"""Repair tasks reconstructed from the repository, not from the trace.

An episode says when and where work happened. The repository says what
changed: the commits in the episode's window (in any git repository at or
under the working directory) give a pre-state, a post-state and the files
touched. A task is admitted when the range touched test files and source,
the post-state's versions of those test files pass on the post-state tree,
and the same frozen files fail on the pre-state tree. The frozen tests are
the verifier; the source diff is the frontier's answer and is never given
to the attempt.

Two verifier kinds. A range of Python source and pytest files, with nothing
that needs a build, is a ``pytest`` task: the frozen files are collected
and run by pytest. A range that touches C, CUDA, Metal or other built
source is a ``make`` task: its verifier is the make gates of the C test
files it touched (``tests/test_x.c`` is ``make test-x`` then ``./test-x``,
the convention the runner's own Makefile follows), built from the frozen
test files and the frozen post-state Makefile. A built range without a C
test file has no gate to verify it and is ineligible.

Everything the attempt receives is: the pre-state tree, the user's request,
and the names of the visible test files (and gates) as they were at the
pre-state.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from xyntetik_runner.shadow.evidence import Disposition
from xyntetik_runner.shadow.importer import Episode
from xyntetik_runner.shadow.verifier import InstrumentError, ProtectedTests, calibrate, check

BUILD_SUFFIXES = (".c", ".h", ".cu", ".m", ".metal", ".cpp", ".rs", ".go")
C_TEST_SUFFIX = ".c"
MAKEFILE_NAMES = frozenset({"Makefile", "makefile", "GNUmakefile"})
SKIP_DIRS = frozenset({"node_modules", "models", ".venv", "venv", "__pycache__", "dist", "build"})
COMMIT_SLACK = timedelta(minutes=30)


@dataclass(frozen=True)
class Rejection:
    disposition: Disposition
    reason: str


@dataclass(frozen=True)
class RepairTask:
    task_id: str
    episode_id: str
    repo: str
    base_sha: str
    solution_sha: str
    request: str
    request_sha256: str
    context: tuple[str, ...]
    test_files: tuple[str, ...]
    visible_test_files: tuple[str, ...]
    src_files: int
    pythonpath: tuple[str, ...]
    protected_dir: str
    expected_tests: int
    baseline_failing: int
    failing_at_base: tuple[str, ...] = ()
    task_class: str = "file"
    """``function`` when every changed source line sits inside one function
    or method; ``file`` for one file beyond that; ``multi-file`` otherwise.
    The class a person delegates is the first one."""
    changed_lines: int = 0
    verifier_kind: str = "pytest"
    """``pytest`` or ``make`` (see the module docstring); a task written
    before kinds existed is pytest."""
    gates: tuple[str, ...] = ()
    """The make gates of a ``make`` task, one per frozen C test file."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def load(cls, path: Path) -> RepairTask:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("test_files", "visible_test_files", "pythonpath", "failing_at_base",
                    "context", "gates"):
            data[key] = tuple(data.get(key) or ())
        return cls(**data)


def canonical(path: Path | str) -> Path:
    """One spelling per directory. ``Path.resolve`` is not enough on every
    Python: a Windows build can keep the separator a path was created with,
    so ``C:/x/proj`` and ``C:/x\\proj`` compare unequal while naming one
    directory. normpath fixes the separators and dots, normcase the case."""
    return Path(os.path.normcase(os.path.normpath(os.path.realpath(str(path)))))


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return proc.stdout


def repository_root(cwd: Path) -> Path | None:
    """Resolve Git's relative root against a native path, including with MSYS Git."""
    try:
        proc = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--show-cdup"],
                              capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # --show-toplevel may be /drive/... while Python expects Drive:/... .
    # The relative path is valid in both path namespaces; empty means cwd.
    return canonical(cwd / proc.stdout.strip())


def repos_under(cwd: Path, *, depth: int = 2) -> list[Path]:
    """Git repositories at or under ``cwd``, at most ``depth`` levels down,
    so a session run from a parent directory still finds its repositories.
    A repository that contains other repositories (an umbrella checkout
    with the real projects untracked inside it) yields all of them: the
    walk does not stop at the first ``.git``."""
    out: list[Path] = []
    if not cwd.is_dir():
        return out
    # Resolved paths throughout: a walked path carries the platform's
    # separator while a caller's may not (pytest's temp paths on Windows
    # keep forward slashes), and two spellings of one directory must be one
    # repository in every set and dict keyed on it.
    stack: list[tuple[Path, int]] = [(canonical(cwd), 0)]
    while stack:
        here, d = stack.pop()
        if (here / ".git").exists():
            out.append(canonical(here))
        if d >= depth:
            continue
        try:
            children = sorted(p for p in here.iterdir() if p.is_dir() and not p.is_symlink())
        except OSError:
            continue
        for child in children:
            if child.name.startswith(".") or child.name in SKIP_DIRS:
                continue
            stack.append((child, d + 1))
    return sorted(out)


def commits_in_window(repo: Path, start: datetime, end: datetime,
                      *, slack: timedelta = COMMIT_SLACK) -> list[str]:
    """Non-merge commits on any ref authored inside the window (newest first)."""
    text = _git(str(repo), "log", "--all", "--no-merges", "--format=%H",
                "--since", start.isoformat(), "--until", (end + slack).isoformat())
    return text.split()


def touched_files(repo: Path, shas: Iterable[str]) -> set[str]:
    files: set[str] = set()
    for sha in shas:
        # -z: names come NUL-separated and unquoted, so a non-ASCII path is
        # the path and not git's C-quoted rendering of it
        text = _git(str(repo), "show", "--name-only", "--format=", "--no-renames", "-z", sha)
        files.update(x for x in text.split("\0") if x)
    return files


def classify(files: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    """(pytest files, Python source, built source) of a range. A C test file
    counts as built source here; ``classify_range`` tells them apart."""
    tests, src, build = [], [], []
    for f in sorted(files):
        name = Path(f).name
        if name.startswith("test_") and name.endswith(".py"):
            tests.append(f)
        elif f.endswith(".py"):
            src.append(f)
        elif f.endswith(BUILD_SUFFIXES):
            build.append(f)
    return tests, src, build


@dataclass(frozen=True)
class Touched:
    py_tests: tuple[str, ...]
    py_src: tuple[str, ...]
    build: tuple[str, ...]
    """built source that is not a C test file"""
    c_tests: tuple[str, ...]
    makefile: bool


def is_c_test(rel: str) -> bool:
    name = Path(rel).name
    return name.startswith("test_") and name.endswith(C_TEST_SUFFIX)


def gate_name(rel: str) -> str:
    """``tests/test_penalty_window.c`` builds and runs as ``test-penalty-window``."""
    return "test-" + Path(rel).stem[len("test_"):].replace("_", "-")


def classify_range(files: Iterable[str]) -> Touched:
    tests, src, build = classify(files)
    c_tests = tuple(f for f in build if is_c_test(f))
    return Touched(py_tests=tuple(tests), py_src=tuple(src),
                   build=tuple(f for f in build if not is_c_test(f)), c_tests=c_tests,
                   makefile=any(Path(f).name in MAKEFILE_NAMES for f in files))


def import_roots(tree: Path) -> tuple[str, ...]:
    """Workspace-relative import roots for a src layout, a python/ client, or
    a packages/* monorepo; the repository's own tests need them."""
    roots: list[str] = []
    for cand in ("python/src", "src"):
        if (tree / cand).is_dir():
            roots.append(cand)
    packages = tree / "packages"
    if packages.is_dir():
        for p in sorted(packages.iterdir()):
            if p.is_dir():
                roots.append(f"packages/{p.name}/src" if (p / "src").is_dir() else f"packages/{p.name}")
    return tuple(roots)


def _worktree(repo: Path, sha: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="xyntetik-shadow-wt-"))
    proc = subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", "-q", str(d), sha],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError(f"git worktree add failed: {proc.stderr.strip()[:200]}")
    return d


def _drop_worktree(repo: Path, d: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(d)],
                   capture_output=True)
    shutil.rmtree(d, ignore_errors=True)


def collect_tests(tree: Path, files: Sequence[str], roots: Sequence[str], *, python: str,
                  timeout_s: float) -> dict[str, list[str]]:
    """Expected test ids per file, from pytest's own collection at the
    post-state; a file that does not collect contributes nothing."""
    import os
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "SystemRoot") if k in os.environ}
    env.update({"HOME": str(tree), "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join([str(tree)] + [str(tree / r) for r in roots])})
    try:
        # rootdir pinned to the tree: pytest otherwise takes the nearest
        # pyproject.toml above a test file, and a repository whose Python
        # lives in a subdirectory then reports nodeids relative to that
        # subdirectory, which never match the paths admitted here
        proc = subprocess.run([python, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only",
                               "-q", "--rootdir", str(Path(tree).resolve()), *files], cwd=tree, env=env,
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {}
    expected: dict[str, list[str]] = {}
    for line in proc.stdout.split("\n"):
        line = line.strip()
        if "::" not in line or line.startswith(("ERROR", "FAILED")):
            continue
        f, name = line.split("::", 1)
        if f in files:
            expected.setdefault(f, []).append(name)
    return expected


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def change_class(repo: Path, base: str, solution: str, src: Sequence[str]) -> tuple[str, int]:
    """(class, changed source lines) for the range, from the diff and the
    post-state's own syntax tree: ``function`` when every changed line of
    the one changed file falls inside one function or method."""
    if len(src) != 1:
        return ("multi-file" if len(src) > 1 else "file", 0)
    rel = src[0]
    diff = _git(str(repo), "diff", "-U0", base, solution, "--", rel)
    post = _git(str(repo), "show", f"{solution}:{rel}")
    if rel.endswith(BUILD_SUFFIXES):
        return class_from_diff_c(diff, post)
    return class_from_diff(diff, post)


def _hunks(diff: str) -> tuple[list[tuple[int, int]], int]:
    ranges: list[tuple[int, int]] = []
    changed = 0
    for line in diff.split("\n"):
        m = _HUNK.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        changed += count
        ranges.append((start, start + max(count, 1) - 1))
    return ranges, changed


def _strip_c(line: str, in_comment: bool) -> tuple[str, bool]:
    """A C line without its comments and string or character literals, and
    whether a block comment is still open after it."""
    out: list[str] = []
    j = 0
    while j < len(line):
        if in_comment:
            k = line.find("*/", j)
            if k < 0:
                return "".join(out), True
            in_comment, j = False, k + 2
            continue
        if line.startswith("//", j):
            break
        if line.startswith("/*", j):
            in_comment, j = True, j + 2
            continue
        ch = line[j]
        if ch in "\"'":
            j += 1
            while j < len(line) and line[j] != ch:
                j += 2 if line[j] == "\\" else 1
            j += 1
            continue
        out.append(ch)
        j += 1
    return "".join(out), in_comment


_C_NAME = re.compile(r"([A-Za-z_]\w*)\s*\(")
_C_NOT_FUNC = ("struct", "enum", "union", "typedef", "if", "for", "while", "switch", "return")


def c_function_spans(text: str) -> list[tuple[int, int, str]]:
    """(first line, last line, name) of every function definition in a C
    file, from brace depth: a definition opens at depth zero with a
    parenthesised head and closes when the depth returns to zero. Comments,
    strings and preprocessor lines are ignored. A heuristic, exact enough
    to say whether a change stayed inside one function."""
    spans: list[tuple[int, int, str]] = []
    depth = 0
    in_comment = False
    start: tuple[int, str] | None = None
    pending: tuple[int, str] | None = None   # a head whose brace is on a later line
    for i, raw in enumerate(text.split("\n"), 1):
        code, in_comment = _strip_c(raw, in_comment)
        stripped = code.strip()
        if depth == 0 and start is None and stripped.startswith("#"):
            continue
        opens, closes = code.count("{"), code.count("}")
        if depth == 0 and start is None and stripped:
            head = code.split("{", 1)[0]
            m = _C_NAME.search(head)
            is_head = (m is not None and "=" not in head and ";" not in head
                       and m.group(1) not in _C_NOT_FUNC)
            if is_head and opens:
                start = (i, m.group(1))
            elif is_head and head.rstrip().endswith(")"):
                pending = (i, m.group(1))
            elif opens and pending is not None and stripped.startswith("{"):
                start = pending
            else:
                pending = None
        depth += opens - closes
        if start is not None and depth <= 0:
            spans.append((start[0], i, start[1]))
            start, pending, depth = None, None, 0
    return spans


def class_from_diff_c(diff: str, post_text: str) -> tuple[str, int]:
    """``class_from_diff`` for a C file: ``function`` when every changed
    line of the file falls inside one function definition; an added
    ``#include`` does not make it a file-level change."""
    ranges, changed = _hunks(diff)
    if not ranges:
        return ("file", 0)
    spans = c_function_spans(post_text)
    lines = post_text.split("\n")
    owners: set[str] = set()
    for a, b in ranges:
        if all(lines[k - 1].lstrip().startswith("#include") for k in range(a, b + 1)
               if 0 < k <= len(lines)):
            continue
        inside = [s for s in spans if s[0] <= a and b <= s[1]]
        if not inside:
            return ("file", changed)
        innermost = min(inside, key=lambda s: s[1] - s[0])
        owners.add(f"{innermost[2]}@{innermost[0]}")
    return ("function" if len(owners) == 1 else "file", changed)


def class_from_diff(diff: str, post_text: str) -> tuple[str, int]:
    """The class of one file's change from its unified diff (zero context)
    and the file's post-state text; shared by the committed range and the
    uncommitted scratch worktree of a delegation."""
    ranges, changed = _hunks(diff)
    if not ranges:
        return ("file", 0)
    try:
        tree = ast.parse(post_text)
    except SyntaxError:
        return ("file", changed)
    spans: list[tuple[int, int, str]] = []
    imports: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
            spans.append((node.lineno, node.end_lineno, node.name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and node.end_lineno:
            imports.append((node.lineno, node.end_lineno))
    owners: set[str] = set()
    for a, b in ranges:
        # an import the fix needed does not make it a file-level change
        if any(i0 <= a and b <= i1 for i0, i1 in imports):
            continue
        inside = [s for s in spans if s[0] <= a and b <= s[1]]
        if not inside:
            return ("file", changed)
        innermost = min(inside, key=lambda s: s[1] - s[0])
        owners.add(f"{innermost[2]}@{innermost[0]}")
    return ("function" if len(owners) == 1 else "file", changed)


def build_task(episode: Episode, repo: Path, shas: Sequence[str], *, out_dir: Path,
               python: str = sys.executable, timeout_s: float = 600.0) -> RepairTask | Rejection:
    """Admit or reject one (episode, repository, commit range)."""
    files = touched_files(repo, shas)
    touched = classify_range(files)
    if touched.build or touched.c_tests:
        return build_make_task(episode, repo, shas, touched, out_dir=out_dir, timeout_s=timeout_s)
    tests, src = list(touched.py_tests), list(touched.py_src)
    if not tests or not src:
        return Rejection(Disposition.INELIGIBLE, "range touches no pytest test file with Python source")
    solution = shas[0]
    # --verify: without it git echoes an unresolvable "<sha>^" back on stdout
    # and a root commit would reach git worktree as an invalid reference.
    base = _git(str(repo), "rev-parse", "--verify", "--quiet", f"{shas[-1]}^").strip()
    if not base:
        return Rejection(Disposition.UNREPLAYABLE, "first commit in the range has no parent")
    task_id = f"{repo.name}-{base[:8]}-{solution[:8]}"
    task_dir = out_dir / task_id
    protected = task_dir / "protected"
    post = pre = None
    admitted = False
    try:
        post = _worktree(repo, solution)
        pre = _worktree(repo, base)
        roots = import_roots(post)
        present = [t for t in tests if (post / t).is_file()]
        if not present:
            return Rejection(Disposition.UNREPLAYABLE, "test files were deleted in the range")
        expected = collect_tests(post, present, roots, python=python, timeout_s=timeout_s)
        if not expected:
            return Rejection(Disposition.UNREPLAYABLE, "post-state test files do not collect")
        if protected.exists():
            shutil.rmtree(protected)
        for rel in expected:
            (protected / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(post / rel, protected / rel)
        frozen = ProtectedTests.freeze(protected, expected, verifier_id=f"commit-tests:{task_id}")
        outcome = check(post, frozen, timeout_s=timeout_s, python=python, pythonpath=roots)
        if outcome.passed is not True:
            return Rejection(Disposition.UNREPLAYABLE,
                             f"frozen tests do not pass on the post-state: {'; '.join(outcome.reasons)}")
        try:
            cal = calibrate(frozen, pre, timeout_s=timeout_s, python=python, pythonpath=roots)
        except InstrumentError as e:
            return Rejection(Disposition.UNREPLAYABLE, f"instrument: {e}")
        visible = tuple(t for t in present if (pre / t).is_file())
        klass, changed = change_class(repo, base, solution, src)
        task = RepairTask(
            task_id=task_id, episode_id=episode.episode_id, repo=str(repo), base_sha=base,
            solution_sha=solution, request=episode.request, request_sha256=episode.request_sha256,
            context=episode.context, test_files=tuple(present), visible_test_files=visible, src_files=len(src),
            pythonpath=roots, protected_dir=str(protected), expected_tests=len(frozen.expected),
            baseline_failing=cal.failing + cal.missing, failing_at_base=cal.failing_ids,
            task_class=klass, changed_lines=changed)
        (task_dir / "task.json").write_text(task.to_json() + "\n", encoding="utf-8")
        admitted = True
        return task
    except RuntimeError as e:
        return Rejection(Disposition.UNREPLAYABLE, str(e))
    finally:
        if post is not None:
            _drop_worktree(repo, post)
        if pre is not None:
            _drop_worktree(repo, pre)
        if not admitted and task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)


def build_tools_present() -> str | None:
    """Why a make task cannot be verified on this machine, or None."""
    if not shutil.which("make"):
        return "no make on this machine"
    cc = os.environ.get("CC") or "cc"
    if not shutil.which(cc):
        return f"no C compiler on this machine ({cc} not found)"
    return None


def build_make_task(episode: Episode, repo: Path, shas: Sequence[str], touched: Touched, *,
                    out_dir: Path, timeout_s: float = 600.0) -> RepairTask | Rejection:
    """Admit or reject a built range: the verifier is the make gates of the
    C test files it touched, frozen with the post-state Makefile."""
    n_built = len(touched.build) + len(touched.c_tests)
    if not touched.c_tests:
        return Rejection(Disposition.INELIGIBLE,
                         f"range touches {n_built} built file(s) and no C test file (tests/test_*.c) "
                         "to verify them")
    why = build_tools_present()
    if why:
        return Rejection(Disposition.UNREPLAYABLE, f"make gate: {why}")
    solution = shas[0]
    base = _git(str(repo), "rev-parse", "--verify", "--quiet", f"{shas[-1]}^").strip()
    if not base:
        return Rejection(Disposition.UNREPLAYABLE, "first commit in the range has no parent")
    task_id = f"{repo.name}-{base[:8]}-{solution[:8]}"
    task_dir = out_dir / task_id
    protected = task_dir / "protected"
    post = pre = None
    admitted = False
    try:
        post = _worktree(repo, solution)
        pre = _worktree(repo, base)
        makefile = next((n for n in ("Makefile", "makefile", "GNUmakefile") if (post / n).is_file()), None)
        if makefile is None:
            return Rejection(Disposition.UNREPLAYABLE, "no Makefile at the post-state to build the gates")
        present = [t for t in touched.c_tests if (post / t).is_file()]
        if not present:
            return Rejection(Disposition.UNREPLAYABLE, "C test files were deleted in the range")
        if protected.exists():
            shutil.rmtree(protected)
        expected = {rel: [gate_name(rel)] for rel in present}
        for rel in [*expected, makefile]:
            (protected / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(post / rel, protected / rel)
        frozen = ProtectedTests.freeze(protected, expected, verifier_id=f"commit-gates:{task_id}",
                                       kind="make")
        outcome = check(post, frozen, timeout_s=timeout_s)
        if outcome.passed is not True:
            return Rejection(Disposition.UNREPLAYABLE,
                             f"frozen gates do not pass on the post-state: {'; '.join(outcome.reasons)}")
        try:
            cal = calibrate(frozen, pre, timeout_s=timeout_s)
        except InstrumentError as e:
            return Rejection(Disposition.UNREPLAYABLE, f"instrument: {e}")
        visible = tuple(t for t in present if (pre / t).is_file())
        src = [*touched.build, *touched.py_src]
        klass, changed = change_class(repo, base, solution, src)
        task = RepairTask(
            task_id=task_id, episode_id=episode.episode_id, repo=str(repo), base_sha=base,
            solution_sha=solution, request=episode.request, request_sha256=episode.request_sha256,
            context=episode.context, test_files=tuple(present), visible_test_files=visible,
            src_files=len(src), pythonpath=(), protected_dir=str(protected),
            expected_tests=len(frozen.expected), baseline_failing=cal.failing + cal.missing,
            failing_at_base=cal.failing_ids, task_class=klass, changed_lines=changed,
            verifier_kind="make", gates=tuple(gate_name(rel) for rel in present))
        (task_dir / "task.json").write_text(task.to_json() + "\n", encoding="utf-8")
        admitted = True
        return task
    except RuntimeError as e:
        return Rejection(Disposition.UNREPLAYABLE, str(e))
    finally:
        if post is not None:
            _drop_worktree(repo, post)
        if pre is not None:
            _drop_worktree(repo, pre)
        if not admitted and task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)


@dataclass(frozen=True)
class Candidate:
    episode: Episode
    repo: Path
    shas: tuple[str, ...]

    @property
    def solution(self) -> str:
        return self.shas[0]


def pair(episode: Episode) -> list[Candidate] | Rejection:
    """The (repository, commit range) pairs an episode's window covers, or
    why it has none."""
    if episode.is_command:
        return Rejection(Disposition.INELIGIBLE, "harness-injected turn, not a request")
    if not episode.request.strip():
        return Rejection(Disposition.INELIGIBLE, "empty request")
    repos = repos_under(Path(episode.cwd))
    if not repos:
        return Rejection(Disposition.INELIGIBLE, "no git repository at or under the working directory")
    if episode.source == "capture":
        # Prospective capture: the exact commits between the two HEADs of
        # every repository the hook saw at both ends; no time window at all.
        before = dict(episode.heads_start)
        after = dict(episode.heads_end)
        if episode.head_start and episode.head_end:
            before.setdefault(episode.cwd, episode.head_start)
            after.setdefault(episode.cwd, episode.head_end)
        found: list[Candidate] = []
        moved = False
        for repo_path, start_sha in before.items():
            end_sha = after.get(repo_path)
            if not end_sha or end_sha == start_sha:
                continue
            moved = True
            repo = Path(repo_path)
            if not (repo / ".git").exists():
                continue
            shas = _git(str(repo), "rev-list", "--no-merges", f"{start_sha}..{end_sha}").split()
            if shas:
                found.append(Candidate(episode, repo, tuple(shas)))
        if found:
            return found
        if not before:
            return Rejection(Disposition.UNREPLAYABLE, "capture recorded no repository HEAD")
        if not moved:
            return Rejection(Disposition.UNREPLAYABLE, "HEAD did not move during the turn")
        return Rejection(Disposition.UNREPLAYABLE,
                         "captured HEADs are not an ancestry range in any repository here")
    start = datetime.fromisoformat(episode.started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(episode.ended_at.replace("Z", "+00:00"))
    found = [Candidate(episode, repo, tuple(commits_in_window(repo, start, end))) for repo in repos]
    found = [c for c in found if c.shas]
    if not found:
        return Rejection(Disposition.UNREPLAYABLE, "no commit in the task window")
    return found


def choose(candidates: Iterable[Candidate]) -> dict[str, Candidate]:
    """One episode per fix commit: the one whose working directory is the
    repository itself over a parent, then the latest prompt before the
    fix. A commit reachable from two clones is still one fix."""
    best: dict[str, Candidate] = {}
    for c in candidates:
        depth = 0 if canonical(c.episode.cwd) == canonical(c.repo) else 1
        key = (depth, -datetime.fromisoformat(c.episode.started_at.replace("Z", "+00:00")).timestamp())
        cur = best.get(c.solution)
        if cur is None:
            best[c.solution] = c
            continue
        cur_depth = 0 if canonical(cur.episode.cwd) == canonical(cur.repo) else 1
        cur_key = (cur_depth, -datetime.fromisoformat(cur.episode.started_at.replace("Z", "+00:00")).timestamp())
        if key < cur_key:
            best[c.solution] = c
    return best


def admit(episode: Episode, *, out_dir: Path, python: str = sys.executable,
          timeout_s: float = 600.0, seen: set[tuple[str, str]] | None = None
          ) -> RepairTask | Rejection:
    """Pair one episode with its repository and commit range, then build.
    For a whole trace use ``pair`` + ``choose`` so overlapping windows are
    attributed once; this per-episode path keys duplicates on the fix sha."""
    paired = pair(episode)
    if isinstance(paired, Rejection):
        return paired
    last: RepairTask | Rejection = Rejection(Disposition.INELIGIBLE, "no candidate")
    for repo, shas in ((c.repo, list(c.shas)) for c in paired):
        # One fix commit is one task, whichever clone it is reachable from and
        # however many turns' windows overlap it: the key is the solution sha.
        key = ("solution", shas[0])
        if seen is not None and key in seen:
            return Rejection(Disposition.INELIGIBLE, "same fix commit as an earlier episode")
        last = build_task(episode, repo, shas, out_dir=out_dir, python=python, timeout_s=timeout_s)
        if isinstance(last, RepairTask):
            if seen is not None:
                seen.add(key)
            return last
    return last
