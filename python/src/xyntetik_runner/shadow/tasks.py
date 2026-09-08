"""Repair tasks reconstructed from the repository, not from the trace.

An episode says when and where work happened. The repository says what
changed: the commits in the episode's window (in any git repository at or
under the working directory) give a pre-state, a post-state and the files
touched. A task is admitted when the range touched pytest test files and
Python source and nothing that needs a build, the post-state's versions of
those test files can be collected and pass on the post-state tree, and the
same frozen files fail on the pre-state tree. The frozen tests are the
verifier; the source diff is the frontier's answer and is never given to
the attempt.

Everything the attempt receives is: the pre-state tree, the user's request,
and the names of the visible test files as they were at the pre-state.
"""

from __future__ import annotations

import json
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

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def load(cls, path: Path) -> RepairTask:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("test_files", "visible_test_files", "pythonpath", "failing_at_base",
                    "context"):
            data[key] = tuple(data.get(key) or ())
        return cls(**data)


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return proc.stdout


def repos_under(cwd: Path, *, depth: int = 2) -> list[Path]:
    """Git repositories at or under ``cwd``, at most ``depth`` levels down,
    so a session run from a parent directory still finds its repositories.
    A repository that contains other repositories (an umbrella checkout
    with the real projects untracked inside it) yields all of them: the
    walk does not stop at the first ``.git``."""
    out: list[Path] = []
    if not cwd.is_dir():
        return out
    stack: list[tuple[Path, int]] = [(cwd, 0)]
    while stack:
        here, d = stack.pop()
        if (here / ".git").exists():
            out.append(here)
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
        text = _git(str(repo), "show", "--name-only", "--format=", "--no-renames", sha)
        files.update(x for x in text.split("\n") if x)
    return files


def classify(files: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
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
        proc = subprocess.run([python, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only",
                               "-q", *files], cwd=tree, env=env, capture_output=True, text=True,
                              timeout=timeout_s)
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


def build_task(episode: Episode, repo: Path, shas: Sequence[str], *, out_dir: Path,
               python: str = sys.executable, timeout_s: float = 600.0) -> RepairTask | Rejection:
    """Admit or reject one (episode, repository, commit range)."""
    files = touched_files(repo, shas)
    tests, src, build = classify(files)
    if not tests or not src:
        return Rejection(Disposition.INELIGIBLE, "range touches no pytest test file with Python source")
    if build:
        return Rejection(Disposition.INELIGIBLE, f"range touches {len(build)} file(s) that need a build")
    solution = shas[0]
    base = _git(str(repo), "rev-parse", f"{shas[-1]}^").strip()
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
        task = RepairTask(
            task_id=task_id, episode_id=episode.episode_id, repo=str(repo), base_sha=base,
            solution_sha=solution, request=episode.request, request_sha256=episode.request_sha256,
            context=episode.context, test_files=tuple(present), visible_test_files=visible, src_files=len(src),
            pythonpath=roots, protected_dir=str(protected), expected_tests=len(frozen.expected),
            baseline_failing=cal.failing + cal.missing, failing_at_base=cal.failing_ids)
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
        depth = 0 if Path(c.episode.cwd).resolve() == c.repo.resolve() else 1
        key = (depth, -datetime.fromisoformat(c.episode.started_at.replace("Z", "+00:00")).timestamp())
        cur = best.get(c.solution)
        if cur is None:
            best[c.solution] = c
            continue
        cur_depth = 0 if Path(cur.episode.cwd).resolve() == cur.repo.resolve() else 1
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
