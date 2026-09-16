"""Synthetic repair tasks from the repository's own gates (suite R14.5.17).

The bank turns commits into tasks and inherits their size: a real commit
touches a median of eight files, which is larger than anything a small
model has completed. This makes the tasks instead: one function of one
source file a make gate links is broken by a small, local mutation, and the
break is kept only when the gate that passed at HEAD fails on it. Each kept
break is a task whose base is the broken tree (a commit on a scratch ref),
whose solution is HEAD, whose verifier is the frozen test file and Makefile
at HEAD, and whose request is the gate's own failure output. Units are
single-function by construction and reachability becomes a dial (the size
and kind of the mutation), which is the ingredient the learning cycle
lacked (the shade closeout of 2026-09-14: no reward was ever reachable).

What a forged task is not: evidence about real work. It is a denominator
for the search and teacher probes (R14.5.16, R14.5.15) and, only if a
model reaches reward on it, a curriculum. Transfer to real commits is
measured against the commit bank, never assumed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from xyntetik_runner.shadow.tasks import (
    RepairTask,
    _drop_worktree,
    _git,
    _worktree,
    c_function_spans,
    gate_name,
)
from xyntetik_runner.shadow.verifier import (
    InstrumentError,
    ProtectedTests,
    calibrate,
    check,
    fixture_targets,
    run_gate,
)

# (name, pattern, replacement): one local edit inside a function body. The
# lookarounds keep `<` out of `<<`, `<=` and templates, `==` out of `===`.
MUTATIONS: tuple[tuple[str, str, str], ...] = (
    ("lt-le", r"(?<![<>=!])<(?![<=])", "<="),
    ("le-lt", r"(?<![<>])<=", "<"),
    ("gt-ge", r"(?<![<>=!-])>(?![>=])", ">="),
    ("ge-gt", r"(?<![<>])>=", ">"),
    ("eq-ne", r"(?<![=!<>])==(?!=)", "!="),
    ("ne-eq", r"!=(?!=)", "=="),
    ("and-or", r"&&", "||"),
    ("or-and", r"\|\|", "&&"),
    ("plus1-minus1", r"\+ 1\b", "- 1"),
    ("minus1-plus1", r"- 1\b", "+ 1"),
    ("ret0-ret1", r"\breturn 0;", "return 1;"),
    ("ret1-ret0", r"\breturn 1;", "return 0;"),
    ("true-false", r"\btrue\b", "false"),
    ("false-true", r"\bfalse\b", "true"),
    ("plus-minus", r"(?<=[A-Za-z0-9_)\]]) \+ (?=[A-Za-z0-9_(])", " - "),
    ("minus-plus", r"(?<=[A-Za-z0-9_)\]]) - (?=[A-Za-z0-9_(])", " + "),
    ("drop-guard", r"^(\s*)if \(.*\) (return[^;]*;|continue;|break;)\s*$", r"\1;"),
)


@dataclass(frozen=True)
class Mutation:
    rel: str
    line: int          # 1-based
    op: str
    before: str
    after: str
    function: str


def gate_sources(tree: Path, gate: str) -> tuple[str, ...]:
    """The source files a gate links, from make's own dry run: every
    `<objdir>/name.o` on its link line is `src/name.c`."""
    make = shutil.which("make") or "make"
    proc = subprocess.run([make, "-n", gate], cwd=tree, capture_output=True, text=True)
    names: list[str] = []
    for m in re.finditer(r"(?:\S+/)?([A-Za-z0-9_]+)\.o\b|\b((?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.c)\b",
                         proc.stdout):
        rel = f"src/{m.group(1)}.c" if m.group(1) else m.group(2)
        if rel.startswith("tests/") or not (tree / rel).is_file() or rel in names:
            continue
        names.append(rel)
    return tuple(names)


def candidate_mutations(rel: str, text: str) -> list[Mutation]:
    """Every single-line mutation inside a function body of ``text``."""
    spans = c_function_spans(text)
    lines = text.split("\n")
    out: list[Mutation] = []
    for start, end, name in spans:
        for ln in range(start + 1, end):   # never the head line or the closing brace
            line = lines[ln - 1]
            s = line.strip()
            if not s or s.startswith(("//", "/*", "*", "#")) or "\"" in line:
                continue
            for op, pat, repl in MUTATIONS:
                new = re.sub(pat, repl, line, count=1)
                if new != line:
                    out.append(Mutation(rel, ln, op, line, new, name))
    return out


def _stamp_newer(tree: Path, p: Path) -> None:
    """make compares whole-second mtimes: a source rewritten in the second
    its gate was built reads as up to date and the OLD binary answers
    (measured: the first forged mutation "passed" its gate that way). Stamp
    the file strictly past everything in the tree."""
    newest = max((q.stat().st_mtime for q in tree.rglob("*") if q.is_file()), default=0.0)
    stamp = max(time.time(), newest + 1.0)
    os.utime(p, (stamp, stamp))


def _apply(tree: Path, m: Mutation) -> None:
    p = tree / m.rel
    lines = p.read_text(encoding="utf-8").split("\n")
    assert lines[m.line - 1] == m.before, (m.rel, m.line)
    lines[m.line - 1] = m.after
    p.write_text("\n".join(lines), encoding="utf-8")
    _stamp_newer(tree, p)


def _restore(tree: Path, m: Mutation) -> None:
    p = tree / m.rel
    lines = p.read_text(encoding="utf-8").split("\n")
    lines[m.line - 1] = m.before
    p.write_text("\n".join(lines), encoding="utf-8")
    _stamp_newer(tree, p)


def _compile_error(out: bytes) -> bool:
    text = out.decode("utf-8", errors="replace")
    return bool(re.search(r"(error:|\*\*\* \[)", text))


@dataclass(frozen=True)
class ForgeEntry:
    mutation: Mutation
    gate: str
    outcome: str        # kept | gate-passed | no-compile | timeout | instrument
    task_id: str = ""


def forge(repo: Path, gates: Sequence[str], *, out_dir: Path, limit: int = 30, seed: int = 0,
          timeout_s: float = 600.0, log: Iterator[str] | None = None) -> list[ForgeEntry]:
    """Break functions until ``limit`` gate-caught breaks are filed as tasks."""
    head = _git(str(repo), "rev-parse", "HEAD").strip()
    stamp = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    rng = random.Random(seed)
    entries: list[ForgeEntry] = []
    kept = 0
    probe = _worktree(repo, head)      # one tree, mutated and restored in place
    try:
        makefile = next((n for n in ("Makefile", "makefile", "GNUmakefile") if (probe / n).is_file()), None)
        if makefile is None:
            raise RuntimeError("no Makefile at HEAD")
        # a warm build directory beside the probe tree: the first gate builds
        # everything, later probes rebuild the one mutated file
        build = Path(tempfile.mkdtemp(prefix="xyntetik-shadow-forge-"))
        shutil.copytree(probe, build, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
        per_gate: list[tuple[str, str, list[Mutation]]] = []
        for gate in gates:
            rel_test = f"tests/test_{gate[len('test-'):].replace('-', '_')}.c"
            if not (probe / rel_test).is_file():
                continue
            state, out, late = run_gate(build, gate, timeout_s=timeout_s,
                                        fixtures=fixture_targets(build, rel_test))
            if state != "passed":
                continue          # a gate that does not pass at HEAD cannot catch anything
            cands: list[Mutation] = []
            for rel in gate_sources(probe, gate):
                cands.extend(candidate_mutations(rel, (probe / rel).read_text(encoding="utf-8")))
            rng.shuffle(cands)
            per_gate.append((gate, rel_test, cands))
        # round-robin over gates so one gate's sources do not fill the bank
        cursors = [0] * len(per_gate)
        used_lines: set[tuple[str, int]] = set()
        while kept < limit and any(c < len(pg[2]) for c, pg in zip(cursors, per_gate)):
            for gi, (gate, rel_test, cands) in enumerate(per_gate):
                if kept >= limit or cursors[gi] >= len(cands):
                    continue
                m = cands[cursors[gi]]
                cursors[gi] += 1
                if (m.rel, m.line) in used_lines:
                    continue
                _apply(build, m)
                state, out, late = run_gate(build, gate, timeout_s=timeout_s,
                                            fixtures=fixture_targets(build, rel_test))
                _restore(build, m)
                if late:
                    entries.append(ForgeEntry(m, gate, "timeout")); continue
                if state == "passed":
                    entries.append(ForgeEntry(m, gate, "gate-passed")); continue
                if _compile_error(out):
                    entries.append(ForgeEntry(m, gate, "no-compile")); continue
                tail = out.decode("utf-8", errors="replace").strip().split("\n")[-6:]
                task = _file_task(repo, probe, head, m, gate, rel_test, makefile, "\n".join(tail),
                                  stamp, out_dir=out_dir, timeout_s=timeout_s, index=kept + 1)
                if isinstance(task, str):
                    entries.append(ForgeEntry(m, gate, task)); continue
                used_lines.add((m.rel, m.line))
                kept += 1
                entries.append(ForgeEntry(m, gate, "kept", task.task_id))
                if log is not None:
                    print(f"  forged {task.task_id}: {m.op} in {m.function} ({m.rel}:{m.line}) caught by {gate}",
                          flush=True)
        shutil.rmtree(build, ignore_errors=True)
    finally:
        _drop_worktree(repo, probe)
    return entries


def _file_task(repo: Path, probe: Path, head: str, m: Mutation, gate: str, rel_test: str,
               makefile: str, failure_tail: str, stamp: str, *, out_dir: Path,
               timeout_s: float, index: int) -> RepairTask | str:
    """Commit the break on the probe tree (detached; the commit is reachable
    by sha), freeze the gate at HEAD, calibrate, and write the task."""
    _apply(probe, m)
    subprocess.run(["git", "-C", str(probe), "add", "-A"], capture_output=True)
    env = {"GIT_AUTHOR_NAME": "shadow forge", "GIT_AUTHOR_EMAIL": "forge@localhost",
           "GIT_COMMITTER_NAME": "shadow forge", "GIT_COMMITTER_EMAIL": "forge@localhost",
           "PATH": os.environ.get("PATH", "")}
    proc = subprocess.run(["git", "-C", str(probe), "commit", "-q", "-m",
                           f"forge: {m.op} in {m.function} ({m.rel}:{m.line})"],
                          capture_output=True, text=True, env=env)
    base = _git(str(probe), "rev-parse", "HEAD").strip()
    # back to HEAD for the next probe, keeping the break reachable by sha
    subprocess.run(["git", "-C", str(probe), "checkout", "-q", "--detach", head], capture_output=True)
    if proc.returncode != 0 or not base or base == head:
        return "instrument"
    # the break must stay reachable after the probe worktree is dropped
    subprocess.run(["git", "-C", str(repo), "update-ref", f"refs/shadow-forge/{base[:12]}", base],
                   capture_output=True)
    task_id = f"{repo.name}-forge-{base[:8]}-{head[:8]}"
    task_dir = out_dir / task_id
    protected = task_dir / "protected"
    if protected.exists():
        shutil.rmtree(protected)
    for rel in (rel_test, makefile):
        (protected / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(probe / rel, protected / rel)
    frozen = ProtectedTests.freeze(protected, {rel_test: [gate]}, verifier_id=f"forge-gate:{task_id}",
                                   kind="make")
    outcome = check(probe, frozen, timeout_s=timeout_s)
    if outcome.passed is not True:
        shutil.rmtree(task_dir, ignore_errors=True)
        return "instrument"
    broken = _worktree(repo, base)
    try:
        try:
            cal = calibrate(frozen, broken, timeout_s=timeout_s)
        except InstrumentError:
            shutil.rmtree(task_dir, ignore_errors=True)
            return "instrument"
    finally:
        _drop_worktree(repo, broken)
    request = (f"`make {gate} && ./{gate}` fails on this tree. The tail of its output:\n\n"
               f"{failure_tail}\n\nThe test is right; the fault is in the source the gate links. "
               f"Make the gate pass without changing the test.")
    task = RepairTask(
        task_id=task_id, episode_id=f"forge:{repo.name}:{index}", repo=str(repo), base_sha=base,
        solution_sha=head, request=request,
        request_sha256=hashlib.sha256(request.encode("utf-8")).hexdigest(), context=(),
        test_files=(rel_test,), visible_test_files=(rel_test,), src_files=1, pythonpath=(),
        protected_dir=str(protected), expected_tests=1,
        baseline_failing=cal.failing + cal.missing, failing_at_base=cal.failing_ids,
        task_class="function", changed_lines=1, verifier_kind="make", gates=(gate,))
    (task_dir / "task.json").write_text(task.to_json() + "\n", encoding="utf-8")
    (task_dir / "forge.json").write_text(json.dumps({
        "mutation": m.__dict__, "gate": gate, "head": head, "stamp": stamp}, indent=2) + "\n",
        encoding="utf-8")
    return task
