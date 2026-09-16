"""A public task bank (R15.0.3): repair tasks from an open-source history.

The same admission rule as the personal ledger, applied to a repository
anyone may clone: a commit that touched test files and source, whose
post-state tests pass and whose frozen tests fail on the parent, as a
``pytest`` task (pytest files beside Python source, nothing built) or a
``make`` task (C test files beside built source, verified by their make
gates). The request is the commit message, the authors' own public words. Nothing in a bank task comes from
a frontier tool, so scaffold learning can run on it without the
training-data firewall's entitlement questions, and a scaffold learned
here is then measured, not trained, on the owner's captured episodes.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from xyntetik_runner.shadow.importer import Episode
from xyntetik_runner.shadow.tasks import RepairTask, Rejection, build_task, classify_range, touched_files


@dataclass(frozen=True)
class BankEntry:
    sha: str
    task: RepairTask | None
    reason: str


def _commits(repo: Path, max_commits: int, since: str = "") -> Iterator[tuple[str, int, str]]:
    cmd = ["git", "-C", str(repo), "log", "--no-merges", "--format=%H%x00%at%x00%B%x1e", f"-n{max_commits}"]
    if since:
        cmd += ["--since", since]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for chunk in proc.stdout.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        sha, at, message = chunk.split("\x00", 2)
        yield sha, int(at), message.strip()


def build_bank(repo: Path, *, out_dir: Path, python: str = sys.executable, limit: int = 20,
               max_commits: int = 400, max_src_files: int = 3, timeout_s: float = 600.0,
               log: Iterator[str] | None = None, kind: str = "any", max_gates: int = 0,
               since: str = "") -> list[BankEntry]:
    """Walk the history newest first and admit up to ``limit`` tasks.
    ``kind`` keeps only ``pytest`` or ``make`` ranges; ``max_gates`` (make
    ranges) caps the C test files a commit may touch, so a single-gate
    bank can be built; ``since`` is git's own date filter."""
    out: list[BankEntry] = []
    admitted = 0
    for i, (sha, at, message) in enumerate(_commits(repo, max_commits, since)):
        if admitted >= limit:
            break
        files = touched_files(repo, [sha])
        touched = classify_range(files)
        built = bool(touched.build or touched.c_tests)
        if built:
            if kind == "pytest":
                out.append(BankEntry(sha, None, "built range, pytest bank"))
                continue
            if not touched.c_tests:
                out.append(BankEntry(sha, None, "built source without a C test file"))
                continue
            if max_gates and len(touched.c_tests) > max_gates:
                out.append(BankEntry(sha, None, f"{len(touched.c_tests)} C test files, above {max_gates}"))
                continue
            src = [*touched.build, *touched.py_src]
        else:
            if kind == "make":
                out.append(BankEntry(sha, None, "python range, make bank"))
                continue
            if not touched.py_tests or not touched.py_src:
                out.append(BankEntry(sha, None, "no test+source pair"))
                continue
            src = list(touched.py_src)
        if len(src) > max_src_files:
            out.append(BankEntry(sha, None, f"{len(src)} source files, above {max_src_files}"))
            continue
        stamp = datetime.fromtimestamp(at, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        request = message or f"commit {sha[:8]}"
        episode = Episode(source="bank", session_id=repo.name, turn=i + 1, cwd=str(repo),
                          started_at=stamp, ended_at=stamp, request=request,
                          request_sha256=hashlib.sha256(request.encode()).hexdigest())
        result = build_task(episode, repo, [sha], out_dir=out_dir, python=python,
                            timeout_s=timeout_s)
        if isinstance(result, Rejection):
            out.append(BankEntry(sha, None, f"{result.disposition.value}: {result.reason}"))
            continue
        admitted += 1
        out.append(BankEntry(sha, result, "admitted"))
    return out
