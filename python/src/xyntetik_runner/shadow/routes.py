"""Where the local model has verified successes (the route table), and
delegation of one request on a scratch worktree.

The route table is the bench, the ledger and the recorded delegations read
per task class: attempted and verified counts for the model stacks recorded
(a delegation counts as verified only when the repository's tests passed
and no test file was touched), and a qualification rule
that is deliberately plain and printed with the numbers, never applied
silently: a class qualifies when it has at least three attempts and at
least one verified success, and verified over attempted is at least one
half. The harness reads the table and decides; the runner reports.

Delegation never touches the working tree: a detached worktree at HEAD, the
bounded attempt, the repository's own tests on the copy, the diff as a
patch file, and a verdict. Applying the patch is the user's act.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xyntetik_runner.shadow.attempt import AttemptResult, Budget, Workspace, attempt, runner_chat
from xyntetik_runner.shadow.baseline import Baseline
from xyntetik_runner.shadow.evidence import Disposition, EpisodeEvidence
from xyntetik_runner.shadow.scaffold import Scaffold
from xyntetik_runner.shadow.tasks import class_from_diff, classify, import_roots

MIN_ATTEMPTS = 3
MIN_VERIFIED = 1
MIN_FRACTION = 0.5
STRONG_ATTEMPTS = 5
STRONG_FRACTION = 0.8


@dataclass(frozen=True)
class Route:
    task_class: str
    model: str
    attempted: int
    verified: int

    @property
    def qualifies(self) -> bool:
        return (self.attempted >= MIN_ATTEMPTS and self.verified >= MIN_VERIFIED
                and self.verified / self.attempted >= MIN_FRACTION)

    @property
    def strong(self) -> bool:
        """Runner first: enough attempts and a high enough verified share
        that the harness should wait for the local result before doing the
        work itself. The rule is printed with the numbers, never applied
        silently."""
        return (self.qualifies and self.attempted >= STRONG_ATTEMPTS
                and self.verified / self.attempted >= STRONG_FRACTION)


def route_table(records: Iterable[EpisodeEvidence]) -> list[Route]:
    counts: dict[tuple[str, str], list[int]] = {}
    for r in records:
        if r.verifier is None:
            continue
        key = (r.identity.task_class, r.identity.model_sha256)
        c = counts.setdefault(key, [0, 0])
        c[0] += 1
        if r.disposition is Disposition.VERIFIED_LOCAL_ATTEMPT:
            c[1] += 1
    return sorted((Route(k[0], k[1], v[0], v[1]) for k, v in counts.items()),
                  key=lambda x: (x.task_class, x.model))


def render_routes(routes: list[Route]) -> str:
    if not routes:
        return ("no attempts recorded yet: nothing qualifies. Run the bench on this repository "
                "(shadow bench --repo . --models ...) to find out.")
    lines = [f"rule: a class qualifies with >= {MIN_ATTEMPTS} attempts, >= {MIN_VERIFIED} verified, "
             f"verified/attempted >= {MIN_FRACTION:g}", "",
             "| class | model | attempted | verified | qualifies |", "|---|---|---:|---:|---|"]
    for r in routes:
        model = r.model.split(":", 1)[-1][:40]
        lines.append(f"| {r.task_class} | {model} | {r.attempted} | {r.verified} | "
                     f"{'yes' if r.qualifies else 'no'} |")
    good = [r for r in routes if r.qualifies]
    lines.append("")
    lines.append("qualifying classes: " + (", ".join(sorted({r.task_class for r in good}))
                                          if good else "none"))
    return "\n".join(lines)


@dataclass(frozen=True)
class Delegation:
    request: str
    repo: str
    head: str
    patch_path: str
    changed_paths: tuple[str, ...]
    tests_exit: int | None
    tests_tail: str
    attempt: AttemptResult
    model: str
    wall_s: float
    task_class: str = "file"
    test_files_changed: tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        if not self.changed_paths:
            return "no change produced"
        if self.tests_exit == 0:
            return "tests passed on the scratch copy"
        if self.tests_exit is None:
            return "tests not run"
        return f"tests failed on the scratch copy (exit {self.tests_exit})"

    def to_json(self) -> str:
        data = asdict(self)
        data["attempt"] = {k: v for k, v in asdict(self.attempt).items() if k != "transcript"}
        data["verdict"] = self.verdict
        return json.dumps(data, indent=2)


def delegate(repo: Path, request: str, post_json: Any, model: str, *, python: str,
             budget: Budget | None = None, scaffold: Scaffold | None = None,
             context: tuple[str, ...] = (), out_dir: Path | None = None) -> Delegation:
    """One bounded attempt at ``request`` on a detached worktree of ``repo``
    at HEAD; the working tree is never touched."""
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if not head:
        raise RuntimeError(f"{repo} is not a git repository with a HEAD")
    budget = budget or Budget(max_turns=12, wall_s=900.0, test_runs=4)
    ws_dir = Path(tempfile.mkdtemp(prefix="xyntetik-shadow-delegate-"))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", "-q",
                               str(ws_dir), head], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {proc.stderr.strip()[:200]}")
        roots = import_roots(ws_dir)
        baseline = Baseline.capture(ws_dir)
        ws = Workspace(ws_dir, visible_tests=(), pythonpath=roots, python=python, budget=budget)
        chat = runner_chat(post_json, model, max_tokens=budget.max_tokens)
        result = attempt(request, ws, chat, budget=budget, context=context, scaffold=scaffold)
        changes = baseline.changes(ws_dir)
        tests_exit: int | None = None
        tail = ""
        if changes:
            ws.test_runs = 0  # the verdict run is not charged to the attempt's budget
            tail = ws.run_tests()
            first = tail.split("\n", 1)[0]
            tests_exit = int(first.split()[-1]) if first.startswith("exit code") else None
        diff = subprocess.run(["git", "-C", str(ws_dir), "diff"], capture_output=True,
                              text=True).stdout
        _tests, src, _other = classify(changes.paths)
        task_class, _n = ("multi-file" if len(src) > 1 else "file", 0)
        if len(src) == 1:
            u0 = subprocess.run(["git", "-C", str(ws_dir), "diff", "-U0", "--", src[0]],
                                capture_output=True, text=True).stdout
            try:
                post = (ws_dir / src[0]).read_text(encoding="utf-8")
            except OSError:
                post = ""
            task_class, _n = class_from_diff(u0, post)
        out_dir = out_dir or (Path.home() / ".xyntetik" / "shadow" / "delegations")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        patch = out_dir / f"{stamp}-{head[:8]}.patch"
        patch.write_text(diff, encoding="utf-8")
        return Delegation(request=request, repo=str(repo), head=head, patch_path=str(patch),
                          changed_paths=changes.paths, tests_exit=tests_exit,
                          tests_tail=tail[-2000:], attempt=result, model=model,
                          wall_s=round(time.monotonic() - t0, 1), task_class=task_class,
                          test_files_changed=tuple(_tests))
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(ws_dir)],
                       capture_output=True)
        shutil.rmtree(ws_dir, ignore_errors=True)
