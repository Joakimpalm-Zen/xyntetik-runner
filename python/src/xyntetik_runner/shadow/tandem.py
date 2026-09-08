"""Tandem: the local model works beside the harness, in the background,
where the evidence says it can.

The prompt hook decides in a few milliseconds: is this a request typed in
a repository where the local model has verified successes on record? If
so, a bounded delegation starts detached on a scratch worktree while the
frontier model works in the foreground, and the harness is told so. When
the local attempt ends verified, the stop hook surfaces the patch once:
the user sees both results and applies or not. Nothing is applied for
them, nothing blocks their prompt, and no attempt starts where the ledger
holds no evidence.

The funnel widens by evidence alone. Every delegation is recorded in the
ledger with the class its own diff had, so a repository where the local
model keeps producing verified patches qualifies in more classes, and an
adapter that raised the held-out count raises the counts here too.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from xyntetik_runner.process import spawn_detached
from xyntetik_runner.shadow.evidence import EpisodeEvidence
from xyntetik_runner.shadow.routes import Route, route_table

STATE_DIR_REL = Path(".xyntetik") / "shadow" / "delegations"
RUNNING_WALL_S = 1200.0  # a delegation older than this without an end is treated as dead
MIN_REQUEST_CHARS = 24


@dataclass
class DelegationState:
    id: str
    session_id: str
    request: str
    repo: str
    started_at: float
    status: str = "running"  # running | done | error
    verdict: str = ""
    patch_path: str = ""
    changed_paths: tuple[str, ...] = ()
    tests_exit: int | None = None
    task_class: str = ""
    model: str = ""
    surfaced: bool = False
    ended_at: float | None = None
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status == "done" and self.tests_exit == 0 and bool(self.changed_paths)

    @property
    def running(self) -> bool:
        return self.status == "running" and time.time() - self.started_at < RUNNING_WALL_S

    def save(self, home: Path) -> Path:
        d = home / STATE_DIR_REL
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> DelegationState | None:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            d["changed_paths"] = tuple(d.get("changed_paths") or ())
            return cls(**d)
        except (ValueError, TypeError, OSError):
            return None


def states(home: Path) -> list[DelegationState]:
    d = home / STATE_DIR_REL
    if not d.is_dir():
        return []
    out = [s for p in sorted(d.glob("*.json")) if (s := DelegationState.load(p)) is not None]
    return sorted(out, key=lambda s: s.started_at)


def new_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid() % 10000:04d}"


# ---------------------------------------------------------------- the gate

def project_routes(records: Sequence[EpisodeEvidence], project: str, model_sha256: str) -> list[Route]:
    """The route table restricted to one repository and one model."""
    return route_table(r for r in records
                       if r.identity.project == project and r.identity.model_sha256 == model_sha256)


def qualifies(records: Sequence[EpisodeEvidence], project: str, model_sha256: str) -> Route | None:
    """The best qualifying route for this repository and model, or None.
    Any qualifying class opens the funnel: the verifier decides per attempt."""
    good = [r for r in project_routes(records, project, model_sha256) if r.qualifies]
    if not good:
        return None
    return max(good, key=lambda r: (r.verified / r.attempted, r.verified))


def looks_like_a_task(prompt: str) -> bool:
    """Cheap and stated: a request long enough to be work, not a slash
    command, not a one-word answer. The verifier judges the rest."""
    p = prompt.strip()
    if len(p) < MIN_REQUEST_CHARS or p.startswith("/"):
        return False
    return True


# ---------------------------------------------------------------- hooks

def hook_prompt(home: Path, *, session_id: str, cwd: str, prompt: str, repo: Path | None,
                records: Sequence[EpisodeEvidence], model: str, model_sha256: str, python: str,
                out: str, spawn: Callable[[list[str]], Any] | None = None) -> dict[str, Any] | None:
    """Decide and start; return the hook's JSON for the harness, or None."""
    spawn = spawn or spawn_detached  # resolved at call time, so a test can stand in
    notes: list[str] = []
    # a finished attempt from an earlier turn of this session, not yet shown
    for s in states(home):
        if s.session_id == session_id and s.status != "running" and not s.surfaced:
            s.surfaced = True
            s.save(home)
            if s.verified:
                notes.append(f"shadow: the local model's background attempt at an earlier request "
                             f"(\"{s.request[:80]}\") ended verified: its tests passed on a scratch copy. "
                             f"Patch: {s.patch_path}. Tell the user in one line and offer "
                             f"`git apply {s.patch_path}`; never apply it yourself.")
    if repo is not None and model and looks_like_a_task(prompt):
        route = qualifies(records, repo.name, model_sha256)
        busy = any(s.running for s in states(home))
        if route is not None and not busy:
            state = DelegationState(id=new_id(), session_id=session_id, request=prompt, repo=str(repo),
                                    started_at=time.time(), model=model)
            state.save(home)
            argv = [python, "-m", "xyntetik_runner.shadow", "delegate", "--repo", str(repo),
                    "--request", prompt, "--home", str(home), "--python", python, "--out", out,
                    "--record", state.id]
            try:
                spawn(argv)
            except OSError as e:
                state.status = "error"
                state.error = str(e)
                state.ended_at = time.time()
                state.save(home)
            else:
                if route.strong:
                    notes.append(f"shadow: runner first. The local model ({Path(model).name}) has "
                                 f"{route.verified} verified of {route.attempted} attempts on record for "
                                 f"{route.task_class} tasks in {repo.name}, so it is attempting this request now "
                                 f"on a scratch copy (delegation {state.id}). Before doing the work yourself, "
                                 f"wait for it: run `{python} -m xyntetik_runner.shadow delegations --wait {state.id}`. "
                                 "If it ends verified, show the patch in one line, offer `git apply <patch>` and "
                                 "stop; if not, do the work as usual.")
                else:
                    notes.append(f"shadow: the local model ({Path(model).name}) is attempting this request in "
                                 f"the background on a scratch copy of {repo.name}, because it has "
                                 f"{route.verified} verified of {route.attempted} attempts on record for "
                                 f"{route.task_class} tasks here (delegation {state.id}). Work as usual; if "
                                 f"it ends verified you will be told. `shadow delegations` shows it.")
    if not notes:
        return None
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                   "additionalContext": "\n".join(notes)}}


def hook_stop(home: Path, *, session_id: str) -> dict[str, Any] | None:
    """Surface a verified background result once, at the end of the turn."""
    for s in states(home):
        if s.session_id == session_id and s.status != "running" and not s.surfaced:
            s.surfaced = True
            s.save(home)
            if s.verified:
                return {"decision": "block",
                        "reason": (f"shadow: the local model's background attempt at \"{s.request[:80]}\" "
                                   f"ended verified: its tests passed on a scratch copy of {Path(s.repo).name}. "
                                   f"Patch: {s.patch_path}. Tell the user in one line and offer "
                                   f"`git apply {s.patch_path}`; never apply it yourself. Then stop.")}
    return None


def wait_for(home: Path, delegation_id: str, *, timeout_s: float = RUNNING_WALL_S,
             poll_s: float = 5.0, sleep: Callable[[float], None] = time.sleep) -> DelegationState | None:
    """Block until the delegation ends or the wall passes; the ended state,
    or the running one when the wall passed, or None if unknown."""
    deadline = time.time() + timeout_s
    while True:
        found = [s for s in states(home) if s.id == delegation_id]
        if not found:
            return None
        s = found[0]
        if s.status != "running" or time.time() >= deadline or not s.running:
            if s.status != "running":
                s.surfaced = True
                s.save(home)
            return s
        sleep(poll_s)


def render_delegations(home: Path, *, session_id: str = "") -> str:
    rows = [s for s in states(home) if not session_id or s.session_id == session_id]
    if not rows:
        return "no background delegations recorded"
    lines = ["| started | repo | request | status | verdict | class | patch |", "|---|---|---|---|---|---|---|"]
    for s in rows[-20:]:
        status = "running" if s.running else ("stale" if s.status == "running" else s.status)
        lines.append(f"| {time.strftime('%Y-%m-%d %H:%M', time.gmtime(s.started_at))} | {Path(s.repo).name} | "
                     f"{s.request[:40]} | {status} | {s.verdict or s.error} | {s.task_class} | {s.patch_path} |")
    return "\n".join(lines)


def emit(payload: dict[str, Any] | None) -> None:
    """The hook's JSON on stdout, nothing else, so the harness can parse it."""
    if payload is not None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
