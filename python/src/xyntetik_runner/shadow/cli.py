"""``python -m xyntetik_runner.shadow``: import, replay, report.

``import`` scans the frontier traces on this machine, pairs each episode with
the commits in its window, admits the ones that can be verified, and writes
one evidence record for every episode it could not admit, so the
denominator is complete before any model runs. ``replay`` runs the bounded
attempt on each admitted task against a runner endpoint and verifies it
with the frozen tests. ``report`` prints counts and both denominators.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xyntetik_runner.endpoint import RunnerEndpoint
from xyntetik_runner.shadow.attempt import Budget, Workspace, attempt, runner_chat
from xyntetik_runner.shadow.baseline import Baseline
from xyntetik_runner.shadow.evidence import (
    Disposition,
    EpisodeEvidence,
    Identity,
    VerifierOutcome,
    render,
    summarize,
)
from xyntetik_runner.shadow.importer import Episode, read_episodes, scan_all, write_episodes
from xyntetik_runner.shadow.tasks import Candidate, RepairTask, Rejection, build_task, choose, pair
from xyntetik_runner.shadow.verifier import ProtectedTests, fixed_ids, verify

HARNESS_VERSION = "shadow-0.1"
DEFAULT_OUT = Path.home() / ".xyntetik" / "shadow"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _band(chars: int) -> str:
    return "<2k" if chars < 2000 else "<8k" if chars < 8000 else ">=8k"


def _instrument_identity(verifier_id: str) -> Identity:
    """The identity of an episode no model touched: the instrument alone."""
    return Identity(project="", task_class="repair", context_band="", tool_set=(),
                    verifier_id=verifier_id, environment_id=platform.node(), model_sha256="none",
                    quant="none", template_sha256="none", runner_build="none", backend="none",
                    harness_version=HARNESS_VERSION)


def _append(path: Path, record: EpisodeEvidence) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


def _read_records(path: Path) -> list[EpisodeEvidence]:
    if not path.is_file():
        return []
    out: list[EpisodeEvidence] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(EpisodeEvidence.from_json(line))
    return out


def cmd_import(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks_dir = out / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    report = scan_all(Path(args.home) if args.home else None)
    episodes = [e for e in report.episodes if not args.source or e.source == args.source]
    n = write_episodes(episodes, out / "episodes.jsonl")
    print(f"scanned {report.files_read} trace files ({len(report.files_skipped)} unreadable), "
          f"{n} episodes", flush=True)
    evidence = out / "evidence.jsonl"
    known = {r.episode_id for r in _read_records(evidence)}
    counts: dict[str, int] = {}
    admitted = 0

    def record(e: Episode, rej: Rejection) -> None:
        _append(evidence, EpisodeEvidence(
            episode_id=e.episode_id, source=e.source, observed_at=e.started_at,
            disposition=rej.disposition, identity=_instrument_identity("none"),
            baseline_sha256="", patch_sha256=None, changed_paths=(), verifier=None, wall_s=0.0,
            reasons=(rej.reason,)))
        counts[rej.disposition.value] = counts.get(rej.disposition.value, 0) + 1

    # Pass 1: pair every episode; pass 2: one episode per fix commit.
    candidates: list[Candidate] = []
    for e in episodes:
        if e.episode_id in known:
            continue
        paired = pair(e)
        if isinstance(paired, Rejection):
            record(e, paired)
        else:
            candidates.extend(paired)
    chosen = choose(candidates)
    chosen_ids = {(c.episode.episode_id, c.solution) for c in chosen.values()}
    attributed: set[str] = set()
    for c in candidates:
        if (c.episode.episode_id, c.solution) in chosen_ids or c.episode.episode_id in attributed:
            continue
        if not any(x.episode.episode_id == c.episode.episode_id for x in chosen.values()):
            attributed.add(c.episode.episode_id)
            record(c.episode, Rejection(Disposition.INELIGIBLE,
                                        "fix commit attributed to a closer prompt"))
    built: set[str] = set()
    for c in chosen.values():
        if c.episode.episode_id in built:
            continue
        built.add(c.episode.episode_id)
        result = build_task(c.episode, c.repo, list(c.shas), out_dir=tasks_dir,
                            python=args.python, timeout_s=args.timeout)
        if isinstance(result, RepairTask):
            admitted += 1
            counts["admitted"] = counts.get("admitted", 0) + 1
            print(f"  admitted {result.task_id}: {result.expected_tests} frozen tests, "
                  f"{result.baseline_failing} fail at base, {result.src_files} source file(s)",
                  flush=True)
            continue
        record(c.episode, result)
    print("dispositions at import:", json.dumps(counts, sort_keys=True), flush=True)
    print(f"{admitted} task(s) admitted under {tasks_dir}", flush=True)
    return 0


def _tasks(out: Path, only: str | None) -> list[RepairTask]:
    tasks = [RepairTask.load(p) for p in sorted((out / "tasks").glob("*/task.json"))]
    return [t for t in tasks if not only or t.task_id == only]


def _worktree(task: RepairTask) -> Path:
    d = Path(tempfile.mkdtemp(prefix="xyntetik-shadow-attempt-"))
    proc = subprocess.run(["git", "-C", task.repo, "worktree", "add", "--detach", "-q", str(d),
                           task.base_sha], capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError(f"git worktree add failed: {proc.stderr.strip()[:200]}")
    return d


def _drop(task: RepairTask, d: Path) -> None:
    subprocess.run(["git", "-C", task.repo, "worktree", "remove", "--force", str(d)],
                   capture_output=True)
    shutil.rmtree(d, ignore_errors=True)


def cmd_replay(args: argparse.Namespace) -> int:
    out = Path(args.out)
    evidence = out / "evidence.jsonl"
    endpoint = RunnerEndpoint(args.endpoint, timeout=args.request_timeout)
    caps = endpoint.capabilities()
    models = [m.get("id") for m in caps.get("models", []) if isinstance(m, dict)]
    model = args.model or (str(models[0]) if models else "")
    if not model:
        print("error: the endpoint reports no model; pass --model", file=sys.stderr)
        return 2
    budget = Budget(max_turns=args.max_turns, wall_s=args.wall, max_tokens=args.max_tokens,
                    test_runs=args.test_runs)
    base_identity = Identity(
        project="", task_class="repair", context_band="", tool_set=(),
        verifier_id="", environment_id=f"{platform.node()}|{args.endpoint}",
        model_sha256=args.model_sha256 or f"unknown:{model}", quant=args.quant or "unknown",
        template_sha256="unknown", runner_build=str(caps.get("version") or "unknown"),
        backend=str(caps.get("backend") or "unknown"), harness_version=HARNESS_VERSION)
    done = {(r.episode_id, r.identity.model_sha256) for r in _read_records(evidence)
            if r.verifier is not None}
    ran = 0
    for task in _tasks(out, args.task):
        if args.limit and ran >= args.limit:
            break
        ident = replace(base_identity, project=Path(task.repo).name,
                        context_band=_band(len(task.request)),
                        tool_set=("list_files", "read_file", "write_file", "run_tests"),
                        verifier_id=f"commit-tests:{task.task_id}")
        if (task.episode_id, ident.model_sha256) in done:
            continue
        ran += 1
        print(f"[{task.task_id}] attempt with {model} ...", flush=True)
        ws_dir = _worktree(task)
        try:
            baseline = Baseline.capture(ws_dir)
            ws = Workspace(ws_dir, visible_tests=task.visible_test_files,
                           pythonpath=task.pythonpath, python=args.python, budget=budget)
            chat = runner_chat(endpoint.post_json, model, max_tokens=budget.max_tokens)
            result = attempt(task.request, ws, chat, budget=budget)
            protected = ProtectedTests.load(Path(task.protected_dir))
            outcome = verify(ws_dir, protected, baseline, timeout_s=args.timeout,
                             python=args.python, pythonpath=task.pythonpath)
            changes = baseline.changes(ws_dir)
            fixed: tuple[str, ...] = ()
            if changes and task.failing_at_base and outcome.passed is not None:
                fixed = fixed_ids(protected, task.failing_at_base, ws_dir, timeout_s=args.timeout,
                                  python=args.python, pythonpath=task.pythonpath)
            if outcome.passed is True:
                disposition = Disposition.VERIFIED_LOCAL_ATTEMPT
            elif outcome.passed is False:
                disposition = Disposition.LOCAL_FAILED
            else:
                disposition = Disposition.VERIFIER_INCONCLUSIVE
            rec = EpisodeEvidence(
                episode_id=task.episode_id, source=task.episode_id.split(":")[0],
                observed_at=_now(), disposition=disposition, identity=ident,
                baseline_sha256=baseline.sha256,
                patch_sha256=baseline.patch_sha256(ws_dir) if changes else None,
                changed_paths=changes.paths, verifier=outcome, wall_s=result.wall_s,
                resources={"prompt_tokens": float(result.prompt_tokens),
                           "completion_tokens": float(result.completion_tokens),
                           "turns": float(result.turns), "tool_calls": float(result.tool_calls),
                           "test_runs": float(result.test_runs),
                           "failing_at_base": float(len(task.failing_at_base)),
                           "fixed": float(len(fixed))},
                reasons=(f"attempt: {result.stop_reason}",
                         f"fixed {len(fixed)} of {len(task.failing_at_base)} failing at base",
                         *outcome.reasons))
            _append(evidence, rec)
            _keep_attempt(out, task, ident, result, ws_dir)
            print(f"[{task.task_id}] {disposition.value}: {result.stop_reason}, "
                  f"{result.turns} turns, {result.tool_calls} calls, {result.wall_s:.0f}s; "
                  f"fixed {len(fixed)}/{len(task.failing_at_base)}, verifier "
                  f"{outcome.passed_count}/{outcome.expected}"
                  f"{' tamper' if outcome.tamper else ''}", flush=True)
        finally:
            _drop(task, ws_dir)
    print(f"{ran} attempt(s) recorded in {evidence}", flush=True)
    return 0


def _keep_attempt(out: Path, task: RepairTask, ident: Identity, result: Any, ws_dir: Path) -> None:
    """Transcript and final diff of one attempt, beside its task: the local
    model's own words and edits, which the evidence record summarizes."""
    d = out / "attempts" / task.task_id
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{_now().replace(':', '')}-{ident.model_sha256[-12:].replace(':', '_')}"
    with (d / f"{stem}.transcript.jsonl").open("w", encoding="utf-8") as f:
        for m in result.transcript:
            f.write(json.dumps(m, sort_keys=True) + "\n")
    diff = subprocess.run(["git", "-C", str(ws_dir), "diff"], capture_output=True, text=True)
    (d / f"{stem}.diff").write_text(diff.stdout, encoding="utf-8")


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    records = _read_records(out / "evidence.jsonl")
    print(render(summarize(records)))
    if args.tasks:
        by_episode: dict[str, list[EpisodeEvidence]] = {}
        for r in records:
            if r.verifier is not None:
                by_episode.setdefault(r.episode_id, []).append(r)
        print()
        for task in _tasks(out, None):
            head = " ".join(task.request.split())[:72]
            print(f"{task.task_id}  ({task.expected_tests} frozen tests, "
                  f"{task.baseline_failing} fail at base, {task.src_files} src file(s))")
            print(f"    request: {head}")
            for r in by_episode.get(task.episode_id, []):
                v = r.verifier
                assert v is not None
                model = r.identity.model_sha256.split(":", 1)[-1]
                fixed = int(r.resources.get("fixed", 0))
                print(f"    {r.disposition.value:24s} {model[:40]:40s} fixed "
                      f"{fixed}/{task.baseline_failing}, verifier {v.passed_count}/{v.expected}"
                      f"{' tamper' if v.tamper else ''}  {r.reasons[0][:40]}")
            if not by_episode.get(task.episode_id):
                print("    (not attempted yet)")
    if args.by_reason:
        reasons: dict[str, int] = {}
        for r in records:
            key = f"{r.disposition.value}: {r.reasons[0] if r.reasons else ''}"
            reasons[key] = reasons.get(key, 0) + 1
        for key, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {key}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m xyntetik_runner.shadow", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="scan traces, admit verifiable tasks, record the rest")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--home", default="")
    p.add_argument("--source", choices=["codex", "claude_code"], default="")
    p.add_argument("--python", default=sys.executable, help="interpreter that runs the tests")
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("replay", help="attempt each admitted task and verify it")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--endpoint", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="")
    p.add_argument("--model-sha256", default="")
    p.add_argument("--quant", default="")
    p.add_argument("--task", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--test-runs", type=int, default=4)
    p.add_argument("--wall", type=float, default=900.0)
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("report", help="counts and both denominators")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--by-reason", action="store_true")
    p.add_argument("--tasks", action="store_true", help="one block per admitted task")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    fn: Any = args.fn
    return int(fn(args))
