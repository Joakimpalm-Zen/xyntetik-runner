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
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xyntetik_runner.endpoint import RunnerEndpoint
from xyntetik_runner.process import ManagedRunner, ServerLaunch
from xyntetik_runner.shadow.attempt import Budget, Workspace, attempt, probe_speed, runner_chat
from xyntetik_runner.shadow.baseline import Baseline, file_sha256
from xyntetik_runner.shadow.evidence import (
    Disposition,
    EpisodeEvidence,
    Identity,
    Summary,
    VerifierOutcome,
    render,
    summarize,
)
from xyntetik_runner.shadow.importer import CAPTURE_FILE, Episode, read_episodes, scan_all, write_episodes
from xyntetik_runner.shadow import adapt
from xyntetik_runner.shadow import tandem
from xyntetik_runner.shadow.install import (harness_present, install, read_config, render_candidates,
                                            set_config_adapter, set_config_key, suggest_model, uninstall)
from xyntetik_runner.process import spawn_detached
from xyntetik_runner.shadow.routes import delegate, render_routes, route_table
from xyntetik_runner.shadow.server import (DEFAULT_TTL, render_status as server_status, served_runner,
                                           stop_server)
from xyntetik_runner.shadow.bank import build_bank
from xyntetik_runner.shadow.optimize import (
    TaskOutcome,
    optimize,
    result_json,
    runner_reflect,
)
from xyntetik_runner.shadow.scaffold import Scaffold
from xyntetik_runner.shadow.tasks import (
    Candidate,
    RepairTask,
    Rejection,
    build_task,
    choose,
    canonical,
    pair,
    repos_under,
)
from xyntetik_runner.shadow.verifier import ProtectedTests, fixed_ids, verify


def _run_attempt_factory(args: argparse.Namespace, endpoint: RunnerEndpoint, model: str,
                         budget: Budget) -> Any:
    """One attempt plus verification on a scratch worktree, as the optimizer's
    reward function: fixed count, verdict and the failure text."""
    def run(task: RepairTask, scaffold: Scaffold) -> TaskOutcome:
        ws_dir = _worktree(task)
        try:
            baseline = Baseline.capture(ws_dir)
            ws = Workspace(ws_dir, visible_tests=task.visible_test_files,
                           pythonpath=task.pythonpath, python=args.python, budget=budget)
            chat = runner_chat(endpoint.post_json, model, max_tokens=budget.max_tokens)
            result = attempt(task.request, ws, chat, budget=budget, context=task.context,
                             scaffold=scaffold)
            protected = ProtectedTests.load(Path(task.protected_dir))
            outcome = verify(ws_dir, protected, baseline, timeout_s=args.timeout,
                             python=args.python, pythonpath=task.pythonpath)
            fixed: tuple[str, ...] = ()
            if baseline.changes(ws_dir) and task.failing_at_base and outcome.passed is not None:
                fixed = fixed_ids(protected, task.failing_at_base, ws_dir, timeout_s=args.timeout,
                                  python=args.python, pythonpath=task.pythonpath)
            tail = [m for m in result.transcript if m.get("role") == "tool"][-2:]
            feedback = (f"stop: {result.stop_reason}; verifier: {'; '.join(outcome.reasons)}\n"
                        + "\n".join(str(m.get("content", ""))[-600:] for m in tail))
            return TaskOutcome(task_id=task.task_id, fixed=len(fixed),
                               failing_at_base=len(task.failing_at_base),
                               verified=outcome.passed is True, feedback=feedback)
        finally:
            _drop(task, ws_dir)
    return run

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
    do_import(Path(args.out), home=Path(args.home) if args.home else None, source=args.source,
              python=args.python, timeout=args.timeout)
    return 0


def do_import(out: Path, *, home: Path | None, source: str = "", python: str = sys.executable,
              timeout: float = 600.0) -> int:
    """Scan the harness traces and the capture file, pair, choose, build the
    tasks; returns how many were admitted this time."""
    out.mkdir(parents=True, exist_ok=True)
    tasks_dir = out / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    report = scan_all(home)
    episodes = [e for e in report.episodes if not source or e.source == source]
    n = write_episodes(episodes, out / "episodes.jsonl")
    print(f"scanned {report.files_read} trace files ({len(report.files_skipped)} unreadable), "
          f"{n} episodes", flush=True)
    evidence = out / "evidence.jsonl"
    known = {r.episode_id for r in _read_records(evidence)}
    known |= {t.episode_id for t in _tasks(out, None)}
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
                            python=python, timeout_s=timeout)
        if isinstance(result, RepairTask):
            admitted += 1
            counts["admitted"] = counts.get("admitted", 0) + 1
            # Eligible and waiting for a model: the record exists now so the
            # denominator is complete before any replay.
            record(c.episode, Rejection(Disposition.NOT_ATTEMPTED_RESOURCE,
                                        f"admitted as {result.task_id}; not attempted yet"))
            print(f"  admitted {result.task_id}: {result.expected_tests} frozen tests, "
                  f"{result.baseline_failing} fail at base, {result.src_files} source file(s)",
                  flush=True)
            continue
        record(c.episode, result)
    print("dispositions at import:", json.dumps(counts, sort_keys=True), flush=True)
    print(f"{admitted} task(s) admitted under {tasks_dir}", flush=True)
    return admitted


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


@dataclass(frozen=True)
class ReplayOptions:
    python: str
    timeout: float
    max_turns: int = 12
    max_tokens: int = 1500
    test_runs: int = 4
    wall: float = 900.0
    min_tps: float = 15.0
    scaffold_path: str = ""
    model_sha256: str = ""
    quant: str = ""
    task: str = ""
    limit: int = 0
    endpoint_label: str = ""


_CLASS_RANK = {"function": 0, "file": 1, "multi-file": 2}


def replay_order(tasks: list[RepairTask]) -> list[RepairTask]:
    """Cheapest evidence first: the class a person delegates (function),
    then file, then multi-file; within a class the fewest failing tests.
    A limited replay therefore spends its wall clock where a verified
    success is likeliest and shortest."""
    return sorted(tasks, key=lambda t: (_CLASS_RANK.get(t.task_class, 3), t.baseline_failing, t.task_id))


def run_replay(out: Path, endpoint: RunnerEndpoint, opts: ReplayOptions, *,
               model: str = "") -> tuple[int, float, str]:
    """Attempt every admitted task under ``out`` against ``endpoint`` and
    record the verdicts; returns (attempts recorded, probe tok/s, model id).
    Raises ``RuntimeError`` when the endpoint has no model or the fit probe
    is below the floor, so a caller can report and move on."""
    evidence = out / "evidence.jsonl"
    caps = endpoint.capabilities()
    models = [m.get("id") for m in caps.get("models", []) if isinstance(m, dict)]
    model = model or (str(models[0]) if models else "")
    if not model:
        raise RuntimeError("the endpoint reports no model; pass --model")
    budget = Budget(max_turns=opts.max_turns, wall_s=opts.wall, max_tokens=opts.max_tokens,
                    test_runs=opts.test_runs)
    scaffold = Scaffold.load(Path(opts.scaffold_path)) if opts.scaffold_path else Scaffold.base()
    scaffold_id = "base" if not opts.scaffold_path else scaffold.sha256
    print(f"scaffold: {scaffold.name} ({scaffold_id[:12]})", flush=True)
    # Fit first: a model that crawls has spilled the device, and an attempt
    # at that speed measures the wall clock, not the model. Refuse it.
    tps = probe_speed(endpoint.post_json, model)
    print(f"probe: {model} decodes at {tps:.1f} tok/s (floor {opts.min_tps:g})", flush=True)
    if tps < opts.min_tps:
        raise RuntimeError(f"fit-first: {tps:.1f} tok/s is below the floor of {opts.min_tps:g}; "
                           "serve a model and context that fit the device (see runner --fit) "
                           "or lower --min-tps")
    base_identity = Identity(
        project="", task_class="file", context_band="", tool_set=(),
        verifier_id="", environment_id=f"{platform.node()}|{opts.endpoint_label or endpoint.base_url}",
        model_sha256=opts.model_sha256 or f"unknown:{model}", quant=opts.quant or "unknown",
        template_sha256="unknown", runner_build=str(caps.get("version") or "unknown"),
        backend=str(caps.get("backend") or "unknown"), harness_version=HARNESS_VERSION,
        scaffold_sha256=scaffold_id)
    done = {(r.episode_id, r.identity.model_sha256, r.identity.scaffold_sha256)
            for r in _read_records(evidence) if r.verifier is not None}
    ran = 0
    for task in replay_order(_tasks(out, opts.task)):
        if opts.limit and ran >= opts.limit:
            break
        ident = replace(base_identity, project=Path(task.repo).name, task_class=task.task_class,
                        context_band=_band(len(task.request)),
                        tool_set=("list_files", "read_file", "write_file", "edit_file", "run_tests"),
                        verifier_id=f"commit-tests:{task.task_id}")
        if (task.episode_id, ident.model_sha256, ident.scaffold_sha256) in done:
            continue
        ran += 1
        print(f"[{task.task_id}] attempt with {model} ...", flush=True)
        ws_dir = _worktree(task)
        try:
            baseline = Baseline.capture(ws_dir)
            ws = Workspace(ws_dir, visible_tests=task.visible_test_files,
                           pythonpath=task.pythonpath, python=opts.python, budget=budget)
            chat = runner_chat(endpoint.post_json, model, max_tokens=budget.max_tokens)
            result = attempt(task.request, ws, chat, budget=budget, context=task.context,
                             scaffold=scaffold)
            protected = ProtectedTests.load(Path(task.protected_dir))
            outcome = verify(ws_dir, protected, baseline, timeout_s=opts.timeout,
                             python=opts.python, pythonpath=task.pythonpath)
            changes = baseline.changes(ws_dir)
            fixed: tuple[str, ...] = ()
            if changes and task.failing_at_base and outcome.passed is not None:
                fixed = fixed_ids(protected, task.failing_at_base, ws_dir, timeout_s=opts.timeout,
                                  python=opts.python, pythonpath=task.pythonpath)
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
                           "fixed": float(len(fixed)), "probe_tps": round(tps, 2)},
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
    return ran, tps, model


def cmd_replay(args: argparse.Namespace) -> int:
    opts = ReplayOptions(python=args.python, timeout=args.timeout, max_turns=args.max_turns,
                         max_tokens=args.max_tokens, test_runs=args.test_runs, wall=args.wall,
                         min_tps=args.min_tps, scaffold_path=args.scaffold,
                         model_sha256=args.model_sha256, quant=args.quant, task=args.task,
                         limit=args.limit, endpoint_label=args.endpoint)
    try:
        run_replay(Path(args.out), RunnerEndpoint(args.endpoint, timeout=args.request_timeout),
                   opts, model=args.model)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """R14.5.3: bank + fit probe + replay per model + the verified table."""
    out = Path(args.out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    if not _tasks(out, None):
        if not args.repo:
            print("error: no tasks under", out / "tasks", "and no --repo to build them from",
                  file=sys.stderr)
            return 2
        entries = build_bank(Path(args.repo), out_dir=out / "tasks", python=args.python,
                             limit=args.limit, max_commits=args.max_commits,
                             max_src_files=args.max_src_files, timeout_s=args.timeout)
        print(f"bank: {sum(1 for e in entries if e.task)} task(s) from {len(entries)} commits",
              flush=True)
    tasks = _tasks(out, None)
    if not tasks:
        print("error: the repository yielded no verifiable task", file=sys.stderr)
        return 2
    classes: dict[str, int] = {}
    for t in tasks:
        classes[t.task_class] = classes.get(t.task_class, 0) + 1
    print(f"{len(tasks)} task(s): " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())),
          flush=True)
    opts = ReplayOptions(python=args.python, timeout=args.timeout, max_turns=args.max_turns,
                         max_tokens=args.max_tokens, test_runs=args.test_runs, wall=args.wall,
                         min_tps=args.min_tps, scaffold_path=args.scaffold)
    rows: list[dict[str, Any]] = []
    arms: list[tuple[str, str]] = [("endpoint", u) for u in args.endpoints.split(",") if u]
    arms += [("model", m) for m in args.models.split(",") if m]
    if not arms:
        print("error: pass --models a.gguf,b.gguf (with --runner) or --endpoints url,url",
              file=sys.stderr)
        return 2
    for kind, target in arms:
        label = target
        managed: ManagedRunner | None = None
        try:
            if kind == "model":
                path = Path(target)
                if not path.is_file():
                    rows.append({"model": label, "error": "no such file"})
                    continue
                sha = file_sha256(path)
                port = _free_port()
                launch = ServerLaunch(executable=args.runner, model=path, port=port,
                                      context_size=args.ctx, gpu=args.gpu,
                                      threads=args.threads or None,
                                      extra_args=tuple(a for a in args.runner_args.split() if a))
                managed = ManagedRunner(launch)
                print(f"[{path.name}] starting runner on port {port} ...", flush=True)
                if not managed.start(timeout=args.start_timeout):
                    rows.append({"model": label, "error": "runner did not start"})
                    continue
                endpoint = RunnerEndpoint(managed.base_url, timeout=args.request_timeout)
                run_opts = replace(opts, model_sha256=sha, quant=_quant_from_name(path.name),
                                   endpoint_label=path.name)
            else:
                endpoint = RunnerEndpoint(target, timeout=args.request_timeout)
                run_opts = replace(opts, endpoint_label=target)
            ran, tps, model = run_replay(out, endpoint, run_opts)
            rows.append({"model": model, "model_sha256": run_opts.model_sha256 or f"unknown:{model}",
                         "tps": round(tps, 1), "attempts": ran})
        except RuntimeError as e:
            rows.append({"model": label, "error": str(e)})
            print(f"[{label}] {e}", flush=True)
        finally:
            if managed is not None:
                managed.stop()
    summary = summarize(_read_records(out / "evidence.jsonl"))
    table = _bench_table(summary, rows, len(tasks), classes)
    (out / "bench.md").write_text(table + "\n", encoding="utf-8")
    (out / "bench.json").write_text(json.dumps({"tasks": len(tasks), "classes": classes,
                                                "arms": rows, "report": render(summary)},
                                               indent=2) + "\n", encoding="utf-8")
    print(table, flush=True)
    print(f"written {out / 'bench.md'} and bench.json", flush=True)
    return 0


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _quant_from_name(name: str) -> str:
    m = re.search(r"(IQ\d_\w+|Q\d_K_[MSL]|Q\d_K|Q\d_\d|F16|BF16|F32|NVFP4|MXFP4)", name, re.I)
    return m.group(1).upper() if m else "unknown"


def _bench_table(summary: Summary, rows: list[dict[str, Any]], n_tasks: int,
                 classes: dict[str, int]) -> str:
    lines = [f"# Local capability bench: {n_tasks} verified tasks "
             f"({', '.join(f'{v} {k}' for k, v in sorted(classes.items()))})", "",
             "| model | quant | tok/s | attempted | verified | failed | per class |",
             "|---|---|---:|---:|---:|---:|---|"]
    by_stack = {c.model: c for c in summary.cohorts}
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | | | | | | refused: {r['error']} |")
            continue
        c = by_stack.get(r["model_sha256"])
        if c is None:
            lines.append(f"| {r['model']} | | {r['tps']} | 0 | 0 | 0 | |")
            continue
        per = "; ".join(f"{k}: {v}/{a}" for k, a, v, _f in c.classes)
        lines.append(f"| {r['model']} | {_quant_from_name(r['model'])} | {r['tps']} | {c.attempted} | "
                     f"{c.verified} | {c.failed} | {per} |")
    lines += ["", "Counts before rates: verified is all frozen tests passing; per class is "
              "verified over attempted for that task class. No percentage before thirty "
              "independent eligible tasks."]
    return "\n".join(lines)


def _keep_attempt(out: Path, task: RepairTask, ident: Identity, result: Any, ws_dir: Path) -> None:
    """Transcript and final diff of one attempt, beside its task: the local
    model's own words and edits, which the evidence record summarizes."""
    d = out / "attempts" / task.task_id
    d.mkdir(parents=True, exist_ok=True)
    stem = (f"{_now().replace(':', '')}-{ident.model_sha256[-12:].replace(':', '_')}"
            f"-{ident.scaffold_sha256[:8]}")
    with (d / f"{stem}.transcript.jsonl").open("w", encoding="utf-8") as f:
        for m in result.transcript:
            f.write(json.dumps(m, sort_keys=True) + "\n")
    diff = subprocess.run(["git", "-C", str(ws_dir), "diff"], capture_output=True, text=True)
    (d / f"{stem}.diff").write_text(diff.stdout, encoding="utf-8")


def cmd_bank(args: argparse.Namespace) -> int:
    out = Path(args.out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    entries = build_bank(Path(args.repo), out_dir=out / "tasks", python=args.python,
                         limit=args.limit, max_commits=args.max_commits,
                         max_src_files=args.max_src_files, timeout_s=args.timeout)
    admitted = [e for e in entries if e.task is not None]
    for e in admitted:
        assert e.task is not None
        print(f"  admitted {e.task.task_id}: {e.task.expected_tests} frozen tests, "
              f"{e.task.baseline_failing} fail at base", flush=True)
    reasons: dict[str, int] = {}
    for e in entries:
        if e.task is None:
            key = e.reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    print(f"{len(admitted)} task(s) admitted from {len(entries)} commits; rejected:",
          json.dumps(reasons, sort_keys=True), flush=True)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    tasks = [t for t in _tasks(out, None)
             if t.baseline_failing <= args.max_failing and t.baseline_failing < t.expected_tests]
    # A task where everything fails at base (a test-suite move, a refactor)
    # carries no gradient for any scaffold; the band keeps the signal.
    if not tasks:
        print("error: no tasks under", out / "tasks", "within --max-failing", file=sys.stderr)
        return 2
    print(f"{len(tasks)} task(s) in the difficulty band", flush=True)
    endpoint = RunnerEndpoint(args.endpoint, timeout=args.request_timeout)
    caps = endpoint.capabilities()
    models = [m.get("id") for m in caps.get("models", []) if isinstance(m, dict)]
    model = args.model or (str(models[0]) if models else "")
    if not model:
        print("error: the endpoint reports no model; pass --model", file=sys.stderr)
        return 2
    tps = probe_speed(endpoint.post_json, model)
    print(f"probe: {model} decodes at {tps:.1f} tok/s (floor {args.min_tps:g})", flush=True)
    if tps < args.min_tps:
        print("error: fit-first: below the floor; serve a model and context that fit",
              file=sys.stderr)
        return 2
    budget = Budget(max_turns=args.max_turns, wall_s=args.wall, max_tokens=args.max_tokens,
                    test_runs=args.test_runs)
    base = Scaffold.load(Path(args.scaffold)) if args.scaffold else Scaffold.base()
    save_dir = out / "scaffolds"
    run = _run_attempt_factory(args, endpoint, model, budget)
    reflect = runner_reflect(endpoint.post_json, model)
    result = optimize(tasks, base, run, reflect, generations=args.generations, batch=args.batch,
                      holdout_fraction=args.holdout, seed=args.seed,
                      rollout_budget=args.rollout_budget, save_dir=save_dir,
                      log=lambda s: print(s, flush=True))
    (out / "optimize.json").write_text(result_json(result) + "\n", encoding="utf-8")
    best_path = save_dir / "best.json"
    result.best.save(best_path)
    print(f"best scaffold {result.best.sha256[:12]} ({result.best.name}) -> {best_path}; "
          f"rollouts {result.rollouts}; held-out gain: {result.held_out_gain}", flush=True)
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    """Append one prompt or stop line from a hook. Reads the hook's JSON on
    stdin (Claude Code passes ``session_id``, ``cwd`` and, on prompt
    submission, ``prompt``); records the request, the directory, the time
    and the repository HEAD, and nothing the assistant produced.
    ``--summary`` prints counts over the capture file instead."""
    if args.summary:
        path = Path(args.file) if args.file else Path.home() / CAPTURE_FILE
        rows: list[dict[str, Any]] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        prompts = [r for r in rows if r.get("event") == "prompt"]
        with_heads = sum(1 for r in prompts if r.get("heads"))
        print(f"captured: {len(rows)} lines, {len(prompts)} prompts, {with_heads} with repository "
              f"heads, {len({r.get('session_id') for r in rows})} sessions ({path})")
        return 0
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if args.event not in ("prompt", "stop"):
        print("error: --event prompt|stop is required (or --summary)", file=sys.stderr)
        return 2
    cwd = str(args.cwd or data.get("cwd") or Path.cwd())
    head = subprocess.run(["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    # Every repository at or under the directory, so a session started from a
    # parent directory still yields exact ranges for the repositories inside.
    heads = {}
    for repo in repos_under(Path(cwd)):
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        if sha:
            heads[str(canonical(repo))] = sha
    rec: dict[str, Any] = {"event": args.event, "timestamp": _now(),
                           "session_id": str(args.session or data.get("session_id") or ""),
                           "cwd": cwd, "head": head, "heads": heads, "tool": args.tool}
    if args.event == "prompt":
        rec["prompt"] = str(data.get("prompt") or args.prompt or "")
    path = Path(args.file) if args.file else Path.home() / CAPTURE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    # tandem: a background attempt where the evidence allows it, and the
    # harness told; every failure here is swallowed, a hook never blocks
    try:
        home = Path(args.home) if args.home else Path.home()
        cfg = read_config(home)
        model = str(cfg.get("model") or "")
        if cfg.get("tandem", True) and model and Path(model).is_file():
            out = str(Path(str(cfg.get("out") or DEFAULT_OUT)).expanduser())
            if args.event == "prompt":
                repo_or_none = _repo_of(Path(cwd))
                records = _read_records(Path(out) / "evidence.jsonl")
                tandem.emit(tandem.hook_prompt(home, session_id=rec["session_id"], cwd=cwd,
                                               prompt=rec["prompt"], repo=repo_or_none, records=records,
                                               model=model, model_sha256=_model_sha(model),
                                               python=args.python or sys.executable, out=out,
                                               turns=int(str(cfg.get("tandem_turns") or tandem.DEFAULT_TURNS)),
                                               wall_s=float(str(cfg.get("tandem_wall") or tandem.DEFAULT_WALL_S))))
            else:
                tandem.emit(tandem.hook_stop(home, session_id=rec["session_id"]))
    except Exception:  # noqa: BLE001 - a hook must never fail the prompt
        pass
    return 0


def _repo_of(cwd: Path) -> Path | None:
    top = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"], capture_output=True,
                         text=True).stdout.strip()
    return Path(top) if top else None


_MODEL_SHA_CACHE: dict[str, tuple[float, str]] = {}


def _model_sha(model: str) -> str:
    """sha256 of the model file, cached on mtime so a hook does not hash
    gigabytes on every prompt."""
    try:
        mtime = Path(model).stat().st_mtime
    except OSError:
        return ""
    cache = Path.home() / ".xyntetik" / "shadow" / "model-sha.json"
    try:
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("model") == model and d.get("mtime") == mtime:
            return str(d.get("sha256") or "")
    except (OSError, ValueError):
        pass
    sha = file_sha256(Path(model))
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"model": model, "mtime": mtime, "sha256": sha}), encoding="utf-8")
    except OSError:
        pass
    return sha


def cmd_install(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    if args.model and not Path(args.model).expanduser().is_file():
        print(f"error: model not found: {args.model}\n"
              "  --shadow-mode -m takes the path of a GGUF file on this machine; nothing was written",
              file=sys.stderr)
        return 2
    has_claude, has_codex = harness_present(home)
    claude = not args.no_claude and (has_claude or args.claude)
    codex = not args.no_codex and (has_codex or args.codex)
    if not (claude or codex):
        print("nothing to install: neither ~/.claude nor ~/.codex exists here "
              "(pass --claude or --codex to force)", file=sys.stderr)
        return 2
    plan = []
    if claude:
        plan.append(f"Claude Code: prompt and stop capture hooks merged into {home / '.claude' / 'settings.json'} "
                    "(backup beside it), a /shadow skill, and a marked note in "
                    f"{home / '.claude' / 'CLAUDE.md'} saying Runner is here and what it can do")
    if codex:
        plan.append(f"Codex: a /shadow prompt under {home / '.codex' / 'prompts'} and a marked note in "
                    f"{home / '.codex' / 'AGENTS.md'} saying Runner is here and what it can do")
    if not args.no_tandem:
        plan.append("tandem: in a repository where the local model has verified successes on record, "
                    "each request also gets a background local attempt on a scratch copy; a verified "
                    "patch is offered, never applied ('shadow tandem off' stops it)")
    plan.append(f"a capability sheet at {home / '.xyntetik' / 'shadow' / 'runner-capabilities.md'} "
                "(from 'runner --help', '--caps' and the README) the assistants read before "
                "proposing another local inference tool")
    picked = ""
    if args.model:
        plan.append(f"model for offloading: {args.model} (served by {args.runner} when asked)")
    elif not args.no_pick:
        best, cands = suggest_model(args.runner, home, ctx=args.ctx)
        if best is not None:
            picked = str(best.path)
            plan.append(f"model for offloading: {picked} (found on disk, the largest that fits at "
                        f"ctx {args.ctx} by 'runner --fit'; pass -m to choose another)")
        else:
            plan.append("no model for offloading yet: " + ("none of the GGUF files found fits at "
                        f"ctx {args.ctx}" if cands else "no GGUF file found") +
                        "; run 'runner --shadow-mode -m MODEL.gguf' later")
        print("models looked at:")
        print(render_candidates(cands))
    print("shadow mode will write:")
    for line in plan:
        print(f"  - {line}")
    print("  Only your own requests, directories, times and commit ids are ever recorded; "
          "nothing the assistant produces. Nothing runs until you submit a prompt, and the "
          "hooks can never block one. 'shadow uninstall' removes exactly this.")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("error: not a terminal; pass --yes to confirm", file=sys.stderr)
            return 2
        answer = input("Install shadow mode now? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("not installed")
            return 1
    done = install(home, python=args.python, pythonpath=args.pythonpath or None, out=args.out,
                   claude=claude, codex=codex, model=args.model or picked, runner=args.runner, ctx=args.ctx,
                   gpu=args.gpu, threads=args.threads)
    if done.config:
        if args.no_tandem:
            set_config_key(home, "tandem", False)
        print(f"config: {done.config}")
    if done.settings:
        print(f"claude code: {done.hooks_added} hook(s) added to {done.settings} "
              f"(backup beside it); skill {done.claude_skill}")
    if done.codex_prompt:
        print(f"codex: prompt {done.codex_prompt}")
    if done.sheet:
        print(f"capabilities: {done.sheet}; noted in {', '.join(str(n) for n in done.notes)}")
    print("nothing runs until a prompt is submitted; the hooks never block one; "
          "'shadow uninstall' removes exactly this")
    print()
    print("what happens next:")
    print("  1. keep working as you do; each prompt and each finished turn is noted with the")
    print("     directory, the time and the repository's commit id, nothing else")
    print("  2. after a few sessions, ask /shadow" + (" in Claude Code" if claude else "") +
          (" (the shadow prompt in Codex)" if codex else "") + ": it imports what you did, says how")
    print("     many tasks the local model can be tried on, and offers to run them; say yes")
    print("  3. once a task class shows verified successes, /shadow can offload such a task to")
    print("     the local model and show you the diff and the test verdict; you apply it")
    if not (args.model or picked):
        print("  the offload needs a model: 'runner --shadow-mode -m MODEL.gguf' when you have one")
    return 0


def cmd_routes(args: argparse.Namespace) -> int:
    records = _read_records(Path(args.out) / "evidence.jsonl")
    records += _read_records(Path(args.bench) / "evidence.jsonl")
    routes = route_table(records)
    if args.json:
        print(json.dumps([{**r.__dict__, "qualifies": r.qualifies} for r in routes], indent=2))
    else:
        print(render_routes(routes))
    return 0


def cmd_delegate(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    cfg = read_config(home)
    out = str(Path(args.out or str(cfg.get("out") or DEFAULT_OUT)).expanduser())
    if args.background:
        model = str(args.model or cfg.get("model") or "")
        if not model:
            print("error: no model in the shadow config", file=sys.stderr)
            return 2
        repo = _repo_of(Path(args.repo).resolve())
        if repo is None:
            print(f"error: {args.repo} is not inside a git repository", file=sys.stderr)
            return 2
        records = _read_records(Path(out) / "evidence.jsonl")
        route = tandem.qualifies(records, repo.name, _model_sha(model))
        if route is None:
            print(f"not started: no task class qualifies for {repo.name} with {Path(model).name} yet "
                  "('shadow routes' shows the counts)")
            return 1
        if any(st.running for st in tandem.states(home)):
            print("not started: a delegation is already running ('shadow delegations')")
            return 1
        bg = tandem.DelegationState(id=tandem.new_id(), session_id=args.session, request=args.request,
                                    repo=str(repo), started_at=__import__("time").time(), model=model)
        bg.save(home)
        spawn_detached([args.python, "-m", "xyntetik_runner.shadow", "delegate", "--repo", str(repo),
                        "--request", args.request, "--home", str(home), "--python", args.python,
                        "--out", out, "--record", bg.id])
        print(f"started delegation {bg.id} in the background ({route.task_class}: {route.verified} verified "
              f"of {route.attempted} on record{'; strong: wait for it first' if route.strong else ''}); "
              f"'shadow delegations --wait {bg.id}' waits for it")
        return 0
    state: tandem.DelegationState | None = None
    if args.record:
        for st in tandem.states(home):
            if st.id == args.record:
                state = st
    try:
        if args.endpoint:
            endpoint = RunnerEndpoint(args.endpoint, timeout=args.request_timeout)
        else:
            endpoint, started = served_runner(home, cfg, model=args.model,
                                              start_timeout=args.start_timeout,
                                              request_timeout=args.request_timeout,
                                              managed_factory=ManagedRunner,
                                              endpoint_factory=RunnerEndpoint)
            print(f"{'started' if started else 'reusing'} the runner at {endpoint.base_url} "
                  f"(warm for {DEFAULT_TTL}s; 'shadow server --stop' ends it)", flush=True)
        caps = endpoint.capabilities()
        models = [m.get("id") for m in caps.get("models", []) if isinstance(m, dict)]
        model = str(models[0]) if models else "model"
        budget = Budget(max_turns=args.max_turns, wall_s=args.wall, test_runs=4)
        d = delegate(Path(args.repo).resolve(), args.request, endpoint.post_json, model,
                     python=args.python, budget=budget)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        if state is not None:
            state.status, state.error, state.ended_at = "error", str(e), __import__("time").time()
            state.save(home)
        return 2
    _record_delegation(Path(out), d, model_path=str(cfg.get("model") or ""), caps=caps,
                       endpoint_label=endpoint.base_url, budget_turns=args.max_turns, budget_wall=args.wall)
    if state is not None:
        state.status, state.verdict, state.patch_path = "done", d.verdict, d.patch_path
        state.changed_paths, state.tests_exit, state.task_class = d.changed_paths, d.tests_exit, d.task_class
        state.ended_at = __import__("time").time()
        state.save(home)
    if args.json:
        print(d.to_json())
        return 0
    print(f"model: {d.model}; attempt: {d.attempt.stop_reason}, {d.attempt.turns} turns, "
          f"{d.attempt.tool_calls} calls, {d.wall_s:.0f}s")
    print(f"changed: {', '.join(d.changed_paths) or '(nothing)'}")
    print(f"verdict: {d.verdict}")
    if d.tests_tail:
        print("tests (tail):")
        print(d.tests_tail[-1200:])
    print(f"patch: {d.patch_path}")
    if d.changed_paths:
        print(f"apply with: git apply {d.patch_path}   (your call; the working tree is untouched)")
    return 0 if d.tests_exit == 0 else 1


def _serve_for(cfg: dict[str, object], model_path: str, *, extra: tuple[str, ...],
               start_timeout: float, request_timeout: float) -> tuple[ManagedRunner, RunnerEndpoint]:
    launch = ServerLaunch(executable=str(cfg.get("runner") or "runner"), model=model_path,
                          port=_free_port(), context_size=int(str(cfg.get("ctx") or 8192)),
                          gpu=str(cfg.get("gpu") or "auto"),
                          threads=int(str(cfg.get("threads") or 0)) or None, extra_args=extra)
    managed = ManagedRunner(launch)
    if not managed.start(timeout=start_timeout):
        raise RuntimeError("the runner did not start; check the model path and 'runner --fit'")
    return managed, RunnerEndpoint(managed.base_url, timeout=request_timeout)


def cmd_adapt(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    if args.status:
        print(adapt.render_status(home))
        return 0
    cfg = read_config(home)
    model_path = str(args.model or cfg.get("model") or "")
    runner = str(args.runner or cfg.get("runner") or "runner")
    if not model_path:
        print("error: no model in the shadow config; run 'runner --shadow-mode -m MODEL.gguf' first",
              file=sys.stderr)
        return 2
    if not Path(model_path).is_file():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 2
    try:
        meta = adapt.gguf_meta(Path(model_path))
    except (OSError, ValueError) as e:
        print(f"error: cannot read the model header: {e}", file=sys.stderr)
        return 2
    family = adapt.template_family(str(meta.get("tokenizer.chat_template") or ""))
    if family is None:
        print("error: this model's chat template is not one adapt can render for training "
              "(ChatML or Llama 3); the prompt at training must match the prompt at serving",
              file=sys.stderr)
        return 2
    tasks: list[RepairTask] = []
    for d in [Path(args.out).expanduser()] + [Path(b).expanduser() for b in args.bank.split(",") if b]:
        if (d / "tasks").is_dir():
            tasks += _tasks(d, None)
    tasks = [t for t in tasks if t.baseline_failing <= args.max_failing and t.baseline_failing < t.expected_tests]
    if not tasks:
        print("error: no admitted tasks under --out or --bank; capture work or run 'shadow bank'",
              file=sys.stderr)
        return 2
    log = lambda s: print(s, flush=True)  # noqa: E731
    print(f"{len(tasks)} task(s); model {Path(model_path).name} ({family} template); "
          f"self-checking the function units ...", flush=True)
    ds = adapt.build_dataset(tasks, family=family, ctx=args.ctx, python=args.python, seed=args.seed,
                             holdout_fraction=args.holdout_fraction, log=log)
    n_units = len(ds.dev) + len(ds.holdout)
    print(f"units: {n_units} (development {len(ds.dev)}, held out {len(ds.holdout)}); "
          f"examples: {len(ds.examples)}; dropped: {len(ds.dropped)}")
    for dropped in ds.dropped:
        print(f"  drop {dropped['task_id']}: {dropped['why']}")
    if n_units < adapt.MIN_UNITS or not ds.examples or not ds.holdout:
        print(f"too few units to adapt: at least {adapt.MIN_UNITS} verified function units are "
              "needed, one to train on and one to hold out; the bench and captured work add units")
        return 1
    steps = args.epochs * len(ds.examples)
    print(f"plan: train {len(ds.examples)} example(s) for {steps} step(s) (rank {args.rank}, "
          f"lr {args.lr}, ctx {args.ctx}); evaluate base and adapter on {len(ds.holdout)} held-out "
          f"unit(s) with K {args.k}; keep the adapter only if held-out verified rises")
    if args.dry_run:
        print("dry run: nothing trained")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("error: not a terminal; pass --yes to confirm", file=sys.stderr)
            return 2
        if input("Start the adaptation run now? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not started")
            return 1
    stop_server(home)  # the warm runner holds the device the training needs
    stamp = _now().replace(":", "").replace("-", "")[:15]
    run_dir = adapt.adapters_dir(home) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    data = ds.write(run_dir)
    print(f"run: {run_dir}", flush=True)
    managed: ManagedRunner | None = None
    try:
        print("serving the base for its evaluation ...", flush=True)
        managed, endpoint = _serve_for(cfg | {"runner": runner}, model_path, extra=(),
                                       start_timeout=args.start_timeout,
                                       request_timeout=args.request_timeout)
        model_id = _served_model(endpoint)
        base_hold = adapt.evaluate(ds.holdout, endpoint.post_json, model_id, label="base held-out",
                                   k=args.k, python=args.python, log=log)
        base_dev = adapt.evaluate(ds.dev, endpoint.post_json, model_id, label="base development",
                                  k=args.k, python=args.python, log=log)
        managed.stop()
        managed = None
        print(f"base: held-out {base_hold.verified_samples}/{base_hold.samples} verified, "
              f"development {base_dev.verified_samples}/{base_dev.samples}", flush=True)
        adapter = run_dir / "adapter.gguf"
        print(f"training {steps} step(s); log {run_dir / 'train.log'} ...", flush=True)
        trained = adapt.train(runner, model_path, data, adapter, ctx=args.ctx, steps=steps, lr=args.lr,
                              rank=args.rank, threads=args.threads, log_path=run_dir / "train.log",
                              gpu=not args.cpu_train, log=log)
        if trained.returncode != 0 or not adapter.is_file():
            print(f"error: training failed (exit {trained.returncode}); see {trained.log}",
                  file=sys.stderr)
            return 2
        print("serving the adapter for its evaluation ...", flush=True)
        managed, endpoint = _serve_for(cfg | {"runner": runner}, model_path,
                                       extra=("--lora", str(adapter)),
                                       start_timeout=args.start_timeout,
                                       request_timeout=args.request_timeout)
        model_id = _served_model(endpoint)
        ada_hold = adapt.evaluate(ds.holdout, endpoint.post_json, model_id, label="adapter held-out",
                                  k=args.k, python=args.python, log=log)
        ada_dev = adapt.evaluate(ds.dev, endpoint.post_json, model_id, label="adapter development",
                                 k=args.k, python=args.python, log=log)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        if managed is not None:
            managed.stop()
    verdict = adapt.decide(base_hold, ada_hold, base_dev, ada_dev)
    adapt.record_run(home, stamp, dataset=ds, base_hold=base_hold, ada_hold=ada_hold, base_dev=base_dev,
                     ada_dev=ada_dev, trained=trained, verdict=verdict, model=model_path)
    if verdict.promoted:
        set_config_adapter(home, str(adapter))
        print(f"kept: {verdict.reason}; the next delegation serves {adapter}")
    else:
        print(f"not kept: {verdict.reason}; recorded under {run_dir}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """The loop the hooks feed: import what was captured, say how many tasks
    wait for the configured model, and replay some of them when told to."""
    home = Path(args.home) if args.home else Path.home()
    out = Path(args.out).expanduser()
    cfg = read_config(home)
    model_path = str(args.model or cfg.get("model") or "")
    admitted = do_import(out, home=home if args.home else None, python=args.python, timeout=args.timeout)
    evidence = out / "evidence.jsonl"
    records = _read_records(evidence)
    sha = file_sha256(Path(model_path)) if model_path and Path(model_path).is_file() else ""
    done = {r.episode_id for r in records if r.verifier is not None and r.identity.model_sha256 == sha}
    waiting = [t for t in _tasks(out, None) if t.episode_id not in done]
    name = Path(model_path).name if model_path else "(no model configured)"
    by_class: dict[str, int] = {}
    for t in waiting:
        by_class[t.task_class] = by_class.get(t.task_class, 0) + 1
    detail = ", ".join(f"{n} {c}" for c, n in sorted(by_class.items(), key=lambda kv: _CLASS_RANK.get(kv[0], 3)))
    print(f"sync: {admitted} task(s) admitted now; {len(waiting)} waiting for a replay with {name}"
          + (f" ({detail}; function first)" if detail else ""))
    if not args.replay or not waiting:
        if waiting and model_path:
            print(f"  'shadow sync --replay {min(len(waiting), 5)}' runs the next ones; each takes up to "
                  f"{int(args.wall)}s and the fit probe runs first")
        elif waiting:
            print("  no model configured: run 'runner --shadow-mode -m MODEL.gguf' first")
        print(render_routes(route_table(records)))
        return 0
    try:
        endpoint, started = served_runner(home, cfg, model=args.model, start_timeout=args.start_timeout,
                                          request_timeout=args.request_timeout,
                                          managed_factory=ManagedRunner, endpoint_factory=RunnerEndpoint)
        print(f"{'started' if started else 'reusing'} the runner at {endpoint.base_url}", flush=True)
        opts = ReplayOptions(python=args.python, timeout=args.timeout, wall=args.wall,
                             min_tps=args.min_tps, model_sha256=sha,
                             quant=_quant_from_name(Path(model_path).name), limit=args.replay,
                             endpoint_label=endpoint.base_url)
        ran, _tps, _model = run_replay(out, endpoint, opts)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    records = _read_records(evidence)
    left = len(waiting) - ran
    print(f"sync: {ran} replayed, {max(left, 0)} still waiting")
    print(render_routes(route_table(records)))
    return 0


def cmd_server(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    if args.stop:
        print("warm runner stopped" if stop_server(home) else "no warm runner was running")
        return 0
    print(server_status(home))
    return 0


def _record_delegation(out: Path, d: Any, *, model_path: str, caps: dict[str, Any],
                       endpoint_label: str, budget_turns: int = 0, budget_wall: float = 0.0) -> None:
    """A delegation is an attempt with a verdict on the user's own request:
    it joins the ledger under its own verifier id (the repository's tests
    at HEAD, not frozen tests), verified only when they passed and no test
    file was touched, so the funnel widens on evidence and never on a
    weakened test."""
    passed = d.tests_exit == 0 if d.tests_exit is not None else None
    tamper = tuple(d.test_files_changed)
    verifier: VerifierOutcome | None
    if not d.changed_paths or passed is None:
        disposition, verifier = Disposition.VERIFIER_INCONCLUSIVE, None
    elif passed and not tamper:
        disposition = Disposition.VERIFIED_LOCAL_ATTEMPT
        verifier = VerifierOutcome("repo-tests", True, 0, 0, 0, 0, 0, reasons=("repository tests passed at HEAD",))
    else:
        disposition = Disposition.LOCAL_FAILED
        verifier = VerifierOutcome("repo-tests", passed, 0, 0, 0 if passed else 1, 0, 0, tamper=tamper,
                                   reasons=(("test files changed by the attempt",) if tamper else ("repository tests failed",)))
    sha = _model_sha(model_path) if model_path else f"unknown:{d.model}"
    ident = Identity(project=Path(d.repo).name, task_class=d.task_class, context_band=_band(len(d.request)),
                     tool_set=("list_files", "read_file", "write_file", "edit_file", "run_tests"),
                     verifier_id="repo-tests", environment_id=f"{platform.node()}|{endpoint_label}",
                     model_sha256=sha, quant=_quant_from_name(Path(model_path).name) if model_path else "unknown",
                     template_sha256="unknown", runner_build=str(caps.get("version") or "unknown"),
                     backend=str(caps.get("backend") or "unknown"), harness_version=HARNESS_VERSION,
                     scaffold_sha256="base")
    rec = EpisodeEvidence(episode_id=f"delegation:{Path(d.patch_path).stem}", source="delegation",
                          observed_at=_now(), disposition=disposition, identity=ident, baseline_sha256="",
                          patch_sha256=None, changed_paths=d.changed_paths, verifier=verifier,
                          wall_s=d.wall_s, resources={"turns": float(d.attempt.turns),
                                                      "tool_calls": float(d.attempt.tool_calls),
                                                      "budget_turns": float(budget_turns),
                                                      "budget_wall_s": float(budget_wall)},
                          reasons=(f"attempt: {d.attempt.stop_reason}", d.verdict))
    out.mkdir(parents=True, exist_ok=True)
    _append(out / "evidence.jsonl", rec)


def cmd_delegations(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    if args.wait:
        st = tandem.wait_for(home, args.wait, timeout_s=args.timeout)
        if st is None:
            print(f"error: no delegation {args.wait}", file=sys.stderr)
            return 2
        if st.status == "running":
            print(f"still running after {int(args.timeout)}s: do the work yourself; 'shadow delegations' later")
            return 1
        print(f"delegation {st.id}: {st.status}; verdict: {st.verdict or st.error}; class: {st.task_class}")
        if st.verified:
            print(f"verified: tests passed on the scratch copy; patch: {st.patch_path}")
            print(f"apply with: git apply {st.patch_path}   (the user's call; the working tree is untouched)")
            return 0
        return 1
    print(tandem.render_delegations(home, session_id=args.session))
    return 0


def cmd_tandem(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    cfg = read_config(home)
    if args.turns is not None:
        set_config_key(home, "tandem_turns", int(args.turns))
    if args.wall is not None:
        set_config_key(home, "tandem_wall", float(args.wall))
    if args.state in ("on", "off"):
        set_config_key(home, "tandem", args.state == "on")
        print(f"tandem {args.state}")
        return 0
    cfg = read_config(home)
    print("tandem " + ("on" if cfg.get("tandem", True) and cfg.get("model") else "off")
          + ("" if cfg.get("model") else " (no model configured)")
          + f"; budget {cfg.get('tandem_turns', tandem.DEFAULT_TURNS)} turns, "
          f"{cfg.get('tandem_wall', tandem.DEFAULT_WALL_S):g}s (measured default; see tandem.py)")
    return 0


def _served_model(endpoint: RunnerEndpoint) -> str:
    caps = endpoint.capabilities()
    models = [m.get("id") for m in caps.get("models", []) if isinstance(m, dict)]
    return str(models[0]) if models else "model"


def cmd_uninstall(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    if stop_server(home):
        print("warm runner stopped")
    done = uninstall(home)
    print(f"removed {-done.hooks_added} hook(s), the /shadow skill, the codex prompt, the capability "
          "sheet and the notes in CLAUDE.md and AGENTS.md")
    return 0


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
                model = f"{model[:28]} +{r.identity.scaffold_sha256[:8]}"
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
    p.add_argument("--min-tps", type=float, default=15.0,
                   help="refuse a model that decodes slower than this (fit-first)")
    p.add_argument("--scaffold", default="", help="scaffold JSON; default is the base scaffold")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("bank", help="build repair tasks from a public repository's history")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT.parent / "shadow-bank"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-commits", type=int, default=400)
    p.add_argument("--max-src-files", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(fn=cmd_bank)
    p = sub.add_parser("optimize", help="evolve a scaffold against the tasks under --out")
    p.add_argument("--out", default=str(DEFAULT_OUT.parent / "shadow-bank"))
    p.add_argument("--endpoint", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="")
    p.add_argument("--scaffold", default="", help="starting scaffold; default base")
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--holdout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout-budget", type=int, default=200)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--test-runs", type=int, default=3)
    p.add_argument("--wall", type=float, default=600.0)
    p.add_argument("--min-tps", type=float, default=15.0)
    p.add_argument("--max-failing", type=int, default=30,
                   help="skip tasks with more failing-at-base tests than this")
    p.set_defaults(fn=cmd_optimize)
    p = sub.add_parser("capture", help="append a prompt or stop event from an agent hook")
    p.add_argument("--event", choices=["prompt", "stop"], default="")
    p.add_argument("--summary", action="store_true", help="print counts over the capture file")
    p.add_argument("--tool", default="claude_code")
    p.add_argument("--cwd", default="")
    p.add_argument("--session", default="")
    p.add_argument("--prompt", default="")
    p.add_argument("--file", default="")
    p.add_argument("--home", default="")
    p.add_argument("--python", default="")
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("bench", help="bank + fit probe + replay per model + the verified table, "
                                     "on your own repository")
    p.add_argument("--repo", default="", help="repository to build tasks from (if none yet)")
    p.add_argument("--out", default=str(DEFAULT_OUT.parent / "shadow-bench"))
    p.add_argument("--models", default="", help="comma-separated GGUF paths, served one at a time")
    p.add_argument("--endpoints", default="", help="comma-separated running runner URLs")
    p.add_argument("--runner", default="runner", help="runner executable for --models")
    p.add_argument("--runner-args", default="", help="extra runner flags, e.g. '--cpu-moe'")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--start-timeout", type=float, default=600.0)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-commits", type=int, default=400)
    p.add_argument("--max-src-files", type=int, default=3)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--test-runs", type=int, default=3)
    p.add_argument("--wall", type=float, default=600.0)
    p.add_argument("--min-tps", type=float, default=15.0)
    p.add_argument("--scaffold", default="")
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("install", help="wire the hooks and a /shadow command into the harnesses "
                                       "(explicit opt-in; reversible with uninstall)")
    p.add_argument("--home", default="")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--pythonpath", default="", help="set when the client is not installed")
    p.add_argument("--out", default="~/.xyntetik/shadow")
    p.add_argument("--no-claude", action="store_true")
    p.add_argument("--no-codex", action="store_true")
    p.add_argument("--claude", action="store_true", help="install for Claude Code even if ~/.claude is absent")
    p.add_argument("--codex", action="store_true", help="install for Codex even if ~/.codex is absent")
    p.add_argument("--model", default="", help="GGUF the /shadow offload serves (runner --shadow-mode -m)")
    p.add_argument("--runner", default="runner", help="runner executable for offloading")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-pick", action="store_true", help="without -m, do not look for a model on disk")
    p.add_argument("--no-tandem", action="store_true", help="no background attempts on prompts")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="print what would be written")
    p.set_defaults(fn=cmd_install)
    p = sub.add_parser("routes", help="per task class, where the local model has verified successes")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--bench", default=str(DEFAULT_OUT.parent / "shadow-bench"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_routes)
    p = sub.add_parser("delegate", help="one bounded attempt at a request on a scratch worktree")
    p.add_argument("--repo", default=".")
    p.add_argument("--request", required=True)
    p.add_argument("--endpoint", default="", help="running runner; default: start from config")
    p.add_argument("--model", default="")
    p.add_argument("--home", default="")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--wall", type=float, default=900.0)
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--start-timeout", type=float, default=600.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default="", help="ledger directory (default: the shadow config's)")
    p.add_argument("--background", action="store_true",
                   help="start detached where the evidence allows it and return at once")
    p.add_argument("--session", default="", help="with --background: the harness session id")
    p.add_argument("--record", default="", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_delegate)
    p = sub.add_parser("delegations", help="background delegations and their verdicts")
    p.add_argument("--home", default="")
    p.add_argument("--session", default="")
    p.add_argument("--wait", default="", help="block until this delegation ends, then print its verdict")
    p.add_argument("--timeout", type=float, default=tandem.RUNNING_WALL_S)
    p.set_defaults(fn=cmd_delegations)
    p = sub.add_parser("tandem", help="show or set whether prompts get a background local attempt")
    p.add_argument("state", nargs="?", choices=("on", "off"), default="")
    p.add_argument("--home", default="")
    p.add_argument("--turns", type=int, default=None, help="background attempt budget in turns")
    p.add_argument("--wall", type=float, default=None, help="background attempt budget in seconds")
    p.set_defaults(fn=cmd_tandem)
    p = sub.add_parser("adapt", help="overnight: train the configured model's adapter on the ledger's "
                                     "own commits, keep it only on a held-out verified rise")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--bank", default=str(DEFAULT_OUT.parent / "shadow-bank"),
                   help="comma-separated task directories to add (the bench bank)")
    p.add_argument("--home", default="")
    p.add_argument("--model", default="", help="default: the shadow config's model")
    p.add_argument("--runner", default="", help="default: the shadow config's runner")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--ctx", type=int, default=4096, help="training window")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--k", type=int, default=4, help="samples per unit in each evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-fraction", type=float, default=0.3)
    p.add_argument("--max-failing", type=int, default=30)
    p.add_argument("--cpu-train", action="store_true", help="do not ask for the GPU training path")
    p.add_argument("--start-timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--dry-run", action="store_true", help="print the plan; train nothing")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--status", action="store_true", help="list the recorded runs and their verdicts")
    p.set_defaults(fn=cmd_adapt)
    p = sub.add_parser("sync", help="import what the hooks captured, count the tasks waiting for the "
                                    "configured model, replay some when asked")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--home", default="")
    p.add_argument("--model", default="", help="default: the shadow config's model")
    p.add_argument("--replay", type=int, default=0, help="replay up to N waiting tasks now")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--wall", type=float, default=900.0)
    p.add_argument("--min-tps", type=float, default=15.0)
    p.add_argument("--start-timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.set_defaults(fn=cmd_sync)
    p = sub.add_parser("server", help="the warm runner shadow mode keeps for delegations")
    p.add_argument("--home", default="")
    p.add_argument("--stop", action="store_true")
    p.set_defaults(fn=cmd_server)
    p = sub.add_parser("uninstall", help="remove exactly what install wrote")
    p.add_argument("--home", default="")
    p.set_defaults(fn=cmd_uninstall)
    p = sub.add_parser("report", help="counts and both denominators")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--by-reason", action="store_true")
    p.add_argument("--tasks", action="store_true", help="one block per admitted task")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    fn: Any = args.fn
    return int(fn(args))
