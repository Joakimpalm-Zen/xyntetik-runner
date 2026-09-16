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
from xyntetik_runner.shadow import recurring
from xyntetik_runner.shadow.importer import CAPTURE_FILE, Episode, read_episodes, scan_all, write_episodes
from xyntetik_runner.shadow import receipt, tandem
from xyntetik_runner.shadow.install import (harness_present, install, read_config, render_candidates,
                                            set_config_key, suggest_model, uninstall)
from xyntetik_runner.process import spawn_detached
from xyntetik_runner.shadow.routes import delegate, render_routes, route_table
from xyntetik_runner.shadow.server import (DEFAULT_TTL, render_status as server_status, served_runner,
                                           stop_server)
from xyntetik_runner.shadow.bank import build_bank

from xyntetik_runner.shadow.scaffold import Scaffold
from xyntetik_runner.shadow.tasks import (
    Candidate,
    RepairTask,
    Rejection,
    build_task,
    choose,
    canonical,
    repository_root,
    pair,
    repos_under,
)
from xyntetik_runner.shadow.verifier import ProtectedTests, fixed_ids, verify



HARNESS_VERSION = "shadow-0.1"
def default_out() -> Path:
    """Where a user's shadow state lives when they name no other place.

    A function and not a constant: a module-level ``Path.home()`` is bound
    at import, which is before a test session can redirect HOME, so the
    suite wrote its fixtures into the developer's own ~/.xyntetik for
    three days without anyone noticing."""
    return Path.home() / ".xyntetik" / "shadow"


# The capture events, in one place. argparse and the runtime check both read
# this: as two separate literals they drifted, and --event post parsed and then
# failed at the runtime check.
CAPTURE_EVENTS: tuple[str, ...] = ("prompt", "stop", "verify", "post")


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
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(EpisodeEvidence.from_json(line))
        except (ValueError, KeyError, TypeError):
            skipped += 1  # a line cut short by a kill mid-append; the ledger is still readable
    if skipped:
        print(f"warning: {skipped} unreadable line(s) in {path} skipped", file=sys.stderr)
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
    latest: dict[str, EpisodeEvidence] = {}
    for r in _read_records(evidence):
        latest[r.episode_id] = r
    # An episode is settled by its newest record, except one a rule that has
    # since widened rejected: that one is paired again and its new record
    # supersedes (the summary reads the newest record of an unattempted
    # episode). Today's widened rule: built ranges, once "need a build".
    known = {eid for eid, r in latest.items() if not _revisit(r)}
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
            print(f"  admitted {result.task_id}: {result.expected_tests} frozen "
                  f"{'gates' if result.verifier_kind == 'make' else 'tests'}, "
                  f"{result.baseline_failing} fail at base, {result.src_files} source file(s)",
                  flush=True)
            continue
        record(c.episode, result)
    print("dispositions at import:", json.dumps(counts, sort_keys=True), flush=True)
    print(f"{admitted} task(s) admitted under {tasks_dir}", flush=True)
    return admitted


REVISIT_REASONS = ("file(s) that need a build",)
"""Rejection reasons of admission rules that have since widened; an episode
whose newest record carries one is imported again."""


def _revisit(r: EpisodeEvidence) -> bool:
    return (r.verifier is None and r.disposition in (Disposition.INELIGIBLE, Disposition.UNREPLAYABLE)
            and any(p in reason for reason in r.reasons for p in REVISIT_REASONS))


def _tasks(out: Path, only: str | None) -> list[RepairTask]:
    tasks = [RepairTask.load(p) for p in sorted((out / "tasks").glob("*/task.json"))]
    return [t for t in tasks if not only or t.task_id == only]


def attempt_key(episode_id: str, ident: Identity) -> tuple[str, str, str, tuple[str, ...]]:
    """What makes an attempt a repeat: the same episode under the same model,
    scaffold AND tool set. The tool set is part of the identity (edit-only,
    visible solution tests), so a cohort that differs there is a new
    attempt, not a rerun; keyed on model and scaffold alone, the first
    show-tests cohort was skipped as already attempted (2026-09-16)."""
    return (episode_id, ident.model_sha256, ident.scaffold_sha256, tuple(ident.tool_set))


def stage_solution_tests(task: RepairTask, ws_dir: Path) -> tuple[str, ...]:
    """Copy the task's frozen test files (never the Makefile) into the
    workspace and return their paths: the failing test as the attempt's
    visible specification. Done before the baseline is captured, so an
    edit to one of them is a change the verifier reports as tamper."""
    src = Path(task.protected_dir)
    staged: list[str] = []
    for rel in task.test_files:
        if not (src / rel).is_file():
            continue
        target = ws_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, target)
        staged.append(rel)
    return tuple(staged)


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
    edit_only: bool = False
    """write_file creates only; existing files change through edit_file.
    Part of the identity's tool set, so cohorts stay apart."""
    show_tests: bool = False
    """Stage the frozen (post-state) test files into the workspace before the
    attempt, so the attempt sees the failing test the fix was written for
    instead of the pre-state test that passes. The protected copy still
    judges, and an edit to a staged file is tamper. Part of the identity's
    tool set (`visible:solution-tests`), so cohorts stay apart."""
    samples: int = 1
    """Attempts per task at ``temperature``, each with its own seed and its
    own ledger record; the first verified sample ends the task. Coverage
    (tasks with a verified sample) is the cohort's metric, and the summary
    already reads it: an episode with any verified record counts verified.
    Part of the identity's tool set (`samples:N@T`), so the cohort stays
    apart from the single greedy attempt."""
    temperature: float = 0.0


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
    # what each task's cohort already holds: how many samples, and whether
    # one verified (a sampling cohort resumes where it stopped)
    seen: dict[tuple[str, str, str, tuple[str, ...]], list[bool]] = {}
    for r in _read_records(evidence):
        if r.verifier is not None:
            seen.setdefault(attempt_key(r.episode_id, r.identity), []).append(
                r.disposition is Disposition.VERIFIED_LOCAL_ATTEMPT)
    done = {k for k, v in seen.items() if any(v) or len(v) >= max(1, opts.samples)}
    ran = 0
    for task in replay_order(_tasks(out, opts.task)):
        if opts.limit and ran >= opts.limit:
            break
        ident = replace(base_identity, project=Path(task.repo).name, task_class=task.task_class,
                        context_band=_band(len(task.request)),
                        tool_set=("list_files", "read_file",
                                  "write_file:new-only" if opts.edit_only else "write_file",
                                  "edit_file", "run_tests",
                                  *(("visible:solution-tests",) if opts.show_tests else ()),
                                  *((f"samples:{opts.samples}@{opts.temperature:g}",)
                                    if opts.samples > 1 or opts.temperature > 0 else ())),
                        verifier_id=f"commit-{'gates' if task.verifier_kind == 'make' else 'tests'}:{task.task_id}")
        if attempt_key(task.episode_id, ident) in done:
            continue
        ran += 1
        first = len(seen.get(attempt_key(task.episode_id, ident), []))
        for sample in range(first, max(1, opts.samples)):
          seed = 1000 + sample if opts.samples > 1 or opts.temperature > 0 else 0
          print(f"[{task.task_id}] attempt with {model}"
                + (f" (sample {sample + 1} of {opts.samples}, seed {seed})" if opts.samples > 1 else "")
                + " ...", flush=True)
          ws_dir = _worktree(task)
          ws = None
          try:
              visible = task.visible_test_files
              if opts.show_tests:
                  visible = stage_solution_tests(task, ws_dir)
              baseline = Baseline.capture(ws_dir)
              ws = Workspace(ws_dir, visible_tests=visible,
                             pythonpath=task.pythonpath, python=opts.python, budget=budget,
                             gates=task.gates, edit_only=opts.edit_only)
              chat = runner_chat(endpoint.post_json, model, max_tokens=budget.max_tokens,
                                 temperature=opts.temperature, seed=seed)
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
                             "fixed": float(len(fixed)), "probe_tps": round(tps, 2),
                             "sample": float(sample + 1), "seed": float(seed)},
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
              if ws is not None:
                  ws.close()
              _drop(task, ws_dir)
          if disposition is Disposition.VERIFIED_LOCAL_ATTEMPT:
              break
    print(f"{ran} attempt(s) recorded in {evidence}", flush=True)
    return ran, tps, model


def cmd_replay(args: argparse.Namespace) -> int:
    opts = ReplayOptions(python=args.python, timeout=args.timeout, max_turns=args.max_turns,
                         max_tokens=args.max_tokens, test_runs=args.test_runs, wall=args.wall,
                         min_tps=args.min_tps, scaffold_path=args.scaffold,
                         model_sha256=args.model_sha256, quant=args.quant, task=args.task,
                         limit=args.limit, endpoint_label=args.endpoint,
                         edit_only=args.edit_only, show_tests=args.show_tests,
                         samples=args.samples, temperature=args.temperature)
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
                         min_tps=args.min_tps, scaffold_path=args.scaffold, edit_only=args.edit_only, show_tests=args.show_tests,
                         samples=args.samples, temperature=args.temperature)
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
                         max_src_files=args.max_src_files, timeout_s=args.timeout,
                         kind=args.kind, max_gates=args.max_gates, since=args.since)
    admitted = [e for e in entries if e.task is not None]
    for e in admitted:
        assert e.task is not None
        what = "gates" if e.task.verifier_kind == "make" else "tests"
        print(f"  admitted {e.task.task_id}: {e.task.expected_tests} frozen {what}, "
              f"{e.task.baseline_failing} fail at base, {e.task.task_class}", flush=True)
    reasons: dict[str, int] = {}
    for e in entries:
        if e.task is None:
            key = e.reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    print(f"{len(admitted)} task(s) admitted from {len(entries)} commits; rejected:",
          json.dumps(reasons, sort_keys=True), flush=True)
    return 0




def cmd_capture(args: argparse.Namespace) -> int:
    """Capture local raw events, bounded state and provenance; summarize on request.

    Verification events record a post-command snapshot, not an atomic execution
    snapshot. Reports keep that association separate from reproduced outcomes.
    """
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
    if getattr(args, "report", False):
        from . import capture_report
        home = Path(args.home) if args.home else Path.home()
        report_path = Path(args.file) if args.file else None
        rep = capture_report.report(home, report_path)
        print(json.dumps(rep, indent=1, default=str) if args.json
              else capture_report.render(rep))
        return 0
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    # Kept in step with the argparse choices above by CAPTURE_EVENTS; they were
    # two independent literals and a pipe test found them disagreeing.
    if args.event not in CAPTURE_EVENTS:
        print("error: --event %s is required (or --summary)" % "|".join(CAPTURE_EVENTS),
              file=sys.stderr)
        return 2
    cwd = str(args.cwd or data.get("cwd") or Path.cwd())
    home_p = Path(args.home) if args.home else Path.home()
    session = str(args.session or data.get("session_id") or "")

    # ---- v2 (capture.py): raw payload preserved, provenance labelled and linked,
    # the actual dirty workspace snapshotted, verification bound to that state.
    # Wrapped whole: a hook must never fail a prompt, and a v2 defect must never
    # cost the v1 record that is still this project's only continuous series.
    def _v2() -> None:
        from . import capture as cap
        verification = None
        if args.event == "verify":
            command, tr = cap.verification_io(data)
            if not cap.looks_like_verification(command):
                return
            snap = cap.snapshot_all(Path(cwd), home_p)
            rc = tr.get("exit_code")
            if rc is None:
                rc = tr.get("returncode")
            if rc is None and isinstance(tr.get("interrupted"), bool):
                rc = None
            tail = str(tr.get("stdout") or tr.get("output") or tr.get("stderr") or "")
            verification = cap.verification_record(
                command=command,
                exit_code=int(rc) if isinstance(rc, (int, float)) else None,
                output_tail=tail, manifest_sha256=snap.get("combined_sha256", ""),
                cwd=cwd)
            rec2 = cap.build("verify", payload=data, cwd=Path(cwd), home=home_p,
                             tool=args.tool, session_id=session, snap=snap,
                             verification=verification)
        else:
            snap = cap.snapshot_all(Path(cwd), home_p)
            rec2 = cap.build(args.event, payload=data, cwd=Path(cwd), home=home_p,
                             tool=args.tool, session_id=session, snap=snap)
        cap.append(home_p, rec2)

    # `post` is v2-only for the same reason `verify` is: the v1 series has run
    # continuously since 2026-09-08 and a new event type in it would make the
    # counts before and after this change incomparable.
    if args.v2_only or args.event in ("verify", "post"):
        try:
            _v2()
        except Exception:  # noqa: BLE001 - a hook never fails the prompt
            pass
        return 0
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
    # v1 is written first and unchanged, so the continuous series since
    # 2026-09-08 stays comparable; v2 is additive.
    try:
        _v2()
    except Exception:  # noqa: BLE001 - a hook never fails the prompt
        pass
    # tandem: a background attempt where the evidence allows it, and the
    # harness told; every failure here is swallowed, a hook never blocks
    try:
        home = Path(args.home) if args.home else Path.home()
        cfg = read_config(home)
        model = str(cfg.get("model") or "")
        if cfg.get("tandem", True) and model and Path(model).is_file():
            out = str(Path(str(cfg.get("out") or default_out())).expanduser())
            if args.event == "prompt":
                repo_or_none = _repo_of(Path(cwd))
                records = _read_records(Path(out) / "evidence.jsonl")
                tandem.emit(tandem.hook_prompt(home, session_id=rec["session_id"], cwd=cwd,
                                               prompt=rec["prompt"], repo=repo_or_none, records=records,
                                               model=model, model_sha256=_model_sha(model, compute=False, home=home),
                                               adapter_sha256=_adapter_sha(cfg, home, compute=False),
                                               python=args.python or sys.executable, out=out,
                                               turns=int(str(cfg.get("tandem_turns") or tandem.DEFAULT_TURNS)),
                                               wall_s=float(str(cfg.get("tandem_wall") or tandem.DEFAULT_WALL_S))))
            else:
                tandem.emit(tandem.hook_stop(home, session_id=rec["session_id"]))
    except Exception:  # noqa: BLE001 - a hook must never fail the prompt
        pass
    return 0


def _repo_of(cwd: Path) -> Path | None:
    return repository_root(cwd)


_MODEL_SHA_CACHE: dict[str, tuple[float, str]] = {}


def _adapter_sha(cfg: dict[str, object], home: Path, *, compute: bool = True) -> str | None:
    adapter = str(cfg.get("adapter") or "")
    if not adapter:
        return None
    return (_model_sha(adapter, home=home, compute=compute, cache_name="adapter-sha.json")
            or "unknown:configured-adapter")


def _model_sha(model: str, *, compute: bool = True, home: Path | None = None,
               cache_name: str = "model-sha.json") -> str:
    """sha256 of the model file, cached on mtime.

    ``compute=False`` is the hook path and it never reads the model. Hashing
    a GGUF is seconds of CPU per gigabyte, a hook runs under a timeout of
    ten seconds, and the two together are a trap: the hash does not finish,
    the process is killed before it can write the cache, so the next prompt
    hashes from scratch and is killed too. The user pays it on every prompt
    and sees a timeout warning for a hook that was supposed to be free.
    Measured here: 3.6 GB, 9.4 s of CPU, every prompt, never cached.

    So the foreground commands warm this cache (`shadow install` at the end
    of installation, `shadow sync` before it counts) and the hook takes the
    warm value or nothing. Nothing is the honest answer: an empty sha does
    not match any record's model identity, so tandem simply does not fire
    this turn, which is what should happen when we cannot say which model
    the evidence was about."""
    try:
        mtime = Path(model).stat().st_mtime
    except OSError:
        return ""
    cache = (home or Path.home()) / ".xyntetik" / "shadow" / cache_name
    try:
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("model") == model and d.get("mtime") == mtime:
            return str(d.get("sha256") or "")
    except (OSError, ValueError):
        pass
    if not compute:
        return ""
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
        plan.append(f"Claude Code: prompt, stop and edit capture hooks merged into {home / '.claude' / 'settings.json'} "
                    "(backup beside it), a /shadow skill, and a marked note in "
                    f"{home / '.claude' / 'CLAUDE.md'} saying Runner is here and what it can do")
    if codex:
        plan.append(f"Codex: prompt, stop, edit and verification capture hooks merged into "
                    f"{home / '.codex' / 'hooks.json'} (backup beside it), a /shadow prompt under "
                    f"{home / '.codex' / 'prompts'}, and a marked note in "
                    f"{home / '.codex' / 'AGENTS.md'} saying Runner is here and what it can do; "
                    "Codex reviews new or changed hooks before trusting them")
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
    print("  Capture retains raw hook payloads and bounded workspace contents locally, "
          "including agent text or tool data present in events. No data is uploaded. "
          "The hooks start after installation; 'shadow uninstall' removes the integration.")
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
        print(f"codex: {done.codex_hooks_added} hook(s) added to {done.codex_settings} "
              f"(backup beside it); prompt {done.codex_prompt}")
    if done.sheet:
        print(f"capabilities: {done.sheet}; noted in {', '.join(str(n) for n in done.notes)}")
    # Warm the model hash here, where there is no timeout. The prompt hook
    # reads this cache and never computes it.
    if done.config and (args.model or picked):
        _model_sha(str(args.model or picked), home=home)
    print("nothing runs until a prompt is submitted; the hooks never block one; "
          "'shadow uninstall' removes exactly this")
    print()
    print("what happens next:")
    print("  1. keep working as you do; each prompt and each finished turn is noted with the")
    print("     raw event, task provenance and bounded local workspace snapshot")
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
    out = str(Path(args.out or str(cfg.get("out") or default_out())).expanduser())
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
        route = tandem.qualifies(records, repo.name, _model_sha(model, home=home),
                                 adapter_sha256=_adapter_sha(cfg, home))
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
        models = [str(m.get("id")) for m in caps.get("models", []) if isinstance(m, dict)]
        wanted = Path(args.model).name if args.model else ""
        model = wanted if wanted in models else (models[0] if models else "model")
        budget = Budget(max_turns=args.max_turns, wall_s=args.wall, test_runs=4)
        d = delegate(Path(args.repo).resolve(), args.request, endpoint.post_json, model,
                     python=args.python, out_dir=Path(out) / "delegations", budget=budget)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        if state is not None:
            state.status, state.error, state.ended_at = "error", str(e), __import__("time").time()
            state.save(home)
        return 2
    rec_path = _record_delegation(Path(out), d, model_path=str(cfg.get("model") or ""), caps=caps,
                                  adapter_sha256=(_adapter_sha(cfg, home) if not args.endpoint else "unknown:external"),
                                  endpoint_label=endpoint.base_url, budget_turns=args.max_turns,
                                  budget_wall=args.wall)
    signed: receipt.Signed | None = None
    key = home / receipt.SIGNKEY_REL
    runner_exe = str(cfg.get("runner") or "runner")
    if key.is_file():
        try:
            body = receipt.statement(
                repo=d.repo, head=d.head, request=d.request, patch_path=d.patch_path,
                patch_sha256=receipt.sha256_file(Path(d.patch_path)) if Path(d.patch_path).is_file() else "",
                changed_paths=d.changed_paths, verdict=d.verdict, tests_exit=d.tests_exit,
                task_class=d.task_class, model=str(cfg.get("model") or d.model),
                model_sha256=_model_sha(str(cfg.get("model") or "")) if cfg.get("model") else "",
                adapter_sha256=receipt.sha256_file(Path(str(cfg.get("adapter")))) if cfg.get("adapter") and Path(str(cfg.get("adapter"))).is_file() else "",
                runner_build=str(caps.get("version") or "unknown"), backend=str(caps.get("backend") or "unknown"),
                budget_turns=args.max_turns, budget_wall_s=args.wall, turns=d.attempt.turns,
                tool_calls=d.attempt.tool_calls, wall_s=d.wall_s, stop_reason=d.attempt.stop_reason,
                observed_at=_now())
            signed = receipt.write_receipt(home, body, runner=runner_exe, key=key)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"receipt: not written: {e}", file=sys.stderr)
    if state is not None:
        state.status, state.verdict, state.patch_path = "done", d.verdict, d.patch_path
        state.changed_paths, state.tests_exit, state.task_class = d.changed_paths, d.tests_exit, d.task_class
        state.ended_at = __import__("time").time()
        state.save(home)
    if args.json:
        print(d.to_json())
        return 0
    if signed is not None:
        print(f"receipt: {signed.path} (chain {signed.chain_hash[:12]}, signed by {signed.public_key[:12]}...)")
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
    # also warms the cache the prompt hook reads: the hook must never hash
    sha = _model_sha(model_path, home=home) if model_path and Path(model_path).is_file() else ""
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
                       endpoint_label: str, adapter_sha256: str | None = None, budget_turns: int = 0, budget_wall: float = 0.0) -> Path:
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
                     scaffold_sha256="base", adapter_sha256=adapter_sha256)
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
    return out / "evidence.jsonl"


def cmd_keygen(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    cfg = read_config(home)
    runner_exe = str(args.runner or cfg.get("runner") or "runner")
    try:
        key = receipt.keygen(home, runner_exe)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    pub = ""
    try:
        pub = str(json.loads(key.read_text(encoding="utf-8")).get("public_key") or "")
    except (OSError, ValueError):
        pass
    print(f"signing key: {key}\npublic key: {pub}\nevery delegation now writes a signed receipt under "
          f"{receipt.receipts_dir(home)}; 'shadow receipts --check' verifies them")
    return 0


def cmd_recurring(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    roots = ([Path(r).expanduser() for r in args.sessions.split(",") if r]
             if args.sessions else recurring.default_roots(home))
    missing = [r for r in roots if not r.is_dir()]
    obs = recurring.observe(roots, min_chars=args.min_chars)
    if args.project:
        obs = [o for o in obs if args.project in o.project or args.project in o.session]
    print(recurring.render(obs, sessions=len({o.session for o in obs})))
    if missing:
        print("\nnot found, so nothing was read from: " + ", ".join(str(m) for m in missing),
              file=sys.stderr)
    return 0


def cmd_receipts(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else Path.home()
    cfg = read_config(home)
    runner_exe = str(args.runner or cfg.get("runner") or "runner")
    print(receipt.render_receipts(home, runner=runner_exe, check=args.check))
    return 0


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
    print(f"removed {-done.hooks_added} Claude Code hook(s); "
          f"removed {-done.codex_hooks_added} Codex hook(s), the /shadow skill, the codex prompt, the capability "
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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("adapt", "optimize"):
        print(f"shadow {argv[0]} moved to the optional Shade learning tools. "
              f"Use python -m xyntetik_shade.shadow {argv[0]}. "
              "No training was started and the serving configuration is unchanged.", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="python -m xyntetik_runner.shadow", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="scan traces, admit verifiable tasks, record the rest")
    p.add_argument("--out", default=str(default_out()))
    p.add_argument("--home", default="")
    p.add_argument("--source", choices=["codex", "claude_code"], default="")
    p.add_argument("--python", default=sys.executable, help="interpreter that runs the tests")
    p.add_argument("--timeout", type=float, default=600.0)
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("replay", help="attempt each admitted task and verify it")
    p.add_argument("--out", default=str(default_out()))
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
    p.add_argument("--edit-only", action="store_true",
                   help="write_file creates files only; existing files change through edit_file")
    p.add_argument("--show-tests", action="store_true",
                   help="stage the frozen post-state test files into the workspace, so the "
                        "attempt sees the failing test; edits to them are tamper")
    p.add_argument("--samples", type=int, default=1,
                   help="attempts per task at --temperature, each with its own seed and record; "
                        "the first verified sample ends the task (coverage is the metric)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("bank", help="build repair tasks from a public repository's history")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", default=str(default_out().parent / "shadow-bank"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-commits", type=int, default=400)
    p.add_argument("--max-src-files", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--kind", choices=("any", "pytest", "make"), default="any",
                   help="keep only pytest ranges or only make ranges (C tests with built source)")
    p.add_argument("--max-gates", type=int, default=0,
                   help="make ranges: at most this many C test files per commit (0 = no cap)")
    p.add_argument("--since", default="", help="git's --since, e.g. 2026-07-15")
    p.set_defaults(fn=cmd_bank)
    p = sub.add_parser("capture", help="append a prompt or stop event from an agent hook")
    p.add_argument("--event", choices=list(CAPTURE_EVENTS), default="")
    p.add_argument("--summary", action="store_true", help="print counts over the capture file")
    p.add_argument("--v2-only", action="store_true",
                   help="write only the v2 record (the verify event has no v1 form)")
    p.add_argument("--report", action="store_true",
                   help="capture completeness, workspace reconstruction and verification binding")
    p.add_argument("--json", action="store_true", help="--report as JSON")
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
    p.add_argument("--out", default=str(default_out().parent / "shadow-bench"))
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
    p.add_argument("--edit-only", action="store_true")
    p.add_argument("--show-tests", action="store_true")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
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
    p.add_argument("--out", default=str(default_out()))
    p.add_argument("--bench", default=str(default_out().parent / "shadow-bench"))
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
    p = sub.add_parser("keygen", help="make the signing key; every delegation then writes a signed receipt")
    p.add_argument("--home", default="")
    p.add_argument("--runner", default="")
    p.set_defaults(fn=cmd_keygen)
    p = sub.add_parser("recurring", help="what your work has been, grouped by what it produced")
    p.add_argument("--home", default="")
    p.add_argument("--sessions", default="", help="comma-separated session directories; "
                                                  "default: the harnesses' own")
    p.add_argument("--min-chars", type=int, default=60,
                   help="ignore requests shorter than this: continuations like 'go ahead' "
                        "inherit their kind from the conversation and describe no work")
    p.add_argument("--project", default="", help="only sessions or requests naming this string")
    p.set_defaults(fn=cmd_recurring)
    p = sub.add_parser("receipts", help="the signed delegation receipts, newest last")
    p.add_argument("--home", default="")
    p.add_argument("--runner", default="")
    p.add_argument("--check", action="store_true", help="verify each signature through the runner")
    p.set_defaults(fn=cmd_receipts)
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
    p = sub.add_parser("sync", help="import what the hooks captured, count the tasks waiting for the "
                                    "configured model, replay some when asked")
    p.add_argument("--out", default=str(default_out()))
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
    p.add_argument("--out", default=str(default_out()))
    p.add_argument("--by-reason", action="store_true")
    p.add_argument("--tasks", action="store_true", help="one block per admitted task")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    fn: Any = args.fn
    return int(fn(args))
