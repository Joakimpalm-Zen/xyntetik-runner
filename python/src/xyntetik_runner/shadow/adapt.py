"""Overnight adaptation from the ledger: the repository's own commits as
verified-SFT data, trained through the runner's adapter path, promoted only
when the held-out verified count rises.

The ledger never holds frontier content, so the only completions it can
offer are the human's commits. The unit is the function: for a task whose
fix changed exactly one function or method, the prompt is the request, the
visible tests at the pre-state and that function's source at the pre-state,
rendered through the model's own chat template; the completion is the same
function at the solution commit. A whole source file does not fit a
training window (a 40k-token module is ordinary), a function does.

Every unit is self-checked before it is used: the human's function, spliced
over the pre-state span, must be verified by the protected tests, or the
unit is dropped. Evaluation splices the model's function the same way and
the verifier judges the whole tree, so "verified" here means the same as
everywhere else in shadow mode: the frozen tests pass and nothing else
broke.

The gate is the held-out slice, seeded and never trained on. The adapter
is kept when its held-out verified count is higher than the base's on the
same tasks and samples; a rise on the development slice alone is
memorization and is reported as such. Nothing here decides for the user:
`adapt` prints its plan and runs only when told to.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .baseline import Baseline
from .tasks import RepairTask, classify, touched_files
from .verifier import ProtectedTests, fixed_ids, verify

SYSTEM = ("You are a software engineer. You receive a change request, the tests, and the "
          "current source of one function. Output the complete new source of that function "
          "and nothing else: no explanation, no code fence, no other code.")
CHARS_PER_TOKEN = 4.2  # measured 4.6 to 4.7 on code with the Qwen2.5 tokenizer; kept conservative
MIN_UNITS = 2  # one to train on, one to hold out; anything less has no gate
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


# ---------------------------------------------------------------- GGUF header

_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
                 11: "q", 12: "d"}


def gguf_meta(path: Path, *, keys: Sequence[str] = ("general.architecture",
                                                     "tokenizer.chat_template")) -> dict[str, Any]:
    """Scalar metadata from a GGUF header, without reading the tensors. Arrays
    are stepped over, so the vocabulary costs a walk and no memory."""
    want = set(keys)
    out: dict[str, Any] = {}
    with open(path, "rb") as f:
        def take(n: int) -> bytes:
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated GGUF header")
            return b

        def string() -> str:
            n = struct.unpack("<Q", take(8))[0]
            return take(n).decode("utf-8", "replace")

        def value(t: int) -> Any:
            if t == 8:
                return string()
            if t == 9:
                et = struct.unpack("<I", take(4))[0]
                n = struct.unpack("<Q", take(8))[0]
                for _ in range(n):
                    value(et)
                return None
            fmt = _GGUF_SCALARS.get(t)
            if fmt is None:
                raise ValueError(f"unknown GGUF value type {t}")
            return struct.unpack("<" + fmt, take(struct.calcsize(fmt)))[0]

        if take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", take(4))[0]
        if version < 2:
            raise ValueError(f"GGUF version {version} is not supported")
        _n_tensors, n_kv = struct.unpack("<QQ", take(16))
        for _ in range(n_kv):
            key = string()
            t = struct.unpack("<I", take(4))[0]
            v = value(t)
            if key in want:
                out[key] = v
            if want <= set(out):
                break
    return out


def template_family(chat_template: str) -> str | None:
    """The chat-template family this module can render byte for byte as the
    runner renders it, or None. Training prompts must match serving."""
    if "<|im_start|>" in chat_template:
        return "chatml"
    if "<|start_header_id|>" in chat_template:
        return "llama3"
    return None


def render_prompt(family: str, system: str, user: str) -> str:
    if family == "chatml":
        return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    if family == "llama3":
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    raise ValueError(f"no renderer for template family {family!r}")


# ---------------------------------------------------------------- the unit

@dataclass(frozen=True)
class FunctionUnit:
    task_id: str
    rel: str
    name: str
    base_span: tuple[int, int]
    base_fn: str
    sol_fn: str


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def function_spans(text: str) -> dict[str, tuple[int, int]]:
    """Qualified name to (first line including decorators, last line)."""
    out: dict[str, tuple[int, int]] = {}
    tree = ast.parse(text)

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if not isinstance(child, ast.ClassDef) and child.end_lineno:
                    first = min([d.lineno for d in child.decorator_list] + [child.lineno])
                    out[name] = (first, child.end_lineno)
                walk(child, name + ".")
            else:
                walk(child, prefix)
    walk(tree, "")
    return out


def function_unit(task: RepairTask) -> FunctionUnit | str:
    """The one changed function of a single-file fix, or why there is none."""
    if task.src_files != 1:
        return f"{task.src_files} source files"
    files = touched_files(Path(task.repo), [task.solution_sha])
    _, src, _ = classify(files)
    if len(src) != 1:
        return f"{len(src)} source files in the range"
    rel = src[0]
    base_text = _git(task.repo, "show", f"{task.base_sha}:{rel}")
    sol_text = _git(task.repo, "show", f"{task.solution_sha}:{rel}")
    if not base_text or not sol_text:
        return "file absent in a state"
    try:
        base_spans, sol_spans = function_spans(base_text), function_spans(sol_text)
    except SyntaxError:
        return "syntax error in a state"
    ranges: list[tuple[int, int]] = []
    for line in _git(task.repo, "diff", "-U0", task.base_sha, task.solution_sha, "--", rel).split("\n"):
        m = _HUNK.match(line)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            ranges.append((start, start + max(count, 1) - 1))
    if not ranges:
        return "no hunks"
    sol_lines = sol_text.split("\n")
    owners: set[str] = set()
    for a, b in ranges:
        inside = [(n, s) for n, s in sol_spans.items() if s[0] <= a and b <= s[1]]
        if not inside:
            probe = sol_lines[a - 1] if 0 < a <= len(sol_lines) else ""
            if probe.startswith(("import ", "from ")):
                continue  # an import the fix needed; the self-check decides whether it mattered
            return f"lines {a}-{b} outside any function"
        owners.add(min(inside, key=lambda x: x[1][1] - x[1][0])[0])
    if len(owners) != 1:
        return f"{len(owners)} functions changed"
    name = owners.pop()
    if name not in base_spans:
        return f"{name} is new at the solution"
    b0, b1 = base_spans[name]
    s0, s1 = sol_spans[name]
    base_fn = "\n".join(base_text.split("\n")[b0 - 1:b1])
    sol_fn = "\n".join(sol_lines[s0 - 1:s1])
    if base_fn == sol_fn:
        return "function text identical (change was elsewhere)"
    return FunctionUnit(task.task_id, rel, name, (b0, b1), base_fn, sol_fn)


def unit_prompt(task: RepairTask, u: FunctionUnit, *, test_chars: int = 6000) -> str:
    visible = "".join(_git(task.repo, "show", f"{task.base_sha}:{vt}")[:test_chars]
                      for vt in task.visible_test_files)
    return (f"Change request:\n{task.request.strip()}\n\nCurrent tests ({', '.join(task.visible_test_files)}):\n"
            f"{visible}\n\nCurrent source of {u.name} in {u.rel}:\n{u.base_fn}\n\n"
            f"Output the complete new source of {u.name}.")


def splice(text: str, span: tuple[int, int], new_fn: str) -> str:
    lines = text.split("\n")
    return "\n".join(lines[:span[0] - 1] + new_fn.rstrip("\n").split("\n") + lines[span[1]:])


def strip_fence(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def _worktree(task: RepairTask) -> Path:
    d = Path(tempfile.mkdtemp(prefix="xyntetik-shadow-adapt-"))
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
class Judgement:
    verified: bool
    fixed: int
    failing: int
    reasons: tuple[str, ...]


def judge(task: RepairTask, u: FunctionUnit, new_fn: str, *, python: str,
          timeout_s: float = 600.0) -> Judgement:
    """Splice ``new_fn`` over the unit's pre-state span in a scratch worktree
    and let the protected verifier judge the whole tree."""
    base_text = _git(task.repo, "show", f"{task.base_sha}:{u.rel}")
    protected = ProtectedTests.load(Path(task.protected_dir))
    ws = _worktree(task)
    try:
        baseline = Baseline.capture(ws)
        (ws / u.rel).write_text(splice(base_text, u.base_span, new_fn), encoding="utf-8")
        outcome = verify(ws, protected, baseline, timeout_s=timeout_s, python=python,
                         pythonpath=task.pythonpath)
        fixed = fixed_ids(protected, task.failing_at_base, ws, timeout_s=timeout_s, python=python,
                          pythonpath=task.pythonpath) \
            if outcome.passed is not None and task.failing_at_base else ()
    finally:
        _drop(task, ws)
    return Judgement(outcome.passed is True, len(fixed), len(task.failing_at_base),
                     tuple(outcome.reasons))


# ---------------------------------------------------------------- dataset

def split(tasks: Sequence[RepairTask], *, holdout_fraction: float, seed: int
          ) -> tuple[list[RepairTask], list[RepairTask]]:
    """Development and held-out slices, seeded; the held-out slice is never
    trained on. The same rule the scaffold optimizer uses."""
    order = list(tasks)
    random.Random(seed).shuffle(order)
    k = max(1, int(round(len(order) * holdout_fraction))) if len(order) > 1 else 0
    return order[k:], order[:k]


@dataclass
class Dataset:
    family: str
    ctx: int
    examples: list[dict[str, object]]
    dev: list[tuple[RepairTask, FunctionUnit]]
    holdout: list[tuple[RepairTask, FunctionUnit]]
    dropped: list[dict[str, str]] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        return {"schema": "xyntetik.shadow.adapt.v1", "unit": "function", "family": self.family,
                "ctx": self.ctx, "chars_per_token": CHARS_PER_TOKEN,
                "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "dev": [{"task_id": t.task_id, "function": u.name, "rel": u.rel} for t, u in self.dev],
                "holdout": [{"task_id": t.task_id, "function": u.name, "rel": u.rel}
                            for t, u in self.holdout],
                "dropped": self.dropped, "examples": len(self.examples)}

    def write(self, out: Path) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(e) + "\n" for e in self.examples)
        (out / "train.jsonl").write_text(data, encoding="utf-8")
        man = self.manifest()
        man["train_sha256"] = hashlib.sha256(data.encode("utf-8")).hexdigest()
        (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        return out / "train.jsonl"


def build_dataset(tasks: Sequence[RepairTask], *, family: str, ctx: int, python: str,
                  seed: int = 0, holdout_fraction: float = 0.3,
                  log: Callable[[str], None] = lambda s: None) -> Dataset:
    """Units for every task that has one and whose human function passes the
    self-check, split into development and held-out; the development units
    that fit the window become training examples."""
    units: list[tuple[RepairTask, FunctionUnit]] = []
    dropped: list[dict[str, str]] = []
    for task in tasks:
        u = function_unit(task)
        if isinstance(u, str):
            dropped.append({"task_id": task.task_id, "why": u})
            continue
        j = judge(task, u, u.sol_fn, python=python)
        if not j.verified:
            dropped.append({"task_id": task.task_id,
                            "why": "the human's function alone is not verified: " + "; ".join(j.reasons)})
            continue
        units.append((task, u))
        log(f"  unit {task.task_id} {u.name} (self-check verified)")
    by_id = {t.task_id: (t, u) for t, u in units}
    dev_t, hold_t = split([t for t, _ in units], holdout_fraction=holdout_fraction, seed=seed)
    dev = [by_id[t.task_id] for t in dev_t]
    holdout = [by_id[t.task_id] for t in hold_t]
    budget = int(ctx * CHARS_PER_TOKEN) - 64
    examples: list[dict[str, object]] = []
    kept: list[tuple[RepairTask, FunctionUnit]] = []
    for task, u in dev:
        prompt = render_prompt(family, SYSTEM, unit_prompt(task, u))
        if len(prompt) + len(u.sol_fn) > budget:
            dropped.append({"task_id": task.task_id,
                            "why": f"{len(prompt) + len(u.sol_fn)} chars over the {budget} window budget"})
            continue
        examples.append({"prompt": prompt, "completion": u.sol_fn, "weight": 1.0})
        kept.append((task, u))
    return Dataset(family, ctx, examples, kept, holdout, dropped)


# ---------------------------------------------------------------- evaluation

@dataclass
class Evaluation:
    label: str
    k: int
    tasks: list[dict[str, Any]]

    @property
    def verified_samples(self) -> int:
        return sum(1 for t in self.tasks for s in t["samples"] if s["verified"])

    @property
    def verified_tasks(self) -> int:
        return sum(1 for t in self.tasks if any(s["verified"] for s in t["samples"]))

    @property
    def samples(self) -> int:
        return sum(len(t["samples"]) for t in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "k": self.k, "tasks": self.tasks,
                "summary": {"tasks": len(self.tasks), "samples": self.samples,
                            "verified_samples": self.verified_samples,
                            "verified_tasks": self.verified_tasks}}


def evaluate(units: Sequence[tuple[RepairTask, FunctionUnit]],
             post_json: Callable[[str, dict[str, Any]], dict[str, Any]], model: str, *,
             label: str, k: int, python: str, temperature: float = 0.8, max_tokens: int = 1500,
             log: Callable[[str], None] = lambda s: None) -> Evaluation:
    """K samples per unit through the served model, each spliced and judged."""
    rows: list[dict[str, Any]] = []
    for task, u in units:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": unit_prompt(task, u)}]
        samples: list[dict[str, Any]] = []
        for i in range(k):
            t0 = time.monotonic()
            resp = post_json("/v1/chat/completions", {
                "model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": 1000 + i})
            text = str((resp.get("choices") or [{}])[0].get("message", {}).get("content") or "")
            wall = time.monotonic() - t0
            new_fn = strip_fence(text)
            j = judge(task, u, new_fn, python=python)
            samples.append({"i": i, "verified": j.verified, "fixed": j.fixed, "failing": j.failing,
                            "gen_wall_s": round(wall, 2), "chars": len(new_fn),
                            "completion_sha256": hashlib.sha256(new_fn.encode("utf-8")).hexdigest()})
            log(f"  {label} {task.task_id} {u.name} sample {i}: verified {j.verified}, "
                f"fixed {j.fixed}/{j.failing}, {wall:.0f}s")
        rows.append({"task_id": task.task_id, "function": u.name, "samples": samples})
    return Evaluation(label, k, rows)


# ---------------------------------------------------------------- training

@dataclass(frozen=True)
class TrainResult:
    adapter: Path
    steps: int
    returncode: int
    log: Path
    first_step_s: float | None
    loss_first: float | None
    loss_last: float | None


def train(runner: str, model: str, data: Path, adapter: Path, *, ctx: int, steps: int, lr: float,
          rank: int, threads: int, log_path: Path, gpu: bool = True,
          log: Callable[[str], None] = lambda s: None) -> TrainResult:
    """One `runner --train` run, its stderr kept in ``log_path``. The GPU
    switch is safe on every build: without CUDA the runner says so and
    trains on the CPU, and the adapter bytes do not depend on it."""
    args = [runner, "-m", model, "--gpu", "off", "--train", str(data), "--train-ctx", str(ctx),
            "--train-steps", str(steps), "--lr", str(lr), "--lora-rank", str(rank),
            "--train-out", str(adapter)]
    if threads > 0:
        args += ["-t", str(threads)]
    env = dict(os.environ)
    env["RUNNER_TRAIN_GPU"] = "1" if gpu else "0"
    env["RUNNER_TRAIN_PROF"] = "1"
    first_step: float | None = None
    loss_first: float | None = None
    loss_last: float | None = None
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
                                text=True, encoding="utf-8", errors="replace")
        assert proc.stderr is not None
        for line in proc.stderr:
            lf.write(line)
            lf.flush()
            if line.startswith("{") and '"step"' in line:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if first_step is None:
                    first_step = float(rec.get("step_s") or 0)
                    loss_first = float(rec.get("loss") or 0)
                    log(f"  step 1: {rec.get('tokens')} tokens in {first_step:.0f}s; "
                        f"{steps} steps is about {first_step * steps / 3600:.1f} h")
                loss_last = float(rec.get("loss") or 0)
                log(f"  step {rec.get('step')}/{steps}: loss {loss_last:.4f}, {rec.get('step_s')}s")
        rc = proc.wait()
    return TrainResult(adapter, steps, rc, log_path, first_step, loss_first, loss_last)


# ---------------------------------------------------------------- verdict

@dataclass(frozen=True)
class Verdict:
    promoted: bool
    holdout_base: int
    holdout_adapter: int
    dev_base: int
    dev_adapter: int
    reason: str


def decide(base_hold: Evaluation, ada_hold: Evaluation, base_dev: Evaluation,
           ada_dev: Evaluation) -> Verdict:
    hb, ha = base_hold.verified_samples, ada_hold.verified_samples
    db, da = base_dev.verified_samples, ada_dev.verified_samples
    if ha > hb:
        return Verdict(True, hb, ha, db, da,
                       f"held-out verified rose {hb} -> {ha} of {ada_hold.samples} samples")
    if da > db:
        return Verdict(False, hb, ha, db, da,
                       f"held-out verified {hb} -> {ha}, development {db} -> {da}: a rise on the "
                       "training slice alone is memorization, not kept")
    return Verdict(False, hb, ha, db, da, f"held-out verified {hb} -> {ha}: no rise, not kept")


def adapters_dir(home: Path) -> Path:
    return home / ".xyntetik" / "shadow" / "adapters"


def record_run(home: Path, stamp: str, *, dataset: Dataset, base_hold: Evaluation,
               ada_hold: Evaluation, base_dev: Evaluation, ada_dev: Evaluation,
               trained: TrainResult, verdict: Verdict, model: str) -> Path:
    d = adapters_dir(home) / stamp
    d.mkdir(parents=True, exist_ok=True)
    if trained.adapter.is_file() and trained.adapter.parent != d:
        shutil.copyfile(trained.adapter, d / "adapter.gguf")
    record = {
        "schema": "xyntetik.shadow.adapt.run.v1", "stamp": stamp, "model": model,
        "model_sha256": _sha256(Path(model)) if Path(model).is_file() else "",
        "dataset": dataset.manifest(),
        "training": {"steps": trained.steps, "returncode": trained.returncode,
                     "first_step_s": trained.first_step_s, "loss_first": trained.loss_first,
                     "loss_last": trained.loss_last, "log": str(trained.log)},
        "base_holdout": base_hold.to_dict(), "adapter_holdout": ada_hold.to_dict(),
        "base_dev": base_dev.to_dict(), "adapter_dev": ada_dev.to_dict(),
        "verdict": asdict(verdict),
        "adapter": str(d / "adapter.gguf") if (d / "adapter.gguf").is_file() else "",
    }
    (d / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_status(home: Path) -> str:
    runs = sorted(p for p in adapters_dir(home).glob("*/run.json")) if adapters_dir(home).is_dir() else []
    if not runs:
        return "no adaptation runs yet"
    lines = ["| run | model | examples | held-out verified base -> adapter | development | kept |",
             "|---|---|---:|---|---|---|"]
    for p in runs:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        v = r.get("verdict", {})
        lines.append(f"| {r.get('stamp')} | {Path(str(r.get('model'))).name} | "
                     f"{r.get('dataset', {}).get('examples')} | "
                     f"{v.get('holdout_base')} -> {v.get('holdout_adapter')} | "
                     f"{v.get('dev_base')} -> {v.get('dev_adapter')} | "
                     f"{'yes' if v.get('promoted') else 'no'} |")
    return "\n".join(lines)
