"""Scaffold optimization (R15.0.4): the local model as its own reflector.

The loop is deliberately the simplest member of the reflective
prompt-evolution family (GEPA, 2025): run the current best scaffold on a
minibatch of tasks, collect the failure text (stop reason, verifier
reasons, the tail of what the model saw), ask the local model for a
revised scaffold as JSON, score the child on the same minibatch, keep it
if it is better. The reward is the protected verifier and the count of
tests turned green, never a judge. A held-out slice, never touched during
the search, decides at the end against the base scaffold under the
display rule; the kill gate is no held-out gain within the rollout budget.

The optimizer writes system text, procedure and tool descriptions. It does
not write exemplars: exemplars are content, and the firewall applies.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xyntetik_runner.shadow.scaffold import Scaffold
from xyntetik_runner.shadow.tasks import RepairTask


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    fixed: int
    failing_at_base: int
    verified: bool
    feedback: str


RunAttempt = Callable[[RepairTask, Scaffold], TaskOutcome]
Reflect = Callable[[Scaffold, Sequence[TaskOutcome]], Scaffold]


@dataclass(frozen=True)
class Score:
    fixed: int
    failing: int
    verified: int
    n: int

    @property
    def fraction(self) -> float:
        return self.fixed / self.failing if self.failing else 0.0

    def better_than(self, other: Score) -> bool:
        return (self.verified, self.fixed) > (other.verified, other.fixed)


def score(tasks: Sequence[RepairTask], scaffold: Scaffold, run: RunAttempt
          ) -> tuple[Score, list[TaskOutcome]]:
    outcomes = [run(t, scaffold) for t in tasks]
    return (Score(fixed=sum(o.fixed for o in outcomes),
                  failing=sum(o.failing_at_base for o in outcomes),
                  verified=sum(1 for o in outcomes if o.verified), n=len(outcomes)), outcomes)


def split(tasks: Sequence[RepairTask], *, holdout_fraction: float, seed: int
          ) -> tuple[list[RepairTask], list[RepairTask]]:
    """Development and held-out slices, seeded; the held-out slice is never
    used during the search."""
    order = list(tasks)
    random.Random(seed).shuffle(order)
    k = max(1, int(round(len(order) * holdout_fraction))) if len(order) > 1 else 0
    return order[k:], order[:k]


REFLECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "system": {"type": "string"},
        "procedure": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "tool_descriptions": {"type": "object",
                              "additionalProperties": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["system", "procedure", "notes"],
}


def reflect_prompt(scaffold: Scaffold, outcomes: Sequence[TaskOutcome]) -> str:
    failures = [o for o in outcomes if not o.verified]
    shown = "\n\n".join(
        f"### {o.task_id}: fixed {o.fixed} of {o.failing_at_base}\n{o.feedback[:1500]}"
        for o in failures[:4]) or "(all attempts passed)"
    current = json.dumps({"system": scaffold.system, "procedure": list(scaffold.procedure),
                          "tool_descriptions": dict(scaffold.tool_descriptions)}, indent=2)
    return (
        "You are improving the instructions a coding agent follows. The agent has five tools "
        "(list_files, read_file, write_file, run_tests, finish) and no shell. Its work is "
        "judged by hidden tests it cannot see or change.\n\n"
        f"Current instructions (JSON):\n{current}\n\n"
        f"What went wrong on recent tasks with these instructions:\n{shown}\n\n"
        "Write improved instructions as JSON with keys system (one paragraph), procedure "
        "(a short list of concrete steps), tool_descriptions (optional, tool name to "
        "description) and notes (one sentence on what you changed and why). Keep what works. "
        "Do not add examples. Output only the JSON object."
    )


def parse_reflection(text: str, parent: Scaffold, name: str) -> Scaffold:
    """The model's JSON, tolerant of a code fence, into a child scaffold that
    keeps the parent's exemplars and budget untouched."""
    body = text.strip()
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        raise ValueError("reflection carried no JSON object")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("reflection JSON is not an object")
    tools = data.get("tool_descriptions") or {}
    return Scaffold(
        name=name, system=str(data.get("system") or parent.system),
        procedure=tuple(str(p) for p in (data.get("procedure") or ()))[:12],
        tool_descriptions={str(k): str(v) for k, v in tools.items()} if isinstance(tools, dict) else {},
        exemplars=parent.exemplars, budget=parent.budget,
        notes=str(data.get("notes") or ""), parent_sha256=parent.sha256)


def runner_reflect(post_json: Callable[..., dict[str, Any]], model: str, *, max_tokens: int = 1200
                   ) -> Reflect:
    """Reflection through the runner, schema-constrained where the server
    accepts ``response_format``; plain JSON parsing otherwise."""
    def reflect(scaffold: Scaffold, outcomes: Sequence[TaskOutcome]) -> Scaffold:
        prompt = reflect_prompt(scaffold, outcomes)
        payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}],
                                   "max_tokens": max_tokens, "temperature": 0.7,
                                   "response_format": {"type": "json_schema", "json_schema": {
                                       "name": "scaffold", "schema": REFLECT_SCHEMA}}}
        try:
            data = post_json("/v1/chat/completions", payload)
        except Exception:
            payload.pop("response_format")
            data = post_json("/v1/chat/completions", payload)
        text = str(data["choices"][0]["message"].get("content") or "")
        return parse_reflection(text, scaffold, name=f"{scaffold.name}>{scaffold.sha256[:6]}")
    return reflect


@dataclass(frozen=True)
class Generation:
    index: int
    parent_sha256: str
    child_sha256: str
    parent: Score
    child: Score
    accepted: bool
    batch: tuple[str, ...]


@dataclass(frozen=True)
class OptimizeResult:
    best: Scaffold
    generations: tuple[Generation, ...]
    rollouts: int
    holdout_base: Score | None
    holdout_best: Score | None

    @property
    def held_out_gain(self) -> bool:
        return (self.holdout_base is not None and self.holdout_best is not None
                and self.holdout_best.better_than(self.holdout_base))


def optimize(tasks: Sequence[RepairTask], base: Scaffold, run: RunAttempt, reflect: Reflect, *,
             generations: int = 4, batch: int = 4, holdout_fraction: float = 0.3, seed: int = 0,
             rollout_budget: int = 200, save_dir: Path | None = None,
             log: Callable[[str], None] | None = None) -> OptimizeResult:
    say = log or (lambda _s: None)
    dev, holdout = split(tasks, holdout_fraction=holdout_fraction, seed=seed)
    rng = random.Random(seed)
    best = base
    rollouts = 0
    gens: list[Generation] = []
    if save_dir is not None:
        base.save(save_dir / f"gen0-{base.sha256[:8]}.json")
    for g in range(1, generations + 1):
        minibatch = rng.sample(dev, min(batch, len(dev))) if dev else []
        if not minibatch or rollouts + 2 * len(minibatch) > rollout_budget:
            say(f"gen {g}: stopping, rollout budget {rollout_budget} would be exceeded")
            break
        parent_score, outcomes = score(minibatch, best, run)
        rollouts += len(minibatch)
        try:
            child = reflect(best, outcomes)
        except (ValueError, KeyError, TypeError) as e:
            say(f"gen {g}: reflection unusable ({e}); parent kept")
            continue
        child_score, _ = score(minibatch, child, run)
        rollouts += len(minibatch)
        accepted = child_score.better_than(parent_score)
        gens.append(Generation(index=g, parent_sha256=best.sha256, child_sha256=child.sha256,
                               parent=parent_score, child=child_score, accepted=accepted,
                               batch=tuple(t.task_id for t in minibatch)))
        say(f"gen {g}: parent fixed {parent_score.fixed}/{parent_score.failing} verified "
            f"{parent_score.verified}; child fixed {child_score.fixed}/{child_score.failing} "
            f"verified {child_score.verified}; {'ACCEPTED' if accepted else 'rejected'}")
        if save_dir is not None:
            child.save(save_dir / f"gen{g}-{child.sha256[:8]}{'-accepted' if accepted else ''}.json")
        if accepted:
            best = child
    hb = hs = None
    if holdout and best.sha256 != base.sha256:
        hb, _ = score(holdout, base, run)
        hs, _ = score(holdout, best, run)
        rollouts += 2 * len(holdout)
        say(f"holdout ({len(holdout)} tasks): base fixed {hb.fixed}/{hb.failing} verified "
            f"{hb.verified}; best fixed {hs.fixed}/{hs.failing} verified {hs.verified}")
    elif holdout:
        say("holdout: no accepted child, nothing to compare")
    return OptimizeResult(best=best, generations=tuple(gens), rollouts=rollouts,
                          holdout_base=hb, holdout_best=hs)


def result_json(r: OptimizeResult) -> str:
    return json.dumps({"best_sha256": r.best.sha256, "best_name": r.best.name, "rollouts": r.rollouts,
                       "held_out_gain": r.held_out_gain,
                       "holdout_base": asdict(r.holdout_base) if r.holdout_base else None,
                       "holdout_best": asdict(r.holdout_best) if r.holdout_best else None,
                       "generations": [asdict(g) for g in r.generations]}, indent=2)
