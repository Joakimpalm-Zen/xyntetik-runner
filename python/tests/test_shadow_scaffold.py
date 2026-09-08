"""The scaffold artifact, the bank and the optimizer, without a model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from xyntetik_runner.shadow import Identity, Scaffold
from xyntetik_runner.shadow.attempt import TOOLS, Budget, Workspace, attempt, budget_with
from xyntetik_runner.shadow.bank import build_bank
from xyntetik_runner.shadow.optimize import (
    Score,
    TaskOutcome,
    optimize,
    parse_reflection,
    reflect_prompt,
    split,
)
from xyntetik_runner.shadow.scaffold import BASE_SYSTEM
from xyntetik_runner.shadow.tasks import RepairTask

FIXTURE = Path(__file__).parent / "fixtures" / "repair_task_v1"


def test_scaffold_hash_is_canonical_and_base_matches_the_harness(tmp_path: Path) -> None:
    a = Scaffold(name="x", system="s", tool_descriptions={"b": "2", "a": "1"}, budget={"z": 1, "y": 2})
    b = Scaffold(name="x", system="s", tool_descriptions={"a": "1", "b": "2"}, budget={"y": 2, "z": 1})
    assert a.sha256 == b.sha256
    assert Scaffold.base().system == BASE_SYSTEM and Scaffold.base().system_text() == BASE_SYSTEM
    p = tmp_path / "s.json"
    sha = a.save(p)
    assert Scaffold.load(p) == a and sha == a.sha256
    with pytest.raises(ValueError):
        Scaffold(name="x", system="   ")


def test_scaffold_shapes_system_text_tools_and_budget() -> None:
    s = Scaffold(name="t", system="Fix it.", procedure=("read the test", "run tests"),
                 exemplars=("good: ...",), tool_descriptions={"run_tests": "Run them."},
                 budget={"max_turns": 3, "wall_s": 99999, "nonsense": 1})
    text = s.system_text()
    assert text.startswith("Fix it.") and "1. read the test" in text and "Examples of good work" in text
    tools = s.apply_tools(TOOLS)
    assert [t["function"]["description"] for t in tools if t["function"]["name"] == "run_tests"] == ["Run them."]
    assert all(t["function"]["name"] for t in tools)
    b = budget_with(s, Budget(max_turns=12, wall_s=900.0))
    assert b.max_turns == 3 and b.wall_s == 900.0, "a scaffold can spend less, never more"


def test_identity_carries_the_scaffold_and_the_stack_key_splits_on_it() -> None:
    base = Identity(project="p", task_class="repair", context_band="<2k", tool_set=(), verifier_id="v",
                    environment_id="e", model_sha256="m", quant="q", template_sha256="t",
                    runner_build="b", backend="cpu", harness_version="h")
    other = Identity(**{**base.__dict__, "scaffold_sha256": "abc"})
    assert base.scaffold_sha256 == "base" and base.stack_key() != other.stack_key()


def test_attempt_uses_the_scaffold_system_text(tmp_path: Path) -> None:
    seen: list[list[dict[str, Any]]] = []

    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        seen.append(messages)
        assert tools[0]["function"]["description"] == "custom listing"
        return {"content": "", "tool_calls": [{"id": "1", "function": {"name": "finish", "arguments": "{}"}}]}
    s = Scaffold(name="t", system="Be brief.", procedure=("finish at once",),
                 tool_descriptions={"list_files": "custom listing"}, budget={"max_turns": 1})
    ws = Workspace(tmp_path, visible_tests=(), pythonpath=(), python=sys.executable, budget=Budget())
    r = attempt("x", ws, chat, scaffold=s)
    assert r.finished and seen[0][0]["content"].startswith("Be brief.") and "1. finish at once" in seen[0][0]["content"]


def git(repo: Path, *args: str, date: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env,
                          check=True).stdout.strip()


def test_bank_builds_tasks_from_a_public_history(tmp_path: Path) -> None:
    r = tmp_path / "lib"
    (r / "calc").mkdir(parents=True)
    (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r), date="2026-09-01T10:00:00+00:00")
    (r / "calc" / "__init__.py").write_text("", encoding="utf-8")
    (r / "calc" / "money.py").write_text((FIXTURE / "workspace" / "calc" / "money.py").read_text(), encoding="utf-8")
    (r / "tests" / "test_money.py").write_text((FIXTURE / "workspace" / "tests" / "test_money.py").read_text(), encoding="utf-8")
    git(r, "add", "-A", date="2026-09-01T10:00:00+00:00")
    git(r, "commit", "-q", "-m", "initial", date="2026-09-01T10:00:00+00:00")
    (r / "README.md").write_text("docs only\n", encoding="utf-8")
    git(r, "add", "-A", date="2026-09-01T11:00:00+00:00")
    git(r, "commit", "-q", "-m", "docs", date="2026-09-01T11:00:00+00:00")
    (r / "calc" / "money.py").write_text('''from decimal import ROUND_HALF_UP, Decimal


def parse_amount(text: str) -> int:
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in "-.")
    return int((Decimal(cleaned) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
''', encoding="utf-8")
    (r / "tests" / "test_money.py").write_text((FIXTURE / "protected" / "tests" / "test_money.py").read_text(), encoding="utf-8")
    git(r, "add", "-A", date="2026-09-01T12:00:00+00:00")
    git(r, "commit", "-q", "-m", "Handle thousands separators, currency prefixes and rounding\n\nFixes #7",
        date="2026-09-01T12:00:00+00:00")
    entries = build_bank(r, out_dir=tmp_path / "tasks", python=sys.executable, limit=5, max_commits=10)
    admitted = [e for e in entries if e.task]
    assert len(admitted) == 1 and admitted[0].task is not None
    task = admitted[0].task
    assert task.request.startswith("Handle thousands separators") and "Fixes #7" in task.request
    assert task.expected_tests == 6 and task.baseline_failing == 3 and task.episode_id.startswith("bank:")
    assert {e.reason for e in entries if not e.task} == {
        "no test+source pair", "unreplayable: first commit in the range has no parent"}


def fake_task(i: int) -> RepairTask:
    return RepairTask(task_id=f"t{i}", episode_id=f"bank:x:{i}", repo="/r", base_sha="a", solution_sha="b",
                      request="fix", request_sha256="s", context=(), test_files=("tests/t.py",),
                      visible_test_files=(), src_files=1, pythonpath=(), protected_dir="/p",
                      expected_tests=2, baseline_failing=2, failing_at_base=("tests/t.py::a", "tests/t.py::b"))


def test_optimizer_keeps_a_child_only_when_it_scores_better_and_reports_holdout(tmp_path: Path) -> None:
    tasks = [fake_task(i) for i in range(10)]

    def run(task: RepairTask, scaffold: Scaffold) -> TaskOutcome:
        good = "run the tests first" in " ".join(scaffold.procedure)
        return TaskOutcome(task.task_id, fixed=2 if good else 0, failing_at_base=2, verified=good,
                           feedback="stop: finish; verifier: 2 expected test(s) failed")
    calls = {"n": 0}

    def reflect(scaffold: Scaffold, outcomes: list[TaskOutcome]) -> Scaffold:
        calls["n"] += 1
        step = "run the tests first" if calls["n"] == 2 else "think harder"
        return Scaffold(name="child", system=scaffold.system, procedure=(*scaffold.procedure, step),
                        parent_sha256=scaffold.sha256)
    log: list[str] = []
    r = optimize(tasks, Scaffold.base(), run, reflect, generations=3, batch=3, holdout_fraction=0.3,
                 seed=1, rollout_budget=100, save_dir=tmp_path, log=log.append)
    assert [g.accepted for g in r.generations] == [False, True, False]
    assert "run the tests first" in r.best.procedure and r.best.parent_sha256 == Scaffold.base().sha256
    assert r.holdout_base is not None and r.holdout_best is not None and r.held_out_gain
    assert r.holdout_best.fixed == 2 * len(split(tasks, holdout_fraction=0.3, seed=1)[1])
    assert any(p.name.endswith("-accepted.json") for p in tmp_path.glob("gen*.json"))
    assert r.rollouts == 3 * 6 + 2 * 3


def test_optimizer_respects_the_rollout_budget_and_survives_bad_reflection() -> None:
    tasks = [fake_task(i) for i in range(6)]
    run = lambda t, s: TaskOutcome(t.task_id, 0, 2, False, "x")  # noqa: E731
    reflect = lambda s, o: (_ for _ in ()).throw(ValueError("garbage"))  # noqa: E731
    r = optimize(tasks, Scaffold.base(), run, reflect, generations=5, batch=2, holdout_fraction=0.3,
                 seed=0, rollout_budget=3)
    assert r.generations == () and r.rollouts == 0 and not r.held_out_gain
    r = optimize(tasks, Scaffold.base(), run, reflect, generations=2, batch=2, holdout_fraction=0.3,
                 seed=0, rollout_budget=100)
    assert r.generations == () and r.best == Scaffold.base()


def test_reflection_parsing_keeps_exemplars_and_budget_from_the_parent() -> None:
    parent = Scaffold(name="p", system="s", exemplars=("keep me",), budget={"max_turns": 2})
    text = "Here you go:\n```json\n" + json.dumps({"system": "new", "procedure": ["a", "b"],
                                                    "tool_descriptions": {"finish": "done"},
                                                    "notes": "shorter"}) + "\n```"
    child = parse_reflection(text, parent, name="c")
    assert child.system == "new" and child.procedure == ("a", "b") and child.exemplars == ("keep me",)
    assert child.budget == {"max_turns": 2} and child.parent_sha256 == parent.sha256
    with pytest.raises(ValueError):
        parse_reflection("no json here", parent, name="c")
    prompt = reflect_prompt(parent, [TaskOutcome("t1", 0, 2, False, "boom" * 1000)])
    assert "Do not add examples" in prompt and len(prompt) < 5000


def test_score_ordering() -> None:
    assert Score(fixed=3, failing=4, verified=1, n=2).better_than(Score(fixed=4, failing=4, verified=0, n=2))
    assert not Score(fixed=1, failing=4, verified=0, n=2).better_than(Score(fixed=1, failing=4, verified=0, n=2))
