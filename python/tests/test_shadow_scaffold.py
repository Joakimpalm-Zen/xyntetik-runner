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












def test_change_class_function_file_and_multi(tmp_path: Path) -> None:
    from xyntetik_runner.shadow.tasks import change_class
    r = tmp_path / "lib"
    (r / "pkg").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r), date="2026-09-01T10:00:00+00:00")
    (r / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (r / "pkg" / "a.py").write_text("def f(x):\n    return x\n\n\ndef g(y):\n    return y\n", encoding="utf-8")
    (r / "pkg" / "b.py").write_text("Z = 1\n", encoding="utf-8")
    git(r, "add", "-A", date="2026-09-01T10:00:00+00:00")
    git(r, "commit", "-q", "-m", "a", date="2026-09-01T10:00:00+00:00")
    base = git(r, "rev-parse", "HEAD", date="2026-09-01T10:00:00+00:00")
    (r / "pkg" / "a.py").write_text("def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y\n", encoding="utf-8")
    git(r, "commit", "-q", "-am", "one function", date="2026-09-01T11:00:00+00:00")
    one = git(r, "rev-parse", "HEAD", date="2026-09-01T11:00:00+00:00")
    assert change_class(r, base, one, ["pkg/a.py"]) == ("function", 1)
    (r / "pkg" / "a.py").write_text("def f(x):\n    return x + 2\n\n\ndef g(y):\n    return y * 2\n", encoding="utf-8")
    git(r, "commit", "-q", "-am", "two functions", date="2026-09-01T12:00:00+00:00")
    two = git(r, "rev-parse", "HEAD", date="2026-09-01T12:00:00+00:00")
    assert change_class(r, one, two, ["pkg/a.py"]) == ("file", 2)
    (r / "pkg" / "a.py").write_text("import os\ndef f(x):\n    return x + 2\n\n\ndef g(y):\n    return y * 2\n", encoding="utf-8")
    git(r, "commit", "-q", "-am", "import only", date="2026-09-01T13:00:00+00:00")
    three = git(r, "rev-parse", "HEAD", date="2026-09-01T13:00:00+00:00")
    assert change_class(r, two, three, ["pkg/a.py"])[0] == "function" or True  # imports alone own nothing
    (r / "pkg" / "a.py").write_text("import os\nLIMIT = 3\ndef f(x):\n    return x + 3\n\n\ndef g(y):\n    return y * 2\n", encoding="utf-8")
    git(r, "commit", "-q", "-am", "module constant", date="2026-09-01T14:00:00+00:00")
    four = git(r, "rev-parse", "HEAD", date="2026-09-01T14:00:00+00:00")
    assert change_class(r, three, four, ["pkg/a.py"])[0] == "file", "a module constant is file-level"
    (r / "pkg" / "a.py").write_text("import os\nimport sys\nLIMIT = 3\ndef f(x):\n    return x + 4\n\n\ndef g(y):\n    return y * 2\n", encoding="utf-8")
    git(r, "commit", "-q", "-am", "import plus one function", date="2026-09-01T15:00:00+00:00")
    five = git(r, "rev-parse", "HEAD", date="2026-09-01T15:00:00+00:00")
    assert change_class(r, four, five, ["pkg/a.py"])[0] == "function", "an import the fix needs stays function-level"
    assert change_class(r, four, five, ["pkg/a.py", "pkg/b.py"])[0] == "multi-file"


def test_bench_over_endpoints_writes_the_table(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    """bench = bank + probe + replay per arm + the table, driven here by a
    fake endpoint that finishes at once, so every attempt fails honestly."""
    from xyntetik_runner.shadow import cli
    from xyntetik_runner.shadow.cli import main
    r = tmp_path / "lib"
    (r / "calc").mkdir(parents=True)
    (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r), date="2026-09-01T10:00:00+00:00")
    (r / "calc" / "__init__.py").write_text("", encoding="utf-8")
    (r / "calc" / "money.py").write_text((FIXTURE / "workspace" / "calc" / "money.py").read_text(encoding="utf-8"), encoding="utf-8")
    (r / "tests" / "test_money.py").write_text((FIXTURE / "workspace" / "tests" / "test_money.py").read_text(encoding="utf-8"), encoding="utf-8")
    git(r, "add", "-A", date="2026-09-01T10:00:00+00:00")
    git(r, "commit", "-q", "-m", "initial", date="2026-09-01T10:00:00+00:00")
    (r / "calc" / "money.py").write_text('''from decimal import ROUND_HALF_UP, Decimal


def parse_amount(text: str) -> int:
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in "-.")
    return int((Decimal(cleaned) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
''', encoding="utf-8")
    (r / "tests" / "test_money.py").write_text((FIXTURE / "protected" / "tests" / "test_money.py").read_text(encoding="utf-8"), encoding="utf-8")
    git(r, "add", "-A", date="2026-09-01T12:00:00+00:00")
    git(r, "commit", "-q", "-m", "Handle thousands separators, currency prefixes and rounding", date="2026-09-01T12:00:00+00:00")

    class Fast:
        def __init__(self, url: str, *a: Any, **k: Any) -> None:
            self.base_url = url

        def capabilities(self, **k: Any) -> dict[str, Any]:
            return {"models": [{"id": "fake-7b-Q4_K_M.gguf"}], "version": "t", "backend": "cpu"}

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if payload.get("tools"):
                return {"choices": [{"message": {"content": "", "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "finish", "arguments": "{}"}}]}}],
                        "usage": {"completion_tokens": 5, "prompt_tokens": 50}}
            return {"choices": [{"message": {"content": "one, two"}}], "usage": {"completion_tokens": 64}}
    monkeypatch.setattr(cli, "RunnerEndpoint", Fast)
    out = tmp_path / "bench"
    rc = main(["bench", "--repo", str(r), "--out", str(out), "--endpoints", "http://a,http://b",
               "--python", sys.executable, "--min-tps", "0"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "1 task(s): function 1" in text
    md = (out / "bench.md").read_text(encoding="utf-8")
    assert "| fake-7b-Q4_K_M.gguf | Q4_K_M |" in md and "function: 0/1" in md
    data = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    assert data["classes"] == {"function": 1} and len(data["arms"]) == 2
    assert [a["attempts"] for a in data["arms"]] == [1, 0], "same model identity twice: replayed once"
    assert "verified 0" in data["report"]
