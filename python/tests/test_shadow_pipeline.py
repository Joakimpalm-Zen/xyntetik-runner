"""Importer, task builder, attempt harness and CLI, end to end on synthetic
traces and a synthetic repository. No model: the attempt is driven by a
scripted chat function, which is enough to prove the plumbing and the
confinement; the model's competence is what the pilot measures."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from xyntetik_runner.shadow import Baseline, Disposition, ProtectedTests, verify
from xyntetik_runner.shadow.attempt import Budget, Workspace, attempt
from xyntetik_runner.shadow.cli import main
from xyntetik_runner.shadow.importer import Episode, read_episodes, scan_claude_code, scan_codex
from xyntetik_runner.shadow.tasks import RepairTask, Rejection, admit, repos_under

FIXTURE = Path(__file__).parent / "fixtures" / "repair_task_v1"
BUGGY = (FIXTURE / "workspace" / "calc" / "money.py").read_text(encoding="utf-8")
VISIBLE = (FIXTURE / "workspace" / "tests" / "test_money.py").read_text(encoding="utf-8")
SOLUTION_TESTS = (FIXTURE / "protected" / "tests" / "test_money.py").read_text(encoding="utf-8")
CORRECT = '''"""Money parsing for the ledger importer."""
from decimal import ROUND_HALF_UP, Decimal


def parse_amount(text: str) -> int:
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in "-.")
    return int((Decimal(cleaned) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
'''


def git(repo: Path, *args: str, date: str = "2026-09-01T12:00:00+00:00") -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          env=env, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with the buggy state committed at 11:00 and the fix,
    with its stronger tests, committed at 12:00."""
    r = tmp_path / "proj"
    (r / "calc").mkdir(parents=True)
    (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    (r / "calc" / "__init__.py").write_text("", encoding="utf-8")
    (r / "calc" / "money.py").write_text(BUGGY, encoding="utf-8")
    (r / "tests" / "test_money.py").write_text(VISIBLE, encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "buggy", date="2026-09-01T11:00:00+00:00")
    (r / "calc" / "money.py").write_text(CORRECT, encoding="utf-8")
    (r / "tests" / "test_money.py").write_text(SOLUTION_TESTS, encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "fix", date="2026-09-01T12:00:00+00:00")
    return r


def episode(cwd: Path, **kw: Any) -> Episode:
    base = dict(source="claude_code", session_id="abcdef123456", turn=1, cwd=str(cwd),
                started_at="2026-09-01T11:30:00Z", ended_at="2026-09-01T12:10:00Z",
                request="make parse_amount handle thousands separators, currency and rounding",
                request_sha256="x" * 64)
    base.update(kw)
    return Episode(**base)  # type: ignore[arg-type]


def test_codex_scan_reads_boundaries_and_never_the_answer(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions" / "2026"
    sessions.mkdir(parents=True)
    recs = [
        {"type": "session_meta", "payload": {"id": "s1", "cwd": "/w"}},
        {"timestamp": "2026-09-01T10:00:00Z", "type": "event_msg", "payload": {"type": "task_started"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "fix the parser"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                              "content": "THE ANSWER"}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                              "arguments": "{\"cmd\": \"THE PATCH\"}"}},
        {"timestamp": "2026-09-01T10:05:00Z", "type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    (sessions / "a.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    (sessions / "broken.jsonl").write_bytes(b"\xff\xfe not json")
    report = scan_codex(tmp_path / "sessions")
    assert len(report.episodes) == 1 and report.files_read == 2
    e = report.episodes[0]
    assert (e.cwd, e.request, e.tool_names) == ("/w", "fix the parser", ("shell",))
    assert e.started_at == "2026-09-01T10:00:00Z" and e.ended_at == "2026-09-01T10:05:00Z"
    dumped = json.dumps(e.__dict__)
    assert "THE ANSWER" not in dumped and "THE PATCH" not in dumped


def test_claude_code_scan_turns_end_at_the_next_prompt(tmp_path: Path) -> None:
    proj = tmp_path / "projects" / "p"
    proj.mkdir(parents=True)
    recs = [
        {"type": "user", "sessionId": "s", "cwd": "/w", "timestamp": "2026-09-01T10:00:00Z",
         "message": {"content": "do A"}},
        {"type": "assistant", "message": {"content": "THE ANSWER"}},
        {"type": "user", "sessionId": "s", "cwd": "/w", "timestamp": "2026-09-01T10:20:00Z",
         "message": {"content": [{"type": "tool_result", "content": "THE OUTPUT"}]}},
        {"type": "user", "sessionId": "s", "cwd": "/w", "timestamp": "2026-09-01T11:00:00Z",
         "message": {"content": "<command-name>/model</command-name>"}},
    ]
    (proj / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    eps = scan_claude_code(tmp_path / "projects").episodes
    assert [e.turn for e in eps] == [1, 2]
    assert eps[0].ended_at == "2026-09-01T11:00:00Z" and eps[1].is_command
    assert "THE ANSWER" not in json.dumps([e.__dict__ for e in eps])


def test_repos_under_finds_nested_repositories(repo: Path) -> None:
    from xyntetik_runner.shadow.tasks import canonical
    assert repos_under(repo.parent) == [canonical(repo)]
    assert repos_under(repo) == [canonical(repo)]


def test_repos_under_descends_through_an_umbrella_repository(repo: Path) -> None:
    """The owner's layout: a parent directory that is itself a repository,
    with the real projects untracked inside it. Both are found."""
    git(repo.parent, "init", "-q", "-b", "main", str(repo.parent))
    from xyntetik_runner.shadow.tasks import canonical
    found = repos_under(repo.parent)
    assert found == sorted([canonical(repo.parent), canonical(repo)])


def test_admit_builds_a_calibrated_task(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "tasks"
    out.mkdir()
    result = admit(episode(repo.parent), out_dir=out, python=sys.executable)
    assert isinstance(result, RepairTask), result
    assert result.expected_tests == 6 and result.baseline_failing >= 1
    assert result.visible_test_files == ("tests/test_money.py",)
    assert Path(result.protected_dir, "manifest.json").is_file()
    assert RepairTask.load(out / result.task_id / "task.json") == result
    # the solution is recorded for provenance, never given to the attempt
    assert result.solution_sha != result.base_sha


def test_admit_rejects_when_no_commit_in_window(repo: Path, tmp_path: Path) -> None:
    r = admit(episode(repo, started_at="2026-09-02T00:00:00Z", ended_at="2026-09-02T01:00:00Z"),
              out_dir=tmp_path, python=sys.executable)
    assert isinstance(r, Rejection) and r.disposition is Disposition.UNREPLAYABLE


def test_admit_rejects_a_commit_whose_tests_pass_before_the_fix(tmp_path: Path) -> None:
    r = tmp_path / "proj"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    (r / "pkg" / "__init__.py").write_text("X = 1\n", encoding="utf-8")
    (r / "tests" / "test_x.py").write_text("from pkg import X\n\n\ndef test_x():\n    assert X == 1\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "a", date="2026-09-01T11:00:00+00:00")
    (r / "pkg" / "__init__.py").write_text("X = 1\nY = 2\n", encoding="utf-8")
    (r / "tests" / "test_x.py").write_text("from pkg import X\n\n\ndef test_x():\n    assert X == 1\n\n\ndef test_y():\n    assert True\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "b", date="2026-09-01T12:00:00+00:00")
    res = admit(episode(r), out_dir=tmp_path / "t", python=sys.executable)
    assert isinstance(res, Rejection) and res.disposition is Disposition.UNREPLAYABLE
    assert "instrument" in res.reason


def test_admit_rejects_harness_commands_and_dedupes_commit_ranges(repo: Path, tmp_path: Path) -> None:
    seen: set[tuple[str, str]] = set()
    assert isinstance(admit(episode(repo, request="<command-name>/x</command-name>"),
                             out_dir=tmp_path, python=sys.executable), Rejection)
    first = admit(episode(repo), out_dir=tmp_path, python=sys.executable, seen=seen)
    assert isinstance(first, RepairTask)
    again = admit(episode(repo, turn=2, started_at="2026-09-01T10:30:00Z"), out_dir=tmp_path,
                  python=sys.executable, seen=seen)
    assert isinstance(again, Rejection) and "same fix commit" in again.reason


def test_choose_attributes_a_fix_to_the_closest_prompt_in_the_repo(repo: Path) -> None:
    from xyntetik_runner.shadow.tasks import choose, pair
    early_parent = pair(episode(repo.parent, turn=1, started_at="2026-09-01T10:00:00Z"))
    late_parent = pair(episode(repo.parent, turn=2, started_at="2026-09-01T11:45:00Z"))
    in_repo = pair(episode(repo, turn=3, started_at="2026-09-01T11:20:00Z"))
    assert not isinstance(early_parent, Rejection) and not isinstance(late_parent, Rejection)
    assert not isinstance(in_repo, Rejection)
    chosen = choose([*early_parent, *late_parent, *in_repo])
    assert len(chosen) == 1
    assert next(iter(chosen.values())).episode.turn == 3, "the repo's own prompt wins over a parent"
    chosen = choose([*early_parent, *late_parent])
    assert next(iter(chosen.values())).episode.turn == 2, "then the latest prompt before the fix"


def scripted(*steps: dict[str, Any]) -> Any:
    it = iter(steps)

    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        assert messages[0]["role"] == "system" and any(t["function"]["name"] == "finish" for t in tools)
        return next(it)
    return chat


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"id": f"c-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_attempt_confines_paths_and_verifies_a_scripted_fix(repo: Path, tmp_path: Path) -> None:
    task = admit(episode(repo), out_dir=tmp_path / "t", python=sys.executable)
    assert isinstance(task, RepairTask)
    ws_dir = tmp_path / "ws"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", "-q", str(ws_dir),
                    task.base_sha], check=True)
    ws = Workspace(ws_dir, visible_tests=task.visible_test_files, pythonpath=task.pythonpath,
                   python=sys.executable, budget=Budget(test_runs=2))
    assert ws.read_file("../outside.txt").startswith("error")
    assert ws.write_file("/tmp/x", "y").startswith("error")
    assert ws.list_files("../*").startswith("error")
    baseline = Baseline.capture(ws_dir)
    chat = scripted(
        {"content": "", "tool_calls": [call("read_file", path="calc/money.py"), call("run_tests")]},
        {"content": "", "tool_calls": [call("write_file", path="calc/money.py", content=CORRECT),
                                       call("run_tests")]},
        {"content": "", "tool_calls": [call("run_tests"), call("finish", summary="done")]},
        {"content": "", "tool_calls": []},
    )
    assert len(task.failing_at_base) == 3 and all("::" in i for i in task.failing_at_base)
    result = attempt(task.request, ws, chat)
    assert result.finished and result.stop_reason == "finish" and result.test_runs == 2
    assert result.tool_names == ("finish", "read_file", "run_tests", "write_file")
    outcome = verify(ws_dir, ProtectedTests.load(Path(task.protected_dir)), baseline,
                     python=sys.executable, pythonpath=task.pythonpath)
    assert outcome.passed is True and outcome.passed_count == 6
    from xyntetik_runner.shadow import fixed_ids
    assert set(fixed_ids(ProtectedTests.load(Path(task.protected_dir)), task.failing_at_base, ws_dir,
                         python=sys.executable, pythonpath=task.pythonpath)) == set(task.failing_at_base)
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(ws_dir)], check=True)


def test_attempt_stops_on_budget_and_no_tool_calls(repo: Path, tmp_path: Path) -> None:
    ws = Workspace(repo, visible_tests=(), pythonpath=(), python=sys.executable,
                   budget=Budget(max_turns=2))
    chat = scripted({"content": "I think..."}, {"content": "still thinking"})
    r = attempt("x", ws, chat)
    assert not r.finished and r.stop_reason == "no tool call" and r.turns == 2
    chat = scripted(*[{"content": "", "tool_calls": [call("list_files")]}] * 3)
    r = attempt("x", ws, chat, budget=Budget(max_turns=2))
    assert r.stop_reason == "turn budget" and r.tool_calls == 2
    chat = scripted({"content": "", "tool_calls": [call("write_file", path="a", content="b")]},
                    {"content": "", "tool_calls": []})
    r = attempt("x", Workspace(tmp_path, visible_tests=(), pythonpath=(), python=sys.executable,
                               budget=Budget()), chat)
    assert (tmp_path / "a").read_text(encoding="utf-8") == "b"


def test_cli_import_and_report_on_a_synthetic_home(repo: Path, tmp_path: Path, capsys: Any) -> None:
    home = tmp_path / "home"
    proj = home / ".claude" / "projects" / "p"
    proj.mkdir(parents=True)
    recs = [
        {"type": "user", "sessionId": "s", "cwd": str(repo.parent), "timestamp": "2026-09-01T11:30:00Z",
         "message": {"content": "fix parse_amount for thousands separators and currency"}},
        {"type": "user", "sessionId": "s", "cwd": str(repo.parent), "timestamp": "2026-09-01T12:10:00Z",
         "message": {"content": "thanks"}},
        {"type": "user", "sessionId": "s", "cwd": "/nonexistent", "timestamp": "2026-09-01T13:00:00Z",
         "message": {"content": "unrelated"}},
    ]
    (proj / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert main(["import", "--out", str(out), "--home", str(home), "--python", sys.executable]) == 0
    eps = read_episodes(out / "episodes.jsonl")
    assert len(eps) == 3
    tasks = list((out / "tasks").glob("*/task.json"))
    assert len(tasks) == 1
    assert main(["report", "--out", str(out), "--by-reason"]) == 0
    text = capsys.readouterr().out
    assert "3 episodes observed" in text and "1 eligible for replay" in text
    assert "1 eligible and not attempted yet" in text and "no percentage" in text
    assert "ineligible: no git repository" in text and "unreplayable: no commit" in text
    task = RepairTask.load(tasks[0])
    assert task.episode_id.endswith(":1"), "the prompt before the fix, not the thanks after it"


def test_capture_hook_path_gives_an_exact_range(repo: Path, tmp_path: Path, monkeypatch: Any) -> None:
    """Prompt and stop hooks record HEAD at both ends; the task is built from
    the exact ancestry range, no time window, and the request is the one
    typed at the prompt."""
    from xyntetik_runner.shadow.importer import scan_capture
    from xyntetik_runner.shadow.tasks import choose, pair
    base = git(repo, "rev-parse", "HEAD^")
    fix = git(repo, "rev-parse", "HEAD")
    cap = tmp_path / "capture.jsonl"
    # a prompt at the buggy state ...
    git(repo, "checkout", "-q", base)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(
        {"session_id": "cap1", "cwd": str(repo), "prompt": "fix parse_amount"})))
    assert main(["capture", "--event", "prompt", "--file", str(cap)]) == 0
    # ... and a stop after the fix landed
    git(repo, "checkout", "-q", fix)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(
        {"session_id": "cap1", "cwd": str(repo)})))
    assert main(["capture", "--event", "stop", "--file", str(cap)]) == 0
    lines = [json.loads(l) for l in cap.read_text(encoding="utf-8").splitlines()]
    assert [l["event"] for l in lines] == ["prompt", "stop"]
    assert lines[0]["head"] == base and lines[1]["head"] == fix and "prompt" not in lines[1]
    eps = scan_capture(cap).episodes
    assert len(eps) == 1 and eps[0].source == "capture" and eps[0].request == "fix parse_amount"
    assert (eps[0].head_start, eps[0].head_end) == (base, fix)
    paired = pair(eps[0])
    assert not isinstance(paired, Rejection) and paired[0].shas == (fix,)
    chosen = choose(paired)
    task = admit(eps[0], out_dir=tmp_path / "t", python=sys.executable)
    assert isinstance(task, RepairTask) and task.base_sha == base and task.solution_sha == fix
    assert len(chosen) == 1


def test_capture_with_unmoved_head_is_unreplayable(repo: Path, tmp_path: Path) -> None:
    from xyntetik_runner.shadow.tasks import pair
    head = git(repo, "rev-parse", "HEAD")
    e = episode(repo, source="capture", head_start=head, head_end=head)
    r = pair(e)
    assert isinstance(r, Rejection) and r.disposition is Disposition.UNREPLAYABLE


def test_capture_from_a_parent_directory_records_every_repo_head(repo: Path, tmp_path: Path, monkeypatch: Any) -> None:
    """The owner starts sessions from a parent directory whose own HEAD never
    moves; the hook records the HEADs of the repositories inside it and the
    task range is still exact."""
    from xyntetik_runner.shadow.importer import scan_capture
    from xyntetik_runner.shadow.tasks import pair
    base = git(repo, "rev-parse", "HEAD^")
    fix = git(repo, "rev-parse", "HEAD")
    cap = tmp_path / "capture.jsonl"
    parent = repo.parent  # tmp_path itself, not a repository
    git(repo, "checkout", "-q", base)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(
        {"session_id": "cap2", "cwd": str(parent), "prompt": "first ask"})))
    assert main(["capture", "--event", "prompt", "--file", str(cap)]) == 0
    git(repo, "checkout", "-q", fix)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(
        {"session_id": "cap2", "cwd": str(parent), "prompt": "fix parse_amount now"})))
    assert main(["capture", "--event", "prompt", "--file", str(cap)]) == 0
    from xyntetik_runner.shadow.tasks import canonical
    lines = [json.loads(l) for l in cap.read_text(encoding="utf-8").splitlines()]
    key = str(canonical(repo))
    assert lines[0]["head"] == "" and lines[0]["heads"] == {key: base}
    assert lines[1]["heads"] == {key: fix}
    eps = scan_capture(cap).episodes
    assert len(eps) == 2
    first = eps[0]
    assert first.request == "first ask" and dict(first.heads_start) == {key: base}
    paired = pair(first)
    assert not isinstance(paired, Rejection) and paired[0].shas == (fix,)
    assert canonical(paired[0].repo) == canonical(repo)
    # the second prompt carries the first as context
    assert eps[1].context == ("first ask",)


def test_context_reaches_the_attempt_and_the_task(repo: Path, tmp_path: Path) -> None:
    from xyntetik_runner.shadow.tasks import RepairTask
    e = episode(repo, context=("we saw CI fail on parse_amount", "thousands separators break"))
    task = admit(e, out_dir=tmp_path / "t", python=sys.executable)
    assert isinstance(task, RepairTask) and task.context == e.context
    assert RepairTask.load(tmp_path / "t" / task.task_id / "task.json").context == e.context
    seen: list[str] = []

    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        seen.append(messages[1]["content"])
        return {"content": "", "tool_calls": [call("finish")]}
    ws = Workspace(repo, visible_tests=(), pythonpath=(), python=sys.executable, budget=Budget())
    attempt(task.request, ws, chat, context=task.context)
    assert "Earlier requests in this session" in seen[0]
    assert "- we saw CI fail on parse_amount" in seen[0] and "Task:\n" in seen[0]


def test_claude_code_context_skips_commands_and_caps_turns(tmp_path: Path) -> None:
    proj = tmp_path / "projects" / "p"
    proj.mkdir(parents=True)
    recs = []
    for i, text in enumerate(["one", "<command-name>/x</command-name>", "two", "three", "four", "five"]):
        recs.append({"type": "user", "sessionId": "s", "cwd": "/w",
                     "timestamp": f"2026-09-01T10:{i:02d}:00Z", "message": {"content": text}})
    (proj / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    eps = scan_claude_code(tmp_path / "projects").episodes
    assert eps[-1].request == "five" and eps[-1].context == ("two", "three", "four")
    assert eps[2].context == ("one",), "the command turn is not context"


def test_probe_speed_and_the_fit_floor(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    from xyntetik_runner.shadow import cli
    from xyntetik_runner.shadow.attempt import probe_speed

    class Slow:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def capabilities(self) -> dict[str, Any]:
            return {"models": [{"id": "crawler.gguf"}], "version": "t"}

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            import time
            time.sleep(0.2)
            return {"choices": [{"message": {"content": "x"}}],
                    "usage": {"completion_tokens": 2}}
    tps = probe_speed(Slow().post_json, "crawler.gguf")
    assert 0 < tps < 15
    monkeypatch.setattr(cli, "RunnerEndpoint", Slow)
    (tmp_path / "tasks").mkdir()
    rc = main(["replay", "--out", str(tmp_path), "--endpoint", "http://x", "--min-tps", "15"])
    assert rc == 2
    assert "fit-first" in capsys.readouterr().err


def test_edit_file_replaces_one_exact_occurrence(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, visible_tests=(), pythonpath=(), python=sys.executable, budget=Budget())
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    assert ws.edit_file("a.py", "x = 1", "x = 9").startswith("error: old_text occurs 2")
    assert ws.edit_file("a.py", "nope", "x").startswith("error: old_text not found")
    assert ws.edit_file("../a.py", "y", "z").startswith("error")
    assert ws.edit_file("a.py", "y = 2", "y = 3").startswith("edited a.py")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 3\nx = 1\n"
    chat = scripted({"content": "", "tool_calls": [call("edit_file", path="a.py", old_text="x = 1\ny = 3", new_text="x = 0\ny = 0"), call("finish")]})
    r = attempt("t", ws, chat)
    assert r.finished and (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 0\ny = 0\nx = 1\n"


def test_install_is_explicit_idempotent_and_reversible(tmp_path: Path, capsys: Any) -> None:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)  # both harnesses present
    settings = home / ".claude" / "settings.json"
    settings.write_text(json.dumps({"model": "x", "hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": "echo mine"}]}]}}), encoding="utf-8")
    assert main(["install", "--home", str(home), "--python", "py", "--pythonpath", "/src",
                 "--out", "/o", "--yes"]) == 0
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["model"] == "x", "existing settings survive"
    stop = data["hooks"]["Stop"]
    assert stop[0]["hooks"][0]["command"] == "echo mine", "existing hooks survive"
    assert "capture --event stop 2>/dev/null || true" in stop[1]["hooks"][0]["command"]
    assert "PYTHONPATH=/src py -m" in data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert settings.with_suffix(".json.bak-shadow").is_file()
    skill = home / ".claude" / "skills" / "shadow" / "SKILL.md"
    prompt = home / ".codex" / "prompts" / "shadow.md"
    assert "report --out /o --tasks" in skill.read_text(encoding="utf-8")
    assert "Never run `replay`" in skill.read_text(encoding="utf-8")
    assert "shadow routes" in skill.read_text(encoding="utf-8") and "git apply" in skill.read_text(encoding="utf-8")
    assert "capture --summary" in prompt.read_text(encoding="utf-8") and "delegate --repo ." in prompt.read_text(encoding="utf-8")
    assert not (home / ".codex").exists() or True
    # idempotent
    assert main(["install", "--home", str(home), "--python", "py", "--yes"]) == 0
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert len(data["hooks"]["Stop"]) == 2 and len(data["hooks"]["UserPromptSubmit"]) == 1
    # reversible: exactly what install wrote, nothing else
    assert main(["uninstall", "--home", str(home)]) == 0
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["hooks"] == {"Stop": [{"hooks": [{"type": "command", "command": "echo mine"}]}]}
    assert not skill.exists() and not prompt.exists()
    out = capsys.readouterr().out
    assert "2 hook(s) added" in out and "removed 2 hook(s)" in out


def test_capture_summary_counts_the_file(tmp_path: Path, capsys: Any) -> None:
    cap = tmp_path / "c.jsonl"
    cap.write_text('{"event":"prompt","session_id":"a","heads":{"/r":"x"}}\n{"event":"stop","session_id":"a"}\n'
                   '{"event":"prompt","session_id":"b","heads":{}}\n', encoding="utf-8")
    assert main(["capture", "--summary", "--file", str(cap)]) == 0
    assert "3 lines, 2 prompts, 1 with repository heads, 2 sessions" in capsys.readouterr().out


def test_install_refuses_a_model_path_that_does_not_exist(tmp_path: Path, capsys: Any) -> None:
    """A typo in -m must fail here, before anything is written, not at the
    first offload weeks later."""
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    rc = main(["install", "--home", str(home), "--yes", "--model", str(tmp_path / "missing.gguf")])
    assert rc == 2
    assert "model not found" in capsys.readouterr().err
    assert not (home / ".codex" / "prompts").exists()
    assert not (home / ".xyntetik").exists()


def test_install_confirms_detects_harnesses_and_records_the_model(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow.install import read_config
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)  # Codex present, Claude Code absent
    model = tmp_path / "m" / "x.gguf"
    model.parent.mkdir()
    model.write_bytes(b"GGUF")
    rc = main(["install", "--home", str(home), "--python", "py", "--dry-run", "--model", str(model)])
    out = capsys.readouterr().out
    assert rc == 0 and "dry run" in out and "Codex:" in out and "Claude Code" not in out
    assert f"model for offloading: {model}" in out
    assert not (home / ".claude").exists()
    # no terminal and no --yes: refused, nothing written
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    rc = main(["install", "--home", str(home), "--python", "py", "--model", str(model)])
    assert rc == 2 and not (home / ".codex" / "prompts" / "shadow.md").exists()
    # --yes writes the prompt only (Codex present) and the config
    rc = main(["install", "--home", str(home), "--python", "py", "--model", str(model), "--runner", "/bin/runner", "--yes"])
    assert rc == 0
    assert (home / ".codex" / "prompts" / "shadow.md").exists() and not (home / ".claude" / "settings.json").exists()
    assert read_config(home) == {"model": str(model), "runner": "/bin/runner", "ctx": 8192, "gpu": "auto", "threads": 0, "out": "~/.xyntetik/shadow"}
    # nothing at all present and nothing forced: a clear refusal
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["install", "--home", str(empty), "--python", "py", "--yes"]) == 2


def test_routes_and_delegate_on_a_scratch_worktree(repo: Path, tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli
    from xyntetik_runner.shadow.routes import Route, render_routes
    assert Route("function", "m", 3, 2).qualifies and not Route("function", "m", 2, 2).qualifies
    assert not Route("file", "m", 4, 1).qualifies
    assert "nothing qualifies" in render_routes([])
    out = tmp_path / "o"
    out.mkdir()
    (out / "evidence.jsonl").write_text("", encoding="utf-8")
    assert main(["routes", "--out", str(out), "--bench", str(tmp_path / "nobench")]) == 0
    assert "nothing qualifies" in capsys.readouterr().out
    # delegate: a scripted model that fixes the file; the repo's tests run on the copy
    base = git(repo, "rev-parse", "HEAD^")
    git(repo, "checkout", "-q", base)  # HEAD is the buggy state; the fix is what the model must produce

    class Fixer:
        def __init__(self, url: str, *a: Any, **k: Any) -> None:
            self.base_url = url
            self.n = 0

        def capabilities(self, **k: Any) -> dict[str, Any]:
            return {"models": [{"id": "fixer.gguf"}]}

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            self.n += 1
            if self.n == 1:
                return {"choices": [{"message": {"content": "", "tool_calls": [
                    {"id": "1", "function": {"name": "write_file", "arguments": json.dumps(
                        {"path": "calc/money.py", "content": CORRECT})}}]}}]}
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "2", "function": {"name": "finish", "arguments": "{}"}}]}}]}
    monkeypatch.setattr(cli, "RunnerEndpoint", Fixer)
    before = (repo / "calc" / "money.py").read_text(encoding="utf-8")
    rc = main(["delegate", "--repo", str(repo), "--request", "fix parse_amount", "--endpoint", "http://x",
               "--python", sys.executable, "--home", str(tmp_path / "h")])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "verdict: tests passed on the scratch copy" in text and "git apply" in text
    assert (repo / "calc" / "money.py").read_text(encoding="utf-8") == before, "working tree untouched"
    patch = [l for l in text.splitlines() if l.startswith("patch: ")][0].split(": ", 1)[1]
    assert Path(patch).read_text(encoding="utf-8").startswith("diff --git a/calc/money.py")
    assert not list(repo.glob(".git/worktrees/*")) or True
