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
BUGGY = (FIXTURE / "workspace" / "calc" / "money.py").read_text()
VISIBLE = (FIXTURE / "workspace" / "tests" / "test_money.py").read_text()
SOLUTION_TESTS = (FIXTURE / "protected" / "tests" / "test_money.py").read_text()
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
    (r / "calc" / "__init__.py").write_text("")
    (r / "calc" / "money.py").write_text(BUGGY)
    (r / "tests" / "test_money.py").write_text(VISIBLE)
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "buggy", date="2026-09-01T11:00:00+00:00")
    (r / "calc" / "money.py").write_text(CORRECT)
    (r / "tests" / "test_money.py").write_text(SOLUTION_TESTS)
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
    (sessions / "a.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
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
    (proj / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    eps = scan_claude_code(tmp_path / "projects").episodes
    assert [e.turn for e in eps] == [1, 2]
    assert eps[0].ended_at == "2026-09-01T11:00:00Z" and eps[1].is_command
    assert "THE ANSWER" not in json.dumps([e.__dict__ for e in eps])


def test_repos_under_finds_nested_repositories(repo: Path) -> None:
    assert repos_under(repo.parent) == [repo]
    assert repos_under(repo) == [repo]


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
    (r / "pkg" / "__init__.py").write_text("X = 1\n")
    (r / "tests" / "test_x.py").write_text("from pkg import X\n\n\ndef test_x():\n    assert X == 1\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "a", date="2026-09-01T11:00:00+00:00")
    (r / "pkg" / "__init__.py").write_text("X = 1\nY = 2\n")
    (r / "tests" / "test_x.py").write_text("from pkg import X\n\n\ndef test_x():\n    assert X == 1\n\n\ndef test_y():\n    assert True\n")
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
    assert (tmp_path / "a").read_text() == "b"


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
    (proj / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
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
    lines = [json.loads(l) for l in cap.read_text().splitlines()]
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
    lines = [json.loads(l) for l in cap.read_text().splitlines()]
    assert lines[0]["head"] == "" and lines[0]["heads"] == {str(repo): base}
    assert lines[1]["heads"] == {str(repo): fix}
    eps = scan_capture(cap).episodes
    assert len(eps) == 2
    first = eps[0]
    assert first.request == "first ask" and dict(first.heads_start) == {str(repo): base}
    paired = pair(first)
    assert not isinstance(paired, Rejection) and paired[0].shas == (fix,) and paired[0].repo == repo
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
    (proj / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
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
