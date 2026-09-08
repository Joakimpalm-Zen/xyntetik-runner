"""`shadow sync` closes the loop the hooks feed (import, count what waits,
replay when told to), and the warm runner is reused across commands. No
model: the served endpoint is scripted; the verifier and the worktrees are
real."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from xyntetik_runner.shadow import server
from xyntetik_runner.shadow.cli import main
from xyntetik_runner.shadow.install import read_config, write_config

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


class FakeManaged:
    started: list[Any] = []

    def __init__(self, launch: Any, **k: Any) -> None:
        self.launch = launch
        self.base_url = f"http://127.0.0.1:{launch.port}"
        self.process: Any = type("P", (), {"pid": 4242})()

    def start(self, **k: Any) -> bool:
        FakeManaged.started.append(self.launch)
        return True

    def stop(self, **k: Any) -> None:
        pass


class Fixer:
    """Answers the fit probe, then fixes the file in two tool calls."""
    live_pids: dict[str, int] = {}

    def __init__(self, url: str, **k: Any) -> None:
        self.base_url = url
        self.n = 0

    def capabilities(self, **k: Any) -> dict[str, Any]:
        return {"object": "runner.capabilities", "models": [{"id": "coder.gguf"}],
                "pid": Fixer.live_pids.get(self.base_url, 4242), "version": "t"}

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("temperature") == 0 and not payload.get("tools"):
            return {"choices": [{"message": {"content": "one, two"}}], "usage": {"completion_tokens": 64}}
        self.n += 1
        if self.n == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "write_file", "arguments": json.dumps(
                    {"path": "calc/money.py", "content": CORRECT})}}]}}]}
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "2", "function": {"name": "finish", "arguments": "{}"}}]}}]}


def capture(repo: Path, cap: Path, monkeypatch: Any, base: str, fix: str) -> None:
    git(repo, "checkout", "-q", base)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": "cap1", "cwd": str(repo), "prompt": "fix parse_amount"})))
    assert main(["capture", "--event", "prompt", "--file", str(cap)]) == 0
    git(repo, "checkout", "-q", fix)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": "cap1", "cwd": str(repo)})))
    assert main(["capture", "--event", "stop", "--file", str(cap)]) == 0


def test_sync_imports_counts_and_replays_when_told(repo: Path, tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    from xyntetik_runner.shadow import cli
    home = tmp_path / "home"
    out = home / ".xyntetik" / "shadow"
    cap = home / ".xyntetik" / "shadow" / "capture.jsonl"
    cap.parent.mkdir(parents=True)
    base = git(repo, "rev-parse", "HEAD^")
    fix = git(repo, "rev-parse", "HEAD")
    capture(repo, cap, monkeypatch, base, fix)
    model = tmp_path / "coder.gguf"
    model.write_bytes(b"GGUF")
    write_config(home, model=str(model), runner="fake-runner", ctx=4096, gpu="auto", threads=0, out=str(out))
    monkeypatch.setattr(cli, "ManagedRunner", FakeManaged)
    monkeypatch.setattr(cli, "RunnerEndpoint", Fixer)
    FakeManaged.started.clear()
    common = ["sync", "--out", str(out), "--home", str(home), "--python", sys.executable]
    # import only: one task admitted, one waiting, nothing served
    assert main(common) == 0
    text = capsys.readouterr().out
    assert "sync: 1 task(s) admitted now; 1 waiting for a replay with coder.gguf" in text, text
    assert "sync --replay 1" in text and not FakeManaged.started
    assert server.read_state(home) is None
    # replay when told: the runner is started warm, the attempt is verified, the state is recorded
    assert main(common + ["--replay", "1"]) == 0
    text = capsys.readouterr().out
    assert "started the runner" in text and "verified_local_attempt" in text, text
    assert "sync: 1 replayed, 0 still waiting" in text
    assert "| repair |" in text or "qualif" in text
    assert len(FakeManaged.started) == 1
    launch = FakeManaged.started[0]
    assert launch.parent_pid is None and launch.ttl == server.DEFAULT_TTL and launch.extra_args == ()
    state = server.read_state(home)
    assert state is not None and state.pid == 4242 and state.model == str(model)
    # the next sync has nothing new and nothing waiting; the warm runner is reused, not restarted
    Fixer.live_pids[state.base_url] = 4242
    assert main(common + ["--replay", "1"]) == 0
    text = capsys.readouterr().out
    assert "0 waiting" in text and len(FakeManaged.started) == 1, text
    # delegate reuses it too, with the same state
    git(repo, "checkout", "-q", base)
    assert main(["delegate", "--repo", str(repo), "--request", "fix parse_amount", "--home", str(home),
                 "--python", sys.executable]) == 0
    text = capsys.readouterr().out
    assert "reusing the runner" in text and "git apply" in text, text
    assert len(FakeManaged.started) == 1
    # server status and stop (status asks the port itself)
    monkeypatch.setattr(server, "RunnerEndpoint", Fixer)
    assert main(["server", "--home", str(home)]) == 0
    assert "alive" in capsys.readouterr().out
    monkeypatch.setattr(server, "alive", lambda *a, **k: False)  # nothing real to kill here
    assert main(["server", "--home", str(home), "--stop"]) == 0
    assert server.read_state(home) is None


def test_replay_order_spends_the_budget_on_function_tasks_first() -> None:
    from dataclasses import replace as _replace
    from xyntetik_runner.shadow.cli import replay_order
    from xyntetik_runner.shadow.tasks import RepairTask
    base = RepairTask(task_id="t", episode_id="e", repo="r", base_sha="a", solution_sha="b", request="q",
                      request_sha256="0" * 64, context=(), test_files=(), visible_test_files=(), src_files=1,
                      pythonpath=(), protected_dir="p", expected_tests=3, baseline_failing=1)
    tasks = [_replace(base, task_id="multi", task_class="multi-file", src_files=3),
             _replace(base, task_id="file-hard", task_class="file", baseline_failing=5),
             _replace(base, task_id="fn-b", task_class="function", baseline_failing=2),
             _replace(base, task_id="file-easy", task_class="file", baseline_failing=1),
             _replace(base, task_id="fn-a", task_class="function", baseline_failing=1)]
    assert [t.task_id for t in replay_order(tasks)] == ["fn-a", "fn-b", "file-easy", "file-hard", "multi"]


def test_warm_runner_is_replaced_when_stale_or_serving_another_model(tmp_path: Path, monkeypatch: Any) -> None:
    home = tmp_path / "home"
    cfg = {"model": str(tmp_path / "coder.gguf"), "runner": "r", "ctx": 4096, "gpu": "auto", "threads": 0}
    FakeManaged.started.clear()
    stopped: list[Path] = []
    monkeypatch.setattr(server, "stop_server", lambda h: stopped.append(h) or server.clear_state(h) or True)
    ep, started = server.served_runner(home, cfg, managed_factory=FakeManaged, endpoint_factory=Fixer)
    assert started and len(FakeManaged.started) == 1
    state = server.read_state(home)
    assert state is not None
    # a recorded runner that does not answer with its pid is stale: replaced
    Fixer.live_pids[state.base_url] = 1
    ep, started = server.served_runner(home, cfg, managed_factory=FakeManaged, endpoint_factory=Fixer)
    assert started and len(FakeManaged.started) == 2 and stopped == [home]
    state2 = server.read_state(home)
    assert state2 is not None
    Fixer.live_pids[state2.base_url] = 4242
    # alive and same model: reused
    ep, started = server.served_runner(home, cfg, managed_factory=FakeManaged, endpoint_factory=Fixer)
    assert not started and len(FakeManaged.started) == 2
    # a promoted adapter changes what must be served: replaced with --lora
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"A")
    ep, started = server.served_runner(home, {**cfg, "adapter": str(adapter)}, managed_factory=FakeManaged,
                                       endpoint_factory=Fixer)
    assert started and FakeManaged.started[-1].extra_args == ("--lora", str(adapter))
    assert "no warm runner" not in server.render_status(home)
    with pytest.raises(RuntimeError):
        server.served_runner(home, {}, managed_factory=FakeManaged, endpoint_factory=Fixer)
    assert read_config(home) == {}
