"""Two commands that find no warm runner at the same moment start ONE.

Startup was unsynchronised and the state file's temporary name was shared:
two simultaneous starts produced two live runners, one FileNotFoundError from
the rename race, and a record pointing at a process the successful caller was
not given (external review, 2026-09-27). Under the lock the second caller
waits through the first's start and then reuses what it recorded.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from xyntetik_runner.shadow import server


class SlowManaged:
    started: list[Any] = []
    next_pid = 100

    def __init__(self, launch: Any, **k: Any) -> None:
        self.launch = launch
        self.base_url = f"http://127.0.0.1:{launch.port}"
        SlowManaged.next_pid += 1
        self.process: Any = type("P", (), {"pid": SlowManaged.next_pid})()

    def start(self, **k: Any) -> bool:
        SlowManaged.started.append(self.launch)
        time.sleep(0.3)   # a model load
        return True


class Live:
    """Every recorded port answers with the pid that was recorded for it."""
    pids: dict[str, int] = {}

    def __init__(self, url: str, **k: Any) -> None:
        self.base_url = url

    def capabilities(self, **k: Any) -> dict[str, Any]:
        return {"models": [{"id": "coder.gguf"}], "pid": Live.pids.get(self.base_url, 0)}


def test_concurrent_starts_share_one_runner(tmp_path: Path, monkeypatch: Any) -> None:
    home = tmp_path / "home"
    cfg = {"model": str(tmp_path / "coder.gguf"), "runner": "r", "ctx": 4096, "gpu": "auto", "threads": 0}
    SlowManaged.started.clear()
    Live.pids.clear()
    orig_write = server.write_state

    def write_and_answer(h: Path, st: server.ServerState) -> Path:
        Live.pids[st.base_url] = st.pid
        return orig_write(h, st)

    monkeypatch.setattr(server, "write_state", write_and_answer)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []

    def go() -> None:
        try:
            ep, started = server.served_runner(home, cfg, managed_factory=SlowManaged,
                                               endpoint_factory=Live)
            results.append((ep.base_url, started))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert len(SlowManaged.started) == 1, "more than one runner was started"
    assert sum(1 for _, s in results if s) == 1
    assert len({url for url, _ in results}) == 1
    state = server.read_state(home)
    assert state is not None and state.base_url == results[0][0]
    assert Live.pids[state.base_url] == state.pid
    # no temporary file was left behind
    assert not [p for p in (home / ".xyntetik" / "shadow").iterdir() if p.suffix == ".tmp"]
