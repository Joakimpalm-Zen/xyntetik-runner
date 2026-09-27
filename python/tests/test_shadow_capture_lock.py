"""Concurrent captures keep every thread and job record.

The thread and job ledgers are read-modify-write files shared by every
session that captures. Without a lock, two sessions linking at the same
moment dropped each other's record and raced a shared temporary name
(sweep, 2026-09-27).
"""
from __future__ import annotations

import threading
from pathlib import Path

from xyntetik_runner.shadow import capture as C


def test_concurrent_links_keep_every_record(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    n = 12
    errors: list[BaseException] = []

    def go(i: int) -> None:
        try:
            sid = f"s{i}"
            C.link(home, session_id=sid, event_id=f"e{i}", prov=C.classify("do the thing"))
            C.link(home, session_id=sid, event_id=f"n{i}",
                   prov=C.classify(f"<task-notification>\n<task-id>job{i}</task-id>\n"),
                   job_ids=[f"job{i}"])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    threads_ledger = C._read_threads(home)
    jobs_ledger = C._read_jobs(home)
    assert sorted(threads_ledger) == sorted(f"s{i}" for i in range(n))
    assert sorted(jobs_ledger) == sorted(f"job{i}" for i in range(n))
    for i in range(n):
        assert threads_ledger[f"s{i}"]["task_id"] == f"e{i}"
        assert jobs_ledger[f"job{i}"]["first_seen_task_id"] == f"e{i}"
    leftovers = [p.name for p in (home / ".xyntetik" / "shadow").iterdir() if p.suffix == ".tmp"]
    assert not leftovers, leftovers
