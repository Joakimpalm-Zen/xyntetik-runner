"""Listing pinned contexts must stay valid while other slots create them.

The expected names come from the requests, independently of the list response.
The sanitized conformance job makes an out-of-bounds read fail even when the
ordinary build happens to read plausible bytes beyond the allocation.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import urllib.request

from harness import RunnerServer, find_runner


def test_list_contexts_during_creation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    model = os.environ.get("RUNNER_TEST_MODEL", str(root / "test.gguf"))
    env = dict(os.environ, RUNNER_PREFIX_CACHE_MB="16")
    with RunnerServer(find_runner(root), model, ctx=128, parallel=4, env=env,
                      extra_args=["--gpu", "off", "-t", "1"],
                      log_path=tmp_path / "contexts.log") as srv:
        def request(payload=None):
            req = urllib.request.Request(
                srv.base_url + "/v1/runner/contexts",
                data=None if payload is None else json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as response:
                assert response.status == 200
                return json.load(response)

        assert request()["data"] == []
        pin = request({"id": "initial", "prompt": "hi"})
        writers, per_writer = 4, 400
        expected = {"initial"} | {
            f"ctx-{w}-{i}" for w in range(writers) for i in range(per_writer)}
        start = threading.Barrier(writers + 2)
        stop = threading.Event()

        def create(writer):
            start.wait(timeout=15)
            for i in range(per_writer):
                if stop.is_set():
                    return
                name = f"ctx-{writer}-{i}"
                assert request({"id": name, "prompt": "hi"})["id"] == name

        def listing():
            start.wait(timeout=15)
            for _ in range(per_writer):
                if stop.is_set():
                    return
                entries = request()["data"]
                names = [entry["id"] for entry in entries]
                assert len(names) == len(set(names))
                assert "initial" in names
                assert set(names) <= expected
                assert all(entry["tokens"] == pin["tokens"] and
                           entry["bytes"] == pin["bytes"] for entry in entries)

        def guarded(fn, *args):
            try:
                fn(*args)
            except BaseException:
                stop.set()
                raise

        with ThreadPoolExecutor(max_workers=writers + 2) as pool:
            futures = [pool.submit(guarded, create, w) for w in range(writers)]
            futures += [pool.submit(guarded, listing) for _ in range(2)]
            for future in futures:
                future.result()

        assert {entry["id"] for entry in request()["data"]} == expected
        srv.assert_alive()
