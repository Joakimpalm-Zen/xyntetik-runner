"""Server slots sharing one thread pool (2026-10-08).

Every slot of `--parallel N` runs its forwards on one pool (tpool_run
serializes them a matvec at a time). An outside review asked for the slot
counts, lone-versus-concurrent and shutdown-under-load cases to be pinned:

  * the banner reports the shared pool and the slot count;
  * concurrent greedy requests answer exactly what a lone request answers,
    at 1, 2 and 4 slots (the pool is shared state; a mixed-up work item
    would show as a different text);
  * stopping the server with requests in flight ends the process promptly
    and without a crash signal (the refcounted pool tears down once).
"""

import concurrent.futures
import contextlib
import json
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
CRASH = {-signal.SIGSEGV, -signal.SIGABRT} | ({-signal.SIGBUS} if hasattr(signal, "SIGBUS") else set())


def _free_port():
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    if not RUNNER.exists():
        pytest.skip("runner binary not built")
    out = tmp_path_factory.mktemp("pool") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", out],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return out


def _start(model, parallel):
    port = _free_port()
    proc = subprocess.Popen(
        [RUNNER, "-m", model, "--serve", "--port", str(port), "--gpu", "off",
         "--no-tray", "-t", "4", "--parallel", str(parallel), "-c", "256"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError("server exited before its health check: "
                                 + proc.stderr.read().decode(errors="replace"))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return proc, port
        except (urllib.error.URLError, OSError):
            time.sleep(0.05)
    proc.kill()
    raise AssertionError("server did not become healthy")


def _complete(port, prompt, n):
    body = json.dumps({"prompt": prompt, "max_tokens": n, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["text"]


def _stop(proc):
    proc.terminate()
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        raise AssertionError("server did not stop within 30 s of SIGTERM")


@pytest.mark.parametrize("parallel", (1, 2, 4))
def test_concurrent_requests_answer_like_a_lone_one(model, parallel):
    proc, port = _start(model, parallel)
    try:
        solo = _complete(port, "the quick brown fox", 24)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2 * parallel) as ex:
            got = list(ex.map(lambda _: _complete(port, "the quick brown fox", 24),
                              range(2 * parallel)))
    finally:
        rc = _stop(proc)
    banner = proc.stderr.read().decode(errors="replace")
    m = re.search(r"(\d+) slots? sharing (\d+) threads", banner)
    assert m and int(m.group(1)) == parallel, banner[-2000:]
    assert all(g == solo for g in got), (solo, got)
    assert rc not in CRASH, f"server exited with {rc}"


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM delivery differs on Windows")
def test_shutdown_with_requests_in_flight(model):
    proc, port = _start(model, 2)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futs = [ex.submit(_complete, port, "a long answer about anything", 200) for _ in range(4)]
    time.sleep(0.5)                       # the slots are busy, two requests queued
    t0 = time.time()
    rc = _stop(proc)
    took = time.time() - t0
    for f in futs:
        with contextlib.suppress(Exception):
            f.result(timeout=5)
    ex.shutdown(wait=False)
    assert rc not in CRASH, f"server crashed on shutdown under load: {rc}"
    assert took < 30, f"shutdown took {took:.1f} s"
