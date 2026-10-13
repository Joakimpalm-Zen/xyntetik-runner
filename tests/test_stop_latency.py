"""A stop never waits for the TTL reaper's poll.

Shutdown joins the reaper thread, and the reaper slept its whole poll (5 s
shipped) in one call: every SIGTERM, Ctrl-C and test teardown waited for
that sleep to end. The conformance harness hid it behind RUNNER_TTL_POLL_S=
0.25; thirty root tests paid the full five seconds each (2026-10-13). The
reaper now sleeps in slices and checks for the stop between them.
"""
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal contract")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / "runner"
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_sigterm_does_not_wait_for_the_reaper_poll(runner_bin, model):
    env = dict(os.environ)
    env.pop("RUNNER_TTL_POLL_S", None)   # the shipped 5 s cadence, deliberately
    port = _free_port()
    proc = subprocess.Popen([str(runner_bin), "-m", str(model), "--serve", "--no-tray",
                             "--port", str(port), "--gpu", "off"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("server never answered /health")
        time.sleep(0.3)   # well inside the reaper's first 5 s sleep
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=10)
        took = time.monotonic() - t0
        assert rc == 0, rc
        assert took < 2.0, f"stop took {took:.2f}s: it waited for the reaper poll"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
