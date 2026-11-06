"""Process-level stop latency and discovery cleanup contracts.

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


@pytest.mark.parametrize("registry", [False, True], ids=["single-model", "swap-set"])
@pytest.mark.parametrize("sweep", ["list", "next-server"])
def test_clean_stop_leaves_record_for_discovery_to_sweep(runner_bin, model, tmp_path,
                                                         registry, sweep):
    # Process death, established by wait(), is the independent liveness
    # anchor. Exit must leave the record alone: unlink can block after the
    # server's shutdown watchdog has released its bound (R10.20).
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from harness import RunnerServer

    helper = ROOT / "test-instances"
    assert helper.is_file(), "build test-instances with make"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), APPDATA=str(home))
    records = home / ".xyntetik" / "runner" / "instances"
    spec = f"a={model},b={model}" if registry else str(model)

    def server():
        return RunnerServer(runner_bin, spec, ctx=128, env=env,
                            extra_args=["--gpu", "off", "-t", "2"])

    with server() as first:
        pid = first.proc.pid
        record = records / f"{pid}.json"
        assert record.is_file(), "a live server publishes its instance record"
    assert first.exit_code == 0 and not first.killed
    assert record.is_file(), "clean exit must not unlink the instance record"

    if sweep == "list":
        result = subprocess.run([helper, "list"], env=env, check=True,
                                capture_output=True, text=True, timeout=10)
        assert str(pid) not in result.stdout.splitlines()
        assert not record.exists(), "the public list sweeps the dead owner"
    else:
        # Do not list first: registration alone must remove the old file,
        # including on hosts that never run a tray or discovery controller.
        with server() as second:
            assert not record.exists(), "the next server sweeps the old file"
            assert (records / f"{second.proc.pid}.json").is_file()
