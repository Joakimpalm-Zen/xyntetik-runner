"""Process-level stop latency and discovery cleanup contracts.

Shutdown joins the reaper thread, and the reaper slept its whole poll (5 s
shipped) in one call: every SIGTERM, Ctrl-C and test teardown waited for
that sleep to end. The conformance harness hid it behind RUNNER_TTL_POLL_S=
0.25; thirty root tests paid the full five seconds each (2026-10-13). The
reaper now sleeps in slices and checks for the stop between them.
"""
import os
import json
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Cleanup is eventual, not a service deadline. This only bounds a hung test
# on slow storage; process startup and exit keep their own measured bounds.
SWEEP_CLEANUP_TIMEOUT_S = 120

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal contract")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if os.name == "nt" else "runner")
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


@posix_only
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
@posix_only
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
    env = dict(os.environ, HOME=str(home), APPDATA=str(home),
               RUNNER_TEST_INSTANCE_SWEEP_DELAY_MS="5000")
    records = home / ".xyntetik" / "runner" / "instances"
    spec = f"a={model},b={model}" if registry else str(model)

    def server():
        return RunnerServer(runner_bin, spec, ctx=128, env=env,
                            extra_args=["--gpu", "off", "-t", "2"])

    with server() as first:
        pid = first.proc.pid
        record = records / f"{pid}.json"
        assert record.is_file(), "a live server publishes its instance record"
        first.proc.send_signal(signal.SIGTERM)
        assert first.proc.wait(timeout=10) == 0
        # Check before __exit__ disposes of the harness log: that filesystem
        # work can outlast the helper's deliberately delayed sweep.
        assert record.is_file(), "clean exit must not unlink the instance record"
    assert first.exit_code == 0 and not first.killed

    if sweep == "list":
        result = subprocess.run([helper, "list"], env=env, check=True,
                                capture_output=True, text=True, timeout=10)
        assert str(pid) not in result.stdout.splitlines()
        deadline = time.monotonic() + SWEEP_CLEANUP_TIMEOUT_S
        while record.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not record.exists(), "the public list sweeps the dead owner"
    else:
        # Do not list first: registration alone must remove the old file,
        # including on hosts that never run a tray or discovery controller.
        with server() as second:
            deadline = time.monotonic() + SWEEP_CLEANUP_TIMEOUT_S
            while record.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not record.exists(), "the next server eventually sweeps the old file"
            assert (records / f"{second.proc.pid}.json").is_file()


def test_blocked_sweep_does_not_hold_start_or_exit(runner_bin, model, tmp_path):
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from harness import RunnerServer

    home = tmp_path / "home"
    records = home / ("xyntetik" if os.name == "nt" else ".xyntetik") / "runner" / "instances"
    records.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    assert dead.wait(timeout=5) == 0  # independent proof that this PID is dead
    stale = records / f"{dead.pid}.json"
    stale.write_text(json.dumps({"pid": dead.pid, "mode": "serve"}))
    temporary = records / f"{dead.pid}.json.tmp"
    temporary.touch()
    live = records / f"{os.getpid()}.json"
    live.write_text(json.dumps({"pid": os.getpid(), "mode": "cli"}))
    marker = tmp_path / "sweeper.pid"
    env = dict(os.environ, HOME=str(home), APPDATA=str(home),
               RUNNER_TEST_INSTANCE_SWEEP_DELAY_MS="5000",
               RUNNER_TEST_INSTANCE_SWEEP_MARKER=str(marker))
    worker = None
    first = RunnerServer(runner_bin, str(model), ctx=128, env=env, start_timeout=3,
                         extra_args=["--gpu", "off", "-t", "2"])
    try:
        first.start()
        deadline = time.monotonic() + SWEEP_CLEANUP_TIMEOUT_S
        while worker is None and time.monotonic() < deadline:
            try:
                worker = int(marker.read_text())
            except (FileNotFoundError, ValueError):
                time.sleep(0.02)
        assert worker is not None, "cleanup helper did not start"
        assert worker != first.proc.pid, "cleanup must run in a separate process"
        assert stale.exists(), "cleanup is still blocked after the server became healthy"
        t0 = time.monotonic()
        first.proc.send_signal(signal.SIGTERM)
        rc = first.proc.wait(timeout=2)
        assert time.monotonic() - t0 < 2, "server exit waited for its cleanup helper"
        if os.name != "nt":
            assert rc == 0, rc
        # RSS sampling, log close/unlink and other harness bookkeeping run in
        # finally, outside the measurement and the blocked-worker window.

        # A second start does not wait for the worker lock or run a duplicate sweep.
        second = RunnerServer(runner_bin, str(model), ctx=128, env=env, start_timeout=3,
                              extra_args=["--gpu", "off", "-t", "2"])
        with second:
            assert int(marker.read_text()) == worker
        # No discovery list runs: the helper alone must clear the readerless backlog,
        # including records left by both servers after they stopped.
        deadline = time.monotonic() + SWEEP_CLEANUP_TIMEOUT_S
        while time.monotonic() < deadline and (stale.exists() or temporary.exists() or
                (records / f"{first.proc.pid}.json").exists() or
                (records / f"{second.proc.pid}.json").exists()):
            time.sleep(0.02)
        assert sorted(p.name for p in records.glob("*.json")) == [live.name]
        assert not temporary.exists()
        worker = None  # successful cleanup needs no signal to an old helper PID
    finally:
        first.stop()
        if worker is not None:
            try:
                os.kill(worker, signal.SIGTERM)
            except OSError:
                pass
