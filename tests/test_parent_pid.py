"""--parent-pid ties the Runner to a PROCESS, never to a thread (R4.12.25).

On Linux the flag used to arm PR_SET_PDEATHSIG, which the kernel fires when
the THREAD that forked the Runner exits. A supervisor that launched from a
short-lived thread (a Rust worker, Python's subprocess from a thread pool) had
its Runner killed seconds after it was ready; the Suite hit exactly that. It
also fired on the direct parent when the flag named a grandparent. The watch is
now on the named process itself: a pidfd on Linux, the 2 s poll elsewhere.
"""

import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import find_runner, free_port  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="the POSIX watch; Windows waits on a process handle")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("parent") / "plain.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _launch(model, parent):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    port = free_port()
    proc = subprocess.Popen([exe, "-m", str(model), "--serve", "--no-tray",
                             "--port", str(port), "--gpu", "off", "-t", "2",
                             "--parent-pid", str(parent)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return proc, port


def _ready(proc, port, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_the_launching_thread_exiting_does_not_stop_the_runner(model):
    """The Suite's case: launched from a thread that returns once the Runner is
    ready, while the supervising process lives on."""
    started = {}

    def launch():
        proc, port = _launch(model, os.getpid())
        started["proc"], started["port"] = proc, port
        started["ready"] = _ready(proc, port)

    t = threading.Thread(target=launch)
    t.start()
    t.join()
    proc = started["proc"]
    try:
        assert started["ready"], proc.stderr.read() if proc.poll() is not None else ""
        # the death signal fired within milliseconds of the thread's exit;
        # a poll interval and some margin later the Runner must still answer
        time.sleep(3)
        assert proc.poll() is None, proc.stderr.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{started['port']}/health",
                                    timeout=5) as r:
            assert r.status == 200
    finally:
        _stop(proc)


def test_the_runner_exits_when_the_named_process_dies(model):
    """The flag's contract: a supervisor that is SIGKILLed leaves no Runner."""
    sup = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    proc, port = _launch(model, sup.pid)
    try:
        assert _ready(proc, port), proc.stderr.read()
        sup.kill()
        sup.wait()
        proc.wait(timeout=10)
        assert b"exited" in proc.stderr.read()
    finally:
        _stop(proc)
        if sup.poll() is None:
            sup.kill()
            sup.wait()


def test_a_process_that_is_already_gone_is_not_watched_forever(model):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    proc, _ = _launch(model, gone.pid)
    try:
        # at once on Linux (the pidfd open refuses a dead pid), within one
        # poll interval elsewhere
        assert proc.wait(timeout=10) == 0
    finally:
        _stop(proc)
