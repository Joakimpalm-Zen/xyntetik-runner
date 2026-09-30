"""The C startup lease against the Python class it ports (R4.12.24).

The tray takes the lease in C (src/instances.c), the Python client's
ManagedRunner and the Suite take it through lease.py and its Rust port. Each
must judge the others' records exactly as its own: the same start identity for
the same process, a held lease refused, a dead owner's record reclaimed. The C
side is driven through the test-instances binary's helper modes.
"""

import os
import pathlib
import signal
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from xyntetik_runner.lease import StartupLease, _process_start_time, lease_holder  # noqa: E402

HELPER = ROOT / ("test-instances.exe" if os.name == "nt" else "test-instances")

pytestmark = pytest.mark.skipif(not HELPER.exists(),
                                reason="test-instances not built (make test builds it)")


def _hold(path):
    """Start the C side holding the lease at `path`; returns (process, first line)."""
    proc = subprocess.Popen([str(HELPER), "lease-hold", str(path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return proc, proc.stdout.readline().strip()


def _done(proc):
    if proc.stdin and not proc.stdin.closed:
        proc.stdin.close()
    assert proc.wait(timeout=30) == 0


def test_both_sides_read_the_same_start_identity():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for pid in (os.getpid(), child.pid):
            c = subprocess.run([str(HELPER), "lease-identity", str(pid)],
                               capture_output=True, text=True, timeout=30)
            assert c.returncode == 0, c.stderr
            assert c.stdout.strip() == _process_start_time(pid), pid
    finally:
        child.kill()
        child.wait()


def test_a_lease_the_c_side_holds_refuses_python(tmp_path):
    path = tmp_path / "runner-8123.pid"
    proc, first = _hold(path)
    try:
        assert first == "held"
        assert lease_holder(path) == proc.pid
        assert not StartupLease(path).acquire()
    finally:
        _done(proc)
    # released on its side: the record is gone and Python claims it
    lease = StartupLease(path)
    assert lease.acquire()
    lease.release()


def test_a_lease_python_holds_refuses_the_c_side(tmp_path):
    path = tmp_path / "runner-8123.pid"
    lease = StartupLease(path)
    assert lease.acquire()
    proc, first = _hold(path)
    _done(proc)
    assert first == f"refused {os.getpid()}"
    lease.release()
    proc, first = _hold(path)
    try:
        assert first == "held"
    finally:
        _done(proc)


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL")
def test_a_dead_c_owner_is_reclaimed_by_python(tmp_path):
    path = tmp_path / "runner-8123.pid"
    proc, first = _hold(path)
    assert first == "held"
    proc.send_signal(signal.SIGKILL)   # dies holding it, as a crashed tray would
    proc.wait()
    assert lease_holder(path) == proc.pid
    lease = StartupLease(path)
    assert lease.acquire()
    lease.release()


def test_a_dead_python_owner_is_reclaimed_by_the_c_side(tmp_path):
    path = tmp_path / "runner-8123.pid"
    code = ("import sys; sys.path.insert(0, sys.argv[2]);"
            "from pathlib import Path; from xyntetik_runner.lease import StartupLease;"
            "assert StartupLease(Path(sys.argv[1])).acquire()")
    subprocess.run([sys.executable, "-c", code, str(path), str(ROOT / "python" / "src")],
                   check=True, timeout=60)
    assert lease_holder(path) is not None   # it exited without releasing
    proc, first = _hold(path)
    try:
        assert first == "held"
    finally:
        _done(proc)
