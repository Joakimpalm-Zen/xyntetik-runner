"""A disconnect during a queued model load (R10.12.3.1).

A load waiting in the --wait-for-vram queue holds a request nobody may want any
more. When its client has gone the wait stops, unless a live client is queued
behind it: stopping would cost that client its place in the VRAM queue.

The wait runs on CUDA hosts only, so RUNNER_TEST_VRAM_FREE_MB stands in for the
device and the real vram_claim runs on any host, against a real ledger with a
made-up holder in it. How a queued socket is judged is the point of several
cases here: only one that closed having sent nothing is gone. Its request is
unread, so a close behind it cannot be seen, and that counts as live.
"""
import contextlib
import json
import os
import pathlib
import socket
import sys
import time
import urllib.request

import pytest

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                       ctypes.POINTER(_Overlapped)]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL
else:
    import fcntl


@contextlib.contextmanager
def _ledger_lock(f, exclusive):
    """The lock the runner takes on the ledger (plat_file_rmw), shared to read and exclusive to write.

    R6.48: on Windows a byte-range lock is mandatory, so a plain read that lands while the runner holds
    LockFileEx over the whole file fails with ERROR_LOCK_VIOLATION, which Python raises as PermissionError
    (Errno 13). The queued load re-claims the ledger while it waits and the test polls it every 20 ms, so
    now and then the two met. Taking the same lock waits for the runner's read-modify-write instead. On
    POSIX the runner's flock is advisory and a plain read never failed; the lock there keeps a test write
    from landing inside the runner's read-modify-write and being lost.
    """
    if os.name == "nt":
        handle = msvcrt.get_osfhandle(f.fileno())
        if not _kernel32.LockFileEx(handle, 2 if exclusive else 0, 0, 0xFFFFFFFF, 0xFFFFFFFF,
                                    ctypes.byref(_Overlapped())):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            _kernel32.UnlockFileEx(handle, 0, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(_Overlapped()))
    else:
        fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_ledger(path):
    with open(path, "rb") as f, _ledger_lock(f, exclusive=False):
        return f.read().decode("utf-8")


def _edit_ledger(path, edit):
    """Rewrites the ledger as edit(old text) returns it, under the exclusive lock, in one step."""
    with open(os.open(path, os.O_RDWR | os.O_CREAT, 0o600), "r+b") as f, _ledger_lock(f, exclusive=True):
        new = edit(f.read().decode("utf-8")).encode("utf-8")
        f.seek(0)
        f.truncate()
        f.write(new)
        f.flush()

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from _process import RunnerServer, find_runner  # noqa: E402

FREE_MB, HELD_MB = 4096, 4000
LEDGER = "xyntetik-vram-test-device.reg"


class Rig:
    def __init__(self, tmp_path):
        self.dir = tmp_path
        self.log = tmp_path / "server.log"
        self.ledger = tmp_path / LEDGER

    # The ledger is the runner's too: every read and write here takes its lock (_ledger_lock).
    def hold(self):
        # 'P': a pending claim is subtracted from the free figure (test_vram_registry.c)
        _edit_ledger(self.ledger, lambda _: f"{os.getpid()}\t1\t0\t{int(time.time())}\tP\t{HELD_MB << 20}"
                                            "\t/models/holder.gguf\t0\n")

    def release(self):
        _edit_ledger(self.ledger, lambda text: "".join(
            l for l in text.splitlines(True) if "holder.gguf" not in l))

    def waiting(self):
        return self.ledger.exists() and "\tW\t" in _read_ledger(self.ledger)

    def text(self):
        return self.log.read_text(errors="replace") if self.log.exists() else ""


def _until(what, check, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError(f"never: {what}")


def _body():
    return json.dumps({"model": "a", "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 2, "temperature": 0}).encode()


def _open(srv, send=True):
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=60)
    if send:
        b = _body()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                  + f"Content-Length: {len(b)}\r\n\r\n".encode() + b)
    return s


def _status(s):
    data = b""
    while True:
        part = s.recv(65536)
        if not part:
            break
        data += part
    s.close()
    return int(data.split(b" ", 2)[1])


def _health(srv):
    with urllib.request.urlopen(srv.base_url + "/health", timeout=5) as r:
        return json.load(r)


@pytest.fixture()
def rig(tmp_path):
    return Rig(tmp_path)


def _serve(rig, *, delay_ms=0):
    model = ROOT / "test.gguf"
    assert model.exists(), "build the generated fixture with make test.gguf"
    env = dict(os.environ, RUNNER_TEST_VRAM_FREE_MB=str(FREE_MB), RUNNER_VRAM_REGISTRY_DIR=str(rig.dir))
    if delay_ms:
        env["RUNNER_TEST_LOAD_DELAY_MS"] = str(delay_ms)
    return RunnerServer(find_runner(str(ROOT)), f"a={model},b={model}", ctx=512, parallel=1,
                        extra_args=["--wait-for-vram", "60"], log_path=str(rig.log), env=env)


def _queued_owner(rig, srv):
    rig.hold()
    owner = _open(srv)
    _until("the load joined the VRAM queue", rig.waiting)
    return owner


def _still_waiting_after(rig, seconds=0.6):
    time.sleep(seconds)
    assert rig.waiting(), rig.text()[-400:]
    assert "discarded" not in rig.text()


def test_the_wait_stops_when_its_client_goes_and_the_server_stays_up(rig):
    with _serve(rig) as srv:
        _queued_owner(rig, srv).close()
        _until("the wait stopped", lambda: "discarded" in rig.text())
        assert not rig.waiting(), "the queue marker must go with the wait"
        assert _health(srv)["status"] == "ok" and _health(srv)["resident"] is None
        rig.release()
        assert _status(_open(srv)) == 200


def test_a_client_that_stays_keeps_its_wait(rig):
    with _serve(rig) as srv:
        owner = _queued_owner(rig, srv)
        _still_waiting_after(rig)
        rig.release()
        assert _status(owner) == 200


def test_a_live_client_behind_it_keeps_the_wait(rig):
    with _serve(rig) as srv:
        owner = _queued_owner(rig, srv)
        queued = _open(srv)
        time.sleep(0.4)
        owner.close()
        _still_waiting_after(rig)
        rig.release()
        assert _status(queued) == 200


def test_a_queued_socket_that_said_nothing_and_left_is_gone(rig):
    with _serve(rig) as srv:
        owner = _queued_owner(rig, srv)
        silent = _open(srv, send=False)
        time.sleep(0.5)   # past the accept path's 250 ms: it is in the queue now
        silent.close()
        owner.close()
        _until("the wait stopped", lambda: "discarded" in rig.text())


@pytest.mark.parametrize("leave", [lambda s: s.close(), lambda s: s.shutdown(socket.SHUT_WR)],
                         ids=["closed", "half-closed"])
def test_a_queued_request_counts_as_live_however_its_client_left(rig, leave):
    """Its bytes are unread, so a close behind them is invisible, and a half
    close looks the same. The safe error is the wait going on, as it did."""
    with _serve(rig) as srv:
        owner = _queued_owner(rig, srv)
        queued = _open(srv)
        time.sleep(0.4)
        leave(queued)
        owner.close()
        _still_waiting_after(rig)
        rig.release()
        _until("the load finished", lambda: _health(srv)["resident"] is not None)
        queued.close()


def test_a_disconnect_after_admission_keeps_the_finished_load(rig):
    with _serve(rig, delay_ms=1500) as srv:
        owner = _open(srv)
        _until("the load was admitted", lambda: "test: load admitted" in rig.text())
        owner.close()
        _until("the load finished", lambda: _health(srv)["resident"] is not None)
        assert "discarded" not in rig.text()
        assert _status(_open(srv)) == 200
        assert rig.text().count("swap: loading a") == 1, "the next request must not load it again"


def test_unload_still_stops_the_wait(rig):
    with _serve(rig) as srv:
        owner = _queued_owner(rig, srv)
        with urllib.request.urlopen(urllib.request.Request(srv.base_url + "/unload", data=b""), timeout=5) as r:
            assert json.load(r)["deferred"] is True
        assert _status(owner) == 503
        assert not rig.waiting()
