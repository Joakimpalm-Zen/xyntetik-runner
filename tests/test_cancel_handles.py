"""Cancels by request id leave no handle behind in the server (R6.68.22).

A Windows soak counted the server's handles: 400 cancels by id took it from
265 to 456, one each, while closed sockets and finished requests left none.
Under winpthreads a thread created detached keeps its Windows HANDLE after it
exits, and every cancel that waited for its request got such a thread. The
answers now come from one long-lived thread, joined at the stop.

Absolute anchor: the operating system's own count of the process's handles,
GetProcessHandleCount. A run of cancels after a warm-up must leave it where it
was, give or take what the runtime moves on its own. Windows only: on POSIX a
detached thread's resources go with it, and the race gates (test-cancel-race,
test-server-race-tsan) cover the answering thread there.
"""
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from _process import RunnerServer, find_runner  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "nt", reason="the Windows process handle count")

WARMUP, CANCELS, SLACK = 10, 100, 20


def _handle_count(pid):
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    assert h, ctypes.get_last_error()
    n = wintypes.DWORD(0)
    try:
        assert k.GetProcessHandleCount(h, ctypes.byref(n)), ctypes.get_last_error()
    finally:
        k.CloseHandle(h)
    return n.value


def _call(srv, path, body=None, headers=None, timeout=60):
    req = urllib.request.Request(srv.base_url + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers=dict({"Content-Type": "application/json"}, **(headers or {})),
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _cancel_once(srv, rid):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", _call(
        srv, "/v1/completions", {"prompt": "hi", "max_tokens": 64, "ignore_eos": True,
                                 "temperature": 0}, {"X-Request-Id": rid})), daemon=True)
    t.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if any(r["phase"] == "generate" for r in _call(srv, "/health")[1].get("requests", [])):
            break
        time.sleep(0.005)
    req = urllib.request.Request(f"{srv.base_url}/v1/runner/requests/{rid}/cancel", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=40) as r:
        assert r.status == 200 and json.load(r)["cancelled"] is True
    t.join(40)
    assert out["r"][0] == 200, out


def test_cancels_by_id_leave_no_handle_behind():
    env = dict(os.environ, RUNNER_TEST_STEP_DELAY_MS="20")
    with RunnerServer(find_runner(str(ROOT)), str(ROOT / "test.gguf"), ctx=256, parallel=1, env=env,
                      extra_args=["--gpu", "off", "-t", "1"]) as srv:
        for i in range(WARMUP):
            _cancel_once(srv, f"warm-{i}")
        time.sleep(0.5)
        before = _handle_count(srv.proc.pid)
        for i in range(CANCELS):
            _cancel_once(srv, f"h-{i}")
        time.sleep(0.5)
        after = _handle_count(srv.proc.pid)
        assert after - before < SLACK, f"{CANCELS} cancels by id moved the handle count {before} -> {after}"
