"""Server smokes that lived as shell steps in .github/workflows/ci.yml until
2026-10-13, as tests. Each starts the server it needs (the shared session
server has one model and no draft), so they are a few seconds each; what they
pin is named in each test.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from harness import RunnerServer, find_runner

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.environ.get("RUNNER_TEST_MODEL", os.path.join(ROOT, "test.gguf"))


def _post(base, path, payload, timeout=60):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def _status(base, path, payload):
    try:
        _post(base, path, payload)
    except urllib.error.HTTPError as e:
        return e.code
    return 200


def test_health_models_and_capabilities_bypass_a_busy_slot():
    """A stalled request pins the only slot; /health, /v1/models and
    /v1/capabilities must still answer from the accept loop (the watchdog
    regression)."""
    srv = RunnerServer(find_runner(ROOT), MODEL, ctx=256, parallel=1,
                       extra_args=["--gpu", "off"])
    srv.start()
    stalled = None
    try:
        stalled = socket.create_connection(("127.0.0.1", srv.port), timeout=30)
        stalled.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")  # never finishes
        time.sleep(1)
        t0 = time.monotonic()
        _get(srv.base_url, "/health", timeout=3)
        _get(srv.base_url, "/v1/models", timeout=3)
        caps = _get(srv.base_url, "/v1/capabilities", timeout=3)
        assert caps["features"]["json_schema"]
        assert time.monotonic() - t0 < 3, "control routes queued behind the pinned slot"
    finally:
        if stalled is not None:
            stalled.close()
        srv.stop()


def test_a_stalled_request_is_reaped_with_408_and_the_slot_is_released():
    """A client that never finishes its request gets a 408 within the 10 s
    read deadline, and the slot it held serves the next request."""
    srv = RunnerServer(find_runner(ROOT), MODEL, ctx=256, parallel=1,
                       extra_args=["--gpu", "off"])
    srv.start()
    try:
        s = socket.create_connection(("127.0.0.1", srv.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n")
        s.settimeout(15)
        data = b""
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
        except socket.timeout:
            pytest.fail("stalled connection not reaped within 15 s")
        finally:
            s.close()
        assert b"408" in data.split(b"\r\n", 1)[0], data[:60]
        r = _post(srv.base_url, "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 4, "temperature": 0}, timeout=10)
        assert r["choices"][0]["message"]
    finally:
        srv.stop()


def test_model_swap_serves_both_names_and_a_broken_entry_is_refused(tmp_path):
    """A `name=file,name=file` registry serves both models, lists both and
    advertises swap; an entry that is not a GGUF is refused on request and
    the good model keeps serving."""
    exe = find_runner(ROOT)
    q8 = tmp_path / "test-q8.gguf"
    subprocess.run([exe, "-m", MODEL, "--quantize", str(q8), "--quant", "q8_0"],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf")
    chat = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 3}

    srv = RunnerServer(exe, f"a={MODEL},b={q8}", ctx=256, parallel=1,
                       extra_args=["--gpu", "off", "--ttl", "0"])
    srv.start()
    try:
        _post(srv.base_url, "/v1/chat/completions", dict(chat, model="a"))
        _post(srv.base_url, "/v1/chat/completions", dict(chat, model="b"))
        assert len(_get(srv.base_url, "/v1/models")["data"]) == 2
        caps = _get(srv.base_url, "/v1/capabilities")
        assert caps["swap"] and len(caps["models"]) == 2
    finally:
        srv.stop()

    srv = RunnerServer(exe, f"good={MODEL},bad={bad}", ctx=256, parallel=1,
                       extra_args=["--gpu", "off", "--ttl", "0"])
    srv.start()
    try:
        assert _status(srv.base_url, "/v1/chat/completions", dict(chat, model="bad")) >= 400
        _post(srv.base_url, "/v1/chat/completions", dict(chat, model="good"))
    finally:
        srv.stop()


def test_a_self_drafting_server_matches_the_plain_server_byte_for_byte():
    """Speculative decoding at temperature 0 reproduces the plain greedy
    output exactly, and a json_object request still succeeds through the
    constrained speculative path."""
    exe = find_runner(ROOT)
    req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 12, "temperature": 0}
    srv = RunnerServer(exe, MODEL, ctx=256, parallel=1, extra_args=["--gpu", "off"])
    srv.start()
    try:
        plain = _post(srv.base_url, "/v1/chat/completions", req)["choices"][0]["message"]["content"]
    finally:
        srv.stop()
    assert plain
    srv = RunnerServer(exe, MODEL, ctx=256, parallel=1,
                       extra_args=["--gpu", "off", "--draft", MODEL])
    srv.start()
    try:
        spec = _post(srv.base_url, "/v1/chat/completions", req)["choices"][0]["message"]["content"]
        assert spec == plain, f"spec output diverged: {plain!r} vs {spec!r}"
        r = _post(srv.base_url, "/v1/chat/completions",
                  dict(req, max_tokens=24, response_format={"type": "json_object"}))
        content = r["choices"][0]["message"]["content"]
        if r["choices"][0].get("finish_reason") == "length" and not content.startswith("{"):
            pytest.skip("the near-random fixture hit the token limit before opening an object")
        json.loads(content)
    finally:
        srv.stop()
