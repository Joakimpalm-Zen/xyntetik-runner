"""A tools[] gemma4 cannot natively constrain falls back, not 400s.

gemma4's native call syntax has no spelling for a free-form value, so a
parameter declared without a `type` (`{"body": {"description": "anything"}}`)
cannot be compiled to it. That used to 400 the request with "has no declared
`type`" -- a schema that worked on gpt-oss failed on gemma4. Decided
2026-08-27 (owner): fall back to the GENERIC strict envelope for that one
request, losing native syntax rather than the request. The switch happens
before rendering, so prompt and grammar stay in agreement.

Pinned here: the untyped request now succeeds end-to-end under a forced
gemma4 template, a typed request still succeeds (native path untouched),
and the server says which protocol it used on its stderr.
"""
import contextlib
import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
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
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve_gemma4(runner_bin, model):
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--port", str(port),
         "-c", "512", "--gpu", "off", "--no-tray",
         "--chat-template", "gemma4"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    "server exited during startup: "
                    + proc.stderr.read().decode(errors="replace"))
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        else:
            raise AssertionError("server never answered /health")
        yield base, proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def _chat(base, tools):
    body = json.dumps({
        "messages": [{"role": "user", "content": "send it"}],
        "tools": tools, "tool_choice": "required",
        "max_tokens": 48, "temperature": 0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


UNTYPED = [{"type": "function", "function": {
    "name": "post_note",
    "parameters": {"type": "object",
                   "properties": {"body": {"description": "anything"}},
                   "required": ["body"]}}}]

TYPED = [{"type": "function", "function": {
    "name": "post_note",
    "parameters": {"type": "object",
                   "properties": {"body": {"type": "string"}},
                   "required": ["body"]}}}]


def test_untyped_parameter_falls_back_instead_of_400(runner_bin, model):
    with _serve_gemma4(runner_bin, model) as (base, proc):
        status, payload = _chat(base, UNTYPED)
        assert status == 200, payload
        calls = payload["choices"][0]["message"].get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "post_note"
        # the arguments must still parse: the guarantee survives the fallback
        json.loads(calls[0]["function"]["arguments"])
        proc.terminate()
        stderr = proc.stderr.read().decode(errors="replace")
        assert "generic envelope" in stderr  # the downgrade is said, not silent


def test_typed_parameter_still_uses_the_native_path(runner_bin, model):
    with _serve_gemma4(runner_bin, model) as (base, proc):
        status, payload = _chat(base, TYPED)
        assert status == 200, payload
        calls = payload["choices"][0]["message"].get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "post_note"
        json.loads(calls[0]["function"]["arguments"])
        proc.terminate()
        stderr = proc.stderr.read().decode(errors="replace")
        assert "generic envelope" not in stderr  # no downgrade on this one
