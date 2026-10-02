"""A tool schema the function-XML syntax cannot enforce falls back, not 400s.

Granite 4.2, Qwen 3.8, Ornith and Qwen3-Coder write a call as
`<function=NAME><parameter=KEY>raw text</parameter>...`. Under a required or
named tool choice the runner holds the turn to a grammar, and that grammar
has no way to enforce a string parameter's `minLength`, `maxLength` or
`pattern`: the value is raw text up to the closing tag. Such a request was
answered 400, so a tool that worked on every JSON family failed on these
four (15 of 120 agent-torture requests on Granite 4.2 8B, 2026-10-02).

It now takes the generic envelope for that one request, where the constraint
is enforced in JSON. The switch is made before rendering, so the prompt and
the grammar agree, and the server says so on stderr.
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

CONSTRAINED = [{"type": "function", "function": {
    "name": "record_finding",
    "parameters": {"type": "object", "additionalProperties": False,
                   "properties": {"hypothesis": {"type": "string", "minLength": 3,
                                                 "maxLength": 40},
                                  "confidence": {"type": "integer"}},
                   "required": ["hypothesis", "confidence"]}}}]

PLAIN = [{"type": "function", "function": {
    "name": "record_finding",
    "parameters": {"type": "object", "additionalProperties": False,
                   "properties": {"hypothesis": {"type": "string"},
                                  "confidence": {"type": "integer"}},
                   "required": ["hypothesis", "confidence"]}}}]


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
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve(runner_bin, model, template):
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--port", str(port),
         "-c", "8192", "--gpu", "off", "--no-tray", "--chat-template", template],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError("server exited during startup: "
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


def _chat(base, tools, choice="required"):
    body = json.dumps({
        "messages": [{"role": "user", "content": "record it"}],
        "tools": tools, "tool_choice": choice,
        "max_tokens": 96, "temperature": 0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


FAMILIES = ["granite42", "qwen38", "ornith", "qwen3-coder"]


@pytest.mark.parametrize("template", FAMILIES)
@pytest.mark.parametrize("choice", ["required",
                                    {"type": "function",
                                     "function": {"name": "record_finding"}}],
                         ids=["required", "named"])
def test_an_unenforceable_schema_falls_back_instead_of_400(runner_bin, model,
                                                           template, choice):
    with _serve(runner_bin, model, template) as (base, proc):
        status, payload = _chat(base, CONSTRAINED, choice)
        assert status == 200, payload
        calls = payload["choices"][0]["message"].get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "record_finding"
        args = json.loads(calls[0]["function"]["arguments"])
        # the constraint the XML syntax could not hold is held
        if payload["choices"][0]["finish_reason"] != "length":
            assert 3 <= len(args["hypothesis"]) <= 40
        proto = payload["runner_telemetry"]["tool_protocol"]
        assert proto["family"] != "qwen3_xml", proto
        proc.terminate()
        stderr = proc.stderr.read().decode(errors="replace")
        assert "generic envelope" in stderr   # the downgrade is said, not silent


@pytest.mark.parametrize("template", FAMILIES)
def test_an_enforceable_schema_keeps_the_native_protocol(runner_bin, model, template):
    with _serve(runner_bin, model, template) as (base, proc):
        status, payload = _chat(base, PLAIN)
        assert status == 200, payload
        proto = payload["runner_telemetry"]["tool_protocol"]
        assert proto["family"] == "qwen3_xml", proto
        proc.terminate()
        stderr = proc.stderr.read().decode(errors="replace")
        assert "generic envelope" not in stderr


@pytest.mark.parametrize("template", FAMILIES)
def test_an_auto_turn_is_untouched(runner_bin, model, template):
    """Auto is parsed, not constrained, so there is nothing to fall back from."""
    with _serve(runner_bin, model, template) as (base, proc):
        status, payload = _chat(base, CONSTRAINED, "auto")
        assert status == 200, payload
        assert payload["runner_telemetry"]["tool_protocol"]["family"] == "qwen3_xml"
