"""POST /unload must never claim success it cannot deliver.

A single model served with --parallel N>1 holds the model in the slots
directly and never joins the registry, so there is nothing an unload could
free: the weights and every slot KV cache stay resident. That configuration
used to answer {"status":"ok"} after freeing only the prefix cache -- an
operator reclaiming memory was told the bytes were gone while all of them
stayed, and nothing on the wire said otherwise. The fix answers 409
unload_unsupported with the remedy in the message. This file pins the
honesty in both directions:

  * --parallel 2, one model: /unload refuses with 409 and the code
    "unload_unsupported", and the server still serves afterwards --
    which is the proof the refusal was truthful (nothing was freed).
  * --parallel 1: /unload answers ok, and a later request still succeeds
    because the registry reloads the model on demand -- the promise the
    README's "give the memory back" section makes in public.
  * GET /unload is 405: a GET was reachable from any web page via
    <img src>, so the method gate is part of the contract too.
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
def _serve(runner_bin, model_spec, *extra):
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model_spec), "--serve", "--port", str(port),
         "-c", "512", "--gpu", "off", "--no-tray", *extra],
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
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def _post(base, path):
    req = urllib.request.Request(base + path, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _chat_ok(base):
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status == 200


def test_parallel_slots_refuse_with_409_and_stay_serving(runner_bin, model):
    with _serve(runner_bin, model, "--parallel", "2") as base:
        status, body = _post(base, "/unload")
        assert status == 409
        assert body["error"]["code"] == "unload_unsupported"
        # the remedy is part of the answer, not just the refusal
        assert "--parallel 1" in body["error"]["message"]
        # truthfulness check: nothing was freed, so the server still serves
        assert _chat_ok(base)


def test_parallel_one_unloads_and_reloads_on_demand(runner_bin, model):
    with _serve(runner_bin, model) as base:
        status, body = _post(base, "/unload")
        assert status == 200
        assert body["status"] == "ok"
        # the registry reloads on the next request: ok was not a lie either
        assert _chat_ok(base)


def test_get_unload_is_method_gated(runner_bin, model):
    with _serve(runner_bin, model) as base:
        try:
            with urllib.request.urlopen(base + "/unload", timeout=10) as r:
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        assert status == 405
