"""A forward that fails on the device is reported as a device failure, never as a full context, and never crashes.

An inkling2_loop forward on CUDA that fails returns no logits, and its instance computes no more rows. The engine
read that empty row as a context overflow in prefill (the CLI said "prompt exceeds context", the server 400
context_length_exceeded), went on feeding the prompt's later chunks into the failed instance, and in decode handed the
empty row to the sampler, a segmentation fault. A CUDA host's test: skipped without a CUDA device. The failures are
injected with the TEST-ONLY RUNNER_INKLING2_CUDA_TEST_*_FAIL indices (docs/inkling2-cuda.md).
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from inkling2_tiny import _kvs, _tensors, _write  # noqa: E402
from harness import RunnerServer  # noqa: E402

PROMPT = "hello there, general"
CTX = 64
# the first expert body fails: the prompt's first chunk, whose forward is batched
PREFILL_FAIL = {"RUNNER_INKLING2_CUDA_TEST_EXP_FAIL": "0"}
# the first resident decode step fails: the CLI's first generated token's forward
DECODE_FAIL = {"RUNNER_INKLING2_CUDA_TEST_STEP_FAIL": "0"}
# the server feeds a prompt's last token alone, its step 0 (a recurrent fold's prefill shape, engine_feed_prompt):
# step 2 is the second generated token's forward
SERVER_DECODE_FAIL = {"RUNNER_INKLING2_CUDA_TEST_STEP_FAIL": "2"}


def _env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_INKLING2_")}
    env["RUNNER_INKLING2_CUDA"] = "1"
    env.update(extra or {})
    return env


def _run(runner_bin, model, extra_env=None, *args):
    return subprocess.run([str(runner_bin), "-m", str(model), "--gpu", "auto", "-c", str(CTX), "-t", "2", "-b", "4",
                           "--temp", "0", "--no-tray", *map(str, args)],
                          capture_output=True, encoding="utf-8", errors="replace", env=_env(extra_env), timeout=600)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, runner_bin):
    m = _write(tmp_path_factory.mktemp("ink2fail") / "tiny.gguf", _kvs(), _tensors())
    r = _run(runner_bin, m, None, "-p", PROMPT, "-n", "2")
    assert r.returncode == 0, r.stderr[-2000:]
    if "inkling2_loop: the weights on CUDA" not in r.stderr:
        pytest.skip("no CUDA device here: the device failures are a CUDA host's")
    return m


def test_a_prefill_failure_is_named_and_stops_the_prompt(runner_bin, tiny):
    r = _run(runner_bin, tiny, PREFILL_FAIL, "-p", PROMPT, "-n", "4")
    assert r.returncode == 1, (r.returncode, r.stderr[-2000:])
    assert "exceeds context" not in r.stderr, r.stderr[-2000:]
    assert "error: the prompt's forward failed on the device" in r.stderr, r.stderr[-2000:]
    # the prompt's later chunks are not fed into the failed instance
    assert "computes no more rows" not in r.stderr, r.stderr[-2000:]


def test_a_decode_failure_ends_the_run_with_an_error_not_a_crash(runner_bin, tiny):
    r = _run(runner_bin, tiny, DECODE_FAIL, "-p", PROMPT, "-n", "4")
    assert r.returncode == 1, (r.returncode, r.stderr[-2000:])
    assert "error: generation stopped: a forward failed on the device" in r.stderr, r.stderr[-2000:]


def test_the_benchmark_names_a_device_failure(runner_bin, tiny):
    r = _run(runner_bin, tiny, PREFILL_FAIL, "--bench-json", "-n", "4", "-p", PROMPT)
    assert r.returncode == 1, (r.returncode, r.stderr[-2000:])
    assert "exceeds context" not in r.stderr, r.stderr[-2000:]
    assert "failed on the device" in r.stderr, r.stderr[-2000:]


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _server(runner_bin, model, extra_env):
    return RunnerServer(str(runner_bin), str(model), ctx=CTX, parallel=1, env=_env(extra_env),
                        extra_args=["--gpu", "auto", "-t", "2", "-b", "4"])


def test_the_server_answers_a_prefill_failure_as_a_device_error(runner_bin, tiny):
    with _server(runner_bin, tiny, PREFILL_FAIL) as srv:
        code, body = _post(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 4, "temperature": 0})
    assert code == 500, (code, body)
    assert body["error"]["code"] == "device_failed", body
    assert "context" not in body["error"]["message"], body


def test_the_server_ends_a_decode_failure_as_an_error_finish(runner_bin, tiny):
    with _server(runner_bin, tiny, SERVER_DECODE_FAIL) as srv:
        code, body = _post(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 4, "temperature": 0})
        # the server outlived the failure
        code2, body2 = _post(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 1, "temperature": 0})
    assert code == 200, (code, body)
    assert body["choices"][0]["finish_reason"] == "error", body
    assert body["usage"]["completion_tokens"] < 4, body
    # the failed instance refuses the next request as a device failure, not a full context
    assert code2 == 500 and body2["error"]["code"] == "device_failed", (code2, body2)
