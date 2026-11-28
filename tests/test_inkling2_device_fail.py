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


# ---- PR #1065 second read: every surface, the flag's scope, and co-batched slots

def _post_h(srv, path, body, headers=()):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **dict(headers)})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _server_args(runner_bin, model, extra_env, *extra, gpu="auto"):
    return RunnerServer(str(runner_bin), str(model), ctx=CTX, parallel=1, env=_env(extra_env),
                        extra_args=["--gpu", gpu, "-t", "2", "-b", "4", *extra])


# (1) /v1/decide and /v1/rerank: a device failure under the scoring is the server's, 500 device_failed
@pytest.mark.parametrize("path,body", [
    ("/v1/decide", {"state": PROMPT, "questions": [{"id": "q", "question": "which?", "options": ["a", "b"]}]}),
    ("/v1/rerank", {"query": "hello", "documents": ["hello there", "general"]}),
])
def test_decide_and_rerank_answer_a_device_failure_as_500(runner_bin, tiny, path, body):
    # rerank's rendered question is longer than CTX: a context that holds it, so the failure is the device's
    with RunnerServer(str(runner_bin), str(tiny), ctx=512, parallel=1, env=_env(PREFILL_FAIL),
                      extra_args=["--gpu", "auto", "-t", "2", "-b", "4"]) as srv:
        code, out = _post_h(srv, path, body)
    assert code == 500, (code, out)
    assert out["error"]["code"] == "device_failed", out


# (2) the Responses and Messages surfaces name the device, never memory, and never invite a retry
def test_responses_names_a_device_failure(runner_bin, tiny):
    with _server(runner_bin, tiny, SERVER_DECODE_FAIL) as srv:
        code, out = _post_h(srv, "/v1/responses", {"input": PROMPT, "max_output_tokens": 4, "temperature": 0})
    assert code == 200, (code, out)
    assert out["status"] == "incomplete", out
    assert out["incomplete_details"]["reason"] == "device_failed", out


def test_messages_names_a_device_failure(runner_bin, tiny):
    with _server(runner_bin, tiny, SERVER_DECODE_FAIL) as srv:
        code, out = _post_h(srv, "/v1/messages",
                            {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4, "temperature": 0},
                            headers={"anthropic-version": "2023-06-01"})
    assert code == 500, (code, out)
    msg = out["error"]["message"]
    assert "device" in msg and "memory" not in msg and "retry it." not in msg, out
    assert out["runner_telemetry"]["finish_detail"] == "device_failed", out


def test_completions_carry_the_device_detail(runner_bin, tiny):
    with _server(runner_bin, tiny, SERVER_DECODE_FAIL) as srv:
        code, out = _post_h(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 4, "temperature": 0})
    assert code == 200, (code, out)
    assert out["choices"][0]["finish_reason"] == "error", out
    assert out["runner_telemetry"]["finish_detail"] == "device_failed", out


# (3) sessions: a device-cut generation is not a "length" answer
def test_a_session_answers_a_device_failure_as_500(runner_bin, tiny, tmp_path):
    with _server_args(runner_bin, tiny, SERVER_DECODE_FAIL, "--sessions", str(tmp_path / "s")) as srv:
        code, out = _post_h(srv, "/v1/runner/sessions", {"prompt": PROMPT, "max_tokens": 4, "temperature": 0})
    assert code == 500, (code, out)
    assert out["error"]["code"] == "device_failed", out


# (4) no receipt for a turn the device cut short (the CLI writes none either); a healthy turn writes one
def test_the_server_writes_no_receipt_for_a_device_cut_turn(runner_bin, tiny, tmp_path):
    ok_dir, cut_dir = tmp_path / "ok", tmp_path / "cut"
    ok_dir.mkdir()
    cut_dir.mkdir()
    with _server_args(runner_bin, tiny, None, "--receipts", str(ok_dir)) as srv:
        code, _ = _post_h(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 4, "temperature": 0})
    assert code == 200 and list(ok_dir.iterdir()), "the control wrote no receipt: the test below would be blind"
    with _server_args(runner_bin, tiny, SERVER_DECODE_FAIL, "--receipts", str(cut_dir)) as srv:
        code, out = _post_h(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 4, "temperature": 0})
    assert code == 200 and out["choices"][0]["finish_reason"] == "error", out
    assert not list(cut_dir.iterdir()), list(cut_dir.iterdir())


# (5) a failed forward that is NOT the device's (TEST-ONLY RUNNER_INKLING2_TEST_FWD_FAIL, any route) never reads as one
CPU_PREFILL_FAIL = {"RUNNER_INKLING2_TEST_FWD_FAIL": "0"}
# the CLI feeds the 20-token prompt in 5 chunks of -b 4 (forwards 0-4), so forward 7 is the third generated token's;
# the server's prompt shape takes 6 (engine_feed_prompt: the chunks, then the last token alone), so 8 is in decode
CPU_DECODE_FAIL = {"RUNNER_INKLING2_TEST_FWD_FAIL": "7"}
SERVER_CPU_DECODE_FAIL = {"RUNNER_INKLING2_TEST_FWD_FAIL": "8"}


def test_a_cpu_route_failure_is_not_a_device_failure_on_the_cli(runner_bin, tiny):
    for extra in (CPU_PREFILL_FAIL, CPU_DECODE_FAIL):
        r = subprocess.run([str(runner_bin), "-m", str(tiny), "--gpu", "off", "-c", str(CTX), "-t", "2", "-b", "4",
                            "--temp", "0", "--no-tray", "-p", PROMPT, "-n", "6"],
                           capture_output=True, encoding="utf-8", errors="replace", env=_env(extra), timeout=600)
        assert r.returncode == 1, (extra, r.returncode, r.stderr[-2000:])
        assert "device" not in r.stderr.split("refused (test injection)")[-1], (extra, r.stderr[-2000:])
        assert "exceeds context" not in r.stderr and "out of memory" in r.stderr, (extra, r.stderr[-2000:])


def test_a_cpu_route_failure_is_not_a_device_failure_on_the_server(runner_bin, tiny):
    with _server_args(runner_bin, tiny, CPU_PREFILL_FAIL, gpu="off") as srv:
        code, out = _post_h(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 4, "temperature": 0})
    assert code == 500 and out["error"].get("code") != "device_failed", (code, out)
    assert "context" not in out["error"]["message"], out


# (6) the flag is the request's own: a failure in one request does not mark the next one
def test_a_failure_does_not_leak_into_the_next_request(runner_bin, tiny):
    with _server_args(runner_bin, tiny, SERVER_CPU_DECODE_FAIL, gpu="off") as srv:
        code, first = _post_h(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 6, "temperature": 0})
        assert code == 200 and first["choices"][0]["finish_reason"] == "error", (code, first)
        code, second = _post_h(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 6, "temperature": 0})
    assert code == 200, (code, second)
    assert second["choices"][0]["finish_reason"] in ("stop", "length"), second
    assert "finish_detail" not in second.get("runner_telemetry", {}), second


# (8) --parallel 2, one slot's failed forward ending its co-batched neighbours: tests/test_inkling2_fast.c (o) pins
# model_batch_decode's per-member rows directly. A server-level race cannot: the scheduler's gather window is
# milliseconds, so two requests meet in one step only by timing.


# (9) /v1/embeddings has no readout on inkling2_loop: refused by name (the server died on it), on any route
@pytest.mark.parametrize("gpu", ["off", "auto"])
def test_embeddings_are_refused_on_inkling2(runner_bin, tiny, gpu):
    with _server_args(runner_bin, tiny, None, gpu=gpu) as srv:
        code, out = _post_h(srv, "/v1/embeddings", {"input": "hello there"})
        alive, _ = _post_h(srv, "/v1/completions", {"prompt": PROMPT, "max_tokens": 1, "temperature": 0})
    assert code == 400 and out["error"]["code"] == "embeddings_unsupported", (code, out)
    assert alive == 200
