"""Cancel a request by id: POST /v1/runner/requests/{id}/cancel (R10.12.3.2).

A client that wants an answer instead of polling /health names the request,
by the X-Request-Id it sent (the id a buffered client knows before its reply
arrives) or by the response id (which a streamed client reads from its first
chunk). The cancel is answered from the accept loop, so it does not queue
behind the request it cancels on a one-slot server, and its reply waits until
the request has stopped. The cancelled request's own client gets what was
generated, with finish_detail "cancelled" and a receipt that replays; a cut
while its prompt was processed answers 409 "cancelled". The server stays up.

Driven with the step and prefill delay hooks: a fixture model generates
thousands of tokens a second, so a request has to be slowed to be caught.
"""
import hashlib
import http.client
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("cancel") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _call(srv, method, path, payload=None, headers=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    h = {"Content-Type": "application/json"} if data is not None else {}
    h.update(headers or {})
    req = urllib.request.Request(srv.base_url + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _health(srv):
    return _call(srv, "GET", "/health")[1]


def _wait(pred, secs):
    end = time.time() + secs
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _cancel(srv, rid):
    return _call(srv, "POST", f"/v1/runner/requests/{rid}/cancel")


def _in_background(fn):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    t.start()
    return t, out


@pytest.fixture(scope="module")
def slow(runner_bin, model, tmp_path_factory):
    d = tmp_path_factory.mktemp("cancel-receipts")
    env = dict(os.environ, RUNNER_TEST_STEP_DELAY_MS="20")
    with RunnerServer(runner_bin, model, ctx=256, parallel=1, env=env, extra_args=[
            "--gpu", "off", "-t", "1", "--receipts", str(d)]) as srv:
        srv.receipts = d
        yield srv


LONG = {"prompt": "the quick brown fox", "max_tokens": 400, "temperature": 0.8,
        "seed": 9, "cache_prompt": False}


def test_a_buffered_request_is_cancelled_by_its_own_id(runner_bin, model, slow):
    t, out = _in_background(lambda: _call(slow, "POST", "/v1/completions", LONG,
                                          {"X-Request-Id": "job-1"}))
    assert _wait(lambda: any(r["phase"] == "generate" and r["generated"] >= 3
                             for r in _health(slow)["requests"]), 10)
    t0 = time.time()
    code, ans = _cancel(slow, "job-1")       # one slot: this must not queue
    assert time.time() - t0 < 5, "the cancel waited behind the request"
    assert code == 200, ans
    assert ans["id"] == "job-1" and ans["cancelled"] is True
    assert ans["phase"] == "generate" and ans["request_id"].startswith("cmpl-")
    t.join(30)
    code, body = out["r"]
    assert code == 200, body
    assert body["id"] == ans["request_id"]
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["runner_telemetry"]["finish_detail"] == "cancelled"
    assert 0 < body["usage"]["completion_tokens"] < LONG["max_tokens"]
    assert body["usage"]["completion_tokens"] == ans["generated"]
    # the receipt says it was cut, and replays the tokens it holds
    rec_path = slow.receipts / body["runner_telemetry"]["receipt"]["file"]
    rec = json.loads(rec_path.read_text())
    assert rec["serve"]["cut"] == "cancelled"
    assert len(rec["output"]["tokens"]) == ans["generated"]
    v = subprocess.run([runner_bin, "-m", model, "--verify", rec_path, "--gpu", "off",
                        "-t", "1", "-c", "256"], capture_output=True, timeout=120)
    assert v.returncode == 0 and b"VERIFIED" in v.stderr, v.stderr[-600:]
    # without the field, as before it, the replay runs on to max_tokens and
    # blames the model for a cut it did not make
    raw = rec_path.read_text()
    cut = raw.rindex(',"chain":')
    old = json.loads(raw[:cut] + "}")
    del old["serve"]["cut"]
    old.pop("signature", None)
    head = json.dumps(old, separators=(",", ":"))[:-1]
    old_path = rec_path.with_name("old.json")
    old_path.write_text(head + ',"chain":' + json.dumps(
        {"algo": "sha256", "prev": rec["chain"]["prev"],
         "hash": hashlib.sha256(head.encode()).hexdigest()}, separators=(",", ":")) + "}")
    v = subprocess.run([runner_bin, "-m", model, "--verify", old_path, "--gpu", "off",
                        "-t", "1", "-c", "256"], capture_output=True, timeout=120)
    old_path.unlink()
    assert v.returncode == 2 and b"DIVERGED" in v.stderr, v.stderr[-600:]
    # the id is free again, and the server serves the next request whole
    assert _wait(lambda: _health(slow)["requests"] == [], 5)
    code, body = _call(slow, "POST", "/v1/completions",
                       dict(LONG, max_tokens=4), {"X-Request-Id": "job-1"})
    assert code == 200 and body["choices"][0]["finish_reason"] == "length", body
    assert "finish_detail" not in body.get("runner_telemetry", {})


def test_a_streamed_request_is_cancelled_by_its_response_id(slow):
    conn = http.client.HTTPConnection("127.0.0.1", slow.port, timeout=60)
    conn.request("POST", "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 400,
        "temperature": 0.8, "seed": 4, "stream": True, "cache_prompt": False}),
        {"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    rid, chunks = None, []
    while True:
        line = resp.readline()
        if not line or line.strip() == b"data: [DONE]":
            break
        if line.startswith(b"data: "):
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            if rid is None:
                rid = chunk["id"]
                code, ans = _cancel(slow, rid)
                assert code == 200 and ans["request_id"] == rid, ans
    conn.close()
    last = [c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    assert last and last[-1]["choices"][0]["finish_reason"] == "length", chunks[-2:]
    assert last[-1]["choices"][0].get("finish_detail") == "cancelled" or \
        chunks[-1].get("runner_telemetry", {}).get("finish_detail") == "cancelled", chunks[-2:]
    assert _health(slow)["status"] == "ok"


def test_unknown_and_malformed_ids(slow):
    code, ans = _cancel(slow, "nobody")
    assert code == 404 and ans["error"]["code"] == "request_not_found", ans
    code, ans = _cancel(slow, "bad%20id")
    assert code == 404, ans
    code, ans = _call(slow, "POST", "/v1/completions", dict(LONG, max_tokens=2),
                      {"X-Request-Id": "has space"})
    assert code == 400 and ans["error"]["param"] == "X-Request-Id", ans
    code, ans = _call(slow, "POST", "/v1/completions", dict(LONG, max_tokens=2),
                      {"X-Request-Id": "x" * 65})
    assert code == 400 and ans["error"]["code"] == "invalid_request_id", ans
    # a cancel takes no body
    code, ans = _call(slow, "POST", "/v1/runner/requests/nobody/cancel", {"x": 1})
    assert code == 400, ans


def test_an_active_id_cannot_be_taken_twice(runner_bin, model):
    env = dict(os.environ, RUNNER_TEST_STEP_DELAY_MS="20")
    with RunnerServer(runner_bin, model, ctx=256, parallel=2, env=env,
                      extra_args=["--gpu", "off", "-t", "2"]) as srv:
        t, out = _in_background(lambda: _call(srv, "POST", "/v1/completions", LONG,
                                              {"X-Request-Id": "twin"}))
        assert _wait(lambda: any(r["phase"] == "generate"
                                 for r in _health(srv)["requests"]), 10)
        code, ans = _call(srv, "POST", "/v1/completions", dict(LONG, max_tokens=2),
                          {"X-Request-Id": "twin"})
        assert code == 409 and ans["error"]["code"] == "duplicate_request_id", ans
        assert _cancel(srv, "twin")[0] == 200
        t.join(30)
        assert out["r"][1]["runner_telemetry"]["finish_detail"] == "cancelled"


def test_a_cut_while_the_prompt_is_processed_answers_409(runner_bin, model):
    env = dict(os.environ, RUNNER_TEST_PREFILL_DELAY_MS="200")
    with RunnerServer(runner_bin, model, ctx=256, parallel=1, env=env,
                      extra_args=["--gpu", "off", "-t", "1", "-b", "8"]) as srv:
        # about 60 tokens on the byte-fallback fixture: eight chunks of 8,
        # with the delay between them
        prompt = "abcd " * 12
        t, out = _in_background(lambda: _call(srv, "POST", "/v1/completions", {
            "prompt": prompt, "max_tokens": 8, "cache_prompt": False},
            {"X-Request-Id": "early"}))
        assert _wait(lambda: any(r["phase"] == "prefill"
                                 for r in _health(srv)["requests"]), 10)
        code, ans = _cancel(srv, "early")
        assert code == 200 and ans["phase"] == "prefill" and ans["generated"] == 0, ans
        t.join(30)
        code, body = out["r"]
        assert code == 409 and body["error"]["code"] == "cancelled", body
        assert _health(srv)["status"] == "ok"
