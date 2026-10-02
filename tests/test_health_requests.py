"""/health lists what each busy slot is doing, and stops listing a request
whose client went away.

A streamed reply's headers are only sent once prefill is over, so a client
waiting on a long prompt cannot tell prefill from a hang, and one that closed
its connection cannot see whether the work stopped. `requests` on /health is
one row per busy slot: the phase, how much of the prompt is in the cache, how
many tokens were generated. An idle server lists nothing.

The fixture model generates thousands of tokens a second, so the server is
started with the RUNNER_TEST_STEP_DELAY_MS hook to give a request a
duration.
"""
import http.client
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import RunnerServer, find_runner  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("health") / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return out


def _health(srv):
    with urllib.request.urlopen(srv.base_url + "/health", timeout=30) as r:
        return json.load(r)


def _open_stream(srv, max_tokens):
    host, port = srv.base_url.split("//")[1].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=60)
    body = {"prompt": "the quick brown fox", "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    conn.request("POST", "/v1/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    return conn, resp


def _wait(pred, seconds):
    end = time.time() + seconds
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    return None


@pytest.fixture(scope="module")
def slow_server(model):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    env = dict(os.environ, RUNNER_TEST_STEP_DELAY_MS="25")
    with RunnerServer(exe, model, ctx=256, parallel=1, env=env,
                      extra_args=["--gpu", "off", "-t", "1"]) as srv:
        yield srv


def test_a_busy_slot_is_listed_and_an_idle_server_lists_none(slow_server):
    srv = slow_server
    assert _health(srv)["requests"] == []
    conn, resp = _open_stream(srv, 60)       # about 1.5 s of generation
    rows = []
    while True:
        line = resp.readline()
        if not line or line.strip() == b"data: [DONE]":
            break
        rows.extend(_health(srv)["requests"])
    conn.close()
    assert rows, "no busy row was observed during a 1.5 s request"
    for r in rows:
        assert r["slot"] == 0 and r["phase"] in ("prefill", "generate")
        assert 0 <= r["prompt_done"] <= r["prompt_tokens"]
        if r["phase"] == "generate":
            assert r["prompt_done"] == r["prompt_tokens"]
    gen = [r["generated"] for r in rows if r["phase"] == "generate"]
    assert gen and gen == sorted(gen) and gen[-1] > gen[0]
    assert _wait(lambda: _health(srv)["requests"] == [], 5)


def test_a_request_whose_client_left_stops_being_listed(slow_server):
    """The confirmation a client cannot get from its own closed socket."""
    srv = slow_server
    conn, resp = _open_stream(srv, 150)      # would run about 4 s
    assert _wait(lambda: any(r["generated"] > 2
                             for r in _health(srv)["requests"]), 5)
    resp.close()
    conn.close()
    t0 = time.time()
    assert _wait(lambda: _health(srv)["requests"] == []
                 and _health(srv)["active_requests"] == 0, 3), \
        "the request was still listed 3 s after its client closed"
    # well short of the 4 s it had left: it stopped, it did not finish
    assert time.time() - t0 < 2.5
