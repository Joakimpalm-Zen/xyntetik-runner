"""The exact-repeat count (R3.66's first slice): how often a server sees the same request again.

An answer cache is only worth building if real agent sessions repeat requests exactly. With
RUNNER_REPEAT_COUNT=1 the server keeps, per request, one 64-bit fingerprint of the model, the rendered
prompt's token ids, the sampling settings, the seed and the token cap (no content), and /metrics
counts the requests it hashed, the exact repeats among them, and the repeats a cache could answer
(greedy, or seeded sampling). Without the switch nothing is hashed and the counts are absent.
"""
import json
import os
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from _process import RunnerServer, find_runner  # noqa: E402

MODEL = ROOT / "test.gguf"


def _chat(srv, text, **extra):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 4, **extra}
    req = urllib.request.Request(srv.base_url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _metrics(srv):
    with urllib.request.urlopen(srv.base_url + "/metrics", timeout=30) as r:
        text = r.read().decode()
    return {line.split()[0]: int(line.split()[1]) for line in text.splitlines()
            if line.startswith("runner_repeat_")}


def test_exact_repeats_are_counted_without_content():
    env = dict(os.environ, RUNNER_REPEAT_COUNT="1")
    with RunnerServer(find_runner(str(ROOT)), str(MODEL), ctx=512, env=env) as srv:
        _chat(srv, "hello there", temperature=0)
        _chat(srv, "hello there", temperature=0)            # a greedy repeat: a cache could answer it
        _chat(srv, "something else", temperature=0)
        _chat(srv, "hello there", temperature=0, max_tokens=5)   # another cap: not the same request
        _chat(srv, "roll", temperature=0.9)
        _chat(srv, "roll", temperature=0.9)                 # unseeded sampling: a repeat, never cacheable
        _chat(srv, "seeded", temperature=0.9, seed=7)
        _chat(srv, "seeded", temperature=0.9, seed=7)       # seeded sampling repeats exactly
        m = _metrics(srv)
    assert m == {"runner_repeat_requests_hashed_total": 8,
                 "runner_repeat_requests_exact_total": 3,
                 "runner_repeat_requests_cacheable_total": 2}, m


def test_nothing_is_counted_without_the_switch():
    env = {k: v for k, v in os.environ.items() if k != "RUNNER_REPEAT_COUNT"}
    with RunnerServer(find_runner(str(ROOT)), str(MODEL), ctx=512, env=env) as srv:
        _chat(srv, "hello there", temperature=0)
        _chat(srv, "hello there", temperature=0)
        assert _metrics(srv) == {}
