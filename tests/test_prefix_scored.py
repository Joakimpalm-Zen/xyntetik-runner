"""A normal request does not reuse the rows a scoring request wrote (R3.60).

An echo request (logprobs over the prompt) and /v1/decide's scored tokens are fed one token a forward on the decode
route, whose rows differ in their last bits from those a prompt's prefill pass writes (tests/test_prefill_batch_
identity.c, gate 3). Since R3.60 a fresh CPU prompt's bits do not depend on -b, and a continued prefix that was
prefilled gives them too. So a normal request that extends a scored prompt must not keep the scored rows. The engine
marks the first one; the request keeps the slot's rows below it, and feeds the rest as a prompt
(`prompt_reuse` "scored_rows").

Checked on a quantized fixture, -b 64, the CPU, one slot, through the server:
- echo P, then a normal request for P plus more: nothing kept (the whole prompt was scored), and the row at a
  position inside P is the cold server's, byte for byte (RUNNER_TEST_KV_DUMP);
- decide (continuation-v1, the option after the state), then a normal request for the state, that option and more:
  the state's rows, prefilled, are kept (the mark is a position, not a flag), the option's are fed again, and the row
  at a position inside the option is the cold server's;
- the controls: each warm request does print the dumped row, so the comparison is of a row it wrote. Before the fix
  the scored rows were kept and the row was never written by the normal request, which this test reports as red.
"""
import json
import pathlib
import re
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer, find_runner  # noqa: E402

SEAM = "RUNNER_TEST_KV_DUMP"
LINE = re.compile(r"kvdump: pos (\d+) layer (\d+) k ([0-9a-f]{16}|-) v ([0-9a-f]{16}|-)")
MODEL = ROOT / "test-q4k.gguf"
P = "the quick brown fox jumps over the lazy dog and keeps running"
STATE = "The quick brown fox"
OPTIONS = [" a cat", " jumps over"]   # sorted by token: the slot ends on " jumps over"'s path
FOLLOW = STATE + " jumps over the lazy dog"


def _server(tmp_path, pos=None):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    if not MODEL.exists():
        pytest.skip(f"{MODEL.name} not generated (make test-q4k.gguf)")
    log = tmp_path / f"server-{pos}-{len(list(tmp_path.iterdir()))}.log"
    env = {SEAM: f"{pos},1"} if pos is not None else {}
    return RunnerServer(exe, MODEL, ctx=256, parallel=1, log_path=str(log),
                        extra_args=["--gpu", "off", "-t", "2", "-b", "64"], env=env), log


def _post(srv, path, body):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _dumped(log, before, pos):
    text = log.read_text(errors="replace")[len(before):]
    return {int(l): (k, v) for p, l, k, v in LINE.findall(text) if int(p) == pos}


def _complete(srv, log, prompt, pos, **extra):
    before = log.read_text(errors="replace") if log.exists() else ""
    body = _post(srv, "/v1/completions", {"prompt": prompt, "max_tokens": 1, "temperature": 0, **extra})
    return body["runner_telemetry"], _dumped(log, before, pos)


def _tokens(tmp_path, text):
    """the prompt's token count, read off a cold request (the fixture's tokenizer is not one token a byte)"""
    srv, log = _server(tmp_path)
    with srv:
        tel, _ = _complete(srv, log, text, -1, cache_prompt=False)
    return tel["prompt_eval_tokens"]


def _cold(tmp_path, prompt, pos):
    srv, log = _server(tmp_path, pos)
    with srv:
        _, rows = _complete(srv, log, prompt, pos)
    assert rows, "the cold server printed no row at %d" % pos
    return rows


def test_a_normal_request_after_an_echo_keeps_none_of_its_rows(tmp_path):
    pos = 5
    cold = _cold(tmp_path, P + " far away", pos)
    srv, log = _server(tmp_path, pos)
    with srv:
        _complete(srv, log, P, pos, echo=True, logprobs=1)
        tel, rows = _complete(srv, log, P + " far away", pos)
    assert rows, ("the normal request did not write row %d: it kept the scored rows" % pos, tel)
    assert tel["prompt_cached_tokens"] == 0 and tel["prompt_reuse"] == "scored_rows", tel
    assert rows == cold, (rows, cold)


def test_after_a_decide_the_state_is_kept_and_the_scored_option_is_fed_again(tmp_path):
    state_tokens = _tokens(tmp_path, STATE)
    pos = state_tokens + 2                          # inside the option " jumps over"
    cold = _cold(tmp_path, FOLLOW, pos)
    srv, log = _server(tmp_path, pos)
    with srv:
        d = _post(srv, "/v1/decide", {"state": STATE, "rendering": "continuation-v1",
                                       "questions": [{"options": OPTIONS}]})
        assert d.get("questions") or d.get("results") or d, d
        tel, rows = _complete(srv, log, FOLLOW, pos)
    assert rows, ("the normal request did not write row %d: it kept the scored option's rows" % pos, tel)
    # the mark is where the scored tokens start, one before the option (its token is re-fed alone): the state's
    # own prefill rows below it are kept
    assert tel["prompt_cached_tokens"] == state_tokens - 1 and tel["prompt_reuse"] == "scored_rows", tel
    assert rows == cold, (rows, cold)
