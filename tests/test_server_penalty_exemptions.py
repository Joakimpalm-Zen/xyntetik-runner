"""The repeat penalty never punishes the model's own turn terminator over HTTP.

engine_init installs the stop tokens as repeat-penalty exemptions in the
slot's live sampler, because a chat template puts the terminator in the
prompt and the prompt seeds the penalty window. The server captured its
request-visible defaults BEFORE that, and restored the whole struct at the
start of every request, so the exemptions were dropped on the first request:
prompt "</s>", top-k 1, penalty 100 stopped at once on the CLI and ran to
max_tokens over HTTP (found by an outside review, 2026-09-05).

Gate: the same request over HTTP stops with no completion tokens, and a
second request after it does too (the restore runs every request).

The 2026-09-14 remedy plan asked for the exemptions to be audited across
request reset, several slots, a model reload, cache reuse and speculative
generation, and for no punctuation to be exempt by fiat. What is exempt is
exactly the engine's stop ids (eos plus the family turn terminators, all
derived from the tokenizer at engine_init), nothing else, on every path:
each slot's engine installs its own set (`--parallel`), a reload after
`keep_alive: 0` re-derives it from the new tokenizer, a prefix fork keeps
the sampler's exemptions while it re-seeds the penalty window, and the
speculative walk scores the target's logits through the same sampler. The
tests below pin the slot and reload legs over HTTP; the fork and the
speculative walk are pinned by `tests/test_prefix.c` and the spec identity
gates, which decode through the same `sample_pick`.

The penalty window has held generated tokens only since the prompt stopped
seeding it, so the first gate (stop at once on the prompt "</s>") passes
with the exemption switched off in the sampler: nothing is in the window
when the first token is picked (measured 2026-11-12, R6.68.52). Each test
therefore also runs the request with ignore_eos, where the terminator is
generated, enters the window and must be picked again: exempt, the four
tokens are the terminator four times; not exempt, the penalty of 100
moves the second pick to another token ([2, 104, 207, 124] on test.gguf).
"""
import json
import pathlib
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402


def _complete(base_url, **fields):
    body = {"prompt": "</s>", "temperature": 0.01, "top_k": 1,
            "repeat_penalty": 100, "max_tokens": 4}
    body.update(fields)
    req = urllib.request.Request(
        base_url + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(base_url, path):
    with urllib.request.urlopen(base_url + path, timeout=30) as r:
        return json.load(r)


def _terminator_stays_exempt(base_url):
    """The terminator, generated under ignore_eos, is picked again each step."""
    r = _complete(base_url, ignore_eos=True, logprobs=1)
    ids = r["choices"][0]["logprobs"]["token_ids"]
    assert len(ids) == 4 and len(set(ids)) == 1, ids


def test_stop_token_is_exempt_from_the_repeat_penalty_over_http(runner_bin):
    model = ROOT / "test.gguf"
    if not model.exists():
        pytest.skip("test.gguf not built")
    with RunnerServer(runner_bin, model, ctx=256,
                      extra_args=["--gpu", "off"]) as srv:
        for _ in range(2):
            r = _complete(srv.base_url)
            assert r["choices"][0]["finish_reason"] == "stop", r
            assert r["usage"]["completion_tokens"] == 0, r["usage"]
            _terminator_stays_exempt(srv.base_url)


def test_every_slot_keeps_the_exemptions(runner_bin):
    """Two slots each install their own set at their own engine_init."""
    model = ROOT / "test.gguf"
    if not model.exists():
        pytest.skip("test.gguf not built")
    with RunnerServer(runner_bin, model, ctx=256, parallel=2,
                      extra_args=["--gpu", "off"]) as srv:
        # four requests in a row land on both slots (the scheduler hands a
        # free slot to each), every one must stop at once
        for _ in range(4):
            r = _complete(srv.base_url)
            assert r["choices"][0]["finish_reason"] == "stop", r
            assert r["usage"]["completion_tokens"] == 0, r["usage"]
            _terminator_stays_exempt(srv.base_url)


def test_a_reloaded_model_keeps_the_exemptions(runner_bin):
    """After a keep_alive:0 release (single slot: the registry path) the
    reloaded engine derives the set again from the tokenizer."""
    model = ROOT / "test.gguf"
    if not model.exists():
        pytest.skip("test.gguf not built")
    with RunnerServer(runner_bin, model, ctx=256, parallel=1,
                      extra_args=["--gpu", "off"]) as srv:
        r = _complete(srv.base_url, keep_alive=0)
        assert r["usage"]["completion_tokens"] == 0, r["usage"]
        deadline = time.monotonic() + 15
        while _get(srv.base_url, "/v1/capabilities")["resident"] is not None:
            assert time.monotonic() < deadline, "model did not unload"
            time.sleep(0.1)
        for _ in range(2):
            r = _complete(srv.base_url)
            assert r["choices"][0]["finish_reason"] == "stop", r
            assert r["usage"]["completion_tokens"] == 0, r["usage"]
            _terminator_stays_exempt(srv.base_url)


def _session(base_url, **fields):
    req = urllib.request.Request(
        base_url + "/v1/runner/sessions", data=json.dumps(fields).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def test_a_session_keeps_the_exemptions(runner_bin, tmp_path):
    """A session request set the slot's sampler from the pristine defaults
    wholesale, the restore the completion path had stopped doing: the
    session itself ran without the exemptions, and so did every request on
    that slot after it, until the model was reloaded."""
    model = ROOT / "test.gguf"
    if not model.exists():
        pytest.skip("test.gguf not built")
    with RunnerServer(runner_bin, model, ctx=256, parallel=1, extra_args=[
            "--gpu", "off", "--sessions", str(tmp_path / "s")]) as srv:
        _terminator_stays_exempt(srv.base_url)
        # the session's own generation: four terminators decode to no text,
        # any other token to some
        s = _session(srv.base_url, prompt="</s>", max_tokens=4, top_k=1,
                     temperature=0.01, repeat_penalty=10, ignore_eos=True)
        assert s["generated"] == 4 and s["text"] == "", s
        # and the slot's next request has the exemptions too
        _terminator_stays_exempt(srv.base_url)
