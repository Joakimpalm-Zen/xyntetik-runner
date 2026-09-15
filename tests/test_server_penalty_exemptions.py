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


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


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
