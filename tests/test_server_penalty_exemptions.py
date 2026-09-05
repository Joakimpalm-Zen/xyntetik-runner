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
"""
import json
import pathlib
import sys
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


def _complete(base_url):
    body = {"prompt": "</s>", "temperature": 0.01, "top_k": 1,
            "repeat_penalty": 100, "max_tokens": 4}
    req = urllib.request.Request(
        base_url + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
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
