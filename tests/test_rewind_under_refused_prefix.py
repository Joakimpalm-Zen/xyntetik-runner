"""The slot still rewinds when the SHARED prefix tier is refused.

A KV ring (RUNNER_KV_RING=1) and tied-V (RUNNER_TIEDV=1) both refuse the
shared prefix cache: a ring's rows are recycled, and a tied-V K cache is
smaller than the flat copy expects. That refusal used to return before the
slot's own engine_rewind, which is the one call between two requests that
resets the engine's position, so with the default cache_prompt:true the
next prompt was appended to the previous one's context: identical prompts
answered differently, and the third request overflowed a 128-token context
(found by an outside review, 2026-09-05; cache_prompt:false was the only
workaround).

Gate: three identical greedy requests on a context sized so that two fit
but three appended ones do not. All three must succeed with the same text,
and the third must report the same prompt length as the first.
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

PROMPT = "the quick brown fox jumps over the lazy dog"  # ~59 fixture tokens
N_CTX = 160  # 59 + 8 fits twice, not three times, when requests append


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _make(tmp_path_factory, name, extra):
    p = tmp_path_factory.mktemp(name) / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(p), *extra], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return p


@pytest.fixture(scope="module")
def ring_model(tmp_path_factory):
    return _make(tmp_path_factory, "ringrewind", ["--swa", "32,2"])


@pytest.fixture(scope="module")
def tiedv_model(tmp_path_factory):
    return _make(tmp_path_factory, "tiedvrewind", ["--gemma4-hetero"])


def _complete(base_url):
    body = {"prompt": PROMPT, "temperature": 0, "max_tokens": 8}
    req = urllib.request.Request(
        base_url + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _three_identical(runner_bin, model, env_key):
    env = dict(os.environ)
    env[env_key] = "1"
    with RunnerServer(runner_bin, model, ctx=N_CTX, env=env,
                      extra_args=["--gpu", "off"]) as srv:
        results = [_complete(srv.base_url) for _ in range(3)]
    for st, r in results:
        assert st == 200, r
    texts = [r["choices"][0]["text"] for _, r in results]
    assert texts[0] == texts[1] == texts[2], texts
    lengths = [r["usage"]["prompt_tokens"] for _, r in results]
    assert lengths[0] == lengths[1] == lengths[2], lengths
    # the setup must be the one the bug needs: appending three would overflow
    assert 3 * (lengths[0] + 8) > N_CTX >= 2 * (lengths[0] + 8), lengths


def test_ring_slot_rewinds_between_requests(runner_bin, ring_model):
    _three_identical(runner_bin, ring_model, "RUNNER_KV_RING")


def test_tiedv_slot_rewinds_between_requests(runner_bin, tiedv_model):
    _three_identical(runner_bin, tiedv_model, "RUNNER_TIEDV")
