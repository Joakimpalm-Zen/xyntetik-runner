"""adapter.lora.alpha is part of the adapter's identity.

The prefix cache keys its snapshots on the engine's model identity, which
folds in the adapter id. That id hashed the adapter path, its tensor bytes
and --lora-scale, but not alpha, although alpha scales every delta. After a
TTL unload the server reloads the adapter from its path; an adapter file
rewritten in between with a different alpha therefore hit the old snapshot
and served rows computed under the old scaling (alpha 8 -> 800 reused the
whole prompt, logprobs off by ~0.018 from a fresh run; found by an outside
review, 2026-09-05).

Gate: serve with the alpha-8 adapter, request logprobs, rewrite the same
path with an alpha-800 adapter built from the same seed, wait for the TTL
reload, request again. The reload must reuse nothing and its logprobs must
equal a fresh server started on the alpha-800 file exactly. A control
confirms the two adapters do produce different logprobs, or the gate could
pass on an adapter that does nothing.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

PROMPT = " ".join(["the quick brown fox jumps over the lazy dog"] * 3)
BODY = {"prompt": PROMPT, "temperature": 0, "max_tokens": 6, "logprobs": 1}


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    base = ROOT / "test.gguf"
    if not base.exists():
        pytest.skip("test.gguf not built")
    d = tmp_path_factory.mktemp("alpha")
    out = {}
    for alpha in ("8", "800"):
        env = dict(os.environ)
        env["LORA_ALPHA"] = alpha
        subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                        str(base), str(d / f"a{alpha}")], check=True,
                       cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        out[alpha] = d / f"a{alpha}.adapter.gguf"
    return base, out


def _post(base_url, path, body):
    req = urllib.request.Request(
        base_url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _resident(base_url):
    with urllib.request.urlopen(base_url + "/health", timeout=10) as r:
        return json.load(r).get("resident")


def _logprobs(r):
    return r["choices"][0]["logprobs"]["token_logprobs"]


def test_alpha_change_invalidates_cached_prefixes(runner_bin, adapters, tmp_path):
    base, ad = adapters
    live = tmp_path / "adapter.gguf"
    shutil.copy(ad["8"], live)
    with RunnerServer(runner_bin, base, ctx=256,
                      extra_args=["--gpu", "off", "--lora", str(live),
                                  "--ttl", "1"]) as srv:
        before = _logprobs(_post(srv.base_url, "/v1/completions", BODY))
        shutil.copy(ad["800"], live)
        deadline = time.monotonic() + 30
        while _resident(srv.base_url) is not None:
            assert time.monotonic() < deadline, "TTL never unloaded"
            time.sleep(0.5)
        r = _post(srv.base_url, "/v1/completions", BODY)
        after = _logprobs(r)
        cached = r["usage"]["prompt_tokens_details"]["cached_tokens"]
    with RunnerServer(runner_bin, base, ctx=256,
                      extra_args=["--gpu", "off", "--lora", str(ad["800"])]) as srv:
        fresh = _logprobs(_post(srv.base_url, "/v1/completions", BODY))
    assert before != fresh, "alpha 8 and 800 must differ or the gate is blind"
    assert cached == 0, f"reload reused {cached} prompt tokens across an alpha change"
    assert after == fresh, (after, fresh)
