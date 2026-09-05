"""TTL expiry releases the draft model with the target.

POST /unload and shutdown free the draft before the resident model; the idle
reaper called unload_resident() alone, so a --ttl expiry reported no
resident model while the draft's weights and KV stayed mapped. Proof from
the outside review (2026-09-05): delete the draft file after the expiry and
the reload still drafted, so the old allocation had survived.

Gate: the same shape. After the TTL unload the draft file is deleted; the
next request reloads the target, cannot reload the draft, and must report
speculative decoding OFF. Skipped on Windows, where a mapped file cannot be
unlinked, which is the very property the gate relies on elsewhere.
"""
import json
import pathlib
import shutil
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


def _chat(base_url):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    req = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(base_url, path):
    with urllib.request.urlopen(base_url + path, timeout=10) as r:
        return json.load(r)


@pytest.mark.skipif(sys.platform == "win32", reason="cannot unlink a mapped file")
def test_ttl_expiry_unloads_the_draft(runner_bin, tmp_path):
    model = ROOT / "test.gguf"
    if not model.exists():
        pytest.skip("test.gguf not built")
    draft = tmp_path / "draft.gguf"
    shutil.copy(model, draft)
    with RunnerServer(runner_bin, model, ctx=256,
                      extra_args=["--gpu", "off", "--draft", str(draft),
                                  "--ttl", "1"]) as srv:
        assert _chat(srv.base_url)["runner_telemetry"]["speculative"] is True
        deadline = time.monotonic() + 30
        while _get(srv.base_url, "/health").get("resident") is not None:
            assert time.monotonic() < deadline, "TTL never unloaded"
            time.sleep(0.5)
        draft.unlink()
        r = _chat(srv.base_url)
        assert r["runner_telemetry"]["speculative"] is False, r["runner_telemetry"]
        caps = _get(srv.base_url, "/v1/capabilities")["draft"]
        assert caps["requested"] is True and caps["active"] is False, caps
