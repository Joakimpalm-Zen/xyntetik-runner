"""The host paths that restore KV rows run on the Apple GPU's growing cache (R3.68, codex's read of PR G).

The gate arm that ran the prefix cache, named-context and session tests at a 16-row step ran most of
them with --gpu off, so no Metal restore was exercised. These run on the Apple GPU at a 16-row step
with RUNNER_GPU_FALLBACK=fail, each restoring more rows than the first step maps: a preamble snapshot
and a named-context snapshot loaded into a fresh process (whose cache maps one step, so the restore
lands above the mapped mark and the next forward must grow past it), and a prefix-cache fork in one
process. Each answer is compared with a cold prefill's.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

STEP = 16
SYSTEM = ("You are a coding agent working in a repository. Read before you edit, keep edits small, run "
          "the tests after a change and report what you did. The tools are read_file, write_file, "
          "edit_file, search and run.")


@pytest.fixture(scope="module")
def model(runner_bin, tmp_path_factory):
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
                                     timeout=60).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("the Metal backend's growing KV cache")
    m = tmp_path_factory.mktemp("restores") / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return m


def _serve(runner_bin, model, tmp_path, *extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_TEST_METAL_KV_")}
    env.update(RUNNER_TEST_METAL_KV_STEP=str(STEP), RUNNER_GPU_FALLBACK="fail")
    log = tmp_path / ("server-%d.log" % time.monotonic_ns())
    return RunnerServer(runner_bin, model, ctx=512, env=env, log_path=str(log),
                        extra_args=["-t", "2", "--chat-template", "chatml", *extra]), log


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _chat(srv, user, **extra):
    st, body = _post(srv, "/v1/chat/completions", {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "max_tokens": 8, "temperature": 0, **extra})
    assert st == 200, body
    t = body["runner_telemetry"]
    return body["choices"][0]["message"]["content"], t["prompt_cached_tokens"], t["prompt_reuse"]


def _on_metal(log):
    text = log.read_text(errors="replace")
    assert "gpu: Metal backend" in text and "%d rows at a time" % STEP in text, text[-800:]
    assert "continuing on the CPU" not in text, text[-800:]


def test_a_preamble_snapshot_restored_into_a_new_process_grows_past_it(runner_bin, model, tmp_path):
    snaps = tmp_path / "snaps"
    a, log_a = _serve(runner_bin, model, tmp_path, "--kv-snapshots", str(snaps), "--kv-snapshots-auto", "64")
    with a as srv:
        _chat(srv, "List the files.")
        for _ in range(200):
            if any(p.is_file() and (snaps / (p.name + ".json")).exists() for p in snaps.glob("pre-*.kv")):
                break
            time.sleep(0.1)
    b, log_b = _serve(runner_bin, model, tmp_path, "--kv-snapshots", str(snaps), "--kv-snapshots-auto", "64")
    with b as srv:
        warm, cached, reuse = _chat(srv, "Which file is the largest?")
    assert reuse == "preamble_snapshot" and cached > STEP, (reuse, cached)
    c, log_c = _serve(runner_bin, model, tmp_path)
    with c as srv:
        cold, cached_c, _ = _chat(srv, "Which file is the largest?", cache_prompt=False)
    assert cached_c == 0 and warm == cold
    for log in (log_a, log_b, log_c):
        _on_metal(log)


def test_a_named_context_snapshot_loaded_into_a_new_process_grows_past_it(runner_bin, model, tmp_path):
    snaps = tmp_path / "snaps"
    a, log_a = _serve(runner_bin, model, tmp_path, "--kv-snapshots", str(snaps))
    with a as srv:
        st, ctx = _post(srv, "/v1/runner/contexts", {"id": "mem", "prompt": SYSTEM})
        assert st == 200 and ctx["tokens"] > STEP, ctx
        st, snap = _post(srv, "/v1/runner/contexts/mem/snapshot", {})
        assert st == 200, snap
        st, cold = _post(srv, "/v1/completions", {"prompt": SYSTEM + " Where is item 42?", "max_tokens": 8,
                                                  "temperature": 0, "cache_prompt": False})
        assert st == 200, cold
    b, log_b = _serve(runner_bin, model, tmp_path, "--kv-snapshots", str(snaps))
    with b as srv:
        st, loaded = _post(srv, "/v1/runner/contexts", {"id": "mem2", "snapshot": "mem"})
        assert st == 200 and loaded["tokens"] == ctx["tokens"], loaded
        st, warm = _post(srv, "/v1/completions", {"prompt": SYSTEM + " Where is item 42?", "max_tokens": 8,
                                                  "temperature": 0, "context_id": "mem2"})
        assert st == 200, warm
    assert warm["runner_telemetry"]["prompt_cached_tokens"] == ctx["tokens"] > STEP
    assert warm["choices"][0]["text"] == cold["choices"][0]["text"]
    for log in (log_a, log_b):
        _on_metal(log)


def test_a_prefix_cache_fork_in_one_process(runner_bin, model, tmp_path):
    """Within one slot the restored rows are ones the slot has already mapped (it prefilled them); the
    fork still restores more rows than one step, on the Apple GPU."""
    a, log = _serve(runner_bin, model, tmp_path)
    with a as srv:
        cold, _, _ = _chat(srv, "Where is item 42?", cache_prompt=False)
        _chat(srv, "Where is item 7?")
        assert _post(srv, "/v1/completions", {"prompt": "unrelated text that shares nothing",
                                              "max_tokens": 4, "temperature": 0})[0] == 200
        warm, cached, reuse = _chat(srv, "Where is item 42?")
    assert reuse == "prefix_cache" and cached > STEP, (reuse, cached)
    assert warm == cold
    _on_metal(log)
