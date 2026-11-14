"""Documented RUNNER_* controls the November deprecation review found no test for (R18.8.3).

docs/deprecation-review-2026-11.md counted seventeen controls MANUAL.md documents that no test read, so
any of them could stop working, or stop being read at all, with every gate green. Three are retired since:
CUDA_ATTN_MLA and CUDA_ATTN_TILE512 (R18.8.2), CUDA_ATTN_TC_FULL (R18.8.1). CUDA_GRAPH_OFF gained its
tests in R6.68.25. ATTN_BATCH is held in tests/test_attn_split.c, where the engine's own load reads it,
and RESPONSES_STORE_MB in tests/test_responses_store.py, beside the store's other tests. The rest are
here.

Each test sets the switch and sees the effect the manual documents, against a run without it: a counter,
a route detail, a warning, the tensor types written, the bytes generated. A switch the engine stopped
reading fails its test. Where the manual promises the switch leaves the output alone, the test also
holds the bytes equal across the two arms.

The CUDA switches (CUDA_GRAPH_RECUR, CUDA_KV_GATHER, CUDA_RECUR_SEQ) skip without a CUDA device, and the
Metal ones (METAL_MV2, METAL_TENSOR) without Metal.
"""
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gguf_fixture import read_header  # noqa: E402

# the switches these tests set: none leaks in from the host's environment
SWITCHES = ("RUNNER_TPOOL_SPIN", "RUNNER_GRAMMAR_FF", "RUNNER_SPEC_STATS", "RUNNER_SCHEMA_ALLOW_WS",
            "RUNNER_FORCE_REQUANT", "RUNNER_REQUANT_ONLY", "RUNNER_METAL_MV2", "RUNNER_METAL_TENSOR",
            "RUNNER_VRAM_PRIORITY", "RUNNER_CUDA_KV_GATHER", "RUNNER_CUDA_GRAPH_RECUR",
            "RUNNER_CUDA_RECUR_SEQ", "RUNNER_DEBUG_ROUTE")
GGML_Q4_0, GGML_Q8_0, GGML_Q4_K = 2, 8, 12
PROMPT = "the quick brown fox jumps over the lazy dog"
# fixed keys, so the grammar pins byte runs: {"colour_name":" and ,"count":
SCHEMA = {"type": "object",
          "properties": {"colour_name": {"type": "string", "maxLength": 6}, "count": {"type": "integer"}},
          "required": ["colour_name", "count"], "additionalProperties": False}


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    env.update(extra)
    return env


def _run(runner_bin, *args, timeout=300, **env):
    p = subprocess.run([str(runner_bin), *map(str, args)], cwd=ROOT, env=_env(**env),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-1500:]
    return p.stdout, p.stderr.decode(errors="replace")


def _gen(runner_bin, model, *extra, **env):
    return _run(runner_bin, "-m", model, "-p", PROMPT, "-n", "16", "--temp", "0", "--no-tray", *extra, **env)


def _find(obj, key):
    """Every value under `key`, anywhere in a decoded JSON document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find(v, key)


def _bench(runner_bin, model, *extra, **env):
    out, _ = _run(runner_bin, "-m", model, "--bench-json", "-n", "8", *extra, **env)
    return json.loads(out)


def _counter(bench, key):
    found = list(_find(bench, key))
    assert len(found) == 1, f"{key} should appear once in --bench-json, found {found}"
    return found[0]


def _backend(runner_bin):
    caps = subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE, timeout=60)
    return (json.loads(caps.stdout).get("gpu") or {}).get("backend") if caps.returncode == 0 else None


def _types(path):
    with open(path, "rb") as f:
        _kv, tensors, _end = read_header(f)
    return {name: ggml_type for name, _dims, ggml_type, _off in tensors}


@pytest.fixture(scope="module")
def fx(runner_bin, tmp_path_factory):
    """test.gguf (F32) and the wide fixture at Q4_K and Q6_K, built the way the Makefile builds
    test-q4k.gguf and test-q6k.gguf."""
    d = tmp_path_factory.mktemp("controls")
    src = d / "wide-src.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-model.py", "--wide", "--n-ff", "2048",
                    str(src)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    out = {"f32": ROOT / "test.gguf", "dir": d}
    assert out["f32"].exists(), "build the generated fixture with make test.gguf"
    for quant in ("q4_k", "q6_k"):
        out[quant] = d / f"wide-{quant}.gguf"
        _run(runner_bin, "-m", src, "--quantize", out[quant], "--quant", quant)
    (d / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    return out


# ---- CPU and engine


def test_tpool_spin_is_parsed_and_leaves_the_bytes(runner_bin, fx):
    """RUNNER_TPOOL_SPIN bounds how long a pool worker spins before parking; 0 is the pure condvar pool.
    The value is parsed strictly, and only scheduling changes: the same tokens either way."""
    base, _ = _gen(runner_bin, fx["f32"], "--gpu", "off", "-t", "4")
    _, err = _gen(runner_bin, fx["f32"], "--gpu", "off", "-t", "4", RUNNER_TPOOL_SPIN="abc")
    assert "RUNNER_TPOOL_SPIN='abc' is not an integer" in err, err[-800:]
    out, err = _gen(runner_bin, fx["f32"], "--gpu", "off", "-t", "4", RUNNER_TPOOL_SPIN="0")
    assert "RUNNER_TPOOL_SPIN" not in err, err[-800:]
    assert out == base


def _grammar_drafts(err):
    m = re.search(r"^spec: .*, grammar (\d+)/(\d+),", err, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def test_grammar_ff_drafts_pinned_bytes_and_keeps_the_output(runner_bin, fx):
    """RUNNER_GRAMMAR_FF=1 proposes the byte runs a schema pins as draft tokens for the batched verify.
    The schema's fixed keys are such runs: with the switch some are drafted, without it none, and the
    document is the same bytes."""
    schema = fx["dir"] / "schema.json"
    args = ("--json-schema", schema, "--gpu", "off", "-n", "40")
    off, err_off = _gen(runner_bin, fx["f32"], *args, RUNNER_SPEC_STATS="1")
    on, err_on = _gen(runner_bin, fx["f32"], *args, RUNNER_SPEC_STATS="1", RUNNER_GRAMMAR_FF="1")
    assert _grammar_drafts(err_off) == (0, 0), err_off[-800:]
    accepted, drafted = _grammar_drafts(err_on)
    assert drafted > 0 and accepted > 0, err_on[-800:]
    assert on == off
    json.loads(on)   # the constrained document is complete either way


def test_schema_allow_ws_restores_the_whitespace_runs(runner_bin, fx):
    """Under a schema a token of nothing but insignificant whitespace is refused (an unsure model emits
    runs of it until -n runs out). RUNNER_SCHEMA_ALLOW_WS=1 restores the old behaviour for an A/B: on
    this fixture the old arm pads the object with newlines, and the default arm writes none."""
    args = ("--json-schema", fx["dir"] / "schema.json", "--gpu", "off", "-n", "40")
    fixed, _ = _gen(runner_bin, fx["f32"], *args)
    old, _ = _gen(runner_bin, fx["f32"], *args, RUNNER_SCHEMA_ALLOW_WS="1")
    doc = fixed.decode().strip()
    assert json.dumps(json.loads(doc), separators=(",", ":")) == doc, doc
    assert re.search(r"\s{4,}", old.decode()), old[:200]


# ---- quantize


def test_force_requant_converts_a_type_that_is_not_smaller(runner_bin, fx, tmp_path):
    """A conversion to a type no smaller than the one a tensor carries (Q4_K to Q4_0) is declined and
    said so; RUNNER_FORCE_REQUANT=1 converts anyway, for diagnosing a quant type."""
    kept, forced = tmp_path / "kept.gguf", tmp_path / "forced.gguf"
    _, err = _run(runner_bin, "-m", fx["q4_k"], "--quantize", kept, "--quant", "q4_0")
    assert "kept their own type" in err and "RUNNER_FORCE_REQUANT=1" in err, err[-800:]
    _, err_f = _run(runner_bin, "-m", fx["q4_k"], "--quantize", forced, "--quant", "q4_0",
                    RUNNER_FORCE_REQUANT="1")
    assert "kept their own type" not in err_f, err_f[-800:]
    src, k, f = _types(fx["q4_k"]), _types(kept), _types(forced)
    q4k = {n for n, t in src.items() if t == GGML_Q4_K}
    assert q4k, src
    assert {n for n, t in k.items() if t == GGML_Q4_K} == q4k
    assert {n for n, t in f.items() if t == GGML_Q4_0} == q4k


def test_requant_only_scopes_the_conversion_to_matching_names(runner_bin, fx, tmp_path):
    """RUNNER_REQUANT_ONLY=SUBSTR converts only the tensors whose name contains SUBSTR; every other
    tensor keeps its type. The whole-file run beside it shows the others would have converted."""
    whole, scoped = tmp_path / "whole.gguf", tmp_path / "scoped.gguf"
    _run(runner_bin, "-m", fx["f32"], "--quantize", whole, "--quant", "q8_0")
    _run(runner_bin, "-m", fx["f32"], "--quantize", scoped, "--quant", "q8_0", RUNNER_REQUANT_ONLY="attn_q")
    src, w, s = _types(fx["f32"]), _types(whole), _types(scoped)
    assert any(t == GGML_Q8_0 and "attn_q" not in n for n, t in w.items()), w
    converted = {n for n, t in s.items() if t == GGML_Q8_0}
    assert converted and all("attn_q" in n for n in converted), converted
    assert all(s[n] == src[n] for n in s if n not in converted)


# ---- the VRAM registry


def _ledger_line(ledger, pid, timeout=20):
    """The registry line of `pid`, read while the load holds its claim open. The runner takes a
    mandatory lock over the file for its read-modify-write on Windows, so a read that lands inside one
    fails with PermissionError and is retried."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            text = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
        except PermissionError:
            text = ""
        for line in text.splitlines():
            f = line.split("\t")
            if f and f[0] == str(pid):
                return f
        time.sleep(0.05)
    raise AssertionError(f"no registry line for pid {pid} in {ledger}")


@pytest.mark.parametrize("env_value,flag,want", [(None, None, "0"), ("7", None, "7"), ("7", "3", "3")])
def test_vram_priority_is_the_flags_baseline(runner_bin, fx, tmp_path, env_value, flag, want):
    """RUNNER_VRAM_PRIORITY is the baseline of --vram-priority, the advisory tag on a VRAM claim, and the
    flag overrides it. RUNNER_TEST_VRAM_FREE_MB stands in for a device's free memory, so the real claim
    runs on any host against a real ledger; the claim's last field is its priority."""
    env = _env(RUNNER_TEST_VRAM_FREE_MB="100000", RUNNER_VRAM_REGISTRY_DIR=str(tmp_path),
               RUNNER_TEST_LOAD_DELAY_MS="3000")
    if env_value is not None:
        env["RUNNER_VRAM_PRIORITY"] = env_value
    args = [str(runner_bin), "-m", str(fx["f32"]), "-p", "hi", "-n", "1", "--temp", "0", "--no-tray"]
    if flag is not None:
        args += ["--vram-priority", flag]
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        line = _ledger_line(tmp_path / "xyntetik-vram-test-device.reg", p.pid)
    finally:
        _, err = p.communicate(timeout=120)
    assert p.returncode == 0, err.decode(errors="replace")[-800:]
    assert len(line) == 8 and line[7] == want, line


# ---- Metal


def _metal_or_skip(runner_bin):
    if _backend(runner_bin) != "metal":
        pytest.skip("needs the Metal backend")


def test_metal_mv2_off_keeps_the_one_row_kernel(runner_bin, fx):
    """A one-column Q6_K decode matvec takes the two-row kernel k_mv2_q6_K once its load-time self-test
    admits it; RUNNER_METAL_MV2=0 keeps the one-row kernel. The two are bit-exact, so the tokens match."""
    _metal_or_skip(runner_bin)
    assert _counter(_bench(runner_bin, fx["q6_k"], "--gpu", "auto"), "metal_mv2") > 0
    assert _counter(_bench(runner_bin, fx["q6_k"], "--gpu", "auto", RUNNER_METAL_MV2="0"), "metal_mv2") == 0
    base, _ = _gen(runner_bin, fx["q6_k"], "--gpu", "auto")
    off, _ = _gen(runner_bin, fx["q6_k"], "--gpu", "auto", RUNNER_METAL_MV2="0")
    assert off == base


def _tensor_device(runner_bin):
    """The device gate of RUNNER_METAL_TENSOR: an Apple M5 or M6 on macOS 26.2 or newer."""
    caps = json.loads(subprocess.run([str(runner_bin), "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
                                     timeout=60).stdout)
    name = (caps.get("gpu") or {}).get("name") or ""
    ver = tuple(int(x) for x in (platform.mac_ver()[0] or "0").split(".")[:2] + ["0"])[:2]
    return name.startswith(("Apple M5", "Apple M6")) and ver >= (26, 2)


def test_metal_tensor_is_an_opt_in_behind_its_device_gate(runner_bin, fx):
    """RUNNER_METAL_TENSOR=1 opts Q4_K, Q8_0 and Q4_0 prefill into the Metal 4 tensor GEMM on an M5 or
    newer with macOS 26.2; it is parked off by default, and 0 is the explicit pin. The route detail
    names it. Where the device gate declines it, the load says so and the opt-in changes nothing else,
    bytes included."""
    _metal_or_skip(runner_bin)
    declined = "Metal 4 tensor GEMM requires M5+ and macOS 26.2+"

    def detail(**env):
        return _counter(_bench(runner_bin, fx["q4_k"], "--gpu", "auto", **env), "backend_detail")

    assert "tensor=off" in detail()
    assert "tensor=off" in detail(RUNNER_METAL_TENSOR="0")
    base, err = _gen(runner_bin, fx["q4_k"], "--gpu", "auto")
    _, err_pin = _gen(runner_bin, fx["q4_k"], "--gpu", "auto", RUNNER_METAL_TENSOR="0")
    assert declined not in err and declined not in err_pin, (err[-600:], err_pin[-600:])
    if _tensor_device(runner_bin):
        assert "tensor=on" in detail(RUNNER_METAL_TENSOR="1")
    else:
        assert "tensor=off" in detail(RUNNER_METAL_TENSOR="1")
        asked, err_asked = _gen(runner_bin, fx["q4_k"], "--gpu", "auto", RUNNER_METAL_TENSOR="1")
        assert declined in err_asked, err_asked[-600:]
        assert asked == base


# ---- CUDA


def _cuda_or_skip(runner_bin):
    if _backend(runner_bin) != "cuda":
        pytest.skip("needs a CUDA device")


def test_cuda_kv_gather_off_keeps_the_per_layer_copies(runner_bin, fx):
    """A decoded token's KV rows come back to the host cache in one gather (X3); RUNNER_CUDA_KV_GATHER=0
    keeps the per-layer copies. The same bytes land in the same host slots, so the tokens match."""
    _cuda_or_skip(runner_bin)
    assert _counter(_bench(runner_bin, fx["f32"], "--gpu", "auto"), "cuda_kv_gather") > 0
    assert _counter(_bench(runner_bin, fx["f32"], "--gpu", "auto", RUNNER_CUDA_KV_GATHER="0"),
                    "cuda_kv_gather") == 0
    base, _ = _gen(runner_bin, fx["f32"], "--gpu", "auto")
    off, _ = _gen(runner_bin, fx["f32"], "--gpu", "auto", RUNNER_CUDA_KV_GATHER="0")
    assert off == base


@pytest.fixture(scope="module")
def qwen35(runner_bin, tmp_path_factory):
    _cuda_or_skip(runner_bin)
    path = tmp_path_factory.mktemp("controls-qwen35") / "qwen35.gguf"
    subprocess.run([sys.executable, ROOT / "scripts" / "make-test-ornith.py", str(path)], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _routes(err):
    return [line for line in err.splitlines() if line.startswith("route:")]


def test_cuda_graph_recur_off_keeps_a_qwen35_decode_eager(runner_bin, qwen35):
    """A qwen35 one-token decode replays a captured CUDA graph; RUNNER_CUDA_GRAPH_RECUR=0 keeps it
    eager, an A/B control. The decode route says which ran, and the tokens match."""
    args = ("--gpu", "auto", "-b", "8", "-t", "2")
    base, err = _gen(runner_bin, qwen35, *args, RUNNER_DEBUG_ROUTE="1")
    eager, err_e = _gen(runner_bin, qwen35, *args, RUNNER_DEBUG_ROUTE="1", RUNNER_CUDA_GRAPH_RECUR="0")
    decode = [r for r in _routes(err) if " rows=1 " in r]
    decode_e = [r for r in _routes(err_e) if " rows=1 " in r]
    assert decode and "graph=replay" in decode[-1], decode[-1:] or err[-1500:]
    assert decode_e and "graph=replay" not in decode_e[-1], decode_e[-1:]
    assert eager == base


def test_cuda_recur_seq_off_keeps_the_per_token_loop(runner_bin, qwen35):
    """A qwen35 prompt tile's recurrent layers run as whole-tile kernels; RUNNER_CUDA_RECUR_SEQ=0 keeps
    the per-token loop. The prompt's route says recur=tile only with the kernels on, and the tokens
    match."""
    args = ("--gpu", "auto", "-b", "8", "-t", "2")
    base, err = _gen(runner_bin, qwen35, *args, RUNNER_DEBUG_ROUTE="1")
    loop, err_l = _gen(runner_bin, qwen35, *args, RUNNER_DEBUG_ROUTE="1", RUNNER_CUDA_RECUR_SEQ="0")
    assert any(" recur=tile" in r for r in _routes(err)), _routes(err)[-3:] or err[-1500:]
    assert not any(" recur=tile" in r for r in _routes(err_l)), _routes(err_l)[-3:]
    assert loop == base
