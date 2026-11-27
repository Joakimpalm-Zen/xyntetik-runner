"""R4.48.8.5: Inkling 2 Mini's --kv q8 cache and its fit, on the tiny inkling2_loop file (tests/inkling2_tiny.py).

Gate (b): q8 engages, and it keeps every bit invariance the float32 cache holds (thread counts, prompt chunk sizes),
asserted pair by pair against what the float32 cache shows. The house's other formats (f16, k8v4, fp4) are not this
engine's: an asked one keeps f32 with a note. Gate (a), bit for bit against the float32 cache over the round-tripped
rows, is tests/test_inkling2_kvq8.c.

The fit: --info fit counts the cache the load allocates, byte for byte, for the kind --kv asks, every slot --parallel
asks, and the sliding applications by the file's own pattern.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling2_tiny import FULL, H, HD, K, LOOP_END, LOOP_START, N_BLOCK, NKV, PASSES, WIN, _kvs, _tensors, _write  # noqa: E402

# 48 byte tokens: six windows of 8, so the sliding buffers slide
PROMPT = "the q8 cache slides past the window, chunk by chunk"
CTX = 64


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("ink2kv")
    return _write(d / "tiny.gguf", _kvs(), _tensors())


def _run(runner_bin, model, *args):
    return subprocess.run([str(runner_bin), "-m", str(model), "--gpu", "off", "-c", str(CTX), *map(str, args)],
                          capture_output=True, encoding="utf-8", errors="replace", timeout=600)


def _score(runner_bin, model, *extra, threads=2, batch=8):
    r = _run(runner_bin, model, "-t", threads, "-b", batch, "--score", "1", "-p", PROMPT, *extra)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)["logprobs"], r.stderr


def test_q8_engages(runner_bin, tiny):
    f32, _ = _score(runner_bin, tiny)
    q8, err = _score(runner_bin, tiny, "--kv", "q8")
    assert "K/V q8" in err and "keeps f32" not in err, err[-2000:]
    assert q8 != f32, "--kv q8 changed no score: it did not engage"


CONFIGS = {"t1 b8": (1, 8), "t4 b8": (4, 8), "t2 b1": (2, 1), "t2 b8": (2, 8)}


def test_q8_keeps_every_bit_invariance_the_f32_cache_holds(runner_bin, tiny):
    f32 = {n: _score(runner_bin, tiny, threads=t, batch=b)[0] for n, (t, b) in CONFIGS.items()}
    q8 = {n: _score(runner_bin, tiny, "--kv", "q8", threads=t, batch=b)[0] for n, (t, b) in CONFIGS.items()}
    names = list(CONFIGS)
    held = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if f32[a] == f32[b]:
                held += 1
                assert q8[a] == q8[b], f"the float32 cache gives {a} and {b} the same bits; q8 does not"
    assert held > 0, "the float32 cache held no invariance here: the test would assert nothing"


@pytest.mark.parametrize("kv", ["fp4", "k8v4", "f16"])
def test_another_format_keeps_f32_with_a_note(runner_bin, tiny, kv):
    f32, _ = _score(runner_bin, tiny)
    got, err = _score(runner_bin, tiny, "--kv", kv)
    assert f"--kv {kv} is not one of its formats, so it keeps f32" in err, err[-2000:]
    assert got == f32


def _bytes(text, where):
    m = re.search(r"cache\s+[0-9.]+ Gi?B \((\d+) bytes", text)
    assert m, f"no exact cache bytes in the {where}:\n{text[-2000:]}"
    return int(m.group(1))


def _fit(runner_bin, model, *extra):
    r = _run(runner_bin, model, "--info", "fit", *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _apps():
    """(global, sliding) applications of the tiny file: its blocks with the loop's passes."""
    blocks = list(range(LOOP_START)) + list(range(LOOP_START, LOOP_END)) * PASSES + list(range(LOOP_END, N_BLOCK))
    g = sum(1 for b in blocks if FULL[b])
    return g, len(blocks) - g


def _expected(kv, batch, slots=1):
    C = NKV * HD
    row = C // 32 * 34 if kv == "q8" else C * 4
    g, sl = _apps()
    local = min(CTX, WIN - 1 + batch)
    conv = (g + sl) * 4 * (K - 1) * max(C, H) * 4
    return slots * (2 * g * CTX * row + 2 * sl * local * row + conv)


@pytest.mark.parametrize("kv", ["f32", "q8"])
@pytest.mark.parametrize("batch", [1, 8])
def test_the_fit_counts_the_cache_the_load_allocates(runner_bin, tiny, kv, batch):
    fit = _bytes(_fit(runner_bin, tiny, "--kv", kv, "-b", batch), "fit")
    r = _run(runner_bin, tiny, "--kv", kv, "-b", batch, "-t", 2, "-p", "hi", "-n", 1, "--temp", 0)
    assert r.returncode == 0, r.stderr[-2000:]
    assert fit == _bytes(r.stderr, "load") == _expected(kv, batch)


def test_the_fit_names_the_sliding_applications_by_the_files_pattern(runner_bin, tiny):
    _, sl = _apps()
    out = _fit(runner_bin, tiny)
    assert f"the {sl} sliding ones" in out, out


def test_the_fit_counts_every_slot(runner_bin, tiny):
    one = _bytes(_fit(runner_bin, tiny, "--kv", "q8", "-b", 8), "fit")
    two = _bytes(_fit(runner_bin, tiny, "--kv", "q8", "-b", 8, "--parallel", 2), "fit")
    assert two == 2 * one == _expected("q8", 8, slots=2)


def test_the_fit_says_what_another_format_and_the_ring_switch_do(runner_bin, tiny):
    out = _fit(runner_bin, tiny, "--kv", "fp4", "--kv-ring", "off", "-b", 8)
    assert "keeps f32" in out and "always keep a window ring" in out, out
    assert _bytes(out, "fit") == _expected("f32", 8)


def _score_route(stderr):
    lines = [ln for ln in stderr.splitlines() if ln.startswith("score: route ")]
    assert len(lines) == 1, f"expected one 'score: route' line:\n{stderr[-2000:]}"
    return json.loads(lines[0][len("score: route "):])


def test_route_line_reports_real_kv_type(runner_bin, tiny):
    """R4.48.32: --score's route line and RUNNER_DEBUG_VRAM report Inkling 2's
    real K/V cache format (f32 or q8), matching the load line rather than
    falling through to model_t's zero-initialized f16 default."""
    import os
    for extra, want in [
        ((), "f32"),
        (("--kv", "f32"), "f32"),
        (("--kv", "q8"), "q8"),
        (("--kv", "f16"), "f32"),
        (("--kv", "fp4"), "f32"),
        (("--kv", "k8v4"), "f32"),
    ]:
        env = dict(os.environ, RUNNER_DEBUG_VRAM="1")
        r = subprocess.run(
            [str(runner_bin), "-m", str(tiny), "--gpu", "off", "-c", str(CTX),
             "-t", "2", "-b", "8", "--score", "1", "-p", PROMPT, *extra],
            capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=600,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        assert f"K/V {want}" in r.stderr, (extra, r.stderr[-2000:])
        route = _score_route(r.stderr)
        assert route["kv"] == want, (extra, route)
        assert f"as {want}," in r.stderr, (extra, r.stderr[-2000:])


@pytest.mark.parametrize("extra,want", [((), "f32"), (("--kv", "q8"), "q8")])
def test_record_verify_and_doctor_report_real_kv_type(runner_bin, tiny, tmp_path, extra, want):
    """R4.48.32: receipts (--record), --verify replay, and --doctor all agree
    with Inkling 2's real K/V cache format (f32 or q8)."""
    rec_path = tmp_path / f"ink2_{want}.json"
    r = _run(runner_bin, tiny, "-t", 2, "-b", 8, "-p", "hi", "-n", 2, "--temp", 0,
             "--record", rec_path, *extra)
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec["profile"]["kv"] == want, rec["profile"]

    v = subprocess.run(
        [str(runner_bin), "-m", str(tiny), "--verify", str(rec_path)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert v.returncode == 0, v.stdout + v.stderr[-2000:]

    d = _run(runner_bin, tiny, "-t", 2, "-b", 8, "--doctor", *extra)
    assert d.returncode in (0, 1, 2) and d.stdout.strip(), d.stderr[-2000:]
    doc = json.loads(d.stdout)
    assert doc["context"]["kv"] == want, doc["context"]


def _restamp_receipt(raw_bytes, *, runner_ver=None, kv=None):
    """Restamp runner version and/or profile.kv in a compact receipt and update
    its chain.hash so --verify exercises run_verify_replay directly."""
    import hashlib
    s = raw_bytes.decode("utf-8")
    if runner_ver is not None:
        s = re.sub(r'"runner":"[^"]+"', f'"runner":"{runner_ver}"', s, count=1)
    if kv is not None:
        s = re.sub(r'"kv":"[^"]+"', f'"kv":"{kv}"', s, count=1)
    b = s.encode("utf-8")
    cut = b.rindex(b',"chain"')
    h = hashlib.sha256(b[:cut]).hexdigest()
    s = re.sub(r'("chain":\{[^}]*"hash":")[0-9a-f]{64}(")', rf"\g<1>{h}\g<2>", s, count=1)
    return s.encode("utf-8")


def test_verify_legacy_f16_receipt_version_bound(runner_bin, tiny, tmp_path):
    """R4.48.32: --verify accepts a legacy profile.kv="f16" on inkling2_loop
    ONLY when the receipt's runner version is <= 1.22.0 (every released Inkling
    build through 1.22.0 ran f32 only and stamped "f16"); post-1.22.0 receipts
    with "f16" stay strictly refused, and new f32/q8 receipts verify strictly."""
    rec_path = tmp_path / "base_f32.json"
    r = _run(runner_bin, tiny, "-t", 2, "-b", 8, "-p", "hi", "-n", 2, "--temp", 0,
             "--record", rec_path)
    assert r.returncode == 0, r.stderr[-2000:]
    raw = rec_path.read_bytes()

    for ver in ("1.21.0", "1.22.0"):
        p = tmp_path / f"legacy_{ver}.json"
        p.write_bytes(_restamp_receipt(raw, runner_ver=ver, kv="f16"))
        v = subprocess.run([str(runner_bin), "-m", str(tiny), "--verify", str(p)],
                           capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        assert v.returncode == 0, (ver, v.stdout + v.stderr[-2000:])

    for ver in ("1.22.1", "1.23.0"):
        p = tmp_path / f"post_{ver}.json"
        p.write_bytes(_restamp_receipt(raw, runner_ver=ver, kv="f16"))
        v = subprocess.run([str(runner_bin), "-m", str(tiny), "--verify", str(p)],
                           capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        assert v.returncode == 3, (ver, v.stdout + v.stderr[-2000:])
        assert "could not reproduce the recorded execution profile" in v.stderr, (ver, v.stderr[-2000:])

    # Post-1.22.0 receipt with honest "f32" still verifies cleanly.
    p_f32 = tmp_path / "post_1.23.0_f32.json"
    p_f32.write_bytes(_restamp_receipt(raw, runner_ver="1.23.0", kv="f32"))
    v_f32 = subprocess.run([str(runner_bin), "-m", str(tiny), "--verify", str(p_f32)],
                           capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert v_f32.returncode == 0, v_f32.stdout + v_f32.stderr[-2000:]


def test_resume_legacy_f16_session_refuses_without_f16_advice(runner_bin, tiny, tmp_path):
    """R4.48.32: resuming a legacy Inkling 2 session image whose header recorded
    kv_type="f16" refuses loudly and does not give the impossible advice
    'resume with --kv f16'."""
    import hashlib
    import struct

    sess = tmp_path / "cur.sess"
    r = _run(runner_bin, tiny, "-t", 2, "-b", 8, "-p", "hi", "-n", 2, "--temp", 0,
             "--session-out", sess)
    assert r.returncode == 0, r.stderr[-2000:]
    raw = sess.read_bytes()
    hl = struct.unpack_from("<I", raw, 16)[0]
    hdr = json.loads(raw[20 : 20 + hl].decode("utf-8"))
    assert hdr["kv_type"] == "f32", hdr
    # Rewrite header as a 1.22.0-style session (kv_type="f16", different model_key)
    hdr["kv_type"] = "f16"
    hdr["model"]["model_key"] = "0000000000000001"
    new_hdr = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    body = raw[:16] + struct.pack("<I", len(new_hdr)) + new_hdr + raw[20 + hl : -32]
    legacy = tmp_path / "legacy_f16.sess"
    legacy.write_bytes(body + hashlib.sha256(body).digest())

    res = subprocess.run([str(runner_bin), "-m", str(tiny), "--gpu", "off", "--resume", str(legacy)],
                         capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert res.returncode != 0, res.stderr[-2000:]
    assert "resume with --kv f16" not in res.stderr, res.stderr[-2000:]
    assert "inkling2_loop only supports f32 and q8" in res.stderr, res.stderr[-2000:]


def test_train_kv_f32_admits_inkling2_f32_and_refuses_q8(runner_bin, tiny, tmp_path):
    """R4.48.32: --train-kv-f32 now admits Inkling 2 (whose K/V cache is f32,
    where it falsely refused before) and still refuses when --kv q8 is active."""
    corpus = tmp_path / "train.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog.\n" * 4, encoding="utf-8")
    out_ok = tmp_path / "ok.gguf"
    r_ok = subprocess.run(
        [str(runner_bin), "-m", str(tiny), "--gpu", "off", "-c", str(CTX), "-t", "2",
         "--train", str(corpus), "--train-steps", "2", "--lora-rank", "4", "--lr", "3e-3",
         "-s", "7", "--train-kv-f32", "--train-out", str(out_ok)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert r_ok.returncode == 0, r_ok.stderr[-2000:]
    rec = json.loads((tmp_path / "ok.gguf.train.json").read_text(encoding="utf-8"))
    assert rec["kv_cache"] == "f32", rec

    out_bad = tmp_path / "bad.gguf"
    r_bad = subprocess.run(
        [str(runner_bin), "-m", str(tiny), "--gpu", "off", "-c", str(CTX), "-t", "2",
         "--train", str(corpus), "--train-steps", "2", "--lora-rank", "4", "--lr", "3e-3",
         "-s", "7", "--train-kv-f32", "--kv", "q8", "--train-out", str(out_bad)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert r_bad.returncode != 0, r_bad.stderr[-2000:]
    assert "--train-kv-f32 keeps an f32 KV cache" in r_bad.stderr, r_bad.stderr[-2000:]


def test_route_reports_real_layers_paths_cpu_act_and_debug_route(runner_bin, tiny):
    """R4.48.32: route_collect / model_route_json / score_route / RUNNER_DEBUG_ROUTE /
    --doctor / --benchmark report Inkling 2's real layer count, MoE layer count,
    forward paths (solo/batch/keep), and CPU activation format (ignoring generic
    RUNNER_CPU_I8* and honoring RUNNER_INKLING2_ACT8 on the fast route)."""
    import os

    def run_env(args, extra_env=None):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("RUNNER_CPU_I8", "RUNNER_INKLING2_", "RUNNER_SCORE_", "RUNNER_DEBUG_"))}
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [str(runner_bin), "-m", str(tiny), "--gpu", "off", "-c", str(CTX),
             "-t", "2", "-b", "8", *map(str, args)],
            capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=600,
        )

    # 1. Default solo --score: layers, moe_layers, paths.solo, load banner
    r_solo = run_env(["--score", "1", "-p", PROMPT])
    assert r_solo.returncode == 0, r_solo.stderr[-2000:]
    assert f"| inkling2_loop | {N_BLOCK} layers |" in r_solo.stderr, r_solo.stderr[-2000:]
    route = _score_route(r_solo.stderr)
    sroute = json.loads(r_solo.stdout)["score_route"]
    assert route["backend"] == "cpu" and route["device"] == "", route
    assert route["layers"] == N_BLOCK and route["gpu_layers"] == 0, route
    assert route["moe_layers"] == N_BLOCK and route["experts_on_host"] == 0, route
    assert route["cpu_act"] == "f32" and route["backend_detail"] == "", route
    assert route["paths"].get("solo", {}).get("forwards", 0) > 0, route["paths"]
    assert route["paths"]["solo"]["rows_max"] == 1, route["paths"]
    assert sroute["paths"] == route["paths"], (sroute, route["paths"])
    assert sroute["backend"] == "cpu" and sroute["cpu_act"] == "f32" and sroute["backend_detail"] == "", sroute

    # 2. Chunked --score + RUNNER_DEBUG_ROUTE=1: paths.keep + route: line on stderr
    r_keep = run_env(["--score", "1", "-p", PROMPT],
                     {"RUNNER_SCORE_CHUNKED": "1", "RUNNER_DEBUG_ROUTE": "1"})
    assert r_keep.returncode == 0, r_keep.stderr[-2000:]
    route_k = _score_route(r_keep.stderr)
    sroute_k = json.loads(r_keep.stdout)["score_route"]
    assert route_k["paths"].get("keep", {}).get("forwards", 0) > 0, route_k["paths"]
    assert route_k["paths"]["keep"]["rows_max"] == 8, route_k["paths"]
    assert sroute_k["paths"] == route_k["paths"], (sroute_k, route_k["paths"])
    dbg_lines = [ln for ln in r_keep.stderr.splitlines() if ln.startswith("route: ")]
    assert dbg_lines, r_keep.stderr[-2000:]
    assert f"backend=cpu" in dbg_lines[0] and f"split=0/{N_BLOCK}" in dbg_lines[0], dbg_lines[0]
    assert f"experts_on_host=0/{N_BLOCK}" in dbg_lines[0] and "path=keep" in dbg_lines[0], dbg_lines[0]

    # 3. Generic RUNNER_CPU_I8* must NOT pollute Inkling 2's cpu_act
    r_i8 = run_env(["--score", "1", "-p", PROMPT],
                   {"RUNNER_CPU_I8": "1", "RUNNER_CPU_I8_PREFILL": "1", "RUNNER_CPU_I8K": "1"})
    assert r_i8.returncode == 0, r_i8.stderr[-2000:]
    assert _score_route(r_i8.stderr)["cpu_act"] == "f32", _score_route(r_i8.stderr)
    assert json.loads(r_i8.stdout)["score_route"]["cpu_act"] == "f32"

    # 4. RUNNER_INKLING2_ACT8=1 reports cpu_act="int8" on the fast route, "f32" on reference
    r_a8 = run_env(["--score", "1", "-p", PROMPT], {"RUNNER_INKLING2_ACT8": "1"})
    assert r_a8.returncode == 0, r_a8.stderr[-2000:]
    assert _score_route(r_a8.stderr)["cpu_act"] == "int8", _score_route(r_a8.stderr)
    assert json.loads(r_a8.stdout)["score_route"]["cpu_act"] == "int8"

    r_ref_a8 = run_env(["--score", "1", "-p", PROMPT],
                       {"RUNNER_INKLING2_ROUTE": "reference", "RUNNER_INKLING2_ACT8": "1"})
    assert r_ref_a8.returncode == 0, r_ref_a8.stderr[-2000:]
    assert _score_route(r_ref_a8.stderr)["cpu_act"] == "f32", _score_route(r_ref_a8.stderr)

    # 5. --doctor and --bench-json report real layer count and forward paths
    r_doc = run_env(["--doctor"])
    assert r_doc.returncode in (0, 1, 2) and r_doc.stdout.strip(), r_doc.stderr[-2000:]
    doc = json.loads(r_doc.stdout)
    assert doc["model"]["layers"] == N_BLOCK, doc["model"]
    assert doc["placement"]["layers"] == N_BLOCK and doc["placement"]["gpu_layers"] == 0, doc["placement"]

    r_bench = run_env(["--bench-json", "-p", "hello world test", "-n", "2"])
    assert r_bench.returncode == 0, r_bench.stderr[-2000:]
    bench = json.loads(r_bench.stdout)
    assert bench["layers"] == N_BLOCK and bench["gpu_layers"] == 0, bench
    assert bench["route"]["layers"] == N_BLOCK and bench["route"]["moe_layers"] == N_BLOCK, bench["route"]
    assert "batch" in bench["route"]["paths"] and "solo" in bench["route"]["paths"], bench["route"]["paths"]


def test_record_verify_tracks_inkling2_act8_cpu_act(runner_bin, tiny, tmp_path):
    """R4.48.32: a receipt recorded with RUNNER_INKLING2_ACT8=1 stamps cpu_act="int8",
    verifies when replayed with RUNNER_INKLING2_ACT8=1, and is refused when replayed
    without it."""
    import os
    env_on = dict(os.environ, RUNNER_INKLING2_ACT8="1")
    env_off = {k: v for k, v in os.environ.items() if k != "RUNNER_INKLING2_ACT8"}
    rec_path = tmp_path / "ink2_act8.json"
    r = subprocess.run(
        [str(runner_bin), "-m", str(tiny), "--gpu", "off", "-c", str(CTX),
         "-t", "2", "-b", "8", "-p", "hi", "-n", "2", "--temp", "0", "--record", str(rec_path)],
        capture_output=True, encoding="utf-8", errors="replace", env=env_on, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec["profile"]["route"]["cpu_act"] == "int8", rec["profile"]["route"]

    v_ok = subprocess.run(
        [str(runner_bin), "-m", str(tiny), "--verify", str(rec_path)],
        capture_output=True, encoding="utf-8", errors="replace", env=env_on, timeout=600,
    )
    assert v_ok.returncode == 0, v_ok.stdout + v_ok.stderr[-2000:]

    v_bad = subprocess.run(
        [str(runner_bin), "-m", str(tiny), "--verify", str(rec_path)],
        capture_output=True, encoding="utf-8", errors="replace", env=env_off, timeout=600,
    )
    assert v_bad.returncode == 3, v_bad.stdout + v_bad.stderr[-2000:]
    assert "did not take the recorded route" in v_bad.stderr, v_bad.stderr[-2000:]


def test_cuda_route_fields_and_receipt_verify(runner_bin, tiny, tmp_path):
    """R4.48.32 (CUDA host): --score route, --record/--verify, and --doctor report
    backend="cuda", non-empty device, gpu_layers==N_BLOCK, experts_on_host under
    --cpu-moe, real paths, and Inkling CUDA backend_detail."""
    import os
    env_cuda = dict(os.environ, RUNNER_INKLING2_CUDA="1")
    env_cuda.pop("RUNNER_INKLING2_ROUTE", None)
    env_cuda.pop("RUNNER_INKLING2_ACT8", None)
    common = [str(runner_bin), "-m", str(tiny), "--gpu", "auto", "-c", str(CTX), "-t", "2", "-b", "8"]
    r_w = subprocess.run(common + ["--score", "1", "-p", PROMPT],
                         capture_output=True, encoding="utf-8", errors="replace", env=env_cuda, timeout=600)
    assert r_w.returncode == 0, r_w.stderr[-2000:]
    if "inkling2_loop: the weights on CUDA" not in r_w.stderr:
        pytest.skip("no CUDA device available for inkling2_loop route test")
    rw = _score_route(r_w.stderr)
    sw = json.loads(r_w.stdout)["score_route"]
    assert rw["backend"] == "cuda" and rw["device"], rw
    assert rw["gpu_layers"] == N_BLOCK and rw["layers"] == N_BLOCK, rw
    assert rw["moe_layers"] == N_BLOCK and rw["experts_on_host"] == 0, rw
    assert rw["cpu_act"] == "f32" and "placed=W" in rw["backend_detail"], rw
    assert rw["paths"].get("solo", {}).get("forwards", 0) > 0, rw["paths"]
    assert sw["backend"] == "cuda" and sw["backend_detail"] == rw["backend_detail"], (sw, rw)
    assert sw["paths"] == rw["paths"], (sw, rw)

    r_m = subprocess.run(common + ["--cpu-moe", "--score", "1", "-p", PROMPT],
                         capture_output=True, encoding="utf-8", errors="replace", env=env_cuda, timeout=600)
    assert r_m.returncode == 0, r_m.stderr[-2000:]
    rm = _score_route(r_m.stderr)
    assert rm["backend"] == "cuda" and rm["gpu_layers"] == N_BLOCK, rm
    assert rm["experts_on_host"] == N_BLOCK and "placed=M" in rm["backend_detail"], rm

    rec_path = tmp_path / "ink2_cuda.json"
    r_rec = subprocess.run(common + ["-p", "hi", "-n", "2", "--temp", "0", "--record", str(rec_path)],
                           capture_output=True, encoding="utf-8", errors="replace", env=env_cuda, timeout=600)
    assert r_rec.returncode == 0, r_rec.stderr[-2000:]
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec["profile"]["gpu"] is True and rec["profile"]["gpu_layers"] == N_BLOCK, rec["profile"]
    assert rec["profile"]["device"] != "cpu", rec["profile"]
    assert rec["profile"]["route"]["backend"] == "cuda", rec["profile"]["route"]
    assert rec["profile"]["route"]["tc_types"] == "", rec["profile"]["route"]
    assert "placed=W" in rec["profile"]["route"]["backend_detail"], rec["profile"]["route"]

    v_ok = subprocess.run([str(runner_bin), "-m", str(tiny), "--verify", str(rec_path)],
                          capture_output=True, encoding="utf-8", errors="replace", env=env_cuda, timeout=600)
    assert v_ok.returncode == 0, v_ok.stdout + v_ok.stderr[-2000:]

    env_cpu = {k: v for k, v in os.environ.items() if k != "RUNNER_INKLING2_CUDA"}
    v_cpu = subprocess.run([str(runner_bin), "-m", str(tiny), "--verify", str(rec_path)],
                           capture_output=True, encoding="utf-8", errors="replace", env=env_cpu, timeout=600)
    assert v_cpu.returncode == 3, v_cpu.stdout + v_cpu.stderr[-2000:]

    r_doc = subprocess.run(common + ["--doctor"],
                           capture_output=True, encoding="utf-8", errors="replace", env=env_cuda, timeout=600)
    assert r_doc.returncode in (0, 1, 2) and r_doc.stdout.strip(), r_doc.stderr[-2000:]
    doc = json.loads(r_doc.stdout)
    assert doc["placement"]["backend"] == "cuda" and doc["placement"]["gpu_layers"] == N_BLOCK, doc["placement"]
    assert not any(f["code"] == "gpu_not_used" for f in doc["findings"]), doc["findings"]


def test_the_cuda_fit_places_the_file_as_its_load_does(runner_bin, tiny):
    """On a CUDA host: --info fit's device section is the load's own placement run dry, so its placement is the load's
    (every weight on the device, or the expert stacks on the host, or the CPU), at --kv q8, and its non-weight line
    carries exact bytes."""
    import os
    env = dict(os.environ, RUNNER_INKLING2_CUDA="1")
    env.pop("RUNNER_INKLING2_ROUTE", None)
    common = [str(runner_bin), "-m", str(tiny), "--gpu", "auto", "-c", str(CTX), "-b", "8", "--kv", "q8"]
    fit = subprocess.run(common + ["--info", "fit"], capture_output=True, encoding="utf-8", errors="replace", env=env,
                         timeout=600)
    m = re.search(r"device\s+CUDA route: (.*)", fit.stdout)
    if not m:
        pytest.skip("no CUDA device here: the device fit is a CUDA host's (" + fit.stdout.strip().splitlines()[-1] + ")")
    assert re.search(r"non-weight\s+[0-9.]+ GiB \(\d+ bytes", fit.stdout), fit.stdout
    # the whole placement's own non-weight bytes, whichever placement the device took
    assert re.search(r"whole\s+[0-9.]+ GiB of weights \(\d+ bytes\) and [0-9.]+ GiB besides them \(non-weight \d+ bytes\)",
                     fit.stdout), fit.stdout
    load = subprocess.run(common + ["-t", "2", "-p", "hi", "-n", "1", "--temp", "0"], capture_output=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)
    assert load.returncode == 0, load.stderr[-2000:]
    on_device = "inkling2_loop: the weights on CUDA" in load.stderr
    assert m.group(1).startswith("every weight on the device") == on_device, (fit.stdout, load.stderr[-2000:])


def test_q8_engages_on_the_device(runner_bin, tiny):
    """On a CUDA host: --kv q8 under RUNNER_INKLING2_CUDA=1 stores q8 rings on the device, with no test switch: the load
    line says K/V q8, no note keeps f32, and the weights are on CUDA (the bind banner). Gate (c) in
    tests/test_inkling2_cuda.c holds those rings to the CPU's q8 cache."""
    import os
    env = dict(os.environ, RUNNER_INKLING2_CUDA="1")
    env.pop("RUNNER_INKLING2_ROUTE", None)
    common = [str(runner_bin), "-m", str(tiny), "--gpu", "auto", "-c", str(CTX), "-b", "8", "--kv", "q8"]
    fit = subprocess.run(common + ["--info", "fit"], capture_output=True, encoding="utf-8", errors="replace", env=env,
                         timeout=600)
    if "CUDA route:" not in fit.stdout:
        pytest.skip("no CUDA device here: the device rings are a CUDA host's")
    load = subprocess.run(common + ["-t", "2", "-p", "hi", "-n", "1", "--temp", "0"], capture_output=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)
    assert load.returncode == 0, load.stderr[-2000:]
    assert "K/V q8" in load.stderr and "keeps f32" not in load.stderr, load.stderr[-2000:]
    assert "inkling2_loop: the weights on CUDA" in load.stderr, load.stderr[-2000:]
    assert "slots' K/V rings in q8" in fit.stdout, fit.stdout
