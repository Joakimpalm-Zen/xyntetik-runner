"""R4.48.8.7: Gemma 4 per-expert ffn_down_exps.scale costing in --quant-error and derive.

In Gemma 4 MoE, blk.N.ffn_down_exps.scale ([n_expert]) is a learned architectural per-expert
output scale applied after the down-projection matvec (src/model.c:10717):
    moe_out += selw[t] * f[e] * (W_e @ h_e).
Unlike Inkling 2 Mini's two-level NVFP4 scale, Gemma 4's .scale is kept untouched by --quantize.
Therefore --quant-error must quantize the raw stored row W_e (the exact row --quantize writes)
and weight expert e's squared round-trip error by f[e]^2 in double.

Tests:
1. Uniform scale factor f scales blk.N.ffn_down_exps.weight error by f^2 (other tensors unchanged).
2. Per-expert heterogeneous decomposition err([a, b]) == a^2 * err([1, 0]) + b^2 * err([0, 1]),
   negative-scale sign invariance, and BF16/F16 .scale decoding matching model.c's tensor_to_f32.
3. Exact equality to independently decoded rewritten Q4_0 bytes on non-power-of-two scales [0.85, 1.15],
   while verifying --quantize preserves blk.N.ffn_down_exps.scale bit-for-bit.
4. --imatrix composes multiplicatively with f[e]^2 per expert, and -t 1 == -t 4 byte-for-byte.
5. derive --target-size ranking reversal + certified-family CPU --score fidelity check: with
   identical weights across layers 0 and 1 (expert 0 continuous/noisy, expert 1 exact on Q4_0) and
   opposite per-expert scales [4.0, 0.25] vs [0.25, 4.0], derive keeps the high-leverage layer at
   q8_0, reverses when scales are swapped, and beats the inverted blind plan on --score drift vs F32.
6. NVFP4 non-clobber guard: --quant-error ... --quant nvfp4 reports null for Gemma 4's
   blk.N.ffn_down_exps.weight and --quantize --quant nvfp4 keeps its [n_expert] .scale intact.
"""
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gguf_fixture as gf  # noqa: E402

T_F32 = 0
T_F16 = 1
T_Q4_0 = 2
T_Q8_0 = 8
T_BF16 = 30
T_NVFP4 = 40

E = 64
FF = 64
HEADS = 2
KV = 2
LAYERS = 2
N_EXP = 2
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256


def _flist(n, seed=1, amp=0.15):
    return [amp * math.sin(0.37 * (i + 1) + 1.13 * seed) + 0.04 * math.cos(1.91 * (i + 1) + seed)
            for i in range(n)]


def _q40_exact_list(n, seed=1, scale=0.05):
    """Values that are exact multiples of scale * (q - 8) with q in 0..15 and hit q=0 per block of 32,
    so Q4_0 represents them with zero error while keeping comparable L2 norm to _flist."""
    out = []
    for blk in range(n // 32):
        for j in range(32):
            if j == 0:
                q = 0
            else:
                q = ((blk * 7 + j * 5 + seed * 3) % 15) + 1
            out.append(scale * (q - 8))
    return out


def _f32_to_bf16_bytes(vals):
    out = bytearray()
    for v in vals:
        u = struct.unpack("<I", struct.pack("<f", v))[0]
        # round-to-nearest-even to bf16
        u += 0x7FFF + ((u >> 16) & 1)
        out += struct.pack("<H", (u >> 16) & 0xFFFF)
    return bytes(out)


def _bf16_roundtrip(vals):
    raw = _f32_to_bf16_bytes(vals)
    return [struct.unpack("<f", struct.pack("<I", h << 16))[0]
            for (h,) in struct.iter_unpack("<H", raw)]


def _f32_to_f16_bytes(vals):
    return struct.pack(f"<{len(vals)}e", *vals)


def _f16_roundtrip(vals):
    return list(struct.unpack(f"<{len(vals)}e", _f32_to_f16_bytes(vals)))


def _make_gemma4_moe(path, layer_scales, down_weights_by_layer=None, scale_dtype=T_F32):
    """Write a tiny valid Gemma 4 dual-branch MoE GGUF fixture."""
    hd = E // HEADS
    kv_dim = KV * hd
    tensors = [
        ("token_embd.weight", [E, len(VOCAB)], gf.pack(_flist(E * len(VOCAB), seed=10))),
        ("output_norm.weight", [E], gf.ones(E)),
    ]
    for i in range(LAYERS):
        if down_weights_by_layer is not None:
            down_vals = down_weights_by_layer[i]
        else:
            down_vals = _flist(FF * E * N_EXP, seed=100 + i, amp=0.25)
        sc = layer_scales[i]
        if scale_dtype == T_F32:
            sc_bytes = gf.pack(sc)
        elif scale_dtype == T_BF16:
            sc_bytes = _f32_to_bf16_bytes(sc)
        elif scale_dtype == T_F16:
            sc_bytes = _f32_to_f16_bytes(sc)
        else:
            raise ValueError(scale_dtype)

        tensors += [
            (f"blk.{i}.attn_norm.weight", [E], gf.ones(E)),
            (f"blk.{i}.attn_q.weight", [E, E], gf.pack(_flist(E * E, seed=20 + i))),
            (f"blk.{i}.attn_k.weight", [E, kv_dim], gf.pack(_flist(E * kv_dim, seed=30 + i))),
            (f"blk.{i}.attn_v.weight", [E, kv_dim], gf.pack(_flist(E * kv_dim, seed=40 + i))),
            (f"blk.{i}.attn_output.weight", [E, E], gf.pack(_flist(E * E, seed=50 + i))),
            (f"blk.{i}.attn_q_norm.weight", [hd], gf.ones(hd)),
            (f"blk.{i}.attn_k_norm.weight", [hd], gf.ones(hd)),
            (f"blk.{i}.post_attention_norm.weight", [E], gf.ones(E)),
            (f"blk.{i}.ffn_norm.weight", [E], gf.ones(E)),
            (f"blk.{i}.post_ffw_norm.weight", [E], gf.ones(E)),
            (f"blk.{i}.ffn_gate.weight", [E, FF], gf.pack(_flist(E * FF, seed=60 + i, amp=0.2))),
            (f"blk.{i}.ffn_up.weight", [E, FF], gf.pack(_flist(E * FF, seed=70 + i, amp=0.2))),
            (f"blk.{i}.ffn_down.weight", [FF, E], gf.pack(_flist(FF * E, seed=80 + i, amp=0.2))),
            # Equal router logits (zeros) so both experts get selw = [0.5, 0.5] cleanly.
            (f"blk.{i}.ffn_gate_inp.weight", [E, N_EXP], gf.pack([0.0] * (E * N_EXP))),
            (f"blk.{i}.ffn_gate_inp.scale", [E], gf.pack([1.0] * E)),
            (f"blk.{i}.ffn_gate_up_exps.weight", [E, 2 * FF, N_EXP],
             gf.pack(_flist(E * 2 * FF * N_EXP, seed=90 + i, amp=0.25))),
            (f"blk.{i}.ffn_down_exps.weight", [FF, E, N_EXP], gf.pack(down_vals)),
            (f"blk.{i}.ffn_down_exps.scale", [N_EXP], sc_bytes, scale_dtype),
            (f"blk.{i}.post_ffw_norm_1.weight", [E], gf.ones(E)),
            (f"blk.{i}.pre_ffw_norm_2.weight", [E], gf.ones(E)),
            (f"blk.{i}.post_ffw_norm_2.weight", [E], gf.ones(E)),
            (f"blk.{i}.layer_output_scale.weight", [1], gf.pack([1.0])),
        ]
    meta = [
        gf.ks("general.architecture", "gemma4"),
        gf.ku("gemma4.block_count", LAYERS),
        gf.ku("gemma4.context_length", 256),
        gf.ku("gemma4.embedding_length", E),
        gf.ku("gemma4.feed_forward_length", FF),
        gf.ku("gemma4.attention.head_count", HEADS),
        gf.ku("gemma4.attention.head_count_kv", KV),
        gf.kf("gemma4.attention.layer_norm_rms_epsilon", 1e-5),
        gf.kf("gemma4.rope.freq_base", 10000.0),
        gf.ku("gemma4.expert_count", N_EXP),
        gf.ku("gemma4.expert_used_count", N_EXP),
        gf.ku("gemma4.expert_feed_forward_length", FF),
        gf.ku("gemma4.attention.key_length", hd),
        gf.ku("gemma4.rope.dimension_count", hd),
        gf.kf("gemma4.final_logit_softcapping", 30.0),
        gf.ks("tokenizer.ggml.model", "llama"),
        gf.kas("tokenizer.ggml.tokens", VOCAB),
        gf.kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
        gf.kai("tokenizer.ggml.token_type", TTYPE),
        gf.ku("tokenizer.ggml.bos_token_id", 1),
        gf.ku("tokenizer.ggml.eos_token_id", 2),
        gf.kb("tokenizer.ggml.add_bos_token", True),
    ]
    gf.write_gguf(str(path), tensors, meta)
    return path


def _gguf(path):
    """Read (kv, {name: (dims, type, raw_bytes)}) of a GGUF v3 file."""
    b = Path(path).read_bytes()
    o = [4]

    def take(fmt):
        v = struct.unpack_from("<" + fmt, b, o[0])
        o[0] += struct.calcsize("<" + fmt)
        return v[0]

    def text():
        n = take("Q")
        o[0] += n
        return b[o[0] - n:o[0]].decode()

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return text()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(scalar[t])

    assert b[:4] == b"GGUF"
    take("I")
    nt, nkv = take("Q"), take("Q")
    kv = {}
    for _ in range(nkv):
        k = text()
        kv[k] = value(take("I"))
    infos = []
    for _ in range(nt):
        name = text()
        dims = [take("Q") for _ in range(take("I"))]
        infos.append((name, dims, take("I"), take("Q")))
    al = kv.get("general.alignment", 32)
    start = (o[0] + al - 1) // al * al
    sizes = {T_F32: (4, 1), T_F16: (2, 1), T_Q4_0: (18, 32), T_Q8_0: (34, 32), T_BF16: (2, 1), T_NVFP4: (36, 64)}
    out = {}
    for name, dims, typ, off in infos:
        n = 1
        for d in dims:
            n *= d
        ts, bs = sizes.get(typ, (0, 1))
        out[name] = (dims, typ, b[start + off:start + off + n // bs * ts] if ts else None)
    return kv, out


def _dequant_q4_0(raw, n):
    out = []
    for off in range(0, len(raw), 18):
        d = struct.unpack_from("<e", raw, off)[0]
        qs = raw[off + 2:off + 18]
        lo = [(byte & 0x0F) - 8 for byte in qs]
        hi = [(byte >> 4) - 8 for byte in qs]
        out.extend(d * q for q in lo)
        out.extend(d * q for q in hi)
    assert len(out) == n
    return out


def _write_imatrix_gguf(path, entries):
    """Write a GGUF imatrix: {tensor_name: (n_per_row, n_mat, flat_values)}."""
    tensors = []
    for name, (npr, nmat, vals) in entries.items():
        dims = [npr] if nmat == 1 else [npr, nmat]
        tensors.append((f"{name}.in_sum2", dims, gf.pack(vals), T_F32))
        tensors.append((f"{name}.counts", [1] if nmat == 1 else [1, nmat], gf.pack([1.0] * nmat), T_F32))
    gf.write_gguf(str(path), tensors, [gf.ku("imatrix.chunk_count", 1)])
    return path


def _run(runner_bin, *args):
    env = dict(os.environ)
    env.pop("RUNNER_FORCE_REQUANT", None)
    env.pop("RUNNER_REQUANT_ONLY", None)
    return subprocess.run(
        [str(runner_bin), *map(str, args)],
        capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=300,
    )


def _report(path):
    return {r["tensor"]: r for r in (json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())}


def _score_logprobs(runner_bin, model, corpus, out_json):
    r = _run(
        runner_bin, "-m", model, "--score", "24", "-f", corpus,
        "-c", "64", "-t", "2", "--gpu", "off",
        "--score-out", out_json,
    )
    assert r.returncode == 0, f"score failed: {r.stderr}"
    data = json.loads(Path(out_json).read_text())
    return data["logprobs"]


def test_uniform_down_exps_scale_weights_error_by_f_squared(runner_bin, tmp_path):
    """Uniform ffn_down_exps.scale = [f, f] multiplies blk.N.ffn_down_exps.weight error by f^2."""
    m_one = _make_gemma4_moe(tmp_path / "one.gguf", [[1.0, 1.0], [1.0, 1.0]])
    m_qtr = _make_gemma4_moe(tmp_path / "qtr.gguf", [[0.25, 0.25], [0.25, 0.25]])
    m_big = _make_gemma4_moe(tmp_path / "big.gguf", [[4.0, 4.0], [4.0, 4.0]])

    reps = {}
    for tag, src in (("one", m_one), ("qtr", m_qtr), ("big", m_big)):
        out = tmp_path / f"qe-{tag}.jsonl"
        r = _run(runner_bin, "-m", src, "--quant-error", out, "--quant", "q8_0,q4_0")
        assert r.returncode == 0, r.stderr
        reps[tag] = _report(out)

    for i in range(LAYERS):
        dn = f"blk.{i}.ffn_down_exps.weight"
        for qt in ("q8_0", "q4_0"):
            e1 = reps["one"][dn]["types"][qt]["error"]
            eq = reps["qtr"][dn]["types"][qt]["error"]
            eb = reps["big"][dn]["types"][qt]["error"]
            assert e1 > 0.0
            assert eq == pytest.approx(e1 / 16.0, rel=1e-7), f"{dn} {qt}: {eq} vs {e1}/16"
            assert eb == pytest.approx(e1 * 16.0, rel=1e-7), f"{dn} {qt}: {eb} vs {e1}*16"
        for other in (f"blk.{i}.ffn_gate_up_exps.weight", f"blk.{i}.ffn_down.weight", f"blk.{i}.attn_q.weight"):
            for qt in ("q8_0", "q4_0"):
                assert reps["qtr"][other]["types"][qt]["error"] == reps["one"][other]["types"][qt]["error"]
                assert reps["big"][other]["types"][qt]["error"] == reps["one"][other]["types"][qt]["error"]


def test_per_expert_asymmetric_decomposition_sign_and_bf16_f16_scale(runner_bin, tmp_path):
    """Per-expert f[e]^2 decomposes linearly across experts, ignores sign, and decodes BF16/F16 .scale."""
    m_e0 = _make_gemma4_moe(tmp_path / "e0.gguf", [[1.0, 0.0], [1.0, 0.0]])
    m_e1 = _make_gemma4_moe(tmp_path / "e1.gguf", [[0.0, 1.0], [0.0, 1.0]])
    m_mix = _make_gemma4_moe(tmp_path / "mix.gguf", [[3.0, 0.5], [0.25, 2.5]])
    m_neg = _make_gemma4_moe(tmp_path / "neg.gguf", [[-3.0, 0.5], [0.25, -2.5]])
    bf_sc = [[0.85, 1.15], [1.75, 0.35]]
    m_bf16 = _make_gemma4_moe(tmp_path / "bf16.gguf", bf_sc, scale_dtype=T_BF16)
    m_bf16_ref = _make_gemma4_moe(tmp_path / "bf16-ref.gguf", [_bf16_roundtrip(s) for s in bf_sc], scale_dtype=T_F32)
    m_f16 = _make_gemma4_moe(tmp_path / "f16.gguf", bf_sc, scale_dtype=T_F16)
    m_f16_ref = _make_gemma4_moe(tmp_path / "f16-ref.gguf", [_f16_roundtrip(s) for s in bf_sc], scale_dtype=T_F32)

    reps = {}
    for tag, src in (("e0", m_e0), ("e1", m_e1), ("mix", m_mix), ("neg", m_neg),
                     ("bf16", m_bf16), ("bf16_ref", m_bf16_ref), ("f16", m_f16), ("f16_ref", m_f16_ref)):
        out = tmp_path / f"qe-{tag}.jsonl"
        r = _run(runner_bin, "-m", src, "--quant-error", out, "--quant", "q4_0")
        assert r.returncode == 0, r.stderr
        reps[tag] = _report(out)

    mix_sc = [[3.0, 0.5], [0.25, 2.5]]
    for i in range(LAYERS):
        dn = f"blk.{i}.ffn_down_exps.weight"
        f0, f1 = mix_sc[i]
        expected = f0 * f0 * reps["e0"][dn]["types"]["q4_0"]["error"] + f1 * f1 * reps["e1"][dn]["types"]["q4_0"]["error"]
        assert reps["mix"][dn]["types"]["q4_0"]["error"] == pytest.approx(expected, rel=1e-7)
        assert reps["neg"][dn]["types"]["q4_0"]["error"] == pytest.approx(expected, rel=1e-7)
        assert reps["bf16"][dn]["types"]["q4_0"]["error"] == pytest.approx(
            reps["bf16_ref"][dn]["types"]["q4_0"]["error"], rel=1e-7)
        assert reps["f16"][dn]["types"]["q4_0"]["error"] == pytest.approx(
            reps["f16_ref"][dn]["types"]["q4_0"]["error"], rel=1e-7)


def test_exact_match_to_rewritten_q4_0_bytes_and_scale_kept_unchanged(runner_bin, tmp_path):
    """--quantize preserves blk.N.ffn_down_exps.scale bit-for-bit, and --quant-error matches
    sum_e f[e]^2 ||W_e - deq(W_e_q4_0)||^2 on non-power-of-two scales [0.85, 1.15]."""
    scales = [[0.85, 1.15], [1.35, 0.65]]
    src = _make_gemma4_moe(tmp_path / "src.gguf", scales)
    rewritten = tmp_path / "q40.gguf"
    r = _run(runner_bin, "-m", src, "--quantize", rewritten, "--quant", "q4_0")
    assert r.returncode == 0, r.stderr

    _, src_t = _gguf(src)
    _, got_t = _gguf(rewritten)
    qe_path = tmp_path / "qe.jsonl"
    r = _run(runner_bin, "-m", src, "--quant-error", qe_path, "--quant", "q4_0")
    assert r.returncode == 0, r.stderr
    rep = _report(qe_path)

    for i in range(LAYERS):
        sc_name = f"blk.{i}.ffn_down_exps.scale"
        w_name = f"blk.{i}.ffn_down_exps.weight"
        assert got_t[sc_name] == src_t[sc_name], f"{sc_name} must be kept bit-for-bit"
        assert got_t[w_name][1] == T_Q4_0

        w_src = list(struct.unpack(f"<{FF * E * N_EXP}f", src_t[w_name][2]))
        w_deq = _dequant_q4_0(got_t[w_name][2], FF * E * N_EXP)
        per_exp = FF * E
        manual = 0.0
        for e in range(N_EXP):
            f = scales[i][e]
            sse = sum((w_src[e * per_exp + k] - w_deq[e * per_exp + k]) ** 2 for k in range(per_exp))
            manual += (f * f) * sse
        assert rep[w_name]["types"]["q4_0"]["error"] == pytest.approx(manual, rel=1e-6)


def test_imatrix_and_multithread_determinism(runner_bin, tmp_path):
    """--imatrix weights compose multiplicatively with f[e]^2 per expert and -t 1 == -t 4."""
    m_one = _make_gemma4_moe(tmp_path / "one.gguf", [[1.0, 1.0], [1.0, 1.0]])
    m_sc = _make_gemma4_moe(tmp_path / "sc.gguf", [[2.0, 2.0], [0.5, 0.5]])
    imx_vals = [1.0 + 0.1 * ((k * 7) % 13) for k in range(FF * N_EXP)]
    imx = _write_imatrix_gguf(tmp_path / "imx.gguf", {
        "blk.0.ffn_down_exps.weight": (FF, N_EXP, imx_vals),
        "blk.1.ffn_down_exps.weight": (FF, N_EXP, imx_vals),
    })

    out_one = tmp_path / "qe-one.jsonl"
    out_sc1 = tmp_path / "qe-sc-t1.jsonl"
    out_sc4 = tmp_path / "qe-sc-t4.jsonl"
    for src, out, th in ((m_one, out_one, "1"), (m_sc, out_sc1, "1"), (m_sc, out_sc4, "4")):
        r = _run(runner_bin, "-m", src, "--quant-error", out, "--quant", "q4_0", "--imatrix", imx, "-t", th)
        assert r.returncode == 0, r.stderr

    assert out_sc1.read_bytes() == out_sc4.read_bytes(), "-t 1 and -t 4 reports must be byte-identical"
    r_one, r_sc = _report(out_one), _report(out_sc1)
    assert r_sc["blk.0.ffn_down_exps.weight"]["weighted"] is True
    assert r_sc["blk.0.ffn_down_exps.weight"]["types"]["q4_0"]["error"] == pytest.approx(
        4.0 * r_one["blk.0.ffn_down_exps.weight"]["types"]["q4_0"]["error"], rel=1e-7)
    assert r_sc["blk.1.ffn_down_exps.weight"]["types"]["q4_0"]["error"] == pytest.approx(
        0.25 * r_one["blk.1.ffn_down_exps.weight"]["types"]["q4_0"]["error"], rel=1e-7)


def test_derive_ranking_and_fidelity_beat_blind_plan(runner_bin, tmp_path):
    """With identical expert weights across layers 0 and 1 (expert 0 continuous/noisy, expert 1 exact on Q4_0)
    and opposite per-expert scales [4.0, 0.25] vs [0.25, 4.0]:
    - derive --target-size promotes the high-leverage layer's ffn_down_exps.weight and reverses on swap;
    - on CPU --score against F32, the scale-aware plan has strictly lower logprob drift than the blind inversion."""
    w_noisy = _flist(FF * E, seed=77, amp=0.25)
    w_clean = _q40_exact_list(FF * E, seed=77, scale=0.04)
    same_pair = w_noisy + w_clean

    m_fwd = _make_gemma4_moe(tmp_path / "fwd.gguf", [[4.0, 0.25], [0.25, 4.0]],
                             down_weights_by_layer=[same_pair, same_pair])
    m_rev = _make_gemma4_moe(tmp_path / "rev.gguf", [[0.25, 4.0], [4.0, 0.25]],
                             down_weights_by_layer=[same_pair, same_pair])

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("The quick brown fox jumps over the lazy dog. " * 20, encoding="utf-8")

    # Measure total q4_0 vs q8_0 floor so we pick a budget that fits all tensors at q4_0 plus enough room
    # to promote roughly half the model (specifically one of the two ffn_down_exps.weight tensors, not both).
    qe_fwd = tmp_path / "qe-fwd.jsonl"
    r = _run(runner_bin, "-m", m_fwd, "--quant-error", qe_fwd, "--quant", "q8_0,q4_0")
    assert r.returncode == 0, r.stderr
    rep = _report(qe_fwd)
    assert rep["blk.0.ffn_down_exps.weight"]["types"]["q4_0"]["error"] > (
        100.0 * rep["blk.1.ffn_down_exps.weight"]["types"]["q4_0"]["error"]
    ), "blk.0 (noisy expert scaled 4.0) must vastly outweigh blk.1 (noisy expert scaled 0.25)"

    # Run runner derive --target-size on the full model (no --match filter) at a budget where
    # all-q8_0 exceeds target by ~one ffn_down_exps.weight (q8_0 -> q4_0) saving, OR on
    # --match ffn_down_exps.weight with ladder q8_0,q4_0 where exactly one of the two
    # ffn_down_exps.weight tensors fits at q8_0 and the other must drop to q4_0.
    one_down_delta = (
        rep["blk.0.ffn_down_exps.weight"]["types"]["q8_0"]["bytes"]
        - rep["blk.0.ffn_down_exps.weight"]["types"]["q4_0"]["bytes"]
    )
    base_q8_down = tmp_path / "base-q8-down.gguf"
    p_all_q8_down = tmp_path / "p-all-q8-down.json"
    p_all_q8_down.write_text(json.dumps({
        "default": "keep",
        "rules": [{"match": "ffn_down_exps.weight", "type": "q8_0"}],
    }))
    r = _run(runner_bin, "-m", m_fwd, "--quantize", base_q8_down, "--type-plan", p_all_q8_down)
    assert r.returncode == 0, r.stderr
    # Target saves exactly one ffn_down_exps.weight's (q8_0 -> q4_0) delta.
    target_bytes = base_q8_down.stat().st_size - one_down_delta

    out_fwd = tmp_path / "der-fwd.gguf"
    r = _run(
        runner_bin, "derive", "-m", m_fwd, "-o", out_fwd,
        "--target-size", str(target_bytes), "--ladder", "q8_0,q4_0",
        "--match", "ffn_down_exps.weight", "--group", "tensor",
        "--base-model", "test/gemma4-moe@0000000", "--license", "apache-2.0",
        "--corpus", corpus, "--window", "2:1:8", "-c", "64", "-t", "2", "--gpu", "off",
    )
    assert r.returncode == 0, f"derive fwd failed: {r.stderr}"
    rec_fwd = json.loads(Path(str(out_fwd) + ".derive.json").read_text())
    plan_fwd = json.loads((out_fwd.parent / rec_fwd["recipe"]["type_plan"]).read_text())
    _, got_fwd = _gguf(out_fwd)
    assert got_fwd["blk.0.ffn_down_exps.weight"][1] == T_Q8_0
    assert got_fwd["blk.1.ffn_down_exps.weight"][1] == T_Q4_0

    # Verify exact plan parity with scripts/type-plan-by-sensitivity.py.
    rp = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "type-plan-by-sensitivity.py"),
         str(out_fwd.parent / rec_fwd["recipe"]["quant_error"]),
         "--ladder", "q8_0,q4_0", "--match", "ffn_down_exps.weight", "--group", "tensor",
         "--save-bytes", str(rec_fwd["recipe"]["save_bytes"])],
        capture_output=True, encoding="utf-8", check=False, cwd=ROOT,
    )
    assert rp.returncode == 0, rp.stderr
    assert plan_fwd == json.loads(rp.stdout)

    # Swapping the two layers' scales reverses derive's choice.
    out_rev = tmp_path / "der-rev.gguf"
    r = _run(
        runner_bin, "derive", "-m", m_rev, "-o", out_rev,
        "--target-size", str(target_bytes), "--ladder", "q8_0,q4_0",
        "--match", "ffn_down_exps.weight", "--group", "tensor",
        "--base-model", "test/gemma4-moe@0000000", "--license", "apache-2.0",
        "--corpus", corpus, "--window", "2:1:8", "-c", "64", "-t", "2", "--gpu", "off",
    )
    assert r.returncode == 0, f"derive rev failed: {r.stderr}"
    _, got_rev = _gguf(out_rev)
    assert got_rev["blk.0.ffn_down_exps.weight"][1] == T_Q4_0
    assert got_rev["blk.1.ffn_down_exps.weight"][1] == T_Q8_0

    # Fidelity gate: compare CPU --score logprob drift vs F32 for the scale-aware derive output
    # (out_fwd: blk.0=q8_0, blk.1=q4_0) against the inverted blind plan (blk.0=q4_0, blk.1=q8_0).
    blind_plan = tmp_path / "blind.json"
    blind_plan.write_text(json.dumps({
        "default": "keep",
        "rules": [
            {"match": "blk.0.ffn_down_exps.weight", "type": "q4_0"},
            {"match": "blk.1.ffn_down_exps.weight", "type": "q8_0"},
        ],
    }))
    m_blind = tmp_path / "blind.gguf"
    r = _run(runner_bin, "-m", m_fwd, "--quantize", m_blind, "--type-plan", blind_plan)
    assert r.returncode == 0, r.stderr

    # Scale tensors stay bit-for-bit untouched in the derived model.
    _, src_t = _gguf(m_fwd)
    for i in range(LAYERS):
        assert got_fwd[f"blk.{i}.ffn_down_exps.scale"] == src_t[f"blk.{i}.ffn_down_exps.scale"]

    score_prompt = tmp_path / "score.txt"
    score_prompt.write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    lp_ref = _score_logprobs(runner_bin, m_fwd, score_prompt, tmp_path / "sc-ref.json")
    lp_aware = _score_logprobs(runner_bin, out_fwd, score_prompt, tmp_path / "sc-aware.json")
    lp_blind = _score_logprobs(runner_bin, m_blind, score_prompt, tmp_path / "sc-blind.json")
    assert len(lp_ref) >= 16 and len(lp_ref) == len(lp_aware) == len(lp_blind)

    mae_aware = sum(abs(a - b) for a, b in zip(lp_ref, lp_aware)) / len(lp_ref)
    mae_blind = sum(abs(a - b) for a, b in zip(lp_ref, lp_blind)) / len(lp_ref)
    assert mae_aware < mae_blind * 0.5, (
        f"scale-aware derive output must clearly beat inverted blind plan on F32 --score drift: "
        f"aware MAE={mae_aware:.6g} vs blind MAE={mae_blind:.6g}"
    )


def test_nvfp4_declines_gemma4_down_exps_and_never_clobbers_scale(runner_bin, tmp_path):
    """NVFP4 must decline blk.N.ffn_down_exps.weight on Gemma 4 so its [n_expert] .scale is never overwritten."""
    src = _make_gemma4_moe(tmp_path / "g4.gguf", [[0.85, 1.15], [1.25, 0.75]])
    qe = tmp_path / "qe-nvfp4.jsonl"
    r = _run(runner_bin, "-m", src, "--quant-error", qe, "--quant", "nvfp4")
    assert r.returncode == 0, r.stderr
    rep = _report(qe)
    for i in range(LAYERS):
        assert rep[f"blk.{i}.ffn_down_exps.weight"]["types"]["nvfp4"] is None

    out = tmp_path / "g4-nvfp4.gguf"
    r = _run(runner_bin, "-m", src, "--quantize", out, "--quant", "nvfp4")
    assert r.returncode == 0, r.stderr
    _, src_t = _gguf(src)
    _, got_t = _gguf(out)
    for i in range(LAYERS):
        assert got_t[f"blk.{i}.ffn_down_exps.scale"] == src_t[f"blk.{i}.ffn_down_exps.scale"]
        assert got_t[f"blk.{i}.ffn_down_exps.weight"][1] != T_NVFP4
