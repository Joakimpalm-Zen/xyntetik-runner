"""R4.48.8.6: IQ3_XXS and two-level NVFP4 writers, --quant-error costing, and derive --ladder keep,...

Tests cover:
- IQ3_XXS writer (unweighted and imatrix-weighted), decode round-trip, and Inkling NVFP4 -> IQ3_XXS fold;
- Two-level NVFP4 writer for BF16 Inkling expert layers (synthesizing per-expert F32 .scale companions),
  all 127 UE4M3 codes (0x00..0x7E) interop with llama.cpp's decode formula, two-level vs single-level RMSE,
  uniform pruning + NVFP4 synthesis, and clean refusal on Inkling dense tensors;
- --quant-error costing for iq3_xxs and nvfp4 (including loop.count and thread determinism);
- derive --target-size --ladder keep,nvfp4,iq3_xxs and keep,q8_0,nvfp4,iq3_xxs on a mixed NVFP4 + BF16
  Inkling fixture, checked against scripts/type-plan-by-sensitivity.py;
- CPU fidelity smoke (--score 20) confirming finite logits and tighter preservation for NVFP4 than IQ3_XXS.
"""
import json
import math
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling2_tiny import (  # noqa: E402
    ARR, BOOL, DREL, E, EXT, F32, FULL, HD, I32, K, LOOP_END, LOOP_START,
    N_BLOCK, NH, NKV, PASSES, RANK, STR, T_BF16, T_F32, T_NVFP4, T_Q8_0, U32,
    USED, VOCAB, VUSED, WIN, _bf16, _byte_tokens, _f32, _nvfp4, _q8_0, _write,
)

T_IQ3_XXS = 18
FACTORS = [0.3, 1.7, 0.9, 0.55]
STACKS = ("ffn_gate_up_exps", "ffn_down_exps")
_SIZE = {
    T_F32: (4, 1),
    T_BF16: (2, 1),
    T_Q8_0: (34, 32),
    T_IQ3_XXS: (98, 256),
    T_NVFP4: (36, 64),
    1: (2, 1),
}
E2M1_TABLE = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _ue4m3_to_f32(x):
    if x == 0 or x == 0x7F:
        return 0.0
    exp = (x >> 3) & 0xF
    man = x & 0x7
    if exp == 0:
        return math.ldexp(float(man), -9)
    return math.ldexp(1.0 + man / 8.0, exp - 7)


def _decode_nvfp4_llama_interop(raw_bytes, n_elem):
    """Exact llama.cpp dequantize_row_nvfp4 formula (block_nvfp4 = 4 UE4M3 scales + 32 nibbles)."""
    assert len(raw_bytes) == (n_elem // 64) * 36
    out = []
    for b in range(n_elem // 64):
        off = b * 36
        scales = raw_bytes[off:off + 4]
        qs = raw_bytes[off + 4:off + 36]
        for s in range(4):
            d = _ue4m3_to_f32(scales[s])
            sub_q = qs[s * 8:(s + 1) * 8]
            lo = [E2M1_TABLE[q & 0xF] * d for q in sub_q]
            hi = [E2M1_TABLE[q >> 4] * d for q in sub_q]
            out.extend(lo)
            out.extend(hi)
    return out


def _gguf(path):
    """Parse a GGUF v3 file into (kv, {name: (dims, type, data bytes)})."""
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
    out = {}
    for name, dims, typ, off in infos:
        n = 1
        for d in dims:
            n *= d
        ts, bs = _SIZE.get(typ, (0, 1))
        out[name] = (dims, typ, b[start + off:start + off + n // bs * ts] if ts else None)
    return kv, out


def _run(runner_bin, *args, force=False):
    env = dict(os.environ)
    env.pop("RUNNER_FORCE_REQUANT", None)
    env.pop("RUNNER_REQUANT_ONLY", None)
    if force:
        env["RUNNER_FORCE_REQUANT"] = "1"
    return subprocess.run(
        [str(runner_bin), *map(str, args)],
        capture_output=True, encoding="utf-8", errors="replace",
        env=env, timeout=300,
    )


def _plan(path, rules, default="keep"):
    path.write_text(json.dumps({"default": default, "rules": [{"match": m, "type": t} for m, t in rules]}))
    return path


def _report(path):
    return {r["tensor"]: r for r in (json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())}


def _mixed_inkling_gguf(path, h=256, ff=256, bf16_layers=(2, 3), bf16_std=0.03):
    """Tiny inkling2_loop GGUF with 256-wide rows (divides IQ3_XXS 256 and NVFP4 64),
    NVFP4 + .scale on layers 0..1 and BF16 (no .scale) on layers 2..3, mirroring Inkling 2 Mini."""
    nh, nkv, hd = h // 32, max(1, (h // 32) // 2), 32
    a = "inkling2_loop."
    kvs = [
        ("general.architecture", STR, "inkling2_loop"), ("general.alignment", U32, 32),
        (a + "block_count", U32, N_BLOCK), (a + "context_length", U32, 512),
        (a + "embedding_length", U32, h), (a + "feed_forward_length", U32, ff),
        (a + "attention.head_count", ARR, (I32, [nh] * N_BLOCK)),
        (a + "attention.head_count_kv", ARR, (I32, [nkv] * N_BLOCK)),
        (a + "attention.layer_norm_rms_epsilon", F32, 1e-6),
        (a + "expert_count", U32, E), (a + "expert_used_count", U32, USED),
        (a + "attention.key_length", U32, hd), (a + "attention.value_length", U32, hd),
        (a + "loop.count", U32, PASSES), (a + "loop.start", U32, LOOP_START),
        (a + "loop.end", U32, LOOP_END), (a + "loop.gate_rank", U32, RANK),
        (a + "vocab_size", U32, VOCAB), (a + "vocab_size_unpadded", U32, VUSED),
        (a + "expert_feed_forward_length", U32, ff),
        (a + "attention.key_length_swa", U32, hd), (a + "attention.value_length_swa", U32, hd),
        (a + "attention.sliding_window", U32, WIN),
        (a + "attention.sliding_window_pattern", ARR, (BOOL, [not x for x in FULL])),
        (a + "shortconv.l_cache", U32, K), (a + "expert_weights_scale", F32, 1.0),
        (a + "activation.situ_beta", F32, 4.0), (a + "activation.situ_linear_beta", F32, 25.0),
        (a + "attn_res.block_size", U32, 1), (a + "final_logit_softcapping", F32, 30.0),
        (a + "attention.relative_length", U32, DREL), (a + "attention.relative_extent", U32, EXT),
        (a + "attention.qk_clip", F32, 22.0), (a + "attention.relative_clip", F32, 8.0),
        (a + "attention.log_scale_floor", U32, 8192), (a + "attention.log_scale_alpha", F32, 0.1),
        (a + "loop.varying", BOOL, False),
        ("tokenizer.ggml.model", STR, "gpt2"), ("tokenizer.ggml.pre", STR, "inkling2"),
        ("tokenizer.ggml.tokens", ARR, (STR, _byte_tokens() + [f"<|s{i}|>" for i in range(VUSED - 256)])),
        ("tokenizer.ggml.token_type", ARR, (I32, [1] * 256 + [3] * (VUSED - 256))),
        ("tokenizer.ggml.merges", ARR, (STR, [])), ("tokenizer.ggml.eos_token_id", U32, 256),
    ]
    t = []
    seed = [100]

    def add_f(name, dims, scale=0.05, value=None, as_q8=False):
        seed[0] += 1
        n = 1
        for d in dims:
            n *= d
        data = struct.pack(f"<{n}f", *([value] * n)) if value is not None else _f32(n, scale, seed[0])
        if as_q8:
            t.append((name, dims, T_Q8_0, _q8_0(data)))
        else:
            t.append((name, dims, T_F32, data))

    def gaussian_bf16(n_el, std, s):
        rng = random.Random(s)
        vals = [rng.gauss(0.0, std) for _ in range(n_el)]
        raw_f32 = struct.pack(f"<{n_el}f", *vals)
        return _bf16(raw_f32)

    add_f("token_embd.weight", [h, VOCAB], 0.3, as_q8=True)
    add_f("token_embd_norm.weight", [h], value=1.0)
    add_f("output_norm.weight", [h], value=1.0)
    add_f("output_res_query.weight", [h], 0.1)
    add_f("output.weight", [h, VOCAB], 0.3, as_q8=True)
    for b in range(N_BLOCK):
        p = f"blk.{b}."
        c_dim = nkv * hd
        add_f(p + "attn_q.weight", [h, nh * hd], as_q8=True)
        add_f(p + "attn_q_clip.weight", [hd, nh], value=1.0)
        add_f(p + "attn_k.weight", [h, c_dim], as_q8=True)
        add_f(p + "attn_v.weight", [h, c_dim], as_q8=True)
        add_f(p + "attn_k_conv.weight", [K, c_dim], 0.3)
        add_f(p + "attn_v_conv.weight", [K, c_dim], 0.3)
        if FULL[b]:
            add_f(p + "attn_k_conv_proj.weight", [h, nkv * K])
            add_f(p + "attn_v_conv_proj.weight", [h, nkv * K])
            for x in ("q", "k", "v"):
                add_f(p + f"attn_{x}_inp_norm.weight", [h], value=1.0)
        else:
            add_f(p + "attn_norm.weight", [h], value=1.0)
        add_f(p + "attn_gate.weight", [h, nh * hd], as_q8=True)
        add_f(p + "attn_output.weight", [nh * hd, h], as_q8=True)
        add_f(p + "attn_sink_proj.weight", [h, nh], 0.22)
        add_f(p + "attn_rel_bias.weight", [DREL, EXT], 0.1)
        add_f(p + "attn_rel.weight", [h, nh * DREL], as_q8=True)
        add_f(p + "attn_rel_clip.weight", [DREL, nh], value=1.0)
        add_f(p + "ffn_gate_inp.weight", [h, E])
        add_f(p + "exp_probs_b.bias", [E], value=0.0)
        add_f(p + "ffn_norm.weight", [h], value=1.0)
        add_f(p + "attn_post_conv.weight", [K, h], 0.3)
        add_f(p + "ffn_post_conv.weight", [K, h], 0.3)
        if b >= 1:
            add_f(p + "attn_res_query.weight", [h, 3] if FULL[b] else [h], 0.1)
        add_f(p + "ffn_res_query.weight", [h], 0.1)
        if LOOP_START <= b < LOOP_END:
            for r in ("q", "k", "v", "ffn"):
                add_f(p + f"loop_{r}_ctrl_down.weight", [2 * h, RANK], 0.15)
                add_f(p + f"loop_{r}_ctrl_forget.weight", [RANK, h], 0.15)
                add_f(p + f"loop_{r}_ctrl_gain.weight", [RANK, h], 0.15)
            add_f(p + "loop_msg_query.weight", [h, 4], 0.2)
        if b in bf16_layers:
            seed[0] += 1
            t.append((p + "ffn_gate_up_exps.weight", [h, 2 * ff, E], T_BF16,
                      gaussian_bf16(h * 2 * ff * E, bf16_std, seed[0])))
            seed[0] += 1
            t.append((p + "ffn_down_exps.weight", [ff, h, E], T_BF16,
                      gaussian_bf16(ff * h * E, bf16_std, seed[0])))
        else:
            seed[0] += 1
            t.append((p + "ffn_gate_up_exps.weight", [h, 2 * ff, E], T_NVFP4,
                      _nvfp4(h * 2 * ff * E, seed[0], varied=True)))
            seed[0] += 1
            t.append((p + "ffn_down_exps.weight", [ff, h, E], T_NVFP4,
                      _nvfp4(ff * h * E, seed[0], varied=True)))
            for which in ("gate_up", "down"):
                t.append((p + f"ffn_{which}_exps.scale", [E], T_F32, struct.pack(f"<{E}f", *FACTORS)))
    return _write(path, kvs, t)


def _write_gguf_imatrix(path, entries):
    """Write a llama.cpp-compatible GGUF imatrix: {tensor_name: (n_per_row, n_mat, flat_values)}."""
    kvs = [("general.architecture", STR, "imatrix"), ("general.alignment", U32, 32)]
    tensors = []
    for name, (n_per_row, n_mat, vals) in entries.items():
        dims = [n_per_row, n_mat] if n_mat > 1 else [n_per_row]
        tensors.append((f"{name}.in_sum2", dims, T_F32, struct.pack(f"<{len(vals)}f", *vals)))
        tensors.append((f"{name}.counts", [1, n_mat] if n_mat > 1 else [1], T_F32,
                        struct.pack(f"<{n_mat}f", *([1.0] * n_mat))))
    return _write(path, kvs, tensors)


def _bf16_to_f32_list(raw_bytes):
    vals = []
    for (u16,) in struct.iter_unpack("<H", raw_bytes):
        vals.append(struct.unpack("<f", struct.pack("<I", u16 << 16))[0])
    return vals


@pytest.fixture(scope="module")
def mixed_ink(tmp_path_factory):
    d = tmp_path_factory.mktemp("r48886")
    return d, _mixed_inkling_gguf(d / "mixed-ink.gguf")


def test_iq3_xxs_roundtrip_and_imatrix_weighting(runner_bin, tmp_path):
    """IQ3_XXS writes valid 98 B / 256-weight blocks (MOSTLY_IQ3_XXS = 23) and imatrix lowers weighted error."""
    cols, rows = 256, 8
    rng = random.Random(20261126)
    weights = [rng.gauss(0.0, 0.25) for _ in range(rows * cols)]
    # Make row 1 all zeros (zero sub-block guard) and row 2 mixed signs.
    for c in range(cols):
        weights[1 * cols + c] = 0.0
    src = _write(
        tmp_path / "f32.gguf",
        [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)],
        [
            ("output_norm.weight", [cols], T_F32, struct.pack(f"<{cols}f", *([1.0] * cols))),
            ("blk.0.ffn_gate.weight", [cols, rows], T_F32, struct.pack(f"<{rows * cols}f", *weights)),
        ],
    )
    # Heavily emphasize first 128 columns in imatrix.
    imp = [100.0] * 128 + [1.0] * 128
    imx = _write_gguf_imatrix(tmp_path / "imx.gguf", {"blk.0.ffn_gate.weight": (cols, 1, imp)})

    out_plain = tmp_path / "iq3-plain.gguf"
    out_imx = tmp_path / "iq3-imx.gguf"
    r1 = _run(runner_bin, "-m", src, "--quantize", out_plain, "--quant", "iq3_xxs")
    assert r1.returncode == 0, r1.stderr
    r2 = _run(runner_bin, "-m", src, "--quantize", out_imx, "--quant", "iq3_xxs", "--imatrix", imx)
    assert r2.returncode == 0, r2.stderr

    kv_p, got_p = _gguf(out_plain)
    _, got_i = _gguf(out_imx)
    assert kv_p["general.file_type"] == 23
    assert got_p["blk.0.ffn_gate.weight"][1] == T_IQ3_XXS
    assert len(got_p["blk.0.ffn_gate.weight"][2]) == rows * 98
    assert got_p["blk.0.ffn_gate.weight"][2] != got_i["blk.0.ffn_gate.weight"][2]

    rep_i = tmp_path / "qe-imx.jsonl"
    r3 = _run(runner_bin, "-m", src, "--quant-error", rep_i, "--quant", "iq3_xxs", "--imatrix", imx)
    assert r3.returncode == 0, r3.stderr
    entry = _report(rep_i)["blk.0.ffn_gate.weight"]
    assert entry["weighted"] is True
    assert entry["types"]["iq3_xxs"]["bytes"] == rows * 98
    assert math.isfinite(entry["types"]["iq3_xxs"]["error"]) and entry["types"]["iq3_xxs"]["error"] > 0.0


def test_nvfp4_all_127_ue4m3_codes_and_llama_interop(runner_bin, tmp_path):
    """NVFP4 encodes all 127 finite UE4M3 codes (0x00..0x7E, never 0x7F) and decodes identically under llama.cpp's formula."""
    # Construct a 128-subblock (2048-element) 2D tensor [256, 8] whose first 127 sub-blocks
    # have sub_amax = 6.0 * ue4m3(u) * f_target so every code u in 0x00..0x7E is hit.
    f_target = 0.01  # arbitrary realistic global scale -> global_amax = 2688 * 0.01 = 26.88
    cols, rows = 256, 8
    vals = [0.0] * (cols * rows)
    for u in range(127):
        base = u * 16
        d_u = _ue4m3_to_f32(u)
        vals[base] = 6.0 * d_u * f_target
        vals[base + 1] = -3.0 * d_u * f_target
        vals[base + 8] = 1.5 * d_u * f_target
    src = _write(
        tmp_path / "ue127.gguf",
        [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)],
        [
            ("output_norm.weight", [cols], T_F32, struct.pack(f"<{cols}f", *([1.0] * cols))),
            ("blk.0.ffn_gate.weight", [cols, rows], T_F32, struct.pack(f"<{cols * rows}f", *vals)),
        ],
    )
    out = tmp_path / "ue127-nvfp4.gguf"
    r = _run(runner_bin, "-m", src, "--quantize", out, "--quant", "nvfp4")
    assert r.returncode == 0, r.stderr

    kv, got = _gguf(out)
    assert kv["general.file_type"] == 39
    dims, typ, raw = got["blk.0.ffn_gate.weight"]
    assert dims == [cols, rows] and typ == T_NVFP4
    assert "blk.0.ffn_gate.scale" in got
    sdims, styp, sraw = got["blk.0.ffn_gate.scale"]
    assert sdims == [1] and styp == T_F32
    (f_got,) = struct.unpack("<f", sraw)
    assert f_got == pytest.approx(f_target, rel=1e-6)

    seen_codes = []
    for blk in range(len(raw) // 36):
        seen_codes.extend(raw[blk * 36:blk * 36 + 4])
    assert 0x7F not in seen_codes
    assert seen_codes[:127] == list(range(127)), f"missing/mismatched UE4M3 codes: {seen_codes[:127]}"

    # Interop: llama.cpp's dequantize_row_nvfp4 formula * f_got reproduces every constructed value exactly.
    dec = _decode_nvfp4_llama_interop(raw, cols * rows)
    for u in range(127):
        base = u * 16
        for idx in (0, 1, 8):
            assert dec[base + idx] * f_got == pytest.approx(vals[base + idx], rel=1e-6, abs=1e-7)


def test_nvfp4_two_level_bf16_experts_synthesize_scales_and_report_rmse(runner_bin, mixed_ink):
    """Converting BF16 Inkling layers 2..3 to NVFP4 synthesizes [E] F32 .scale, keeps layers 0..1 bit-for-bit,
    loads cleanly, decodes identically under llama.cpp's NVFP4 formula, and beats single-level RMSE."""
    d, src = mixed_ink
    out = d / "ink-nvfp4.gguf"
    plan = _plan(d / "bf16-to-nvfp4.json", [("blk.2.ffn_", "nvfp4"), ("blk.3.ffn_", "nvfp4")])
    r = _run(runner_bin, "-m", src, "--quantize", out, "--type-plan", plan)
    assert r.returncode == 0, r.stderr

    _, s_ten = _gguf(src)
    _, g_ten = _gguf(out)

    # Layers 0..1 kept bit-for-bit (both weight and scale).
    for b in (0, 1):
        for s in STACKS:
            for part in ("weight", "scale"):
                n = f"blk.{b}.{s}.{part}"
                assert g_ten[n] == s_ten[n], n

    # Layers 2..3 converted to NVFP4 + synthesized [E] F32 .scale, with tight two-level RMSE.
    for b in (2, 3):
        for s in STACKS:
            wn, sn = f"blk.{b}.{s}.weight", f"blk.{b}.{s}.scale"
            assert g_ten[wn][1] == T_NVFP4
            assert sn in g_ten, f"missing synthesized {sn}"
            assert g_ten[sn][0] == [E] and g_ten[sn][1] == T_F32
            facs = struct.unpack(f"<{E}f", g_ten[sn][2])
            assert all(math.isfinite(f) and f > 0.0 for f in facs)

            # Check two-level RMSE via llama.cpp's decode formula against the BF16 source,
            # and compare against single-level (f = 1.0) on the same weights.
            src_f32 = _bf16_to_f32_list(s_ten[wn][2])
            dec_raw = _decode_nvfp4_llama_interop(g_ten[wn][2], len(src_f32))
            per_exp = len(src_f32) // E
            sq_err = 0.0
            sq_ref = 0.0
            for idx, (ref_v, raw_v) in enumerate(zip(src_f32, dec_raw)):
                got_v = raw_v * facs[idx // per_exp]
                diff = got_v - ref_v
                sq_err += diff * diff
                sq_ref += ref_v * ref_v
            rel_rmse = math.sqrt(sq_err / sq_ref)
            # Theoretical E2M1 Gaussian relative RMSE is ~0.095; two-level hits it cleanly.
            assert rel_rmse < 0.11, f"{wn}: two-level NVFP4 rel_rmse={rel_rmse:.4f}"

            # Single-level (f = 1.0) on N(0, 0.03^2) falls into UE4M3 subnormals; verify two-level wins clearly.
            sq_single = 0.0
            for blk_off in range(0, len(src_f32), 16):
                sub = src_f32[blk_off:blk_off + 16]
                d_id = max(abs(v) for v in sub) / 6.0
                best_u = min(range(127), key=lambda u: abs(_ue4m3_to_f32(u) - d_id))
                d1 = _ue4m3_to_f32(best_u)
                for v in sub:
                    q_v = min((lvl * d1 for lvl in E2M1_TABLE), key=lambda c: abs(c - v))
                    sq_single += (q_v - v) ** 2
            single_rmse = math.sqrt(sq_single / sq_ref)
            assert rel_rmse < single_rmse, f"{wn}: two-level {rel_rmse:.4f} vs single-level {single_rmse:.4f}"

    assert _run(runner_bin, "-m", out, "--info", "fit", "-c", "64", "--gpu", "off").returncode == 0
    gen = _run(runner_bin, "-m", out, "--gpu", "off", "-t", "2", "-c", "64", "-p", "hello", "-n", "3", "--temp", "0")
    assert gen.returncode == 0, gen.stderr


def test_nvfp4_declined_on_inkling_dense_and_null_in_quant_error(runner_bin, mixed_ink):
    """Inkling loader refuses NVFP4 on dense tensors: quantize declines it cleanly and --quant-error emits null."""
    d, src = mixed_ink
    out = d / "ink-dense-nvfp4.gguf"
    plan = _plan(d / "dense-nvfp4.json", [("blk.0.attn_q.weight", "nvfp4")])
    r = _run(runner_bin, "-m", src, "--quantize", out, "--type-plan", plan)
    assert r.returncode == 0, r.stderr
    _, s_ten = _gguf(src)
    _, g_ten = _gguf(out)
    assert g_ten["blk.0.attn_q.weight"] == s_ten["blk.0.attn_q.weight"]
    assert "blk.0.attn_q.scale" not in g_ten
    assert _run(runner_bin, "-m", out, "--info", "fit", "-c", "64", "--gpu", "off").returncode == 0

    rep = d / "qe-nvfp4-dense.jsonl"
    r2 = _run(runner_bin, "-m", src, "--quant-error", rep, "--quant", "nvfp4,iq3_xxs")
    assert r2.returncode == 0, r2.stderr
    rows = _report(rep)
    assert rows["blk.0.attn_q.weight"]["types"]["nvfp4"] is None
    assert rows["blk.2.ffn_down_exps.weight"]["types"]["nvfp4"] is not None
    assert rows["blk.0.ffn_down_exps.weight"]["types"]["nvfp4"]["error"] == 0.0


def test_pruned_mixed_inkling_with_nvfp4_and_iq3_xxs_loads(runner_bin, mixed_ink):
    """Uniform prune (E=4->2) + BF16->NVFP4 (synthesizing [2] .scale) + NVFP4->IQ3_XXS (folding & dropping .scale)."""
    d, src = mixed_ink
    prune = d / "uprune.json"
    prune.write_text(json.dumps({"layer_0": [0, 2], "layer_1": [3, 1], "layer_2": [1, 2], "layer_3": [2, 0]}))
    plan = _plan(d / "mixed-plan.json", [
        ("blk.1.ffn_down_exps.weight", "iq3_xxs"),
        ("blk.2.ffn_down_exps.weight", "nvfp4"),
        ("blk.3.ffn_gate_up_exps.weight", "nvfp4"),
    ])
    out = d / "pruned-mixed.gguf"
    r = _run(runner_bin, "-m", src, "--quantize", out, "--prune-experts", prune, "--type-plan", plan)
    assert r.returncode == 0, r.stderr

    kv, got = _gguf(out)
    assert kv["inkling2_loop.expert_count"] == 2
    assert got["blk.1.ffn_down_exps.weight"][1] == T_IQ3_XXS
    assert "blk.1.ffn_down_exps.scale" not in got
    assert got["blk.2.ffn_down_exps.weight"][1] == T_NVFP4
    assert got["blk.2.ffn_down_exps.scale"][0] == [2]
    assert got["blk.3.ffn_gate_up_exps.weight"][1] == T_NVFP4
    assert got["blk.3.ffn_gate_up_exps.scale"][0] == [2]
    assert got["blk.0.ffn_down_exps.weight"][1] == T_NVFP4
    assert got["blk.0.ffn_down_exps.scale"][0] == [2]
    assert _run(runner_bin, "-m", out, "--info", "fit", "-c", "64", "--gpu", "off").returncode == 0
    gen = _run(runner_bin, "-m", out, "--gpu", "off", "-t", "2", "-c", "64", "-p", "hello", "-n", "3", "--temp", "0")
    assert gen.returncode == 0, gen.stderr


def test_quant_error_device_types_ordering_and_thread_determinism(runner_bin, mixed_ink):
    """--quant-error orders q8_0 < nvfp4 < iq3_xxs on BF16 expert stacks and matches across -t 1 and -t 4."""
    d, src = mixed_ink
    r1_path, r4_path = d / "qe-t1.jsonl", d / "qe-t4.jsonl"
    r1 = _run(runner_bin, "-m", src, "--quant-error", r1_path, "--quant", "q8_0,nvfp4,iq3_xxs", "-t", "1")
    r4 = _run(runner_bin, "-m", src, "--quant-error", r4_path, "--quant", "q8_0,nvfp4,iq3_xxs", "-t", "4")
    assert r1.returncode == 0, r1.stderr
    assert r4.returncode == 0, r4.stderr
    assert r1_path.read_bytes() == r4_path.read_bytes()

    rows = _report(r1_path)
    for b in (2, 3):
        for s in STACKS:
            tmap = rows[f"blk.{b}.{s}.weight"]["types"]
            e_q8 = tmap["q8_0"]["error"]
            e_nv = tmap["nvfp4"]["error"]
            e_iq = tmap["iq3_xxs"]["error"]
            assert 0.0 < e_q8 < e_nv < e_iq, (b, s, e_q8, e_nv, e_iq)


def test_derive_ladder_keep_rung_zero_on_mixed_nvfp4_bf16_inkling(runner_bin, mixed_ink):
    """R4.48.8.6 red-first gate: derive --target-size --ladder keep,nvfp4,iq3_xxs and keep,q8_0,nvfp4,iq3_xxs
    on a mixed NVFP4 (layers 0..1) + BF16 (layers 2..3) Inkling parent."""
    d, src = mixed_ink
    parent_size = src.stat().st_size

    # Pre-cost to choose two realistic targets:
    # (1) target_mid: fits by compressing BF16 layers 2..3 to nvfp4 while keeping NVFP4 layers 0..1 untouched;
    # (2) target_tight: requires demoting layers 2..3 to nvfp4 AND demoting at least one layer to iq3_xxs.
    pre = d / "pre-qe.jsonl"
    r = _run(runner_bin, "-m", src, "--quant-error", pre, "--quant", "q8_0,nvfp4,iq3_xxs")
    assert r.returncode == 0, r.stderr
    rows = _report(pre)
    _, s_ten = _gguf(src)
    bf16_saving_to_nvfp4 = sum(
        len(s_ten[f"blk.{b}.{s}.weight"][2]) - rows[f"blk.{b}.{s}.weight"]["types"]["nvfp4"]["bytes"]
        for b in (2, 3) for s in STACKS
    )
    extra_iq3_saving = sum(
        rows[f"blk.{b}.{s}.weight"]["types"]["nvfp4"]["bytes"] - rows[f"blk.{b}.{s}.weight"]["types"]["iq3_xxs"]["bytes"]
        for b in range(N_BLOCK) for s in STACKS
    )

    corpus = d / "derive-corpus.txt"
    corpus.write_text("The quick brown fox jumps over the lazy dog. " * 4)

    # Target 1: keep,nvfp4,iq3_xxs with target between parent and all-nvfp4 floor
    target_mid = parent_size - int(bf16_saving_to_nvfp4 * 0.75)
    out_mid = d / "derived-mid.gguf"
    r_mid = _run(
        runner_bin, "derive", "-m", src, "-o", out_mid,
        "--target-size", str(target_mid),
        "--ladder", "keep,nvfp4,iq3_xxs",
        "--match", "_exps.weight",
        "--group", "tensor",
        "--base-model", "test/inkling-mixed@0000000",
        "--license", "apache-2.0",
        "--corpus", corpus, "--window", "2:1:8", "-c", "64", "-t", "2", "--gpu", "off",
    )
    assert r_mid.returncode == 0, r_mid.stderr
    assert out_mid.stat().st_size <= target_mid
    rec_mid = json.loads(Path(str(out_mid) + ".derive.json").read_text())
    plan_mid = json.loads((out_mid.parent / rec_mid["recipe"]["type_plan"]).read_text())
    assert plan_mid["default"] == "keep"
    # Verify layers 0..1 stayed NVFP4 bit-for-bit and the file loads.
    _, g_mid = _gguf(out_mid)
    for b in (0, 1):
        for s in STACKS:
            assert g_mid[f"blk.{b}.{s}.weight"] == s_ten[f"blk.{b}.{s}.weight"]
            assert g_mid[f"blk.{b}.{s}.scale"] == s_ten[f"blk.{b}.{s}.scale"]
    assert _run(runner_bin, "-m", out_mid, "--info", "fit", "-c", "64", "--gpu", "off").returncode == 0

    # Target 2: keep,q8_0,nvfp4,iq3_xxs tighter than all-nvfp4 -> must use iq3_xxs on some stacks too.
    target_tight = parent_size - bf16_saving_to_nvfp4 - int(extra_iq3_saving * 0.45)
    out_tight = d / "derived-tight.gguf"
    r_tight = _run(
        runner_bin, "derive", "-m", src, "-o", out_tight,
        "--target-size", str(target_tight),
        "--ladder", "keep,q8_0,nvfp4,iq3_xxs",
        "--match", "_exps.weight",
        "--group", "tensor",
        "--base-model", "test/inkling-mixed@0000000",
        "--license", "apache-2.0",
        "--corpus", corpus, "--window", "2:1:8", "-c", "64", "-t", "2", "--gpu", "off",
    )
    assert r_tight.returncode == 0, r_tight.stderr
    assert out_tight.stat().st_size <= target_tight
    _, g_tight = _gguf(out_tight)
    types_seen = {g_tight[f"blk.{b}.{s}.weight"][1] for b in range(N_BLOCK) for s in STACKS}
    assert T_IQ3_XXS in types_seen, f"expected some stacks demoted to IQ3_XXS, got {types_seen}"
    assert _run(runner_bin, "-m", out_tight, "--info", "fit", "-c", "64", "--gpu", "off").returncode == 0
    gen = _run(runner_bin, "-m", out_tight, "--gpu", "off", "-t", "2", "-c", "64", "-p", "hello", "-n", "3", "--temp", "0")
    assert gen.returncode == 0, gen.stderr

    # Cross-check plan parity with scripts/type-plan-by-sensitivity.py.
    rec_tight = json.loads(Path(str(out_tight) + ".derive.json").read_text())
    ours = json.loads((out_tight.parent / rec_tight["recipe"]["type_plan"]).read_text())
    py = subprocess.run(
        [sys.executable, str(ROOT / "scripts/type-plan-by-sensitivity.py"),
         str(out_tight.parent / rec_tight["recipe"]["quant_error"]),
         "--ladder", "keep,q8_0,nvfp4,iq3_xxs",
         "--match", "_exps.weight", "--group", "tensor",
         "--save-bytes", str(rec_tight["recipe"]["save_bytes"])],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert py.returncode == 0, py.stderr
    assert ours == json.loads(py.stdout)

    # Refusal: keep in a non-zero rung is refused.
    r_bad = _run(
        runner_bin, "derive", "-m", src, "-o", d / "bad-keep.gguf",
        "--target-size", str(target_mid),
        "--ladder", "nvfp4,keep",
        "--match", "_exps.weight",
        "--base-model", "test/inkling-mixed@0000000",
        "--license", "apache-2.0",
        "--corpus", corpus, "--window", "2:1:8", "-c", "64", "-t", "2", "--gpu", "off",
    )
    assert r_bad.returncode != 0


def test_fidelity_smoke_nvfp4_and_iq3_xxs(runner_bin, mixed_ink):
    """Fidelity read (--score 20) on mixed Inkling: zero NaN/Inf and NVFP4 preserves parent log-probs closer than IQ3_XXS."""
    d, src = mixed_ink
    corpus = d / "smoke.txt"
    corpus.write_text("The quick brown fox jumps over the lazy dog. " * 6)

    nv_out = d / "fid-nvfp4.gguf"
    iq_out = d / "fid-iq3.gguf"
    p_nv = _plan(d / "fid-nv.json", [("blk.2.ffn_", "nvfp4"), ("blk.3.ffn_", "nvfp4")])
    p_iq = _plan(d / "fid-iq.json", [("blk.2.ffn_", "iq3_xxs"), ("blk.3.ffn_", "iq3_xxs")])
    assert _run(runner_bin, "-m", src, "--quantize", nv_out, "--type-plan", p_nv).returncode == 0
    assert _run(runner_bin, "-m", src, "--quantize", iq_out, "--type-plan", p_iq).returncode == 0

    scores = {}
    for name, mdl in (("ref", src), ("nvfp4", nv_out), ("iq3_xxs", iq_out)):
        sp = d / f"score-{name}.json"
        r = _run(runner_bin, "-m", mdl, "--score", "20", "-f", corpus, "--score-tokens", "24",
                 "-c", "64", "-t", "2", "--gpu", "off", "--score-out", sp)
        assert r.returncode == 0, r.stderr
        scores[name] = json.loads(sp.read_text())

    def mean_logprob_l1(cand):
        ref_lp = scores["ref"]["logprobs"]
        cand_lp = cand["logprobs"]
        assert len(ref_lp) == len(cand_lp) and len(ref_lp) > 0
        for v in cand_lp:
            assert math.isfinite(v)
        return sum(abs(a - b) for a, b in zip(ref_lp, cand_lp)) / len(ref_lp)

    d_nv = mean_logprob_l1(scores["nvfp4"])
    d_iq = mean_logprob_l1(scores["iq3_xxs"])
    assert math.isfinite(d_nv) and math.isfinite(d_iq)
    assert d_nv < d_iq, f"expected NVFP4 ({d_nv:.6f}) closer to BF16 parent than IQ3_XXS ({d_iq:.6f})"


def test_verify_llama_harness_contract(runner_bin, tmp_path):
    """scripts/verify-device-writers-llama.py satisfies llama-quantize's real CLI/metadata contract:
    (1) passes --imatrix and --pure for IQ3_XXS (no separate unweighted leg, since llama-quantize refuses
        IQ3_XXS without --imatrix; token_embd.weight and output.weight have no imatrix entry and verify the
        unweighted path);
    (2) writes imatrix.datasets, imatrix.chunk_count, and imatrix.chunk_size metadata keys;
    (3) asserts every expected IQ3_XXS tensor is compared (9 total: 7 weighted + 2 unweighted);
    (4) surfaces subprocess stderr on failure."""
    mock_py = tmp_path / "mock_llama_quantize.py"
    mock_py.write_text(
        f"""#!{sys.executable}
import struct, subprocess, sys
from pathlib import Path

def read_kv(path):
    b = Path(path).read_bytes()
    o = [4]
    def take(fmt):
        v = struct.unpack_from("<" + fmt, b, o[0]); o[0] += struct.calcsize("<" + fmt); return v[0]
    def text():
        n = take("Q"); o[0] += n; return b[o[0] - n:o[0]].decode("utf-8")
    sc = {{0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}}
    def val(t):
        if t == 8: return text()
        if t == 9: et, n = take("I"), take("Q"); return [val(et) for _ in range(n)]
        return take(sc[t])
    take("I"); _nt, nkv = take("Q"), take("Q")
    kv = {{}}
    for _ in range(nkv):
        k = text(); kv[k] = val(take("I"))
    return kv

args = sys.argv[1:]
pure = False
imx = None
pos = []
i = 0
while i < len(args):
    if args[i] == "--pure":
        pure = True; i += 1
    elif args[i] == "--imatrix":
        imx = args[i + 1]; i += 2
    else:
        pos.append(args[i]); i += 1
src, dst, ftype = pos[0], pos[1], pos[2]
if ftype == "IQ3_XXS" and imx is None:
    sys.stderr.write("error: IQ3_XXS requires --imatrix\\n")
    sys.exit(1)
if imx is not None:
    kv = read_kv(imx)
    for req in ("imatrix.datasets", "imatrix.chunk_count", "imatrix.chunk_size"):
        if req not in kv:
            sys.stderr.write(f"load_imatrix: missing imatrix metadata ({{req}})\\n")
            sys.exit(1)
    if not isinstance(kv["imatrix.datasets"], list) or not kv["imatrix.datasets"]:
        sys.stderr.write("load_imatrix: imatrix.datasets must be non-empty string array\\n")
        sys.exit(1)
if not pure:
    # Mimic llama-quantize without --pure: only 4 tensors get IQ3_XXS
    plan = Path(dst).with_suffix(".mix.json")
    plan.write_text('{{"default":"keep","rules":[{{"match":"attn_q","type":"iq3_xxs"}},{{"match":"attn_k","type":"iq3_xxs"}},{{"match":"attn_output","type":"iq3_xxs"}},{{"match":"ffn_up","type":"iq3_xxs"}}]}}')
    cmd = [{str(runner_bin)!r}, "-m", src, "--quantize", dst, "--type-plan", str(plan), "--imatrix", imx]
else:
    cmd = [{str(runner_bin)!r}, "-m", src, "--quantize", dst, "--quant", "iq3_xxs", "--imatrix", imx]
r = subprocess.run(cmd, capture_output=True, text=True)
sys.stderr.write(r.stderr)
sys.exit(r.returncode)
""",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        mock_exe = tmp_path / "mock_llama_quantize.bat"
        mock_exe.write_text(f'@"{sys.executable}" "{mock_py}" %*\r\n', encoding="utf-8")
    else:
        mock_py.chmod(0o755)
        mock_exe = mock_py

    script = ROOT / "scripts" / "verify-device-writers-llama.py"
    r = subprocess.run(
        [sys.executable, str(script), "--runner", str(runner_bin), "--llama-quantize", str(mock_exe)],
        capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 0, f"harness failed:\nstdout={r.stdout}\nstderr={r.stderr}"
    data = json.loads(r.stdout)
    iq = data["iq3_xxs_vs_llama_quantize"]
    assert iq["total_tensors"] == 9
    assert iq["weighted_tensors"] == 7
    assert iq["unweighted_tensors"] == 2
    assert iq["diff_bytes"] == 0

    # Verify stderr is printed when a subprocess fails.
    bad_py = tmp_path / "bad_llama_quantize.py"
    bad_py.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stderr.write('SENTINEL_LLAMA_ERR\\n')\nsys.exit(2)\n",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        bad_exe = tmp_path / "bad_llama_quantize.bat"
        bad_exe.write_text(f'@"{sys.executable}" "{bad_py}" %*\r\n', encoding="utf-8")
    else:
        bad_py.chmod(0o755)
        bad_exe = bad_py

    rb = subprocess.run(
        [sys.executable, str(script), "--runner", str(runner_bin), "--llama-quantize", str(bad_exe)],
        capture_output=True, text=True, timeout=60,
    )
    assert rb.returncode != 0
    assert "SENTINEL_LLAMA_ERR" in rb.stderr, f"stderr not surfaced:\n{rb.stderr}"


def test_nvfp4_8scale_scan_invariants(runner_bin, tmp_path):
    """R4.48.8.12: NVFP4 writer uses 8-candidate UE4M3 scan (ue_6 + [-1..+6]) by plain unweighted MSE:
    - red-first block where static-6 is not MSE-optimal;
    - Four Over Six scale (amax / (4*f)) is inside the candidate set and achieves 0 MSE on a 4/6-exact block;
    - ties go to candidate nearest ue_6, static-6 first;
    - outer scale stays amax / (6 * 448);
    - per-block scan MSE <= static-6 MSE on 100% of blocks of an Inkling-shaped BF16 matrix (>22% total MSE drop);
    - -t 1 equals -t 4 byte-for-byte;
    - imatrix does NOT change NVFP4 bytes (imatrix_weighted_type(T_NVFP4) stays false)."""
    ue_tbl = [_ue4m3_to_f32(u) for u in range(127)]

    def to_ue(x):
        if not (x > 0.0):
            return 0
        if x >= 448.0:
            return 126
        lo, hi = 0, 126
        while hi - lo > 1:
            m = (lo + hi) >> 1
            if ue_tbl[m] <= x:
                lo = m
            else:
                hi = m
        dlo, dhi = x - ue_tbl[lo], ue_tbl[hi] - x
        if dlo < dhi:
            return lo
        if dhi < dlo:
            return hi
        return hi if (lo & 1) else lo

    # Exhaustive check: Four Over Six's ue_4 always lies in ue_6 + [-1..+6].
    for i in range(126):
        for x6 in (ue_tbl[i], 0.5 * (ue_tbl[i] + ue_tbl[i + 1]), ue_tbl[i + 1]):
            u6, u4 = to_ue(x6), to_ue(1.5 * x6)
            assert -1 <= (u4 - u6) <= 6, f"x6={x6}: u6={u6}, u4={u4}"

    # Crafted 4-sub-block row (64 values) with amax = 6*448 = 2688 in sub-block 3 so f_got == 1.0 exactly:
    # - sub-block 0: [6.0, 5.0, ..., 5.0] -> static-6 (d=1.0) MSE=15.0; scan picks d=1.625 (ue_6+5, code 61)
    #   where 4*1.625=6.5 and 3*1.625=4.875 -> MSE = 0.25 + 15*0.015625 = 0.484375 (31x lower than static-6)!
    # - sub-block 1: [6.0, 4.5, ..., 4.5] -> 4/6 exact at d_4=1.5 (ue_6+4), MSE=0.0!
    # - sub-block 2: [6.0, 3.0, 1.5, 0, ...] -> exact at BOTH d_6=1.0 and d_4=1.5; tie goes to static-6 (d=1.0)!
    # - sub-block 3: [2688.0, 0, ...] -> sets outer scale f = 2688 / (6*448) = 1.0 exactly.
    row = (
        [6.0] + [5.0] * 15
        + [6.0] + [4.5] * 15
        + [6.0, 3.0, 1.5] + [0.0] * 13
        + [2688.0] + [0.0] * 15
    )
    ones64 = struct.pack("<64f", *([1.0] * 64))
    src_c = _write(
        tmp_path / "crafted.gguf",
        [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)],
        [("output_norm.weight", [64], T_F32, ones64),
         ("blk.0.ffn_gate.weight", [64, 1], T_F32, struct.pack("<64f", *row))],
    )
    out_c = tmp_path / "crafted-nvfp4.gguf"
    r = _run(runner_bin, "-m", src_c, "--quantize", out_c, "--quant", "nvfp4")
    assert r.returncode == 0, r.stderr
    _, got_c = _gguf(out_c)
    (f_c,) = struct.unpack("<f", got_c["blk.0.ffn_gate.scale"][2])
    assert f_c == pytest.approx(1.0, rel=1e-6), f"outer scale must stay amax/(6*448)=1.0, got {f_c}"
    raw_c = got_c["blk.0.ffn_gate.weight"][2]
    dec_c = [v * f_c for v in _decode_nvfp4_llama_interop(raw_c, 64)]

    u_100 = to_ue(1.0)
    u_1625 = to_ue(1.625)
    u_150 = to_ue(1.5)
    assert raw_c[0] == u_1625, f"sub-block 0 must pick d=1.625 (code {u_1625}), got code {raw_c[0]}"
    mse0 = sum((a - b) ** 2 for a, b in zip(row[0:16], dec_c[0:16]))
    assert mse0 == pytest.approx(0.484375, abs=1e-5), f"expected MSE 0.484375 (< static-6 15.0), got {mse0}"

    assert raw_c[1] == u_150, f"sub-block 1 (4/6 exact) must pick d_4=1.5 (code {u_150}), got code {raw_c[1]}"
    mse1 = sum((a - b) ** 2 for a, b in zip(row[16:32], dec_c[16:32]))
    assert mse1 == 0.0, f"sub-block 1 must decode with 0.0 MSE under 4/6, got {mse1}"

    assert raw_c[2] == u_100, f"sub-block 2 (tie between d=1.0 and d=1.5) must keep static-6 (code {u_100}), got {raw_c[2]}"
    mse2 = sum((a - b) ** 2 for a, b in zip(row[32:48], dec_c[32:48]))
    assert mse2 == 0.0

    # Per-block invariant + >22% total MSE reduction + -t 1 == -t 4 + imatrix invariance on BF16 matrix.
    rng = random.Random(20261126)
    cols, rows = 256, 128
    n_el = cols * rows
    raw_f32 = struct.pack(f"<{n_el}f", *[rng.gauss(0.0, 0.03) for _ in range(n_el)])
    bf16_raw = bytearray()
    bf16_vals = []
    for (u,) in struct.iter_unpack("<I", raw_f32):
        u += 0x7FFF + ((u >> 16) & 1)
        h16 = (u >> 16) & 0xFFFF
        bf16_raw += struct.pack("<H", h16)
        bf16_vals.append(struct.unpack("<f", struct.pack("<I", h16 << 16))[0])
    ones256 = struct.pack("<256f", *([1.0] * 256))
    src_bf = _write(
        tmp_path / "bf.gguf",
        [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)],
        [("output_norm.weight", [cols], T_F32, ones256),
         ("blk.0.ffn_gate.weight", [cols, rows], T_BF16, bytes(bf16_raw))],
    )
    imx = _write_gguf_imatrix(tmp_path / "bf-imx.gguf", {
        "blk.0.ffn_gate.weight": (cols, 1, [0.1 + 20.0 * ((c * 17) % 13) for c in range(cols)]),
    })
    out_t1 = tmp_path / "bf-t1.gguf"
    out_t4 = tmp_path / "bf-t4.gguf"
    out_imx = tmp_path / "bf-imx-out.gguf"
    assert _run(runner_bin, "-m", src_bf, "--quantize", out_t1, "--quant", "nvfp4", "-t", "1").returncode == 0
    assert _run(runner_bin, "-m", src_bf, "--quantize", out_t4, "--quant", "nvfp4", "-t", "4").returncode == 0
    assert _run(runner_bin, "-m", src_bf, "--quantize", out_imx, "--quant", "nvfp4", "--imatrix", imx, "-t", "1").returncode == 0
    assert out_t1.read_bytes() == out_t4.read_bytes(), "-t 1 and -t 4 must be byte-identical"
    assert out_t1.read_bytes() == out_imx.read_bytes(), "imatrix must not alter NVFP4 bytes (imatrix_weighted_type stays false)"

    _, got_bf = _gguf(out_t1)
    (f_bf,) = struct.unpack("<f", got_bf["blk.0.ffn_gate.scale"][2])
    expected_f = max(abs(v) for v in bf16_vals) / (6.0 * 448.0)
    assert f_bf == pytest.approx(expected_f, rel=1e-6)
    dec_bf = [v * f_bf for v in _decode_nvfp4_llama_interop(got_bf["blk.0.ffn_gate.weight"][2], n_el)]

    tot_scan, tot_s6 = 0.0, 0.0
    for off in range(0, n_el, 16):
        sub = bf16_vals[off:off + 16]
        d6 = ue_tbl[to_ue(max(abs(v) for v in sub) / (6.0 * f_bf))]
        m_s6 = sum((min(((lvl * d6) * f_bf for lvl in E2M1_TABLE), key=lambda q: abs(q - v)) - v) ** 2 for v in sub)
        m_sc = sum((dec_bf[off + j] - sub[j]) ** 2 for j in range(16))
        assert m_sc <= m_s6 + 1e-12, f"block {off // 16}: scan MSE {m_sc} > static-6 MSE {m_s6}"
        tot_scan += m_sc
        tot_s6 += m_s6
    assert tot_scan < 0.78 * tot_s6, f"expected >22% total MSE reduction, got {(tot_scan / tot_s6 - 1.0) * 100:.2f}%"
