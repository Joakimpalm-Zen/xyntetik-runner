"""R4.48.6 / R4.48.15: a tiny but complete inkling2_loop GGUF, for the loader's and the engine's tests.

4 layers, a loop of 3 passes over layers 1 and 2, NVFP4 expert stacks with their per-expert .scale, a byte-level
vocabulary so the tokenizer loads. The dense projections are F32, BF16 or Q8_0 (--dense), so every weight type a real
file carries reaches the engine; F16 too, a type the CUDA route's products do not take (its device test keeps such
projections on the host). No pytest here: the Makefile writes the variants with any Python.

  python tests/inkling2_tiny.py OUT.gguf [--dense f32|bf16|q8_0|f16]
"""

import struct

T_F32, T_F16, T_Q8_0, T_BF16, T_NVFP4 = 0, 1, 8, 30, 40
U32, I32, F32, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9

H, NH, NKV, HD, E, USED, FF, VOCAB, VUSED, K, WIN, DREL, EXT, RANK = 64, 2, 1, 32, 4, 2, 64, 264, 260, 4, 8, 4, 8, 4
N_BLOCK, LOOP_START, LOOP_END, PASSES = 4, 1, 3, 3
FULL = [False, True, False, True]   # layer types: sliding, full, sliding, full


def _byte_tokens():
    """GPT-2's byte-to-unicode map: 256 printable stand-ins, one per byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m = dict(zip(bs, (chr(c) for c in cs)))
    return [m[b] for b in range(256)]


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key, typ, val):
    out = _s(key) + struct.pack("<I", typ)
    if typ == U32:
        return out + struct.pack("<I", val)
    if typ == I32:
        return out + struct.pack("<i", val)
    if typ == F32:
        return out + struct.pack("<f", val)
    if typ == BOOL:
        return out + struct.pack("<B", 1 if val else 0)
    if typ == STR:
        return out + _s(val)
    et, items = val
    out += struct.pack("<IQ", et, len(items))
    for it in items:
        if et == STR:
            out += _s(it)
        elif et == I32:
            out += struct.pack("<i", it)
        elif et == BOOL:
            out += struct.pack("<B", 1 if it else 0)
        else:
            raise ValueError(et)
    return out


def _f32(n, scale, seed):
    import random
    r = random.Random(seed)
    return struct.pack(f"<{n}f", *[(r.random() - 0.5) * 2 * scale for _ in range(n)])


def _nvfp4(n_el, seed):
    import random
    r = random.Random(seed)
    out = bytearray()
    for _ in range(n_el // 64):
        out += bytes([0x30, 0x30, 0x30, 0x30])                 # UE4M3 0.5 for each sub-block
        out += bytes(r.randrange(256) for _ in range(32))      # E2M1 codes
    return bytes(out)


def _bf16(data):
    """float32 bytes to BF16, rounded to nearest even (as a converter writes it)."""
    out = bytearray()
    for (u,) in struct.iter_unpack("<I", data):
        u += 0x7FFF + ((u >> 16) & 1)
        out += struct.pack("<H", (u >> 16) & 0xFFFF)
    return bytes(out)


def _q8_0(data):
    """float32 bytes to Q8_0: 32 values a block, an f16 scale amax / 127 and the rounded int8 codes."""
    xs = [v for (v,) in struct.iter_unpack("<f", data)]
    out = bytearray()
    for b in range(0, len(xs), 32):
        blk = xs[b:b + 32]
        d = max(abs(v) for v in blk) / 127.0
        d16 = struct.unpack("<e", struct.pack("<e", d))[0]
        inv = 1.0 / d16 if d16 else 0.0
        out += struct.pack("<e", d16) + struct.pack("<32b", *[max(-127, min(127, round(v * inv))) for v in blk])
    return bytes(out)


def _tensors(no_scale_layer=None, scale_len=E, dense_nvfp4=False, dense="f32"):
    """(name, ggml dims, type, bytes) for every tensor of the tiny model. dense: the type of the 2-D projections whose
    rows hold a whole number of 32-value blocks ("f32", "bf16", "q8_0" or "f16"); norms, vectors and short rows stay F32, as
    a converter leaves them."""
    t = []
    seed = [0]

    def f(name, dims, scale=0.05, value=None):
        seed[0] += 1
        n = 1
        for d in dims:
            n *= d
        data = struct.pack(f"<{n}f", *([value] * n)) if value is not None else _f32(n, scale, seed[0])
        if dense != "f32" and value is None and len(dims) == 2 and dims[0] % 32 == 0 and dims[1] > 1:
            if dense == "f16":
                xs = [v for (v,) in struct.iter_unpack("<f", data)]
                t.append((name, dims, T_F16, struct.pack(f"<{n}e", *xs)))
            else:
                t.append((name, dims, T_BF16, _bf16(data)) if dense == "bf16" else (name, dims, T_Q8_0, _q8_0(data)))
            return
        t.append((name, dims, T_F32, data))

    f("token_embd.weight", [H, VOCAB], 0.5)
    f("token_embd_norm.weight", [H], value=1.0)
    f("output_norm.weight", [H], value=1.0)
    f("output_res_query.weight", [H], 0.1)
    f("output.weight", [H, VOCAB], 0.5)
    for b in range(N_BLOCK):
        p = f"blk.{b}."
        C = NKV * HD
        if dense_nvfp4 and b == 0:
            seed[0] += 1
            t.append((p + "attn_q.weight", [H, NH * HD], T_NVFP4, _nvfp4(H * NH * HD, seed[0])))
        else:
            f(p + "attn_q.weight", [H, NH * HD])
        f(p + "attn_q_clip.weight", [HD, NH], value=1.0)
        f(p + "attn_k.weight", [H, C])
        f(p + "attn_v.weight", [H, C])
        f(p + "attn_k_conv.weight", [K, C], 0.3)
        f(p + "attn_v_conv.weight", [K, C], 0.3)
        if FULL[b]:
            f(p + "attn_k_conv_proj.weight", [H, NKV * K])
            f(p + "attn_v_conv_proj.weight", [H, NKV * K])
            for x in ("q", "k", "v"):
                f(p + f"attn_{x}_inp_norm.weight", [H], value=1.0)
        else:
            f(p + "attn_norm.weight", [H], value=1.0)
        f(p + "attn_gate.weight", [H, NH * HD])
        f(p + "attn_output.weight", [NH * HD, H])
        f(p + "attn_sink_proj.weight", [H, NH], 0.22)
        f(p + "attn_rel_bias.weight", [DREL, EXT], 0.1)
        f(p + "attn_rel.weight", [H, NH * DREL])
        f(p + "attn_rel_clip.weight", [DREL, NH], value=1.0)
        f(p + "ffn_gate_inp.weight", [H, E])
        f(p + "exp_probs_b.bias", [E], value=0.0)
        f(p + "ffn_norm.weight", [H], value=1.0)
        f(p + "attn_post_conv.weight", [K, H], 0.3)
        f(p + "ffn_post_conv.weight", [K, H], 0.3)
        if b >= 1:
            f(p + "attn_res_query.weight", [H, 3] if FULL[b] else [H], 0.1)
        f(p + "ffn_res_query.weight", [H], 0.1)
        if LOOP_START <= b < LOOP_END:
            for r in ("q", "k", "v", "ffn"):
                f(p + f"loop_{r}_ctrl_down.weight", [2 * H, RANK], 0.15)
                f(p + f"loop_{r}_ctrl_forget.weight", [RANK, H], 0.15)
                f(p + f"loop_{r}_ctrl_gain.weight", [RANK, H], 0.15)
            f(p + "loop_msg_query.weight", [H, 4], 0.2)
        seed[0] += 1
        t.append((p + "ffn_gate_up_exps.weight", [H, 2 * FF, E], T_NVFP4, _nvfp4(H * 2 * FF * E, seed[0])))
        seed[0] += 1
        t.append((p + "ffn_down_exps.weight", [FF, H, E], T_NVFP4, _nvfp4(FF * H * E, seed[0])))
        for which in ("gate_up", "down"):
            if no_scale_layer == b and which == "down":
                continue
            n = scale_len if b == 0 and which == "gate_up" else E
            t.append((p + f"ffn_{which}_exps.scale", [n], T_F32, struct.pack(f"<{n}f", *([0.25] * n))))
    return t


def _kvs(drop=None, used=USED):
    a = "inkling2_loop."
    kv = [("general.architecture", STR, "inkling2_loop"), ("general.alignment", U32, 32),
          (a + "block_count", U32, N_BLOCK), (a + "context_length", U32, 512), (a + "embedding_length", U32, H),
          (a + "feed_forward_length", U32, FF), (a + "attention.head_count", ARR, (I32, [NH] * N_BLOCK)),
          (a + "attention.head_count_kv", ARR, (I32, [NKV] * N_BLOCK)),
          (a + "attention.layer_norm_rms_epsilon", F32, 1e-6), (a + "expert_count", U32, E),
          (a + "expert_used_count", U32, used), (a + "attention.key_length", U32, HD),
          (a + "attention.value_length", U32, HD), (a + "loop.count", U32, PASSES), (a + "loop.start", U32, LOOP_START),
          (a + "loop.end", U32, LOOP_END), (a + "loop.gate_rank", U32, RANK), (a + "vocab_size", U32, VOCAB),
          (a + "vocab_size_unpadded", U32, VUSED), (a + "expert_feed_forward_length", U32, FF),
          (a + "attention.key_length_swa", U32, HD), (a + "attention.value_length_swa", U32, HD),
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
          ("tokenizer.ggml.merges", ARR, (STR, [])), ("tokenizer.ggml.eos_token_id", U32, 256)]
    return [x for x in kv if x[0] != drop]


def _write(path, kvs, tensors, cut=0):
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, typ, v in kvs:
        head += _kv(k, typ, v)
    off, infos, blobs = 0, b"", []
    for name, dims, typ, data in tensors:
        infos += _s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off)
        pad = (-len(data)) % 32
        blobs.append(data + b"\0" * pad)
        off += len(data) + pad
    body = head + infos
    body += b"\0" * ((-len(body)) % 32)
    data = b"".join(blobs)
    path.write_bytes(body + (data[:len(data) - cut] if cut else data))
    return path


if __name__ == "__main__":
    import argparse
    import pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dense", choices=("f32", "bf16", "q8_0", "f16"), default="f32")
    a = ap.parse_args()
    _write(pathlib.Path(a.out), _kvs(), _tensors(dense=a.dense))
