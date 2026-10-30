#!/usr/bin/env python3
"""Generate a tiny Qwen3.5/Ornith-shaped GGUF for the CPU tracer test."""
import struct
import sys
import os

from gguf_fixture import s, ku, kf, ks, kb, kas, kaf, kai, kau, ones

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-ornith.gguf"
# the first argument is the output path: an option there (`--help`) used to
# be taken as a file name prefix and wrote a set of "--help.*" files
if str(OUT).startswith("-"):
    sys.exit(f"usage: {sys.argv[0]} [OUTPUT]: writes test fixtures to OUTPUT (got {str(OUT)!r})")
E, HEADS, KV, FF = 32, 4, 2, 64
LAYERS = int(os.environ.get("ORNITH_TEST_LAYERS", "4"))
LEGACY_DT = os.environ.get("ORNITH_LEGACY_DT") == "1"
# ORNITH_TEST_MTP=1: emit the declared NextN/MTP block (an attention block at
# index LAYERS plus its nextn.* head tensors), so the head can be bound and run
MTP = os.environ.get("ORNITH_TEST_MTP") == "1"
# ORNITH_TEST_MOE=1: the qwen35moe shape (Qwen3.5/3.6-35B-A3B): every block's
# FFN is routed experts plus a sigmoid-gated shared expert, as in the
# publishers' GGUFs (ffn_*_exps, ffn_gate_inp, ffn_*_shexp,
# ffn_gate_inp_shexp), under the arch name qwen35moe
MOE = os.environ.get("ORNITH_TEST_MOE") == "1"
ARCH = "qwen35moe" if MOE else "qwen35"
NEXP, NUSED, FFE, FFS = 8, 3, 16, 24
STATE, GROUPS, VHEADS, CONV = 8, 2, 4, 4
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256


seed = int(os.environ.get("ORNITH_TEST_SEED", "0x35"), 0)
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return (seed / 0x7fffffff - .5) * .08
def data(n): return struct.pack(f"<{n}f", *(rnd() for _ in range(n)))
def vals(xs): return struct.pack(f"<{len(xs)}f", *xs)
def add(ts, name, dims, payload=None):
    n = 1
    for d in dims: n *= d
    ts.append((name, dims, payload if payload is not None else data(n)))

t = []
add(t, "token_embd.weight", [E, len(VOCAB)])
add(t, "output_norm.weight", [E], ones(E))
for i in range(LAYERS + (1 if MTP else 0)):
    add(t, f"blk.{i}.attn_norm.weight", [E], ones(E))
    add(t, f"blk.{i}.post_attention_norm.weight", [E], ones(E))
    if i < LAYERS and (i + 1) % 4:
        keydim, valuedim = STATE * GROUPS, STATE * VHEADS
        add(t, f"blk.{i}.attn_qkv.weight", [E, keydim * 2 + valuedim])
        add(t, f"blk.{i}.attn_gate.weight", [E, valuedim])
        add(t, f"blk.{i}.ssm_conv1d.weight", [CONV, keydim * 2 + valuedim])
        dt_name = "ssm_dt" if LEGACY_DT else "ssm_dt.bias"
        add(t, f"blk.{i}.{dt_name}", [VHEADS], vals([0.] * VHEADS))
        add(t, f"blk.{i}.ssm_a", [VHEADS], vals([-1.] * VHEADS))
        add(t, f"blk.{i}.ssm_beta.weight", [E, VHEADS])
        add(t, f"blk.{i}.ssm_alpha.weight", [E, VHEADS])
        add(t, f"blk.{i}.ssm_norm.weight", [STATE], ones(STATE))
        add(t, f"blk.{i}.ssm_out.weight", [valuedim, E])
    else:
        hd, kvdim = E // HEADS, (E // HEADS) * KV
        add(t, f"blk.{i}.attn_q.weight", [E, E * 2])
        add(t, f"blk.{i}.attn_k.weight", [E, kvdim])
        add(t, f"blk.{i}.attn_v.weight", [E, kvdim])
        add(t, f"blk.{i}.attn_output.weight", [E, E])
        add(t, f"blk.{i}.attn_q_norm.weight", [hd], ones(hd))
        add(t, f"blk.{i}.attn_k_norm.weight", [hd], ones(hd))
    if MOE:
        add(t, f"blk.{i}.ffn_gate_inp.weight", [E, NEXP])
        add(t, f"blk.{i}.ffn_gate_exps.weight", [E, FFE, NEXP])
        add(t, f"blk.{i}.ffn_up_exps.weight", [E, FFE, NEXP])
        add(t, f"blk.{i}.ffn_down_exps.weight", [FFE, E, NEXP])
        add(t, f"blk.{i}.ffn_gate_inp_shexp.weight", [E])
        add(t, f"blk.{i}.ffn_gate_shexp.weight", [E, FFS])
        add(t, f"blk.{i}.ffn_up_shexp.weight", [E, FFS])
        add(t, f"blk.{i}.ffn_down_shexp.weight", [FFS, E])
    else:
        add(t, f"blk.{i}.ffn_gate.weight", [E, FF])
        add(t, f"blk.{i}.ffn_up.weight", [E, FF])
        add(t, f"blk.{i}.ffn_down.weight", [FF, E])
    if i == LAYERS:
        add(t, f"blk.{i}.nextn.eh_proj.weight", [2 * E, E])
        add(t, f"blk.{i}.nextn.enorm.weight", [E], ones(E))
        add(t, f"blk.{i}.nextn.hnorm.weight", [E], ones(E))
        add(t, f"blk.{i}.nextn.shared_head_norm.weight", [E], ones(E))

kvs = [
    # Current Qwen3.5 GGUFs include the auxiliary NextN/MTP predictor in
    # block_count.  It is not part of the autoregressive backbone and has a
    # different tensor layout, so the loader must subtract the declared count.
    ks("general.architecture", ARCH), ku(f"{ARCH}.block_count", LAYERS + 1),
    ku(f"{ARCH}.nextn_predict_layers", 1),
    ku(f"{ARCH}.context_length", 256), ku(f"{ARCH}.embedding_length", E),
    ku(f"{ARCH}.feed_forward_length", FF), ku(f"{ARCH}.attention.head_count", HEADS),
    ku(f"{ARCH}.attention.head_count_kv", KV), ku(f"{ARCH}.attention.key_length", E // HEADS),
    ku(f"{ARCH}.rope.dimension_count", E // HEADS), kf(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5),
    kf(f"{ARCH}.rope.freq_base", 10000.), ku(f"{ARCH}.full_attention_interval", 4),
    kau(f"{ARCH}.rope.dimension_sections", [2, 2, 0, 0]),
    ku(f"{ARCH}.ssm.conv_kernel", CONV), ku(f"{ARCH}.ssm.inner_size", E),
    ku(f"{ARCH}.ssm.state_size", STATE), ku(f"{ARCH}.ssm.time_step_rank", VHEADS),
    ku(f"{ARCH}.ssm.group_count", GROUPS), ks("tokenizer.ggml.model", "llama"),
    kas("tokenizer.ggml.tokens", VOCAB), kaf("tokenizer.ggml.scores", [0.] * len(VOCAB)),
    kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
    ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
]
if MOE:
    kvs += [ku(f"{ARCH}.expert_count", NEXP), ku(f"{ARCH}.expert_used_count", NUSED),
            ku(f"{ARCH}.expert_feed_forward_length", FFE),
            ku(f"{ARCH}.expert_shared_feed_forward_length", FFS)]
meta = b"".join(kvs)
info, off = b"", 0
for name, dims, payload in t:
    info += s(name) + struct.pack("<I", len(dims))
    info += b"".join(struct.pack("<Q", d) for d in dims)
    info += struct.pack("<IQ", 0, off)
    off = (off + len(payload) + 31) & ~31
head = struct.pack("<IIQQ", 0x46554747, 3, len(t), len(kvs)) + meta + info
with open(OUT, "wb") as f:
    f.write(head + b"\0" * ((-len(head)) % 32))
    for _, _, payload in t:
        f.write(payload)
        f.write(b"\0" * ((-len(payload)) % 32))
print(f"wrote {OUT}")
