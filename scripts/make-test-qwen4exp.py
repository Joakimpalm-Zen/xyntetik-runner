#!/usr/bin/env python3
"""A tiny qwen4exp (Qwen3.8-Flash-Next shaped) GGUF for the CPU forward gate.

Every tensor and key llama.cpp b11433's qwen4exp loader requires, at toy
dims and F32, so the same file loads in both engines and the greedy-agreement
anchor runs without the 84 GB release file. Gated DeltaNet blocks every
block but each full_attention_interval-th, routed experts with a gated shared
expert on every block, hyper-connections (HC streams), a PLE n-gram block at
block 1 with a hashed table sized to its head ranges, and the indexer tensors
on the attention blocks with compress_ratios 0 (dense attention in both
engines). Environment: QWEN4EXP_TEST_LAYERS (default 4), QWEN4EXP_TEST_SEED.
"""
import os
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-qwen4exp.gguf"
E, HEADS, KV, HD = 32, 4, 2, 8            # HD must be a multiple of 4 for the rope sections below
LAYERS = int(os.environ.get("QWEN4EXP_TEST_LAYERS", "4"))
STATE, GROUPS, VHEADS, CONV = 8, 2, 4, 4
HC, HC_LR = 4, 16
N_EXP, N_USED, FF_EXP, FF_SHEXP = 8, 2, 16, 16
IDX_HEADS, IDX_DIM, IDX_TOPK = 2, 8, 2048
PLE_LAYER, PLE_NGRAM, PLE_PER, PLE_K, PLE_DIM = 1, 3, 2, 4, 8
PLE_HEADS = (PLE_NGRAM - 1) * PLE_PER
PLE_VOCAB = [37, 41, 43, 47]              # one small prime per head
PLE_OFF = [sum(PLE_VOCAB[:h]) for h in range(PLE_HEADS)]
PLE_ROWS = PLE_OFF[-1] + PLE_VOCAB[-1]
PLE_MULT = [23703573157769, 20109073645365, 8052911324071]   # the release file's
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b
def ku(k, v): return s(k) + struct.pack("<II", U32, v)
def kf(k, v): return s(k) + struct.pack("<If", F32, v)
def ks(k, v): return s(k) + struct.pack("<I", STR) + s(v)
def kb(k, v): return s(k) + struct.pack("<IB", BOOL, bool(v))
def kas(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, STR, len(xs)) + b"".join(s(x) for x in xs)
def kaf(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, F32, len(xs)) + struct.pack(f"<{len(xs)}f", *xs)
def kai(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, I32, len(xs)) + struct.pack(f"<{len(xs)}i", *xs)
def kau(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, U32, len(xs)) + struct.pack(f"<{len(xs)}I", *xs)
def kau64(k, xs):
    return s(k) + struct.pack("<IIQ", ARR, U64, len(xs)) + struct.pack(f"<{len(xs)}Q", *xs)


seed = int(os.environ.get("QWEN4EXP_TEST_SEED", "0x4e"), 0)
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return (seed / 0x7fffffff - .5) * .08
def data(n): return struct.pack(f"<{n}f", *(rnd() for _ in range(n)))
def ones(n): return struct.pack(f"<{n}f", *([1.] * n))
def vals(xs): return struct.pack(f"<{len(xs)}f", *xs)
# QWEN4EXP_TEST_ZERO=ple,hc,moe,attn,lin zeroes a component's output so a
# two-engine comparison can name the component that disagrees: the PLE table
# (its block adds nothing), the hc mixers (down/up/inject zero: the mix is
# mean/2, the combine weight 1), the expert and shared-expert down
# projections, the attention output projection, the DeltaNet out projection.
ZERO = set(filter(None, os.environ.get("QWEN4EXP_TEST_ZERO", "").split(",")))
ZERO_NAMES = {
    "ple":  ("per_layer_token_embd.weight",),
    "hc":   ("hc_attn_down", "hc_attn_up", "hc_attn_inject", "hc_ffn_down",
             "hc_ffn_up", "hc_ffn_inject", "output_hc_down", "output_hc_up"),
    "moe":  ("ffn_down_exps", "ffn_down_shexp"),
    "attn": ("attn_output.weight",),
    "lin":  ("ssm_out.weight",),
}
def add(ts, name, dims, payload=None):
    n = 1
    for d in dims: n *= d
    if payload is None:
        payload = data(n)
        for z in ZERO:
            if any(k in name for k in ZERO_NAMES[z]):
                payload = struct.pack(f"<{n}f", *([0.] * n))
    ts.append((name, dims, payload))


HCD = HC * E
t = []
add(t, "token_embd.weight", [E, len(VOCAB)])
add(t, "output.weight", [E, len(VOCAB)])
# the head mixer replaces output_norm; gammas are stored (1 + w) folded
add(t, "output_hc_norm.weight", [HCD], ones(HCD))
add(t, "output_hc_down.weight", [HCD, HC_LR])
add(t, "output_hc_up.weight", [HC_LR, HCD])
add(t, "per_layer_token_embd.weight", [PLE_DIM, PLE_ROWS])
for i in range(LAYERS):
    for part in ("attn", "ffn"):
        add(t, f"blk.{i}.hc_{part}_norm.weight", [HCD], ones(HCD))
        add(t, f"blk.{i}.hc_{part}_down.weight", [HCD, HC_LR])
        add(t, f"blk.{i}.hc_{part}_up.weight", [HC_LR, HCD])
        add(t, f"blk.{i}.hc_{part}_inject.weight", [HCD, HC])
    if (i + 1) % 4:
        keydim, valuedim = STATE * GROUPS, STATE * VHEADS
        add(t, f"blk.{i}.attn_qkv.weight", [E, keydim * 2 + valuedim])
        add(t, f"blk.{i}.attn_gate.weight", [E, valuedim])
        add(t, f"blk.{i}.ssm_conv1d.weight", [CONV, keydim * 2 + valuedim])
        add(t, f"blk.{i}.ssm_dt.bias", [VHEADS], vals([0.] * VHEADS))
        add(t, f"blk.{i}.ssm_a", [VHEADS], vals([-1.] * VHEADS))
        add(t, f"blk.{i}.ssm_beta.weight", [E, VHEADS])
        add(t, f"blk.{i}.ssm_alpha.weight", [E, VHEADS])
        add(t, f"blk.{i}.ssm_norm.weight", [STATE], ones(STATE))
        add(t, f"blk.{i}.ssm_out.weight", [valuedim, E])
    else:
        kvdim = HD * KV
        add(t, f"blk.{i}.attn_q.weight", [E, HD * HEADS * 2])
        add(t, f"blk.{i}.attn_k.weight", [E, kvdim])
        add(t, f"blk.{i}.attn_v.weight", [E, kvdim])
        add(t, f"blk.{i}.attn_output.weight", [HD * HEADS, E])
        add(t, f"blk.{i}.attn_q_norm.weight", [HD], ones(HD))
        add(t, f"blk.{i}.attn_k_norm.weight", [HD], ones(HD))
        add(t, f"blk.{i}.indexer.q_proj.weight", [E, IDX_HEADS * IDX_DIM])
        add(t, f"blk.{i}.indexer.k_proj.weight", [E, IDX_DIM])
        add(t, f"blk.{i}.indexer.q_norm.weight", [IDX_DIM], ones(IDX_DIM))
        add(t, f"blk.{i}.indexer.k_norm.weight", [IDX_DIM], ones(IDX_DIM))
    if i == PLE_LAYER:
        add(t, f"blk.{i}.ple_key.weight", [PLE_HEADS * PLE_DIM, HCD])
        add(t, f"blk.{i}.ple_value.weight", [PLE_HEADS * PLE_DIM, E])
        add(t, f"blk.{i}.ple_norm_key.weight", [HCD], ones(HCD))
        add(t, f"blk.{i}.ple_norm_query.weight", [HCD], ones(HCD))
        add(t, f"blk.{i}.ple_norm_conv.weight", [HCD], ones(HCD))
        add(t, f"blk.{i}.ple_conv1d.weight", [PLE_K, HCD])
    add(t, f"blk.{i}.ffn_gate_inp.weight", [E, N_EXP])
    add(t, f"blk.{i}.ffn_gate_exps.weight", [E, FF_EXP, N_EXP])
    add(t, f"blk.{i}.ffn_up_exps.weight", [E, FF_EXP, N_EXP])
    add(t, f"blk.{i}.ffn_down_exps.weight", [FF_EXP, E, N_EXP])
    add(t, f"blk.{i}.ffn_gate_inp_shexp.weight", [E])
    add(t, f"blk.{i}.ffn_gate_shexp.weight", [E, FF_SHEXP])
    add(t, f"blk.{i}.ffn_up_shexp.weight", [E, FF_SHEXP])
    add(t, f"blk.{i}.ffn_down_shexp.weight", [FF_SHEXP, E])

A = "qwen4exp"
kvs = [
    ks("general.architecture", A), ku(f"{A}.block_count", LAYERS),
    ku(f"{A}.context_length", 256), ku(f"{A}.embedding_length", E),
    ku(f"{A}.attention.head_count", HEADS), ku(f"{A}.attention.head_count_kv", KV),
    ku(f"{A}.attention.key_length", HD), ku(f"{A}.attention.value_length", HD),
    ku(f"{A}.rope.dimension_count", HD), # [t, h, w, extra] must give every rotary pair a live section: llama.cpp's
    # interleaved M-RoPE sends pair p to section p % 3 and a pair whose section
    # is empty falls to the 4th position, which is 0 for text (unrotated).
    # [2, 2, 0, 0] left pair 2 unrotated there; the release file's
    # [11, 11, 10, 0] has no such pair.
    kau(f"{A}.rope.dimension_sections", [2, 1, 1, 0]),
    kf(f"{A}.rope.freq_base", float(os.environ.get("QWEN4EXP_TEST_FREQ_BASE", "10000"))), kf(f"{A}.attention.layer_norm_rms_epsilon", 1e-6),
    ku(f"{A}.expert_count", N_EXP), ku(f"{A}.expert_used_count", N_USED),
    ku(f"{A}.expert_feed_forward_length", FF_EXP),
    ku(f"{A}.expert_shared_feed_forward_length", FF_SHEXP),
    ku(f"{A}.ssm.conv_kernel", CONV), ku(f"{A}.ssm.state_size", STATE),
    ku(f"{A}.ssm.group_count", GROUPS), ku(f"{A}.ssm.time_step_rank", VHEADS),
    ku(f"{A}.ssm.inner_size", E), ku(f"{A}.full_attention_interval", 4),
    ku(f"{A}.hyper_connection.count", HC), ku(f"{A}.hyper_connection.low_rank", HC_LR),
    ku(f"{A}.attention.indexer.head_count", IDX_HEADS),
    ku(f"{A}.attention.indexer.key_length", IDX_DIM),
    ku(f"{A}.attention.indexer.top_k", IDX_TOPK),
    kau(f"{A}.attention.compress_ratios", [0] * LAYERS),
    kau(f"{A}.ple.layers", [PLE_LAYER]), ku(f"{A}.ple.ngram_size", PLE_NGRAM),
    ku(f"{A}.ple.heads_per_ngram", PLE_PER), ku(f"{A}.ple.conv_kernel", PLE_K),
    ku(f"{A}.ple.eos_token_id", 2), ku(f"{A}.embedding_length_per_layer_input", PLE_DIM),
    kau64(f"{A}.ple.layer_multipliers", PLE_MULT),
    kau64(f"{A}.ple.head_offsets", PLE_OFF), kau64(f"{A}.ple.head_vocab_sizes", PLE_VOCAB),
    ks("tokenizer.ggml.model", "llama"),
    kas("tokenizer.ggml.tokens", VOCAB), kaf("tokenizer.ggml.scores", [0.] * len(VOCAB)),
    kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
    ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
]
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
