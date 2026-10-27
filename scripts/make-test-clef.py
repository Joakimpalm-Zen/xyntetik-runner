#!/usr/bin/env python3
"""Generate a tiny Clef-shaped GGUF: the Ornith-sized qwen35 backbone under
the "clef" key prefix, with a joint schema head laid out as llama.cpp's
converter writes Cloudflare/clef's (dec.blk.N with the routing blocks first,
decision.*, token_types, decision.scales).

    scripts/make-test-clef.py OUTPUT

The vocabulary is the byte fallback alone with no space prefix, so a record's
token ids decode back to its prompt byte for byte; tests/test_clef.py checks
the encoder's rendering that way. CLEF_TEST_DROP names one tensor to leave
out (a malformed-head fixture).
"""
import os
import struct
import sys

from gguf_fixture import s, ku, kf, ks, kb, kas, kaf, kai, kau, ones

OUT = sys.argv[1] if len(sys.argv) > 1 else "test-clef.gguf"
if str(OUT).startswith("-"):
    sys.exit(f"usage: {sys.argv[0]} [OUTPUT]: writes the fixture to OUTPUT (got {str(OUT)!r})")
DROP = os.environ.get("CLEF_TEST_DROP", "")
E, HEADS, KV, FF, LAYERS = 32, 4, 2, 64, 4
STATE, GROUPS, VHEADS, CONV = 8, 2, 4, 4
W, H, HF, ROUTING, JOINT = 16, 4, 32, 2, 2       # the head
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256


seed = 0x35
def rnd(scale=0.08):
    global seed
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return (seed / 0x7fffffff - .5) * scale
def data(n, scale=0.08): return struct.pack(f"<{n}f", *(rnd(scale) for _ in range(n)))
def near_one(n): return struct.pack(f"<{n}f", *(1. + rnd(0.2) for _ in range(n)))
def vals(xs): return struct.pack(f"<{len(xs)}f", *xs)


t = []
def add(name, dims, payload=None, scale=0.08):
    if name == DROP:
        return
    n = 1
    for d in dims: n *= d
    t.append((name, dims, payload if payload is not None else data(n, scale)))


# the backbone: make-test-ornith.py's
add("token_embd.weight", [E, len(VOCAB)])
add("output_norm.weight", [E], ones(E))
for i in range(LAYERS):
    add(f"blk.{i}.attn_norm.weight", [E], ones(E))
    add(f"blk.{i}.post_attention_norm.weight", [E], ones(E))
    if (i + 1) % 4:
        keydim, valuedim = STATE * GROUPS, STATE * VHEADS
        add(f"blk.{i}.attn_qkv.weight", [E, keydim * 2 + valuedim])
        add(f"blk.{i}.attn_gate.weight", [E, valuedim])
        add(f"blk.{i}.ssm_conv1d.weight", [CONV, keydim * 2 + valuedim])
        add(f"blk.{i}.ssm_dt.bias", [VHEADS], vals([0.] * VHEADS))
        add(f"blk.{i}.ssm_a", [VHEADS], vals([-1.] * VHEADS))
        add(f"blk.{i}.ssm_beta.weight", [E, VHEADS])
        add(f"blk.{i}.ssm_alpha.weight", [E, VHEADS])
        add(f"blk.{i}.ssm_norm.weight", [STATE], ones(STATE))
        add(f"blk.{i}.ssm_out.weight", [valuedim, E])
    else:
        hd, kvdim = E // HEADS, (E // HEADS) * KV
        add(f"blk.{i}.attn_q.weight", [E, E * 2])
        add(f"blk.{i}.attn_k.weight", [E, kvdim])
        add(f"blk.{i}.attn_v.weight", [E, kvdim])
        add(f"blk.{i}.attn_output.weight", [E, E])
        add(f"blk.{i}.attn_q_norm.weight", [hd], ones(hd))
        add(f"blk.{i}.attn_k_norm.weight", [hd], ones(hd))
    add(f"blk.{i}.ffn_gate.weight", [E, FF])
    add(f"blk.{i}.ffn_up.weight", [E, FF])
    add(f"blk.{i}.ffn_down.weight", [FF, E])

# the joint schema head; larger weights than the backbone's so the scores
# spread and a wrong wire shows up in them
def norm(base, n):
    add(f"{base}.weight", [n], near_one(n))
    add(f"{base}.bias", [n], data(n, 0.2))
def attn(prefix, i):
    for part in "qkvo":
        add(f"dec.blk.{i}.{prefix}_{part}.weight", [W, W], scale=0.6)
        add(f"dec.blk.{i}.{prefix}_{part}.bias", [W], data(W, 0.2))
for i in range(ROUTING + JOINT):
    if i < ROUTING:
        norm(f"dec.blk.{i}.cross_attn_norm_kv", W)
    else:
        norm(f"dec.blk.{i}.attn_norm", W)
        attn("attn", i)
    norm(f"dec.blk.{i}.cross_attn_norm", W)
    attn("cross_attn", i)
    norm(f"dec.blk.{i}.ffn_norm", W)
    add(f"dec.blk.{i}.ffn_up.weight", [W, HF], scale=0.6)
    add(f"dec.blk.{i}.ffn_up.bias", [HF], data(HF, 0.2))
    add(f"dec.blk.{i}.ffn_down.weight", [HF, W], scale=0.6)
    add(f"dec.blk.{i}.ffn_down.bias", [W], data(W, 0.2))
norm("decision.hidden_norm", E)
for n_ in ("option_summary_norm", "field_norm", "option_norm"):
    norm(f"decision.{n_}", W)
for p in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
    add(f"decision.proj_{p}.weight", [E, W], scale=0.8)
add("token_types.weight", [W, 3], scale=0.8)
add("decision.scorer.weight", [4 * W, W], scale=0.6)
add("decision.scorer.bias", [W], data(W, 0.2))
add("decision.scorer_out.weight", [W, 1], scale=0.8)
add("decision.scorer_out.bias", [1], vals([0.1]))
# exp(min(x, ln 100)) twice and a sigmoid, as the converter stores them
add("decision.scales", [3], vals([1.5, 3.0, 0.6]))

A = "clef"
kvs = [
    ks("general.architecture", A), ku(f"{A}.block_count", LAYERS),
    ku(f"{A}.context_length", 8192), ku(f"{A}.embedding_length", E),
    ku(f"{A}.feed_forward_length", FF), ku(f"{A}.attention.head_count", HEADS),
    ku(f"{A}.attention.head_count_kv", KV), ku(f"{A}.attention.key_length", E // HEADS),
    ku(f"{A}.rope.dimension_count", E // HEADS), kf(f"{A}.attention.layer_norm_rms_epsilon", 1e-5),
    kf(f"{A}.rope.freq_base", 10000.), ku(f"{A}.full_attention_interval", 4),
    kau(f"{A}.rope.dimension_sections", [2, 2, 0, 0]),
    ku(f"{A}.ssm.conv_kernel", CONV), ku(f"{A}.ssm.inner_size", E),
    ku(f"{A}.ssm.state_size", STATE), ku(f"{A}.ssm.time_step_rank", VHEADS),
    ku(f"{A}.ssm.group_count", GROUPS),
    ks(f"{A}.decision.type", "clef"), ku(f"{A}.decision.routing_block_count", ROUTING),
    ku(f"{A}.decision.block_count", JOINT), ku(f"{A}.decision.head_count", H),
    kf(f"{A}.attention.layer_norm_epsilon", 1e-5),
    ks("tokenizer.ggml.model", "llama"),
    kas("tokenizer.ggml.tokens", VOCAB), kaf("tokenizer.ggml.scores", [0.] * len(VOCAB)),
    kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
    ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", False),
    kb("tokenizer.ggml.add_space_prefix", False),
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
