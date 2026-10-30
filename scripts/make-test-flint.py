#!/usr/bin/env python3
"""A tiny flint GGUF for the architecture's fixture tests.

    scripts/make-test-flint.py OUT.gguf [--rope-mismatch] [--no-rope-pattern]
                                        [--sinks-off] [--window N]

Four blocks: two leading dense SiLU blocks, then two blocks of four routed
experts (top 2 by sigmoid score, renormalised) with an always-on shared
expert. Every block has per-head QK norms and attention sinks; blocks 1 and 2
use a sliding window of 4 and rope, blocks 0 and 3 are full attention without
rope, carried as the two boolean arrays flint's GGUFs carry. Weights are
seeded pseudo-random F32. `--rope-mismatch` makes the rope pattern disagree
with the sliding pattern on one block, `--no-rope-pattern` drops the rope
array: both are files Runner refuses. `--sinks-off` sets every sink to
-1e4 (no mass on the sink) and `--window N` changes the window, so a test can
show each takes part.

The arithmetic is checked elsewhere, against transformers' own flint in
float32 on a converted random model
(docs/compat-reports/flint-2026-10-30/README.md); these fixtures pin the
binding: what loads, what is refused, and that the loaded model is finite
and deterministic.
"""
import sys

from gguf_fixture import (s, ku, kf, ks, kb, kas, kaf, kai, kab, pack,
                          write_gguf)

argv = sys.argv[1:]
WINDOW = 4
if "--window" in argv:
    k = argv.index("--window")
    WINDOW = int(argv[k + 1])
    del argv[k:k + 2]
SINKS_OFF = "--sinks-off" in argv
args = [a for a in argv if not a.startswith("--")]
if len(args) != 1:
    sys.exit(f"usage: {sys.argv[0]} OUT.gguf [--rope-mismatch] [--no-rope-pattern]")
OUT = args[0]
MISMATCH = "--rope-mismatch" in sys.argv
NO_ROPE = "--no-rope-pattern" in sys.argv

E, HEADS, KV, HD, LAYERS = 32, 4, 2, 8, 4
FF, FF_EXP, N_EXP, USED, FF_SH = 48, 16, 4, 2, 32
DENSE = 2
SWA = [False, True, True, False]
ROPE = list(SWA)
if MISMATCH:
    ROPE[3] = True
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256

_seed = 0xF117
def rnd(scale=0.08):
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7fffffff
    return (_seed / 0x7fffffff - .5) * scale


def flist(n, scale=0.08):
    return [rnd(scale) for _ in range(n)]


p = "flint"
kvs = [
    ks("general.architecture", p),
    ku(f"{p}.block_count", LAYERS), ku(f"{p}.context_length", 512),
    ku(f"{p}.embedding_length", E), ku(f"{p}.feed_forward_length", FF),
    ku(f"{p}.attention.head_count", HEADS), ku(f"{p}.attention.head_count_kv", KV),
    ku(f"{p}.attention.key_length", HD), ku(f"{p}.attention.value_length", HD),
    kf(f"{p}.attention.layer_norm_rms_epsilon", 1e-6),
    kf(f"{p}.rope.freq_base", 10000.0),
    ku(f"{p}.expert_count", N_EXP), ku(f"{p}.expert_used_count", USED),
    ku(f"{p}.expert_gating_func", 2),
    ku(f"{p}.expert_feed_forward_length", FF_EXP),
    ku(f"{p}.expert_shared_feed_forward_length", FF_SH),
    ku(f"{p}.expert_shared_count", FF_SH // FF_EXP),
    ku(f"{p}.leading_dense_block_count", DENSE),
    kb(f"{p}.expert_weights_norm", True), kf(f"{p}.expert_weights_scale", 1.0),
    ku(f"{p}.attention.sliding_window", WINDOW),
    kab(f"{p}.attention.sliding_window_pattern", SWA),
    ks("tokenizer.ggml.model", "llama"), kas("tokenizer.ggml.tokens", VOCAB),
    kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
    kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
    ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
]
if not NO_ROPE:
    kvs.append(kab(f"{p}.attention.rope_pattern", ROPE))

V = len(VOCAB)
t = [("token_embd.weight", [E, V], pack(flist(E * V))),
     ("output_norm.weight", [E], pack([1.0] * E)),
     ("output.weight", [E, V], pack(flist(E * V)))]
for i in range(LAYERS):
    t += [
        (f"blk.{i}.attn_norm.weight", [E], pack([1.0] * E)),
        (f"blk.{i}.attn_q.weight", [E, HEADS * HD], pack(flist(E * HEADS * HD, 0.5))),
        (f"blk.{i}.attn_k.weight", [E, KV * HD], pack(flist(E * KV * HD, 0.5))),
        (f"blk.{i}.attn_v.weight", [E, KV * HD], pack(flist(E * KV * HD, 0.5))),
        (f"blk.{i}.attn_output.weight", [HEADS * HD, E], pack(flist(HEADS * HD * E, 0.5))),
        (f"blk.{i}.attn_q_norm.weight", [HD], pack([1.0 + rnd(0.6) for _ in range(HD)])),
        (f"blk.{i}.attn_k_norm.weight", [HD], pack([1.0 + rnd(0.6) for _ in range(HD)])),
        (f"blk.{i}.attn_sinks.weight", [HEADS],
         pack([-1e4] * HEADS if SINKS_OFF else flist(HEADS, 4.0))),
        (f"blk.{i}.ffn_norm.weight", [E], pack([1.0] * E)),
    ]
    if i < DENSE:
        t += [
            (f"blk.{i}.ffn_gate.weight", [E, FF], pack(flist(E * FF, 0.5))),
            (f"blk.{i}.ffn_up.weight", [E, FF], pack(flist(E * FF, 0.5))),
            (f"blk.{i}.ffn_down.weight", [FF, E], pack(flist(FF * E, 0.5))),
        ]
    else:
        t += [
            (f"blk.{i}.ffn_gate_inp.weight", [E, N_EXP], pack(flist(E * N_EXP, 2.0))),
            (f"blk.{i}.ffn_gate_exps.weight", [E, FF_EXP, N_EXP], pack(flist(E * FF_EXP * N_EXP, 0.5))),
            (f"blk.{i}.ffn_up_exps.weight", [E, FF_EXP, N_EXP], pack(flist(E * FF_EXP * N_EXP, 0.5))),
            (f"blk.{i}.ffn_down_exps.weight", [FF_EXP, E, N_EXP], pack(flist(FF_EXP * E * N_EXP, 0.5))),
            (f"blk.{i}.ffn_gate_shexp.weight", [E, FF_SH], pack(flist(E * FF_SH, 0.5))),
            (f"blk.{i}.ffn_up_shexp.weight", [E, FF_SH], pack(flist(E * FF_SH, 0.5))),
            (f"blk.{i}.ffn_down_shexp.weight", [FF_SH, E], pack(flist(FF_SH * E, 0.5))),
        ]
write_gguf(OUT, t, kvs)
