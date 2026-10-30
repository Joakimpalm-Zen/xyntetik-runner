#!/usr/bin/env python3
"""A tiny lfm2 GGUF for the architecture's fixture tests.

    scripts/make-test-lfm2.py OUT.gguf [--moe] [--conv-bias] [--all-attention]
                                       [--l-cache N]

Five blocks typed by the attention.head_count_kv array: blocks 2 and 4 are
GQA attention with per-head QK norms, the other three short convolutions
(in_proj into B, C and x, a causal depthwise kernel of shortconv.l_cache
taps, out_proj). Every block has a gated SiLU FFN; `--moe` writes the
lfm2moe form instead, one leading dense block then four sigmoid-routed
experts, top 2, with a selection bias. The final norm is token_embd_norm
and the output is tied. Weights are seeded pseudo-random F32.
`--conv-bias` adds a convolution bias and `--all-attention` types every
block as attention: both are files Runner refuses. `--l-cache N` changes
the kernel length.

The arithmetic is checked elsewhere, against transformers' float32
reference on the publisher's files
(docs/compat-reports/lfm2.5-reference-2026-10-30/README.md); these
fixtures pin the binding and the convolution state: what loads, what is
refused, and that a restored or rewound state decodes bit for bit as a
cold one.
"""
import sys

from gguf_fixture import (s, ku, kf, ks, kb, kas, kaf, kai, kau, pack,
                          write_gguf)

argv = sys.argv[1:]
LCACHE = 3
if "--l-cache" in argv:
    k = argv.index("--l-cache")
    LCACHE = int(argv[k + 1])
    del argv[k:k + 2]
MOE = "--moe" in argv
CONV_BIAS = "--conv-bias" in argv
ALL_ATTN = "--all-attention" in argv
args = [a for a in argv if not a.startswith("--")]
if len(args) != 1:
    sys.exit(f"usage: {sys.argv[0]} OUT.gguf [--moe] [--conv-bias] "
             "[--all-attention] [--l-cache N]")
OUT = args[0]

E, HEADS, KV, HD, LAYERS = 32, 4, 2, 8, 5
FF, FF_EXP, N_EXP, USED, DENSE = 48, 16, 4, 2, 1
ATTN = [False, False, True, False, True]
if ALL_ATTN:
    ATTN = [True] * LAYERS
VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256

_seed = 0x1F2A
def rnd(scale=0.08):
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7fffffff
    return (_seed / 0x7fffffff - .5) * scale


def flist(n, scale=0.08):
    return [rnd(scale) for _ in range(n)]


p = "lfm2moe" if MOE else "lfm2"
kvs = [
    ks("general.architecture", p),
    ku(f"{p}.block_count", LAYERS), ku(f"{p}.context_length", 2048),
    ku(f"{p}.embedding_length", E), ku(f"{p}.feed_forward_length", FF),
    ku(f"{p}.attention.head_count", HEADS),
    kau(f"{p}.attention.head_count_kv", [KV if a else 0 for a in ATTN]),
    kf(f"{p}.attention.layer_norm_rms_epsilon", 1e-5),
    kf(f"{p}.rope.freq_base", 1000000.0),
    ku(f"{p}.shortconv.l_cache", LCACHE),
    ks("tokenizer.ggml.model", "llama"), kas("tokenizer.ggml.tokens", VOCAB),
    kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
    kai("tokenizer.ggml.token_type", TTYPE), ku("tokenizer.ggml.bos_token_id", 1),
    ku("tokenizer.ggml.eos_token_id", 2), kb("tokenizer.ggml.add_bos_token", True),
]
if MOE:
    kvs += [ku(f"{p}.expert_count", N_EXP), ku(f"{p}.expert_used_count", USED),
            ku(f"{p}.expert_feed_forward_length", FF_EXP),
            ku(f"{p}.leading_dense_block_count", DENSE),
            ku(f"{p}.expert_gating_func", 2)]

V = len(VOCAB)
t = [("token_embd.weight", [E, V], pack(flist(E * V, 0.5))),
     ("token_embd_norm.weight", [E], pack([1.0 + rnd(0.4) for _ in range(E)]))]
for i in range(LAYERS):
    t += [(f"blk.{i}.attn_norm.weight", [E], pack([1.0 + rnd(0.4) for _ in range(E)])),
          (f"blk.{i}.ffn_norm.weight", [E], pack([1.0 + rnd(0.4) for _ in range(E)]))]
    if ATTN[i]:
        t += [
            (f"blk.{i}.attn_q.weight", [E, HEADS * HD], pack(flist(E * HEADS * HD, 0.5))),
            (f"blk.{i}.attn_k.weight", [E, KV * HD], pack(flist(E * KV * HD, 0.5))),
            (f"blk.{i}.attn_v.weight", [E, KV * HD], pack(flist(E * KV * HD, 0.5))),
            (f"blk.{i}.attn_output.weight", [HEADS * HD, E], pack(flist(HEADS * HD * E, 0.5))),
            (f"blk.{i}.attn_q_norm.weight", [HD], pack([1.0 + rnd(0.6) for _ in range(HD)])),
            (f"blk.{i}.attn_k_norm.weight", [HD], pack([1.0 + rnd(0.6) for _ in range(HD)])),
        ]
    else:
        t += [
            (f"blk.{i}.shortconv.in_proj.weight", [E, 3 * E], pack(flist(E * 3 * E, 0.8))),
            (f"blk.{i}.shortconv.conv.weight", [LCACHE, E], pack(flist(LCACHE * E, 1.5))),
            (f"blk.{i}.shortconv.out_proj.weight", [E, E], pack(flist(E * E, 0.8))),
        ]
        if CONV_BIAS:
            t.append((f"blk.{i}.shortconv.conv.bias", [E], pack(flist(E))))
    if MOE and i >= DENSE:
        t += [
            (f"blk.{i}.ffn_gate_inp.weight", [E, N_EXP], pack(flist(E * N_EXP, 2.0))),
            (f"blk.{i}.exp_probs_b.bias", [N_EXP], pack(flist(N_EXP, 1.0))),
            (f"blk.{i}.ffn_gate_exps.weight", [E, FF_EXP, N_EXP], pack(flist(E * FF_EXP * N_EXP, 0.5))),
            (f"blk.{i}.ffn_up_exps.weight", [E, FF_EXP, N_EXP], pack(flist(E * FF_EXP * N_EXP, 0.5))),
            (f"blk.{i}.ffn_down_exps.weight", [FF_EXP, E, N_EXP], pack(flist(FF_EXP * E * N_EXP, 0.5))),
        ]
    else:
        t += [
            (f"blk.{i}.ffn_gate.weight", [E, FF], pack(flist(E * FF, 0.5))),
            (f"blk.{i}.ffn_up.weight", [E, FF], pack(flist(E * FF, 0.5))),
            (f"blk.{i}.ffn_down.weight", [FF, E], pack(flist(FF * E, 0.5))),
        ]
write_gguf(OUT, t, kvs)
