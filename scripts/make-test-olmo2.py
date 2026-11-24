#!/usr/bin/env python3
"""A tiny olmo2 GGUF (OLMo 2 / OLMo 3) for the CPU forward's gates (R4.50.1).

The keys and tensor names are the ones llama.cpp's converter writes for
allenai/Olmo-3-7B-Think (read from that file's own header, 2026-11-24):
`olmo2.*` hyperparameters, `rope.scaling.type yarn` with `factor`,
`attn_factor` and `original_context_length`, `attention.sliding_window` and
`attention.sliding_window_pattern` as one bool a layer (true: sliding), and
per block attn_q / attn_k / attn_v / attn_output, the full-width
attn_q_norm / attn_k_norm, post_attention_norm, post_ffw_norm and
ffn_gate / ffn_up / ffn_down. There is no attn_norm and no ffn_norm: an OLMo 2
block normalises only after attention and after the MLP.

Shape: 4 layers (sliding, sliding, sliding, full, as the publisher's 3-in-4),
n_embd 64, 4 heads of 16, no GQA, FFN 128, the generator's byte vocabulary
(259 tokens), F32 throughout. The window is 8 tokens, so a text of more than
8 tokens is masked on the sliding layers. YaRN is factor 8 over an original
context of 4096 at theta 500000: at head_dim 16 its correction range is
dimensions 1 to 4 of 8, so the ramp mixes interpolated and extrapolated
frequencies rather than taking one of them whole. Every norm weight is
1 +- 0.5 (deterministic), so a norm read at the wrong place or width moves
the logits.

    make-test-olmo2.py OUT.gguf [--llama] [--plain] [--no-pattern-array]
                       [--attn-factor F]

  --llama             the same sizes and weights as a plain llama file
                      (pre-norms in place of the post-norms, no QK norm, no
                      window, no YaRN): the reference script's calibration
                      on an admitted family
  --plain             OLMo 2's layout: no sliding window, no RoPE scaling
  --no-pattern-array  leave out attention.sliding_window_pattern (the
                      default 3-in-4 pattern then applies)
  --attn-factor F     write rope.scaling.attn_factor as F instead of the
                      factor's default 1 + 0.1 ln(factor)
"""
import math
import struct
import sys

from gguf_fixture import kas, kab, kaf, kai, kb, kf, ks, ku, pack, write_gguf

N_EMBD, N_HEAD, N_KV, N_FF, N_LAYER = 64, 4, 4, 128, 4
HEAD_DIM = N_EMBD // N_HEAD
WINDOW = 8
PATTERN = [(i + 1) % 4 != 0 for i in range(N_LAYER)]   # True: sliding
THETA = 500000.0
YARN_FACTOR, YARN_ORIG = 8.0, 4096
CTX_TRAIN = 32768
EPS = 1e-6

args = sys.argv[1:]
if not args or args[0].startswith("-"):
    sys.exit(__doc__)
OUT = args.pop(0)
LLAMA = "--llama" in args
PLAIN = "--plain" in args
NO_PATTERN = "--no-pattern-array" in args
ATTN_FACTOR = 1.0 + 0.1 * math.log(YARN_FACTOR)
if "--attn-factor" in args:
    ATTN_FACTOR = float(args[args.index("--attn-factor") + 1])
known = {"--llama", "--plain", "--no-pattern-array", "--attn-factor"}
for k, a in enumerate(args):
    if a.startswith("-") and a not in known:
        sys.exit(f"make-test-olmo2: unknown option {a!r}")
    if not a.startswith("-") and (k == 0 or args[k - 1] != "--attn-factor"):
        sys.exit(f"make-test-olmo2: unexpected argument {a!r}")

_state = 0x2F0A1B3C


def rnd():
    """The generator's deterministic draw in [-0.04, 0.04) (make-test-model's LCG)."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0x7FFFFFFF
    return (_state / 0x7FFFFFFF - 0.5) * 0.08


def data(n, scale=1.0):
    return pack([rnd() * scale for _ in range(n)])


def norm_w(n):
    return pack([1.0 + rnd() * 12.5 for _ in range(n)])


VOCAB = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
TTYPE = [2, 3, 3] + [6] * 256
q_dim, kv_dim = N_HEAD * HEAD_DIM, N_KV * HEAD_DIM

# Every draw happens in one order whatever the options, so --llama and the
# olmo2 file hold the same projection weights and the same norm weights.
tensors = [("token_embd.weight", [N_EMBD, len(VOCAB)], data(N_EMBD * len(VOCAB), 12.5)),
           ("output_norm.weight", [N_EMBD], norm_w(N_EMBD)),
           ("output.weight", [N_EMBD, len(VOCAB)], data(N_EMBD * len(VOCAB), 4.0))]
for i in range(N_LAYER):
    wq, wk, wv = data(N_EMBD * q_dim, 4.0), data(N_EMBD * kv_dim, 4.0), data(N_EMBD * kv_dim, 4.0)
    wo = data(q_dim * N_EMBD, 4.0)
    qn, kn = norm_w(q_dim), norm_w(kv_dim)
    n_attn, n_ffn = norm_w(N_EMBD), norm_w(N_EMBD)
    wg, wu, wd = data(N_EMBD * N_FF, 4.0), data(N_EMBD * N_FF, 4.0), data(N_FF * N_EMBD, 4.0)
    tensors += [(f"blk.{i}.attn_q.weight", [N_EMBD, q_dim], wq),
                (f"blk.{i}.attn_k.weight", [N_EMBD, kv_dim], wk),
                (f"blk.{i}.attn_v.weight", [N_EMBD, kv_dim], wv),
                (f"blk.{i}.attn_output.weight", [q_dim, N_EMBD], wo)]
    if LLAMA:
        tensors += [(f"blk.{i}.attn_norm.weight", [N_EMBD], n_attn),
                    (f"blk.{i}.ffn_norm.weight", [N_EMBD], n_ffn)]
    else:
        tensors += [(f"blk.{i}.attn_q_norm.weight", [q_dim], qn),
                    (f"blk.{i}.attn_k_norm.weight", [kv_dim], kn),
                    (f"blk.{i}.post_attention_norm.weight", [N_EMBD], n_attn),
                    (f"blk.{i}.post_ffw_norm.weight", [N_EMBD], n_ffn)]
    tensors += [(f"blk.{i}.ffn_gate.weight", [N_EMBD, N_FF], wg),
                (f"blk.{i}.ffn_up.weight", [N_EMBD, N_FF], wu),
                (f"blk.{i}.ffn_down.weight", [N_FF, N_EMBD], wd)]

A = "llama" if LLAMA else "olmo2"
kvs = [ks("general.architecture", A),
       ks("general.name", "olmo2-test" if not LLAMA else "olmo2-test-llama"),
       ku(f"{A}.block_count", N_LAYER),
       ku(f"{A}.context_length", CTX_TRAIN),
       ku(f"{A}.embedding_length", N_EMBD),
       ku(f"{A}.feed_forward_length", N_FF),
       ku(f"{A}.attention.head_count", N_HEAD),
       ku(f"{A}.attention.head_count_kv", N_KV),
       kf(f"{A}.rope.freq_base", THETA),
       kf(f"{A}.attention.layer_norm_rms_epsilon", EPS)]
if not LLAMA and not PLAIN:
    kvs += [ks(f"{A}.rope.scaling.type", "yarn"),
            kf(f"{A}.rope.scaling.factor", YARN_FACTOR),
            kf(f"{A}.rope.scaling.attn_factor", ATTN_FACTOR),
            ku(f"{A}.rope.scaling.original_context_length", YARN_ORIG),
            ku(f"{A}.attention.sliding_window", WINDOW)]
    if not NO_PATTERN:
        kvs.append(kab(f"{A}.attention.sliding_window_pattern", PATTERN))
kvs += [ks("tokenizer.ggml.model", "llama"),
        kas("tokenizer.ggml.tokens", VOCAB),
        kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
        kai("tokenizer.ggml.token_type", TTYPE),
        ku("tokenizer.ggml.bos_token_id", 1),
        ku("tokenizer.ggml.eos_token_id", 2),
        kb("tokenizer.ggml.add_bos_token", True)]
write_gguf(OUT, tensors, kvs)
