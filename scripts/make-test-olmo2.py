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

    make-test-olmo2.py OUT.gguf [--llama] [--plain] [--gqa] [--wide] [--head-dim 16|32] [--olmo-specials | --tulu3-specials]
                       [--instruct-template | --tulu3-template]
                       [--no-pattern-array] [--attn-factor F] [--bad KIND]

  --llama             the same sizes and weights as a plain llama file
                      (pre-norms in place of the post-norms, no QK norm, no
                      window, no YaRN): the reference script's calibration
                      on an admitted family
  --plain             OLMo 2's layout: no sliding window, no RoPE scaling
  --gqa               half the KV heads (2 of 4: kv_dim 32 != q_dim 64 !=
                      head_dim 16; with --head-dim 32, 1 of 2)
  --wide              256-wide rows (n_embd 256, FFN 512): every 2-D matrix has
                      inner dimension divisible by 256 so K-quants quantize
                      without width fallback
  --olmo-specials     OLMo 2's <|endoftext|> as one control token, the file's BOS and EOS (add_bos_token
                      stays true: the case a template that writes its own BOS must not double)
  --instruct-template write OLMo 2 Instruct's chat template (revision 48d788ec) into the file
  --tulu3-specials    Tulu 3's <|begin_of_text|> (BOS) and <|end_of_text|> (EOS), two control tokens
                      (add_bos_token true, as Tulu 3's tokenizer adds its BOS)
  --tulu3-template    write Tulu 3's chat template (allenai/Llama-3.1-Tulu-3-8B revision 66694379, the same
                      bytes as OLMo-2-0325-32B-Instruct's) into the file
  --head-dim 32       2 heads of 32 in place of 4 of 16 (n_embd stays 64):
                      a KV row then holds whole 32-value blocks, so the q8
                      and k8v4 caches apply (at 16 they keep f16)
  --no-pattern-array  leave out attention.sliding_window_pattern (the
                      default 3-in-4 pattern then applies)
  --attn-factor F     write rope.scaling.attn_factor as F instead of the
                      factor's default 1 + 0.1 ln(factor)
  --bad KIND          a file the loader must refuse: pre-norm (blk.0 carries
                      attn_norm), no-qk-norm (blk.0 has no attn_q_norm),
                      narrow-qk-norm (blk.0's QK norms one head wide),
                      no-post-norm (blk.0 has no post_ffw_norm),
                      pattern-period (the pattern as a u32 period),
                      pattern-short (a bool array one layer short) or
                      experts (declares expert_count 4: olmo2 is dense)

The options come in any order around the one output path.
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

BAD_KINDS = ("pre-norm", "no-qk-norm", "narrow-qk-norm", "no-post-norm", "pattern-period", "pattern-short",
             "experts")
args = sys.argv[1:]
LLAMA = PLAIN = GQA = WIDE = NO_PATTERN = OLMO_SPECIALS = TULU3_SPECIALS = False
INSTRUCT_TEMPLATE = TULU3_TEMPLATE = False
ATTN_FACTOR = 1.0 + 0.1 * math.log(YARN_FACTOR)
BAD = None
OUT = None
k = 0
while k < len(args):
    a = args[k]
    if a == "--llama":
        LLAMA = True
    elif a == "--plain":
        PLAIN = True
    elif a == "--gqa":
        GQA = True
    elif a == "--wide":
        WIDE = True
    elif a == "--head-dim" and k + 1 < len(args) and args[k + 1] in ("16", "32"):
        k += 1
        HEAD_DIM = int(args[k])
    elif a == "--no-pattern-array":
        NO_PATTERN = True
    elif a == "--olmo-specials":
        OLMO_SPECIALS = True
    elif a == "--instruct-template":
        INSTRUCT_TEMPLATE = True
    elif a == "--tulu3-specials":
        TULU3_SPECIALS = True
    elif a == "--tulu3-template":
        TULU3_TEMPLATE = True
    elif a == "--attn-factor" and k + 1 < len(args):
        k += 1
        ATTN_FACTOR = float(args[k])
    elif a == "--bad" and k + 1 < len(args) and args[k + 1] in BAD_KINDS:
        k += 1
        BAD = args[k]
    elif a.startswith("-"):
        sys.exit(f"make-test-olmo2: unknown option {a!r}")
    elif OUT is None:
        OUT = a
    else:
        sys.exit(f"make-test-olmo2: a second output path {a!r}")
    k += 1
if OUT is None:
    sys.exit(__doc__)
if WIDE:
    N_EMBD, N_FF = 256, 512
N_HEAD = N_KV = N_EMBD // HEAD_DIM
if GQA:
    N_KV = N_HEAD // 2

# allenai/OLMo-2-0425-1B-Instruct's chat template at revision 48d788eca847d4d7548f375ad03d3c9312f6139e (its
# tokenizer_config.json, the same bytes as its publisher GGUF's tokenizer.chat_template)
OLMO2_INSTRUCT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + "
    "message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' "
    "}}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] "
    "+ eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif "
    "%}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}")

# allenai/Llama-3.1-Tulu-3-8B's chat template at revision 666943798adbde0b1aff34626007e26986a3c107 (70B, 405B and
# allenai/OLMo-2-0325-32B-Instruct ship the same bytes)
TULU3_TEMPLATE_TEXT = (
    "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + "
    "'\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif "
    "message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + "
    "eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif "
    "%}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}")

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
if OLMO_SPECIALS:   # OLMo 2's own BOS and EOS, one control token, as its vocabulary holds them
    VOCAB.append("<|endoftext|>")
    TTYPE.append(3)
BOS_ID = EOS_ID = len(VOCAB) - 1 if OLMO_SPECIALS else None
if TULU3_SPECIALS:   # Tulu 3's (Llama 3.1's) BOS and EOS, two control tokens
    VOCAB += ["<|begin_of_text|>", "<|end_of_text|>"]
    TTYPE += [3, 3]
    BOS_ID, EOS_ID = len(VOCAB) - 2, len(VOCAB) - 1
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
        norms = [(f"blk.{i}.attn_q_norm.weight", [q_dim], qn),
                 (f"blk.{i}.attn_k_norm.weight", [kv_dim], kn),
                 (f"blk.{i}.post_attention_norm.weight", [N_EMBD], n_attn),
                 (f"blk.{i}.post_ffw_norm.weight", [N_EMBD], n_ffn)]
        if i == 0 and BAD == "pre-norm":
            norms.append((f"blk.{i}.attn_norm.weight", [N_EMBD], n_attn))
        if i == 0 and BAD == "no-qk-norm":
            norms = [t for t in norms if "attn_q_norm" not in t[0]]
        if i == 0 and BAD == "no-post-norm":
            norms = [t for t in norms if "post_ffw_norm" not in t[0]]
        if i == 0 and BAD == "narrow-qk-norm":
            norms[0] = (f"blk.{i}.attn_q_norm.weight", [HEAD_DIM], qn[:4 * HEAD_DIM])
            norms[1] = (f"blk.{i}.attn_k_norm.weight", [HEAD_DIM], kn[:4 * HEAD_DIM])
        tensors += norms
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
if BAD == "experts":   # the metadata alone: the loader must refuse before it looks for an expert tensor
    kvs += [ku(f"{A}.expert_count", 4), ku(f"{A}.expert_used_count", 2)]
if not LLAMA and not PLAIN:
    kvs += [ks(f"{A}.rope.scaling.type", "yarn"),
            kf(f"{A}.rope.scaling.factor", YARN_FACTOR),
            kf(f"{A}.rope.scaling.attn_factor", ATTN_FACTOR),
            ku(f"{A}.rope.scaling.original_context_length", YARN_ORIG),
            ku(f"{A}.attention.sliding_window", WINDOW)]
    if BAD == "pattern-period":
        kvs.append(ku(f"{A}.attention.sliding_window_pattern", 4))
    elif BAD == "pattern-short":
        kvs.append(kab(f"{A}.attention.sliding_window_pattern", PATTERN[:-1]))
    elif not NO_PATTERN:
        kvs.append(kab(f"{A}.attention.sliding_window_pattern", PATTERN))
kvs += [ks("tokenizer.ggml.model", "llama"),
        kas("tokenizer.ggml.tokens", VOCAB),
        kaf("tokenizer.ggml.scores", [0.0] * len(VOCAB)),
        kai("tokenizer.ggml.token_type", TTYPE),
        ku("tokenizer.ggml.bos_token_id", BOS_ID if BOS_ID is not None else 1),
        ku("tokenizer.ggml.eos_token_id", EOS_ID if EOS_ID is not None else 2),
        kb("tokenizer.ggml.add_bos_token", True)]
if INSTRUCT_TEMPLATE:   # allenai/OLMo-2-0425-1B-Instruct's chat template (revision 48d788ec), byte for byte
    kvs.append(ks("tokenizer.chat_template", OLMO2_INSTRUCT_TEMPLATE))
elif TULU3_TEMPLATE:   # allenai/Llama-3.1-Tulu-3-8B's chat template (revision 66694379), byte for byte
    kvs.append(ks("tokenizer.chat_template", TULU3_TEMPLATE_TEXT))
write_gguf(OUT, tensors, kvs)
