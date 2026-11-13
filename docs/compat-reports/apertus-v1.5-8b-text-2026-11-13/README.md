# Apertus v1.5 8B: a text-only extraction's Q8_0 against swiss-ai's BF16 checkpoint in float32 (2026-11-13)

Apertus v1.5 8B is swiss-ai's multimodal model: `Apertus1p5ForConditionalGeneration`, a text model on the apertus
architecture with a vision and an audio tokenizer, whose input vocabulary (266,752) carries image and audio codebook
tokens and whose output vocabulary is the text's 131,072. swiss-ai publishes no GGUF of it. The row is a third party's
text-only extraction: andreasmartin took the text model out of the checkpoint (andreasmartin/apertus-v1.5-8b-text) and
converted it to this Q8_0. A second conversion of the same extraction, Colby's Q4_K_M, was measured and not admitted.

**What is not flattering, first:**
- **These are not the publisher's files, and they are the text model only.** The publisher's model reads images and
  audio; this file and Runner do not. The extraction's own repository is gated and was not read; what is known of it is
  read from the GGUFs and checked against the publisher's weights (below).
- **The extraction added 200 output rows the publisher's model does not have.** It kept the publisher's first 131,272
  input rows, which include the 200 omni special tokens at ids 131072 to 131271, and gave those ids output rows of its
  own: untrained, near the mean of the real rows. Nothing excludes them: the file carries no suppress list, and a
  request's `logit_bias` is refused. With sampling at temperature 1, about one generated token in 3 million is one of
  these control tokens, which decode to no text. That rate is measured over teacher-forced corpus text, not the model's
  own generations. Greedy decoding never emits one.
- **Colby's Q4_K_M fails the bar:** pooled 0.1658 ± 0.0086 mean KLD, 89.8% margin-qualified top-1. Its header carries
  no importance matrix (an observation; the cause is not measured). The Q8_0 is the only file of the row.
- **The publisher's tokenizer rewrites eleven spelled patterns in input text, and Runner does not.** Its tokenizer.json
  carries a normalizer, which GGUF has no field for, that turns `<think>`, `<thought>` and `<|channel|>thought` with a
  line break into `<|inner_prefix|>`; `</think>`, `</thought>` and `<channel|>` into `<|inner_suffix|>`; removes
  `<answer>` and `</answer>`; turns `<image>` and `<audio>` into the media tokens; and removes whitespace after
  `<|inner_suffix|>`. Runner, like any engine reading this GGUF, tokenizes them as the characters they are; a user
  cannot inject a control token by typing it.
- **With no system message, the publisher's default persona is not written.** It reads "You are Apertus 1.5 Omni, a
  multimodal assistant ... you understand images and audio and respond in text."; Runner does not invent a persona the
  caller did not ask for, and this one claims senses the file does not have.
- **Metal is not measured on this file.** It carries 2509's xIELU activation, which the Metal backend does not have.
- **Training is refused** at load: "training does not cover this model yet: FFN activation not in the backward
  (gpt-oss/apertus)". Serving an adapter is supported.
- **Scope:** two windows over one corpus, CPU score files. Task quality is not measured.
- **Lab instruments.** The mass on the 200 ids was read through two lab-only copies of the files that suppress those
  ids; they are instruments, never a file of the row. The compat run used a lab wrapper (the cells, below).

The kills were declared before each read: the extraction against the publisher's weights and the 200 rows explained,
then the tokenizer, then the template, then the owner's bar (mean KLD at most 0.05 and margin-qualified top-1 at least
97%), and the smallest passing file is the row's.

## The files and the reference

- **Reference:** `swiss-ai/Apertus-v1.5-8B` at `1519de0ae7f3d114879291810a58259966f64c99` (apache-2.0, gated "auto";
  the gate accepted for this account), held whole: the four language-model shards, the vision and audio tokenizer
  files, the index and the configs, every file's sha256 equal to the Hub's LFS record.
- **Reference path:** no released transformers has the apertus1p5 classes (5.22.2, 2026-11-06, has `apertus` only), so
  the reference ran on swiss-ai's own branch, `swiss-ai/transformers` at `3797303dda74844e3d1f8977ff5518bb91f818b4`
  (5.14.0.dev0, the commit the model card installs), in its own environment (torch 2.14.0+cpu, tokenizers 0.22.2):
  `scripts/gold-logits.py --upcast-at-use`, the publisher's `Apertus1p5ForConditionalGeneration` driven with text only,
  float32 arithmetic on the stored bfloat16, cast at use per decoder block (32 blocks).
- **Files:**
  - Q8_0: `andreasmartin/apertus-v1.5-8b-text-Q8_0-GGUF` at `2028c3184c0fa060b4259b33ee25e836da6896e5`,
    `apertus-v1.5-8b-text-q8_0.gguf`, 8,567,121,856 bytes, sha256
    `dea904ad80cd725ec89962abae464477f5d8114d1f4377514f381f7dc2ed202d`.
  - Q4_K_M (measured, not admitted): `Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF` at
    `248ec68a63e219e3f061e4c946fc8a20d63b9975`, `apertus-v1.5-8b-text-q4_k_m.gguf`, 5,059,027,136 bytes, sha256
    `a037df8d87ff6caacee794ee85f55342f2152e0d359b7389033300c3bee5f299`.
  - Each file's sha256 was checked against the Hub's LFS record on download. Both headers name `general.architecture`
    apertus, `general.license` apache-2.0, the pre-tokenizer tekken, a vocabulary of 131,272, context 262,144 and rope
    base 4,000,000, and the base model swiss-ai/Apertus-v1.5-8B.
- **Windows:** two windows of 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, the first starting
  at position 8 (8 to 4493), the second at 15 (15 to 4500), so no position is shared. Token-id keyed: every position
  matched by id, no token spelling mismatches, no dropped reference mass. 195 and 218 positions are margin-qualified.
- **Runner:** CPU score files (`--score --score-top 20 --gpu off`, `RUNNER_SCORE_CHUNKED=1`, the batched prefill route,
  4,520 tokens) from the clean release-recipe build of main 919da757 (gcc 13.3.0).

## The extraction against the publisher

Every tensor of both GGUFs against the publisher's weights at 1519de0a (`trunk-q8_0.json`, `trunk-q4_k_m.json`):

- 324 tensors in each; none missing, none extra; every shape equal. `token_embd` is the publisher's input rows 0 to
  131,271 of 266,752; `output` is the publisher's `lm_head` (131,072 rows) and 200 rows past it.
- Q8_0: 251,561,878 of 251,683,840 blocks are byte-equal to llama.cpp's Q8_0 of the publisher's values. Every other
  block is in `token_embd`, in 1,147 rows of special tokens whose publisher rows are near zero (norm under 0.04; 909
  under 1e-6): their scales are at most 1.32e-5, below float16's smallest normal, so the codes shift; the largest
  absolute error is 1.07e-5, against about 0.09 per dimension in an ordinary row.
- The 130 float32 tensors (the norms) are exact. The per-layer xIELU parameters are the publisher's `act_fn`
  parameters exactly (Runner applies the softplus at load). `rope_freqs` equals the llama3 factors from the publisher's
  `rope_parameters` within 3.2e-7.
- Q4_K_M: the same names and shapes; float32 exact; Q4_K tensors 7.1 to 8.4% relative RMS error, Q6_K 1.8%, uniform by
  type. The two files carry the same tokens, token types, merges and template.

## The 200 rows

- Ids 131072 to 131271 are the publisher's omni special block (`<|RESERVED_OMNI_000|>` to `_199|>` with the image,
  speech and media markers among them), control tokens. The publisher's ids past them, 131,072 visual and 4,100 audio
  codebook tokens, are not in the file.
- Their input rows are the publisher's (within Q8_0 error). Their output rows exist only in the extraction: norm 0.67
  each (the mean of the publisher's rows has norm 0.66), cosine 0.98 with that mean, a per-dimension spread of 0.0020
  around it against the real rows' 0.0905. The recipe is not identified: transformers' mean resizing samples with 1e-9
  times the covariance, which would give about 3e-6 per dimension.
- The chat template names two of them, `<|image|>` (131079) and `<|audio|>` (131085), for image and audio parts only.
  Runner spells a control token in user text as characters.
- **The mass on them** (`mass200.json`), measured two ways over the corpus's 4,519 predicted positions:
  - from Runner's score files, the file against a lab-only copy with the 200 ids suppressed: mean 3.2e-7, max 4.9e-4;
    the argmax is the same at every position and the top-20 ids are the same, shifted by the removed mass;
  - from the publisher's float32 hidden states times the 200 rows, at full precision: mean 3.1e-7, median 6.4e-8,
    max 4.85e-4 at position 0 (the first token after BOS), every other position at most 4.5e-6 (mean 2.1e-7).
  - The rate in the row (one generated token in about 3 million at temperature 1) is the reciprocal of that mean.
- **The KLD both ways:** against the publisher's distribution, the file as it is and the suppressed copy read the same
  pooled 0.0036 to five decimals, at most 2.5e-10 apart at any position (3.4e-9 on the Q4_K_M). The number is
  quantisation; none of it is the extraction's rows. The claim reads the file as a user gets it.

## Against the reference

| File | Bytes | Window | Mean KLD | Median | Max | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| **Q8_0** | 8,567,121,856 | 1 (from 8) | 0.0035 | 0.0008 | 0.036 | 97.00% | 100% (195/195) |
| | | 2 (from 15) | 0.0037 | 0.0007 | 0.031 | 98.67% | 100% (218/218) |
| | | **pooled 600** | **0.0036 ± 0.0003** | 0.0008 | 0.036 | 97.83% | **100% (413/413)** |
| Q4_K_M (not admitted) | 5,059,027,136 | 1 (from 8) | 0.1629 | 0.1032 | 1.682 | 75.33% | 87.69% (171/195) |
| | | 2 (from 15) | 0.1686 | 0.1143 | 1.955 | 79.00% | 91.74% (200/218) |
| | | **pooled 600** | **0.1658 ± 0.0086** | 0.1063 | 1.955 | 77.17% | **89.83% (371/413)** |

± is the standard error of the pooled mean over the 600 per-position values. A percentile bootstrap of the pooled mean
(10,000 resamples, seed 20261113) gives 0.0031 to 0.0041 for the Q8_0 and 0.1496 to 0.1831 for the Q4_K_M, every one of
whose resamples is over 0.05. Both treat the 600 positions as independent draws; they come from one text. The Q4_K_M's
perplexity on the corpus is 23.05 against the Q8_0's 21.25.

Per position: `gold-q8_0-w1.json`, `gold-q8_0-w2.json`, `gold-q4_k_m-w1.json`, `gold-q4_k_m-w2.json` (FILES.sha256).

## The tokenizer

- difftok against swiss-ai's tokenizer.json at 1519de0a (sha256 `1f2f6198`): 0 of 721 corpus strings differ on both
  files. The vocabulary pass runs over the publisher's base vocabulary, ids 0 to 131,071, every one of them in the file
  (998 added and 1,437 partial-UTF-8 entries left out): 2 of 128,637 differ, `<think>` and `</think>`, which the
  publisher's normalizer turns into `<|inner_prefix|>` (32) and `<|inner_suffix|>` (33) and Runner reads as their
  characters. The compat check gates that count (`expect_vocab_divergences` 2).
- The normalizer's eleven rules are listed above. Neither corpus contains any of them, so the fidelity windows are not
  affected.
- BOS 1; the turn ends on `<|assistant_end|>` (68). The reference ids are captured in
  `tests/compatibility/tokenizer-references/apertus-v1.5-8b.tokenizer-json.json`.

## The template

- Both GGUFs carry the publisher's chat_template.jinja at 1519de0a byte for byte (16,614 bytes, sha256 `d23af285`).
  It is 2509's with its own default system message, tool results between `<|tool_output_start|>` and
  `<|tool_output_end|>` (2509 writes them as one `[`...`]` list), OpenAI-wrapped declarations and calls accepted, and
  image and audio content parts.
- Runner picks its `apertus15` renderer (R4.38.1), selected by `<|tool_output_start|>` in the template: 2509's framing
  with v1.5's tool-result wrapping.
- `scripts/template-conformance.py --family apertus15 --require-tokens`: 25 cases compared, 25 identical in text and
  tokens; one is not expressible (system-mid-history, which the reference refuses). Accounted deviations, as for 2509:
  BOS is the tokenizer's, and the default system turn is omitted (above).
- An assistant turn in the history that carries its reasoning: in a `reasoning_content` field, both the publisher's
  template and Runner drop it (the template reads no such field), token for token the same; inline as
  `<think>x</think>y` in the content, the text is the same, and the publisher's normalizer then makes the tags control
  tokens where Runner keeps the characters (the tokenizer limitation above).

## Thinking, tools and structured output

Measured on the Q8_0 through the server (a CUDA MIG slice, 1g.24gb, sm_120, temperature 0):

- **Thinking** (R2.15, main 5a2c58fb): with `enable_thinking` true the model deliberates between `<|inner_prefix|>` and
  `<|inner_suffix|>`; the deliberation comes back in `reasoning_content` and the answer in `content`. A stream sends
  the reasoning chunks, then the answer's. A stop string applies to the answer, not the deliberation. The Responses API
  (reasoning with `enable_thinking`) returns a reasoning item and a message; the Messages API with thinking enabled
  returns a thinking block and a text block. A raw completion's text is the decoded tokens, the two tags not spelled.
- **Tools and structured output:** a tool call (`tool_choice` auto and required, thinking on and off), a JSON schema
  turn and a JSON mode turn generate the same token ids as main before R2.15 (compared on the server's receipts), with
  the call delivered as `tool_calls`.

## The cells

On a CUDA MIG slice, 1g.24gb, sm_120, the host's CPUs shared with other work, from the clean release-recipe build of
this branch's first commit, a11f45ee (gcc 13.3.0, runner sha256 `a7ba0fe1dd82d636c7fe8fa0cfd27b3fdf28ac159b9321dc8ed5b88d379fb06e`;
the build reports commit a11f45ee, not dirty), built before the row was added (R6.67).
- **tier-smoke**, `scripts/tier-smoke.py --row apertus-v1.5-8b-text-q8_0 --model <file> --backend cpu|cuda
  --device-class linux-x86_64-cuda`, `--pin` then `--record` (a compare run against the new pin): both cells pass with
  the fact found; the CPU and CUDA replies are byte-identical (sha256 `77abae05`). All 32 layers run on the device.
- **compat** (`scripts/compat_matrix.py --verify-files --load --execute-checks --require-complete`,
  `1.17.0-2026-11-13-linux-x86_64-apertus-v1.5-8b-text.json`): load, tokenizer (corpus 0 of 721; vocabulary 2 of
  128,637, as declared), chat ("What is 2+2?" answered and stopped) and cpu_cuda (9 of 9 prompts identical, none
  tolerated as a near tie, `RUNNER_CUDA_TC=0`, `RUNNER_CUDA_ROUTE=reference`;
  `cpu_cuda/apertus-v1.5-8b-text-q8_0.json`) all pass. The file was held in memory for the run; the compat script
  resolves a model path, which a memory-held file's link does not survive, so a lab wrapper kept the link for that path
  only. The sha256 check read the held bytes; the repository script is unchanged.
- **Metal:** not a cell (above).

## Training and adapters

- `--adapt-info` (recorded by tier-smoke): adapter serving supported; training not supported, "FFN activation not in
  the backward (gpt-oss/apertus)". The xIELU activation has no backward.
