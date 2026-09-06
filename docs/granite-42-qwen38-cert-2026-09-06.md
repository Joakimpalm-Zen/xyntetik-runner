# Granite 4.2 and Qwen 3.8 admission, 2026-09-06

Environment: runner branch `granite42-qwen38` (this document's commit),
llama.cpp reference **master 73a43d1 (2026-09-06)**, built on the Blackwell
box with `-DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF`. The tokenizer reference
is each model's own Hugging Face tokenizer through the `tokenizers` library.
Raw gate outputs are in `docs/granite-42-qwen38-evidence/`, the compat-matrix
rows in `docs/compat-reports/0.4.10-2026-09-06-blackwell-*.json`.

Files under test, SHA-256 pinned in `tests/compatibility/models.json`:

| file | source | bytes |
|---|---|---|
| `granite-4.2-3b-Q4_K_M.gguf` | `ibm-granite/granite-4.2-3b-GGUF` (official) | 2.0 GB |
| `granite-4.2-8b-Q4_K_M.gguf` | `ibm-granite/granite-4.2-8b-GGUF` (official) | 5.1 GB |
| `granite-4.2-8b-Q8_0.gguf` | same repository, the quant bar's baseline | 8.7 GB |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | `unsloth/Qwen3.8-27B-GGUF` | 16.6 GB |
| `Qwen3.8-27B-Q8_0.gguf` | same repository, the quant bar's baseline | 29 GB |

## Verdicts

**Granite 4.2 (3B and 8B, Q4_K_M): ADMITTED.** The architecture is 4.1's
(`granite`, identical tensor set); what changed is the header and the
template, and both needed runner work: a new pre-tokenizer name and a new
chat template family. After that work the tokenizer differential is
**0/721** on both sizes, the chat template renders **22/22** reference cases
byte- and token-identical, and the cross-engine logit comparison sits in
the same class as certified Granite 4.1. Greedy identity against llama.cpp
is 1/6 on the 8B by the strict count, and the token-divergence gate reads
every one of those divergences as a numerical tie (**0 real of 9**, max
0.187 nats against the 0.25-nat bar). The 8B Q4_K_M passes the quant
fidelity bar against its own Q8_0 (mean KLD **0.0036**, top-1 **99.75%**).
The 3B is the weaker identity row: 0/6 strict on the cert set (the
certified 4.1-3b control reads 3/6 on the same set today) and 5 of 11
divergences past the tie bar on the 16-prompt gate, where the control
reads 4 of 8 with one-sided preferences of the same size; its same-file
logits are cleaner than that sibling's. It is admitted on the 8B's evidence for the shared code and on
its own tokenizer, template, cpu_cuda (9/9) and chat results, with its
identity row stated as measured.

**Qwen 3.8 27B (UD-Q4_K_M): ADMITTED.** The `qwen35` path is the whole
admission, as the plan's narrowed premise predicted: 48 Gated DeltaNet plus
16 full-attention blocks, and one NextN/MTP block that `--mtp` drafts from.
The tokenizer differential was **3/721** on the shelf rule and is **0/721**
after the `qwen35` pre-tokenizer learned the regex's `[\p{L}\p{M}]+` letter
class. Greedy identity vs llama.cpp is **4/6** (the two 256-token runs part
late, at bytes 408 and 398), cross-engine logits mean KLD 0.012 with every
top-1 disagreement inside the tie margin. CPU only on this box: the CUDA
slice was held by another project's job for the whole window, so the
`cpu_cuda` check is recorded as not executed, not as a pass.

## Gate table

| gate | granite-4.2-3b Q4_K_M | granite-4.2-8b Q4_K_M | Qwen3.8-27B UD-Q4_K_M |
|---|---|---|---|
| tokenizer differential (721 strings) | **0/721** (5/721 before the fix) | **0/721** | **0/721** (3/721 before the fix) |
| greedy identity vs 73a43d1 (6 prompts) | 0/6 strict (first partings at bytes 19, 113, 47, 2; see the 3B section) | 1/6 | 4/6 (256-token runs diverge at bytes 408, 398) |
| token divergence vs reference (tie bar 0.25 nats, 16 prompts x 32 tokens) | 5 identical, 6 at a tie, **5 real** (see below) | 7 identical, **9 at a tie, 0 real**, max 0.187 | 13 identical, **3 at a tie, 0 real**, max 0.129 |
| cross-engine logits, same file (kld-raw, 100+ positions) | mean 0.0146, top-1 90.5%, margin-qualified 100% | mean 0.012, top-1 94.5%, margin-qualified 100% | mean 0.012, top-1 94%, margin-qualified 100% |
| quant bar vs Q8_0 (top-1 >= 97%, mean KLD <= 0.05) | not run | **PASS**: 400 positions, mean KLD 0.0036, top-1 99.75%, top-8 overlap 0.921 | **PASS**: 100 positions, mean KLD 0.0016, top-1 100%, top-8 overlap 0.835 |
| chat template conformance (jinja2 reference) | **22/22 text and token identical** | same template | **21/21 text and token identical** under the new `qwen38` family (its own template, not Qwen3's; see below) |
| compat-matrix row (Blackwell) | load, tokenizer, **cpu_cuda 9/9**, chat pass; greedy 3/5 at 8 tokens (both partings at ties) | load, tokenizer, **greedy 5/5**, chat pass; cpu_cuda 8/9, one flip at token 27 on a 0.0019-nat tie | load, tokenizer, **greedy 5/5**, chat pass; cpu_cuda not executed (insufficient VRAM) |

For scale, the certified granite-4.1-3b measured on the same kld-raw gate
today reads mean 0.017, top-1 93%, margin-qualified 99%: the 4.2 numbers
are the family's cross-engine floor, not a 4.2-specific defect.

## What changed in the runner

### Granite 4.2 header delta (R4.12.1)

Read off the three official GGUFs (3B, 8B, 30B) against granite-4.1:

- `tokenizer.ggml.pre` is **`granite-docling`** (4.1: `dbrx`). The runner
  refused it by name; it now maps to the GPT-2 family rule. That rule also
  learned to use the regex's `\p{L}` letter class instead of ASCII
  alphabetics, which is what the 5 divergences (curly apostrophes and
  Devanagari marks) were.
- The muP scalars are **all 1.0** (`embedding_scale`, `logit_scale`,
  `residual_scale`; 4.1 shipped 12, 16 and 0.22). The existing loader reads
  them from the header, so no code changed; the values are recorded here
  because a reader of the 4.1 certification would otherwise expect them.
- `tokenizer.ggml.bos_token_id` is 100283 and `add_bos_token` is absent,
  which both engines read as false.
- The chat template is new: ChatML framing, a `<think>` channel with
  `enable_thinking` defaulting to true, `truncate_history_thinking`, an
  optional `{reasoning effort: low}` marker, and the Qwen3-Coder function
  XML tool protocol (`<tool_call><function=NAME>...`), with tool results
  grouped into a user turn. The `granite` template (`<|start_of_role|>`)
  no longer applies to this generation.
- The tensor set is identical to 4.1's; the 30B is the same architecture
  at 40 blocks and was loaded but not measured (the 16 GB tier answer from
  the 4.1 certification stands: certify the 8B).

### The `granite42` template family

The template shares its declaration text and call syntax with Ornith's,
so `template_detect` could not tell them apart by those; it now checks for
the literal `<think></think>` first, which Granite 4.2 writes and Ornith
never does. The renderer is Granite 4.2's own, not Ornith's with edits:

- a system turn is always emitted, empty when none was given (the
  reference defines `system_message` as `""` and then tests `is defined`);
- a system turn is representable anywhere in the history;
- an assistant turn without a thought block is seeded with the closed
  `<think></think>`; a turn before the last user turn that carried a
  complete block keeps only what followed its last `</think>` behind that
  same seed; the latest assistant turn is replayed verbatim;
- a call turn is the trimmed content, one newline, and each call ending in
  its own newline; results are `<tool_response>` blocks in one user turn,
  each ending in a newline;
- the generation prompt opens `<think>\n`, or is the closed
  `<think></think>` when the caller turns thinking off, so the stream
  starts inside reasoning in every mode but that one.

Declarations unwrap the OpenAI `function` envelope and drop `defer_loading`
and `strict`, as the template's `tool_to_json` macro does. Gate:
`scripts/template-conformance.py --family granite42 --require-tokens`, 22
cases, and the `granite42` rows of `tests/test_template.c`,
`tests/test_tool_attribution.c` and `tests/test_tool_info.py` (native
protocol `qwen3_xml`). Tool calling on the agent-torture matrix is R4.12.3
and remains open; the protocol renders conformantly, the model's behaviour
under it is not measured here.

### Qwen 3.8 (R4.13.1, R4.13.2)

- Header: arch `qwen35`, 65 blocks of which one is NextN
  (`nextn_predict_layers 1`); the runner excludes it from the backbone and
  offers it to `--mtp`. The recurrent-layer device limits of `qwen35`
  carry over unchanged. The vision tower in the safetensors release has no
  GGUF counterpart and is out of scope.
- Pre-tokenizer: the `qwen35` rule now keeps combining marks (`\p{M}`)
  inside letter runs, the difference between the Qwen3 regex and the
  Qwen3.5 one. The 3 divergences were Devanagari and Thai strings.
- Side effect, declared in the manifest: Ornith's GGUF says `qwen35` while
  Ornith's own `tokenizer.json` still carries the Qwen3 regex, so its
  differential against the Hugging Face reference now reads 3/721 while
  the runner and llama.cpp master agree with each other on those strings
  (`[58069, 84237, 150104, 153348]`). The row's
  `check_params.tokenizer.expect_divergences: 3` records the expected count
  and the reason; `compat_matrix.py` passes it to `difftok.py --expect`.
- Fixtures: `vocab-bpe-granite-docling.gguf` is new, and the BPE fixtures
  gained curly-apostrophe, Devanagari-mark and digit-run cases for the
  GPT-2, llama3, qwen2 and qwen35 rules
  (`scripts/make-vocab-fixture.py`, `tests/test_tokenizer.c`).

## The 3B's token divergences, with a control

The 3B row is the one place the tie gate does not clear the table by
itself: 5 of its 11 divergences are "real" by the gate's definition, which
is that at least one engine prefers its own token by more than 0.25 nats.
The largest such preference is 0.935 nats (the reference, for "We" over
"Sentence" at position 7 of the France prompt; the runner sees a 0.04-nat
tie at the same position), and the position-by-position log-probability
deltas over the matched prefixes stay at or under 0.27 nats.

Whether that is a 4.2 defect or the floor of a 3B model at 4 bits against
this reference build is a question with a control: the **certified
granite-4.1-3b Q4_K_M** run through the same gate, the same runner build
and the same llama.cpp 73a43d1 on the same afternoon reads **8 identical,
1 length mismatch, 4 at a tie, 4 real**, max delta 0.248, with reference-
side preferences up to 0.639 nats
(`docs/granite-42-qwen38-evidence/tokdiv-granite-4.1-3b-control.txt`).
The one-sided preferences at the parting positions are of the same size on
both generations (0.35 to 1.0 nats summed over the two engines); the 4.2-3b
meets them on more prompts and earlier. On the six-prompt greedy set the
control reads **3/6** (partings at bytes 238, 48 and 976) where the 4.2-3b
reads 0/6 (bytes 19, 113, 47 and 2), so on that set the 4.2-3b is the
weaker of the two 3Bs, and this document says so rather than folding it
into the tie count.
The same-file logit comparison points the same way: over 200 corpus
positions the 4.2-3b's largest per-position KLD is 0.098 and its largest
top-1 disagreement margin 0.32 nats, where the certified 4.1-3b's are
0.212 and 1.26 nats. The 4.2-3b is not the noisier of the two. The
reading this document records is therefore the 4.1 certification's:
engine agreement inside the model's own numerical floor, that floor now
measured on a certified sibling rather than assumed, and the 8B on the
identical code reading 0 real of 9.

## The reference's own configuration, and a gold reference

Every identity number above compares the runner with llama.cpp, and
llama.cpp is a second implementation, not the truth. Two things were
measured on the 3B to say what the strict counts are counting
(`docs/granite-42-qwen38-evidence/`, all on the Blackwell, 2026-09-06).

**The reference's flash attention.** llama.cpp's CPU flash-attention
kernel (ggml-cpu/ops.cpp at 73a43d1, `flash_attn_ext_f16_one_chunk`)
converts Q to f16 for the K dot and accumulates the attention output in
f16 (`ggml_vec_mad_f16` into `VKQ16`) whenever the V cache is f16, which is
the default; the runner keeps Q and both accumulators in f32 over the f16
cache. With `-fa auto` the CPU build enables it. The gate's reference ran
that way, so `scripts/token_divergence.py` gained `--reference-args` and
the 3B was re-run:

| reference configuration | identical | at a tie | real | max delta |
|---|---|---|---|---|
| default (`-fa auto`, f16 KV) | 5 | 6 | 5 | 0.265 |
| `-fa off` | 9 | 4 | 3 | 0.229 |
| `-fa off -ctk f32 -ctv f32` | 6 | 6 | 4 | 0.248 |
| bf16 weights, `-fa off` | **16** | 0 | **0** | **0.020** |

The 8B under `-fa off` reads 13 / 2 / 1 (from 7 / 9 / 0), and the greedy
sets move without improving: 3B 0/6 either way, 8B 1/6 either way, Qwen
3.8 3/6 with flash off against 4/6 with it on. llama.cpp against itself,
flash on versus off, on the 3B over 200 corpus positions: mean KLD
**0.020**, max **0.84**, top-1 90.5%, which is more than it disagrees with
the runner (0.015, max 0.098). The runner against itself, batched prefill
versus one token at a time: mean KLD 0.00003, top-1 100%, max 0.005. The
strict counts on this model are counting the reference's configuration.

**A gold reference.** With the weights unquantized (IBM's own bf16 GGUF)
the two engines agree on all 16 prompts to 0.02 nats, so everything above
that is the two engines' different Q4_K arithmetic (llama.cpp quantizes
the activations to 8 bits per 256-block for the dot; the runner's dot
takes the f32 activations). Which arithmetic is closer to the model is a
question for the model's own implementation: `scripts/gold-logits.py`
runs transformers 5.16 in float32 on the CPU from IBM's safetensors and
scores both engines against it, 99 corpus positions:

| weights | engine | mean KLD vs fp32 | max | top-1 | margin-qualified |
|---|---|---|---|---|---|
| bf16 | runner | **0.0016** | 0.149 | **100%** | 100% |
| bf16 | llama.cpp `-fa off` | 0.0016 | 0.150 | 99.0% | 100% |
| bf16 | llama.cpp `-fa on` | 0.0018 | 0.151 | 97.0% | 100% |
| Q4_K_M | runner | **0.109** | 0.499 | 87.9% | 94.7% |
| Q4_K_M | llama.cpp `-fa off` | 0.119 | 0.613 | 85.9% | 97.3% |
| Q4_K_M | llama.cpp `-fa on` | 0.115 | 0.549 | 88.9% | 97.3% |

At bf16 the runner reproduces the reference implementation to the noise of
its own float arithmetic and picks its token on every position; at the
served quant the runner is the closer of the two engines to the model.
That is the reading this document records for the 3B: the architecture
is right on both engines, the residual is the quantization each engine
performs, and the runner's is the smaller.

## What the model authors say

- **IBM, Granite 4.2** (model card): `temperature=1.0` and `top_p=0.95`
  "across all tasks and serving backends", `do_sample=True`; thinking on
  by default, `enable_thinking=False` for direct answers,
  `low_effort=True` for brief reasoning; historical thinking stripped
  (`truncate_history_thinking`). The muP scalars, the attention scale
  (`attention_multiplier` used as the score scale, 1/64 on the 3B and
  1/128 on the 8B, in place of 1/sqrt(d)), the residual and embedding
  multipliers and the division by `logits_scaling` are what
  transformers' `modeling_granite.py` does and what both engines read
  from the header. The GGUF carries the same values as `config.json`
  (`rope_theta` 1e7, no rope scaling, untied embeddings, bos 100283).
- **Qwen, Qwen 3.8** (model card): thinking mode `temperature=1.0`,
  `top_p=0.95`, `top_k=20`, `min_p=0`; instruct mode `temperature=0.7`,
  `top_p=0.8`, `presence_penalty=1.5`; thinking on and `preserve_thinking`
  on by default; `reasoning_effort` xhigh (default), medium, low. The
  architecture in `config.json` (`qwen3_5_text`: Gated DeltaNet with
  L2-normalised q/k, `g = -exp(A_log) * softplus(a + dt_bias)`, sigmoid
  beta, a sigmoid output gate on attention, partial rotary 0.25 with
  interleaved multimodal rope sections, one MTP layer) is what
  llama.cpp's `qwen35.cpp` builds (`ggml_l2_norm`, `ggml_softplus`,
  `ggml_sigmoid` gate, `ggml_rope_multi` with the sections) and what the
  runner's `qwen35` path was measured against.
- **Qwen 3.8's chat template is not Qwen3's.** The GGUF's template was
  detected as Qwen3's and rendered through the `chatml-think` family;
  pointed at `Qwen/Qwen3.8-27B`, the conformance gate showed 20 of 21
  cases drifting. The template opens every conversation with a
  reasoning-effort preamble in the system turn whenever thinking is on,
  keeps every historical assistant turn's thought block, places tool
  declarations between the preamble and the caller's system text, and
  uses the function-XML call protocol. The new `qwen38` family renders
  all of that: 21 of 21 cases text- and token-identical
  (`conformance-qwen38-granite42-tokens.txt`), `reasoning_effort`
  honoured on the chat and Responses surfaces with the template's own
  three values. The Qwen 3.8 matrix row was re-run under it (chat pass,
  greedy 5/5; `docs/compat-reports/0.4.10-2026-09-06-blackwell-qwen3.8-27b-ud-q4_k_m.json`).

## Reading the greedy numbers

The strict greedy count (1/6 on the 8B, 4/6 on the 27B) is the number a
reader would compare with the 4.1 certification's 6/6, and it should be
read with the gate that was built for exactly this question. Both
references moved since 2026-08-11: llama.cpp master 73a43d1 is 250 builds
past b10353, and its Granite path is not byte-for-byte the one certified
against. `scripts/token_divergence.py` classifies every divergence by the
log-probability gap at the parting token: a gap under 0.25 nats is a tie
that either engine's rounding may break. On the 8B all 9 divergences are
ties and the largest gap is 0.187 nats; the same-file logit comparison
(`kld-compare-raw.py`) then says the two engines agree on every position
where the model itself is confident (margin-qualified top-1 100%). That is
the reading this document certifies: engine agreement inside the model's
own numerical floor, with the floor measured rather than assumed.

## Not measured

- `cpu_cuda` on Qwen 3.8: the CUDA slice was occupied. The `qwen35`
  device path is unchanged from the Ornith and Qwen3.5 rows.
- Granite 4.2 30B: loaded, not scored.
- Tool calling behaviour (R4.12.3, R4.13.3): protocol rendering is
  conformant; the agent-torture matrix was not run for either family.
- Metal: neither family was run on Apple silicon in this window; the
  Granite 4.2 architecture is 4.1's, which is certified there.
