# Changelog

Current release and unreleased changes. Compatibility follows the
[versioning policy](docs/versioning.md).

## Unreleased

- **`--imatrix FILE` weights the rounding of q3_k and q4_0 rows.** A rewrite
  can now take an importance matrix (llama.cpp's GGUF or legacy `.dat`
  imatrix, so a published one serves as it is) and round each q3_k and q4_0
  row with ggml's weighted quantizers: a column the calibration text drives
  hard keeps its precision at the expense of one it barely uses. Other types
  are written as before; a matrix with no calibration data is written
  unweighted; a misfit entry fails the rewrite. The `.quant.json` record
  names the imatrix by sha256. Without the flag every byte is what it was.

- **`--quant mxfp4` and `"type":"mxfp4"` write OCP microscaling FP4.** The
  rewriter read MXFP4 (gpt-oss ships its experts in it) but could not write
  it, so a gpt-oss artifact that went through a merge or a requantization
  left its native format. The writer is ggml's reference one: one E8M0
  power-of-two scale per 32 values, 2^(floor(log2(amax)) - 2), each value
  the nearest signed E2M1 level. A source already at MXFP4 is copied, not
  rewritten.

- **The KV gather covers short forwards and the decode microbatch.** With
  full offload the host KV cache stays authoritative, so every forward's new
  K and V rows go back to it. A decoded token's came back in one copy since
  the gather landed; a verify round, a short prompt and each member of a
  decode microbatch still took two synchronous copies per layer. A forward of
  up to 16 rows now gathers each row on the device and comes back in one
  copy, and a microbatch's members are gathered together behind one wait.
  On the RTX 3070, Qwen2.5-3B Q4_K_M at eight concurrent requests decodes at
  about 173 tok/s against 160 (two rounds each), and the copy time of a
  drafted Qwen2.5-Coder-7B run's verify rounds falls from 31 ms to 7. The
  bytes and host slots are the per-layer copies'; `RUNNER_CUDA_KV_GATHER=0`
  restores those (R3.4.1).

- **Codebook i-quant decode on CUDA: IQ2_XXS twice as fast, IQ3 6 to 14%
  faster.** The batch-1 IQ2 GEMVs converted each grid byte to float while it
  was a 64-bit integer, which the GPU runs as I2F.U64 on a slow pipe:
  granite-4.2-3b IQ2_XXS decoded at 48 tok/s on the RTX 3070 against 90 at
  IQ3_S. A byte now becomes a float exactly through the mantissa of 2^23 (two
  full-rate instructions, the same float), and the lane GEMVs keep four
  blocks of loads in flight. IQ2_XXS decodes at 104 tok/s (60% of llama.cpp,
  from 28%), IQ3_XXS at 102 (63%, from 56%), IQ3_S at 96 (62%, from 58%), and
  prefill gains up to 21%; output is byte-identical. IQ1_M (39%) keeps a
  one-lane-a-block layout whose fix changes the reduction order
  (`docs/compat-reports/cuda-iq-decode-2026-10-31/README.md`).
- **`--draft mtp` on CUDA is documented as such.** The NextN head has run on
  the device with the whole trunk offloaded since 2026-10-07, and the verify
  walk is target-exact there, but `--help` still called it the CPU path. On
  the RTX 3070 a Qwen3.5-4B Q4_K_M with its NextN block decodes at 112.8
  tok/s at `--draft-k 2` against 94.8 plain once the exact-row twins are
  faster (R3.3.4), output byte-identical
  (`docs/compat-reports/cuda-mtp-2026-10-31/README.md`).

- **Speculative decoding pays on CUDA at full offload.** A drafted run on a
  fully offloaded target was slower than plain decoding: on the RTX 3070,
  Qwen2.5-Coder-7B Q4_K_M with its 1.5B draft decoded at 61.0 tok/s at
  `--draft-k 4` against 70.9 plain, because a verify round's five rows cost
  3.2 one-token steps. The exact-row GEMV twins (each column the batch-1
  kernel's bits) waited on their loads at low occupancy. The Q4_K, Q5_K and
  Q6_K twins now stage x double-buffered with the next block prefetched,
  load each row's next weights while the current block computes, take their
  rows a warp and warps a block by width, and gain exact widths 5 to 7. A
  five-column Q4_K pass costs about 1.8 batch-1 passes, from about 4. The
  same run reaches 86.0 tok/s at k 4 (1.21x plain), 84.3 at k 6 and 74.5 at
  k 8, every output byte-identical to the plain one. The decode microbatch
  and the MoE expert matmuls use the same kernels
  (`docs/compat-reports/cuda-spec-verify-2026-10-31/README.md`).
- **A guide to measuring a quantized file against its publisher's
  reference** (`docs/measure-a-quantized-file.md`), for anyone who publishes
  GGUF quantizations: the publisher's checkpoint in float32 as the
  reference, one `--score` pass per file, `scripts/gold-logits.py` for mean
  KL and margin-qualified top-1 with a paired test between two files, and
  the instrument's own floor stated first (BF16 files of 11 of 12 models
  read 2.7e-5 to 2.4e-4 mean KL against their float32 references).

- **An example snapshot test for a model's output.**
  `examples/deterministic-ci/` replays a committed `--record` of a greedy
  completion with a pinned Runner release and a pinned model file (both
  checked by sha256), so a model's answer can be asserted in CI like any
  other snapshot. A workflow runs it on Linux x86_64 (another build: tier
  T2, token ids) and macOS arm64 (the recording binary: T1, bytes) when the
  example changes and weekly.

- **Every roster row with a file states its licence.** `docs/model-tiers.json`
  gains `licence` beside `upstream`: the licence as the publisher's model
  card states it on the Hub (its `license`, or its `license_name` when that
  is `other`), and the repository it was read from. A re-packer's tag does
  not change the parent's terms, so the parent is named: one row's GGUF
  repository is tagged `apache-2.0` while its parent, Qwen3.8-Flash-Next,
  is under `qwen-community-1.0`, and the row says the latter. The roster
  check refuses a row with a file and no licence, and
  `docs/model-support.md` shows it per file. It is an attribute to read and
  filter on; nothing is filtered by it. Today: 42 rows `apache-2.0`, 6
  `lfm1.0`, 3 `openmdw-1.1`, 3 `qwen-community-1.0`, 1
  `nvidia-open-model-license`, 1 `mit`.

- **Receipts carry the tool results a request sent, as data.** A served
  receipt named the calls its turn delivered (`tool_calls`), but the results
  of earlier calls reached it only inside the rendered prompt text, so a
  policy over the record could not see that a call retries a reported
  failure or repeats a write already done. A record now carries
  `tool_results`: per result the tool's text, verbatim, its parse as `json`
  when it is one JSON value, the Messages `is_error` flag, and the id, name
  and arguments of the call it answers, on the Chat, Responses and Messages
  surfaces, under the same chain hash and signature. It is written when the
  request carries a result or declares tools; an older record verifies as
  before.

- **Spark X2.5 and X2.6 run on Metal and CUDA.** The GPU backends refused
  `spark2_5` by name for two pieces, and both are now device kernels: the
  per-head attention output gate (one sigmoid per head over its outputs)
  and the gated exact GELU. Metal has no erf, so its GELU computes erf by
  Abramowitz and Stegun 7.1.26 (absolute error at most 1.5e-7, below float
  rounding inside the GELU); the CPU and CUDA use erff. The rest of the
  architecture (Q, K and V fused in one tensor, two rope widths and bases,
  the sliding window) already ran there. GPU against CPU at logit
  precision (limit 0.002 of the range): on the fixture 2.1e-06 mean
  |dlogit| on Metal and 3.3e-06 on CUDA, one token at a time, with every
  layer sliding and quantized to Q8_0 and Q4_0 likewise; on XHToken's
  Spark-X2.6-4B Q4_K_M 1.9e-05 (Metal) and 3.0e-05 (CUDA) of its range, on
  X2.5 2.3e-05 (CUDA). Both rows' Metal and CUDA cells pass, each reply
  byte-identical to the CPU pins
  (`docs/compat-reports/spark2_5-gpu-2026-10-30/README.md`). An expert FFN
  with the exact GELU still runs on the CPU (no such model exists).

- **Tool-call arguments that are not strings replay as the publishers'
  templates write them, on six families.** A list or object argument in an
  assistant turn's history was written as compact JSON; Qwen3-Coder's
  template family (Ornith, Qwen 3.5, Qwen 3.6, Qwen 3.8, Granite 4.2,
  Nemotron 3.5) writes it through `tojson`, whose separators are `, ` and
  `: `, and Ornith, Qwen 3.5, Granite 4.2 and Nemotron 3.5 write a boolean or
  null through Python's `str()` (`False`, `None`). The template conformance
  gate now runs a case with such arguments on each of these families, and all
  conform in text (Qwen 3.6 in tokens too).
- **Qwen 3.6's own template (`qwen36`).** Qwen 3.5's rules with tool-call
  scalars written as JSON and the `preserve_thinking` switch, read at the top
  level or inside `chat_template_kwargs` as `enable_thinking` is: `true`
  replays the thought of every assistant turn. Detected from the file's
  template, so a Qwen3.6 GGUF needs no flag.
- **Qwen3.6-35B-A3B admitted at Tier 2 on the CPU; bartowski's Q4_K_M passes
  the fidelity bar by a hair.** Against Qwen's BF16 parent with float32
  arithmetic cast at use per decoder block (300 positions), the standard
  Q4_K_M reads 0.0475 mean KLD and 97.2% margin-qualified top-1 (bars 0.05
  and 97%); the Q8_0 reads 0.0349 and 99.1%, its error concentrated at two
  positions another engine reproduces on the same file. The file carries
  Qwen's NextN head, and `--mtp` drafts from it with output unchanged, but
  slower than plain decoding on the CPU
  (`docs/compat-reports/qwen3.6-35b-a3b-reference-2026-10-31/README.md`).
- **The qwen35moe architecture on the CPU: Qwen3.5 and Qwen3.6-35B-A3B
  and the models built on them.** Qwen 3.5's Gated DeltaNet and gated
  attention with routed experts (softmax, top-k, renormalized) and a
  sigmoid-gated shared expert on every block. Runner's qwen35 path already
  carried that combination for qwen4exp; qwen35moe takes it without
  qwen4exp's hyper-connections, PLE and indexer. Against transformers' own
  float32 Qwen3_5MoeForCausalLM on a converted random model, Runner reads
  2.9e-08 mean KLD with top-1 on all 199 positions, where taking away any
  one feature (the shared expert or its gate, the renormalization, the
  DeltaNet's SiLU gate, the partial rope, the convolution's history) moves
  the reference by 0.16 to 8.7
  (`docs/compat-reports/qwen35moe-toy-2026-10-30/README.md`). `--mtp` now
  drafts from the files' NextN head, an attention block with the trunk's
  MoE FFN, where it refused every MoE backbone without hyper-connections;
  greedy output is unchanged by it. The GPU backends and the files' rows
  follow.
- **LFM2.5-24B-A2B Q4_K_M admitted at Tier 2; it fails the fidelity bar,
  and the publisher's Q8_0 is the fidelity file.** Against LiquidAI's BF16
  parent with float32 arithmetic cast at use per decoder block (300
  positions on the lab CPU lane), the Q4_K_M reads 0.144 mean KLD and 90.5%
  margin-qualified top-1 against bars of 0.05 and 97%; the Q8_0 reads 0.0297
  and 99.1% (`docs/compat-reports/lfm2.5-24b-a2b-reference-2026-10-30/README.md`).
  Its CPU cell passes on an x86-64 host with the GPU hidden. Its template,
  the same bytes in the repository and the GGUF, is the 1.2B's with two
  differences, and Runner rendered it as the 1.2B's: a history turn's thought
  is kept on every assistant turn after the last user turn, not on the last
  assistant turn only, and the generation prompt opens the thought
  (`<|im_start|>assistant` and a newline, then `<think>`), whatever
  `enable_thinking` says. It is now `lfm2-24b`, and its reply's reasoning is
  split from the opener the prompt wrote. Template conformance against the
  24B's own chat_template.jinja is 32 of 32 in text and tokens; 21 of the
  cases fail when it is rendered with the 1.2B's rules. Its tokenizer.json is
  byte-identical to the 8B-A1B's, and the Q4_K_M reads as the publisher's
  does on all 721 corpus strings, all 107 gold passages and every
  vocabulary entry.

## v1.9.0 - 2026-10-30

- **LFM2.5-8B-A1B Q4_K_M admitted at Tier 2; it fails the fidelity bar, and
  the publisher's Q8_0 is the fidelity file, narrowly.** Against LiquidAI's
  BF16 parent computed in float32 (300 positions on the lab CPU lane), the
  Q4_K_M reads 0.237 mean KLD and 85.4% margin-qualified top-1 against bars
  of 0.05 and 97%; the Q8_0 reads 0.0443 and 97.6%. That gap is the file's
  quantization: Runner on the publisher's BF16 GGUF reads 0.0078 (100%
  margin-qualified), and llama.cpp on the same Q8_0 reads 0.0572 and 96.7%,
  outside the bar (`docs/compat-reports/lfm2.5-8b-a1b-reference-2026-10-30/README.md`).
  The file loads, tokenizes as the publisher's tokenizer.json does (a
  128,000-token vocabulary of its own: 0 of 721 corpus strings, 0 of the gold
  corpus's 107 passages and 0 vocabulary entries with the whole-word rule and
  the Unicode class tables) and answers on the CPU of the Metal laptop. Its
  GGUF carries an earlier revision of the LFM2.5 template, which Runner
  rendered with the 1.2B's rules; it is now its own family, `lfm2-8b`: string
  arguments written raw, lists as Python's `str()`, the thought from
  `thinking` alone and kept on every assistant turn after the last user turn.
  Its model writes string arguments raw too, and they are read raw
  (`lfm2_python_raw`), up to the first quote a valid continuation follows.
  Template conformance against the GGUF's own template is clean in text and
  tokens.
- **The BPE splits read the Unicode classes their publishers' regexes name,
  and four split rules now match the publishers' exactly.** The splits are
  regexes over letters by case, marks, numbers, punctuation, symbols and
  white space, and Runner read them from ranges written by hand that took
  every code point above U+007F for a letter unless a listed block said
  otherwise: full-width punctuation, U+FEFF, the Arabic-Indic digits and most
  combining marks were letters, and case was known for Latin, Greek and
  Cyrillic only, so "：“" and "）\n" split where the publishers keep one
  piece. The classes are now a table generated from the Unicode 16.0
  database (`scripts/gen-unicode-classes.py`), checked on every code point
  against the reference tokenizer's own regex engine. With it: Tekken, Llama 4
  and gpt-oss try their case alternatives in the regex's order (" 亚洲AV" is
  " 亚洲" then "AV"); Trinity groups only decimal digits by three ("1²" is "1"
  then "²"), keeps Thai as the whole block and each script group apart;
  Trinity and Spark let a format character lead a letter run and keep text no
  alternative matches as one piece; and a merge whose key passes 512 bytes is
  built instead of skipped (MiniCPM5's 160 spaces were two tokens). Over
  every vocabulary entry decoded to text, 23 of 24 files checked now match
  their publishers on every entry (Llama 3.2 was 86, LFM2.5-8B-A1B 108,
  Qwen3 128, MiMo 1,176, the Tekken files 202), and two of granite 4.2's gold
  passages and two NFC strings each on LFM2.5, Mellum and MiniCPM5 now match;
  Ornith's file differs only where its tokenizer.json's older split does, as
  its row declares. The LFM2.5-1.2B and 350M rows declare 0
  (`docs/compat-reports/unicode-classes-2026-10-30/README.md`).
- **Validation runs at 1.8.0 on the Windows CUDA host: 8 rows** (RTX 3070,
  2026-10-30, built at the v1.8.0 tag, 9bb83e7e). `k2-horizon-0.9b-q8_0` and
  `k2-horizon-0.9b-q4_k_m` pass on the CPU and on CUDA with the whole file on
  the card. `lfm2.5-350m-q8_0`, `lfm2.5-350m-q4_k_m`, `spark-x2.6-4b-q4_k_m`,
  `spark-x2.5-4b-q4_k_m`, `k2-horizon-7b-q4_k_m` and `k2-horizon-7b-q8_0` pass
  on the CPU; their CUDA cells are declared limits (CUDA refuses lfm2 and
  spark2_5 by name, and the 7B's grouped norm has no GPU form). Every reply
  that has a pin on another device class is byte-identical to it.
- **A U+0000 in text is encoded as the publisher's tokenizer does, or
  refused, never cut there unseen.** The tokenizer gains `tok_encode_n`, which
  reads its text by length: "a\u0000b" was "a" alone and is now the three
  tokens the publishers' tokenizers give, on every vocabulary checked (Llama
  3.2, LFM2.5, SmolLM2, granite, MiniCPM5, Spark, K2-Horizon, Mistral-Nemo,
  gpt-oss, TinyLlama's SentencePiece and gemma-4's). Where the text goes on
  as a NUL-terminated string, a NUL is refused by name: a `-f` prompt file
  and a `--detect-watermark` text, as training text already was and as the
  server's JSON refuses `\u0000`. A chat line piped with a NUL in it was cut
  at the NUL and its rest sent as the next turn; the line is now refused whole
  and the next one is read. `difftok --vocab` compares each vocabulary's
  U+0000 entry too.
- **Llama 3's and LFM2.5-8B-A1B's tokenizers take a whole vocabulary word
  before merging, as their publishers' do.** The tokenizer.json files of
  Llama 3, Llama 4, gpt-oss, the Tekken vocabularies (Mistral-Nemo, Apertus,
  Nemotron 3.5 Lightning) and LFM2.5-8B-A1B and 24B-A2B set
  `ignore_merges`: a pre-token that is itself a vocabulary entry is that
  token. Runner's BPE always merged, and the merges do not always reach the
  word. LFM2.5-8B-A1B read " läsa" as " l" + "äs" + "a", so 7 of the gold
  corpus's 107 passages tokenized differently from the publisher's tokenizer,
  and Llama-3.2-3B read " Việt" as three tokens (1 of the 721 corpus
  strings); both now match on every one. The GGUF carries no key for the
  flag, so the split's name decides, each family read from its publisher's
  file. "lfm2" also names LFM2.5-1.2B's and 350M's vocabulary, whose
  tokenizer.json leaves the flag off; there the rule changes nothing, since
  the only entries its merges do not build whole are its 507 special
  tokens, which no split yields as a pre-token. `scripts/difftok.py --vocab`
  compares over every entry of the publisher's vocabulary decoded to text:
  Llama-3.2-3B goes from 674 differing entries to 86 and LFM2.5-8B-A1B from
  2,567 to 108, every changed entry now the publisher's; the Tekken, Llama 4,
  gpt-oss and LFM2.5-1.2B vocabularies and the families that keep their
  merges change on none. The entries still differing had a separate cause,
  the split's character classes, which this release also fixes (above): with
  both, the declared rows read 0. The compatibility matrix runs the
  vocabulary comparison for every row with a tokenizer reference and gates
  where a row declares its count; a new admission declares one
  (`docs/compat-reports/bpe-ignore-merges-2026-10-30/README.md`). The rows'
  smoke prompts tokenize as before.
- **Clef-Flash pinned at Tier 2; its Q4_K_M changes 3 of 29 decisions.**
  Cloudflare publishes no GGUF, so `cloudflare-clef-flash` pins ggml-org's
  Q4_K_M (`ggml-org/Clef-Flash-GGUF` at 6e2a164d, converted by
  ggml-org/convert from `Cloudflare/clef-flash` at f0b7d3cd), and
  `cloudflare-clef-flash-q8_0` its Q8_0. Against Cloudflare's own
  joint_schema_model.py in float32 (15 records, 29 questions), the Q4_K_M
  picks the reference's option on 26 of 29, every flip where the reference's
  top two are within 0.066 of each other, and the Q8_0 on 29 of 29 (max
  |dprob| 0.012): the file for decisions that must match
  (docs/compat-reports/clef-flash-reference-2026-10-30/README.md). The
  tokenizer matches Cloudflare's tokenizer.json (0 of 721, 0 of 13,264 on the
  NFC corpus). The file's own template, which is Qwen3.5's, conforms in 22
  of 22 cases in text and tokens. The Q4_K_M passes on the CPU and on CUDA
  with byte-identical replies, its 15 fixture decisions agreeing between the
  two (max |dlogit| 1.8e-4); the Q8_0 passes on CUDA as a layer split and on
  an x86-64 CPU.
- **The flint architecture, its tokenizer rule and its template.**
  RecursiveSI's flint loads on the CPU: QK norms and learned attention sinks
  on every block, a sliding window with rope on the blocks the file's pattern
  marks and no rope on the full-attention blocks, two dense blocks, then
  sigmoid-routed experts with an always-on shared expert. Checked against
  transformers' own flint in float32 on a converted random model: top-1 on
  all 199 positions, log-probabilities within the f16 KV cache's rounding
  (`docs/compat-reports/flint-2026-10-30/README.md`). The tokenizer pre `flint`
  reads 0 of 721 corpus strings differently from the publisher's
  tokenizer.json, and the `flint` template renders its own template token for
  token (23 of 23), with Spark's tool calls and its own `<|developer|>` turn
  (every other family still serves `developer` as its system turn).
  Generation stops at the role tokens the file lists as end-of-generation. On
  the publisher's Q4_1, Runner agrees with another engine at 0.028 mean KLD
  and 99.0% margin-qualified top-1 over 200 positions, both on the CPU; that
  is agreement, not fidelity, since a float32 reference of the 39.8B model
  (about 160 GB) is not feasible here. The file's admitting CPU cell passes on
  an x86-64 host with the GPU hidden; the GPU path is the next change.
- **The CUDA gate's reports are written with LF on every host.**
  `scripts/cuda-smoke.py` and `scripts/cuda-act-probe.py` opened their JSON
  reports with the platform's line ending, so the 1.8.0 gate on the Windows
  CUDA host wrote CRLF and the files were converted by hand before commit.
  Both now write `\n`, as `release-cuda-gate.py`'s redaction already did.

## v1.8.0 - 2026-10-30

- **A sigmoid or SiLU below -88 no longer turns into NaN on x86-64 CPU (a
  wrong-output fix in 1.7.1).** The build's fast math made the plain
  1/(1+exp(-x)) and x/(1+exp(-x)) return NaN on x86-64 for every input below
  about -88; arm64 builds returned 0, so no Apple Silicon run met it. On an
  x86-64 CPU, 1.7.1 wrote NaN wherever a router logit, a gate or a
  convolution output went that negative, and one NaN poisons the whole
  block: RecursiveSI flint's layer-46 router reaches -270 and every reply was
  "!". Whether an input goes that negative depends on the model; the admitted
  rows' x86-64 CPU cells matched their arm64 pins, so none is known to have
  reached it. Every CPU sigmoid and SiLU now goes through one guarded form
  (src/act.h); scripts/check-fpclass.py refuses the plain form and
  tests/test_act.c checks both under the build's own flags. The GPU kernels
  keep IEEE results there: Metal measured on an M1 GPU (exact 0 below -88),
  CUDA by its IEEE division (no fast-math flag), checked on the device by
  scripts/cuda-act-probe.py in the release's CUDA gate (#473).
- **YaRN reads the ramp and the attention factor the file declares.** A YaRN
  model's `beta_fast`, `beta_slow` and `attention_factor` (written as
  `rope.scaling.yarn_beta_fast`, `_slow` and `rope.scaling.yarn_attn_factor`)
  were ignored for 32, 1 and Runner's own `1 + 0.1 ln(factor)`, so a model
  trained with other ends roped with the ramp in the wrong place; IFM's
  K2-Horizon-0.9B declares 128 and 4. A file without the keys is unchanged.
- **The k2-horizon architecture and its tokenizer rule.** IFM's dense
  K2-Horizon models are llama-shaped except for their RMSNorms, which
  normalise the hidden vector in equal groups before one full-width weight
  (four on the 7B, one on the 0.9B). The grouped norm runs on the CPU, so a
  grouped file falls back from the GPU by name and refuses LoRA training by
  name; the family's MoE, value-expert and gated members are refused by name.
  The tokenizer pre `k2-horizon` (the Llama 3 split with combining marks and
  zero-width joiners inside letter runs, under NFC) reads 0 of 721 corpus
  strings differently from the publisher's tokenizer.json on the 0.9B.
- **K2-Horizon-0.9B and 7B, Q8_0 and Q4_K_M, admitted at Tier 2; both Q4_K_M
  files fail the fidelity bar.** IFM's own Q8_0 loads, tokenizes as the publisher's
  tokenizer.json does (0 of 721), renders the template its GGUF carries token
  for token (23 of 23) and answers through it on the Metal laptop, on the GPU
  and on the CPU; against the BF16 checkpoint computed in float32 it reads
  0.0037 and 97.3% top-1 (100% margin-qualified), the fidelity file. The
  publisher's Q4_K_M reads 0.108 against a 0.05 bar (97.1% margin-qualified),
  the file's quantization; it is a Tier 2 row of its own with that failure
  first, as MiniCPM5's Q4_K_M is
  (`docs/compat-reports/k2-horizon-0.9b-reference-2026-10-30/README.md`). The
  7B's two files are Tier 2 as well: their tokenizer reads 0 of 721, the
  template is the 0.9B's, and against the BF16 parent in float32 the Q8_0
  reads 0.0159, the fidelity file, while the Q4_K_M reads 0.1225 and fails the
  bar (`docs/compat-reports/k2-horizon-7b-reference-2026-10-30/README.md`).
  Both answer on the CPU of an x86-64 host with the GPU hidden; the 7B's
  grouped norm has no GPU kernel yet.
- **YaRN's attention factor follows the factor in use.** A file's
  `rope.scaling.yarn_attn_factor` is the magnitude for its own factor; under
  `--yarn-factor` or an automatic context extension Runner now takes the
  default for the factor it applies.
- **The k2-horizon chat template.** Turns framed by `<|ifm|im_start|>` and
  `<|ifm|im_end|>`; declared tools written as the template's markdown, or as
  one JSON line per tool when a schema uses a construct its markdown renderer
  does not handle; a think block on every assistant turn, empty when the turn
  carries no reasoning, as the template writes it; calls as `<ifm|tool_calls>`
  blocks, parsed on the way out, buffered and streamed (`ifm_argkv_xml`).
  Schemas the template refuses are answered 400 with its reason on all three
  surfaces, and a 44-case tool-schema set is checked byte for byte against the
  template's own renders. `reasoning_effort` medium and low are refused: their
  think blocks are not split from the answer yet.
- **LFM2 reads a history turn's thought from the fields its template reads.**
  LFM2.5's chat template takes the thought from `thinking`, then
  `reasoning`, then `reasoning_content`, the first one set and not empty, and
  keeps nothing when that one is not a string. Runner read
  `reasoning_content` only, so a client that sends `thinking` or `reasoning`
  had the last assistant turn's thought dropped from the prompt. Template
  conformance for `lfm2` is 29 of 29 in text and tokens, with a case for each
  spelling, their order, an empty field passed over and a non-string winner.
- **Spark-X2.6-4B and Spark-X2.5-4B Q4_K_M admitted at Tier 2; both fail the
  fidelity bar.** Against XHToken's BF16 checkpoints computed in float32 the
  publisher's Q4_K_M files read 0.113 (X2.6) and 0.1136 (X2.5) mean KLD over
  300 positions, against a 0.05 bar, each with 87.0% top-1. XHToken's Q8_0
  files read 0.0132 and 0.0044 on the same build, so the gap is the files'
  quantization and the Q8_0 files are the fidelity files
  (`docs/compat-reports/spark-2026-10-30/README.md`). The Q4_K_M files load,
  tokenize as the publisher's tokenizer.json does (0 of 721), render the
  publisher's templates token for token (23 of 23 each) and answer through
  them on the CPU of the Metal laptop. The architecture also agrees with
  llama.cpp on the X2.6 1.7B Q8_0 (0.0065 mean KLD over 200 positions).
- **The spark2_5 architecture loads, on the CPU.** XHToken's Spark 2.5 and 2.6
  are a dense transformer with three things Runner did not have together: Q,
  K and V fused in one `attn_qkv` tensor with the FFN left separate, a
  per-head sigmoid output gate (one value per head, where afmoe's gate has one
  per element), and a gated GELU in its exact erf form rather than the tanh
  approximation. The sliding layers rope over all 256 dims at base 10,000 and
  the full layers over 64 at 5,000,000. The per-head gate and the exact GELU
  have no CUDA or Metal kernels yet, so a GPU run falls back to the CPU and
  says so; LoRA training is refused by name. Pinned by fixture tests (loads,
  decodes deterministically, the gate and the window pattern participate,
  training refused).
- **Spark's tokenizer rule, chat templates and tool calls.** XHToken's Spark
  X2.5 and X2.6 files name the pre-tokenizer `spark2_5`, which Runner did not
  know; it is now its own rule (every digit alone, CJK runs apart, newlines
  one at a time) and 0 of 721 corpus strings differ from the publisher's
  tokenizer.json. Their two chat templates are families of their own
  (`spark`, `spark25`): turns framed by `<｜start▁of▁sentence｜>` and role
  tags, a system block always written, thinking on by default, X2.6's
  thinking-effort text, and GLM-style calls
  (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`)
  parsed on the way out, buffered and streamed, with a value that spells the
  closing tags kept whole and a truncated call never completed. Template
  conformance is 23 of 23 cases for each in text and tokens. The MiniCPM5
  parser now shares its buffered mapper and streaming demultiplexer with this
  one.
- **Validation runs on the Windows CUDA host: 4 rows** (RTX 3070, 2026-10-30,
  built at 64cdc0dc, MiMo at 80c30df3). `trinity-nano-preview-q8_0` re-runs
  its CPU cell after the template's blank lines (#464), and its reply is
  byte-identical to the pin. `lfm2.5-1.2b-instruct-q8_0` and
  `lfm2.5-1.2b-instruct-q4_k_m` pass on the CPU, both replies byte-identical
  to the Mac pins; their CUDA cells are declared limits, since CUDA refuses
  lfm2 by name. `mimo-v2.6-distill-qwen-9b-q4_k_m` passes on the CPU and on
  CUDA with the whole file on the card, both replies byte-identical to the
  Mac pin.
- **LFM2.5-350M Q8_0 and Q4_K_M admitted at Tier 2; the Q4_K_M fails the
  fidelity bar by far.** Against LiquidAI's BF16 parent computed in float32
  the publisher's Q4_K_M reads 0.874 mean KLD over 300 positions (54.3%
  top-1), and llama.cpp reads the same file at 0.924; asked for the planet
  closest to the Sun it answers Earth. The Q8_0 reads 0.0081 and 95.3% top-1
  and is the fidelity file
  (`docs/compat-reports/lfm2.5-reference-2026-10-30/README.md`). Both files
  share the 1.2B-Instruct's tokenizer and template byte for byte and answer
  through them on the CPU. The LFM Open License v1.0 limits commercial use to
  entities under US$10M annual revenue.
- **MiMo-V2.6-Distill-Qwen-9B Q4_K_M admitted at Tier 2; it fails the
  fidelity bar.** Against XiaomiMiMo's BF16 checkpoint computed in float32 it
  reads 0.0717 mean KLD over 300 positions, against a 0.05 bar (91.7% top-1,
  98.2% margin-qualified), and 0.064 against bartowski's own Q8_0. The Q8_0
  reads 0.0075 and 99.0% on the same build, so the gap is the file's
  quantization and the Q8_0 is the fidelity file
  (`docs/compat-reports/mimo-v2.6-distill-qwen-9b-2026-10-29/README.md`).
  bartowski's Q4_K_M of XiaomiMiMo's Qwen3.5-9B distill loads, tokenizes as
  the publisher's tokenizer.json does (0 of 721), renders the publisher's
  template token for token (23 of 23) and answers through it on the CPU of
  the Metal laptop; the Windows CUDA host's CPU and CUDA cells follow below.
  Runner's `qwen35` path has no Metal backend, so the row has no Metal cell.
- **MiMo-V2.6-Distill-Qwen-9B renders with its own template and tokenizes as
  its publisher does.** Its chat template was detected as Qwen3-Coder's, so
  every turn was framed with newlines the template never writes and no
  assistant turn carried its thought block. It is a family of its own
  (`mimo`): turns joined with no newline, `<think>`reasoning`</think>` on every
  assistant turn, a kept `tool` role, tools as a system turn of their own and
  Qwen3-Coder's XML calls written compact, parsed by the Qwen3-Coder parser.
  Template conformance against XiaomiMiMo's template is 23 of 23 in text and
  tokens. Its tokenizer.json splits with the qwen2 rule while the GGUF
  declares `qwen35`; a file carrying MiMo's template now reads with the qwen2
  split, as Clef does, and 0 of 721 corpus strings differ (3 did, all
  Devanagari or Thai). A `required` or named tool choice takes the generic
  envelope on this family.
- **LFM2.5-1.2B-Instruct Q8_0 and Q4_K_M admitted at Tier 2; the Q4_K_M fails
  the fidelity bar.** Against LiquidAI's BF16 parent computed in float32 the
  publisher's Q4_K_M reads 0.115 mean KLD over 300 positions, against a 0.05
  bar, and llama.cpp reads the same file at 0.119, so the gap is the file's
  quantization; the Q8_0 reads 0.0042 and 99.0% top-1 and is the fidelity
  file. The publisher's QAD-Q4_0 fails as well (0.128) and is not a row
  (`docs/compat-reports/lfm2.5-reference-2026-10-30/README.md`). Both files
  load, tokenize as the publisher's tokenizer.json does (0 of 721), render
  its template token for token (23 of 23) and answer through it on the CPU.
  The LFM Open License v1.0 limits commercial use to entities under US$10M
  annual revenue.
- **The `lfm2` and `lfm2moe` architectures, on the CPU.** Liquid's
  short-convolution layers interleaved with GQA attention: `in_proj` into B,
  C and x, B·x through a causal depthwise convolution, C gating it,
  `out_proj`. The convolution ring is the whole recurrent state, so snapshot,
  rewind, the prefix cache and the speculative round checkpoint use the
  Mamba-2 hybrids' seam, each pinned bit-identical to a cold decode on a dense
  and an MoE fixture. `lfm2moe`'s sigmoid-routed experts with a selection
  bias match transformers on a converted random model (max |Δlog p| 5.2e-4
  with the f16 KV cache). The GPU backends refuse both by name; LoRA serving,
  training and sublayer removal refuse them too. The tokenizer pre `lfm2` is
  the Llama 3 split with BOS on, as the publisher's tokenizer.json does.
- **The `lfm2` template family and its Python call list.** ChatML with the
  declarations as one `List of tools: [...]` line, the thought kept on the
  last assistant turn only, results as their own `tool` turn, and calls as
  `<|tool_call_start|>[name(key=value)]<|tool_call_end|>`, read back by a
  strict literal parser that evaluates nothing. Anything that is not a call
  list is served as text. Its two markers are control tokens; while the
  protocol is in force the engine spells them for the parser, as
  `special_tokens` does for completions. Template conformance gains two kinds
  of case: a part-array content handed to a template that reads the array
  itself, and a thought in history.
- **A file's own end-of-generation ids are stops.** Runner now reads
  `tokenizer.ggml.eog_token_ids` when a file carries it and ends the turn on
  each id it lists (Harmony's `<|end|>` stays out, as before). No llama.cpp
  converter writes the key; flint's publisher does. Of the roster files
  scanned, only flint's carries it, so no admitted model's stops change by
  this half.
- **An ordinary `</s>` no longer ends a turn.** The turn ends Runner knows by
  spelling now count only when the token is control or user-defined. Qwen3's
  vocabulary (id 128247, on the Qwen3 and Qwen3 MoE rows) and Gemma 4's
  (id 212, on every Gemma 4 row and Nemotron 3.5 Bolt) hold `</s>` as an
  ordinary token, which their tokenizers never produce from text (`</s>`
  encodes as `</`, `s`, `>`); a model that samples one now gets `</s>`
  printed instead of its turn ended. Every other roster file's stop set is
  unchanged.

- **Trinity Nano's prompt now has the blank lines its template writes.** Its
  chat template wraps the assistant span in `{% generation %}` tags that do
  not trim the whitespace around them; read as transformers reads them,
  every assistant turn in the history is followed by a blank line after its
  header and after its `<|im_end|>`. Runner served Trinity through the
  generic ChatML renderer, which wrote neither, so every multi-turn
  conversation reached the model two newlines short per assistant turn. The
  new `trinity` template (detected by the untrimmed tag) writes them; a
  single-turn prompt is unchanged.
- **The template-conformance harness parses `{% generation %}` as a block.**
  It deleted the tag from the template source, which joined the whitespace on
  both sides of it and hid the Trinity difference above. A template's own
  raise message also reaches the report now (it surfaced as a NameError).
- **`reasoning_effort` reads the same on the Responses surface as on chat.**
  `/v1/responses` honoured it only for Qwen 3.8, so Muse's
  `reasoning_strength` was dropped there and a value its template refuses was
  served. One reading now serves both surfaces.

- **Validation runs at 1.7.1 on the Windows CUDA host: 2 rows** (MiniCPM5-2B,
  RTX 3070, 2026-10-29, built at 77192431). `minicpm5-2b-q8_0` and
  `minicpm5-2b-q4_k_m` pass on the CPU and on CUDA, both files resident on
  the card. Every reply is byte-identical to the Mac pins except the
  Q4_K_M's x86 CPU reply, which picks a different word in its first
  sentence ("The smallest planet" against "The closest planet") and is
  pinned on its own; it still names the fact.
- **A rotated-basis ternary file is refused instead of answering garbage.**
  Prism ML's Ternary-Bonsai-2 GGUFs store every weight matrix in a
  Hadamard-rotated basis (`prism.hadamard.*`) and need the matching
  activation rotation at run time. Their Q2_0 packing is a type Runner reads
  and their architecture (`qwen35`) is admitted, so
  `Ternary-Bonsai-2-2B-Q2_0.gguf` loaded and answered with garbage at exit 0.
  Any file that declares the rotation is now refused at load, naming the key.
  The PQ2_0 and PTQ1_0 packings use tensor types this build does not read.
  Supporting the format is a separate decision.
- **MiniCPM5-2B admitted at Tier 2; its Q4_K_M fails the fidelity bar.**
  openbmb's own Q4_K_M against openbmb's BF16 parent computed in float32 reads
  0.102 mean KLD and 86.0% top-1 over 300 positions, against a 0.05 bar, and
  0.098 against its own Q8_0. The Q8_0 (2.68 GB) is the fidelity file: 0.0033
  and 98.0% against the same parent on the same build. Both rows carry Metal and
  CPU cells from the Metal laptop with byte-identical replies; the Windows CUDA
  host's cells and a llama.cpp column are not run
  (`docs/compat-reports/minicpm5-2b-reference-2026-10-29/README.md`).
- **MiniCPM5 loads with its own tokenizer rule and chat template.** The
  GGUF names the pre-tokenizer `minicpm5`, which Runner did not know and read
  with the GPT-2 split, so 210 of the 721 tokenizer corpus strings came out
  differently from the publisher's tokenizer.json; it is now the Llama 3
  split applied after the digit groups are cut out (`src/tokenizer.c`), and 0
  of 721 differ. Every prompt starts with `<s>`, as the publisher's tokenizer
  prepends it on every encode (the file's `add_bos_token=false` describes a
  tokenizer the publisher does not ship). Its chat template, which Runner
  rendered as Qwen3's, is a family of its own (`minicpm5`): a thought block
  on every historical assistant turn, `enable_thinking: true` pre-seeding an
  open `<think>`, and an XML tool protocol (`<function name=...><param
  name=...>`, CDATA for a value holding `<`, `&` or a newline) declared under
  the template's own guideline text and parsed on the way out, buffered and
  streamed, with a truncated call never completed. Template conformance
  against `openbmb/MiniCPM5-2B` is 23 of 23 in text and tokens; the BOS the
  template spells is the tokenizer's, as on Llama 3 (allowlist
  `minicpm5/bos`). A `required` or named tool choice takes the generic
  envelope on this family.
- **`-hf` and `--pull` refuse a name the Hub cannot hold.** An owner or
  repository name was only checked for its one `/`, so a space, `?`, `#`, `%`,
  a backslash, a `:` or a non-ASCII byte went into the API URL as typed and
  into the cache directory's name (`a\b--c` on a POSIX host, a stream separator
  on an NTFS one), and `.` or `..` as a whole segment was accepted. Both
  segments are now checked against the Hub's own alphabet (letters, digits,
  `-`, `_`, `.`) before any request or directory exists, and the refusal is the
  existing `wants owner/repo[:TAG]` one. Pinned by thirteen cases in
  `tests/test_hf_spec_bounds.py`, each also asserting that the cache stays
  empty.
- **The README and the manual state the counts the binary reports.** Both
  said how many architectures Runner admits (20 and 19) and both had drifted
  from the 22 that `--info caps` lists. The counts are corrected and
  `tests/test_doc_counts.py` ties them, and the 24 weight formats, to
  `--info caps` so they cannot drift again.
- **Validation runs at 1.7.1 on the Windows CUDA host: 8 rows** (Tier 2, RTX
  3070, 2026-10-29). CPU and CUDA cells pass for `gemma-4-e2b-it-q4_k_m-q4_0-mix`,
  `qwen3-4b-q4_k_m-bartowski`, `qwen3-4b-runner-tooluse-lora` (on that base),
  `qwen3-4b-runner-tooluse-merged-q8_0`, `gemma-4-12b-it-qat-q4_0` (CUDA 47 of
  48 layers), `granite-4.2-8b-q8_0` (32 of 40) and
  `nvidia-nemotron-nano-9b-v2-q8_0` (43 of 56); `trinity-nano-preview-q8_0`
  passes on the CPU (its CUDA cell is a declared limit). Every reply is
  byte-identical to the row's earlier pins. The rows of 13.8 GB and up are
  out of this host's reach and listed in the PR.
- **Validation runs at 1.7.1 on the Windows CUDA host: 3 rows** (Tier 1, RTX
  3070, 2026-10-29). `qwen3.8-27b-gsq-rco-iq3_s-recovered` CUDA (a layer
  split, 38 of 64 layers), `xyntetik-kvist-14b-q5_0-mix` CUDA (43 of 52) and
  `xyntetik-kvist-14b-iq4_nl-mix` CPU and CUDA (51 of 52) pass, each reply
  byte-identical to the row's linux-x86_64-cuda pin. The other Tier 1 rows
  are out of this host's reach and listed in the PR for the lab box.
- **Mellum2.1-12B-A2.5B-Thinking Q4_K_M and Nemotron 3.5 Bolt E4B Q4_K_M move to
  Tier 1** on the owner's decision of 2026-10-29: both are publisher files, so
  the 2026-10-27 pull rule yields to the owner's call, and the tier policy now
  says so. Their admitting cells stand as the record; the Mellum Metal row and
  Q8_0 cells and the Bolt BF16-parent fidelity measurement are still owed.
- **Mellum2.1-12B-A2.5B-Thinking admitted at Tier 2; its Q4_K_M fails the
  fidelity bar.** JetBrains' own Q4_K_M against JetBrains' BF16 parent,
  computed in float32 at 300 positions (token-id keyed): mean KLD 0.128 and
  top-1 85.7% on Runner, 0.131 and 85.9% on llama.cpp b11748. The file fails
  the 0.05 bar on both engines, and the gap grows with depth (about 0.08 in
  the first 750 words, 0.16 past 2,000). Against JetBrains' own Q8_0 it fails
  the quant bar too (mean KLD 0.131, top-1 80.0%, 400 positions). The engine
  reproduces the files as published: the two engines are 0.0177 apart on
  the Q4_K_M, and against the parent Runner's Q8_0 reads 0.0187 and 94.3%,
  llama.cpp's 0.0234 and 93.6%. So the gap is the file. Q8_0 (12.9 GB) is
  the fidelity file, and the publisher's MXFP4_MOE file is untested. The new `mellum` architecture is Qwen3-MoE
  attention and routing with a 1,024-token sliding window on three of every
  four layers and YaRN on the full layers only. Its tokenizer pre `mellum2`
  is the SmolLM split (0 of 721 corpus strings differ from the publisher's
  tokenizer.json). A needle was retrieved at 3,568 tokens. CUDA runs its MoE
  on the scalar route; the tensor-core and grouped routes are not measured
  for this family. The 8.07 GB file does not load resident on an 8 GB
  machine.
- **The `mellum` chat template.** Mellum2.1 renders as Qwen3's
  `chatml-think` except at one site. It writes a thought block on an
  assistant turn after the last user query only when that turn's reasoning
  is non-empty. So a trailing historical answer gets no empty
  `<think>\n\n</think>`, and non-empty `reasoning_content` is replayed on
  every assistant turn after the last user query. Auto-detected from the
  file's template; template-conformance passes 23 of 23 cases in text and
  tokens.
- **`/v1/completions` `top_logprobs` never repeats a key.** Two candidate ids
  whose text escapes the same (a byte-level vocabulary renders each lone
  partial UTF-8 byte as `�`) were written as two members with one key,
  which most JSON parsers collapse to the last. They are now one member
  holding the log of their summed probability. `top_token_ids` keeps every
  id, and the new `top_token_logprobs`, aligned with it, carries each id's
  own logprob; `stop_logprobs` gains the same field. The Mellum2.1 parity
  record found five positions that repeated a key.
- **`scripts/tier-smoke.py`: `backend_args`.** A row's manifest entry can
  name server arguments for one backend only, for example the layer split
  of a file larger than the card. The CPU cell does not receive them, and
  the pin records them, so the `ran` line shows the split.
- **A VRAM queue with nobody to wait for returns at once.** `--wait-for-vram`
  queues behind other registered runners. When the registry listed no holder at
  all and the file was larger than the card, the queue still polled its whole
  budget (300 s by default under `--serve`) before the load fell back to what
  was free and the adaptive split trimmed layers onto the CPU; a layer-split
  server, and the compat matrix's load and chat checks, lost five minutes a
  start on the Windows CUDA host. The claim now refuses immediately when the
  ledger is empty, the same verdict `--wait-for-vram 0` gives, and the fallback
  runs at once. A queue behind a holder, including another slot of the same
  server, waits as before. Pinned by two registry tests.

## v1.7.1 - 2026-10-29

- **Load notices state the engine's own evidence.** The architecture lines a
  load prints for the Gemma 4 variants, `granitehybrid` and `nemotron_h` now say
  what was validated and where the record is (`docs/model-support.md`,
  `docs/moe-support.md`) instead of naming another engine as the yardstick; the
  same for two `--help` lines. The evidence itself is unchanged and stays in
  the docs, where every comparison keeps its second column.
- **`--mlock` leaves the E-series per-layer token table pageable.** It
  wired the whole file, including the 2.99 GB table of a 6.18 GB E4B Q4_K_M
  that the forward reads one row per token. It now wires the hot ranges on
  either side of that table, keeping the pages it shares with its neighbours.
  Slots that share weights share one lock, held until the last of them is
  freed; before, a slot that joined locked weights did not know they were
  locked, so its GPU load could hand their host pages back. On Windows each
  range now adds to the working-set minimum instead of resizing it alone.
  `test-mlock` checks the wired bytes against the kernel's locked-memory
  limit and hand-computed page boundaries. On an 8 GB Apple Silicon laptop
  the Bolt E4B server wired 3.19 GB, and its first response after 8 minutes
  idle took 4.7 s with `--mlock` against 24.9 s without it; the measurement
  is the new `measurements` line of that row in `docs/model-support.md`. Help and manual updated; README,
  site and model cards do not describe `--mlock` and are unchanged.
- **E-series fit reports distinguish row-gathered weights.** `--info fit`
  retains the whole file size and reports the per-layer token table separately
  from the hot set used for its RAM verdict, including on discrete GPU splits.
  The tied output head and model projection still count in full. Header-only
  tests anchor the subtraction to Q8_0 block sizes. README, manual and site
  updated; model cards unchanged because this changes the estimator, not an
  artifact's measured performance or fidelity.
- **The engine merge to main is green in about 12 minutes instead of 55.**
  `make test` is now three targets run in order, `test-gates-core`,
  `test-gates-batch` and `test-python`, and CI runs them as three parallel
  jobs on macOS and on Windows instead of one fifty-minute step; the root
  pytest runs under pytest-xdist there (`PYTEST_WORKERS=auto`, whole files
  per worker; 1145 s to 490-603 s on the hosted Mac). Two dispatched runs
  of the whole workflow took 10 min 30 s and 12 min 8 s with the same 2050
  cases and the same 98 skips. A local `make test` is unchanged in what it
  runs. No check was removed: the microbatch gates, the Metal microbatch
  identity and the attention split moved to `test-gates-batch`, the client
  tests and the root pytest to `test-python`, everything else stayed in
  `test-gates-core`; one test that compared two fit reports byte for byte
  now masks the live free-RAM figure they quote.
- **The Windows suite's python job no longer waits on one test file.**
  `scripts/check-release.py` runs its tree scans (two `git grep` passes and
  the KL-table walk over the docs) once per tree state instead of once per
  `check()` call, the state being the checkout, its HEAD and every changed
  or untracked path with its size and mtime: the release workflow calls
  `check()` once, so it scans exactly as before; the test module calls it
  twenty-eight times on one unchanged tree and now scans once. On the hosted Windows runner every repeat cost about sixteen
  seconds, so `tests/test_release_check.py` was 628 of the root pytest's
  1378 case-seconds and the whole of its 647 s wall, and
  `windows-suite-python` took 14 min 37 s on the 2026-10-29 merge against
  the 15-minute gate. Three tests pin the verdict's lifetime (one tree, one scan; an
  edit, a new file, a deletion or a commit is a new state), that a tree git
  cannot describe is never cached, and that a failed scan stays failed.
  Same cases, same skips.
- **Nemotron 3.5 Bolt E4B admitted at Tier 2.** NVIDIA's post-train of
  Gemma-4-E4B-it in the publisher's own Q4_K_M
  (`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`), on the owner's request for the
  Metal laptop: pulled and verified through `--pull`, loaded on Metal,
  tokenized as the publisher's tokenizer.json does (0 of 721 corpus strings
  differ) and answered through its chat template; the validation smoke is
  pinned and recorded on `macos-arm64-metal`, and on the Windows CUDA host
  as CUDA and CPU cells (`windows-x86_64-cuda`) with a byte-identical reply
  across the three cells, a CPU-vs-CUDA agreement of 9 of 9 prompts, and a
  20-step `runner train --train-gpu` smoke whose adapter changes the answer.
  No fidelity measurement against the BF16 parent yet, and the repository's
  DFlash drafter file is not a Runner artifact. `scripts/compat_matrix.py`
  now decodes every child's pipes as UTF-8 on every platform; the Windows
  report had stored runner's em dashes and an emoji as cp1252 mojibake. `scripts/tier-smoke.py --row` and
  `scripts/model-tiers.py --record` now accept a Tier 2 row, so an
  admission below Tier 1 records the pass that admitted it as dated
  evidence; the release gate still reads only Tier 1 rows.
- **Qwen3-4B base and tool-use adapter recorded on the Metal laptop.** The
  bartowski Q4_K_M the published adapter runs on had no envelope, no smoke
  class and no recorded cell; it now has a compat row of its own (load,
  tokenizer against Qwen's tokenizer.json with 0 of 721 corpus strings
  differing, chat), Metal and CPU smoke cells with byte-identical replies,
  and `macos-arm64-metal` as its smoke class. The adapter's composed run
  (base plus `--lora`) is pinned and recorded on Metal, which closes its
  missing line; its reply is byte-identical to the CPU reply another host
  recorded with 1.5.1. Both rows stay Tier 2.

## v1.7.0 - 2026-10-28

- **A slow weight upload says so.** The CUDA backend copies the weights to the
  device straight from the model file's mapping, at whatever speed the storage
  faults the pages in; a cold 8 GB file on a throttled disk took longer than
  anyone waited, with nothing on stderr after the `gpu-split` lines, and was
  killed as a hang (reported as `--score` with `--lora` hanging; the adapter
  had not loaded yet). The copy now goes in 64 MB pieces and, once it has run
  for two seconds, a `gpu: weights ... on the device` line every five seconds
  gives the bytes placed, the rate from the file and the time left, then one
  closing line with the whole upload's rate. A warm load prints nothing new.
  `tests/test_upload_report.c` drives the policy with a synthetic clock;
  `tests/test_cuda_upload_report.py` holds every weight copy to the reported
  path.
- **Model validation on demand, with a paper trail.** The tiers stay; the
  age limit on their records goes. A Tier 1 file is no longer re-tested on
  a schedule or when a record turns four months old: validation runs
  happen in the bigger validation runs the project starts and on immediate
  need, each run writes its Runner version and date into the roster, and a
  new admission records the pass that admitted it. `docs/model-support.md`
  now opens with every validated file at a glance, Tier 1 and Tier 2 by
  family, each cell the version and date of the file's last run, and the
  release check (`scripts/model-tiers.py --gate`, formerly `--stale`)
  refuses only a Tier 1 row with no recorded pass on its backend or a
  recorded failure on any. No model was re-tested for this change; the
  marks are the runs already recorded.
- **One command line, one mode, shorter spellings.** The flags that were
  really verbs are refused in pairs instead of resolved by dispatch order
  (`-p hi --serve` used to start the server and drop the prompt; `--score
  --decide F` decided and never scored). Two groups moved to subcommands:
  `runner attest keygen|sign|check|lineage|bundle|check-bundle|pack|
  check-pack|sign-model|watermark-keygen|detect` holds the record, key,
  bundle and pack utilities, and `runner train DATA -m MODEL` the trainer
  with short option names (`--steps`, `--out`, `--dpo`, ...); both are
  rewrites of the flags behind them, so the records, exit codes and tests
  are unchanged. Merged spellings: `--json [FILE]`, `--think on|off|auto`,
  `--tray [off|auto]`, `--gpu auto|off|N`, `--reserve P` or
  `vram=P,ram=P,cpu=P`, `--rope linear=F,yarn=F,base=F`, `--draft
  PATH|mtp|mtp:FILE|lookup`, `--score [N]`, `--record PATH` (the transcript
  of a one-shot run, the receipts directory of a server; `--record-keep`,
  `--record-prev`), `--rewrite OUT` with `--merge-lora` or
  `--context-surgery` as the trigger (each pass keeps its own record kind),
  and `--info caps|tools|adapter|doctor|fit`. `--trust-key` also takes a
  file holding the key. Serve-mode defaults: a server queues for a held GPU
  (`--wait-for-vram 300`; `0` refuses at once) and yields its model on
  request (`--yield-on-request on`; `off` keeps it); `--port` and `--ttl`
  are refused without `--serve`. Every older spelling is still accepted and
  means the same (MANUAL.md, "Older spellings"); `runner --help` lists the
  current ones and its attestation and training sections moved to the two
  subcommands' `--help`. README, manual, site and guides use the current
  spellings; dated reports keep the ones they were run with.

- **Tier 1 follows pull.** The actively validated list held 27 files, a
  maintenance promise sized to the proof thesis rather than to use. It now
  holds the project's own published files whose repository shows at least
  1,000 downloads on the Hub in the last 30 days, read at each release and
  written into the row's transition: 8 files today (Xyntetik-Kvist-14B,
  the Qwen3.8-27B GSQ-RCO recovery with its MTP head, the Flash-Next pair).
  The other 19 move to Tier 2, maintained on report, with their records
  kept as dated evidence and their transition stating the count or that the
  file is its publisher's. Nothing stops loading, and no published file is
  withdrawn. The Clef IQ3_S recovery repository
  (`Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF`: the base and two
  adapters, below the house bar as its card states) is admitted at Tier 2,
  and on 2026-10-28 the repository's full decision model
  (`Clef-IQ3_S.clef.gguf`: the IQ3_S backbone with the publisher's BF16
  decision head) joins it there, with its decision figures as the anchor.

- **One rendered table, in docs/model-support.md.** The Tier 1 files at a
  glance (one block per family, then the Upcoming models) open
  [docs/model-support.md](docs/model-support.md), above every row's
  envelope; the README's "Actively validated models" section says what the
  release check is and links there instead of carrying the table. Each
  published file's card carries one sentence and a link per file to its row
  on that page, with no marks or dates, so it cannot drift; the
  `cards-drift` workflow and the `--cards`, `--cards-check` and
  `--cards-push` commands are gone. The site's Runner page links to the
  overview and to the README's section in place of its own copy of the
  table, and the landing page's model cards link to their Hub repositories.

- **The validation gates run on change, not on the calendar.** The device
  ledger and the model roster held their records to 30 or 45 days, which
  re-ran every class and every Tier 1 file on a schedule whether or not
  anything on its path had changed. Both now hold a record for 120 days as a
  backstop, with no per-class override, and a record is renewed earlier when
  a pull request changes a path it covers and names it. Releases are cut
  when a user-visible change exists, not on a schedule
  ([docs/versioning.md](docs/versioning.md)).

- **The release gate no longer depends on borrowed hardware.** Each device
  class in `docs/device-evidence.json` now says whose machine it is
  (`ownership`: owned or borrowed; absent means owned). A borrowed class
  (lent with no terms, usable while it is there, never depended on) is
  reported by both gates like any other, but a missing or old record on it
  never refuses a tag; a recorded failure on it still does, because a
  failure is a fact about the engine. The Linux CUDA host is marked
  borrowed. The overview in [docs/model-support.md](docs/model-support.md)
  shows a gate recorded on borrowed hardware with a superscript b, and the
  row names it. The Hugging Face cards do not change.

- **`--train-gpu`: the attention backward runs on the GPU, and every
  model's adapter changes bytes again.** The score and softmax adjoint of
  every attention layer ran on the CPU beside the device's products. Three
  kernels take it over; they recompute the attention probabilities in
  another order than the CPU, so a `--train-gpu` adapter changes (within
  rounding of a CPU run, deterministic). The CPU trainer's bytes are
  unchanged. On an RTX 3070 beside 4 threads, medians of three: two
  Qwen2.5-7B Q4_K_M steps of 512 tokens 15.9 s to 14.4 s, two Qwen3.5-4B
  steps 12.8 s to 12.4 s; the attention backward 0.22-0.31 s to 0.04 s a
  step. On the 26B Clef IQ3_S with 6 threads of a datacenter MIG slice,
  interleaved runs of two steps: a step 28.5 s to 25.7 s, the attention
  layers 3.2-3.5 s to 0.4 s a step, the same losses. The record counts the
  calls (`attention_device_calls`, `attention_failed_calls`).

- **`make ptx` has one pinned toolchain, and CI holds the committed PTX to
  it.** `tools/ptx-toolchain/make-ptx.sh` builds the toolchain from a conda
  lockfile (nvcc 13.0.88, crt 13.0.88, cccl 13.0.85, cudart-dev 13.0.96,
  gcc 15.2.0) and reproduces the committed PTX byte for byte; the
  `ptx-reproducible` workflow runs it on every kernel change and fails on
  any difference. The same nvcc with MSVC as the host compiler changes 58
  kernels, so PTX built any other way is not committed. A new test holds
  the CPU attention backward to a double-precision reference: the gradient
  checks could not see an error of a tenth of a percent there.

- **README: the actively validated models are listed by family.** One
  collapsed block per family (Gemma 4, Qwen3, Qwen3.8 and the others), in
  the roster's order, its line a plain overview per backend and for Train:
  ✅ when every file of its models is recorded, 🟨 *k of n* when some are,
  a hyphen when none is, n/a when every file has a declared limit there.
  Each model inside keeps its per-file table. For Train, recorded means the
  LoRA backward admits the file and a refusal counts as a declared limit.
  [docs/model-support.md](docs/model-support.md) groups the envelopes the
  same way, every Tier 1 and Tier 2 row of `docs/model-tiers.json` names
  its `family` (`--check` requires one), and the site's table follows the
  README's order. The Hugging Face cards do not change.

- **`--train-gpu`: the backward's transposed products run 2-3x faster, and
  every model's adapter changes bytes again.** They ran the CPU trainer's
  own chain (one serial sum an element, skipping zero gradients), which the
  CUDA assist (`RUNNER_TRAIN_GPU=1`) needs and `--train-gpu` does not.
  Device mode now takes the forward product's contract (four chains of
  16-row chunks, folded in order, then added to the incoming gradient), so
  a `--train-gpu` adapter changes (within rounding of a CPU run,
  deterministic and batch-invariant); the assist keeps the CPU's bytes.
  Kernel time against the chain at 64 and 512 columns: on a datacenter MIG
  slice 2.90x and 2.58x (IQ3_S), 2.78x and 2.61x (Q4_K), 2.53x and 2.31x
  (Q6_K), 2.56x and 2.21x (Q8_0); on an RTX 3070 2.1-2.7x. A 26B Clef IQ3_S
  step on the slice with 6 threads: 32.0 s to 27.8 s (two interleaved
  rounds on a quiet host), its transposed kernels 7.25 s to 3.22 s.
  `RUNNER_TRAIN_MVT_KERNEL=chain` runs the chain in device mode too, for
  comparison.

- **`--train-gpu`: every model's adapter is not 1.6.0's byte for byte, and
  the forward products run about twice as fast.** The forward's weight
  products (the taped forward, the recompute and the heads) move to a
  register-blocked kernel that sums each output in another order: four
  chains of 16-input chunks, each a serial fused multiply-add, folded in
  order. It stays within rounding of a CPU run, deterministic and
  batch-invariant (a column's value does not depend on the columns sharing
  its call), and the CPU trainer's bytes are unchanged. IQ3_S, Q4_K and
  Q6_K weights are decoded eight at a time. Kernel time at 64 and 512
  columns against the previous kernel: on a datacenter MIG slice 2.07x and
  2.74x (IQ3_S), 1.76x and 2.27x (Q4_K), 1.57x and 2.07x (Q6_K), 1.81x and
  2.30x (Q8_0); on an RTX 3070 2.05x and 2.99x, 1.87x and 2.82x, 1.70x and
  2.39x, 1.82x and 2.59x. A 26B Clef IQ3_S step on the slice with 6 threads:
  59.1 s to 50.4 s (two interleaved rounds), its forward kernels 15.4 s to
  6.0 s. `RUNNER_TRAIN_MM_KERNEL=warp` runs the previous kernels, for
  comparison.

- **`--train-gpu` on Qwen3.5 and Clef: the Gated DeltaNet recurrence runs
  on the GPU too, and these models' adapters are not 1.6.0's byte for
  byte.** The recurrence of the DeltaNet layers ran on the CPU beside the
  device's weight products, in all three places a step takes it: the
  forward, the backward's recompute and its sweep back through time. It now
  runs on the GPU through one step function, so the forward and the
  recompute still see the same states, bit for bit. Its sums run in another
  order than before, so a `--train-gpu` adapter on these models changes
  (within rounding of a CPU run, as before, and deterministic); on every
  other model it is byte-identical to 1.6.0's, and the CPU trainer's bytes
  are unchanged on every model. On an RTX 3070 beside 4 CPU threads, medians
  of three: six Qwen3.5-0.8B distillation steps 16.5 s to 14.4 s, two
  Qwen3.5-4B Q4_K_M steps of 512 tokens 16.1 s to 14.9 s. On the 26B Clef
  IQ3_S with 6 threads of a datacenter MIG slice the recurrence's part of a
  step fell from 2.9-5.6 s to 1.2-1.7 s; the whole step (50 to 65 s on that
  shared host) moved by less than the host's own noise. The record's
  `products` count the recurrence's calls on the device
  (`deltanet_device_calls`) and any that failed there and ran on the CPU
  (`deltanet_failed_calls`). A new test holds the CPU recurrence and its
  sweep to a double-precision reference (`tests/test_dn_recurrence.c`): the
  gradient gates could not see an error of a tenth of a percent there.

- **SHA-256 on the CPU's own instructions: a model file hashes about ten
  times faster.** The training, quantize and merge records, the download
  check, bundles and lineage all hash whole model files, and the portable
  code ran at about 0.2 GB/s on an Apple M1 and 0.5 GB/s on an x86 core:
  a 12 GB file took about a minute on the M1 and 23 s on the x86 core. Where the running CPU has SHA-256
  instructions (every arm64 Mac, ARMv8 Linux CPUs that report them, x86
  CPUs with SHA-NI) they are used, at 2.1 and 2.5 GB/s on the same
  machines; elsewhere the portable code runs as before. The digests are
  the same by definition, and `tests/test_sha256.c` checks every backend a
  host has against the FIPS 180-4 example messages and reference digests
  at each block-padding edge. The 12.4 GB Clef backbone file hashed in
  5-6 s instead of 23 s on an AMD Threadripper core; a one-step training
  run of Qwen3-0.6B Q8_0 on an Apple M1 went from 9.2 to 5.9 s, the
  difference the record's hash.

- **The CPU trainer: a step takes about a third to a half of the time, with
  the same bytes.** Most of a CPU training step was the backward's
  transposed weight products, and the largest was the LM head's: its
  16-position rounds were split into position chunks of 2, so each worker
  re-walked every vocabulary row for two positions, under 2 GFLOP/s. Chunks
  now keep at least 32 positions, rows are decoded a panel at a time, and
  the gradient is updated in register tiles down the panel instead of
  being reloaded and stored once per row. Every element keeps its
  summation order. On an Apple M1, Qwen3-0.6B Q8_0 over 256 positions: 76
  s to 25 s a step on one thread, 24-26 s to 9-11 s on four. Qwen3-4B
  Q4_K_M on 12 threads of a datacenter host: 27-32 s to 15-16 s.
  Qwen2.5-7B Q4_K_M on an i7-7700K's 4 threads: 97-101 s to 84-94 s; six
  Qwen3.5-0.8B distillation steps there: 73.5 s to 51.1 s. Adapters
  are byte-identical to before on every fixture family and objective, on
  those models, and against the per-element contract in
  `tests/test_mvt_chains.c` (all 25 weight types the trainer decodes).

- **`type-plan-size` reads `general.alignment` as the quantizer reads it.**
  The size estimator took any value under that key as the alignment (a
  boolean gave 0 or 1, a 64-bit value predicted a 4 GiB file) where Runner
  reads it as an unsigned 32-bit integer and wrote 288 bytes in each case.
  The estimator now keeps each header value's type and mirrors
  `gguf_get_u32`, refuses a zero or non-power-of-two alignment with the
  quantizer's message, and nine cases compare its prediction with Runner's
  actual output bytes.

- **`--bench-json` names CUDA's attention routes.** Its kernels object now
  counts the tensor-core prefill tiles (`cuda_attn_tc`, among
  `cuda_attn_tile`), the grouped decode launches (`cuda_attn_dec_gqa`) and
  the decoded tokens whose K and V came back in one copy
  (`cuda_kv_gather`), so a record says which route ran. The three were
  counted and never reported, until a compiler flagged them as set and
  never read; CPU and Metal runs report 0.

## v1.6.1 - 2026-10-25

**Linux compatibility hotfix.** The loose Linux binary restores the previous
GLIBC 2.38 ceiling and starts on Ubuntu 24.04. The 1.6.0 asset requires
GLIBC 2.43 through `sqrtf` and fails before startup on that system.
Upgrade to 1.6.1 on affected hosts.

**Build-sensitive outputs remain a limitation.** This patch keeps the 1.6.0
inference and training engine sources, apart from the version constant,
while returning Linux builds to Ubuntu 24.04 / GCC 13.3 with x86-64-v3.
Compiler and architecture flags independently change scores. Exact replay
stays bound to the binary. The published 1.5.0 binary's 20/23 and local Clang
native 1.6.0 build's 19/23 remain separate measurements; the published 1.6.0
and 1.6.1 Golden counts are unknown. All-row source equivalence remains
unproven. The five instrument rows and full 23-row asset pass remain queued.
[Golden evidence and build controls](docs/golden-pass-2026-10-25.md).

- Linux packaging pins Ubuntu 24.04, rejects imported GLIBC symbols above
  2.38, and records compiler, libc, hosted image and flags in BUILD-INFO.
- Six focused real-ELF tests pass. Linux CI verifies SHA256SUMS, extracted
  archive bytes, startup and fixture inference on Ubuntu 24.04 and a
  checksum-verified isolated GLIBC 2.38 runtime. The release build repeats
  the Linux ABI and runtime smokes before packaging.
- The pinned Gemma-4 E4B QAT Q4_0 file passes load and chat on the 1.6.1
  macOS CPU build. Its paging warning and slow timings are retained in the
  [new compatibility report](docs/compat-reports/1.6.1-2026-10-25-macos-arm64-gemma-4-e4b-it-qat-q4_0.json).
- The complete required PR CI matrix and local release-check pass. Device
  and Tier 1 records remain current; the existing 1.6.0 device evidence is
  retained with the same engine sources, rather than relabeled as a new
  CUDA measurement. Later main training optimizations are excluded.
- README, manual and site carry the compatibility fix and measured-build
  qualifications together. Model files and adapters are unchanged; their
  existing roster validation marks retain the versions actually measured.

## v1.6.0 - 2026-10-24

Numbered 1.6.0 under the [versioning policy](docs/versioning.md): a 1.x minor
release adds, and this one adds flags and record fields.

- **Windows: a pooled CPU result no longer depends on which thread computed
  it.** Windows only; Linux and macOS are unchanged, since their threads
  already inherited this state. Runner's 1.5.1 release and earlier have it.
  `-ffast-math` sets flush-to-zero and denormals-are-zero in the main
  thread's floating-point control register at startup; Linux and macOS
  threads start with their creator's state, Windows threads in the default
  one. So on Windows the thread pool's workers, the server's slot threads
  and the scheduler's decode thread kept subnormals the calling thread
  flushed, and a row computed on a worker could round differently from the
  same row on the calling thread (pooled LoRA rows, the batched prompt pass,
  a server slot against the CLI), or differ outright with extreme inputs
  (inf where the calling thread gave NaN). Every compute thread now installs
  its creator's control state when it starts (x86 MXCSR without its sticky
  exception flags, arm64 FPCR), and `tests/test_thread_default.c` requires
  each pool worker to compute in the state its pool was made in. CPU outputs
  on Windows can change where a subnormal reached a worker.

- **The x86 release binaries are gcc builds, so their adapter bytes are
  bound to that build and differ from a clang build's.** Measured on 14
  fixture families at `x86-64-v3`, the release target: clang 23 and gcc
  16.1 (the Windows release's toolchain) built from the same source trained
  identical adapters on 0 of 14, also with gcc's `-fno-associative-math`.
  Part of the gap is gcc's `-ffast-math` reassociating some of the
  trainer's explicit fused multiply-add chains into vector reductions: a
  normal gcc build and the same source built with `-fno-associative-math`
  agreed on 0 of 14. A gcc build is deterministic (the same binary twice:
  2 of 2 identical; the thread count does not change them), but its
  adapters can move when the code around a chain changes while the chain's
  own order in the source does not. At `-march=native` on a Zen-class CPU
  gcc reassociates more (the LoRA delta's chains as well, measured on a
  Zen 5). Replay is bound to the build
  ([versioning policy](docs/versioning.md)).

- **`--lora` on the CPU: the adapter's delta costs about a third of what it
  did, with the same bytes.** Each projection's y += scale * B (A x) ran its
  rank and output sums one chain at a time, every step waiting on the one
  before (2-3 GFLOP/s a thread). It now runs up to 8 of those chains side
  by side, each with its own order unchanged. On an Apple M1, 4 threads,
  Qwen3-0.6B Q8_0 with a rank-8 adapter: decode 37.0 to 42.6 tok/s (44.7 to
  47.0 without the adapter), prefill 149-154 to 164-173 tok/s. Training's
  forward passes take the same path. The CUDA and Metal backends apply
  adapters with their own kernels and are unchanged. Outputs are
  byte-identical to before: text and scoring with an adapter, every
  training fixture family and objective (ranks 1, 2, 4, 5, 8, 12, 16 and
  33), and `tests/test_lora_chains.c` against the old loop over 915 shapes.

- **Training: the adapter's own backward passes run 4 to 11 times faster,
  with the same bytes.** Every site's rank-r products (B^T dy, A x, the
  dA and dB updates, the adapter's share of dx) walked one strided chain at
  a time, about 3 GFLOP/s a thread. They now run several chains side by
  side over contiguous rows, with the accumulators held in registers. On
  an Apple M1, one thread, Qwen3-0.6B Q8_0 at rank 8 over 256 positions:
  1.96 s to 0.17 s a step. On the 26B Clef backbone with `--train-gpu` and
  6 threads of a datacenter MIG slice: 6.2-6.5 s to 1.3-1.5 s a step, and
  steady steps from 59-61 s to 53-55 s. Each element keeps its summation
  order in the source, so adapters are byte-identical to before in clang
  builds (on every fixture family and objective, at ranks 1, 5, 16 and 33
  beside the default 8, and on the 26B pair) and in a gcc build that keeps
  the source order (`-fno-associative-math`: 14 of 14 fixture families). A
  gcc release build's adapters do change: gcc reassociates some of those
  chains, and the new loops change where (gcc 16.1 at `x86-64-v3`, before
  against after: 0 of 14 identical).

- **`--train-gpu`: the device budget is an input you can name, and the
  record names it.** The weights are cached as long as the budget allows,
  and the rest train on the CPU, where a tensor's products take the CPU's
  order, so the adapter's bits depend on the split. The budget was the free
  memory at the start, which no flag named, and allocations could fail
  later in the run, so two runs of one command could write different
  adapters with records that looked the same.
  `--train-gpu-budget BYTES` now sets the budget (default: the free memory,
  less 512 MB). The budget is reserved on the device at the start, so a run
  gets it or stops with an error before training. `products` records
  `budget_bytes`, `budget_from` and the split (`tensors_on_device`,
  `device_bytes`, `tensors_on_cpu`, `cpu_bytes`). Passing a record's
  `budget_bytes` back reproduces its adapter.

- **Release gate: refuses to run in a degraded launch environment (system
  temp directory, no user name) and records the environment it ran in.** On
  the Windows CUDA host a tool shell started MSYS2 bash without `USERNAME`,
  `TEMP` or `TMP`: python's tempfile fell back to `C:\Windows\Temp`,
  `getpass.getuser()` raised, and the gate ran about 27 times slower than
  under a login shell, with two subprocess timeouts that were not hangs.
  `cuda-gate.sh` now asks its own python for the temp directory and the user
  before its first step and finishes not-run when the temp directory is
  under the Windows directory (on POSIX, `/`, not writable, or not the
  `TMPDIR` the environment names), there is no user name at all (`USERNAME`
  or `USER` empty and `getuser()` raising; an empty variable that
  `getuser()` still resolves, as in a Linux container without `USER`, is a
  warning), or an MSYS2 bash has no `MSYSTEM` (or no `HOME` outside a login
  shell); it names each check, the values it saw and the remedy (launch from
  a login shell, or set `TEMP`/`TMP` and `USERNAME`).
  `GATE_ALLOW_DEGRADED_ENV=1` turns the refusal into a warning. The summary
  JSON moves to `xyntetik.runner.cuda-gate.v2` and gains an `env` object:
  temp directory, `user_resolved` (whether a user name resolved, never the
  name itself, since the summary is committed as device evidence), shell,
  login shell, `MSYSTEM`, the checks that failed and the checks that only
  warned.

- **Release gate: cuda-gate.sh explains an empty GATE_MODEL.** On the
  Windows CUDA host variables exported in a Git Bash shell never reached the
  MSYS2 bash the gate ran under, and the gate said only "no model for the
  cpu-cuda step (set GATE_MODEL)". A gate that cannot run for want of a
  setting now lists every gap at once, then the env file it looked for and
  whether it was found and loaded, the settings it ended up with (a value
  ending in a carriage return from a CRLF env file is named), and the shell
  running it. `RUNNER_LLAMA_CPP_BIN` is checked with the others: moe-legacy
  cannot pass without it, so the gate now stops before its first step
  instead of failing there.

- **Release gate: the llama-quantize binary is resolved by name, not by
  glob.** A llama.cpp Windows release directory holds
  `llama-quantize-impl.dll` beside `llama-quantize.exe`, and under C
  collation the DLL sorted first, so the CUDA gate's `test-cuda-moe-legacy`
  step tried to execute it and failed. The four steps that quantize fixtures
  now run `llama-quantize.exe` or `llama-quantize` from
  `RUNNER_LLAMA_CPP_BIN`, the `.exe` first (under MSYS2 a bare name resolves
  to its `.exe`, which would otherwise be reported under the wrong name), and
  name the directory and both names when neither is there;
  `make llama-quantize-path` prints the one they run. In the three
  steps that narrow their coverage without the tool, a quantize that fails
  now prints a line naming the binary and the type instead of dropping the
  type silently.

- **Nemotron 3.5 Lightning tokenizes as its publisher's tokenizer, and an
  unknown pre-tokenizer name is no longer silent.** The file declares
  `tokenizer.ggml.pre` "pixtral", llama.cpp's name for the Tekken split
  Runner already implements. Runner had no row for that name and fell back
  to the GPT-2 split: `<p` became `<` + `p`, a 1000-byte README prompt took
  353 tokens against the publisher's 298, and 150 of the 721 corpus strings
  tokenized differently from NVIDIA's own tokenizer. Now 0 of 721, gated in
  the compatibility matrix against a pinned capture of that tokenizer.
  Generated text still read normally, which is why nothing flagged it. A BPE
  file whose pre-tokenizer name has no rule here still loads with the GPT-2
  split, but now prints one line at load (`tokenizer: pre 'NAME' unknown,
  GPT-2 split assumed ...`), and `--doctor` reports it as a `degraded`
  finding, `tokenizer_pre_unknown`. SentencePiece vocabularies, which do not
  use the name, and files without the key stay quiet.

- **Windows: non-ASCII prompts, file paths and environment values were read
  in the ANSI code page; they are UTF-8 now.** The C runtime built the
  command line in the system's legacy code page, so a `-p` prompt reached
  the tokenizer as cp1252 bytes on a Western European system, and as `?` for
  what cp1252 cannot spell, while the same text through `-f` arrived intact:
  on the Windows CUDA host a 17-character prompt mixing accented letters, a
  dash and two Japanese ideographs was 8 tokens through `-f` and 10
  different ones through `-p`, and the README's own prompt 822 tokens
  against 816 on macOS. `runner.exe` now carries an application manifest
  that makes UTF-8 the process code page on Windows 10 version 1903 and
  later; building it needs `windres` (binutils). `--caps` reports the code
  page. Where Windows ignores the manifest (older versions, or a stripped
  one) the result is the same: the command line and the environment are read
  in their UTF-16 form and every file is opened through the UTF-16 APIs. A
  console on stdout or stderr is switched to the UTF-8 output code page
  while runner runs and restored at exit, and a line typed into the
  interactive chat at a console is read as UTF-16 (it arrived in the
  console's input code page, 437 or 850, whatever the process's own). Two
  Windows defects on the same paths went with it: `--shadow-mode` passed a
  model path containing a space to the Python client as two arguments, and
  the desktop tray named every model over 2 GiB "model file missing",
  because MinGW's `stat()` fails past 2 GiB.

- **Seed 0 is accepted.** `seed: 0`, `-s 0`, `--fork-seed 0` and a
  session's `seed` or `fork_seed` of 0 now sample and replay like any other
  seed. They were refused because 0 was the sampler RNG's fixed point;
  seeds are mixed before use now and the mix is never 0.

- **Seeded sampling: the request seed is now mixed before use; with small
  seeds the first sampled token was always the most likely one.** Every
  seeded output changes once; determinism within a version is unchanged.
  The seed (a request's `seed`, `-s`, `--fork-seed`, a session's `seed` and
  `fork_seed`) was installed raw as the sampler's xorshift64 state, whose
  first draw from a small state is 0.0, so seeds 1 to about 1000 always took
  the top candidate first: a one-token classification sampled under
  `seed: i` was greedy for every `i`. Seeds now pass through splitmix64, the
  mix unseeded requests already used. A `--transcript` record's seed is the
  state the sampler started from and `--verify` installs it as it is, so
  records written before this change still verify.

- **`choice_logprobs` reports the mass the legal branches held.** Each
  decision point now carries `legal_mass`: the whole-vocabulary probability
  of its legal candidates before the posteriors were renormalized over them.
  The posteriors alone read the same whether the model wanted one of the
  schema's branches or none of them: on the Windows CUDA host Qwen3-4B, after
  writing `"E` under an enum holding `"E"` and `"EAT"`, gave its legal
  continuations about 1e-4 of its probability and the record showed a 0.84
  posterior with `coverage` 1.0. A low `legal_mass` marks a choice the schema
  forced. Added to the decision-record v1 contract
  ([docs/decision-record.md](docs/decision-record.md)). Both masses are now
  held at 1: float rounding in the shared log-sum-exp printed `coverage` up
  to about 1.4e-6 above it, against the contract's maximum. Nothing else in
  the record or in sampling changes.

- **Windows: `make` builds `runner.exe` in a bash started with a minimal
  environment.** The Makefile chose its Windows branch only from the `OS`
  environment variable, which a bash.exe started directly does not carry, so
  such a shell built a `runner` with no `.exe` and failed linking `-ldl`.
  The platform is now decided once from the shell's `uname` (`MINGW64_NT-...`,
  `MSYS_NT-...`, `CYGWIN_NT-...`, busybox-w32's `Windows_NT`) or
  `OS=Windows_NT`, and every platform branch reads that one decision; plain
  `make` replaces `make OS=Windows_NT` in the build notes and CI.

- **Inside a container, Runner sizes from the container's limits.** CPU and
  RAM came from the affinity mask, the physical page count and
  `/proc/meminfo`, none of which sees a cgroup's CPU quota or memory limit:
  a container held to 26 CPUs and 80 GiB on a 128-CPU, 268 GB host was sized
  as the whole host, from the default thread count and the batch default to
  `--fit` and the load-time RAM checks. On Linux these now take the cgroup v2
  `cpu.max` and `memory.max` (or the v1 equivalents) into account, at every
  level from the process's own cgroup up; available memory under a limit is
  the headroom plus the cgroup's reclaimable page cache, as `MemAvailable`
  counts it for the host. `--caps` keeps its keys, now carrying the effective
  numbers, and adds `host` (the machine's own) and `limits` (the source, the
  CPU quota and the memory limit). `--doctor` reports the same effective
  memory and adds an `info` finding, `container_limits`, with both sets of
  numbers; an `info` finding does not change the verdict. macOS and Windows
  are unchanged.

- **qwen35 and Clef: the DeltaNet q/k L2 norm takes the publisher's
  epsilon.** The Gated DeltaNet's query and key L2 norm used the model's RMS
  epsilon; the publisher's implementation hard-codes 1e-6 (transformers'
  `l2norm`). Every qwen35 and Clef file known ships `rms_norm_eps` 1e-6, so
  their outputs do not change. A file with another RMS epsilon read its
  query and key at a slightly wrong scale, on the CPU, in the training
  recompute and on CUDA alike; all three now use 1e-6. The test fixtures
  carry an RMS epsilon of 1e-5, so their DeltaNet outputs move, and the Clef
  fixture's pinned head scores were re-pinned.

- **Model support tiers.** [`docs/model-tiers.json`](docs/model-tiers.json)
  records which model files Runner re-checks on every release (Tier 1, 27
  files: the owner's roster and every published file), which are maintained
  on report (Tier 2), and which are planned (Upcoming, 12, listed by what
  each is missing); Tier 3 has no members. Each Tier 1 row carries its
  envelope, the device class its smoke runs on, a mark per backend (a dated
  gate, a declared limit, or not run) and the file's `--adapt-info` LoRA
  answer, all written by the gate run, never by hand.
  `scripts/tier-smoke.py` is the per-release smoke: the file hashed against
  its row, served and asked one user turn through its own chat template
  (temperature 0, `enable_thinking` false), the reply's bytes compared with a
  committed pin under `tests/compatibility/tier-smoke/` that always names the
  prompt's fact, a head or an adapter composed with the row it runs with.
  A mark is a reproduced greedy smoke on that class, not a quality or
  fidelity claim. `make release-check` now refuses a tag while a Tier 1
  file's smoke is missing, recorded on another class, failing, or older than
  45 days. All 27 are recorded for this release, every reply matching its
  pin: 25 on `linux-x86_64-cuda` (20 on its CUDA backend, 5 on its CPU
  backend), one on `windows-x86_64-cuda`, and the published gemma-4-E2B mix
  on CPU and Metal on `macos-arm64-metal`. They ran on main builds made
  between Runner's 1.5.1 release and this one, which still named their
  version 1.5.1, so the marks read 1.5.1. The Flash-Next MTP Q8_0 head
  carries a declared CUDA limit instead of a CUDA run (`--mtp` needs the
  whole trunk on the device, and the IQ3_S trunk keeps 18 of its 48 layers
  on a 24 GB device), so its smoke runs on the same class's CPU backend. The
  Train column is each file's `--adapt-info` answer: 21 of the 27 train an
  adapter, and the other six name why not (the gemma-4 dual-branch MoE FFN,
  Mamba-2, removed sublayers, a MoE router that is not softmax, and the
  hyper-connections of the two Flash-Next files). The README's "Actively
  validated models" section (one collapsed block per model, its line
  summing up each backend, a row per file inside) and
  [docs/model-support.md](docs/model-support.md) are rendered from the
  roster and held to it by a parity test. Tiers change
  no loading path and do not appear in `--caps`. Also fixed: the device
  ledger's release-check refusal printed its summary above the rows it
  summarises. A machine whose `--caps` matches no device class (a GPU
  hidden with an empty `CUDA_VISIBLE_DEVICES`, say) is refused with every
  class, the caps fields each expected otherwise, and how a GPU class's CPU
  cell is run.

- **The last-validated line on the model cards and the site.** Every
  published model file's Hugging Face card (13 repositories) now carries a
  `xyntetik-validated` block rendered from the roster: per roster file, its
  CPU, CUDA, Metal and Train marks with their version and date, or "not
  run", and a link to the file's row in
  [docs/model-support.md](docs/model-support.md#cards-and-site); the Tier 2
  gpt-oss-20b keep-30 file says its tier and reason instead. A mark is a
  reproduced greedy smoke at that version and date, not a quality claim.
  `scripts/model-tiers.py --cards OUTDIR` renders the cards for review,
  `--cards-push OUTDIR` uploads them (a dry run unless `--yes`, one commit
  per repository, refused when the card changed on the Hub outside the
  block) and `--cards-check` reports drift; the new `cards-drift` workflow
  runs the check on main and weekly, never on a pull request. xyntetik.com
  renders the same table on the Runner page and a one-line summary on the
  landing page's model cards at build time, and rebuilds when the roster
  changes. `site/shot.mjs` is the site's layout check, now in the
  repository.

- **Training: distillation toward a teacher (`--train-teacher DIR`).** With
  `--train`, each completion position is fitted to a teacher's top-k
  distribution instead of the next token: the KL divergence from the
  teacher's top-k plus one tail bucket to the student's, under the same
  prompt mask and weights. The use is bringing a quantized file back toward
  the model it came from, with an adapter on the frozen quantized base. DIR
  holds `topk_idx.npy` (int32) and `topk_logp.npy` (float32), each
  `[lines, positions, k]`, read a line at a time and checked in full before
  the first step; the `--train` file uses the id form.
  `scripts/kd-teacher.py` writes all three from a parent GGUF through
  `runner --score --score-top K`. On Qwen3-0.6B (Q4_0 student, Q8_0 teacher,
  48 steps) the held-out KL to the teacher fell from 0.2006 to 0.1346, where
  cross-entropy on the same tokens left it at 0.8357; a step costs 0.4%
  more. A one-hot teacher trains the cross-entropy's adapter byte for byte.
  The record's `objective` is `kl-to-teacher`, with both arrays hashed.
  It covers every model the trainer covers, qwen35 and Clef included (the
  next entry).

- **Training: qwen35 and Clef train an adapter (backward through Gated
  DeltaNet).** They served adapters and refused training. The backward now
  runs through each DeltaNet layer: backpropagation through time over every
  head's state (kept every square root of the window and rebuilt a stretch
  at a time on the way back, so memory grows with the square root of the
  window), the causal convolution, the L2-normed queries and keys, the decay
  and update gates and the gated output norm, with an adapter on all five
  DeltaNet projections; and through the full-attention layers' fused query
  and output-gate projection. It matches PyTorch autograd element by element
  on the qwen35 and Clef fixtures (every one of the 62 adapter gradient
  buffers within 1e-6 relative; `tests/test_deltanet_autograd.py`, run where
  torch is installed), and in `make test` the central-difference gate checks
  each projection slot on its own, which caught two dropped terms the
  whole-adapter check passed. `RUNNER_TRAIN_GPU=1` keeps the CPU's adapter
  bytes on CUDA. On Qwen3.5-0.8B against its Q8_0 file as the teacher
  (top-64, 24 documents, 48 steps), the held-out KL fell from 0.1269 to
  0.0914 with a Q4_0 student (requantized from the Q8_0) at `--lr 2e-4`, and
  from 0.0319 to 0.0257 with the Q4_K_M student at `--lr 5e-5`. At
  `--lr 2e-4` that student diverged (0.0447): it starts 0.03 nats from its
  teacher, and the trainer's AdamW moves each parameter by about the learning
  rate per step whatever the loss (no warmup or decay, one example per step),
  while its gradient on that model matches central differences projection by
  projection. The Mamba-2 hybrids, qwen4exp and MoE on a
  DeltaNet model still refuse training by name.

- **Training: partial-dimension RoPE trains; LayerNorm refuses by name.**
  The refusal of a rope that covers part of each head had no reason behind
  it (the backward already rotated the same dimensions the forward does) and
  is lifted, gated on fixtures rotating half of each head in both layouts.
  It had also been what kept StableLM out of training, and the backward has
  no LayerNorm adjoint, so `--adapt-info` and `--train` now refuse
  "LayerNorm (stablelm)" by name.

- **Training: a distillation pipeline's recipe in `--train`.** A learning-rate
  schedule (`--lr-warmup N`, then `--lr-schedule constant|cosine` down to
  `--lr-floor`), batches (`--train-batch N`: N rows a step, the loss and
  gradient the mean over their positions), `--weight-decay` (0 is plain Adam),
  checkpoints that carry the optimizer state (`--save-every` writes
  `OUT.optim` and keeps `OUT.step-NNNNN` copies) and `--train-resume CKPT`,
  which continues such a run byte for byte as if it had never stopped,
  held-out sets (`--train-eval LABEL=ROWS[,TEACHER]`, `--eval-every`: mean
  loss, `top1_topk`, `top1_vocab`), a kill rule (`--kill-at`, exit 3) and a
  stop file (`--stop-file`, save and exit 75). Each step line adds `lr` and
  the per-layer gradient norms (`grad_norm_max`, `grad_norm_mean`), and a step
  whose loss or gradient is not finite stops the run before it touches the
  adapter. A run with none of the new flags writes the same adapter bytes as
  before. The recipe is the one a distillation pipeline uses for quantization
  recovery, so such a run can move from PyTorch to Runner.

- **Training: a CPU step 1.7x faster, the same adapter bytes.** The
  forward a step trains through, its head and the backward's recompute ran
  one position at a time, decoding every weight once per position. The
  forward (and a `--train-eval` set's) now runs 16 positions at once, every
  product column computed in a single decode step's arithmetic; the
  recompute takes one product per weight over the window the same way; the
  head's gradient comes from the logits the forward already computed; the
  adapter's own rank-r math runs on every thread, and so does the attention
  backward (it was split by kv-head group, a few threads on a GQA model).
  Every gradient element keeps its accumulation order, so the adapter is the
  same byte for byte as the trainer before this change: on all 13 trainable
  fixture families with CE, weighted rows, the batch recipe and KD on an
  Apple M1 and on x86-64 under clang and gcc, and with DPO on Apple M1; at
  round widths 1, 5 and 16 (`RUNNER_TRAIN_ROWS=N`); and at `-t 1` and `-t 3`,
  which a new test now holds (the manual's claim had none). Qwen3.5-0.8B
  Q4_K_M with KD on an Apple M1, `-t 4`: a 134-token step from 15.2 s to
  8.9 s.

- **Training on the GPU (`--train-gpu`).** With `--train` or `--train-dpo`,
  every weight product of a step runs on an NVIDIA GPU: the forward, the
  heads, the backward's recompute and its transposed products, for every
  weight type the trainer decodes (25, the IQ and FP4 formats included), the
  weights uploaded once and cached for the run. The CPU keeps the norms,
  rope, attention, the DeltaNet recurrence, the adapter's rank-r math and
  the optimizer. On a 24 GB datacenter MIG slice with 6 CPU threads, six
  Qwen3.5-0.8B distillation steps took 10.3 s against 43.1 s for the CPU
  trainer on the same threads; on an RTX 3070 beside 4 CPU threads, a
  Qwen2.5-7B Q4_K_M step took 9.7 s against 95 s. The adapter is not the CPU
  run's byte for byte (the device adds in another order); it matches within
  rounding: Qwen3.5-0.8B's losses to 1e-6 and its held-out evals to six
  digits after 6 steps, on both hosts; Qwen2.5-7B's to 3e-5 relative. Its
  forward product gives a position the same value however the window is
  batched, which the backward needs, and the gradient gates run in this mode
  on CUDA hosts at a batching that would expose a product that depends on it
  (all 14 pass on a datacenter MIG slice and on an RTX 3070).
  The record names the device under `products`. Without a CUDA build and
  an NVIDIA GPU the flag is an error, not a CPU run. `RUNNER_TRAIN_GPU=1`,
  the same-bytes assist that moves only the backward's transposed products,
  now covers every weight type (it covered 7). `RUNNER_TRAIN_PROF=1` prints
  each phase's share of device time, the forward's included, and gives a
  DeltaNet layer's mixer sites their own column (`dn_sites`) instead of
  counting them as attention; `RUNNER_DEBUG_TIME=1` splits each forward
  stage's time (attention, DeltaNet or SSM, dense FFN and the rest) into
  device and host when a run used the device.

- **`scripts/kld-compare.py` and `kld-compare-raw.py` listen below the
  ephemeral port ranges.** Their default ports, 58601/58602 and 58611/58612,
  sat inside Linux's ephemeral range (32768-60999). A readiness poll sent
  before the server had bound its port could be given that same port as its
  own source port, connect to itself and read back its own request, which
  ended a run with `BadStatusLine`. The defaults are now 28601/28602 and
  28611/28612, below every ephemeral range, and the poll refuses a socket
  connected to itself. The conformance suite's server helper picks its ports
  from 20000-32767 for the same reason.

- **The manual and the site link the GENESIS genome record.** A research
  dataset with no weights, not a Runner model: the restarted GENESIS thesis,
  its prior-art review and its first kill test, GK-001, with every run. The
  existence record's description now covers every counted run from GE-001 to
  GE-014.

- **`scripts/decide-calibrate.py --rendering raw-v1|continuation-v1`.** The
  calibration instrument sent no `rendering`, so every run measured the
  server's default, `raw-v1`. The flag names the rendering in every request
  and records it in the result and the report's header; `continuation-v1`
  with an `--answer-prefix` is refused before any request, as the server
  would refuse each one. Without the flag the requests and the output are
  byte for byte what they were.

## v1.5.1 - 2026-10-23

- **Metal: Q4_K decode matvecs take the fast kernel by default on dense
  models.** The reassociating `k_mvf_q4_K` (two rows a simdgroup, vector
  accumulation), opt-in since the performance follow-up, is the default for
  the Q4_K weights of dense models on the owner's decision of 2026-10-23
  (MoE models keep the identity kernel: the fidelity rows are dense): decode +10.8% (Llama-3.2-1B) and
  +12.5% (Llama-3.2-3B Q4_K_M) on an Apple M1, -0.9% on Qwen2.5-1.5B; mean
  KL 0.9993x and 0.9998x of the identity route's against the publisher's
  float32 model, paired tests not distinguishable. Promoting it found that
  its microbatch columns did not equal solo decode (`test-batch-identity`
  failed on a Q4_K fixture: `--parallel` output would have depended on
  concurrency); the kernel now walks columns with the one-column arithmetic,
  so microbatch and speculative verify rows stay bit-identical, and `make
  test` runs both gates on a new Q4_K fixture. `RUNNER_METAL_MV=0` restores
  the identity kernel for every type (every byte comparison sets it);
  `RUNNER_METAL_MV=1` adds the fast q4_0 / q8_0 kernels, which stay opt-in.

- **gemma4: a forced tool call with a number argument is returned as a call,
  not as text.** The native grammar let a number start with a space
  (`[9, 0, 0]`, `scale: 1.5`) where booleans and nulls already could not,
  and format_argument never writes one; the call reader refused the spelling, so
  the whole call came back as `content` in gemma4's native framing with
  finish_reason "stop". gemma-4-E4B did this on 2 of 120 agent-torture
  cases under a named tool_choice, on 1.5.0 as well. Number values now
  refuse leading whitespace like the other bare values;
  `tests/test_tools.c` holds the spaced spellings unreachable and the
  reference spelling reachable.

- **CUDA: a MoE prompt tile routes its tokens in one launch.** Every MoE
  prefill route (fused, grouped and by-expert, and gemma-4's grouped and
  by-expert routes) ran the router as one matvec launch per prompt token.
  An F32 or F16 router, and a BF16 one off its split-K route, now runs through
  the batched twin of its one-token kernel, so each token's router logits are
  the same bits from one launch per MoE layer per tile; other router types
  keep the per-token launches, and gemma-4's fused per-token path is
  unchanged. Chunked `--score` is byte-identical before and after on
  Qwen3-30B-A3B, gpt-oss-20b and gemma-4-26B-A4B across every MoE route, and
  an 875-token Qwen3-30B-A3B prefill makes about 42,000 fewer launches.
  Prefill on the datacenter MIG slice at about 600 and 2,600 tokens (5
  interleaved rounds): Qwen3-30B-A3B +7.4% and +6.0%, gemma-4-26B-A4B +2.5%
  and +2.3%, gpt-oss-20b +2.0% at both; decode unchanged.

- **CUDA: gemma-4 files whose global layers reuse K as V decode through the
  CUDA graph.** Such a layer builds V from K by zeroing it and adding K, and
  the zeroing was a synchronous memset, which a graph capture cannot record,
  so these files (gemma-4-12B and gemma-4-26B-A4B among them) decoded with
  plain launches, and their decode microbatch did too. The zeroing now runs
  in stream order (`cuMemsetD8Async`) before the same add, so V is the same
  bits and both graphs capture; a driver without that call keeps the
  previous path. Greedy output is byte-identical with the graph on and off on
  gemma-4-12B and gemma-4-26B-A4B. Decode on the datacenter MIG slice (5
  interleaved rounds): gemma-4-12B +1.2% at about 700 and 2,800 prompt
  tokens, gemma-4-26B-A4B +0.2% and +1.6%.

- **CUDA: a Mamba-2 prompt tile gates and normalizes its tokens in one
  launch.** The tile path ran the convolution and the scan once over the
  tile but the gate and group norm once per token, a launch per token in
  every Mamba layer. One launch with a grid row per token runs the same
  arithmetic for each token: greedy text, `--score` and chunked `--score` are
  byte-identical on granite-4.0-h-micro, granite-4.0-h-tiny and
  Nemotron-Nano-9B-v2. Prefill on the datacenter MIG slice (5 interleaved
  rounds): granite-4.0-h-micro +135% at about 600 tokens and +145% at 2,500,
  Nemotron-Nano-9B-v2 +17% and +19%.

- **CUDA: the dense Mamba-2 hybrids decode through the CUDA graph.**
  granitehybrid and nemotron_h files without experts (granite-4.0-h-micro,
  Nemotron-Nano-9B-v2) decoded with plain launches: the graph plan admitted
  qwen35 alone among the recurrent families. Their one-token step advances
  the conv ring and the scan state in place on buffers held for the
  instance, with the rollback copy taken before the forward, the ownership
  qwen35's graph relies on, and nemotron_h's MLP-only blocks run no
  attention; both now replay. Greedy output is byte-identical with the graph
  on and off; `RUNNER_CUDA_GRAPH_RECUR=0` keeps every recurrent family eager.
  Decode on the datacenter MIG slice (5 interleaved rounds):
  granite-4.0-h-micro +1.2%, Nemotron-Nano-9B-v2 +0.4%.

- **Metal: Q6_K decode matvecs compute two rows a simdgroup.** A Q4_K_M file
  keeps some tensors in Q6_K (the output head and part of the V and FFN-down
  tensors), and their one-column matvec computed one row a simdgroup, each
  row loading the same activations again. `k_mv2_q6_K` computes two rows from
  one activation load a lane, with each row's multiply-adds written as the
  `fma` the one-row kernel compiles to, so the outputs are the same bits. The
  engine checks that at load against the one-row kernel on the running device
  and keeps the one-row kernel where any bit differs; `RUNNER_METAL_MV2=0`
  keeps it everywhere. Greedy text and token-by-token `--score` are
  byte-identical with it on and off on Llama-3.2-1B, Llama-3.2-3B and
  Qwen2.5-1.5B Q4_K_M. Decode on an Apple M1 (5 interleaved rounds, 512-token
  prompt, 128 tokens): Llama-3.2-1B +8.6%, Llama-3.2-3B +16.1%, Qwen2.5-1.5B
  +3.4%; prefill unchanged.

- **Metal: the opt-in fast decode matvec covers Q4_K** (made the Q4_K default on
  dense models by the entry at the top of this section). `RUNNER_METAL_MV=1`
  (tolerance class, off by default) had kernels for Q4_0 and Q8_0 only; a
  Q4_K kernel now computes two rows a simdgroup with vector accumulation.
  Over the new default on the M1: Llama-3.2-1B +10.8%, Llama-3.2-3B +12.5%,
  Qwen2.5-1.5B -0.9%. `./test-mv-tol` now runs on Q4_K files (it skipped
  them before) and passes on all three: no top-1 change in 64
  teacher-forced positions, mean logit change 1e-5 of the logit range. It
  stays off by default because it does not reproduce the identity kernel's
  bits.

- **CUDA: K and V reuse the activations TC2 staged for Q.** The pipelined
  tensor-core GEMM (TC2) prepares a prompt tile's input before every GEMM:
  each column's largest magnitude, then fp16 copies scaled by it. K and V
  read the same columns Q just prepared (so does qwen35's output gate), as
  do up after gate and a grouped MoE up after its gate, and prepared them
  again. Such a GEMM now reuses the preparation when nothing has written
  those columns since, which the GEMM checks by address, shape and that the
  preparation came from the call right before it. The prepared values are
  the same, so chunked `--score` is byte-identical with the reuse on and off
  on Qwen2.5-7B, Llama-3.2-3B, Qwen3-30B-A3B and gemma-4-26B-A4B (grouped),
  with 4,452 to 11,902 reuses a run (Llama-3.2-3B and Qwen2.5-7B also on an
  RTX 3070), and `make test-cuda-tc2-bytes` now checks it.
  `RUNNER_CUDA_XH_REUSE=0` prepares every GEMM's input. Prefill on the
  datacenter MIG slice (5 interleaved rounds, about 600 and 2,600 tokens):
  Llama-3.2-3B +1.7% and +1.3%, Qwen2.5-7B +1.2% and +1.0%, Qwen3-30B-A3B
  +2.6% and +2.1% (best of five: that file's runs scattered under host
  load); decode unchanged.

- **CUDA: a Mamba-2 hybrid's scan state is double-buffered instead of
  copied before every forward.** The engine copied every layer's SSD scan
  state to a rollback snapshot before each forward (4.8% of
  granite-4.0-h-micro's and 5.5% of Nemotron-Nano-9B-v2's decode time on the
  datacenter MIG slice). The state is now two halves: a forward's first tile
  reads the half the previous forward wrote and writes the other, a rollback
  returns to the half the forward read, and the decode graph keeps one
  capture per half. The scan's arithmetic is unchanged: greedy text with the
  graph on and off, a 2,600-token prompt, and token-by-token and chunked
  `--score` are byte-identical before and after on granite-4.0-h-micro,
  granite-4.0-h-tiny and Nemotron-Nano-9B-v2 (granite-4.0-h-micro also on an
  RTX 3070). Decode on the slice (5 interleaved rounds): granite-4.0-h-micro
  +6.9%, Nemotron-Nano-9B-v2 +5.7%; prefill unchanged.

- **CUDA: two attention opt-ins for layers the default leaves alone.**
  `RUNNER_CUDA_ATTN_TILE512=1` lets the tiled prefill attention take head
  widths 257-512 (`k_attn_tile8`, 8 query rows of 8-key tiles), so gemma-4's
  global layers stop falling back to the one-row `k_attn`;
  `RUNNER_CUDA_ATTN_TC_FULL=1` puts the tensor-core attention on the layers
  without a window in a family the default keeps on the tile (gpt-oss's
  full-attention layers). Both are tolerance class and off by default until
  their long-context rows against the float32 reference pass. 8,191-token
  prefill on an RTX 3070, 3 rounds: gemma-4-E4B +41.8% with the first; phi3
  +135.6% under the existing `RUNNER_CUDA_ATTN_TC=1`.

- **CUDA: the tensor-core prefill GEMM reads Q6_K weights eight bytes at a
  time.** A Q6_K block is 210 bytes, so its rows are only 2-byte aligned,
  and the pipelined tensor-core GEMM (TC2) read each segment's low and high
  bits as 48 two-byte loads; a profile on an RTX 3070 showed 4.1 of every
  32 bytes a sector used. It now reads them as 14 aligned 8-byte loads and
  shifts the same words into place, so the dequantized values, the
  tensor-core operands and every output are unchanged: greedy text and
  token-by-token and chunked `--score` are byte-identical on Q4_K_M, Q5_K_M
  and Q6_K files on an RTX 3070. Prefill there, 10 interleaved rounds
  (medians):
  Qwen3-4B Q6_K +12.6%, Qwen2.5-7B Q4_K_M +4.4%, Llama-3.2-3B Q4_K_M +4.1%;
  decode unchanged.

- **CPU: a prompt matvec serves each activation chunk to more weight rows.**
  The batched CPU matvec dequantized four weight rows at a time and ran
  them against the prompt's activations in 16-column chunks, so each chunk
  was fetched from cache again for every four rows. Rows now go in groups
  of sixteen on AVX2 (four 4 x 3 register tiles against each chunk; the
  4 x 4 tile kept five to eight of its sixteen accumulators in memory at
  x86-64-v3) and eight on NEON (one 8 x 3 tile), then smaller groups for a
  range's remainder. Every output is the same dot in the same order, so
  chunked and token-by-token `--score` and greedy text are byte-identical.
  512-token CPU prefill, 5 interleaved rounds: Llama-3.2-3B Q4_K_M +26.0% and
  Qwen2.5-7B Q4_K_M +40.7% on 12 Zen 5 cores at x86-64-v3, the release
  build (medians; the 4 x 3 tile alone +15.8% and +7.3%); Llama-3.2-3B +5.8%
  and Qwen2.5-3B Q4_K_M +4.1% on an i7-7700K; Llama-3.2-3B +6.1%,
  Qwen2.5-1.5B Q4_K_M +10.6% and Q8_0 +10.1% on an Apple M1.

- **CPU: the AVX2 multi-column dots keep their accumulators in registers.**
  The 2- to 7-column K-quant and codebook dots that speculative verify
  rounds and the CPU decode microbatch run kept one accumulator per column
  in an array indexed by the column count, a runtime value, so the compiler
  kept the array in memory. Each count from two to seven now has its own
  instantiation with the same per-column operations, and the outputs are
  unchanged. Q4_K dot on an i7-7700K: +12 to +14% at two columns, +27% at
  four; `--draft-lookup` decode +3.9% on Llama-3.2-3B Q4_K_M, with the same
  text as plain decoding. NEON measured no gain and keeps its form.

- **CPU: on NEON a prompt matvec walks 32 activation columns a pass.** With
  the row groups above, an Apple M1 read prefill +2.5% (Llama-3.2-3B
  Q4_K_M), +3.5% (Qwen2.5-1.5B Q4_K_M) and +4.2% (Q8_0) with 32-column
  chunks over 16 (5 interleaved rounds, medians), byte-identical; on x86
  (Zen 5, i7-7700K) 8 and 32 columns read within noise of 16, which stays.

- **CUDA: a speculative round on a Mamba-2 hybrid restores its kept rows
  instead of running them again.** With `--draft-lookup` or a draft model on
  a granite-4.0-h or Nemotron-H hybrid, a round whose drafts were partly
  rejected rolled the recurrent state back to the round's start and ran the
  kept tokens through the model a second time. A verify round now files the
  state after each of its rows (the tile scan writes them as it goes; the
  last row's state is the live one), and a partial round restores the row
  it kept, as Qwen3.5 models already did. The slots are allocated by the
  first verify round (about 0.5 GB on granite-4.0-h-micro, 1 GB on
  Nemotron-Nano-9B-v2), so plain decoding holds none, and without room a
  round re-runs its kept tokens as before. Output is unchanged: greedy text
  with and without lookup is byte-identical on three hybrids, and every
  verify row is byte-identical to solo steps in
  `make test-cuda-spec-identity`. Lookup decode on the datacenter MIG slice
  (5 interleaved rounds, medians): granite-4.0-h-micro +8.1% on a
  repetitive prompt (now 7.8% faster than plain decoding, from 0.3% slower)
  and +4.7% on an open one, Nemotron-Nano-9B-v2 +5.0% and +0.8%; plain
  decoding unchanged.

- **CUDA: three prefill attention routes become defaults.** The
  tensor-core prefill attention now also covers the qwen3moe and phi3
  families (the phi3 files that publish a window longer than their context,
  Phi-3.5-mini and Phi-4-mini, load without one), and the tiled prefill
  attention takes gemma-4's 512-wide global layers (`k_attn_tile8`) instead
  of the plain kernel. Each passed the long-context rule against the route
  it replaces (2,000 positions at 4,000 to 6,000 tokens of context against
  the publisher's float32 weights): mean KL 0.07584 vs 0.07580 on
  Qwen3-30B-A3B, 0.07912 vs 0.07912 on Phi-3.5-mini, 0.13998 vs 0.13996 on
  Phi-4-mini, 0.31995 vs 0.32108 on gemma-4-26B-A4B and 0.14722 vs 0.14714
  on gemma-4-E2B, with no paired test separating them. Prefill on the
  datacenter MIG slice at about 700 and 3,500 tokens (4 interleaved rounds,
  medians): Phi-3.5-mini +21% and +78%, Phi-4-mini +31% and +121%,
  Qwen3-30B-A3B +13% and +60%, gemma-4-E4B +8% and +26%, gemma-4-12B +6%
  and +21%, gemma-4-26B-A4B +1.5% and +6%; decode unchanged.
  `RUNNER_CUDA_ATTN_TC=0` and `RUNNER_CUDA_ATTN_TILE512=0` restore the
  previous routes. gpt-oss keeps the tile: its full-attention layers on the
  tensor cores (`RUNNER_CUDA_ATTN_TC_FULL=1`, still opt-in) read further
  from the reference (0.04065 vs 0.04039, Wilcoxon p 3e-18).

- **`-f FILE` reads the prompt file as written.** The file's text was
  appended to the `-p` text before `-p`'s escapes (`\n`, `\t`, `\r`, `\\`)
  were applied to the whole prompt, so a backslash sequence in a prompt file
  was rewritten: a scored corpus's `\times` reached the tokenizer as a tab
  and `imes`, and code or LaTeX in a one-shot or `--bench` prompt file
  changed the same way. `-p` text still takes the escapes; a file's bytes
  are now the prompt. `--score` comparisons against a reference tokenizer
  (`scripts/gold-logits.py`) refused such files rather than scoring them,
  which is how this was found.

- **CPU: a native AVX-512 build takes the 4 x 4 register tile.** AVX-512VL
  doubles the vector registers to 32, enough for the 4 x 4 tile's 21 live
  values, so a build for such a host (`-march=native` on Zen 4 / Zen 5 or a
  recent Xeon) runs its prompt matvecs in 4 x 4 tiles; the x86-64-v3 release
  build keeps the 4 x 3 tile. Each output is the same dot in the same order:
  chunked `--score` is byte-identical on Llama-3.2-3B Q4_K_M and Qwen3-8B
  Q8_0. 512-token prefill on 12 Zen 5 cores, native builds, 7 interleaved
  rounds (medians): +4.5% (Llama-3.2-3B Q4_K_M), +8.1% (Qwen2.5-7B Q4_K_M)
  and +8.9% (Qwen3-8B Q8_0) over the 4 x 3 tile, A/A within 1.6%.

- **`--pull owner/repo[:TAG]` downloads a model without loading it.** It
  fetches into the cache exactly what `-hf` would (same selection, cache,
  `HF_TOKEN`, multipart handling and verification), prints the path to give
  `-m` and the total size, and exits; `--pull ... --json` prints one JSON
  object instead, with each file's path, size, local SHA-256 and the Hub
  record that verified it (`hub_sha256` or `hub_size`). Where `-hf` loads a
  file the Hub lists with neither a digest nor a size, `--pull` refuses it.
  It is a mode of its own: any other option beside it is refused by name.
  `tests/test_pull.py` pins it against a fake hub, every assertion anchored
  on the fixtures' own sizes and digests.

- **Windows: `-hf` reuses a cached model larger than 2 GiB.** The cache
  sized files with `stat()`, whose size MinGW keeps in 32 bits and which
  fails past 2 GiB, as every real checkpoint is: such a file was never seen
  as cached, so it was fetched again and the rename over the file already
  there failed, and a file the Hub sizes without a digest read as
  "downloaded 0 bytes". Sizes now come from `_stat64` on Windows, as the
  model loader's file identity does; `tests/test_pull.py` holds a 2 GiB
  file's size, digest and reuse by `--pull` and `-hf`.

## v1.5.0 - 2026-10-22

- **`scripts/cuda-nightly.sh` sees a held GPU lock however MSYS2's bash was
  started.** It checks a lock's Windows pid with `tasklist`, found only
  through `PATH`; a bash started from a minimal environment has no Windows
  directory there, fell back to `kill -0`, which cannot see a Windows pid,
  and read a held lock as free, so the guest job could start over another
  user's run. It now reaches `tasklist` by its own path under MSYS2.

- **`scripts/compat_matrix.py`: a report made on Windows names its evidence
  files as one made anywhere else does.** Rewriting a machine path relative
  to its root kept the host's separator below it, so a ledger written on
  Windows recorded `cpu_cuda\m.json`, the same evidence under another name,
  and its own per-row reference did not open. The part below the root is now
  written with forward slashes.

- **CUDA: MoE experts in Q5_0, Q4_1 and Q5_1 take the fused device
  routing.** The fused MoE path had no indirect expert kernel for the
  32-element legacy types, so a model with any such expert tensor ran every
  MoE layer on the eager path (host routing, no decode graph).
  llama-quantize's Q4_K_M of gemma-4-26B-A4B keeps 17% of its experts in
  Q5_0, so a file quantized that way did. `k_moe_mv_q5_0`, `_q4_1` and
  `_q5_1` are the bodies of the types' batch-1 kernels; `make
  test-cuda-moe-legacy` (in the CUDA gate) holds each to the eager path with
  `test-moe-tol` and fails if the fused path does not engage. On the
  datacenter MIG slice, gemma-4-26B-A4B Q4_K_M reads a 2,048-token prompt at
  175 tok/s instead of 139 and decodes at 69.6 tok/s instead of 63.1.

- **CUDA: tensor-core prefill for Q5_K, Q3_K, Q2_K, Q4_1, Q5_0 and Q5_1.**
  These six types had no tensor-core kernel, so a prompt ran their weights
  through the scalar GEMM (Q5_K, Q3_K) or the matvec a column at a time
  (Q4_1, Q5_0, Q5_1, Q2_K): every Q5_K_M and Q3_K_M file, and the Q4_1 and
  Q5_K tensors that Q4_0 exports carry, prefilled far slower than the types
  beside them. They now have the TC route's kernels and TC2's (compute_75
  and compute_80), over decoders shared with a host test that holds each one
  to the CPU dequantizer bit for bit (`test-kq-stage`), and `make
  test-cuda-tc-kq` (in the CUDA gate) holds each type to the tensor-core
  tolerance gate on all three kernel families. They are the default on every
  family of the tensor-core list except the experimental qwen4exp: on 2,000
  positions of Qwen3-4B quantized to each type, and of Llama-3.2-3B at
  Q5_K_M and Q3_K_M, the tensor-core route read no further from the
  publisher's float32 model than the scalar one (closer on Q5_K_M, Q4_1 and
  Q5_1). Qwen3-4B on the datacenter MIG slice at 2,048 tokens: Q5_K_M 234 ->
  985 tok/s, Q3_K_M 282 -> 1,231-1,269, Q2_K 41 -> 1,118-1,281, Q4_1 22 ->
  1,235, Q5_0 22 -> 1,231-1,268, Q5_1 19 -> 1,033-1,060 (two rounds).
  Qwen3.8-27B Q4_0, whose Q4_1 and Q5_K tensors held its whole prompt back,
  prefills 2,048 tokens at 326 tok/s where it ran 60, and the 6,406-token
  tool call that timed out at a client's 90 seconds takes 27 (116 before).

- **CUDA: the one-token Gated DeltaNet step spreads each head over four
  blocks.** A qwen35 decode step and each checkpointed row of a verify round
  now run `k_q35_delta1`, which gives each block 32 of a head's value
  columns (four times the blocks of the one-block-a-head kernels, which kept
  32 of an RTX 3070's 46 SMs busy), and `k_q35_delta_norm`, the head's
  output norm in its own launch; each block repeats the head's q and k
  scales with the same reduction, so the output bytes are unchanged. The
  step went from 54 to 25 microseconds a call; Qwen3.5-4B decodes at 93.7
  tok/s where it ran 88.7, and `--mtp` at 85.0 where it ran 79.9.

- **CUDA: one-token Gated DeltaNet steps take the whole-tile kernels.** A
  qwen35 decode step and each checkpointed row of a speculative verify round
  ran the per-token kernel pair; they now run the whole-tile kernels over
  one token, which keep a thread's state row in registers and compute the
  same arithmetic (the identity `test-gpu-identity GPU_ID_RECUR` holds).
  Output bytes are unchanged; on an RTX 3070 Qwen3.5-4B decodes at 88.7
  tok/s where it ran 87.2, and `--mtp` at 79.6 where it ran 77.9.
  `RUNNER_CUDA_RECUR_SEQ=0` keeps every step on the per-token pair.

- **CUDA: the one-token Gated DeltaNet step reads its state coalesced.**
  `k_q35_delta`, the per-token step of the qwen35 recurrent layers (Qwen3.5,
  Qwen3.8), had each thread walk its own state row in device memory, with
  rows 512 bytes apart, so every access of a warp touched 32 cache lines.
  The rows now pass through a shared tile 32 keys at a time, loaded and
  stored by the whole block in order; every element takes the same
  operations in the same order, so the output bytes are unchanged. On an RTX
  3070 the step went from 82 to 59 microseconds a call and Qwen3.5-4B
  decodes at 87.2 tok/s where it ran 83.3.

- **CUDA: the tensor-core prefill attention is the default for the llama,
  qwen2, qwen3 and granite families.** `attn-tc` (`k_attn_tc` with Q and P
  carried as two fp16 halves, f16 caches) prefills an 8,000-token prompt
  2.5-3.2x faster than the tiled kernel and a 2,048-token one 1.5-1.6x:
  Llama-3.2-3B 620 -> 1,660 tok/s and Qwen2.5-7B 460 -> 1,165 at 8,192
  tokens on an RTX 3070. On each family's row of 2,000 positions between
  4,000 and 6,000 tokens it measured no further from the publisher's float32
  model than the tiled kernel it replaces (Llama-3.2-3B, Mistral-7B,
  Qwen2.5-3B, Qwen3-4B, granite-4.2-8b), and it does not touch the decode
  step. gpt-oss (attention sinks: its row read further, Wilcoxon p 0.017)
  and every sliding-window family keep the tiled kernel, as does a quantized
  KV cache. `RUNNER_CUDA_ATTN_TC=1` or `0` forces it on or off for every
  model; records name the route as before, and `--verify` replays a record
  with the route it names.

- **CUDA: qwen35 models decode through a captured graph.** The qwen35
  architecture (Qwen3.5 and Qwen3.8, with Gated DeltaNet layers) decoded
  every token with eager launches since it reached CUDA: the recurrent
  layers kept the graph off, with no hazard on record. A Gated DeltaNet step
  is two launches on buffers fixed for the instance, and the rollback copy
  is taken before the forward, so the one-token decode now replays the
  captured graph like the other families (`RUNNER_CUDA_GRAPH_RECUR=0` keeps
  it eager). A speculative keep-forward, whose per-row checkpoints the graph
  does not hold, still runs eagerly. `make test-cuda-spec-identity` compares
  the graph-decoded steps of its qwen35 fixture with the eager verify rows
  byte for byte. On an RTX 3070 Qwen3.5-4B decodes at 83.8 tok/s where it
  ran 79.1, with the greedy text identical to the eager build.

- **CUDA: a wide prompt runs a small tile matvec as one launch.** The tile
  matvecs a prompt takes for weights without a GEMM (the F32 and BF16 Gated
  DeltaNet alpha/beta projections of the qwen35 family among them) ran one
  launch per 16 prompt columns, eight blocks each for a 32-row matrix, so a
  2,048-token prompt issued 128 nearly empty launches per projection; they
  were 15% of a Qwen3.5-4B prefill on an RTX 3070. A wide prompt now runs as
  one launch with a column tile per grid row, the same arithmetic per
  element (scores byte-identical): Qwen3.5-4B prefills at 1,060, 976 and 710
  tok/s at 512, 2,048 and 8,192 tokens where it ran 888, 839 and 633.

- **`test-tc-tol` says how close a free-running divergence was.** When the
  tensor-core and scalar prefill paths' greedy outputs part, the gate still
  fails, and now prints how far the scalar pick led the other at that step
  as a fraction of the logit range, so a dead tie reads as one without
  rerunning both paths.

- **Metal: `--parallel` slots decoding together stay bit-identical to
  sequential decoding past 128 tokens.** A Metal decode step's attention
  switches to its chunked kernels once a layer's span passes 128 positions,
  but the decode microbatch ran the single-pass kernel for every column, so
  past that point a slot's step differed from its sequential step in the
  last bits (logit 9.64625 against 9.64631 at position 303 on Qwen2.5-1.5B),
  against the microbatch's documented contract. The microbatch's columns now
  take the decode step's own attention selection, and `test-batch-identity`
  (in `make test` and the CUDA gate) decodes behind a 300-token prefix as
  well as from short prompts; it never reached position 20 before, which is
  how the break stayed hidden.

- **CUDA: batched decode reaches the codebook i-quants and Q3_K.** The
  decode microbatch (`--parallel` slots decoding together) admitted a weight
  type only beside a coalesced single-step GEMV, so files in IQ2_XXS,
  IQ2_XS, IQ2_S, IQ3_XXS or IQ3_S decoded their slots one at a time,
  although the width-classed twins of those types' single-step kernels were
  there and a speculative verify round already ran them. Q3_K had no twin at
  all: a note from 2026-08-20 declined it until a single-step kernel in the
  GEMV geometry existed, and `k_gemv_q3_K` has been that kernel since
  qwen4exp; so a Q3_K_M file decoded its slots one at a time and ran a
  verify round's Q3_K rows a column at a time. The microbatch now asks what
  the verify rows ask (is the tensor on the single-step route the twins
  reproduce: no per-tensor scale, not the experimental int8-activation
  route, which also closes a latent mismatch for IQ4_XS under
  `RUNNER_CUDA_Q8X=1`), and `k_gemvb_q3_K` mirrors `k_gemv_q3_K`. On an RTX
  3070 (Qwen3-4B, every slot's text equal to sequential decoding) four and
  eight slots decode at 134 and 170 tok/s together where they ran 78 one at
  a time at Q3_K_M, and 114 and 130 against 75 at IQ3_S; prompt lookup on a
  Q3_K_M file decodes at 83-90 tok/s where it ran 71 (plain 79-81). `make
  test-cuda-batch-identity` runs all five codebook types from new
  random-block fixtures (`scripts/make-test-model.py --quant iq2_xxs` and
  the others, which need no quantizer; llama-quantize writes IQ2 only from
  an importance matrix) and a Q3_K fixture, and `make
  test-cuda-spec-identity` adds Q3_K, IQ2_XXS and IQ3_S.

- **Metal: tiled prefill GEMMs for Q5_K, Q4_1, Q5_0 and Q5_1.** Metal's
  tiled GEMM covered every common type but these four, so a Q5_K_M file (and
  the Q4_1 / Q5 legacy tensors some exports carry) prefilled through the
  matvec, which reads every weight once per token column. `k_mm_q5_K`,
  `k_mm_q4_1`, `k_mm_q5_0` and `k_mm_q5_1` decode each value as the CPU
  dequantizer does, and `test-metal-kquants` sweeps them with the other
  types. On an Apple M1 (Qwen3-0.6B quantized from the publisher's weights)
  prompt processing runs at 520 tok/s where it ran 97-102 at Q5_K_M (512
  tokens; 291 against 87-89 at 2K), and at 377-384 where it ran 97-106 at
  Q4_1, Q5_0 and Q5_1 (243-249 against 90-92 at 2K). Against the publisher's
  float32 model over 2,000 positions each type reads as the matvec route did
  (mean KL within 0.03%, top-1 equal or one apart, no paired test at p <
  0.05), which is the bar a tolerance route meets before it becomes a
  default.

- **CUDA: a speculative round that drafted nothing runs as a plain decode
  step.** With `--draft-lookup`, most rounds on text unlike the prompt find
  no match and verify a single row. That row went through the verify
  forward, which runs its kernels without the captured decode graph, so it
  cost more than the plain step it equals. Such a round now takes the decode
  step itself: the same bits (the verify rows are the decode step's), and on
  an RTX 3070 Qwen3-4B with lookup where about one token a round is accepted
  decodes at 102 tok/s where it ran 94 (plain 106), and Qwen2.5-7B at 73
  where it ran 71 (plain 71). MTP rounds keep the verify forward, since the
  head reads the round's hidden rows.

- **Speculative decoding gives the plain path's output on every backend, bit
  for bit, and pays on CUDA where drafts are accepted.** The draft only
  proposes; the target's verify round decides every token, so the output is
  the plain path's only if each verify row is the bits a single decode step
  at that position would produce. They were not: a CUDA round ran the
  batched prompt kernels (on the datacenter MIG slice the drafted text
  differed from plain decoding in 4 of 12 runs on the mixture-of-experts
  pairs), a Metal round took other kernels than a decode step from its
  second row on, and a CPU round of 8 or more rows (`--draft-k` 7 and up,
  grammar fast-forward) took the batched f32 tile. Now a CUDA round runs the
  decode kernels' multi-row twins (the GEMVs, the decode attention per row
  at its own position, the per-token MoE), up to 16 rows a pass; a Metal
  round runs its rows as decode steps; a CPU round takes the single-step dot
  per column; and the re-fold after a recurrent model rolls a round back
  runs the same way. `test-spec-identity` (in `make test`) and `make
  test-cuda-spec-identity` (in the CUDA gate) hold every row of rounds 2 to
  16 rows wide, with the walk's rewinds and fold rollbacks, byte-identical
  to single steps on dense, mixture-of-experts (Q8_0, Q4_0, MXFP4, gemma-4's
  dual branch), 256-wide K-quant and i-quant, and recurrent fixtures. Speed
  on an RTX 3070 (256 greedy tokens, two rounds within 1%): prompt lookup on
  an input-grounded prompt decodes Qwen3-4B at 143 tok/s with `--draft-k 4`
  and 137 with `--draft-k 12`, where the batched verify gave 69 and 134, and
  Qwen2.5-7B at 84, against 103 and 71 for plain decoding; the Qwen3.5-4B
  MTP head runs at 54 where it ran 33, against 79 plain, and gemma-3-4b with
  lookup at 105 where it ran 75 (with the two KV changes below), against
  102. Where about one token a round is accepted speculation still costs
  more than it saves (Qwen3-4B at 93-94 against 106). Metal is unchanged
  (its batched verify already cost about a decode step a row).

- **CUDA: an IQ4_XS decode microbatch gives single-step bits.** Since IQ4_XS
  gained its coalesced single-step GEMV (`k_gemv_iq4_xs`, with qwen4exp),
  its width-classed twins were still the twins of the lane kernel it
  replaced, a different reduction, so `--parallel` slots decoding an IQ4_XS
  file together differed from single steps in the last bits. The twins now
  mirror `k_gemv_iq4_xs` (four rows a warp, as the K-quant twins), and `make
  test-cuda-batch-identity` runs 256-wide Q4_K, Q6_K and Q4_0 fixtures, and
  Q5_K and IQ4_XS ones where llama.cpp's quantizer is at hand; no gate had
  an IQ4_XS file before.

- **CUDA decode twins take several rows a warp.** The decode microbatch
  (`--parallel` slots decoding together) runs width-classed twins of the
  decode GEMVs, each column the single step's bits. They staged every
  column's activations once per eight rows and had each row re-read them,
  re-forming each column's activation sum per row, and an 8-wide pass
  computed all eight columns whatever the batch held. A warp now takes four
  rows (two in the new 16-column class) and reads each staged value once for
  all of them, and a pass computes only the columns it has. With four
  Qwen3-4B Q4_K_M slots an RTX 3070 serves 159 tok/s where it served 135,
  and 178 where it served 147 with eight; every slot's text equals its
  single-request run.

- **CUDA keeps the token embedding in RAM and uploads only the tensors it
  reads.** The forward stages each token's embedding row from the host, yet
  the device held the whole embedding table, and its weight image ran from
  the file's first tensor (through the end of the file on a full split), so
  whatever lay in front of the layers rode along without being planned:
  gemma-4-E4B's 1.94 GB per-layer embedding table, which only the host's
  pre-pass reads. The image is now the stretch of the tensors the device
  reads (the layers, and on a full split the output head; a tied head is the
  embedding table); where they do not lie together (a head stored in front
  of the embedding, a name-ordered file) they upload as one allocation per
  stretch. On an RTX 3070, Xyntetik-Kvist-14B at an 8K context holds 51 of
  its 52 layers where it held 47 and reads the prompt at 105.2 tok/s where
  it read 34.4, decoding at 10.25 where it ran 5.78 (two rounds);
  gemma-4-E4B leaves 4.48 GB free where it left 2.47; granite-4.2-8b at a
  16K context places 36 of 40 layers where it placed 35, and all 40 with
  `--kv q8` where 39 fitted. `--score` is byte-identical to the previous
  upload at equal splits on thirteen files and splits, and `--fit` predicts
  the loader's split in 48 of 48 cases (six models, two contexts, four KV
  formats).

- **CPU IQ3_S dot fused.** IQ3_S was the slowest format on the CPU: its dot
  dequantized each block into a 256-float scratch row, multiplying every
  code by its sub-block scale, and then took a second pass to dot it. It now
  reads the grid as floats into the multiply-adds and applies the sub-block
  scale once per 32 weights, as the IQ3_XXS dot does: one core 1.44x on the
  Linux host and 1.49x on an Apple M1, and an Apple M1 decodes Llama-3.2-1B
  IQ3_S on its CPU at 18.3-19.3 tok/s where it decoded 14.0-14.4. Applying
  the scale once reorders the arithmetic; against the publisher's float32
  forward over 2,000 positions granite-4.2-3b IQ3_S reads mean KL 0.76508
  against 0.76502 (Wilcoxon p 0.85), not distinguishable.

- **CPU verify rounds and small batches decode each weight block once.** A
  speculative verify round on the CPU computes every row with the decode
  step's own dot, one column at a time, and a format without a multi-column
  form of that dot decoded each weight block again for every column: Q8_0,
  Q4_0 and MXFP4 on both architectures, and on arm64 also the K-quants,
  IQ4_NL, IQ4_XS, Q2_0, BF16 and IQ3_XXS. Each now has one that widens a
  block once and gives every column the single dot's operations, so the rows
  keep their bits. On an Apple M1's CPU, prompt lookup (`--draft-lookup
  --draft-k 4`) runs Qwen2.5-1.5B Q8_0 at 25.0 and 29.4 tok/s on two prompts
  where it ran 22.2 and 22.7 (plain decoding 22.6-24.1), and Llama-3.2-1B
  Q4_K_M at 28.7-29.9 where it ran 19.5-20.4 (plain 21.0-23.2).

- **CPU MXFP4 and IQ3_XXS dots read their scales and codes from tables, bit
  for bit as before.** The MXFP4 dots evaluated each block's E8M0 scale with
  a libm `ldexpf` call every 32 weights; the scale now comes from a
  256-entry table of the same values, and one core reads MXFP4 rows 3.1x as
  fast on the Linux host (7.3 to 22.8 G weights/s) and 1.7x on an Apple M1.
  The IQ3_XXS dot assembled each grid word's bytes into a vector and widened
  them to floats; it now loads the grid as floats and applies the signs as a
  bit mask: 1.28x on the Linux host, 1.12x on an Apple M1. The same values
  meet the same operations, so the outputs are unchanged; a new test pins
  every table entry against the dequantizer (a one-hot activation turns a
  dot into the weight it multiplies).

- **CPU K-quant dots keep one accumulator a row.** The Q4_K, Q5_K and Q6_K
  dot products (the AVX2 and NEON single and group-sum forms, and the AVX2
  multi-column ones) added each 32-weight sub-block into the row's sum with
  a horizontal reduction, two of them per 64 weights; each sub-block's sum
  is now scaled into one vector accumulator (the min term kept apart) and
  the row reduced once. One core of the Linux host reads Q4_K rows on the
  decode route 1.44x as fast (14.4 to 20.7 G weights/s), an Apple M1 core
  1.17x, and an Apple M1 prefills Llama-3.2-1B Q4_K_M on its CPU at 88 tok/s
  where it read 75 (decode up to 3% faster). The same terms in another order
  move the last bits (about 1e-6 in a logprob); against the publishers'
  float32 forwards over 2,000 positions the two orders are not
  distinguishable (Qwen3-4B Q4_K_M mean KL 0.13987 against 0.13988, Wilcoxon
  p 0.50; Llama-3.2-3B Q4_K_M 0.04391 against 0.04389, p 0.07).

- **Llama 3 files without an add-BOS flag get their BOS token.** A GGUF
  that does not carry `tokenizer.ggml.add_bos_token` was read as "no BOS"
  for every BPE vocabulary, and Llama 3 conversions commonly omit the key
  (both Llama-3.2 files on our hosts do). The runner's Llama 3 chat format
  leaves `<|begin_of_text|>` to the tokenizer, so chat prompts on such a
  file started without it, as did `--score` and raw prompts. A missing key
  now means what the family's own tokenizer does: a BOS for SentencePiece
  vocabularies and for the Llama 3 (`llama-bpe`) and Mistral Tekken BPE
  families; an explicit key still decides. Llama-3.2-1B's perplexity on
  the manual's first 6,000 characters (`--score`, CPU) is 18.64 with the
  token where it read 19.02 without it.

- **CUDA sliding-window layers copy only the rows a forward writes.** A
  layer that keeps its KV in a ring mirrored the whole ring between host
  and device on every copy, so a speculative round or a microbatch step on
  a gemma model moved every slot of every sliding layer across the bus;
  the copies now move the slots the forward's positions occupy (at most
  two stretches when they wrap). `--score` and generation are
  byte-identical to the previous copies on gemma-3-4b, gemma-4-E4B and
  gemma-4-E2B at an 8K context.

- **CUDA no longer re-uploads the KV cache when a sequence rewinds, and a
  resumed session is never decoded against the previous request's rows.**
  The device keeps a mirror of the host's KV cache and took the host's
  rows again whenever a forward did not continue at the previous position
  plus one: every speculative round that kept only part of its draft, and
  every request reusing a slot's own prompt prefix, re-uploaded the whole
  cache. A forward continuing exactly there was trusted instead, so a
  session image (`/v1/runner/sessions/.../resume`) installed into the host
  cache of a slot whose previous request had stopped one row short of it
  decoded against that request's device rows. The mirror now counts the
  leading rows it holds as the host does, uploads only the rows past them,
  and is told when the host writes rows itself (a prefix fork, a session
  image). `tests/test_server_sessions.py` resumes an image on a slot left
  one row short by another session (it failed on the previous build).

- **CUDA: Qwen3 MoE prefill groups its experts on the tensor cores, and
  Qwen3 MoE and gpt-oss run their dense projections there too.** The
  tensor-core route was kept off the MoE families because routing can
  amplify its rounding. Held to the publisher's float32 models over 2,000
  positions instead, under the rule fixed before the runs (mean KL within
  1.01x, paired Wilcoxon and McNemar not worse at p < 0.05), neither read
  further from the reference: Qwen3-30B-A3B mean KL 0.09167 against 0.09157
  (p 0.36), gpt-oss-20b 0.08142 against 0.08312 (closer, p 1e-5), and
  Qwen3-30B-A3B with its Q4_K / Q6_K experts grouped on the tensor cores as
  well, scored in 64-row chunks so those GEMMs ran, 0.09178 against
  0.09170 (p 0.83). On the datacenter MIG slice Qwen3-30B-A3B reads a
  512-token prompt at 845 tok/s where it read 215 (454 against 174 at 2K)
  and gpt-oss-20b at 216 where it read 174 by expert, three rounds each.
  They run as the routes `moe-tc` and `moe-grouped`: a record of these models names
  them, `--verify` replays them, and `--gpu-route reference` (or a record
  made before them) runs without them.

- **CUDA MoE prefill runs by expert on gemma-4 and gpt-oss, with the same
  bytes.** The fused MoE prefill computed each token's selected experts one
  token at a time, so a prompt tile read every routed expert's weights once
  for each token that chose it. On gemma-4 MoE and gpt-oss the prefill now
  groups the tile's (token, expert) pairs by expert, runs each expert's
  columns eight at a time through matvec twins that give every column the
  single-token kernel's arithmetic, and sums each token's expert outputs in
  its router order, so the output is the fused path's bit for bit
  (`GPU_ID_MOE_TWIN=1 ./test-gpu-identity`, launch count required; `make
  test-cuda-moe-twin` in the CUDA gate on routed Q8_0 and Q4_0 fixtures,
  the gpt-oss MXFP4 fixture and gemma-4's dual-branch layout at Q8_0 and
  Q4_0). On the datacenter MIG slice prefill rose 13-14% at a 512-token
  prompt on gemma-4-26B-A4B and gpt-oss-20b (11-13% at 2K, three rounds).
  Qwen3-30B-A3B measured 8-9% slower by expert, so qwen3moe, and families
  not yet measured, keep the per-token path; `RUNNER_MOE_TWIN=1` takes the
  by-expert prefill on any family and `=0` on none.
- **CUDA decode for IQ4_NL, NVFP4, Q2_K, IQ1_S, IQ1_M, Q4_1, Q5_0, Q5_1
  and Q2_0 reads the activations once per block of rows, with the same
  bytes.** These formats have no coalesced decode kernel; their matvec,
  like Q4_0's and MXFP4's before the previous change, gave each lane a
  whole quant block and read its activations 128 bytes or more from the
  next lane's. They, and the MoE expert copies for NVFP4, IQ4_NL, IQ4_XS
  and Q2_0, now take the same staging through shared memory, eight rows a
  block, every lane adding the same blocks in the same order. On an RTX
  3070, Xyntetik-Kvist-14B IQ4_NL decodes at 22.9 tok/s where it ran 6.8
  and granite-4.2-3b IQ1_M at 60.8 where it ran 24.6 (three rounds); the
  matvecs themselves run 3.5-9.0x faster. `--score` is byte-identical to
  the previous kernels on Kvist-14B IQ4_NL-mix, Qwen3.5-4B NVFP4 and
  granite-4.2-3b IQ1_M and IQ2_XXS.

- **CUDA MoE experts in Q4_0 and MXFP4 read their activations once per
  block of rows, with the same bytes.** These two formats' expert matvec
  gives each lane a whole 32-weight block, so a lane walked its block's 32
  activations 128 bytes from its neighbours and every warp load touched 32
  cache lines; MXFP4 also fetched its 16-entry codebook from global memory
  for every nibble. The rows of a launch now copy 32 blocks of activations
  at a time into shared memory with coalesced loads (padded so the lanes'
  reads do not conflict) and keep the codebook there. Every lane still adds
  the same blocks in the same order with the same expression, so the output
  is unchanged bit for bit: `--score` (both routes) and 64 greedy tokens
  are byte-identical to the previous kernels on gemma-4-26B-A4B Q4_0 and
  gpt-oss-20b MXFP4. The expert matvec itself runs 5.1-7.0x faster on
  those two models' shapes (RTX 3070 and the datacenter MIG slice), and on
  the slice, with 512-token prompts, gemma-4-26B-A4B decoded at 75.6 tok/s
  where it ran 39.6 and read prompts at 203.6 where it read 58.8, and
  gpt-oss-20b decoded at 96.0 where it ran 32.6 and read prompts at 152.7
  where it read 37.3 (three rounds each).

- **Phi-3.5 ropes every layer with its LongRoPE factors.** phi3 exports
  publish a sliding window without a layer pattern (Phi-3.5: 262144 over a
  131072 context), and the loader read it as gemma2's alternation, so every
  other layer roped without the LongRoPE short factors and attention
  factor. Against the publisher's float32 model over 2,000 positions,
  Phi-3.5-mini Q4_K_M went from mean KL 0.275 to 0.045 (another engine on
  the same file: 0.049) and margin-qualified top-1 from 85.6% to 98.5%. A
  phi3 window now covers every layer alike under the one rotation, and a
  window at least the trained context covers none; `tests/test_phi3_rope.py`
  holds a phi3 fixture with non-identity factors to that.

- **gemma-4 E2B runs on CUDA.** E2B publishes a different FFN width per
  layer, and the CUDA backend sized its FFN path off one global width, so it
  refused these files and they ran on the CPU. Every FFN projection now takes
  its layer's width (the buffers keep the widest layer's stride). On an RTX
  3070, gemma-4-E2B went from 24-33 to 800 tok/s prefill and from 11 to 104
  tok/s decode; 64 greedy tokens on the device match the CPU byte for byte on
  E2B and E4B, and `make test-cuda-varff` holds a fixture with alternating
  widths to the CPU in the CUDA gate.

- **CPU decode on arm64: Q4_K and Q5_K dots widen their weights by table
  lookup.** Most of the instructions in the NEON Q4_K/Q5_K dot went to
  widening each weight byte to a float through two widening moves. A byte
  table lookup now places four bytes into four 32-bit lanes in one
  instruction; the integers, and so every float and every output, are the
  same (CPU `--score` byte-identical, solo and chunked, on Q4_K_M files and
  a Q5_K_M fixture). Apple M1, four threads, five interleaved rounds:
  decode +4.0% on Llama-3.2-1B Q4_K_M, +6.5% on Llama-3.2-3B Q4_K_M and
  +4.1% on Qwen2.5-1.5B Q4_K_M; prefill unchanged.

- **Gates can see which route ran, and a device run that gives up says so
  by failing.** `--score` ends with one stderr line, `score: route {...}`,
  carrying the paths and counted kernels of its forwards, so a byte
  comparison of two score files can show that the switched route ran (a
  tensor-core route check had compared five files byte for byte with no
  sign the new kernels launched). `RUNNER_GPU_FALLBACK=fail` ends the
  process with exit status 3 when a device forward fails at runtime instead
  of releasing the GPU and continuing on the CPU; the comparison and
  certification scripts set it, so a fault part-way through a run (the Q3_K
  fault below) can no longer be recorded as a device result.

- **CUDA decode: the RMS norm takes a third of its time.** A decode
  token's norm is one 256-thread block on an otherwise idle card, run twice
  a layer. It now reads its row once into registers and reduces with five
  fewer barriers, in the same order, so every output is unchanged. On an RTX
  3070 the kernel went from 6-7 us to about 2 us and decode rose 1.1-2.6% on
  three models (Llama-3.2-3B Q4_K_M 113.3 to 116.3 tok/s).

- **`--gpu-route` names the CUDA arithmetic routes, and records keep them.**
  The tensor-core and attention kernels this release adds compute in another
  order or precision than the reference kernels (each held to a tolerance
  gate). `--gpu-route reference` runs none of them, a comma list of `tc2`,
  `sm80`, `attn-tile`, `attn-dec`, `attn-tc` runs those, and
  `RUNNER_CUDA_ROUTE` takes the same value from the environment. The set a
  CUDA run used goes into transcripts and receipts (`profile.gpu_route`), and
  `--verify` replays with the record's set; older records replay with
  `reference`. Off CUDA the flag is accepted and does nothing.

- **Q3_K models decode on CUDA again.** The Q3_K decode kernel read its
  quant and high-mask bytes eight at a time from 110-byte blocks, so three
  of every four loads were misaligned and the first decoded token raised a
  device fault; the runner then released the GPU and decoded on the CPU
  (Qwen2.5-32B Q3_K_S at 0.85 tok/s on the datacenter MIG slice, where
  another engine decodes 26). The bytes are now read as 16-bit words, the
  same values. `test-gpu-identity` had read ok throughout, because it
  recorded GPU use at load and the faulted run finished on the CPU; it now
  fails a run that loses its backend part-way, and `make test-cuda-q3k`
  (in the CUDA gate) runs a Q3_K fixture on the device.

- **TC2 on compute capability 8.0 and newer uses the newer tensor-core
  instruction.** The embedded kernels target compute capability 7.5, so TC2
  issued two `m16n8k8` multiplies per 16-deep step even on cards that have
  the `m16n8k16` instruction its operands already match. A second, small
  module built for compute capability 8.0 carries the TC2 kernels for Q4_K,
  Q6_K, Q8_0, Q4_0 and the IQ1-IQ4 i-quants and is loaded on such a device
  unless TC2 is switched off; every other kernel keeps the existing module. `RUNNER_CUDA_SM80=0` leaves the second module unloaded, so TC2
  runs only when forced (`RUNNER_CUDA_TC2=1`), on its compute_75 kernels.

- **The int8 CPU route runs on arm64.** `RUNNER_CPU_I8=1` now has
  dot-product (SDOT) kernels for Q8_0, Q4_0 and Q4_K on arm64 (every Apple
  M-series core) beside the x86 ones, and quantizes the activations with the
  same bytes as the scalar definition. It stays off by default under the
  same tolerance gate (`./test-i8-tol MODEL.gguf`).

- **CPU decode: Q4_K and Q5_K rows no longer re-sum the activations.** The
  dot subtracts each 32-element group's minimum times the sum of the
  activations in that group, and that sum was recomputed for every row of a
  matrix. A decode matvec now computes the sums once and every row reuses
  them, byte-identical (`test-quants-simd` compares the two on random and
  boundary inputs). On an Apple M1 at four threads, decode went from 20.27
  to 20.79 tok/s on Llama-3.2-1B Q4_K_M and from 16.01 to 16.49 on
  Qwen2.5-1.5B Q4_K_M.

- **`--score-top N` reports each position's distribution head.** With
  `--score`, a `top` array lists every position's N most likely next tokens
  with their logprobs. `scripts/gold-logits.py --score-a FILE` compares such
  a file with the publisher's float32 model in one pass over the text, and
  with `RUNNER_SCORE_CHUNKED=1` every position comes from the batched prefill
  route, which a request per position cannot reach.

- **CUDA prefill of Gated DeltaNet and Mamba-2 layers in one launch per
  tile.** The recurrent layers of Qwen3.5/3.8, granite-4.x-h and Nemotron-H
  advanced a prompt one token per kernel launch, reloading each head's state
  from device memory every time. A prompt tile's convolution and state
  updates now run in one launch per kernel with the state held in
  registers, the per-token arithmetic unchanged: byte-identical logits and
  state (`GPU_ID_RECUR=1 ./test-gpu-identity`). With the experimental
  tensor-core and attention kernels on, Qwen3.5-4B Q4_K_M prefill went from
  333 to 869 tok/s at a 512-token prompt and from 288 to 622 at 8K on an
  RTX 3070, and from 434 to 791 and 375 to 612 on the datacenter MIG slice,
  where Qwen3.8-27B IQ3_S went from 53 to 66 and 50 to 61; decode is
  unchanged. `RUNNER_CUDA_RECUR_SEQ=0` keeps the per-token launches.

- **CUDA decode: one KV copy-back per token.** The host KV cache stays
  authoritative, so each decoded token's K and V rows went back to it as two
  copies per offloaded layer. With every layer on the device, the new rows
  are now gathered there as the last node of the decode graph and brought
  back in one copy, the same bytes to the same host slots. On an RTX 3070
  the copies per token went from 57 to 2, and with the experimental
  attention kernels on, decode rose 5.5% on Llama-3.2-3B Q4_K_M (120.3 to
  127.0 tok/s), 7.2% on Qwen2.5-3B and 3.5% on Qwen2.5-7B. On the
  datacenter MIG slice, default route, decode rose 5.0% on Llama-3.2-3B,
  4.3% on granite-4.1-3b Q8_0 and 2.6% on Qwen2.5-7B at a 512-token context
  (2.6%, 2.5% and 1.7% at 4K). `RUNNER_CUDA_KV_GATHER=0` keeps the
  per-layer copies.

- **CUDA tensor-core prefill attention (`attn-tc`), Q and the softmax
  weights in two fp16 halves.** It was added opt-in and is now the default
  for the llama, qwen2, qwen3 and granite families (the entry above). With
  Q and the softmax weights rounded to single fp16 operands, Mistral-7B
  v0.3's long-context float32-reference row read 0.19% further from the
  reference than the default kernel's (a significant paired difference, p
  0.014, top-1 unchanged), which the promotion rule does not admit. Both now
  go in as two fp16 halves (the value and its rounding remainder, about 22
  bits), and that row reads the same as the original kernel's (mean KL
  0.02069 against 0.02069, p 0.60); on the datacenter MIG slice the split
  costs 27% of the kernel's speed (a 7,841-token prompt on Llama-3.2-3B:
  3,262 to 2,377 tok/s, against 781 with the tiled kernel).
  `RUNNER_CUDA_ATTN_TC=1` computes the prefill attention scores and the
  value sums of an f16 cache on the tensor cores (64 query rows a block, two
  passes over the cached rows), for head widths that are multiples of 16 up
  to 128; quantized caches keep the tiled kernel. Before the split, on an
  RTX 3070, a Llama-3.2-3B Q4_K_M request with a 16K-token prompt and 64 generated
  tokens took 20.8 s against 59.5 s with the tiled kernel and 170 s with the
  original one, generating the same 64 tokens (one run each).

- **CUDA pipelined tensor-core prefill GEMM (TC2), on by default on
  compute capability 8.0 and newer.** The batched projections run through a
  128 x 64 tile GEMM that converts the next 64-deep stage of quantized
  weights into shared memory while the tensor cores multiply the current
  one, with the activations converted to scaled fp16 once per GEMM instead
  of once per row block. On a compute capability 8.0+ card it runs the
  kernels built for that architecture (above), whose fp32 sums are the
  tensor-core route's own: chunked `--score` logprobs came out byte for byte
  the tensor-core route's, with TC2 launches counted, on four files
  on an RTX 3070 and three on the datacenter MIG slice, and the CUDA gate
  now checks the same on its device (`make test-cuda-tc2-bytes`, which reads
  the device's `compute_capability`, now in `--caps`). On an RTX
  3070 at a 512-token prompt, the default route's prefill went from 1090 to
  1824 tok/s on Llama-3.2-3B Q4_K_M and from 639 to 1159 on Qwen2.5-7B
  Q4_K_M (2K: 653 to 862 and 453 to 667; 8K, where attention takes most of
  the time: 235 to 259 and 197 to 231); decode is unchanged. Where a GEMM
  spans more than 64 prompt columns, the compute_80 kernels take a
  128-column tile, which converts each weight stage once for twice the
  columns with each column's sums unchanged (`GPU_ID_TC2_WIDE=1
  ./test-gpu-identity`: byte-identical to the 64-column tile with launches
  counted, four files on the RTX 3070 and three on the datacenter MIG slice):
  another 3.6-6.1% at a 512-token prompt on four files (0.6-0.8% at 8K);
  `RUNNER_CUDA_TC2_WIDE=0` keeps the 64-column tile. Cards below
  8.0 keep the tensor-core route by default;
  `RUNNER_CUDA_TC2=1` forces the compute_75 TC2 kernels there (the same
  numerical class as the tensor-core route, not the same bytes:
  `test-tc-tol` 0 flips of 64 and a token-identical free-running arm on
  five files), and `RUNNER_CUDA_TC2=0` or `--gpu-route reference` switches
  TC2 off. A run records the kernels it took (`profile.gpu_route` reads
  `tc2,sm80` on a default 8.0+ run), and `--bench-json` counts the launches
  (`route.kernels.cuda_tc2`). The kernels cover Q4_K, Q6_K, Q8_0, Q4_0 and
  the IQ1-IQ4 i-quants on both modules; the codebook decoders moved into
  headers both modules share (the compute_75 PTX byte-identical after the
  move), and the i-quants' compute_80 kernels give the tensor-core route's
  bytes too (chunked `--score` on IQ3_S, IQ3_XXS, IQ2_XXS and IQ1_M on an RTX
  3070 and IQ3_S on the datacenter MIG slice, launches counted; their
  128-column twins byte-identical to the 64-column tile, IQ3_S and IQ2_XXS
  on the RTX 3070 and IQ3_S on the slice). On an RTX 3070, granite-4.2-3b IQ3_S prefill went from 814
  to 1265 tok/s at a 512-token prompt (513 to 663 at 2K).

- **CUDA long-context attention, on by default.** Prefill attention runs
  one strip of query rows at a time over K and V tiles staged in shared
  memory (each score computed as the existing kernel computes it), and
  decode attention reads each cached KV row once for all the query heads
  that share it, instead of once per head; the grouped decode splits the
  cached rows into whole 32-row tiles, so a short context costs nothing.
  Both add in another order than the kernels they replace, so they were
  held to the publisher's float32 model first: at positions 4,000 to 6,000
  of a long document (2,000 positions; Qwen3-4B, Mistral-7B v0.3, and
  gpt-oss-20b with its attention sinks and alternating 128-token window),
  mean KL and margin-qualified top-1 were never worse than the previous
  kernels' (paired tests p 0.15 to 0.60 on the first two; on gpt-oss-20b
  the tiled prefill read closer to the reference, p 4e-6, and the grouped
  decode matched it, p 0.24). On an RTX 3070 at a 16K-token prompt,
  Llama-3.2-3B read the prompt at 347 tok/s where it read 106 and decoded
  at 57.6 tok/s where it decoded 25.2 (Qwen2.5-7B: 264 against 120, and
  43.4 against 21.7); at a 512-token prompt prefill gains 17-28% and decode
  4-7% (three rounds). `test-gpu-identity` holds them to the CPU bound on
  every KV format (`GPU_ID_KV=q8|fp4|k8v4`). The server's decode
  microbatch (slots decoding together) runs the grouped decode attention's
  multi-sequence twin, so a batched step stays byte-identical to the slots'
  sequential steps; until the twin it kept the previous kernel and was not,
  and no CUDA gate step held that contract (`make test-cuda-batch-identity`
  now does: F32, Q8_0, BF16 and a Q4_K model, the grouped attention on and
  off). A run records them
  (`profile.gpu_route`: `attn-tile`, `attn-dec`); `--gpu-route reference`,
  `RUNNER_CUDA_ATTN_TILE=0` and `RUNNER_CUDA_ATTN_DEC=0` switch them off.

- **CUDA prefill reads the prompt in wider windows.** The device ran a
  prompt in tiles of at most 64 tokens, fixed at build time, so every weight
  matrix was read once per 64 tokens. The window is now chosen at load from
  the memory the layer split leaves spare: up to 512 tokens, never at the
  cost of a layer, still capped by `-b`, and `--fit` and the loader charge
  the same scratch. The default tensor-core GEMM stages 64 token columns
  per block whatever the window, so on its own the wider window measured
  within 1% of the 64-token one on an RTX 3070 (512 to 8K tokens); what it
  pays for is the pipelined GEMM (TC2, below), which runs every column of
  the window in one launch: with it and the experimental
  attention kernels on, prefill at a 512-token prompt rose from 1692 to 2711
  tok/s on Llama-3.2-3B Q4_K_M and at 8K from 1138 to 2044 (window 64
  against 512). The window decides which kernel
  route a prompt's last partial tile takes, so transcripts and receipts
  record it (`profile.gpu_window`) and `--verify` rebuilds the instance at
  the recorded window, refusing as unverifiable when it cannot; a record
  written before this change replays at the old window of 64.

- **CUDA prefill: the tensor-core GEMMs no longer conflict on shared-memory
  banks.** They staged 128-element fp16 rows at a 256-byte stride, so every
  row of a tile read from the same banks. Rows are now padded by 16 bytes.
  The arithmetic is unchanged and the output byte-identical (`test-tc-tol`
  deviations equal before and after; chunked `--score` logprobs over 1,540
  to 1,605 positions on three models compared byte for byte). On an RTX
  3070 prefill at a 512-token prompt went from 600 to 1086 tok/s on
  Llama-3.2-3B Q4_K_M, 627 to 1091 on Qwen2.5-3B and 324 to 640 on
  Qwen2.5-7B Q4_K_M (five interleaved rounds, the same prompt window).

- **Qwen3.5/3.8 on CUDA: a race in the Gated DeltaNet kernel is closed.**
  The kernel read the query-norm sum from a shared reduction slot and then
  let every thread store its share of the key-norm sum into the same slots
  with no barrier between, so one warp could overwrite the query sum before
  another warp had read it and give that head a wrong query scale. It needs
  one warp to run a whole global load ahead of another, which is why no gate
  had caught it. A barrier closes it; no other kernel reuses a reduction slot
  without one.

- **Unloading a model on CUDA frees all of its device memory.** Teardown
  frees device buffers from hand-kept lists, and those lists had fallen
  behind the allocation sites: a model with hyper-connections (Qwen3.8), a
  shared expert, Mamba-2 layers or an MTP block left device buffers behind
  on every unload, in the context every model on the card shares, so a
  server that swaps models lost VRAM until it exited. Teardown now frees
  them. A count of live device allocations shows the lists agreeing
  (`test-cuda-teardown`: three load and free cycles of the qwen4exp test
  model return to the starting count, in the CUDA gate), and a source test
  holds each free list against its allocation sites.

- **CUDA runs a file with a removed sublayer.** A file made by
  `--remove-sublayer`, such as the released Gemma 4 31B with block 48's
  attention removed, ran only on the CPU under a CUDA build. The CUDA walk
  now skips the removed branch and leaves its KV unallocated, as the CPU and
  Metal paths do. On the 31B at `-c 4096` that frees 64 MiB of KV, and the
  output matches the CPU path within the usual CUDA envelope (9 of 9 prompts
  exact over 64 tokens; 0.0012 nats mean over 326 scored positions). A
  removed attention on a MoE file keeps its FFN path, experts on the device
  or the host. The `-v` line no longer says such a file runs on the CPU path.

- **LoRA adapters serve on Qwen3.5/3.6/3.8 hybrids and on Clef.** An
  adapter on a qwen35 model or a Clef file was refused ("recurrent
  architectures"). It now sits on each Gated DeltaNet layer's five
  projections (`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`,
  `ssm_out`, which is how llama.cpp's converter names PEFT's
  `in_proj_qkv`/`z`/`b`/`a` and `out_proj`), on the full-attention
  projections and on every FFN, on the CPU and on CUDA. On Clef it reaches
  the decision head through the backbone, and a `qwen35` adapter, which the
  converter writes from that backbone, is accepted. `--merge-lora` folds
  the same projections. A served adapter matches the weights merged by hand
  to within 1e-5. Training on these models, and the Mamba-2 hybrids, are
  still refused.

- **`--fit` charges each layer its own KV cache.** It charged every layer
  the scalar KV heads times the scalar head width. On gemma-4-31B that is
  about four times what a load allocates (120 GiB at 32k), and a file with a
  removed sublayer was refused outright. It now reads per-layer KV heads,
  gemma-4's sliding head width, removed blocks and E-series shared-KV layers
  the way the loader does, and the GPU split it predicts uses the same
  per-layer figures. A test holds `--fit`, the loader's allocation and a
  reading of the header to the same bytes.

## v1.4.2 - 2026-10-17

- **Reinforcement fine-tuning trains on exactly what it sampled.**
  `scripts/train-grpo-lite.py` promised that the policy that samples is the
  policy that trains, bit for bit, but nothing tested it, and two things
  broke it. The loop handed the trainer the sampled text, which the trainer
  tokenized again: that is not an identity (a byte token that ends in the
  middle of a character decodes to U+FFFD), so it could train on tokens the
  model never produced. And the rollout server read each prompt in one
  batch, which from 8 rows up takes a different CPU route than the trainer's
  one token per forward: the logprobs after the prompt then differed, by
  up to 3.1e-3 on MiniCPM5-1B Q4_K_M after the loop's 60-token prompt. The loop now passes token ids, reads prompts with `-b 1`, and
  stops if the trainer's first step scores a rollout differently from the
  server. `tests/test_rollout_identity.py` holds the identity on every
  family the trainer admits, in `make test`.

- **`--train` takes token ids, and reports the logprobs it trains on.** A
  `.jsonl` line can give `{"prompt_tokens":[...],"completion_tokens":[...]}`
  instead of text, used exactly as given. `--train-logprobs FILE` writes,
  per step, the logprob of every completion token under the forward that
  step trains through, before its update.

- **A continued adapter's record names the adapter it started from.**
  `--train` with `--lora` continues an existing adapter (the reinforcement
  loop does it every round), but the record named only the base and the
  data, and gave the `--lora-rank` default as the rank whatever the loaded
  adapter's was. Such a record could not reproduce its run. It now records
  `from_adapter` (hash and its own record, taken before the first step, so
  training in place is recorded right) and the adapter's real rank, and
  `--lineage` walks back through every earlier run.

- **`--train-dpo` no longer needs `--gpu off`.** On a CUDA or Metal build it
  loaded the model onto the GPU and then refused, naming `--train`. It now
  takes the CPU path the way `--train` does.

- **Every adapter feature, explained in one place.**
  [docs/lora.md](docs/lora.md) walks through checking a model, data formats,
  training, continuing, DPO, the reinforcement loop, evaluation, serving one
  or many adapters, merging, records, signatures, lineage, receipts and
  interop, each with its commands, limits and measurements. The README,
  manual, site and the other adapter pages link to it.
- **The gold logit check no longer floors stop tokens.** A server spells a
  stop or control token "" (it decodes to no bytes), while the float32
  reference spells it `<|endoftext|>`, so `scripts/gold-logits.py`, which
  matched by spelling, scored the reference's mass on it at a floor: one
  qwen35 position read KL 0.112 where its 19 shared tokens agree to 8.2e-9.
  It now matches by token id where the endpoint gives ids, reads a greedy
  stop from `stop_logprobs` instead of skipping the position, and, where an
  endpoint gives no ids, drops an unlisted special token and reports the
  dropped mass (report schema v4). Earlier reports may overstate absolute
  KL at such positions (docs/golden-pass-2026-09-07.md).

- **A gemma4 MoE file no longer prints the dense variant's verification
  line.** The load banner said "dense variant verified against llama.cpp"
  on gemma-4-26B-A4B-it too; it now names what the MoE variant was verified
  on.

- **Logprobs are reported at full precision.** The completion and chat
  logprob fields printed six decimals, which hid differences in the last
  bits and lost most digits of a small logprob. They now print the
  engine's float at nine significant digits, which round-trips it exactly,
  as `prompt_logprobs` and `--score` already did.

## v1.4.1 - 2026-10-16

- **Several engines on one CUDA GPU keep the GPU.** A program that ran
  Runner engines on separate threads, sharing one CUDA device, could lose
  all but one of them to the CPU. One engine capturing its decode step into
  a CUDA graph while another made a synchronous copy is a combination CUDA
  refuses: it fails the copy and invalidates the capture, and the failed
  forward dropped the copying engine to the CPU for good. The answers stayed
  correct, so only the speed showed it. Captures and the driver calls that
  would collide with them now take turns. A `--parallel` server was not
  affected, because its scheduler drives the device from one thread.

- **Parallel slots no longer race on state the process sets up on first
  use.** Runner reads a few dozen settings once and keeps them: kernel
  switches from the environment, dispatch counters, once-only notices, debug
  trace files and the CUDA driver's function table. Each was a plain variable
  that whichever thread got there first wrote, so two slots, or an
  embedder's two engines, starting together raced on it. Two of these races
  had visible effects. The CUDA driver table was marked loaded before it was
  filled, so a second engine starting at that moment could call into an
  empty table. The per-layer similarity diagnostic (`RUNNER_LAYER_SIM`)
  shared one buffer between concurrent forwards. All of this state is now
  atomic, set up once, or per thread. `make test-decide-race-gpu` runs four
  engines on the GPU at once, under ThreadSanitizer where the platform has
  it.

- **A run of combining marks no longer stalls a Qwen tokenizer.** NFC puts
  each run of combining marks in combining-class order, and Runner sorted the
  run with an insertion sort, whose time grows with the square of the run's
  length. A request made of marks out of class order therefore held its slot
  before inference began: 128 KB took 8.4 s in `/v1/messages/count_tokens`.
  A run longer than 32 marks is now sorted by counting, in linear time, and
  256 KB takes about 10 ms. The output is unchanged: over the 13,264-string
  NFC corpus and 5,902 more strings built around the sort's boundaries, the
  old and new code agree byte for byte, and both match the `tokenizers`
  package's NFC.

- **A stop exits within its 10-second bound even when nobody reads the
  log.** The bound's watchdog printed its last line through the C library's
  stream before exiting, and stopped watching once the requests were
  joined. With the server's output on a pipe nobody reads, that line waited
  on the full pipe, so a step that did not end held the process past the
  bound. A swap set with nothing in flight hung after the drain the same
  way, logging its unload once the watchdog was gone. The watchdog now
  watches the server's whole shutdown, writes its line from a separate
  thread with a plain `write`, and exits 200 ms later whether the line went
  out or not. When the shutdown finishes in time, the watchdog stands down,
  so a program that runs the server again in the same process keeps
  running.

- **Integers longer than fifteen digits keep every digit.** Runner kept the
  digits of such an integer when it parsed a request, but wrote the
  rounded double back out, so 9007199254740993 (2^53 + 1) became
  9007199254740992. It showed in Clef's score legend, which no longer named
  the criterion the model had scored. The same writer renders tool schemas
  and history into chat prompts, compiles `enum` and `const` literals into
  the output grammar, and echoes tool-call arguments, so each of those now
  keeps the caller's integer too. A JSON schema holding such an integer now
  has a sha256 of its own (it used to share the rounded schema's), so a
  record or session image made under one by 1.4.0 names a digest this
  release does not reproduce; both are bound to the build that wrote them.

- **`oneOf` admits only the values exactly one alternative lists.** JSON
  Schema's `oneOf` holds when exactly one alternative validates, and Runner
  merged literal alternatives into a union, which is what `anyOf` means. So
  `oneOf: [{"enum": [1, 2]}, {"enum": [2, 3]}]` let the model emit 2, which
  both alternatives admit. A value two alternatives share is now left out,
  and a `oneOf` that leaves no value (two identical `const` branches, say) is
  refused with a 400 instead of answered. `anyOf` is unchanged. Other
  alternatives could not overlap: they must already differ in their first
  character.

- **Schema literals compare by value, exactly.** A `const` keeps only the
  `enum` members equal to it, and `oneOf` drops the values two alternatives
  share, so both need number equality, which JSON Schema defines by value.
  The comparison used the rounded double, so
  `{"const": 9007199254740992, "enum": [9007199254740993]}` emitted
  9007199254740993 with HTTP 200, while the small-integer contradiction was
  a 400. It now compares exact values: an integer of any length by its
  digits, a whole double past 2^63 by its exact expansion, so `1e20` and
  `100000000000000000000` are one value and 2^53 and 2^53 + 1 are two.

- **A request's `seed` runs as sent, and `fork_seed` keeps the session seed
  rule.** The chat and completion `seed` was read as a double, so any seed
  above 2^53 could run and be reported as a neighbour: 9007199254740993 ran
  as 9007199254740992, and 18446744073709551615 (2^64 - 1) was refused as
  2^64. A seed written in digits is now read exactly over the whole uint64
  range, and the error names that range. A session resume's `fork_seed` was
  checked only for being a whole number, so 2^53 and 2^53 + 1 forked into one
  image, and a value past 2^63 reached an undefined cast. It is now held to
  the rule a session's own `seed` follows (a positive integer below 2^53)
  and refused by name otherwise.

- **Concurrent `/v1/decide` and `/v1/rerank` answers get distinct ids.**
  Each response id (`decide-N`, `rerank-N`) came from a counter that every
  parallel slot bumped without synchronization, so two answers produced at
  the same moment could carry one id: a probe saw 3 duplicates in 16,000
  concurrent calls. The counters are atomic now, and a ThreadSanitizer test
  holds them there.

## v1.4.0 - 2026-10-16

- **`--train` says when a run is about to memorize.** A user trained four
  examples that all answered the same passphrase for 60 steps at five times
  the default learning rate, and the adapter then answered the passphrase to
  "what is the capital of Japan". A step is one example, so that was 15
  passes over the data. Measured against a reference PEFT run with the same
  settings, the two behaved the same at every checkpoint: the data and the
  settings did it, not the trainer. `--train` now warns before the first step
  when a run is likely to end that way: more than three passes over fewer than
  200 examples, one completion shared by more than half of them, `--lr` above
  twice the default, or short completions without an end-of-turn token. The
  manual explains it with the numbers under "Train the GGUF you actually
  serve". Training itself is unchanged; the adapter bytes are the same.

- **Clef decision models load and answer at `/v1/systemone`.** Cloudflare's
  Clef (Qwen3.8-27B) and Clef-Flash (Qwen3.5-9B) put a trained joint head
  on a Qwen backbone: a state and typed questions in, a probability for
  every allowed option out, in one forward pass. Every public Clef GGUF
  carries the `clef` architecture, which Runner refused. It now loads them
  and answers `POST /v1/systemone` and `--systemone FILE` with the body
  Cloudflare's own `systemone()` returns. The prompt is built exactly as
  the publisher's encoder builds it and the head runs its module graph in
  float32: on a small Clef checkpoint scored against that code, all 15
  edge-case records are token-identical and the per-option scores agree to
  1.5e-4, on the CPU and with full CUDA offload. Text input only for now;
  a request with images or videos is refused. `"runner_logits": true`
  returns the raw scores beside the answers.

- **Qwen tokenizers normalize to NFC, as the publisher's do.** Qwen's
  `tokenizer.json` (Qwen2 through Qwen3.8) declares an NFC normalizer, so
  the reference composes decomposed text before it splits it, and Runner
  did not: "cafe" followed by a combining accent was two tokens here and one
  (" café") there. Runner now applies NFC to text between special tokens
  on the qwen2 and qwen35 tokenizers. Over 13,264 strings built to stress it
  (every character with a canonical decomposition written decomposed, plus
  marks out of order, conjoining Hangul jamo and Indic, Thai, Hebrew and
  Vietnamese text), Runner and the `tokenizers` package now produce the same
  ids for every one. Text that was already composed, which is almost all
  text, tokenizes as before.

- **`--decide` escapes the message in an error line.** A failing request
  was reported as `{"error":{...,"message":"..."}}` with the message
  written raw, so a message containing a quote produced a line that was not
  JSON. `--decide` and `--systemone` now share one reader that escapes it.

- **One SIGTERM stops a busy server in seconds.** The drain after the
  first SIGINT or SIGTERM waited for every active request to finish on its
  own, with no bound, so a server in the middle of a long generation or a
  slow prompt looked as if it ignored the signal: a 1,200-token generation
  held it 111.5 s, and a prompt paging a 26B model in from a slow disk needed
  SIGKILL. Active requests now get 2 seconds, then end at their next prompt
  chunk or token: a generating turn returns what it has (`finish_reason`
  `length`, `runner_telemetry.finish_detail` `shutdown`), and a request still
  in its prompt is answered `503 server shutting down`. If one step is still
  running 10 seconds after the signal, the process exits without it, with 128
  plus the signal number. A second signal still exits at once with 130. The
  manual documents all of it under "Stopping the server".

- **The Runner and Evidence pages open with the numbers and a way in.**
  The Runner page's first screen now carries the download button, the
  sixty-second start and the evidence link, four facts, and a strip of four
  measured results that link to their sections; the Evidence page opens
  with its four headline results, losing row first, and its table of
  contents gains the publisher-reference section that was only reachable
  by scrolling. Every number was already on the page with its date. Each
  hero now also carries a drawn visual generated from the site's own data:
  on Runner, a terminal with the three questions the page opens with and
  one real command and answer for each; on Evidence, the six-engine
  truncation grid at hero size, linked to its section.

- **The site tells search engines who publishes it and what Runner is.**
  Every page of xyntetik.com now carries schema.org structured data: the
  publishing organization with the repository, the container image and the
  model-file account as the same entity, and on the home and Runner pages
  the software itself with its version, license and download link. A
  `llms.txt` at the site root describes the program in plain text for the
  answer engines, next to the sitemap and robots file.

- **The container image, the release notes and the Python client say the
  same first sentence as the README.** The image labels carry the
  positioning line, the documentation and site URLs; every GitHub release
  body opens with it; `xyntetik-runner-client` on PyPI carries it with the
  project links.

## v1.3.1 - 2026-10-14

- **`scripts/bench-corpus.py` measures at the context its record states.**
  The corpus driver launched its calibration server with `-c` but every
  `--bench-json` measurement without it, so each bench row ran at the
  runner's default context (the model's trained length, at most 4096) while
  the record said `"ctx": --ctx`. The KV allocation and the GPU split both
  follow the context, so a context-sensitive comparison from 1.3.0's driver
  measured something other than what it reported. One argument builder now
  serves every launch, and a row whose own reported `context` differs from
  `--ctx` is recorded as an error, not a measurement. The idle case's
  command is quoted for the parser that reads it, so a path with a space
  works. The engine is unchanged: 1.3.1 and 1.3.0 differ in this script,
  its test and the version string.

## v1.3.0 - 2026-10-14

- **A model on a CUDA GPU no longer stays resident in host RAM.** Loading
  copies the offloaded weights out of the mapped model file, and every page
  that copy touched stayed in the process's resident set: a fully offloaded
  Qwen2.5-7B Q4_K_M server held 4.86 GB of host RAM after a request. Once the
  device is ready, the host pages of what it holds are handed back: 0.32 GB
  for the same server, and Llama 3.2 3B on an RTX 3070 under Windows went
  from 2,108 to 191 MiB. Output is unchanged, and a CPU fallback reads the
  same bytes back from the file. The load itself still reads the whole file
  once, and `--mlock` keeps every page resident as before.

- **A number in a function-XML tool parameter can end.** With a required or
  named tool choice, a parameter holding a number (`<parameter=a>` then
  `13`) could not be followed by the newline that starts its closing tag, so
  the model kept writing digits: "use add to add 13 and 29" on Qwen3.8 27B
  came back as `a: 1.3e+101`. The newline now ends the number.

- **A Qwen2.5 model gets the system prompt its template writes.** A request
  with no system message reached a Qwen2.5 model without the default its own
  chat template supplies ("You are Qwen, created by Alibaba Cloud. You are a
  helpful assistant."), 21 tokens short of the reference, and with tools
  declared the default was missing before the tool declarations too. Both
  are written now, as they already were for SmolLM2, Hermes 4 and Granite
  4.0-H. System-less Qwen2.5 prompts are longer than in earlier releases,
  so their output can differ.

- **`runner_telemetry.closure` names the fields the closer wrote on
  XML-style calls.** When the token budget runs out inside a native tool
  call, the closer completes it. Its provenance listed the fields it
  synthesized only for JSON calls and read `"fields": null` for Qwen3-Coder
  XML and Muse atem calls; it now names the call's `/name` and
  `/arguments/KEY` there too.

- **`RUNNER_DEBUG_ROUTE=1` and `--bench-json` say which route a forward
  took.** With the variable set, every forward prints, after it ran, the
  backend it ended on, the device, the split, experts on the host, rows,
  path, KV type, weight types, graph and tensor-core state, the fallback
  reason and the build. `--bench-json` carries the same as `route`, with
  every path the run took and how many forwards each served.

- **On Windows, a server with no console behind it no longer raises a
  tray.** Any character device counted as a terminal, so a server started
  with stdin on NUL (a scheduled task, a service wrapper) looked
  interactive and raised one; a model download from such a process also
  drew its progress bar into NUL. A terminal on Windows is now a console.

- **Every `--parallel` slot gets the CUDA split the first one picks.**
  The split was planned for one slot: the first filled the card with
  layers, and the next was then refused its VRAM at startup, or found no
  room for its scratch and served from the CPU, answering the same request
  with different logits from the slot beside it. The plan now charges
  each slot's KV cache and scratch and the weights once, so a server with
  `--parallel N` may put fewer layers on the GPU than one slot would, and
  the split line says how many slots it planned for. `--fit --parallel N`
  answers for the same N slots. `--fit` on a discrete GPU also bills the
  whole KV cache to RAM, offloaded layers included: the host copy is the
  authoritative one and is written back after every device forward, so
  the RAM figure it reported was short by the offloaded layers' cache.

- **A server seed makes unseeded sampled requests reproducible at any
  `--parallel`.** Each slot kept its own sampling stream, and which idle
  slot serves a request is the scheduler's timing, so two runs of the same
  requests under the same `-s` seed could sample differently with two or
  more slots. Every request that samples now takes the next place in one
  server-wide order, and an unseeded one draws from the server seed and
  that place. A request's own `seed` still wins, and a greedy or refused
  request still leaves the next request's draw alone. Unseeded sampled
  output under a given `-s` differs from earlier releases.

- **A loaded model that sits idle no longer costs a pass over its file.**
  The provenance endpoint's model digest was taken in the background after
  every load, reading the whole file: 6 to 11 CPU seconds in the first idle
  minute on a 3 to 4 GB model, and the file pulled through the page cache. It
  is now taken when something first needs it (`GET /v1/runner/provenance`, an
  `expect_resident` digest), and at load only when the server runs with
  `--receipts`, `--sessions` or `--kv-snapshots`, which need it on a request's
  path. A reload or swap back to the same unchanged file reuses its digest.
  The first idle minute after loading Qwen3 4B Q4_K_M on an Apple GPU went
  from 6.25 to 0.01 CPU seconds. The route's states are unchanged.

- **A model whose chat template is not recognised reports it as
  `llama2-fallback`, not `raw`.** Such a model is rendered with Llama-2
  markup, but `--caps`, the doctor report, the server's capabilities and
  receipts named its template `raw`, the template that adds no framing at
  all. A receipt therefore claimed an unframed prompt for a framed one. They
  now say `llama2-fallback`. `--chat-template` still does not accept the
  name (only detection produces it), and replay is unaffected.

- **A partial CUDA split of a small model uploads its weights in one
  block again.** A partial split switches to one upload per tensor when
  the file's tensor order spreads the offloaded layers across the file.
  The check counted the GGUF header as spread, so on a small model with a
  large vocabulary the header alone tipped it, and the weights were
  uploaded tensor by tensor for no reason (each upload rounded to the
  driver's 2 MiB granularity). It now counts tensor bytes only.
  `tests/test_gpu_split_order.py`, which failed on every CUDA host, passes.

- **`--cpu-moe` on CUDA runs Granite 4 hybrid MoE models on the GPU.** The
  CUDA backend kept four lists of a layer's device tensors, one each for
  the split plan, the contiguous upload, the per-tensor upload and the
  type check at load, and none had the Mamba-2 input projection
  (`ssm_in`). Under `--cpu-moe` the first recurrent layer found its
  projection missing, the forward failed, and the run continued on the
  CPU. The split plan also never counted those bytes, and the contiguous
  upload's reach left out the hyper-connection mixers and the shared
  expert. One list now serves all four, and `tests/test_cuda_tensor_roles.py`
  holds every tensor a layer binds to a device role. With `--cpu-moe` or
  `--cpu-moe auto`, granite-4.0-h-tiny and h-small now run attention and
  Mamba-2 on the GPU with the experts on the host (their shared expert has
  no gate, which the device cannot run, so an auto fit keeps every bank on
  the host). Granite 4 hybrid and Nemotron-H splits now budget the input
  projections, so a tight card can fit one layer fewer than before.

- **`--cpu-moe auto` on CUDA no longer crashes on a MoE model whose shared
  expert has no gate.** The device runs a shared expert only with its
  scalar gate (Qwen 3.5 and 3.8). The check that keeps an ungated one (the
  DeepSeek shape, which the llama architecture admits) on the host ran at
  load, before the auto fit decided where the expert banks go, and the fit
  then placed them on the GPU, where the forward read the absent gate and
  the process died. The fit now keeps such a bank on the host, and the
  advice `--cpu-moe` prints counts only banks the GPU can run.

- **`RUNNER_TRAIN_GPU=1` trains a MoE adapter to the CPU's bytes.** The
  device side of the backward's transposed matvec caches each weight it
  uploads, and it knew a weight by its descriptor's address. A routed
  expert reaches the backward as a view of the fused expert tensor, built
  for each call, so every expert of a layer arrived at the same address and
  the first expert's weights served every later expert: the MoE adapters'
  gradients through the expert down projections were wrong whenever the
  CUDA assist was on (dense models were unaffected). The cache now knows a
  weight by the bytes it holds, and the MoE adapter fixtures are held to the
  CPU's bytes on CUDA hosts.

- **`--cpu-moe` on CUDA runs each MoE layer's FFN exactly as the CPU forward
  does.** With the experts on the host, the device hands each MoE layer's
  residual to a host function that runs the FFN and its tail. That function
  was a copy of the forward's FFN, and on x86-64 it compiled differently from
  the CPU forward, so `--cpu-moe` results on CUDA move slightly: gpt-oss 20B's
  teacher-forced mean NLL by 3e-5 and Qwen3 30B-A3B's by up to 1e-3 with
  every expert layer on the host. The copy also lacked two later families'
  rules (Granite 4 hybrid's residual multiplier, Nemotron-H's gate-less shared
  expert), but neither family reaches it in this release: both run on the CPU
  under `--cpu-moe`. The host path now runs the forward's own FFN stages, and
  `tests/test_moe_host_bridge.c` holds it equal, bit for bit, to the CPU
  forward on every MoE layer of the fixtures.

- **`runner -i` shows a reasoning model's thinking as thinking.** The chat
  loop started its reasoning split inside the thought only for Harmony,
  whatever the thinking mode. Qwen 3.5 and 3.8, Granite 4.2, Nemotron and
  Ornith, whose prompts open `<think>` themselves, printed their reasoning
  as the answer, and Harmony under `--no-think` started primed. The chat
  loop now reads where the rendered prompt left the turn, as the server
  does.

- **A Gemma 4 turn that continues after a tool result with thinking on is
  read from the thought its prompt opened.** The prompt ends by opening
  `<|channel>thought`, so the model's first words are its thought. The
  check meant to notice that compared the raw prompt's last bytes, which
  end in an ownership mark, and never matched. The tool grammar therefore
  started outside the thought, and the stream demultiplexer and the
  buffered mapper waited for an opener that would never come, so a
  continuation's thought and its next call could not be read as reasoning
  and a call. All three now start inside the thought the prompt opened.

- **A Messages request's `thinking` object now decides where its reply
  starts.** The prompt was rendered under Anthropic's own `thinking: {"type":
  "enabled" | "disabled"}`, but where the reply starts, inside a thought
  block or in content, was rebuilt from `enable_thinking` alone. So
  `thinking: {"type": "disabled"}` on Granite 4.2, Nemotron 3.5, Qwen 3.8,
  Qwen 3.5, Ornith or gpt-oss returned the answer as a `thinking` block,
  and `{"type": "enabled"}` on Qwen 3.5-0.8B or Muse returned the
  reasoning as `text`. The renderer now reports the start
  of the turn it wrote, and the reasoning split, the think budget and the
  tool grammars read that report on every surface. A Harmony turn whose
  tools are declared but not enforced starts in content, because its prompt
  leaves the channel to the model.

- **A request's recipient or effort bits no longer change another family's
  thinking.** The thinking argument carries the mode with the effort bits
  (read by Qwen 3.8 and Muse) and the recipient bits (read by Muse) above
  it, and a request's `bare_recipients` is added for every family. Six
  renderers compared the whole value with `THINK_OFF` or `THINK_ON`, so
  `bare_recipients` with thinking off rendered an open thought block for
  ChatML-think, Granite 4.2 and Nemotron 3.5 Lightning and primed Harmony's
  analysis channel, and an effort bit with thinking on dropped Gemma 4's
  `<|think|>`. They now compare the mode bits only; every render of a
  family that does not read a bit is identical with and without it.

- **A Responses turn that hit an allocation failure no longer reports
  `completed`.** A generation fault is a turn with nothing usable behind it,
  and chat (`finish_reason: "error"`) and Messages (an `error` object, or
  the `error` event on a stream) already said so. Responses only treated an
  unmappable tool envelope as a fault, so an allocation failure
  mid-generation came back `status: "completed"`, and on a stream the turn
  ended on `response.completed`. Both faults now report `status:
  "incomplete"` with `incomplete_details.reason` naming the fault
  (`envelope_unmapped` or `out_of_memory`), the message item is
  `incomplete`, and a stream ends on `response.incomplete`.

- **A refused `--serve` no longer leaves a tray behind.** The tray is
  raised only after every argument check a serve or interactive run can
  still fail, which is what the code already said, but the serving
  configuration (`--receipts-keep` without `--receipts`, `--adapter` with
  `--lora`, the `--kv-snapshots` and `--sessions` checks), the rewrite flags
  and `--require-eval` were checked after the raise. A refused run on a
  terminal left a detached tray running, holding the executable open
  against the next relink. The raise now follows the last of them; a run
  that fails later, when the model will not load, still raises one.

- **Shadow's verification replay runs in the recorded directory on
  Windows.** `capture_report.reproduce_verification` ran the recorded
  command in a login shell and set the directory on the process, but
  MSYS2's login hook moves to `$HOME` whenever `SHLVL` is at most 1, which
  is every shell started from cmd, PowerShell or make. The command then ran
  in the home directory, failed to find its files (exit 127) and was
  reported as an instrument error instead of reproduced or not reproduced.
  The shell now enters the recorded directory itself after its startup
  files run, so no profile can move it.

- **Byte-level BPE tokenization matches the reference on Latin-1 symbols,
  superscripts, currency and U+FFFD.** The pre-tokenizers read every code
  point above U+007F as a letter unless it sat in a few symbol blocks, so
  `°`, `±`, `§`, `«`, `¿`, `×`, `€`, `²`, the replacement character `�` and
  their neighbours joined or split pre-tokens where every family's reference
  regex does not: `°)` became two tokens on Kvist-14B and gpt-oss where the
  reference emits one, and `.txt` after a `"�` became one where it emits
  two. U+0080..U+00FF and U+2070..U+20FF are now classified from the Unicode
  database (letters, numbers, whitespace, everything else) and the Specials
  block (U+FFF0..U+FFFF) is a symbol range. Against the Hugging Face
  reference tokenizers of seven families (Kvist-14B, gpt-oss-20b, Llama-3.2,
  Qwen2.5, SmolLM2, Mistral Nemo, Granite 4.1): 24 to 36 divergences per
  family on 1,642 strings over these characters before, none after; the 13
  Kvist-14B requests of the typed-decisions study that differed now agree;
  the 721-string corpus is unchanged. Token ids change for text containing
  these characters on every byte-level BPE model.

- **A refused request no longer reseeds its slot.** A request that carried a
  `seed` and was then refused with a 400 (a bad `max_tokens`, `stream` or
  `stop`, an unknown `response_format`, a prompt longer than the context)
  had already installed that seed in the slot's sampler, so the next
  unseeded request on the slot drew exactly the refused request's stream
  instead of continuing its own. The seed is now installed where generation
  starts, after the last refusal.

## v1.2.1 - 2026-10-13

- **A client stalled mid-request no longer holds the server's shutdown.**
  A connection that had sent half a request kept its slot reading until the
  10-second request-read deadline, so a SIGTERM or Ctrl-C waited up to 10 s
  for a request nothing had admitted. The read now waits in quarter-second
  slices and gives up when a stop is requested: a stop with such a client
  open finishes in about a second. The stop flag and the listener descriptor
  are now atomics, which removes a data race ThreadSanitizer reported
  between a stop requested from another thread and the accept loop.

- **Tools are no longer dropped from Llama 2, Gemma and Mistral v0.1
  prompts that carry a system prompt.** The generic tool envelope puts its
  teaching turn (the tool declarations and the call format) in front of the
  caller's system message, and these three framings fold only one system
  text into the first user turn: they kept the caller's and discarded the
  tools. A request with a system prompt and tools rendered the same prompt as
  one without tools (Llama 2: 31 tokens either way), and only the grammar
  forced a call out of a model that had never been shown the tools. The
  leading system messages are now joined, the envelope's first, a blank line
  apart (Llama 2 now 214 tokens, Mistral v0.1 207, Gemma 219, on Qwen2.5's
  tokenizer). A single system message renders exactly as before.

- **Mistral v0.3 and Mistral-Nemo speak their own tool protocol.** Their
  tools rode the generic JSON envelope, whose teaching turn the Mistral
  renderer dropped whenever the caller also sent a system prompt: the model
  was then never shown the tools, and only the grammar forced a call out.
  Runner now declares tools in the publisher templates' `[AVAILABLE_TOOLS]`
  block before the last user turn, replays calls as `[TOOL_CALLS]` lists and
  results as `[TOOL_RESULTS]` blocks with 9-character call ids (any other id
  maps to one by its hash, call and result alike, on the Chat, Responses and
  Anthropic surfaces), and constrains and parses a turn's `[TOOL_CALLS]`
  list, buffered and streamed. The marker is a control token, so `auto` is
  constrained too: prose, or a hand-off to the list. Template conformance:
  both families clean in text and tokens. Agent-torture on the real Q4_K_M
  files: 40 of 40 on both arms for both models (suite R2.4.3). The v0.1
  framing stays generic.

- **Mistral v0.3 files with the publisher's old template get the v0.3
  framing.** v0.3 GGUFs converted before the publisher replaced its
  template embed the v0.1 form (`' [/INST]'`), and detection served them
  v0.1's framing, which the model was not trained on, and no tool
  protocol. A v0.1-form template over a vocabulary holding
  `[AVAILABLE_TOOLS]` (a v0.3 token; v0.1 and v0.2 have none) now detects as
  `mistral`, so the pinned v0.3 file is served its own framing and native
  tools without `--chat-template` (suite R4.22.7).

- **Nemotron Nano speaks its own tool protocol.** Its tools rode the generic
  JSON envelope, which also replaced the caller's system prompt with the
  envelope's instructions; the template's own `<AVAILABLE_TOOLS>` list and
  `<TOOLCALL>[...]` calls were the last open rows of the template
  conformance backlog. Runner now renders the declarations after the
  caller's system text and replays calls in NVIDIA's form, parses a
  `<TOOLCALL>` block back into `tool_calls`, buffered and streamed, and holds
  a required or named turn to one entry (one to eight when parallel) with a
  grammar; an auto turn is the model's own, parsed. Template conformance:
  20 of 20 cases identical to NVIDIA's template in text and tokens.
  Agent-torture on the real 9B Q8_0 file: generic 39 of 40, native 39 of
  39 with one case excused (suite R2.4.4).

- **`/v1/completions` takes `prompt` as an array of token ids.** OpenAI's
  API, vLLM and llama.cpp accept it; Runner answered "missing prompt". The
  ids are used exactly as sent (no BOS added, nothing re-tokenized), so a
  receipt records them and `--verify` replays them. A request sent as text
  and the same request sent as the ids its receipt recorded leave identical
  token lists and identical greedy output. An empty array, an id outside
  the vocabulary, a non-integer and a batch are refused with a 400. Found
  while probing Muse's unconstrained output (suite R4.26.19(e)).

- **Muse: a parallel tool turn carries one to eight calls in the model's
  own form, and an absent `parallel_tool_calls` now means several there
  too.** The parallel grammar was a fixed pair joined by a bare
  `assistant`, which refused the model's own separator (`<|eom|><|start|>`,
  control tokens admitted only by their spelling) and pushed it to type a
  turn header as text. On Muse-Glimmer-30B (CPU, temperature 0, `required`)
  it then gave two calls every time, the second a repeat of the first city
  in 5 of 6 turns. The separator is now spelled in the grammar, and the turn
  ends at the model's `<|eot|>` or opens the next call, up to eight. The same
  prompts now give one call: unconstrained, the model writes "one at a time"
  in its reasoning and calls once per turn, so one call is its own choice.

- **A lineage walkthrough on a real model (R17.1.5).**
  `docs/lineage-walkthrough.md` runs one chain on Qwen3-0.6B: quantize to
  Q8_0, train a LoRA, merge, record a fidelity evaluation, serve with
  receipts, then `--lineage` from the file, from the receipts and from a
  copy with one changed byte, every output as printed. It found that a
  Q8_0 base merged back into Q8_0 keeps only 18.2% of a 20-step adapter's
  delta; the merge refusal's advice, which named q8_0 as a remedy, now
  points to a type wider than the base's own (f16). The README's "Prove
  what a model did" and the site's receipts page gain a sentence on
  lineage.

- **Gemma 4: a parallel tool turn carries one to eight calls, and an absent
  `parallel_tool_calls` now means several there too.** The native grammar
  compiled a parallel turn as a fixed pair: on Gemma 4 E4B Q4_K_M (CPU,
  temperature 0, `required`), a prompt needing one city's weather got
  `Oslo, Oslo` and one needing three lost the third. Each call is now
  followed by the model's own stop or the next call, up to eight: the same
  prompts give one, two and three calls, thinking on and off. Gemma 4 ends
  a turn of calls with `<|tool_response>`, which Google's generation_config
  lists as end of generation (id 50, beside `<turn|>` and `<eos>`); Runner
  now stops on it too, where unconstrained output could run past a call.
  The calls share the grammar's expansion budget, so a tool whose
  arguments fit a pair but not eight copies steps the cap down (8, 4, 2,
  1) instead of failing. Muse keeps its fixed pair and the one-call
  default for now (suite R4.26.19(b)).

- **A required tool turn with thinking on keeps the model's own blank line
  after `</think>`.** On the Qwen JSON and function/parameter XML protocols
  a required or named turn opened on the literal `<tool_call>`, so the
  newline the template itself writes after a closed thought
  (`</think>\n\n`) was vetoed and the model was pushed straight to the
  opener, off its own distribution (with `auto` the newline was already
  admitted). The first call now admits up to two newlines ahead of its
  opener and nothing else: no text, no third newline, and still no turn
  without a call. Found with a scripted reply, which the veto ended with no
  call at all (suite R4.26.19(d)).

- **`--merge-lora` measures what the merge kept, and refuses a merge that
  rounded the fine-tune away.** Each adapted row is also written from the
  base alone at the same output type; the projection of the difference onto
  the intended delta (`delta_retained`) and the adapted bytes that differ
  are printed and recorded under `survival` in the `.merge.json` record. A
  merge that keeps under 50% of the delta is refused with the destination
  untouched; `--merge-allow-erased` writes it anyway. Calibrated on the
  merge study's Qwen3-4B Q4_K_M and tool-use adapter: merged back onto Q4_K
  it kept 1.8% of the delta (3.0, 4.7 and 8.9% at 2, 4 and 8x scale), into
  Q8_0 99.5%. A merge onto a different 4-bit grid (Q4_K_M into `q4_0`) keeps
  the delta on average; its effect on behaviour is not measured. The
  study's 8x merge, which scored like the adapter only because the grid cut
  an overshoot, is now refused by default.

- **Nemotron 3.5 Lightning gets its own chat template family,
  `nemotron35`.** Its template is Granite 4.2's with Qwen3-Coder's
  nested-XML tool declarations, and Runner detected and served it as
  `granite42`: a request with tools declared them as JSON inside `<tools>`,
  a format the model was not trained on, and telemetry named the family
  and sampling preset `granite42`. It now writes the declarations as the
  file's template does (`<function><name>`, `<parameters>`, `enum` after the
  description, `required` after the other keys), folds `reasoning_content`
  as `<think>\n...</think>` and keeps the text after `<think></think>` on a
  truncated tool-call turn untrimmed, as that template does; calls are
  parsed as before. Its sampling preset `nemotron35` is NVIDIA's
  generation_config (temperature 1.0, top_p 0.95), the same numbers it got
  before. Template conformance against the file's own template: 22 of 22
  cases identical in text and in tokens on the real 30B file, where the
  four tool cases drifted. Found by the lab (suite R4.22.5).
- **Granite 4.2 trims the text of a truncated tool-call turn.** On an
  assistant turn with tool calls before the last user message, Granite
  4.2's template trims the text after `<think></think>`; Runner kept its
  leading whitespace. Pinned against the template's own render, with and
  without a thought block.

## v1.2.0 - 2026-10-09

- **A request that leaves `parallel_tool_calls` out now gets several calls
  per turn, as OpenAI's API does (owner, 2026-10-09).** Runner used to read
  an absent flag as false, and on the Hermes-style JSON families (Qwen 2.5,
  Qwen 3) every tool turn is constrained, so any client that omitted the
  flag, `tool_choice: auto` included, got at most one call per turn; on
  required turns it held the XML families to one as well. The same rule
  applies to `/v1/responses` (whose echo now reads `true` when the request
  is silent) and to `/v1/messages` (`disable_parallel_tool_use` absent means
  allowed, Anthropic's default). Exception: Gemma 4 and Muse, whose parallel
  grammar is still a fixed pair of calls, keep one call when the flag is
  absent. A `tool_choice` that names one function also keeps one call when
  the flag is absent, on both OpenAI surfaces: OpenAI's forced function calls
  exactly one (Anthropic's named `tool` choice is not changed, its default
  allows several). An explicit value always wins. The conformance tests that pin the
  one-call path now say `parallel_tool_calls: false` (or
  `disable_parallel_tool_use: true`) explicitly; new tests pin the default.

- **`--train` on sparse-MoE models: an attention-only adapter (R17.3).**
  Qwen3-MoE and Mixtral routing (softmax router, top-k, with or without
  renormalization and a weight scale) train now: the adapter sits on the
  attention projections, and the backward recomputes the forward's routing
  and runs through the frozen experts and the router's softmax weights.
  Pinned by a strict finite-difference gate on two router-sensitive
  fixtures (`make-test-moe.py`'s `moe4-train` and `moe4-train-nonorm`,
  worst per-coordinate error 0 against a 0.5% limit; dropping the router
  term reads 1.3 to 2.0% and fails), and end to end by
  `tests/test_moe_lora.py`: the loss falls, a rerun is byte-identical, the
  adapter serves back, a merge scores exactly like base plus adapter. Still
  refused by name: adapters on the experts, sigmoid, group-limited or biased
  routers, shared experts. `make-test-lora.py` builds attention-only
  adapters for bases without dense FFN tensors.

- **Repository cleanup.** The 17 raw records of the `tooluse-shifted` eval
  (122 MB, about half of a checkout) moved to the
  [Qwen3-4B-tooluse-shifted-eval-record](https://huggingface.co/datasets/Joakimpalm-Zen/Qwen3-4B-tooluse-shifted-eval-record)
  dataset, checksums verified after the upload; the instrument, its sets and
  its summary stay in `evals/tooluse-shifted/`, and new records there are
  ignored by git. The fixture scripts (`make-test-*.py`,
  `make-tooluse-data.py`, `tokenizer-corpus.py`) refuse an option where the
  output path goes (`--help` used to write a set of `--help.*` files).
  `docs/README.md` indexes every document and evidence folder by topic.
  `scripts/outcome-table.py`, used only by the comparison with another engine, is gone.
- **`--lineage FILE`: where a model file came from, checked (R17.1).**
  `--quantize` (and its prune, sublayer-removal and type-plan forms) now
  writes a provenance record beside its output, `OUT.gguf.quant.json`, as
  `--merge-lora`, `--train` and `--context-surgery` already did; every
  record's inputs link the record that produced them (path and sha256), and
  `--sign-key` beside any of these steps signs the record in place.
  `--lineage` walks a file's chain back to its origins: each link re-hashed,
  each signature checked (`--trust-key` pins the signer), exit 0 verified
  and signed, 1 consistent but not all signed, 2 broken or no record. The
  records alone verify after the model files are gone, copied together to
  another folder. `tests/test_lineage.py` builds a train, merge and quantize
  chain and checks a signed chain, an unsigned one, an edited record, a
  swapped file, another signer, a file with no record and a records-only
  copy.
- **A receipt links the model's and adapter's records, and `--lineage`
  starts from it (R17.1).** A `--transcript` record and every `--receipts`
  record of a served answer add a `record` link (path and sha256) to the
  `model` and `adapter` objects when a step made that file, so `--lineage
  run.json` walks from an answer back to the base model and the training
  data, checking the receipt's own signature first. Replay (`--verify`) is
  unchanged.
- **Evaluations that travel with the file (R17.2).**
  `scripts/eval-record.py fidelity` (the file against a reference through
  `kld-compare-raw.py`) and `scripts/eval-record.py agent` (an agent-bank
  run's evidence, with multi-call turns from the attempt transcripts) write
  `<model>.eval.<kind>.json`: subject and inputs by sha256 with their
  lineage links, method, numbers, thresholds and pass, signed with
  `--sign-key`. `--lineage` shows a file's evaluations (STALE when the file
  changed since), and `--require-eval KIND` refuses to load a model without
  a passing, current one, signed by `--trust-key` when given.
  `tests/test_eval_record.py`.
- **`--lineage DIR` on a `--receipts` folder: which version answered what
  (R17.4).** The answers grouped by model and adapter sha256, each version's
  count and time span, the order they served in (a rollback reads as a
  version served again), the receipt chain's continuity and every receipt's
  signature, then each version's lineage once.

- **`scripts/decide-calibrate.py` counts both splits the same way and can
  emit per-question predictions.** The held-out group count used
  permutation groups while the train count used split keys, so the two
  disagreed on any question file with `split_group` pairs; both now count
  the split unit, with `n_train_permutation_groups` and
  `n_holdout_permutation_groups` beside them. `--emit-predictions FILE`
  writes one JSONL record per question with its probabilities, logprobs and
  label, which the aggregate report never kept. Found by the lab while
  drafting its typed-decisions baselines (TD-000).
- **A gate on the agent rule files.** On 2026-10-06 an edit meant to
  replace one paragraph of AGENTS.md inserted it after every character of
  the file (372 lines became 150,523, every rule cut apart) and nothing
  noticed for two days. `scripts/check-agent-docs.py` fails on a rule file
  past 800 lines, a paragraph that occurs twice, or a long line repeated;
  it runs in `make test` and in the commit-hygiene workflow, which every
  pull request waits for, docs-only ones included. Its self-test includes
  the shredded shape.
