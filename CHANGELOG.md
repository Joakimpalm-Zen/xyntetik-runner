# Changelog

Current release and unreleased changes. Compatibility follows the
[versioning policy](docs/versioning.md).

## Unreleased

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
  Training still covers the models it covered: qwen35 and Clef refuse it.

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
