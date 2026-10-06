# qwen4exp admission evidence (R4.26.1)

CPU forward of the `qwen4exp` architecture (Qwen3.8-Flash-Next) anchored
against llama.cpp on the toy fixture `scripts/make-test-qwen4exp.py`
writes (every tensor family of the release file: four hyper-connection
streams, the Engram n-gram PLE block, Gated DeltaNet and full-attention
blocks, routed + shared experts; random weights, seed 0x4e, sha256
b0ec6265eb4998cd...). This is the llama.cpp column; the publisher's fp32
reference is the primary anchor and lands in R4.26.3 on the real file.

## Procedure

Same file on both engines, bytes verified by sha256. Runner:
`runner -b 1 -m test-qwen4exp.gguf --score -p "hello world one two three
four five six" --gpu off -c 128` (57 tokens, 56 scored positions,
`toy-runner-score.json`). llama.cpp build 11433 (commit 50569eb87,
`llama-server`, CPU only via `CUDA_VISIBLE_DEVICES=""`, `-c 128 -t 4 -b 64
-ub 64 -ctk f32 -ctv f32 -fa off`): for every prefix of runner's token ids
one `/completion` with `n_probs` = the whole vocabulary (259), temperature
0, `cache_prompt false`, `ignore_eos`, and `logit_bias [[1, 100]]` so the
sampled token is always valid UTF-8 (llama-server drops
`completion_probabilities` when the sampled token is an incomplete
multibyte sequence; the bias changes the sample, not the reported
pre-sampling distribution, checked at a position that works both ways).
The next token's logprob is read from that distribution
(`scripts/anchor-llama-logprobs.py`, `toy-llama-b11433-score.json`).

## Result

56 of 56 positions compared, max |delta logprob| 3.33e-06, which is
float reassociation noise. The two-engine bisect that got there, by
zeroing one component's output projection on both engines
(`QWEN4EXP_TEST_ZERO`):

| step | finding | max delta before | after |
|---|---|---|---|
| gate | PLE conv kernel dequantized into activation scratch too small for it (overwrote the gated value at every batch size, overran the allocation at `-b 1`); per-row head (`--score`, spec verify) normed the stream mean instead of mixing the streams | chunked != single-shot | batch-invariant |
| bisect: only zeroing `ssm_out` closed the gap | Qwen3.8's Gated DeltaNet gates the normed output with sigmoid(z), Qwen3.5's with silu(z) (llama.cpp qwen4exp.cpp `build_norm_gated`: "the one numerical difference from Qwen3.5's GDN") | 5.12e-02 | 6.22e-04 |
| bisect: only zeroing `attn_output` closed the rest; `freq_base 1e9` closed it too | toy artifact: `rope.dimension_sections [2,2,0,0]` leaves pair 2 with an empty section, and llama.cpp's interleaved M-RoPE then uses the 4th position (0 for text), so that pair is unrotated there and rotated in runner. The release file's `[11,11,10,0]` gives every pair a live position. Toy changed to `[2,1,1,0]`. | 6.22e-04 | 3.33e-06 |

Not covered here: QSA (the toy declares no compress ratio, both engines
run dense attention), CUDA, Metal, quantized tensors, the real file.

## The release file (Blackwell, CPU only, 2026-10-06)

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (two parts, 83.6 GB; part 1 sha256
4c1eb2ce..., part 2 316b46f3...), runner at `c127504`, 16 threads on
CPUs 40-55/104-119, `--gpu off -c 4096 --temp 0`:

- 1.0.2 refused the file: `unsupported type 42` (the routed experts are
  Q2_0, a ggml type newer than the i-quants; now read).
- First load produced "The capital of France is Paris." and then token 0
  ("!") forever: the hyper-connection gate reached -94.7 at layer 42 of
  the third decode step and the unguarded sigmoid overflowed under
  -ffast-math (`RUNNER_DEBUG_ACT=3`, `hc-gate ... min=-94.69`, then
  `inf=2` in the mixed block input). Guarded; see sigmoid_f.
- After the guard, greedy 12 tokens: "The capital of France is Paris. The
  capital of Germany is Berlin. The capital of", identical to llama.cpp
  b11433's CPU-only greedy continuation of the same prompt. Prefill 7.6,
  decode 5.7 tok/s (generic dequantize-per-block dot, no Q2_0 SIMD kernel).

The 200-position p512 anchor and the 64-token greedy comparison on this
file wait for a lane G gap on the Blackwell (filed with the lab queue,
2026-10-06 11:40); the fp32 publisher reference stays R4.26.3's primary
anchor.
