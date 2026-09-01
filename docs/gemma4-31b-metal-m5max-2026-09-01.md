# gemma-4-31B dense Metal validation — M5 Max, 2026-09-01

Machine-locked validation on Apple M5 Max (18 cores), 128 GB unified RAM,
macOS 26.5. The artifact was `models/gemma-4-31B-it-Q4_0.gguf`,
17,992,313,088 bytes, SHA-256
`031dc1c5fa9c5a0abbf3c39c5173fb2af65f5ac2dc2a090268561d3c72dcd834`.
`scripts/verify-gguf.py` found a complete GGUF v3: gemma4 architecture, 60
layers, 833 tensors (422 F32 / 410 Q4_0 / 1 Q8_0), no trailing or missing
bytes. The digest is recorded as a local artifact digest.

This closes the last uncertified artifact on the 128 GB host's model shelf:
gemma4 **dense** at scale, beside the already-certified 26B-A4B MoE.

`test-gpu-identity` compared the scalar and Metal routes over 24
teacher-forced positions. Of 6,291,456 logits, the mean absolute delta was
0.00719, or **0.000167 of the mean 43 logit range** (limit 0.002); the worst
individual delta was 0.0796. The gate passed.

Cross-engine, per the compatibility program (`scripts/compare_llamacpp.py`
against llama.cpp `010be968` llama-server, greedy, 64 tokens from the default
prompt): **64/64 shared tokens — token-identical** — with a maximum common
top-logprob delta of 0.1445. The report is under
`tests/compatibility/out/gemma4-31b-m5max/`.

A sustained Metal run read the determinism note as a 729-token prompt at
batch 512 and generated 128 greedy tokens:

- prefill: 107.88 tok/s;
- decode: 19.03 tok/s;
- swap before and after: 1.12 MB, unchanged (zero growth).

Honest output note: on this raw `-p`/`-f` completion (no chat framing, greedy,
repeat penalty inactive at `--temp 0` by design), the instruction-tuned 31B
degenerates into a token loop after a coherent prefix. The degeneration is the
model's own raw-completion behavior, not a backend defect: the CPU path
produces the byte-identical loop, and the llama.cpp comparison above is
token-identical over its 64-token window. Chat-framed use is unaffected.

## Tied-V evidence (LAB-KV2)

The 31B is the first local model carrying the V-less condition: exactly **10
of 60 layers** (5, 11, …, 59 — every 6th, the full-attention globals) ship no
`attn_v.weight`, verified by reading the GGUF tensor index directly. That
unblocked the tied-V cache work; see `RUNNER_TIEDV` in the README and
`tests/test_tiedv.py`. Measured here:

- derivation check (`RUNNER_TIEDV_CHECK=1`, flat cache still storing real K):
  derived rope(V·w) rows agree with the stored K rows at **99.95–100.00% per
  f16 row** (most rows 100%), worst relative f32 error 1.7e-4 — pure
  rounding, no structural disagreement;
- K-cache allocation at `-c 32768`: **14,763,950,080 → 13,421,772,800 bytes**
  (14.76 → 13.42 GB), reproducing the lab measurement bit-for-bit;
- teacher-forced fidelity (`--score`, 297 positions of the determinism note,
  CPU): 99.0% of individual logprobs move, mean |Δlogprob| **1.33e-3**, max
  2.2e-2, `nll_mean` 8.091037 → 8.090950 (Δ 8.7e-5), absolute top-1 count
  unchanged at 39/297. The perturbation is fp-rounding-shaped, exactly the
  E10 regime: nearly every logprob moves, none of them far. No house-bar
  verdict is claimed and none may be read from a `--score` delta.
- scale, from the lossy alternative on the same text: a q8 KV cache costs
  mean |Δlogprob| 8.13e-2 (max 1.22) and moves `nll_mean` by 4.7e-2 — the
  tied-V perturbation is **61× smaller** than the cache format users already
  accept, while q8 saves ~47% of both caches and tied-V saves 9.1% of one.
