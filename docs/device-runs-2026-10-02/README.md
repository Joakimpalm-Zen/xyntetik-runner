# Device runs, 2026-10-02

What ran on the lab's RTX 3070 box (Windows 11, i7-7700K, CUDA 13.3) against
the unreleased tree on 2026-10-02. The files here are the machine-written
records; the numbers below are read from them.

| run | result |
|---|---|
| CUDA smoke (`scripts/cuda-smoke.py`) | 13 of 13 checks pass |
| CUDA gate (`scripts/cuda-gate.sh`) | pass: tc-overflow, device i-quants, CPU vs CUDA identity on Qwen3-0.6B Q8_0 |
| int8 decode route (`test-i8-tol`), granite-4.1-8b Q4_0 | 1 of 64 top-1 flips, mean deviation 0.00075 of the logit range: not promotable |
| int8 decode route, Llama-3.2-3B Q4_K_M | 1 of 64 top-1 flips, 0.00055 of the range: not promotable |
| CPU vs CUDA log-probability delta, gemma-4 E4B Q4_K_M | 317 positions, mean 2.3e-4, max 3.7e-3, first over 1e-4 at position 3 |
| CPU vs CUDA log-probability delta, gemma-3 4B Q4_K_M | 322 positions, mean 2.3e-4, max 7.8e-3, first over 1e-4 at position 5 |

## The int8 route stays opt-in

The promotion bar for `RUNNER_CPU_I8=1` is zero flips in 64 teacher-forced
positions on both official rows. On this AVX2 box each row flips one
near-tie (worst margin 0.0009 and 0.0005 of the logit range), as the Zen 5
box did on 2026-08-13. The route is not promoted on either CPU class.

## The gemma-4 E4B flip is a near-tie, not an E-series defect

The 0.5.6 compatibility row for gemma-4 E4B on this card fails the dense
identity check on one deterministic flip (margins 0.00056 and 0.00016 nats).
`scripts/cpu_cuda_delta.py` teacher-forces the CPU's own continuation of that
prompt through both backends. The E-series model and gemma-3 4B, whose row
passes 9 of 9, show the same profile: the two backends differ by about 2e-4
nats on average from the first positions on, under the gate's pins. The flip
therefore sits inside the ordinary CPU-versus-CUDA difference on this card
and happens to land on a tie that narrow; the E-series path (per-layer
embeddings, shared KV) is not an outlier. Whether a dense row may tolerate a
margin-qualified flip is a policy question and is not decided here.

## The Linux CUDA box (Blackwell, MIG 1g.24gb slice)

Run on 2026-10-02 in a window the lab gave this work, on the unreleased
branch `backlog-batch-3` (`06d7d07`) unless noted. Records in `blackwell/`.

| run | result |
|---|---|
| CUDA gate on main `9f066a5`, run by the lab | pass: tc-overflow 12 s, device i-quants 34 s, CPU vs CUDA identity 12 s |
| agent-torture, Granite 4.1 8B Q4_0, the family's own protocol (`granite4`) | 120 of 120, 147.8 s |
| agent-torture, the same file, main's generic JSON envelope | 120 of 120, 120.5 s |
| agent-torture, Granite 4.2 8B Q4_K_M (function XML) | 102 of 120: 15 refused with 400, 3 declined (below) |
| new template families on their real files, CPU | 171 checks pass, 0 fail, 1 note, nine files |
| the same, GPU (Qwen 3.5 4B, Phi-4-mini, Hermes 4 14B, Granite 4.1 8B) | 69 of 69 pass; Hermes 4 on a second run alone (below) |
| agent-torture, Granite 4.2 8B, the branch tip with the function-XML fallback | 117 of 120: the 15 refusals gone, the 3 declined remain |
| CUDA prefill profile, Llama-3.2-3B Q4_K_M, 541 tokens | 667 tok/s; quantized matmuls 85% of GPU time, attention 14% |
| sampled `--json`, Llama-3.2-3B Q4_K_M, CUDA | 39.6 tok/s on main, 123.1 with the new sampler (133 unconstrained) |

**Granite 4.1 keeps its own protocol.** Both protocols pass every request;
the publisher's is about 23% slower over the matrix, because its tool
preamble is longer. The default stays the publisher's (the provider
reference rule), and the generic envelope remains one template flag away.

**Granite 4.2.** The 15 refusals were one defect: under a required tool
choice, a string parameter with a length or pattern constraint cannot be
enforced by the function-XML grammar, and the request was answered 400. It
now uses the generic envelope for that request (`tests/test_xml_schema_fallback.py`).
The 3 declined cases ask for a tool call with `max_tokens: 1`; the model's
one token went to its reasoning channel. That is the model's turn shape
under a one-token budget, recorded rather than changed.

**Hermes 4 on the GPU, and what a second slot asked for.** The first GPU
run of the Hermes 4 14B row failed at server start: the sweep serves with
`--parallel 2`, and slot 1 asked for another 10.2 GB of VRAM beside the
9.8 GB slot 0 held, while this window's Granite 4.2 server occupied the
rest. Run again alone it passed all 18 checks. The second slot does not
upload the weights again (CUDA shares one upload between instances of a
file), but the VRAM claim made before the upload asked for the whole file
anyway and was refused when less than that was free. The claim now counts
a resident upload as already paid, so a second slot asks for its KV cache
and scratch only.

**Where CUDA prefill goes.** With tensor cores on, 794 ms of GPU time for
541 tokens: 671 ms in the quantized matmuls, 109 ms in attention, the rest
under 2%. With them off, 1,967 ms. The next prefill lever is the matmul
tiles, not attention or launches.
