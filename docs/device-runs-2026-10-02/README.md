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
