# spark2_5 on Metal and CUDA (2026-10-30)

The GPU backends refused XHToken's Spark X2.5 and X2.6 (`spark2_5`) by name for two pieces. Both are now device
kernels on Metal and CUDA. The rest of the architecture already ran on the device: Q, K and V as row slices of the
fused `attn_qkv`, two rope widths and bases, and the sliding window on three of every four layers.

- **The per-head output gate.** `blk.N.attn_gate` projects the normed input to one value per head, and
  sigmoid(g[h]) scales that head's outputs before `attn_output`: `k_head_attn_gate` (CUDA), `k_head_sigmoid_mul`
  (Metal). The gate projection is sized `n_head`, not `q_dim`.
- **The exact GELU of the gated FFN,** 0.5 * g * (1 + erf(g / sqrt 2)) * u: `k_gelu_erf_mul` on both. **The erf is
  not the same function on every backend.** The CPU uses libm's `erff` and CUDA its own `erff` (the PTX is built without
  fast math). Metal's standard library has no erf (a runtime compile of `erf(x)` fails: "use of undeclared identifier
  'erf'"), so Metal computes it by Abramowitz and Stegun 7.1.26, with absolute error at most 1.5e-7. Inside the GELU that
  is below float rounding relative to the result. The identity numbers below include it.
- An expert FFN with the exact GELU stays on the CPU on both backends: the MoE kernels' activation selector does not
  know it, and no such model exists.

## GPU against CPU at logit precision

`test-gpu-identity`, 24 teacher-forced positions; the limit is 0.002 of the mean logit range.

| Fixture or file | Metal (Apple M1) | CUDA (RTX 3070) |
|---|---|---|
| spark fixture, batch 32 | 2.1e-06 mean \|dlogit\|, worst 1.6e-05 | 3.3e-06, worst 2.4e-05 |
| the same, one token at a time | 2.0e-06, worst 1.9e-05 | 3.9e-06, worst 2.8e-05 |
| every layer sliding | 1.9e-06, worst 2.2e-05 | 4.8e-07, worst 1.2e-05 |
| quantized to Q8_0 | 7.1e-06, worst 4.3e-05 | 6.4e-06, worst 6.1e-05 |
| quantized to Q4_0 | 2.5e-06, worst 2.0e-05 | 1.9e-06, worst 2.0e-05 |
| Spark-X2.6-4B Q4_K_M (XHToken) | 7.5e-04 = 1.9e-05 of range 38.8, worst 0.0115 | 0.00118 = 3.0e-05 of range 38.8, worst 0.0135 |
| Spark-X2.5-4B Q4_K_M (XHToken) | | 0.000898 = 2.3e-05 of range 39.9, worst 0.0099 |

Every CUDA line ran with the whole model on the card (36 of 36 layers on the real files).

## Cells

`scripts/tier-smoke.py --pin` then `--record` on each row's own file:

| Row | Metal (Metal laptop) | CUDA (Windows CUDA host) |
|---|---|---|
| `spark-x2.6-4b-q4_k_m` | pass, 19.1 s, fact found | pass, 2.99 s |
| `spark-x2.5-4b-q4_k_m` | pass, 7.43 s, fact found | pass, 2.41 s |

Each GPU reply is byte-identical to the row's CPU pins on both hosts: X2.6 sha256 `42a4c173...`, X2.5 `b872096a...`.

## The CUDA kernels' PTX

`tools/ptx-toolchain/make-ptx.sh` with the locked environment (nvcc 13.0.88, conda-forge gcc 15.2.0), on a Linux CPU
host with no GPU, first rebuilt main's committed `src/kernels_ptx.h` and `src/kernels_ptx80.h` byte for byte. It then
regenerated them on the branch. The compute_75 header gains the two kernels, and the rest of its diff is nvcc's branch
labels renumbered after them. The compute_80 header is unchanged.
