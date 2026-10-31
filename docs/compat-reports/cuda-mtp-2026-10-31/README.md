# MTP head consumption on CUDA (R3.3.3), 2026-10-31

Measured on the Windows CUDA host (RTX 3070, 8 GB). Model: a Qwen3.5-4B Q4_K_M that carries its NextN block
(`qwen35`, 32 trunk layers plus one predictor block; Q4_K, Q5_K and Q6_K weights), fully offloaded. One prompt
("Write a Python function that parses an ISO 8601 date string and returns a datetime object, with error handling
and docstring."), 200 tokens, temperature 0, seed 1, `--draft mtp --draft-k K`.

## Status

The plan item asked for head consumption on CUDA after R3.3.4. Both pieces were already on main: the NextN head runs
on the device since 9a42cac9 (2026-10-07), and a fully offloaded target serves the verify walk target-exact. What
held it back was the verify cost, which R3.3.4 (PR #492) addresses. This change corrects the `--help` text and the
two `model.h` comments that still described the head as CPU-only. MANUAL.md already said CPU and full CUDA.

| Arm | main f61f6961 | with PR #492 |
|---|---|---|
| plain | 94.7 tok/s | 94.8 tok/s |
| mtp, k 1 | 87.8 (1.90 tok/round) | 98.8 |
| mtp, k 2 | 98.4 (2.74 tok/round) | 112.8 (1.19x plain) |
| mtp, k 3 | 92.6 (3.12 tok/round) | 107.4 |

Every output is byte-identical to the plain run (sha256 0e7f844c...). Drafting costs about 2.3 ms a drafted token
on the device (338 ms for 146 at k 2). At k 2 the verify forwards take 1405 ms with PR #492 against 1660 on main.

## Where the rest goes (k 2, PR #492)

`RUNNER_CUDA_PROFILE=1`: a three-row verify tile spends 15.4 ms in matvec against 8.3 for a decoded token, and
13% of the tile is the output head (Q6_K, 2560 x 248320). Exact 2- and 3-wide twins for the narrow rounds
measured 10 to 13% faster than the 4-wide class on that head and 0 to 5% on the Q4_K shapes, about 2 to 3% of the
run: noted for later, not done here. R3.3.1 (per-row recurrent checkpoints in the verify batch) is the other lever
on hybrid models.
