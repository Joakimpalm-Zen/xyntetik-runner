# `--train-gpu` on a sparse-MoE model: Qwen3-30B-A3B (R17.3.3), 2026-11-02

**No speedup on this device.** On a 24 GB MIG slice, a `--train-gpu` step took 52 to 55 s, against 53 s for the
CPU trainer on 16 threads of the same host. The step is device-bound. What this record does show: the whole 30B
MoE model fits on the slice and trains there, its losses match the CPU trainer's, and the gradient gates hold in
device mode on the MoE fixtures.

## Setup

- **Device:** NVIDIA RTX PRO 6000, one MIG 1g.24gb slice (sm_120). **Host:** Threadripper 9980X, Linux x86-64,
  16 threads on cores 40-55. The host is shared, so CPU timings move with its load.
- **Build:** runner main 32163d9, Clang 22, CUDA build.
- **Model:** `Qwen/Qwen3-30B-A3B-GGUF` revision e4d4bafdfb96a411a163846265362aceb0b9c63a,
  `Qwen3-30B-A3B-Q4_K_M.gguf`, 18,556,685,824 bytes, sha256
  0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5 (checked after the fetch).
- **Run:** the first 6,000 bytes of `tests/fixtures/gold-corpus-2k.txt`, 3 steps, `--train-ctx 256`,
  `--lr 1e-4`, `-s 7`, rank 8, alpha 16; adapters on the attention projections (the MoE rule in `docs/lora.md`).

## Fit

The products record of the `--train-gpu` run: budget 24,585,568,256 bytes (the free memory), 16,054 tensors and
15,889,434,624 bytes on the device, 0 on the CPU, 144 attention calls on the device and none failed.

## Losses

| Step | `--train-gpu` | CPU trainer | Relative difference |
|---|---|---|---|
| 1 | 3.003649 | 3.001648 | 6.7e-4 |
| 2 | 4.530073 | 4.530299 | 5.0e-5 |
| 3 | 3.777729 | 3.783909 | 1.6e-3 |

The device products add in another order, so the adapter matches within rounding, not byte for byte (the same
contract as on dense models). Two more `--train-gpu` runs from the same seed repeated the first steps' losses
exactly (3.003649 at step 1 in both).

## Step time

| Run | Step times (s) |
|---|---|
| `--train-gpu`, 3 steps | 54.81, 54.17, 53.70 |
| `--train-gpu`, 2 steps, `RUNNER_DEBUG_TIME=1` | 54.32, 53.67 |
| `--train-gpu`, 2 steps, `RUNNER_TRAIN_PROF=1` | 53.34, 52.26 |
| CPU trainer, 3 steps | 52.98; then 68.02 and 58.73 with a build on the same cores |
| CPU trainer, 1 step, `RUNNER_DEBUG_TIME=1`, host load 30 to 46 | 93.35 |

The one CPU step with the cores to itself took 53 s. The device steps hold steady at 52 to 55 s whatever the
host's load.

**Where a device step goes** (`RUNNER_TRAIN_PROF=1`, the second step): forward and head 6.4 s, the layers'
backward 45.9 s (the recompute 22.8 s, the backward's products 22.4 s, attention 0.3 s). Device time was
about 49 s of it: 372,340 forward products (31.8 s) and 307,396 transposed ones (17.2 s). The forward's stage timer puts 83% of the forward in the experts, 10.8 s of 11.2 s of it on the
device. The slice is one seventh of the card, and the products are many and small.

## Gradient gates

Device mode (`RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 test-lora-grad ... strict`) on the slice, runner 32163d9:

- `test-moe-train.moe4-train.gguf`: pass, directional derivative rel 8.7e-4 at eps 2e-4 (281 coordinates).
- `test-moe-train.moe4-train-nonorm.gguf`: pass, rel 1.0e-3 at eps 2e-4 (282 coordinates). This one was not
  in `make test`'s device-mode list; it is now.
- On the fixture, a 3-step `--train-gpu` run matched the CPU trainer's losses (5.556804 and 5.531299 at steps 2
  and 3 on both).

## Not measured

A full card, or any other GPU. The 30B model does not fit an 8 GB card whole: the budget would put most experts
on the CPU, which this record does not cover.
