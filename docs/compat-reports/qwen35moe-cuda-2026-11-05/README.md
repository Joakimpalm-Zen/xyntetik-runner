# qwen35moe on CUDA: the R4.27 A2 gate and Qwen3.6-35B-A3B's cells (lab, 2026-11-05)

**Unflattering first: CUDA's evidence here is agreement with the CPU, not fidelity to the publisher.** The row's
fidelity reads (0.0475 mean KLD, 97.2% margin-qualified top-1) are the CPU's against Qwen's BF16 parent in float32. No
gold read ran on CUDA. What CUDA shows is that it computes what the CPU computes: 3.6e-6 of the logit range on the
real file with routing pinned. The Windows CUDA host's cells are not run, and will not be (no CUDA consumer host). The
compatibility check's cpu_cuda arm is 8 of 9 prompts byte-identical, not 9 of 9: the ninth splits on a 0.085-nat near
tie. No speed was measured.

The kill numbers and the four mutants were declared before the first run (PLAN-a2 in the lab's record store, r427-a2).
The owner's kill was the toy gate not reaching 1e-5 within 2 working days. It passed on its first run.

## The code

The A2 commit (`R4.27 A2: lift CUDA's by-name qwen35moe hold; test-cuda-qwen35moe gates it`), cherry-picked onto
main a0babf8. It is this branch's first commit, and every cell below ran on it (commit 88dd8f7 in the records). Two
conflicts, resolved keeping main's newer code:
- `scripts/cuda-gate.sh`: main's DEVICE_STEPS, with `qwen35moe:test-cuda-qwen35moe` after qwen4exp.
- `src/cuda.c`: cuda_admit keeps the deepseek2 q-lora check and `model_moe_router_is_plain_any_norm`, and loses only
  the qwen35moe block.

The device path is a composition that already existed: dense qwen35's trunk, the routed MoE kernels and
`tile_shexp`, the gated shared expert.

## The toy gate: `make test-cuda-qwen35moe`

[test-cuda-qwen35moe.log](test-cuda-qwen35moe.log). The target's bars, unchanged:

| Check | Bar | Read |
|---|---|---|
| gpu-identity: the shexpg fixture and the ornith qwen35moe toy at state 8, 40, 72 and 128; fused, eager and grouped expert routes; batched and one token at a time | 1e-5 of the logit range, free and pinned routing | 30 of 30 ok |
| Routing pin | every slot forced, 0 differing | 30 of 30 (1,140 of 1,140 slots on s72 and s128) |
| Speculative identity | output unchanged | 5 of 5 |
| --score placements on s40 | 2e-5 in logprob, no argmax flip | 1.91e-6 all on the device, 1.91e-6 --cpu-moe, 9.60e-7 --gpu-layers 2, 2.38e-6 both; 0 flips |

## The mutants: does the gate see a wrong composition?

Each was applied alone to the passing tree, rebuilt and run through the gate. Expected: fail. 4 of 4 failed.

| Mutant | Where | Gate | How it failed |
|---|---|---|---|
| 1. The shared expert's sigmoid gate dropped (gated treated as ungated) | `tile_shexp` | exit 2 | CPU and GPU disagree beyond reduction-order rounding ([log](mutant-1-gate.log)) |
| 2. The shared expert skipped | the MoE tile | exit 2 | the same ([log](mutant-2-gate.log)) |
| 3. The DeltaNet gate set to Qwen3.8's sigmoid form | `gate_sigmoid` | exit 2 | the device's router chose another expert at 309 of 1,140 slots, 0.271 against 0.01 ([log](mutant-3-gate.log)) |
| 4. `k_axpy_sigmoid` adds `g * v`, not `sigmoid(g) * v` | kernel, PTX rebuilt | exit 2 | CPU and GPU disagree ([log](mutant-4-gate.log)) |

For mutant 4 the PTX was rebuilt with the pinned toolchain (nvcc 13.0.88, tools/ptx-toolchain/versions.txt). The unmutated
`make ptx` first reproduced the committed headers byte for byte (kernels_ptx.h a0e46b7f, kernels_ptx80.h 48bdb5be).
After the mutants the tree had no tracked change, the PTX equalled the committed headers, and the gate passed again
([log](test-cuda-qwen35moe-restored.log)). Not mutated: the routed-expert kernels themselves, which other gated
architectures share.

## The real file: Qwen3.6-35B-A3B Q4_K_M on the slice

bartowski's `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` at 5c2410d7, 22,285,080,192 bytes. Held in memory from the Hub, its
sha256 b46fedd3 equal to the Hub's LFS sha256. The host was a CUDA MIG slice, 1g.24gb, sm_120 (25.13 GB offload
budget).

- **Placement:** all 40 blocks and the head on the device (`G=40/40 full=1 used=21.72GB`). Prompt window 64 columns
  at batch 32, with 3.00 GB spare.
- **gpu-identity** ([log](gpu-identity-q4_k_m.log)), 96 tokens, 24 teacher-forced positions, 5,959,680 logits:
  - pinned routing: mean |dlogit| 3.6e-6 of the mean range 33.2 (limit 1e-3), worst 0.00413;
  - free routing: 3.85e-6 (limit 2e-3), worst 0.00442; 3 of 30,400 expert slots differed from the CPU's choice.
- **tier-smoke** on CUDA: pinned, then compared against that pin: match
  ([pin](tier-smoke-cuda-pin.json), [record](tier-smoke-cuda-record.json)). The CUDA pin's sha256 (f3a25e63) is the
  CPU pin's from 2026-10-30, so the 48 tokens are byte-identical across backends. The CPU re-recorded against its
  existing pin: match ([record](tier-smoke-cpu-record.json)). The CPU pin was not redone.
- **compat_matrix** ([compat.json](compat.json), [cpu_cuda](../cpu_cuda/qwen3.6-35b-a3b-q4_k_m.json)): load,
  tokenizer and chat pass. cpu_cuda (128 tokens; RUNNER_CUDA_TC=0, RUNNER_CUDA_ROUTE=reference, RUNNER_MOE_EAGER=1) is
  `pass_margin_qualified`: 8 of 9 exact. On "1 2 3 4 5 6 7 8" the two sides split at token 96 where the CPU's margin to
  the GPU's pick is 0.085 nats (the tie band is 0.5).

## Not measured

- The Windows CUDA host: not run, no CUDA consumer host.
- A 24 GB consumer card: this file's 21.72 GB on the device leaves less room there than the slice's budget.
- Fidelity on CUDA against the float32 reference, speed, task quality, Metal on this file.
- The three upcoming qwen35moe rows (Qwen3.8-35B-A3B, Ornith-1.5-35B-A3B, MiniCPM-V-4.7-35B-A3B), each of which needs
  its own gate on its own file.

`FILES.sha256` lists the sha256 of every file here except this README.
