# Inkling 2 Mini on CUDA

The fast text route runs on CUDA with `RUNNER_INKLING2_CUDA=1`, a CUDA build
and GPU use allowed. It is opt-in: the default and the `reference` weight
route stay on the CPU. This inference route adds no Metal path, GPU training,
speculative decoding or shared prefix cache for `inkling2_loop`.

## Placement and execution

The loader uploads supported stored matrices and expert stacks, with their
per-expert factors. Server slots share one device weight copy and keep their
own caches and histories. The fit counts every slot before placement: it
tries the device, then expert stacks on the host, then CPU with a reason.
`--cpu-moe` keeps the expert stacks on the host. Unsupported device weight
types also keep their products on the host. Check the load's placement note;
setting the switch does not guarantee a file fits.

On the non-resident path, CUDA runs the supported products, attention blocks
and their K/V rings, expert bodies and loop message gates. The router runs
on the device for multiple positions, on the host for one. The host gathers
embeddings, orchestrates the loop and block states, and runs input norms,
post convolutions and residual work. LoRA attention terms are added on the
host, and an adapter declines the resident step.

An eligible plain single-position forward runs the whole decode step on
the device: all 48 layer applications, block-state reads, message gates,
norms, attention, post convolutions, routing, experts, final read, head and
softcap. Only the embedding row and position go up and the logits row comes
back: three transfers a step, none inside the layer loop. Diagnostic
per-application outputs and expert-selection observers use the
non-resident path. Histories are synchronized when execution switches paths.
A failed device operation refuses that forward and later forwards on the
instance; it does not silently produce CPU rows.

The CPU fast route retains its scalar `ink2_swiglu`, as on main. CUDA uses
its own `tanhf`/`expf` and reduction orders; CUDA versus CPU logits are not
claimed byte-identical. The resident step is held byte for byte to the
non-resident CUDA route across prefill, decode, chunk transitions and
concurrent slots. That equivalence is separate from the publisher's CPU
fidelity measurements in the [file roster](model-support.md#inkling-2-mini-ud-q8_k_xl).

## Real-file resident-step window

Measured on an sm_120 GPU (a 24 GB MIG slice), on this exact file:

- Repository: `unsloth/Inkling-2-Mini-GGUF`.
- Revision: `45fd08764f2b0bf16b03a5483a80e5c7aaec8772`.
- Path: `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`.
- Size: 22,867,452,896 bytes.
- SHA256: `2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68`.

The window at `be4e210a` used ten rotated rounds per column. Every column
applied `-falign-functions=64 -falign-loops=64` to the Inkling objects,
verified on the compile lines. The shipped flags are unchanged. Host load
ranged from 14.55 to 49.32; residency never refused a run. The rule stated
before measuring was decode at least 1.05x and prefill not lower.

| tok/s | Resident step, `be4e210a` | Baseline `37b140a7` | Baseline `08d20c62` |
|---|---:|---:|---:|
| Decode | 23.269 | 21.308 | 21.307 |
| Prefill | 77.741 | 77.554 | 77.477 |

Decode is 1.092x against each baseline. Prefill is not lower and is equal
within noise: its differences' intervals are `[-0.067, +0.429]` and
`[-0.009, +0.590]` tok/s. This is not a faster-prefill claim. The earlier
unaligned prefill gap was traced to host code layout (R4.48.9.4): device
work was identical, and aligning all columns alike removed the gap. These
figures describe one file and one aligned measurement build, not every GPU
or a guarantee about another build's layout. Q4_K_XL was a separate ladder
smoke, not the timed file.

## Gates and scope

At `1fb25712`, `make test-inkling2-cuda` passed 284 of 284 checks on Linux
sm_120 and Windows sm_120; Windows passed both STRICT and release-recipe
builds. Earlier 244- and 262-check results are smaller versions of the
suite, not skips at this head. The gate covers the stored types and expert
factors, float64 product anchors, CPU/CUDA tolerances, resident/non-resident
identity, transfer counts, slot ownership and injected failure refusals.

Compute Sanitizer at the same head was filtered to `k_ink2_rms_rows`,
`k_ink2_msg_norm` and `k_ink2_depth_attn_ix`. A positive control first proved
the filters/instrument. Racecheck found zero hazards, errors and warnings;
synccheck found zero errors. All 284 suite checks passed under each. This
is scoped to those three changed kernels, not an unfiltered whole-backend
sanitizer claim.

The rebase preserves main's caller-sized attention scratch, atomic error
status, success-only output projection, scratch routing tables and scalar
SwiGLU. The relative-probe allocation-failure regression was reproduced
without CUDA and fixed: the host slot takes its projected K/V chunk, advances
`n_past`, leaves output untouched and returns `INK2_BAD_INPUT`; the next
chunk can continue. The final replay onto `b670e732` has 39 identical patches.

These fixture/kernel checks and speed measurements do not create a backend
validation mark. Each [support-roster](model-support.md#overview) cell comes
from that exact file's pinned chat smoke and comparison on its named device
class. The smoke manifest declares `RUNNER_INKLING2_CUDA=1` for the CUDA
backend; the pin settings and recorded command preserve that opt-in.

## Test-only controls

These seams belong to `make test-inkling2-cuda`; they are not operator
settings. They are unset in a normal run. The gate uses each control to
prove that a missing factor, declined path or device failure is detected.

| Environment variable | Test effect |
|---|---|
| `RUNNER_INKLING2_CUDA_TEST_NO_FACTOR=1` | Omit the expert factors, the fidelity control. |
| `RUNNER_INKLING2_CUDA_TEST_FREE` | Cap the fit's reported free device memory in bytes. |
| `RUNNER_INKLING2_CUDA_TEST_PUT_FAIL` | Fail the indexed upload; check rollback and CPU fallback. |
| `RUNNER_INKLING2_CUDA_TEST_LIN_FAIL` | Fail the indexed device product; check forward refusal. |
| `RUNNER_INKLING2_CUDA_TEST_EXP_FAIL` | Fail the indexed expert-body call. |
| `RUNNER_INKLING2_CUDA_TEST_ATTN_FAIL` | Fail the indexed attention-core call. |
| `RUNNER_INKLING2_CUDA_TEST_MSG_FAIL` | Fail the indexed message-step call. |
| `RUNNER_INKLING2_CUDA_TEST_STEP_FAIL` | Fail the indexed resident-step attempt. |
| `RUNNER_INKLING2_CUDA_TEST_NO_EXP=1` | Keep the per-product expert-body path. |
| `RUNNER_INKLING2_CUDA_TEST_NO_ATTN=1` | Keep the attention core and rings on the host. |
| `RUNNER_INKLING2_CUDA_TEST_NO_BLOCK=1` | Keep host attention glue with the device core. |
| `RUNNER_INKLING2_CUDA_TEST_NO_MSG=1` | Keep the message-step host path. |
| `RUNNER_INKLING2_CUDA_TEST_NO_ROUTER=1` | Keep the non-resident router on the host. |
| `RUNNER_INKLING2_CUDA_TEST_NO_STEP=1` | Keep the non-resident forward rather than the resident decode step. |
| `RUNNER_INKLING2_CUDA_TEST_DEVFORM` | `1` passes private device buffers to the cores; `2`, `3` and `4` additionally make a required attention, expert or message buffer appear short, to check refusal. |
| `RUNNER_INKLING2_CUDA_TEST_DUP_IDX=1` | Repeat a scatter-list position, to check that the call refuses it. |
