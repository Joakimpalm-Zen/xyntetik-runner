# Buffer owners, transfers and lifetimes

Who owns each buffer, when its contents are valid, which transfers block,
and what signal allows reuse or release. The ownership classes come from
`model_t` in `src/model.h`, whose comment is the authority; the release
functions run in the order `model_free()` calls them. This page adds the
backend side and the timing. Nothing here describes a stream, event or
staging ring that does not exist today.

## Owners

| Group | Owner | Allocated by | Released by | Shared |
| --- | --- | --- | --- | --- |
| Weight mapping, parse, layer array, f32 norm and bias conversions | a refcounted `model_weights` record, one per file and weight-side parameters | the bind phase | `model_free_weights()`, by the last holder | by every `model_t` of that file |
| KV cache, activations, family scratch, recurrent state, rope tables, worker pool | the instance | `model_alloc_runtime()`'s stages (`runtime_placement`, `runtime_kv_layout`, `runtime_buffers`, `runtime_threads_rope`, `runtime_gpu_admit`) | `runtime_free()` | never |
| Buffers a forward adds on first use (`all_logits`, the MTP queue, recurrent and PLE keep rows, the MoE probe) | the instance | the forward path | `runtime_free()` | never |
| CUDA device weights and constants | a refcounted `gpu_weights` entry keyed on file identity (path, size, mtime, inode) and geometry | `shared_upload_weights()` | `shared_release()` by the last `gpu_t` | across `--parallel` slots: uploaded once |
| CUDA device KV, activations, recurrent, MoE scratch, host staging | the instance's `gpu_t` | `cuda_alloc_kv`, `cuda_alloc_activations`, `cuda_alloc_recurrent`, `cuda_alloc_moe`, `cuda_alloc_host_staging` at init | `gpu_ctx_free()` | never |
| CUDA microbatch logits, positions and KV pointer tables | the `gpu_batch` object | `gpu_batch_create()` | `gpu_batch_free()` | borrows the lead's weights |
| Metal weight wraps | the instance's `gpu_t`, aliasing the shared mapping without a copy | `metal_wrap_weights()` | `gpu_release_state()` | the bytes are the mapping's |
| Metal KV cache | the backend: shared-storage buffers whose contents are `m->kcache` and `m->vcache` (`KV_OWNER_GPU_BACKEND`) | `metal_alloc_state()`, published by `metal_publish()` after every other allocation succeeded | `gpu_free()`, which detaches the pointers first | never |
| Adapter tables | the instance, or a request when `lora_borrowed` is set | `--lora` load, or the request | `model_lora_free()`; a borrowed table is never freed by the model | a borrowed table is the request's |
| Training tape and backward scratch | one training step | the step | the step | never |

On the CPU and CUDA, the instance's host KV is `malloc` memory
(`KV_OWNER_MALLOC`). `gpu_owner` can outlive `m->gpu` so that a CPU fallback
can keep reading backend-owned KV; clearing `m->gpu` never implies the
backend's memory may be released.

## Transfers and completion

**CPU.** No transfers. A matvec job's temporary tile (`mv_rows`) is
allocated and freed inside the worker call; if that allocation fails the job
falls back to the buffer-free dot, so a forward never writes through NULL.

**CUDA.** Every host transfer is synchronous: host staging is pageable
`malloc` memory, and `cuMemcpyHtoD` and `cuMemcpyDtoH` return when the copy is
done. One forward runs in this order:

1. `cuda_fwd_prepare()`: per-forward state, and the recurrent checkpoints
   (device-to-device copies) a rollback needs.
2. If the position does not follow the last one (a CPU-run batch, a reset,
   a rewind), `kv_upload()` copies host KV rows `[0, pos)` to the device.
3. `stage_x()` copies the token embeddings to the device.
4. Either the captured decode graph is launched and the stream synchronized,
   or the tiles are launched and `cuda_fwd_finish()` calls
   `cuCtxSynchronize()`.
5. `kv_copyback()` copies the new KV rows back. **The host KV is the
   authoritative copy**: the CPU path can take over after any forward.
6. The logits (or a verify walk's rows) are copied to `h_logits`.

The synchronize in step 4 is the completion signal: after it, every device
scratch buffer may be reused by the next forward. The logits pointer a
forward returns belongs to the instance and is overwritten by its next
forward. The only asynchronous copies are the qwen35 per-row checkpoints
(`cuMemcpyDtoDAsync` on the forward's stream), which the same synchronize
orders.

**Metal.** Unified memory and shared storage: the CPU and the GPU read the
same bytes, so there are no weight or KV transfers. Each forward encodes into
one command buffer, commits it, and waits (`waitUntilCompleted`) before it
returns; that wait is the completion signal. Per-column scratch grows only
between forwards (`metal_ensure_batch()`, the single sizing site), which is
safe because nothing is in flight then. A partial layer split hands the
boundary activation back through the host `x` buffer.

## CUDA graph capture and replay

- **What is captured.** One-token decode with full offload, captured on its
  first use and replayed for every later token. The position lives in device
  memory (`pos_dev`), so one capture serves every position. The microbatch
  keeps one graph per width, with positions and KV pointers in device
  tables.
- **What the graph assumes.** Every device address it recorded stays valid
  and unmoved for the life of the `gpu_t`. All per-instance device buffers
  are allocated at init and released only by `gpu_ctx_free()`, which
  destroys the graph first.
- **When capture is refused** (`graph_bad`): no stream; a layer without
  `attn_v.weight` (a synchronous memset that cannot be captured); a MoE layer
  without fused device routing; recurrent layers; qwen35; experts on the
  host; a partial split; the profiler on; `RUNNER_CUDA_GRAPH_OFF`.
- **When a graph is dropped.** Binding or unbinding an adapter
  (`gpu_lora_bind`, `gpu_lora_unbind`) turns graphs off for that instance,
  because the captured launches predate the adapter.
- **Capture mode** is thread-local, so another slot's forward on its own
  thread can make synchronous calls during a capture.

## Planner accounting

`--fit` (`model_fit_report()`, `model_fit_gpu()`) answers from the header
alone. It separates the whole file (`weights`), the bytes touched per token
(`hot`, smaller for a sparse MoE), KV per token and at the requested context
for each KV format, and on a discrete GPU a fixed charge for the token
embedding and headroom plus each slot's activation scratch (`gpu_fixed`)
before it counts how many leading layers fit. With `--parallel N`
(`model_fit.n_seq`) every slot pays its own KV cache, on the host and for
the offloaded layers on the device, and its own scratch; the weights are
paid once. The CUDA loader plans its split the same way
(`shared_plan_budget()`, `m->gpu_instances`), so every slot fits the split
the first one decides. The host bill includes every layer's KV, because the
host copy is the authoritative one.

It does **not** separate:

- peak simultaneously-live temporary scratch (a matvec tile, a widened
  Metal batch, the CUDA microbatch tables);
- device memory a running server already holds: the answer is "right now",
  so a resident server shows up only as less free memory.

Those are where a future planner would add accounting; measure against
`--doctor` and the idle tool before relying on an estimate
([performance](performance.md)).
