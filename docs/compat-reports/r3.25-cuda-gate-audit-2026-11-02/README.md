# R3.25, CUDA half: structural mutants against the CUDA device identity gates, 2026-11-02

**Question.** Does each tolerance-bounded identity gate a CUDA host runs catch a real mistake in a CUDA kernel or in
the host encoder? This is the CUDA half of the Metal audit (`../r3.25-metal-gate-audit-2026-10-31/`). The trigger
was R4.10: on the toy Mamba-2 fixture, a scan with its D skip dropped passed test-gpu-identity's cross-backend 2e-3.

**Host.** The datacenter MIG slice: an RTX PRO 6000, MIG 1g.24gb, sm_120. Runner main 550164c0ff13,
clang 22.1.8. To rebuild mutated kernels, the mutant runs regenerate the PTX with the local nvcc 13.2; the pinned
toolchain is 13.0. Healthy residue on the regenerated PTX matches the committed PTX's: spark's three runs are
within about 10% (3.3e-6 against 3.7e-6), and every other fixture is the same to three digits. The final check
(below) runs on the committed PTX.

**Method.** `cuda-mutant-audit.py`, run from the runner tree root:
1. Apply one structural mutant to `src/kernels.cu` (then regenerate the PTX) or `src/cuda.c`, and check the number
   of replacements.
2. Rebuild, then run every gate below. The script then restores the files and rebuilds.

Mutants:
- the attention residual dropped (host encoder, 3 sites);
- the SSM D skip dropped (both scan kernels);
- SiLU replaced by tanh-GELU in the dense gated FFN;
- the RMS norm skipped (scale 1);
- its eps dropped;
- rope one position later in `k_rope`;
- the attention scale ×1.05 (host, 2 sites);
- the matvec bias dropped (all 14 sites);
- K left unrotated (host);
- rope skipped for Q and K (`k_rope`, `k_rope_seq`).

Gates (19):
- test-gpu-identity on 14 fixtures. These are the 11 that `make test` runs, plus the three hybrids that only
  Metal's targets run: ornith (qwen35), test-m2-s40-g1 (Mamba-2) and nemotron-h.
- test-prefix at chunk 4 and 64.
- `make test-cuda-nvfp4`, `test-cuda-tc-kq` and `test-cuda-moe-replay`.

Two notes on scope:
- `test-cuda-tc-kq` ran its Q3_K and NVFP4 cases only, because this host has no llama-quantize.
- test-prefix and tc-kq compare CUDA with CUDA: a mutant on the shared path cannot show there, by design ("s" in
  the table).

Pass 1 ran the gates at their own bounds (`pass1-default-bound.log`, plus `pass1b-rope-default-bound.log` for the
two rope mutants added after pass 1). Pass 2 ran the gpu-identity gates at the proposed bounds, because the limit
moves nothing else:
- `pass2-tight.log` is the first run, with k_temp_live still at 2e-3;
- `pass2-fixed.log` reran it with the decode-graph fix below and k_temp_live at 1e-5.

**A fault in the first runs.** Until 18:30 the audit script left stale PTX after a kernel mutant. It restored
`kernels.cu` but rebuilt the PTX only for the next kernel mutant, so a following `cuda.c` mutant ran on the previous
one's kernels. Two rows were affected:
- **The attention scale ×1.05** (in passes 1, 2 and 2-fixed) also carried rope +1. That mutant moves the residue by
  1.5e-5 at most. The clean rerun matches the old row to within 0.2%, with the same catches.
- **K unrotated** (in passes 2 and 2-fixed) also carried the bias dropped. gptoss-top1, the one fixture with biases,
  gave 2.4e-3 instead of its true 7.8e-4. Pass 1b ran K unrotated first, so it was clean.

The script is fixed, and the two mutants were rerun clean:
- `pass1c-rerun-default-bound.log`, the attention scale at the default bounds. It ran on the patched tree, so the
  nvfp4 cell there is at the new 1e-4.
- `pass2c-rerun.log`, both mutants at the proposed bounds.

The table uses `pass2-fixed.log` and, for those two rows, `pass2c-rerun.log`.

## Table

Each cell gives the deviation, as a fraction of the mean logit range for gpu-identity and prefix, and as the max
logprob difference for nvfp4. The marks:
- **C**: caught at the gate's own bound.
- **T**: caught only at the proposed bound.
- **n**: the deviation is the healthy residue, so the mutated path is not on this fixture.
- **s**: a CUDA-against-CUDA gate.
- **-**: the mutant moved the result and passed.

| gate | healthy | attn residual | SSM D skip | silu to gelu | norm skipped | norm eps | rope +1 (Q and K) | attn scale x1.05 | bias dropped | K rope skipped | rope skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gpu-id test.gguf` | 2.55e-08 | 4.46e-02 **C** | 2.55e-08 n | 2.02e-03 **C** | 8.31e-02 **C** | 7.71e-04 **T** | 3.66e-07 - | 5.84e-05 **T** | 2.55e-08 n | 9.56e-04 **T** | 9.00e-04 **T** |
| `gpu-id test.gguf 0 1` | 3.02e-07 | 4.46e-02 **C** | 3.02e-07 n | 2.02e-03 **C** | 8.31e-02 **C** | 7.71e-04 **T** | 4.75e-07 - | 5.84e-05 **T** | 3.02e-07 n | 9.56e-04 **T** | 9.00e-04 **T** |
| `gpu-id noisy-norms` | 1.75e-08 | 5.42e-02 **C** | 1.75e-08 n | 1.81e-03 **T** | 8.77e-02 **C** | 7.95e-04 **T** | 1.86e-08 - | 1.60e-06 - | 1.75e-08 n | 1.94e-05 **T** | 1.92e-05 **T** |
| `gpu-id gptoss-top1` | 4.47e-08 | 2.06e-01 **C** | 4.47e-08 n | 4.47e-08 n | 5.16e-01 **C** | 1.20e-03 **T** | 3.32e-07 - | 6.31e-05 **T** | 2.07e-03 **C** | 7.83e-04 **T** | 5.93e-04 **T** |
| `gpu-id k_off` | 2.55e-08 | 4.46e-02 **C** | 2.55e-08 n | 2.02e-03 **C** | 8.31e-02 **C** | 7.71e-04 **T** | 3.66e-07 - | 5.84e-05 **T** | 2.55e-08 n | 9.56e-04 **T** | 9.00e-04 **T** |
| `gpu-id k_nope` | 2.39e-08 | 4.47e-02 **C** | 2.39e-08 n | 2.02e-03 **C** | 8.32e-02 **C** | 7.72e-04 **T** | 2.39e-08 n | 6.88e-05 **T** | 2.39e-08 n | 2.39e-08 n | 2.39e-08 n |
| `gpu-id k_half` | 2.43e-08 | 4.46e-02 **C** | 2.43e-08 n | 2.02e-03 **C** | 8.31e-02 **C** | 7.71e-04 **T** | 3.54e-07 - | 6.40e-05 **T** | 2.43e-08 n | 7.38e-04 **T** | 7.40e-04 **T** |
| `gpu-id k_temp_live` | 1.02e-07 | 4.70e-02 **C** | 1.02e-07 n | 2.01e-03 **C** | 8.43e-02 **C** | 8.71e-04 **T** | 1.02e-07 n | 1.14e-03 **T** | 1.02e-07 n | 1.02e-07 n | 1.02e-07 n |
| `gpu-id spark` | 3.33e-06 | 1.44e-01 **C** | 3.33e-06 n | 3.33e-06 n | 1.45e-01 **C** | 1.10e-03 **T** | 1.05e-05 - | 1.75e-04 **T** | 3.33e-06 n | 3.49e-03 **C** | 1.52e-03 **T** |
| `gpu-id spark 0 1` | 3.32e-06 | 1.44e-01 **C** | 3.32e-06 n | 3.32e-06 n | 1.45e-01 **C** | 1.10e-03 **T** | 1.05e-05 - | 1.75e-04 **T** | 3.32e-06 n | 3.49e-03 **C** | 1.52e-03 **T** |
| `gpu-id spark-swa` | 1.56e-06 | 1.51e-01 **C** | 1.56e-06 n | 1.56e-06 n | 1.45e-01 **C** | 1.10e-03 **T** | 1.52e-05 - | 1.80e-04 **T** | 1.56e-06 n | 3.75e-03 **C** | 1.58e-03 **T** |
| `gpu-id ornith (qwen35)` | 4.56e-08 | 1.92e-02 **C** | 4.56e-08 n | 9.32e-04 **T** | 1.01e-01 **C** | 1.13e-03 **T** | 1.48e-06 - | 4.33e-04 **T** | 4.56e-08 n | 8.77e-03 **C** | 7.25e-03 **C** |
| `gpu-id mamba2 s40` | 1.24e-08 | 1.86e-02 **C** | 8.17e-04 **T** | 1.72e-03 **T** | 2.64e+00 **C** | 4.11e-08 - | 3.11e-08 - | 8.69e-06 - | 1.24e-08 n | 1.84e-04 **T** | 1.67e-04 **T** |
| `gpu-id nemotron-h` | 1.73e-08 | 2.56e-02 **C** | 8.94e-03 **C** | 1.73e-08 n | 1.45e-01 **C** | 1.39e-03 **T** | 1.73e-08 n | 2.31e-05 **T** | 1.73e-08 n | 1.73e-08 n | 1.73e-08 n |
| `prefix chunk 4` | 0.00e+00 | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s |
| `prefix chunk 64` | 0.00e+00 | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s | 0.00e+00 s |
| `cuda-nvfp4` | 9.60e-07 | **C** | 9.60e-07 n | 8.42e-03 - | **C** | 4.06e-03 - | 4.76e-06 - | 5.20e-04 - | 9.60e-07 n | 7.94e-03 - | 6.02e-03 - |
| `cuda-tc-kq (TC prefill)` | ok | s | s | s | s | s | s | s | s | s | s |
| `cuda-moe-replay` | 1.90e-03 | **C** | **C** | 1.50e-04 **C** | **C** | **C** | 2.10e-03 - | 1.00e-03 **C** | 3.20e-04 **C** | **C** | 3.60e-03 **C** |

| mutant | caught at 2e-3 (19 gates) | gpu-id caught at the tight bound (14) | reachable and still missed at the tight bound |
|---|---|---|---|
| attn residual | 16 | 14 | none |
| SSM D skip | 2 | 2 | none |
| silu to gelu | 7 | 9 | none |
| norm skipped | 16 | 14 | none |
| norm eps | 1 | 13 | mamba2 s40 4.11e-08 |
| rope +1 (Q and K) | 0 | 0 | test.gguf 3.66e-07, test.gguf 0 1 4.75e-07, noisy-norms 1.86e-08, gptoss-top1 3.32e-07, k_off 3.66e-07, k_half 3.54e-07, spark 1.05e-05, spark 0 1 1.05e-05, spark-swa 1.52e-05, ornith (qwen35) 1.48e-06, mamba2 s40 3.11e-08 |
| attn scale x1.05 | 1 | 12 | noisy-norms 1.60e-06, mamba2 s40 8.69e-06 |
| bias dropped | 2 | 1 | none |
| K rope skipped | 5 | 11 | none |
| rope skipped | 2 | 11 | none |

The `cuda-moe-replay` column is not counted as evidence; see below.

The k_temp_live row's pass-1 cells (all at 2e-3) were measured before the fix, on top of its 8.0e-4.

Every deviation repeats exactly on a given build: five runs of gptoss-top1 under K unrotated all give 7.83e-4. The
2.4e-3 that first looked like a router tie was the stale-PTX fault above. CPU and CUDA also route identically: on
gptoss-top1 the two sides select the same expert at all 234 (position, layer) points of a scored prompt (the CUDA
trace leaves the `norms` field empty, which is the only difference).

## What the table says

**The cross-backend 2e-3 is blind on CUDA, as it was on Metal.** At 2e-3, the gpu-identity gates catch only the
gross mistakes, the residual dropped and the norm skipped, on every fixture.
- **No gpu-identity fixture catches two mistakes:** the norm's eps dropped (7.7e-4 to 1.4e-3, mamba2 aside) and
  the attention scale ×1.05 (1.6e-6 to 5.4e-4).
- **Rope:** K unrotated passes everything but spark and ornith, and rope skipped everything but ornith.
- **The SSM D skip** passes mamba2 s40 (8.2e-4), and only nemotron-h catches it.
- **SiLU to GELU** passes noisy-norms, ornith and mamba2.

That is a blind bound, not a blind fixture: the healthy residue is 1e-8 to 3e-7 (spark 3.7e-6).

**At the proposed bound, most fixtures that reach a mutant catch it.** Only rope one position later escapes every
fixture, and it is an equivalent mutant: `k_rope` rotates Q and K at the same position, so the shift leaves every
relative angle unchanged. The residue moves (to 1.5e-5 at most), and the program is still correct; the Metal half
saw the same on its decode path. The two rope mutants that break the rotation are caught by 11 of 14 fixtures.
The other three use no rope: k_nope, nemotron-h, and k_temp_live, which is NoPE throughout.

**What remains, by cause:**
- **Fixture: biases.** Only gptoss-top1 has biases. A dense fixture with QKV or FFN biases would close it; the
  Metal half saw the same.
- **Fixture: the norm's eps on mamba2 s40.** `HYBRID_TEST_AMP=1` makes the hidden state's mean square far larger
  than eps, so dropping eps moves the logits by only 3e-8. Every other fixture catches it.
- **Bound: the attention scale on noisy-norms (1.6e-6) and mamba2 s40 (8.7e-6).** Under 1e-5. A 1e-6 limit for
  those two would catch it: their residue is 1.8e-8 and 1.2e-8, so the margins are 57× and 80×. This patch does not
  propose it, because 12 other fixtures already catch the mutant.
- **No CUDA gate runs the hybrids.** On a CUDA host, `make test` holds no Mamba-2, nemotron-h or qwen35 fixture to
  the CPU's logits; those gpu-identity runs are Metal targets (`test-metal-mamba2`, `test-metal-qwen35`). Of the
  gates a CUDA host does run, none caught the D skip dropped (moe-replay's one failure did not reproduce; see
  below).
- **`test-cuda-moe-replay` is not reliable evidence here.** It failed under the D-skip mutant in pass 1, although
  none of its fixtures has an SSM. With the same mutant applied, a rerun with full output passed. So did four clean
  runs (108 to 301 s), and then twenty more on runner main, every one of them (89 to 124 s). That makes 25 straight
  passes; at the rule of three, the failure rate is under about 12%.
  - **What failed is not known.** The audit kept only a gate's last line, which for a make target is make's own
    "Error 1". The failing run took 443 s, against about 100 s since, while the CPUs were contended. The script now
    keeps a failing gate's full output.
  - **Its other catches in the table are not counted as evidence.** Most of them failed within seconds, which looks
    like a real trace mismatch, but only a repeat could tell.

## k_temp_live: the decode graph replayed a stale temperature (fixed in this patch)

**The symptom.** k_temp_live's CUDA residue was 8.0e-4, against Metal's 1.9e-7. CUDA's ordinary residue matches
Metal's (test.gguf 2.6e-8 against 2.3e-8), so this was not rounding. The cross-backend 2e-3 had been sized on that
8.0e-4 (2026-08-16), and at that level the fixture hid the norm's eps and the attention scale.

**The bisection.**
- **Not the position.** The temperature at position ±1 gives 8.4e-4 and 8.5e-4.
- **Not the prefill tiles.** One-token forwards give 8.0e-4 too.
- **The decode graph.** Per-row dumps of Q and the attention output on both sides needed `RUNNER_CUDA_GRAPH_OFF=1`,
  and with it the residue fell to 1.0e-7. With graphs off and no dumps, it is 1.0e-7 (batch 1: 3.6e-8). With graphs
  on, it is 8.0e-4.

**The mechanism.** `cuda_fwd_graph_decode` captures the decode forward once and replays it for every later token.
Its comment says why that is safe: "position-invariant because pos lives in device memory". Rope and the attention
window read the position that way, but the temperature does not. `enc_rope_or_temp` computes it on the host
(`model_attn_temp(m, pos + b)`), passes it to `k_scale` as a kernel argument, and skips the launch where it is 1.0.
A replayed graph therefore applies the capture position's factor to every later token.
- On the fixture (floor 4), the capture at position 72 used 15.72 for tokens whose own factor reaches 16.89.
- At llama 4's floor of 8192, the factor is 1.0 below position 8191, so a graph captured there holds no scale launch
  at all. Past 8191, decode through that graph applied none.

That affects llama 4 and mistral3 files that declare the temperature, when decode runs through the graph: the
model fully on the device, with no MoE layer on the host. Prefill, and decode with graphs off, were right.

**The fix** (`src/cuda.c`, host only, no kernel change). The decode graph records the factor it was captured with
(`graph_temp`). When the factor for the next token differs, the graph is destroyed and captured again. The factor
steps every floor_scale positions, so a real file captures again once per 8,192 tokens, and a model without the
knob never does (its factor is always 1.0). The fixture now captures at positions 72, 75, 79, 83, 87 and 91 over
its 24 decode tokens, where k_off captures once (a temporary counter, removed). The residue is 1.02e-7 with graphs
on and off alike. At 1e-5, k_temp_live then catches:
- the norm's eps (8.7e-4);
- the attention scale (1.1e-3);
- SiLU to GELU (2.0e-3).

**The test.** k_temp_live runs at 1e-5 in `make test`, like the other toy fixtures. Its decode crosses six
temperature steps, and the old graph misses that bound 80 times over.

**`GPU_DEV_FRAC`.** `tests/test_gpu_identity.c` explained the 8.02e-4 as "its reductions round differently again".
The patch corrects that comment. The cross-backend 2e-3 itself is unchanged: its sizing rested on the 8.02e-4,
but lowering it would need real-file residues on CUDA, a separate decision.

## The proposed tightening (the patch)

- **`GPU_ID_TIGHT = GPU_ID_LIMIT=1e-5`** for the toy fixtures, on every GPU host. This replaces `METAL_ID_TIGHT`:
  Metal's values carry over unchanged, and CUDA now uses them too. The CUDA margin is 33× over the worst toy
  residue (`test.gguf 0 1`, 3.0e-7), and 19× over the worst hybrid (ornith-s128, 5.2e-7).
- **`GPU_ID_TIGHT_SPARK = GPU_ID_LIMIT=2e-5`** for the three spark runs. The CUDA residue is 3.7e-6, a 5.4× margin.
- **k_temp_live** takes `GPU_ID_TIGHT` too, now that the decode-graph fix is in: CUDA residue 1.0e-7, a 98× margin.
- **The decode-graph fix** in `src/cuda.c` and the corrected comment in `tests/test_gpu_identity.c` (above).
- **test-cuda-nvfp4:** the generated fixture is held to a logprob difference of 1e-4. Its residue is 9.6e-7, and at
  2e-2 the eps (4.1e-3), the activation (8.4e-3), both rope mutants (6e-3 and 7.9e-3) and the attention scale
  (5.2e-4) passed. A real file named in `NVFP4_MODEL` keeps 2e-2.
- **`make test-cuda-hybrids`** (new): the hybrid fixtures of `test-metal-mamba2` and `test-metal-qwen35` on CUDA
  at 1e-5. Those are the Mamba-2 hybrids at d_state 40, 72 and 128, nemotron-h, and ornith at its default plus
  states 40, 72 and 128. All eight pass, at 1.2e-8 to 5.2e-7. It is not yet hooked into a suite, because a CUDA
  host runs the `test-cuda-*` targets by name.

## The CUDA MoE routing trace's norms (follow-up)

**The gap.** The routing trace (`RUNNER_MOE_TRACE`) has carried a `norms` field since JC-R3: the L2 norm of each
selected expert's output, after its down projection and bias and before the routing weight. On CUDA, every record
had `"norms": []`. The device routers wrote their records at routing time, before any expert output existed. The
check above found the gap: on gptoss-top1, CPU and CUDA route identically at all 234 points, but only the CPU record
has norms.

**The fix** (`src/cuda.c`, host only).
- **Routing holds its records.** `enc_moe_route` and the eager routers now keep the tile's records instead of writing
  them (`moe_trace_add`).
- **Every MoE path fills in the norms** where its expert outputs exist, and `gpu_moe_ffn` and `gpu_gemma_moe_ffn`
  write the records (`moe_trace_flush`). The paths are fused, grouped, eager and by-expert, each in its gemma-4
  variant too.
- **The norm matches the CPU's definition.** Where the device folded the routing weight (and gemma-4's down scale)
  into the hidden state, it is divided back out, and where the bias is added later, it is added.
- **A missing norm is visible.** A norm a path leaves uncomputed writes an empty array, as before.
- **Trace runs only.** Tracing was already eager (`graph_bad`). The trace syncs and copies back each output it
  measures, and nothing else changes; moe-replay's exact "tracing changed the scores" check still passes.
- `model_moe_trace_route` takes the norms (`src/model.h`, `src/model.c`).

**The test** (`scripts/cuda-moe-replay.py`, run by `make test-cuda-moe-replay`). For every model, arm and mode, the
device trace is now compared with the CPU trace field by field:
- the experts, exactly, as before;
- every record holds as many gates and norms as experts;
- the gates within 1e-3;
- the norms within 2e-2, relative.

Those bounds come from `trace-fields.log` (three fixtures × four arms × two modes):

| | solo | chunked |
|---|---|---|
| gates, absolute | 1e-6 | 5.2e-5 |
| norms, relative | 3.5e-4 | 4.0e-3 |

The chunked tiles stage activations at half precision. A norm of the weighted output, a dropped bias or a slot
mix-up is off by a tenth or more.

The test passes on the patch (24 cases, 108 s, as before; `moe-replay-trace-fields.log`) and fails on runner main
without it: "the device trace at (0, 0) holds 2 gates and 0 norms for 2 experts". It also fails a mutant that
leaves the routing weight in the norm (device 4.27 against the CPU's 8.51). `test-cuda-moe-twin`, `test-cuda-keepn`
and the gpu-identity gates pass unchanged. `test-cuda-moe-legacy` needs llama-quantize, which this host lacks
(runner main fails the same way).

## GPU_DEV_FRAC on real files (follow-up): measured, kept at 2e-3

**Why measure.** The cross-backend constant (2e-3) was sized on k_temp_live's 8.02e-4, which turned out to be the
decode-graph bug above. Every `test-gpu-identity` run without `GPU_ID_LIMIT` uses it: real models, the CUDA gate's
quantized fixtures, and `tests/test_iquants.py`. The question was whether real files' honest residue allows a
tighter bound.

**Healthy residue on real files** (`real-files.log`, `iq-files.log`, `moe-flips.log`). These ran on CUDA sm_120
with the patch (the graph fix included), each batched (prefill batch 32) and one token at a time. The IQ files were
quantized here from Qwen3-0.6B BF16 with an importance matrix (llama.cpp 73a43d1's llama-quantize and
llama-imatrix). The Qwen3-4B ones were requantized from its Q4_K_M.

| file | type | batched | one token |
|---|---|---|---|
| SmolLM2-135M-Instruct | Q8_0 | 7.26e-5 | 4.68e-5 |
| Qwen2.5-0.5B-Instruct (QKV bias) | Q4_K_M | 3.62e-5 | 3.75e-5 |
| Qwen3-0.6B (q/k norm) | Q4_0 | 2.82e-5 | 2.51e-5 |
| Qwen3-0.6B | BF16 | 2.43e-5 | 2.69e-5 |
| TinyLlama-1.1B-Chat | Q2_K | 2.12e-5 | 2.04e-5 |
| Qwen3.5-0.8B (Gated DeltaNet) | Q4_K_M | 3.5e-6 | 3.63e-6 |
| granite-4.0-h-micro (Mamba-2 hybrid) | Q4_K_M | 6.57e-6 | 3.95e-6 |
| Qwen3-4B | Q4_K_M | 2.67e-5 | 3.5e-5 |
| granite-4.1-8b (the gate's documented example) | Q4_0 | 3.87e-5 | 3.49e-5 |
| Qwen3-30B-A3B (top-8 of 128 MoE) | Q4_K_M | **1.06e-3** | **1.46e-3** |
| Qwen3-0.6B | IQ3_S | 2.59e-5 | 2.75e-5 |
| Qwen3-0.6B | IQ2_XS | 5.93e-5 | 5.96e-5 |
| Qwen3-0.6B | IQ2_XXS | 1.04e-4 | 8.87e-5 |
| Qwen3-0.6B | IQ1_M | 2.65e-4 | 3.12e-4 |
| Qwen3-0.6B | IQ1_S | **3.99e-4** | 3.59e-4 |
| Qwen3-4B | IQ2_XS | 6.49e-5 | 6.39e-5 |
| Qwen3-4B | IQ1_S | 2.85e-4 | 2.90e-4 |

**What the table shows:**
- **The k-quants and BF16** stay at 7.3e-5 and below. The worst is SmolLM2-135M Q8_0, and Metal measured 5.4e-5 on
  the same file.
- **The 1-bit codebooks** reach 4.0e-4. That is not a decoder difference: `tests/test_iquants.py`'s toy IQ files,
  built here the way the test builds them, measure 2e-7 to 1.9e-6 for every type, IQ1_S at 3.5e-7. The real file's
  depth amplifies an ordinary difference.
- **The CUDA gate's quantized fixtures** stay at 1.34e-5 or below: varff 1.3e-5, moe-twin 2.1e-6, q3k 1.0e-6,
  keepn 2.0e-7, qwen4exp 1.9e-7.
- **Qwen3-30B-A3B reaches 1.46e-3**, a different kind of residue: routing flips at near-ties.
  - **The flips.** On a 40-token prompt scored with `RUNNER_MOE_TRACE` on both sides, 21 of 1,872 routings differ.
    18 of them are only a different slot order, and 3 change the expert set, all at one position. The first is
    layer 39 of position 22: the CPU picks expert 28 (gate 0.0738257), the device expert 89 (0.0738349), 1.2e-4
    apart. Layers 41 and 44 then follow.
  - **They are not arithmetic.** The CPU-vs-CUDA |dlogprob| is 1.14 at most. With the CPU's routing replayed on the
    device (`RUNNER_MOE_REPLAY`), it falls to 1.5e-3 at most and 2.1e-4 on average.
  - **Any backend will do this.** A top-8-of-128 router turns any cross-backend rounding into a different expert
    wherever two gates nearly tie. The MoE fixtures (2.1e-6) are too small to show it, and how often it happens
    depends on the prompt.

**Real mistakes on real files.** Two of the audit's host mutants, on SmolLM2, Qwen2.5-0.5B and Qwen3-0.6B
(`real-files.log`):
- **The attention scale ×1.05:** 2.05e-3 to 5.6e-3. That is 1.03× to 2.8× over 2e-3.
- **K unrotated:** 7.5e-2 to 9.8e-2.

**The decision: `GPU_DEV_FRAC` stays at 2e-3 for now.** 1e-3 would suit every dense file: 2.5× over IQ1_S, 13.8×
over the k-quants, and 2.05× under the attention-scale mistake. But it fails Qwen3-30B-A3B on a tie flip, a false
alarm for anyone gating a MoE file. At 2e-3 that file has 1.37× of room, prompt-dependent.

**The follow-up.** A routing-pinned mode for MoE files in `test-gpu-identity`:
- measure the arithmetic with the CPU's routing replayed on the device;
- report the flip rate separately;
- then take 1e-3 for dense and pinned-MoE files.

The comment at `GPU_DEV_FRAC` (`tests/test_gpu_identity.c`) now records these measurements. An earlier draft
proposed 5e-4 (before the IQ files) and then 1e-3 (before the MoE). Both were withdrawn on the measurements above.

**Still to measure:** sm_86 and Metal on the same files.

## Final check: the patch on the committed PTX (`final-check2.log`)

The whole patch, graph fix included, on runner main 550164c0ff13 with the committed PTX rebuilt:
- **gpu-identity:** the 11 recipe lines, as `make -n test-gates-core` expands them, all pass:
  - the toy fixtures at 1.75e-8 to 3.02e-7, k_temp_live among them at 1.02e-7 (limit 1e-5);
  - spark at 1.41e-6 to 3.7e-6 (limit 2e-5).
- **`make test-cuda-nvfp4`:** passes, at 9.60e-7 against 1e-4.
- **`make test-cuda-hybrids`:** passes on all eight fixtures.
- **`make test-makefile-sane`:** passes.
- **k_temp_live:** 1.02e-7 with graphs on and with `RUNNER_CUDA_GRAPH_OFF=1`.

`final-check.log` is the same check before the fix, with k_temp_live at 2e-3.

**Before landing:** another CUDA architecture (sm_86) needs to run the unmutated gates at these bounds, because
residue depends on the device's rounding. Spark's 5.4× is the thinnest margin, and the nvfp4 fixture's 1e-4 has
been measured on sm_120 only.

## Files

- `cuda-mutant-audit.py` runs the audit: `cuda-mutant-audit.py OUT.log [LIMIT]`. Set `ONLY=name,...` for a
  supplementary pass over named mutants.
- `make-table.py` builds the table above from the pass logs.
- The logs: `pass1-default-bound.log`, `pass1b-rope-default-bound.log`, `pass1c-rerun-default-bound.log`,
  `pass2-tight.log`, `pass2-fixed.log`, `pass2c-rerun.log`, `hybrids-1e-5.log`, `final-check.log`,
  `final-check2.log`, `trace-fields.log`, `moe-replay-trace-fields.log`, `real-files.log`, `iq-files.log`,
  `moe-flips.log`.
- `trace-fields.py RUNNER MODEL...` measures the CPU-against-CUDA trace fields per arm and mode.
- `make-table.py pass1-default-bound.log,pass1b-rope-default-bound.log pass2-fixed.log,pass2c-rerun.log` rebuilds
  the table.
