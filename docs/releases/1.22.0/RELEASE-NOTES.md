# 1.22.0 release notes (draft, not published)

**Limits first:** Claim 1 measures complete greedy-sequence identity, not
task quality. Its pre-registered result is Runner 40/100 versus llama.cpp
20/100, including 58 incomplete Runner records. Erratum 3, decided after
the verdicts, reports 94/100 versus 20/100 after completing the omitted
stop-token positions; it is reported beside, not instead of, that result.
Inkling's measured CPU prefill remains 0.70x llama.cpp's. Inkling CUDA is opt-in, with its speed result scoped to one
Q8_K_XL file and an aligned build; the available Windows device's 12 GB
cannot hold either admitted Inkling file. OLMo GPU training is refused,
compact KV has a cross-backend limit, and the 7B file does not fit the 8 GB
Metal device. Claim 2 is the provisional sixteen-thread half; its one-thread cells run
after the release. The OLMo CPU speed rows remain pending.
These notes are not a release-gate pass or a publication announcement.

## What changed

### Inkling 2 Mini: fast CPU by default

Run 3 of the pre-registered R4.48.24 bar passed at `7bf8b64d`, after the fast
route restored scalar `ink2_swiglu`. The measured file is
`unsloth/Inkling-2-Mini-GGUF` at `45fd0876`,
`Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` (SHA256 `2da51788`).
The native and STRICT builds were byte-identical. The reference is the
publisher's float32 implementation, eager attention on the CPU, with
weights dequantized/cast at use. These are that build's measurements, not
numbers from the still-unbuilt release artifacts.

| Fresh 2,000-position read | Fast mean KL vs publisher | Reference mean KL vs publisher | Fast / reference margin-qualified top-1 |
|---|---:|---:|---|
| Chat | 3.311354e-05 | 3.311425e-05 | 1707/1707 on both |
| Raw | 2.438582e-12 | 1.116652e-11 | 1659/1659 on both |

The bar was fast mean KL at most twice reference and 100% margin-qualified
top-1 on both fresh reads. Both 8-bit activation controls failed it, as
required. Fast rows were byte-identical at 1, 4 and 16 threads. The earlier
R4.48.19 envelope still failed on both reads, as the reference route's would;
that verdict is retained. Runs 1 and 2 of the new bar failed chat; run 3
passed after the scalar SwiGLU restoration.

At 16 threads, prefill at 512 tokens was 52.7 against llama.cpp `56eea9e4`'s
75.5 tok/s (0.70x); decode was 11.77 against 12.154 (0.968x), above the
predeclared 0.5x floor. Timed-row load ranged 21.43 to 71.59; engines ran in
blocks, not interleaved repetitions. Start residency was 1.0 at every gate.
The final rebased source `1fb25712` reproduced the four fast row captures
and the 100 answers' deterministic fields; decode was 12.05 against
12.217 tok/s (0.986x). These are not faster-than-llama.cpp claims.
[File identity, reference configuration, intervals and the full limits](../../inkling2-cpu-default.md).

### Claim 1: Runner versus llama.cpp, with an additional Unsloth column

**Limits:** complete greedy-sequence identity to the publisher's float32
reference on 100 pre-registered prompts, not task quality. The original
Runner measurement and its post-verdict correction are distinct; neither
is R4.48.24's newer CPU-route comparison. The exact UD-Q8_K_XL file was run
greedily, at most 256 tokens, with `reasoning_effort=low`, and verified by
teacher-forced reference forwards.

Unsloth's additional column uses its published CPU prebuilt
`b12649-mix-c7059d6`, run as published. A source build at the release
tag commit could not load Inkling 2 Mini.

**Pre-registered result, first and unchanged:**

| Engine | Identical / 100 [95% Wilson] |
|---|---|
| Runner's first run | 40/100 [0.309, 0.498] |
| llama.cpp | 20/100 [0.133, 0.289] |
| Unsloth published CPU prebuilt, additional | 16/100 [0.101, 0.244] |

| Pre-registered Runner pair | Difference [Newcombe Method 10 95%] | Exact McNemar p | Outcome |
|---|---|---:|---|
| Versus llama.cpp | +0.200 [+0.049, +0.339] | 0.0135 | higher |
| Versus Unsloth prebuilt | +0.240 [+0.099, +0.369] | 0.0015 | higher |

Runner had 58 incomplete records because its v1 client omitted the stop
token from its stored IDs. llama.cpp had no incomplete record and one
encoding discrepancy. The pre-registered rule counts these records as
they stood; the result above is not replaced.

**Beside it: erratum 3, decided after the verdicts:**

| Engine | Identical / 100 [95% Wilson] |
|---|---|
| Runner's completed original records | 94/100 [0.875, 0.972] |
| llama.cpp, unchanged | 20/100 [0.133, 0.289] |
| Unsloth published CPU prebuilt, additional | 16/100 [0.101, 0.244] |

| Erratum-3 Runner pair, decided after the verdicts | Difference [Newcombe 95%] | Exact McNemar p | Outcome |
|---|---|---:|---|
| Versus llama.cpp | +0.740 [+0.626, +0.816] | 2.039e-20 | higher |
| Versus Unsloth prebuilt | +0.780 [+0.668, +0.851] | 1.4e-21 | higher |

The response separately recorded the stop token. The erratum appended it
to the 58 early-stopped original records, without changing a prompt,
engine answer, setting, metric or llama.cpp record. Forty completed
sequences were teacher-forced afresh and 18 reused verified rows for
token-identical llama.cpp sequences, including the final stop. The other
42 Runner sequences were unchanged. Future cells use the v2 client.
Unsloth had zero encoding discrepancies and exact ties; one head-shape flag
affected no verdict. These remain dated original Claim 1 figures, not a
new measurement of the candidate fast route.

### Shipped fast route: separate R4.48.24 rule (3) result

R4.48.24 used the prompt set to check the proposed fast CPU default against
Runner's reference-route baseline. Each route was scored against the
publisher's greedy reference: fast 99/100, Runner reference 98/100. This is
the CPU-default gate, not release Claim 1's Runner-versus-llama.cpp result.
The paired table is 98 both, one fast-only, zero reference-only and one
neither; difference +1 percentage point, Newcombe 95% [-3.18, +5.78]
percentage points, exact McNemar p = 1.0: a statistical tie. It does not
show fast is better. The fast miss is c1_090 at token 88. No encoding
discrepancy, incomplete sequence, exact-tie position or head-shape flag was
reported in that separate run.

### Inkling's opt-in CUDA route

Set `RUNNER_INKLING2_CUDA=1` on the fast route in a CUDA build. Attention,
expert bodies and the loop's message gates run on the device; an eligible
single-position step keeps all 48 layer applications and the head resident,
with three host/device transfers per step. Adapters take the non-resident
path with their attention terms on the host. The default and reference
route remain CPU. Placement can keep expert stacks on the host or select
CPU with a reason.

The aligned real-file window measured Q8_K_XL (`2da51788`) at `be4e210a`, on
an sm_120 GPU (a 24 GB MIG slice). Decode was 23.269 against the two recorded
baselines' 21.308 / 21.307 tok/s: 1.092x. Prefill was 77.741 against
77.554 / 77.477, not lower and equal within noise. Every column aligned the
Inkling objects alike; shipped flags did not change. The earlier unaligned
gap was traced to host code layout.

The final implementation's CUDA gate passed 284/284 on Linux sm_120 and
Windows sm_120; Windows passed STRICT and release-recipe builds. A positive
control preceded racecheck and synccheck filtered to the three changed
norm/read kernels: zero errors, all 284 checks passing under each. These
are scoped kernel checks. Both Q8_K_XL and Q4_K_XL also have independent
CUDA file-smoke passes on `linux-x86_64-cuda`, recorded at `299cba11` with
the opt-in in their provenance. The Windows device's 12 GB memory limit
still prevents real-file recordings there. [Execution and evidence](../../inkling2-cuda.md).

### OLMo 2 and OLMo 3 admitted

The `olmo2` architecture is admitted on CPU, Metal and CUDA, with LoRA
training on the CPU. Real-file CPU and CUDA records cover
OLMo-2-0425-1B-Instruct-F16 (`376249df`) and Olmo-3-7B-Think-Q8_0
(`3a4d7dda`); Metal's real-file record is the 1B. OLMo 3's Metal path is
checked on generated files only, and the 7B does not fit with runtime state
on the available 8 GB Metal device. The base 1B's evidence is separate,
not inherited by the Instruct row.

| File, CPU / f16 KV / 2,000 positions | Runner mean KL [95% interval] vs publisher float32 | Plain / margin-qualified top-1 | llama.cpp mean KL [95% interval] vs publisher float32 | Plain / margin-qualified top-1 |
|---|---|---|---|---|
| 1B Instruct F16 | 1.0642e-04 [6.1904e-05, 1.5615e-04] | 100% / 100% | 1.1261e-04 [6.6509e-05, 1.6433e-04] | 100% / 100% |
| 7B Think Q8_0 | 1.3671e-03 [1.1975e-03, 1.5472e-03] | 98.55% / 100% | 9.1434e-02 [8.5543e-02, 9.8364e-02] | 85.25% / 95.77% |

The 7B bar passed: 100% margin-qualified top-1 and separated intervals in
Runner's favour on the file as published. The reference uses the publisher
checkpoint in float32 on CPU (the BF16 7B weights upcast at use). Templates
are checked token for token; modern tool requests to the native OLMo
protocol are refused rather than dropped. A non-default YaRN attention
factor, experts and GPU training are refused by name. The compact-KV limit
remains: fp4 KV can amplify cross-backend differences at wide direct-model
batches. [Exact files, reference configurations and both reads](../../compat-reports/olmo2-admission-2026-11-24/README.md).

Q8_0 CUDA tensor-core prefill is on by default for `olmo2`; other weight
types stay scalar until measured. Against TC off on one sm_120 GPU, the
1B Instruct Q8_0 prefill medians are 7.31x / 5.30x at 512 / 2048 tokens;
7B Think Q8_0 is 9.07x / 6.80x. Decode medians are 0.996x / 1.001x.
These are tolerance routes, not bit identity. TC-on prefill at 512 is still
0.27x / 0.29x llama.cpp `b12640`'s, context outside that promotion bar.
The 1B TC speed file is Q8_0, distinct from the admitted F16 file; the speed
row does not admit a quantized sibling. The 7B's forced-TC float32 read is
1.3625e-03 versus 1.3717e-03 with TC off, 100% margin-qualified on both;
Wilcoxon p 0.103, McNemar p 1.0. [Promotion evidence](../../tc-promotion-evidence.md#olmo2-q8_0-r4509-2026-11-24).

**PLACEHOLDER-OLMO2-CPU-SPEED:** insert the fixed per-file CPU speed rows,
baselines, build/flags, settings, ranges and limits when the measurements
finish. No CPU throughput improvement is claimed here yet.

### Windows gate fixes

- The health-wait and timeout tests start the base Python interpreter,
  avoiding a Windows venv launcher's child retaining a killed process's log.
  The health-wait reproduction moved from 1/10 to 10/10 passes.
- Recurrent context tests read the CPU-only declarations from `arch.h`.
  CUDA's lfm2 cases now test the promised CPU fallback instead of reporting
  a forbidden device skip: 528 passed, 15 allowed skips in the recorded gate.
- GCC's AVX-512 fast-math test build avoids the GCC 16.2 vectorizer crash
  with a scoped `novector` pragma. Shipped non-fast-math `quants.o` is
  unchanged by that workaround. These test fixes are not release-artifact
  validation; the candidate's Windows gates still have to run.

### Claim 2: sixteen-thread half, provisional INFORMATION

**Limits first:** on this Inkling CPU route, Runner decodes concurrent MoE
requests one sequence at a time; gathering slots is not a shared
multi-sequence matrix product. Only the sixteen-thread half has run.
One-thread cells run after the release. No final full-matrix Claim 2
wording or cross-engine speed conclusion is drawn.

Unsloth's additional column uses its published CPU prebuilt
`b12649-mix-c7059d6`, run as published. A source build at the release
tag commit could not load Inkling 2 Mini.

Runner's two cells lost up to 3.7% of the file's pages to host reclaim while
llama.cpp's stayed resident, so the times are information, not a speed
comparison.

The timing table names the default cells selected by the prewritten rule,
flash attention on and micro-batch 512, as “quoted”. The other measured
settings are shown beside them as information, not selected by their
speed. Throughput is the generated-ID count divided by wall time, including
prefill, over 20 rounds. A batch wall runs from barrier release through
the ninth response; an alone wall covers its one response. The usage-based
median is shown when the client count differs, including Runner's appended
stop token. Every timing has its host-wide load and sampled file residency
beside it.

| Sixteen-thread cell | Generated-ID tok/s median [IQR] | Usage-based median when different | Load median / peak | File residency minimum | Samples below 0.99 |
|---|---:|---:|---:|---:|---:|
| Runner alone (quoted) | 11.18 [10.69, 12.49] | 11.18 | 49.87 / 60.50 | 0.963599 | 32.4% |
| Runner batch (quoted) | 11.80 [11.44, 12.07] | 11.76 | 56.31 / 158.32 | 0.962821 | 92.4% |
| llama.cpp alone, FA on, ub 512 (quoted) | 10.49 [9.65, 11.11] | - | 51.84 / 80.02 | 1.000000 | 0.0% |
| llama.cpp alone, FA on, ub 1 (information) | 8.23 [6.91, 9.40] | - | 32.18 / 44.14 | 1.000000 | 0.0% |
| llama.cpp alone, FA off, ub 512 (information) | 10.64 [9.91, 11.28] | - | 38.56 / 66.20 | 1.000000 | 0.0% |
| llama.cpp alone, FA off, ub 1 (information) | 8.70 [6.50, 9.35] | - | 24.48 / 25.90 | 1.000000 | 0.0% |
| llama.cpp batch, FA on, ub 512 (quoted) | 18.55 [17.98, 20.95] | - | 29.80 / 195.27 | 0.999989 | 0.0% |
| llama.cpp batch, FA on, ub 1 (information) | 9.92 [9.82, 10.17] | - | 41.16 / 1356.19 | 1.000000 | 0.0% |
| llama.cpp batch, FA off, ub 512 (information) | 17.02 [16.38, 19.45] | - | 58.40 / 337.58 | 1.000000 | 0.0% |
| llama.cpp batch, FA off, ub 1 (information) | 9.19 [8.87, 9.39] | - | 45.20 / 115.22 | 0.999461 | 0.0% |
| Unsloth prebuilt alone, FA on, ub 512 (quoted) | 10.86 [10.29, 11.22] | - | 36.46 / 69.35 | 1.000000 | 0.0% |
| Unsloth prebuilt alone, FA on, ub 1 (information) | 8.60 [6.91, 9.46] | - | 46.93 / 86.08 | 1.000000 | 0.0% |
| Unsloth prebuilt alone, FA off, ub 512 (information) | 10.53 [10.03, 11.31] | - | 64.03 / 79.75 | 0.999541 | 0.0% |
| Unsloth prebuilt alone, FA off, ub 1 (information) | 8.14 [6.14, 8.96] | - | 60.89 / 97.39 | 1.000000 | 0.0% |
| Unsloth prebuilt batch, FA on, ub 512 (quoted) | 18.50 [17.79, 19.47] | - | 50.23 / 102.02 | 1.000000 | 0.0% |
| Unsloth prebuilt batch, FA on, ub 1 (information) | 10.49 [9.68, 10.82] | - | 35.65 / 79.54 | 0.999855 | 0.0% |
| Unsloth prebuilt batch, FA off, ub 512 (information) | 18.38 [17.92, 18.87] | - | 33.75 / 72.73 | 1.000000 | 0.0% |
| Unsloth prebuilt batch, FA off, ub 1 (information) | 9.70 [9.47, 9.87] | - | 27.57 / 72.70 | 0.999694 | 0.0% |

A host-wide load spike overlaps the non-default llama.cpp
flash-on/micro-batch-1 batch cell; no request there timed out or retried.

Each measured cell has 20 complete target rounds, with prompt IDs equal
across each engine's cells and no incomplete or torn record. Identity
checks generated IDs and their decoded bytes, including the stop token.
The counts below cover the measured cells only.

| Engine | Measured sixteen-thread cells | Targets identical across those cells [95% Wilson] |
|---|---:|---|
| Runner | 2 | 20/20 [0.839, 1.000] |
| llama.cpp | 8 | 2/20 [0.028, 0.301] |
| Unsloth prebuilt, additional | 8 | 2/20 [0.028, 0.301] |

The one-thread factors remain pending. Measured factor counts use the
alone, sixteen-thread, flash-attention-on, micro-batch-512 baseline (Runner
has no flash/micro-batch factor).

| Factor changing only that setting, at sixteen threads | Runner differing targets | llama.cpp differing targets | Unsloth prebuilt differing targets, additional |
|---|---:|---:|---:|
| Alone versus batched | 0/20 | 15/20 | 15/20 |
| Flash attention on versus off | n/a | 18/20 | 15/20 |
| Micro-batch 512 versus 1 | n/a | 17/20 | 16/20 |

llama.cpp's co-batching at micro-batch 512 is **inferred from the source
and the slot log**. At micro-batch 1, concurrently active slots still decode
one token at a time inside the call. The additional Unsloth column's
co-batching is reported with the same inference qualification. Runner's
batch counters count gathered slots; the measured-build source check
confirms sequential one-row forwards. The debug route lines named in the
pre-registration cannot appear on this architecture, so those counters
and the source check are the disclosed evidence.

Companions are information: the eight fixed companions were invariant
across measured batched cells for Runner 8/8, llama.cpp 0/8 and Unsloth
0/8. Runner has one measured batched configuration. Unsloth's column has
no Claim 2 pairs and no combined label. The status remains provisional
INFORMATION until the one-thread cells complete.

### Claim 3: FINAL, pre-registered outcome TIE

**Limits first:** the pre-registered outcome is **TIE**, not superiority to
the budgeted comparator. Runner with loop guard and reasoning budget has
37/40 usable turns and zero observed loops, versus budgeted llama.cpp's
39/40 usable turns; the paired difference is -0.050, Newcombe 95%
[-0.176, +0.067], exact McNemar p 0.625. “Usable” is the specified output-form
metric, not proof that a code fix works. `unit_test_pass` was **not
computed**: there was no isolation for executing model-written code.
This is a disclosed omission of the secondary diagnostic.

Unsloth's additional column uses its published CPU prebuilt
`b12649-mix-c7059d6`, run as published. A source build at the release
tag commit could not load Inkling 2 Mini.

The set has 40 code-debugging prompts, 20 plain and 20 tool prompts, greedy
at `reasoning_effort=high` on UD-Q8_K_XL. `off` leaves the controls off;
`lg` enables Runner's loop guard and `lg-rb` combines it with the
reasoning budget. The comparator `rb` arms have a reasoning budget.
The loop metric combines span repetition with unclosed-thinking exhaustion;
zero observed cases is scoped to this set.

| Arm | Usable turns [95% Wilson] | Loop/exhaustion turns [95% Wilson] |
|---|---|---|
| runner-off | 24/40 [0.446, 0.737] | 15/40 [0.242, 0.530] |
| runner-lg | 25/40 [0.470, 0.758] | 14/40 [0.221, 0.505] |
| runner-lg-rb | 37/40 [0.801, 0.974] | 0/40 [0.000, 0.088] |
| llama-off | 25/40 [0.470, 0.758] | 14/40 [0.221, 0.505] |
| llama-rb | 39/40 [0.871, 0.996] | 1/40 [0.004, 0.129] |
| unsloth-off | 26/40 [0.495, 0.779] | 14/40 [0.221, 0.505] |
| unsloth-rb | 37/40 [0.801, 0.974] | 2/40 [0.014, 0.165] |

The combined Runner arm is higher than unguarded Runner and unrestricted
llama.cpp, but has no statistically distinguishable usable-turn difference
from budgeted llama.cpp. The primary outcome above stays TIE.

| Pre-registered pair: Runner loop guard + budget versus | Usable-turn difference [Newcombe 95%] | Exact McNemar p | Outcome |
|---|---|---:|---|
| Runner off | +0.325 [+0.166, +0.474] | 0.0002441 | higher |
| llama.cpp off | +0.300 [+0.125, +0.459] | 0.001831 | higher |
| llama.cpp budgeted | -0.050 [-0.176, +0.067] | 0.625 | tie |

In the combined Runner arm the loop guard fired on 0/40 turns and the
reasoning budget forced a close on 18/40. The zero-loop result must not be
described as observed loop-guard interventions. Metric-field interpretations
were documented after the five Runner/mainline arms existed, with no change
to thresholds, arms, settings or interval methods; the omitted secondary
unit-test diagnostic is disclosed above.

The Unsloth arms and pairs are additional information, not a combined
outcome or a change to the pre-registered TIE label.

| Additional Unsloth pair: Runner loop guard + budget versus | Usable-turn difference [Newcombe 95%] | Exact McNemar p | Pair outcome only |
|---|---|---:|---|
| Unsloth prebuilt off | +0.275 [+0.105, +0.433] | 0.003418 | higher |
| Unsloth prebuilt budgeted | +0.000 [-0.128, +0.128] | 1 | tie |

## Behaviour changes

- The new Inkling family uses the fast CPU route by default; reference is
  still selectable with `RUNNER_INKLING2_ROUTE=reference`. CUDA stays opt-in.
  Generated tokens may change with numerical paths; replay remains bound to
  the binary/model named by the record, as the 1.x policy already states.
- `olmo2` leaves `architectures_pending`. Protocol and layout refusals are
  explicit; admitting these files does not admit unmeasured quantized siblings.
- The Windows gate fixes above do not alter runtime behavior. The minor
  version adds model/execution support without removing 1.x interfaces.

## Limits

The first paragraph and linked records are part of each claim. No claim of
universal speed, cross-backend bit identity, all-100 greedy identity, GPU
training for OLMo, or real-file Metal validation of the 7B is made. The CPU
and CUDA timing windows are different measurements on named builds. The OLMo CPU-speed
placeholder and release gates must be resolved before these notes are final;
Claim 2 must retain its provisional sixteen-thread qualification.

## Gates

**DRAFT / NOT RELEASE-READY.** Source main was `83da2dc3`; the candidate
version pin is `0ea15c6e`. This prep has no tagged assets or version-specific
1.22.0 compatibility/full-CUDA records. The generic Make dry run prints the
plan only. See [the gate record](GATE-RECORD.md) for artifact, Windows and
Mac prerequisites and the remaining publication steps. No pending gate is
counted as a pass, and no tag, release, container or site was published.
