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
Metal device. Claims 2 and 3 and the OLMo CPU speed rows below are pending.
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

### Claim 1: Runner versus llama.cpp

**Limits:** this is complete greedy-sequence identity to the publisher's
float32 reference on 100 pre-registered prompts, not task quality. The
original Runner measurement and its post-verdict correction are distinct;
neither is R4.48.24's newer fast-versus-reference-route comparison.
The exact UD-Q8_K_XL file was run greedily, at most 256 tokens, with
`reasoning_effort=low`, and verified by teacher-forced reference forwards.

| Analysis, in reporting order | Runner identical / 100 [95% Wilson] | llama.cpp identical / 100 [95% Wilson] |
|---|---|---|
| Pre-registered result, unchanged | 40/100 [0.309, 0.498] | 20/100 [0.133, 0.289] |
| Erratum 3, decided after the verdicts | 94/100 [0.875, 0.972] | 20/100 [0.133, 0.289] |

**Pre-registered paired result:** +0.200, Newcombe Method 10 95% interval
[+0.049, +0.339], exact McNemar p = 0.0135, outcome **higher**. The paired
table is zero both, 40 Runner-only, 20 llama.cpp-only and 40 neither.
Runner had 58 incomplete records; llama.cpp had no incomplete record and
one encoding discrepancy. These are the result and counting rule as
pre-registered, reported first and unchanged.

**Erratum 3 disclosure, decided after the verdicts:** Runner's v1 client
stored the receipt's generated IDs without the stop token even though the
response separately recorded `stop_token_id`. Its 58 early-stopped answers
therefore failed the completeness rule. The erratum read that field from
the original records and appended the omitted stop token; no prompt,
engine answer, setting, metric or llama.cpp record was changed. Of the
58 completed sequences, 40 were teacher-forced afresh and 18 reused verified
rows for token-identical llama.cpp sequences, including the final stop.
The other 42 Runner sequences and all llama.cpp sequences were unchanged.

The resulting 94/100 versus 20/100 is the adjacent erratum figure, not a
replacement pre-registered verdict. Its paired difference is +0.740,
Newcombe 95% [+0.626, +0.816], exact McNemar p = 2.039e-20; the paired table
is 18 both, 76 Runner-only, two llama.cpp-only and four neither. Future
cells use the v2 client, which retains the stop token. These figures refer
to the original Claim 1 records; they are not a new measurement of the
candidate fast route.

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

### Measurements awaiting their final analysis

**PLACEHOLDER-CLAIM-2:** alone-versus-batched answer identity. Insert the
pre-registered verdict, per-engine and batch results, denominators, failures
and provenance. Do not infer this result from thread identity or kernel gates.

**PLACEHOLDER-CLAIM-3:** reasoning-loop guard on the model's own loops.
Insert the pre-registered analysis, controls, per-arm outcomes, failures and
provenance. Interim arm completion is not a final result.

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
and CUDA timing windows are different measurements on named builds. Pending
placeholders and release gates must be resolved before these notes are final.

## Gates

**DRAFT / NOT RELEASE-READY.** Source main was `83da2dc3`; the candidate
version pin is `0ea15c6e`. This prep has no tagged assets or version-specific
1.22.0 compatibility/full-CUDA records. The generic Make dry run prints the
plan only. See [the gate record](GATE-RECORD.md) for artifact, Windows and
Mac prerequisites and the remaining publication steps. No pending gate is
counted as a pass, and no tag, release, container or site was published.
