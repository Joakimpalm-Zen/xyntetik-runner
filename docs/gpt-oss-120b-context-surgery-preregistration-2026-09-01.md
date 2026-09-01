# gpt-oss-120B context surgery — preregistration

Date: 2026-09-01 UTC

This ledger was frozen before any inference, retrieval, scoring, or benchmark
result from a context-extended candidate existed. The 262K GGUF file write had
already started; its only observed output was the mechanical header pre-pass
(`context_length 131072 -> 262144`, YaRN factor `32 -> 64`, and the unchanged
tensor-type histogram). No model output or quality measurement had run.

## Question

Can the released `gpt-oss-120b` be made usefully reliable at 262,144 tokens,
then 524,288 tokens, while preserving its short-context coding and Harmony
tool behavior? The context length is the requirement. Compounded YaRN is the
first cheap candidate, not a prescribed answer.

## Frozen parent and compiler

- Parent: `models/gpt-oss-120b-MXFP4/gpt-oss-120b-MXFP4.gguf`
- Parent bytes: 63,387,346,208
- Parent SHA-256 recorded by the existing M5 validation:
  `582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d`
- Architecture metadata: context 131,072; native YaRN factor 32 over original
  context 4,096; 36 layers; alternating SWA-128 and full attention.
- Runner branch: `context-surgery`
- Runner compiler commit before this ledger: `59e512e`
- Final serving reference: a pinned stock llama.cpp revision, if a candidate
  survives Runner's gates.

## Candidate ladder

candidate | compile contract | status before measurement
--- | --- | ---
parent control | context 131,072; YaRN x32 | frozen control
262K-A | context 262,144; YaRN x64 | compile in progress
524K-A | context 524,288; YaRN x128 | compile only; evaluation gated on 262K-A
trained 262K | method selected after no-training bakeoff | not authorized by this ledger
trained 524K | progressive extension of a passing 262K model | not authorized by this ledger

The two A candidates are metadata-only. Runner must independently reparse the
parent and output and byte-compare all tensor payloads before it may report a
successful build. QAT and expert requantization are forbidden on this lane.

## Hypotheses and predictions

1. **H1 — mechanical identity.** Both A candidates retain all 687 tensor
   payloads byte-for-byte and change only context/YaRN metadata. Expected:
   held, because this is a compiler invariant rather than a model-quality
   claim.
2. **H2 — short-context preservation.** 262K-A remains coherent and stays
   inside the parent self-sensitivity envelope at short context. Lean: held at
   2K, increasingly uncertain at 8K/32K because changing the effective YaRN
   factor also changes the magnitude correction.
3. **H3 — 262K retrieval.** Inference-only x64 retrieves exact random passkeys
   at 250K across shallow, middle, and deep placements. Lean: fails or is
   brittle; this screen sizes the training requirement.
4. **H4 — 524K retrieval.** Inference-only x128 remains useful at 500K. Lean:
   fails. The run is forbidden unless 262K-A first clears coherence and the
   intermediate retrieval gate.
5. **H5 — task integration.** A candidate that retrieves one passkey may still
   fail multi-needle ordering or coding/tool behavior. Expected: retrieval
   degrades before short task behavior; measured separately, never inferred.

## Fixed measurements

### Mechanical and runtime anchors

- SHA-256 for parent, candidate, Runner binary, and every machine-written
  provenance record.
- Direct GGUF metadata read for context, YaRN type/factor/original context,
  tensor count, tensor shapes/types, and tensor-payload equality.
- Runner `--fit` at 131K/262K/524K with both F16 and Q8 KV estimates.
- A one-token absolute anchor whose expected next token is recorded from the
  frozen parent before comparing variants.
- Parent control must pass every probe at 131K. A harness that fails the parent
  is invalid, not evidence against a candidate.

### Coherence screen

At each admitted length, run three deterministic Harmony prompts: factual
prose, a small coding task, and a structured tool choice. A candidate is killed
at that length if at least two of three are empty, repetitive, protocol-leaking,
or plainly incoherent. Raw outputs are retained.

### Retrieval screen

- Deterministic synthetic documents with cryptographically generated passkeys;
  the answer is known outside Runner.
- Lengths: 32K, 64K, 128K, 160K, 192K, 224K, 250K, 262K; add 384K/500K/524K
  only after the 262K gate.
- Needle depths: 10%, 50%, 90%; three seeds per cell.
- Exact-match scoring after normalization fixed in the harness before runs.
- Multi-needle control at the longest surviving length: five key/value pairs,
  query order shuffled, all five required in the requested order.

Kill 262K-A before expensive evaluation if it scores below 50% exact at or
below 192K, or if it fails all three depths at two consecutive lengths. Promote
it to full RULER only if it scores at least 8/9 at 250K and passes the
multi-needle control. These are screening criteria, not the final product bar.

### Short-context regression

- Teacher-forced parent/candidate comparison on the repository mixed corpus at
  2K, 8K, and 32K prefixes.
- Report mean KLD, plain top-1, margin-qualified top-1, and per-domain rows.
- Coding and Harmony tool-call cases are paired and greedily decoded.
- The final product bar remains no more than two percentage points of paired
  coding/tool regression; a small smoke cannot establish that bar and must be
  labelled a smoke.

### Final product gate

Only a trained or inference-only candidate that survives the screens may run
the expensive acceptance suite:

- RULER at 262K at least 85% of the same model's 131K baseline, plus absolute
  per-task floors and confidence intervals.
- Multi-needle retrieval at 250K in the served GGUF under pinned llama.cpp.
- No more than two percentage points paired coding/tool regression.
- Runner-to-llama.cpp comparison inside the model's measured self-sensitivity
  envelope; literal `KL = 0` is not assumed for sensitive MXFP4 routing.

## Stop and publication rules

- Stop a candidate on its first preregistered kill condition. Do not tune the
  probe after seeing its answer.
- A failed A candidate is a useful negative result and may ship as a report,
  manifest, and commands, but not as a recommended model artifact.
- Do not begin 1B-token continued training from these results alone. First run
  a 25–100M-token attention/norm pilot with experts and router frozen, compare
  at least two position-scaling methods, and preregister that experiment in a
  new ledger.
- Preserve corrections and failed expectations in place. Do not delete or
  relabel an unfavorable row.

## Environment recorded before measurement

- Host RAM: 250 GiB; 229 GiB available at initialization.
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
- Build toolchain: Conda environment `ccbuild`.
- Target deployment evidence already in tree: Apple M5 Max, 128 GB unified
  memory, fully resident parent with zero swap.

## Amendments after measurement began

### 2026-09-01T16:57:15Z — invalid parent control, evaluator corrected

The first 4K parent retrieval control returned empty visible content for all
nine cases and is **invalid**, not a 0/9 model result. Every request generated
exactly the evaluator's 24-token ceiling. Runner's documented Harmony mapping
placed those tokens in `reasoning_content`; the unconstrained request exhausted
its shared ceiling before reaching the final channel. The untouched raw record
is `parent-retrieval-4k.json`, SHA-256
`8ec4c41221cbce472ff15d47328d359e0602c3f2d0ce4067a0ab2e608e1e30ef`.

Before viewing any candidate output, `scripts/kv-quality.py` was changed to
send `enable_thinking:false`. Retrieval is the measured task and the 24-token
budget is intended for its short answer, not hidden chain-of-thought. The
parent control will be rerun under that fixed request shape. Hypotheses,
lengths, seeds, depths, scoring, and kill thresholds are unchanged; the invalid
record remains retained as an instrument correction.
