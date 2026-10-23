# Model support tiers

A tier is a maintenance and validation commitment: it says which model files
Runner re-checks on every release and what happens when one of them breaks. It
is not a compatibility list and not a correctness verdict. What loads is the
[support matrix](../MANUAL.md#support-matrix), `runner --caps` and
`runner --doctor`; what was measured is in the
[model evidence](model-evidence.md) and the admission documents each row
links. Every row on this page is rendered from
[`docs/model-tiers.json`](model-tiers.json), and every mark carries the Runner
version and date of the run that wrote it.

Three rules no tier overrides:

- An architecture or quantization format that loaded in 1.0.0 keeps loading in
  every 1.x release ([versioning policy](versioning.md)). A tier never removes
  a path or waives a compatibility regression.
- No unmeasured configuration inherits a pass from a neighbouring row. A row is
  a claim about the file and the backends it names, nothing wider.
- A historical result is never presented as a current one: a mark names the
  version and date of its run, and an old mark stays dated.

## The four statuses

| Status | Runner's commitment |
|---|---|
| Tier 1, actively validated | Scheduled validation and proactive maintenance for the configurations the row names. Correctness claims are tied to recorded evidence with a version and date. Gated on every release. |
| Tier 2, maintained on report | Previously admitted; no scheduled full-model revalidation. Existing regression tests keep running, and a change to a shared execution path (kernel, architecture code, tokenizer, template, scheduler) is tested with representative coverage that includes the affected families. A reproducible report is investigated, the fix is validated on the affected configuration where feasible, and it leaves a regression test behind. |
| Tier 3, unmaintained | No dedicated validation or model-specific development. The implementation and the dated evidence remain; current-build correctness is not claimed. The 1.x loading promise still applies. No members: the first entry needs an owner decision and a transition-log entry. |
| Upcoming, planned admission | A Tier 1 candidate whose implementation or validation is incomplete. Listed apart from Tier 1 and never counted as actively supported. Promotion is by evidence, not by whether the weights fit in memory. |

Moving a model down a tier changes the commitment; it does not erase earlier
measurements or say the model is broken. Every move is appended to the row's
history and appears in the [transition log](#transition-log) with its tier,
date, reason and last validated release.

## What Tier 1 claims, and what it does not

Tier 1 promises ongoing validation against stated criteria. It does not promise
that a model's answers are correct. Each row keeps three things apart:

- **Numerical correctness**, against the publisher's float32 implementation.
  Where no float32 reference is feasible, the row says "anchor: agreement with
  another engine, not fidelity" in those words.
- **Protocol correctness**: tokenizer, chat template, tool protocol and API
  surfaces, for the features the row lists. A feature not listed is not
  claimed.
- **Task quality**: benchmarks, which do not follow from the fidelity bar and
  which no tier claims.

A row leads with what it does and does not claim. A row for a model that ships
a vision component Runner does not load starts with "Text path only", and an
experimental row starts with "Experimental".

## The envelope and the evidence bar

Each Tier 1 row names the upstream checkpoint, the file with its sha256 and
the quantization, the features it claims, its numerical anchor, the context
tested and the known limits, what is still missing, the device class and
backend of its per-release smoke, a mark per backend, the file's LoRA
compatibility, and its tier history. A head or an adapter also names the row
it runs with.

The LoRA compatibility is the file's own `runner --adapt-info` answer,
recorded by the gate run: whether the LoRA backward admits the file
(training) and whether an adapter can be served on it, each with the refusal
sentence when not. It says the backward admits the file; it is not a claim
about the quality of training it. For a head or an adapter the answer is the
trunk's or the base's, the model the configuration trains and serves on.

The evidence bar for Tier 1, and for promotion from Upcoming:

1. Numerical: `scripts/gold-logits.py` against the publisher's float32
   implementation on at least the family's smallest member where one exists;
   otherwise the row says "anchor: agreement with another engine, not
   fidelity".
2. Protocol: a tokenizer differential and chat-template conformance against
   the publisher's template, token for token
   (`scripts/template-conformance.py --require-tokens`); the tool protocol
   where the family has one.
3. Repeatable: a pinned greedy output on the named file, which a later build is
   compared against (the per-release smoke below).
4. A recorded run on a real device class, written by the machine.

External hardware and community contributions count only as reproducible
records with a realistic path to revalidation.

## Two gates, and the release refusal

**The per-release smoke, every Tier 1 file**, on the device class its row
names. `scripts/tier-smoke.py` hashes the file against its row, records the
file's `--adapt-info` answer, runs one greedy prompt (the eight planets of the
Solar System in order, 48 tokens, `--temp 0 -c 512 -b 64`) and compares stdout
byte for byte with the pin under
[`tests/compatibility/tier-smoke/`](../tests/compatibility/tier-smoke.json),
one pin per device class and backend. The continuation must name "Mercury",
a fact from outside Runner, and a pin holds at least 96 generated bytes, so
the comparison has something to catch. The anchors are the file's sha256 and
that fact; the pin itself only says whether this build reproduces the one
that wrote it.

**Full re-certification**, the evidence bar above, only when a path the model
uses changes: its kernels, its architecture code, its tokenizer, its template
or the scheduler on its path. The pull request that changes such a path names
the Tier 1 rows it re-certifies.

`scripts/check-release.py` refuses a tag while any Tier 1 row is not current,
the same way it refuses stale device evidence: the row's smoke cell (the
backend and device class the row names) was never recorded or was recorded on
another class, it is older than the row's age limit (45 days unless the row
sets a tighter one), or any backend of the row carries a recorded failure. A
head or an adapter runs with the row it names, so every Tier 1 row can be
gated; `scripts/model-tiers.py --check` refuses one that cannot.

## Run the smoke and record it

On the device class the row names, from a clean checkout of the commit being
checked and a `runner` built from it. Give every file by its path: the lab
hosts keep models in a workspace directory, not in the Hugging Face cache, and
the smoke hashes the path it is given against the row before it runs
anything.

```sh
# optional, on a host whose disk throttles: read the file once, so the
# memory-mapped load does not fault in slowly during the run
dd if=/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf of=/dev/null bs=64M

# compare with the pin and write the result into the roster
python3 scripts/tier-smoke.py --row qwen3.8-27b-gsq-rco-iq3_s-recovered \
    --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf --record

# a head or an adapter: name the file of the row it runs with, too
python3 scripts/tier-smoke.py --row qwen3.8-27b-mtp-q8_0 \
    --model /models/Qwen3.8-27B-MTP-Q8_0.gguf \
    --with-model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf --record

# every row this machine's class gates, files found by name under a directory
python3 scripts/tier-smoke.py --models /models --record
```

The device class comes from the binary's own `--caps`, matched against
[`docs/device-evidence.json`](device-evidence.json). The backend defaults to
the row's; `--backend cpu` adds a CPU cell on a GPU host. `--record` writes the
cell and the `--adapt-info` answer into the roster and re-renders the README
table and this page: commit `docs/model-tiers.json`, `README.md` and this file
together.

The first run on a row and backend has no pin: run it once with `--pin`
instead of `--record`, commit the pin, then run with `--record` from the clean
tree that holds it. When a release changes generated tokens on purpose (the
[versioning policy](versioning.md) allows it), the pin mismatches. Read the
difference; then `scripts/tier-smoke.py --retire --row ID --backend B
--device-class C` removes the pin, its manifest entry and the cell that cites
it, and the row is pinned again, with the reason in the commit. A pin is never
overwritten in place.

The checks: `scripts/model-tiers.py --check` (the roster's structure),
`--stale` (the release gate's question; `--today` fixes the day for a test)
and `--docs-check` (the README table and this page against the roster).

<!-- model-tiers:rows:begin (rendered from docs/model-tiers.json by scripts/model-tiers.py --write-docs; do not edit) -->

<a id="tier1"></a>
## Tier 1, actively validated

### Text path only: Cloudflare Clef (clef, on Qwen3.8-27B)

<a id="cloudflare-clef-iq3_m"></a>
#### `Cloudflare_clef-IQ3_M.gguf`

Text path only: a decision model served at /v1/systemone; requests with images or videos are refused. The model card's two examples reproduce on the CPU with this file, which shows the head binds and decides; it is not a fidelity measurement of the quantized file.

- **File:** `Cloudflare_clef-IQ3_M.gguf` (sha256 `4977cdedb8784ed414ae03e17ff0ee33f1be1d6bc89370114ebcbc6c5f818975`, 13,776,408,640 bytes, from [bartowski/Cloudflare_clef-GGUF](https://huggingface.co/bartowski/Cloudflare_clef-GGUF) at revision `9d3b28af7cfe`).
- **Upstream checkpoint:** Cloudflare/clef at revision `0b331204bb13`.
- **Quantization:** IQ3_M.
- **Features claimed:** decisions. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** The publisher's joint_schema_model.py in float32 on a small Clef (the Qwen3.5-0.8B backbone with a joint head): 15/15 input ids and spans, max |dlogit| 1.5e-4 on CUDA. Not on this checkpoint.
- **Context tested:** not recorded.
- **Known limitations:** Text only: a request with images or videos is refused.
- **Still missing:** A float32 gold run on a released checkpoint against its BF16 GGUF; the decision surface in the per-release smoke.
- **Evidence:** [docs/clef-admission-2026-10-16.md](clef-admission-2026-10-16.md).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** [n/a](../MANUAL.md#support-matrix): Clef's qwen35 backbone has no Metal path; a Metal build runs it on the CPU.
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Qwen3.8-27B

<a id="qwen3.8-27b-ud-q4_k_m"></a>
#### `Qwen3.8-27B-UD-Q4_K_M.gguf`

Unsloth's dynamic Q4_K_M of Qwen3.8-27B, text path: it passes the quant bar against the Q8_0 sibling; the qwen35 CUDA path was measured on other quantizations of the same model.

- **File:** `Qwen3.8-27B-UD-Q4_K_M.gguf` (sha256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`, 16,464,440,224 bytes, from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision `4ca720788d1e`).
- **Upstream checkpoint:** Qwen/Qwen3.8-27B.
- **Quantization:** UD-Q4_K_M (with 4 IQ3_S tensors).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: Qwen3.5-0.8B bf16 against Qwen's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass). This file: UD-Q4_K_M vs Q8_0 bar PASS (100 positions, mean KLD 0.0016, top-1 100%); tokenizer 0/721, template 21/21.
- **Context tested:** not recorded.
- **Still missing:** A CPU row on the file; pinned greedy output.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** [n/a](../MANUAL.md#support-matrix): The qwen35 Gated DeltaNet hybrid has no Metal path; a Metal build runs this file on the CPU.
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="qwen3.8-27b-gsq-rco-iq3_s-recovered"></a>
#### `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf`

ISTA-DASLab's IQ3_S file with every quantised block's fp16 scales retrained against the BF16 parent, text path: it reads inside the fidelity bar against that parent and is not a lossless Qwen3.8-27B.

- **File:** `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf` (sha256 `873df48dd3ecf205f73c29716e80fbe766e4202ef4ed394b2c24607a8628458f`, 11,771,546,784 bytes, from [Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF) at revision `7b10f923a7f7`).
- **Upstream checkpoint:** Qwen/Qwen3.8-27B.
- **Quantization:** IQ3_S codebook mix, recovered scales.
- **Features claimed:** text, tools (qwen38). A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Against the BF16 parent: mean KLD 0.0450, margin-qualified top-1 97.80% on 500 held-out positions, inside the bar. Family: Qwen3.5-0.8B bf16 against Qwen's float32 reference (1.5.0 golden pass).
- **Context tested:** not recorded.
- **Still missing:** A CPU row on the file; pinned greedy output.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** [n/a](../MANUAL.md#support-matrix): The qwen35 Gated DeltaNet hybrid has no Metal path; a Metal build runs this file on the CPU.
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="qwen3.8-27b-mtp-q8_0"></a>
#### `Qwen3.8-27B-MTP-Q8_0.gguf`

The NextN/MTP predictor head for Qwen3.8-27B, loaded with --mtp-file beside a trunk: it only proposes tokens, so the text is the trunk's own greedy output; it is a speed feature, not a model.

- **File:** `Qwen3.8-27B-MTP-Q8_0.gguf` (sha256 `89e31f13400bac27b888f10bab02ff4d8da1f8610fa078dc32f9f3f0d2b6f6c9`, 3,164,006,784 bytes, from [Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF) at revision `7b10f923a7f7`).
- **Upstream checkpoint:** Qwen/Qwen3.8-27B.
- **Quantization:** Q8_0.
- **Runs with:** [Qwen3.8-27B](#qwen3.8-27b-gsq-rco-iq3_s-recovered), loaded beside it with `--mtp-file`; the smoke and its pin are of that composed run.
- **Features claimed:** mtp. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** The trunk's greedy output is the reference: the head proposes, the trunk's verify walk decides, so greedy text with and without the head is the same.
- **Context tested:** not recorded.
- **Still missing:** Pinned output of the composed run (the trunk plus this head).
- **Evidence:** [MANUAL.md#placement-and-memory](../MANUAL.md#placement-and-memory).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** [n/a](../MANUAL.md#support-matrix): MTP/NextN draft-head consumption is not implemented on Metal.
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Text path only: Gemma 4 12B

<a id="gemma-4-12b-it-qat-q4_0"></a>
#### `gemma-4-12b-it-qat-q4_0.gguf`

Text path only: Google's quantization-aware Q4_0 release, its mmproj file not loaded; the gemma4 tool protocol passes report cases A and B on CUDA, and case C (prose then call) fails on the family's own call-first format.

- **File:** `gemma-4-12b-it-qat-q4_0.gguf` (sha256 `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`, 6,975,879,296 bytes, from [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) at revision `29d097773436`).
- **Upstream checkpoint:** google/gemma-4-12B-it.
- **Quantization:** Q4_0 (Google QAT).
- **Features claimed:** text, tools (gemma4). A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: gemma-4-E2B bf16 against Google's float32 reference (1.5.0 golden pass). This file: 4/6 greedy against llama.cpp (2026-08-05), agreement with another engine.
- **Context tested:** not recorded.
- **Known limitations:** No vision: the mmproj file is not loaded.
- **Still missing:** Metal and CPU rows on the 12B; pinned output; Metal re-gate hardware.
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Text path only: Gemma 4 26B-A4B

<a id="gemma-4-26b-a4b-it-qat-q4_0"></a>
#### `gemma-4-26B_q4_0-it.gguf`

Text path only: Google's quantization-aware Q4_0 release of the MoE model, its mmproj file not loaded; no float32 reference has been run on the MoE layout.

- **File:** `gemma-4-26B_q4_0-it.gguf` (sha256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`, 14,439,363,584 bytes, from [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) at revision `d1c082be9cf3`).
- **Upstream checkpoint:** google/gemma-4-26B-A4B-it.
- **Quantization:** Q4_0 (Google QAT).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison is feasible and owed):** anchor: agreement with another engine, not fidelity. The smallest MoE member is this model (104 GB in float32, feasible on the lab CPU box and owed); llama.cpp agreement sits inside the documented routing-chaos KL floor.
- **Context tested:** not recorded.
- **Known limitations:** Metal last validated 2026-09-01 on a 128 GB loaner; no Metal device in the lab can re-gate it.
- **Still missing:** A float32 run or keeping the another-engine label; the tool gate on the file; pinned output; Metal re-gate hardware.
- **Evidence:** [docs/model-evidence.md](model-evidence.md), [tests/compatibility/models.json](../tests/compatibility/models.json).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Text path only: Gemma 4 31B

<a id="gemma-4-31b-it-q4_0"></a>
#### `gemma-4-31B-it-Q4_0.gguf`

Text path only: ggml-org's post-training Q4_0 of Gemma 4 31B, not Google's QAT file; the existing 31B evidence is bound to this file's sha256.

- **File:** `gemma-4-31B-it-Q4_0.gguf` (sha256 `031dc1c5fa9c5a0abbf3c39c5173fb2af65f5ac2dc2a090268561d3c72dcd834`, 17,992,313,088 bytes, from [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) at revision `3d47a7e680fb`).
- **Upstream checkpoint:** google/gemma-4-31B-it.
- **Quantization:** Q4_0 (ggml-org, post-training).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: gemma-4-E2B bf16 against Google's float32 reference (1.5.0 golden pass). This file: 5/6 greedy against llama.cpp, 64/64 token-identical against llama.cpp on Metal (2026-09-01), agreement with another engine; a float32 run on the 31B (124 GB) is feasible on the lab CPU box and owed.
- **Context tested:** not recorded.
- **Known limitations:** Metal last validated 2026-09-01 on a 128 GB loaner; no Metal device in the lab can re-gate it.
- **Still missing:** A float32 run or the another-engine label; the tool gate; pinned output; Metal re-gate hardware.
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [docs/model-evidence.md](model-evidence.md).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Text path only: Muse-Glimmer-30B

<a id="muse-glimmer-30b-kquant"></a>
#### `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`

Text path only: Meta's own Q4_K_M, which passes the fidelity bar against the BF16 parent; the vision encoder is outside the admitted path.

- **File:** `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (sha256 `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e`, 16,756,683,904 bytes, from [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) at revision `b44f1072cf75`).
- **Upstream checkpoint:** meta-models/Muse-Glimmer-30B.
- **Quantization:** Q4_K_M (Meta).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. The only member is 30B (120 GB in float32); the bar is measured against the BF16 parent (quant frontier report, 2026-08-20) and greedy against llama.cpp b10353.
- **Context tested:** not recorded.
- **Known limitations:** No vision encoder. Metal last validated 2026-08-11; no lab Metal device fits the file.
- **Still missing:** A live tool-protocol run on one file; backend rows; pinned output; Metal re-gate hardware.
- **Evidence:** [docs/model-evidence.md](model-evidence.md), [tests/compatibility/models.json](../tests/compatibility/models.json).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Granite-4.2-8B

<a id="granite-4.2-8b-q4_k_m"></a>
#### `granite-4.2-8b-Q4_K_M.gguf`

IBM's official Q4_K_M of Granite 4.2 8B, text path: the granite42 tool protocol and two client loops pass on an 8 GB CUDA card; case C is open.

- **File:** `granite-4.2-8b-Q4_K_M.gguf` (sha256 `16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311`, 5,347,917,952 bytes, from [ibm-granite/granite-4.2-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) at revision `93f3f6a8938e`).
- **Upstream checkpoint:** ibm-granite/granite-4.2-8b.
- **Quantization:** Q4_K_M (IBM).
- **Features claimed:** text, tools (granite42). A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: granite-4.2-3B bf16 against IBM's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass). This file: Q4_K_M vs Q8_0 bar PASS (400 positions, mean KLD 0.0036); tokenizer 0/721, template 22/22.
- **Context tested:** not recorded.
- **Still missing:** Metal and CPU rows; pinned output.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Per-release smoke:** on windows-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="granite-4.2-8b-q8_0"></a>
#### `granite-4.2-8b-Q8_0.gguf`

IBM's official Q8_0 of Granite 4.2 8B, text path: the reference sibling of the Q4_K_M row's quant bar.

- **File:** `granite-4.2-8b-Q8_0.gguf` (sha256 `fb66ad5750680c77c76b9dc095961375f14fc88ba15fd5fc084ef6e6701bad77`, 9,345,613,952 bytes, from [ibm-granite/granite-4.2-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) at revision `93f3f6a8938e`).
- **Upstream checkpoint:** ibm-granite/granite-4.2-8b.
- **Quantization:** Q8_0 (IBM).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: granite-4.2-3B bf16 against IBM's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass).
- **Context tested:** not recorded.
- **Still missing:** Backend rows on the file; pinned output.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Trinity Nano

<a id="trinity-nano-preview-q8_0"></a>
#### `Trinity-Nano-Preview-Q8_0.gguf`

Arcee's Q8_0 of Trinity Nano Preview, text path, CPU only by admission: the one afmoe file Runner actively validates.

- **File:** `Trinity-Nano-Preview-Q8_0.gguf` (sha256 `5fcc2428c325e8b2035d280ff28c4c659ea240ce317a9536f230acfc352b79d1`, 6,531,428,640 bytes, from [arcee-ai/Trinity-Nano-Preview-GGUF](https://huggingface.co/arcee-ai/Trinity-Nano-Preview-GGUF) at revision `2aa08593b792`).
- **Upstream checkpoint:** arcee-ai/Trinity-Nano-Preview.
- **Quantization:** Q8_0.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** This file against Arcee's float32 reference (1.5.0 golden pass, 2,000 positions): margin-qualified top-1 97.43%, mean KL 0.0626 (llama.cpp 97.80%, 0.0610); at bf16 both engines sit about 0.045 of mean KL from the publisher, a shared reproduction gap not yet explained.
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [docs/model-evidence.md](model-evidence.md).
- **Per-release smoke:** on linux-x86_64-cuda, cpu backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** [n/a](../MANUAL.md#support-matrix): afmoe (gated attention with sparse MoE) has no CUDA path; CUDA refuses it and the file runs on the CPU.
- **Metal:** [n/a](../MANUAL.md#support-matrix): afmoe (gated attention with sparse MoE) has no Metal path; Metal refuses it and the file runs on the CPU.
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Xyntetik-Kvist-14B

<a id="xyntetik-kvist-14b-bf16"></a>
#### `Xyntetik-Kvist-14B-BF16.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its BF16 file.

- **File:** `Xyntetik-Kvist-14B-BF16.gguf` (sha256 `daf422db9d0853d53ec8abb8fe9cd3f1e33ea53dffe7cca476225fbe1bc5d5c6`, 28,903,141,184 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** BF16.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cpu backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="xyntetik-kvist-14b-q8_0"></a>
#### `Xyntetik-Kvist-14B-Q8_0.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its Q8_0 file.

- **File:** `Xyntetik-Kvist-14B-Q8_0.gguf` (sha256 `cdb73543a6fc22f7faa5bbab0176ab2d5b1b590b46870d2d22f2a2510caaacb6`, 15,363,224,384 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** Q8_0.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="xyntetik-kvist-14b-q5_0-mix"></a>
#### `Xyntetik-Kvist-14B-Q5_0-mix.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its Q5_0 mix file.

- **File:** `Xyntetik-Kvist-14B-Q5_0-mix.gguf` (sha256 `37af54b4cef4445bbbf23e8fbf0ceec8dc9b01b38dab0dc7f9e2bd2fd79f3a56`, 10,295,023,424 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** Q5_0 mix.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="xyntetik-kvist-14b-iq4_nl-mix"></a>
#### `Xyntetik-Kvist-14B-IQ4_NL-mix.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its IQ4_NL mix file.

- **File:** `Xyntetik-Kvist-14B-IQ4_NL-mix.gguf` (sha256 `c74b27a5a39c894c4d1a1a2ae140de0cee125fc5cc9a7c96db9b7355bf8709a3`, 7,612,384,384 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** IQ4_NL mix.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Qwen3-30B-A3B selective precision

<a id="qwen3-30b-a3b-expq4_0-attnq8_0"></a>
#### `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf`

Qwen3-30B-A3B with attention Q8_0 and experts Q4_0, text path: it passes the fidelity bar against the first-party Q8_0 where the official uniform Q4_K_M fails it.

- **File:** `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf` (sha256 `df02efa815a74e6207f9e01f2d38272c1e2718bb23aec544709a02a2c7d5b206`, 17,988,417,024 bytes, from [Joakimpalm-Zen/Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF) at revision `6cdb710bcd55`).
- **Upstream checkpoint:** Qwen/Qwen3-30B-A3B.
- **Quantization:** Q8_0 attention / Q4_0 experts.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. A float32 reference of the 30B (120 GB) is not run; the bar is against the byte-verified first-party Q8_0 (v2 top-1 99.50%, mean KLD 0.0345).
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [tests/compatibility/models.json](../tests/compatibility/models.json), [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Qwen3-Coder-30B-A3B keep-120

<a id="qwen3-coder-30b-a3b-instruct-keep120-q4_k_m"></a>
#### `Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf`

An expert-pruned Qwen3-Coder-30B-A3B (120 of 128 experts kept) at Q4_K_M, text path: it passes the original and the current fidelity bar against its parent.

- **File:** `Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf` (sha256 `3ac8568224a15ea49db6affdd195ba48e8b2d7c6325c7849b9829381e89163a1`, 17,531,967,808 bytes, from [Joakimpalm-Zen/Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M-GGUF) at revision `3fd968c75417`).
- **Upstream checkpoint:** Qwen/Qwen3-Coder-30B-A3B-Instruct.
- **Quantization:** Q4_K_M, 120 experts.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. A float32 reference of the 30B is not run; the bars are against the unpruned parent.
- **Context tested:** not recorded.
- **Still missing:** Pinned output; a tool gate on this file (the recorded qwen3-coder tool run is on the unpruned Q4_K_M).
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Text path only: Gemma 4 E2B

<a id="gemma-4-e2b-it-q4_k_m-q4_0-mix"></a>
#### `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`

Text path only: the smoke-test file from the quickstart; it fails the fidelity bar against its own BF16 parent (77.75% of tokens agree, mean KLD 0.286) and is a way to try Runner on an 8 GB machine, not a faithful gemma-4-E2B.

- **File:** `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf` (sha256 `16ffaa2ea568e7a0c9b5c6ddbd39a376879f9ac007861334974de39f4cc52e3f`, 2,631,015,904 bytes, from [Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF](https://huggingface.co/Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF) at revision `37c8fa438245`).
- **Upstream checkpoint:** google/gemma-4-E2B-it.
- **Quantization:** Q4_K_M/Q4_0 mix.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: gemma-4-E2B bf16 against Google's float32 reference, where Runner is the closer engine (1.5.0 golden pass); that pass's served row is a different file (q4_0 throughout), not this mix.
- **Context tested:** not recorded.
- **Still missing:** nothing the audit named
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on macos-arm64-metal, metal backend, the policy's age limit.
- **CPU:** ✅ v1.5.1 2026-10-23 on macos-arm64-metal at commit `a72068140625`, output sha256 `eadb99d6bfee9881`, [pin](../tests/compatibility/tier-smoke/gemma-4-e2b-it-q4_k_m-q4_0-mix.macos-arm64-metal.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ✅ v1.5.1 2026-10-23 on macos-arm64-metal at commit `a72068140625`, output sha256 `eadb99d6bfee9881`, [pin](../tests/compatibility/tier-smoke/gemma-4-e2b-it-q4_k_m-q4_0-mix.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.5.1, 2026-10-23).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-23).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### NVIDIA Nemotron Nano 9B v2

<a id="nvidia-nemotron-nano-9b-v2-q8_0"></a>
#### `NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf`

A plain Q8_0 of NVIDIA's Nemotron-Nano-9B-v2 (Mamba-2 hybrid), text path: 100% margin-qualified top-1 against NVIDIA's float32 reference at Q8_0 over 100 positions; the 2,000-position rerun did not finish.

- **File:** `NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf` (sha256 `d3224286484df1ae79c2b74d2a0913e1bbb5a364b0dd7c1061c95fa97ee608d7`, 9,458,093,632 bytes, from [Joakimpalm-Zen/NVIDIA-Nemotron-Nano-9B-v2-Q8_0-GGUF](https://huggingface.co/Joakimpalm-Zen/NVIDIA-Nemotron-Nano-9B-v2-Q8_0-GGUF) at revision `3b6a1cf5035f`).
- **Upstream checkpoint:** nvidia/NVIDIA-Nemotron-Nano-9B-v2.
- **Quantization:** Q8_0.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** NVIDIA's float32 reference (golden pass 2026-09-07, 100 positions): margin-qualified top-1 100% at bf16 and Q8_0; Runner's mean KL 0.0041 at Q8_0 against llama.cpp's 0.0054. CPU and CUDA greedy byte-identical on this file.
- **Context tested:** not recorded.
- **Still missing:** The tool gate (native protocol since 2026-10-09); pinned output; the 2,000-position rerun.
- **Evidence:** [docs/golden-pass-2026-09-07.md](golden-pass-2026-09-07.md).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** [n/a](../MANUAL.md#support-matrix): Metal has no SSM path; a Metal build runs this file on the CPU.
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Text path only: Gemma 4 31B, attention block 48 removed

<a id="gemma-4-31b-it-attn48-removed-q4_0"></a>
#### `gemma-4-31B-it-attn48-removed-q4_0.gguf`

Text path only: Google's QAT Q4_0 of Gemma 4 31B with one attention sublayer physically removed; it tracks its parent at clean KLD 0.0223 on 44,413 held-out positions and loads only in Runner (llama.cpp refuses it by name).

- **File:** `gemma-4-31B-it-attn48-removed-q4_0.gguf` (sha256 `d7eaef9a8c05d8ad5cc63e4b37767113498b6ec6a31863519f5399f12cdb3134`, 17,576,681,600 bytes, from [Joakimpalm-Zen/gemma-4-31B-it-attn48-removed-Q4_0-GGUF](https://huggingface.co/Joakimpalm-Zen/gemma-4-31B-it-attn48-removed-Q4_0-GGUF) at revision `c704744772b4`).
- **Upstream checkpoint:** google/gemma-4-31B-it.
- **Quantization:** Q4_0 (Google QAT), one attention sublayer removed.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Against its parent, Google's QAT Q4_0: clean KLD 0.0223 on 44,413 held-out positions (2026-08-30, on the identical zeroed form). Family: gemma-4-E2B bf16 against Google's float32 reference.
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [docs/sublayer-removal.md](sublayer-removal.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Text path only: Muse-Glimmer-30B surgical

<a id="muse-glimmer-30b-surgical-4p75-q4_k"></a>
#### `Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf`

Text path only: Muse-Glimmer-30B with 4.75% of the decoder removed and healed, at Q4_K; it passes the bar against the unmodified BF16 parent on a held-out split.

- **File:** `Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf` (sha256 `103aa117dfce88974bd78e03ee6ca6601c720f01a1ec24ee1cfd0e5fe2f77ce6`, 15,686,266,304 bytes, from [Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q4_K-GGUF](https://huggingface.co/Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q4_K-GGUF) at revision `38f6284ec919`).
- **Upstream checkpoint:** meta-models/Muse-Glimmer-30B.
- **Quantization:** Q4_K.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. The family's only member is 30B (120 GB in float32). Against the unmodified BF16 parent: 97.74% / 0.04810 on a held-out split.
- **Context tested:** not recorded.
- **Still missing:** Backend rows; pinned output; a live tool-protocol run.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="muse-glimmer-30b-surgical-6p34-q8_0"></a>
#### `Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf`

Text path only: Muse-Glimmer-30B with 6.34% of the decoder removed (the FFN sublayers of layers 4, 7, 9 and 48), at Q8_0; it passes the bar against the unmodified BF16 parent on a held-out split.

- **File:** `Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf` (sha256 `d7bb5c6f3b7170160322c684f338c0db630ea24ab09b896a6c97c8c415f58d4d`, 29,612,961,216 bytes, from [Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q8_0-GGUF](https://huggingface.co/Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q8_0-GGUF) at revision `e95fcd1a298c`).
- **Upstream checkpoint:** meta-models/Muse-Glimmer-30B.
- **Quantization:** Q8_0.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. The family's only member is 30B (120 GB in float32). Against the unmodified BF16 parent: 98.34% / 0.03883 on a held-out split.
- **Context tested:** not recorded.
- **Still missing:** Backend rows; pinned output; a live tool-protocol run.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cpu backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="muse-glimmer-30b-surgical-6p34-q4_k-recovered"></a>
#### `Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf`

Text path only: the 6.34% surgical Muse-Glimmer-30B at Q4_K with retrained scales; it passes the bar (0.04949, 97.90%) where plain Q4_K of the same model fails it (0.05815).

- **File:** `Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf` (sha256 `12a429483b18aefcfd9cdc41afb156f339d610be0fdd66fdaf5391e9df1cbb19`, 15,686,266,304 bytes, from [Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-6p34-Q4_K-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-6p34-Q4_K-recovered-GGUF) at revision `164d823e32a3`).
- **Upstream checkpoint:** meta-models/Muse-Glimmer-30B.
- **Quantization:** Q4_K, recovered scales.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. The family's only member is 30B (120 GB in float32). Against the unmodified BF16 parent: mean KLD 0.04949, margin-qualified top-1 97.90%.
- **Context tested:** not recorded.
- **Still missing:** Backend rows; pinned output; a live tool-protocol run.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Qwen3-4B Runner ToolUse

<a id="qwen3-4b-runner-tooluse-merged-q8_0"></a>
#### `merged-Q8_0.gguf`

Qwen3-4B with the tool-use LoRA merged and written at Q8_0, text path: a qwen3 client loop passes on CUDA; it is an adaptation demonstration, not a general improvement over Qwen3-4B.

- **File:** `merged-Q8_0.gguf` (sha256 `b9d21e525d91b20a1fc3437cf34c7ecfa03afda722d960b95bcfd113f2b7ea02`, 4,280,405,440 bytes, from [Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M) at revision `5902b309ee51`).
- **Upstream checkpoint:** Qwen/Qwen3-4B.
- **Quantization:** Q8_0.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: Qwen3-4B bf16 against Qwen's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass; its served row is Qwen3-4B Q4_K_M, not this file).
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [docs/adaptation-engine.md](adaptation-engine.md).
- **Per-release smoke:** on linux-x86_64-cuda, cuda backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="qwen3-4b-runner-tooluse-lora"></a>
#### `Qwen3-4B-ToolUse-LoRA.gguf`

The tool-use LoRA adapter for Qwen3-4B, served with --lora on the base its card names (bartowski's Q4_K_M): an adapter, not a model.

- **File:** `Qwen3-4B-ToolUse-LoRA.gguf` (sha256 `ea38f80c33d381c9aa62c874920d2fa058dd307e7186e9b7c347d782f35b88e4`, 66,094,752 bytes, from [Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M) at revision `5902b309ee51`).
- **Upstream checkpoint:** Qwen/Qwen3-4B.
- **Quantization:** adapter (F32/F16 LoRA tensors).
- **Runs with:** [Qwen3-4B](#qwen3-4b-q4_k_m-bartowski), loaded beside it with `--lora`; the smoke and its pin are of that composed run.
- **Features claimed:** adapters. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** The base model's anchor: Qwen3-4B bf16 against Qwen's float32 reference (1.5.0 golden pass). The adapter itself has no float32 reference.
- **Context tested:** not recorded.
- **Still missing:** Pinned output of the composed run (the base named on the card plus this adapter).
- **Evidence:** [docs/adaptation-engine.md](adaptation-engine.md), [docs/lora.md](lora.md).
- **Per-release smoke:** on linux-x86_64-cuda, cpu backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Experimental: Qwen3.8-Flash-Next

<a id="qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head"></a>
#### `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf` (4 parts)

Experimental: the qwen4exp architecture is admitted with a recorded publisher-reference gap, and no gate has run on this 4-part file; the admission evidence is on the upstream 2-part trunk.

- **File:** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf` (sha256 `522632bbb6f2c353621762be5bd48b270b1ff9fe60a9630bd47d5ef097aa0fa2`, 26,481,462,400 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00002-of-00004.gguf` (sha256 `f82cc8f151e15b73bbb84c75baf78f179df1b9b368437d7c75d8b3a7618cb8c8`, 28,347,013,600 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00003-of-00004.gguf` (sha256 `53ada4d2c3f2e189ce209520ecad0ae9649028b19e6bb9a55a021613b204e93b`, 28,800,138,432 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00004-of-00004.gguf` (sha256 `3a8c77474c82c5e81212ef3c5ef2330e75413cc60241d600fcfd87e4e2143b66`, 2,803,798,784 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`).
- **Upstream checkpoint:** Qwen/Qwen3.8-Flash-Next.
- **Quantization:** IQ3_S codebook mix with the MTP head (4 parts).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison is feasible and owed):** CPU forward anchored on llama.cpp on a toy fixture and on the upstream 2-part trunk (greedy 12 tokens identical to llama.cpp b11433), agreement with another engine; the publisher's float32 comparison is owed.
- **Context tested:** not recorded.
- **Still missing:** Every gate on this file; the publisher float32 comparison.
- **Evidence:** [docs/qwen4exp-admission-evidence/README.md](qwen4exp-admission-evidence/README.md).
- **Per-release smoke:** on linux-x86_64-cuda, cpu backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** [n/a](../tests/compatibility/coverage.json): qwen4exp has no Metal path (no qwen35-family path on Metal).
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="qwen3.8-flash-next-mtp-q8_0"></a>
#### `Qwen3.8-Flash-Next-MTP-Q8_0.gguf`

Experimental: the NextN/MTP head for Qwen3.8-Flash-Next as a companion file for --mtp-file, built from the publisher's BF16 checkpoint for ISTA-DASLab's IQ3_S trunk; no gate has run on it.

- **File:** `Qwen3.8-Flash-Next-MTP-Q8_0.gguf` (sha256 `cd2f837a63f0b6ea1d951976e161fc40648932ffaf59194aed3557175638000b`, 4,165,605,056 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`).
- **Upstream checkpoint:** Qwen/Qwen3.8-Flash-Next.
- **Quantization:** Q8_0.
- **Runs with:** [Qwen3.8-Flash-Next](#qwen3.8-flash-next-gsq-rco-iq3_s), loaded beside it with `--mtp-file`; the smoke and its pin are of that composed run.
- **Features claimed:** mtp. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison is feasible and owed):** The trunk's greedy output is the reference: the head proposes and the trunk decides. The trunk itself is experimental (see the trunk row).
- **Context tested:** not recorded.
- **Still missing:** Every gate, run beside ISTA-DASLab's 2-part trunk (the 4-part set already carries a head, and --mtp-file refuses a second one).
- **Evidence:** [MANUAL.md#placement-and-memory](../MANUAL.md#placement-and-memory).
- **Per-release smoke:** on linux-x86_64-cuda, cpu backend, the policy's age limit.
- **CPU:** ☐ not run
- **CUDA:** ☐ not run
- **Metal:** [n/a](../MANUAL.md#support-matrix): MTP/NextN draft-head consumption is not implemented on Metal.
- **LoRA:** ☐ not run (no gate has recorded the file's `--adapt-info` answer).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="tier2"></a>
## Tier 2, maintained on report

### gpt-oss-20b keep-30

<a id="gpt-oss-20b-keep30-mxfp4"></a>
#### `gpt-oss-20b-keep30-MXFP4.gguf`

A 32-to-30-expert derivative of gpt-oss-20b that does not pass the current fidelity bar; its card leads with the measured status and it stays published as a disclosed near-miss.

- **File:** `gpt-oss-20b-keep30-MXFP4.gguf` (sha256 `e6b884b3de28a78858097bbee25c9c503975992a2bbc8943e0ef2c4bbbc14312`, 11,472,833,184 bytes, from [Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF](https://huggingface.co/Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF) at revision `d6bfb80ce279`).
- **Published below Tier 1:** Owner decision 2026-10-22: a published file may be Tier 2 when its card already leads with its measured shortfall.
- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Qwen3-4B

<a id="qwen3-4b-q4_k_m-bartowski"></a>
#### `Qwen_Qwen3-4B-Q4_K_M.gguf`

The base the published tool-use adapter was trained through and runs on, named by sha256 on the adapter's card: maintained on report, and run on every release only as the adapter's base in its composed smoke.

- **File:** `Qwen_Qwen3-4B-Q4_K_M.gguf` (sha256 `fbe1d5edd4ce802ae3ae7c7e4ab7d09789d697fdac1fc7929f8df4ca3c41bae3`, 2,497,280,960 bytes, from [bartowski/Qwen_Qwen3-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF) at revision `cb76885dc66d`).
- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Experimental: Qwen3.8-Flash-Next

<a id="qwen3.8-flash-next-gsq-rco-iq3_s"></a>
#### `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (2 parts)

Experimental: ISTA-DASLab's IQ3_S trunk of Qwen3.8-Flash-Next, the file the published MTP head was built for and the one the qwen4exp admission evidence measured; maintained on report, and run on every release only as that head's trunk in its composed smoke.

- **File:** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (sha256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`, 54,817,524,224 bytes, from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at revision `ed59f92082b1`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`, 28,800,138,432 bytes, from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at revision `ed59f92082b1`).
- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Llama-style dense families (llama)

<a id="arch-llama"></a>
#### Llama-style dense families (llama)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Mistral (mistral)

<a id="arch-mistral"></a>
#### Mistral (mistral)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### SmolLM (smollm)

<a id="arch-smollm"></a>
#### SmolLM (smollm)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### StableLM (stablelm)

<a id="arch-stablelm"></a>
#### StableLM (stablelm)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Qwen2 (qwen2)

<a id="arch-qwen2"></a>
#### Qwen2 (qwen2)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Gemma 3 (gemma3)

<a id="arch-gemma3"></a>
#### Gemma 3 (gemma3)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Phi-3 (phi3)

<a id="arch-phi3"></a>
#### Phi-3 (phi3)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Apertus (apertus)

<a id="arch-apertus"></a>
#### Apertus (apertus)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Granite 4 h-series hybrid (granitehybrid)

<a id="arch-granitehybrid"></a>
#### Granite 4 h-series hybrid (granitehybrid)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

<a id="tier3"></a>
## Tier 3, unmaintained

No members.

<a id="upcoming"></a>
## Upcoming, planned admission

### Qwen3.8-35B-A3B

<a id="qwen3.8-35b-a3b"></a>
#### Qwen3.8-35B-A3B

Not admitted: Runner does not load the qwen35moe architecture yet.

- **Numerical anchor:** anchor: agreement with another engine, not fidelity. A float32 reference (140 GB) is not feasible; Qwen3.5-0.8B covers the recurrent path only.
- **Missing for Tier 1:** The qwen35moe admission (Gated DeltaNet hybrid plus routed experts; the qwen35 recurrent path and the qwen3moe expert path both exist, the hybrid does not). One piece of engine work covers Qwen3.8-35B-A3B, Ornith-1.5-35B-A3B and MiniCPM-V-4.7-35B-A3B; the first Upcoming item. Then tokenizer, template and quant-bar gates, a pinned output and a device record.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Text path only: Ornith-1.5-35B-A3B

<a id="ornith-1.5-35b-a3b"></a>
#### Ornith-1.5-35B-A3B

Text path only: the vision projector is not loaded, and Runner does not load the qwen35moe architecture yet.

- **Numerical anchor:** anchor: agreement with another engine, not fidelity.
- **Missing for Tier 1:** The qwen35moe admission (see Qwen3.8-35B-A3B), then template conformance for Ornith 1.5 (the ornith template exists for 1.0 9B dense).
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Text path only: MiniCPM-V-4.7-35B-A3B

<a id="minicpm-v-4.7-35b-a3b"></a>
#### MiniCPM-V-4.7-35B-A3B

Text path only: the vision projector is not loaded, and Runner does not load the qwen35moe architecture yet.

- **Numerical anchor:** anchor: agreement with another engine, not fidelity.
- **Missing for Tier 1:** The qwen35moe admission (see Qwen3.8-35B-A3B); its tokenizer and template are unknown.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Granite-4.2-30B

<a id="granite-4.2-30b"></a>
#### Granite-4.2-30B

Not validated: the granite architecture is admitted and IBM's official GGUF exists, but no gate has run on a 30B file.

- **Numerical anchor:** Family: granite-4.2-3B bf16 against IBM's float32 reference covers the family.
- **Missing for Tier 1:** One gate run on the Linux CUDA lab box: quant bar, tool gate, pinned output, device record; a file pinned by sha256.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Nemotron 3.5 Lightning-30B-A3B

<a id="nemotron-3.5-lightning-30b-a3b-q4_0"></a>
#### `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`

Not validated: the file loads and its template conforms, but the tool gate, a pinned output and a device record are missing.

- **File:** `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (sha256 `61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c`, 18,898,091,584 bytes, from [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) at revision `8cd082170ed3`).
- **Numerical anchor:** anchor: agreement with another engine, not fidelity. The family's smallest member is 30B; 4/5 greedy against llama.cpp.
- **Missing for Tier 1:** Tool gate, pinned output, device record, and the another-engine label.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Qwen3.6-27B

<a id="qwen3.6-27b"></a>
#### Qwen3.6-27B

Not validated: the qwen35 family loads, but no gate has run on a Qwen3.6-27B file.

- **Numerical anchor:** Family: Qwen3.5-0.8B bf16 against Qwen's float32 reference covers the family.
- **Missing for Tier 1:** One gate run: tokenizer, template, quant bar, pinned output, device record; a file pinned by sha256.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Cloudflare Clef Flash (clef, on Qwen3.5-9B)

<a id="cloudflare-clef-flash"></a>
#### Cloudflare Clef Flash (clef, on Qwen3.5-9B)

Not validated: the clef architecture is admitted, but no clef-flash file is pinned or gated.

- **Missing for Tier 1:** Pin one clef-flash file by sha256 with its own evidence (the audit names clef and clef-flash as separate checkpoints); a float32 gold run on a released checkpoint.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Nemotron 3 Nano (30B-A3B)

<a id="nemotron-3-nano-30b-a3b"></a>
#### Nemotron 3 Nano (30B-A3B)

Not admitted as a file: never loaded; it declares the same architecture id as Lightning, with the layout unverified.

- **Numerical anchor:** Nano-9B-v2 bf16 covers the SSM path; the MoE path has no small member.
- **Missing for Tier 1:** A load test and a layout check, then the Lightning gate list.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Text path only: Nemotron 3.5 Bolt-74B-A9B

<a id="nemotron-3.5-bolt-74b-a9b"></a>
#### Nemotron 3.5 Bolt-74B-A9B

Text path only: Bolt ships a vision projector that is not loaded, and the model has never been loaded.

- **Numerical anchor:** anchor: agreement with another engine, not fidelity.
- **Missing for Tier 1:** A sharded load test (22 GB is tight on a 24 GB slice), then the Lightning gate list.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Text path only: Nemotron 3.5 Super VL-120B-A12B

<a id="nemotron-3.5-super-vl-120b-a12b"></a>
#### Nemotron 3.5 Super VL-120B-A12B

Text path only: image input is not implemented in Runner and a VL name never implies vision support; the model has never been loaded.

- **Numerical anchor:** anchor: agreement with another engine, not fidelity.
- **Missing for Tier 1:** A load test on the lab CPU box (Q4_K_M about 65 GB in two parts), then the Lightning gate list.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Kolibri-1 (Aleph Alpha, 78B)

<a id="kolibri-1"></a>
#### Kolibri-1 (Aleph Alpha, 78B)

Not admitted: a new architecture.

- **Numerical anchor:** anchor: agreement with another engine, not fidelity. A float32 reference is not feasible.
- **Missing for Tier 1:** Read the model code first, then an admission; 78B at Q4 is about 45 GB, the lab CPU box only.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

### Mellum2.1-12B-A2.5B-Thinking (JetBrains)

<a id="mellum2.1-12b-a2.5b-thinking"></a>
#### Mellum2.1-12B-A2.5B-Thinking (JetBrains)

Not admitted: a new architecture.

- **Numerical anchor:** A float32 reference is feasible.
- **Missing for Tier 1:** An admission; it fits every box and a float32 reference (48 GB) is feasible on the lab box, so it is a full admission candidate.
- **Tier history:** Upcoming, planned admission from 2026-10-22 ([log](#transition-log)).

<a id="exempt"></a>
## Published, not a Runner model file

- [Joakimpalm-Zen/Xyntetik-Kvist-14B](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B): Training weights in safetensors, not a Runner model file; the GGUF files are rows.
- [Joakimpalm-Zen/Xyntetik-Kvist-14B-research-checkpoints](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-research-checkpoints): Weights to resume the training from, not a Runner model file.
- [Joakimpalm-Zen/Xyntetik-Blad-Genesis-0.7M](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Blad-Genesis-0.7M): A toy-scale research artifact with no GGUF, not a Runner model (the manual says so).
- [Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M) (`study/`): Research adapters from the training study, not served artifacts.
- [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) (`evidence/`): The importance matrix behind the IQ4_NL mix, evidence and not a model file.

<a id="transition-log"></a>
## Transition log

Every tier a row has held, oldest first. Last validated release is the Runner version of the newest passing gate the roster had recorded on the row when it moved; none recorded means the roster held no gate for it then, which is every row at the 2026-10-22 migration.

| Effective | Row | Tier | Reason | Last validated release |
|---|---|---|---|---|
| 2026-10-22 | [Cloudflare Clef (clef, on Qwen3.8-27B), `Cloudflare_clef-IQ3_M.gguf`](#cloudflare-clef-iq3_m) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Qwen3.8-27B, `Qwen3.8-27B-UD-Q4_K_M.gguf`](#qwen3.8-27b-ud-q4_k_m) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Gemma 4 12B, `gemma-4-12b-it-qat-q4_0.gguf`](#gemma-4-12b-it-qat-q4_0) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Gemma 4 26B-A4B, `gemma-4-26B_q4_0-it.gguf`](#gemma-4-26b-a4b-it-qat-q4_0) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Gemma 4 31B, `gemma-4-31B-it-Q4_0.gguf`](#gemma-4-31b-it-q4_0) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Muse-Glimmer-30B, `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`](#muse-glimmer-30b-kquant) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Granite-4.2-8B, `granite-4.2-8b-Q4_K_M.gguf`](#granite-4.2-8b-q4_k_m) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Granite-4.2-8B, `granite-4.2-8b-Q8_0.gguf`](#granite-4.2-8b-q8_0) | Tier 1, actively validated | Migration: on the owner's Tier 1 roster (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Trinity Nano, `Trinity-Nano-Preview-Q8_0.gguf`](#trinity-nano-preview-q8_0) | Tier 1, actively validated | Migration: owner decision 2026-10-22, Tier 1 as the pinned afmoe file (the manual's 8 GB recommendation and a 1.5.0 golden-pass row). | none recorded |
| 2026-10-22 | [Qwen3.8-27B, `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf`](#qwen3.8-27b-gsq-rco-iq3_s-recovered) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Qwen3.8-27B, `Qwen3.8-27B-MTP-Q8_0.gguf`](#qwen3.8-27b-mtp-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-BF16.gguf`](#xyntetik-kvist-14b-bf16) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-Q8_0.gguf`](#xyntetik-kvist-14b-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-Q5_0-mix.gguf`](#xyntetik-kvist-14b-q5_0-mix) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-IQ4_NL-mix.gguf`](#xyntetik-kvist-14b-iq4_nl-mix) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Qwen3-30B-A3B selective precision, `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf`](#qwen3-30b-a3b-expq4_0-attnq8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Qwen3-Coder-30B-A3B keep-120, `Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf`](#qwen3-coder-30b-a3b-instruct-keep120-q4_k_m) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Gemma 4 E2B, `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`](#gemma-4-e2b-it-q4_k_m-q4_0-mix) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [NVIDIA Nemotron Nano 9B v2, `NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf`](#nvidia-nemotron-nano-9b-v2-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Gemma 4 31B, attention block 48 removed, `gemma-4-31B-it-attn48-removed-q4_0.gguf`](#gemma-4-31b-it-attn48-removed-q4_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf`](#muse-glimmer-30b-surgical-4p75-q4_k) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf`](#muse-glimmer-30b-surgical-6p34-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf`](#muse-glimmer-30b-surgical-6p34-q4_k-recovered) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Qwen3-4B Runner ToolUse, `merged-Q8_0.gguf`](#qwen3-4b-runner-tooluse-merged-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Qwen3-4B Runner ToolUse, `Qwen3-4B-ToolUse-LoRA.gguf`](#qwen3-4b-runner-tooluse-lora) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by owner decision 1. | none recorded |
| 2026-10-22 | [Qwen3.8-Flash-Next, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf`](#qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head) | Tier 1, actively validated | Migration: published file, Tier 1 by owner decision 1; owner decision 2026-10-22: stays Tier 1 with 'experimental' first. | none recorded |
| 2026-10-22 | [Qwen3.8-Flash-Next, `Qwen3.8-Flash-Next-MTP-Q8_0.gguf`](#qwen3.8-flash-next-mtp-q8_0) | Tier 1, actively validated | Migration: published file, Tier 1 by owner decision 1; owner decision 2026-10-22: stays Tier 1 with 'experimental' first. | none recorded |
| 2026-10-22 | [gpt-oss-20b keep-30, `gpt-oss-20b-keep30-MXFP4.gguf`](#gpt-oss-20b-keep30-mxfp4) | Tier 2, maintained on report | Migration: owner decision 2026-10-22, Tier 2 with its near-miss disclosure; it stays published. | none recorded |
| 2026-10-22 | [Qwen3-4B, `Qwen_Qwen3-4B-Q4_K_M.gguf`](#qwen3-4b-q4_k_m-bartowski) | Tier 2, maintained on report | Migration: the base of a Tier 1 adapter, added 2026-10-23 so the adapter's smoke can run; not on the owner's Tier 1 roster. | none recorded |
| 2026-10-22 | [Qwen3.8-Flash-Next, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`](#qwen3.8-flash-next-gsq-rco-iq3_s) | Tier 2, maintained on report | Migration: the trunk of a Tier 1 MTP head, added 2026-10-23 so the head's smoke can run; not on the owner's Tier 1 roster. | none recorded |
| 2026-10-22 | [Llama-style dense families (llama)](#arch-llama) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [Mistral (mistral)](#arch-mistral) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [SmolLM (smollm)](#arch-smollm) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [StableLM (stablelm)](#arch-stablelm) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [Qwen2 (qwen2)](#arch-qwen2) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [Gemma 3 (gemma3)](#arch-gemma3) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [Phi-3 (phi3)](#arch-phi3) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [Apertus (apertus)](#arch-apertus) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [Granite 4 h-series hybrid (granitehybrid)](#arch-granitehybrid) | Tier 2, maintained on report | Migration: an admitted architecture with no Tier 1 file (audit 2026-10-22, section C). | none recorded |
| 2026-10-22 | [Qwen3.8-35B-A3B](#qwen3.8-35b-a3b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Ornith-1.5-35B-A3B](#ornith-1.5-35b-a3b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [MiniCPM-V-4.7-35B-A3B](#minicpm-v-4.7-35b-a3b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Granite-4.2-30B](#granite-4.2-30b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Nemotron 3.5 Lightning-30B-A3B, `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`](#nemotron-3.5-lightning-30b-a3b-q4_0) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Qwen3.6-27B](#qwen3.6-27b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Cloudflare Clef Flash (clef, on Qwen3.5-9B)](#cloudflare-clef-flash) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Nemotron 3 Nano (30B-A3B)](#nemotron-3-nano-30b-a3b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Nemotron 3.5 Bolt-74B-A9B](#nemotron-3.5-bolt-74b-a9b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Nemotron 3.5 Super VL-120B-A12B](#nemotron-3.5-super-vl-120b-a12b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Kolibri-1 (Aleph Alpha, 78B)](#kolibri-1) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Mellum2.1-12B-A2.5B-Thinking (JetBrains)](#mellum2.1-12b-a2.5b-thinking) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |

<!-- model-tiers:rows:end -->
