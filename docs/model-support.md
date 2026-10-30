<<<<<<< HEAD
=======
# Model support tiers

A tier is a maintenance and validation commitment: it says which model files
Runner validates, when each one last passed a validation run, and what happens
when one of them breaks. It is not a compatibility list and not a correctness
verdict. What loads is the
[support matrix](../MANUAL.md#support-matrix), `runner --info caps` and
`runner --info doctor`; what was measured is in the
[model evidence](model-evidence.md) and the admission documents each row
links. Every row on this page is rendered from
[`docs/model-tiers.json`](model-tiers.json), and every mark carries the Runner
version and date of the run that wrote it. Nothing is re-tested on a schedule
(policy decision 2026-10-28): a mark stands until the next validation run or
an immediate need re-runs it, and the page is the paper trail of when each
file was last validated, with which Runner.

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
| Tier 1, actively validated | Validated in the bigger validation runs and on immediate need, with proactive maintenance for the configurations the row names. Correctness claims are tied to recorded evidence with a version and date. A row with no recorded pass, or a recorded failure, refuses a release. |
| Tier 2, maintained on report | Previously admitted; no scheduled full-model revalidation. Existing regression tests keep running, and a change to a shared execution path (kernel, architecture code, tokenizer, template, scheduler) is tested with representative coverage that includes the affected families. A reproducible report is investigated, the fix is validated on the affected configuration where feasible, and it leaves a regression test behind. |
| Tier 3, unmaintained | No dedicated validation or model-specific development. The implementation and the dated evidence remain; current-build correctness is not claimed. The 1.x loading promise still applies. No members: the first entry needs a policy decision and a transition-log entry. |
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
backend of its validation run with the version and date of the last one, a
mark per backend, the file's LoRA compatibility, and its tier history. A head or an adapter also names the row
it runs with.

The LoRA compatibility is the file's own `runner --info adapter` answer,
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
   compared against (the validation smoke below).
4. A recorded run on a real device class, written by the machine.

External hardware and community contributions count only as reproducible
records with a realistic path to revalidation.

## The validation run, and the release refusal

**The validation smoke, every Tier 1 file and any Tier 2 file a run
includes**, on the device class its row names. `scripts/tier-smoke.py` hashes the file against its row, records the
file's `--info adapter` answer, serves the file and asks it one user turn
through its own chat template, as a client would (`POST /v1/chat/completions`:
list the eight planets of the Solar System in order, temperature 0, 48
tokens, not streamed, `enable_thinking` false so a thinking family answers
within the budget; `-c 512 -b 64`), and compares the reply's content, as UTF-8
bytes, with the pin under
[`tests/compatibility/tier-smoke/`](../tests/compatibility/tier-smoke.json),
one pin per device class and backend. The reply must name "Mercury", a fact
from outside Runner, and hold at least 96 bytes, or it is never pinned, so
the comparison has something to catch. The anchors are the file's sha256 and
that fact; the pin itself only says whether this build reproduces the one
that wrote it. The one-shot `-p` is not what the smoke measures: it is a raw
prompt with no chat template, which an instruct model continues instead of
answering.

**When it runs.** Not on a schedule and not by age. The smoke, and the full
evidence bar above where a row's path changed, run in the bigger validation
runs the project starts (a release campaign, a kernel or architecture
rewrite) and on immediate need (a reported regression on a path the file
uses). A pull request does not trigger a re-run by itself. Each run writes
its Runner version and date into the row, which is the paper trail: the
[overview](#overview) shows for every file when it was last validated and
with which Runner, and an old mark stays dated rather than being re-run.

**A new admission** records the pass that admitted it: the row joins Tier 1,
or Tier 2 when the pull bar is not met, with the version, date and device
class of its validation run written by `--record`, never typed.

`scripts/check-release.py` refuses a tag while any Tier 1 row has no recorded
pass on the backend its smoke names, or any backend of the row carries a
recorded failure. A record's age never refuses. A row whose smoke class is
borrowed hardware (ownership in `docs/device-evidence.json`: lent with no
terms, usable while it is there, never depended on) is reported the same
way, but a missing record on it does not refuse the tag; a recorded failure
does on any class. A head or an adapter runs with the row it names, so every
Tier 1 row can be gated; `scripts/model-tiers.py --check` refuses one that
cannot.

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

The device class comes from the binary's own `--info caps`, matched against
[`docs/device-evidence.json`](device-evidence.json). The backend defaults to
the row's; `--backend cpu` adds a CPU cell on a GPU host. Run that cell with
the GPU visible, since the smoke serves it with `--gpu off` itself: a GPU
hidden with an empty `CUDA_VISIBLE_DEVICES` leaves `--info caps` with no GPU, which
matches no GPU class. When no class matches, the refusal lists every class
with the caps fields it expected otherwise; `--device-class` names the class
outright. `--record` writes the
cell and the `--info adapter` answer into the roster and re-renders this page:
commit `docs/model-tiers.json` and this file together.

The first run on a row and backend has no pin: run it once with `--pin`
instead of `--record`, commit the pin, then run with `--record` from the clean
tree that holds it. Each run starts `runner --serve` on a free local port,
sends the one request and stops the server. A `--pin` run whose reply does
not name the fact, or is shorter than the minimum, refuses and prints the
reply with how the turn ended; give that row its own `prompt`, `n_predict` or
`request` in [`tests/compatibility/tier-smoke.json`](../tests/compatibility/tier-smoke.json)
rather than lowering the minimum. A pin is the reply's bytes as sent and the
manifest is written with LF, so a Windows host writes the same files as any
other. When a release changes generated tokens on purpose (the
[versioning policy](versioning.md) allows it), the pin mismatches. Read the
difference; then `scripts/tier-smoke.py --retire --row ID --backend B
--device-class C` removes the pin, its manifest entry and the cell that cites
it, and the row is pinned again, with the reason in the commit. A pin is never
overwritten in place.

The checks: `scripts/model-tiers.py --check` (the roster's structure),
`--gate` (the release gate's question; `--today` fixes the day the printed
ages count from) and `--docs-check` (this page against the roster).

<a id="cards-and-site"></a>
## The overview, the README, the cards and the site

This page opens with the validation runs at a glance: one collapsed block
per family in the roster's order, Tier 1 families first and Tier 2 families
under their own line, each family's models sorted by name, then the Upcoming
models as one line; the tiers below group every row the same way. The table
is the record of when each file was last validated and with which Runner. A
family's line sums up every file of its models, per backend and for Train:
✅ when every file is recorded, 🟨 *k of n* when k of its n files are (a
declared limit beside recorded files counts as not run), a hyphen when none
is, n/a when every file has a declared limit there, and failed when any
file recorded a failure. For Train, recorded means the LoRA backward admits
the file, and a refusal counts as a declared limit. Inside, each model's
line names it with its label (text path only, experimental), and its table
gives one row per file with its CPU, CUDA, Metal and Train cells: a
recorded gate as its version and date, `failed` with its version and date
for a recorded failure, `n/a` linked to the manual's support matrix, and a
blank cell where nobody has run the gate yet. The Train cell is the
recorded `--info adapter` answer, ✅ with its version and date or `no:` and
the refusal. The file name links to its row further down this page.

This page is the only render of the marks (2026-10-27: one rendered table,
moved here from the README the same day). Three more places link to it
instead of carrying a copy, so none can drift and nothing watches them:

- **The README.** Its "Actively validated models" section says what the
  release check is and links to the overview here.
- **The Hugging Face cards.** Every repository that holds a published row's
  file carries a `xyntetik-validated` block on its card, the last bullets
  of the card's own `xyntetik-status` block: one sentence, then one line
  per roster file of the repository naming it and linking to its row on
  this page (the row's id is the anchor). The block carries no marks or
  dates. A new published file's card gets the same block by hand, copied
  from any other card with the file name and anchor changed; nothing
  renders or pushes it.
- **xyntetik.com.** The Runner page's "Actively validated models" section
  links to the overview here and to the README's section; the landing
  page's model cards link to their Hub repositories, whose cards link here.

<!-- model-tiers:rows:begin (rendered from docs/model-tiers.json by scripts/model-tiers.py --write-docs; do not edit) -->

<a id="overview"></a>
## Validation runs at a glance

✅ *version date*: recorded gate · blank: nobody has run it yet · n/a: declared limit, linked to the manual · failed *version date*: recorded failure · Train: the file's own `--info adapter` answer, where ✅ means the LoRA backward admits the file, not that training it gives good results. A family's line sums up every file of its models, per backend and for Train: ✅ all recorded · 🟨 *k of n*: k of its n files recorded, a declared limit beside them counted as not run · a hyphen: none run · n/a: a declared limit on every file · failed: a recorded failure on any file. For Train, recorded means the LoRA backward admits the file, and a refusal counts as a declared limit. A mark with <sup>b</sup> was recorded on borrowed hardware: dated evidence the project may use and does not depend on. Each file name links to its envelope below.

**Tier 1, actively validated** (the release gate holds Runner to these):

<details>
<summary><b>Gemma 4</b>, 1 model: CPU ✅ · CUDA ✅ · Metal ✅ · Train ✅</summary>
<p><b>Nemotron 3.5 Bolt E4B</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#nemotron-3.5-bolt-e4b-q4_k_m"><code>Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.0<br>2026-10-29</td><td nowrap>✅ 1.7.0<br>2026-10-28</td><td nowrap>✅ 1.7.0<br>2026-10-29</td><td nowrap>✅ 1.7.0<br>2026-10-29</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Qwen3.8</b>, 2 models: CPU 🟨 1 of 4 · CUDA 🟨 3 of 4 · Metal n/a · Train 🟨 2 of 4</summary>
<p><b>Qwen3.8-27B</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3.8-27b-gsq-rco-iq3_s-recovered"><code>Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf</code></a></td><td></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
<tr><td><a href="#qwen3.8-27b-mtp-q8_0"><code>Qwen3.8-27B-MTP-Q8_0.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
<p><b>Qwen3.8-Flash-Next</b>, experimental</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head"><code>Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf</code></a> (4 parts)</td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td>no: hyper-connections (qwen4exp)</td></tr>
<tr><td><a href="#qwen3.8-flash-next-mtp-q8_0"><code>Qwen3.8-Flash-Next-MTP-Q8_0.gguf</code></a></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td>no: hyper-connections (qwen4exp)</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Xyntetik-Kvist-14B</b>, 1 model: CPU 🟨 2 of 4 · CUDA 🟨 3 of 4 · Metal - · Train ✅</summary>
<p><b>Xyntetik-Kvist-14B</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#xyntetik-kvist-14b-bf16"><code>Xyntetik-Kvist-14B-BF16.gguf</code></a></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
<tr><td><a href="#xyntetik-kvist-14b-q8_0"><code>Xyntetik-Kvist-14B-Q8_0.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
<tr><td><a href="#xyntetik-kvist-14b-q5_0-mix"><code>Xyntetik-Kvist-14B-Q5_0-mix.gguf</code></a></td><td></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td></td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
<tr><td><a href="#xyntetik-kvist-14b-iq4_nl-mix"><code>Xyntetik-Kvist-14B-IQ4_NL-mix.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td></td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Mellum2.1</b>, 1 model: CPU ✅ · CUDA ✅ · Metal - · Train ✅</summary>
<p><b>Mellum2.1-12B-A2.5B-Thinking (JetBrains)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#mellum2.1-12b-a2.5b-thinking-q4_k_m"><code>Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.0<br>2026-10-29</td><td nowrap>✅ 1.7.0<br>2026-10-29</td><td></td><td nowrap>✅ 1.7.0<br>2026-10-29</td></tr>
</tbody>
</table>
</details>

**Tier 2, maintained on report** (dated evidence, not gated):

<details>
<summary><b>Cloudflare Clef</b>, 3 models: CPU 🟨 2 of 7 · CUDA 🟨 3 of 7 · Metal - · Train 🟨 3 of 7</summary>
<p><b>Cloudflare Clef (clef, on Qwen3.8-27B)</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#cloudflare-clef-iq3_m"><code>Cloudflare_clef-IQ3_M.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
<p><b>Cloudflare Clef Flash (clef, on Qwen3.5-9B)</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#cloudflare-clef-flash"><code>Clef-Flash-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td nowrap>✅ 1.7.1<br>2026-10-30</td></tr>
<tr><td><a href="#cloudflare-clef-flash-q8_0"><code>Clef-Flash-Q8_0.gguf</code></a></td><td nowrap>✅ 1.7.1<sup>b</sup><br>2026-10-30</td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td nowrap>✅ 1.7.1<br>2026-10-30</td></tr>
</tbody>
</table>
<p><b>Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B)</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#cloudflare-clef-iq3_s-recovered"><code>Clef-backbone-IQ3_S.gguf</code></a></td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="#cloudflare-clef-iq3_s-native"><code>Clef-IQ3_S.clef.gguf</code></a></td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="#cloudflare-clef-iq3_s-recovery-qr-001e-lora"><code>clef-IQ3_S-recovery-QR-001e.lora.F16.gguf</code></a></td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="#cloudflare-clef-iq3_s-recovery-qr-001b-lora"><code>clef-IQ3_S-recovery-QR-001b.lora.F16.gguf</code></a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Qwen3.8</b>, 2 models: CPU - · CUDA 🟨 1 of 2 · Metal - · Train 🟨 1 of 2</summary>
<p><b>Qwen3.8-27B</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3.8-27b-ud-q4_k_m"><code>Qwen3.8-27B-UD-Q4_K_M.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
<p><b>Qwen3.8-Flash-Next</b>, experimental</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3.8-flash-next-gsq-rco-iq3_s"><code>Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf</code></a> (2 parts)</td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Gemma 4</b>, 5 models: CPU 🟨 2 of 5 · CUDA ✅ · Metal 🟨 1 of 5 · Train 🟨 3 of 5</summary>
<p><b>Gemma 4 12B</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#gemma-4-12b-it-qat-q4_0"><code>gemma-4-12b-it-qat-q4_0.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td></td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
</tbody>
</table>
<p><b>Gemma 4 26B-A4B</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#gemma-4-26b-a4b-it-qat-q4_0"><code>gemma-4-26B_q4_0-it.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td>no: MoE FFN (gemma-4 dual-branch)</td></tr>
</tbody>
</table>
<p><b>Gemma 4 31B</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#gemma-4-31b-it-q4_0"><code>gemma-4-31B-it-Q4_0.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
<p><b>Gemma 4 31B, attention block 48 removed</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#gemma-4-31b-it-attn48-removed-q4_0"><code>gemma-4-31B-it-attn48-removed-q4_0.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td>no: removed sublayers (--remove-sublayer artifact)</td></tr>
</tbody>
</table>
<p><b>Gemma 4 E2B</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#gemma-4-e2b-it-q4_k_m-q4_0-mix"><code>gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.5.1<br>2026-10-23</td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Muse-Glimmer-30B</b>, 2 models: CPU 🟨 1 of 4 · CUDA 🟨 3 of 4 · Metal - · Train ✅</summary>
<p><b>Muse-Glimmer-30B</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#muse-glimmer-30b-kquant"><code>Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
<p><b>Muse-Glimmer-30B surgical</b>, text path only</p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#muse-glimmer-30b-surgical-4p75-q4_k"><code>Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
<tr><td><a href="#muse-glimmer-30b-surgical-6p34-q8_0"><code>Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf</code></a></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
<tr><td><a href="#muse-glimmer-30b-surgical-6p34-q4_k-recovered"><code>Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Granite-4.2-8B</b>, 1 model: CPU ✅ · CUDA ✅ · Metal - · Train ✅</summary>
<p><b>Granite-4.2-8B</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#granite-4.2-8b-q4_k_m"><code>granite-4.2-8b-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.0<br>2026-10-29</td><td nowrap>✅ 1.7.0<br>2026-10-29</td><td></td><td nowrap>✅ 1.7.0<br>2026-10-29</td></tr>
<tr><td><a href="#granite-4.2-8b-q8_0"><code>granite-4.2-8b-Q8_0.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td></td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Trinity Nano</b>, 1 model: CPU ✅ · CUDA n/a · Metal n/a · Train n/a</summary>
<p><b>Trinity Nano</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#trinity-nano-preview-q8_0"><code>Trinity-Nano-Preview-Q8_0.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td>no: MoE router gating other than softmax</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Qwen3</b>, 4 models: CPU 🟨 3 of 5 · CUDA ✅ · Metal 🟨 2 of 5 · Train ✅</summary>
<p><b>Qwen3-30B-A3B selective precision</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3-30b-a3b-expq4_0-attnq8_0"><code>Qwen3-30B-A3B-expq4_0-attnq8_0.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
<p><b>Qwen3-4B</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3-4b-q4_k_m-bartowski"><code>Qwen_Qwen3-4B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.0<br>2026-10-28</td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
</tbody>
</table>
<p><b>Qwen3-4B Runner ToolUse</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3-4b-runner-tooluse-merged-q8_0"><code>merged-Q8_0.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td></td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
<tr><td><a href="#qwen3-4b-runner-tooluse-lora"><code>Qwen3-4B-ToolUse-LoRA.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.0<br>2026-10-28</td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
</tbody>
</table>
<p><b>Qwen3-Coder-30B-A3B keep-120</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#qwen3-coder-30b-a3b-instruct-keep120-q4_k_m"><code>Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf</code></a></td><td></td><td nowrap>✅ 1.5.1<sup>b</sup><br>2026-10-24</td><td></td><td nowrap>✅ 1.5.1<br>2026-10-24</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>NVIDIA Nemotron Nano 9B v2</b>, 1 model: CPU ✅ · CUDA ✅ · Metal n/a · Train n/a</summary>
<p><b>NVIDIA Nemotron Nano 9B v2</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#nvidia-nemotron-nano-9b-v2-q8_0"><code>NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td>no: recurrent architecture (Mamba-2)</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>gpt-oss</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>gpt-oss-20b keep-30</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#gpt-oss-20b-keep30-mxfp4"><code>gpt-oss-20b-keep30-MXFP4.gguf</code></a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Llama</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>Llama-style dense families (llama)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-llama">Llama-style dense families (llama)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Mistral</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>Mistral (mistral)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-mistral">Mistral (mistral)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>SmolLM</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>SmolLM (smollm)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-smollm">SmolLM (smollm)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>StableLM</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>StableLM (stablelm)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-stablelm">StableLM (stablelm)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Qwen2</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>Qwen2 (qwen2)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-qwen2">Qwen2 (qwen2)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Gemma 3</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>Gemma 3 (gemma3)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-gemma3">Gemma 3 (gemma3)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Phi-3</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>Phi-3 (phi3)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-phi3">Phi-3 (phi3)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Apertus</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>Apertus (apertus)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-apertus">Apertus (apertus)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Granite 4</b>, 1 model: CPU - · CUDA - · Metal - · Train -</summary>
<p><b>Granite 4 h-series hybrid (granitehybrid)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#arch-granitehybrid">Granite 4 h-series hybrid (granitehybrid)</a></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>MiniCPM5</b>, 1 model: CPU ✅ · CUDA ✅ · Metal ✅ · Train ✅</summary>
<p><b>MiniCPM5-2B (openbmb)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#minicpm5-2b-q8_0"><code>MiniCPM5-2B-Q8_0.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
<tr><td><a href="#minicpm5-2b-q4_k_m"><code>MiniCPM5-2B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td><td nowrap>✅ 1.7.1<br>2026-10-29</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>LFM2.5</b>, 2 models: CPU ✅ · CUDA - · Metal - · Train n/a</summary>
<p><b>LFM2.5-1.2B-Instruct (LiquidAI)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#lfm2.5-1.2b-instruct-q8_0"><code>LFM2.5-1.2B-Instruct-Q8_0.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td></td><td>no: recurrent architecture (short convolution)</td></tr>
<tr><td><a href="#lfm2.5-1.2b-instruct-q4_k_m"><code>LFM2.5-1.2B-Instruct-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td><a href="../MANUAL.md#support-matrix">n/a</a></td><td></td><td>no: recurrent architecture (short convolution)</td></tr>
</tbody>
</table>
<p><b>LFM2.5-350M (LiquidAI)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#lfm2.5-350m-q8_0"><code>LFM2.5-350M-Q8_0.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td></td><td></td><td>no: recurrent architecture (short convolution)</td></tr>
<tr><td><a href="#lfm2.5-350m-q4_k_m"><code>LFM2.5-350M-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td></td><td></td><td>no: recurrent architecture (short convolution)</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>MiMo</b>, 1 model: CPU ✅ · CUDA ✅ · Metal - · Train ✅</summary>
<p><b>MiMo-V2.6-Distill-Qwen-9B (XiaomiMiMo)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#mimo-v2.6-distill-qwen-9b-q4_k_m"><code>MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td></td><td nowrap>✅ 1.7.1<br>2026-10-30</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>Spark</b>, 2 models: CPU ✅ · CUDA - · Metal - · Train n/a</summary>
<p><b>Spark-X2.5-4B (XHToken)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#spark-x2.5-4b-q4_k_m"><code>Spark-X2.5-4B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td></td><td></td><td>no: per-head attention output gate (spark2_5) not in the backward</td></tr>
</tbody>
</table>
<p><b>Spark-X2.6-4B (XHToken)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#spark-x2.6-4b-q4_k_m"><code>Spark-X2.6-4B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td></td><td></td><td>no: per-head attention output gate (spark2_5) not in the backward</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>K2-Horizon</b>, 2 models: CPU ✅ · CUDA 🟨 2 of 4 · Metal 🟨 2 of 4 · Train 🟨 2 of 4</summary>
<p><b>K2-Horizon-0.9B (IFM)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#k2-horizon-0.9b-q8_0"><code>K2-Horizon-1B-Q8_0.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td nowrap>✅ 1.8.0<br>2026-10-30</td></tr>
<tr><td><a href="#k2-horizon-0.9b-q4_k_m"><code>K2-Horizon-1B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td nowrap>✅ 1.7.1<br>2026-10-30</td><td nowrap>✅ 1.8.0<br>2026-10-30</td></tr>
</tbody>
</table>
<p><b>K2-Horizon-7B (IFM)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#k2-horizon-7b-q8_0"><code>K2-Horizon-7B-Q8_0.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td></td><td></td><td>no: grouped RMSNorm (k2-horizon) not in the backward</td></tr>
<tr><td><a href="#k2-horizon-7b-q4_k_m"><code>K2-Horizon-7B-Q4_K_M.gguf</code></a></td><td nowrap>✅ 1.8.0<br>2026-10-30</td><td></td><td></td><td>no: grouped RMSNorm (k2-horizon) not in the backward</td></tr>
</tbody>
</table>
</details>

<details>
<summary><b>flint</b>, 1 model: CPU ✅ · CUDA - · Metal - · Train -</summary>
<p><b>flint (RecursiveSI)</b></p>
<table>
<thead><tr><th>File</th><th>CPU</th><th>CUDA</th><th>Metal</th><th>Train</th></tr></thead>
<tbody>
<tr><td><a href="#flint-int4-q4_1"><code>flint-int4-q4_1.gguf</code></a></td><td nowrap>✅ 1.7.1<sup>b</sup><br>2026-10-30</td><td></td><td></td><td></td></tr>
</tbody>
</table>
</details>

**Upcoming** (planned, not supported yet): Qwen3.8-35B-A3B, Ornith-1.5-35B-A3B (text path only), MiniCPM-V-4.7-35B-A3B (text path only), Granite-4.2-30B, Nemotron 3.5 Lightning-30B-A3B, Qwen3.6-27B, Nemotron 3 Nano (30B-A3B), Nemotron 3.5 Bolt-74B-A9B (text path only), Nemotron 3.5 Super VL-120B-A12B (text path only), Kolibri-1 (Aleph Alpha, 78B).

<a id="tier1"></a>
## Tier 1, actively validated

### Gemma 4

#### Text path only: Nemotron 3.5 Bolt E4B

<a id="nemotron-3.5-bolt-e4b-q4_k_m"></a>
##### `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`

Text path only: NVIDIA's post-train of Gemma-4-E4B-it in the publisher's own Q4_K_M, admitted on the owner's request to run on the Metal laptop; it loads, tokenizes as the publisher's tokenizer.json does and answers through its chat template, and no fidelity measurement against its BF16 parent has been made.

- **File:** `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` (sha256 `c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612`, 6,180,446,208 bytes, from [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) at revision `1db54d5b1253`).
- **Upstream checkpoint:** nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 at revision `49843eed8db1`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison is feasible and owed):** Family: gemma-4-E2B bf16 against Google's float32 reference (1.5.0 golden pass). This file: load, publisher tokenizer agreement (0 of 721 corpus strings differ) and a chat answer at temperature 0 on the Metal laptop and on the Windows CUDA host (CUDA, and CPU with the GPU off); the smoke reply is byte-identical across the Metal, CUDA and CPU cells (one output sha256), and the CPU-vs-CUDA check agreed on 9 of 9 prompts. A 20-step LoRA smoke with --train-gpu wrote an adapter that changes the answer. No fidelity or cross-engine number.
- **Context tested:** 4096.
- **Known limitations:** No vision: the publisher's mmproj is not part of this repository and would not be loaded. 6.18 GB of weights page from disk on an 8 GB machine with other applications open; the validation run recorded that state.
- **Still missing:** Fidelity against the BF16 parent (nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16); the repository's DFlash drafter file (Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf), which Runner has no path for.
- **Evidence:** [docs/compat-reports/1.7.0-2026-10-28-macos-arm64-nemotron-3.5-bolt-e4b-q4_k_m.json](compat-reports/1.7.0-2026-10-28-macos-arm64-nemotron-3.5-bolt-e4b-q4_k_m.json), [docs/compat-reports/1.7.0-2026-10-29-windows-x86_64-nemotron-3.5-bolt-e4b-q4_k_m.json](compat-reports/1.7.0-2026-10-29-windows-x86_64-nemotron-3.5-bolt-e4b-q4_k_m.json), [docs/compat-reports/train-smoke-1.7.0-2026-10-29-windows-x86_64-nemotron-3.5-bolt-e4b.md](compat-reports/train-smoke-1.7.0-2026-10-29-windows-x86_64-nemotron-3.5-bolt-e4b.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, metal backend; last run 1.7.0 on 2026-10-29.
- **CPU:** ✅ v1.7.0 2026-10-29 on windows-x86_64-cuda at commit `659df503e435`, output sha256 `0ae60d3e43338350`, [pin](../tests/compatibility/tier-smoke/nemotron-3.5-bolt-e4b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.0 2026-10-28 on windows-x86_64-cuda at commit `4a0a2c46ce39`, output sha256 `0ae60d3e43338350`, [pin](../tests/compatibility/tier-smoke/nemotron-3.5-bolt-e4b-q4_k_m.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.7.0 2026-10-29 on macos-arm64-metal at commit `0800f47350d5`, output sha256 `0ae60d3e43338350`, [pin](../tests/compatibility/tier-smoke/nemotron-3.5-bolt-e4b-q4_k_m.macos-arm64-metal.metal.txt)
- **8 GB laptop fit (Metal, 2026-10-29, runner 1.7.0 at commit `0800f47350d5`):** one resident server at `-c 4096 -b 32 --kv q8 --mlock` on an 8 GB Apple Silicon laptop wires 3.19 GB of weights while the 2.99 GB row-gathered table stays pageable; a 262-token prompt answers in 4.0-5.1 s to the first token and decodes 128 tokens at 8.5-9.0 tok/s; the first response after 8 minutes idle took 4.7 s with `--mlock` and 24.9 s without it (the weights were evicted while idle); the laptop swaps other applications to hold the weights, and the sliding-window KV ring (`RUNNER_KV_RING=1`) saved 77 MB of cache without changing these figures while losing the prefix cache.
- **LoRA training:** supported (runner 1.7.0, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.0, 2026-10-29).
- **Tier history:** Tier 2, maintained on report from 2026-10-28; Tier 1, actively validated from 2026-10-29 ([log](#transition-log)).

### Qwen3.8

#### Qwen3.8-27B

<a id="qwen3.8-27b-gsq-rco-iq3_s-recovered"></a>
##### `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf`

ISTA-DASLab's IQ3_S file with every quantised block's fp16 scales retrained against the BF16 parent, text path: it reads inside the fidelity bar against that parent and is not a lossless Qwen3.8-27B.

- **File:** `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf` (sha256 `873df48dd3ecf205f73c29716e80fbe766e4202ef4ed394b2c24607a8628458f`, 11,771,546,784 bytes, from [Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF) at revision `7b10f923a7f7`).
- **Upstream checkpoint:** Qwen/Qwen3.8-27B.
- **Quantization:** IQ3_S codebook mix, recovered scales.
- **Features claimed:** text, tools (qwen38). A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Against the BF16 parent: mean KLD 0.0450, margin-qualified top-1 97.80% on 500 held-out positions, inside the bar. Family: Qwen3.5-0.8B bf16 against Qwen's float32 reference (1.5.0 golden pass).
- **Context tested:** not recorded.
- **Still missing:** A CPU row on the file; pinned greedy output.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `6bc726d37ed11e94`, [pin](../tests/compatibility/tier-smoke/qwen3.8-27b-gsq-rco-iq3_s-recovered.windows-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): The qwen35 Gated DeltaNet hybrid has no Metal path; a Metal build runs this file on the CPU.
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="qwen3.8-27b-mtp-q8_0"></a>
##### `Qwen3.8-27B-MTP-Q8_0.gguf`

The NextN/MTP predictor head for Qwen3.8-27B, loaded with --draft mtp:FILE beside a trunk: it only proposes tokens, so the text is the trunk's own greedy output; it is a speed feature, not a model.

- **File:** `Qwen3.8-27B-MTP-Q8_0.gguf` (sha256 `89e31f13400bac27b888f10bab02ff4d8da1f8610fa078dc32f9f3f0d2b6f6c9`, 3,164,006,784 bytes, from [Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF) at revision `7b10f923a7f7`).
- **Upstream checkpoint:** Qwen/Qwen3.8-27B.
- **Quantization:** Q8_0.
- **Runs with:** [Qwen3.8-27B](#qwen3.8-27b-gsq-rco-iq3_s-recovered), loaded beside it with `--draft mtp:FILE`; the smoke and its pin are of that composed run.
- **Features claimed:** mtp. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** The trunk's greedy output is the reference: the head proposes, the trunk's verify walk decides, so greedy text with and without the head is the same.
- **Context tested:** not recorded.
- **Still missing:** Pinned output of the composed run (the trunk plus this head).
- **Evidence:** [MANUAL.md#placement-and-memory](../MANUAL.md#placement-and-memory).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `6bc726d37ed11e94`, [pin](../tests/compatibility/tier-smoke/qwen3.8-27b-mtp-q8_0.linux-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): MTP/NextN draft-head consumption is not implemented on Metal.
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

#### Experimental: Qwen3.8-Flash-Next

<a id="qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head"></a>
##### `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf` (4 parts)

Experimental: the qwen4exp architecture is admitted with its publisher float32 comparison still unmeasured, and no gate has run on this 4-part file; the admission evidence is on the upstream 2-part trunk.

- **File:** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf` (sha256 `522632bbb6f2c353621762be5bd48b270b1ff9fe60a9630bd47d5ef097aa0fa2`, 26,481,462,400 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00002-of-00004.gguf` (sha256 `f82cc8f151e15b73bbb84c75baf78f179df1b9b368437d7c75d8b3a7618cb8c8`, 28,347,013,600 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00003-of-00004.gguf` (sha256 `53ada4d2c3f2e189ce209520ecad0ae9649028b19e6bb9a55a021613b204e93b`, 28,800,138,432 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00004-of-00004.gguf` (sha256 `3a8c77474c82c5e81212ef3c5ef2330e75413cc60241d600fcfd87e4e2143b66`, 2,803,798,784 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`).
- **Upstream checkpoint:** Qwen/Qwen3.8-Flash-Next.
- **Quantization:** IQ3_S codebook mix with the MTP head (4 parts).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison is feasible and owed):** CPU forward anchored on llama.cpp on a toy fixture and on the upstream 2-part trunk (greedy 12 tokens identical to llama.cpp b11433), agreement with another engine; the publisher's float32 comparison is owed.
- **Context tested:** not recorded.
- **Still missing:** Every gate on this file; the publisher float32 comparison.
- **Evidence:** [docs/qwen4exp-admission-evidence/README.md](qwen4exp-admission-evidence/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `d10ba25da7047454`, [pin](../tests/compatibility/tier-smoke/qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head.linux-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): qwen4exp's Gated DeltaNet blocks have no Metal kernel (the support matrix names Gated DeltaNet among the operations Metal lacks); coverage.json records the architecture's Metal path as unsupported.
- **LoRA training:** no: hyper-connections (qwen4exp) (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="qwen3.8-flash-next-mtp-q8_0"></a>
##### `Qwen3.8-Flash-Next-MTP-Q8_0.gguf`

Experimental: the NextN/MTP head for Qwen3.8-Flash-Next as a companion file for --draft mtp:FILE, built from the publisher's BF16 checkpoint for ISTA-DASLab's IQ3_S trunk, whose qwen4exp architecture is admitted with its publisher float32 comparison still unmeasured; no gate has run on it.

- **File:** `Qwen3.8-Flash-Next-MTP-Q8_0.gguf` (sha256 `cd2f837a63f0b6ea1d951976e161fc40648932ffaf59194aed3557175638000b`, 4,165,605,056 bytes, from [Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF) at revision `c104e6579031`).
- **Upstream checkpoint:** Qwen/Qwen3.8-Flash-Next.
- **Quantization:** Q8_0.
- **Runs with:** [Qwen3.8-Flash-Next](#qwen3.8-flash-next-gsq-rco-iq3_s), loaded beside it with `--draft mtp:FILE`; the smoke and its pin are of that composed run.
- **Features claimed:** mtp. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison is feasible and owed):** The trunk's greedy output is the reference: the head proposes and the trunk decides. The trunk itself is experimental (see the trunk row).
- **Context tested:** not recorded.
- **Still missing:** Every gate, run beside ISTA-DASLab's 2-part trunk (the 4-part set already carries a head, and --draft mtp:FILE refuses a second one).
- **Evidence:** [MANUAL.md#placement-and-memory](../MANUAL.md#placement-and-memory).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `d10ba25da7047454`, [pin](../tests/compatibility/tier-smoke/qwen3.8-flash-next-mtp-q8_0.linux-x86_64-cuda.cpu.txt)
- **CUDA:** [n/a](../MANUAL.md#support-matrix): --draft mtp decoding on CUDA needs the whole trunk on the device; the IQ3_S trunk this head runs with holds 18 of 48 layers on a 24 GB device, and the partial split refuses with an error naming --gpu off (Linux CUDA host, 2026-10-25, runner f960e470). The head is smoked on the CPU path of the same class.
- **Metal:** [n/a](../MANUAL.md#support-matrix): MTP/NextN draft-head consumption is not implemented on Metal.
- **LoRA training:** no: hyper-connections (qwen4exp) (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Xyntetik-Kvist-14B

#### Xyntetik-Kvist-14B

<a id="xyntetik-kvist-14b-bf16"></a>
##### `Xyntetik-Kvist-14B-BF16.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its BF16 file.

- **File:** `Xyntetik-Kvist-14B-BF16.gguf` (sha256 `daf422db9d0853d53ec8abb8fe9cd3f1e33ea53dffe7cca476225fbe1bc5d5c6`, 28,903,141,184 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** BF16.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `3b2c21c82306`, output sha256 `c89c26e516c5d4dd`, [pin](../tests/compatibility/tier-smoke/xyntetik-kvist-14b-bf16.linux-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="xyntetik-kvist-14b-q8_0"></a>
##### `Xyntetik-Kvist-14B-Q8_0.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its Q8_0 file.

- **File:** `Xyntetik-Kvist-14B-Q8_0.gguf` (sha256 `cdb73543a6fc22f7faa5bbab0176ab2d5b1b590b46870d2d22f2a2510caaacb6`, 15,363,224,384 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** Q8_0.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `c89c26e516c5d4dd`, [pin](../tests/compatibility/tier-smoke/xyntetik-kvist-14b-q8_0.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="xyntetik-kvist-14b-q5_0-mix"></a>
##### `Xyntetik-Kvist-14B-Q5_0-mix.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its Q5_0 mix file.

- **File:** `Xyntetik-Kvist-14B-Q5_0-mix.gguf` (sha256 `37af54b4cef4445bbbf23e8fbf0ceec8dc9b01b38dab0dc7f9e2bd2fd79f3a56`, 10,295,023,424 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** Q5_0 mix.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `c89c26e516c5d4dd`, [pin](../tests/compatibility/tier-smoke/xyntetik-kvist-14b-q5_0-mix.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

<a id="xyntetik-kvist-14b-iq4_nl-mix"></a>
##### `Xyntetik-Kvist-14B-IQ4_NL-mix.gguf`

Built for one job, tool-calling agent loops on this runner from a 24 GB card, and not a drop-in replacement for Muse-Glimmer-30B or any general-purpose model: a dense 14B student distilled from Muse-Glimmer-30B, this is its IQ4_NL mix file.

- **File:** `Xyntetik-Kvist-14B-IQ4_NL-mix.gguf` (sha256 `c74b27a5a39c894c4d1a1a2ae140de0cee125fc5cc9a7c96db9b7355bf8709a3`, 7,612,384,384 bytes, from [Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF](https://huggingface.co/Joakimpalm-Zen/Xyntetik-Kvist-14B-GGUF) at revision `e6f66cdda4c5`).
- **Upstream checkpoint:** Joakimpalm-Zen/Xyntetik-Kvist-14B.
- **Quantization:** IQ4_NL mix.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (this project is the publisher and its BF16 file is the reference):** This project is the publisher: the BF16 file is the reference. The release gate was task-level (five arms, 57 of 60 held-out tasks where the parent solves 60), not a fidelity bar.
- **Context tested:** not recorded.
- **Still missing:** Per-file backend rows; pinned output.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `c89c26e516c5d4dd`, [pin](../tests/compatibility/tier-smoke/xyntetik-kvist-14b-iq4_nl-mix.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `c89c26e516c5d4dd`, [pin](../tests/compatibility/tier-smoke/xyntetik-kvist-14b-iq4_nl-mix.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22 ([log](#transition-log)).

### Mellum2.1

#### Mellum2.1-12B-A2.5B-Thinking (JetBrains)

<a id="mellum2.1-12b-a2.5b-thinking-q4_k_m"></a>
##### `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf`

Text path, JetBrains' own Q4_K_M. Publisher Q4_K_M against the publisher's BF16 parent: mean KLD 0.128, top-1 85.7% (Runner), 0.131 / 85.9% (llama.cpp); the file fails the 0.05 bar on both engines and the gap grows with depth. Against the publisher Q8_0 it fails the quant bar too (mean KLD 0.131, top-1 80.0%, 400 positions); Q8_0 (12.9 GB) is the fidelity file (Runner Q8_0 vs the parent 0.0187 / 94.3%), and the publisher's MXFP4_MOE file is untested.

- **File:** `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf` (sha256 `ecc4d5b8107fc219e23c494b2d107552137637b6c6002ee713e0ba8af7fa2755`, 8,071,295,264 bytes, from [JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF) at revision `4159d684ffdb`).
- **Upstream checkpoint:** JetBrains/Mellum2.1-12B-A2.5B-Thinking at revision `92ddae9fc766`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** This file against the BF16 parent computed in float32 (transformers' own mellum code, 300 positions to 7,420 tokens, token-id keyed): Runner 0.1282 mean KLD / 85.7% top-1 / 93.7% margin-qualified, llama.cpp b11748 0.1305 / 85.9% / 94.0% (298 scored), about 0.08 in words 0-750 and 0.16 past 2,000; Runner vs llama.cpp on this file 0.0177. Q8_0 against the parent: Runner 0.0187 / 94.3% / 100%, llama.cpp 0.0234 / 93.6% / 100%. Q4_K_M vs Q8_0 bar FAIL (400 positions, 0.131, 80.0%); tokenizer 0/721, template 23/23, a needle retrieved at 3,568 tokens. Record: shade research/evaluation/mellum-parity-2026-10-29 (README.md, reference/README.md, windows-q4/README.md).
- **Context tested:** not recorded.
- **Known limitations:** No fidelity claim for Q4_K_M: it fails the 0.05 bar against the BF16 parent on both engines, and the gap is the file, not the engine. The publisher's MXFP4_MOE file is the untested alternative. CUDA MoE on the scalar route; tensor-core and grouped routes not measured for this family. The 8.07 GB file does not load resident on an 8 GB machine, and on an 8 GB card it runs as a layer split.
- **Still missing:** The Metal row; the Q8_0 cells (12.9 GB is not servable on the Windows CUDA host's 16 GB beside its other work).
- **Evidence:** [tests/compatibility/coverage.json](../tests/compatibility/coverage.json), [docs/compat-reports/1.7.0-2026-10-29-windows-x86_64-mellum2.1-12b-a2.5b-thinking-q4_k_m.json](compat-reports/1.7.0-2026-10-29-windows-x86_64-mellum2.1-12b-a2.5b-thinking-q4_k_m.json), [docs/compat-reports/cpu_cuda/mellum2.1-12b-a2.5b-thinking-q4_k_m.json](compat-reports/cpu_cuda/mellum2.1-12b-a2.5b-thinking-q4_k_m.json).
- **Validation run:** on windows-x86_64-cuda, cuda backend; last run 1.7.0 on 2026-10-29.
- **CPU:** ✅ v1.7.0 2026-10-29 on windows-x86_64-cuda at commit `502805d33ecc`, output sha256 `d419bee5d375409e`, [pin](../tests/compatibility/tier-smoke/mellum2.1-12b-a2.5b-thinking-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.0 2026-10-29 on windows-x86_64-cuda at commit `502805d33ecc`, output sha256 `d419bee5d375409e`, [pin](../tests/compatibility/tier-smoke/mellum2.1-12b-a2.5b-thinking-q4_k_m.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.0, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.0, 2026-10-29).
- **Tier history:** Upcoming, planned admission from 2026-10-22; Tier 2, maintained on report from 2026-10-29; Tier 1, actively validated from 2026-10-29 ([log](#transition-log)).

<a id="tier2"></a>
## Tier 2, maintained on report

### Cloudflare Clef

#### Text path only: Cloudflare Clef (clef, on Qwen3.8-27B)

<a id="cloudflare-clef-iq3_m"></a>
##### `Cloudflare_clef-IQ3_M.gguf`

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
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `a38540f4ff7fd6c7`, [pin](../tests/compatibility/tier-smoke/cloudflare-clef-iq3_m.linux-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): Clef's qwen35 backbone has no Metal path; a Metal build runs it on the CPU.
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Text path only: Cloudflare Clef Flash (clef, on Qwen3.5-9B)

<a id="cloudflare-clef-flash"></a>
##### `Clef-Flash-Q4_K_M.gguf`

Text path only; ggml-org's Q4_K_M changes 3 of 29 decisions against Cloudflare's float32 reference (26 of 29 agree, max |dprob| 0.199), each where the reference's top two are within 0.066 of each other; the Q8_0 agrees on 29 of 29 and is the file for decisions that must match. A decision model served at /v1/systemone, in ggml-org's conversion of Cloudflare/clef-flash at f0b7d3cd (converter ggml-org/convert, the revision its .src_sha names; ggml-org/Clef-Flash-GGUF at 6e2a164d), as Cloudflare publishes no GGUF; requests with images or videos are refused.

- **File:** `Clef-Flash-Q4_K_M.gguf` (sha256 `eef2968c238a80d49d340430f22410ea3db9934ad1c6fdb233a503cbf9a9b54e`, 6,732,781,824 bytes, from [ggml-org/Clef-Flash-GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF) at revision `6e2a164d70aa`).
- **Upstream checkpoint:** Cloudflare/clef-flash at revision `f0b7d3cd6529`.
- **Quantization:** Q4_K_M (ggml-org).
- **Features claimed:** decisions. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both ggml-org files against Cloudflare's own joint_schema_model.py at f0b7d3cd in float32 on the lab CPU lane (15 records, 29 questions; input ids and spans equal on every record; Runner --systemone at 686a31da, the GPU off): Q8_0 29 of 29 decisions, max |dprob| 0.012, mean 0.0027; Q4_K_M 26 of 29, max 0.199, mean 0.025, its three flips where the reference's top two are 0.020 to 0.066 apart. Record: docs/compat-reports/clef-flash-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** Float32 reference owed, so no fidelity claim: ggml-org's Q4_K_M keeps attention, the SSM projections, the output matrix and the decision head at Q8_0 and puts the feed-forward and the token embedding at Q4_K; it is not bartowski's mix of the same name. Text only: a request with images or videos is refused. No llama.cpp column.
- **Still missing:** The Metal cell (a declared limit today); a llama.cpp column against the same reference.
- **Evidence:** [docs/compat-reports/clef-flash-reference-2026-10-30/README.md](compat-reports/clef-flash-reference-2026-10-30/README.md), [docs/compat-reports/clef-flash-2026-10-30/README.md](compat-reports/clef-flash-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json), [docs/compat-reports/1.7.1-2026-10-30-windows-x86_64-clef-flash-q4_k_m.json](compat-reports/1.7.1-2026-10-30-windows-x86_64-clef-flash-q4_k_m.json).
- **Validation run:** on windows-x86_64-cuda, cuda backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `b9ad4e9dc9a6`, output sha256 `0b9c736613433a3c`, [pin](../tests/compatibility/tier-smoke/cloudflare-clef-flash.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `b9ad4e9dc9a6`, output sha256 `0b9c736613433a3c`, [pin](../tests/compatibility/tier-smoke/cloudflare-clef-flash.windows-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): Clef's qwen35 backbone has no Metal path; a Metal build runs it on the CPU.
- **LoRA training:** supported (runner 1.7.1, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-22; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

<a id="cloudflare-clef-flash-q8_0"></a>
##### `Clef-Flash-Q8_0.gguf`

Text path only; the fidelity file of ggml-org's two: against Cloudflare's float32 reference it picks the reference's option on 29 of 29 questions (max |dprob| 0.012). A decision model served at /v1/systemone, in ggml-org's conversion of Cloudflare/clef-flash at f0b7d3cd (converter ggml-org/convert, the revision its .src_sha names; ggml-org/Clef-Flash-GGUF at 6e2a164d), as Cloudflare publishes no GGUF; requests with images or videos are refused.

- **File:** `Clef-Flash-Q8_0.gguf` (sha256 `25f801e2a27610b0ccc0cb1558d7eda3d7a304e34e2c4a04e3079357ff7d906e`, 9,657,260,288 bytes, from [ggml-org/Clef-Flash-GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF) at revision `6e2a164d70aa`).
- **Upstream checkpoint:** Cloudflare/clef-flash at revision `f0b7d3cd6529`.
- **Quantization:** Q8_0 (ggml-org).
- **Features claimed:** decisions. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both ggml-org files against Cloudflare's own joint_schema_model.py at f0b7d3cd in float32 on the lab CPU lane (15 records, 29 questions; input ids and spans equal on every record; Runner --systemone at 686a31da, the GPU off): Q8_0 29 of 29 decisions, max |dprob| 0.012, mean 0.0027; Q4_K_M 26 of 29, max 0.199, mean 0.025, its three flips where the reference's top two are 0.020 to 0.066 apart. Record: docs/compat-reports/clef-flash-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** Float32 reference owed, so no fidelity claim: nothing is measured against the parent for either clef-flash file yet. Text only: a request with images or videos is refused. No llama.cpp column.
- **Still missing:** The Metal cell (a declared limit today); its compat report's CPU-against-CUDA check (the lab CPU lane has no CUDA device); a llama.cpp column against the same reference.
- **Evidence:** [docs/compat-reports/clef-flash-reference-2026-10-30/README.md](compat-reports/clef-flash-reference-2026-10-30/README.md), [docs/compat-reports/clef-flash-2026-10-30/README.md](compat-reports/clef-flash-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json), [docs/compat-reports/1.7.1-2026-10-30-linux-x86_64-clef-flash-q8_0.json](compat-reports/1.7.1-2026-10-30-linux-x86_64-clef-flash-q8_0.json).
- **Validation run:** on windows-x86_64-cuda, cuda backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.7.1 2026-10-30 on linux-x86_64-cuda (borrowed hardware) at commit `686a31da93e8`, output sha256 `1b3a370279fc586b`, [pin](../tests/compatibility/tier-smoke/cloudflare-clef-flash-q8_0.linux-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `b9ad4e9dc9a6`, output sha256 `1b3a370279fc586b`, [pin](../tests/compatibility/tier-smoke/cloudflare-clef-flash-q8_0.windows-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): Clef's qwen35 backbone has no Metal path; a Metal build runs it on the CPU.
- **LoRA training:** supported (runner 1.7.1, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-30).
- **Tier history:** Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

#### Text path only: Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B)

<a id="cloudflare-clef-iq3_s-recovered"></a>
##### `Clef-backbone-IQ3_S.gguf`

Text path only: a decision model served at /v1/systemone; requests with images or videos are refused. The project's own IQ3_S quantization of the Clef backbone with two recovery adapters trained on it; the house bar against the BF16 twin (mean KLD at most 0.05 and margin-qualified top-1 at least 97% on 500 held-out positions) is NOT met: the base reads 0.0879 / 94.0%, the best adapter 0.0736 / 95.6%, stated on the card with the evidence.

- **File:** `Clef-backbone-IQ3_S.gguf` (sha256 `dafa8752f756f21c941ffee1df6ab93fec499eaeddfea7e5ba5ba601abf747cb`, 12,419,329,216 bytes, from [Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF) at revision `9e9ce879b6cb`).
- **Upstream checkpoint:** Cloudflare/clef at revision `0b331204bb13`.
- **Quantization:** IQ3_S.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; a new repository has no downloads yet, and the card leads with the measured shortfall.
- **Tier history:** Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

<a id="cloudflare-clef-iq3_s-native"></a>
##### `Clef-IQ3_S.clef.gguf`

Text path only: the full Clef decision model in one file, the project's IQ3_S backbone with Cloudflare's own BF16 joint decision head unchanged, served at /v1/systemone; requests with images or videos are refused. Measured against the BF16 parent with the same head on 2,000 typed decisions: 92.25% the parent's answer (92.95% with the QR-001e adapter), none of the parent's 704 confident decisions changed with the adapter; descriptive, there is no bar for decisions, and as text the backbone does NOT meet the house bar, as its card states.

- **File:** `Clef-IQ3_S.clef.gguf` (sha256 `1e142e60b7067673971bbcbcb7af316409f530a24857770d30dae492932f0efe`, 12,675,711,328 bytes, from [Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF) at revision `0a24dd670d52`).
- **Upstream checkpoint:** Cloudflare/clef at revision `0b331204bb13`.
- **Quantization:** IQ3_S backbone, BF16 decision head.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull, and this repository is under the download bar. Published on the owner's direct request (2026-10-28) as the full decision model of the project's Clef IQ3_S files; its card leads with the measured shortfall against the house bar and reports decisions descriptively.
- **Numerical anchor:** Against the BF16 parent as a native Clef GGUF with the same head, 2,000 decisions of LocalLLaMA/typed-decisions at 7189f46b: 92.25% same answer, accuracy 73.45% (parent 72.30%), mean total variation 0.0501; the published Runner 1.6.1 asset reproduced 50 of 50 recorded answers with max |dlogit| 3.2e-4 (the card's evidence files).
- **Tier history:** Tier 2, maintained on report from 2026-10-28 ([log](#transition-log)).

<a id="cloudflare-clef-iq3_s-recovery-qr-001e-lora"></a>
##### `clef-IQ3_S-recovery-QR-001e.lora.F16.gguf`

Text path only: the QR-001e recovery adapter for the project's Clef IQ3_S backbone, served with --lora on that file: an adapter, not a model; the best recovery round (house reading 0.0736 / 95.6%), below the house bar, as its card states.

- **File:** `clef-IQ3_S-recovery-QR-001e.lora.F16.gguf` (sha256 `67e249a6d95c20e1128f4f1ceabde9b1c311988a1154766562c5c44e56380b64`, 233,523,840 bytes, from [Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF) at revision `9e9ce879b6cb`).
- **Upstream checkpoint:** Cloudflare/clef at revision `0b331204bb13`.
- **Quantization:** adapter (F16 LoRA tensors).
- **Runs with:** [Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B)](#cloudflare-clef-iq3_s-recovered), loaded beside it with `--lora`; the smoke and its pin are of that composed run.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; a new repository has no downloads yet, and the card leads with the measured shortfall.
- **Tier history:** Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

<a id="cloudflare-clef-iq3_s-recovery-qr-001b-lora"></a>
##### `clef-IQ3_S-recovery-QR-001b.lora.F16.gguf`

Text path only: the QR-001b recovery adapter for the project's Clef IQ3_S backbone, served with --lora on that file: an adapter, not a model; the first recovery round (house reading 0.0757 / 95.6%), below the house bar, as its card states.

- **File:** `clef-IQ3_S-recovery-QR-001b.lora.F16.gguf` (sha256 `2816dff4dbe0db506cf9c1b21d89f164a851f8a0b12fb253da3762a7928e1ae9`, 233,523,936 bytes, from [Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Cloudflare-clef-IQ3_S-recovered-GGUF) at revision `9e9ce879b6cb`).
- **Upstream checkpoint:** Cloudflare/clef at revision `0b331204bb13`.
- **Quantization:** adapter (F16 LoRA tensors).
- **Runs with:** [Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B)](#cloudflare-clef-iq3_s-recovered), loaded beside it with `--lora`; the smoke and its pin are of that composed run.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; a new repository has no downloads yet, and the card leads with the measured shortfall.
- **Tier history:** Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

### Qwen3.8

#### Qwen3.8-27B

<a id="qwen3.8-27b-ud-q4_k_m"></a>
##### `Qwen3.8-27B-UD-Q4_K_M.gguf`

Unsloth's dynamic Q4_K_M of Qwen3.8-27B, text path: it passes the quant bar against the Q8_0 sibling; the qwen35 CUDA path was measured on other quantizations of the same model.

- **File:** `Qwen3.8-27B-UD-Q4_K_M.gguf` (sha256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`, 16,464,440,224 bytes, from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision `4ca720788d1e`).
- **Upstream checkpoint:** Qwen/Qwen3.8-27B.
- **Quantization:** UD-Q4_K_M (with 4 IQ3_S tensors).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: Qwen3.5-0.8B bf16 against Qwen's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass). This file: UD-Q4_K_M vs Q8_0 bar PASS (100 positions, mean KLD 0.0016, top-1 100%); tokenizer 0/721, template 21/21.
- **Context tested:** not recorded.
- **Still missing:** A CPU row on the file; pinned greedy output.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `44ffbc5ccee57adc`, [pin](../tests/compatibility/tier-smoke/qwen3.8-27b-ud-q4_k_m.linux-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): The qwen35 Gated DeltaNet hybrid has no Metal path; a Metal build runs this file on the CPU.
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Experimental: Qwen3.8-Flash-Next

<a id="qwen3.8-flash-next-gsq-rco-iq3_s"></a>
##### `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (2 parts)

Experimental: ISTA-DASLab's IQ3_S trunk of Qwen3.8-Flash-Next, the file the published MTP head was built for and the one the qwen4exp admission evidence measured; maintained on report, and run on every release only as that head's trunk in its composed smoke.

- **File:** `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (sha256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`, 54,817,524,224 bytes, from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at revision `ed59f92082b1`); `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`, 28,800,138,432 bytes, from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at revision `ed59f92082b1`).
- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Gemma 4

#### Text path only: Gemma 4 12B

<a id="gemma-4-12b-it-qat-q4_0"></a>
##### `gemma-4-12b-it-qat-q4_0.gguf`

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
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `109c0663eb8c13c0`, [pin](../tests/compatibility/tier-smoke/gemma-4-12b-it-qat-q4_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `109c0663eb8c13c0`, [pin](../tests/compatibility/tier-smoke/gemma-4-12b-it-qat-q4_0.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Text path only: Gemma 4 26B-A4B

<a id="gemma-4-26b-a4b-it-qat-q4_0"></a>
##### `gemma-4-26B_q4_0-it.gguf`

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
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `c697fa9c64f4a370`, [pin](../tests/compatibility/tier-smoke/gemma-4-26b-a4b-it-qat-q4_0.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** no: MoE FFN (gemma-4 dual-branch) (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** no: --lora does not cover the gemma-4 dual-branch FFN yet (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Text path only: Gemma 4 31B

<a id="gemma-4-31b-it-q4_0"></a>
##### `gemma-4-31B-it-Q4_0.gguf`

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
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `6e01ea7bc489371c`, [pin](../tests/compatibility/tier-smoke/gemma-4-31b-it-q4_0.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Text path only: Gemma 4 31B, attention block 48 removed

<a id="gemma-4-31b-it-attn48-removed-q4_0"></a>
##### `gemma-4-31B-it-attn48-removed-q4_0.gguf`

Text path only: Google's QAT Q4_0 of Gemma 4 31B with one attention sublayer physically removed; it tracks its parent at clean KLD 0.0223 on 44,413 held-out positions and loads only in Runner (llama.cpp refuses it by name).

- **File:** `gemma-4-31B-it-attn48-removed-q4_0.gguf` (sha256 `d7eaef9a8c05d8ad5cc63e4b37767113498b6ec6a31863519f5399f12cdb3134`, 17,576,681,600 bytes, from [Joakimpalm-Zen/gemma-4-31B-it-attn48-removed-Q4_0-GGUF](https://huggingface.co/Joakimpalm-Zen/gemma-4-31B-it-attn48-removed-Q4_0-GGUF) at revision `c704744772b4`).
- **Upstream checkpoint:** google/gemma-4-31B-it.
- **Quantization:** Q4_0 (Google QAT), one attention sublayer removed.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 454 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Against its parent, Google's QAT Q4_0: clean KLD 0.0223 on 44,413 held-out positions (2026-08-30, on the identical zeroed form). Family: gemma-4-E2B bf16 against Google's float32 reference.
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [docs/sublayer-removal.md](sublayer-removal.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `fbb8f3caf0ab7e53`, [pin](../tests/compatibility/tier-smoke/gemma-4-31b-it-attn48-removed-q4_0.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** no: removed sublayers (--remove-sublayer artifact) (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** no: --lora does not cover a model with removed sublayers yet (1 removed by --remove-sublayer) (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Text path only: Gemma 4 E2B

<a id="gemma-4-e2b-it-q4_k_m-q4_0-mix"></a>
##### `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`

Text path only: the smoke-test file from the quickstart; it fails the fidelity bar against its own BF16 parent (77.75% of tokens agree, mean KLD 0.286) and is a way to try Runner on an 8 GB machine, not a faithful gemma-4-E2B.

- **File:** `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf` (sha256 `16ffaa2ea568e7a0c9b5c6ddbd39a376879f9ac007861334974de39f4cc52e3f`, 2,631,015,904 bytes, from [Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF](https://huggingface.co/Joakimpalm-Zen/gemma-4-E2B-it-Q4_0-GGUF) at revision `37c8fa438245`).
- **Upstream checkpoint:** google/gemma-4-E2B-it.
- **Quantization:** Q4_K_M/Q4_0 mix.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 552 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: gemma-4-E2B bf16 against Google's float32 reference, where Runner is the closer engine (1.5.0 golden pass); that pass's served row is a different file (q4_0 throughout), not this mix.
- **Context tested:** not recorded.
- **Still missing:** nothing the audit named
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on macos-arm64-metal, metal backend; last run 1.5.1 on 2026-10-23.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `4132fa520bc517ed`, [pin](../tests/compatibility/tier-smoke/gemma-4-e2b-it-q4_k_m-q4_0-mix.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `4132fa520bc517ed`, [pin](../tests/compatibility/tier-smoke/gemma-4-e2b-it-q4_k_m-q4_0-mix.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.5.1 2026-10-23 on macos-arm64-metal at commit `7660980968b4`, output sha256 `4132fa520bc517ed`, [pin](../tests/compatibility/tier-smoke/gemma-4-e2b-it-q4_k_m-q4_0-mix.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

### Muse-Glimmer-30B

#### Text path only: Muse-Glimmer-30B

<a id="muse-glimmer-30b-kquant"></a>
##### `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`

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
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `bfe1008c84ebfae4`, [pin](../tests/compatibility/tier-smoke/muse-glimmer-30b-kquant.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Text path only: Muse-Glimmer-30B surgical

<a id="muse-glimmer-30b-surgical-4p75-q4_k"></a>
##### `Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf`

Text path only: Muse-Glimmer-30B with 4.75% of the decoder removed and healed, at Q4_K; it passes the bar against the unmodified BF16 parent on a held-out split.

- **File:** `Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf` (sha256 `103aa117dfce88974bd78e03ee6ca6601c720f01a1ec24ee1cfd0e5fe2f77ce6`, 15,686,266,304 bytes, from [Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q4_K-GGUF](https://huggingface.co/Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q4_K-GGUF) at revision `38f6284ec919`).
- **Upstream checkpoint:** meta-models/Muse-Glimmer-30B.
- **Quantization:** Q4_K.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 298 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. The family's only member is 30B (120 GB in float32). Against the unmodified BF16 parent: 97.74% / 0.04810 on a held-out split.
- **Context tested:** not recorded.
- **Still missing:** Backend rows; pinned output; a live tool-protocol run.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `9d806b71f48e1049`, [pin](../tests/compatibility/tier-smoke/muse-glimmer-30b-surgical-4p75-q4_k.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

<a id="muse-glimmer-30b-surgical-6p34-q8_0"></a>
##### `Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf`

Text path only: Muse-Glimmer-30B with 6.34% of the decoder removed (the FFN sublayers of layers 4, 7, 9 and 48), at Q8_0; it passes the bar against the unmodified BF16 parent on a held-out split.

- **File:** `Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf` (sha256 `d7bb5c6f3b7170160322c684f338c0db630ea24ab09b896a6c97c8c415f58d4d`, 29,612,961,216 bytes, from [Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q8_0-GGUF](https://huggingface.co/Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-Q8_0-GGUF) at revision `e95fcd1a298c`).
- **Upstream checkpoint:** meta-models/Muse-Glimmer-30B.
- **Quantization:** Q8_0.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 339 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. The family's only member is 30B (120 GB in float32). Against the unmodified BF16 parent: 98.34% / 0.03883 on a held-out split.
- **Context tested:** not recorded.
- **Still missing:** Backend rows; pinned output; a live tool-protocol run.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `3b2c21c82306`, output sha256 `40a1110e1716fb3d`, [pin](../tests/compatibility/tier-smoke/muse-glimmer-30b-surgical-6p34-q8_0.linux-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

<a id="muse-glimmer-30b-surgical-6p34-q4_k-recovered"></a>
##### `Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf`

Text path only: the 6.34% surgical Muse-Glimmer-30B at Q4_K with retrained scales; it passes the bar (0.04949, 97.90%) where plain Q4_K of the same model fails it (0.05815).

- **File:** `Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf` (sha256 `12a429483b18aefcfd9cdc41afb156f339d610be0fdd66fdaf5391e9df1cbb19`, 15,686,266,304 bytes, from [Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-6p34-Q4_K-recovered-GGUF](https://huggingface.co/Joakimpalm-Zen/Muse-Glimmer-30B-Surgical-6p34-Q4_K-recovered-GGUF) at revision `164d823e32a3`).
- **Upstream checkpoint:** meta-models/Muse-Glimmer-30B.
- **Quantization:** Q4_K, recovered scales.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 292 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. The family's only member is 30B (120 GB in float32). Against the unmodified BF16 parent: mean KLD 0.04949, margin-qualified top-1 97.90%.
- **Context tested:** not recorded.
- **Still missing:** Backend rows; pinned output; a live tool-protocol run.
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `80e114dfae593916`, [pin](../tests/compatibility/tier-smoke/muse-glimmer-30b-surgical-6p34-q4_k-recovered.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

### Granite-4.2-8B

#### Granite-4.2-8B

<a id="granite-4.2-8b-q4_k_m"></a>
##### `granite-4.2-8b-Q4_K_M.gguf`

IBM's official Q4_K_M of Granite 4.2 8B, text path: the granite42 tool protocol and two client loops pass on an 8 GB CUDA card; case C is open.

- **File:** `granite-4.2-8b-Q4_K_M.gguf` (sha256 `16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311`, 5,347,917,952 bytes, from [ibm-granite/granite-4.2-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) at revision `93f3f6a8938e`).
- **Upstream checkpoint:** ibm-granite/granite-4.2-8b.
- **Quantization:** Q4_K_M (IBM).
- **Features claimed:** text, tools (granite42). A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: granite-4.2-3B bf16 against IBM's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass). This file: Q4_K_M vs Q8_0 bar PASS (400 positions, mean KLD 0.0036); tokenizer 0/721, template 22/22.
- **Context tested:** not recorded.
- **Still missing:** The Metal row.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json), [docs/compat-reports/1.7.0-2026-10-29-windows-x86_64-granite-4.2-8b-q4_k_m.json](compat-reports/1.7.0-2026-10-29-windows-x86_64-granite-4.2-8b-q4_k_m.json), [docs/compat-reports/cpu_cuda/granite-4.2-8b-q4_k_m.json](compat-reports/cpu_cuda/granite-4.2-8b-q4_k_m.json).
- **Validation run:** on windows-x86_64-cuda, cuda backend; last run 1.7.0 on 2026-10-29.
- **CPU:** ✅ v1.7.0 2026-10-29 on windows-x86_64-cuda at commit `d397f67af886`, output sha256 `ff1cd210d7a92cd2`, [pin](../tests/compatibility/tier-smoke/granite-4.2-8b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.0 2026-10-29 on windows-x86_64-cuda at commit `047e8e346312`, output sha256 `ff1cd210d7a92cd2`, [pin](../tests/compatibility/tier-smoke/granite-4.2-8b-q4_k_m.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.0, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.0, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

<a id="granite-4.2-8b-q8_0"></a>
##### `granite-4.2-8b-Q8_0.gguf`

IBM's official Q8_0 of Granite 4.2 8B, text path: the reference sibling of the Q4_K_M row's quant bar.

- **File:** `granite-4.2-8b-Q8_0.gguf` (sha256 `fb66ad5750680c77c76b9dc095961375f14fc88ba15fd5fc084ef6e6701bad77`, 9,345,613,952 bytes, from [ibm-granite/granite-4.2-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) at revision `93f3f6a8938e`).
- **Upstream checkpoint:** ibm-granite/granite-4.2-8b.
- **Quantization:** Q8_0 (IBM).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: granite-4.2-3B bf16 against IBM's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass).
- **Context tested:** not recorded.
- **Still missing:** Backend rows on the file; pinned output.
- **Evidence:** [docs/granite-42-qwen38-cert-2026-09-06.md](granite-42-qwen38-cert-2026-09-06.md), [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `9e4082576850d0a8`, [pin](../tests/compatibility/tier-smoke/granite-4.2-8b-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `9e4082576850d0a8`, [pin](../tests/compatibility/tier-smoke/granite-4.2-8b-q8_0.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

### Trinity Nano

#### Trinity Nano

<a id="trinity-nano-preview-q8_0"></a>
##### `Trinity-Nano-Preview-Q8_0.gguf`

Arcee's Q8_0 of Trinity Nano Preview, text path, CPU only by admission: the one afmoe file Runner actively validates.

- **File:** `Trinity-Nano-Preview-Q8_0.gguf` (sha256 `5fcc2428c325e8b2035d280ff28c4c659ea240ce317a9536f230acfc352b79d1`, 6,531,428,640 bytes, from [arcee-ai/Trinity-Nano-Preview-GGUF](https://huggingface.co/arcee-ai/Trinity-Nano-Preview-GGUF) at revision `2aa08593b792`).
- **Upstream checkpoint:** arcee-ai/Trinity-Nano-Preview.
- **Quantization:** Q8_0.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** This file against Arcee's float32 reference (1.5.0 golden pass, 2,000 positions): margin-qualified top-1 97.43%, mean KL 0.0626 (llama.cpp 97.80%, 0.0610); at bf16 both engines sit about 0.045 of mean KL from the publisher, a shared reproduction gap not yet explained.
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [docs/model-evidence.md](model-evidence.md).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `64cdc0dc6bab`, output sha256 `b26e284a3452dbba`, [pin](../tests/compatibility/tier-smoke/trinity-nano-preview-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** [n/a](../MANUAL.md#support-matrix): afmoe (gated attention with sparse MoE) has no CUDA path; CUDA refuses it and the file runs on the CPU.
- **Metal:** [n/a](../MANUAL.md#support-matrix): afmoe (gated attention with sparse MoE) has no Metal path; Metal refuses it and the file runs on the CPU.
- **LoRA training:** no: MoE router gating other than softmax (runner 1.7.1, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-30).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

### Qwen3

#### Qwen3-30B-A3B selective precision

<a id="qwen3-30b-a3b-expq4_0-attnq8_0"></a>
##### `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf`

Qwen3-30B-A3B with attention Q8_0 and experts Q4_0, text path: it passes the fidelity bar against the first-party Q8_0 where the official uniform Q4_K_M fails it.

- **File:** `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf` (sha256 `df02efa815a74e6207f9e01f2d38272c1e2718bb23aec544709a02a2c7d5b206`, 17,988,417,024 bytes, from [Joakimpalm-Zen/Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF) at revision `6cdb710bcd55`).
- **Upstream checkpoint:** Qwen/Qwen3-30B-A3B.
- **Quantization:** Q8_0 attention / Q4_0 experts.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 760 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. A float32 reference of the 30B (120 GB) is not run; the bar is against the byte-verified first-party Q8_0 (v2 top-1 99.50%, mean KLD 0.0345).
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [tests/compatibility/models.json](../tests/compatibility/models.json), [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `9c80990c12a550f6`, [pin](../tests/compatibility/tier-smoke/qwen3-30b-a3b-expq4_0-attnq8_0.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Qwen3-4B

<a id="qwen3-4b-q4_k_m-bartowski"></a>
##### `Qwen_Qwen3-4B-Q4_K_M.gguf`

The base the published tool-use adapter was trained through and runs on, named by sha256 on the adapter's card: maintained on report, and run on every release only as the adapter's base in its composed smoke.

- **File:** `Qwen_Qwen3-4B-Q4_K_M.gguf` (sha256 `fbe1d5edd4ce802ae3ae7c7e4ab7d09789d697fdac1fc7929f8df4ca3c41bae3`, 2,497,280,960 bytes, from [bartowski/Qwen_Qwen3-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF) at revision `cb76885dc66d`).
- **Upstream checkpoint:** Qwen/Qwen3-4B at revision `1cfa9a720891`.
- **Quantization:** Q4_K_M (bartowski).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: Qwen3-4B bf16 against Qwen's float32 reference (1.5.0 golden pass). This file: load, publisher tokenizer agreement and a chat answer at temperature 0 on the Metal laptop; no fidelity number of its own.
- **Context tested:** 4096.
- **Known limitations:** 2.33 GiB of weights page from disk on an 8 GB machine with other applications open; the validation run recorded that state.
- **Still missing:** CUDA cell; a fidelity measurement of this file against the BF16 parent.
- **Evidence:** [docs/compat-reports/1.7.0-2026-10-28-macos-arm64-qwen3-4b-q4_k_m-bartowski.json](compat-reports/1.7.0-2026-10-28-macos-arm64-qwen3-4b-q4_k_m-bartowski.json), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, metal backend; last run 1.7.0 on 2026-10-28.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `95be9f33ca7dbafb`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-q4_k_m-bartowski.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `95be9f33ca7dbafb`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-q4_k_m-bartowski.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.7.0 2026-10-28 on macos-arm64-metal at commit `b31730a9dc45`, output sha256 `95be9f33ca7dbafb`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-q4_k_m-bartowski.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

#### Qwen3-4B Runner ToolUse

<a id="qwen3-4b-runner-tooluse-merged-q8_0"></a>
##### `merged-Q8_0.gguf`

Qwen3-4B with the tool-use LoRA merged and written at Q8_0, text path: a qwen3 client loop passes on CUDA; it is an adaptation demonstration, not a general improvement over Qwen3-4B.

- **File:** `merged-Q8_0.gguf` (sha256 `b9d21e525d91b20a1fc3437cf34c7ecfa03afda722d960b95bcfd113f2b7ea02`, 4,280,405,440 bytes, from [Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M) at revision `5902b309ee51`).
- **Upstream checkpoint:** Qwen/Qwen3-4B.
- **Quantization:** Q8_0.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 639 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Family: Qwen3-4B bf16 against Qwen's float32 reference, 100% margin-qualified top-1 (1.5.0 golden pass; its served row is Qwen3-4B Q4_K_M, not this file).
- **Context tested:** not recorded.
- **Still missing:** Pinned output.
- **Evidence:** [docs/golden-pass-2026-10-22.md](golden-pass-2026-10-22.md), [docs/adaptation-engine.md](adaptation-engine.md).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `aa1a1647a3e4b534`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-runner-tooluse-merged-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `aa1a1647a3e4b534`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-runner-tooluse-merged-q8_0.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

<a id="qwen3-4b-runner-tooluse-lora"></a>
##### `Qwen3-4B-ToolUse-LoRA.gguf`

The tool-use LoRA adapter for Qwen3-4B, served with --lora on the base its card names (bartowski's Q4_K_M): an adapter, not a model.

- **File:** `Qwen3-4B-ToolUse-LoRA.gguf` (sha256 `ea38f80c33d381c9aa62c874920d2fa058dd307e7186e9b7c347d782f35b88e4`, 66,094,752 bytes, from [Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M) at revision `5902b309ee51`).
- **Upstream checkpoint:** Qwen/Qwen3-4B.
- **Quantization:** adapter (F32/F16 LoRA tensors).
- **Runs with:** [Qwen3-4B](#qwen3-4b-q4_k_m-bartowski), loaded beside it with `--lora`; the smoke and its pin are of that composed run.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 639 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** adapters. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** The base model's anchor: Qwen3-4B bf16 against Qwen's float32 reference (1.5.0 golden pass). The adapter itself has no float32 reference. The composed run's pinned reply on the Metal laptop (1.7.0, 2026-10-28) is byte-identical to the CPU reply recorded on another host with 1.5.1.
- **Context tested:** 512.
- **Still missing:** nothing the audit named
- **Evidence:** [docs/adaptation-engine.md](adaptation-engine.md), [docs/lora.md](lora.md), [tests/compatibility/tier-smoke/qwen3-4b-runner-tooluse-lora.macos-arm64-metal.metal.txt](../tests/compatibility/tier-smoke/qwen3-4b-runner-tooluse-lora.macos-arm64-metal.metal.txt).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `9a878ce06241ebaa`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-runner-tooluse-lora.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `9a878ce06241ebaa`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-runner-tooluse-lora.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.7.0 2026-10-28 on macos-arm64-metal at commit `b31730a9dc45`, output sha256 `9a878ce06241ebaa`, [pin](../tests/compatibility/tier-smoke/qwen3-4b-runner-tooluse-lora.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

#### Qwen3-Coder-30B-A3B keep-120

<a id="qwen3-coder-30b-a3b-instruct-keep120-q4_k_m"></a>
##### `Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf`

An expert-pruned Qwen3-Coder-30B-A3B (120 of 128 experts kept) at Q4_K_M, text path: it passes the original and the current fidelity bar against its parent.

- **File:** `Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf` (sha256 `3ac8568224a15ea49db6affdd195ba48e8b2d7c6325c7849b9829381e89163a1`, 17,531,967,808 bytes, from [Joakimpalm-Zen/Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M-GGUF](https://huggingface.co/Joakimpalm-Zen/Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M-GGUF) at revision `3fd968c75417`).
- **Upstream checkpoint:** Qwen/Qwen3-Coder-30B-A3B-Instruct.
- **Quantization:** Q4_K_M, 120 experts.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 813 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine, not fidelity. A float32 reference of the 30B is not run; the bars are against the unpruned parent.
- **Context tested:** not recorded.
- **Still missing:** Pinned output; a tool gate on this file (the recorded qwen3-coder tool run is on the unpruned Q4_K_M).
- **Evidence:** [MANUAL.md#published-artifacts](../MANUAL.md#published-artifacts).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.5.1 on 2026-10-24.
- **CPU:** ☐ not run
- **CUDA:** ✅ v1.5.1 2026-10-24 on linux-x86_64-cuda (borrowed hardware) at commit `a3a741236eb0`, output sha256 `f049d30f2e67e4a9`, [pin](../tests/compatibility/tier-smoke/qwen3-coder-30b-a3b-instruct-keep120-q4_k_m.linux-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.5.1, 2026-10-24).
- **LoRA adapter serving:** supported (runner 1.5.1, 2026-10-24).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

### NVIDIA Nemotron Nano 9B v2

#### NVIDIA Nemotron Nano 9B v2

<a id="nvidia-nemotron-nano-9b-v2-q8_0"></a>
##### `NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf`

A plain Q8_0 of NVIDIA's Nemotron-Nano-9B-v2 (Mamba-2 hybrid), text path: 100% margin-qualified top-1 against NVIDIA's float32 reference at Q8_0 over 100 positions; the 2,000-position rerun did not finish.

- **File:** `NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf` (sha256 `d3224286484df1ae79c2b74d2a0913e1bbb5a364b0dd7c1061c95fa97ee608d7`, 9,458,093,632 bytes, from [Joakimpalm-Zen/NVIDIA-Nemotron-Nano-9B-v2-Q8_0-GGUF](https://huggingface.co/Joakimpalm-Zen/NVIDIA-Nemotron-Nano-9B-v2-Q8_0-GGUF) at revision `3b6a1cf5035f`).
- **Upstream checkpoint:** nvidia/NVIDIA-Nemotron-Nano-9B-v2.
- **Quantization:** Q8_0.
- **Published below Tier 1:** Policy decision 2026-10-27: Tier 1 follows pull; 496 downloads in 30 days, under the 1,000 bar.
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** NVIDIA's float32 reference (golden pass 2026-09-07, 100 positions): margin-qualified top-1 100% at bf16 and Q8_0; Runner's mean KL 0.0041 at Q8_0 against llama.cpp's 0.0054. CPU and CUDA greedy byte-identical on this file.
- **Context tested:** not recorded.
- **Still missing:** The tool gate (native protocol since 2026-10-09); pinned output; the 2,000-position rerun.
- **Evidence:** [docs/golden-pass-2026-09-07.md](golden-pass-2026-09-07.md).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cuda backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `e23a8c77690649fb`, [pin](../tests/compatibility/tier-smoke/nvidia-nemotron-nano-9b-v2-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `d07e80b85e6d`, output sha256 `e23a8c77690649fb`, [pin](../tests/compatibility/tier-smoke/nvidia-nemotron-nano-9b-v2-q8_0.windows-x86_64-cuda.cuda.txt)
- **Metal:** [n/a](../MANUAL.md#support-matrix): Metal has no SSM path; a Metal build runs this file on the CPU.
- **LoRA training:** no: recurrent architecture (Mamba-2) (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** no: --lora does not cover Mamba-2 hybrid architectures yet (nemotron_h) (runner 1.7.1, 2026-10-29).
- **Tier history:** Tier 1, actively validated from 2026-10-22; Tier 2, maintained on report from 2026-10-27 ([log](#transition-log)).

### gpt-oss

#### gpt-oss-20b keep-30

<a id="gpt-oss-20b-keep30-mxfp4"></a>
##### `gpt-oss-20b-keep30-MXFP4.gguf`

A 32-to-30-expert derivative of gpt-oss-20b that does not pass the current fidelity bar; its card leads with the measured status and it stays published as a disclosed near-miss.

- **File:** `gpt-oss-20b-keep30-MXFP4.gguf` (sha256 `e6b884b3de28a78858097bbee25c9c503975992a2bbc8943e0ef2c4bbbc14312`, 11,472,833,184 bytes, from [Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF](https://huggingface.co/Joakimpalm-Zen/gpt-oss-20b-keep30-MXFP4-GGUF) at revision `d6bfb80ce279`).
- **Published below Tier 1:** Policy decision 2026-10-22: a published file may be Tier 2 when its card already leads with its measured shortfall.
- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Llama

#### Llama-style dense families (llama)

<a id="arch-llama"></a>
##### Llama-style dense families (llama)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Mistral

#### Mistral (mistral)

<a id="arch-mistral"></a>
##### Mistral (mistral)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### SmolLM

#### SmolLM (smollm)

<a id="arch-smollm"></a>
##### SmolLM (smollm)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### StableLM

#### StableLM (stablelm)

<a id="arch-stablelm"></a>
##### StableLM (stablelm)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Qwen2

#### Qwen2 (qwen2)

<a id="arch-qwen2"></a>
##### Qwen2 (qwen2)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Gemma 3

#### Gemma 3 (gemma3)

<a id="arch-gemma3"></a>
##### Gemma 3 (gemma3)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Phi-3

#### Phi-3 (phi3)

<a id="arch-phi3"></a>
##### Phi-3 (phi3)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Apertus

#### Apertus (apertus)

<a id="arch-apertus"></a>
##### Apertus (apertus)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### Granite 4

#### Granite 4 h-series hybrid (granitehybrid)

<a id="arch-granitehybrid"></a>
##### Granite 4 h-series hybrid (granitehybrid)

Admitted and maintained on report: existing regression tests keep running and shared-path changes are tested with representative coverage; no file of this architecture is re-validated on every release.

- **Tier history:** Tier 2, maintained on report from 2026-10-22 ([log](#transition-log)).

### MiniCPM5

#### MiniCPM5-2B (openbmb)

<a id="minicpm5-2b-q8_0"></a>
##### `MiniCPM5-2B-Q8_0.gguf`

Text path, the publisher's Q8_0, the fidelity file of the two: against the publisher's BF16 parent computed in float32 it reads 0.0033 mean KLD and 98.0% top-1 (300 positions, Metal).

- **File:** `MiniCPM5-2B-Q8_0.gguf` (sha256 `c5415f8989bf88a8288f1b55a3cc371af53c07b0faa220a63bd7a990cfaba078`, 2,679,710,688 bytes, from [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) at revision `2079a22f3bea`).
- **Upstream checkpoint:** openbmb/MiniCPM5-2B at revision `f97400052a43`.
- **Quantization:** Q8_0 (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against openbmb's BF16 parent computed in float32 on the CPU (transformers' own llama code, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from Metal score files): Q8_0 0.0033 mean KLD / 98.0% top-1 / 100% margin-qualified, Q4_K_M 0.102 / 86.0% / 94.6%; Q4_K_M against Q8_0 from the same score files 0.098 / 85.3%. Tokenizer 0/721 against the publisher's tokenizer.json at f9740005, template conformance 23/23 in text and tokens. Record: docs/compat-reports/minicpm5-2b-reference-2026-10-29/README.md.
- **Context tested:** 4096.
- **Known limitations:** 2.68 GB of weights page from disk on an 8 GB machine with other applications open. No llama.cpp column. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The Windows CUDA host's CPU and CUDA cells and the CPU-vs-CUDA check; a llama.cpp column against the same reference (not run); task quality and the native tool protocol on the real checkpoint (the parser is pinned on fixtures only).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-29-macos-arm64-minicpm5-2b-q8_0.json](compat-reports/1.7.1-2026-10-29-macos-arm64-minicpm5-2b-q8_0.json), [docs/compat-reports/minicpm5-2b-reference-2026-10-29/README.md](compat-reports/minicpm5-2b-reference-2026-10-29/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, metal backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `771924318c51`, output sha256 `d163df7b65c07bae`, [pin](../tests/compatibility/tier-smoke/minicpm5-2b-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `771924318c51`, output sha256 `d163df7b65c07bae`, [pin](../tests/compatibility/tier-smoke/minicpm5-2b-q8_0.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.7.1 2026-10-29 on macos-arm64-metal at commit `a83670a85fdc`, output sha256 `d163df7b65c07bae`, [pin](../tests/compatibility/tier-smoke/minicpm5-2b-q8_0.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Upcoming, planned admission from 2026-10-29; Tier 2, maintained on report from 2026-10-29 ([log](#transition-log)).

<a id="minicpm5-2b-q4_k_m"></a>
##### `MiniCPM5-2B-Q4_K_M.gguf`

Text path, the publisher's Q4_K_M, which fails the fidelity bar: against the publisher's BF16 parent computed in float32 it reads 0.102 mean KLD and 86.0% top-1 (300 positions, Metal), against a 0.05 bar; the Q8_0 is the fidelity file.

- **File:** `MiniCPM5-2B-Q4_K_M.gguf` (sha256 `ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd`, 1,561,318,368 bytes, from [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) at revision `2079a22f3bea`).
- **Upstream checkpoint:** openbmb/MiniCPM5-2B at revision `f97400052a43`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against openbmb's BF16 parent computed in float32 on the CPU (transformers' own llama code, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from Metal score files): Q8_0 0.0033 mean KLD / 98.0% top-1 / 100% margin-qualified, Q4_K_M 0.102 / 86.0% / 94.6%; Q4_K_M against Q8_0 from the same score files 0.098 / 85.3%. Tokenizer 0/721 against the publisher's tokenizer.json at f9740005, template conformance 23/23 in text and tokens. Record: docs/compat-reports/minicpm5-2b-reference-2026-10-29/README.md.
- **Context tested:** 4096.
- **Known limitations:** No fidelity claim: the file fails the 0.05 bar against its parent on Runner, and the gap is the file's quantization, not the arithmetic (the Q8_0 of the same parent reads 0.0033 on the same build). No llama.cpp column. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The Windows CUDA host's CPU and CUDA cells and the CPU-vs-CUDA check; a llama.cpp column against the same reference (not run); task quality and the native tool protocol on the real checkpoint (the parser is pinned on fixtures only).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-29-macos-arm64-minicpm5-2b-q4_k_m.json](compat-reports/1.7.1-2026-10-29-macos-arm64-minicpm5-2b-q4_k_m.json), [docs/compat-reports/minicpm5-2b-reference-2026-10-29/README.md](compat-reports/minicpm5-2b-reference-2026-10-29/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, metal backend; last run 1.7.1 on 2026-10-29.
- **CPU:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `771924318c51`, output sha256 `4ed9b06a2b3a14c9`, [pin](../tests/compatibility/tier-smoke/minicpm5-2b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-29 on windows-x86_64-cuda at commit `771924318c51`, output sha256 `16ee8166ecb8972e`, [pin](../tests/compatibility/tier-smoke/minicpm5-2b-q4_k_m.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.7.1 2026-10-29 on macos-arm64-metal at commit `a83670a85fdc`, output sha256 `16ee8166ecb8972e`, [pin](../tests/compatibility/tier-smoke/minicpm5-2b-q4_k_m.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.7.1, 2026-10-29).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-29).
- **Tier history:** Upcoming, planned admission from 2026-10-29; Tier 2, maintained on report from 2026-10-29 ([log](#transition-log)).

### LFM2.5

#### LFM2.5-1.2B-Instruct (LiquidAI)

<a id="lfm2.5-1.2b-instruct-q8_0"></a>
##### `LFM2.5-1.2B-Instruct-Q8_0.gguf`

Text path, LiquidAI's own Q8_0 of LFM2.5-1.2B-Instruct, the fidelity file of the publisher's two: against the BF16 parent computed in float32 it reads 0.0042 mean KLD and 99.0% top-1 (300 positions, CPU).

- **File:** `LFM2.5-1.2B-Instruct-Q8_0.gguf` (sha256 `f6b981dcb86917fa463f78a362320bd5e2dc45445df147287eedb85e5a30d26a`, 1,246,253,888 bytes, from [LiquidAI/LFM2.5-1.2B-Instruct-GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) at revision `8ed288026e23`).
- **Upstream checkpoint:** LiquidAI/LFM2.5-1.2B-Instruct at revision `0f604ada3f76`.
- **Quantization:** Q8_0 (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against LiquidAI's BF16 parent computed in float32 on the CPU (transformers 5.22, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from CPU score files at d66f0f7a, the pre-rebase form of 67e75e8e): Q8_0 0.0042 mean KLD / 99.0% top-1 / 100% margin-qualified, Q4_K_M 0.115 / 85.7% / 94.7%; llama.cpp (3bb63cb) on the same Q4_K_M against the same reference 0.119 / 84.3%. Tokenizer 0/721 against the publisher's tokenizer.json at 0f604ada, template conformance 23/23 in text and tokens against the publisher's chat_template.jinja. Record: docs/compat-reports/lfm2.5-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: the GPU backends refuse lfm2 by name and serve it on the CPU (its short-convolution layers are not checked there yet). A required or named tool choice takes the generic JSON envelope on this family. Licence: LFM Open License v1.0, which limits commercial use to entities with under US$10M annual revenue; the limit binds whoever runs the model.
- **Still missing:** A GPU path; task quality and the native Python-call tool protocol on the real checkpoint (the parser is pinned on fixtures only).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-1.2b-instruct-q8_0.json](compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-1.2b-instruct-q8_0.json), [docs/compat-reports/lfm2.5-reference-2026-10-30/README.md](compat-reports/lfm2.5-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, cpu backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `64cdc0dc6bab`, output sha256 `587d226ab6e1654f`, [pin](../tests/compatibility/tier-smoke/lfm2.5-1.2b-instruct-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** [n/a](../MANUAL.md#support-matrix): lfm2 (short-convolution layers interleaved with GQA attention) has no validated CUDA path; CUDA refuses it by name and the file runs on the CPU.
- **Metal:** ☐ not run
- **LoRA training:** no: recurrent architecture (short convolution) (runner 1.7.1, 2026-10-30).
- **LoRA adapter serving:** no: --lora does not cover the short-convolution hybrid yet (lfm2) (runner 1.7.1, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

<a id="lfm2.5-1.2b-instruct-q4_k_m"></a>
##### `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`

Text path, LiquidAI's own Q4_K_M of LFM2.5-1.2B-Instruct, which fails the fidelity bar: against the BF16 parent computed in float32 it reads 0.115 mean KLD against 0.05 (85.7% top-1, 94.7% margin-qualified against 97%), and llama.cpp on the same file reads 0.119, so the gap is the file's quantization. The Q8_0 is the fidelity file.

- **File:** `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` (sha256 `b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5`, 730,895,168 bytes, from [LiquidAI/LFM2.5-1.2B-Instruct-GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) at revision `8ed288026e23`).
- **Upstream checkpoint:** LiquidAI/LFM2.5-1.2B-Instruct at revision `0f604ada3f76`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against LiquidAI's BF16 parent computed in float32 on the CPU (transformers 5.22, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from CPU score files at d66f0f7a, the pre-rebase form of 67e75e8e): Q8_0 0.0042 mean KLD / 99.0% top-1 / 100% margin-qualified, Q4_K_M 0.115 / 85.7% / 94.7%; llama.cpp (3bb63cb) on the same Q4_K_M against the same reference 0.119 / 84.3%. Tokenizer 0/721 against the publisher's tokenizer.json at 0f604ada, template conformance 23/23 in text and tokens against the publisher's chat_template.jinja. Record: docs/compat-reports/lfm2.5-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: the GPU backends refuse lfm2 by name and serve it on the CPU (its short-convolution layers are not checked there yet). A required or named tool choice takes the generic JSON envelope on this family. Licence: LFM Open License v1.0, which limits commercial use to entities with under US$10M annual revenue; the limit binds whoever runs the model.
- **Still missing:** A GPU path; task quality and the native Python-call tool protocol on the real checkpoint (the parser is pinned on fixtures only). The publisher's QAD-Q4_0 fails the bar too (0.128) and is not a row.
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-1.2b-instruct-q4_k_m.json](compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-1.2b-instruct-q4_k_m.json), [docs/compat-reports/lfm2.5-reference-2026-10-30/README.md](compat-reports/lfm2.5-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, cpu backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `64cdc0dc6bab`, output sha256 `3bb0f080e0b57ed8`, [pin](../tests/compatibility/tier-smoke/lfm2.5-1.2b-instruct-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** [n/a](../MANUAL.md#support-matrix): lfm2 (short-convolution layers interleaved with GQA attention) has no validated CUDA path; CUDA refuses it by name and the file runs on the CPU.
- **Metal:** ☐ not run
- **LoRA training:** no: recurrent architecture (short convolution) (runner 1.7.1, 2026-10-30).
- **LoRA adapter serving:** no: --lora does not cover the short-convolution hybrid yet (lfm2) (runner 1.7.1, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

#### LFM2.5-350M (LiquidAI)

<a id="lfm2.5-350m-q8_0"></a>
##### `LFM2.5-350M-Q8_0.gguf`

Text path, LiquidAI's own Q8_0 of LFM2.5-350M, the fidelity file of the publisher's two: against the BF16 parent computed in float32 it reads 0.0081 mean KLD and 95.3% top-1 (300 positions, CPU).

- **File:** `LFM2.5-350M-Q8_0.gguf` (sha256 `be036a757295e550098b85e13f6af2735d0fa73b41e1156a40c7d8e8e32a5766`, 379,217,632 bytes, from [LiquidAI/LFM2.5-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF) at revision `657e078c9408`).
- **Upstream checkpoint:** LiquidAI/LFM2.5-350M at revision `9e6c6ccf47cd`.
- **Quantization:** Q8_0 (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against LiquidAI's BF16 parent computed in float32 on the CPU (transformers 5.22, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from CPU score files at d66f0f7a, the pre-rebase form of 67e75e8e): Q8_0 0.0081 mean KLD / 95.3% top-1 / 100% margin-qualified, Q4_K_M 0.874 / 54.3% / 61.9%; llama.cpp (3bb63cb) on the same Q4_K_M against the same reference 0.924 / 53.0%. Tokenizer 0/721 against the publisher's tokenizer.json, template conformance 23/23 in text and tokens against the publisher's chat_template.jinja (the 350M's is byte-identical to the 1.2B-Instruct's). Record: docs/compat-reports/lfm2.5-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: the GPU backends refuse lfm2 by name and serve it on the CPU (its short-convolution layers are not checked there yet). A required or named tool choice takes the generic JSON envelope on this family. Licence: LFM Open License v1.0, which limits commercial use to entities with under US$10M annual revenue; the limit binds whoever runs the model.
- **Still missing:** The Windows CUDA host's CPU cell; a GPU path; task quality and the native Python-call tool protocol on the real checkpoint (the parser is pinned on fixtures only).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-350m-q8_0.json](compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-350m-q8_0.json), [docs/compat-reports/lfm2.5-reference-2026-10-30/README.md](compat-reports/lfm2.5-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, cpu backend; last run 1.8.0 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `a3dd7b514a3e7cb8`, [pin](../tests/compatibility/tier-smoke/lfm2.5-350m-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** no: recurrent architecture (short convolution) (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** no: --lora does not cover the short-convolution hybrid yet (lfm2) (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

<a id="lfm2.5-350m-q4_k_m"></a>
##### `LFM2.5-350M-Q4_K_M.gguf`

Text path, LiquidAI's own Q4_K_M of LFM2.5-350M, which fails the fidelity bar: against the BF16 parent computed in float32 it reads 0.874 mean KLD against 0.05 (54.3% top-1, 61.9% margin-qualified against 97%), and llama.cpp on the same file reads 0.924, so the gap is the file's quantization. The Q8_0 is the fidelity file.

- **File:** `LFM2.5-350M-Q4_K_M.gguf` (sha256 `7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4`, 229,312,224 bytes, from [LiquidAI/LFM2.5-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF) at revision `657e078c9408`).
- **Upstream checkpoint:** LiquidAI/LFM2.5-350M at revision `9e6c6ccf47cd`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against LiquidAI's BF16 parent computed in float32 on the CPU (transformers 5.22, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from CPU score files at d66f0f7a, the pre-rebase form of 67e75e8e): Q8_0 0.0081 mean KLD / 95.3% top-1 / 100% margin-qualified, Q4_K_M 0.874 / 54.3% / 61.9%; llama.cpp (3bb63cb) on the same Q4_K_M against the same reference 0.924 / 53.0%. Tokenizer 0/721 against the publisher's tokenizer.json, template conformance 23/23 in text and tokens against the publisher's chat_template.jinja (the 350M's is byte-identical to the 1.2B-Instruct's). Record: docs/compat-reports/lfm2.5-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** Asked for the planet closest to the Sun it answers Earth, where the Q8_0 answers Mercury: the quantization's error reaches facts. CPU only: the GPU backends refuse lfm2 by name and serve it on the CPU (its short-convolution layers are not checked there yet). A required or named tool choice takes the generic JSON envelope on this family. Licence: LFM Open License v1.0, which limits commercial use to entities with under US$10M annual revenue; the limit binds whoever runs the model.
- **Still missing:** The Windows CUDA host's CPU cell; a GPU path; task quality and the native Python-call tool protocol on the real checkpoint (the parser is pinned on fixtures only). The publisher's QAD-Q4_0 fails the bar too (0.483) and is not a row.
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-350m-q4_k_m.json](compat-reports/1.7.1-2026-10-30-macos-arm64-lfm2.5-350m-q4_k_m.json), [docs/compat-reports/lfm2.5-reference-2026-10-30/README.md](compat-reports/lfm2.5-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, cpu backend; last run 1.8.0 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `a3dd7b514a3e7cb8`, [pin](../tests/compatibility/tier-smoke/lfm2.5-350m-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** no: recurrent architecture (short convolution) (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** no: --lora does not cover the short-convolution hybrid yet (lfm2) (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

### MiMo

#### MiMo-V2.6-Distill-Qwen-9B (XiaomiMiMo)

<a id="mimo-v2.6-distill-qwen-9b-q4_k_m"></a>
##### `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf`

Text path, bartowski's Q4_K_M of XiaomiMiMo's Qwen3.5-9B distill, which fails the fidelity bar: against the BF16 checkpoint computed in float32 it reads 0.0717 mean KLD against 0.05 (91.7% top-1, 98.2% margin-qualified, inside its 97%), and 0.064 against its own Q8_0. The same build reads the Q8_0 at 0.0075, the fidelity file. It loads, tokenizes as the publisher's tokenizer.json does, renders the publisher's own template token for token and answers through it on the CPU.

- **File:** `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf` (sha256 `4bca6f18c73f72270c7a20c2ea2bea581de8246e318714277120369d34048c81`, 5,841,049,120 bytes, from [bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF](https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF) at revision `4371da10c84f`).
- **Upstream checkpoint:** XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B at revision `1f09ddcc1a0e`.
- **Quantization:** Q4_K_M (bartowski).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** This file: load, publisher tokenizer agreement (0 of 721 corpus strings differ from tokenizer.json at 1f09ddcc, read with the qwen2 split the publisher's tokenizer uses), template conformance 23 of 23 in text and tokens against the publisher's chat_template.jinja (the GGUF carries it byte for byte), and a chat answer at temperature 0 on the Metal laptop's CPU. Both bartowski files against XiaomiMiMo's BF16 checkpoint computed in float32 on the lab CPU lane (transformers 5.16.1, gold-logits.py at fcd461ef, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from CPU score files at f5b25f0a): Q4_K_M 0.0717 mean KLD / 91.7% top-1 / 98.2% margin-qualified, FAIL on the mean; Q8_0 0.0075 / 99.0% / 100%. Quant bar against the Q8_0 from the same score files: 0.064 mean KLD (median 0.022), 92.0% top-1, 97.7% margin-qualified, FAIL on the mean. No cross-engine number. Record: docs/compat-reports/mimo-v2.6-distill-qwen-9b-2026-10-29/README.md.
- **Context tested:** 4096.
- **Known limitations:** No Metal cell: Runner's qwen35 hybrid path runs on the CPU and CUDA only. 5.84 GB of weights page from disk on an 8 GB machine (the CPU smoke took 13 minutes there). No vision: the mmproj file is not loaded. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** A Q8_0 row (the fidelity file, measured here, with no device record yet).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-29-macos-arm64-mimo-v2.6-distill-qwen-9b-q4_k_m.json](compat-reports/1.7.1-2026-10-29-macos-arm64-mimo-v2.6-distill-qwen-9b-q4_k_m.json), [docs/compat-reports/mimo-v2.6-distill-qwen-9b-2026-10-29/README.md](compat-reports/mimo-v2.6-distill-qwen-9b-2026-10-29/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, cpu backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `80c30df3d188`, output sha256 `eea4d971ebfaa42d`, [pin](../tests/compatibility/tier-smoke/mimo-v2.6-distill-qwen-9b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.7.1 2026-10-30 on windows-x86_64-cuda at commit `80c30df3d188`, output sha256 `eea4d971ebfaa42d`, [pin](../tests/compatibility/tier-smoke/mimo-v2.6-distill-qwen-9b-q4_k_m.windows-x86_64-cuda.cuda.txt)
- **Metal:** ☐ not run
- **LoRA training:** supported (runner 1.7.1, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.7.1, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-29; Tier 2, maintained on report from 2026-10-29 ([log](#transition-log)).

### Spark

#### Spark-X2.5-4B (XHToken)

<a id="spark-x2.5-4b-q4_k_m"></a>
##### `Spark-X2.5-4B-Q4_K_M.gguf`

Text path, XHToken's own Q4_K_M of Spark X2.5 4B, which fails the fidelity bar: against the BF16 checkpoint computed in float32 it reads 0.1136 mean KLD against 0.05 (87.0% top-1, 95.6% margin-qualified, under its 97%). The same build reads XHToken's Q8_0 at 0.0044, the fidelity file. It loads, tokenizes as the publisher's tokenizer.json does, renders the publisher's own template token for token and answers through it on the CPU.

- **File:** `Spark-X2.5-4B-Q4_K_M.gguf` (sha256 `adfcfa19a4ed6a5985da8bf565fe15f8e1a7e131d79bae2d19d48d1c40109428`, 2,600,224,352 bytes, from [XHToken/Spark-X2.5-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF) at revision `9826e0be84e6`).
- **Upstream checkpoint:** XHToken/Spark-X2.5-4B at revision `0bcb35678590`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** This file: load, publisher tokenizer agreement (0 of 721 corpus strings differ from tokenizer.json, identical across X2.5 and X2.6), template conformance 23 of 23 in text and tokens against the publisher's chat_template.jinja, and a chat answer at temperature 0 on the Metal laptop's CPU. The architecture is checked against llama.cpp on the X2.6 1.7B Q8_0 (the family's smallest member): 0.0065 mean KLD, 96.0% top-1, 100% margin-qualified over 200 raw-completion positions on the CPU; agreement with another engine, not fidelity. Both XHToken files (the GGUF repository at 9826e0be) against XHToken's BF16 checkpoint at 0bcb3567 computed in float32 on the lab CPU lane (the publisher's modeling code, transformers 4.57.1, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from CPU score files): Q4_K_M 0.1136 mean KLD / 87.0% top-1 / 95.6% margin-qualified, FAIL; Q8_0 0.0044 / 98.7% / 100%. Record: docs/compat-reports/spark-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: the per-head attention gate and the exact GELU have no CUDA or Metal kernels yet, so a GPU run falls back to the CPU by name. LoRA training is refused (the per-head gate is not in the backward). A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The CUDA and Metal paths (the next change); the Windows CUDA host's CPU cell; a Q8_0 row (the fidelity file, measured here, with no device record yet).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-spark-x2.5-4b-q4_k_m.json](compat-reports/1.7.1-2026-10-30-macos-arm64-spark-x2.5-4b-q4_k_m.json), [docs/compat-reports/spark-2026-10-30/README.md](compat-reports/spark-2026-10-30/README.md), [docs/compat-reports/spark-2026-10-30/gold-logits-cpu-x2.5-q8-q4.json](compat-reports/spark-2026-10-30/gold-logits-cpu-x2.5-q8-q4.json), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, cpu backend; last run 1.8.0 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `b872096a90ef40b7`, [pin](../tests/compatibility/tier-smoke/spark-x2.5-4b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** no: per-head attention output gate (spark2_5) not in the backward (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

#### Spark-X2.6-4B (XHToken)

<a id="spark-x2.6-4b-q4_k_m"></a>
##### `Spark-X2.6-4B-Q4_K_M.gguf`

Text path, XHToken's own Q4_K_M of Spark X2.6 4B, which fails the fidelity bar: against the BF16 checkpoint computed in float32 it reads 0.113 mean KLD against 0.05 (87.0% top-1, 93.6% margin-qualified, under its 97%). The same build reads XHToken's Q8_0 at 0.0132, the fidelity file. It loads, tokenizes as the publisher's tokenizer.json does, renders the publisher's own template token for token and answers through it on the CPU.

- **File:** `Spark-X2.6-4B-Q4_K_M.gguf` (sha256 `91cb2811aebb47ac916aa619d27d4cdd7f4143106b7917accb8ead4cd7f5cbd9`, 3,543,441,312 bytes, from [XHToken/Spark-X2.6-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF) at revision `0e8aa39a3406`).
- **Upstream checkpoint:** XHToken/Spark-X2.6-4B at revision `d1b5f2607110`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** This file: load, publisher tokenizer agreement (0 of 721 corpus strings differ from tokenizer.json, identical across X2.5 and X2.6), template conformance 23 of 23 in text and tokens against the publisher's chat_template.jinja, and a chat answer at temperature 0 on the Metal laptop's CPU. The architecture is checked against llama.cpp on the X2.6 1.7B Q8_0 (the family's smallest member): 0.0065 mean KLD, 96.0% top-1, 100% margin-qualified over 200 raw-completion positions on the CPU; agreement with another engine, not fidelity. Both XHToken files (the GGUF repository at 0e8aa39a) against XHToken's BF16 checkpoint at d1b5f260 computed in float32 on the lab CPU lane (the publisher's modeling code, transformers 4.57.1, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from CPU score files): Q4_K_M 0.113 mean KLD / 87.0% top-1 / 93.6% margin-qualified, FAIL; Q8_0 0.0132 / 98.7% / 100%. Record: docs/compat-reports/spark-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: the per-head attention gate and the exact GELU have no CUDA or Metal kernels yet, so a GPU run falls back to the CPU by name. LoRA training is refused (the per-head gate is not in the backward). A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The CUDA and Metal paths (the next change); the Windows CUDA host's CPU cell; a Q8_0 row (the fidelity file, measured here, with no device record yet).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-spark-x2.6-4b-q4_k_m.json](compat-reports/1.7.1-2026-10-30-macos-arm64-spark-x2.6-4b-q4_k_m.json), [docs/compat-reports/spark-2026-10-30/README.md](compat-reports/spark-2026-10-30/README.md), [docs/compat-reports/spark-2026-10-30/gold-logits-cpu-x2.6-q8-q4.json](compat-reports/spark-2026-10-30/gold-logits-cpu-x2.6-q8-q4.json), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on macos-arm64-metal, cpu backend; last run 1.8.0 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `42a4c173422d4070`, [pin](../tests/compatibility/tier-smoke/spark-x2.6-4b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** no: per-head attention output gate (spark2_5) not in the backward (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

### K2-Horizon

#### K2-Horizon-0.9B (IFM)

<a id="k2-horizon-0.9b-q8_0"></a>
##### `K2-Horizon-1B-Q8_0.gguf`

Text path, IFM's own Q8_0 of K2-Horizon 0.9B, the fidelity file of the publisher's two: against the BF16 parent computed in float32 it reads 0.0037 mean KLD and 97.3% top-1 (300 positions, Metal).

- **File:** `K2-Horizon-1B-Q8_0.gguf` (sha256 `741fb9ad263956c003883cb7caddeb14b4dfb9b4b67b5414ad12967e965dd547`, 1,148,614,016 bytes, from [IFM/K2-Horizon-0.9B-GGUF](https://huggingface.co/IFM/K2-Horizon-0.9B-GGUF) at revision `118965081cf1`).
- **Upstream checkpoint:** IFM/K2-Horizon-0.9B at revision `d50881f4f8dc`.
- **Quantization:** Q8_0 (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against IFM's BF16 parent computed in float32 on the CPU (the publisher's own modeling code, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from Metal score files): Q8_0 0.0037 mean KLD / 97.3% top-1 / 100% margin-qualified, Q4_K_M 0.108 / 81.7% / 97.1%; Q4_K_M against Q8_0 from the same score files 0.098 / 83.3%. Tokenizer 0/721 against the publisher's tokenizer.json at d50881f4, template conformance 23/23 in text and tokens against the template the GGUF carries, its 44-case tool-schema torture set byte for byte. Record: docs/compat-reports/k2-horizon-0.9b-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** reasoning_effort medium and low are refused (their think blocks are not split from the answer yet); high, the default, and thinking off are served. A tool without a parameters object is refused, as the template refuses it. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The Windows CUDA host's CPU and CUDA cells; a llama.cpp column against the same reference (not run); task quality and the native tool protocol on the real checkpoint (the parser is pinned on fixtures only). The publisher's Q4_K_M fails the fidelity bar (0.108); it is its own row, with that failure first.
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-k2-horizon-0.9b-q8_0.json](compat-reports/1.7.1-2026-10-30-macos-arm64-k2-horizon-0.9b-q8_0.json), [docs/compat-reports/k2-horizon-0.9b-reference-2026-10-30/README.md](compat-reports/k2-horizon-0.9b-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json), [tests/fixtures/k2-horizon-tool-schemas.json](../tests/fixtures/k2-horizon-tool-schemas.json).
- **Validation run:** on macos-arm64-metal, metal backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `b9af23d3578b00d9`, [pin](../tests/compatibility/tier-smoke/k2-horizon-0.9b-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `b9af23d3578b00d9`, [pin](../tests/compatibility/tier-smoke/k2-horizon-0.9b-q8_0.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.7.1 2026-10-30 on macos-arm64-metal at commit `c159098d0998`, output sha256 `b9af23d3578b00d9`, [pin](../tests/compatibility/tier-smoke/k2-horizon-0.9b-q8_0.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

<a id="k2-horizon-0.9b-q4_k_m"></a>
##### `K2-Horizon-1B-Q4_K_M.gguf`

Text path, IFM's own Q4_K_M of K2-Horizon 0.9B, which fails the fidelity bar: against the BF16 parent computed in float32 it reads 0.108 mean KLD against 0.05 (81.7% top-1, 97.1% margin-qualified, inside its 97%), and 0.098 against its own Q8_0. The Q8_0 is the fidelity file.

- **File:** `K2-Horizon-1B-Q4_K_M.gguf` (sha256 `78bff592aeba235c767c93a0bda5870d5a4a8dca21c2a2d502930dd3078c79f2`, 666,184,064 bytes, from [IFM/K2-Horizon-0.9B-GGUF](https://huggingface.co/IFM/K2-Horizon-0.9B-GGUF) at revision `118965081cf1`).
- **Upstream checkpoint:** IFM/K2-Horizon-0.9B at revision `d50881f4f8dc`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against IFM's BF16 parent computed in float32 on the CPU (the publisher's own modeling code, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, the Runner sides from Metal score files): Q8_0 0.0037 mean KLD / 97.3% top-1 / 100% margin-qualified, Q4_K_M 0.108 / 81.7% / 97.1%; Q4_K_M against Q8_0 from the same score files 0.098 / 83.3%. Tokenizer 0/721 against the publisher's tokenizer.json at d50881f4, template conformance 23/23 in text and tokens against the template the GGUF carries, its 44-case tool-schema torture set byte for byte. Record: docs/compat-reports/k2-horizon-0.9b-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** reasoning_effort medium and low are refused (their think blocks are not split from the answer yet); high, the default, and thinking off are served. A tool without a parameters object is refused, as the template refuses it. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The Windows CUDA host's CPU and CUDA cells; a llama.cpp column against the same reference (not run); task quality and the native tool protocol on the real checkpoint (the parser is pinned on fixtures only).
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-macos-arm64-k2-horizon-0.9b-q4_k_m.json](compat-reports/1.7.1-2026-10-30-macos-arm64-k2-horizon-0.9b-q4_k_m.json), [docs/compat-reports/k2-horizon-0.9b-reference-2026-10-30/README.md](compat-reports/k2-horizon-0.9b-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json), [tests/fixtures/k2-horizon-tool-schemas.json](../tests/fixtures/k2-horizon-tool-schemas.json).
- **Validation run:** on macos-arm64-metal, metal backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `128cdfebbb63b10f`, [pin](../tests/compatibility/tier-smoke/k2-horizon-0.9b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `128cdfebbb63b10f`, [pin](../tests/compatibility/tier-smoke/k2-horizon-0.9b-q4_k_m.windows-x86_64-cuda.cuda.txt)
- **Metal:** ✅ v1.7.1 2026-10-30 on macos-arm64-metal at commit `5cc154e261a7`, output sha256 `128cdfebbb63b10f`, [pin](../tests/compatibility/tier-smoke/k2-horizon-0.9b-q4_k_m.macos-arm64-metal.metal.txt)
- **LoRA training:** supported (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

#### K2-Horizon-7B (IFM)

<a id="k2-horizon-7b-q8_0"></a>
##### `K2-Horizon-7B-Q8_0.gguf`

Text path, IFM's own Q8_0 of K2-Horizon 7B, the fidelity file of the publisher's two: against the BF16 parent computed in float32 it reads 0.0159 mean KLD and 95.0% top-1 (300 positions, CPU).

- **File:** `K2-Horizon-7B-Q8_0.gguf` (sha256 `5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf`, 9,573,964,160 bytes, from [IFM/K2-Horizon-7B-GGUF](https://huggingface.co/IFM/K2-Horizon-7B-GGUF) at revision `7a75dbda05e3`).
- **Upstream checkpoint:** IFM/K2-Horizon-7B at revision `4622f94a6632`.
- **Quantization:** Q8_0 (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against IFM's BF16 parent computed in float32 on the lab CPU lane (the publisher's own modeling code, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, BOS native, the Runner sides from CPU score files at 5cc154e2): Q8_0 0.0159 mean KLD / 95.0% top-1 / 100% margin-qualified, Q4_K_M 0.1225 / 84.3% / 95.8%. Tokenizer 0/721 against the publisher's tokenizer.json at 4622f94a, template conformance as the 0.9B's (the same template). Record: docs/compat-reports/k2-horizon-7b-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: the grouped RMSNorm (four groups) has no GPU kernel yet, so the GPU backends serve it on the CPU. reasoning_effort medium and low are refused (their think blocks are not split from the answer yet); high, the default, and thinking off are served. A tool without a parameters object is refused, as the template refuses it. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The Windows CUDA host's CPU cell; a GPU path for the grouped norm; a llama.cpp column against the same reference (not run); task quality and the native tool protocol on the real checkpoint.
- **Evidence:** [docs/compat-reports/k2-horizon-7b-reference-2026-10-30/README.md](compat-reports/k2-horizon-7b-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.8.0 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `e9b3df046270dbf7`, [pin](../tests/compatibility/tier-smoke/k2-horizon-7b-q8_0.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** no: grouped RMSNorm (k2-horizon) not in the backward (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

<a id="k2-horizon-7b-q4_k_m"></a>
##### `K2-Horizon-7B-Q4_K_M.gguf`

Text path, IFM's own Q4_K_M of K2-Horizon 7B, which fails the fidelity bar: against the BF16 parent computed in float32 it reads 0.1225 mean KLD against 0.05 (84.3% top-1, 95.8% margin-qualified against 97%). The Q8_0 is the fidelity file.

- **File:** `K2-Horizon-7B-Q4_K_M.gguf` (sha256 `eb89c15a0ae9712be2ee462bf43802de14200f20f93b73da6eb68c2ebdd28e4e`, 5,592,217,984 bytes, from [IFM/K2-Horizon-7B-GGUF](https://huggingface.co/IFM/K2-Horizon-7B-GGUF) at revision `7a75dbda05e3`).
- **Upstream checkpoint:** IFM/K2-Horizon-7B at revision `4622f94a6632`.
- **Quantization:** Q4_K_M (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (a publisher float32 comparison exists):** Both publisher files against IFM's BF16 parent computed in float32 on the lab CPU lane (the publisher's own modeling code, gold-logits.py, 300 positions at stride 15 over gold-corpus-2k, token-id keyed, BOS native, the Runner sides from CPU score files at 5cc154e2): Q8_0 0.0159 mean KLD / 95.0% top-1 / 100% margin-qualified, Q4_K_M 0.1225 / 84.3% / 95.8%. Tokenizer 0/721 against the publisher's tokenizer.json at 4622f94a, template conformance as the 0.9B's (the same template). Record: docs/compat-reports/k2-horizon-7b-reference-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: the grouped RMSNorm (four groups) has no GPU kernel yet, so the GPU backends serve it on the CPU. reasoning_effort medium and low are refused (their think blocks are not split from the answer yet); high, the default, and thinking off are served. A tool without a parameters object is refused, as the template refuses it. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The Windows CUDA host's CPU cell; a GPU path for the grouped norm; a llama.cpp column against the same reference (not run); task quality and the native tool protocol on the real checkpoint.
- **Evidence:** [docs/compat-reports/k2-horizon-7b-reference-2026-10-30/README.md](compat-reports/k2-horizon-7b-reference-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.8.0 on 2026-10-30.
- **CPU:** ✅ v1.8.0 2026-10-30 on windows-x86_64-cuda at commit `9bb83e7e80b4`, output sha256 `537c65d7aee2a58d`, [pin](../tests/compatibility/tier-smoke/k2-horizon-7b-q4_k_m.windows-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **LoRA training:** no: grouped RMSNorm (k2-horizon) not in the backward (runner 1.8.0, 2026-10-30).
- **LoRA adapter serving:** supported (runner 1.8.0, 2026-10-30).
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

### flint

#### flint (RecursiveSI)

<a id="flint-int4-q4_1"></a>
##### `flint-int4-q4_1.gguf`

Text path, RecursiveSI's own Q4_1 of flint (39.8B, mixture of experts): it loads, tokenizes as the publisher's tokenizer.json does, renders the publisher's own template token for token and answers through it on the CPU, and agrees with another engine on this file (0.028 mean KLD, 99.0% margin-qualified top-1, 200 positions). No float32 reference is feasible (about 160 GB), so that is agreement, not fidelity.

- **File:** `flint-int4-q4_1.gguf` (sha256 `43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b`, 25,992,966,976 bytes, from [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) at revision `9c47151036bc`).
- **Upstream checkpoint:** RecursiveSI/flint at revision `2a247be0c0ae`.
- **Quantization:** Q4_1 (publisher).
- **Features claimed:** text. A feature not listed is not claimed.
- **Numerical anchor (no publisher float32 reference is feasible):** anchor: agreement with another engine on the publisher's file, not fidelity: Runner against mainline llama.cpp 36a0e2f on this Q4_1, both on the CPU of the lab CPU lane, 0.028 mean KLD, 90.0% top-1, 99.0% margin-qualified, 0.924 top-8 overlap over 200 raw-completion positions. A float32 reference (about 160 GB) is not feasible on any machine this project runs; the architecture's arithmetic is checked against transformers' own flint in float32 on a converted random model. Tokenizer 0/721 against the publisher's tokenizer.json at 2a247be0, template conformance 23 of 23 in text and tokens against its chat_template.jinja. Record: docs/compat-reports/flint-2026-10-30/README.md.
- **Context tested:** 4096.
- **Known limitations:** CPU only: attention sinks with a sliding window, NoPE full layers and sigmoid routing with a shared expert have no CUDA or Metal path yet, so the GPU backends serve it on the CPU. LoRA training is refused (sigmoid router). 26 GB of weights: the lab CPU lane is the only machine here that holds the file in memory. A required or named tool choice takes the generic JSON envelope on this family.
- **Still missing:** The CUDA path; task quality and the native tool protocol on the real checkpoint.
- **Evidence:** [docs/compat-reports/1.7.1-2026-10-30-linux-x86_64-flint-int4-q4_1.json](compat-reports/1.7.1-2026-10-30-linux-x86_64-flint-int4-q4_1.json), [docs/compat-reports/flint-2026-10-30/README.md](compat-reports/flint-2026-10-30/README.md), [tests/compatibility/coverage.json](../tests/compatibility/coverage.json).
- **Validation run:** on linux-x86_64-cuda (borrowed hardware: dated evidence, its absence does not hold a release), cpu backend; last run 1.7.1 on 2026-10-30.
- **CPU:** ✅ v1.7.1 2026-10-30 on linux-x86_64-cuda (borrowed hardware) at commit `f5f61a2e8d25`, output sha256 `8ce1eee2a9e1f4fc`, [pin](../tests/compatibility/tier-smoke/flint-int4-q4_1.linux-x86_64-cuda.cpu.txt)
- **CUDA:** ☐ not run
- **Metal:** ☐ not run
- **Tier history:** Upcoming, planned admission from 2026-10-30; Tier 2, maintained on report from 2026-10-30 ([log](#transition-log)).

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
- **Missing for Tier 1:** One gate run on the Linux CUDA host: quant bar, tool gate, pinned output, device record; a file pinned by sha256.
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
| 2026-10-22 | [Trinity Nano, `Trinity-Nano-Preview-Q8_0.gguf`](#trinity-nano-preview-q8_0) | Tier 1, actively validated | Migration: policy decision 2026-10-22, Tier 1 as the pinned afmoe file (the manual's 8 GB recommendation and a 1.5.0 golden-pass row). | none recorded |
| 2026-10-22 | [Qwen3.8-27B, `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf`](#qwen3.8-27b-gsq-rco-iq3_s-recovered) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Qwen3.8-27B, `Qwen3.8-27B-MTP-Q8_0.gguf`](#qwen3.8-27b-mtp-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-BF16.gguf`](#xyntetik-kvist-14b-bf16) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-Q8_0.gguf`](#xyntetik-kvist-14b-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-Q5_0-mix.gguf`](#xyntetik-kvist-14b-q5_0-mix) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Xyntetik-Kvist-14B, `Xyntetik-Kvist-14B-IQ4_NL-mix.gguf`](#xyntetik-kvist-14b-iq4_nl-mix) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Qwen3-30B-A3B selective precision, `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf`](#qwen3-30b-a3b-expq4_0-attnq8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Qwen3-Coder-30B-A3B keep-120, `Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf`](#qwen3-coder-30b-a3b-instruct-keep120-q4_k_m) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Gemma 4 E2B, `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`](#gemma-4-e2b-it-q4_k_m-q4_0-mix) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [NVIDIA Nemotron Nano 9B v2, `NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf`](#nvidia-nemotron-nano-9b-v2-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Gemma 4 31B, attention block 48 removed, `gemma-4-31B-it-attn48-removed-q4_0.gguf`](#gemma-4-31b-it-attn48-removed-q4_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf`](#muse-glimmer-30b-surgical-4p75-q4_k) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf`](#muse-glimmer-30b-surgical-6p34-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf`](#muse-glimmer-30b-surgical-6p34-q4_k-recovered) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Qwen3-4B Runner ToolUse, `merged-Q8_0.gguf`](#qwen3-4b-runner-tooluse-merged-q8_0) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Qwen3-4B Runner ToolUse, `Qwen3-4B-ToolUse-LoRA.gguf`](#qwen3-4b-runner-tooluse-lora) | Tier 1, actively validated | Migration: published under the maintainer's Hugging Face account, Tier 1 by policy decision 1. | none recorded |
| 2026-10-22 | [Qwen3.8-Flash-Next, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-with-MTP-head-00001-of-00004.gguf`](#qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head) | Tier 1, actively validated | Migration: published file, Tier 1 by policy decision 1; policy decision 2026-10-22: stays Tier 1 with 'experimental' first. | none recorded |
| 2026-10-22 | [Qwen3.8-Flash-Next, `Qwen3.8-Flash-Next-MTP-Q8_0.gguf`](#qwen3.8-flash-next-mtp-q8_0) | Tier 1, actively validated | Migration: published file, Tier 1 by policy decision 1; policy decision 2026-10-22: stays Tier 1 with 'experimental' first. | none recorded |
| 2026-10-22 | [gpt-oss-20b keep-30, `gpt-oss-20b-keep30-MXFP4.gguf`](#gpt-oss-20b-keep30-mxfp4) | Tier 2, maintained on report | Migration: policy decision 2026-10-22, Tier 2 with its near-miss disclosure; it stays published. | none recorded |
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
| 2026-10-22 | [Cloudflare Clef Flash (clef, on Qwen3.5-9B), `Clef-Flash-Q4_K_M.gguf`](#cloudflare-clef-flash) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Nemotron 3 Nano (30B-A3B)](#nemotron-3-nano-30b-a3b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Nemotron 3.5 Bolt-74B-A9B](#nemotron-3.5-bolt-74b-a9b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Nemotron 3.5 Super VL-120B-A12B](#nemotron-3.5-super-vl-120b-a12b) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Kolibri-1 (Aleph Alpha, 78B)](#kolibri-1) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-22 | [Mellum2.1-12B-A2.5B-Thinking (JetBrains), `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf`](#mellum2.1-12b-a2.5b-thinking-q4_k_m) | Upcoming, planned admission | Migration: on the owner's roster; the envelope is not complete (audit 2026-10-22). | none recorded |
| 2026-10-27 | [Cloudflare Clef (clef, on Qwen3.8-27B), `Cloudflare_clef-IQ3_M.gguf`](#cloudflare-clef-iq3_m) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Qwen3.8-27B, `Qwen3.8-27B-UD-Q4_K_M.gguf`](#qwen3.8-27b-ud-q4_k_m) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Gemma 4 12B, `gemma-4-12b-it-qat-q4_0.gguf`](#gemma-4-12b-it-qat-q4_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Gemma 4 26B-A4B, `gemma-4-26B_q4_0-it.gguf`](#gemma-4-26b-a4b-it-qat-q4_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Gemma 4 31B, `gemma-4-31B-it-Q4_0.gguf`](#gemma-4-31b-it-q4_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Muse-Glimmer-30B, `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`](#muse-glimmer-30b-kquant) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Granite-4.2-8B, `granite-4.2-8b-Q4_K_M.gguf`](#granite-4.2-8b-q4_k_m) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Granite-4.2-8B, `granite-4.2-8b-Q8_0.gguf`](#granite-4.2-8b-q8_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Trinity Nano, `Trinity-Nano-Preview-Q8_0.gguf`](#trinity-nano-preview-q8_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; this is its publisher's file and the repository's downloads are the publisher's. Maintained on report; its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Qwen3-30B-A3B selective precision, `Qwen3-30B-A3B-expq4_0-attnq8_0.gguf`](#qwen3-30b-a3b-expq4_0-attnq8_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 760 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Qwen3-Coder-30B-A3B keep-120, `Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M.gguf`](#qwen3-coder-30b-a3b-instruct-keep120-q4_k_m) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 813 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Gemma 4 E2B, `gemma-4-E2B-it-Q4_K_M-Q4_0-mix.gguf`](#gemma-4-e2b-it-q4_k_m-q4_0-mix) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 552 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [NVIDIA Nemotron Nano 9B v2, `NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf`](#nvidia-nemotron-nano-9b-v2-q8_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 496 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Gemma 4 31B, attention block 48 removed, `gemma-4-31B-it-attn48-removed-q4_0.gguf`](#gemma-4-31b-it-attn48-removed-q4_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 454 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-4p75-Q4_K.gguf`](#muse-glimmer-30b-surgical-4p75-q4_k) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 298 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-6p34-Q8_0.gguf`](#muse-glimmer-30b-surgical-6p34-q8_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 339 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Muse-Glimmer-30B surgical, `Muse-Glimmer-30B-surgical-6p34-Q4_K-recovered.gguf`](#muse-glimmer-30b-surgical-6p34-q4_k-recovered) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 292 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Qwen3-4B Runner ToolUse, `merged-Q8_0.gguf`](#qwen3-4b-runner-tooluse-merged-q8_0) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 639 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Qwen3-4B Runner ToolUse, `Qwen3-4B-ToolUse-LoRA.gguf`](#qwen3-4b-runner-tooluse-lora) | Tier 2, maintained on report | Policy decision 2026-10-27: Tier 1 follows pull; the repository showed 639 downloads in the 30 days to 2026-10-27, under the 1,000 bar. Maintained on report; it stays published and its records stay as dated evidence. | 1.5.1 |
| 2026-10-27 | [Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B), `Clef-backbone-IQ3_S.gguf`](#cloudflare-clef-iq3_s-recovered) | Tier 2, maintained on report | Admitted 2026-10-27 at Tier 2 (policy decision 2026-10-27: Tier 1 follows pull, and a new repository has no downloads yet). The card leads with the measured shortfall against the house bar. | none recorded |
| 2026-10-27 | [Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B), `clef-IQ3_S-recovery-QR-001e.lora.F16.gguf`](#cloudflare-clef-iq3_s-recovery-qr-001e-lora) | Tier 2, maintained on report | Admitted 2026-10-27 at Tier 2 (policy decision 2026-10-27: Tier 1 follows pull, and a new repository has no downloads yet). The card leads with the measured shortfall against the house bar. | none recorded |
| 2026-10-27 | [Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B), `clef-IQ3_S-recovery-QR-001b.lora.F16.gguf`](#cloudflare-clef-iq3_s-recovery-qr-001b-lora) | Tier 2, maintained on report | Admitted 2026-10-27 at Tier 2 (policy decision 2026-10-27: Tier 1 follows pull, and a new repository has no downloads yet). The card leads with the measured shortfall against the house bar. | none recorded |
| 2026-10-28 | [Nemotron 3.5 Bolt E4B, `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`](#nemotron-3.5-bolt-e4b-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-28 at Tier 2 on the owner's request (policy decision 2026-10-27: Tier 1 follows pull; this is the publisher's file and its downloads are the publisher's). Maintained on report, with the admitting run recorded. | none recorded |
| 2026-10-28 | [Cloudflare Clef IQ3_S recovered (clef, on Qwen3.8-27B), `Clef-IQ3_S.clef.gguf`](#cloudflare-clef-iq3_s-native) | Tier 2, maintained on report | Admitted 2026-10-28 at Tier 2 on the owner's direct request (the head added to the 2026-10-27 repository; policy decision 2026-10-27: Tier 1 follows pull, and the repository is under the bar). The card leads with the measured shortfall against the house bar and reports decisions descriptively. | none recorded |
| 2026-10-29 | [Nemotron 3.5 Bolt E4B, `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`](#nemotron-3.5-bolt-e4b-q4_k_m) | Tier 1, actively validated | Owner decision 2026-10-29: promoted to Tier 1 for active validation (the publisher's file; the 2026-10-27 pull rule yields to the owner's call). The admitting Metal, CUDA and CPU cells and the train smoke stand as the record; the BF16-parent fidelity measurement is still owed. | 1.7.0 |
| 2026-10-29 | [Mellum2.1-12B-A2.5B-Thinking (JetBrains), `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf`](#mellum2.1-12b-a2.5b-thinking-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-29 at Tier 2 (policy decision 2026-10-27: Tier 1 follows pull; this is the publisher's file and its downloads are the publisher's). The BF16-parent result is stated first; maintained on report, with the admitting run recorded. | none recorded |
| 2026-10-29 | [Mellum2.1-12B-A2.5B-Thinking (JetBrains), `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf`](#mellum2.1-12b-a2.5b-thinking-q4_k_m) | Tier 1, actively validated | Owner decision 2026-10-29: promoted to Tier 1 for active validation (the publisher's file; the 2026-10-27 pull rule yields to the owner's call). The admitting CPU and CUDA cells stand as the record; the row keeps its fidelity reading (the Q4_K_M fails the bar against the BF16 parent, Q8_0 is the fidelity file); the Metal row and the Q8_0 cells are still owed. | 1.7.0 |
| 2026-10-29 | [MiniCPM5-2B (openbmb), `MiniCPM5-2B-Q8_0.gguf`](#minicpm5-2b-q8_0) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the first of nine; the files did not tokenize as the publisher's tokenizer does until the minicpm5 pre-tokenizer rule, the BOS rule and the minicpm5 template family landed. | none recorded |
| 2026-10-29 | [MiniCPM5-2B (openbmb), `MiniCPM5-2B-Q8_0.gguf`](#minicpm5-2b-q8_0) | Tier 2, maintained on report | Admitted 2026-10-29 at Tier 2 with the admitting Metal and CPU cells and a float32 reference measurement (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-29 | [MiniCPM5-2B (openbmb), `MiniCPM5-2B-Q4_K_M.gguf`](#minicpm5-2b-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the first of nine; the files did not tokenize as the publisher's tokenizer does until the minicpm5 pre-tokenizer rule, the BOS rule and the minicpm5 template family landed. | none recorded |
| 2026-10-29 | [MiniCPM5-2B (openbmb), `MiniCPM5-2B-Q4_K_M.gguf`](#minicpm5-2b-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-29 at Tier 2 with the admitting Metal and CPU cells and a float32 reference measurement (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-29 | [MiMo-V2.6-Distill-Qwen-9B (XiaomiMiMo), `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf`](#mimo-v2.6-distill-qwen-9b-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the second of nine; the file rendered as Qwen3-Coder's template and tokenized combining marks differently from the publisher's tokenizer until the mimo template family and the qwen2 split landed. | none recorded |
| 2026-10-29 | [MiMo-V2.6-Distill-Qwen-9B (XiaomiMiMo), `MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf`](#mimo-v2.6-distill-qwen-9b-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-29 at Tier 2 with the admitting CPU cell from the Metal laptop and a float32 reference measurement (policy 2026-10-27). Maintained on report. | none recorded |
| 2026-10-30 | [Cloudflare Clef Flash (clef, on Qwen3.5-9B), `Clef-Flash-Q4_K_M.gguf`](#cloudflare-clef-flash) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the Windows CUDA host's CPU and CUDA cells, the tokenizer and template checks; the float32 reference is owed (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-30 | [Cloudflare Clef Flash (clef, on Qwen3.5-9B), `Clef-Flash-Q8_0.gguf`](#cloudflare-clef-flash-q8_0) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the Windows CUDA host's CUDA cell (a layer split, --gpu 28 of 32), and the tokenizer and template checks of the Q4_K_M, which shares its vocabulary and template; the float32 reference is owed (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-30 | [LFM2.5-1.2B-Instruct (LiquidAI), `LFM2.5-1.2B-Instruct-Q8_0.gguf`](#lfm2.5-1.2b-instruct-q8_0) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29 (LFM2.5-24B-A2B); scope decided 2026-10-30 to admit the shared architecture through the dense 1.2B-Instruct first. The files did not load until the lfm2 architecture, its tokenizer rule and the lfm2 template family landed. | none recorded |
| 2026-10-30 | [LFM2.5-1.2B-Instruct (LiquidAI), `LFM2.5-1.2B-Instruct-Q8_0.gguf`](#lfm2.5-1.2b-instruct-q8_0) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell and a float32 reference measurement (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-30 | [LFM2.5-1.2B-Instruct (LiquidAI), `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`](#lfm2.5-1.2b-instruct-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29 (LFM2.5-24B-A2B); scope decided 2026-10-30 to admit the shared architecture through the dense 1.2B-Instruct first. The files did not load until the lfm2 architecture, its tokenizer rule and the lfm2 template family landed. | none recorded |
| 2026-10-30 | [LFM2.5-1.2B-Instruct (LiquidAI), `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`](#lfm2.5-1.2b-instruct-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell and a float32 reference measurement (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-30 | [LFM2.5-350M (LiquidAI), `LFM2.5-350M-Q8_0.gguf`](#lfm2.5-350m-q8_0) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29 (LFM2.5-24B-A2B); scope decided 2026-10-30 to admit the shared architecture through the dense 1.2B-Instruct first; the 350M follows as its own row. The files did not load until the lfm2 architecture, its tokenizer rule and the lfm2 template family landed. | none recorded |
| 2026-10-30 | [LFM2.5-350M (LiquidAI), `LFM2.5-350M-Q8_0.gguf`](#lfm2.5-350m-q8_0) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell and a float32 reference measurement (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-30 | [LFM2.5-350M (LiquidAI), `LFM2.5-350M-Q4_K_M.gguf`](#lfm2.5-350m-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29 (LFM2.5-24B-A2B); scope decided 2026-10-30 to admit the shared architecture through the dense 1.2B-Instruct first; the 350M follows as its own row. The files did not load until the lfm2 architecture, its tokenizer rule and the lfm2 template family landed. | none recorded |
| 2026-10-30 | [LFM2.5-350M (LiquidAI), `LFM2.5-350M-Q4_K_M.gguf`](#lfm2.5-350m-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell and a float32 reference measurement (policy 2026-10-27: Tier 1 follows pull or the owner's call). Maintained on report. | none recorded |
| 2026-10-30 | [Spark-X2.6-4B (XHToken), `Spark-X2.6-4B-Q4_K_M.gguf`](#spark-x2.6-4b-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the third item: a new architecture (spark2_5), its tokenizer rule and its chat templates, implemented before validation. | none recorded |
| 2026-10-30 | [Spark-X2.6-4B (XHToken), `Spark-X2.6-4B-Q4_K_M.gguf`](#spark-x2.6-4b-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell from the Metal laptop (policy 2026-10-27). Maintained on report. | none recorded |
| 2026-10-30 | [Spark-X2.5-4B (XHToken), `Spark-X2.5-4B-Q4_K_M.gguf`](#spark-x2.5-4b-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the third item: a new architecture (spark2_5), its tokenizer rule and its chat templates, implemented before validation. | none recorded |
| 2026-10-30 | [Spark-X2.5-4B (XHToken), `Spark-X2.5-4B-Q4_K_M.gguf`](#spark-x2.5-4b-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell from the Metal laptop (policy 2026-10-27). Maintained on report. | none recorded |
| 2026-10-30 | [K2-Horizon-0.9B (IFM), `K2-Horizon-1B-Q8_0.gguf`](#k2-horizon-0.9b-q8_0) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the fourth item: a new architecture (k2-horizon), its tokenizer rule and its chat template, implemented before validation. | none recorded |
| 2026-10-30 | [K2-Horizon-0.9B (IFM), `K2-Horizon-1B-Q8_0.gguf`](#k2-horizon-0.9b-q8_0) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting cells from the Metal laptop (policy 2026-10-27). Maintained on report. | none recorded |
| 2026-10-30 | [K2-Horizon-0.9B (IFM), `K2-Horizon-1B-Q4_K_M.gguf`](#k2-horizon-0.9b-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the fourth item: a new architecture (k2-horizon), its tokenizer rule and its chat template, implemented before validation. | none recorded |
| 2026-10-30 | [K2-Horizon-0.9B (IFM), `K2-Horizon-1B-Q4_K_M.gguf`](#k2-horizon-0.9b-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting cells from the Metal laptop (policy 2026-10-27). Maintained on report. | none recorded |
| 2026-10-30 | [K2-Horizon-7B (IFM), `K2-Horizon-7B-Q8_0.gguf`](#k2-horizon-7b-q8_0) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the fourth item: the 7B member, which carries the grouped RMSNorm the 0.9B does not. | none recorded |
| 2026-10-30 | [K2-Horizon-7B (IFM), `K2-Horizon-7B-Q8_0.gguf`](#k2-horizon-7b-q8_0) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell from the lab CPU lane and a float32 reference measurement (policy 2026-10-27). Maintained on report. | none recorded |
| 2026-10-30 | [K2-Horizon-7B (IFM), `K2-Horizon-7B-Q4_K_M.gguf`](#k2-horizon-7b-q4_k_m) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the fourth item: the 7B member, which carries the grouped RMSNorm the 0.9B does not. | none recorded |
| 2026-10-30 | [K2-Horizon-7B (IFM), `K2-Horizon-7B-Q4_K_M.gguf`](#k2-horizon-7b-q4_k_m) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell from the lab CPU lane and a float32 reference measurement (policy 2026-10-27). Maintained on report. | none recorded |
| 2026-10-30 | [flint (RecursiveSI), `flint-int4-q4_1.gguf`](#flint-int4-q4_1) | Upcoming, planned admission | Owner's Hub-search validation queue of 2026-10-29, the fifth item: a new architecture (flint), its tokenizer rule and its chat template, implemented before validation. | none recorded |
| 2026-10-30 | [flint (RecursiveSI), `flint-int4-q4_1.gguf`](#flint-int4-q4_1) | Tier 2, maintained on report | Admitted 2026-10-30 at Tier 2 with the admitting CPU cell from the lab CPU lane (policy 2026-10-27); no float32 reference is feasible, so the anchor is agreement with another engine. Maintained on report. | none recorded |

<!-- model-tiers:rows:end -->
>>>>>>> 50ad0859 (flint: developer turns keep their own role)
