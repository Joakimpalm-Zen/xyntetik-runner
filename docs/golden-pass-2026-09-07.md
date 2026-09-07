# The golden pass over every admitted family, 2026-09-07

The standing rule (AGENTS.md, "The provider's reference implementation is
the primary anchor") was proven on one family. This pass applies it to
every family already admitted, because each of those was certified under
the llama.cpp yardstick and therefore carries the same two blind spots: a
chat template that is not the publisher's, and a numerical reading both
engines could have copied from each other.

Two phases, both on the Blackwell against runner `golden-pass`:

- **Phase A**, template conformance: every family's renderer against the
  publisher's own template (or, where the publisher's page is gated, the
  template embedded in the pinned file), text and token for token.
- **Phase B**, gold logits: each family's smallest member against the
  publisher's reference implementation in float32, with llama.cpp scored
  beside it on the same file.

Raw outputs in `docs/golden-pass-evidence/`.

## The instrument was wrong three times first

Phase B's first run reported mean KL around 2 for gemma-3, 5 for gemma-4
and 2 for Phi-3.5, with **both engines agreeing with each other** and both
disagreeing with the reference, while qwen3 and granite read exactly
0.0000. That pattern is the signature of a broken instrument, not two
broken engines, and the split says which: qwen3 and granite are the
families whose tokenizer prepends nothing.

The gold script scored the reference on the bare token ids while the
serving endpoints tokenized the prefix text and prepended their
beginning-of-sequence token, so every BOS-prepending family was compared
across a one-token offset. Fixed by taking the special prefix from the
tokenizer itself and giving it to the reference
(`scripts/gold-logits.py`, `special_prefix_tokens` now in every report).
Recorded here rather than quietly corrected: without it this pass would
have filed false numerical defects against three families, and the rule
that caught it is the project's own, that a gate needs an anchor outside
the system under test.

Two more followed, both worth writing down because both produced
confident wrong output rather than an error:

- **A stale report read as a fresh one.** The re-run left the previous
  run's JSON in place when a server failed to bind a port, and the
  summary printer read it, reprinting the old numbers to four decimals
  under the new run's heading. The tell was that the corrected pass
  reproduced the uncorrected pass exactly, digit for digit, on seven
  families. The harness now deletes the target report before each run and
  reports its absence afterwards as a failure, and a server that cannot
  bind aborts its family loudly.
- **A position with no logprobs killed a family.** A request whose next
  token is the end of text comes back from at least one server without
  the logprobs block; the script raised and lost every position after it.
  Such a position is now skipped and counted
  (`positions_skipped_no_logprobs`).

Three harness faults in one morning, all found by the same reflex: when a
number looks impossible, or two things that should differ agree exactly,
suspect the instrument before the subject. The measurements below are the
ones taken after all three were fixed.

## Phase A: what the publishers' templates say

Sixteen families were already in the gate. Thirteen more were added, one
per admitted architecture that had none. Result over 29 families and 503
cases: **18 clean, 11 with findings**, and 487 of those cases compared at
the token level as well as the text level (345 identical, 142 differing).

Clean: `chatml`, `chatml-think`, `llama2`, `llama3`, `zephyr`, `gemma`,
`gemma4-mainline`, `mistral` (publisher's current template), `phi3`,
`apertus`, `ornith`, `qwen38`, `granite42`, `muse`, `trinity`, `eurollm`,
`granite`, `harmony`. One family, `gemma4`, could not be compared at all:
the E2B file the gate reads its oracle from carries no chat template, so
its E-series renderer is unmeasured and that is a gap, not a pass.

`lucie` is the eleventh finding and it is not a defect: its 16 cases are
the beginning-of-sequence token, which the runner leaves to the tokenizer
by design, and it now carries the same allowlist entry llama3 has. The
ten below are the real ones.

The findings sort into five kinds.

### 1. A fixed default system prompt the runner never sends

`smollm2` (10 cases), `hermes4` (5 of 15), `granite40h` (10 of 14). Each
publisher's template injects a fixed system turn when the caller sends
none:

| family | text the model was trained to see |
|---|---|
| SmolLM2 | `You are a helpful AI assistant named SmolLM, trained by Hugging Face` |
| Hermes 4 | `You are Hermes, created by Nous Research.` |
| Granite 4.0-h | `You are a helpful assistant. ...` |

The runner sends the user turn alone. This is not the live-date case the
allowlist already forgives for llama3, apertus and muse: there is nothing
unreproducible about a fixed string, and omitting it puts the model
outside its training distribution on every system-less request. Genuine
defect, one mechanism, three families.

### 2. Two Qwen 3.5 divergences, on a survivor-suite family

`qwen35-4b` (7 cases), `qwen35-0.8b` (17). Qwen 3.5 is detected as
`ornith` and rendered by ornith's renderer. Two differences:

- Ornith opens every historical assistant turn with a thought block, so
  the runner writes `<think>\n\n</think>\n\n` in front of replayed
  assistant content. Qwen 3.5's own template writes nothing there.
- The 0.8B's template **defaults thinking off**; the runner's ornith path
  defaults it on, so the generation prompt is `<think>\n` where the
  publisher writes `<think>\n\n</think>\n\n`.

This is the same shape as the Qwen 3.8 finding of 2026-09-06, on the
family the survivor suite is built around (Qwen3.5 4B and 9B are
survivors). Genuine defect; the fix is a `qwen35` template family, the
work `qwen38` already scoped.

### 3. Phi-4-mini is not Phi-3.5

`phi4mini`, 20 of 22 cases. Phi-3.5's template writes `<|user|>\n` and a
newline before `<|end|>`; Phi-4-mini's writes `<|user|>` and no newlines
at all. The runner serves both through `phi3`, so every turn of a
Phi-4-mini conversation carries newline tokens the model never saw.
Genuine defect, and one the `phi3` row could not show because that row
renders Phi-3.5, which the runner matches exactly.

### 4. Tool declarations in the runner's words, not the publisher's

`hermes4`, `granite40h`, `nemotron-lightning` (4 cases each). The runner
writes its own tools preamble where each publisher ships one. The
Nemotron 3.5 Lightning case is the sharpest: its template declares tools
as nested `<function><name>...</name><parameters>` XML, and the runner,
which detects that file as `granite42`, writes `{"name": ...}` JSON
inside the same `<tools>` block. Genuine; it belongs to the same
provider-contract work as finding 1.

### 5. Nemotron Nano is not recognised at all

`nemotron-nano`, 19 of 22 cases, and the runner says so at load: *this
model ships a chat template this build does not recognise; falling back
to llama2 markup, which the model was not trained on*. Its framing is
`<SPECIAL_10>System`, `<SPECIAL_11>User`, `<SPECIAL_12>`, with
`/think` and `/no_think` control strings and an `<AVAILABLE_TOOLS>` JSON
block. The refusal is honest and loud, and the file is pinned in the
compatibility manifest with `chat` as a declared check. Genuine gap, and
the largest single one this pass found.

### The two Mistral rows, which are not renderer defects

Worth stating precisely, because the first reading was wrong. The pinned
`Mistral-7B-Instruct-v0.3` GGUF and the `Mistral-Nemo-Instruct-2407` GGUF
each embed a template **older than the one their publisher ships today**.
The gate's `mistral` row, which renders the publisher's current template,
is clean. Against the embedded templates the differences are a leading
space on the v0.3 file (the three-way discriminator reads its
`content + ' [/INST]'` literal as the v0.1 form) and the beginning-of-
sequence token, which the runner leaves to the tokenizer by design and
which the allowlist already accounts for. Filed as an artifact-freshness
question, not as a renderer defect.

## Phase B: against the publishers' own implementations

Transformers 5.16, float32, eager attention, on the Blackwell's CPUs,
from each publisher's safetensors. llama.cpp 73a43d1 with flash attention
off is scored on the same file beside the runner. 100 corpus positions.

Mean KL from the reference distribution, lower is closer; top-1 is how
often the engine picks the reference's token. `bf16` is the publisher's
own weights converted without quantization, so it isolates arithmetic;
`served` is the quantized file the manifest pins.

| family | weights | runner KL | runner top-1 | runner mq | llama.cpp KL | llama.cpp top-1 | llama.cpp mq |
|---|---|---|---|---|---|---|---|
| SmolLM2 135M | bf16 | **0.0000** | 100% | 100.0% | 0.0168 | 100% | 100.0% |
| SmolLM2 135M | Q8_0 | **0.0095** | 98.98% | 100.0% | 0.0338 | 95.92% | 100.0% |
| Qwen2.5 0.5B | bf16 | **0.0004** | 100% | 100.0% | 0.0017 | 100% | 100.0% |
| Qwen2.5 0.5B | Q4_K_M | **0.0694** | 88% | **98.4%** | 0.0762 | 87% | 96.7% |
| Qwen3 0.6B | bf16 | **0.0000** | 100% | 100.0% | 0.0009 | 100% | 100.0% |
| Qwen3 0.6B | Q4_0 | **0.3535** | 58% | 69.8% | 0.3582 | 58% | 69.8% |
| Qwen3 4B | bf16 | **0.0000** | 100% | 100.0% | 0.0002 | 100% | 100.0% |
| Qwen3 4B | Q4_K_M | **0.1278** | 81% | **94.4%** | 0.1315 | 82% | 93.1% |
| Qwen3.5 0.8B | bf16 | **0.0015** | 100% | 100.0% | 0.0023 | 99% | 100.0% |
| Qwen3.5 0.8B | Q4_K_M | **0.0659** | 87% | 98.4% | 0.0730 | 89% | 98.4% |
| Granite 4.0-h micro | bf16 | **0.0000** | 100% | 100.0% | 0.0005 | 100% | 100.0% |
| Granite 4.0-h micro | Q4_K_M | **0.1006** | 78% | 88.1% | 0.1078 | 79% | **89.6%** |
| Gemma 3 4B | bf16 | **0.0000** | 100% | 100.0% | 0.0001 | 100% | 100.0% |
| Gemma 3 4B | Q4_K_M | **0.0607** | 85% | **91.4%** | 0.0685 | 81% | 88.9% |
| Trinity Nano (afmoe) | bf16 | 0.1100 | 81.82% | 92.2% | **0.1105** | 83.84% | **93.8%** |
| Trinity Nano (afmoe) | Q8_0 | 0.1491 | 80% | 90.8% | **0.1223** | 82% | **93.8%** |
| Gemma 4 E2B | bf16 | **0.0009** | 100% | 100.0% | 0.0012 | 100% | 100.0% |
| Gemma 4 E2B | Q4_0 | 0.8558 | 61% | 67.9% | **0.8252** | 62% | **69.0%** |
| Apertus 8B | Q4_K_M | **0.0467** | 90% | **97.1%** | 0.0849 | 84% | 92.9% |
| Nemotron Nano 9B v2 | bf16 | **0.0027** | 98% | 100.0% | 0.0031 | 98% | 100.0% |
| Nemotron Nano 9B v2 | Q8_0 | **0.0041** | 97% | 100.0% | 0.0054 | 95% | 100.0% |
| StableLM 2 1.6B | bf16 | 0.6099 | 55% | 67.1% | **0.0007** | **100%** | **100.0%** |

Granite 4.2 3B, measured the day before under the same gate, reads 0.0016
at bf16 with top-1 100%. At Q4_K_M it has the lower KL, 0.109 against
0.115, and it loses both of the other two columns: see "The column this
report nearly buried" below.

**The pattern holds and it is narrow.** The runner is the closer engine on
**ten of eleven** families at bf16 and **nine of eleven** at the served
quant, and on unquantized weights it reproduces the publisher's own
implementation to 0.0000 or near it on eight of them. Apertus is the widest margin at the served quant,
0.0467 against 0.0849, and Nemotron Nano 9B is worth naming twice: its
numerics are among the best in the table while its chat template is one
the runner does not recognise at all. The arithmetic and the protocol
fail independently, which is the argument for measuring both. That is the f32 activation and f32 attention accumulation showing
up as a measurable thing rather than a claim. But the margin is small in
absolute terms, and at the quant people actually serve, quantization
dominates both engines by one to two orders of magnitude: on Qwen3 4B the
two engines sit 0.128 and 0.132 from the reference, a gap of 0.004
between them inside a 0.13 gap to the model. The honest form of the
public claim is that the runner is the closer of the two engines *on
this measure*, not that it is close to the model at 4 bits, and not that
it is the more accurate engine at the token a user receives. The next
section is why that last qualifier is there.

**And three rows go the other way.** Trinity Nano (afmoe) at its served
Q8_0, 0.1223 against the runner's 0.1491, with better top-1 at both
weights; Gemma 4 E2B at Q4_0, 0.8252 against 0.8558; and `stablelm`,
which is not a near-miss but a defect and has its own section below.
Recorded, not smoothed over.

### The column this report nearly buried

Every sentence above is about mean KL, and mean KL is the measure that
favours the runner. This project's own quality bar is not mean KL. It is
**margin-qualified top-1**: the share of positions where the reference
had a clear preference (its top two tokens separated by more than the
0.5-nat tie band) and the engine picked it. That column was measured in
every run and it was left out of the first draft of this table. It is in
the table now, and it does not tell the same story.

On the 22 rows above, margin-qualified top-1 splits **four to the runner,
five to llama.cpp, thirteen tied**, and the section after next shows that
none of those nine differences except StableLM is distinguishable from
noise at this corpus size. The split is reported here because it is what
the measure says, not because it is a finding. The runner takes Qwen2.5 0.5B,
Qwen3 4B and Gemma 3 4B at Q4_K_M and Apertus 8B; llama.cpp takes
Granite 4.0-h micro at Q4_K_M, Trinity Nano at both weights, Gemma 4 E2B
at Q4_0, and StableLM. Every one of the thirteen ties is a row where both
engines score 100% except Qwen3 0.6B at Q4_0 and Qwen3.5 0.8B at Q4_K_M,
so the disagreements are the whole signal.

The two measures disagree for a reason that is worth stating rather than
explaining away. The runner's f32 activations and f32 attention
accumulation move the *whole distribution* slightly closer to the
reference, which is what mean KL rewards. They do not reliably change
*which token comes out on top*, which is what a user sees and what
margin-qualified top-1 measures. A defensible claim from this table is
that the runner reproduces the reference distribution more closely. The
claim that it is the more accurate engine at the token a user receives
is not supported by these 22 rows.

**Flash attention makes it narrower still.** llama.cpp was run here with
its flash-attention kernel off, on the reasoning that the f16 accumulation
in that kernel is one of the shortcuts this report is about. At 4 bits
that reasoning is wrong in llama.cpp's favour: turning the kernel on
improves it. The Granite 4.2 3B certification of 2026-09-06 measured all
three configurations on the same file.

| weights | engine | mean KL | top-1 | margin-qualified |
|---|---|---|---|---|
| Q4_K_M | runner | **0.109** | 87.9% | 94.7% |
| Q4_K_M | llama.cpp `-fa off` | 0.119 | 85.9% | **97.3%** |
| Q4_K_M | llama.cpp `-fa on` | 0.115 | **88.9%** | **97.3%** |

Read the table carefully, because a first pass over it (including the
first version of this section) gets the causation wrong. **Flash
attention does not move the margin-qualified column at all**: llama.cpp
reads 97.3% with the kernel on and 97.3% with it off. What the kernel
changes is mean KL and plain top-1. So llama.cpp clears this project's
97% bar on that family at that quant in either configuration, and the
runner does not in either. Turning the kernel on is still the right
thing to measure, because it closes the runner's mean-KL lead (see the
significance table below), but it is not the reason llama.cpp is ahead
on the bar we gate on. Re-running the served tier with `-fa on` is
filed as R6.7.7.

### How much of this is real: n=100 is too small to say

Every comparison above rests on 100 corpus positions, of which 61 to 84
qualify for the margin-qualified column. At that size **one position is
worth 1.2 to 1.6 percentage points**, so every headline gap in this
report is a handful of tokens. Written out as counts rather than
percentages:

| family, served tier | qualified | runner hits | llama.cpp hits | gap | McNemar exact p |
|---|---|---|---|---|---|
| Apertus 8B | 70 | **68** | 65 | 3 to runner | 0.250 |
| Gemma 3 4B | 81 | **74** | 72 | 2 to runner | 0.500 |
| Qwen2.5 0.5B | 61 | **60** | 59 | 1 to runner | 1.000 |
| Qwen3 4B | 72 | **68** | 67 | 1 to runner | 1.000 |
| Gemma 4 E2B | 84 | 57 | **58** | 1 to llama.cpp | 1.000 |
| Granite 4.0-h micro | 67 | 59 | **60** | 1 to llama.cpp | 1.000 |
| Trinity Nano | 65 | 59 | **61** | 2 to llama.cpp | 0.500 |
| Granite 4.2 3B (`-fa on`) | 75 | 71 | **73** | 2 to llama.cpp | 0.500 |
| StableLM 2 1.6B | 70 | 47 | **70** | 23 to llama.cpp | <0.001 |

**Not one of these differences reaches significance except StableLM.**
The 4 / 5 / 13 split reported above is a tally of coin flips. That
includes the rows the runner wins: Apertus at p=0.250 is our best result
in the table and it is not a result. And it includes the Granite 4.2
flash-attention row that prompted this section, at two positions and
p=0.500.

The continuous measure has roughly ten times the power, because it uses
every position instead of only the ones where the argmax flips. A
Wilcoxon signed-rank test on the per-position KL difference does find
real structure:

| | families |
|---|---|
| runner genuinely closer (p<0.05) | SmolLM2, Qwen2.5 0.5B, Qwen3.5 0.8B, Gemma 3 4B, Apertus 8B, Nemotron Nano 9B, Granite 4.0-h and Granite 4.2 3B at bf16 |
| no measurable difference | Granite 4.0-h at Q4_K_M, Gemma 4 E2B at Q4_0, Qwen3 4B at Q4_K_M, Qwen3 0.6B at Q4_0, Granite 4.2 3B at Q4_K_M with `-fa on` |
| **llama.cpp genuinely closer** | **Trinity Nano (afmoe) at bf16 p=0.012 and at Q8_0 p=0.012**, StableLM p<0.001 |

Three conclusions follow, and they are not the ones the percentage
tables suggested.

1. **There is no general 4-bit deficit.** What quantization does to the
   runner is erase its advantage, not reverse it. Five served rows move
   from "genuinely closer" at bf16 to "no measurable difference" at the
   pinned quant, and none moves to "genuinely worse". The f32 activation
   and f32 attention accumulation buy a real edge that weight-rounding
   error then swamps.
2. **afmoe is a second defect, not a lost coin flip.** Trinity Nano is
   the only family besides StableLM where llama.cpp is genuinely closer,
   and it is genuinely closer **at bf16**, where no quantization is
   involved. That makes it arithmetic, in the same class as StableLM and
   an order of magnitude smaller. This report previously listed it as
   "behind at its served quant", which understated it. Filed as R4.24.
3. **The instrument needs a bigger corpus before any of this is quoted
   comparatively.** At the observed discordance rate of about 2.7% of
   qualified positions, 100 positions has a power of 0.02 to detect the
   gap it appears to show. 1,000 positions reaches 0.54 and 2,000 reaches
   0.88 for a lopsided true effect, less for a moderate one. **2,000
   positions is the floor for a comparative claim** and is a few hours of
   Blackwell time, not a research program. Filed as R6.7.9.

### StableLM is wrong, and this is the pass's most important result

On unquantized StableLM 2 1.6B weights, through the same harness, on the
same file, in the same run:

| engine | mean KL from the reference | top-1 |
|---|---|---|
| llama.cpp | **0.0007** | **100%** |
| Runner | **0.6099** | **55%** |

llama.cpp reproduces the publisher's implementation essentially exactly.
The runner does not, and not subtly: at corpus position 110 the reference
and llama.cpp both predict ` token` and the runner predicts ` fö`, a KL of
7.27 at that position; at 185 the reference says a paragraph break and the
runner says a line break; at 191 the reference says ` language` and the
runner says ` runner`. The runner loads the file as `stablelm`, 24 layers,
with no warning of any kind.

Because llama.cpp is exact on the same file through the same instrument,
this cannot be the harness and it cannot be the artifact. It is the
runner's `stablelm` path.

`stablelm` is named as supported in the README's architecture table and
has **no row in the compatibility manifest**, so no pinned file, no
tokenizer differential, no greedy check, no CPU/GPU identity check has
ever run against it. It was claimed and never gated. That is the failure
mode the compatibility program exists to prevent, and the golden pass
found it in an afternoon because it was the first time the architecture
was measured against anything other than itself.

## What we gained and what we lost

Both halves matter, so both are stated.

**Gained.**

- A measured, repeatable statement of what the runner's arithmetic
  actually does: on unquantized weights it reproduces seven publishers'
  own implementations to 0.0000 or near it, and has the lower KL from
  the reference on every one of those families. That was one family
  yesterday and is eight today.
- Ten genuine chat-template defects across four mechanisms, none of which
  any engine-to-engine comparison could have surfaced, on families that
  were all considered admitted. One of them, Qwen 3.5, is what the
  survivor suite is built around.
- One numerical defect, `stablelm`, where the runner is plainly wrong and
  llama.cpp is right.
- The structural lesson behind it: an architecture can be named as
  supported in the README with no manifest row and therefore no gate at
  all. That gap is now a rule to close, not an accident to repeat.
- A gold harness that three separate faults made honest, and which now
  refuses to read a report it did not just write.

**Lost.**

- The standing of "admitted". Eleven of twenty-nine families needed
  attention, and the ones this pass could not measure are not thereby
  fine, they are unmeasured. The word did less work than it appeared to.
- Any claim to be close to the model at the quant people serve. At 4 bits
  both engines sit 0.06 to 0.35 from the reference and are separated from
  each other by 0.004 to 0.008. The runner is the closer engine on that
  measure; it is not a close engine.
- The whole accuracy pitch on the landing page. It read that the runner
  "comes out the closer of the two engines", which was true of the
  measure this report led with and not of the measure this project gates
  on. The comparative clause went first, and then the owner removed the
  paragraph outright the same day: a claim that needs a column selected
  to hold is not a reason anyone lands on the page. The hero is back to
  what the runner is and what it records; the accuracy comparison lives
  on the evidence page with both columns beside it, where a reader who
  wants it can weigh it. The lesson is the ordinary one, and it has two
  halves: the metric that flatters you is published last, and a claim
  that only survives one metric does not belong in the first screen.
- Three rows where the runner is behind: afmoe at its served quant and on
  top-1 at both, Gemma 4 E2B at Q4_0, and `stablelm` badly.
- Roughly a day of measurement to three harness faults, one of which
  (the stale report) would have published false numbers had the corrected
  run not reproduced the uncorrected one digit for digit.
- Two families the instrument still cannot measure, diagnosed but not
  fixed: Phi-3.5 and Mistral v0.3 declare `add_space_prefix`, so the
  servers prepend a space token the reference never sees, and both read
  around 1.6 to 2.0 with the two engines agreeing to four decimals (on
  Mistral, to all four: 1.6114 and 1.6114). That is the instrument, not
  the engines, and it is filed rather than reported as fidelity. Phi-3.5
  has a second obstacle worth its own note: its publisher's modeling code
  does not run on transformers 5.16 (`DynamicCache.from_legacy_cache` is
  gone), so for that family the reference implementation the standing
  rule points at is not currently runnable at all.

## What this pass did not measure

- Phi-3.5 and Mistral v0.3, for the space-prefix reason above, and the
  Apertus bf16 tier, whose two 16 GB servers did not come up inside the
  30-minute window (its served tier did, and is in the table).
- Families whose smallest published member is too large for a float32
  reference on this box: gpt-oss (120B and 20B), Muse Glimmer 30B,
  Nemotron 3.5 Lightning 30B. gpt-oss is anchored on OpenAI's own
  evaluation guide instead; the other two keep their existing
  llama.cpp-anchored envelopes until a smaller member exists.
- The tool-calling behaviour behind the declarations (R4.12.3, R4.13.3).
- Sampling defaults from each publisher's card, which is R4.12.6 and
  needs an owner decision before the runner reads them.
