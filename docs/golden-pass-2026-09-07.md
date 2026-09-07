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

GOLDTABLE

## What this pass did not measure

- Families whose smallest published member is too large for a float32
  reference on this box: gpt-oss (120B and 20B), Muse Glimmer 30B,
  Nemotron 3.5 Lightning 30B. gpt-oss is anchored on OpenAI's own
  evaluation guide instead; the other two keep their existing
  llama.cpp-anchored envelopes until a smaller member exists.
- The tool-calling behaviour behind the declarations (R4.12.3, R4.13.3).
- Sampling defaults from each publisher's card, which is R4.12.6 and
  needs an owner decision before the runner reads them.
