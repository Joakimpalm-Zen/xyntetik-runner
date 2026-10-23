# The decision record

A constrained turn (a JSON schema, a tool call under a grammar, JSON mode) can
say how sure the model was at each choice the constraint left it. This page
is the contract for that report, version 1; the machine-readable form is
[`schemas/decision-record.v1.json`](schemas/decision-record.v1.json).

## Two shapes

**A point** is one constrained step at which at least two of the probed
candidates were legal. `choice_logprobs: true` lists every point of a
buffered turn in `choices[0].choice_logprobs`:

| field | meaning |
|---|---|
| `index` | generated-token index of the step |
| `n_legal` | legal candidates among the probed ones |
| `n_probed` | candidates probed (`choice_logprobs_probe`, 32 by default, at most 64) |
| `coverage` | whole-vocabulary probability mass of the probed candidates |
| `legal_mass` | whole-vocabulary probability mass of the legal probed candidates: the number each `prob` was divided by |
| `alternatives` | the legal candidates, most probable first, at most 8: `token`, `id`, `prob` (posterior over the legal probed set), `logprob` (over the whole vocabulary) |

`n_legal < n_probed` means the grammar removed some candidates: the step was
a choice between the schema's branches (which tool, which enum value, a key).
`n_legal == n_probed` is a step inside free text, where the grammar allowed
everything probed and the step is the model's wording.

**A turn** is the summary `confirm_below: p` asks for, in
`runner_telemetry.decision`, on Chat Completions, Responses and Messages:

| field | meaning |
|---|---|
| `schema` | `xyntetik.runner.decision.v1` |
| `confirm_below` | the threshold the request gave |
| `decisions` | grammar-shaped points in the turn |
| `min_chosen_prob` | the lowest posterior the CHOSEN token had at one of them (a chosen token outside the probed set counts as 0); null with no decision |
| `min_margin` | that token's posterior minus the best other legal one's |
| `at_token` | the generated-token index where it fell |
| `needs_confirmation` | `min_chosen_prob < confirm_below` |

## What reads it

The same record serves four consumers, which is why it is one contract:

- a client that routes on it (confirm before acting when
  `needs_confirmation`);
- a validator that auto-approves a call only above a margin;
- labeling: a person marks whether the chosen branch was right, and
  `scripts/cl-calibration.py` turns labeled records into accuracy, Brier and
  ECE;
- a per-model threshold derived from that calibration. Not shipped: the
  threshold today is the caller's, and a threshold written into a model's
  envelope waits for a labeled corpus.

## Limits

The posteriors are over the probed candidates, not the whole vocabulary:
`coverage` says how much of the distribution the probe saw, and `legal_mass`
how much of it the grammar left. They differ when the model's own next token
is outside the grammar: on the Windows CUDA host, Qwen3-4B Q4_K_M after
writing `"E` under an enum holding `"E"` and `"EAT"` put `coverage` 1.0 but a
`legal_mass` near 1e-4 on the legal continuations, so its posterior of 0.84
for `AT` was a forced choice between two strings it barely considered
([the measurements](enum-prefix-2026-10-23.md)). A low `legal_mass` says the
schema, not the model, made that choice. They are the
model's, at the temperature-free logits, not a calibrated probability of
being right; calibration is a measurement on labeled decisions, per model.
A native protocol's auto turn is parsed rather than constrained and has no
points to report.
