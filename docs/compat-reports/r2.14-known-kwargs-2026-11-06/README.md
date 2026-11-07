# R2.14: known `chat_template_kwargs` keys, against the references

**Result: no known key is silently dropped on the 33 families compared.** For each family and key, the reference
template and Runner rendered one history with the key absent and set, and the two changes were compared. Before: five
family and key pairs moved the reference and not Runner. After: none, and all 25 changes both sides make are the same bytes.

**Limits, first:**
- **19 of 52 families were not compared.** 17 keep their template in a local GGUF not on this host; harmony and raw have
  no Jinja source. They are covered by the refusal, which reads the file's own text at request time, not by a
  measurement. `audit.py` runs on any host with the model shelf.
- **One history, no tools.** A key that only matters with tool calls was not exercised, so `bare_recipients` shows no
  change. No reference reads it: it is Runner's own.
- **`preserve_thinking: null` is a known difference.** Runner counts null as absent. Qwen 3.8's `is undefined` is false
  for null, so its template behaves as for `false`. Left as it is, as null means absent for every key.
- **Apertus renders `Deliberation: enabled` now.** Its reasoning is not split from the answer.

## What the audit found

| Family | Key | Reference | Runner before | Now |
|---|---|---|---|---|
| lfm2, lfm2-24b | `preserve_thinking: true` | replays past thoughts | ignored | honoured |
| qwen38 | `preserve_thinking: false` | drops thoughts before the last query | ignored | honoured |
| apertus | `enable_thinking: true` | `Deliberation: enabled` | ignored | honoured |
| granite42 | `reasoning_effort: low` | `{reasoning effort: low}` on the last user turn | ignored | honoured |
| spark | `reasoning_effort: xhigh` | raises | served as high | refused |
| 30 of 33 | `add_generation_prompt: false` | no assistant opener | always opened | refused |

**A second axis the audit's reading found:** surfaces. Only Chat Completions reads `preserve_thinking`, and Messages
reads no effort, so those were dropped on families where chat honours them. They are refused there now, where the
template reads the key.

**Honoured beyond the reference, kept and now documented:** Hermes 4 (`enable_thinking` to its `thinking`), Nemotron
(`enable_thinking: false` to the `/no_think` state; its `ns.enable_thinking` is a namespace field, not the kwarg), and
Muse (`enable_thinking` to its reasoning turn, `reasoning_effort` to `reasoning_strength`). Each is an explicit request
served through the family's native control, not a dropped one.

## The rule

A key takes effect if the family's renderer acts on it on that surface (measured for all 43 renderers, `honours.py`).
Otherwise the file's template text decides: read as a variable means a 400, not read means ignored, as the reference
does. So a fine-tune whose template reads a known key its family ignores is refused, with no list to maintain.

## The gates

- `tests/test_known_kwargs.py`, 30 tests, expected bytes taken by hand from the published templates. 15 failed before.
- `tests/test_template_kwargs.py`, 37 tests, through a real server, including the surface refusals.
- `scripts/template-conformance.py --network-only`: conformant, 33 families, text only (no tokenizers on this host).
- `--cites-only`, `check-ci-roster`, `check-fpclass`, `help-parity`, `test-template`: clean.
- Related suites and `tests/conformance`: 652 passed, 0 failed. Full `make test` was not run.

## Files

- `audit.py`, `honours.py`: the measurements. Paths come from `R214_ROOT`.
- `audit-before.log`, `audit-after.log`: the audit on either side of the change.
