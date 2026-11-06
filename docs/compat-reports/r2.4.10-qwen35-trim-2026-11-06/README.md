# The qwen35-family templates trim message content (R2.4.10, 2026-11-06)

**What is not flattering, first:**
- Runner had never trimmed message content for these templates, and no conformance case carried padded content, so
  every release served a padded turn with whitespace the references strip. Found by R4.27.6's MiniCPM-V check, whose
  reasoning-history case was the first to pad an assistant turn.
- The change is scoped to the four templates measured here (ornith, qwen35, qwen35-nothink, qwen36). Other families'
  references may trim too; their padded cases are not written yet.
- The last-query scan of these templates also trims a user turn before it tests for `<tool_response>`; Runner keys the
  last user turn on the role alone. Unchanged here.

## The references

Seven templates, read from the Hub, share the same six trim sites: the system message (with and without tools), the
last-query scan, every message's content, `reasoning_content`, and the call separator (`if content|trim`):
Qwen/Qwen3.5-4B (851bf6e8), Qwen/Qwen3.5-0.8B (2fc06364), Qwen/Qwen3.5-9B (c2022362, Clef-Flash's bytes),
Qwen/Qwen3.6-35B-A3B (995ad96e, empero-ai's distill's bytes), Qwen/Qwen3.6-27B (6a9e13bd), ornith-ai/Ornith-1.0-9B
(83dc1f5e), and the template embedded in ornith-ai/Ornith-1.5-35B-A3B-GGUF (12393612). jinja's `trim` is python's
`str.strip()`: every code point `str.isspace()` calls whitespace.

## The change

- `py_strip` (src/template.c) narrows a span to the text between python whitespace at either end, on the set Runner
  already used for K2-Horizon.
- chat/completions (server.c, message_text), /v1/responses and /v1/messages trim a message's text, its reasoning, a
  tool result, `instructions` and `system` where they flatten them, before the thought block, the calls or the
  result wrap is added, for the ornith-like templates only.

## The checks, each declared before its read

- **K1:** template-conformance `--require-tokens`, two new house cases for families that set `trims_content`
  (padded-content: system, user and assistant history; padded-tool-turn: padded reasoning, a call and a padded
  result), ASCII whitespace with U+00A0 and U+3000. On ornith, ornith15-35b-a3b, qwen35-4b, qwen35-0.8b, clef-flash,
  qwen36-35b-a3b, qwen36-27b and empero-qwen38-35b-a3b-distill: 201 cases, 201 identical in text and tokens.
  - With the trim switched off, exactly the padded cases fail (+11 and +7 tokens on qwen36-35b-a3b and ornith).
- **K2a:** every registered family's house matrix as it was (50 families, 1,057 renders, 784 with the family's own
  template), main's render driver against this change's: byte-identical. With the padded cases included, exactly
  those 16 differ.
- **K2b:** tests/test_tool_attribution.c, `test_padded_content_contract`: the same tool conversation padded round the
  system text, the user text and the tool result renders as the plain one on chat/completions, /v1/responses and
  /v1/messages for ornith, qwen35 and qwen36; on ChatML, whose reference does not trim, the padding stays. With the
  trim off, the nine trimming checks fail.
- **K2c:** `--info tools` on the real header of every roster file (75 rows, 79 files): byte-identical before and
  after.
