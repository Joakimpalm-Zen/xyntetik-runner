# MiniCPM-V 4.7's chat templates: two renderers, text path (R4.27.6, 2026-11-06)

**What is not flattering, first:**
- Text path only: the templates' image and video placeholders (`<|image_pad|>`, `<|video_pad|>`) are not rendered.
- No MiniCPM-V row is admitted here; R4.27.2 (the row) continues on these renderers.
- Two conformance cases tokenize differently from the reference: a history turn's content carries caller-spelled
  `<think>...</think>`, which Runner reads as text by design (tests/test_prompt_marks.c) and the reference reads as its
  two added tokens. The renders are byte-identical; the cases are accounted under the token-only kind (R2.4.11).
- The `minicpmv` sampling preset carries openbmb's repetition_penalty of 1.03 as published; it is the publisher's
  value, not a Runner calibration.

## The files

openbmb/MiniCPM-V-4.7-35B-A3B-gguf at cf943fbe ships two sets with byte-identical weights that differ only in the
embedded template (its README):
- the root files, "thinking disabled (fixed)", openbmb's recommended default ("not trained as a reasoning model"): the
  repository's template with the generation prompt's thinking test replaced by a comment padded to the same length, so
  the prompt is always the closed `<think>\n\n</think>\n\n` (sha256 5aaa30dc...);
- `thinking/`: openbmb/MiniCPM-V-4.7-35B-A3B's chat_template.jinja at 98e79ba9 byte for byte (7,249 bytes, 49c37bcc).

## The templates against Qwen 3.5-0.8B's

Against Qwen/Qwen3.5-0.8B's chat_template.jinja (2fc06364), the one `qwen35-nothink` renders, the repository's
template changes two things:
1. every assistant turn opens with a thought block: its reasoning when the turn is after the last user query and its
   reasoning is non-empty, otherwise the empty `<think>\n\n</think>\n\n` (Qwen 3.5 writes the turns at or before the
   last query without one);
2. content parts are joined with '\n' (Runner's message flattening already joins them so).
The thinking default (closed unless enable_thinking is true), the tool declarations, the tool-call argument rule
(Qwen 3.5's) and the content trim are the same. The root files add one: the generation prompt is always closed.

llama.cpp renders each file's own template through its jinja engine and always injects `enable_thinking`; openbmb's
README serves the repository's template through transformers' apply_chat_template.

## The change

`TMPL_MINICPMV` (`minicpmv`) and `TMPL_MINICPMV_FIXED` (`minicpmv-fixed`): `qwen35-nothink` with rule 1, and for the
root files the closed prompt. Detected before the Qwen 3.5 tests by text only these templates write (the parts join
beside the history line's empty block; the root files' comment). They join the qwen35 family's handling (tool
protocol, trim, think tags). A `minicpmv` sampling preset from openbmb's README.

## The checks, each declared before its read

- **K1:** template-conformance `--require-tokens`, families `minicpmv47` (the thinking/ Q4_K_M's own template) and
  `minicpmv47-fixed` (the root Q4_K_M's), both on the root Q4_K_M's tokenizer, with content parts, reasoning history
  and padded content: 56 cases conformant in text, 2 not expressible (a system message mid-history, which the
  reference refuses); tokens 54 identical and the two reasoning-history cases accounted as above.
- **K2a:** every other registered family's renders (52 families, 1,073 renders) byte-identical before and after.
- **K2b:** `--info tools` on the real header of every roster file (75 rows) byte-identical before and after. On the
  MiniCPM-V files' own headers the root file now reads `minicpmv-fixed` and the thinking file `minicpmv`.
