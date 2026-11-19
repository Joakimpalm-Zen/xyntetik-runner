# gpt-oss-20b MXFP4: measured and not admitted (2026-11-19)

**Not admitted: 0.0949 mean KLD in the chat frame (fails the 0.05 bar);
0.0709 on raw text (undecided).** Admission requires the chat-frame read to
pass. Both reads are recorded on the publisher-file row in the
[support list](../../model-support.md#gpt-oss-20b-mxfp4), with no gate cells.
The keep-30 derivative is a different file and keeps its own disclosure.

## Both reads against the publisher's reference

| Input | Positions | Mean KLD | 95% interval | Margin-qualified top-1 | 95% interval | Interval-rule verdict |
|---|---:|---:|---|---:|---|---|
| Chat frame | 534 | 0.0949 | [0.0704, 0.1229] | 95.12% | [92.99%, 97.03%] | Fail |
| Raw text | 544 | 0.0709 | [0.0471, 0.1001] | 98.02% | [96.81%, 99.05%] | Undecided |

The whole chat-frame KLD interval is above the 0.05 bar. The raw-text
interval straddles it, so that read is undecided under the interval rule.
The [2026-11-13 record](../gpt-oss-20b-2026-11-13/README.md) called raw text
a failure using that release's point-estimate bar; its per-position rows
remain the raw-text evidence. The top-1 bar is 97% margin-qualified.

Paired on 534 shared positions, chat frame minus raw text is **+0.0233
[-0.0066, +0.0536]** mean KLD; the ratio is **1.33 [0.93, 1.99]**.
The frame did not help in the point estimate; the difference is unresolved.

## File, reference and route

- File: `ggml-org/gpt-oss-20b-GGUF` at
  `41e8a94d15d62070b36cf4553a0250717ecf46b8`,
  `gpt-oss-20b-MXFP4.gguf`, 12,109,566,624 bytes, sha256
  `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.
- Reference for both reads: `openai/gpt-oss-20b` at
  `6cee5e81ee83917806bbde320786a8fb61efebee`, float32 arithmetic,
  transformers 5.16.1. As described in the 2026-11-13 record, the
  publisher's MXFP4 experts dequantize exactly to bf16, then run in
  float32 at use. Both sides retain the publisher's 4-bit experts.
- Both reads share main `92ef2e92`'s release-recipe build (binary sha256
  prefix `50cca98c4747fd20`), CPU, `RUNNER_SCORE_CHUNKED=1`,
  `--gpu off`, `-c 4096`.
- Chat frame: the publisher's chat template, one user turn, the assistant's
  final channel, then the corpus text. The raw-text read uses the
  2026-11-13 rows. The paired comparison uses only their shared positions.

## Control and limits

The dense control, Qwen3-0.6B, was read raw and in its own chat template.
On 540 shared positions its framed/raw KLD ratio is **0.9717
[0.8418, 1.1256]**, which contains 1; both reads pass. This control does
not show a systematic frame effect.

For context, llama.cpp read the same gpt-oss file at 0.0672 mean KLD on
raw text in the 2026-11-13 record. That comparison does not establish the
cause of the gap. These measurements do not measure task quality or admit
the file on any backend.

This public summary records the aggregate results. The raw-text
per-position files are in the linked 2026-11-13 record; chat-frame
per-position files are not included here.
