# 27B Hub card, variant A (GSQ-RCO): replacement blocks

Card: `Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF`, README.md as
of revision `4b19c412`. Only the blocks below change. Every section not
named here stays as it is.

## Front matter

No change.

## Status block (replaces the `xyntetik-status` block's first four bullets)

```markdown
> **Research artifact, not a general-purpose release. Read the first bullet before using it.**
> - **Limit first: this file is not a faithful copy of its parent.** Against Qwen's float32 reference (`Qwen/Qwen3.8-27B` at revision `1d4bf0f2`), it reads mean KLD 0.41, top-1 77.7-80.4% and margin-qualified top-1 85-86%, both in the house window and at positions 4,000-5,999. That is outside the fidelity bar (KLD at most 0.05, margin-qualified top-1 at least 97%). It is a learned post-training quantization fitted on calibration data. On the same corpus it predicts the text better than its parent does (perplexity 21.5 against the parent's 73.3), so its next-token distribution is a different one, not a noisier copy of the parent's. ISTA-DASLab's unmodified source file reads [lab: source KLD / margin-qualified top-1 / perplexity] on the same positions, so the divergence comes from the source quantization, and scale recovery did not cause it.
> - **Earlier anchor, kept as what it measured:** against the project's BF16 GGUF of the same parent, on 500 short held-out Wikitext-2 prefixes with a 0.5-nat tie band, mean KLD 0.0450 and margin-qualified top-1 97.80% (plain top-1 87.20%). That number holds for those positions only. It does not describe the file on longer or other text, and it is not the reference reading above.
> - **What was changed:** only the fp16 `d`/`dmin` scale fields of every quantised block in 399 tensors, retrained by distillation against the BF16 parent. Integer codes, format, byte length and offsets are identical to the source file.
> - **Parent model:** [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (IQ3_S), from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2` (the parent's only weight set; the file's unquantized tensors match it bit for bit).
> - **Useful for normal inference?** As a small IQ3_S of Qwen3.8-27B, with the limit above: it serves wherever the source file does, but it is not a stand-in for the parent's distribution.
```

The remaining bullets (experiment, evidence, Runner compatibility, MTP head,
validation status) stay.

## Opening paragraphs (replace the two paragraphs after the actions block)

```markdown
ISTA-DASLab's GSQ-RCO IQ3_S file of Qwen3.8-27B, with every quantised
block's fp16 scales retrained end to end against the BF16 parent by knowledge
distillation on a half-prose corpus. **Same integer codes, same format, same
11,771,546,784 bytes, same loader path as the source file.** The only bytes
that changed are the `d`/`dmin` scale fields.

**Against Qwen's float32 reference it is outside the fidelity bar:** mean
KLD 0.41 and margin-qualified top-1 85-86% in the house window and at
positions 4,000-5,999, with a corpus perplexity of 21.5 against the
parent's 73.3. GSQ fits codes and scales on calibration data, and the file
tracks that data, not the parent. The unmodified source file reads
[lab: source row] on the same positions. The earlier anchor (0.0450 / 97.80%
against our BF16 GGUF on 500 short Wikitext-2 prefixes) is in the envelope
below, under the reference rows.
```

## Measured envelope (a new first table, before "Runner fidelity")

```markdown
**Against Qwen's float32 reference** (`Qwen/Qwen3.8-27B@1d4bf0f2`, the
house window and positions 4,000-5,999, measured 2026-11-04):

| vs the float32 reference | mean KLD | top-1 | margin-qualified top-1 | corpus perplexity |
| --- | ---: | ---: | ---: | ---: |
| BF16 GGUF of the parent (engine check) | 0.00002 | 100% | 100% | 73.29 |
| source GSQ-RCO IQ3_S, unmodified | [lab] | [lab] | [lab] | [lab] |
| **this file** | **0.41** | **77.7-80.4%** | **85-86%** | **21.54** |

House bar: KLD at most 0.05 and margin-qualified top-1 at least 97%. This
file: **outside the bar.** A perplexity below the parent's means the file
predicts this corpus with a different distribution from the parent's. It
does not mean the file is better than the parent.

**Earlier anchor, kept as what it measured** (500 short held-out
Wikitext-2 prefixes, against the project's BF16 GGUF, tie band 0.5 nats):
```

Then the existing 500-position table follows unchanged. Its "House bar ...
**PASS**" sentence becomes:

```markdown
On these 500 short prefixes the file is inside the bar. That reading does
not carry to the reference rows above, and the reference rows are the
file's fidelity claim.
```

## Limits (a new first bullet; one bullet changes)

```markdown
- **Outside the fidelity bar against the parent's float32 reference** (0.41 /
  85-86%), and below the parent's perplexity on that corpus. Quote the
  reference rows, never the 500-prefix anchor alone.
```

The "Thin margin on agreement" bullet becomes:

```markdown
- **The 97.80% is a short-prefix, margin-qualified reading** against our BF16
  GGUF (plain top-1 87.20%). It was inside a 97% bar on 500 positions and is
  superseded by the reference rows.
```

## RECOVERY.json

No change to the record of what was done. Append a `reference_fidelity`
object with the lab's numbers and date, and set `claims.passes_house_bar`
to `"superseded: outside the bar against the float32 reference (2026-11-04)"`.
