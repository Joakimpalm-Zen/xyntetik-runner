# 27B Hub card, variant B (recovery): withdrawal notice

Card: `Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF`, README.md as
of revision `4b19c412`. The file stays downloadable, so the card says why it
left the roster. The `Qwen3.8-27B-MTP-Q8_0.gguf` head in the same repository
is the publisher's weights and is unaffected: it drafts for ISTA-DASLab's
unmodified IQ3_S.

## Front matter

Add the tag `withdrawn`. Nothing else changes.

## A new block directly under the title, above `xyntetik-status`

```markdown
> **Withdrawn from the Xyntetik Runner roster on [date]. Do not use this file as a Qwen3.8-27B.**
> Against Qwen's float32 reference (`Qwen/Qwen3.8-27B` at revision `1d4bf0f2`), this file reads mean KLD 0.41, top-1 77.7-80.4% and margin-qualified top-1 85-86%, both in the house window and at positions 4,000-5,999. That is outside the fidelity bar (KLD at most 0.05, margin-qualified top-1 at least 97%). The unmodified ISTA-DASLab source file reads [lab: source KLD / margin-qualified top-1] on the same positions, so the scale recovery made the distance. On the same corpus the file predicts the text better than its parent (perplexity 21.5 against the parent's 73.3), which a faithful copy cannot do.
> The card's earlier claim, 0.0450 / 97.80%, was measured against the project's BF16 GGUF on 500 short Wikitext-2 prefixes with a 0.5-nat tie band. It holds for those positions only and is withdrawn as the file's fidelity claim.
> **Use instead:** [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`, with `Qwen3.8-27B-MTP-Q8_0.gguf` from this repository as its draft head.
```

## Status block

- The first bullet becomes: `**Research artifact, withdrawn [date]; see the notice above.**`
- The "Useful for normal inference?" bullet becomes: `no: see the notice above; use the source file`.
- The validation-status sub-bullet for the trunk becomes:
  `` `Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered.gguf`: withdrawn [date], [roster note] ``.
  The MTP head's sub-bullet stays.

## Opening paragraphs

The second paragraph (the 0.0450 / 97.80% "inside the house bar" claim)
becomes:

```markdown
**Withdrawn.** Against Qwen's float32 reference the file reads 0.41 /
85-86%, outside the bar. The source file reads [lab: source row] on the
same positions. The 500-prefix reading below is kept as the record of what
was measured at release, not as a claim.
```

## Measured envelope

Add the reference table from variant A as the first table, with the source
row filled from the lab's read. In the 500-position table, the "This file:
**PASS**" line becomes:

```markdown
Release reading (2026-09-13), superseded by the reference rows above: PASS
on these 500 short prefixes only.
```

## Limits

The new first bullet:

```markdown
- **Withdrawn: outside the fidelity bar against the parent's float32
  reference, further from it than the unmodified source file.** The scale
  recovery moved the file away from the parent on text the 500-prefix
  harness did not cover.
```

## MTP head section

The two example command lines use the source file's name
(`Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`) in place of the recovered one, with one
sentence: "Measured with the recovered trunk, withdrawn [date]; the head's
tensors are the publisher's and do not depend on the trunk's scales."

## RECOVERY.json

The record of what was done stays. Append `"withdrawn": {"date": "[date]",
"reason": "outside the fidelity bar against the float32 reference; the
source file reads [lab] on the same positions", "reference_fidelity": {...}}`.
