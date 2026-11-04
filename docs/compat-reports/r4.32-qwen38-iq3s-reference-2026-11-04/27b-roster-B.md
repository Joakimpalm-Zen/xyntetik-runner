# 27B roster row, variant B (recovery): withdrawn

The roster has no withdrawn state (`TIERS` in `scripts/model-tiers.py` is
tier1, tier2, tier3 and upcoming). The owner chooses one of two shapes, and
the tier is left blank:

1. **Remove the row.** The transition below goes into the commit message and
   the R4.32 report, and the row's tier-smoke pins are retired with
   `--retire`, which drops their manifest entries and texts.
2. **Keep the row in an unmaintained tier** with the claim below, so the
   published card's roster link still resolves.

```json
{
  "tier": "",
  "claim": "Withdrawn: outside the fidelity bar against Qwen's float32 reference (mean KLD 0.41, margin-qualified top-1 85-86%), further from the parent than the unmodified ISTA-DASLab source file ([lab]); the scale recovery moved it. Not a Qwen3.8-27B; use the source file.",
  "upstream": {
    "checkpoint": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
  },
  "envelope": {
    "fp32_reference": "measured",
    "anchor": "Against Qwen/Qwen3.8-27B@1d4bf0f2 in float32, in the house window and at positions 4000-5999: mean KLD 0.41, top-1 77.7-80.4%, margin-qualified top-1 85-86%, corpus perplexity 21.54 against the parent's 73.29. The source file reads [lab] on the same positions. The release reading (0.0450 / 97.80% against the project's BF16 GGUF on 500 short Wikitext-2 prefixes) is withdrawn as the claim.",
    "limitations": "Withdrawn [date]."
  },
  "transitions": [
    "...existing entries...",
    {
      "tier": "",
      "effective": "[date]",
      "reason": "R4.32: withdrawn. The float32 reference read is 0.41 / 85-86% against the source file's [lab], so the scale recovery took the file outside the fidelity bar. Tier: owner's decision.",
      "last_validated_release": "1.7.1"
    }
  ]
}
```

## What else moves with a withdrawal

- `qwen3.8-27b-mtp-q8_0` has `runs_with: {"row": "qwen3.8-27b-gsq-rco-iq3_s-recovered"}`.
  It needs a trunk row for ISTA-DASLab's unmodified `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`
  (sha256 `64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810`,
  11,771,546,784 bytes, repo revision `d562806d`), with the reference read
  as its anchor. Otherwise the head row goes with this one.
- The pins under `tests/compatibility/tier-smoke/`: this row's linux and
  windows CUDA pins, the head's composed linux CUDA pin, and the
  `docs/model-support.md` section, which is regenerated from the roster.
- The cross-family remedy evidence files that name this file stay as dated
  records.
