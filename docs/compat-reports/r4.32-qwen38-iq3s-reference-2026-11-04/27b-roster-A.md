# 27B roster row, variant A (GSQ-RCO)

Row `qwen3.8-27b-gsq-rco-iq3_s-recovered` in `docs/model-tiers.json`. The
fields shown change and every other field stays. The tier is the owner's.

```json
{
  "tier": "",
  "claim": "Outside the fidelity bar against Qwen's float32 reference (mean KLD 0.41, margin-qualified top-1 85-86%): ISTA-DASLab's learned IQ3_S quantization with every quantised block's fp16 scales retrained, text path. It predicts the reference corpus better than its parent (perplexity 21.5 against 73.3), so it is not a faithful Qwen3.8-27B.",
  "upstream": {
    "checkpoint": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
  },
  "envelope": {
    "fp32_reference": "measured",
    "anchor": "Against Qwen/Qwen3.8-27B@1d4bf0f2 in float32, in the house window and at positions 4000-5999: mean KLD 0.41, top-1 77.7-80.4%, margin-qualified top-1 85-86%, corpus perplexity 21.54 against the parent's 73.29; a fresh BF16 GGUF of the parent reads 0.00002 / 100% on the same positions. The unmodified source file reads [lab]. Earlier anchor, under the reference: 0.0450 / 97.80% margin-qualified against the project's BF16 GGUF on 500 short Wikitext-2 prefixes.",
    "limitations": "Outside the fidelity bar against the publisher's float32 reference. It is a learned quantization fitted on calibration data, and its perplexity on the reference corpus is below the parent's. The 500-prefix anchor does not describe longer or other text.",
    "evidence": [
      "docs/compat-reports/r4.32-qwen38-iq3s-reference-2026-11-04/README.md",
      "docs/granite-42-qwen38-cert-2026-09-06.md",
      "docs/golden-pass-2026-10-22.md",
      "tests/compatibility/coverage.json"
    ]
  },
  "transitions": [
    "...existing entries...",
    {
      "tier": "",
      "effective": "[date]",
      "reason": "R4.32: the float32 reference read (0.41 / 85-86%) puts the file outside the fidelity bar; the source quantization reads [lab], so the limit is GSQ-RCO's. Tier: owner's decision.",
      "last_validated_release": "1.7.1"
    }
  ]
}
```

The MTP head row (`qwen3.8-27b-mtp-q8_0`) keeps `runs_with` on this row.
Its claim does not change, because the head's text is the trunk's.
