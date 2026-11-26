# Claim 1 Statistical Report (Pre-Registered Analysis)

Method: PREREG 0.2, 1.3, 1.4 (95% Wilson score CI with z=1.959963984540054; Newcombe 1998 Method 10 paired 95% CI; exact two-sided McNemar p; Erratum 2 head-shape bound 0.0001); pure Python stdlib, no third-party statistics packages.

## Per-Engine Summary

| Engine | Identical | 95% Wilson CI | Exact ties | Encoding diff | Incomplete | Flagged pos (prompts) | Verdicts depending on flagged |
|---|---:|---|---:|---:|---:|---:|---:|
| runner | 40/100 | [0.3094, 0.4980] | 0 | 0 | 58 | 0 (0) | 0 |
| unsloth | 16/100 | [0.1010, 0.2442] | 0 | 0 | 0 | 1 (1) | 0 |

## Per-Category Breakdown

| Engine | Category | Identical | 95% Wilson CI |
|---|---|---:|---|
| runner | chat | 15/25 | [0.4074, 0.7660] |
| runner | code | 21/25 | [0.6535, 0.9360] |
| runner | arithmetic | 3/25 | [0.0417, 0.2996] |
| runner | short_tool_use | 1/25 | [0.0071, 0.1954] |
| unsloth | chat | 1/25 | [0.0071, 0.1954] |
| unsloth | code | 0/25 | [0.0000, 0.1332] |
| unsloth | arithmetic | 8/25 | [0.1721, 0.5159] |
| unsloth | short_tool_use | 7/25 | [0.1428, 0.4758] |

## First Differing Token Distribution and Margin

| Engine | Non-id w/ diff | first_diff (min / p25 / med / p75 / max) | margin med (IQR) | j=0 | 1-15 | 16-63 | 64-255 |
|---|---:|---|---|---:|---:|---:|---:|
| runner | 6 | 42 / 61.25 / 67.00 / 127.50 / 148 | 1.324e+01 (1.007e+01) | 0 | 0 | 3 | 3 |
| unsloth | 84 | 2 / 13.00 / 31.00 / 54.00 / 198 | 7.692e-02 (1.276e-01) | 0 | 23 | 47 | 14 |

## Paired Comparisons and Claim Wording

### runner vs unsloth
- Paired 2x2 (n=100): n_11=1, n_10=39, n_01=15, n_00=45 (m=54)
- Difference delta = +0.2400, phi = -0.300669, Newcombe Method 10 95% CI = [+0.0990, +0.3687], exact McNemar p = 0.0015, outcome = higher
- Claim wording: On 100 pre-registered everyday prompts (greedy, max 256 tokens, reasoning_effort=low) using the exact UD-Q8_K_XL GGUF verified by teacher-forced forward against the float32 transformers reference, Runner matched the reference's complete greedy generation on more prompts than Unsloth llama.cpp: 40/100 (95% Wilson [0.309, 0.498]) vs 16/100 (95% Wilson [0.101, 0.244]; paired difference +0.240 [Newcombe 95% CI +0.099, +0.369], McNemar p=0.0015, exact ties Runner 0, Unsloth llama.cpp 0); first token divergence fell at median index 67 vs 31.
