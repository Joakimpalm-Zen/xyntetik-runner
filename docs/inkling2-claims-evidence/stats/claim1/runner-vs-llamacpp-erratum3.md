# Claim 1 Statistical Report (Pre-Registered Analysis)

Method: PREREG 0.2, 1.3, 1.4 (95% Wilson score CI with z=1.959963984540054; Newcombe 1998 Method 10 paired 95% CI; exact two-sided McNemar p; Erratum 2 head-shape bound 0.0001); pure Python stdlib, no third-party statistics packages.

## Per-Engine Summary

| Engine | Identical | 95% Wilson CI | Exact ties | Encoding diff | Incomplete | Flagged pos (prompts) | Verdicts depending on flagged |
|---|---:|---|---:|---:|---:|---:|---:|
| runner | 94/100 | [0.8752, 0.9722] | 0 | 0 | 0 | 0 (0) | 0 |
| llama | 20/100 | [0.1334, 0.2888] | 0 | 1 | 0 | 0 (0) | 0 |

## Per-Category Breakdown

| Engine | Category | Identical | 95% Wilson CI |
|---|---|---:|---|
| runner | chat | 25/25 | [0.8668, 1.0000] |
| runner | code | 24/25 | [0.8046, 0.9929] |
| runner | arithmetic | 25/25 | [0.8668, 1.0000] |
| runner | short_tool_use | 20/25 | [0.6087, 0.9114] |
| llama | chat | 1/25 | [0.0071, 0.1954] |
| llama | code | 0/25 | [0.0000, 0.1332] |
| llama | arithmetic | 7/25 | [0.1428, 0.4758] |
| llama | short_tool_use | 12/25 | [0.3003, 0.6650] |

## First Differing Token Distribution and Margin

| Engine | Non-id w/ diff | first_diff (min / p25 / med / p75 / max) | margin med (IQR) | j=0 | 1-15 | 16-63 | 64-255 |
|---|---:|---|---|---:|---:|---:|---:|
| runner | 6 | 42 / 61.25 / 67.00 / 127.50 / 148 | 1.324e+01 (1.007e+01) | 0 | 0 | 3 | 3 |
| llama | 80 | 2 / 15.50 / 35.50 / 74.00 / 252 | 6.419e-02 (1.137e-01) | 0 | 20 | 36 | 24 |

## Paired Comparisons and Claim Wording

### runner vs llama
- Paired 2x2 (n=100): n_11=18, n_10=76, n_01=2, n_00=4 (m=78)
- Difference delta = +0.7400, phi = -0.084215, Newcombe Method 10 95% CI = [+0.6257, +0.8164], exact McNemar p = 2.039e-20, outcome = higher
- Claim wording: On 100 pre-registered everyday prompts (greedy, max 256 tokens, reasoning_effort=low) using the exact UD-Q8_K_XL GGUF verified by teacher-forced forward against the float32 transformers reference, Runner matched the reference's complete greedy generation on more prompts than llama.cpp: 94/100 (95% Wilson [0.875, 0.972]) vs 20/100 (95% Wilson [0.133, 0.289]; paired difference +0.740 [Newcombe 95% CI +0.626, +0.816], McNemar p=2.039e-20, exact ties Runner 0, llama.cpp 0); first token divergence fell at median index 67 vs 35.5.
