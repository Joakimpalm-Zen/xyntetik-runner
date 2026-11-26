# Claim 1 Statistical Report (Pre-Registered Analysis)

Method: PREREG 0.2, 1.3, 1.4 (95% Wilson score CI with z=1.959963984540054; Newcombe 1998 Method 10 paired 95% CI; exact two-sided McNemar p; Erratum 2 head-shape bound 0.0001); pure Python stdlib, no third-party statistics packages.

## Per-Engine Summary

| Engine | Identical | 95% Wilson CI | Exact ties | Encoding diff | Incomplete | Flagged pos (prompts) | Verdicts depending on flagged |
|---|---:|---|---:|---:|---:|---:|---:|
| runner | 40/100 | [0.3094, 0.4980] | 0 | 0 | 58 | 0 (0) | 0 |
| llama | 20/100 | [0.1334, 0.2888] | 0 | 1 | 0 | 0 (0) | 0 |

## Per-Category Breakdown

| Engine | Category | Identical | 95% Wilson CI |
|---|---|---:|---|
| runner | chat | 15/25 | [0.4074, 0.7660] |
| runner | code | 21/25 | [0.6535, 0.9360] |
| runner | arithmetic | 3/25 | [0.0417, 0.2996] |
| runner | short_tool_use | 1/25 | [0.0071, 0.1954] |
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
- Paired 2x2 (n=100): n_11=0, n_10=40, n_01=20, n_00=40 (m=60)
- Difference delta = +0.2000, phi = -0.408248, Newcombe Method 10 95% CI = [+0.0494, +0.3392], exact McNemar p = 0.0135, outcome = higher
- Claim wording: On 100 pre-registered everyday prompts (greedy, max 256 tokens, reasoning_effort=low) using the exact UD-Q8_K_XL GGUF verified by teacher-forced forward against the float32 transformers reference, Runner matched the reference's complete greedy generation on more prompts than llama.cpp: 40/100 (95% Wilson [0.309, 0.498]) vs 20/100 (95% Wilson [0.133, 0.289]; paired difference +0.200 [Newcombe 95% CI +0.049, +0.339], McNemar p=0.0135, exact ties Runner 0, llama.cpp 0); first token divergence fell at median index 67 vs 35.5.
