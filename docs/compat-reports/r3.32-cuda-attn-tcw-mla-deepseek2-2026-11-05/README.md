# R3.32: deepseek2's absorbed prefill on CUDA's tensor cores, and the re-pinned CUDA cells (2026-11-05)

**What is not flattering, first:**
- **Stale cells on main:** both deepseek2 roster rows' CUDA tier-smoke cells were stale on main from #631
  (7504d0a8) until this change. R3.31's decode kernel moved their replies, and #631 did not compare them against
  their pins. No release was cut in that window. This change re-pins both cells, and the decode kernel's own D1
  read (below), missing from #631, passes.
- **Scope:** one file family (DeepSeek-V2-Lite-Chat Q6_K and Q8_0), one host (a CUDA MIG slice, 1g.24gb,
  sm_120), contexts up to 8K for the speed read, and f16 caches only.
- **Not here:** quantized caches keep the previous path. So do Metal and the CPU.

The kill was declared before the first read: absorbed prefill under 0.9x of the naive form's at 4K or 8K, or any
identity miss against the pinned CPU arm.

## The change

The absorbed form keeps one 576-wide [k_pe 64 | latent 512] row per token that all 16 query heads share. That width
is past every tiled prefill kernel (tensor cores up to 512 in quarters of 128, the tile up to 512), so its prompt
tiles ran `k_attn`: one block per head and column, every cached row read again for each head.
- **The kernel:** `k_attn_tcw_mla` is `k_attn_tcw`'s twin (R3.6.3.1). S runs over the whole row, V is the staged
  row's latent (the one table, R3.31, so no V tile is staged), and the 16 heads are a block's 16 rows. That is the
  packing of llama.cpp's MMA f16 kernel at <576, 512, 16>.
- **The fit beside the kill:** from R3.31's prompt rates (time per token = c + a x mean context), absorbed attention
  cost 7.7x naive's per unit of context, at 3.4x its multiply-adds. To pass, it had to come down to 2.4x at 4K and
  1.7x at 8K.

## The kill's read

DeepSeek-V2-Lite-Chat Q6_K (mradermacher, sha256 `3f634b8f08f4ef10...`), 64 tokens at temperature 0, back to back.
Prompt tok/s:

| Context | Absorbed, new | Absorbed, before | Naive | New / naive (bar 0.9) | Load average (1 min) |
|---|---|---|---|---|---|
| 4,096 | 201.64 | 114.69 | 191.57 | **1.053** | 10.1 / 8.8 / 13.5 |
| 8,192 | 192.43 | 78.24 | 173.38 | **1.110** | 10.1 / 7.9 / 12.1 |

- **Against the fit:** the new kernel costs 0.12 ms per 1K of context, 0.44x naive's.
- **Decode:** unchanged, 100.04 against 100.02 tok/s at 4K.
- **Outputs:** byte for byte the previous path's at both lengths.

## Identity

- `--mla-wide` fixture, gpu-identity at 512 tokens: 1.55e-6 of the logit range (limit 1e-5).
- The Q6_K with routing pinned: 1.45e-5 (limit 1e-3).
- **Mutants**, each alone with the PTX rebuilt by the pinned toolchain:
  - V read from the row's start: 0.0575 of the range, caught.
  - The rescale dropped: 3.35e-4, caught.
  - The restored build's PTX and binary are the passing ones by sha256.
- `k_attn_tcw`'s existing families (256 and 512 wide, G 2 and 12) write the same score bytes as before the change.

## The D1 reads (the attention routes' rule, 2026-10-20)

- **Reference:** `deepseek-ai/DeepSeek-V2-Lite-Chat` at 85864749, the publisher's float32 model code
  (transformers 5.16.1, bf16 weights cast to float32 at use).
- **Positions:** 4,000 to 5,999 at stride 1, BOS forced: 1,999 positions, every one matched by id, no dropped mass.
- **Score files:** the Q6_K on the slice, 6,002 tokens.
  - The prefill pair is chunked (64-row tiles), so every position goes through the prefill route.
  - The decode pair is unchunked, so every position goes through decode attention.

| Route | Mean KL before | Mean KL after | Ratio (at most 1.01) | Wilcoxon (not worse) | McNemar (not worse) |
|---|---|---|---|---|---|
| Prefill: `k_attn_tcw_mla` against `k_attn` | 0.040823 | 0.040243 | 0.986 | z 0.27, p 0.79 | 3 / 2, p 1.0 |
| Decode: `k_attn_dec_mla` (R3.31) against `k_attn_dec` | 0.040741 | 0.040371 | 0.991 | z -0.89, p 0.37 | 1 / 1, p 1.0 |

Both pass, so the prefill kernel is deepseek2's default and the decode kernel stays the default.

## The roster's CUDA cells

After the change both deepseek2 rows' CUDA replies differed from their pins, with "Mercury" still found. Bisected on
the Q6_K with tier-smoke, no write:

| Build and switches | Reply sha256 | Pin |
|---|---|---|
| This change, prefill switch off (`RUNNER_CUDA_ATTN_TC_WIDE=0`) | `eae8d81a` | mismatch |
| This change, prefill off and the old decode kernel (`RUNNER_CUDA_ATTN_MLA=0`) | `70827d50` | match |
| main at #631, default | `eae8d81a` | mismatch |
| main at #631, the old decode kernel | `70827d50` | match |
| R4.4.5 (d)'s build, which made the pin | `70827d50` | match |

So R3.31's decode kernel moved the reply, and this change's prefill route did not. With the decode kernel's D1 read
passing, both cells are retired, pinned and recorded again on this change's merged build.

| Row | CUDA reply before (old pin) | Re-pinned reply | No-write check after | CPU cell |
|---|---|---|---|---|
| `deepseek-v2-lite-chat-q6_k` | mismatch (pin `70827d50`) | `eae8d81a` | match | match, unchanged |
| `deepseek-v2-lite-chat-q8_0` | mismatch (pin `eae8d81a`) | `cedb5bd6` | match | match, unchanged |

"Mercury" was found in every reply. The cells now name runner 1.13.0 on 2026-11-05.
