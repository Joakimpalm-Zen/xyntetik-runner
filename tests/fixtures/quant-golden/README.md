# Quantizer byte-identity fixture

`tests/test_quantize.c` writes this slice with Runner's reference writers and compares every byte with ggml's.

- `smollm2-135m-l0-ffn_down-16rows.bf16`: 16 rows (0, 1, 2, 5, 9, 11, 21, 22, 23, 26, 28, 33, 39, 48, 61, 82) of
  `model.layers.0.mlp.down_proj.weight` (576 x 1536, BF16, row-major, little-endian) from
  `HuggingFaceTB/SmolLM2-135M-Instruct` at 12fd25f77366fa6b3b4b768ec3050bf629380bac (`model.safetensors`, sha256
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`), Apache-2.0. The rows were chosen because
  an FMA-contracted build writes them differently from the reference for q4_0, q4_k and q6_k, so the test fails
  on a build that contracts.
- `ggml-b10353-<type>.bin`: what ggml's own writers (`quantize_q8_0`, `quantize_q4_0`, `quantize_q3_K`,
  `quantize_q4_K`, `quantize_q6_K`, `quantize_mxfp4`, the functions `ggml_quantize_chunk` calls, no importance
  matrix) write for the slice widened exactly to float32. They come from llama.cpp b10353 (f8def7fe)
  `ggml/src/ggml-quants.c` compiled with `-O3 -fno-fast-math -ffp-contract=off` (MIT, see NOTICE).

ggml's own K-quant bytes depend on floating-point contraction: built with multiply-adds fused, it writes the q4_k
and q6_k files here differently. The reference is the uncontracted arithmetic, which every compiler gives under
`-ffp-contract=off`.

sha256:

```
3030c6ad3a5d358f7c4657def717e6f7169ae89e986a54008db1770a75749728  smollm2-135m-l0-ffn_down-16rows.bf16
bba686455a2bd94f438568cbfced47fe1afe30f5ca181c256715b548249ce07b  ggml-b10353-q8_0.bin
884cf4f16f596781d4433e90093b2679d5c7dc1492f5d5b9339bdef94f3e0c9e  ggml-b10353-q4_0.bin
67789073bde56324150e7d6133d6bd9604637e5a81070179ea31ae4479e020ae  ggml-b10353-q3_k.bin
ba1afdbc11f640903f15a615eea65cbd510405c180bad9acfeec54a921d3dbc6  ggml-b10353-q4_k.bin
59ed9fda6db400af97011f538c6a71065f297e35b9cdd6e8e6cf91a5b08b74b9  ggml-b10353-q6_k.bin
88fdf0a94b38bdfb5f2477dab52b11d7c65ba152a798afdcfde647f7a4f82095  ggml-b10353-mxfp4.bin
```
