# Quantizer byte-identity fixture

`tests/test_quantize.c` writes this slice with Runner's reference and weighted writers and compares every byte with ggml's.

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

- `ggml-c1d0e7a-q5_0.bin`: what ggml's `quantize_row_q5_0_ref` (the function `quantize_q5_0` calls without an
  importance matrix) writes for the same slice. Its text was taken from llama.cpp c1d0e7a `ggml/src/ggml-quants.c`
  unchanged, with `GGML_FP32_TO_FP16` as the hardware half conversion, and compiled with `-O3 -fno-fast-math
  -ffp-contract=off` (MIT, see NOTICE). Q5_0 is the 32-block fallback of a Q4_K target.
- `ggml-c1d0e7a-{q5_1,q5_k,iq4_nl}.bin` (R4.33): what ggml's `quantize_q5_1`, `quantize_q5_K` and `quantize_iq4_nl`
  write for the same slice with no importance matrix. Those reach `quantize_row_q5_1_ref`, `quantize_row_q5_K_ref`
  and `quantize_row_iq4_nl_impl(32, 32, ..., 7)`. They come from llama.cpp c1d0e7a's `ggml/src/ggml-quants.c`
  compiled whole with `-O3 -fno-fast-math -ffp-contract=off -mf16c` (gcc 13.3), beside a harness that widens the
  slice to float32 and stubs the four ggml.c symbols the file links against (MIT, see NOTICE). Built with
  multiply-adds fused (`-ffp-contract=fast -mfma`), ggml writes these three files byte for byte the same on these
  rows. So the slice does not guard contraction for these types; the q4_k and q6_k files still do.

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
8a95b5d1cdf2748bafbbffab8ee5e12d649fb35aace4474a93a65fd584447b45  ggml-c1d0e7a-q5_0.bin
3db45d3fc28d59456fcef1fae2cc304dd29e72838872697e8b462d7657ffcfa4  ggml-c1d0e7a-q5_1.bin
a13633339e21eb3b7713dab1e53d05ed2f9f9515fa6b384410d5ec8c3762216a  ggml-c1d0e7a-q5_k.bin
e5ed0cd27ed13825715886f697e4f93aa694a8c6d5e3a5575525be908f2be129  ggml-c1d0e7a-iq4_nl.bin
```

## Importance-weighted pins (R4.34.3)

`ggml-c1d0e7a-weighted-{q4_k,q5_k,q6_k,q5_1,iq4_nl}.bin` comes from the whole, unmodified
[`ggml/src/ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/ggml/src/ggml-quants.c)
at `c1d0e7a004015f23bc0233470b747b596f29b264` (source sha256
`07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d`), MIT. The reference was built
with Clang, `-O2 -fno-fast-math -ffp-contract=off`. Its `quantize_q4_K`, `quantize_q5_K`,
`quantize_q6_K`, `quantize_q5_1` and `quantize_iq4_nl` consume the same exact float32 widening of the
16 BF16 rows above. No Runner quantization code participates in generating these bytes.

Every row uses this stated skewed float32 importance vector, for column `c` from 0 through 1535
(`/` below is integer division):

```text
w[c] = 0  if c % 64 < 16
       50 if (c + 5 * (c / 256)) % 16 < 3
       1  otherwise
```

The zero 16-column groups exercise zero-weight fits. Heavy columns rotate between 256-value blocks,
so reusing the first block's weights for later blocks is observable.

The matching `-edges.bin` files use `weighted-edges.f32`: eight deterministic rows, each 1536 float32
values, little-endian, with the same importance vector. `generate-weighted.c` defines the exact input:
zero; constant +1; constant -1; alternating signed endpoints; tiny scales; large scales; mixed zero and
unequal-scale groups; and exact half steps with sparse outliers. These protect zero/underflow scales,
code extremes, ties, and offsets across 16-, 32- and 256-value groups beside the real tensor.

`generate-weighted.c` supplies only the four ggml.c link dependencies (row/type sizes, type name,
and abort). It first reproduces all five existing unweighted pins before generating weighted ones.
To regenerate, set `GGML_SRC` to a clean llama.cpp checkout at the pinned commit, then run from Runner:

```sh
test "$(git -C "$GGML_SRC" rev-parse HEAD)" = c1d0e7a004015f23bc0233470b747b596f29b264
mkdir -p .build/golden
cc -O2 -std=gnu11 -fno-fast-math -ffp-contract=off \
    -I"$GGML_SRC/ggml/include" -I"$GGML_SRC/ggml/src" \
    tests/fixtures/quant-golden/generate-weighted.c "$GGML_SRC/ggml/src/ggml-quants.c" \
    -lm -o .build/golden/generate-weighted
.build/golden/generate-weighted tests/fixtures/quant-golden
```

The test also checks that GGUF `in_sum2 / counts` calibration agrees with the legacy `.dat` matrix,
that all-zero calibration retains unweighted bytes, and that a wrong matrix width fails the rewrite.
The original unweighted pins remain unchanged.

Weighted fixture sha256:

```text
a8c80df47a3ad2b831ce39278d57db4a21e0f34fc7974e5f86c0f9875dabfc17  ggml-c1d0e7a-weighted-iq4_nl-edges.bin
4b9c143e3432ecde531575a1e59c60b5847b66748995b044d6e28571837ce022  ggml-c1d0e7a-weighted-iq4_nl.bin
f61b57ce643dbc1360ce10e7aa85d5272670aeea42545e18dfa33a4f01fd6629  ggml-c1d0e7a-weighted-q4_k-edges.bin
b27e5cd31b89c8b3eb65fd9e322edd05b28e175ee8dcb67c6ba91fa8418aa1fb  ggml-c1d0e7a-weighted-q4_k.bin
326f6d67797b3ab675ed6a02d51242f693126c07db95adf550bed328dcdfc923  ggml-c1d0e7a-weighted-q5_1-edges.bin
3fc51e729ce082297390f387a0a8edb6e090a79f14412c808a193cab07b55b42  ggml-c1d0e7a-weighted-q5_1.bin
7cbc7d4da8525ae19a80cb6b5c0ede335774bb9c11b9ce95828768c1afe24a0f  ggml-c1d0e7a-weighted-q5_k-edges.bin
afae0adb92b30422785040b44c98c0d40d4d2893593c25faffd03a7a91e97c0c  ggml-c1d0e7a-weighted-q5_k.bin
63de1e3f792e3db92e28c77f4923bcba1d4fa9265d5e29718b323878d4099e55  ggml-c1d0e7a-weighted-q6_k-edges.bin
d4a119433d4b5377df739ab322e6ab644c4aea49159fc4c4be7397ba9cb3a3fc  ggml-c1d0e7a-weighted-q6_k.bin
e45284d01bebda2fd10d0b4850f5c20a4d9521d8eddce98e62652bb7d356b7fe  weighted-edges.f32
```
