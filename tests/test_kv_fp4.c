// The --kv fp4 cache codec against its own inverse and against the
// dequantised dot: the block scale is the UE4M3 ceiling of amax/6 so no code
// clips, nibbles decode through the same E2M1 table as NVFP4 weights, and the
// fused score and accumulate equal the dequantise-then-multiply form. The
// CUDA and Metal store kernels mirror this arithmetic, so this is the
// host-side reference every backend's cache bytes are held to.
#include "../src/quants.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static uint32_t st = 0x2545F491u;
static float rnd(void) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return (float)(int32_t)st / 2147483648.0f; }

int main(void) {
    // 1. the scale encoder is a true ceiling: value(code) >= s and the code
    //    below it is < s, for every representable value and points between
    for (int c = 0; c < 0x7F; c++) {
        float v = kv_ue4m3_to_f32((uint8_t)c);
        uint8_t back = kv_ue4m3_ceil(v);
        CK(kv_ue4m3_to_f32(back) == v, "code %02x value %g re-encodes to %02x (%g)",
           c, v, back, kv_ue4m3_to_f32(back));
        if (c > 0) {
            float below = kv_ue4m3_to_f32((uint8_t)(c - 1));
            float mid = 0.5f * (below + v);
            if (mid > below) {
                uint8_t mc = kv_ue4m3_ceil(mid);
                CK(kv_ue4m3_to_f32(mc) >= mid, "ceil(%g) = %02x (%g) is below", mid, mc, kv_ue4m3_to_f32(mc));
                CK(kv_ue4m3_to_f32(mc) == v, "ceil(%g) should be %02x, got %02x", mid, c, mc);
            }
        }
    }
    CK(kv_ue4m3_ceil(0.0f) == 0, "zero scale");
    CK(kv_ue4m3_ceil(1e6f) == 0x7E, "overflow clamps to the largest finite");
    CK(kv_ue4m3_ceil(1e-9f) == 1, "tiny scale is the smallest subnormal");

    // 2. quantise/dequantise: no code clips (|x| <= 6d) and the error is
    //    within half an E2M1 step at that magnitude; zero rows are all-zero
    static float x[1024], y[1024], out[1024], ref[1024];
    static uint8_t blk[1024 / 16 * 9];
    int n_sizes[] = { 16, 64, 128, 1024 };
    for (size_t s = 0; s < sizeof n_sizes / sizeof *n_sizes; s++) {
        int n = n_sizes[s];
        for (int trial = 0; trial < 50; trial++) {
            float mag = trial % 5 == 0 ? 1e-3f : trial % 5 == 1 ? 300.0f : 8.0f;
            for (int i = 0; i < n; i++) x[i] = rnd() * mag;
            if (trial == 3) memset(x, 0, sizeof(float) * n);
            fp4_quant_row(x, blk, n);
            fp4_dequant_row(blk, y, n);
            for (int b = 0; b < n / 16; b++) {
                float d = kv_ue4m3_to_f32(blk[b * 9]);
                float amax = 0;
                for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(x[b * 16 + j]));
                CK(d * 6.0f >= amax - 1e-6f * amax, "block %d: scale %g cannot cover amax %g", b, d, amax);
                if (amax == 0) CK(blk[b * 9] == 0, "zero block has nonzero scale");
                for (int j = 0; j < 16; j++) {
                    float xv = x[b * 16 + j], yv = y[b * 16 + j];
                    float a = fabsf(xv) / (d > 0 ? d : 1);
                    // half an E2M1 step at this magnitude, in units of d
                    float step = a < 2 ? 0.5f : a < 4 ? 1.0f : 2.0f;
                    CK(fabsf(yv - xv) <= 0.5f * step * d + 1e-6f,
                       "n=%d block %d elem %d: x=%g y=%g d=%g", n, b, j, xv, yv, d);
                    if (amax == 0) CK(yv == 0, "zero row decodes nonzero");
                }
            }
            // 3. fused forms equal the dequantised forms
            for (int i = 0; i < n; i++) ref[i] = rnd();
            float dot = fp4_dot_row(blk, ref, n), dref = 0;
            for (int i = 0; i < n; i++) dref += y[i] * ref[i];
            CK(fabsf(dot - dref) <= 1e-4f * (1 + fabsf(dref)), "dot %g vs %g", dot, dref);
            for (int i = 0; i < n; i++) out[i] = ref[i];
            fp4_accum_row(blk, 0.37f, out, n);
            for (int i = 0; i < n; i++)
                CK(fabsf(out[i] - (ref[i] + 0.37f * y[i])) <= 1e-5f * (1 + fabsf(out[i])),
                   "accum elem %d", i);
        }
    }
    // 4. nibble layout matches the NVFP4 sub-block convention: element j in
    //    the low nibble of byte j, element j+8 in the high nibble
    for (int i = 0; i < 16; i++) x[i] = 0;
    x[3] = 6.0f; x[11] = -3.0f;
    fp4_quant_row(x, blk, 16);
    CK((blk[1 + 3] & 0xF) == 7, "element 3 (6.0) code %x", blk[1 + 3] & 0xF);
    CK((blk[1 + 3] >> 4) == (5 | 8), "element 11 (-3.0) code %x", blk[1 + 3] >> 4);
    if (fails) { fprintf(stderr, "kv fp4 codec: %d failures\n", fails); return 1; }
    printf("kv fp4 codec: ue4m3 ceiling 127/127, quant/dequant bounds, fused dot+accum, nibble layout ok\n");
    return 0;
}
