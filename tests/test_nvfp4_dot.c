// R4.48.15: the NVFP4 kernels Inkling 2 Mini's fast route takes, against a decode and a dot computed here.
//
// An NVFP4 block (ggml's block_nvfp4): 64 values as four 16-value sub-blocks, four UE4M3 scale bytes then 32 bytes of
// E2M1 codes; in sub-block s the byte j holds value j (low nibble) and value j + 8 (high nibble). A value is
// E2M1[code] x UE4M3[scale], exact in float32.
//
//  1. this build's vec_dot for NVFP4 is a SIMD kernel (vec_dot_simd), not the scalar loop (fails before the kernel)
//  2. dequant_row (the prefill tile's decode) equals the decode computed here bit for bit, on random blocks and on
//     blocks with every scale byte and every code
//  3. vec_dot is within 1e-5 of the double-precision dot of the exact weights, relative to sum |w x|, at the lengths
//     a real file has (64 to 6144) and on ragged lengths of whole blocks
//  4. vec_dot_multi is byte-equal to vec_dot for one to eight columns
#include "quants.h"
#include "fp16.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BLK = 64, BLK_BYTES = 36 };
// code 8 (negative zero in E2M1) decodes to +0, as ggml's integer table and src/quants.c's kvalues_mxfp4 give it;
// no dot product can tell the two zeros apart
static const float E2M1[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rng = 20261123;
static uint32_t next(void) { rng = rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(rng >> 33); }

static void decode_here(const uint8_t *row, int n, float *y) {
    for (int b = 0; b < n / BLK; b++) {
        const uint8_t *blk = row + (size_t)b * BLK_BYTES;
        for (int s = 0; s < 4; s++) {
            const float d = kv_ue4m3_to_f32(blk[s]);
            const uint8_t *q = blk + 4 + s * 8;
            for (int j = 0; j < 8; j++) {
                y[b * BLK + s * 16 + j] = E2M1[q[j] & 0xF] * d;
                y[b * BLK + s * 16 + j + 8] = E2M1[q[j] >> 4] * d;
            }
        }
    }
}
// scale bytes drawn from the finite, non-zero UE4M3 codes a converter writes for weights (1 to 0x7E), codes uniform
static void random_row(uint8_t *row, int n) {
    for (int b = 0; b < n / BLK; b++) {
        uint8_t *blk = row + (size_t)b * BLK_BYTES;
        for (int s = 0; s < 4; s++) blk[s] = (uint8_t)(1 + next() % 0x7E);
        for (int j = 0; j < 32; j++) blk[4 + j] = (uint8_t)next();
    }
}

int main(void) {
    f16_init();
    CHECK(vec_dot_simd(T_NVFP4), "1. vec_dot for NVFP4 is a SIMD kernel on this build");

    // 2. the decode
    enum { NMAX = 6144 };
    uint8_t *row = malloc((size_t)NMAX / BLK * BLK_BYTES);
    float *a = malloc(sizeof(float) * NMAX), *b = malloc(sizeof(float) * NMAX), *x = malloc(sizeof(float) * NMAX);
    long bad = 0;
    for (int r = 0; r < 200; r++) {
        random_row(row, NMAX);
        dequant_row(T_NVFP4, row, a, NMAX);
        decode_here(row, NMAX, b);
        bad += memcmp(a, b, sizeof(float) * NMAX) != 0;
    }
    // every scale byte against every code: 256 blocks, block k's four scales all k, its codes 0..15 in turn
    float *a2 = malloc(sizeof(float) * 256 * BLK), *b2 = malloc(sizeof(float) * 256 * BLK);
    uint8_t *all = malloc((size_t)256 * BLK_BYTES);
    for (int k = 0; k < 256; k++) {
        uint8_t *blk = all + (size_t)k * BLK_BYTES;
        for (int s = 0; s < 4; s++) blk[s] = (uint8_t)k;
        for (int j = 0; j < 32; j++) blk[4 + j] = (uint8_t)((j & 15) | ((15 - (j & 15)) << 4));
    }
    dequant_row(T_NVFP4, all, a2, 256 * BLK);
    decode_here(all, 256 * BLK, b2);
    for (int i = 0; i < 256 * BLK; i++) bad += memcmp(&a2[i], &b2[i], 4) != 0;
    CHECK(bad == 0, "2. dequant_row equals the decode here bit for bit (200 random rows of 6144, every scale x code): %ld differ", bad);

    // 3. the dot
    static const int lens[] = { 64, 128, 192, 576, 1024, 1536, 2048, 2880, 4096, 6144 };
    double worst = 0;
    for (size_t li = 0; li < sizeof lens / sizeof *lens; li++) {
        const int n = lens[li];
        for (int r = 0; r < 50; r++) {
            random_row(row, n);
            for (int i = 0; i < n; i++) x[i] = ((float)(next() % 20001) - 10000.0f) / 2500.0f;
            decode_here(row, n, b);
            double exact = 0, mag = 0;
            for (int i = 0; i < n; i++) { exact += (double)b[i] * x[i]; mag += fabs((double)b[i] * x[i]); }
            const float got = vec_dot(T_NVFP4, row, x, n);
            const double rel = mag > 0 ? fabs(got - exact) / mag : fabs(got - exact);
            if (rel > worst) worst = rel;
        }
    }
    CHECK(worst <= 1e-5, "3. vec_dot within %.2e of the double dot (relative to sum |w x|), want <= 1e-5", worst);

    // 4. the multi-column dot
    const int n = 2880;
    random_row(row, n);
    float *cols[8];
    for (int c = 0; c < 8; c++) {
        cols[c] = malloc(sizeof(float) * n);
        for (int i = 0; i < n; i++) cols[c][i] = ((float)(next() % 20001) - 10000.0f) / 2500.0f;
    }
    long mism = 0;
    for (int nc = 1; nc <= 8; nc++) {
        float out[8];
        vec_dot_multi(T_NVFP4, row, (const float *const *)cols, nc, n, out);
        for (int c = 0; c < nc; c++) {
            const float one = vec_dot(T_NVFP4, row, cols[c], n);
            mism += memcmp(&one, &out[c], 4) != 0;
        }
    }
    CHECK(mism == 0, "4. vec_dot_multi byte-equal to vec_dot for 1 to 8 columns: %ld differ", mism);
    for (int c = 0; c < 8; c++) free(cols[c]);
    free(row); free(a); free(b); free(x); free(a2); free(b2); free(all);
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
