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
//  5. vec_dot_multi for NVFP4 takes the per-column fallback for 0 columns across nc = 2..8 (observed counter, fails before the kernel)
//  6. vec_dot_multi matches vec_dot bit for bit across block counts, non-contiguous column strides, output sentinels,
//     and nc > VEC_DOT_MULTI_MAX
//  7. vec_dot_multi matches vec_dot bit for bit and stays within 1e-5 of the double dot on every UE4M3 scale byte
//     (0..255) x every E2M1 code (0..15) and alternating-sign sub-blocks
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

static double rel_err_double(const float *w, const float *x, int n, float got) {
    double exact = 0.0, mag = 0.0;
    for (int i = 0; i < n; i++) {
        exact += (double)w[i] * x[i];
        mag += fabs((double)w[i] * x[i]);
    }
    return mag > 0.0 ? fabs((double)got - exact) / mag : fabs((double)got - exact);
}

static long check_multi_widths_and_strides(const int *lens, size_t nlens) {
    enum { NMAX = 6144, STRIDE = 6167, MAXC = VEC_DOT_MULTI_MAX + 2 };
    uint8_t *wrow = malloc((size_t)NMAX / BLK * BLK_BYTES);
    float *buf = malloc(sizeof(float) * (size_t)MAXC * STRIDE);
    long diff = 0;
    const uint32_t guard_bits = 0x7FC01234u;
    float guard;
    memcpy(&guard, &guard_bits, sizeof guard);
    for (size_t li = 0; li < nlens; li++) {
        const int len = lens[li];
        random_row(wrow, len);
        for (int c = 0; c < MAXC; c++) {
            float *col = buf + (size_t)c * STRIDE;
            for (int i = 0; i < len; i++) col[i] = ((float)(next() % 20001) - 10000.0f) / 2500.0f;
        }
        // asymmetric non-adjacent column pointers (as grouped expert positions in ink2_experts_fast)
        const float *xs[MAXC];
        for (int c = 0; c < MAXC; c++) {
            const int src = (c * 3 + 1) % MAXC;
            xs[c] = buf + (size_t)src * STRIDE;
        }
        for (int nc = 1; nc <= VEC_DOT_MULTI_MAX + 1; nc++) {
            float raw[MAXC + 2];
            for (int g = 0; g < MAXC + 2; g++) raw[g] = guard;
            vec_dot_multi(T_NVFP4, wrow, xs, nc, len, raw + 1);
            if (memcmp(&raw[0], &guard, 4) != 0) diff++;
            for (int g = nc + 1; g < MAXC + 2; g++) if (memcmp(&raw[g], &guard, 4) != 0) diff++;
            for (int c = 0; c < nc; c++) {
                const float one = vec_dot(T_NVFP4, wrow, xs[c], len);
                if (memcmp(&one, &raw[1 + c], 4) != 0) diff++;
            }
        }
    }
    free(wrow);
    free(buf);
    return diff;
}

static long check_multi_corners(const uint8_t *all_scales, double *worst_rel) {
    enum { N_ALL = 256 * BLK, N_ALT = 256 };
    float *wdec = malloc(sizeof(float) * N_ALL);
    uint8_t *alt = malloc((size_t)(N_ALT / BLK) * BLK_BYTES);
    float *cols[VEC_DOT_MULTI_MAX];
    for (int c = 0; c < VEC_DOT_MULTI_MAX; c++) {
        cols[c] = malloc(sizeof(float) * N_ALL);
        for (int i = 0; i < N_ALL; i++) cols[c][i] = ((float)(next() % 20001) - 10000.0f) / 2500.0f;
    }
    long diff = 0;
    *worst_rel = 0.0;
    decode_here(all_scales, N_ALL, wdec);
    for (int nc = 1; nc <= VEC_DOT_MULTI_MAX; nc++) {
        float out[VEC_DOT_MULTI_MAX];
        vec_dot_multi(T_NVFP4, all_scales, (const float *const *)cols, nc, N_ALL, out);
        for (int c = 0; c < nc; c++) {
            const float one = vec_dot(T_NVFP4, all_scales, cols[c], N_ALL);
            if (memcmp(&one, &out[c], 4) != 0) diff++;
            const double rel = rel_err_double(wdec, cols[c], N_ALL, out[c]);
            if (rel > *worst_rel) *worst_rel = rel;
        }
    }
    // alternating-sign even vs odd sub-blocks (+6.0 / -6.0 and code 8 -0), testing acc0/acc1 separation
    for (int blk = 0; blk < N_ALT / BLK; blk++) {
        uint8_t *p = alt + (size_t)blk * BLK_BYTES;
        p[0] = 0x7E; p[1] = 0x7E; p[2] = 0x01; p[3] = 0x7F;
        for (int j = 0; j < 8; j++) p[4 + j] = 0x77;          // sub 0: +6.0
        for (int j = 8; j < 16; j++) p[4 + j] = 0xFF;         // sub 1: -6.0
        for (int j = 16; j < 24; j++) p[4 + j] = 0x88;        // sub 2: code 8 (+0.0)
        for (int j = 24; j < 32; j++) p[4 + j] = 0x19;        // sub 3 (0x7F -> 0.0): codes 9, 1
    }
    decode_here(alt, N_ALT, wdec);
    for (int nc = 1; nc <= VEC_DOT_MULTI_MAX; nc++) {
        float out[VEC_DOT_MULTI_MAX];
        vec_dot_multi(T_NVFP4, alt, (const float *const *)cols, nc, N_ALT, out);
        for (int c = 0; c < nc; c++) {
            const float one = vec_dot(T_NVFP4, alt, cols[c], N_ALT);
            if (memcmp(&one, &out[c], 4) != 0) diff++;
            const double rel = rel_err_double(wdec, cols[c], N_ALT, out[c]);
            if (rel > *worst_rel) *worst_rel = rel;
        }
    }
    for (int c = 0; c < VEC_DOT_MULTI_MAX; c++) free(cols[c]);
    free(alt);
    free(wdec);
    return diff;
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
    const unsigned long fb0 = vec_dot_multi_fallbacks();
    for (int nc = 2; nc <= VEC_DOT_MULTI_MAX; nc++) {
        float out[VEC_DOT_MULTI_MAX];
        vec_dot_multi(T_NVFP4, row, (const float *const *)cols, nc, n, out);
    }
    const unsigned long fb_nvfp4 = vec_dot_multi_fallbacks() - fb0;
    CHECK(fb_nvfp4 == 0,
          "5. vec_dot_multi for NVFP4 takes the per-column fallback for 0 columns over nc=2..%d (got %lu)",
          VEC_DOT_MULTI_MAX, fb_nvfp4);
    const long multi_diff = check_multi_widths_and_strides(lens, sizeof lens / sizeof *lens);
    CHECK(multi_diff == 0, "6. vec_dot_multi byte-equal across widths 64..6144, strides, guards and nc=9: %ld differ",
          multi_diff);
    double corner_rel = 0.0;
    const long corner_diff = check_multi_corners(all, &corner_rel);
    CHECK(corner_diff == 0 && corner_rel <= 1e-5,
          "7. vec_dot_multi corner scales/codes byte-equal (%ld differ) and within %.2e <= 1e-5",
          corner_diff, corner_rel);
    // 8. Tier C exact wide-batch tile for NVFP4: 0 bits differ against vec_dot * scale
    // across row counts (1..17), columns (1..64), widths (64..2880), strides, scales,
    // and corner scales/codes.
    {
        enum { TR = 17, TC = 64, TN = 2880, XS = 2893, YS = 23 };
        size_t rsz = (size_t)(TN / BLK) * BLK_BYTES;
        uint8_t *wt = malloc(rsz * TR);
        float *xt = malloc(sizeof(float) * TC * XS);
        float *yt = malloc(sizeof(float) * TC * YS);
        for (int r = 0; r < TR; r++) random_row(wt + (size_t)r * rsz, TN);
        for (int c = 0; c < TC; c++)
            for (int i = 0; i < TN; i++) xt[(size_t)c * XS + i] = ((float)(next() % 20001) - 10000.0f) / 2500.0f;
        static const int t_nrs[] = { 1, 3, 4, 15, 16, 17 };
        static const int t_nbs[] = { 1, 8, 9, 16, 33, 64 };
        static const int t_ns[]  = { 64, 128, 576, 1536, 2880 };
        static const float t_scs[] = { 1.0f, 0.25f };
        long t_diff = 0;
        const unsigned long tc0 = vec_dot_exact_tile_cols();
        for (size_t ri = 0; ri < sizeof(t_nrs) / sizeof(*t_nrs); ri++)
            for (size_t bi = 0; bi < sizeof(t_nbs) / sizeof(*t_nbs); bi++)
                for (size_t ni = 0; ni < sizeof(t_ns) / sizeof(*t_ns); ni++) {
                    int nr = t_nrs[ri], nb = t_nbs[bi], n_len = t_ns[ni];
                    float sc = t_scs[(ri + bi + ni) % 2];
                    for (size_t i = 0; i < (size_t)TC * YS; i++) yt[i] = -777.0f;
                    vec_dot_exact_tile(T_NVFP4, wt, rsz, nr, xt, XS, nb, n_len, sc, yt, YS);
                    for (int c = 0; c < nb; c++)
                        for (int r = 0; r < nr; r++) {
                            float want = vec_dot(T_NVFP4, wt + (size_t)r * rsz,
                                                 xt + (size_t)c * XS, n_len) * sc;
                            t_diff += memcmp(&yt[(size_t)c * YS + r], &want, 4) != 0;
                        }
                }
        // all 256 UE4M3 scales x 16 codes in rows (split across 4 rows x 4096)
        size_t crsz = (size_t)(4096 / BLK) * BLK_BYTES;
        free(xt); free(wt);
        float *xc = malloc(sizeof(float) * 64 * 4096);
        for (size_t i = 0; i < (size_t)64 * 4096; i++) xc[i] = ((float)(next() % 20001) - 10000.0f) / 2500.0f;
        for (size_t i = 0; i < (size_t)TC * YS; i++) yt[i] = -777.0f;
        vec_dot_exact_tile(T_NVFP4, all, crsz, 4, xc, 4096, 64, 4096, 0.5f, yt, YS);
        for (int c = 0; c < 64; c++)
            for (int r = 0; r < 4; r++) {
                float want = vec_dot(T_NVFP4, all + (size_t)r * crsz, xc + (size_t)c * 4096, 4096) * 0.5f;
                t_diff += memcmp(&yt[(size_t)c * YS + r], &want, 4) != 0;
            }
        free(xc); free(yt);
        const unsigned long tc_nvfp4 = vec_dot_exact_tile_cols() - tc0;
#if !defined(RUNNER_CANON_KERNELS) && ((defined(__AVX2__) && defined(__FMA__) && defined(__F16C__) && defined(__AVX512F__)) || (defined(__aarch64__) && defined(__ARM_NEON)))
        CHECK(t_diff == 0 && vec_dot_exact_tile_ok(T_NVFP4, TN) && tc_nvfp4 > 0,
              "8. vec_dot_exact_tile for NVFP4 byte-equal to vec_dot*scale over 1..64 cols (%lu tile cols): %ld differ",
              tc_nvfp4, t_diff);
#else
        CHECK(t_diff == 0 && !vec_dot_exact_tile_ok(T_NVFP4, TN) && tc_nvfp4 == 0,
              "8. vec_dot_exact_tile for NVFP4 declines AVX2/canon and falls back byte-equal (%lu tile cols): %ld differ",
              tc_nvfp4, t_diff);
#endif
    }
    for (int c = 0; c < 8; c++) free(cols[c]);
    free(row); free(a); free(b); free(x); free(a2); free(b2); free(all);
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
