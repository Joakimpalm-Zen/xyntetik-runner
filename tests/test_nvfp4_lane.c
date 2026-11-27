// R4.48.33: the coalesced NVFP4 GEMV's lane map and arithmetic (src/nvfp4_lane.h), held on the host.
//
// nvfp4_lane.h is compiled into the CUDA kernels (kernels.cu k_gemv_nvfp4, k_gemv_nvfp4_a2) and into this test,
// so what this test computes is what a lane computes: every multiply-add is an explicit fmaf and the tail's multiply
// and add are explicitly rounded (nvcc contracts by default, so nothing is left for it to fuse). It holds:
//   1. both lane maps against the tables below, element and byte coverage;
//   2. the register E2M1 decode against the kernels' table, all 16 nibbles bit for bit (nibble 8 is +0.0);
//   3. the UE4M3 decode against dequant_row(), all 256 codes;
//   4. a row's emulated dot, both maps, against dequant_row()'s dot in double, within float rounding;
//   5. an all-zero row reads +0.0 exactly, in both maps, any activations;
//   6. B4, the identity rule: a C-column dot (the twins' order) equals C single-column dots bit for bit;
//   7. the tail, scale then bias, each rounded once.
#include "runner.h"
#include "quants.h"
#include "nvfp4_lane.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// One warp pass covers 4 blocks (256 values): lane l takes block l / 8 and sub-block (l % 8) / 2.
// x0/n0 and x1/n1: the block-relative activations of its products, in summation order (n contiguous from x).
// byte0/nbytes: the nibble bytes it loads (block-relative, after the 4 scale bytes); nib: which nibbles.
enum { LOW = 0, HIGH = 1, BOTH = 2 };
typedef struct { int blk, sub, x0, n0, x1, n1, byte0, nbytes, nib; } lane_row;

// Map A: a lane pair shares a sub-block's 8 bytes; the even lane takes the low nibbles (elements 0 to 7 of the
// sub-block), the odd lane the high nibbles (8 to 15).
static const lane_row MAP_A[32] = {
    {0, 0,  0, 8, -1, 0,  4, 8, LOW}, {0, 0,  8, 8, -1, 0,  4, 8, HIGH},
    {0, 1, 16, 8, -1, 0, 12, 8, LOW}, {0, 1, 24, 8, -1, 0, 12, 8, HIGH},
    {0, 2, 32, 8, -1, 0, 20, 8, LOW}, {0, 2, 40, 8, -1, 0, 20, 8, HIGH},
    {0, 3, 48, 8, -1, 0, 28, 8, LOW}, {0, 3, 56, 8, -1, 0, 28, 8, HIGH},
    {1, 0,  0, 8, -1, 0,  4, 8, LOW}, {1, 0,  8, 8, -1, 0,  4, 8, HIGH},
    {1, 1, 16, 8, -1, 0, 12, 8, LOW}, {1, 1, 24, 8, -1, 0, 12, 8, HIGH},
    {1, 2, 32, 8, -1, 0, 20, 8, LOW}, {1, 2, 40, 8, -1, 0, 20, 8, HIGH},
    {1, 3, 48, 8, -1, 0, 28, 8, LOW}, {1, 3, 56, 8, -1, 0, 28, 8, HIGH},
    {2, 0,  0, 8, -1, 0,  4, 8, LOW}, {2, 0,  8, 8, -1, 0,  4, 8, HIGH},
    {2, 1, 16, 8, -1, 0, 12, 8, LOW}, {2, 1, 24, 8, -1, 0, 12, 8, HIGH},
    {2, 2, 32, 8, -1, 0, 20, 8, LOW}, {2, 2, 40, 8, -1, 0, 20, 8, HIGH},
    {2, 3, 48, 8, -1, 0, 28, 8, LOW}, {2, 3, 56, 8, -1, 0, 28, 8, HIGH},
    {3, 0,  0, 8, -1, 0,  4, 8, LOW}, {3, 0,  8, 8, -1, 0,  4, 8, HIGH},
    {3, 1, 16, 8, -1, 0, 12, 8, LOW}, {3, 1, 24, 8, -1, 0, 12, 8, HIGH},
    {3, 2, 32, 8, -1, 0, 20, 8, LOW}, {3, 2, 40, 8, -1, 0, 20, 8, HIGH},
    {3, 3, 48, 8, -1, 0, 28, 8, LOW}, {3, 3, 56, 8, -1, 0, 28, 8, HIGH},
};

// Map A' (the advisor's): a lane owns 4 consecutive bytes of a sub-block and takes both nibbles of each, so
// elements 4k to 4k+3 (low nibbles, summed first) and 8+4k to 8+4k+3 (high), k = l % 2: one 4-byte load a lane,
// every byte loaded once.
static const lane_row MAP_A2[32] = {
    {0, 0,  0, 4,  8, 4,  4, 4, BOTH}, {0, 0,  4, 4, 12, 4,  8, 4, BOTH},
    {0, 1, 16, 4, 24, 4, 12, 4, BOTH}, {0, 1, 20, 4, 28, 4, 16, 4, BOTH},
    {0, 2, 32, 4, 40, 4, 20, 4, BOTH}, {0, 2, 36, 4, 44, 4, 24, 4, BOTH},
    {0, 3, 48, 4, 56, 4, 28, 4, BOTH}, {0, 3, 52, 4, 60, 4, 32, 4, BOTH},
    {1, 0,  0, 4,  8, 4,  4, 4, BOTH}, {1, 0,  4, 4, 12, 4,  8, 4, BOTH},
    {1, 1, 16, 4, 24, 4, 12, 4, BOTH}, {1, 1, 20, 4, 28, 4, 16, 4, BOTH},
    {1, 2, 32, 4, 40, 4, 20, 4, BOTH}, {1, 2, 36, 4, 44, 4, 24, 4, BOTH},
    {1, 3, 48, 4, 56, 4, 28, 4, BOTH}, {1, 3, 52, 4, 60, 4, 32, 4, BOTH},
    {2, 0,  0, 4,  8, 4,  4, 4, BOTH}, {2, 0,  4, 4, 12, 4,  8, 4, BOTH},
    {2, 1, 16, 4, 24, 4, 12, 4, BOTH}, {2, 1, 20, 4, 28, 4, 16, 4, BOTH},
    {2, 2, 32, 4, 40, 4, 20, 4, BOTH}, {2, 2, 36, 4, 44, 4, 24, 4, BOTH},
    {2, 3, 48, 4, 56, 4, 28, 4, BOTH}, {2, 3, 52, 4, 60, 4, 32, 4, BOTH},
    {3, 0,  0, 4,  8, 4,  4, 4, BOTH}, {3, 0,  4, 4, 12, 4,  8, 4, BOTH},
    {3, 1, 16, 4, 24, 4, 12, 4, BOTH}, {3, 1, 20, 4, 28, 4, 16, 4, BOTH},
    {3, 2, 32, 4, 40, 4, 20, 4, BOTH}, {3, 2, 36, 4, 44, 4, 24, 4, BOTH},
    {3, 3, 48, 4, 56, 4, 28, 4, BOTH}, {3, 3, 52, 4, 60, 4, 32, 4, BOTH},
};

// kernels.cu kv_mxfp4, the table the one-lane-per-block kernels read (nibble 8 is +0.0 there and in quants.c)
static const float KV_MXFP4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float frnd(void) { return ((float)(rnd() & 0xffffff) / 16777216.0f - 0.5f) * 4.0f; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void check_maps(void) {
    static const struct { int map; const lane_row *t; const char *name; } M[] = {
        { NV_MAP_A, MAP_A, "A" }, { NV_MAP_A2, MAP_A2, "A'" },
    };
    for (size_t k = 0; k < sizeof M / sizeof *M; k++) {
        int elem_hits[4][64] = {{0}}, byte_hits[4][36] = {{0}};
        for (int l = 0; l < 32; l++) {
            nv_lane_t m;
            nv_lane_map(M[k].map, l, &m);
            const lane_row *r = &M[k].t[l];
            if (m.blk != r->blk || m.sub != r->sub || m.x0 != r->x0 || m.n0 != r->n0 || m.x1 != r->x1 ||
                m.n1 != r->n1 || m.byte0 != r->byte0 || m.nbytes != r->nbytes || m.nib != r->nib) {
                fprintf(stderr, "FAIL: map %s lane %d differs from its table\n", M[k].name, l);
                exit(1);
            }
            for (int j = 0; j < m.n0; j++) elem_hits[m.blk][m.x0 + j]++;
            for (int j = 0; j < m.n1; j++) elem_hits[m.blk][m.x1 + j]++;
            for (int j = 0; j < m.nbytes; j++) byte_hits[m.blk][m.byte0 + j]++;
        }
        for (int b = 0; b < 4; b++) {
            for (int e = 0; e < 64; e++) assert(elem_hits[b][e] == 1);           // every element exactly once
            for (int j = 4; j < 36; j++) assert(byte_hits[b][j] == (M[k].map == NV_MAP_A ? 2 : 1));
        }
    }
    printf("ok: lane maps A and A' match their tables; every element once, every byte %s\n",
           "twice in A (a pair), once in A'");
}

static void check_decoders(void) {
    for (unsigned n = 0; n < 16; n++) {
        float v = nv_e2m1(n);
        if (fbits(v) != fbits(KV_MXFP4[n])) {
            fprintf(stderr, "FAIL: nv_e2m1(%u) = %a, table %a\n", n, (double)v, (double)KV_MXFP4[n]);
            exit(1);
        }
    }
    // UE4M3 against dequant_row: a block whose element 0 is nibble 2 (1.0) under scale code c
    for (unsigned c = 0; c < 256; c++) {
        uint8_t blk[36] = {0};
        blk[0] = (uint8_t)c;
        blk[4] = 2;
        float out[64];
        dequant_row(T_NVFP4, blk, out, 64);
        if (fbits(nv_ue4m3(c)) != fbits(out[0])) {
            fprintf(stderr, "FAIL: nv_ue4m3(0x%02x) = %a, dequant_row %a\n", c, (double)nv_ue4m3(c), (double)out[0]);
            exit(1);
        }
    }
    printf("ok: E2M1 in registers equals the kernels' table (16 nibbles, bit for bit); UE4M3 equals dequant_row "
           "(256 codes)\n");
}

static void fill_row(uint8_t *row, int nb, int zero) {
    for (int b = 0; b < nb; b++) {
        uint8_t *blk = row + 36 * b;
        for (int s = 0; s < 4; s++) blk[s] = zero ? (uint8_t)(rnd() & 0x7f) : (uint8_t)(0x20 + rnd() % 0x40);
        for (int j = 4; j < 36; j++) blk[j] = zero ? (uint8_t)((rnd() & 1) ? 0x88 : 0x00) : (uint8_t)rnd();
    }
}

static void check_rows(void) {
    static const int N[] = { 64, 128, 192, 256, 320, 4096, 4160, 11008 };
    for (size_t k = 0; k < sizeof N / sizeof *N; k++) {
        int n = N[k], nb = n / 64;
        uint8_t *row = malloc((size_t)nb * 36);
        float *x = malloc(sizeof(float) * (size_t)n), *w = malloc(sizeof(float) * (size_t)n);
        assert(row && x && w);
        for (int t = 0; t < 20; t++) {
            fill_row(row, nb, 0);
            for (int i = 0; i < n; i++) x[i] = frnd();
            dequant_row(T_NVFP4, row, w, n);
            double ref = 0, mag = 0;
            for (int i = 0; i < n; i++) { ref += (double)w[i] * x[i]; mag += fabs((double)w[i] * x[i]); }
            for (int map = NV_MAP_A; map <= NV_MAP_A2; map++) {
                float got = nv_emul_row(map, row, x, n);
                if (fabs((double)got - ref) > 1e-5 * mag + 1e-30) {
                    fprintf(stderr, "FAIL: map %d n %d: %.9g against %.9g (sum |wx| %.3g)\n", map, n,
                            (double)got, ref, mag);
                    exit(1);
                }
            }
        }
        // all-zero rows (nibbles 0 and 8, any scales) read +0.0 exactly, whatever the activations' signs
        fill_row(row, nb, 1);
        for (int i = 0; i < n; i++) x[i] = frnd();
        for (int map = NV_MAP_A; map <= NV_MAP_A2; map++) {
            float z = nv_emul_row(map, row, x, n);
            if (fbits(z) != 0u) {
                fprintf(stderr, "FAIL: map %d n %d: an all-zero row reads %a, not +0.0\n", map, n, (double)z);
                exit(1);
            }
        }
        free(row); free(x); free(w);
    }
    printf("ok: rows of 64 to 11008 values (tails of 1 to 3 blocks), both maps, within float rounding of "
           "dequant_row's dot; all-zero rows read +0.0\n");
}

static void check_twins(void) {
    enum { N = 4160, C = 16 };
    static uint8_t row[(N / 64) * 36];
    static float x[C * N], ys[C];
    fill_row(row, N / 64, 0);
    for (int i = 0; i < C * N; i++) x[i] = frnd();
    for (int map = NV_MAP_A; map <= NV_MAP_A2; map++) {
        for (int cols = 1; cols <= C; cols++) {
            nv_emul_rows_cols(map, row, x, N, N, cols, ys);
            for (int c = 0; c < cols; c++) {
                float one = nv_emul_row(map, row, x + (size_t)c * N, N);
                if (fbits(one) != fbits(ys[c])) {
                    fprintf(stderr, "FAIL: map %d: column %d of %d reads %a, alone %a\n", map, c, cols,
                            (double)ys[c], (double)one);
                    exit(1);
                }
            }
        }
    }
    printf("ok: B4, 1 to 16 columns in the twins' order equal each column alone, bit for bit, both maps\n");
}

static void check_tail(void) {
    for (int t = 0; t < 1000; t++) {
        float s = frnd(), sc = (t & 1) ? 1.0f : frnd(), b = frnd();
        volatile float m = s * sc;          // rounded once
        volatile float y = m + b;           // rounded once
        assert(fbits(nv_tail(s, sc, 1, b)) == fbits(y));
        assert(fbits(nv_tail(s, sc, 0, b)) == fbits(m));
    }
    printf("ok: the tail rounds the scale's multiply and the bias add separately\n");
}

int main(void) {
    f16_init();
    check_maps();
    check_decoders();
    check_rows();
    check_twins();
    check_tail();
    printf("all nvfp4 lane tests passed\n");
    return 0;
}
