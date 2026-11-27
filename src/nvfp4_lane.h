// NVFP4's coalesced GEMV (R4.48.33): the lane map and the arithmetic of one lane, shared by the CUDA kernels
// (kernels.cu k_gemv_nvfp4 and k_gemv_nvfp4_a2) and tests/test_nvfp4_lane.c, the way kq_stage.h and iq_decode.h
// are shared. nvcc compiles this file into the kernels; the C compiler into the test.
//
// The block is NVFP4's 36 bytes: four UE4M3 sub-block scales d[4], then 32 bytes of E2M1 nibbles. Within sub-block
// s, byte j's low nibble is element j and its high nibble element j + 8 (quants.c dq_nvfp4).
//
// One warp computes one row. A pass covers 4 consecutive blocks (256 values): lane l takes block l / 8 and
// sub-block (l % 8) / 2, and 8 of that sub-block's 16 elements, as its map says. A pass whose blocks run past the
// row's end leaves those lanes idle (the tail is part of the order). Each lane's sum is, in this order:
//   t = 0; t = fmaf(w_j, x_j, t) for its 8 products in map order;  s = fmaf(d_sub, t, s) per pass;
// then the warp's shfl_down tree (kernels.cu warp_sum), then y = s * scale, then + bias, each rounded once.
// Every multiply-add is an explicit fmaf and the tail's multiply and add are explicitly rounded, so nvcc's default
// contraction has nothing to fuse and the host computes exactly what a lane does.
#ifndef RUNNER_NVFP4_LANE_H
#define RUNNER_NVFP4_LANE_H

#if defined(__CUDACC__)
#define NV_FN static __device__ __forceinline__
#define NV_MUL(a, b) __fmul_rn((a), (b))
#define NV_ADD(a, b) __fadd_rn((a), (b))
#else
#include <math.h>
#include <string.h>
#define NV_FN static inline
#define NV_MUL(a, b) ((a) * (b))   // the host test is built -ffp-contract=off
#define NV_ADD(a, b) ((a) + (b))
#endif

typedef unsigned char nv_byte;

enum { NV_MAP_A = 0, NV_MAP_A2 = 1 };
enum { NV_LOW = 0, NV_HIGH = 1, NV_BOTH = 2 };

// What lane l of a pass reads: block blk of the pass, sub-block sub; activations x0..x0+n0-1, then x1..x1+n1-1
// (block-relative, in summation order); bytes byte0..byte0+nbytes-1 of the block; which nibbles of them.
typedef struct { int blk, sub, x0, n0, x1, n1, byte0, nbytes, nib; } nv_lane_t;

NV_FN void nv_lane_map(int map, int lane, nv_lane_t *m) {
    int sub = (lane & 7) >> 1, p = lane & 1;
    m->blk = lane >> 3;
    m->sub = sub;
    if (map == NV_MAP_A) {
        // a lane pair shares the sub-block's 8 bytes: the even lane the low nibbles, the odd lane the high
        m->x0 = sub * 16 + p * 8; m->n0 = 8; m->x1 = -1; m->n1 = 0;
        m->byte0 = 4 + sub * 8; m->nbytes = 8; m->nib = p ? NV_HIGH : NV_LOW;
    } else {
        // a lane owns 4 consecutive bytes and both their nibbles: elements 4p..4p+3, then 8+4p..8+4p+3
        m->x0 = sub * 16 + 4 * p; m->n0 = 4; m->x1 = sub * 16 + 8 + 4 * p; m->n1 = 4;
        m->byte0 = 4 + sub * 8 + 4 * p; m->nbytes = 4; m->nib = NV_BOTH;
    }
}

// E2M1 in registers: the magnitudes doubled (0, 1, 2, 3, 4, 6, 8, 12) packed 4 bits each, the sign applied to the
// integer, then halved: exact, and nibble 8 is +0.0 as in the kernels' kv_mxfp4 table and quants.c.
NV_FN float nv_e2m1(unsigned n) {
    int m = (int)((0xC8643210u >> (4u * (n & 7u))) & 0xFu);
    if (n & 8u) m = -m;
    return (float)m * 0.5f;
}

// UE4M3 as kernels.cu ue4m3f and quants.c ue4m3_to_fp32: 0 and 0x7F read 0, exponent 0 is subnormal
NV_FN float nv_ue4m3(unsigned x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    int e = (int)((x >> 3) & 0xFu), m = (int)(x & 0x7u);
    if (e == 0) return ldexpf((float)m, -9);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// a 4-byte little-endian load; every offset this file asks for is a multiple of 4 from a block's start
NV_FN unsigned nv_ld32(const nv_byte *p) {
#if defined(__CUDACC__)
    return *(const unsigned *)p;
#else
    unsigned u;
    memcpy(&u, p, 4);
    return u;
#endif
}

// The lane's 8 weights, in its summation order, from the block at bp
NV_FN void nv_lane_w(int map, int lane, const nv_byte *bp, float *w) {
    int sub = (lane & 7) >> 1, p = lane & 1;
    if (map == NV_MAP_A) {
        unsigned q0 = nv_ld32(bp + 4 + sub * 8), q1 = nv_ld32(bp + 8 + sub * 8), sh = p ? 4u : 0u;
        for (int j = 0; j < 4; j++) {
            w[j]     = nv_e2m1((q0 >> (8u * (unsigned)j + sh)) & 0xFu);
            w[4 + j] = nv_e2m1((q1 >> (8u * (unsigned)j + sh)) & 0xFu);
        }
    } else {
        unsigned q = nv_ld32(bp + 4 + sub * 8 + 4 * p);
        for (int i = 0; i < 4; i++) {
            w[i]     = nv_e2m1((q >> (8u * (unsigned)i)) & 0xFu);
            w[4 + i] = nv_e2m1((q >> (8u * (unsigned)i + 4u)) & 0xFu);
        }
    }
}

// The lane's 8 activations from the block's 64 at xb, in the same order (the kernels load them as two float4s)
NV_FN void nv_lane_x(int map, int lane, const float *xb, float *x) {
    nv_lane_t m;
    nv_lane_map(map, lane, &m);
    for (int j = 0; j < m.n0; j++) x[j] = xb[m.x0 + j];
    for (int j = 0; j < m.n1; j++) x[m.n0 + j] = xb[m.x1 + j];
}

NV_FN float nv_dot8(const float *w, const float *x) {
    float t = 0.0f;
    for (int j = 0; j < 8; j++) t = fmaf(w[j], x[j], t);
    return t;
}

NV_FN float nv_tail(float s, float scale, int has_bias, float bias) {
    float m = NV_MUL(s, scale);
    return has_bias ? NV_ADD(m, bias) : m;
}

#if !defined(__CUDACC__)
// kernels.cu warp_sum: s += __shfl_down_sync(s, off) for off = 16, 8, 4, 2, 1; lane 0's value
static inline float nv_emul_warp_sum(const float *lanes) {
    float v[32], n[32];
    memcpy(v, lanes, sizeof v);
    for (int off = 16; off > 0; off >>= 1) {
        for (int i = 0; i < 32; i++) n[i] = v[i] + (i + off < 32 ? v[i + off] : v[i]);
        memcpy(v, n, sizeof v);
    }
    return v[0];
}

// The GEMV's order: one row, one column, as k_gemv_nvfp4* computes it, before the tail
static inline float nv_emul_row(int map, const nv_byte *row, const float *x, int n_in) {
    float s[32] = {0};
    int nb = n_in / 64;
    for (int b0 = 0; b0 < nb; b0 += 4)
        for (int lane = 0; lane < 32; lane++) {
            int b = b0 + (lane >> 3);
            if (b >= nb) continue;
            const nv_byte *bp = row + (size_t)b * 36;
            float w[8], xv[8];
            nv_lane_w(map, lane, bp, w);
            nv_lane_x(map, lane, x + (size_t)b * 64, xv);
            s[lane] = fmaf(nv_ue4m3(bp[(lane & 7) >> 1]), nv_dot8(w, xv), s[lane]);
        }
    return nv_emul_warp_sum(s);
}

// The twins' order: a lane decodes its weights once a pass and applies them to every column (x columns xs apart),
// each column summed exactly as the GEMV sums it. ys[c] is column c's value before the tail.
static inline void nv_emul_rows_cols(int map, const nv_byte *row, const float *x, int n_in, int xs, int cols,
                                     float *ys) {
    float s[16][32];
    memset(s, 0, sizeof s);
    int nb = n_in / 64;
    for (int b0 = 0; b0 < nb; b0 += 4)
        for (int lane = 0; lane < 32; lane++) {
            int b = b0 + (lane >> 3);
            if (b >= nb) continue;
            const nv_byte *bp = row + (size_t)b * 36;
            float w[8];
            nv_lane_w(map, lane, bp, w);
            float d = nv_ue4m3(bp[(lane & 7) >> 1]);
            for (int c = 0; c < cols; c++) {
                float xv[8];
                nv_lane_x(map, lane, x + (size_t)c * xs + (size_t)b * 64, xv);
                s[c][lane] = fmaf(d, nv_dot8(w, xv), s[c][lane]);
            }
        }
    for (int c = 0; c < cols; c++) ys[c] = nv_emul_warp_sum(s[c]);
}
#endif

#endif
