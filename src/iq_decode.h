// Codebook i-quant decode primitives shared by the CUDA kernels and a host
// test. The arithmetic the device runs is this file, compiled by nvcc into
// kernels.cu; the same file compiled by the C compiler is what
// tests/test_iq_decode.c executes against the host tables in
// quants_iq_grids.h, so a change here is measured without a GPU (the
// v0.5.3 review showed the sign helper could be broken with every test
// still green: the tests checked the table's arithmetic, never the code).
//
// Host-side this is plain C11; device-side every function is a
// __forceinline__ __device__ helper. No engine header is included: the
// only inputs are packed bytes and grid entries, and the only outputs are
// floats and unsigned integers.
#ifndef RUNNER_IQ_DECODE_H
#define RUNNER_IQ_DECODE_H

#if defined(__CUDACC__)
#define IQ_FN static __device__ __forceinline__
#define IQ_POPC(x) __popc(x)
// staging writes fp16 on the device (the tensor-core operand) and f32 on
// the host, where the test compares the value before the fp16 rounding
typedef __half iq_stage_t;
#define IQ_STORE(dst, i, v) ((dst)[i] = __float2half(v))
#else
#include <math.h>
#include <string.h>
#define IQ_FN static inline
#define IQ_POPC(x) __builtin_popcount(x)
typedef float iq_stage_t;
#define IQ_STORE(dst, i, v) ((dst)[i] = (v))
#endif

typedef unsigned char iq_byte;
typedef unsigned long long iq_u64;

// The block scale, an IEEE half at a 2-byte aligned address. The device
// converts in hardware; the host conversion below is exact for every
// half (subnormals scaled by 2^-24, Inf and NaN built from their bits).
IQ_FN float iq_f16(const iq_byte *p) {
#if defined(__CUDACC__)
    return __half2float(*(const __half *)p);
#else
    unsigned h = p[0] | ((unsigned)p[1] << 8);
    unsigned sign = h >> 15, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) {
        v = ldexpf((float)m, -24);
    } else if (e == 31) {
        unsigned bits = 0x7f800000u | (m << 13);
        memcpy(&v, &bits, sizeof v);
    } else {
        v = ldexpf((float)(m | 0x400), (int)e - 25);
    }
    return sign ? -v : v;
#endif
}

// Block fields are 2-byte aligned (every block size is even and a row is a
// whole number of blocks), so 32-bit fields are read as two 16-bit halves.
IQ_FN unsigned iq_ld16(const iq_byte *p) {
    return *(const unsigned short *)p;
}
IQ_FN unsigned iq_ld32a2(const iq_byte *p) {
    return iq_ld16(p) | (iq_ld16(p + 2) << 16);
}

// ksigns_iq2xs[i] in quants_iq_grids.h is i with its odd parity in bit 7
// (tests/test_iq_decode.c holds this function to the table for all 128
// entries), so a 7-bit sign index expands arithmetically instead of
// through a divergent table read.
IQ_FN unsigned iq_signs7(unsigned idx) {
    return idx | ((IQ_POPC(idx) & 1u) << 7);
}

// weight j of an 8-magnitude grid entry (byte j), negated by sign bit j
IQ_FN float iq_w8(iq_u64 grid, int j, unsigned signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return (signs >> j) & 1 ? -mag : mag;
}

// weight j of a 4-magnitude grid entry (byte j), negated by sign bit j
IQ_FN float iq_w4(unsigned grid, int j, unsigned signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return (signs >> j) & 1 ? -mag : mag;
}

// IQ1 grid bytes are signed (-1/0/1) and carry a per-index delta
IQ_FN float iq1_w(iq_u64 grid, int j, float delta) {
    return (float)(signed char)((grid >> (8 * j)) & 0xFF) + delta;
}
#define IQ1_DELTA 0.125f

// ------------------------------------------------- tensor-core staging
// One 64-element segment of a 256-weight block, decoded to the staging
// type: segment s is elements [64s, 64s + 64). The tensor-core GEMMs stage
// a 64-row x 128-K tile per step with two threads per row, one segment
// each, so this is the unit of work; the host test decodes all four
// segments of a block and holds them to dequant_row() bit for bit.

// IQ3_S: segment s is the sub-block pair p = s of dq_iq3_s (quants.c):
// scale nibbles of scales[p] (1 + 2*nibble, low nibble first), high index
// bits qh[2p] and qh[2p+1] (one bit per index), indices qs + 16p, sign
// bytes signs + 8p; the arithmetic of k_mv_iq3_s_b, value = d * scale *
// (+/- magnitude).
IQ_FN void iq3s_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                        const unsigned *grid) {
    float d = iq_f16(blk);
    const iq_byte *qs = blk + 2 + 16 * seg, *qh = blk + 66 + 2 * seg,
                  *sg = blk + 74 + 8 * seg;
    unsigned scb = blk[106 + seg];
    for (int h = 0; h < 2; h++) {
        float db = d * (float)(1 + 2 * (h ? (scb >> 4) : (scb & 0xF)));
        unsigned hb = qh[h];
        const iq_byte *q = qs + 8 * h, *sgn = sg + 4 * h;
        for (int l = 0; l < 4; l++) {
            unsigned g1 = grid[q[2 * l + 0] | ((hb << (8 - 2 * l)) & 256)];
            unsigned g2 = grid[q[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
            unsigned signs = sgn[l];
            for (int j = 0; j < 4; j++) {
                IQ_STORE(dst, h * 32 + l * 8 + j, db * iq_w4(g1, j, signs));
                IQ_STORE(dst, h * 32 + l * 8 + 4 + j, db * iq_w4(g2, j, signs >> 4));
            }
        }
    }
}

#endif
