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
IQ_FN float iq_f16_bits(unsigned h) {
#if defined(__CUDACC__)
    return __half2float(__ushort_as_half((unsigned short)h));
#else
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
IQ_FN float iq_f16(const iq_byte *p) {
#if defined(__CUDACC__)
    return __half2float(*(const __half *)p);
#else
    return iq_f16_bits(p[0] | ((unsigned)p[1] << 8));
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


// IQ3_XXS: 98-byte block; segment s is sub-blocks 2s and 2s+1 of
// dq_iq3_xxs: eight grid indices per sub-block (qs + 8*ib, four magnitudes
// each), one word of four 7-bit sign indices and a 4-bit scale at
// 66 + 4*ib, scale (0.5 + nibble) / 2.
IQ_FN void iq3xxs_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                          const unsigned *grid) {
    float d = iq_f16(blk);
    for (int h = 0; h < 2; h++) {
        int ib = 2 * seg + h;
        const iq_byte *qs = blk + 2 + 8 * ib;
        unsigned aux = iq_ld32a2(blk + 66 + 4 * ib);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            unsigned g1 = grid[qs[2 * l + 0]], g2 = grid[qs[2 * l + 1]];
            unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
            for (int j = 0; j < 4; j++) {
                IQ_STORE(dst, h * 32 + l * 8 + j, db * iq_w4(g1, j, signs));
                IQ_STORE(dst, h * 32 + l * 8 + 4 + j, db * iq_w4(g2, j, signs >> 4));
            }
        }
    }
}

// IQ2_S: 82-byte block; per sub-block ib: four low index bytes at 2 + 4*ib,
// four sign bytes at 34 + 4*ib (one bit per weight, no table), two high
// index bits per index in the byte at 66 + ib, two 4-bit scales in the
// byte at 74 + ib, scale (0.5 + nibble) / 4, the low nibble for the first
// 16 weights.
IQ_FN void iq2s_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                        const iq_u64 *grid) {
    float d = iq_f16(blk);
    for (int h = 0; h < 2; h++) {
        int ib = 2 * seg + h;
        const iq_byte *qs = blk + 2 + 4 * ib, *sg = blk + 34 + 4 * ib;
        unsigned hb = blk[66 + ib], sc = blk[74 + ib];
        float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            iq_u64 g = grid[qs[l] | ((hb << (8 - 2 * l)) & 0x300)];
            unsigned signs = sg[l];
            float db = l < 2 ? db0 : db1;
            for (int j = 0; j < 8; j++)
                IQ_STORE(dst, h * 32 + l * 8 + j, db * iq_w8(g, j, signs));
        }
    }
}

// IQ2_XS: 74-byte block; per sub-block ib four 16-bit words at 2 + 8*ib,
// each a 9-bit grid index and a 7-bit sign index, and the scale byte at
// 66 + ib as in IQ2_S.
IQ_FN void iq2xs_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                         const iq_u64 *grid) {
    float d = iq_f16(blk);
    for (int h = 0; h < 2; h++) {
        int ib = 2 * seg + h;
        const iq_byte *qs = blk + 2 + 8 * ib;
        unsigned sc = blk[66 + ib];
        float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            unsigned q = iq_ld16(qs + 2 * l);
            iq_u64 g = grid[q & 511];
            unsigned signs = iq_signs7(q >> 9);
            float db = l < 2 ? db0 : db1;
            for (int j = 0; j < 8; j++)
                IQ_STORE(dst, h * 32 + l * 8 + j, db * iq_w8(g, j, signs));
        }
    }
}

// IQ2_XXS: 66-byte block; per sub-block ib eight bytes at 2 + 8*ib: four
// grid indices, then one word of four 7-bit sign indices and a 4-bit
// scale, scale (0.5 + nibble) / 4.
IQ_FN void iq2xxs_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                          const iq_u64 *grid) {
    float d = iq_f16(blk);
    for (int h = 0; h < 2; h++) {
        int ib = 2 * seg + h;
        const iq_byte *qs = blk + 2 + 8 * ib;
        unsigned aux = iq_ld32a2(qs + 4);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            iq_u64 g = grid[qs[l]];
            unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
            for (int j = 0; j < 8; j++)
                IQ_STORE(dst, h * 32 + l * 8 + j, db * iq_w8(g, j, signs));
        }
    }
}


// IQ1_S: 50-byte block; per sub-block ib four low index bytes at 2 + 4*ib
// and one 16-bit word at 34 + 2*ib: three high index bits per index (bits
// 0-11), a 3-bit scale (bits 12-14, decoded 1 + 2*scale) and the sign of
// the shared 1/8 delta (bit 15). Grid bytes are signed.
IQ_FN void iq1s_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                        const iq_u64 *grid) {
    float d = iq_f16(blk);
    for (int h = 0; h < 2; h++) {
        int ib = 2 * seg + h;
        const iq_byte *qs = blk + 2 + 4 * ib;
        unsigned w = iq_ld16(blk + 34 + 2 * ib);
        float dl = d * (float)(2 * ((w >> 12) & 7) + 1);
        float delta = w & 0x8000 ? -IQ1_DELTA : IQ1_DELTA;
        for (int l = 0; l < 4; l++) {
            iq_u64 g = grid[qs[l] | (((w >> (3 * l)) & 7) << 8)];
            for (int j = 0; j < 8; j++)
                IQ_STORE(dst, h * 32 + l * 8 + j, dl * iq1_w(g, j, delta));
        }
    }
}

// IQ1_M: 56-byte block with no leading half: 32 low index bytes, 16 bytes
// of high bits (per byte two 3-bit index extensions at bits 0-2 and 4-6,
// and two delta signs at bits 3 and 7), then four 16-bit scale words. Each
// word holds two 3-bit scales per sub-block of 32 (one per 16 weights,
// decoded 1 + 2*scale) in its low 12 bits, and the block's half scale is
// scattered across the four top nibbles.
IQ_FN void iq1m_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                        const iq_u64 *grid) {
    const iq_byte *scb = blk + 48;
    unsigned sc0 = iq_ld16(scb), sc1 = iq_ld16(scb + 2),
             sc2 = iq_ld16(scb + 4), sc3 = iq_ld16(scb + 6);
    float d = iq_f16_bits((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
                          ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
    for (int h = 0; h < 2; h++) {
        int ib = 2 * seg + h;
        const iq_byte *qs = blk + 4 * ib, *qh = blk + 32 + 2 * ib;
        unsigned sw = iq_ld16(scb + 2 * (ib >> 1)) >> (6 * (ib & 1));
        float dl0 = d * (float)(2 * (sw & 7) + 1);
        float dl1 = d * (float)(2 * ((sw >> 3) & 7) + 1);
        for (int l = 0; l < 4; l++) {
            unsigned hb = qh[l >> 1];
            unsigned idx = qs[l] | ((hb << ((l & 1) ? 4 : 8)) & 0x700);
            float delta = hb & ((l & 1) ? 0x80 : 0x08) ? -IQ1_DELTA : IQ1_DELTA;
            iq_u64 g = grid[idx];
            float dl = l < 2 ? dl0 : dl1;
            for (int j = 0; j < 8; j++)
                IQ_STORE(dst, h * 32 + l * 8 + j, dl * iq1_w(g, j, delta));
        }
    }
}


// ------------------------------------------------- the IQ4 codebook types
// IQ4_NL and IQ4_XS index a fixed 16-entry signed codebook with each
// nibble; the caller passes the table (kv_iq4 on the device, the host
// test's own copy of the format's values).

// IQ4_NL: 18-byte block of 32 weights, d (half) + 16 bytes of nibbles; the
// low nibble of byte j is weight j, the high nibble weight j + 16.
IQ_FN void iq4nl_stage32(iq_stage_t *dst, const iq_byte *blk, const signed char *kv) {
    float d = iq_f16(blk);
    const iq_byte *q = blk + 2;
    for (int j = 0; j < 16; j++) {
        IQ_STORE(dst, j, d * (float)kv[q[j] & 0xF]);
        IQ_STORE(dst, j + 16, d * (float)kv[q[j] >> 4]);
    }
}

// IQ4_XS: 136-byte block of 256 weights: d (half), a 16-bit word of the
// scales' high two bits, four bytes of their low nibbles, then 128 bytes
// of nibbles; sub-block ib (32 weights) has scale (ls - 32) with ls the
// 6-bit value assembled from both fields. Segment s is sub-blocks 2s, 2s+1.
IQ_FN void iq4xs_stage64(iq_stage_t *dst, const iq_byte *blk, int seg,
                         const signed char *kv) {
    float d = iq_f16(blk);
    unsigned sh = iq_ld16(blk + 2);
    const iq_byte *sl = blk + 4;
    for (int h = 0; h < 2; h++) {
        int ib = 2 * seg + h;
        int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
        float dl = d * (float)(ls - 32);
        const iq_byte *q = blk + 8 + 16 * ib;
        for (int j = 0; j < 16; j++) {
            IQ_STORE(dst, h * 32 + j, dl * (float)kv[q[j] & 0xF]);
            IQ_STORE(dst, h * 32 + j + 16, dl * (float)kv[q[j] >> 4]);
        }
    }
}

#endif
