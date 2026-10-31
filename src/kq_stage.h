// K-quant and legacy-quant decoders for the tensor-core prefill, shared by
// the CUDA kernels and a host test, the way iq_decode.h is for the codebook
// i-quants: nvcc compiles this file into the stagers of kernels_kq_stage.cuh,
// and the C compiler into tests/test_kq_stage.c, which holds every segment
// to quants.c's dequant_row() bit for bit. Each function writes what that
// decoder computes for the same elements, in its per-element expression;
// on the device nvcc may contract a multiply-add into one FMA, which moves
// the last bit of a value before it is rounded to fp16 for the tensor-core
// operand (a tolerance-gated route either way).
//
// Until 2026-10-21 these six types had no tensor-core kernel, so a prompt
// ran their weights through the scalar GEMM (Q5_K, Q3_K) or the tile matvec
// (Q4_1, Q5_0, Q5_1, Q2_K): Q5_K_M and Q3_K_M files, and the Q5_K / Q4_1
// tensors a Q4_0 export mixes in, prefilled far slower than the types
// beside them.
#ifndef RUNNER_KQ_STAGE_H
#define RUNNER_KQ_STAGE_H

#include "iq_decode.h"   // IQ_FN, iq_stage_t, IQ_STORE, iq_f16, iq_byte

// get_scale_min_k4 of quants.c: the 6-bit scale and min of sub-block j
// (0..7) from a Q4_K / Q5_K block's 12 packed scale bytes
IQ_FN void kq_scale_min_k4(int j, const iq_byte *q, unsigned *d, unsigned *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

// ------------------------------------- 256-element K-quants, a segment each
// Segment s is elements [64s, 64s + 64) of the block, as in iq_decode.h.

// Q5_K: 176-byte block (d, dmin, 12 scale bytes, qh[32], qs[128]). Segment
// s is dq_q5_K's iteration j = 64s: sub-blocks 2s and 2s + 1, quant bytes
// qs + 32s (low nibbles, then high), high bits 2s and 2s + 1 of qh.
IQ_FN void q5k_stage64(iq_stage_t *dst, const iq_byte *blk, int seg) {
    float d = iq_f16(blk), dmin = iq_f16(blk + 2);
    unsigned sc, mn;
    kq_scale_min_k4(2 * seg, blk + 4, &sc, &mn);
    float d1 = d * sc, m1 = dmin * mn;
    kq_scale_min_k4(2 * seg + 1, blk + 4, &sc, &mn);
    float d2 = d * sc, m2 = dmin * mn;
    const iq_byte *q = blk + 48 + 32 * seg, *qh = blk + 16;
    unsigned u1 = 1u << (2 * seg), u2 = 2u << (2 * seg);
    for (int l = 0; l < 32; l++)
        IQ_STORE(dst, l, d1 * ((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1);
    for (int l = 0; l < 32; l++)
        IQ_STORE(dst, l + 32, d2 * ((q[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2);
}

// Q3_K: 110-byte block (hmask[32], qs[64], 12 scale bytes, d). Segment s is
// half n = s / 2 of dq_q3_K and its shifts j = 2 (s % 2) and j + 1: quant
// bytes qs + 32n, high-mask bit 4n + j, signed 6-bit scales 8n + 2j (and + 1
// for the second 16 elements), unpacked as dq_q3_K does.
IQ_FN void q3k_stage64(iq_stage_t *dst, const iq_byte *blk, int seg) {
    const unsigned kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const iq_byte *s = blk + 96;
    unsigned a0 = s[0] | (s[1] << 8) | (s[2] << 16) | ((unsigned)s[3] << 24);
    unsigned a1 = s[4] | (s[5] << 8) | (s[6] << 16) | ((unsigned)s[7] << 24);
    unsigned a2 = s[8] | (s[9] << 8) | (s[10] << 16) | ((unsigned)s[11] << 24);
    unsigned aux[4];
    aux[0] = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4);
    aux[1] = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4);
    aux[2] = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4);
    aux[3] = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4);
    float d_all = iq_f16(blk + 108);
    int n = seg >> 1;
    const iq_byte *q = blk + 32 + 32 * n, *hm = blk;
    for (int jj = 0; jj < 2; jj++) {
        int j = 2 * (seg & 1) + jj, shift = 2 * j, is = 8 * n + 2 * j;
        unsigned m = 1u << (4 * n + j);
        int sc0 = (signed char)(aux[is >> 2] >> (8 * (is & 3)));
        int sc1 = (signed char)(aux[(is + 1) >> 2] >> (8 * ((is + 1) & 3)));
        float dl = d_all * (sc0 - 32);
        for (int l = 0; l < 16; l++)
            IQ_STORE(dst, 32 * jj + l,
                     dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
        dl = d_all * (sc1 - 32);
        for (int l = 0; l < 16; l++)
            IQ_STORE(dst, 32 * jj + 16 + l,
                     dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)));
    }
}

// Q2_K: 84-byte block (16 scale bytes, qs[64], d, dmin). Segment s is half
// n = s / 2 of dq_q2_K and its shifts j = 2 (s % 2) and j + 1: quant bytes
// qs + 32n, scale bytes 8n + 2j and 8n + 2j + 1 (scale low, min high).
IQ_FN void q2k_stage64(iq_stage_t *dst, const iq_byte *blk, int seg) {
    float d = iq_f16(blk + 80), dmin = iq_f16(blk + 82);
    int n = seg >> 1;
    const iq_byte *q = blk + 16 + 32 * n;
    for (int jj = 0; jj < 2; jj++) {
        int j = 2 * (seg & 1) + jj, shift = 2 * j;
        unsigned sc = blk[8 * n + 2 * j];
        float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
        for (int l = 0; l < 16; l++)
            IQ_STORE(dst, 32 * jj + l, dl * ((q[l] >> shift) & 3) - ml);
        sc = blk[8 * n + 2 * j + 1];
        dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
        for (int l = 0; l < 16; l++)
            IQ_STORE(dst, 32 * jj + 16 + l, dl * ((q[l + 16] >> shift) & 3) - ml);
    }
}

// ------------------------------------------ 32-element legacy quants, a block
// Q4_1: 20-byte block (d, m, qs[16]): dq_q4_1.
IQ_FN void q41_stage32(iq_stage_t *dst, const iq_byte *blk) {
    float d = iq_f16(blk), m = iq_f16(blk + 2);
    const iq_byte *qs = blk + 4;
    for (int j = 0; j < 16; j++) {
        IQ_STORE(dst, j, (qs[j] & 0xF) * d + m);
        IQ_STORE(dst, j + 16, (qs[j] >> 4) * d + m);
    }
}

// Q5_0: 22-byte block (d, qh as four bytes, qs[16]): dq_q5_0.
IQ_FN void q50_stage32(iq_stage_t *dst, const iq_byte *blk) {
    float d = iq_f16(blk);
    unsigned qh = blk[2] | (blk[3] << 8) | (blk[4] << 16) | ((unsigned)blk[5] << 24);
    const iq_byte *qs = blk + 6;
    for (int j = 0; j < 16; j++) {
        int x0 = (qs[j] & 0xF) | (((qh >> j) & 1) << 4);
        int x1 = (qs[j] >> 4)  | (((qh >> (j + 16)) & 1) << 4);
        IQ_STORE(dst, j, (x0 - 16) * d);
        IQ_STORE(dst, j + 16, (x1 - 16) * d);
    }
}

// Q5_1: 24-byte block (d, m, qh as four bytes, qs[16]): dq_q5_1.
IQ_FN void q51_stage32(iq_stage_t *dst, const iq_byte *blk) {
    float d = iq_f16(blk), m = iq_f16(blk + 2);
    unsigned qh = blk[4] | (blk[5] << 8) | (blk[6] << 16) | ((unsigned)blk[7] << 24);
    const iq_byte *qs = blk + 8;
    for (int j = 0; j < 16; j++) {
        int x0 = (qs[j] & 0xF) | (((qh >> j) & 1) << 4);
        int x1 = (qs[j] >> 4)  | (((qh >> (j + 16)) & 1) << 4);
        IQ_STORE(dst, j, x0 * d + m);
        IQ_STORE(dst, j + 16, x1 * d + m);
    }
}

// ------------------------------------------ NVFP4, a 64-element block
// 36-byte block: four UE4M3 scale bytes (one per 16 elements), then 32 bytes
// of E2M1 codes; within sub-block s, byte j's low nibble is element j and its
// high nibble element j + 8 (llama.cpp block_nvfp4): dq_nvfp4. The scale and
// the code values are built from their bits, the same on both sides, so the
// staged floats equal ue4m3_to_fp32() and kvalues_mxfp4[] exactly.
IQ_FN float kq_f32_bits(unsigned b) {
#if defined(__CUDACC__)
    return __int_as_float((int)b);
#else
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
#endif
}
// ue4m3_to_fp32 (quants.c): bits 3..6 the exponent, bits 0..2 the mantissa,
// bit 7 ignored, 0 and 0x7F zero
IQ_FN float kq_ue4m3(unsigned x) {
    x &= 0xFF;
    if (x == 0 || x == 0x7F) return 0.0f;
    unsigned e = (x >> 3) & 0xF, man = x & 7;
    if (e == 0) return (float)man * (1.0f / 512.0f);     // man * 2^-9, exact
    return kq_f32_bits(((e + 120u) << 23) | (man << 20));  // (1 + man/8) * 2^(e-7)
}
// kvalues_mxfp4[q]: {0, .5, 1, 1.5, 2, 3, 4, 6}, negated for q >= 8 except
// that entry 8 is +0
IQ_FN float kq_e2m1(unsigned q) {
    unsigned m = q & 7;
    float v = m < 4 ? 0.5f * (float)m : m == 4 ? 2.0f : m == 5 ? 3.0f : m == 6 ? 4.0f : 6.0f;
    return (q & 8) && m ? -v : v;
}
IQ_FN void nvfp4_stage64(iq_stage_t *dst, const iq_byte *blk) {
    for (int s = 0; s < 4; s++) {
        float d = kq_ue4m3(blk[s]);
        const iq_byte *q = blk + 4 + 8 * s;
        for (int j = 0; j < 8; j++) {
            IQ_STORE(dst, 16 * s + j, kq_e2m1(q[j] & 0xF) * d);
            IQ_STORE(dst, 16 * s + j + 8, kq_e2m1(q[j] >> 4) * d);
        }
    }
}

#endif // RUNNER_KQ_STAGE_H
