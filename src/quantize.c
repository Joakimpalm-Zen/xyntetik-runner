// GGUF requantizer: rewrite a model with its weight matrices converted to a
// smaller quantization so one downloaded model can be fitted to each node's
// RAM/VRAM. Metadata is copied verbatim; norms/biases/1-D tensors stay F32.
#include "quants.h"
#include "fp16.h"
#include "gguf.h"
#include "compat.h"
#include "json.h"
#include "model.h"   // model_lora_arch_ok (static inline)

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

// ---------------------------------------------------------------- rows
// The reference quantizers below (Q8_0, Q4_0, Q3_K, Q4_K, Q6_K, and the
// make_qx_quants / make_qkx2_quants searches) are ports of ggml's
// ggml-quants.c, byte-identical to it when both are built without
// floating-point contraction (this file always is: QUANTIZE_CFLAGS in the
// Makefile; tests/fixtures/quant-golden pins it on a real tensor): MIT,
// Copyright (c) 2023-2026 The ggml authors; the notice is
// LICENSES/ggml-MIT.txt (see NOTICE).

// ------------------------------------------------ IEEE subnormals for quant
// ggml computes quant block scales with IEEE subnormals (it is built without
// -ffast-math). The runner engine links crtfastmath.o (-ffast-math), which
// sets FTZ/DAZ in MXCSR process-wide; a subnormal scale d = amax/127 then
// flushes to zero and that block's codes come out zero. Clearing FTZ/DAZ for
// the offline quantize makes runner's Q8_0/q4_0 bytes match ggml's exactly.
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
typedef unsigned fp_denormal_state;
static inline fp_denormal_state fp_denormals_disable(void) {
    unsigned csr = _mm_getcsr();
    _mm_setcsr(csr & ~0x8040u); // clear FTZ (bit 15) and DAZ (bit 6)
    return csr;
}
static inline void fp_denormals_restore(fp_denormal_state s) { _mm_setcsr(s); }
#elif defined(__aarch64__)
typedef unsigned long fp_denormal_state;
static inline fp_denormal_state fp_denormals_disable(void) {
    unsigned long fpcr; __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr & ~(1UL << 24))); // clear FZ
    return fpcr;
}
static inline void fp_denormals_restore(fp_denormal_state s) { __asm__ volatile("msr fpcr, %0" :: "r"(s)); }
#else
typedef int fp_denormal_state;
static inline fp_denormal_state fp_denormals_disable(void) { return 0; }
static inline void fp_denormals_restore(fp_denormal_state s) { (void)s; }
#endif


typedef struct { f16_t d; uint8_t qs[16]; } wblock_q4_0;
typedef struct { uint8_t e; uint8_t qs[16]; } wblock_mxfp4;   // 17 bytes
typedef struct { f16_t d; int8_t qs[32]; }  wblock_q8_0;
typedef struct {
    uint8_t hmask[32], qs[64], scales[12];
    f16_t d;
} wblock_q3_K;
typedef struct {
    f16_t d, dmin;
    uint8_t scales[12], qs[128];
} wblock_q4_K;
typedef struct {
    uint8_t ql[128], qh[64];
    int8_t scales[16];
    f16_t d;
} wblock_q6_K;

static void quantize_row_q8_0(const float *x, uint8_t *dst, int n) {
    wblock_q8_0 *b = (wblock_q8_0 *)dst;
    for (int i = 0; i < n / 32; i++, b++) {
        const float *xp = x + i * 32;
        float amax = 0;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(xp[j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        b->d = f32_to_f16(d);
        for (int j = 0; j < 32; j++) b->qs[j] = (int8_t)roundf(xp[j] * id);
    }
}

// QAT grid-exact repack.
//
// A quantization-aware-trained checkpoint's weights already sit on the q4_0
// grid: every value is d*(q-8) for one per-block scale d and integer q in
// [0,15]. Requantizing such a block should be LOSSLESS, because the answer is
// already there to be read off.
//
// The derived-scale path below cannot read it off. `d = vmax / -8` recovers
// the true scale only when the block actually contains q == 0; a QAT block
// whose codes never reach 0 gets a scale that is too small, and the values at
// the far end of the range then saturate at q == 15 and come back wrong. On a
// 64-block fixture with codes 1..15 that is 1770 of 2048 values changed, worst
// error 0.4375 -- on a repack that should have been exact.
//
// So try to RECOVER the grid first. The smallest nonzero magnitude in the
// block is m*d for some integer m in [1,8], which gives eight candidate
// scales; take the largest that works, since a coarser grid that still
// reproduces every value is equally exact and leaves the codes better centred.
//
// The acceptance test is the whole safety argument: a candidate is accepted
// only if the value the DEQUANTIZER will compute -- f16_to_f32 of the stored
// scale, times the code -- equals the source float bit for bit, for all 32
// values. That folds the fp16 storage of the scale into the same check (a
// bf16-scaled source is exactly representable, since bf16's 7 mantissa bits
// fit fp16's 10, but only inside fp16's narrower exponent range), and it means
// imprecision anywhere can only cause a REJECTION, never a wrong write. On any
// rejection the derived-scale path runs unchanged, so a non-QAT model gets
// byte-identical output to before.
static bool q4_0_grid_exact(const float *xp, wblock_q4_0 *b) {
    float amin = 0;
    bool any = false;
    for (int j = 0; j < 32; j++) {
        float a = fabsf(xp[j]);
        if (a != 0.0f && (!any || a < amin)) { amin = a; any = true; }
    }
    if (!any) return false;      // all-zero block: the derived path is already exact

    for (int m = 1; m <= 8; m++) {
        f16_t dh = f32_to_f16(amin / (float)m);
        float ds = f16_to_f32(dh);
        if (ds == 0.0f) continue;            // underflowed fp16, or no f16 table
        uint8_t q[32];
        bool ok = true;
        for (int j = 0; j < 32; j++) {
            float kf = roundf(xp[j] / ds);
            if (kf < -8.0f || kf > 7.0f) { ok = false; break; }
            int k = (int)kf;
            if (ds * (float)k != xp[j]) { ok = false; break; }
            q[j] = (uint8_t)(k + 8);
        }
        if (!ok) continue;
        b->d = dh;
        for (int j = 0; j < 16; j++) b->qs[j] = (uint8_t)(q[j] | (q[j + 16] << 4));
        return true;
    }
    return false;
}

static void quantize_row_q4_0(const float *x, uint8_t *dst, int n) {
    wblock_q4_0 *b = (wblock_q4_0 *)dst;
    for (int i = 0; i < n / 32; i++, b++) {
        const float *xp = x + i * 32;
        if (q4_0_grid_exact(xp, b)) continue;
        float amax = 0, vmax = 0;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(xp[j]);
            if (v > amax) { amax = v; vmax = xp[j]; }
        }
        float d = vmax / -8.0f;
        float id = d ? 1.0f / d : 0.0f;
        b->d = f32_to_f16(d);
        for (int j = 0; j < 16; j++) {
            int q0 = (int)(xp[j] * id + 8.5f);
            int q1 = (int)(xp[j + 16] * id + 8.5f);
            if (q0 > 15) q0 = 15;
            if (q1 > 15) q1 = 15;
            if (q0 < 0) q0 = 0;
            if (q1 < 0) q1 = 0;
            b->qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

static int nearest_int(float v) {
    // Exact ggml reference rounding for K-quants (nearest, ties-to-even for
    // the default IEEE environment), avoiding a libm implementation choice.
    float biased = v + 12582912.0f;
    int bits;
    memcpy(&bits, &biased, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

static float make_q3_quants(const float *x, int8_t *q) {
    float vmax = 0.0f, amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        float a = fabsf(x[i]);
        if (a > amax) { amax = a; vmax = x[i]; }
    }
    if (amax < 1e-15f) {
        memset(q, 0, 16);
        return 0.0f;
    }
    float iscale = -4.0f / vmax;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < 16; i++) {
        int l = nearest_int(iscale * x[i]);
        if (l < -4) l = -4;
        if (l > 3) l = 3;
        q[i] = (int8_t)l;
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int changed = 0;
        for (int i = 0; i < 16; i++) {
            float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * q[i];
            if (slx <= 0.0f) continue;
            float sl2 = suml2 - w * q[i] * q[i];
            int l = nearest_int(x[i] * sl2 / slx);
            if (l < -4) l = -4;
            if (l > 3) l = 3;
            if (l == q[i]) continue;
            slx += w * x[i] * l;
            sl2 += w * l * l;
            if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                q[i] = (int8_t)l;
                sumlx = slx;
                suml2 = sl2;
                changed++;
            }
        }
        if (!changed) break;
    }
    for (int i = 0; i < 16; i++) q[i] += 4;
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

// GGML Q3_K's 256-value super-block: sixteen signed 3-bit groups, with the
// group scales themselves quantized to signed 6-bit values.  The layout is
// deliberately the inverse of dq_q3_K in quants.c; the hermetic writer test
// drives that existing reader, so a layout disagreement cannot pass.
static void quantize_row_q3_K(const float *x, uint8_t *dst, int n) {
    wblock_q3_K *out = (wblock_q3_K *)dst;
    for (int ib = 0; ib < n / 256; ib++, out++, x += 256) {
        int8_t q[256];
        float scales[16];
        float max_scale = 0.0f, amax = 0.0f;
        for (int j = 0; j < 16; j++) {
            scales[j] = make_q3_quants(x + 16 * j, q + 16 * j);
            float a = fabsf(scales[j]);
            if (a > amax) { amax = a; max_scale = scales[j]; }
        }
        memset(out, 0, sizeof(*out));
        if (max_scale != 0.0f) {
            float iscale = -32.0f / max_scale;
            for (int j = 0; j < 16; j++) {
                int l = nearest_int(iscale * scales[j]);
                if (l < -32) l = -32;
                if (l > 31) l = 31;
                l += 32;
                if (j < 8) out->scales[j] = (uint8_t)(l & 15);
                else out->scales[j - 8] |= (uint8_t)((l & 15) << 4);
                out->scales[j % 4 + 8] |= (uint8_t)((l >> 4) << (2 * (j / 4)));
            }
            out->d = f32_to_f16(1.0f / iscale);
        }
        for (int j = 0; j < 16; j++) {
            int sc = j < 8 ? out->scales[j] & 15 : out->scales[j - 8] >> 4;
            sc = (sc | (((out->scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
            float d = f16_to_f32(out->d) * sc;
            if (d == 0.0f) continue;
            for (int k = 0; k < 16; k++) {
                int l = nearest_int(x[16 * j + k] / d);
                if (l < -4) l = -4;
                if (l > 3) l = 3;
                q[16 * j + k] = (int8_t)(l + 4);
            }
        }
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < 256; j++) {
            if (q[j] > 3) { out->hmask[m] |= hm; q[j] -= 4; }
            if (++m == 32) { m = 0; hm <<= 1; }
        }
        for (int j = 0; j < 256; j += 128)
            for (int k = 0; k < 32; k++)
                out->qs[j / 4 + k] = (uint8_t)(q[j + k] | (q[j + k + 32] << 2) |
                    (q[j + k + 64] << 4) | (q[j + k + 96] << 6));
    }
}

// ggml's make_qx_quants specialized to the one call Q6_K makes of it
// (x^2 error weighting, the full ±9 candidate-scale search). Faithful port:
// a K-quant tensor this file writes must carry the bytes llama.cpp's own
// quantizer would emit for the same floats.
static float make_q6_quants(int n, int nmax, const float *x, int8_t *L) {
    float vmax = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; vmax = x[i]; }
    }
    if (amax < 1e-15f) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = -nmax / vmax;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; i++) {
        int l = nearest_int(iscale * x[i]);
        if (l < -nmax) l = -nmax;
        if (l > nmax - 1) l = nmax - 1;
        L[i] = (int8_t)(l + nmax);
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; is++) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / vmax;
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = nearest_int(iscale * x[i]);
            if (l < -nmax) l = -nmax;
            if (l > nmax - 1) l = nmax - 1;
            float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; i++) {
                int l = nearest_int(iscale * x[i]);
                if (l < -nmax) l = -nmax;
                if (l > nmax - 1) l = nmax - 1;
                L[i] = (int8_t)(l + nmax);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// GGML Q6_K: sixteen 16-value groups of signed 6-bit codes, group scales
// quantized to signed 8-bit against one f16 super-scale. Inverse of
// dq_q6_K in quants.c; the hermetic round-trip test drives that reader.
static void quantize_row_q6_K(const float *x, uint8_t *dst, int n) {
    wblock_q6_K *out = (wblock_q6_K *)dst;
    for (int ib = 0; ib < n / 256; ib++, out++, x += 256) {
        int8_t L[256];
        float scales[16];
        float max_scale = 0.0f, max_abs = 0.0f;
        for (int j = 0; j < 16; j++) {
            scales[j] = make_q6_quants(16, 32, x + 16 * j, L + 16 * j);
            float a = fabsf(scales[j]);
            if (a > max_abs) { max_abs = a; max_scale = scales[j]; }
        }
        memset(out, 0, sizeof(*out));
        if (max_abs < 1e-15f) continue;      // all-zero block, d stays 0
        float iscale = -128.0f / max_scale;
        out->d = f32_to_f16(1.0f / iscale);
        for (int j = 0; j < 16; j++) {
            int l = nearest_int(iscale * scales[j]);
            out->scales[j] = (int8_t)(l < 127 ? l : 127);
        }
        for (int j = 0; j < 16; j++) {
            // a group whose quantized scale rounded to 0 keeps its
            // make_q6_quants codes (they decode to 0 either way) — the
            // reference does the same, and byte fidelity to it matters
            float d = f16_to_f32(out->d) * (float)out->scales[j];
            if (d == 0.0f) continue;
            for (int k = 0; k < 16; k++) {
                int l = nearest_int(x[16 * j + k] / d);
                if (l < -32) l = -32;
                if (l > 31) l = 31;
                L[16 * j + k] = (int8_t)(l + 32);
            }
        }
        uint8_t *ql = out->ql, *qh = out->qh;
        for (int j = 0; j < 256; j += 128) {
            for (int k = 0; k < 32; k++) {
                uint8_t q1 = L[j + k]      & 0xF;
                uint8_t q2 = L[j + k + 32] & 0xF;
                uint8_t q3 = L[j + k + 64] & 0xF;
                uint8_t q4 = L[j + k + 96] & 0xF;
                ql[k]      = (uint8_t)(q1 | (q3 << 4));
                ql[k + 32] = (uint8_t)(q2 | (q4 << 4));
                qh[k] = (uint8_t)((L[j + k] >> 4) |
                                  ((L[j + k + 32] >> 4) << 2) |
                                  ((L[j + k + 64] >> 4) << 4) |
                                  ((L[j + k + 96] >> 4) << 6));
            }
            ql += 64; qh += 32;
        }
    }
}

// ggml's make_qkx2_quants with Q4_K's fixed search parameters (rmin -1,
// rdelta 0.1, 20 steps, squared-error weighting).
static float make_qkx2_quants(int n, int nmax, const float *x,
                              const float *weights, uint8_t *L,
                              float *the_min, uint8_t *Laux) {
    float min = x[0], max = x[0];
    float sum_w = weights[0], sum_x = weights[0] * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if (min > 0.0f) min = 0.0f;
    if (max == min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1.0f / iscale;
    float best_mad = 0.0f;
    for (int i = 0; i < n; i++) {
        int l = nearest_int(iscale * (x[i] - min));
        if (l < 0) l = 0;
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
        float diff = scale * l + min - x[i];
        best_mad += weights[i] * diff * diff;
    }
    for (int is = 0; is <= 20; is++) {
        iscale = (-1.0f + 0.1f * is + nmax) / (max - min);
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = nearest_int(iscale * (x[i] - min));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            Laux[i] = (uint8_t)l;
            sum_l  += weights[i] * l;
            sum_l2 += weights[i] * l * l;
            sum_xl += weights[i] * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D <= 0.0f) continue;
        float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
        float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
        if (this_min > 0.0f) {
            this_min = 0.0f;
            this_scale = sum_xl / sum_l2;   // D>0 guarantees sum_l2>0
        }
        float mad = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = this_scale * Laux[i] + this_min - x[i];
            mad += weights[i] * diff * diff;
        }
        if (mad < best_mad) {
            memcpy(L, Laux, (size_t)n);
            best_mad = mad;
            scale = this_scale;
            min = this_min;
        }
    }
    *the_min = -min;
    return scale;
}

// mirror of quants.c get_scale_min_k4 — the reader this writer must invert
static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

// GGML Q4_K: eight 32-value groups of unsigned 4-bit codes with per-group
// scale+min pairs packed to 6 bits against f16 d/dmin super-scales.
// Inverse of dq_q4_K in quants.c.
static void quantize_row_q4_K(const float *x, uint8_t *dst, int n) {
    wblock_q4_K *out = (wblock_q4_K *)dst;
    for (int ib = 0; ib < n / 256; ib++, out++, x += 256) {
        uint8_t L[256], Laux[32];
        float weights[32], mins[8], scales[8];
        float max_scale = 0.0f, max_min = 0.0f;
        for (int j = 0; j < 8; j++) {
            float sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++)
                sum_x2 += x[32 * j + l] * x[32 * j + l];
            float av_x = sqrtf(sum_x2 / 32.0f);
            for (int l = 0; l < 32; l++)
                weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights,
                                         L + 32 * j, &mins[j], Laux);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
        float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
        memset(out, 0, sizeof(*out));
        for (int j = 0; j < 8; j++) {
            uint8_t ls = (uint8_t)nearest_int(inv_scale * scales[j]);
            uint8_t lm = (uint8_t)nearest_int(inv_min * mins[j]);
            if (ls > 63) ls = 63;
            if (lm > 63) lm = 63;
            if (j < 4) {
                out->scales[j] = ls;
                out->scales[j + 4] = lm;
            } else {
                out->scales[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
                out->scales[j - 4] |= (uint8_t)((ls >> 4) << 6);
                out->scales[j]     |= (uint8_t)((lm >> 4) << 6);
            }
        }
        out->d = f32_to_f16(max_scale / 63.0f);
        out->dmin = f32_to_f16(max_min / 63.0f);
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            scale_min_k4(j, out->scales, &sc, &mn);
            float d = f16_to_f32(out->d) * sc;
            if (d == 0.0f) continue;
            float dm = f16_to_f32(out->dmin) * mn;
            for (int l = 0; l < 32; l++) {
                int q = nearest_int((x[32 * j + l] + dm) / d);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                L[32 * j + l] = (uint8_t)q;
            }
        }
        uint8_t *qs = out->qs;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; l++)
                qs[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
            qs += 32;
        }
    }
}

// OCP microscaling FP4 (R4.7.1), ggml's quantize_row_mxfp4_ref (MIT, The
// ggml authors; see NOTICE): one E8M0
// power-of-two scale per 32 values, 2^(floor(log2(amax)) - 2), and each value
// the nearest E2M1 level {0, .5, 1, 1.5, 2, 3, 4, 6} with its sign (the first
// of two equally near wins, as in ggml's best_index_mxfp4). floor(log2) comes
// from frexpf, exact on every libm. Byte layout as dq_mxfp4 in quants.c
// reads it: element j in the low nibble of qs[j], element j+16 in the high.
static const float mxfp4_levels[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static uint8_t mxfp4_nearest(float x, float d) {
    int best = 0;
    float err = fabsf(mxfp4_levels[0] * d - x);
    for (int i = 1; i < 16; i++) {
        float e = fabsf(mxfp4_levels[i] * d - x);
        if (e < err) { err = e; best = i; }
    }
    return (uint8_t)best;
}

static void quantize_row_mxfp4(const float *x, uint8_t *dst, int n) {
    wblock_mxfp4 *b = (wblock_mxfp4 *)dst;
    for (int i = 0; i < n / 32; i++, b++, x += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++)
            if (fabsf(x[j]) > amax) amax = fabsf(x[j]);
        int e = 0;
        if (amax > 0.0f) {
            int ex;
            frexpf(amax, &ex);              // amax = m * 2^ex, m in [0.5, 1)
            e = (ex - 1) - 2 + 127;
            if (e < 0) e = 0;
            if (e > 254) e = 254;           // 255 is E8M0's NaN
        }
        b->e = (uint8_t)e;
        float d = ldexpf(1.0f, e - 127);
        for (int j = 0; j < 16; j++)
            b->qs[j] = (uint8_t)(mxfp4_nearest(x[j], d) |
                                 (mxfp4_nearest(x[j + 16], d) << 4));
    }
}

// ---------------------------------------------- importance-weighted rows (R4.7.2)
// ggml's make_qx_quants with explicit weights (rmse_type 1 and a quant_weights
// array): the scale that minimises the WEIGHTED squared error over the
// nearest-integer levels, searched at nineteen candidate scales. With an
// importance matrix the weight of column i is qw[i] * sqrt(sigma2 + x[i]^2),
// so a column the calibration text drives hard keeps its precision at the
// expense of one it barely uses. Ported from ggml's quantize_row_q3_K_impl
// and quantize_row_q4_0_impl (MIT, The ggml authors; see NOTICE); without an
// imatrix nothing here runs and the reference writers above emit their usual
// bytes.
static float make_qx_quants_w(int n, int nmax, const float *x, int8_t *L,
                              const float *w) {
    float vmax = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; vmax = x[i]; }
    }
    if (amax < 1e-15f) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = -nmax / vmax;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; i++) {
        int l = nearest_int(iscale * x[i]);
        if (l < -nmax) l = -nmax;
        if (l > nmax - 1) l = nmax - 1;
        L[i] = (int8_t)(l + nmax);
        sumlx += w[i] * x[i] * l;
        suml2 += w[i] * l * l;
    }
    float scale = suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; is++) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / vmax;
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = nearest_int(iscale * x[i]);
            if (l < -nmax) l = -nmax;
            if (l > nmax - 1) l = nmax - 1;
            sumlx += w[i] * x[i] * l;
            suml2 += w[i] * l * l;
        }
        if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; i++) {
                int l = nearest_int(iscale * x[i]);
                if (l < -nmax) l = -nmax;
                if (l > nmax - 1) l = nmax - 1;
                L[i] = (int8_t)(l + nmax);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

static void quantize_row_q3_K_w(const float *x, uint8_t *dst, int n, const float *qw) {
    wblock_q3_K *out = (wblock_q3_K *)dst;
    for (int ib = 0; ib < n / 256; ib++, out++, x += 256, qw += 256) {
        int8_t L[256], Ls[16];
        float scales[16], weight[16], sw[16];
        float sumx2 = 0.0f;
        for (int j = 0; j < 256; j++) sumx2 += x[j] * x[j];
        float sigma2 = 2.0f * sumx2 / 256.0f;
        for (int j = 0; j < 16; j++) {
            float sumw = 0.0f;
            for (int l = 0; l < 16; l++) {
                float v = x[16 * j + l];
                weight[l] = qw[16 * j + l] * sqrtf(sigma2 + v * v);
                sumw += weight[l];
            }
            sw[j] = sumw;
            scales[j] = make_qx_quants_w(16, 4, x + 16 * j, L + 16 * j, weight);
        }
        memset(out, 0, sizeof(*out));
        float d_block = make_qx_quants_w(16, 32, scales, Ls, sw);
        for (int j = 0; j < 16; j++) {
            int l = Ls[j];
            if (j < 8) out->scales[j] = (uint8_t)(l & 15);
            else out->scales[j - 8] |= (uint8_t)((l & 15) << 4);
            out->scales[j % 4 + 8] |= (uint8_t)((l >> 4) << (2 * (j / 4)));
        }
        out->d = f32_to_f16(d_block);
        for (int j = 0; j < 16; j++) {
            int sc = j < 8 ? out->scales[j] & 15 : out->scales[j - 8] >> 4;
            sc = (sc | (((out->scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
            float d = f16_to_f32(out->d) * sc;
            if (d == 0.0f) continue;
            for (int k = 0; k < 16; k++) {
                int l = nearest_int(x[16 * j + k] / d);
                if (l < -4) l = -4;
                if (l > 3) l = 3;
                L[16 * j + k] = (int8_t)(l + 4);
            }
        }
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < 256; j++) {
            if (L[j] > 3) { out->hmask[m] |= hm; L[j] -= 4; }
            if (++m == 32) { m = 0; hm <<= 1; }
        }
        for (int j = 0; j < 256; j += 128)
            for (int k = 0; k < 32; k++)
                out->qs[j / 4 + k] = (uint8_t)(L[j + k] | (L[j + k + 32] << 2) |
                    (L[j + k + 64] << 4) | (L[j + k + 96] << 6));
    }
}

static void quantize_row_q4_0_w(const float *x, uint8_t *dst, int n, const float *qw) {
    wblock_q4_0 *b = (wblock_q4_0 *)dst;
    float sumx2 = 0.0f;
    for (int j = 0; j < n; j++) sumx2 += x[j] * x[j];
    float sigma2 = sumx2 / (float)n;
    for (int ib = 0; ib < n / 32; ib++, b++) {
        const float *xb = x + 32 * ib, *wb = qw + 32 * ib;
        float weight[32];
        int8_t L[32];
        for (int j = 0; j < 32; j++) weight[j] = wb[j] * sqrtf(sigma2 + xb[j] * xb[j]);
        b->d = f32_to_f16(make_qx_quants_w(32, 8, xb, L, weight));
        for (int j = 0; j < 16; j++) b->qs[j] = (uint8_t)(L[j] | (L[j + 16] << 4));
    }
}

static void quantize_row_f16(const float *x, uint8_t *dst, int n) {
    f16_t *h = (f16_t *)dst;
    for (int i = 0; i < n; i++) h[i] = f32_to_f16(x[i]);
}

// ggml_compute_fp32_to_bf16 semantics: round to nearest even, NaN forced
// quiet — a truncation here would not match what a bf16 GGUF built by the
// reference converter carries.
static void quantize_row_bf16(const float *x, uint8_t *dst, int n) {
    uint16_t *h = (uint16_t *)dst;
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &x[i], 4);
        if ((u & 0x7fffffff) > 0x7f800000)
            h[i] = (uint16_t)((u >> 16) | 64);
        else
            h[i] = (uint16_t)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
    }
}

// One row, f32 -> `type`. Everything this file can write goes through here;
// type_from_name, the whole-file target validator and quantize_row_writable
// must stay in sync with this switch.
static void quantize_row_any(int type, const float *rowf, uint8_t *rowq, int n) {
    switch (type) {
        case T_Q8_0: quantize_row_q8_0(rowf, rowq, n); break;
        case T_Q4_0: quantize_row_q4_0(rowf, rowq, n); break;
        case T_Q3_K: quantize_row_q3_K(rowf, rowq, n); break;
        case T_Q4_K: quantize_row_q4_K(rowf, rowq, n); break;
        case T_Q6_K: quantize_row_q6_K(rowf, rowq, n); break;
        case T_MXFP4: quantize_row_mxfp4(rowf, rowq, n); break;
        case T_F16:  quantize_row_f16(rowf, rowq, n); break;
        case T_BF16: quantize_row_bf16(rowf, rowq, n); break;
        default:     memcpy(rowq, rowf, sizeof(float) * (size_t)n); break;
    }
}

static bool quantize_row_writable(int type) {
    return type == T_F32 || type == T_F16 || type == T_BF16 ||
           type == T_Q8_0 || type == T_Q4_0 || type == T_Q3_K ||
           type == T_Q4_K || type == T_Q6_K || type == T_MXFP4;
}

// The types an importance matrix shapes; every other type is written as
// without one.
static bool imatrix_weighted_type(int type) {
    return type == T_Q3_K || type == T_Q4_0;
}

// One row with its importance weights (NULL: the reference writer).
static void quantize_row_any_w(int type, const float *rowf, uint8_t *rowq, int n,
                               const float *qw) {
    if (qw && type == T_Q3_K) quantize_row_q3_K_w(rowf, rowq, n, qw);
    else if (qw && type == T_Q4_0) quantize_row_q4_0_w(rowf, rowq, n, qw);
    else quantize_row_any(type, rowf, rowq, n);
}

// ---------------------------------------------------------------- imatrix
// An importance matrix: per weight tensor, the mean square of each input
// column over a calibration text (one row of n_per_row values per matrix;
// a stacked expert tensor has one per expert). Read from llama.cpp's two
// file formats, so a published imatrix serves as it is:
//   - GGUF: tensors "<name>.in_sum2" (F32, [n_per_row, n_mat]) and
//     "<name>.counts" (F32, [1, n_mat]); the values are in_sum2 / counts;
//   - legacy .dat: int32 n_entries, then per entry int32 name length, the
//     name, int32 ncall, int32 nval, nval float32 values.
// Only the ratios inside one matrix shape the quantization, so the two
// formats' different normalisations give the same bytes. A matrix whose
// values are all zero (an expert the calibration never routed to) carries
// no information and its rows are written as without an imatrix.
typedef struct {
    char   name[128];
    int64_t n_per_row, n_mat;
    float *v;                       // n_per_row * n_mat
} imatrix_entry;

typedef struct {
    imatrix_entry *e;
    int n;
    // what it did, for the summary
    uint64_t tensors_weighted, rows_weighted, rows_without_data;
    uint64_t tensors_other_type;
} imatrix_t;

static imatrix_t g_imx;

static void imatrix_free(void) {
    for (int i = 0; i < g_imx.n; i++) free(g_imx.e[i].v);
    free(g_imx.e);
    memset(&g_imx, 0, sizeof(g_imx));
}

static imatrix_entry *imatrix_add(const char *name, int64_t n_per_row, int64_t n_mat) {
    if (strlen(name) >= sizeof(g_imx.e[0].name) || n_per_row <= 0 || n_mat <= 0 ||
        (uint64_t)n_per_row > SIZE_MAX / sizeof(float) / (uint64_t)n_mat)
        return NULL;
    imatrix_entry *ne = realloc(g_imx.e, sizeof(*ne) * (size_t)(g_imx.n + 1));
    if (!ne) return NULL;
    g_imx.e = ne;
    imatrix_entry *e = &g_imx.e[g_imx.n];
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->n_per_row = n_per_row;
    e->n_mat = n_mat;
    e->v = malloc(sizeof(float) * (size_t)(n_per_row * n_mat));
    if (!e->v) return NULL;
    g_imx.n++;
    return e;
}

static bool imatrix_load_gguf(const char *path) {
    gguf_file g;
    if (!gguf_open(&g, path)) return false;
    bool ok = true;
    for (uint64_t i = 0; ok && i < g.n_tensors; i++) {
        const gguf_tensor *t = &g.tensors[i];
        size_t ln = strlen(t->name);
        if (ln <= 8 || strcmp(t->name + ln - 8, ".in_sum2")) continue;
        char base[128], cname[140];
        snprintf(base, sizeof(base), "%.*s", (int)(ln - 8), t->name);
        snprintf(cname, sizeof(cname), "%s.counts", base);
        const gguf_tensor *c = NULL;
        for (uint64_t k = 0; k < g.n_tensors; k++)
            if (!strcmp(g.tensors[k].name, cname)) { c = &g.tensors[k]; break; }
        int64_t npr = (int64_t)t->ne[0], nmat = (int64_t)(t->n_dims > 1 ? t->ne[1] : 1);
        if (t->type != T_F32 || !c || c->type != T_F32 ||
            (int64_t)(c->ne[0] * (c->n_dims > 1 ? c->ne[1] : 1)) != nmat) {
            fprintf(stderr, "error: imatrix %s: %s is not an F32 in_sum2/counts pair\n",
                    path, base);
            ok = false;
            break;
        }
        imatrix_entry *e = imatrix_add(base, npr, nmat);
        if (!e) { ok = false; break; }
        const float *s2 = t->data, *cnt = c->data;
        for (int64_t m = 0; m < nmat; m++)
            for (int64_t j = 0; j < npr; j++)
                e->v[m * npr + j] = cnt[m] > 0.0f ? s2[m * npr + j] / cnt[m] : 0.0f;
    }
    gguf_close(&g);
    return ok;
}

static bool imatrix_load_dat(const char *path) {
    FILE *f = plat_fopen(path, "rb");
    if (!f) return false;
    int32_t n = 0;
    bool ok = fread(&n, 4, 1, f) == 1 && n > 0 && n < 1000000;
    for (int32_t i = 0; ok && i < n; i++) {
        int32_t len = 0, ncall = 0, nval = 0;
        char name[128];
        ok = fread(&len, 4, 1, f) == 1 && len > 0 && len < (int32_t)sizeof(name) &&
             fread(name, 1, (size_t)len, f) == (size_t)len;
        if (!ok) break;
        name[len] = 0;
        ok = fread(&ncall, 4, 1, f) == 1 && fread(&nval, 4, 1, f) == 1 && nval > 0;
        if (!ok) break;
        // the row length is the tensor's; the matrix count is resolved
        // against the model when the tensor is written (imatrix_row)
        imatrix_entry *e = imatrix_add(name, nval, 1);
        ok = e && fread(e->v, sizeof(float), (size_t)nval, f) == (size_t)nval;
    }
    fclose(f);
    return ok;
}

bool quantize_set_imatrix(const char *path) {
    imatrix_free();
    if (!path) return true;
    FILE *f = plat_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open imatrix %s\n", path);
        return false;
    }
    char magic[4] = {0};
    size_t got = fread(magic, 1, 4, f);
    fclose(f);
    bool ok = got == 4 && !memcmp(magic, "GGUF", 4) ? imatrix_load_gguf(path)
                                                    : imatrix_load_dat(path);
    for (int i = 0; ok && i < g_imx.n; i++)
        for (int64_t j = 0; j < g_imx.e[i].n_per_row * g_imx.e[i].n_mat; j++)
            if (!isfinite(g_imx.e[i].v[j]) || g_imx.e[i].v[j] < 0.0f) {
                fprintf(stderr, "error: imatrix %s: %s holds a negative or "
                        "non-finite value\n", path, g_imx.e[i].name);
                ok = false;
                break;
            }
    if (!ok || g_imx.n == 0) {
        if (ok) fprintf(stderr, "error: imatrix %s holds no matrices\n", path);
        else fprintf(stderr, "error: %s is not a readable imatrix (GGUF or "
                     "legacy .dat)\n", path);
        imatrix_free();
        return false;
    }
    return true;
}

static const imatrix_entry *imatrix_find(const char *tname) {
    for (int i = 0; i < g_imx.n; i++)
        if (!strcmp(g_imx.e[i].name, tname)) return &g_imx.e[i];
    return NULL;
}

// The weights for row `row` of a tensor with `mats` matrices of
// `rows_per_mat` rows each, or NULL (no imatrix, no entry, an all-zero
// matrix). *bad is set when the entry exists and does not fit the tensor.
static const float *imatrix_row(const imatrix_entry *e, int64_t row_len,
                                uint64_t row, uint64_t rows_per_mat, uint64_t mats,
                                bool *bad) {
    if (!e) return NULL;
    int64_t npr = e->n_per_row, nmat = e->n_mat;
    if (nmat == 1 && npr == row_len * (int64_t)mats) { npr = row_len; nmat = (int64_t)mats; }
    if (npr != row_len || (nmat != 1 && nmat != (int64_t)mats)) { *bad = true; return NULL; }
    uint64_t m = nmat == 1 ? 0 : row / rows_per_mat;
    const float *w = e->v + m * (uint64_t)npr;
    for (int64_t j = 0; j < npr; j++)
        if (w[j] > 0.0f) return w;
    return NULL;
}

// ---------------------------------------------------------------- writer

typedef struct {
    FILE *f;
    bool ok;
} writer;

static void wr(writer *w, const void *p, size_t n) {
    if (w->ok && n > 0 && fwrite(p, 1, n, w->f) != n) w->ok = false;
}
static void wr_u32(writer *w, uint32_t v) { wr(w, &v, 4); }
static void wr_u64(writer *w, uint64_t v) { wr(w, &v, 8); }
static void wr_str(writer *w, const char *s, uint64_t n) {
    wr_u64(w, n);
    if (n > SIZE_MAX) { w->ok = false; return; }
    wr(w, s, (size_t)n);
}

static int64_t wr_tell(writer *w) {
    if (!w->ok) return -1;
#ifdef _WIN32
    __int64 pos = _ftelli64(w->f);
#else
    off_t pos = ftello(w->f);
#endif
    if (pos < 0) w->ok = false;
    return (int64_t)pos;
}

static void wr_pad_to(writer *w, uint64_t target) {
    int64_t signed_pos = wr_tell(w);
    if (signed_pos < 0) return;
    uint64_t pos = (uint64_t)signed_pos;
    if (pos > target) { w->ok = false; return; }
    static const uint8_t zeros[4096] = {0};
    while (w->ok && pos < target) {
        size_t n = target - pos < sizeof(zeros) ? (size_t)(target - pos)
                                                 : sizeof(zeros);
        wr(w, zeros, n);
        pos += n;
    }
}

static bool wr_close(writer *w) {
    bool ok = w->ok;
    // The caller installs this file with rename() on success. rename() orders
    // the DIRECTORY entry, not the data: after a crash the destination can
    // exist full-length with an unwritten tail of zeros -- a correctly-named
    // model that fails checksum (or worse, loads). Flush user-space buffers
    // and force the data to stable storage BEFORE the atomic install, so the
    // rename can only ever publish bytes that survive a power cut.
    if (ok && fflush(w->f) != 0) ok = false;
#ifndef _WIN32
    if (ok && fsync(fileno(w->f)) != 0) ok = false;
#else
    if (ok && _commit(_fileno(w->f)) != 0) ok = false;
#endif
    if (fclose(w->f) != 0) ok = false;
    w->f = NULL;
    w->ok = ok;
    return ok;
}

static const size_t kv_scalar_size[] = {
    [GGUF_T_U8] = 1, [GGUF_T_I8] = 1, [GGUF_T_U16] = 2, [GGUF_T_I16] = 2,
    [GGUF_T_U32] = 4, [GGUF_T_I32] = 4, [GGUF_T_F32] = 4, [GGUF_T_BOOL] = 1,
    [GGUF_T_U64] = 8, [GGUF_T_I64] = 8, [GGUF_T_F64] = 8,
};

static void wr_scalar(writer *w, const gguf_kv *kv, uint32_t type) {
    uint8_t b1; uint16_t b2; uint32_t b4; uint64_t b8;
    switch (type) {
        case GGUF_T_U8:   b1 = (uint8_t)kv->v.u64;  wr(w, &b1, 1); break;
        case GGUF_T_I8:   b1 = (uint8_t)(int8_t)kv->v.i64; wr(w, &b1, 1); break;
        case GGUF_T_BOOL: b1 = kv->v.b ? 1 : 0;     wr(w, &b1, 1); break;
        case GGUF_T_U16:  b2 = (uint16_t)kv->v.u64; wr(w, &b2, 2); break;
        case GGUF_T_I16:  b2 = (uint16_t)(int16_t)kv->v.i64; wr(w, &b2, 2); break;
        case GGUF_T_U32:  b4 = (uint32_t)kv->v.u64; wr(w, &b4, 4); break;
        case GGUF_T_I32:  b4 = (uint32_t)(int32_t)kv->v.i64; wr(w, &b4, 4); break;
        case GGUF_T_F32:  { uint32_t u = (uint32_t)kv->raw; wr(w, &u, 4); break; }
        case GGUF_T_U64:  b8 = kv->v.u64;           wr(w, &b8, 8); break;
        case GGUF_T_I64:  b8 = (uint64_t)kv->v.i64; wr(w, &b8, 8); break;
        case GGUF_T_F64:  wr(w, &kv->raw, 8); break;
    }
}

// Keys the output writes itself and must therefore NOT copy from its input.
//
// general.file_type is re-derived from the histogram of what was actually
// written (see the pre-pass below). The split.* trio has to go for a different
// reason: it describes a MULTI-PART set, and gguf_open merges every part
// before this writer ever runs, so what comes out is one whole file. Copying
// the trio declares an N-part model inside a single file, and gguf_open then
// refuses to open it at all -- "split GGUF filename does not follow
// <prefix>-00001-of-000NN.gguf" -- while the quantize that produced it exited
// 0 and printed its tensor counts. llama.cpp reads the same keys, so the
// artifact was unreadable everywhere, and a split checkpoint is how every
// model past a few tens of GB ships.
static bool kv_is_reauthored(const char *key) {
    static const char *const keys[] = {
        "general.file_type", "split.no", "split.count", "split.tensors.count",
    };
    for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); i++)
        if (!strcmp(key, keys[i])) return true;
    return false;
}

static void wr_kv(writer *w, const gguf_kv *kv) {
    wr_str(w, kv->key, strlen(kv->key));
    wr_u32(w, kv->type);
    if (kv->type == GGUF_T_STR) {
        wr_str(w, kv->str.s, kv->str.n);
    } else if (kv->type == GGUF_T_ARR) {
        wr_u32(w, kv->arr_type);
        wr_u64(w, kv->arr_n);
        if (kv->arr_type == GGUF_T_STR) {
            for (uint64_t i = 0; i < kv->arr_n; i++)
                wr_str(w, kv->arr_str[i].s, kv->arr_str[i].n);
        } else {
            uint64_t bytes;
            if (!checked_u64_mul(kv_scalar_size[kv->arr_type], kv->arr_n, &bytes) ||
                bytes > SIZE_MAX) w->ok = false;
            else wr(w, kv->arr_raw, (size_t)bytes);
        }
    } else {
        wr_scalar(w, kv, kv->type);
    }
}

// --------------------------------------------------------- sublayer removal
//
// --remove-sublayer attn:N[,mlp:M,...]: physically drop one block's attention
// or FFN tensors and declare the absence the way llama.cpp's per-layer
// arrays already can (its Nemotron-51B "deci" graph skips attention where
// attention.head_count[il] == 0 and the FFN where feed_forward_length[il]
// == 0). The runner loader reads the same zeros. Scope is the branch proper:
// every blk.N.attn_* tensor except the attn_norm* pre-norm, every
// blk.N.ffn_* tensor except the ffn_norm* pre-norm. The norms are kilobytes
// and keeping them leaves the residual plumbing identical to the zeroed
// form, which is what the removal is gated against.
enum { RM_ATTN = 0, RM_MLP = 1 };
typedef struct { int part, layer; } rm_item_t;
typedef struct { rm_item_t *items; int n; } rm_plan_t;

static bool rm_plan_parse(const char *spec, rm_plan_t *out) {
    out->items = NULL; out->n = 0;
    const char *p = spec;
    while (*p) {
        const char *q = strchr(p, ',');
        size_t len = q ? (size_t)(q - p) : strlen(p);
        int part;
        const char *num;
        if (len > 5 && strncmp(p, "attn:", 5) == 0) { part = RM_ATTN; num = p + 5; }
        else if (len > 4 && strncmp(p, "mlp:", 4) == 0) { part = RM_MLP; num = p + 4; }
        else {
            fprintf(stderr, "error: --remove-sublayer expects attn:N or mlp:N "
                    "entries, got '%.*s'\n", (int)len, p);
            free(out->items); out->items = NULL; out->n = 0;
            return false;
        }
        char *end;
        long v = strtol(num, &end, 10);
        if (end == num || end != p + len || v < 0 || v > INT_MAX ||
            !(*num >= '0' && *num <= '9')) {
            fprintf(stderr, "error: --remove-sublayer entry '%.*s' has no "
                    "valid block index\n", (int)len, p);
            free(out->items); out->items = NULL; out->n = 0;
            return false;
        }
        for (int i = 0; i < out->n; i++)
            if (out->items[i].part == part && out->items[i].layer == (int)v) {
                fprintf(stderr, "error: --remove-sublayer lists %.*s twice\n",
                        (int)len, p);
                free(out->items); out->items = NULL; out->n = 0;
                return false;
            }
        rm_item_t *grown = realloc(out->items, sizeof(rm_item_t) * (size_t)(out->n + 1));
        if (!grown) { free(out->items); out->items = NULL; out->n = 0; return false; }
        out->items = grown;
        out->items[out->n].part = part;
        out->items[out->n].layer = (int)v;
        out->n++;
        p = q ? q + 1 : p + len;
    }
    if (out->n == 0) {
        fprintf(stderr, "error: --remove-sublayer needs at least one entry\n");
        return false;
    }
    return true;
}

static bool rm_suffix_in_scope(int part, const char *suffix) {
    if (part == RM_ATTN)
        return strncmp(suffix, "attn_", 5) == 0 &&
               strncmp(suffix, "attn_norm", 9) != 0;
    return strncmp(suffix, "ffn_", 4) == 0 &&
           strncmp(suffix, "ffn_norm", 8) != 0;
}

// One re-authored u32-array header key: written after the copied keys,
// replacing any same-named key in the source.
typedef struct { char key[160]; uint32_t *vals; } reauth_arr_t;

// Fill `vals` (n_blocks entries) from the source key: an existing array
// entry by entry, a scalar everywhere, `dflt` when the key is absent.
static bool reauth_fill(gguf_file *g, const char *key, uint64_t n_blocks,
                        uint32_t dflt, uint32_t *vals) {
    gguf_kv *kv = gguf_get(g, key);
    if (kv && kv->type == GGUF_T_ARR && kv->arr_n != n_blocks) {
        fprintf(stderr, "error: %s has %llu entries for %llu blocks\n", key,
                (unsigned long long)kv->arr_n, (unsigned long long)n_blocks);
        return false;
    }
    for (uint64_t i = 0; i < n_blocks; i++)
        vals[i] = gguf_get_u32_idx(g, key, i, dflt);
    return true;
}

// ------------------------------------------------------------- expert prune

// --prune-experts LIST.json: {"layer_N": [kept expert ids...], ...}. A layer
// absent from the file keeps every expert. Stacked-layout pruning needs no
// per-expert tensor split: every per-expert tensor (router weight/bias, the
// three fused expert projections and their biases, gemma's fused gate+up and
// per-expert down scale) stores expert e as a contiguous block along its
// LAST dimension (see moe_expert_weight/model.c's check_shape3 comment) —
// pruning is just "keep these blocks, in this order, drop the rest."
typedef struct { int layer; int *ids; int n_ids; } prune_layer_t;
typedef struct { prune_layer_t *layers; int n; } prune_plan_t;

static void prune_plan_free(prune_plan_t *p) {
    if (!p) return;
    for (int i = 0; i < p->n; i++) free(p->layers[i].ids);
    free(p->layers);
    p->layers = NULL;
    p->n = 0;
}

// Returns false (plan left zeroed) on any parse/validation failure — a
// malformed plan must error out, not silently prune nothing or prune the
// wrong experts. Per-layer id bounds are re-checked later against each
// tensor's own declared expert count (not known until it's read).

// ------------------------------------------------------- selective precision
//
// A --type-plan is a per-TENSOR type override. Per-tensor is the finest
// granularity GGUF can express, and that limit is the whole story of the
// hot/cold expert experiment this was written for: MoE experts are stored
// STACKED, one 3-D tensor per layer holding every expert
// (blk.N.ffn_down_exps.weight is (768, 2048, 128) on Qwen3-30B-A3B), and a
// tensor carries exactly one type. Giving hot and cold experts different
// precisions would mean splitting that tensor into 128, which no loader
// expects and which would stop the file being a GGUF anyone can run. So
// per-expert precision is not a thing this format does, and the lever that IS
// available is per tensor class: keep attention and embeddings high, move the
// expert bulk down.
//
//   {"default": "keep",
//    "rules": [{"match": "_exps.weight", "type": "q4_0"}]}
//
// Rules are tried in order and the first substring match wins; "default"
// applies to everything unmatched. Types are the ones this quantizer can
// actually write (q8_0, q4_0, q3_k, q4_k, q6_k, f16, bf16, keep) — naming
// one it cannot is an error at load, not a silent fallback.
typedef struct { char *match; int type; } type_rule_t;
typedef struct { type_rule_t *rules; int n; int dflt; } type_plan_t;

static int type_from_name(const char *s) {
    if (!s) return -2;
    if (!strcmp(s, "keep")) return T_KEEP;
    if (!strcmp(s, "q8_0")) return T_Q8_0;
    if (!strcmp(s, "q4_0")) return T_Q4_0;
    if (!strcmp(s, "q3_k")) return T_Q3_K;
    if (!strcmp(s, "q4_k")) return T_Q4_K;
    if (!strcmp(s, "q6_k")) return T_Q6_K;
    if (!strcmp(s, "mxfp4")) return T_MXFP4;
    if (!strcmp(s, "f16"))  return T_F16;
    if (!strcmp(s, "bf16")) return T_BF16;
    return -2;
}

static void type_plan_free(type_plan_t *p) {
    for (int i = 0; i < p->n; i++) free(p->rules[i].match);
    free(p->rules);
    memset(p, 0, sizeof(*p));
}

static bool type_plan_load(const char *path, type_plan_t *plan) {
    memset(plan, 0, sizeof(*plan));
    plan->dflt = T_KEEP;
    FILE *f = plat_fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open --type-plan file %s\n", path); return false; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz < 0 || sz > 8 * 1024 * 1024 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "error: %s is not a readable type plan\n", path);
        return false;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return false; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    jv *root = json_parse(buf, rd);
    free(buf);
    if (!root || root->type != J_OBJ) {
        fprintf(stderr, "error: %s is not a JSON object\n", path);
        if (root) jv_free(root);
        return false;
    }
    bool ok = true;
    // The plan's CONTAINER is validated as strictly as its individual rules
    // already are. Every shape that used to be silently ignored -- a
    // misspelled key, a non-string "default", a "rules" that is not an array
    // -- loaded as a plan with zero rules and a keep-everything default: the
    // quantize then exited 0, printed "0 tensors converted", and wrote a
    // byte-identical copy of the input while the caller believed they had
    // built a selective-precision artifact. One character ("rule" for
    // "rules") was enough.
    for (int i = 0; ok && i < root->n; i++) {
        if (strcmp(root->keys[i], "default") == 0 ||
            strcmp(root->keys[i], "rules") == 0) continue;
        fprintf(stderr, "error: type-plan key \"%.40s\" is not one of "
                "\"default\" or \"rules\"\n", root->keys[i]);
        ok = false;
    }
    jv *d = jv_get(root, "default");
    bool has_default = false;
    if (ok && d) {
        if (d->type != J_STR) {
            fprintf(stderr, "error: type-plan \"default\" must be a type "
                    "name string\n");
            ok = false;
        } else {
            plan->dflt = type_from_name(jv_str(d, NULL));
            has_default = true;
            if (plan->dflt == -2) {
                fprintf(stderr, "error: type-plan default \"%s\" is not one of "
                        "keep|q8_0|q4_0|q3_k|q4_k|q6_k|mxfp4|f16|bf16\n", jv_str(d, "?"));
                ok = false;
            }
        }
    }
    jv *rules = jv_get(root, "rules");
    if (ok && rules && rules->type != J_ARR) {
        fprintf(stderr, "error: type-plan \"rules\" must be an array\n");
        ok = false;
    }
    if (ok && rules && rules->type == J_ARR && rules->n > 0) {
        plan->rules = calloc((size_t)rules->n, sizeof(type_rule_t));
        if (!plan->rules) { jv_free(root); return false; }
        for (int i = 0; ok && i < rules->n; i++) {
            jv *r = rules->items[i];
            jv *m = r && r->type == J_OBJ ? jv_get(r, "match") : NULL;
            jv *t = r && r->type == J_OBJ ? jv_get(r, "type") : NULL;
            if (!m || m->type != J_STR || !t || t->type != J_STR) {
                fprintf(stderr, "error: type-plan rule %d needs string "
                        "\"match\" and \"type\"\n", i);
                ok = false;
                break;
            }
            int ty = type_from_name(jv_str(t, NULL));
            if (ty == -2) {
                fprintf(stderr, "error: type-plan rule %d type \"%s\" is not "
                        "one of keep|q8_0|q4_0|q3_k|q4_k|q6_k|mxfp4|f16|bf16\n", i, jv_str(t, "?"));
                ok = false;
                break;
            }
            plan->rules[plan->n].match = strdup(jv_str(m, ""));
            if (!plan->rules[plan->n].match) { ok = false; break; }
            plan->rules[plan->n].type = ty;
            plan->n++;
        }
    }
    // A plan that names no rule and states no default selects nothing: it
    // would copy the model byte for byte and report success.
    if (ok && plan->n == 0 && !has_default) {
        fprintf(stderr, "error: %s selects nothing: it declares no \"rules\" "
                "and no \"default\", so it would copy the model unchanged\n",
                path);
        ok = false;
    }
    jv_free(root);
    if (!ok) type_plan_free(plan);
    return ok;
}

// first matching rule wins; T_KEEP means "leave this tensor exactly as it is"
static int type_plan_pick(const type_plan_t *plan, const char *name) {
    for (int i = 0; i < plan->n; i++)
        if (strstr(name, plan->rules[i].match)) return plan->rules[i].type;
    return plan->dflt;
}

static bool prune_plan_load(const char *path, prune_plan_t *plan) {
    memset(plan, 0, sizeof(*plan));
    FILE *f = plat_fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open --prune-experts file %s\n", path); return false; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    if (sz > 64 * 1024 * 1024) {
        fclose(f);
        fprintf(stderr, "error: %s is too large to be a prune plan\n", path);
        return false;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return false; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    jv *root = json_parse(buf, rd);
    free(buf);
    if (!root || root->type != J_OBJ) {
        fprintf(stderr, "error: %s is not a JSON object\n", path);
        if (root) jv_free(root);
        return false;
    }
    plan->layers = root->n > 0 ? calloc((size_t)root->n, sizeof(prune_layer_t)) : NULL;
    if (root->n > 0 && !plan->layers) { jv_free(root); return false; }
    bool ok = true;
    for (int i = 0; ok && i < root->n; i++) {
        long long parsed_layer;
        const char *key = root->keys[i];
        if (strncmp(key, "layer_", 6) != 0 ||
            key[6] < '0' || key[6] > '9' ||
            !parse_i64(key + 6, 0, INT_MAX, &parsed_layer)) {
            fprintf(stderr, "error: prune-plan key \"%s\" is not \"layer_N\"\n", root->keys[i]);
            ok = false;
            break;
        }
        int layer = (int)parsed_layer;
        jv *arr = root->items[i];
        if (arr->type != J_ARR || arr->n < 1) {
            fprintf(stderr, "error: prune-plan[\"layer_%d\"] must be a "
                    "non-empty array of expert ids\n", layer);
            ok = false;
            break;
        }
        int *ids = malloc(sizeof(int) * (size_t)arr->n);
        if (!ids) { ok = false; break; }
        for (int j = 0; j < arr->n; j++) {
            double id = arr->items[j]->type == J_NUM
                      ? arr->items[j]->num : -1.0;
            if (!isfinite(id) || id < 0 || id > INT_MAX || trunc(id) != id) {
                fprintf(stderr, "error: prune-plan[\"layer_%d\"][%d] is not "
                        "a non-negative integer\n", layer, j);
                free(ids);
                ok = false;
                break;
            }
            ids[j] = (int)id;
        }
        if (!ok) break;
        for (int a = 0; ok && a < arr->n; a++)
            for (int b = a + 1; b < arr->n; b++)
                if (ids[a] == ids[b]) {
                    fprintf(stderr, "error: prune-plan[\"layer_%d\"] lists "
                            "expert %d twice\n", layer, ids[a]);
                    ok = false;
                    break;
                }
        if (!ok) { free(ids); break; }
        plan->layers[plan->n].layer = layer;
        plan->layers[plan->n].ids = ids;
        plan->layers[plan->n].n_ids = arr->n;
        plan->n++;
    }
    jv_free(root);
    if (!ok) { prune_plan_free(plan); return false; }
    return true;
}

static const prune_layer_t *prune_plan_find(const prune_plan_t *plan, int layer) {
    if (!plan) return NULL;
    for (int i = 0; i < plan->n; i++)
        if (plan->layers[i].layer == layer) return &plan->layers[i];
    return NULL;
}

// Every per-expert tensor role this quantizer knows how to prune, matched by
// name (not shape — a coincidental last-dim size is not license to slice a
// tensor this tool doesn't understand). ffn_gate_inp.scale is deliberately
// absent: it is gemma-4's per-EMBEDDING-CHANNEL router input scale
// ([n_embd]), not per-expert, and must never be sliced.
static bool is_expert_tensor_suffix(const char *suffix) {
    static const char *names[] = {
        "ffn_gate_inp.weight", "ffn_gate_inp.bias",
        "ffn_gate_exps.weight", "ffn_gate_exps.bias",
        "ffn_up_exps.weight", "ffn_up_exps.bias",
        "ffn_down_exps.weight", "ffn_down_exps.bias",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.scale",
        // Both on-disk spellings of the selection bias: model.c reads
        // `exp_probs_b.weight` OR `exp_probs_b.bias` with the layer's expert
        // count, and nemotron_h_moe ships the `.bias` one. Slicing only
        // `.weight` left a full-length bias beside a pruned router, which the
        // loader then (correctly) refuses.
        "exp_probs_b.weight", "exp_probs_b.bias",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
        if (!strcmp(suffix, names[i])) return true;
    return false;
}

// "blk.N.suffix" -> layer=N, *suffix -> "suffix". False for anything not
// per-layer (token embedding, output head, ...).
static bool parse_blk_name(const char *name, int *layer, const char **suffix) {
    if (strncmp(name, "blk.", 4) != 0) return false;
    char *end;
    long v = strtol(name + 4, &end, 10);
    if (end == name + 4 || *end != '.' || v < 0 || v > INT_MAX) return false;
    *layer = (int)v;
    *suffix = end + 1;
    return true;
}

// Per-expert tensors store expert e as one contiguous block of `rows` rows
// (ggml_row_size(type, row_len) bytes each) at block index e along their
// LAST dimension — uniformly across the 1D (bias/scale: row_len=1, rows=1),
// 2D (router/exps-bias: row_len=ne[0], rows=1), and 3D (fused expert
// weights: row_len=ne[0], rows=ne[1]) shapes this tool prunes.
static void expert_block_geometry(const gguf_tensor *t, int64_t *row_len, int64_t *rows) {
    // The expert index is the OUTERMOST dimension; everything between it and
    // ne[0] belongs to one expert's block. Taking only ne[1] under-wrote a
    // 4-D per-expert tensor to a zero-padded stub while reporting success --
    // no shipping model has that shape, but a crafted file must produce a
    // refusal or a correct file, never a silently short tensor. The 3-D and
    // 2-D shapes are unchanged: the product degenerates to ne[1] and 1.
    *row_len = t->n_dims >= 2 ? t->ne[0] : 1;
    *rows = 1;
    for (uint32_t d = 1; d + 1 < t->n_dims; d++) *rows *= t->ne[d];
}

// ---------------------------------------------------------------- lora merge
// --merge-lora: fold a trained adapter into the base weights, producing a
// standalone GGUF (W' = W + (alpha/r)*B·A per adapted projection). The
// merged file runs anywhere a GGUF runs — but merging into a quantized
// type ROUNDS THE DELTA through that type's grid, so base + --lora stays
// the exact form; a merged artifact's fidelity is a measurement, not a
// given. Provenance (.merge.json, written by the caller) carries the
// base/adapter shas that make the merged blob auditable.

typedef struct {
    float *a, *b;         // owned f32 copies of the pair: a is r rows of
                          // n_in, b is n_out rows of r — the orientation
                          // lora_apply consumes, so the merged weight
                          // computes W'x = Wx + scale*B(Ax) up to
                          // summation order. Copied (not mmap-aliased)
                          // because F16/BF16 adapters convert at resolve
    int r;                // 0 = this base tensor is untouched
    float scale;          // (alpha / r) * user scale
    bool zero;            // delta is exactly zero (b all-zero or scale 0):
                          // copy the base tensor verbatim instead of
                          // churning it through dequant->requant
} merge_pair_t;

typedef struct {
    gguf_file g;          // adapter file; stays open while rows stream
    bool open;
    merge_pair_t *map;    // one entry per BASE tensor index
    uint64_t n_base;      // base tensor count the map was resolved against
    int n_pairs, n_zero;
    // What the output grid kept of the delta, over the adapted tensors:
    // each row is written twice, base + delta (the merged row) and the base
    // alone at the same output type, and the two are compared.
    merge_survival surv;
    float min_retained;   // refuse below this delta_retained; < 0 = never
} merge_set_t;

// mirror of model.c lora_slot_name — the projections --lora hooks. The
// merge accepts exactly what serving accepts, so a mergeable adapter and a
// servable adapter are the same set.
static const char *const merge_slot_name[] = {
    "attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
};
// the qwen35 DeltaNet projections, hooked only on that backbone (qwen35, or
// clef on it): afmoe and Qwen-G1 have an attn_gate of their own that
// serving does not hook, so the merge must not accept the name there
static const char *const merge_slot_name_q35[] = {
    "attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out",
};

static void merge_set_free(merge_set_t *ms) {
    if (ms->open) gguf_close(&ms->g);
    for (uint64_t i = 0; ms->map && i < ms->n_base; i++) {
        free(ms->map[i].a);
        free(ms->map[i].b);
    }
    free(ms->map);
    memset(ms, 0, sizeof(*ms));
}

// model_lora_load's validation, run against the base FILE instead of a
// loaded model: every adapter tensor must name a hooked projection that
// exists in the base with matching geometry, or the whole merge refuses —
// a silently skipped tensor would emit a merged model that is not
// base+adapter.
static bool merge_set_resolve(merge_set_t *ms, const char *adapter_path,
                              float user_scale, gguf_file *base) {
    memset(ms, 0, sizeof(*ms));
    if (!gguf_open(&ms->g, adapter_path)) {
        fprintf(stderr, "error: cannot open adapter %s\n", adapter_path);
        return false;
    }
    ms->open = true;
    const char *ftype = gguf_get_str(&ms->g, "general.type", "");
    const char *atype = gguf_get_str(&ms->g, "adapter.type", "");
    if (strcmp(ftype, "adapter") != 0 || strcmp(atype, "lora") != 0) {
        fprintf(stderr, "error: %s is not a LoRA adapter GGUF "
                "(general.type=%s adapter.type=%s)\n", adapter_path, ftype,
                atype);
        return false;
    }
    const char *aarch = gguf_get_str(&ms->g, "general.architecture", "");
    const char *barch = gguf_get_str(base, "general.architecture", "");
    if (!model_lora_arch_ok(barch, aarch)) {
        fprintf(stderr, "error: adapter architecture '%s' does not match "
                "the model's '%s'\n", aarch, barch);
        return false;
    }
    float alpha = gguf_get_f32(&ms->g, "adapter.lora.alpha", 0.0f);
    if (!(alpha > 0.0f)) {
        fprintf(stderr, "error: adapter.lora.alpha missing or not "
                "positive\n");
        return false;
    }
    ms->n_base = base->n_tensors;
    ms->map = calloc(base->n_tensors ? base->n_tensors : 1,
                     sizeof(merge_pair_t));
    if (!ms->map) return false;
    for (uint64_t ti = 0; ti < ms->g.n_tensors; ti++) {
        const gguf_tensor *t = &ms->g.tensors[ti];
        int l = -1;
        char pname[64];
        char side = 0;
        // %n after the last field: sscanf stops at `side` and never reports
        // that anything followed it, so "...weight.lora_a2" parsed as side
        // 'a' and returned 3. Two adapter tensors then mapped to the same
        // (projection, side) slot and the second silently overwrote the
        // first, emitting a merged model that is not base+adapter -- the one
        // outcome this loader exists to refuse.
        int consumed = -1;
        if (sscanf(t->name, "blk.%d.%40[a-z_].weight.lora_%c%n", &l, pname,
                   &side, &consumed) != 3 || consumed < 0 ||
            t->name[consumed] != 0 || (side != 'a' && side != 'b') || l < 0) {
            fprintf(stderr, "error: adapter tensor '%s' does not name a "
                    "hooked projection\n", t->name);
            return false;
        }
        bool hooked = false;
        for (size_t k = 0; k < sizeof(merge_slot_name) /
                               sizeof(*merge_slot_name); k++)
            if (strcmp(pname, merge_slot_name[k]) == 0) { hooked = true; break; }
        bool q35 = model_lora_arch_ok("qwen35", barch);   // qwen35 itself, or clef
        if (!q35) q35 = strcmp(barch, "clef") == 0;
        for (size_t k = 0; q35 && !hooked && k < sizeof(merge_slot_name_q35) /
                                                  sizeof(*merge_slot_name_q35); k++)
            if (strcmp(pname, merge_slot_name_q35[k]) == 0) hooked = true;
        char bname[96];
        snprintf(bname, sizeof(bname), "blk.%d.%s.weight", l, pname);
        const gguf_tensor *bt = NULL;
        uint64_t bi = 0;
        for (uint64_t j = 0; hooked && j < base->n_tensors; j++)
            if (strcmp(base->tensors[j].name, bname) == 0) {
                bt = &base->tensors[j];
                bi = j;
                break;
            }
        if (!bt || bt->n_dims != 2) {
            fprintf(stderr, "error: adapter tensor '%s' targets a "
                    "projection this model does not carry\n", t->name);
            return false;
        }
        // F32 from our writer, F16 from llama.cpp's converter (the way
        // community adapters ship), BF16 for completeness — anything else
        // refuses by name
        if (t->type != T_F32 && t->type != T_F16 && t->type != T_BF16) {
            fprintf(stderr, "error: adapter tensor '%s' is %s — adapters "
                    "must be F32, F16 or BF16\n", t->name,
                    ggml_type_name(t->type));
            return false;
        }
        uint64_t n_in = bt->ne[0], n_out = bt->ne[1];
        uint64_t r = side == 'a' ? t->ne[1] : t->ne[0];
        uint64_t want0 = side == 'a' ? n_in : r;
        uint64_t want1 = side == 'a' ? r : n_out;
        // 512 = LORA_R_MAX, model.c — the merge accepts what --lora serves
        if (t->n_dims != 2 || r < 1 || r > 512 || t->ne[0] != want0 ||
            t->ne[1] != want1) {
            fprintf(stderr, "error: adapter tensor '%s' is %u-D "
                    "[%llu,%llu]; base '%s' is [%llu,%llu]\n", t->name,
                    t->n_dims, (unsigned long long)t->ne[0],
                    (unsigned long long)t->ne[1], bt->name,
                    (unsigned long long)n_in, (unsigned long long)n_out);
            return false;
        }
        merge_pair_t *mp = &ms->map[bi];
        if (mp->r && mp->r != (int)r) {
            fprintf(stderr, "error: adapter '%s' rank %llu disagrees with "
                    "its pair (rank %d)\n", t->name,
                    (unsigned long long)r, mp->r);
            return false;
        }
        mp->r = (int)r;
        uint64_t n_el;
        if (!checked_u64_mul(t->ne[0], t->ne[1], &n_el) ||
            n_el > SIZE_MAX / sizeof(float)) return false;
        float *dat = malloc(sizeof(float) * (size_t)n_el);
        if (!dat) return false;
        size_t row_size = ggml_row_size(t->type, (int64_t)t->ne[0]);
        if (row_size == 0 || t->ne[1] > t->nbytes / row_size) {
            free(dat);
            return false;
        }
        for (uint64_t row = 0; row < t->ne[1]; row++)
            dequant_row(t->type,
                        (const uint8_t *)t->data + row * row_size,
                        dat + row * t->ne[0], (int)t->ne[0]);
        if (side == 'a' ? mp->a != NULL : mp->b != NULL) {
            fprintf(stderr, "error: adapter declares lora_%c for %s twice\n",
                    side, bname);
            free(dat);
            return false;
        }
        if (side == 'a') mp->a = dat;
        else             mp->b = dat;
    }
    for (uint64_t i = 0; i < base->n_tensors; i++) {
        merge_pair_t *mp = &ms->map[i];
        if (!mp->r) continue;
        if (!mp->a || !mp->b) {
            fprintf(stderr, "error: adapter has only half of the "
                    "lora_a/lora_b pair for %s\n", base->tensors[i].name);
            return false;
        }
        mp->scale = alpha / (float)mp->r * user_scale;
        int64_t n_out = base->tensors[i].ne[1];
        mp->zero = mp->scale == 0.0f;
        if (!mp->zero) {
            mp->zero = true;
            const uint32_t *bw = (const uint32_t *)mp->b;
            for (int64_t c = 0; c < (int64_t)mp->r * n_out; c++)
                if (bw[c] != 0) { mp->zero = false; break; }
        }
        ms->n_pairs++;
        if (mp->zero) ms->n_zero++;
    }
    if (!ms->n_pairs) {
        fprintf(stderr, "error: adapter carries no lora_a/lora_b pairs\n");
        return false;
    }
    return true;
}

// Fold the low-rank delta into one dequantized row: as a weight update,
// row j of scale*B(Ax) is rowf[i] += scale * sum_k B[j][k]*A[k][i].
// Fixed-order fmaf chain — the merged bytes must be reproducible across
// runs and hosts, the same discipline as lora_apply.
static void merge_row(const merge_pair_t *mp, uint64_t j, float *rowf,
                      int n_in) {
    const float *br = mp->b + (size_t)j * mp->r;
    for (int i = 0; i < n_in; i++) {
        float acc = 0.0f;
        for (int k = 0; k < mp->r; k++)
            acc = fmaf(br[k], mp->a[(size_t)k * n_in + i], acc);
        rowf[i] = fmaf(mp->scale, acc, rowf[i]);
    }
}

// ---------------------------------------------------------------- main

static bool should_quantize(const gguf_tensor *t) {
    if (t->n_dims < 2) return false;                 // norms, biases
    if (t->ne[0] % 32 != 0) return false;
    size_t nl = strlen(t->name);
    if (nl < 7 || strcmp(t->name + nl - 7, ".weight") != 0) return false;
    if (strstr(t->name, "_norm.")) return false;
    if (strstr(t->name, "rope_freqs")) return false;
    return true;
}

// Tensors that leave a requant with their on-disk type untouched, whatever
// the target or the plan says. The MoE router picks WHICH expert runs: an
// error there swaps a whole FFN instead of perturbing one output, and it is
// the one weight matrix whose cost of staying exact is negligible
// (n_embd x n_expert — 0.3% of a 30B-A3B checkpoint). That is why llama.cpp
// excludes ffn_gate_inp from quantization and why every reference MoE GGUF
// ships it at F32 beside 4-bit experts. `ffn_gate_inp_shexp` is the same
// decision for the shared expert, so the substring covers both.
static bool keep_source_type(const gguf_tensor *t) {
    return strstr(t->name, "ffn_gate_inp") != NULL;
}

// A quantized row must divide into a whole number of blocks. should_quantize
// only clears ne[0] % 32, which is the block width of q8_0/q4_0 but not of the
// 256-wide K-quants a --type-plan can name: writing a 288-wide row as Q3_K
// emits one block (256 values, 32 silently dropped) under a header that still
// says 288, and gguf.c refuses ne[0] % block_size != 0 at open — so the requant
// would report success and produce a file nothing can load.
static bool type_fits_row(int type, int64_t n) {
    return n % ggml_block_size(type) == 0;
}

// A 256-wide K-quant or i-quant cannot describe a row that only divides by
// 32. Leaving such a tensor at its source type made a "Q4_K_M" of a model
// with 5760-wide rows keep 62% of its bytes at BF16 (Kvist-14B, the lab,
// 2026-09-23), where llama.cpp writes the 32-block type of the nearest bit
// budget instead. Same map here, reported per tensor: the file that comes
// out is what its label says within one bit per weight, not a bf16 file
// with a quantized name.
static int width_fallback(int want, int64_t n) {
    if (type_fits_row(want, n)) return want;
    int fb;
    switch (want) {
    case T_Q2_K: case T_Q3_K:
    case T_IQ1_S: case T_IQ1_M: case T_IQ2_XXS: case T_IQ2_XS: case T_IQ2_S:
    case T_IQ3_XXS: case T_IQ3_S:
        fb = T_Q4_0; break;
    case T_Q4_K: case T_IQ4_XS: case T_IQ4_NL:
        fb = T_Q5_0; break;
    case T_Q5_K:
        fb = T_Q5_1; break;
    case T_Q6_K:
        fb = T_Q8_0; break;
    default:
        return want;                       // no 32-block cousin: decline as before
    }
    return type_fits_row(fb, n) ? fb : want;
}

static bool quantize_install_injected(void) {
    const char *inject = getenv("RUNNER_QUANTIZE_INSTALL_FAIL");
    return inject && *inject && strcmp(inject, "0");
}

static bool g_type_plan_strict = false;
void quantize_set_type_plan_strict(bool on) { g_type_plan_strict = on; }

static int install_finished_file(const char *tmp_path, const char *out_path) {
    if (quantize_install_injected()) return -1;
    return plat_replace_file(tmp_path, out_path) ? 0 : -1;
}

// If tensor `t` is a per-expert tensor for a layer the (already fully
// validated — see the pre-pass in quantize_gguf) prune plan covers, returns
// that layer's kept-id list; else NULL ("not pruned, write unchanged").
static const prune_layer_t *resolve_prune(const gguf_tensor *t, const prune_plan_t *plan) {
    if (!plan || plan->n == 0) return NULL;
    int layer; const char *suffix;
    if (!parse_blk_name(t->name, &layer, &suffix)) return NULL;
    if (!is_expert_tensor_suffix(suffix)) return NULL;
    return prune_plan_find(plan, layer);
}

static void quantize_plans_free(prune_plan_t *prune, type_plan_t *types) {
    prune_plan_free(prune);
    type_plan_free(types);
}

int quantize_gguf(const char *in_path, const char *out_path, int target,
                  const char *prune_path) {
    return quantize_gguf_plan(in_path, out_path, target, prune_path, NULL, NULL);
}

int quantize_type_from_name(const char *s) {
    return type_from_name(s);
}

static int quantize_gguf_plan_inner(const char *in_path, const char *out_path, int target,
                       const char *prune_path, const char *type_plan_path,
                       const char *remove_spec,
                       merge_set_t *ms, uint32_t surgery_context,
                       float surgery_factor, context_surgery_result *surgery_result);

// FTZ/DAZ are process-wide (set by the engine's -ffast-math startup); clear
// them around the quantize so a subnormal block scale matches ggml, and
// restore on exit so no later caller sees perturbed denormal handling.
int quantize_gguf_plan(const char *in_path, const char *out_path, int target,
                       const char *prune_path, const char *type_plan_path,
                       const char *remove_spec) {
    fp_denormal_state fpst = fp_denormals_disable();
    int rc = quantize_gguf_plan_inner(in_path, out_path, target, prune_path,
                                      type_plan_path, remove_spec, NULL, 0, 0, NULL);
    fp_denormals_restore(fpst);
    return rc;
}

static bool tensor_payloads_equal(const char *a_path, const char *b_path) {
    gguf_file a, b;
    if (!gguf_open(&a, a_path)) return false;
    if (!gguf_open(&b, b_path)) { gguf_close(&a); return false; }
    bool same = a.n_tensors == b.n_tensors;
    for (uint64_t i = 0; same && i < a.n_tensors; i++) {
        const gguf_tensor *x = &a.tensors[i], *y = &b.tensors[i];
        same = strcmp(x->name, y->name) == 0 && x->type == y->type &&
               x->n_dims == y->n_dims && x->nbytes == y->nbytes &&
               x->nbytes <= SIZE_MAX;
        for (uint32_t d = 0; same && d < x->n_dims; d++)
            same = x->ne[d] == y->ne[d];
        if (same && x->nbytes)
            same = memcmp(x->data, y->data, (size_t)x->nbytes) == 0;
    }
    gguf_close(&b);
    gguf_close(&a);
    return same;
}

int context_surgery_gguf(const char *in_path, const char *out_path,
                         uint32_t target_context, float target_factor,
                         context_surgery_result *result) {
    context_surgery_result local = {0};
    if (!result) result = &local;
    memset(result, 0, sizeof(*result));
    fp_denormal_state fpst = fp_denormals_disable();
    int rc = quantize_gguf_plan_inner(in_path, out_path, T_KEEP, NULL, NULL, NULL,
                                      NULL, target_context, target_factor,
                                      result);
    fp_denormals_restore(fpst);
    if (rc != 0) return rc;
    if (!tensor_payloads_equal(in_path, out_path)) {
        fprintf(stderr, "error: context-surgery tensor-payload verification "
                        "failed; refusing %s\n", out_path);
        plat_remove(out_path);
        return 1;
    }
    result->tensors_byte_identical = true;
    fprintf(stderr, "context-surgery: independently reparsed tensor payloads "
                    "are byte-identical\n");
    return 0;
}

// The adapter is resolved against a first open of the base, which then
// closes; the inner writer re-opens the same path itself. Tensor indices
// are stable across the two opens of one file, and the adapter's mmap (the
// a/b pointers in the map) stays alive in `ms` for the whole write.
int merge_lora_gguf(const char *in_path, const char *adapter_path,
                    float user_scale, const char *out_path, int target,
                    float min_retained, merge_survival *surv) {
    gguf_file base;
    if (!gguf_open(&base, in_path)) return 1;
    merge_set_t ms;
    bool ok = merge_set_resolve(&ms, adapter_path, user_scale, &base);
    gguf_close(&base);
    if (!ok) {
        merge_set_free(&ms);
        return 1;
    }
    ms.min_retained = min_retained;
    fp_denormal_state fpst = fp_denormals_disable();
    int rc = quantize_gguf_plan_inner(in_path, out_path, target, NULL, NULL, NULL,
                                      &ms, 0, 0, NULL);
    fp_denormals_restore(fpst);
    if (surv) *surv = ms.surv;
    merge_set_free(&ms);
    return rc;
}

// Everything one file transformation owns. Each stage of
// quantize_gguf_plan_inner() adds to it and every exit releases all of it in
// one place, so a refusal at any stage unwinds every stage before it.
typedef struct {
    prune_plan_t plan;
    type_plan_t tplan;
    rm_plan_t rm;
    gguf_file g;
    bool g_open;
    bool *drop;               // tensors the removal does not write
    reauth_arr_t reauth[4];   // per-layer arrays the removal re-authors
    int n_reauth;
    uint64_t n_dropped;       // tensors the removal drops
    // Per-layer expert counts after the prune, authored into the header as
    // <arch>.expert_count_per_layer (u32 per block, 0 for a non-MoE block).
    // GGUF has one global expert_count; a non-uniform plan makes that key a
    // ceiling rather than a description, and a consumer that trusts it
    // mis-sizes the model. The array is what lets the file describe itself.
    uint32_t *layer_counts;
    uint64_t n_blocks;        // the length of layer_counts
    char pl_key[128];         // <arch>.expert_count_per_layer
    bool uniform_prune;       // every MoE layer pruned to one count
    int uniform_count;
    char *tmp_path;           // OUT.partial
    bool tmp_created;         // OUT.partial exists and is not installed
    uint64_t amask;           // the data alignment, less one
    uint64_t data_start;      // where the tensor data begins
    writer w;
    int *out_type;
    uint64_t *out_off;
    int64_t (*eff_ne)[4];
    float *rowf, *basef, *deqm, *deqb;
    uint8_t *rowq, *rowqb;
    uint64_t quantized, kept; // tensors converted, tensors copied
} quantize_job;

// Everything that reads the source, released before the output is installed.
static void quantize_job_close_source(quantize_job *J) {
    free(J->basef); free(J->deqm); free(J->deqb); free(J->rowqb);
    free(J->rowf); free(J->rowq);
    free(J->out_type); free(J->out_off); free(J->eff_ne);
    J->basef = J->deqm = J->deqb = J->rowf = NULL;
    J->rowqb = J->rowq = NULL;
    J->out_type = NULL; J->out_off = NULL; J->eff_ne = NULL;
    for (int k = 0; k < J->n_reauth; k++) free(J->reauth[k].vals);
    J->n_reauth = 0;
    free(J->layer_counts); J->layer_counts = NULL;
    free(J->drop); J->drop = NULL;
    free(J->rm.items); J->rm.items = NULL; J->rm.n = 0;
    if (J->g_open) gguf_close(&J->g);
    J->g_open = false;
    quantize_plans_free(&J->plan, &J->tplan);
}

// Everything, the partial file included unless it was installed.
static void quantize_job_release(quantize_job *J) {
    quantize_job_close_source(J);
    if (J->w.f) fclose(J->w.f);
    J->w.f = NULL;
    if (J->tmp_created) plat_remove(J->tmp_path);
    J->tmp_created = false;
    free(J->tmp_path); J->tmp_path = NULL;
}

// The plans parsed and the source open, every tensor in a type this file
// can read.
static bool quantize_job_open_source(quantize_job *J, const char *in_path,
                                     const char *prune_path,
                                     const char *type_plan_path,
                                     const char *remove_spec) {
    if (prune_path && !prune_plan_load(prune_path, &J->plan)) return false;
    if (type_plan_path && !type_plan_load(type_plan_path, &J->tplan)) return false;
    if (remove_spec && !rm_plan_parse(remove_spec, &J->rm)) return false;

    if (!gguf_open(&J->g, in_path)) return false;
    J->g_open = true;
    for (uint64_t i = 0; i < J->g.n_tensors; i++) {
        if (!ggml_type_supported(J->g.tensors[i].type)) {
            fprintf(stderr, "error: tensor %s has unsupported type %s\n",
                    J->g.tensors[i].name, ggml_type_name(J->g.tensors[i].type));
            return false;
        }
    }
    return true;
}

// Sublayer-removal pre-pass: decide every dropped tensor and every
// re-authored per-layer array before a byte of output exists, so a bad
// request fails clean. `drop[i]` marks a tensor that is not written.
static bool quantize_job_remove_sublayers(quantize_job *J) {
    const char *rarch = gguf_get_str(&J->g, "general.architecture", "");
    char key[160];
    snprintf(key, sizeof key, "%s.block_count", rarch);
    uint64_t nb = gguf_get_u32(&J->g, key, 0);
    const char *why = NULL;
    if (!strcmp(rarch, "nemotron_h") || !strcmp(rarch, "nemotron_h_moe") ||
        !strcmp(rarch, "granitehybrid") || !strcmp(rarch, "qwen35") ||
        !strcmp(rarch, "lfm2") || !strcmp(rarch, "lfm2moe") ||
        (snprintf(key, sizeof key, "%s.ssm.conv_kernel", rarch),
         gguf_get(&J->g, key) != NULL))
        why = "hybrid or recurrent block typing";
    // gemma-4 exports carry both E-series keys on every model, with 0
    // on the dense ones (the 31B has embedding_length_per_layer_input
    // = 0); it is the VALUE that makes a file E-series, as in the loader.
    else if ((snprintf(key, sizeof key, "%s.embedding_length_per_layer_input",
                       rarch), gguf_get_u32(&J->g, key, 0) > 0) ||
             (snprintf(key, sizeof key, "%s.attention.shared_kv_layers",
                       rarch), gguf_get_u32(&J->g, key, 0) > 0))
        why = "gemma-4 E-series per-layer embeddings or shared KV";
    else if ((snprintf(key, sizeof key, "%s.nextn_predict_layers", rarch),
              gguf_get_u32(&J->g, key, 0) > 0))
        why = "a declared NextN/MTP head";
    else if (nb == 0)
        why = "no block_count";
    if (why) {
        fprintf(stderr, "error: --remove-sublayer does not support %s (%s)\n",
                rarch, why);
        return false;
    }
    J->drop = calloc(J->g.n_tensors ? J->g.n_tensors : 1, sizeof(bool));
    if (!J->drop) return false;
    char hc_key[160], hk_key[160], ff_key[160];
    snprintf(hc_key, sizeof hc_key, "%s.attention.head_count", rarch);
    snprintf(hk_key, sizeof hk_key, "%s.attention.head_count_kv", rarch);
    snprintf(ff_key, sizeof ff_key, "%s.feed_forward_length", rarch);
    uint32_t *hc = NULL, *hk = NULL, *ff = NULL;
    bool bad = false;
    for (int k = 0; k < J->rm.n && !bad; k++) {
        int layer = J->rm.items[k].layer, part = J->rm.items[k].part;
        const char *pname = part == RM_ATTN ? "attn" : "mlp";
        if ((uint64_t)layer >= nb) {
            fprintf(stderr, "error: --remove-sublayer %s:%d names block %d, "
                    "but %s.block_count is %llu\n", pname, layer, layer,
                    rarch, (unsigned long long)nb);
            bad = true; break;
        }
        char nm[160];
        if (part == RM_ATTN) {
            if (!hc) {
                hc = malloc(sizeof(uint32_t) * nb);
                hk = malloc(sizeof(uint32_t) * nb);
                uint32_t nh = gguf_get_u32(&J->g, hc_key, 0);
                if (!hc || !hk || !reauth_fill(&J->g, hc_key, nb, nh, hc) ||
                    !reauth_fill(&J->g, hk_key, nb, nh, hk)) { bad = true; break; }
            }
            if (hc[layer] == 0) {
                fprintf(stderr, "error: --remove-sublayer attn:%d: block %d "
                        "already has its attention removed\n", layer, layer);
                bad = true; break;
            }
            snprintf(nm, sizeof nm, "blk.%d.attn_qkv.weight", layer);
            if (gguf_find_tensor(&J->g, nm)) {
                fprintf(stderr, "error: --remove-sublayer attn:%d: fused-QKV "
                        "families are not supported\n", layer);
                bad = true; break;
            }
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", layer);
            if (!gguf_find_tensor(&J->g, nm)) {
                fprintf(stderr, "error: --remove-sublayer attn:%d: block %d "
                        "has no attention tensors\n", layer, layer);
                bad = true; break;
            }
            hc[layer] = 0;
            hk[layer] = 0;
        } else {
            if (!ff) {
                ff = malloc(sizeof(uint32_t) * nb);
                uint32_t nf = gguf_get_u32(&J->g, ff_key, 0);
                if (!ff || !reauth_fill(&J->g, ff_key, nb, nf, ff)) { bad = true; break; }
            }
            if (ff[layer] == 0) {
                fprintf(stderr, "error: --remove-sublayer mlp:%d: block %d "
                        "already has its FFN removed\n", layer, layer);
                bad = true; break;
            }
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", layer);
            bool moe = gguf_find_tensor(&J->g, nm) != NULL;
            for (uint64_t i = 0; !moe && i < J->g.n_tensors; i++) {
                int tl; const char *suf;
                if (parse_blk_name(J->g.tensors[i].name, &tl, &suf) &&
                    tl == layer && strncmp(suf, "ffn_", 4) == 0 &&
                    strstr(suf, "_exps"))
                    moe = true;
            }
            if (moe) {
                fprintf(stderr, "error: --remove-sublayer mlp:%d: block %d is "
                        "a MoE FFN, which is not supported for removal\n",
                        layer, layer);
                bad = true; break;
            }
            snprintf(nm, sizeof nm, "blk.%d.ffn_down.weight", layer);
            if (!gguf_find_tensor(&J->g, nm)) {
                fprintf(stderr, "error: --remove-sublayer mlp:%d: block %d "
                        "has no dense FFN tensors\n", layer, layer);
                bad = true; break;
            }
            ff[layer] = 0;
        }
        uint64_t n_t = 0, bytes = 0;
        for (uint64_t i = 0; i < J->g.n_tensors; i++) {
            int tl; const char *suf;
            if (!parse_blk_name(J->g.tensors[i].name, &tl, &suf) || tl != layer ||
                !rm_suffix_in_scope(part, suf) || J->drop[i])
                continue;
            J->drop[i] = true;
            n_t++;
            bytes += J->g.tensors[i].nbytes;
        }
        J->n_dropped += n_t;
        fprintf(stderr, "remove-sublayer: %s:%d dropped %llu tensors, %llu "
                "bytes of tensor data\n", pname, layer,
                (unsigned long long)n_t, (unsigned long long)bytes);
    }
    // A file with no attention (or no FFN) left in ANY block is not a
    // model the loader can run: it reads the head count (or FFN width)
    // as the largest per-block entry, finds 0 and refuses the file. Say
    // so here, before writing an artifact that cannot load. Counted
    // over the whole array, so blocks the source already lacked count.
    if (!bad) {
        uint64_t attn_left = 0, ffn_left = 0;
        for (uint64_t i = 0; i < nb; i++) {
            if (hc && hc[i]) attn_left++;
            if (ff && ff[i]) ffn_left++;
        }
        const char *none = hc && !attn_left ? "attention" :
                           ff && !ffn_left  ? "FFN" : NULL;
        if (none) {
            fprintf(stderr, "error: --remove-sublayer would leave no %s in "
                    "any of the %llu blocks; the loader cannot run such a "
                    "file, so it is not written\n", none,
                    (unsigned long long)nb);
            bad = true;
        }
    }
    if (bad) {
        free(hc); free(hk); free(ff);
        return false;
    }
    if (hc) {
        snprintf(J->reauth[J->n_reauth].key, sizeof J->reauth[J->n_reauth].key, "%s", hc_key);
        J->reauth[J->n_reauth++].vals = hc;
        snprintf(J->reauth[J->n_reauth].key, sizeof J->reauth[J->n_reauth].key, "%s", hk_key);
        J->reauth[J->n_reauth++].vals = hk;
    }
    if (ff) {
        snprintf(J->reauth[J->n_reauth].key, sizeof J->reauth[J->n_reauth].key, "%s", ff_key);
        J->reauth[J->n_reauth++].vals = ff;
    }
    return true;
}

// --context-surgery: the header's YaRN context and factor rewritten in
// memory, after checking that the file declares native YaRN and that the
// request only extends it and agrees with the original context.
static bool quantize_job_context_surgery(quantize_job *J, uint32_t surgery_context,
                                         float surgery_factor,
                                         context_surgery_result *surgery_result) {
    const char *arch = gguf_get_str(&J->g, "general.architecture", "");
    char ctx_key[128], type_key[128], factor_key[128], orig_key[128];
    snprintf(ctx_key, sizeof(ctx_key), "%s.context_length", arch);
    snprintf(type_key, sizeof(type_key), "%s.rope.scaling.type", arch);
    snprintf(factor_key, sizeof(factor_key), "%s.rope.scaling.factor", arch);
    snprintf(orig_key, sizeof(orig_key),
             "%s.rope.scaling.original_context_length", arch);
    gguf_kv *ctx = gguf_get(&J->g, ctx_key);
    gguf_kv *fac = gguf_get(&J->g, factor_key);
    uint32_t source_context = gguf_get_u32(&J->g, ctx_key, 0);
    uint32_t original_context = gguf_get_u32(&J->g, orig_key, 0);
    float source_factor = gguf_get_f32(&J->g, factor_key, 0);
    const char *scaling_type = gguf_get_str(&J->g, type_key, "");
    double implied = (double)original_context * (double)surgery_factor;
    uint64_t implied_context = implied >= 0.0 ? (uint64_t)(implied + 0.5) : 0;
    bool valid = arch[0] && ctx && ctx->type == GGUF_T_U32 &&
                 fac && fac->type == GGUF_T_F32 &&
                 strcmp(scaling_type, "yarn") == 0 &&
                 original_context > 0 && source_context > 0 &&
                 source_factor > 1.0f;
    if (!valid) {
        fprintf(stderr, "error: --context-surgery requires complete native "
                        "YaRN metadata with u32 context/original context "
                        "and f32 factor\n");
        return false;
    }
    if (surgery_context <= source_context ||
        surgery_factor <= source_factor) {
        fprintf(stderr, "error: --context-surgery only extends context "
                        "(source %u/x%.9g, requested %u/x%.9g)\n",
                source_context, (double)source_factor, surgery_context,
                (double)surgery_factor);
        return false;
    }
    if (implied_context != surgery_context ||
        fabs(implied - (double)surgery_context) > 0.0001) {
        fprintf(stderr, "error: --context-surgery target %u conflicts "
                        "with original context %u x YaRN factor %.9g "
                        "(implies %.9g)\n", surgery_context,
                original_context, (double)surgery_factor, implied);
        return false;
    }
    ctx->v.u64 = surgery_context;
    uint32_t factor_bits;
    memcpy(&factor_bits, &surgery_factor, sizeof(factor_bits));
    fac->raw = factor_bits;
    fac->v.f64 = surgery_factor;
    if (surgery_result) {
        surgery_result->source_context = source_context;
        surgery_result->original_context = original_context;
        surgery_result->source_factor = source_factor;
        surgery_result->tensors_byte_identical = false;
    }
    fprintf(stderr, "context-surgery: %s %u -> %u; %s %.9g -> %.9g\n",
            ctx_key, source_context, surgery_context, factor_key,
            (double)source_factor, (double)surgery_factor);
    return true;
}

// Prune-plan pre-pass: resolve every plan entry against its layer's own
// router tensor (the one tensor every is_moe layer always has), validate
// ids are in range, and decide whether the WHOLE model ends up with one
// uniform post-prune expert count — the only case expert_count metadata
// can honestly be rewritten to a single new value. Done before any
// output is written so a bad plan fails clean, before touching the
// destination file at all.
static bool quantize_job_check_prune(quantize_job *J, const char *in_path) {
    const char *parch = gguf_get_str(&J->g, "general.architecture", "");
    char bc_key[128];
    snprintf(bc_key, sizeof(bc_key), "%s.block_count", parch);
    snprintf(J->pl_key, sizeof(J->pl_key), "%s.expert_count_per_layer", parch);
    J->n_blocks = gguf_get_u32(&J->g, bc_key, 0);
    J->layer_counts = calloc(J->n_blocks ? J->n_blocks : 1, sizeof(uint32_t));
    if (!J->layer_counts) return false;
    int n_moe_layers = 0, n_moe_seen = 0;
    bool any_pruned = false, all_equal = true;
    int first_count = -1;
    for (uint64_t i = 0; i < J->g.n_tensors; i++) {
        int layer; const char *suffix;
        if (!parse_blk_name(J->g.tensors[i].name, &layer, &suffix)) continue;
        if (strcmp(suffix, "ffn_gate_inp.weight") != 0) continue;
        n_moe_layers++;
        gguf_tensor *router = &J->g.tensors[i];
        if (router->n_dims < 2) {
            fprintf(stderr, "error: %s has %d dims, expected 2\n",
                    router->name, router->n_dims);
            return false;
        }
        int64_t orig_n = router->ne[router->n_dims - 1];
        const prune_layer_t *pl = prune_plan_find(&J->plan, layer);
        int final_count = (int)orig_n;
        if (pl) {
            n_moe_seen++;
            for (int j = 0; j < pl->n_ids; j++)
                if (pl->ids[j] >= orig_n) {
                    fprintf(stderr, "error: prune-plan layer_%d lists "
                            "expert %d, but blk.%d only has %lld experts\n",
                            layer, pl->ids[j], layer, (long long)orig_n);
                    return false;
                }
            // Every tensor resolve_prune() will slice must describe the
            // same expert axis as this router. Validating ids against the
            // router alone is insufficient: a shorter bias/bank would
            // make the copy loop index beyond that tensor's declared
            // data and produce an artifact that cannot load.
            for (uint64_t j = 0; j < J->g.n_tensors; j++) {
                int tensor_layer; const char *tensor_suffix;
                gguf_tensor *et = &J->g.tensors[j];
                if (!parse_blk_name(et->name, &tensor_layer,
                                    &tensor_suffix) ||
                    tensor_layer != layer ||
                    !is_expert_tensor_suffix(tensor_suffix))
                    continue;
                int64_t tensor_n = et->ne[et->n_dims - 1];
                if (tensor_n != orig_n) {
                    fprintf(stderr, "error: %s expert axis has %lld "
                            "entries, but the blk.%d router has %lld\n",
                            et->name, (long long)tensor_n, layer,
                            (long long)orig_n);
                    return false;
                }
            }
            final_count = pl->n_ids;
            if (final_count < orig_n) any_pruned = true;
        }
        if (first_count < 0) first_count = final_count;
        else if (final_count != first_count) all_equal = false;
        if ((uint64_t)layer >= J->n_blocks) {
            fprintf(stderr, "error: %s names block %d but %s.block_count is %llu\n",
                    router->name, layer, parch, (unsigned long long)J->n_blocks);
            return false;
        }
        J->layer_counts[layer] = (uint32_t)final_count;
    }
    if (n_moe_seen < J->plan.n) {
        // at least one "layer_N" in the plan matched no MoE layer in
        // this file at all — almost certainly the plan was built for a
        // different model or has a typo'd layer index
        fprintf(stderr, "error: --prune-experts lists %d layer(s) but "
                "only %d matched an MoE layer in %s\n",
                J->plan.n, n_moe_seen, in_path);
        return false;
    }
    if (any_pruned && all_equal && n_moe_seen == n_moe_layers) {
        J->uniform_prune = true;
        J->uniform_count = first_count;
    }
    return true;
}

// The output's data alignment decided and its partial file open.
static bool quantize_job_open_output(quantize_job *J, const char *out_path) {
    // Honor the file's declared data alignment. The reader derives every
    // tensor's data offset from general.alignment (gguf.c), and we copy that
    // KV verbatim — so the data MUST be laid out at the same alignment, or
    // every offset past the first misaligned tensor is read at the wrong
    // address. (RNR-002: hardcoding 32 silently corrupts any model that
    // declares 64/128.)
    uint64_t align = gguf_get_u32(&J->g, "general.alignment", 32);
    if (align == 0 || (align & (align - 1))) {
        fprintf(stderr, "error: general.alignment=%llu is not a power of two\n",
                (unsigned long long)align);
        return false;
    }
    J->amask = align - 1;

    // Write beside the destination and rename on success, so a failed or
    // interrupted requant never destroys an existing good model — and an
    // in-place requant (in_path == out_path) no longer truncates its own
    // input. (RNR-015.)
    size_t tlen = strlen(out_path) + sizeof(".partial");
    J->tmp_path = malloc(tlen);
    if (!J->tmp_path) return false;
    snprintf(J->tmp_path, tlen, "%s.partial", out_path);

    J->w.f = plat_fopen(J->tmp_path, "wb");
    if (!J->w.f) {
        fprintf(stderr, "error: cannot write %s\n", J->tmp_path);
        return false;
    }
    J->w.ok = true;
    J->tmp_created = true;
    return true;
}

// Rewrite {arch}.expert_count in place (still in memory, not yet
// written) when pruning left every MoE layer at the SAME new count —
// the only case a single global KV can honestly describe. A plan that
// prunes layers to different counts, or leaves some layers unpruned,
// makes expert_count ambiguous by construction; runner reads each
// layer's real count from its own router tensor regardless (model.c),
// so leaving the metadata at its original value there is correct, not
// stale — it's still every layer's true ceiling, just no longer tight
// for the pruned ones.
static void quantize_job_uniform_expert_count(quantize_job *J) {
    const char *arch = gguf_get_str(&J->g, "general.architecture", "");
    char ec_key[128];
    snprintf(ec_key, sizeof(ec_key), "%s.expert_count", arch);
    gguf_kv *ec = gguf_get(&J->g, ec_key);
    if (ec && ec->type == GGUF_T_U32) {
        fprintf(stderr, "quantize: %s %llu -> %d (uniform prune)\n",
                ec_key, (unsigned long long)ec->v.u64, J->uniform_count);
        ec->v.u64 = (uint64_t)J->uniform_count;
    }
}

// Output types are decided BEFORE the header goes out, because
// general.file_type must describe the OUTPUT: copying the parent's
// declaration shipped a mixed Q4_0/Q4_K artifact labeled by its parent
// (finding B of the 2026-08-11 external evaluation — the same
// mixed-tensor trap this project's own cert matrix fails third parties
// for). The declared value is computed from the histogram of what is
// actually written, and the histogram itself is printed so a mixed
// result is visible in the log, not just in a header field.
static bool quantize_job_decide_types(quantize_job *J, int target,
                                      const char *type_plan_path,
                                      const merge_set_t *ms) {
    J->out_type = malloc(sizeof(int) * J->g.n_tensors);
    if (J->g.n_tensors > 0 && !J->out_type) J->w.ok = false;
    // RI-3: a declined rule names the tensor and the type it asked for.
    // An aggregate count told the author of a plan that something was dropped
    // but not what, so the only way to find it was to diff the built histogram
    // against the plan. A few names cost nothing and answer it directly; the
    // cap keeps a MoE's hundreds of expert tensors from burying the summary,
    // and the count still reports the true total.
    // enum, not #define: a macro here would leak to the end of the
    // translation unit for a constant that is meaningful only inside
    // this function. Same hand-maintained hazard as the seq capacities
    // in schema_tools.c, just smaller.
    enum { DECLINE_NAMES = 6 };
    const char *decl_w_name[DECLINE_NAMES]; uint32_t decl_w_type[DECLINE_NAMES];
    // tensors written in a 32-block fallback type because the requested
    // block does not divide their row (width_fallback)
    const char *fb_name[DECLINE_NAMES]; uint32_t fb_want[DECLINE_NAMES], fb_got[DECLINE_NAMES];
    int n_fb = 0; uint64_t fell_back = 0;
    const char *decl_s_name[DECLINE_NAMES]; uint32_t decl_s_type[DECLINE_NAMES];
    int n_decl_w = 0, n_decl_s = 0;
    uint64_t declined_width = 0;
    // Declines because the requested type is no SMALLER than the one already
    // there. That is usually the right call, but two pairs are exactly the
    // same width and materially different quality -- q4_0/q4_K are both 144 B
    // per 256 values, q5_0/q5_K both 176 -- so `--quant q4_k` over a Q4_0 file
    // is a real request that came back as a byte-identical copy with nothing
    // said about it. Counted and reported, as declined_width already is.
    uint64_t declined_size = 0;
    bool merge_unwritable = false;
    for (uint64_t i = 0; J->w.ok && i < J->g.n_tensors; i++) {
        gguf_tensor *t = &J->g.tensors[i];
        if (J->drop && J->drop[i]) { J->out_type[i] = t->type; continue; }
        if (type_plan_path) {
            // The plan decides per tensor. T_KEEP leaves the tensor byte for
            // byte; anything else goes through the same should_quantize,
            // block-width and never-grow guards as a whole-file target, so a
            // plan cannot make a tensor bigger, quantize something that must
            // stay f32, or write a row the target type cannot describe.
            int want = type_plan_pick(&J->tplan, t->name);
            if (keep_source_type(t) || want == T_KEEP) {
                // "keep" means exactly that, for every tensor. Falling through
                // to the whole-file rule below pushed a non-f16 tensor (a bf16
                // norm) to F32 — a converted, grown tensor in a plan whose
                // whole instruction was to convert nothing.
                J->out_type[i] = t->type;
            } else if (!should_quantize(t)) {
                J->out_type[i] = t->type == T_F16 ? T_F16 : T_F32;
            } else if (!type_fits_row(want, t->ne[0]) &&
                       width_fallback(want, t->ne[0]) == want) {
                J->out_type[i] = t->type;
                if (n_decl_w < DECLINE_NAMES) {
                    decl_w_name[n_decl_w] = t->name;
                    decl_w_type[n_decl_w++] = (uint32_t)want;
                }
                declined_width++;
            } else if (!type_fits_row(want, t->ne[0])) {
                int fb = width_fallback(want, t->ne[0]);
                if (n_fb < DECLINE_NAMES) {
                    fb_name[n_fb] = t->name; fb_want[n_fb] = (uint32_t)want;
                    fb_got[n_fb++] = (uint32_t)fb;
                }
                fell_back++;
                want = fb;
                J->out_type[i] = ggml_row_size(t->type, t->ne[0]) <=
                              ggml_row_size(fb, t->ne[0]) ? (int)t->type : fb;
            } else if (ggml_row_size(t->type, t->ne[0]) <=
                       ggml_row_size(want, t->ne[0])) {
                J->out_type[i] = t->type;   // never grow
                if ((uint32_t)want != t->type) {
                    if (n_decl_s < DECLINE_NAMES) {
                        decl_s_name[n_decl_s] = t->name;
                        decl_s_type[n_decl_s++] = (uint32_t)want;
                    }
                    declined_size++;
                }
            } else {
                J->out_type[i] = want;
            }
        } else if (target == T_KEEP) {
            J->out_type[i] = t->type;
        } else {
            const char *only = ms ? NULL : getenv("RUNNER_REQUANT_ONLY");
            bool filtered = keep_source_type(t) ||
                            (only && *only && !strstr(t->name, only));
            if (filtered)
                J->out_type[i] = t->type;
            else if (should_quantize(t) && !type_fits_row(target, t->ne[0]) &&
                     width_fallback(target, t->ne[0]) == target) {
                J->out_type[i] = t->type;
                filtered = true;          // and skip the never-grow rule below
                if (n_decl_w < DECLINE_NAMES) {
                    decl_w_name[n_decl_w] = t->name;
                    decl_w_type[n_decl_w++] = (uint32_t)target;
                }
                declined_width++;
            } else if (should_quantize(t) && !type_fits_row(target, t->ne[0])) {
                int fb = width_fallback(target, t->ne[0]);
                if (n_fb < DECLINE_NAMES) {
                    fb_name[n_fb] = t->name; fb_want[n_fb] = (uint32_t)target;
                    fb_got[n_fb++] = (uint32_t)fb;
                }
                fell_back++;
                J->out_type[i] = fb;         // the never-grow rule below still applies
            } else
                J->out_type[i] = should_quantize(t) ? target
                            : (t->type == T_F16 ? T_F16 : T_F32);
            // never grow a tensor that is already smaller than the target —
            // RUNNER_FORCE_REQUANT overrides this to build same-size
            // conversions (e.g. Q4_K -> Q4_0) for diagnosing per-quant-type
            // correctness bugs. A merge to a wider type is exempt too: asking
            // for a merged f16 from a 4-bit base is a preservation request,
            // not an accident.
            if (!filtered && !ms && should_quantize(t) && !getenv("RUNNER_FORCE_REQUANT") &&
                ggml_row_size(t->type, t->ne[0]) <= ggml_row_size(target, t->ne[0])) {
                if ((uint32_t)target != t->type) {
                    if (n_decl_s < DECLINE_NAMES) {
                        decl_s_name[n_decl_s] = t->name;
                        decl_s_type[n_decl_s++] = (uint32_t)target;
                    }
                    declined_size++;
                }
                J->out_type[i] = t->type;
            }
        }
        // an adapted tensor cannot keep a type this file has no quantizer
        // for — refuse before writing a byte rather than skip the delta
        if (ms && ms->map[i].r && !ms->map[i].zero &&
            !quantize_row_writable(J->out_type[i])) {
            fprintf(stderr, "error: cannot merge into %s — no %s "
                    "quantizer; pass --quant to pick a writable output "
                    "type\n", t->name, ggml_type_name(J->out_type[i]));
            merge_unwritable = true;
        }
    }
    if (merge_unwritable) return false;
    bool strict_refuse = g_type_plan_strict && type_plan_path &&
                         (fell_back || declined_width || declined_size);
    if (J->w.ok && fell_back) {
        fprintf(stderr, "quantize: %llu tensor(s) written in a 32-block fallback "
                "type — the requested type's block does not divide their row "
                "width\n", (unsigned long long)fell_back);
        for (int k = 0; k < n_fb; k++)
            fprintf(stderr, "quantize:   %s wanted %s, wrote %s\n", fb_name[k],
                    ggml_type_name((int)fb_want[k]), ggml_type_name((int)fb_got[k]));
        if (fell_back > (uint64_t)n_fb)
            fprintf(stderr, "quantize:   ... and %llu more\n",
                    (unsigned long long)(fell_back - (uint64_t)n_fb));
    }
    if (J->w.ok && declined_width) {
        fprintf(stderr, "quantize: %llu tensor(s) kept their own type — the "
                "requested type's block does not divide their row width\n",
                (unsigned long long)declined_width);
        for (int k = 0; k < n_decl_w; k++)
            fprintf(stderr, "quantize:   %s wanted %s\n",
                    decl_w_name[k], ggml_type_name(decl_w_type[k]));
        if (declined_width > (uint64_t)n_decl_w)
            fprintf(stderr, "quantize:   ... and %llu more\n",
                    (unsigned long long)(declined_width - (uint64_t)n_decl_w));
    }
    if (J->w.ok && declined_size) {
        fprintf(stderr, "quantize: %llu tensor(s) kept their own type — the "
                "requested type is not smaller than what they already carry "
                "(set RUNNER_FORCE_REQUANT=1 to convert anyway)\n",
                (unsigned long long)declined_size);
        for (int k = 0; k < n_decl_s; k++)
            fprintf(stderr, "quantize:   %s wanted %s\n",
                    decl_s_name[k], ggml_type_name(decl_s_type[k]));
        if (declined_size > (uint64_t)n_decl_s)
            fprintf(stderr, "quantize:   ... and %llu more\n",
                    (unsigned long long)(declined_size - (uint64_t)n_decl_s));
    }
    if (strict_refuse) {
        fprintf(stderr, "error: --type-plan-strict: %llu plan rule(s) would "
                "not be applied as written (listed above); nothing was "
                "written\n", (unsigned long long)(fell_back + declined_width +
                                                  declined_size));
        return false;
    }
    return true;
}

// dominant non-f32 output type -> its MOSTLY_* code. The K-quant
// S/M/L distinction is not recoverable from a one-type histogram;
// the M code is the closest honest label and the printed histogram
// carries the exact truth.
static uint32_t quantize_job_file_type(const quantize_job *J) {
    if (!J->w.ok || J->g.n_tensors == 0) return 0;
    static const struct { int t; uint32_t ftype; const char *name; } FT[] = {
        { T_F16, 1, "F16" }, { T_BF16, 32, "BF16" },
        { T_Q4_0, 2, "Q4_0" }, { T_Q4_1, 3, "Q4_1" },
        { T_Q5_0, 8, "Q5_0" }, { T_Q5_1, 9, "Q5_1" },
        { T_Q8_0, 7, "Q8_0" },
        { T_Q2_K, 10, "Q2_K" }, { T_Q3_K, 12, "Q3_K" },
        { T_Q4_K, 15, "Q4_K" }, { T_Q5_K, 17, "Q5_K" },
        { T_Q6_K, 18, "Q6_K" },
        { T_IQ2_XXS, 19, "IQ2_XXS" }, { T_IQ2_XS, 20, "IQ2_XS" },
        { T_IQ2_S, 28, "IQ2_S" }, { T_IQ3_XXS, 23, "IQ3_XXS" },
        { T_IQ3_S, 26, "IQ3_S" }, { T_IQ1_S, 24, "IQ1_S" },
        { T_IQ1_M, 31, "IQ1_M" }, { T_IQ4_NL, 25, "IQ4_NL" },
        { T_IQ4_XS, 30, "IQ4_XS" }, { T_MXFP4, 38, "MXFP4" },
        { T_NVFP4, 39, "NVFP4" }, { T_Q2_0, 41, "Q2_0" },   // llama.h LLAMA_FTYPE
    };
    uint64_t counts[sizeof(FT) / sizeof(*FT)] = {0};
    uint64_t f32s = 0;
    for (uint64_t i = 0; i < J->g.n_tensors; i++) {
        if (J->drop && J->drop[i]) continue;
        if (J->out_type[i] == T_F32) { f32s++; continue; }
        for (size_t k = 0; k < sizeof(FT) / sizeof(*FT); k++)
            if (FT[k].t == J->out_type[i]) { counts[k]++; break; }
    }
    size_t dom = 0;
    uint64_t best = 0;
    for (size_t k = 0; k < sizeof(FT) / sizeof(*FT); k++)
        if (counts[k] > best) { best = counts[k]; dom = k; }
    uint32_t ftype_out = best ? FT[dom].ftype : 0;   // all-f32 -> ALL_F32
    fprintf(stderr, "quantize: general.file_type %u (MOSTLY_%s) — output histogram:",
            ftype_out, best ? FT[dom].name : "F32");
    if (f32s) fprintf(stderr, " F32:%llu", (unsigned long long)f32s);
    for (size_t k = 0; k < sizeof(FT) / sizeof(*FT); k++)
        if (counts[k]) fprintf(stderr, " %s:%llu", FT[k].name,
                               (unsigned long long)counts[k]);
    fprintf(stderr, "\n");
    return ftype_out;
}

// The header: every source KV the transformation does not re-author, then
// the re-authored ones.
static void quantize_job_write_header(quantize_job *J) {
    uint32_t ftype_out = quantize_job_file_type(J);
    // A pruned source may already carry the per-layer declaration; a new
    // plan re-authors it (never duplicates it), a plain requant copies it
    // through like every other KV.
    uint64_t dropped_kv = 0;
    for (uint64_t i = 0; i < J->g.n_kv; i++) {
        bool re = kv_is_reauthored(J->g.kv[i].key) ||
                  (J->layer_counts && !strcmp(J->g.kv[i].key, J->pl_key));
        for (int k = 0; !re && k < J->n_reauth; k++)
            re = !strcmp(J->g.kv[i].key, J->reauth[k].key);
        if (re) dropped_kv++;
    }

    wr_u32(&J->w, 0x46554747);
    wr_u32(&J->w, 3);
    wr_u64(&J->w, J->g.n_tensors - J->n_dropped);
    wr_u64(&J->w, J->g.n_kv - dropped_kv + 1 + (J->layer_counts ? 1 : 0) + (uint64_t)J->n_reauth);
    for (uint64_t i = 0; i < J->g.n_kv; i++) {
        bool re = kv_is_reauthored(J->g.kv[i].key) ||
                  (J->layer_counts && !strcmp(J->g.kv[i].key, J->pl_key));
        for (int k = 0; !re && k < J->n_reauth; k++)
            re = !strcmp(J->g.kv[i].key, J->reauth[k].key);
        if (re) continue;
        wr_kv(&J->w, &J->g.kv[i]);
    }
    for (int k = 0; k < J->n_reauth; k++) {
        // per-layer arrays declaring the removed sublayers (0 = absent)
        char bc_key[160];
        snprintf(bc_key, sizeof bc_key, "%s.block_count",
                 gguf_get_str(&J->g, "general.architecture", ""));
        uint64_t nb = gguf_get_u32(&J->g, bc_key, 0);
        wr_str(&J->w, J->reauth[k].key, strlen(J->reauth[k].key));
        wr_u32(&J->w, 9);        // GGUF_T_ARR
        wr_u32(&J->w, 4);        // of GGUF_T_U32
        wr_u64(&J->w, nb);
        for (uint64_t i = 0; i < nb; i++) wr_u32(&J->w, J->reauth[k].vals[i]);
        fprintf(stderr, "quantize: %s written as a per-block array (0 = removed)\n",
                J->reauth[k].key);
    }
    wr_str(&J->w, "general.file_type", strlen("general.file_type"));
    wr_u32(&J->w, 4);            // GGUF_T_U32
    wr_u32(&J->w, ftype_out);
    if (J->layer_counts) {
        wr_str(&J->w, J->pl_key, strlen(J->pl_key));
        wr_u32(&J->w, 9);        // GGUF_T_ARR
        wr_u32(&J->w, 4);        // of GGUF_T_U32
        wr_u64(&J->w, J->n_blocks);
        for (uint64_t i = 0; i < J->n_blocks; i++) wr_u32(&J->w, J->layer_counts[i]);
        fprintf(stderr, "quantize: %s written (%llu blocks; the global "
                "expert_count stays the ceiling)\n",
                J->pl_key, (unsigned long long)J->n_blocks);
    }
}

// The tensor table (each tensor's type, written shape and data offset)
// and the padding up to the data.
static void quantize_job_write_layout(quantize_job *J) {
    // tensor table with new types/offsets, at the file's data alignment
    uint64_t off = 0;
    J->out_off = malloc(sizeof(uint64_t) * J->g.n_tensors);
    // effective (possibly pruned) ne[] this tensor is WRITTEN with — equal
    // to t->ne[] unless resolve_prune() slices its expert axis
    J->eff_ne = malloc(sizeof(int64_t[4]) * (J->g.n_tensors ? J->g.n_tensors : 1));
    if ((J->g.n_tensors > 0 && (!J->out_type || !J->out_off || !J->eff_ne))) J->w.ok = false;
    for (uint64_t i = 0; i < J->g.n_tensors; i++) {
        gguf_tensor *t = &J->g.tensors[i];
        if (!J->w.ok) break;
        if (J->drop && J->drop[i]) { J->out_off[i] = 0; continue; }
        for (int d = 0; d < 4; d++) J->eff_ne[i][d] = t->ne[d];
        const prune_layer_t *pl = resolve_prune(t, &J->plan);
        if (pl) J->eff_ne[i][t->n_dims - 1] = pl->n_ids;
        // out_type[i] was decided by quantize_job_decide_types(), before
        // the header (general.file_type is derived from it)
        uint64_t rows, bytes, end;
        if (!checked_u64_mul((uint64_t)J->eff_ne[i][1], (uint64_t)J->eff_ne[i][2], &rows) ||
            !checked_u64_mul(rows, (uint64_t)J->eff_ne[i][3], &rows) ||
            !checked_u64_mul(ggml_row_size(J->out_type[i], J->eff_ne[i][0]), rows, &bytes) ||
            !checked_u64_add(off, bytes, &end) ||
            !checked_u64_add(end, J->amask, &end)) {
            J->w.ok = false;
            break;
        }
        J->out_off[i] = off;
        off = end & ~J->amask;

        wr_str(&J->w, t->name, strlen(t->name));
        wr_u32(&J->w, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) wr_u64(&J->w, (uint64_t)J->eff_ne[i][d]);
        wr_u32(&J->w, (uint32_t)J->out_type[i]);
        wr_u64(&J->w, J->out_off[i]);
    }
    // pad to data section
    int64_t hdr_end = wr_tell(&J->w);
    J->data_start = hdr_end >= 0 ? ((uint64_t)hdr_end + J->amask) & ~J->amask : 0;
    if (hdr_end >= 0 && J->data_start < (uint64_t)hdr_end) J->w.ok = false;
    wr_pad_to(&J->w, J->data_start);
}

// One pruned tensor: its kept experts' blocks, in the plan's order.
// The tensor's imatrix entry when its output type takes one, counted for the
// summary; NULL otherwise.
static const imatrix_entry *imatrix_for_tensor(const gguf_tensor *t, int out_type) {
    if (!g_imx.n) return NULL;
    if (!imatrix_weighted_type(out_type)) {
        g_imx.tensors_other_type++;
        return NULL;
    }
    const imatrix_entry *e = imatrix_find(t->name);
    if (e) g_imx.tensors_weighted++;
    return e;
}

// One row's weights, or NULL; a misfit entry fails the write.
static const float *imatrix_row_checked(quantize_job *J, const gguf_tensor *t,
                                        const imatrix_entry *e, uint64_t row) {
    if (!e) return NULL;
    uint64_t mats = t->n_dims > 2 && t->ne[2] > 1 ? t->ne[2] : 1;
    uint64_t per = mats > 1 ? t->ne[1] : t->ne[1] * t->ne[2] * t->ne[3];
    bool bad = false;
    const float *w = imatrix_row(e, (int64_t)t->ne[0], row, per, mats, &bad);
    if (bad) {
        fprintf(stderr, "error: the imatrix entry for %s (%lld values x %lld) does "
                "not fit the tensor ([%llu, %llu, %llu])\n", t->name,
                (long long)e->n_per_row, (long long)e->n_mat,
                (unsigned long long)t->ne[0], (unsigned long long)t->ne[1],
                (unsigned long long)t->ne[2]);
        J->w.ok = false;
        return NULL;
    }
    if (w) g_imx.rows_weighted++;
    else g_imx.rows_without_data++;
    return w;
}

static void quantize_job_write_pruned(quantize_job *J, uint64_t i,
                                      const prune_layer_t *pl) {
    gguf_tensor *t = &J->g.tensors[i];
    // Pruned: copy/convert only the kept experts' blocks, IN PLAN
    // ORDER, instead of the tensor's full contiguous row range —
    // each expert is one contiguous block of exp_rows rows (1 for
    // the router/exps-bias 2D shapes, 1 element for the 1D
    // bias/scale shapes, n_ff_exp rows for the fused 3D expert
    // weights), so this is the pruning itself: the survivors, in
    // the plan's order, is the new expert index space — no
    // separate id remapping happens anywhere at runtime.
    int64_t row_len, exp_rows;
    expert_block_geometry(t, &row_len, &exp_rows);
    size_t in_row = ggml_row_size(t->type, row_len);
    size_t per_expert_in = in_row * (size_t)exp_rows;
    if ((uint32_t)J->out_type[i] == t->type) {
        for (int j = 0; J->w.ok && j < pl->n_ids; j++)
            wr(&J->w, (const uint8_t *)t->data +
                   (size_t)pl->ids[j] * per_expert_in, per_expert_in);
        J->kept++;
        return;
    }
    size_t out_rs = ggml_row_size(J->out_type[i], row_len);
    const imatrix_entry *ie = imatrix_for_tensor(t, J->out_type[i]);
    for (int j = 0; J->w.ok && j < pl->n_ids; j++) {
        for (int64_t r = 0; J->w.ok && r < exp_rows; r++) {
            int64_t src_row = (int64_t)pl->ids[j] * exp_rows + r;
            dequant_row(t->type, (const uint8_t *)t->data +
                        (size_t)src_row * in_row, J->rowf, (int)row_len);
            // the source expert's weights: the imatrix was made on the
            // unpruned model
            const float *qw = imatrix_row_checked(J, t, ie, (uint64_t)src_row);
            quantize_row_any_w(J->out_type[i], J->rowf, J->rowq, (int)row_len, qw);
            wr(&J->w, J->rowq, out_rs);
        }
    }
    J->quantized++;
}

// One tensor written whole: copied when its type stays and no delta
// applies, else row by row through f32, with a merge's delta folded in
// and its survival measured.
static void quantize_job_write_rows(quantize_job *J, uint64_t i, merge_set_t *ms) {
    gguf_tensor *t = &J->g.tensors[i];
    // a zero-delta pair leaves its tensor on the verbatim path below —
    // copying the base bytes is exactly the merge result, without a
    // gratuitous dequant->requant of untouched weights
    const merge_pair_t *mp = ms && ms->map[i].r && !ms->map[i].zero
                             ? &ms->map[i] : NULL;
    uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
    size_t in_rs = ggml_row_size(t->type, t->ne[0]);
    size_t out_rs = ggml_row_size(J->out_type[i], t->ne[0]);
    if (!mp && (uint32_t)J->out_type[i] == t->type) {
        uint64_t bytes;
        if (!checked_u64_mul(in_rs, rows, &bytes) || bytes > SIZE_MAX)
            J->w.ok = false;
        else
            wr(&J->w, t->data, (size_t)bytes);
        J->kept++;
        return;
    }
    if (mp && t->scale != 1.0f) {
        // The base rows are stored unscaled with a `.scale` companion the
        // loader applies; folding a delta into them would put the delta
        // under that companion too. Refuse rather than write a merged
        // file that is wrong by 1/scale on the adapted part.
        fprintf(stderr, "error: %s carries a per-tensor scale companion; "
                "--merge-lora into a scaled base is not supported "
                "(serve base + --lora instead)\n", t->name);
        J->w.ok = false;
        return;
    }
    int n0 = (int)t->ne[0];
    const imatrix_entry *ie = imatrix_for_tensor(t, J->out_type[i]);
    for (uint64_t r = 0; J->w.ok && r < rows; r++) {
        dequant_row(t->type, (uint8_t *)t->data + r * in_rs, J->rowf, n0);
        if (mp) {
            memcpy(J->basef, J->rowf, sizeof(float) * (size_t)n0);
            merge_row(mp, r, J->rowf, n0);
        }
        const float *qw = imatrix_row_checked(J, t, ie, r);
        quantize_row_any_w(J->out_type[i], J->rowf, J->rowq, n0, qw);
        wr(&J->w, J->rowq, out_rs);
        if (mp) {
            // Did the grid keep the delta? The base alone, written at
            // the same type, is what the merge would be had the adapter
            // done nothing; the projection of (merged - that) onto the
            // intended delta is the part of the delta that survived.
            quantize_row_any_w(J->out_type[i], J->basef, J->rowqb, n0, qw);
            merge_survival *sv = &ms->surv;
            sv->bytes += out_rs;
            for (size_t k = 0; k < out_rs; k++)
                sv->bytes_changed += J->rowq[k] != J->rowqb[k];
            dequant_row(J->out_type[i], J->rowq, J->deqm, n0);
            dequant_row(J->out_type[i], J->rowqb, J->deqb, n0);
            for (int k = 0; k < n0; k++) {
                double d = (double)J->rowf[k] - J->basef[k];
                sv->ret_num += ((double)J->deqm[k] - J->deqb[k]) * d;
                sv->ret_den += d * d;
            }
        }
    }
    J->quantized++;
}

// Every written tensor's data, each at its offset.
static void quantize_job_write_tensors(quantize_job *J, merge_set_t *ms) {
    // tensor data
    size_t max_row = 0;
    for (uint64_t i = 0; i < J->g.n_tensors; i++)
        if (J->g.tensors[i].ne[0] > max_row) max_row = J->g.tensors[i].ne[0];
    J->rowf = malloc(sizeof(float) * max_row);
    J->rowq = malloc(ggml_row_size(T_F32, max_row) + 64);
    if (!J->rowf || !J->rowq) J->w.ok = false;
    // the survival measurement's second write of each adapted row
    if (ms) {
        J->basef = malloc(sizeof(float) * max_row);
        J->deqm = malloc(sizeof(float) * max_row);
        J->deqb = malloc(sizeof(float) * max_row);
        J->rowqb = malloc(ggml_row_size(T_F32, max_row) + 64);
        if (!J->basef || !J->deqm || !J->deqb || !J->rowqb) J->w.ok = false;
    }

    for (uint64_t i = 0; J->w.ok && i < J->g.n_tensors; i++) {
        gguf_tensor *t = &J->g.tensors[i];
        if (J->drop && J->drop[i]) continue;
        uint64_t want;
        if (!checked_u64_add(J->data_start, J->out_off[i], &want)) {
            J->w.ok = false;
            break;
        }
        wr_pad_to(&J->w, want);

        const prune_layer_t *pl = resolve_prune(t, &J->plan);
        if (pl) quantize_job_write_pruned(J, i, pl);
        else quantize_job_write_rows(J, i, ms);
    }
}

// The partial file closed and checked, a merge held to its survival floor,
// the output installed over the destination.
static int quantize_job_finish(quantize_job *J, const char *in_path,
                               const char *out_path, int target,
                               const char *type_plan_path,
                               const merge_set_t *ms) {
    bool write_ok = wr_close(&J->w);
    // The source closes before the install: an in-place requant replaces
    // the very file the source maps.
    quantize_job_close_source(J);

    if (!write_ok) {
        fprintf(stderr, "error: failed writing quantized model %s "
                "(destination left untouched)\n", out_path);
        return 1;
    }

    // A merge whose output grid rounded the delta away is not the adapted
    // model: refuse it before it replaces anything (the 2026-08-22 study's
    // 4-bit merge scored the base's 0.69 again with 98.55% of the adapted
    // bytes on the base's own codes). --merge-allow-erased writes it anyway.
    if (ms && ms->surv.ret_den > 0) {
        double ret = ms->surv.ret_num / ms->surv.ret_den;
        if (ms->min_retained >= 0 && ret < ms->min_retained) {
            fprintf(stderr, "error: the merge kept %.1f%% of the adapter's "
                    "delta (%.2f%% of the adapted bytes differ from the base "
                    "written alone at the same type), under the %.0f%% "
                    "floor: the output grid rounded the fine-tune away, so "
                    "%s would behave like the base. Merge into a type wider "
                    "than the base's own (--quant f16 keeps the delta as "
                    "computed; q8_0 is enough only over a narrower base, such "
                    "as a 4-bit one), serve base + --lora, or pass "
                    "--merge-allow-erased to write it anyway (destination "
                    "left untouched)\n", 100.0 * ret,
                    100.0 * (double)ms->surv.bytes_changed /
                        (double)ms->surv.bytes,
                    100.0 * ms->min_retained, out_path);
            return 1;
        }
    }

    // Install the finished file only after the temp is complete. POSIX rename()
    // atomically replaces an existing destination; Windows needs MoveFileEx
    // with replace semantics (plat_replace_file). Never remove the destination first: if the final
    // install fails, the caller must still have the previous model.
    if (install_finished_file(J->tmp_path, out_path) != 0) {
        fprintf(stderr, "error: wrote %s but could not install it as %s "
                "(destination unchanged)\n", J->tmp_path, out_path);
        return 1;
    }
    J->tmp_created = false;   // installed: it is the destination now

    // Under a --type-plan the whole-file `target` decided nothing, so naming
    // it here described the output as a type no tensor was written in.
    fprintf(stderr, "quantize: %s -> %s (%s): %llu tensors converted, %llu kept\n",
            in_path, out_path,
            type_plan_path ? "per-tensor plan"
                           : target == T_KEEP ? "per-tensor unchanged"
                                              : ggml_type_name(target),
            (unsigned long long)J->quantized, (unsigned long long)J->kept);
    if (g_imx.n) {
        // R4.7.2: what the importance matrix shaped, and what it could not
        fprintf(stderr, "imatrix: %llu tensors weighted (%llu rows; %llu rows "
                "without calibration data written unweighted); %llu converted "
                "tensors are of a type the imatrix does not shape (only q3_k "
                "and q4_0 are weighted)\n",
                (unsigned long long)g_imx.tensors_weighted,
                (unsigned long long)g_imx.rows_weighted,
                (unsigned long long)g_imx.rows_without_data,
                (unsigned long long)g_imx.tensors_other_type);
        if (!g_imx.tensors_weighted)
            fprintf(stderr, "warning: the imatrix named none of the tensors "
                    "written as q3_k or q4_0\n");
    }
    if (ms)
        fprintf(stderr, "merge: %d adapted projection%s folded into the "
                "weights%s\n", ms->n_pairs - ms->n_zero,
                ms->n_pairs - ms->n_zero == 1 ? "" : "s",
                ms->n_zero ? " (zero-delta pairs copied verbatim)" : "");
    if (ms && ms->surv.ret_den > 0)
        fprintf(stderr, "merge: the output kept %.1f%% of the adapter's delta; "
                "%.2f%% of the adapted bytes differ from the base written "
                "alone\n", 100.0 * ms->surv.ret_num / ms->surv.ret_den,
                100.0 * (double)ms->surv.bytes_changed / (double)ms->surv.bytes);
    return 0;
}

// One file transformation: --quantize (with a type plan, an expert prune
// or a sublayer removal), --merge-lora (ms) or --context-surgery
// (surgery_context). Each stage adds to one quantize_job and refuses by
// returning false; the job's one release path unwinds every stage before
// the one that refused.
static int quantize_gguf_plan_inner(const char *in_path, const char *out_path, int target,
                       const char *prune_path, const char *type_plan_path,
                       const char *remove_spec,
                       merge_set_t *ms, uint32_t surgery_context,
                       float surgery_factor, context_surgery_result *surgery_result) {
    if (target != T_KEEP && !quantize_row_writable(target)) {
        fprintf(stderr, "error: quantize target must be q8_0, q4_0, q3_k, "
                "q4_k, q6_k, mxfp4, f16, or bf16\n");
        return 1;
    }
    quantize_job job;
    memset(&job, 0, sizeof(job));
    quantize_job *J = &job;
    int rc = 1;
    if (!quantize_job_open_source(J, in_path, prune_path, type_plan_path,
                                  remove_spec))
        goto done;
    if (J->rm.n > 0 && !quantize_job_remove_sublayers(J)) goto done;
    if (ms && ms->n_base != J->g.n_tensors) {
        // the base changed between the resolve open and this one
        fprintf(stderr, "error: %s changed while merging\n", in_path);
        goto done;
    }
    if (surgery_context &&
        !quantize_job_context_surgery(J, surgery_context, surgery_factor,
                                      surgery_result))
        goto done;
    if (J->plan.n > 0 && !quantize_job_check_prune(J, in_path)) goto done;
    if (!quantize_job_open_output(J, out_path)) goto done;
    if (J->uniform_prune) quantize_job_uniform_expert_count(J);
    if (!quantize_job_decide_types(J, target, type_plan_path, ms)) goto done;
    // a failed write clears J->w.ok, and the finish refuses the file
    quantize_job_write_header(J);
    quantize_job_write_layout(J);
    quantize_job_write_tensors(J, ms);
    rc = quantize_job_finish(J, in_path, out_path, target, type_plan_path, ms);
done:
    quantize_job_release(J);
    return rc;
}

// ---------------------------------------------------------------- error report
// R4.7.3: what each candidate type would cost each tensor, measured with the
// writers a rewrite uses (the importance-weighted ones when an imatrix is
// set). Per tensor that a rewrite would quantize and per type in `types`,
// one JSON line carries the type's bytes and the weighted squared round-trip
// error, sum over rows of sum_i w_i (x_i - deq(quant(x))_i)^2, with w_i the
// imatrix's column weight (1 without one) and the tensor's scale companion
// applied. With w the mean squared activation of each input column, that is
// the expected squared output perturbation the type adds to the matrix: the
// proxy scripts/type-plan-by-sensitivity.py ranks tensors by. Nothing is
// written but the report.
int quantize_error_report(const char *in_path, const int *types, int n_types,
                          const char *out_path) {
    gguf_file g;
    if (!gguf_open(&g, in_path)) return 1;
    FILE *out = plat_fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        gguf_close(&g);
        return 1;
    }
    size_t max_row = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++)
        if (g.tensors[i].ne[0] > max_row) max_row = g.tensors[i].ne[0];
    float *rowf = malloc(sizeof(float) * (max_row ? max_row : 1));
    float *deq = malloc(sizeof(float) * (max_row ? max_row : 1));
    uint8_t *rowq = malloc(ggml_row_size(T_F32, max_row) + 64);
    double *err = calloc((size_t)n_types, sizeof(double));
    int rc = rowf && deq && rowq && err ? 0 : 1;
    fp_denormal_state fps = fp_denormals_disable();
    uint64_t reported = 0;
    for (uint64_t i = 0; rc == 0 && i < g.n_tensors; i++) {
        gguf_tensor *t = &g.tensors[i];
        if (!should_quantize(t) || keep_source_type(t)) continue;
        if (!ggml_type_supported(t->type)) {
            fprintf(stderr, "error: tensor %s has unsupported type %s\n",
                    t->name, ggml_type_name(t->type));
            rc = 1;
            break;
        }
        int n0 = (int)t->ne[0];
        uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
        uint64_t mats = t->n_dims > 2 && t->ne[2] > 1 ? t->ne[2] : 1;
        uint64_t per = mats > 1 ? t->ne[1] : rows;
        size_t in_rs = ggml_row_size(t->type, t->ne[0]);
        const imatrix_entry *ie = g_imx.n ? imatrix_find(t->name) : NULL;
        bool weighted = false;
        memset(err, 0, sizeof(double) * (size_t)n_types);
        for (uint64_t r = 0; rc == 0 && r < rows; r++) {
            dequant_row(t->type, (const uint8_t *)t->data + r * in_rs, rowf, n0);
            bool bad = false;
            const float *w = ie ? imatrix_row(ie, n0, r, per, mats, &bad) : NULL;
            if (bad) {
                fprintf(stderr, "error: the imatrix entry for %s does not fit the tensor\n",
                        t->name);
                rc = 1;
                break;
            }
            weighted |= w != NULL;
            for (int k = 0; k < n_types; k++) {
                if (!type_fits_row(types[k], n0)) continue;
                quantize_row_any_w(types[k], rowf, rowq, n0,
                                   imatrix_weighted_type(types[k]) ? w : NULL);
                dequant_row(types[k], rowq, deq, n0);
                double s = 0;
                for (int j = 0; j < n0; j++) {
                    double d = (double)rowf[j] - deq[j];
                    s += (w ? (double)w[j] : 1.0) * d * d;
                }
                err[k] += s;
            }
        }
        if (rc) break;
        double sc2 = (double)t->scale * t->scale;
        fprintf(out, "{\"tensor\":\"%s\",\"source_type\":\"%s\",\"rows\":%llu,\"cols\":%d,"
                "\"weighted\":%s,\"types\":{", t->name, ggml_type_name(t->type),
                (unsigned long long)rows, n0, weighted ? "true" : "false");
        for (int k = 0; k < n_types; k++) {
            bool fits = type_fits_row(types[k], n0);
            // the --type-plan spelling (lower case), so a planner can write
            // the report's keys straight into a plan
            char tn[16];
            snprintf(tn, sizeof tn, "%s", ggml_type_name(types[k]));
            for (char *c = tn; *c; c++) *c = (char)tolower((unsigned char)*c);
            fprintf(out, "%s\"%s\":", k ? "," : "", tn);
            if (fits)
                fprintf(out, "{\"bytes\":%llu,\"error\":%.9g}",
                        (unsigned long long)(ggml_row_size(types[k], t->ne[0]) * rows),
                        err[k] * sc2);
            else
                fprintf(out, "null");
        }
        fprintf(out, "}}\n");
        reported++;
    }
    fp_denormals_restore(fps);
    if (fclose(out) != 0) rc = 1;
    free(rowf); free(deq); free(rowq); free(err);
    gguf_close(&g);
    if (rc == 0)
        fprintf(stderr, "quantize: error report of %llu tensors -> %s%s\n",
                (unsigned long long)reported, out_path,
                g_imx.n ? " (imatrix-weighted)" : "");
    return rc;
}
