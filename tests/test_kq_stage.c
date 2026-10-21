// The CUDA tensor-core stagers of the K-quants and legacy quants without a
// raw-register loader, executed on the host.
//
// src/kq_stage.h is the arithmetic the device runs to decode a Q5_K, Q3_K
// or Q2_K segment and a Q4_1, Q5_0 or Q5_1 block into the tensor-core
// operand; nvcc compiles it into kernels_kq_stage.cuh, and this test
// compiles the same file with the C compiler and holds every element to
// quants.c's dequant_row(), bit for bit, on random blocks and on the scale
// corners (on the host; the device's FMA contraction can move a last bit
// before the fp16 rounding). A stager that read the wrong scale, the wrong
// high bit or the wrong quarter of a block fails here without a GPU; the
// reference is dequant_row itself, which the CPU path and its own tests
// hold.
//
//     ./test-kq-stage
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_stage.h"
#include "quants.h"
#include "fp16.h"

static int g_fail = 0;

static uint32_t g_rng = 0x2545f491u;
static unsigned rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

static void put_f16(iq_byte *p, unsigned bits) { p[0] = bits & 0xFF; p[1] = bits >> 8; }
// a finite half that is not tiny: the scale of a real file
static unsigned rnd_f16_scale(void) {
    unsigned e = 1 + rnd() % 29;
    return (rnd() & 0x8000u) | (e << 10) | (rnd() & 0x3ffu);
}

static int f32_bits_equal(float a, float b) {
    uint32_t x, y;
    memcpy(&x, &a, 4); memcpy(&y, &b, 4);
    return x == y;
}

// one format: block bytes, elements a block, where its f16 scales sit, and
// its stager over the whole block (four segments, or the one block)
typedef struct {
    const char *name;
    int type, nbytes, nelems;
    int scale_at[2];               // byte offsets of d (and dmin / m), -1 none
    void (*stage)(float *dst, const iq_byte *blk);
} kq_format;

static void st_q5k(float *dst, const iq_byte *blk) {
    for (int s = 0; s < 4; s++) q5k_stage64(dst + 64 * s, blk, s);
}
static void st_q3k(float *dst, const iq_byte *blk) {
    for (int s = 0; s < 4; s++) q3k_stage64(dst + 64 * s, blk, s);
}
static void st_q2k(float *dst, const iq_byte *blk) {
    for (int s = 0; s < 4; s++) q2k_stage64(dst + 64 * s, blk, s);
}
static void st_q41(float *dst, const iq_byte *blk) { q41_stage32(dst, blk); }
static void st_q50(float *dst, const iq_byte *blk) { q50_stage32(dst, blk); }
static void st_q51(float *dst, const iq_byte *blk) { q51_stage32(dst, blk); }

static const kq_format FORMATS[] = {
    { "Q5_K", T_Q5_K, 176, 256, { 0, 2 },   st_q5k },
    { "Q3_K", T_Q3_K, 110, 256, { 108, -1 }, st_q3k },
    { "Q2_K", T_Q2_K, 84,  256, { 80, 82 },  st_q2k },
    { "Q4_1", T_Q4_1, 20,  32,  { 0, 2 },   st_q41 },
    { "Q5_0", T_Q5_0, 22,  32,  { 0, -1 },  st_q50 },
    { "Q5_1", T_Q5_1, 24,  32,  { 0, 2 },   st_q51 },
};

static int block_matches(const kq_format *f, const iq_byte *blk, const char *what) {
    float want[256], got[256];
    dequant_row(f->type, blk, want, f->nelems);
    f->stage(got, blk);
    for (int i = 0; i < f->nelems; i++)
        if (!f32_bits_equal(want[i], got[i])) {
            fprintf(stderr, "FAIL: %s %s element %d: staged %.9g, dequant_row %.9g\n",
                    f->name, what, i, (double)got[i], (double)want[i]);
            g_fail = 1;
            return 0;
        }
    return 1;
}

static void fill(const kq_format *f, iq_byte *blk, unsigned d_bits, unsigned m_bits) {
    for (int i = 0; i < f->nbytes; i++) blk[i] = (iq_byte)rnd();
    put_f16(blk + f->scale_at[0], d_bits);
    if (f->scale_at[1] >= 0) put_f16(blk + f->scale_at[1], m_bits);
}

int main(void) {
    f16_init();
    static const unsigned corners[] = { 0x0000, 0x0001, 0x7bff, 0x8001, 0xfbff };
    int checked = 0;
    for (size_t k = 0; k < sizeof FORMATS / sizeof *FORMATS; k++) {
        const kq_format *f = &FORMATS[k];
        iq_byte blk[176];
        int ok = 1;
        // 2,000 random blocks: every scale, high-bit and quant field random
        for (int n = 0; n < 2000 && ok; n++) {
            fill(f, blk, rnd_f16_scale(), rnd_f16_scale());
            ok = block_matches(f, blk, "random block");
        }
        // the scale corners (zero, the smallest subnormal, the largest
        // half, both signs) on d and on the min
        for (size_t c = 0; c < sizeof corners / sizeof *corners && ok; c++) {
            fill(f, blk, corners[c], rnd_f16_scale());
            ok = block_matches(f, blk, "d corner");
            if (ok && f->scale_at[1] >= 0) {
                fill(f, blk, rnd_f16_scale(), corners[c]);
                ok = block_matches(f, blk, "min corner");
            }
        }
        // every quant and high bit set, then every one clear
        for (int v = 0; v < 2 && ok; v++) {
            memset(blk, v ? 0xFF : 0x00, sizeof blk);
            put_f16(blk + f->scale_at[0], 0x3c00);
            if (f->scale_at[1] >= 0) put_f16(blk + f->scale_at[1], 0x3800);
            ok = block_matches(f, blk, v ? "all-ones block" : "all-zeros block");
        }
        checked += ok;
    }
    if (g_fail) {
        fprintf(stderr, "test_kq_stage: FAILED\n");
        return 1;
    }
    printf("test_kq_stage: OK (%d formats, every element bit-equal to dequant_row)\n", checked);
    return 0;
}
