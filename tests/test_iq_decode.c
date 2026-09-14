// The CUDA codebook decode primitives, executed on the host.
//
// src/iq_decode.h is the arithmetic the device runs for the sign expansion
// and the grid-byte magnitudes of the seven codebook i-quant kernels. nvcc
// compiles it into kernels.cu; this test compiles the same file with the C
// compiler and holds every function to the host tables in
// quants_iq_grids.h, exhaustively over its inputs. Until 2026-09-14 the
// only test in this area compared the host table to a Python formula and
// never touched the device code: the v0.5.3 review changed the helper's
// parity term to a constant and all tests stayed green.
//
//     ./test-iq-decode
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iq_decode.h"
#include "quants.h"
#include "fp16.h"
#include "quants_iq_grids.h"

static int g_fail = 0;
#define CK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static uint32_t g_rng = 0x9e3779b9u;
static unsigned rnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

// A random half that is finite and not tiny: the block scale of a real
// file. Subnormal and zero scales are covered by the explicit cases.
static void put_f16(iq_byte *p, unsigned bits) { p[0] = bits & 0xFF; p[1] = bits >> 8; }
static unsigned rnd_f16_scale(void) {
    unsigned e = 1 + rnd() % 29;              // exponents 1..29: 2^-14 .. 2^14
    return (rnd() & 0x8000u) | (e << 10) | (rnd() & 0x3ffu);
}

static int f32_bits_equal(float a, float b) {
    uint32_t x, y;
    memcpy(&x, &a, 4); memcpy(&y, &b, 4);
    return x == y;
}

// The tensor-core staging decoder of one 256-weight block, all four
// segments, against the host decoder the CUDA kernels are twins of.
// Bitwise: both compute d * scale * (+/- magnitude) in f32 in the same order.
typedef void (*stage64_fn)(float *dst, const iq_byte *blk, int seg);
static int stage_block_matches(int type, const iq_byte *blk, stage64_fn stage,
                               const char *what) {
    float want[256], got[256];
    dequant_row(type, blk, want, 256);
    for (int seg = 0; seg < 4; seg++) stage(got + 64 * seg, blk, seg);
    for (int i = 0; i < 256; i++)
        if (!f32_bits_equal(want[i], got[i])) {
            fprintf(stderr, "FAIL: %s element %d: staged %.9g, dequant_row %.9g\n",
                    what, i, (double)got[i], (double)want[i]);
            g_fail = 1;
            return 0;
        }
    return 1;
}

static void stage_iq3s(float *dst, const iq_byte *blk, int seg) {
    iq3s_stage64(dst, blk, seg, iq3s_grid);
}

// IQ3_S hand-built block: every field set to a value whose decoded weight
// is computed here from the format's definition, not from any decoder.
//   d = 2.0 (0x4000); scales[0] = 0x21 (sub-block 0 scale 1 -> 3, sub-block
//   1 scale 2 -> 5); scales[1..3] = 0x00 (scale 1 each); every index byte
//   0, qh[0] bit 0 set (index 0 of sub-block 0 becomes 256 -> grid entry
//   256), signs[0] = 0x81 (weight 0 and weight 7 of the first octet
//   negative), all other signs 0.
//   grid[0] = 0x01010101 (magnitudes 1,1,1,1), grid[256] is the table's
//   own entry: its byte j is the magnitude of weight j.
static void iq3s_known_block(void) {
    iq_byte blk[110];
    memset(blk, 0, sizeof blk);
    put_f16(blk, 0x4000);              // d = 2.0
    blk[106] = 0x21;                   // scales: sub-block 0 -> 1+2*1, sub-block 1 -> 1+2*2
    blk[66] = 0x01;                    // qh[0] bit 0: index 0 of sub-block 0 gets +256
    blk[74] = 0x81;                    // signs[0]: weights 0 and 7 of octet 0 negative
    float got[256];
    for (int seg = 0; seg < 4; seg++) stage_iq3s(got + 64 * seg, blk, seg);
    unsigned g256 = iq3s_grid[256], g0 = iq3s_grid[0];
    CK(g0 == 0x01010101u, "iq3s_grid[0] is the all-ones entry (%08x)", g0);
    // octet 0 of sub-block 0: first four from grid[256] (index 0 | 256),
    // scale 3, d 2 -> 6 * mag; weight 0 negative
    for (int j = 0; j < 4; j++) {
        float mag = (float)((g256 >> (8 * j)) & 0xFF);
        float want = 6.0f * (j == 0 ? -mag : mag);
        CK(got[j] == want, "IQ3_S known block: weight %d = %g, want %g", j, (double)got[j], (double)want);
    }
    // next four from grid[0] (index 0, qh bit 1 clear): 6 * 1, weight 7 negative
    for (int j = 4; j < 8; j++) {
        float want = j == 7 ? -6.0f : 6.0f;
        CK(got[j] == want, "IQ3_S known block: weight %d = %g, want %g", j, (double)got[j], (double)want);
    }
    // the rest of sub-block 0: 6 * 1, positive
    for (int j = 8; j < 32; j++) CK(got[j] == 6.0f, "IQ3_S known block: weight %d = %g", j, (double)got[j]);
    // sub-block 1: scale 5 -> 10 * 1
    for (int j = 32; j < 64; j++) CK(got[j] == 10.0f, "IQ3_S known block: weight %d = %g", j, (double)got[j]);
    // sub-blocks 2..7: scale 1 -> 2 * 1
    for (int j = 64; j < 256; j++) CK(got[j] == 2.0f, "IQ3_S known block: weight %d = %g", j, (double)got[j]);
    // and the same block through dequant_row, bit for bit
    stage_block_matches(T_IQ3_S, blk, stage_iq3s, "IQ3_S known block");
}

static void iq3s_random_blocks(void) {
    iq_byte blk[110];
    for (int n = 0; n < 500; n++) {
        for (size_t i = 0; i < sizeof blk; i++) blk[i] = (iq_byte)rnd();
        put_f16(blk, rnd_f16_scale());
        if (!stage_block_matches(T_IQ3_S, blk, stage_iq3s, "IQ3_S random block")) return;
    }
    // the scale corners: zero, the smallest subnormal, the largest half
    static const unsigned corners[] = { 0x0000, 0x0001, 0x7bff, 0x8001, 0xfbff };
    for (size_t c = 0; c < sizeof corners / sizeof *corners; c++) {
        for (size_t i = 0; i < sizeof blk; i++) blk[i] = (iq_byte)rnd();
        put_f16(blk, corners[c]);
        if (!stage_block_matches(T_IQ3_S, blk, stage_iq3s, "IQ3_S scale corner")) return;
    }
}

int main(void) {
    f16_init();
    // 1. the 7-bit sign index expands to the table entry, all 128
    for (unsigned i = 0; i < 128; i++)
        CK(iq_signs7(i) == ksigns_iq2xs[i],
           "iq_signs7(%u) = %u, ksigns_iq2xs = %u", i, iq_signs7(i), ksigns_iq2xs[i]);
    // and the expansion is closed: bit 7 is the only bit it may add
    for (unsigned i = 0; i < 128; i++)
        CK((iq_signs7(i) & 0x7F) == i, "iq_signs7(%u) changed the low bits", i);

    // 2. an 8-magnitude grid entry: byte j is weight j, sign bit j (the host
    //    decoders test signs & kmask_iq2xs[j]) negates it. Every sign byte.
    const iq_u64 g8 = 0x0807060504030201ull;
    for (unsigned signs = 0; signs < 256; signs++)
        for (int j = 0; j < 8; j++) {
            float mag = (float)(j + 1);
            float want = (signs & kmask_iq2xs[j]) ? -mag : mag;
            CK(iq_w8(g8, j, signs) == want, "iq_w8 signs %u j %d: %g, want %g",
               signs, j, (double)iq_w8(g8, j, signs), (double)want);
        }
    // the 0 and 255 magnitudes survive the byte extraction
    CK(iq_w8(0x00ull, 3, 0) == 0.0f, "iq_w8 zero magnitude");
    CK(iq_w8(0xFF00000000000000ull, 7, 0) == 255.0f, "iq_w8 byte 7");
    CK(iq_w8(0xFF00000000000000ull, 7, 0x80) == -255.0f, "iq_w8 byte 7 negated");

    // 3. a 4-magnitude entry, and the kernels' second half convention: the
    //    upper four sign bits reach iq_w4 as (signs >> 4)
    const unsigned g4 = 0x04030201u;
    for (unsigned signs = 0; signs < 256; signs++)
        for (int j = 0; j < 4; j++) {
            float mag = (float)(j + 1);
            float want_lo = (signs & kmask_iq2xs[j]) ? -mag : mag;
            float want_hi = (signs & kmask_iq2xs[j + 4]) ? -mag : mag;
            CK(iq_w4(g4, j, signs) == want_lo, "iq_w4 low signs %u j %d", signs, j);
            CK(iq_w4(g4, j, signs >> 4) == want_hi, "iq_w4 high signs %u j %d", signs, j);
        }

    // 4. IQ1 grid bytes are signed and carry the delta: every byte value,
    //    both delta signs, every byte position
    for (unsigned b = 0; b < 256; b++)
        for (int j = 0; j < 8; j++) {
            iq_u64 g = (iq_u64)b << (8 * j);
            float sb = (float)(int8_t)b;
            CK(iq1_w(g, j, IQ1_DELTA) == sb + 0.125f, "iq1_w byte %u j %d +delta", b, j);
            CK(iq1_w(g, j, -IQ1_DELTA) == sb - 0.125f, "iq1_w byte %u j %d -delta", b, j);
        }
    // the grid values the format actually stores
    CK(iq1_w(0xffull, 0, IQ1_DELTA) == -0.875f, "iq1_w -1 + delta");
    CK(iq1_w(0x01ull, 0, -IQ1_DELTA) == 0.875f, "iq1_w 1 - delta");

    // 5. the 2-byte aligned field reads are little-endian, matching the
    //    host decoders' memcpy of the same bytes
    iq_byte buf[6] = { 0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a };
    CK(iq_ld16(buf) == 0x1234u, "iq_ld16");
    CK(iq_ld16(buf + 2) == 0x5678u, "iq_ld16 offset 2");
    CK(iq_ld32a2(buf) == 0x56781234u, "iq_ld32a2");
    CK(iq_ld32a2(buf + 2) == 0x9abc5678u, "iq_ld32a2 offset 2");

    // 6. the half conversion the host stager reads block scales with, against
    //    the engine's own table, every one of the 65536 bit patterns
    for (unsigned h = 0; h < 0x10000; h++) {
        iq_byte p[2] = { (iq_byte)(h & 0xFF), (iq_byte)(h >> 8) };
        float a = iq_f16(p), b = f16_to_f32((uint16_t)h);
        if (!f32_bits_equal(a, b) && !(a != a && b != b)) {
            CK(0, "iq_f16(%04x) = %.9g, f16_to_f32 = %.9g", h, (double)a, (double)b);
            break;
        }
    }

    // 7. the tensor-core staging decoders, block by block
    iq3s_known_block();
    iq3s_random_blocks();

    printf(g_fail ? "iq-decode: FAILED\n"
                  : "iq-decode: ok (128 sign expansions, 2048+2048 grid signs, "
                    "4096 IQ1 values, field reads, 65536 halves, IQ3_S staging: "
                    "1 known + 500 random + 5 corner blocks bitwise)\n");
    return g_fail;
}
