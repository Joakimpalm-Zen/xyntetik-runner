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
#include <string.h>

#include "iq_decode.h"
#include "quants_iq_grids.h"

static int g_fail = 0;
#define CK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); g_fail = 1; } } while (0)

int main(void) {
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

    printf(g_fail ? "iq-decode: FAILED\n"
                  : "iq-decode: ok (128 sign expansions, 2048+2048 grid signs, "
                    "4096 IQ1 values, field reads)\n");
    return g_fail;
}
