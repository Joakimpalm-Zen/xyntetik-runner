// R6.86: src/pmath.h's p_expf, written with no branch and no loop so that a
// compiler can vectorize the loops that call it, gives the bits of the p_expf
// it replaced for every float input.
//
// The oracle is that p_expf word for word, (float)pm_exp_d((double)x):
// pm_exp_d is unchanged and is still the exp of tanhf and powf.
//
//   ./test-pmath-expf         the named inputs below, then every 256th bit
//                             pattern (2^24 inputs: every sign, exponent and
//                             high mantissa, the low byte varied), then the
//                             mutant. make test runs this.
//   ./test-pmath-expf --all   all 2^32 bit patterns: make test-pmath-expf-all
//                             (strict flags) and make test-pmath-expf-all-default
//                             (the engine's own flags)
//
// The comparison is of the bits: a NaN must come back with the oracle's sign
// and payload, a zero with its sign. Each input goes through p_expf twice:
//   - in a buffer loop, the shape a compiler vectorizes, as the softmax's is;
//   - in a scalar call.
// Under the engine's default flags (-ffast-math, so __FAST_MATH__) the
// compiler may assume that no NaN exists, and both functions' NaN handling is
// undefined there. NaN inputs are then counted apart and reported, not failed,
// and every other input must match.
//
// The mutant is a live control for the sample. copy_expf is p_expf's body
// with ln2_lo, the low half of the Cody-Waite split, as a parameter:
//   - with the right value it must equal p_expf on the whole sample (so the copy
//     is p_expf; edit both together);
//   - with ln2_lo = 0, one wrong constant whose results are off in their last bit
//     only, it must differ from the oracle somewhere in the sample.
// So the sample sees an error that test_pmath.c's 1-ulp accuracy sweeps would
// pass. Strict build only: under fast-math a constant passed at run time and
// a literal one may fold differently.
#include "pmath.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;

// the p_expf this change replaced, as it stood: the oracle
static float oracle(float x) { return (float)pm_exp_d((double)x); }

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float flt(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static int nan_bits(uint32_t u) { return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0; }

enum { CHUNK = 4096 };
static float g_in[CHUNK], g_new[CHUNK], g_old[CHUNK], g_one[CHUNK];

// p_expf over a buffer, the loop shape the engine's softmax and SiLU have
static __attribute__((noinline)) void run_new(int n) {
    for (int i = 0; i < n; i++) g_new[i] = p_expf(g_in[i]);
}
static __attribute__((noinline)) void run_old(int n) {
    for (int i = 0; i < n; i++) g_old[i] = oracle(g_in[i]);
}
// p_expf alone, as a scalar call site compiles it
static __attribute__((noinline)) float one_new(float x) { return p_expf(x); }

typedef struct {
    uint64_t n, diff, nan_n, nan_diff;
    uint32_t first, first_nan;   // the first input that differed (not a NaN, a NaN)
} tally;

static void compare(tally *t, int n) {
    run_new(n);
    run_old(n);
    for (int i = 0; i < n; i++) g_one[i] = one_new(g_in[i]);
    for (int i = 0; i < n; i++) {
        uint32_t x = bits(g_in[i]), o = bits(g_old[i]);
        int differs = bits(g_new[i]) != o || bits(g_one[i]) != o;
        int nan = nan_bits(x);
        t->n++;
        t->nan_n += (uint64_t)nan;
        if (!differs) continue;
        if (nan) { if (!t->nan_diff) t->first_nan = x; t->nan_diff++; }
        else { if (!t->diff) t->first = x; t->diff++; }
    }
}

// verdict on a tally: every input that is not a NaN must match; NaNs too in a
// strict build. Prints one line.
static void judge(const char *what, const tally *t) {
#ifdef __FAST_MATH__
    int nan_fails = 0;
#else
    int nan_fails = t->nan_diff != 0;
#endif
    printf("pmath-expf: %s: %" PRIu64 " inputs, %" PRIu64 " differ; NaN inputs %" PRIu64 ", %" PRIu64 " differ%s\n",
           what, t->n, t->diff, t->nan_n, t->nan_diff,
#ifdef __FAST_MATH__
           t->nan_diff ? " (reported, not failed: -ffast-math assumes no NaN)" : ""
#else
           ""
#endif
    );
    if (t->diff) {
        uint32_t x = t->first;
        fprintf(stderr, "FAIL: %s: p_expf(%a, bits %08" PRIx32 ") = %08" PRIx32 " (buffer) and %08" PRIx32
                " (scalar), the oracle %08" PRIx32 "\n", what, (double)flt(x), x, bits(p_expf(flt(x))),
                bits(one_new(flt(x))), bits(oracle(flt(x))));
        g_fail = 1;
    }
    if (nan_fails) {
        uint32_t x = t->first_nan;
        fprintf(stderr, "FAIL: %s: the NaN %08" PRIx32 " gives %08" PRIx32 ", the oracle %08" PRIx32 "\n",
                what, x, bits(one_new(flt(x))), bits(oracle(flt(x))));
        g_fail = 1;
    }
}

// The named inputs: every class and every edge the change could move.
static const struct { uint32_t x; const char *why; } NAMED[] = {
    { 0x00000000u, "+0" }, { 0x80000000u, "-0" },
    { 0x7f800000u, "+inf" }, { 0xff800000u, "-inf" },
    { 0x7fc00000u, "the quiet NaN" }, { 0xffc00000u, "a negative quiet NaN" },
    { 0x7fc12345u, "a quiet NaN with a payload" }, { 0xffd2468au, "a negative quiet NaN with a payload" },
    { 0x7f800001u, "the smallest signaling NaN" }, { 0x7fa00000u, "a signaling NaN" },
    { 0xff800001u, "a negative signaling NaN" }, { 0x7fffffffu, "the largest NaN" }, { 0xffffffffu, "-NaN, all ones" },
    { 0x00000001u, "the smallest subnormal" }, { 0x80000001u, "its negative" },
    { 0x007fffffu, "the largest subnormal" }, { 0x807fffffu, "its negative" },
    { 0x00800000u, "the smallest normal" }, { 0x80800000u, "its negative" },
    { 0x7f7fffffu, "FLT_MAX" }, { 0xff7fffffu, "-FLT_MAX" },
    { 0x3f800000u, "1" }, { 0xbf800000u, "-1" }, { 0x34000000u, "FLT_EPSILON" },
    { 0x2edbe6ffu, "1e-10" }, { 0xaedbe6ffu, "-1e-10" },
    { 0x42b1ffffu, "under the clamp at 89" }, { 0x42b20000u, "89, the clamp" }, { 0x42b20001u, "over the clamp at 89" },
    { 0xc2cfffffu, "over the clamp at -104" }, { 0xc2d00000u, "-104, the clamp" }, { 0xc2d00001u, "under the clamp at -104" },
    { 0x42b17217u, "the largest x with a finite exp" }, { 0x42b17218u, "the smallest x whose exp is +inf" },
    { 0xc2cff1b4u, "the smallest x whose exp is not 0" }, { 0xc2cff1b5u, "the largest x whose exp is 0" },
    { 0xc2b40000u, "-90, a subnormal result" }, { 0xc2c80000u, "-100, a subnormal result" },
    { 0x44313fffu, "under pm_exp_d's guard at 709" }, { 0x44314000u, "709" }, { 0x44314001u, "over 709" },
    { 0xc43a3fffu, "over pm_exp_d's guard at -745" }, { 0xc43a4000u, "-745" }, { 0xc43a4001u, "under -745" },
    { 0x442d5f99u, "under k = 1001 (pm_ldexp2's first loop)" }, { 0x442d5f9au, "at k = 1001" },
    { 0xc42d5f99u, "over k = -1001 (its second loop)" }, { 0xc42d5f9au, "at k = -1001" },
    { 0x3eb17217u, "under ln2/2, where k turns 1" }, { 0x3eb17218u, "ln2/2" }, { 0x3eb17219u, "over ln2/2" },
    { 0xbeb17217u, "over -ln2/2" }, { 0xbeb17218u, "-ln2/2" }, { 0xbeb17219u, "under -ln2/2" },
    { 0x42b0c0a5u, "under 127.5 ln2, where k turns 128" }, { 0x42b0c0a6u, "127.5 ln2" }, { 0x42b0c0a7u, "over it" },
    { 0xc2cf4041u, "over -149.5 ln2, where k turns -150" }, { 0xc2cf4042u, "-149.5 ln2" }, { 0xc2cf4043u, "under it" },
};
enum { NNAMED = (int)(sizeof(NAMED) / sizeof(NAMED[0])) };

#ifndef __FAST_MATH__
// p_expf's body with ln2_lo as a parameter: the mutant's frame (see the top)
static __attribute__((noinline)) float copy_expf(float x, double ln2_lo) {
    const double ln2_hi = 6.93147180369123816490e-01;
    const double inv_ln2 = 1.44269504088896338700e+00;
    double xd = (double)x;
    double xc = xd < 89.0 ? xd : 89.0;
    xc = xc > -104.0 ? xc : -104.0;
    double kd = xc * inv_ln2;
    int k = (int)(kd + (kd >= 0 ? 0.5 : -0.5));
    double r = (xc - (double)k * ln2_hi) - (double)k * ln2_lo;
    double p = 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    union { double d; uint64_t u; } s;
    s.u = (uint64_t)(k + 1023) << 52;
    double e = p * s.d;
    return (float)(xd == xd ? e : xd);
}
#endif

// the sample's i-th input: bit pattern i * 256, its low byte varied
static uint32_t sample_bits(uint32_t i) { return (i << 8) | ((i * 151u) & 0xffu); }

int main(int argc, char **argv) {
    int all = argc > 1 && strcmp(argv[1], "--all") == 0;
#ifdef __FAST_MATH__
    printf("pmath-expf: built with -ffast-math (the engine's default flags), %s\n", __VERSION__);
#else
    printf("pmath-expf: built with strict float (the T3 build's flags), %s\n", __VERSION__);
#endif

    // the named inputs
    tally named = { 0 };
    for (int i = 0; i < NNAMED; i++) g_in[i] = flt(NAMED[i].x);
    compare(&named, NNAMED);
    for (int i = 0; i < NNAMED; i++) {
        float x = flt(NAMED[i].x);
        uint32_t o = bits(oracle(x)), a = bits(one_new(x));
#ifdef __FAST_MATH__
        if (nan_bits(NAMED[i].x)) continue;
#endif
        if (a != o) {
            fprintf(stderr, "FAIL: named %08" PRIx32 " (%s): p_expf %08" PRIx32 ", the oracle %08" PRIx32 "\n",
                    NAMED[i].x, NAMED[i].why, a, o);
            g_fail = 1;
        }
    }
    judge("named inputs", &named);
#ifndef __FAST_MATH__
    // the edges are what their names say (in the oracle, so a wrong name fails here, not silently).
    // Not under fast-math: gcc's -ffast-math also flushes subnormal results to zero.
    {
        float big = oracle(flt(0x42b17217u)), inf = oracle(flt(0x42b17218u));
        float tiny = oracle(flt(0xc2cff1b4u)), zero = oracle(flt(0xc2cff1b5u));
        if (!(big < flt(0x7f800000u)) || bits(inf) != 0x7f800000u || bits(tiny) != 0x00000001u || bits(zero) != 0)
            { fprintf(stderr, "FAIL: the overflow and underflow edges are not where the names say\n"); g_fail = 1; }
    }
#endif

    // the sample, or every float
    tally t = { 0 };
    uint64_t total = all ? (UINT64_C(1) << 32) : (UINT64_C(1) << 24);
    for (uint64_t base = 0; base < total; base += CHUNK) {
        for (int i = 0; i < CHUNK; i++) {
            uint32_t j = (uint32_t)(base + (uint64_t)i);
            g_in[i] = flt(all ? j : sample_bits(j));
        }
        compare(&t, CHUNK);
    }
    judge(all ? "every float (2^32 inputs)" : "the sample (every 256th bit pattern)", &t);

#ifndef __FAST_MATH__
    if (!all) {
        // the mutant, on the sample
        uint64_t copy_diff = 0, mutant_diff = 0, mutant_named = 0;
        for (uint32_t j = 0; j < (UINT32_C(1) << 24); j++) {
            float x = flt(sample_bits(j));
            copy_diff += bits(copy_expf(x, 1.90821492927058770002e-10)) != bits(p_expf(x));
            mutant_diff += bits(copy_expf(x, 0.0)) != bits(oracle(x));
        }
        for (int i = 0; i < NNAMED; i++)
            mutant_named += bits(copy_expf(flt(NAMED[i].x), 0.0)) != bits(oracle(flt(NAMED[i].x)));
        printf("pmath-expf: the copy of p_expf differs from it at %" PRIu64 " of the sample; the mutant (ln2_lo = 0) "
               "differs from the oracle at %" PRIu64 " of the sample and %" PRIu64 " of the named inputs\n",
               copy_diff, mutant_diff, mutant_named);
        if (copy_diff) { fprintf(stderr, "FAIL: copy_expf is not p_expf: edit both together\n"); g_fail = 1; }
        if (!mutant_diff) { fprintf(stderr, "FAIL: the sample does not tell the mutant from the oracle\n"); g_fail = 1; }
    }
#endif
    if (g_fail) { fprintf(stderr, "pmath-expf: FAILED\n"); return 1; }
    printf("pmath-expf: OK\n");
    return 0;
}
