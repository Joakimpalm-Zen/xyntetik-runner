// moe-limb (R3.81): a gemma-4 Q4_0 expert's product is the same bits at any
// width, in any prompt, as the decode computes it, on the real file's tensors.
//
//     ./test-cuda-moe-limb MODEL.gguf [LAYERS]      LAYERS: all (default) or 0,15,29
//
// For each layer, both expert products (gate_up, down) of four experts:
// - inputs: 64 columns x_j = w_j z_j (w the layer's pre_ffw_norm_2 weight, z a
//   fixed normal draw: the shape rmsnorm(h) * w gives the expert input); the
//   down inputs are GELU(gate) * up of today's gate_up outputs on them; and 12
//   special columns (zeros, denormals, a block of tiny values, NaN, +-inf, a
//   block max above 2^64, at 2^64 and at 2^-64, FLT_MAX, negative zeros).
// - the subject, moe-limb: each column alone through the decode's expert
//   matvec is the reference; the by-expert prefill's product over subsets of
//   widths 1 to 76, in rotating orders, must give every column those bits (a
//   NaN only as a NaN). Every special column must have today's result class
//   (NaN, +inf, -inf, zero, finite) in every output, and a flagged column
//   (NaN, inf, a block max outside [2^-64, 2^64]) today's bits.
// - the control, today's route: the twins equal the decode in the same way.
// - the negative control, the grouped prefill (enc_moe_gemm): it must differ
//   somewhere, or the gate could not see a route that is not an identity.
// - the selection (R3.83, the route is the default): the load takes moe-limb
//   unless the environment holds it off (RUNNER_MOE_LIMB=0, or a
//   RUNNER_CUDA_ROUTE without moe-limb such as `reference`), and the
//   instance's own route (the hook's route 3) gives that route's bits, decode
//   and prefill, on the first layer's first expert. Held off, the instance
//   has no moe-limb at all (route 2 is refused) and the subject's checks,
//   which need the route, are not run; the controls still are.
// Skips (prints "ok (skipped)", exits 0) in a non-CUDA build or without a CUDA
// device that took the file.
#include "runner.h"
#include "compat.h"
#include "gpu.h"
#include "gate_gpu.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NCOL = 64, NSPEC = 12, NALL = NCOL + NSPEC, NEXP = 4 };
static int g_fail = 0;
static void ck(int cond, const char *what) {
    printf("%s: %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) g_fail = 1;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static double urand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((rng >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static float nrand(void) {
    double u = urand(), v = urand();
    return (float)(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v));
}

// Special values are made, and read, through their bit patterns: this file
// builds with the default flags, whose -ffast-math lets a compiler fold
// isnan/isinf (and x != x) to false and refuse the NAN/INFINITY macros
// (Apple clang: -Wnan-infinity-disabled). class_self_test proves the reading.
static float fbits(uint32_t u) { float v; memcpy(&v, &u, 4); return v; }
static uint32_t ubits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }
// 0 NaN, 1 +inf, 2 -inf, 3 zero, 4 finite nonzero
static int fclass(float v) {
    uint32_t u = ubits(v), e = (u >> 23) & 0xffu, mant = u & 0x7fffffu;
    if (e == 0xffu) return mant ? 0 : (u >> 31) ? 2 : 1;
    return (u & 0x7fffffffu) == 0 ? 3 : 4;
}
static bool same_bits(float a, float b) {
    if (fclass(a) == 0 || fclass(b) == 0) return fclass(a) == 0 && fclass(b) == 0;
    return ubits(a) == ubits(b);
}
static bool class_self_test(void) {
    static const uint32_t u[] = { 0x7fc00000u, 0xffc00001u, 0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u,
                                  0x00000001u, 0x3f800000u, 0x7f7fffffu };
    static const int want[] = { 0, 0, 1, 2, 3, 3, 4, 4, 4 };
    for (int i = 0; i < 9; i++)
        if (fclass(fbits(u[i])) != want[i]) return false;
    return !same_bits(fbits(0x7fc00000u), fbits(0x3f800000u)) && same_bits(fbits(0x7fc00000u), fbits(0xffc00001u)) &&
           !same_bits(fbits(0x00000000u), fbits(0x80000000u));
}

// the 12 special columns from a base column of n elements (n / 32 blocks)
static void specials(const float *base, int n, float *out) {
    for (int s = 0; s < NSPEC; s++) {
        float *c = out + (size_t)s * n;
        memcpy(c, base, sizeof(float) * (size_t)n);
        int nb = n / 32, b = (3 + s % 5) % nb, j = b * 32 + 7;
        switch (s) {
        case 0: memset(c, 0, sizeof(float) * (size_t)n); break;                 // all zeros
        case 1: memset(c + b * 32, 0, sizeof(float) * 32); break;               // a zero block
        case 2: c[j] = fbits(0x000116c2u); c[j + 1] = fbits(0x80003cf6u); break;  // denormals (1e-40, -2.2e-41) in a normal block
        case 3: for (int k = 0; k < 32; k++) c[b * 32 + k] = 1e-30f * (float)(k - 15); break;  // max < 2^-64
        case 4: c[j] = fbits(0x7fc00000u); break;                               // NaN
        case 5: c[j] = fbits(0x7f800000u); break;                               // +inf
        case 6: c[j] = fbits(0xff800000u); break;                               // -inf
        case 7: c[j] = ldexpf(1.0f, 70); break;                                 // max > 2^64
        case 8: c[j] = ldexpf(1.0f, 64); break;                                 // max at 2^64
        case 9: memset(c + b * 32, 0, sizeof(float) * 32); c[j] = ldexpf(1.0f, -64); break;
        case 10: c[j] = FLT_MAX; break;
        case 11: for (int k = 0; k < n; k++) c[k] = fbits(0x80000000u); break;  // negative zeros
        }
    }
}
static bool flagged_case(int s) { return s == 3 || s == 4 || s == 5 || s == 6 || s == 7 || s == 10; }

static double gelu(double v) {
    return 0.5 * v * (1 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
}

// |limb - today| per output against the two bounds (R3.81 design, point 1):
// |limb - ref| <= 3uA + 12uS and |today - ref| <= 25uS, u = 2^-24, S = sum_b
// |d_b| sum_j |q_bj x_bj|, A = sum_b |d_b| mb_b sum_j |q_bj| (mb_b the block's
// max |x|). Host arithmetic in double from the Q4_0 slice. Returns the number
// of outputs over 3uA + 37uS; *worst gets the largest |limb - today| / that.
static long accuracy(const uint8_t *w, int n_in, int n_out, const float *x, const float *lim,
                     const float *tod, double *worst) {
    int nb = n_in / 32;
    double mb[512];
    for (int b = 0; b < nb && b < 512; b++) {
        double m = 0;
        for (int j = 0; j < 32; j++) m = fmax(m, fabs((double)x[b * 32 + j]));
        mb[b] = m;
    }
    const double u = ldexp(1.0, -24);
    long over = 0;
    for (int r = 0; r < n_out; r++) {
        const uint8_t *row = w + (size_t)r * nb * 18;
        double S = 0, A = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + (size_t)b * 18;
            uint16_t h;
            memcpy(&h, blk, 2);
            // f16 to double
            int ex = (h >> 10) & 31, mant = h & 1023;
            double d = ex ? ldexp(1.0 + mant / 1024.0, ex - 15) : ldexp(mant / 1024.0, -14);
            double sq = 0, sqx = 0;
            for (int j = 0; j < 16; j++) {
                int lo = (blk[2 + j] & 15) - 8, hi = (blk[2 + j] >> 4) - 8;
                sq += abs(lo) + abs(hi);
                sqx += abs(lo) * fabs((double)x[b * 32 + j]) + abs(hi) * fabs((double)x[b * 32 + 16 + j]);
            }
            S += d * sqx;
            A += d * mb[b] * sq;
        }
        double bound = 3 * u * A + 37 * u * S, e = fabs((double)lim[r] - (double)tod[r]);
        if (e > bound) over++;
        if (bound > 0 && e / bound > *worst) *worst = e / bound;
    }
    return over;
}

static const int WIDTHS[] = { 1, 2, 3, 5, 8, 9, 16, 17, 31, 32, 33, 64, NALL };
enum { NW = sizeof WIDTHS / sizeof WIDTHS[0] };

// route's prefill over every width against dec (each column's decode bits);
// returns the number of differing columns
static long widths_check(model_t *m, int l, int proj, int e, int route, const float *x, int n_in,
                         int n_out, const float *dec, float *xs, float *ys, int *bad_calls) {
    long diff = 0;
    for (int wi = 0; wi < NW; wi++) {
        int w = WIDTHS[wi], idx[NALL];
        for (int k = 0; k < w; k++) idx[k] = (k * 29 + w * 7 + l) % NALL;   // a rotating order
        if (w == NALL) for (int k = 0; k < w; k++) idx[k] = (k * 29) % NALL;  // a permutation (29 odd vs 76)
        for (int k = 0; k < w; k++) memcpy(xs + (size_t)k * n_in, x + (size_t)idx[k] * n_in, sizeof(float) * (size_t)n_in);
        if (!gpu_test_moe_q4_0(m, l, proj, e, route, 1, xs, w, ys)) { (*bad_calls)++; continue; }
        for (int k = 0; k < w; k++) {
            const float *a = ys + (size_t)k * n_out, *b = dec + (size_t)idx[k] * n_out;
            for (int r = 0; r < n_out; r++)
                if (!same_bits(a[r], b[r])) { diff++; break; }
        }
    }
    return diff;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL.gguf [all|0,15,29]\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
#ifndef RUNNER_GPU_CUDA
    // the route and its test hook are CUDA's; a Metal build's device would
    // take the file and refuse every call
    fprintf(stderr, "test-cuda-moe-limb: skipped (not a CUDA build)\n");
    printf("test-cuda-moe-limb: ok (skipped)\n");
    return 0;
#endif
    ck(class_self_test(), "the result classes are read from the bits (NaN, +-inf, zeros, denormals, finite)");
    // what the environment asks of the load (R3.83): a route pin first, then
    // the switch, then the default (on)
    const char *sw = getenv("RUNNER_MOE_LIMB"), *rt = getenv("RUNNER_CUDA_ROUTE");
    int rmask = rt && *rt ? gpu_route_parse(rt) : -1;
    bool want = rmask >= 0 ? (rmask & GR_MOE_LIMB) != 0
                           : !(sw && (!strcmp(sw, "0") || !strcmp(sw, "off")));
    model_t m;
    memset(&m, 0, sizeof m);
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_AUTO;
    p.n_ctx = 256;
    p.n_batch = 64;
    if (!model_load(&m, argv[1], &p)) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    if (!m.gpu) {
        if (gate_no_device("test-cuda-moe-limb")) return 1;
        printf("test-cuda-moe-limb: skipped (no CUDA device took %s)\n", argv[1]);
        model_free(&m);
        return 0;
    }
    if (!m.moe_gemma) { fprintf(stderr, "%s is not a gemma-4 MoE file\n", argv[1]); return 1; }
    char what[256], gr[96];
    gpu_route_format(gpu_route_active(&m), gr, sizeof gr);
    bool took = (gpu_route_active(&m) & GR_MOE_LIMB) != 0;
    snprintf(what, sizeof what, "the load chose %s (gpu_route %s; RUNNER_MOE_LIMB %s, RUNNER_CUDA_ROUTE %s)",
             took ? "moe-limb" : "today's expert route", gr, sw ? sw : "unset", rt ? rt : "unset");
    ck(took == want, what);
    int layers[256], nl = 0;
    if (argc < 3 || !strcmp(argv[2], "all")) for (int l = 0; l < m.n_layer && nl < 256; l++) layers[nl++] = l;
    else for (char *s = argv[2]; *s && nl < 256; ) { layers[nl++] = atoi(s); s = strchr(s, ','); if (!s) break; s++; }
    int n_embd = m.n_embd, nff = m.n_ff_exp, mx = n_embd > 2 * nff ? n_embd : 2 * nff;
    float *xg = malloc(sizeof(float) * (size_t)NALL * n_embd), *xd = malloc(sizeof(float) * (size_t)NALL * nff);
    float *dec = malloc(sizeof(float) * (size_t)NALL * mx), *tod = malloc(sizeof(float) * (size_t)NALL * mx);
    float *xs = malloc(sizeof(float) * (size_t)NALL * mx), *ys = malloc(sizeof(float) * (size_t)NALL * mx);
    float *gu0 = malloc(sizeof(float) * (size_t)NCOL * 2 * nff);
    float *own = malloc(sizeof(float) * (size_t)NALL * mx);
    if (!xg || !xd || !dec || !tod || !xs || !ys || !gu0 || !own) return 2;
    long subj = 0, ctrl = 0, grp = 0, cls = 0, flg = 0, bad_flag_rows = 0, moved = 0, plain = 0, acc = 0;
    long own_diff = 0, own_cols = 0;
    double worst = 0;
    int bad_calls = 0, cases = 0, limb_ran = 0, limb_asked = 0;
    for (int li = 0; li < nl; li++) {
        int l = layers[li];
        const layer_t *ly = &m.layers[l];
        if (!ly->ffn_gate_up_exps || ly->ffn_gate_up_exps->type != T_Q4_0 || !ly->ffn_pre_norm2_w) {
            fprintf(stderr, "layer %d: no Q4_0 gate_up experts\n", l);
            g_fail = 1;
            continue;
        }
        int exps[NEXP] = { 0, ly->n_expert / 2 - 1, ly->n_expert - 1, (l * 37 + 11) % ly->n_expert };
        for (int c = 0; c < NCOL; c++)
            for (int j = 0; j < n_embd; j++) xg[(size_t)c * n_embd + j] = ly->ffn_pre_norm2_w[j] * nrand();
        specials(xg, n_embd, xg + (size_t)NCOL * n_embd);
        for (int ei = 0; ei < NEXP; ei++) {
            int e = exps[ei];
            // the down inputs from today's gate_up outputs (the decode's route)
            if (!gpu_test_moe_q4_0(&m, l, 0, e, 0, 0, xg, NCOL, gu0)) { bad_calls++; continue; }
            for (int c = 0; c < NCOL; c++)
                for (int j = 0; j < nff; j++)
                    xd[(size_t)c * nff + j] = (float)(gelu(gu0[(size_t)c * 2 * nff + j]) *
                                                      gu0[(size_t)c * 2 * nff + nff + j]);
            specials(xd, nff, xd + (size_t)NCOL * nff);
            for (int proj = 0; proj < 2; proj++) {
                const float *x = proj ? xd : xg;
                int n_in = proj ? nff : n_embd, n_out = proj ? n_embd : 2 * nff;
                cases++;
                // today: the decode, then its twins
                if (!gpu_test_moe_q4_0(&m, l, proj, e, 0, 0, x, NALL, tod)) { bad_calls++; continue; }
                ctrl += widths_check(&m, l, proj, e, 0, x, n_in, n_out, tod, xs, ys, &bad_calls);
                grp += widths_check(&m, l, proj, e, 1, x, n_in, n_out, tod, xs, ys, &bad_calls);
                if (!took) {
                    // held off: the instance's own route is today's, and it has no moe-limb
                    if (li == 0 && ei == 0)
                        for (int pre = 0; pre < 2; pre++) {
                            if (!gpu_test_moe_q4_0(&m, l, proj, e, 3, pre, x, NALL, own)) { bad_calls++; continue; }
                            for (int c = 0; c < NALL; c++, own_cols++)
                                for (int r = 0; r < n_out; r++)
                                    if (!same_bits(own[(size_t)c * n_out + r], tod[(size_t)c * n_out + r])) {
                                        own_diff++;
                                        break;
                                    }
                        }
                    limb_asked++;
                    limb_ran += gpu_test_moe_q4_0(&m, l, proj, e, 2, 0, x, 1, dec);
                    continue;
                }
                // the subject
                if (!gpu_test_moe_q4_0(&m, l, proj, e, 2, 0, x, NALL, dec)) { bad_calls++; continue; }
                // the instance's own route is moe-limb: its decode and its prefill are the subject's bits
                if (li == 0 && ei == 0)
                    for (int pre = 0; pre < 2; pre++) {
                        if (!gpu_test_moe_q4_0(&m, l, proj, e, 3, pre, x, NALL, own)) { bad_calls++; continue; }
                        for (int c = 0; c < NALL; c++, own_cols++)
                            for (int r = 0; r < n_out; r++)
                                if (!same_bits(own[(size_t)c * n_out + r], dec[(size_t)c * n_out + r])) {
                                    own_diff++;
                                    break;
                                }
                    }
                subj += widths_check(&m, l, proj, e, 2, x, n_in, n_out, dec, xs, ys, &bad_calls);
                // accuracy against today, on 8 ordinary columns
                {
                    const gguf_tensor *bt = proj ? ly->ffn_down_exps : ly->ffn_gate_up_exps;
                    size_t es = (size_t)n_out * (size_t)(n_in / 32) * 18;
                    const uint8_t *wexp = (const uint8_t *)bt->data + (size_t)e * es;
                    for (int c = 0; c < 8; c++)
                        acc += accuracy(wexp, n_in, n_out, x + (size_t)c * n_in, dec + (size_t)c * n_out,
                                        tod + (size_t)c * n_out, &worst);
                }
                // the route engaged: its decode is not today's on the ordinary columns
                for (int c = 0; c < NCOL; c++) {
                    plain++;
                    for (int r = 0; r < n_out; r++)
                        if (!same_bits(dec[(size_t)c * n_out + r], tod[(size_t)c * n_out + r])) { moved++; break; }
                }
                for (int s = 0; s < NSPEC; s++) {
                    const float *a = dec + (size_t)(NCOL + s) * n_out, *b = tod + (size_t)(NCOL + s) * n_out;
                    long cdiff = 0, fdiff = 0;
                    for (int r = 0; r < n_out; r++) {
                        cdiff += fclass(a[r]) != fclass(b[r]);
                        fdiff += flagged_case(s) && !same_bits(a[r], b[r]);
                    }
                    cls += cdiff > 0;
                    flg += fdiff > 0;
                    bad_flag_rows += fdiff;
                    if (cdiff || fdiff)
                        fprintf(stderr, "layer %d expert %d %s special %d: %ld outputs of another class, "
                                "%ld flagged outputs not today's bits\n", l, e, proj ? "down" : "gate_up",
                                s, cdiff, fdiff);
                }
            }
        }
        fprintf(stderr, "layer %d: running totals subject %ld, control %ld, grouped %ld, class %ld, "
                "flagged %ld, failed calls %d\n", l, subj, ctrl, grp, cls, flg, bad_calls);
    }
    snprintf(what, sizeof what, "%d layer(s), %d (expert, product) cases, widths 1 to %d: every call ran",
             nl, cases, NALL);
    ck(bad_calls == 0 && cases > 0, what);
    ck(ctrl == 0, "control: today's twins give every column the decode's bits at every width");
    snprintf(what, sizeof what, "negative control: the grouped prefill differs from the decode (%ld columns)", grp);
    ck(grp > 0, what);
    snprintf(what, sizeof what, "the instance's own route gives %s's bits, decode and prefill (%ld of %ld columns "
             "differ)", took ? "moe-limb" : "today's route", own_diff, own_cols);
    ck(own_diff == 0 && own_cols > 0 && bad_calls == 0, what);
    if (!took) {
        snprintf(what, sizeof what, "held off, the instance has no moe-limb: %d of %d calls of it ran (the "
                 "subject's checks need the route and are not run)", limb_ran, limb_asked);
        ck(limb_ran == 0 && limb_asked > 0, what);
        free(xg); free(xd); free(dec); free(tod); free(xs); free(ys); free(gu0); free(own);
        model_free(&m);
        printf(g_fail ? "test-cuda-moe-limb: FAILED\n" : "test-cuda-moe-limb: ok\n");
        return g_fail;
    }
    // the prefill's path: the int8 MMA tile where the compute_80 module loaded,
    // else the decode kernel a column at a time (RUNNER_CUDA_SM80=0 forces it)
    bool tile = gpu_moe_limb_has_tile(&m);
    snprintf(what, sizeof what, "moe-limb engaged: its decode differs from today's on %ld of %ld ordinary columns, "
             "and its prefill ran %s (%lu tiles)", moved, plain,
             tile ? "the int8 MMA tile" : "the decode kernel a column at a time (no compute_80 module)",
             gpu_moe_limb_tiles());
    ck(moved > 0 && (tile ? gpu_moe_limb_tiles() > 0 : gpu_moe_limb_tiles() == 0), what);
    snprintf(what, sizeof what, "moe-limb: every output of 8 ordinary columns a case is within 3uA + 37uS of "
             "today's (%ld are not; the largest is %.4f of its bound)", acc, worst);
    ck(acc == 0 && bad_calls == 0, what);
    snprintf(what, sizeof what, "moe-limb: every prefill column is the decode's bits at every width and order "
             "(%ld columns differ)", subj);
    ck(subj == 0 && bad_calls == 0, what);
    snprintf(what, sizeof what, "moe-limb: every special column has today's result class in every output "
             "(%ld columns do not)", cls);
    ck(cls == 0 && bad_calls == 0, what);
    snprintf(what, sizeof what, "moe-limb: every flagged column is today's bits (%ld columns, %ld outputs, are not)",
             flg, bad_flag_rows);
    ck(flg == 0 && bad_calls == 0, what);
    free(xg); free(xd); free(dec); free(tod); free(xs); free(ys); free(gu0); free(own);
    model_free(&m);
    printf(g_fail ? "test-cuda-moe-limb: FAILED\n" : "test-cuda-moe-limb: ok\n");
    return g_fail;
}
