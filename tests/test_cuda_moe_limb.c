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
// Skips (prints so) without a CUDA device that took the file.
#include "runner.h"
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

// 0 NaN, 1 +inf, 2 -inf, 3 zero, 4 finite nonzero
static int fclass(float v) {
    return isnan(v) ? 0 : isinf(v) ? (v > 0 ? 1 : 2) : v == 0.0f ? 3 : 4;
}
static bool same_bits(float a, float b) {
    if (isnan(a) || isnan(b)) return isnan(a) && isnan(b);
    uint32_t x, y;
    memcpy(&x, &a, 4); memcpy(&y, &b, 4);
    return x == y;
}

// the 12 special columns from a base column of n elements (n / 32 blocks)
static void specials(const float *base, int n, float *out) {
    for (int s = 0; s < NSPEC; s++) {
        float *c = out + (size_t)s * n;
        memcpy(c, base, sizeof(float) * (size_t)n);
        int b = 3 + s % 5, j = b * 32 + 7;
        switch (s) {
        case 0: memset(c, 0, sizeof(float) * (size_t)n); break;                 // all zeros
        case 1: memset(c + b * 32, 0, sizeof(float) * 32); break;               // a zero block
        case 2: c[j] = 1e-40f; c[j + 1] = -2e-41f; break;                       // denormals in a normal block
        case 3: for (int k = 0; k < 32; k++) c[b * 32 + k] = 1e-30f * (float)(k - 15); break;  // max < 2^-64
        case 4: c[j] = NAN; break;
        case 5: c[j] = INFINITY; break;
        case 6: c[j] = -INFINITY; break;
        case 7: c[j] = ldexpf(1.0f, 70); break;                                 // max > 2^64
        case 8: c[j] = ldexpf(1.0f, 64); break;                                 // max at 2^64
        case 9: memset(c + b * 32, 0, sizeof(float) * 32); c[j] = ldexpf(1.0f, -64); break;
        case 10: c[j] = FLT_MAX; break;
        case 11: for (int k = 0; k < n; k++) c[k] = -0.0f; break;               // negative zeros
        }
    }
}
static bool flagged_case(int s) { return s == 3 || s == 4 || s == 5 || s == 6 || s == 7 || s == 10; }

static double gelu(double v) {
    return 0.5 * v * (1 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
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
    setenv("RUNNER_MOE_LIMB", "1", 1);   // the route's scratch and kernels; each call names its route
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
    int layers[256], nl = 0;
    if (argc < 3 || !strcmp(argv[2], "all")) for (int l = 0; l < m.n_layer && nl < 256; l++) layers[nl++] = l;
    else for (char *s = argv[2]; *s && nl < 256; ) { layers[nl++] = atoi(s); s = strchr(s, ','); if (!s) break; s++; }
    int n_embd = m.n_embd, nff = m.n_ff_exp, mx = n_embd > 2 * nff ? n_embd : 2 * nff;
    float *xg = malloc(sizeof(float) * (size_t)NALL * n_embd), *xd = malloc(sizeof(float) * (size_t)NALL * nff);
    float *dec = malloc(sizeof(float) * (size_t)NALL * mx), *tod = malloc(sizeof(float) * (size_t)NALL * mx);
    float *xs = malloc(sizeof(float) * (size_t)NALL * mx), *ys = malloc(sizeof(float) * (size_t)NALL * mx);
    float *gu0 = malloc(sizeof(float) * (size_t)NCOL * 2 * nff);
    if (!xg || !xd || !dec || !tod || !xs || !ys || !gu0) return 2;
    long subj = 0, ctrl = 0, grp = 0, cls = 0, flg = 0, bad_flag_rows = 0;
    int bad_calls = 0, cases = 0;
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
                // the subject
                if (!gpu_test_moe_q4_0(&m, l, proj, e, 2, 0, x, NALL, dec)) { bad_calls++; continue; }
                subj += widths_check(&m, l, proj, e, 2, x, n_in, n_out, dec, xs, ys, &bad_calls);
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
    char what[256];
    snprintf(what, sizeof what, "%d layer(s), %d (expert, product) cases, widths 1 to %d: every call ran",
             nl, cases, NALL);
    ck(bad_calls == 0 && cases > 0, what);
    ck(ctrl == 0, "control: today's twins give every column the decode's bits at every width");
    snprintf(what, sizeof what, "negative control: the grouped prefill differs from the decode (%ld columns)", grp);
    ck(grp > 0, what);
    snprintf(what, sizeof what, "moe-limb: every prefill column is the decode's bits at every width and order "
             "(%ld columns differ)", subj);
    ck(subj == 0 && bad_calls == 0, what);
    snprintf(what, sizeof what, "moe-limb: every special column has today's result class in every output "
             "(%ld columns do not)", cls);
    ck(cls == 0 && bad_calls == 0, what);
    snprintf(what, sizeof what, "moe-limb: every flagged column is today's bits (%ld columns, %ld outputs, are not)",
             flg, bad_flag_rows);
    ck(flg == 0 && bad_calls == 0, what);
    free(xg); free(xd); free(dec); free(tod); free(xs); free(ys); free(gu0);
    model_free(&m);
    printf(g_fail ? "test-cuda-moe-limb: FAILED\n" : "test-cuda-moe-limb: ok\n");
    return g_fail;
}
