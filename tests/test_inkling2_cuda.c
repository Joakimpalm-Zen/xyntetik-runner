// R4.48.9: Inkling 2 Mini on CUDA, G1 (the weights on the device) and G2 (the forward's pieces) on the tiny
// inkling2_loop files (tests/inkling2_tiny.py: the dense projections F32, BF16 and Q8_0; the experts NVFP4 with their
// per-expert .scale on both stacks). Needs the CUDA backend and a CUDA device; without either it says so and exits 77
// (skipped), never a pass. The backend is the build's (RUNNER_GPU_CUDA): a Metal build has a device but no CUDA route.
//
// TOLERANCES, fixed before the first run (the supervisor's G2 rule):
//   - rows: the CUDA rows within 1e-4 of the reference route's rows, relative to the largest reference logit (the
//     fast CPU route's T1 bound), and NOT byte-identical to them (a different order of sums);
//   - invariance: byte-identical at -b 64 and -b 8, and across two loads (one fixed order per op);
//   - finite: a value that is not finite (counted by its bits) fails (c), (d) and (e), whatever the distance says;
//   - control: the expert factor left out on the device (RUNNER_INKLING2_CUDA_TEST_NO_FACTOR=1, test-only) is at least
//     10 times further from the reference than the CUDA route.
//
// The reference here is the CPU's reference route (RUNNER_INKLING2_ROUTE=reference: the serial float32 engine).
//  G1 (a) the route: loaded with the GPU allowed, the instance runs on CUDA (model_inkling2_cuda true; GPU off: false);
//     (b) every matrix on the device (projections, head) and both expert stacks, read back from the device through the
//         test-only accessor model_inkling2_cuda_readback, equal the host decode (dequant_row; for the stacks the
//         stored rows, the factor separate) bit for bit;
//  G2 (c) rows near the reference and not equal (above); (d) invariance (above); (e) the control (above).
//     (f) a failed upload (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_PUT_FAIL=k: the k-th device tensor's put fails) after a
//         successful bind in the same process: the instance falls back to the CPU and gives the fast CPU route's rows
//         byte for byte, or refuses; never a row that is not finite (the second read's P0).
//     (g) what goes up is what the device computes (the second read's P1): the token embedding (read on the host)
//         stays on the host and the head goes up; on the F16 file (a type the device products do not take) the F16
//         projections stay on the host and the rows are the (c) rows' kind; a --cpu-moe load keeps every expert
//         stack on the host, and its rows too are within 1e-4 of the reference and finite.
//     (h) a device product that fails (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_LIN_FAIL=k: the k-th product fails) refuses
//         that forward and every later one on the instance, never a row (the second read's P2).
//     (i) each device kernel the products take (F32, BF16, Q8_0, NVFP4 with and without its factor, IQ3_XXS: the
//         tiny file's 64-value rows cannot hold IQ3_XXS's 256-value blocks) through gpu.h's gpu_ink2_put and
//         gpu_ink2_lin, from row 5 of a stored matrix, 3 columns, within 1e-4 of float64 sums over dequant_row's
//         decode (relative to the largest), finite. (d) and (e) run on all three files.
//  G2b step 1 (the expert body on the device, the router on the host):
//     (j) a forward runs each application's expert body as ONE device call (model_inkling2_cuda_calls: n_app calls
//         for a one-chunk forward), on F32, BF16 and Q8_0; the rows still pass (c), (d), (e);
//     (k) gpu_ink2_experts on a synthetic NVFP4 layer at the real model's widths (H 2304, F 1152: more than one block
//         in x for every kernel; E 5, one expert with no positions; 6 positions, 2 experts each) against float64 on the
//         host (dequant_row's decode; Inkling's SwiGLU in double), within 1e-4 of each position's largest value
//         (itself above 1e-6), finite. x is one-hot, so each gate value is one decoded weight times the expert's
//         factor: random codes (all 16 E2M1 values), the largest (448) and the smallest UE4M3 scale and all-zero
//         blocks, gate rows unlike their up rows, and the factors 1e-3, 1, 10, 0.05 put gates near 0, in range, beyond
//         +-10. A launch's position list with a repeat (test-only RUNNER_INKLING2_CUDA_TEST_DUP_IDX) refuses;
//     (l) a --cpu-moe load runs no device expert body (its stacks are on the host).
//  G2b step 2's B (one device copy per file and placement; the server's slots are its own instances):
//     (p) two loads of the BF16 file share ONE device copy (the placement accessor's handle) and give the same rows;
//         two instances forwarding at once on it give the sequential rows; the first freed, the second's rows do
//         not move; both freed, the device's free memory is back within 2 MiB of the figure before the first bind.
//         With the free memory capped (test-only RUNNER_INKLING2_CUDA_TEST_FREE) between the --cpu-moe need and
//         the whole need, both loads take the expert stacks on the host (M) and share; below both, both run on the
//         CPU (C). Never one instance on the device beside another on the CPU.
//  LoRA on the CUDA route (the supervisor's check C, before G2b step 2): the adapter's term is added on the host after
//  each attention product returns (ink2_lora_apply after ink2_lin_rows), on either route. Bars fixed before the run:
//     (q) the BF16 file with tests/inkling2_tiny.py --lora's adapter (all four attention slots of every layer): the
//         CUDA rows within 1e-4 of the CPU fast route's rows with the same adapter, finite; the adapter moves the CUDA
//         rows at least 10 times further than that distance; at scale 0 the CUDA rows equal the base CUDA rows byte
//         for byte.
//  R4.48.9.2, an operation-level gate (tests/test_inkling2_vmath.c's SwiGLU case, on the device):
//     (sw) the expert body's own SwiGLU, k_ink2_swiglu and the fused body's k_ink2_swiglu_n, against pmath.h's double
//         forms within 20 u (u = 2^-24), over g and u at every sign and 41 magnitudes from 1e-6 to 1e4 and 0, every
//         pair, at Inkling's caps 4 / 25 and the tiny files' 1 / 2, F 64 and 67. The rule is the host gate's, and its
//         anchor the CUDA C Programming Guide's maximum errors (tanhf and expf 2 ulp, at most 4 u); its live control,
//         the kernel's statements with the cancelling float32 tanh, must exceed the bound.
//
// usage: test_inkling2_cuda TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf TINY_F16.gguf TINY_LORA.gguf
#include "model.h"
#include "gpu.h"
#include "inkling2_model.h"
#include "inkling2.h"
#include "compat.h"
#include "quants.h"
#include "pmath.h"   // (sw): pm_exp_d, the SwiGLU's double reference
#include <float.h>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>   // D1a's tanh check: the reference with the FTZ / DAZ modes off (_mm_getcsr / _mm_setcsr)
#endif
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TOK = 24 };
static const int32_t PROMPT[N_TOK] = { 1, 5, 9, 2, 7, 3, 8, 4, 6, 0, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61 };
static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static bool cpu_moe;   // the next loads' --cpu-moe
static int n_seq;      // the next loads' slot count (model_params.n_seq: a server's --parallel)
static int ctx_len;    // the next loads' -c (0: 64)
static float *rows(model_t *m);
// (p): forwards on two instances at once, each on its own thread, over one shared device copy
typedef struct { model_t *m; float *out[4]; } par_job;
static void *par_run(void *a) {
    par_job *j = a;
    for (int k = 0; k < 4; k++) j->out[k] = rows(j->m);
    return NULL;
}
// the selection observer's C check: calls, positions, positions without exactly n_used experts, negative margins,
// and the distinct experts summed over calls (each distinct expert of a call is two products on the per-product path)
static struct { long calls, positions, bad_count, bad_margin, distinct; int want_used; } obs;
static void obs_fn(void *ctx, int block, int T, int E, const unsigned char *chosen, const float *margin) {
    (void)ctx; (void)block;
    obs.calls++;
    for (int e = 0; e < E; e++) {
        int any = 0;
        for (int t = 0; t < T; t++) any |= chosen[(size_t)t * E + e];
        obs.distinct += any;
    }
    for (int t = 0; t < T; t++) {
        int k = 0;
        for (int e = 0; e < E; e++) k += chosen[(size_t)t * E + e] != 0;
        obs.positions++;
        obs.bad_count += k != obs.want_used;
        obs.bad_margin += !(margin[t] >= 0.0f);
    }
}
static model_t *load_route(const char *path, bool gpu, bool no_factor, int n_batch, const char *route) {
    plat_setenv("RUNNER_INKLING2_ROUTE", route);
    plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_FACTOR", no_factor ? "1" : "");
    plat_setenv("RUNNER_INKLING2_CUDA", gpu ? "1" : "");   // opt-in until the CUDA forward is complete
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = gpu ? GPU_AUTO : GPU_OFF;
    p.n_threads = 2;
    p.n_ctx = ctx_len > 0 ? ctx_len : 64;
    p.n_batch = n_batch;
    p.cpu_moe = cpu_moe;
    p.n_seq = n_seq;
    model_t *m = calloc(1, sizeof *m);
    if (!m || !model_load(m, path, &p)) { free(m); return NULL; }
    return m;
}
static model_t *load(const char *path, bool gpu, bool no_factor, int n_batch) {
    // the CPU side is the reference route (the serial float32 engine), not the CPU's default fast route
    return load_route(path, gpu, no_factor, n_batch, gpu ? "" : "reference");
}
// NaN and infinity by their bits (all exponent bits set): this file builds with -ffast-math, under which isfinite()
// may fold to true
static size_t n_nonfinite(const float *a, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, a + i, sizeof u);
        k += (u & 0x7f800000u) == 0x7f800000u;
    }
    return k;
}
static float *rows_n(model_t *m, const int32_t *prompt, int n_tok) {
    float *r = malloc(sizeof(float) * (size_t)n_tok * (size_t)m->n_vocab);
    for (int pos = 0; r && pos < n_tok;) {
        const int n = n_tok - pos < m->n_batch ? n_tok - pos : m->n_batch;
        if (!model_inkling2_forward(m, prompt + pos, n, pos, true, true)) { free(r); return NULL; }
        for (int b = 0; b < n; b++)
            memcpy(r + (size_t)(pos + b) * (size_t)m->n_vocab, model_inkling2_row(m, b), sizeof(float) * (size_t)m->n_vocab);
        pos += n;
    }
    return r;
}
static float *rows_p(model_t *m, const int32_t *prompt) { return rows_n(m, prompt, N_TOK); }
// D1d.3b (ak2): the prompt one position a call whatever the instance's -b (each a resident step when the step is on)
static float *rows_steps(model_t *m) {
    float *r = malloc(sizeof(float) * (size_t)N_TOK * (size_t)m->n_vocab);
    for (int pos = 0; r && pos < N_TOK; pos++) {
        if (!model_inkling2_forward(m, PROMPT + pos, 1, pos, true, true)) { free(r); return NULL; }
        memcpy(r + (size_t)pos * (size_t)m->n_vocab, model_inkling2_row(m, 0), sizeof(float) * (size_t)m->n_vocab);
    }
    return r;
}
static void *par_run_steps(void *a) {
    par_job *j = a;
    for (int k = 0; k < 4; k++) j->out[k] = rows_steps(j->m);
    return NULL;
}
static float *rows(model_t *m) { return rows_p(m, PROMPT); }
static void drop(model_t *m) { if (m) { model_free(m); free(m); } }
// a value that is not finite on either side scores as a failure (1e30), whatever the compiler makes of a NaN compare
// (an IEEE compare with NaN is false, so a plain maximum would skip it; the second read's P1)
static double rel_diff(const float *a, const float *ref, size_t n) {
    if (n_nonfinite(a, n) || n_nonfinite(ref, n)) return 1e30;
    double d = 0, top = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabs((double)a[i] - ref[i]) > d) d = fabs((double)a[i] - ref[i]);
        if (fabs((double)ref[i]) > top) top = fabs((double)ref[i]);
    }
    return top > 0 ? d / top : d;
}

// G2b 2a (n): the device core over a K/V ring through the given chunks (consecutive positions from 0), each chunk's
// output against float64 on the host over the same keys (4 heads on 2 K/V heads, ext 16, the sinks varied). The
// largest difference relative to the largest value; *nf the values not finite, *cmp the values compared; 1 if the
// device did not run.
// *ndiff: the output values whose bits differ from the TEST-ONLY reference kernel's (k_ink2_attn_ref, the core before
// the decode fix) run on its own ring over the same chunks (want 0: the fix changes no bit).
// (ae), the second read's P2 3: what the helpers below observed of their last device call (gpu_ink2_test_stats): the
// batched dispatches it ran, the handle's buffer capacities after it (caps: x, attention output, gathered x,
// activations, stored logits), an expert call's columns and experts of 16+ columns, the attention's largest logits
// buffer and its smallest attention-output buffer short of T + 16 columns ((size_t)-1: none short)
static struct {
    long nb;
    size_t caps[5];
    long nall, nbig;
    int Q;
    bool core_unreserved;
    size_t alg_max, aatt_min;
} g_obs;

static double core_case(int hd, int global, int window, int R, const int *chunks, int n_chunks, size_t *nf, long *cmp,
                        long *ndiff) {
    enum { NH = 4, NKV = 2, EXT = 16 };
    const int C = NKV * hd, Q = NH * hd;
    int NP = 0, Tmax = 0;
    for (int c = 0; c < n_chunks; c++) { NP += chunks[c]; if (chunks[c] > Tmax) Tmax = chunks[c]; }
    const float scale = 1.0f / sqrtf((float)hd);
    uint32_t seed = 4242u + (uint32_t)hd;
#define RND() ((float)((int)((seed = seed * 1664525u + 1013904223u) >> 9) % 2001 - 1000) / 1000.0f)
    float *q = malloc(sizeof(float) * (size_t)NP * Q), *k = malloc(sizeof(float) * (size_t)NP * C);
    float *v = malloc(sizeof(float) * (size_t)NP * C), *rl = malloc(sizeof(float) * (size_t)NP * NH * EXT);
    float *sk = malloc(sizeof(float) * (size_t)NP * NH), *att = malloc(sizeof(float) * (size_t)Tmax * Q);
    float *att_ref = malloc(sizeof(float) * (size_t)Tmax * Q);
    double *lg = malloc(sizeof(double) * (size_t)NP), *o = malloc(sizeof(double) * (size_t)hd);
    int *pos = malloc(sizeof(int) * (size_t)NP);
    for (int i = 0; i < NP * Q; i++) q[i] = RND();
    for (int i = 0; i < NP * C; i++) { k[i] = RND(); v[i] = RND(); }
    for (int i = 0; i < NP * NH * EXT; i++) rl[i] = 2.0f * RND();
    for (int i = 0; i < NP * NH; i++) sk[i] = i % 7 == 0 ? -20.0f : i % 11 == 0 ? 5.0f : RND();
    for (int i = 0; i < NP; i++) pos[i] = i;
#undef RND
    gpu_ink2 *h = gpu_ink2_bind();
    const int rows[1] = { R }, cw[1] = { C };
    gpu_ink2_kv *kv = h ? gpu_ink2_kv_new(h, 1, rows, cw, 0) : NULL, *kvr = h ? gpu_ink2_kv_new(h, 1, rows, cw, 0) : NULL;
    double worst = 0, top = 0;
    // reserved as the fit does (the stored logits at most 16 MiB): 1040 queries x 4 heads x 1040 keys then run as two
    // groups of queries, so the grouping is under the byte identity too
    // (ae) g_obs.core_unreserved: no reservation (a direct caller); the stored logits must still stay within 16 MiB
    bool ran = kv != NULL && kvr != NULL && att_ref != NULL &&
               (g_obs.core_unreserved || gpu_ink2_reserve_attn(h, Tmax, Q, NH, EXT, NP));
    g_obs.Q = Q;
    g_obs.alg_max = 0;
    g_obs.aatt_min = (size_t)-1;
    for (int c = 0, p0 = 0; ran && c < n_chunks; p0 += chunks[c], c++) {
        const int T = chunks[c];
        ran = gpu_ink2_attn(h, kv, 0, T, pos + p0, q + (size_t)p0 * Q, k + (size_t)p0 * C, v + (size_t)p0 * C,
                            rl + (size_t)p0 * NH * EXT, sk + (size_t)p0 * NH, att, NH, NKV, hd, window, global, EXT, scale);
        gpu_ink2_test_attn_ref(h, 1);
        ran = ran && gpu_ink2_attn(h, kvr, 0, T, pos + p0, q + (size_t)p0 * Q, k + (size_t)p0 * C, v + (size_t)p0 * C,
                                   rl + (size_t)p0 * NH * EXT, sk + (size_t)p0 * NH, att_ref, NH, NKV, hd, window, global,
                                   EXT, scale);
        gpu_ink2_test_attn_ref(h, 0);
        if (!ran) break;
        {
            size_t cp[5];
            if (gpu_ink2_test_stats(h, cp) >= 0) {
                if (cp[4] > g_obs.alg_max) g_obs.alg_max = cp[4];
                const size_t want = sizeof(float) * ((size_t)T + 16) * (size_t)Q;   // the output, padded by INK2_XPAD
                if (cp[1] < want && cp[1] < g_obs.aatt_min) g_obs.aatt_min = cp[1];
            }
        }
        for (size_t x = 0; x < (size_t)T * Q; x++) {
            uint32_t ua, ub;
            memcpy(&ua, att + x, sizeof ua);
            memcpy(&ub, att_ref + x, sizeof ub);
            *ndiff += ua != ub;
        }
        *nf += n_nonfinite(att, (size_t)T * Q);
        for (int t = p0; t < p0 + T; t++)
            for (int hh = 0; hh < NH; hh++) {
                const int kh = hh / (NH / NKV), lo = global ? 0 : (t - window + 1 > 0 ? t - window + 1 : 0);
                double mx = sk[t * NH + hh];
                for (int j = lo; j <= t; j++) {
                    double sdot = 0;
                    for (int e = 0; e < hd; e++) sdot += (double)q[(size_t)t * Q + hh * hd + e] * k[(size_t)j * C + kh * hd + e];
                    lg[j] = sdot * scale + (t - j < EXT ? rl[((size_t)t * NH + hh) * EXT + (t - j)] : 0.0);
                    if (lg[j] > mx) mx = lg[j];
                }
                double sum = exp(sk[t * NH + hh] - mx);
                for (int d = 0; d < hd; d++) o[d] = 0;
                for (int j = lo; j <= t; j++) {
                    const double e = exp(lg[j] - mx);
                    sum += e;
                    for (int d = 0; d < hd; d++) o[d] += e * v[(size_t)j * C + kh * hd + d];
                }
                for (int d = 0; d < hd; d++) {
                    const double rr = o[d] / sum, g = att[(size_t)(t - p0) * Q + hh * hd + d];
                    if (fabs(rr) > top) top = fabs(rr);
                    if (fabs(g - rr) > worst) worst = fabs(g - rr);
                    (*cmp)++;
                }
            }
    }
    gpu_ink2_kv_free(kv);
    gpu_ink2_kv_free(kvr);
    gpu_ink2_free(h);
    free(q); free(k); free(v); free(rl); free(sk); free(att); free(att_ref); free(lg); free(o); free(pos);
    return ran && top > 0 ? worst / top : 1;
}

// D2 (x2): the fused expert body against the former per-expert launches (TEST-ONLY gpu_ink2_test_experts_ref) on the
// same inputs, bit for bit. Dense random x (the one-hot x of (k) makes most sums trivial); a stack of E experts of
// `type` (NVFP4: random codes and scales, the factors given; BF16: random values, no factors); chosen: position t takes
// `used` experts starting at (t * 5 + 1) % E stepping 3 (so experts repeat across positions), plus expert `all` at
// every position (-1: none) and expert `none` never. *nval the values compared; returns the values whose bits differ,
// -1 if a call did not run.
static long exp_identity(int type, int H, int F, int E, int T, int used, int all, int none, long *nval) {
    const size_t rb13 = type == T_NVFP4 ? (size_t)H / 64 * 36 : (size_t)H * 2, rb2 = type == T_NVFP4 ? (size_t)F / 64 * 36 : (size_t)F * 2;
    unsigned char *w13 = malloc((size_t)E * 2 * F * rb13), *w2 = malloc((size_t)E * H * rb2);
    float *x = malloc(sizeof(float) * (size_t)T * H), *o1 = malloc(sizeof(float) * (size_t)T * H);
    float *o2 = malloc(sizeof(float) * (size_t)T * H), *wt = calloc((size_t)T * E, sizeof(float));
    float *s13 = malloc(sizeof(float) * (size_t)E), *s2 = malloc(sizeof(float) * (size_t)E);
    unsigned char *ch = calloc((size_t)T * E, 1);
    long diff = -1;
    *nval = 0;
    if (!w13 || !w2 || !x || !o1 || !o2 || !wt || !s13 || !s2 || !ch) goto done;
    uint32_t seed = 991u + (uint32_t)type * 7u + (uint32_t)T;
#define RNDU() (seed = seed * 1664525u + 1013904223u, seed >> 8)
#define RNDF() ((float)((int)(RNDU() % 2001u) - 1000) / 1000.0f)
    if (type == T_NVFP4) {
        for (size_t i = 0; i < (size_t)E * 2 * F * rb13; i++) w13[i] = (unsigned char)RNDU();
        for (size_t i = 0; i < (size_t)E * H * rb2; i++) w2[i] = (unsigned char)RNDU();
        for (size_t r = 0; r < (size_t)E * 2 * F; r++)   // scale bytes in a sane range (no NaN code 0x7F)
            for (size_t b = 0; b < rb13; b += 36) for (int k = 0; k < 4; k++) w13[r * rb13 + b + k] = (unsigned char)(0x30 + RNDU() % 16);
        for (size_t r = 0; r < (size_t)E * H; r++)
            for (size_t b = 0; b < rb2; b += 36) for (int k = 0; k < 4; k++) w2[r * rb2 + b + k] = (unsigned char)(0x30 + RNDU() % 16);
    } else {   // BF16: a float's top half
        for (size_t i = 0; i < (size_t)E * 2 * F * (size_t)H; i++) { const float f = 0.05f * RNDF(); uint32_t u; memcpy(&u, &f, 4); ((uint16_t *)w13)[i] = (uint16_t)(u >> 16); }
        for (size_t i = 0; i < (size_t)E * H * (size_t)F; i++) { const float f = 0.05f * RNDF(); uint32_t u; memcpy(&u, &f, 4); ((uint16_t *)w2)[i] = (uint16_t)(u >> 16); }
    }
    for (size_t i = 0; i < (size_t)T * H; i++) x[i] = RNDF();
    for (int e = 0; e < E; e++) { s13[e] = 0.5f + 0.25f * (float)(e % 5); s2[e] = 1.5f - 0.125f * (float)(e % 7); }
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < used; k++) {
            const int e = ((t * 5 + 1) % E + 3 * k) % E;
            if (e == none) continue;
            ch[(size_t)t * E + e] = 1; wt[(size_t)t * E + e] = 0.1f + 0.05f * (float)k + 0.001f * (float)t;
        }
        if (all >= 0) { ch[(size_t)t * E + all] = 1; wt[(size_t)t * E + all] = 0.3f; }
    }
#undef RNDU
#undef RNDF
    {
        gpu_ink2 *h = gpu_ink2_bind();
        const float *f13 = type == T_NVFP4 ? s13 : NULL, *f2 = type == T_NVFP4 ? s2 : NULL;
        const int i13 = h ? gpu_ink2_put(h, w13, (size_t)E * 2 * F * rb13, f13, f13 ? E : 0) : -1;
        const int i2 = h ? gpu_ink2_put(h, w2, (size_t)E * H * rb2, f2, f2 ? E : 0) : -1;
        const long nb0 = gpu_ink2_test_stats(h, NULL);
        bool ran = i13 >= 0 && i2 >= 0 &&
                   gpu_ink2_experts(h, i13, i2, type, type, rb13, rb2, H, F, E, T, x, ch, wt, f13, f2, 4.0f, 8.0f, o1);
        g_obs.nb = gpu_ink2_test_stats(h, g_obs.caps) - nb0;   // (ae): the tiled products, the gathered buffers
        g_obs.nall = 0;
        g_obs.nbig = 0;
        for (int e = 0; e < E; e++) {
            int n = 0;
            for (int t = 0; t < T; t++) n += ch[(size_t)t * E + e];
            g_obs.nall += n;
            g_obs.nbig += n >= 16;
        }
        gpu_ink2_test_experts_ref(h, 1);   // the reference: the former per-expert launches, each product per column
        gpu_ink2_test_mv_ref(h, 1);
        ran = ran && gpu_ink2_experts(h, i13, i2, type, type, rb13, rb2, H, F, E, T, x, ch, wt, f13, f2, 4.0f, 8.0f, o2);
        gpu_ink2_test_experts_ref(h, 0);
        gpu_ink2_test_mv_ref(h, 0);
        if (ran) {
            diff = 0;
            for (size_t i = 0; i < (size_t)T * H; i++) {
                uint32_t a, b;
                memcpy(&a, o1 + i, 4); memcpy(&b, o2 + i, 4);
                diff += a != b;
            }
            *nval = (long)T * H;
        }
        gpu_ink2_free(h);
    }
done:
    free(w13); free(w2); free(x); free(o1); free(o2); free(wt); free(s13); free(s2); free(ch);
    return diff;
}
// D1d.3a (ai): the device-routed expert body (gpu_ink2_route_experts: the device router, its tables, the body) on its
// SIZED launches (the counts read back to the host) and on its BOUNDED ones (gpu_ink2_test_bounded: each launch sized
// by its bound, the live counts read on the device), same inputs, one handle: the rows, the choices and the margins
// compared bit for bit; *rb_sized / *rb_bounded the body's count readbacks each call made (want 1 / 0); *ntiled the
// experts with 16+ columns (the tiled launches live). Returns the differing values, -1 if a call failed.
static long body_bounded_identity(int type, int H, int F, int E, int K, int T, long *nval, long *rb_sized, long *rb_bounded,
                           int *ntiled) {
    const size_t rb13 = type == T_NVFP4 ? (size_t)H / 64 * 36 : (size_t)H * 2, rb2 = type == T_NVFP4 ? (size_t)F / 64 * 36 : (size_t)F * 2;
    unsigned char *w13 = malloc((size_t)E * 2 * F * rb13), *w2 = malloc((size_t)E * H * rb2);
    float *wr = malloc(sizeof(float) * (size_t)E * H), *br = malloc(sizeof(float) * (size_t)E);
    float *x = malloc(sizeof(float) * (size_t)T * H), *o1 = malloc(sizeof(float) * (size_t)T * H);
    float *o2 = malloc(sizeof(float) * (size_t)T * H), *m1 = malloc(sizeof(float) * (size_t)T), *m2 = malloc(sizeof(float) * (size_t)T);
    float *s13 = malloc(sizeof(float) * (size_t)E), *s2 = malloc(sizeof(float) * (size_t)E);
    unsigned char *c1 = malloc((size_t)T * E), *c2 = malloc((size_t)T * E);
    long diff = -1;
    *nval = 0; *rb_sized = *rb_bounded = -1; *ntiled = 0;
    if (!w13 || !w2 || !wr || !br || !x || !o1 || !o2 || !m1 || !m2 || !s13 || !s2 || !c1 || !c2) goto done;
    uint32_t seed = 4441u + (uint32_t)type * 7u + (uint32_t)T * 13u + (uint32_t)E;
#define RNDU() (seed = seed * 1664525u + 1013904223u, seed >> 8)
#define RNDF() ((float)((int)(RNDU() % 2001u) - 1000) / 1000.0f)
    if (type == T_NVFP4) {
        for (size_t i = 0; i < (size_t)E * 2 * F * rb13; i++) w13[i] = (unsigned char)RNDU();
        for (size_t i = 0; i < (size_t)E * H * rb2; i++) w2[i] = (unsigned char)RNDU();
        for (size_t r = 0; r < (size_t)E * 2 * F; r++)   // scale bytes in a sane range (no NaN code 0x7F)
            for (size_t b = 0; b < rb13; b += 36) for (int k = 0; k < 4; k++) w13[r * rb13 + b + k] = (unsigned char)(0x30 + RNDU() % 16);
        for (size_t r = 0; r < (size_t)E * H; r++)
            for (size_t b = 0; b < rb2; b += 36) for (int k = 0; k < 4; k++) w2[r * rb2 + b + k] = (unsigned char)(0x30 + RNDU() % 16);
    } else {
        for (size_t i = 0; i < (size_t)E * 2 * F * (size_t)H; i++) { const float f = 0.05f * RNDF(); uint32_t u; memcpy(&u, &f, 4); ((uint16_t *)w13)[i] = (uint16_t)(u >> 16); }
        for (size_t i = 0; i < (size_t)E * H * (size_t)F; i++) { const float f = 0.05f * RNDF(); uint32_t u; memcpy(&u, &f, 4); ((uint16_t *)w2)[i] = (uint16_t)(u >> 16); }
    }
    for (size_t i = 0; i < (size_t)E * H; i++) wr[i] = 0.05f * RNDF();
    for (int e = 0; e < E; e++) { br[e] = 0.01f * RNDF(); s13[e] = 0.5f + 0.25f * (float)(e % 5); s2[e] = 1.5f - 0.125f * (float)(e % 7); }
    for (size_t i = 0; i < (size_t)T * H; i++) x[i] = RNDF();
    for (int e = 0; e < 2 && e < E; e++) br[e] = 5.0f;   // two experts chosen everywhere (the bias picks): at T >= 16
                                                         // they reach a tile's 16 columns
#undef RNDU
#undef RNDF
    {
        gpu_ink2 *h = gpu_ink2_bind();
        const float *f13 = type == T_NVFP4 ? s13 : NULL, *f2 = type == T_NVFP4 ? s2 : NULL;
        const int i13 = h ? gpu_ink2_put(h, w13, (size_t)E * 2 * F * rb13, f13, f13 ? E : 0) : -1;
        const int i2 = h ? gpu_ink2_put(h, w2, (size_t)E * H * rb2, f2, f2 ? E : 0) : -1;
        const int ir = h ? gpu_ink2_put(h, wr, sizeof(float) * (size_t)E * H, NULL, 0) : -1;
        const long r0 = gpu_ink2_test_body_readbacks(h);
        bool ran = i13 >= 0 && i2 >= 0 && ir >= 0 &&
                   gpu_ink2_route_experts(h, ir, br, 2.5f, K, i13, i2, type, type, rb13, rb2, H, F, E, T, x, f13, f2,
                                          4.0f, 8.0f, o1, c1, m1);
        const long r1 = gpu_ink2_test_body_readbacks(h);
        ran = ran && gpu_ink2_test_poison_body(h);   // the second call cannot reuse the first's scratch
        gpu_ink2_test_bounded(h, 1);
        ran = ran && gpu_ink2_route_experts(h, ir, br, 2.5f, K, i13, i2, type, type, rb13, rb2, H, F, E, T, x, f13, f2,
                                            4.0f, 8.0f, o2, c2, m2);
        gpu_ink2_test_bounded(h, 0);
        const long r2 = gpu_ink2_test_body_readbacks(h);
        *rb_sized = r1 - r0; *rb_bounded = r2 - r1;
        if (ran) {
            diff = 0;
            for (size_t i = 0; i < (size_t)T * H; i++) { uint32_t a, b; memcpy(&a, o1 + i, 4); memcpy(&b, o2 + i, 4); diff += a != b; }
            for (size_t i = 0; i < (size_t)T; i++) { uint32_t a, b; memcpy(&a, m1 + i, 4); memcpy(&b, m2 + i, 4); diff += a != b; }
            for (size_t i = 0; i < (size_t)T * E; i++) diff += c1[i] != c2[i];
            *nval = (long)T * H + T + (long)T * E;
            for (int e = 0; e < E; e++) {
                int n = 0;
                for (int t = 0; t < T; t++) n += c1[(size_t)t * E + e] != 0;
                *ntiled += n >= 16;
            }
        }
        gpu_ink2_free(h);
    }
done:
    free(w13); free(w2); free(wr); free(br); free(x); free(o1); free(o2); free(m1); free(m2); free(s13); free(s2);
    free(c1); free(c2);
    return diff;
}

// P1a/P1b (i2): a BF16 or NVFP4 product through gpu_ink2_lin at T columns, batched (k_mv_bf16_b / k_ink2_mv_nvfp4_b from 16
// columns) against every column its own batch-1 launch (TEST-ONLY gpu_ink2_test_mv_ref), bit for bit; random weights
// (BF16: exponents kept sane; NVFP4: random codes, scale bytes in a sane range, the factor 0.75) and dense random x.
// *nval the values compared; returns the values whose bits differ, -1 if a call did not run.
static long lin_identity(int type, int n_in, int n_out, int T, long *nval) {
    const size_t rb = type == T_NVFP4 ? (size_t)n_in / 64 * 36 : (size_t)n_in * 2;
    unsigned char *w = malloc(rb * (size_t)n_out);
    float *x = malloc(sizeof(float) * (size_t)T * n_in), *y1 = malloc(sizeof(float) * (size_t)T * n_out);
    float *y2 = malloc(sizeof(float) * (size_t)T * n_out);
    const float fac = type == T_NVFP4 ? 0.75f : 1.0f;
    long diff = -1;
    *nval = 0;
    if (w && x && y1 && y2) {
        uint32_t seed = 555u + (uint32_t)T + 31u * (uint32_t)type;
        if (type == T_NVFP4) {
            for (size_t i = 0; i < rb * (size_t)n_out; i++) { seed = seed * 1664525u + 1013904223u; w[i] = (unsigned char)(seed >> 24); }
            for (size_t r = 0; r < (size_t)n_out; r++)
                for (size_t b = 0; b < rb; b += 36)
                    for (int k = 0; k < 4; k++) { seed = seed * 1664525u + 1013904223u; w[r * rb + b + k] = (unsigned char)(0x30 + (seed >> 28)); }
        } else {
            for (size_t i = 0; i < (size_t)n_in * n_out; i++) {
                seed = seed * 1664525u + 1013904223u;
                const float f = 0.05f * (float)((int)(seed >> 9) % 2001 - 1000) / 1000.0f;
                uint32_t u;
                memcpy(&u, &f, 4);
                ((uint16_t *)w)[i] = (uint16_t)(u >> 16);
            }
        }
        for (size_t i = 0; i < (size_t)T * n_in; i++) {
            seed = seed * 1664525u + 1013904223u;
            x[i] = (float)((int)(seed >> 9) % 2001 - 1000) / 1000.0f;
        }
        gpu_ink2 *h = gpu_ink2_bind();
        const int i0 = h ? gpu_ink2_put(h, w, rb * (size_t)n_out, NULL, 0) : -1;
        const long nb0 = gpu_ink2_test_stats(h, NULL);
        bool ran = i0 >= 0 && gpu_ink2_lin(h, i0, type, rb, 0, n_out, n_in, T, x, y1, fac);
        g_obs.nb = gpu_ink2_test_stats(h, g_obs.caps) - nb0;   // (ae): the batched dispatches of the call, its x buffer
        gpu_ink2_test_mv_ref(h, 1);
        ran = ran && gpu_ink2_lin(h, i0, type, rb, 0, n_out, n_in, T, x, y2, fac);
        gpu_ink2_test_mv_ref(h, 0);
        if (ran) {
            diff = 0;
            for (size_t i = 0; i < (size_t)T * n_out; i++) diff += memcmp(y1 + i, y2 + i, 4) != 0;
            *nval = (long)T * n_out;
        }
        gpu_ink2_free(h);
    }
    free(w); free(x); free(y1); free(y2);
    return diff;
}

// D1a (y2): the reference for the device message step is the host's ink2_message_steps_c AS THE CUDA ROUTE RUNS IT:
// its norms and combination on the host, its products on the device through the view's lin op. This test instance's
// lin op runs a product on the device as the route's dev_lin does (gpu_ink2_lin on the view's device index).
static gpu_ink2 *g_mh;
static int t_lin(const ink2_tensor *W, int64_t row0, int n_out, int n_in, int T, const float *x, size_t xs, float *y,
                 size_t ys, float scale) {
    float *xc = (float *)x, *yc = y;
    if (xs != (size_t)n_in) {
        xc = malloc(sizeof(float) * (size_t)T * n_in);
        for (int t = 0; t < T; t++) memcpy(xc + (size_t)t * n_in, x + (size_t)t * xs, sizeof(float) * (size_t)n_in);
    }
    if (ys != (size_t)n_out) yc = malloc(sizeof(float) * (size_t)T * n_out);
    const bool ok = gpu_ink2_lin(g_mh, W->dev_i, W->qtype, W->q_row_bytes, row0, n_out, n_in, T, xc, yc, scale);
    if (ok && yc != y) for (int t = 0; t < T; t++) memcpy(y + (size_t)t * ys, yc + (size_t)t * n_out, sizeof(float) * (size_t)n_out);
    if (xc != x) free(xc);
    if (yc != y) free(yc);
    return ok ? 0 : -1;
}
static const ink2_dev_ops g_mops = { t_lin, NULL, NULL, NULL, NULL, NULL, NULL };
static ink2_dev_hdr g_minst = { &g_mops };
static ink2_tensor g_mt[9];
static char g_mname[9][128];
static const ink2_tensor *t_get(void *ctx, const char *name) {
    (void)ctx;
    for (int k = 0; k < 9; k++) if (g_mname[k][0] && !strcmp(name, g_mname[k])) return &g_mt[k];
    return NULL;
}
// one message step (block 1, roles q/k/v or mlp) on random BF16 gates whose forget and message rows are scaled in
// thirds by 1000, 1e-6 and 1 (so their outputs land in tanh's three branches: |v| > 20, |v| < 1e-4, between), dense
// random x (shared: all roles one x, as a sliding layer passes it) and messages [T][4][H] at stride 4H: the device step
// against the host's, bit for bit. *nval the values compared; returns the values whose bits differ, -1: did not run.
static long msg_identity(int H, int r, int T, int n_roles, bool shared_x, long *nval) {
    static const char *const qkv[3] = { "q", "k", "v" }, *const mlp[1] = { "mlp" };
    const char *const *roles = n_roles == 3 ? qkv : mlp;
    static const char *const part[3] = { "down_proj", "forget_gate_proj", "message_gate_proj" };
    gpu_ink2 *h = gpu_ink2_bind();
    g_mh = h;
    long diff = -1;
    *nval = 0;
    uint16_t *wb[9] = { 0 };
    float *x = malloc(sizeof(float) * 3 * (size_t)T * H), *msg = malloc(sizeof(float) * (size_t)T * 4 * H);
    float *o1 = malloc(sizeof(float) * 3 * (size_t)T * H), *o2 = malloc(sizeof(float) * 3 * (size_t)T * H);
    uint32_t seed = 4321u + (uint32_t)(T * 7 + n_roles + (shared_x ? 100 : 0) + H);
#define RNDF() (seed = seed * 1664525u + 1013904223u, (float)((int)(seed >> 9) % 2001 - 1000) / 1000.0f)
    bool ok = h && x && msg && o1 && o2;
    memset(g_mt, 0, sizeof g_mt);
    memset(g_mname, 0, sizeof g_mname);
    gpu_ink2_msg_role ro[3];
    for (int k = 0; ok && k < n_roles; k++) {
        int idx[3];
        for (int u = 0; u < 3; u++) {
            const int n_in = u == 0 ? 2 * H : r, n_out = u == 0 ? r : H, slot = 3 * k + u;
            wb[slot] = malloc(sizeof(uint16_t) * (size_t)n_in * n_out);
            if (!wb[slot]) { ok = false; break; }
            for (int o = 0; o < n_out; o++)
                for (int i = 0; i < n_in; i++) {
                    const float sc = u == 0 ? 0.05f : o % 3 == 0 ? 50.0f : o % 3 == 1 ? 5e-8f : 0.05f;
                    const float f = sc * RNDF();
                    uint32_t bits;
                    memcpy(&bits, &f, 4);
                    wb[slot][(size_t)o * n_in + i] = (uint16_t)(bits >> 16);
                }
            idx[u] = gpu_ink2_put(h, wb[slot], sizeof(uint16_t) * (size_t)n_in * n_out, NULL, 0);
            snprintf(g_mname[slot], sizeof g_mname[slot], "model.layers.1.%s_message_gate.%s.weight", roles[k], part[u]);
            g_mt[slot] = (ink2_tensor){ g_mname[slot], NULL, 2, { n_in, n_out, 1, 1 }, T_BF16, wb[slot],
                                        sizeof(uint16_t) * (size_t)n_in, NULL, 0, &g_minst, idx[u] };
            ok = idx[u] >= 0;
        }
        ro[k] = (gpu_ink2_msg_role){ idx[0], idx[1], idx[2], T_BF16, T_BF16, T_BF16, 1.0f, 1.0f, 1.0f };
    }
    if (ok) {
        for (size_t i = 0; i < 3 * (size_t)T * H; i++) x[i] = RNDF();
        for (size_t i = 0; i < (size_t)T * 4 * H; i++) msg[i] = RNDF();
        ink2_shape s;
        memset(&s, 0, sizeof s);
        s.hidden = H; s.loop_gate_rank = r; s.eps = 1e-6f;
        ink2_weights w = { t_get, NULL, NULL };
        const float *xs[3], *ms[3];
        float *r1[3], *r2[3];
        for (int k = 0; k < n_roles; k++) {
            xs[k] = x + (shared_x ? 0 : (size_t)k * T * H); ms[k] = msg + (size_t)k * H;
            r1[k] = o1 + (size_t)k * T * H; r2[k] = o2 + (size_t)k * T * H;
        }
        ok = ink2_test_message_steps(&s, &w, 1, roles, n_roles, T, xs, ms, (size_t)4 * H, r1) == INK2_OK &&
             gpu_ink2_msg_steps(h, n_roles, T, H, r, s.eps, ro, xs, ms, (size_t)4 * H, r2);
        if (ok) {
            diff = 0;
            for (size_t i = 0; i < (size_t)n_roles * T * H; i++) diff += memcmp(o1 + i, o2 + i, 4) != 0;
            *nval = (long)n_roles * T * H;
        }
    }
#undef RNDF
    for (int k = 0; k < 9; k++) free(wb[k]);
    free(x); free(msg); free(o1); free(o2);
    gpu_ink2_free(h);
    g_mh = NULL;
    return diff;
}

// D1b B1: the device router against the host's ink2_route_one on the same inputs, bit for bit: scores, choices and
// weights. Random router rows and x, with CONSTRUCTED TIES: experts 0..K-2 get bias +10 (chosen), among them 1 and 2
// share a row (a tie inside the top K); experts K-1 and K share a row and bias 1.0 (an exact tie for the K-th place:
// the lower index must win); with near = true expert K's bias is one ulp above (the higher index wins by one ulp);
// expert K+1's row is 1000x (its score 0 or 1: the sigmoid's extremes), K+2's row zero (l = 0); the rest bias -10.
// The host reference with the CPU's FTZ / DAZ off (the sigmoid's far tail is subnormal; see D1a's tanh check).
// *nval the values compared; returns the values that differ (scores and weights by their bits), -1: did not run.
// what route_identity's last call saw: the values that differ by kind (scores, choices, weights, constructed ties
// gone the other way on the host) and the first differing value's place and values
static struct {
    long n[4], first;
    float sh, sd, wh, wd;
    unsigned char ch, cd;
} g_robs;

// D1b.1 (the second read's F3, before any kernel change): kind 0 the construction above; the special values, each on
// top of it, where a parallel argmax would part from the serial scan (ink2_route_one: `best < 0 || v > bv`, so a NaN
// in a pass's FIRST unchosen slot is chosen, a NaN anywhere else never is, ties go to the lowest index):
//   1 NANFIRST  router row 0 all NaN: expert 0's score NaN, the first slot of the first pass
//   2 NANLATE   row K + 2 NaN (not a first slot until the passes reach it) and expert K - 1's bias NaN
//   3 INF       bias +inf on expert 1 and -inf on expert K; row K + 1 x 1e38 (its logits overflow to +-inf: score 1 or 0)
//   4 ZERO      experts K - 1 and K zero rows with bias -0.5 (v = 0.5 + -0.5 = +0 exactly, a tie at +0); experts K + 1
//               and K + 2 zero rows with bias +0.0 and -0.0 (v = 0.5 both: the K-th place's tie by a signed zero).
//               (v = -0 itself cannot arise: a score is >= +0 and +0 + -0 rounds to +0.)
//   5 KEQE      K == E: every expert chosen (the constructions above skipped; they need K + 3 <= E)
// A NaN score or weight compares equal to a NaN on the other side, whatever its payload (x86 keeps a NaN's payload,
// CUDA returns its canonical NaN); every other value by its bits.
// (NaN by its bits: this test builds with -ffast-math, whose finite-math assumption folds isnan() to false)
static bool nan_bits(float v) { uint32_t u; memcpy(&u, &v, 4); return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu); }
static bool same_f(float a, float b) { return memcmp(&a, &b, 4) == 0 || (nan_bits(a) && nan_bits(b)); }

static long route_identity_k(int E, int K, int H, int T, bool near, int kind, long *nval);
static long route_identity(int E, int K, int H, int T, bool near, long *nval) {
    return route_identity_k(E, K, H, T, near, 0, nval);
}
static long route_identity_k(int E, int K, int H, int T, bool near, int kind, long *nval) {
    memset(&g_robs, 0, sizeof g_robs);
    g_robs.first = -1;
    float *wr = malloc(sizeof(float) * (size_t)E * H), *br = malloc(sizeof(float) * (size_t)E);
    float *x = malloc(sizeof(float) * (size_t)T * H), *sh = malloc(sizeof(float) * (size_t)T * E);
    float *sd = malloc(sizeof(float) * (size_t)T * E), *wh = malloc(sizeof(float) * (size_t)T * E);
    float *wd = malloc(sizeof(float) * (size_t)T * E);
    unsigned char *ch = malloc((size_t)T * E), *cd = malloc((size_t)T * E);
    long diff = -1;
    *nval = 0;
    if (wr && br && x && sh && sd && wh && wd && ch && cd &&
        (kind == 5 ? K == E : kind == 6 ? K + 32 <= E && K + 3 <= E : K + 3 <= E)) {
        uint32_t seed = 2468u + (uint32_t)(E * 3 + T + (near ? 1 : 0) + 7 * kind);
#define RNDF() (seed = seed * 1664525u + 1013904223u, (float)((int)(seed >> 9) % 2001 - 1000) / 1000.0f)
        for (size_t i = 0; i < (size_t)E * H; i++) wr[i] = 0.02f * RNDF();
        for (size_t i = 0; i < (size_t)T * H; i++) x[i] = RNDF();
#undef RNDF
        for (int e = 0; e < E; e++) br[e] = e < K - 1 ? 10.0f : -10.0f;
        if (kind == 5) goto constructed;
        if (K >= 3) memcpy(wr + 2 * (size_t)H, wr + 1 * (size_t)H, sizeof(float) * H);    // a tie inside the top K
        memcpy(wr + (size_t)K * H, wr + (size_t)(K - 1) * H, sizeof(float) * H);           // the K-th place's tie
        // near: a one-ulp gap in the ranked value v = score + bias needs the score exact, or the two sums can round to
        // one float (v = sc + 1 at a round-to-even tie: a one-ulp gap in the bias alone collapsed on 2 of 17 positions,
        // on the host as on the device). Zero rows give l = 0, sc = sigmoid(0) = 0.5 exactly: v = 1.5 and 1.5 + 2^-23.
        if (near) {
            memset(wr + (size_t)(K - 1) * H, 0, sizeof(float) * H);
            memset(wr + (size_t)K * H, 0, sizeof(float) * H);
        }
        br[K - 1] = 1.0f;
        br[K] = near ? nextafterf(1.0f, 2.0f) : 1.0f;
        for (int i = 0; i < H; i++) wr[(size_t)(K + 1) * H + i] *= 1000.0f;                 // the sigmoid's extremes
        memset(wr + (size_t)(K + 2) * H, 0, sizeof(float) * H);                            // l = 0
        if (kind == 1) for (int i = 0; i < H; i++) wr[i] = NAN;
        if (kind == 2) { for (int i = 0; i < H; i++) wr[(size_t)(K + 2) * H + i] = NAN; br[K - 1] = NAN; }
        if (kind == 3) {
            br[1] = INFINITY; br[K] = -INFINITY;
            for (int i = 0; i < H; i++) wr[(size_t)(K + 1) * H + i] = (i % 2 ? -1e38f : 1e38f);
        }
        if (kind == 4) {
            for (int k = K - 1; k <= K + 2; k++) memset(wr + (size_t)k * H, 0, sizeof(float) * H);
            br[K - 1] = -0.5f; br[K] = -0.5f; br[K + 1] = 0.0f; br[K + 2] = -0.0f;
        }
        // 6 LANE (D1b.1: the warp top-k scans experts lane, lane + 32, ... in a lane): the K-th place's tie three ways,
        // K - 1, K and K - 1 + 32 (the same lane as K - 1) identical rows and biases; the lowest index, K - 1, wins
        if (kind == 6) {
            memcpy(wr + (size_t)(K - 1 + 32) * H, wr + (size_t)(K - 1) * H, sizeof(float) * H);
            br[K - 1 + 32] = br[K - 1];
        }
    constructed:;
        ink2_shape s;
        memset(&s, 0, sizeof s);
        s.hidden = H; s.n_expert = E; s.n_used = K; s.route_scale = 2.5f;
        gpu_ink2 *h = gpu_ink2_bind();
        const int ir = h ? gpu_ink2_put(h, wr, sizeof(float) * (size_t)E * H, NULL, 0) : -1;
        const bool ran = ir >= 0 && gpu_ink2_test_route(h, ir, br, s.route_scale, K, E, H, T, x, sd, cd, wd);
#if defined(__x86_64__) || defined(__i386__)
        const unsigned csr = _mm_getcsr();
        _mm_setcsr(csr & ~0x8040u);
#endif
        ink2_test_route(&s, wr, br, T, x, sh, ch, wh);
#if defined(__x86_64__) || defined(__i386__)
        _mm_setcsr(csr);
#endif
        if (ran) {
            diff = 0;
            for (size_t i = 0; i < (size_t)T * E; i++)
            {
                const int ds = !same_f(sh[i], sd[i]), dc = ch[i] != cd[i], dw = !same_f(wh[i], wd[i]);
                g_robs.n[0] += ds; g_robs.n[1] += dc; g_robs.n[2] += dw;
                if ((ds || dc || dw) && g_robs.first < 0) {
                    g_robs.first = (long)i; g_robs.sh = sh[i]; g_robs.sd = sd[i]; g_robs.wh = wh[i]; g_robs.wd = wd[i];
                    g_robs.ch = ch[i]; g_robs.cd = cd[i];
                }
                diff += ds + dc + dw;
            }
            *nval = 3L * T * E;
            // the constructed ties went the host's way (the lowest index; with near, the higher by one ulp)
            for (int t = 0; kind == 0 && t < T; t++) {
                const int bad = ch[(size_t)t * E + (near ? K : K - 1)] != 1;
                g_robs.n[3] += bad;
                diff += bad;
            }
            // the special constructions did what they claim, on the host (else the case tests nothing): NANFIRST
            // chose expert 0; ZERO gave the K-th place to K + 1 (bias +0.0) over K + 2 (-0.0), the lower index; KEQE
            // chose all
            for (int t = 0; t < T; t++) {
                const unsigned char *c = ch + (size_t)t * E;
                int bad = 0;
                if (kind == 1) bad = c[0] != 1;
                if (kind == 4) bad = c[K + 1] != 1 || c[K + 2] != 0;
                if (kind == 5) for (int e = 0; e < E; e++) bad |= c[e] != 1;
                if (kind == 6) bad = c[K - 1] != 1 || c[K] != 0 || c[K - 1 + 32] != 0;   // the lowest of the three
                g_robs.n[3] += bad;
                diff += bad;
            }
        }
        gpu_ink2_free(h);
    }
    free(wr); free(br); free(x); free(sh); free(sd); free(wh); free(wd); free(ch); free(cd);
    return diff;
}

// (sw), R4.48.9.2: tests/test_inkling2_vmath.c's SwiGLU reference and control, for the device's kernels. pmath.h's
// double forms: p_tanhf's body before its float rounding, and the exp in double. The softcap divides as the kernel
// does, a correctly rounded float32 quotient, then takes the tanh of it in double.
static const double SW_U = 5.9604644775390625e-08;   // 2^-24
static double sw_ref_tanh(float x) {
    double xd = (double)x;
    if (xd > 20.0) return 1.0;
    if (xd < -20.0) return -1.0;
    if (xd < 1e-4 && xd > -1e-4) return xd * (1.0 - xd * xd * (1.0 / 3.0));
    double e = pm_exp_d(2.0 * xd);
    return (e - 1.0) / (e + 1.0);
}
static double sw_ref(float g, float u, float gc, float uc) {
    const double sg = sw_ref_tanh((float)((double)g / gc)) * gc, su = sw_ref_tanh((float)((double)u / uc)) * uc;
    return sg * (1.0 / (1.0 + pm_exp_d(-(double)g))) * su;
}
// THE CONTROL: R4.48.20's cancelling tanh, (1 - e^-2|x|) / (1 + e^-2|x|) in float32, one lane
static float sw_tanh_cancel(float x) {
    float ax = fabsf(x);
    if (ax < 1e-3f) return x * (1.0f - x * x * (1.0f / 3.0f));
    if (ax >= 10.0f) return copysignf(1.0f, x);
    float e = p_expf(-2.0f * ax);
    return copysignf((1.0f - e) / (1.0f + e), x);
}
// The worst relative error in u of the device's SwiGLU (bounded: k_ink2_swiglu_n; ctl: the control on the host,
// through the kernel's own float32 statements) over every (g, u) pair of 0 and +-1e-6 .. 1e4 at 41 magnitudes; -1 if
// the hook did not run. *wg / *wu: the worst pair.
static double sw_worst(gpu_ink2 *h, float gc, float uc, int F, int bounded, int ctl, float *wg, float *wu) {
    enum { NMAG = 41 };
    float vals[1 + 2 * NMAG];
    int nv = 0;
    vals[nv++] = 0.0f;
    for (int m = 0; m < NMAG; m++) {
        const float a = (float)pow(10.0, -6.0 + 10.0 * m / (NMAG - 1));
        vals[nv++] = a;
        vals[nv++] = -a;
    }
    const int npairs = nv * nv, cols = (npairs + F - 1) / F;
    float *act = calloc((size_t)cols * 2 * F, sizeof(float));
    float *gi = calloc((size_t)cols * F, sizeof(float)), *ui = calloc((size_t)cols * F, sizeof(float));
    if (!act || !gi || !ui) { free(act); free(gi); free(ui); return -1.0; }
    for (int p = 0; p < cols * F; p++) {
        const int q = p % npairs, c = p / F, j = p % F;   // the last column repeats the first pairs
        gi[p] = vals[q / nv];
        ui[p] = vals[q % nv];
        act[(size_t)c * 2 * F + j] = gi[p];
        act[(size_t)c * 2 * F + F + j] = ui[p];
    }
    bool ran = true;
    if (ctl) {
        for (int p = 0; p < cols * F; p++) {
            const float g = gi[p], u = ui[p];
            const float sg = sw_tanh_cancel(g / gc) * gc, sig = 1.0f / (1.0f + p_expf(-g));
            const float su = sw_tanh_cancel(u / uc) * uc;
            act[(size_t)(p / F) * 2 * F + p % F] = sg * sig * su;
        }
    } else {
        ran = gpu_ink2_test_swiglu(h, act, F, cols, gc, uc, bounded);
    }
    const double floor_abs = (double)gc * uc * FLT_MIN;
    double worst = ran ? 0.0 : -1.0;
    for (int p = 0; ran && p < cols * F; p++) {
        const double r = sw_ref(gi[p], ui[p], gc, uc);
        const float y = act[(size_t)(p / F) * 2 * F + p % F];
        double e;
        uint32_t yb;
        memcpy(&yb, &y, 4);
        if ((yb & 0x7f800000u) == 0x7f800000u) e = INFINITY;   // not finite, by its bits (-ffast-math)
        else if (r == 0.0) e = y == 0.0f ? 0.0 : INFINITY;
        else if (fabs(y - r) <= floor_abs) e = 0.0;
        else e = fabs(y - r) / fabs(r) / SW_U;
        if (e > worst) { worst = e; *wg = gi[p]; *wu = ui[p]; }
    }
    free(act); free(gi); free(ui);
    return worst;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf TINY_F16.gguf TINY_LORA.gguf\n", argv[0]);
        return 2;
    }
    // D1d.3b: the resident decode step is on by default; every test before (aj) holds the NON-resident route's own
    // parts (its per-call counters, its byte accounting, its host forms), so the file runs with the step off and (aj)
    // / (ak) turn it on for their own instances
    plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "1");
#ifndef RUNNER_GPU_CUDA
    printf("SKIP: this build's GPU backend is not CUDA (not a pass)\n");
    return 77;
#endif
    // the checker's own check (the second read's P1): a row of NaN must score as a failure, never 0
    {
        float nanrow[4], ref4[4] = { 1, -2, 3, -4 };
        uint32_t qnan = 0x7fc00000u;
        for (int i = 0; i < 4; i++) memcpy(nanrow + i, &qnan, sizeof qnan);
        const double dn = rel_diff(nanrow, ref4, 4);
        CHECK(!(dn <= 1e-4), "(0) the row checker scores a NaN row %.2e, not within 1e-4", dn);
        float one[4] = { 1, -2, 3, -4 };   // one NaN among values equal to the reference
        memcpy(one + 1, &qnan, sizeof qnan);
        const double d1 = rel_diff(one, ref4, 4);
        CHECK(!(d1 <= 1e-4), "(0) the row checker scores a row with one NaN %.2e, not within 1e-4", d1);
    }
    char dev[160] = "";
    if (!gpu_available(dev, sizeof dev)) { printf("SKIP: no CUDA device on this host (not a pass)\n"); return 77; }
    size_t fr = 0, tot = 0;
    gpu_mem_info(&fr, &tot);
    printf("device %s: %zu of %zu bytes free\n", dev, fr, tot);
    const char *kind[3] = { "F32", "BF16", "Q8_0" };
    for (int f = 0; f < 3; f++) {
        model_t *g = load(argv[1 + f], true, false, 64), *r = load(argv[1 + f], false, false, 64);
        if (!g || !r) { CHECK(0, "%s: the file loads on the GPU and on the reference route", kind[f]); drop(g); drop(r); continue; }
        // G1 (a)
        CHECK(model_inkling2_cuda(g) && !model_inkling2_cuda(r), "(a) %s: the GPU-allowed load runs on CUDA (%d), the GPU-off one "
              "on the CPU (%d)", kind[f], model_inkling2_cuda(g), model_inkling2_cuda(r));
        // G1 (b)
        int n_t = 0, n_bad = 0;
        char bad[160] = "";
        CHECK(model_inkling2_cuda_readback(g, &n_t, &n_bad, bad, sizeof bad) && n_t > 0 && n_bad == 0,
              "(b) %s: %d device tensors read back, %d differ from the host decode%s%s", kind[f], n_t, n_bad,
              *bad ? ", first " : "", bad);
        // G2 (g): the embedding stays on the host, the head goes up
        CHECK(!model_inkling2_cuda_holds(g, "model.embed_tokens.weight") && model_inkling2_cuda_holds(g, "lm_head.weight"),
              "(g) %s: the token embedding stays on the host (%d on the device), the head goes up (%d)", kind[f],
              model_inkling2_cuda_holds(g, "model.embed_tokens.weight"), model_inkling2_cuda_holds(g, "lm_head.weight"));
        // G2 (c): only rows a CUDA instance computed count (a CPU fallback's rows would pass (c) and (d) for nothing)
        const bool on_cuda = model_inkling2_cuda(g);
        const size_t n = (size_t)N_TOK * (size_t)g->n_vocab;
        const void *sp = NULL;
        model_inkling2_parts(g, &sp, NULL);
        memset(&obs, 0, sizeof obs);
        obs.want_used = sp ? ((const ink2_shape *)sp)->n_used : -1;
        ink2_set_sel_observer(obs_fn, NULL);
        const long hc0 = ink2_attn_host_core_calls(), hg0 = ink2_attn_host_glue_calls();
        float *a = rows(g);
        const long hc1 = ink2_attn_host_core_calls(), hg1 = ink2_attn_host_glue_calls();
        ink2_set_sel_observer(NULL, NULL);
        {   // G2b 2b (r): each application's whole attention block on the device, no host q/k/v/bank work for it
            const int n_app_r = sp ? ((const ink2_shape *)sp)->n_app : -1;
            const long nb = model_inkling2_cuda_block_calls(g);
            CHECK(on_cuda && a && nb == n_app_r && hg1 - hg0 == 0, "(r) %s: %ld device attention-block calls for one forward "
                  "of %d applications, the host's block work %ld times (want 0)", kind[f], nb, n_app_r, hg1 - hg0);
        }
        {   // G2b 2a (m): each application's attention core on the device, the host core not run
            const int n_app_m = sp ? ((const ink2_shape *)sp)->n_app : -1;
            size_t kvb = 0;
            const long na = model_inkling2_cuda_attn_calls(g, &kvb);
            CHECK(on_cuda && a && na == n_app_m && hc1 - hc0 == 0 && kvb > 0, "(m) %s: %ld device attention-core calls for one "
                  "forward of %d applications, the host core %ld times (want 0), the K/V rings %zu bytes on the device",
                  kind[f], na, n_app_m, hc1 - hc0, kvb);
        }
        float *b = rows(r);
        // G2b (j): the one-chunk forward above ran each application's expert body as one device call, and each
        // distinct expert of a call is two single products fewer than the per-product path (TEST-ONLY NO_EXP baseline)
        {
            long n_lin = 0, n_exp = 0, b_lin = 0, b_exp = -1;
            model_inkling2_cuda_calls(g, &n_lin, &n_exp);
            const int n_app = sp ? ((const ink2_shape *)sp)->n_app : -1;
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_EXP", "1");
            model_t *gb = load(argv[1 + f], true, false, 64);
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_EXP", "");
            float *ab = gb ? rows(gb) : NULL;
            if (gb) model_inkling2_cuda_calls(gb, &b_lin, &b_exp);
            CHECK(on_cuda && a && n_exp == n_app && ab && b_exp == 0 && b_lin - n_lin == 2 * obs.distinct,
                  "(j) %s: %ld device expert-body calls for one forward of %d applications; single products %ld, the "
                  "per-product path's %ld (%ld body calls): %ld fewer, want 2 x %ld distinct experts", kind[f], n_exp, n_app,
                  n_lin, b_lin, b_exp, b_lin - n_lin, obs.distinct);
            // the observer's own check: one call per application, n_used experts at every position, margins >= 0
            CHECK(obs.calls == n_app && obs.positions == (long)n_app * N_TOK && obs.bad_count == 0 && obs.bad_margin == 0,
                  "(j) %s: the selection observer saw %ld calls, %ld positions, %ld without %d experts, %ld negative margins",
                  kind[f], obs.calls, obs.positions, obs.bad_count, obs.want_used, obs.bad_margin);
            free(ab);
            drop(gb);
        }
        const double rd = a && b ? rel_diff(a, b, n) : 1;
        const size_t nf = a ? n_nonfinite(a, n) : 0;
        CHECK(on_cuda && a && b && nf == 0 && rd <= 1e-4, "(c) %s: CUDA rows within %.2e of the reference rows, want <= 1e-4; %zu "
              "values not finite, want 0", kind[f], rd, nf);
        CHECK(on_cuda && a && b && memcmp(a, b, sizeof(float) * n), "(c) %s: the CUDA rows are not the reference rows byte for byte",
              kind[f]);
        {   // (d) and (e) on each file
            model_t *g8 = load(argv[1 + f], true, false, 8), *g2 = load(argv[1 + f], true, false, 64);
            model_t *gq = load(argv[1 + f], true, true, 64);
            float *x8 = g8 ? rows(g8) : NULL, *x2 = g2 ? rows(g2) : NULL, *xq = gq ? rows(gq) : NULL;
            CHECK(on_cuda && a && nf == 0 && x8 && !memcmp(a, x8, sizeof(float) * n), "(d) %s: the CUDA rows are byte-identical at -b 64 "
                  "and -b 8", kind[f]);
            CHECK(on_cuda && a && nf == 0 && x2 && !memcmp(a, x2, sizeof(float) * n), "(d) %s: the CUDA rows are byte-identical across two "
                  "loads", kind[f]);
            const double qe = xq && b && !n_nonfinite(xq, n) ? rel_diff(xq, b, n) : 0;   // NaN rows are no control
            CHECK(on_cuda && qe >= 10 * rd && qe > 0 && qe < 1e30, "(e) %s: control: the factor left out %.2e from the reference, CUDA "
                  "%.2e, want >= 10x", kind[f], qe, rd);
            free(x8); free(x2); free(xq); drop(g8); drop(g2); drop(gq);
        }
        {   // G2b 2b (s): -b 1, every chunk after the first a continuation (the K/V convolutions read the device
            // history at each step, the sliding window of 8 wraps over the 24 positions): byte-identical to -b 64,
            // within 1e-4 of the reference engine's rows
            model_t *g1 = load(argv[1 + f], true, false, 1);
            float *x1 = g1 ? rows(g1) : NULL;
            const double d1 = x1 && b ? rel_diff(x1, b, n) : 1;
            // (D1d.3b: the file runs with the step off, so this is the block path's own -b 1 and the step is absent)
            CHECK(on_cuda && g1 && model_inkling2_cuda_block_calls(g1) > 0 && model_inkling2_cuda_step_calls(g1) == -1 && x1 &&
                  a && !memcmp(a, x1, sizeof(float) * n) &&
                  d1 <= 1e-4, "(s) %s: at -b 1 (the history read at every step) the device block's rows equal -b 64's byte "
                  "for byte, %.2e from the reference, want <= 1e-4", kind[f], d1);
            free(x1); drop(g1);
        }
        {   // (v) the CPU route's decode fusion (RUNNER_INK2_DECODE_FUSE, T = 1: the fused projections, the fused expert
            // body, the fused attention branch) never takes a device view's work: at -b 1 (every chunk T = 1), on the
            // block path and on 2a's (test-only NO_BLOCK), with the fusion on and off: one device call per application
            // and position, no host product over a device view, no host core, and the same rows byte for byte
            const int n_app_v = sp ? ((const ink2_shape *)sp)->n_app : -1;
            const long want = (long)n_app_v * N_TOK;
            const char *fe = getenv("RUNNER_INK2_DECODE_FUSE");   // the run's own setting, restored after (v)
            char fu_run[8] = "";
            if (fe) snprintf(fu_run, sizeof fu_run, "%s", fe);
            for (int path = 0; path < 2; path++) {
                float *xv[2] = { NULL, NULL };
                for (int fu = 0; fu < 2; fu++) {
                    plat_setenv("RUNNER_INK2_DECODE_FUSE", fu ? "1" : "0");
                    plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_BLOCK", path ? "1" : "");
                    model_t *gv = load(argv[1 + f], true, false, 1);
                    plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_BLOCK", "");
                    const long hp0 = ink2_host_dev_products(), hc0v = ink2_attn_host_core_calls();
                    xv[fu] = gv ? rows(gv) : NULL;
                    const long hp = ink2_host_dev_products() - hp0, hcv = ink2_attn_host_core_calls() - hc0v;
                    plat_setenv("RUNNER_INK2_DECODE_FUSE", fu_run);
                    long nl = 0, ne = 0;
                    size_t kvb = 0;
                    if (gv) model_inkling2_cuda_calls(gv, &nl, &ne);
                    const long nattn = gv ? (path ? model_inkling2_cuda_attn_calls(gv, &kvb) : model_inkling2_cuda_block_calls(gv)) : -1;
                    CHECK(on_cuda && gv && model_inkling2_cuda(gv) && xv[fu] && nattn == want && ne == want && hp == 0 && hcv == 0,
                          "(v) %s, %s, fusion %s, -b 1: %ld device %s calls and %ld expert-body calls (want %ld each), %ld host "
                          "products over device views and %ld host cores (want 0)", kind[f], path ? "2a's path" : "the block path",
                          fu ? "on" : "off", nattn, path ? "attention-core" : "attention-block", ne, want, hp, hcv);
                    drop(gv);
                }
                const double dv = xv[1] && b ? rel_diff(xv[1], b, n) : 1;
                CHECK(xv[0] && xv[1] && !memcmp(xv[0], xv[1], sizeof(float) * n) && dv <= 1e-4, "(v) %s, %s: the rows with the "
                      "fusion on equal its rows with it off byte for byte, %.2e from the reference, want <= 1e-4", kind[f],
                      path ? "2a's path" : "the block path", dv);
                free(xv[0]); free(xv[1]);
            }
        }
        {   // (w) the default chunk (256 since the CPU route's R4.48.20) on CUDA, at -c 384 over 280 tokens (a chunk
            // boundary inside: 256 + 24): -b 0 resolves to 256; the rows at -b 256, 64 and 8 byte-identical and within
            // 1e-4 of the reference engine's; the chunk boundaries live (a test-only bump at each chunk's first position
            // past 0 moves -b 256's rows away from -b 64's); every sliding K/V ring window - 1 + 256 rows (a full one
            // -c); the products' output buffer T = 256 rows of the widest product but the head and one head row, and a
            // keep-all forward's head (260 rows, T = 256) leaves it at that size (its columns in groups)
            enum { NL = 280, CL = 384 };
            int32_t *lp = malloc(sizeof(int32_t) * NL);
            for (int i = 0; lp && i < NL; i++) lp[i] = (int32_t)((i * 37 + 11) % 256);
            const size_t nl = (size_t)NL * (size_t)g->n_vocab;
            ctx_len = CL;
            model_t *w0 = load(argv[1 + f], true, false, 0), *w64 = load(argv[1 + f], true, false, 64);
            model_t *w8 = load(argv[1 + f], true, false, 8), *wr = load(argv[1 + f], false, false, 0);
            ctx_len = 0;
            const void *wsp = NULL;
            if (w0) model_inkling2_parts(w0, &wsp, NULL);
            const ink2_shape *ws = wsp;
            const size_t y0 = model_inkling2_cuda_lin_out_bytes(w0);
            float *l0 = w0 && lp ? rows_n(w0, lp, NL) : NULL, *l64 = w64 && lp ? rows_n(w64, lp, NL) : NULL;
            float *l8 = w8 && lp ? rows_n(w8, lp, NL) : NULL, *lr = wr && lp ? rows_n(wr, lp, NL) : NULL;
            const size_t y1 = model_inkling2_cuda_lin_out_bytes(w0);
            const double dl = l0 && lr ? rel_diff(l0, lr, nl) : 1;
            const size_t nfl = l0 ? n_nonfinite(l0, nl) : 0;
            CHECK(w0 && w0->n_batch == 256 && model_inkling2_cuda(w0) && l0 && l64 && l8 && lr && nfl == 0 &&
                  !memcmp(l0, l64, sizeof(float) * nl) && !memcmp(l0, l8, sizeof(float) * nl) && dl <= 1e-4,
                  "(w) %s, %d tokens at -c %d: -b 0 is -b %d (want 256); the CUDA rows at -b 256, 64 and 8 byte-identical, "
                  "%.2e from the reference, want <= 1e-4; %zu not finite", kind[f], NL, CL, w0 ? w0->n_batch : -1, dl, nfl);
            // the control: the bump at each chunk's first position past 0 (256 at -b 256; 64, 128, ... at -b 64)
            ink2_set_test_chunk_bump(1e-2f);
            ctx_len = CL;
            model_t *c0 = load(argv[1 + f], true, false, 0), *c64 = load(argv[1 + f], true, false, 64);
            ctx_len = 0;
            float *k0 = c0 && lp ? rows_n(c0, lp, NL) : NULL, *k64 = c64 && lp ? rows_n(c64, lp, NL) : NULL;
            ink2_set_test_chunk_bump(0.0f);
            CHECK(k0 && k64 && memcmp(k0, k64, sizeof(float) * nl), "(w) %s: control: a bump at each chunk's start moves "
                  "-b 256's rows away from -b 64's (the boundaries are live)", kind[f]);
            // the sliding rings: window - 1 + 256 rows; a full layer's -c; plus each layer's two histories (conv_k - 1)
            size_t want_kv = 0, kvb = 0;
            for (int ap = 0; ws && ap < ws->n_app; ap++) {
                const int gl = ink2_app_global(ws, ap);
                const size_t C = (size_t)(gl ? ws->n_kv * ws->head_dim : ws->swa_n_kv * ws->swa_head_dim);
                const size_t rws = gl || ws->window - 1 + 256 >= CL ? (size_t)CL : (size_t)(ws->window - 1 + 256);
                want_kv += 2 * sizeof(float) * C * (rws + (size_t)(ws->conv_k - 1));
            }
            if (w0) model_inkling2_cuda_attn_calls(w0, &kvb);
            CHECK(ws && kvb == want_kv && want_kv > 0, "(w) %s: the device K/V rings and histories at -b 256: %zu bytes, want "
                  "%zu (each sliding ring window %d - 1 + 256 rows)", kind[f], kvb, want_kv, ws ? ws->window : -1);
            // the output buffer: 256 rows of the tiny file's widest product but the head (one expert's gate_up, 2 x ff
            // = 128) beside one head row (260): 256 x 128 floats; the keep-all head over 256 positions did not grow it
            const size_t want_y = ws ? sizeof(float) * (256 * (size_t)(2 * ws->ff) > (size_t)ws->vocab_used
                                                            ? 256 * (size_t)(2 * ws->ff) : (size_t)ws->vocab_used) : 0;
            CHECK(ws && y0 == want_y && y1 == y0, "(w) %s: the products' output buffer at -b 256 %zu bytes (want %zu: T rows "
                  "of the widest product but the head, at least one head row; T x the head's %d rows would be %zu), %zu "
                  "after a keep-all forward (want unchanged)", kind[f], y0, want_y, ws ? ws->vocab_used : -1,
                  ws ? sizeof(float) * 256 * (size_t)ws->vocab_used : 0, y1);
            free(l0); free(l64); free(l8); free(lr); free(k0); free(k64); free(lp);
            drop(w0); drop(w64); drop(w8); drop(wr); drop(c0); drop(c64);
        }
        free(a); free(b); drop(g); drop(r);
    }
    // G2 (g): the F16 file (its projections stay on the host), and --cpu-moe on the BF16 file (its stacks stay)
    for (int k = 0; k < 2; k++) {
        const char *path = k ? argv[2] : argv[4], *what = k ? "BF16 --cpu-moe" : "F16";
        const char *stack = "model.layers.0.mlp.experts.gate_up_proj", *proj = "model.layers.0.self_attn.q_proj.linear.weight";
        cpu_moe = k == 1;
        model_t *g = load(path, true, false, 64);
        cpu_moe = false;
        model_t *r = load(path, false, false, 64);
        if (!g || !r) { CHECK(0, "(g) %s: the file loads on the GPU and on the reference route", what); drop(g); drop(r); continue; }
        const size_t n = (size_t)N_TOK * (size_t)g->n_vocab;
        float *a = rows(g), *b = rows(r);
        const double rd = a && b ? rel_diff(a, b, n) : 1;
        const size_t nf = a ? n_nonfinite(a, n) : 0;
        const bool held = k ? model_inkling2_cuda_holds(g, stack) : model_inkling2_cuda_holds(g, proj);
        CHECK(model_inkling2_cuda(g) && !held && a && b && nf == 0 && rd <= 1e-4,
              "(g) %s: on CUDA (%d) with %s on the host (%d on the device); rows %s, within %.2e of the reference, want "
              "<= 1e-4; %zu not finite", what, model_inkling2_cuda(g), k ? "the expert stacks" : "the F16 projections",
              held, a ? "given" : "refused", rd, nf);
        if (k == 1) {   // G2b (l): --cpu-moe runs no device expert body
            long n_exp = -1;
            model_inkling2_cuda_calls(g, NULL, &n_exp);
            CHECK(a && n_exp == 0, "(l) --cpu-moe: %ld device expert-body calls, want 0", n_exp);
        }
        free(a); free(b); drop(g); drop(r);
    }
    // G2 (f), on the BF16 file: a bind that succeeds first, so the process's device hook is set, then partial uploads
    {
        model_t *ok = load(argv[2], true, false, 64), *cpu = load_route(argv[2], false, false, 64, "");
        int n_t = 0, n_bad = 0;
        const bool bound = ok && model_inkling2_cuda(ok) && model_inkling2_cuda_readback(ok, &n_t, &n_bad, NULL, 0) && n_t > 1;
        float *want = cpu ? rows(cpu) : NULL;
        CHECK(bound && want, "(f) a first bind succeeds in this process (%d device tensors) and the fast CPU route gives rows", n_t);
        const int at[4] = { 0, 1, n_t / 2, n_t - 1 };
        for (int k = 0; bound && want && k < 4; k++) {
            char v[32];
            snprintf(v, sizeof v, "%d", at[k]);
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_PUT_FAIL", v);
            model_t *pf = load(argv[2], true, false, 64);
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_PUT_FAIL", "");
            const size_t n = pf ? (size_t)N_TOK * (size_t)pf->n_vocab : 0;
            float *got = pf ? rows(pf) : NULL;
            const size_t bad = got ? n_nonfinite(got, n) : 0;
            const bool same = got && !memcmp(got, want, sizeof(float) * n);
            CHECK(pf && !model_inkling2_cuda(pf) && (!got || (bad == 0 && same)),
                  "(f) the put of tensor %d of %d fails: CPU fallback (%d), rows %s, %zu not finite, %s the fast CPU rows",
                  at[k], n_t, pf ? !model_inkling2_cuda(pf) : -1, got ? "given" : "refused", bad, same ? "equal to" : "not");
            free(got);
            drop(pf);
        }
        free(want);
        drop(ok);
        drop(cpu);
    }
    // G2 (h), on the BF16 file: the first device product fails; that forward and the next refuse. (Since D1a the
    // message gates' products run inside the device message step, so a forward of the tiny file makes one product
    // through the product hook, the head's: the injection takes the first; (y) injects a failed message step.)
    {
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_LIN_FAIL", "0");
        model_t *g = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_LIN_FAIL", "");
        const bool on = g && model_inkling2_cuda(g);
        // the thread's pool binding before the forward (ink2_set_tpool returns the one it replaces)
        struct tpool *tp0 = ink2_set_tpool(NULL);
        ink2_set_tpool(tp0);
        const float *first = on ? model_inkling2_forward(g, PROMPT, N_TOK, 0, true, true) : NULL;
        struct tpool *tp1 = ink2_set_tpool(tp0);
        const float *again = on ? model_inkling2_forward(g, PROMPT, 1, 0, true, true) : NULL;
        CHECK(on && !first && !again, "(h) a failed device product: on CUDA (%d), that forward %s, the next %s", on,
              first ? "gave rows" : "refused", again ? "gave rows" : "refused");
        // the rebase onto 317a128c: the forward that fails on the device gives the thread back its pool binding, as
        // every other return of the forward does (left bound, the thread kept this instance's pool past its life)
        CHECK(on && tp1 == tp0, "(h) after the forward that failed on the device, the thread's pool binding is the one "
              "it had before (%s)", tp1 == tp0 ? "restored" : "LEFT BOUND to the instance's pool");
        drop(g);
        // the same inside the expert body (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_EXP_FAIL=k: the k-th body call fails)
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_EXP_FAIL", "2");
        g = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_EXP_FAIL", "");
        long ne = 0;
        const bool on2 = g && model_inkling2_cuda(g);
        const float *f2 = on2 ? model_inkling2_forward(g, PROMPT, N_TOK, 0, true, true) : NULL;
        const float *a2 = on2 ? model_inkling2_forward(g, PROMPT, 1, 0, true, true) : NULL;
        if (g) model_inkling2_cuda_calls(g, NULL, &ne);
        CHECK(on2 && ne >= 3 && !f2 && !a2, "(h) a failed device expert body (call %ld reached): that forward %s, the next %s",
              ne, f2 ? "gave rows" : "refused", a2 ? "gave rows" : "refused");
        drop(g);
    }
    // G2 (i): each device kernel through gpu.h, against float64 sums over dequant_row's decode
    {
        CHECK(ggml_type_size(T_Q8_0) == 34 && ggml_block_size(T_Q8_0) == 32 && ggml_type_size(T_NVFP4) == 36 &&
              ggml_block_size(T_NVFP4) == 64 && ggml_type_size(T_IQ3_XXS) == 98 && ggml_block_size(T_IQ3_XXS) == 256,
              "(i) the block layouts written here: Q8_0 34 bytes a 32-value block, NVFP4 36 a 64, IQ3_XXS 98 a 256");
        enum { NI = 512, NR = 40, R0 = 5, NO = 24, NT = 3 };   // 24 rows: several blocks of every launch shape
        const struct { int type; const char *name; float scale; } ty[] = {
            { T_F32, "F32", 1.0f }, { T_BF16, "BF16", 1.0f }, { T_Q8_0, "Q8_0", 1.0f }, { T_NVFP4, "NVFP4", 1.0f },
            { T_NVFP4, "NVFP4 factor 0.75", 0.75f }, { T_IQ3_XXS, "IQ3_XXS", 1.0f },
        };
        uint32_t seed = 12345u;
#define RND() (seed = seed * 1664525u + 1013904223u, seed >> 8)
        float x[NT * NI];
        for (int i = 0; i < NT * NI; i++) x[i] = (float)((int)(RND() % 2001u) - 1000) / 1000.0f;
        gpu_ink2 *h = gpu_ink2_bind();
        CHECK(h && !gpu_ink2_runs(h, T_F16), "(i) a handle binds, and its products take no F16 (the F16 file's premise)");
        for (size_t k = 0; h && k < sizeof ty / sizeof *ty; k++) {
            const int type = ty[k].type;
            const size_t rb = (size_t)NI / (size_t)ggml_block_size(type) * ggml_type_size(type);
            unsigned char *w = calloc(NR, rb);
            for (int r = 0; w && r < NR; r++) {
                unsigned char *row = w + (size_t)r * rb;
                for (size_t b = 0; b < rb; b++) row[b] = (unsigned char)RND();
                if (type == T_F32 || type == T_BF16)
                    for (int i = 0; i < NI; i++) {
                        const float v = (float)((int)(RND() % 2001u) - 1000) / 4000.0f;
                        uint32_t u;
                        memcpy(&u, &v, sizeof u);
                        if (type == T_F32) memcpy(row + 4 * (size_t)i, &u, 4);
                        else { const uint16_t hb = (uint16_t)(u >> 16); memcpy(row + 2 * (size_t)i, &hb, 2); }
                    }
                else if (type == T_Q8_0)
                    for (size_t b = 0; b < rb; b += 34) { row[b] = 0x1f; row[b + 1] = 0x21; }   // d = f16 0x211f (0.01)
                else if (type == T_NVFP4)
                    for (size_t b = 0; b < rb; b += 36) memset(row + b, 0x30, 4);   // each sub-block's UE4M3 scale 0.5
                else
                    for (size_t b = 0; b < rb; b += 98) { row[b] = 0x1f; row[b + 1] = 0x21; }   // d = 0.01
            }
            double ref[NT * NO], top = 0;
            float dq[NI], y[NT * NO];
            for (int o = 0; w && o < NO; o++) {
                dequant_row(type, w + (size_t)(R0 + o) * rb, dq, NI);
                for (int t = 0; t < NT; t++) {
                    double s = 0;
                    for (int i = 0; i < NI; i++) s += (double)dq[i] * x[t * NI + i];
                    ref[t * NO + o] = s * ty[k].scale;
                    if (fabs(ref[t * NO + o]) > top) top = fabs(ref[t * NO + o]);
                }
            }
            const int idx = w ? gpu_ink2_put(h, w, (size_t)NR * rb, NULL, 0) : -1;
            const bool ran = idx >= 0 && gpu_ink2_runs(h, type) &&
                             gpu_ink2_lin(h, idx, type, rb, R0, NO, NI, NT, x, y, ty[k].scale);
            double d = 0;
            for (int i = 0; ran && i < NT * NO; i++)
                if (fabs(y[i] - ref[i]) > d) d = fabs(y[i] - ref[i]);
            const size_t nf = ran ? n_nonfinite(y, NT * NO) : 0;
            const double rel = top > 0 ? d / top : d;
            CHECK(ran && nf == 0 && top > 0 && rel <= 1e-4, "(i) %s: the device product of rows %d..%d over %d columns %s, within "
                  "%.2e of the float64 sums (largest %.3g), want <= 1e-4; %zu not finite", ty[k].name, R0, R0 + NO - 1, NT,
                  ran ? "ran" : "failed", rel, top, nf);
            free(w);
        }
#undef RND
        gpu_ink2_free(h);
    }
    // (q): LoRA on the CUDA route, against the CPU fast route with the same adapter
    {
        // an active adapter keeps 2a's path (its terms are host code; G2b 2b), so the base it is held to byte for byte
        // at scale 0 runs that path too (test-only RUNNER_INKLING2_CUDA_TEST_NO_BLOCK=1): the same path, the same bits
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_BLOCK", "1");
        model_t *gb = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_BLOCK", "");
        model_t *gl = load(argv[2], true, false, 64);
        model_t *g0 = load(argv[2], true, false, 64), *cl = load_route(argv[2], false, false, 64, "");
        const bool l1 = gl && model_lora_load(gl, argv[5], 1.0f), l0 = g0 && model_lora_load(g0, argv[5], 0.0f);
        const bool lc = cl && model_lora_load(cl, argv[5], 1.0f);
        const bool on = gb && gl && g0 && model_inkling2_cuda(gb) && model_inkling2_cuda(gl) && model_inkling2_cuda(g0);
        const size_t n = gb ? (size_t)N_TOK * (size_t)gb->n_vocab : 0;
        float *rb = on ? rows(gb) : NULL, *rl = on && l1 ? rows(gl) : NULL, *r0 = on && l0 ? rows(g0) : NULL;
        float *rc = lc ? rows(cl) : NULL;
        const double dq = rl && rc ? rel_diff(rl, rc, n) : 1, mv = rl && rb ? rel_diff(rl, rb, n) : 0;
        const size_t nf = rl ? n_nonfinite(rl, n) : 0;
        CHECK(on && l1 && lc && rl && rc && nf == 0 && dq <= 1e-4, "(q) the adapter on CUDA (loaded %d, CPU %d): rows within "
              "%.2e of the CPU fast route's with the same adapter, want <= 1e-4; %zu not finite", l1, lc, dq, nf);
        CHECK(rl && rb && mv >= 10 * dq && mv > 0 && mv < 1e30, "(q) the adapter moves the CUDA rows %.2e, want >= 10x %.2e", mv, dq);
        CHECK(r0 && rb && !memcmp(r0, rb, sizeof(float) * n), "(q) the adapter at scale 0 (loaded %d): the CUDA rows equal the "
              "base CUDA rows byte for byte", l0);
        free(rb); free(rl); free(r0); free(rc);
        drop(gb); drop(gl); drop(g0); drop(cl);
    }
    // G2b (k): gpu_ink2_experts against float64 on the host, SwiGLU's boundary inputs reached through one-hot x
    {
        // the real model's widths (H 2304, F 1152: every kernel runs more than one block in x), a fifth expert no
        // position chooses (n = 0), and gate rows unlike their up rows (independent random codes; the scale class
        // cycles by 7, which F does not divide), so a kernel that swaps or mixes up gate and up cannot pass
        enum { KH = 2304, KF = 1152, KE = 5, KT = 6 };
        const size_t rb13 = (size_t)KH / 64 * 36, rb2 = (size_t)KF / 64 * 36;
        const float s13[KE] = { 1e-3f, 1.0f, 10.0f, 0.05f, 1.0f }, s2[KE] = { 1.0f, 0.5f, 2.0f, 1.0f, 1.0f };
        const float gcap = 4.0f, ucap = 8.0f;
        unsigned char *w13 = calloc((size_t)KE * 2 * KF, rb13), *w2 = calloc((size_t)KE * KH, rb2);
        uint32_t seed = 777u;
#define RND() (seed = seed * 1664525u + 1013904223u, seed >> 8)
        for (int r = 0; w13 && r < KE * 2 * KF; r++) {   // gate_up: all 16 codes at random, the scales' extremes
            unsigned char *row = w13 + (size_t)r * rb13;
            const int cls = r % 7;
            const unsigned char sc = cls == 0 ? 0x7e : cls == 1 ? 0x01 : 0x38;   // 448, the smallest, 1.0
            for (size_t b = 0; b < rb13; b += 36) {
                memset(row + b, sc, 4);
                for (int j = 0; j < 32; j++) row[b + 4 + j] = cls == 2 ? 0 : (unsigned char)RND();   // class 2: zero block
            }
        }
        for (int r = 0; w2 && r < KE * KH; r++) {   // down: scale 1.0, codes from the generator
            unsigned char *row = w2 + (size_t)r * rb2;
            for (size_t b = 0; b < rb2; b += 36) {
                memset(row + b, 0x38, 4);
                for (int j = 0; j < 32; j++) row[b + 4 + j] = (unsigned char)RND();
            }
        }
#undef RND
        static float x[KT * KH], out[KT * KH];
        float wt[KT * KE];
        unsigned char ch[KT * KE];
        memset(x, 0, sizeof x); memset(wt, 0, sizeof wt); memset(ch, 0, sizeof ch);
        for (int t = 0; t < KT; t++) {
            x[t * KH + (7 * t + 1) % KH] = 1.0f;   // one-hot: each gate value is one decoded weight times the factor
            for (int j = 0; j < 2; j++) {
                const int e = (t + j) % 4;   // experts 0..3; expert 4 has no positions
                ch[t * KE + e] = 1;
                wt[t * KE + e] = 0.25f + 0.1f * (float)e;
            }
        }
        // the reference, float64 over dequant_row's decode, experts in ascending index
        static double ref[KT * KH], act[KF];
        static float row[KH], gu[2 * KF];
        memset(ref, 0, sizeof ref);
        for (int t = 0; w13 && w2 && t < KT; t++)
            for (int e = 0; e < KE; e++) {
                if (!ch[t * KE + e]) continue;
                for (int r = 0; r < 2 * KF; r++) {
                    dequant_row(T_NVFP4, w13 + ((size_t)e * 2 * KF + r) * rb13, row, KH);
                    double s = 0;
                    for (int i = 0; i < KH; i++) s += (double)row[i] * x[t * KH + i];
                    gu[r] = (float)(s * s13[e]);
                }
                for (int j = 0; j < KF; j++) {
                    const double g = gu[j], u = gu[KF + j];
                    act[j] = tanh(g / gcap) * gcap * (1.0 / (1.0 + exp(-g))) * tanh(u / ucap) * ucap;
                }
                for (int o = 0; o < KH; o++) {
                    dequant_row(T_NVFP4, w2 + ((size_t)e * KH + o) * rb2, row, KF);
                    double s = 0;
                    for (int j = 0; j < KF; j++) s += (double)row[j] * act[j];
                    ref[t * KH + o] += wt[t * KE + e] * s * s2[e];
                }
            }
        gpu_ink2 *h = gpu_ink2_bind();
        const int i13 = h && w13 ? gpu_ink2_put(h, w13, (size_t)KE * 2 * KF * rb13, s13, KE) : -1;
        const int i2 = h && w2 ? gpu_ink2_put(h, w2, (size_t)KE * KH * rb2, s2, KE) : -1;
        const bool ran = i13 >= 0 && i2 >= 0 && gpu_ink2_experts_ok(h) &&
                         gpu_ink2_experts(h, i13, i2, T_NVFP4, T_NVFP4, rb13, rb2, KH, KF, KE, KT, x, ch, wt, s13, s2,
                                          gcap, ucap, out);
        double worst = 0, least_top = 1e30;
        size_t nf = 0;
        for (int t = 0; ran && t < KT; t++) {
            double top = 0, d = 0;
            for (int o = 0; o < KH; o++) {
                if (fabs(ref[t * KH + o]) > top) top = fabs(ref[t * KH + o]);
                if (fabs(out[t * KH + o] - ref[t * KH + o]) > d) d = fabs(out[t * KH + o] - ref[t * KH + o]);
            }
            const double rel = top > 0 ? d / top : 1e30;
            if (rel > worst) worst = rel;
            if (top < least_top) least_top = top;
        }
        if (ran) nf = n_nonfinite(out, KT * KH);
        CHECK(ran && nf == 0 && worst <= 1e-4 && least_top > 1e-6, "(k) the device expert body %s (kernels in the module: %d), "
              "within %.2e of the float64 sums at its worst position, want <= 1e-4; the smallest position's largest "
              "reference value %.3g, want > 1e-6; %zu not finite", ran ? "ran" : "did not run", gpu_ink2_experts_ok(h), worst,
              least_top, nf);
        // the scatter's uniqueness guard (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_DUP_IDX=1 repeats a position in one list)
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_DUP_IDX", "1");
        const bool dup_ran = i13 >= 0 && i2 >= 0 &&
                             gpu_ink2_experts(h, i13, i2, T_NVFP4, T_NVFP4, rb13, rb2, KH, KF, KE, KT, x, ch, wt, s13, s2,
                                              gcap, ucap, out);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_DUP_IDX", "");
        CHECK(i13 >= 0 && !dup_ran, "(k) a launch's position list with a repeat: the call %s", dup_ran ? "RAN" : "refused");
        gpu_ink2_free(h);
        free(w13); free(w2);
    }
    // P1a/P1b (i): a BF16 or NVFP4 product batched from 16 columns equals its per-column launches bit for bit
    {
        static const int Ts[] = { 1, 15, 16, 17, 31, 33, 64, 100 };
        static const struct { int type; const char *name; } ty[] = { { T_BF16, "BF16" }, { T_NVFP4, "NVFP4 (factor 0.75)" } };
        for (size_t u = 0; u < 2; u++)
            for (size_t k = 0; k < sizeof Ts / sizeof *Ts; k++) {
                long nv = 0;
                const long d = lin_identity(ty[u].type, 512, 384, Ts[k], &nv);
                CHECK(d == 0 && nv > 0, "(i) a %s product (512 -> 384) at %d columns, batched from 16: %ld of %ld values "
                      "differ in their bits from the per-column launches' (want 0)", ty[u].name, Ts[k], d, nv);
            }
        long nv = 0;
        long d = lin_identity(T_BF16, 2304, 3072, 48, &nv);
        CHECK(d == 0 && nv > 0, "(i) a BF16 product at q_proj's widths (2304 -> 3072), 48 columns: %ld of %ld values "
              "differ in their bits from the per-column launches' (want 0)", d, nv);
        d = lin_identity(T_NVFP4, 2304, 2304, 48, &nv);
        CHECK(d == 0 && nv > 0, "(i) an NVFP4 product at gate_up's widths (2304 -> 2304), 48 columns: %ld of %ld values "
              "differ in their bits from the per-column launches' (want 0)", d, nv);
    }
    // D1a (y): the loop's message gates on the device. First the ported tanh against pmath.h's p_tanhf over its three
    // branches and their edges (|x| > 20, |x| < 1e-4 and the boundaries' neighbours), the signed zeros, subnormals,
    // the infinities and 4000 random values; a NaN in gives a NaN out on both (their payloads are the hardware's, not
    // compared)
    {
        enum { NV = 4096 };
        static float xv[NV], yd[NV];
        int n = 0;
        const float edges[] = { 0.0f, -0.0f, 20.0f, -20.0f, 1e-4f, -1e-4f, 1e-30f, -1e-30f, 1e-45f, -1e-45f, 0.5f,
                                -0.5f, 1.0f, -1.0f, 9.0f, -9.0f, 19.99f, -19.99f, 1e10f, -1e10f, INFINITY, -INFINITY, NAN };
        for (size_t k = 0; k < sizeof edges / sizeof *edges; k++) xv[n++] = edges[k];
        const float near[] = { 20.0f, -20.0f, 1e-4f, -1e-4f };
        for (size_t k = 0; k < 4; k++) { xv[n++] = nextafterf(near[k], 0.0f); xv[n++] = nextafterf(near[k], near[k] * 2.0f); }
        uint32_t seed = 99u;
        while (n < NV) {
            seed = seed * 1664525u + 1013904223u;
            const float u = (float)((int)(seed >> 9) % 2000001 - 1000000) / 1000000.0f;
            xv[n] = (n % 3 == 0) ? u * 25.0f : (n % 3 == 1) ? u * 2e-4f : u * 3.0f;
            n++;
        }
        gpu_ink2 *h = gpu_ink2_bind();
        const bool ran = h && gpu_ink2_test_ptanh(h, xv, yd, NV);
        long nd = 0;
        // the reference with the CPU's flush-to-zero and denormals-are-zero modes OFF: a binary linked with -ffast-math
        // (this test, and the default build's runner) starts with them on (crtfastmath), which flushes a subnormal
        // operand even in inkling2.o's strictly compiled p_tanhf; the device kernels (-ftz=false) and the strict T3
        // build do not flush. NaN by its bits (-ffast-math folds x != x).
#if defined(__x86_64__) || defined(__i386__)
        const unsigned csr = _mm_getcsr();
        _mm_setcsr(csr & ~0x8040u);   // FTZ (bit 15) and DAZ (bit 6) off
#endif
        for (int i = 0; ran && i < NV; i++) {
            const float yh = ink2_test_p_tanhf(xv[i]);   // the engine's own build of it (this file has -ffast-math)
            uint32_t bh, bd;
            memcpy(&bh, &yh, 4); memcpy(&bd, yd + i, 4);
            const bool nh = (bh & 0x7f800000u) == 0x7f800000u && (bh & 0x007fffffu), nd_ = (bd & 0x7f800000u) == 0x7f800000u && (bd & 0x007fffffu);
            if (nh || nd_) { nd += nh != nd_; continue; }
            nd += bh != bd;
        }
#if defined(__x86_64__) || defined(__i386__)
        _mm_setcsr(csr);
#endif
        CHECK(ran && nd == 0, "(y) the device's p_tanhf over %d values (its three branches, their edges, zeros, "
              "subnormals, infinities, NaN): %ld differ in their bits from pmath.h's (want 0)", NV, nd);
        gpu_ink2_free(h);
    }
    // R4.48.9.2 (sw): the expert body's own SwiGLU on the device, k_ink2_swiglu and the fused body's k_ink2_swiglu_n,
    // within 20 u of pmath.h's double forms (tests/test_inkling2_vmath.c's rule: two softcaps of a correctly rounded
    // division, a tanh and a multiply, 6 u each; the sigmoid's exp, add and division, 6 u; two multiplies; with a
    // transcendental at 4 u, the CUDA C Programming Guide's 2 ulp for tanhf and expf), at Inkling's caps and the tiny
    // files', F 64 and 67 (the columns past F); then the control, the cancelling float32 tanh, past the bound
    {
        gpu_ink2 *h = gpu_ink2_bind();
        const float caps[][2] = { { 4.0f, 25.0f }, { 1.0f, 2.0f } };   // Inkling 2 Mini's, the tiny files'
        for (int bounded = 0; bounded < 2; bounded++)
            for (int c = 0; c < 2; c++)
                for (int F = 64; F <= 67; F += 3) {
                    float wg = 0, wu = 0;
                    const double w = sw_worst(h, caps[c][0], caps[c][1], F, bounded, 0, &wg, &wu);
                    CHECK(w >= 0.0 && w <= 20.0, "(sw) the device's %s, caps %g/%g, F %d, 6889 (g, u) pairs: worst "
                          "%.2f u (g %.3g u %.3g) <= 20 u of pmath.h's double forms",
                          bounded ? "k_ink2_swiglu_n" : "k_ink2_swiglu", caps[c][0], caps[c][1], F, w, wg, wu);
                }
        float wg = 0, wu = 0;
        const double w = sw_worst(h, 4.0f, 25.0f, 64, 0, 1, &wg, &wu);
        CHECK(w > 20.0, "(sw) CONTROL: the kernel's statements with the cancelling float32 tanh fail the sweep (worst "
              "%.2f u, g %.3g u %.3g)", w, wg, wu);
        gpu_ink2_free(h);
    }
    {
        static const struct { const char *what; int H, r, T, n_roles; bool shared; } mc[] = {
            { "q/k/v (a full layer's three reads), 1 position", 256, 64, 1, 3, false },
            { "q/k/v on one read (a sliding layer), 1 position", 256, 64, 1, 3, true },
            { "mlp, 5 positions", 256, 64, 5, 1, false },
            { "q/k/v, 20 positions (the products batched)", 256, 64, 20, 3, false },
            { "mlp, 20 positions", 256, 64, 20, 1, false },
            { "q/k/v at the real widths (H 2304, rank 64), 1 position", 2304, 64, 1, 3, false },
        };
        for (size_t k = 0; k < sizeof mc / sizeof *mc; k++) {
            long nv = 0;
            const long d = msg_identity(mc[k].H, mc[k].r, mc[k].T, mc[k].n_roles, mc[k].shared, &nv);
            CHECK(d == 0 && nv > 0, "(y) the device message step, %s: %ld of %ld values differ in their bits from the "
                  "host's ink2_message_steps_c (its products on the device, as the route runs it; want 0)", mc[k].what, d, nv);
        }
    }
    // (y) the model: the loop's message steps on the device give the rows of the host message steps byte for byte
    // (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_NO_MSG=1, the baseline), at -b 64 and -b 1; two device steps for every
    // application with messages (the loop's later passes) a forward; the host message path 0 times
    {
        model_t *gm = load(argv[2], true, false, 64), *gm1 = load(argv[2], true, false, 1);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_MSG", "1");
        model_t *gh = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_MSG", "");
        const void *ysp = NULL;
        if (gm) model_inkling2_parts(gm, &ysp, NULL);
        const ink2_shape *ys = ysp;
        int n_msg_app = 0;
        for (int a = 0; ys && a < ys->n_app; a++) {
            const int b = ys->app_block[a];
            n_msg_app += b >= ys->loop_start && b < ys->loop_end && ys->app_pass[a] > 0;
        }
        const long hm0 = ink2_msg_host_calls();
        float *rm = gm ? rows(gm) : NULL, *rm1 = gm1 ? rows(gm1) : NULL;
        const long hm1 = ink2_msg_host_calls();
        float *rh = gh ? rows(gh) : NULL;
        const size_t n = gm ? (size_t)N_TOK * (size_t)gm->n_vocab : 0;
        const long c64 = model_inkling2_cuda_msg_calls(gm), c1 = model_inkling2_cuda_msg_calls(gm1);
        CHECK(rm && rm1 && rh && n_msg_app > 0 && !memcmp(rm, rh, sizeof(float) * n) && !memcmp(rm1, rh, sizeof(float) * n) &&
              c64 == 2L * n_msg_app && c1 == 2L * n_msg_app * N_TOK && hm1 - hm0 == 0,
              "(y) the model's message steps on the device: rows equal the host steps' byte for byte at -b 64 and -b 1; "
              "%ld and %ld device steps (want %d and %d: two for each of the %d applications with messages a forward); "
              "the host message path %ld times (want 0)", c64, c1, 2 * n_msg_app, 2 * n_msg_app * N_TOK, n_msg_app,
              hm1 - hm0);
        free(rm); free(rm1); free(rh);
        drop(gm); drop(gm1); drop(gh);
        // a failed device message step (TEST-ONLY RUNNER_INKLING2_CUDA_TEST_MSG_FAIL=0: the first) refuses that forward
        // and the next, as a failed product does
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_MSG_FAIL", "0");
        model_t *gf = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_MSG_FAIL", "");
        const bool onf = gf && model_inkling2_cuda_msg_calls(gf) >= 0;
        const float *f1 = onf ? model_inkling2_forward(gf, PROMPT, N_TOK, 0, true, true) : NULL;
        const float *f2 = onf ? model_inkling2_forward(gf, PROMPT, 1, 0, true, true) : NULL;
        CHECK(onf && !f1 && !f2, "(y) a failed device message step: that forward %s, the next %s",
              f1 ? "gave rows" : "refused", f2 ? "gave rows" : "refused");
        drop(gf);
    }
    // D1b (z) B1: the device router bit for bit against the host's, its p_expf port first (the engine's own build of
    // p_expf, FTZ / DAZ off, over its clamps, their neighbours, zeros, subnormals, infinities, NaN and random values)
    {
        enum { NV = 4096 };
        static float xv[NV], yd[NV];
        int n = 0;
        const float edges[] = { 0.0f, -0.0f, 89.0f, -104.0f, 88.72f, -87.3f, -103.9f, 100.0f, -200.0f, 1e-30f, -1e-45f,
                                1.0f, -1.0f, INFINITY, -INFINITY, NAN };
        for (size_t k = 0; k < sizeof edges / sizeof *edges; k++) xv[n++] = edges[k];
        xv[n++] = nextafterf(89.0f, 0.0f); xv[n++] = nextafterf(89.0f, 100.0f);
        xv[n++] = nextafterf(-104.0f, 0.0f); xv[n++] = nextafterf(-104.0f, -200.0f);
        uint32_t seed = 77u;
        while (n < NV) {
            seed = seed * 1664525u + 1013904223u;
            xv[n] = (float)((int)(seed >> 9) % 2000001 - 1000000) / 1000000.0f * (n % 2 ? 110.0f : 5.0f);
            n++;
        }
        gpu_ink2 *h = gpu_ink2_bind();
        const bool ran = h && gpu_ink2_test_pexpf(h, xv, yd, NV);
        long nd = 0;
#if defined(__x86_64__) || defined(__i386__)
        const unsigned csr = _mm_getcsr();
        _mm_setcsr(csr & ~0x8040u);
#endif
        for (int i = 0; ran && i < NV; i++) {
            const float yh = ink2_test_p_expf(xv[i]);
            uint32_t bh, bd;
            memcpy(&bh, &yh, 4); memcpy(&bd, yd + i, 4);
            const bool nh = (bh & 0x7f800000u) == 0x7f800000u && (bh & 0x007fffffu), nd_ = (bd & 0x7f800000u) == 0x7f800000u && (bd & 0x007fffffu);
            if (nh || nd_) { nd += nh != nd_; continue; }
            nd += bh != bd;
        }
#if defined(__x86_64__) || defined(__i386__)
        _mm_setcsr(csr);
#endif
        CHECK(ran && nd == 0, "(z) the device's p_expf over %d values (its clamps and their neighbours, zeros, "
              "subnormals, infinities, NaN): %ld differ in their bits from pmath.h's (want 0)", NV, nd);
        gpu_ink2_free(h);
    }
    {
        static const struct { const char *what; int E, K, H, T; bool near; } rc[] = {
            { "the file's widths (128 experts, 8 used, H 2304), 1 position, exact ties", 128, 8, 2304, 1, false },
            { "the file's widths, 17 positions, a near tie (one ulp)", 128, 8, 2304, 17, true },
            { "the file's widths, 512 positions, exact ties", 128, 8, 2304, 512, false },
            { "6 experts, 2 used (the tiny file's 2 used; its 4 experts leave no room for the constructions), H 64, 17 "
              "positions", 6, 2, 64, 17, false },
            { "6 experts, 2 used, H 64, 1 position, a near tie", 6, 2, 64, 1, true },
            // D1b.1 (the second read's F1): the scores' grid is 1-D: 600000 positions are 75000 blocks of 8, past
            // grid.y's 65535 even with 8 positions a block (70000 positions, 8750 blocks, would not tell)
            { "6 experts, 2 used, H 16, 600000 positions (75000 blocks of 8: past grid.y's 65535)", 6, 2, 16, 600000, false },
        };
        for (size_t k = 0; k < sizeof rc / sizeof *rc; k++) {
            long nv = 0;
            const long d = route_identity(rc[k].E, rc[k].K, rc[k].H, rc[k].T, rc[k].near, &nv);
            CHECK(d == 0 && nv > 0, "(z) B1 the device router, %s: %ld of %ld scores, choices and weights differ from the "
                  "host's ink2_route_one, or a constructed tie went the other way (want 0) [scores %ld, choices %ld, "
                  "weights %ld, ties %ld; first at %ld: score %a / %a, chosen %d / %d, weight %a / %a (host / device)]",
                  rc[k].what, d, nv, g_robs.n[0], g_robs.n[1], g_robs.n[2], g_robs.n[3], g_robs.first, g_robs.sh,
                  g_robs.sd, (int)g_robs.ch, (int)g_robs.cd, g_robs.wh, g_robs.wd);
        }
        // D1b.1 (the second read's F3): the special values, where a parallel argmax would part from the serial scan
        static const struct { const char *what; int E, K, H, T, kind; } sc_[] = {
            { "a NaN score in the first slot (router row 0 NaN), the file's widths, 17 positions", 128, 8, 2304, 17, 1 },
            { "a NaN score in the first slot, 6 experts, 2 used", 6, 2, 64, 17, 1 },
            { "a NaN score later and a NaN bias, the file's widths, 17 positions", 128, 8, 2304, 17, 2 },
            { "a NaN score later and a NaN bias, 6 experts, 2 used", 6, 2, 64, 17, 2 },
            { "biases +inf and -inf, logits overflowing to +-inf, the file's widths, 17 positions", 128, 8, 2304, 17, 3 },
            { "biases +inf and -inf, logits overflowing, 6 experts, 2 used", 6, 2, 64, 17, 3 },
            { "ties at v = +0 and by a signed-zero bias, the file's widths, 17 positions", 128, 8, 2304, 17, 4 },
            { "ties at v = +0 and by a signed-zero bias, 6 experts, 2 used", 6, 2, 64, 17, 4 },
            { "K == E (8 of 8), H 64, 17 positions", 8, 8, 64, 17, 5 },
            { "K == E (128 of 128), the file's H, 3 positions", 128, 128, 2304, 3, 5 },
            { "a three-way tie for the K-th place in one lane and across lanes (K - 1, K, K - 1 + 32), the file's "
              "widths, 17 positions", 128, 8, 2304, 17, 6 },
            { "the same lane tie, 512 positions", 128, 8, 2304, 512, 6 },
        };
        for (size_t k = 0; k < sizeof sc_ / sizeof *sc_; k++) {
            long nv = 0;
            const long d = route_identity_k(sc_[k].E, sc_[k].K, sc_[k].H, sc_[k].T, false, sc_[k].kind, &nv);
            CHECK(d == 0 && nv > 0, "(z) B1 the device router, %s: %ld of %ld scores, choices and weights differ from the "
                  "host's (NaN = NaN), or the construction failed on the host (want 0) [scores %ld, choices %ld, weights "
                  "%ld, constructions %ld; first at %ld: score %a / %a, chosen %d / %d, weight %a / %a (host / device)]",
                  sc_[k].what, d, nv, g_robs.n[0], g_robs.n[1], g_robs.n[2], g_robs.n[3], g_robs.first, g_robs.sh,
                  g_robs.sd, (int)g_robs.ch, (int)g_robs.cd, g_robs.wh, g_robs.wd);
        }
    }
    // (z) B2: the model's rows with the device router equal the host router's (TEST-ONLY ..._NO_ROUTER=1) byte for byte
    // at -b 64 and -b 1 on each dense file. D1b.1's split: at -b 64 (T > 1) the device router runs, once an
    // application a forward, and the host router 0 times; at -b 1 (T == 1, a decode step's shape) the HOST router runs,
    // once an application a position, and the device router 0 times
    for (int f = 0; f < 3; f++) {
        static const char *kn[3] = { "F32", "BF16", "Q8_0" };
        model_t *gr = load(argv[1 + f], true, false, 64), *gr1 = load(argv[1 + f], true, false, 1);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_ROUTER", "1");
        model_t *gn = load(argv[1 + f], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_ROUTER", "");
        const void *zsp = NULL;
        if (gr) model_inkling2_parts(gr, &zsp, NULL);
        const int napp = zsp ? ((const ink2_shape *)zsp)->n_app : -1;
        const long hr0 = ink2_route_host_calls();
        float *a = gr ? rows(gr) : NULL;
        const long hr1 = ink2_route_host_calls();
        float *a1 = gr1 ? rows(gr1) : NULL;
        const long hr2 = ink2_route_host_calls();
        float *b = gn ? rows(gn) : NULL;
        const size_t n = gr ? (size_t)N_TOK * (size_t)gr->n_vocab : 0;
        const long c64 = model_inkling2_cuda_route_calls(gr), c1 = model_inkling2_cuda_route_calls(gr1);
        CHECK(a && a1 && b && !memcmp(a, b, sizeof(float) * n) && !memcmp(a1, b, sizeof(float) * n) && c64 == napp &&
              hr1 - hr0 == 0 && c1 == 0 && hr2 - hr1 == (long)napp * N_TOK,
              "(z) B2 %s: the rows equal the host router's byte for byte at -b 64 and -b 1; at -b 64 %ld device router "
              "calls (want %d) and the host router %ld times (want 0); at -b 1 %ld device router calls (want 0) and the "
              "host router %ld times (want %d)", kn[f], c64, napp, hr1 - hr0, c1, hr2 - hr1, napp * N_TOK);
        free(a); free(a1); free(b);
        drop(gr); drop(gr1); drop(gn);
    }
    // R4.48.9 D1d.2b (ah), the second read's F4 and F3: the device-pointer cores (the attention block, the expert
    // body, the message steps) called on PRIVATE device scratch (RUNNER_INKLING2_CUDA_TEST_DEVFORM=1: the host forms
    // upload there, not into the handle's buffers; a sliding layer's shared q / k / v input as one pointer): on each
    // dense tiny file at -b 64 (the device router) and -b 1 (one position: the host router's body), the rows
    // byte-identical to the normal path; the seam's calls on its handle equal the block / message-step calls the
    // instance ran and the body's are > 0; none on the normal handle. F3: at levels 2 / 3 / 4 one core at a time sees
    // a short buffer and must REFUSE it (the forward fails), never run or grow it.
    {
        static const int bs[2] = { 64, 1 };
        static const char *const fname[5] = { "", "F32", "BF16", "Q8_0", "F16" };
        for (int f = 1; f <= 4; f++)
            for (int k = 0; k < 2; k++) {
                model_t *g = load(argv[f], true, false, bs[k]);
                const void *hv = NULL;
                char pl = 0;
                size_t need = 0;
                long c0[3] = { -1, -1, -1 }, c1[3] = { -1, -1, -1 };
                float *a = g ? rows(g) : NULL;
                const int l0 = g && model_inkling2_cuda_placement(g, &pl, &need, &hv) ? gpu_ink2_test_devform((gpu_ink2 *)(uintptr_t)hv, c0) : -9;
                drop(g);
                plat_setenv("RUNNER_INKLING2_CUDA_TEST_DEVFORM", "1");
                model_t *d = load(argv[f], true, false, bs[k]);
                plat_setenv("RUNNER_INKLING2_CUDA_TEST_DEVFORM", "");
                float *b = d ? rows(d) : NULL;
                hv = NULL;
                const int l1 = d && model_inkling2_cuda_placement(d, &pl, &need, &hv) ? gpu_ink2_test_devform((gpu_ink2 *)(uintptr_t)hv, c1) : -9;
                const long nblk = d ? model_inkling2_cuda_block_calls(d) : -1, nmsg = d ? model_inkling2_cuda_msg_calls(d) : -1;
                const size_t n = d ? (size_t)N_TOK * (size_t)d->n_vocab : 0;
                const bool same = a && b && !memcmp(a, b, sizeof(float) * n);
                // (the F16 file runs 2a's path, no block calls (-1), and no device message steps: its counts 0; the
                // other three files run every kind)
                const bool every = f != 4;
                CHECK(same && l0 == 0 && c0[0] == 0 && c0[1] == 0 && c0[2] == 0 && l1 == 1 && c1[1] > 0 &&
                      c1[0] == (nblk > 0 ? nblk : 0) && c1[2] == (nmsg > 0 ? nmsg : 0) && (!every || (nblk > 0 && nmsg > 0)),
                      "(ah) the device-pointer cores on private scratch, %s at -b %d: rows byte-identical to the normal "
                      "path (%s); seam level %d / %d (want 0 / 1); its calls block %ld (of %ld), body %ld, message steps "
                      "%ld (of %ld); on the normal handle %ld / %ld / %ld (want 0)", fname[f], bs[k], same ? "yes" : "no",
                      l0, l1, c1[0], nblk, c1[1], c1[2], nmsg, c0[0], c0[1], c0[2]);
                free(a); free(b);
                drop(d);
            }
        static const char *const lvn[3] = { "the block's aq", "the expert body's exe", "the message steps' mgh" };
        for (int lv = 2; lv <= 4; lv++) {
            char v[2] = { (char)('0' + lv), 0 };
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_DEVFORM", v);
            model_t *d = load(argv[2], true, false, 64);
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_DEVFORM", "");
            float *r = d ? rows(d) : NULL;
            CHECK(d && !r, "(ah) F3: with %s short for its core (level %d), the forward is refused (%s)", lvn[lv - 2], lv,
                  !d ? "the load FAILED" : r ? "it RAN" : "refused");
            free(r);
            drop(d);
        }
    }
    // R4.48.9 D1d.3b (aj): the RESIDENT decode step (on by default) against the same
    // file's instance without it, over one sequence that crosses between the two forms: a prefill of 10, 8 decode
    // steps, a chunk of 4 (the post convolutions' histories down to the host first, up after), a step (they went up
    // right), then a step the step must decline (the selection observer is on: the device's histories down first). Every
    // row returned byte-identical to the control's; the step counter 9; the handle's host transfers 3 a step (the
    // embedding row and the position up, the logits row down) and 0 inside the applications' loop. The F16 file runs
    // 2a's path (no block calls): its instance has no step (counter -1), its rows unchanged.
    {
        static const struct { int p0, n; bool obs; } seq[] = {
            { 0, 10, false }, { 10, 1, false }, { 11, 1, false }, { 12, 1, false }, { 13, 1, false }, { 14, 1, false },
            { 15, 1, false }, { 16, 1, false }, { 17, 1, false }, { 18, 4, false }, { 22, 1, false }, { 23, 1, true } };
        const int ns = (int)(sizeof seq / sizeof *seq);
        static const char *const fname[5] = { "", "F32", "BF16", "Q8_0", "F16" };
        for (int f = 1; f <= 4; f++) {
            model_t *c0 = load(argv[f], true, false, 64);   // the control: no resident step (the file's setting)
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "");
            model_t *rs = load(argv[f], true, false, 64);   // the step on, the default
            plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "1");
            const size_t V = c0 ? (size_t)c0->n_vocab : 0;
            long nd = 0, bad = 0;
            int first = -1;
            for (int k = 0; c0 && rs && k < ns; k++) {
                if (seq[k].obs) ink2_set_sel_observer(obs_fn, NULL);
                const float *a = model_inkling2_forward(c0, PROMPT + seq[k].p0, seq[k].n, seq[k].p0, true, false);
                float *ac = a ? malloc(sizeof(float) * V) : NULL;
                if (ac) memcpy(ac, a, sizeof(float) * V);
                const float *b = model_inkling2_forward(rs, PROMPT + seq[k].p0, seq[k].n, seq[k].p0, true, false);
                if (seq[k].obs) ink2_set_sel_observer(NULL, NULL);
                if (!ac || !b) bad++;
                else if (memcmp(ac, b, sizeof(float) * V)) { nd++; if (first < 0) first = k; }
                free(ac);
            }
            const long steps = rs ? model_inkling2_cuda_step_calls(rs) : -2;
            const void *hv = NULL;
            char pl = 0;
            size_t need = 0;
            long st[3] = { -1, -1, -1 };
            if (rs && model_inkling2_cuda_placement(rs, &pl, &need, &hv)) gpu_ink2_test_step_stats((gpu_ink2 *)(uintptr_t)hv, st);
            const bool has = f != 4;
            CHECK(c0 && rs && bad == 0 && nd == 0 && (has ? steps == 9 && st[0] == 9 && st[1] == 27 && st[2] == 0 : steps == -1),
                  "(aj) the resident decode step, %s: %d forwards (a prefill, 8 steps, a chunk, a step, a declined step) "
                  "byte-identical to the control's (%ld differ, the first at forward %d; %ld failed); steps %ld (want %s); "
                  "host transfers %ld (want 27: 3 a step), inside the applications' loop %ld (want 0)", fname[f], ns, nd,
                  first, bad, steps, has ? "9" : "-1: no step", st[1], st[2]);
            // and a sequence of single steps FROM POSITION 0 (the step begins the sequence itself: the case -b 1 takes)
            long nd0 = 0, bad0 = 0;
            int first0 = -1;
            for (int p = 0; c0 && rs && p < N_TOK; p++) {
                const float *a = model_inkling2_forward(c0, PROMPT + p, 1, p, true, false);
                float *ac = a ? malloc(sizeof(float) * V) : NULL;
                if (ac) memcpy(ac, a, sizeof(float) * V);
                const float *b = model_inkling2_forward(rs, PROMPT + p, 1, p, true, false);
                if (!ac || !b) bad0++;
                else if (memcmp(ac, b, sizeof(float) * V)) { nd0++; if (first0 < 0) first0 = p; }
                free(ac);
            }
            CHECK(c0 && rs && nd0 == 0 && bad0 == 0, "(aj) the resident decode step, %s, single steps from position 0: %ld of "
                  "%d positions differ from the control's (the first at position %d; %ld failed; want 0)", fname[f], nd0,
                  N_TOK, first0, bad0);
            drop(c0); drop(rs);
        }
    }
    // R4.48.9 D1d.3b (ak), the second read's F1: the step holds the shared handle's lock for the WHOLE step, so two
    // instances of one file (one device copy, its scratch shared) stepping at once from two threads each get their own
    // rows: at -b 1 every position is a resident step; a control instance of the same copy runs alone first; then two
    // instances run 4 sequences each concurrently. Every one of their 8 row sets byte-identical to the control's, the
    // three on one handle, and each concurrent instance's step counter 4 x 24.
    {
        model_t *nr = load(argv[2], true, false, 1);   // -b 1 WITHOUT the step (the file's setting): the bytes to match
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "");
        model_t *c = load(argv[2], true, false, 1), *a = load(argv[2], true, false, 1), *b = load(argv[2], true, false, 1);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "1");
        float *ref = c ? rows(c) : NULL, *nref = nr ? rows(nr) : NULL;
        CHECK(ref && nref && !memcmp(ref, nref, sizeof(float) * (size_t)N_TOK * (size_t)c->n_vocab) &&
              model_inkling2_cuda_step_calls(c) == N_TOK && model_inkling2_cuda_step_calls(nr) == -1,
              "(ak) at -b 1 the resident step's rows (every position a step from 0) equal the non-resident route's byte for "
              "byte (%s); steps %ld / %ld (want %d / -1)", ref && nref && !memcmp(ref, nref, sizeof(float) * (size_t)N_TOK *
              (size_t)c->n_vocab) ? "yes" : "NO", c ? model_inkling2_cuda_step_calls(c) : -2,
              nr ? model_inkling2_cuda_step_calls(nr) : -2, N_TOK);
        free(nref); drop(nr);
        par_job ja = { a, { 0 } }, jb = { b, { 0 } };
        pthread_t ta, tb;
        const bool st = ref && a && b && !pthread_create(&ta, NULL, par_run, &ja) && !pthread_create(&tb, NULL, par_run, &jb);
        if (st) { pthread_join(ta, NULL); pthread_join(tb, NULL); }
        const size_t n = c ? (size_t)N_TOK * (size_t)c->n_vocab : 0;
        int same = 0;
        for (int k = 0; st && k < 4; k++) {
            same += ja.out[k] && !memcmp(ja.out[k], ref, sizeof(float) * n);
            same += jb.out[k] && !memcmp(jb.out[k], ref, sizeof(float) * n);
        }
        const void *hc = NULL, *ha = NULL, *hb = NULL;
        char pl = 0;
        size_t need = 0;
        if (c) model_inkling2_cuda_placement(c, &pl, &need, &hc);
        if (a) model_inkling2_cuda_placement(a, &pl, &need, &ha);
        if (b) model_inkling2_cuda_placement(b, &pl, &need, &hb);
        const long sa = a ? model_inkling2_cuda_step_calls(a) : -2, sb = b ? model_inkling2_cuda_step_calls(b) : -2;
        CHECK(st && same == 8 && hc && hc == ha && ha == hb && sa == 4 * N_TOK && sb == 4 * N_TOK,
              "(ak) two instances stepping at once on one device copy (BF16, -b 1): %d of 8 row sets byte-identical to "
              "the control's (want 8); one handle: %s; steps %ld / %ld (want %d each)", same,
              hc && hc == ha && ha == hb ? "yes" : "NO", sa, sb, 4 * N_TOK);
        for (int k = 0; k < 4; k++) { free(ja.out[k]); free(jb.out[k]); }
        free(ref);
        drop(a); drop(b); drop(c);
    }
    // R4.48.9 D1d.3b (ak2), the stage-3 read's F3: one slot's PREFILL (its non-resident chunks, then the post
    // convolutions' host-to-device sync) interleaved with another slot's resident STEPS on ONE device copy (both at
    // -b 64: a server's slots share their -b). Slot A runs the prompt as one chunk (rows); slot B runs it one position a
    // call (every position a step); each alone first, then both at once from two threads, 4 sequences each. Every row
    // set byte-identical to its slot's alone run; one handle; B's steps 5 x 24, A's 0.
    {
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "");
        model_t *a = load(argv[2], true, false, 64), *b = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "1");
        float *ra = a ? rows(a) : NULL, *rb = b ? rows_steps(b) : NULL;
        par_job ja = { a, { 0 } }, jb = { b, { 0 } };
        pthread_t ta, tb;
        const bool st = ra && rb && !pthread_create(&ta, NULL, par_run, &ja) && !pthread_create(&tb, NULL, par_run_steps, &jb);
        if (st) { pthread_join(ta, NULL); pthread_join(tb, NULL); }
        const size_t n = a ? (size_t)N_TOK * (size_t)a->n_vocab : 0;
        int sa = 0, sb = 0;
        for (int k = 0; st && k < 4; k++) {
            sa += ja.out[k] && !memcmp(ja.out[k], ra, sizeof(float) * n);
            sb += jb.out[k] && !memcmp(jb.out[k], rb, sizeof(float) * n);
        }
        const void *ha = NULL, *hb = NULL;
        char pl = 0;
        size_t need = 0;
        if (a) model_inkling2_cuda_placement(a, &pl, &need, &ha);
        if (b) model_inkling2_cuda_placement(b, &pl, &need, &hb);
        const long na = a ? model_inkling2_cuda_step_calls(a) : -2, nb = b ? model_inkling2_cuda_step_calls(b) : -2;
        CHECK(st && sa == 4 && sb == 4 && ha && ha == hb && na == 0 && nb == 5 * N_TOK,
              "(ak2) one slot's prefill (chunks + the post-convolution sync) beside another's resident steps on one copy "
              "(BF16, -b 64): A %d / 4, B %d / 4 row sets byte-identical to each alone (want 4 / 4); one handle: %s; steps "
              "A %ld (want 0), B %ld (want %d)", sa, sb, ha && ha == hb ? "yes" : "NO", na, nb, 5 * N_TOK);
        for (int k = 0; k < 4; k++) { free(ja.out[k]); free(jb.out[k]); }
        free(ra); free(rb);
        drop(a); drop(b);
    }
    // R4.48.9 D1d.3b (al), the stage-3 read's F2: the resident step's FAILURE path with the step on (every other op's
    // injection runs step-off, and at T == 1 the step replaces those ops): RUNNER_INKLING2_CUDA_TEST_STEP_FAIL=2 fails
    // the third step attempt. Single steps from 0: positions 0 and 1 give rows, position 2's forward is refused, and the
    // instance refuses every forward after (its device failed: never a CPU row in its place)
    {
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "");
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_STEP_FAIL", "2");
        model_t *g = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_STEP_FAIL", "");
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "1");
        const bool on = g && model_inkling2_cuda_step_calls(g) >= 0;
        const float *s0 = on ? model_inkling2_forward(g, PROMPT, 1, 0, true, false) : NULL;
        const float *s1 = s0 ? model_inkling2_forward(g, PROMPT + 1, 1, 1, true, false) : NULL;
        const float *s2 = s1 ? model_inkling2_forward(g, PROMPT + 2, 1, 2, true, false) : NULL;
        const float *s3 = s1 ? model_inkling2_forward(g, PROMPT, 4, 0, true, false) : NULL;   // a chunk: refused too
        CHECK(on && s0 && s1 && !s2 && !s3 && model_inkling2_cuda_step_calls(g) == 2,
              "(al) a failed resident step (step on, the third attempt injected): positions 0 / 1 %s / %s, position 2 %s, "
              "the next forward (a chunk from 0) %s; steps %ld (want 2)", s0 ? "rows" : "REFUSED", s1 ? "rows" : "REFUSED",
              s2 ? "GAVE ROWS" : "refused", s3 ? "GAVE ROWS" : "refused", g ? model_inkling2_cuda_step_calls(g) : -2);
        drop(g);
    }
    // R4.48.9 D1d.1 (af): the decode step's remaining ops on the device, each against the host function the forward
    // runs (inkling2.c's own, through its TEST-ONLY wrappers) on the same inputs, BIT FOR BIT (NaN by its bits), the
    // host with the CPU's FTZ / DAZ off (the device never flushes; D1a's tanh check): the RMS norm rows (weighted and
    // not; a zero row, a row of huge values, subnormals), the depth attention (the block reads' 1 and 3 queries and
    // the message input's 4, 1 to 26 states, a zero state), the short convolution with its history and store (a
    // prefill without history, chunks continuing a sequence, decode steps around the ring), the residual add and the
    // head's softcap (huge, small, zero, signed).
    {
        uint32_t seed = 4211u;
#define RNDF() (seed = seed * 1664525u + 1013904223u, (float)((int)(seed >> 9) % 2001 - 1000) / 1000.0f)
        gpu_ink2 *h = gpu_ink2_bind();
#if defined(__x86_64__) || defined(__i386__)
        const unsigned csr = _mm_getcsr();
#define FTZ_OFF() _mm_setcsr(csr & ~0x8040u)
#define FTZ_BACK() _mm_setcsr(csr)
#else
#define FTZ_OFF() ((void)0)
#define FTZ_BACK() ((void)0)
#endif
        // the norm rows (R4.48.9.3: one block a row, the row staged in tiles of INK2_NORM_TILE = 1024 by a block of
        // 256 threads: the lengths around one block's width and around the tile, beside the real width 2304)
        static const int nh[] = { 64, 255, 256, 257, 1023, 1024, 1025, 2304 };
        for (size_t hk = 0; hk < sizeof nh / sizeof *nh; hk++) {
            const int H = nh[hk], R = 6;
            float *x = malloc(sizeof(float) * (size_t)R * H), *wt = malloc(sizeof(float) * (size_t)H);
            float *yd = malloc(sizeof(float) * (size_t)R * H), *yh = malloc(sizeof(float) * (size_t)R * H);
            for (int i = 0; i < R * H; i++) x[i] = RNDF();
            for (int i = 0; i < H; i++) { x[i] = 0.0f; x[H + i] = 3.0e18f * RNDF(); x[2 * H + i] = 1e-39f * RNDF(); wt[i] = RNDF(); }
            for (int wk = 0; wk < 2; wk++) {
                const float *wp = wk ? wt : NULL;
                const float eps = wk ? 1e-5f : 1e-6f;
                const bool ran = h && gpu_ink2_test_rms_rows(h, x, wp, yd, R, H, eps);
                FTZ_OFF();
                for (int r = 0; r < R; r++) ink2_test_rms_norm(x + (size_t)r * H, H, eps, wp, yh + (size_t)r * H);
                FTZ_BACK();
                long nd = 0;
                for (int i = 0; ran && i < R * H; i++) nd += !same_f(yd[i], yh[i]);
                CHECK(ran && nd == 0, "(af) the RMS norm rows (%s, H %d, 6 rows: zero, huge, subnormal, random): %ld of %d "
                      "differ from the host's (want 0)", wk ? "weighted" : "unweighted", H, nd, R * H);
            }
            free(x); free(wt); free(yd); free(yh);
        }
        // the depth attention
        // (D1d.2b, the second read's F6: lay 1 the [T][nq][H] output layout beside [nq][T][H]; qs 60 spreads a query's
        // scores past 104, so some weights underflow to subnormals and to 0, each checked by the scores' spread)
        static const struct { int nq, n_kv, H, T, lay; float qs; } dc[] = {
            { 1, 1, 2304, 1, 0, 0.1f }, { 3, 7, 2304, 1, 0, 0.1f }, { 4, 26, 2304, 3, 0, 0.1f }, { 1, 13, 64, 5, 0, 0.1f },
            { 4, 2, 64, 1, 0, 0.1f }, { 3, 7, 2304, 2, 1, 0.1f }, { 4, 26, 2304, 3, 0, 60.0f }, { 1, 13, 64, 5, 1, 60.0f } };
        for (size_t k = 0; k < sizeof dc / sizeof *dc; k++) {
            const int nq = dc[k].nq, nkv = dc[k].n_kv, H = dc[k].H, T = dc[k].T, lay = dc[k].lay;
            float *q = malloc(sizeof(float) * (size_t)nq * H), *st = malloc(sizeof(float) * (size_t)T * nkv * H);
            float *od = malloc(sizeof(float) * (size_t)nq * T * H), *oh = malloc(sizeof(float) * (size_t)nq * T * H);
            for (int i = 0; i < nq * H; i++) q[i] = dc[k].qs * RNDF();
            for (int i = 0; i < T * nkv * H; i++) st[i] = RNDF();
            if (nkv > 1) memset(st, 0, sizeof(float) * (size_t)H);   // a zero state (its norm 0 / sqrt(eps))
            const float scale = ink2_inv_sqrt(H);
            // the widest spread of one query's scores at one position (float64, the states normed as the op does)
            double spread = 0;
            for (int t = 0; t < T; t++)
                for (int i = 0; i < nq; i++) {
                    double lo = 1e300, hi = -1e300;
                    for (int j = 0; j < nkv; j++) {
                        const float *sj = st + ((size_t)t * nkv + j) * H;
                        double ss = 0, d = 0;
                        for (int u = 0; u < H; u++) { ss += (double)sj[u] * sj[u]; d += (double)q[(size_t)i * H + u] * sj[u]; }
                        const double s_ = scale * d / sqrt(ss / H + 1e-6);
                        if (s_ < lo) lo = s_;
                        if (s_ > hi) hi = s_;
                    }
                    if (hi - lo > spread) spread = hi - lo;
                }
            const int ois = lay ? H : T * H, ots = lay ? nq * H : H;
            const bool ran = h && gpu_ink2_test_depth_attn(h, q, nq, st, nkv, H, T, scale, od, ois, ots);
            int hb = 0;
            FTZ_OFF();
            for (int t = 0; t < T; t++) {
                float *tmp = malloc(sizeof(float) * (size_t)nq * H);
                hb |= !tmp || ink2_test_depth_attn(q, nq, st + (size_t)t * nkv * H, nkv, H, tmp);
                for (int i = 0; tmp && i < nq; i++)
                    memcpy(oh + (size_t)i * ois + (size_t)t * ots, tmp + (size_t)i * H, sizeof(float) * H);
                free(tmp);
            }
            FTZ_BACK();
            long nd = 0;
            for (int i = 0; ran && !hb && i < nq * T * H; i++) nd += !same_f(od[i], oh[i]);
            const bool spread_ok = dc[k].qs < 1.0f || spread > 104.0;
            CHECK(ran && !hb && nd == 0 && spread_ok, "(af) the depth attention (%d queries, %d states, H %d, %d positions, "
                  "layout %s, scores spread %.1f%s): %ld of %d differ from the host's (want 0)", nq, nkv, H, T,
                  lay ? "[T][nq][H]" : "[nq][T][H]", spread, dc[k].qs < 1.0f ? "" : " (want > 104: weights underflow)", nd,
                  nq * T * H);
            free(q); free(st); free(od); free(oh);
        }
        // (am) R4.48.9.3: the resident step's INDEXED depth read (k_ink2_depth_attn_ix, its scores through shared memory
        // in tiles sized by the state count) against the host's depth attention over the same states in the list's
        // order, bit for bit: one tile, a tile and a one-element tail, several tiles with a tail, the floor tile of 32
        // (257 states), a list that skips and repeats pool rows, a zero state, and the spread whose weights underflow.
        // A lone state's weight is 1 and 0.1-scaled queries leave too little spread for a score's last bits to reach the
        // output, so the one-element tail and the single tile also run with several states at qs 0.9 (R4.48.9.3 F1)
        static const struct { int nq, n_kv, n_pool, H; float qs; } ic[] = {
            { 1, 1, 3, 1025, 0.1f }, { 1, 5, 5, 1024, 0.1f }, { 3, 12, 12, 1023, 0.1f }, { 3, 7, 9, 2304, 0.1f },
            { 4, 26, 30, 2304, 0.1f }, { 4, 40, 41, 4096, 0.1f }, { 1, 257, 260, 64, 0.1f }, { 4, 26, 30, 2304, 60.0f },
            { 1, 3, 4, 1025, 0.9f }, { 1, 5, 5, 1024, 0.9f } };
        for (size_t k = 0; k < sizeof ic / sizeof *ic; k++) {
            const int nq = ic[k].nq, nkv = ic[k].n_kv, np = ic[k].n_pool, H = ic[k].H;
            float *q = malloc(sizeof(float) * (size_t)nq * H), *pool = malloc(sizeof(float) * (size_t)np * H);
            float *sg = malloc(sizeof(float) * (size_t)nkv * H), *od = malloc(sizeof(float) * (size_t)nq * H);
            float *oh = malloc(sizeof(float) * (size_t)nq * H);
            int *ix = malloc(sizeof(int) * (size_t)nkv);
            for (int i = 0; i < nq * H; i++) q[i] = ic[k].qs * RNDF();
            for (int i = 0; i < np * H; i++) pool[i] = RNDF();
            memset(pool, 0, sizeof(float) * (size_t)H);   // pool row 0: a zero state
            for (int j = 0; j < nkv; j++) ix[j] = (int)(((size_t)j * 7 + (size_t)(np - 1)) % (size_t)np);   // skips, wraps
            if (nkv > 2) ix[nkv - 1] = ix[0];   // a repeated row
            for (int j = 0; j < nkv; j++) memcpy(sg + (size_t)j * H, pool + (size_t)ix[j] * H, sizeof(float) * H);
            bool zero = false;
            for (int j = 0; j < nkv; j++) zero |= ix[j] == 0;
            const float scale = ink2_inv_sqrt(H);
            double spread = 0;
            for (int i = 0; i < nq; i++) {
                double lo = 1e300, hi = -1e300;
                for (int j = 0; j < nkv; j++) {
                    const float *sj = sg + (size_t)j * H;
                    double ss = 0, d = 0;
                    for (int u = 0; u < H; u++) { ss += (double)sj[u] * sj[u]; d += (double)q[(size_t)i * H + u] * sj[u]; }
                    const double s_ = scale * d / sqrt(ss / H + 1e-6);
                    if (s_ < lo) lo = s_;
                    if (s_ > hi) hi = s_;
                }
                if (hi - lo > spread) spread = hi - lo;
            }
            const bool ran = h && gpu_ink2_test_depth_attn_ix(h, q, nq, pool, np, ix, nkv, H, scale, od);
            FTZ_OFF();
            const int hb = ink2_test_depth_attn(q, nq, sg, nkv, H, oh);
            FTZ_BACK();
            long nd = 0;
            for (int i = 0; ran && !hb && i < nq * H; i++) nd += !same_f(od[i], oh[i]);
            const bool spread_ok = ic[k].qs < 1.0f || spread > 104.0;
            const bool zero_ok = nkv < 3 || zero || ic[k].qs > 0.5f;   // the qs 0.9 cases are the score path's, not the zero state's
            CHECK(ran && !hb && nd == 0 && spread_ok && zero_ok, "(am) the step's indexed depth read (%d queries, "
                  "%d of %d pool states%s, H %d, scores spread %.1f%s): %ld of %d differ from the host's (want 0)", nq, nkv,
                  np, zero ? " with the zero state" : "", H, spread, ic[k].qs < 1.0f ? "" : " (want > 104: weights underflow)",
                  nd, nq * H);
            free(q); free(pool); free(sg); free(od); free(oh); free(ix);
        }
        // the short convolution with its history: one sequence through chunks (a prefill, a chunk, decode steps
        // around the ring), the device's history carried between calls as the host's is
        {
            const int C = 2304, K = 4;
            float *cw = malloc(sizeof(float) * (size_t)C * K), *hd = calloc((size_t)(K - 1) * C, sizeof(float));
            float *hh = calloc((size_t)(K - 1) * C, sizeof(float));
            for (int i = 0; i < C * K; i++) cw[i] = RNDF();
            static const int chunks[] = { 5, 2, 1, 1, 1, 1, 3 };
            int p0 = 0;
            long nd = 0, nv = 0;
            bool ran = h != NULL;
            for (size_t c = 0; ran && c < sizeof chunks / sizeof *chunks; c++) {
                const int T = chunks[c];
                float *lin = malloc(sizeof(float) * (size_t)T * C), *od = malloc(sizeof(float) * (size_t)T * C);
                float *oh = malloc(sizeof(float) * (size_t)T * C);
                for (int i = 0; i < T * C; i++) lin[i] = RNDF();
                const float div = (float)sqrt((double)(K + 1));
                ran = gpu_ink2_test_short_conv(h, cw, C, K, T, lin, od, p0, hd, div);
                FTZ_OFF();
                ink2_test_short_conv(cw, C, K, T, lin, oh, p0, hh);
                FTZ_BACK();
                for (int i = 0; ran && i < T * C; i++) nd += !same_f(od[i], oh[i]);
                for (int i = 0; ran && i < (K - 1) * C; i++) nd += !same_f(hd[i], hh[i]);
                nv += (long)T * C + (long)(K - 1) * C;
                p0 += T;
                free(lin); free(od); free(oh);
            }
            CHECK(ran && nd == 0, "(af) the short convolution with its history (C %d, K %d; chunks 5, 2, then decode steps "
                  "around the ring, then 3): %ld of %ld outputs and history values differ from the host's (want 0)", C, K, nd, nv);
            // F6: a prefill without history (hist NULL: p0 taken as 0, no store), at a nonzero p0 and at K 4 and 1
            for (int kk = 0; kk < 2; kk++) {
                const int K1 = kk ? 1 : K, T = 6;
                float *lin = malloc(sizeof(float) * (size_t)T * C), *od = malloc(sizeof(float) * (size_t)T * C);
                float *oh = malloc(sizeof(float) * (size_t)T * C);
                for (int i = 0; i < T * C; i++) lin[i] = RNDF();
                const float div = (float)sqrt((double)(K1 + 1));
                const bool r1 = h && gpu_ink2_test_short_conv(h, cw, C, K1, T, lin, od, 9, NULL, div);
                FTZ_OFF();
                ink2_test_short_conv(cw, C, K1, T, lin, oh, 9, NULL);
                FTZ_BACK();
                long n1 = 0;
                for (int i = 0; r1 && i < T * C; i++) n1 += !same_f(od[i], oh[i]);
                CHECK(r1 && n1 == 0, "(af) the short convolution without history (a prefill; C %d, K %d, T %d, p0 9 taken "
                      "as 0): %ld of %d differ from the host's (want 0)", C, K1, T, n1, T * C);
                free(lin); free(od); free(oh);
            }
            free(cw); free(hd); free(hh);
        }
        // the residual add and the softcap
        {
            enum { N = 4096 };
            static float a[N], b[N], sd[N], vd[N], vh[N];
            for (int i = 0; i < N; i++) { a[i] = RNDF(); b[i] = 1e-3f * RNDF(); vd[i] = vh[i] = 60.0f * RNDF(); }
            vd[0] = vh[0] = 0.0f; vd[1] = vh[1] = -0.0f; vd[2] = vh[2] = 1e30f; vd[3] = vh[3] = -1e30f; vd[4] = vh[4] = 1e-30f;
            // F6: subnormal inputs (the device never flushes; the host checked with FTZ / DAZ off)
            vd[5] = vh[5] = 1e-40f; vd[6] = vh[6] = -3e-42f; vd[7] = vh[7] = 1.4e-45f; a[8] = 2e-39f; b[8] = -1e-39f;
            const float cap = 30.0f;
            const bool ran = h && gpu_ink2_test_add_softcap(h, a, b, sd, vd, N, cap);
            long nd = 0;
            FTZ_OFF();
            for (int i = 0; ran && i < N; i++) {
                const float s_ = a[i] + b[i], c_ = ink2_test_softcap(vh[i], cap);
                nd += !same_f(sd[i], s_) + !same_f(vd[i], c_);
            }
            FTZ_BACK();
            CHECK(ran && nd == 0, "(af) the residual add and the softcap (cap 30, %d values; 0, -0, +-1e30, 1e-30, subnormals): %ld "
                  "differ from the host's (want 0)", N, nd);
        }
#undef FTZ_OFF
#undef FTZ_BACK
#undef RNDF
        gpu_ink2_free(h);
    }
    // D2 (k): the fused expert body equals the former per-expert launches bit for bit
    {
        static const struct { const char *what; int type, H, F, E, T, used, all, none; } xc[] = {
            { "NVFP4 at the real widths, one position, 8 of 16 experts (a decode step)", T_NVFP4, 2304, 1152, 16, 1, 8, -1, -1 },
            { "NVFP4, 37 positions, 4 experts each plus expert 1 at every position, expert 2 never", T_NVFP4, 256, 128, 16, 37, 4, 1, 2 },
            { "NVFP4, 300 positions, 8 of 16 each (2400 columns)", T_NVFP4, 256, 128, 16, 300, 8, -1, -1 },
            { "BF16 (no factors), 6 positions, 2 of 8 each", T_BF16, 256, 128, 8, 6, 2, -1, -1 },
            { "BF16 at the real widths, one position, 8 of 16 experts", T_BF16, 2304, 1152, 16, 1, 8, -1, -1 },
            // BF16 experts run per column whatever their column count (P1a's BF16 tile measured slower and went): these
            // hold that path to the reference at the counts where an NVFP4 expert tiles
            { "BF16, 300 positions, 8 of 16 each (~150 columns an expert)", T_BF16, 256, 128, 16, 300, 8, -1, -1 },
            { "BF16, 20 positions, 8 of 16 each plus expert 1 at every position", T_BF16, 256, 128, 16, 20, 8, 1, -1 },
            { "BF16, 16 positions, expert 3 at every position (16 columns)", T_BF16, 256, 128, 16, 16, 2, 3, -1 },
            { "BF16, 17 positions, expert 3 at every position (17 columns)", T_BF16, 256, 128, 16, 17, 2, 3, -1 },
            { "BF16 at the real widths, 64 positions, 8 of 16 each", T_BF16, 2304, 1152, 16, 64, 8, -1, -1 },
            // P1b: NVFP4 experts with at least 16 columns run as batched tiles too
            { "NVFP4, 16 positions, expert 3 at every position (exactly one full tile)", T_NVFP4, 256, 128, 16, 16, 2, 3, -1 },
            { "NVFP4, 17 positions, expert 3 at every position (a tile of 16 and one of 1)", T_NVFP4, 256, 128, 16, 17, 2, 3, -1 },
            { "NVFP4 at the real widths, 64 positions, 8 of 16 each", T_NVFP4, 2304, 1152, 16, 64, 8, -1, -1 },
        };
        for (size_t k = 0; k < sizeof xc / sizeof *xc; k++) {
            long nv = 0;
            const long d = exp_identity(xc[k].type, xc[k].H, xc[k].F, xc[k].E, xc[k].T, xc[k].used, xc[k].all, xc[k].none, &nv);
            CHECK(d == 0 && nv > 0, "(k) the fused expert body, %s: %ld of %ld values differ in their bits from the former "
                  "per-expert launches' (want 0)", xc[k].what, d, nv);
        }
    }
    // D1d.3a (ai), ruling (i)'s red-first bar: the device-routed body's BOUNDED launches byte-identical to its SIZED
    // ones at T = 1, 17 and 64 (rows, choices, margins), NVFP4 and BF16, at the real widths and at Inkling's 128
    // experts; the bounded call reads no count back to the host (0 readbacks), the sized one reads one; at T 64 the
    // NVFP4 case's tiled launches are live (experts with 16+ columns)
    {
        static const struct { int type, H, F, E, K, T; } rc[] = {
            { T_NVFP4, 2304, 1152, 16, 8, 1 }, { T_NVFP4, 2304, 1152, 16, 8, 17 }, { T_NVFP4, 2304, 1152, 16, 8, 64 },
            { T_BF16, 2304, 1152, 16, 8, 1 }, { T_BF16, 2304, 1152, 16, 8, 17 }, { T_BF16, 2304, 1152, 16, 8, 64 },
            { T_NVFP4, 256, 128, 128, 8, 1 }, { T_NVFP4, 256, 128, 128, 8, 17 }, { T_NVFP4, 256, 128, 128, 8, 64 },
        };
        for (size_t k = 0; k < sizeof rc / sizeof *rc; k++) {
            long nv = 0, rs = -1, rbd = -1;
            int nt = 0;
            const long d = body_bounded_identity(rc[k].type, rc[k].H, rc[k].F, rc[k].E, rc[k].K, rc[k].T, &nv, &rs, &rbd, &nt);
            const bool tiles = rc[k].type == T_NVFP4 && rc[k].T == 64 && rc[k].E == 16;
            CHECK(d == 0 && nv > 0 && rs == 1 && rbd == 0 && (!tiles || nt > 0),
                  "(ai) the bounded expert body, %s H %d E %d K %d at T %d: %ld of %ld values differ from the sized "
                  "launches' (want 0); count readbacks sized %ld / bounded %ld (want 1 / 0); experts tiled %d%s",
                  rc[k].type == T_NVFP4 ? "NVFP4" : "BF16", rc[k].H, rc[k].E, rc[k].K, rc[k].T, d, nv, rs, rbd, nt,
                  tiles ? " (want > 0)" : "");
        }
    }
    // G2b 2a (n): gpu_ink2_attn against float64 on the host, by core_case below. Bar fixed before the run: 1e-4 of the
    // largest value, finite. A full layer over 1040 positions in one call (every key count 1..1040: around each
    // 32-key chunk); a sliding layer (window 40, ring 103) through chunks of 64, 8 and 64 that straddle the ring's
    // wraps at 103 and 206, then single steps; head_dim 32, 64 and 256 beside 128 (the range the kernel takes); GQA
    // (4 heads on 2 K/V heads), the bias within distance 16, sinks from -20 to +5
    {
        static const int c_full[] = { 1040 }, c_slide[] = { 64, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 64, 1, 1, 1, 1, 1, 1, 1, 1 };
        static const int c_small[] = { 70, 8, 8 }, c_mid[] = { 32, 8, 8, 8, 8, 32 }, c_wide[] = { 64, 1, 1 };
        static const int c_steps[40] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        static const int c_long[41] = { 1000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        static const struct { const char *what; int hd, global, window, R; const int *ch; int nch; } cs[] = {
            { "a full layer, 1040 queries in one call (head_dim 128)", 128, 1, 40, 1100, c_full, 1 },
            { "a full layer, 40 single steps from position 0 (1..40 keys, head_dim 128)", 128, 1, 40, 64, c_steps, 40 },
            { "a full layer, 1000 positions then 40 single steps (1001..1040 keys, head_dim 128)", 128, 1, 40, 1100,
              c_long, 41 },
            { "a sliding layer (window 40, ring 103), chunks of 64, 8 and 64 across both of the ring's wraps, then single "
              "steps (head_dim 128)", 128, 0, 40, 103, c_slide, (int)(sizeof c_slide / sizeof *c_slide) },
            { "a full layer, head_dim 32", 32, 1, 40, 200, c_small, 3 },
            { "a sliding layer (window 16, ring 47), head_dim 64", 64, 0, 16, 47, c_mid, 6 },
            { "a full layer, head_dim 256", 256, 1, 40, 100, c_wide, 3 },
        };
        for (size_t k = 0; k < sizeof cs / sizeof *cs; k++) {
            size_t nf = 0;
            long cmp = 0, nd = 0;
            const double w = core_case(cs[k].hd, cs[k].global, cs[k].window, cs[k].R, cs[k].ch, cs[k].nch, &nf, &cmp, &nd);
            CHECK(w <= 1e-4 && nf == 0 && cmp > 0, "(n) the device core, %s: within %.2e of the float64 sums, want <= 1e-4; "
                  "%zu not finite (%ld values)", cs[k].what, w, nf, cmp);
            CHECK(cmp > 0 && nd == 0, "(n) the device core, %s: %ld of %ld values differ in their bits from the reference "
                  "kernel's (the core before the decode fix; want 0)", cs[k].what, nd, cmp);
        }
    }
    // (ae) the second read's P2 3 (and 2): the 16-column threshold observed, not only its bits: a dense product runs
    // batched from 16 columns and per column below (one batched dispatch a product), an NVFP4 expert body tiles only
    // an expert of 16+ columns (one tiled dispatch each for gate_up and down); the padded buffers hold INK2_XPAD
    // columns past their last (x of a product, the gathered x and the activations, the attention output); the wide
    // products at partial tiles (17, 33 columns) bit for bit; and P2 2: an attention call without a reservation keeps
    // its stored logits within INK2_LOGITS_CAP (16 MiB) and still equals the reference kernel bit for bit
    {
        static const int Ts[] = { 1, 15, 16, 17, 48 };
        static const struct { int type; const char *name; } ty[] = { { T_BF16, "BF16" }, { T_NVFP4, "NVFP4" } };
        for (size_t u = 0; u < 2; u++)
            for (size_t k = 0; k < sizeof Ts / sizeof *Ts; k++) {
                long nv = 0;
                const long d = lin_identity(ty[u].type, 512, 384, Ts[k], &nv);
                const long want = Ts[k] >= 16;
                const size_t xw = sizeof(float) * ((size_t)Ts[k] + 16) * 512;
                CHECK(d == 0 && nv > 0 && g_obs.nb == want && g_obs.caps[0] >= xw, "(ae) a %s product at %d columns: %ld "
                      "batched dispatches (want %ld), its x buffer %zu bytes (want >= %zu, %d columns + 16)", ty[u].name,
                      Ts[k], g_obs.nb, want, g_obs.caps[0], xw, Ts[k]);
            }
        static const struct { int type, n_in, n_out, T; const char *what; } wc[] = {
            { T_BF16, 2304, 3072, 17, "BF16 at q_proj's widths (2304 -> 3072), 17 columns (a tile of 16 and one of 1)" },
            { T_NVFP4, 2304, 2304, 33, "NVFP4 at gate_up's widths (2304 -> 2304), 33 columns (two tiles and one of 1)" },
        };
        for (size_t k = 0; k < sizeof wc / sizeof *wc; k++) {
            long nv = 0;
            const long d = lin_identity(wc[k].type, wc[k].n_in, wc[k].n_out, wc[k].T, &nv);
            CHECK(d == 0 && nv > 0 && g_obs.nb == 1, "(ae) %s: %ld of %ld values differ from the per-column launches' "
                  "(want 0); %ld batched dispatches (want 1)", wc[k].what, d, nv, g_obs.nb);
        }
        static const struct { int T, all; const char *what; } ec[] = {
            { 17, 3, "17 positions, expert 3 at every position (17 columns: tiled)" },
            { 15, 3, "15 positions, expert 3 at every position (15 columns: no expert tiles)" },
        };
        for (size_t k = 0; k < sizeof ec / sizeof *ec; k++) {
            long nv = 0;
            const long d = exp_identity(T_NVFP4, 256, 128, 16, ec[k].T, 2, ec[k].all, -1, &nv);
            const long want = g_obs.nbig > 0 ? 2 : 0;
            const size_t we = sizeof(float) * ((size_t)g_obs.nall + 16) * 256, wa = sizeof(float) * ((size_t)g_obs.nall + 16) * (2 * 128);
            CHECK(d == 0 && nv > 0 && g_obs.nb == want && (ec[k].T >= 16) == (g_obs.nbig > 0) && g_obs.caps[2] >= we &&
                  g_obs.caps[3] >= wa, "(ae) the NVFP4 expert body, %s: %ld tiled dispatches (want %ld; %ld experts of 16+ "
                  "columns); the gathered x %zu bytes (want >= %zu) and the activations %zu (want >= %zu: %ld columns + "
                  "16)", ec[k].what, g_obs.nb, want, g_obs.nbig, g_obs.caps[2], we, g_obs.caps[3], wa, g_obs.nall);
        }
        static const int c_one[1] = { 1040 };
        for (int un = 0; un < 2; un++) {
            size_t nf = 0;
            long cmp = 0, nd = 0;
            g_obs.core_unreserved = un;
            const double w = core_case(128, 1, 40, 1100, c_one, 1, &nf, &cmp, &nd);
            g_obs.core_unreserved = false;
            CHECK(cmp > 0 && nd == 0 && nf == 0 && w <= 1e-4 && g_obs.alg_max <= ((size_t)16 << 20) &&
                  g_obs.aatt_min == (size_t)-1, "(ae) the device core %s, 1040 queries x 4 heads x 1040 keys (17.3 MB "
                  "of logits at once): the stored logits' buffer %zu bytes (want <= 16 MiB); an attention output short "
                  "of T + 16 columns: %s; %ld of %ld values differ from the reference kernel's (want 0)",
                  un ? "WITHOUT a reservation" : "reserved as the fit does", g_obs.alg_max,
                  g_obs.aatt_min == (size_t)-1 ? "none" : "YES", nd, cmp);
        }
    }
    // G2b 2a (o): a failed device core refuses that forward and the next; and the REWIND path through the device
    // ring: a sequence A, then B from position 0, then HALF (A's first 12 tokens, then others) from 0 gives a fresh
    // instance's HALF rows byte for byte (nothing of A or B leaks from the ring), within 1e-4 of the CPU fast route's
    {
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_ATTN_FAIL", "2");
        model_t *g = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_ATTN_FAIL", "");
        const bool on = g && model_inkling2_cuda_attn_calls(g, NULL) >= 0;
        const float *f1 = on ? model_inkling2_forward(g, PROMPT, N_TOK, 0, true, true) : NULL;
        const float *f2 = on ? model_inkling2_forward(g, PROMPT, 1, 0, true, true) : NULL;
        CHECK(on && !f1 && !f2, "(o) a failed device attention core: that forward %s, the next %s",
              f1 ? "gave rows" : "refused", f2 ? "gave rows" : "refused");
        drop(g);
        int32_t other[N_TOK], half[N_TOK];
        for (int t = 0; t < N_TOK; t++) { other[t] = (int32_t)((PROMPT[t] * 7 + 3) % 200); half[t] = t < 12 ? PROMPT[t] : other[t]; }
        // at -b 8 (several chunks) and at -b 1 (each chunk shorter than the convolution's history, conv_k - 1 = 3: the
        // device history ring is read at every step and only partly rewritten by each)
        for (int bi = 0; bi < 2; bi++) {
            const int bt = bi ? 1 : 8;
            model_t *gs = load(argv[2], true, false, bt), *gf = load(argv[2], true, false, bt);
            model_t *cf = load_route(argv[2], false, false, bt, "");
            float *ra = gs ? rows_p(gs, PROMPT) : NULL, *rb = gs ? rows_p(gs, other) : NULL, *rh = gs ? rows_p(gs, half) : NULL;
            float *fh = gf ? rows_p(gf, half) : NULL, *ch = cf ? rows_p(cf, half) : NULL;
            const size_t n = gs ? (size_t)N_TOK * (size_t)gs->n_vocab : 0;
            const double dc = rh && ch ? rel_diff(rh, ch, n) : 1;
            // (D1d.3b: the file runs with the step off, so the step is absent here)
            CHECK(gs && model_inkling2_cuda_attn_calls(gs, NULL) + model_inkling2_cuda_block_calls(gs) > 0 &&
                  model_inkling2_cuda_step_calls(gs) == -1 && ra && rb && rh &&
                  fh && !memcmp(rh, fh, sizeof(float) * n) && dc <= 1e-4, "(o) rewind through the device ring (-b %d): A, B, "
                  "then HALF from 0 equal a fresh instance's HALF byte for byte, %.2e from the CPU fast route's, want <= 1e-4",
                  bt, dc);
            free(ra); free(rb); free(rh); free(fh); free(ch);
            drop(gs); drop(gf); drop(cf);
        }
    }
    // (x2) the stage-3 read's F1: the fit's resident-step terms with the step ON, against the bytes actually allocated,
    // each by the driver's rounding: the copy's part (the step's small tensors + its program's 16 allocations, each
    // equal to the fit's term in order, the index list at most its bound) and one slot's post-convolution copies
    {
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "");
        model_t *g = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_STEP", "1");
        const float *f = g ? model_inkling2_forward(g, PROMPT, 1, 0, true, false) : NULL;   // one step: the program is up
        size_t sb[5] = { 0 };
        const bool on = f && model_inkling2_cuda_step_calls(g) == 1 && model_inkling2_cuda_step_bytes(g, sb);
        CHECK(on && sb[4] == 1 && sb[1] <= sb[0] && sb[1] > 0 && sb[2] == sb[3] && sb[3] > 0,
              "(x2) the fit's step terms with the step on: the copy's part counted %zu, allocated %zu (want <= and > 0); the "
              "16 allocations each the fit's term: %s; one slot's post copies counted %zu, allocated %zu (want equal, > 0)",
              sb[0], sb[1], sb[4] ? "yes" : "NO", sb[2], sb[3]);
        drop(g);
    }
    // (x) the second read of 2b's P2: the fit counts what a slot and the block allocate; the block refuses an adapter
    // in the middle of a sequence it began and leaves the slot where it was
    {
        model_t *g = load(argv[2], true, false, 64);
        const void *xsp = NULL;
        if (g) model_inkling2_parts(g, &xsp, NULL);
        const ink2_shape *xs = xsp;
        // one slot by hand (the driver's rounding: 512 bytes under 1 MiB, 2 MiB pages from it): per application its
        // K and V rings (rows: -c 64, a sliding ring's window - 1 + 64 = 71 capped there) and its two histories
        // (conv_k - 1 = 3 rows); every row 32 floats on the tiny file
        size_t want = 0;
        for (int ap = 0; xs && ap < xs->n_app; ap++) {
            const int gl = ink2_app_global(xs, ap);
            const size_t C = (size_t)(gl ? xs->n_kv * xs->head_dim : xs->swa_n_kv * xs->swa_head_dim);
            const size_t rws = gl || xs->window - 1 + 64 >= 64 ? 64 : (size_t)(xs->window - 1 + 64);
            const size_t b[2] = { sizeof(float) * rws * C, sizeof(float) * (size_t)(xs->conv_k - 1) * C };
            for (int k = 0; k < 2; k++) {
                const size_t gr = b[k] >= ((size_t)1 << 20) ? (size_t)2 << 20 : 512;
                want += 2 * ((b[k] + gr - 1) / gr * gr);
            }
        }
        const size_t slot = model_inkling2_cuda_slot_fit_bytes(g);
        CHECK(g && model_inkling2_cuda_block_calls(g) >= 0 && slot == want && want > 0, "(x) the fit counts one slot's K/V "
              "rings and convolution histories: %zu bytes, want %zu", slot, want);
        // the block's ten buffers reserved before the fit read the free memory: their bytes before any forward are
        // T = 64 x (4 H + 3 C + nkv K + nh dr + Q) floats, and a forward does not grow them
        const size_t bb0 = model_inkling2_cuda_block_buf_bytes(g);
        float *r0 = g ? rows(g) : NULL;
        const size_t bb1 = model_inkling2_cuda_block_buf_bytes(g);
        // (the three inputs q, k, v with P1a's 16 columns of padding the batched BF16 product reads)
        const size_t want_b = xs ? sizeof(float) * ((64 + 16) * (size_t)(3 * xs->hidden) +
                                                    64 * (size_t)(xs->hidden + 3 * xs->n_kv * xs->head_dim +
                                                                  xs->n_kv * xs->conv_k + xs->n_head * xs->d_rel +
                                                                  xs->n_head * xs->head_dim)) : 0;
        CHECK(g && r0 && bb0 == want_b && bb1 == bb0 && want_b > 0, "(x) the attention block's buffers before the first "
              "forward %zu bytes (want %zu, reserved with the fit), %zu after it (want unchanged)", bb0, want_b, bb1);
        // THE CONTRACT: an adapter active at a position past 0 of a sequence the device block began is refused (a
        // sentence on stderr: "start it again from position 0"), because the host holds no convolution history for that
        // sequence; the slot stays at that position, so without the adapter the sequence continues there, and with it a
        // sequence begun again at 0 runs (2a's path, its terms on the host)
        const size_t nv = g ? (size_t)g->n_vocab : 0;
        const float *p1 = g ? model_inkling2_forward(g, PROMPT, 12, 0, true, true) : NULL;
        const bool lo = p1 && model_lora_load(g, argv[5], 1.0f);
        const float *p2 = lo ? model_inkling2_forward(g, PROMPT + 12, 12, 12, true, true) : NULL;
        if (lo) model_lora_free(g);
        const float *p3 = g ? model_inkling2_forward(g, PROMPT + 12, 12, 12, true, true) : NULL;
        const bool same = p3 && r0 && !memcmp(model_inkling2_row(g, 11), r0 + 23 * nv, sizeof(float) * nv) &&
                          !memcmp(model_inkling2_row(g, 0), r0 + 12 * nv, sizeof(float) * nv);
        CHECK(p1 && lo && !p2 && same, "(x) an adapter at position 12 of a sequence the device block began: %s (want "
              "refused); without it the sequence %s at 12, rows %s the one-call forward's", p2 ? "ran" : "refused",
              p3 ? "continues" : "does not continue", same ? "equal to" : "not equal to");
        const bool l2 = g && model_lora_load(g, argv[5], 1.0f);
        model_t *gl = load(argv[2], true, false, 64);
        const bool l3 = gl && model_lora_load(gl, argv[5], 1.0f);
        float *ra = l2 ? rows(g) : NULL, *rf = l3 ? rows(gl) : NULL;
        const size_t n = (size_t)N_TOK * nv;
        CHECK(ra && rf && !n_nonfinite(ra, n) && !memcmp(ra, rf, sizeof(float) * n), "(x) the same instance with the adapter "
              "from position 0: rows equal a fresh instance's with it byte for byte (not refused, not failed)");
        free(r0); free(ra); free(rf);
        drop(g); drop(gl);
    }
    // G2b 2a (p) at N slots: each slot holds its own K/V rings, so the fit counts them N times. With the free figure
    // capped at the whole need + 2 rings + the reserve: one slot takes every weight (W); three slots do not.
    {
        model_t *one = load(argv[2], true, false, 64);
        char p1 = 0;
        size_t n1 = 0, ring = 0;
        model_inkling2_cuda_placement(one, &p1, &n1, NULL);
        model_inkling2_cuda_attn_calls(one, &ring);
        drop(one);
        char v[32];
        snprintf(v, sizeof v, "%zu", (size_t)(512ull << 20) + n1 + 2 * ring);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_FREE", v);
        n_seq = 1;
        model_t *s1 = load(argv[2], true, false, 64);
        n_seq = 3;
        model_t *s3a = load(argv[2], true, false, 64), *s3b = load(argv[2], true, false, 64);
        n_seq = 0;
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_FREE", "");
        char q1 = 0, q3a = 0, q3b = 0;
        model_inkling2_cuda_placement(s1, &q1, NULL, NULL);
        model_inkling2_cuda_placement(s3a, &q3a, NULL, NULL);
        model_inkling2_cuda_placement(s3b, &q3b, NULL, NULL);
        CHECK(p1 == 'W' && ring > 0 && q1 == 'W' && q3a != 'W' && q3a == q3b && q3a != 0,
              "(p) N slots in the fit (rings %zu bytes a slot, the whole need %zu): 1 slot %c (want W), 3 slots %c %c (want "
              "not W, the same for both)", ring, n1, q1 ? q1 : '-', q3a ? q3a : '-', q3b ? q3b : '-');
        drop(s1); drop(s3a); drop(s3b);
    }
    // G2b step 2's B (p): one device copy of the weights per file and placement, shared and refcounted; the fit's
    // outcome (W whole, M expert stacks on the host, C the CPU) the same for every instance, never mixed. Bars fixed
    // before the run: the free memory back within 2 MiB once both are freed; the M rows within 1e-4 of the reference.
    {
        const size_t RES = 512ull << 20;   // the fit's reserve (src/inkling2_model.c)
        size_t f0 = 0, tt = 0;
        gpu_mem_info(&f0, &tt);
        model_t *a = load(argv[2], true, false, 64), *b = load(argv[2], true, false, 64);
        char pa = 0, pb = 0;
        size_t na = 0, nb = 0;
        const void *ha = NULL, *hb = NULL;
        model_inkling2_cuda_placement(a, &pa, &na, &ha);
        model_inkling2_cuda_placement(b, &pb, &nb, &hb);
        size_t ring_b = 0;   // one slot's K/V rings (step 2a), which the fit adds to every placement's need
        model_inkling2_cuda_attn_calls(a, &ring_b);
        const size_t n = b ? (size_t)N_TOK * (size_t)b->n_vocab : 0;
        float *ra = a ? rows(a) : NULL, *rb = b ? rows(b) : NULL;
        CHECK(pa == 'W' && pb == 'W' && ha && ha == hb, "(p) two loads of one file: placements %c %c, %s device copy",
              pa ? pa : '-', pb ? pb : '-', ha && ha == hb ? "ONE shared" : "two");
        CHECK(ra && rb && !memcmp(ra, rb, sizeof(float) * n), "(p) the second instance's rows equal the first's byte for byte");
        {   // both instances forwarding at once on the shared copy (its buffers are used under its lock)
            par_job ja = { a, { 0 } }, jb = { b, { 0 } };
            pthread_t ta, tb;
            const bool st = a && b && !pthread_create(&ta, NULL, par_run, &ja) && !pthread_create(&tb, NULL, par_run, &jb);
            if (st) { pthread_join(ta, NULL); pthread_join(tb, NULL); }
            int bad = 0;
            for (int k = 0; k < 4; k++) {
                bad += !ja.out[k] || memcmp(ja.out[k], ra, sizeof(float) * n);
                bad += !jb.out[k] || memcmp(jb.out[k], ra, sizeof(float) * n);
                free(ja.out[k]); free(jb.out[k]);
            }
            CHECK(st && bad == 0, "(p) two instances forwarding at once on one device copy, 4 forwards each: %d of 8 differ "
                  "from the sequential rows", bad);
        }
        {   // a bind in other instances while one forwards (the second read of 2a: the device hooks must be the
            // instances' own, never a process-wide pointer a bind writes while another instance's forward reads it;
            // functionally the old race was benign, so ThreadSanitizer over this case is its red)
            par_job jc = { a, { 0 } };
            pthread_t tc;
            const bool st = a && !pthread_create(&tc, NULL, par_run, &jc);
            for (int k = 0; st && k < 3; k++) drop(load(argv[2], true, false, 64));
            if (st) pthread_join(tc, NULL);
            int bad = 0;
            for (int k = 0; k < 4; k++) { bad += !jc.out[k] || memcmp(jc.out[k], ra, sizeof(float) * n); free(jc.out[k]); }
            CHECK(st && bad == 0, "(p) three binds while another instance forwards 4 times: %d of 4 differ from its sequential "
                  "rows", bad);
        }
        drop(a);
        float *rb2 = b ? rows(b) : NULL;
        CHECK(rb && rb2 && !memcmp(rb, rb2, sizeof(float) * n), "(p) the first instance freed: the second's rows unchanged");
        drop(b);
        size_t f1 = 0;
        gpu_mem_info(&f1, &tt);
        const long long df = (long long)f0 - (long long)f1;
        CHECK(df <= (2ll << 20) && df >= -(2ll << 20), "(p) both freed: the device's free memory %zu, before the first bind %zu "
              "(want within 2 MiB)", f1, f0);
        free(ra); free(rb); free(rb2);
        // the --cpu-moe placement's need, then a cap between it and the whole need: both loads take M and share
        cpu_moe = true;
        model_t *mm = load(argv[2], true, false, 64);
        cpu_moe = false;
        char pm = 0;
        size_t nm = 0;
        model_inkling2_cuda_placement(mm, &pm, &nm, NULL);
        drop(mm);
        CHECK(pm == 'M' && nm > 0 && na > nm, "(p) the needs: whole %zu bytes, the stacks on the host %zu bytes", na, nm);
        char v[32];
        snprintf(v, sizeof v, "%zu", RES + (na + nm) / 2 + ring_b);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_FREE", v);
        model_t *c1 = load(argv[2], true, false, 64), *c2 = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_FREE", "");
        model_t *ref = load(argv[2], false, false, 64);
        char p1 = 0, p2 = 0;
        const void *h1 = NULL, *h2 = NULL;
        model_inkling2_cuda_placement(c1, &p1, NULL, &h1);
        model_inkling2_cuda_placement(c2, &p2, NULL, &h2);
        float *r1 = c1 ? rows(c1) : NULL, *rr = ref ? rows(ref) : NULL;
        const double d1 = r1 && rr ? rel_diff(r1, rr, n) : 1;
        CHECK(p1 == 'M' && p2 == 'M' && h1 && h1 == h2 && !model_inkling2_cuda_holds(c1, "model.layers.0.mlp.experts.gate_up_proj") &&
              d1 <= 1e-4, "(p) free between the two needs: placements %c %c (want M M), %s copy, the stacks on the host, rows "
              "within %.2e of the reference; the deciding fit read %zu bytes free (the cap %s)", p1 ? p1 : '-',
              p2 ? p2 : '-', h1 && h1 == h2 ? "one shared" : "not one", d1, model_inkling2_cuda_fit_free(c1), v);
        free(r1); free(rr);
        drop(c1); drop(c2); drop(ref);
        // a cap below the --cpu-moe need: the CPU for every instance
        snprintf(v, sizeof v, "%zu", RES + nm / 2);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_FREE", v);
        model_t *c3 = load(argv[2], true, false, 64), *c4 = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_FREE", "");
        char p3 = 0, p4 = 0;
        model_inkling2_cuda_placement(c3, &p3, NULL, NULL);
        model_inkling2_cuda_placement(c4, &p4, NULL, NULL);
        CHECK(p3 == 'C' && p4 == 'C' && c3 && c4 && !model_inkling2_cuda(c3) && !model_inkling2_cuda(c4),
              "(p) free below both needs: placements %c %c (want C C), neither on the device", p3 ? p3 : '-', p4 ? p4 : '-');
        drop(c3); drop(c4);
    }
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
