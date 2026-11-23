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
//
// usage: test_inkling2_cuda TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf TINY_F16.gguf TINY_LORA.gguf
#include "model.h"
#include "gpu.h"
#include "inkling2_model.h"
#include "inkling2.h"
#include "compat.h"
#include "quants.h"
#include <math.h>
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
static double core_case(int hd, int global, int window, int R, const int *chunks, int n_chunks, size_t *nf, long *cmp) {
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
    gpu_ink2_kv *kv = h ? gpu_ink2_kv_new(h, 1, rows, cw, 0) : NULL;
    double worst = 0, top = 0;
    bool ran = kv != NULL;
    for (int c = 0, p0 = 0; ran && c < n_chunks; p0 += chunks[c], c++) {
        const int T = chunks[c];
        ran = gpu_ink2_attn(h, kv, 0, T, pos + p0, q + (size_t)p0 * Q, k + (size_t)p0 * C, v + (size_t)p0 * C,
                            rl + (size_t)p0 * NH * EXT, sk + (size_t)p0 * NH, att, NH, NKV, hd, window, global, EXT, scale);
        if (!ran) break;
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
    gpu_ink2_free(h);
    free(q); free(k); free(v); free(rl); free(sk); free(att); free(lg); free(o); free(pos);
    return ran && top > 0 ? worst / top : 1;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf TINY_F16.gguf TINY_LORA.gguf\n", argv[0]);
        return 2;
    }
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
            CHECK(on_cuda && g1 && model_inkling2_cuda_block_calls(g1) > 0 && x1 && a && !memcmp(a, x1, sizeof(float) * n) &&
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
    // G2 (h), on the BF16 file: the third device product fails; that forward and the next refuse
    {
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_LIN_FAIL", "2");
        model_t *g = load(argv[2], true, false, 64);
        plat_setenv("RUNNER_INKLING2_CUDA_TEST_LIN_FAIL", "");
        const bool on = g && model_inkling2_cuda(g);
        const float *first = on ? model_inkling2_forward(g, PROMPT, N_TOK, 0, true, true) : NULL;
        const float *again = on ? model_inkling2_forward(g, PROMPT, 1, 0, true, true) : NULL;
        CHECK(on && !first && !again, "(h) a failed device product: on CUDA (%d), that forward %s, the next %s", on,
              first ? "gave rows" : "refused", again ? "gave rows" : "refused");
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
    // G2b 2a (n): gpu_ink2_attn against float64 on the host, by core_case below. Bar fixed before the run: 1e-4 of the
    // largest value, finite. A full layer over 1040 positions in one call (every key count 1..1040: around each
    // 32-key chunk); a sliding layer (window 40, ring 103) through chunks of 64, 8 and 64 that straddle the ring's
    // wraps at 103 and 206, then single steps; head_dim 32, 64 and 256 beside 128 (the range the kernel takes); GQA
    // (4 heads on 2 K/V heads), the bias within distance 16, sinks from -20 to +5
    {
        static const int c_full[] = { 1040 }, c_slide[] = { 64, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 64, 1, 1, 1, 1, 1, 1, 1, 1 };
        static const int c_small[] = { 70, 8, 8 }, c_mid[] = { 32, 8, 8, 8, 8, 32 }, c_wide[] = { 64, 1, 1 };
        static const struct { const char *what; int hd, global, window, R; const int *ch; int nch; } cs[] = {
            { "a full layer, 1040 queries in one call (head_dim 128)", 128, 1, 40, 1100, c_full, 1 },
            { "a sliding layer (window 40, ring 103), chunks of 64, 8 and 64 across both of the ring's wraps, then single "
              "steps (head_dim 128)", 128, 0, 40, 103, c_slide, (int)(sizeof c_slide / sizeof *c_slide) },
            { "a full layer, head_dim 32", 32, 1, 40, 200, c_small, 3 },
            { "a sliding layer (window 16, ring 47), head_dim 64", 64, 0, 16, 47, c_mid, 6 },
            { "a full layer, head_dim 256", 256, 1, 40, 100, c_wide, 3 },
        };
        for (size_t k = 0; k < sizeof cs / sizeof *cs; k++) {
            size_t nf = 0;
            long cmp = 0;
            const double w = core_case(cs[k].hd, cs[k].global, cs[k].window, cs[k].R, cs[k].ch, cs[k].nch, &nf, &cmp);
            CHECK(w <= 1e-4 && nf == 0 && cmp > 0, "(n) the device core, %s: within %.2e of the float64 sums, want <= 1e-4; "
                  "%zu not finite (%ld values)", cs[k].what, w, nf, cmp);
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
            CHECK(gs && model_inkling2_cuda_attn_calls(gs, NULL) + model_inkling2_cuda_block_calls(gs) > 0 && ra && rb && rh &&
                  fh && !memcmp(rh, fh, sizeof(float) * n) && dc <= 1e-4, "(o) rewind through the device ring (-b %d): A, B, "
                  "then HALF from 0 equal a fresh instance's HALF byte for byte, %.2e from the CPU fast route's, want <= 1e-4",
                  bt, dc);
            free(ra); free(rb); free(rh); free(fh); free(ch);
            drop(gs); drop(gf); drop(cf);
        }
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
        const size_t want_b = xs ? sizeof(float) * 64 * (size_t)(4 * xs->hidden + 3 * xs->n_kv * xs->head_dim +
                                                                 xs->n_kv * xs->conv_k + xs->n_head * xs->d_rel +
                                                                 xs->n_head * xs->head_dim) : 0;
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
