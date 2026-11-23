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
//     (b) every matrix (projections, head, embedding) and both expert stacks, read back from the device through the
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
//
// usage: test_inkling2_cuda TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf TINY_F16.gguf
#include "model.h"
#include "gpu.h"
#include "inkling2_model.h"
#include "compat.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TOK = 24 };
static const int32_t PROMPT[N_TOK] = { 1, 5, 9, 2, 7, 3, 8, 4, 6, 0, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61 };
static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static bool cpu_moe;   // the next loads' --cpu-moe
static model_t *load_route(const char *path, bool gpu, bool no_factor, int n_batch, const char *route) {
    plat_setenv("RUNNER_INKLING2_ROUTE", route);
    plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_FACTOR", no_factor ? "1" : "");
    plat_setenv("RUNNER_INKLING2_CUDA", gpu ? "1" : "");   // opt-in until the CUDA forward is complete
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = gpu ? GPU_AUTO : GPU_OFF;
    p.n_threads = 2;
    p.n_ctx = 64;
    p.n_batch = n_batch;
    p.cpu_moe = cpu_moe;
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
static float *rows(model_t *m) {
    float *r = malloc(sizeof(float) * (size_t)N_TOK * (size_t)m->n_vocab);
    for (int pos = 0; pos < N_TOK;) {
        const int n = N_TOK - pos < m->n_batch ? N_TOK - pos : m->n_batch;
        if (!model_inkling2_forward(m, PROMPT + pos, n, pos, true, true)) { free(r); return NULL; }
        for (int b = 0; b < n; b++)
            memcpy(r + (size_t)(pos + b) * (size_t)m->n_vocab, model_inkling2_row(m, b), sizeof(float) * (size_t)m->n_vocab);
        pos += n;
    }
    return r;
}
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

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf TINY_F16.gguf\n", argv[0]); return 2; }
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
        float *a = rows(g), *b = rows(r);
        const double rd = a && b ? rel_diff(a, b, n) : 1;
        const size_t nf = a ? n_nonfinite(a, n) : 0;
        CHECK(on_cuda && a && b && nf == 0 && rd <= 1e-4, "(c) %s: CUDA rows within %.2e of the reference rows, want <= 1e-4; %zu "
              "values not finite, want 0", kind[f], rd, nf);
        CHECK(on_cuda && a && b && memcmp(a, b, sizeof(float) * n), "(c) %s: the CUDA rows are not the reference rows byte for byte",
              kind[f]);
        if (f == 1) {   // (d) and (e) on the BF16 file (the exact file's dense type)
            model_t *g8 = load(argv[2], true, false, 8), *g2 = load(argv[2], true, false, 64);
            model_t *gq = load(argv[2], true, true, 64);
            float *x8 = g8 ? rows(g8) : NULL, *x2 = g2 ? rows(g2) : NULL, *xq = gq ? rows(gq) : NULL;
            CHECK(on_cuda && a && nf == 0 && x8 && !memcmp(a, x8, sizeof(float) * n), "(d) the CUDA rows are byte-identical at -b 64 and -b 8");
            CHECK(on_cuda && a && nf == 0 && x2 && !memcmp(a, x2, sizeof(float) * n), "(d) the CUDA rows are byte-identical across two loads");
            const double qe = xq && b && !n_nonfinite(xq, n) ? rel_diff(xq, b, n) : 0;   // NaN rows are no control
            CHECK(on_cuda && qe >= 10 * rd && qe > 0 && qe < 1e30, "(e) control: the factor left out %.2e from the reference, CUDA %.2e, want >= 10x",
                  qe, rd);
            free(x8); free(x2); free(xq); drop(g8); drop(g2); drop(gq);
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
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
