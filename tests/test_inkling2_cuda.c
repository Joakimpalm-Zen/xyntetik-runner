// R4.48.9: Inkling 2 Mini on CUDA, G1 (the weights on the device) and G2 (the forward's pieces) on the tiny
// inkling2_loop files (tests/inkling2_tiny.py: the dense projections F32, BF16 and Q8_0; the experts NVFP4 with their
// per-expert .scale on both stacks). Needs a CUDA device; without one it says so and exits 77 (skipped), never a pass.
//
// TOLERANCES, fixed before the first run (the supervisor's G2 rule):
//   - rows: the CUDA rows within 1e-4 of the reference route's rows, relative to the largest reference logit (the
//     fast CPU route's T1 bound), and NOT byte-identical to them (a different order of sums);
//   - invariance: byte-identical at -b 64 and -b 8, and across two loads (one fixed order per op);
//   - control: the expert factor left out on the device (RUNNER_INKLING2_CUDA_TEST_NO_FACTOR=1, test-only) is at least
//     10 times further from the reference than the CUDA route.
//
// The reference here is the CPU's reference route (RUNNER_INKLING2_ROUTE=reference: the serial float32 engine).
//  G1 (a) the route: loaded with the GPU allowed, the instance runs on CUDA (model_inkling2_cuda true; GPU off: false);
//     (b) every matrix (projections, head, embedding) and both expert stacks, read back from the device through the
//         test-only accessor model_inkling2_cuda_readback, equal the host decode (dequant_row; for the stacks the
//         stored rows, the factor separate) bit for bit;
//  G2 (c) rows near the reference and not equal (above); (d) invariance (above); (e) the control (above).
//
// usage: test_inkling2_cuda TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf
#include "model.h"
#include "gpu.h"
#include "inkling2_model.h"
#include "compat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TOK = 24 };
static const int32_t PROMPT[N_TOK] = { 1, 5, 9, 2, 7, 3, 8, 4, 6, 0, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61 };
static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static model_t *load(const char *path, bool gpu, bool no_factor, int n_batch) {
    // the CPU side is the reference route (the serial float32 engine), not the CPU's default fast route
    plat_setenv("RUNNER_INKLING2_ROUTE", gpu ? "" : "reference");
    plat_setenv("RUNNER_INKLING2_CUDA_TEST_NO_FACTOR", no_factor ? "1" : "");
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = gpu ? GPU_AUTO : GPU_OFF;
    p.n_threads = 2;
    p.n_ctx = 64;
    p.n_batch = n_batch;
    model_t *m = calloc(1, sizeof *m);
    if (!m || !model_load(m, path, &p)) { free(m); return NULL; }
    return m;
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
static double rel_diff(const float *a, const float *ref, size_t n) {
    double d = 0, top = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabs((double)a[i] - ref[i]) > d) d = fabs((double)a[i] - ref[i]);
        if (fabs((double)ref[i]) > top) top = fabs((double)ref[i]);
    }
    return top > 0 ? d / top : d;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s TINY_F32.gguf TINY_BF16.gguf TINY_Q8_0.gguf\n", argv[0]); return 2; }
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
        // G2 (c): only rows a CUDA instance computed count (a CPU fallback's rows would pass (c) and (d) for nothing)
        const bool on_cuda = model_inkling2_cuda(g);
        const size_t n = (size_t)N_TOK * (size_t)g->n_vocab;
        float *a = rows(g), *b = rows(r);
        const double rd = a && b ? rel_diff(a, b, n) : 1;
        CHECK(on_cuda && a && b && rd <= 1e-4, "(c) %s: CUDA rows within %.2e of the reference rows, want <= 1e-4", kind[f], rd);
        CHECK(on_cuda && a && b && memcmp(a, b, sizeof(float) * n), "(c) %s: the CUDA rows are not the reference rows byte for byte",
              kind[f]);
        if (f == 1) {   // (d) and (e) on the BF16 file (the exact file's dense type)
            model_t *g8 = load(argv[2], true, false, 8), *g2 = load(argv[2], true, false, 64);
            model_t *gq = load(argv[2], true, true, 64);
            float *x8 = g8 ? rows(g8) : NULL, *x2 = g2 ? rows(g2) : NULL, *xq = gq ? rows(gq) : NULL;
            CHECK(on_cuda && a && x8 && !memcmp(a, x8, sizeof(float) * n), "(d) the CUDA rows are byte-identical at -b 64 and -b 8");
            CHECK(on_cuda && a && x2 && !memcmp(a, x2, sizeof(float) * n), "(d) the CUDA rows are byte-identical across two loads");
            const double qe = xq && b ? rel_diff(xq, b, n) : 0;
            CHECK(on_cuda && qe >= 10 * rd && qe > 0, "(e) control: the factor left out %.2e from the reference, CUDA %.2e, want >= 10x",
                  qe, rd);
            free(x8); free(x2); free(xq); drop(g8); drop(g2); drop(gq);
        }
        free(a); free(b); drop(g); drop(r);
    }
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
