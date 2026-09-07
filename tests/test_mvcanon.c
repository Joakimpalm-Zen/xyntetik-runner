// R8.7.1 slice 1: the forward matvec on the device, in the CANONICAL order.
//
// The trainer's forward is 80% of a training step and sits on the CPU only
// because moving it would cost the byte contract: the device forward reduces
// in a different order from the host's, so an adapter trained through it
// would stop being byte-identical to one trained on the CPU. This gate is the
// argument that the cost is avoidable. RUNNER_CANON_KERNELS already defines
// the reduction as an eight-lane tree, for cross-ISA bit-exactness on the CPU
// (experiment R12.1) -- and eight lanes with a three-step tree is a shape a
// warp runs natively. So the device can compute the SAME association rather
// than a good approximation of it.
//
// The comparison is memcmp, not a tolerance. A tolerance here would pass the
// ordinary CPU/GPU divergence this exists to remove.
//
// quants.c is compiled INTO this test with the define, in the engine's strict
// float regime, exactly as test_canon_kernels.c does: the shared object is
// built without it and would answer a different question.
//
// Skips (not passes) without a CUDA device or without a kernel for the type.
// F32, F16 and Q8_0 have a canonical order; the k-quants do not yet, and
// gpu_mvcanon refuses them by name rather than guessing one.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static double clk(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint64_t rng = 0x243F6A8885A308D3ull;
static float frnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)(int32_t)(uint32_t)(rng >> 32) / 2147483648.0f;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    f16_init();
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_AUTO;
    p.n_threads = 2;
    p.n_ctx = 64;
    model_t m;
    memset(&m, 0, sizeof(m));
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "FAIL: cannot load %s\n", path);
        return 1;
    }
    if (!m.gpu) {
        printf("mvcanon: no GPU backend — skipping (not passing)\n");
        model_free(&m);
        return 0;
    }

    int tested = 0;
    for (int l = 0; l < m.n_layer && l < 2; l++) {
        gguf_tensor *ws[] = { m.layers[l].wq, m.layers[l].w_down,
                              m.layers[l].w_up };
        for (size_t k = 0; k < sizeof(ws) / sizeof(*ws); k++) {
            gguf_tensor *w = ws[k];
            if (!w) continue;
            int n_in = (int)w->ne[0], n_out = (int)w->ne[1];
            float *x = malloc(sizeof(float) * (size_t)n_in);
            float *y_cpu = malloc(sizeof(float) * (size_t)n_out);
            float *y_gpu = malloc(sizeof(float) * (size_t)n_out);
            float *y_gpu2 = malloc(sizeof(float) * (size_t)n_out);
            if (!x || !y_cpu || !y_gpu || !y_gpu2) return 1;
            for (int i = 0; i < n_in; i++) x[i] = frnd();
            memset(y_gpu, 0xA5, sizeof(float) * (size_t)n_out);
            memset(y_gpu2, 0x5A, sizeof(float) * (size_t)n_out);

            if (!gpu_mvcanon(&m, w, x, y_gpu, n_in, n_out)) {
                printf("mvcanon: %s — no canonical kernel (type %d), "
                       "skipping\n", w->name, w->type);
                free(x); free(y_cpu); free(y_gpu); free(y_gpu2);
                continue;
            }
            size_t rs = ggml_row_size(w->type, n_in);
            for (int r = 0; r < n_out; r++)
                y_cpu[r] = vec_dot(w->type,
                                   (const uint8_t *)w->data + (size_t)r * rs,
                                   x, n_in);
            CHECK(memcmp(y_cpu, y_gpu, sizeof(float) * (size_t)n_out) == 0,
                  "%s [%d x %d]: device result differs from the canonical "
                  "host dot", w->name, n_in, n_out);
            CHECK(gpu_mvcanon(&m, w, x, y_gpu2, n_in, n_out) &&
                  memcmp(y_gpu, y_gpu2, sizeof(float) * (size_t)n_out) == 0,
                  "%s: device result not deterministic across runs", w->name);
            tested++;
            free(x); free(y_cpu); free(y_gpu); free(y_gpu2);
        }
    }
    if (!tested) {
        printf("mvcanon: no canonical kernels for this model's types — "
               "skipping (not passing)\n");
        model_free(&m);
        return 0;
    }

    // throughput on the widest tensor, printed rather than asserted: the
    // question this slice exists to answer is whether the exact order costs
    // anything, and the honest comparison is against the host doing the SAME
    // association, which is what this binary's vec_dot is
    if (m.output) {
        gguf_tensor *w = m.output;
        int n_in = (int)w->ne[0], n_out = (int)w->ne[1];
        float *x = malloc(sizeof(float) * (size_t)n_in);
        float *y = malloc(sizeof(float) * (size_t)n_out);
        for (int i = 0; i < n_in; i++) x[i] = frnd();
        if (x && y && gpu_mvcanon(&m, w, x, y, n_in, n_out)) {
            int reps = 10;
            double t0 = clk();
            for (int r = 0; r < reps; r++)
                gpu_mvcanon(&m, w, x, y, n_in, n_out);
            double gpu_s = (clk() - t0) / reps;
            size_t rs = ggml_row_size(w->type, n_in);
            t0 = clk();
            for (int r = 0; r < n_out; r++)
                y[r] = vec_dot(w->type,
                               (const uint8_t *)w->data + (size_t)r * rs,
                               x, n_in);
            double cpu_s = clk() - t0;
            printf("mvcanon: head [%d x %d] gpu %.3f ms  cpu(1 thread, canon) "
                   "%.3f ms  (%.1fx)\n", n_in, n_out, gpu_s * 1e3,
                   cpu_s * 1e3, cpu_s / gpu_s);
        }
        free(x); free(y);
    }
    model_free(&m);
    printf(g_fail ? "mvcanon: FAILED\n" : "mvcanon: ok\n");
    return g_fail;
}
