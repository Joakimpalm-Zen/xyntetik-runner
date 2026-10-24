// R8.19: the training products on the device, every weight type the CPU
// trainer decodes, on random blocks (scale fields made finite).
//
//   transposed  gpu_train_mvt == the CPU trainer's chain (dequant_row, then
//               an ascending-j fmaf per element, a zero dy skipped), memcmp
//   forward     gpu_train_mm is batch-invariant: every column of a 77-column
//               call (the staged kernel, two column tiles) equals the same
//               column computed alone and in a 5-column call (the 16-column
//               kernel) and in the 77-column call with the staged kernel off
//               (RUNNER_TRAIN_MMS=0: the 32-column kernel), memcmp; and it is
//               the product (each column within 2^-16 of sum|w x| of a
//               double-precision reference)
//
// Skips (not passes) without a CUDA device. Prints each type's throughput.
// RUNNER_TRAIN_GPU_BENCH=1 adds a kernel benchmark: four types at 4096 x
// 5120, 64 and 512 columns, the staged kernel against the per-warp ones
// (the kernels' time alone, best of five).
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

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static float frnd(void) { return (float)(int32_t)rnd32() / 2147483648.0f; }

static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static uint16_t get16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
// a finite block scale, either sign, 2^-10 .. 2^-4
static uint16_t rnd_scale(void) {
    float v = ldexpf(1.0f + (float)(rnd32() % 1024) / 1024.0f, -10 + (int)(rnd32() % 7));
    return f32_to_f16(rnd32() & 1 ? -v : v);
}

static const struct { int type; const char *name; int f16[2]; } TYPES[] = {
    { T_F32, "f32", {-1, -1} },        { T_F16, "f16", {-1, -1} },
    { T_BF16, "bf16", {-1, -1} },      { T_Q8_0, "q8_0", {0, -1} },
    { T_Q4_0, "q4_0", {0, -1} },       { T_Q4_1, "q4_1", {0, 2} },
    { T_Q5_0, "q5_0", {0, -1} },       { T_Q5_1, "q5_1", {0, 2} },
    { T_Q2_K, "q2_K", {80, 82} },      { T_Q3_K, "q3_K", {108, -1} },
    { T_Q4_K, "q4_K", {0, 2} },        { T_Q5_K, "q5_K", {0, 2} },
    { T_Q6_K, "q6_K", {208, -1} },     { T_IQ4_NL, "iq4_nl", {0, -1} },
    { T_IQ4_XS, "iq4_xs", {0, -1} },   { T_IQ2_XXS, "iq2_xxs", {0, -1} },
    { T_IQ2_XS, "iq2_xs", {0, -1} },   { T_IQ2_S, "iq2_s", {0, -1} },
    { T_IQ3_XXS, "iq3_xxs", {0, -1} }, { T_IQ3_S, "iq3_s", {0, -1} },
    { T_IQ1_S, "iq1_s", {0, -1} },     { T_IQ1_M, "iq1_m", {-1, -1} },
    { T_MXFP4, "mxfp4", {-1, -1} },    { T_NVFP4, "nvfp4", {-1, -1} },
    { T_Q2_0, "q2_0", {0, -1} },
};

// n_out rows of n_in random weights of the type
static uint8_t *make_rows(int type, const int *f16, int n_in, int n_out, uint64_t *nbytes) {
    size_t rs = ggml_row_size(type, n_in);
    *nbytes = rs * (size_t)n_out;
    uint8_t *w = malloc(*nbytes);
    if (type == T_F32 || type == T_F16 || type == T_BF16) {
        for (size_t e = 0; e < (size_t)n_in * n_out; e++) {
            float v = frnd();
            if (type == T_F32) memcpy(w + 4 * e, &v, 4);
            else if (type == T_F16) put16(w + 2 * e, f32_to_f16(v));
            else { uint32_t u; memcpy(&u, &v, 4); put16(w + 2 * e, (uint16_t)(u >> 16)); }
        }
        return w;
    }
    for (size_t b = 0; b < *nbytes; b++) w[b] = (uint8_t)rnd32();
    size_t ts = ggml_type_size(type), nb = *nbytes / ts;
    for (size_t k = 0; k < nb; k++) {
        uint8_t *blk = w + k * ts;
        for (int f = 0; f < 2; f++)
            if (f16[f] >= 0) put16(blk + f16[f], rnd_scale());
        if (type == T_IQ1_M) {
            // the block's fp16 scale lives in the top nibbles of the four
            // 16-bit scale words at byte 48
            uint16_t h = rnd_scale();
            for (int s = 0; s < 4; s++) {
                uint16_t sw = get16(blk + 48 + 2 * s);
                sw = (uint16_t)((sw & 0x0FFF) | (((h >> (4 * s)) & 0xF) << 12));
                put16(blk + 48 + 2 * s, sw);
            }
        }
        if (type == T_MXFP4) blk[0] = (uint8_t)(118 + rnd32() % 16);   // 2^-9 .. 2^6
    }
    return w;
}

// the CPU trainer's chain (model.c mvt_worker)
static void cpu_mvt(const gguf_tensor *w, const float *dy, float *dx,
                    int n_in, int n_out, int batch) {
    float *row = malloc(sizeof(float) * (size_t)n_in);
    size_t rs = ggml_row_size(w->type, n_in);
    for (int t = 0; t < batch; t++)
        for (int j = 0; j < n_out; j++) {
            float d = dy[(size_t)t * n_out + j];
            if (d == 0.0f) continue;
            dequant_row(w->type, (const uint8_t *)w->data + (size_t)j * rs, row, n_in);
            for (int i = 0; i < n_in; i++)
                dx[(size_t)t * n_in + i] = fmaf(row[i], d, dx[(size_t)t * n_in + i]);
        }
    free(row);
}

static double clk(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(void) {
    f16_init();
    model_t m;
    memset(&m, 0, sizeof(m));
    m.train_gpu_budget = (uint64_t)1 << 30;   // no model file: the arena is this
    if (!gpu_train_init(&m)) {
        printf("train-gpu: no CUDA device, skipping (not passing)\n");
        return 0;
    }
    // 77 positions: two of k_tmvt's 64-position tiles and three of k_tmmw's
    // 32-column tiles, each with a remainder
    enum { N_IN = 512, N_OUT = 100, B = 77 };   // 100 rows: a partial row block
    float *dy = malloc(sizeof(float) * B * N_OUT);
    float *dx0 = malloc(sizeof(float) * B * N_IN);
    float *dxc = malloc(sizeof(float) * B * N_IN), *dxg = malloc(sizeof(float) * B * N_IN);
    float *x = malloc(sizeof(float) * B * N_IN);
    float *y = malloc(sizeof(float) * B * N_OUT), *y5 = malloc(sizeof(float) * 5 * N_OUT);
    float *y1 = malloc(sizeof(float) * N_OUT), *row = malloc(sizeof(float) * N_IN);
    for (int i = 0; i < B * N_OUT; i++) dy[i] = i % 7 == 0 ? 0.0f : frnd();  // the skip
    for (int i = 0; i < B * N_IN; i++) { dx0[i] = frnd(); x[i] = frnd(); }
    int tested = 0;
    // every type's rows stay allocated until the context is freed: the
    // context knows a tensor by its host address and size (weights live for
    // the run in a mapped file), so a freed buffer the next type's malloc
    // got back would be served the previous type's device copy
    void *keep[sizeof(TYPES) / sizeof(*TYPES) + 1] = {0};
    for (size_t k = 0; k < sizeof(TYPES) / sizeof(*TYPES); k++) {
        gguf_tensor w;
        memset(&w, 0, sizeof(w));
        snprintf(w.name, sizeof w.name, "%s", TYPES[k].name);
        w.type = (uint32_t)TYPES[k].type;
        w.ne[0] = N_IN; w.ne[1] = N_OUT;
        w.scale = 1.0f;
        w.data = make_rows(TYPES[k].type, TYPES[k].f16, N_IN, N_OUT, &w.nbytes);
        // transposed: the CPU chain's bytes
        memcpy(dxc, dx0, sizeof(float) * B * N_IN);
        memcpy(dxg, dx0, sizeof(float) * B * N_IN);
        cpu_mvt(&w, dy, dxc, N_IN, N_OUT, B);
        double t0 = clk();
        bool ok = gpu_train_mvt(&m, &w, dy, dxg, N_IN, N_OUT, B);
        double t1 = clk();
        CHECK(ok, "%s: gpu_train_mvt declined", TYPES[k].name);
        if (ok) {
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < (size_t)B * N_IN; i++)
                if (memcmp(&dxc[i], &dxg[i], 4) && !bad++) first = i;
            CHECK(!bad, "%s: transposed differs from the CPU chain at %zu of %d "
                  "elements (first [%zu]: cpu %.9g gpu %.9g)", TYPES[k].name, bad,
                  B * N_IN, first, dxc[first], dxg[first]);
        }
        // forward: batch-invariant columns, and the product
        ok = gpu_train_mm(&m, &w, x, N_IN, y, N_OUT, N_IN, N_OUT, NULL, B);
        double t2 = clk();
        CHECK(ok, "%s: gpu_train_mm declined", TYPES[k].name);
        if (ok) {
            // the same 77 columns through the per-warp kernels
            float *yw = malloc(sizeof(float) * B * N_OUT);
            setenv("RUNNER_TRAIN_MMS", "0", 1);
            bool okw = gpu_train_mm(&m, &w, x, N_IN, yw, N_OUT, N_IN, N_OUT, NULL, B);
            unsetenv("RUNNER_TRAIN_MMS");
            CHECK(okw && !memcmp(yw, y, sizeof(float) * B * N_OUT),
                  "%s: the staged kernel's %d columns differ from the per-warp "
                  "kernel's", TYPES[k].name, B);
            free(yw);
        }
        bool ok5 = ok && gpu_train_mm(&m, &w, x + 3 * N_IN, N_IN, y5, N_OUT, N_IN,
                                      N_OUT, NULL, 5);
        CHECK(!ok || ok5, "%s: gpu_train_mm declined 5 columns", TYPES[k].name);
        if (ok5) {
            CHECK(!memcmp(y5, y + 3 * N_OUT, sizeof(float) * 5 * N_OUT),
                  "%s: columns 3..7 differ between a %d- and a 5-column call",
                  TYPES[k].name, B);
            double worst = 0;
            for (int c = 0; c < B; c++) {
                bool one = gpu_train_mm(&m, &w, x + (size_t)c * N_IN, N_IN, y1, N_OUT,
                                        N_IN, N_OUT, NULL, 1);
                CHECK(one && !memcmp(y1, y + (size_t)c * N_OUT, sizeof(float) * N_OUT),
                      "%s: column %d alone differs from the %d-column call",
                      TYPES[k].name, c, B);
                for (int j = 0; j < N_OUT; j++) {
                    dequant_row(w.type, (const uint8_t *)w.data +
                                (size_t)j * ggml_row_size(w.type, N_IN), row, N_IN);
                    double s = 0, a = 0;
                    for (int i = 0; i < N_IN; i++) {
                        s += (double)row[i] * x[(size_t)c * N_IN + i];
                        a += fabs((double)row[i] * x[(size_t)c * N_IN + i]);
                    }
                    double e = a > 0 ? fabs(y[(size_t)c * N_OUT + j] - s) / a : 0;
                    if (e > worst) worst = e;
                }
            }
            CHECK(worst < 1.0 / 65536, "%s: forward is not the product (relative "
                  "error %.3g of sum|w x|)", TYPES[k].name, worst);
            printf("train-gpu %-8s transposed bytes == CPU, forward invariant, "
                   "err %.2g; %.2f / %.2f ms\n", TYPES[k].name, worst,
                   (t1 - t0) * 1e3, (t2 - t1) * 1e3);
        }
        keep[k] = w.data;
        tested++;
    }
    // a bias rides on the host after the product; a strided x and y move
    // only their own rows
    {
        gguf_tensor w;
        memset(&w, 0, sizeof(w));
        w.type = T_Q8_0; w.scale = 1.0f; w.ne[0] = N_IN; w.ne[1] = N_OUT;
        int f16[2] = {0, -1};
        w.data = make_rows(T_Q8_0, f16, N_IN, N_OUT, &w.nbytes);
        float bias[N_OUT];
        for (int j = 0; j < N_OUT; j++) bias[j] = frnd();
        enum { XS = N_IN + 8, YS = N_OUT + 5 };
        float *xs = calloc((size_t)3 * XS, sizeof(float)), *ys = malloc(sizeof(float) * 3 * YS);
        for (int c = 0; c < 3; c++) memcpy(xs + (size_t)c * XS, x + (size_t)c * N_IN, sizeof(float) * N_IN);
        for (int i = 0; i < 3 * YS; i++) ys[i] = -7.0f;
        bool ok = gpu_train_mm(&m, &w, xs, XS, ys, YS, N_IN, N_OUT, bias, 3) &&
                  gpu_train_mm(&m, &w, x, N_IN, y, N_OUT, N_IN, N_OUT, NULL, 3);
        CHECK(ok, "strided call declined");
        for (int c = 0; ok && c < 3; c++) {
            for (int j = 0; j < N_OUT; j++)
                CHECK(ys[(size_t)c * YS + j] == y[(size_t)c * N_OUT + j] + bias[j],
                      "strided column %d row %d", c, j);
            for (int j = N_OUT; j < YS; j++)
                CHECK(ys[(size_t)c * YS + j] == -7.0f, "stride gap %d,%d written", c, j);
        }
        // a scaled tensor is the CPU's
        w.scale = 2.0f;
        CHECK(!gpu_train_mm(&m, &w, x, N_IN, y, N_OUT, N_IN, N_OUT, NULL, 1),
              "a scaled tensor went to the device");
        free(xs); free(ys);
        keep[sizeof(TYPES) / sizeof(*TYPES)] = w.data;
    }
    const char *bench = getenv("RUNNER_TRAIN_GPU_BENCH");
    void *bkeep[4] = {0};   // freed after the context, like keep[]
    if (bench && strcmp(bench, "0") != 0) {
        static const struct { int type; const char *name; int f16[2]; } BT[] = {
            { T_IQ3_S, "iq3_s", {0, -1} }, { T_Q4_K, "q4_K", {0, 2} },
            { T_Q6_K, "q6_K", {208, -1} }, { T_Q8_0, "q8_0", {0, -1} },
        };
        enum { BN_IN = 5120, BN_OUT = 4096, BB = 512 };
        float *bx = malloc(sizeof(float) * (size_t)BB * BN_IN);
        float *by = malloc(sizeof(float) * (size_t)BB * BN_OUT);
        for (size_t i = 0; i < (size_t)BB * BN_IN; i++) bx[i] = frnd();
        for (size_t k = 0; k < sizeof(BT) / sizeof(*BT); k++) {
            gguf_tensor w;
            memset(&w, 0, sizeof(w));
            w.type = (uint32_t)BT[k].type; w.scale = 1.0f;
            w.ne[0] = BN_IN; w.ne[1] = BN_OUT;
            w.data = make_rows(BT[k].type, BT[k].f16, BN_IN, BN_OUT, &w.nbytes);
            for (int nb = 64; nb <= BB; nb *= 8)
                for (int staged = 1; staged >= 0; staged--) {
                    if (!staged) setenv("RUNNER_TRAIN_MMS", "0", 1);
                    double best = 1e9;
                    for (int rep = 0; rep < 5; rep++) {
                        double t_mm, t_mvt, k_mm, k_mvt;
                        long n_mm, n_mvt;
                        gpu_train_prof(&m, &t_mm, &n_mm, &t_mvt, &n_mvt, &k_mm, &k_mvt);
                        if (!gpu_train_mm(&m, &w, bx, BN_IN, by, BN_OUT, BN_IN,
                                          BN_OUT, NULL, nb)) break;
                        gpu_train_prof(&m, &t_mm, &n_mm, &t_mvt, &n_mvt, &k_mm, &k_mvt);
                        if (k_mm < best) best = k_mm;
                    }
                    unsetenv("RUNNER_TRAIN_MMS");
                    printf("bench %-6s %4d columns %-8s %7.3f ms  %6.2f TFLOP/s\n",
                           BT[k].name, nb, staged ? "staged" : "per-warp", best * 1e3,
                           2.0 * BN_IN * BN_OUT * nb / best / 1e12);
                }
            bkeep[k] = w.data;
        }
        free(bx); free(by);
    }
    gpu_train_free(&m);
    for (size_t k = 0; k < sizeof(keep) / sizeof(*keep); k++) free(keep[k]);
    for (size_t k = 0; k < sizeof(bkeep) / sizeof(*bkeep); k++) free(bkeep[k]);
    printf("%s: %d types\n", g_fail ? "FAIL" : "PASS", tested);
    return g_fail;
}
