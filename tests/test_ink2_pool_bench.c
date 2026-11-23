// R4.48.20: TEST-ONLY microbenchmark comparing ink2_par (src/inkling2.c) against
// the house tpool (src/quants.c) on empty barriers and a 2304x2304 BF16 product
// across 1, 4, 8, and 16 threads. Interleaved rounds take each path's best time
// so thermal drift hits both equally. No bits or defaults in the engine change.

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling2.h"
#include "quants.h"
#include "tpool.h"

typedef void (*ink2_range_fn)(void *ctx, int64_t i0, int64_t i1);
void ink2_par(int64_t n, int64_t work, ink2_range_fn fn, void *ctx);

static void noop_ink2(void *ctx, int64_t i0, int64_t i1) {
    (void)ctx; (void)i0; (void)i1;
}

static void noop_tp(void *ctx, int i0, int i1) {
    (void)ctx; (void)i0; (void)i1;
}

typedef struct {
    const uint16_t *w;
    const float *x;
    float *y;
    int n_in;
} bf16_mv_ctx;

static void mv_ink2(void *c, int64_t r0, int64_t r1) {
    const bf16_mv_ctx *ctx = (const bf16_mv_ctx *)c;
    const int n = ctx->n_in;
    for (int64_t r = r0; r < r1; r++)
        ctx->y[r] = vec_dot(T_BF16, ctx->w + (size_t)r * (size_t)n, ctx->x, n);
}

static void mv_tp(void *c, int i0, int i1) {
    const bf16_mv_ctx *ctx = (const bf16_mv_ctx *)c;
    const int n = ctx->n_in;
    for (int r = i0; r < i1; r++)
        ctx->y[r] = vec_dot(T_BF16, ctx->w + (size_t)r * (size_t)n, ctx->x, n);
}

static uint16_t f32_bits_bf16(float v) {
    uint32_t u = 0;
    memcpy(&u, &v, sizeof(u));
    return (uint16_t)(u >> 16);
}

static void bench_pair_us(tpool *tp, int64_t n_ink2, int64_t work, ink2_range_fn fn_ink2,
                          void *ctx_ink2, int n_tp, tp_fn fn_tp, void *ctx_tp,
                          int rounds, int iters, double *best_ink2, double *best_tp) {
    double b0 = 1e30, b1 = 1e30;
    for (int r = 0; r < rounds; r++) {
        double t0, dt;
        for (int pass = 0; pass < 2; pass++) {
            int run_ink2 = ((r & 1) == 0) ? (pass == 0) : (pass == 1);
            t0 = tprof_now();
            if (run_ink2) {
                for (int i = 0; i < iters; i++) ink2_par(n_ink2, work, fn_ink2, ctx_ink2);
                dt = (tprof_now() - t0) * 1e6 / (double)iters;
                if (dt < b0) b0 = dt;
            } else {
                for (int i = 0; i < iters; i++) tpool_run(tp, fn_tp, ctx_tp, n_tp);
                dt = (tprof_now() - t0) * 1e6 / (double)iters;
                if (dt < b1) b1 = dt;
            }
        }
    }
    *best_ink2 = b0;
    *best_tp = b1;
}

static int bench_empty(const int *nts, int n_nts, int rounds, int iters) {
    printf("EMPTY_BARRIER n=256 work=16384 (rounds=%d iters=%d best us/call):\n", rounds, iters);
    for (int ti = 0; ti < n_nts; ti++) {
        int nt = nts[ti];
        int got = ink2_set_threads(nt);
        if (got != nt) {
            fprintf(stderr, "FAIL: ink2_set_threads(%d) returned %d\n", nt, got);
            return 1;
        }
        tpool *tp = tpool_create(nt);
        if (!tp) { fprintf(stderr, "FAIL: tpool_create(%d) failed\n", nt); return 1; }

        long split0 = 0, split1 = 0;
        int last_pool = 0;
        ink2_par_stats(&split0, &last_pool);
        for (int i = 0; i < 64; i++) {
            ink2_par(256, 16384, noop_ink2, NULL);
            tpool_run(tp, noop_tp, NULL, 256);
        }

        double best_ink2 = 0.0, best_tp = 0.0;
        bench_pair_us(tp, 256, 16384, noop_ink2, NULL, 256, noop_tp, NULL,
                      rounds, iters, &best_ink2, &best_tp);
        ink2_par_stats(&split1, &last_pool);
        long expected_splits = (nt > 1) ? (long)(64 + rounds * iters) : 0L;
        if (split1 - split0 != expected_splits || (nt > 1 && last_pool != nt)) {
            fprintf(stderr, "FAIL: nt=%d splits=%ld (want %ld) last_pool=%d\n",
                    nt, split1 - split0, expected_splits, last_pool);
            tpool_destroy(tp);
            return 1;
        }
        printf("  nt=%2d  ink2_par=%7.2f us  tpool=%7.2f us  ratio(ink2/tpool)=%6.2fx\n",
               nt, best_ink2, best_tp, best_tp > 0.0 ? best_ink2 / best_tp : 0.0);
        tpool_destroy(tp);
    }
    return 0;
}

static int bench_bf16_matvec(const int *nts, int n_nts, int rounds, int iters) {
    const int rows = 2304;
    const int n_in = 2304;
    uint16_t *w = (uint16_t *)malloc((size_t)rows * (size_t)n_in * sizeof(uint16_t));
    float *x = (float *)malloc((size_t)n_in * sizeof(float));
    float *y_ref = (float *)malloc((size_t)rows * sizeof(float));
    float *y_ink2 = (float *)malloc((size_t)rows * sizeof(float));
    float *y_tp = (float *)malloc((size_t)rows * sizeof(float));
    if (!w || !x || !y_ref || !y_ink2 || !y_tp) {
        fprintf(stderr, "FAIL: out of memory allocating 2304x2304 BF16 bench buffers\n");
        free(w); free(x); free(y_ref); free(y_ink2); free(y_tp);
        return 1;
    }

    uint32_t s = 0x13579bdfu;
    for (int j = 0; j < n_in; j++) {
        s = s * 1664525u + 1013904223u;
        x[j] = ((float)(int32_t)(s >> 16) / 32768.0f) - 1.0f;
    }
    for (size_t k = 0; k < (size_t)rows * (size_t)n_in; k++) {
        s = s * 1664525u + 1013904223u;
        float v = (((float)(int32_t)(s >> 16) / 32768.0f) - 1.0f) * 0.05f;
        w[k] = f32_bits_bf16(v);
    }

    bf16_mv_ctx ctx_ref = { w, x, y_ref, n_in };
    mv_ink2(&ctx_ref, 0, rows);

    printf("BF16_MATVEC rows=%d n_in=%d (rounds=%d iters=%d best us/call):\n", rows, n_in, rounds, iters);
    for (int ti = 0; ti < n_nts; ti++) {
        int nt = nts[ti];
        int got = ink2_set_threads(nt);
        if (got != nt) {
            fprintf(stderr, "FAIL: ink2_set_threads(%d) returned %d\n", nt, got);
            free(w); free(x); free(y_ref); free(y_ink2); free(y_tp);
            return 1;
        }
        tpool *tp = tpool_create(nt);
        if (!tp) {
            fprintf(stderr, "FAIL: tpool_create(%d) failed\n", nt);
            free(w); free(x); free(y_ref); free(y_ink2); free(y_tp);
            return 1;
        }

        bf16_mv_ctx ctx_ink2 = { w, x, y_ink2, n_in };
        bf16_mv_ctx ctx_tp = { w, x, y_tp, n_in };

        for (int i = 0; i < 2; i++) {
            ink2_par(rows, n_in, mv_ink2, &ctx_ink2);
            tpool_run(tp, mv_tp, &ctx_tp, rows);
        }
        if (memcmp(y_ink2, y_ref, (size_t)rows * sizeof(float)) != 0 ||
            memcmp(y_tp, y_ref, (size_t)rows * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: bit mismatch vs 1T reference at nt=%d\n", nt);
            tpool_destroy(tp);
            free(w); free(x); free(y_ref); free(y_ink2); free(y_tp);
            return 1;
        }

        double best_ink2 = 0.0, best_tp = 0.0;
        bench_pair_us(tp, rows, n_in, mv_ink2, &ctx_ink2, rows, mv_tp, &ctx_tp,
                      rounds, iters, &best_ink2, &best_tp);
        printf("  nt=%2d  ink2_par=%7.2f us  tpool=%7.2f us  delta=%+7.2f us  ratio(ink2/tpool)=%6.2fx  (0 diff)\n",
               nt, best_ink2, best_tp, best_ink2 - best_tp, best_tp > 0.0 ? best_ink2 / best_tp : 0.0);
        tpool_destroy(tp);
    }

    free(w);
    free(x);
    free(y_ref);
    free(y_ink2);
    free(y_tp);
    return 0;
}

int main(int argc, char **argv) {
    const int nts[4] = { 1, 4, 8, 16 };
    int rounds = argc > 1 ? atoi(argv[1]) : 25;
    int empty_iters = argc > 2 ? atoi(argv[2]) : 1000;
    int matvec_iters = argc > 3 ? atoi(argv[3]) : 20;
    if (rounds < 1) rounds = 1;
    if (empty_iters < 1) empty_iters = 1;
    if (matvec_iters < 1) matvec_iters = 1;

    if (bench_empty(nts, 4, rounds, empty_iters) != 0) return 1;
    if (bench_bf16_matvec(nts, 4, rounds, matvec_iters) != 0) return 1;
    printf("PASS: test-ink2-pool-bench\n");
    return 0;
}
