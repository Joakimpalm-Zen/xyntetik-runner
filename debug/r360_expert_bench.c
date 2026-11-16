// EXPERIMENT R3.60 / R3.73 (not for merge): one weight product through the
// CPU routes a batched prefill can take, timed per width.
//
// R3.60's third variant moves the routed experts' products from the f32 tile
// to the per-column native dot at every width. Its kill line K2 allows those
// products, at -b 512, up to 1.37x the tile's time (the lab profile, shade
// 342aff4b). This bench reads that ratio on one product of gpt-oss-120b's
// expert shape, 2,880 x 2,880 MXFP4, at widths 1 to 128.
// R3.73's premise is the decode's share of a batched product for a type whose
// x86 decode is C (Q5_K), against a type with a SIMD decode (Q4_K, the
// control: its share should be the smaller one).
//
// Routes, each one pool run per product unless said:
//   tile    mv_rows' f32 tile as cb82532b takes it in a prefill at any width,
//           copied from src/model.c mv_rows at cb82532b: groups of
//           VEC_DOT_F32_TILE_ROWS weight rows dequantized into a buffer, then
//           vec_dot_f32_tile over TCOLS-column chunks; the tail rows one at a
//           time (dequant_row, vec_dot_f32_multi)
//   passes  the third variant: per worker, column passes of
//           VEC_DOT_MULTI_MAX outer and the worker's rows inner, vec_dot_multi
//           (vec_dot for a one-column pass)
//   split   the verify's exact mode (t_mv_exact): one pool run per
//           VEC_DOT_MULTI_MAX columns, mv_rows' native branch, rows outer
//   decode  the tile's dequantize alone, without the dot (R3.73's share)
//
// The weights are spread over enough copies (512 MB by default) that a
// product reads them from memory, as an expert of a 59 GB file does, and not
// from a cache warmed by the previous repeat.
//
// Identity first: passes and split must equal the single-column vec_dot of
// every column byte for byte at every width. The live control: the same
// comparison must fail when one activation moves by a quarter (a one-ulp
// move of one term in 2,880 rounds away in the sum). Whether the tile
// equals vec_dot is printed, not judged (it sums in another order, so it
// should not).
// Timing: interleaved rounds in a rotating arm order, each a calibrated
// number of products, best of the rounds per arm and width.
// Exit: 0 done, 1 an identity failure, 2 a control not live, 3 no memory.
//
//   cc -O3 -ffast-math -std=gnu11 -march=native -I src
//      debug/r360_expert_bench.c src/quants.c -o r360-bench -lm -lpthread
//   (one command line)
//   ./r360-bench [threads (0 = every CPU)] [rounds] [MB of weight copies]
#include "quants.h"
#include "fp16.h"
#include "tpool.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && \
    !(defined(__AVX2__) && defined(__FMA__) && defined(__F16C__))
enum { TCOLS = 32 };   // as mv_rows
#else
enum { TCOLS = 16 };
#endif
enum { TROWS = VEC_DOT_F32_TILE_ROWS, MAXW = 128 };

typedef struct {
    int type, n_in, n_out, n;   // n: the product's width (columns)
    const uint8_t *w;           // this product's weight copy
    size_t rsz;
    const float *x;             // columns of n_in floats, stride n_in
    float *y;                   // columns of n_out floats, stride n_out
    int c0, nc;                 // split: this run's columns
} job;

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void oom(void) { fprintf(stderr, "r360-bench: out of memory\n"); exit(3); }

// src/model.c mv_rows at cb82532b, the f32 tile and its tail (no bias, no
// scale companion: neither occurs for these types)
static void tile_rows(void *ctx, int i0, int i1) {
    job *j = ctx;
    int n_in = j->n_in;
    float *tbuf = NULL;
    const float *trow[TROWS];
    if (i1 - i0 >= 4) tbuf = malloc(sizeof(float) * (size_t)n_in * TROWS);
    if (tbuf) {
        int r = i0;
        for (int g = TROWS; g >= 4; g /= 2)
            for (; r + g <= i1; r += g) {
                for (int k = 0; k < g; k++) {
                    float *dst = tbuf + (size_t)k * n_in;
                    dequant_row(j->type, j->w + (size_t)(r + k) * j->rsz, dst, n_in);
                    trow[k] = dst;
                }
                for (int c = 0; c < j->n; c += TCOLS) {
                    int nb = j->n - c < TCOLS ? j->n - c : TCOLS;
                    float tout[TROWS][TCOLS];
                    vec_dot_f32_tile(trow, g, j->x + (size_t)c * n_in, n_in, nb, n_in,
                                     &tout[0][0], TCOLS);
                    for (int k = 0; k < g; k++)
                        for (int b = 0; b < nb; b++)
                            j->y[(size_t)(c + b) * j->n_out + r + k] = tout[k][b];
                }
            }
        i0 = r;
        free(tbuf);
        if (i0 >= i1) return;
    }
    float *buf = malloc(sizeof(float) * (size_t)n_in);
    if (!buf) oom();
    float outs[64];
    for (int r = i0; r < i1; r++) {
        dequant_row(j->type, j->w + (size_t)r * j->rsz, buf, n_in);
        for (int c = 0; c < j->n; c += 64) {
            int nb = j->n - c < 64 ? j->n - c : 64;
            vec_dot_f32_multi(buf, j->x + (size_t)c * n_in, n_in, nb, n_in, outs);
            for (int b = 0; b < nb; b++) j->y[(size_t)(c + b) * j->n_out + r] = outs[b];
        }
    }
    free(buf);
}

// mv_rows' native branch over columns [c0, c0 + nc), nc <= VEC_DOT_MULTI_MAX,
// rows outer
static void native_chunk(const job *j, int c0, int nc, int i0, int i1) {
    const float *xs[VEC_DOT_MULTI_MAX];
    float v[VEC_DOT_MULTI_MAX];
    for (int c = 0; c < nc; c++) xs[c] = j->x + (size_t)(c0 + c) * j->n_in;
    for (int r = i0; r < i1; r++) {
        const void *row = j->w + (size_t)r * j->rsz;
        if (nc == 1) v[0] = vec_dot(j->type, row, xs[0], j->n_in);
        else vec_dot_multi(j->type, row, xs, nc, j->n_in, v);
        for (int c = 0; c < nc; c++) j->y[(size_t)(c0 + c) * j->n_out + r] = v[c];
    }
}

static void passes_rows(void *ctx, int i0, int i1) {
    job *j = ctx;
    for (int c0 = 0; c0 < j->n; c0 += VEC_DOT_MULTI_MAX)
        native_chunk(j, c0, j->n - c0 < VEC_DOT_MULTI_MAX ? j->n - c0 : VEC_DOT_MULTI_MAX,
                     i0, i1);
}

static void split_rows(void *ctx, int i0, int i1) {
    job *j = ctx;
    native_chunk(j, j->c0, j->nc, i0, i1);
}

static void decode_rows(void *ctx, int i0, int i1) {
    job *j = ctx;
    int n_in = j->n_in;
    float *tbuf = malloc(sizeof(float) * (size_t)n_in * TROWS);
    if (!tbuf) oom();
    float s = 0;
    int r = i0;
    for (int g = TROWS; g >= 4; g /= 2)
        for (; r + g <= i1; r += g) {
            for (int k = 0; k < g; k++)
                dequant_row(j->type, j->w + (size_t)(r + k) * j->rsz,
                            tbuf + (size_t)k * n_in, n_in);
            s += tbuf[0] + tbuf[(size_t)g * n_in - 1];
        }
    for (; r < i1; r++) {
        dequant_row(j->type, j->w + (size_t)r * j->rsz, tbuf, n_in);
        s += tbuf[n_in - 1];
    }
    free(tbuf);
    j->y[i0] = s;   // each worker its own slot: the work stays observable
}

enum { A_TILE, A_PASSES, A_SPLIT, A_DECODE, N_ARMS };

static void product(tpool *tp, int arm, job *j) {
    switch (arm) {
    case A_TILE:   tpool_run(tp, tile_rows, j, j->n_out); break;
    case A_PASSES: tpool_run(tp, passes_rows, j, j->n_out); break;
    case A_DECODE: tpool_run(tp, decode_rows, j, j->n_out); break;
    case A_SPLIT:
        for (int c0 = 0; c0 < j->n; c0 += VEC_DOT_MULTI_MAX) {
            j->c0 = c0;
            j->nc = j->n - c0 < VEC_DOT_MULTI_MAX ? j->n - c0 : VEC_DOT_MULTI_MAX;
            tpool_run(tp, split_rows, j, j->n_out);
        }
        break;
    }
}

static uint32_t g_rnd = 12345u;
static uint32_t rnd(void) { g_rnd = g_rnd * 1103515245u + 12345u; return g_rnd >> 8; }

// random blocks with sane scales: a random scale byte can be Inf, NaN or a
// subnormal, which would time the FPU's slow path, not the route
static void fill_weights(int type, uint8_t *w, size_t bytes) {
    for (size_t i = 0; i < bytes; i++) w[i] = (uint8_t)rnd();
    size_t bs = ggml_type_size(type);
    for (size_t b = 0; b + bs <= bytes; b += bs) {
        if (type == T_MXFP4) {
            w[b] = (uint8_t)(120 + rnd() % 15);              // E8M0: 2^-7 .. 2^7
        } else {                                              // Q4_K, Q5_K: d, dmin
            w[b] = 0x00; w[b + 1] = 0x2c; w[b + 2] = 0x00; w[b + 3] = 0x2c;   // 1/16
        }
    }
}

typedef struct { int type; const char *name; int n_in, n_out; const char *what; } shape;

static int g_status = 0;

static void run_shape(tpool *tp, const shape *s, int rounds, size_t copy_mb) {
    size_t rsz = ggml_row_size(s->type, s->n_in), mat = rsz * (size_t)s->n_out;
    int ncopy = (int)((copy_mb << 20) / mat);
    if (ncopy < 1) ncopy = 1;
    if (ncopy > 256) ncopy = 256;
    uint8_t *w = malloc(mat * (size_t)ncopy);
    float *x = malloc(sizeof(float) * (size_t)s->n_in * MAXW);
    float *y = malloc(sizeof(float) * (size_t)s->n_out * MAXW);
    float *ref = malloc(sizeof(float) * (size_t)s->n_out * MAXW);
    if (!w || !x || !y || !ref) oom();
    fill_weights(s->type, w, mat);
    for (int c = 1; c < ncopy; c++) memcpy(w + mat * (size_t)c, w, mat);
    for (size_t i = 0; i < (size_t)s->n_in * MAXW; i++) x[i] = (rnd() % 2048) / 2048.0f - 0.5f;
    printf("== %s %d x %d (%s): %d weight copies of %.1f MB\n", s->name, s->n_out, s->n_in,
           s->what, ncopy, mat / 1048576.0);

    // the reference: every column's single-column vec_dot, one thread
    for (int c = 0; c < MAXW; c++)
        for (int r = 0; r < s->n_out; r++)
            ref[(size_t)c * s->n_out + r] =
                vec_dot(s->type, w + (size_t)r * rsz, x + (size_t)c * s->n_in, s->n_in);

    static const int widths[] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    enum { NW = sizeof(widths) / sizeof(widths[0]) };
    int bad_passes = 0, bad_split = 0, tile_differs = 0;
    for (int k = 0; k < NW; k++) {
        job j = { s->type, s->n_in, s->n_out, widths[k], w, rsz, x, y, 0, 0 };
        size_t nb = sizeof(float) * (size_t)s->n_out * widths[k];
        // each arm starts from a poisoned output, so an arm that writes
        // nothing cannot pass for the one before it
        memset(y, 0xff, nb);
        product(tp, A_PASSES, &j);
        if (memcmp(y, ref, nb)) bad_passes++;
        memset(y, 0xff, nb);
        product(tp, A_SPLIT, &j);
        if (memcmp(y, ref, nb)) bad_split++;
        memset(y, 0xff, nb);
        product(tp, A_TILE, &j);
        if (memcmp(y, ref, nb)) tile_differs++;
    }
    // the live control: one activation moved by a quarter, and the comparison must fail
    int seen;
    {
        job j = { s->type, s->n_in, s->n_out, 8, w, rsz, x, y, 0, 0 };
        float keep = x[0];
        x[0] = keep + 0.25f;
        memset(y, 0xff, sizeof(float) * (size_t)s->n_out * 8);
        product(tp, A_PASSES, &j);
        seen = memcmp(y, ref, sizeof(float) * (size_t)s->n_out * 8) != 0;
        x[0] = keep;
    }
    printf("identity: passes = vec_dot at all %d widths: %s; split = vec_dot: %s; "
           "a changed activation is seen: %s (the live control)\n",
           NW, bad_passes ? "NO" : "yes", bad_split ? "NO" : "yes", seen ? "yes" : "NO");
    printf("information: the tile differs from vec_dot at %d of %d widths\n", tile_differs, NW);
    if (bad_passes || bad_split) g_status = 1;
    else if (!seen && g_status == 0) g_status = 2;

    // the decode alone: one width-free number
    int it = 0;
    double dbest = 1e9;
    {
        job j = { s->type, s->n_in, s->n_out, 1, w, rsz, x, y, 0, 0 };
        double t0 = now(); product(tp, A_DECODE, &j); double t1 = now();
        int reps = (int)(0.03 / (t1 - t0 + 1e-9)); if (reps < 1) reps = 1; if (reps > 2000) reps = 2000;
        for (int r = 0; r < rounds; r++) {
            t0 = now();
            for (int q = 0; q < reps; q++) { j.w = w + mat * (size_t)(it++ % ncopy); product(tp, A_DECODE, &j); }
            t1 = now();
            if ((t1 - t0) / reps < dbest) dbest = (t1 - t0) / reps;
        }
    }
    printf("decode alone (the tile's dequantize, no dot): %.3f ms a product\n", dbest * 1e3);
    printf("width  reps   tile_ms passes_ms  split_ms  passes/tile  split/tile  decode/tile\n");
    double worst_k2 = 0;
    for (int k = 0; k < NW; k++) {
        job j = { s->type, s->n_in, s->n_out, widths[k], w, rsz, x, y, 0, 0 };
        double t0 = now(); product(tp, A_TILE, &j); double t1 = now();
        int reps = (int)(0.03 / (t1 - t0 + 1e-9)); if (reps < 1) reps = 1; if (reps > 2000) reps = 2000;
        double best[3] = { 1e9, 1e9, 1e9 };
        for (int r = 0; r < rounds; r++)
            for (int a = 0; a < 3; a++) {
                int arm = (a + r) % 3;   // A_TILE, A_PASSES, A_SPLIT in a rotating order
                t0 = now();
                for (int q = 0; q < reps; q++) { j.w = w + mat * (size_t)(it++ % ncopy); product(tp, arm, &j); }
                t1 = now();
                if ((t1 - t0) / reps < best[arm]) best[arm] = (t1 - t0) / reps;
            }
        double rp = best[A_PASSES] / best[A_TILE], rs = best[A_SPLIT] / best[A_TILE];
        if (widths[k] >= 16 && widths[k] <= 64 && rp > worst_k2) worst_k2 = rp;
        printf("%5d %5d %9.3f %9.3f %9.3f %12.2f %11.2f %12.2f\n", widths[k], reps,
               best[A_TILE] * 1e3, best[A_PASSES] * 1e3, best[A_SPLIT] * 1e3, rp, rs,
               dbest / best[A_TILE]);
    }
    if (s->type == T_MXFP4)
        printf("R3.60 K2 first look: passes over tile at 16 to 64 columns, worst %.2f "
               "(the margin K2 leaves: 1.37): %s\n", worst_k2,
               worst_k2 <= 1.37 ? "inside" : "OVER");
    free(w); free(x); free(y); free(ref);
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 0;
    int rounds = argc > 2 ? atoi(argv[2]) : 7;
    long mb = argc > 3 ? atol(argv[3]) : 512;
    if (threads <= 0) threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (rounds < 1) rounds = 1;
    if (mb < 1) mb = 1;
    f16_init();   // the fp16 and E8M0 tables: without them every weight decodes to 0
    printf("r360-bench: avx2 %d avx512f %d avx512vl %d neon %d | tile rows %d, tile cols %d, "
           "multi max %d | threads %d | rounds %d | weight copies %ld MB\n",
#if defined(__AVX2__)
           1,
#else
           0,
#endif
#if defined(__AVX512F__)
           1,
#else
           0,
#endif
#if defined(__AVX512VL__)
           1,
#else
           0,
#endif
#if defined(__ARM_NEON)
           1,
#else
           0,
#endif
           TROWS, TCOLS, VEC_DOT_MULTI_MAX, threads, rounds, mb);
    tpool *tp = tpool_create(threads);
    if (!tp) oom();
    static const shape shapes[] = {
        { T_MXFP4, "MXFP4", 2880, 2880, "R3.60: a gpt-oss-120b expert" },
        { T_Q5_K,  "Q5_K",  4096, 4096, "R3.73: a C decode on x86" },
        { T_Q4_K,  "Q4_K",  4096, 4096, "R3.73's control: a SIMD decode on x86" },
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        run_shape(tp, &shapes[i], rounds, (size_t)mb);
        fflush(stdout);
    }
    tpool_destroy(tp);
    printf("exit %d (0 done, 1 identity failure, 2 a control not live)\n", g_status);
    return g_status;
}
