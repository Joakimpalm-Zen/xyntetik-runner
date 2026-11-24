// The CPU dot route gate (2026-10-08). Byte-identity tests cannot see a
// route that is correct and slow: 1.1.1 sent the one-row decode through
// vec_dot_multi, every output stayed byte-equal, and dense IQ3_S decode fell
// from 1.45 to 0.83 tok/s until a hotfix. Two checks:
//  1. dispatch: a one-token forward never calls vec_dot_multi; a 3-row batch
//     does (the small-batch route exists and is used);
//  2. a same-process ratio canary: for 2..7 columns vec_dot_multi must not
//     cost more than 1.5x the per-column vec_dot loop it replaces. A ratio
//     on one machine in one process, not an absolute tok/s. Best of 5
//     interleaved rounds removes a single spike but not a stretch of
//     contention on a shared hosted runner (two macos-suite failures, each
//     at ratio ~2 and ~1.0 on re-run), so a case over the limit is measured
//     again with 25 rounds and fails only when that confirms it: a slow
//     route is slow every time, a busy neighbour is not.
//  3. the same kind of canary for a weight decode (R3.73): on an x86 AVX2
//     build, dequant_row's MXFP4 rows (every expert row a batched prefill
//     of gpt-oss decodes) must be at least 1.3x as fast as the C decode they
//     replaced, quants.c's dequant_row_mxfp4_ref, on an L2-sized set in the
//     same process. A fallback to C reads 1.0 by construction. Measured on
//     hosted runners, 2026-11-16, at -march=native: EPYC 7763 (AVX2 only,
//     Linux and Windows) 1.73x to 1.77x, the narrowest; Xeon 6975P-C 2.26x;
//     EPYC 9V74 4.96x (2.92x at x86-64-v3). The first line, at most 1.3x
//     Q4_K's per weight, failed on the EPYC 7763 (1.51x to 1.62x) with the
//     SIMD decode in place: Q4_K's decode is another instruction mix (byte
//     unpacks where MXFP4 takes two permutes and a blend), so that ratio
//     measured the core's shuffle throughput, not whether the decode is
//     SIMD. Its bytes are checked too, over all 256 exponents: this binary
//     is built with the runner's own flags (-ffast-math, so flush-to-zero
//     and denormals-are-zero are on), which test-quants-simd's are not.
#include "model.h"
#include "quants.h"
#include "fp16.h"
#include "runner.h"
#include "tpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

static int g_fail = 0;
static void ck(int cond, const char *what) {
    fprintf(stderr, "%s: %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) g_fail = 1;
}

static double route_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// the process's page faults so far (Windows: every fault; POSIX: the minor ones, a demand-zero page's kind), or -1
static long route_faults(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pc;
    pc.cb = sizeof pc;
    return GetProcessMemoryInfo(GetCurrentProcess(), &pc, sizeof pc) ? (long)pc.PageFaultCount : -1;
#else
    struct rusage ru;
    return getrusage(RUSAGE_SELF, &ru) == 0 ? (long)ru.ru_minflt : -1;
#endif
}

static void dispatch(const char *path) {
    model_t m; memset(&m, 0, sizeof m);
    model_params p; memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF; p.n_threads = 1; p.n_ctx = 64; p.n_batch = 8;
    if (!model_load(&m, path, &p)) { fprintf(stderr, "cannot load %s\n", path); g_fail = 1; return; }
    unsigned long c0 = vec_dot_multi_calls();
    model_forward(&m, 1, 0);
    unsigned long c1 = vec_dot_multi_calls();
    ck(c1 == c0, "a one-row forward never takes the multi-column dot");
    int32_t toks[3] = { 2, 3, 4 };
    model_forward_batch(&m, toks, 3, 1, true);
    ck(vec_dot_multi_calls() > c1, "a 3-row batch takes the multi-column dot");
    model_free(&m);
}

static void ratio(int type, const char *name) {
    const int n = 4096, rows = 256;
    size_t rs = ggml_row_size(type, n);
    unsigned char *w = malloc(rs * rows);
    float *x = malloc(sizeof(float) * n * VEC_DOT_MULTI_MAX);
    unsigned s = 12345;
    for (size_t i = 0; i < rs * rows; i++) { s = s * 1103515245u + 12345u; w[i] = (unsigned char)(s >> 16); }
    // keep fp16 block scales finite and small: every format here leads its
    // block with an fp16 scale, and a random one can be Inf/NaN
    size_t bs = ggml_row_size(type, ggml_block_size(type));
    if (type == T_NVFP4) {
        for (size_t b = 0; b < rs * rows; b += bs) memset(w + b, 0x38, 4);   // finite unit UE4M3 scales (1.0)
    } else {
        for (size_t b = 0; b < rs * rows; b += bs) { w[b] = 0x00; w[b + 1] = 0x3c; w[b + 2] = 0x00; w[b + 3] = 0x3c; }
    }
    for (int i = 0; i < n * VEC_DOT_MULTI_MAX; i++) { s = s * 1103515245u + 12345u; x[i] = ((s >> 9) & 1023) / 1024.0f - 0.5f; }
    float out[VEC_DOT_MULTI_MAX], sink = 0;
    for (int nc = 2; nc < VEC_DOT_MULTI_MAX; nc++) {
        const float *xs[VEC_DOT_MULTI_MAX];
        for (int c = 0; c < nc; c++) xs[c] = x + (size_t)c * n;
        double best_m, best_l;
        for (int attempt = 0, rounds = 5; attempt < 2; attempt++, rounds = 25) {
            best_m = 1e9; best_l = 1e9;
            for (int rep = 0; rep < rounds; rep++) {
                double t0 = route_now();
                for (int r = 0; r < rows; r++) { vec_dot_multi(type, w + rs * r, xs, nc, n, out); sink += out[0]; }
                double t1 = route_now();
                for (int r = 0; r < rows; r++) for (int c = 0; c < nc; c++) sink += vec_dot(type, w + rs * r, xs[c], n);
                double t2 = route_now();
                if (t1 - t0 < best_m) best_m = t1 - t0;
                if (t2 - t1 < best_l) best_l = t2 - t1;
            }
            if (best_m <= 1.5 * best_l) break;
            if (attempt == 0)
                fprintf(stderr, "re-measuring: %s, %d columns: ratio %.2f in 5 rounds\n",
                        name, nc, best_m / best_l);
        }
        char msg[160];
        snprintf(msg, sizeof msg, "%s, %d columns: multi %.3f ms, per-column loop %.3f ms (ratio %.2f, limit 1.50)",
                 name, nc, best_m * 1e3, best_l * 1e3, best_m / best_l);
        ck(best_m <= 1.5 * best_l, msg);
    }
    if (sink == 12345.678f) puts("");   // keep the work observable
    free(w); free(x);
}

// Tier C wide-batch tile ratio canary against VEC_DOT_MULTI_MAX=8 chunked multi:
// guards regressions at <= 1.50x (no hard < 1.0 assertion) and reports measured ratio.
static void tile_ratio(int type, const char *name) {
    const int n = 2880, rows = 64, max_nb = 64;
    size_t rs = ggml_row_size(type, n);
    unsigned char *w = malloc(rs * rows);
    float *x = malloc(sizeof(float) * (size_t)n * max_nb);
    float *y = malloc(sizeof(float) * (size_t)rows * max_nb);
    unsigned s = 67890;
    for (size_t i = 0; i < rs * rows; i++) { s = s * 1103515245u + 12345u; w[i] = (unsigned char)(s >> 16); }
    if (type == T_NVFP4) {
        size_t bs = ggml_row_size(type, ggml_block_size(type));
        for (size_t b = 0; b < rs * rows; b += bs) memset(w + b, 0x38, 4);
    } else if (type == T_BF16) {
        uint16_t *h = (uint16_t *)w;
        for (size_t i = 0; i < (size_t)n * rows; i++) h[i] = (uint16_t)((h[i] & 0x807fu) | 0x3f00u);
    }
    for (int i = 0; i < n * max_nb; i++) { s = s * 1103515245u + 12345u; x[i] = ((s >> 9) & 1023) / 1024.0f - 0.5f; }
    static const int nbs[] = { 16, 32, 64 };
    float sink = 0;
    for (size_t bi = 0; bi < sizeof(nbs) / sizeof(*nbs); bi++) {
        const int nb = nbs[bi];
        double best_t = 1e9, best_m = 1e9;
        long flt_t = 0, flt_m = 0, calls = 0;   // page faults in the tile's calls and in the multi-8 loops
        for (int attempt = 0, rounds = 5; attempt < 2; attempt++, rounds = 25) {
            best_t = 1e9; best_m = 1e9;
            for (int rep = 0; rep < rounds; rep++) {
                const long f0 = route_faults();
                double t0 = route_now();
                vec_dot_exact_tile(type, w, rs, rows, x, n, nb, n, 1.0f, y, rows);
                sink += y[0];
                double t1 = route_now();
                const long f1 = route_faults();
                for (int r = 0; r < rows; r++) {
                    const void *row = w + rs * r;
                    for (int c0 = 0; c0 < nb; c0 += VEC_DOT_MULTI_MAX) {
                        const int nc = nb - c0 < VEC_DOT_MULTI_MAX ? nb - c0 : VEC_DOT_MULTI_MAX;
                        const float *xs[VEC_DOT_MULTI_MAX];
                        float out[VEC_DOT_MULTI_MAX];
                        for (int k = 0; k < nc; k++) xs[k] = x + (size_t)(c0 + k) * n;
                        vec_dot_multi(type, row, xs, nc, n, out);
                        sink += out[0];
                    }
                }
                double t2 = route_now();
                const long f2 = route_faults();
                if (t1 - t0 < best_t) best_t = t1 - t0;
                if (t2 - t1 < best_m) best_m = t2 - t1;
                if (f0 >= 0 && f1 >= 0 && f2 >= 0) { flt_t += f1 - f0; flt_m += f2 - f1; calls++; }
            }
            if (best_t <= 1.5 * best_m) break;
        }
        // R4.48.20.4: the page faults a call, so a slow tile on a host says whether its time went to faults (a
        // per-call buffer's demand-zero pages) or to its arithmetic
        char msg[256], flt[96] = "";
        if (calls) snprintf(flt, sizeof flt, "; page faults a call: tile %.1f, multi-8 loop %.1f",
                            (double)flt_t / calls, (double)flt_m / calls);
        snprintf(msg, sizeof msg,
                 "%s exact tile, %d cols: tile %.3f ms, multi-8 loop %.3f ms (ratio %.2f, limit 1.50)%s",
                 name, nb, best_t * 1e3, best_m * 1e3, best_t / best_m, flt);
        ck(best_t <= 1.5 * best_m, msg);
    }
    if (sink == 12345.678f) puts("");
    free(y); free(x); free(w);
}

// R4.48.20.4: the tile's scratch is a thread's own, kept across calls. It was a malloc and a free of the rows'
// decoded weights (16 x n floats, 184 KB at K 2880) on every call: a block that large a heap returns to the system
// (Windows' heap decommits it on free, glibc maps one past its 128 KB mmap threshold), so each call faulted its pages
// in again (45 demand-zero pages a call, about 0.03 ms on a bare-metal Zen 4, more in a VM). On one thread: after
// the first call, repeats of it and of smaller calls allocate nothing and a call that needs more allocates once; on
// the house's thread pool (persistent workers), the first run allocates at most once a worker and the runs after it
// nothing.
typedef struct { int type; const unsigned char *w; size_t rs; int rows; const float *x; int n, nb; float *y; } tile_job;
static void tile_job_fn(void *ctx, int i0, int i1) {
    const tile_job *j = ctx;
    for (int i = i0; i < i1; i++)
        vec_dot_exact_tile(j->type, j->w, j->rs, j->rows, j->x, (size_t)j->n, j->nb, j->n, 1.0f,
                           j->y + (size_t)i * j->rows * j->nb, (size_t)j->rows);
}
static void tile_scratch(void) {
    const int type = T_BF16, n = 2880, n2 = 4096, rows = 64, nb = 16, items = 8;
    if (!vec_dot_exact_tile_ok(type, n)) {
        fprintf(stderr, "ok: the tile's scratch: skipped (this build has no exact tile)\n");
        return;
    }
    const size_t rs = ggml_row_size(type, n), rs2 = ggml_row_size(type, n2);
    unsigned char *w = malloc(rs2 * rows);
    float *x = malloc(sizeof(float) * (size_t)n2 * nb), *y = malloc(sizeof(float) * (size_t)rows * nb * items);
    unsigned s = 24680;
    uint16_t *h = (uint16_t *)w;
    for (size_t i = 0; i < (size_t)n2 * rows; i++) {
        s = s * 1103515245u + 12345u;
        h[i] = (uint16_t)(0x3f00u | (s >> 25));
    }
    for (size_t i = 0; i < (size_t)n2 * nb; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = ((s >> 9) & 1023) / 1024.0f - 0.5f;
    }
    vec_dot_exact_tile(type, w, rs, rows, x, n, nb, n, 1.0f, y, rows);   // this thread's first call
    const unsigned long a0 = vec_dot_exact_tile_allocs();
    for (int k = 0; k < 50; k++) vec_dot_exact_tile(type, w, rs, rows, x, n, nb, n, 1.0f, y, rows);
    for (int k = 0; k < 50; k++) vec_dot_exact_tile(type, w, rs, 5, x, n, 9, n, 1.0f, y, rows);
    const unsigned long a1 = vec_dot_exact_tile_allocs();
    vec_dot_exact_tile(type, w, rs2, rows, x, n2, nb, n2, 1.0f, y, rows);   // more than the scratch holds
    const unsigned long a2 = vec_dot_exact_tile_allocs();
    for (int k = 0; k < 20; k++) {
        vec_dot_exact_tile(type, w, rs2, rows, x, n2, nb, n2, 1.0f, y, rows);
        vec_dot_exact_tile(type, w, rs, rows, x, n, nb, n, 1.0f, y, rows);
    }
    const unsigned long a3 = vec_dot_exact_tile_allocs();
    char msg[256];
    snprintf(msg, sizeof msg, "the tile's scratch on one thread: 100 calls after its first allocate %lu times "
             "(want 0); a call needing more %lu (want 1), then 40 calls %lu (want 0)", a1 - a0, a2 - a1, a3 - a2);
    ck(a1 == a0 && a2 - a1 == 1 && a3 == a2, msg);
    // R4.48.20.4: NVFP4 tile scratch and allocation-failure fallback. At this point the thread holds exact r_grp * n2
    // floats from the BF16 call above. On NEON, T_NVFP4 needs r_grp * (n2 + n2/16) > r_grp * n2, so a failed growth
    // must fall back to vec_dot_multi (preserving exact vec_dot_nvfp4 bits) rather than falling through to the
    // generic dequant_row + exact_tile_block branch (which would reuse the held r_grp * n2 buffer with different bits).
    if (vec_dot_exact_tile_ok(T_NVFP4, n2)) {
        const int n3 = n2 + 64;
        const size_t rsn2 = ggml_row_size(T_NVFP4, n2), rsn3 = ggml_row_size(T_NVFP4, n3);
        const size_t bsn = ggml_row_size(T_NVFP4, ggml_block_size(T_NVFP4));
        unsigned char *wn = malloc(rsn3 * rows);
        float *xn = malloc(sizeof(float) * (size_t)n3 * nb);
        float *ref = malloc(sizeof(float) * (size_t)rows * nb);
        unsigned sn = 13579;
        for (size_t i = 0; i < rsn3 * rows; i++) { sn = sn * 1103515245u + 12345u; wn[i] = (unsigned char)(sn >> 16); }
        // Non-power-of-two UE4M3 scales (0x25..0x4d) so pre-scaling vs post-scaling differs in float32 rounding.
        for (size_t b = 0; b < rsn3 * rows; b += bsn)
            for (int k = 0; k < 4; k++) { sn = sn * 1103515245u + 12345u; wn[b + k] = (unsigned char)(0x25 + ((sn >> 16) % 41)); }
        for (size_t i = 0; i < (size_t)n3 * nb; i++) { sn = sn * 1103515245u + 12345u; xn[i] = ((sn >> 9) & 1023) / 1024.0f - 0.5f; }
        for (int c = 0; c < nb; c++)
            for (int r = 0; r < rows; r++)
                ref[(size_t)c * rows + r] = vec_dot(T_NVFP4, wn + (size_t)r * rsn2, xn + (size_t)c * n2, n2);
        // Trigger growth failure: on NEON n2 needs r_grp*(n2+n2/16) > held r_grp*n2; on AVX-512 test at n3 > n2.
        const int nf =
#if !defined(RUNNER_CANON_KERNELS) && !defined(RUNNER_NO_SIMD) && defined(__aarch64__) && defined(__ARM_NEON)
            n2;
#else
            n3;
#endif
        const size_t rsnf = nf == n2 ? rsn2 : rsn3;
        if (nf == n3)
            for (int c = 0; c < nb; c++)
                for (int r = 0; r < rows; r++)
                    ref[(size_t)c * rows + r] = vec_dot(T_NVFP4, wn + (size_t)r * rsn3, xn + (size_t)c * n3, n3);
        const unsigned long c0 = vec_dot_exact_tile_cols(), m0 = vec_dot_multi_calls();
        vec_dot_exact_tile_fail_next_alloc(1);
        vec_dot_exact_tile(T_NVFP4, wn, rsnf, rows, xn, nf, nb, nf, 1.0f, y, rows);
        vec_dot_exact_tile_fail_next_alloc(0);
        const unsigned long c1 = vec_dot_exact_tile_cols(), m1 = vec_dot_multi_calls();
        int bad_bits = 0;
        for (size_t i = 0; i < (size_t)rows * nb; i++)
            if (memcmp(&y[i], &ref[i], sizeof(float)) != 0) bad_bits++;
        snprintf(msg, sizeof msg, "NVFP4 tile alloc failure falls back to vec_dot_multi: tile cols +%lu (want 0), "
                 "multi calls +%lu (want > 0), mismatched floats %d (want 0)", c1 - c0, m1 - m0, bad_bits);
        ck(c1 == c0 && m1 > m0 && bad_bits == 0, msg);
        // Normal NVFP4 tile path at n3 > n2: grows once, then 20 repeats allocate 0 and take the tile.
        const unsigned long na0 = vec_dot_exact_tile_allocs(), nc0 = vec_dot_exact_tile_cols();
        vec_dot_exact_tile(T_NVFP4, wn, rsn3, rows, xn, n3, nb, n3, 1.0f, y, rows);
        const unsigned long na1 = vec_dot_exact_tile_allocs(), nc1 = vec_dot_exact_tile_cols();
        for (int k = 0; k < 20; k++) {
            vec_dot_exact_tile(T_NVFP4, wn, rsn3, rows, xn, n3, nb, n3, 1.0f, y, rows);
            vec_dot_exact_tile(T_NVFP4, wn, rsn2, rows, xn, n2, nb, n2, 1.0f, y, rows);
        }
        const unsigned long na2 = vec_dot_exact_tile_allocs();
        snprintf(msg, sizeof msg, "NVFP4 tile scratch on one thread: first larger call allocates %lu (want 1, tile "
                 "cols +%lu), 40 repeats after it %lu (want 0)", na1 - na0, nc1 - nc0, na2 - na1);
        ck(na1 - na0 == 1 && nc1 > nc0 && na2 == na1, msg);
        free(ref); free(xn); free(wn);
    }
    tpool *tp = tpool_create(4);
    if (tp) {
        tile_job j = { type, w, rs, rows, x, n, nb, y };
        const long l0 = vec_dot_exact_tile_live();   // this thread's scratch
        const unsigned long p0 = vec_dot_exact_tile_allocs();
        tpool_run(tp, tile_job_fn, &j, items);
        const unsigned long p1 = vec_dot_exact_tile_allocs();
        for (int k = 0; k < 20; k++) tpool_run(tp, tile_job_fn, &j, items);
        const unsigned long p2 = vec_dot_exact_tile_allocs();
        const long l1 = vec_dot_exact_tile_live();
        snprintf(msg, sizeof msg, "the tile's scratch on the thread pool (%d workers): its first run allocates %lu "
                 "times (want at most one a worker), 20 runs after it %lu (want 0)", tpool_size(tp), p1 - p0, p2 - p1);
        ck(p1 - p0 <= (unsigned long)tpool_size(tp) && p2 == p1, msg);
        tpool_destroy(tp);
        // the workers' scratches go with their threads (the key's destructor), so none is left as a leak
        const long l2 = vec_dot_exact_tile_live();
        snprintf(msg, sizeof msg, "the tile's scratch is freed when its thread exits: %ld threads held one with the "
                 "pool, %ld after it is destroyed (want this thread's %ld)", l1, l2, l0);
        ck(l0 == 1 && l1 > l0 && l2 == l0, msg);
    }
    free(w); free(x); free(y);
}

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
static void random_mxfp4(unsigned char *w, size_t bytes, unsigned s) {
    for (size_t i = 0; i < bytes; i++) { s = s * 1103515245u + 12345u; w[i] = (unsigned char)(s >> 16); }
    for (size_t b = 0; b + 17 <= bytes; b += 17) w[b] = (unsigned char)(117 + (b / 17) % 21);
}
static void decode_ratio_mxfp4(void) {
    // an L2-sized set (64 K weights, 256 KB of f32), decoded 32 times a
    // timing: a memory-sized one measures the store stream on some hosts,
    // which hides the decode
    const int n = 4096, rows = 16, reps = 32;
    f16_init();   // quants.c's tables (once; a loaded model built them already)
    size_t rs = ggml_row_size(T_MXFP4, n);
    unsigned char *w = malloc(rs * rows);
    float *y = malloc(sizeof(float) * (size_t)n * rows);
    random_mxfp4(w, rs * rows, 54321);
    double best_s = 1e9, best_c = 1e9;
    for (int attempt = 0, rounds = 5; attempt < 2; attempt++, rounds = 25) {
        best_s = 1e9; best_c = 1e9;
        for (int rep = 0; rep < rounds; rep++) {
            double t0 = route_now();
            for (int q = 0; q < reps; q++)
                for (int r = 0; r < rows; r++) dequant_row(T_MXFP4, w + rs * r, y + (size_t)n * r, n);
            double t1 = route_now();
            for (int q = 0; q < reps; q++)
                for (int r = 0; r < rows; r++) dequant_row_mxfp4_ref(w + rs * r, y + (size_t)n * r, n);
            double t2 = route_now();
            if (t1 - t0 < best_s) best_s = t1 - t0;
            if (t2 - t1 < best_c) best_c = t2 - t1;
        }
        if (best_c >= 1.3 * best_s) break;
        if (attempt == 0)
            fprintf(stderr, "re-measuring: MXFP4 row decode at %.2fx the C decode's speed in 5 rounds\n",
                    best_c / best_s);
    }
    double per = 1e9 / ((double)n * rows * reps);
    char msg[220];
    snprintf(msg, sizeof msg, "MXFP4 row decode: dequant_row %.3f ns a weight, the C decode %.3f "
             "(%.2fx faster, at least 1.30x)", best_s * per, best_c * per, best_c / best_s);
    ck(best_c >= 1.3 * best_s, msg);
    // the C decode's bytes under this binary's float environment, every
    // exponent (0 is the subnormal scale denormals-are-zero flushes, 255 the
    // infinite one)
    float *z = malloc(sizeof(float) * (size_t)n);
    int bad = 0;
    for (int e = 0; e < 256; e++) {
        for (int k = 0; k < n / 32; k++) w[(size_t)k * 17] = (unsigned char)e;
        dequant_row(T_MXFP4, w, y, n);
        dequant_row_mxfp4_ref(w, z, n);
        if (memcmp(y, z, sizeof(float) * (size_t)n)) bad++;
    }
    snprintf(msg, sizeof msg, "MXFP4 row decode under the runner's float flags: the C decode's "
             "bytes at %d of 256 exponents", 256 - bad);
    ck(bad == 0, msg);
    free(z); free(w); free(y);
}
#endif

int main(int argc, char **argv) {
    dispatch(argc > 1 ? argv[1] : "test-q8.gguf");
    ratio(T_Q8_0, "Q8_0");
    ratio(T_Q4_K, "Q4_K");
    ratio(T_IQ3_S, "IQ3_S");
    ratio(T_IQ4_XS, "IQ4_XS");
    ratio(T_NVFP4, "NVFP4");
    tile_scratch();
    tile_ratio(T_BF16, "BF16");
    tile_ratio(T_NVFP4, "NVFP4");
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    decode_ratio_mxfp4();
#endif
    if (!g_fail) puts("dot route gate ok");
    return g_fail;
}
