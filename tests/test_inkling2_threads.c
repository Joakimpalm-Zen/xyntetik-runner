// R4.48.6: Inkling 2 Mini's forward runs at its instance's pool size, the way a server sets it up, and two slots'
// forwards running at once give the same bits.
//
// A server loads each slot with a one-thread pool and swaps the shared pool in after the load (src/server.c). The
// engine's pool is its own (src/inkling2.c), so the count it runs at has to come from the instance at use. The
// observables are the engine's own: the count it runs at (ink2_get_threads) and what its pool really did
// (ink2_par_stats: loops split over more than one thread, and the threads the last one ran on). INK2_PAR_MIN=1
// makes every loop of the tiny file split, so a count above one is seen in the stats.
//
//  1. load with n_threads=1 (no pool swap): after a load m->tp is NULL on inkling2_loop, so the count comes from the
//     load's n_threads (1) and splits nothing (the CONTROL that the observable is not constant)
//  2. the pool swapped for four threads, as the server does: the forward runs on four, its loops split over four,
//     and its rows equal case 1's bit for bit (fails before the fix: the count stayed one)
//  3. two instances, pools of four and two, running forwards at once on their own threads: every row equals case
//     1's bit for bit (the count flips between loops; a loop runs with the count it read)
//  4. load with n_threads=4 WITHOUT a pool swap (the CLI / --score path where m->tp is NULL): runs on four
//     threads, splits loops over four, and matches case 1's rows bit for bit
//  5. CONTROL that the comparison sees a difference: a different prompt's rows differ from case 1's
//
// usage: test_inkling2_threads TINY.gguf (tests/fuzz/corpus/inkling_open/tiny-valid.gguf)
#include "model.h"
#include "gpu.h"
#include "inkling2.h"
#include "tpool.h"
#include "compat.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TOK = 8, ROUNDS = 25 };
static const int32_t PROMPT[N_TOK] = { 1, 5, 9, 2, 7, 3, 8, 4 };
static const int32_t OTHER[N_TOK] = { 4, 8, 3, 7, 2, 9, 5, 1 };
static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static bool load(model_t *m, const char *path, int threads) {
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = threads;
    p.n_ctx = 64;
    p.n_batch = 64;
    memset(m, 0, sizeof *m);
    return model_load(m, path, &p);
}
static void swap_pool(model_t *m, int threads) {   // what src/server.c does to a slot after its load
    tpool_destroy(m->tp);
    m->tp = tpool_create(threads);
}
static float *rows(model_t *m, const int32_t *ids) {
    const float *l = model_forward_batch(m, ids, N_TOK, 0, true);
    if (!l) return NULL;
    float *c = malloc(sizeof(float) * (size_t)m->n_vocab);
    memcpy(c, l, sizeof(float) * (size_t)m->n_vocab);
    return c;
}

typedef struct { model_t *m; const float *want; int bad, ran; } job;
static void *run(void *arg) {
    job *j = arg;
    for (int r = 0; r < ROUNDS; r++) {
        float *got = rows(j->m, PROMPT);
        if (!got || memcmp(got, j->want, sizeof(float) * (size_t)j->m->n_vocab)) j->bad++;
        j->ran++;
        free(got);
    }
    return NULL;
}

typedef struct {
    pthread_t main_th;
    _Atomic int worker_hits;
    _Atomic int nested_dispatches;
    _Atomic long inner_done;
} nested_ctx;

typedef struct {
    nested_ctx *nc;
    pthread_t outer_th;
} nested_inner_ctx;

static void nested_inner(void *arg, int64_t j0, int64_t j1) {
    nested_inner_ctx *ic = arg;
    if (!pthread_equal(pthread_self(), ic->outer_th))
        atomic_fetch_add(&ic->nc->nested_dispatches, 1);
    atomic_fetch_add(&ic->nc->inner_done, (long)(j1 - j0));
}

static void nested_outer(void *arg, int64_t i0, int64_t i1) {
    nested_ctx *nc = arg;
    pthread_t self = pthread_self();
    if (!pthread_equal(self, nc->main_th))
        atomic_fetch_add(&nc->worker_hits, 1);
    nested_inner_ctx ic = { nc, self };
    for (int64_t i = i0; i < i1; i++)
        ink2_par(4, 1, nested_inner, &ic);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s TINY.gguf\n", argv[0]); return 2; }
    plat_setenv("INK2_PAR_MIN", "1");
    plat_setenv("INK2_THREADS", "4");
    plat_setenv("RUNNER_INKLING2_EXPERT_CACHE", "64M");
    model_t *a = malloc(sizeof *a), *b = malloc(sizeof *b), *c = malloc(sizeof *c);
    if (!load(a, argv[1], 1)) { printf("FAIL: the tiny file does not load\n"); return 1; }
    long s0, s1, s2;
    int last;

    // 1. loaded with n_threads=1 (no pool swap: m->tp is NULL, count is the load's n_threads = 1)
    ink2_par_stats(&s0, NULL);
    float *one = rows(a, PROMPT);
    ink2_par_stats(&s1, NULL);
    CHECK(one != NULL, "case 1: the forward runs");
    if (!one) return 1;
    CHECK(ink2_get_threads() == 1, "case 1 (control): a one-thread pool runs the forward at %d thread(s), want 1",
          ink2_get_threads());
    CHECK(s1 == s0, "case 1 (control): %ld loops split over more than one thread, want 0", s1 - s0);

    // 2. the server swaps in the shared pool
    swap_pool(a, 4);
    float *four = rows(a, PROMPT);
    ink2_par_stats(&s2, &last);
    CHECK(four != NULL, "case 2: the forward runs");
    CHECK(ink2_get_threads() == 4, "case 2: after the swap to a four-thread pool the forward runs at %d thread(s), want 4",
          ink2_get_threads());
    CHECK(s2 > s1 && last == 4, "case 2: %ld loops split, the last over %d threads, want > 0 and 4", s2 - s1, last);
    CHECK(four && !memcmp(one, four, sizeof(float) * (size_t)a->n_vocab),
          "case 2: four threads give one thread's rows bit for bit (%d logits)", a->n_vocab);

    // 3. two slots at once, pools of four and two
    if (!load(b, argv[1], 1)) { printf("FAIL: the second instance does not load\n"); return 1; }
    swap_pool(b, 2);
    long t0, t1;
    ink2_par_stats(&t0, NULL);
    job ja = { a, one, 0, 0 }, jb = { b, one, 0, 0 };
    pthread_t ta, tb;
    pthread_create(&ta, NULL, run, &ja);
    pthread_create(&tb, NULL, run, &jb);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    ink2_par_stats(&t1, NULL);
    CHECK(ja.ran == ROUNDS && jb.ran == ROUNDS && !ja.bad && !jb.bad,
          "case 3: two slots at once (4 and 2 threads), %d + %d forwards, %d + %d differ from one thread's rows",
          ja.ran, jb.ran, ja.bad, jb.bad);
    CHECK(t1 > t0, "case 3: %ld loops split while the two ran", t1 - t0);

    // 4. load with n_threads=4 WITHOUT a pool swap (CLI path: CLI binds its own 4-thread tpool onto c->tp)
    if (!load(c, argv[1], 4)) { printf("FAIL: the third instance does not load\n"); return 1; }
    CHECK(c->tp != NULL && tpool_size(c->tp) == 4, "case 4: CLI load bound 4-thread tpool (%d)", tpool_size(c->tp));
    long s3 = 0; int last3 = 1;
    float *cli4 = rows(c, PROMPT);
    ink2_par_stats(&s3, &last3);
    CHECK(cli4 != NULL, "case 4: the CLI 4-thread forward runs");
    CHECK(ink2_get_threads() == 4, "case 4: CLI load with n_threads=4 runs at %d thread(s), want 4", ink2_get_threads());
    CHECK(s3 > t1 && last3 == 4, "case 4: %ld loops split, the last over %d threads, want > 0 and 4", s3 - t1, last3);
    CHECK(cli4 && !memcmp(one, cli4, sizeof(float) * (size_t)a->n_vocab),
          "case 4: CLI 4 threads match case 1's rows bit for bit (%d logits)", a->n_vocab);

    // 5. the comparison sees a difference
    float *other = rows(a, OTHER);
    CHECK(other && memcmp(one, other, sizeof(float) * (size_t)a->n_vocab),
          "case 5 (control): a different prompt's rows differ from case 1's");

    // 6. R4.48.21: inkling2_loop counts as recurrent (model_has_recurrent: the engine re-prefills rather than rewind
    //    part of the way) but holds no SSM fold, so every fold function declines it. blob_save and blob_load did not:
    //    they divided by the model's zero SSM heads (a context pin on a served inkling2_loop file died of SIGFPE).
    unsigned char blob[64];
    CHECK(model_has_recurrent(a) && model_recurrent_blob_bytes(a) == 0 && !model_recurrent_blob_save(a, blob) &&
          !model_recurrent_blob_load(a, blob, PROMPT, 1) && !model_recurrent_restore_row(a, 0),
          "case 6: the fold functions decline inkling2_loop (blob bytes %zu, save, load and restore refused)",
          model_recurrent_blob_bytes(a));

    // 6b. nested ink2_par from inside tpool workers runs serially even with INK2_THREADS=4 fallback active
    nested_ctx nc = { .main_th = pthread_self() };
    long s5 = 0, s6 = 0;
    int last6 = 1;
    ink2_par_stats(&s5, NULL);
    struct tpool *prev_tp = ink2_set_tpool(c->tp);
    ink2_par(4, 1, nested_outer, &nc);
    ink2_set_tpool(prev_tp);
    ink2_par_stats(&s6, &last6);
    CHECK(s6 - s5 == 1 && last6 == 4,
          "case 6b: nested ink2_par from 4-thread workers split %ld loop(s), last_pool %d (want 1 and 4)",
          s6 - s5, last6);
    CHECK(atomic_load(&nc.worker_hits) == 3 && atomic_load(&nc.nested_dispatches) == 0 &&
          atomic_load(&nc.inner_done) == 16,
          "case 6b: workers ran %d slices serially inside caller workers (nested dispatches %d, inner_done %ld)",
          atomic_load(&nc.worker_hits), atomic_load(&nc.nested_dispatches), atomic_load(&nc.inner_done));

    free(one); free(four); free(cli4); free(other);
    model_free(a); model_free(b); model_free(c);
    free(a); free(b); free(c);
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
