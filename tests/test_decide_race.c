// /v1/decide and /v1/rerank number each response ("decide-N", "rerank-N")
// from a counter every slot shares, and parallel slots answer on their own
// threads. The counters were plain statics bumped with ++, so two requests
// answered at once could read the same value and carry one id: a review
// probe, 2026-10-16, saw 3 duplicates in 16,000 concurrent handler calls,
// and a client correlating or deduplicating by id would merge two answers.
//
// Built with -fsanitize=thread by `make test-decide-race`: TSan halting on
// the unsynchronized increment is the gate, whatever the timing. The check
// that every id came out distinct rides along. Each thread loads its own
// model and engine, as a slot does. With "gpu" after the model the engines
// run on the build's GPU backend (`make test-decide-race-gpu`), so the
// backend's own first-use state is raced too: kernel-selection flags,
// dispatch counters, once-only notices. There every engine must start on the
// device and still be on it at the end: on CUDA, one engine's graph capture
// and another's synchronous copy collided, and the copier's forward failed
// and dropped it to the CPU for good (2026-10-16), with every answer still
// correct, so only this check could see it.
#include "runner.h"
#include "decide.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { THREADS = 4, ROUNDS = 25 };

static const char *g_model = "test.gguf";
static int g_gpu_mode = GPU_OFF;
static unsigned long long g_ids[2][THREADS * ROUNDS];
static int g_n[2];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_fail;

static const char DECIDE[] =
    "{\"state\":\"The quick brown fox\",\"questions\":[{\"question\":\"What does it do?\","
    "\"options\":[\"jumps\",\"sleeps\"]}]}";
static const char RERANK[] =
    "{\"query\":\"fox\",\"documents\":[\"a fox\",\"a dog\"],\"rendering\":\"raw-v1\"}";

static void record(int kind, const char *body, const char *prefix) {
    const char *p = body ? strstr(body, prefix) : NULL;
    pthread_mutex_lock(&g_mu);
    if (!p) {
        fprintf(stderr, "FAIL: no %s id in %s\n", prefix, body ? body : "(null)");
        g_fail = 1;
    } else {
        g_ids[kind][g_n[kind]++] = strtoull(p + strlen(prefix), NULL, 10);
    }
    pthread_mutex_unlock(&g_mu);
}

static void *worker(void *arg) {
    (void)arg;
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = g_gpu_mode;
    p.n_threads = 1;
    p.n_ctx = 512;
    p.n_batch = 8;
    model_t m;
    tokenizer tok;
    sampler smp = {0};
    engine e;
    memset(&m, 0, sizeof m);
    memset(&e, 0, sizeof e);
    sampler_reset(&smp);
    if (!model_load(&m, g_model, &p) || !tokenizer_init(&tok, &m.gf) ||
        !engine_init(&e, &m, &tok, &smp)) {
        fprintf(stderr, "FAIL: cannot load %s\n", g_model);
        g_fail = 1;
        return NULL;
    }
    bool on_gpu = m.gpu != NULL;
    if (g_gpu_mode == GPU_AUTO && !on_gpu) {
        fprintf(stderr, "FAIL: an engine did not start on the GPU\n");
        g_fail = 1;
    }
    jv *dq = json_parse(DECIDE, strlen(DECIDE)), *rq = json_parse(RERANK, strlen(RERANK));
    for (int r = 0; r < ROUNDS; r++) {
        const char *err = NULL;
        sbuf out = {0};
        int st = decide_handle(&e, dq, "test", &out, &err);
        if (st != 200) { fprintf(stderr, "FAIL: decide %d %s\n", st, err ? err : ""); g_fail = 1; }
        else record(0, out.s, "\"id\":\"decide-");
        free(out.s);
        sbuf rout = {0};
        st = rerank_handle(&e, rq, "test", 0, &rout, &err);
        if (st != 200) { fprintf(stderr, "FAIL: rerank %d %s\n", st, err ? err : ""); g_fail = 1; }
        else record(1, rout.s, "\"id\":\"rerank-");
        free(rout.s);
    }
    if (on_gpu && (!m.gpu || m.route_fallback[0])) {
        fprintf(stderr, "FAIL: an engine left the GPU mid-run (%s)\n",
                m.route_fallback[0] ? m.route_fallback : "backend released");
        g_fail = 1;
    }
    jv_free(dq);
    jv_free(rq);
    free(e.hist);
    free(e.cdoc);
    tokenizer_free(&tok);
    model_free(&m);
    return NULL;
}

static int cmp_u64(const void *a, const void *b) {
    unsigned long long x = *(const unsigned long long *)a, y = *(const unsigned long long *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    if (argc > 1) g_model = argv[1];
    if (argc > 2 && !strcmp(argv[2], "gpu")) g_gpu_mode = GPU_AUTO;
    f16_init();
    pthread_t th[THREADS];
    for (int i = 0; i < THREADS; i++) pthread_create(&th[i], NULL, worker, NULL);
    for (int i = 0; i < THREADS; i++) pthread_join(th[i], NULL);
    static const char *const kind[2] = { "decide", "rerank" };
    for (int k = 0; k < 2; k++) {
        if (g_n[k] != THREADS * ROUNDS) {
            fprintf(stderr, "FAIL: %d %s answers, want %d\n", g_n[k], kind[k], THREADS * ROUNDS);
            g_fail = 1;
        }
        qsort(g_ids[k], (size_t)g_n[k], sizeof g_ids[k][0], cmp_u64);
        for (int i = 1; i < g_n[k]; i++)
            if (g_ids[k][i] == g_ids[k][i - 1]) {
                fprintf(stderr, "FAIL: two %s answers carry id %llu\n", kind[k], g_ids[k][i]);
                g_fail = 1;
            }
    }
    if (g_fail) return 1;
    printf("decide race: %d decide and %d rerank answers from %d threads, every id distinct\n",
           g_n[0], g_n[1], THREADS);
    return 0;
}
