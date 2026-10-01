// Split attention on the CPU (R3.1.7): the same bits with more threads busy.
//
// attn_heads parallelizes one token's attention over query heads, so a box
// with more threads than heads idles through a long-context decode. The split
// path spreads the work over (head, position chunk) and (head, channel slice)
// items instead. It never merges partial softmaxes -- that would change the
// summation order and every pinned output with it -- so the claim is
// BIT-IDENTITY with the head-parallel path, and this is its gate:
//
//   the reference is one thread, split off (attn_heads, exactly as shipped);
//   every other run -- 1, 2, 3, 4 and 7 threads, split forced on at every
//   span and split left to its own decision -- must produce the same logits,
//   byte for byte, at every one of N_POS decode positions, and the same
//   last-token logits from a batched prefill.
//
// Forced mode exercises the split at spans of 1 and at ragged chunk edges;
// 7 threads on a 4-head fixture is where the automatic decision splits.
// The cases cover every branch of the attention inner loops: f16, q8_0,
// fp4 and k8v4 caches, a head width that is not a multiple of the channel
// slice (24), a sliding-window ring, attention sinks and tied-V (K derived
// from the cached V).
//
// Usage: test-attn-split PATH[:opt,...] ...
//   opts: q8 | fp4 | k8v4 (KV type), ring (RUNNER_KV_RING), tied (RUNNER_TIEDV)
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MinGW has no setenv/unsetenv; an empty value removes the variable there
#ifdef _WIN32
#define env_set(k, v) _putenv_s(k, v)
#define env_unset(k)  _putenv_s(k, "")
#else
#define env_set(k, v) setenv(k, v, 1)
#define env_unset(k)  unsetenv(k)
#endif

enum { N_POS = 600, N_CTX = 1024, N_BATCH_CHUNK = 64 };

typedef struct { bool q8, fp4, k8v4, ring, tied; } opts;

static bool load(model_t *m, const char *path, const opts *o, int threads) {
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_ctx = N_CTX;
    p.n_threads = threads;
    p.kv_q8 = o->q8;
    p.kv_fp4 = o->fp4;
    p.kv_split = o->k8v4;
    if (o->ring) env_set("RUNNER_KV_RING", "1"); else env_unset("RUNNER_KV_RING");
    if (o->tied) env_set("RUNNER_TIEDV", "1"); else env_unset("RUNNER_TIEDV");
    return model_load(m, path, &p);
}

static int32_t tok_at(int i, int nv) { return (int32_t)(3 + (i * 131 + 17) % (nv - 3)); }

// every position's logits, solo decode; then the last token's logits from a
// batched prefill of the same tokens (N_BATCH_CHUNK at a time)
static bool run(const char *path, const opts *o, int threads, int split,
                float *solo, float *batch, int *nv_out, char *what, size_t what_n,
                long *split_runs, int *n_head) {
    model_t m;
    if (!load(&m, path, o, threads)) { fprintf(stderr, "attn-split: cannot load %s\n", path); return false; }
    m.attn_split = split;
    int nv = m.n_vocab;
    *nv_out = nv;
    if (o->ring && m.kv_ring <= 0) { fprintf(stderr, "attn-split: %s: ring did not engage\n", path); model_free(&m); return false; }
    if (o->tied && !m.tied_v) { fprintf(stderr, "attn-split: %s: tied-V did not engage\n", path); model_free(&m); return false; }
    for (int t = 0; t < N_POS; t++) {
        float *lg = model_forward(&m, tok_at(t, nv), t);
        if (!lg) { fprintf(stderr, "attn-split: forward failed\n"); model_free(&m); return false; }
        memcpy(solo + (size_t)t * nv, lg, sizeof(float) * (size_t)nv);
    }
    int32_t *toks = malloc(sizeof(int32_t) * N_POS);
    if (!toks) { model_free(&m); return false; }
    for (int t = 0; t < N_POS; t++) toks[t] = tok_at(t, nv);
    float *lg = NULL;
    for (int at = 0; at < N_POS; at += N_BATCH_CHUNK) {
        int k = N_POS - at < N_BATCH_CHUNK ? N_POS - at : N_BATCH_CHUNK;
        lg = model_forward_batch(&m, toks + at, k, at, at + k == N_POS);
    }
    free(toks);
    if (!lg) { fprintf(stderr, "attn-split: batch forward failed\n"); model_free(&m); return false; }
    memcpy(batch, lg, sizeof(float) * (size_t)nv);
    snprintf(what, what_n, "%d heads", m.n_head);
    *split_runs = m.attn_split_runs;
    *n_head = m.n_head;
    model_free(&m);
    return true;
}

static int check(const char *arg) {
    char path[512];
    opts o = {0};
    snprintf(path, sizeof path, "%s", arg);
    char *c = strchr(path, ':');
    if (c) {
        *c++ = 0;
        for (char *t = strtok(c, ","); t; t = strtok(NULL, ",")) {
            if (!strcmp(t, "q8")) o.q8 = true;
            else if (!strcmp(t, "fp4")) o.fp4 = true;
            else if (!strcmp(t, "k8v4")) o.k8v4 = true;
            else if (!strcmp(t, "ring")) o.ring = true;
            else if (!strcmp(t, "tied")) o.tied = true;
            else { fprintf(stderr, "attn-split: unknown option %s\n", t); return 1; }
        }
    }
    int nv = 0;
    char what[64] = "";
    // the reference fixes n_vocab; buffers are sized from it
    model_t probe;
    if (!load(&probe, path, &o, 1)) { fprintf(stderr, "attn-split: cannot load %s\n", path); return 1; }
    nv = probe.n_vocab;
    model_free(&probe);
    float *ref = malloc(sizeof(float) * (size_t)N_POS * nv), *refb = malloc(sizeof(float) * nv);
    float *got = malloc(sizeof(float) * (size_t)N_POS * nv), *gotb = malloc(sizeof(float) * nv);
    if (!ref || !refb || !got || !gotb) return 1;
    long runs = 0;
    int n_head = 0;
    if (!run(path, &o, 1, -1, ref, refb, &nv, what, sizeof what, &runs, &n_head)) return 1;
    if (runs) { printf("attn-split: FAIL %s: split off, yet %ld split passes ran\n", arg, runs); return 1; }

    static const int THREADS[] = { 1, 2, 3, 4, 7 };
    static const int MODES[] = { 1, 0 };   // forced split, automatic
    int bad = 0;
    for (size_t ti = 0; ti < sizeof THREADS / sizeof *THREADS; ti++)
        for (size_t mi = 0; mi < sizeof MODES / sizeof *MODES; mi++) {
            int n = THREADS[ti], mode = MODES[mi];
            if (!run(path, &o, n, mode, got, gotb, &nv, what, sizeof what, &runs, &n_head)) return 1;
            // engagement: an identity that holds because the split never ran
            // proves nothing. Forced mode must split; automatic mode must
            // split exactly when more than a quarter of the threads would
            // idle under the head split (the spans here pass the length bar)
            int rounds = (n_head + n - 1) / n;
            bool want = mode == 1 || (n > 1 && n_head * 4 < rounds * n * 3);
            if (want != (runs > 0)) {
                printf("attn-split: FAIL %s %d threads, split %s: %ld split passes, "
                       "expected %s\n", arg, n, mode ? "forced" : "auto", runs,
                       want ? "some" : "none");
                bad = 1;
            }
            int first = -1, differ = 0;
            for (int t = 0; t < N_POS; t++)
                if (memcmp(got + (size_t)t * nv, ref + (size_t)t * nv, sizeof(float) * (size_t)nv)) {
                    differ++;
                    if (first < 0) first = t;
                }
            bool bdiff = memcmp(gotb, refb, sizeof(float) * (size_t)nv) != 0;
            printf("attn-split: %s (%s) %d thread%s, split %s: %d/%d decode positions "
                   "differ", arg, what, n, n == 1 ? "" : "s",
                   mode ? "forced" : "auto", differ, N_POS);
            if (first >= 0) printf(", the first at %d", first);
            printf("\n");
            if (bdiff) printf("attn-split:   batched prefill's last logits differ\n");
            if (differ || bdiff) bad = 1;
        }
    free(ref); free(refb); free(got); free(gotb);
    return bad;
}

int main(int argc, char **argv) {
    int rc = 0;
    if (argc < 2) rc |= check("test.gguf");
    for (int i = 1; i < argc; i++) rc |= check(argv[i]);
    printf(rc ? "attn-split: FAIL\n" : "attn-split: ok\n");
    return rc;
}
