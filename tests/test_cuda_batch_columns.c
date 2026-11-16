// A microbatch column's place moves no bits on the device (R4.40.3).
//
// gpu_batch_decode orders a step's columns by LongRoPE factor set so that each
// set's columns rotate in one launch. That moves sequences to other columns of
// every batched matrix product, not only of the rope, so this gate checks the
// premise the ordering rests on: on a batch whose columns are all on one set,
// RUNNER_TEST_BATCH_REVERSE=1 reverses the column order, and every sequence
// must decode the same bits as in the caller's order, at widths 2 to 8.
//
//   test-batch-columns-cuda FILE CTX   (make test-cuda-batch-columns)
//
// Anti-vacuity: the batch must engage on the device, and the reversed run,
// on the same batch object, must hand sequence i the logits buffer the plain
// run handed sequence w-1-i (column c's logits live at a fixed place), so a
// seam that reverses nothing fails here instead of passing.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define env_set(k, v) _putenv_s(k, v)
#define env_unset(k)  _putenv_s(k, "")
#else
#define env_set(k, v) setenv(k, v, 1)
#define env_unset(k)  unsetenv(k)
#endif

enum { NS = 8, STEPS = 5 };   // widths 2 to 8: the exact twins at 5 to 7 and the class kernels at 8
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

static const int32_t PROMPT[NS][6] = { { 5, 9, 12 }, { 7, 3, 20, 11, 6 }, { 4, 4, 8, 15 }, { 2, 30, 17, 9, 1, 13 },
                                        { 6, 14, 3, 22 }, { 9, 1, 27, 5, 18, 2 }, { 11, 23 }, { 16, 8, 4, 29, 7 } };
static const int PLEN[NS] = { 3, 5, 4, 6, 4, 6, 2, 5 };

static bool prefill(model_t *m, int nv) {
    for (int i = 0; i < NS; i++) {
        int32_t t[6];
        for (int j = 0; j < PLEN[i]; j++) t[j] = PROMPT[i][j] % nv;
        if (!model_forward_batch(&m[i], t, PLEN[i], 0, true)) return false;   // NULL without logits
    }
    return true;
}

// STEPS steps of a w-wide batch over sequences 0..w-1, fixed tokens; logits
// copied to cap[s][i], the buffers each step returned in ptr[s][i]
static bool run(model_batch *mb, model_t *m, int w, int nv, float *cap, const float **ptr) {
    if (!prefill(m, nv)) return false;
    for (int s = 0; s < STEPS; s++) {
        int idx[NS], pos[NS];
        int32_t tok[NS];
        float *out[NS] = { NULL };
        for (int i = 0; i < w; i++) {
            idx[i] = i; pos[i] = PLEN[i] + s;
            tok[i] = (int32_t)(3 + (s * 7 + i * 5) % (nv - 4));
        }
        if (!model_batch_decode(mb, idx, tok, pos, w, out)) return false;
        for (int i = 0; i < w; i++) {
            if (!out[i]) return false;
            memcpy(cap + ((size_t)s * NS + i) * nv, out[i], sizeof(float) * (size_t)nv);
            ptr[s * NS + i] = out[i];
        }
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test-batch-columns-cuda FILE CTX\n"); return 2; }
    const char *path = argv[1];
    int ctx = atoi(argv[2]);
    env_unset("RUNNER_TEST_BATCH_REVERSE");
    model_t m[NS];
    for (int i = 0; i < NS; i++) {
        model_params p;
        memset(&p, 0, sizeof p);
        p.gpu_mode = GPU_AUTO;
        p.n_ctx = ctx;
        if (!model_load(&m[i], path, &p)) { fprintf(stderr, "cannot load %s\n", path); return 1; }
        if (!m[i].gpu) { fprintf(stderr, "test-cuda-batch-columns: skipped (no CUDA device took %s)\n", path); return 1; }
    }
    int nv = m[0].n_vocab;
    float *a = malloc(sizeof(float) * (size_t)STEPS * NS * nv), *r = malloc(sizeof(float) * (size_t)STEPS * NS * nv);
    if (!a || !r) return 1;
    for (int w = 2; w <= NS; w++) {
        model_t *seqs[NS];
        for (int i = 0; i < w; i++) seqs[i] = &m[i];
        model_batch *mb = model_batch_create(seqs, w);
        char what[200];
        snprintf(what, sizeof what, "%s, width %d: the microbatch engages on the device", path, w);
        ck(mb && model_batch_engaged(mb), what);
        const float *pa[STEPS * NS], *pr[STEPS * NS];
        bool ok_a = mb && run(mb, m, w, nv, a, pa);
        env_set("RUNNER_TEST_BATCH_REVERSE", "1");
        bool ok_r = mb && run(mb, m, w, nv, r, pr);
        env_unset("RUNNER_TEST_BATCH_REVERSE");
        int moved = 0, diff = 0;
        for (int s = 0; ok_a && ok_r && s < STEPS; s++)
            for (int i = 0; i < w; i++) {
                moved += pr[s * NS + i] == pa[s * NS + (w - 1 - i)];
                diff += memcmp(a + ((size_t)s * NS + i) * nv, r + ((size_t)s * NS + i) * nv,
                               sizeof(float) * (size_t)nv) != 0;
            }
        fprintf(stderr, "%s, width %d: %d of %d steps' columns reversed, %d of %d sequence steps differ\n",
                path, w, moved, STEPS * w, diff, STEPS * w);
        snprintf(what, sizeof what, "%s, width %d: the reversed run put every sequence in the mirrored column", path, w);
        ck(ok_a && ok_r && moved == STEPS * w, what);
        snprintf(what, sizeof what, "%s, width %d: every sequence decodes the same bits in either column order", path, w);
        ck(ok_a && ok_r && diff == 0, what);
        if (mb) model_batch_free(mb);
    }
    free(a); free(r);
    for (int i = 0; i < NS; i++) model_free(&m[i]);
    return g_fail;
}
