// The Metal microbatch's whole contract in one word: BIT-IDENTICAL.
//
// model_batch_decode exists so N server slots share one weight sweep per
// decode step. The CUDA backend established the discipline (batched kernels
// are TWINS — same per-element order, byte-equal to sequential decode) and
// this gate holds the Metal implementation to it: three sequences with
// different prompts and different positions are decoded twice in one
// process, once through the microbatch and once sequentially, and every
// logit of every step of every sequence must match byte for byte.
//
// Anti-vacuity: if the microbatch never engages (gpu_batch_create returned
// NULL), the comparison is sequential-vs-sequential and proves nothing — on
// a Metal host with an eligible dense fixture that is a FAIL, not a skip.
// No GPU at all skips loudly.
//
// Eleven sequences, not three, since 2026-10-03: a step with more than
// MODEL_BATCH_MAX (8) ready sequences runs as consecutive microbatches, and
// the backend's logits buffer is reused by each. With three the split never
// happened, so the gate could not see the second microbatch overwriting the
// first one's rows -- which served one request another request's next-token
// distribution on the lab's 16-slot CUDA server (garbled code, 25% of
// same-seed pairs different) and reproduced on Metal at 4 of 12.
//
// Twice since 2026-10-21: once from the short prompts above and once behind a
// LONG_PREFIX-token prefix, so the steps run past position 128, where a Metal
// decode step's attention switches to its chunked kernels. The microbatch ran
// the single-pass kernel for every column there, so a slot's step differed
// from its sequential step in the last bits, and decoding never got that far
// in this gate (a speculative verify through the microbatch found it).
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NSEQ = 11, STEPS = 12, PROMPT_MAX = 8, LONG_PREFIX = 300, CTX = 512 };

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// distinct prompts, distinct lengths — positions must differ per column
static const int32_t PROMPTS[NSEQ][PROMPT_MAX] = {
    { 5, 9, 12 },
    { 7, 3, 20, 11, 6 },
    { 4, 4, 8, 15, 16, 23, 42 },
    { 2, 30 },
    { 9, 1, 44, 17 },
    { 13, 13, 13, 6, 31, 8 },
    { 21 },
    { 40, 2, 9, 33, 5, 18, 27, 3 },
    { 6, 11, 25 },
    { 38, 14, 7, 22, 9 },
    { 3, 36, 10, 28 },
};
static const int PLEN[NSEQ] = { 3, 5, 7, 2, 4, 6, 1, 8, 3, 5, 4 };

static bool load_all(const char *path, model_t *m, int n) {
    for (int i = 0; i < n; i++) {
        model_params p;
        memset(&p, 0, sizeof(p));
        p.gpu_mode = GPU_AUTO;
        p.n_ctx = CTX;
        if (!model_load(&m[i], path, &p)) {
            fprintf(stderr, "cannot load %s (instance %d)\n", path, i);
            return false;
        }
    }
    return true;
}

// prefill each sequence with `prefix` filler tokens (distinct per sequence)
// and then its prompt; return per-seq next position
static bool prefill(model_t *m, int *pos, int prefix) {
    static int32_t fill[LONG_PREFIX];
    for (int i = 0; i < NSEQ; i++) {
        int nv = m[i].n_vocab;
        for (int t = 0; t < prefix; t++) fill[t] = (int32_t)((t * 37 + i * 11 + 5) % nv);
        for (int t = 0; t < prefix; t += 32) {
            int k = prefix - t < 32 ? prefix - t : 32;
            if (!model_forward_batch(&m[i], fill + t, k, t, true)) return false;
        }
        for (int t = 0; t < PLEN[i]; t++)
            if (!model_forward(&m[i], PROMPTS[i][t], prefix + t)) return false;
        pos[i] = prefix + PLEN[i];
    }
    return true;
}

static int argmax(const float *v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

// one microbatch-vs-sequential comparison behind `prefix` filler tokens;
// returns 0 on a pass, 1 on a failure, 2 when no GPU backend loaded
static int run(const char *path, int prefix) {
    model_t m[NSEQ];
    if (!load_all(path, m, NSEQ)) return 1;
    if (!m[0].gpu) {
        for (int i = 0; i < NSEQ; i++) model_free(&m[i]);
        return 2;
    }
    int nv = m[0].n_vocab;
    float *cap = malloc(sizeof(float) * (size_t)STEPS * NSEQ * nv);
    int32_t toks[STEPS][NSEQ];
    if (!cap) return 1;

    // ---- pass A: microbatch ------------------------------------------------
    int pos[NSEQ];
    if (!prefill(m, pos, prefix)) return 1;
    model_t *seqp[NSEQ];
    for (int i = 0; i < NSEQ; i++) seqp[i] = &m[i];
    model_batch *mb = model_batch_create(seqp, NSEQ);
    ck(mb != NULL, "model_batch_create");
    bool engaged = mb && model_batch_engaged(mb);
    int32_t cur[NSEQ];
    for (int i = 0; i < NSEQ; i++) cur[i] = PROMPTS[i][PLEN[i] - 1] % nv;
    for (int s = 0; s < STEPS && !g_fail; s++) {
        int idx[NSEQ];
        float *out[NSEQ];
        for (int i = 0; i < NSEQ; i++) idx[i] = i;
        ck(model_batch_decode(mb, idx, cur, pos, NSEQ, out),
           "model_batch_decode step");
        for (int i = 0; i < NSEQ && !g_fail; i++) {
            memcpy(cap + ((size_t)s * NSEQ + i) * nv, out[i],
                   sizeof(float) * (size_t)nv);
            toks[s][i] = argmax(out[i], nv);
            cur[i] = toks[s][i];
            pos[i]++;
        }
    }
    model_batch_free(mb);
    for (int i = 0; i < NSEQ; i++) model_free(&m[i]);
    if (g_fail) { printf("batch-identity: FAILED\n"); free(cap); return 1; }

    // ---- pass B: sequential reference, fresh state -------------------------
    if (!load_all(path, m, NSEQ)) return 1;
    if (!prefill(m, pos, prefix)) return 1;
    for (int i = 0; i < NSEQ; i++) cur[i] = PROMPTS[i][PLEN[i] - 1] % nv;
    size_t n_diff = 0;
    for (int s = 0; s < STEPS; s++) {
        for (int i = 0; i < NSEQ; i++) {
            float *lg = model_forward(&m[i], cur[i], pos[i]);
            ck(lg != NULL, "sequential forward");
            if (!lg) break;
            const float *want = cap + ((size_t)s * NSEQ + i) * nv;
            if (memcmp(lg, want, sizeof(float) * (size_t)nv) != 0) {
                n_diff++;
                if (n_diff == 1)
                    for (int k = 0; k < nv; k++)
                        if (lg[k] != want[k]) {
                            fprintf(stderr, "first diff: prefix %d step %d "
                                    "seq %d (position %d) elem %d: seq=%g "
                                    "batch=%g\n", prefix, s, i, pos[i], k,
                                    lg[k], want[k]);
                            break;
                        }
            }
            int t = argmax(lg, nv);
            ck(t == toks[s][i], "greedy token matches the microbatch");
            cur[i] = t;
            pos[i]++;
        }
    }
    for (int i = 0; i < NSEQ; i++) model_free(&m[i]);
    free(cap);

    ck(n_diff == 0, "every step's logits are byte-identical across the paths");
    if (!engaged) {
        // sequential fallback compared against sequential: correct, but the
        // microbatch was never on trial. On a Metal host with a dense
        // fixture that is a broken gate, not a pass.
        printf("FAIL: the microbatch never engaged — this compared the "
               "sequential path against itself\nbatch-identity: FAILED\n");
        return 1;
    }
    return g_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    f16_init();
    int r = run(path, 0);
    if (r == 2) {
        printf("batch-identity: ok (skipped — no GPU backend for %s)\n", path);
        return 0;
    }
    if (r == 0) r = run(path, LONG_PREFIX);
    if (r != 0 || g_fail) { printf("batch-identity: FAILED\n"); return 1; }
    printf("batch-identity: ok (%d seqs x %d steps at positions from %d and "
           "from %d, byte-identical, microbatch engaged)\n", NSEQ, STEPS,
           PROMPT_MAX, LONG_PREFIX);
    return 0;
}
