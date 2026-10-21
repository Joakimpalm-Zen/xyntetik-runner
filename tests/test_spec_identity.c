// A speculative verify round's rows are the plain path's bits.
//
// The speculative walk decides every token from the verify rows' logits, so
// it is the plain path only if row r of a verify round (model_forward_verify
// at pos) is BYTE for byte what a solo decode step at pos + r computes after
// the same history. Two instances of one model follow the same token stream:
//
//   V verifies a round of n rows, then rewinds to the round's accepted prefix
//     (pos + a + 1) as the walk does, so the next round overwrites the rows
//     it did not keep; on CUDA its device mirror keeps the rows it already
//     holds (kv_rows) instead of taking the whole prefix from the host again.
//   R decodes the same rows one solo step at a time, and before each round
//     marks its whole cache as written by the host (model_kv_host_wrote), so
//     its device reads the authoritative host rows every round.
//
// Every row of every round must match. Round widths run 2..16 (the walk's
// widest is SPEC_DRAFT_MAX = 16), acceptance runs from 0 to every draft, and
// the drafted tokens are arbitrary ids: the contract does not depend on them
// being good drafts. A recurrent model rolls its fold back after a round as
// the walk does (spec_fold_sync: the per-row checkpoint, else the round-start
// snapshot and an exact re-fold of the kept rows), and R reloads its fold
// as it stood at the round's start and re-runs the kept rows one solo step
// at a time, so the next round starts both from the fold plain decoding
// would hold. R saves that fold as a blob (model_recurrent_blob_save, the
// prefix cache's copy): the snapshot the walk uses is, on CUDA, a mark on
// the device's pre-forward copy, valid for the one verify forward after it,
// and R runs several forwards before it rolls back.
//
// Anti-vacuity: on a CUDA build with a device, the exact verify tiles must
// have run (gpu_exact_verify_tiles), or the rows came from some other path.
//
//   test-spec-identity MODEL.gguf [cpu]
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROUNDS = 24, PROMPT = 9, ROWS_MAX = 16 };

static int argmax(const float *v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

static bool load(model_t *m, const char *path, bool cpu) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = cpu ? GPU_OFF : GPU_AUTO;
    p.n_ctx = 256;
    if (!model_load(m, path, &p)) {
        fprintf(stderr, "cannot load %s\n", path);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    bool cpu = argc > 2 && strcmp(argv[2], "cpu") == 0;
    f16_init();

    model_t V, R;
    if (!load(&V, path, cpu) || !load(&R, path, cpu)) return 1;
    const char *where = V.gpu ? "gpu" : "cpu";
    if (!model_spec_verify_ok(&V)) {
        printf("spec-identity: ok (skipped: %s cannot serve the verify walk for %s)\n",
               where, path);
        model_free(&V); model_free(&R);
        return 0;
    }
    int nv = V.n_vocab;
    float *rows = malloc(sizeof(float) * (size_t)ROWS_MAX * nv);
    if (!rows) return 1;
    unsigned long tiles0 = gpu_exact_verify_tiles();
    size_t blob_n = model_recurrent_blob_bytes(&R);
    uint8_t *blob = blob_n ? malloc(blob_n) : NULL;
    if (blob_n && !blob) return 1;

    int32_t prompt[PROMPT];
    for (int i = 0; i < PROMPT; i++) prompt[i] = (int32_t)((5 + 7 * i) % nv);
    float *lv = model_forward_batch(&V, prompt, PROMPT, 0, true);
    float *lr = model_forward_batch(&R, prompt, PROMPT, 0, true);
    if (!lv || !lr || memcmp(lv, lr, sizeof(float) * (size_t)nv) != 0) {
        fprintf(stderr, "FAIL: the two instances' prefills differ\n");
        return 1;
    }
    int32_t cur = argmax(lv, nv);
    int pos = PROMPT;
    uint32_t lcg = 12345u;
    int fails = 0, checked = 0;
    for (int round = 0; round < ROUNDS && !fails; round++) {
        int n = 2 + round % (ROWS_MAX - 1);
        if (n > V.spec_batch) n = V.spec_batch;
        if (pos + n >= V.n_ctx) break;
        int32_t b[ROWS_MAX];
        b[0] = cur;
        for (int r = 1; r < n; r++) {
            lcg = lcg * 1664525u + 1013904223u;
            b[r] = (int32_t)((lcg >> 8) % (uint32_t)nv);
        }
        bool recur = model_has_recurrent(&V);
        if (recur && (!model_recurrent_snapshot(&V, pos) || !model_recurrent_blob_save(&R, blob))) {
            fprintf(stderr, "FAIL: no round-start snapshot at %d\n", pos);
            return 1;
        }
        if (!model_forward_verify(&V, b, n, pos)) {
            fprintf(stderr, "FAIL: model_forward_verify refused a %d-row round at %d\n", n, pos);
            return 1;
        }
        for (int r = 0; r < n; r++) {
            float *lg = model_spec_row_logits(&V, r);
            if (!lg) { fprintf(stderr, "FAIL: no row %d\n", r); return 1; }
            memcpy(rows + (size_t)r * nv, lg, sizeof(float) * (size_t)nv);
        }
        model_kv_host_wrote(&R, 0);
        for (int r = 0; r < n; r++) {
            float *ls = model_forward(&R, b[r], pos + r);
            if (!ls) { fprintf(stderr, "FAIL: solo step at %d\n", pos + r); return 1; }
            const float *want = rows + (size_t)r * nv;
            checked++;
            if (memcmp(ls, want, sizeof(float) * (size_t)nv) != 0) {
                for (int k = 0; k < nv; k++)
                    if (ls[k] != want[k]) {
                        fprintf(stderr, "FAIL: round %d (%d rows at %d) row %d elem %d: "
                                "solo=%.9g verify=%.9g\n", round, n, pos, r, k, ls[k], want[k]);
                        break;
                    }
                fails++;
                break;
            }
        }
        // the walk keeps a drafts and the token row a samples: the next
        // round starts after them, over the rows it did not keep
        int a = round % n;
        int acc = a + 1;
        if (recur && acc < n) {
            if (!model_recurrent_restore_row(&V, acc - 1)) {
                if (!model_recurrent_restore(&V, pos)) {
                    fprintf(stderr, "FAIL: no fold to roll back to at %d\n", pos);
                    return 1;
                }
                model_forward_refold(&V, b, acc, pos);
            }
            if (!model_recurrent_blob_load(&R, blob)) {
                fprintf(stderr, "FAIL: the reference cannot reload its fold at %d\n", pos);
                return 1;
            }
            for (int r = 0; r < acc; r++)
                if (!model_forward(&R, b[r], pos + r)) {
                    fprintf(stderr, "FAIL: re-run step at %d\n", pos + r);
                    return 1;
                }
        }
        cur = argmax(rows + (size_t)a * nv, nv);
        pos += acc;
    }
    unsigned long tiles = gpu_exact_verify_tiles() - tiles0;
    free(rows);
    free(blob);
    model_free(&V);
    model_free(&R);
    if (fails) {
        printf("spec-identity: FAILED (%s, %s)\n", where, path);
        return 1;
    }
#ifndef __APPLE__
    if (!cpu && strcmp(where, "gpu") == 0 && tiles == 0) {
        printf("FAIL: no exact verify tile ran on the device: the rows came from "
               "another path\nspec-identity: FAILED\n");
        return 1;
    }
#endif
    printf("spec-identity: ok (%s, %d rows byte-identical to solo steps, %lu exact tiles)\n",
           where, checked, tiles);
    return 0;
}
