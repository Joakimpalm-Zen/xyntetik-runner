// R4.48.8.5 gate (a): Inkling 2 Mini's q8 K/V cache, on the tiny inkling2_loop file (tests/inkling2_tiny.py, window 8).
// The cache stores each K and V row as q8_0 (quants.c q8_quant_row) and the cores read value e of a row as
// (float)q * d, exact in float32. So a q8 cache must give, bit for bit, the logits of a float32 cache whose rows hold
// those same round-tripped values (TEST-ONLY RUNNER_INKLING2_TEST_KV_RT=1: each stored row quantized and decoded back
// to float32), on both CPU routes. The comparison crosses the sliding buffers' slide (a 40-position sequence over an
// 8-position window, in calls of 5, and the one-position decode path at -b 1), and the q8 rows must differ from the
// plain float32 cache's, or the format did not engage.
// usage: test_inkling2_kvq8 TINY_F32.gguf
#include "model.h"
#include "gpu.h"
#include "inkling2.h"
#include "inkling2_model.h"
#include "compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_TOK = 40 };
static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

// kv: 0 the float32 cache, 1 --kv q8, 2 the float32 cache holding each row's q8 round trip (TEST-ONLY)
static model_t *load(const char *path, const char *route, int kv, int n_batch) {
    plat_setenv("RUNNER_INKLING2_ROUTE", route);
    plat_setenv("RUNNER_INKLING2_TEST_KV_RT", kv == 2 ? "1" : "");
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = 2;
    p.n_ctx = 64;
    p.n_batch = n_batch;
    p.kv_q8 = kv == 1;
    model_t *m = calloc(1, sizeof *m);
    if (!m || !model_load(m, path, &p)) { free(m); return NULL; }
    return m;
}
// every row of the sequence, in calls of at most -b positions, as the engine's prefill makes them
static float *rows(model_t *m, const int32_t *tok) {
    float *r = malloc(sizeof(float) * (size_t)N_TOK * (size_t)m->n_vocab);
    for (int pos = 0; r && pos < N_TOK;) {
        const int n = N_TOK - pos < m->n_batch ? N_TOK - pos : m->n_batch;
        if (!model_inkling2_forward(m, tok + pos, n, pos, true, true)) { free(r); return NULL; }
        for (int b = 0; b < n; b++)
            memcpy(r + (size_t)(pos + b) * (size_t)m->n_vocab, model_inkling2_row(m, b), sizeof(float) * (size_t)m->n_vocab);
        pos += n;
    }
    return r;
}
static void drop(model_t *m) { if (m) { model_free(m); free(m); } }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s TINY_F32.gguf\n", argv[0]); return 2; }
    int32_t tok[N_TOK];
    uint32_t r = 2463534242u;
    for (int t = 0; t < N_TOK; t++) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; tok[t] = (int32_t)(r % 256u); }
    static const char *const route[2] = { "reference", "fast" };
    static const int batch[2] = { 5, 1 };
    for (int ri = 0; ri < 2; ri++)
        for (int bi = 0; bi < 2; bi++) {
            float *out[3] = { NULL, NULL, NULL };
            size_t nv = 0;
            for (int kv = 0; kv < 3; kv++) {
                model_t *m = load(argv[1], route[ri], kv, batch[bi]);
                if (!m) { printf("FAIL: %s -b %d kv %d: the load failed\n", route[ri], batch[bi], kv); fails++; continue; }
                nv = (size_t)m->n_vocab;
                out[kv] = rows(m, tok);
                if (!out[kv]) { printf("FAIL: %s -b %d kv %d: a forward failed\n", route[ri], batch[bi], kv); fails++; }
                drop(m);
            }
            if (out[0] && out[1] && out[2]) {
                const size_t n = sizeof(float) * (size_t)N_TOK * nv;
                CHECK(memcmp(out[1], out[2], n) == 0,
                      "%s route -b %d: the q8 cache's logits are the float32 cache's over the round-tripped rows, bit for bit",
                      route[ri], batch[bi]);
                CHECK(memcmp(out[1], out[0], n) != 0, "%s route -b %d: the q8 cache engaged (its logits are not the "
                      "float32 cache's)", route[ri], batch[bi]);
                CHECK(memcmp(out[2], out[0], n) != 0, "%s route -b %d: the round trip engaged", route[ri], batch[bi]);
            }
            for (int kv = 0; kv < 3; kv++) free(out[kv]);
        }
    plat_setenv("RUNNER_INKLING2_TEST_KV_RT", "");
    if (fails) { printf("test_inkling2_kvq8: %d FAILED\n", fails); return 1; }
    printf("test_inkling2_kvq8: ok\n");
    return 0;
}
