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
    // the device bytes the CUDA load and --info fit count for one slot (model_inkling2_dev_slot_bytes), on Inkling 2
    // Mini's own shape at 64K (24 blocks, the loop 6..18 x 3: 8 full applications and 40 sliding; 8 K/V heads of 128;
    // window 1024; hidden 2304; conv_k 4), each figure worked by hand from the driver's rounding (2 MiB from 1 MiB up,
    // else 512 bytes) and the packed sliding rings (one allocation a side, 256-byte offsets)
    {
        ink2_shape rs;
        memset(&rs, 0, sizeof rs);
        rs.n_block = 24; rs.attn_res_block = 6; rs.loop_start = 6; rs.loop_end = 18; rs.loop_passes = 3;
        for (int b = 0; b < 24; b++) rs.block_global[b] = (b + 1) % 6 == 0;
        rs.hidden = 2304; rs.n_kv = rs.swa_n_kv = 8; rs.head_dim = rs.swa_head_dim = 128; rs.window = 1024; rs.conv_k = 4;
        char why[128];
        const int derived = ink2_shape_derive(&rs, why, sizeof why);
        int n_full = 0;
        for (int a = 0; derived == 0 && a < rs.n_app; a++) n_full += ink2_app_global(&rs, a);
        CHECK(derived == 0 && rs.n_app == 48 && n_full == 8, "the real shape: %d applications, %d full (want 48 and 8)",
              rs.n_app, n_full);
        static const struct { int kind, batch; uint64_t want; const char *what; } dv[] = {
            // 8 x 2 x 256 MiB full rings + 2 x 400 MiB packed (40 x 1279 rows x 4096) + 96 x 12 KiB + 4 MiB post
            { INK2_KV_F32, 256, 4719771648ull, "f32, -b 256" },
            // 8 x 2 x 68 MiB + 2 x 108 MiB packed (40 x 1279 x 1088, 256-aligned) + histories + post
            { INK2_KV_Q8, 256, 1259470848ull, "q8, -b 256" },
            { INK2_KV_Q8, 1, 1238499328ull, "q8, -b 1" },
        };
        for (size_t k = 0; derived == 0 && k < sizeof dv / sizeof *dv; k++) {
            const uint64_t got = model_inkling2_dev_slot_bytes(&rs, 65536, dv[k].batch, dv[k].kind, true, 3, true);
            CHECK(got == dv[k].want, "the device bytes of one slot at 64K, %s: %llu (want %llu)", dv[k].what,
                  (unsigned long long)got, (unsigned long long)dv[k].want);
        }
    }
    // R4.48.32: placement-fails fallback sentence with a 159-byte device name (~360 bytes) and escaped-quote
    // worst case (>520 escaped bytes) must appear untruncated in model_route_json().
    {
        model_t *m = load(argv[1], "fast", 0, 5);
        CHECK(m != NULL, "route_fallback truncation check: load tiny model");
        if (m) {
            char dev[160];
            memset(dev, 'A', 152);
            memcpy(dev + 152, "GPU-END", 8);   // 159 bytes + NUL
            char why[512];
            int wn = snprintf(why, sizeof why,
                              "inkling2_loop's weights for the device do not fit the %.2f GB free on %s "
                              "for %d slot%s (after its context, module and buffers, with 0.54 GB kept), "
                              "not with its expert stacks on the host either",
                              12.34, dev, 2, "s");
            snprintf(m->route_fallback, sizeof m->route_fallback, "%s", why);
            char jbuf[6144], needle[600];
            model_route_json(m, jbuf, (int)sizeof jbuf);
            snprintf(needle, sizeof needle, "\"fallback\":\"%s\"", why);
            CHECK(wn > 300 && strstr(jbuf, needle) != NULL,
                  "route_json preserves full %d-byte placement-fails fallback with 159-byte device name", wn);

            // Escaping can double the length: 340 quotes -> 680 escaped bytes (exceeds old why[520]).
            char qwhy[341], qesc[700];
            memset(qwhy, '"', 340);
            qwhy[340] = 0;
            snprintf(m->route_fallback, sizeof m->route_fallback, "%s", qwhy);
            model_route_json(m, jbuf, (int)sizeof jbuf);
            size_t p = 0;
            p += (size_t)snprintf(qesc + p, sizeof qesc - p, "\"fallback\":\"");
            for (int i = 0; i < 340; i++) { qesc[p++] = '\\'; qesc[p++] = '"'; }
            qesc[p++] = '"'; qesc[p] = 0;
            CHECK(strstr(jbuf, qesc) != NULL,
                  "route_json escape buffer preserves 340 escaped quotes (680 bytes) without truncation");
            drop(m);
        }
    }
    if (fails) { printf("test_inkling2_kvq8: %d FAILED\n", fails); return 1; }
    printf("test_inkling2_kvq8: ok\n");
    return 0;
}
