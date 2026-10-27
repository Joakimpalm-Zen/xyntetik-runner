// The CPU trainer's attention backward (model.c attn_bw_run: the score and
// softmax adjoint into dq, dk, dv) against the double-precision reference in
// tests/attn_ref.h: each within 2^-12 of its largest magnitude, at GQA, a
// sliding window, windows past one block of ATB positions and head widths
// 64 and 256; and the run on a pool of three threads writes the serial
// run's bytes.
//
// The finite-difference gradient gates (tests/test_lora_grad.c) resolve
// about a tenth of a percent at fixture scale: scaling the softmax adjoint's
// sum_pd by 0.999 passed them (R8.20, the attention backward's device
// port); this test fails it.
//
// model.c is compiled into this test (tests/test_dn_recurrence.c's pattern)
// to reach attn_bw_run.
#include <stdio.h>
#include "../src/model.c"
#include "attn_ref.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static void run(tpool *tp, int T, int H, int kv_mul, int hd, int swa, uint64_t seed) {
    ab_shape s = ab_shape_of(T, H, kv_mul, hd, swa);
    size_t row = (size_t)s.kv_dim * 2, nq = (size_t)T * s.q_dim, nk = (size_t)T * s.kv_dim;
    uint8_t *kc = malloc(row * T), *vc = malloc(row * T);
    float *q = malloc(sizeof(float) * nq), *dao = malloc(sizeof(float) * nq);
    float *o[2][3];
    size_t on[3] = { nq, nk, nk };
    for (int r = 0; r < 2; r++)
        for (int i = 0; i < 3; i++) o[r][i] = calloc(on[i], sizeof(float));
    double *ref[3];
    for (int i = 0; i < 3; i++) ref[i] = calloc(on[i], sizeof(double));
    ab_ref_fill(&s, seed, kc, vc, q, dao);
    float scale = 1.0f / sqrtf((float)hd);
    ab_ref_run(&s, scale, kc, vc, q, dao, ref[0], ref[1], ref[2]);
    bool ok = true;
    for (int r = 0; r < 2; r++) {
        attn_bw_job aj = { kc, vc, q, dao, o[r][0], o[r][1], o[r][2], T, H, kv_mul,
                           hd, s.q_dim, s.kv_dim, row, row, scale, swa };
        ok = attn_bw_run(r ? tp : NULL, &aj) && ok;
    }
    CHECK(ok, "hd %d: attn_bw_run failed", hd);
    static const char *const NAMES[3] = { "dq", "dk", "dv" };
    for (int i = 0; ok && i < 3; i++) {
        double rel = ab_ref_rel(o[0][i], ref[i], on[i]);
        CHECK(rel <= 1.0 / 4096, "T %d hd %d swa %d: %s off by %.3g of its largest "
              "magnitude", T, hd, swa, NAMES[i], rel);
        CHECK(!memcmp(o[0][i], o[1][i], sizeof(float) * on[i]),
              "T %d hd %d: %s on three threads differs from the serial run", T, hd,
              NAMES[i]);
    }
    printf("attn backward T %3d H %d kv_mul %d hd %3d swa %2d: checked\n", T, H,
           kv_mul, hd, swa);
    for (int r = 0; r < 2; r++) for (int i = 0; i < 3; i++) free(o[r][i]);
    for (int i = 0; i < 3; i++) free(ref[i]);
    free(kc); free(vc); free(q); free(dao);
}

int main(void) {
    f16_init();
    tpool *tp = tpool_create(3);
    run(tp, 77, 4, 2, 64, 0, 0x2545F4914F6CDD1Dull);
    run(tp, 150, 8, 4, 64, 0, 0x9E3779B97F4A7C15ull);
    run(tp, 90, 4, 1, 256, 24, 0xD1B54A32D192ED03ull);
    run(tp, 1, 2, 2, 32, 0, 0x8CB92BA72F3D8DD7ull);
    tpool_destroy(tp);
    printf("%s\n", g_fail ? "FAIL" : "attn backward: the CPU matches the "
           "double-precision reference");
    return g_fail;
}
