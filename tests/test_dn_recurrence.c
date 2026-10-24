// The CPU trainer's Gated DeltaNet recurrence (model.c dn_fw_worker: the
// recompute's states, checkpoints and readouts; dn_bw_worker: the reverse
// sweep through time) against the double-precision reference in
// tests/dn_ref.h: y within 2^-14 of the reference, relative to the largest
// |y|, and dq, dk, dv, dbeta and ddecay within 2^-12, at the fixtures' head
// width (8), the models' (128) and between; and the sweep split over the
// heads in two calls (two pool workers) writes the one call's bytes.
//
// The finite-difference gradient gates (tests/test_lora_grad.c) resolve
// about a tenth of a percent at fixture scale. Dropping one term of dk from
// the sweep moved the derivative they check by about that much and passed
// them, on the CPU and on the device (R8.21); this test fails it.
//
// model.c is compiled into this test (tests/test_mvt_chains.c's pattern) to
// reach the workers.
#include <stdio.h>
#include "../src/model.c"
#include "dn_ref.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); g_fail = 1; } } while (0)

static void close_to(const char *what, int hv, const float *got,
                     const double *ref, size_t n, double tol) {
    size_t at;
    double mx, worst = dn_ref_err(got, ref, n, &at, &mx);
    CHECK(worst <= tol * mx,   // an all-zero reference wants exact zeros
          "hv %d: %s off by %.3g of max %.3g at "
          "[%zu] (cpu %.9g reference %.9g)", hv, what, worst, mx, at,
          (double)got[at], ref[at]);
}

static void run(int nh, int ng, int hv, int T, uint64_t seed) {
    dn_shape s = dn_shape_of(nh, ng, hv, T);
    static model_t m;
    memset(&m, 0, sizeof m);
    m.ssm.state = hv; m.ssm.groups = ng; m.ssm.v_heads = nh;
    m.ssm.inner = nh * hv; m.ssm.conv_kernel = 4; m.rms_eps = 1e-6f;
    layer_t ly;
    memset(&ly, 0, sizeof ly);
    float *nw = malloc(sizeof(float) * (size_t)hv);
    for (int i = 0; i < hv; i++) nw[i] = 1.0f;
    ly.ssm_norm_w = nw;
    lbw_dn d;
    if (!lbw_dn_init(&m, T, &d)) {
        CHECK(0, "hv %d: lbw_dn_init failed", hv);
        free(nw);
        return;
    }
    dn_ref_fill(&s, seed, d.cv, d.beta, d.decay, d.dy);
    for (size_t i = 0; i < (size_t)T * s.inner; i++) d.z[i] = 0.5f;
    dn_ref_out r;
    dn_ref_run(&s, d.cv, d.beta, d.decay, d.dy, &r);

    dn_job J = { &m, &ly, &d, T, true };
    dn_fw_worker(&J, 0, nh);
    CHECK(J.ok, "hv %d: the recompute failed", hv);
    close_to("y", hv, d.y, r.y, (size_t)T * s.inner, 1.0 / 16384);
    dn_bw_worker(&J, 0, nh);
    CHECK(J.ok, "hv %d: the sweep failed", hv);
    size_t nq = (size_t)nh * T * s.sk, ny = (size_t)T * s.inner, ng_ = (size_t)T * nh;
    close_to("dq", hv, d.dqh, r.dq, nq, 1.0 / 4096);
    close_to("dk", hv, d.dkh, r.dk, nq, 1.0 / 4096);
    close_to("dv", hv, d.dv, r.dv, ny, 1.0 / 4096);
    close_to("dbeta", hv, d.dbeta, r.dbeta, ng_, 1.0 / 4096);
    close_to("ddecay", hv, d.ddecay, r.ddecay, ng_, 1.0 / 4096);

    // the heads in two calls, as two workers would take them
    float *keep[5] = { d.dqh, d.dkh, d.dv, d.dbeta, d.ddecay };
    size_t kn[5] = { nq, nq, ny, ng_, ng_ };
    float *copy[5];
    for (int i = 0; i < 5; i++) {
        copy[i] = malloc(sizeof(float) * kn[i]);
        memcpy(copy[i], keep[i], sizeof(float) * kn[i]);
        memset(keep[i], 0, sizeof(float) * kn[i]);
    }
    dn_bw_worker(&J, 0, nh / 2);
    dn_bw_worker(&J, nh / 2, nh);
    for (int i = 0; i < 5; i++) {
        CHECK(!memcmp(copy[i], keep[i], sizeof(float) * kn[i]),
              "hv %d: output %d depends on how the heads were split", hv, i);
        free(copy[i]);
    }
    printf("dn recurrence hv %3d nh %d ng %d T %d: checked\n", hv, nh, ng, T);
    dn_ref_free(&r);
    lbw_dn_free(&d);
    free(nw);
}

int main(void) {
    run(4, 2, 128, 77, 0x2545F4914F6CDD1Dull);
    run(3, 3, 64, 40, 0x9E3779B97F4A7C15ull);
    run(4, 2, 40, 33, 0xD1B54A32D192ED03ull);
    run(2, 1, 32, 9, 0x8CB92BA72F3D8DD7ull);
    run(4, 2, 8, 26, 0xC2B2AE3D27D4EB4Full);
    run(2, 2, 16, 1, 0x165667B19E3779F9ull);
    printf("%s\n", g_fail ? "FAIL" : "dn recurrence: the CPU recompute and sweep "
           "match the double-precision reference");
    return g_fail;
}
