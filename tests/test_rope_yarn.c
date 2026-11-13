// YaRN's NTK-by-parts ramp against the publishers' own definitions.
//
// The ramp blends interpolated and extrapolated frequencies between two
// "correction dimensions". transformers rounds them outward (floor/ceil) by
// default, and llama.cpp always does; OpenAI's gpt-oss reference does not
// (gpt_oss/torch/model.py, RotaryEmbedding: `low`/`high` used as computed,
// and the model's config.json says so with `"truncate": false`). gpt-oss
// runs YaRN x32 at every position, so rounding moved the rotation frequency
// of every dimension inside the ramp -- by 1.75x at dimension 17 of the real
// 64-wide heads -- and agreeing with llama.cpp could never show it, because
// llama.cpp makes the same choice.
//
// Anchor: the expected tables below were computed OUTSIDE this code, in
// double precision from the two reference formulas (OpenAI's for gpt-oss,
// transformers' `_compute_yarn_parameters` with its default truncate=True
// for llama), for each fixture's own geometry: head dim 8 (gpt-oss) and 16
// (llama), base 10000, factor 32, original context 4096, beta 32/1. The
// magnitude factor is YaRN's 0.1*ln(32) + 1 in both.
#include "model.h"
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

static bool close_rel(float got, double want) {
    return fabs((double)got - want) <= 1e-5 * fabs(want);
}

static bool load(model_t *m, const char *path) {
    model_params p; memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF; p.n_threads = 1; p.n_ctx = 64; p.n_batch = 4;
    memset(m, 0, sizeof *m);
    if (!model_load(m, path, &p)) { fprintf(stderr, "cannot load %s\n", path); return false; }
    return true;
}

static void check_table(const char *what, const float *got, const double *want,
                        int n) {
    char msg[160];
    for (int j = 0; j < n; j++) {
        snprintf(msg, sizeof msg, "%s: inv_freq[%d] = %.9g, reference %.9g",
                 what, j, (double)got[j], want[j]);
        ck(close_rel(got[j], want[j]), msg);
    }
}

int main(int argc, char **argv) {
    const char *gptoss = argc > 1 ? argv[1] : "test-moe-fixture.gptoss-yarn.gguf";
    const char *llama  = argc > 2 ? argv[2] : "test-yarn.gguf";
    const char *local  = argc > 3 ? argv[3] : "test-rope-local.gguf";
    const double mscale = 1.3465735902799727;   // 0.1 * ln(32) + 1

    // gpt-oss: OpenAI's ramp, correction range 1.3090..2.8142 unrounded.
    // Rounded to 1..3 (the old behaviour) dimension 2 read 0.00515625.
    static const double want_oss[4] = { 1.0, 0.1, 0.005552754881458853, 3.125e-05 };
    model_t m;
    if (!load(&m, gptoss)) return 1;
    ck(m.rope_dim == 8, "gpt-oss fixture rotates 8 dims");
    check_table("gpt-oss global", m.rope_inv_freq, want_oss, 4);
    // its sliding layers rope in the same regime (swa_rope_global)
    ck(m.rope_inv_freq_local != NULL, "gpt-oss has a sliding-layer table");
    if (m.rope_inv_freq_local)
        check_table("gpt-oss sliding", m.rope_inv_freq_local, want_oss, 4);
    ck(close_rel(m.rope_mscale, mscale), "gpt-oss YaRN magnitude 0.1*ln(32)+1");
    model_free(&m);

    // llama with YaRN metadata: transformers' default rounds the range out,
    // 2.618..5.628 -> 2..6. Unrounded, dims 3..5 would move.
    static const double want_llama[8] = {
        1.0, 0.31622776601683794, 0.1, 0.0239641353934635,
        0.00515625, 0.0008646852977022911, 3.125e-05, 9.882117688026186e-06 };
    if (!load(&m, llama)) return 1;
    ck(m.rope_dim == 16, "llama fixture rotates 16 dims");
    check_table("llama", m.rope_inv_freq, want_llama, 8);
    ck(close_rel(m.rope_mscale, mscale), "llama YaRN magnitude 0.1*ln(32)+1");
    model_free(&m);

    // A file that declares the sliding layers' base under both spellings:
    // rope.local.freq_base 500 wins over rope.freq_base_swa 20000. The two
    // reads used to share one key buffer inside one argument list, so the
    // order the compiler evaluates arguments in chose the key: clang's left
    // to right read rope.freq_base_swa twice. Anchor: base^(-2j/d) in double.
    if (!load(&m, local)) return 1;
    ck(m.swa_window > 0 && m.rope_inv_freq_local != NULL, "the local fixture slides");
    if (m.rope_inv_freq_local) {
        int d = m.rope_dim_local;
        double want_local[64];
        for (int j = 0; j < d / 2 && j < 64; j++) want_local[j] = pow(500.0, -2.0 * j / d);
        check_table("rope.local.freq_base over rope.freq_base_swa", m.rope_inv_freq_local,
                    want_local, d / 2 < 64 ? d / 2 : 64);
    }
    model_free(&m);

    if (!g_fail) puts("rope yarn tests ok");
    return g_fail;
}
