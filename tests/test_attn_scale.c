// Gemma 3's attention scale is `query_pre_attn_scalar ** -0.5`
// (transformers' Gemma3Attention.scaling), and query_pre_attn_scalar is NOT
// always the head width: it is 256 = head_dim on the 270M/1B/4B/12B models
// and 168 = hidden_size / num_attention_heads on the 27B, whose heads are 128
// wide. The GGUF does not carry the scalar; llama.cpp recovers it from the
// size (62 blocks is the 27B, gemma_pytorch config.py). The runner used
// 1/sqrt(head_dim) for every size, so the 27B scored attention 1.146x too
// sharp at every layer.
//
// Anchor: the two expected scales are the published configs' rule applied
// to each fixture's geometry by hand -- n_embd 64 over 4 heads is 16, the
// head width is 8 -- not values read back from the loader.
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

static bool load(model_t *m, const char *path) {
    model_params p; memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF; p.n_threads = 1; p.n_ctx = 64; p.n_batch = 4;
    memset(m, 0, sizeof *m);
    if (!model_load(m, path, &p)) { fprintf(stderr, "cannot load %s\n", path); return false; }
    return true;
}

static void check_scale(const char *path, double want, const char *what) {
    model_t m;
    if (!load(&m, path)) { g_fail = 1; return; }
    char msg[160];
    for (int l = 0; l < m.n_layer; l++) {
        float got = model_attn_scale(&m, l);
        if (fabs((double)got - want) > 1e-6) {
            snprintf(msg, sizeof msg, "%s: layer %d scale %.6f, reference %.6f",
                     what, l, (double)got, want);
            ck(0, msg);
            model_free(&m);
            return;
        }
    }
    snprintf(msg, sizeof msg, "%s: every layer scales by %.6f", what, want);
    ck(1, msg);
    model_free(&m);
}

int main(int argc, char **argv) {
    const char *g27 = argc > 1 ? argv[1] : "test-gemma3-62.gguf";
    const char *g1  = argc > 2 ? argv[2] : "test-gemma3-26.gguf";
    // 62 blocks: the 27B's rule, 1 / sqrt(n_embd / n_head) = 1 / sqrt(16)
    check_scale(g27, 0.25, "gemma3 27B shape");
    // 26 blocks: the 1B's rule, 1 / sqrt(head_dim) = 1 / sqrt(8)
    check_scale(g1, 0.35355339059327373, "gemma3 1B shape");
    if (!g_fail) puts("attention scale tests ok");
    return g_fail;
}
