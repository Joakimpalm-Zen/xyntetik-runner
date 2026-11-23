// R4.48.6: the GGUF loader (src/inkling2_gguf.c) against the test-side provider on the publisher's safetensors
// (tests/inkling2_provider.c, the identity-tested reference loader order), tensor by tensor, bit for bit.
//   test_inkling2_gguf FILE.gguf CKPT.safetensors MAP SHAPES
// 1. the shape from the GGUF's metadata equals the one SHAPES gives, field for field;
// 2. every map entry the provider serves: the GGUF loader serves the same name with the same dims, and the same
//    float32 bits; every expert of every stack (fill_expert) the same bits;
// 3. CONTROLS, each must make step 2 fail on the real files: the factor dropped, the static convolution's taps not
//    reversed, the NVFP4 nibbles swapped. A control that passes means the test is blind: FAIL.
// The absolute anchor is outside Runner: the provider's order is the reference loader's (its own identity test
// against transformers), and the GGUF's bytes are the publisher's (R4.48.5 gate (a)).
#include "inkling2.h"
#include "inkling2_gguf.h"
#include "inkling2_provider.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int shapes_load(ink2_shape *s, const char *path);   // below

typedef struct { long tensors, tensor_diff_elems, tensors_differ, experts, expert_diff_elems, experts_differ, missing; char first[512]; } tally;

static void note(tally *t, const char *fmt, const char *a, long long b, double x, double y) {
    if (!t->first[0]) snprintf(t->first, sizeof t->first, fmt, a, b, x, y);
}

static void compare(ink2g *g, ink2p *p, const ink2_shape *s, tally *t, int experts_too) {
    memset(t, 0, sizeof *t);
    const int n = ink2p_count(p);
    float *a = NULL, *b = NULL;
    size_t cap = 0;
    for (int i = 0; i < n; i++) {
        int nd;
        int64_t dims[4];
        const char *ops, *dt;
        const char *name = ink2p_name(p, i, &nd, dims, &ops, &dt);
        if (ink2p_is_stack(p, i)) {
            if (!experts_too) continue;
            for (int e = 0; e < s->n_expert; e++) {
                int64_t r1, c1, r2, c2;
                if (ink2p_fill_expert(p, name, e, NULL, &r1, &c1) || ink2g_fill_expert(g, name, e, NULL, &r2, &c2) || r1 != r2 || c1 != c2) {
                    t->missing++; if (!t->first[0]) snprintf(t->first, sizeof t->first, "%s expert %d: no counterpart or a different shape", name, e); continue;
                }
                const size_t m = (size_t)(r1 * c1);
                if (m > cap) { cap = m; a = realloc(a, cap * 4); b = realloc(b, cap * 4); }
                ink2p_fill_expert(p, name, e, a, &r1, &c1);
                ink2g_fill_expert(g, name, e, b, &r2, &c2);
                long d = 0;
                for (size_t k = 0; k < m; k++)
                    if (memcmp(&a[k], &b[k], 4)) { if (!d) { char nm[256]; snprintf(nm, sizeof nm, "%s e%d", name, e); note(t, "%s element %lld: safetensors %.9g, gguf %.9g", nm, (long long)k, a[k], b[k]); } d++; }
                t->experts++;
                t->expert_diff_elems += d;
                t->experts_differ += d > 0;
            }
            continue;
        }
        const ink2_tensor *x = ink2p_get(p, name), *y = ink2g_get(g, name);
        if (!x || !y) { t->missing++; if (!t->first[0]) snprintf(t->first, sizeof t->first, "%s: served by %s only", name, x ? "the safetensors provider" : "the GGUF loader"); continue; }
        int same_dims = x->n_dims == y->n_dims;
        for (int k = 0; k < 4; k++) same_dims &= x->ne[k] == y->ne[k];
        if (!same_dims) { t->missing++; if (!t->first[0]) snprintf(t->first, sizeof t->first, "%s: dims differ", name); continue; }
        const size_t m = (size_t)(x->ne[0] * x->ne[1] * x->ne[2] * x->ne[3]);
        long d = 0;
        for (size_t k = 0; k < m; k++)
            if (memcmp(&x->data[k], &y->data[k], 4)) { if (!d) note(t, "%s element %lld: safetensors %.9g, gguf %.9g", name, (long long)k, x->data[k], y->data[k]); d++; }
        t->tensors++;
        t->tensor_diff_elems += d;
        t->tensors_differ += d > 0;
    }
    free(a);
    free(b);
}

static void report(const char *tag, const tally *t) {
    printf("%s: %ld tensors (%ld differ, %ld elements), %ld experts (%ld differ, %ld elements), %ld missing or misshapen%s%s\n",
           tag, t->tensors, t->tensors_differ, t->tensor_diff_elems, t->experts, t->experts_differ, t->expert_diff_elems, t->missing,
           t->first[0] ? "; first: " : "", t->first);
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s FILE.gguf CKPT.safetensors MAP SHAPES\n", argv[0]); return 2; }
    char why[512];
    ink2_shape sg, sr;
    ink2g *g = ink2g_open(argv[1], &sg, why, sizeof why);
    if (!g) { printf("FAIL the GGUF is refused: %s\n", why); return 1; }
    if (shapes_load(&sr, argv[4])) { printf("FAIL cannot read %s\n", argv[4]); return 1; }
    if (memcmp(&sg, &sr, sizeof sg)) { printf("FAIL the shape from the GGUF's metadata differs from %s's\n", argv[4]); return 1; }
    printf("ok   the shape from the GGUF's metadata equals the reference's, field for field (%d applications)\n", sg.n_app);
    ink2p *p = ink2p_open(argv[2], argv[3], (size_t)8 << 30, (size_t)256 << 20, why, sizeof why);
    if (!p) { printf("FAIL provider: %s\n", why); return 1; }
    int fails = 0;
    tally t;
    compare(g, p, &sg, &t, 1);
    report("identity", &t);
    if (t.tensors_differ || t.experts_differ || t.missing || !t.tensors || !t.experts) { printf("FAIL the GGUF does not serve the provider's bits\n"); fails++; }
    else printf("ok   every tensor and every expert bit-equal\n");
    static const struct { int flag; const char *what; int experts; } C[] = {
        { INK2G_CTRL_NO_FACTOR, "NVFP4 experts without the factor", 1 },
        { INK2G_CTRL_NO_FLIP, "static convolution not reversed", 0 },
        { INK2G_CTRL_SWAP_NIBBLE, "NVFP4 nibbles swapped", 1 },
    };
    for (size_t i = 0; i < sizeof C / sizeof C[0]; i++) {
        ink2g_set_control(g, C[i].flag);
        compare(g, p, &sg, &t, C[i].experts);
        char tag[96];
        snprintf(tag, sizeof tag, "control (%s)", C[i].what);
        report(tag, &t);
        if (!t.tensors_differ && !t.experts_differ) { printf("FAIL control passed: the test is blind to %s\n", C[i].what); fails++; }
        else printf("ok   the control fails, as it must\n");
    }
    ink2g_set_control(g, 0);
    ink2p_close(p);
    ink2g_close(g);
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}

// the config lines of SHAPES, as tests/inkling2_read.c reads them
static int shapes_load(ink2_shape *s, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    memset(s, 0, sizeof *s);
    char line[1 << 14];
    int have_l = 0;
    while (fgets(line, sizeof line, f)) {
        char k[128];
        double v;
        if (line[0] == 'I' || line[0] == 'F') {
            if (sscanf(line + 2, "%127s %lf", k, &v) != 2) continue;
#define SI(name, field) if (!strcmp(k, name)) s->field = (int)v;
#define SF(name, field) if (!strcmp(k, name)) s->field = (float)v;
            SI("num_hidden_layers", n_block) SI("n_layers_per_attn_res_block", attn_res_block)
            SI("loop_layer_index_start", loop_start) SI("loop_layer_index_end", loop_end)
            SI("loop_num_passes", loop_passes) SI("loop_gate_rank", loop_gate_rank)
            SI("hidden_size", hidden) SI("vocab_size", vocab) SI("unpadded_vocab_size", vocab_used)
            SI("num_attention_heads", n_head) SI("num_key_value_heads", n_kv) SI("head_dim", head_dim)
            SI("swa_num_attention_heads", swa_n_head) SI("swa_num_key_value_heads", swa_n_kv)
            SI("swa_head_dim", swa_head_dim) SI("sliding_window_size", window) SI("d_rel", d_rel)
            SI("rel_extent", rel_extent) SI("conv_kernel_size", conv_k) SI("n_routed_experts", n_expert)
            SI("num_experts_per_tok", n_used) SI("moe_intermediate_size", ff) SI("log_scaling_n_floor", log_floor)
            SF("log_scaling_alpha", log_alpha) SF("rms_norm_eps", eps) SF("route_scale", route_scale)
            SF("mlp_gate_softcap", gate_cap) SF("mlp_up_softcap", up_cap) SF("final_logit_softcapping", final_cap)
            SF("qk_l2_clip_threshold", qk_clip) SF("rel_l2_clip_threshold", rel_clip)
#undef SI
#undef SF
        } else if (line[0] == 'L') {
            char *q = line + 1;
            for (int b = 0; b < INK2_MAX_BLOCK; b++) {
                char *end = NULL;
                long x = strtol(q, &end, 10);
                if (end == q) break;
                s->block_global[b] = (unsigned char)x;
                q = end;
                have_l = b + 1;
            }
        }
    }
    fclose(f);
    if (have_l != s->n_block) return -1;
    char why[256];
    return ink2_shape_derive(s, why, sizeof why);
}
