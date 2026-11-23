// R4.48.6: the read harness for src/inkling2.c (test-side): token ids in, the FULL logits rows out, one sequence.
//   inkling2_read real CKPT.safetensors MAP SHAPES IDS OUT N
//   inkling2_read gguf FILE.gguf IDS OUT N          (src/inkling2_gguf.c: the shape from the file's metadata; with
//                                                   INK2_SHAPES=SHAPES set, that shape is checked against SHAPES's)
// The weights come from the test-side provider (tests/inkling2_provider.c) on the publisher's checkpoint, behind the
// decoded-expert cache (tests/inkling2_ecache.c) when INK2_ECACHE_BYTES is set. The forward runs as chunks of
// INK2_READ_CHUNK positions (default 500) over one cache (ink2_forward_chunk, p0 = 0, chunk, 2 x chunk, ...).
// INK2_READ_PREFILL=P: one chunk of P positions, then ONE position per call (the decode route), each step timed.
// INK2_THREADS sets the engine's threads (the outputs do not depend on it: threads split outputs, not sums).
// INK2P_TENSOR_BYTES and INK2P_EXPERT_BYTES (suffixes K, M, G as powers of 1024) size the provider's caches.
// OUT.logits.f32: raw little-endian float32 [N][vocab_used], row-major; OUT.json: the run's numbers.
#include "inkling2.h"
#include "inkling2_ecache.h"
#include "inkling2_gguf.h"
#include "inkling2_provider.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static size_t env_bytes(const char *k, size_t dflt) {
    const char *v = getenv(k);
    if (!v || !*v) return dflt;
    char *end = NULL;
    double x = strtod(v, &end);
    size_t m = 1;
    if (end && (*end == 'K' || *end == 'k')) m = 1024;
    else if (end && (*end == 'M' || *end == 'm')) m = 1024 * 1024;
    else if (end && (*end == 'G' || *end == 'g')) m = 1024ull * 1024 * 1024;
    return (size_t)(x * (double)m);
}

// the config lines of SHAPES (I name value, F name value, L the layer types as 0/1 = sliding/full)
static int shapes_load(ink2_shape *s, const char *path, char *why, size_t wn) {
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(why, wn, "cannot open %s", path); return -1; }
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
            char *p = line + 1;
            for (int b = 0; b < INK2_MAX_BLOCK; b++) {
                char *end = NULL;
                long x = strtol(p, &end, 10);
                if (end == p) break;
                s->block_global[b] = (unsigned char)x;
                p = end;
                have_l = b + 1;
            }
        }
    }
    fclose(f);
    if (have_l != s->n_block) { snprintf(why, wn, "layer types: %d of %d", have_l, s->n_block); return -1; }
    return ink2_shape_derive(s, why, wn);
}

static int *ids_load(const char *path, int *n) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    int cap = 4096, k = 0;
    int *v = malloc(sizeof(int) * (size_t)cap);
    long x;
    while (fscanf(f, "%ld", &x) == 1) {
        if (k == cap) { cap *= 2; v = realloc(v, sizeof(int) * (size_t)cap); }
        v[k++] = (int)x;
    }
    fclose(f);
    *n = k;
    return v;
}

int main(int argc, char **argv) {
    const int real = argc == 8 && !strcmp(argv[1], "real"), gg = argc == 6 && !strcmp(argv[1], "gguf");
    if (!real && !gg) {
        fprintf(stderr, "usage: %s real CKPT.safetensors MAP SHAPES IDS OUT N\n       %s gguf FILE.gguf IDS OUT N\n", argv[0], argv[0]);
        return 2;
    }
    const char *idsf = real ? argv[5] : argv[3], *out = real ? argv[6] : argv[4];
    const int N = atoi(real ? argv[7] : argv[5]);
    char why[512] = {0};
    ink2_shape s;
    const char *th = getenv("INK2_THREADS");
    if (th) ink2_set_threads(atoi(th));
    const int chunk = getenv("INK2_READ_CHUNK") ? atoi(getenv("INK2_READ_CHUNK")) : 500;
    if (chunk < 1) { fprintf(stderr, "INK2_READ_CHUNK must be >= 1\n"); return 2; }

    double t0 = now_s();
    ink2p *p = NULL;
    ink2g *g = NULL;
    ink2_weights pw, w;
    ink2_ec_fill_fn fill;
    void *fctx;
    if (real) {
        if (shapes_load(&s, argv[4], why, sizeof why)) { fprintf(stderr, "shapes: %s\n", why); return 1; }
        p = ink2p_open(argv[2], argv[3], env_bytes("INK2P_TENSOR_BYTES", (size_t)4 << 30),
                       env_bytes("INK2P_EXPERT_BYTES", (size_t)256 << 20), why, sizeof why);
        if (!p) { fprintf(stderr, "provider: %s\n", why); return 1; }
        ink2p_weights(p, &pw);
        fill = ink2p_fill_expert; fctx = p;
    } else {
        g = ink2g_open(argv[2], &s, why, sizeof why);
        if (!g) { fprintf(stderr, "REFUSED: %s\n", why); return 4; }
        const char *sh = getenv("INK2_SHAPES");
        if (sh) {
            ink2_shape r;
            if (shapes_load(&r, sh, why, sizeof why)) { fprintf(stderr, "shapes: %s\n", why); return 1; }
            if (memcmp(&r, &s, sizeof s)) { fprintf(stderr, "FAIL the shape from the file's metadata differs from %s\n", sh); return 1; }
            fprintf(stderr, "the shape from the file's metadata equals %s's, field for field\n", sh);
        }
        ink2g_weights(g, &pw);
        fill = ink2g_fill_expert; fctx = g;
        fprintf(stderr, "experts by layer:");
        for (int b = 0; b < s.n_block; b++) fprintf(stderr, " %c", ink2g_expert_kind(g, b));
        fprintf(stderr, "\n");
    }
    int n_ids = 0;
    int *ids = ids_load(idsf, &n_ids);
    if (!ids || n_ids < N || N < 1) { fprintf(stderr, "ids: %d in %s, need %d\n", n_ids, idsf, N); return 1; }
    for (int t = 0; t < N; t++)
        if (ids[t] < 0 || ids[t] >= s.vocab_used) { fprintf(stderr, "id %d at %d out of range\n", ids[t], t); return 1; }
    ink2_ecache *ec = NULL;
    ink2_ec_cfg cfg = ink2_ec_cfg_env();
    if (cfg.enabled) {
        ec = ink2_ec_new(&pw, fill, fctx, cfg);
        ink2_ec_weights(ec, &w);
    } else {
        w = pw;
    }
    const double t_open = now_s() - t0;

    ink2_cache c;
    const int prefill = getenv("INK2_READ_PREFILL") ? atoi(getenv("INK2_READ_PREFILL")) : 0;
    // INK2_READ_WINDOW=1: the sliding applications' K/V held for the window only (ink2_cache_init_window)
    const int windowed = getenv("INK2_READ_WINDOW") ? atoi(getenv("INK2_READ_WINDOW")) : 0;
    const int widest = prefill > chunk ? prefill : chunk;
    if (windowed ? ink2_cache_init_window(&c, &s, N, widest) : ink2_cache_init(&c, &s, N)) {
        fprintf(stderr, "cache init failed\n");
        return 1;
    }
    if (windowed == 2) c.test_drop_key = 1;   // INK2_READ_WINDOW=2: the cap's control (must change the rows)
    fprintf(stderr, "cache: %s, %.1f MiB\n", windowed == 2 ? "CONTROL: windowed, first admitted key dropped" :
            windowed ? "sliding K/V for the window only" : "full size",
            ink2_cache_bytes(&c, &s) / 1048576.0);
    const size_t V = (size_t)s.vocab_used;
    float *rows = malloc(sizeof(float) * V * (size_t)widest);   // the widest call: a prefill wider than the chunk too
    char path[4096];
    snprintf(path, sizeof path, "%s.logits.f32", out);
    FILE *fo = fopen(path, "wb");
    if (!rows || !fo) { fprintf(stderr, "cannot allocate or open %s\n", path); return 1; }
    long long nonfinite = 0;
    double t1 = now_s();
    double step_sum = 0, step_max = 0;
    int steps = 0;
    for (int p0 = 0; p0 < N;) {
        int T = p0 + chunk <= N ? chunk : N - p0;
        if (prefill > 0) T = p0 == 0 ? (prefill < N ? prefill : N) : 1;
        const double ts = now_s();
        ink2_status st = ink2_forward_chunk(&s, &w, &c, p0, T, ids + p0, rows, 0, NULL);
        if (st != INK2_OK) { fprintf(stderr, "forward at %d: status %d\n", p0, (int)st); return 1; }
        for (size_t i = 0; i < V * (size_t)T; i++) nonfinite += !isfinite(rows[i]);
        if (fwrite(rows, sizeof(float), V * (size_t)T, fo) != V * (size_t)T) { fprintf(stderr, "write failed\n"); return 1; }
        const double dt = now_s() - ts;
        if (prefill > 0 && p0 > 0) { step_sum += dt; if (dt > step_max) step_max = dt; steps++; }
        fprintf(stderr, "chunk %d..%d done, %.3f s (total %.1f s)\n", p0, p0 + T - 1, dt, now_s() - t1);
        p0 += T;
    }
    if (steps) fprintf(stderr, "decode: %d single-position steps, mean %.3f s, max %.3f s (%.2f tokens/s)\n", steps,
                       step_sum / steps, step_max, steps / step_sum);
    fclose(fo);
    const double t_fwd = now_s() - t1;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    // what the pool really did (not the count asked for): loops split over more than one thread, and the last one's
    long split_loops = 0;
    int last_pool = 1;
    ink2_par_stats(&split_loops, &last_pool);
    snprintf(path, sizeof path, "%s.json", out);
    FILE *fj = fopen(path, "w");
    if (fj) {
        fprintf(fj, "{\"source\": \"%s\", \"T\": %d, \"V\": %zu, \"chunk\": %d, \"threads\": %d, \"open_s\": %.3f, "
                    "\"forward_s\": %.3f, \"s_per_token\": %.4f, \"nonfinite_logits\": %lld, \"peak_rss_gb\": %.3f, "
                    "\"ecache\": %s, \"split_loops\": %ld, \"last_pool\": %d}\n",
                real ? "real" : "gguf", N, V, chunk, ink2_get_threads(), t_open, t_fwd, t_fwd / N, nonfinite, ru.ru_maxrss / 1e6,
                ec ? "true" : "false", split_loops, last_pool);
        fclose(fj);
    }
    fprintf(stderr, "forward %.1f s (%.3f s a token), non-finite %lld, peak RSS %.2f GB\n", t_fwd, t_fwd / N, nonfinite,
            ru.ru_maxrss / 1e6);
    if (ec) ink2_ec_print(ec, "expert cache", stderr);
    ink2_cache_free(&c);
    if (ec) ink2_ec_free(ec);
    if (p) ink2p_close(p);
    if (g) ink2g_close(g);
    free(rows);
    free(ids);
    return nonfinite ? 3 : 0;
}
