// R4.48.6: Inkling 2 Mini on Runner's normal path (see inkling2_model.h).
#include "inkling2_model.h"
#include "inkling2.h"
#include "inkling2_ecache.h"
#include "inkling2_gguf.h"
#include "gpu.h"
#include "compat.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ink2g *g;
    ink2_shape s;
    ink2_weights inner, w;
    ink2_ecache *ec;
    ink2_cache c;
    bool cache_ok;
    float *rows;        // [n_batch][n_vocab]: the last chunk's rows (the keep path reads row b of the last forward)
    int rows_cap;       // rows the buffer holds
    int kept_from;      // the first position the buffer's row 0 is, for model_inkling2_row
    int kept_n;         // rows kept by the last keep_all forward
    size_t dense_bytes;
    size_t ec_budget;
} ink2_rt;

static ink2_rt *rt_of(model_t *m) { return (ink2_rt *)m->ink2; }

bool model_inkling2_bind(model_t *m, const model_params *p) {
    (void)p;
    char why[512] = { 0 };
    ink2_rt *r = calloc(1, sizeof *r);
    if (!r) { fprintf(stderr, "error: out of memory loading an inkling2_loop model\n"); return false; }
    r->g = ink2g_attach(&m->gf, &r->s, why, sizeof why);
    if (!r->g) {
        fprintf(stderr, "error: this inkling2_loop file is refused: %s\n", why);
        free(r);
        return false;
    }
    if (ink2g_decode_all(r->g, &r->dense_bytes)) {
        fprintf(stderr, "error: an inkling2_loop tensor could not be decoded to float32\n");
        ink2g_close(r->g);
        free(r);
        return false;
    }
    ink2g_weights(r->g, &r->inner);
    m->ink2 = r;
    snprintf(m->arch, sizeof m->arch, "inkling2_loop");
    m->ad = arch_find("inkling2_loop");
    // what the engine, the sampler and the server read from model_t; no llama-style layer is bound (n_layer 0)
    m->n_layer = 0;
    m->n_embd = r->s.hidden;
    m->n_head = r->s.n_head;
    m->n_head_kv = r->s.n_kv;
    m->head_dim = r->s.head_dim;
    m->n_ff = r->s.ff;
    m->n_vocab = r->s.vocab_used;
    m->n_ctx_train = (int)gguf_get_u32(&m->gf, "inkling2_loop.context_length", 0);
    if (m->n_ctx_train <= 0) m->n_ctx_train = 4096;
    m->n_expert = r->s.n_expert;
    m->n_expert_used = r->s.n_used;
    fprintf(stderr, "inkling2_loop: %d layers, %d applications (loop %d..%d x %d passes), %d experts (%d used), "
                    "the serial float32 engine on the CPU; %.2f GB of dense float32 tensors decoded\n",
            r->s.n_block, r->s.n_app, r->s.loop_start, r->s.loop_end, r->s.loop_passes, r->s.n_expert, r->s.n_used,
            r->dense_bytes / 1e9);
    return true;
}

static size_t env_bytes(const char *k, size_t dflt) {
    const char *v = getenv(k);
    if (!v || !*v) return dflt;
    char *end = NULL;
    double x = strtod(v, &end);
    size_t mul = 1;
    if (end && (*end == 'K' || *end == 'k')) mul = 1024;
    else if (end && (*end == 'M' || *end == 'm')) mul = 1024 * 1024;
    else if (end && (*end == 'G' || *end == 'g')) mul = 1024ull * 1024 * 1024;
    return (size_t)(x * (double)mul);
}

bool model_inkling2_runtime(model_t *m, const model_params *p) {
    ink2_rt *r = rt_of(m);
    if (!r) return false;
    if (p->gpu_mode != GPU_OFF)
        fprintf(stderr, "note: %s; running on the CPU\n", m->ad && m->ad->cpu_only ? m->ad->cpu_only : "inkling2_loop runs on the CPU");
    int n_ctx = p->n_ctx > 0 ? p->n_ctx : (m->n_ctx_train < 4096 ? m->n_ctx_train : 4096);
    if (n_ctx > m->n_ctx_train) {
        fprintf(stderr, "error: -c %d is past this model's training context %d\n", n_ctx, m->n_ctx_train);
        return false;
    }
    m->n_ctx = n_ctx;
    m->n_batch = p->n_batch > 0 ? p->n_batch : 64;
    if (m->n_batch > n_ctx) m->n_batch = n_ctx;
    m->spec_batch = m->n_batch;
    if (ink2_cache_init_window(&r->c, &r->s, n_ctx, m->n_batch)) {
        fprintf(stderr, "error: the inkling2_loop cache for -c %d does not fit in memory\n", n_ctx);
        return false;
    }
    r->cache_ok = true;
    r->rows_cap = m->n_batch;
    r->rows = malloc(sizeof(float) * (size_t)r->rows_cap * (size_t)m->n_vocab);
    m->logits = malloc(sizeof(float) * (size_t)m->n_vocab);
    if (!r->rows || !m->logits) { fprintf(stderr, "error: out of memory for the logits rows\n"); return false; }
    // the decoded-expert cache: RUNNER_INKLING2_EXPERT_CACHE bytes (default 4 GiB; 0 decodes every expert on use)
    r->ec_budget = env_bytes("RUNNER_INKLING2_EXPERT_CACHE", (size_t)4 << 30);
    if (r->ec_budget) {
        ink2_ec_cfg cfg = { 1, r->ec_budget, 0, 64 };
        r->ec = ink2_ec_new(&r->inner, ink2g_fill_expert, r->g, cfg);
        if (!r->ec) { fprintf(stderr, "error: out of memory for the expert cache\n"); return false; }
        ink2_ec_weights(r->ec, &r->w);
    } else {
        r->w = r->inner;
    }
    // no thread count here: a server's slot is loaded with a one-thread pool, and the forward takes the count at use
    fprintf(stderr, "inkling2_loop: -c %d -b %d; cache %.2f GB (sliding K/V for the window only), expert "
                    "cache %.2f GB\n", n_ctx, m->n_batch, ink2_cache_bytes(&r->c, &r->s) / 1e9, r->ec_budget / 1e9);
    return true;
}

// RUNNER_INKLING2_ROWS, read once for every slot
static const char *rows_dump;
static pthread_once_t rows_dump_once = PTHREAD_ONCE_INIT;
static void rows_dump_read(void) { rows_dump = getenv("RUNNER_INKLING2_ROWS"); }

float *model_inkling2_forward(model_t *m, const int32_t *tokens, int n, int pos, bool want_logits, bool keep_all) {
    ink2_rt *r = rt_of(m);
    if (!r || !r->cache_ok || n <= 0 || pos < 0 || pos + n > m->n_ctx) {
        fprintf(stderr, "error: inkling2_loop: %d tokens at %d do not fit -c %d\n", n, pos, m->n_ctx);
        return NULL;
    }
    if (keep_all && n > r->rows_cap) return NULL;
    // the engine's own pool takes the instance's thread count at use, the one place it is set: a server loads each
    // slot with a one-thread pool and swaps in the shared pool afterwards, so a count read at bind would be one. Two
    // slots both set it from the one shared pool; a parallel loop runs with the count it read (src/inkling2.c). The
    // rows do not depend on the count (tests/test_inkling2_threads.c).
    const int nt = m->tp ? tpool_size(m->tp) : 1;
    if (nt != ink2_get_threads()) ink2_set_threads(nt);
    // chunks of n_batch: the engine's rows do not depend on the split (measured bit for bit)
    for (int done = 0; done < n;) {
        const int T = n - done < r->rows_cap ? n - done : r->rows_cap;
        ink2_status st = ink2_forward_chunk(&r->s, &r->w, &r->c, pos + done, T, (const int *)(tokens + done), r->rows, 0,
                                            NULL);   // int32_t is int on every target Runner builds for
        if (st != INK2_OK) {
            fprintf(stderr, "error: inkling2_loop: the forward at position %d refused (status %d%s)\n", pos + done, (int)st,
                    st == INK2_REWIND ? ": not where the sequence ends, and this model does not rewind" : "");
            return NULL;
        }
        // RUNNER_INKLING2_ROWS=FILE (TEST-ONLY, the wired path's bit-for-bit check): every forward's rows, appended
        // in call order, raw float32 [T][n_vocab], as the read harness writes them
        pthread_once(&rows_dump_once, rows_dump_read);
        const char *dump = rows_dump;
        if (dump && *dump) {
            FILE *f = plat_fopen(dump, "ab");
            if (f) {
                fwrite(r->rows, sizeof(float), (size_t)T * (size_t)m->n_vocab, f);
                fclose(f);
            }
            // beside it, FILE.threads: what the pool really did for this chunk (one line a chunk)
            char side[4096];
            snprintf(side, sizeof side, "%s.threads", dump);
            FILE *ft = plat_fopen(side, "a");
            if (ft) {
                long split = 0;
                int last = 1;
                ink2_par_stats(&split, &last);
                fprintf(ft, "pos %d T %d count %d split_loops %ld last_pool %d\n", pos + done, T, ink2_get_threads(), split,
                        last);
                fclose(ft);
            }
        }
        done += T;
        if (done == n) {
            r->kept_from = pos + done - T;
            r->kept_n = T;
        }
    }
    if (!want_logits) return NULL;
    memcpy(m->logits, r->rows + (size_t)(r->kept_n - 1) * (size_t)m->n_vocab, sizeof(float) * (size_t)m->n_vocab);
    return m->logits;
}

float *model_inkling2_row(model_t *m, int b) {
    ink2_rt *r = rt_of(m);
    if (!r || b < 0 || b >= r->kept_n) return NULL;
    return r->rows + (size_t)b * (size_t)m->n_vocab;
}

bool model_inkling2_fit(model_t *m, inkling2_fit *f) {
    ink2_rt *r = rt_of(m);
    memset(f, 0, sizeof *f);
    if (!r) return false;
    f->dense_f32 = r->dense_bytes;
    f->cache = r->cache_ok ? ink2_cache_bytes(&r->c, &r->s) : 0;
    f->expert_cache = r->ec_budget;
    f->mapped = gguf_mapped_size(&m->gf);
    f->logits = sizeof(float) * (uint64_t)m->n_vocab * (uint64_t)(r->rows_cap + 1);
    return true;
}

void model_inkling2_free(model_t *m) {
    ink2_rt *r = rt_of(m);
    if (!r) return;
    if (r->cache_ok) ink2_cache_free(&r->c);
    if (r->ec) ink2_ec_free(r->ec);
    if (r->g) ink2g_close(r->g);
    free(r->rows);
    free(r);
    m->ink2 = NULL;
}

int model_inkling2_fit_print(gguf_file *g, const char *path, int n_ctx_want, int n_batch) {
    ink2_shape s;
    char why[512] = { 0 };
    if (ink2g_shape(g, &s, why, sizeof why)) {
        printf("fit: %s\n  verdict       WILL NOT LOAD: %s\n", path, why);
        return 1;
    }
    int train = (int)gguf_get_u32(g, "inkling2_loop.context_length", 4096);
    int n_ctx = n_ctx_want > 0 ? n_ctx_want : (train < 4096 ? train : 4096);
    if (n_batch <= 0) n_batch = 64;
    if (n_batch > n_ctx) n_batch = n_ctx;
    // the dense float32 tensors the load decodes: every tensor but the expert stacks and their factors, the head at
    // its unpadded rows
    uint64_t dense = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor *t = &g->tensors[i];
        if (strstr(t->name, "_exps.")) continue;
        uint64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        if (!strcmp(t->name, "output.weight") && t->ne[1] > (uint64_t)s.vocab_used) n = t->ne[0] * (uint64_t)s.vocab_used;
        dense += 4 * n;
    }
    // the engine's cache under ink2_cache_init_window: K and V per application, the sliding ones for the window and
    // the batch, plus each application's four convolution histories
    uint64_t cache = 0;
    for (int a = 0; a < s.n_app; a++) {
        int gl = ink2_app_global(&s, a);
        uint64_t kv = (uint64_t)(gl ? s.n_kv * s.head_dim : s.swa_n_kv * s.swa_head_dim);
        uint64_t rows = (uint64_t)n_ctx;
        if (!gl && (uint64_t)s.window - 1 + (uint64_t)n_batch < rows) rows = (uint64_t)s.window - 1 + (uint64_t)n_batch;
        uint64_t wmax = kv > (uint64_t)s.hidden ? kv : (uint64_t)s.hidden;
        cache += 2 * rows * kv * 4 + 4ull * (uint64_t)(s.conv_k - 1) * wmax * 4;
    }
    size_t ec = env_bytes("RUNNER_INKLING2_EXPERT_CACHE", (size_t)4 << 30);
    uint64_t logits = 4ull * (uint64_t)s.vocab_used * (uint64_t)(n_batch + 1);
    uint64_t mapped = gguf_mapped_size(g);
    if (!mapped) for (uint64_t i = 0; i < g->n_tensors; i++) mapped += g->tensors[i].nbytes;
    uint64_t total = dense + cache + ec + logits + mapped;
    uint64_t avail = plat_ram_available_bytes();
    printf("fit: %s\n", path);
    printf("  arch          inkling2_loop, %d layers, %d applications, %d experts (%d used); the serial float32 engine on "
           "the CPU (no GPU path)\n", s.n_block, s.n_app, s.n_expert, s.n_used);
    printf("  context       %d (trained %d), batch %d\n", n_ctx, train, n_batch);
    printf("  file mapped   %.2f GiB (the experts are read from it)\n", mapped / 1073741824.0);
    printf("  dense f32     %.2f GiB (every non-expert tensor, decoded once)\n", dense / 1073741824.0);
    printf("  cache         %.2f GiB (K/V per application, the %d sliding ones for window %d + batch; convolutions)\n",
           cache / 1073741824.0, s.n_app - 8 > 0 ? s.n_app - 8 : 0, s.window);
    printf("  expert cache  %.2f GiB (RUNNER_INKLING2_EXPERT_CACHE)\n", ec / 1073741824.0);
    printf("  logits        %.2f GiB\n", logits / 1073741824.0);
    printf("  total         %.2f GiB of RAM (%.2f GB)\n", total / 1073741824.0, total / 1e9);
    if (avail)
        printf("  verdict       %s (%.2f GiB available)\n", total <= avail ? "FITS" : "DOES NOT FIT", avail / 1073741824.0);
    else
        printf("  verdict       unknown (available RAM unreadable)\n");
    return avail && total > avail ? 1 : 0;
}
