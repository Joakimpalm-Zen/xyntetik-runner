// R4.48: the decoded-expert cache; see inkling2_ecache.h.
#define _GNU_SOURCE
#include "inkling2_ecache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define EC_MAX_LAYER  64
#define EC_MAX_EXPERT 1024
#define EC_KEYS (EC_MAX_LAYER * 2 * EC_MAX_EXPERT)

typedef struct { int key; float *data; size_t mapped; uint64_t tick; ink2_tensor t; char name[96]; } ec_ent;
typedef struct { float *data; size_t mapped; ink2_tensor t; char name[96]; } ec_buf;

struct ink2_ecache {
    ink2_weights inner;
    ink2_ec_fill_fn fill; void *fctx;
    ink2_ec_cfg cfg;
    int *slot;                 // [EC_KEYS] -> index into ent, -1 none
    ec_ent **ent; int n_ent, cap_ent;
    uint64_t now;
    int last[2];               // the keys returned by the last two lookups (pinned), -1 none
    ec_buf bypass[2]; int bypass_k;
    ec_buf scratch;
    ink2_ec_stats st;
};

static size_t page_round(size_t n) { size_t pg = (size_t)sysconf(_SC_PAGESIZE); return (n + pg - 1) / pg * pg; }
static float *map_floats(size_t mapped) {
    void *p = mmap(NULL, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}
static int buf_fit(ink2_ecache *c, ec_buf *b, size_t bytes) {
    size_t m = page_round(bytes);
    if (b->mapped >= m) return 0;
    if (b->data) { munmap(b->data, b->mapped); c->st.aux_bytes -= b->mapped; }
    b->data = map_floats(m); b->mapped = b->data ? m : 0;
    if (!b->data) return -1;
    c->st.aux_bytes += m;
    return 0;
}

static size_t parse_size(const char *s) {
    char *e; double v = strtod(s, &e);
    if (*e == 'K' || *e == 'k') v *= 1024.0; else if (*e == 'M' || *e == 'm') v *= 1048576.0; else if (*e == 'G' || *e == 'g') v *= 1073741824.0;
    return v > 0 ? (size_t)v : 0;
}
ink2_ec_cfg ink2_ec_cfg_env(void) {
    ink2_ec_cfg c = { 0, 0, 0, 64 };
    const char *b = getenv("INK2_ECACHE_BYTES"), *n = getenv("INK2_ECACHE_ENTRIES"), *k = getenv("INK2_ECACHE_CHECK");
    if (b && *b) { c.budget_bytes = parse_size(b); c.enabled = 1; }
    if (n && *n) { c.max_entries = atoi(n); c.enabled = 1; }
    if (k && *k) c.check_every = !strcmp(k, "all") ? 1 : atoi(k);
    return c;
}

static const ink2_tensor *ec_get(void *ctx, const char *name) {
    ink2_ecache *c = ctx;
    return c->inner.get(c->inner.ctx, name);
}

// the key of a stack name, or -1
static int ec_key(const char *stack, int e) {
    int b, n = 0;
    if (sscanf(stack, "model.layers.%d.mlp.experts.%n", &b, &n) != 1 || !n) return -1;
    const char *kind = stack + n;
    int k = !strcmp(kind, "gate_up_proj") ? 0 : !strcmp(kind, "down_proj") ? 1 : -1;
    if (k < 0 || b < 0 || b >= EC_MAX_LAYER || e < 0 || e >= EC_MAX_EXPERT) return -1;
    return (b * 2 + k) * EC_MAX_EXPERT + e;
}

static void ec_drop(ink2_ecache *c, int i) {
    ec_ent *x = c->ent[i];
    c->slot[x->key] = -1;
    munmap(x->data, x->mapped);
    c->st.entry_bytes -= x->mapped; c->st.entries--;
    free(x); c->st.aux_bytes -= sizeof(ec_ent);
    c->ent[i] = c->ent[--c->n_ent];
    if (i < c->n_ent) c->slot[c->ent[i]->key] = i;
}
void ink2_ec_clear(ink2_ecache *c) {
    while (c->n_ent) ec_drop(c, c->n_ent - 1);
    c->last[0] = c->last[1] = -1;
    c->st.clears++;
}
static int fits(const ink2_ecache *c, size_t mapped) {
    return (!c->cfg.budget_bytes || c->st.entry_bytes + mapped <= c->cfg.budget_bytes) &&
           (!c->cfg.max_entries || c->st.entries + 1 <= c->cfg.max_entries);
}
static void set_t(ink2_tensor *t, char *keep, const char *name, const float *d, int64_t rows, int64_t cols) {
    snprintf(keep, 96, "%s", name); t->name = keep; t->data = d; t->n_dims = 2; t->ne[0] = cols; t->ne[1] = rows; t->ne[2] = t->ne[3] = 1;
}

static const ink2_tensor *ec_get_expert(void *ctx, const char *stack, int e) {
    ink2_ecache *c = ctx;
    c->st.lookups++; c->now++;
    int key = ec_key(stack, e);
    if (key < 0) { c->st.uncached++; return c->inner.get_expert ? c->inner.get_expert(c->inner.ctx, stack, e) : NULL; }
    int64_t rows, cols;
    int i = c->slot[key];
    if (i >= 0) {
        ec_ent *x = c->ent[i];
        c->st.hits++; x->tick = c->now;
        if (c->cfg.check_every && c->st.hits % (uint64_t)c->cfg.check_every == 0) {
            size_t bytes = (size_t)x->t.ne[0] * (size_t)x->t.ne[1] * sizeof(float);
            if (buf_fit(c, &c->scratch, bytes) || c->fill(c->fctx, stack, e, c->scratch.data, &rows, &cols) ||
                rows != x->t.ne[1] || cols != x->t.ne[0] || memcmp(c->scratch.data, x->data, bytes)) {
                fprintf(stderr, "INK2 ECACHE MISMATCH: %s expert %d, the cached copy differs from a fresh decode "
                        "(hit %llu)\n", stack, e, (unsigned long long)c->st.hits);
                abort();
            }
            c->st.checks++;
        }
        c->last[1] = c->last[0]; c->last[0] = key;
        return &x->t;
    }
    c->st.misses++;
    if (c->fill(c->fctx, stack, e, NULL, &rows, &cols)) return NULL;
    size_t bytes = (size_t)rows * (size_t)cols * sizeof(float), mapped = page_round(bytes);
    // room: evict the least recently used entries other than the two returned last
    while (!fits(c, mapped)) {
        int v = -1;
        for (int j = 0; j < c->n_ent; j++) {
            int k = c->ent[j]->key;
            if (k == c->last[0] || k == c->last[1]) continue;
            if (v < 0 || c->ent[j]->tick < c->ent[v]->tick) v = j;
        }
        if (v < 0) break;
        ec_drop(c, v); c->st.evictions++;
    }
    if (!fits(c, mapped)) {   // the bound holds: decode beside the cache
        ec_buf *b = &c->bypass[c->bypass_k++ & 1];
        if (buf_fit(c, b, bytes) || c->fill(c->fctx, stack, e, b->data, &rows, &cols)) return NULL;
        c->st.bypasses++;
        set_t(&b->t, b->name, stack, b->data, rows, cols);
        c->last[1] = c->last[0]; c->last[0] = -1;
        return &b->t;
    }
    ec_ent *x = calloc(1, sizeof *x);
    float *d = map_floats(mapped);
    if (!x || !d || c->fill(c->fctx, stack, e, d, &rows, &cols)) { free(x); if (d) munmap(d, mapped); return NULL; }
    if (c->n_ent == c->cap_ent) {
        int nc = c->cap_ent ? 2 * c->cap_ent : 256;
        ec_ent **ne = realloc(c->ent, sizeof *ne * (size_t)nc);
        if (!ne) { free(x); munmap(d, mapped); return NULL; }
        c->st.aux_bytes += sizeof *ne * (size_t)(nc - c->cap_ent);
        c->ent = ne; c->cap_ent = nc;
    }
    x->key = key; x->data = d; x->mapped = mapped; x->tick = c->now;
    set_t(&x->t, x->name, stack, d, rows, cols);
    c->ent[c->n_ent] = x; c->slot[key] = c->n_ent++;
    c->st.inserts++; c->st.entries++; c->st.entry_bytes += mapped; c->st.aux_bytes += sizeof *x;
    if (c->st.entries > c->st.peak_entries) c->st.peak_entries = c->st.entries;
    if (c->st.entry_bytes > c->st.peak_entry_bytes) c->st.peak_entry_bytes = c->st.entry_bytes;
    c->last[1] = c->last[0]; c->last[0] = key;
    return &x->t;
}

ink2_ecache *ink2_ec_new(const ink2_weights *inner, ink2_ec_fill_fn fill, void *fctx, ink2_ec_cfg cfg) {
    ink2_ecache *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->inner = *inner; c->fill = fill; c->fctx = fctx; c->cfg = cfg;
    c->slot = malloc(sizeof(int) * EC_KEYS);
    if (!c->slot) { free(c); return NULL; }
    for (int i = 0; i < EC_KEYS; i++) c->slot[i] = -1;
    c->last[0] = c->last[1] = -1;
    c->st.aux_bytes = sizeof(int) * EC_KEYS + sizeof *c;
    return c;
}
void ink2_ec_free(ink2_ecache *c) {
    if (!c) return;
    ink2_ec_clear(c);
    for (int i = 0; i < 2; i++) if (c->bypass[i].data) munmap(c->bypass[i].data, c->bypass[i].mapped);
    if (c->scratch.data) munmap(c->scratch.data, c->scratch.mapped);
    free(c->ent); free(c->slot); free(c);
}
void ink2_ec_weights(ink2_ecache *c, ink2_weights *w) { w->get = ec_get; w->get_expert = ec_get_expert; w->ctx = c; }
void ink2_ec_get_stats(const ink2_ecache *c, ink2_ec_stats *st) { *st = c->st; }
size_t ink2_ec_resident(const ink2_ecache *c) { return c->st.entry_bytes + c->st.aux_bytes; }
void ink2_ec_print(const ink2_ecache *c, const char *tag, void *file) {
    const ink2_ec_stats *s = &c->st;
    fprintf((FILE *)file, "%s: budget %zu bytes, %d entries max, check every %d%s; %llu lookups: %llu hits (%llu memcmp-checked, "
            "0 differ), %llu misses (%llu inserted, %llu bypassed), %llu evictions, %llu uncached, %llu clears; entries now %d "
            "(%zu bytes), peak %d (%zu bytes); auxiliary %zu bytes; resident %zu bytes\n",
            tag, c->cfg.budget_bytes, c->cfg.max_entries, c->cfg.check_every, c->cfg.check_every == 1 ? " (debug: every hit)" : "",
            (unsigned long long)s->lookups, (unsigned long long)s->hits, (unsigned long long)s->checks,
            (unsigned long long)s->misses, (unsigned long long)s->inserts, (unsigned long long)s->bypasses,
            (unsigned long long)s->evictions, (unsigned long long)s->uncached, (unsigned long long)s->clears, s->entries,
            s->entry_bytes, s->peak_entries, s->peak_entry_bytes, s->aux_bytes, ink2_ec_resident(c));
}
