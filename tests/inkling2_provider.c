// R4.48: the test-side Inkling-2-Mini weight provider; see inkling2_provider.h. Built strict, as inkling2.c is
// (-fno-fast-math -ffp-contract=off): every decode below is one float32 operation at a time, in the loader's order.
#ifdef _WIN32
#error "a POSIX-only lab tool (mmap, getline, posix_memalign, setenv): not built on Windows"
#else
#define _GNU_SOURCE
#include "inkling2_provider.h"
#include "json.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { DT_BF16, DT_F32, DT_F8E4M3, DT_U8, DT_BAD };
enum { OP_RENAME, OP_UNPAD, OP_MERGE_MQ, OP_DEQ };
static const char *dt_names[] = { "BF16", "F32", "F8_E4M3", "U8" };
static const int dt_size[] = { 2, 4, 1, 1 };

typedef struct { char key[192]; int dt, nd; int64_t dims[4]; uint64_t b, e; const unsigned char *p; } src_t;
typedef struct {
    char name[192], ops[96];
    int op, nd, nsrc, il, nvfp4;   // il: Interleave(dim=1) after the dequantization; nvfp4: amax present
    int64_t dims[4];
    int has[4];
    src_t src[4];
} ent_t;

typedef struct { int ent, e, used; float *data; size_t bytes; uint64_t tick; ink2_tensor t; } slot_t;
typedef struct { slot_t *s; int n; size_t budget, bytes, peak; uint64_t now; int pin; } pool_t;

struct ink2p {
    int fd; unsigned char *map; size_t fsize; uint64_t hn; const unsigned char *data0;
    ent_t *ent; int n_ent; int *order;   // order: indices sorted by name
    pool_t tp, ep;
    int ctrl;
    ink2p_stats st;
    float e4m3[256], e2m1[16], p2[256];
};

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int64_t numel(const int64_t *d, int nd) { int64_t n = 1; for (int i = 0; i < nd; i++) n *= d[i]; return n; }

// float8_e4m3fn: sign, 4 exponent bits (bias 7), 3 mantissa bits; no infinities, S.1111.111 is NaN
static float e4m3fn(unsigned b) {
    int s = b >> 7, ex = (b >> 3) & 15, m = b & 7;
    float v;
    if (ex == 15 && m == 7) v = NAN;
    else if (ex == 0) v = ldexpf((float)m, -9);              // m/8 * 2^-6
    else v = ldexpf((float)(8 + m), ex - 10);                // (1 + m/8) * 2^(ex-7)
    return s ? -v : v;
}
static float bf16f(const unsigned char *p) {
    uint16_t h; memcpy(&h, p, 2); uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static float f32at(const unsigned char *p) { float f; memcpy(&f, p, 4); return f; }

// ---------------------------------------------------------------- the map
static int dt_parse(const char *s) {
    for (int i = 0; i < 4; i++) if (!strcmp(s, dt_names[i])) return i;
    return DT_BAD;
}
static const ent_t *cmp_ents;   // qsort's table (qsort takes no context)
static int cmp_ord(const void *a, const void *b) { return strcmp(cmp_ents[*(const int *)a].name, cmp_ents[*(const int *)b].name); }

int ink2p_find(const ink2p *p, const char *name) {
    int lo = 0, hi = p->n_ent - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(name, p->ent[p->order[mid]].name);
        if (!c) return p->order[mid];
        if (c < 0) hi = mid - 1; else lo = mid + 1;
    }
    return -1;
}

// the shape rules of each op chain; 0 ok
static int ent_check(ent_t *E, char *why, size_t wn) {
    const int64_t *d = E->dims;
    src_t *s = E->src;
#define BAD(...) do { snprintf(why, wn, __VA_ARGS__); return -1; } while (0)
    if (E->op == OP_RENAME) {
        if (E->nsrc != 1 || !E->has[0]) BAD("%s: rename wants one source", E->name);
        if (s[0].dt != DT_BF16 && s[0].dt != DT_F32) BAD("%s: rename of %s", E->name, dt_names[s[0].dt]);
        if (s[0].nd != E->nd || memcmp(s[0].dims, d, sizeof(int64_t) * E->nd)) BAD("%s: source shape differs", E->name);
    } else if (E->op == OP_UNPAD) {
        if (E->nsrc != 1 || !E->has[0] || E->nd != 2 || s[0].nd != 2 || s[0].dims[1] != d[1] || s[0].dims[0] < d[0] ||
            (s[0].dt != DT_BF16 && s[0].dt != DT_F32)) BAD("%s: unpad shapes", E->name);
    } else if (E->op == OP_MERGE_MQ) {
        if (E->nsrc != 2 || !E->has[0] || !E->has[1] || E->nd != 3 || d[1] != 4) BAD("%s: merge sources", E->name);
        for (int i = 0; i < 2; i++)
            if (s[i].dt != DT_F32 || s[i].nd != 2 || s[i].dims[0] != 2 * d[0] || s[i].dims[1] != d[2]) BAD("%s: merge shapes", E->name);
    } else {   // OP_DEQ: weight, scale, amax, denominator (by the converter's source patterns)
        if (E->nsrc != 4 || !E->has[0] || !E->has[1] || E->nd != 3) BAD("%s: dequantize sources", E->name);
        int64_t n_e = d[0], R = d[1], C = d[2];
        if (E->il && R % 2) BAD("%s: interleave of odd rows", E->name);
        E->nvfp4 = E->has[2];
        if (E->nvfp4) {
            if (!E->has[3] || s[0].dt != DT_U8 || s[1].dt != DT_F8E4M3 || s[2].dt != DT_F32 || s[3].dt != DT_F32 || C % 16 ||
                s[0].nd != 3 || s[0].dims[0] != n_e || s[0].dims[1] != R || s[0].dims[2] * 2 != C ||
                s[1].nd != 3 || s[1].dims[0] != n_e || s[1].dims[1] != R || s[1].dims[2] * 16 != C ||
                s[2].nd != 1 || s[2].dims[0] != n_e || numel(s[3].dims, s[3].nd) != 1) BAD("%s: NVFP4 shapes", E->name);
        } else {
            if (E->has[3] || s[0].dt != DT_F8E4M3 || s[1].dt != DT_U8 || C % 32 ||
                s[0].nd != 3 || s[0].dims[0] != n_e || s[0].dims[1] != R || s[0].dims[2] != C ||
                s[1].nd != 3 || s[1].dims[0] != n_e || s[1].dims[1] != R || s[1].dims[2] * 32 != C) BAD("%s: MXFP8 shapes", E->name);
        }
    }
#undef BAD
    return 0;
}

static int map_load(ink2p *p, const char *path, char *why, size_t wn) {
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(why, wn, "cannot open %s", path); return -1; }
    size_t cap = 1024; p->ent = calloc(cap, sizeof *p->ent);
    char *line = NULL; size_t ln = 0; ssize_t r; int have_h = 0;
    uint64_t hn = 0, fs = 0;
    while ((r = getline(&line, &ln, f)) > 0) {
        char *save = NULL, *tok = strtok_r(line, " \n", &save);
        if (!tok) continue;
        if (!strcmp(tok, "H")) {
            hn = strtoull(strtok_r(NULL, " \n", &save), NULL, 10); fs = strtoull(strtok_r(NULL, " \n", &save), NULL, 10);
            have_h = 1; continue;
        }
        if (strcmp(tok, "T")) continue;   // U lines: checkpoint keys no parameter takes (the vision tower)
        if ((size_t)p->n_ent == cap) { cap *= 2; p->ent = realloc(p->ent, cap * sizeof *p->ent); }
        ent_t *E = &p->ent[p->n_ent]; memset(E, 0, sizeof *E);
#define NX() strtok_r(NULL, " \n", &save)
        char *t;
        if (!(t = NX())) goto bad;
        snprintf(E->name, sizeof E->name, "%s", t);
        if (!(t = NX())) goto bad;
        snprintf(E->ops, sizeof E->ops, "%s", t);
        if (!(t = NX())) goto bad;
        E->nd = atoi(t); if (E->nd < 1 || E->nd > 4) goto bad;
        for (int i = 0; i < E->nd; i++) { if (!(t = NX())) goto bad; E->dims[i] = strtoll(t, NULL, 10); }
        if (!(t = NX())) goto bad;
        E->nsrc = atoi(t); if (E->nsrc < 1 || E->nsrc > 4) goto bad;
        for (int k = 0; k < E->nsrc; k++) {
            if (!(t = NX())) goto bad;
            if (!strcmp(t, "-")) continue;
            src_t *s = &E->src[k]; E->has[k] = 1;
            snprintf(s->key, sizeof s->key, "%s", t);
            if (!(t = NX())) goto bad;
            s->dt = dt_parse(t); if (s->dt == DT_BAD) goto bad;
            if (!(t = NX())) goto bad;
            s->nd = atoi(t); if (s->nd < 1 || s->nd > 4) goto bad;
            for (int i = 0; i < s->nd; i++) { if (!(t = NX())) goto bad; s->dims[i] = strtoll(t, NULL, 10); }
            if (!(t = NX())) goto bad;
            s->b = strtoull(t, NULL, 10);
            if (!(t = NX())) goto bad;
            s->e = strtoull(t, NULL, 10);
        }
#undef NX
        if (!strcmp(E->ops, "rename")) E->op = OP_RENAME;
        else if (!strcmp(E->ops, "Inkling2UnpadVocab")) E->op = OP_UNPAD;
        else if (!strcmp(E->ops, "Inkling2LoopMergeMessageQueries")) E->op = OP_MERGE_MQ;
        else if (!strcmp(E->ops, "Inkling2LoopDequantizeExperts")) E->op = OP_DEQ;
        else if (!strcmp(E->ops, "Inkling2LoopDequantizeExperts,Interleave(dim=1)")) { E->op = OP_DEQ; E->il = 1; }
        else { snprintf(why, wn, "%s: op chain %s has no decode here", E->name, E->ops); goto fail; }
        if (ent_check(E, why, wn)) goto fail;
        p->n_ent++;
        continue;
bad:
        snprintf(why, wn, "map line %d malformed", p->n_ent + 1);
        goto fail;
    }
    free(line); fclose(f);
    if (!have_h) { snprintf(why, wn, "map has no H line"); return -1; }
    p->hn = hn;
    if (fs != p->fsize) { snprintf(why, wn, "map says file size %llu, file is %zu", (unsigned long long)fs, p->fsize); return -1; }
    return 0;
fail:
    free(line); fclose(f);
    return -1;
}

// every source against the file's own header: dtype, shape, offsets; and the byte range inside the file
static int map_verify(ink2p *p, char *why, size_t wn) {
    uint64_t hn; memcpy(&hn, p->map, 8);
    if (hn != p->hn || 8 + hn > p->fsize) { snprintf(why, wn, "header length %llu, map says %llu", (unsigned long long)hn, (unsigned long long)p->hn); return -1; }
    jv *h = json_parse((const char *)p->map + 8, (size_t)hn);
    if (!h || h->type != J_OBJ) { snprintf(why, wn, "header does not parse"); return -1; }
    p->data0 = p->map + 8 + hn;
    int rc = 0;
    for (int i = 0; i < p->n_ent && !rc; i++) {
        ent_t *E = &p->ent[i];
        for (int k = 0; k < E->nsrc && !rc; k++) {
            if (!E->has[k]) continue;
            src_t *s = &E->src[k];
            jv *e = jv_get(h, s->key), *sh = e ? jv_get(e, "shape") : NULL, *off = e ? jv_get(e, "data_offsets") : NULL;
            if (!e || !sh || !off || sh->type != J_ARR || off->type != J_ARR || off->n != 2 ||
                strcmp(jv_str(jv_get(e, "dtype"), ""), dt_names[s->dt]) || sh->n != s->nd) { rc = -1; break; }
            for (int d = 0; d < s->nd; d++) if ((int64_t)sh->items[d]->num != s->dims[d]) rc = -1;
            if ((uint64_t)off->items[0]->num != s->b || (uint64_t)off->items[1]->num != s->e) rc = -1;
            if (s->e < s->b || s->e - s->b != (uint64_t)numel(s->dims, s->nd) * dt_size[s->dt] ||
                8 + hn + s->e > p->fsize) rc = -1;
            if (rc) snprintf(why, wn, "%s: source %s disagrees with the file's header", E->name, s->key);
            s->p = p->data0 + s->b;
        }
        if (rc && !why[0]) snprintf(why, wn, "%s: a source is not in the file's header", E->name);
    }
    jv_free(h);
    return rc;
}

ink2p *ink2p_open(const char *ckpt, const char *map, size_t tensor_budget, size_t expert_budget, char *why, size_t wn) {
    why[0] = 0;
    ink2p *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->fd = open(ckpt, O_RDONLY);
    struct stat sb;
    if (p->fd < 0 || fstat(p->fd, &sb)) { snprintf(why, wn, "cannot open %s: %s", ckpt, strerror(errno)); free(p); return NULL; }
    p->fsize = (size_t)sb.st_size;
    p->map = mmap(NULL, p->fsize, PROT_READ, MAP_SHARED, p->fd, 0);
    if (p->map == MAP_FAILED) { snprintf(why, wn, "mmap: %s", strerror(errno)); close(p->fd); free(p); return NULL; }
    if (map_load(p, map, why, wn) || map_verify(p, why, wn)) { ink2p_close(p); return NULL; }
    p->order = malloc(sizeof(int) * (size_t)p->n_ent);
    for (int i = 0; i < p->n_ent; i++) p->order[i] = i;
    cmp_ents = p->ent; qsort(p->order, (size_t)p->n_ent, sizeof(int), cmp_ord);
    for (int i = 1; i < p->n_ent; i++)
        if (!strcmp(p->ent[p->order[i]].name, p->ent[p->order[i - 1]].name)) { snprintf(why, wn, "%s twice", p->ent[p->order[i]].name); ink2p_close(p); return NULL; }
    for (int b = 0; b < 256; b++) { p->e4m3[b] = e4m3fn((unsigned)b); p->p2[b] = ldexpf(1.0f, b - 127); }
    // Inkling2LoopDequantizeExperts.E2M1_VALUES, by code
    static const float e2m1[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
    memcpy(p->e2m1, e2m1, sizeof e2m1);
    p->tp.n = 1024; p->tp.s = calloc((size_t)p->tp.n, sizeof(slot_t)); p->tp.budget = tensor_budget; p->tp.pin = INK2P_PIN_TENSORS;
    p->ep.n = 4096; p->ep.s = calloc((size_t)p->ep.n, sizeof(slot_t)); p->ep.budget = expert_budget; p->ep.pin = INK2P_PIN_EXPERTS;
    return p;
}

static void pool_clear(pool_t *P) {
    for (int i = 0; i < P->n; i++) if (P->s[i].used) { free(P->s[i].data); P->s[i].used = 0; }
    P->bytes = 0;
}
void ink2p_close(ink2p *p) {
    if (!p) return;
    if (p->tp.s) { pool_clear(&p->tp); free(p->tp.s); }
    if (p->ep.s) { pool_clear(&p->ep); free(p->ep.s); }
    if (p->map && p->map != MAP_FAILED) munmap(p->map, p->fsize);
    if (p->fd >= 0) close(p->fd);
    free(p->ent); free(p->order); free(p);
}
void ink2p_weights(ink2p *p, ink2_weights *w) { w->get = ink2p_get; w->get_expert = ink2p_get_expert; w->ctx = p; }
int ink2p_count(const ink2p *p) { return p->n_ent; }
const char *ink2p_name(const ink2p *p, int i, int *n_dims, int64_t *dims, const char **ops, const char **src_dtype) {
    const ent_t *E = &p->ent[i];
    if (n_dims) *n_dims = E->nd;
    if (dims) for (int d = 0; d < 4; d++) dims[d] = d < E->nd ? E->dims[d] : 1;
    if (ops) *ops = E->ops;
    if (src_dtype) *src_dtype = dt_names[E->src[0].dt];
    return E->name;
}
int ink2p_is_stack(const ink2p *p, int i) { return p->ent[i].op == OP_DEQ; }
void ink2p_set_control(ink2p *p, int flags) { p->ctrl = flags; pool_clear(&p->tp); pool_clear(&p->ep); }
void ink2p_get_stats(const ink2p *p, ink2p_stats *st) {
    *st = p->st;
    st->tensor_bytes = p->tp.bytes; st->tensor_peak = p->tp.peak; st->expert_bytes = p->ep.bytes; st->expert_peak = p->ep.peak;
}

// ---------------------------------------------------------------- the caches
static slot_t *pool_find(pool_t *P, int ent, int e) {
    for (int i = 0; i < P->n; i++) if (P->s[i].used && P->s[i].ent == ent && P->s[i].e == e) return &P->s[i];
    return NULL;
}
// a slot for `bytes` more: evict least recently used slots outside the pin window until the budget holds
static slot_t *pool_new(pool_t *P, size_t bytes) {
    while (P->bytes + bytes > P->budget) {
        slot_t *v = NULL;
        for (int i = 0; i < P->n; i++) {
            slot_t *s = &P->s[i];
            if (!s->used || P->now - s->tick <= (uint64_t)P->pin) continue;   // pinned: one of the last `pin` lookups
            if (!v || s->tick < v->tick) v = s;
        }
        if (!v) break;   // everything pinned: run over the budget
        free(v->data); v->used = 0; P->bytes -= v->bytes;
    }
    for (int i = 0; i < P->n; i++) if (!P->s[i].used) {
        slot_t *s = &P->s[i];
        void *d = NULL;
        if (posix_memalign(&d, 64, bytes ? bytes : 64)) return NULL;
        s->data = d; s->bytes = bytes; s->used = 1;
        P->bytes += bytes; if (P->bytes > P->peak) P->peak = P->bytes;
        return s;
    }
    return NULL;
}
// drop the decoded source pages from this process (they stay in the page cache): the mapping is the whole file
static void release(const ink2p *p, const unsigned char *a, size_t n) {
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t lo = (uintptr_t)a & ~(uintptr_t)(pg - 1), hi = ((uintptr_t)a + n + pg - 1) & ~(uintptr_t)(pg - 1);
    madvise((void *)lo, hi - lo, MADV_DONTNEED);
    (void)p;
}

// ---------------------------------------------------------------- the decodes
// one expert e of a stack into dst [R][C], in the loader's order (see the header)
static void deq_expert(const ink2p *p, const ent_t *E, int e, float *dst) {
    const int64_t R = E->dims[1], C = E->dims[2];
    const int il = E->il && !(p->ctrl & INK2P_CTRL_NO_INTERLEAVE);
    if (E->nvfp4) {
        const unsigned char *W = E->src[0].p + (size_t)e * R * (C / 2), *S = E->src[1].p + (size_t)e * R * (C / 16);
        const float amax = f32at(E->src[2].p + 4 * (size_t)e), den = f32at(E->src[3].p);
        const float factor = amax / den;                        // expert_factor = amax.float() / denominator.float()
        const int nof = p->ctrl & INK2P_CTRL_NO_FACTOR, swp = (p->ctrl & INK2P_CTRL_SWAP_NIBBLE) ? 4 : 0;
        for (int64_t o = 0; o < R; o++) {
            int64_t r = il ? (o < R / 2 ? 2 * o : 2 * (o - R / 2) + 1) : o;   // Interleave(dim=1)
            const unsigned char *wr = W + (size_t)r * (C / 2), *sr = S + (size_t)r * (C / 16);
            float *d = dst + (size_t)o * C;
            for (int64_t j = 0; j < C / 2; j++) {
                const unsigned b = wr[j];
                const float sc = p->e4m3[sr[j >> 3]];             // elements 2j, 2j+1 share block (2j) / 16 = j / 8
                const float lo = p->e2m1[(b >> swp) & 15], hi = p->e2m1[(b >> (4 - swp)) & 15];
                float v0 = lo * sc, v1 = hi * sc;                  // byte_to_values[codes] * block_scale
                if (!nof) { v0 = v0 * factor; v1 = v1 * factor; } // * expert_factor
                d[2 * j] = v0; d[2 * j + 1] = v1;
            }
        }
        release(p, W, (size_t)R * (C / 2)); release(p, S, (size_t)R * (C / 16));
    } else {
        const unsigned char *W = E->src[0].p + (size_t)e * R * C, *S = E->src[1].p + (size_t)e * R * (C / 32);
        for (int64_t o = 0; o < R; o++) {
            int64_t r = il ? (o < R / 2 ? 2 * o : 2 * (o - R / 2) + 1) : o;
            const unsigned char *wr = W + (size_t)r * C, *sr = S + (size_t)r * (C / 32);
            float *d = dst + (size_t)o * C;
            for (int64_t j = 0; j < C; j++) d[j] = p->e4m3[wr[j]] * p->p2[sr[j >> 5]];   // values * exp2(scale - 127)
        }
        release(p, W, (size_t)R * C); release(p, S, (size_t)R * (C / 32));
    }
}
static void dec_rows(const src_t *s, size_t n, float *dst) {
    if (s->dt == DT_F32) memcpy(dst, s->p, n * 4);
    else for (size_t i = 0; i < n; i++) dst[i] = bf16f(s->p + 2 * i);
}
static void dec_tensor(const ink2p *p, const ent_t *E, float *dst) {
    size_t n = (size_t)numel(E->dims, E->nd);
    if (E->op == OP_RENAME || E->op == OP_UNPAD) {   // unpad: the first rows, n of them
        dec_rows(&E->src[0], n, dst);
        release(p, E->src[0].p, n * dt_size[E->src[0].dt]);
    } else if (E->op == OP_MERGE_MQ) {
        const int64_t L = E->dims[0], H = E->dims[2];
        const unsigned char *q = E->src[0].p, *kv = E->src[1].p;
        for (int64_t l = 0; l < L; l++) {
            const unsigned char *rows[4] = { q + (size_t)(2 * l) * H * 4, kv + (size_t)(2 * l) * H * 4,
                                             kv + (size_t)(2 * l + 1) * H * 4, q + (size_t)(2 * l + 1) * H * 4 };
            for (int k = 0; k < 4; k++) memcpy(dst + ((size_t)l * 4 + k) * H, rows[k], (size_t)H * 4);
        }
    } else {
        const size_t per = (size_t)E->dims[1] * E->dims[2];
        for (int64_t e = 0; e < E->dims[0]; e++) deq_expert(p, E, (int)e, dst + (size_t)e * per);
    }
}

const ink2_tensor *ink2p_get(void *ctx, const char *name) {
    ink2p *p = ctx;
    int i = ink2p_find(p, name);
    if (i < 0) return NULL;
    const ent_t *E = &p->ent[i];
    p->tp.now++; p->st.gets++;
    slot_t *s = pool_find(&p->tp, i, -1);
    if (s) { p->st.get_hits++; s->tick = p->tp.now; return &s->t; }
    size_t n = (size_t)numel(E->dims, E->nd);
    s = pool_new(&p->tp, n * sizeof(float));
    if (!s) return NULL;
    double t0 = now_s();
    dec_tensor(p, E, s->data);
    p->st.decode_s_tensor += now_s() - t0; p->st.decoded_floats_tensor += n;
    s->ent = i; s->e = -1; s->tick = p->tp.now;
    s->t.name = E->name; s->t.data = s->data; s->t.n_dims = E->nd;
    for (int d = 0; d < 4; d++) s->t.ne[d] = d < E->nd ? E->dims[E->nd - 1 - d] : 1;
    return &s->t;
}

const ink2_tensor *ink2p_get_expert(void *ctx, const char *stack, int e) {
    ink2p *p = ctx;
    int i = ink2p_find(p, stack);
    if (i < 0 || p->ent[i].op != OP_DEQ || e < 0 || e >= p->ent[i].dims[0]) return NULL;
    const ent_t *E = &p->ent[i];
    p->ep.now++; p->st.experts++;
    slot_t *s = pool_find(&p->ep, i, e);
    if (s) { p->st.expert_hits++; s->tick = p->ep.now; return &s->t; }
    size_t n = (size_t)E->dims[1] * E->dims[2];
    s = pool_new(&p->ep, n * sizeof(float));
    if (!s) return NULL;
    double t0 = now_s();
    deq_expert(p, E, e, s->data);
    p->st.decode_s_expert += now_s() - t0; p->st.decoded_floats_expert += n;
    s->ent = i; s->e = e; s->tick = p->ep.now;
    s->t.name = E->name; s->t.data = s->data; s->t.n_dims = 2;
    s->t.ne[0] = E->dims[2]; s->t.ne[1] = E->dims[1]; s->t.ne[2] = s->t.ne[3] = 1;
    return &s->t;
}

int ink2p_fill_expert(void *ctx, const char *stack, int e, float *dst, int64_t *rows, int64_t *cols) {
    ink2p *p = ctx;
    int i = ink2p_find(p, stack);
    if (i < 0 || p->ent[i].op != OP_DEQ || e < 0 || e >= p->ent[i].dims[0]) return -1;
    const ent_t *E = &p->ent[i];
    *rows = E->dims[1]; *cols = E->dims[2];
    if (dst) {
        double t0 = now_s();
        deq_expert(p, E, e, dst);
        p->st.decode_s_expert += now_s() - t0; p->st.decoded_floats_expert += (uint64_t)E->dims[1] * E->dims[2];
    }
    return 0;
}
#endif   // !_WIN32
