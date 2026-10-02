// Session images (R1.3). See session.h.
#include "session.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#endif

#include "compat.h"
#include "envelope.h"
#include "runner.h"
#include "json.h"

#ifndef O_BINARY
#define O_BINARY 0
#endif

#define SESS_MAGIC "runner.session.1"
#define SESS_SCHEMA "xyntetik.runner.session.v1"

// the image's bytes are built in memory and hashed as they are, then written
// once: an image is at most one KV cache plus a logits row
typedef struct { uint8_t *b; size_t n, cap; bool oom; } wbuf;

static void wput(wbuf *w, const void *p, size_t n) {
    if (w->oom) return;
    if (w->n + n > w->cap) {
        size_t cap = w->cap ? w->cap : 1 << 16;
        while (cap < w->n + n) cap *= 2;
        uint8_t *g = realloc(w->b, cap);
        if (!g) { w->oom = true; return; }
        w->b = g; w->cap = cap;
    }
    memcpy(w->b + w->n, p, n);
    w->n += n;
}

bool session_write(const char *path, const engine *e, const session_meta *mt,
                   const float *logits, int n_vocab, char sha_out[65]) {
    size_t sn = engine_state_bytes(e);
    uint8_t *state = malloc(sn ? sn : 1);
    if (!state || !engine_state_save(e, state)) {
        free(state);
        fprintf(stderr, "error: session: this model's KV layout (a ring or tied-V "
                "cache) cannot be imaged\n");
        return false;
    }
    sbuf h = {0};
    sb_fmt(&h, "{\"schema_version\":\"%s\",\"model\":{\"sha256\":\"%s\","
               "\"model_key\":\"%016llx\"},\"n_ctx\":%d,\"kv_type\":\"%s\","
               "\"n_prompt\":%d,\"n_tokens\":%d,\"max_new\":%d,\"generated\":%d,"
               "\"sampler\":{\"temp\":%.9g,\"top_k\":%d,\"top_p\":%.9g,"
               "\"min_p\":%.9g,\"repeat_penalty\":%.9g,\"rng\":\"%llu\"},"
               "\"constraint\":{\"json_mode\":%s,\"json_schema_sha256\":",
           SESS_SCHEMA, mt->model_sha256, (unsigned long long)mt->model_key,
           mt->n_ctx, mt->kv_type, mt->n_prompt, mt->n_tokens, mt->max_new,
           mt->generated, (double)mt->temp, mt->top_k, (double)mt->top_p,
           (double)mt->min_p, (double)mt->repeat_penalty,
           (unsigned long long)mt->rng, mt->json_mode ? "true" : "false");
    if (mt->schema_sha256[0]) sb_fmt(&h, "\"%s\"", mt->schema_sha256);
    else sb_lit(&h, "null");
    sb_fmt(&h, "},\"ignore_eos\":%s,\"runner\":\"%s\",\"binary_sha256\":\"%s\"}",
           mt->ignore_eos ? "true" : "false", RUNNER_VERSION, mt->binary_sha256);
    wbuf w = {0};
    uint32_t hl = (uint32_t)h.n, nt = (uint32_t)mt->n_tokens, nv = (uint32_t)n_vocab;
    uint64_t sl = sn;
    wput(&w, SESS_MAGIC, 16);
    wput(&w, &hl, 4);
    wput(&w, h.s, h.n);
    wput(&w, &nt, 4);
    wput(&w, e->hist, sizeof(int32_t) * (size_t)mt->n_tokens);
    wput(&w, &sl, 8);
    wput(&w, state, sn);
    wput(&w, &nv, 4);
    wput(&w, logits, sizeof(float) * (size_t)n_vocab);
    free(state);
    bool ok = !w.oom && !h.failed;
    free(h.s);
    if (ok) {
        uint8_t d[32];
        envelope_data_sha256_raw(w.b, w.n, d);
        wput(&w, d, 32);
        for (int i = 0; i < 32; i++) snprintf(sha_out + 2 * i, 3, "%02x", d[i]);
        ok = !w.oom;
    }
    if (!ok) {
        free(w.b);
        fprintf(stderr, "error: session: out of memory building the image\n");
        return false;
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_BINARY, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (fd >= 0 && !f) close(fd);
    if (!f) {
        free(w.b);
        fprintf(stderr, "error: session: %s %s\n", path,
                errno == EEXIST ? "already exists; an image is never overwritten"
                                : "cannot be created");
        return false;
    }
    ok = fwrite(w.b, 1, w.n, f) == w.n;
    ok = (fclose(f) == 0) && ok;
    free(w.b);
    if (!ok) {
        remove(path);
        fprintf(stderr, "error: session: cannot write %s\n", path);
    }
    return ok;
}

void session_image_free(session_image *img) {
    free(img->tokens);
    free(img->state);
    free(img->logits);
    memset(img, 0, sizeof *img);
}

static bool hex64(const char *s) {
    if (!s || strlen(s) != 64) return false;
    for (int i = 0; i < 64; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return false;
    return true;
}

bool session_read(const char *path, session_image *img) {
    memset(img, 0, sizeof *img);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: session: cannot read %s\n", path); return false; }
    uint8_t *b = NULL;
    size_t n = 0, cap = 0, got;
    uint8_t chunk[1 << 16];
    while ((got = fread(chunk, 1, sizeof chunk, f)) > 0) {
        if (n + got > cap) {
            cap = cap ? cap * 2 : 1 << 20;
            while (cap < n + got) cap *= 2;
            uint8_t *g = realloc(b, cap);
            if (!g) { free(b); fclose(f); fprintf(stderr, "error: session: out of memory\n"); return false; }
            b = g;
        }
        memcpy(b + n, chunk, got);
        n += got;
    }
    fclose(f);
    const char *why = NULL;
    jv *h = NULL;
    size_t off = 0;
#define NEED(k) do { if (off + (k) > n - 32) { why = "truncated"; goto out; } } while (0)
    if (n < 16 + 4 + 32 || memcmp(b, SESS_MAGIC, 16) != 0) { why = "not a session image"; goto out; }
    {
        uint8_t d[32];
        envelope_data_sha256_raw(b, n - 32, d);
        if (memcmp(d, b + n - 32, 32) != 0) { why = "its SHA-256 trailer does not match its bytes"; goto out; }
        for (int i = 0; i < 32; i++) snprintf(img->sha256 + 2 * i, 3, "%02x", d[i]);
    }
    off = 16;
    uint32_t hl, nt, nv;
    uint64_t sl;
    NEED(4); memcpy(&hl, b + off, 4); off += 4;
    NEED(hl);
    h = json_parse((const char *)b + off, hl);
    off += hl;
    if (!h || strcmp(jv_str(jv_get(h, "schema_version"), ""), SESS_SCHEMA) != 0) {
        why = "its header is not " SESS_SCHEMA; goto out;
    }
    session_meta *m = &img->meta;
    jv *model = jv_get(h, "model"), *smp = jv_get(h, "sampler"), *con = jv_get(h, "constraint");
    const char *msha = jv_str(jv_get(model, "sha256"), "");
    const char *mkey = jv_str(jv_get(model, "model_key"), "");
    const char *rng = jv_str(jv_get(smp, "rng"), "");
    const char *ssha = jv_str(jv_get(con, "json_schema_sha256"), "");
    const char *kv = jv_str(jv_get(h, "kv_type"), "");
    if (!hex64(msha) || strlen(mkey) != 16 || !*rng || strlen(kv) >= sizeof m->kv_type ||
        (*ssha && !hex64(ssha))) { why = "its header is malformed"; goto out; }
    snprintf(m->model_sha256, sizeof m->model_sha256, "%s", msha);
    m->model_key = strtoull(mkey, NULL, 16);
    m->rng = strtoull(rng, NULL, 10);
    snprintf(m->kv_type, sizeof m->kv_type, "%s", kv);
    snprintf(m->schema_sha256, sizeof m->schema_sha256, "%s", ssha);
    snprintf(m->binary_sha256, sizeof m->binary_sha256, "%s",
             jv_str(jv_get(h, "binary_sha256"), ""));
    m->n_ctx = (int)jv_num(jv_get(h, "n_ctx"), -1);
    m->n_prompt = (int)jv_num(jv_get(h, "n_prompt"), -1);
    m->n_tokens = (int)jv_num(jv_get(h, "n_tokens"), -1);
    m->max_new = (int)jv_num(jv_get(h, "max_new"), -1);
    m->generated = (int)jv_num(jv_get(h, "generated"), -1);
    m->temp = (float)jv_num(jv_get(smp, "temp"), 0);
    m->top_k = (int)jv_num(jv_get(smp, "top_k"), 0);
    m->top_p = (float)jv_num(jv_get(smp, "top_p"), 1);
    m->min_p = (float)jv_num(jv_get(smp, "min_p"), 0);
    m->repeat_penalty = (float)jv_num(jv_get(smp, "repeat_penalty"), 1);
    jv *jm = jv_get(con, "json_mode"), *ie = jv_get(h, "ignore_eos");
    m->json_mode = jm && jm->type == J_BOOL && jm->b;
    m->ignore_eos = ie && ie->type == J_BOOL && ie->b;
    if (m->n_ctx < 1 || m->n_prompt < 1 || m->n_tokens < m->n_prompt ||
        m->n_tokens > m->n_ctx || m->generated != m->n_tokens - m->n_prompt ||
        m->max_new < m->generated) { why = "its counts are inconsistent"; goto out; }
    NEED(4); memcpy(&nt, b + off, 4); off += 4;
    if ((int)nt != m->n_tokens) { why = "its token count disagrees with its header"; goto out; }
    NEED(4 * (size_t)nt);
    img->tokens = malloc(4 * (size_t)nt);
    if (!img->tokens) { why = "out of memory"; goto out; }
    memcpy(img->tokens, b + off, 4 * (size_t)nt); off += 4 * (size_t)nt;
    NEED(8); memcpy(&sl, b + off, 8); off += 8;
    if (sl > (uint64_t)1 << 40) { why = "its state is implausibly large"; goto out; }
    NEED((size_t)sl);
    img->state = malloc((size_t)sl ? (size_t)sl : 1);
    if (!img->state) { why = "out of memory"; goto out; }
    memcpy(img->state, b + off, (size_t)sl); off += (size_t)sl;
    img->state_n = (size_t)sl;
    NEED(4); memcpy(&nv, b + off, 4); off += 4;
    NEED(4 * (size_t)nv);
    img->logits = malloc(nv ? 4 * (size_t)nv : 1);
    if (!img->logits) { why = "out of memory"; goto out; }
    memcpy(img->logits, b + off, 4 * (size_t)nv); off += 4 * (size_t)nv;
    img->n_vocab = (int)nv;
    if (off != n - 32) why = "it has bytes after its last section";
#undef NEED
out:
    jv_free(h);
    free(b);
    if (why) {
        fprintf(stderr, "error: session: %s: %s\n", path, why);
        session_image_free(img);
        return false;
    }
    return true;
}


bool session_run(engine *e, float *logits, int stop_at, gen_cb cb, void *ud,
                 const float **last) {
    int32_t tok;
    int pos;
    while (stop_at <= 0 || e->gen_count < stop_at) {
        if (engine_gen_step(e, logits, cb, ud, &tok, &pos) != ENGINE_STEP_MORE) break;
        logits = model_forward(e->m, tok, pos);
        if (!logits) { e->oom = true; break; }
    }
    e->pending_pos = -1;
    *last = logits;
    return logits && !e->hit_stop && !e->oom;
}

bool session_image_write(const char *path, const engine *e, const char *model_path,
                   const char *model_sha256, const char *binary_sha256,
                   const float *logits, int n_prompt, int max_new,
                   bool json_mode, bool ignore_eos, const char *schema_sha256,
                   char sha_out[65]) {
    session_meta mt;
    memset(&mt, 0, sizeof mt);
    if (model_sha256 && strlen(model_sha256) == 64) {
        memcpy(mt.model_sha256, model_sha256, 65);
    } else if (!envelope_file_sha256(model_path, mt.model_sha256)) {
        fprintf(stderr, "error: session: cannot hash %s\n", model_path);
        return false;
    }
    if (binary_sha256 && strlen(binary_sha256) == 64) {
        memcpy(mt.binary_sha256, binary_sha256, 65);
    } else {
        char *exe = plat_executable_path();
        if (exe) envelope_file_sha256(exe, mt.binary_sha256);
        free(exe);
    }
    if (schema_sha256 && *schema_sha256)
        snprintf(mt.schema_sha256, sizeof mt.schema_sha256, "%s", schema_sha256);
    const model_t *m = e->m;
    mt.model_key = e->model_key;
    mt.n_ctx = m->n_ctx;
    snprintf(mt.kv_type, sizeof mt.kv_type, "%s",
             m->kv_fp4 ? "fp4" : m->kv_split ? "k8v4" : m->kv_q8 ? "q8" : "f16");
    mt.n_prompt = n_prompt;
    mt.n_tokens = e->pos;
    mt.max_new = max_new;
    mt.generated = e->gen_count;
    mt.temp = e->smp->temp; mt.top_k = e->smp->top_k; mt.top_p = e->smp->top_p;
    mt.min_p = e->smp->min_p; mt.repeat_penalty = e->smp->repeat_penalty;
    // A greedy generation never draws from the rng, and an unseeded run's
    // rng is the wall clock: recorded, it made two images of the same greedy
    // state differ whenever the runs straddled a second. The image holds no
    // timestamp, so a state that does not include the rng does not carry it.
    mt.rng = e->smp->temp > 0 ? e->smp->rng : 0;
    mt.json_mode = json_mode;
    mt.ignore_eos = ignore_eos;
    return session_write(path, e, &mt, logits, m->n_vocab, sha_out);
}


static char g_sessions_dir[1024];

bool sessions_configure(const char *dir) {
    if (!dir || !*dir || strlen(dir) >= sizeof g_sessions_dir - 80) {
        fprintf(stderr, "error: --sessions: a directory path is needed\n");
        return false;
    }
#ifdef _WIN32
    if (_mkdir(dir) != 0 && errno != EEXIST) {
#else
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) {
#endif
        fprintf(stderr, "error: --sessions %s: %s\n", dir, strerror(errno));
        return false;
    }
    snprintf(g_sessions_dir, sizeof g_sessions_dir, "%s", dir);
    return true;
}

const char *sessions_dir(void) {
    return g_sessions_dir[0] ? g_sessions_dir : NULL;
}
