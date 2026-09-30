// Tournament-sampling watermark (R1.8.1, R1.8.2). See watermark.h.
#include "watermark.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "envelope.h"
#include "json.h"

#ifndef O_BINARY
#define O_BINARY 0
#endif

static const char KEY_SCHEMA[] = "xyntetik.runner.watermark_key.v1";
static const char ID_DOMAIN[] = "xyntetik.runner.watermark.key_id.v1";
static const char CTX_DOMAIN[] = "xyntetik.wm.ctx.v1";

// ---- SipHash-2-4 (Aumasson & Bernstein), 64-bit output ------------------------

#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))
#define SIPROUND do {                                                      \
        v0 += v1; v1 = ROTL(v1, 13); v1 ^= v0; v0 = ROTL(v0, 32);          \
        v2 += v3; v3 = ROTL(v3, 16); v3 ^= v2;                             \
        v0 += v3; v3 = ROTL(v3, 21); v3 ^= v0;                             \
        v2 += v1; v1 = ROTL(v1, 17); v1 ^= v2; v2 = ROTL(v2, 32);          \
    } while (0)

static uint64_t le64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

uint64_t wm_siphash24(const uint64_t k[2], const uint8_t *m, size_t n) {
    uint64_t v0 = 0x736f6d6570736575ULL ^ k[0];
    uint64_t v1 = 0x646f72616e646f6dULL ^ k[1];
    uint64_t v2 = 0x6c7967656e657261ULL ^ k[0];
    uint64_t v3 = 0x7465646279746573ULL ^ k[1];
    size_t full = n - n % 8;
    for (size_t i = 0; i < full; i += 8) {
        uint64_t mi = le64(m + i);
        v3 ^= mi;
        SIPROUND; SIPROUND;
        v0 ^= mi;
    }
    uint64_t b = (uint64_t)n << 56;
    for (size_t i = 0; i < n % 8; i++) b |= (uint64_t)m[full + i] << (8 * i);
    v3 ^= b;
    SIPROUND; SIPROUND;
    v0 ^= b;
    v2 ^= 0xff;
    SIPROUND; SIPROUND; SIPROUND; SIPROUND;
    return v0 ^ v1 ^ v2 ^ v3;
}

// ---- keys ---------------------------------------------------------------------

void wm_key_id(const uint8_t key[32], char id[17]) {
    uint8_t buf[sizeof ID_DOMAIN - 1 + 32], d[32];
    memcpy(buf, ID_DOMAIN, sizeof ID_DOMAIN - 1);
    memcpy(buf + sizeof ID_DOMAIN - 1, key, 32);
    envelope_data_sha256_raw(buf, sizeof buf, d);
    for (int i = 0; i < 8; i++) snprintf(id + 2 * i, 3, "%02x", d[i]);
    memset(buf, 0, sizeof buf);
}

bool wm_key_write(const char *path, const uint8_t key[32], char id_out[17]) {
    char hex[65];
    for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", key[i]);
    wm_key_id(key, id_out);
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_BINARY, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (fd >= 0 && !f) close(fd);
    bool ok = f != NULL;
    if (ok) {
        ok = fprintf(f, "{\"schema_version\":\"%s\",\"key\":\"%s\",\"key_id\":\"%s\"}\n",
                     KEY_SCHEMA, hex, id_out) > 0;
        ok = (fclose(f) == 0) && ok;
        if (!ok) remove(path);
    }
    memset(hex, 0, sizeof hex);
    return ok;
}

static int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool wm_key_load(const char *path, wm_key *k, char *err, size_t cap) {
    memset(k, 0, sizeof *k);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, cap, "cannot read %s", path); return false; }
    char buf[1024];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    jv *j = json_parse(buf, n);
    memset(buf, 0, sizeof buf);
    const char *sv = jv_str(jv_get(j, "schema_version"), "");
    const char *kh = jv_str(jv_get(j, "key"), "");
    const char *id = jv_str(jv_get(j, "key_id"), "");
    bool ok = j && strcmp(sv, KEY_SCHEMA) == 0 && strlen(kh) == 64;
    for (int i = 0; ok && i < 32; i++) {
        int a = hexv(kh[2 * i]), b = hexv(kh[2 * i + 1]);
        if (a < 0 || b < 0) ok = false;
        else k->key[i] = (uint8_t)(a * 16 + b);
    }
    if (!ok) snprintf(err, cap, "%s is not a %s file", path, KEY_SCHEMA);
    if (ok) {
        wm_key_id(k->key, k->id);
        if (strcmp(id, k->id) != 0) {
            snprintf(err, cap, "%s: key_id %s does not name its key (that key's "
                     "id is %s)", path, id, k->id);
            ok = false;
        }
    }
    jv_free(j);
    if (!ok) memset(k, 0, sizeof *k);
    return ok;
}

// ---- g-values -----------------------------------------------------------------

void wm_seed(const wm_key *k, const int32_t *toks, int t, uint64_t sk[2]) {
    int lo = t > WM_CONTEXT ? t - WM_CONTEXT : 0, n = t - lo;
    uint8_t buf[32 + sizeof CTX_DOMAIN - 1 + 1 + 4 * WM_CONTEXT], d[32];
    size_t b = 0;
    memcpy(buf, k->key, 32); b += 32;
    memcpy(buf + b, CTX_DOMAIN, sizeof CTX_DOMAIN - 1); b += sizeof CTX_DOMAIN - 1;
    buf[b++] = (uint8_t)n;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)toks[lo + i];
        for (int j = 0; j < 4; j++) buf[b++] = (uint8_t)(v >> (8 * j));
    }
    envelope_data_sha256_raw(buf, b, d);
    sk[0] = le64(d);
    sk[1] = le64(d + 8);
    memset(buf, 0, sizeof buf);
}

uint64_t wm_gword(const uint64_t sk[2], int32_t tok) {
    uint8_t m[4];
    uint32_t v = (uint32_t)tok;
    for (int j = 0; j < 4; j++) m[j] = (uint8_t)(v >> (8 * j));
    return wm_siphash24(sk, m, 4);
}

bool wm_context_repeats(const int32_t *toks, int start, int t) {
    int n = t > WM_CONTEXT ? WM_CONTEXT : t;
    for (int u = start; u < t; u++) {
        int nu = u > WM_CONTEXT ? WM_CONTEXT : u;
        if (nu != n || (n && toks[u - 1] != toks[t - 1])) continue;
        if (memcmp(toks + u - n, toks + t - n, sizeof(int32_t) * (size_t)n) == 0)
            return true;
    }
    return false;
}

bool wm_reweight(const uint64_t sk[2], const int32_t *ids, float *p, int n,
                 int layers) {
    if (n <= 1) return true;
    uint64_t stackg[256];
    double stackp[256];
    uint64_t *g = n <= 256 ? stackg : malloc(sizeof *g * (size_t)n);
    double *q = n <= 256 ? stackp : malloc(sizeof *q * (size_t)n);
    if (!g || !q) {
        if (g != stackg) free(g);
        if (q != stackp) free(q);
        return false;
    }
    for (int i = 0; i < n; i++) { g[i] = wm_gword(sk, ids[i]); q[i] = p[i]; }
    for (int l = 0; l < layers; l++) {
        double G = 0;
        for (int i = 0; i < n; i++) if ((g[i] >> l) & 1) G += q[i];
        for (int i = 0; i < n; i++) q[i] *= 1.0 + (double)((g[i] >> l) & 1) - G;
    }
    for (int i = 0; i < n; i++) p[i] = (float)q[i];
    if (g != stackg) free(g);
    if (q != stackp) free(q);
    return true;
}

// ---- the engine hook ------------------------------------------------------------

static bool wm_reweight_cb(void *ud, const int32_t *ids, float *p, int n) {
    wm_state *w = ud;
    if (!wm_reweight(w->sk, ids, p, n, WM_LAYERS)) return false;
    w->marked++;
    return true;
}

void wm_prepare(void *ud, sampler *s, const int32_t *hist, int start, int t) {
    wm_state *w = ud;
    s->reweight = NULL;
    s->reweight_ud = NULL;
    if (!w || !w->key || !hist || t < 1) return;
    if (wm_context_repeats(hist, start, t)) { w->repeats++; return; }
    wm_seed(w->key, hist, t, w->sk);
    s->reweight = wm_reweight_cb;
    s->reweight_ud = w;
}

// ---- detection ------------------------------------------------------------------

void wm_detect(const wm_key *k, const int32_t *toks, int n, int start,
               wm_score *out) {
    memset(out, 0, sizeof *out);
    if (start < 0) start = 0;
    const uint64_t mask = (1ULL << WM_LAYERS) - 1;
    for (int t = start; t < n; t++) {
        out->tokens++;
        if (wm_context_repeats(toks, start, t)) continue;
        uint64_t sk[2];
        wm_seed(k, toks, t, sk);
        uint64_t w = wm_gword(sk, toks[t]) & mask;
        int ones = 0;
        while (w) { ones += (int)(w & 1); w >>= 1; }
        out->g_sum += ones;
        out->g_n += WM_LAYERS;
        out->scored++;
    }
    if (out->g_n > 0) {
        out->mean = (double)out->g_sum / (double)out->g_n;
        out->z = ((double)out->g_sum - out->g_n / 2.0) / sqrt(out->g_n / 4.0);
        out->p_value = 0.5 * erfc(out->z / sqrt(2.0));
    } else {
        out->p_value = 1;
    }
}

const char *wm_verdict(const wm_score *s) {
    if (s->scored < WM_MIN_SCORED) return "INSUFFICIENT";
    return s->z >= WM_Z_DETECT ? "WATERMARKED" : "NOT_DETECTED";
}
