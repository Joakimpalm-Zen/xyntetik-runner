// Signed KV snapshots (R1.12.1, R1.12.2). See kvsnap.h.
#include "kvsnap.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#define ks_mkdir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define ks_mkdir(p) mkdir(p, 0700)
#endif

#include "envelope.h"
#include "provenance.h"
#include "receipts.h"
#include "runner.h"

#define KS_SCHEMA "xyntetik.runner.kv_snapshot.v1"
// the runner.prefix.v1 layout: magic, entry count, then the entry's model
// key, token count and KV size before its token ids
#define KS_TOKENS_AT (16 + 4 + 8 + 4 + 8)

static struct {
    bool on;
    char dir[1024];
    char sign_key[1024];
    // The key a loaded manifest must be signed with (R1.12.3): --trust-key,
    // else this server's own --sign-key. Hex of the public key, or
    // "sha256:" + the hex digest of its bytes; "" = no anchor.
    char trust[SIGN_PUBHEX_CAP + 8];
    pthread_mutex_t mu;   // one writer at a time: names are checked then made
} KS = { .mu = PTHREAD_MUTEX_INITIALIZER };

static bool hex_eq(const char *a, const char *b) {
    for (; *a && *b; a++, b++) {
        char x = *a >= 'A' && *a <= 'F' ? (char)(*a + 32) : *a;
        char y = *b >= 'A' && *b <= 'F' ? (char)(*b + 32) : *b;
        if (x != y) return false;
    }
    return !*a && !*b;
}

// Is `pub` (hex) the trusted key? The forms --verify's --trust-key takes.
static bool trusted_key(const char *pub) {
    if (strncmp(KS.trust, "sha256:", 7) != 0) return hex_eq(KS.trust, pub);
    uint8_t pkb[SIGN_PK_MAX];
    size_t hn = strlen(pub);
    if (hn % 2 || hn / 2 > sizeof pkb) return false;
    for (size_t i = 0; i < hn / 2; i++) {
        unsigned v = 0;
        if (sscanf(pub + 2 * i, "%2x", &v) != 1) return false;
        pkb[i] = (uint8_t)v;
    }
    char digest[65];
    envelope_data_sha256(pkb, hn / 2, digest);
    return hex_eq(KS.trust + 7, digest);
}

bool kvsnap_configure(const char *dir, const char *sign_key, const char *trust_key) {
    KS.on = false;
    KS.trust[0] = 0;
    if (trust_key && strlen(trust_key) >= sizeof KS.trust) {
        fprintf(stderr, "error: --kv-snapshots: --trust-key is not a public key "
                "or sha256:<digest>\n");
        return false;
    }
    if (trust_key && *trust_key) snprintf(KS.trust, sizeof KS.trust, "%s", trust_key);
    if (!dir || !*dir || strlen(dir) >= sizeof KS.dir) {
        fprintf(stderr, "error: --kv-snapshots needs a directory\n");
        return false;
    }
    snprintf(KS.dir, sizeof KS.dir, "%s", dir);
    size_t n = strlen(KS.dir);
    while (n > 1 && (KS.dir[n - 1] == '/' || KS.dir[n - 1] == '\\')) KS.dir[--n] = 0;
    if (ks_mkdir(KS.dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "error: --kv-snapshots: cannot create %s\n", KS.dir);
        return false;
    }
    KS.sign_key[0] = 0;
    if (sign_key && *sign_key) {
        signkey k;
        if (!signkey_load(sign_key, &k)) {
            fprintf(stderr, "error: --kv-snapshots: cannot load signing key %s\n",
                    sign_key);
            return false;
        }
        // a server that signs its snapshots trusts its own key by default:
        // without an anchor a manifest was only held to the key it named,
        // so an unsigned one, or one signed by anyone, loaded as well
        if (!KS.trust[0])
            for (size_t i = 0; i < k.pk_n && 2 * i + 2 < sizeof KS.trust; i++)
                snprintf(KS.trust + 2 * i, 3, "%02x", k.pk[i]);
        memset(&k, 0, sizeof k);
        snprintf(KS.sign_key, sizeof KS.sign_key, "%s", sign_key);
    }
    KS.on = true;
    fprintf(stderr, "kv snapshots: %s (%s manifests; loads %s)\n", KS.dir,
            KS.sign_key[0] ? "signed" : "unsigned",
            KS.trust[0] ? "only manifests signed by the trusted key"
                        : "any manifest that verifies against the key it names");
    return true;
}

bool kvsnap_enabled(void) { return KS.on; }

static bool fail(kvsnap_err *err, int status, const char *code, const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));
static bool fail(kvsnap_err *err, int status, const char *code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err->msg, sizeof err->msg, fmt, ap);
    va_end(ap);
    err->status = status;
    err->code = code;
    return false;
}

// a snapshot name is a plain file stem: [A-Za-z0-9._-], 1..64, no leading dot
static bool name_ok(const char *s) {
    size_t n = s ? strlen(s) : 0;
    if (n < 1 || n > 64 || s[0] == '.') return false;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return false;
    }
    return true;
}

static bool exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static char *slurp(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || sz > (16L << 20)) { fclose(f); return NULL; }
    rewind(f);
    char *b = malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t n = fread(b, 1, (size_t)sz, f);
    fclose(f);
    b[n] = 0;
    *n_out = n;
    return b;
}

// the chain object of a record recomputes over the bytes before it
static bool chain_ok(const char *buf, size_t n, char hash_out[65]) {
    const char *c = NULL;
    for (const char *p = buf; (p = strstr(p, ",\"chain\":")) != NULL; p++) c = p;
    if (!c) return false;
    jv *j = json_parse(buf, n);
    const char *h = jv_str(jv_get(jv_get(j, "chain"), "hash"), "");
    char want[65];
    envelope_data_sha256(buf, (size_t)(c - buf), want);
    bool ok = strlen(h) == 64 && strcmp(h, want) == 0;
    if (ok) memcpy(hash_out, want, 65);
    jv_free(j);
    return ok;
}

static const char *kv_type_of(const model_t *m) {
    return m->kv_fp4 ? "fp4" : m->kv_split ? "k8v4" : m->kv_q8 ? "q8" : "f16";
}

bool kvsnap_save(const engine *e, const char *ctx, jv *req, sbuf *out,
                 kvsnap_err *err) {
    if (!KS.on)
        return fail(err, 400, "snapshots_off", "KV snapshots are off: start the "
                    "server with --kv-snapshots DIR");
    const char *name = jv_str(jv_get(req, "name"), ctx);
    if (!name_ok(name))
        return fail(err, 400, "invalid_value", "a snapshot name is 1 to 64 "
                    "characters of [A-Za-z0-9._-], not starting with a dot");
    // the producer: a receipt this server wrote, whose chain recomputes
    const char *rfile = jv_str(jv_get(req, "receipt"), NULL);
    char rchain[65] = "";
    if (rfile) {
        char rpath[1200];
        if (!receipts_record_path(rfile, rpath, sizeof rpath))
            return fail(err, 400, "invalid_value", "receipt must name a "
                        "receipt-<sequence>.json this server writes (--receipts)");
        size_t rn = 0;
        char *rb = slurp(rpath, &rn);
        if (!rb) return fail(err, 404, "receipt_not_found", "no receipt %s", rfile);
        bool ok = chain_ok(rb, rn, rchain);
        free(rb);
        if (!ok)
            return fail(err, 409, "receipt_invalid", "receipt %s does not "
                        "recompute its own chain hash", rfile);
    }
    char mhex[65], bhex[65];
    if (!provenance_digests(mhex, bhex))
        return fail(err, 409, "model_digest_unavailable", "the served model's "
                    "digest is unavailable; a snapshot must name it");
    char kvpath[1200], manpath[1200], tmp[1300];
    snprintf(kvpath, sizeof kvpath, "%s/%s.kv", KS.dir, name);
    snprintf(manpath, sizeof manpath, "%s/%s.kv.json", KS.dir, name);
    snprintf(tmp, sizeof tmp, "%s.partial", manpath);
    pthread_mutex_lock(&KS.mu);
    bool ok = false;
    sbuf m = {0};
    do {
        if (exists(kvpath) || exists(manpath)) {
            fail(err, 409, "snapshot_exists", "a snapshot named %s already "
                 "exists; it is never overwritten", name);
            break;
        }
        int n = prefix_context_export(ctx, kvpath);
        if (n == PFX_CTX_UNKNOWN || n == PFX_CTX_BADNAME) {
            fail(err, 404, "context_not_found", "no context \"%s\" is pinned "
                 "(POST /v1/runner/contexts)", ctx);
            break;
        }
        if (n < 0) {
            fail(err, 500, "snapshot_write_failed", "cannot write %s.kv", name);
            break;
        }
        // describe the file as written: its digest, and the ids it carries
        char kvhex[65], thex[65];
        size_t kn = 0;
        if (!envelope_file_sha256(kvpath, kvhex)) {
            remove(kvpath);
            fail(err, 500, "snapshot_write_failed", "cannot read back %s.kv", name);
            break;
        }
        FILE *f = fopen(kvpath, "rb");
        int32_t *toks = malloc(sizeof(int32_t) * (size_t)n);
        bool tok_ok = f && toks && fseek(f, KS_TOKENS_AT, SEEK_SET) == 0 &&
                      fread(toks, sizeof(int32_t), (size_t)n, f) == (size_t)n;
        if (f) { fseek(f, 0, SEEK_END); kn = (size_t)ftell(f); fclose(f); }
        if (tok_ok) envelope_data_sha256(toks, sizeof(int32_t) * (size_t)n, thex);
        free(toks);
        if (!tok_ok) {
            remove(kvpath);
            fail(err, 500, "snapshot_write_failed", "cannot read back %s.kv", name);
            break;
        }
        char utc[32];
        time_t now = time(NULL);
        struct tm g;
#ifdef _WIN32
        gmtime_s(&g, &now);
#else
        gmtime_r(&now, &g);
#endif
        strftime(utc, sizeof utc, "%Y-%m-%dT%H:%M:%SZ", &g);
        sb_fmt(&m, "{\"schema_version\":\"%s\",\"name\":\"%s\",\"context\":\"",
               KS_SCHEMA, name);
        sb_esc(&m, ctx, strlen(ctx));
        sb_fmt(&m, "\",\"kv_file\":\"%s.kv\",\"kv_sha256\":\"%s\",\"kv_bytes\":%zu,"
                   "\"tokens\":%d,\"token_sha256\":\"%s\",\"model\":{\"sha256\":"
                   "\"%s\",\"model_key\":\"%016llx\"},\"kv_type\":\"%s\","
                   "\"n_ctx\":%d,\"binary_sha256\":\"%s\",\"runner\":\"%s\","
                   "\"producer\":{\"receipt\":",
               name, kvhex, kn, n, thex, mhex,
               (unsigned long long)e->model_key, kv_type_of(e->m), e->m->n_ctx,
               bhex, RUNNER_VERSION);
        if (rfile) sb_fmt(&m, "{\"file\":\"%s\",\"chain_hash\":\"%s\"}", rfile, rchain);
        else sb_lit(&m, "null");
        sb_fmt(&m, "},\"created_utc\":\"%s\"", utc);
        if (m.failed) {
            remove(kvpath);
            fail(err, 500, "out_of_memory", "out of memory");
            break;
        }
        // chained like any record; signed through the record signer when a
        // key was given, else the chain object is written here
        bool wrote;
        if (KS.sign_key[0]) {
            FILE *mf = fopen(tmp, "wb");
            wrote = mf && fwrite(m.s, 1, m.n, mf) == m.n && fputc('}', mf) != EOF;
            if (mf && fclose(mf) != 0) wrote = false;
            wrote = wrote && record_sign(tmp, KS.sign_key, NULL);
        } else {
            char chain[65];
            envelope_data_sha256(m.s, m.n, chain);
            sb_fmt(&m, ",\"chain\":{\"algo\":\"sha256\",\"prev\":\"\",\"hash\":"
                       "\"%s\"}}", chain);
            FILE *mf = fopen(tmp, "wb");
            wrote = !m.failed && mf && fwrite(m.s, 1, m.n, mf) == m.n;
            if (mf && fclose(mf) != 0) wrote = false;
        }
        if (!wrote || rename(tmp, manpath) != 0) {
            remove(tmp);
            remove(kvpath);
            fail(err, 500, "snapshot_write_failed", "cannot write %s.kv.json", name);
            break;
        }
        size_t mn = 0;
        char *mb = slurp(manpath, &mn);
        char mchain[65] = "";
        bool mok = mb && chain_ok(mb, mn, mchain);
        free(mb);
        if (!mok) {
            fail(err, 500, "snapshot_write_failed", "the manifest written does "
                 "not recompute");
            break;
        }
        sb_fmt(out, "{\"object\":\"runner.kv_snapshot\",\"name\":\"%s\","
                    "\"context\":\"", name);
        sb_esc(out, ctx, strlen(ctx));
        sb_fmt(out, "\",\"tokens\":%d,\"kv_sha256\":\"%s\",\"kv_bytes\":%zu,"
                    "\"manifest_chain_hash\":\"%s\",\"signed\":%s}",
               n, kvhex, kn, mchain, KS.sign_key[0] ? "true" : "false");
        ok = true;
        fprintf(stderr, "kv snapshot %s: context %s, %d tokens, kv sha256 %.16s\n",
                name, ctx, n, kvhex);
    } while (0);
    pthread_mutex_unlock(&KS.mu);
    free(m.s);
    return ok;
}

bool kvsnap_load(const engine *e, const char *id, const char *name, sbuf *out,
                 kvsnap_err *err) {
    if (!KS.on)
        return fail(err, 400, "snapshots_off", "KV snapshots are off: start the "
                    "server with --kv-snapshots DIR");
    if (!name_ok(name))
        return fail(err, 400, "invalid_value", "snapshot must name a snapshot in "
                    "the --kv-snapshots directory ([A-Za-z0-9._-], 1 to 64)");
    char kvpath[1200], manpath[1200];
    snprintf(kvpath, sizeof kvpath, "%s/%s.kv", KS.dir, name);
    snprintf(manpath, sizeof manpath, "%s/%s.kv.json", KS.dir, name);
    size_t mn = 0;
    char *mb = slurp(manpath, &mn);
    if (!mb) return fail(err, 404, "snapshot_not_found", "no snapshot %s", name);
    jv *man = json_parse(mb, mn);
    char mchain[65] = "";
    char pub[SIGN_PUBHEX_CAP] = "";
    bool ok = false;
    do {
        // the manifest first: it is the claim everything else is held to
        receipt_sig_state rs = RSIG_MALFORMED;
        if (!man || strcmp(jv_str(jv_get(man, "schema_version"), ""), KS_SCHEMA) ||
            !chain_ok(mb, mn, mchain) ||
            (rs = receipt_signature_check(mb, mn, pub)) == RSIG_BAD ||
            rs == RSIG_MALFORMED ||
            strcmp(jv_str(jv_get(man, "name"), ""), name) != 0) {
            fail(err, 409, "snapshot_record_invalid", "the manifest of %s does not "
                 "recompute its chain hash or verify its signature", name);
            break;
        }
        if (rs != RSIG_OK) pub[0] = 0;
        if (KS.trust[0] && (rs != RSIG_OK || !trusted_key(pub))) {
            fail(err, 409, "snapshot_untrusted", "snapshot %s is %s; this "
                 "server loads only snapshots signed by its trusted key "
                 "(--trust-key, else its own --sign-key)", name,
                 rs == RSIG_OK ? "signed by another key" : "unsigned");
            break;
        }
        char want_file[80];
        snprintf(want_file, sizeof want_file, "%s.kv", name);
        const char *kvsha = jv_str(jv_get(man, "kv_sha256"), "");
        if (strcmp(jv_str(jv_get(man, "kv_file"), ""), want_file) != 0 ||
            strlen(kvsha) != 64) {
            fail(err, 409, "snapshot_record_invalid", "the manifest of %s names "
                 "no KV file of its own", name);
            break;
        }
        char have[65];
        if (!envelope_file_sha256(kvpath, have)) {
            fail(err, 404, "snapshot_not_found", "the KV file of snapshot %s is "
                 "missing", name);
            break;
        }
        if (strcmp(have, kvsha) != 0) {
            fail(err, 409, "snapshot_digest_mismatch", "%s.kv is not the file its "
                 "manifest names (sha256 %.16s..., manifest %.16s...)", name,
                 have, kvsha);
            break;
        }
        char mhex[65], bhex[65];
        if (!provenance_digests(mhex, bhex)) {
            fail(err, 409, "model_digest_unavailable", "the served model's digest "
                 "is unavailable, so the snapshot's model cannot be checked");
            break;
        }
        const char *msha = jv_str(jv_get(jv_get(man, "model"), "sha256"), "");
        if (strcmp(msha, mhex) != 0) {
            fail(err, 409, "snapshot_model_mismatch", "snapshot %s was made with "
                 "model sha256 %.16s..., this server serves %.16s...", name,
                 msha, mhex);
            break;
        }
        const char *kt = jv_str(jv_get(man, "kv_type"), "");
        if (strcmp(kt, kv_type_of(e->m)) != 0) {
            fail(err, 409, "snapshot_kv_type_mismatch", "snapshot %s holds %s KV, "
                 "this server's cache is %s (--kv)", name, kt, kv_type_of(e->m));
            break;
        }
        char origin[256];
        snprintf(origin, sizeof origin, "{\"name\":\"%s\",\"kv_sha256\":\"%s\","
                 "\"manifest_chain_hash\":\"%s\"}", name, kvsha, mchain);
        int n = prefix_context_import(e, id, kvpath, origin);
        if (n == PFX_CTX_MISMATCH) {
            fail(err, 409, "snapshot_model_mismatch", "snapshot %s was made for "
                 "another model key or context length than this engine's", name);
            break;
        }
        if (n == PFX_CTX_NOSPACE) {
            fail(err, 507, "context_budget", "the prefix-cache budget cannot hold "
                 "this context beside the ones already pinned");
            break;
        }
        if (n == PFX_CTX_UNSUPPORTED) {
            fail(err, 409, "context_unsupported", "this model cannot fork a "
                 "pinned prefix (a ring or tied-V KV layout, or a recurrent "
                 "model on a device)");
            break;
        }
        if (n < 0) {
            fail(err, 409, "snapshot_corrupt", "%s.kv is not one whole "
                 "runner.prefix.v1 entry", name);
            break;
        }
        sb_lit(out, "{\"object\":\"runner.context\",\"id\":\"");
        sb_esc(out, id, strlen(id));
        sb_fmt(out, "\",\"tokens\":%d,\"bytes\":%zu,\"snapshot\":{\"name\":\"%s\","
                    "\"kv_sha256\":\"%s\",\"manifest_chain_hash\":\"%s\","
                    "\"signed_by\":", n, prefix_cache_entry_bytes(e->m, n), name,
               kvsha, mchain);
        if (pub[0]) sb_fmt(out, "\"%s\"}}", pub);
        else sb_lit(out, "null}}");
        ok = true;
        fprintf(stderr, "kv snapshot %s: loaded as context %s (%d tokens)\n",
                name, id, n);
    } while (0);
    jv_free(man);
    free(mb);
    return ok;
}
