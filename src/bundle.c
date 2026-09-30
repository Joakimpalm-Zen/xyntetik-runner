// Receipt bundles (R1.2.1). See bundle.h.
#include "bundle.h"
#include "envelope.h"
#include "json.h"
#include "oms.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#define bundle_mkdir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define bundle_mkdir(p) mkdir(p, 0700)
#endif

#define BUNDLE_SCHEMA "xyntetik.runner.bundle.v1"
#define BUNDLE_MAX_FILE (256u * 1024u * 1024u)

static char *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || (unsigned long)sz > BUNDLE_MAX_FILE || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return NULL;
    }
    char *b = malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)sz, f);
    bool err = ferror(f) != 0;
    fclose(f);
    if (err || got != (size_t)sz) { free(b); return NULL; }
    b[got] = 0;
    *n = got;
    return b;
}

static bool spill(const char *path, const void *data, size_t n) {
    FILE *f = fopen(path, "wb");
    bool ok = f && fwrite(data, 1, n, f) == n && fflush(f) == 0;
    if (f && fclose(f) != 0) ok = false;
    return ok;
}

static void join(char *out, size_t cap, const char *dir, const char *name) {
    size_t dn = strlen(dir);
    bool slash = dn && (dir[dn - 1] == '/' || dir[dn - 1] == '\\');
    snprintf(out, cap, "%s%s%s", dir, slash ? "" : "/", name);
}

static const char *base_of(const char *p) {
    const char *b = p;
    for (const char *c = p; *c; c++)
        if (*c == '/' || *c == '\\') b = c + 1;
    return b;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// sha256 of the key bytes a hex string spells: the receipt key's fingerprint
static bool key_fingerprint(const char *hex, char out[65]) {
    size_t n = strlen(hex);
    if (n == 0 || n % 2) return false;
    uint8_t *b = malloc(n / 2);
    if (!b) return false;
    for (size_t i = 0; i < n / 2; i++) {
        int hi = hexval(hex[2 * i]), lo = hexval(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) { free(b); return false; }
        b[i] = (uint8_t)(hi << 4 | lo);
    }
    envelope_data_sha256(b, n / 2, out);
    free(b);
    return true;
}

// A record's chain hash, recomputed: sha256 of every byte before the LAST
// `,"chain":`. False when the record has no chain object.
static bool chain_recompute(const char *rec, size_t n, char out[65]) {
    const char *pos = NULL, *scan = rec;
    while ((scan = strstr(scan, ",\"chain\":")) != NULL) { pos = scan; scan++; }
    if (!pos || (size_t)(pos - rec) > n) return false;
    envelope_data_sha256(rec, (size_t)(pos - rec), out);
    return true;
}

static void put_str(sbuf *b, const char *s) {
    if (!s) { sb_lit(b, "null"); return; }
    sb_lit(b, "\"");
    sb_esc(b, s, strlen(s));
    sb_lit(b, "\"");
}

// ------------------------------------------------------------------ export

int bundle_export(const bundle_opts *o) {
    size_t rn = 0;
    char *rec = slurp(o->receipt, &rn);
    if (!rec) {
        fprintf(stderr, "error: bundle: cannot read receipt %s\n", o->receipt);
        return 1;
    }
    int rc = 1;
    char *sig = NULL, *pub = NULL;
    jv *j = json_parse(rec, rn);
    const char *schema = jv_str(jv_get(j, "schema_version"), "");
    const char *stated = jv_str(jv_get(jv_get(j, "chain"), "hash"), NULL);
    char chain[65];
    char pk[SIGN_PUBHEX_CAP] = "", fp[65] = "";
    const char *algo = NULL;
    if (!j || strcmp(schema, "xyntetik.runner.transcript.v1")) {
        fprintf(stderr, "error: bundle: %s is not a transcript receipt "
                "(xyntetik.runner.transcript.v1)\n", o->receipt);
        goto out;
    }
    // A bundle vouches for the receipt it carries; one whose own chain does
    // not recompute, or whose signature does not verify, is refused rather
    // than packaged into something that looks checked.
    if (!stated || !chain_recompute(rec, rn, chain) || strcmp(chain, stated)) {
        fprintf(stderr, "error: bundle: the receipt's chain hash does not "
                "recompute; refusing to bundle it\n");
        goto out;
    }
    receipt_sig_state st = receipt_signature_check(rec, rn, pk);
    if (st == RSIG_BAD || st == RSIG_MALFORMED) {
        fprintf(stderr, "error: bundle: the receipt's signature does not "
                "verify; refusing to bundle it\n");
        goto out;
    }
    if (st == RSIG_OK) {
        algo = jv_str(jv_get(jv_get(j, "signature"), "algo"), NULL);
        if (!key_fingerprint(pk, fp)) goto out;
    }
    if (bundle_mkdir(o->out_dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "error: bundle: cannot create %s\n", o->out_dir);
        goto out;
    }
    char p_manifest[4096], p_rec[4096], p_sig[4096], p_pub[4096];
    join(p_manifest, sizeof p_manifest, o->out_dir, "bundle.json");
    join(p_rec, sizeof p_rec, o->out_dir, "receipt.json");
    join(p_sig, sizeof p_sig, o->out_dir, "model.sig");
    join(p_pub, sizeof p_pub, o->out_dir, "model-pubkey.pem");
    FILE *exists = fopen(p_manifest, "rb");
    if (exists) {
        fclose(exists);
        fprintf(stderr, "error: bundle: %s already holds a bundle\n", o->out_dir);
        goto out;
    }
    size_t sn = 0, pn = 0;
    if (o->model_sig && !(sig = slurp(o->model_sig, &sn))) {
        fprintf(stderr, "error: bundle: cannot read %s\n", o->model_sig);
        goto out;
    }
    char key_fp[65] = "";
    if (o->model_pubkey) {
        if (!(pub = slurp(o->model_pubkey, &pn)) ||
            !oms_pubkey_fingerprint(o->model_pubkey, key_fp)) {
            fprintf(stderr, "error: bundle: %s is not a readable PEM public "
                    "key\n", o->model_pubkey);
            goto out;
        }
    }
    char h_rec[65], h_sig[65] = "", h_pub[65] = "";
    envelope_data_sha256(rec, rn, h_rec);
    if (sig) envelope_data_sha256(sig, sn, h_sig);
    if (pub) envelope_data_sha256(pub, pn, h_pub);
    if (!spill(p_rec, rec, rn) || (sig && !spill(p_sig, sig, sn)) ||
        (pub && !spill(p_pub, pub, pn))) {
        fprintf(stderr, "error: bundle: cannot write into %s\n", o->out_dir);
        remove(p_rec); remove(p_sig); remove(p_pub);
        goto out;
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

    jv *model = jv_get(j, "model"), *build = jv_get(j, "build");
    sbuf m = {0};
    sb_lit(&m, "{\"schema_version\":\"" BUNDLE_SCHEMA "\",\"created_utc\":");
    put_str(&m, utc);
    sb_fmt(&m, ",\"files\":[{\"name\":\"receipt.json\",\"sha256\":\"%s\","
               "\"bytes\":%zu}", h_rec, rn);
    if (sig) sb_fmt(&m, ",{\"name\":\"model.sig\",\"sha256\":\"%s\",\"bytes\":%zu}",
                    h_sig, sn);
    if (pub) sb_fmt(&m, ",{\"name\":\"model-pubkey.pem\",\"sha256\":\"%s\","
                        "\"bytes\":%zu}", h_pub, pn);
    sb_lit(&m, "],\"receipt\":{\"file\":\"receipt.json\",\"chain_hash\":");
    put_str(&m, chain);
    sb_lit(&m, ",\"chain_prev\":");
    put_str(&m, jv_str(jv_get(jv_get(j, "chain"), "prev"), NULL));
    // never a key spelled "signature" inside the manifest: a signed record's
    // own signature is its LAST `,"signature"` (envelope.c), so a nested one
    // would be read as the manifest's
    sb_lit(&m, ",\"signed_by\":");
    if (st == RSIG_OK) {
        sb_lit(&m, "{\"algo\":");
        put_str(&m, algo);
        sb_lit(&m, ",\"public_key\":");
        put_str(&m, pk);
        sb_lit(&m, ",\"fingerprint\":");
        put_str(&m, fp);
        sb_lit(&m, "}");
    } else {
        sb_lit(&m, "null");
    }
    sb_lit(&m, "},\"model\":{\"name\":");
    put_str(&m, base_of(jv_str(jv_get(model, "path"), "")));
    sb_lit(&m, ",\"sha256\":");
    put_str(&m, jv_str(jv_get(model, "sha256"), NULL));
    sb_lit(&m, ",\"oms\":");
    jv *ms = jv_get(j, "model_signature");
    if (ms) jv_dump(ms, &m);
    else    sb_lit(&m, "null");
    sb_lit(&m, ",\"signature_file\":");
    put_str(&m, sig ? "model.sig" : NULL);
    sb_lit(&m, ",\"key_file\":");
    put_str(&m, pub ? "model-pubkey.pem" : NULL);
    sb_lit(&m, ",\"key_fingerprint\":");
    put_str(&m, pub ? key_fp : NULL);
    sb_lit(&m, "},\"binary\":{\"runner\":");
    put_str(&m, jv_str(jv_get(j, "runner"), NULL));
    sb_lit(&m, ",\"sha256\":");
    put_str(&m, jv_str(jv_get(build, "binary_sha256"), NULL));
    sb_lit(&m, ",\"os\":");
    put_str(&m, jv_str(jv_get(build, "os"), NULL));
    sb_lit(&m, ",\"arch\":");
    put_str(&m, jv_str(jv_get(build, "arch"), NULL));
    sb_lit(&m, "},\"check\":\"runner --check-bundle DIR\",\"replay\":");
    {
        sbuf r = {0};
        sb_lit(&r, "runner -m <the GGUF whose sha256 is model.sha256> --verify "
                   "DIR/receipt.json");
        if (st == RSIG_OK) { sb_lit(&r, " --trust-key "); sb_put(&r, pk, strlen(pk)); }
        sb_put(&r, "", 1);
        put_str(&m, r.failed ? "" : r.s);
        free(r.s);
    }
    sb_lit(&m, ",\"statement\":\"--check-bundle proves these files are the "
               "ones listed, the receipt's chain recomputes and its signature "
               "verifies with the key named here. It does not replay the "
               "inference: that takes the model and the binary named by digest, "
               "and --verify.\"}\n");
    if (m.failed || !spill(p_manifest, m.s, m.n)) {
        fprintf(stderr, "error: bundle: cannot write %s\n", p_manifest);
        free(m.s);
        remove(p_rec); remove(p_sig); remove(p_pub);
        goto out;
    }
    free(m.s);
    if (o->sign_key && !record_sign(p_manifest, o->sign_key, NULL)) {
        remove(p_manifest); remove(p_rec); remove(p_sig); remove(p_pub);
        goto out;
    }
    fprintf(stderr, "bundle -> %s (receipt chain %.16s…%s%s)\n", o->out_dir,
            chain, st == RSIG_OK ? ", signed" : ", unsigned",
            o->sign_key ? ", manifest signed" : "");
    rc = 0;
out:
    jv_free(j);
    free(rec); free(sig); free(pub);
    return rc;
}

// ------------------------------------------------------------------- check

static int bad(const char *dir, const char *why, const char *what) {
    printf("BAD: bundle %s: %s%s%s\n", dir, why, what ? ": " : "", what ? what : "");
    return 2;
}

static bool plain_name(const char *n) {
    return n && *n && !strchr(n, '/') && !strchr(n, '\\') && strcmp(n, ".") &&
           strcmp(n, "..");
}

int bundle_check(const char *dir, const char *trust_hex) {
    char path[4096];
    join(path, sizeof path, dir, "bundle.json");
    size_t mn = 0;
    char *man = slurp(path, &mn);
    if (!man) {
        printf("UNVERIFIABLE: bundle %s: no readable bundle.json\n", dir);
        return 3;
    }
    int rc = 2;
    char *rec = NULL;
    jv *m = json_parse(man, mn), *r = NULL;
    if (!m || strcmp(jv_str(jv_get(m, "schema_version"), ""), BUNDLE_SCHEMA)) {
        printf("UNVERIFIABLE: bundle %s: bundle.json is not %s\n", dir, BUNDLE_SCHEMA);
        rc = 3;
        goto out;
    }
    // the manifest's own signature, when it has one: over every byte before
    // its `,"signature"`, like any signed record
    char mpk[SIGN_PUBHEX_CAP] = "";
    receipt_sig_state mst = receipt_signature_check(man, mn, mpk);
    if (mst == RSIG_BAD || mst == RSIG_MALFORMED) {
        rc = bad(dir, "the manifest's signature does not verify", NULL);
        goto out;
    }
    if (mst == RSIG_OK) {
        char mc[65];
        const char *ms = jv_str(jv_get(jv_get(m, "chain"), "hash"), "");
        if (!chain_recompute(man, mn, mc) || strcmp(mc, ms)) {
            rc = bad(dir, "the manifest's chain hash does not recompute", NULL);
            goto out;
        }
    }
    if (trust_hex && *trust_hex && (mst != RSIG_OK || strcmp(trust_hex, mpk))) {
        rc = bad(dir, mst == RSIG_OK ? "the manifest is signed by another key"
                                     : "the manifest is unsigned", trust_hex);
        goto out;
    }
    // every listed file is the one listed
    jv *files = jv_get(m, "files");
    if (!files || files->type != J_ARR || files->n == 0) {
        rc = bad(dir, "the manifest lists no files", NULL);
        goto out;
    }
    for (int i = 0; i < files->n; i++) {
        const char *name = jv_str(jv_get(files->items[i], "name"), NULL);
        const char *want = jv_str(jv_get(files->items[i], "sha256"), "");
        if (!plain_name(name)) { rc = bad(dir, "a listed file name is not a plain name", name); goto out; }
        char fp[4096], have[65];
        join(fp, sizeof fp, dir, name);
        if (!envelope_file_sha256(fp, have)) { rc = bad(dir, "a listed file is missing", name); goto out; }
        if (strcmp(have, want)) { rc = bad(dir, "a file differs from its listed sha256", name); goto out; }
    }
    // the receipt: its chain recomputes, its signature is the one named
    jv *mr = jv_get(m, "receipt");
    const char *rname = jv_str(jv_get(mr, "file"), NULL);
    if (!plain_name(rname)) { rc = bad(dir, "no receipt file named", NULL); goto out; }
    join(path, sizeof path, dir, rname);
    size_t rn = 0;
    if (!(rec = slurp(path, &rn))) { rc = bad(dir, "the receipt is unreadable", rname); goto out; }
    char chain[65];
    const char *mchain = jv_str(jv_get(mr, "chain_hash"), "");
    r = json_parse(rec, rn);
    const char *rchain = jv_str(jv_get(jv_get(r, "chain"), "hash"), "");
    if (!chain_recompute(rec, rn, chain) || strcmp(chain, rchain) ||
        strcmp(chain, mchain)) {
        rc = bad(dir, "the receipt's chain hash does not recompute", NULL);
        goto out;
    }
    char pk[SIGN_PUBHEX_CAP] = "";
    receipt_sig_state st = receipt_signature_check(rec, rn, pk);
    jv *msig = jv_get(mr, "signed_by");
    bool named = msig && msig->type == J_OBJ;
    if (st == RSIG_BAD || st == RSIG_MALFORMED) {
        rc = bad(dir, "the receipt's signature does not verify", NULL);
        goto out;
    }
    if (named != (st == RSIG_OK) ||
        (named && strcmp(pk, jv_str(jv_get(msig, "public_key"), "")))) {
        rc = bad(dir, "the receipt's signature is not the one the manifest names", NULL);
        goto out;
    }
    char fp[65];
    if (named && (!key_fingerprint(pk, fp) ||
                  strcmp(fp, jv_str(jv_get(msig, "fingerprint"), "")))) {
        rc = bad(dir, "the receipt key's fingerprint does not recompute", NULL);
        goto out;
    }
    // the model key's fingerprint, and the digests the replay will need
    jv *mm = jv_get(m, "model");
    const char *kf = jv_str(jv_get(mm, "key_file"), NULL);
    if (kf) {
        char kp[4096], kfp[65];
        join(kp, sizeof kp, dir, kf);
        if (!plain_name(kf) || !oms_pubkey_fingerprint(kp, kfp) ||
            strcmp(kfp, jv_str(jv_get(mm, "key_fingerprint"), ""))) {
            rc = bad(dir, "the model key's fingerprint does not recompute", kf);
            goto out;
        }
    }
    const char *msha = jv_str(jv_get(jv_get(r, "model"), "sha256"), "");
    const char *bsha = jv_str(jv_get(jv_get(r, "build"), "binary_sha256"), "");
    if (strcmp(msha, jv_str(jv_get(mm, "sha256"), "")) ||
        strcmp(bsha, jv_str(jv_get(jv_get(m, "binary"), "sha256"), ""))) {
        rc = bad(dir, "the manifest's model or binary digest is not the receipt's", NULL);
        goto out;
    }
    printf("OK: bundle %s: receipt chain %s, %s%s; manifest %s; replay needs "
           "model %s and binary %s\n", dir, chain,
           named ? "signed by key " : "unsigned", named ? fp : "",
           mst == RSIG_OK ? "signed" : "unsigned", msha, bsha);
    rc = 0;
out:
    jv_free(m);
    jv_free(r);
    free(man);
    free(rec);
    return rc;
}
