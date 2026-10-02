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
#include <io.h>
#define bundle_mkdir(p) _mkdir(p)
#else
#include <dirent.h>
#include <sys/stat.h>
#define bundle_mkdir(p) mkdir(p, 0700)
#endif

#define BUNDLE_SCHEMA "xyntetik.runner.bundle.v1"
#define PACK_SCHEMA "xyntetik.runner.evidence-pack.v1"
#define PACK_MAX_RECEIPTS 100000
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
    // Nothing in the directory is overwritten, and so nothing this call did
    // not write is removed by a failure below: exporting a receipt into the
    // directory it lives in used to rewrite it onto itself and then delete
    // it when the manifest could not be signed.
    const char *members[] = { p_manifest, p_rec, p_sig, p_pub };
    for (int i = 0; i < 4; i++) {
        FILE *exists = fopen(members[i], "rb");
        if (!exists) continue;
        fclose(exists);
        fprintf(stderr, "error: bundle: %s already exists; a bundle is "
                "written into a directory that holds none of its files\n",
                members[i]);
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

// =========================================================== evidence packs
//
// R1.6.1. A bundle carries one receipt; a pack carries every receipt of a
// chain segment, so a reader can see that nothing between the first and the
// last was left out, plus the envelope manifest the receipts name by digest
// and any further records a reviewer needs (oversight decisions, a tool
// cassette) as opaque attachments listed by sha256. One id per inference:
// the receipt's chain hash.

static bool is_receipt_name(const char *n) {
    size_t len = strlen(n);
    return len > 13 && !strncmp(n, "receipt-", 8) && !strcmp(n + len - 5, ".json");
}

static int name_cmp(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

// receipt-*.json in dir, sorted by name (the server writes a zero-padded
// sequence, so name order is write order). Returns the count, -1 on error.
static int list_receipts(const char *dir, char ***out) {
    char **v = NULL;
    int n = 0, cap = 0;
#ifdef _WIN32
    char pat[4200];
    join(pat, sizeof pat, dir, "receipt-*.json");
    struct _finddata_t fd;
    intptr_t h = _findfirst(pat, &fd);
    if (h != -1) {
        do {
            const char *nm = fd.name;
#else
    DIR *d = opendir(dir);
    if (!d) return -1;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        {
            const char *nm = de->d_name;
#endif
            if (!is_receipt_name(nm)) continue;
            if (n == PACK_MAX_RECEIPTS) break;
            if (n == cap) {
                cap = cap ? cap * 2 : 64;
                char **g = realloc(v, sizeof *v * (size_t)cap);
                if (!g) break;
                v = g;
            }
            if (!(v[n] = strdup(nm))) break;
            n++;
#ifdef _WIN32
        } while (_findnext(h, &fd) == 0);
        _findclose(h);
    }
#else
        }
    }
    closedir(d);
#endif
    if (n) qsort(v, (size_t)n, sizeof *v, name_cmp);
    *out = v;
    return n;
}

static void free_names(char **v, int n) {
    for (int i = 0; i < n; i++) free(v[i]);
    free(v);
}

static const char *PACK_RESERVED[] = { "pack.json", "model.sig",
                                       "model-pubkey.pem", "envelope.json" };

static bool reserved_name(const char *n) {
    for (size_t i = 0; i < sizeof PACK_RESERVED / sizeof *PACK_RESERVED; i++)
        if (!strcmp(n, PACK_RESERVED[i])) return true;
    return is_receipt_name(n);
}

// one receipt as the pack sees it
typedef struct {
    char *name, *bytes;
    size_t n;
    jv *j;
    char hash[65], fp[65];
    bool signed_ok;
} pack_rec;

int pack_export(const pack_opts *o) {
    int rc = 1, nr = 0;
    char **names = NULL;
    pack_rec *r = NULL;
    char *sig = NULL, *pub = NULL, *env = NULL;
    char *att[PACK_MAX_ATTACH] = {0};
    size_t attn[PACK_MAX_ATTACH] = {0};
    if (o->n_attach > PACK_MAX_ATTACH) {
        fprintf(stderr, "error: pack: at most %d attachments\n", PACK_MAX_ATTACH);
        return 1;
    }
    nr = list_receipts(o->receipts_dir, &names);
    if (nr <= 0) {
        fprintf(stderr, "error: pack: no receipt-*.json in %s\n", o->receipts_dir);
        return 1;
    }
    r = calloc((size_t)nr, sizeof *r);
    if (!r) goto out;
    const char *msha = NULL, *bsha = NULL, *esha = NULL;
    for (int i = 0; i < nr; i++) {
        char path[4200];
        join(path, sizeof path, o->receipts_dir, names[i]);
        r[i].name = names[i];
        if (!(r[i].bytes = slurp(path, &r[i].n)) ||
            !(r[i].j = json_parse(r[i].bytes, r[i].n)) ||
            strcmp(jv_str(jv_get(r[i].j, "schema_version"), ""),
                   "xyntetik.runner.transcript.v1")) {
            fprintf(stderr, "error: pack: %s is not a transcript receipt\n", names[i]);
            goto out;
        }
        const char *stated = jv_str(jv_get(jv_get(r[i].j, "chain"), "hash"), NULL);
        if (!stated || !chain_recompute(r[i].bytes, r[i].n, r[i].hash) ||
            strcmp(r[i].hash, stated)) {
            fprintf(stderr, "error: pack: %s: the chain hash does not recompute; "
                    "refusing to pack it\n", names[i]);
            goto out;
        }
        char pk[SIGN_PUBHEX_CAP] = "";
        receipt_sig_state st = receipt_signature_check(r[i].bytes, r[i].n, pk);
        if (st == RSIG_BAD || st == RSIG_MALFORMED) {
            fprintf(stderr, "error: pack: %s: the signature does not verify; "
                    "refusing to pack it\n", names[i]);
            goto out;
        }
        r[i].signed_ok = st == RSIG_OK;
        if (r[i].signed_ok && !key_fingerprint(pk, r[i].fp)) goto out;
        // one model and one binary per pack: a replay is one GGUF and one
        // build, and a pack that mixed them would need a verifier to sort it
        const char *m = jv_str(jv_get(jv_get(r[i].j, "model"), "sha256"), "");
        const char *b = jv_str(jv_get(jv_get(r[i].j, "build"), "binary_sha256"), "");
        const char *e = jv_str(jv_get(jv_get(r[i].j, "envelope"), "manifest_sha256"), "");
        if (!i) { msha = m; bsha = b; esha = e; }
        else if (strcmp(m, msha) || strcmp(b, bsha)) {
            fprintf(stderr, "error: pack: %s names another model or binary than "
                    "%s; one pack holds one model and one build\n", names[i], names[0]);
            goto out;
        }
        else if (strcmp(e, esha)) {
            fprintf(stderr, "error: pack: %s ran under another envelope manifest "
                    "than %s\n", names[i], names[0]);
            goto out;
        }
    }
    size_t sn = 0, pn = 0, en = 0;
    char key_fp[65] = "", h_env[65] = "";
    if (o->model_sig && !(sig = slurp(o->model_sig, &sn))) {
        fprintf(stderr, "error: pack: cannot read %s\n", o->model_sig);
        goto out;
    }
    if (o->model_pubkey && (!(pub = slurp(o->model_pubkey, &pn)) ||
                            !oms_pubkey_fingerprint(o->model_pubkey, key_fp))) {
        fprintf(stderr, "error: pack: %s is not a readable PEM public key\n",
                o->model_pubkey);
        goto out;
    }
    if (o->envelope) {
        if (!(env = slurp(o->envelope, &en))) {
            fprintf(stderr, "error: pack: cannot read %s\n", o->envelope);
            goto out;
        }
        envelope_data_sha256(env, en, h_env);
        // the receipts name the sidecar their load read; a pack that put a
        // different one beside them would misattribute the verdict
        if (strcmp(h_env, esha)) {
            fprintf(stderr, "error: pack: %s is not the envelope manifest the "
                    "receipts name (%s)\n", o->envelope, *esha ? esha : "none");
            goto out;
        }
    }
    for (int a = 0; a < o->n_attach; a++) {
        const char *bn = base_of(o->attach[a]);
        if (!plain_name(bn) || reserved_name(bn)) {
            fprintf(stderr, "error: pack: attachment name %s is reserved or not a "
                    "plain file name\n", bn);
            goto out;
        }
        for (int b2 = 0; b2 < a; b2++)
            if (!strcmp(bn, base_of(o->attach[b2]))) {
                fprintf(stderr, "error: pack: two attachments are named %s\n", bn);
                goto out;
            }
        if (!(att[a] = slurp(o->attach[a], &attn[a]))) {
            fprintf(stderr, "error: pack: cannot read %s\n", o->attach[a]);
            goto out;
        }
    }
    if (bundle_mkdir(o->out_dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "error: pack: cannot create %s\n", o->out_dir);
        goto out;
    }
    // nothing in the directory is overwritten (see bundle_export)
    {
        char p[4200];
        join(p, sizeof p, o->out_dir, "pack.json");
        char **have = NULL;
        int nh = list_receipts(o->out_dir, &have);
        FILE *ex = fopen(p, "rb");
        if (ex || nh > 0) {
            if (ex) fclose(ex);
            free_names(have, nh > 0 ? nh : 0);
            fprintf(stderr, "error: pack: %s already holds a pack or receipts; a "
                    "pack is written into a directory that holds none\n", o->out_dir);
            goto out;
        }
        free_names(have, nh > 0 ? nh : 0);
    }

    sbuf m = {0};
    char utc[32];
    time_t now = time(NULL);
    struct tm g;
#ifdef _WIN32
    gmtime_s(&g, &now);
#else
    gmtime_r(&now, &g);
#endif
    strftime(utc, sizeof utc, "%Y-%m-%dT%H:%M:%SZ", &g);
    sb_lit(&m, "{\"schema_version\":\"" PACK_SCHEMA "\",\"created_utc\":");
    put_str(&m, utc);
    sb_lit(&m, ",\"files\":[");
    bool first = true;
    bool wrote_ok = true;
    for (int i = 0; i < nr && wrote_ok; i++) {
        char p[4200], h[65];
        join(p, sizeof p, o->out_dir, r[i].name);
        envelope_data_sha256(r[i].bytes, r[i].n, h);
        wrote_ok = spill(p, r[i].bytes, r[i].n);
        sb_fmt(&m, "%s{\"name\":\"%s\",\"role\":\"receipt\",\"sha256\":\"%s\","
                   "\"bytes\":%zu}", first ? "" : ",", r[i].name, h, r[i].n);
        first = false;
    }
    struct { const char *name, *role; char *data; size_t n; } extra[3] = {
        { "model.sig", "model_signature", sig, sn },
        { "model-pubkey.pem", "model_key", pub, pn },
        { "envelope.json", "envelope", env, en },
    };
    for (int k = 0; k < 3 && wrote_ok; k++) {
        if (!extra[k].data) continue;
        char p[4200], h[65];
        join(p, sizeof p, o->out_dir, extra[k].name);
        envelope_data_sha256(extra[k].data, extra[k].n, h);
        wrote_ok = spill(p, extra[k].data, extra[k].n);
        sb_fmt(&m, ",{\"name\":\"%s\",\"role\":\"%s\",\"sha256\":\"%s\",\"bytes\":%zu}",
               extra[k].name, extra[k].role, h, extra[k].n);
    }
    for (int a = 0; a < o->n_attach && wrote_ok; a++) {
        char p[4200], h[65];
        const char *bn = base_of(o->attach[a]);
        join(p, sizeof p, o->out_dir, bn);
        envelope_data_sha256(att[a], attn[a], h);
        wrote_ok = spill(p, att[a], attn[a]);
        sb_lit(&m, ",{\"name\":");
        put_str(&m, bn);
        sb_fmt(&m, ",\"role\":\"attachment\",\"sha256\":\"%s\",\"bytes\":%zu}", h, attn[a]);
    }
    // the inferences, in write order, and whether they form one unbroken
    // segment of the server's chain
    int breaks = 0;
    sb_lit(&m, "],\"inferences\":[");
    for (int i = 0; i < nr; i++) {
        jv *j = r[i].j, *sv = jv_get(j, "serve");
        const char *prev = jv_str(jv_get(jv_get(j, "chain"), "prev"), NULL);
        if (i && (!prev || strcmp(prev, r[i - 1].hash))) breaks++;
        sb_fmt(&m, "%s{\"id\":\"%s\",\"file\":\"%s\",\"chain_prev\":",
               i ? "," : "", r[i].hash, r[i].name);
        put_str(&m, prev);
        sb_lit(&m, ",\"api\":");
        put_str(&m, jv_str(jv_get(sv, "api"), NULL));
        sb_lit(&m, ",\"request_id\":");
        put_str(&m, jv_str(jv_get(sv, "request_id"), NULL));
        sb_lit(&m, ",\"shaped_by\":");
        jv *sb2 = jv_get(sv, "shaped_by");
        if (sb2) jv_dump(sb2, &m); else sb_lit(&m, "null");
        sb_lit(&m, ",\"signer_fingerprint\":");
        put_str(&m, r[i].signed_ok ? r[i].fp : NULL);
        sb_lit(&m, ",\"envelope_verdict\":");
        put_str(&m, jv_str(jv_get(jv_get(j, "envelope"), "verdict"), NULL));
        sb_lit(&m, "}");
    }
    // "segment", never "chain": a signed record's own chain is its LAST
    // `,"chain":` (see the manifest note in bundle_export)
    sb_fmt(&m, "],\"segment\":{\"receipts\":%d,\"contiguous\":%s,\"breaks\":%d,"
               "\"first_prev\":", nr, breaks ? "false" : "true", breaks);
    put_str(&m, jv_str(jv_get(jv_get(r[0].j, "chain"), "prev"), NULL));
    sb_fmt(&m, ",\"last_hash\":\"%s\"}", r[nr - 1].hash);
    jv *model = jv_get(r[0].j, "model"), *build = jv_get(r[0].j, "build");
    sb_lit(&m, ",\"model\":{\"name\":");
    put_str(&m, base_of(jv_str(jv_get(model, "path"), "")));
    sb_lit(&m, ",\"sha256\":");
    put_str(&m, jv_str(jv_get(model, "sha256"), NULL));
    sb_lit(&m, ",\"oms\":");
    jv *ms = jv_get(r[0].j, "model_signature");
    if (ms) jv_dump(ms, &m); else sb_lit(&m, "null");
    sb_lit(&m, ",\"signature_file\":");
    put_str(&m, sig ? "model.sig" : NULL);
    sb_lit(&m, ",\"key_file\":");
    put_str(&m, pub ? "model-pubkey.pem" : NULL);
    sb_lit(&m, ",\"key_fingerprint\":");
    put_str(&m, pub ? key_fp : NULL);
    sb_lit(&m, "},\"binary\":{\"runner\":");
    put_str(&m, jv_str(jv_get(r[0].j, "runner"), NULL));
    sb_lit(&m, ",\"sha256\":");
    put_str(&m, jv_str(jv_get(build, "binary_sha256"), NULL));
    sb_lit(&m, ",\"os\":");
    put_str(&m, jv_str(jv_get(build, "os"), NULL));
    sb_lit(&m, ",\"arch\":");
    put_str(&m, jv_str(jv_get(build, "arch"), NULL));
    sb_lit(&m, "},\"envelope\":");
    if (env) {
        sb_fmt(&m, "{\"file\":\"envelope.json\",\"sha256\":\"%s\",\"verdict\":", h_env);
        put_str(&m, jv_str(jv_get(jv_get(r[0].j, "envelope"), "verdict"), NULL));
        sb_lit(&m, "}");
    } else {
        sb_lit(&m, "null");
    }
    sb_lit(&m, ",\"check\":\"runner --check-pack DIR\",\"replay\":\"runner -m "
               "<the GGUF whose sha256 is model.sha256> --verify DIR/<receipt "
               "file> for each inference\",\"statement\":\"--check-pack proves "
               "these files are the ones listed, every receipt's chain hash "
               "recomputes and is its inference id, every signature verifies "
               "with the key named, the receipts form the chain segment stated "
               "here, and every receipt names this model, this binary and this "
               "envelope. Attachments are listed by sha256 and not interpreted. "
               "It does not replay the inferences: that takes the model and the "
               "binary named by digest, and --verify.\"}\n");
    char pm[4200];
    join(pm, sizeof pm, o->out_dir, "pack.json");
    if (!wrote_ok || m.failed || !spill(pm, m.s, m.n) ||
        (o->sign_key && !record_sign(pm, o->sign_key, NULL))) {
        fprintf(stderr, "error: pack: cannot write the pack into %s\n", o->out_dir);
        free(m.s);
        // remove what this call wrote; the directory held none of it before
        remove(pm);
        for (int i = 0; i < nr; i++) {
            char p[4200];
            join(p, sizeof p, o->out_dir, r[i].name);
            remove(p);
        }
        for (int k = 0; k < 3; k++) if (extra[k].data) {
            char p[4200];
            join(p, sizeof p, o->out_dir, extra[k].name);
            remove(p);
        }
        for (int a = 0; a < o->n_attach; a++) {
            char p[4200];
            join(p, sizeof p, o->out_dir, base_of(o->attach[a]));
            remove(p);
        }
        goto out;
    }
    free(m.s);
    fprintf(stderr, "pack -> %s (%d receipt%s, chain %s%s%s)\n", o->out_dir, nr,
            nr == 1 ? "" : "s", breaks ? "with breaks" : "contiguous",
            env ? ", envelope" : "", o->sign_key ? ", manifest signed" : "");
    rc = 0;
out:
    if (r) for (int i = 0; i < nr; i++) { jv_free(r[i].j); free(r[i].bytes); }
    free(r);
    free_names(names, nr > 0 ? nr : 0);
    free(sig); free(pub); free(env);
    for (int a = 0; a < PACK_MAX_ATTACH; a++) free(att[a]);
    return rc;
}

static int pbad(const char *dir, const char *why, const char *what) {
    printf("BAD: pack %s: %s%s%s\n", dir, why, what ? ": " : "", what ? what : "");
    return 2;
}

int pack_check(const char *dir, const char *trust_hex) {
    char path[4200];
    join(path, sizeof path, dir, "pack.json");
    size_t mn = 0;
    char *man = slurp(path, &mn);
    if (!man) {
        printf("UNVERIFIABLE: pack %s: no readable pack.json\n", dir);
        return 3;
    }
    int rc = 2;
    jv *m = json_parse(man, mn);
    if (!m || strcmp(jv_str(jv_get(m, "schema_version"), ""), PACK_SCHEMA)) {
        printf("UNVERIFIABLE: pack %s: pack.json is not %s\n", dir, PACK_SCHEMA);
        rc = 3;
        goto out;
    }
    char mpk[SIGN_PUBHEX_CAP] = "";
    receipt_sig_state mst = receipt_signature_check(man, mn, mpk);
    if (mst == RSIG_BAD || mst == RSIG_MALFORMED) {
        rc = pbad(dir, "the manifest's signature does not verify", NULL);
        goto out;
    }
    if (mst == RSIG_OK) {
        char mc[65];
        const char *ms = jv_str(jv_get(jv_get(m, "chain"), "hash"), "");
        if (!chain_recompute(man, mn, mc) || strcmp(mc, ms)) {
            rc = pbad(dir, "the manifest's chain hash does not recompute", NULL);
            goto out;
        }
    }
    if (trust_hex && *trust_hex && (mst != RSIG_OK || strcmp(trust_hex, mpk))) {
        rc = pbad(dir, mst == RSIG_OK ? "the manifest is signed by another key"
                                      : "the manifest is unsigned", trust_hex);
        goto out;
    }
    jv *files = jv_get(m, "files");
    if (!files || files->type != J_ARR || files->n == 0) {
        rc = pbad(dir, "the manifest lists no files", NULL);
        goto out;
    }
    for (int i = 0; i < files->n; i++) {
        const char *name = jv_str(jv_get(files->items[i], "name"), NULL);
        const char *want = jv_str(jv_get(files->items[i], "sha256"), "");
        if (!plain_name(name)) { rc = pbad(dir, "a listed file name is not a plain name", name); goto out; }
        char fp[4200], have[65];
        join(fp, sizeof fp, dir, name);
        if (!envelope_file_sha256(fp, have)) { rc = pbad(dir, "a listed file is missing", name); goto out; }
        if (strcmp(have, want)) { rc = pbad(dir, "a file differs from its listed sha256", name); goto out; }
    }
    // a receipt in the directory that the manifest does not list is a
    // receipt someone added, or one the manifest leaves out of its count
    {
        char **have = NULL;
        int nh = list_receipts(dir, &have);
        for (int i = 0; i < nh; i++) {
            bool listed = false;
            for (int k = 0; k < files->n && !listed; k++)
                listed = !strcmp(have[i], jv_str(jv_get(files->items[k], "name"), ""));
            if (!listed) {
                rc = pbad(dir, "a receipt in the directory is not in the manifest", have[i]);
                free_names(have, nh);
                goto out;
            }
        }
        free_names(have, nh > 0 ? nh : 0);
    }
    jv *inf = jv_get(m, "inferences");
    if (!inf || inf->type != J_ARR || inf->n == 0) {
        rc = pbad(dir, "the manifest lists no inferences", NULL);
        goto out;
    }
    jv *mm = jv_get(m, "model"), *mb = jv_get(m, "binary"), *me = jv_get(m, "envelope");
    const char *want_m = jv_str(jv_get(mm, "sha256"), "");
    const char *want_b = jv_str(jv_get(mb, "sha256"), "");
    const char *want_e = me && me->type == J_OBJ ? jv_str(jv_get(me, "sha256"), "") : NULL;
    char prev_hash[65] = "";
    int breaks = 0, n_receipts = 0;
    for (int i = 0; i < inf->n; i++) {
        jv *it = inf->items[i];
        const char *name = jv_str(jv_get(it, "file"), NULL);
        const char *id = jv_str(jv_get(it, "id"), "");
        bool listed = false;
        for (int k = 0; k < files->n && !listed; k++)
            listed = name && !strcmp(name, jv_str(jv_get(files->items[k], "name"), "")) &&
                     !strcmp(jv_str(jv_get(files->items[k], "role"), ""), "receipt");
        if (!listed) { rc = pbad(dir, "an inference's receipt is not a listed receipt", name); goto out; }
        char fp[4200];
        join(fp, sizeof fp, dir, name);
        size_t rn = 0;
        char *rec = slurp(fp, &rn);
        jv *rj = rec ? json_parse(rec, rn) : NULL;
        char h[65], pk[SIGN_PUBHEX_CAP] = "", kfp[65] = "";
        const char *why = NULL;
        if (!rj) why = "a receipt is unreadable";
        else if (!chain_recompute(rec, rn, h) ||
                 strcmp(h, jv_str(jv_get(jv_get(rj, "chain"), "hash"), "")) ||
                 strcmp(h, id))
            why = "a receipt's chain hash does not recompute to its inference id";
        else {
            receipt_sig_state st = receipt_signature_check(rec, rn, pk);
            const char *named = jv_str(jv_get(it, "signer_fingerprint"), NULL);
            if (st == RSIG_BAD || st == RSIG_MALFORMED)
                why = "a receipt's signature does not verify";
            else if ((st == RSIG_OK) != (named != NULL) ||
                     (named && (!key_fingerprint(pk, kfp) || strcmp(kfp, named))))
                why = "a receipt's signer is not the one the manifest names";
            else if (strcmp(jv_str(jv_get(jv_get(rj, "model"), "sha256"), ""), want_m) ||
                     strcmp(jv_str(jv_get(jv_get(rj, "build"), "binary_sha256"), ""), want_b))
                why = "a receipt names another model or binary than the manifest";
            else if (want_e && strcmp(jv_str(jv_get(jv_get(rj, "envelope"),
                                                    "manifest_sha256"), ""), want_e))
                why = "a receipt names another envelope manifest than the pack's";
        }
        const char *prev = rj ? jv_str(jv_get(jv_get(rj, "chain"), "prev"), NULL) : NULL;
        if (!why && i && (!prev || strcmp(prev, prev_hash))) breaks++;
        if (!why) snprintf(prev_hash, sizeof prev_hash, "%s", h);
        jv_free(rj);
        free(rec);
        if (why) { rc = pbad(dir, why, name); goto out; }
        n_receipts++;
    }
    jv *ch = jv_get(m, "segment");
    jv *contig = jv_get(ch, "contiguous");
    bool said = contig && contig->type == J_BOOL && contig->b;
    if (said != (breaks == 0) || jv_num(jv_get(ch, "breaks"), -1) != breaks ||
        jv_num(jv_get(ch, "receipts"), -1) != n_receipts) {
        rc = pbad(dir, "the chain the manifest states is not the receipts' chain", NULL);
        goto out;
    }
    const char *kf = jv_str(jv_get(mm, "key_file"), NULL);
    if (kf) {
        char kp[4200], kfp[65];
        join(kp, sizeof kp, dir, kf);
        if (!plain_name(kf) || !oms_pubkey_fingerprint(kp, kfp) ||
            strcmp(kfp, jv_str(jv_get(mm, "key_fingerprint"), ""))) {
            rc = pbad(dir, "the model key's fingerprint does not recompute", kf);
            goto out;
        }
    }
    printf("OK: pack %s: %d inference%s, chain %s; manifest %s; replay needs "
           "model %s and binary %s\n", dir, n_receipts, n_receipts == 1 ? "" : "s",
           breaks ? "with breaks" : "contiguous",
           mst == RSIG_OK ? "signed" : "unsigned", want_m, want_b);
    rc = 0;
out:
    jv_free(m);
    free(man);
    return rc;
}
