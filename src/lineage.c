#include "lineage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "envelope.h"
#include "receipts.h"

#include <sys/stat.h>

bool lineage_sidecar(const char *artifact, char *out, size_t cap) {
    return record_sidecar(artifact, out, cap);
}

bool lineage_put_input(sbuf *b, const char *path, const char *sha) {
    char own[65];
    if (!sha) {
        if (!envelope_file_sha256(path, own)) return false;
        sha = own;
    }
    sb_lit(b, "{\"path\":\"");
    sb_esc(b, path, strlen(path));
    sb_lit(b, "\",\"sha256\":\"");
    sb_lit(b, sha);
    sb_lit(b, "\"");
    char rec[4096], rsha[65];
    if (lineage_sidecar(path, rec, sizeof rec) && envelope_file_sha256(rec, rsha)) {
        sb_lit(b, ",\"record\":{\"path\":\"");
        sb_esc(b, rec, strlen(rec));
        sb_lit(b, "\",\"sha256\":\"");
        sb_lit(b, rsha);
        sb_lit(b, "\"}");
    }
    sb_lit(b, "}");
    return !b->failed;
}

bool lineage_sign(const char *record_path, const char *sign_key) {
    if (!sign_key) return true;
    return record_sign(record_path, sign_key, NULL);
}

// ------------------------------------------------------------------ walker --

static jv *read_json(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, n = 0, r;
    char *buf = malloc(cap);
    while (buf && (r = fread(buf + n, 1, cap - n, f)) > 0) {
        n += r;
        if (n == cap) {
            if (cap > (16u << 20)) { free(buf); buf = NULL; break; }
            char *nb = realloc(buf, cap * 2);
            if (!nb) { free(buf); buf = NULL; break; }
            buf = nb;
            cap *= 2;
        }
    }
    fclose(f);
    if (!buf) return NULL;
    jv *j = json_parse(buf, n);
    free(buf);
    return j;
}

// The step a record's schema names, and the key its OUTPUT file sits under.
static const char *step_of(jv *rec, const char **out_key) {
    const char *sv = jv_str(jv_get(rec, "schema_version"), "");
    static const struct { const char *schema, *step, *out; } K[] = {
        { "xyntetik.runner.quant.v1",          "quantize",       "output" },
        { "xyntetik.runner.merge.v1",          "merge",          "merged" },
        { "xyntetik.runner.train.v1",          "train",          "adapter" },
        { "xyntetik.runner.context-surgery.v1", "context-surgery", "target" },
    };
    for (size_t i = 0; i < sizeof K / sizeof K[0]; i++)
        if (!strcmp(sv, K[i].schema)) { *out_key = K[i].out; return K[i].step; }
    *out_key = NULL;
    return NULL;
}

// The input keys each step records, in the order they print.
static const char *const INPUT_KEYS[] = {
    "base", "adapter", "data", "prune_experts", "type_plan",
};

typedef struct { int worst; } walk_state;   // 0 verified, 1 unsigned, 2 broken

static void note(walk_state *w, int level) { if (level > w->worst) w->worst = level; }

static void indent(int depth) { for (int i = 0; i < depth; i++) fputs("  ", stdout); }

static void short_sha(const char *sha, char out[13]) {
    snprintf(out, 13, "%.12s", sha ? sha : "?");
}

static void walk_record(const char *rec_path, const char *want_rec_sha,
                        const char *want_out_sha, const char *trust_hex,
                        int depth, walk_state *w);

// The evaluations a file carries (<file>.eval.<kind>.json, R17.2): shown
// beside its chain. An evaluation made for another file (its subject hash
// differs) is STALE and does not count; a bad signature is BROKEN.
static const char *const EVAL_KINDS[] = { "fidelity", "agent" };

static void walk_evals(const char *file, const char *sha, const char *trust_hex,
                       int depth, walk_state *w) {
    for (size_t k = 0; k < sizeof EVAL_KINDS / sizeof EVAL_KINDS[0]; k++) {
        char path[4096];
        int n = snprintf(path, sizeof path, "%s.eval.%s.json", file, EVAL_KINDS[k]);
        if (n < 0 || (size_t)n >= sizeof path) continue;
        jv *ev = read_json(path);
        if (!ev) continue;
        indent(depth);
        const char *subj = jv_str(jv_get(jv_get(ev, "subject"), "sha256"), "");
        jv *m = jv_get(ev, "metrics");
        jv *pass = jv_get(ev, "pass");
        const char *pw = !pass || pass->type == J_NULL ? "no threshold set"
                       : pass->type == J_BOOL && pass->b ? "pass" : "FAIL";
        printf("eval %s: ", EVAL_KINDS[k]);
        if (!strcmp(EVAL_KINDS[k], "fidelity"))
            printf("vs %s, %d positions, top-1 %.1f%%, mean KL %.4g; %s",
                   jv_str(jv_get(jv_get(ev, "reference"), "path"), "?"),
                   (int)jv_num(jv_get(m, "positions_scored"), 0),
                   jv_num(jv_get(m, "top1_agreement_pct"), 0),
                   jv_num(jv_get(m, "mean_kld"), 0), pw);
        else
            printf("%d of %d tasks verified, %d aborts; %s",
                   (int)jv_num(jv_get(m, "tasks_verified"), 0),
                   (int)jv_num(jv_get(m, "tasks_attempted"), 0),
                   (int)jv_num(jv_get(m, "engine_error_aborts"), 0), pw);
        if (!sha || strcmp(subj, sha) != 0) {
            printf("  STALE (made for another file)\n");
            jv_free(ev);
            continue;
        }
        char pub[SIGN_PUBHEX_CAP];
        int sig = record_signature_state(path, trust_hex, pub);
        if (sig == 0) printf("  VERIFIED (signed %.16s...)\n", pub);
        else if (sig == 1) { printf("  UNSIGNED\n"); note(w, 1); }
        else { printf("  BROKEN: bad, malformed or untrusted signature\n"); note(w, 2); }
        jv_free(ev);
    }
}

bool lineage_require_eval(const char *model, const char *kind, const char *trust_hex,
                          char *why, size_t cap) {
    char path[4096], sha[65], pub[SIGN_PUBHEX_CAP];
    snprintf(path, sizeof path, "%s.eval.%s.json", model, kind);
    jv *ev = read_json(path);
    if (!ev) { snprintf(why, cap, "no %s evaluation beside the model (%s)", kind, path); return false; }
    bool ok = false;
    jv *pass = jv_get(ev, "pass");
    if (strcmp(jv_str(jv_get(ev, "schema_version"), ""), "xyntetik.runner.eval.v1") ||
        strcmp(jv_str(jv_get(ev, "kind"), ""), kind))
        snprintf(why, cap, "%s is not a %s evaluation record", path, kind);
    else if (!envelope_file_sha256(model, sha))
        snprintf(why, cap, "cannot hash %s", model);
    else if (strcmp(jv_str(jv_get(jv_get(ev, "subject"), "sha256"), ""), sha))
        snprintf(why, cap, "%s was made for another file (its subject sha256 differs)", path);
    else if (!pass || pass->type != J_BOOL || !pass->b)
        snprintf(why, cap, "%s did not pass (or set no threshold)", path);
    else {
        int sig = record_signature_state(path, trust_hex, pub);
        if (sig == 2) snprintf(why, cap, "%s: bad, malformed or untrusted signature", path);
        else if (sig == 1 && trust_hex && *trust_hex)
            snprintf(why, cap, "%s is unsigned and --trust-key asks for a signer", path);
        else ok = true;
    }
    jv_free(ev);
    return ok;
}

// A record cited by path that is not there any more: a chain copied to
// another machine or folder keeps its records together, so look for the same
// file name beside the record that cites it. The hash still decides whether
// it is the record that was cited.
static const char *relocate(const char *cited, const char *citing, char *buf, size_t cap) {
    char probe[65];
    if (envelope_file_sha256(cited, probe)) return cited;
    const char *name = strrchr(cited, '/');
    const char *bs = strrchr(cited, '\\');
    if (bs && (!name || bs > name)) name = bs;
    name = name ? name + 1 : cited;
    const char *dir_end = strrchr(citing, '/');
    const char *bs2 = strrchr(citing, '\\');
    if (bs2 && (!dir_end || bs2 > dir_end)) dir_end = bs2;
    int dn = dir_end ? (int)(dir_end - citing + 1) : 0;
    int n = snprintf(buf, cap, "%.*s%s", dn, citing, name);
    return (n > 0 && (size_t)n < cap) ? buf : cited;
}

// One input of a record: a file by hash, and the record that made it if any.
static void walk_input(const char *key, jv *in, const char *citing, const char *trust_hex,
                       int depth, walk_state *w) {
    const char *path = jv_str(jv_get(in, "path"), "?");
    const char *sha = jv_str(jv_get(in, "sha256"), NULL);
    char s[13];
    short_sha(sha, s);
    jv *link = jv_get(in, "record");
    indent(depth);
    if (!link) {
        printf("%s: %s sha256 %s  ORIGIN (no record: a download, or made before records)\n",
               key, path, s);
        walk_evals(path, sha, trust_hex, depth + 1, w);
        return;
    }
    printf("%s: %s sha256 %s\n", key, path, s);
    walk_evals(path, sha, trust_hex, depth + 1, w);
    char moved[4096];
    walk_record(relocate(jv_str(jv_get(link, "path"), ""), citing, moved, sizeof moved),
                jv_str(jv_get(link, "sha256"), NULL), sha, trust_hex, depth + 1, w);
}

static void walk_record(const char *rec_path, const char *want_rec_sha,
                        const char *want_out_sha, const char *trust_hex,
                        int depth, walk_state *w) {
    indent(depth);
    if (depth > 32) { printf("BROKEN: chain deeper than 32 links (a loop?)\n"); note(w, 2); return; }
    char rsha[65];
    if (!envelope_file_sha256(rec_path, rsha)) {
        printf("BROKEN: record %s is missing\n", rec_path);
        note(w, 2);
        return;
    }
    // the record must be the one the child recorded: altered later is broken
    if (want_rec_sha && strcmp(want_rec_sha, rsha) != 0) {
        printf("BROKEN: record %s changed after the next step recorded it\n", rec_path);
        note(w, 2);
        return;
    }
    jv *rec = read_json(rec_path);
    const char *out_key = NULL;
    const char *step = rec ? step_of(rec, &out_key) : NULL;
    if (!step) {
        printf("BROKEN: %s is not a lineage record\n", rec_path);
        note(w, 2);
        jv_free(rec);
        return;
    }
    jv *out = jv_get(rec, out_key);
    const char *out_path = jv_str(jv_get(out, "path"), "?");
    const char *out_sha = jv_str(jv_get(out, "sha256"), NULL);
    char pub[SIGN_PUBHEX_CAP];
    int sig = record_signature_state(rec_path, trust_hex, pub);
    char s[13];
    short_sha(out_sha, s);
    // the link itself: the child's input hash is this record's output hash
    bool link_ok = !want_out_sha || (out_sha && !strcmp(want_out_sha, out_sha));
    // and the output file, where it is still on disk, is that file
    char disk[65];
    bool on_disk = envelope_file_sha256(out_path, disk);
    bool disk_ok = !on_disk || (out_sha && !strcmp(disk, out_sha));
    const char *verdict;
    if (!link_ok) { verdict = "BROKEN: its output is not the file the next step used"; note(w, 2); }
    else if (!disk_ok) { verdict = "BROKEN: the file on disk no longer matches its record"; note(w, 2); }
    else if (sig == 2) { verdict = "BROKEN: bad, malformed or untrusted signature"; note(w, 2); }
    else if (sig == 1) { verdict = "UNSIGNED"; note(w, 1); }
    else verdict = "VERIFIED";
    const char *rec_name = strrchr(rec_path, '/');
#ifdef _WIN32
    const char *bs = strrchr(rec_path, '\\');
    if (bs && (!rec_name || bs > rec_name)) rec_name = bs;
#endif
    rec_name = rec_name ? rec_name + 1 : rec_path;
    printf("%s -> %s sha256 %s  %s", step, out_path, s, verdict);
    if (sig == 0) printf(" (signed %.16s...)", pub);
    if (!on_disk) printf(" (output file not present; checked by its recorded hash)");
    printf("  [runner %s, record %s]\n", jv_str(jv_get(rec, "runner"), "?"), rec_name);
    for (size_t i = 0; i < sizeof INPUT_KEYS / sizeof INPUT_KEYS[0]; i++) {
        if (!strcmp(INPUT_KEYS[i], out_key)) continue;   // a train record's "adapter" is its output
        jv *in = jv_get(rec, INPUT_KEYS[i]);
        if (in && in->type == J_OBJ && jv_get(in, "sha256"))
            walk_input(INPUT_KEYS[i], in, rec_path, trust_hex, depth + 1, w);
    }
    jv_free(rec);
}

// --lineage DIR on a --receipts directory (R17.4): which model and adapter
// versions served the answers, when, and in what order (a rollback shows as
// a version served again), the receipt chain's continuity, then each
// version's lineage once.
typedef struct { char model[65], adapter[65]; int count; int idx; char first[32], last[32]; } version;

static int walk_receipts_dir(const char *dir, const char *trust_hex) {
    walk_state w = { 0 };
    char **paths = NULL;
    int n = receipts_list_dir(dir, &paths);
    printf("lineage of the receipts in %s: %d answers\n", dir, n);
    if (n == 0) { printf("  NO RECORD: no receipt-*.json files\n"); free(paths); return 2; }
    version *v = calloc((size_t)n, sizeof *v);
    int nv = 0, *seq_v = calloc((size_t)n, sizeof *seq_v);
    jv **first_rec = calloc((size_t)n, sizeof *first_rec);   // per version
    int signed_n = 0, unsigned_n = 0, bad_n = 0, breaks = 0;
    char prev_hash[65] = "";
    for (int i = 0; i < n && v && seq_v && first_rec; i++) {
        jv *r = read_json(paths[i]);
        if (!r) { bad_n++; seq_v[i] = -1; continue; }
        char pub[SIGN_PUBHEX_CAP];
        int sig = record_signature_state(paths[i], trust_hex, pub);
        if (sig == 0) signed_n++; else if (sig == 1) unsigned_n++; else bad_n++;
        jv *ch = jv_get(r, "chain");
        const char *ph = jv_str(jv_get(ch, "prev"), ""), *hh = jv_str(jv_get(ch, "hash"), "");
        if (i > 0 && prev_hash[0] && strcmp(ph, prev_hash)) breaks++;
        snprintf(prev_hash, sizeof prev_hash, "%s", hh);
        const char *ms = jv_str(jv_get(jv_get(r, "model"), "sha256"), "");
        jv *ad = jv_get(r, "adapter");
        const char *as = ad && ad->type == J_OBJ ? jv_str(jv_get(ad, "sha256"), "") : "";
        const char *utc = jv_str(jv_get(r, "generated_utc"), "?");
        int k = 0;
        while (k < nv && (strcmp(v[k].model, ms) || strcmp(v[k].adapter, as))) k++;
        if (k == nv) {
            snprintf(v[k].model, 65, "%s", ms);
            snprintf(v[k].adapter, 65, "%s", as);
            snprintf(v[k].first, 32, "%s", utc);
            v[k].idx = k + 1;
            first_rec[k] = r;
            r = NULL;
            nv++;
        }
        v[k].count++;
        snprintf(v[k].last, 32, "%s", utc);
        seq_v[i] = k;
        jv_free(r);
    }
    if (breaks) note(&w, 2);
    if (bad_n) note(&w, 2);
    if (unsigned_n) note(&w, 1);
    printf("  receipts: %d signed, %d unsigned, %d bad; chain %s\n", signed_n, unsigned_n, bad_n,
           breaks ? "BROKEN (a receipt's prev is not the one before it)" : "continuous");
    // the order versions served in: a version that returns is a rollback
    printf("  timeline:");
    for (int i = 0; i < n; ) {
        int k = seq_v[i], run = 0;
        while (i < n && seq_v[i] == k) { i++; run++; }
        if (k >= 0) printf(" v%d x%d", k + 1, run);
    }
    printf("\n");
    for (int k = 0; k < nv; k++) {
        char ms[13], as[13];
        short_sha(v[k].model, ms);
        short_sha(v[k].adapter[0] ? v[k].adapter : "none", as);
        printf("  v%d: model sha256 %s, adapter %s: %d answers, %s .. %s\n", k + 1, ms,
               v[k].adapter[0] ? as : "none", v[k].count, v[k].first, v[k].last);
        static const char *const KEYS[] = { "model", "adapter" };
        for (size_t i = 0; i < 2; i++) {
            jv *in = jv_get(first_rec[k], KEYS[i]);
            if (in && in->type == J_OBJ && jv_get(in, "sha256"))
                walk_input(KEYS[i], in, paths[0], trust_hex, 2, &w);
        }
        jv_free(first_rec[k]);
    }
    for (int i = 0; i < n; i++) free(paths[i]);
    free(paths); free(v); free(seq_v); free(first_rec);
    printf("RESULT: %s\n", w.worst == 0 ? "VERIFIED (every link consistent and signed)"
                         : w.worst == 1 ? "CONSISTENT, NOT ALL SIGNED"
                                        : "BROKEN");
    return w.worst;
}

int lineage_walk(const char *start, const char *trust_hex) {
    struct stat dst;
    if (stat(start, &dst) == 0 && S_ISDIR(dst.st_mode)) return walk_receipts_dir(start, trust_hex);
    walk_state w = { 0 };
    char rec[4096];
    const char *want_out = NULL;
    char sha[65];
    // a record named directly, or an artifact whose sidecar is found
    jv *probe = NULL;
    size_t sl = strlen(start);
    if (sl > 5 && !strcmp(start + sl - 5, ".json")) probe = read_json(start);
    const char *ok_key = NULL;
    if (probe && !strcmp(jv_str(jv_get(probe, "schema_version"), ""),
                         "xyntetik.runner.transcript.v1")) {
        // a receipt: a served or one-shot answer. Its signature, then the
        // model and adapter it names, walked back through their records.
        char pub[SIGN_PUBHEX_CAP];
        int sig = record_signature_state(start, trust_hex, pub);
        printf("lineage of receipt %s: %s", start,
               sig == 0 ? "signed" : sig == 1 ? "UNSIGNED" : "BROKEN: bad, malformed or untrusted signature");
        if (sig == 0) printf(" (%.16s...)", pub);
        printf("\n");
        note(&w, sig);
        static const char *const KEYS[] = { "model", "adapter" };
        for (size_t i = 0; i < 2; i++) {
            jv *in = jv_get(probe, KEYS[i]);
            if (in && in->type == J_OBJ && jv_get(in, "sha256"))
                walk_input(KEYS[i], in, start, trust_hex, 1, &w);
        }
        jv_free(probe);
        printf("RESULT: %s\n", w.worst == 0 ? "VERIFIED (every link consistent and signed)"
                             : w.worst == 1 ? "CONSISTENT, NOT ALL SIGNED"
                                            : "BROKEN");
        return w.worst;
    }
    if (probe && step_of(probe, &ok_key)) {
        snprintf(rec, sizeof rec, "%s", start);
        printf("lineage of record %s\n", start);
    } else {
        if (!envelope_file_sha256(start, sha)) {
            printf("BROKEN: cannot read %s\n", start);
            jv_free(probe);
            return 2;
        }
        char s[13];
        short_sha(sha, s);
        printf("lineage of %s (sha256 %s)\n", start, s);
        if (!lineage_sidecar(start, rec, sizeof rec)) {
            printf("  NO RECORD: no step on this machine wrote %s (a download, or made "
                   "before records)\n", start);
            jv_free(probe);
            return 2;
        }
        want_out = sha;
        walk_evals(start, sha, trust_hex, 1, &w);
    }
    jv_free(probe);
    walk_record(rec, NULL, want_out, trust_hex, 1, &w);
    printf("RESULT: %s\n", w.worst == 0 ? "VERIFIED (every link consistent and signed)"
                         : w.worst == 1 ? "CONSISTENT, NOT ALL SIGNED"
                                        : "BROKEN");
    return w.worst;
}
