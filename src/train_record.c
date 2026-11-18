#include "train_record.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "compat.h"
#include "envelope.h"
#include "fpclass.h"
#include "lineage.h"

#define MAX_INPUT_FILE (256u * 1024u * 1024u)

static bool record_num(jv *obj, const char *key, double min, double max,
                       double *out) {
    jv *v = jv_get(obj, key);
    if (!v || v->type != J_NUM || !fp_isfinite(v->num) ||
        v->num < min || v->num > max)
        return false;
    *out = v->num;
    return true;
}

static bool record_int(jv *obj, const char *key, int min, int max, int *out) {
    jv *v = jv_get(obj, key);
    if (!v || v->type != J_NUM || !v->num_int || v->num_text ||
        !fp_isfinite(v->num) || v->num < (double)min || v->num > (double)max ||
        v->num != trunc(v->num))
        return false;
    *out = (int)v->num;
    return true;
}

bool train_record_num(jv *obj, const char *key, double min, double max, double *out) {
    return record_num(obj, key, min, max, out);
}

bool train_record_int(jv *obj, const char *key, int min, int max, int *out) {
    return record_int(obj, key, min, max, out);
}

static const char *const TRAIN_ENV_INERT[] = {
    // read only by the test harness, never by this binary
    "RUNNER_EXE", "RUNNER_CONFORMANCE_OUT", "RUNNER_TEST_GPU_OFF", "RUNNER_TEST_TIMEOUT_SCALE",
    "RUNNER_REQUIRE_TURN_GUARD", "RUNNER_SWEEP_MODELS", "RUNNER_LLAMA_CPP_BIN",
    // set by GitHub Actions' runner on every job, never read by this binary
    "RUNNER_NAME", "RUNNER_OS", "RUNNER_ARCH", "RUNNER_TEMP", "RUNNER_TOOL_CACHE", "RUNNER_WORKSPACE",
    "RUNNER_ENVIRONMENT", "RUNNER_TRACKING_ID", "RUNNER_DEBUG", "RUNNER_PERFLOG",
    // read, and held not to move a training run's bytes
    "RUNNER_TRAIN_ROWS",       // every width trains the same bytes (test_batched_taped_forward_is_the_solo_bytes)
    "RUNNER_TPOOL_SPIN",       // how idle pool threads wait
    "RUNNER_HF_CACHE", "RUNNER_CURL", "RUNNER_PREFIX_CACHE_MB",   // downloads, the server's prefix cache
    "RUNNER_TEST_BUILD_DIRTY", // marks the build dirty: it can only take the claim down
};

static bool train_env_inert(const char *name, size_t n) {
    for (size_t i = 0; i < sizeof TRAIN_ENV_INERT / sizeof *TRAIN_ENV_INERT; i++)
        if (strlen(TRAIN_ENV_INERT[i]) == n && !strncmp(TRAIN_ENV_INERT[i], name, n)) return true;
    return false;
}

// Whether the variable `name` (n bytes) is one a record names: a RUNNER_ one
// outside the inert set, or one of the two the binary reads outside that
// prefix that reach arithmetic (the recurrent scan's chunking, src/model.c).
bool train_env_counts(const char *name, size_t n) {
    if (n == 12 && !strncmp(name, "XR_SSM_CHUNK", 12)) return true;
    if (n == 13 && !strncmp(name, "XR_SSM_SERIAL", 13)) return true;
    return n > 7 && !strncmp(name, "RUNNER_", 7) && !train_env_inert(name, n);
}

bool train_env_is_secret(const char *name, size_t n) {
    static const char *const words[] = { "KEY", "TOKEN", "SECRET", "PASS" };
    for (size_t w = 0; w < sizeof words / sizeof *words; w++) {
        size_t wl = strlen(words[w]);
        for (size_t i = 0; i + wl <= n; i++)
            if (!memcmp(name + i, words[w], wl)) return true;
    }
    return false;
}

static bool replay_fail(int *rc, const char *field, const char *want, const char *got) {
    fputs("MISMATCH ", stdout);
    envelope_print_safe(stdout, field, 128, "missing", true);
    fputs(": expected ", stdout);
    envelope_print_safe(stdout, want, 128, "missing", true);
    fputs(", got ", stdout);
    envelope_print_safe(stdout, got, 128, "missing", true);
    putchar('\n');
    *rc = 2;
    return true;
}

static bool safe_relname(const char *s, size_t max_len) {
    if (!s || !*s) return false;
    if (!strcmp(s, ".") || !strcmp(s, "..")) return false;
    size_t n = strlen(s);
    if (n > max_len) return false;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x20 || c == 0x7f || c == '/' || c == '\\' || c == ':') return false;
    }
    return true;
}

static bool fmt_checked(char *dst, size_t cap, const char *fmt, ...) RUNNER_PRINTF(3, 4);
static bool fmt_checked(char *dst, size_t cap, const char *fmt, ...) {
    if (!cap) return false;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(dst, cap, fmt, ap);
    va_end(ap);
    return n >= 0 && (size_t)n < cap;
}

static bool relocate_local(const char *cited, const char *citing, char *buf, size_t cap) {
    if (!safe_relname(cited, 255)) return false;
    const char *s = strrchr(citing, '/');
    const char *b = strrchr(citing, '\\');
    if (b && (!s || b > s)) s = b;
    int dn = s ? (int)(s - citing + 1) : 0;
    return fmt_checked(buf, cap, "%.*s%s", dn, citing, cited);
}

static bool replay_check_sha_opt(const char *field, const char *path, const char *want,
                                 bool nofollow, int *rc) {
    if (!envelope_is_sha256_hex(want)) return !replay_fail(rc, field, "64-hex sha256", want);
    char got[65] = "";
    bool ok = path && *path &&
              (nofollow ? envelope_cited_regular_sha256(path, got) : envelope_cited_sha256(path, got));
    if (!ok) return !replay_fail(rc, field, want, "missing or unreadable");
    if (strcmp(got, want) != 0) return !replay_fail(rc, field, want, got);
    return true;
}

static bool replay_check_sha(const char *field, const char *path, const char *want, int *rc) {
    return replay_check_sha_opt(field, path, want, false, rc);
}

static bool replay_parent_dir(const char *path, char *out, size_t cap) {
    const char *s = strrchr(path, '/');
    const char *b = strrchr(path, '\\');
    if (b && (!s || b > s)) s = b;
    if (!s) return fmt_checked(out, cap, ".");
    return fmt_checked(out, cap, "%.*s", (int)(s - path), path);
}

static bool safe_dir_nofollow(const char *dir) {
    plat_stat_t st;
    bool is_link = false;
    return dir && *dir && plat_stat_nofollow(dir, &st, &is_link) == 0 && !is_link && S_ISDIR(st.st_mode);
}

static bool probe_teacher_subdir(const char *base_dir, const char *fname, char *out, size_t cap,
                                 bool *saw_symlink) {
    char tdir[4096];
    if (!fmt_checked(tdir, sizeof tdir, "%s/teacher", base_dir)) return false;
    plat_stat_t st;
    bool is_link = false;
    if (plat_stat_nofollow(tdir, &st, &is_link) == 0 && is_link) {
        if (saw_symlink) *saw_symlink = true;
        return false;
    }
    if (!safe_dir_nofollow(tdir)) return false;
    return fmt_checked(out, cap, "%s/%s", tdir, fname) && record_regular_file_nofollow(out);
}

static bool replay_resolve_teacher(const train_record_opts *o, cli_replay *rp, jv *tch, int *rc) {
    jv *ti = jv_get(tch, "topk_idx"), *tl = jv_get(tch, "topk_logp");
    const char *inm = jv_str(jv_get(ti, "path"), NULL), *isha = jv_str(jv_get(ti, "sha256"), NULL);
    const char *lnm = jv_str(jv_get(tl, "path"), NULL), *lsha = jv_str(jv_get(tl, "sha256"), NULL);
    if (!safe_relname(inm, 255) || !safe_relname(lnm, 255))
        return !replay_fail(rc, "teacher", "safe basenames for topk_idx and topk_logp", "invalid");
    char idx_p[4096] = "", logp_p[4096] = "";
    bool explicit_dir = o->train_teacher != NULL;
    if (explicit_dir) {
        if (!fmt_checked(rp->teacher_dir, sizeof rp->teacher_dir, "%s", o->train_teacher) ||
            !fmt_checked(idx_p, sizeof idx_p, "%s/%s", rp->teacher_dir, inm) ||
            !fmt_checked(logp_p, sizeof logp_p, "%s/%s", rp->teacher_dir, lnm))
            return !replay_fail(rc, "teacher", "path fitting limit", "too long");
    } else {
        char cand[4096], sub[4096], rdir[4096], ddir[4096];
        if (!relocate_local(inm, o->replay_train, cand, sizeof cand))
            return !replay_fail(rc, "teacher", "valid local path", inm);
        if (!record_regular_file_nofollow(cand)) {
            bool sym = false;
            if (replay_parent_dir(o->replay_train, rdir, sizeof rdir) &&
                probe_teacher_subdir(rdir, inm, sub, sizeof sub, &sym)) {
                if (!fmt_checked(idx_p, sizeof idx_p, "%s", sub))
                    return !replay_fail(rc, "teacher", "path fitting limit", "too long");
            } else if (!sym && replay_parent_dir(rp->data_path, ddir, sizeof ddir) &&
                       probe_teacher_subdir(ddir, inm, sub, sizeof sub, &sym)) {
                if (!fmt_checked(idx_p, sizeof idx_p, "%s", sub))
                    return !replay_fail(rc, "teacher", "path fitting limit", "too long");
            }
            if (sym)
                return !replay_fail(rc, "teacher", "regular non-symlink teacher directory", "symlink");
        }
        if (!idx_p[0] && !fmt_checked(idx_p, sizeof idx_p, "%s", cand))
            return !replay_fail(rc, "teacher", "path fitting limit", "too long");
        if (!replay_parent_dir(idx_p, rp->teacher_dir, sizeof rp->teacher_dir) ||
            !fmt_checked(logp_p, sizeof logp_p, "%s/%s", rp->teacher_dir, lnm))
            return !replay_fail(rc, "teacher", "path fitting limit", "too long");
    }
    return replay_check_sha_opt("teacher.topk_idx.sha256", idx_p, isha, !explicit_dir, rc) &&
           replay_check_sha_opt("teacher.topk_logp.sha256", logp_p, lsha, !explicit_dir, rc);
}

static bool safe_eval_label(const char *s) {
    if (!s || !*s || strlen(s) >= 64) return false;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p < 0x20 || *p == 0x7f || *p == '=' || *p == ',') return false;
    }
    return true;
}

static bool validate_evals_and_kill_meta(jv *rec, size_t kill_cap, int *rc) {
    jv *evals = jv_get(rec, "evals");
    jv *kill_v = jv_get(rec, "kill");
    if (evals) {
        if (evals->type != J_ARR || evals->n > 8)
            return !replay_fail(rc, "evals", "array of <= 8 eval sets", "invalid");
        for (int i = 0; i < evals->n; i++) {
            jv *e = evals->items[i];
            if (!e || e->type != J_OBJ)
                return !replay_fail(rc, "evals", "eval set objects", "invalid");
            const char *lbl = jv_str(jv_get(e, "label"), NULL);
            if (!safe_eval_label(lbl))
                return !replay_fail(rc, "evals.label", "safe label under 64 bytes", lbl);
            for (int j = 0; j < i; j++) {
                const char *prev = jv_str(jv_get(evals->items[j], "label"), "");
                if (strcmp(lbl, prev) == 0)
                    return !replay_fail(rc, "evals.label", "unique eval labels", lbl);
            }
            jv *rw = jv_get(e, "rows");
            if (!rw || rw->type != J_OBJ)
                return !replay_fail(rc, "evals.rows", "rows object", "missing");
            jv *ti = jv_get(e, "topk_idx"), *tl = jv_get(e, "topk_logp");
            if ((ti || tl) && (!ti || !tl || ti->type != J_OBJ || tl->type != J_OBJ))
                return !replay_fail(rc, "evals.teacher", "topk_idx and topk_logp objects", "invalid");
        }
    }
    if (kill_v) {
        if (kill_v->type != J_OBJ)
            return !replay_fail(rc, "kill", "kill object", "invalid");
        int kat = 0;
        double kmin = 0;
        const char *km = jv_str(jv_get(kill_v, "metric"), NULL);
        bool match = false;
        if (km && *km && evals) {
            for (int i = 0; i < evals->n; i++) {
                const char *lbl = jv_str(jv_get(evals->items[i], "label"), "");
                if (strcmp(lbl, km) == 0) { match = true; break; }
            }
        }
        if (!km || !*km || strlen(km) >= kill_cap ||
            !record_int(kill_v, "at", 1, INT_MAX, &kat) ||
            !record_num(kill_v, "min_rel", -1, 1, &kmin) ||
            !evals || evals->n == 0 || !match)
            return !replay_fail(rc, "kill", "valid kill rule matching an eval label", km ? km : "invalid");
    }
    return true;
}

static bool replay_resolve_evals(const train_record_opts *o, cli_replay *rp, jv *rec, jv *kill_v, int *rc) {
    jv *evals = jv_get(rec, "evals");
    if (!evals) return true;
    for (int i = 0; i < evals->n; i++) {
        jv *e = evals->items[i], *rw = jv_get(e, "rows");
        const char *lbl = jv_str(jv_get(e, "label"), NULL);
        const char *rpath = jv_str(jv_get(rw, "path"), NULL), *rsha = jv_str(jv_get(rw, "sha256"), NULL);
        char rbuf[4096];
        const char *rf = relocate_local(rpath, o->replay_train, rbuf, sizeof rbuf) ? rbuf : NULL;
        if (!replay_check_sha_opt("evals.rows.sha256", rf, rsha, true, rc)) return false;
        jv *ti = jv_get(e, "topk_idx"), *tl = jv_get(e, "topk_logp");
        char tdir[4096] = "";
        if (ti || tl) {
            const char *inm = jv_str(jv_get(ti, "path"), NULL), *isha = jv_str(jv_get(ti, "sha256"), NULL);
            const char *lnm = jv_str(jv_get(tl, "path"), NULL), *lsha = jv_str(jv_get(tl, "sha256"), NULL);
            if (!safe_relname(inm, 255) || !safe_relname(lnm, 255))
                return !replay_fail(rc, "evals.teacher", "safe basenames", "invalid");
            char ibuf[4096], lbuf[4096];
            const char *ip = relocate_local(inm, o->replay_train, ibuf, sizeof ibuf) ? ibuf : NULL;
            if (ip && !record_regular_file_nofollow(ip)) {
                char edir[4096];
                if (!replay_parent_dir(rf, edir, sizeof edir) ||
                    !fmt_checked(lbuf, sizeof lbuf, "%s/%s", edir, inm) ||
                    (record_regular_file_nofollow(lbuf) && !fmt_checked(ibuf, sizeof ibuf, "%s", lbuf)))
                    return !replay_fail(rc, "evals", "path fitting limit", "too long");
                if (record_regular_file_nofollow(lbuf)) ip = ibuf;
            }
            if (!ip || !replay_check_sha_opt("evals.topk_idx.sha256", ip, isha, true, rc)) return false;
            if (!replay_parent_dir(ip, tdir, sizeof tdir) ||
                !fmt_checked(lbuf, sizeof lbuf, "%s/%s", tdir, lnm))
                return !replay_fail(rc, "evals", "path fitting limit", "too long");
            if (!replay_check_sha_opt("evals.topk_logp.sha256", lbuf, lsha, true, rc)) return false;
        }
        if (kill_v) {
            bool ok = tdir[0]
                ? fmt_checked(rp->eval_specs[i], sizeof rp->eval_specs[i], "%s=%s,%s", lbl, rf, tdir)
                : fmt_checked(rp->eval_specs[i], sizeof rp->eval_specs[i], "%s=%s", lbl, rf);
            if (!ok) return !replay_fail(rc, "evals", "spec fitting limit", "too long");
        }
    }
    return true;
}

static jv *read_bounded_record(const char *path, bool nofollow, char *err, size_t err_cap,
                               char **buf_out, size_t *len_out) {
    if (nofollow ? !record_regular_file_nofollow(path) : !record_regular_file(path)) {
        fmt_checked(err, err_cap, "cannot read %s", path);
        return NULL;
    }
    plat_stat_t st;
    if (plat_stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        fmt_checked(err, err_cap, "cannot read %s", path);
        return NULL;
    }
    if ((uint64_t)st.st_size > TRAIN_RECORD_MAX_BYTES) {
        fmt_checked(err, err_cap, "%s is too large (records up to %u MiB are supported)",
                    path, TRAIN_RECORD_MAX_BYTES / (1024u * 1024u));
        return NULL;
    }
    size_t n = 0;
    char *b = plat_read_file(path, TRAIN_RECORD_MAX_BYTES, &n);
    if (!b) {
        fmt_checked(err, err_cap, "cannot read %s", path);
        return NULL;
    }
    jv *rec = json_parse(b, n);
    if (!rec || rec->type != J_OBJ) {
        jv_free(rec);
        free(b);
        fmt_checked(err, err_cap, "%s is not a valid JSON training record", path);
        return NULL;
    }
    if (buf_out) *buf_out = b;
    else free(b);
    if (len_out) *len_out = n;
    return rec;
}

static bool replay_check_compat(const train_record_host *host, const train_record_opts *o,
                                jv *rec, const char *rec_buf, size_t rec_len,
                                bool *want_t3_exact, int *rc) {
    if (strcmp(jv_str(jv_get(rec, "schema_version"), ""), "xyntetik.runner.train.v1"))
        return !replay_fail(rc, "schema_version", "xyntetik.runner.train.v1",
                            jv_str(jv_get(rec, "schema_version"), ""));
    char pub[SIGN_PUBHEX_CAP];
    receipt_sig_state st = receipt_signature_check(rec_buf, rec_len, pub);
    int sig = st == RSIG_NONE ? 1 : (st != RSIG_OK ? 2 :
              (o->trust_key && *o->trust_key && strcmp(o->trust_key, pub) != 0 ? 2 : 0));
    if (sig == 2) return !replay_fail(rc, "signature", "valid signature", "bad, malformed or untrusted");
    if (sig == 1 && o->trust_key)
        return !replay_fail(rc, "signature", "signed record (--trust-key)", "unsigned");
    const char *det = jv_str(jv_get(rec, "determinism"), "");
    if (jv_get(rec, "products") || strcmp(det, "gpu-within-tolerance") == 0)
        return !replay_fail(rc, "determinism", "cpu-t3-exact or cpu-same-binary", "gpu-within-tolerance");
    if (strcmp(det, "cpu-t3-exact") != 0 && strcmp(det, "cpu-same-binary") != 0)
        return !replay_fail(rc, "determinism", "cpu-t3-exact or cpu-same-binary", det);
    *want_t3_exact = strcmp(det, "cpu-t3-exact") == 0;
    const char *rec_runner = jv_str(jv_get(rec, "runner"), "");
    if (strcmp(rec_runner, host->runner_version) != 0)
        return !replay_fail(rc, "runner", rec_runner, host->runner_version);
    jv *bld = jv_get(rec, "build");
    const char *rec_flavor = jv_str(jv_get(bld, "flavor"), "");
    const char *cur_flavor = host->build_flavor ? host->build_flavor : "default";
    if (strcmp(det, "cpu-t3-exact") == 0) {
        if (strcmp(rec_flavor, "t3") != 0 || strcmp(cur_flavor, "t3") != 0)
            return !replay_fail(rc, "build.flavor", "t3",
                                strcmp(rec_flavor, "t3") ? rec_flavor : cur_flavor);
        // R17.9: exact means a clean stamped commit on both sides. A record
        // that says otherwise beside the word, or names a switch or a reason,
        // is not one a T3 run writes; a dirty or unstamped replaying binary
        // is not the commit's.
        const char *rcm = jv_str(jv_get(bld, "commit"), NULL);
        const char *ccm = host->build_commit;
        jv *rdirty = jv_get(bld, "dirty");
        if (!rdirty || rdirty->type != J_BOOL || rdirty->b)
            return !replay_fail(rc, "build.dirty", "false", rdirty && rdirty->type == J_BOOL ? "true" : "missing");
        if (host->build_dirty)
            return !replay_fail(rc, "build.dirty", "a clean replaying build", "this binary is dirty");
        if (!rcm || !ccm)
            return !replay_fail(rc, "build.commit", rcm ? rcm : "a commit", ccm ? ccm : "none");
        if (strcmp(rcm, ccm) != 0)
            return !replay_fail(rc, "build.commit", rcm, ccm);
        if (jv_get(rec, "environment") || jv_get(rec, "determinism_reasons"))
            return !replay_fail(rc, "determinism", "cpu-t3-exact with no switch and no reason",
                                "a record that names one");
    } else {
        if (!o->allow_same_binary)
            return !replay_fail(rc, "determinism", "cpu-t3-exact (or --allow-same-binary)", det);
        char *exe = plat_executable_path();
        char cur_sha[65] = "";
        bool xok = exe && envelope_file_sha256(exe, cur_sha);
        free(exe);
        const char *want_sha = jv_str(jv_get(bld, "binary_sha256"), "");
        if (!envelope_is_sha256_hex(want_sha))
            return !replay_fail(rc, "build.binary_sha256", "64-hex sha256", want_sha);
        if (!xok || strcmp(cur_sha, want_sha) != 0)
            return !replay_fail(rc, "build.binary_sha256", want_sha, xok ? cur_sha : "unreadable");
    }
    return true;
}

// R17.9: the replay runs with the record's switches and no other: each of the
// record's must be set here to its value ("(set)", a credential's, only set),
// and none a record names (train_env_counts) may be set here that it lacks.
static bool replay_check_environment(const train_record_host *host, jv *rec, int *rc) {
    jv *want = jv_get(rec, "environment");
    if (want && want->type != J_OBJ) return !replay_fail(rc, "environment", "an object", "invalid");
    char wb[600], gb[600];
    for (int i = 0; want && i < want->n; i++) {
        const char *k = want->keys[i];
        size_t kl = strlen(k);
        if (strchr(k, '=') || !train_env_counts(k, kl)) {
            snprintf(gb, sizeof gb, "%.200s not a training switch", k);
            return !replay_fail(rc, "environment", "known training switches only", gb);
        }
        const char *v = jv_str(want->items[i], NULL), *here = getenv(k);
        bool set_ok = v && !strcmp(v, "(set)") && train_env_is_secret(k, kl);
        if (v && here && (!strcmp(v, here) || set_ok)) continue;
        snprintf(wb, sizeof wb, "%.200s=%.200s", k, v ? v : "?");
        snprintf(gb, sizeof gb, "%.200s %s", k, here ? "set otherwise" : "unset");
        return !replay_fail(rc, "environment", wb, gb);
    }
    for (char **e = host->environ_list; e && *e; e++) {
        const char *eq = strchr(*e, '=');
        if (!eq) continue;
        size_t kl = (size_t)(eq - *e);
        if (!train_env_counts(*e, kl)) continue;   // the record's own rule
        char *kbuf = malloc(kl + 1);
        if (!kbuf) { *rc = 1; return false; }
        memcpy(kbuf, *e, kl);
        kbuf[kl] = 0;
        bool present = want && jv_get(want, kbuf) != NULL;
        free(kbuf);
        if (!present) {
            snprintf(gb, sizeof gb, "%.*s set", (int)(kl > 500 ? 500 : kl), *e);
            return !replay_fail(rc, "environment", "no such switch", gb);
        }
    }
    return true;
}

// R17.9: the model overrides the run took, applied from the record. One the
// caller passes is refused unless it is the record's own.
static bool replay_apply_overrides(train_record_opts *o, jv *rec, int *rc) {
    jv *mo = jv_get(rec, "model_overrides");
    if (mo && mo->type != J_OBJ) return !replay_fail(rc, "model_overrides", "an object", "invalid");
    static const char *const keys[] = { "rope_base", "rope_scale", "yarn_factor" };
    float *fields[] = { &o->mp.rope_base, &o->mp.rope_scale, &o->mp.yarn_factor };
    char wb[64], gb[64], field[64];
    for (int i = 0; i < 3; i++) {
        double want = 0;
        jv *v = mo ? jv_get(mo, keys[i]) : NULL;
        if (v && !record_num(mo, keys[i], 0, FLT_MAX, &want))
            return !replay_fail(rc, "model_overrides", "numbers", "invalid");
        float given = *fields[i];
        if (given > 0 && given != (float)want) {
            snprintf(field, sizeof field, "model_overrides.%s", keys[i]);
            snprintf(wb, sizeof wb, v ? "%.9g" : "none", want);
            snprintf(gb, sizeof gb, "%.9g", (double)given);
            return !replay_fail(rc, field, wb, gb);
        }
        *fields[i] = (float)want;
    }
    // --mtp is not one: training never reads the predictor (#925's measurement)
    return true;
}

// R17.9(c), exact unlearning as a retrain: DATA1's lines less every occurrence
// of D's lines, the rest in their order, each closed by "\n", into *kept. A
// line is dropped when it equals one of D's non-empty lines byte for byte (its
// newline aside). *count is the rows removed, *distinct the lines D lists.
// False, with why, for a text corpus (it has no rows), a line of D that is not
// in DATA1, or a file that cannot be read.
typedef struct {
    const char *p;
    size_t      len;
} unlearn_line;

static int unlearn_line_cmp(const void *va, const void *vb) {
    const unlearn_line *a = va, *b = vb;
    size_t m = a->len < b->len ? a->len : b->len;
    int c = m ? memcmp(a->p, b->p, m) : 0;
    if (c) return c;
    return a->len < b->len ? -1 : (a->len > b->len ? 1 : 0);
}

static bool unlearn_split(const char *data1, const char *drop, sbuf *kept, int *count, int *distinct,
                          char *why, size_t cap) {
    size_t n1 = 0, n2 = 0;
    char *a = plat_read_file(data1, MAX_INPUT_FILE, &n1);
    char *d = a ? plat_read_file(drop, MAX_INPUT_FILE, &n2) : NULL;
    unlearn_line *dl = NULL;
    bool *found = NULL, ok = false;
    int nd = 0, capn = 0;
    *count = *distinct = 0;
    if (!a || !d) { snprintf(why, cap, "cannot read %s", !a ? data1 : drop); goto out; }
    if (memchr(a, '\0', n1)) { snprintf(why, cap, "%s contains a NUL byte", data1); goto out; }
    if (memchr(d, '\0', n2)) { snprintf(why, cap, "%s contains a NUL byte", drop); goto out; }
    size_t i0 = 0;
    while (i0 < n1 && (a[i0] == ' ' || a[i0] == '\t' || a[i0] == '\r' || a[i0] == '\n')) i0++;
    if (i0 == n1 || a[i0] != '{') {
        snprintf(why, cap, "%s is not JSONL rows: a text corpus has no row to drop", data1);
        goto out;
    }
    for (size_t p = 0; p < n2; ) {
        const char *e = memchr(d + p, '\n', n2 - p);
        size_t len = e ? (size_t)(e - (d + p)) : n2 - p;
        if (len) {
            if (nd == INT_MAX) { snprintf(why, cap, "%s has too many rows", drop); goto out; }
            if (nd == capn) {
                if (capn > INT_MAX / 2) { snprintf(why, cap, "%s has too many rows", drop); goto out; }
                capn = capn ? 2 * capn : 16;
                unlearn_line *g = realloc(dl, sizeof *dl * (size_t)capn);
                if (!g) { snprintf(why, cap, "out of memory"); goto out; }
                dl = g;
            }
            dl[nd++] = (unlearn_line){ d + p, len };
        }
        p += e ? len + 1 : len;
    }
    if (nd == 0) { snprintf(why, cap, "%s lists no row", drop); goto out; }
    qsort(dl, (size_t)nd, sizeof *dl, unlearn_line_cmp);
    int u = 1;   // distinct, in place
    for (int i = 1; i < nd; i++)
        if (unlearn_line_cmp(&dl[i], &dl[u - 1]) != 0) dl[u++] = dl[i];
    nd = u;
    found = calloc((size_t)nd, sizeof *found);
    if (!found) { snprintf(why, cap, "out of memory"); goto out; }
    for (size_t p = 0; p < n1; ) {
        const char *e = memchr(a + p, '\n', n1 - p);
        size_t len = e ? (size_t)(e - (a + p)) : n1 - p;
        unlearn_line key = { a + p, len };
        unlearn_line *hit = len ? bsearch(&key, dl, (size_t)nd, sizeof *dl, unlearn_line_cmp) : NULL;
        if (hit) { found[hit - dl] = true; (*count)++; }
        else if (len || e) { sb_put(kept, a + p, len); sb_lit(kept, "\n"); }
        p += e ? len + 1 : len;
    }
    for (int i = 0; i < nd; i++)
        if (!found[i]) {
            snprintf(why, cap, "a line of %s is not in the parent's data: %.*s",
                     drop, (int)(dl[i].len > 60 ? 60 : dl[i].len), dl[i].p);
            goto out;
        }
    *distinct = nd;
    ok = !kept->failed;
    if (!ok) snprintf(why, cap, "out of memory");
out:
    free(dl);
    free(found);
    free(a);
    free(d);
    return ok;
}

static bool has_jsonl_suffix(const char *p) {
    size_t n = p ? strlen(p) : 0;
    return n >= 6 && strcmp(p + n - 6, ".jsonl") == 0;
}

static const char *unlearn_record_eligible(jv *rec) {
    if (!rec || rec->type != J_OBJ)
        return "is not a valid JSON training record";
    if (strcmp(jv_str(jv_get(rec, "schema_version"), ""), "xyntetik.runner.train.v1"))
        return "is not a xyntetik.runner.train.v1 training record";
    if (strcmp(jv_str(jv_get(rec, "objective"), ""), "weighted-cross-entropy"))
        return "is not a cross-entropy run (a teacher's arrays and DPO pairs are not rows to drop)";
    if (jv_get(rec, "from_adapter") || jv_get(rec, "resumed_at_step"))
        return "continued another adapter (an unlearned run starts from the base)";
    if (jv_get(rec, "full_weight"))
        return "trained the full weights";
    if (jv_get(rec, "evals") || jv_get(rec, "kill"))
        return "has evals or a kill rule, whose files would need the rows dropped too";
    const char *dpath = jv_str(jv_get(jv_get(rec, "data"), "path"), "");
    if (!has_jsonl_suffix(dpath))
        return "does not train on .jsonl rows";
    return NULL;
}

static bool unlearn_check_eligible(const train_record_opts *o, jv *rec, int *rc) {
    const char *why = unlearn_record_eligible(rec);
    if (!why) return true;
    fputs("error: --unlearn: ", stderr);
    envelope_print_safe(stderr, o->unlearn_parent, 0, "?", false);
    fprintf(stderr, " %s\n", why);
    *rc = 1;
    return false;
}

// --unlearn: the parent record a retrain may follow, and the retained rows,
// written beside the output. A fresh cross-entropy adapter only: a teacher's
// arrays are indexed by row, a continued or resumed run starts from another
// adapter, and evals and a kill rule would need their own files dropped too.
static bool unlearn_prepare(train_record_opts *o, jv *rec, int *rc) {
    char have[65] = "";
    const char *want = jv_str(jv_get(jv_get(rec, "data"), "sha256"), "");
    if (!envelope_is_sha256_hex(want)) {
        fputs("error: --unlearn: parent record has invalid data.sha256: ", stderr);
        envelope_print_safe(stderr, want, 128, "missing", true);
        fputc('\n', stderr);
        *rc = 1;
        return false;
    }
    if (!envelope_file_sha256(o->train_path, have) || strcmp(have, want)) {
        fputs("error: --unlearn: ", stderr);
        envelope_print_safe(stderr, o->train_path, 0, "?", false);
        fprintf(stderr, " is not the parent's data (sha256 %s, the record names ",
                have[0] ? have : "unreadable");
        envelope_print_safe(stderr, want, 128, "missing", true);
        fputs(")\n", stderr);
        *rc = 1;
        return false;
    }
    sbuf kept = {0};
    char swhy[512];
    if (!unlearn_split(o->train_path, o->unlearn_drop, &kept, &o->unlearn_count, &o->unlearn_distinct,
                       swhy, sizeof swhy)) {
        fputs("error: --unlearn: ", stderr);
        envelope_print_safe(stderr, swhy, 0, "failed", false);
        fputc('\n', stderr);
        free(kept.s);
        *rc = 1;
        return false;
    }
    if (kept.n == 0) {
        fputs("error: --unlearn: ", stderr);
        envelope_print_safe(stderr, o->unlearn_drop, 0, "?", false);
        fputs(" drops every row of ", stderr);
        envelope_print_safe(stderr, o->train_path, 0, "?", false);
        fputs(": nothing is left to train on\n", stderr);
        free(kept.s);
        *rc = 1;
        return false;
    }
    static char path[4200];
    if (!fmt_checked(path, sizeof path, "%s.retained.jsonl", o->train_out)) {
        fputs("error: --unlearn: retained path exceeds buffer for ", stderr);
        envelope_print_safe(stderr, o->train_out, 0, "?", false);
        fputc('\n', stderr);
        free(kept.s);
        *rc = 1;
        return false;
    }
    FILE *f = plat_fopen(path, "wb");
    bool ok = f && (kept.n == 0 || fwrite(kept.s, 1, kept.n, f) == kept.n);
    if (f && fclose(f) != 0) ok = false;
    free(kept.s);
    if (!ok) {
        fputs("error: --unlearn: cannot write ", stderr);
        envelope_print_safe(stderr, path, 0, "?", false);
        fputc('\n', stderr);
        *rc = 1;
        return false;
    }
    o->unlearn_retained = path;
    return true;
}

// Two JSON values the same: type, number, string, bool, and objects and arrays
// member for member (an absent value is the same only as another absent one).
static bool jv_same(const jv *a, const jv *b) {
    if (!a || !b) return a == b;
    if (a->type != b->type) return false;
    switch (a->type) {
    case J_NULL: return true;
    case J_BOOL: return a->b == b->b;
    case J_NUM:  return a->num_int == b->num_int &&
                      (a->num_text || b->num_text
                           ? a->num_text && b->num_text && !strcmp(a->num_text, b->num_text)
                           : a->num == b->num);
    case J_STR:  return !strcmp(a->str, b->str);
    case J_ARR:
        if (a->n != b->n) return false;
        for (int i = 0; i < a->n; i++) if (!jv_same(a->items[i], b->items[i])) return false;
        return true;
    case J_OBJ:
        if (a->n != b->n) return false;
        for (int i = 0; i < a->n; i++)
            if (!jv_same(a->items[i], jv_get((jv *)b, a->keys[i]))) return false;
        return true;
    }
    return false;
}

// attest unlearn: the record's unlearn object against the files the auditor
// holds, before any replay: the parent record, its data, the dropped rows, and
// DATA1 less D as the record's own data.
static bool unlearn_audit_check(const train_record_opts *o, jv *rec, const char *rec_path, int *rc) {
    const char *c_why = unlearn_record_eligible(rec);
    if (c_why)
        return !replay_fail(rc, "unlearn", "eligible weighted-cross-entropy LoRA .jsonl run", c_why);
    jv *un = jv_get(rec, "unlearn");
    if (!un || un->type != J_OBJ)
        return !replay_fail(rc, "unlearn", "a record of runner train --unlearn", "none");
    int c = 0, dd = 0;
    if (!record_int(un, "dropped_count", 1, INT_MAX, &c) ||
        !record_int(un, "dropped_distinct", 1, INT_MAX, &dd))
        return !replay_fail(rc, "unlearn.dropped_count", "positive integers", "invalid");
    jv *pr = jv_get(un, "parent_record");
    if (!pr || pr->type != J_OBJ)
        return !replay_fail(rc, "unlearn.parent_record", "parent_record object", "missing");
    const char *ppath = jv_str(jv_get(pr, "path"), NULL);
    char pbuf[4096];
    const char *pf = relocate_local(ppath, rec_path, pbuf, sizeof pbuf) ? pbuf : NULL;
    if (!replay_check_sha_opt("unlearn.parent_record.sha256", pf, jv_str(jv_get(pr, "sha256"), NULL), true, rc))
        return false;
    char rerr[512] = "";
    jv *parent = read_bounded_record(pf, true, rerr, sizeof rerr, NULL, NULL);
    if (!parent)
        return !replay_fail(rc, "unlearn.parent_record", "valid JSON training record", rerr[0] ? rerr : "unreadable");
    const char *p_why = unlearn_record_eligible(parent);
    if (p_why) {
        jv_free(parent);
        return !replay_fail(rc, "unlearn.parent_record", "eligible weighted-cross-entropy LoRA .jsonl run", p_why);
    }
    const char *pdata_raw = jv_str(jv_get(jv_get(parent, "data"), "sha256"), NULL);
    if (!envelope_is_sha256_hex(pdata_raw)) {
        replay_fail(rc, "unlearn.parent_data.sha256", "64-hex sha256", pdata_raw);
        jv_free(parent);
        return false;
    }
    char pdata[65];
    memcpy(pdata, pdata_raw, 65);
    // the same base and the same recipe as the parent: every field replay reads
    // to train, the switches and the model overrides. Otherwise a from-scratch
    // run under another recipe, given an unlearn block, would replay to its own
    // bytes and verify.
    static const char *const same[] = {
        "objective", "lora_rank", "alpha", "lora_scale", "lr", "steps", "ctx", "seed", "kv_cache",
        "no_bos", "end_of_turn", "eot_id", "adamw", "recipe", "model_overrides", "environment",
    };
    const char *pbase = jv_str(jv_get(jv_get(parent, "base"), "sha256"), "");
    const char *rbase = jv_str(jv_get(jv_get(rec, "base"), "sha256"), "");
    if (!envelope_is_sha256_hex(pbase)) {
        replay_fail(rc, "unlearn.parent.base.sha256", "64-hex sha256", pbase);
        jv_free(parent);
        return false;
    }
    if (!envelope_is_sha256_hex(rbase)) {
        replay_fail(rc, "base.sha256", "64-hex sha256", rbase);
        jv_free(parent);
        return false;
    }
    bool same_ok = parent && !strcmp(pbase, rbase);
    const char *differs = same_ok ? NULL : "base.sha256";
    for (size_t i = 0; same_ok && i < sizeof same / sizeof *same; i++)
        if (!jv_same(jv_get(parent, same[i]), jv_get(rec, same[i]))) { same_ok = false; differs = same[i]; }
    jv_free(parent);
    if (!same_ok) {
        char field[64];
        snprintf(field, sizeof field, "unlearn.%s", differs);
        return !replay_fail(rc, field, "the parent's", "another");
    }
    if (!replay_check_sha("unlearn.parent_data.sha256", o->unlearn_parent_data, pdata, rc)) return false;
    if (!replay_check_sha("unlearn.dropped_sha256", o->unlearn_dropped,
                          jv_str(jv_get(un, "dropped_sha256"), NULL), rc))
        return false;
    sbuf kept = {0};
    int count = 0, distinct = 0;
    char why[512], got[65] = "";
    if (!unlearn_split(o->unlearn_parent_data, o->unlearn_dropped, &kept, &count, &distinct, why, sizeof why)) {
        free(kept.s);
        return !replay_fail(rc, "unlearn.retained", "DATA1 less D's lines", why);
    }
    if (kept.n == 0) {
        free(kept.s);
        return !replay_fail(rc, "unlearn.retained", "non-empty retained rows", "empty retained set");
    }
    envelope_data_sha256(kept.s, kept.n, got);
    free(kept.s);
    const char *want = jv_str(jv_get(jv_get(rec, "data"), "sha256"), "");
    if (!envelope_is_sha256_hex(want)) return !replay_fail(rc, "data.sha256", "64-hex sha256", want);
    if (strcmp(got, want)) return !replay_fail(rc, "unlearn.retained", want, got);
    if (c != count || dd != distinct) {
        char w[64], g[64];
        snprintf(w, sizeof w, "%d rows of %d lines", c, dd);
        snprintf(g, sizeof g, "%d rows of %d lines", count, distinct);
        return !replay_fail(rc, "unlearn.dropped_count", w, g);
    }
    return true;
}

// R17.9 step 4: attest replay RECORD --segment A..B re-runs steps A+1..B from
// the step-A checkpoint beside the record and compares every checkpoint the
// segment writes with the record's. Before anything trains: the modes segments
// do not cover, refused by name; the segment's shape; the record's
// checkpoints and their chain, whose head is recomputed from the entries and
// whose h_0 from the record's own bytes before the chain fields (the record
// read once, bounded, by read_bounded_record); then the start files, hashed
// before the resume parses them. Each refusal says what it saw.
static bool seg_refuse(int *rc, int code, const char *what) {
    fputs("error: --segment: ", stderr);
    fputs(what, stderr);
    fputc('\n', stderr);
    *rc = code;
    return false;
}

static const char *seg_find(const char *buf, size_t n, const char *mark) {
    size_t m = strlen(mark);
    for (size_t i = 0; i + m <= n; i++)
        if (buf[i] == mark[0] && memcmp(buf + i, mark, m) == 0) return buf + i;
    return NULL;
}

static bool seg_modes_ok(jv *rec, int *rc) {
    if (jv_get(rec, "teacher") || !strcmp(jv_str(jv_get(rec, "objective"), ""), "kl-to-teacher"))
        return seg_refuse(rc, 2, "a record of a run with a teacher (distillation) is not replayed by segments");
    if (jv_get(rec, "dpo") || !strcmp(jv_str(jv_get(rec, "objective"), ""), "dpo"))
        return seg_refuse(rc, 2, "a DPO record is not replayed by segments");
    if (jv_get(rec, "variant") || jv_get(rec, "full_weight"))
        return seg_refuse(rc, 2, "a --train-full record is not replayed by segments");
    if (jv_get(rec, "resumed_at_step"))
        return seg_refuse(rc, 2, "a record of a resumed run is not replayed by segments: replay the run it resumed");
    return true;
}

static bool seg_chain(jv *rec, const char *rbuf, size_t rlen, cli_replay *rp, int *rc);

static bool segment_precheck(const train_record_opts *o, jv *rec, const char *rbuf, size_t rlen,
                             const char *rec_path, cli_replay *rp, int *rc) {
    if (!seg_modes_ok(rec, rc)) return false;
    const char *s = o->replay_segment;
    char *e1 = NULL, *e2 = NULL;
    long a = strtol(s, &e1, 10), b = -1;
    if (e1 != s && strncmp(e1, "..", 2) == 0) b = strtol(e1 + 2, &e2, 10);
    if (e1 == s || b < 0 || e2 == e1 + 2 || *e2 || a < 0 || a > INT_MAX || b > INT_MAX)
        return seg_refuse(rc, 1, "wants A..B: two checkpoint steps the record names (A may be 0, the run's start)");
    if (b <= a) return seg_refuse(rc, 1, "the segment ends before it starts");
    if (!seg_chain(rec, rbuf, rlen, rp, rc)) return false;
    int ia = -1, ib = -1;
    for (int i = 0; i < rp->seg_n; i++) {
        if (rp->seg_ck[i].step == a) ia = i;
        if (rp->seg_ck[i].step == b) ib = i;
    }
    char msg[128];
    if ((a != 0 && ia < 0) || ib < 0) {
        snprintf(msg, sizeof msg, "step %ld is not a checkpoint the record names", a != 0 && ia < 0 ? a : b);
        return seg_refuse(rc, 1, msg);
    }
    if (a > 0) {
        const char *ap = jv_str(jv_get(jv_get(rec, "adapter"), "path"), "");
        char name[320], opt[4200], have[65] = "", have_o[65] = "";
        if (!safe_relname(ap, 255) || !fmt_checked(name, sizeof name, "%s.step-%05ld", ap, a) ||
            !relocate_local(name, rec_path, rp->seg_start, sizeof rp->seg_start) ||
            !fmt_checked(opt, sizeof opt, "%s.optim", rp->seg_start))
            return !replay_fail(rc, "adapter.path", "a safe base name", ap);
        // the record's other cited files' rule: a regular file by its base
        // name, never a symlink (envelope_cited_regular_sha256: lstat, no-follow)
        bool ha = envelope_cited_regular_sha256(rp->seg_start, have);
        bool ho = envelope_cited_regular_sha256(opt, have_o);
        if (!ha || !ho) {
            fputs("error: --segment: ", stderr);
            envelope_print_safe(stderr, !ha ? rp->seg_start : opt, 0, "?", false);
            fprintf(stderr, " is missing or not a regular file (a symlink is not followed) (step %ld)\n", a);
            *rc = 2;
            return false;
        }
        if (strcmp(have, rp->seg_ck[ia].adapter) != 0 || strcmp(have_o, rp->seg_ck[ia].optim) != 0) {
            fputs("error: --segment: ", stderr);
            envelope_print_safe(stderr, strcmp(have, rp->seg_ck[ia].adapter) != 0 ? rp->seg_start : opt,
                                0, "?", false);
            fprintf(stderr, " is not the checkpoint the record names (step %ld)\n", a);
            *rc = 2;
            return false;
        }
    }
    rp->segment = true;
    rp->seg_a = (int)a;
    rp->seg_b = (int)b;
    return true;
}

// The record's checkpoints and their chain: each entry validated, the head
// recomputed from h0 and the entries, h0 from the record's own bytes before
// the chain fields, and the interval the run wrote them at (every K steps and
// the last). Shared by --segment and --sample.
static bool seg_chain(jv *rec, const char *rbuf, size_t rlen, cli_replay *rp, int *rc) {
    jv *cks = jv_get(rec, "checkpoints"), *ch = jv_get(rec, "checkpoint_chain");
    if (!cks || cks->type != J_ARR || cks->n < 1 || !ch || ch->type != J_OBJ)
        return seg_refuse(rc, 2, "the record names no checkpoints (a run without --save-every, or older than segment records)");
    if (cks->n > (int)(sizeof rp->seg_ck / sizeof rp->seg_ck[0]))
        return seg_refuse(rc, 2, "the record names more checkpoints than a segment replay takes (256)");
    const char *h0 = jv_str(jv_get(ch, "h0"), NULL), *head = jv_str(jv_get(ch, "head"), NULL);
    if (strcmp(jv_str(jv_get(ch, "basis"), ""), "xyntetik.runner.train.chain.v1") != 0 ||
        !envelope_is_sha256_hex(h0) || !envelope_is_sha256_hex(head))
        return !replay_fail(rc, "checkpoint_chain", "a v1 chain with a hex h0 and head", "invalid");
    char h[65], line[512];
    memcpy(h, h0, sizeof h);
    for (int i = 0; i < cks->n; i++) {
        jv *c = cks->items[i];
        int step = 0;
        double cur = 0;
        const char *ad = c ? jv_str(jv_get(c, "adapter_sha256"), NULL) : NULL;
        const char *op = c ? jv_str(jv_get(c, "optim_sha256"), NULL) : NULL;
        if (!c || c->type != J_OBJ || !record_int(c, "step", 1, INT_MAX, &step) ||
            !record_num(c, "cursor", 0, 9007199254740992.0, &cur) || cur != trunc(cur) ||
            !envelope_is_sha256_hex(ad) || !envelope_is_sha256_hex(op) ||
            (i > 0 && step <= rp->seg_ck[i - 1].step))
            return !replay_fail(rc, "checkpoints", "ascending {step, cursor, adapter_sha256, optim_sha256}",
                                "invalid");
        rp->seg_ck[i].step = step;
        rp->seg_ck[i].cursor = (long long)cur;
        memcpy(rp->seg_ck[i].adapter, ad, 65);
        memcpy(rp->seg_ck[i].optim, op, 65);
        int n = snprintf(line, sizeof line, "%s|%d|%lld|%s|%s", h, step, (long long)cur, ad, op);
        envelope_data_sha256(line, (size_t)n, h);
    }
    rp->seg_n = cks->n;
    if (strcmp(h, head) != 0) return !replay_fail(rc, "checkpoint_chain.head", head, h);
    const char *at = seg_find(rbuf, rlen, ",\"checkpoints\":[");
    if (!at) return !replay_fail(rc, "checkpoint_chain.h0", h0, "no chain fields in the record's bytes");
    size_t bl = (size_t)(at - rbuf);
    char *body = malloc(bl + 1);
    if (!body) return seg_refuse(rc, 1, "out of memory");
    memcpy(body, rbuf, bl);
    body[bl] = '}';
    char bh[65], hh[65];
    envelope_data_sha256(body, bl + 1, bh);
    free(body);
    int n = snprintf(line, sizeof line, "xyntetik.runner.train.chain.v1|%s", bh);
    envelope_data_sha256(line, (size_t)n, hh);
    if (strcmp(hh, h0) != 0) return !replay_fail(rc, "checkpoint_chain.h0", h0, hh);
    int steps = 0;
    // the run wrote them every K steps and at its last: the replay writes the same
    int every = rp->seg_ck[0].step;
    if (!record_int(rec, "steps", 1, INT_MAX, &steps))
        return !replay_fail(rc, "steps", "integer", "invalid");
    for (int i = 0; i < rp->seg_n; i++)
        if ((long long)rp->seg_ck[i].step != (long long)(i + 1) * every &&
            !(i == rp->seg_n - 1 && rp->seg_ck[i].step == steps))
            return seg_refuse(rc, 2, "the record's checkpoints are not every K steps and the last");
    rp->seg_every = every;
    return true;
}

// R17.9 step 4b, --sample: the record's segments, for the draw. Read as a
// replay reads it (bounded, no-follow), the modes segments do not cover
// refused by name, the chain checked whole. bounds[0] is 0 (the run's start),
// bounds[1..m] the checkpoint steps; head is the chain's head. Each drawn
// segment is then replayed by --segment, which checks the rest.
bool train_record_sample_plan(const train_record_host *host, const train_record_opts *o, int *bounds, int cap,
                               int *m, char head[65], int *rc) {
    const char *rec_path = o->replay_train;
    char err[512] = "";
    char *rbuf = NULL;
    size_t rlen = 0;
    jv *rec = read_bounded_record(rec_path, true, err, sizeof err, &rbuf, &rlen);
    if (!rec) {
        fputs("error: ", stderr);
        envelope_print_safe(stderr, err[0] ? err : rec_path, 0, "?", false);
        fputc('\n', stderr);
        *rc = 1;
        return false;
    }
    // the record's own checks first (schema, signature, determinism,
    // environment), once, before any segment is drawn
    cli_replay *rp = calloc(1, sizeof *rp);
    bool t3 = false;
    bool ok = rp && replay_check_compat(host, o, rec, rbuf, rlen, &t3, rc) && replay_check_environment(host, rec, rc) &&
              seg_modes_ok(rec, rc) && seg_chain(rec, rbuf, rlen, rp, rc);
    if (ok && rp->seg_n + 1 > cap) ok = seg_refuse(rc, 2, "the record names more segments than a sample takes");
    if (ok) {
        bounds[0] = 0;
        for (int i = 0; i < rp->seg_n; i++) bounds[i + 1] = rp->seg_ck[i].step;
        *m = rp->seg_n;
        memcpy(head, jv_str(jv_get(jv_get(rec, "checkpoint_chain"), "head"), ""), 65);
    }
    if (!rp && *rc == 0) *rc = 1;
    free(rp);
    free(rbuf);
    jv_free(rec);
    return ok;
}

bool train_record_prepare(const train_record_host *host, train_record_opts *o, cli_replay *rp, int *rc) {
    // R17.9(c): attest unlearn is a replay of the record after its unlearn
    // checks; --unlearn reads the parent record here for its recipe, retrains
    // into the caller's --train-out, and is no replay (nothing is compared).
    if (o->unlearn_audit && !o->replay_train) o->replay_train = o->unlearn_audit;
    const bool retrain = !o->replay_train && o->unlearn_parent;
    const char *rec_path = o->replay_train ? o->replay_train : o->unlearn_parent;
    if (!rec_path) return false;
    char err[512] = "";
    char *rbuf = NULL;
    size_t rlen = 0;
    jv *rec = read_bounded_record(rec_path, !retrain, err, sizeof err, &rbuf, &rlen);
    if (!rec) {
        fputs("error: ", stderr);
        envelope_print_safe(stderr, err[0] ? err : rec_path, 0, "?", false);
        fputc('\n', stderr);
        *rc = 1;
        return true;
    }
    const char *caller_sign_key = o->sign_key;
    if (retrain) o->replay_data = o->train_path;   // DATA1, hashed as the parent's data below
    bool ok = (retrain ? unlearn_check_eligible(o, rec, rc)
                       : replay_check_compat(host, o, rec, rbuf, rlen, &rp->want_t3_exact, rc)) &&
              replay_check_environment(host, rec, rc) && replay_apply_overrides(o, rec, rc);
    // R17.9 step 4: the segment's checks after the record's own (schema,
    // signature, determinism, environment), before its bytes go
    if (ok && o->replay_segment && !segment_precheck(o, rec, rbuf, rlen, rec_path, rp, rc)) {
        free(rbuf);
        jv_free(rec);
        return true;
    }
    free(rbuf);
    if (!ok && *rc == 0) *rc = 1;
    jv *fw_v = jv_get(rec, "full_weight");
    if (ok && fw_v && fw_v->type != J_OBJ)
        ok = !replay_fail(rc, "full_weight", "full_weight object", "invalid");
    rp->is_full = fw_v && fw_v->type == J_OBJ;
    jv *base = jv_get(rec, "base");
    jv *data = jv_get(rec, "data");
    jv *out_v = jv_get(rec, rp->is_full ? "variant" : "adapter");
    jv *res_v = jv_get(rec, "resumed_at_step");
    jv *from_v = jv_get(rec, "from_adapter");
    jv *tch = jv_get(rec, "teacher");
    jv *kill_v = jv_get(rec, "kill");
    jv *un_v = jv_get(rec, "unlearn");
    jv *bld_v = jv_get(rec, "build");
    jv *adm = jv_get(rec, "adamw"), *rcp = jv_get(rec, "recipe");
    if (ok && (!base || base->type != J_OBJ))
        ok = !replay_fail(rc, "base", "base object", "invalid");
    if (ok && (!data || data->type != J_OBJ))
        ok = !replay_fail(rc, "data", "data object", "invalid");
    if (ok && (!out_v || out_v->type != J_OBJ))
        ok = !replay_fail(rc, rp->is_full ? "variant" : "adapter", "output object", "invalid");
    if (ok && from_v && from_v->type != J_OBJ)
        ok = !replay_fail(rc, "from_adapter", "adapter object", "invalid");
    if (ok && tch && tch->type != J_OBJ)
        ok = !replay_fail(rc, "teacher", "teacher object", "invalid");
    if (ok && kill_v && kill_v->type != J_OBJ)
        ok = !replay_fail(rc, "kill", "kill object", "invalid");
    if (ok && un_v && un_v->type != J_OBJ)
        ok = !replay_fail(rc, "unlearn", "unlearn object", "invalid");
    if (ok && !retrain && (!bld_v || bld_v->type != J_OBJ))
        ok = !replay_fail(rc, "build", "build object", "invalid");
    if (ok && (!adm || adm->type != J_OBJ))
        ok = !replay_fail(rc, "adamw", "adamw object", "invalid");
    if (ok && (!rcp || rcp->type != J_OBJ))
        ok = !replay_fail(rc, "recipe", "recipe object", "invalid");

    const char *out_name = ok ? jv_str(jv_get(out_v, "path"), NULL) : NULL;
    const char *out_sha = ok ? jv_str(jv_get(out_v, "sha256"), NULL) : NULL;
    if (ok && !safe_relname(out_name, 255))
        ok = !replay_fail(rc, rp->is_full ? "variant.path" : "adapter.path",
                          "safe basename", out_name);
    if (ok && !envelope_is_sha256_hex(out_sha))
        ok = !replay_fail(rc, rp->is_full ? "variant.sha256" : "adapter.sha256",
                          "64-hex sha256", out_sha);
    if (ok && (!fmt_checked(rp->want_out_path, sizeof rp->want_out_path, "%s", out_name) ||
               !fmt_checked(rp->want_out_sha, sizeof rp->want_out_sha, "%s", out_sha)))
        ok = !replay_fail(rc, rp->is_full ? "variant.path" : "adapter.path", "path fitting limit", "too long");

    // Validate hyperparameters and cheap scalar fields before any large I/O,
    // unlearning split, or model load.
    const char *obj = jv_str(jv_get(rec, "objective"), "");
    const char *kv = jv_str(jv_get(rec, "kv_cache"), NULL);
    jv *nb = jv_get(rec, "no_bos"), *eot = jv_get(rec, "end_of_turn");
    jv *sv = jv_get(rec, "seed");
    int rank = 0, steps = 0, ctx = 0, eot_id = -1, warm = 0, batch = 1;
    double alpha = 0, lscale = 0, lr = 0, beta = 0.1, wd = 0, lr_floor_d = 0;
    double b1 = 0, b2 = 0, eps = 0;
    uint64_t seed_u64 = 0;
    if (ok) {
        if (!kv || (strcmp(kv, "f16") && strcmp(kv, "f32")))
            ok = !replay_fail(rc, "kv_cache", "f16 or f32", kv);
        else if (!nb || nb->type != J_BOOL)
            ok = !replay_fail(rc, "no_bos", "bool", "missing");
        else if (!eot || eot->type != J_BOOL)
            ok = !replay_fail(rc, "end_of_turn", "bool", "missing");
        else if (!record_int(rec, "lora_rank", rp->is_full ? 0 : 1, 512, &rank) ||
                 !record_num(rec, "alpha", 0, FLT_MAX, &alpha) ||
                 (rp->is_full && (rank != 0 || alpha != 0.0)) ||
                 !record_num(rec, "lora_scale", 0, FLT_MAX, &lscale) ||
                 (!from_v && !res_v && fabs(lscale - 1.0) > 1e-6) ||
                 !record_num(rec, "lr", 0, 1, &lr) ||
                 !record_int(rec, "steps", 1, INT_MAX, &steps) ||
                 !record_int(rec, "ctx", 2, INT_MAX, &ctx) ||
                 !record_int(rec, "eot_id", -1, INT_MAX, &eot_id))
            ok = !replay_fail(rc, "recipe", "valid training hyperparameters", "invalid");
        else if (!sv || sv->type != J_NUM || !sv->num_int ||
                 !(sv->num_text ? parse_u64(sv->num_text, 0, UINT64_MAX, &seed_u64)
                                : (sv->num >= 0 && sv->num <= 9007199254740992.0 &&
                                   sv->num == trunc(sv->num) && (seed_u64 = (uint64_t)sv->num, true))))
            ok = !replay_fail(rc, "seed", "exact uint64", "invalid");
        else if (!record_num(adm, "beta1", 0, 1, &b1) || !record_num(adm, "beta2", 0, 1, &b2) ||
                 !record_num(adm, "eps", 0, 1, &eps) || !record_num(adm, "weight_decay", 0, 1, &wd) ||
                 fabs(b1 - TRAIN_ADAM_BETA1) > 1e-6 || fabs(b2 - TRAIN_ADAM_BETA2) > 1e-6 ||
                 fabs(eps - TRAIN_ADAM_EPS) > 1e-12)
            ok = !replay_fail(rc, "adamw", "matching AdamW constants", "mismatch");
        else if (!record_int(rcp, "warmup", 0, INT_MAX, &warm) ||
                 !record_num(rcp, "floor", 0, 1, &lr_floor_d) ||
                 !record_int(rcp, "batch", 1, 65536, &batch))
            ok = !replay_fail(rc, "recipe", "valid lr/batch config", "invalid");
    }
    if (ok) {
        const char *sched = jv_str(jv_get(rcp, "schedule"), "");
        const char *loss_mode = jv_str(jv_get(rcp, "loss"), "");
        if (strcmp(sched, "constant") && strcmp(sched, "cosine"))
            ok = !replay_fail(rc, "recipe.schedule", "constant or cosine", sched);
        else if (strcmp(loss_mode, "row-sum") && strcmp(loss_mode, "position-mean"))
            ok = !replay_fail(rc, "recipe.loss", "row-sum or position-mean", loss_mode);
        else if (batch > 1 && strcmp(loss_mode, "position-mean") != 0)
            ok = !replay_fail(rc, "recipe.loss", "position-mean", loss_mode);
        else if (rp->want_t3_exact && (rp->is_full || strcmp(obj, "weighted-cross-entropy") != 0))
            ok = !replay_fail(rc, "determinism", "cpu-same-binary", "cpu-t3-exact");
        else if (!strcmp(obj, "dpo")) {
            if (!record_num(rec, "dpo_beta", 0, FLT_MAX, &beta))
                ok = !replay_fail(rc, "dpo_beta", "number", "missing");
        } else if (strcmp(obj, "weighted-cross-entropy") != 0 && strcmp(obj, "kl-to-teacher") != 0) {
            ok = !replay_fail(rc, "objective", "weighted-cross-entropy, kl-to-teacher or dpo", obj);
        }
        if (ok && !validate_evals_and_kill_meta(rec, sizeof rp->kill_metric, rc))
            ok = false;
    }
    if (ok && retrain && !unlearn_prepare(o, rec, rc))
        ok = false;
    else if (ok && o->unlearn_audit && !unlearn_audit_check(o, rec, rec_path, rc))
        ok = false;

    // Locate & hash every input before touching the model.
    if (ok) {
        bool explicit_m = o->model_path != NULL;
        const char *bp = explicit_m ? o->model_path
                                    : (relocate_local(jv_str(jv_get(base, "path"), ""),
                                                      rec_path, rp->base_path, sizeof rp->base_path)
                                       ? rp->base_path : NULL);
        if (!bp || (bp != rp->base_path && !fmt_checked(rp->base_path, sizeof rp->base_path, "%s", bp)))
            ok = !replay_fail(rc, "base.path", "safe path fitting limit", jv_str(jv_get(base, "path"), NULL));
        else
            ok = replay_check_sha_opt("base.sha256", rp->base_path, jv_str(jv_get(base, "sha256"), NULL), !explicit_m, rc);
        if (ok) o->model_path = rp->base_path;
    }
    if (ok) {
        bool explicit_d = o->replay_data != NULL;
        const char *dp = explicit_d ? o->replay_data
                                    : (relocate_local(jv_str(jv_get(data, "path"), ""),
                                                      rec_path, rp->data_path, sizeof rp->data_path)
                                       ? rp->data_path : NULL);
        if (!dp || (dp != rp->data_path && !fmt_checked(rp->data_path, sizeof rp->data_path, "%s", dp)))
            ok = !replay_fail(rc, "data.path", "safe path fitting limit", jv_str(jv_get(data, "path"), NULL));
        else
            ok = replay_check_sha_opt("data.sha256", rp->data_path, jv_str(jv_get(data, "sha256"), NULL), !explicit_d, rc);
    }

    if (ok && (from_v || res_v)) {
        if (res_v && o->lora_path) {
            fprintf(stderr, "error: record resumes a checkpoint (--train-resume), not --lora\n");
            *rc = 1;
            ok = false;
        } else if (!res_v && o->train_resume) {
            fprintf(stderr, "error: record continues from --lora adapter, not --train-resume\n");
            *rc = 1;
            ok = false;
        }
        const char *cli_f = o->train_resume ? o->train_resume : o->lora_path;
        bool explicit_f = cli_f != NULL;
        const char *fp = !ok ? ""
                       : explicit_f ? cli_f
                       : (relocate_local(jv_str(jv_get(from_v, "path"), ""),
                                         rec_path, rp->from_path, sizeof rp->from_path)
                          ? rp->from_path : NULL);
        if (ok && (!fp || (fp != rp->from_path && !fmt_checked(rp->from_path, sizeof rp->from_path, "%s", fp))))
            ok = !replay_fail(rc, "from_adapter.path", "safe path fitting limit", jv_str(jv_get(from_v, "path"), NULL));
        if (ok) ok = replay_check_sha_opt("from_adapter.sha256", rp->from_path, jv_str(jv_get(from_v, "sha256"), NULL), !explicit_f, rc);
        if (ok && res_v) {
            int rs = 0;
            if (!record_int(rec, "resumed_at_step", 0, INT_MAX, &rs))
                ok = !replay_fail(rc, "resumed_at_step", "integer", "invalid");
            char opt_p[4120];
            if (ok && !fmt_checked(opt_p, sizeof opt_p, "%s.optim", rp->from_path))
                ok = !replay_fail(rc, "resume_optim_sha256", "path fitting limit", "too long");
            else if (ok)
                ok = replay_check_sha_opt("resume_optim_sha256", opt_p,
                                          jv_str(jv_get(rec, "resume_optim_sha256"), NULL), !explicit_f, rc);
            if (ok) {
                o->train_resume = rp->from_path;
                o->lora_path = NULL;
                o->replay_resumed_at = rp->want_resumed_at = rs;
            }
        } else if (ok) {
            o->lora_path = rp->from_path;
            o->train_resume = NULL;
        }
    } else if (ok && (o->lora_path || o->train_resume)) {
        fprintf(stderr, "error: fresh training record does not take --lora or --train-resume\n");
        *rc = 1;
        ok = false;
    }

    if (ok && !strcmp(obj, "kl-to-teacher")) {
        if (!tch || !replay_resolve_teacher(o, rp, tch, rc)) {
            if (*rc != 2) replay_fail(rc, "teacher", "teacher object", "missing");
            ok = false;
        } else {
            o->train_teacher = rp->teacher_dir;
        }
    }

    if (ok) ok = replay_resolve_evals(o, rp, rec, kill_v, rc);

    if (ok) {
        const char *sched = jv_str(jv_get(rcp, "schedule"), "");
        const char *loss_mode = jv_str(jv_get(rcp, "loss"), "");
        if (!strcmp(obj, "dpo")) {
            o->dpo_path = rp->data_path;
            o->dpo_beta = (float)beta;
            o->train_path = NULL;
        } else {
            o->train_path = rp->data_path;
            o->dpo_path = NULL;
        }
        {
            o->mp.kv_f32 = strcmp(kv, "f32") == 0;
            o->train_kv_f32 = o->mp.kv_f32;
            o->train_full = rp->is_full;
            o->no_bos = nb->b;
            o->train_eot = eot->b;
            o->lora_rank = rank;
            o->lora_scale = (float)lscale;
            o->train_lr = (float)lr;
            o->train_steps = steps;
            o->train_ctx = ctx;
            o->seed = seed_u64;
            o->seed_given = true;
            o->weight_decay = (float)wd;
            o->lr_schedule = strcmp(sched, "cosine") == 0 ? 1 : 0;
            o->lr_warmup = warm;
            o->lr_floor = (float)lr_floor_d;
            o->train_batch = (!strcmp(loss_mode, "position-mean") || batch > 1) ? batch : 0;
            o->train_gpu = false;
            o->train_gpu_budget = 0;
            o->save_every = 0;
            o->sign_key = NULL;
            o->stop_file = NULL;
            o->train_lp_path = NULL;
            rp->want_ctx = ctx;
            rp->want_eot_id = eot_id;
            rp->want_alpha = (float)alpha;
            if (kill_v) {
                int kat = 0; double kmin = 0;
                const char *km = jv_str(jv_get(kill_v, "metric"), NULL);
                jv *evals = jv_get(rec, "evals");
                if (!km || !*km || strlen(km) >= sizeof rp->kill_metric ||
                    !record_int(kill_v, "at", 1, INT_MAX, &kat) ||
                    !record_num(kill_v, "min_rel", -1, 1, &kmin) || !evals)
                    ok = !replay_fail(rc, "kill", "valid kill object", "invalid");
                else {
                    snprintf(rp->kill_metric, sizeof rp->kill_metric, "%s", km);
                    o->kill_at = kat;
                    o->kill_metric = rp->kill_metric;
                    o->kill_min_rel = (float)kmin;
                    o->n_eval = evals->n;
                    for (int i = 0; i < evals->n; i++) o->eval_spec[i] = rp->eval_specs[i];
                }
            } else {
                o->n_eval = 0;
                o->kill_at = 0;
                o->kill_metric = NULL;
            }
            if (ok && retrain) {
                // the retrain trains the retained rows into the caller's output, signed as asked
                o->train_path = o->unlearn_retained;
                o->sign_key = caller_sign_key;
                o->replay_data = NULL;
            } else if (ok && snprintf(rp->tmp_out, sizeof rp->tmp_out, "%s/runner-replay-%ld.gguf",
                               plat_runtime_dir(), plat_pid_self()) < (int)sizeof rp->tmp_out) {
                plat_remove(rp->tmp_out);
                o->train_out = rp->tmp_out;
            } else if (ok) {
                fprintf(stderr, "error: replay output path too long\n");
                *rc = 1;
                ok = false;
            }
        }
    }
    if (ok && rp->segment) {
        // the segment's run: the record's recipe whole (its schedule spans
        // every step), resumed at A, a checkpoint every K steps as the run
        // wrote them, stopped after B
        o->save_every = rp->seg_every;
        o->replay_stop_at = rp->seg_b;
        if (rp->seg_a > 0) {
            o->train_resume = rp->seg_start;
            o->lora_path = NULL;
            o->replay_resumed_at = rp->want_resumed_at = rp->seg_a;
        }
    }
    jv_free(rec);
    return !ok;
}

bool train_record_jv_same(const jv *a, const jv *b) {
    return jv_same(a, b);
}

bool train_record_unlearn_split(const char *data1, const char *drop, sbuf *kept, int *count,
                                int *distinct, char *why, size_t cap) {
    return unlearn_split(data1, drop, kept, count, distinct, why, cap);
}

bool train_record_fail(int *rc, const char *field, const char *want, const char *got) {
    return replay_fail(rc, field, want, got);
}
