// Streaming pre-computed reference logits reader & exact float64 KL scorer
// (R4.48.8.3).
#include "ref_logits.h"

#include "compat.h"
#include "envelope.h"
#include "fpclass.h"
#include "json.h"
#include "npy.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

enum { REF_MANIFEST_MAX_BYTES = 64 * 1024 * 1024 };

static void set_why(char *why, size_t cap, const char *fmt, ...) {
    if (!why || !cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(why, cap, fmt, ap);
    va_end(ap);
}

static bool hex_lower_eq(const char *a, const char *b) {
    if (!a || !b) return false;
    for (int i = 0; i < 64; i++) {
        if (!a[i] || !b[i]) return false;
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i])) return false;
    }
    return a[64] == '\0' && b[64] == '\0';
}

static void copy_hex_lower(char dst[65], const char *src) {
    for (int i = 0; i < 64; i++) dst[i] = (char)tolower((unsigned char)src[i]);
    dst[64] = '\0';
}

// Validate a manifest-cited relative path: must stay strictly inside the
// manifest's directory (no leading slash/backslash, no drive colon, no "..",
// no empty segments, no control bytes).
static bool valid_rel_path(const char *rel) {
    if (!rel || !*rel) return false;
    if (rel[0] == '/' || rel[0] == '\\') return false;
    const char *seg = rel;
    for (const char *p = rel; ; p++) {
        unsigned char c = (unsigned char)*p;
        if (c != '\0' && (c < 0x20 || c == 0x7f || c == ':')) return false;
        if (c == '/' || c == '\\' || c == '\0') {
            size_t len = (size_t)(p - seg);
            if (len == 0) return false;
            if (len == 2 && seg[0] == '.' && seg[1] == '.') return false;
            if (len == 1 && seg[0] == '.') return false;
            if (c == '\0') break;
            seg = p + 1;
        }
    }
    return true;
}

static bool resolve_manifest_rel(const char *manifest_path, const char *rel,
                                 char *out, size_t out_cap) {
    if (!manifest_path || !rel || !out || !out_cap) return false;
    const char *slash = strrchr(manifest_path, '/');
    const char *bslash = strrchr(manifest_path, '\\');
    const char *sep = slash;
    if (bslash && (!sep || bslash > sep)) sep = bslash;
    int n;
    if (!sep) {
        n = snprintf(out, out_cap, "%s", rel);
    } else {
        size_t dlen = (size_t)(sep - manifest_path);
        n = snprintf(out, out_cap, "%.*s/%s", (int)dlen, manifest_path, rel);
    }
    return n > 0 && (size_t)n < out_cap;
}

static bool read_manifest_bytes(const char *path, char **buf_out, size_t *len_out,
                                char *why, size_t why_cap) {
    *buf_out = NULL;
    *len_out = 0;
    if (!record_regular_file(path)) {
        set_why(why, why_cap, "reference manifest is not a readable regular file: %s", path);
        return false;
    }
    FILE *f = plat_fopen(path, "rb");
    if (!f) {
        set_why(why, why_cap, "cannot open reference manifest: %s", path);
        return false;
    }
    int64_t sz = npy_file_size(f);
    if (sz <= 0 || sz > REF_MANIFEST_MAX_BYTES || !npy_seek(f, 0)) {
        fclose(f);
        set_why(why, why_cap, "reference manifest size (%lld bytes) is empty or exceeds %d bytes",
                (long long)sz, REF_MANIFEST_MAX_BYTES);
        return false;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        set_why(why, why_cap, "out of memory reading reference manifest");
        return false;
    }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) {
        free(buf);
        set_why(why, why_cap, "short read on reference manifest: %s", path);
        return false;
    }
    buf[sz] = '\0';
    *buf_out = buf;
    *len_out = (size_t)sz;
    return true;
}

static bool jv_exact_i64(const jv *v, int64_t *out) {
    if (!v || v->type != J_NUM || !fp_finite_f64(v->num)) return false;
    if (v->num < -9007199254740992.0 || v->num > 9007199254740992.0) return false;
    double tr = floor(v->num);
    if (tr != v->num) return false;
    *out = (int64_t)tr;
    return true;
}

static char *dup_opt_str(jv *obj, const char *key) {
    jv *v = jv_get(obj, key);
    if (!v || v->type != J_STR || !v->str || !*v->str) return NULL;
    return strdup(v->str);
}

void ref_logits_manifest_free(ref_logits_manifest *m) {
    if (!m) return;
    free(m->tokens);
    free(m->row_indices);
    free(m->ref_repo);
    free(m->ref_revision);
    free(m->ref_class);
    free(m->ref_dtype);
    free(m->ref_device);
    free(m->ref_producer);
    free(m->ref_producer_mode);
    free(m->ref_weights_sha256);
    memset(m, 0, sizeof *m);
}

bool ref_logits_load_manifest(const char *manifest_path, bool verify_sha256,
                              ref_logits_manifest *out, char *why, size_t why_cap) {
    if (!manifest_path || !*manifest_path || !out) {
        set_why(why, why_cap, "missing reference manifest path");
        return false;
    }
    memset(out, 0, sizeof *out);

    char *text = NULL;
    size_t text_len = 0;
    if (!read_manifest_bytes(manifest_path, &text, &text_len, why, why_cap)) return false;
    envelope_data_sha256(text, text_len, out->manifest_sha256);

    bool oom = false;
    jv *root = json_parse_ex(text, text_len, &oom);
    free(text);
    if (!root || root->type != J_OBJ) {
        jv_free(root);
        set_why(why, why_cap, "%s", oom ? "out of memory parsing reference manifest"
                                        : "reference manifest is not a valid JSON object");
        return false;
    }

    const char *schema = jv_str(jv_get(root, "schema_version"), NULL);
    if (!schema || strcmp(schema, REF_LOGITS_SCHEMA) != 0) {
        set_why(why, why_cap, "reference manifest schema_version must be %s", REF_LOGITS_SCHEMA);
        jv_free(root);
        return false;
    }

    jv *lg = jv_get(root, "logits");
    if (!lg || lg->type != J_OBJ) {
        set_why(why, why_cap, "reference manifest is missing logits object");
        jv_free(root);
        return false;
    }

    const char *lrel = jv_str(jv_get(lg, "path"), NULL);
    if (!valid_rel_path(lrel) || strlen(lrel) >= sizeof out->logits_rel) {
        set_why(why, why_cap, "reference manifest logits.path must be a relative path inside the manifest directory (no '..', absolute path, or drive letter)");
        jv_free(root);
        return false;
    }
    snprintf(out->logits_rel, sizeof out->logits_rel, "%s", lrel);
    if (!resolve_manifest_rel(manifest_path, lrel, out->logits_path, sizeof out->logits_path)) {
        set_why(why, why_cap, "resolved reference logits path is too long");
        jv_free(root);
        return false;
    }

    const char *fmt = jv_str(jv_get(lg, "format"), NULL);
    if (!fmt || (strcmp(fmt, "npy") != 0 && strcmp(fmt, "raw-f32") != 0)) {
        set_why(why, why_cap, "reference manifest logits.format must be \"npy\" or \"raw-f32\"");
        jv_free(root);
        return false;
    }
    snprintf(out->logits_format, sizeof out->logits_format, "%s", fmt);

    const char *dtype = jv_str(jv_get(lg, "dtype"), NULL);
    if (!dtype || strcmp(dtype, "float32") != 0) {
        set_why(why, why_cap, "reference manifest logits.dtype must be \"float32\"");
        jv_free(root);
        return false;
    }

    const char *lsha = jv_str(jv_get(lg, "sha256"), NULL);
    if (!lsha || !envelope_is_sha256_hex(lsha)) {
        set_why(why, why_cap, "reference manifest logits.sha256 must be a 64-character hex SHA-256 digest");
        jv_free(root);
        return false;
    }
    copy_hex_lower(out->logits_sha256, lsha);

    jv *shape = jv_get(lg, "shape");
    int64_t sh0 = 0, sh1 = 0, nv_top = 0, nspec = 0;
    if (!shape || shape->type != J_ARR || shape->n != 2 ||
        !jv_exact_i64(shape->items[0], &sh0) || !jv_exact_i64(shape->items[1], &sh1) ||
        sh0 < 1 || sh1 < 2 || sh1 > INT32_MAX) {
        set_why(why, why_cap, "reference manifest logits.shape must be [N, V] with N >= 1 and V >= 2");
        jv_free(root);
        return false;
    }
    if (!jv_exact_i64(jv_get(root, "n_vocab"), &nv_top) || nv_top != sh1) {
        set_why(why, why_cap, "reference manifest n_vocab does not match logits.shape[1]");
        jv_free(root);
        return false;
    }
    if (!jv_exact_i64(jv_get(root, "n_special"), &nspec) || nspec < 0 || nspec > INT32_MAX) {
        set_why(why, why_cap, "reference manifest n_special must be a non-negative integer");
        jv_free(root);
        return false;
    }
    // Check overflow of N * V * 4 in int64_t.
    if (sh0 > (INT64_MAX / 4) / sh1) {
        set_why(why, why_cap, "reference manifest logits byte size overflows 64-bit integer");
        jv_free(root);
        return false;
    }
    int64_t data_bytes = sh0 * sh1 * 4;
    out->n_rows = sh0;
    out->n_vocab = (int32_t)sh1;
    out->n_special = (int32_t)nspec;

    // Parse tokens[] array.
    jv *toks = jv_get(root, "tokens");
    if (!toks || toks->type != J_ARR || toks->n < 2) {
        set_why(why, why_cap, "reference manifest tokens must be an array of at least 2 token IDs");
        jv_free(root);
        return false;
    }
    out->tokens = malloc(sizeof(int32_t) * (size_t)toks->n);
    if (!out->tokens) {
        set_why(why, why_cap, "out of memory allocating reference manifest tokens");
        jv_free(root);
        ref_logits_manifest_free(out);
        return false;
    }
    out->n_tokens = toks->n;
    for (int i = 0; i < toks->n; i++) {
        int64_t tid = 0;
        if (!jv_exact_i64(toks->items[i], &tid) || tid < 0 || tid > INT32_MAX) {
            set_why(why, why_cap, "reference manifest tokens[%d] is not a valid token ID", i);
            jv_free(root);
            ref_logits_manifest_free(out);
            return false;
        }
        out->tokens[i] = (int32_t)tid;
    }

    // Parse optional sparse row_indices[].
    jv *ridx = jv_get(lg, "row_indices");
    if (ridx && ridx->type != J_NULL) {
        if (ridx->type != J_ARR || (int64_t)ridx->n != out->n_rows) {
            set_why(why, why_cap, "reference manifest logits.row_indices length must equal shape[0]");
            jv_free(root);
            ref_logits_manifest_free(out);
            return false;
        }
        out->row_indices = malloc(sizeof(int32_t) * (size_t)ridx->n);
        if (!out->row_indices) {
            set_why(why, why_cap, "out of memory allocating reference manifest row_indices");
            jv_free(root);
            ref_logits_manifest_free(out);
            return false;
        }
        int32_t prev = -1;
        for (int i = 0; i < ridx->n; i++) {
            int64_t pos = 0;
            if (!jv_exact_i64(ridx->items[i], &pos) || pos < 0 || pos >= (int64_t)(out->n_tokens - 1) ||
                (int32_t)pos <= prev) {
                set_why(why, why_cap, "reference manifest logits.row_indices must be strictly increasing within [0, len(tokens)-2]");
                jv_free(root);
                ref_logits_manifest_free(out);
                return false;
            }
            out->row_indices[i] = (int32_t)pos;
            prev = (int32_t)pos;
        }
    }

    // Optional corpus.sha256 pin.
    jv *corp = jv_get(root, "corpus");
    if (corp && corp->type == J_OBJ) {
        jv *csha = jv_get(corp, "sha256");
        if (csha && csha->type != J_NULL) {
            const char *cs = jv_str(csha, NULL);
            if (!cs || !envelope_is_sha256_hex(cs)) {
                set_why(why, why_cap, "reference manifest corpus.sha256 must be a 64-character hex SHA-256 digest");
                jv_free(root);
                ref_logits_manifest_free(out);
                return false;
            }
            copy_hex_lower(out->corpus_sha256, cs);
        }
    }

    // Optional reference provenance metadata.
    jv *ref = jv_get(root, "reference");
    if (ref && ref->type == J_OBJ) {
        out->ref_repo           = dup_opt_str(ref, "repo");
        out->ref_revision       = dup_opt_str(ref, "revision");
        out->ref_class          = dup_opt_str(ref, "class");
        out->ref_dtype          = dup_opt_str(ref, "dtype");
        out->ref_device         = dup_opt_str(ref, "device");
        out->ref_producer       = dup_opt_str(ref, "producer");
        out->ref_producer_mode  = dup_opt_str(ref, "producer_mode");
        out->ref_weights_sha256 = dup_opt_str(ref, "weights_sha256");
    }
    jv_free(root);

    // Check logits file on disk (must be regular, non-symlink).
    if (!record_regular_file_nofollow(out->logits_path)) {
        set_why(why, why_cap, "reference logits file is missing or not a regular file (symlinks refused): %s",
                out->logits_path);
        ref_logits_manifest_free(out);
        return false;
    }

    FILE *lf = plat_fopen(out->logits_path, "rb");
    if (!lf) {
        set_why(why, why_cap, "cannot open reference logits file: %s", out->logits_path);
        ref_logits_manifest_free(out);
        return false;
    }
    int64_t fsz = npy_file_size(lf);
    if (strcmp(out->logits_format, "npy") == 0) {
        npy_header nh;
        char nwhy[256] = "";
        if (!npy_seek(lf, 0) || !npy_read_header(lf, &nh, nwhy, sizeof nwhy)) {
            fclose(lf);
            set_why(why, why_cap, "invalid .npy reference logits header: %s", nwhy);
            ref_logits_manifest_free(out);
            return false;
        }
        if ((strcmp(nh.descr, "<f4") != 0 && strcmp(nh.descr, "=f4") != 0) ||
            nh.fortran_order || nh.ndim != 2 ||
            nh.shape[0] != out->n_rows || nh.shape[1] != (int64_t)out->n_vocab) {
            fclose(lf);
            set_why(why, why_cap, "reference .npy header (descr=%s ndim=%d shape=[%lld,%lld] fortran=%d) does not match manifest float32 [%lld,%d]",
                    nh.descr, nh.ndim, (long long)nh.shape[0], (long long)nh.shape[1],
                    nh.fortran_order ? 1 : 0, (long long)out->n_rows, (int)out->n_vocab);
            ref_logits_manifest_free(out);
            return false;
        }
        out->data_offset = nh.data_offset;
        if (fsz < 0 || nh.data_offset > INT64_MAX - data_bytes || fsz != nh.data_offset + data_bytes) {
            fclose(lf);
            set_why(why, why_cap, "reference .npy file size (%lld) does not match header + [%lld,%d] float32 (%lld)",
                    (long long)fsz, (long long)out->n_rows, (int)out->n_vocab,
                    (long long)(nh.data_offset + data_bytes));
            ref_logits_manifest_free(out);
            return false;
        }
    } else {
        out->data_offset = 0;
        if (fsz != data_bytes) {
            fclose(lf);
            set_why(why, why_cap, "reference raw-f32 file size (%lld) does not match [%lld,%d] float32 (%lld)",
                    (long long)fsz, (long long)out->n_rows, (int)out->n_vocab, (long long)data_bytes);
            ref_logits_manifest_free(out);
            return false;
        }
    }
    fclose(lf);

    if (verify_sha256) {
        char got_sha[65];
        if (!envelope_cited_regular_sha256(out->logits_path, got_sha)) {
            set_why(why, why_cap, "failed to hash reference logits file: %s", out->logits_path);
            ref_logits_manifest_free(out);
            return false;
        }
        if (!hex_lower_eq(got_sha, out->logits_sha256)) {
            set_why(why, why_cap, "reference logits sha256 mismatch: manifest expects %s, file has %s",
                    out->logits_sha256, got_sha);
            ref_logits_manifest_free(out);
            return false;
        }
    }
    return true;
}

static bool has_row_index(const ref_logits_manifest *m, int32_t pos) {
    if (pos < 0) return false;
    if (!m->row_indices) return (int64_t)pos < m->n_rows;
    int64_t lo = 0, hi = m->n_rows - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int32_t v = m->row_indices[mid];
        if (v == pos) return true;
        if (v < pos) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

bool ref_logits_covers_window(const ref_logits_manifest *m,
                              int start, int stride, int count,
                              char *why, size_t why_cap) {
    if (!m || start < 1 || stride < 1 || count < 1) {
        set_why(why, why_cap, "invalid scoring window for reference logits");
        return false;
    }
    for (int n = 0; n < count; n++) {
        int64_t e64 = (int64_t)m->n_special + (int64_t)start + (int64_t)n * (int64_t)stride - 1;
        if (e64 < 0 || e64 >= (int64_t)(m->n_tokens - 1)) {
            set_why(why, why_cap, "window position %lld (sample %d/%d) exceeds reference manifest tokens (%d)",
                    (long long)e64, n + 1, count, m->n_tokens);
            return false;
        }
        if (!has_row_index(m, (int32_t)e64)) {
            set_why(why, why_cap, "window position %lld (sample %d/%d) is not present in reference logits (%lld rows)",
                    (long long)e64, n + 1, count, (long long)m->n_rows);
            return false;
        }
    }
    return true;
}

bool ref_logits_open(const char *manifest_path, int model_n_vocab, int n_special,
                     const int32_t *toks, int n_prompt,
                     ref_logits_reader *r, char *why, size_t why_cap) {
    if (!r) return false;
    memset(r, 0, sizeof *r);

    if (!ref_logits_load_manifest(manifest_path, true, &r->m, why, why_cap)) return false;

    if (r->m.n_vocab != model_n_vocab) {
        set_why(why, why_cap, "reference logits vocab size (%d) does not match model n_vocab (%d)",
                (int)r->m.n_vocab, model_n_vocab);
        ref_logits_close(r);
        return false;
    }
    if (r->m.n_special != n_special) {
        set_why(why, why_cap, "reference manifest n_special (%d) does not match model tokenization (%d)",
                (int)r->m.n_special, n_special);
        ref_logits_close(r);
        return false;
    }
    if (n_prompt < 2 || n_prompt > r->m.n_tokens) {
        set_why(why, why_cap, "prompt token count (%d) exceeds reference manifest tokens (%d)",
                n_prompt, (int)r->m.n_tokens);
        ref_logits_close(r);
        return false;
    }
    for (int i = 0; i < n_prompt; i++) {
        if (toks[i] != r->m.tokens[i]) {
            set_why(why, why_cap, "token mismatch at index %d: model tokenized %d, reference manifest has %d",
                    i, (int)toks[i], (int)r->m.tokens[i]);
            ref_logits_close(r);
            return false;
        }
    }

    r->row_buf = malloc(sizeof(float) * (size_t)r->m.n_vocab);
    if (!r->row_buf) {
        set_why(why, why_cap, "out of memory allocating reference logit row buffer (%d floats)",
                (int)r->m.n_vocab);
        ref_logits_close(r);
        return false;
    }
    r->fp = plat_fopen(r->m.logits_path, "rb");
    if (!r->fp) {
        set_why(why, why_cap, "cannot open reference logits file for streaming: %s", r->m.logits_path);
        ref_logits_close(r);
        return false;
    }
    r->cursor = 0;
    return true;
}

void ref_logits_close(ref_logits_reader *r) {
    if (!r) return;
    if (r->fp) fclose(r->fp);
    free(r->row_buf);
    ref_logits_manifest_free(&r->m);
    memset(r, 0, sizeof *r);
}

bool ref_logits_want_pos(ref_logits_reader *r, int32_t emitted_pos, int64_t *file_row) {
    if (!r || emitted_pos < 0) return false;
    if (!r->m.row_indices) {
        if ((int64_t)emitted_pos >= r->m.n_rows) return false;
        if (file_row) *file_row = (int64_t)emitted_pos;
        return true;
    }
    while (r->cursor < r->m.n_rows && r->m.row_indices[r->cursor] < emitted_pos) {
        r->cursor++;
    }
    if (r->cursor < r->m.n_rows && r->m.row_indices[r->cursor] == emitted_pos) {
        if (file_row) *file_row = r->cursor;
        r->cursor++;
        return true;
    }
    return false;
}

bool ref_logits_score_row(ref_logits_reader *r, int32_t emitted_pos, int64_t file_row,
                          const float *cand_logits, ref_logits_row_result *out,
                          char *why, size_t why_cap) {
    if (!r || !r->fp || !r->row_buf || !cand_logits || !out) {
        set_why(why, why_cap, "internal error: invalid ref_logits_score_row state");
        return false;
    }
    if (file_row < 0 || file_row >= r->m.n_rows) {
        set_why(why, why_cap, "reference logits row %lld out of bounds (%lld)",
                (long long)file_row, (long long)r->m.n_rows);
        return false;
    }
    int32_t V = r->m.n_vocab;
    int64_t off = r->m.data_offset + file_row * (int64_t)V * 4;
    if (!npy_seek(r->fp, off)) {
        set_why(why, why_cap, "seek failed on reference logits file at row %lld", (long long)file_row);
        return false;
    }
    if (fread(r->row_buf, sizeof(float), (size_t)V, r->fp) != (size_t)V) {
        set_why(why, why_cap, "short read on reference logits file at row %lld", (long long)file_row);
        return false;
    }

    const float *xr = r->row_buf;
    // Pass 1: fail-closed finiteness check on both rows + max, top-1, top-2, max_abs.
    if (!fp_finite_f32(xr[0]) || !fp_finite_f32(cand_logits[0])) {
        set_why(why, why_cap, "non-finite %s logit at position %d, vocab 0",
                !fp_finite_f32(xr[0]) ? "reference" : "candidate", (int)emitted_pos);
        return false;
    }
    double r_best1 = (double)xr[0];
    double r_best2 = 0.0;
    bool have_r2 = false;
    int32_t r_best1_i = 0;

    double c_best1 = (double)cand_logits[0];
    int32_t c_best1_i = 0;
    double max_abs = fabs(r_best1 - c_best1);

    for (int32_t i = 1; i < V; i++) {
        float fr_f = xr[i];
        float fo_f = cand_logits[i];
        if (!fp_finite_f32(fr_f) || !fp_finite_f32(fo_f)) {
            set_why(why, why_cap, "non-finite %s logit at position %d, vocab %d",
                    !fp_finite_f32(fr_f) ? "reference" : "candidate", (int)emitted_pos, (int)i);
            return false;
        }
        double fr = (double)fr_f;
        double fo = (double)fo_f;
        if (fr > r_best1) {
            r_best2 = r_best1;
            have_r2 = true;
            r_best1 = fr;
            r_best1_i = i;
        } else if (!have_r2 || fr > r_best2) {
            r_best2 = fr;
            have_r2 = true;
        }
        if (fo > c_best1) {
            c_best1 = fo;
            c_best1_i = i;
        }
        double d = fabs(fr - fo);
        if (d > max_abs) max_abs = d;
    }

    // Pass 2: exact float64 log-sum-exp (compare_full.py formula).
    double sr = 0.0, so = 0.0;
    for (int32_t i = 0; i < V; i++) {
        sr += exp((double)xr[i] - r_best1);
        so += exp((double)cand_logits[i] - c_best1);
    }
    double lse_r = r_best1 + log(sr);
    double lse_o = c_best1 + log(so);

    // Pass 3: exact full-vocabulary float64 KL(ref || cand).
    double kld = 0.0;
    for (int32_t i = 0; i < V; i++) {
        double lr = (double)xr[i] - lse_r;
        double lo = (double)cand_logits[i] - lse_o;
        kld += exp(lr) * (lr - lo);
    }
    if (!fp_finite_f64(kld) || !fp_finite_f64(lse_r) || !fp_finite_f64(lse_o)) {
        set_why(why, why_cap, "non-finite float64 KL computed at position %d", (int)emitted_pos);
        return false;
    }

    out->pos         = emitted_pos;
    out->kld         = kld;
    out->gold_margin = r_best1 - r_best2;
    out->gold_top1   = r_best1_i;
    out->cand_top1   = c_best1_i;
    out->agree       = (r_best1_i == c_best1_i);
    out->max_abs     = max_abs;
    return true;
}
