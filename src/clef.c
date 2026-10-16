// Clef joint schema head and the Jev/SystemOne contract, ported from the
// publisher's joint_schema_model.py (Cloudflare/clef at 0b331204, the same
// file in Cloudflare/clef-flash). Line references below are to that file.
#include "clef.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "envelope.h"
#include "fpclass.h"
#include "runner.h"

// ------------------------------------------------------- python spellings
//
// encode_record renders every non-string value with json.dumps(value,
// ensure_ascii=False, separators=(",", ":"), sort_keys=True) (render, l.35)
// and tokenizes the result, so those bytes are model input. Three things
// differ from this program's own JSON writer and each changes tokens: a
// float is written with float.__repr__ (`1250.0`, `1e-05`, `1e+16`), an
// integer literal stays an integer, and the string escapes are Python's.

// D (p significant digits) times 10^E written as text strtod reads back
static void sci_text(uint64_t D, int p, int E, char *buf, size_t cap) {
    char dig[24];
    snprintf(dig, sizeof dig, "%0*llu", p, (unsigned long long)D);
    if (p == 1) snprintf(buf, cap, "%se%d", dig, E);
    else        snprintf(buf, cap, "%c.%se%d", dig[0], dig + 1, E);
}

static uint64_t pow10u(int n) {
    uint64_t r = 1;
    while (n-- > 0) r *= 10;
    return r;
}

void clef_py_float_repr(double d, char out[40]) {
    uint64_t bits = fp_bits_f64(d);
    bool neg = bits >> 63;
    if (fp_isnan(d)) { strcpy(out, "NaN"); return; }
    if (!fp_isfinite(d)) { strcpy(out, neg ? "-Infinity" : "Infinity"); return; }
    if ((bits << 1) == 0) { strcpy(out, neg ? "-0.0" : "0.0"); return; }
    double a = neg ? -d : d;
    // The shortest digit string that reads back to `a`, as dtoa's mode 0
    // (float.__repr__) finds it. At each length the correctly rounded
    // spelling is the nearest one; when it misses, only its neighbour on the
    // other side can still land inside the (asymmetric, at a power of two)
    // rounding interval, so that is the one other candidate tried.
    uint64_t D = 0;
    int E = 0, p = 1;
    char buf[48];
    for (p = 1; p <= 17; p++) {
        snprintf(buf, sizeof buf, "%.*e", p - 1, a);
        const char *ep = strchr(buf, 'e');
        if (!ep) continue;
        D = 0;
        for (const char *c = buf; c < ep; c++)
            if (*c >= '0' && *c <= '9') D = D * 10 + (uint64_t)(*c - '0');
        E = atoi(ep + 1);
        if (strtod(buf, NULL) == a) break;
        uint64_t lo = pow10u(p - 1), hi = pow10u(p);
        char cand[48];
        if (D + 1 < hi) {
            sci_text(D + 1, p, E, cand, sizeof cand);
            if (strtod(cand, NULL) == a) { D = D + 1; break; }
        }
        uint64_t Dd = D == lo ? hi - 1 : D - 1;
        int Ed = D == lo ? E - 1 : E;
        sci_text(Dd, p, Ed, cand, sizeof cand);
        if (strtod(cand, NULL) == a) { D = Dd; E = Ed; break; }
    }
    if (p > 17) p = 17;   // unreachable: seventeen digits always read back
    char dig[24];
    snprintf(dig, sizeof dig, "%0*llu", p, (unsigned long long)D);
    int nd = p;
    while (nd > 1 && dig[nd - 1] == '0') nd--;
    dig[nd] = 0;
    // repr switches to exponent form when decpt <= -4 or decpt > 16
    // (pystrtod.c format_float_short, 'r'), decpt being where the point
    // falls in 0.<digits>
    int decpt = E + 1;
    char *o = out;
    if (neg) *o++ = '-';
    if (decpt <= -4 || decpt > 16) {
        *o++ = dig[0];
        if (nd > 1) { *o++ = '.'; memcpy(o, dig + 1, (size_t)nd - 1); o += nd - 1; }
        // e, the sign, and at least two digits (a double's exponent has three
        // at most)
        int x = decpt - 1, ax = x < 0 ? -x : x;
        *o++ = 'e';
        *o++ = x < 0 ? '-' : '+';
        if (ax >= 100) *o++ = (char)('0' + ax / 100);
        *o++ = (char)('0' + ax / 10 % 10);
        *o++ = (char)('0' + ax % 10);
        *o = 0;
        return;
    }
    if (decpt <= 0) {
        *o++ = '0'; *o++ = '.';
        for (int i = 0; i < -decpt; i++) *o++ = '0';
        memcpy(o, dig, (size_t)nd); o += nd;
    } else if (decpt < nd) {
        memcpy(o, dig, (size_t)decpt); o += decpt;
        *o++ = '.';
        memcpy(o, dig + decpt, (size_t)(nd - decpt)); o += nd - decpt;
    } else {
        memcpy(o, dig, (size_t)nd); o += nd;
        for (int i = nd; i < decpt; i++) *o++ = '0';
        *o++ = '.'; *o++ = '0';
    }
    *o = 0;
}

// json.dumps(str, ensure_ascii=False): the five short escapes, \uXXXX in
// lowercase hex for the other control characters, everything else as is
static void py_dumps_str(sbuf *o, const char *s) {
    sb_lit(o, "\"");
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
        case '"':  sb_lit(o, "\\\""); break;
        case '\\': sb_lit(o, "\\\\"); break;
        case '\n': sb_lit(o, "\\n"); break;
        case '\r': sb_lit(o, "\\r"); break;
        case '\t': sb_lit(o, "\\t"); break;
        case '\b': sb_lit(o, "\\b"); break;
        case '\f': sb_lit(o, "\\f"); break;
        default:
            if (c < 0x20) sb_fmt(o, "\\u%04x", c);
            else sb_put(o, p, 1);
        }
    }
    sb_lit(o, "\"");
}

static int key_cmp(const void *a, const void *b) {
    // python sorts str by code point, which is UTF-8 byte order
    const char *x = *(const char *const *)a, *y = *(const char *const *)b;
    return strcmp(x, y);
}

typedef struct { const char *key; int idx; } keyed;

static int keyed_cmp(const void *a, const void *b) {
    return key_cmp(&((const keyed *)a)->key, &((const keyed *)b)->key);
}

void clef_py_dumps(const jv *v, sbuf *o) {
    if (!v) { sb_lit(o, "null"); return; }
    switch (v->type) {
    case J_NULL: sb_lit(o, "null"); break;
    case J_BOOL: sb_lit(o, v->b ? "true" : "false"); break;
    case J_NUM:
        if (v->num_int) {
            // an int literal: its digits (python ints are exact); -0 is 0
            if (v->num_text) sb_lit(o, v->num_text);
            else sb_fmt(o, "%lld", (long long)v->num);
        } else {
            char f[40];
            clef_py_float_repr(v->num, f);
            sb_lit(o, f);
        }
        break;
    case J_STR: py_dumps_str(o, v->str); break;
    case J_ARR:
        sb_lit(o, "[");
        for (int i = 0; i < v->n; i++) {
            if (i) sb_lit(o, ",");
            clef_py_dumps(v->items[i], o);
        }
        sb_lit(o, "]");
        break;
    case J_OBJ: {
        keyed *ks = v->n ? malloc(sizeof *ks * (size_t)v->n) : NULL;
        if (v->n && !ks) { o->failed = true; return; }
        for (int i = 0; i < v->n; i++) { ks[i].key = v->keys[i]; ks[i].idx = i; }
        if (v->n) qsort(ks, (size_t)v->n, sizeof *ks, keyed_cmp);
        sb_lit(o, "{");
        for (int i = 0; i < v->n; i++) {
            if (i) sb_lit(o, ",");
            py_dumps_str(o, ks[i].key);
            sb_lit(o, ":");
            clef_py_dumps(v->items[ks[i].idx], o);
        }
        sb_lit(o, "}");
        free(ks);
        break;
    }
    }
}

// render(value), l.35: a string as is, anything else as compact JSON
static void py_render(const jv *v, sbuf *o) {
    if (v && v->type == J_STR) sb_lit(o, v->str);
    else clef_py_dumps(v, o);
}

// Python's truthiness, for `question.get("criteria")` and `or []`
static bool py_truthy(const jv *v) {
    if (!v) return false;
    switch (v->type) {
    case J_NULL: return false;
    case J_BOOL: return v->b;
    case J_NUM:  return v->num != 0;
    case J_STR:  return v->str[0] != 0;
    case J_ARR:
    case J_OBJ:  return v->n > 0;
    }
    return false;
}

// ------------------------------------------------------------- the record

typedef struct { int32_t *v; int n, cap; bool failed; } ivec;

static void iv_push(ivec *a, const int32_t *src, int n) {
    if (a->failed || n <= 0) return;
    if (a->n + n > a->cap) {
        int cap = a->cap ? a->cap : 256;
        while (cap < a->n + n) cap *= 2;
        int32_t *g = realloc(a->v, sizeof(int32_t) * (size_t)cap);
        if (!g) { a->failed = true; return; }
        a->v = g;
        a->cap = cap;
    }
    memcpy(a->v + a->n, src, sizeof(int32_t) * (size_t)n);
    a->n += n;
}

// _tokens(tokenizer, text), l.77: tokenizer(text, add_special_tokens=False),
// one piece at a time. The HF tokenizer matches its added tokens anywhere in
// the text, so every piece is encoded with special tokens recognized
// (TOK_RAW), the caller's state and descriptions included: that is the
// reference's encoding, and token-for-token agreement with it is the bar.
static bool put_text(tokenizer *tok, ivec *a, const char *text) {
    int32_t *t = NULL;
    int n = tok_encode_fit(tok, text, false, TOK_RAW, 0, &t);
    if (n < 0) { a->failed = true; return false; }
    iv_push(a, t, n);
    free(t);
    return !a->failed;
}

static bool put_sb(tokenizer *tok, ivec *a, sbuf *s) {
    if (s->failed) { a->failed = true; return false; }
    bool ok = put_text(tok, a, s->s ? s->s : "");
    s->n = 0;
    if (s->s) s->s[0] = 0;
    return ok;
}

void clef_record_free(clef_record *r) {
    if (!r) return;
    for (int i = 0; r->q && i < r->n_q; i++) {
        clef_question *q = &r->q[i];
        free(q->id);
        free(q->o0); free(q->o1);
        for (int k = 0; q->opt_id && k < q->n_opt; k++) free(q->opt_id[k]);
        free(q->opt_id);
    }
    free(r->q);
    free(r->ids);
    memset(r, 0, sizeof *r);
}

static const char *const NOUL_TRUE  = "The proposition is true or the answer is yes.";
static const char *const NOUL_FALSE = "The proposition is false or the answer is no.";

static int qtype(const char *t) {
    if (!strcmp(t, "noul"))   return 0;
    if (!strcmp(t, "choice")) return 1;
    if (!strcmp(t, "score"))  return 2;
    return -1;
}

// static error text for the formatted ones below (one request at a time per
// slot, but slots run concurrently: thread-local)
static _Thread_local char g_err[256];

static const char *errf(const char *qid, const char *what) {
    snprintf(g_err, sizeof g_err, "%.120s: %s", qid, what);
    return g_err;
}

// The option list of one question (question_options, l.46), as
// (id, description) with description NULL for python's None.
typedef struct { const char *id; const jv *desc; char buf[24]; } clef_opt;

static int question_options(const jv *q, int type, clef_opt **out, const char **err,
                            const char *qid) {
    jv *crit = jv_get((jv *)q, "criteria");
    *out = NULL;
    if (type == 0) {
        // the two defaults, overridden by the caller's criteria
        if (crit && crit->type != J_NULL && crit->type != J_OBJ) {
            *err = errf(qid, "noul criteria must be an object with true and/or false");
            return -1;
        }
        clef_opt *o = calloc(2, sizeof *o);
        if (!o) { *err = "out of memory"; return -1; }
        o[0].id = "true";
        o[1].id = "false";
        jv *t = crit && crit->type == J_OBJ ? jv_get(crit, "true") : NULL;
        jv *f = crit && crit->type == J_OBJ ? jv_get(crit, "false") : NULL;
        o[0].desc = t;   // NULL = the default text below
        o[1].desc = f;
        *out = o;
        return 2;
    }
    if (type == 1) {
        if (!crit || crit->type != J_OBJ) {
            *err = errf(qid, "choice criteria must be an object of option ids to descriptions");
            return -1;
        }
        clef_opt *o = calloc((size_t)crit->n, sizeof *o);
        if (!o) { *err = "out of memory"; return -1; }
        keyed *ks = malloc(sizeof *ks * (size_t)crit->n);
        if (!ks) { free(o); *err = "out of memory"; return -1; }
        for (int i = 0; i < crit->n; i++) { ks[i].key = crit->keys[i]; ks[i].idx = i; }
        qsort(ks, (size_t)crit->n, sizeof *ks, keyed_cmp);
        for (int i = 0; i < crit->n; i++) {
            o[i].id = ks[i].key;
            o[i].desc = crit->items[ks[i].idx];
        }
        free(ks);
        *out = o;
        return crit->n;
    }
    if (!crit || crit->type != J_ARR) {
        *err = errf(qid, "score criteria must be an array of level descriptions");
        return -1;
    }
    clef_opt *o = calloc((size_t)crit->n, sizeof *o);
    if (!o) { *err = "out of memory"; return -1; }
    for (int i = 0; i < crit->n; i++) {
        snprintf(o[i].buf, sizeof o[i].buf, "%d", i);
        o[i].id = o[i].buf;
        o[i].desc = crit->items[i];
    }
    *out = o;
    return crit->n;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

bool clef_encode(tokenizer *tok, const jv *req_c, int max_length,
                 clef_record *out, const char **err) {
    jv *req = (jv *)req_c;
    memset(out, 0, sizeof *out);
    // systemone(), l.552: the request checks, in its order
    jv *model = jv_get(req, "model");
    jv *state = jv_get(req, "state");
    if (!model || model->type != J_STR || !state) {
        *err = "model and state are required";
        return false;
    }
    jv *qs = jv_get(req, "questions");
    if (!qs || qs->type != J_OBJ || qs->n == 0) {
        *err = "at least one question is required";
        return false;
    }
    for (int i = 0; i < qs->n; i++) {
        jv *q = qs->items[i];
        if (!q || q->type != J_OBJ) { *err = errf(qs->keys[i], "a question must be an object"); return false; }
        jv *t = jv_get(q, "type");
        if (!t || t->type != J_STR || qtype(t->str) < 0) {
            *err = errf(qs->keys[i], "type must be noul, choice, or score");
            return false;
        }
        if (qtype(t->str) != 0 && !py_truthy(jv_get(q, "criteria"))) {
            *err = errf(qs->keys[i], "criteria must not be empty");
            return false;
        }
    }
    if (py_truthy(jv_get(req, "images")) || py_truthy(jv_get(req, "videos"))) {
        *err = "images and videos are not supported yet: this runner serves Clef's text input only";
        return false;
    }

    // encode_record, l.103
    ivec schema = {0}, ids = {0};
    sbuf s = {0};
    out->q = calloc((size_t)qs->n, sizeof *out->q);
    if (!out->q) { *err = "out of memory"; return false; }
    out->n_q = qs->n;
    bool ok = put_text(tok, &schema, "\n\nSCHEMA FIELDS:\n");
    for (int qi = 0; ok && qi < qs->n; qi++) {
        const char *qid = qs->keys[qi];
        jv *q = qs->items[qi];
        const char *type = jv_get(q, "type")->str;
        clef_question *cq = &out->q[qi];
        cq->id = xstrdup(qid);
        cq->type = qtype(type);
        if (!cq->id) { ok = false; break; }
        sb_fmt(&s, "\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: ", qi + 1, qid, type);
        ok = put_sb(tok, &schema, &s);
        cq->q0 = schema.n;
        jv *ins = jv_get(q, "instructions");
        if (!ins || ins->type == J_NULL || (ins->type == J_STR && !ins->str[0]))
            sb_lit(&s, qid);   // the question id stands in (l.121)
        else
            py_render(ins, &s);
        ok = ok && put_sb(tok, &schema, &s);
        cq->q1 = schema.n;
        if (ok && cq->q1 == cq->q0) {
            *err = errf(qid, "the instructions encode to no tokens");
            clef_record_free(out); free(schema.v); free(s.s);
            return false;
        }
        ok = ok && put_text(tok, &schema, "\nALLOWED OPTIONS:\n");
        clef_opt *opts = NULL;
        int n_opt = ok ? question_options(q, cq->type, &opts, err, qid) : -1;
        if (n_opt < 0) {
            if (ok) { clef_record_free(out); free(schema.v); free(s.s); return false; }
            break;
        }
        cq->n_opt = n_opt;
        cq->o0 = calloc((size_t)n_opt, sizeof(int));
        cq->o1 = calloc((size_t)n_opt, sizeof(int));
        cq->opt_id = calloc((size_t)n_opt, sizeof(char *));
        if (!cq->o0 || !cq->o1 || !cq->opt_id) { free(opts); ok = false; break; }
        for (int k = 0; ok && k < n_opt; k++) {
            sb_fmt(&s, "OPTION %d: ", k + 1);
            ok = put_sb(tok, &schema, &s);
            cq->o0[k] = schema.n;
            // semantics = {"option_id": id[, "description": d]}, rendered
            // with sorted keys (l.132)
            const jv *d = opts[k].desc;
            bool has_desc;
            if (cq->type == 0 && !d) has_desc = true;   // a default text
            else has_desc = d && d->type != J_NULL;
            sb_lit(&s, "{");
            if (has_desc) {
                sb_lit(&s, "\"description\":");
                if (cq->type == 0 && !d) py_dumps_str(&s, k == 0 ? NOUL_TRUE : NOUL_FALSE);
                else clef_py_dumps(d, &s);
                sb_lit(&s, ",");
            }
            sb_lit(&s, "\"option_id\":");
            py_dumps_str(&s, opts[k].id);
            sb_lit(&s, "}");
            ok = ok && put_sb(tok, &schema, &s);
            cq->o1[k] = schema.n;
            cq->opt_id[k] = xstrdup(opts[k].id);
            if (!cq->opt_id[k]) ok = false;
            ok = ok && put_text(tok, &schema, "\n");
        }
        free(opts);
        ok = ok && put_text(tok, &schema, "END FIELD\n");
    }

    ivec prefix = {0}, suffix = {0}, st = {0};
    sb_fmt(&s, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\nSTATE:\n",
           "Read the complete state and schema. Decide every field jointly. Each answer "
           "must be exactly one of that field's allowed options.");
    ok = ok && put_sb(tok, &prefix, &s);
    ok = ok && put_text(tok, &suffix,
                        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    py_render(state, &s);
    ok = ok && put_sb(tok, &st, &s);
    free(s.s);
    if (!ok || schema.failed || prefix.failed || suffix.failed || st.failed) {
        free(schema.v); free(prefix.v); free(suffix.v); free(st.v);
        clef_record_free(out);
        *err = "out of memory encoding the record";
        return false;
    }
    int fixed = prefix.n + schema.n + suffix.n;
    if (fixed > max_length) {
        snprintf(g_err, sizeof g_err,
                 "schema requires %d tokens before state; maximum is %d", fixed, max_length);
        *err = g_err;
        free(schema.v); free(prefix.v); free(suffix.v); free(st.v);
        clef_record_free(out);
        return false;
    }
    int n_state = st.n;
    if (n_state > max_length - fixed) {
        n_state = max_length - fixed;
        out->state_truncated = true;
    }
    int off = prefix.n + n_state;
    iv_push(&ids, prefix.v, prefix.n);
    iv_push(&ids, st.v, n_state);
    iv_push(&ids, schema.v, schema.n);
    iv_push(&ids, suffix.v, suffix.n);
    free(schema.v); free(prefix.v); free(suffix.v); free(st.v);
    if (ids.failed || ids.n == 0) {
        free(ids.v);
        clef_record_free(out);
        *err = ids.failed ? "out of memory encoding the record" : "record produced no model input";
        return false;
    }
    for (int qi = 0; qi < out->n_q; qi++) {
        clef_question *cq = &out->q[qi];
        cq->q0 += off; cq->q1 += off;
        for (int k = 0; k < cq->n_opt; k++) { cq->o0[k] += off; cq->o1[k] += off; }
    }
    out->ids = ids.v;
    out->n_ids = ids.n;
    return true;
}

// --------------------------------------------------------------- the head
//
// JointSchemaHead.forward (l.344) for one record, in f32. Shapes: L tokens,
// E backbone width, W head width, Q questions, O options in all.

static void layer_norm(float *y, const float *x, const clef_norm *nm, int dim, float eps) {
    double mean = 0, var = 0;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    for (int i = 0; i < dim; i++) { double d = x[i] - mean; var += d * d; }
    var /= dim;   // biased, as torch.nn.LayerNorm
    float inv = (float)(1.0 / sqrt(var + eps));
    for (int i = 0; i < dim; i++)
        y[i] = ((float)(x[i] - mean) * inv) * nm->w[i] + nm->b[i];
}

static void layer_norm_rows(float *y, const float *x, int rows, int dim,
                            const clef_norm *nm, float eps) {
    for (int r = 0; r < rows; r++)
        layer_norm(y + (size_t)r * dim, x + (size_t)r * dim, nm, dim, eps);
}

// torch.nn.GELU() / activation="gelu": the exact erf form
static void gelu_rows(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752440f));
}

static void linear(model_t *m, float *y, const gguf_tensor *w, const float *x,
                   int rows, int n_in, int n_out, const float *bias) {
    model_matmul(m, y, n_out, w, x, n_in, n_in, n_out, bias, rows);
}

static float dotf(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void softmax_inplace(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float z = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); z += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= z;
}

// torch.nn.MultiheadAttention(q_in, kv_in, kv_in), no masks: out[nq][W]
static bool mha(model_t *m, const clef_attn *a, const float *q_in, int nq,
                const float *kv_in, int nk, int W, int H, float *out) {
    int dh = W / H;
    float *Q = malloc(sizeof(float) * (size_t)nq * W);
    float *K = malloc(sizeof(float) * (size_t)nk * W);
    float *V = malloc(sizeof(float) * (size_t)nk * W);
    float *C = malloc(sizeof(float) * (size_t)nq * W);
    float *p = malloc(sizeof(float) * (size_t)nk);
    bool ok = Q && K && V && C && p;
    if (ok) {
        linear(m, Q, a->wq, q_in, nq, W, W, a->bq);
        linear(m, K, a->wk, kv_in, nk, W, W, a->bk);
        linear(m, V, a->wv, kv_in, nk, W, W, a->bv);
        float scale = 1.0f / sqrtf((float)dh);
        for (int i = 0; i < nq; i++) {
            for (int h = 0; h < H; h++) {
                const float *qh = Q + (size_t)i * W + (size_t)h * dh;
                for (int j = 0; j < nk; j++)
                    p[j] = dotf(qh, K + (size_t)j * W + (size_t)h * dh, dh) * scale;
                softmax_inplace(p, nk);
                float *ch = C + (size_t)i * W + (size_t)h * dh;
                for (int d = 0; d < dh; d++) ch[d] = 0;
                for (int j = 0; j < nk; j++) {
                    const float *vh = V + (size_t)j * W + (size_t)h * dh;
                    for (int d = 0; d < dh; d++) ch[d] += p[j] * vh[d];
                }
            }
        }
        linear(m, out, a->wo, C, nq, W, W, a->bo);
    }
    free(Q); free(K); free(V); free(C); free(p);
    return ok;
}

// x += FF(norm(x)): Linear(W, F), GELU, Linear(F, W)
static bool ffn_residual(model_t *m, const struct clef_head *h, const clef_block *b,
                         float *x, int rows) {
    int W = h->width, F = h->n_ff;
    float *n = malloc(sizeof(float) * (size_t)rows * W);
    float *u = malloc(sizeof(float) * (size_t)rows * F);
    float *d = malloc(sizeof(float) * (size_t)rows * W);
    bool ok = n && u && d;
    if (ok) {
        layer_norm_rows(n, x, rows, W, &b->ffn_norm, h->eps);
        linear(m, u, b->ffn_up, n, rows, W, F, b->ffn_up_b);
        gelu_rows(u, (size_t)rows * F);
        linear(m, d, b->ffn_down, u, rows, F, W, b->ffn_down_b);
        for (size_t i = 0; i < (size_t)rows * W; i++) x[i] += d[i];
    }
    free(n); free(u); free(d);
    return ok;
}

// mean of rows [a, b) of x (width dim) into y
static void mean_span(float *y, const float *x, int a, int b, int dim) {
    for (int i = 0; i < dim; i++) y[i] = 0;
    for (int r = a; r < b; r++) {
        const float *row = x + (size_t)r * dim;
        for (int i = 0; i < dim; i++) y[i] += row[i];
    }
    float inv = 1.0f / (float)(b - a);
    for (int i = 0; i < dim; i++) y[i] *= inv;
}

// F.normalize(v, dim=-1): v / max(||v||, 1e-12)
static void normalize(float *y, const float *x, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    double nrm = sqrt(ss);
    float inv = (float)(1.0 / (nrm > 1e-12 ? nrm : 1e-12));
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

bool clef_head_forward(model_t *m, const clef_record *r, const float *hidden,
                       float *logits) {
    const struct clef_head *h = m->clef;
    if (!h || r->n_ids <= 0) return false;
    const int L = r->n_ids, E = h->hidden, W = h->width, H = h->n_head;
    const int Q = r->n_q;
    int O = 0, max_opt = 0;
    for (int i = 0; i < Q; i++) {
        O += r->q[i].n_opt;
        if (r->q[i].n_opt > max_opt) max_opt = r->q[i].n_opt;
    }
    if (O == 0) return false;
    int *owner = malloc(sizeof(int) * (size_t)O);
    float *rw   = malloc(sizeof(float) * (size_t)max_opt);  // routing weights
    float *nh   = malloc(sizeof(float) * (size_t)L * E);   // hidden_norm(hidden)
    float *mem  = malloc(sizeof(float) * (size_t)L * W);   // memory_projection
    float *memn = malloc(sizeof(float) * (size_t)L * W);   // memory_norm(mem)
    float *qv   = malloc(sizeof(float) * (size_t)Q * E);   // question vectors
    float *oc   = malloc(sizeof(float) * (size_t)O * E);   // option contexts
    float *lx   = malloc(sizeof(float) * (size_t)O * E);   // lexical options
    float *ta   = malloc(sizeof(float) * (size_t)O * W);
    float *tb   = malloc(sizeof(float) * (size_t)O * W);
    float *tq   = malloc(sizeof(float) * (size_t)Q * W);
    float *opt  = malloc(sizeof(float) * (size_t)O * W);   // routed options
    float *qn   = malloc(sizeof(float) * (size_t)O * W);
    float *att  = malloc(sizeof(float) * (size_t)O * W);
    float *fld  = malloc(sizeof(float) * (size_t)Q * W);   // fields
    float *fn   = malloc(sizeof(float) * (size_t)Q * W);
    float *fa   = malloc(sizeof(float) * (size_t)Q * W);
    float *glob = malloc(sizeof(float) * (size_t)W);
    float *row  = malloc(sizeof(float) * (size_t)(E > 4 * W ? E : 4 * W));
    float *vec  = malloc(sizeof(float) * (size_t)(E > W ? E : W));
    float *anc  = malloc(sizeof(float) * (size_t)E);
    float *hid1 = malloc(sizeof(float) * (size_t)W);
    bool ok = owner && rw && nh && mem && memn && qv && oc && lx && ta && tb && tq &&
              opt && qn && att && fld && fn && fa && glob && row && vec && anc && hid1;
    if (!ok) goto done;

    // normalized_hidden, memory, global_vector (l.353-358)
    layer_norm_rows(nh, hidden, L, E, &h->hidden_norm, h->eps);
    linear(m, mem, h->proj_memory, nh, L, E, W, NULL);
    const float *gv = nh + (size_t)(L - 1) * E;

    // question vectors, option contexts, lexical options (l.359-386)
    {
        int o = 0;
        const gguf_tensor *out_w = m->output;
        size_t rs = ggml_row_size(out_w->type, E);
        for (int i = 0; i < Q; i++) {
            const clef_question *cq = &r->q[i];
            mean_span(qv + (size_t)i * E, nh, cq->q0, cq->q1, E);
            for (int k = 0; k < cq->n_opt; k++, o++) {
                owner[o] = i;
                mean_span(oc + (size_t)o * E, nh, cq->o0[k], cq->o1[k], E);
                // output_embedding_weight[token_ids].mean(0): the LM head's rows
                float *lo = lx + (size_t)o * E;
                for (int c = 0; c < E; c++) lo[c] = 0;
                for (int t = cq->o0[k]; t < cq->o1[k]; t++) {
                    int32_t id = r->ids[t];
                    if (id < 0 || id >= m->n_vocab) id = 0;
                    dequant_row(out_w->type, (const uint8_t *)out_w->data + (size_t)id * rs,
                                row, E);
                    if (out_w->scale != 1.0f)
                        for (int c = 0; c < E; c++) row[c] *= out_w->scale;
                    for (int c = 0; c < E; c++) lo[c] += row[c];
                }
                float inv = 1.0f / (float)(cq->o1[k] - cq->o0[k]);
                for (int c = 0; c < E; c++) lo[c] *= inv;
            }
        }
    }

    // option queries (l.388-399)
    linear(m, ta, h->proj_option_context, oc, O, E, W, NULL);
    linear(m, tb, h->proj_option_lexical, lx, O, E, W, NULL);
    linear(m, tq, h->proj_option_question, qv, Q, E, W, NULL);
    for (int o = 0; o < O; o++)
        for (int c = 0; c < W; c++)
            opt[(size_t)o * W + c] = (ta[(size_t)o * W + c] + tb[(size_t)o * W + c]) +
                                     tq[(size_t)owner[o] * W + c];

    // evidence routing (EvidenceRoutingLayer.forward, l.269)
    for (int l = 0; ok && l < h->n_routing; l++) {
        const clef_block *b = &h->blocks[l];
        layer_norm_rows(qn, opt, O, W, &b->cross_norm, h->eps);
        layer_norm_rows(memn, mem, L, W, &b->cross_norm_kv, h->eps);
        ok = mha(m, &b->cross_attn, qn, O, memn, L, W, H, att);
        if (!ok) break;
        for (size_t i = 0; i < (size_t)O * W; i++) opt[i] += att[i];
        ok = ffn_residual(m, h, b, opt, O);
    }
    if (!ok) goto done;

    // fields: base, routed option summary, global, type (l.405-420)
    linear(m, fld, h->proj_question, qv, Q, E, W, NULL);
    linear(m, glob, h->proj_global, gv, 1, E, W, NULL);
    {
        int o0 = 0;
        float root = sqrtf((float)W);
        for (int i = 0; i < Q; i++) {
            const clef_question *cq = &r->q[i];
            const float *f = fld + (size_t)i * W;
            float *w = rw;   // routing weights over this question's options
            for (int k = 0; k < cq->n_opt; k++)
                w[k] = dotf(opt + (size_t)(o0 + k) * W, f, W) / root;
            softmax_inplace(w, cq->n_opt);
            for (int c = 0; c < W; c++) vec[c] = 0;
            for (int k = 0; k < cq->n_opt; k++) {
                const float *ok_ = opt + (size_t)(o0 + k) * W;
                for (int c = 0; c < W; c++) vec[c] += w[k] * ok_[c];
            }
            layer_norm(fa + (size_t)i * W, vec, &h->option_summary_norm, W, h->eps);
            o0 += cq->n_opt;
        }
        for (int i = 0; i < Q; i++) {
            const float *te = h->type_embd + (size_t)r->q[i].type * W;
            for (int c = 0; c < W; c++) {
                size_t x = (size_t)i * W + c;
                fld[x] = ((fld[x] + fa[x]) + glob[c]) + te[c];
            }
        }
    }

    // TransformerDecoderLayer, norm_first (l.422)
    for (int l = h->n_routing; ok && l < h->n_routing + h->n_joint; l++) {
        const clef_block *b = &h->blocks[l];
        layer_norm_rows(fn, fld, Q, W, &b->self_norm, h->eps);
        ok = mha(m, &b->self_attn, fn, Q, fn, Q, W, H, fa);
        if (!ok) break;
        for (size_t i = 0; i < (size_t)Q * W; i++) fld[i] += fa[i];
        layer_norm_rows(fn, fld, Q, W, &b->cross_norm, h->eps);
        ok = mha(m, &b->cross_attn, fn, Q, mem, L, W, H, fa);
        if (!ok) break;
        for (size_t i = 0; i < (size_t)Q * W; i++) fld[i] += fa[i];
        ok = ffn_residual(m, h, b, fld, Q);
    }
    if (!ok) goto done;
    layer_norm_rows(fld, fld, Q, W, &h->field_norm, h->eps);

    // scores (l.426-457)
    {
        int o = 0;
        for (int i = 0; i < Q; i++) {
            const clef_question *cq = &r->q[i];
            const float *f = fld + (size_t)i * W;
            for (int c = 0; c < E; c++) vec[c] = qv[(size_t)i * E + c] + gv[c];
            normalize(anc, vec, E);
            double fss = 0;
            for (int c = 0; c < W; c++) fss += (double)f[c] * f[c];
            float fnrm = (float)sqrt(fss);
            for (int k = 0; k < cq->n_opt; k++, o++) {
                normalize(vec, lx + (size_t)o * E, E);
                float prior = h->prior_scale * dotf(vec, anc, E);
                float *on = qn;   // option_norm(routed), one row
                layer_norm(on, opt + (size_t)o * W, &h->option_norm, W, h->eps);
                // F.cosine_similarity: each side over its clamped norm
                double oss = 0;
                for (int c = 0; c < W; c++) oss += (double)on[c] * on[c];
                float onrm = (float)sqrt(oss);
                float ia = 1.0f / (fnrm > 1e-8f ? fnrm : 1e-8f);
                float ib = 1.0f / (onrm > 1e-8f ? onrm : 1e-8f);
                float cosv = 0;
                for (int c = 0; c < W; c++) cosv += (f[c] * ia) * (on[c] * ib);
                float *feat = row;
                for (int c = 0; c < W; c++) {
                    feat[c]         = f[c];
                    feat[W + c]     = on[c];
                    feat[2 * W + c] = f[c] * on[c];
                    feat[3 * W + c] = fabsf(f[c] - on[c]);
                }
                linear(m, hid1, h->scorer, feat, 1, 4 * W, W, h->scorer_b);
                gelu_rows(hid1, (size_t)W);
                float residual;
                linear(m, &residual, h->scorer_out, hid1, 1, W, 1, h->scorer_out_b);
                float joint = h->joint_scale * cosv + residual;
                logits[o] = prior + h->gate * joint;
            }
        }
    }

done:
    free(owner); free(rw); free(nh); free(mem); free(memn); free(qv); free(oc); free(lx);
    free(ta); free(tb); free(tq); free(opt); free(qn); free(att); free(fld);
    free(fn); free(fa); free(glob); free(row); free(vec); free(anc); free(hid1);
    return ok;
}

// ---------------------------------------------------------------- the API

// round(x, 4) as CPython computes it: correctly rounded to four decimals
// (half to even on an exact tie), then read back as the nearest double
static double py_round4(double x) {
    if (!fp_isfinite(x) || x == 0) return x;
    bool neg = x < 0;
    double a = neg ? -x : x;
    if (a >= 1e14) return x;   // integral at this precision already
#ifdef __SIZEOF_INT128__
    int e2;
    double fr = frexp(a, &e2);                  // a = fr * 2^e2, fr in [0.5, 1)
    uint64_t mant = (uint64_t)ldexp(fr, 53);    // exact
    int sh = e2 - 53;                           // a = mant * 2^sh, sh < 0 here
    unsigned __int128 num = (unsigned __int128)mant * 10000u;
    uint64_t q = 0;
    if (sh >= 0) {
        q = (uint64_t)(num << sh);
    } else if (-sh < 127) {
        int rs = -sh;
        unsigned __int128 qq = num >> rs;
        unsigned __int128 rem = num - (qq << rs);
        unsigned __int128 half = (unsigned __int128)1 << (rs - 1);
        if (rem > half || (rem == half && (qq & 1))) qq++;
        q = (uint64_t)qq;
    }
    char buf[48];
    snprintf(buf, sizeof buf, "%s%llu.%04llu", neg ? "-" : "",
             (unsigned long long)(q / 10000), (unsigned long long)(q % 10000));
    return strtod(buf, NULL);
#else
    char buf[48];
    snprintf(buf, sizeof buf, "%.4f", x);
    return strtod(buf, NULL);
#endif
}

static void put_num(sbuf *o, double x) {
    char f[40];
    clef_py_float_repr(py_round4(x), f);
    sb_lit(o, f);
}

int clef_systemone_handle(engine *e, const jv *req_c, sbuf *out, const char **err) {
    jv *req = (jv *)req_c;
    model_t *m = e->m;
    if (!m->clef) {
        *err = "this model has no decision head: /v1/systemone serves Clef models "
               "(architecture clef)";
        return 400;
    }
    // encode_record's default bound, or the context when that is smaller
    int max_length = 16384 < m->n_ctx ? 16384 : m->n_ctx;
    clef_record r;
    if (!clef_encode(e->tok, req, max_length, &r, err)) {
        return strncmp(*err, "out of memory", 13) == 0 ? 500 : 400;
    }
    int O = 0;
    for (int i = 0; i < r.n_q; i++) O += r.q[i].n_opt;
    float *hidden = malloc(sizeof(float) * (size_t)r.n_ids * m->n_embd);
    float *logits = malloc(sizeof(float) * (size_t)(O > 0 ? O : 1));
    float *raw = NULL;
    int status = 200;
    if (!hidden || !logits) { *err = "out of memory"; status = 500; goto done; }
    if (!model_hidden_states(m, r.ids, r.n_ids, hidden)) {
        *err = "the backbone forward failed";
        status = 500;
        goto done;
    }
    if (!clef_head_forward(m, &r, hidden, logits)) {
        *err = "out of memory in the decision head";
        status = 500;
        goto done;
    }
    for (int o = 0; o < O; o++)
        if (!fp_isfinite(logits[o])) {
            *err = "the decision head produced a non-finite score";
            status = 500;
            goto done;
        }
    raw = malloc(sizeof(float) * (size_t)(O > 0 ? O : 1));
    if (!raw) { *err = "out of memory"; status = 500; goto done; }
    memcpy(raw, logits, sizeof(float) * (size_t)O);

    // systemone_answer (l.523) per question; the body of systemone (l.565)
    jv *qs = jv_get(req, "questions");
    sb_lit(out, "{\"model\":\"");
    const char *mn = jv_get(req, "model")->str;
    sb_esc(out, mn, strlen(mn));
    sb_lit(out, "\",\"answers\":{");
    int o0 = 0;
    for (int i = 0; i < r.n_q; i++) {
        clef_question *cq = &r.q[i];
        jv *q = qs->items[i];
        int n = cq->n_opt;
        // question_logits.float().softmax(-1): f32, then python floats
        float *pf = logits + o0;
        softmax_inplace(pf, n);
        if (i) sb_lit(out, ",");
        sb_lit(out, "\"");
        sb_esc(out, cq->id, strlen(cq->id));
        sb_lit(out, "\":");
        if (cq->type == 0) {
            sb_lit(out, "{\"type\":\"noul\",\"noul\":");
            put_num(out, (double)pf[0]);   // option 0 is "true"
            sb_lit(out, "}");
        } else if (cq->type == 1) {
            // options in the caller's criteria order; max() keeps the first
            jv *crit = jv_get(q, "criteria");
            double *p = malloc(sizeof(double) * (size_t)crit->n);
            if (!p) { *err = "out of memory"; status = 500; goto done; }
            for (int k = 0; k < crit->n; k++) {
                p[k] = 0;
                for (int j = 0; j < n; j++)
                    if (!strcmp(cq->opt_id[j], crit->keys[k])) { p[k] = (double)pf[j]; break; }
            }
            int best = 0;
            for (int k = 1; k < crit->n; k++) if (p[k] > p[best]) best = k;
            sb_lit(out, "{\"type\":\"choice\",\"choice\":\"");
            sb_esc(out, crit->keys[best], strlen(crit->keys[best]));
            sb_lit(out, "\",\"confidence\":");
            put_num(out, p[best]);
            sb_lit(out, ",\"probabilities\":{");
            for (int k = 0; k < crit->n; k++) {
                sb_lit(out, k ? ",\"" : "\"");
                sb_esc(out, crit->keys[k], strlen(crit->keys[k]));
                sb_lit(out, "\":");
                put_num(out, p[k]);
            }
            sb_lit(out, "}}");
            free(p);
        } else {
            jv *crit = jv_get(q, "criteria");
            double score = 0, mx = 0;
            for (int k = 0; k < n; k++) {
                score += k * (double)pf[k];
                if (k == 0 || (double)pf[k] > mx) mx = (double)pf[k];
            }
            sb_lit(out, "{\"type\":\"score\",\"score\":");
            put_num(out, score);
            sb_lit(out, ",\"confidence\":");
            put_num(out, mx);
            sb_lit(out, ",\"legend\":{");
            for (int k = 0; k < n; k++) {
                sb_fmt(out, "%s\"%d\":", k ? "," : "", k);
                jv_dump(crit->items[k], out);
            }
            sb_lit(out, "},\"probabilities\":{");
            for (int k = 0; k < n; k++) {
                sb_fmt(out, "%s\"%d\":", k ? "," : "", k);
                put_num(out, (double)pf[k]);
            }
            sb_lit(out, "}}");
        }
        o0 += n;
    }
    sb_lit(out, "}");
    // "runner_logits": true (not part of SystemOne): the encoded record and
    // the head's raw scores, for scoring the head against a reference
    if (jv_bool(jv_get(req, "runner_logits"), false)) {
        sb_lit(out, ",\"runner_logits\":{\"input_ids\":[");
        for (int t = 0; t < r.n_ids; t++) sb_fmt(out, "%s%d", t ? "," : "", (int)r.ids[t]);
        sb_lit(out, "],\"questions\":[");
        int ob = 0;
        for (int i = 0; i < r.n_q; i++) {
            clef_question *cq = &r.q[i];
            sb_lit(out, i ? ",{\"id\":\"" : "{\"id\":\"");
            sb_esc(out, cq->id, strlen(cq->id));
            sb_fmt(out, "\",\"question_span\":[%d,%d],\"option_ids\":[", cq->q0, cq->q1);
            for (int k = 0; k < cq->n_opt; k++) {
                sb_lit(out, k ? ",\"" : "\"");
                sb_esc(out, cq->opt_id[k], strlen(cq->opt_id[k]));
                sb_lit(out, "\"");
            }
            sb_lit(out, "],\"option_spans\":[");
            for (int k = 0; k < cq->n_opt; k++)
                sb_fmt(out, "%s[%d,%d]", k ? "," : "", cq->o0[k], cq->o1[k]);
            sb_lit(out, "],\"logits\":[");
            for (int k = 0; k < cq->n_opt; k++)
                sb_fmt(out, "%s%.9g", k ? "," : "", (double)raw[ob + k]);
            sb_lit(out, "]}");
            ob += cq->n_opt;
        }
        sb_lit(out, "]}");
    }
    {
        char ids_sha[65];
        envelope_data_sha256(r.ids, sizeof(int32_t) * (size_t)r.n_ids, ids_sha);
        sb_fmt(out, ",\"usage\":{\"input_tokens\":%d,\"output_tokens\":0},"
                    "\"envelope\":{\"runner_version\":\"%s\",\"encoding\":\"joint_schema_model.encode_record\","
                    "\"input_ids_sha256\":\"%s\",\"max_length\":%d,\"state_truncated\":%s}}",
               r.n_ids, RUNNER_VERSION, ids_sha, max_length,
               r.state_truncated ? "true" : "false");
    }
    if (out->failed) { *err = "out of memory building the response"; status = 500; }

done:
    free(hidden);
    free(logits);
    free(raw);
    clef_record_free(&r);
    return status;
}
