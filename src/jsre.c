// JSON Schema `pattern` as a byte automaton (R2.18); see jsre.h.
#include "jsre.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- code points

typedef struct { uint32_t lo, hi; } cprange;
typedef struct { cprange *r; int n, cap; } cpset;

enum { CP_MAX = 0x10FFFF, SUR_LO = 0xD800, SUR_HI = 0xDFFF };

static bool cp_add(cpset *s, uint32_t lo, uint32_t hi) {
    if (lo > hi) return true;
    if (s->n == s->cap) {
        int cap = s->cap ? s->cap * 2 : 8;
        cprange *g = realloc(s->r, sizeof *g * (size_t)cap);
        if (!g) return false;
        s->r = g; s->cap = cap;
    }
    s->r[s->n++] = (cprange){ lo, hi };
    return true;
}

static int cp_cmp(const void *a, const void *b) {
    uint32_t x = ((const cprange *)a)->lo, y = ((const cprange *)b)->lo;
    return x < y ? -1 : x > y;
}

// Sorted, merged, and with the surrogate block taken out: a range in a pattern
// is over scalar values, and no surrogate has a UTF-8 spelling.
static bool cp_norm(cpset *s) {
    if (!s->n) return true;
    qsort(s->r, (size_t)s->n, sizeof *s->r, cp_cmp);
    int k = 0;
    for (int i = 1; i < s->n; i++) {
        if (s->r[i].lo <= s->r[k].hi + 1) {
            if (s->r[i].hi > s->r[k].hi) s->r[k].hi = s->r[i].hi;
        } else s->r[++k] = s->r[i];
    }
    s->n = k + 1;
    // a failure part-way frees the set being built (R2.18, codex on #875)
    cpset o = { 0 };
    for (int i = 0; i < s->n; i++) {
        uint32_t lo = s->r[i].lo, hi = s->r[i].hi;
        if (hi < SUR_LO || lo > SUR_HI) { if (!cp_add(&o, lo, hi)) { free(o.r); return false; } continue; }
        if (lo < SUR_LO && !cp_add(&o, lo, SUR_LO - 1)) { free(o.r); return false; }
        if (hi > SUR_HI && !cp_add(&o, SUR_HI + 1, hi)) { free(o.r); return false; }
    }
    free(s->r);
    *s = o;
    return true;
}

// The complement over every scalar value. `s` must be normalized.
static bool cp_not(cpset *s) {
    cpset o = { 0 };
    uint32_t next = 0;
    for (int i = 0; i < s->n; i++) {
        if (s->r[i].lo > next && !cp_add(&o, next, s->r[i].lo - 1)) { free(o.r); return false; }
        next = s->r[i].hi + 1;
    }
    if (next <= CP_MAX && !cp_add(&o, next, CP_MAX)) { free(o.r); return false; }
    free(s->r);
    *s = o;
    return cp_norm(s);
}

static bool cp_union(cpset *d, const cpset *a) {
    for (int i = 0; i < a->n; i++) if (!cp_add(d, a->r[i].lo, a->r[i].hi)) return false;
    return cp_norm(d);
}

// What a JSON string spells without an escape: no control character below
// U+0020, no quote, no backslash. Every class is cut to this before it is
// compiled, so the automaton never asks for a byte the validator refuses.
static bool cp_producible(cpset *s) {
    cpset o = { 0 };
    static const cprange allow[] = { { 0x20, 0x21 }, { 0x23, 0x5B }, { 0x5D, CP_MAX } };
    for (int i = 0; i < s->n; i++)
        for (size_t j = 0; j < sizeof allow / sizeof *allow; j++) {
            uint32_t lo = s->r[i].lo > allow[j].lo ? s->r[i].lo : allow[j].lo;
            uint32_t hi = s->r[i].hi < allow[j].hi ? s->r[i].hi : allow[j].hi;
            if (lo <= hi && !cp_add(&o, lo, hi)) { free(o.r); return false; }
        }
    free(s->r);
    *s = o;
    return cp_norm(s);
}

// ECMA-262 classes (Unicode mode): \d and \w ASCII, \s white space and line
// terminators, dot every code point but the four line terminators.
static bool cp_class_escape(cpset *s, char c) {
    cpset t = { 0 };
    bool ok = true;
    switch (c) {
    case 'd': case 'D': ok = cp_add(&t, '0', '9'); break;
    case 'w': case 'W':
        ok = cp_add(&t, '0', '9') && cp_add(&t, 'A', 'Z') && cp_add(&t, '_', '_') && cp_add(&t, 'a', 'z');
        break;
    case 's': case 'S':
        ok = cp_add(&t, 0x09, 0x0D) && cp_add(&t, 0x20, 0x20) && cp_add(&t, 0xA0, 0xA0) &&
             cp_add(&t, 0x1680, 0x1680) && cp_add(&t, 0x2000, 0x200A) && cp_add(&t, 0x2028, 0x2029) &&
             cp_add(&t, 0x202F, 0x202F) && cp_add(&t, 0x205F, 0x205F) && cp_add(&t, 0x3000, 0x3000) &&
             cp_add(&t, 0xFEFF, 0xFEFF);
        break;
    default: free(t.r); return false;
    }
    ok = ok && cp_norm(&t) && (c >= 'a' || cp_not(&t)) && cp_union(s, &t);
    free(t.r);
    return ok;
}

static bool cp_dot(cpset *s) {
    return cp_add(s, 0, 0x09) && cp_add(s, 0x0B, 0x0C) && cp_add(s, 0x0E, 0x2027) &&
           cp_add(s, 0x202A, CP_MAX) && cp_norm(s);
}

// ---------------------------------------------------------------- the parse

enum { N_EMPTY, N_CLASS, N_CAT, N_ALT, N_REP, N_BOL, N_EOL };

typedef struct node {
    int kind;
    struct node *a, *b;
    int min, max;      // N_REP; max -1: unbounded
    cpset cls;         // N_CLASS
} node;

typedef struct {
    const char *p, *end;
    char *err;
    size_t errcap;
    bool failed;
    int depth;
} parser;

static node *mk(parser *ps, int kind) {
    node *n = calloc(1, sizeof *n);
    if (!n) { snprintf(ps->err, ps->errcap, "out of memory compiling a pattern"); ps->failed = true; }
    else n->kind = kind;
    return n;
}

static void node_free(node *n) {
    if (!n) return;
    node_free(n->a);
    node_free(n->b);
    free(n->cls.r);
    free(n);
}

static node *refuse(parser *ps, const char *what) {
    if (!ps->failed) snprintf(ps->err, ps->errcap, "unsupported pattern: %s", what);
    ps->failed = true;
    return NULL;
}

// One code point of the pattern's own text (UTF-8), or -1.
static int32_t take_utf8(parser *ps) {
    const unsigned char *s = (const unsigned char *)ps->p;
    if (ps->p >= ps->end) return -1;
    uint32_t c = s[0];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!n || ps->p + n > ps->end) return -1;
    if (n > 1) {
        c &= (uint32_t)(0x7F >> n);
        for (int i = 1; i < n; i++) {
            if ((s[i] & 0xC0) != 0x80) return -1;
            c = c << 6 | (s[i] & 0x3F);
        }
    }
    ps->p += n;
    return (int32_t)c;
}

static int hexval(char c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10
         : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

static int32_t take_hex(parser *ps, int n) {
    int32_t v = 0;
    for (int i = 0; i < n; i++) {
        if (ps->p >= ps->end || hexval(*ps->p) < 0) return -1;
        v = v * 16 + hexval(*ps->p++);
    }
    return v;
}

// An escape after the backslash, as a code point; a class escape goes into
// `cls` (returns -2). `in_class`: \b is a backspace there and an assertion
// outside. Returns -1 having refused.
static int32_t escape(parser *ps, cpset *cls, bool in_class) {
    if (ps->p >= ps->end) { refuse(ps, "a trailing backslash"); return -1; }
    char c = *ps->p++;
    switch (c) {
    case 'd': case 'D': case 'w': case 'W': case 's': case 'S':
        if (!cp_class_escape(cls, c)) { refuse(ps, "out of memory"); return -1; }
        return -2;
    case 'p': case 'P': refuse(ps, "a Unicode property class (\\p{...})"); return -1;
    case 'b':
        if (in_class) return 0x08;
        refuse(ps, "a word boundary (\\b)"); return -1;
    case 'B': refuse(ps, "a word boundary (\\B)"); return -1;
    case 'k': refuse(ps, "a named backreference (\\k)"); return -1;
    case 'f': return 0x0C;
    case 'n': return 0x0A;
    case 'r': return 0x0D;
    case 't': return 0x09;
    case 'v': return 0x0B;
    case '0':
        if (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') {
            refuse(ps, "an octal escape"); return -1;
        }
        return 0;
    case 'c':
        if (ps->p < ps->end && ((*ps->p >= 'a' && *ps->p <= 'z') || (*ps->p >= 'A' && *ps->p <= 'Z')))
            return *ps->p++ % 32;
        refuse(ps, "a \\c escape without a letter"); return -1;
    case 'x': {
        int32_t v = take_hex(ps, 2);
        if (v < 0) refuse(ps, "a \\x escape without two hex digits");
        return v;
    }
    case 'u': {
        int32_t v;
        if (ps->p < ps->end && *ps->p == '{') {
            ps->p++;
            v = 0;
            int digits = 0;
            while (ps->p < ps->end && hexval(*ps->p) >= 0 && v <= CP_MAX) { v = v * 16 + hexval(*ps->p++); digits++; }
            if (!digits || ps->p >= ps->end || *ps->p != '}' || v > CP_MAX) {
                refuse(ps, "a malformed \\u{...} escape"); return -1;
            }
            ps->p++;
            return v;
        }
        v = take_hex(ps, 4);
        if (v < 0) { refuse(ps, "a \\u escape without four hex digits"); return -1; }
        // a surrogate pair spelled as two escapes is one code point
        if (v >= 0xD800 && v <= 0xDBFF && ps->end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
            const char *save = ps->p;
            ps->p += 2;
            int32_t lo = take_hex(ps, 4);
            if (lo >= 0xDC00 && lo <= 0xDFFF) return 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
            ps->p = save;
        }
        return v;
    }
    default:
        if (c >= '1' && c <= '9') { refuse(ps, "a backreference"); return -1; }
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            refuse(ps, "an unknown letter escape"); return -1;
        }
        if ((unsigned char)c >= 0x80) { refuse(ps, "an escaped non-ASCII character"); return -1; }
        return (unsigned char)c;   // \. \- \/ \$ ...: the character itself
    }
}

static node *class_node(parser *ps, cpset *s) {
    node *n = mk(ps, N_CLASS);
    if (!n) { free(s->r); return NULL; }
    n->cls = *s;
    return n;
}

static node *parse_class(parser *ps) {
    bool neg = false;
    if (ps->p < ps->end && *ps->p == '^') { neg = true; ps->p++; }
    cpset s = { 0 };
    // ECMA has no POSIX first-place ]: [] is the empty class (it matches
    // nothing) and [^] matches any code point
    for (;;) {
        if (ps->p >= ps->end) { free(s.r); return refuse(ps, "an unterminated character class"); }
        if (*ps->p == ']') { ps->p++; break; }
        int32_t lo;
        if (*ps->p == '\\') {
            ps->p++;
            lo = escape(ps, &s, true);
            if (lo == -1) { free(s.r); return NULL; }
            if (lo == -2) {
                if (ps->p + 1 < ps->end && ps->p[0] == '-' && ps->p[1] != ']') {
                    free(s.r); return refuse(ps, "a class range from a class escape");
                }
                continue;
            }
        } else {
            lo = take_utf8(ps);
            if (lo < 0) { free(s.r); return refuse(ps, "invalid UTF-8 in the pattern"); }
        }
        int32_t hi = lo;
        if (ps->p + 1 < ps->end && ps->p[0] == '-' && ps->p[1] != ']') {
            ps->p++;
            if (*ps->p == '\\') {
                ps->p++;
                cpset dummy = { 0 };
                hi = escape(ps, &dummy, true);
                free(dummy.r);
                if (hi == -1) { free(s.r); return NULL; }
                if (hi == -2) { free(s.r); return refuse(ps, "a class range to a class escape"); }
            } else {
                hi = take_utf8(ps);
                if (hi < 0) { free(s.r); return refuse(ps, "invalid UTF-8 in the pattern"); }
            }
            if (hi < lo) { free(s.r); return refuse(ps, "a class range out of order"); }
        }
        if (!cp_add(&s, (uint32_t)lo, (uint32_t)hi)) { free(s.r); return refuse(ps, "out of memory"); }
    }
    if (!cp_norm(&s) || (neg && !cp_not(&s))) { free(s.r); return refuse(ps, "out of memory"); }
    return class_node(ps, &s);
}

static node *parse_alt(parser *ps);
static bool parse_braces(parser *ps, int *min, int *max);
static bool malformed_count(const parser *ps);

static node *parse_atom(parser *ps) {
    char c = *ps->p;
    if (c == '(') {
        ps->p++;
        if (ps->p < ps->end && *ps->p == '?') {
            ps->p++;
            if (ps->p < ps->end && *ps->p == ':') ps->p++;
            else if (ps->p < ps->end && (*ps->p == '=' || *ps->p == '!'))
                return refuse(ps, "a lookahead");
            else if (ps->p < ps->end && *ps->p == '<' && ps->p + 1 < ps->end &&
                     (ps->p[1] == '=' || ps->p[1] == '!'))
                return refuse(ps, "a lookbehind");
            else if (ps->p < ps->end && (*ps->p == '<' || *ps->p == 'P'))
                return refuse(ps, "a named group");
            else return refuse(ps, "an inline flag");
        }
        if (++ps->depth > 200) return refuse(ps, "groups nested too deep");
        node *inner = parse_alt(ps);
        ps->depth--;
        if (!inner) return NULL;
        if (ps->p >= ps->end || *ps->p != ')') { node_free(inner); return refuse(ps, "an unclosed group"); }
        ps->p++;
        return inner;
    }
    if (c == '[') { ps->p++; return parse_class(ps); }
    cpset s = { 0 };
    if (c == '.') {
        ps->p++;
        if (!cp_dot(&s)) { free(s.r); return refuse(ps, "out of memory"); }
        return class_node(ps, &s);
    }
    if (c == '^') { ps->p++; return mk(ps, N_BOL); }
    if (c == '$') { ps->p++; return mk(ps, N_EOL); }
    if (c == '\\') {
        ps->p++;
        int32_t v = escape(ps, &s, false);
        if (v == -1) { free(s.r); return NULL; }
        if (v >= 0 && !cp_add(&s, (uint32_t)v, (uint32_t)v)) { free(s.r); return refuse(ps, "out of memory"); }
        if (!cp_norm(&s)) { free(s.r); return refuse(ps, "out of memory"); }
        return class_node(ps, &s);
    }
    if (c == '*' || c == '+' || c == '?') return refuse(ps, "a quantifier with nothing to repeat");
    if (c == '{') {
        // a count needs an atom before it (both ECMA modes refuse {2} there);
        // any other brace is text, as legacy mode reads it
        int lo, hi;
        parser probe = *ps;
        if (parse_braces(&probe, &lo, &hi)) return refuse(ps, "a quantifier with nothing to repeat");
        if (malformed_count(ps)) return refuse(ps, "a malformed count ({}, {,n} or {n,m,k})");
    }
    if (c == ')') return refuse(ps, "an unbalanced parenthesis");
    int32_t v = take_utf8(ps);
    if (v < 0) return refuse(ps, "invalid UTF-8 in the pattern");
    if (!cp_add(&s, (uint32_t)v, (uint32_t)v) || !cp_norm(&s)) { free(s.r); return refuse(ps, "out of memory"); }
    return class_node(ps, &s);
}

// {n}, {n,} or {n,m} at ps->p, or false (then `{` is a literal: Annex B).
static bool parse_braces(parser *ps, int *min, int *max) {
    const char *q = ps->p + 1;
    long n = 0, m;
    int digits = 0;
    while (q < ps->end && *q >= '0' && *q <= '9') { n = n * 10 + (*q++ - '0'); digits++; if (n > 100000) return false; }
    if (!digits || q >= ps->end) return false;
    if (*q == '}') { m = n; q++; }
    else if (*q == ',') {
        q++;
        if (q < ps->end && *q == '}') { m = -1; q++; }
        else {
            m = 0; digits = 0;
            while (q < ps->end && *q >= '0' && *q <= '9') { m = m * 10 + (*q++ - '0'); digits++; if (m > 100000) return false; }
            if (!digits || q >= ps->end || *q != '}') return false;
            q++;
        }
    } else return false;
    *min = (int)n; *max = (int)m;
    ps->p = q;
    return true;
}

// A run of digits and commas in braces that is not a count: {}, {,3},
// {2,3,4}. Legacy ECMA reads it as text, Python 3.11 reads {,3} as {0,3}, the
// u flag refuses it: engines disagree on what it means, so it is refused.
static bool malformed_count(const parser *ps) {
    const char *q = ps->p + 1;
    while (q < ps->end && ((*q >= '0' && *q <= '9') || *q == ',')) q++;
    return q < ps->end && *q == '}';   // reached only after parse_braces refused it as a count
}

static node *parse_term(parser *ps) {
    node *a = parse_atom(ps);
    if (!a) return NULL;
    for (bool quantified = false;; quantified = true) {
        if (ps->p >= ps->end) return a;
        int min, max;
        char c = *ps->p;
        if (c == '*') { min = 0; max = -1; ps->p++; }
        else if (c == '+') { min = 1; max = -1; ps->p++; }
        else if (c == '?') { min = 0; max = 1; ps->p++; }
        else if (c == '{' && parse_braces(ps, &min, &max)) {
            if (max >= 0 && max < min) { node_free(a); return refuse(ps, "a count range out of order"); }
        } else if (c == '{' && malformed_count(ps)) {
            node_free(a); return refuse(ps, "a malformed count ({}, {,n} or {n,m,k})");
        } else return a;
        // a{2}{3}, a**: ECMA (and Python) refuse a quantifier on a quantifier
        if (quantified) { node_free(a); return refuse(ps, "a quantifier on a quantifier"); }
        if (a->kind == N_BOL || a->kind == N_EOL) { node_free(a); return refuse(ps, "a quantified anchor"); }
        if (ps->p < ps->end && *ps->p == '?') ps->p++;   // lazy: the same language as greedy
        node *r = mk(ps, N_REP);
        if (!r) { node_free(a); return NULL; }
        r->a = a; r->min = min; r->max = max;
        a = r;
    }
}

// A list of terms (or alternatives) as one node, joined under `kind` in a
// BALANCED tree: compiling and freeing recurse on the tree, so a sequence of
// k terms recurses log2(k) deep rather than k (a schema compiles on a server
// thread, whose stack can be small, and a pattern can be thousands of
// characters). Adjacent pairs join first, so a sequence keeps its order.
static node *join(parser *ps, node **v, int k, int kind) {
    while (k > 1) {
        int j = 0;
        for (int i = 0; i < k; i += 2) {
            if (i + 1 == k) { v[j++] = v[i]; continue; }
            node *c = mk(ps, kind);
            if (!c) {
                for (int m = 0; m < j; m++) node_free(v[m]);
                for (int m = i; m < k; m++) node_free(v[m]);
                return NULL;
            }
            c->a = v[i]; c->b = v[i + 1];
            v[j++] = c;
        }
        k = j;
    }
    return v[0];
}

static bool push(parser *ps, node ***v, int *k, int *cap, node *t) {
    if (*k == *cap) {
        int c = *cap ? *cap * 2 : 8;
        node **g = realloc(*v, sizeof *g * (size_t)c);
        if (!g) { snprintf(ps->err, ps->errcap, "out of memory compiling a pattern"); ps->failed = true; return false; }
        *v = g; *cap = c;
    }
    (*v)[(*k)++] = t;
    return true;
}

static void free_list(node **v, int k) { for (int i = 0; i < k; i++) node_free(v[i]); free(v); }

static node *parse_seq(parser *ps) {
    node **v = NULL;
    int k = 0, cap = 0;
    while (ps->p < ps->end && *ps->p != '|' && *ps->p != ')') {
        node *t = parse_term(ps);
        if (!t || !push(ps, &v, &k, &cap, t)) { node_free(t); free_list(v, k); return NULL; }
    }
    if (!k) { free(v); return mk(ps, N_EMPTY); }   // an empty alternative matches the empty string
    node *r = join(ps, v, k, N_CAT);
    free(v);
    return r;
}

static node *parse_alt(parser *ps) {
    node **v = NULL;
    int k = 0, cap = 0;
    for (;;) {
        node *b = parse_seq(ps);
        if (!b || !push(ps, &v, &k, &cap, b)) { node_free(b); free_list(v, k); return NULL; }
        if (ps->p >= ps->end || *ps->p != '|') break;
        ps->p++;
    }
    node *r = join(ps, v, k, N_ALT);
    free(v);
    return r;
}

// ---------------------------------------------------------------- the automaton

enum { S_BYTE, S_SPLIT, S_BOL, S_EOL, S_MATCH, S_DEAD };

typedef struct { uint8_t kind, lo, hi; int out, out1; } state;

struct jsre {
    state *s;
    int n, cap;
    int start;
    size_t words;
    uint64_t *live;      // a state from which a match is reachable through literal bytes
    uint64_t *end_ok;    // an S_EOL whose continuation matches at the end of the string
    uint8_t cls[256];    // byte -> byte class: bytes no transition tells apart share one
    uint8_t rep[256];    // a byte of each class
    int ncls;
    // exact completion lengths (jsre_lengths): lv[l] is the states from which
    // an accepting completion of exactly l more code points exists; from
    // lv_tail on the sequence repeats with period lv_period (0: computed only
    // up to the largest length a caller asks, lv_n - 1)
    uint64_t *lv;
    int lv_n, lv_tail, lv_period;
    bool too_big;
};

static int st_new(jsre *re, int kind, int lo, int hi, int out, int out1) {
    if (re->n >= JSRE_MAX_STATES) { re->too_big = true; return -1; }
    if (re->n == re->cap) {
        int cap = re->cap ? re->cap * 2 : 64;
        state *g = realloc(re->s, sizeof *g * (size_t)cap);
        if (!g) { re->too_big = true; return -1; }
        re->s = g; re->cap = cap;
    }
    re->s[re->n] = (state){ (uint8_t)kind, (uint8_t)lo, (uint8_t)hi, out, out1 };
    return re->n++;
}

static int utf8_encode(uint32_t c, uint8_t *b) {
    if (c < 0x80) { b[0] = (uint8_t)c; return 1; }
    if (c < 0x800) { b[0] = (uint8_t)(0xC0 | c >> 6); b[1] = (uint8_t)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) {
        b[0] = (uint8_t)(0xE0 | c >> 12); b[1] = (uint8_t)(0x80 | (c >> 6 & 0x3F));
        b[2] = (uint8_t)(0x80 | (c & 0x3F)); return 3;
    }
    b[0] = (uint8_t)(0xF0 | c >> 18); b[1] = (uint8_t)(0x80 | (c >> 12 & 0x3F));
    b[2] = (uint8_t)(0x80 | (c >> 6 & 0x3F)); b[3] = (uint8_t)(0x80 | (c & 0x3F)); return 4;
}

// A range of code points with one UTF-8 length, split until every piece is a
// product of byte ranges (the construction RE2 and Rust's regex use), each
// piece a chain of byte states ending at `next`; the pieces joined by splits.
static int utf8_range(jsre *re, uint32_t lo, uint32_t hi, int next, int acc) {
    int len = lo < 0x80 ? 1 : lo < 0x800 ? 2 : lo < 0x10000 ? 3 : 4;
    for (int i = 1; i < len; i++) {
        uint32_t m = (1u << (6 * i)) - 1;
        if ((lo & ~m) != (hi & ~m)) {
            if (lo & m) {
                acc = utf8_range(re, lo, lo | m, next, acc);
                return acc < 0 ? -1 : utf8_range(re, (lo | m) + 1, hi, next, acc);
            }
            if ((hi & m) != m) {
                acc = utf8_range(re, lo, (hi & ~m) - 1, next, acc);
                return acc < 0 ? -1 : utf8_range(re, hi & ~m, hi, next, acc);
            }
        }
    }
    uint8_t a[4], b[4];
    utf8_encode(lo, a);
    utf8_encode(hi, b);
    int t = next;
    for (int i = len - 1; i >= 0; i--) {
        t = st_new(re, S_BYTE, a[i], b[i], t, -1);
        if (t < 0) return -1;
    }
    return acc < 0 ? t : st_new(re, S_SPLIT, 0, 0, t, acc);
}

static int compile_class(jsre *re, const cpset *s, int next) {
    static const uint32_t bound[] = { 0x7F, 0x7FF, 0xFFFF, CP_MAX };
    int acc = -1;
    for (int i = 0; i < s->n; i++) {
        uint32_t lo = s->r[i].lo;
        for (int k = 0; k < 4 && lo <= s->r[i].hi; k++) {
            if (lo > bound[k]) continue;
            uint32_t hi = s->r[i].hi < bound[k] ? s->r[i].hi : bound[k];
            int a = utf8_range(re, lo, hi, next, acc < 0 ? -1 : acc);
            if (a < 0) return -1;
            acc = a;
            lo = hi + 1;
        }
    }
    return acc < 0 ? st_new(re, S_DEAD, 0, 0, -1, -1) : acc;
}

// Compiled backwards: `next` is where a match of `n` continues.
static int compile(jsre *re, const node *n, int next) {
    if (next < 0) return -1;
    switch (n->kind) {
    case N_EMPTY: return next;
    case N_CLASS: return compile_class(re, &n->cls, next);
    case N_CAT: return compile(re, n->a, compile(re, n->b, next));
    case N_ALT: {
        int a = compile(re, n->a, next), b = compile(re, n->b, next);
        return a < 0 || b < 0 ? -1 : st_new(re, S_SPLIT, 0, 0, a, b);
    }
    case N_BOL: return st_new(re, S_BOL, 0, 0, next, -1);
    case N_EOL: return st_new(re, S_EOL, 0, 0, next, -1);
    case N_REP: {
        int t = next;
        if (n->max < 0) {
            int loop = st_new(re, S_SPLIT, 0, 0, -1, next);   // out filled below
            if (loop < 0) return -1;
            int body = compile(re, n->a, loop);
            if (body < 0) return -1;
            re->s[loop].out = body;
            t = loop;
        } else {
            for (int i = 0; i < n->max - n->min && t >= 0; i++) {
                int body = compile(re, n->a, t);
                t = body < 0 ? -1 : st_new(re, S_SPLIT, 0, 0, body, next);
            }
        }
        for (int i = 0; i < n->min && t >= 0; i++) t = compile(re, n->a, t);
        return t;
    }
    }
    return -1;
}

static inline bool bit(const uint64_t *s, int i) { return s[i >> 6] >> (i & 63) & 1; }
static inline void setb(uint64_t *s, int i) { s[i >> 6] |= 1ull << (i & 63); }

// Every state reachable from `i` without a byte: ^ passes only at the start,
// $ is kept (it passes only in the acceptance check).
static void closure(const jsre *re, uint64_t *set, int i, bool at_start, int *stack) {
    int sp = 0;
    stack[sp++] = i;
    while (sp) {
        int k = stack[--sp];
        if (k < 0 || bit(set, k)) continue;
        setb(set, k);
        const state *s = &re->s[k];
        if (s->kind == S_SPLIT) { stack[sp++] = s->out; stack[sp++] = s->out1; }
        else if (s->kind == S_BOL && at_start) stack[sp++] = s->out;
    }
}

static bool end_reaches_match(const jsre *re, int k, int *stack, uint64_t *seen) {
    memset(seen, 0, re->words * sizeof *seen);
    int sp = 0;
    stack[sp++] = k;
    while (sp) {
        int i = stack[--sp];
        if (i < 0 || bit(seen, i)) continue;
        setb(seen, i);
        const state *s = &re->s[i];
        if (s->kind == S_MATCH) return true;
        if (s->kind == S_SPLIT) { stack[sp++] = s->out; stack[sp++] = s->out1; }
        else if (s->kind == S_EOL) stack[sp++] = s->out;   // $ after $ at the end still holds
    }
    return false;
}

static bool restrict_producible(node *n) {
    if (!n) return true;
    if (n->kind == N_CLASS && !cp_producible(&n->cls)) return false;
    return restrict_producible(n->a) && restrict_producible(n->b);
}

// `source`, not `pattern`: schema.h has a type of that name, and a test that
// compiles both files as one unit fails gcc's -Wshadow on it (clang accepts it)
jsre *jsre_compile(const char *source, char *err, size_t errcap) {
    parser ps = { source, source + strlen(source), err, errcap, false, 0 };
    node *ast = parse_alt(&ps);
    if (ast && ps.p < ps.end) { node_free(ast); ast = refuse(&ps, "an unbalanced parenthesis"); }
    if (!ast) return NULL;
    // search semantics: any literal code points around the pattern
    cpset any = { 0 };
    node *pre = NULL, *post = NULL, *top = NULL, *mid = NULL;
    bool ok = cp_add(&any, 0, CP_MAX) && cp_norm(&any);
    jsre *re = calloc(1, sizeof *re);
    if (ok && re) {
        pre = mk(&ps, N_REP); post = mk(&ps, N_REP);
        node *ca = mk(&ps, N_CLASS), *cb = mk(&ps, N_CLASS);
        mid = mk(&ps, N_CAT); top = mk(&ps, N_CAT);
        if (pre && post && ca && cb && mid && top) {
            cpset dup = { 0 };
            ok = cp_union(&dup, &any);
            ca->cls = any; any.r = NULL;
            cb->cls = dup;
            pre->a = ca; pre->min = 0; pre->max = -1;
            post->a = cb; post->min = 0; post->max = -1;
            mid->a = ast; mid->b = post;
            top->a = pre; top->b = mid;
            ast = NULL;
        } else { node_free(ca); node_free(cb); node_free(pre); node_free(post); node_free(mid);
                 node_free(top); pre = post = mid = top = NULL; ok = false; }
    } else ok = false;
    free(any.r);
    if (!ok || !re) {
        node_free(ast); node_free(top); free(re);
        snprintf(err, errcap, "out of memory compiling a pattern");
        return NULL;
    }
    // every class cut to what JSON spells without an escape: every node,
    // never a bounded walk that could skip one (a skipped class would be wider)
    ok = restrict_producible(top);
    int match = st_new(re, S_MATCH, 0, 0, -1, -1);
    re->start = ok ? compile(re, top, match) : -1;
    node_free(top);
    if (re->start < 0) {
        snprintf(err, errcap, re->too_big ? "unsupported pattern: more than %d automaton states"
                                          : "out of memory compiling a pattern", JSRE_MAX_STATES);
        jsre_free(re);
        return NULL;
    }
    re->words = ((size_t)re->n + 63) / 64;
    // byte classes: a cut wherever some transition's range starts or ends, so
    // two bytes of one class step every state set alike (a DFA table then
    // needs one entry per class, not 256)
    bool cut[257] = { false };
    for (int k = 0; k < re->n; k++)
        if (re->s[k].kind == S_BYTE) { cut[re->s[k].lo] = true; cut[re->s[k].hi + 1] = true; }
    int c = 0;
    for (int b = 0; b < 256; b++) {
        if (b > 0 && cut[b]) c++;
        if (b == 0 || cut[b]) re->rep[c] = (uint8_t)b;
        re->cls[b] = (uint8_t)c;
    }
    re->ncls = c + 1;
    re->live = calloc(re->words, sizeof *re->live);
    re->end_ok = calloc(re->words, sizeof *re->end_ok);
    int *stack = malloc(sizeof *stack * (size_t)(re->n * 2 + 4));
    uint64_t *seen = calloc(re->words, sizeof *seen);
    if (!re->live || !re->end_ok || !stack || !seen) {
        free(stack); free(seen); jsre_free(re);
        snprintf(err, errcap, "out of memory compiling a pattern");
        return NULL;
    }
    for (int k = 0; k < re->n; k++)
        if (re->s[k].kind == S_EOL && end_reaches_match(re, k, stack, seen)) setb(re->end_ok, k);
    // live: a match is reachable through literal bytes (^ is not passable
    // after the first byte; the start set passes it before this mask applies)
    bool changed = true;
    while (changed) {
        changed = false;
        for (int k = 0; k < re->n; k++) {
            if (bit(re->live, k)) continue;
            const state *s = &re->s[k];
            bool l = s->kind == S_MATCH || (s->kind == S_EOL && bit(re->end_ok, k)) ||
                     (s->kind == S_BYTE && bit(re->live, s->out)) ||
                     (s->kind == S_SPLIT && (bit(re->live, s->out) || bit(re->live, s->out1)));
            if (l) { setb(re->live, k); changed = true; }
        }
    }
    uint64_t *set = seen;
    memset(set, 0, re->words * sizeof *set);
    closure(re, set, re->start, true, stack);
    bool any_live = false;
    for (size_t w = 0; w < re->words; w++) any_live |= (set[w] & re->live[w]) != 0;
    // The empty string is the one place where ^ and $ both hold, so there a
    // path may pass $ and then ^ ($^, (a|$)(^|b)), which the start closure
    // (^ only) and the end check ($ only) never walk. ECMA matches "" there;
    // refuse such a pattern rather than miss it.
    bool start_acc = jsre_accepts(re, set);
    memset(set, 0, re->words * sizeof *set);
    int sp = 0;
    stack[sp++] = re->start;
    bool empty_ok = false;
    while (sp && !empty_ok) {
        int k = stack[--sp];
        if (k < 0 || bit(set, k)) continue;
        setb(set, k);
        const state *s = &re->s[k];
        if (s->kind == S_MATCH) empty_ok = true;
        else if (s->kind == S_SPLIT) { stack[sp++] = s->out; stack[sp++] = s->out1; }
        else if (s->kind == S_BOL || s->kind == S_EOL) stack[sp++] = s->out;
    }
    free(stack); free(seen);
    if (empty_ok && !start_acc) {
        snprintf(err, errcap, "unsupported pattern: an end assertion before a start assertion that "
                 "only the empty string can pass ($ then ^)");
        jsre_free(re);
        return NULL;
    }
    if (!any_live) {
        snprintf(err, errcap, "pattern describes no string JSON can spell without an escape "
                 "(it matches none, or every match needs a control character, a quote or a backslash)");
        jsre_free(re);
        return NULL;
    }
    return re;
}

void jsre_free(jsre *re) {
    if (!re) return;
    free(re->s); free(re->live); free(re->end_ok); free(re->lv); free(re);
}

// ---------------------------------------------------------------- lengths
//
// minLength and maxLength count code points, and with a pattern they must hold
// together with it: a prefix is admitted only when some completion the pattern
// accepts ends inside the bounds, or the model is led where nothing can close
// the string. The shortest and longest completions are not enough: ^(ab)+$
// between 3 and 3 code points has a shortest of 2 and no longest, and no
// string at all. So the lengths are exact: lv[l] holds the states with an
// accepting completion of exactly l more code points, a byte that starts a
// code point costing one and a continuation byte none (so a state inside a
// code point already counted needs only what follows it). lv[0] reaches an
// accepting state through splits and continuation bytes; lv[l + 1] is what
// reaches, the same way, a byte that starts a code point and leads into
// lv[l]. One sequence serves every state set: a set has a completion of l
// code points when it meets lv[l]. A sequence of sets over finitely many
// states repeats; it is kept up to its first repeat, or up to the largest
// length a caller asks when that comes first.

static bool starts_cp(const state *s) { return s->lo < 0x80 || s->lo >= 0xC0; }

static uint64_t words_hash(const uint64_t *s, size_t w) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < w; i++) { h ^= s[i]; h *= 0x100000001B3ull; h ^= h >> 31; }
    return h;
}

bool jsre_lengths(jsre *re, int upto, char *err, size_t errcap) {
    int n = re->n;
    size_t w = re->words;
    // upto + 1 in long long: a maxLength reaches INT_MAX (R2.18, codex on #875),
    // and long is 32 bits on Windows
    int cap = upto >= 0 && (long long)upto + 1 < JSRE_LEN_MAX ? upto + 1 : JSRE_LEN_MAX;
    int *off0 = calloc((size_t)n + 1, sizeof *off0), *off1 = calloc((size_t)n + 1, sizeof *off1);
    int *src0 = NULL, *src1 = NULL, *queue = malloc(sizeof *queue * (size_t)n);
    uint64_t *lv = malloc(sizeof *lv * w * (size_t)cap), *y = malloc(sizeof *y * w);
    int *hidx = malloc(sizeof *hidx * (size_t)cap * 2);
    bool ok = off0 && off1 && queue && lv && y && hidx;
    // reverse edges: free moves (splits, continuation bytes) and code-point starts
    for (int k = 0; ok && k < n; k++) {
        const state *s = &re->s[k];
        if (s->kind == S_SPLIT) { if (s->out >= 0) off0[s->out + 1]++; if (s->out1 >= 0) off0[s->out1 + 1]++; }
        else if (s->kind == S_BYTE && s->out >= 0) (starts_cp(s) ? off1 : off0)[s->out + 1]++;
    }
    for (int k = 0; ok && k < n; k++) { off0[k + 1] += off0[k]; off1[k + 1] += off1[k]; }
    if (ok) {
        src0 = malloc(sizeof *src0 * ((size_t)off0[n] + 1));
        src1 = malloc(sizeof *src1 * ((size_t)off1[n] + 1));
        ok = src0 && src1;
    }
    if (ok) {
        int *f0 = calloc((size_t)n, sizeof *f0), *f1 = calloc((size_t)n, sizeof *f1);
        ok = f0 && f1;
        for (int k = 0; ok && k < n; k++) {
            const state *s = &re->s[k];
            if (s->kind == S_SPLIT) {
                if (s->out >= 0) src0[off0[s->out] + f0[s->out]++] = k;
                if (s->out1 >= 0) src0[off0[s->out1] + f0[s->out1]++] = k;
            } else if (s->kind == S_BYTE && s->out >= 0) {
                if (starts_cp(s)) src1[off1[s->out] + f1[s->out]++] = k;
                else src0[off0[s->out] + f0[s->out]++] = k;
            }
        }
        free(f0); free(f1);
    }
    if (!ok) {
        free(off0); free(off1); free(src0); free(src1); free(queue); free(lv); free(y); free(hidx);
        snprintf(err, errcap, "out of memory compiling a pattern's lengths");
        return false;
    }
    for (int i = 0; i < cap * 2; i++) hidx[i] = -1;
    int count = 0, tail = -1;
    // y: the seeds of the next set; the set is what reaches them by free moves
    memset(y, 0, w * sizeof *y);
    for (int k = 0; k < n; k++)
        if (re->s[k].kind == S_MATCH || (re->s[k].kind == S_EOL && bit(re->end_ok, k))) setb(y, k);
    while (count < cap) {
        uint64_t *v = lv + (size_t)count * w;
        memcpy(v, y, w * sizeof *v);
        int qh = 0, qt = 0;
        for (int k = 0; k < n; k++) if (bit(v, k)) queue[qt++] = k;
        while (qh < qt) {
            int t = queue[qh++];
            for (int e = off0[t]; e < off0[t + 1]; e++)
                if (!bit(v, src0[e])) { setb(v, src0[e]); queue[qt++] = src0[e]; }
        }
        // a repeat of an earlier set: the sequence cycles from there
        size_t hs = (size_t)cap * 2, h = words_hash(v, w) % hs;
        for (; hidx[h] >= 0; h = (h + 1) % hs)
            if (!memcmp(lv + (size_t)hidx[h] * w, v, w * sizeof *v)) { tail = hidx[h]; break; }
        if (tail >= 0) break;
        hidx[h] = count++;
        memset(y, 0, w * sizeof *y);
        for (int t = 0; t < n; t++)
            if (bit(v, t)) for (int e = off1[t]; e < off1[t + 1]; e++) setb(y, src1[e]);
    }
    free(off0); free(off1); free(src0); free(src1); free(queue); free(y); free(hidx);
    if (tail < 0 && (upto < 0 || (long long)upto + 1 > count)) {
        free(lv);
        snprintf(err, errcap, "unsupported pattern with a length bound: the lengths it allows do not "
                 "settle within %d code points", JSRE_LEN_MAX);
        return false;
    }
    uint64_t *g = realloc(lv, sizeof *lv * w * (size_t)(count ? count : 1));
    free(re->lv);
    re->lv = g ? g : lv;
    re->lv_n = count;
    re->lv_tail = tail >= 0 ? tail : count;
    re->lv_period = tail >= 0 ? count - tail : 0;
    return true;
}

static bool meets(const uint64_t *a, const uint64_t *b, size_t w) {
    for (size_t i = 0; i < w; i++) if (a[i] & b[i]) return true;
    return false;
}

// Does `set` have an accepting completion of some l code points, lo <= l <= hi
// (hi < 0: no upper bound)? A length past the kept sequence maps into its cycle.
bool jsre_len_ok(const jsre *re, const uint64_t *set, int lo, int hi) {
    if (!re->lv) return hi < 0 || hi >= (lo > 0 ? lo : 0);
    int a = lo > 0 ? lo : 0;
    if (hi >= 0 && hi < a) return false;
    size_t w = re->words;
    // lengths below lv_n index the sequence directly
    for (int l = a; l < re->lv_n && (hi < 0 || l <= hi); l++)
        if (meets(set, re->lv + (size_t)l * w, w)) return true;
    if (!re->lv_period) return false;           // never asked past lv_n - 1
    long long from = a > re->lv_n ? a : re->lv_n;    // the rest is the cycle, at most once round
    for (long long l = from; l < from + re->lv_period && (hi < 0 || l <= hi); l++)
        if (meets(set, re->lv + (size_t)(re->lv_tail + (l - re->lv_tail) % re->lv_period) * w, w)) return true;
    return false;
}

int jsre_len_entries(const jsre *re) { return re->lv_n; }

// The bytes a closer writes to finish a string: the shortest completion from
// `set` (NULL: the start) that the pattern accepts, of between lo and hi more
// code points (hi < 0: no upper bound). Built a byte at a time keeping one
// invariant, that a completion of exactly r more code points exists from
// here, so it cannot dead-end. Bytes are tried letters and digits first, for
// a filler a reader can read. -1 when none fits or `cap` is too small.
int jsre_witness(const jsre *re, const uint64_t *set, int lo, int hi, char *out, int cap) {
    if (!re->lv) return -1;
    size_t w = re->words;
    uint64_t *cur = malloc(w * sizeof *cur), *nx = malloc(w * sizeof *nx);
    if (!cur || !nx) { free(cur); free(nx); return -1; }
    if (set) memcpy(cur, set, w * sizeof *cur);
    else jsre_start(re, cur);
    // in long long (long is 32 bits on Windows, where lo near INT_MAX wrapped
    // and no witness was found), and only as far as a new answer can come: past one cycle beyond
    // the kept lengths every length repeats one already asked
    long long a = lo > 0 ? lo : 0, top = a + re->lv_n + re->lv_period, r = -1;
    if (hi >= 0 && hi < top) top = hi;
    for (long long l = a; l <= top && r < 0; l++) if (jsre_len_ok(re, cur, (int)(l > INT_MAX ? INT_MAX : l), (int)(l > INT_MAX ? INT_MAX : l))) r = l;
    static const char order[] = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-_. ";
    int n = 0;
    while (r >= 0) {
        if (r == 0 && jsre_accepts(re, cur)) { free(cur); free(nx); return n; }
        int took = -1;
        for (int k = 0; k < 256 + (int)sizeof order - 1 && took < 0; k++) {
            int x = k < (int)sizeof order - 1 ? (unsigned char)order[k] : k - ((int)sizeof order - 1);
            if (x == 0) continue;
            memcpy(nx, cur, w * sizeof *nx);
            if (!jsre_step(re, nx, (uint8_t)x)) continue;
            int c = x < 0x80 || x >= 0xC0;
            if (r - c < 0 || !jsre_len_ok(re, nx, (int)(r - c), (int)(r - c))) continue;
            took = x;
            r -= c;
            memcpy(cur, nx, w * sizeof *cur);
        }
        if (took < 0) break;
        if (n >= cap) { free(cur); free(nx); return n; }   // as much as fits: the caller stops short
        out[n++] = (char)took;
    }
    free(cur); free(nx);
    return -1;
}

int jsre_states(const jsre *re) { return re->n; }

size_t jsre_bytes(const jsre *re) {
    return sizeof *re + (size_t)re->cap * sizeof *re->s + 2 * re->words * sizeof(uint64_t) +
           (size_t)re->lv_n * re->words * sizeof(uint64_t);
}
size_t jsre_words(const jsre *re) { return re->words; }

void jsre_start(const jsre *re, uint64_t *set) {
    int stack[JSRE_MAX_STATES * 2 + 4];
    memset(set, 0, re->words * sizeof *set);
    closure(re, set, re->start, true, stack);
    for (size_t w = 0; w < re->words; w++) set[w] &= re->live[w];
}

bool jsre_step(const jsre *re, uint64_t *set, uint8_t b) {
    uint64_t next[(JSRE_MAX_STATES + 63) / 64];
    int stack[JSRE_MAX_STATES * 2 + 4];
    memset(next, 0, re->words * sizeof *next);
    for (size_t w = 0; w < re->words; w++) {
        uint64_t m = set[w];
        while (m) {
            int k = (int)(w * 64) + __builtin_ctzll(m);
            m &= m - 1;
            const state *s = &re->s[k];
            if (s->kind == S_BYTE && b >= s->lo && b <= s->hi) closure(re, next, s->out, false, stack);
        }
    }
    bool any = false;
    for (size_t w = 0; w < re->words; w++) { set[w] = next[w] & re->live[w]; any |= set[w] != 0; }
    return any;
}

bool jsre_accepts(const jsre *re, const uint64_t *set) {
    for (size_t w = 0; w < re->words; w++) {
        uint64_t m = set[w];
        while (m) {
            int k = (int)(w * 64) + __builtin_ctzll(m);
            m &= m - 1;
            if (re->s[k].kind == S_MATCH || (re->s[k].kind == S_EOL && bit(re->end_ok, k))) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------- lazy DFA
//
// One DFA state per NFA state set reached, interned by the set; its transition
// table (one entry per byte class) is filled on first use. Every entry is a
// pure function of the set and the byte, so the cache can only change speed:
// a flush that drops it, or a budget too small to hold more than one table,
// gives the same answers (R2.18, cache policy).
//
// A state is KEPT once a validator that persists has passed through it
// (jsre_dfa_step with keep set: sval_feed). Kept states hold their slot for
// the pool's life, because copies of a fed validator can sit anywhere (the
// engine's own, a probe's copy, a replay). Every other state was reached only
// inside a trial, whose scratch validator is the one copy holding it and is
// discarded when the trial returns; those are the cache a flush may drop.
//
// A handle is tag << 32 | slot. A kept state's tag is KEPT_TAG; any other
// handle carries the pool's epoch, which every flush bumps, and slots are
// freed only by a flush. So a handle whose tag is neither KEPT_TAG nor the
// epoch names a state a flush may have dropped: it is refused (JSRE_FAIL),
// never read as whatever the slot holds now, and the check reads no slot.

enum { KEPT_TAG = 0xFFFFFFFEu };

typedef struct {
    uint64_t *set;       // NULL: a free slot
    int32_t  *next;      // per byte class: a slot, -1 dead, -2 not built; NULL: no table
    uint64_t *lf;        // lengths: bit l when the set meets lv[l]; NULL: not built
    int32_t   lmin;      // the shortest accepting completion, -1 none (with lf)
    bool      acc, kept;
} dstate;

struct jsre_dfa {
    const jsre *re;
    jsre_pool  *pool;
    dstate     *st;
    int         n, cap, live;
    uint32_t    epoch;   // the pool's, copied here at each flush
    int32_t    *freel;   // slots a flush freed
    int         nfree, freecap;
    int32_t    *table;   // open addressing over slots, by the set's hash
    size_t      tcap;
    const char *label;   // the caller's name for it, for a failure's message
    uint8_t     cls[256];
};

struct jsre_pool {
    size_t      budget;          // bytes of cache and books before a flush
    size_t      cache, kept;     // bytes now: tables, length bits and unkept sets; kept sets
    size_t      books;           // bytes now: the pool, its DFAs and their slot
                                 //   arrays, hash tables and free lists, by capacity
    size_t      high;            // the most cache + kept + books ever held
    uint64_t    flushes;
    uint32_t    epoch;
    int         kept_states;
    int         kept_max;        // JSRE_DFA_MAX, or lower (jsre_pool_set_kept_max)
    jsre_dfa  **dfa;
    int         ndfa, dcap;
    const void *owner;           // the filling thread's mark (pool_owner_ok)
    const jsre_dfa *failed_dfa;  // the DFA whose step failed (failed's subject)
    const char *failed;
};

// What the pool charges is what it asks the allocator for, block by block:
// a set, a table, a state's length bits (the cache, or `kept` for a kept
// set), and the books: the pool itself, each DFA, and each DFA's slot array,
// hash table and free list at their CAPACITY (R2.18, codex on #875: these
// were left out, and grew past the budget across a pool's DFAs). The budget
// holds cache + books; a flush frees the cache and shrinks the books to what
// still lives. Kept sets are bounded apart, by the kept limit.
static size_t set_cost(const jsre *re) { return re->words * sizeof(uint64_t); }
static size_t table_cost(const jsre *re) { return (size_t)re->ncls * sizeof(int32_t); }
static size_t lf_nwords(const jsre *re) { size_t nw = ((size_t)re->lv_n + 63) / 64; return nw ? nw : 1; }
static size_t lf_cost(const jsre *re) { return lf_nwords(re) * sizeof(uint64_t); }
static void books(jsre_pool *p, size_t add, size_t sub) { p->books = p->books + add - sub; }

static uint64_t set_hash(const uint64_t *s, size_t w) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < w; i++) { h ^= s[i]; h *= 0x100000001B3ull; h ^= h >> 29; }
    return h;
}

static jsre_st pool_fail(jsre_pool *p, const jsre_dfa *d, const char *why) {
    if (!p->failed) { p->failed = why; p->failed_dfa = d; }
    return JSRE_FAIL;
}

static void note_high(jsre_pool *p) {
    size_t held = p->cache + p->kept + p->books;
    if (held > p->high) p->high = held;
}

jsre_pool *jsre_pool_new(size_t budget) {
    jsre_pool *p = calloc(1, sizeof *p);
    if (p) { p->budget = budget; p->epoch = 1; p->kept_max = JSRE_DFA_MAX; p->books = sizeof *p; note_high(p); }
    return p;
}

void jsre_pool_set_kept_max(jsre_pool *p, int kept_max) {
    if (kept_max > 0 && kept_max < JSRE_DFA_MAX) p->kept_max = kept_max;
}

void jsre_pool_free(jsre_pool *p) {
    if (!p) return;
    free(p->dfa);
    free(p);
}

void jsre_pool_stats(const jsre_pool *p, jsre_pool_stat *s) {
    s->budget = p->budget;
    s->cache = p->cache; s->kept = p->kept; s->books = p->books; s->high = p->high;
    s->flushes = p->flushes; s->kept_states = p->kept_states; s->failed = p->failed;
    s->failed_label = p->failed_dfa ? p->failed_dfa->label : NULL;
    s->failed_states = p->failed_dfa ? p->failed_dfa->live : 0;
}

void jsre_dfa_label(jsre_dfa *d, const char *label) { d->label = label; }

static void hash_insert(jsre_dfa *d, int id) {
    size_t i = set_hash(d->st[id].set, d->re->words) & (d->tcap - 1);
    while (d->table[i] >= 0) i = (i + 1) & (d->tcap - 1);
    d->table[i] = id;
}

static bool hash_rebuild(jsre_dfa *d, size_t ncap) {
    int32_t *nt = malloc(ncap * sizeof *nt);
    if (!nt) return false;
    for (size_t k = 0; k < ncap; k++) nt[k] = -1;
    // both tables live for a moment: the peak counts them both
    books(d->pool, ncap * sizeof *nt, 0);
    note_high(d->pool);
    free(d->table);
    books(d->pool, 0, d->tcap * sizeof *nt);
    d->table = nt; d->tcap = ncap;
    for (int k = 0; k < d->n; k++)
        if (d->st[k].set) hash_insert(d, k);
    return true;
}

// After a flush, a DFA's books shrink to what still lives: the slot array to
// the highest slot holding a state, the free list to the slots free below it
// (the lowest on top, so new states fill from the bottom and the array stays
// short), the hash table to twice the live states. A slot that still lives
// keeps its number, so every handle a flush keeps stays good; a failed shrink
// keeps the larger block, charged as it is.
static bool dfa_shrink(jsre_pool *p, jsre_dfa *d) {
    int top = 0, nf = 0;
    for (int i = 0; i < d->n; i++) if (d->st[i].set) top = i + 1;
    for (int i = 0; i < top; i++) nf += !d->st[i].set;
    d->n = top;
    int fcap = nf > 16 ? nf : 16;
    if (fcap != d->freecap) {
        int32_t *g = realloc(d->freel, sizeof *g * (size_t)fcap);
        if (g || !fcap) {
            books(p, sizeof *g * (size_t)fcap, sizeof *g * (size_t)d->freecap);
            d->freel = g; d->freecap = fcap;
        } else if (d->freecap < nf) { d->nfree = 0; return false; }
    }
    d->nfree = 0;
    for (int i = top - 1; i >= 0; i--) if (!d->st[i].set) d->freel[d->nfree++] = i;
    int scap = 64;
    while (scap < top) scap *= 2;
    if (scap < d->cap) {
        dstate *g = realloc(d->st, sizeof *g * (size_t)scap);
        if (g) { books(p, sizeof *g * (size_t)scap, sizeof *g * (size_t)d->cap); d->st = g; d->cap = scap; }
    }
    size_t tc = 256;
    while (tc < (size_t)d->live * 2 + 2) tc *= 2;
    return hash_rebuild(d, tc);
}

// Drop the cache: every table, every state's length bits, and every unkept
// set but `keep_id`'s in `keep_dfa` (the state being stepped from); then
// shrink each DFA's books. Kept states keep their slots and sets; `keep_id`'s
// table stays allocated with its entries unbuilt, since they may name slots
// this frees.
static void pool_flush(jsre_pool *p, jsre_dfa *keep_dfa, int keep_id) {
    size_t tb = 0;
    p->flushes++;
    p->epoch = p->epoch + 1 >= KEPT_TAG ? 1 : p->epoch + 1;   // 0 never: a zeroed handle is never good
    for (int k = 0; k < p->ndfa; k++) {
        jsre_dfa *d = p->dfa[k];
        size_t tbytes = table_cost(d->re), sbytes = set_cost(d->re);
        d->epoch = p->epoch;
        for (int i = 0; i < d->n; i++) {
            dstate *s = &d->st[i];
            if (!s->set) continue;
            bool cur = d == keep_dfa && i == keep_id;
            free(s->lf); s->lf = NULL;   // a pure cache, rebuilt on demand
            if (s->next) {
                if (cur) { for (int c = 0; c < d->re->ncls; c++) s->next[c] = -2; tb += tbytes; }
                else { free(s->next); s->next = NULL; }
            }
            if (cur && !s->kept) tb += sbytes;
            if (s->kept || cur) continue;
            free(s->set); s->set = NULL; d->live--;
        }
        // a rebuild that cannot allocate leaves a table naming free slots:
        // linear probing cannot drop entries in place without breaking
        // chains, so the pool fails loudly instead of answering from it
        if (!dfa_shrink(p, d)) pool_fail(p, d, "out of memory in a pattern automaton");
    }
    p->cache = tb;
}

// Room for `need` more bytes; a flush when cache and books would pass the
// budget. A step always gets the room it needs after that flush, so a budget
// of one byte still steps, one table at a time.
static void pool_room(jsre_pool *p, jsre_dfa *d, int cur, size_t need) {
    if (p->cache + p->books + need > p->budget && p->cache > 0) pool_flush(p, d, cur);
}

// A thread is named by the address of its copy of a thread-local: distinct
// per live thread, and no thread library to link.
static _Thread_local char thread_mark;

static bool pool_owner_ok(jsre_pool *p) {
    if (!p->owner) { p->owner = &thread_mark; return true; }
    return p->owner == (const void *)&thread_mark;
}

jsre_dfa *jsre_dfa_new(const jsre *re, jsre_pool *pool) {
    if (!re || !pool) return NULL;
    // making a DFA writes the pool too: the same thread as every fill
    if (!pool_owner_ok(pool)) { pool_fail(pool, NULL, "pattern automaton filled from a second thread"); return NULL; }
    jsre_dfa *d = calloc(1, sizeof *d);
    if (!d) return NULL;
    d->re = re; d->pool = pool; d->epoch = pool->epoch;
    memcpy(d->cls, re->cls, sizeof d->cls);
    d->tcap = 256;
    d->table = malloc(d->tcap * sizeof *d->table);
    if (pool->ndfa == pool->dcap) {
        int cap = pool->dcap ? pool->dcap * 2 : 8;
        jsre_dfa **g = realloc(pool->dfa, sizeof *g * (size_t)cap);
        if (g) { books(pool, sizeof *g * (size_t)cap, sizeof *g * (size_t)pool->dcap); pool->dfa = g; pool->dcap = cap; }
    }
    if (!d->table || pool->ndfa == pool->dcap) { free(d->table); free(d); return NULL; }
    for (size_t i = 0; i < d->tcap; i++) d->table[i] = -1;
    books(pool, sizeof *d + d->tcap * sizeof *d->table, 0);
    note_high(pool);
    pool->dfa[pool->ndfa++] = d;
    return d;
}

void jsre_dfa_free(jsre_dfa *d) {
    if (!d) return;
    jsre_pool *p = d->pool;
    size_t tbytes = table_cost(d->re), sbytes = set_cost(d->re);
    for (int i = 0; i < d->n; i++) {
        dstate *s = &d->st[i];
        if (s->next) p->cache -= tbytes;
        if (s->lf) p->cache -= lf_cost(d->re);
        free(s->lf);
        if (s->set) { if (s->kept) { p->kept -= sbytes; p->kept_states--; } else p->cache -= sbytes; }
        free(s->set); free(s->next);
    }
    for (int k = 0; k < p->ndfa; k++)
        if (p->dfa[k] == d) { p->dfa[k] = p->dfa[--p->ndfa]; break; }
    if (p->failed_dfa == d) p->failed_dfa = NULL;
    books(p, 0, sizeof *d + d->tcap * sizeof *d->table + (size_t)d->cap * sizeof *d->st +
                (size_t)d->freecap * sizeof *d->freel);
    free(d->st); free(d->freel); free(d->table); free(d);
}

// Keep a state: the one place a state becomes kept, so the one place the
// kept limit is held. A trial usually reached the state first (unkept) and the
// feed that follows only promotes it; the limit counts promotions too.
static __attribute__((noinline)) jsre_st keep_handle(jsre_dfa *d, int id) {
    dstate *s = &d->st[id];
    if (!s->kept) {
        // a promotion writes the pool: a cached transition taken from a
        // second thread is refused here too (R2.18, codex on #875)
        if (!pool_owner_ok(d->pool))
            return pool_fail(d->pool, d, "pattern automaton filled from a second thread");
        if (d->pool->kept_states >= d->pool->kept_max)
            return pool_fail(d->pool, d, "more states than one request may keep");
        size_t sbytes = set_cost(d->re);
        s->kept = true;
        d->pool->cache -= sbytes; d->pool->kept += sbytes; d->pool->kept_states++;
    }
    return (jsre_st)KEPT_TAG << 32 | (uint32_t)id;
}

// The slot of `set`, interned if new (a kept state when `keep`). -1 with the
// pool's reason when the pool refuses.
static int intern(jsre_dfa *d, const uint64_t *set, bool keep, int cur) {
    jsre_pool *p = d->pool;
    size_t w = d->re->words, sbytes = set_cost(d->re), bytes = w * sizeof(uint64_t);
    uint64_t h = set_hash(set, w);
    for (size_t i = h & (d->tcap - 1);; i = (i + 1) & (d->tcap - 1)) {
        int id = d->table[i];
        if (id < 0) break;
        if (!memcmp(d->st[id].set, set, bytes)) return id;
    }
    if (!keep) {
        // the set, and the books this state may grow: a doubled slot array, a
        // doubled hash table (built beside the old one)
        size_t need = sbytes;
        if (!d->nfree && d->n == d->cap) need += sizeof(dstate) * (size_t)(d->cap ? d->cap : 64);
        if ((size_t)(d->live + 1) * 2 > d->tcap) need += d->tcap * 2 * sizeof(int32_t);
        pool_room(p, d, cur, need);
    }
    int id;
    if (d->nfree) id = d->freel[--d->nfree];
    else {
        if (d->n == d->cap) {
            int cap = d->cap ? d->cap * 2 : 64;
            dstate *g = realloc(d->st, sizeof *g * (size_t)cap);
            if (!g) { pool_fail(p, d, "out of memory in a pattern automaton"); return -1; }
            books(p, sizeof *g * (size_t)cap, sizeof *g * (size_t)d->cap);
            d->st = g; d->cap = cap;
        }
        id = d->n++;
        d->st[id] = (dstate){ 0 };
    }
    dstate *s = &d->st[id];
    s->set = malloc(bytes);
    if (!s->set) { pool_fail(p, d, "out of memory in a pattern automaton"); return -1; }
    memcpy(s->set, set, bytes);
    s->next = NULL;
    s->lf = NULL;
    s->acc = jsre_accepts(d->re, set);
    s->kept = false;
    p->cache += sbytes;
    d->live++;
    if ((size_t)d->live * 2 > d->tcap) {
        if (!hash_rebuild(d, d->tcap * 2)) { pool_fail(p, d, "out of memory in a pattern automaton"); return -1; }
    } else hash_insert(d, id);
    note_high(p);
    return id;
}

jsre_st jsre_dfa_start(jsre_dfa *d) {
    if (!pool_owner_ok(d->pool)) return pool_fail(d->pool, d, "pattern automaton filled from a second thread");
    uint64_t set[(JSRE_MAX_STATES + 63) / 64];
    jsre_start(d->re, set);
    int id = intern(d, set, false, -1);
    return id < 0 ? JSRE_FAIL : keep_handle(d, id);
}

static bool handle_ok(const jsre_dfa *d, jsre_st st) {
    uint32_t tag = (uint32_t)(st >> 32);
    return (tag == KEPT_TAG || tag == d->epoch) && (uint32_t)st < (uint32_t)d->n;
}

static __attribute__((noinline)) jsre_st step_refused(jsre_dfa *d, jsre_st st) {
    if (st == JSRE_DEAD || st == JSRE_FAIL) return st;
    return pool_fail(d->pool, d, "pattern automaton: a stale state handle (a trial's state outlived its trial)");
}

// A fill: the one place the cache changes. Out of line, so the cached step
// above it needs no stack frame.
static __attribute__((noinline)) jsre_st step_fill(jsre_dfa *d, uint32_t cur, int c, bool keep) {
    jsre_pool *p = d->pool;
    if (!pool_owner_ok(p)) return pool_fail(p, d, "pattern automaton filled from a second thread");
    size_t tbytes = table_cost(d->re);
    if (!d->st[cur].next) {
        pool_room(p, d, (int)cur, tbytes);
        int32_t *t = malloc((size_t)d->re->ncls * sizeof(int32_t));
        if (!t) return pool_fail(p, d, "out of memory in a pattern automaton");
        for (int k = 0; k < d->re->ncls; k++) t[k] = -2;
        d->st[cur].next = t;
        p->cache += tbytes;
        note_high(p);
    }
    uint64_t set[(JSRE_MAX_STATES + 63) / 64];
    memcpy(set, d->st[cur].set, d->re->words * sizeof *set);
    if (!jsre_step(d->re, set, d->re->rep[c])) { d->st[cur].next[c] = -1; return JSRE_DEAD; }
    int id = intern(d, set, keep, (int)cur);   // may flush: cur's slot and table survive it
    if (id < 0) return JSRE_FAIL;
    d->st[cur].next[c] = id;
    return keep ? keep_handle(d, id) : (jsre_st)d->epoch << 32 | (uint32_t)id;
}

jsre_st jsre_dfa_step(jsre_dfa *d, jsre_st st, uint8_t b, bool keep) {
    uint32_t tag = (uint32_t)(st >> 32), cur = (uint32_t)st;
    // n never shrinks, so a tag that is good leaves the slot in range
    if (__builtin_expect(tag != KEPT_TAG && tag != d->epoch, 0)) return step_refused(d, st);
    const int32_t *next = d->st[cur].next;
    int c = d->cls[b];
    int32_t nx = next ? next[c] : -2;
    if (__builtin_expect(nx >= 0 && !keep, 1)) return (jsre_st)d->epoch << 32 | (uint32_t)nx;
    if (nx == -1) return JSRE_DEAD;
    if (nx >= 0) return keep_handle(d, nx);
    return step_fill(d, cur, c, keep);
}

bool jsre_dfa_accepts(const jsre_dfa *d, jsre_st st) {
    return handle_ok(d, st) && d->st[(uint32_t)st].acc;
}

const uint64_t *jsre_dfa_set(const jsre_dfa *d, jsre_st st) {
    return handle_ok(d, st) ? d->st[(uint32_t)st].set : NULL;
}

bool jsre_dfa_kept(const jsre_dfa *d, jsre_st st) {
    return handle_ok(d, st) && d->st[(uint32_t)st].kept;
}

int jsre_dfa_states(const jsre_dfa *d) { return d->live; }

// Cache the state's lengths if the budget has room. Never a flush: the caller
// keeps the handle it asked about, and a flush would leave a trial's handle
// stale. Without room the ask is answered from the set itself.
static __attribute__((noinline)) bool len_fill(jsre_dfa *d, uint32_t id) {
    jsre_pool *p = d->pool;
    const jsre *re = d->re;
    if (!pool_owner_ok(p)) { pool_fail(p, d, "pattern automaton filled from a second thread"); return false; }
    if (p->cache + p->books + lf_cost(re) > p->budget) return false;
    uint64_t *lf = calloc(lf_nwords(re), sizeof *lf);
    if (!lf) return false;   // uncached, answered from the set
    dstate *s = &d->st[id];
    s->lmin = -1;
    for (int l = 0; l < re->lv_n; l++)
        if (meets(s->set, re->lv + (size_t)l * re->words, re->words)) {
            lf[l >> 6] |= 1ull << (l & 63);
            if (s->lmin < 0) s->lmin = l;
        }
    s->lf = lf;
    p->cache += lf_cost(re);
    note_high(p);
    return true;
}

// The first length in [from, to] the state's lengths hold, or -1.
static int lf_first(const uint64_t *lf, int from, int to) {
    for (int l = from; l <= to; ) {
        uint64_t m = lf[l >> 6] >> (l & 63);
        if (m) { int r = l + __builtin_ctzll(m); return r <= to ? r : -1; }
        l = (l | 63) + 1;
    }
    return -1;
}

// jsre_len_ok for a DFA state, from a cached bit per length: the shortest
// completion answers most asks at once (it is the least length there is).
bool jsre_dfa_len_ok(jsre_dfa *d, jsre_st st, int lo, int hi) {
    if (!handle_ok(d, st)) { (void)step_refused(d, st); return false; }
    const jsre *re = d->re;
    int a = lo > 0 ? lo : 0;
    if (!re->lv) return hi < 0 || hi >= a;
    if (hi >= 0 && hi < a) return false;
    uint32_t id = (uint32_t)st;
    if (!d->st[id].lf && !len_fill(d, id)) return !d->pool->failed && jsre_len_ok(re, d->st[id].set, lo, hi);
    const dstate *s = &d->st[id];
    if (s->lmin < 0 || (hi >= 0 && s->lmin > hi)) return false;
    if (s->lmin >= a) return true;
    int top = re->lv_n - 1;
    if (lf_first(s->lf, a, hi >= 0 && hi < top ? hi : top) >= 0) return true;
    if (!re->lv_period) return false;
    // past the kept lengths: the cycle, at most once round
    long long from = a > re->lv_n ? a : re->lv_n;
    for (long long l = from; l < from + re->lv_period && (hi < 0 || l <= hi); l++) {
        int ix = re->lv_tail + (int)((l - re->lv_tail) % re->lv_period);
        if (s->lf[ix >> 6] >> (ix & 63) & 1) return true;
    }
    return false;
}

int jsre_classes(const jsre *re) { return re->ncls; }
