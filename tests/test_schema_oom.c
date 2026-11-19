// Allocation-failure tests for the JSON-Schema compiler.
//
// schema_compile is driven by response_format.json_schema out of an untrusted
// HTTP request body: the property counts, enum sizes, key strings and nesting
// depth are all attacker-chosen, and each of them sizes an allocation. Runner
// deliberately operates near memory limits, so a failed allocation is a normal
// condition — an unchecked one here is a remote crash.
//
// schema.c, schema_tools.c and json.c are compiled straight into this test
// with their allocators macro-substituted, the same technique as
// test_json_oom.c. schema_tools.c holds the native tool grammars the second
// sweep drives. json.c has to come along because the compiler serialises
// enum/const literals through jv_dump/sbuf, so a failure inside the builder
// is a schema.c failure path too.
// jsonmode.c is linked normally: it never allocates, and only the validator
// (not the compiler under test here) calls into it.
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long alloc_calls;   // allocations attempted since the last reset
static long alloc_live;    // outstanding blocks; must return to its baseline
static long fail_at = -1;  // index of the allocation to fail, -1 for none
// and the bytes asked for, now and at the most, from a size header on every
// block (R2.18: the per-request cap on compiled automata is checked in bytes)
static size_t bytes_live, bytes_high;

static void *t_malloc(size_t n) {
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    size_t *p = malloc(n + 16);
    if (!p) return NULL;
    *p = n;
    alloc_live++;
    bytes_live += n;
    if (bytes_live > bytes_high) bytes_high = bytes_live;
    return (char *)p + 16;
}

static void *t_calloc(size_t a, size_t b) {
    if (b && a > ((size_t)-1 - 16) / b) return NULL;
    void *p = t_malloc(a * b);
    if (p) memset(p, 0, a * b);
    return p;
}

static void *t_realloc(void *p, size_t n) {
    if (!p) return t_malloc(n);   // realloc(NULL, n) behaves as malloc
    if (fail_at >= 0 && alloc_calls++ == fail_at) return NULL;
    if (fail_at < 0) alloc_calls++;
    size_t *h = (size_t *)((char *)p - 16), old = *h;
    size_t *q = realloc(h, n + 16);
    if (!q) return NULL;
    *q = n;
    bytes_live = bytes_live - old + n;
    if (bytes_live > bytes_high) bytes_high = bytes_live;
    return (char *)q + 16;
}

// strdup must be instrumented too, not merely for the failure injection: its
// blocks are released through schema_free's (substituted) free, so leaving it
// alone would unbalance the counter and hide real leaks. strndup is the same
// story -- compile_ascii_pattern owns its segment prefixes through it.
static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static char *t_strndup(const char *s, size_t cap) {
    size_t n = strnlen(s, cap);
    char *p = t_malloc(n + 1);
    if (p) { memcpy(p, s, n); p[n] = 0; }
    return p;
}

static void t_free(void *p) {
    if (!p) return;
    size_t *h = (size_t *)((char *)p - 16);
    alloc_live--;
    bytes_live -= *h;
    free(h);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define strdup  t_strdup
#define strndup t_strndup
#define free    t_free
#include "../src/json.c"
#include "../src/schema.c"
#include "../src/schema_tools.c"
#include "../src/jsre.c"     // R2.18: the pattern automaton allocates under the same injection
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef strndup
#undef free

// One schema per compile path: plain object with required keys, arrays with
// bounds, enum and const literals (which run through jv_dump), a scalar-const
// oneOf, a first-byte-dispatched union, a `type` array union, the discriminated
// tool/args shape compile_discriminated_action recognises, the general
// const-discriminated union (R2.8), and deep nesting.
static const char *const schemas[] = {
    "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"},"
        "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"]}",
    "{\"type\":\"array\",\"items\":{\"type\":\"string\",\"minLength\":2,"
        "\"maxLength\":8},\"minItems\":1,\"maxItems\":4}",
    "{\"type\":\"string\",\"enum\":[\"alpha\",\"beta\",\"gamma\"]}",
    "{\"const\":\"fixed\"}",
    "{\"oneOf\":[{\"const\":\"x\"},{\"const\":\"y\"},{\"const\":3}]}",
    "{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"},{\"type\":\"null\"}]}",
    "{\"type\":[\"string\",\"null\"]}",
    "{\"oneOf\":["
        "{\"properties\":{\"tool\":{\"const\":\"read\"},"
            "\"args\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}},"
         "\"required\":[\"tool\",\"args\"]},"
        "{\"properties\":{\"tool\":{\"const\":\"write\"},"
            "\"args\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}},"
         "\"required\":[\"tool\",\"args\"]}]}",
    // R2.8: the const-discriminated union merge_discriminated builds, with
    // a property only one branch has and a null beside the objects
    "{\"anyOf\":["
        "{\"properties\":{\"op\":{\"const\":\"MAP\"},\"fn\":{\"enum\":[\"INC\",\"DEC\"]},"
            "\"args\":{\"type\":\"array\",\"minItems\":1,\"items\":{\"type\":\"string\"}}},"
         "\"required\":[\"op\",\"fn\",\"args\"]},"
        "{\"type\":\"null\"},"
        "{\"properties\":{\"op\":{\"const\":\"HEAD\"},"
            "\"args\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},"
         "\"required\":[\"op\",\"args\"]}],"
     "\"discriminator\":{\"propertyName\":\"op\"}}",
    "{\"type\":\"object\",\"properties\":{\"deep\":{\"type\":\"object\","
        "\"properties\":{\"deeper\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"object\",\"properties\":{\"leaf\":{\"enum\":[1,2,3]}}}}}}}}",
    // R2.18: patterns the segments cannot take, so the automaton compiles
    // them (jsre_compile, jsre_lengths, the source copy), with bounds
    "{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"string\","
        "\"pattern\":\"^(ab|cd){1,4}-[0-9]{2}$\",\"minLength\":4}},\"required\":[\"id\"]}",
    "{\"type\":\"string\",\"pattern\":\"[a-z]+@[a-z]+\\\\.(com|org)\",\"maxLength\":40}",
};

// R2.18: the automaton allocates while it validates too (its DFA states, the
// tables, the length caches, the pool), so every allocation a constrained
// feed makes is failed in turn. A failed one refuses the byte and records why
// (sval_pattern_fault), never crashes, and schema_free leaves nothing behind.

// R2.18 (codex on #875): a temporary code-point set (cp_norm, cp_not,
// cp_producible) left its block behind when an allocation part-way through
// failed. Multi-range Unicode classes build several of them; every one of
// their allocations is failed once, and nothing may be stranded.

// R2.18: the bytes one request's compiled pattern automata may hold (a
// schema, or a request's tools together), 64 MiB
// (JSRE_SCHEMA_BYTES), counted as the schema compiles and refused by name past
// it, so a request's schema cannot make the server hold more however many
// patterns it carries. Each property here is ^(a|b){0,1000}$ with maxLength
// 1000, about half a MiB of automaton (4,083 states, 1,001 lengths): a schema of
// as many as fit compiles, one more is refused, and the heap, counted by the
// allocator, never holds much past the cap while either compiles.
// k string properties in groups of 50 (an object holds at most 60), one
// group an object of the outer one
static char *cap_schema(int k) {
    size_t cap = (size_t)k * 96 + (size_t)(k / 50 + 1) * 64 + 64;
    char *s = malloc(cap);
    int n = snprintf(s, cap, "{\"type\":\"object\",\"properties\":{");
    for (int g = 0; g * 50 < k; g++) {
        n += snprintf(s + n, cap - (size_t)n, "%s\"g%d\":{\"type\":\"object\",\"properties\":{", g ? "," : "", g);
        for (int i = g * 50; i < k && i < g * 50 + 50; i++)
            n += snprintf(s + n, cap - (size_t)n, "%s\"p%d\":{\"type\":\"string\",\"pattern\":\"^(a|b){0,1000}$\",\"maxLength\":1000}",
                          i > g * 50 ? "," : "", i);
        n += snprintf(s + n, cap - (size_t)n, "}}");
    }
    snprintf(s + n, cap - (size_t)n, "}}");
    return s;
}

static void test_schema_automata_cap(void) {
    char err[256];
    fail_at = -1;
    // one pattern's automaton, as jsre counts it
    char *one = cap_schema(1);
    jv *j = json_parse(one, strlen(one));
    snode *n = schema_compile(j, err, sizeof(err));
    assert(n && n->n_props == 1 && n->props[0]->n_props == 1 && n->props[0]->props[0]->re);
    size_t b = jsre_bytes(n->props[0]->props[0]->re);
    schema_free(n); jv_free(j); free(one);
    assert(b > 256 * 1024 && b < 4 << 20);
    int fit = (int)(JSRE_SCHEMA_BYTES / b);
    // as many as fit: compiles, and holds about what it counts
    char *under = cap_schema(fit);
    j = json_parse(under, strlen(under));
    size_t base = bytes_live;
    bytes_high = bytes_live;
    n = schema_compile(j, err, sizeof(err));
    if (!n) fprintf(stderr, "automata cap: %d patterns refused: %s\n", fit, err);
    assert(n != NULL);
    size_t held = bytes_live - base;
    assert(held >= (size_t)fit * b && held <= JSRE_SCHEMA_BYTES + (size_t)fit * 4096);
    schema_free(n);
    jv_free(j); free(under);
    // one more: refused by name, the compile stopping near the cap
    char *over = cap_schema(fit + 1);
    j = json_parse(over, strlen(over));
    base = bytes_live;
    bytes_high = bytes_live;
    n = schema_compile(j, err, sizeof(err));
    assert(n == NULL && strstr(err, "pattern automata") && strstr(err, "64 MiB"));
    assert(bytes_high - base <= JSRE_SCHEMA_BYTES + 2 * b + (size_t)(fit + 1) * 4096);
    assert(bytes_live == base);   // the refused compile left nothing behind
    jv_free(j); free(over);
    printf("ok: the compiled automata cap: %d patterns of %zu bytes compile, %d are refused by name\n", fit, b, fit + 1);
}

static void test_pattern_classes_survive_allocation_failure(void) {
    const char *pats[] = { "\\s", "\\S+", "[^\\s\\d]{2}", "[a-z\\u00e9\\u4e00-\\u9fff]{2,3}",
                           "(\\w|\\s)*\\u00e9", ".[^a-c\\u2028]", "[\\s\\S]x" };
    for (size_t i = 0; i < sizeof(pats) / sizeof(*pats); i++) {
        char err[256];
        fail_at = -1; alloc_calls = 0; alloc_live = 0;
        jsre *re = jsre_compile(pats[i], err, sizeof(err));
        assert(re && jsre_lengths(re, -1, err, sizeof(err)));
        long total = alloc_calls;
        jsre_free(re);
        assert(alloc_live == 0 && total > 3);
        for (long k = 0; k < total; k++) {
            fail_at = k; alloc_calls = 0; err[0] = 0;
            jsre *r = jsre_compile(pats[i], err, sizeof(err));
            if (r && !jsre_lengths(r, -1, err, sizeof(err))) assert(err[0]);
            if (!r) assert(err[0]);
            jsre_free(r);
            if (alloc_live != 0) {
                fprintf(stderr, "pattern %s, failing allocation %ld of %ld: %ld block(s) leaked\n",
                        pats[i], k, total, alloc_live);
                abort();
            }
        }
        fail_at = -1;
    }
    printf("ok: pattern classes survive each of their allocations failing\n");
}

static void test_pattern_feed_survives_allocation_failure(void) {
    const char *src = "{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"string\","
                      "\"pattern\":\"^(ab|cd){1,4}-[0-9]{2}$\",\"minLength\":4}},\"required\":[\"id\"]}";
    const char *doc = "{\"id\":\"abcd-42\"}";
    fail_at = -1;
    alloc_calls = 0;
    alloc_live = 0;
    jv *j = json_parse(src, strlen(src));
    long base_live = alloc_live;
    char err[256];
    snode *n = schema_compile(j, err, sizeof(err));
    assert(n);
    alloc_calls = 0;
    sval v; sval_init(&v, n);
    assert(sval_feed(&v, doc, (int)strlen(doc)) && v.done);
    long total = alloc_calls;
    assert(total > 0);                     // the feed did allocate: the sweep is not empty
    schema_free(n);
    assert(alloc_live == base_live);
    int refused = 0;
    for (long k = 0; k < total; k++) {
        fail_at = -1;
        n = schema_compile(j, err, sizeof(err));
        assert(n);
        alloc_calls = 0;
        fail_at = k;
        sval w; sval_init(&w, n);
        bool ok = sval_feed(&w, doc, (int)strlen(doc)) && w.done;
        fail_at = -1;
        char msg[400];
        if (!ok) { refused++; assert(sval_pattern_fault(&w, msg, sizeof(msg)) && strstr(msg, "out of memory")); }
        schema_free(n);
        if (alloc_live != base_live) {
            fprintf(stderr, "pattern feed, failing allocation %ld of %ld: %ld block(s) leaked\n",
                    k, total, alloc_live - base_live);
            abort();
        }
    }
    assert(refused > 0);
    jv_free(j);
    assert(alloc_live == 0);
    printf("ok: a pattern feed survives each of its %ld allocations failing (%d refused, named)\n", total, refused);
}

// Every allocation on the compile path is failed in turn. schema_compile may
// succeed or return NULL, but it must never crash and never strand a block —
// and when it does fail it must say so, because both callers (main.c and
// server.c) print that message and would otherwise report an empty reason.
static void test_compile_survives_allocation_failure(void) {
    for (size_t i = 0; i < sizeof(schemas) / sizeof(*schemas); i++) {
        const char *s = schemas[i];

        fail_at = -1;
        alloc_calls = 0;
        alloc_live = 0;
        jv *doc = json_parse(s, strlen(s));
        assert(doc != NULL);              // the schema text itself must be valid
        long base_live = alloc_live;      // the parsed document stays live

        char err[256];
        alloc_calls = 0;
        snode *n = schema_compile(doc, err, sizeof(err));
        if (!n) {
            fprintf(stderr, "schema %zu does not compile cleanly: %s\n", i, err);
            abort();
        }
        long total = alloc_calls;
        assert(total > 0);
        schema_free(n);
        assert(alloc_live == base_live);  // a clean compile must not leak

        for (long k = 0; k < total; k++) {
            fail_at = k;
            alloc_calls = 0;
            err[0] = 0;
            snode *r = schema_compile(doc, err, sizeof(err));
            if (!r && err[0] == 0) {
                fprintf(stderr, "schema %zu, failing allocation %ld of %ld: "
                                "rejected with an empty error message\n",
                        i, k, total);
                abort();
            }
            schema_free(r);
            if (alloc_live != base_live) {
                fprintf(stderr, "schema %zu, failing allocation %ld of %ld: "
                                "%ld block(s) leaked\n", i, k, total,
                        alloc_live - base_live);
                abort();
            }
        }
        fail_at = -1;

        jv_free(doc);
        assert(alloc_live == 0);
    }
}

// The NATIVE compilers reach the same allocator from the same untrusted
// request -- tools[] out of a chat request, response_format out of the same
// body -- but they build their trees by hand instead of through compile_node,
// with an unwind label per function. That is exactly the code an
// allocation-failure sweep is for, and it was never swept: it turned up a
// leaked raw-value node and two rejections that came back with no reason at
// all for a caller to print.
static const char *const NATIVE_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"},"
        "\"units\":{\"type\":\"string\",\"enum\":[\"c\",\"f\"]},"
        "\"days\":{\"type\":\"integer\",\"minimum\":1,\"maximum\":7},"
        "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},"
                  "\"maxItems\":3}},"
      "\"required\":[\"city\"]}}},"
     "{\"type\":\"function\",\"function\":{\"name\":\"add\",\"parameters\":"
      "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},"
      "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"]}}}]";
static const char *const NATIVE_FINAL =
    "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},"
    "\"required\":[\"answer\"]}";

typedef snode *(*native_fn)(jv *tools, jv *final, char *err, int errcap);

static snode *n_atem_tools(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_atem_tools(t, e, c);
}
static snode *n_atem_turn(jv *t, jv *f, char *e, int c) {
    return schema_compile_atem_turn(t, true, NULL, f, ATEM_TURN_EITHER, e, c);
}
static snode *n_atem_named(jv *t, jv *f, char *e, int c) {
    (void)f;
    return schema_compile_atem_turn(t, false, "add", NULL, ATEM_TURN_DIRECT, e, c);
}
static snode *n_atem_parallel(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_atem_parallel(t, NULL, e, c);
}
static snode *n_muse_user(jv *t, jv *f, char *e, int c) {
    (void)t; return schema_compile_muse_user_payload(f, e, c);
}
static snode *n_harmony(jv *t, jv *f, char *e, int c) {
    return schema_compile_harmony_turn(t, true, NULL, f, true, e, c);
}
static snode *n_harmony_required(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_harmony_turn(t, false, NULL, NULL, true, e, c);
}
static snode *n_gemma4(jv *t, jv *f, char *e, int c) {
    return schema_compile_gemma4_turn(t, true, NULL, f, true, false, false, e, c);
}
static snode *n_gemma4_primed(jv *t, jv *f, char *e, int c) {
    (void)f;
    return schema_compile_gemma4_turn(t, true, NULL, NULL, true, true, false, e, c);
}
static snode *n_gemma4_parallel(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_gemma4_parallel(t, NULL, e, c);
}
static snode *n_gemma4_auto_parallel(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_gemma4_turn(t, true, NULL, NULL,
                                             true, false, true, e, c);
}
// Qwen: the reasoning turn (thought or call, then the un-thought turn as
// its continuation), the un-thought turn, and the parallel form.
static snode *n_qwen(jv *t, jv *f, char *e, int c) {
    return schema_compile_qwen_turn(t, true, NULL, f, true, false, e, c);
}
static snode *n_qwen_no_think(jv *t, jv *f, char *e, int c) {
    return schema_compile_qwen_turn(t, true, NULL, f, false, false, e, c);
}
static snode *n_qwen_parallel(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_qwen_parallel(t, NULL, e, c);
}
static snode *n_qwen_named(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_qwen_turn(t, false, "add", NULL, false, false, e, c);
}
// Qwen3-Coder's function/parameter XML: the auto turn with a final schema,
// the named call, and the parallel required turn.
static snode *n_qwen_xml(jv *t, jv *f, char *e, int c) {
    return schema_compile_qwen_xml_turn(t, true, NULL, f, false, e, c);
}
static snode *n_qwen_xml_named(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_qwen_xml_turn(t, false, "add", NULL, false, e, c);
}
static snode *n_qwen_xml_parallel(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_qwen_xml_turn(t, false, NULL, NULL, true, e, c);
}
// Nemotron Nano's <TOOLCALL> list: one entry, a named entry, and the
// parallel list.
static snode *n_nemotron(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_nemotron_turn(t, NULL, false, e, c);
}
static snode *n_nemotron_named(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_nemotron_turn(t, "add", false, e, c);
}
static snode *n_nemotron_parallel(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_nemotron_turn(t, NULL, true, e, c);
}
// Mistral's [TOOL_CALLS] list: the auto turn with prose, the auto turn with a
// final schema, the named call, and the parallel required turn.
static snode *n_mistral_text(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_mistral_turn(t, true, NULL, NULL, false, e, c);
}
static snode *n_mistral_final(jv *t, jv *f, char *e, int c) {
    return schema_compile_mistral_turn(t, true, NULL, f, false, e, c);
}
static snode *n_mistral_named(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_mistral_turn(t, false, "add", NULL, false, e, c);
}
static snode *n_mistral_parallel(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_mistral_turn(t, false, NULL, NULL, true, e, c);
}
static snode *n_harmony_named(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_harmony_turn(t, false, "add", NULL, false, e, c);
}
static snode *n_gemma4_named(jv *t, jv *f, char *e, int c) {
    (void)f; return schema_compile_gemma4_turn(t, false, "add", NULL, false, false, false, e, c);
}

static const struct { const char *name; native_fn fn; } NATIVE[] = {
    { "atem_tools",       n_atem_tools },
    { "atem_turn",        n_atem_turn },
    { "atem_turn_named",  n_atem_named },
    { "atem_parallel",    n_atem_parallel },
    { "muse_user",        n_muse_user },
    { "harmony",          n_harmony },
    { "harmony_required", n_harmony_required },
    { "gemma4",           n_gemma4 },
    { "gemma4_primed",    n_gemma4_primed },
    { "gemma4_parallel",  n_gemma4_parallel },
    { "gemma4_auto_parallel", n_gemma4_auto_parallel },
    { "qwen",             n_qwen },
    { "qwen_no_think",    n_qwen_no_think },
    { "qwen_parallel",    n_qwen_parallel },
    { "qwen_named",       n_qwen_named },
    { "qwen_xml",         n_qwen_xml },
    { "qwen_xml_named",   n_qwen_xml_named },
    { "qwen_xml_parallel", n_qwen_xml_parallel },
    { "nemotron",         n_nemotron },
    { "nemotron_named",   n_nemotron_named },
    { "nemotron_parallel", n_nemotron_parallel },
    { "mistral_text",     n_mistral_text },
    { "mistral_final",    n_mistral_final },
    { "mistral_named",    n_mistral_named },
    { "mistral_parallel", n_mistral_parallel },
    { "harmony_named",    n_harmony_named },
    { "gemma4_named",     n_gemma4_named },
};

static void test_native_compilers_survive_allocation_failure(void) {
    for (size_t i = 0; i < sizeof(NATIVE) / sizeof(*NATIVE); i++) {
        fail_at = -1;
        alloc_calls = 0;
        alloc_live = 0;
        jv *tools = json_parse(NATIVE_TOOLS, strlen(NATIVE_TOOLS));
        jv *final = json_parse(NATIVE_FINAL, strlen(NATIVE_FINAL));
        assert(tools != NULL && final != NULL);
        long base_live = alloc_live;

        char err[256];
        err[0] = 0;
        alloc_calls = 0;
        snode *n = NATIVE[i].fn(tools, final, err, sizeof(err));
        if (!n) {
            fprintf(stderr, "%s does not compile cleanly: %s\n",
                    NATIVE[i].name, err);
            abort();
        }
        long total = alloc_calls;
        assert(total > 0);
        schema_free(n);
        assert(alloc_live == base_live);

        for (long k = 0; k < total; k++) {
            fail_at = k;
            alloc_calls = 0;
            err[0] = 0;
            snode *r = NATIVE[i].fn(tools, final, err, sizeof(err));
            if (!r && err[0] == 0) {
                fprintf(stderr, "%s, failing allocation %ld of %ld: rejected "
                                "with an empty error message\n",
                        NATIVE[i].name, k, total);
                abort();
            }
            schema_free(r);
            if (alloc_live != base_live) {
                fprintf(stderr, "%s, failing allocation %ld of %ld: "
                                "%ld block(s) leaked\n",
                        NATIVE[i].name, k, total, alloc_live - base_live);
                abort();
            }
        }
        fail_at = -1;

        jv_free(tools);
        jv_free(final);
        assert(alloc_live == 0);
    }
}

int main(void) {
    test_schema_automata_cap();
    test_pattern_classes_survive_allocation_failure();
    test_pattern_feed_survives_allocation_failure();
    test_compile_survives_allocation_failure();
    test_native_compilers_survive_allocation_failure();
    puts("schema oom tests ok");
    return 0;
}
