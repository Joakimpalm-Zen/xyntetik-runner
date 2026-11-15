// Allocation-failure tests for the atem tool-call mapper (R2.19.1, codex's
// read of #897).
//
// The mapper recovers a cut-off number argument from the parameter's
// compiled schema (schema_fill), after resolving the tool's declarations
// (schema_resolve_refs). An allocation failure in either was read as "no
// constraint": the recovery wrote 0 where minimum 10 holds, and a failed
// resolution typed a $ref'd number as a string (""), both returned as a
// successful call. The stream reader and the call sink had the same shape
// one level up: a failed buffer ended the turn as a clean one with no call,
// and a failed argument buffer wrote "arguments":"". Each failure must now
// fail the call; a mapper that succeeds writes exactly what it writes when
// nothing fails.
//
// Every allocation of one mapping is failed in turn. json.c, schema.c,
// schema_tools.c, jsre.c and template.c are compiled straight into this test
// with their allocators macro-substituted, the technique of
// test_template_oom.c and test_schema_oom.c; tokenizer.c, gguf.c and the
// rest link normally.
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long alloc_calls;   // allocations attempted since the last reset
static long alloc_live;    // outstanding blocks; must return to the baseline
static long fail_at = -1;  // index of the allocation to fail, -1 for none

static void *t_malloc(size_t n) {
    if (alloc_calls++ == fail_at) return NULL;
    void *p = malloc(n);
    if (p) alloc_live++;
    return p;
}

static void *t_calloc(size_t a, size_t b) {
    if (alloc_calls++ == fail_at) return NULL;
    void *p = calloc(a, b);
    if (p) alloc_live++;
    return p;
}

static void *t_realloc(void *p, size_t n) {
    if (alloc_calls++ == fail_at) return NULL;
    void *q = realloc(p, n);
    if (q && !p) alloc_live++;   // realloc(NULL, n) behaves as malloc
    return q;
}

static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

__attribute__((unused)) static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

__attribute__((unused)) static char *t_strndup(const char *s, size_t cap) {
    size_t n = strnlen(s, cap);
    char *p = t_malloc(n + 1);
    if (p) { memcpy(p, s, n); p[n] = 0; }
    return p;
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
// jsre.c names its NFA node type `state` at file scope, and template.c has a
// local of that name, which gcc's -Wshadow refuses in one unit (Apple clang
// does not flag it). The type is renamed for this test's unit only; every
// other use of the word in jsre.c and jsre.h is a comment.
#define state jsre_nfa_state
#include "../src/jsre.c"
#undef state
// schema_tools.c and template.c each define their own G4_MAX_DEPTH; each
// file's code expands its own, so the first goes before the second
#undef G4_MAX_DEPTH
#include "../src/template.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef strndup
#undef free

// Blocks the instrumented units allocated (the mapped outputs, a dump) go back
// through t_free below, or they would count as leaks.

// A cut-off argument (empty), so the mapper reaches the recovery.
static const char *DOC =
    " to=probe<|message|><atem:function_calls>\n<atem:invoke name=\"probe\">\n"
    "<atem:parameter name=\"x\"></atem:parameter>\n</atem:invoke>\n</atem:function_calls>";

static int sweep_doc(const char *doc, const char *param, const char *defs, const char *want_args) {
    char src[2048];
    snprintf(src, sizeof(src),
             "[{\"type\":\"function\",\"function\":{\"name\":\"probe\",\"parameters\":{\"type\":"
             "\"object\",\"properties\":{\"x\":%s},\"required\":[\"x\"]%s}}}]", param, defs);
    fail_at = -1;
    jv *tools = json_parse(src, strlen(src));
    assert(tools);
    sbuf before = {0};
    jv_dump(tools, &before);
    assert(before.s);
    tool_envelope env = {.proto = TP_ATEM, .tools = tools};
    sbuf content = {0}, calls = {0};
    alloc_calls = 0;
    int rc = tool_envelope_map(&env, doc, strlen(doc), &content, &calls);
    long total = alloc_calls;
    if (rc != 1 || !calls.s || !strstr(calls.s, want_args)) {
        fprintf(stderr, "R2.19.1: parameter %s maps to %s (rc %d), want arguments %s\n",
                param, calls.s ? calls.s : "(nothing)", rc, want_args);
        assert(0);
    }
    char *want = calls.s;
    t_free(content.s);
    long baseline = alloc_live;
    int wrong = 0;
    for (long k = 0; k < total; k++) {
        sbuf c = {0}, t = {0};
        alloc_calls = 0;
        fail_at = k;
        int got = tool_envelope_map(&env, doc, strlen(doc), &c, &t);
        fail_at = -1;
        // failed, or exactly the output an unfailed mapping writes
        if (got >= 0 && (got != 1 || !t.s || strcmp(t.s, want))) {
            if (wrong < 4)
                fprintf(stderr, "R2.19.1: parameter %s, allocation %ld of %ld failed: rc %d, calls %s\n",
                        param, k, total, got, t.s ? t.s : "(none)");
            wrong++;
        }
        t_free(c.s);
        t_free(t.s);
        if (alloc_live != baseline) {
            fprintf(stderr, "R2.19.1: parameter %s, allocation %ld failed: %ld blocks leaked\n",
                    param, k, alloc_live - baseline);
            wrong++;
            alloc_live = baseline;
        }
        sbuf after = {0};
        jv_dump(tools, &after);
        assert(after.s && !strcmp(after.s, before.s));   // the declarations are not changed
        t_free(after.s);
    }
    printf("atem oom: %s (%s argument): %ld allocations failed in turn, %d wrong\n", param,
           doc == DOC ? "cut-off" : "whole", total, wrong);
    t_free(want);
    t_free(before.s);
    jv_free(tools);
    return wrong;
}

static int sweep(const char *param, const char *defs, const char *want_args) {
    return sweep_doc(DOC, param, defs, want_args);
}

// R2.21: a WHOLE argument, valid JSON, whose json_parse lost an allocation was
// read as unparseable and replaced by the recovery's fill (12 became 10 under
// minimum 10): an allocation failure must fail the call, not change the value.
static char *whole_doc(const char *value) {
    static char doc[512];
    snprintf(doc, sizeof(doc),
             " to=probe<|message|><atem:function_calls>\n<atem:invoke name=\"probe\">\n"
             "<atem:parameter name=\"x\">%s</atem:parameter>\n</atem:invoke>\n</atem:function_calls>", value);
    return doc;
}

// schema_fill's own contract (codex's second look at #897): -1 on any
// allocation failure inside it, in the compile or the emit; 1 only with a value
// the compiled schema accepts. Every allocation of one fill is failed in turn,
// and each outcome is judged by the schema's own validator (built once, with
// nothing failed). A pattern's witness that lost its buffer came back as 1 with
// "", and the pattern compiler reports some allocation failures as refusals.
static bool value_accepted(const snode *n, const char *s) {
    sval v;
    sval_init(&v, n);
    bool ok = sval_feed(&v, s, (int)strlen(s));
    if (ok && !v.done) (void)sval_feed(&v, " ", 1);   // a bare number ends here
    return ok && v.done;
}

static int fill_sweep(const char *src) {
    fail_at = -1;
    jv *j = json_parse(src, strlen(src));
    assert(j);
    char err[256], out[512];
    snode *check = schema_compile(j, err, sizeof(err));
    if (!check) { fprintf(stderr, "fill sweep: %s does not compile: %s\n", src, err); assert(0); }
    alloc_calls = 0;
    int rc = schema_fill(j, out, (int)sizeof(out));
    long total = alloc_calls;
    if (rc != 1 || !value_accepted(check, out)) {
        fprintf(stderr, "fill sweep: %s fills %s (rc %d) with nothing failed\n", src, out, rc);
        assert(0);
    }
    long baseline = alloc_live;
    int wrong = 0;
    for (long k = 0; k < total; k++) {
        alloc_calls = 0;
        fail_at = k;
        rc = schema_fill(j, out, (int)sizeof(out));
        fail_at = -1;
        if (rc == -1 ? out[0] != 0 : rc != 1 || !value_accepted(check, out)) {
            if (wrong < 4)
                fprintf(stderr, "fill sweep: %s, allocation %ld of %ld failed: rc %d, out \"%s\"\n",
                        src, k, total, rc, out);
            wrong++;
        }
        if (alloc_live != baseline) {
            fprintf(stderr, "fill sweep: %s, allocation %ld failed: %ld blocks leaked\n",
                    src, k, alloc_live - baseline);
            wrong++;
            alloc_live = baseline;
        }
    }
    printf("fill sweep: %s: %ld allocations failed in turn, %d wrong\n", src, total, wrong);
    schema_free(check);
    jv_free(j);
    return wrong;
}

int main(void) {
    int bad = 0;
    static const char *const FILLS[] = {
        "{\"type\":\"number\",\"minimum\":10}",
        "{\"type\":\"number\",\"exclusiveMinimum\":0,\"maximum\":5e-308}",
        "{\"type\":\"integer\",\"minimum\":5,\"maximum\":9}",
        "{\"type\":\"string\",\"pattern\":\"^(a|b)$\"}",
        "{\"type\":\"string\",\"minLength\":3}",
        "{\"type\":\"number\",\"const\":5}",
        "{\"enum\":[3,\"x\"]}",
        "{\"type\":\"boolean\"}",
        "{\"$ref\":\"#/$defs/n\",\"$defs\":{\"n\":{\"type\":\"number\",\"minimum\":10}}}",
        "{\"allOf\":[{\"type\":\"string\",\"pattern\":\"^(a|b)$\"}]}",
        "{\"type\":\"object\",\"properties\":{\"k\":{\"type\":\"integer\",\"minimum\":2}},\"required\":[\"k\"]}",
        "{\"type\":\"array\",\"items\":{\"type\":\"number\",\"maximum\":-1},\"minItems\":2}",
    };
    for (size_t i = 0; i < sizeof(FILLS) / sizeof(FILLS[0]); i++) bad += fill_sweep(FILLS[i]);
    bad += sweep_doc(whole_doc("12"), "{\"type\":\"number\",\"minimum\":10}", "", "{\\\"x\\\":12}");
    bad += sweep_doc(whole_doc("true"), "{\"type\":\"boolean\"}", "", "{\\\"x\\\":true}");
    bad += sweep_doc(whole_doc("{\"k\":3}"),
                     "{\"type\":\"object\",\"properties\":{\"k\":{\"type\":\"integer\",\"minimum\":2}},\"required\":[\"k\"]}",
                     "", "{\\\"x\\\":{\\\"k\\\":3}}");
    bad += sweep("{\"type\":\"number\",\"minimum\":10}", "", "{\\\"x\\\":10}");
    bad += sweep("{\"$ref\":\"#/$defs/N\"}", ",\"$defs\":{\"N\":{\"type\":\"number\",\"minimum\":10}}",
                 "{\\\"x\\\":10}");
    bad += sweep("{\"allOf\":[{\"type\":\"integer\",\"maximum\":-4}]}", "", "{\\\"x\\\":-4}");
    bad += sweep("{\"type\":\"number\",\"const\":5}", "", "{\\\"x\\\":5}");
    if (bad) {
        fprintf(stderr, "atem oom: %d wrong outputs or leaks\n", bad);
        return 1;
    }
    printf("atem oom: every failed allocation fails the call or changes nothing\n");
    return 0;
}
