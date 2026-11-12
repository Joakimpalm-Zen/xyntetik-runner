// `runner derive --target-size` planning with every allocation of derive.c
// failed in turn (R6.68.5). sized_plan() grew three arrays with realloc
// assigned over their own pointers (the held tensors, the groups, a group's
// tensors): a failure dereferenced NULL and lost the old block, and no return
// freed the groups, the success path included. Each failure must now end in a
// refusal with every block derive.c allocated given back, and a clean plan
// must give back everything except what its result owns.
//
// derive.c is compiled into this test with its allocators macro-substituted,
// the pattern of tests/test_quantize_oom.c, and the test links every other
// runner object. sized_plan() runs RUNNER --quant-error on PARENT as a child
// process, whose allocations are its own; blocks other units allocated and
// derive.c frees (the report text) are not counted.
// Usage: test-derive-oom WORKDIR RUNNER PARENT.gguf
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRACK_CAP 65536
static void *tracked[TRACK_CAP];
static long n_tracked;
static long alloc_calls;   // allocations attempted since the last reset
static long fail_at = -1;  // the allocation to fail, -1 for none
static bool overflow;

static void track(void *p) {
    if (!p) return;
    if (n_tracked < TRACK_CAP) tracked[n_tracked++] = p;
    else overflow = true;
}

static bool untrack(void *p) {
    for (long i = n_tracked - 1; i >= 0; i--)
        if (tracked[i] == p) { tracked[i] = tracked[--n_tracked]; return true; }
    return false;
}

static void *t_malloc(size_t n) {
    if (alloc_calls++ == fail_at) return NULL;
    void *p = malloc(n);
    track(p);
    return p;
}

static void *t_calloc(size_t a, size_t b) {
    if (alloc_calls++ == fail_at) return NULL;
    void *p = calloc(a, b);
    track(p);
    return p;
}

static void *t_realloc(void *q, size_t n) {
    if (alloc_calls++ == fail_at) return NULL;
    void *p = realloc(q, n);
    if (p) { if (q) untrack(q); track(p); }
    return p;
}

__attribute__((unused)) static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static void t_free(void *p) {
    if (p) untrack(p);
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define strdup  t_strdup
#define free    t_free
#include "../src/derive.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

static int g_fail = 0;

static void ck(int cond, const char *what, long k) {
    if (cond) return;
    if (k >= 0) fprintf(stderr, "FAIL: %s (allocation %ld failed)\n", what, k);
    else fprintf(stderr, "FAIL: %s\n", what);
    g_fail = 1;
}

// What a sized_result owns, given back through the counted free.
static void result_free(sized_result *r) {
    t_free(r->report);
    t_free(r->plan);
    for (int k = 0; r->demoted && k < r->n_demoted; k++) t_free(r->demoted[k]);
    t_free(r->demoted);
    t_free((void *)r->demoted_to);
    t_free(r->skipped);
    for (int k = 0; r->held && k < r->n_held; k++) t_free(r->held[k]);
    t_free(r->held);
    memset(r, 0, sizeof *r);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: test-derive-oom WORKDIR RUNNER PARENT.gguf\n"); return 2; }
    char out[1024];
    snprintf(out, sizeof out, "%s/derive-oom.gguf", argv[1]);
    derive_opts o;
    memset(&o, 0, sizeof o);
    o.parent = argv[3];
    o.out = out;
    o.rungs[0] = "q8_0";
    o.rungs[1] = "q4_0";
    o.n_rungs = 2;
    o.threads = "1";
    long long pbytes = file_bytes(o.parent);
    sized_result r;
    memset(&r, 0, sizeof r);

    // a target of one byte is out of reach, and tells the size at the top rung
    o.target_size = 1;
    alloc_calls = 0; fail_at = -1;
    ck(!sized_plan(argv[2], &o, pbytes, &r), "a one-byte target is refused as out of reach", -1);
    long long at_high = r.save_bytes + 1;
    result_free(&r);
    ck(n_tracked == 0, "a refused plan gives back every block", -1);

    // one byte under it: the cheapest group is demoted and the plan is written
    o.target_size = at_high - 1;
    alloc_calls = 0; n_tracked = 0;
    bool clean = sized_plan(argv[2], &o, pbytes, &r);
    long total = alloc_calls;
    ck(clean, "a target one byte under the top rung is planned", -1);
    ck(r.n_demoted >= 1, "the plan demotes at least one group", -1);
    result_free(&r);
    ck(n_tracked == 0, "a clean plan gives back every block its result does not own", -1);
    ck(total >= 4, "the plan allocates its groups, their names and the result", -1);

    for (long k = 0; k < total; k++) {
        alloc_calls = 0; fail_at = k; n_tracked = 0;
        bool ok = sized_plan(argv[2], &o, pbytes, &r);
        result_free(&r);
        ck(!ok, "a failed allocation is a refusal", k);
        ck(n_tracked == 0, "a failed allocation gives back every block", k);
    }
    fail_at = -1;
    ck(!overflow, "the tracking table held every live block", -1);
    if (!g_fail) fprintf(stderr, "derive oom ok: %ld allocations, each failed in turn\n", total);
    return g_fail;
}
