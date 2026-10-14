// Allocation failure anywhere in a file transformation, and a refusal at any
// stage: refused, the destination absent, no partial file left behind, no
// block leaked.
//
// quantize_gguf_plan_inner() serves --quantize (with a type plan, an expert
// prune plan or a sublayer removal), --merge-lora and --context-surgery, and
// allocates in every stage: the plans, the removal pre-pass, the layout, the
// per-tensor conversion buffers, the output's staging. A failure in any of
// them has to unwind what the stages before it built and leave the
// destination as it was. The property is pinned from outside, per command:
// for each allocation a successful run makes, failing that one allocation
// must end either in a refusal that leaves no OUT and no OUT.partial and no
// live block, or (for an allocation the command can do without) in a run
// whose output is byte-identical to the clean run's.
//
// The refusals that are not allocation failures go through the same exits:
// a prune plan naming an expert the file lacks, a destination directory that
// does not exist, a --type-plan-strict plan that would not apply as written.
// Each runs after a stage that allocated (the sublayer removal's per-layer
// arrays, the prune's per-layer counts), so each pins that the exit releases
// it.
//
// quantize.c is compiled into this test with its allocators
// macro-substituted, the pattern of tests/test_model_load_oom.c; the system
// headers come first, so the macros rewrite only quantize.c's own calls.
// Usage: test-quantize-oom WORKDIR DENSE.gguf MOE.gguf YARN.gguf ADAPTER.gguf
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static long alloc_calls;   // allocations attempted since the last reset
static long alloc_live;    // outstanding blocks
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

static void *t_realloc(void *q, size_t n) {
    if (alloc_calls++ == fail_at) return NULL;
    void *p = realloc(q, n);
    if (p && !q) alloc_live++;
    return p;
}

static char *t_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = t_malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define realloc t_realloc
#define strdup  t_strdup
#define free    t_free
#include "../src/quantize.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

static int g_fail = 0;

static void ck(int cond, const char *what, const char *detail, long k) {
    if (cond) return;
    if (k >= 0) fprintf(stderr, "FAIL: %s: %s (allocation %ld failed)\n", what, detail, k);
    else fprintf(stderr, "FAIL: %s: %s\n", what, detail);
    g_fail = 1;
}

static bool exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

static char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)*n + 1);   // the test's own, not counted
    if (b && fread(b, 1, (size_t)*n, f) != (size_t)*n) { free(b); b = NULL; }
    fclose(f);
    return b;
}

static void write_file(const char *p, const char *text) {
    FILE *f = fopen(p, "w");
    if (f) { fputs(text, f); fclose(f); }
}

typedef struct {
    const char *name;
    int kind;              // 0 quantize, 1 merge, 2 context surgery
    const char *in, *prune, *tplan, *remove, *adapter;
    int target;
} qcase;

static int run_case(const qcase *c, const char *out) {
    switch (c->kind) {
    case 0: return quantize_gguf_plan(c->in, out, c->target, c->prune, c->tplan, c->remove);
    case 1: {
        merge_survival s;
        return merge_lora_gguf(c->in, c->adapter, 1.0f, out, c->target, -1.0f, &s);
    }
    default: {
        context_surgery_result r;
        return context_surgery_gguf(c->in, out, 262144, 64.0f, &r);
    }
    }
}

static void remove_outputs(const char *out) {
    char p[1100];
    remove(out);
    snprintf(p, sizeof p, "%s.partial", out);
    remove(p);
}

static void sweep(const qcase *c, const char *work) {
    char out[1024], partial[1100];
    snprintf(out, sizeof out, "%s/test-quantize-oom-%s.gguf", work, c->name);
    snprintf(partial, sizeof partial, "%s.partial", out);
    remove_outputs(out);
    long live0 = alloc_live;
    fail_at = -1;
    alloc_calls = 0;
    int rc = run_case(c, out);
    long n = alloc_calls;
    ck(rc == 0, c->name, "the clean run succeeds", -1);
    ck(alloc_live == live0, c->name, "a clean run frees every block", -1);
    long clean_n = 0;
    char *clean = slurp(out, &clean_n);
    ck(clean != NULL, c->name, "the clean run wrote its output", -1);
    remove_outputs(out);
    long refused = 0, optional = 0;
    for (long k = 0; k < n && clean; k++) {
        fail_at = k;
        alloc_calls = 0;
        rc = run_case(c, out);
        fail_at = -1;
        if (rc != 0) {
            refused++;
            ck(!exists(out), c->name, "a refused run leaves no output", k);
            ck(!exists(partial), c->name, "a refused run leaves no partial file", k);
        } else {
            // an allocation the command can do without: the output must
            // be the clean run's
            optional++;
            long got_n = 0;
            char *got = slurp(out, &got_n);
            ck(got && got_n == clean_n && memcmp(got, clean, (size_t)clean_n) == 0,
               c->name, "a run that survived a failure writes the clean output", k);
            free(got);
        }
        ck(alloc_live == live0, c->name, "the failed allocation leaves no block behind", k);
        remove_outputs(out);
    }
    free(clean);
    fprintf(stderr, "test-quantize-oom: %-14s %ld allocations, %ld refused, %ld optional\n",
            c->name, n, refused, optional);
}

// A refusal that is not an allocation failure: refused, nothing written,
// nothing leaked.
static void refusal(const qcase *c, const char *out) {
    char partial[1100];
    snprintf(partial, sizeof partial, "%s.partial", out);
    remove_outputs(out);
    long live0 = alloc_live;
    int failed0 = g_fail;
    g_fail = 0;
    fail_at = -1;
    int rc = run_case(c, out);
    ck(rc != 0, c->name, "the request is refused", -1);
    ck(!exists(out), c->name, "a refused run leaves no output", -1);
    ck(!exists(partial), c->name, "a refused run leaves no partial file", -1);
    ck(alloc_live == live0, c->name, "a refused run frees every block", -1);
    remove_outputs(out);
    if (!g_fail)
        fprintf(stderr, "test-quantize-oom: %-14s refused, nothing written, nothing leaked\n",
                c->name);
    g_fail |= failed0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: test-quantize-oom WORKDIR DENSE.gguf MOE.gguf YARN.gguf ADAPTER.gguf\n");
        return 2;
    }
    const char *work = argv[1], *dense = argv[2], *moe = argv[3], *yarn = argv[4],
               *adapter = argv[5];
    char tplan[1024], prune[1024];
    snprintf(tplan, sizeof tplan, "%s/test-quantize-oom-type-plan.json", work);
    snprintf(prune, sizeof prune, "%s/test-quantize-oom-prune.json", work);
    write_file(tplan, "{\"default\": \"q8_0\", \"rules\": [{\"match\": \"ffn\", \"type\": \"q4_0\"}]}");
    write_file(prune, "{\"layer_0\": [0, 1], \"layer_1\": [2, 3]}");
    int q8 = quantize_type_from_name("q8_0"), keep = quantize_type_from_name("keep");
    const qcase cases[] = {
        { "quantize-q8",   0, dense, NULL,  NULL,  NULL,     NULL,    q8 },
        { "type-plan",     0, dense, NULL,  tplan, NULL,     NULL,    keep },
        { "remove-attn1",  0, dense, NULL,  NULL,  "attn:1", NULL,    keep },
        { "prune-experts", 0, moe,   prune, NULL,  NULL,     NULL,    keep },
        { "merge-q8",      1, dense, NULL,  NULL,  NULL,     adapter, q8 },
        { "surgery",       2, yarn,  NULL,  NULL,  NULL,     NULL,    keep },
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) sweep(&cases[i], work);

    char bad_prune[1024], strict_plan[1024], out[1024], no_dir[1100];
    snprintf(bad_prune, sizeof bad_prune, "%s/test-quantize-oom-prune-range.json", work);
    snprintf(strict_plan, sizeof strict_plan, "%s/test-quantize-oom-strict-plan.json", work);
    write_file(bad_prune, "{\"layer_0\": [0, 99]}");
    // q4_k's 256-wide block divides none of the fixture's rows: the plan
    // falls back, which --type-plan-strict refuses
    write_file(strict_plan, "{\"default\": \"q4_k\"}");
    snprintf(out, sizeof out, "%s/test-quantize-oom-refused.gguf", work);
    snprintf(no_dir, sizeof no_dir, "%s/test-quantize-oom-no-such-dir/refused.gguf", work);
    const qcase refused[] = {
        { "prune-range",   0, moe,   bad_prune, NULL,        NULL,     NULL, keep },
        { "no-out-dir",    0, dense, NULL,      NULL,        "attn:1", NULL, keep },
        { "strict-plan",   0, dense, NULL,      strict_plan, "attn:1", NULL, keep },
    };
    refusal(&refused[0], out);
    refusal(&refused[1], no_dir);
    quantize_set_type_plan_strict(true);
    refusal(&refused[2], out);
    quantize_set_type_plan_strict(false);
    remove(bad_prune);
    remove(strict_plan);
    remove(tplan);
    remove(prune);
    if (g_fail) { fprintf(stderr, "test-quantize-oom FAILED\n"); return 1; }
    fprintf(stderr, "test-quantize-oom: every allocation of six transformations failed in turn, "
            "each refused cleanly or wrote the clean output; three refusals; no leaks\n");
    return 0;
}
