// `runner lineage DIR` with every allocation of lineage.c failed in turn
// (R6.68.5). walk_receipts_dir() allocates three per-answer arrays; with any
// of them missing it skipped its fill loop and then read seq_v[i] through
// NULL. Each failure must end in a refusal (exit 2, an error line) with every
// block lineage.c allocated given back, and the clean run must verify.
//
// lineage.c is compiled into this test with its allocators macro-substituted,
// the pattern of tests/test_quantize_oom.c. It also frees blocks other units
// allocated (receipts_list_dir's paths, plat_read_file's text), so a block is
// counted only while it is one lineage.c allocated itself.
// Usage: test-lineage-oom WORKDIR
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
    // The old block is forgotten before the call: q may not be read after it
    // (gcc's -Wuse-after-free). An injected failure returns above; a real one
    // would only drop the old block from the table, never count a leak.
    if (q) untrack(q);
    void *p = realloc(q, n);
    track(p);
    return p;
}

__attribute__((unused)) static char *t_strdup(const char *s) {   // lineage.c may not call it
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
#include "../src/lineage.c"
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

static bool write_receipt(const char *dir, int i, const char *prev, const char *hash) {
    char path[1024];
    snprintf(path, sizeof path, "%s/receipt-%010d.json", dir, i);   // receipts.c: ten digits
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "{\"generated_utc\":\"2026-11-12T00:00:0%dZ\",\"chain\":{\"prev\":\"%s\",\"hash\":\"%s\"},"
               "\"model\":{\"sha256\":\"%064d\"}}\n", i, prev, hash, i < 2 ? 1 : 2);
    return fclose(f) == 0;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: test-lineage-oom WORKDIR\n"); return 2; }
    char dir[1024];
    snprintf(dir, sizeof dir, "%s/lineage-oom-receipts", argv[1]);
    plat_mkdir_exist_ok(dir);
    if (!write_receipt(dir, 0, "", "aaaa") || !write_receipt(dir, 1, "aaaa", "bbbb") ||
        !write_receipt(dir, 2, "bbbb", "cccc")) {
        fprintf(stderr, "test-lineage-oom: cannot write the receipts under %s\n", argv[1]);
        return 2;
    }
    char sink[1100];
    snprintf(sink, sizeof sink, "%s/lineage-oom-out.txt", argv[1]);
    if (!freopen(sink, "w", stdout)) return 2;   // the walk's report is not this test's output

    alloc_calls = 0; fail_at = -1;
    int clean = walk_receipts_dir(dir, NULL);
    long total = alloc_calls;
    ck(clean != 2, "the clean walk of three receipts is not BROKEN", -1);
    ck(n_tracked == 0, "the clean walk gives back every block it allocated", -1);
    ck(total >= 3, "the walk allocates its three per-answer arrays", -1);

    for (long k = 0; k < total; k++) {
        alloc_calls = 0; fail_at = k; n_tracked = 0;
        int rc = walk_receipts_dir(dir, NULL);
        ck(rc == 2 || rc == clean, "a failed allocation is a refusal or the clean answer", k);
        ck(n_tracked == 0, "a failed allocation gives back every block", k);
    }
    fail_at = -1;
    ck(!overflow, "the tracking table held every live block", -1);
    if (!g_fail) fprintf(stderr, "lineage oom ok: %ld allocations, each failed in turn\n", total);
    return g_fail;
}
