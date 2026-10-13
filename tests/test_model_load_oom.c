// Allocation failure anywhere in a model load: refused or loaded, never a leak.
//
// model_load binds the weights (model_bind_weights), publishes them for other
// instances of the same file (the model_weights record), then sizes this
// instance's runtime (model_alloc_runtime). Every stage allocates, and a
// failure in any of them has to unwind exactly what the stages before it
// built. The property is pinned from outside, per fixture: for each
// allocation a successful load makes, failing that one allocation must end
// either in a refused load that leaves no block behind, or, for the shared
// record alone (sharing is an optimization, never a reason to fail), in a
// working load that does not share. A second sweep repeats this with another
// instance of the file already loaded, so the failure lands in a load that
// reuses the published weights: the holder's reference must survive every
// failed load, the holder must still decode, and freeing it must release
// every block.
//
// model.c is compiled into this test with its allocators macro-substituted,
// the pattern of tests/test_instances_oom.c: the system headers come first,
// so the macros rewrite only model.c's own calls (it allocates through
// malloc, calloc and free alone). Every other object is the engine's own.
// Fixture paths come from the command line (Makefile).
#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
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

static void t_free(void *p) {
    if (p) alloc_live--;
    free(p);
}

#define malloc  t_malloc
#define calloc  t_calloc
#define free    t_free
#include "../src/model.c"
#undef malloc
#undef calloc
#undef free

static int g_fail = 0;

static void ck(int cond, const char *path, const char *what, long k) {
    if (cond) return;
    if (k >= 0) fprintf(stderr, "FAIL: %s: %s (allocation %ld failed)\n", path, what, k);
    else fprintf(stderr, "FAIL: %s: %s\n", path, what);
    g_fail = 1;
}

static model_params params(void) {
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = 128;
    p.n_batch = 4;
    return p;
}

// One clean load: its allocation count, after checking it frees everything.
static long clean_load_count(const char *path, const model_params *p) {
    static model_t m;
    long live0 = alloc_live;
    fail_at = -1;
    alloc_calls = 0;
    if (!model_load(&m, path, p)) {
        ck(0, path, "the fixture loads", -1);
        return -1;
    }
    long n = alloc_calls;
    model_free(&m);
    ck(alloc_live == live0, path, "a clean load frees every block", -1);
    return n;
}

static void sweep_alone(const char *path, const model_params *p) {
    static model_t m;
    long n = clean_load_count(path, p);
    if (n < 0) return;
    long live0 = alloc_live;
    int loaded_unshared = 0;
    for (long k = 0; k < n; k++) {
        fail_at = k;
        alloc_calls = 0;
        bool ok = model_load(&m, path, p);
        fail_at = -1;
        if (ok) {
            // only the shared record may fail without failing the load, and
            // then the instance holds its weights privately
            loaded_unshared++;
            ck(m.W == NULL, path, "a load that survived a failure does not share", k);
            model_free(&m);
        }
        ck(alloc_live == live0, path, "the failed allocation leaves no block behind", k);
    }
    ck(loaded_unshared <= 1, path, "at most one allocation (the shared record) is optional", -1);
    ck(clean_load_count(path, p) == n, path, "a clean load after the sweep is unchanged", -1);
}

static void sweep_with_holder(const char *path, const model_params *p) {
    static model_t holder, m;
    long live0 = alloc_live;
    fail_at = -1;
    if (!model_load(&holder, path, p)) { ck(0, path, "the holder loads", -1); return; }
    ck(holder.W != NULL, path, "the holder publishes its weights", -1);
    long live1 = alloc_live;
    alloc_calls = 0;
    if (!model_load(&m, path, p)) { ck(0, path, "a second instance loads", -1); model_free(&holder); return; }
    long n = alloc_calls;
    ck(m.W == holder.W && m.layers == holder.layers, path, "the second instance shares", -1);
    model_free(&m);
    ck(alloc_live == live1, path, "freeing the second instance keeps the holder's blocks", -1);
    for (long k = 0; k < n; k++) {
        fail_at = k;
        alloc_calls = 0;
        bool ok = model_load(&m, path, p);
        fail_at = -1;
        ck(!ok, path, "a load reusing the weights fails with its allocation", k);
        if (ok) model_free(&m);
        ck(alloc_live == live1, path, "the failed load leaves the holder's blocks only", k);
    }
    ck(holder.W && holder.W->refs == 1, path, "the holder's reference survives every failed load", -1);
    ck(model_forward(&holder, 1, 0) != NULL, path, "the holder still decodes", -1);
    model_free(&holder);
    ck(alloc_live == live0, path, "freeing the holder releases every block", -1);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test-model-load-oom FIXTURE.gguf...\n");
        return 2;
    }
    model_params p = params();
    for (int i = 1; i < argc; i++) {
        sweep_alone(argv[i], &p);
        sweep_with_holder(argv[i], &p);
    }
    if (g_fail) { fprintf(stderr, "test-model-load-oom FAILED\n"); return 1; }
    fprintf(stderr, "test-model-load-oom: %d fixtures, every allocation failed in turn, no leaks\n",
            argc - 1);
    return 0;
}
