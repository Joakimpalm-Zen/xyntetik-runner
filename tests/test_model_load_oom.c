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
// The same sweep then covers the adapter lifecycle on every fixture the
// trainer admits: an adapter load and a per-request set load (refused, the
// attached adapter untouched, no block left), train-init (refused, the
// attached adapter untouched), a save (refused, the destination unchanged)
// and a first backward step (refused, the tape released, every block freed
// with the adapter).
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

static bool read_file(const char *path, char **buf, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    *buf = (char *)malloc((size_t)*n);   // the test's own block, not counted
    bool ok = *buf && fread(*buf, 1, (size_t)*n, f) == (size_t)*n;
    fclose(f);
    return ok;
}

static bool same_file(const char *path, const char *want, long wn) {
    char *got = NULL;
    long gn = 0;
    bool same = read_file(path, &got, &gn) && gn == wn && memcmp(got, want, (size_t)wn) == 0;
    free(got);
    return same;
}

// Allocations made by f(), after checking a clean run of it succeeds.
#define CLEAN_COUNT(n, call) do { fail_at = -1; alloc_calls = 0; \
        bool ok_ = (call); (n) = ok_ ? alloc_calls : -1; } while (0)

static void sweep_adapter(const char *path, const model_params *p) {
    static model_t m;
    char why[256];
    fail_at = -1;
    long live0 = alloc_live;
    if (!model_load(&m, path, p)) { ck(0, path, "the fixture loads", -1); return; }
    if (!model_lora_serve_supported(&m, why, sizeof why) ||
        !model_lora_train_supported(&m, why, sizeof why)) {
        model_free(&m);
        return;
    }
    char lpath[512];
    snprintf(lpath, sizeof lpath, "%s.oom-lora.gguf", path);
    long base_live = alloc_live;

    // the adapter file every sweep below reads
    ck(model_lora_train_init(&m, 2, 4.0f, 7), path, "train-init", -1);
    ck(model_lora_save(&m, lpath), path, "save", -1);
    char *saved = NULL;
    long saved_n = 0;
    ck(read_file(lpath, &saved, &saved_n), path, "the saved adapter reads back", -1);
    model_lora_free(&m);
    ck(alloc_live == base_live, path, "releasing a trained adapter frees every block", -1);

    // load: attached over a train-init adapter, which a failure must keep
    long n;
    CLEAN_COUNT(n, model_lora_load(&m, lpath, 1.0f));
    ck(n >= 0, path, "the adapter loads", -1);
    uint64_t loaded_id = m.lora_id;
    model_lora_free(&m);
    ck(model_lora_train_init(&m, 2, 4.0f, 7), path, "train-init", -1);
    struct lora_w *held = m.lora;
    uint64_t held_id = m.lora_id;
    long live1 = alloc_live;
    for (long k = 0; k < n; k++) {
        fail_at = k; alloc_calls = 0;
        bool ok = model_lora_load(&m, lpath, 1.0f);
        fail_at = -1;
        ck(!ok, path, "an adapter load fails with its allocation", k);
        ck(m.lora == held && m.lora_id == held_id, path,
           "a failed adapter load keeps the attached adapter", k);
        ck(alloc_live == live1, path, "a failed adapter load leaves no block", k);
        if (ok) break;
    }
    ck(model_lora_load(&m, lpath, 1.0f) && m.lora_id == loaded_id, path,
       "a clean load after the sweep has the same identity", -1);

    // a per-request set: refused or loaded, never a leak, never attached
    fail_at = -1; alloc_calls = 0;
    lora_set *ls = model_lora_set_load(&m, lpath, 1.0f);
    n = alloc_calls;
    ck(ls && model_lora_set_id(ls) == loaded_id, path, "a set loads under the file's identity", -1);
    model_lora_set_free(ls);
    long live2 = alloc_live;
    for (long k = 0; k < n; k++) {
        fail_at = k; alloc_calls = 0;
        ls = model_lora_set_load(&m, lpath, 1.0f);
        fail_at = -1;
        ck(ls == NULL, path, "a set load fails with its allocation", k);
        if (ls) { model_lora_set_free(ls); break; }
        ck(alloc_live == live2, path, "a failed set load leaves no block", k);
    }

    // train-init over the loaded adapter
    CLEAN_COUNT(n, model_lora_train_init(&m, 2, 4.0f, 7));
    ck(n >= 0, path, "train-init", -1);
    model_lora_free(&m);
    ck(model_lora_load(&m, lpath, 1.0f), path, "the adapter reloads", -1);
    held = m.lora;
    held_id = m.lora_id;
    long live3 = alloc_live;
    for (long k = 0; k < n; k++) {
        fail_at = k; alloc_calls = 0;
        bool ok = model_lora_train_init(&m, 2, 4.0f, 7);
        fail_at = -1;
        ck(!ok, path, "train-init fails with its allocation", k);
        if (ok) break;
        ck(m.lora == held && m.lora_id == held_id, path,
           "a failed train-init keeps the attached adapter", k);
        ck(alloc_live == live3, path, "a failed train-init leaves no block", k);
    }

    // save: a failure leaves the destination's bytes as they were
    CLEAN_COUNT(n, model_lora_save(&m, lpath));
    ck(n >= 0, path, "the adapter saves", -1);
    ck(same_file(lpath, saved, saved_n), path, "a loaded adapter saves to the same bytes", -1);
    for (long k = 0; k < n; k++) {
        fail_at = k; alloc_calls = 0;
        bool ok = model_lora_save(&m, lpath);
        fail_at = -1;
        ck(!ok, path, "a save fails with its allocation", k);
        ck(same_file(lpath, saved, saved_n), path, "a failed save leaves the destination", k);
        ck(alloc_live == live3, path, "a failed save leaves no block", k);
    }

    // a first backward step on a fresh adapter: gradient buffers, the tape
    // and every layer's scratch are allocated during it. Two tokens, one
    // scored transition: every allocation site, at a fraction of the count
    // (a step's allocations grow with the window, and the sweep with their
    // square)
    int32_t toks[2] = {1, 1 + 7 % (m.n_vocab - 1)};
    double loss = 0;
    ck(model_lora_train_init(&m, 2, 4.0f, 7), path, "train-init", -1);
    CLEAN_COUNT(n, model_lora_backward(&m, toks, 2, &loss));
    ck(n >= 0, path, "a backward step", -1);
    for (long k = 0; k < n; k++) {
        ck(model_lora_train_init(&m, 2, 4.0f, 7), path, "train-init", k);
        fail_at = k; alloc_calls = 0;
        bool ok = model_lora_backward(&m, toks, 2, &loss);
        fail_at = -1;
        ck(!ok, path, "a backward step fails with its allocation", k);
        ck(m.tape == NULL && m.tape_T == 0, path, "a failed backward releases the tape", k);
        ck(!m.bw_dk_share && !m.bw_dv_share && !m.bw_ple, path,
           "a failed backward releases its shared scratch", k);
        if (ok) break;
    }
    model_lora_free(&m);
    ck(alloc_live == base_live, path, "the interrupted steps' blocks go with the adapter", -1);
    free(saved);
    remove(lpath);
    model_free(&m);
    ck(alloc_live == live0, path, "freeing the model after the adapter sweeps releases every block", -1);
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
        sweep_adapter(argv[i], &p);
    }
    if (g_fail) { fprintf(stderr, "test-model-load-oom FAILED\n"); return 1; }
    fprintf(stderr, "test-model-load-oom: %d fixtures, every allocation of a load and of the "
            "adapter lifecycle failed in turn, no leaks\n",
            argc - 1);
    return 0;
}
