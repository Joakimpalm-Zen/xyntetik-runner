// --mlock wires the hot ranges of the weights and leaves a row-gathered table
// pageable, once per shared mapping.
//
//   1. model_lock_plan, against page boundaries worked out by hand: a cold
//      range keeps only its whole pages, so a page it shares with a hot
//      neighbour stays locked.
//   2. On POSIX, the kernel's own locked-memory accounting (RLIMIT_MEMLOCK,
//      enforced cumulatively on Linux and macOS) is the anchor for what a
//      load actually wired: exactly the hot pages of an E-series fixture,
//      nothing left behind by a lock that failed halfway, and a shared lock
//      that lives until the last instance holding the weights is freed.
//
// The fixture is test-es-mlock.gguf: its per-layer token table sits between
// the layers and the BF16-sized projection, so the hot set is two ranges.
#include "runner.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/resource.h>
#endif

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else       printf("ok: %s\n", what);
}

static int same(const model_lock_range *r, size_t n, const model_lock_range *want,
                size_t n_want) {
    if (n != n_want) return 0;
    for (size_t i = 0; i < n; i++)
        if (r[i].off != want[i].off || r[i].len != want[i].len) return 0;
    return 1;
}

static void test_plan(void) {
    enum { P = 4096 };
    model_lock_range r[4];
    size_t n;

    n = model_lock_plan(10000, P, NULL, 0, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 0, 10000 } }, 1),
       "plan: no cold range locks the whole mapping");

    // cold 5000..14000: whole pages 8192..12288 only; pages 1 and 3 are
    // shared with hot neighbours and stay locked
    n = model_lock_plan(5 * P, P, &(model_lock_range){ 5000, 9000 }, 1, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 0, 8192 }, { 12288, 8192 } }, 2),
       "plan: a mid-mapping cold range keeps its boundary pages hot");

    n = model_lock_plan(5 * P, P, &(model_lock_range){ 4096, 8192 }, 1, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 0, 4096 }, { 12288, 8192 } }, 2),
       "plan: a page-aligned cold range drops exactly its pages");

    n = model_lock_plan(5 * P, P, &(model_lock_range){ 5000, 3000 }, 1, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 0, 5 * P } }, 1),
       "plan: a cold range without a whole page leaves the mapping whole");

    // ends at the mapping's unaligned end: its last partial page has no hot
    // neighbour, so it goes too
    n = model_lock_plan(20000, P, &(model_lock_range){ 6000, 14000 }, 1, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 0, 8192 } }, 1),
       "plan: a cold tail drops through the mapping's end");

    n = model_lock_plan(5 * P, P, &(model_lock_range){ 16384, 99999 }, 1, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 0, 16384 } }, 1),
       "plan: a cold range past the mapping is clipped to it");

    n = model_lock_plan(5 * P, P, &(model_lock_range){ 0, 4096 }, 1, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 4096, 16384 } }, 1),
       "plan: a cold head starts the lock at its end");

    n = model_lock_plan(5 * P, P, &(model_lock_range){ 0, 5 * P }, 1, r, 4);
    ck(n == 0, "plan: an all-cold mapping has nothing to lock");

    // two overlapping cold ranges merge: 4096..12288 and 8192..16384
    n = model_lock_plan(5 * P, P,
                        (model_lock_range[]){ { 8192, 8192 }, { 4096, 8192 } }, 2, r, 4);
    ck(same(r, n, (model_lock_range[]){ { 0, 4096 }, { 16384, 4096 } }, 2),
       "plan: overlapping cold ranges merge, in any order");

    n = model_lock_plan(5 * P, P, &(model_lock_range){ 5000, 9000 }, 1, r, 1);
    ck(n == 2 && r[0].off == 0 && r[0].len == 8192,
       "plan: a short output still reports how many ranges there are");

    ck(model_lock_plan(5 * P, 0, NULL, 0, r, 4) == 0 &&
       model_lock_plan(5 * P, 3000, NULL, 0, r, 4) == 0 &&
       model_lock_plan(0, P, NULL, 0, r, 4) == 0,
       "plan: no page size, a non-power-of-two page or an empty mapping plan nothing");
}

#ifndef _WIN32
// Can this process lock `n` more bytes right now? Asks the kernel with
// memory of its own, so the answer is the limit minus whatever is wired.
static int can_lock(size_t n) {
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) return -1;
    int ok = mlock(p, n) == 0;
    if (ok) munlock(p, n);
    munmap(p, n);
    return ok;
}

static model_params mlock_params(bool lock) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;     // the lock is about the host mapping
    p.n_threads = 1;
    p.n_ctx = 64;
    p.n_batch = 8;
    p.mlock = lock;
    return p;
}

static void test_lifecycle(const char *path) {
    size_t page = plat_page_size();
    gguf_file g;
    if (!page || !gguf_open(&g, path)) { ck(0, "open the E-series fixture"); return; }
    gguf_tensor *t = gguf_find_tensor(&g, "per_layer_token_embd.weight");
    size_t size;
    const uint8_t *map = gguf_map_part(&g, 0, &size);
    if (!t || !map || gguf_map_count(&g) != 1) {
        ck(0, "fixture has a per-layer token table in one mapping");
        gguf_close(&g);
        return;
    }
    // The hot pages by hand: up to the table's first whole page, and from
    // its last whole page to the end of the mapping.
    size_t lo = (size_t)((const uint8_t *)t->data - map), hi = lo + t->nbytes;
    size_t head = (lo + page - 1) / page * page;
    size_t tail = (size + page - 1) / page * page - hi / page * page;
    size_t hot = head + tail, whole = (size + page - 1) / page * page;
    gguf_close(&g);
    printf("note: page %zu, hot %zu of %zu bytes (head %zu, tail %zu)\n",
           page, hot, whole, head, tail);
    ck(tail > page && whole - hot > 4 * page,
       "fixture: hot pages on both sides of a table of several pages");

    struct rlimit old;
    if (getrlimit(RLIMIT_MEMLOCK, &old) != 0) { ck(0, "getrlimit"); return; }
    const size_t slack = 4 * page;
    if (old.rlim_max != RLIM_INFINITY && old.rlim_max < hot + slack) {
        printf("skip: RLIMIT_MEMLOCK hard limit %llu is below the fixture's hot set\n",
               (unsigned long long)old.rlim_max);
        return;
    }
    struct rlimit lim = old;

    // ---- a lock that fails halfway leaves nothing wired. The head range
    // fits under this limit and the tail does not, so a missing rollback
    // would leave the head counted and the probe below refused.
    lim.rlim_cur = head + page;
    if (setrlimit(RLIMIT_MEMLOCK, &lim) != 0) { ck(0, "setrlimit"); return; }
    if (can_lock(head + 2 * page)) {
        printf("skip: RLIMIT_MEMLOCK is not enforced for this process (privileged?)\n");
        setrlimit(RLIMIT_MEMLOCK, &old);
        return;
    }
    model_t a;
    model_params p = mlock_params(true);
    ck(model_load(&a, path, &p), "a refused lock does not fail the load");
    ck(!a.weights_locked, "a refused lock is not reported as locked");
    ck(can_lock(head + page) == 1, "a lock refused halfway is rolled back in full");
    model_free(&a);

    // ---- the hot set, and only the hot set, is wired.
    lim.rlim_cur = hot + slack;
    setrlimit(RLIMIT_MEMLOCK, &lim);
    model_t x, y, z;
    ck(model_load(&x, path, &p) && x.weights_locked, "x locks the hot set");
    ck(can_lock(slack) == 1 && can_lock(slack + page) == 0,
       "exactly the hand-computed hot pages are wired");

    // ---- a second instance shares the lock rather than taking another.
    ck(model_load(&y, path, &p) && y.weights_locked, "y shares x's lock");
    ck(x.layers == y.layers, "x and y share one set of weights");
    ck(can_lock(slack) == 1 && can_lock(slack + page) == 0,
       "the shared lock is counted once");
    // An instance that joins locked weights without --mlock must still see
    // them as locked: the flag is what stops its GPU load handing the host
    // pages back, which on Windows is the call that unlocks them.
    model_params plain = mlock_params(false);
    model_t w;
    ck(model_load(&w, path, &plain) && w.weights_locked,
       "an instance joining locked weights knows they are locked");
    model_free(&w);
    model_free(&x);
    ck(can_lock(slack + page) == 0,
       "freeing the instance that locked keeps the lock while y holds the weights");
    model_free(&y);
    ck(can_lock(hot + slack) == 1, "the last holder's free unlocks");

    // ---- an instance loaded without --mlock keeps a later lock alive too.
    ck(model_load(&z, path, &plain) && !z.weights_locked, "z loads unlocked");
    ck(can_lock(hot + slack) == 1, "nothing is wired without --mlock");
    ck(model_load(&x, path, &p) && x.weights_locked, "x locks z's shared weights");
    model_free(&x);
    ck(can_lock(slack + page) == 0, "z keeps the lock x took");
    model_free(&z);
    ck(can_lock(hot + slack) == 1, "and its free releases it");

    setrlimit(RLIMIT_MEMLOCK, &old);
}
#endif

int main(int argc, char **argv) {
    test_plan();
#ifndef _WIN32
    test_lifecycle(argc > 1 ? argv[1] : "test-es-mlock.gguf");
#else
    (void)argc; (void)argv;
    printf("skip: no locked-memory limit to measure against on Windows\n");
#endif
    if (g_fail) return 1;
    printf("test-mlock: all checks passed\n");
    return 0;
}
