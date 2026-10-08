#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "compat.h"
#include "tpool.h"
#ifdef __linux__
#include <sched.h>
#endif

#include <pthread.h>
#include <stdio.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}


// the shared pool: several holders (server slots) run one pool from their
// own threads; tpool_run serializes, every run must see only its own items
typedef struct { long long *sum; } sum_ctx;
static void sum_items(void *ctx, int i0, int i1) {
    sum_ctx *c = ctx;
    long long s = 0;
    for (int i = i0; i < i1; i++) s += i;
    __atomic_add_fetch(c->sum, s, __ATOMIC_RELAXED);
}
typedef struct { tpool *tp; int bad; } caller;
static void *call_pool(void *arg) {
    caller *c = arg;
    for (int r = 0; r < 300; r++) {
        long long sum = 0;
        sum_ctx ctx = { &sum };
        tpool_run(c->tp, sum_items, &ctx, 1000);
        if (sum != 999LL * 1000 / 2) c->bad++;
    }
    return NULL;
}

int main(void) {
    int logical = plat_cpu_count();
    int def = plat_default_thread_count();
    ck(logical >= 1, "logical CPU count is positive");
    ck(def >= 1, "default thread count is positive");
    ck(def <= logical, "default thread count does not exceed logical CPUs");
    ck(def <= PLAT_THREAD_DEFAULT_MAX, "default thread count is capped");
#ifdef __linux__
    // the count follows the affinity MASK, not the machine: a taskset, cpuset
    // or container that narrows it must narrow the default with it (a mask
    // of 32 logical CPUs got the machine's 32-thread default and decoded at
    // 1.7 tok/s where 16 threads gave 9.3, 2026-10-07)
    cpu_set_t saved;
    if (logical >= 2 && sched_getaffinity(0, sizeof(saved), &saved) == 0) {
        cpu_set_t two; CPU_ZERO(&two);
        int picked = 0;
        for (int c = 0; c < CPU_SETSIZE && picked < 2; c++)
            if (CPU_ISSET(c, &saved)) { CPU_SET(c, &two); picked++; }
        if (sched_setaffinity(0, sizeof(two), &two) == 0) {
            ck(plat_cpu_count() == 2, "cpu count follows a 2-cpu affinity mask");
            ck(plat_default_thread_count() <= 2, "default thread count follows the mask");
            sched_setaffinity(0, sizeof(saved), &saved);
        }
    }
#endif

    // The CEILING is the measured part of this policy, and it cannot be
    // exercised on a host with few cores -- the machine this test usually runs
    // on never reaches it. plat_thread_default_for() is the policy as a pure
    // function so the many-core regime is testable everywhere.
    //
    // Two independent sweeps found decode PEAKS at 32 threads and regresses
    // above it: tpbench on a 64-core Zen 5 box (2026-08-13, 65us handoff per
    // run at 32 threads against 138us at 64) and the Blackwell 128-core sweep
    // (2026-08-29, both models peaking at t=32; the old default of 64 cost
    // -41.0% decode and -55.3% prefill on the smaller of them).
    ck(plat_thread_default_for(128) == 32, "128 logical CPUs -> 32, not 64");
    ck(plat_thread_default_for(96) == 32, "96 logical CPUs -> 32");
    ck(plat_thread_default_for(65) == 32, "just past the ceiling -> 32");
    // below the ceiling the halving rule is untouched, so every machine that
    // was not measured keeps exactly the default it had
    ck(plat_thread_default_for(64) == 32, "64 logical CPUs -> 32, unchanged");
    ck(plat_thread_default_for(32) == 16, "32 logical CPUs -> 16, unchanged");
    ck(plat_thread_default_for(8) == 4, "8 logical CPUs -> 4, unchanged");
    ck(plat_thread_default_for(3) == 3, "below 4 the halving does not apply");
    ck(plat_thread_default_for(1) == 1, "a single CPU still gets one thread");
    ck(plat_thread_default_for(0) >= 1, "a bad count never yields zero threads");

#ifdef __APPLE__
    int perf = 0;
    size_t len = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, NULL, 0) == 0 &&
        len == sizeof(perf) && perf > 0) {
        int want = perf > PLAT_THREAD_DEFAULT_MAX
                     ? PLAT_THREAD_DEFAULT_MAX : perf;
        ck(def == want, "Apple asymmetric default uses performance cores");
    }
#endif

    // An over-large request must be CLAMPED AND SAID SO. Silently discarding
    // an explicit -t value is the failure mode the Syntetik-MoE run lost time
    // to: `-t 128` behaved exactly like `-t 64` with nothing on stderr.
    // tpool_create() caps at TP_MAX; here we pin that the cap holds and that
    // the pool reports the clamped size rather than the requested one.
    tpool *big = tpool_create(1024);
    ck(big != NULL, "tpool_create survives an over-large request");
    if (big) {
        ck(tpool_size(big) <= 64, "over-large thread request is clamped");
        ck(tpool_size(big) == 64, "clamp lands on TP_MAX, not something else");
        tpool_destroy(big);
    }

    // the shared-pool size (tpool_shared_threads): the one policy server.c uses
    ck(tpool_shared_threads(16, 1, 16, 0) == 16, "one slot keeps the full count");
    ck(tpool_shared_threads(16, 2, 16, 0) == 15, "two slots leave one CPU of the mask free");
    ck(tpool_shared_threads(8, 4, 16, 0) == 8, "a count under the mask is kept");
    ck(tpool_shared_threads(4, 2, 16, 12) == 12, "the CPU-forced fallback floor raises it");
    ck(tpool_shared_threads(16, 2, 1, 0) == 16, "a single-CPU mask is not cut to zero");
    ck(tpool_shared_threads(0, 2, 8, 0) == 1, "never below one thread");

    // several holders running one retained pool concurrently: every run sees
    // exactly its own items, and the references come back in balance
    tpool *sp = tpool_create(4);
    ck(sp != NULL, "shared pool created");
    if (sp) {
        tpool_retain(sp); tpool_retain(sp);
        caller cs[3] = { { sp, 0 }, { sp, 0 }, { sp, 0 } };
        pthread_t th[3];
        for (int i = 0; i < 3; i++) pthread_create(&th[i], NULL, call_pool, &cs[i]);
        for (int i = 0; i < 3; i++) pthread_join(th[i], NULL);
        ck(cs[0].bad + cs[1].bad + cs[2].bad == 0,
           "three holders x 300 concurrent runs: every sum exact");
        tpool_destroy(sp); tpool_destroy(sp);
        ck(tpool_size(sp) == 4, "the last holder still has a live pool after two releases");
        tpool_destroy(sp);
    }

    if (g_fail) {
        fprintf(stderr, "thread defaults: FAILED\n");
        return 1;
    }
    puts("thread defaults ok");
    return 0;
}
