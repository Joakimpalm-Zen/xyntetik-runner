#include "compat.h"
#include "tpool.h"

#include <stdio.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

int main(void) {
    int logical = plat_cpu_count();
    int def = plat_default_thread_count();
    ck(logical >= 1, "logical CPU count is positive");
    ck(def >= 1, "default thread count is positive");
    ck(def <= logical, "default thread count does not exceed logical CPUs");
    ck(def <= PLAT_THREAD_DEFAULT_MAX, "default thread count is capped");

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

    if (g_fail) {
        fprintf(stderr, "thread defaults: FAILED\n");
        return 1;
    }
    puts("thread defaults ok");
    return 0;
}
