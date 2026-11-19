// plat_spawn_wait retries waitpid on EINTR (R6.100): a signal delivered to the
// parent while a child runs must not turn the child's exit status into -1.
#include "compat.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
int main(void) {
    printf("spawn-eintr: skip (POSIX EINTR waitpid test)\n");
    return 0;
}
#else
#include <signal.h>
#include <sys/time.h>

static volatile sig_atomic_t alarm_hits;

static void on_alarm(int sig) {
    (void)sig;
    alarm_hits++;
}

int main(void) {
    struct sigaction sa = {0}, old = {0};
    sa.sa_handler = on_alarm;
    sigemptyset(&sa.sa_mask);
    // Explicitly no SA_RESTART so SIGALRM interrupts waitpid with EINTR.
    if (sigaction(SIGALRM, &sa, &old) != 0) {
        fprintf(stderr, "FAIL: sigaction(SIGALRM)\n");
        return 1;
    }

    struct itimerval it = {0};
    it.it_value.tv_usec = 20000;
    it.it_interval.tv_usec = 20000;
    alarm_hits = 0;
    if (setitimer(ITIMER_REAL, &it, NULL) != 0) {
        fprintf(stderr, "FAIL: setitimer\n");
        sigaction(SIGALRM, &old, NULL);
        return 1;
    }

    const char *const argv1[] = { "/bin/sh", "-c", "sleep 0.15; exit 7", NULL };
    int rc1 = plat_spawn_wait(argv1);

    struct itimerval off = {0};
    setitimer(ITIMER_REAL, &off, NULL);
    sig_atomic_t hits1 = alarm_hits;

    if (rc1 != 7 || hits1 < 1) {
        fprintf(stderr, "FAIL: plat_spawn_wait under SIGALRM: rc=%d want=7 hits=%d\n",
                rc1, (int)hits1);
        sigaction(SIGALRM, &old, NULL);
        return 1;
    }

    alarm_hits = 0;
    if (setitimer(ITIMER_REAL, &it, NULL) != 0) {
        fprintf(stderr, "FAIL: setitimer\n");
        sigaction(SIGALRM, &old, NULL);
        return 1;
    }
    const char *const argv2[] = { "/bin/sh", "-c", "read line; sleep 0.15; exit 5", NULL };
    int rc2 = plat_spawn_wait_stdin(argv2, "ok\n", 3);
    setitimer(ITIMER_REAL, &off, NULL);
    sig_atomic_t hits2 = alarm_hits;

    sigaction(SIGALRM, &old, NULL);

    if (rc2 != 5 || hits2 < 1) {
        fprintf(stderr, "FAIL: plat_spawn_wait_stdin under SIGALRM: rc=%d want=5 hits=%d\n",
                rc2, (int)hits2);
        return 1;
    }

    printf("spawn-eintr: ok (plat_spawn_wait hits=%d, stdin hits=%d)\n",
           (int)hits1, (int)hits2);
    return 0;
}
#endif
