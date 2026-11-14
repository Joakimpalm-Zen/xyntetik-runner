// Discovery-registry gate: register/list roundtrip, port rewrite, stale
// sweep, multi-record listing, unregister. Runs against a private HOME
// (or APPDATA on Windows) so it never touches the user's real registry.
#include "instances.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <process.h>
#include <direct.h>
#include <sys/utime.h>
#define getpid _getpid
#define setenv_compat(k, v) _putenv_s(k, v)
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utime.h>
#define setenv_compat(k, v) setenv(k, v, 1)
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); fails++; } \
} while (0)

static void await_sweep(const char *path) {
    for (int i = 0; i < 500; i++) {
        FILE *f = fopen(path, "rb");
        if (!f) return;
        fclose(f);
        plat_sleep_ms(10);
    }
}

static void write_fake(const char *dir, long pid, const char *name) {
    char path[1300];
    snprintf(path, sizeof path, "%s/%ld.json", dir, pid);
    FILE *f = fopen(path, "wb");
    fprintf(f, "{\"pid\": %ld, \"started\": %lld, \"mode\": \"serve\","
               " \"port\": 8123, \"version\": \"test\","
               " \"models\": [{\"name\": \"%s\", \"path\": \"/x/%s\"}]}\n",
            pid, (long long)time(NULL), name, name);
    fclose(f);
}

static void write_lease_record(const char *path, long pid, const char *start,
                               const char *token) {
    char f[1300];
#ifdef _WIN32
    _mkdir(path);
    snprintf(f, sizeof f, "%s\\owner.json", path);
#else
    mkdir(path, 0755);
    snprintf(f, sizeof f, "%s/owner.json", path);
#endif
    FILE *o = fopen(f, "wb");
    if (!o) return;
    if (start)
        fprintf(o, "{\"owner_pid\": %ld, \"owner_start\": \"%s\", \"token\": \"%s\"}",
                pid, start, token);
    else
        fprintf(o, "{\"owner_pid\": %ld, \"owner_start\": null, \"token\": \"%s\"}",
                pid, token);
    fclose(o);
}

// the lease record's mtime set to `stamp`, as test_tray_core pins a file's
static void pin_lease_mtime(const char *path, time_t stamp) {
    char f[1300];
#ifdef _WIN32
    snprintf(f, sizeof f, "%s\\owner.json", path);
    struct __utimbuf64 t = { (__time64_t)stamp, (__time64_t)stamp };
    _utime64(f, &t);
#else
    snprintf(f, sizeof f, "%s/owner.json", path);
    struct utimbuf t = { stamp, stamp };
    utime(f, &t);
#endif
}

// Helper modes for test_stop_latency.py and test_lease_interop.py, which
// holds this C lease against the Python class it ports:
//   test-instances lease-identity PID   print PID's start identity
//   test-instances list                 print live pids through the public list
//   test-instances lease-hold PATH      acquire; print "held" or "refused PID";
//                                       when held, keep it until stdin closes
static int lease_helper(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "list")) {
        int n = 0;
        instance_rec *records = instances_list(&n);
        for (int i = 0; i < n; i++) printf("%ld\n", records[i].pid);
        instances_list_free(records, n);
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "lease-identity")) {
        char id[96];
        if (!runner_lease_start_identity(atol(argv[2]), id, sizeof id)) return 1;
        printf("%s\n", id);
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "lease-hold")) {
        runner_lease l;
        if (!runner_lease_acquire(&l, argv[2])) {
            printf("refused %ld\n", runner_lease_holder(argv[2]));
            fflush(stdout);
            return 0;
        }
        printf("held\n");
        fflush(stdout);
        while (getchar() != EOF) {}
        runner_lease_release(&l);
        return 0;
    }
    fprintf(stderr, "usage: test-instances [list | lease-identity PID | lease-hold PATH]\n");
    return 2;
}

#ifndef _WIN32
static double cpu_seconds(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (double)r.ru_utime.tv_sec + (double)r.ru_stime.tv_sec
         + ((double)r.ru_utime.tv_usec + (double)r.ru_stime.tv_usec) / 1e6;
}

// This binary as the sweep helper: before it sweeps, write down what it
// started with. A process keeps its CPU time across exec, so whatever ran
// between the caller's spawn and this line is in the figure.
static void report_start(void) {
    const char *out = getenv("RUNNER_TEST_SWEEP_REPORT");
    if (!out) return;
    double cpu = cpu_seconds();
    const char *probe = getenv("RUNNER_TEST_SWEEP_PROBE_FD");
    int fd = probe ? atoi(probe) : -1;
    // read before fopen below: a new stream could take the probe's number
    int stdin_open = fcntl(0, F_GETFD) != -1;
    int probe_open = fd >= 0 && fcntl(fd, F_GETFD) != -1;
    int leader = getsid(0) == getpid();
    // the sweep's own first act, for a kernel whose spawn cannot promise it
    plat_close_inherited();
    int probe_left = fd >= 0 && fcntl(fd, F_GETFD) != -1;
    char tmp[1300];
    snprintf(tmp, sizeof tmp, "%s.tmp", out);
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    fprintf(f, "%.6f %d %d %d %d\n", cpu, stdin_open, probe_open, probe_left, leader);
    fclose(f);
    rename(tmp, out);
}

// The sweep helper starts on every registration, so what it costs is paid by
// every run (R6.75). It closed one descriptor number after another up to the
// limit before it exec'd: 1,048,576 calls where the limit is that high, and
// short runs started helpers faster than the machine retired them.
static void helper_starts_clean(const char *home) {
    struct rlimit old, lim;
    if (getrlimit(RLIMIT_NOFILE, &old) != 0) return;
    static const rlim_t want[] = { 1 << 20, 1 << 18, 1 << 16 };
    for (size_t i = 0; i < sizeof want / sizeof want[0]; i++) {
        if (old.rlim_cur >= want[i]) break;
        if (old.rlim_max != RLIM_INFINITY && want[i] > old.rlim_max) continue;
        lim = old;
        lim.rlim_cur = want[i];
        if (setrlimit(RLIMIT_NOFILE, &lim) == 0) break;
    }
    long limit = sysconf(_SC_OPEN_MAX);

    // what one close() per possible descriptor costs here. Nothing of this
    // test's is open from `far` up: the probe below is made after it.
    int far = limit > 2000 ? 1000 : (int)(limit / 2);
    double before = cpu_seconds();
    for (long fd = far; fd < limit; fd++) close((int)fd);
    double loop = cpu_seconds() - before;

    int p[2];
    int probe = pipe(p) == 0 ? fcntl(p[1], F_DUPFD, far) : -1;
    CHECK(probe >= far, "a descriptor without close-on-exec, far from the low numbers");
    char report[1300], num[32];
    snprintf(report, sizeof report, "%s/sweep-report", home);
    snprintf(num, sizeof num, "%d", probe);
    setenv("RUNNER_TEST_SWEEP_REPORT", report, 1);
    setenv("RUNNER_TEST_SWEEP_PROBE_FD", num, 1);
    CHECK(plat_spawn_self_detached("--internal-instance-sweep"), "the helper is launched");
    FILE *f = NULL;
    for (int i = 0; i < 1000 && !(f = fopen(report, "rb")); i++) plat_sleep_ms(10);
    unsetenv("RUNNER_TEST_SWEEP_REPORT");
    unsetenv("RUNNER_TEST_SWEEP_PROBE_FD");
    double cpu = -1;
    int stdin_open = 0, probe_open = 1, probe_left = 1, leader = 0;
    CHECK(f && fscanf(f, "%lf %d %d %d %d", &cpu, &stdin_open, &probe_open, &probe_left,
                      &leader) == 5,
          "the helper reports how it started");
    if (f) fclose(f);
    // the control: the probe does see a descriptor that is open
    CHECK(stdin_open, "the helper has a standard input");
    CHECK(!probe_left, "the helper holds no descriptor of its caller's when it sweeps");
#ifdef __APPLE__
    // here the spawn itself hands none over
    CHECK(!probe_open, "the helper inherits no descriptor of its caller's");
#endif
    CHECK(leader, "the helper leads its own session");
    printf("[sweep] descriptor limit %ld: one close per number costs %.1f ms of CPU, "
           "the helper had used %.1f ms when it started (inherited at the spawn: %s)\n",
           limit, loop * 1e3, cpu * 1e3, probe_open ? "yes" : "no");
    // armed where the loop is dear enough to tell from a process start
    if (loop >= 0.040)
        CHECK(cpu >= 0 && cpu < loop / 2,
              "the helper does not pay for every descriptor number the limit allows");
    else
        printf("[sweep] the cost check is not armed at this limit\n");
    if (probe >= 0) close(probe);
    if (p[0] >= 0) { close(p[0]); close(p[1]); }
    setrlimit(RLIMIT_NOFILE, &old);
    remove(report);
}

// Where the kernel cannot close everything in one call at the spawn, the
// helper closes what it inherited itself. Run in a child: it closes this
// test's descriptors too.
static void inherited_descriptors_close(void) {
    int p[2];
    if (pipe(p) != 0) { CHECK(0, "a pipe"); return; }
    long limit = sysconf(_SC_OPEN_MAX);
    int high = fcntl(p[1], F_DUPFD, limit > 2000 ? 1000 : (int)(limit / 2));
    pid_t child = fork();
    if (child == 0) {
        // the control: all three are open before the call
        if (fcntl(p[0], F_GETFD) == -1 || fcntl(p[1], F_GETFD) == -1
            || fcntl(high, F_GETFD) == -1) _exit(2);
        plat_close_inherited();
        if (fcntl(p[0], F_GETFD) != -1 || fcntl(p[1], F_GETFD) != -1
            || fcntl(high, F_GETFD) != -1) _exit(3);
        for (int i = 0; i < 3; i++) if (fcntl(i, F_GETFD) == -1) _exit(4);
        _exit(0);
    }
    int status = -1;
    waitpid(child, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) != 2, "the descriptors were open first");
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) != 3,
          "every descriptor above the standard three is closed, a high one too");
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) != 4, "the standard three stay open");
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the child ran to its end");
    close(p[0]); close(p[1]);
    if (high >= 0) close(high);
}
#endif

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--internal-instance-sweep")) {
#ifndef _WIN32
        report_start();
#endif
        return instances_sweep_worker();
    }
    if (argc > 1) return lease_helper(argc, argv);
    char tmpl[512];
#ifdef _WIN32
    snprintf(tmpl, sizeof tmpl, "%s\\xyntetik-test-%ld", getenv("TEMP"), (long)getpid());
    _mkdir(tmpl);
    setenv_compat("APPDATA", tmpl);
#else
    snprintf(tmpl, sizeof tmpl, "/tmp/xyntetik-test-%ld", (long)getpid());
    mkdir(tmpl, 0755);
    setenv_compat("HOME", tmpl);
#endif

    const char *dir = instances_dir();
    CHECK(dir != NULL, "instances_dir resolves and creates the directory");
    if (!dir) return 1;

#ifndef _WIN32
    helper_starts_clean(tmpl);
    inherited_descriptors_close();
#endif

    // 1. register + list roundtrip
    const char *const names[] = { "modelA.gguf", "modelB.gguf" };
    const char *const paths[] = { "/m/modelA.gguf", "/m/modelB.gguf" };
    CHECK(instances_register("cli", 0, names, paths, 2), "register succeeds");
    int n = 0;
    instance_rec *r = instances_list(&n);
    CHECK(n == 1, "one live record after register");
    if (n == 1) {
        CHECK(r[0].pid == (long)getpid(), "record carries our pid");
        uint64_t live_start = 0;
        CHECK(r[0].procstart > 0 &&
              plat_pid_start_time(r[0].pid, &live_start) &&
              r[0].procstart == live_start,
              "record carries the owner's process-creation stamp");
        CHECK(strcmp(r[0].mode, "cli") == 0, "mode roundtrips");
        CHECK(r[0].port == 0, "port roundtrips");
        CHECK(r[0].n_models == 2, "both models roundtrip");
        CHECK(r[0].n_models == 2 && strcmp(r[0].model_names[1], "modelB.gguf") == 0,
              "model name content roundtrips");
        CHECK(r[0].version[0] != 0, "version present");
    }
    instances_list_free(r, n);

    // A live PID is not sufficient ownership proof: after a crash and PID
    // reuse, an old record would otherwise identify an unrelated process as a
    // runner and the tray's Stop action could signal it. A recorded creation
    // time that disagrees with the live process makes the record stale.
    {
        uint64_t start = 0;
        CHECK(plat_pid_start_time((long)getpid(), &start) && start > 0,
              "current process creation time is available");
        char reused[1300];
        snprintf(reused, sizeof reused, "%s/reused-pid.json", dir);
        FILE *f = fopen(reused, "wb");
        fprintf(f, "{\"pid\": %ld, \"procstart\": %llu, \"started\": 1,"
                   " \"mode\": \"serve\", \"port\": 8124,"
                   " \"version\": \"t\", \"models\": []}\n",
                (long)getpid(), (unsigned long long)(start + 1));
        fclose(f);
        r = instances_list(&n);
        bool found_reused = false;
        for (int i = 0; i < n; i++)
            if (r[i].port == 8124) found_reused = true;
        CHECK(!found_reused, "reused-PID record is not attributed to its new owner");
        instances_list_free(r, n);
        await_sweep(reused);
        f = fopen(reused, "rb");
        CHECK(f == NULL, "reused-PID record is swept");
        if (f) fclose(f);
    }

    // 2. set_port rewrites in place
    CHECK(instances_set_port(9099), "set_port succeeds");
    r = instances_list(&n);
    CHECK(n == 1 && r[0].port == 9099, "port rewrite visible to readers");
    instances_list_free(r, n);

    // 3. a second record for a live pid is listed alongside ours
    write_fake(dir, (long)getpid() + 2000000, "ghost.gguf"); // dead pid
    char alive_path[1300];
    // reuse OUR pid under a different filename: filename is identity, pid
    // field is what liveness checks — this models a second live process
    snprintf(alive_path, sizeof alive_path, "%s/%ld.json", dir, (long)getpid() + 1);
    {
        FILE *f = fopen(alive_path, "wb");
        fprintf(f, "{\"pid\": %ld, \"started\": 1, \"mode\": \"serve\","
                   " \"port\": 8123, \"version\": \"t\", \"models\": []}\n",
                (long)getpid());
        fclose(f);
    }
    r = instances_list(&n);
    CHECK(n == 2, "live foreign record listed, dead one swept");
    bool found_serve = false;
    for (int i = 0; i < n; i++)
        if (strcmp(r[i].mode, "serve") == 0 && r[i].port == 8123) found_serve = true;
    CHECK(found_serve, "foreign serve record fields intact");
    instances_list_free(r, n);

    // 4. the dead record's file is gone (swept, not just skipped)
    {
        char dead[1300];
        snprintf(dead, sizeof dead, "%s/%ld.json", dir, (long)getpid() + 2000000);
        await_sweep(dead);
        FILE *f = fopen(dead, "rb");
        CHECK(f == NULL, "stale record file deleted by the sweep");
        if (f) fclose(f);
    }

    // 5. unregister removes ours; double-unregister is safe
    instances_unregister();
    instances_unregister();
    remove(alive_path);
    r = instances_list(&n);
    CHECK(n == 0, "no records after unregister + cleanup");
    instances_list_free(r, n);

    // 5b. a registering process sweeps what dead processes left: their
    // records and the temporaries of a write that never reached its rename.
    // A machine that never runs the tray has no other reader to do it.
    {
        long deadpid = (long)getpid() + 2000000;
        char dead[1300], deadtmp[1300], livetmp[1300];
        write_fake(dir, deadpid, "ghost.gguf");
        snprintf(dead, sizeof dead, "%s/%ld.json", dir, deadpid);
        snprintf(deadtmp, sizeof deadtmp, "%s/%ld.json.tmp", dir, deadpid);
        // the parent, not this process: our own registration writes and
        // renames `<our pid>.json.tmp`
#ifdef _WIN32
        livetmp[0] = 0;   // no getppid here; the dead-pid half still runs
#else
        snprintf(livetmp, sizeof livetmp, "%s/%ld.json.tmp", dir, (long)getppid());
#endif
        FILE *f = fopen(deadtmp, "wb"); if (f) fclose(f);
        if (livetmp[0]) { f = fopen(livetmp, "wb"); if (f) fclose(f); }
        CHECK(instances_register("cli", 0, NULL, NULL, 0), "register sweeps");
        await_sweep(dead);
        await_sweep(deadtmp);
        f = fopen(dead, "rb");
        CHECK(f == NULL, "a dead process's record is swept at registration");
        if (f) fclose(f);
        f = fopen(deadtmp, "rb");
        CHECK(f == NULL, "a dead process's temporary is swept at registration");
        if (f) fclose(f);
        if (livetmp[0]) {
            f = fopen(livetmp, "rb");
            CHECK(f != NULL, "a live process's temporary is left alone");
            if (f) fclose(f);
            remove(livetmp);
        }
        instances_unregister();
    }

    // 6. corrupt record is swept, not fatal
    {
        char junk[1300];
        snprintf(junk, sizeof junk, "%s/999.json", dir);
        FILE *f = fopen(junk, "wb");
        fprintf(f, "not json at all");
        fclose(f);
        r = instances_list(&n);
        CHECK(n == 0, "corrupt record ignored");
        instances_list_free(r, n);
        await_sweep(junk);
        f = fopen(junk, "rb");
        CHECK(f == NULL, "corrupt record swept");
        if (f) fclose(f);
    }

    // 6b. a record's numbers are checked before they are cast (R6.68.10): a
    // pid, started, procstart or port that is not a whole number in its range
    // is refused like a corrupt record, never converted. With this process's
    // own pid the record would otherwise be live, so a converted started or
    // port used to list it; a converted 1e300 was undefined behaviour, which
    // test-instances-ubsan stops on (float-cast-overflow).
    {
        const char *fields[] = {
            "\"pid\": 1e300",
            "\"pid\": %ld, \"started\": 1e300",
            "\"pid\": %ld, \"procstart\": 1e300",
            "\"pid\": %ld, \"port\": 1e300",
            "\"pid\": %ld, \"port\": 8080.5",
            "\"pid\": %ld, \"port\": -1",
            "\"pid\": %ld, \"started\": \"soon\"",
            // infinite and NaN values: json.c refuses the overflowing literal
            // and has no NaN spelling, so these die at the parse; pinned so a
            // parser that ever accepted them still meets a refusal here
            "\"pid\": 1e999",
            "\"pid\": %ld, \"port\": -1e999",
            "\"pid\": NaN",
        };
        for (size_t i = 0; i < sizeof fields / sizeof fields[0]; i++) {
            char path[1300], body[256], rec[512];
            snprintf(path, sizeof path, "%s/bad-%zu.json", dir, i);
            snprintf(body, sizeof body, fields[i], (long)getpid());
            snprintf(rec, sizeof rec, "{%s, \"mode\": \"serve\", \"version\": \"t\"}", body);
            FILE *f = fopen(path, "wb");
            CHECK(f != NULL, "a bad record can be written");
            if (!f) continue;
            fputs(rec, f);
            fclose(f);
            r = instances_list(&n);
            CHECK(n == 0, rec);
            instances_list_free(r, n);
            remove(path);
        }
    }

#ifndef _WIN32
    // 7. a zombie answers kill(pid, 0) but must count as dead: a record
    // pointing at one would otherwise be an unsweepable ghost row that
    // Stop can never clear (found live on the M1, 2026-08-05)
    {
        pid_t z = fork();
        if (z == 0) _exit(0);
        // give it time to die; do NOT reap — the zombie is the fixture
        for (int t = 0; t < 100 && instance_pid_alive((long)z); t++) {
            struct timespec ts = { 0, 10 * 1000 * 1000 };
            nanosleep(&ts, NULL);
        }
        CHECK(!instance_pid_alive((long)z), "zombie counts as dead");
        write_fake(dir, (long)z, "ghost.gguf");
        r = instances_list(&n);
        CHECK(n == 0, "zombie record not listed");
        instances_list_free(r, n);
        char zp[1300];
        snprintf(zp, sizeof zp, "%s/%ld.json", dir, (long)z);
        await_sweep(zp);
        FILE *f = fopen(zp, "rb");
        CHECK(f == NULL, "zombie record swept");
        if (f) fclose(f);
        waitpid(z, NULL, 0);
    }
#endif

    // 8. the startup lease (R4.12.24), lease.py's rules in C
    {
        char lp[1100];
        CHECK(runner_lease_path(8123, lp, sizeof lp), "lease path resolves");
        CHECK(strstr(lp, "leases") && strstr(lp, "runner-8123.pid"),
              "lease path is <root>/leases/runner-<port>.pid");
        char me[96];
        CHECK(runner_lease_start_identity((long)getpid(), me, sizeof me),
              "this process has a start identity");

        runner_lease a, b;
        CHECK(runner_lease_acquire(&a, lp), "a free lease is claimed");
        CHECK(runner_lease_holder(lp) == (long)getpid(), "the record names this process");
        CHECK(!runner_lease_acquire(&b, lp), "a held lease refuses a second claim");
        b.held = true;                  // a release without the token is ignored
        runner_lease_release(&b);
        CHECK(runner_lease_holder(lp) == (long)getpid(), "a foreign token releases nothing");
        runner_lease_release(&a);
        CHECK(runner_lease_holder(lp) == 0, "the owner's release removes the record");
        CHECK(runner_lease_acquire(&b, lp), "a released lease is claimed again");
        runner_lease_release(&b);

        // a live pid whose start identity moved: the pid was reused
        write_lease_record(lp, (long)getpid(), "not this process", "t1");
        CHECK(runner_lease_acquire(&a, lp), "a reused pid's record is stale");
        runner_lease_release(&a);

        // unverifiable ownership: honoured while fresh, then it ages out
        write_lease_record(lp, (long)getpid(), NULL, "t2");
        CHECK(!runner_lease_acquire(&a, lp), "a fresh unverifiable record is honoured");
        setenv_compat("RUNNER_UNVERIFIED_LEASE_TTL", "0");
        CHECK(runner_lease_acquire(&a, lp), "an unverifiable record ages out");
        setenv_compat("RUNNER_UNVERIFIED_LEASE_TTL", "900");
        runner_lease_release(&a);
        // a record whose mtime reads ahead of the clock is no fresher than
        // one written now. Linux can stamp a file finer than the coarse
        // clock time() reads, so a record written this instant can read a
        // second ahead: its age came out -1, below a TTL of 0, and the
        // record was honoured (a CI flake, 2026-10-31)
        write_lease_record(lp, (long)getpid(), NULL, "t3");
        pin_lease_mtime(lp, time(NULL) + 1);
        setenv_compat("RUNNER_UNVERIFIED_LEASE_TTL", "0");
        CHECK(runner_lease_acquire(&a, lp), "a record dated ahead of the clock ages out");
        setenv_compat("RUNNER_UNVERIFIED_LEASE_TTL", "900");
        runner_lease_release(&a);

#ifndef _WIN32
        // a dead owner's record is moved aside and the claim retried
        pid_t d = fork();
        if (d == 0) _exit(0);
        waitpid(d, NULL, 0);
        write_lease_record(lp, (long)d, me, "t3");
        CHECK(runner_lease_acquire(&a, lp), "a dead owner's record is stale");
        runner_lease_release(&a);
#endif
    }

    if (fails) { fprintf(stderr, "test_instances: %d FAILURES\n", fails); return 1; }
    printf("instances registry tests ok\n");
    return 0;
}
