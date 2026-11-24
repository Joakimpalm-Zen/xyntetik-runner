// Windows-visible manifest replace test and planted-symlink refusal for
// kvsnap_write_manifest_file() (R6.101).
#include "kvsnap.h"
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#else
#include <io.h>
#include <windows.h>
#endif

static int g_fail = 0;
static void ck(bool cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

#ifdef _WIN32
// R6.118: many takers of one publish lock at once. Each thread opens its own
// handle, as each process does, so they contend as processes do.
#define PL_THREADS 8
#define PL_ITERS 150
static volatile LONG pl_errors, pl_inside, pl_overlap, pl_taken;
static DWORD WINAPI pl_worker(LPVOID arg) {
    (void)arg;
    for (int i = 0; i < PL_ITERS; i++) {
        void *lk = NULL;
        plat_lock_status st = plat_publish_lock("test-kvsnap-contend.out", &lk);
        if (st != PLAT_LOCK_OK || !lk) { InterlockedIncrement(&pl_errors); continue; }
        if (InterlockedIncrement(&pl_inside) != 1) InterlockedIncrement(&pl_overlap);
        InterlockedIncrement(&pl_taken);
        if ((i & 7) == 0) Sleep(1);
        InterlockedDecrement(&pl_inside);
        plat_publish_unlock(lk);
    }
    return 0;
}

typedef struct {
    HANDLE ready;
    DWORD  hold_ms;
    DWORD  open_err;
} share_holder_ctx;

// Hold the lock path with sharing disabled to simulate a scanner holding a
// new lock file briefly while plat_publish_lock attempts to open it.
static DWORD WINAPI share_holder(LPVOID arg) {
    share_holder_ctx *ctx = arg;
    HANDLE h = CreateFileW(L"test-kvsnap-share.out.publish.lock",
                           GENERIC_READ, 0, NULL, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        ctx->open_err = GetLastError();
        SetEvent(ctx->ready);
        return 1;
    }
    SetEvent(ctx->ready);
    Sleep(ctx->hold_ms);
    CloseHandle(h);
    return 0;
}

// Run plat_publish_lock(target, lk) with stderr captured into buf[0..cap) and
// echoed afterward so CI keeps diagnostics visible.
static plat_lock_status lock_capture_stderr(const char *target, void **lk,
                                            char *buf, size_t cap,
                                            long *elapsed_ms) {
    if (cap) buf[0] = '\0';
    fflush(stderr);
    int err_fd = _fileno(stderr);
    int saved = err_fd >= 0 ? _dup(err_fd) : -1;
    FILE *capf = plat_fopen("test-kvsnap-stderr.tmp", "wb");
    if (saved >= 0 && capf) _dup2(_fileno(capf), err_fd);
    double t0 = plat_now();
    plat_lock_status st = plat_publish_lock(target, lk);
    if (elapsed_ms) *elapsed_ms = (long)((plat_now() - t0) * 1000.0);
    fflush(stderr);
    if (saved >= 0) {
        if (capf) _dup2(saved, err_fd);
        _close(saved);
    }
    if (capf) fclose(capf);
    FILE *rf = plat_fopen("test-kvsnap-stderr.tmp", "rb");
    if (rf) {
        if (cap > 1) {
            size_t n = fread(buf, 1, cap - 1, rf);
            buf[n] = '\0';
        }
        fclose(rf);
    }
    remove("test-kvsnap-stderr.tmp");
    if (cap && buf[0]) fprintf(stderr, "%s", buf);
    return st;
}

static void check_lasting_denied(const char *target, const char *label) {
    void *lk = NULL;
    char errbuf[512];
    long elapsed_ms = 0;
    plat_lock_status st = lock_capture_stderr(target, &lk, errbuf, sizeof errbuf, &elapsed_ms);
    char msg[160];
    snprintf(msg, sizeof msg, "lasting access-denied (%.40s): fails with PLAT_LOCK_ERROR", label);
    ck(st == PLAT_LOCK_ERROR && lk == NULL, msg);
    snprintf(msg, sizeof msg, "lasting access-denied (%.40s): bounded wait %ld ms (want 3500..30000 ms)",
             label, elapsed_ms);
    ck(elapsed_ms >= 3500 && elapsed_ms < 30000, msg);
    snprintf(msg, sizeof msg, "lasting access-denied (%.40s): prints step and Windows error 5", label);
    ck(strstr(errbuf, "opening the lock file") != NULL &&
       strstr(errbuf, "Windows error 5") != NULL, msg);
}
#endif

static int kv_fail_step_b(const char *tmp, const char *out) {
    (void)tmp; (void)out;
    return -1;
}

static bool file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) { fclose(f); return true; }
    return false;
}

static bool has_partial(const char *prefix) {
    size_t pn = strlen(prefix);
    plat_dir *dp = plat_opendir(".");
    if (!dp) return false;
    bool found = false;
    for (const char *nm; (nm = plat_readdir(dp)) != NULL; ) {
        if (strncmp(nm, prefix, pn) == 0) { found = true; break; }
    }
    plat_closedir(dp);
    return found;
}

int main(void) {
    const char *man = "test-kvsnap-replace.kv.json";
    const char *pfx = "test-kvsnap-replace.kv.json.partial";
    remove(man);

    const char *body1 = "{\"name\":\"first\"";
    ck(kvsnap_write_manifest_file(man, body1, strlen(body1), NULL),
       "initial kvsnap manifest write succeeds");
    ck(!has_partial(pfx), "initial manifest write leaves no .partial-*");

    // On Windows a plain rename refuses an existing destination; plat_replace_file
    // overwrites the existing manifest cleanly.
    const char *body2 = "{\"name\":\"second\"";
    ck(kvsnap_write_manifest_file(man, body2, strlen(body2), NULL),
       "second kvsnap manifest write replaces existing destination");
    ck(!has_partial(pfx), "second manifest write leaves no .partial-*");

    size_t got_n = 0;
    char *got = plat_read_file(man, 4096, &got_n);
    ck(got && strstr(got, "\"second\"") != NULL && strstr(got, "\"chain\"") != NULL,
       "replaced manifest holds second payload and chain");
    free(got);

#ifndef _WIN32
    const char *lnk = "test-kvsnap-replace.kv.json.partial-fixed";
    const char *victim = "test-kvsnap-replace-victim.txt";
    remove(lnk);
    remove(victim);
    FILE *vf = fopen(victim, "wb");
    if (vf) { fputs("untouched", vf); fclose(vf); }
    if (symlink(victim, lnk) == 0) {
        const char *body3 = "{\"name\":\"third\"";
        setenv("RUNNER_TEST_PARTIAL_SUFFIX", "fixed", 1);
        bool wr = kvsnap_write_manifest_file(man, body3, strlen(body3), NULL);
        unsetenv("RUNNER_TEST_PARTIAL_SUFFIX");
        ck(!wr, "kvsnap manifest write refuses symlink planted at .partial-fixed");

        size_t vn = 0;
        char *vbuf = plat_read_file(victim, 256, &vn);
        ck(vbuf && vn == 9 && memcmp(vbuf, "untouched", 9) == 0,
           "kvsnap manifest symlink refusal leaves victim untouched");
        free(vbuf);

        size_t mn = 0;
        char *mbuf = plat_read_file(man, 4096, &mn);
        ck(mbuf && strstr(mbuf, "\"second\"") != NULL,
           "kvsnap manifest symlink refusal preserves existing manifest");
        free(mbuf);
        remove(lnk);
    }

    // Legacy-name case (no RUNNER_TEST_PARTIAL_SUFFIX): succeeds without
    // touching `<man>.partial` or overwriting its canary.
    const char *leg_lnk = "test-kvsnap-replace.kv.json.partial";
    remove(leg_lnk);
    vf = fopen(victim, "wb");
    if (vf) { fputs("untouched", vf); fclose(vf); }
    if (symlink(victim, leg_lnk) == 0) {
        unsetenv("RUNNER_TEST_PARTIAL_SUFFIX");
        const char *body4 = "{\"name\":\"legacy_ok\"";
        ck(kvsnap_write_manifest_file(man, body4, strlen(body4), NULL),
           "kvsnap manifest write succeeds with legacy .partial symlink present");
        size_t vn = 0;
        char *vbuf = plat_read_file(victim, 256, &vn);
        ck(vbuf && vn == 9 && memcmp(vbuf, "untouched", 9) == 0,
           "kvsnap legacy .partial canary untouched");
        free(vbuf);
        struct stat lst;
        ck(lstat(leg_lnk, &lst) == 0 && S_ISLNK(lst.st_mode),
           "kvsnap legacy .partial remains a symlink");
        remove(leg_lnk);
    }
    remove(victim);
#endif

    remove(man);

    // Pair publish tests for KV snapshots (.kv + .kv.json via plat_publish_pair):
    // 1. Fresh pair succeeds, cleans temps and lock file.
    const char *kv_f = "test-kvsnap-pair.kv";
    const char *man_f = "test-kvsnap-pair.kv.json";
    const char *kv_t = "test-kvsnap-pair.kv.partial-a";
    const char *man_t = "test-kvsnap-pair.kv.json.partial-a";
    remove(kv_f); remove(man_f); remove(kv_t); remove(man_t);
    FILE *kf = fopen(kv_t, "wb"); if (kf) { fputs("KV_BYTES_1", kf); fclose(kf); }
    FILE *mf = fopen(man_t, "wb"); if (mf) { fputs("{\"n\":1}", mf); fclose(mf); }
    ck(plat_publish_pair(kv_t, kv_f, man_t, man_f, true, NULL, NULL) == 0,
       "kvsnap pair publish succeeds on fresh destination");
    // 2. no_overwrite under lock returns -2 and cleans partials without overwriting.
    kf = fopen(kv_t, "wb"); if (kf) { fputs("KV_BYTES_2", kf); fclose(kf); }
    mf = fopen(man_t, "wb"); if (mf) { fputs("{\"n\":2}", mf); fclose(mf); }
    ck(plat_publish_pair(kv_t, kv_f, man_t, man_f, true, NULL, NULL) == -2,
       "kvsnap pair publish refuses overwrite when destination exists");
    FILE *chk = fopen(kv_t, "rb");
    ck(chk == NULL, "kvsnap refused overwrite unlinks staged kv partial");
    if (chk) fclose(chk);
    // 3. Step B failure on overwrite restores old manifest and keeps staged kv partial.
    kf = fopen(kv_t, "wb"); if (kf) { fputs("KV_BYTES_2", kf); fclose(kf); }
    mf = fopen(man_t, "wb"); if (mf) { fputs("{\"n\":2}", mf); fclose(mf); }
    plat_publish_ops ops = { .install_out = kv_fail_step_b };
    bool kept_kv = false;
    ck(plat_publish_pair(kv_t, kv_f, man_t, man_f, false, &kept_kv, &ops) == -1,
       "kvsnap pair publish reports Step B failure");
    ck(kept_kv, "kvsnap Step B failure preserves staged kv partial");
    size_t kvn = 0, mn = 0;
    char *kvb = plat_read_file(kv_f, 256, &kvn);
    char *mb = plat_read_file(man_f, 256, &mn);
    ck(kvb && kvn == 10 && memcmp(kvb, "KV_BYTES_1", 10) == 0,
       "kvsnap Step B failure preserves old .kv bytes");
    ck(mb && mn == 7 && memcmp(mb, "{\"n\":1}", 7) == 0,
       "kvsnap Step B failure atomically restores old .kv.json bytes");
    free(kvb); free(mb);
    ck(!has_partial("test-kvsnap-pair.kv.json.bak"),
       "kvsnap Step B failure leaves no .bak temp");
    ck(file_exists(kv_t) && !file_exists(man_t),
       "kvsnap Step B failure keeps kv partial and consumes manifest partial");
    remove(kv_t);
    remove(kv_f); remove(man_f);
    remove("test-kvsnap-pair.kv.publish.lock");
#ifdef _WIN32
    // R6.118 / R6.119: main's Windows job failed under contention because opens
    // meeting an in-flight delete in plat_publish_unlock get ERROR_ACCESS_DENIED
    // (or ERROR_SHARING_VIOLATION if a scanner holds a new lock file briefly
    // without sharing). Eight takers, 150 rounds each: every take must succeed
    // and no two may hold it at once.
    {
        HANDLE th[PL_THREADS];
        int started = 0;
        for (int t = 0; t < PL_THREADS; t++) {
            th[t] = CreateThread(NULL, 0, pl_worker, NULL, 0, NULL);
            if (th[t]) started++;
        }
        for (int t = 0; t < PL_THREADS; t++)
            if (th[t]) { WaitForSingleObject(th[t], INFINITE); CloseHandle(th[t]); }
        ck(started == PL_THREADS, "contended lock: eight takers started");
        printf("contended lock: %ld taken, %ld failed, %ld overlaps\n",
               (long)pl_taken, (long)pl_errors, (long)pl_overlap);
        ck(pl_errors == 0, "contended lock: every take of the publish lock succeeds");
        ck(pl_overlap == 0, "contended lock: no two takers hold it at once");
        ck(pl_taken == PL_THREADS * PL_ITERS, "contended lock: 1200 takes in all");
        DeleteFileW(L"test-kvsnap-contend.out.publish.lock");
    }
    // R6.119 (1): transient ERROR_SHARING_VIOLATION (scanner holding lock file
    // without sharing) waits and succeeds once released.
    {
        DeleteFileW(L"test-kvsnap-share.out.publish.lock");
        share_holder_ctx sctx = { .ready = CreateEventW(NULL, TRUE, FALSE, NULL),
                                  .hold_ms = 120, .open_err = 0 };
        HANDLE sth = sctx.ready ? CreateThread(NULL, 0, share_holder, &sctx, 0, NULL) : NULL;
        ck(sctx.ready != NULL && sth != NULL, "sharing violation: holder thread started");
        if (sth) {
            WaitForSingleObject(sctx.ready, INFINITE);
            void *lk = NULL;
            plat_lock_status st = plat_publish_lock("test-kvsnap-share.out", &lk);
            WaitForSingleObject(sth, INFINITE);
            CloseHandle(sth);
            ck(sctx.open_err == 0 && st == PLAT_LOCK_OK && lk != NULL,
               "sharing violation: waits through unshared handle and acquires lock");
            plat_publish_unlock(lk);
        }
        if (sctx.ready) CloseHandle(sctx.ready);
        DeleteFileW(L"test-kvsnap-share.out.publish.lock");
    }
    // R6.119 (2): lasting ERROR_ACCESS_DENIED (read-only lock file and a
    // directory at the lock path) fails after the bounded wait with the named error.
    {
        SetFileAttributesW(L"test-kvsnap-ro.out.publish.lock", FILE_ATTRIBUTE_NORMAL);
        DeleteFileW(L"test-kvsnap-ro.out.publish.lock");
        HANDLE roh = CreateFileW(L"test-kvsnap-ro.out.publish.lock", GENERIC_WRITE, 0, NULL,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_READONLY, NULL);
        ck(roh != INVALID_HANDLE_VALUE, "lasting access-denied: read-only lock file created");
        if (roh != INVALID_HANDLE_VALUE) {
            CloseHandle(roh);
            check_lasting_denied("test-kvsnap-ro.out", "read-only lock file");
            SetFileAttributesW(L"test-kvsnap-ro.out.publish.lock", FILE_ATTRIBUTE_NORMAL);
            DeleteFileW(L"test-kvsnap-ro.out.publish.lock");
        }

        RemoveDirectoryW(L"test-kvsnap-dir.out.publish.lock");
        DeleteFileW(L"test-kvsnap-dir.out.publish.lock");
        BOOL dir_ok = CreateDirectoryW(L"test-kvsnap-dir.out.publish.lock", NULL);
        ck(dir_ok != 0, "lasting access-denied: directory at lock path created");
        if (dir_ok) {
            check_lasting_denied("test-kvsnap-dir.out", "directory at lock path");
            RemoveDirectoryW(L"test-kvsnap-dir.out.publish.lock");
        }
    }
    // R6.119 (5): reparse-point refusal names the reparse point without "Windows error 0".
    {
        DeleteFileW(L"test-kvsnap-sym.out.publish.lock");
        DeleteFileW(L"test-kvsnap-sym-target.txt");
        HANDLE tf = CreateFileW(L"test-kvsnap-sym-target.txt", GENERIC_WRITE, 0, NULL,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (tf != INVALID_HANDLE_VALUE) CloseHandle(tf);
        typedef BOOLEAN (WINAPI *sym_fn)(LPCWSTR, LPCWSTR, DWORD);
        sym_fn mk_sym = (sym_fn)(void *)GetProcAddress(GetModuleHandleW(L"kernel32.dll"),
                                                       "CreateSymbolicLinkW");
        // SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE == 0x2
        if (mk_sym && mk_sym(L"test-kvsnap-sym.out.publish.lock",
                             L"test-kvsnap-sym-target.txt", 0x2)) {
            void *lk = NULL;
            char errbuf[512];
            plat_lock_status st = lock_capture_stderr("test-kvsnap-sym.out", &lk,
                                                      errbuf, sizeof errbuf, NULL);
            ck(st == PLAT_LOCK_ERROR && lk == NULL,
               "reparse-point refusal: returns PLAT_LOCK_ERROR");
            ck(strstr(errbuf, "reparse point") != NULL &&
               strstr(errbuf, "Windows error 0") == NULL,
               "reparse-point refusal: prints real cause without Windows error 0");
            DeleteFileW(L"test-kvsnap-sym.out.publish.lock");
        } else {
            printf("skip: reparse-point test (symlink creation refused, error %lu)\n",
                   (unsigned long)GetLastError());
        }
        DeleteFileW(L"test-kvsnap-sym-target.txt");
    }
#endif
    return g_fail ? 1 : 0;
}
