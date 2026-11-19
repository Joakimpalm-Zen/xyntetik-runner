// Failure-injection regression for plat_fopen_new() / plat_fopen_partial() (R6.101):
// if fdopen / _open_osfhandle / _fdopen fails after the file has already been
// created exclusively (O_CREAT|O_EXCL / CREATE_NEW), the newly created empty
// file must be unlinked, *made_out stay false, and errno be preserved; when
// O_EXCL / CREATE_NEW itself fails on a pre-existing file, that file must
// remain untouched.
#ifndef _CRT_RAND_S
#define _CRT_RAND_S
#endif
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>

static int g_fail_osfhandle = 0;
static int g_fail_fdopen = 0;

static int test_open_osfhandle(intptr_t h, int flags) {
    if (g_fail_osfhandle) {
        errno = EMFILE;
        return -1;
    }
    return _open_osfhandle(h, flags);
}

static FILE *test_wfdopen(int fd, const char *mode) {
    if (g_fail_fdopen) {
        errno = ENOMEM;
        return NULL;
    }
    return _fdopen(fd, mode);
}

#define _open_osfhandle test_open_osfhandle
#define _fdopen test_wfdopen
#else
#include <unistd.h>

static int g_fail_fdopen = 0;

static FILE *test_fdopen(int fd, const char *mode) {
    if (g_fail_fdopen) {
        errno = ENOMEM;
        return NULL;
    }
    return fdopen(fd, mode);
}

#define fdopen test_fdopen
#endif

#undef _CRT_RAND_S
#include "../src/compat.c"

static int g_fail = 0;
static void ck(bool cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else       printf("ok: %s\n", what);
}

static bool file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

int main(void) {
    const char *base = "test-partial-fdopen-target.bin";
    const char *fixed = "test-partial-fdopen-target.bin.partial-fixed";
    char tmp[256];
    bool made = false;

    remove(fixed);
    remove(base);

    // 1. POSIX fdopen failure (or Win32 _fdopen failure) after exclusive create
    g_fail_fdopen = 1;
#ifdef _WIN32
    _putenv_s("RUNNER_TEST_PARTIAL_SUFFIX", "fixed");
#else
    setenv("RUNNER_TEST_PARTIAL_SUFFIX", "fixed", 1);
#endif
    errno = 0;
    FILE *f = plat_fopen_partial(base, "partial", tmp, sizeof tmp, &made);
    int saved_err = errno;
    g_fail_fdopen = 0;
    ck(f == NULL, "plat_fopen_partial returns NULL when fdopen fails");
    ck(!made, "made stays false when fdopen fails");
    ck(saved_err == ENOMEM, "errno ENOMEM preserved across cleanup");
    ck(!file_exists(fixed), "just-created empty file is removed when fdopen fails");

#ifdef _WIN32
    // 1b. Win32 _open_osfhandle failure after CREATE_NEW
    g_fail_osfhandle = 1;
    errno = 0;
    f = plat_fopen_partial(base, "partial", tmp, sizeof tmp, &made);
    saved_err = errno;
    g_fail_osfhandle = 0;
    ck(f == NULL, "plat_fopen_partial returns NULL when _open_osfhandle fails");
    ck(!made, "made stays false when _open_osfhandle fails");
    ck(saved_err == EMFILE, "errno EMFILE preserved across cleanup");
    ck(!file_exists(fixed), "just-created empty file is removed when _open_osfhandle fails");
#endif

    // 2. Pre-existing file at the partial path is refused without unlinking it
    FILE *pre = fopen(fixed, "wb");
    ck(pre != NULL, "create pre-existing file");
    if (pre) { fputs("keep", pre); fclose(pre); }
    g_fail_fdopen = 1;
    f = plat_fopen_partial(base, "partial", tmp, sizeof tmp, &made);
    g_fail_fdopen = 0;
    ck(f == NULL && !made, "pre-existing file is refused");
    ck(file_exists(fixed), "pre-existing file is never removed on EEXIST");
    remove(fixed);

    // 3. Normal success writes cleanly
    f = plat_fopen_partial(base, "partial", tmp, sizeof tmp, &made);
    ck(f != NULL && made, "plat_fopen_partial succeeds normally");
    if (f) { fputs("ok", f); fclose(f); }
    ck(file_exists(fixed), "normal partial file created");
    remove(fixed);

#ifndef _WIN32
    unsetenv("RUNNER_TEST_PARTIAL_SUFFIX");
#else
    _putenv_s("RUNNER_TEST_PARTIAL_SUFFIX", "");
#endif
    return g_fail ? 1 : 0;
}
