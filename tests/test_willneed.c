// plat_willneed on Linux brings a whole unaligned range in (R3.41). Before the
// fix an expert's hint started mid-page, madvise refused it with EINVAL, and
// --moe-prefetch on read nothing; and an aligned hint longer than the device's
// readahead window read only its first window.
//
// The check: a file several windows long, mapped with plat_mmap_ro, evicted,
// then hinted from an address 32 bytes past a page (a GGUF tensor's
// alignment). Every page of the range must be resident within 5 s. Where the
// file system keeps the file resident anyway (tmpfs) the eviction does not
// take, and the residency check is reported as not informative, not passed.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double resident(const uint8_t *base, size_t off, size_t len, long ps) {
    uintptr_t lo = ((uintptr_t)base + off) & ~((uintptr_t)ps - 1);
    uintptr_t hi = ((uintptr_t)base + off + len + ps - 1) & ~((uintptr_t)ps - 1);
    size_t n = (hi - lo) / ps, r = 0;
    unsigned char *v = malloc(n);
    if (!v || mincore((void *)lo, hi - lo, v) != 0) { free(v); return -1; }
    for (size_t i = 0; i < n; i++) r += v[i] & 1;
    free(v);
    return (double)r / n;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test-willneed.bin";
    long ps = sysconf(_SC_PAGESIZE);
    size_t sz = 12u << 20;
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "FAIL: cannot write %s\n", path); return 1; }
    char *buf = malloc(1 << 20);
    for (int i = 0; i < (1 << 20); i++) buf[i] = (char)(i * 31);
    for (size_t w = 0; w < sz; w += 1 << 20) fwrite(buf, 1, 1 << 20, f);
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    free(buf);
    size_t msz = 0;
    uint8_t *m = plat_mmap_ro(path, &msz);
    if (!m || msz != sz) { fprintf(stderr, "FAIL: plat_mmap_ro\n"); return 1; }
    size_t win = plat_willneed_window();
    printf("willneed: window %zu KB\n", win >> 10);
    // a range three windows long (at least 6 MB), starting 32 bytes into a page
    size_t off = (size_t)ps + 32, len = 3 * win > (6u << 20) ? 3 * win : (6u << 20);
    if (off + len > sz) len = sz - off;
    int fd = open(path, O_RDONLY);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    double before = resident(m, off, len, ps);
    int fail = 0;
    if (before > 0.5) {
        printf("willneed: the file stayed %.0f%% resident after eviction (tmpfs or similar): "
               "residency not informative here\n", 100 * before);
    } else {
        plat_willneed(m + off, len);
        double r = 0;
        struct timespec t0, t;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        do {
            usleep(10000);
            r = resident(m, off, len, ps);
            clock_gettime(CLOCK_MONOTONIC, &t);
        } while (r < 1.0 && (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) * 1e-9 < 5.0);
        printf("willneed: %.1f%% of a %zu KB range at page offset 32 resident after the hint (%.0f%% before)\n",
               100 * r, len >> 10, 100 * before);
        if (r < 1.0) {
            fprintf(stderr, "FAIL: plat_willneed left %.1f%% of the range out\n", 100 * (1 - r));
            fail = 1;
        }
    }
    munmap(m, msz);
    remove(path);
    if (!fail) printf("willneed: ok\n");
    return fail;
}
#else
int main(void) {
    printf("willneed: Linux only (macOS and Windows keep their own hint)\n");
    return 0;
}
#endif
