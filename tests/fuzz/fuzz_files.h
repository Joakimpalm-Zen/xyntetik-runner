// File-backed fuzz targets use an isolated directory and retain sanitizer
// diagnostics via log_path while ordinary malformed-input messages are muted.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

static char fuzz_dir[] = "/tmp/runner-fuzz-XXXXXX";
static int fuzz_err, fuzz_null;
static void fuzz_files_init(void) {
    assert(mkdtemp(fuzz_dir));
    fuzz_err = dup(2); fuzz_null = open("/dev/null", O_WRONLY);
    assert(fuzz_err >= 0 && fuzz_null >= 0);
}
static void fuzz_write(const char *path, const unsigned char *data, size_t size) {
    FILE *f = fopen(path, "wb"); assert(f);
    assert(fwrite(data, 1, size, f) == size); assert(fclose(f) == 0);
}
static void fuzz_mute(void) { assert(dup2(fuzz_null, 2) >= 0); }
static void fuzz_unmute(void) { assert(dup2(fuzz_err, 2) >= 0); }
static void fuzz_files_close(void) { close(fuzz_err); close(fuzz_null); rmdir(fuzz_dir); }
