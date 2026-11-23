// libFuzzer harness for the inkling2_loop header and tensor-table checks (src/inkling2_gguf.c ink2g_attach).
//
// The same shape as fuzz_gguf_open.c: each input is written to one reused file and opened with gguf_open. A file the
// GGUF reader accepts is then handed to the Inkling 2 Mini source, which reads the shape from the metadata
// (read_shape: the loop keys, the per-layer head counts, the sliding-window pattern) and checks every tensor of the
// text model against it (presence, dims, type, the per-expert .scale beside an NVFP4 stack). The contract under test
// is "refuse with a sentence, never crash": an accepted file is walked further, every reference tensor decoded once,
// so a check that let a misshapen tensor through shows up as an out-of-bounds read here.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdbool.h>

#include "gguf.h"
#include "inkling2_gguf.h"

static char g_path[] = "/tmp/fuzz-inkling-XXXXXX";
static int  g_fd = -1;
static int  g_null = -1, g_stderr = -1;

static void cleanup(void) {
    if (g_fd >= 0) close(g_fd);
    unlink(g_path);
}
static void mute_stderr(void)   { if (g_null   >= 0) dup2(g_null, 2); }
static void unmute_stderr(void) { if (g_stderr >= 0) dup2(g_stderr, 2); }

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv;
    g_fd = mkstemp(g_path);
    if (g_fd < 0) { perror("mkstemp"); exit(1); }
    atexit(cleanup);
    g_stderr = dup(2);
    g_null   = open("/dev/null", O_WRONLY);
    if (g_stderr < 0 || g_null < 0) { perror("dup/open"); exit(1); }
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (ftruncate(g_fd, 0) != 0) return 0;
    if (lseek(g_fd, 0, SEEK_SET) != 0) return 0;
    for (size_t off = 0; off < size; ) {
        ssize_t w = write(g_fd, data + off, size - off);
        if (w <= 0) return 0;
        off += (size_t)w;
    }
    gguf_file g;
    mute_stderr();
    bool ok = gguf_open(&g, g_path);
    unmute_stderr();
    if (!ok) return 0;
    ink2_shape s;
    char why[512];
    mute_stderr();
    ink2g *src = ink2g_attach(&g, &s, why, sizeof why);
    if (src) {
        // an accepted file: every dense tensor decoded, and one expert of every stack
        (void)ink2g_decode_all(src, NULL);
        for (int b = 0; b < s.n_block; b++)
            for (int which = 0; which < 2; which++) {
                char name[96];
                snprintf(name, sizeof name, "model.layers.%d.mlp.experts.%s", b, which ? "down_proj" : "gate_up_proj");
                (void)ink2g_get_expert(src, name, s.n_expert - 1);
            }
        ink2g_close(src);
    }
    unmute_stderr();
    gguf_close(&g);
    return 0;
}
