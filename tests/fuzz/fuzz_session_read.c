// libFuzzer harness for the session image reader (src/session.c). An image
// is a file that travels: it is written on one machine and resumed on
// another, so the reader sees bytes this process did not write. Its SHA-256
// trailer proves the file is whole, not that it is honest, since anyone can
// compute it; the harness therefore fixes the trailer up for half the
// inputs, so the mutator reaches the parser behind it instead of stopping at
// the digest.
//
// What a successful read must guarantee, whatever the bytes: the counts in
// the header agree with the sections, and every section the caller will
// index is allocated at the size the header states.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "envelope.h"
#include "session.h"
#include "fuzz_files.h"

static char path[256];
static void cleanup(void) { unlink(path); fuzz_files_close(); }
int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv; fuzz_files_init();
    snprintf(path, sizeof path, "%s/image.session", fuzz_dir);
    atexit(cleanup); return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 1 || size > 1024 * 1024) return 0;
    bool fix_trailer = data[0] & 1;
    size_t n = size - 1;
    uint8_t *img_bytes = malloc(n + 32);
    if (!img_bytes) return 0;
    memcpy(img_bytes, data + 1, n);
    if (fix_trailer) {
        envelope_data_sha256_raw(img_bytes, n, img_bytes + n);
        n += 32;
    }
    fuzz_write(path, img_bytes, n);
    free(img_bytes);
    fuzz_mute();
    session_image img;
    if (session_read(path, &img)) {
        const session_meta *m = &img.meta;
        assert(m->n_ctx >= 1 && m->n_prompt >= 1);
        assert(m->n_tokens >= m->n_prompt && m->n_tokens <= m->n_ctx);
        assert(m->generated == m->n_tokens - m->n_prompt && m->max_new >= m->generated);
        assert(img.tokens && img.state && img.logits && img.n_vocab >= 0);
        // every section is readable at the size the header gave it
        volatile int32_t t = img.tokens[m->n_tokens - 1]; (void)t;
        if (img.state_n) { volatile uint8_t s = img.state[img.state_n - 1]; (void)s; }
        if (img.n_vocab) { volatile float l = img.logits[img.n_vocab - 1]; (void)l; }
        assert(strlen(img.sha256) == 64 && strlen(m->model_sha256) == 64);
        session_image_free(&img);
    }
    fuzz_unmute();
    return 0;
}
