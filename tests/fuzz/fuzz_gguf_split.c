// Input: u32 first-part length, first GGUF, second GGUF. Unlike a single
// temporary path, this reaches normal cross-part metadata/binding checks.
#include "gguf.h"
#include "fuzz_files.h"
#include <stdint.h>

static char paths[2][256];
static void cleanup(void) {
    unlink(paths[0]); unlink(paths[1]); fuzz_files_close();
}
int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv; fuzz_files_init();
    for (int i = 0; i < 2; i++)
        snprintf(paths[i], sizeof paths[i], "%s/model-%05d-of-00002.gguf", fuzz_dir, i + 1);
    atexit(cleanup); return 0;
}
int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
    if (size < 4 || size > 1024 * 1024) return 0;
    uint32_t n = (uint32_t)data[0] | (uint32_t)data[1] << 8 |
                 (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
    if (n > size - 4) return 0;
    fuzz_write(paths[0], data + 4, n);
    fuzz_write(paths[1], data + 4 + n, size - 4 - n);
    fuzz_mute();
    for (int start = 0; start < 2; start++) {
        gguf_file g;
        if (gguf_open(&g, paths[start])) {
            uint64_t total = 0;
            for (uint32_t i = 0; i < gguf_map_count(&g); i++) {
                size_t bytes = 0;
                assert(gguf_map_part(&g, i, &bytes)); total += bytes;
            }
            assert(total == gguf_mapped_size(&g));
            for (uint64_t i = 0; i < g.n_tensors; i++) {
                gguf_tensor *t = &g.tensors[i];
                if (t->data && t->nbytes) {
                    volatile unsigned char x = ((unsigned char *)t->data)[t->nbytes - 1];
                    (void)x;
                }
            }
            gguf_close(&g);
        }
    }
    fuzz_unmute(); return 0;
}
