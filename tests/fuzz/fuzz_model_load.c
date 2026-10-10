// Reach the real CPU binder/runtime and cleanup with bounded model geometry.
// Extreme geometry remains covered by the unfiltered GGUF parser and hostile
// geometry tests; this target spends its budget past those rejection checks.
#include "runner.h"
#include "fuzz_files.h"
#include <string.h>

static char path[256];
static uint64_t parsed, attempted, loaded;
static void cleanup(void) {
    fprintf(stderr, "model-load reachability: parsed=%llu bind-attempts=%llu loaded=%llu\n",
            (unsigned long long)parsed, (unsigned long long)attempted, (unsigned long long)loaded);
    unlink(path); fuzz_files_close();
}
int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv; fuzz_files_init();
    snprintf(path, sizeof path, "%s/model.gguf", fuzz_dir);
    atexit(cleanup); return 0;
}
static bool bounded(gguf_file *g) {
    if (g->n_tensors > 256 || g->n_kv > 256) return false;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gguf_kv *k = &g->kv[i];
        if (k->type <= GGUF_T_I32 || k->type == GGUF_T_U64 || k->type == GGUF_T_I64)
            if (k->v.u64 > 512) return false;
        if (k->arr_n > 1024) return false;
        if (strstr(k->key, ".block_count") && k->v.u64 > 8) return false;
    }
    for (uint64_t i = 0; i < g->n_tensors; i++)
        for (uint32_t j = 0; j < g->tensors[i].n_dims; j++)
            if (g->tensors[i].ne[j] > 1024) return false;
    const char *arch = gguf_get_str(g, "general.architecture", "");
    return !strcmp(arch, "llama") || !strcmp(arch, "qwen2");
}
int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
    if (size > 1024 * 1024) return 0;
    fuzz_write(path, data, size);
    fuzz_mute();
    gguf_file g;
    bool ok = gguf_open(&g, path), run = false;
    if (ok) { parsed++; run = bounded(&g); gguf_close(&g); }
    if (run) {
        attempted++;
        model_params p = {0}; p.gpu_mode = GPU_OFF;
        p.n_threads = 1; p.n_ctx = 16; p.n_batch = 1;
        model_t m = {0};
        if (model_load(&m, path, &p)) loaded++;
        model_free(&m); // success and partial rejection share the public cleanup
    }
    fuzz_unmute(); return 0;
}
