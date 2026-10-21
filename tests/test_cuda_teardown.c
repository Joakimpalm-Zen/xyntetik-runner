// Device buffers come back when a model is freed.
//
//     ./test-cuda-teardown RESIDENT.gguf CYCLED.gguf [cycles]
//
// cuda.c allocates an instance's scratch in its cuda_alloc_* functions and
// gives it back in gpu_ctx_free, from a list kept by hand beside them. When
// the two drift, what a family allocates but never frees stays allocated in
// the process's CUDA context. Every model on a device shares that context, so
// a server that unloads one model while another stays resident keeps those
// bytes until it exits: the hyper-connection streams, the split-K partials and
// the shared-expert buffers did, until 2026-10-20 (qwen4exp, Qwen3.8).
//
// The anchor is the allocation count itself (gpu_live_buffers: every
// cuMemAlloc the engine made and has not freed), not the free-byte figure,
// which the driver's pooling hides small leaks from: loading CYCLED, running a
// prompt and a decode step through it and freeing it, with RESIDENT loaded
// the whole time, must return the count to where it was before the load.
// Skips (prints so) where the backend does not count, or nothing engaged CUDA.
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool load(model_t *m, const char *path) {
    memset(m, 0, sizeof(*m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_AUTO;
    p.n_ctx = 256;
    p.n_batch = 64;
    return model_load(m, path, &p);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s RESIDENT.gguf CYCLED.gguf [cycles]\n", argv[0]);
        return 2;
    }
    int cycles = argc > 3 ? atoi(argv[3]) : 3;
    model_t res;
    if (!load(&res, argv[1])) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    if (!res.gpu || gpu_live_buffers() < 0) {
        printf("test-cuda-teardown: ok (skipped: no CUDA device engaged)\n");
        model_free(&res);
        return 0;
    }
    long base = gpu_live_buffers();
    int fails = 0;
    for (int c = 0; c < cycles; c++) {
        model_t m;
        if (!load(&m, argv[2])) { fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
        if (!m.gpu) {
            fprintf(stderr, "FAIL: %s did not run on the device\n", argv[2]);
            model_free(&m);
            fails++;
            break;
        }
        long held = gpu_live_buffers();
        int32_t toks[6] = { 1, 2, 3, 4, 5, 6 };
        if (!model_forward_batch(&m, toks, 5, 0, true) || !model_forward(&m, toks[5], 5))
            fprintf(stderr, "warning: a forward of %s failed\n", argv[2]);
        model_free(&m);
        long after = gpu_live_buffers();
        printf("cycle %d: %ld device buffers resident, %ld while %s was loaded, %ld after it was freed\n",
               c, base, held, argv[2], after);
        if (after != base) {
            fprintf(stderr, "FAIL: freeing %s left %ld device buffers allocated\n",
                    argv[2], after - base);
            fails++;
        }
    }
    model_free(&res);
    if (fails) return 1;
    printf("test-cuda-teardown: ok (%d load/free cycles, no device buffer left behind)\n", cycles);
    return 0;
}
