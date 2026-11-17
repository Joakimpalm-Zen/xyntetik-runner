// R6.90: what a GPU gate asks before it skips for want of a device.
//
// "No device" is a skip. A device that is up and a kernel module that did not
// load (src/cuda.c module_load_failed: the driver's JIT rejected the embedded
// PTX) is a FAILURE, printed with the driver's log: before R6.90 the two read
// alike, every gate that found no engaged device skipped, and a module that
// never loaded passed them all. Call it where the gate would skip; a nonzero
// return means fail.
#ifndef RUNNER_GATE_GPU_H
#define RUNNER_GATE_GPU_H

#include <stdio.h>
#include "gpu.h"

static inline int gate_no_device(const char *gate) {
    const char *why = gpu_module_failure();
    if (!why) return 0;
    fprintf(stderr, "%s: FAIL: a GPU device is up but %s\n", gate, why);
    printf("%s: FAILED (a kernel module did not load; see stderr)\n", gate);
    return 1;
}

#endif // RUNNER_GATE_GPU_H
