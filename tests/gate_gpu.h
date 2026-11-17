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
#include <string.h>
#include "runner.h"
#include "gpu.h"

static inline int gate_no_device(const char *gate) {
    const char *why = gpu_module_failure();
    if (!why) return 0;
    fprintf(stderr, "%s: FAIL: a GPU device is up but %s\n", gate, why);
    printf("%s: FAILED (a kernel module did not load; see stderr)\n", gate);
    return 1;
}

// R6.90 (review P2): the minimum compute capability's decision, pinned with
// no device. A device below an image's minimum is no usable device, a skip
// (src/cuda.c module_cc_ok), never a module failure. Anchors: runner.h's
// RUNNER_CUDA_MIN_GPU and the manual's support matrix ("NVIDIA Turing /
// compute capability 7.5 or newer"), and NVIDIA's numbering: Maxwell 5.2,
// Pascal 6.1 (GTX 10-series), Volta 7.0 (V100) and Xavier 7.2 below it; Turing
// 7.5, Ampere 8.0 and 8.6, Ada 8.9, Hopper 9.0 and the RTX 50 series 12.0 at or above.
// The compute_80 image's minimum is 8.0. Nonzero: a pin that does not hold.
static inline int gate_cc_pins(const char *gate) {
    static const struct { int maj, min; bool ok, ok80; } P[] = {
        { 5, 2, false, false }, { 6, 1, false, false }, { 7, 0, false, false },
        { 7, 2, false, false }, { 7, 5, true, false },  { 8, 0, true, true },
        { 8, 6, true, true },   { 8, 9, true, true },   { 9, 0, true, true },
        { 12, 0, true, true },
    };
    int n = (int)(sizeof(P) / sizeof(P[0])), bad = 0;
    for (int i = 0; i < n; i++) {
        bool u = gpu_cc_usable(P[i].maj, P[i].min), u80 = gpu_cc_at_least(P[i].maj, P[i].min, 8, 0);
        if ((u != P[i].ok || u80 != P[i].ok80) && !bad++)
            fprintf(stderr, "%s: FAIL: compute capability %d.%d reads as %s for compute_75 and %s "
                    "for compute_80\n", gate, P[i].maj, P[i].min, u ? "usable" : "not usable",
                    u80 ? "usable" : "not usable");
    }
    char s[16];
    snprintf(s, sizeof s, "%d.%d", GPU_CUDA_MIN_CC_MAJOR, GPU_CUDA_MIN_CC_MINOR);
    if (strcmp(s, RUNNER_CUDA_MIN_CC) != 0 && !bad++)
        fprintf(stderr, "%s: FAIL: gpu.h's minimum %s is not runner.h's " RUNNER_CUDA_MIN_CC "\n",
                gate, s);
    if (!bad) printf("%s: compute capability minimum %s (compute_75), 8.0 (compute_80): %d pins hold\n",
                     gate, s, n);
    return bad;
}

#endif // RUNNER_GATE_GPU_H
