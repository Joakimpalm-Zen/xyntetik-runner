#ifndef RUNNER_DN_EPS_H
#define RUNNER_DN_EPS_H

// The Gated DeltaNet's q/k L2 norm epsilon: x / sqrt(sum x^2 + 1e-6), the
// publisher's own (transformers' l2norm, eps=1e-6 hard-coded), not the
// model's RMS epsilon, which only the RMS norms take. Every qwen35 and Clef
// file known ships rms_norm_eps 1e-6, so the two were equal there; a file
// with another value read its q and k at the wrong scale (R8.18). The CPU
// forward, the backward's recompute, the CUDA steps and the Metal mixer all
// take it here.
#define DN_L2_EPS 1e-6f

#endif
