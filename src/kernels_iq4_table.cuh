#ifndef RUNNER_KERNELS_IQ4_TABLE_CUH
#define RUNNER_KERNELS_IQ4_TABLE_CUH
// Shared by kernels.cu (compute_75) and kernels_sm80.cu (compute_80), moved
// out of kernels.cu verbatim. Includers define uchar, ushort16, uint and
// ulong64 and include cuda_fp16.h first.

// IQ4: the nibble indexes a fixed 16-entry codebook
static __device__ const signed char kv_iq4[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
#endif // RUNNER_KERNELS_IQ4_TABLE_CUH
