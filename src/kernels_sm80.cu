// Tensor-core kernels for compute capability 8.0 and newer (E1 of the
// performance campaign). The pipelined TC2 GEMM of kernels_tc2.cuh, built for
// the compute_80 target so each 16-deep step is one mma.sync.m16n8k16 instead
// of two m16n8k8 (the only shape the compute_75 PTX can name). Same operands,
// same fp32 sums in another hardware order: the TC2 tolerance class.
//
// Compiled to its own PTX (make ptx -> src/kernels_ptx80.h) and loaded only on
// a device that reports 8.0 or newer (cuda.c shared_load_module). kernels.cu
// keeps every kernel for every supported device, so this module accelerates
// and is never required: when it is absent, fails to load, or
// RUNNER_CUDA_SM80=0 is set, TC2 runs its compute_75 kernels.
#include <cuda_fp16.h>

#include "kernel_args.h"

typedef unsigned char  uchar;
typedef unsigned short ushort16;
typedef unsigned int   uint;
typedef unsigned long long ulong64;

#define G2_MMA_K16 1
#include "kernels_tc2.cuh"

#define G2_80(NAME, F)                                                              \
extern "C" __global__ void __launch_bounds__(256)                                   \
NAME(const uchar *wb, const __half *xh, float *y, mv_args a, const float *bias,     \
     const float *xsc) {                                                            \
    gemm2_body<F>(wb, xh, y, a, bias, xsc);                                         \
}
G2_80(k_gemm2_80_q4_K, G2Q4K)
G2_80(k_gemm2_80_q6_K, G2Q6K)
G2_80(k_gemm2_80_q8_0, G2Q80)
G2_80(k_gemm2_80_q4_0, G2Q40)
