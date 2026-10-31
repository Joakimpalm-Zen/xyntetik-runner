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

// the codebook i-quants and the K-quants / legacy quants without a
// raw-register loader: their tables, decoders and stagers, shared with
// kernels.cu, then their direct-staging TC2 loaders
#include "kernels_iq4_table.cuh"
#include "kernels_iq_grids.cuh"
#include "iq_decode.h"
#include "kernels_iq_stage.cuh"
#include "kernels_kq_stage.cuh"

#define G2_MMA_K16 1
#include "kernels_tc2.cuh"
#include "kernels_tc2_iq.cuh"
#include "kernels_tc2_kq.cuh"

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
G2_80(k_gemm2_80_iq3_s, G2IQ3S)
G2_80(k_gemm2_80_iq3_xxs, G2IQ3XXS)
G2_80(k_gemm2_80_iq2_s, G2IQ2S)
G2_80(k_gemm2_80_iq2_xs, G2IQ2XS)
G2_80(k_gemm2_80_iq2_xxs, G2IQ2XXS)
G2_80(k_gemm2_80_iq1_s, G2IQ1S)
G2_80(k_gemm2_80_iq1_m, G2IQ1M)
G2_80(k_gemm2_80_iq4_xs, G2IQ4XS)
G2_80(k_gemm2_80_iq4_nl, G2IQ4NL)
G2_80(k_gemm2_80_q5_K, G2Q5K)
G2_80(k_gemm2_80_q3_K, G2Q3K)
G2_80(k_gemm2_80_q2_K, G2Q2K)
G2_80(k_gemm2_80_q4_1, G2Q41)
G2_80(k_gemm2_80_q5_0, G2Q50)
G2_80(k_gemm2_80_q5_1, G2Q51)
G2_80(k_gemm2_80_nvfp4, G2NV4)

// The 128-column tile: each weight dequantized once for 128 token columns
// instead of 64, the integer work that bounds the 64-column kernel halved per
// output. Byte-identical to the kernels above (gemm2_body). For GEMMs over
// more than 64 columns; it needs 73,728 bytes of dynamic shared memory, which
// every compute capability 8.0+ device grants (7.5 stops at 64 KB).
#define G2_80W(NAME, F)                                                             \
extern "C" __global__ void __launch_bounds__(256)                                   \
NAME(const uchar *wb, const __half *xh, float *y, mv_args a, const float *bias,     \
     const float *xsc) {                                                            \
    gemm2_body<F, G2_BN_W>(wb, xh, y, a, bias, xsc);                                \
}
G2_80W(k_gemm2_80w_q4_K, G2Q4K)
G2_80W(k_gemm2_80w_q6_K, G2Q6K)
G2_80W(k_gemm2_80w_q8_0, G2Q80)
G2_80W(k_gemm2_80w_q4_0, G2Q40)
G2_80W(k_gemm2_80w_iq3_s, G2IQ3S)
G2_80W(k_gemm2_80w_iq3_xxs, G2IQ3XXS)
G2_80W(k_gemm2_80w_iq2_s, G2IQ2S)
G2_80W(k_gemm2_80w_iq2_xs, G2IQ2XS)
G2_80W(k_gemm2_80w_iq2_xxs, G2IQ2XXS)
G2_80W(k_gemm2_80w_iq1_s, G2IQ1S)
G2_80W(k_gemm2_80w_iq1_m, G2IQ1M)
G2_80W(k_gemm2_80w_iq4_xs, G2IQ4XS)
G2_80W(k_gemm2_80w_iq4_nl, G2IQ4NL)
G2_80W(k_gemm2_80w_q5_K, G2Q5K)
G2_80W(k_gemm2_80w_q3_K, G2Q3K)
G2_80W(k_gemm2_80w_q2_K, G2Q2K)
G2_80W(k_gemm2_80w_q4_1, G2Q41)
G2_80W(k_gemm2_80w_q5_0, G2Q50)
G2_80W(k_gemm2_80w_q5_1, G2Q51)
G2_80W(k_gemm2_80w_nvfp4, G2NV4)
