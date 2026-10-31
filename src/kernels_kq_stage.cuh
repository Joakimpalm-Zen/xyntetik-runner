#ifndef RUNNER_KERNELS_KQ_STAGE_CUH
#define RUNNER_KERNELS_KQ_STAGE_CUH
// The tensor-core stagers of the K-quants and legacy quants that have no
// raw-register TC2 loader (Q5_K, Q3_K, Q2_K, Q4_1, Q5_0, Q5_1): one
// 64-element segment of a row from global into shared, decoded by
// kq_stage.h, which tests/test_kq_stage.c holds to dequant_row() bit for
// bit. Shared by kernels.cu (compute_75) and kernels_sm80.cu (compute_80);
// includers define uchar and ulong64 and include cuda_fp16.h first.
#include "kq_stage.h"

// 256-element blocks: segment (e0 % 256) / 64 of block e0 / 256. A K-quant
// row is a whole number of blocks, so no segment reaches past n_in.
#define KQ_STAGE_256(NAME, FN, BLKBYTES)                                       \
static __device__ __forceinline__ void NAME(__half *dst, const uchar *rw,     \
                                            int nb, int e0, int n_in) {       \
    FN(dst, rw + (ulong64)(e0 >> 8) * (BLKBYTES), (e0 & 255) >> 6);           \
    (void)nb; (void)n_in;                                                      \
}
KQ_STAGE_256(tc_stage_q5_K, q5k_stage64, 176)
KQ_STAGE_256(tc_stage_q3_K, q3k_stage64, 110)
KQ_STAGE_256(tc_stage_q2_K, q2k_stage64, 84)

// 32-element blocks: two a segment, tail-safe past n_in as tc_stage_q4_0
#define KQ_STAGE_32(NAME, FN, BLKBYTES)                                        \
static __device__ __forceinline__ void NAME(__half *dst, const uchar *rw,     \
                                            int nb, int e0, int n_in) {       \
    _Pragma("unroll")                                                          \
    for (int half = 0; half < 2; half++) {                                     \
        int base = e0 + half * 32;                                             \
        if (base >= n_in) {                                                    \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 32; j++)                                       \
                dst[half * 32 + j] = __float2half(0.0f);                       \
            continue;                                                          \
        }                                                                      \
        FN(dst + half * 32, rw + (ulong64)(base / 32) * (BLKBYTES));           \
    }                                                                          \
    (void)nb;                                                                  \
}
KQ_STAGE_32(tc_stage_q4_1, q41_stage32, 20)
KQ_STAGE_32(tc_stage_q5_0, q50_stage32, 22)
KQ_STAGE_32(tc_stage_q5_1, q51_stage32, 24)

// NVFP4 (R4.9.5): a 64-element block is one segment; a row is a whole
// number of blocks, so no segment reaches past n_in
static __device__ __forceinline__ void tc_stage_nvfp4(__half *dst, const uchar *rw, int nb,
                                                      int e0, int n_in) {
    nvfp4_stage64(dst, rw + (ulong64)(e0 >> 6) * 36);
    (void)nb; (void)n_in;
}
#endif // RUNNER_KERNELS_KQ_STAGE_CUH
