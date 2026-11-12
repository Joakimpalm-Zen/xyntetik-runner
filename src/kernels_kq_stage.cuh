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

// 32-element blocks (two a segment) and NVFP4's 64-element blocks (one): the
// shared row stager of kq_stage.h, whose one tail rule zeroes a block at or
// past n_in without reading it (R6.68.111; NVFP4 had no tail and read the next
// row's first block, past the tensor on the last row, for a row 64 x odd wide)
#define KQ_STAGE_ROW(NAME, ROW)                                                \
static __device__ __forceinline__ void NAME(__half *dst, const uchar *rw,     \
                                            int nb, int e0, int n_in) {       \
    ROW(dst, rw, e0, n_in);                                                    \
    (void)nb;                                                                  \
}
KQ_STAGE_ROW(tc_stage_q4_1, kq_row_q4_1)
KQ_STAGE_ROW(tc_stage_q5_0, kq_row_q5_0)
KQ_STAGE_ROW(tc_stage_q5_1, kq_row_q5_1)
KQ_STAGE_ROW(tc_stage_nvfp4, kq_row_nvfp4)

// TC2's NVFP4 loader (G2NV4) stages whole blocks only, so it keeps the
// unchecked stager: TC2 is admitted only for n_in % G2_BK == 0 (cuda.c, the
// f2 route) and steps nst = n_in / G2_BK segments of G2_BK = 64, one NVFP4
// block each (gemm2_body, kernels_tc2.cuh), so no segment starts at n_in
static __device__ __forceinline__ void tc2_stage_nvfp4(__half *dst, const uchar *rw, int nb,
                                                       int e0, int n_in) {
    nvfp4_stage64(dst, rw + (ulong64)(e0 >> 6) * 36);
    (void)nb; (void)n_in;
}

#endif // RUNNER_KERNELS_KQ_STAGE_CUH
