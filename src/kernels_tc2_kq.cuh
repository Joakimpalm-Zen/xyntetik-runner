#ifndef RUNNER_KERNELS_TC2_KQ_CUH
#define RUNNER_KERNELS_TC2_KQ_CUH
// The TC2 loaders of the K-quants and legacy quants without a raw-register
// one: the codebook i-quants' direct staging (G2_DIRECT, kernels_tc2_iq.cuh)
// over the stagers of kernels_kq_stage.cuh. Include both of those first.
G2_DIRECT(G2Q5K, tc_stage_q5_K, 176, 256)
G2_DIRECT(G2Q3K, tc_stage_q3_K, 110, 256)
G2_DIRECT(G2Q2K, tc_stage_q2_K, 84, 256)
G2_DIRECT(G2Q41, tc_stage_q4_1, 20, 32)
G2_DIRECT(G2Q50, tc_stage_q5_0, 22, 32)
G2_DIRECT(G2Q51, tc_stage_q5_1, 24, 32)
G2_DIRECT(G2NV4, tc_stage_nvfp4, 36, 64)
#endif // RUNNER_KERNELS_TC2_KQ_CUH
