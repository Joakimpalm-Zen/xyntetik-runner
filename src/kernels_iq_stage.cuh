#ifndef RUNNER_KERNELS_IQ_STAGE_CUH
#define RUNNER_KERNELS_IQ_STAGE_CUH
// Shared by kernels.cu (compute_75) and kernels_sm80.cu (compute_80), moved
// out of kernels.cu verbatim. Includers define uchar, ushort16, uint and
// ulong64 and include cuda_fp16.h first.

// The codebook types' tensor-core stagers (iq_decode.h does the decoding):
// one 64-element segment of one row into shared fp16, the same per-element
// values as dequant_row().
static __device__ __forceinline__ void tc_stage_iq3_s(__half *dst,
                                                      const uchar *rw, int nb,
                                                      int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 110;
    iq3s_stage64(dst, blk, (e0 & 255) >> 6, kiq3s_grid);
    (void)nb; (void)n_in;
}

static __device__ __forceinline__ void tc_stage_iq3_xxs(__half *dst,
                                                        const uchar *rw, int nb,
                                                        int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 98;
    iq3xxs_stage64(dst, blk, (e0 & 255) >> 6, kiq3xxs_grid);
    (void)nb; (void)n_in;
}

static __device__ __forceinline__ void tc_stage_iq2_s(__half *dst,
                                                      const uchar *rw, int nb,
                                                      int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 82;
    iq2s_stage64(dst, blk, (e0 & 255) >> 6, kiq2s_grid);
    (void)nb; (void)n_in;
}

static __device__ __forceinline__ void tc_stage_iq2_xs(__half *dst,
                                                       const uchar *rw, int nb,
                                                       int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 74;
    iq2xs_stage64(dst, blk, (e0 & 255) >> 6, kiq2xs_grid);
    (void)nb; (void)n_in;
}

static __device__ __forceinline__ void tc_stage_iq2_xxs(__half *dst,
                                                        const uchar *rw, int nb,
                                                        int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 66;
    iq2xxs_stage64(dst, blk, (e0 & 255) >> 6, kiq2xxs_grid);
    (void)nb; (void)n_in;
}

static __device__ __forceinline__ void tc_stage_iq1_s(__half *dst,
                                                      const uchar *rw, int nb,
                                                      int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 50;
    iq1s_stage64(dst, blk, (e0 & 255) >> 6, kiq1s_grid);
    (void)nb; (void)n_in;
}

static __device__ __forceinline__ void tc_stage_iq1_m(__half *dst,
                                                      const uchar *rw, int nb,
                                                      int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 56;
    iq1m_stage64(dst, blk, (e0 & 255) >> 6, kiq1s_grid);
    (void)nb; (void)n_in;
}

// the IQ4 codebook types: the same shape, the nibble codebook kv_iq4
static __device__ __forceinline__ void tc_stage_iq4_xs(__half *dst,
                                                       const uchar *rw, int nb,
                                                       int e0, int n_in) {
    const uchar *blk = rw + (ulong64)(e0 >> 8) * 136;
    iq4xs_stage64(dst, blk, (e0 & 255) >> 6, kv_iq4);
    (void)nb; (void)n_in;
}

// IQ4_NL has 32-element blocks like Q4_0, so its stager is the Q4_0 one's
// shape: two blocks per 64-element segment, tail-safe past n_in
static __device__ __forceinline__ void tc_stage_iq4_nl(__half *dst,
                                                       const uchar *rw, int nb,
                                                       int e0, int n_in) {
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        int base = e0 + half * 32;
        if (base >= n_in) {
            #pragma unroll
            for (int j = 0; j < 32; j++) dst[half * 32 + j] = __float2half(0.0f);
            continue;
        }
        iq4nl_stage32(dst + half * 32, rw + (ulong64)(base / 32) * 18, kv_iq4);
    }
    (void)nb;
}
#endif // RUNNER_KERNELS_IQ_STAGE_CUH
