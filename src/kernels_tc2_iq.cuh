#ifndef RUNNER_KERNELS_TC2_IQ_CUH
#define RUNNER_KERNELS_TC2_IQ_CUH
// Shared by kernels.cu (compute_75) and kernels_sm80.cu (compute_80), moved
// out of kernels.cu verbatim. Includers define uchar, ushort16, uint and
// ulong64 and include cuda_fp16.h first.

// The codebook i-quants keep their TC stagers (tc_stage_*, one 64-element
// segment from global into shared, the same per-element values): no raw
// register prefetch, so gemm2_body stages their next buffer before the
// multiplies instead of after, overlapping the stager's reads with other
// warps' tensor work.
#define G2_DIRECT(NAME, STAGE, BLKBYTES, BLKELEMS)                                     \
struct NAME {                                                                          \
    struct raw { const uchar *rowp; int n_in; };                                       \
    enum { kDirect = 1 };                                                              \
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks,     \
                                                int n_in) {                            \
        r.rowp = rowp; r.n_in = n_in; (void)ks;                                        \
    }                                                                                  \
    static __device__ __forceinline__ void store(__half *dst, const raw &r, int ks) {  \
        STAGE(dst, r.rowp, r.n_in / (BLKELEMS), ks, r.n_in);                           \
    }                                                                                  \
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) {                    \
        return (ulong64)(n_in / (BLKELEMS)) * (BLKBYTES);                              \
    }                                                                                  \
};
G2_DIRECT(G2IQ3S, tc_stage_iq3_s, 110, 256)
G2_DIRECT(G2IQ3XXS, tc_stage_iq3_xxs, 98, 256)
G2_DIRECT(G2IQ2S, tc_stage_iq2_s, 82, 256)
G2_DIRECT(G2IQ2XS, tc_stage_iq2_xs, 74, 256)
G2_DIRECT(G2IQ2XXS, tc_stage_iq2_xxs, 66, 256)
G2_DIRECT(G2IQ1S, tc_stage_iq1_s, 50, 256)
G2_DIRECT(G2IQ1M, tc_stage_iq1_m, 56, 256)
G2_DIRECT(G2IQ4XS, tc_stage_iq4_xs, 136, 256)
G2_DIRECT(G2IQ4NL, tc_stage_iq4_nl, 18, 32)
#endif // RUNNER_KERNELS_TC2_IQ_CUH
