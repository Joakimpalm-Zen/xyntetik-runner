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
#include "kernels_i8.cuh"

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

// the int8 tensor-core prefill (R3.14, kernels_i8.cuh)
GEMM_I8(k_gemm_i8_q4_K, I8Q4K)
GEMM_I8(k_gemm_i8_q6_K, I8Q6K)
GEMM_I8(k_gemm_i8_q8_0, I8Q80)

// ---- moe-limb's prefill (R3.81; the route is described in kernels.cu at
// k_moe_limb_quant): the int8 MMA here gives the exact integer block sums the
// decode's dp4a gives, and the same f32 tail.
static __device__ __forceinline__ void mma_s8_16832(int c[4], const unsigned a[4], const unsigned b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The prefill: one expert's slice (wb + w_off) over cnt limb columns (column
// c of lq/lsc; output y + c*ys). A warp a 16-row by 8-column tile; a thread
// holds rows gid, gid + 8 and columns 2*tig, 2*tig + 1. Leaf l (a decode
// lane) is fma(t, c, .) over blocks l, l + 32, ..., leaves visited in
// bit-reversed order and folded on a stack, which is the decode's
// shfl_down tree. Flagged columns are written by k_moe_limb_q4_0's
// flagged-only pass afterwards.
#define LIMB_TILE_WARPS 4
extern "C" __global__ void k_moe_limb_tile_q4_0(const uchar *wb, ulong64 w_off, int n_in, int n_out,
                                                int cnt, int ys, float *y, const signed char *lq,
                                                const float *lsc) {
    unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned gid = lane >> 2, tig = lane & 3;
    unsigned tile = blockIdx.x * LIMB_TILE_WARPS + warp;
    int nb = n_in / 32;
    unsigned r0 = tile * 16 + gid, r1 = r0 + 8;
    bool l0 = r0 < (unsigned)n_out, l1 = r1 < (unsigned)n_out;
    const uchar *w0 = wb + w_off + (ulong64)(l0 ? r0 : 0) * nb * 18;
    const uchar *w1 = wb + w_off + (ulong64)(l1 ? r1 : 0) * nb * 18;
    unsigned cb = blockIdx.y * 8;
    unsigned bc = cb + gid;                       // this thread's B column
    bool bl = bc < (unsigned)cnt;
    const int *bq = (const int *)(lq + (ulong64)(bl ? bc : 0) * n_in * 3);
    unsigned c0 = cb + tig * 2, c1 = c0 + 1;      // this thread's C columns
    bool cl0 = c0 < (unsigned)cnt, cl1 = c1 < (unsigned)cnt;
    const float *sc0 = lsc + (ulong64)(cl0 ? c0 : 0) * nb, *sc1 = lsc + (ulong64)(cl1 ? c1 : 0) * nb;
    float st[6][4];
    int top = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int leaf = (int)(__brev((unsigned)i) >> 27);
        float p[4] = { 0.f, 0.f, 0.f, 0.f };
        for (int b = leaf; b < nb; b += 32) {
            const uchar *p0 = w0 + (ulong64)b * 18, *p1 = w1 + (ulong64)b * 18;
            float d0 = __half2float(*(const __half *)p0), d1 = __half2float(*(const __half *)p1);
            const ushort16 *h0 = (const ushort16 *)(p0 + 2 + tig * 4), *h1 = (const ushort16 *)(p1 + 2 + tig * 4);
            unsigned u0 = (unsigned)h0[0] | ((unsigned)h0[1] << 16), u1 = (unsigned)h1[0] | ((unsigned)h1[1] << 16);
            unsigned A[4] = { __vsub4(u0 & 0x0F0F0F0Fu, 0x08080808u), __vsub4(u1 & 0x0F0F0F0Fu, 0x08080808u),
                              __vsub4((u0 >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                              __vsub4((u1 >> 4) & 0x0F0F0F0Fu, 0x08080808u) };
            int S[3][4];
            #pragma unroll
            for (int l = 0; l < 3; l++) {
                unsigned B[2] = { bl ? (unsigned)bq[(ulong64)b * 24 + l * 8 + tig] : 0u,
                                  bl ? (unsigned)bq[(ulong64)b * 24 + l * 8 + 4 + tig] : 0u };
                S[l][0] = S[l][1] = S[l][2] = S[l][3] = 0;
                mma_s8_16832(S[l], A, B);
            }
            float s0 = cl0 ? sc0[b] : 0.f, s1 = cl1 ? sc1[b] : 0.f;
            float cc[4] = { __fmul_rn(d0, s0), __fmul_rn(d0, s1), __fmul_rn(d1, s0), __fmul_rn(d1, s1) };
            #pragma unroll
            for (int o = 0; o < 4; o++) {
                long long tot = (long long)S[0][o] * 65536 + (long long)S[1][o] * 256 + S[2][o];
                p[o] = __fmaf_rn(__ll2float_rn(tot), cc[o], p[o]);
            }
        }
        #pragma unroll
        for (int o = 0; o < 4; o++) st[top][o] = p[o];
        top++;
        #pragma unroll
        for (int j = i + 1; (j & 1) == 0; j >>= 1) {
            top--;
            #pragma unroll
            for (int o = 0; o < 4; o++) st[top - 1][o] = st[top - 1][o] + st[top][o];
        }
    }
    if (l0 && cl0) y[(ulong64)c0 * ys + r0] = st[0][0];
    if (l0 && cl1) y[(ulong64)c1 * ys + r0] = st[0][1];
    if (l1 && cl0) y[(ulong64)c0 * ys + r1] = st[0][2];
    if (l1 && cl1) y[(ulong64)c1 * ys + r1] = st[0][3];
}
