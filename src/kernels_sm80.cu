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
//
// One expert's slice (wb + w_off) over cnt limb columns (column c of lq/lsc;
// output y + c*ys). A block is 8 warps over R row tiles of 16 and NT column
// tiles of 8; a thread holds rows gid, gid + 8 and columns 2*tig, 2*tig + 1 of
// each. Warp w computes the decode's leaves (lanes) w, w+8, w+16, w+24: leaf
// l is fma(t, c, .) over blocks l, l + 32, ..., the block's limb fragments
// kept across the R row tiles. Those four leaves are the decode tree's subtree
// under root w (level 16 adds w+16 to w and w+24 to w+8, level 8 adds w+8 to
// w), folded in registers in that order; the tree's levels 4, 2, 1 then run
// over the 8 roots in shared memory. Every leaf and every add is the decode's,
// so a column's bits do not depend on R, NT or the other columns.
// Flagged columns are written afterwards by k_moe_limb_fix_q4_0.
static __device__ __forceinline__ void mma_s8_16832(int c[4], const unsigned a[4], const unsigned b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int R, int NT>
static __device__ __forceinline__ void limb_tile_body(const uchar *wb, ulong64 w_off, int n_in, int n_out, int cnt, int ys,
                                            float *y, const signed char *lq, const float *lsc) {
    __shared__ float P[8][R * 16][8 * NT];
    unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    int nb = n_in / 32;
    unsigned rbase = blockIdx.x * R * 16, cb = blockIdx.y * 8 * NT;
    bool bl[NT], cl0[NT], cl1[NT];
    const int *bq[NT];
    const float *sc0[NT], *sc1[NT];
    #pragma unroll
    for (int n = 0; n < NT; n++) {
        unsigned bc = cb + n * 8 + gid; bl[n] = bc < (unsigned)cnt;
        bq[n] = (const int *)(lq + (ulong64)(bl[n] ? bc : 0) * n_in * 3);
        unsigned c0 = cb + n * 8 + tig * 2, c1 = c0 + 1; cl0[n] = c0 < (unsigned)cnt; cl1[n] = c1 < (unsigned)cnt;
        sc0[n] = lsc + (ulong64)(cl0[n] ? c0 : 0) * nb; sc1[n] = lsc + (ulong64)(cl1[n] ? c1 : 0) * nb;
    }
    float lv[4][R][NT][4];     // the four leaves' values, in the order w, w+8, w+16, w+24
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int leaf = w + 8 * j;
        #pragma unroll
        for (int r = 0; r < R; r++)
            #pragma unroll
            for (int n = 0; n < NT; n++) lv[j][r][n][0] = lv[j][r][n][1] = lv[j][r][n][2] = lv[j][r][n][3] = 0.f;
        for (int b = leaf; b < nb; b += 32) {
            unsigned B[NT][3][2];
            float s0[NT], s1[NT];
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                #pragma unroll
                for (int l = 0; l < 3; l++) {
                    B[n][l][0] = bl[n] ? (unsigned)bq[n][(ulong64)b * 24 + l * 8 + tig] : 0u;
                    B[n][l][1] = bl[n] ? (unsigned)bq[n][(ulong64)b * 24 + l * 8 + 4 + tig] : 0u;
                }
                s0[n] = cl0[n] ? sc0[n][b] : 0.f; s1[n] = cl1[n] ? sc1[n][b] : 0.f;
            }
            #pragma unroll
            for (int r = 0; r < R; r++) {
                unsigned r0 = rbase + r * 16 + gid, r1 = r0 + 8;
                bool l0 = r0 < (unsigned)n_out, l1 = r1 < (unsigned)n_out;
                const uchar *p0 = wb + w_off + ((ulong64)(l0 ? r0 : 0) * nb + b) * 18;
                const uchar *p1 = wb + w_off + ((ulong64)(l1 ? r1 : 0) * nb + b) * 18;
                float d0 = __half2float(*(const __half *)p0), d1 = __half2float(*(const __half *)p1);
                const ushort16 *h0 = (const ushort16 *)(p0 + 2 + tig * 4), *h1 = (const ushort16 *)(p1 + 2 + tig * 4);
                unsigned u0 = (unsigned)h0[0] | ((unsigned)h0[1] << 16), u1 = (unsigned)h1[0] | ((unsigned)h1[1] << 16);
                unsigned A[4] = { __vsub4(u0 & 0x0F0F0F0Fu, 0x08080808u), __vsub4(u1 & 0x0F0F0F0Fu, 0x08080808u),
                                  __vsub4((u0 >> 4) & 0x0F0F0F0Fu, 0x08080808u), __vsub4((u1 >> 4) & 0x0F0F0F0Fu, 0x08080808u) };
                #pragma unroll
                for (int n = 0; n < NT; n++) {
                    int S[3][4];
                    #pragma unroll
                    for (int l = 0; l < 3; l++) { S[l][0] = S[l][1] = S[l][2] = S[l][3] = 0; mma_s8_16832(S[l], A, B[n][l]); }
                    float cc[4] = { __fmul_rn(d0, s0[n]), __fmul_rn(d0, s1[n]), __fmul_rn(d1, s0[n]), __fmul_rn(d1, s1[n]) };
                    #pragma unroll
                    for (int o = 0; o < 4; o++) {
                        // exact in int32: |s0*65536 + s1*256 + s2| <= 2,139,127,808 < 2^31
                        int tot = S[0][o] * 65536 + S[1][o] * 256 + S[2][o];
                        lv[j][r][n][o] = __fmaf_rn(__int2float_rn(tot), cc[o], lv[j][r][n][o]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int n = 0; n < NT; n++) {
            float root[4];
            #pragma unroll
            for (int o = 0; o < 4; o++) {
                float qa = lv[0][r][n][o] + lv[2][r][n][o];      // level 16: w += w+16
                float qb = lv[1][r][n][o] + lv[3][r][n][o];      // level 16: w+8 += w+24
                root[o] = qa + qb;                               // level 8: w += w+8
            }
            P[w][r * 16 + gid][n * 8 + tig * 2] = root[0]; P[w][r * 16 + gid][n * 8 + tig * 2 + 1] = root[1];
            P[w][r * 16 + gid + 8][n * 8 + tig * 2] = root[2]; P[w][r * 16 + gid + 8][n * 8 + tig * 2 + 1] = root[3];
        }
    __syncthreads();
    for (int o = threadIdx.x; o < R * 16 * 8 * NT; o += blockDim.x) {
        int rr = o / (8 * NT), c = o % (8 * NT);
        float s[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) s[k] = P[k][rr][c];
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            #pragma unroll
            for (int k = 0; k < off; k++) s[k] = s[k] + s[k + off];
        }
        unsigned row = rbase + rr, col = cb + c;
        if (row < (unsigned)n_out && col < (unsigned)cnt) y[(ulong64)col * ys + row] = s[0];
    }
}
#define LIMB_TILE(NAME, R, NT)                                                                     \
extern "C" __global__ void __launch_bounds__(256) NAME(const uchar *wb, ulong64 w_off, int n_in,  \
        int n_out, int cnt, int ys, float *y, const signed char *lq, const float *lsc) {         \
    limb_tile_body<R, NT>(wb, w_off, n_in, n_out, cnt, ys, y, lq, lsc);                          \
}
LIMB_TILE(k_moe_limb_tile_q4_0, 1, 1)
LIMB_TILE(k_moe_limb_tile2_q4_0, 2, 2)
