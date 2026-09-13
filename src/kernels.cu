// CUDA compute kernels: the full forward pass, one token or a prompt tile.
// 1:1 port of kernels.metal for the layout, with two additions for prompt
// tiles of up to MVB tokens:
//   - k_mv_*_b matvec variants decode each weight element once and FMA it
//     against every token column (weight-bandwidth reuse; generation keeps
//     the single-column k_mv_* variants, which are faster at batch 1)
//   - every small kernel takes the token index from blockIdx.y/z, so a tile
//     costs the same number of kernel launches as a single token — launch
//     overhead (severe under Windows WDDM) does not scale with tile size
// Compiled to PTX at development time (make ptx) and embedded via
// kernels_ptx.h; the driver JIT-compiles for the resident GPU.
#include <cuda_fp16.h>
#include <mma.h>

// mv_args / moe_args / rope_args / attn_args: ONE definition, shared with
// src/cuda.c. They are passed to kernels by value, so host and device must
// agree on the layout exactly (see the header).
#include "kernel_args.h"

typedef unsigned char  uchar;
typedef unsigned short ushort16;
typedef unsigned int   uint;
typedef unsigned long long ulong64;

static __device__ __forceinline__ float f16f(const uchar *p) {
    return __half2float(*(const __half *)p);
}

// ---------------------------------------------------------------- rmsnorm
// grid.y = token column; xs/ys = element stride between columns

extern "C" __global__ void k_rmsnorm(const float *x, float *y, const float *w,
                                     int n, float eps, int xs, int ys) {
    __shared__ float red[256];
    int tid = threadIdx.x, tpg = blockDim.x;
    x += (ulong64)blockIdx.y * xs;
    y += (ulong64)blockIdx.y * ys;
    float s = 0;
    for (int i = tid; i < n; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float r = rsqrtf(red[0] / n + eps);
    for (int i = tid; i < n; i += tpg) y[i] = x[i] * r * w[i];
}

// per-head RMSNorm (qwen3 Q/K norm): one block per (head, token)
// LAUNCH INVARIANT: the tree reduction below (off = tpg/2; off >>= 1) and the
// red[] indexing require blockDim.x to be a power of two and <= 128 (red[]'s
// size). Host launches this with 64 — do not exceed the buffer or break pow2.
extern "C" __global__ void k_qknorm(float *v, const float *w, int hd, float eps,
                                    int vs) {
    __shared__ float red[128];
    int tid = threadIdx.x, tpg = blockDim.x;
    float *x = v + (ulong64)blockIdx.y * vs + blockIdx.x * hd;
    float s = 0;
    for (int i = tid; i < hd; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float r = rsqrtf(red[0] / hd + eps);
    for (int i = tid; i < hd; i += tpg) x[i] = x[i] * r * w[i];
}

// ---------------------------------------------------------------- matvec
// One warp (32 lanes) per output row; lanes stride over blocks.
// 128 threads = 4 warps = 4 rows per block (same shape as the Metal version).
// k_mv_* handles one column (generation); k_mv_*_b applies each decoded
// weight to all MVB columns (x buffers are always MVB columns wide, so the
// unguarded reads for t >= batch touch valid, ignored memory).

#define MVT 16  // scalar-kernel tile: register arrays and xsm size
#define MVB 64  // activation BUFFER columns (keep in sync with cuda.c)


static __device__ __forceinline__ float warp_sum(float s) {
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
    return s;
}

#define MV_HEAD \
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); \
    unsigned lane = threadIdx.x & 31; \
    if (row >= (unsigned)a.n_out) return;

#define MV_TAIL \
    s = warp_sum(s); \
    if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;

#define MV_HEAD_B \
    MV_HEAD; \
    float s[MVT] = {0};

// apply one decoded weight w at element index idx to every token column
#define MV_FMA(w, idx) do { \
    float _w = (w); ulong64 _i = (idx); \
    _Pragma("unroll") \
    for (int t = 0; t < MVT; t++) s[t] += _w * x[(ulong64)t * a.xs + _i]; \
} while (0)

#define MV_TAIL_B \
    for (int t = 0; t < a.batch; t++) { \
        float r = warp_sum(s[t]); \
        if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r; \
    }

#define MV_PARAMS const uchar *wb, const float *x, float *y, mv_args a, const float *bias

extern "C" __global__ void k_mv_f32(MV_PARAMS) {
    MV_HEAD;
    const float *rw = (const float *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += rw[i] * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_f32_b(MV_PARAMS) {
    MV_HEAD_B;
    const float *rw = (const float *)(wb + a.w_off) + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) MV_FMA(rw[i], i);
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_f16(MV_PARAMS) {
    MV_HEAD;
    const __half *rw = (const __half *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += __half2float(rw[i]) * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_f16_b(MV_PARAMS) {
    MV_HEAD_B;
    const __half *rw = (const __half *)(wb + a.w_off) + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) MV_FMA(__half2float(rw[i]), i);
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 34;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 32; j++) t += (float)q[j] * xp[j];
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q8_0_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 34;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 32; j++) MV_FMA(d * (float)q[j], base + j);
    }
    MV_TAIL_B;
}

// -------------------------------------------------------------- prefill GEMM
// Real tiled GEMM replacements for the two prefill formats that matter on this
// machine (Q8_0, Q4_K). The batch _b kernels are compute/latency bound: each
// decoded weight issues MVB scattered global x-loads. These variants stage the
// x-tile columns into shared memory once per block, so every decoded weight
// FMAs against smem instead of global memory, and multiple warps (rows) reuse
// the same staged x.
//
// Correctness: the reduction is kept BIT-IDENTICAL to the _b kernels — lane b
// still owns k-blocks b, b+32, ...; the inner j-order is 0..31; the per-weight
// term is the same d*(float)q[j]; the final warp_sum tree is unchanged. Only
// the *source* of x changes (smem vs global), so results match the _b kernels
// exactly and greedy tokens are identical.

#define GEMM_WARPS 8            // output rows per block (warps)
#define Q8_CHUNK   32           // q8 blocks staged per k-iteration (== warp lanes)
#define SMPAD      33           // 32 + 1: makes per-lane smem reads conflict-free

// MVB=16 note: a 16-column f32 x-tile of 32 q8 blocks does not fit shared
// memory (16*1024*4 = 64 KB), and every narrower restructure measured slower
// than this proven 8-column shape (the kernel is smem-FMA bound, so per-weight
// work scales with columns and wider tiles buy nothing on this path). The
// kernel therefore keeps its 8-column tile and cuda.c splits a 16-token tile
// into two launches — identical arithmetic to the MVB=8 build.
#define Q8_COLS 8               // fixed column width of this kernel's tile

// xsm[t][blk_in_chunk*SMPAD + j] holds x column t, element (chunk*1024)+blk*32+j
extern "C" __global__ void k_gemm_q8_0(MV_PARAMS) {
    __shared__ float xsm[Q8_COLS][Q8_CHUNK * SMPAD];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 32;
    float s[Q8_COLS] = {0};
    const uchar *rw = wb + a.w_off + (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 34;

    for (int cs = 0; cs < nb; cs += Q8_CHUNK) {
        int cblocks = nb - cs < Q8_CHUNK ? nb - cs : Q8_CHUNK;
        int celems  = cblocks * 32;
        int base_e  = cs * 32;
        // coalesced cooperative load of this chunk's x into padded smem
        #pragma unroll
        for (int t = 0; t < Q8_COLS; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < celems; e += blockDim.x)
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out && lane < (unsigned)cblocks) {
            const uchar *blk = rw + (ulong64)(cs + lane) * 34;
            float d = f16f(blk);
            const signed char *q = (const signed char *)(blk + 2);
            int soff = (int)lane * SMPAD;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                float w = d * (float)q[j];
                #pragma unroll
                for (int t = 0; t < Q8_COLS; t++) s[t] += w * xsm[t][soff + j];
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        int cols = a.batch < Q8_COLS ? a.batch : Q8_COLS;
        for (int t = 0; t < cols; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

// -------------------------------------------------------------- decode GEMV
// Batch-1 (decode) matvec replacements for the two formats that matter here
// (Q8_0, Q4_K). The generic k_mv_* decode kernel maps one lane to a whole
// quant block, so consecutive lanes read 34-byte-strided (Q8) addresses: the
// loads never coalesce into 32-byte segments and the kernel tops out at ~18 %
// of peak weight bandwidth (memory-latency/coalescing bound, see diagnosis).
//
// These variants flip the mapping to LANE-PER-ELEMENT: within each block lane
// l owns element l, so the 32 lanes read 32 consecutive bytes -> one coalesced
// transaction per block. Each lane accumulates its own element-position across
// all blocks, then a single warp_sum reduces at the end. This REORDERS the
// k-reduction relative to k_mv_* (per-element partials summed once, vs the
// per-block d*(Sum q*x) of the originals), so identity is not bitwise and is
// established empirically by kernel-verify on the real models. Same block
// shape as k_mv_* (4 rows/block, 128 threads) so occupancy is unchanged; the
// win is purely coalescing.

// v2 (decode-bandwidth pass): the warp covers FOUR blocks per iteration —
// lane l owns bytes [(l&7)*4, +4) of block b0+(l>>3) — so each lane issues one
// aligned float4 x-load and two ushort quant loads per iteration instead of a
// single scalar element. Quart of the iterations, 4x the loads in flight; the
// per-lane partial is reduced by the same warp_sum. Reduction order differs
// from v1 (per-lane running s over its byte-quarter vs per-element), so
// identity vs CPU is established empirically by kernel-verify, like v1 was.
extern "C" __global__ void k_gemv_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 34;
    int bsub = (int)(lane >> 3);          // which of the 4 blocks this lane works
    int boff = ((int)lane & 7) * 4;       // this lane's 4-element chunk
    float s = 0;
    int b4 = nb & ~3;
    for (int b0 = 0; b0 < b4; b0 += 4) {
        const uchar *blk = rw + (ulong64)(b0 + bsub) * 34;
        float d = f16f(blk);
        // 34-byte stride keeps quants only 2-aligned: two ushort loads, then
        // sign-extend the four int8 lanes
        const uchar *qp = blk + 2 + boff;
        ushort16 u0 = *(const ushort16 *)qp, u1 = *(const ushort16 *)(qp + 2);
        int q0 = (int)(signed char)(u0 & 0xFF), q1 = (int)(signed char)(u0 >> 8);
        int q2 = (int)(signed char)(u1 & 0xFF), q3 = (int)(signed char)(u1 >> 8);
        const float4 xv = *(const float4 *)(x + (ulong64)(b0 + bsub) * 32 + boff);
        s += d * ((float)q0 * xv.x + (float)q1 * xv.y +
                  (float)q2 * xv.z + (float)q3 * xv.w);
    }
    // tail blocks (nb not a multiple of 4): v1's element-per-lane mapping
    for (int b = b4; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        s += d * ((float)q[lane] * x[(ulong64)b * 32 + lane]);
    }
    MV_TAIL;
}

// Q4_0 decode GEMV — the twin k_gemv_q8_0 never got (2026-08-13). Without it
// Q4_0 decode fell through to k_mv_q4_0, where one lane walks a whole 32-element
// block with a serial 16-iteration scalar loop: measured ~55 GB/s of implied
// weight bandwidth against 250-330 GB/s for every Q4_K/Q8_0 model on the same
// slice, a 5x gap that was pure kernel coverage, not arithmetic.
//
// Same shape as k_gemv_q8_0: four blocks in flight across the warp, eight lanes
// per block. A q4_0 block packs element j in the low nibble of byte j and
// element j+16 in the high nibble, so a lane taking two adjacent quant bytes
// owns four elements — two contiguous at boff and two contiguous at boff+16 —
// which is two aligned float2 activation loads and one 2-aligned ushort weight
// load (the 18-byte block stride never gives more than 2-alignment).
//
// The reduction is reordered relative to k_mv_q4_0, exactly as k_gemv_q4_K is
// relative to k_mv_q4_K, so identity is an empirical gate (kernel-verify +
// cpu_cuda_check), not an accumulation-order argument.
extern "C" __global__ void k_gemv_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    int bsub = (int)(lane >> 3);          // which of the 4 blocks this lane works
    int boff = ((int)lane & 7) * 2;       // this lane's 2 quant bytes
    float s = 0;
    int b4 = nb & ~3;
    for (int b0 = 0; b0 < b4; b0 += 4) {
        const uchar *blk = rw + (ulong64)(b0 + bsub) * 18;
        float d = f16f(blk);
        ushort16 u = *(const ushort16 *)(blk + 2 + boff);
        int q0 = (int)( u        & 0xF) - 8;   // byte boff  low  -> elem boff
        int q1 = (int)((u >>  4) & 0xF) - 8;   // byte boff  high -> elem boff+16
        int q2 = (int)((u >>  8) & 0xF) - 8;   // byte boff+1 low  -> elem boff+1
        int q3 = (int)((u >> 12) & 0xF) - 8;   // byte boff+1 high -> elem boff+17
        const float *xp = x + (ulong64)(b0 + bsub) * 32 + boff;
        float2 xlo = *(const float2 *)xp;
        float2 xhi = *(const float2 *)(xp + 16);
        s += d * ((float)q0 * xlo.x + (float)q2 * xlo.y +
                  (float)q1 * xhi.x + (float)q3 * xhi.y);
    }
    // tail blocks (nb not a multiple of 4): one element per lane
    for (int b = b4; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        int j = (int)lane & 15, hi = (int)lane >> 4;
        int qv = hi ? (q[j] >> 4) : (q[j] & 0xF);
        s += d * (float)(qv - 8) * x[(ulong64)b * 32 + lane];
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_0_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)((int)(q[j] & 0xF) - 8), base + j);
            MV_FMA(d * (float)((int)(q[j] >> 4)  - 8), base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q4_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 20;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 20;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        const uchar *q = blk + 4;
        const float *xp = x + b * 32;
        float t = 0, sx = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)(q[j] & 0xF) * xp[j] + (float)(q[j] >> 4) * xp[j + 16];
            sx += xp[j] + xp[j + 16];
        }
        s += d * t + mm * sx;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_1_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 20;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 20;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        const uchar *q = blk + 4;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)(q[j] & 0xF) + mm, base + j);
            MV_FMA(d * (float)(q[j] >> 4)  + mm, base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q5_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 22;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 22;
        float d = f16f(blk);
        uint qh = (uint)blk[2] | ((uint)blk[3] << 8) |
                  ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        const uchar *q = blk + 6;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            int x0 = (int)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
            int x1 = (int)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) - 16;
            t += x0 * xp[j] + x1 * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_0_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 22;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 22;
        float d = f16f(blk);
        uint qh = (uint)blk[2] | ((uint)blk[3] << 8) |
                  ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        const uchar *q = blk + 6;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            int x0 = (int)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
            int x1 = (int)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) - 16;
            MV_FMA(d * (float)x0, base + j);
            MV_FMA(d * (float)x1, base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q5_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 24;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 24;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        uint qh = (uint)blk[4] | ((uint)blk[5] << 8) |
                  ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
        const uchar *q = blk + 8;
        const float *xp = x + b * 32;
        float t = 0, sx = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) * xp[j] +
                 (float)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) * xp[j + 16];
            sx += xp[j] + xp[j + 16];
        }
        s += d * t + mm * sx;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_1_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 24;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 24;
        float d  = f16f(blk);
        float mm = f16f(blk + 2);
        uint qh = (uint)blk[4] | ((uint)blk[5] << 8) |
                  ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
        const uchar *q = blk + 8;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) + mm, base + j);
            MV_FMA(d * (float)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) + mm, base + j + 16);
        }
    }
    MV_TAIL_B;
}

static __device__ __forceinline__ void get_scale_min_k4(int j, const uchar *q,
                                                        uchar *d, uchar *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

// bf16 is the top 16 bits of an f32: widening is a shift and a reinterpret,
// with no rounding, so this matches bf16_to_f32() in fp16.h exactly. Read as
// ushort and widened by hand rather than via __nv_bfloat16, which would pull
// in a header the PTX build does not otherwise need.
static __device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float((unsigned int)h << 16);
}

extern "C" __global__ void k_mv_bf16(MV_PARAMS) {
    MV_HEAD;
    const unsigned short *rw =
        (const unsigned short *)(wb + a.w_off) + (ulong64)row * a.n_in;
    float s = 0;
    for (int i = lane; i < a.n_in; i += 32) s += bf16f(rw[i]) * x[i];
    MV_TAIL;
}

extern "C" __global__ void k_mv_bf16_b(MV_PARAMS) {
    MV_HEAD_B;
    const unsigned short *rw =
        (const unsigned short *)(wb + a.w_off) + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) MV_FMA(bf16f(rw[i]), i);
    MV_TAIL_B;
}

// ports quants.c dq_q2_K into the warp-per-row dot. Unlike q3_K the scale byte
// carries BOTH a scale (low nibble) and a min (high nibble), and the min is
// subtracted from the dequantized weight rather than scaling it -- so the value
// handed to the accumulate is (dl*q - ml), not a product.
#define Q2K_SETUP \
    float d = f16f(blk + 80), dmin = f16f(blk + 82); \
    const uchar *q2scales = blk; \
    const uchar *qbase = blk + 16;

extern "C" __global__ void k_mv_q2_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 84;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 84;
        Q2K_SETUP;
        const float *xp = x + (ulong64)b * 256;
        int pos = 0, is = 0; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uchar scb = q2scales[is++];
                float dl = d * (scb & 0xF), ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    s += (dl * (float)((q[l] >> shift) & 3) - ml) * xp[pos++];
                scb = q2scales[is++];
                dl = d * (scb & 0xF); ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    s += (dl * (float)((q[l+16] >> shift) & 3) - ml) * xp[pos++];
                shift += 2;
            }
            q += 32;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q2_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 84;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 84;
        Q2K_SETUP;
        ulong64 base = (ulong64)b * 256;
        int pos = 0, is = 0; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uchar scb = q2scales[is++];
                float dl = d * (scb & 0xF), ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)((q[l] >> shift) & 3) - ml, base + pos++);
                scb = q2scales[is++];
                dl = d * (scb & 0xF); ml = dmin * (scb >> 4);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)((q[l+16] >> shift) & 3) - ml, base + pos++);
                shift += 2;
            }
            q += 32;
        }
    }
    MV_TAIL_B;
}

// Q3_K: 110-byte block, 256 elements. Layout: hmask[32] (one high bit per
// weight), qs[64] (2-bit low bits), scales[12] (packed 16x 6-bit, bias -32),
// d (f16 super-scale). Weight = d*(scale-32)*((2bit) - (highbit?0:4)). This
// ports quants.c dq_q3_K into the warp-per-row dot (dequant fused into the
// accumulate), matching the CPU reference bit-for-bit modulo reduction order.
#define Q3K_UNPACK_SCALES \
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu; \
    const uchar *sc = blk + 96; \
    uint a0 = sc[0] | (sc[1]<<8) | (sc[2]<<16) | ((uint)sc[3]<<24); \
    uint a1 = sc[4] | (sc[5]<<8) | (sc[6]<<16) | ((uint)sc[7]<<24); \
    uint a2 = sc[8] | (sc[9]<<8) | (sc[10]<<16) | ((uint)sc[11]<<24); \
    uint xs[4]; \
    xs[2] = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4); \
    xs[3] = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4); \
    xs[0] = ( a0       & kmask2) | (((a2 >> 0) & kmask1) << 4); \
    xs[1] = ( a1       & kmask2) | (((a2 >> 2) & kmask1) << 4); \
    const signed char *q3scales = (const signed char *)xs; \
    float d_all = f16f(blk + 108); \
    const uchar *hm = blk; \
    const uchar *qbase = blk + 32;

extern "C" __global__ void k_mv_q3_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 110;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 110;
        Q3K_UNPACK_SCALES;
        const float *xp = x + (ulong64)b * 256;
        int pos = 0, is = 0; uchar mbit = 1; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    s += dl * (float)(((q[l] >> shift) & 3) - ((hm[l] & mbit) ? 0 : 4)) * xp[pos++];
                dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    s += dl * (float)(((q[l+16] >> shift) & 3) - ((hm[l+16] & mbit) ? 0 : 4)) * xp[pos++];
                shift += 2; mbit <<= 1;
            }
            q += 32;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q3_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 110;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 110;
        Q3K_UNPACK_SCALES;
        ulong64 base = (ulong64)b * 256;
        int pos = 0, is = 0; uchar mbit = 1; const uchar *q = qbase;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)(((q[l] >> shift) & 3) - ((hm[l] & mbit) ? 0 : 4)), base + pos++);
                dl = d_all * (q3scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    MV_FMA(dl * (float)(((q[l+16] >> shift) & 3) - ((hm[l+16] & mbit) ? 0 : 4)), base + pos++);
                shift += 2; mbit <<= 1;
            }
            q += 32;
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_q4_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 144;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 144;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *q  = blk + 16;
        const float *xp = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
            const uint4 *q16 = (const uint4 *)q;   // blk+16 is 16B-aligned
            for (int v = 0; v < 2; v++) {
                uint4 w = q16[v];
                uint ws[4] = { w.x, w.y, w.z, w.w };
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int l = v * 16 + c * 4 + k;
                        uint b8 = (ws[c] >> (8 * k)) & 0xFFu;
                        t1 += (float)(b8 & 0xF) * xp[l];      sx1 += xp[l];
                        t2 += (float)(b8 >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
                    }
                }
            }
            s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
            q += 32; is += 2; xp += 64;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q4_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 144;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 144;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *q  = blk + 16;
        ulong64 base = (ulong64)b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            const uint4 *q16 = (const uint4 *)q;   // blk+16 is 16B-aligned
            for (int v = 0; v < 2; v++) {
                uint4 w = q16[v];
                uint ws[4] = { w.x, w.y, w.z, w.w };
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int l = v * 16 + c * 4 + k;
                        uint b8 = (ws[c] >> (8 * k)) & 0xFFu;
                        MV_FMA(d1 * (float)(b8 & 0xF) - mm1, base + j + l);
                        MV_FMA(d2 * (float)(b8 >> 4)  - mm2, base + j + l + 32);
                    }
                }
            }
            q += 32; is += 2;
        }
    }
    MV_TAIL_B;
}

// Q4_K prefill GEMM. Unlike Q8_0 the 256-element block is too wide to keep
// lane==k-block (32 blocks -> 8192 x elements won't fit in smem), so the warp
// instead walks k-blocks sequentially and its 32 lanes cooperatively reduce the
// 256 elements of each block (8 elements/lane). x for the current block is
// staged into smem (small, 256*MVB floats) so decoded weights FMA against smem.
// This REORDERS the k-reduction relative to k_mv_q4_K_b, so it is not bitwise
// identical — token-identity is verified empirically by kernel-verify on the
// real Q4_K model. Lane l owns elements [l*8, l*8+8): all in scale group l/4,
// quant segment l/8, byte offset (l&3)*8, lower nibble iff group even.
extern "C" __global__ void k_gemm_q4_K(MV_PARAMS) {
    __shared__ float xsm[MVT][256];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off +
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 144;
    float s[MVT] = {0};
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int lo    = (((int)lane >> 2) & 1) == 0;
    int bbase = ((int)lane & 3) * 8;      // byte offset within the segment

    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 144;
        int base_e = b * 256;
        #pragma unroll
        for (int t = 0; t < MVT; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < 256; e += blockDim.x) xsm[t][e] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out) {
            float dd   = f16f(blk);
            float dmin = f16f(blk + 2);
            const uchar *sc = blk + 4;
            const uchar *q  = blk + 16 + ji * 32;
            uchar sg, mg;
            get_scale_min_k4(g, sc, &sg, &mg);
            float dg = dd * (float)sg, mmg = dmin * (float)mg;
            int el = (int)lane * 8;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                uchar byte = q[bbase + k];
                int nib = lo ? (byte & 0xF) : (byte >> 4);
                float w = dg * (float)nib - mmg;
                #pragma unroll
                for (int t = 0; t < MVT; t++) s[t] += w * xsm[t][el + k];
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        for (int t = 0; t < a.batch; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

// ---------------------------------------------------- tensor-core Q4_K GEMM
// v2, MMQ-style (suite plan P1-prefill lever b; supersedes the per-warp v1,
// which re-dequantized per 16-element K-step through a scalar index helper and
// measured ~6-7x SLOWER than the scalar GEMM at 8 tokens). The economics that
// make this one fast:
//   - the whole BLOCK cooperates on one 64-row x 128-K fp16 weight tile per
//     step, dequantized ONCE into shared memory with 8-byte quant loads —
//     two threads per row, each unpacking one contiguous 64-element segment
//     (exactly two scale groups) — then all four warps' MMAs reuse it;
//   - activations are staged as a 128-K x 16-token fp16 tile (vectorized
//     float4 reads, zero-padded past a.batch), tiny enough to live in L2
//     across the row-block sweep;
//   - each warp owns a 16-row m16n16k16 fragment strip: 4 warps x 16 = 64
//     rows per block, accumulated in fp32 across the full K without ever
//     leaving registers (the "register-accumulate" of the plan item).
// Numerics: identical per-element weight values to k_gemm_q4_K, rounded to
// fp16 operands with fp32 accumulation — same numeric class as v1, still
// behind the RUNNER_CUDA_TC opt-in + tolerance gate, never the default path.
// Compiles to compute_75 PTX (fp16 WMMA is Turing+; no bf16 in sm_75).

#define TC_ROWS 64    // output rows per block (4 warps x 16-row fragments)
#define TC_K    128   // K-elements staged per step (half a q4_K super-block)
#define TC_N    64    // token columns per tile (was 16; widened to amortise the weight dequantisation)

extern "C" __global__ void k_gemm_q4_K_tc(MV_PARAMS) {
    using namespace nvcuda::wmma;
    const int tid  = threadIdx.x;         // 128 threads = 4 warps
    const int warp = tid >> 5;
    const unsigned row0 = blockIdx.x * TC_ROWS;
    __shared__ __half sh_w[TC_ROWS * TC_K];   // row-major, ldm = TC_K
    __shared__ __half sh_x[TC_N * TC_K];      // col-major (k,t) at [t*TC_K+k]
    // The epilogue tile ALIASES the weight tile: sh_w is 64*128 halves = 16 KB
    // and sh_c needs TC_ROWS*TC_N floats, and their lifetimes are disjoint
    // (weights are dead once the last MMA has run). Separate arrays would push
    // the block past the 48 KB static shared cap at a wide tile.
    float *sh_c = (float *)sh_w;

    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc[TC_N / 16];
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++) fill_fragment(fc[n], 0.0f);

    int nb = a.n_in / 256;
    // this thread stages one 64-element segment of one row per K-step:
    // row srow, segment sseg (0 or 1) of the 128-K tile
    int srow = tid >> 1, sseg = tid & 1;

    for (int b = 0; b < nb; b++) {
        #pragma unroll
        for (int koff = 0; koff < 256; koff += TC_K) {
            // ---- stage weights: 64 rows x 128 K, dequantized once ----
            {
                unsigned gr = row0 + srow;
                __half *dst = sh_w + srow * TC_K + sseg * 64;
                if (gr < (unsigned)a.n_out) {
                    const uchar *blk = wb + a.w_off +
                                       (ulong64)gr * nb * 144 + (ulong64)b * 144;
                    int e0 = koff + sseg * 64;         // segment base element
                    float dd   = f16f(blk);
                    float dmin = f16f(blk + 2);
                    uchar sg0, mg0, sg1, mg1;
                    get_scale_min_k4((e0 >> 5) + 0, blk + 4, &sg0, &mg0);
                    get_scale_min_k4((e0 >> 5) + 1, blk + 4, &sg1, &mg1);
                    float dg0 = dd * (float)sg0, mm0 = dmin * (float)mg0;
                    float dg1 = dd * (float)sg1, mm1 = dmin * (float)mg1;
                    // 64 consecutive elements live in ONE 32-byte quant
                    // segment: bytes b8 carry the low nibbles of elements
                    // 0..31 and the high nibbles of elements 32..63
                    const uint2 *q8 = (const uint2 *)(blk + 16 + (e0 >> 6) * 32);
                    #pragma unroll
                    for (int v = 0; v < 4; v++) {
                        uint2 qv = q8[v];
                        #pragma unroll
                        for (int w = 0; w < 2; w++) {
                            uint bits = w ? qv.y : qv.x;
                            int base = v * 8 + w * 4;
                            uint lo = bits & 0x0F0F0F0Fu;
                            uint hi = (bits >> 4) & 0x0F0F0F0Fu;
                            dst[base + 0]  = __float2half(dg0 * (float)(lo & 0xFF) - mm0);
                            dst[base + 1]  = __float2half(dg0 * (float)((lo >> 8) & 0xFF) - mm0);
                            dst[base + 2]  = __float2half(dg0 * (float)((lo >> 16) & 0xFF) - mm0);
                            dst[base + 3]  = __float2half(dg0 * (float)(lo >> 24) - mm0);
                            dst[base + 32] = __float2half(dg1 * (float)(hi & 0xFF) - mm1);
                            dst[base + 33] = __float2half(dg1 * (float)((hi >> 8) & 0xFF) - mm1);
                            dst[base + 34] = __float2half(dg1 * (float)((hi >> 16) & 0xFF) - mm1);
                            dst[base + 35] = __float2half(dg1 * (float)(hi >> 24) - mm1);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            // ---- stage activations: 128 K x 16 tokens, vectorized ----
            {
                int col = tid / 2, part = tid % 2;
                __half *dst = sh_x + col * TC_K + part * 64;
                if (col < a.batch) {
                    const float *xg = x + (ulong64)col * a.xs + b * 256 + koff
                                      + part * 64;
                    #pragma unroll
                    for (int v = 0; v < 16; v++) {
                        float4 xv = *(const float4 *)(xg + v * 4);
                        dst[v * 4 + 0] = __float2half(xv.x);
                        dst[v * 4 + 1] = __float2half(xv.y);
                        dst[v * 4 + 2] = __float2half(xv.z);
                        dst[v * 4 + 3] = __float2half(xv.w);
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            __syncthreads();
            // ---- 8 MMA K-steps over the staged tile ----
            const __half *wt = sh_w + warp * 16 * TC_K;
            #pragma unroll
            for (int k = 0; k < TC_K; k += 16) {
                load_matrix_sync(fa, wt + k, TC_K);
                // one staged weight tile, reused across every n-tile: this is
                // the entire point of widening TC_N
                #pragma unroll
                for (int n = 0; n < TC_N / 16; n++) {
                    load_matrix_sync(fb, sh_x + n * 16 * TC_K + k, TC_K);
                    mma_sync(fc[n], fa, fb, fc[n]);
                }
            }
            __syncthreads();
        }
    }
    // sh_c aliases sh_w: every warp must be done reading weights first
    __syncthreads();
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++)
        store_matrix_sync(sh_c + warp * 16 * TC_N + n * 16, fc[n], TC_N,
                          mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < TC_ROWS * TC_N; idx += blockDim.x) {
        int rr = idx / TC_N, tt = idx % TC_N;
        unsigned gr = row0 + rr;
        if (gr < (unsigned)a.n_out && tt < a.batch) {
            float r = sh_c[rr * TC_N + tt];
            y[(ulong64)tt * a.ys + gr] = a.has_bias ? r + bias[gr] : r;
        }
    }
}

// ------------------------------------- tensor-core Q8_0 / Q4_0 GEMM twins
// P3 of the moe-gpu-routing spec: the same MMQ-style structure as
// k_gemm_q4_K_tc — a 64-row x 128-K fp16 weight tile dequantized once per
// step by the whole block, a 128-K x 16-token fp16 activation tile, four
// warps of m16n16k16 MMA strips with fp32 accumulation. Per-element weight
// values match the scalar kernels' dequantization exactly, rounded to fp16
// operands. Both are OPT-IN (RUNNER_CUDA_TC / per-(type, arch) promotion
// via make test-tc-tol); tc_promoted() does not list them, so the default
// path is untouched.
//
// Unlike Q4_K (256-element super-blocks, so every K-step is block-aligned),
// these formats have 32-element blocks and real models carry K dims that
// are not 128-multiples (gemma-4-MoE n_ff_exp = 704), so the K loop is
// tail-safe: elements past a.n_in stage as zeros, which the MMA then
// accumulates harmlessly.

// Stage one 64-element segment of one row: two 32-element quant blocks,
// dequantized with the scalar kernels' exact per-element arithmetic.
static __device__ __forceinline__ void tc_stage_q8_0(__half *dst,
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
        const uchar *blk = rw + (ulong64)(base / 32) * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        #pragma unroll
        for (int j = 0; j < 32; j++)
            dst[half * 32 + j] = __float2half(d * (float)q[j]);
    }
    (void)nb;
}

static __device__ __forceinline__ void tc_stage_q4_0(__half *dst,
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
        const uchar *blk = rw + (ulong64)(base / 32) * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            dst[half * 32 + j]      = __float2half(d * (float)((int)(q[j] & 0xF) - 8));
            dst[half * 32 + j + 16] = __float2half(d * (float)((int)(q[j] >> 4)  - 8));
        }
    }
    (void)nb;
}

// 32-byte-block twin of k_gemm_q4_K_tc (Q8_0, Q4_0): same 64-row x TC_K fp16
// weight tile, staged through STAGE instead of the q4_K super-block decoder.
//
// FIXED 2026-08-13 — this macro was left at the original TC_N=16 shape when
// TC_N was widened 16 -> 64 in 6cf8c70 (2026-08-08 — NOT 2026-07-29, which is
// the Q4_K promotion date; the correction matters because it sets the exposure
// window: 2026-08-08 to 2026-08-13, and only for a type promoted or forced in
// it — Q8_0 from 2026-08-09, gemma4 Q4_0 from 2026-08-12). It staged 16
// activation columns,
// accumulated a single 16-wide fragment and stored one 16x16 tile, then the
// epilogue wrote sh_c columns 16..batch-1 — never written, so UNINITIALISED
// shared memory — into y. Q8_0 is promoted by default, so the default CUDA
// prefill path produced corrupt logits for every prompt batch above 16 (the
// runner's own default is -b 64). It reproduced as: greedy output identical
// to the scalar path at -b 16, divergent at -b 32 and -b 64.
//
// The tolerance gate did not catch it, and "it was measured before the
// widening" turned out to be the wrong explanation: re-run on 2026-08-13, the
// pre-fix kernel PASSES the teacher-forced gate on phi3 and gemma4 q4_0
// ("BIT-IDENTICAL over 448/820 dispatches") while the same binary diverges in
// free-running greedy at -b 64. The gate ran at n_ctx = n_tok + 8; at a
// production context the block inherits ZEROED shared memory and the
// corruption surfaces. test_tc_tol now carries a free-running arm at ctx 4096
// that fails against this kernel.
#define TC_GEMM_32B(NAME, STAGE, BLKBYTES)                                     \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    using namespace nvcuda::wmma;                                              \
    const int tid  = threadIdx.x;                                              \
    const int warp = tid >> 5;                                                 \
    const unsigned row0 = blockIdx.x * TC_ROWS;                                \
    __shared__ __half sh_w[TC_ROWS * TC_K];                                    \
    __shared__ __half sh_x[TC_N * TC_K];                                       \
    __shared__ float  sh_c[TC_ROWS * TC_N];                                    \
    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;                      \
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;                      \
    fragment<accumulator, 16, 16, 16, float> fc[TC_N / 16];                    \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < TC_N / 16; n++) fill_fragment(fc[n], 0.0f);            \
    int nb = a.n_in / 32;                                                      \
    int srow = tid >> 1, sseg = tid & 1;                                       \
    for (int ks = 0; ks < a.n_in; ks += TC_K) {                                \
        {                                                                      \
            unsigned gr = row0 + srow;                                         \
            __half *dst = sh_w + srow * TC_K + sseg * 64;                      \
            if (gr < (unsigned)a.n_out) {                                      \
                const uchar *rw = wb + a.w_off +                               \
                                  (ulong64)gr * nb * BLKBYTES;                 \
                STAGE(dst, rw, nb, ks + sseg * 64, a.n_in);                    \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);      \
            }                                                                  \
        }                                                                      \
        {                                                                      \
            /* TC_N columns x TC_K elements with 128 threads: two 64-element */\
            /* halves per thread, guarded at 32-element granularity because  */\
            /* n_in is a 32-multiple (quant blocks) but need not be a        */\
            /* 64-multiple — a 64-wide part can straddle the end of the row. */\
            int col = tid >> 1, part = tid & 1;                                \
            __half *dst = sh_x + col * TC_K + part * 64;                       \
            _Pragma("unroll")                                                  \
            for (int h = 0; h < 2; h++) {                                      \
                __half *d2 = dst + h * 32;                                     \
                int e0 = ks + part * 64 + h * 32;                              \
                if (col < a.batch && e0 < a.n_in) {                            \
                    const float *xg = x + (ulong64)col * a.xs + e0;            \
                    _Pragma("unroll")                                          \
                    for (int v = 0; v < 8; v++) {                              \
                        float4 xv = *(const float4 *)(xg + v * 4);             \
                        d2[v * 4 + 0] = __float2half(xv.x);                    \
                        d2[v * 4 + 1] = __float2half(xv.y);                    \
                        d2[v * 4 + 2] = __float2half(xv.z);                    \
                        d2[v * 4 + 3] = __float2half(xv.w);                    \
                    }                                                          \
                } else {                                                       \
                    _Pragma("unroll")                                          \
                    for (int e = 0; e < 32; e++) d2[e] = __float2half(0.0f);   \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const __half *wt = sh_w + warp * 16 * TC_K;                            \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < TC_K; k += 16) {                                   \
            load_matrix_sync(fa, wt + k, TC_K);                                \
            _Pragma("unroll")                                                  \
            for (int n = 0; n < TC_N / 16; n++) {                              \
                load_matrix_sync(fb, sh_x + n * 16 * TC_K + k, TC_K);          \
                mma_sync(fc[n], fa, fb, fc[n]);                                \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < TC_N / 16; n++)                                        \
        store_matrix_sync(sh_c + warp * 16 * TC_N + n * 16, fc[n], TC_N,       \
                          mem_row_major);                                      \
    __syncthreads();                                                           \
    for (int idx = tid; idx < TC_ROWS * TC_N; idx += blockDim.x) {             \
        int rr = idx / TC_N, tt = idx % TC_N;                                  \
        unsigned gr = row0 + rr;                                               \
        if (gr < (unsigned)a.n_out && tt < a.batch) {                          \
            float r = sh_c[rr * TC_N + tt];                                    \
            y[(ulong64)tt * a.ys + gr] = a.has_bias ? r + bias[gr] : r;        \
        }                                                                      \
    }                                                                          \
}

TC_GEMM_32B(k_gemm_q8_0_tc, tc_stage_q8_0, 34)
TC_GEMM_32B(k_gemm_q4_0_tc, tc_stage_q4_0, 18)

// Q4_K decode GEMV: lane-per-element coalesced variant of k_mv_q4_K. Lane l
// owns the 8 elements [l*8, l*8+8), reusing the exact per-element weight
// geometry of k_gemm_q4_K (which passed identity empirically) -> group l/4,
// quant segment l/8, byte offset (l&3)*8, lower nibble iff group even. x is
// read from global (single decode column, small + L1-cached); the 128-byte
// quant region of each block is read coalesced across the warp. Reduction is
// reordered vs k_mv_q4_K -> identity is verified empirically.
// v2 (decode-bandwidth pass): same lane-per-8-elements geometry, but the 8
// quant bytes arrive as one aligned uint2 (blk+16 + ji*32 + bbase is 8-aligned
// within the 144-byte block) and the 8 x elements as two aligned float4s. The
// per-group affine is factored out — s += dg*sum(nib*x) - mmg*sum(x) — so the
// inner 8 elements are pure FMAs on unpacked nibbles. Per-element weights are
// unchanged from v1; only load shape and summation order differ, so identity
// vs CPU stays empirically gated by kernel-verify.
extern "C" __global__ void k_gemv_q4_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 144;
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;   // low or high nibble
    int bbase = ((int)lane & 3) * 8;      // byte offset within the segment
    float s = 0;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 144;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 16 + ji * 32 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 176;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 176;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *qh = blk + 16;
        const uchar *q  = blk + 48;
        const float *xp = x + b * 256;
        int is = 0;
        uchar u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            for (int l = 0; l < 32; l++) {
                s += (d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1) * xp[l];
                s += (d2 * (float)((q[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - mm2) * xp[l + 32];
            }
            q += 32; is += 2; xp += 64; u1 <<= 2; u2 <<= 2;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q5_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 176;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 176;
        float d    = f16f(blk);
        float dmin = f16f(blk + 2);
        const uchar *sc = blk + 4;
        const uchar *qh = blk + 16;
        const uchar *q  = blk + 48;
        ulong64 base = (ulong64)b * 256;
        int is = 0;
        uchar u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uchar s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            float d1 = d * s1, mm1 = dmin * m1;
            float d2 = d * s2, mm2 = dmin * m2;
            for (int l = 0; l < 32; l++) {
                MV_FMA(d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1, base + j + l);
                MV_FMA(d2 * (float)((q[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - mm2, base + j + l + 32);
            }
            q += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
    MV_TAIL_B;
}

// Q5_K decode GEMV: Q4_K's lane-per-element geometry with Q5_K's extra high
// bit. Lane l owns eight consecutive elements in scale/min group l/4. The
// nibble comes from quant segment l/8 and the fifth bit is bit l/4 of qh.
// A warp therefore processes each 256-element block cooperatively, coalescing
// the 128-byte qs region and 32-byte qh region instead of reading 176-byte-
// strided blocks across lanes.
// v2 (decode-bandwidth pass): the Q4_K v2 load shape — one aligned uint2 for
// the 8 nibbles, one aligned uint2 for the 8 qh bytes (blk+16+bbase is
// 8-aligned in the 176-byte block), two float4 x loads, and the factored
// s += dg*sum(qv*x) - mmg*sum(x). Weights per element unchanged; identity vs
// CPU empirically gated by kernel-verify as before.
extern "C" __global__ void k_gemv_q5_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 176;
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;
    int bbase = ((int)lane & 3) * 8;      // byte offset within segment/qh
    int hshift = g;                       // qh bit for this group
    float s = 0;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 176;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 48 + ji * 32 + bbase);
        uint2 hv = *(const uint2 *)(blk + 16 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        // fifth bit: bit `g` of each qh byte, moved to value 16
        uint h0 = ((hv.x >> hshift) & 0x01010101u) << 4;
        uint h1 = ((hv.y >> hshift) & 0x01010101u) << 4;
        v0 += h0; v1 += h1;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MV_TAIL;
}

// Q5_K prefill GEMM: the decode geometry above with the current x block staged
// in shared memory. Eight warps reuse that tile for eight output rows; each
// warp reduces one 256-element weight block cooperatively.
extern "C" __global__ void k_gemm_q5_K(MV_PARAMS) {
    __shared__ float xsm[MVT][256];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off +
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 176;
    int g     = (int)(lane >> 2);         // scale/min group 0..7
    int ji    = (int)(lane >> 3);         // 32-byte quant segment 0..3
    int lo    = (((int)lane >> 2) & 1) == 0;
    int bbase = ((int)lane & 3) * 8;      // byte offset within segment/qh
    int hmask = 1 << g;
    float s[MVT] = {0};

    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 176;
        int base_e = b * 256;
        #pragma unroll
        for (int t = 0; t < MVT; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < 256; e += blockDim.x) xsm[t][e] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out) {
            float dd   = f16f(blk);
            float dmin = f16f(blk + 2);
            const uchar *sc = blk + 4;
            const uchar *qh = blk + 16;
            const uchar *q  = blk + 48 + ji * 32;
            uchar sg, mg;
            get_scale_min_k4(g, sc, &sg, &mg);
            float dg = dd * (float)sg, mmg = dmin * (float)mg;
            int el = (int)lane * 8;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                uchar byte = q[bbase + k];
                int qv = (lo ? (byte & 0xF) : (byte >> 4)) +
                         ((qh[bbase + k] & hmask) ? 16 : 0);
                float w = dg * (float)qv - mmg;
                #pragma unroll
                for (int t = 0; t < MVT; t++) s[t] += w * xsm[t][el + k];
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        for (int t = 0; t < a.batch; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

extern "C" __global__ void k_mv_q6_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 210;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 210;
        const uchar *ql = blk;
        const uchar *qh = blk + 128;
        const signed char *sc = (const signed char *)(blk + 192);
        float d = f16f(blk + 208);
        const float *xp = x + b * 256;
        for (int half_i = 0; half_i < 2; half_i++) {
            float t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int l = 0; l < 32; l++) {
                int is = (l / 16) & 1;
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                t[is * 4 + 0] += q1 * xp[l];
                t[is * 4 + 1] += q2 * xp[l + 32];
                t[is * 4 + 2] += q3 * xp[l + 64];
                t[is * 4 + 3] += q4 * xp[l + 96];
            }
            s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                      sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
            ql += 64; qh += 32; sc += 8; xp += 128;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_q6_K_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 210;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 210;
        const uchar *ql = blk;
        const uchar *qh = blk + 128;
        const signed char *sc = (const signed char *)(blk + 192);
        float d = f16f(blk + 208);
        ulong64 base = (ulong64)b * 256;
        for (int half_i = 0; half_i < 2; half_i++) {
            for (int l = 0; l < 32; l++) {
                int is = (l / 16) & 1;
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                MV_FMA(d * (float)(sc[is] * q1),     base + l);
                MV_FMA(d * (float)(sc[2 + is] * q2), base + l + 32);
                MV_FMA(d * (float)(sc[4 + is] * q3), base + l + 64);
                MV_FMA(d * (float)(sc[6 + is] * q4), base + l + 96);
            }
            ql += 64; qh += 32; sc += 8; base += 128;
        }
    }
    MV_TAIL_B;
}

// Q6_K decode GEMV: lane-per-element coalesced variant of k_mv_q6_K. The
// generic k_mv_q6_K maps one lane to a whole 210-byte block, so consecutive
// lanes read 210-byte-strided addresses (uncoalesced). This variant makes the
// warp process each block cooperatively: lane l owns the four sub-positions
// {l, l+32, l+64, l+96} within each of the block's two 128-element halves (8
// elements total). Then ql[l]/ql[l+32]/qh[l] each read 32 consecutive bytes
// across the warp -> coalesced. Per-element weight d*sc[base+is]*q matches
// k_mv_q6_K exactly; only the k-reduction is reordered (per-lane partials +
// one warp_sum), so identity is verified empirically by kernel-verify.
// v2 (decode-bandwidth pass): same cooperative geometry — the per-lane loads
// already coalesce across the warp — but two blocks per iteration with
// independent accumulators so twice as many loads are in flight per loop trip
// (the 210-byte stride is only 2-aligned, so wider per-lane loads are not
// available). Identity vs CPU stays empirically gated by kernel-verify.
static __device__ __forceinline__ float q6k_block_dot(const uchar *blk,
                                                      const float *xb,
                                                      int lane, int is) {
    float d = f16f(blk + 208);
    float acc = 0;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const uchar *ql = blk + half * 64;
        const uchar *qh = blk + 128 + half * 32;
        const signed char *sc = (const signed char *)(blk + 192) + half * 8;
        int q1 = (int)((ql[lane]      & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
        int q2 = (int)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
        int q3 = (int)((ql[lane]      >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
        int q4 = (int)((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
        const float *xp = xb + half * 128;
        acc += d * ((float)(sc[0 + is] * q1) * xp[lane] +
                    (float)(sc[2 + is] * q2) * xp[lane + 32] +
                    (float)(sc[4 + is] * q3) * xp[lane + 64] +
                    (float)(sc[6 + is] * q4) * xp[lane + 96]);
    }
    return acc;
}

extern "C" __global__ void k_gemv_q6_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 210;
    int is = (int)(lane >> 4);          // (lane/16)&1 for lane 0..31 -> 0 or 1
    float s0 = 0, s1 = 0;
    int b2 = nb & ~1;
    for (int b = 0; b < b2; b += 2) {
        s0 += q6k_block_dot(rw + (ulong64)b * 210,       x + (ulong64)b * 256,
                            (int)lane, is);
        s1 += q6k_block_dot(rw + (ulong64)(b + 1) * 210, x + (ulong64)(b + 1) * 256,
                            (int)lane, is);
    }
    if (b2 < nb)
        s0 += q6k_block_dot(rw + (ulong64)b2 * 210, x + (ulong64)b2 * 256,
                            (int)lane, is);
    float s = s0 + s1;
    MV_TAIL;
}

// Q6_K prefill GEMM: same shared-memory x staging as k_gemm_q4_K, with the
// lane-per-element geometry of k_gemv_q6_K. Warp walks blocks sequentially;
// its 32 lanes cooperatively reduce the 256 elements of each block (8/lane,
// positions {l,l+32,l+64,l+96} per half). x for the current block is staged in
// smem so decoded weights FMA against smem. Reordered k-reduction vs
// k_mv_q6_K_b -> token identity verified empirically.
extern "C" __global__ void k_gemm_q6_K(MV_PARAMS) {
    __shared__ float xsm[MVT][256];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off +
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 210;
    int is = (int)(lane >> 4);
    float s[MVT] = {0};

    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 210;
        int base_e = b * 256;
        #pragma unroll
        for (int t = 0; t < MVT; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < 256; e += blockDim.x) xsm[t][e] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out) {
            float d = f16f(blk + 208);
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const uchar *ql = blk + half * 64;
                const uchar *qh = blk + 128 + half * 32;
                const signed char *sc = (const signed char *)(blk + 192) + half * 8;
                int q1 = (int)((ql[lane]      & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[lane]      >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
                float w1 = d * (float)(sc[0 + is] * q1);
                float w2 = d * (float)(sc[2 + is] * q2);
                float w3 = d * (float)(sc[4 + is] * q3);
                float w4 = d * (float)(sc[6 + is] * q4);
                int e0 = half * 128;
                #pragma unroll
                for (int t = 0; t < MVT; t++) {
                    const float *xs = xsm[t];
                    s[t] += w1 * xs[e0 + lane]      + w2 * xs[e0 + lane + 32] +
                            w3 * xs[e0 + lane + 64] + w4 * xs[e0 + lane + 96];
                }
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        for (int t = 0; t < a.batch; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}

// Q3_K prefill GEMM: same shared-memory x staging as k_gemm_q4_K, with 8
// CONTIGUOUS elements per lane. That split keeps every per-element input
// constant across the lane's strip: elements el..el+7 share one 16-element
// scale group (el is a multiple of 8), one 2-bit shift (el's 32-quarter),
// one hmask bit and one qs quarter — so the inner loop is one scale mul and
// eight mask/shift ops. Reordered k-reduction vs k_mv_q3_K_b -> token
// identity verified empirically, the k_gemm_q6_K precedent. This kernel is
// what retires the measured Q3_K prefill pathology (6.7-15.5x slower than
// Q4_K on the full card): batch>1 previously fell back to per-token matvec
// because f_gemm[T_Q3_K] did not exist.
extern "C" __global__ void k_gemm_q3_K(MV_PARAMS) {
    __shared__ float xsm[MVT][256];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off +
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 110;
    float s[MVT] = {0};
    int el    = (int)lane * 8;          // this lane's 8 contiguous elements
    int half  = el >> 7;                // the mv loop's n = 0 / n = 128 halves
    int shift = ((el & 127) >> 5) * 2;  // 2-bit group within the half
    int idx   = el & 31;                // byte index within the quarter
    uchar mbit = (uchar)(1 << (half * 4 + (shift >> 1)));
    int sidx  = el >> 4;                // 16-element scale group

    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 110;
        int base_e = b * 256;
        #pragma unroll
        for (int t = 0; t < MVT; t++) {
            const float *xg = x + (ulong64)t * a.xs + base_e;
            for (int e = threadIdx.x; e < 256; e += blockDim.x) xsm[t][e] = xg[e];
        }
        __syncthreads();
        if (row < (unsigned)a.n_out) {
            Q3K_UNPACK_SCALES;
            float dl = d_all * (q3scales[sidx] - 32);
            const uchar *q = qbase + half * 32;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int i = idx + k;
                float w = dl * (float)(((q[i] >> shift) & 3) -
                                       ((hm[i] & mbit) ? 0 : 4));
                #pragma unroll
                for (int t = 0; t < MVT; t++) s[t] += w * xsm[t][el + k];
            }
        }
        __syncthreads();
    }
    if (row < (unsigned)a.n_out) {
        for (int t = 0; t < a.batch; t++) {
            float r = warp_sum(s[t]);
            if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
        }
    }
}


// ---------------------------------------------------------------------------
// Transposed matvec for training (adaptation D8): dx[t][i] += sum_j W[j][i] *
// dy[t][j], computed as the EXACT fmaf chain the CPU trainer runs — the
// accumulator STARTS from dx's incoming value, j advances serially per output
// element, and a zero dy[j] is skipped before the fmaf exactly as the CPU
// path skips it. One thread per output element i and one grid row per
// position t: outputs are independent, the serial-j reduction order is
// fixed, so the result is deterministic and
// byte-comparable against the CPU chain. Adjacent threads read adjacent
// elements of each weight row (coalesced) and share the row's block headers
// through L2. Reuses mv_args: x = dy [batch][xs], y = dx in-out [batch][ys].

#define MVT_HEAD \
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); \
    if (i >= a.n_in) return; \
    (void)bias;

// Positions are the grid's second dimension (D8 slice 4). Every dx[t][i]
// starts from its own incoming value and advances j serially, so the t
// iterations touch disjoint outputs and depend on nothing the other t's
// write: spreading them over blocks cannot move a bit, and the strided
// form stays correct for ANY gridDim.y, including the 1 the pre-slice-4
// host used. What it buys is occupancy — a projection with n_in 1536 is
// six 256-thread blocks, six of the 3070's 46 SMs, until t widens it.
#define MVT_T_LOOP \
    for (int t = (int)blockIdx.y; t < a.batch; t += (int)gridDim.y)

// ---------------------------------------------------------------------------
// Canonical-order forward matvec (R8.7.1 slice 1): y[row] = dot(W[row], x)
// computed in EXACTLY the association `vec_dot` uses under
// RUNNER_CANON_KERNELS, so the device result is bit-identical to the host's
// rather than merely close.
//
// Why this is not the usual determinism trade. The published cost of
// deterministic GPU inference is batch-invariant kernels, which give up
// tiling and split reductions and lose most of the throughput. This gives up
// neither: the canonical order is ALREADY an eight-lane tree, so the shape
// the CPU must use for cross-ISA bit-exactness is a shape a warp runs
// natively. Eight lanes hold the eight accumulators and three shuffles are
// canon_tree8:
//
//     c[l] = acc[l] + acc[l+4]      shfl_down 4   (l < 4)
//     d0   = c0 + c2, d1 = c1 + c3  shfl_down 2
//     out  = d0 + d1                shfl_down 1
//
// which is that function's association written out, not an approximation of
// it. Lane l reads element l of each eight-wide group, so the eight lanes of
// a row read 32 contiguous bytes: coalesced, and every weight byte is read
// once.
//
// The tail (n not a multiple of the group) is summed by lane 0 alone, in
// ascending order with `+` rather than fma, because that is what the scalar
// tail does after the tree.

#define CANON_LANES 8

// the three-step reduction, in canon_tree8's exact association
static __device__ __forceinline__ float canon_tree8_warp(float v) {
    float c = v + __shfl_down_sync(0xffffffffu, v, 4, CANON_LANES);
    float d = c + __shfl_down_sync(0xffffffffu, c, 2, CANON_LANES);
    return d + __shfl_down_sync(0xffffffffu, d, 1, CANON_LANES);
}

#define CANON_HEAD(ROWBYTES) \
    int lane = (int)(threadIdx.x & (CANON_LANES - 1)); \
    int row  = (int)((blockIdx.x * blockDim.x + threadIdx.x) / CANON_LANES); \
    if (row >= n_out) return; \
    const uchar *rw = wb + w_off + (ulong64)row * (ulong64)(ROWBYTES);

// canon_dot_q8_0: per 32-element block, lane l takes elements l, 8+l, 16+l,
// 24+l -- a multiply then three fmaf -- and the block scale folds in with a
// fourth fmaf into the lane accumulator.
extern "C" __global__ void k_mvcanon_q8_0(const uchar *wb, const float *x,
                                          float *y, int n_in, int n_out,
                                          ulong64 w_off, int row_bytes) {
    CANON_HEAD(row_bytes);
    int nb = n_in / 32;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        const float *xp = x + b * 32;
        float t = (float)q[lane] * xp[lane];
        t = fmaf((float)q[8 + lane],  xp[8 + lane],  t);
        t = fmaf((float)q[16 + lane], xp[16 + lane], t);
        t = fmaf((float)q[24 + lane], xp[24 + lane], t);
        acc = fmaf(d, t, acc);
    }
    float s = canon_tree8_warp(acc);
    if (lane == 0) y[row] = s;
}

// canon_dot_f32: four accumulators over a 32-element stride, combined as
// (a+b) + (c+d) per lane before the tree, then a scalar tail.
extern "C" __global__ void k_mvcanon_f32(const uchar *wb, const float *x,
                                         float *y, int n_in, int n_out,
                                         ulong64 w_off, int row_bytes) {
    CANON_HEAD(row_bytes);
    const float *w = (const float *)rw;
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    int i = 0;
    for (; i + 32 <= n_in; i += 32) {
        a = fmaf(w[i + lane],      x[i + lane],      a);
        b = fmaf(w[i + 8 + lane],  x[i + 8 + lane],  b);
        c = fmaf(w[i + 16 + lane], x[i + 16 + lane], c);
        d = fmaf(w[i + 24 + lane], x[i + 24 + lane], d);
    }
    float ab = a + b, cd = c + d;
    float s = canon_tree8_warp(ab + cd);
    if (lane == 0) {
        for (int j = i; j < n_in; j++) s += w[j] * x[j];
        y[row] = s;
    }
}

// canon_dot_f16: two accumulators over a 16-element stride, a+b per lane
// before the tree, then a scalar tail.
extern "C" __global__ void k_mvcanon_f16(const uchar *wb, const float *x,
                                         float *y, int n_in, int n_out,
                                         ulong64 w_off, int row_bytes) {
    CANON_HEAD(row_bytes);
    float a = 0.0f, b = 0.0f;
    int i = 0;
    for (; i + 16 <= n_in; i += 16) {
        a = fmaf(f16f(rw + (ulong64)(i + lane) * 2),     x[i + lane],     a);
        b = fmaf(f16f(rw + (ulong64)(i + 8 + lane) * 2), x[i + 8 + lane], b);
    }
    float s = canon_tree8_warp(a + b);
    if (lane == 0) {
        for (int j = i; j < n_in; j++) s += f16f(rw + (ulong64)j * 2) * x[j];
        y[row] = s;
    }
}

// ---------------------------------------------------------------------------
// LoRA at inference on the device (adaptation D2 on CUDA, R8.7.2):
// y[b][j] += scale * sum_k B[j][k] * (sum_i A[k][i] * x[b][i]).
//
// Two launches rather than one fused kernel: the inner projection is r
// reductions over n_in and the outer is n_out independent dot products over
// r, so their natural grids have nothing in common. A and B are F32 (the
// adapter loader converts F16/BF16 at load), and the rank is bounded by
// LORA_R_MAX, which is what lets the second kernel stage the whole inner
// vector in static shared memory.
//
// This is not the CPU hook's arithmetic: the reduction below is a warp tree
// where the CPU walks a serial fmaf chain. That difference is the ordinary
// CPU/GPU one and is bounded by the same merged-reference gate the CPU hook
// answers to. What IS exact either way: a zero B, or a zero scale,
// contributes fmaf(scale, 0, y) == y, so an unadapted answer stays
// bit-for-bit unadapted.

#define LORA_R_MAX_DEV 512   // keep in sync with LORA_R_MAX in model.c

// t[b][k] = sum_i A[k][i] * x[b][i]; one 128-thread block per (k, position)
extern "C" __global__ void k_lora_a(const float *A, const float *x, float *t,
                                    int n_in, int r, int xs) {
    int k = (int)blockIdx.x, b = (int)blockIdx.y;
    const float *ar = A + (ulong64)k * n_in;
    const float *xr = x + (ulong64)b * xs;
    float s = 0.0f;
    for (int i = (int)threadIdx.x; i < n_in; i += (int)blockDim.x)
        s = fmaf(ar[i], xr[i], s);
    __shared__ float part[4];
    s = warp_sum(s);
    unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) part[w] = s;
    __syncthreads();
    if (threadIdx.x == 0)
        t[(ulong64)b * r + k] = (part[0] + part[1]) + (part[2] + part[3]);
}

// y[b][j] += scale * sum_k B[j][k] * t[b][k]; one thread per output element
extern "C" __global__ void k_lora_b(const float *B, const float *t, float *y,
                                    int n_out, int r, int ys, float scale) {
    __shared__ float ts[LORA_R_MAX_DEV];
    int b = (int)blockIdx.y;
    for (int k = (int)threadIdx.x; k < r; k += (int)blockDim.x)
        ts[k] = t[(ulong64)b * r + k];
    __syncthreads();
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n_out) return;
    const float *br = B + (ulong64)j * r;
    float acc = 0.0f;
    for (int k = 0; k < r; k++) acc = fmaf(br[k], ts[k], acc);
    y[(ulong64)b * ys + j] = fmaf(scale, acc, y[(ulong64)b * ys + j]);
}

extern "C" __global__ void k_mvt_f32(MV_PARAMS) {
    MVT_HEAD;
    MVT_T_LOOP {
        float acc = y[(ulong64)t * a.ys + i];
        for (int j = 0; j < a.n_out; j++) {
            float v = x[(ulong64)t * a.xs + j];
            if (v == 0.0f) continue;
            const float *row = (const float *)(wb + a.w_off +
                                               (ulong64)j * a.n_in * 4);
            acc = fmaf(row[i], v, acc);
        }
        y[(ulong64)t * a.ys + i] = acc;
    }
}

extern "C" __global__ void k_mvt_f16(MV_PARAMS) {
    MVT_HEAD;
    MVT_T_LOOP {
        float acc = y[(ulong64)t * a.ys + i];
        for (int j = 0; j < a.n_out; j++) {
            float v = x[(ulong64)t * a.xs + j];
            if (v == 0.0f) continue;
            const uchar *row = wb + a.w_off + (ulong64)j * a.n_in * 2;
            acc = fmaf(f16f(row + (ulong64)i * 2), v, acc);
        }
        y[(ulong64)t * a.ys + i] = acc;
    }
}

extern "C" __global__ void k_mvt_bf16(MV_PARAMS) {
    MVT_HEAD;
    MVT_T_LOOP {
        float acc = y[(ulong64)t * a.ys + i];
        for (int j = 0; j < a.n_out; j++) {
            float v = x[(ulong64)t * a.xs + j];
            if (v == 0.0f) continue;
            const ushort16 *row = (const ushort16 *)(wb + a.w_off +
                                                     (ulong64)j * a.n_in * 2);
            uint u = ((uint)row[i]) << 16;
            acc = fmaf(__uint_as_float(u), v, acc);
        }
        y[(ulong64)t * a.ys + i] = acc;
    }
}

extern "C" __global__ void k_mvt_q8_0(MV_PARAMS) {
    MVT_HEAD;
    int nb = a.n_in / 32;
    int bi = i >> 5, el = i & 31;
    MVT_T_LOOP {
        float acc = y[(ulong64)t * a.ys + i];
        for (int j = 0; j < a.n_out; j++) {
            float v = x[(ulong64)t * a.xs + j];
            if (v == 0.0f) continue;
            const uchar *blk = wb + a.w_off + (ulong64)j * nb * 34 +
                               (ulong64)bi * 34;
            float d = f16f(blk);
            signed char q = ((const signed char *)(blk + 2))[el];
            acc = fmaf(d * (float)q, v, acc);
        }
        y[(ulong64)t * a.ys + i] = acc;
    }
}

extern "C" __global__ void k_mvt_q4_0(MV_PARAMS) {
    MVT_HEAD;
    int nb = a.n_in / 32;
    int bi = i >> 5, el = i & 31;
    MVT_T_LOOP {
        float acc = y[(ulong64)t * a.ys + i];
        for (int j = 0; j < a.n_out; j++) {
            float v = x[(ulong64)t * a.xs + j];
            if (v == 0.0f) continue;
            const uchar *blk = wb + a.w_off + (ulong64)j * nb * 18 +
                               (ulong64)bi * 18;
            float d = f16f(blk);
            const uchar *q = blk + 2;
            int nib = el < 16 ? (q[el] & 0xF) : (q[el - 16] >> 4);
            acc = fmaf(d * (float)(nib - 8), v, acc);
        }
        y[(ulong64)t * a.ys + i] = acc;
    }
}

extern "C" __global__ void k_mvt_q6_K(MV_PARAMS) {
    MVT_HEAD;
    int nb = a.n_in / 256;
    int bi = i >> 8, e = i & 255;
    int half = e >> 7, r = e & 127, seg = r >> 5, l = r & 31, is = l >> 4;
    MVT_T_LOOP {
        float acc = y[(ulong64)t * a.ys + i];
        for (int j = 0; j < a.n_out; j++) {
            float v = x[(ulong64)t * a.xs + j];
            if (v == 0.0f) continue;
            const uchar *blk = wb + a.w_off + (ulong64)j * nb * 210 +
                               (ulong64)bi * 210;
            const uchar *ql = blk + half * 64;
            const uchar *qh = blk + 128 + half * 32;
            const signed char *sc = (const signed char *)(blk + 192) +
                                    half * 8;
            float d = f16f(blk + 208);
            int q;
            signed char s;
            if (seg == 0) {
                q = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                s = sc[0 + is];
            } else if (seg == 1) {
                q = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                s = sc[2 + is];
            } else if (seg == 2) {
                q = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                s = sc[4 + is];
            } else {
                q = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                s = sc[6 + is];
            }
            acc = fmaf(d * (float)((int)s * q), v, acc);
        }
        y[(ulong64)t * a.ys + i] = acc;
    }
}

extern "C" __global__ void k_mvt_q4_K(MV_PARAMS) {
    MVT_HEAD;
    int nb = a.n_in / 256;
    int bi = i >> 8, e = i & 255;
    int g = e >> 5;           // 32-element scale group 0..7
    int w32 = e & 31;
    int pair = g >> 1;        // qs stores groups in nibble pairs
    int hi = g & 1;
    MVT_T_LOOP {
        float acc = y[(ulong64)t * a.ys + i];
        for (int j = 0; j < a.n_out; j++) {
            float v = x[(ulong64)t * a.xs + j];
            if (v == 0.0f) continue;
            const uchar *blk = wb + a.w_off + (ulong64)j * nb * 144 +
                               (ulong64)bi * 144;
            float dd   = f16f(blk);
            float dmin = f16f(blk + 2);
            uchar sg, mg;
            get_scale_min_k4(g, blk + 4, &sg, &mg);
            uchar byte = blk[16 + pair * 32 + w32];
            int nib = hi ? (byte >> 4) : (byte & 0xF);
            float wv = dd * (float)sg * (float)nib - dmin * (float)mg;
            acc = fmaf(wv, v, acc);
        }
        y[(ulong64)t * a.ys + i] = acc;
    }
}

// IQ4: the nibble indexes a fixed 16-entry codebook
static __device__ const signed char kv_iq4[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

extern "C" __global__ void k_mv_iq4_nl(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)kv_iq4[q[j] & 0xF] * xp[j];
            t += (float)kv_iq4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq4_nl_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * (float)kv_iq4[q[j] & 0xF], base + j);
            MV_FMA(d * (float)kv_iq4[q[j] >> 4],  base + j + 16);
        }
    }
    MV_TAIL_B;
}

extern "C" __global__ void k_mv_iq4_xs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 136;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 136;
        float d = f16f(blk);
        unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
        const uchar *sl = blk + 4;
        const uchar *q  = blk + 8;
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
            float dl = d * (ls - 32);
            float t = 0;
            for (int j = 0; j < 16; j++) {
                t += (float)kv_iq4[q[j] & 0xF] * xp[j];
                t += (float)kv_iq4[q[j] >> 4]  * xp[j + 16];
            }
            s += dl * t;
            q += 16; xp += 32;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq4_xs_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 136;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 136;
        float d = f16f(blk);
        unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
        const uchar *sl = blk + 4;
        const uchar *q  = blk + 8;
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; j++) {
                MV_FMA(dl * (float)kv_iq4[q[j] & 0xF], base + ib * 32 + j);
                MV_FMA(dl * (float)kv_iq4[q[j] >> 4],  base + ib * 32 + j + 16);
            }
            q += 16;
        }
    }
    MV_TAIL_B;
}

// ------------------------------------------------------ codebook i-quants
// IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S: every 8 (IQ1/IQ2) or 4
// (IQ3) weights are one index into a fixed grid of magnitude patterns, one
// byte per weight, flipped by a sign bit per weight and scaled per sub-block
// of 32. The grids are the formats' codebooks, copied verbatim from
// quants_iq_grids.h (tests/test_cuda_iq_grids.py holds the two copies
// equal); each kernel walks its block exactly as the matching dq_* in
// quants.c does, one lane per block, in the shape of the k_mv_* kernels
// above. Block bytes are 2-aligned (even block sizes, row = whole blocks),
// so 32-bit fields are read as two 16-bit halves.
static __device__ const ulong64 kiq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static __device__ const ulong64 kiq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

static __device__ const ulong64 kiq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

static __device__ const unsigned kiq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static __device__ const unsigned kiq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

static __device__ const ulong64 kiq1s_grid[2048] = {
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
};


static __device__ __forceinline__ unsigned ld16(const uchar *p) {
    return *(const unsigned short *)p;
}
static __device__ __forceinline__ unsigned ld32a2(const uchar *p) {
    return ld16(p) | (ld16(p + 2) << 16);
}
// ksigns_iq2xs[i] in quants_iq_grids.h is i with its odd parity in bit 7
// (tests/test_cuda_iq_grids.py checks all 128 entries), so a 7-bit sign
// index expands arithmetically instead of through a divergent table read.
static __device__ __forceinline__ unsigned iq_signs7(unsigned idx) {
    return idx | ((__popc(idx) & 1u) << 7);
}
// weight j of an 8-magnitude grid entry (byte j), negated by sign bit j
static __device__ __forceinline__ float iq_w8(ulong64 grid, int j, unsigned signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return (signs >> j) & 1 ? -mag : mag;
}
// weight j of a 4-magnitude grid entry (byte j), negated by sign bit j
static __device__ __forceinline__ float iq_w4(unsigned grid, int j, unsigned signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return (signs >> j) & 1 ? -mag : mag;
}
// IQ1 grid bytes are signed (-1/0/1) and carry a per-index delta
static __device__ __forceinline__ float iq1_w(ulong64 grid, int j, float delta) {
    return (float)(signed char)((grid >> (8 * j)) & 0xFF) + delta;
}
#define IQ1_DELTA 0.125f

// IQ2_XXS: 66-byte block, d (fp16) + 8 sub-blocks of 8 bytes: four grid
// indices, then one word of four 7-bit sign indices (bits 0-27) and a 4-bit
// scale (bits 28-31) decoded as (0.5 + scale) / 4.
extern "C" __global__ void k_mv_iq2_xxs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 66;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 66;
        float d = f16f(blk);
        const uchar *qs = blk + 2;
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 8) {
            unsigned aux = ld32a2(qs + 4);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
            float t = 0;
            for (int l = 0; l < 4; l++, xp += 8) {
                ulong64 g = kiq2xxs_grid[qs[l]];
                unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
                for (int j = 0; j < 8; j++) t += iq_w8(g, j, signs) * xp[j];
            }
            s += db * t;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq2_xxs_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 66;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 66;
        float d = f16f(blk);
        const uchar *qs = blk + 2;
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 8) {
            unsigned aux = ld32a2(qs + 4);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++, base += 8) {
                ulong64 g = kiq2xxs_grid[qs[l]];
                unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
                for (int j = 0; j < 8; j++) MV_FMA(db * iq_w8(g, j, signs), base + j);
            }
        }
    }
    MV_TAIL_B;
}

// IQ2_XS: 74-byte block, d (fp16) + 32 16-bit words (9-bit grid index, 7-bit
// sign index) + 8 scale bytes (two 4-bit scales per sub-block of 32, one per
// 16 weights) decoded as (0.5 + scale) / 4.
extern "C" __global__ void k_mv_iq2_xs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 74;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 74;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *sc = blk + 66;
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 8) {
            float db0 = d * (0.5f + (float)(sc[ib] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++, xp += 8) {
                unsigned q = ld16(qs + 2 * l);
                ulong64 g = kiq2xs_grid[q & 511];
                unsigned signs = iq_signs7(q >> 9);
                float t = 0;
                for (int j = 0; j < 8; j++) t += iq_w8(g, j, signs) * xp[j];
                s += (l < 2 ? db0 : db1) * t;
            }
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq2_xs_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 74;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 74;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *sc = blk + 66;
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 8) {
            float db0 = d * (0.5f + (float)(sc[ib] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++, base += 8) {
                unsigned q = ld16(qs + 2 * l);
                ulong64 g = kiq2xs_grid[q & 511];
                unsigned signs = iq_signs7(q >> 9);
                float db = l < 2 ? db0 : db1;
                for (int j = 0; j < 8; j++) MV_FMA(db * iq_w8(g, j, signs), base + j);
            }
        }
    }
    MV_TAIL_B;
}

// IQ2_S: 82-byte block, d (fp16) + 32 low index bytes + 32 sign bytes (one
// bit per weight) + 8 bytes of high index bits (two per index) + 8 scale
// bytes as in IQ2_XS.
extern "C" __global__ void k_mv_iq2_s(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 82;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 82;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *sg = blk + 34, *qh = blk + 66, *sc = blk + 74;
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 4, sg += 4) {
            float db0 = d * (0.5f + (float)(sc[ib] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
            unsigned hb = qh[ib];
            for (int l = 0; l < 4; l++, xp += 8) {
                ulong64 g = kiq2s_grid[qs[l] | ((hb << (8 - 2 * l)) & 0x300)];
                unsigned signs = sg[l];
                float t = 0;
                for (int j = 0; j < 8; j++) t += iq_w8(g, j, signs) * xp[j];
                s += (l < 2 ? db0 : db1) * t;
            }
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq2_s_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 82;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 82;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *sg = blk + 34, *qh = blk + 66, *sc = blk + 74;
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 4, sg += 4) {
            float db0 = d * (0.5f + (float)(sc[ib] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
            unsigned hb = qh[ib];
            for (int l = 0; l < 4; l++, base += 8) {
                ulong64 g = kiq2s_grid[qs[l] | ((hb << (8 - 2 * l)) & 0x300)];
                unsigned signs = sg[l];
                float db = l < 2 ? db0 : db1;
                for (int j = 0; j < 8; j++) MV_FMA(db * iq_w8(g, j, signs), base + j);
            }
        }
    }
    MV_TAIL_B;
}

// IQ3_XXS: 98-byte block, d (fp16) + 64 grid indices (four magnitudes each)
// + 8 words of four 7-bit sign indices (bits 0-27) and a 4-bit scale (bits
// 28-31) decoded as (0.5 + scale) / 2.
extern "C" __global__ void k_mv_iq3_xxs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 98;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 98;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *ss = blk + 66;
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 8) {
            unsigned aux = ld32a2(ss + 4 * ib);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            float t = 0;
            for (int l = 0; l < 4; l++, xp += 8) {
                unsigned g1 = kiq3xxs_grid[qs[2 * l + 0]];
                unsigned g2 = kiq3xxs_grid[qs[2 * l + 1]];
                unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
                for (int j = 0; j < 4; j++) {
                    t += iq_w4(g1, j, signs) * xp[j];
                    t += iq_w4(g2, j, signs >> 4) * xp[j + 4];
                }
            }
            s += db * t;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq3_xxs_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 98;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 98;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *ss = blk + 66;
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 8) {
            unsigned aux = ld32a2(ss + 4 * ib);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++, base += 8) {
                unsigned g1 = kiq3xxs_grid[qs[2 * l + 0]];
                unsigned g2 = kiq3xxs_grid[qs[2 * l + 1]];
                unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
                for (int j = 0; j < 4; j++) {
                    MV_FMA(db * iq_w4(g1, j, signs), base + j);
                    MV_FMA(db * iq_w4(g2, j, signs >> 4), base + j + 4);
                }
            }
        }
    }
    MV_TAIL_B;
}

// IQ3_S: 110-byte block, d (fp16) + 64 grid indices (low 8 bits) + 8 bytes
// of high index bits (one per index, two sub-blocks of 32 per byte) + 32
// sign bytes (one bit per weight) + 4 scale bytes (a 4-bit scale per
// sub-block of 32, decoded as 1 + 2*scale).
extern "C" __global__ void k_mv_iq3_s(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 110;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 110;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *qh = blk + 66, *sg = blk + 74, *sc = blk + 106;
        const float *xp = x + b * 256;
        for (int pair = 0; pair < 4; pair++, qh += 2) {
            for (int h = 0; h < 2; h++, qs += 8, sg += 4) {
                float db = d * (float)(1 + 2 * (h ? (sc[pair] >> 4) : (sc[pair] & 0xF)));
                unsigned hb = qh[h];
                float t = 0;
                for (int l = 0; l < 4; l++, xp += 8) {
                    unsigned g1 = kiq3s_grid[qs[2 * l + 0] | ((hb << (8 - 2 * l)) & 256)];
                    unsigned g2 = kiq3s_grid[qs[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
                    unsigned signs = sg[l];
                    for (int j = 0; j < 4; j++) {
                        t += iq_w4(g1, j, signs) * xp[j];
                        t += iq_w4(g2, j, signs >> 4) * xp[j + 4];
                    }
                }
                s += db * t;
            }
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq3_s_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 110;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 110;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *qh = blk + 66, *sg = blk + 74, *sc = blk + 106;
        ulong64 base = (ulong64)b * 256;
        for (int pair = 0; pair < 4; pair++, qh += 2) {
            for (int h = 0; h < 2; h++, qs += 8, sg += 4) {
                float db = d * (float)(1 + 2 * (h ? (sc[pair] >> 4) : (sc[pair] & 0xF)));
                unsigned hb = qh[h];
                for (int l = 0; l < 4; l++, base += 8) {
                    unsigned g1 = kiq3s_grid[qs[2 * l + 0] | ((hb << (8 - 2 * l)) & 256)];
                    unsigned g2 = kiq3s_grid[qs[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
                    unsigned signs = sg[l];
                    for (int j = 0; j < 4; j++) {
                        MV_FMA(db * iq_w4(g1, j, signs), base + j);
                        MV_FMA(db * iq_w4(g2, j, signs >> 4), base + j + 4);
                    }
                }
            }
        }
    }
    MV_TAIL_B;
}

// IQ1_S: 50-byte block, d (fp16) + 32 low index bytes + 8 16-bit words per
// sub-block of 32: three high index bits per index (bits 0-11), a 3-bit
// scale (bits 12-14) decoded as 1 + 2*scale, and the sign of the shared
// 1/8 delta (bit 15).
extern "C" __global__ void k_mv_iq1_s(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 50;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 50;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *qh = blk + 34;
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 4) {
            unsigned h = ld16(qh + 2 * ib);
            float dl = d * (float)(2 * ((h >> 12) & 7) + 1);
            float delta = h & 0x8000 ? -IQ1_DELTA : IQ1_DELTA;
            float t = 0;
            for (int l = 0; l < 4; l++, xp += 8) {
                ulong64 g = kiq1s_grid[qs[l] | (((h >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; j++) t += iq1_w(g, j, delta) * xp[j];
            }
            s += dl * t;
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq1_s_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 50;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 50;
        float d = f16f(blk);
        const uchar *qs = blk + 2, *qh = blk + 34;
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 4) {
            unsigned h = ld16(qh + 2 * ib);
            float dl = d * (float)(2 * ((h >> 12) & 7) + 1);
            float delta = h & 0x8000 ? -IQ1_DELTA : IQ1_DELTA;
            for (int l = 0; l < 4; l++, base += 8) {
                ulong64 g = kiq1s_grid[qs[l] | (((h >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; j++) MV_FMA(dl * iq1_w(g, j, delta), base + j);
            }
        }
    }
    MV_TAIL_B;
}

// IQ1_M: 56-byte block with no leading fp16: 32 low index bytes + 16 bytes
// of high bits (per byte: two 3-bit index extensions and two delta signs)
// + 4 16-bit scale words. Each word holds two 3-bit scales per sub-block of
// 32 (one per 16 weights, decoded as 1 + 2*scale) in its low 12 bits, and
// the block's fp16 scale is scattered across the four top nibbles.
extern "C" __global__ void k_mv_iq1_m(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 56;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 56;
        const uchar *qs = blk, *qh = blk + 32, *scb = blk + 48;
        unsigned sc0 = ld16(scb), sc1 = ld16(scb + 2), sc2 = ld16(scb + 4), sc3 = ld16(scb + 6);
        unsigned short sd = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
                            ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
        float d = __half2float(__ushort_as_half(sd));
        const float *xp = x + b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 4, qh += 2) {
            unsigned sw = ld16(scb + 2 * (ib >> 1)) >> (6 * (ib & 1));
            float dl0 = d * (float)(2 * (sw & 7) + 1);
            float dl1 = d * (float)(2 * ((sw >> 3) & 7) + 1);
            for (int l = 0; l < 4; l++, xp += 8) {
                unsigned hb = qh[l >> 1];
                unsigned idx = qs[l] | ((hb << ((l & 1) ? 4 : 8)) & 0x700);
                float delta = hb & ((l & 1) ? 0x80 : 0x08) ? -IQ1_DELTA : IQ1_DELTA;
                ulong64 g = kiq1s_grid[idx];
                float t = 0;
                for (int j = 0; j < 8; j++) t += iq1_w(g, j, delta) * xp[j];
                s += (l < 2 ? dl0 : dl1) * t;
            }
        }
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_iq1_m_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 56;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 56;
        const uchar *qs = blk, *qh = blk + 32, *scb = blk + 48;
        unsigned sc0 = ld16(scb), sc1 = ld16(scb + 2), sc2 = ld16(scb + 4), sc3 = ld16(scb + 6);
        unsigned short sd = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
                            ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
        float d = __half2float(__ushort_as_half(sd));
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 4, qh += 2) {
            unsigned sw = ld16(scb + 2 * (ib >> 1)) >> (6 * (ib & 1));
            float dl0 = d * (float)(2 * (sw & 7) + 1);
            float dl1 = d * (float)(2 * ((sw >> 3) & 7) + 1);
            for (int l = 0; l < 4; l++, base += 8) {
                unsigned hb = qh[l >> 1];
                unsigned idx = qs[l] | ((hb << ((l & 1) ? 4 : 8)) & 0x700);
                float delta = hb & ((l & 1) ? 0x80 : 0x08) ? -IQ1_DELTA : IQ1_DELTA;
                ulong64 g = kiq1s_grid[idx];
                float dl = l < 2 ? dl0 : dl1;
                for (int j = 0; j < 8; j++) MV_FMA(dl * iq1_w(g, j, delta), base + j);
            }
        }
    }
    MV_TAIL_B;
}

// MXFP4 (gpt-oss expert tensors): 17-byte block = one E8M0 scale byte (a
// biased power-of-two exponent, 2^(e-127), NOT an fp16) + 32 packed E2M1
// nibbles indexing a fixed signed codebook. Table and decode are 1:1 with
// dq_mxfp4 in quants.c — ldexpf keeps 2^(e-127) exact down into the
// subnormal range, where exp2f of a float could flush to zero.
static __device__ const float kv_mxfp4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

extern "C" __global__ void k_mv_mxfp4(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 17;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 17;
        float d = ldexpf(1.0f, (int)blk[0] - 127);
        const uchar *q = blk + 1;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}

extern "C" __global__ void k_mv_mxfp4_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 17;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 17;
        float d = ldexpf(1.0f, (int)blk[0] - 127);
        const uchar *q = blk + 1;
        ulong64 base = (ulong64)b * 32;
        for (int j = 0; j < 16; j++) {
            MV_FMA(d * kv_mxfp4[q[j] & 0xF], base + j);
            MV_FMA(d * kv_mxfp4[q[j] >> 4],  base + j + 16);
        }
    }
    MV_TAIL_B;
}

// NVFP4 (NVIDIA ModelOpt / llama.cpp type 40): a 36-byte super-block of four
// 16-element sub-blocks, one UE4M3 scale byte each (d[4]), then 32 packed
// E2M1 nibbles over the same signed codebook as MXFP4. Within sub-block s,
// byte j's low nibble is element j and its high nibble element j+8. Decode
// is 1:1 with ue4m3_to_fp32 / dq_nvfp4 in quants.c: the UE4M3 scale is
// unsigned, 4 exponent bits biased by 7, 3 mantissa bits, 0x7F is NaN and
// decodes to 0 like llama.cpp, and exponent 0 is subnormal (man * 2^-9).
// The dot sums each sub-block's 16 products before scaling (s += d * t),
// the same association as the CPU kernel. The per-tensor companion (the
// export's second level, gguf_tensor.scale, 1.0 when the file has none) is
// this kernel family's own trailing parameter, applied to the finished dot
// before the bias exactly as the CPU seam does (dot(w*s, x) = s*dot(w, x)).
// A parameter of these three kernels only, NOT a field of mv_args: widening
// the shared struct re-lays the registers of every kernel that takes it,
// and the committed PTX of kernels nobody touched must not move.
static __device__ __forceinline__ float ue4m3f(uchar x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 0x7;
    if (e == 0) return ldexpf((float)m, -9);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

extern "C" __global__ void k_mv_nvfp4(MV_PARAMS, float scale) {
    MV_HEAD;
    int nb = a.n_in / 64;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 36;
    float s = 0;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 36;
        for (int sub = 0; sub < 4; sub++) {
            float d = ue4m3f(blk[sub]);
            const uchar *q = blk + 4 + sub * 8;
            const float *xp = x + b * 64 + sub * 16;
            float t = 0;
            for (int j = 0; j < 8; j++) {
                t += kv_mxfp4[q[j] & 0xF] * xp[j];
                t += kv_mxfp4[q[j] >> 4]  * xp[j + 8];
            }
            s += d * t;
        }
    }
    s = warp_sum(s) * scale;
    if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
}

extern "C" __global__ void k_mv_nvfp4_b(MV_PARAMS, float scale) {
    MV_HEAD_B;
    int nb = a.n_in / 64;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 36;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 36;
        for (int sub = 0; sub < 4; sub++) {
            float d = ue4m3f(blk[sub]);
            const uchar *q = blk + 4 + sub * 8;
            ulong64 base = (ulong64)b * 64 + sub * 16;
            for (int j = 0; j < 8; j++) {
                MV_FMA(d * kv_mxfp4[q[j] & 0xF], base + j);
                MV_FMA(d * kv_mxfp4[q[j] >> 4],  base + j + 8);
            }
        }
    }
    for (int t = 0; t < a.batch; t++) {
        float r = warp_sum(s[t]) * scale;
        if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? r + bias[row] : r;
    }
}

// ---------------------------------------------------------------- rope
// grid: (ceil(half_dim/32), n_heads, batch); vs = element stride per column


extern "C" __global__ void k_rope(float *v, const float *fr, rope_args a,
                                  const int *posp, int vs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (j >= a.half_dim || h >= a.n_heads) return;
    int pos = *posp + blockIdx.z;
    float ang = pos * fr[j];
    float c = cosf(ang) * a.mscale, s = sinf(ang) * a.mscale;
    float *p = v + (ulong64)blockIdx.z * vs + h * a.head_dim;
    int i0 = a.neox ? j : 2 * j;
    int i1 = a.neox ? j + a.half_dim : i0 + 1;
    float x0 = p[i0], x1 = p[i1];
    p[i0] = x0 * c - x1 * s;
    p[i1] = x0 * s + x1 * c;
}

// ------------------------------------------------------------- kv storage
// The cache is either fp16 (2 bytes/value) or q8_0 (32 values per 34-byte
// block: one fp16 scale + 32 int8 quants), selected at load and identical in
// layout to the CPU cache so the two paths share the same host buffer. All
// cache offsets below are therefore BYTE offsets, and the pointers are byte
// pointers: a q8_0 row is only 2-byte aligned, so no wider load is legal.
//
// q8_0 quantization here is the same arithmetic as q8_quant_row() in quants.c
// (amax/127 scale, round-half-away-from-zero, RN fp16 scale), so a row
// quantized on the GPU is bit-identical to the same row quantized on the CPU.

struct q8_blk { __half d; signed char qs[32]; };   // 34 bytes, 2-byte aligned

#define KV_ROW_BYTES(kv_dim, q8) \
    ((q8) ? (ulong64)((kv_dim) / 32) * 34 : (ulong64)(kv_dim) * 2)

__device__ __forceinline__ void kv_store_row(unsigned char *cache,
                                             const float *src, int kv_dim,
                                             int q8, int i) {
    if (q8) {
        q8_blk *b = (q8_blk *)(cache + (ulong64)i * 34);
        const float *x = src + i * 32;
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        float d  = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        b->d = __float2half(d);
        for (int j = 0; j < 32; j++) b->qs[j] = (signed char)roundf(x[j] * id);
    } else {
        ((__half *)cache)[i] = __float2half(src[i]);
    }
}

// q * k for one head: paired accumulation, mirroring the fp16 path and
// vec_dot(T_Q8_0) in quants.c (per-block int sum, then scaled)
__device__ __forceinline__ float kv_dot(const unsigned char *row,
                                        const float *qh, int hd, int q8) {
    float s = 0;
    if (q8) {
        for (int b = 0; b < hd / 32; b++) {
            const q8_blk *blk = (const q8_blk *)(row + (ulong64)b * 34);
            const float *xp = qh + b * 32;
            float t = 0;
            for (int j = 0; j < 32; j += 2)
                t += xp[j] * blk->qs[j] + xp[j + 1] * blk->qs[j + 1];
            s += __half2float(blk->d) * t;
        }
    } else {
        const __half2 *k2 = (const __half2 *)row;
        for (int i = 0; i < hd / 2; i++) {
            float2 kf = __half22float2(k2[i]);
            // paired add reassociates FP vs. a sequential accumulation;
            // temp-0 gate covered it on tested models
            s += qh[2 * i] * kf.x + qh[2 * i + 1] * kf.y;
        }
    }
    return s;
}

// the value pair at element offset 2*i2 of one head's row. Element pairs never
// straddle a q8 block (32 is even), so one block lookup serves both.
__device__ __forceinline__ float2 kv_pair(const unsigned char *row,
                                          int i2, int q8) {
    if (q8) {
        const q8_blk *blk = (const q8_blk *)(row + (ulong64)(i2 / 16) * 34);
        float d = __half2float(blk->d);
        int j = (2 * i2) & 31;
        return make_float2(d * blk->qs[j], d * blk->qs[j + 1]);
    }
    return __half22float2(((const __half2 *)row)[i2]);
}

// Absolute position -> cache row. A ring layer owns `ring` rows and recycles
// them, so position t lives at t % ring; ring == 0 is the flat layout where the
// row IS the position. Every KV address in the attention kernels goes through
// this: the CPU path proved bit-identical with the same mapping, and a kernel
// that skipped it read a row holding a different token (nan, RTX 3070, partial
// split, 2026-08-30).
__device__ __forceinline__ ulong64 kv_slot(int t, int ring) {
    return (ulong64)(ring > 0 ? t % ring : t);
}

// grid.y = token column; cache rows for consecutive positions are contiguous

extern "C" __global__ void k_store_kv(const float *k, const float *v,
                                      unsigned char *kc, unsigned char *vc,
                                      int kv_dim, ulong64 l_off,
                                      const int *posp, int q8, int ring) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = q8 ? kv_dim / 32 : kv_dim;
    if (i < n) {
        ulong64 row_b = KV_ROW_BYTES(kv_dim, q8);
        ulong64 dst = l_off + kv_slot(*posp + blockIdx.y, ring) * row_b;
        const float *ks = k + (ulong64)blockIdx.y * kv_dim;
        const float *vs = v + (ulong64)blockIdx.y * kv_dim;
        kv_store_row(kc + dst, ks, kv_dim, q8, i);
        kv_store_row(vc + dst, vs, kv_dim, q8, i);
    }
}

// ---------------------------------------------------------------- attention
// One block per (head, token): scores -> softmax -> weighted value sum.
// att scratch is MVB planes of [n_head][n_ctx].


// byte offset of head kvh's slice within a cache row
__device__ __forceinline__ ulong64 kv_head_off(int kvh, int hd, int q8) {
    return q8 ? (ulong64)(kvh * hd / 32) * 34 : (ulong64)(kvh * hd) * 2;
}

// sinks: gpt-oss per-head learned attention-sink logits for this layer, or
// NULL. Transcribed from softmax_sink() in model.c: the sink joins the max
// scan and the denominator but has NO value row, so the probabilities over
// real positions sum to < 1 and the head's output shrinks. The sink competes
// against ALREADY-SCALED scores; it is not itself scaled.
// Position-chunk cap for the V accumulation. nchunk*lanes never exceeds the
// block, so the partial buffer is bounded at tpg*2 floats regardless.
#define ATTN_VMAX 8

extern "C" __global__ void k_attn(const float *q, const unsigned char *kc,
                                  const unsigned char *vc, float *att, float *out,
                                  attn_args a, const int *posp,
                                  const float *sinks) {
    __shared__ float red[256];
    __shared__ float vpart[256 * 2];   // nchunk*lanes <= tpg <= 256
    int h = blockIdx.x, tid = threadIdx.x, tpg = blockDim.x;
    int tk = blockIdx.y;                 // token column in the tile
    int pos = *posp + tk;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8);
    ulong64 base  = a.l_off + kv_head_off(kvh, hd, a.q8);
    int t0 = 0;                          // sliding-window start
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    const float *qh = q + (ulong64)tk * a.qs + h * hd;
    float *ah = att + ((ulong64)tk * a.n_head + h) * a.n_ctx;

    for (int t = t0 + tid; t <= pos; t += tpg)
        ah[t] = kv_dot(kc + base + kv_slot(t, a.ring) * row_b, qh, hd, a.q8) * a.scale;
    __syncthreads();

    // max
    float mx = -1e30f;
    for (int t = t0 + tid; t <= pos; t += tpg) mx = fmaxf(mx, ah[t]);
    red[tid] = mx;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    mx = red[0];
    if (sinks && sinks[h] > mx) mx = sinks[h];
    __syncthreads();
    // exp + sum
    float sum = 0;
    for (int t = t0 + tid; t <= pos; t += tpg) {
        float e = expf(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    sum = red[0];
    if (sinks) sum += expf(sinks[h] - mx);
    __syncthreads();

    // V accumulation, spread over the whole block. `lanes` float2 dims x
    // `nchunk` disjoint position strides; each thread owns one (lane, chunk)
    // pair, and the partials are summed per lane below.
    const int lanes  = hd / 2;
    int nchunk = lanes > 0 ? tpg / lanes : 1;
    if (nchunk < 1) nchunk = 1;
    if (nchunk > ATTN_VMAX) nchunk = ATTN_VMAX;
    const int vlane  = lanes > 0 ? tid % lanes : 0;
    const int vchunk = lanes > 0 ? tid / lanes : 0;

    // Parallel path only when every lane is covered by the block. With
    // lanes > tpg (head_dim > 512, which gemma4 can reach) the chunking would
    // leave the upper dims unwritten, so that case keeps the original loop.
    if (lanes > 0 && lanes <= tpg) {
    if (vchunk < nchunk && vlane < lanes) {
        float o0 = 0, o1 = 0;
        for (int t = t0 + vchunk; t <= pos; t += nchunk) {
            float2 vf = kv_pair(vc + base + kv_slot(t, a.ring) * row_b, vlane, a.q8);
            o0 += ah[t] * vf.x;
            o1 += ah[t] * vf.y;
        }
        vpart[(vchunk * lanes + vlane) * 2 + 0] = o0;
        vpart[(vchunk * lanes + vlane) * 2 + 1] = o1;
    }
    __syncthreads();
    for (int i2 = tid; i2 < lanes; i2 += tpg) {
        float s0 = 0, s1 = 0;
        for (int c = 0; c < nchunk; c++) {
            s0 += vpart[(c * lanes + i2) * 2 + 0];
            s1 += vpart[(c * lanes + i2) * 2 + 1];
        }
        out[(ulong64)tk * a.os + h * hd + 2 * i2]     = s0 / sum;
        out[(ulong64)tk * a.os + h * hd + 2 * i2 + 1] = s1 / sum;
    }
    } else {
        for (int i2 = tid; i2 < lanes; i2 += tpg) {
            float o0 = 0, o1 = 0;
            for (int t = t0; t <= pos; t++) {
                float2 vf = kv_pair(vc + base + kv_slot(t, a.ring) * row_b, i2, a.q8);
                o0 += ah[t] * vf.x;
                o1 += ah[t] * vf.y;
            }
            out[(ulong64)tk * a.os + h * hd + 2 * i2]     = o0 / sum;
            out[(ulong64)tk * a.os + h * hd + 2 * i2 + 1] = o1 / sum;
        }
    }
}

// --------------------------------------------------- flash-decoding attention
// Decode (batch==1, one query token, long KV) attention. The plain k_attn runs
// one block per (head, token): a 4B decode step is 32 blocks on a 46-SM GPU,
// each re-reading the whole fp16 KV cache serially. These two kernels split the
// KV range across ATTN_SPLITS blocks per head (fixed, compile-time constant, so
// the CUDA graph stays valid across positions) and merge the partials:
//
//   k_attn_dec  : grid (n_head, ATTN_SPLITS, tn). Each (head, split) block
//                 computes softmax over its on-device-computed KV slice with
//                 the SAME within-slice reduction structure as k_attn (paired
//                 q*k, strided-then-tree max/sum, sequential weighted-V), and
//                 writes an un-normalised partial: weighted-V + local max +
//                 local sum. Empty slices write a -inf-max sentinel.
//   k_attn_merge: grid (n_head, tn). Combines the ATTN_SPLITS partials with a
//                 global max and the standard exp(m_j - M) rescale, divides by
//                 the merged sum, writes out. Merge order is fixed (0..SPLITS)
//                 so it is deterministic across positions.
//
// The cross-slice merge reassociates the softmax sum (extra exp(m_j - M) and a
// regrouped add) relative to k_attn's single global reduction, so identity is
// not bitwise and is verified empirically by kernel-verify. Within a slice the
// order is preserved. Partials scratch layout per (tk, head, split):
//   [0..hd)  un-normalised weighted V ; [hd] local max ; [hd+1] local sum.

#define ATTN_SPLITS 8

// LAUNCH INVARIANT: blockDim.x must be a power of two and <= 128 (red[]'s
// size) for the tree reduction and shared indexing here. Host launches with 128.
extern "C" __global__ void k_attn_dec(const float *q, const unsigned char *kc,
                                      const unsigned char *vc, float *att, float *part,
                                      attn_args a, const int *posp) {
    __shared__ float red[128];
    int h = blockIdx.x, sp = blockIdx.y, tk = blockIdx.z;
    int tid = threadIdx.x, tpg = blockDim.x;
    int pos = *posp + tk;
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8);
    ulong64 base  = a.l_off + kv_head_off(kvh, hd, a.q8);
    int t0 = 0;
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    int total = pos + 1 - t0;
    int slice = (total + ATTN_SPLITS - 1) / ATTN_SPLITS;     // on-device from pos
    int s0 = t0 + sp * slice;
    int s1 = s0 + slice;
    if (s1 > pos + 1) s1 = pos + 1;
    float *P = part + (((ulong64)tk * a.n_head + h) * ATTN_SPLITS + sp) * (hd + 2);
    if (s0 >= s1) {                       // empty slice: sentinel, skipped in merge
        if (tid == 0) { P[hd] = -1e30f; P[hd + 1] = 0.f; }
        return;
    }
    const float *qh = q + (ulong64)tk * a.qs + h * hd;
    float *ah = att + ((ulong64)tk * a.n_head + h) * a.n_ctx;

    for (int t = s0 + tid; t < s1; t += tpg)
        ah[t] = kv_dot(kc + base + kv_slot(t, a.ring) * row_b, qh, hd, a.q8) * a.scale;
    __syncthreads();
    float mx = -1e30f;
    for (int t = s0 + tid; t < s1; t += tpg) mx = fmaxf(mx, ah[t]);
    red[tid] = mx;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0;
    for (int t = s0 + tid; t < s1; t += tpg) {
        float e = expf(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    sum = red[0];
    __syncthreads();
    for (int i2 = tid; i2 < hd / 2; i2 += tpg) {
        float o0 = 0, o1 = 0;
        for (int t = s0; t < s1; t++) {
            float2 vf = kv_pair(vc + base + kv_slot(t, a.ring) * row_b, i2, a.q8);
            o0 += ah[t] * vf.x;
            o1 += ah[t] * vf.y;
        }
        P[2 * i2]     = o0;               // un-normalised (merge divides by sum)
        P[2 * i2 + 1] = o1;
    }
    if (tid == 0) { P[hd] = mx; P[hd + 1] = sum; }
}

// sinks: same contract as k_attn. The sink joins at the GLOBAL reduction —
// it competes in the merged max and adds one exp term to the merged
// denominator, and since it has no value row the numerator is untouched, so
// the split partials (k_attn_dec / k_attn_dec_seq) need no sink awareness.
extern "C" __global__ void k_attn_merge(float *out, const float *part,
                                        attn_args a, const int *posp,
                                        const float *sinks) {
    int h = blockIdx.x, tk = blockIdx.y;
    int tid = threadIdx.x, tpg = blockDim.x;
    int hd = a.head_dim;
    const float *base = part + ((ulong64)tk * a.n_head + h) * ATTN_SPLITS * (hd + 2);
    float M = -1e30f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        M = fmaxf(M, base[sp * (hd + 2) + hd]);
    if (sinks && sinks[h] > M) M = sinks[h];
    float L = 0.f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        float m = base[sp * (hd + 2) + hd];
        if (m <= -1e29f) continue;
        L += base[sp * (hd + 2) + hd + 1] * expf(m - M);
    }
    if (sinks) L += expf(sinks[h] - M);
    for (int i = tid; i < hd; i += tpg) {
        float acc = 0.f;
        for (int sp = 0; sp < ATTN_SPLITS; sp++) {
            const float *P = base + sp * (hd + 2);
            float m = P[hd];
            if (m <= -1e29f) continue;
            acc += P[i] * expf(m - M);
        }
        out[(ulong64)tk * a.os + h * hd + i] = acc / L;
    }
}

// ============================================================================
// Batched decode: one token for each of N *independent* sequences (Phase 6)
// ============================================================================
//
// The prefill tile kernels above batch N tokens of ONE sequence: consecutive
// positions, one KV cache, so a single base position and a single cache
// pointer describe the whole tile. Continuous batching needs the other shape
// — N tokens of N DIFFERENT sequences, each at its own position, each writing
// and reading its own KV region — and that is the only thing the kernels below
// change. Every column is still computed exactly as a lone token would be.
//
// Two mechanical differences from the tile kernels, and nothing else:
//
//   posp is an ARRAY indexed by the token column, not a base + column offset.
//   kcp/vcp are ARRAYS of device pointers, one KV cache per sequence, so no
//   sequence's rows are reachable from another's column.
//
// The numerical contract is the point. `k_gemv_*_b` below decode each weight
// once and FMA it into MODEL_BATCH_MAX accumulators, in the same lane mapping
// and the same warp-reduction tree as the batch-1 `k_gemv_*` they twin — so
// column t's result is BITWISE what k_gemv_* computes for that column alone.
// That is why cuda.c pairs each batched kernel with the batch-1 kernel it
// mirrors rather than reusing the prefill GEMMs, which are faster and would
// reassociate. Identity is not an accident here; it is the selection rule —
// and when a k_gemv_* body is rewritten, its twin below must be rewritten with
// it or the rule quietly stops holding (it did, for three weeks in 2026-07).
// What holds it down is tests/test_batch.c run on a QUANTIZED model: `make
// test` runs it on test-q8.gguf for exactly that reason, because the F32
// fixture takes k_mv_f32/k_mv_f32_b and never touches this family at all.

// grid: (ceil(half_dim/32), n_heads, batch); pos per column
extern "C" __global__ void k_rope_seq(float *v, const float *fr, rope_args a,
                                      const int *posp, int vs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (j >= a.half_dim || h >= a.n_heads) return;
    int pos = posp[blockIdx.z];
    float ang = pos * fr[j];
    float c = cosf(ang) * a.mscale, s = sinf(ang) * a.mscale;
    float *p = v + (ulong64)blockIdx.z * vs + h * a.head_dim;
    int i0 = a.neox ? j : 2 * j;
    int i1 = a.neox ? j + a.half_dim : i0 + 1;
    float x0 = p[i0], x1 = p[i1];
    p[i0] = x0 * c - x1 * s;
    p[i1] = x0 * s + x1 * c;
}

// grid.y = sequence column; each column stores into its OWN cache at its OWN
// position, so two sequences at the same position never collide
extern "C" __global__ void k_store_kv_seq(const float *k, const float *v,
                                          const ulong64 *kcp, const ulong64 *vcp,
                                          int kv_dim, ulong64 l_off,
                                          const int *posp, int q8, int ring) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = q8 ? kv_dim / 32 : kv_dim;
    if (i < n) {
        int sq = blockIdx.y;
        ulong64 row_b = KV_ROW_BYTES(kv_dim, q8);
        ulong64 dst = l_off + kv_slot(posp[sq], ring) * row_b;
        const float *ks = k + (ulong64)sq * kv_dim;
        const float *vs = v + (ulong64)sq * kv_dim;
        kv_store_row((unsigned char *)kcp[sq] + dst, ks, kv_dim, q8, i);
        kv_store_row((unsigned char *)vcp[sq] + dst, vs, kv_dim, q8, i);
    }
}

// Flash-decoding attention over N sequences. Body is k_attn_dec verbatim with
// pos and the cache pointers taken per column; the within-slice reduction, the
// split count and the partial layout are untouched, so k_attn_merge (which
// reads neither position nor cache) serves this path unchanged and each
// column's result is bitwise what the unbatched decode produces.
// LAUNCH INVARIANT: blockDim.x must be a power of two and <= 128 (red[]'s
// size) for the tree reduction and shared indexing here. Host launches with 128.
extern "C" __global__ void k_attn_dec_seq(const float *q, const ulong64 *kcp,
                                          const ulong64 *vcp, float *att,
                                          float *part, attn_args a,
                                          const int *posp) {
    __shared__ float red[128];
    int h = blockIdx.x, sp = blockIdx.y, tk = blockIdx.z;
    int tid = threadIdx.x, tpg = blockDim.x;
    int pos = posp[tk];
    const unsigned char *kc = (const unsigned char *)kcp[tk];
    const unsigned char *vc = (const unsigned char *)vcp[tk];
    int hd = a.head_dim;
    int kvh = h / (a.n_head / a.n_head_kv);
    int kv_dim = a.n_head_kv * hd;
    ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8);
    ulong64 base  = a.l_off + kv_head_off(kvh, hd, a.q8);
    int t0 = 0;
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    int total = pos + 1 - t0;
    int slice = (total + ATTN_SPLITS - 1) / ATTN_SPLITS;
    int s0 = t0 + sp * slice;
    int s1 = s0 + slice;
    if (s1 > pos + 1) s1 = pos + 1;
    float *P = part + (((ulong64)tk * a.n_head + h) * ATTN_SPLITS + sp) * (hd + 2);
    if (s0 >= s1) {                       // empty slice: sentinel, skipped in merge
        if (tid == 0) { P[hd] = -1e30f; P[hd + 1] = 0.f; }
        return;
    }
    const float *qh = q + (ulong64)tk * a.qs + h * hd;
    float *ah = att + ((ulong64)tk * a.n_head + h) * a.n_ctx;

    for (int t = s0 + tid; t < s1; t += tpg)
        ah[t] = kv_dot(kc + base + kv_slot(t, a.ring) * row_b, qh, hd, a.q8) * a.scale;
    __syncthreads();
    float mx = -1e30f;
    for (int t = s0 + tid; t < s1; t += tpg) mx = fmaxf(mx, ah[t]);
    red[tid] = mx;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0;
    for (int t = s0 + tid; t < s1; t += tpg) {
        float e = expf(ah[t] - mx);
        ah[t] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    sum = red[0];
    __syncthreads();
    for (int i2 = tid; i2 < hd / 2; i2 += tpg) {
        float o0 = 0, o1 = 0;
        for (int t = s0; t < s1; t++) {
            float2 vf = kv_pair(vc + base + kv_slot(t, a.ring) * row_b, i2, a.q8);
            o0 += ah[t] * vf.x;
            o1 += ah[t] * vf.y;
        }
        P[2 * i2]     = o0;
        P[2 * i2 + 1] = o1;
    }
    if (tid == 0) { P[hd] = mx; P[hd + 1] = sum; }
}

// ---- multi-column twins of the decode GEMVs ----
//
// Two requirements pull in opposite directions here.
//
// IDENTITY says each column must emit the same FMA sequence, in the same
// order, with the same warp_sum tree, as the batch-1 k_gemv_* kernel it
// replaces. That fixes the arithmetic completely; the only freedom left is
// where x is read from, and reading it from shared memory changes no value.
//
// SPEED says the column loop must be unrolled — a runtime trip count costs
// more than the wasted columns it saves, measured. But an unrolled loop has a
// fixed width, so a kernel that always does eight columns costs the same for
// two sequences as for eight, and a half-full microbatch pays full price.
//
// Both are satisfied by generating each kernel at several fixed widths and
// letting cuda.c launch the narrowest one that covers the microbatch. The
// bodies below are macros instantiated per width for exactly that reason;
// NC is the compile-time column count, always <= MVB, and the buffers stay
// MVB columns wide so the strides are unchanged.
//
// Q4_K and Q5_K also have a width-8 twin already in the tree — k_gemm_q4_K and
// k_gemm_q5_K were built on this same lane geometry for prefill — but they are
// left alone and re-derived here so the prefill path keeps the kernels it was
// verified with, and so every width comes from one source.
//
// DERIVE EACH BODY FROM THE CURRENT k_gemv_* SOURCE, NEVER FROM THE COMMENT
// ABOVE IT. These macros were exact twins when they were written (d0439ea,
// 2026-07-20) and stopped being twins eight days later, when 7ef0209 rewrote
// k_gemv_q8_0/_q4_K/_q5_K/_q6_K into their v2 load shapes and left the macros
// at v1. The comments still claimed twinhood the whole time, so for three weeks
// every quantized microbatch silently returned different bits than a lone
// decode — diagnosed in docs/cuda-microbatch-identity-2026-08-18.md. The macros
// below are re-derived from the v2 bodies; what holds them down now is
// `./test-batch test-q8.gguf`, a QUANTIZED fixture that make test runs, because
// the F32 fixture alone could never have caught it.
//
// Only two things may differ from the batch-1 kernel: x is read from shared
// memory rather than global, and per-block quantities the columns share are
// hoisted. Neither changes a value. The lane->element mapping, the per-lane
// accumulation order and the warp_sum partition must match exactly.

// ---- Q8_0: k_gemv_q8_0 v2 — lane l takes elements [(l&7)*4, +4) of block
// b0+(l>>3), four blocks per trip, then v1's element-per-lane tail. x staged.
// (k_gemm_q8_0 maps a lane to a whole block instead, so it is not a twin.)
// Blocks are staged SMPAD floats apart, not 32: with a 32-float stride the
// four block-groups of a warp land on the same eight banks (4-way conflict);
// the odd stride spreads them across all 32.
#define GEMVB_Q8_0(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][Q8_CHUNK * SMPAD];                                \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 32;                                                      \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 34;  \
    int bsub = (int)(lane >> 3);                                               \
    int boff = ((int)lane & 7) * 4;                                            \
    float s[NC] = {0};                                                         \
    for (int cs = 0; cs < nb; cs += Q8_CHUNK) {                                \
        int cblocks = nb - cs < Q8_CHUNK ? nb - cs : Q8_CHUNK;                 \
        int celems  = cblocks * 32;                                            \
        int base_e  = cs * 32;                                                 \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            /* chunks are Q8_CHUNK-aligned and Q8_CHUNK is a multiple of 4, so \
               the batch-1 kernel's b4 = nb & ~3 boundary falls inside the     \
               LAST chunk at exactly cblocks & ~3 — every lane therefore sees  \
               the same four-block trips, then the same tail, in that order */ \
            int c4 = cblocks & ~3;                                             \
            for (int bi = 0; bi < c4; bi += 4) {                               \
                const uchar *blk = rw + (ulong64)(cs + bi + bsub) * 34;        \
                float d = f16f(blk);                                           \
                const uchar *qp = blk + 2 + boff;                              \
                ushort16 u0 = *(const ushort16 *)qp,                           \
                         u1 = *(const ushort16 *)(qp + 2);                     \
                float q0 = (float)(int)(signed char)(u0 & 0xFF);               \
                float q1 = (float)(int)(signed char)(u0 >> 8);                 \
                float q2 = (float)(int)(signed char)(u1 & 0xFF);               \
                float q3 = (float)(int)(signed char)(u1 >> 8);                 \
                int xo = (bi + bsub) * SMPAD + boff;                           \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++) {                                 \
                    const float *xp = xsm[t] + xo;                             \
                    s[t] += d * (q0 * xp[0] + q1 * xp[1] +                     \
                                 q2 * xp[2] + q3 * xp[3]);                     \
                }                                                              \
            }                                                                  \
            for (int bi = c4; bi < cblocks; bi++) {                            \
                const uchar *blk = rw + (ulong64)(cs + bi) * 34;               \
                float d = f16f(blk);                                           \
                const signed char *q = (const signed char *)(blk + 2);         \
                float qv = (float)q[lane];                                     \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++)                                   \
                    s[t] += d * (qv * xsm[t][bi * SMPAD + lane]);              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// ---- Q4_0: k_gemv_q4_0's twin (2026-08-19). Q4_0 had no width-classed twin
// at all, so enc_mv_batch fell through to f_mvb: a different reduction AND no
// x staging, measured at 0.11x of sequential decode. Same four-blocks-in-
// flight shape as Q8_0; a lane's two quant bytes carry elements boff, boff+1
// (low nibbles) and boff+16, boff+17 (high nibbles).
#define GEMVB_Q4_0(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][Q8_CHUNK * SMPAD];                                \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 32;                                                      \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 18;  \
    int bsub = (int)(lane >> 3);                                               \
    int boff = ((int)lane & 7) * 2;                                            \
    float s[NC] = {0};                                                         \
    for (int cs = 0; cs < nb; cs += Q8_CHUNK) {                                \
        int cblocks = nb - cs < Q8_CHUNK ? nb - cs : Q8_CHUNK;                 \
        int celems  = cblocks * 32;                                            \
        int base_e  = cs * 32;                                                 \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            int c4 = cblocks & ~3;                                             \
            for (int bi = 0; bi < c4; bi += 4) {                               \
                const uchar *blk = rw + (ulong64)(cs + bi + bsub) * 18;        \
                float d = f16f(blk);                                           \
                ushort16 u = *(const ushort16 *)(blk + 2 + boff);              \
                float q0 = (float)((int)( u        & 0xF) - 8);                \
                float q1 = (float)((int)((u >>  4) & 0xF) - 8);                \
                float q2 = (float)((int)((u >>  8) & 0xF) - 8);                \
                float q3 = (float)((int)((u >> 12) & 0xF) - 8);                \
                int xo = (bi + bsub) * SMPAD + boff;                           \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++) {                                 \
                    const float *xp = xsm[t] + xo;                             \
                    s[t] += d * (q0 * xp[0] + q2 * xp[1] +                     \
                                 q1 * xp[16] + q3 * xp[17]);                   \
                }                                                              \
            }                                                                  \
            for (int bi = c4; bi < cblocks; bi++) {                            \
                const uchar *blk = rw + (ulong64)(cs + bi) * 18;               \
                float d = f16f(blk);                                           \
                const uchar *q = blk + 2;                                      \
                int j = (int)lane & 15, hi = (int)lane >> 4;                   \
                int qv = hi ? (q[j] >> 4) : (q[j] & 0xF);                      \
                float w = d * (float)(qv - 8);                                 \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++)                                   \
                    s[t] += w * xsm[t][bi * SMPAD + lane];                     \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// Q4_K/Q5_K stage x in groups of the EIGHT elements a lane owns, one padding
// float per group: with a flat 256-float row all 32 lanes read el = lane*8,
// i.e. only four distinct banks (8-way conflict). A 9-float group stride makes
// bank (9*lane + k) & 31 a bijection over the warp — no conflict, same values.
#define KG8      8              // elements one lane owns in a k-quant block
#define KG8PAD   9              // 8 + 1: makes per-lane smem reads conflict-free
#define KG8ROW   (32 * KG8PAD)  // staged floats per column per 256-element block

// ---- Q4_K: k_gemv_q4_K v2 — lane l owns elements [l*8, l*8+8), the per-group
// affine factored out as dg*Sum(nib*x) - mmg*Sum(x). The v1 form this macro
// used to carry (a separate `dg*nib - mmg` FMA per element straight into s)
// is a different expression, hence different bits.
#define GEMVB_Q4_K(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][KG8ROW];                                          \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 256;                                                     \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 144; \
    float s[NC] = {0};                                                         \
    int g     = (int)(lane >> 2);                                              \
    int ji    = (int)(lane >> 3);                                              \
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;                         \
    int bbase = ((int)lane & 3) * 8;                                           \
    int el    = (int)lane * KG8PAD;                                            \
    for (int b = 0; b < nb; b++) {                                             \
        const uchar *blk = rw + (ulong64)b * 144;                              \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][(e >> 3) * KG8PAD + (e & 7)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            float dd   = f16f(blk);                                            \
            float dmin = f16f(blk + 2);                                        \
            uchar sg, mg;                                                      \
            get_scale_min_k4(g, blk + 4, &sg, &mg);                            \
            float dg = dd * (float)sg, mmg = dmin * (float)mg;                 \
            uint2 qv = *(const uint2 *)(blk + 16 + ji * 32 + bbase);           \
            uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu;                              \
            uint v1 = (qv.y >> sh) & 0x0F0F0F0Fu;                              \
            float n0 = (float)(v0 & 0xFF),        n1 = (float)((v0 >>  8) & 0xFF); \
            float n2 = (float)((v0 >> 16) & 0xFF), n3 = (float)((v0 >> 24));    \
            float n4 = (float)(v1 & 0xFF),        n5 = (float)((v1 >>  8) & 0xFF); \
            float n6 = (float)((v1 >> 16) & 0xFF), n7 = (float)((v1 >> 24));    \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                const float *xp = xsm[t] + el;                                 \
                float tt = n0 * xp[0] + n1 * xp[1] + n2 * xp[2] + n3 * xp[3]   \
                         + n4 * xp[4] + n5 * xp[5] + n6 * xp[6] + n7 * xp[7];  \
                float sx = xp[0] + xp[1] + xp[2] + xp[3]                       \
                         + xp[4] + xp[5] + xp[6] + xp[7];                      \
                s[t] += dg * tt - mmg * sx;                                    \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// ---- Q5_K: k_gemv_q5_K v2, i.e. Q4_K's factored form with the fifth bit —
// bit g of each qh byte, added as 16 before the conversion to float.
#define GEMVB_Q5_K(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][KG8ROW];                                          \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 256;                                                     \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 176; \
    float s[NC] = {0};                                                         \
    int g     = (int)(lane >> 2);                                              \
    int ji    = (int)(lane >> 3);                                              \
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;                         \
    int bbase = ((int)lane & 3) * 8;                                           \
    int hshift = g;                                                            \
    int el    = (int)lane * KG8PAD;                                            \
    for (int b = 0; b < nb; b++) {                                             \
        const uchar *blk = rw + (ulong64)b * 176;                              \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][(e >> 3) * KG8PAD + (e & 7)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            float dd   = f16f(blk);                                            \
            float dmin = f16f(blk + 2);                                        \
            uchar sg, mg;                                                      \
            get_scale_min_k4(g, blk + 4, &sg, &mg);                            \
            float dg = dd * (float)sg, mmg = dmin * (float)mg;                 \
            uint2 qv = *(const uint2 *)(blk + 48 + ji * 32 + bbase);           \
            uint2 hv = *(const uint2 *)(blk + 16 + bbase);                     \
            uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu;                              \
            uint v1 = (qv.y >> sh) & 0x0F0F0F0Fu;                              \
            uint h0 = ((hv.x >> hshift) & 0x01010101u) << 4;                   \
            uint h1 = ((hv.y >> hshift) & 0x01010101u) << 4;                   \
            v0 += h0; v1 += h1;                                                \
            float n0 = (float)(v0 & 0xFF),        n1 = (float)((v0 >>  8) & 0xFF); \
            float n2 = (float)((v0 >> 16) & 0xFF), n3 = (float)((v0 >> 24));    \
            float n4 = (float)(v1 & 0xFF),        n5 = (float)((v1 >>  8) & 0xFF); \
            float n6 = (float)((v1 >> 16) & 0xFF), n7 = (float)((v1 >> 24));    \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                const float *xp = xsm[t] + el;                                 \
                float tt = n0 * xp[0] + n1 * xp[1] + n2 * xp[2] + n3 * xp[3]   \
                         + n4 * xp[4] + n5 * xp[5] + n6 * xp[6] + n7 * xp[7];  \
                float sx = xp[0] + xp[1] + xp[2] + xp[3]                       \
                         + xp[4] + xp[5] + xp[6] + xp[7];                      \
                s[t] += dg * tt - mmg * sx;                                    \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// ---- Q6_K: k_gemv_q6_K v2 — d factored out of the four-term group, a whole
// BLOCK reduced into its own accumulator, and those block sums split across
// TWO running accumulators (even blocks into s0, odd into s1, s0 + s1 at the
// end) because v2 keeps two blocks in flight. Folding both into one running
// sum, as this macro used to, is a different reduction tree.
// (k_gemm_q6_K premultiplies d into each weight instead, which agrees in exact
// arithmetic but not necessarily in floating point, so it is not a twin.)
#define GEMVB_Q6_K(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][256];                                             \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    int nb = a.n_in / 256;                                                     \
    const uchar *rw = wb + a.w_off +                                           \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 210; \
    float s0[NC] = {0}, s1[NC] = {0};                                          \
    int is = (int)(lane >> 4);                                                 \
    for (int b = 0; b < nb; b++) {                                             \
        const uchar *blk = rw + (ulong64)b * 210;                              \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][e] = xg[e];                                             \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out) {                                         \
            float d = f16f(blk + 208);                                         \
            float acc[NC] = {0};                                               \
            _Pragma("unroll")                                                  \
            for (int half = 0; half < 2; half++) {                             \
                const uchar *ql = blk + half * 64;                             \
                const uchar *qh = blk + 128 + half * 32;                       \
                const signed char *sc =                                        \
                    (const signed char *)(blk + 192) + half * 8;               \
                int q1 = (int)((ql[lane]      & 0xF) |                         \
                               (((qh[lane] >> 0) & 3) << 4)) - 32;             \
                int q2 = (int)((ql[lane + 32] & 0xF) |                         \
                               (((qh[lane] >> 2) & 3) << 4)) - 32;             \
                int q3 = (int)((ql[lane]      >> 4)  |                         \
                               (((qh[lane] >> 4) & 3) << 4)) - 32;             \
                int q4 = (int)((ql[lane + 32] >> 4)  |                         \
                               (((qh[lane] >> 6) & 3) << 4)) - 32;             \
                float c1 = (float)(sc[0 + is] * q1);                           \
                float c2 = (float)(sc[2 + is] * q2);                           \
                float c3 = (float)(sc[4 + is] * q3);                           \
                float c4 = (float)(sc[6 + is] * q4);                           \
                int e0 = half * 128;                                           \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++) {                                 \
                    const float *xp = xsm[t] + e0;                             \
                    acc[t] += d * (c1 * xp[lane]      + c2 * xp[lane + 32] +   \
                                   c3 * xp[lane + 64] + c4 * xp[lane + 96]);   \
                }                                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                if (b & 1) s1[t] += acc[t]; else s0[t] += acc[t];              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s0[t] + s1[t]);                                 \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// ---- BF16: k_mv_bf16's twin (2026-08-20). bf16 decodes at batch 1 through
// k_mv_bf16 (there is no k_gemv_bf16), whose lane l accumulates elements
// l, l+32, l+64, ... of the row in ascending order. bf16 is element-strided —
// no quant blocks — so a chunk of BF16_CHUNK elements (a multiple of 32)
// preserves exactly that partition and order: lane l's elements ascend within
// each chunk and chunks ascend, so only the x source (shared vs global)
// differs, which changes no value. Reads are lane-consecutive per trip, so no
// SMPAD padding is needed. Without this twin BF16 fell to f_mvb, whose
// scattered global x-loads measured well below sequential decode (the same
// 0.11x-class loss the Q4_0 comment above records).
#define BF16_CHUNK 1024
#define GEMVB_BF16(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][BF16_CHUNK];                                      \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    const unsigned short *rw = (const unsigned short *)(wb + a.w_off) +        \
                      (ulong64)(row < (unsigned)a.n_out ? row : 0) * a.n_in;   \
    float s[NC] = {0};                                                         \
    for (int cs = 0; cs < a.n_in; cs += BF16_CHUNK) {                          \
        int celems = a.n_in - cs < BF16_CHUNK ? a.n_in - cs : BF16_CHUNK;      \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            const float *xg = x + (ulong64)t * a.xs + cs;                      \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][e] = xg[e];                                             \
        }                                                                      \
        __syncthreads();                                                       \
        if (row < (unsigned)a.n_out)                                           \
            for (int i = lane; i < celems; i += 32) {                          \
                float w = bf16f(rw[cs + i]);                                   \
                _Pragma("unroll")                                              \
                for (int t = 0; t < NC; t++) s[t] += w * xsm[t][i];            \
            }                                                                  \
        __syncthreads();                                                       \
    }                                                                          \
    if (row < (unsigned)a.n_out)                                               \
        for (int t = 0; t < a.batch && t < NC; t++) {                          \
            float r = warp_sum(s[t]);                                          \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}

// Widths cuda.c can pick from. Two are enough to cover 2..8 without a
// half-empty batch paying much: a microbatch of 3 runs the 4-wide kernel.
GEMVB_BF16(k_gemvb_bf16_x4, 4)
GEMVB_BF16(k_gemvb_bf16_x8, 8)
GEMVB_Q8_0(k_gemvb_q8_0_x4, 4)
GEMVB_Q8_0(k_gemvb_q8_0_x8, 8)
GEMVB_Q4_0(k_gemvb_q4_0_x4, 4)
GEMVB_Q4_0(k_gemvb_q4_0_x8, 8)
GEMVB_Q4_K(k_gemvb_q4_K_x4, 4)
GEMVB_Q4_K(k_gemvb_q4_K_x8, 8)
GEMVB_Q5_K(k_gemvb_q5_K_x4, 4)
GEMVB_Q5_K(k_gemvb_q5_K_x8, 8)
GEMVB_Q6_K(k_gemvb_q6_K_x4, 4)
GEMVB_Q6_K(k_gemvb_q6_K_x8, 8)

// ---------------------------------------------------------------- elementwise
// grid.y = token column for k_add (different x/d strides); silu operates on
// the contiguous [batch][n_ff] region in one launch

extern "C" __global__ void k_silu_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = g[i];
        g[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}

extern "C" __global__ void k_gelu_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = g[i];
        float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
        g[i] = 0.5f * x * (1.0f + t) * u[i];
    }
}

// gpt-oss clamped alpha-sigmoid GLU, 1:1 with gated_act(ACT_SWIGLU_OAI) in
// model.c (itself transcribed from llama.cpp's swiglu_oai): gate clamped
// above only, up clamped both sides, up carries a +1 shift. The x <= -50
// early-zero mirrors the CPU guard exactly so both backends emit the same
// value there (0.0f rather than the -0.0f the division limit would give).
static __device__ __forceinline__ float swiglu_oai(float g, float u) {
    const float alpha = 1.702f, limit = 7.0f;
    float x = g < limit ? g : limit;
    float y = u < -limit ? -limit : (u > limit ? limit : u);
    float gl = x < -50.0f ? 0.0f : x / (1.0f + expf(alpha * -x));
    return gl * (y + 1.0f);
}

extern "C" __global__ void k_swiglu_oai_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = swiglu_oai(g[i], u[i]);
}

extern "C" __global__ void k_add(float *x, const float *d, int n, int xs, int ds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(ulong64)blockIdx.y * xs + i] += d[(ulong64)blockIdx.y * ds + i];
}

// whole-layer output scalar (gemma4): x *= s, grid.y = token column
extern "C" __global__ void k_scale(float *x, float s, int n, int xs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[(ulong64)blockIdx.y * xs + i] *= s;
}

// ------------------------------------------------- sparse-MoE device routing
// Device-side softmax -> top-k -> renormalize, replacing the per-token
// DtoH round-trip that forced MoE decode onto the eager (graph_bad) path.
//
// BIT-IDENTITY CONTRACT (moe-gpu-routing spec): this kernel is the device
// mirror of moe_route() in model.c, and the byte-identical CPU==GPU greedy
// cert depends on it selecting exactly as the host would given the same
// logits. It therefore runs SERIALLY, one thread per token, with the host's
// exact arithmetic: same max-scan order, one expf per element in element
// order, same summation order, division (not reciprocal-multiply), and the
// strict `>` compare that sends ties to the lowest expert index. At
// n_expert <= 256 a serial thread is microseconds per token; do not
// restructure this into a parallel reduction — reordering floats here is
// what the contract forbids.
//
// logits: [tokens][ls]; sel/selw: [tokens][used]. lg[] is a local (per-
// thread) working copy so the router logits buffer itself stays intact.
extern "C" __global__ void k_moe_route(const float *logits, int *sel,
                                       float *selw, int ne, int used,
                                       int tokens, int ls) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;
    float lg[256];
    const float *src = logits + (ulong64)t * ls;
    for (int e = 0; e < ne; e++) lg[e] = src[e];
    float mx = lg[0];
    for (int e = 1; e < ne; e++)
        if (lg[e] > mx) mx = lg[e];
    float ssum = 0.0f;
    for (int e = 0; e < ne; e++) {
        // Device expf differs from the host libm by ~1-2 ulp, so the fused
        // path is NOT byte-identical to the host routing — the certified
        // MoE byte-identity property is therefore defined over the EAGER
        // path (RUNNER_MOE_EAGER=1, pinned in the certification harnesses;
        // see docs/compatibility-program.md). The fused default's contract
        // is the weaker, verified class: expert selection identical, selw
        // within 1 ulp of the host reference (the reciprocal-multiply
        // mirror below keeps the divergence to expf alone). History: a
        // correctly-rounded (float)exp((double)x) here DID bit-match
        // correctly-rounded hosts (UCRT), but a fast-math/libmvec host is
        // ~4 ulp and unreachable by construction, so certification pinned
        // eager and this kernel keeps the plain fp32 expf.
        float p = expf(lg[e] - mx);
        lg[e] = p;
        ssum += p;
    }
    // Per-element division, not a hoisted reciprocal-multiply. A
    // reciprocal mirror of the host's -freciprocal-math codegen briefly
    // lived here chasing selw bit-identity; that goal was retired by the
    // eager certification pin (RUNNER_MOE_EAGER=1, bf93510), and the
    // Blackwell splice-proof (suite plan, 2026-07-29 evening: per-kernel
    // PTX hashing + splicing the old body restores 102.9 tok/s from 79.7)
    // showed the rcp.rn.f32 form JITs ~58 us/launch slower on the MIG —
    // x48 layers = 23% of MoE decode. Plain division satisfies the fused
    // contract (selection identical, selw within ~2 ulp of the host).
    for (int e = 0; e < ne; e++) lg[e] /= ssum;
    int   *ts = sel  + (ulong64)t * used;
    float *tw = selw + (ulong64)t * used;
    float denom = 0.0f;
    for (int s = 0; s < used; s++) {
        int best = 0;
        float bp = -1.0f;
        for (int e = 0; e < ne; e++)
            if (lg[e] > bp) { bp = lg[e]; best = e; }
        ts[s] = best;
        tw[s] = bp;
        denom += bp;
        lg[best] = -1.0f;
    }
    // per-element division, same rationale as the softmax normalization
    for (int s = 0; s < used; s++) tw[s] /= denom;
}

// -------------------------------------------- indirect expert matvec (MoE)
// One launch computes ALL selected experts' matvecs for one token: grid.y is
// the expert slot, the weight base is resolved in-kernel from sel[slot] and
// the fused-3D expert stride (the same arithmetic moe_expert_weight does on
// the host — moved into the kernel args so no host round-trip remains).
//
// Numerics: each body below is a verbatim copy of the kernel enc_mv would
// have launched for that quant type at batch 1 (k_gemv_* where the coalesced
// variant exists, k_mv_* otherwise) — only the weight-base computation and
// the slot-indexed x/y columns differ, so per-row results are bit-identical
// to the eager path's launches and every existing per-model cert carries
// over. Expert FFNs carry no bias in any supported arch, so there is no bias
// parameter.
//
// xs = x column stride per slot (0: all slots read the same input, the
// gate/up case; nff: per-slot hidden, the down case); ys = y column stride.


#define MOE_MV_HEAD \
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); \
    unsigned lane = threadIdx.x & 31; \
    if (row >= (unsigned)a.n_out) return; \
    const uchar *wbase = wb + a.w_off + (ulong64)sel[blockIdx.y] * a.estride; \
    x += (ulong64)blockIdx.y * a.xs; \
    float s = 0;

#define MOE_MV_TAIL \
    s = warp_sum(s); \
    if (lane == 0) y[(ulong64)blockIdx.y * a.ys + row] = s;

#define MOE_MV_PARAMS \
    const uchar *wb, const float *x, float *y, moe_args a, const int *sel

// body of k_mv_f32
extern "C" __global__ void k_moe_mv_f32(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    const float *rw = (const float *)wbase + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) s += rw[i] * x[i];
    MOE_MV_TAIL;
}

// body of k_mv_f16
extern "C" __global__ void k_moe_mv_f16(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    const __half *rw = (const __half *)wbase + (ulong64)row * a.n_in;
    for (int i = lane; i < a.n_in; i += 32) s += __half2float(rw[i]) * x[i];
    MOE_MV_TAIL;
}

// body of k_gemv_q8_0 (the batch-1 kernel enc_mv picks for Q8_0)
extern "C" __global__ void k_moe_mv_q8_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)row * nb * 34;
    int bsub = (int)(lane >> 3);
    int boff = ((int)lane & 7) * 4;
    int b4 = nb & ~3;
    for (int b0 = 0; b0 < b4; b0 += 4) {
        const uchar *blk = rw + (ulong64)(b0 + bsub) * 34;
        float d = f16f(blk);
        const uchar *qp = blk + 2 + boff;
        ushort16 u0 = *(const ushort16 *)qp, u1 = *(const ushort16 *)(qp + 2);
        int q0 = (int)(signed char)(u0 & 0xFF), q1 = (int)(signed char)(u0 >> 8);
        int q2 = (int)(signed char)(u1 & 0xFF), q3 = (int)(signed char)(u1 >> 8);
        const float4 xv = *(const float4 *)(x + (ulong64)(b0 + bsub) * 32 + boff);
        s += d * ((float)q0 * xv.x + (float)q1 * xv.y +
                  (float)q2 * xv.z + (float)q3 * xv.w);
    }
    for (int b = b4; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 34;
        float d = f16f(blk);
        const signed char *q = (const signed char *)(blk + 2);
        s += d * ((float)q[lane] * x[(ulong64)b * 32 + lane]);
    }
    MOE_MV_TAIL;
}

// body of k_mv_q4_0 (no coalesced GEMV exists for Q4_0 yet)
extern "C" __global__ void k_moe_mv_q4_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)row * nb * 18;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
        s += d * t;
    }
    MOE_MV_TAIL;
}

// body of k_mv_mxfp4 (gpt-oss experts; no coalesced GEMV exists for MXFP4)
extern "C" __global__ void k_moe_mv_mxfp4(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)row * nb * 17;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 17;
        float d = ldexpf(1.0f, (int)blk[0] - 127);
        const uchar *q = blk + 1;
        const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MOE_MV_TAIL;
}

// body of k_mv_nvfp4 (ModelOpt NVFP4 expert tensors)
extern "C" __global__ void k_moe_mv_nvfp4(MOE_MV_PARAMS, float scale) {
    MOE_MV_HEAD;
    int nb = a.n_in / 64;
    const uchar *rw = wbase + (ulong64)row * nb * 36;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 36;
        for (int sub = 0; sub < 4; sub++) {
            float d = ue4m3f(blk[sub]);
            const uchar *q = blk + 4 + sub * 8;
            const float *xp = x + b * 64 + sub * 16;
            float t = 0;
            for (int j = 0; j < 8; j++) {
                t += kv_mxfp4[q[j] & 0xF] * xp[j];
                t += kv_mxfp4[q[j] >> 4]  * xp[j + 8];
            }
            s += d * t;
        }
    }
    s = warp_sum(s) * scale;
    if (lane == 0) y[(ulong64)blockIdx.y * a.ys + row] = s;
}

// body of k_gemv_q4_K
extern "C" __global__ void k_moe_mv_q4_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wbase + (ulong64)row * nb * 144;
    int g     = (int)(lane >> 2);
    int ji    = (int)(lane >> 3);
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;
    int bbase = ((int)lane & 3) * 8;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 144;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 16 + ji * 32 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MOE_MV_TAIL;
}

// body of k_gemv_q5_K
extern "C" __global__ void k_moe_mv_q5_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wbase + (ulong64)row * nb * 176;
    int g     = (int)(lane >> 2);
    int ji    = (int)(lane >> 3);
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;
    int bbase = ((int)lane & 3) * 8;
    int hshift = g;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 176;
        float dd   = f16f(blk);
        float dmin = f16f(blk + 2);
        uchar sg, mg;
        get_scale_min_k4(g, blk + 4, &sg, &mg);
        float dg = dd * (float)sg, mmg = dmin * (float)mg;
        uint2 qv = *(const uint2 *)(blk + 48 + ji * 32 + bbase);
        uint2 hv = *(const uint2 *)(blk + 16 + bbase);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu, v1 = (qv.y >> sh) & 0x0F0F0F0Fu;
        uint h0 = ((hv.x >> hshift) & 0x01010101u) << 4;
        uint h1 = ((hv.y >> hshift) & 0x01010101u) << 4;
        v0 += h0; v1 += h1;
        float t  = (float)(v0 & 0xFF)         * x0.x
                 + (float)((v0 >>  8) & 0xFF) * x0.y
                 + (float)((v0 >> 16) & 0xFF) * x0.z
                 + (float)((v0 >> 24)       ) * x0.w
                 + (float)(v1 & 0xFF)         * x1.x
                 + (float)((v1 >>  8) & 0xFF) * x1.y
                 + (float)((v1 >> 16) & 0xFF) * x1.z
                 + (float)((v1 >> 24)       ) * x1.w;
        float sx = x0.x + x0.y + x0.z + x0.w + x1.x + x1.y + x1.z + x1.w;
        s += dg * t - mmg * sx;
    }
    MOE_MV_TAIL;
}

// body of k_gemv_q6_K
extern "C" __global__ void k_moe_mv_q6_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wbase + (ulong64)row * nb * 210;
    int is = (int)(lane >> 4);
    float s0 = 0, s1 = 0;
    int b2 = nb & ~1;
    for (int b = 0; b < b2; b += 2) {
        s0 += q6k_block_dot(rw + (ulong64)b * 210,       x + (ulong64)b * 256,
                            (int)lane, is);
        s1 += q6k_block_dot(rw + (ulong64)(b + 1) * 210, x + (ulong64)(b + 1) * 256,
                            (int)lane, is);
    }
    if (b2 < nb)
        s0 += q6k_block_dot(rw + (ulong64)b2 * 210, x + (ulong64)b2 * 256,
                            (int)lane, is);
    s = s0 + s1;
    MOE_MV_TAIL;
}

// Gated activation + routing-weight fold for every expert slot in one launch.
// grid.y = expert slot. Per element: the exact k_silu_mul / k_gelu_mul
// arithmetic, then a separate multiply by the slot's routing weight — the
// same two rounding steps the eager path produced with its actmul + scale
// launch pair, so values are bit-identical to it. dscale is the per-expert
// down-projection scale table (gemma-4), indexed via sel; ones when the
// model has none, so the multiply is exact and harmless.
// gbuf column stride gss, ubuf column stride uss (gemma reads gate and up
// out of one fused 2*nff column: gss = uss = 2*nff, ubuf = gbuf + nff).
// act: 0 = SiLU, 1 = tanh-GELU, 2 = gpt-oss swiglu_oai — same selector values
// as ACT_* in runner.h. gb/ub are per-expert gate/up bias tables ([n_expert]
// rows of nff, gpt-oss) or NULL; the slot's expert index picks the row, and
// the bias lands BEFORE the activation, matching moe_ffn_token on the CPU.
extern "C" __global__ void k_moe_actmul(float *gbuf, const float *ubuf,
                                        int nff, int gss, int uss, int act,
                                        const float *selw, const float *dscale,
                                        const int *sel, const float *gb,
                                        const float *ub) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nff) return;
    float *gp = gbuf + (ulong64)blockIdx.y * gss;
    const float *up = ubuf + (ulong64)blockIdx.y * uss;
    float xg = gp[i], xu = up[i], v;
    if (gb) xg += gb[(ulong64)sel[blockIdx.y] * nff + i];
    if (ub) xu += ub[(ulong64)sel[blockIdx.y] * nff + i];
    if (act == 2) {
        v = swiglu_oai(xg, xu);
    } else if (act == 1) {
        float t = tanhf(0.7978845608f * (xg + 0.044715f * xg * xg * xg));
        v = 0.5f * xg * (1.0f + t) * xu;
    } else {
        v = (xg / (1.0f + expf(-xg))) * xu;
    }
    float w = selw[blockIdx.y] * dscale[sel[blockIdx.y]];
    gp[i] = v * w;
}

// Sum the per-slot down-projections into the token's FFN output, slot 0
// first then ascending — the same accumulation order as the eager path's
// write-then-add sequence, so the sum is bit-identical to it.
// db: per-expert down-bias table ([n_expert] rows of n, gpt-oss) or NULL.
// The routing weight was folded into the hidden before the down matvec, so
// each slot's eout is already w*down(h); the CPU computes w*(down(h)+db),
// and w*down(h) + selw*db is the same quantity with the fold's association.
extern "C" __global__ void k_moe_sum(float *out, const float *eout, int n,
                                     int nslots, int es, const float *db,
                                     const int *sel, const float *selw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = eout[i];
    if (db) s += selw[0] * db[(ulong64)sel[0] * n + i];
    for (int sl = 1; sl < nslots; sl++) {
        s += eout[(ulong64)sl * es + i];
        if (db) s += selw[sl] * db[(ulong64)sel[sl] * n + i];
    }
    out[i] = s;
}

// ---------------------------------------- expert-grouped prefill (MoE P2)
// CUDA port of the CPU cabdad1 grouping: route the tile, then run each
// active expert ONCE over all the tokens routed to it as a batched GEMM.
// These three kernels are the glue around the existing k_gemm family:
// gather the expert's token columns, and scatter its weighted down outputs
// back into the per-token accumulator. grid.y = position in the expert's
// token list.

// dst[c][i] = src[idx[c]][i]; ss/ds = element stride between columns
extern "C" __global__ void k_moe_gather(float *dst, const float *src,
                                        const int *idx, int n, int ss,
                                        int ds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(ulong64)blockIdx.y * ds + i] = src[(ulong64)idx[blockIdx.y] * ss + i];
}

// out[idx[c]][i] += w[c] * src[c][i]. Launched once per expert on the
// stream, so no two active launches write the same token column (a token
// appears at most once in one expert's list) — no atomics needed.
extern "C" __global__ void k_moe_scatter_add(float *out, const float *src,
                                             const int *idx, const float *w,
                                             int n, int os, int ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[(ulong64)idx[blockIdx.y] * os + i] +=
        w[blockIdx.y] * src[(ulong64)blockIdx.y * ss + i];
}

// Gated activation over per-column gate/up at arbitrary column strides,
// with NO weight fold (the grouped path applies selw in the scatter, like
// the CPU grouping). Same per-element arithmetic as k_silu_mul/k_gelu_mul.
// Covers gemma's fused gate_up layout (gss = uss = 2*nff, ubuf = gbuf+nff),
// where the contiguous k_silu_mul cannot run across columns.
extern "C" __global__ void k_moe_actmul_plain(float *gbuf, const float *ubuf,
                                              int nff, int gss, int uss,
                                              int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nff) return;
    float *gp = gbuf + (ulong64)blockIdx.y * gss;
    const float *up = ubuf + (ulong64)blockIdx.y * uss;
    float xg = gp[i];
    if (act == 2) {
        gp[i] = swiglu_oai(xg, up[i]);
    } else if (act == 1) {
        float t = tanhf(0.7978845608f * (xg + 0.044715f * xg * xg * xg));
        gp[i] = 0.5f * xg * (1.0f + t) * up[i];
    } else {
        gp[i] = (xg / (1.0f + expf(-xg))) * up[i];
    }
}

// strided column copy: dst[c][i] = src[c][i] (accumulator -> xb columns)
extern "C" __global__ void k_moe_copy_cols(float *dst, const float *src,
                                           int n, int ds, int ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(ulong64)blockIdx.y * ds + i] = src[(ulong64)blockIdx.y * ss + i];
}

// ---------------------------------------------------------- Qwen3.5 hybrid
// The generic matvec kernels above perform every learned projection.  These
// small kernels implement only the architecture-specific stateful operators,
// keeping the runtime compact and the quantized weight support in one place.

// Qwen3.5 full-attention Q is stored head-interleaved as [Q_h, gate_h].
extern "C" __global__ void k_q35_split_q(const float *packed, float *q,
                                          float *gate, int heads, int hd,
                                          int packed_stride, int q_stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int qdim = heads * hd;
    if (i >= qdim) return;
    int h = i / hd, j = i - h * hd;
    const float *src = packed + (ulong64)blockIdx.y * packed_stride;
    q[(ulong64)blockIdx.y * q_stride + i] = src[h * 2 * hd + j];
    gate[(ulong64)blockIdx.y * q_stride + i] = src[h * 2 * hd + hd + j];
}

extern "C" __global__ void k_q35_attn_gate(float *x, const float *gate,
                                            int n, int xs, int gs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ulong64 xo = (ulong64)blockIdx.y * xs + i;
        ulong64 go = (ulong64)blockIdx.y * gs + i;
        x[xo] *= 1.0f / (1.0f + expf(-gate[go]));
    }
}

// One launch per token is intentional: convolution history is recurrent, so
// prompt columns cannot update it concurrently. Each channel is independent.
extern "C" __global__ void k_q35_conv(const float *mix, float *cv,
                                       const float *weight, float *history,
                                       int convdim, int kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= convdim) return;
    int histn = kernel - 1;
    const float *cw = weight + (ulong64)c * kernel;
    float sum = cw[histn] * mix[c];
    for (int k = 0; k < histn; k++)
        sum += cw[k] * history[(ulong64)k * convdim + c];
    cv[c] = sum / (1.0f + expf(-sum));
    for (int k = 0; k + 1 < histn; k++)
        history[(ulong64)k * convdim + c] =
            history[(ulong64)(k + 1) * convdim + c];
    if (histn) history[(ulong64)(histn - 1) * convdim + c] = mix[c];
}

// One block per value head. The model geometry guarantees hv == state/key
// width. State is [head][value-column][key-row], matching the CPU reference.
extern "C" __global__ void k_q35_delta(float *cv, const float *z,
                                        const float *beta_in,
                                        const float *alpha_in,
                                        const float *dt, const float *a,
                                        const float *norm, float *state,
                                        float *out, int state_dim, int groups,
                                        int heads, float eps) {
    __shared__ float red[256];
    int tid = threadIdx.x, h = blockIdx.x;
    if (h >= heads) return;
    int keydim = state_dim * groups;
    int group = h % groups;
    float *q = cv + group * state_dim;
    float *k = cv + keydim + group * state_dim;
    const float *v = cv + 2 * keydim + h * state_dim;
    float *st = state + (ulong64)h * state_dim * state_dim;
    float *yo = out + h * state_dim;

    float qs = 0.0f;
    for (int i = tid; i < state_dim; i += blockDim.x) qs += q[i] * q[i];
    red[tid] = qs;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float qscale = rsqrtf(red[0] + eps);
    float ks = 0.0f;
    for (int i = tid; i < state_dim; i += blockDim.x) ks += k[i] * k[i];
    red[tid] = ks;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float kscale = rsqrtf(red[0] + eps);
    float beta = 1.0f / (1.0f + expf(-beta_in[h]));
    float av = alpha_in[h] + dt[h];
    float softplus = av > 20.0f ? av : log1pf(expf(av));
    float decay = expf(a[h] * softplus);

    for (int j = tid; j < state_dim; j += blockDim.x) {
        float *row = st + (ulong64)j * state_dim;
        float pred = 0.0f;
        for (int i = 0; i < state_dim; i++) {
            row[i] *= decay;
            pred += row[i] * (k[i] * kscale);
        }
        float delta = (v[j] - pred) * beta;
        float y = 0.0f;
        for (int i = 0; i < state_dim; i++) {
            row[i] += delta * (k[i] * kscale);
            y += row[i] * (q[i] * qscale);
        }
        yo[j] = y * rsqrtf((float)state_dim);
    }
    __syncthreads();
    float ss = 0.0f;
    for (int j = tid; j < state_dim; j += blockDim.x) ss += yo[j] * yo[j];
    red[tid] = ss;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / state_dim + eps);
    for (int j = tid; j < state_dim; j += blockDim.x) {
        int o = h * state_dim + j;
        float zv = z[o];
        yo[j] = yo[j] * rms * norm[j] * (zv / (1.0f + expf(-zv)));
    }
}

// Apertus ungated activation, matching model.h:xielu(). `an` and `ap` are
// already the effective softplus-transformed parameters from model_load.
// Kept after the existing kernels so adding it does not renumber their PTX
// labels and obscure review of the generated header.
extern "C" __global__ void k_xielu(float *x, int n, float an, float ap,
                                    float b, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        if (v > 0.0f) {
            x[i] = ap * v * v + b * v;
        } else {
            float mn = v < eps ? v : eps;
            x[i] = (expm1f(mn) - v) * an + b * v;
        }
    }
}
// --------------------------------------------- tensor-core Q6_K prefill GEMM
//
// The 2026-08-08 profile said prefill is 42.7% k_gemm_q4_K_tc, 30.2% k_attn
// and 26.0% k_gemm_q6_K -- and q6_K had NO tensor-core variant at all, so
// `attn_v` and `ffn_down` in every Q4_K_M file (40% of GEMM calls) ran the
// scalar path.
//
// Same structure as k_gemm_q4_K_tc: the block dequantizes a 64-row x 128-K
// fp16 weight tile once, its four warps' MMAs share it, and the epilogue tile
// aliases the weight tile to stay under the 48 KB shared cap.
//
// Only the unpacking differs, and it is taken verbatim from k_gemm_q6_K so the
// arithmetic matches the scalar path element for element -- including the
// INTEGER `sc * q` before the float multiply:
//
//   ql = blk + half*64, qh = blk + 128 + half*32,
//   sc = (int8*)(blk+192) + half*8, d = f16(blk+208)
//   K = half*128 + l      : (ql[l]    & 0xF) | ((qh[l]>>0 & 3)<<4) - 32, sc[0+is]
//   K = half*128 + l + 32 : (ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4) - 32, sc[2+is]
//   K = half*128 + l + 64 : (ql[l]     >> 4) | ((qh[l]>>4 & 3)<<4) - 32, sc[4+is]
//   K = half*128 + l + 96 : (ql[l+32]  >> 4) | ((qh[l]>>6 & 3)<<4) - 32, sc[6+is]
//   is = l >> 4
//
// A thread stages 64 consecutive K of one row, so sseg 0 covers the first two
// quarters of the 128-K half and sseg 1 the last two.
//
// Not bit-identical to k_gemm_q6_K by construction (the weight is dequantized
// before the multiply and the sum is reassociated into 8-element matrix
// steps), exactly like the q4_K tensor-core path, so it answers to the same
// tolerance gate.

extern "C" __global__ void k_gemm_q6_K_tc(MV_PARAMS) {
    using namespace nvcuda::wmma;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const unsigned row0 = blockIdx.x * TC_ROWS;
    __shared__ __half sh_w[TC_ROWS * TC_K];
    __shared__ __half sh_x[TC_N * TC_K];
    float *sh_c = (float *)sh_w;

    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc[TC_N / 16];
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++) fill_fragment(fc[n], 0.0f);

    int nb = a.n_in / 256;
    int srow = tid >> 1, sseg = tid & 1;

    for (int b = 0; b < nb; b++) {
        #pragma unroll
        for (int koff = 0; koff < 256; koff += TC_K) {
            {   // ---- stage weights: 64 rows x 128 K, dequantized once ----
                unsigned gr = row0 + srow;
                __half *dst = sh_w + srow * TC_K + sseg * 64;
                if (gr < (unsigned)a.n_out) {
                    const uchar *blk = wb + a.w_off +
                                       (ulong64)gr * nb * 210 + (ulong64)b * 210;
                    const int half_ = koff >> 7;          // 0 or 1
                    const uchar *ql = blk + half_ * 64;
                    const uchar *qh = blk + 128 + half_ * 32;
                    const signed char *sc =
                        (const signed char *)(blk + 192) + half_ * 8;
                    float d = f16f(blk + 208);
                    // sseg 0 -> quarters 0,1 (low nibbles, qh shifts 0,2)
                    // sseg 1 -> quarters 2,3 (high nibbles, qh shifts 4,6)
                    const int shA = sseg * 4 + 0, shB = sseg * 4 + 2;
                    const int scA = sseg * 4 + 0, scB = sseg * 4 + 2;
                    #pragma unroll
                    for (int l = 0; l < 32; l++) {
                        int is = l >> 4;
                        uchar a0 = ql[l], b0 = ql[l + 32], h = qh[l];
                        int qa = (int)((sseg ? (a0 >> 4) : (a0 & 0xF))
                                       | (((h >> shA) & 3) << 4)) - 32;
                        int qb = (int)((sseg ? (b0 >> 4) : (b0 & 0xF))
                                       | (((h >> shB) & 3) << 4)) - 32;
                        dst[l]      = __float2half(d * (float)(sc[scA + is] * qa));
                        dst[l + 32] = __float2half(d * (float)(sc[scB + is] * qb));
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            {   // ---- stage activations: 128 K x TC_N tokens ----
                int col = tid / 2, part = tid % 2;
                __half *dst = sh_x + col * TC_K + part * 64;
                if (col < a.batch) {
                    const float *xg = x + (ulong64)col * a.xs + b * 256 + koff
                                      + part * 64;
                    #pragma unroll
                    for (int v = 0; v < 16; v++) {
                        float4 xv = *(const float4 *)(xg + v * 4);
                        dst[v * 4 + 0] = __float2half(xv.x);
                        dst[v * 4 + 1] = __float2half(xv.y);
                        dst[v * 4 + 2] = __float2half(xv.z);
                        dst[v * 4 + 3] = __float2half(xv.w);
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            __syncthreads();
            const __half *wt = sh_w + warp * 16 * TC_K;
            #pragma unroll
            for (int k = 0; k < TC_K; k += 16) {
                load_matrix_sync(fa, wt + k, TC_K);
                #pragma unroll
                for (int n = 0; n < TC_N / 16; n++) {
                    load_matrix_sync(fb, sh_x + n * 16 * TC_K + k, TC_K);
                    mma_sync(fc[n], fa, fb, fc[n]);
                }
            }
            __syncthreads();
        }
    }

    __syncthreads();
    #pragma unroll
    for (int n = 0; n < TC_N / 16; n++)
        store_matrix_sync(sh_c + warp * 16 * TC_N + n * 16, fc[n], TC_N,
                          mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < TC_ROWS * TC_N; idx += blockDim.x) {
        int rr = idx / TC_N, tt = idx % TC_N;
        unsigned gr = row0 + rr;
        if (gr < (unsigned)a.n_out && tt < a.batch)
            y[(ulong64)tt * a.ys + gr] =
                a.has_bias ? sh_c[idx] + bias[gr] : sh_c[idx];
    }
}

// ==========================================================================
// Mamba-2 selective SSD scan (granitehybrid / nemotron_h) — decode + prefill.
// Per-token kernels launched in sequence by gpu_mamba2_recurrent (cuda.c),
// which loops tokens on the host. The conv ring and the SSD state are
// persistent device buffers updated in place, so an eager decode continues
// bit-for-bit like the CPU serial core (model.c: mamba2_ssd_core_serial):
//   conv1d over [x,B,C] + bias + silu ; h_t = exp(dt*A)*h + (dt*B)*x_t ;
//   y = C*h + D*x ; then silu(z) gate + grouped RMSNormGated.
// B/C are shared across a group of n_head/n_group heads (g = h / (nh/ng)).
// ==========================================================================

// Phase 1: causal depthwise conv1d for token t, then bias + silu, then advance
// the conv ring (drop the oldest column, append this token's PRE-conv input).
// grid = ceil(conv_dim/256), block 256; one thread per channel c. Weight row c
// is [conv_kernel] floats; tap histn multiplies the current input (llama concats
// the ring BEFORE the new column). Only thread c touches ring column c.
extern "C" __global__ void k_mamba2_conv(
        const float *proj, float *xBC, const float *conv_w, const float *conv_b,
        float *ring, int conv_dim, int inner, int d_in_proj,
        int conv_kernel, int histn, int t) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const float *cw = conv_w + (size_t)c * conv_kernel;
    float cur = proj[(size_t)t * d_in_proj + inner + c];
    float sum = conv_b[c] + cw[histn] * cur;
    for (int k = 0; k < histn; k++)
        sum += cw[k] * ring[(size_t)k * conv_dim + c];
    xBC[(size_t)t * conv_dim + c] = sum / (1.0f + expf(-sum));   // silu
    for (int k = 0; k < histn - 1; k++)                          // shift ring left
        ring[(size_t)k * conv_dim + c] = ring[(size_t)(k + 1) * conv_dim + c];
    if (histn) ring[(size_t)(histn - 1) * conv_dim + c] = cur;   // append current
}

// Phase 2: SSD recurrence for token t. grid = n_head blocks (blockIdx.x = h),
// block = head_dim threads (threadIdx.x = p). Each (h,p) sweeps d_state
// sequentially in the SAME order as the CPU inner loop, so the float rounding
// matches. dt/A/D are per head; B/C are per group g = h/gsz.
extern "C" __global__ void k_mamba2_ssd(
        const float *xBC, const float *proj, const float *A, const float *dt_b,
        const float *D, float *state, float *y,
        int inner, int hd, int ds, int ng, int gsz, int conv_dim,
        int d_in_proj, int t) {
    int h = blockIdx.x;
    int p = threadIdx.x;
    if (p >= hd) return;
    int g = h / gsz;
    float dtr = proj[(size_t)t * d_in_proj + inner + conv_dim + h] + dt_b[h];
    float dt = dtr > 20.0f ? dtr : logf(1.0f + expf(dtr));       // softplus_f32
    float dA = expf(dt * A[h]);
    const float *xrow = xBC + (size_t)t * conv_dim;
    const float *Bg = xrow + inner + (size_t)g * ds;
    const float *Cg = xrow + inner + (size_t)ng * ds + (size_t)g * ds;
    float x = xrow[(size_t)h * hd + p];
    float x_dt = x * dt;
    float *sp = state + ((size_t)h * hd + p) * ds;
    float sumf = 0.0f;
    for (int j = 0; j < ds; j++) {
        float s = sp[j] * dA + Bg[j] * x_dt;
        sumf += s * Cg[j];
        sp[j] = s;
    }
    y[(size_t)t * inner + (size_t)h * hd + p] = sumf + D[h] * x;
}

// Phase 3: gate by silu(z) then grouped RMS norm (Mamba-2 RMSNormGated) for
// token t. grid = n_group blocks (blockIdx.x = group), block 256. The group's
// per_g = inner/n_group activations are gated then normalized together; the
// gate is elementwise so restricting it to the group's slice is exact. Sum of
// squares is accumulated in double, matching the CPU norm (tree order differs).
extern "C" __global__ void k_mamba2_gate_norm(
        const float *proj, float *y, const float *gnorm_w,
        int inner, int ng, int per_g, float eps, int d_in_proj, int t) {
    int gidx = blockIdx.x;
    int tid = threadIdx.x, blk = blockDim.x;
    const float *zg = proj + (size_t)t * d_in_proj + (size_t)gidx * per_g;
    float *yg = y + (size_t)t * inner + (size_t)gidx * per_g;
    const float *wg = gnorm_w + (size_t)gidx * per_g;
    for (int i = tid; i < per_g; i += blk) {                     // gate: y *= silu(z)
        float zz = zg[i];
        yg[i] *= zz / (1.0f + expf(-zz));
    }
    __syncthreads();
    double part = 0.0;
    for (int i = tid; i < per_g; i += blk)
        part += (double)yg[i] * (double)yg[i];
    __shared__ double red[256];
    red[tid] = part;
    __syncthreads();
    for (int s = blk >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float mean = (float)(red[0] / per_g);
    float scale = 1.0f / sqrtf(mean + eps);
    __syncthreads();
    for (int i = tid; i < per_g; i += blk)
        yg[i] = yg[i] * scale * wg[i];
}

// gate-less squared-ReLU FFN activation (nemotron_h): x = relu(x)^2 in place.
extern "C" __global__ void k_relu2(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r = x[i] > 0.0f ? x[i] : 0.0f;
    x[i] = r * r;
}
