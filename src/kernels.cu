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

// The RMS scale from the block's per-thread partials: their sum in the
// pairwise order of a shared-memory tree (red[t] += red[t + off] for off =
// tpg/2 down to 1). Levels of 32 and up go through shared memory; the last
// five run as shuffles inside warp 0 on the same operands in the same order
// (lane t adds lane t + off), so the sum is the all-shared tree's bit for
// bit with five fewer barriers. tpg is a power of two from 32 to 256.
static __device__ __forceinline__ float rms_scale(float s, float *red, int tid,
                                                  int tpg, int n, float eps) {
    red[tid] = s;
    __syncthreads();
    for (int off = tpg / 2; off >= 32; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid < 32) {
        float v = red[tid];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    return rsqrtf(red[0] / n + eps);
}

// RMSNorm of one row per block (blockIdx.y the column), 256 threads. A
// decode token's norm is one block on an otherwise idle card, so its time
// is latency: rows of up to RMS_REG elements a thread are read once, x and
// the weights together, kept in registers across the reduction and scaled
// from them, one round trip to memory instead of six. Each thread still sums
// its elements tid, tid + tpg, ... in increasing order, so every output is
// the reloading loop's bit for bit; longer rows take that loop.
#define RMS_REG 16
extern "C" __global__ void k_rmsnorm(const float *x, float *y, const float *w,
                                     int n, float eps, int xs, int ys) {
    __shared__ float red[256];
    int tid = threadIdx.x, tpg = blockDim.x;
    x += (ulong64)blockIdx.y * xs;
    y += (ulong64)blockIdx.y * ys;
    float s = 0;
    if (n <= RMS_REG * tpg) {
        float xv[RMS_REG], wv[RMS_REG];
        #pragma unroll
        for (int k = 0; k < RMS_REG; k++) {
            int i = tid + k * tpg;
            xv[k] = i < n ? x[i] : 0.0f;
            wv[k] = i < n ? w[i] : 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < RMS_REG; k++)
            if (tid + k * tpg < n) s += xv[k] * xv[k];
        float r = rms_scale(s, red, tid, tpg, n, eps);
        #pragma unroll
        for (int k = 0; k < RMS_REG; k++) {
            int i = tid + k * tpg;
            if (i < n) y[i] = xv[k] * r * wv[k];
        }
        return;
    }
    for (int i = tid; i < n; i += tpg) s += x[i] * x[i];
    float r = rms_scale(s, red, tid, tpg, n, eps);
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

// Activation staging for the one-lane-per-block matvecs (see LANE_XPAD at
// k_mv_q4_0) at any block width: EL-element blocks in rows of
// EL + 1 floats (lane l's element j in bank (l + j) % 32), for the remaining
// one-lane-per-block kernels (X21b: the decode kernels of the types with no
// coalesced GEMV, and their MoE copies). The block's rows share each copy;
// dead rows join the barriers and skip the math and the store.
#define LANE_STAGE_XN(xsm, xsrc, b0, nblk, EL)                                 \
    for (int i_ = threadIdx.x; i_ < (nblk) * (EL); i_ += blockDim.x)           \
        (xsm)[(i_ / (EL)) * ((EL) + 1) + (i_ % (EL))] = (xsrc)[(ulong64)(b0) * (EL) + i_];
#define LANE_DENSE_HEAD(EL)                                                    \
    __shared__ float xsm[32 * ((EL) + 1)];                                     \
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);        \
    unsigned lane = threadIdx.x & 31;                                          \
    bool live = row < (unsigned)a.n_out;


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

// Q2_0 (ggml type 42): 64 codes per 18-byte block, code j at bits 2*(j%4) of
// byte j/4, value (code - 1) * d; the device twin of quants.c dq_q2_0.
extern "C" __global__ void k_mv_q2_0(MV_PARAMS) {
    LANE_DENSE_HEAD(64)
    int nb = a.n_in / 64;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 18;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 64)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (64 + 1);
            const uchar *blk = rw + (ulong64)b * 18;
            float d = f16f(blk);
            const uchar *q = blk + 2;
            const float *xp = xs_;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                uchar c = q[j];
                t += ((int)(c & 3) - 1) * xp[4 * j]
                   + ((int)((c >> 2) & 3) - 1) * xp[4 * j + 1]
                   + ((int)((c >> 4) & 3) - 1) * xp[4 * j + 2]
                   + ((int)(c >> 6) - 1) * xp[4 * j + 3];
            }
            s += d * t;
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
}

extern "C" __global__ void k_mv_q2_0_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 64;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 18;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 18;
        float d = f16f(blk);
        const uchar *q = blk + 2;
        ulong64 base = (ulong64)b * 64;
        for (int j = 0; j < 16; j++) {
            uchar c = q[j];
            MV_FMA(d * (float)((int)(c & 3) - 1),        base + 4 * j);
            MV_FMA(d * (float)((int)((c >> 2) & 3) - 1), base + 4 * j + 1);
            MV_FMA(d * (float)((int)((c >> 4) & 3) - 1), base + 4 * j + 2);
            MV_FMA(d * (float)((int)(c >> 6) - 1),       base + 4 * j + 3);
        }
    }
    MV_TAIL_B;
}

// One lane per 32-element block (k_mv_q4_0 and k_mv_mxfp4, their MoE copies
// and their by-expert twins), with the activations staged. Read straight from
// global memory, lane l walks its own block's 32 floats of x, 128 bytes from
// its neighbours, so each warp load touched 32 cache lines. The block's warps
// (four rows a block in the dense kernels, eight in the MoE copies and the
// twins; every row of a launch reads the same column) copy 32 blocks of x at a time
// into shared memory with coalesced loads, at a row stride of LANE_XPAD = 33
// floats so lane l's element j sits in bank (l + j) % 32, conflict-free. Each
// lane still takes blocks lane, lane + 32, ... in order with the same
// expression, so the bits are the unstaged kernels'. No early return: a row
// past n_out joins the barriers and skips the math and the store. 5-7x on the
// gemma-4-26B Q4_0 and gpt-oss MXFP4 expert shapes (RTX 3070 microbench).
#define LANE_XPAD 33
#define LANE_STAGE_X(xsm, xsrc, b0, nblk)                                      \
    for (int i_ = threadIdx.x; i_ < (nblk) * 32; i_ += blockDim.x)             \
        (xsm)[(i_ >> 5) * LANE_XPAD + (i_ & 31)] = (xsrc)[(ulong64)(b0) * 32 + i_];

extern "C" __global__ void k_mv_q4_0(MV_PARAMS) {
    __shared__ float xsm[32 * LANE_XPAD];
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    unsigned lane = threadIdx.x & 31;
    bool live = row < (unsigned)a.n_out;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 18;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_X(xsm, x, b0, nblk)
        __syncthreads();
        if (live && (int)lane < nblk) {
            const uchar *blk = rw + (ulong64)(b0 + lane) * 18;
            float d = f16f(blk);
            const uchar *q = blk + 2;
            const float *xp = xsm + lane * LANE_XPAD;
            float t = 0;
            for (int j = 0; j < 16; j++)
                t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
            s += d * t;
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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
    LANE_DENSE_HEAD(32)
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 20;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 32)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (32 + 1);
            const uchar *blk = rw + (ulong64)b * 20;
            float d  = f16f(blk);
            float mm = f16f(blk + 2);
            const uchar *q = blk + 4;
            const float *xp = xs_;
            float t = 0, sx = 0;
            for (int j = 0; j < 16; j++) {
                t += (float)(q[j] & 0xF) * xp[j] + (float)(q[j] >> 4) * xp[j + 16];
                sx += xp[j] + xp[j + 16];
            }
            s += d * t + mm * sx;
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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
    LANE_DENSE_HEAD(32)
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 22;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 32)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (32 + 1);
            const uchar *blk = rw + (ulong64)b * 22;
            float d = f16f(blk);
            uint qh = (uint)blk[2] | ((uint)blk[3] << 8) |
                      ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
            const uchar *q = blk + 6;
            const float *xp = xs_;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                int x0 = (int)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
                int x1 = (int)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                t += x0 * xp[j] + x1 * xp[j + 16];
            }
            s += d * t;
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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
    LANE_DENSE_HEAD(32)
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 24;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 32)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (32 + 1);
            const uchar *blk = rw + (ulong64)b * 24;
            float d  = f16f(blk);
            float mm = f16f(blk + 2);
            uint qh = (uint)blk[4] | ((uint)blk[5] << 8) |
                      ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
            const uchar *q = blk + 8;
            const float *xp = xs_;
            float t = 0, sx = 0;
            for (int j = 0; j < 16; j++) {
                t += (float)((q[j] & 0xF) | (((qh >> j) & 1u) << 4)) * xp[j] +
                     (float)((q[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) * xp[j + 16];
                sx += xp[j] + xp[j + 16];
            }
            s += d * t + mm * sx;
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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

// Split-K GEMV for the few-long-rows shape (qwen4exp's hyper-connection
// down projections: 320 rows of 10,240, and the 4-row inject). A warp per
// row leaves most of the device idle and the call latency-bound; here block
// (row, split) reduces one K-slice into partial[row*splits+split] in a fixed
// tree, and k_splitk_reduce sums the splits in index order, so the result is
// deterministic (no atomics) and the same for every launch of the same shape.
extern "C" __global__ void k_mv_bf16_splitk(const uchar *wb, const float *x, float *partial,
                                            mv_args a, int splits) {
    __shared__ float red[256];
    int row = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x;
    int slice = a.n_in / splits, i0 = sp * slice;
    const unsigned short *rw = (const unsigned short *)(wb + a.w_off) + (ulong64)row * a.n_in + i0;
    const float *xr = x + i0;
    float s = 0;
    for (int i = tid; i < slice; i += blockDim.x) s += bf16f(rw[i]) * xr[i];
    red[tid] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) partial[(ulong64)row * splits + sp] = red[0];
}

extern "C" __global__ void k_splitk_reduce(const float *partial, float *y, const float *bias,
                                           int n_out, int splits, int has_bias) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_out) return;
    float s = 0;
    for (int sp = 0; sp < splits; sp++) s += partial[(ulong64)row * splits + sp];
    y[row] = has_bias ? s + bias[row] : s;
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
    LANE_DENSE_HEAD(256)
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 84;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 256)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (256 + 1);
            const uchar *blk = rw + (ulong64)b * 84;
            Q2K_SETUP;
            const float *xp = xs_;
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
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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
// Leading dimension of the staged tiles, in halves. A row of TC_K halves is
// 256 bytes, so the eight 16-byte rows one ldmatrix phase reads (and the
// lanes of a staging store, one row each) all fall in the same four banks:
// every fragment load was an 8-way conflict and every scalar staging store
// up to 32-way. Eight halves of padding moves row r by 4r banks. Only shared
// addresses change: the operands, the MMAs and their order are the same, so
// the outputs are bit-identical to the unpadded kernels.
#define TC_KP   (TC_K + 8)

extern "C" __global__ void k_gemm_q4_K_tc(MV_PARAMS, const float *xsc) {
    using namespace nvcuda::wmma;
    const int tid  = threadIdx.x;         // 128 threads = 4 warps
    const int warp = tid >> 5;
    const unsigned row0 = blockIdx.x * TC_ROWS;
    __shared__ __align__(32) __half sh_w[TC_ROWS * TC_KP];   // row-major, ldm = TC_KP
    __shared__ __align__(32) __half sh_x[TC_N * TC_KP];      // col-major (k,t) at [t*TC_KP+k]
    // The epilogue tile ALIASES the weight tile: sh_w is 64*136 halves = 17 KB
    // and sh_c needs TC_ROWS*TC_N floats = 16 KB; their lifetimes are disjoint
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
                __half *dst = sh_w + srow * TC_KP + sseg * 64;
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
            // ---- stage activations: 128 K x TC_N tokens, vectorized ----
            // Each column scaled to fp16's range by its own max|x| (xsc,
            // from k_colabsmax): a model's activation outliers can exceed
            // 65504 (Phi-4-mini requantized to IQ3_S reaches 1.6e5), and an
            // Inf operand is a NaN logit. Scaled back in the epilogue.
            {
                int col = tid / 2, part = tid % 2;
                __half *dst = sh_x + col * TC_KP + part * 64;
                if (col < a.batch) {
                    const float xinv = 16384.0f / xsc[col];
                    const float *xg = x + (ulong64)col * a.xs + b * 256 + koff
                                      + part * 64;
                    #pragma unroll
                    for (int v = 0; v < 16; v++) {
                        float4 xv = *(const float4 *)(xg + v * 4);
                        dst[v * 4 + 0] = __float2half(xv.x * xinv);
                        dst[v * 4 + 1] = __float2half(xv.y * xinv);
                        dst[v * 4 + 2] = __float2half(xv.z * xinv);
                        dst[v * 4 + 3] = __float2half(xv.w * xinv);
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            __syncthreads();
            // ---- 8 MMA K-steps over the staged tile ----
            const __half *wt = sh_w + warp * 16 * TC_KP;
            #pragma unroll
            for (int k = 0; k < TC_K; k += 16) {
                load_matrix_sync(fa, wt + k, TC_KP);
                // one staged weight tile, reused across every n-tile: this is
                // the entire point of widening TC_N
                #pragma unroll
                for (int n = 0; n < TC_N / 16; n++) {
                    load_matrix_sync(fb, sh_x + n * 16 * TC_KP + k, TC_KP);
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
            float r = sh_c[rr * TC_N + tt] * (xsc[tt] * (1.0f / 16384.0f));
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
// operands. Both were opt-in when written; tc_promoted() in cuda.c now
// lists Q8_0 (2026-08-09) and Q4_0 (2026-08-13) on their make test-tc-tol
// rows, and RUNNER_CUDA_TC=0 pins the scalar path.
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
// TC_GEMM_BLK is the same kernel for any block size: BLKELEMS elements per
// BLKBYTES-byte block, nb blocks per row, STAGE decoding one 64-element
// segment at element offset e0 of a row. TC_GEMM_32B is its 32-element
// instance (Q8_0, Q4_0, unchanged PTX); the 256-element codebook i-quants
// instantiate it below with their own stagers.
#define TC_GEMM_32B(NAME, STAGE, BLKBYTES) TC_GEMM_BLK_X(NAME, STAGE, BLKBYTES, 32, 1)
#define TC_GEMM_BLK(NAME, STAGE, BLKBYTES, BLKELEMS) TC_GEMM_BLK_X(NAME, STAGE, BLKBYTES, BLKELEMS, 1)
// XSCALED = 1, every tensor-core kernel's form since 2026-09-14: fp16 has
// no room for a model's activation outliers (Phi-4-mini requantized to
// IQ3_S reaches |x| = 1.6e5 at its first position in layer 30, found by the
// non-finite check the review asked for: every logit came out NaN), so
// each token column is staged as x * (2^14 / max|x|) and the product is
// scaled back in the epilogue. The per-column maxima arrive in the trailing
// parameter (k_colabsmax fills them right before the launch). XSCALED = 0
// is the previous unscaled form, kept for reference and unused.
#define TC_XPARAM_0
#define TC_XPARAM_1 , const float *xsc
#define TC_XSCALE_STAGE_0(col) const float xinv = 1.0f;
#define TC_XSCALE_STAGE_1(col) const float xinv = 16384.0f / xsc[col];
#define TC_XSCALE_OUT_0(tt) const float xmul = 1.0f;
#define TC_XSCALE_OUT_1(tt) const float xmul = xsc[tt] * (1.0f / 16384.0f);
#define TC_GEMM_BLK_X(NAME, STAGE, BLKBYTES, BLKELEMS, XSCALED)                \
extern "C" __global__ void NAME(MV_PARAMS TC_XPARAM_##XSCALED) {               \
    using namespace nvcuda::wmma;                                              \
    const int tid  = threadIdx.x;                                              \
    const int warp = tid >> 5;                                                 \
    const unsigned row0 = blockIdx.x * TC_ROWS;                                \
    __shared__ __align__(32) __half sh_w[TC_ROWS * TC_KP];                     \
    __shared__ __align__(32) __half sh_x[TC_N * TC_KP];                        \
    /* the epilogue tile aliases the weight tile (dead after the last MMA, */  \
    /* which the loop's closing barrier orders), as in k_gemm_q4_K_tc      */  \
    float *sh_c = (float *)sh_w;                                               \
    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;                      \
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;                      \
    fragment<accumulator, 16, 16, 16, float> fc[TC_N / 16];                    \
    _Pragma("unroll")                                                          \
    for (int n = 0; n < TC_N / 16; n++) fill_fragment(fc[n], 0.0f);            \
    int nb = a.n_in / BLKELEMS;                                                \
    int srow = tid >> 1, sseg = tid & 1;                                       \
    for (int ks = 0; ks < a.n_in; ks += TC_K) {                                \
        {                                                                      \
            unsigned gr = row0 + srow;                                         \
            __half *dst = sh_w + srow * TC_KP + sseg * 64;                     \
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
            __half *dst = sh_x + col * TC_KP + part * 64;                      \
            TC_XSCALE_STAGE_##XSCALED(col < a.batch ? col : 0)                 \
            _Pragma("unroll")                                                  \
            for (int h = 0; h < 2; h++) {                                      \
                __half *d2 = dst + h * 32;                                     \
                int e0 = ks + part * 64 + h * 32;                              \
                if (col < a.batch && e0 < a.n_in) {                            \
                    const float *xg = x + (ulong64)col * a.xs + e0;            \
                    _Pragma("unroll")                                          \
                    for (int v = 0; v < 8; v++) {                              \
                        float4 xv = *(const float4 *)(xg + v * 4);             \
                        d2[v * 4 + 0] = __float2half(xv.x * xinv);             \
                        d2[v * 4 + 1] = __float2half(xv.y * xinv);             \
                        d2[v * 4 + 2] = __float2half(xv.z * xinv);             \
                        d2[v * 4 + 3] = __float2half(xv.w * xinv);             \
                    }                                                          \
                } else {                                                       \
                    _Pragma("unroll")                                          \
                    for (int e = 0; e < 32; e++) d2[e] = __float2half(0.0f);   \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const __half *wt = sh_w + warp * 16 * TC_KP;                           \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < TC_K; k += 16) {                                   \
            load_matrix_sync(fa, wt + k, TC_KP);                               \
            _Pragma("unroll")                                                  \
            for (int n = 0; n < TC_N / 16; n++) {                              \
                load_matrix_sync(fb, sh_x + n * 16 * TC_KP + k, TC_KP);        \
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
            TC_XSCALE_OUT_##XSCALED(tt)                                        \
            float r = sh_c[rr * TC_N + tt] * xmul;                             \
            y[(ulong64)tt * a.ys + gr] = a.has_bias ? r + bias[gr] : r;        \
        }                                                                      \
    }                                                                          \
}

// per-column max|x| over n_in elements of each of `batch` columns (stride
// xs), for the XSCALED kernels; a zero (or vanishing) column scales by 1.
// One block per column, 256 threads.
extern "C" __global__ void k_colabsmax(const float *x, float *out, int n_in,
                                       int xs, int batch) {
    __shared__ float red[256];
    int col = blockIdx.x, tid = threadIdx.x;
    if (col >= batch) return;
    const float *xc = x + (ulong64)col * xs;
    float m = 0.0f;
    for (int i = tid; i < n_in; i += blockDim.x) m = fmaxf(m, fabsf(xc[i]));
    red[tid] = m;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    if (tid == 0) out[col] = red[0] > 1e-30f ? red[0] : 1.0f;   /* a tiny max would overflow 2^14 / max */
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
// Q3_K decode GEMV in the k_gemv_q5_K geometry: lane owns 8 consecutive
// elements of each 256-block, so a 32-lane warp covers the block in one pass
// with coalesced x loads. Element e's rules from k_mv_q3_K: half = e/128,
// j = (e%128)/32 (shift 2j, high-bit mask 1<<(4*half+j)), byte l = e%32 of
// qs + 32*half, scale index 8*half + 2j + (l >= 16); value
// ((q >> shift) & 3) - (hbit ? 0 : 4), times d_all * (scale - 32).
// A Q3_K block is 110 bytes, so block k starts at 6k mod 8: only every
// fourth block is 8-byte aligned and an 8-byte load faults on the rest (a
// misaligned-address error, sticky for the context). The 8 quant and 8
// high-mask bytes a lane takes are read as four 16-bit words instead (every
// block offset is even); the same bytes, so the same values.
static __device__ __forceinline__ uint2 ld8_a2(const uchar *p) {
    const unsigned short *h = (const unsigned short *)p;
    return make_uint2((uint)h[0] | ((uint)h[1] << 16), (uint)h[2] | ((uint)h[3] << 16));
}
extern "C" __global__ void k_gemv_q3_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 110;
    int half  = (int)(lane >> 4);
    int j     = (int)(lane >> 2) & 3;
    int lseg  = ((int)lane & 3) * 8;
    int shift = 2 * j, hsh = 4 * half + j;
    int sidx  = 8 * half + 2 * j + (lseg >= 16 ? 1 : 0);
    float s = 0;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 110;
        Q3K_UNPACK_SCALES;
        float dl = d_all * (float)(q3scales[sidx] - 32);
        uint2 qv = ld8_a2(qbase + 32 * half + lseg);
        uint2 hv = ld8_a2(hm + lseg);
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        uint v0 = (qv.x >> shift) & 0x03030303u, v1 = (qv.y >> shift) & 0x03030303u;
        uint h0 = (hv.x >> hsh) & 0x01010101u,  h1 = (hv.y >> hsh) & 0x01010101u;
        // value = q + 4*hbit - 4
        float t = ((float)(v0 & 0xFF)         + 4.0f * (float)(h0 & 0xFF)         - 4.0f) * x0.x
                + ((float)((v0 >>  8) & 0xFF) + 4.0f * (float)((h0 >>  8) & 0xFF) - 4.0f) * x0.y
                + ((float)((v0 >> 16) & 0xFF) + 4.0f * (float)((h0 >> 16) & 0xFF) - 4.0f) * x0.z
                + ((float)((v0 >> 24)       ) + 4.0f * (float)((h0 >> 24)       ) - 4.0f) * x0.w
                + ((float)(v1 & 0xFF)         + 4.0f * (float)(h1 & 0xFF)         - 4.0f) * x1.x
                + ((float)((v1 >>  8) & 0xFF) + 4.0f * (float)((h1 >>  8) & 0xFF) - 4.0f) * x1.y
                + ((float)((v1 >> 16) & 0xFF) + 4.0f * (float)((h1 >> 16) & 0xFF) - 4.0f) * x1.z
                + ((float)((v1 >> 24)       ) + 4.0f * (float)((h1 >> 24)       ) - 4.0f) * x1.w;
        s += dl * t;
    }
    MV_TAIL;
}

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

#include "kernels_iq4_table.cuh"

extern "C" __global__ void k_mv_iq4_nl(MV_PARAMS) {
    LANE_DENSE_HEAD(32)
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 18;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 32)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (32 + 1);
            const uchar *blk = rw + (ulong64)b * 18;
            float d = f16f(blk);
            const uchar *q = blk + 2;
            const float *xp = xs_;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                t += (float)kv_iq4[q[j] & 0xF] * xp[j];
                t += (float)kv_iq4[q[j] >> 4]  * xp[j + 16];
            }
            s += d * t;
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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

// IQ4_XS decode GEMV, same geometry. Block 136 bytes: d f16, scales_h u16,
// scales_l[4], qs[128]. Sub-block ib (32 values) reads qs[16*ib + j]: low
// nibble -> value j, high nibble -> value j+16, through the IQ4_NL table;
// scale ls = (scales_l nibble) | (scales_h 2 bits << 4), dl = d*(ls-32).
// Lane l: ib = l/4, quarter q = l%4 owns values 8q..8q+7 of that sub-block:
// q < 2 reads the low nibbles of bytes 8q..8q+7, q >= 2 the high nibbles of
// bytes 8(q-2)..8(q-2)+7.
extern "C" __global__ void k_gemv_iq4_xs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 136;
    int ib = (int)(lane >> 2), qd = (int)(lane & 3);
    int boff = 16 * ib + 8 * (qd & 1);
    int hi = qd >= 2;
    float s = 0;
    for (int b = 0; b < nb; b++) {
        const uchar *blk = rw + (ulong64)b * 136;
        float d = f16f(blk);
        unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
        int ls = ((blk[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
        float dl = d * (float)(ls - 32);
        uint2 qv = *(const uint2 *)(blk + 8 + boff);
        uint v0 = hi ? (qv.x >> 4) & 0x0F0F0F0Fu : qv.x & 0x0F0F0F0Fu;
        uint v1 = hi ? (qv.y >> 4) & 0x0F0F0F0Fu : qv.y & 0x0F0F0F0Fu;
        const float *xp = x + (ulong64)b * 256 + (int)lane * 8;
        float4 x0 = *(const float4 *)xp, x1 = *(const float4 *)(xp + 4);
        float t = (float)kv_iq4[v0 & 0xFF]         * x0.x
                + (float)kv_iq4[(v0 >>  8) & 0xFF] * x0.y
                + (float)kv_iq4[(v0 >> 16) & 0xFF] * x0.z
                + (float)kv_iq4[(v0 >> 24)       ] * x0.w
                + (float)kv_iq4[v1 & 0xFF]         * x1.x
                + (float)kv_iq4[(v1 >>  8) & 0xFF] * x1.y
                + (float)kv_iq4[(v1 >> 16) & 0xFF] * x1.z
                + (float)kv_iq4[(v1 >> 24)       ] * x1.w;
        s += dl * t;
    }
    MV_TAIL;
}

// k_mv_iq4_xs: lane-level since 2026-10-08, defined after the lane dots below

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

#include "kernels_iq_grids.cuh"


// The decode primitives (sign expansion, grid-byte magnitudes, the 2-byte
// aligned field reads) live in iq_decode.h, compiled here as device code
// and by the C compiler for tests/test_iq_decode.c.
#include "iq_decode.h"

// IQ2_XXS: 66-byte block, d (fp16) + 8 sub-blocks of 8 bytes: four grid
// indices, then one word of four 7-bit sign indices (bits 0-27) and a 4-bit
// scale (bits 28-31) decoded as (0.5 + scale) / 4.

// ---------------------------------------- codebook i-quants: lane-level dots
//
// The i-quant GEMVs below used to give each lane a whole 256-weight block
// (`for (b = lane; b < nb; b += 32)`), a shape for dense rows of 4096 and
// more. A routed expert row has n_in 2560 (gate/up: 10 of 32 lanes busy) or
// n_ff_exp (down: 4 lanes), so the device expert time of a MoE token was
// mostly idle lanes: Qwen3.8-Flash-Next IQ2_XS decoded at 8.4 tok/s with 28
// of 48 expert layers on the device against IQ3_S's 13.2 with 19 (2026-10-07).
// Here every block is split across the warp: lane L takes sub-block ib = L/4
// and the 8-weight group l = L%4 of it, the natural unit of every codebook
// format (one grid entry and its sign word), so all 32 lanes work on every
// block and the index loads coalesce. The per-lane partial sums are reduced
// by MV_TAIL's warp_sum; the summation order differs from the old kernels,
// which is inside the tolerance these device kernels are gated at (CPU score
// at logit precision, the nine gpu-identity fixtures), not a bit contract.

static __device__ __forceinline__ float iq2xxs_lane(const uchar *blk, const float *xb, int lane) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8;
    unsigned aux = iq_ld32a2(qs + 4);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
    ulong64 g = kiq2xxs_grid[qs[l]];
    unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
    const float *xp = xb + ib * 32 + l * 8;
    float t = 0;
    for (int j = 0; j < 8; j++) t += iq_w8(g, j, signs) * xp[j];
    return db * t;
}

static __device__ __forceinline__ float iq2xs_lane(const uchar *blk, const float *xb, int lane) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *sc = blk + 66;
    float db = d * (0.5f + (float)(l < 2 ? (sc[ib] & 0xF) : (sc[ib] >> 4))) * 0.25f;
    unsigned q = iq_ld16(qs + 2 * l);
    ulong64 g = kiq2xs_grid[q & 511];
    unsigned signs = iq_signs7(q >> 9);
    const float *xp = xb + ib * 32 + l * 8;
    float t = 0;
    for (int j = 0; j < 8; j++) t += iq_w8(g, j, signs) * xp[j];
    return db * t;
}

static __device__ __forceinline__ float iq2s_lane(const uchar *blk, const float *xb, int lane) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 4, *sg = blk + 34 + ib * 4, *qh = blk + 66, *sc = blk + 74;
    float db = d * (0.5f + (float)(l < 2 ? (sc[ib] & 0xF) : (sc[ib] >> 4))) * 0.25f;
    unsigned hb = qh[ib];
    ulong64 g = kiq2s_grid[qs[l] | ((hb << (8 - 2 * l)) & 0x300)];
    unsigned signs = sg[l];
    const float *xp = xb + ib * 32 + l * 8;
    float t = 0;
    for (int j = 0; j < 8; j++) t += iq_w8(g, j, signs) * xp[j];
    return db * t;
}

static __device__ __forceinline__ float iq3xxs_lane(const uchar *blk, const float *xb, int lane) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *ss = blk + 66;
    unsigned aux = iq_ld32a2(ss + 4 * ib);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    unsigned g1 = kiq3xxs_grid[qs[2 * l + 0]];
    unsigned g2 = kiq3xxs_grid[qs[2 * l + 1]];
    unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
    const float *xp = xb + ib * 32 + l * 8;
    float t = 0;
    for (int j = 0; j < 4; j++) {
        t += iq_w4(g1, j, signs) * xp[j];
        t += iq_w4(g2, j, signs >> 4) * xp[j + 4];
    }
    return db * t;
}

static __device__ __forceinline__ float iq3s_lane(const uchar *blk, const float *xb, int lane) {
    int ib = lane >> 2, l = lane & 3, pair = ib >> 1, h = ib & 1;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *qh = blk + 66, *sg = blk + 74 + ib * 4, *sc = blk + 106;
    float db = d * (float)(1 + 2 * (h ? (sc[pair] >> 4) : (sc[pair] & 0xF)));
    unsigned hb = qh[ib];
    unsigned g1 = kiq3s_grid[qs[2 * l + 0] | ((hb << (8 - 2 * l)) & 256)];
    unsigned g2 = kiq3s_grid[qs[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
    unsigned signs = sg[l];
    const float *xp = xb + ib * 32 + l * 8;
    float t = 0;
    for (int j = 0; j < 4; j++) {
        t += iq_w4(g1, j, signs) * xp[j];
        t += iq_w4(g2, j, signs >> 4) * xp[j + 4];
    }
    return db * t;
}

// N-column forms of the lane dots: the lane's grid words, signs and scale are
// decoded once and the batch-1 inner loop runs per column, so column c of a
// width-classed twin emits exactly the FMA sequence the batch-1 kernel emits
// for that column (the batch-1 kernels below call these with nc = 1: one
// source, the same bits, which the speculative verify and the decode
// microbatch depend on).
static __device__ __forceinline__ void iq2xxs_lane_n(const uchar *blk, const float *xb, ulong64 xs, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8;
    unsigned aux = iq_ld32a2(qs + 4);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
    ulong64 g = kiq2xxs_grid[qs[l]];
    unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
    const float *xp0 = xb + ib * 32 + l * 8;
    float w[8];
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) w[j] = iq_w8(g, j, signs);
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const float *xp = xp0 + (ulong64)c * xs;
        float t = 0;
        _Pragma("unroll")
        for (int j = 0; j < 8; j++) t += w[j] * xp[j];
        acc[c] += db * t;
    }
}

static __device__ __forceinline__ void iq2xs_lane_n(const uchar *blk, const float *xb, ulong64 xs, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *sc = blk + 66;
    float db = d * (0.5f + (float)(l < 2 ? (sc[ib] & 0xF) : (sc[ib] >> 4))) * 0.25f;
    unsigned q = iq_ld16(qs + 2 * l);
    ulong64 g = kiq2xs_grid[q & 511];
    unsigned signs = iq_signs7(q >> 9);
    const float *xp0 = xb + ib * 32 + l * 8;
    float w[8];
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) w[j] = iq_w8(g, j, signs);
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const float *xp = xp0 + (ulong64)c * xs;
        float t = 0;
        _Pragma("unroll")
        for (int j = 0; j < 8; j++) t += w[j] * xp[j];
        acc[c] += db * t;
    }
}

static __device__ __forceinline__ void iq2s_lane_n(const uchar *blk, const float *xb, ulong64 xs, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 4, *sg = blk + 34 + ib * 4, *qh = blk + 66, *sc = blk + 74;
    float db = d * (0.5f + (float)(l < 2 ? (sc[ib] & 0xF) : (sc[ib] >> 4))) * 0.25f;
    unsigned hb = qh[ib];
    ulong64 g = kiq2s_grid[qs[l] | ((hb << (8 - 2 * l)) & 0x300)];
    unsigned signs = sg[l];
    const float *xp0 = xb + ib * 32 + l * 8;
    float w[8];
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) w[j] = iq_w8(g, j, signs);
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const float *xp = xp0 + (ulong64)c * xs;
        float t = 0;
        _Pragma("unroll")
        for (int j = 0; j < 8; j++) t += w[j] * xp[j];
        acc[c] += db * t;
    }
}

static __device__ __forceinline__ void iq3xxs_lane_n(const uchar *blk, const float *xb, ulong64 xs, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *ss = blk + 66;
    unsigned aux = iq_ld32a2(ss + 4 * ib);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    unsigned g1 = kiq3xxs_grid[qs[2 * l + 0]];
    unsigned g2 = kiq3xxs_grid[qs[2 * l + 1]];
    unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
    const float *xp0 = xb + ib * 32 + l * 8;
    float w1[4], w2[4];
    _Pragma("unroll")
    for (int j = 0; j < 4; j++) { w1[j] = iq_w4(g1, j, signs); w2[j] = iq_w4(g2, j, signs >> 4); }
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const float *xp = xp0 + (ulong64)c * xs;
        float t = 0;
        _Pragma("unroll")
        for (int j = 0; j < 4; j++) {
            t += w1[j] * xp[j];
            t += w2[j] * xp[j + 4];
        }
        acc[c] += db * t;
    }
}

static __device__ __forceinline__ void iq3s_lane_n(const uchar *blk, const float *xb, ulong64 xs, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3, pair = ib >> 1, h = ib & 1;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *qh = blk + 66, *sg = blk + 74 + ib * 4, *sc = blk + 106;
    float db = d * (float)(1 + 2 * (h ? (sc[pair] >> 4) : (sc[pair] & 0xF)));
    unsigned hb = qh[ib];
    unsigned g1 = kiq3s_grid[qs[2 * l + 0] | ((hb << (8 - 2 * l)) & 256)];
    unsigned g2 = kiq3s_grid[qs[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
    unsigned signs = sg[l];
    const float *xp0 = xb + ib * 32 + l * 8;
    float w1[4], w2[4];
    _Pragma("unroll")
    for (int j = 0; j < 4; j++) { w1[j] = iq_w4(g1, j, signs); w2[j] = iq_w4(g2, j, signs >> 4); }
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const float *xp = xp0 + (ulong64)c * xs;
        float t = 0;
        _Pragma("unroll")
        for (int j = 0; j < 4; j++) {
            t += w1[j] * xp[j];
            t += w2[j] * xp[j + 4];
        }
        acc[c] += db * t;
    }
}

// IQ4_XS on the same lane geometry (it kept the whole-block-per-lane shape
// after the codebook formats moved, 2026-10-08): 136-byte block, d, the high
// scale bits, the low scale nibbles, 128 quant bytes; sub-block ib's 16 bytes
// carry elements 0-15 (low nibbles) and 16-31 (high). Lane L takes sub-block
// L/4, bytes 4*(L%4).. +4: elements 4l..4l+3 and 16+4l..16+4l+3.
static __device__ __forceinline__ void iq4xs_lane_n(const uchar *blk, const float *xb, ulong64 xs, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
    int ls = ((blk[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
    float dl = d * (ls - 32);
    const uchar *q = blk + 8 + ib * 16 + l * 4;
    const float *xp0 = xb + ib * 32 + l * 4;
    float wl[4], wh[4];
    _Pragma("unroll")
    for (int j = 0; j < 4; j++) { wl[j] = (float)kv_iq4[q[j] & 0xF]; wh[j] = (float)kv_iq4[q[j] >> 4]; }
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const float *xp = xp0 + (ulong64)c * xs;
        float t = 0;
        _Pragma("unroll")
        for (int j = 0; j < 4; j++) {
            t += wl[j] * xp[j];
            t += wh[j] * xp[j + 16];
        }
        acc[c] += dl * t;
    }
}

// The twins' GEMV body. Every warp of a block reads
// the same activation, so x is staged in shared memory once per chunk of CH
// 256-weight blocks for the block's rows (4 at batch 1, GEMM_WARPS for a
// twin) instead of once per row from global: the batch-1 kernels re-read the
// whole activation per row, which on the dense Qwen3.8-27B rows held decode
// to ~150 GB/s (2026-10-08). Shared memory holds the same values as global,
// so no output changes. Columns past a.batch stage zeros (a caller's x need
// not be NC columns wide).
// Two rows per warp in the twins too: the batch-1 kernels gained 16% from a
// second row per warp, and a verify tile is the same latency-bound chain per
// column. Qwen3.8-27B IQ3_S + MTP head on the 24 GB slice: --draft-k 2
// 14.73 -> 15.43, k=3 13.62 -> 15.25, k=4 12.27 -> 13.61 tok/s, plain
// unchanged, text identical (2026-10-08). Each row's arithmetic is the
// one-row twin's, so the bits do not move. enc_mv_batch launches
// 8 * IQ_TWR rows per block for these six formats.
#define IQ_TWR 2
#define IQ_LANE_BODY(fnn, BS, NC, CH) \
    __shared__ float xsm[(NC) * (CH) * 256]; \
    unsigned row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * IQ_TWR; \
    unsigned lane = threadIdx.x & 31; \
    int nb = a.n_in / 256; \
    float acc[IQ_TWR][NC]; \
    _Pragma("unroll") for (int r = 0; r < IQ_TWR; r++) _Pragma("unroll") for (int c = 0; c < (NC); c++) acc[r][c] = 0.0f; \
    int nbat = (NC) == 1 ? 1 : a.batch; \
    for (int cs = 0; cs < nb; cs += (CH)) { \
        int cb = nb - cs < (CH) ? nb - cs : (CH); \
        int ce = cb * 256; \
        for (int t = 0; t < (NC); t++) { \
            const float *xg = x + (ulong64)t * a.xs + (ulong64)cs * 256; \
            for (int e = threadIdx.x; e < ce; e += blockDim.x) \
                xsm[t * (CH) * 256 + e] = t < nbat ? xg[e] : 0.0f; \
        } \
        __syncthreads(); \
        for (int bi = 0; bi < cb; bi++) { \
            _Pragma("unroll") \
            for (int r = 0; r < IQ_TWR; r++) \
                if (row0 + r < (unsigned)a.n_out) \
                    fnn(wb + a.w_off + ((ulong64)(row0 + r) * nb + cs + bi) * BS, xsm + bi * 256, (ulong64)(CH) * 256, lane, (NC), acc[r]); \
        } \
        __syncthreads(); \
    } \
    _Pragma("unroll") \
    for (int r = 0; r < IQ_TWR; r++) { \
        unsigned row = row0 + r; \
        if (row < (unsigned)a.n_out) \
            for (int t = 0; t < nbat && t < (NC); t++) { \
                float s_ = warp_sum(acc[r][t]); \
                if (lane == 0) y[(ulong64)t * a.ys + row] = a.has_bias ? s_ + bias[row] : s_; \
            } \
    }

// batch-1: x from global (staging for four rows measured slower: 27B decode
// 11.9 -> 10.0 tok/s); the same values as the twins' staged copy
// Two rows per warp: the batch-1 i-quant GEMV is latency-bound (a chain of
// small dependent loads per block: index bytes, then the grid word, then x),
// and a second independent row in flight per warp hides part of it. The rows
// share x through L1; each row's arithmetic is the one-row kernel's, so the
// output bits are unchanged. Qwen3.8-27B IQ3_S on the 24 GB slice: decode
// 11.96 -> 13.84 tok/s, text identical; four rows per warp 11.57 (register
// pressure), two rows at eight warps per block 13.72 (2026-10-08).
// enc_mv launches these kernels with 4 * IQ_ROWS rows per 128-thread block.
#define IQ_ROWS 2
#define IQ_LANE_MV1(fnn, BS) \
    unsigned row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * IQ_ROWS; \
    unsigned lane = threadIdx.x & 31; \
    if (row0 >= (unsigned)a.n_out) return; \
    int nb = a.n_in / 256; \
    float acc[IQ_ROWS] = {0}; \
    for (int b = 0; b < nb; b++) { \
        _Pragma("unroll") \
        for (int r = 0; r < IQ_ROWS; r++) \
            if (row0 + r < (unsigned)a.n_out) \
                fnn(wb + a.w_off + ((ulong64)(row0 + r) * nb + b) * BS, x + b * 256, 0, lane, 1, acc + r); \
    } \
    _Pragma("unroll") \
    for (int r = 0; r < IQ_ROWS; r++) { \
        float s = warp_sum(acc[r]); \
        unsigned row = row0 + r; \
        if (lane == 0 && row < (unsigned)a.n_out) y[row] = a.has_bias ? s + bias[row] : s; \
    }

// width-classed twins (launched GEMM_WARPS rows per block like the GEMVB family)
#define IQ_LANE_MVB(NAME, fnn, BS, NC) \
extern "C" __global__ void NAME(MV_PARAMS) { IQ_LANE_BODY(fnn, BS, NC, 32 / (NC)) }

#define IQ_LANE_MV(fn, BS) \
    MV_HEAD; \
    int nb = a.n_in / 256; \
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * BS; \
    float s = 0; \
    for (int b = 0; b < nb; b++) s += fn(rw + (ulong64)b * BS, x + b * 256, lane); \
    MV_TAIL;

#define IQ_LANE_MOE_MV(fn, BS) \
    MOE_MV_HEAD; \
    int nb = a.n_in / 256; \
    const uchar *rw = wbase + (ulong64)row * nb * BS; \
    for (int b = 0; b < nb; b++) s += fn(rw + (ulong64)b * BS, x + b * 256, lane); \
    MOE_MV_TAIL;

extern "C" __global__ void k_mv_iq2_xxs(MV_PARAMS) {
    IQ_LANE_MV1(iq2xxs_lane_n, 66)
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
            unsigned aux = iq_ld32a2(qs + 4);
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
    IQ_LANE_MV1(iq2xs_lane_n, 74)
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
                unsigned q = iq_ld16(qs + 2 * l);
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
    IQ_LANE_MV1(iq2s_lane_n, 82)
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
    IQ_LANE_MV1(iq3xxs_lane_n, 98)
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
            unsigned aux = iq_ld32a2(ss + 4 * ib);
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
    IQ_LANE_MV1(iq3s_lane_n, 110)
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
    LANE_DENSE_HEAD(256)
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 50;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 256)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (256 + 1);
            const uchar *blk = rw + (ulong64)b * 50;
            float d = f16f(blk);
            const uchar *qs = blk + 2, *qh = blk + 34;
            const float *xp = xs_;
            for (int ib = 0; ib < 8; ib++, qs += 4) {
                unsigned h = iq_ld16(qh + 2 * ib);
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
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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
            unsigned h = iq_ld16(qh + 2 * ib);
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
    LANE_DENSE_HEAD(256)
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 56;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 256)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (256 + 1);
            const uchar *blk = rw + (ulong64)b * 56;
            const uchar *qs = blk, *qh = blk + 32, *scb = blk + 48;
            unsigned sc0 = iq_ld16(scb), sc1 = iq_ld16(scb + 2), sc2 = iq_ld16(scb + 4), sc3 = iq_ld16(scb + 6);
            unsigned short sd = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
                                ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
            float d = __half2float(__ushort_as_half(sd));
            const float *xp = xs_;
            for (int ib = 0; ib < 8; ib++, qs += 4, qh += 2) {
                unsigned sw = iq_ld16(scb + 2 * (ib >> 1)) >> (6 * (ib & 1));
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
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
}

extern "C" __global__ void k_mv_iq1_m_b(MV_PARAMS) {
    MV_HEAD_B;
    int nb = a.n_in / 256;
    const uchar *rw = wb + a.w_off + (ulong64)row * nb * 56;
    for (int b = lane; b < nb; b += 32) {
        const uchar *blk = rw + (ulong64)b * 56;
        const uchar *qs = blk, *qh = blk + 32, *scb = blk + 48;
        unsigned sc0 = iq_ld16(scb), sc1 = iq_ld16(scb + 2), sc2 = iq_ld16(scb + 4), sc3 = iq_ld16(scb + 6);
        unsigned short sd = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
                            ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
        float d = __half2float(__ushort_as_half(sd));
        ulong64 base = (ulong64)b * 256;
        for (int ib = 0; ib < 8; ib++, qs += 4, qh += 2) {
            unsigned sw = iq_ld16(scb + 2 * (ib >> 1)) >> (6 * (ib & 1));
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

// ------------------------------------ tensor-core GEMMs for the codebook types
// The TC_GEMM_BLK shape with a 256-element block: the 64-row x 128-K fp16
// weight tile is staged two segments per row per K-step, each segment
// decoded by the stager in iq_decode.h that the host test holds to
// dequant_row() bit for bit, then rounded to fp16 for the MMA operand (the
// same numeric class as the Q4_K/Q6_K/Q8_0/Q4_0 tensor-core kernels: exact
// per-element weight values, fp16 operands, fp32 accumulation). Registered
// in cuda.c's TC_KERNELS and promoted with the codebook family on
// 2026-09-14 (tc_promoted); RUNNER_CUDA_TC=0 pins the scalar path.
#include "kernels_iq_stage.cuh"
TC_GEMM_BLK_X(k_gemm_iq3_s_tc, tc_stage_iq3_s, 110, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq3_xxs_tc, tc_stage_iq3_xxs, 98, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq2_s_tc, tc_stage_iq2_s, 82, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq2_xs_tc, tc_stage_iq2_xs, 74, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq2_xxs_tc, tc_stage_iq2_xxs, 66, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq1_s_tc, tc_stage_iq1_s, 50, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq1_m_tc, tc_stage_iq1_m, 56, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq4_xs_tc, tc_stage_iq4_xs, 136, 256, 1)
TC_GEMM_BLK_X(k_gemm_iq4_nl_tc, tc_stage_iq4_nl, 18, 32, 1)

// MXFP4 (gpt-oss expert tensors): 17-byte block = one E8M0 scale byte (a
// biased power-of-two exponent, 2^(e-127), NOT an fp16) + 32 packed E2M1
// nibbles indexing a fixed signed codebook. Table and decode are 1:1 with
// dq_mxfp4 in quants.c — ldexpf keeps 2^(e-127) exact down into the
// subnormal range, where exp2f of a float could flush to zero.
static __device__ const float kv_mxfp4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// activations staged as k_mv_q4_0's (LANE_XPAD), the codebook in shared
// memory instead of a global load per nibble
extern "C" __global__ void k_mv_mxfp4(MV_PARAMS) {
    __shared__ float xsm[32 * LANE_XPAD];
    __shared__ float kv[16];
    if (threadIdx.x < 16) kv[threadIdx.x] = kv_mxfp4[threadIdx.x];
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    unsigned lane = threadIdx.x & 31;
    bool live = row < (unsigned)a.n_out;
    int nb = a.n_in / 32;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 17;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_X(xsm, x, b0, nblk)
        __syncthreads();
        if (live && (int)lane < nblk) {
            const uchar *blk = rw + (ulong64)(b0 + lane) * 17;
            float d = ldexpf(1.0f, (int)blk[0] - 127);
            const uchar *q = blk + 1;
            const float *xp = xsm + lane * LANE_XPAD;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                t += kv[q[j] & 0xF] * xp[j];
                t += kv[q[j] >> 4]  * xp[j + 16];
            }
            s += d * t;
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s);
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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
    __shared__ float kv[16];
    if (threadIdx.x < 16) kv[threadIdx.x] = kv_mxfp4[threadIdx.x];
    LANE_DENSE_HEAD(64)
    int nb = a.n_in / 64;
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 36;
    float s = 0;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 64)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (64 + 1);
            const uchar *blk = rw + (ulong64)b * 36;
            for (int sub = 0; sub < 4; sub++) {
                float d = ue4m3f(blk[sub]);
                const uchar *q = blk + 4 + sub * 8;
                const float *xp = xs_ + sub * 16;
                float t = 0;
                for (int j = 0; j < 8; j++) {
                    t += kv[q[j] & 0xF] * xp[j];
                    t += kv[q[j] >> 4]  * xp[j + 8];
                }
                s += d * t;
            }
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s) * scale;
        if (lane == 0) y[row] = a.has_bias ? s + bias[row] : s;
    }
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

// The `q8` argument of every KV helper is the cache KIND: 0 fp16, 1 q8_0,
// 2 fp4 (16 values per 9-byte block: one UE4M3 scale byte + 16 E2M1
// nibbles, the --kv fp4 cache). The fp4 encoder repeats quants.c's
// fp4_quant_row step for step (UE4M3 ceiling of amax/6, nearest E2M1 with
// ties up, sign in bit 3, element j in the low nibble of byte j and element
// j+8 in its high nibble), so a row quantised here is byte-identical to the
// CPU's and to Metal's. An fp4 row is only byte-aligned.
__device__ __forceinline__ unsigned kv_ue4m3_ceil(float s) {
    if (!(s > 0.0f)) return 0;
    if (s >= 448.0f) return 0x7E;
    int e;
    float f = frexpf(s, &e);
    int E = e - 1 + 7;
    if (E >= 1) {
        int mn = (int)ceilf((f * 2.0f - 1.0f) * 8.0f);
        if (mn == 8) { mn = 0; E++; }
        if (E > 15 || (E == 15 && mn == 7)) return 0x7E;
        return (unsigned)((E << 3) | mn);
    }
    int mn = (int)ceilf(s * 512.0f);
    if (mn >= 8) return 0x08;
    return (unsigned)mn;
}
__device__ __forceinline__ unsigned kv_e2m1_code(float v) {
    float a = fabsf(v);
    unsigned i = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3
               : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.0f ? 6 : 7;
    return (v < 0.0f && i) ? (i | 8) : i;
}

#define KV_ROW_BYTES(kv_dim, q8) \
    ((q8) == 2 ? (ulong64)((kv_dim) / 16) * 9 : \
     (q8) ? (ulong64)((kv_dim) / 32) * 34 : (ulong64)(kv_dim) * 2)

__device__ __forceinline__ void kv_store_row(unsigned char *cache,
                                             const float *src, int kv_dim,
                                             int q8, int i) {
    if (q8 == 2) {
        unsigned char *blk = cache + (ulong64)i * 9;
        const float *x = src + i * 16;
        float amax = 0;
        for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(x[j]));
        unsigned sc = kv_ue4m3_ceil(amax / 6.0f);
        float d = ue4m3f((uchar)sc);
        float id = d > 0 ? 1.0f / d : 0.0f;
        blk[0] = (unsigned char)sc;
        for (int j = 0; j < 8; j++)
            blk[1 + j] = (unsigned char)(kv_e2m1_code(x[j] * id) |
                                         (kv_e2m1_code(x[j + 8] * id) << 4));
    } else if (q8) {
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
    if (q8 == 2) {
        for (int b = 0; b < hd / 16; b++) {
            const unsigned char *blk = row + (ulong64)b * 9;
            float d = ue4m3f(blk[0]);
            const float *xp = qh + b * 16;
            float t = 0;
            for (int j = 0; j < 8; j++)
                t += xp[j] * kv_mxfp4[blk[1 + j] & 0xF]
                   + xp[j + 8] * kv_mxfp4[blk[1 + j] >> 4];
            s += d * t;
        }
    } else if (q8) {
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
    if (q8 == 2) {
        const unsigned char *blk = row + (ulong64)(i2 / 8) * 9;
        float d = ue4m3f(blk[0]);
        int j = (2 * i2) & 15;
        unsigned c0 = j < 8 ? (blk[1 + j] & 0xF) : (blk[1 + j - 8] >> 4);
        unsigned c1 = j + 1 < 8 ? (blk[1 + j + 1] & 0xF) : (blk[1 + j + 1 - 8] >> 4);
        return make_float2(d * kv_mxfp4[c0], d * kv_mxfp4[c1]);
    }
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

// q8 / l_off describe the K cache, vq8 / v_off the V cache: the two differ
// under --kv k8v4 (K q8_0, V fp4), so each side computes its own row and
// unit count. The grid covers max(units_k, units_v) threads.
__device__ __forceinline__ int kv_units(int kv_dim, int q8) {
    return q8 == 2 ? kv_dim / 16 : q8 ? kv_dim / 32 : kv_dim;
}

extern "C" __global__ void k_store_kv(const float *k, const float *v,
                                      unsigned char *kc, unsigned char *vc,
                                      int kv_dim, ulong64 l_off,
                                      const int *posp, int q8, int ring,
                                      ulong64 v_off, int vq8) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    ulong64 slot = kv_slot(*posp + blockIdx.y, ring);
    const float *ks = k + (ulong64)blockIdx.y * kv_dim;
    const float *vs = v + (ulong64)blockIdx.y * kv_dim;
    if (i < kv_units(kv_dim, q8))
        kv_store_row(kc + l_off + slot * KV_ROW_BYTES(kv_dim, q8), ks, kv_dim, q8, i);
    if (i < kv_units(kv_dim, vq8))
        kv_store_row(vc + v_off + slot * KV_ROW_BYTES(kv_dim, vq8), vs, kv_dim, vq8, i);
}

// ---------------------------------------------------------------- attention
// One block per (head, token): scores -> softmax -> weighted value sum.
// att scratch is MVB planes of [n_head][n_ctx].


// byte offset of head kvh's slice within a cache row
__device__ __forceinline__ ulong64 kv_head_off(int kvh, int hd, int q8) {
    return q8 == 2 ? (ulong64)(kvh * hd / 16) * 9
         : q8      ? (ulong64)(kvh * hd / 32) * 34 : (ulong64)(kvh * hd) * 2;
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
    ulong64 row_v = KV_ROW_BYTES(kv_dim, a.vq8);
    ulong64 base_v = a.v_off + kv_head_off(kvh, hd, a.vq8);
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
            float2 vf = kv_pair(vc + base_v + kv_slot(t, a.ring) * row_v, vlane, a.vq8);
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
                float2 vf = kv_pair(vc + base_v + kv_slot(t, a.ring) * row_v, i2, a.vq8);
                o0 += ah[t] * vf.x;
                o1 += ah[t] * vf.y;
            }
            out[(ulong64)tk * a.os + h * hd + 2 * i2]     = o0 / sum;
            out[(ulong64)tk * a.os + h * hd + 2 * i2 + 1] = o1 / sum;
        }
    }
}

// ------------------------------------------------ tiled prefill attention
// C2 step 1 (RUNNER_CUDA_ATTN_TILE, see tile_attn in cuda.c). k_attn above
// runs one block per (head, token) and reads every K and V row of the
// context from global memory for each of them: a prompt of L tokens on a
// model with G query heads per KV head reads the cache G * L times, and the
// scores make a round trip through global scratch. This kernel gives one
// block a KV head, up to R / G of its query heads' rows (or R heads at one
// token when G >= R) and a strip of query tokens, and walks the strip's key
// range in tiles of TK rows staged in shared memory, so each cache row is
// read once per R query rows.
//
// The tiling follows the design of the published flash-attention kernels
// (tiles of keys staged on chip, every query row of the block reusing them);
// the arithmetic is k_attn's own, kept to the same two passes:
//   pass 1: every score q.k * scale, through the unchanged kv_dot on the
//           staged row (the cache's own bytes, so q8_0 and fp4 rows keep
//           their per-block association), and each row's maximum;
//   pass 2: the same scores recomputed by the same call, expf(s - max),
//           their sum, and the weighted V rows through kv_pair.
// Scores and maxima are therefore bit-identical to k_attn's; only the order
// of the softmax denominator's sum and of the V accumulation differ, so the
// obligation is k_attn's: token-identical to the CPU scalar route, logits
// not bit-identical, gated empirically (test-gpu-identity, cpu_cuda_check).
// Keys a row may not see (causal, sliding window) contribute exp = 0, which
// leaves an fp32 sum unchanged; rows past the strip are zero.
//
// Shared memory: q rows [R][hd+1] floats, one K or V tile [TK][bws] words
// (row bytes rounded up to an odd word count so lanes reading different rows
// hit different banks), the probabilities [R][TK] and the row max and sum.
// Rows are r = hi * qs + ti (query head hi of the block, token ti of the
// strip). Launch: 256 threads, grid (n_kv * ngh, ceil(tn / qs)).
#define AT_THREADS 256
template <int R, int TK>
__device__ __forceinline__ void attn_tile_body(const float *q, const unsigned char *kc,
                                               const unsigned char *vc, float *out,
                                               const attn_args &a, const int *posp,
                                               const float *sinks, int tn, int hb,
                                               int qs) {
    extern __shared__ float at_sm[];
    const int hd = a.head_dim, tid = threadIdx.x;
    const int G = a.n_head / a.n_head_kv;
    const int ngh = (G + hb - 1) / hb;
    const int kvh = blockIdx.x / ngh, hg = blockIdx.x % ngh;
    const int h0 = kvh * G + hg * hb;
    const int nh = G - hg * hb < hb ? G - hg * hb : hb;
    const int tk0 = blockIdx.y * qs;
    const int nt = tn - tk0 < qs ? tn - tk0 : qs;
    const int pos0 = *posp;
    const int kv_dim = a.n_head_kv * hd;
    const ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8), row_v = KV_ROW_BYTES(kv_dim, a.vq8);
    const ulong64 base = a.l_off + kv_head_off(kvh, hd, a.q8);
    const ulong64 base_v = a.v_off + kv_head_off(kvh, hd, a.vq8);
    const int kbytes = (int)KV_ROW_BYTES(hd, a.q8), vbytes = (int)KV_ROW_BYTES(hd, a.vq8);
    const int kws = ((kbytes + 3) / 4) | 1, vws = ((vbytes + 3) / 4) | 1;
    const int bws = kws > vws ? kws : vws;
    const int qst = hd + 1;
    float *Q = at_sm;                                 // [R][qst]
    unsigned *KV = (unsigned *)(Q + R * qst);         // [TK][bws]
    float *P = (float *)(KV + TK * bws);              // [R][TK]
    float *rmax = P + R * TK;                         // [R]
    float *rsum = rmax + R;                           // [R]

    for (int i = tid; i < R * hd; i += AT_THREADS) {
        int r = i / hd, d = i - r * hd;
        int hi = r / qs, ti = r - hi * qs;
        float v = 0.0f;
        if (hi < nh && ti < nt) v = q[(ulong64)(tk0 + ti) * a.qs + (ulong64)(h0 + hi) * hd + d];
        Q[r * qst + d] = v;
    }
    // the strip's key range: the first row's window start to the last row
    const int plast = pos0 + tk0 + nt - 1;
    int klo = 0;
    if (a.window > 0) {
        int f = pos0 + tk0 - a.window + 1;
        klo = f > 0 ? f : 0;
    }
    enum { DPT = (R * TK + AT_THREADS - 1) / AT_THREADS };      // scores per thread
    float mx[DPT];
    #pragma unroll
    for (int j = 0; j < DPT; j++) mx[j] = -1e30f;

    // stage the cache rows [kt, kt + TK) of one side (K or V) as raw bytes
    auto stage = [&](const unsigned char *cache, ulong64 b0, ulong64 rb, int nbytes,
                     int ws, int kt) {
        // whole words when the head slice is word aligned and a whole number
        // of words long (fp16 always is; a q8_0 row at head width 96 or an
        // fp4 row at 80 is not, and its tail bytes would go unstaged)
        if (((b0 | rb | (ulong64)nbytes) & 3) == 0) {
            int nw = nbytes / 4;
            for (int i = tid; i < TK * nw; i += AT_THREADS) {
                int k = i / nw, w = i - k * nw, t = kt + k;
                unsigned v = 0u;
                if (t <= plast)
                    v = *(const unsigned *)(cache + b0 + kv_slot(t, a.ring) * rb + (ulong64)w * 4);
                KV[k * ws + w] = v;
            }
        } else {
            for (int i = tid; i < TK * nbytes; i += AT_THREADS) {
                int k = i / nbytes, c = i - k * nbytes, t = kt + k;
                unsigned char v = 0;
                if (t <= plast) v = cache[b0 + kv_slot(t, a.ring) * rb + (ulong64)c];
                ((unsigned char *)(KV + k * ws))[c] = v;
            }
        }
    };

    // ---- pass 1: row maxima ----
    for (int kt = klo; kt <= plast; kt += TK) {
        __syncthreads();
        stage(kc, base, row_b, kbytes, bws, kt);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < DPT; j++) {
            int idx = tid + j * AT_THREADS;
            if (idx >= R * TK) break;
            int r = idx / TK, k = idx - r * TK, t = kt + k;
            int hi = r / qs, ti = r - hi * qs;
            int pr = pos0 + tk0 + ti;
            int t0r = (a.window > 0 && pr - a.window + 1 > 0) ? pr - a.window + 1 : 0;
            if (hi < nh && ti < nt && t >= t0r && t <= pr) {
                float sc = kv_dot((const unsigned char *)(KV + k * bws), Q + r * qst, hd, a.q8) * a.scale;
                mx[j] = fmaxf(mx[j], sc);
            }
        }
    }
    // rows share a lane group of TK consecutive lanes (TK divides 32)
    #pragma unroll
    for (int j = 0; j < DPT; j++) {
        float v = mx[j];
        for (int off = TK / 2; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        int idx = tid + j * AT_THREADS;
        if (idx < R * TK && idx % TK == 0) {
            int r = idx / TK, hi = r / qs;
            if (sinks && hi < nh && sinks[h0 + hi] > v) v = sinks[h0 + hi];
            rmax[r] = v;
        }
    }

    // ---- pass 2: probabilities, their sum and the weighted values ----
    // (row, value pair) outputs per thread: the host launches only when
    // R * head_dim <= 4096 (32 rows at <= 128, 16 rows at <= 256)
    enum { OPT = 4096 / 2 / AT_THREADS };
    const int P2 = hd / 2;
    float2 acc[OPT];
    #pragma unroll
    for (int j = 0; j < OPT; j++) acc[j] = make_float2(0.0f, 0.0f);
    float sm_[DPT];
    #pragma unroll
    for (int j = 0; j < DPT; j++) sm_[j] = 0.0f;
    for (int kt = klo; kt <= plast; kt += TK) {
        __syncthreads();
        stage(kc, base, row_b, kbytes, bws, kt);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < DPT; j++) {
            int idx = tid + j * AT_THREADS;
            if (idx >= R * TK) break;
            int r = idx / TK, k = idx - r * TK, t = kt + k;
            int hi = r / qs, ti = r - hi * qs;
            int pr = pos0 + tk0 + ti;
            int t0r = (a.window > 0 && pr - a.window + 1 > 0) ? pr - a.window + 1 : 0;
            float pv = 0.0f;
            if (hi < nh && ti < nt && t >= t0r && t <= pr) {
                float sc = kv_dot((const unsigned char *)(KV + k * bws), Q + r * qst, hd, a.q8) * a.scale;
                pv = expf(sc - rmax[r]);
            }
            sm_[j] += pv;
            P[r * TK + k] = pv;
        }
        __syncthreads();
        stage(vc, base_v, row_v, vbytes, bws, kt);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < OPT; j++) {
            int idx = tid + j * AT_THREADS;
            if (idx >= R * P2) break;
            int r = idx / P2, i2 = idx - r * P2;
            float2 o = acc[j];
            for (int k = 0; k < TK; k++) {
                float pk = P[r * TK + k];
                float2 vf = kv_pair((const unsigned char *)(KV + k * bws), i2, a.vq8);
                o.x += pk * vf.x;
                o.y += pk * vf.y;
            }
            acc[j] = o;
        }
    }
    #pragma unroll
    for (int j = 0; j < DPT; j++) {
        float v = sm_[j];
        for (int off = TK / 2; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        int idx = tid + j * AT_THREADS;
        if (idx < R * TK && idx % TK == 0) {
            int r = idx / TK, hi = r / qs;
            if (sinks && hi < nh) v += expf(sinks[h0 + hi] - rmax[r]);
            rsum[r] = v;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < OPT; j++) {
        int idx = tid + j * AT_THREADS;
        if (idx >= R * P2) break;
        int r = idx / P2, i2 = idx - r * P2;
        int hi = r / qs, ti = r - hi * qs;
        if (hi >= nh || ti >= nt) continue;
        float s = rsum[r];
        ulong64 o = (ulong64)(tk0 + ti) * a.os + (ulong64)(h0 + hi) * hd + 2 * i2;
        out[o]     = acc[j].x / s;
        out[o + 1] = acc[j].y / s;
    }
}

// head_dim <= 128: 32 query rows, 32-key tiles
extern "C" __global__ void __launch_bounds__(AT_THREADS)
k_attn_tile32(const float *q, const unsigned char *kc, const unsigned char *vc,
              float *out, attn_args a, const int *posp, const float *sinks,
              int tn, int hb, int qs) {
    attn_tile_body<32, 32>(q, kc, vc, out, a, posp, sinks, tn, hb, qs);
}
// head_dim <= 256: 16 query rows, 16-key tiles (the same shared footprint)
extern "C" __global__ void __launch_bounds__(AT_THREADS)
k_attn_tile16(const float *q, const unsigned char *kc, const unsigned char *vc,
              float *out, attn_args a, const int *posp, const float *sinks,
              int tn, int hb, int qs) {
    attn_tile_body<16, 16>(q, kc, vc, out, a, posp, sinks, tn, hb, qs);
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
    ulong64 row_v = KV_ROW_BYTES(kv_dim, a.vq8);
    ulong64 base_v = a.v_off + kv_head_off(kvh, hd, a.vq8);
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
            float2 vf = kv_pair(vc + base_v + kv_slot(t, a.ring) * row_v, i2, a.vq8);
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
                                          const int *posp, int q8, int ring,
                                          ulong64 v_off, int vq8) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sq = blockIdx.y;
    ulong64 slot = kv_slot(posp[sq], ring);
    const float *ks = k + (ulong64)sq * kv_dim;
    const float *vs = v + (ulong64)sq * kv_dim;
    if (i < kv_units(kv_dim, q8))
        kv_store_row((unsigned char *)kcp[sq] + l_off + slot * KV_ROW_BYTES(kv_dim, q8),
                     ks, kv_dim, q8, i);
    if (i < kv_units(kv_dim, vq8))
        kv_store_row((unsigned char *)vcp[sq] + v_off + slot * KV_ROW_BYTES(kv_dim, vq8),
                     vs, kv_dim, vq8, i);
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
    ulong64 row_v = KV_ROW_BYTES(kv_dim, a.vq8);
    ulong64 base_v = a.v_off + kv_head_off(kvh, hd, a.vq8);
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
            float2 vf = kv_pair(vc + base_v + kv_slot(t, a.ring) * row_v, i2, a.vq8);
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
#define GEMVB_Q8_0_C(NAME, NC, CH)                                             \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][(CH) * SMPAD];                                    \
    enum { R = DQ_TWR(NC) };                                                   \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row0 = (blockIdx.x * GEMM_WARPS + warp) * R;                      \
    int nb = a.n_in / 32;                                                      \
    int nc = a.batch < (NC) ? a.batch : (NC);                                  \
    const uchar *rw[R];                                                        \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        rw[r] = wb + a.w_off +                                                 \
                (ulong64)(row0 + r < (unsigned)a.n_out ? row0 + r : 0) * nb * 34; \
    int bsub = (int)(lane >> 3);                                               \
    int boff = ((int)lane & 7) * 4;                                            \
    float s[R][NC];                                                            \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        _Pragma("unroll") for (int t = 0; t < NC; t++) s[r][t] = 0.0f;         \
    for (int cs = 0; cs < nb; cs += (CH)) {                                    \
        int cblocks = nb - cs < (CH) ? nb - cs : (CH);                         \
        int celems  = cblocks * 32;                                            \
        int base_e  = cs * 32;                                                 \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        /* chunks are CH-aligned and CH is a multiple of 4, so the batch-1     \
           kernel's b4 = nb & ~3 boundary falls inside the LAST chunk at       \
           exactly cblocks & ~3: every lane sees the same four-block trips,    \
           then the same tail, in that order */                                \
        int c4 = cblocks & ~3;                                                 \
        for (int bi = 0; bi < c4; bi += 4) {                                   \
            float d[R], q[R][4];                                               \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < R; r++) {                                      \
                const uchar *blk = rw[r] + (ulong64)(cs + bi + bsub) * 34;     \
                d[r] = f16f(blk);                                              \
                const uchar *qp = blk + 2 + boff;                              \
                ushort16 u0 = *(const ushort16 *)qp,                           \
                         u1 = *(const ushort16 *)(qp + 2);                     \
                q[r][0] = (float)(int)(signed char)(u0 & 0xFF);                \
                q[r][1] = (float)(int)(signed char)(u0 >> 8);                  \
                q[r][2] = (float)(int)(signed char)(u1 & 0xFF);                \
                q[r][3] = (float)(int)(signed char)(u1 >> 8);                  \
            }                                                                  \
            int xo = (bi + bsub) * SMPAD + boff;                               \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                if (t >= nc) continue;                                         \
                const float *xp = xsm[t] + xo;                                 \
                float x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];          \
                _Pragma("unroll")                                              \
                for (int r = 0; r < R; r++)                                    \
                    s[r][t] += d[r] * (q[r][0] * x0 + q[r][1] * x1 +           \
                                       q[r][2] * x2 + q[r][3] * x3);           \
            }                                                                  \
        }                                                                      \
        for (int bi = c4; bi < cblocks; bi++) {                                \
            float d[R], qv[R];                                                 \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < R; r++) {                                      \
                const uchar *blk = rw[r] + (ulong64)(cs + bi) * 34;            \
                d[r] = f16f(blk);                                              \
                qv[r] = (float)((const signed char *)(blk + 2))[lane];         \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                if (t >= nc) continue;                                         \
                float xv = xsm[t][bi * SMPAD + lane];                          \
                _Pragma("unroll")                                              \
                for (int r = 0; r < R; r++) s[r][t] += d[r] * (qv[r] * xv);    \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < R; r++) {                                              \
        unsigned row = row0 + r;                                               \
        if (row >= (unsigned)a.n_out) continue;                                \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            float v = warp_sum(s[r][t]);                                       \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? v + bias[row] : v;                                \
        }                                                                      \
    }                                                                          \
}

// ---- Q4_0: k_gemv_q4_0's twin (2026-08-19). Q4_0 had no width-classed twin
// at all, so enc_mv_batch fell through to f_mvb: a different reduction AND no
// x staging, measured at 0.11x of sequential decode. Same four-blocks-in-
// flight shape as Q8_0; a lane's two quant bytes carry elements boff, boff+1
// (low nibbles) and boff+16, boff+17 (high nibbles).
#define GEMVB_Q4_0_C(NAME, NC, CH)                                             \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][(CH) * SMPAD];                                    \
    enum { R = DQ_TWR(NC) };                                                   \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row0 = (blockIdx.x * GEMM_WARPS + warp) * R;                      \
    int nb = a.n_in / 32;                                                      \
    int nc = a.batch < (NC) ? a.batch : (NC);                                  \
    const uchar *rw[R];                                                        \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        rw[r] = wb + a.w_off +                                                 \
                (ulong64)(row0 + r < (unsigned)a.n_out ? row0 + r : 0) * nb * 18; \
    int bsub = (int)(lane >> 3);                                               \
    int boff = ((int)lane & 7) * 2;                                            \
    float s[R][NC];                                                            \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        _Pragma("unroll") for (int t = 0; t < NC; t++) s[r][t] = 0.0f;         \
    for (int cs = 0; cs < nb; cs += (CH)) {                                    \
        int cblocks = nb - cs < (CH) ? nb - cs : (CH);                         \
        int celems  = cblocks * 32;                                            \
        int base_e  = cs * 32;                                                 \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][(e >> 5) * SMPAD + (e & 31)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        int c4 = cblocks & ~3;                                                 \
        for (int bi = 0; bi < c4; bi += 4) {                                   \
            float d[R], q[R][4];                                               \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < R; r++) {                                      \
                const uchar *blk = rw[r] + (ulong64)(cs + bi + bsub) * 18;     \
                d[r] = f16f(blk);                                              \
                ushort16 u = *(const ushort16 *)(blk + 2 + boff);              \
                q[r][0] = (float)((int)( u        & 0xF) - 8);                 \
                q[r][1] = (float)((int)((u >>  4) & 0xF) - 8);                 \
                q[r][2] = (float)((int)((u >>  8) & 0xF) - 8);                 \
                q[r][3] = (float)((int)((u >> 12) & 0xF) - 8);                 \
            }                                                                  \
            int xo = (bi + bsub) * SMPAD + boff;                               \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                if (t >= nc) continue;                                         \
                const float *xp = xsm[t] + xo;                                 \
                float x0 = xp[0], x1 = xp[1], x16 = xp[16], x17 = xp[17];      \
                _Pragma("unroll")                                              \
                for (int r = 0; r < R; r++)                                    \
                    s[r][t] += d[r] * (q[r][0] * x0 + q[r][2] * x1 +           \
                                       q[r][1] * x16 + q[r][3] * x17);         \
            }                                                                  \
        }                                                                      \
        for (int bi = c4; bi < cblocks; bi++) {                                \
            float w[R];                                                        \
            int j = (int)lane & 15, hi = (int)lane >> 4;                       \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < R; r++) {                                      \
                const uchar *blk = rw[r] + (ulong64)(cs + bi) * 18;            \
                float d = f16f(blk);                                           \
                const uchar *q = blk + 2;                                      \
                int qv = hi ? (q[j] >> 4) : (q[j] & 0xF);                      \
                w[r] = d * (float)(qv - 8);                                    \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                if (t >= nc) continue;                                         \
                float xv = xsm[t][bi * SMPAD + lane];                          \
                _Pragma("unroll")                                              \
                for (int r = 0; r < R; r++) s[r][t] += w[r] * xv;              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < R; r++) {                                              \
        unsigned row = row0 + r;                                               \
        if (row >= (unsigned)a.n_out) continue;                                \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            float v = warp_sum(s[r][t]);                                       \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? v + bias[row] : v;                                \
        }                                                                      \
    }                                                                          \
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
// Q5_K: k_gemv_q5_K v2, i.e. Q4_K's factored form with the fifth bit — bit g
// of each qh byte, added as 16 before the conversion to float.
//
// R rows a warp (KQ_TWR, kernel_args.h; 2026-10-21). A block's staged x and
// each column's Sum(x) are the same for every row, so a warp reads a lane's
// eight x values and forms their sum ONCE per column and applies them to its
// R rows. With one row a warp they were re-read and re-summed per row, and a
// speculative verify pass over five columns cost about six batch-1 passes
// (RTX 3070, Qwen2.5-7B Q4_K_M: 318 us a launch against 53 for k_gemv_q4_K).
// Columns past a.batch are skipped rather than computed on stale x. Every
// (row, column) keeps k_gemv_q4_K's (k_gemv_q5_K's) expressions in their
// order, so its bits.
#define KG_NIBS(n, v0, v1)                                                     \
    n[0] = (float)(v0 & 0xFF);         n[1] = (float)((v0 >>  8) & 0xFF);       \
    n[2] = (float)((v0 >> 16) & 0xFF); n[3] = (float)((v0 >> 24));              \
    n[4] = (float)(v1 & 0xFF);         n[5] = (float)((v1 >>  8) & 0xFF);       \
    n[6] = (float)((v1 >> 16) & 0xFF); n[7] = (float)((v1 >> 24));
#define GEMVB_KQ(NAME, NC, BB, QOFF, HBIT)                                     \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][KG8ROW];                                          \
    enum { R = KQ_TWR(NC) };                                                   \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row0 = (blockIdx.x * GEMM_WARPS + warp) * R;                      \
    int nb = a.n_in / 256;                                                     \
    int nc = a.batch < (NC) ? a.batch : (NC);                                  \
    float s[R][NC];                                                            \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        _Pragma("unroll") for (int t = 0; t < NC; t++) s[r][t] = 0.0f;         \
    int g     = (int)(lane >> 2);                                              \
    int ji    = (int)(lane >> 3);                                              \
    int sh    = ((((int)lane >> 2) & 1) == 0) ? 0 : 4;                         \
    int bbase = ((int)lane & 3) * 8;                                           \
    int el    = (int)lane * KG8PAD;                                            \
    for (int b = 0; b < nb; b++) {                                             \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][(e >> 3) * KG8PAD + (e & 7)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        float dg[R], mmg[R], n[R][8];                                          \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < R; r++) {                                          \
            unsigned row = row0 + r;                                           \
            const uchar *blk = wb + a.w_off +                                  \
                (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * (BB) +     \
                (ulong64)b * (BB);                                             \
            float dd   = f16f(blk);                                            \
            float dmin = f16f(blk + 2);                                        \
            uchar sg, mg;                                                      \
            get_scale_min_k4(g, blk + 4, &sg, &mg);                            \
            dg[r] = dd * (float)sg; mmg[r] = dmin * (float)mg;                 \
            uint2 qv = *(const uint2 *)(blk + (QOFF) + ji * 32 + bbase);       \
            uint v0 = (qv.x >> sh) & 0x0F0F0F0Fu;                              \
            uint v1 = (qv.y >> sh) & 0x0F0F0F0Fu;                              \
            if (HBIT) {                                                        \
                uint2 hv = *(const uint2 *)(blk + 16 + bbase);                 \
                v0 += ((hv.x >> g) & 0x01010101u) << 4;                        \
                v1 += ((hv.y >> g) & 0x01010101u) << 4;                        \
            }                                                                  \
            KG_NIBS(n[r], v0, v1)                                              \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xp = xsm[t] + el;                                     \
            float x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];              \
            float x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];              \
            float sx = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;                  \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < R; r++) {                                      \
                float tt = n[r][0] * x0 + n[r][1] * x1 + n[r][2] * x2          \
                         + n[r][3] * x3 + n[r][4] * x4 + n[r][5] * x5          \
                         + n[r][6] * x6 + n[r][7] * x7;                        \
                s[r][t] += dg[r] * tt - mmg[r] * sx;                           \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < R; r++) {                                              \
        unsigned row = row0 + r;                                               \
        if (row >= (unsigned)a.n_out) continue;                                \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            float v = warp_sum(s[r][t]);                                       \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? v + bias[row] : v;                                \
        }                                                                      \
    }                                                                          \
}
#define GEMVB_Q4_K(NAME, NC) GEMVB_KQ(NAME, NC, 144, 16, 0)
#define GEMVB_Q5_K(NAME, NC) GEMVB_KQ(NAME, NC, 176, 48, 1)

// ---- IQ4_XS: k_gemv_iq4_xs's twin (2026-10-21). The batch-1 decode of
// IQ4_XS runs that coalesced GEMV (lane L owns elements 8L..8L+7 of a block:
// sub-block L/4, low or high nibbles of 8 bytes), but its width-classed
// twins were the codebook family's lane kernels, twins of k_mv_iq4_xs, a
// different reduction: a decode microbatch or a verify round on an IQ4_XS
// file differed from single steps in the last bits. Same staging and R rows
// a warp (KQ_TWR) as the K-quant twins; every (row, column) keeps
// k_gemv_iq4_xs's expressions in their order.
#define GEMVB_IQ4XS(NAME, NC)                                                  \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][KG8ROW];                                          \
    enum { R = KQ_TWR(NC) };                                                   \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row0 = (blockIdx.x * GEMM_WARPS + warp) * R;                      \
    int nb = a.n_in / 256;                                                     \
    int nc = a.batch < (NC) ? a.batch : (NC);                                  \
    float s[R][NC];                                                            \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        _Pragma("unroll") for (int t = 0; t < NC; t++) s[r][t] = 0.0f;         \
    int ib = (int)(lane >> 2), qd = (int)(lane & 3);                           \
    int boff = 16 * ib + 8 * (qd & 1);                                         \
    int hi = qd >= 2;                                                          \
    int el = (int)lane * KG8PAD;                                               \
    for (int b = 0; b < nb; b++) {                                             \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][(e >> 3) * KG8PAD + (e & 7)] = xg[e];                   \
        }                                                                      \
        __syncthreads();                                                       \
        float dl[R], w[R][8];                                                  \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < R; r++) {                                          \
            unsigned row = row0 + r;                                           \
            const uchar *blk = wb + a.w_off +                                  \
                (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 136 +      \
                (ulong64)b * 136;                                              \
            float d = f16f(blk);                                               \
            unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);          \
            int ls = ((blk[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) |               \
                     (((sh >> 2 * ib) & 3) << 4);                              \
            dl[r] = d * (float)(ls - 32);                                      \
            uint2 qv = *(const uint2 *)(blk + 8 + boff);                       \
            uint v0 = hi ? (qv.x >> 4) & 0x0F0F0F0Fu : qv.x & 0x0F0F0F0Fu;     \
            uint v1 = hi ? (qv.y >> 4) & 0x0F0F0F0Fu : qv.y & 0x0F0F0F0Fu;     \
            w[r][0] = (float)kv_iq4[v0 & 0xFF];                                \
            w[r][1] = (float)kv_iq4[(v0 >>  8) & 0xFF];                        \
            w[r][2] = (float)kv_iq4[(v0 >> 16) & 0xFF];                        \
            w[r][3] = (float)kv_iq4[(v0 >> 24)       ];                        \
            w[r][4] = (float)kv_iq4[v1 & 0xFF];                                \
            w[r][5] = (float)kv_iq4[(v1 >>  8) & 0xFF];                        \
            w[r][6] = (float)kv_iq4[(v1 >> 16) & 0xFF];                        \
            w[r][7] = (float)kv_iq4[(v1 >> 24)       ];                        \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xp = xsm[t] + el;                                     \
            float x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];              \
            float x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];              \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < R; r++) {                                      \
                float tt = w[r][0] * x0 + w[r][1] * x1 + w[r][2] * x2          \
                         + w[r][3] * x3 + w[r][4] * x4 + w[r][5] * x5          \
                         + w[r][6] * x6 + w[r][7] * x7;                        \
                s[r][t] += dl[r] * tt;                                         \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < R; r++) {                                              \
        unsigned row = row0 + r;                                               \
        if (row >= (unsigned)a.n_out) continue;                                \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            float v = warp_sum(s[r][t]);                                       \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? v + bias[row] : v;                                \
        }                                                                      \
    }                                                                          \
}

// ---- Q6_K: k_gemv_q6_K v2 — d factored out of the four-term group, a whole
// BLOCK reduced into its own accumulator, and those block sums split across
// TWO running accumulators (even blocks into s0, odd into s1, s0 + s1 at the
// end) because v2 keeps two blocks in flight. Folding both into one running
// sum, as this macro used to, is a different reduction tree.
// (k_gemm_q6_K premultiplies d into each weight instead, which agrees in exact
// arithmetic but not necessarily in floating point, so it is not a twin.)
// R rows a warp (Q6_TWR, as the K-quant twins above): a lane's four x values
// of each half are read once per column for all R rows.
#define GEMVB_Q6_K(NAME, NC)                                                   \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][256];                                             \
    enum { R = Q6_TWR(NC) };                                                   \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row0 = (blockIdx.x * GEMM_WARPS + warp) * R;                      \
    int nb = a.n_in / 256;                                                     \
    int nc = a.batch < (NC) ? a.batch : (NC);                                  \
    float s0[R][NC], s1[R][NC];                                                \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        _Pragma("unroll") for (int t = 0; t < NC; t++) s0[r][t] = s1[r][t] = 0.0f; \
    int is = (int)(lane >> 4);                                                 \
    for (int b = 0; b < nb; b++) {                                             \
        int base_e = b * 256;                                                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xg = x + (ulong64)t * a.xs + base_e;                  \
            for (int e = threadIdx.x; e < 256; e += blockDim.x)                \
                xsm[t][e] = xg[e];                                             \
        }                                                                      \
        __syncthreads();                                                       \
        float d[R], c[R][2][4];                                                \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < R; r++) {                                          \
            unsigned row = row0 + r;                                           \
            const uchar *blk = wb + a.w_off +                                  \
                (ulong64)(row < (unsigned)a.n_out ? row : 0) * nb * 210 +      \
                (ulong64)b * 210;                                              \
            d[r] = f16f(blk + 208);                                            \
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
                c[r][half][0] = (float)(sc[0 + is] * q1);                      \
                c[r][half][1] = (float)(sc[2 + is] * q2);                      \
                c[r][half][2] = (float)(sc[4 + is] * q3);                      \
                c[r][half][3] = (float)(sc[6 + is] * q4);                      \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            float acc[R];                                                      \
            _Pragma("unroll") for (int r = 0; r < R; r++) acc[r] = 0.0f;       \
            _Pragma("unroll")                                                  \
            for (int half = 0; half < 2; half++) {                             \
                const float *xp = xsm[t] + half * 128;                         \
                float xa = xp[lane], xb = xp[lane + 32];                       \
                float xc = xp[lane + 64], xd = xp[lane + 96];                  \
                _Pragma("unroll")                                              \
                for (int r = 0; r < R; r++)                                    \
                    acc[r] += d[r] * (c[r][half][0] * xa + c[r][half][1] * xb + \
                                      c[r][half][2] * xc + c[r][half][3] * xd); \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < R; r++) {                                      \
                if (b & 1) s1[r][t] += acc[r]; else s0[r][t] += acc[r];        \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < R; r++) {                                              \
        unsigned row = row0 + r;                                               \
        if (row >= (unsigned)a.n_out) continue;                                \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            float v = warp_sum(s0[r][t] + s1[r][t]);                           \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? v + bias[row] : v;                                \
        }                                                                      \
    }                                                                          \
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
#define GEMVB_BF16_C(NAME, NC, CH)                                             \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][(CH)];                                            \
    enum { R = DQ_TWR(NC) };                                                   \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row0 = (blockIdx.x * GEMM_WARPS + warp) * R;                      \
    int nc = a.batch < (NC) ? a.batch : (NC);                                  \
    const unsigned short *rw[R];                                               \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        rw[r] = (const unsigned short *)(wb + a.w_off) +                       \
                (ulong64)(row0 + r < (unsigned)a.n_out ? row0 + r : 0) * a.n_in; \
    float s[R][NC];                                                            \
    _Pragma("unroll") for (int r = 0; r < R; r++)                              \
        _Pragma("unroll") for (int t = 0; t < NC; t++) s[r][t] = 0.0f;         \
    for (int cs = 0; cs < a.n_in; cs += (CH)) {                                \
        int celems = a.n_in - cs < (CH) ? a.n_in - cs : (CH);                  \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            const float *xg = x + (ulong64)t * a.xs + cs;                      \
            for (int e = threadIdx.x; e < celems; e += blockDim.x)             \
                xsm[t][e] = xg[e];                                             \
        }                                                                      \
        __syncthreads();                                                       \
        for (int i = lane; i < celems; i += 32) {                              \
            float w[R];                                                        \
            _Pragma("unroll") for (int r = 0; r < R; r++) w[r] = bf16f(rw[r][cs + i]); \
            _Pragma("unroll")                                                  \
            for (int t = 0; t < NC; t++) {                                     \
                if (t >= nc) continue;                                         \
                float xv = xsm[t][i];                                          \
                _Pragma("unroll") for (int r = 0; r < R; r++) s[r][t] += w[r] * xv; \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < R; r++) {                                              \
        unsigned row = row0 + r;                                               \
        if (row >= (unsigned)a.n_out) continue;                                \
        _Pragma("unroll")                                                      \
        for (int t = 0; t < NC; t++) {                                         \
            if (t >= nc) continue;                                             \
            float v = warp_sum(s[r][t]);                                       \
            if (lane == 0) y[(ulong64)t * a.ys + row] =                        \
                a.has_bias ? v + bias[row] : v;                                \
        }                                                                      \
    }                                                                          \
}

#define GEMVB_Q8_0(NAME, NC) GEMVB_Q8_0_C(NAME, NC, Q8_CHUNK)
#define GEMVB_Q4_0(NAME, NC) GEMVB_Q4_0_C(NAME, NC, Q8_CHUNK)
#define GEMVB_BF16(NAME, NC) GEMVB_BF16_C(NAME, NC, BF16_CHUNK)

// Widths cuda.c can pick from. Two are enough to cover 2..8 without a
// half-empty batch paying much: a microbatch of 3 runs the 4-wide kernel.
// The 16-wide ones serve a speculative verify round of 9 to 16 rows in one
// pass over the weights (cuda.c exact_rows); they stage fewer elements per
// pass so sixteen columns stay inside the static shared memory: Q8_0 and
// Q4_0 sixteen blocks (any multiple of four keeps every lane's trips and
// tail), BF16 512 elements (a lane's elements stay lane, lane + 32, ...).
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
GEMVB_BF16_C(k_gemvb_bf16_x16, 16, 512)
GEMVB_Q8_0_C(k_gemvb_q8_0_x16, 16, 16)
GEMVB_Q4_0_C(k_gemvb_q4_0_x16, 16, 16)
GEMVB_Q4_K(k_gemvb_q4_K_x16, 16)
GEMVB_Q5_K(k_gemvb_q5_K_x16, 16)
GEMVB_Q6_K(k_gemvb_q6_K_x16, 16)

// ---- MXFP4: k_mv_mxfp4's twin (gpt-oss experts; the by-expert MoE prefill
// runs their columns through it). The batch-1 kernel gives each lane whole
// blocks (b = lane, lane + 32, ...) and reads x from global; so does this one,
// with one accumulator per column and the same inner order and warp_sum.
#define GEMVB_MXFP4(NAME, NC)                                                  \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][32 * LANE_XPAD];                                  \
    __shared__ float kv[16];                                                   \
    if (threadIdx.x < 16) kv[threadIdx.x] = kv_mxfp4[threadIdx.x];             \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    bool live = row < (unsigned)a.n_out;                                       \
    int nb = a.n_in / 32;                                                      \
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 17;      \
    float s[NC] = {0};                                                         \
    for (int b0 = 0; b0 < nb; b0 += 32) {                                      \
        int nblk = nb - b0 < 32 ? nb - b0 : 32;                                \
        for (int c = 0; c < NC && c < a.batch; c++)                            \
            LANE_STAGE_X(xsm[c], x + (ulong64)c * a.xs, b0, nblk)              \
        __syncthreads();                                                       \
        if (live && (int)lane < nblk) {                                        \
            const uchar *blk = rw + (ulong64)(b0 + lane) * 17;                 \
            float d = ldexpf(1.0f, (int)blk[0] - 127);                         \
            const uchar *q = blk + 1;                                          \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < NC; c++) {                                     \
                if (c >= a.batch) break;                                       \
                const float *xp = xsm[c] + lane * LANE_XPAD;                   \
                float t = 0;                                                   \
                for (int j = 0; j < 16; j++) {                                 \
                    t += kv[q[j] & 0xF] * xp[j];                               \
                    t += kv[q[j] >> 4]  * xp[j + 16];                          \
                }                                                              \
                s[c] += d * t;                                                 \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (live)                                                                  \
        for (int c = 0; c < a.batch && c < NC; c++) {                          \
            float r = warp_sum(s[c]);                                          \
            if (lane == 0) y[(ulong64)c * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}
GEMVB_MXFP4(k_gemvb_mxfp4_x4, 4)
GEMVB_MXFP4(k_gemvb_mxfp4_x8, 8)
// Sixteen columns: a lane owns whole blocks (block b0 + lane), so the group
// of 32 blocks cannot shrink without moving blocks between lanes; the columns
// are staged eight at a time instead, each block's scale and quants read once
// per half. A column's sum is the same terms in the same order either way.
#define GEMVB_MXFP4_HALVES(NAME, NC)                                           \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[(NC) / 2][32 * LANE_XPAD];                            \
    __shared__ float kv[16];                                                   \
    if (threadIdx.x < 16) kv[threadIdx.x] = kv_mxfp4[threadIdx.x];             \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    bool live = row < (unsigned)a.n_out;                                       \
    int nb = a.n_in / 32;                                                      \
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 17;      \
    float s[NC] = {0};                                                         \
    for (int b0 = 0; b0 < nb; b0 += 32) {                                      \
        int nblk = nb - b0 < 32 ? nb - b0 : 32;                                \
        _Pragma("unroll")                                                      \
        for (int h = 0; h < 2; h++) {                                          \
            for (int c = 0; c < (NC) / 2 && h * ((NC) / 2) + c < a.batch; c++) \
                LANE_STAGE_X(xsm[c], x + (ulong64)(h * ((NC) / 2) + c) * a.xs, b0, nblk) \
            __syncthreads();                                                   \
            if (live && (int)lane < nblk) {                                    \
                const uchar *blk = rw + (ulong64)(b0 + lane) * 17;             \
                float d = ldexpf(1.0f, (int)blk[0] - 127);                     \
                const uchar *q = blk + 1;                                      \
                _Pragma("unroll")                                              \
                for (int c = 0; c < (NC) / 2; c++) {                           \
                    int cc = h * ((NC) / 2) + c;                               \
                    if (cc >= a.batch) break;                                  \
                    const float *xp = xsm[c] + lane * LANE_XPAD;               \
                    float t = 0;                                               \
                    for (int j = 0; j < 16; j++) {                             \
                        t += kv[q[j] & 0xF] * xp[j];                           \
                        t += kv[q[j] >> 4]  * xp[j + 16];                      \
                    }                                                          \
                    s[cc] += d * t;                                            \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
    }                                                                          \
    if (live)                                                                  \
        for (int c = 0; c < a.batch && c < NC; c++) {                          \
            float r = warp_sum(s[c]);                                          \
            if (lane == 0) y[(ulong64)c * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}
GEMVB_MXFP4_HALVES(k_gemvb_mxfp4_x16, 16)

// ---- Q4_0: k_mv_q4_0's twin. The fused MoE prefill's Q4_0 expert kernel is
// k_mv_q4_0's lane-per-block body (k_moe_mv_q4_0), not the coalesced
// k_gemv_q4_0 that k_gemvb_q4_0_x* reproduce; the by-expert prefill runs Q4_0
// expert columns through this one: whole blocks per lane (b = lane, lane +
// 32, ...), one accumulator per column, the same inner expression and warp_sum.
#define GEMVB_Q4_0_MV(NAME, NC)                                                \
extern "C" __global__ void NAME(MV_PARAMS) {                                   \
    __shared__ float xsm[NC][32 * LANE_XPAD];                                  \
    unsigned warp = threadIdx.x >> 5;                                          \
    unsigned lane = threadIdx.x & 31;                                          \
    unsigned row  = blockIdx.x * GEMM_WARPS + warp;                            \
    bool live = row < (unsigned)a.n_out;                                       \
    int nb = a.n_in / 32;                                                      \
    const uchar *rw = wb + a.w_off + (ulong64)(live ? row : 0) * nb * 18;      \
    float s[NC] = {0};                                                         \
    for (int b0 = 0; b0 < nb; b0 += 32) {                                      \
        int nblk = nb - b0 < 32 ? nb - b0 : 32;                                \
        for (int c = 0; c < NC && c < a.batch; c++)                            \
            LANE_STAGE_X(xsm[c], x + (ulong64)c * a.xs, b0, nblk)              \
        __syncthreads();                                                       \
        if (live && (int)lane < nblk) {                                        \
            const uchar *blk = rw + (ulong64)(b0 + lane) * 18;                 \
            float d = f16f(blk);                                               \
            const uchar *q = blk + 2;                                          \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < NC; c++) {                                     \
                if (c >= a.batch) break;                                       \
                const float *xp = xsm[c] + lane * LANE_XPAD;                   \
                float t = 0;                                                   \
                for (int j = 0; j < 16; j++)                                   \
                    t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16]; \
                s[c] += d * t;                                                 \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
    if (live)                                                                  \
        for (int c = 0; c < a.batch && c < NC; c++) {                          \
            float r = warp_sum(s[c]);                                          \
            if (lane == 0) y[(ulong64)c * a.ys + row] =                        \
                a.has_bias ? r + bias[row] : r;                                \
        }                                                                      \
}
GEMVB_Q4_0_MV(k_gemvb_q4_0_mv_x4, 4)
GEMVB_Q4_0_MV(k_gemvb_q4_0_mv_x8, 8)

extern "C" __global__ void k_mv_iq4_xs(MV_PARAMS) {
    IQ_LANE_MV1(iq4xs_lane_n, 136)
}

// width-classed twins of the codebook GEMVs (2026-10-08): before these a
// 2-8 row verify tile ran the batch-1 kernel once per column
IQ_LANE_MVB(k_gemvb_iq2_xxs_x4, iq2xxs_lane_n, 66, 4)
IQ_LANE_MVB(k_gemvb_iq2_xxs_x8, iq2xxs_lane_n, 66, 8)
IQ_LANE_MVB(k_gemvb_iq2_xs_x4, iq2xs_lane_n, 74, 4)
IQ_LANE_MVB(k_gemvb_iq2_xs_x8, iq2xs_lane_n, 74, 8)
IQ_LANE_MVB(k_gemvb_iq2_s_x4, iq2s_lane_n, 82, 4)
IQ_LANE_MVB(k_gemvb_iq2_s_x8, iq2s_lane_n, 82, 8)
IQ_LANE_MVB(k_gemvb_iq3_xxs_x4, iq3xxs_lane_n, 98, 4)
IQ_LANE_MVB(k_gemvb_iq3_xxs_x8, iq3xxs_lane_n, 98, 8)
IQ_LANE_MVB(k_gemvb_iq3_s_x4, iq3s_lane_n, 110, 4)
IQ_LANE_MVB(k_gemvb_iq3_s_x8, iq3s_lane_n, 110, 8)
GEMVB_IQ4XS(k_gemvb_iq4_xs_x4, 4)
GEMVB_IQ4XS(k_gemvb_iq4_xs_x8, 8)
IQ_LANE_MVB(k_gemvb_iq2_xxs_x16, iq2xxs_lane_n, 66, 16)
IQ_LANE_MVB(k_gemvb_iq2_xs_x16, iq2xs_lane_n, 74, 16)
IQ_LANE_MVB(k_gemvb_iq2_s_x16, iq2s_lane_n, 82, 16)
IQ_LANE_MVB(k_gemvb_iq3_xxs_x16, iq3xxs_lane_n, 98, 16)
IQ_LANE_MVB(k_gemvb_iq3_s_x16, iq3s_lane_n, 110, 16)
GEMVB_IQ4XS(k_gemvb_iq4_xs_x16, 16)

// ---- int8-activation codebook GEMVs (RUNNER_CUDA_Q8X=1, EXPERIMENTAL) ------
// x is quantized once per call to int8 in 32-element blocks (scale amax/127),
// and a lane's eight codebook weights are packed into two signed int8x4 words
// for __dp4a, the route llama.cpp's MMVQ takes. A verify column then costs
// two integer dot instructions per lane-block instead of eight float FMAs and
// eight float loads. NOT the float route's bits: the activation is rounded to
// int8, so this is tolerance-gated against the CPU like every reassociating
// kernel; within the route, batch-1 and the 2..8-column kernels share one
// lane function, so a verify row is still the plain decode's bits.
extern "C" __global__ void k_quant_q8x(const float *x, signed char *xq, float *dx,
                                       int n, int ncols, int xs) {
    int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int nbk = n / 32;
    if (gw >= nbk * ncols) return;
    int c = gw / nbk, bk = gw - c * nbk;
    float v = x[(ulong64)c * xs + bk * 32 + lane];
    float am = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
    float d = am / 127.0f;
    int q = d > 0.0f ? __float2int_rn(v / d) : 0;
    xq[(ulong64)c * n + bk * 32 + lane] = (signed char)q;
    if (lane == 0) dx[(ulong64)c * nbk + bk] = d;
}

// four magnitude bytes, negated where sign bits 0..3 are set, as int8x4
static __device__ __forceinline__ int iq_pack4(unsigned g, unsigned s) {
    unsigned m = ((s & 1u) ? 0x000000ffu : 0u) | ((s & 2u) ? 0x0000ff00u : 0u) |
                 ((s & 4u) ? 0x00ff0000u : 0u) | ((s & 8u) ? 0xff000000u : 0u);
    return (int)__vsub4(g ^ m, m);
}

#define Q8X_COLS(W1, OFF1, W2, OFF2, SCALE) \
    _Pragma("unroll") \
    for (int c = 0; c < nc; c++) { \
        const signed char *xc = xq + (ulong64)c * n; \
        int s = __dp4a(W1, *(const int *)(xc + (OFF1)), 0); \
        s = __dp4a(W2, *(const int *)(xc + (OFF2)), s); \
        acc[c] += (SCALE) * dx[(ulong64)c * (n / 32) + ib] * (float)s; \
    }

// xq/dx point at this block's 256 elements / 8 scales of column 0; n is the
// column stride (the row length)
static __device__ __forceinline__ void iq3s_q8(const uchar *blk, const signed char *xq, const float *dx,
                                               int n, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3, pair = ib >> 1, h = ib & 1;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *qh = blk + 66, *sg = blk + 74 + ib * 4, *sc = blk + 106;
    float db = d * (float)(1 + 2 * (h ? (sc[pair] >> 4) : (sc[pair] & 0xF)));
    unsigned hb = qh[ib];
    unsigned g1 = kiq3s_grid[qs[2 * l + 0] | ((hb << (8 - 2 * l)) & 256)];
    unsigned g2 = kiq3s_grid[qs[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
    unsigned signs = sg[l];
    int w1 = iq_pack4(g1, signs), w2 = iq_pack4(g2, signs >> 4);
    Q8X_COLS(w1, ib * 32 + l * 8, w2, ib * 32 + l * 8 + 4, db)
}

static __device__ __forceinline__ void iq3xxs_q8(const uchar *blk, const signed char *xq, const float *dx,
                                                 int n, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    const uchar *qs = blk + 2 + ib * 8, *ss = blk + 66;
    unsigned aux = iq_ld32a2(ss + 4 * ib);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    unsigned g1 = kiq3xxs_grid[qs[2 * l + 0]];
    unsigned g2 = kiq3xxs_grid[qs[2 * l + 1]];
    unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
    int w1 = iq_pack4(g1, signs), w2 = iq_pack4(g2, signs >> 4);
    Q8X_COLS(w1, ib * 32 + l * 8, w2, ib * 32 + l * 8 + 4, db)
}

static __device__ __forceinline__ void iq4xs_q8(const uchar *blk, const signed char *xq, const float *dx,
                                                int n, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk);
    unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
    int ls = ((blk[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
    float dl = d * (ls - 32);
    const uchar *q = blk + 8 + ib * 16 + l * 4;
    unsigned lo = 0, hi = 0;
    _Pragma("unroll")
    for (int j = 0; j < 4; j++) {
        lo |= (unsigned)(uchar)kv_iq4[q[j] & 0xF] << (8 * j);
        hi |= (unsigned)(uchar)kv_iq4[q[j] >> 4] << (8 * j);
    }
    int wl = (int)lo, wh = (int)hi;
    Q8X_COLS(wl, ib * 32 + l * 4, wh, ib * 32 + 16 + l * 4, dl)
}

#define Q8X_PARAMS const uchar *wb, const signed char *xq, const float *dx, float *y, mv_args a, const float *bias
// ROWS rows per warp, NC columns; 4 warps per 128-thread block
#define Q8X_MV(NAME, fn, BS, NC, ROWS) \
extern "C" __global__ void NAME(Q8X_PARAMS) { \
    unsigned row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * (ROWS); \
    unsigned lane = threadIdx.x & 31; \
    if (row0 >= (unsigned)a.n_out) return; \
    int nb = a.n_in / 256; \
    float acc[ROWS][NC]; \
    _Pragma("unroll") for (int r = 0; r < (ROWS); r++) _Pragma("unroll") for (int c = 0; c < (NC); c++) acc[r][c] = 0.0f; \
    for (int b = 0; b < nb; b++) { \
        _Pragma("unroll") \
        for (int r = 0; r < (ROWS); r++) \
            if (row0 + r < (unsigned)a.n_out) \
                fn(wb + a.w_off + ((ulong64)(row0 + r) * nb + b) * BS, xq + b * 256, dx + b * 8, \
                   a.n_in, lane, (NC), acc[r]); \
    } \
    _Pragma("unroll") \
    for (int r = 0; r < (ROWS); r++) { \
        unsigned row = row0 + r; \
        for (int t = 0; t < a.batch && t < (NC); t++) { \
            float s = warp_sum(acc[r][t]); \
            if (lane == 0 && row < (unsigned)a.n_out) y[(ulong64)t * a.ys + row] = a.has_bias ? s + bias[row] : s; \
        } \
    } \
}
Q8X_MV(k_q8x_iq3_s_x1, iq3s_q8, 110, 1, 2)
Q8X_MV(k_q8x_iq3_s_x2, iq3s_q8, 110, 2, 1)
Q8X_MV(k_q8x_iq3_s_x3, iq3s_q8, 110, 3, 1)
Q8X_MV(k_q8x_iq3_s_x4, iq3s_q8, 110, 4, 1)
Q8X_MV(k_q8x_iq3_s_x5, iq3s_q8, 110, 5, 1)
Q8X_MV(k_q8x_iq3_s_x6, iq3s_q8, 110, 6, 1)
Q8X_MV(k_q8x_iq3_s_x7, iq3s_q8, 110, 7, 1)
Q8X_MV(k_q8x_iq3_s_x8, iq3s_q8, 110, 8, 1)
Q8X_MV(k_q8x_iq3_xxs_x1, iq3xxs_q8, 98, 1, 2)
Q8X_MV(k_q8x_iq3_xxs_x2, iq3xxs_q8, 98, 2, 1)
Q8X_MV(k_q8x_iq3_xxs_x3, iq3xxs_q8, 98, 3, 1)
Q8X_MV(k_q8x_iq3_xxs_x4, iq3xxs_q8, 98, 4, 1)
Q8X_MV(k_q8x_iq3_xxs_x5, iq3xxs_q8, 98, 5, 1)
Q8X_MV(k_q8x_iq3_xxs_x6, iq3xxs_q8, 98, 6, 1)
Q8X_MV(k_q8x_iq3_xxs_x7, iq3xxs_q8, 98, 7, 1)
Q8X_MV(k_q8x_iq3_xxs_x8, iq3xxs_q8, 98, 8, 1)
Q8X_MV(k_q8x_iq4_xs_x1, iq4xs_q8, 136, 1, 2)
Q8X_MV(k_q8x_iq4_xs_x2, iq4xs_q8, 136, 2, 1)
Q8X_MV(k_q8x_iq4_xs_x3, iq4xs_q8, 136, 3, 1)
Q8X_MV(k_q8x_iq4_xs_x4, iq4xs_q8, 136, 4, 1)
Q8X_MV(k_q8x_iq4_xs_x5, iq4xs_q8, 136, 5, 1)
Q8X_MV(k_q8x_iq4_xs_x6, iq4xs_q8, 136, 6, 1)
Q8X_MV(k_q8x_iq4_xs_x7, iq4xs_q8, 136, 7, 1)
Q8X_MV(k_q8x_iq4_xs_x8, iq4xs_q8, 136, 8, 1)

// Q4_K: d*sc*q - dmin*m per 32-element sub-block; lane L takes sub-block L/4,
// elements 8*(L%4).. +8 (one nibble half of the pair's 32 quant bytes)
static __device__ __forceinline__ void q4k_q8(const uchar *blk, const signed char *xq, const float *dx,
                                              int n, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    float d = f16f(blk), dmin = f16f(blk + 2);
    uchar sc, mn;
    get_scale_min_k4(ib, blk + 4, &sc, &mn);
    const uchar *q = blk + 16 + (ib >> 1) * 32 + l * 8;
    int shift = (ib & 1) * 4;
    int w0 = (int)((iq_ld32a2(q) >> shift) & 0x0F0F0F0Fu);
    int w1 = (int)((iq_ld32a2(q + 4) >> shift) & 0x0F0F0F0Fu);
    float dsc = d * (float)sc, dm = dmin * (float)mn;
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const signed char *xc = xq + (ulong64)c * n + ib * 32 + l * 8;
        int x0 = *(const int *)xc, x1 = *(const int *)(xc + 4);
        int s = __dp4a(w0, x0, 0); s = __dp4a(w1, x1, s);
        int sx = __dp4a(0x01010101, x0, 0); sx = __dp4a(0x01010101, x1, sx);
        acc[c] += dx[(ulong64)c * (n / 32) + ib] * (dsc * (float)s - dm * (float)sx);
    }
}

// Q8_0 in groups of eight 34-byte blocks (256 elements): lane L takes block
// L/4, elements 8*(L%4).. +8
static __device__ __forceinline__ void q80_q8(const uchar *grp, const signed char *xq, const float *dx,
                                              int n, int lane, int nc, float *acc) {
    int ib = lane >> 2, l = lane & 3;
    const uchar *blk = grp + ib * 34;
    float d = f16f(blk);
    int w0 = (int)iq_ld32a2(blk + 2 + l * 8), w1 = (int)iq_ld32a2(blk + 6 + l * 8);
    _Pragma("unroll")
    for (int c = 0; c < nc; c++) {
        const signed char *xc = xq + (ulong64)c * n + ib * 32 + l * 8;
        int s = __dp4a(w0, *(const int *)xc, 0); s = __dp4a(w1, *(const int *)(xc + 4), s);
        acc[c] += d * dx[(ulong64)c * (n / 32) + ib] * (float)s;
    }
}
Q8X_MV(k_q8x_q4_K_x1, q4k_q8, 144, 1, 2)
Q8X_MV(k_q8x_q4_K_x2, q4k_q8, 144, 2, 1)
Q8X_MV(k_q8x_q4_K_x3, q4k_q8, 144, 3, 1)
Q8X_MV(k_q8x_q4_K_x4, q4k_q8, 144, 4, 1)
Q8X_MV(k_q8x_q4_K_x5, q4k_q8, 144, 5, 1)
Q8X_MV(k_q8x_q4_K_x6, q4k_q8, 144, 6, 1)
Q8X_MV(k_q8x_q4_K_x7, q4k_q8, 144, 7, 1)
Q8X_MV(k_q8x_q4_K_x8, q4k_q8, 144, 8, 1)
Q8X_MV(k_q8x_q8_0_x1, q80_q8, 272, 1, 2)
Q8X_MV(k_q8x_q8_0_x2, q80_q8, 272, 2, 1)
Q8X_MV(k_q8x_q8_0_x3, q80_q8, 272, 3, 1)
Q8X_MV(k_q8x_q8_0_x4, q80_q8, 272, 4, 1)
Q8X_MV(k_q8x_q8_0_x5, q80_q8, 272, 5, 1)
Q8X_MV(k_q8x_q8_0_x6, q80_q8, 272, 6, 1)
Q8X_MV(k_q8x_q8_0_x7, q80_q8, 272, 7, 1)
Q8X_MV(k_q8x_q8_0_x8, q80_q8, 272, 8, 1)

// exact widths 2, 3, 5, 6 and 7 (2026-10-08): the x4/x8 classes made a 5-row
// verify tile (--draft-k 4) pay for eight columns; the 27B head decoded at
// 10.6 tok/s at k=4 against 13.6 at k=3 for that reason
IQ_LANE_MVB(k_gemvx_iq2_xxs_x2, iq2xxs_lane_n, 66, 2)
IQ_LANE_MVB(k_gemvx_iq2_xxs_x3, iq2xxs_lane_n, 66, 3)
IQ_LANE_MVB(k_gemvx_iq2_xxs_x5, iq2xxs_lane_n, 66, 5)
IQ_LANE_MVB(k_gemvx_iq2_xxs_x6, iq2xxs_lane_n, 66, 6)
IQ_LANE_MVB(k_gemvx_iq2_xxs_x7, iq2xxs_lane_n, 66, 7)
IQ_LANE_MVB(k_gemvx_iq2_xs_x2, iq2xs_lane_n, 74, 2)
IQ_LANE_MVB(k_gemvx_iq2_xs_x3, iq2xs_lane_n, 74, 3)
IQ_LANE_MVB(k_gemvx_iq2_xs_x5, iq2xs_lane_n, 74, 5)
IQ_LANE_MVB(k_gemvx_iq2_xs_x6, iq2xs_lane_n, 74, 6)
IQ_LANE_MVB(k_gemvx_iq2_xs_x7, iq2xs_lane_n, 74, 7)
IQ_LANE_MVB(k_gemvx_iq2_s_x2, iq2s_lane_n, 82, 2)
IQ_LANE_MVB(k_gemvx_iq2_s_x3, iq2s_lane_n, 82, 3)
IQ_LANE_MVB(k_gemvx_iq2_s_x5, iq2s_lane_n, 82, 5)
IQ_LANE_MVB(k_gemvx_iq2_s_x6, iq2s_lane_n, 82, 6)
IQ_LANE_MVB(k_gemvx_iq2_s_x7, iq2s_lane_n, 82, 7)
IQ_LANE_MVB(k_gemvx_iq3_xxs_x2, iq3xxs_lane_n, 98, 2)
IQ_LANE_MVB(k_gemvx_iq3_xxs_x3, iq3xxs_lane_n, 98, 3)
IQ_LANE_MVB(k_gemvx_iq3_xxs_x5, iq3xxs_lane_n, 98, 5)
IQ_LANE_MVB(k_gemvx_iq3_xxs_x6, iq3xxs_lane_n, 98, 6)
IQ_LANE_MVB(k_gemvx_iq3_xxs_x7, iq3xxs_lane_n, 98, 7)
IQ_LANE_MVB(k_gemvx_iq3_s_x2, iq3s_lane_n, 110, 2)
IQ_LANE_MVB(k_gemvx_iq3_s_x3, iq3s_lane_n, 110, 3)
IQ_LANE_MVB(k_gemvx_iq3_s_x5, iq3s_lane_n, 110, 5)
IQ_LANE_MVB(k_gemvx_iq3_s_x6, iq3s_lane_n, 110, 6)
IQ_LANE_MVB(k_gemvx_iq3_s_x7, iq3s_lane_n, 110, 7)

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
    // MOE_ROUTE_MAX experts per token in local memory (one thread per token);
    // 512 for Qwen3.8-Flash-Next (the host clamps ne to this bound)
    float lg[512];
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

// The staged form of the lane-per-block bodies (LANE_XPAD, at k_mv_q4_0):
// a launch's rows all read the slot's same x, so they share each copy.
#define MOE_LANE_HEAD                                                          \
    __shared__ float xsm[32 * LANE_XPAD];                                      \
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);        \
    unsigned lane = threadIdx.x & 31;                                          \
    bool live = row < (unsigned)a.n_out;                                       \
    const uchar *wbase = wb + a.w_off + (ulong64)sel[blockIdx.y] * a.estride;  \
    x += (ulong64)blockIdx.y * a.xs;                                           \
    float s = 0;
#define MOE_LANE_HEAD_N(EL)                                                    \
    __shared__ float xsm[32 * ((EL) + 1)];                                     \
    unsigned row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);        \
    unsigned lane = threadIdx.x & 31;                                          \
    bool live = row < (unsigned)a.n_out;                                       \
    const uchar *wbase = wb + a.w_off + (ulong64)sel[blockIdx.y] * a.estride;  \
    x += (ulong64)blockIdx.y * a.xs;                                           \
    float s = 0;
#define MOE_LANE_TAIL                                                          \
    if (live) {                                                                \
        s = warp_sum(s);                                                       \
        if (lane == 0) y[(ulong64)blockIdx.y * a.ys + row] = s;                \
    }

// body of k_mv_q4_0, lane per block (k_gemv_q4_0, the coalesced GEMV, came
// later and sums in another order; the by-expert prefill's Q4_0 twins below
// reproduce this body, not that one)
extern "C" __global__ void k_moe_mv_q4_0(MOE_MV_PARAMS) {
    MOE_LANE_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)(live ? row : 0) * nb * 18;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_X(xsm, x, b0, nblk)
        __syncthreads();
        if (live && (int)lane < nblk) {
            const uchar *blk = rw + (ulong64)(b0 + lane) * 18;
            float d = f16f(blk);
            const uchar *q = blk + 2;
            const float *xp = xsm + lane * LANE_XPAD;
            float t = 0;
            for (int j = 0; j < 16; j++)
                t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
            s += d * t;
        }
        __syncthreads();
    }
    MOE_LANE_TAIL;
}

// body of k_mv_mxfp4 (gpt-oss experts; no coalesced GEMV exists for MXFP4)
extern "C" __global__ void k_moe_mv_mxfp4(MOE_MV_PARAMS) {
    __shared__ float kv[16];
    if (threadIdx.x < 16) kv[threadIdx.x] = kv_mxfp4[threadIdx.x];
    MOE_LANE_HEAD;
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)(live ? row : 0) * nb * 17;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_X(xsm, x, b0, nblk)
        __syncthreads();
        if (live && (int)lane < nblk) {
            const uchar *blk = rw + (ulong64)(b0 + lane) * 17;
            float d = ldexpf(1.0f, (int)blk[0] - 127);
            const uchar *q = blk + 1;
            const float *xp = xsm + lane * LANE_XPAD;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                t += kv[q[j] & 0xF] * xp[j];
                t += kv[q[j] >> 4]  * xp[j + 16];
            }
            s += d * t;
        }
        __syncthreads();
    }
    MOE_LANE_TAIL;
}

// body of k_mv_nvfp4 (ModelOpt NVFP4 expert tensors)
extern "C" __global__ void k_moe_mv_nvfp4(MOE_MV_PARAMS, float scale) {
    __shared__ float kv[16];
    if (threadIdx.x < 16) kv[threadIdx.x] = kv_mxfp4[threadIdx.x];
    MOE_LANE_HEAD_N(64)
    int nb = a.n_in / 64;
    const uchar *rw = wbase + (ulong64)(live ? row : 0) * nb * 36;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 64)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (64 + 1);
            const uchar *blk = rw + (ulong64)b * 36;
            for (int sub = 0; sub < 4; sub++) {
                float d = ue4m3f(blk[sub]);
                const uchar *q = blk + 4 + sub * 8;
                const float *xp = xs_ + sub * 16;
                float t = 0;
                for (int j = 0; j < 8; j++) {
                    t += kv[q[j] & 0xF] * xp[j];
                    t += kv[q[j] >> 4]  * xp[j + 8];
                }
                s += d * t;
            }
        }
        __syncthreads();
    }
    if (live) {
        s = warp_sum(s) * scale;
        if (lane == 0) y[(ulong64)blockIdx.y * a.ys + row] = s;
    }
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

// body of k_mv_iq3_xxs (Qwen3.8-Flash-Next gate/up experts)
extern "C" __global__ void k_moe_mv_iq3_xxs(MOE_MV_PARAMS) {
    IQ_LANE_MOE_MV(iq3xxs_lane, 98)
}

// body of k_mv_iq4_nl (Qwen3.8-Flash-Next down experts)
extern "C" __global__ void k_moe_mv_iq4_nl(MOE_MV_PARAMS) {
    MOE_LANE_HEAD_N(32)
    int nb = a.n_in / 32;
    const uchar *rw = wbase + (ulong64)(live ? row : 0) * nb * 18;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 32)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (32 + 1);
            const uchar *blk = rw + (ulong64)b * 18;
            float d = f16f(blk);
            const uchar *q = blk + 2;
            const float *xp = xs_;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                t += (float)kv_iq4[q[j] & 0xF] * xp[j];
                t += (float)kv_iq4[q[j] >> 4]  * xp[j + 16];
            }
            s += d * t;
        }
        __syncthreads();
    }
    MOE_LANE_TAIL;
}

// body of k_mv_iq2_s (Qwen3.8-Flash-Next gate/up experts on 20 of 48 blocks)
extern "C" __global__ void k_moe_mv_iq2_s(MOE_MV_PARAMS) {
    IQ_LANE_MOE_MV(iq2s_lane, 82)
}

extern "C" __global__ void k_moe_mv_iq2_xs(MOE_MV_PARAMS) {
    IQ_LANE_MOE_MV(iq2xs_lane, 74)
}

extern "C" __global__ void k_moe_mv_iq2_xxs(MOE_MV_PARAMS) {
    IQ_LANE_MOE_MV(iq2xxs_lane, 66)
}

// body of k_mv_iq3_s (Qwen3.8-Flash-Next gate/up experts on 10 of 48 blocks)
extern "C" __global__ void k_moe_mv_iq3_s(MOE_MV_PARAMS) {
    IQ_LANE_MOE_MV(iq3s_lane, 110)
}

// body of k_mv_iq4_xs (Qwen3.8-Flash-Next gate/up experts on its last block)
extern "C" __global__ void k_moe_mv_iq4_xs(MOE_MV_PARAMS) {
    MOE_LANE_HEAD_N(256)
    int nb = a.n_in / 256;
    const uchar *rw = wbase + (ulong64)(live ? row : 0) * nb * 136;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 256)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (256 + 1);
            const uchar *blk = rw + (ulong64)b * 136;
            float d = f16f(blk);
            unsigned sh = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
            const uchar *sl = blk + 4;
            const uchar *q  = blk + 8;
            const float *xp = xs_;
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
        __syncthreads();
    }
    MOE_LANE_TAIL;
}

// body of k_mv_q2_0 (some blocks' down experts in the GSQ release)
extern "C" __global__ void k_moe_mv_q2_0(MOE_MV_PARAMS) {
    MOE_LANE_HEAD_N(64)
    int nb = a.n_in / 64;
    const uchar *rw = wbase + (ulong64)(live ? row : 0) * nb * 18;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        int nblk = nb - b0 < 32 ? nb - b0 : 32;
        LANE_STAGE_XN(xsm, x, b0, nblk, 64)
        __syncthreads();
        if (live && (int)lane < nblk) {
            int b = b0 + (int)lane;
            const float *xs_ = xsm + lane * (64 + 1);
            const uchar *blk = rw + (ulong64)b * 18;
            float d = f16f(blk);
            const uchar *q = blk + 2;
            const float *xp = xs_;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                uchar c = q[j];
                t += ((int)(c & 3) - 1) * xp[4 * j]
                   + ((int)((c >> 2) & 3) - 1) * xp[4 * j + 1]
                   + ((int)((c >> 4) & 3) - 1) * xp[4 * j + 2]
                   + ((int)(c >> 6) - 1) * xp[4 * j + 3];
            }
            s += d * t;
        }
        __syncthreads();
    }
    MOE_LANE_TAIL;
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
static __device__ __forceinline__ void moe_sum_body(float *out, const float *eout, int n,
                                                    int nslots, int es, const float *db,
                                                    const int *sel, const float *selw, int i) {
    float s = eout[i];
    if (db) s += selw[0] * db[(ulong64)sel[0] * n + i];
    for (int sl = 1; sl < nslots; sl++) {
        s += eout[(ulong64)sl * es + i];
        if (db) s += selw[sl] * db[(ulong64)sel[sl] * n + i];
    }
    out[i] = s;
}
extern "C" __global__ void k_moe_sum(float *out, const float *eout, int n,
                                     int nslots, int es, const float *db,
                                     const int *sel, const float *selw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    moe_sum_body(out, eout, n, nslots, es, db, sel, selw, i);
}

// k_moe_sum for every token of a tile in one launch, grid.y = token: token
// t's slots are eout rows t * nslots .. t * nslots + nslots - 1 (es apart) and
// its routing at sel/selw + t * nslots; the sum lands in column t of out
// (stride os). Each token's arithmetic is k_moe_sum's (one body).
extern "C" __global__ void k_moe_sum_tile(float *out, const float *eout, int n,
                                          int nslots, int es, const float *db,
                                          const int *sel, const float *selw,
                                          int os) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ulong64 t = blockIdx.y;
    moe_sum_body(out + t * os, eout + t * (ulong64)nslots * es, n, nslots, es, db,
                 sel + t * nslots, selw + t * nslots, i);
}

// dst[pos[c]][i] = src[c][i]: an expert's per-column outputs placed at their
// (token, slot) rows of the tile's slot buffer, for k_moe_sum_tile. A copy.
extern "C" __global__ void k_moe_place(float *dst, const float *src,
                                       const int *pos, int n, int ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(ulong64)pos[blockIdx.y] * n + i] = src[(ulong64)blockIdx.y * ss + i];
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
                                        int heads, float eps, int gate_sigmoid) {
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
    // every thread has read red[0] before any overwrites it with its k
    // partial: without this barrier warp 0 could store its ks over the q sum
    // while another warp had yet to read it (a silent wrong q scale, found
    // 2026-10-20 while reading this kernel for a whole-tile variant)
    __syncthreads();
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
        // Qwen3.5 gates the normed output with silu(z); Qwen3.8 (qwen4exp)
        // with sigmoid(z), mirroring the CPU path's one numerical difference
        float sg = 1.0f / (1.0f + expf(-zv));
        yo[j] = yo[j] * rms * norm[j] * (gate_sigmoid ? sg : zv * sg);
    }
}

// ---- qwen4exp hyper-connections (CPU twins: hc_mix / hc_combine in model.c)
// Grouped RMSNorm: row r of x is `groups` streams of n values, each normed
// over its own n and scaled by that stream's gamma slice (w has groups*n).
// One block per (stream, row).
extern "C" __global__ void k_rmsnorm_grouped(const float *x, float *y, const float *w,
                                             int n, int groups, float eps, int xs, int ys) {
    __shared__ float red[256];
    int tid = threadIdx.x, tpg = blockDim.x, c = blockIdx.x;
    x += (ulong64)blockIdx.y * xs + (ulong64)c * n;
    y += (ulong64)blockIdx.y * ys + (ulong64)c * n;
    w += (ulong64)c * n;
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

// NextN/MTP head input per stream: cat[row*hc + c] = [en[row] ; hn[row][c]],
// the concat graph_mtp feeds eh_proj for every hyper-connection stream
// (CPU mtp_run builds the same rows on the host).
extern "C" __global__ void k_mtp_cat(const float *en, const float *hn, float *cat,
                                     int E, int hc) {
    int row = blockIdx.y, c = blockIdx.x;
    const float *e = en + (ulong64)row * E;
    const float *h = hn + (ulong64)row * hc * E + (ulong64)c * E;
    float *o = cat + ((ulong64)row * hc + c) * 2 * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { o[i] = e[i]; o[E + i] = h[i]; }
}

// x[row][i] = silu(x[row][i] * s), the mixer's low-rank activation (CPU silu_f
// with its -80 cutoff: exp(80+) is inf there and g/(1+inf) is 0 here as well).
extern "C" __global__ void k_silu_scale(float *x, float s, int n, int xs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *xr = x + (ulong64)blockIdx.y * xs;
    float g = xr[i] * s;
    xr[i] = g < -80.0f ? 0.0f : g / (1.0f + expf(-g));
}

// shared expert: y[row][i] += sigmoid(glogit[row]) * v[row][i] (the CPU's
// shexp_add tail; glogit is the 1-wide ffn_gate_inp_shexp projection)
extern "C" __global__ void k_axpy_sigmoid(float *y, const float *v, const float *glogit,
                                          int n, int ys, int vs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gsig = 1.0f / (1.0f + expf(-glogit[blockIdx.y]));
    y[(ulong64)blockIdx.y * ys + i] += gsig * v[(ulong64)blockIdx.y * vs + i];
}

// mixed[row][i] = mean over streams c of xn[row][c*n+i] * sigmoid(gate[row][c*n+i]).
// sigmoid as 1/(1+exp(-g)): exp overflow gives 1/inf = 0 and underflow gives
// exactly 1 in IEEE float, the values the CPU's guarded sigmoid_f returns.
extern "C" __global__ void k_hc_gate_mean(const float *xn, const float *gate, float *mixed,
                                          int n, int groups, int xs, int ms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *xr = xn + (ulong64)blockIdx.y * xs;
    const float *gr = gate + (ulong64)blockIdx.y * xs;
    float acc = 0.0f;
    for (int c = 0; c < groups; c++) {
        float gv = gr[(ulong64)c * n + i];
        acc += xr[(ulong64)c * n + i] * (1.0f / (1.0f + expf(-gv)));
    }
    mixed[(ulong64)blockIdx.y * ms + i] = acc * (1.0f / (float)groups);
}

// x[row][c*n+i] += 2*sigmoid(inject[row][c]/groups) * out[row][i], every stream.
extern "C" __global__ void k_hc_combine(float *x, const float *out, const float *inject,
                                        int n, int groups, int xs, int os) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *xr = x + (ulong64)blockIdx.y * xs;
    float ov = out[(ulong64)blockIdx.y * os + i];
    const float *inj = inject + (ulong64)blockIdx.y * groups;
    float inv = 1.0f / (float)groups;
    for (int c = 0; c < groups; c++) {
        float w = 2.0f / (1.0f + expf(-inj[c] * inv));
        xr[(ulong64)c * n + i] += w * ov;
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

extern "C" __global__ void k_gemm_q6_K_tc(MV_PARAMS, const float *xsc) {
    using namespace nvcuda::wmma;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const unsigned row0 = blockIdx.x * TC_ROWS;
    __shared__ __align__(32) __half sh_w[TC_ROWS * TC_KP];
    __shared__ __align__(32) __half sh_x[TC_N * TC_KP];
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
                __half *dst = sh_w + srow * TC_KP + sseg * 64;
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
            {   // ---- stage activations: 128 K x TC_N tokens, each column
                //      scaled to fp16's range (see k_gemm_q4_K_tc) ----
                int col = tid / 2, part = tid % 2;
                __half *dst = sh_x + col * TC_KP + part * 64;
                if (col < a.batch) {
                    const float xinv = 16384.0f / xsc[col];
                    const float *xg = x + (ulong64)col * a.xs + b * 256 + koff
                                      + part * 64;
                    #pragma unroll
                    for (int v = 0; v < 16; v++) {
                        float4 xv = *(const float4 *)(xg + v * 4);
                        dst[v * 4 + 0] = __float2half(xv.x * xinv);
                        dst[v * 4 + 1] = __float2half(xv.y * xinv);
                        dst[v * 4 + 2] = __float2half(xv.z * xinv);
                        dst[v * 4 + 3] = __float2half(xv.w * xinv);
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 64; e++) dst[e] = __float2half(0.0f);
                }
            }
            __syncthreads();
            const __half *wt = sh_w + warp * 16 * TC_KP;
            #pragma unroll
            for (int k = 0; k < TC_K; k += 16) {
                load_matrix_sync(fa, wt + k, TC_KP);
                #pragma unroll
                for (int n = 0; n < TC_N / 16; n++) {
                    load_matrix_sync(fb, sh_x + n * 16 * TC_KP + k, TC_KP);
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
        if (gr < (unsigned)a.n_out && tt < a.batch) {
            float r = sh_c[idx] * (xsc[tt] * (1.0f / 16384.0f));
            y[(ulong64)tt * a.ys + gr] = a.has_bias ? r + bias[gr] : r;
        }
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

// ------------------------------------------- pipelined tensor-core GEMM (TC2)
// C1 of the performance campaign (RUNNER_CUDA_TC2, see enc_mv in cuda.c).
// The same numerical class as the TC kernels above, built for throughput:
//
//   operands  every weight element is dequantized by the arithmetic the TC
//             kernel of its type uses, every activation is
//             half(x * 16384 / max|x|) of its column (k_x_half_scaled, run
//             once per GEMM input instead of once per row block), and the
//             products accumulate in fp32. Only the order of the fp32 sums
//             differs from the TC kernels, so the obligation is theirs:
//             tolerance (test-tc-tol), with RUNNER_CUDA_TC2=0 restoring them.
//   tiles     a block owns 128 rows x 64 token columns, its eight warps
//             32 x 32 each, and walks K in stages of 64. Two stages of
//             shared tiles alternate: while the warps multiply stage s, the
//             raw bytes of stage s + 1 are already in registers (loaded
//             before the multiplies), and are converted into the other
//             buffer after them, so the global load latency hides under the
//             tensor work. Rows are padded by 8 halves so ldmatrix and the
//             staging stores are free of bank conflicts.
//   mma       mma.sync m16n8k8 (f16 in, f32 accumulate) with ldmatrix,
//             both available on the compute_75 target the PTX is built for.
//
// The design follows the published register-prefetch double-buffered GEMMs
// for Turing-class tensor cores (no cp.async before compute_80); the
// layouts, the dequantization and the epilogue are this file's own.
#include "kernels_tc2.cuh"

// The scaled fp16 activations of a GEMM input: xh[c][k] =
// half(x[c * xs + k] * (16384 / xsc[c])), the TC kernels' staging value.
// grid (ceil(n_in / 256), batch), 256 threads.
extern "C" __global__ void k_x_half_scaled(const float *x, const float *xsc, __half *xh,
                                           int n_in, int xs, int batch) {
    int c = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= batch || k >= n_in) return;
    const float xinv = 16384.0f / xsc[c];
    xh[(ulong64)c * n_in + k] = __float2half(x[(ulong64)c * xs + k] * xinv);
}

#include "kernels_tc2_iq.cuh"

// the activations arrive pre-scaled in half (k_x_half_scaled), so the
// kernel's x parameter is that buffer
extern "C" __global__ void __launch_bounds__(256)
k_gemm2_q4_K(const uchar *wb, const __half *xh, float *y, mv_args a, const float *bias,
             const float *xsc) {
    gemm2_body<G2Q4K>(wb, xh, y, a, bias, xsc);
}
extern "C" __global__ void __launch_bounds__(256)
k_gemm2_q6_K(const uchar *wb, const __half *xh, float *y, mv_args a, const float *bias,
             const float *xsc) {
    gemm2_body<G2Q6K>(wb, xh, y, a, bias, xsc);
}
extern "C" __global__ void __launch_bounds__(256)
k_gemm2_q8_0(const uchar *wb, const __half *xh, float *y, mv_args a, const float *bias,
             const float *xsc) {
    gemm2_body<G2Q80>(wb, xh, y, a, bias, xsc);
}
extern "C" __global__ void __launch_bounds__(256)
k_gemm2_q4_0(const uchar *wb, const __half *xh, float *y, mv_args a, const float *bias,
             const float *xsc) {
    gemm2_body<G2Q40>(wb, xh, y, a, bias, xsc);
}

// ------------------------------------------ grouped decode attention (C2)
// RUNNER_CUDA_ATTN_DEC (cuda.c tile_attn, tn == 1). k_attn_dec gives each
// QUERY head its own blocks, and every one of them reads the whole K and V
// of its KV head from global memory: a model with G query heads per KV head
// reads the cache G times per token (8 on Qwen2.5-3B, where decode at a 16K
// context fell to 17% of another engine's rate in the 2026-10-20 baseline).
// This kernel gives one block a KV head and a slice of the positions, stages
// each tile of TK cache rows once in shared memory (the cache's own bytes),
// and serves all G query heads from it:
//   pass 1: every score through the unchanged kv_dot, each head's maximum;
//   pass 2: the same scores recomputed, expf(s - max), their sum, and the
//           weighted V rows through kv_pair;
// then writes k_attn_dec's partial (un-normalised V, max, sum) per head, so
// the merge is k_attn_dec's arithmetic over AD_SPLITS slices.
// Splits: AD_SPLITS is fixed (the decode graph's grid must not depend on the
// position), the slice length computed on the device from pos as before.
// Obligation: k_attn_dec's (deterministic within the route; the slice and
// reduction grouping differ, so token-identical to the CPU scalar route and
// gated empirically, not bit-identical to k_attn_dec).
#define AD_SPLITS 32
#define AD_TK 32
#define AD_THREADS 256
#define AD_GMAX 8   // query heads per KV head a block serves (larger: k_attn_dec)

// The body for one query column: q, the cache pair, part and pos are that
// column's own, so k_attn_dec_gqa (the solo decode) and k_attn_dec_gqa_seq
// (the decode microbatch, one column per sequence) run the same arithmetic.
__device__ __forceinline__ void
attn_dec_gqa_body(const float *q, const unsigned char *kc, const unsigned char *vc,
                  float *part, const attn_args &a, const int pos) {
    extern __shared__ float ad_sm[];
    const int kvh = blockIdx.x, sp = blockIdx.y, tid = threadIdx.x;
    const int hd = a.head_dim, G = a.n_head / a.n_head_kv, h0 = kvh * G;
    const int kv_dim = a.n_head_kv * hd;
    const ulong64 row_b = KV_ROW_BYTES(kv_dim, a.q8), row_v = KV_ROW_BYTES(kv_dim, a.vq8);
    const ulong64 base = a.l_off + kv_head_off(kvh, hd, a.q8);
    const ulong64 base_v = a.v_off + kv_head_off(kvh, hd, a.vq8);
    const int kbytes = (int)KV_ROW_BYTES(hd, a.q8), vbytes = (int)KV_ROW_BYTES(hd, a.vq8);
    const int kws = ((kbytes + 3) / 4) | 1, vws = ((vbytes + 3) / 4) | 1;
    const int bws = kws > vws ? kws : vws;
    int t0 = 0;
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1;
    const int total = pos + 1 - t0;
    // whole AD_TK tiles per slice: a short context then fills a few splits
    // with full tiles (the rest take the empty-slice exit below) instead of
    // giving all AD_SPLITS blocks a few rows each of a mostly idle tile
    int slice = (total + AD_SPLITS - 1) / AD_SPLITS;
    slice = (slice + AD_TK - 1) / AD_TK * AD_TK;
    const int s0 = t0 + sp * slice;
    int s1 = s0 + slice;
    if (s1 > pos + 1) s1 = pos + 1;
    if (s0 >= s1) {                      // empty slice: the merge's sentinel
        if (tid < G) {
            float *P = part + ((ulong64)(h0 + tid) * AD_SPLITS + sp) * (hd + 2);
            P[hd] = -1e30f;
            P[hd + 1] = 0.0f;
        }
        return;
    }
    float *Q = ad_sm;                                   // [G][hd + 1]
    unsigned *KV = (unsigned *)(Q + AD_GMAX * (hd + 1)); // [AD_TK][bws]
    float *Pr = (float *)(KV + AD_TK * bws);            // [G][AD_TK]
    float *hmax = Pr + AD_GMAX * AD_TK;                 // [G]
    for (int i = tid; i < G * hd; i += AD_THREADS) {
        int g = i / hd, d = i - g * hd;
        Q[g * (hd + 1) + d] = q[(ulong64)(h0 + g) * hd + d];
    }
    auto stage = [&](const unsigned char *cache, ulong64 b0, ulong64 rb, int nbytes, int kt) {
        // words only for a word-aligned slice of whole words (see attn_tile_body)
        if (((b0 | rb | (ulong64)nbytes) & 3) == 0) {
            int nw = nbytes / 4;
            for (int i = tid; i < AD_TK * nw; i += AD_THREADS) {
                int k = i / nw, w = i - k * nw, t = kt + k;
                KV[k * bws + w] = t < s1 ? *(const unsigned *)(cache + b0 + kv_slot(t, a.ring) * rb + (ulong64)w * 4)
                                         : 0u;
            }
        } else {
            for (int i = tid; i < AD_TK * nbytes; i += AD_THREADS) {
                int k = i / nbytes, c = i - k * nbytes, t = kt + k;
                ((unsigned char *)(KV + k * bws))[c] = t < s1 ? cache[b0 + kv_slot(t, a.ring) * rb + (ulong64)c] : 0;
            }
        }
    };
    // score thread mapping: head g = tid / AD_TK, key k = tid % AD_TK
    const int sg = tid / AD_TK, sk = tid - sg * AD_TK;
    float mx = -1e30f;
    for (int kt = s0; kt < s1; kt += AD_TK) {
        __syncthreads();
        stage(kc, base, row_b, kbytes, kt);
        __syncthreads();
        if (sg < G && kt + sk < s1)
            mx = fmaxf(mx, kv_dot((const unsigned char *)(KV + sk * bws), Q + sg * (hd + 1), hd, a.q8) * a.scale);
    }
    for (int off = AD_TK / 2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if (sk == 0 && sg < G) hmax[sg] = mx;
    // value mapping: (head, pair) outputs, AD_GMAX * 128 pairs at most
    const int P2 = hd / 2;
    float2 acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) acc[j] = make_float2(0.0f, 0.0f);
    float sum = 0.0f;
    for (int kt = s0; kt < s1; kt += AD_TK) {
        __syncthreads();
        stage(kc, base, row_b, kbytes, kt);
        __syncthreads();
        float pv = 0.0f;
        if (sg < G && kt + sk < s1)
            pv = expf(kv_dot((const unsigned char *)(KV + sk * bws), Q + sg * (hd + 1), hd, a.q8) * a.scale - hmax[sg]);
        sum += pv;
        if (sg < G) Pr[sg * AD_TK + sk] = pv;
        __syncthreads();
        stage(vc, base_v, row_v, vbytes, kt);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int idx = tid + j * AD_THREADS;
            if (idx >= G * P2) break;
            int g = idx / P2, i2 = idx - g * P2;
            float2 o = acc[j];
            for (int k = 0; k < AD_TK; k++) {
                float pk = Pr[g * AD_TK + k];
                float2 vf = kv_pair((const unsigned char *)(KV + k * bws), i2, a.vq8);
                o.x += pk * vf.x;
                o.y += pk * vf.y;
            }
            acc[j] = o;
        }
    }
    for (int off = AD_TK / 2; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (sk == 0 && sg < G) {
        float *P = part + ((ulong64)(h0 + sg) * AD_SPLITS + sp) * (hd + 2);
        P[hd] = hmax[sg];
        P[hd + 1] = sum;
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int idx = tid + j * AD_THREADS;
        if (idx >= G * P2) break;
        int g = idx / P2, i2 = idx - g * P2;
        float *P = part + ((ulong64)(h0 + g) * AD_SPLITS + sp) * (hd + 2);
        P[2 * i2] = acc[j].x;
        P[2 * i2 + 1] = acc[j].y;
    }
}

extern "C" __global__ void __launch_bounds__(AD_THREADS)
k_attn_dec_gqa(const float *q, const unsigned char *kc, const unsigned char *vc,
               float *part, attn_args a, const int *posp) {
    attn_dec_gqa_body(q, kc, vc, part, a, *posp);
}

// The decode microbatch's twin (fwd_batch): grid (KV head, slice, sequence),
// each column at its own position over its own cache (posp, kcp, vcp are
// per-column arrays, as k_attn_dec_seq takes them) and its own partials, so
// every column is bitwise what k_attn_dec_gqa computes for it alone.
extern "C" __global__ void __launch_bounds__(AD_THREADS)
k_attn_dec_gqa_seq(const float *q, const ulong64 *kcp, const ulong64 *vcp,
                   float *part, attn_args a, const int *posp) {
    const int tk = blockIdx.z;
    attn_dec_gqa_body(q + (ulong64)tk * a.qs, (const unsigned char *)kcp[tk],
                      (const unsigned char *)vcp[tk],
                      part + (ulong64)tk * a.n_head * AD_SPLITS * (a.head_dim + 2), a, posp[tk]);
}

// k_attn_merge over AD_SPLITS partials (its arithmetic, a different count),
// for one column's partials into that column's output
__device__ __forceinline__ void
attn_merge32_body(float *out, const float *part, const attn_args &a, const float *sinks) {
    int h = blockIdx.x;
    int tid = threadIdx.x, tpg = blockDim.x;
    int hd = a.head_dim;
    const float *base = part + (ulong64)h * AD_SPLITS * (hd + 2);
    float M = -1e30f;
    for (int sp = 0; sp < AD_SPLITS; sp++)
        M = fmaxf(M, base[sp * (hd + 2) + hd]);
    if (sinks && sinks[h] > M) M = sinks[h];
    float L = 0.f;
    for (int sp = 0; sp < AD_SPLITS; sp++) {
        float m = base[sp * (hd + 2) + hd];
        if (m <= -1e29f) continue;
        L += base[sp * (hd + 2) + hd + 1] * expf(m - M);
    }
    if (sinks) L += expf(sinks[h] - M);
    for (int i = tid; i < hd; i += tpg) {
        float acc = 0.f;
        for (int sp = 0; sp < AD_SPLITS; sp++) {
            const float *P = base + sp * (hd + 2);
            float m = P[hd];
            if (m <= -1e29f) continue;
            acc += P[i] * expf(m - M);
        }
        out[h * hd + i] = acc / L;
    }
}

extern "C" __global__ void k_attn_merge32(float *out, const float *part,
                                          attn_args a, const int *posp,
                                          const float *sinks) {
    attn_merge32_body(out, part, a, sinks);
    (void)posp;
}

// k_attn_merge32 per microbatch column: grid (n_head, sequence)
extern "C" __global__ void k_attn_merge32_seq(float *out, const float *part,
                                              attn_args a, const int *posp,
                                              const float *sinks) {
    const int tk = blockIdx.y;
    attn_merge32_body(out + (ulong64)tk * a.os,
                      part + (ulong64)tk * a.n_head * AD_SPLITS * (a.head_dim + 2), a, sinks);
    (void)posp;
}
#define G2_KERNEL(NAME, F)                                                              \
extern "C" __global__ void __launch_bounds__(256)                                       \
NAME(const uchar *wb, const __half *xh, float *y, mv_args a, const float *bias,          \
     const float *xsc) {                                                                 \
    gemm2_body<F>(wb, xh, y, a, bias, xsc);                                              \
}
G2_KERNEL(k_gemm2_iq3_s, G2IQ3S)
G2_KERNEL(k_gemm2_iq3_xxs, G2IQ3XXS)
G2_KERNEL(k_gemm2_iq2_s, G2IQ2S)
G2_KERNEL(k_gemm2_iq2_xs, G2IQ2XS)
G2_KERNEL(k_gemm2_iq2_xxs, G2IQ2XXS)
G2_KERNEL(k_gemm2_iq1_s, G2IQ1S)
G2_KERNEL(k_gemm2_iq1_m, G2IQ1M)
G2_KERNEL(k_gemm2_iq4_xs, G2IQ4XS)
G2_KERNEL(k_gemm2_iq4_nl, G2IQ4NL)

// --------------------------------------- tensor-core prefill attention (C2)
// C2 step 2 (RUNNER_CUDA_ATTN_TC, tile_attn in cuda.c): k_attn_tile's two
// passes with both products on tensor cores, for fp16 caches. A block owns 64
// query rows (heads of one KV group times a strip of tokens, k_attn_tile's
// rows), four warps of 16, and walks the strip's keys in tiles of 64:
//   pass 1: S = Q K^T (mma.sync m16n8k8, f16 in, f32 accumulate), the scale,
//           the causal / sliding mask, each row's maximum;
//   pass 2: S again, p = expf(s - max) in fp32, its row sum in fp32, then
//           O += P V with P as half operands in registers (an m16n8
//           accumulator tile is exactly the A operand of the next m16n8k8,
//           so P never leaves the warp) and V read through ldmatrix.trans.
// Q is scaled by an exact power of two per row (max|q| brought to at most
// 2^14, undone on the score), so no row overflows fp16. Q and P each go in
// as two halves, hi = half(x) and lo = half(x - hi), both multiplied against
// the exact fp16 K or V: a product carries about 22 bits of the operand
// instead of 11. With one half each, Mistral-7B's long rows read further from
// the float32 reference than k_attn's (Wilcoxon p 0.014; Q alone split, p
// 0.009); with both split they do not (X23b). K and V are the fp16 cache rows
// themselves. Obligation: tolerance, like the TC GEMM: fp32 accumulation in a
// different summation order from k_attn_tile, gated by the CPU identity
// bound, the long-context greedy rows and the fp32 distance rows. Shared
// memory (dynamic, set by cuda.c): Q hi, Q lo and the K/V tile, 3 x 64 x 136
// halves, plus the row scales: AQ_SMEM = 52,480 bytes.
#define AQ_ROWS 64
#define AQ_TK   64
#define AQ_LD   (128 + 8)     // halves per staged row (head_dim <= 128, padded)
#define AQ_SMEM (3 * AQ_ROWS * AQ_LD * 2 + AQ_ROWS * 4)

extern "C" __global__ void __launch_bounds__(128)
k_attn_tc(const float *q, const unsigned char *kc, const unsigned char *vc, float *out,
          attn_args a, const int *posp, const float *sinks, int tn, int hb, int qs) {
    extern __shared__ __align__(16) unsigned char aq_dyn[];
    __half *Qs  = (__half *)aq_dyn;
    __half *Qlo = Qs + AQ_ROWS * AQ_LD;
    __half *KVs = Qlo + AQ_ROWS * AQ_LD;
    float *qscale = (float *)(KVs + AQ_TK * AQ_LD);
    const int hd = a.head_dim, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int G = a.n_head / a.n_head_kv;
    const int ngh = (G + hb - 1) / hb;
    const int kvh = blockIdx.x / ngh, hg = blockIdx.x % ngh;
    const int h0 = kvh * G + hg * hb;
    const int nh = G - hg * hb < hb ? G - hg * hb : hb;
    const int tk0 = blockIdx.y * qs;
    const int nt = tn - tk0 < qs ? tn - tk0 : qs;
    const int pos0 = *posp;
    const int kv_dim = a.n_head_kv * hd;
    const ulong64 row_b = (ulong64)kv_dim * 2;            // fp16 rows only
    const ulong64 base = a.l_off + (ulong64)kvh * hd * 2;
    const ulong64 base_v = a.v_off + (ulong64)kvh * hd * 2;

    // ---- Q rows, each scaled by an exact power of two into fp16's range ----
    for (int r = warp; r < AQ_ROWS; r += 4) {
        int hi = r / qs, ti = r - hi * qs;
        bool ok = hi < nh && ti < nt;
        const float *qr = q + (ulong64)(tk0 + ti) * a.qs + (ulong64)(h0 + hi) * hd;
        float mx = 0.0f;
        if (ok) for (int d = lane; d < hd; d += 32) mx = fmaxf(mx, fabsf(qr[d]));
        for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        int e = 0;
        if (mx > 0.0f) { frexpf(mx, &e); }              // mx < 2^e
        float s = mx > 0.0f ? ldexpf(1.0f, 14 - e) : 1.0f;
        if (lane == 0) qscale[r] = s;
        for (int d = lane; d < AQ_LD - 8; d += 32) {
            float v = ok && d < hd ? qr[d] * s : 0.0f;
            __half hi = __float2half(v);
            Qs[r * AQ_LD + d] = hi;
            Qlo[r * AQ_LD + d] = __float2half(v - __half2float(hi));
        }
    }
    // the warp's 16 rows: g = lane / 4 holds rows g and g + 8 of each tile
    const int r0 = warp * 16 + (lane >> 2), r1 = r0 + 8;
    auto rowpos = [&](int r, int *lo, int *hi_) {
        int hi = r / qs, ti = r - hi * qs;
        bool ok = hi < nh && ti < nt;
        int pr = pos0 + tk0 + ti;
        int t0r = (a.window > 0 && pr - a.window + 1 > 0) ? pr - a.window + 1 : 0;
        *lo = ok ? t0r : 1;
        *hi_ = ok ? pr : 0;                              // empty range for unused rows
    };
    int lo0, hi0, lo1, hi1;
    rowpos(r0, &lo0, &hi0);
    rowpos(r1, &lo1, &hi1);
    const int plast = pos0 + tk0 + nt - 1;
    int klo = 0;
    if (a.window > 0) { int f = pos0 + tk0 - a.window + 1; klo = f > 0 ? f : 0; }
    const int nkh = hd / 8;      // k8 steps over head_dim
    __syncthreads();
    const float qs0 = qscale[r0], qs1 = qscale[r1];
    auto stage = [&](const unsigned char *cache, ulong64 b0, int kt) {
        // 64 rows x hd halves, 16 bytes a thread a step
        const int per_row = hd / 8;
        for (int i = tid; i < AQ_TK * per_row; i += 128) {
            int k = i / per_row, c = i - k * per_row, t = kt + k;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t <= plast) v = *(const uint4 *)(cache + b0 + kv_slot(t, a.ring) * row_b + (ulong64)c * 16);
            *(uint4 *)(KVs + k * AQ_LD + c * 8) = v;
        }
    };
    // S for this warp's 16 rows against the 64 staged keys: 8 n8 tiles
    auto scores = [&](float (&S)[8][4]) {
        #pragma unroll
        for (int j = 0; j < 8; j++) S[j][0] = S[j][1] = S[j][2] = S[j][3] = 0.0f;
        for (int kk = 0; kk < nkh; kk += 2) {          // k16 per step
            unsigned qa[4], ql[4];
            g2_ldsm_x4(qa[0], qa[1], qa[2], qa[3],
                       Qs + (warp * 16 + (lane & 15)) * AQ_LD + kk * 8 + (lane >> 4) * 8);
            g2_ldsm_x4(ql[0], ql[1], ql[2], ql[3],
                       Qlo + (warp * 16 + (lane & 15)) * AQ_LD + kk * 8 + (lane >> 4) * 8);
            #pragma unroll
            for (int np = 0; np < 4; np++) {           // two n8 key tiles a load
                unsigned kb[4];
                g2_ldsm_x4(kb[0], kb[1], kb[2], kb[3],
                           KVs + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * AQ_LD
                               + kk * 8 + ((lane >> 3) & 1) * 8);
                g2_mma(S[2 * np], qa[0], qa[1], kb[0]);
                g2_mma(S[2 * np], qa[2], qa[3], kb[1]);
                g2_mma(S[2 * np], ql[0], ql[1], kb[0]);
                g2_mma(S[2 * np], ql[2], ql[3], kb[1]);
                g2_mma(S[2 * np + 1], qa[0], qa[1], kb[2]);
                g2_mma(S[2 * np + 1], qa[2], qa[3], kb[3]);
                g2_mma(S[2 * np + 1], ql[0], ql[1], kb[2]);
                g2_mma(S[2 * np + 1], ql[2], ql[3], kb[3]);
            }
        }
    };
    // ---- pass 1: row maxima ----
    float m0 = -1e30f, m1 = -1e30f;
    for (int kt = klo; kt <= plast; kt += AQ_TK) {
        __syncthreads();
        stage(kc, base, kt);
        __syncthreads();
        float S[8][4];
        scores(S);
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int t = kt + j * 8 + (lane & 3) * 2 + (e & 1);
                bool top = e < 2;
                int lo = top ? lo0 : lo1, hi = top ? hi0 : hi1;
                if (t >= lo && t <= hi) {
                    float v = S[j][e] / (top ? qs0 : qs1) * a.scale;
                    if (top) m0 = fmaxf(m0, v); else m1 = fmaxf(m1, v);
                }
            }
    }
    for (int off = 1; off < 4; off <<= 1) {
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, off));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, off));
    }
    if (sinks) {
        int hiA = r0 / qs, hiB = r1 / qs;
        if (hiA < nh && sinks[h0 + hiA] > m0) m0 = sinks[h0 + hiA];
        if (hiB < nh && sinks[h0 + hiB] > m1) m1 = sinks[h0 + hiB];
    }
    // ---- pass 2: probabilities, sums, O = P V ----
    float O[16][4];                                     // hd / 8 n8 tiles (hd <= 128)
    #pragma unroll
    for (int j = 0; j < 16; j++) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.0f;
    float l0 = 0.0f, l1 = 0.0f;
    for (int kt = klo; kt <= plast; kt += AQ_TK) {
        __syncthreads();
        stage(kc, base, kt);
        __syncthreads();
        float S[8][4];
        scores(S);
        unsigned pa[8][2];                              // P as A operands, one k8 key chunk each
        unsigned pl[8][2];                              // P's low halves
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float p[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int t = kt + j * 8 + (lane & 3) * 2 + (e & 1);
                bool top = e < 2;
                int lo = top ? lo0 : lo1, hi = top ? hi0 : hi1;
                p[e] = (t >= lo && t <= hi)
                     ? expf(S[j][e] / (top ? qs0 : qs1) * a.scale - (top ? m0 : m1)) : 0.0f;
            }
            l0 += p[0] + p[1];
            l1 += p[2] + p[3];
            pa[j][0] = g2_pack(p[0], p[1]);
            pa[j][1] = g2_pack(p[2], p[3]);
            pl[j][0] = g2_pack(p[0] - __half2float(__float2half(p[0])),
                               p[1] - __half2float(__float2half(p[1])));
            pl[j][1] = g2_pack(p[2] - __half2float(__float2half(p[2])),
                               p[3] - __half2float(__float2half(p[3])));
        }
        __syncthreads();
        stage(vc, base_v, kt);
        __syncthreads();
        // V as B operands: ldmatrix.trans of [key][d], one k8 key chunk x
        // two n8 d tiles per load
        #pragma unroll
        for (int dp = 0; dp < 16; dp += 4) {        // compile-time trip, so O stays in registers
            if (dp >= nkh) break;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned vb[4];
                unsigned addr = (unsigned)__cvta_generic_to_shared(
                    KVs + (j * 8 + (lane & 7)) * AQ_LD + dp * 8 + (lane >> 3) * 8);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3]) : "r"(addr));
                // vb[i]: d tile dp + i (8 wide) for this key chunk; x4 covers 4 d tiles
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    if (dp + i < nkh) {
                        g2_mma(O[dp + i], pa[j][0], pa[j][1], vb[i]);
                        g2_mma(O[dp + i], pl[j][0], pl[j][1], vb[i]);
                    }
            }
        }
    }
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    if (sinks) {
        int hiA = r0 / qs, hiB = r1 / qs;
        if (hiA < nh) l0 += expf(sinks[h0 + hiA] - m0);
        if (hiB < nh) l1 += expf(sinks[h0 + hiB] - m1);
    }
    // ---- O / sum into the row's token and head ----
    #pragma unroll
    for (int half_ = 0; half_ < 2; half_++) {
        int r = half_ ? r1 : r0;
        int hi = r / qs, ti = r - hi * qs;
        if (hi >= nh || ti >= nt) continue;
        float inv = 1.0f / (half_ ? l1 : l0);
        float *o = out + (ulong64)(tk0 + ti) * a.os + (ulong64)(h0 + hi) * hd;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            if (j >= nkh) break;
            int d = j * 8 + (lane & 3) * 2;
            o[d]     = O[j][half_ * 2 + 0] * inv;
            o[d + 1] = O[j][half_ * 2 + 1] * inv;
        }
    }
}

// ------------------------------------------------ decode KV row gather (X3)
// RUNNER_CUDA_KV_GATHER (cuda.c kv_gather_*): the host KV cache stays
// authoritative, so every decoded token's K and V rows go back to it. That
// was two synchronous copies per offloaded layer (and a whole ring for a ring
// layer); this kernel, the last node of the decode graph, copies each
// layer's single new row (slot kv_slot(pos)) of both caches into one staging
// buffer, which one copy then brings to the host. Bytes move unchanged.
// tab: per layer {k byte offset, v byte offset, k row bytes, v row bytes,
// ring rows, staging offset}; grid (n_layers, 2), 256 threads.
extern "C" __global__ void k_kv_gather(const unsigned char *kc, const unsigned char *vc,
                                       const kvg_entry *tab, const int *posp,
                                       unsigned char *out) {
    const kvg_entry e = tab[blockIdx.x];
    const bool v = blockIdx.y == 1;
    const unsigned n = v ? e.vrow : e.krow;
    const ulong64 slot = kv_slot(*posp, e.ring);
    const unsigned char *src = (v ? vc + e.voff : kc + e.koff) + slot * n;
    unsigned char *dst = out + e.stage + (v ? e.krow : 0u);
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// ---------------------------------------- whole-window recurrent prefill (C2.3)
// RUNNER_CUDA_RECUR_SEQ (cuda.c gpu_q35_recurrent / gpu_mamba2_recurrent):
// the per-token conv and state kernels above, run over all tn columns of a
// prompt tile in ONE launch each. A thread keeps what it carried between
// launches (its conv history, its state row) in registers across the tile and
// writes it back once at the end, instead of a launch per token that reloads
// and stores the whole state through global memory. The per-token arithmetic
// is the per-token kernels' statement for statement (same reductions, same
// block widths, same order), so the outputs and the state after the tile are
// byte-equal to tn per-token launches; `make test-cuda-qwen4exp` and
// test-recurrent-rewind hold the state, GPU_ID_RECUR the logits. Conv kernels
// up to RS_MAX_HIST history taps; state rows up to RS_MAX_SD.
#define RS_MAX_HIST 7
#define RS_MAX_SD 128

extern "C" __global__ void k_q35_conv_seq(const float *mix, float *cv,
                                           const float *weight, float *history,
                                           int convdim, int kernel, int tn) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= convdim) return;
    int histn = kernel - 1;
    const float *cw = weight + (ulong64)c * kernel;
    float h[RS_MAX_HIST];
    #pragma unroll
    for (int k = 0; k < RS_MAX_HIST; k++)
        if (k < histn) h[k] = history[(ulong64)k * convdim + c];
    for (int t = 0; t < tn; t++) {
        float mx = mix[(ulong64)t * convdim + c];
        float sum = cw[histn] * mx;
        #pragma unroll
        for (int k = 0; k < RS_MAX_HIST; k++)
            if (k < histn) sum += cw[k] * h[k];
        cv[(ulong64)t * convdim + c] = sum / (1.0f + expf(-sum));
        #pragma unroll
        for (int k = 0; k + 1 < RS_MAX_HIST; k++)
            if (k + 1 < histn) h[k] = h[k + 1];
        #pragma unroll
        for (int k = 0; k < RS_MAX_HIST; k++)
            if (histn && k == histn - 1) h[k] = mx;
    }
    #pragma unroll
    for (int k = 0; k < RS_MAX_HIST; k++)
        if (k < histn) history[(ulong64)k * convdim + c] = h[k];
}

// k_q35_delta over tn tokens: one block per value head, 128 threads, thread j
// holding state row j (state_dim <= RS_MAX_SD, so at most one row a thread,
// as in k_q35_delta). cv/z/beta/alpha are [tn] rows of their per-token
// strides, out rows are xdim apart.
extern "C" __global__ void __launch_bounds__(128)
k_q35_delta_seq(const float *cv, const float *z, const float *beta_in,
                const float *alpha_in, const float *dt, const float *a,
                const float *norm, float *state, float *out, int state_dim,
                int groups, int heads, float eps, int gate_sigmoid, int tn,
                int convdim, int inner, int xdim) {
    __shared__ float red[256];
    int tid = threadIdx.x, h = blockIdx.x;
    if (h >= heads) return;
    int keydim = state_dim * groups;
    int group = h % groups;
    float *st = state + (ulong64)h * state_dim * state_dim;
    const bool owner = tid < state_dim;
    float row[RS_MAX_SD];
    if (owner) {
        #pragma unroll
        for (int i = 0; i < RS_MAX_SD; i++)
            if (i < state_dim) row[i] = st[(ulong64)tid * state_dim + i];
    }
    for (int t = 0; t < tn; t++) {
        const float *cvt = cv + (ulong64)t * convdim;
        const float *q = cvt + group * state_dim;
        const float *k = cvt + keydim + group * state_dim;
        const float *v = cvt + 2 * keydim + h * state_dim;
        float *yo = out + (ulong64)t * xdim + h * state_dim;

        float qs = 0.0f;
        for (int i = tid; i < state_dim; i += blockDim.x) qs += q[i] * q[i];
        red[tid] = qs;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        float qscale = rsqrtf(red[0] + eps);
        __syncthreads();
        float ks = 0.0f;
        for (int i = tid; i < state_dim; i += blockDim.x) ks += k[i] * k[i];
        red[tid] = ks;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        float kscale = rsqrtf(red[0] + eps);
        float beta = 1.0f / (1.0f + expf(-beta_in[(ulong64)t * heads + h]));
        float av = alpha_in[(ulong64)t * heads + h] + dt[h];
        float softplus = av > 20.0f ? av : log1pf(expf(av));
        float decay = expf(a[h] * softplus);

        if (owner) {
            int j = tid;
            float pred = 0.0f;
            #pragma unroll
            for (int i = 0; i < RS_MAX_SD; i++) {
                if (i < state_dim) {
                    row[i] *= decay;
                    pred += row[i] * (k[i] * kscale);
                }
            }
            float delta = (v[j] - pred) * beta;
            float y = 0.0f;
            #pragma unroll
            for (int i = 0; i < RS_MAX_SD; i++) {
                if (i < state_dim) {
                    row[i] += delta * (k[i] * kscale);
                    y += row[i] * (q[i] * qscale);
                }
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
        const float *zt = z + (ulong64)t * inner;
        for (int j = tid; j < state_dim; j += blockDim.x) {
            int o = h * state_dim + j;
            float zv = zt[o];
            float sg = 1.0f / (1.0f + expf(-zv));
            yo[j] = yo[j] * rms * norm[j] * (gate_sigmoid ? sg : zv * sg);
        }
        __syncthreads();   // red[] and yo[] are reused by the next token
    }
    if (owner) {
        #pragma unroll
        for (int i = 0; i < RS_MAX_SD; i++)
            if (i < state_dim) st[(ulong64)tid * state_dim + i] = row[i];
    }
}

// k_mamba2_conv over tn tokens (one thread per channel, its ring column in
// registers)
extern "C" __global__ void k_mamba2_conv_seq(
        const float *proj, float *xBC, const float *conv_w, const float *conv_b,
        float *ring, int conv_dim, int inner, int d_in_proj,
        int conv_kernel, int histn, int tn) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const float *cw = conv_w + (size_t)c * conv_kernel;
    float r[RS_MAX_HIST];
    #pragma unroll
    for (int k = 0; k < RS_MAX_HIST; k++)
        if (k < histn) r[k] = ring[(size_t)k * conv_dim + c];
    for (int t = 0; t < tn; t++) {
        float cur = proj[(size_t)t * d_in_proj + inner + c];
        float sum = conv_b[c] + cw[histn] * cur;
        #pragma unroll
        for (int k = 0; k < RS_MAX_HIST; k++)
            if (k < histn) sum += cw[k] * r[k];
        xBC[(size_t)t * conv_dim + c] = sum / (1.0f + expf(-sum));
        #pragma unroll
        for (int k = 0; k + 1 < RS_MAX_HIST; k++)
            if (k < histn - 1) r[k] = r[k + 1];
        #pragma unroll
        for (int k = 0; k < RS_MAX_HIST; k++)
            if (histn && k == histn - 1) r[k] = cur;
    }
    #pragma unroll
    for (int k = 0; k < RS_MAX_HIST; k++)
        if (k < histn) ring[(size_t)k * conv_dim + c] = r[k];
}

// k_mamba2_ssd over tn tokens: thread (h, p) keeps its d_state row in
// registers (ds <= RS_MAX_SD)
extern "C" __global__ void k_mamba2_ssd_seq(
        const float *xBC, const float *proj, const float *A, const float *dt_b,
        const float *D, float *state, float *y,
        int inner, int hd, int ds, int ng, int gsz, int conv_dim,
        int d_in_proj, int tn) {
    int h = blockIdx.x;
    int p = threadIdx.x;
    if (p >= hd) return;
    int g = h / gsz;
    float *sp = state + ((size_t)h * hd + p) * ds;
    float s_[RS_MAX_SD];
    #pragma unroll
    for (int j = 0; j < RS_MAX_SD; j++)
        if (j < ds) s_[j] = sp[j];
    for (int t = 0; t < tn; t++) {
        float dtr = proj[(size_t)t * d_in_proj + inner + conv_dim + h] + dt_b[h];
        float dt = dtr > 20.0f ? dtr : logf(1.0f + expf(dtr));
        float dA = expf(dt * A[h]);
        const float *xrow = xBC + (size_t)t * conv_dim;
        const float *Bg = xrow + inner + (size_t)g * ds;
        const float *Cg = xrow + inner + (size_t)ng * ds + (size_t)g * ds;
        float x = xrow[(size_t)h * hd + p];
        float x_dt = x * dt;
        float sumf = 0.0f;
        #pragma unroll
        for (int j = 0; j < RS_MAX_SD; j++) {
            if (j < ds) {
                float s = s_[j] * dA + Bg[j] * x_dt;
                sumf += s * Cg[j];
                s_[j] = s;
            }
        }
        y[(size_t)t * inner + (size_t)h * hd + p] = sumf + D[h] * x;
    }
    #pragma unroll
    for (int j = 0; j < RS_MAX_SD; j++)
        if (j < ds) sp[j] = s_[j];
}
