// Metal compute kernels: the full single-token forward pass.
// Mirrors the CPU implementations in quants.c/model.c bit-layout for bit-layout.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------- rmsnorm

// One threadgroup per column: a prompt batch normalizes every token in one
// dispatch instead of n. Strides are explicit because x and y scratch can be
// strided differently (n_embd vs xdim).
struct norm_args { int n, x_stride, y_stride; float eps; };

kernel void k_rmsnorm(device const float *x_all [[buffer(0)]],
                      device float       *y_all [[buffer(1)]],
                      device const float *w     [[buffer(2)]],
                      constant norm_args &a     [[buffer(3)]],
                      uint3 tid3 [[thread_position_in_threadgroup]],
                      uint3 tpg3 [[threads_per_threadgroup]],
                      uint3 tgpig [[threadgroup_position_in_grid]]) {
    threadgroup float red[256];
    uint tid = tid3.x, tpg = tpg3.x;
    int n = a.n;
    float eps = a.eps;
    device const float *x = x_all + (ulong)tgpig.x * a.x_stride;
    device float       *y = y_all + (ulong)tgpig.x * a.y_stride;
    float s = 0;
    for (int i = tid; i < n; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(red[0] / n + eps);
    for (int i = tid; i < n; i += tpg) y[i] = x[i] * r * w[i];
}

// per-head RMSNorm (qwen3 Q/K norm): one threadgroup per head
// grid.y is the prompt batch: threadgroup (h, col) normalizes head h of column
// col. Each (head, column) pair was already an independent reduction when this
// was encoded once per token, so taking the batch in grid.y is BIT-IDENTICAL,
// not a tolerance trade -- the per-pair arithmetic is character for character
// the same and only the encoding changes. `stride` is elements between columns
// (q_dim for Q, kv_dim for K); at n_col == 1 it is never read.
kernel void k_qknorm(device float       *v   [[buffer(0)]],
                     device const float *w   [[buffer(1)]],
                     constant int       &hd  [[buffer(2)]],
                     constant float     &eps [[buffer(3)]],
                     constant int       &stride [[buffer(4)]],
                     uint2 tgpig [[threadgroup_position_in_grid]],
                     uint2 tpitg [[thread_position_in_threadgroup]],
                     uint2 ntg   [[threads_per_threadgroup]]) {
    uint tid = tpitg.x, tpg = ntg.x;
    threadgroup float red[128];
    device float *x = v + (ulong)tgpig.y * stride + tgpig.x * hd;
    float s = 0;
    for (int i = tid; i < hd; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(red[0] / hd + eps);
    for (int i = tid; i < hd; i += tpg) x[i] = x[i] * r * w[i];
}

kernel void k_head_rmsnorm(device const float *src [[buffer(0)]],
                           device float       *dst [[buffer(1)]],
                           device const float *w   [[buffer(2)]],
                           constant int       &hd  [[buffer(3)]],
                           constant float     &eps [[buffer(4)]],
                           constant int       &has_weight [[buffer(5)]],
                           constant int       &stride [[buffer(6)]],
                           uint2 tgpig [[threadgroup_position_in_grid]],
                           uint2 tpitg [[thread_position_in_threadgroup]],
                           uint2 ntg   [[threads_per_threadgroup]]) {
    uint tid = tpitg.x, tpg = ntg.x;
    threadgroup float red[128];
    // Batched in grid.y for the same reason, and with the same bit-identity
    // argument, as k_qknorm above. src and dst share a stride: both the K and V
    // staging buffers are strided by kv_dim.
    ulong col = (ulong)tgpig.y * stride;
    device const float *x = src + col + tgpig.x * hd;
    device float *y = dst + col + tgpig.x * hd;
    float s = 0;
    for (int i = tid; i < hd; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(red[0] / hd + eps);
    for (int i = tid; i < hd; i += tpg)
        y[i] = x[i] * r * (has_weight ? w[i] : 1.0f);
}

// ---------------------------------------------------------------- matvec
// One simdgroup (32 lanes) per output row; lanes stride over blocks.
//
// n_col > 1 turns this into a matmul: the simdgroup walks its weight row once
// per column, so for a prompt batch the row is fetched from device memory on
// the first column and served from cache for the rest — prefill stops paying
// the whole weight matrix per token. The arithmetic per output element is
// character-for-character what n_col == 1 does (same lane striding, same
// simd_sum), so batched prefill stays bit-identical to per-token submits,
// which is what the CPU==GPU gate requires.
//
// col_tile trades the two off: one threadgroup per row over ALL columns gets
// maximum cache reuse but serializes the batch and starves the GPU of
// threadgroups; col_tile columns per threadgroup keeps grid.y parallelism
// while still amortizing each weight fetch col_tile ways.
//
// x and y strides are explicit because they are NOT always n_in/n_out: the
// xb scratch is strided by xdim = max(q_dim, n_embd).

struct mv_args {
    int   n_in;
    int   n_out;
    ulong w_off;      // tensor byte offset inside the weight buffer
    int   has_bias;
    int   n_col;      // columns (prompt tokens) processed by this dispatch
    int   x_stride;   // elements between consecutive columns of x
    int   y_stride;   // elements between consecutive columns of y
    int   col_tile;   // columns per threadgroup (grid.y tiles the rest)
};

// The loop bound is uniform across the simdgroup, so every lane reaches each
// simd_sum in MV_TAIL the same number of times — a divergent simd_sum would
// be undefined.
#define MV_HEAD \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    if (row >= (uint)a.n_out) return; \
    int col_lo = (int)tgpig.y * a.col_tile; \
    int col_hi = min(col_lo + a.col_tile, a.n_col); \
    for (int col = col_lo; col < col_hi; col++) { \
    device const float *x = x_all + (ulong)col * a.x_stride; \
    device float       *y = y_all + (ulong)col * a.y_stride;

#define MV_TAIL \
    s = simd_sum(s); \
    if (tiisg == 0) y[row] = a.has_bias ? s + bias[row] : s; \
    }

#define MV_PARAMS \
    device const uchar *wb    [[buffer(0)]], \
    device const float *x_all [[buffer(1)]], \
    device float       *y_all [[buffer(2)]], \
    constant mv_args   &a     [[buffer(3)]], \
    device const float *bias  [[buffer(4)]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]], \
    uint  tiisg [[thread_index_in_simdgroup]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint3 ntg   [[threads_per_threadgroup]]

kernel void k_mv_f32(MV_PARAMS) {
    MV_HEAD;
    device const float *rw = (device const float *)(wb + a.w_off) + (ulong)row * a.n_in;
    float s = 0;
    for (int i = tiisg; i < a.n_in; i += 32) s += rw[i] * x[i];
    MV_TAIL;
}

// Widened for the same reason as k_mv_bf16 above: one half per lane step left
// f16 decode slower on the GPU than on the CPU.
kernel void k_mv_f16(MV_PARAMS) {
    MV_HEAD;
    device const half *rw = (device const half *)(wb + a.w_off) + (ulong)row * a.n_in;
    float s = 0;
    if ((a.n_in & 3) == 0) {
        device const packed_half4 *w4 = (device const packed_half4 *)rw;
        device const packed_float4 *x4 = (device const packed_float4 *)x;
        int n4 = a.n_in >> 2;
        float4 acc = 0;
        for (int i = tiisg; i < n4; i += 32) acc += float4(w4[i]) * x4[i];
        s = acc.x + acc.y + acc.z + acc.w;
    } else {
        for (int i = tiisg; i < a.n_in; i += 32) s += (float)rw[i] * x[i];
    }
    MV_TAIL;
}

// Widened loads, unchanged arithmetic.
//
// The scalar forms below issued one memory instruction per BYTE of quant and
// one per float of activation: 48 loads to consume an 18-byte q4_0 block, 64
// for a 34-byte q8_0 one. Reading the same bytes as uchar4/float4 cuts that
// roughly fourfold and is the only bandwidth lever this route has — the lane
// that owns a block is pinned by the identity contract (see MV_HEAD), so the
// work cannot be redistributed.
//
// The rule that makes it safe: an accumulating EXPRESSION is copied character
// for character from the scalar form, and only the way its operands are LOADED
// changes. `t += A * xp[j] + B * xp[j + 16]` keeps that exact shape with xp[j]
// spelled as a component of an already-loaded float4, because the Metal
// compiler contracts multiply-add by expression shape. Resurfacing the same
// products through a float4 accumulator and a horizontal sum rounds
// differently — that is what the 2026-08-07 q4_K attempt did, and it was both
// slower and off-contract.
//
// What it is worth, so nobody re-derives it: on an 8-core M1 this is NEUTRAL.
// Five interleaved rounds against e2b-q40 — 2.6 GB of q4_0, the one local
// model actually bandwidth-bound at ~40 of the M1's ~68 GB/s — read 15.14
// tok/s against 15.11 for the byte-at-a-time form, +0.2%, well inside the
// run-to-run spread. SmolLM2-135M-Q8_0 cannot see it at all (107.13 vs 106.68,
// +0.4%): 145 MB at ~107 tok/s is 19 GB/s, i.e. dispatch-bound, not
// bandwidth-bound. It is kept because it is a strict reduction in issued loads
// at identical arithmetic, not because it was measured faster here.
// See docs/negative-result-metal-multirow-matvec.md for the rest of the sweep,
// including the multi-row arrangement that measured WORSE and is not here.
kernel void k_mv_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 34;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        // packed_char4: the quants start two bytes into a 34-byte block, so
        // nothing about their address is 4-byte aligned.
        device const packed_char4 *q = (device const packed_char4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float t = 0;
        for (int k = 0; k < 8; k++) {
            char4 qq = q[k];
            float4 xv = xp[k];
            t += (float)qq.x * xv.x;
            t += (float)qq.y * xv.y;
            t += (float)qq.z * xv.z;
            t += (float)qq.w * xv.w;
        }
        s += d * t;
    }
    MV_TAIL;
}

kernel void k_mv_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 18;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const packed_uchar4 *q = (device const packed_uchar4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float t = 0;
        for (int k = 0; k < 4; k++) {
            uchar4 qq = q[k];
            float4 xl = xp[k], xh = xp[k + 4];
            t += ((int)(qq.x & 0xF) - 8) * xl.x + ((int)(qq.x >> 4) - 8) * xh.x;
            t += ((int)(qq.y & 0xF) - 8) * xl.y + ((int)(qq.y >> 4) - 8) * xh.y;
            t += ((int)(qq.z & 0xF) - 8) * xl.z + ((int)(qq.z >> 4) - 8) * xh.z;
            t += ((int)(qq.w & 0xF) - 8) * xl.w + ((int)(qq.w >> 4) - 8) * xh.w;
        }
        s += d * t;
    }
    MV_TAIL;
}

kernel void k_mv_q4_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 20;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 20;
        float d = (float)*(device const half *)blk;
        float mm = (float)*(device const half *)(blk + 2);
        device const uchar *q = blk + 4;
        device const float *xp = x + b * 32;
        float t = 0, sx = 0;
        for (int j = 0; j < 16; j++) {
            t += (float)(q[j] & 0xF) * xp[j] + (float)(q[j] >> 4) * xp[j + 16];
            sx += xp[j] + xp[j + 16];
        }
        s += d * t + mm * sx;
    }
    MV_TAIL;
}

kernel void k_mv_q5_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 22;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 22;
        float d = (float)*(device const half *)blk;
        uint qh = (uint)blk[2] | ((uint)blk[3] << 8) |
                  ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        device const uchar *q = blk + 6;
        device const float *xp = x + b * 32;
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

kernel void k_mv_q5_1(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 24;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 24;
        float d = (float)*(device const half *)blk;
        float mm = (float)*(device const half *)(blk + 2);
        uint qh = (uint)blk[4] | ((uint)blk[5] << 8) |
                  ((uint)blk[6] << 16) | ((uint)blk[7] << 24);
        device const uchar *q = blk + 8;
        device const float *xp = x + b * 32;
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

static inline void get_scale_min_k4(int j, device const uchar *q,
                                    thread uchar *d, thread uchar *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j    ] >> 6) << 4);
    }
}

// bf16 is the top 16 bits of an f32, so widening is a shift and a reinterpret
// with no rounding anywhere -- exactly what bf16_to_f32() in fp16.h does, which
// is why this path can be bit-identical to the CPU rather than merely close.
// Metal has no bfloat on every target this ships to, so the weights are read as
// ushort and widened by hand.
static inline float bf16_to_f32_m(ushort h) {
    return as_type<float>((uint)h << 16);
}

// Four weights per lane step, not one. The scalar form left this kernel LOSING
// to the CPU (measured 0.91x on a 135M model) while same-size quantised models
// won 1.5-2.1x on the same device -- the difference was never model size, it
// was that every quant kernel here already loads wide and these did not.
// packed_ushort4 because the weight pointer is row-offset and carries no
// 8-byte alignment guarantee; x is float4-indexed the way k_mv_q5_K does it.
kernel void k_mv_bf16(MV_PARAMS) {
    MV_HEAD;
    device const ushort *rw =
        (device const ushort *)(wb + a.w_off) + (ulong)row * a.n_in;
    float s = 0;
    if ((a.n_in & 3) == 0) {
        device const packed_ushort4 *w4 = (device const packed_ushort4 *)rw;
        device const packed_float4 *x4 = (device const packed_float4 *)x;
        int n4 = a.n_in >> 2;
        float4 acc = 0;
        for (int i = tiisg; i < n4; i += 32) {
            ushort4 h = w4[i];
            acc += float4(as_type<float>((uint)h.x << 16),
                          as_type<float>((uint)h.y << 16),
                          as_type<float>((uint)h.z << 16),
                          as_type<float>((uint)h.w << 16)) * x4[i];
        }
        s = acc.x + acc.y + acc.z + acc.w;
    } else {
        for (int i = tiisg; i < a.n_in; i += 32) s += bf16_to_f32_m(rw[i]) * x[i];
    }
    MV_TAIL;
}

// The IQ4 non-linear codebook. Both iq4_nl and iq4_xs index it with a plain
// 4-bit code, which is the whole difference from q4_0: same 16 codes per byte
// pair, but the value is looked up rather than being the code minus eight.
// Must stay identical to kvalues_iq4nl in quants.c -- these kernels are gated
// on producing bit-identical output to that path.
constant int kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// iq4_nl: 18-byte blocks of 32 weights, one f16 scale, no sub-block scales.
kernel void k_mv_iq4_nl(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 18;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        device const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += (float)kvalues_iq4nl[q[j] & 0xF] * xp[j]
               + (float)kvalues_iq4nl[q[j] >> 4]  * xp[j + 16];
        s += d * t;
    }
    MV_TAIL;
}

// iq4_xs: 136-byte superblocks of 256, worked one 32-weight sub-block per lane
// iteration. Each sub-block carries a 6-bit scale split across two places --
// four low bits packed two-per-byte in scales_l, and the top two bits in a
// 16-bit scales_h field -- biased by 32 like q3_K's. scales_h is read as two
// bytes rather than a ushort because a 136-byte block gives offset 2 no
// two-byte alignment guarantee.
kernel void k_mv_iq4_xs(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 136;
    float s = 0;
    int nq = nb * 8;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 3, ib = u & 7;
        device const uchar *blk = rw + (ulong)b * 136;
        float d = (float)*(device const half *)blk;
        uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
        int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) |
                 (((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        device const uchar *q  = blk + 8 + ib * 16;
        device const float *xp = x + b * 256 + ib * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += (float)kvalues_iq4nl[q[j] & 0xF] * xp[j]
               + (float)kvalues_iq4nl[q[j] >> 4]  * xp[j + 16];
        s += dl * t;
    }
    MV_TAIL;
}

// ---------------------------------------------------------------- codebook i-quants
// The seven codebook formats (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S). Every block holds 256 weights as eight 32-weight sub-blocks, each
// four groups of 8 whose magnitudes come from a shared grid indexed by the
// packed bits; the group's signs come from a 7-bit parity-expanded index or
// a plain sign byte, and the IQ1 grids are signed (-1/0/1) with a shared
// 1/8 delta. The arithmetic is src/iq_decode.h's, the file the CUDA kernels
// and the host test compile; the tables below are the third copy of
// src/quants_iq_grids.h (device code cannot read the host arrays) and
// tests/test_metal_iq_kernels.py holds them element-for-element equal to
// the host, the same way tests/test_cuda_iq_grids.py holds the CUDA copy.
// Data, not code: do not edit the tables by hand.
constant ulong kiq2xxs_grid[256] = {
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
constant ulong kiq2xs_grid[512] = {
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
constant ulong kiq2s_grid[1024] = {
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
constant uint kiq3xxs_grid[256] = {
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
constant uint kiq3s_grid[512] = {
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
constant ulong kiq1s_grid[2048] = {
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

// ksigns_iq2xs[i] is i with its odd parity in bit 7 (the host test pins the
// table to this identity), so a 7-bit sign index expands arithmetically.
static inline uint iq_signs7(uint idx) { return idx | ((popcount(idx) & 1u) << 7); }
static inline float iq_w8(ulong grid, int j, uint signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return ((signs >> j) & 1) ? -mag : mag;
}
static inline float iq_w4(uint grid, int j, uint signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return ((signs >> j) & 1) ? -mag : mag;
}
static inline float iq1_w(ulong grid, int j, float delta) {
    return (float)(char)((grid >> (8 * j)) & 0xFF) + delta;
}
#define IQ1_DELTA 0.125f
static inline uint iq_ld16(device const uchar *p) { return (uint)p[0] | ((uint)p[1] << 8); }
static inline uint iq_ld32(device const uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}
static inline float iq_f16(device const uchar *p) { return (float)*(device const half *)p; }
static inline float iq_f16_bits(uint h) { return (float)as_type<half>((ushort)h); }

// Each decoder writes the 8 weights of group l (0..3) of sub-block ib (0..7)
// of one 256-weight block, scale folded in: element 32*ib + 8*l + j.

// IQ2_XXS, 66 bytes: per sub-block eight bytes at 2 + 8*ib, four grid
// indices then a word of four 7-bit sign indices and a 4-bit scale.
static inline void iq2xxs_dec8(device const uchar *blk, int ib, int l, thread float *w) {
    float d = iq_f16(blk);
    device const uchar *qs = blk + 2 + 8 * ib;
    uint aux = iq_ld32(qs + 4);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
    ulong g = kiq2xxs_grid[qs[l]];
    uint signs = iq_signs7((aux >> (7 * l)) & 127);
    for (int j = 0; j < 8; j++) w[j] = db * iq_w8(g, j, signs);
}

// IQ2_XS, 74 bytes: per sub-block four 16-bit words at 2 + 8*ib (9-bit
// grid index, 7-bit sign index) and a scale byte at 66 + ib (two nibbles).
static inline void iq2xs_dec8(device const uchar *blk, int ib, int l, thread float *w) {
    float d = iq_f16(blk);
    device const uchar *qs = blk + 2 + 8 * ib;
    uint sc = blk[66 + ib];
    float db = d * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
    uint q = iq_ld16(qs + 2 * l);
    ulong g = kiq2xs_grid[q & 511];
    uint signs = iq_signs7(q >> 9);
    for (int j = 0; j < 8; j++) w[j] = db * iq_w8(g, j, signs);
}

// IQ2_S, 82 bytes: four low index bytes at 2 + 4*ib, four sign bytes at
// 34 + 4*ib, two high index bits per index in the byte at 66 + ib, two
// scale nibbles at 74 + ib.
static inline void iq2s_dec8(device const uchar *blk, int ib, int l, thread float *w) {
    float d = iq_f16(blk);
    device const uchar *qs = blk + 2 + 4 * ib, *sg = blk + 34 + 4 * ib;
    uint hb = blk[66 + ib], sc = blk[74 + ib];
    float db = d * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
    ulong g = kiq2s_grid[qs[l] | ((hb << (8 - 2 * l)) & 0x300)];
    uint signs = sg[l];
    for (int j = 0; j < 8; j++) w[j] = db * iq_w8(g, j, signs);
}

// IQ3_XXS, 98 bytes: eight grid indices per sub-block at 2 + 8*ib (four
// magnitudes each), a word of four sign indices and a 4-bit scale at
// 66 + 4*ib, scale (0.5 + nibble) / 2.
static inline void iq3xxs_dec8(device const uchar *blk, int ib, int l, thread float *w) {
    float d = iq_f16(blk);
    device const uchar *qs = blk + 2 + 8 * ib;
    uint aux = iq_ld32(blk + 66 + 4 * ib);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    uint g1 = kiq3xxs_grid[qs[2 * l]], g2 = kiq3xxs_grid[qs[2 * l + 1]];
    uint signs = iq_signs7((aux >> (7 * l)) & 127);
    for (int j = 0; j < 4; j++) {
        w[j]     = db * iq_w4(g1, j, signs);
        w[4 + j] = db * iq_w4(g2, j, signs >> 4);
    }
}

// IQ3_S, 110 bytes: sub-block pair p = ib/2 shares scale nibbles at 106 + p
// (1 + 2*nibble, low nibble first), one high-bit byte per sub-block at
// 66 + 2p + h, indices at 2 + 16p + 8h, sign bytes at 74 + 8p + 4h.
static inline void iq3s_dec8(device const uchar *blk, int ib, int l, thread float *w) {
    float d = iq_f16(blk);
    int p = ib >> 1, h = ib & 1;
    device const uchar *q = blk + 2 + 16 * p + 8 * h, *sgn = blk + 74 + 8 * p + 4 * h;
    uint hb = blk[66 + 2 * p + h], scb = blk[106 + p];
    float db = d * (float)(1 + 2 * (h ? (scb >> 4) : (scb & 0xF)));
    uint g1 = kiq3s_grid[q[2 * l]     | ((hb << (8 - 2 * l)) & 256)];
    uint g2 = kiq3s_grid[q[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
    uint signs = sgn[l];
    for (int j = 0; j < 4; j++) {
        w[j]     = db * iq_w4(g1, j, signs);
        w[4 + j] = db * iq_w4(g2, j, signs >> 4);
    }
}

// IQ1_S, 50 bytes: four low index bytes at 2 + 4*ib and a word at 34 + 2*ib
// with three high bits per index, a 3-bit scale and the delta sign.
static inline void iq1s_dec8(device const uchar *blk, int ib, int l, thread float *w) {
    float d = iq_f16(blk);
    device const uchar *qs = blk + 2 + 4 * ib;
    uint sw = iq_ld16(blk + 34 + 2 * ib);
    float dl = d * (float)(2 * ((sw >> 12) & 7) + 1);
    float delta = (sw & 0x8000) ? -IQ1_DELTA : IQ1_DELTA;
    ulong g = kiq1s_grid[qs[l] | (((sw >> (3 * l)) & 7) << 8)];
    for (int j = 0; j < 8; j++) w[j] = dl * iq1_w(g, j, delta);
}

// IQ1_M, 56 bytes, no leading half: 32 index bytes, 16 high-bit bytes (two
// 3-bit extensions and two delta signs each), four scale words whose top
// nibbles assemble the block scale and whose low 12 bits hold two 3-bit
// scales per sub-block (one per 16 weights).
static inline void iq1m_dec8(device const uchar *blk, int ib, int l, thread float *w) {
    device const uchar *scb = blk + 48;
    uint sc0 = iq_ld16(scb), sc1 = iq_ld16(scb + 2), sc2 = iq_ld16(scb + 4), sc3 = iq_ld16(scb + 6);
    float d = iq_f16_bits((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
    device const uchar *qs = blk + 4 * ib, *qh = blk + 32 + 2 * ib;
    uint sw = iq_ld16(scb + 2 * (ib >> 1)) >> (6 * (ib & 1));
    float dl = d * (float)(2 * (l < 2 ? (sw & 7) : ((sw >> 3) & 7)) + 1);
    uint hb = qh[l >> 1];
    uint idx = qs[l] | ((hb << ((l & 1) ? 4 : 8)) & 0x700);
    float delta = (hb & ((l & 1) ? 0x80 : 0x08)) ? -IQ1_DELTA : IQ1_DELTA;
    ulong g = kiq1s_grid[idx];
    for (int j = 0; j < 8; j++) w[j] = dl * iq1_w(g, j, delta);
}

// Matvec: one 32-weight sub-block per lane iteration, the k_mv_iq4_xs shape.
#define IQ_MV(NAME, BS, DEC) \
kernel void NAME(MV_PARAMS) { \
    MV_HEAD; \
    int nb = a.n_in / 256; \
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * BS; \
    float s = 0; \
    int nq = nb * 8; \
    for (int u = tiisg; u < nq; u += 32) { \
        int b = u >> 3, ib = u & 7; \
        device const uchar *blk = rw + (ulong)b * BS; \
        device const float *xp = x + b * 256 + ib * 32; \
        float t = 0; \
        for (int l = 0; l < 4; l++) { \
            float w[8]; \
            DEC(blk, ib, l, w); \
            for (int j = 0; j < 8; j++) t += w[j] * xp[8 * l + j]; \
        } \
        s += t; \
    } \
    MV_TAIL; \
}
IQ_MV(k_mv_iq2_xxs, 66,  iq2xxs_dec8)
IQ_MV(k_mv_iq2_xs,  74,  iq2xs_dec8)
IQ_MV(k_mv_iq2_s,   82,  iq2s_dec8)
IQ_MV(k_mv_iq3_xxs, 98,  iq3xxs_dec8)
IQ_MV(k_mv_iq3_s,   110, iq3s_dec8)
IQ_MV(k_mv_iq1_s,   50,  iq1s_dec8)
IQ_MV(k_mv_iq1_m,   56,  iq1m_dec8)

// q3_K's 16 six-bit scales live packed in 12 bytes: the low 4 bits of each of
// the first 8 bytes, with the top 2 bits of each scale distributed through the
// last 4. This is the same shuffle the CPU path does with three uint32 words;
// the bytes are assembled by hand rather than cast because a 110-byte block
// leaves scales at blk+96 with no alignment guarantee.
static inline void q3k_scales(device const uchar *s12, thread char *out16) {
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint aux[4];
    for (int i = 0; i < 3; i++)
        aux[i] = (uint)s12[i * 4 + 0]        | ((uint)s12[i * 4 + 1] << 8) |
                ((uint)s12[i * 4 + 2] << 16) | ((uint)s12[i * 4 + 3] << 24);
    uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 4; b++)
            out16[i * 4 + b] = (char)((aux[i] >> (8 * b)) & 0xFF);
}

// q3_K, quarter-superblocks, same decomposition as q2_K below. Two differences
// from q2_K: the scales are the packed 6-bit set above (biased by 32, and
// signed after the bias, so a scale can be negative), and every weight carries
// a third bit in a separate 32-byte hmask that SUBTRACTS 4 when clear. That
// mask is indexed by weight position within the whole superblock, so unlike qs
// it does not advance with the half -- both halves read the same 32 bytes and
// pick different bits, one bit per (half, shift) pair.
//
// Scalar for the same measured reason as k_mv_q4_K. Do not vectorise blind.
kernel void k_mv_q3_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 110;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 110;
        device const uchar *hm  = blk;                        // hmask[32]
        device const uchar *q   = blk + 32 + (jj >> 1) * 32;  // qs[64]
        float d_all = (float)*(device const half *)(blk + 108);
        char sc[16];
        q3k_scales(blk + 96, sc);
        device const float *xp = x + b * 256 + jj * 64;
        int si = (jj >> 1) * 8 + (jj & 1) * 4;
        uchar sh0 = (uchar)((jj & 1) * 4), sh1 = (uchar)(sh0 + 2);
        uchar m0 = (uchar)(1u << ((jj >> 1) * 4 + (jj & 1) * 2));
        uchar m1 = (uchar)(m0 << 1);
        float d0 = d_all * (float)((int)sc[si + 0] - 32);
        float d1 = d_all * (float)((int)sc[si + 1] - 32);
        float d2 = d_all * (float)((int)sc[si + 2] - 32);
        float d3 = d_all * (float)((int)sc[si + 3] - 32);
        float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (int l = 0; l < 16; l++) {
            uchar qa = q[l],  qb = q[l + 16];
            uchar ha = hm[l], hb = hm[l + 16];
            t0 += (float)((int)((qa >> sh0) & 3) - ((ha & m0) ? 0 : 4)) * xp[l];
            t1 += (float)((int)((qb >> sh0) & 3) - ((hb & m0) ? 0 : 4)) * xp[l + 16];
            t2 += (float)((int)((qa >> sh1) & 3) - ((ha & m1) ? 0 : 4)) * xp[l + 32];
            t3 += (float)((int)((qb >> sh1) & 3) - ((hb & m1) ? 0 : 4)) * xp[l + 48];
        }
        s += d0 * t0 + d1 * t1 + d2 * t2 + d3 * t3;
    }
    MV_TAIL;
}

// q2_K, in quarter-superblocks for the same occupancy reason as q4_K below.
//
// The block is 84 bytes: scales[16], qs[64], then d and dmin as halves at the
// END (offsets 80 and 82), unlike q4_K/q5_K which lead with them. Its 256
// weights are two halves of 128; each half owns 32 bytes of qs and 8 scale
// bytes, and within a half the four 2-bit shifts (0,2,4,6) each cover 32
// weights as two groups of 16 (qs[0..15] and qs[16..31]). Each scale byte
// carries BOTH a scale in its low nibble and a min in its high nibble, so
// unlike q4_K there is no 6-bit unpacking helper to call.
//
// A quarter (64 weights) is therefore one half's worth of two adjacent shifts:
// h = jj>>1 picks the half, and the four scale bytes at h*8 + (jj&1)*4 are
// exactly (shift0,group0), (shift0,group1), (shift1,group0), (shift1,group1).
// The 32 qs bytes are re-read by the other quarter of the same half, which is
// the same cache-hit-for-occupancy trade k_mv_q5_K documents for its qh.
//
// Kept scalar deliberately, matching k_mv_q4_K: hand-vectorising that one was
// measured SLOWER (see its comment). Do not vectorise this without measuring.
kernel void k_mv_q2_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 84;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 84;
        device const uchar *sc = blk + (jj >> 1) * 8 + (jj & 1) * 4;
        device const uchar *q  = blk + 16 + (jj >> 1) * 32;
        float d    = (float)*(device const half *)(blk + 80);
        float dmin = (float)*(device const half *)(blk + 82);
        device const float *xp = x + b * 256 + jj * 64;
        uchar sh0 = (uchar)((jj & 1) * 4), sh1 = (uchar)(sh0 + 2);
        uchar c0 = sc[0], c1 = sc[1], c2 = sc[2], c3 = sc[3];
        float d0 = d * (c0 & 0xF), m0 = dmin * (c0 >> 4);
        float d1 = d * (c1 & 0xF), m1 = dmin * (c1 >> 4);
        float d2 = d * (c2 & 0xF), m2 = dmin * (c2 >> 4);
        float d3 = d * (c3 & 0xF), m3 = dmin * (c3 >> 4);
        float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        float sx0 = 0, sx1 = 0, sx2 = 0, sx3 = 0;
        for (int l = 0; l < 16; l++) {
            uchar qa = q[l], qb = q[l + 16];
            float x0 = xp[l], x1 = xp[l + 16], x2 = xp[l + 32], x3 = xp[l + 48];
            t0 += (float)((qa >> sh0) & 3) * x0; sx0 += x0;
            t1 += (float)((qb >> sh0) & 3) * x1; sx1 += x1;
            t2 += (float)((qa >> sh1) & 3) * x2; sx2 += x2;
            t3 += (float)((qb >> sh1) & 3) * x3; sx3 += x3;
        }
        s += d0 * t0 - m0 * sx0 + d1 * t1 - m1 * sx1
           + d2 * t2 - m2 * sx2 + d3 * t3 - m3 * sx3;
    }
    MV_TAIL;
}

// Work is handed out in QUARTER-superblocks (64 weights), not whole ones.
// Whole-superblock striding starved the simdgroup on every matvec whose input
// is n_embd: nb = n_in/256 is 10 for a 2560-wide model, so 10 lanes worked and
// 22 idled — 31% occupancy on q/k/v/o/gate/up, and only 62% on the wider
// ffn_down (40 blocks over 32 lanes). Idle lanes cost more than their
// arithmetic here: a lane that does nothing also issues no load, so the
// simdgroup had 10 memory requests in flight where it could have had 32, and
// decode never came close to memory peak. Quartering lifts those to 62% and
// 100%.
//
// A quarter is the smallest unit that does not re-read weights: 64 weights
// live in 32 bytes of q as low and high nibbles, so splitting finer would
// fetch the same bytes twice.
kernel void k_mv_q4_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 144;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 144;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const packed_uchar4 *q4 =
            (device const packed_uchar4 *)(blk + 16 + jj * 32);
        device const packed_float4 *xp4 =
            (device const packed_float4 *)(x + b * 256 + jj * 64);
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        // Hand-vectorising this loop was tried on 2026-08-07 and measured
        // SLOWER (6.58 -> 6.43 tok/s) — but that attempt moved the SUM into
        // float4 accumulators and paid for four horizontal reductions the
        // scalar form never needs. This one widens only the LOADS: the four
        // accumulators stay scalar and every `+=` keeps its original
        // expression, so the arithmetic is untouched and what disappears is 96
        // of the 128 load instructions per quarter-superblock.
        float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
        for (int k = 0; k < 8; k++) {
            uchar4 qq = q4[k];
            float4 xl = xp4[k], xh = xp4[k + 8];
            t1 += (float)(qq.x & 0xF) * xl.x;  sx1 += xl.x;
            t2 += (float)(qq.x >> 4)  * xh.x;  sx2 += xh.x;
            t1 += (float)(qq.y & 0xF) * xl.y;  sx1 += xl.y;
            t2 += (float)(qq.y >> 4)  * xh.y;  sx2 += xh.y;
            t1 += (float)(qq.z & 0xF) * xl.z;  sx1 += xl.z;
            t2 += (float)(qq.z >> 4)  * xh.z;  sx2 += xh.z;
            t1 += (float)(qq.w & 0xF) * xl.w;  sx1 += xl.w;
            t2 += (float)(qq.w >> 4)  * xh.w;  sx2 += xh.w;
        }
        s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
    }
    MV_TAIL;
}

// Quarter-superblocks per lane and a vectorised body, the same two changes
// that took q4_K and q6_K decode from behind the CPU path to ahead of it. The
// only wrinkle is qh: unlike q4_K's q pointer it does NOT advance per quarter
// — all four quarters read the same 32 bytes and pick different bit pairs out
// of them — so quartering re-reads 32 of the block's 176 bytes three extra
// times. That is a cache hit against a 22-lane occupancy win, and the
// measurement agrees.
kernel void k_mv_q5_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 176;
    float s = 0;
    int nq = nb * 4;                       // quarter-superblocks on this row
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 176;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *qh = blk + 16;
        device const uchar *q  = blk + 48 + jj * 32;
        device const float *xp = x + b * 256 + jj * 64;
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        uchar u1 = (uchar)(1u << (2 * jj)), u2 = (uchar)(2u << (2 * jj));
        // packed_uchar4 for the same reason as q6_K: 176 is a multiple of 16,
        // but q sits at blk+48+jj*32 and qh at blk+16, so only packed types
        // are safe without re-deriving alignment for every offset. The
        // activations are packed too, and for a reason that had been missed:
        // x is x_all + col * x_stride and x_stride is a model dimension, so a
        // plain float4 load off it was asserting a 16-byte alignment the
        // caller never promised.
        device const packed_uchar4 *q4  = (device const packed_uchar4 *)q;
        device const packed_uchar4 *qh4 = (device const packed_uchar4 *)qh;
        device const packed_float4 *xlo = (device const packed_float4 *)xp;
        device const packed_float4 *xhi = (device const packed_float4 *)(xp + 32);
        float4 a1v = 0, a2v = 0;
        for (int k = 0; k < 8; k++) {
            uchar4 qq = q4[k], hh = qh4[k];
            float4 hi1 = select(float4(0.0f), float4(16.0f), (hh & u1) != 0);
            float4 hi2 = select(float4(0.0f), float4(16.0f), (hh & u2) != 0);
            a1v += (d1 * (float4(qq & 0xF) + hi1) - mm1) * xlo[k];
            a2v += (d2 * (float4(qq >> 4)  + hi2) - mm2) * xhi[k];
        }
        s += a1v.x + a1v.y + a1v.z + a1v.w + a2v.x + a2v.y + a2v.z + a2v.w;
    }
    MV_TAIL;
}

// Half-superblocks per lane, for the same reason k_mv_q4_K uses quarters: at
// nb = n_in/256 = 10, whole-block striding left 22 of 32 lanes idle and the
// simdgroup with a third of the memory requests it could have had in flight.
// A half (128 weights) is the natural unit here — ql[l] and ql[l+32] both feed
// one l-iteration, and qh packs four weights' high bits into one byte, so
// splitting finer would re-read those bytes. 31% -> 62% occupancy on the
// n_embd-wide matvecs, 62% -> 100% on ffn_down.
kernel void k_mv_q6_K(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 256;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 210;
    float s = 0;
    int nh = nb * 2;                       // half-superblocks on this row
    for (int u = tiisg; u < nh; u += 32) {
        int b = u >> 1, half_i = u & 1;
        device const uchar *blk = rw + (ulong)b * 210;
        device const uchar *ql = blk + half_i * 64;
        device const uchar *qh = blk + 128 + half_i * 32;
        device const char  *sc = (device const char *)(blk + 192) + half_i * 8;
        float d = (float)*(device const half *)(blk + 208);
        device const float *xp = x + b * 256 + half_i * 128;
        float t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int is = 0; is < 2; is++) {
            // packed_uchar4, NOT uchar4: a q6_K superblock is 210 bytes, which
            // is not a multiple of 4, so ql/qh sit at odd alignments for odd
            // blocks and a uchar4 load there is undefined. It happened to
            // return correct results on an M1 and passed the tolerance gate —
            // which is exactly why it needed catching by reading the block
            // size rather than by trusting the test. packed_uchar4 has
            // 1-byte alignment and compiles to the same loads where the
            // address does happen to be aligned.
            device const packed_uchar4 *l4  = (device const packed_uchar4 *)(ql + is * 16);
            device const packed_uchar4 *l4b = (device const packed_uchar4 *)(ql + 32 + is * 16);
            device const packed_uchar4 *h4  = (device const packed_uchar4 *)(qh + is * 16);
            device const packed_float4 *x0 = (device const packed_float4 *)(xp + is * 16);
            device const packed_float4 *x1 = (device const packed_float4 *)(xp + 32 + is * 16);
            device const packed_float4 *x2 = (device const packed_float4 *)(xp + 64 + is * 16);
            device const packed_float4 *x3 = (device const packed_float4 *)(xp + 96 + is * 16);
            float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int k = 0; k < 4; k++) {
                uchar4 lo = l4[k], hi = l4b[k], h = h4[k];   // widen on load
                a0 += (float4(lo & 0xF) + float4((h >> 0) & 3) * 16.0f - 32.0f) * x0[k];
                a1 += (float4(hi & 0xF) + float4((h >> 2) & 3) * 16.0f - 32.0f) * x1[k];
                a2 += (float4(lo >> 4)  + float4((h >> 4) & 3) * 16.0f - 32.0f) * x2[k];
                a3 += (float4(hi >> 4)  + float4((h >> 6) & 3) * 16.0f - 32.0f) * x3[k];
            }
            t[is * 4 + 0] = a0.x + a0.y + a0.z + a0.w;
            t[is * 4 + 1] = a1.x + a1.y + a1.z + a1.w;
            t[is * 4 + 2] = a2.x + a2.y + a2.z + a2.w;
            t[is * 4 + 3] = a3.x + a3.y + a3.z + a3.w;
        }
        s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                  sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
    }
    MV_TAIL;
}

constant float kv_mxfp4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
     0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

kernel void k_mv_mxfp4(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 17;
    float s = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        device const float *xp = x + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    MV_TAIL;
}


// ------------------------------------------- fast (reassociating) decode matvec
// The k_mv_* family above IS the CPU<->GPU byte-identity contract, and that
// contract is what caps it. Each output row accumulates over
// `for (i = tiisg; i < n; i += 32)` into one scalar and finishes with one
// simd_sum, so which lane owns which block decides which partial sums enter
// the reduction tree. The 2026-08-12 sweep
// (docs/negative-result-metal-multirow-matvec.md) measured the two levers that
// survive that pin — wider loads (neutral) and rows-per-simdgroup (zero-sum,
// because simdgroups x rows is fixed at n_out) — and concluded that the
// roofline gap needs the transformations identity forbids, on a SECOND kernel
// promoted by teacher-forced tolerance the way the tiled GEMM below already is.
//
// This is that kernel family. It answers to tests/test_mv_tol.c through
// gpu_mv_force(); the identity family stays reachable and byte-for-byte
// unchanged behind RUNNER_METAL_MV=0.
//
// What it does that identity forbids:
//
//   * float4 accumulation. The identity form must spell each product as its
//     own scalar `+=` in the original expression order; here four lanes of a
//     float4 accumulate independently and are summed horizontally at the end.
//     That is a different reduction tree, and it is the point: it turns the
//     per-element scalar FMA chain into one vector FMA per four elements.
//   * the q4_0 zero-point factored out of the inner loop. sum (q-8)*y*d
//     becomes d*(sum q*y) - 8d*(sum y), so the per-element integer subtract
//     disappears and the nibble goes straight to float.
//
// The hypothesis being tested, stated so a later reader can check it rather
// than re-derive it: at 36% of roofline this route is ISSUE-bound, not
// traffic-bound — weight bytes are irreducible (every byte is read exactly
// once either way), so the only thing left to cut is issued instructions per
// byte consumed. If that hypothesis is wrong the gate still passes and the
// bench simply shows nothing, which is a result worth recording either way.
//
// Decode only: enc_mv_n selects this at n_col == 1. Prefill already has the
// tiled GEMM, which is a strictly better answer for a real batch.

kernel void k_mvf_q4_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 18;
    float4 acc = 0;      // sum over blocks of d * (q . y)
    float  corr = 0;     // sum over blocks of d * (sum y), the -8 zero-point
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const packed_uchar4 *q = (device const packed_uchar4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float4 t = 0, sy = 0;
        for (int k = 0; k < 4; k++) {
            uchar4 qq = q[k];
            float4 xl = xp[k], xh = xp[k + 4];
            t  += float4(qq & 0xF) * xl;
            t  += float4(qq >> 4)  * xh;
            sy += xl + xh;
        }
        acc  += d * t;
        corr += d * (sy.x + sy.y + sy.z + sy.w);
    }
    float s = (acc.x + acc.y + acc.z + acc.w) - 8.0f * corr;
    MV_TAIL;
}

// q8_0 has no zero-point to factor, so this is the float4-accumulator half of
// the change alone: 8 vector FMAs per block against 32 scalar ones.
kernel void k_mvf_q8_0(MV_PARAMS) {
    MV_HEAD;
    int nb = a.n_in / 32;
    device const uchar *rw = wb + a.w_off + (ulong)row * nb * 34;
    float4 acc = 0;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        device const packed_char4 *q = (device const packed_char4 *)(blk + 2);
        device const packed_float4 *xp = (device const packed_float4 *)(x + b * 32);
        float4 t = 0;
        for (int k = 0; k < 8; k++) t += float4(q[k]) * xp[k];
        acc += d * t;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    MV_TAIL;
}

// ------------------------------------------------------- tiled prefill GEMM
// The matvec kernels above give one output element per simdgroup: 32 lanes
// each do one FMA and then a log-depth reduction, so almost none of the work
// is reuse. This computes an output TILE per threadgroup with Apple's
// simdgroup matrix units instead, which is where prompt processing has its
// real headroom.
//
// It is deliberately NOT bit-identical to the matvec path and cannot be: the
// weight is dequantized before the multiply (d*q)*x rather than d*(q*x), and
// the sum is reassociated into 8-element matrix steps. That is the same trade
// the CUDA tensor-core prefill makes, so it answers to the same kind of gate —
// tests/test_tc_tol.c, driven through gpu_tc_force() — rather than to an
// identity claim it cannot honour.
//
// Orientation: out[c][r] = sum_k W[r][k] * X[c][k], i.e. out = X * W^T. That
// makes the result tile row-major in the COLUMN index, which is exactly how y
// is laid out (column c at offset c*y_stride), so the store is contiguous.
// W is loaded transposed straight out of threadgroup memory.

// MM_TM/MM_TN must stay in step with the dispatch grid in metal.m
// (enc_mv_n's dispatchThreadgroups) and with tests/test_metal_kquants.m's
// `submit()` helper -- none of the three can see the other two, so a change
// here that is not mirrored there is a silent correctness bug, not a build
// error (the grid would just cover the wrong number of output tiles).
#define MM_TM 64    // output rows per threadgroup: multiple of 32 (4 simdgroups x 8)
#define MM_TN 32    // output columns (prompt tokens) per threadgroup: multiple of 8
#define MM_TK 32    // k-step: one q8_0/q4_0/mxfp4 block, and a q4_K sub-block
// Row-groups per simdgroup and column-groups per tile; every simdgroup owns a
// distinct MM_RPS-tall slice of the weight tile and all four cover the same
// MM_CG activation column-groups, for MM_RPS*MM_CG accumulators per thread.
#define MM_RPS (MM_TM / 32)
#define MM_CG  (MM_TN / 8)

struct mm_args {
    int   n_in, n_out, n_col;
    ulong w_off;
    int   has_bias, x_stride, y_stride;
};

#define MM_PARAMS \
    device const uchar *wb   [[buffer(0)]], \
    device const float *x    [[buffer(1)]], \
    device float       *y    [[buffer(2)]], \
    constant mm_args   &a    [[buffer(3)]], \
    device const float *bias [[buffer(4)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint3 tpitg [[thread_position_in_threadgroup]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]]

// Shared prologue/epilogue; DEQ_CHUNK fills tg_w[r][0..MM_TK) for this
// thread's row/sub-range from the type's own block layout.
//
// tg_w/tg_x stage in HALF, not float: the simdgroup A/B operands are loaded
// as simdgroup_half8x8 and multiply-accumulated into a simdgroup_float8x8
// accumulator (MSL's mixed-precision simdgroup_multiply_accumulate, the same
// pattern llama.cpp's Metal mul_mm uses). That halves the threadgroup-memory
// traffic per tile and doubles the matrix-unit throughput on hardware that
// runs half-operand GEMM faster than float-operand GEMM, at the cost of a
// half-precision rounding step on every staged weight/activation value. That
// is exactly the trade this kernel family already answers to
// tests/test_tc_tol.c for (see the file comment above): not bit-identical to
// the scalar path, gated on teacher-forced logit deviation instead. The
// output accumulator (tg_c) and the final store stay float32 — only the
// per-tile operands round to half, not the reduction.
#define MM_BODY(...) \
    threadgroup half  tg_w[MM_TM * MM_TK]; \
    threadgroup half  tg_x[MM_TN * MM_TK]; \
    threadgroup float tg_c[MM_TN * MM_TM]; \
    const int row0 = (int)tgpig.x * MM_TM; \
    const int col0 = (int)tgpig.y * MM_TN; \
    const int tid  = (int)tpitg.x; \
    simdgroup_float8x8 acc[MM_RPS][MM_CG]; \
    for (int rg = 0; rg < MM_RPS; rg++) \
        for (int cg = 0; cg < MM_CG; cg++) acc[rg][cg] = simdgroup_float8x8(0.0f); \
    for (int k0 = 0; k0 < a.n_in; k0 += MM_TK) { \
        /* 128 threads x 8 values/pass = one 32-row x MM_TK weight slab;
           MM_RPS passes cover the full MM_TM-row tile */ \
        for (int p = 0; p < MM_RPS; p++) { \
            int r = p * 32 + (tid >> 2), sub = (tid & 3) * 8; \
            int row = row0 + r; \
            threadgroup half *dst = tg_w + r * MM_TK + sub; \
            if (row < a.n_out) { __VA_ARGS__; } \
            else { for (int j = 0; j < 8; j++) dst[j] = 0.0h; } \
        } \
        /* 128 threads x 4 values/pass = the MM_TN x MM_TK activation tile */ \
        for (int i = tid * 4; i < MM_TN * MM_TK; i += 128 * 4) { \
            for (int j = 0; j < 4; j++) { \
                int idx = i + j, cc = idx / MM_TK, kk = idx % MM_TK; \
                tg_x[idx] = (half)((col0 + cc < a.n_col) \
                    ? x[(ulong)(col0 + cc) * a.x_stride + k0 + kk] : 0.0f); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int kk = 0; kk < MM_TK / 8; kk++) { \
            simdgroup_half8x8 acols[MM_CG]; \
            for (int cg = 0; cg < MM_CG; cg++) \
                simdgroup_load(acols[cg], tg_x + cg * 8 * MM_TK + kk * 8, MM_TK); \
            for (int rg = 0; rg < MM_RPS; rg++) { \
                simdgroup_half8x8 B; \
                int wrg = (int)sgitg * MM_RPS + rg; \
                simdgroup_load(B, tg_w + wrg * 8 * MM_TK + kk * 8, MM_TK, \
                               ulong2(0, 0), true); \
                for (int cg = 0; cg < MM_CG; cg++) \
                    simdgroup_multiply_accumulate(acc[rg][cg], acols[cg], B, \
                                                  acc[rg][cg]); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (int rg = 0; rg < MM_RPS; rg++) { \
        int wrg = (int)sgitg * MM_RPS + rg; \
        for (int cg = 0; cg < MM_CG; cg++) \
            simdgroup_store(acc[rg][cg], tg_c + cg * 8 * MM_TM + wrg * 8, MM_TM); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int idx = tid; idx < MM_TN * MM_TM; idx += 128) { \
        int cc = idx / MM_TM, rr = idx % MM_TM; \
        if (col0 + cc < a.n_col && row0 + rr < a.n_out) \
            y[(ulong)(col0 + cc) * a.y_stride + row0 + rr] = \
                a.has_bias ? tg_c[idx] + bias[row0 + rr] : tg_c[idx]; \
    }

kernel void k_mm_f32(MM_PARAMS) {
    MM_BODY({
        device const float *rw = (device const float *)(wb + a.w_off)
                               + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = rw[j];
    })
}

kernel void k_mm_f16(MM_PARAMS) {
    MM_BODY({
        device const half *rw = (device const half *)(wb + a.w_off)
                              + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = (float)rw[j];
    })
}

kernel void k_mm_q8_0(MM_PARAMS) {
    MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nb + k0 / 32) * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2) + sub;
        for (int j = 0; j < 8; j++) dst[j] = d * (float)q[j];
    })
}

kernel void k_mm_q4_0(MM_PARAMS) {
    MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nb + k0 / 32) * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        /* low nibbles hold lanes 0-15, high nibbles lanes 16-31 */
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            int v = lane < 16 ? (int)(q[lane] & 0xF) : (int)(q[lane - 16] >> 4);
            dst[j] = d * (float)(v - 8);
        }
    })
}

// Tiled-GEMM variants for the formats that only had a matvec. Without these,
// prefill fell back to the matvec path and ran SLOWER on the GPU than on the
// CPU (measured: tinyllama q2_K 36.9 -> 28.2 tok/s prompt, iq4_xs 101.5 ->
// 87.5) because the matvec re-reads every weight once per token column, while
// this reads each weight tile once for sixteen. Types that already had a
// k_mm gained 3.7-5.3x on the same measurement, which is the prize here.
//
// MM_TK is 32, and `sub` is a multiple of 8, so each chunk of 8 stays inside
// one block AND inside one nibble half -- which is what makes these as simple
// as picking the right base pointer and shift.

kernel void k_mm_bf16(MM_PARAMS) {
    MM_BODY({
        device const ushort *rw = (device const ushort *)(wb + a.w_off)
                                + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = bf16_to_f32_m(rw[j]);
    })
}

kernel void k_mm_iq4_nl(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, b = k >> 5, t0 = k & 31;
        device const uchar *blk =
            wb + a.w_off + ((ulong)row * (a.n_in >> 5) + b) * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        bool hi = t0 >= 16;
        int base = hi ? t0 - 16 : t0;
        for (int j = 0; j < 8; j++) {
            uchar by = q[base + j];
            dst[j] = d * (float)kvalues_iq4nl[hi ? (by >> 4) : (by & 0xF)];
        }
    })
}

kernel void k_mm_iq4_xs(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255, ib = sbk >> 5, t0 = sbk & 31;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 136;
        float d = (float)*(device const half *)blk;
        uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
        int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) |
                 (((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        device const uchar *q = blk + 8 + ib * 16;
        bool hi = t0 >= 16;
        int base = hi ? t0 - 16 : t0;
        for (int j = 0; j < 8; j++) {
            uchar by = q[base + j];
            dst[j] = dl * (float)kvalues_iq4nl[hi ? (by >> 4) : (by & 0xF)];
        }
    })
}

// Codebook i-quant tiled GEMMs: a chunk of 8 at k = k0 + sub is one group of
// one sub-block (MM_TK is 32, sub a multiple of 8), decoded by the same
// helpers the matvecs use.
#define IQ_MM(NAME, BS, DEC) \
kernel void NAME(MM_PARAMS) { \
    MM_BODY({ \
        int k = k0 + sub, nsb = a.n_in / 256; \
        int sb = k >> 8, sbk = k & 255, ib = sbk >> 5, l = (sbk & 31) >> 3; \
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * BS; \
        float w[8]; \
        DEC(blk, ib, l, w); \
        for (int j = 0; j < 8; j++) dst[j] = w[j]; \
    }) \
}
IQ_MM(k_mm_iq2_xxs, 66,  iq2xxs_dec8)
IQ_MM(k_mm_iq2_xs,  74,  iq2xs_dec8)
IQ_MM(k_mm_iq2_s,   82,  iq2s_dec8)
IQ_MM(k_mm_iq3_xxs, 98,  iq3xxs_dec8)
IQ_MM(k_mm_iq3_s,   110, iq3s_dec8)
IQ_MM(k_mm_iq1_s,   50,  iq1s_dec8)
IQ_MM(k_mm_iq1_m,   56,  iq1m_dec8)

kernel void k_mm_q2_K(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255;
        int h = sbk >> 7, rem = sbk & 127, j32 = rem >> 5, rem2 = rem & 31;
        int g = rem2 >> 4, l0 = rem2 & 15;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 84;
        float d    = (float)*(device const half *)(blk + 80);
        float dmin = (float)*(device const half *)(blk + 82);
        uchar scb  = blk[h * 8 + j32 * 2 + g];
        float dl = d * (scb & 0xF), ml = dmin * (scb >> 4);
        device const uchar *q = blk + 16 + h * 32 + g * 16;
        uchar shift = (uchar)(2 * j32);
        for (int j = 0; j < 8; j++)
            dst[j] = dl * (float)((q[l0 + j] >> shift) & 3) - ml;
    })
}

kernel void k_mm_q3_K(MM_PARAMS) {
    MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255;
        int h = sbk >> 7, rem = sbk & 127, j32 = rem >> 5, rem2 = rem & 31;
        int g = rem2 >> 4, l0 = rem2 & 15;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 110;
        float d_all = (float)*(device const half *)(blk + 108);
        char sc[16];
        q3k_scales(blk + 96, sc);
        float dl = d_all * (float)((int)sc[h * 8 + j32 * 2 + g] - 32);
        device const uchar *hm = blk + g * 16;          /* not offset by half */
        device const uchar *q  = blk + 32 + h * 32 + g * 16;
        uchar shift = (uchar)(2 * j32);
        uchar mbit  = (uchar)(1u << (h * 4 + j32));
        for (int j = 0; j < 8; j++)
            dst[j] = dl * (float)((int)((q[l0 + j] >> shift) & 3)
                                  - ((hm[l0 + j] & mbit) ? 0 : 4));
    })
}

kernel void k_mm_q4_K(MM_PARAMS) {
    MM_BODY({
        int nsb = a.n_in / 256;
        int sb  = k0 / 256;              /* superblock */
        int j32 = (k0 % 256) / 32;       /* which 32-lane sub-block */
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 144;
        float dall = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        uchar s1, m1;
        get_scale_min_k4(j32, sc, &s1, &m1);
        float d = dall * s1, mm = dmin * m1;
        device const uchar *q = blk + 16 + (j32 / 2) * 32;
        bool hi = (j32 & 1) != 0;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            int v = hi ? (int)(q[lane] >> 4) : (int)(q[lane] & 0xF);
            dst[j] = d * (float)v - mm;
        }
    })
}

kernel void k_mm_q6_K(MM_PARAMS) {
    MM_BODY({
        int nsb = a.n_in / 256;
        int sb  = k0 / 256;
        int off = k0 % 256;              /* 0,32,...,224 */
        device const uchar *blk = wb + a.w_off + ((ulong)row * nsb + sb) * 210;
        float dall = (float)*(device const half *)(blk + 208);
        device const char *scs = (device const char *)(blk + 192);
        /* q6_K packs 128 lanes per half: ql[0..63] low nibbles + qh 2 bits */
        int half_i = off / 128, within = off % 128;
        device const uchar *ql = blk + half_i * 64;
        device const uchar *qh = blk + 128 + half_i * 32;
        device const char  *sc = scs + half_i * 8;
        for (int j = 0; j < 8; j++) {
            int lane = within + sub + j;      /* 0..127 within this half */
            int l = lane % 32, grp = lane / 32;
            /* groups 0,1 take low nibbles and 2,3 high; the byte index cycles
               l, l+32, l, l+32 — matching the reference's four q1..q4 lanes */
            int qb = (int)ql[l + (grp & 1) * 32];
            int qlv = (grp >= 2) ? (qb >> 4) : (qb & 0xF);
            int qhv = ((int)(qh[l] >> (grp * 2)) & 3) << 4;
            int is = (l / 16) & 1;
            dst[j] = dall * (float)sc[grp * 2 + is] * (float)((qlv | qhv) - 32);
        }
    })
}

kernel void k_mm_mxfp4(MM_PARAMS) {
    MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + a.w_off + ((ulong)row * nb + k0 / 32) * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            uchar nib = lane < 16 ? (q[lane] & 0xF) : (q[lane - 16] >> 4);
            dst[j] = d * kv_mxfp4[nib];
        }
    })
}

// ---------------------------------------------------------------- rope

// pos is the FIRST column's position; column c rotates at pos + c, so a whole
// prompt batch ropes in one dispatch (grid.z selects the column).
struct rope_args {
    int   head_dim, n_heads, half_dim, pos, neox;
    float mscale;
    int   stride;     // elements between consecutive columns
};

// The rotation itself, factored so every kernel that ropes — k_rope, the
// fused k_rope_store, the attention-front megakernel — computes the SAME
// IEEE sequence. "Same expression, same contraction" is not a fact across
// kernels: under fast math cos/sin inline as polynomial chains whose fma
// contraction the compiler chooses per call site, and on M1 it chose
// differently for k_rope and k_rope_store (M5 compiled them alike), which
// broke the fusion byte-identity contract. precise::cos/sin compile to one
// defined routine instead of a per-kernel polynomial, and the pragma pins
// the rotation to two multiplies and a sub/add. Rope is a vanishing
// fraction of a decode step, so precise trig costs nothing measurable.
static inline float2 rope_rot(float x0, float x1, float ang, float mscale) {
#pragma clang fp contract(off)
    float c = precise::cos(ang) * mscale, s = precise::sin(ang) * mscale;
    return float2(x0 * c - x1 * s, x0 * s + x1 * c);
}

kernel void k_rope(device float       *v_all [[buffer(0)]],
                   device const float *fr    [[buffer(1)]],
                   constant rope_args &a     [[buffer(2)]],
                   uint3 gid [[thread_position_in_grid]]) {
    int j = gid.x, h = gid.y, col = gid.z;
    if (j >= a.half_dim || h >= a.n_heads) return;
    float ang = (a.pos + col) * fr[j];
    device float *v = v_all + (ulong)col * a.stride;
    device float *p = v + h * a.head_dim;
    int i0 = a.neox ? j : 2 * j;
    int i1 = a.neox ? j + a.half_dim : i0 + 1;
    float2 y = rope_rot(p[i0], p[i1], ang, a.mscale);
    p[i0] = y.x;
    p[i1] = y.y;
}

// ---------------------------------------------------------------- kv store
// The KV cache is either fp16 (2 bytes/value) or q8_0 (32 values per 34-byte
// block: one fp16 scale + 32 int8 quants). Offsets are byte offsets so the CPU
// and Metal paths share the same cache layout.

static inline ulong kv_row_bytes(int kv_dim, int q8) {
    return q8 ? (ulong)(kv_dim / 32) * 34 : (ulong)kv_dim * 2;
}

static inline ulong kv_head_off(int kvh, int hd, int q8) {
    return q8 ? (ulong)(kvh * hd / 32) * 34 : (ulong)(kvh * hd) * 2;
}

static inline void kv_store_row(device uchar *cache, device const float *src,
                                int q8, uint i) {
    if (q8) {
        device uchar *blk = cache + (ulong)i * 34;
        device half *dptr = (device half *)blk;
        device char *q = (device char *)(blk + 2);
        device const float *x = src + i * 32;
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = max(amax, fabs(x[j]));
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0.0f;
        *dptr = (half)d;
        for (int j = 0; j < 32; j++) q[j] = (char)round(x[j] * id);
    } else {
        ((device half *)cache)[i] = (half)src[i];
    }
}

static inline float kv_dot(device const uchar *row, device const float *qh,
                           int hd, int q8) {
    float s = 0;
    if (q8) {
        for (int b = 0; b < hd / 32; b++) {
            device const uchar *blk = row + (ulong)b * 34;
            float d = (float)*(device const half *)blk;
            device const char *q = (device const char *)(blk + 2);
            device const float *xp = qh + b * 32;
            float t = 0;
            for (int j = 0; j < 32; j += 2)
                t += xp[j] * (float)q[j] + xp[j + 1] * (float)q[j + 1];
            s += d * t;
        }
    } else {
        device const half *k = (device const half *)row;
        for (int i = 0; i < hd; i++) s += qh[i] * (float)k[i];
    }
    return s;
}

static inline float2 kv_pair(device const uchar *row, int i2, int q8) {
    if (q8) {
        device const uchar *blk = row + (ulong)(i2 / 16) * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2);
        int j = (2 * i2) & 31;
        return float2(d * (float)q[j], d * (float)q[j + 1]);
    }
    device const half *h = (device const half *)row + 2 * i2;
    return float2((float)h[0], (float)h[1]);
}

// off is the FIRST column's byte offset; column c lands row_b bytes further on,
// so a prompt batch stores every token's K/V in one dispatch (grid.y = column).
// off is the LAYER's byte offset; the kernel places column c at position
// pos + c, modulo kv_rows when a ring is active (0 = flat absolute rows).
struct store_args { int kv_dim, q8, stride, pos, kv_rows; ulong off, row_b; };

kernel void k_store_kv(device const float *k_all [[buffer(0)]],
                       device const float *v_all [[buffer(1)]],
                       device uchar       *kc [[buffer(2)]],
                       device uchar       *vc [[buffer(3)]],
                       constant store_args &a [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int n = a.q8 ? a.kv_dim / 32 : a.kv_dim;
    uint i = gid.x, col = gid.y;
    if ((int)i < n) {
        int p = a.pos + (int)col;
        if (a.kv_rows > 0) p %= a.kv_rows;
        ulong off = a.off + (ulong)p * a.row_b;
        kv_store_row(kc + off, k_all + (ulong)col * a.stride, a.q8, i);
        kv_store_row(vc + off, v_all + (ulong)col * a.stride, a.q8, i);
    }
}

// ------------------------------------------------- decode fusion (n == 1)
//
// The 120B decode budget (docs/metal-decode-dispatch-budget-2026-09-01.md)
// measured 686 dispatches per token with a 3.4-4.8 us serialized chain cost
// each. These kernels fold the per-layer chain: every fused form reproduces
// the unfused kernels' arithmetic element for element — same expressions,
// same order, same conversions — so the contract is BYTE IDENTITY against
// the unfused path, gated, not argued. Justified by budget line items, not
// by what any other engine does (each kernel's saving is printed in the
// host comment beside its dispatch).

// rope(q) + rope(k) + f16-store(k,v): three dispatches -> one. Thread roles
// by flat index: q rope pairs, then k rope pairs (which also store the two
// elements they produced), then unroped k tail elements (head_dim past the
// roped span), then v pairs (straight f16 conversion). q8 caches keep the
// unfused path: a rope PAIR spans block boundaries, and the 32-wide block
// quant needs the whole roped row first.
struct rope_store_args {
    int   n_head, n_kv, head_dim, half_dim, pos, neox;
    float mscale;
    int   q_off_e, k_off_e, v_off_e;   // element offsets into the stage bufs
    ulong kc_off, vc_off;              // byte offsets of this row in the cache
};

kernel void k_rope_store(device float        *q    [[buffer(0)]],
                         device float        *kt   [[buffer(1)]],
                         device const float  *vt   [[buffer(2)]],
                         device uchar        *kc   [[buffer(3)]],
                         device uchar        *vc   [[buffer(4)]],
                         device const float  *fr   [[buffer(5)]],
                         constant rope_store_args &a [[buffer(6)]],
                         uint t [[thread_position_in_grid]]) {
    int hd = a.head_dim, hf = a.half_dim;
    int rd = 2 * hf;                         // roped dims per head
    int q_pairs = a.n_head * hf;
    int k_pairs = a.n_kv * hf;
    int k_tail  = a.n_kv * (hd - rd);
    int kv_dim  = a.n_kv * hd;
    device half *kh = (device half *)(kc + a.kc_off);
    device half *vh = (device half *)(vc + a.vc_off);
    int i = (int)t;
    if (i < q_pairs) {
        int h = i / hf, j = i % hf;
        float ang = a.pos * fr[j];
        device float *p = q + a.q_off_e + h * hd;
        int i0 = a.neox ? j : 2 * j;
        int i1 = a.neox ? j + hf : i0 + 1;
        float2 y = rope_rot(p[i0], p[i1], ang, a.mscale);
        p[i0] = y.x;
        p[i1] = y.y;
        return;
    }
    i -= q_pairs;
    if (i < k_pairs) {
        int h = i / hf, j = i % hf;
        float ang = a.pos * fr[j];
        device float *p = kt + a.k_off_e + h * hd;
        int i0 = a.neox ? j : 2 * j;
        int i1 = a.neox ? j + hf : i0 + 1;
        float2 y = rope_rot(p[i0], p[i1], ang, a.mscale);
        p[i0] = y.x;
        p[i1] = y.y;
        kh[h * hd + i0] = (half)y.x;
        kh[h * hd + i1] = (half)y.y;
        return;
    }
    i -= k_pairs;
    if (i < k_tail) {
        int per = hd - rd;
        int h = i / per, j = rd + i % per;
        kh[h * hd + j] = (half)kt[a.k_off_e + h * hd + j];
        return;
    }
    i -= k_tail;
    if (2 * i < kv_dim) {
        vh[2 * i]     = (half)vt[a.v_off_e + 2 * i];
        if (2 * i + 1 < kv_dim)
            vh[2 * i + 1] = (half)vt[a.v_off_e + 2 * i + 1];
    }
}

// The attention-front megakernel: residual-stream norm + Q/K/V matvecs +
// (optional qwen-style qk-norm) + rope + f16 KV store, ONE dispatch. This is
// the safe form of the persistent layer walk on Metal: a literal whole-layer
// kernel needs cross-threadgroup synchronization Metal does not guarantee
// (no forward-progress contract; a deadlock is a GPU watchdog event), so the
// walk is cut exactly where the data dependencies allow — whole heads per
// threadgroup, every dependency inside one barrier scope. Byte identity with
// the unfused chain is the gate: the norm replays k_rmsnorm's 256-thread
// tree, the dots replay the k_mv_* packed bodies through AFX (x served from
// threadgroup memory — same values, same order, no device re-reads), the
// qk-norm replays k_qknorm at its dispatched 64-thread shape, rope and store
// replay k_rope_store. n_embd caps at 7936 so the staged row and the
// reduction tree fit the 32 KB threadgroup budget; wider models keep the
// split path. A "wide" form that skipped staging and re-derived each normed
// element inline (x[i]*r*nw[i]) was built and measured 2026-09-01: the
// shader compiler contracts the inline renorm into the dot's fma chains, so
// it is NOT byte-identical on real data (the 70B caught it; fixtures did
// not — their small exact values round the same fused or not), and the 70B
// is bandwidth-bound anyway (no speed to win). Removed, negative recorded
// in docs/metal-decode-dispatch-budget-2026-09-01.md.
struct attn_front_args {
    int   n_embd, n_head, n_kv, head_dim, half_dim, pos, neox;
    float mscale, eps;
    ulong wq_off, wk_off, wv_off;
    int   has_bq, has_bk, has_bv;
    ulong kc_off, vc_off;
    int   has_qn, has_kn;
};

// Dot bodies: character-for-character the k_mv_* arithmetic with the
// activation load abstracted as AFX(i). Every accumulation keeps its
// original expression and order — that is the whole byte-identity argument.
#define AF_DOT_Q8_0(SV) { \
    int nb = NE / 32; \
    device const uchar *rw = wb + w_off + (ulong)row * nb * 34; \
    float s = 0; \
    for (int b = tiisg; b < nb; b += 32) { \
        device const uchar *blk = rw + (ulong)b * 34; \
        float d = (float)*(device const half *)blk; \
        device const packed_char4 *qq4 = (device const packed_char4 *)(blk + 2); \
        float t = 0; \
        for (int kk = 0; kk < 8; kk++) { \
            char4 qq = qq4[kk]; \
            t += (float)qq.x * AFX(b * 32 + 4 * kk + 0); \
            t += (float)qq.y * AFX(b * 32 + 4 * kk + 1); \
            t += (float)qq.z * AFX(b * 32 + 4 * kk + 2); \
            t += (float)qq.w * AFX(b * 32 + 4 * kk + 3); \
        } \
        s += d * t; \
    } \
    SV = s; }

#define AF_DOT_Q4_0(SV) { \
    int nb = NE / 32; \
    device const uchar *rw = wb + w_off + (ulong)row * nb * 18; \
    float s = 0; \
    for (int b = tiisg; b < nb; b += 32) { \
        device const uchar *blk = rw + (ulong)b * 18; \
        float d = (float)*(device const half *)blk; \
        device const packed_uchar4 *q4 = (device const packed_uchar4 *)(blk + 2); \
        float t = 0; \
        for (int kk = 0; kk < 4; kk++) { \
            uchar4 qq = q4[kk]; \
            int lo = b * 32 + 4 * kk, hi = lo + 16; \
            float4 xl = float4(AFX(lo + 0), AFX(lo + 1), AFX(lo + 2), AFX(lo + 3)); \
            float4 xh = float4(AFX(hi + 0), AFX(hi + 1), AFX(hi + 2), AFX(hi + 3)); \
            t += ((int)(qq.x & 0xF) - 8) * xl.x + ((int)(qq.x >> 4) - 8) * xh.x; \
            t += ((int)(qq.y & 0xF) - 8) * xl.y + ((int)(qq.y >> 4) - 8) * xh.y; \
            t += ((int)(qq.z & 0xF) - 8) * xl.z + ((int)(qq.z >> 4) - 8) * xh.z; \
            t += ((int)(qq.w & 0xF) - 8) * xl.w + ((int)(qq.w >> 4) - 8) * xh.w; \
        } \
        s += d * t; \
    } \
    SV = s; }

#define AF_DOT_Q4_K(SV) { \
    int nb = NE / 256; \
    device const uchar *rw = wb + w_off + (ulong)row * nb * 144; \
    float s = 0; \
    int nq = nb * 4; \
    for (int uq = tiisg; uq < nq; uq += 32) { \
        int b = uq >> 2, jj = uq & 3; \
        device const uchar *blk = rw + (ulong)b * 144; \
        float d    = (float)*(device const half *)blk; \
        float dmin = (float)*(device const half *)(blk + 2); \
        device const uchar *sc = blk + 4; \
        device const packed_uchar4 *q4 = \
            (device const packed_uchar4 *)(blk + 16 + jj * 32); \
        int xo = b * 256 + jj * 64; \
        uchar s1, m1, s2, m2; \
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1); \
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2); \
        float d1 = d * s1, mm1 = dmin * m1; \
        float d2 = d * s2, mm2 = dmin * m2; \
        float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0; \
        for (int kk = 0; kk < 8; kk++) { \
            uchar4 qq = q4[kk]; \
            float xlv, xhv; \
            xlv = AFX(xo + 4 * kk + 0); xhv = AFX(xo + 32 + 4 * kk + 0); \
            t1 += (float)(qq.x & 0xF) * xlv;  sx1 += xlv; \
            t2 += (float)(qq.x >> 4)  * xhv;  sx2 += xhv; \
            xlv = AFX(xo + 4 * kk + 1); xhv = AFX(xo + 32 + 4 * kk + 1); \
            t1 += (float)(qq.y & 0xF) * xlv;  sx1 += xlv; \
            t2 += (float)(qq.y >> 4)  * xhv;  sx2 += xhv; \
            xlv = AFX(xo + 4 * kk + 2); xhv = AFX(xo + 32 + 4 * kk + 2); \
            t1 += (float)(qq.z & 0xF) * xlv;  sx1 += xlv; \
            t2 += (float)(qq.z >> 4)  * xhv;  sx2 += xhv; \
            xlv = AFX(xo + 4 * kk + 3); xhv = AFX(xo + 32 + 4 * kk + 3); \
            t1 += (float)(qq.w & 0xF) * xlv;  sx1 += xlv; \
            t2 += (float)(qq.w >> 4)  * xhv;  sx2 += xhv; \
        } \
        s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2; \
    } \
    SV = s; }

#define AF_DOT_Q6_K(SV) { \
    int nb = NE / 256; \
    device const uchar *rw = wb + w_off + (ulong)row * nb * 210; \
    float s = 0; \
    int nh = nb * 2; \
    for (int uh = tiisg; uh < nh; uh += 32) { \
        int b = uh >> 1, half_i = uh & 1; \
        device const uchar *blk = rw + (ulong)b * 210; \
        device const uchar *ql = blk + half_i * 64; \
        device const uchar *qh = blk + 128 + half_i * 32; \
        device const char  *sc = (device const char *)(blk + 192) + half_i * 8; \
        float d = (float)*(device const half *)(blk + 208); \
        int xo = b * 256 + half_i * 128; \
        float t[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
        for (int is = 0; is < 2; is++) { \
            device const packed_uchar4 *l4  = (device const packed_uchar4 *)(ql + is * 16); \
            device const packed_uchar4 *l4b = (device const packed_uchar4 *)(ql + 32 + is * 16); \
            device const packed_uchar4 *h4  = (device const packed_uchar4 *)(qh + is * 16); \
            float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0; \
            for (int kk = 0; kk < 4; kk++) { \
                uchar4 lo = l4[kk], hi = l4b[kk], h = h4[kk]; \
                int x0o = xo + is * 16 + 4 * kk; \
                float4 xv0 = float4(AFX(x0o +  0), AFX(x0o +  1), AFX(x0o +  2), AFX(x0o +  3)); \
                float4 xv1 = float4(AFX(x0o + 32), AFX(x0o + 33), AFX(x0o + 34), AFX(x0o + 35)); \
                float4 xv2 = float4(AFX(x0o + 64), AFX(x0o + 65), AFX(x0o + 66), AFX(x0o + 67)); \
                float4 xv3 = float4(AFX(x0o + 96), AFX(x0o + 97), AFX(x0o + 98), AFX(x0o + 99)); \
                a0 += (float4(lo & 0xF) + float4((h >> 0) & 3) * 16.0f - 32.0f) * xv0; \
                a1 += (float4(hi & 0xF) + float4((h >> 2) & 3) * 16.0f - 32.0f) * xv1; \
                a2 += (float4(lo >> 4)  + float4((h >> 4) & 3) * 16.0f - 32.0f) * xv2; \
                a3 += (float4(hi >> 4)  + float4((h >> 6) & 3) * 16.0f - 32.0f) * xv3; \
            } \
            t[is * 4 + 0] = a0.x + a0.y + a0.z + a0.w; \
            t[is * 4 + 1] = a1.x + a1.y + a1.z + a1.w; \
            t[is * 4 + 2] = a2.x + a2.y + a2.z + a2.w; \
            t[is * 4 + 3] = a3.x + a3.y + a3.z + a3.w; \
        } \
        s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] + \
                  sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]); \
    } \
    SV = s; }

// k_qknorm replica at the shape enc_qknorm_n actually dispatches it: 64
// threads, red tree from 32. All 256 threads reach every barrier (role and
// has_qn/has_kn are threadgroup-uniform); only tid < 64 touch data, exactly
// the lanes the unfused dispatch had.
#define AF_QKNORM(P, W) { \
    device float *pp = P; \
    float sq = 0; \
    if (tid < 64) { \
        for (int i = tid; i < hd; i += 64) sq += pp[i] * pp[i]; \
        red[tid] = sq; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = 32; off > 0; off >>= 1) { \
        if (tid < off) red[tid] += red[tid + off]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    float rq = rsqrt(red[0] / hd + a.eps); \
    if (tid < 64) \
        for (int i = tid; i < hd; i += 64) pp[i] = pp[i] * rq * W[i]; \
    threadgroup_barrier(mem_flags::mem_device); \
}

#define AF_PARAMS \
    device const uchar *wb    [[buffer(0)]], \
    device const float *x     [[buffer(1)]], \
    device const float *nw    [[buffer(2)]], \
    device float       *q_out [[buffer(3)]], \
    device float       *kt    [[buffer(4)]], \
    device float       *vt    [[buffer(5)]], \
    device uchar       *kc    [[buffer(6)]], \
    device uchar       *vc    [[buffer(7)]], \
    device const float *fr    [[buffer(8)]], \
    device const float *bq    [[buffer(9)]], \
    device const float *bk    [[buffer(10)]], \
    device const float *bv    [[buffer(11)]], \
    constant attn_front_args &a [[buffer(12)]], \
    device const float *qnw   [[buffer(13)]], \
    device const float *knw   [[buffer(14)]], \
    uint3 tid3  [[thread_position_in_threadgroup]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]], \
    uint  tiisg [[thread_index_in_simdgroup]]

#define AF_BODY(DQ, DK, DV) \
    int u = (int)tgpig.x, hd = a.head_dim; \
    int role, head; \
    if (u < a.n_head)               { role = 0; head = u; } \
    else if (u < a.n_head + a.n_kv) { role = 1; head = u - a.n_head; } \
    else                            { role = 2; head = u - a.n_head - a.n_kv; } \
    ulong w_off = role == 0 ? a.wq_off : role == 1 ? a.wk_off : a.wv_off; \
    device float *yout = role == 0 ? q_out : role == 1 ? kt : vt; \
    device const float *bias = role == 0 ? bq : role == 1 ? bk : bv; \
    int has_bias = role == 0 ? a.has_bq : role == 1 ? a.has_bk : a.has_bv; \
    for (int rr = (int)sgitg; rr < hd; rr += 8) { \
        int row = head * hd + rr; \
        float sv = 0; \
        if (role == 0)      DQ(sv) \
        else if (role == 1) DK(sv) \
        else                DV(sv) \
        sv = simd_sum(sv); \
        if (tiisg == 0) yout[row] = has_bias ? sv + bias[row] : sv; \
    } \
    threadgroup_barrier(mem_flags::mem_device); \
    int hf = a.half_dim, rd = 2 * hf; \
    device half *kh = (device half *)(kc + a.kc_off); \
    device half *vh = (device half *)(vc + a.vc_off); \
    if (role == 0) { \
        if (a.has_qn) AF_QKNORM(q_out + head * hd, qnw) \
        for (int j = tid; j < hf; j += 256) { \
            float ang = a.pos * fr[j]; \
            device float *p = q_out + head * hd; \
            int i0 = a.neox ? j : 2 * j; \
            int i1 = a.neox ? j + hf : i0 + 1; \
            float2 y = rope_rot(p[i0], p[i1], ang, a.mscale); \
            p[i0] = y.x; \
            p[i1] = y.y; \
        } \
    } else if (role == 1) { \
        if (a.has_kn) AF_QKNORM(kt + head * hd, knw) \
        for (int j = tid; j < hf; j += 256) { \
            float ang = a.pos * fr[j]; \
            device float *p = kt + head * hd; \
            int i0 = a.neox ? j : 2 * j; \
            int i1 = a.neox ? j + hf : i0 + 1; \
            float2 y = rope_rot(p[i0], p[i1], ang, a.mscale); \
            p[i0] = y.x; \
            p[i1] = y.y; \
            kh[head * hd + i0] = (half)y.x; \
            kh[head * hd + i1] = (half)y.y; \
        } \
        for (int j = rd + tid; j < hd; j += 256) \
            kh[head * hd + j] = (half)kt[head * hd + j]; \
    } else { \
        for (int j = tid; j < hd; j += 256) \
            vh[head * hd + j] = (half)vt[head * hd + j]; \
    }

// Staged form: k_rmsnorm's exact strided sum and 256-slot tree, normed row
// parked in threadgroup memory for all three projections.
#define AF_KERNEL(NAME, DQ, DK, DV) \
kernel void NAME(AF_PARAMS) { \
    threadgroup float xb[7936]; \
    threadgroup float red[256]; \
    uint tid = tid3.x; \
    int NE = a.n_embd; \
    float s0 = 0; \
    for (int i = tid; i < NE; i += 256) s0 += x[i] * x[i]; \
    red[tid] = s0; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = 128; off > 0; off >>= 1) { \
        if (tid < off) red[tid] += red[tid + off]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    float r = rsqrt(red[0] / NE + a.eps); \
    for (int i = tid; i < NE; i += 256) xb[i] = x[i] * r * nw[i]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    AF_BODY(DQ, DK, DV) \
}

#define AFX(i) xb[i]
AF_KERNEL(k_attn_front_q8_0,    AF_DOT_Q8_0, AF_DOT_Q8_0, AF_DOT_Q8_0)
AF_KERNEL(k_attn_front_q4_0,    AF_DOT_Q4_0, AF_DOT_Q4_0, AF_DOT_Q4_0)
AF_KERNEL(k_attn_front_q4_k,    AF_DOT_Q4_K, AF_DOT_Q4_K, AF_DOT_Q4_K)
AF_KERNEL(k_attn_front_q6_k,    AF_DOT_Q6_K, AF_DOT_Q6_K, AF_DOT_Q6_K)
AF_KERNEL(k_attn_front_q4k_q6k, AF_DOT_Q4_K, AF_DOT_Q4_K, AF_DOT_Q6_K)
#undef AFX

// residual add + the FOLLOWING rmsnorm: two dispatches -> one. The add and
// the reduction reproduce k_add and k_rmsnorm exactly (tid-strided square
// sum, 256-slot tree, same rounding path); x is updated in place and the
// normed row lands in y, exactly as the pair of kernels left them.
struct add_norm_args { int n; float eps; };

kernel void k_add_rmsnorm(device float       *x [[buffer(0)]],
                          device const float *d [[buffer(1)]],
                          device float       *y [[buffer(2)]],
                          device const float *w [[buffer(3)]],
                          constant add_norm_args &a [[buffer(4)]],
                          uint3 tid3 [[thread_position_in_threadgroup]],
                          uint3 tpg3 [[threads_per_threadgroup]]) {
    threadgroup float red[256];
    uint tid = tid3.x, tpg = tpg3.x;
    int n = a.n;
    for (int i = tid; i < n; i += tpg) x[i] += d[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (int i = tid; i < n; i += tpg) s += x[i] * x[i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = tpg / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = rsqrt(red[0] / n + a.eps);
    for (int i = tid; i < n; i += tpg) y[i] = x[i] * r * w[i];
}

// ---------------------------------------------------------------- attention
// One threadgroup per head: scores -> softmax -> weighted value sum.

// pos is the FIRST column's position; column c attends over [.., pos + c], so
// a prompt batch runs every token's attention in one dispatch (grid.y = col).
// Each column keeps its own score row and output slice, so the arithmetic per
// (head, position) is exactly what a per-token dispatch computed.
struct attn_args {
    int   head_dim, n_head, n_head_kv, n_ctx, pos;
    ulong l_off;      // this layer's byte offset into the kv cache
    float scale;
    int   q8;
    int   window;     // sliding-window size for this layer (0 = full)
    int   has_sinks;  // gpt-oss: per-head sink joins softmax denominator only
    int   q_stride, att_stride, out_stride;
    int   kv_rows;    // ring capacity for this layer's rows (0 = flat)
};

// Ring-aware KV row index: absolute position t lives at t % kv_rows when the
// window-aware ring is on. Same arithmetic as the CPU's model_kv_row_at and
// CUDA's kv_slot — three backends, one layout rule.
static inline ulong kv_row_off(int t, int kv_rows, ulong row_b) {
    return (ulong)(kv_rows > 0 ? t % kv_rows : t) * row_b;
}

// Cooperative-read variant of the K-score loop (RUNNER_METAL_ATTN_COOP=1).
//
// The 2026-08-15 diagnosis measured decode's KV read at 1.52 GB/s, ~18x worse
// per byte than the weight read in the same forward, and named the cause:
// k_attn gives each THREAD a whole KV row, so adjacent lanes address memory
// `row_b` apart -- 1024 B on e2b -- and one simdgroup load touches 32 distinct
// cache lines. (The V accumulation below is already coalesced: consecutive
// threads take consecutive elements of the SAME row, so only the score loop
// scatters, and only that loop changes here.)
//
// Cooperative form: one SIMDGROUP owns a row and its 32 lanes split the head
// dimension, so a load covers 32 consecutive elements instead of 32 rows, and
// the per-row dot finishes with one simd_sum.
//
// That reduction is why this is a separate kernel rather than an edit: summing
// 32 lane partials through a reduction tree is not the sequential accumulation
// k_attn performs, so the result is NOT bit-identical and the identity route
// cannot host it. It answers to tests/test_attn_tol.c the way the fast matvec
// answers to test_mv_tol.c, and ships off until a measurement promotes it.
static inline float kv_dot_coop(device const uchar *row, device const float *qh,
                                int hd, int q8, uint lane) {
    float s = 0;
    if (q8) {
        // One 32-element block per lane step: the block's scale is read once
        // and the 32 quants beneath it are contiguous.
        for (int b = (int)lane; b < hd / 32; b += 32) {
            device const uchar *blk = row + (ulong)b * 34;
            float d = (float)*(device const half *)blk;
            device const char *q = (device const char *)(blk + 2);
            device const float *xp = qh + b * 32;
            float t = 0;
            for (int j = 0; j < 32; j++) t += xp[j] * (float)q[j];
            s += d * t;
        }
    } else {
        device const half *k = (device const half *)row;
        for (int i = (int)lane; i < hd; i += 32) s += qh[i] * (float)k[i];
    }
    return simd_sum(s);
}

// Shared prologue and epilogue for the two attention kernels below. They
// differ ONLY in how the K scores are computed; keeping the rest in one place
// is what stops the gated variant from drifting from the route it is measured
// against.
#define ATTN_PROLOGUE \
    /* Decode gives this kernel only n_head threadgroups (8 on gemma-3-4b) on */ \
    /* an 8-core GPU, and its work grows linearly with context, so */ \
    /* threads-per-group is the only parallelism lever short of splitting the */ \
    /* KV range across threadgroups. 128 -> 256 is worth ~11% at 3k context; */ \
    /* 512 and 1024 measured IDENTICAL to 256 on an M1 (the pipeline allows */ \
    /* 1024), because per-thread work shrinks as fast as the reduction's */ \
    /* barrier count grows. 256 is therefore the measured size, and the extra */ \
    /* threadgroup scratch for 1024 buys nothing. The host still clamps to the */ \
    /* pipeline limit and to a power of two, which these reductions require. */ \
    threadgroup float red[256]; \
    uint tid = tid3.x, tpg = tpg3.x; \
    uint h = tgpig.x, col = tgpig.y; \
    int hd = a.head_dim; \
    int kvh = h / (a.n_head / a.n_head_kv); \
    int kv_dim = a.n_head_kv * hd; \
    ulong row_b = kv_row_bytes(kv_dim, a.q8); \
    ulong base = a.l_off + kv_head_off(kvh, hd, a.q8); \
    int pos = a.pos + (int)col; \
    device const float *q = q_all + (ulong)col * a.q_stride; \
    device float *att = att_all + (ulong)col * a.att_stride; \
    device float *out = out_all + (ulong)col * a.out_stride; \
    device const float *qh = q + h * hd; \
    device float *ah = att + (ulong)h * a.n_ctx; \
    int t0 = 0; \
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1; \


#define ATTN_EPILOGUE \
 \
    /* max */ \
    float mx = -1e30f; \
    for (int t = t0 + tid; t <= pos; t += tpg) mx = max(mx, ah[t]); \
    if (a.has_sinks && tid == 0) mx = max(mx, sinks[h]); \
    red[tid] = mx; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] = max(red[tid], red[tid + off]); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    mx = red[0]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    /* exp + sum */ \
    float sum = 0; \
    for (int t = t0 + tid; t <= pos; t += tpg) { \
        float e = exp(ah[t] - mx); \
        ah[t] = e; \
        sum += e; \
    } \
    if (a.has_sinks && tid == 0) sum += exp(sinks[h] - mx); \
    red[tid] = sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] += red[tid + off]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    sum = red[0]; \
    threadgroup_barrier(mem_flags::mem_device); \
 \
    for (int i = tid; i < hd; i += tpg) { \
        float o = 0; \
        for (int t = t0; t <= pos; t++) \
            o += ah[t] * kv_pair(vc + base + kv_row_off(t, a.kv_rows, row_b), i / 2, a.q8)[i & 1]; \
        out[h * hd + i] = o / sum; \
    }

kernel void k_attn(device const float *q_all   [[buffer(0)]],
                   device const uchar *kc      [[buffer(1)]],
                   device const uchar *vc      [[buffer(2)]],
                   device float       *att_all [[buffer(3)]],
                   device float       *out_all [[buffer(4)]],
                   constant attn_args &a   [[buffer(5)]],
                   device const float *sinks [[buffer(6)]],
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   uint3 tid3 [[thread_position_in_threadgroup]],
                   uint3 tpg3 [[threads_per_threadgroup]]) {
    ATTN_PROLOGUE
    for (int t = t0 + tid; t <= pos; t += tpg) {
        ah[t] = kv_dot(kc + base + kv_row_off(t, a.kv_rows, row_b), qh, hd, a.q8) * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
    ATTN_EPILOGUE
}

// Cooperative-read twin: one simdgroup owns a KV row and its lanes split
// head_dim, so a load covers 32 consecutive elements instead of 32 rows.
// Everything after the score loop is the shared epilogue.
kernel void k_attn_coop(device const float *q_all   [[buffer(0)]],
                   device const uchar *kc      [[buffer(1)]],
                   device const uchar *vc      [[buffer(2)]],
                   device float       *att_all [[buffer(3)]],
                   device float       *out_all [[buffer(4)]],
                   constant attn_args &a   [[buffer(5)]],
                   device const float *sinks [[buffer(6)]],
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   uint3 tid3 [[thread_position_in_threadgroup]],
                   uint3 tpg3 [[threads_per_threadgroup]]) {
    ATTN_PROLOGUE
    {
        uint lane = tid & 31u, sg = tid >> 5, n_sg = tpg >> 5;
        for (int t = t0 + (int)sg; t <= pos; t += (int)n_sg) {
            float sc = kv_dot_coop(kc + base + kv_row_off(t, a.kv_rows, row_b), qh, hd,
                                   a.q8, lane) * a.scale;
            if (lane == 0) ah[t] = sc;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    ATTN_EPILOGUE
}

// ------------------------------------------------- chunked decode attention
//
// k_attn gets n_head threadgroups and nothing else, so decode attention runs
// 8 threadgroups on an 8-core GPU while its work grows linearly with context.
// Widening threads-per-group bought ~11% at 3k and then saturated; the ceiling
// is threadgroup COUNT. These two kernels split the position range instead, so
// the grid becomes n_head * n_chunks.
//
// Standard online-softmax decomposition: each chunk reduces its own slice to
// (max, sum, unnormalised weighted V), and the combine pass rescales every
// chunk onto a common maximum. exp(s-m) never sees a positive argument, so the
// split cannot overflow where the single-pass kernel would not.
//
// The scores still land in the shared att buffer: chunks own disjoint position
// ranges, so they cannot collide.

struct attn_chunk_args {
    int   head_dim, n_head, n_head_kv, n_ctx, pos;
    ulong l_off;
    float scale;
    int   q8;
    int   window;
    int   chunk;       // positions per chunk
    int   n_chunks;
    int   kv_rows;     // ring capacity (0 = flat); see kv_row_off
    // No column strides: the host takes this path at n == 1 only (see the
    // n == 1 guard on the chunk dispatch in metal.m), so q, att and the
    // partials are all indexed from column zero.
};

/* Shared halves of the chunked attention kernel; the two variants below
   differ only in how the K scores are read. */
#define ATTNC_PROLOGUE \
    threadgroup float red[256]; \
    uint tid = tid3.x, tpg = tpg3.x; \
    uint h = tgpig.x, z = tgpig.y; \
    int hd = a.head_dim; \
    int kvh = h / (a.n_head / a.n_head_kv); \
    int kv_dim = a.n_head_kv * hd; \
    ulong row_b = kv_row_bytes(kv_dim, a.q8); \
    ulong base = a.l_off + kv_head_off(kvh, hd, a.q8); \
    int pos = a.pos; \
    device const float *qh = q_all + h * hd; \
    device float *ah = att_all + (ulong)h * a.n_ctx; \
 \
    int t0 = 0; \
    if (a.window > 0 && pos - a.window + 1 > 0) t0 = pos - a.window + 1; \
    int lo = t0 + (int)z * a.chunk; \
    int hi = min(lo + a.chunk - 1, pos); \
 \
    device float *acc = acc_all + ((ulong)h * a.n_chunks + z) * hd; \
    device float *ms  = ms_all  + ((ulong)h * a.n_chunks + z) * 2; \
 \
    if (lo > hi) { /* empty chunk: neutral partial */ \
        for (int i = tid; i < hd; i += tpg) acc[i] = 0; \
        if (tid == 0) { ms[0] = -1e30f; ms[1] = 0; } \
        return; \
    } \


#define ATTNC_EPILOGUE \
 \
    float mx = -1e30f; \
    for (int t = lo + (int)tid; t <= hi; t += tpg) mx = max(mx, ah[t]); \
    red[tid] = mx; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] = max(red[tid], red[tid + off]); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    mx = red[0]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
 \
    float sum = 0; \
    for (int t = lo + (int)tid; t <= hi; t += tpg) { \
        float e = exp(ah[t] - mx); \
        ah[t] = e; \
        sum += e; \
    } \
    red[tid] = sum; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint off = tpg / 2; off > 0; off >>= 1) { \
        if (tid < off) red[tid] += red[tid + off]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    sum = red[0]; \
    threadgroup_barrier(mem_flags::mem_device); \
 \
    /* UNNORMALISED: the combine pass owns the division, because the divisor is */ \
    /* not known until every chunk's maximum is. */ \
    for (int i = tid; i < hd; i += tpg) { \
        float o = 0; \
        for (int t = lo; t <= hi; t++) \
            o += ah[t] * kv_pair(vc + base + kv_row_off(t, a.kv_rows, row_b), i / 2, a.q8)[i & 1]; \
        acc[i] = o; \
    } \
    if (tid == 0) { ms[0] = mx; ms[1] = sum; }

kernel void k_attn_chunk(device const float *q_all   [[buffer(0)]],
                         device const uchar *kc      [[buffer(1)]],
                         device const uchar *vc      [[buffer(2)]],
                         device float       *att_all [[buffer(3)]],
                         device float       *acc_all [[buffer(4)]],
                         device float       *ms_all  [[buffer(5)]],
                         constant attn_chunk_args &a [[buffer(6)]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint3 tid3  [[thread_position_in_threadgroup]],
                         uint3 tpg3  [[threads_per_threadgroup]]) {
    ATTNC_PROLOGUE
    for (int t = lo + (int)tid; t <= hi; t += tpg)
        ah[t] = kv_dot(kc + base + kv_row_off(t, a.kv_rows, row_b), qh, hd, a.q8) * a.scale;
    threadgroup_barrier(mem_flags::mem_device);
    ATTNC_EPILOGUE
}

// Cooperative-read twin of the chunked kernel. This is the one that matters
// at long context: decode takes the chunked path whenever the span is worth
// splitting, so k_attn_coop above only ever runs on short spans.
kernel void k_attn_chunk_coop(device const float *q_all   [[buffer(0)]],
                         device const uchar *kc      [[buffer(1)]],
                         device const uchar *vc      [[buffer(2)]],
                         device float       *att_all [[buffer(3)]],
                         device float       *acc_all [[buffer(4)]],
                         device float       *ms_all  [[buffer(5)]],
                         constant attn_chunk_args &a [[buffer(6)]],
                         uint3 tgpig [[threadgroup_position_in_grid]],
                         uint3 tid3  [[thread_position_in_threadgroup]],
                         uint3 tpg3  [[threads_per_threadgroup]]) {
    ATTNC_PROLOGUE
    {
        uint lane = tid & 31u, sg = tid >> 5, n_sg = tpg >> 5;
        for (int t = lo + (int)sg; t <= hi; t += (int)n_sg) {
            float sc = kv_dot_coop(kc + base + kv_row_off(t, a.kv_rows, row_b), qh, hd,
                                   a.q8, lane) * a.scale;
            if (lane == 0) ah[t] = sc;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    ATTNC_EPILOGUE
}

struct attn_comb_args { int head_dim, n_head, n_chunks, has_sinks; };

kernel void k_attn_combine(device const float *acc_all [[buffer(0)]],
                           device const float *ms_all  [[buffer(1)]],
                           device float       *out_all [[buffer(2)]],
                           constant attn_comb_args &a  [[buffer(3)]],
                           device const float *sinks   [[buffer(4)]],
                           uint3 tgpig [[threadgroup_position_in_grid]],
                           uint3 tid3  [[thread_position_in_threadgroup]],
                           uint3 tpg3  [[threads_per_threadgroup]]) {
    uint tid = tid3.x, tpg = tpg3.x;
    uint h = tgpig.x;
    int hd = a.head_dim;
    device const float *ms = ms_all + (ulong)h * a.n_chunks * 2;

    float M = -1e30f;
    for (int z = 0; z < a.n_chunks; z++) M = max(M, ms[z * 2]);
    if (a.has_sinks) M = max(M, sinks[h]);

    float denom = 0;
    for (int z = 0; z < a.n_chunks; z++) denom += ms[z * 2 + 1] * exp(ms[z * 2] - M);
    // A sink joins the denominator only -- it contributes no value vector --
    // exactly as the single-pass kernel treats it.
    if (a.has_sinks) denom += exp(sinks[h] - M);

    for (int i = tid; i < hd; i += tpg) {
        float o = 0;
        for (int z = 0; z < a.n_chunks; z++)
            o += acc_all[((ulong)h * a.n_chunks + z) * hd + i] * exp(ms[z * 2] - M);
        out_all[h * hd + i] = o / denom;
    }
}

// ---------------------------------------------------------------- elementwise

// The elementwise family takes the prompt batch in grid.y: thread (i, col)
// touches element i of column col, at gs/us elements per column. Each element
// was always independent, so this is BIT-IDENTICAL -- only the encoding
// changes. It exists because these ops were 175n of the 240n per-token
// dispatches measured in docs/metal-dispatch-census-2026-08-13.md, and
// batching the 65n of qknorm/headnorm was already worth +2.1% prefill.
// At n_col == 1 the strides are never read.
kernel void k_silu_mul(device float       *g [[buffer(0)]],
                       device const float *u [[buffer(1)]],
                       constant int       &n [[buffer(2)]],
                       constant int       &gs [[buffer(3)]],
                       constant int       &us [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) {
        device float *gp = g + (ulong)gid.y * gs;
        device const float *up = u + (ulong)gid.y * us;
        float x = gp[gid.x];
        gp[gid.x] = (x / (1.0f + exp(-x))) * up[gid.x];
    }
}

// attention output gate (afmoe / muse-glimmer): x *= sigmoid(gate), the
// same arithmetic as the CPU's 1/(1+expf(-g)) and CUDA's k_q35_attn_gate.
// exp() overflow on a large negative gate saturates to 0 exactly as libm's
// expf does, so no clamp is needed for identity with the CPU oracle.
kernel void k_sigmoid_mul(device float       *x [[buffer(0)]],
                          device const float *g [[buffer(1)]],
                          constant int       &n [[buffer(2)]],
                          constant int       &xs [[buffer(3)]],
                          constant int       &gs [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) {
        device float *xp = x + (ulong)gid.y * xs;
        device const float *gp = g + (ulong)gid.y * gs;
        xp[gid.x] *= 1.0f / (1.0f + exp(-gp[gid.x]));
    }
}

kernel void k_gelu_mul(device float       *g [[buffer(0)]],
                       device const float *u [[buffer(1)]],
                       constant int       &n [[buffer(2)]],
                       constant int       &gs [[buffer(3)]],
                       constant int       &us [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) {
        device float *gp = g + (ulong)gid.y * gs;
        device const float *up = u + (ulong)gid.y * us;
        float x = gp[gid.x];
        // Metal compiles with fast math, where tanh() is evaluated through
        // exp(2a): for large |a| that overflows to inf and inf/inf yields NaN,
        // while the CPU oracle's libm tanhf saturates. Gemma-class models
        // reach it — gemma-3-4b's layer-0 gate produced NaN logits here, and
        // the model emitted only token 0. Clamping to a magnitude where tanh
        // is already exactly +/-1.0f in fp32 cannot change any representable
        // result, so this guard is invisible to the identity gates. Same
        // hazard, same fix as the `g < -80` early-out in the CPU silu path.
        float a = 0.7978845608f * (x + 0.044715f * x * x * x);
        float t = tanh(clamp(a, -16.0f, 16.0f));
        gp[gid.x] = 0.5f * x * (1.0f + t) * up[gid.x];
    }
}

kernel void k_add(device float       *x [[buffer(0)]],
                  device const float *d [[buffer(1)]],
                  constant int       &n [[buffer(2)]],
                  constant int       &xs [[buffer(3)]],
                  constant int       &ds [[buffer(4)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n)
        x[(ulong)gid.y * xs + gid.x] += d[(ulong)gid.y * ds + gid.x];
}

kernel void k_scale(device float       *x [[buffer(0)]],
                    constant float     &s [[buffer(1)]],
                    constant int       &n [[buffer(2)]],
                    constant int       &xs [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < n) x[(ulong)gid.y * xs + gid.x] *= s;
}

// -------------------------------------------------------------- sparse MoE
// Plain router + fused-3D expert layout. This is the Metal twin of the first
// CUDA MoE slice: softmax -> top-k -> renormalize on device, then one indirect
// expert matvec launch per projection.

kernel void k_moe_route(device const float *logits [[buffer(0)]],
                        device int         *sel    [[buffer(1)]],
                        device float       *selw   [[buffer(2)]],
                        constant int       &ne     [[buffer(3)]],
                        constant int       &used   [[buffer(4)]],
                        constant int       &tokens [[buffer(5)]],
                        constant int       &ls     [[buffer(6)]],
                        uint t [[thread_position_in_grid]]) {
    if ((int)t >= tokens) return;
    float lg[256];
    device const float *src = logits + (ulong)t * ls;
    for (int e = 0; e < ne; e++) lg[e] = src[e];

    float mx = lg[0];
    for (int e = 1; e < ne; e++)
        if (lg[e] > mx) mx = lg[e];
    float sum = 0.0f;
    for (int e = 0; e < ne; e++) {
        float p = exp(lg[e] - mx);
        lg[e] = p;
        sum += p;
    }
    for (int e = 0; e < ne; e++) lg[e] /= sum;

    device int *ts = sel + (ulong)t * used;
    device float *tw = selw + (ulong)t * used;
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
    if (denom < 6.103515625e-5f) denom = 6.103515625e-5f;
    for (int s = 0; s < used; s++) tw[s] /= denom;
}

kernel void k_trace_copy_f32(device const float *src [[buffer(0)]],
                             device float       *dst [[buffer(1)]],
                             constant int       &n   [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if ((int)i < n) dst[i] = src[i];
}

struct moe_args {
    int   n_in;
    int   n_out;
    ulong w_off;
    ulong estride;
    int   xs;
    int   ys;
    int   has_bias;
    int   bias_stride;
    int   slots_per_token;
    int   n_slots;   // total (token, selected-expert) slots; expert-major only
};

// One dot body per quant type, shared verbatim by BOTH dispatch shapes below
// (slot-major k_moe_mv_* and expert-major k_moe_mv_em_*). One definition is
// what makes the two shapes bit-identical: the per-(row, slot) arithmetic and
// its accumulation order are the same instructions either way — only which
// threadgroup executes them changes.

static inline float moe_dot_f32(device const uchar *wbase, uint row, int n_in,
                                device const float *xp0, uint tiisg) {
    float s = 0;
    device const float *rw = (device const float *)wbase + (ulong)row * n_in;
    for (int i = tiisg; i < n_in; i += 32) s += rw[i] * xp0[i];
    return s;
}

static inline float moe_dot_f16(device const uchar *wbase, uint row, int n_in,
                                device const float *xp0, uint tiisg) {
    float s = 0;
    device const half *rw = (device const half *)wbase + (ulong)row * n_in;
    for (int i = tiisg; i < n_in; i += 32) s += (float)rw[i] * xp0[i];
    return s;
}

static inline float moe_dot_q8_0(device const uchar *wbase, uint row, int n_in,
                                 device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 34;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2);
        device const float *xp = xp0 + b * 32;
        float t = 0;
        for (int j = 0; j < 32; j++) t += (float)q[j] * xp[j];
        s += d * t;
    }
    return s;
}

static inline float moe_dot_q4_0(device const uchar *wbase, uint row, int n_in,
                                 device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 18;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        device const float *xp = xp0 + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++)
            t += ((int)(q[j] & 0xF) - 8) * xp[j] + ((int)(q[j] >> 4) - 8) * xp[j + 16];
        s += d * t;
    }
    return s;
}

#define MOE_MV_HEAD \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    if (row >= (uint)a.n_out) return; \
    uint slot = tgpig.y; \
    device const uchar *wbase = wb + a.w_off + (ulong)sel[slot] * a.estride; \
    uint xrow = a.slots_per_token ? slot / a.slots_per_token : slot; \
    device const float *xp0 = x + (ulong)xrow * a.xs; \
    float s = 0;

#define MOE_MV_TAIL \
    s = simd_sum(s); \
    if (tiisg == 0) { \
        if (a.has_bias) s += bias[(ulong)sel[slot] * a.bias_stride + row]; \
        y[(ulong)slot * a.ys + row] = s; \
    }

#define MOE_MV_PARAMS \
    device const uchar *wb  [[buffer(0)]], \
    device const float *x   [[buffer(1)]], \
    device float       *y   [[buffer(2)]], \
    constant moe_args  &a   [[buffer(3)]], \
    device const int   *sel [[buffer(4)]], \
    device const float *bias [[buffer(5)]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]], \
    uint  tiisg [[thread_index_in_simdgroup]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint3 ntg   [[threads_per_threadgroup]]

kernel void k_moe_mv_f32(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_f32(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

kernel void k_moe_mv_f16(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_f16(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q8_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_q8_0(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

kernel void k_moe_mv_q4_0(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_q4_0(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

// q2_K uses the same quarter-superblock decomposition as k_mv_q2_K. Keeping
// that layout here matters for narrow expert FFNs: a 1536-wide Qwen3 expert
// has only six whole superblocks, but 24 independent quarters for the lanes.
static inline float moe_dot_q2_K(device const uchar *wbase, uint row, int n_in,
                                 device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 84;
    int nq = nb * 4;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 84;
        device const uchar *sc = blk + (jj >> 1) * 8 + (jj & 1) * 4;
        device const uchar *q  = blk + 16 + (jj >> 1) * 32;
        float d    = (float)*(device const half *)(blk + 80);
        float dmin = (float)*(device const half *)(blk + 82);
        device const float *xp = xp0 + b * 256 + jj * 64;
        uchar sh0 = (uchar)((jj & 1) * 4), sh1 = (uchar)(sh0 + 2);
        uchar c0 = sc[0], c1 = sc[1], c2 = sc[2], c3 = sc[3];
        float d0 = d * (c0 & 0xF), m0 = dmin * (c0 >> 4);
        float d1 = d * (c1 & 0xF), m1 = dmin * (c1 >> 4);
        float d2 = d * (c2 & 0xF), m2 = dmin * (c2 >> 4);
        float d3 = d * (c3 & 0xF), m3 = dmin * (c3 >> 4);
        float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        float sx0 = 0, sx1 = 0, sx2 = 0, sx3 = 0;
        for (int l = 0; l < 16; l++) {
            uchar qa = q[l], qb = q[l + 16];
            float x0 = xp[l], x1 = xp[l + 16], x2 = xp[l + 32], x3 = xp[l + 48];
            t0 += (float)((qa >> sh0) & 3) * x0; sx0 += x0;
            t1 += (float)((qb >> sh0) & 3) * x1; sx1 += x1;
            t2 += (float)((qa >> sh1) & 3) * x2; sx2 += x2;
            t3 += (float)((qb >> sh1) & 3) * x3; sx3 += x3;
        }
        s += d0 * t0 - m0 * sx0 + d1 * t1 - m1 * sx1
           + d2 * t2 - m2 * sx2 + d3 * t3 - m3 * sx3;
    }
    return s;
}

kernel void k_moe_mv_q2_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_q2_K(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

static inline float moe_dot_q3_K(device const uchar *wbase, uint row, int n_in,
                                 device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 110;
    int nq = nb * 4;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 110;
        device const uchar *hm  = blk;
        device const uchar *q   = blk + 32 + (jj >> 1) * 32;
        float d_all = (float)*(device const half *)(blk + 108);
        char sc[16];
        q3k_scales(blk + 96, sc);
        device const float *xp = xp0 + b * 256 + jj * 64;
        int si = (jj >> 1) * 8 + (jj & 1) * 4;
        uchar sh0 = (uchar)((jj & 1) * 4), sh1 = (uchar)(sh0 + 2);
        uchar m0 = (uchar)(1u << ((jj >> 1) * 4 + (jj & 1) * 2));
        uchar m1 = (uchar)(m0 << 1);
        float d0 = d_all * (float)((int)sc[si + 0] - 32);
        float d1 = d_all * (float)((int)sc[si + 1] - 32);
        float d2 = d_all * (float)((int)sc[si + 2] - 32);
        float d3 = d_all * (float)((int)sc[si + 3] - 32);
        float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (int l = 0; l < 16; l++) {
            uchar qa = q[l], qb = q[l + 16];
            uchar ha = hm[l], hb = hm[l + 16];
            t0 += (float)((int)((qa >> sh0) & 3) - ((ha & m0) ? 0 : 4)) * xp[l];
            t1 += (float)((int)((qb >> sh0) & 3) - ((hb & m0) ? 0 : 4)) * xp[l + 16];
            t2 += (float)((int)((qa >> sh1) & 3) - ((ha & m1) ? 0 : 4)) * xp[l + 32];
            t3 += (float)((int)((qb >> sh1) & 3) - ((hb & m1) ? 0 : 4)) * xp[l + 48];
        }
        s += d0 * t0 + d1 * t1 + d2 * t2 + d3 * t3;
    }
    return s;
}

kernel void k_moe_mv_q3_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_q3_K(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

// The MoE kernels are twins of the dense k_mv_* family and carried the same
// occupancy defect: a whole 256-weight superblock per lane means nb =
// n_in/256 lanes work, which is 11 of 32 for a 2816-wide expert. Same fix,
// same units -- quarters for q4_K/q5_K, halves for q6_K -- chosen so no
// weight byte is read twice. The dense side measured 5.43 -> 9.22 tok/s from
// this plus vectorisation.
static inline float moe_dot_q4_K(device const uchar *wbase, uint row, int n_in,
                                 device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 144;
    int nq = nb * 4;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 144;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *q  = blk + 16 + jj * 32;
        device const float *xp = xp0 + b * 256 + jj * 64;
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        float t1 = 0, t2 = 0, sx1 = 0, sx2 = 0;
        for (int l = 0; l < 32; l++) {
            t1 += (float)(q[l] & 0xF) * xp[l];      sx1 += xp[l];
            t2 += (float)(q[l] >> 4)  * xp[l + 32]; sx2 += xp[l + 32];
        }
        s += d1 * t1 - mm1 * sx1 + d2 * t2 - mm2 * sx2;
    }
    return s;
}

kernel void k_moe_mv_q4_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_q4_K(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

static inline float moe_dot_q5_K(device const uchar *wbase, uint row, int n_in,
                                 device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 176;
    int nq = nb * 4;
    for (int u = tiisg; u < nq; u += 32) {
        int b = u >> 2, jj = u & 3;
        device const uchar *blk = rw + (ulong)b * 176;
        float d    = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        device const uchar *qh = blk + 16;   // shared by all four quarters
        device const uchar *q  = blk + 48 + jj * 32;
        device const float *xp = xp0 + b * 256 + jj * 64;
        uchar s1, m1, s2, m2;
        get_scale_min_k4(jj * 2 + 0, sc, &s1, &m1);
        get_scale_min_k4(jj * 2 + 1, sc, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        uchar u1 = (uchar)(1u << (2 * jj)), u2 = (uchar)(2u << (2 * jj));
        for (int l = 0; l < 32; l++) {
            s += (d1 * (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1) * xp[l];
            s += (d2 * (float)((q[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - mm2) * xp[l + 32];
        }
    }
    return s;
}

kernel void k_moe_mv_q5_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_q5_K(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

static inline float moe_dot_q6_K(device const uchar *wbase, uint row, int n_in,
                                 device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 256;
    device const uchar *rw = wbase + (ulong)row * nb * 210;
    int nh = nb * 2;
    for (int u = tiisg; u < nh; u += 32) {
        int b = u >> 1, half_i = u & 1;
        device const uchar *blk = rw + (ulong)b * 210;
        device const uchar *ql = blk + half_i * 64;
        device const uchar *qh = blk + 128 + half_i * 32;
        device const char  *sc = (device const char *)(blk + 192) + half_i * 8;
        float d = (float)*(device const half *)(blk + 208);
        device const float *xp = xp0 + b * 256 + half_i * 128;
        float t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        // Vectorised exactly as the dense twin, which this shape is worth the
        // most on: four outputs per iteration assembled from ql and qh with
        // per-element shifts is more than the compiler folds. packed_uchar4
        // because a 210-byte superblock leaves ql/qh at odd alignments.
        for (int is = 0; is < 2; is++) {
            device const packed_uchar4 *l4  = (device const packed_uchar4 *)(ql + is * 16);
            device const packed_uchar4 *l4b = (device const packed_uchar4 *)(ql + 32 + is * 16);
            device const packed_uchar4 *h4  = (device const packed_uchar4 *)(qh + is * 16);
            device const packed_float4 *x0 = (device const packed_float4 *)(xp + is * 16);
            device const packed_float4 *x1 = (device const packed_float4 *)(xp + 32 + is * 16);
            device const packed_float4 *x2 = (device const packed_float4 *)(xp + 64 + is * 16);
            device const packed_float4 *x3 = (device const packed_float4 *)(xp + 96 + is * 16);
            float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int k = 0; k < 4; k++) {
                uchar4 lo = l4[k], hi = l4b[k], h = h4[k];
                a0 += (float4(lo & 0xF) + float4((h >> 0) & 3) * 16.0f - 32.0f) * x0[k];
                a1 += (float4(hi & 0xF) + float4((h >> 2) & 3) * 16.0f - 32.0f) * x1[k];
                a2 += (float4(lo >> 4)  + float4((h >> 4) & 3) * 16.0f - 32.0f) * x2[k];
                a3 += (float4(hi >> 4)  + float4((h >> 6) & 3) * 16.0f - 32.0f) * x3[k];
            }
            t[is * 4 + 0] = a0.x + a0.y + a0.z + a0.w;
            t[is * 4 + 1] = a1.x + a1.y + a1.z + a1.w;
            t[is * 4 + 2] = a2.x + a2.y + a2.z + a2.w;
            t[is * 4 + 3] = a3.x + a3.y + a3.z + a3.w;
        }
        s += d * (sc[0] * t[0] + sc[2] * t[1] + sc[4] * t[2] + sc[6] * t[3] +
                  sc[1] * t[4] + sc[3] * t[5] + sc[5] * t[6] + sc[7] * t[7]);
    }
    return s;
}

kernel void k_moe_mv_q6_K(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_q6_K(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

static inline float moe_dot_mxfp4(device const uchar *wbase, uint row, int n_in,
                                  device const float *xp0, uint tiisg) {
    float s = 0;
    int nb = n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 17;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        device const float *xp = xp0 + b * 32;
        float t = 0;
        for (int j = 0; j < 16; j++) {
            t += kv_mxfp4[q[j] & 0xF] * xp[j];
            t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
        }
        s += d * t;
    }
    return s;
}

kernel void k_moe_mv_mxfp4(MOE_MV_PARAMS) {
    MOE_MV_HEAD;
    s += moe_dot_mxfp4(wbase, row, a.n_in, xp0, tiisg);
    MOE_MV_TAIL;
}

// ------------------------------------------------ expert-major MoE matvec
//
// The slot-major family above runs one threadgroup row per (token, selected
// expert) SLOT, so at prefill batch N every expert's weight rows are re-read
// once per slot that selected it — ~N*used/n_expert redundant reads of the
// dominant traffic. This family inverts the grid: one threadgroup row per
// EXPERT. The threadgroup cooperatively compacts the slots routed to its
// expert into threadgroup memory, then each simdgroup walks that list with
// the SAME moe_dot_* body, reading the expert's weight rows once and reusing
// them across every matching token from cache.
//
// Output placement is unchanged (y[slot * ys + row]) and each (slot, row) has
// exactly one writer, so no atomics touch the results and the outputs are
// bit-identical to the slot-major family — gated, not assumed. Cold experts
// cost one compaction scan and exit before any weight byte is read.
#define MOE_EM_CHUNK 1024

#define MOE_EM_KERNEL(NAME, DOTFN) \
kernel void NAME(MOE_MV_PARAMS) { \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    int  ex = (int)tgpig.y; \
    uint tpitg = sgitg * 32 + tiisg; \
    uint nth = ntg.x; \
    threadgroup int list[MOE_EM_CHUNK]; \
    threadgroup atomic_int cnt; \
    device const uchar *wbase = wb + a.w_off + (ulong)ex * a.estride; \
    for (int base = 0; base < a.n_slots; base += MOE_EM_CHUNK) { \
        int lim = a.n_slots - base; \
        if (lim > MOE_EM_CHUNK) lim = MOE_EM_CHUNK; \
        if (tpitg == 0) \
            atomic_store_explicit(&cnt, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int i = (int)tpitg; i < lim; i += (int)nth) \
            if (sel[base + i] == ex) { \
                int k = atomic_fetch_add_explicit(&cnt, 1, \
                                                  memory_order_relaxed); \
                list[k] = base + i; \
            } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        int c = atomic_load_explicit(&cnt, memory_order_relaxed); \
        if (row < (uint)a.n_out) \
            for (int k = 0; k < c; k++) { \
                int slot = list[k]; \
                uint xrow = a.slots_per_token ? (uint)slot / a.slots_per_token \
                                              : (uint)slot; \
                device const float *xp0 = x + (ulong)xrow * a.xs; \
                float s = DOTFN(wbase, row, a.n_in, xp0, tiisg); \
                s = simd_sum(s); \
                if (tiisg == 0) { \
                    if (a.has_bias) \
                        s += bias[(ulong)ex * a.bias_stride + row]; \
                    y[(ulong)slot * a.ys + row] = s; \
                } \
            } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
}

// Column-tiled twins for the hottest expert quant types: one weight block
// load feeds up to 8 gathered token columns, with a separate accumulator per
// column so every (row, slot) dot keeps the exact block order and reduction
// of the single-column body — still byte-identical, now with the weight
// reuse in registers where the flat expert-major family measured that cache
// alone does not pay.
static inline void moe_dot8_q8_0(device const uchar *wbase, uint row, int n_in,
                                 const thread device const float **xps,
                                 int ncol, uint tiisg, thread float *acc) {
    int nb = n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 34;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2);
        for (int c = 0; c < ncol; c++) {
            device const float *xp = xps[c] + b * 32;
            float t = 0;
            for (int j = 0; j < 32; j++) t += (float)q[j] * xp[j];
            acc[c] += d * t;
        }
    }
}

static inline void moe_dot8_mxfp4(device const uchar *wbase, uint row, int n_in,
                                  const thread device const float **xps,
                                  int ncol, uint tiisg, thread float *acc) {
    int nb = n_in / 32;
    device const uchar *rw = wbase + (ulong)row * nb * 17;
    for (int b = tiisg; b < nb; b += 32) {
        device const uchar *blk = rw + (ulong)b * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        for (int c = 0; c < ncol; c++) {
            device const float *xp = xps[c] + b * 32;
            float t = 0;
            for (int j = 0; j < 16; j++) {
                t += kv_mxfp4[q[j] & 0xF] * xp[j];
                t += kv_mxfp4[q[j] >> 4]  * xp[j + 16];
            }
            acc[c] += d * t;
        }
    }
}

#define MOE_EM8_KERNEL(NAME, DOTFN8) \
kernel void NAME(MOE_MV_PARAMS) { \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    int  ex = (int)tgpig.y; \
    uint tpitg = sgitg * 32 + tiisg; \
    uint nth = ntg.x; \
    threadgroup int list[MOE_EM_CHUNK]; \
    threadgroup atomic_int cnt; \
    device const uchar *wbase = wb + a.w_off + (ulong)ex * a.estride; \
    for (int base = 0; base < a.n_slots; base += MOE_EM_CHUNK) { \
        int lim = a.n_slots - base; \
        if (lim > MOE_EM_CHUNK) lim = MOE_EM_CHUNK; \
        if (tpitg == 0) \
            atomic_store_explicit(&cnt, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int i = (int)tpitg; i < lim; i += (int)nth) \
            if (sel[base + i] == ex) { \
                int k = atomic_fetch_add_explicit(&cnt, 1, \
                                                  memory_order_relaxed); \
                list[k] = base + i; \
            } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        int c = atomic_load_explicit(&cnt, memory_order_relaxed); \
        if (row < (uint)a.n_out) \
            for (int k0 = 0; k0 < c; k0 += 8) { \
                int ncol = c - k0 < 8 ? c - k0 : 8; \
                device const float *xps[8]; \
                int slots8[8]; \
                for (int cc = 0; cc < ncol; cc++) { \
                    int slot = list[k0 + cc]; \
                    slots8[cc] = slot; \
                    uint xrow = a.slots_per_token \
                                    ? (uint)slot / a.slots_per_token \
                                    : (uint)slot; \
                    xps[cc] = x + (ulong)xrow * a.xs; \
                } \
                float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
                DOTFN8(wbase, row, a.n_in, xps, ncol, tiisg, acc); \
                for (int cc = 0; cc < ncol; cc++) { \
                    float s = simd_sum(acc[cc]); \
                    if (tiisg == 0) { \
                        if (a.has_bias) \
                            s += bias[(ulong)ex * a.bias_stride + row]; \
                        y[(ulong)slots8[cc] * a.ys + row] = s; \
                    } \
                } \
            } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
}

MOE_EM8_KERNEL(k_moe_mv_em8_q8_0,  moe_dot8_q8_0)
MOE_EM8_KERNEL(k_moe_mv_em8_mxfp4, moe_dot8_mxfp4)

// ---------------------------------------- grouped simdgroup-MMA MoE prefill
//
// The third grouping shape, and the one that mirrors what dense prefill
// already does: group the batch's slots by expert (k_moe_group), then run
// each expert's token group through the SAME simdgroup-MMA tile structure as
// k_mm_* — the x tile stage gathers columns through the group's slot map, and
// the store scatters them back, so no physical repack of x ever happens.
// Like dense k_mm this stages operands in half and reassociates the k-sum,
// so it is NOT byte-identical to the matvec path: it answers to the
// tolerance gates (test-moe-tol, test-gpu-identity), exactly as
// RUNNER_METAL_MM did before its promotion.
//
// Column order within an expert's group comes from atomic appends and is not
// deterministic — and does not need to be: every output column of a
// simdgroup MMA is an independent dot in fixed k-order, so a slot's value is
// the same wherever it lands in the tile, and zero-padded columns touch
// nothing. Outputs are therefore bit-stable across runs even though the
// grouping is not.

kernel void k_moe_group(device const int *sel    [[buffer(0)]],
                        device int       *colmap [[buffer(1)]],
                        device int       *eoff   [[buffer(2)]],
                        constant int     &ne     [[buffer(3)]],
                        constant int     &nslots [[buffer(4)]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint nth [[threads_per_threadgroup]]) {
    threadgroup atomic_int hist[257];   // n_expert is admitted at <= 256
    for (int e = (int)tid; e <= ne; e += (int)nth)
        atomic_store_explicit(&hist[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)tid; i < nslots; i += (int)nth)
        atomic_fetch_add_explicit(&hist[sel[i]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {   // exclusive scan; hist becomes the running cursors
        int run = 0;
        for (int e = 0; e < ne; e++) {
            int c = atomic_load_explicit(&hist[e], memory_order_relaxed);
            eoff[e] = run;
            atomic_store_explicit(&hist[e], run, memory_order_relaxed);
            run += c;
        }
        eoff[ne] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)tid; i < nslots; i += (int)nth) {
        int pos = atomic_fetch_add_explicit(&hist[sel[i]], 1,
                                            memory_order_relaxed);
        colmap[pos] = i;
    }
}

struct moe_mm_args {
    int   n_in, n_out;
    ulong w_off, estride;
    int   x_stride, y_stride, has_bias, bias_stride, slots_per_token;
};

#define MOE_MM_PARAMS \
    device const uchar *wb     [[buffer(0)]], \
    device const float *x      [[buffer(1)]], \
    device float       *y      [[buffer(2)]], \
    constant moe_mm_args &a    [[buffer(3)]], \
    device const float *bias   [[buffer(4)]], \
    device const int   *colmap [[buffer(5)]], \
    device const int   *eoff   [[buffer(6)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    uint3 tpitg [[thread_position_in_threadgroup]], \
    uint  sgitg [[simdgroup_index_in_threadgroup]]

// MM_BODY with three substitutions: the weight base carries the expert's
// stride, the x stage gathers its columns through the group map, and the
// store scatters through the same map — plus one deliberate departure from
// the dense kernel: operands stage in FLOAT, not half. The half staging is
// an NVIDIA-shaped economy (their half tensor cores run 2x float); on
// Apple's simdgroup units float matmul runs within ~10% of half (measured
// across M1-M4 in the community metal-benchmarks), so on this backend the
// operand-rounding term of the tolerance budget can simply be bought back.
// What remains vs the matvec path is k-tile reduction order alone — the
// class the identity gates are calibrated for. The half-staged twins are
// kept selectable (RUNNER_METAL_MOE_MM=half) so the trade stays measurable.
#define MOE_MM_BODY(...) \
    const int ex   = (int)tgpig.z; \
    const int cbeg = eoff[ex]; \
    const int count = eoff[ex + 1] - cbeg; \
    const int col0 = (int)tgpig.y * MM_TN; \
    if (col0 >= count) return; \
    const ulong w_off_ex = a.w_off + (ulong)ex * a.estride; \
    threadgroup float tg_w[MM_TM * MM_TK]; \
    threadgroup float tg_x[MM_TN * MM_TK]; \
    threadgroup float tg_c[MM_TN * MM_TM]; \
    const int row0 = (int)tgpig.x * MM_TM; \
    const int tid  = (int)tpitg.x; \
    simdgroup_float8x8 acc[MM_RPS][MM_CG]; \
    for (int rg = 0; rg < MM_RPS; rg++) \
        for (int cg = 0; cg < MM_CG; cg++) acc[rg][cg] = simdgroup_float8x8(0.0f); \
    for (int k0 = 0; k0 < a.n_in; k0 += MM_TK) { \
        for (int p = 0; p < MM_RPS; p++) { \
            int r = p * 32 + (tid >> 2), sub = (tid & 3) * 8; \
            int row = row0 + r; \
            threadgroup float *dst = tg_w + r * MM_TK + sub; \
            if (row < a.n_out) { __VA_ARGS__; } \
            else { for (int j = 0; j < 8; j++) dst[j] = 0.0f; } \
        } \
        for (int i = tid * 4; i < MM_TN * MM_TK; i += 128 * 4) { \
            for (int j = 0; j < 4; j++) { \
                int idx = i + j, cc = idx / MM_TK, kk = idx % MM_TK; \
                float v = 0.0f; \
                if (col0 + cc < count) { \
                    int slot = colmap[cbeg + col0 + cc]; \
                    uint xrow = a.slots_per_token \
                                    ? (uint)slot / a.slots_per_token \
                                    : (uint)slot; \
                    v = x[(ulong)xrow * a.x_stride + k0 + kk]; \
                } \
                tg_x[idx] = v; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int kk = 0; kk < MM_TK / 8; kk++) { \
            simdgroup_float8x8 acols[MM_CG]; \
            for (int cg = 0; cg < MM_CG; cg++) \
                simdgroup_load(acols[cg], tg_x + cg * 8 * MM_TK + kk * 8, MM_TK); \
            for (int rg = 0; rg < MM_RPS; rg++) { \
                simdgroup_float8x8 B; \
                int wrg = (int)sgitg * MM_RPS + rg; \
                simdgroup_load(B, tg_w + wrg * 8 * MM_TK + kk * 8, MM_TK, \
                               ulong2(0, 0), true); \
                for (int cg = 0; cg < MM_CG; cg++) \
                    simdgroup_multiply_accumulate(acc[rg][cg], acols[cg], B, \
                                                  acc[rg][cg]); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (int rg = 0; rg < MM_RPS; rg++) { \
        int wrg = (int)sgitg * MM_RPS + rg; \
        for (int cg = 0; cg < MM_CG; cg++) \
            simdgroup_store(acc[rg][cg], tg_c + cg * 8 * MM_TM + wrg * 8, MM_TM); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int idx = tid; idx < MM_TN * MM_TM; idx += 128) { \
        int cc = idx / MM_TM, rr = idx % MM_TM; \
        if (col0 + cc < count && row0 + rr < a.n_out) { \
            int slot = colmap[cbeg + col0 + cc]; \
            float v = tg_c[idx]; \
            if (a.has_bias) v += bias[(ulong)ex * a.bias_stride + row0 + rr]; \
            y[(ulong)slot * a.y_stride + row0 + rr] = v; \
        } \
    }

// The half-staged twins (dense-kernel economics), kept for measurement.
#define MOE_MMH_BODY(...) \
    const int ex   = (int)tgpig.z; \
    const int cbeg = eoff[ex]; \
    const int count = eoff[ex + 1] - cbeg; \
    const int col0 = (int)tgpig.y * MM_TN; \
    if (col0 >= count) return; \
    const ulong w_off_ex = a.w_off + (ulong)ex * a.estride; \
    threadgroup half  tg_w[MM_TM * MM_TK]; \
    threadgroup half  tg_x[MM_TN * MM_TK]; \
    threadgroup float tg_c[MM_TN * MM_TM]; \
    const int row0 = (int)tgpig.x * MM_TM; \
    const int tid  = (int)tpitg.x; \
    simdgroup_float8x8 acc[MM_RPS][MM_CG]; \
    for (int rg = 0; rg < MM_RPS; rg++) \
        for (int cg = 0; cg < MM_CG; cg++) acc[rg][cg] = simdgroup_float8x8(0.0f); \
    for (int k0 = 0; k0 < a.n_in; k0 += MM_TK) { \
        for (int p = 0; p < MM_RPS; p++) { \
            int r = p * 32 + (tid >> 2), sub = (tid & 3) * 8; \
            int row = row0 + r; \
            threadgroup half *dst = tg_w + r * MM_TK + sub; \
            if (row < a.n_out) { __VA_ARGS__; } \
            else { for (int j = 0; j < 8; j++) dst[j] = 0.0h; } \
        } \
        for (int i = tid * 4; i < MM_TN * MM_TK; i += 128 * 4) { \
            for (int j = 0; j < 4; j++) { \
                int idx = i + j, cc = idx / MM_TK, kk = idx % MM_TK; \
                float v = 0.0f; \
                if (col0 + cc < count) { \
                    int slot = colmap[cbeg + col0 + cc]; \
                    uint xrow = a.slots_per_token \
                                    ? (uint)slot / a.slots_per_token \
                                    : (uint)slot; \
                    v = x[(ulong)xrow * a.x_stride + k0 + kk]; \
                } \
                tg_x[idx] = (half)v; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int kk = 0; kk < MM_TK / 8; kk++) { \
            simdgroup_half8x8 acols[MM_CG]; \
            for (int cg = 0; cg < MM_CG; cg++) \
                simdgroup_load(acols[cg], tg_x + cg * 8 * MM_TK + kk * 8, MM_TK); \
            for (int rg = 0; rg < MM_RPS; rg++) { \
                simdgroup_half8x8 B; \
                int wrg = (int)sgitg * MM_RPS + rg; \
                simdgroup_load(B, tg_w + wrg * 8 * MM_TK + kk * 8, MM_TK, \
                               ulong2(0, 0), true); \
                for (int cg = 0; cg < MM_CG; cg++) \
                    simdgroup_multiply_accumulate(acc[rg][cg], acols[cg], B, \
                                                  acc[rg][cg]); \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (int rg = 0; rg < MM_RPS; rg++) { \
        int wrg = (int)sgitg * MM_RPS + rg; \
        for (int cg = 0; cg < MM_CG; cg++) \
            simdgroup_store(acc[rg][cg], tg_c + cg * 8 * MM_TM + wrg * 8, MM_TM); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int idx = tid; idx < MM_TN * MM_TM; idx += 128) { \
        int cc = idx / MM_TM, rr = idx % MM_TM; \
        if (col0 + cc < count && row0 + rr < a.n_out) { \
            int slot = colmap[cbeg + col0 + cc]; \
            float v = tg_c[idx]; \
            if (a.has_bias) v += bias[(ulong)ex * a.bias_stride + row0 + rr]; \
            y[(ulong)slot * a.y_stride + row0 + rr] = v; \
        } \
    }

kernel void k_moe_mm_q8_0(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nb + k0 / 32) * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2) + sub;
        for (int j = 0; j < 8; j++) dst[j] = d * (float)q[j];
    })
}

kernel void k_moe_mm_mxfp4(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nb + k0 / 32) * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            uchar nib = lane < 16 ? (q[lane] & 0xF) : (q[lane - 16] >> 4);
            dst[j] = d * kv_mxfp4[nib];
        }
    })
}

// The rest of the roster, ratified 2026-09-01 when the owner adopted the
// house fidelity bar as the sparse-routing prefill instrument: each chunk is
// its dense k_mm_* twin verbatim with the per-expert weight base. q5_K has no
// dense mm chunk to twin, so it stays on the matvec path per-tensor.
kernel void k_moe_mm_f32(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        device const float *rw = (device const float *)(wb + w_off_ex)
                               + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = rw[j];
    })
}

kernel void k_moe_mm_f16(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        device const half *rw = (device const half *)(wb + w_off_ex)
                              + (ulong)row * a.n_in + k0 + sub;
        for (int j = 0; j < 8; j++) dst[j] = (float)rw[j];
    })
}

kernel void k_moe_mm_q4_0(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nb + k0 / 32) * 18;
        float d = (float)*(device const half *)blk;
        device const uchar *q = blk + 2;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            int v = lane < 16 ? (int)(q[lane] & 0xF) : (int)(q[lane - 16] >> 4);
            dst[j] = d * (float)(v - 8);
        }
    })
}

kernel void k_moe_mm_q2_K(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255;
        int h = sbk >> 7, rem = sbk & 127, j32 = rem >> 5, rem2 = rem & 31;
        int g = rem2 >> 4, l0 = rem2 & 15;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nsb + sb) * 84;
        float d    = (float)*(device const half *)(blk + 80);
        float dmin = (float)*(device const half *)(blk + 82);
        uchar scb  = blk[h * 8 + j32 * 2 + g];
        float dl = d * (scb & 0xF), ml = dmin * (scb >> 4);
        device const uchar *q = blk + 16 + h * 32 + g * 16;
        uchar shift = (uchar)(2 * j32);
        for (int j = 0; j < 8; j++)
            dst[j] = dl * (float)((q[l0 + j] >> shift) & 3) - ml;
    })
}

kernel void k_moe_mm_q3_K(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        int k = k0 + sub, nsb = a.n_in / 256;
        int sb = k >> 8, sbk = k & 255;
        int h = sbk >> 7, rem = sbk & 127, j32 = rem >> 5, rem2 = rem & 31;
        int g = rem2 >> 4, l0 = rem2 & 15;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nsb + sb) * 110;
        float d_all = (float)*(device const half *)(blk + 108);
        char sc[16];
        q3k_scales(blk + 96, sc);
        float dl = d_all * (float)((int)sc[h * 8 + j32 * 2 + g] - 32);
        device const uchar *hm = blk + g * 16;
        device const uchar *q  = blk + 32 + h * 32 + g * 16;
        uchar shift = (uchar)(2 * j32);
        uchar mbit  = (uchar)(1u << (h * 4 + j32));
        for (int j = 0; j < 8; j++)
            dst[j] = dl * (float)((int)((q[l0 + j] >> shift) & 3)
                                  - ((hm[l0 + j] & mbit) ? 0 : 4));
    })
}

kernel void k_moe_mm_q4_K(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        int nsb = a.n_in / 256;
        int sb  = k0 / 256;
        int j32 = (k0 % 256) / 32;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nsb + sb) * 144;
        float dall = (float)*(device const half *)blk;
        float dmin = (float)*(device const half *)(blk + 2);
        device const uchar *sc = blk + 4;
        uchar s1, m1;
        get_scale_min_k4(j32, sc, &s1, &m1);
        float d = dall * s1, mm2 = dmin * m1;
        device const uchar *q = blk + 16 + (j32 / 2) * 32;
        bool hi = (j32 & 1) != 0;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            int v = hi ? (int)(q[lane] >> 4) : (int)(q[lane] & 0xF);
            dst[j] = d * (float)v - mm2;
        }
    })
}

kernel void k_moe_mm_q6_K(MOE_MM_PARAMS) {
    MOE_MM_BODY({
        int nsb = a.n_in / 256;
        int sb  = k0 / 256;
        int off = k0 % 256;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nsb + sb) * 210;
        float dall = (float)*(device const half *)(blk + 208);
        device const char *scs = (device const char *)(blk + 192);
        int half_i = off / 128, within = off % 128;
        device const uchar *ql = blk + half_i * 64;
        device const uchar *qh = blk + 128 + half_i * 32;
        device const char  *sc = scs + half_i * 8;
        for (int j = 0; j < 8; j++) {
            int lane = within + sub + j;
            int l = lane % 32, grp = lane / 32;
            int qb = (int)ql[l + (grp & 1) * 32];
            int qlv = (grp >= 2) ? (qb >> 4) : (qb & 0xF);
            int qhv = ((int)(qh[l] >> (grp * 2)) & 3) << 4;
            int is = (l / 16) & 1;
            dst[j] = dall * (float)sc[grp * 2 + is] * (float)((qlv | qhv) - 32);
        }
    })
}

kernel void k_moe_mmh_q8_0(MOE_MM_PARAMS) {
    MOE_MMH_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nb + k0 / 32) * 34;
        float d = (float)*(device const half *)blk;
        device const char *q = (device const char *)(blk + 2) + sub;
        for (int j = 0; j < 8; j++) dst[j] = d * (float)q[j];
    })
}

kernel void k_moe_mmh_mxfp4(MOE_MM_PARAMS) {
    MOE_MMH_BODY({
        int nb = a.n_in / 32;
        device const uchar *blk = wb + w_off_ex + ((ulong)row * nb + k0 / 32) * 17;
        float d = ldexp(1.0f, (int)blk[0] - 127);
        device const uchar *q = blk + 1;
        for (int j = 0; j < 8; j++) {
            int lane = sub + j;
            uchar nib = lane < 16 ? (q[lane] & 0xF) : (q[lane - 16] >> 4);
            dst[j] = d * kv_mxfp4[nib];
        }
    })
}

MOE_EM_KERNEL(k_moe_mv_em_f32,   moe_dot_f32)
MOE_EM_KERNEL(k_moe_mv_em_f16,   moe_dot_f16)
MOE_EM_KERNEL(k_moe_mv_em_q8_0,  moe_dot_q8_0)
MOE_EM_KERNEL(k_moe_mv_em_q4_0,  moe_dot_q4_0)
MOE_EM_KERNEL(k_moe_mv_em_q2_K,  moe_dot_q2_K)
MOE_EM_KERNEL(k_moe_mv_em_q3_K,  moe_dot_q3_K)
MOE_EM_KERNEL(k_moe_mv_em_q4_K,  moe_dot_q4_K)
MOE_EM_KERNEL(k_moe_mv_em_q5_K,  moe_dot_q5_K)
MOE_EM_KERNEL(k_moe_mv_em_q6_K,  moe_dot_q6_K)
MOE_EM_KERNEL(k_moe_mv_em_mxfp4, moe_dot_mxfp4)

static inline float swiglu_oai(float g, float u) {
    const float alpha = 1.702f, limit = 7.0f;
    float x = g < limit ? g : limit;
    float y = clamp(u, -limit, limit);
    float gl = x < -50.0f ? 0.0f : x / (1.0f + exp(alpha * -x));
    return gl * (y + 1.0f);
}

kernel void k_moe_actmul(device float       *gbuf [[buffer(0)]],
                         device const float *ubuf [[buffer(1)]],
                         constant int4      &args [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int nff = args.x, gss = args.y, uss = args.z, act = args.w;
    int i = gid.x, slot = gid.y;
    if (i >= nff) return;
    device float *g = gbuf + (ulong)slot * gss;
    device const float *u = ubuf + (ulong)slot * uss;
    float x = g[i];
    if (act == 2) {
        g[i] = swiglu_oai(x, u[i]);
    } else if (act == 1) {
        // Same fast-math tanh overflow as k_gelu_mul, and the same fix. This
        // kernel is the ROUTED-expert twin of that dense path: the clamp was
        // applied there when gemma-3-4b's layer-0 gate produced NaN, but this
        // copy was missed, so gemma-4-26B-A4B reached it at layer 3 and the
        // model emitted only token 0 on Metal while the CPU arm was correct.
        // tanh is already exactly +/-1.0f in fp32 well inside +/-16, so the
        // clamp cannot change a representable result.
        float t = tanh(clamp(0.7978845608f * (x + 0.044715f * x * x * x),
                             -16.0f, 16.0f));
        g[i] = 0.5f * x * (1.0f + t) * u[i];
    } else {
        g[i] = (x / (1.0f + exp(-x))) * u[i];
    }
}

// gate-mv + up-mv + actmul: three dispatches -> one. Per (row, slot) one
// simdgroup computes the gate dot and the up dot with the SAME per-type dot
// bodies the separate matvecs use (same order, simd_sum reduction), adds the
// per-expert biases exactly where the matvec tails added them, and applies
// the activation copied verbatim from k_moe_actmul (including the tanh
// clamp that kernel's history mandates). hb receives exactly the bytes the
// three-kernel chain produced.
struct moe_gua_args {
    int   n_in, n_out;
    ulong g_off, u_off, estride_g, estride_u;
    int   xs, has_bias, act, slots_per_token, ys;
};

#define MOE_GUA_KERNEL(NAME, DOTFN) \
kernel void NAME(device const uchar *wb    [[buffer(0)]], \
                 device const float *x     [[buffer(1)]], \
                 device float       *y     [[buffer(2)]], \
                 constant moe_gua_args &a  [[buffer(3)]], \
                 device const int   *sel   [[buffer(4)]], \
                 device const float *gbias [[buffer(5)]], \
                 device const float *ubias [[buffer(6)]], \
                 uint  sgitg [[simdgroup_index_in_threadgroup]], \
                 uint  tiisg [[thread_index_in_simdgroup]], \
                 uint3 tgpig [[threadgroup_position_in_grid]], \
                 uint3 ntg   [[threads_per_threadgroup]]) { \
    uint row = tgpig.x * (ntg.x / 32) + sgitg; \
    if (row >= (uint)a.n_out) return; \
    uint slot = tgpig.y; \
    int  ex = sel[slot]; \
    uint xrow = a.slots_per_token ? slot / a.slots_per_token : slot; \
    device const float *xp0 = x + (ulong)xrow * a.xs; \
    float sg = DOTFN(wb + a.g_off + (ulong)ex * a.estride_g, row, a.n_in, \
                     xp0, tiisg); \
    float su = DOTFN(wb + a.u_off + (ulong)ex * a.estride_u, row, a.n_in, \
                     xp0, tiisg); \
    sg = simd_sum(sg); \
    su = simd_sum(su); \
    if (tiisg != 0) return; \
    if (a.has_bias) { \
        sg += gbias[(ulong)ex * a.n_out + row]; \
        su += ubias[(ulong)ex * a.n_out + row]; \
    } \
    float out; \
    if (a.act == 2) { \
        out = swiglu_oai(sg, su); \
    } else if (a.act == 1) { \
        float t = tanh(clamp(0.7978845608f * (sg + 0.044715f * sg * sg * sg), \
                             -16.0f, 16.0f)); \
        out = 0.5f * sg * (1.0f + t) * su; \
    } else { \
        out = (sg / (1.0f + exp(-sg))) * su; \
    } \
    y[(ulong)slot * a.ys + row] = out; \
}

MOE_GUA_KERNEL(k_moe_gua_f32,   moe_dot_f32)
MOE_GUA_KERNEL(k_moe_gua_f16,   moe_dot_f16)
MOE_GUA_KERNEL(k_moe_gua_q8_0,  moe_dot_q8_0)
MOE_GUA_KERNEL(k_moe_gua_q4_0,  moe_dot_q4_0)
MOE_GUA_KERNEL(k_moe_gua_q2_K,  moe_dot_q2_K)
MOE_GUA_KERNEL(k_moe_gua_q3_K,  moe_dot_q3_K)
MOE_GUA_KERNEL(k_moe_gua_q4_K,  moe_dot_q4_K)
MOE_GUA_KERNEL(k_moe_gua_q5_K,  moe_dot_q5_K)
MOE_GUA_KERNEL(k_moe_gua_q6_K,  moe_dot_q6_K)
MOE_GUA_KERNEL(k_moe_gua_mxfp4, moe_dot_mxfp4)

kernel void k_moe_sum(device float       *out    [[buffer(0)]],
                      device const float *eout   [[buffer(1)]],
                      device const float *selw   [[buffer(2)]],
                      device const float *dscale [[buffer(3)]],
                      device const int   *sel    [[buffer(4)]],
                      constant int       &n      [[buffer(5)]],
                      constant int       &nslots [[buffer(6)]],
                      constant int       &es     [[buffer(7)]],
                      constant int       &has_dscale [[buffer(8)]],
                      constant int       &tokens [[buffer(9)]],
                      constant int       &out_stride [[buffer(10)]],
                      constant int       &fuse_add   [[buffer(11)]],
                      uint2 gid [[thread_position_in_grid]]) {
    int i = gid.x, token = gid.y;
    if ((int)i >= n) return;
    if (token >= tokens) return;
    int slot0 = token * nslots;
    float s = 0.0f;
    for (int slot = 0; slot < nslots; slot++) {
        int si = slot0 + slot;
        float w = selw[si] * (has_dscale ? dscale[sel[si]] : 1.0f);
        s += w * eout[(ulong)si * es + i];
    }
    // Decode fusion F5: the post-FFN residual add folded into the expert
    // sum — out IS the residual stream and x[i] + s is character for
    // character what k_add computed from the stored sum. One fewer
    // dispatch per layer; byte identity is the gate, as ever.
    device float *o = out + (ulong)token * out_stride + i;
    *o = fuse_add ? *o + s : s;
}
