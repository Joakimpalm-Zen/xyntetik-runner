// Training products on the device (R8.19), included at the end of kernels.cu
// (every codebook and table above is in scope).
//
// One element decoder per weight type, tdq_<type>(row, i): weight i of a row,
// the value quants.c dequant_row() writes for it. Every type's value is exact
// in f32 (a half-precision scale times small integers, at most 24 significant
// bits; a K-quant's minimum and Q4_1/Q5_1's offset subtract or add one more
// exact value, a single rounding on both sides), so the association the CPU
// writes and a contracted multiply-add here land on the same bits;
// test-train-gpu holds every type to the CPU chain byte for byte.
//
// Two kernels on each decoder:
//
//   k_tmvt_<type>  dx[t][i] += sum_j W[j][i] dy[t][j], the CPU trainer's
//                  chain exactly (model.c mvt_worker): each element starts
//                  from dx's incoming value, walks j serially with fmaf and
//                  skips a zero dy. A block decodes a weight tile once for
//                  TV_TT positions; the chains are per element, so the
//                  tiling cannot move a bit.
//
//   k_tmm_<type>   y[t][j] = sum_i W[j][i] x[t][i] (k_tmmw_: 32 columns a
//                  block instead of 16), batch-invariant: lane L
//                  of the row's warp sums i = L, L + 32, ... with fmaf, the
//                  warp folds the 32 partials in one fixed tree, so column t's
//                  value is the same whatever columns share the launch. The
//                  taped forward and the backward's recompute batch the
//                  window differently and must see the same values; this is
//                  what makes them agree. Not the CPU's bits (another order).

#define TQ_BLK(BS, TS) (rw + (ulong64)((i) / (BS)) * (TS))

static __device__ __forceinline__ float tdq_f32(const uchar *rw, int i) {
    return ((const float *)rw)[i];
}
static __device__ __forceinline__ float tdq_f16(const uchar *rw, int i) {
    return f16f(rw + (ulong64)i * 2);
}
static __device__ __forceinline__ float tdq_bf16(const uchar *rw, int i) {
    return __uint_as_float(((uint)((const ushort16 *)rw)[i]) << 16);
}
static __device__ __forceinline__ float tdq_q8_0(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(32, 34);
    return (float)((const signed char *)(b + 2))[i & 31] * f16f(b);
}
static __device__ __forceinline__ float tdq_q4_0(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(32, 18);
    int e = i & 31;
    int q = e < 16 ? (b[2 + e] & 0xF) : (b[2 + e - 16] >> 4);
    return (float)(q - 8) * f16f(b);
}
static __device__ __forceinline__ float tdq_q4_1(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(32, 20);
    int e = i & 31;
    int q = e < 16 ? (b[4 + e] & 0xF) : (b[4 + e - 16] >> 4);
    return (float)q * f16f(b) + f16f(b + 2);
}
static __device__ __forceinline__ float tdq_q5_0(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(32, 22);
    int e = i & 31;
    unsigned qh = iq_ld32a2(b + 2);
    int q = (e < 16 ? (b[6 + e] & 0xF) : (b[6 + e - 16] >> 4)) | (((qh >> e) & 1) << 4);
    return (float)(q - 16) * f16f(b);
}
static __device__ __forceinline__ float tdq_q5_1(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(32, 24);
    int e = i & 31;
    unsigned qh = iq_ld32a2(b + 4);
    int q = (e < 16 ? (b[8 + e] & 0xF) : (b[8 + e - 16] >> 4)) | (((qh >> e) & 1) << 4);
    return (float)q * f16f(b) + f16f(b + 2);
}
// Q2_K: 16 scale bytes, qs[64], d, dmin. Half n = e / 128 has quant bytes
// qs + 32n; its 32-element run j = (e % 128) / 32 takes shift 2j and scale
// byte 8n + 2j (+ 1 for the run's second 16).
static __device__ __forceinline__ float tdq_q2_K(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 84);
    int e = i & 255, n = e >> 7, r = e & 127, j = r >> 5, l = r & 31;
    unsigned sc = b[8 * n + 2 * j + (l >> 4)];
    float dl = f16f(b + 80) * (float)(sc & 0xF), ml = f16f(b + 82) * (float)(sc >> 4);
    return dl * (float)((b[16 + 32 * n + l] >> (2 * j)) & 3) - ml;
}
// Q3_K: hmask[32], qs[64], 12 scale bytes (sixteen 6-bit signed scales, the
// unpack of dq_q3_K), d. Element e: half n, run j as Q2_K, high-mask bit
// 4n + j of hmask[l].
static __device__ __forceinline__ float tdq_q3_K(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 110);
    int e = i & 255, n = e >> 7, r = e & 127, j = r >> 5, l = r & 31;
    int is = 8 * n + 2 * j + (l >> 4);
    const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
    unsigned a0 = iq_ld32a2(b + 96), a1 = iq_ld32a2(b + 100), a2 = iq_ld32a2(b + 104);
    unsigned aux;
    switch (is >> 2) {
    case 0:  aux = (a0 & km2) | (((a2 >> 0) & km1) << 4); break;
    case 1:  aux = (a1 & km2) | (((a2 >> 2) & km1) << 4); break;
    case 2:  aux = ((a0 >> 4) & km2) | (((a2 >> 4) & km1) << 4); break;
    default: aux = ((a1 >> 4) & km2) | (((a2 >> 6) & km1) << 4); break;
    }
    int sc = (signed char)(aux >> (8 * (is & 3)));
    float dl = f16f(b + 108) * (float)(sc - 32);
    int q = ((b[32 + 32 * n + l] >> (2 * j)) & 3) - ((b[l] & (1u << (4 * n + j))) ? 0 : 4);
    return dl * (float)q;
}
// Q4_K / Q5_K: d, dmin, 12 scale bytes, (Q5_K: qh[32]), qs. Sub-block
// g = e / 32 is nibble half g % 2 of quant bytes 32 (g / 2) + l.
static __device__ __forceinline__ float tdq_q4_K(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 144);
    int e = i & 255, g = e >> 5, l = e & 31;
    uchar sc, mn;
    get_scale_min_k4(g, b + 4, &sc, &mn);
    uchar byte = b[16 + 32 * (g >> 1) + l];
    int q = (g & 1) ? (byte >> 4) : (byte & 0xF);
    return f16f(b) * (float)sc * (float)q - f16f(b + 2) * (float)mn;
}
static __device__ __forceinline__ float tdq_q5_K(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 176);
    int e = i & 255, g = e >> 5, l = e & 31;
    uchar sc, mn;
    get_scale_min_k4(g, b + 4, &sc, &mn);
    uchar byte = b[48 + 32 * (g >> 1) + l];
    int q = ((g & 1) ? (byte >> 4) : (byte & 0xF)) + ((b[16 + l] >> g) & 1 ? 16 : 0);
    return f16f(b) * (float)sc * (float)q - f16f(b + 2) * (float)mn;
}
// Q6_K: ql[128], qh[64], 16 signed scales, d; the arithmetic of k_mvt_q6_K
static __device__ __forceinline__ float tdq_q6_K(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 210);
    int e = i & 255, half = e >> 7, r = e & 127, seg = r >> 5, l = r & 31;
    const uchar *ql = b + half * 64, *qh = b + 128 + half * 32;
    int s = ((const signed char *)(b + 192))[half * 8 + 2 * seg + (l >> 4)];
    int lo = (seg & 1) ? ql[l + 32] : ql[l];
    int q = (int)(((seg & 2) ? (lo >> 4) : (lo & 0xF)) | (((qh[l] >> (2 * seg)) & 3) << 4)) - 32;
    return f16f(b + 208) * (float)(s * q);
}
static __device__ __forceinline__ float tdq_iq4_nl(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(32, 18);
    int e = i & 31;
    int q = e < 16 ? (b[2 + e] & 0xF) : (b[2 + e - 16] >> 4);
    return f16f(b) * (float)kv_iq4[q];
}
// IQ4_XS: d, scales_h (16 bits), scales_l[4], qs[128]
static __device__ __forceinline__ float tdq_iq4_xs(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 136);
    int e = i & 255, ib = e >> 5, l = e & 31;
    int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) | (((iq_ld16(b + 2) >> 2 * ib) & 3) << 4);
    uchar byte = b[8 + 16 * ib + (l & 15)];
    int q = l < 16 ? (byte & 0xF) : (byte >> 4);
    return f16f(b) * (float)(ls - 32) * (float)kv_iq4[q];
}
// The codebook i-quants: element e is weight j = e % 8 of group l = (e % 32)
// / 8 of sub-block ib = e / 32, as dq_iq* lays them out.
static __device__ __forceinline__ float tdq_iq2_xxs(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 66);
    int e = i & 255, ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned aux = iq_ld32a2(b + 2 + 8 * ib + 4);
    float db = f16f(b) * (0.5f + (float)(aux >> 28)) * 0.25f;
    return db * iq_w8(kiq2xxs_grid[b[2 + 8 * ib + l]], j, iq_signs7((aux >> (7 * l)) & 127));
}
static __device__ __forceinline__ float tdq_iq2_xs(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 74);
    int e = i & 255, ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned sc = b[66 + ib];
    float db = f16f(b) * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
    unsigned q = iq_ld16(b + 2 + 2 * (4 * ib + l));
    return db * iq_w8(kiq2xs_grid[q & 511], j, iq_signs7(q >> 9));
}
static __device__ __forceinline__ float tdq_iq2_s(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 82);
    int e = i & 255, ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned sc = b[74 + ib];
    float db = f16f(b) * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
    unsigned idx = b[2 + 4 * ib + l] | ((b[66 + ib] << (8 - 2 * l)) & 0x300);
    return db * iq_w8(kiq2s_grid[idx], j, b[34 + 4 * ib + l]);
}
static __device__ __forceinline__ float tdq_iq3_xxs(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 98);
    int e = i & 255, ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned aux = iq_ld32a2(b + 66 + 4 * ib);
    float db = f16f(b) * (0.5f + (float)(aux >> 28)) * 0.5f;
    unsigned signs = iq_signs7((aux >> (7 * l)) & 127);
    unsigned g = kiq3xxs_grid[b[2 + 8 * ib + 2 * l + (j >> 2)]];
    float mag = (float)((g >> (8 * (j & 3))) & 0xFF);
    return db * ((signs >> j) & 1 ? -mag : mag);
}
static __device__ __forceinline__ float tdq_iq3_s(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 110);
    int e = i & 255, ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned sc = b[106 + (ib >> 1)];
    float db = f16f(b) * (float)(1 + 2 * ((ib & 1) ? (sc >> 4) : (sc & 0xF)));
    unsigned hb = b[66 + ib];
    unsigned idx = j < 4 ? (b[2 + 8 * ib + 2 * l] | ((hb << (8 - 2 * l)) & 256))
                         : (b[2 + 8 * ib + 2 * l + 1] | ((hb << (7 - 2 * l)) & 256));
    float mag = (float)((kiq3s_grid[idx] >> (8 * (j & 3))) & 0xFF);
    return db * ((b[74 + 4 * ib + l] >> j) & 1 ? -mag : mag);
}
static __device__ __forceinline__ float tdq_iq1_s(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 50);
    int e = i & 255, ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned qh = iq_ld16(b + 34 + 2 * ib);
    float dl = f16f(b) * (float)(2 * ((qh >> 12) & 7) + 1);
    float delta = qh & 0x8000 ? -IQ1_DELTA : IQ1_DELTA;
    return dl * iq1_w(kiq1s_grid[b[2 + 4 * ib + l] | (((qh >> 3 * l) & 7) << 8)], j, delta);
}
// IQ1_M: qs[32], qh[16], four 16-bit scale words whose top nibbles hold the
// block's fp16 scale (dq_iq1_m)
static __device__ __forceinline__ float tdq_iq1_m(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(256, 56);
    int e = i & 255, ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned s0 = iq_ld16(b + 48), s1 = iq_ld16(b + 50), s2 = iq_ld16(b + 52), s3 = iq_ld16(b + 54);
    float d = iq_f16_bits((s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000));
    unsigned sw = iq_ld16(b + 48 + 2 * (ib / 2));
    float dl = d * (float)(2 * ((sw >> (6 * (ib % 2) + (l < 2 ? 0 : 3))) & 0x7) + 1);
    unsigned qh = b[32 + 2 * ib + (l >> 1)];
    unsigned idx = b[4 * ib + l] | ((qh << ((l & 1) ? 4 : 8)) & 0x700);
    float delta = qh & ((l & 1) ? 0x80 : 0x08) ? -IQ1_DELTA : IQ1_DELTA;
    return dl * iq1_w(kiq1s_grid[idx], j, delta);
}
static __device__ __forceinline__ float tdq_mxfp4(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(32, 17);
    int e = i & 31;
    int q = e < 16 ? (b[1 + e] & 0xF) : (b[1 + e - 16] >> 4);
    return kv_mxfp4[q] * ldexpf(1.0f, (int)b[0] - 127);
}
static __device__ __forceinline__ float tdq_nvfp4(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(64, 36);
    int e = i & 63, s = e >> 4, r = e & 15;
    uchar byte = b[4 + 8 * s + (r & 7)];
    return kv_mxfp4[r < 8 ? (byte & 0xF) : (byte >> 4)] * ue4m3f(b[s]);
}
static __device__ __forceinline__ float tdq_q2_0(const uchar *rw, int i) {
    const uchar *b = TQ_BLK(64, 18);
    int e = i & 63;
    return (float)((int)((b[2 + (e >> 2)] >> ((e & 3) * 2)) & 3) - 1) * f16f(b);
}

// a row's bytes from its element count
#define TQ_ROW(BS, TS) ((ulong64)(a.n_in / (BS)) * (TS))

// The activation chunk a k_tmm block stages; TV_*, TMM_NC, TMM_NCW and
// TMM_WARPS are kernel_args.h's, read by the launches too.
#define TMM_KC 128

// k_tmvt: a block owns TV_IT inputs i and TV_TT positions t; per TV_JT rows j
// it decodes that weight tile into shared memory once and stages dy's tile,
// then thread (i, group g) runs its 8 positions' chains over the tile's rows
// in ascending j. Every element is still the CPU chain (start at dx, j
// ascending, a zero dy skipped), and each weight is decoded once per TV_TT
// positions instead of once per 8.
// k_tmm / k_tmmw: the same per-column arithmetic at two widths (TMM_NC and
// TMM_NCW columns a block), so a column's value does not depend on which one
// computed it: lane L sums i = L, L + 32, ... with fmaf, then warp_sum. A
// warp carries TMM_RW rows, so each staged activation it loads feeds TMM_RW
// FMAs (a row's chain is the same whichever rows share its warp). Measured
// on IQ3_S and Q4_K rows of 5120 x 17408 at 64 columns: 1.5 to 1.9 TFLOP/s
// at one row a warp and 64 columns a block, 2.9 to 3.1 at four rows and 16
// or 32 columns (one load of shared memory per FMA was the bound).
#define TRAIN_MM(fname, name, BS, TS, NC, RW)                                 \
extern "C" __global__ void fname(MV_PARAMS) {                                 \
    (void)bias;                                                                \
    __shared__ float xsm[NC][TMM_KC];                                          \
    int warp = (int)(threadIdx.x >> 5), lane = (int)(threadIdx.x & 31);        \
    int row0 = ((int)blockIdx.x * TMM_WARPS + warp) * (RW);                    \
    int t0 = (int)blockIdx.y * NC;                                             \
    int nc = a.batch - t0 < NC ? a.batch - t0 : NC;                            \
    const uchar *rw[RW];                                                       \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < (RW); r++)                                             \
        rw[r] = wb + a.w_off +                                                 \
                (ulong64)(row0 + r < a.n_out ? row0 + r : 0) * TQ_ROW(BS, TS); \
    float acc[RW][NC];                                                         \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < (RW); r++)                                             \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < NC; c++) acc[r][c] = 0.0f;                         \
    for (int k0 = 0; k0 < a.n_in; k0 += TMM_KC) {                              \
        int kn = a.n_in - k0 < TMM_KC ? a.n_in - k0 : TMM_KC;                  \
        __syncthreads();                                                       \
        for (int s = (int)threadIdx.x; s < NC * TMM_KC; s += (int)blockDim.x) { \
            int c = s / TMM_KC, k = s % TMM_KC;                                \
            xsm[c][k] = c < nc && k < kn ? x[(ulong64)(t0 + c) * a.xs + k0 + k] : 0.0f; \
        }                                                                      \
        __syncthreads();                                                       \
        if (row0 < a.n_out)                                                    \
            for (int k = lane; k < kn; k += 32) {                              \
                float w[RW];                                                   \
                _Pragma("unroll")                                              \
                for (int r = 0; r < (RW); r++) w[r] = tdq_##name(rw[r], k0 + k); \
                _Pragma("unroll")                                              \
                for (int c = 0; c < NC; c++) {                                 \
                    float xv = xsm[c][k];                                      \
                    _Pragma("unroll")                                          \
                    for (int r = 0; r < (RW); r++)                             \
                        acc[r][c] = fmaf(w[r], xv, acc[r][c]);                 \
                }                                                              \
            }                                                                  \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < (RW); r++)                                             \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < NC; c++) {                                         \
            float v = warp_sum(acc[r][c]);                                     \
            if (lane == 0 && row0 + r < a.n_out && c < nc)                     \
                y[(ulong64)(t0 + c) * a.ys + row0 + r] = v;                    \
        }                                                                      \
}

// k_tmms: k_tmm's arithmetic with the weight tile staged. A block's
// TMS_RG * TMS_RW rows are decoded into shared memory once per TMS_KC chunk,
// cooperatively, and read by TMS_CG warps a row group, each carrying TMS_NC
// columns, so a weight is decoded once per TMS_CG * TMS_NC columns instead of
// once per 16 or 32, and each staged activation feeds TMS_RW FMAs. Lane L
// still sums i = L, L + 32, ... with fmaf in ascending order and the warp
// folds the 32 partials with warp_sum: every output is k_tmm's value bit for
// bit (test-train-gpu compares the two), so the dispatch may pick either by
// width without moving a column's value.
#define TMS_KC 128
#define TRAIN_MMS(name, BS, TS)                                                \
extern "C" __global__ void k_tmms_##name(MV_PARAMS) {                         \
    (void)bias;                                                                \
    __shared__ float wsm[TMS_RG * TMS_RW][TMS_KC];                             \
    __shared__ float xsm[TMS_CG * TMS_NC][TMS_KC];                             \
    int warp = (int)(threadIdx.x >> 5), lane = (int)(threadIdx.x & 31);        \
    int rg = warp / TMS_CG, cg = warp % TMS_CG;                                \
    int brow = (int)blockIdx.x * (TMS_RG * TMS_RW), row0 = brow + rg * TMS_RW; \
    int tb = (int)blockIdx.y * (TMS_CG * TMS_NC), t0 = tb + cg * TMS_NC;       \
    int nbk = a.batch - tb < TMS_CG * TMS_NC ? a.batch - tb : TMS_CG * TMS_NC; \
    ulong64 rb = TQ_ROW(BS, TS);                                               \
    const uchar *w0 = wb + a.w_off;                                            \
    float acc[TMS_RW][TMS_NC];                                                 \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < TMS_RW; r++)                                           \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < TMS_NC; c++) acc[r][c] = 0.0f;                     \
    for (int k0 = 0; k0 < a.n_in; k0 += TMS_KC) {                              \
        int kn = a.n_in - k0 < TMS_KC ? a.n_in - k0 : TMS_KC;                  \
        __syncthreads();                                                       \
        for (int e = (int)threadIdx.x; e < TMS_RG * TMS_RW * TMS_KC;           \
             e += (int)blockDim.x) {                                           \
            int r = e / TMS_KC, k = e % TMS_KC;                                \
            wsm[r][k] = brow + r < a.n_out && k < kn                           \
                ? tdq_##name(w0 + (ulong64)(brow + r) * rb, k0 + k) : 0.0f;    \
        }                                                                      \
        for (int e = (int)threadIdx.x; e < TMS_CG * TMS_NC * TMS_KC;           \
             e += (int)blockDim.x) {                                           \
            int c = e / TMS_KC, k = e % TMS_KC;                                \
            xsm[c][k] = c < nbk && k < kn                                      \
                ? x[(ulong64)(tb + c) * a.xs + k0 + k] : 0.0f;                 \
        }                                                                      \
        __syncthreads();                                                       \
        for (int k = lane; k < kn; k += 32) {                                  \
            float w[TMS_RW];                                                   \
            _Pragma("unroll")                                                  \
            for (int r = 0; r < TMS_RW; r++) w[r] = wsm[rg * TMS_RW + r][k];   \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < TMS_NC; c++) {                                 \
                float xv = xsm[cg * TMS_NC + c][k];                            \
                _Pragma("unroll")                                              \
                for (int r = 0; r < TMS_RW; r++)                               \
                    acc[r][c] = fmaf(w[r], xv, acc[r][c]);                     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < TMS_RW; r++)                                           \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < TMS_NC; c++) {                                     \
            float v = warp_sum(acc[r][c]);                                     \
            if (lane == 0 && row0 + r < a.n_out && t0 + c < a.batch)           \
                y[(ulong64)(t0 + c) * a.ys + row0 + r] = v;                    \
        }                                                                      \
}

#define TRAIN_KERNELS(name, BS, TS)                                            \
extern "C" __global__ void k_tmvt_##name(MV_PARAMS) {                         \
    (void)bias;                                                                \
    __shared__ float wsm[TV_JT][TV_IT];                                        \
    __shared__ float dys[TV_TT][TV_JT];                                        \
    int ti = (int)(threadIdx.x % TV_IT), tg = (int)(threadIdx.x / TV_IT);      \
    int i0 = (int)blockIdx.x * TV_IT, tb0 = (int)blockIdx.y * TV_TT;           \
    int i = i0 + ti, t0 = tb0 + tg * 8;                                        \
    bool live = i < a.n_in;                                                    \
    ulong64 rb = TQ_ROW(BS, TS);                                               \
    const uchar *w0 = wb + a.w_off;                                            \
    float acc[8];                                                              \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < 8; c++)                                                \
        acc[c] = live && t0 + c < a.batch ? y[(ulong64)(t0 + c) * a.ys + i] : 0.0f; \
    for (int j0 = 0; j0 < a.n_out; j0 += TV_JT) {                              \
        int jn = a.n_out - j0 < TV_JT ? a.n_out - j0 : TV_JT;                  \
        __syncthreads();                                                       \
        for (int e = (int)threadIdx.x; e < TV_JT * TV_IT; e += (int)blockDim.x) { \
            int jj = e / TV_IT, ii = e % TV_IT;                                \
            wsm[jj][ii] = jj < jn && i0 + ii < a.n_in                          \
                ? tdq_##name(w0 + (ulong64)(j0 + jj) * rb, i0 + ii) : 0.0f;    \
        }                                                                      \
        for (int e = (int)threadIdx.x; e < TV_TT * TV_JT; e += (int)blockDim.x) { \
            int tt = e / TV_JT, jj = e % TV_JT;                                \
            dys[tt][jj] = jj < jn && tb0 + tt < a.batch                        \
                ? x[(ulong64)(tb0 + tt) * a.xs + j0 + jj] : 0.0f;              \
        }                                                                      \
        __syncthreads();                                                       \
        if (live)                                                              \
            for (int jj = 0; jj < jn; jj++) {                                  \
                float w = wsm[jj][ti];                                         \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 8; c++) {                                  \
                    float v = dys[tg * 8 + c][jj];                             \
                    if (v != 0.0f) acc[c] = fmaf(w, v, acc[c]);                \
                }                                                              \
            }                                                                  \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < 8; c++)                                                \
        if (live && t0 + c < a.batch) y[(ulong64)(t0 + c) * a.ys + i] = acc[c]; \
}                                                                              \
TRAIN_MM(k_tmm_##name, name, BS, TS, TMM_NC, TMM_RW)                           \
TRAIN_MM(k_tmmw_##name, name, BS, TS, TMM_NCW, TMM_RW)                         \
TRAIN_MMS(name, BS, TS)

TRAIN_KERNELS(f32, 1, 4)
TRAIN_KERNELS(f16, 1, 2)
TRAIN_KERNELS(bf16, 1, 2)
TRAIN_KERNELS(q8_0, 32, 34)
TRAIN_KERNELS(q4_0, 32, 18)
TRAIN_KERNELS(q4_1, 32, 20)
TRAIN_KERNELS(q5_0, 32, 22)
TRAIN_KERNELS(q5_1, 32, 24)
TRAIN_KERNELS(q2_K, 256, 84)
TRAIN_KERNELS(q3_K, 256, 110)
TRAIN_KERNELS(q4_K, 256, 144)
TRAIN_KERNELS(q5_K, 256, 176)
TRAIN_KERNELS(q6_K, 256, 210)
TRAIN_KERNELS(iq4_nl, 32, 18)
TRAIN_KERNELS(iq4_xs, 256, 136)
TRAIN_KERNELS(iq2_xxs, 256, 66)
TRAIN_KERNELS(iq2_xs, 256, 74)
TRAIN_KERNELS(iq2_s, 256, 82)
TRAIN_KERNELS(iq3_xxs, 256, 98)
TRAIN_KERNELS(iq3_s, 256, 110)
TRAIN_KERNELS(iq1_s, 256, 50)
TRAIN_KERNELS(iq1_m, 256, 56)
TRAIN_KERNELS(mxfp4, 32, 17)
TRAIN_KERNELS(nvfp4, 64, 36)
TRAIN_KERNELS(q2_0, 64, 18)
