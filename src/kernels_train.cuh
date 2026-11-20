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

// tdq8_<type>(row, i0, o): weights i0 .. i0 + 7 of a row (i0 a multiple of
// 8), each the value tdq_<type> gives it by the same expression, so the
// same bits. The register-blocked kernels decode a run at a time: IQ3_S,
// Q4_K and Q6_K share the run's block header, scale and grid words instead
// of reading them once a weight; the rest take their weights one by one.
// tdq_run_<type>: whether the kernels decode the type in runs (else one
// weight a thread, consecutive threads on consecutive weights of a row,
// which keeps a simple type's byte reads coalesced).
#define TDQ8_EACH(name)                                                        \
static const bool tdq_run_##name = false;                                      \
static __device__ __forceinline__ void tdq8_##name(const uchar *rw, int i0,   \
                                                   float *o) {                \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; j++) o[j] = tdq_##name(rw, i0 + j);                 \
}
TDQ8_EACH(f32)
TDQ8_EACH(f16)
TDQ8_EACH(bf16)
TDQ8_EACH(q8_0)
TDQ8_EACH(q4_0)
TDQ8_EACH(q4_1)
TDQ8_EACH(q5_0)
TDQ8_EACH(q5_1)
TDQ8_EACH(q2_K)
TDQ8_EACH(q3_K)
TDQ8_EACH(q5_K)
TDQ8_EACH(iq4_nl)
TDQ8_EACH(iq4_xs)
TDQ8_EACH(iq2_xxs)
TDQ8_EACH(iq2_xs)
TDQ8_EACH(iq2_s)
TDQ8_EACH(iq3_xxs)
TDQ8_EACH(iq1_s)
TDQ8_EACH(iq1_m)
TDQ8_EACH(mxfp4)
TDQ8_EACH(nvfp4)
TDQ8_EACH(q2_0)

static const bool tdq_run_q4_K = true;
static __device__ __forceinline__ void tdq8_q4_K(const uchar *rw, int i0, float *o) {
    int i = i0;
    const uchar *b = TQ_BLK(256, 144);
    int e = i0 & 255, g = e >> 5, l0 = e & 31;
    uchar sc, mn;
    get_scale_min_k4(g, b + 4, &sc, &mn);
    float d = f16f(b), dm = f16f(b + 2);
    const uchar *qs = b + 16 + 32 * (g >> 1) + l0;
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) {
        int q = (g & 1) ? (qs[j] >> 4) : (qs[j] & 0xF);
        o[j] = d * (float)sc * (float)q - dm * (float)mn;
    }
}
static const bool tdq_run_q6_K = true;
static __device__ __forceinline__ void tdq8_q6_K(const uchar *rw, int i0, float *o) {
    int i = i0;
    const uchar *b = TQ_BLK(256, 210);
    int e = i0 & 255, half = e >> 7, r = e & 127, seg = r >> 5, l0 = r & 31;
    const uchar *ql = b + half * 64, *qh = b + 128 + half * 32;
    int s = ((const signed char *)(b + 192))[half * 8 + 2 * seg + (l0 >> 4)];
    float d = f16f(b + 208);
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) {
        int l = l0 + j;
        int lo = (seg & 1) ? ql[l + 32] : ql[l];
        int q = (int)(((seg & 2) ? (lo >> 4) : (lo & 0xF)) | (((qh[l] >> (2 * seg)) & 3) << 4)) - 32;
        o[j] = d * (float)(s * q);
    }
}
static const bool tdq_run_iq3_s = true;
static __device__ __forceinline__ void tdq8_iq3_s(const uchar *rw, int i0, float *o) {
    int i = i0;
    const uchar *b = TQ_BLK(256, 110);
    int e = i0 & 255, ib = e >> 5, l = (e >> 3) & 3;
    unsigned sc = b[106 + (ib >> 1)];
    float db = f16f(b) * (float)(1 + 2 * ((ib & 1) ? (sc >> 4) : (sc & 0xF)));
    unsigned hb = b[66 + ib];
    unsigned g0 = kiq3s_grid[b[2 + 8 * ib + 2 * l] | ((hb << (8 - 2 * l)) & 256)];
    unsigned g1 = kiq3s_grid[b[2 + 8 * ib + 2 * l + 1] | ((hb << (7 - 2 * l)) & 256)];
    unsigned sg = b[74 + 4 * ib + l];
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) {
        float mag = (float)(((j < 4 ? g0 : g1) >> (8 * (j & 3))) & 0xFF);
        o[j] = db * ((sg >> j) & 1 ? -mag : mag);
    }
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

// k_tmmr / k_tmmrn (R8.20 attempt 3): the same product, register-blocked.
// Output (j, t) is the sum of TMR_KS chains: chain g takes the input chunks
// c with c % TMR_KS == g (TMR_BK inputs a chunk), each chunk's inputs in
// order, one fmaf a term from zero; the chains fold as ((c0 + c1) + c2) + c3.
// That contract names neither the columns sharing a launch nor the block
// shape, so a column's value is the same in either twin and at any batch:
// batch-invariant, and deterministic. A block holds TMR_BM rows and BN
// columns; group g of its 64 threads runs chain g, decoding its chunk of the
// tile's weights into shared memory once for all BN columns, and each thread
// keeps a TM x TN tile of the outputs in registers, so one staged weight and
// one staged activation feed TM x TN multiply-adds between them.
#define TRAIN_MMR(fname, name, BS, TS, BN, TM, TN)                            \
extern "C" __global__ void __launch_bounds__(TMR_KS * 64) fname(MV_PARAMS) {  \
    (void)bias;                                                                \
    __shared__ __align__(16) float ws[TMR_KS][TMR_BK][TMR_BM + 4];             \
    __shared__ __align__(16) float xs[TMR_KS][TMR_BK][(BN) + 4];               \
    const int TX = (BN) / (TN);                                                \
    int g = (int)(threadIdx.x >> 6), lt = (int)(threadIdx.x & 63);             \
    int tx = lt % TX, ty = lt / TX;                                            \
    int r0 = (int)blockIdx.x * TMR_BM, c0 = (int)blockIdx.y * (BN);            \
    ulong64 rb = TQ_ROW(BS, TS);                                               \
    const uchar *w0 = wb + a.w_off;                                            \
    float acc[TM][TN];                                                         \
    _Pragma("unroll")                                                          \
    for (int m = 0; m < (TM); m++)                                             \
        _Pragma("unroll")                                                      \
        for (int n = 0; n < (TN); n++) acc[m][n] = 0.0f;                       \
    int nch = (a.n_in + TMR_BK - 1) / TMR_BK;                                  \
    for (int it = 0; it * TMR_KS < nch; it++) {                                \
        int ch = it * TMR_KS + g, k0 = ch * TMR_BK;                            \
        int kn = ch < nch ? (a.n_in - k0 < TMR_BK ? a.n_in - k0 : TMR_BK) : 0; \
        __syncthreads();                                                       \
        if (!tdq_run_##name)                                                   \
            for (int e = lt; e < TMR_BM * TMR_BK; e += 64) {                   \
                int kk = e % TMR_BK, r = e / TMR_BK;                           \
                ws[g][kk][r] = kk < kn && r0 + r < a.n_out                     \
                    ? tdq_##name(w0 + (ulong64)(r0 + r) * rb, k0 + kk) : 0.0f; \
            }                                                                  \
        /* runs of 8 inputs of one row a thread (TMR_BK / 8 runs a row) */    \
        else for (int e = lt; e < TMR_BM * (TMR_BK / 8); e += 64) {            \
            int r = e / (TMR_BK / 8), kk0 = (e % (TMR_BK / 8)) * 8;            \
            float wv[8];                                                       \
            if (r0 + r < a.n_out && kk0 + 8 <= kn) {                           \
                tdq8_##name(w0 + (ulong64)(r0 + r) * rb, k0 + kk0, wv);         \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 8; j++)                                    \
                    wv[j] = r0 + r < a.n_out && kk0 + j < kn                   \
                        ? tdq_##name(w0 + (ulong64)(r0 + r) * rb, k0 + kk0 + j) : 0.0f; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 8; j++) ws[g][kk0 + j][r] = wv[j];             \
        }                                                                      \
        for (int e = lt; e < (BN) * TMR_BK; e += 64) {                         \
            int kk = e % TMR_BK, c = e / TMR_BK;                               \
            xs[g][kk][c] = kk < kn && c0 + c < a.batch                         \
                ? x[(ulong64)(c0 + c) * a.xs + k0 + kk] : 0.0f;                \
        }                                                                      \
        __syncthreads();                                                       \
        for (int kk = 0; kk < kn; kk++) {                                      \
            float av[TM], bv[TN];                                              \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m += 4) {                                \
                float4 v = *(const float4 *)&ws[g][kk][ty * (TM) + m];         \
                av[m] = v.x; av[m + 1] = v.y; av[m + 2] = v.z; av[m + 3] = v.w; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int n = 0; n < (TN); n += 4) {                                \
                float4 v = *(const float4 *)&xs[g][kk][tx * (TN) + n];         \
                bv[n] = v.x; bv[n + 1] = v.y; bv[n + 2] = v.z; bv[n + 3] = v.w; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m++)                                     \
                _Pragma("unroll")                                              \
                for (int n = 0; n < (TN); n++)                                 \
                    acc[m][n] = fmaf(av[m], bv[n], acc[m][n]);                 \
        }                                                                      \
    }                                                                          \
    /* the chains fold in order through shared memory (the staging, reused) */ \
    float *red = &ws[0][0][0];                                                 \
    for (int s = 1; s < TMR_KS; s++) {                                         \
        __syncthreads();                                                       \
        if (g == s) {                                                          \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m++)                                     \
                _Pragma("unroll")                                              \
                for (int n = 0; n < (TN); n++)                                 \
                    red[(ty * (TM) + m) * (BN) + tx * (TN) + n] = acc[m][n];   \
        }                                                                      \
        __syncthreads();                                                       \
        if (g == 0) {                                                          \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m++)                                     \
                _Pragma("unroll")                                              \
                for (int n = 0; n < (TN); n++)                                 \
                    acc[m][n] += red[(ty * (TM) + m) * (BN) + tx * (TN) + n];  \
        }                                                                      \
    }                                                                          \
    if (g == 0) {                                                              \
        _Pragma("unroll")                                                      \
        for (int m = 0; m < (TM); m++) {                                       \
            int r = r0 + ty * (TM) + m;                                        \
            _Pragma("unroll")                                                  \
            for (int n = 0; n < (TN); n++) {                                   \
                int c = c0 + tx * (TN) + n;                                    \
                if (r < a.n_out && c < a.batch) y[(ulong64)c * a.ys + r] = acc[m][n]; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// k_tmvtr / k_tmvtrn: the transposed product for --train-gpu, dx[t][i] +=
// sum_j W[j][i] dy[t][j], on k_tmmr's contract: the sum over j is four
// chains of TMR_BK-row chunks dealt round-robin, each a serial fmaf from
// zero, folded in order, then added to dx's incoming value. Not the CPU
// chain, which k_tmvt keeps for RUNNER_TRAIN_GPU's assist (whose adapters
// are the CPU's bytes); batch-invariant and deterministic, the same bits
// from either twin. A block holds TMR_BM inputs and BT positions; a chunk's
// rows are decoded in runs of 8 consecutive inputs.
#define TRAIN_MVTR(fname, name, BS, TS, BT, TM, TN)                           \
extern "C" __global__ void __launch_bounds__(TMR_KS * 64) fname(MV_PARAMS) {  \
    (void)bias;                                                                \
    __shared__ __align__(16) float ws[TMR_KS][TMR_BK][TMR_BM + 4];             \
    __shared__ __align__(16) float ds[TMR_KS][TMR_BK][(BT) + 4];               \
    const int TX = (BT) / (TN);                                                \
    int g = (int)(threadIdx.x >> 6), lt = (int)(threadIdx.x & 63);             \
    int tx = lt % TX, ty = lt / TX;                                            \
    int i0 = (int)blockIdx.x * TMR_BM, t0 = (int)blockIdx.y * (BT);            \
    ulong64 rb = TQ_ROW(BS, TS);                                               \
    const uchar *w0 = wb + a.w_off;                                            \
    float acc[TM][TN];                                                         \
    _Pragma("unroll")                                                          \
    for (int m = 0; m < (TM); m++)                                             \
        _Pragma("unroll")                                                      \
        for (int n = 0; n < (TN); n++) acc[m][n] = 0.0f;                       \
    int nch = (a.n_out + TMR_BK - 1) / TMR_BK;                                 \
    for (int it = 0; it * TMR_KS < nch; it++) {                                \
        int ch = it * TMR_KS + g, j0 = ch * TMR_BK;                            \
        int kn = ch < nch ? (a.n_out - j0 < TMR_BK ? a.n_out - j0 : TMR_BK) : 0; \
        __syncthreads();                                                       \
        for (int e = lt; e < TMR_BK * (TMR_BM / 8); e += 64) {                 \
            int kk = e / (TMR_BM / 8), i8 = (e % (TMR_BM / 8)) * 8;            \
            float wv[8];                                                       \
            if (kk < kn && i0 + i8 + 8 <= a.n_in) {                            \
                tdq8_##name(w0 + (ulong64)(j0 + kk) * rb, i0 + i8, wv);         \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int j = 0; j < 8; j++)                                    \
                    wv[j] = kk < kn && i0 + i8 + j < a.n_in                    \
                        ? tdq_##name(w0 + (ulong64)(j0 + kk) * rb, i0 + i8 + j) : 0.0f; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int j = 0; j < 8; j++) ws[g][kk][i8 + j] = wv[j];             \
        }                                                                      \
        for (int e = lt; e < (BT) * TMR_BK; e += 64) {                         \
            int kk = e % TMR_BK, c = e / TMR_BK;                               \
            ds[g][kk][c] = kk < kn && t0 + c < a.batch                         \
                ? x[(ulong64)(t0 + c) * a.xs + j0 + kk] : 0.0f;                \
        }                                                                      \
        __syncthreads();                                                       \
        for (int kk = 0; kk < kn; kk++) {                                      \
            float av[TM], bv[TN];                                              \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m += 4) {                                \
                float4 v = *(const float4 *)&ws[g][kk][ty * (TM) + m];         \
                av[m] = v.x; av[m + 1] = v.y; av[m + 2] = v.z; av[m + 3] = v.w; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int n = 0; n < (TN); n += 4) {                                \
                float4 v = *(const float4 *)&ds[g][kk][tx * (TN) + n];         \
                bv[n] = v.x; bv[n + 1] = v.y; bv[n + 2] = v.z; bv[n + 3] = v.w; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m++)                                     \
                _Pragma("unroll")                                              \
                for (int n = 0; n < (TN); n++)                                 \
                    acc[m][n] = fmaf(av[m], bv[n], acc[m][n]);                 \
        }                                                                      \
    }                                                                          \
    float *red = &ws[0][0][0];                                                 \
    for (int s = 1; s < TMR_KS; s++) {                                         \
        __syncthreads();                                                       \
        if (g == s) {                                                          \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m++)                                     \
                _Pragma("unroll")                                              \
                for (int n = 0; n < (TN); n++)                                 \
                    red[(ty * (TM) + m) * (BT) + tx * (TN) + n] = acc[m][n];   \
        }                                                                      \
        __syncthreads();                                                       \
        if (g == 0) {                                                          \
            _Pragma("unroll")                                                  \
            for (int m = 0; m < (TM); m++)                                     \
                _Pragma("unroll")                                              \
                for (int n = 0; n < (TN); n++)                                 \
                    acc[m][n] += red[(ty * (TM) + m) * (BT) + tx * (TN) + n];  \
        }                                                                      \
    }                                                                          \
    if (g == 0) {                                                              \
        _Pragma("unroll")                                                      \
        for (int m = 0; m < (TM); m++) {                                       \
            int i = i0 + ty * (TM) + m;                                        \
            _Pragma("unroll")                                                  \
            for (int n = 0; n < (TN); n++) {                                   \
                int c = t0 + tx * (TN) + n;                                    \
                if (i < a.n_in && c < a.batch) y[(ulong64)c * a.ys + i] += acc[m][n]; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
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
TRAIN_MMR(k_tmmr_##name, name, BS, TS, TMR_BN, 8, 8)                           \
TRAIN_MMR(k_tmmrn_##name, name, BS, TS, TMR_BNN, 4, 4)                         \
TRAIN_MVTR(k_tmvtr_##name, name, BS, TS, TMR_BN, 8, 8)                         \
TRAIN_MVTR(k_tmvtrn_##name, name, BS, TS, TMR_BNN, 4, 4)

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

// ---- R8.21: the Gated DeltaNet recurrence of a training step ----
//
// model.c dn_step's step, per head h and position t, on the state S [hv][hv]
// (row j a value dim, column i a key dim):
//   S <- decay S;  pred_j = S_j . k;  delta_j = beta (v_j - pred_j);
//   S_j <- S_j + delta_j k;  y_j = S_j . q / sqrt(sk)
// The rows do not touch each other, so a warp owns TDN_ROWS rows and lane L
// holds columns L, L + 32, ... of each (a lane past the row's end holds
// zeros): a row's dot product is the lane's fmaf chain over its columns,
// then one butterfly (tdn_allsum) that leaves every lane the same sum. tdn_rows is the one step: the taped forward's
// rounds and the backward's recompute (k_tdn_fw) and the reverse sweep's
// rebuild (k_tdn_bw) all take it, so they see the same states, bit for bit.
// Not the CPU's bits (another order); deterministic, and no value depends
// on the launch shape or on how the window was split into calls.

#define TDN_CPL (TDN_MAX_HV / 32)

static __device__ __forceinline__ float tdn_allsum(float s) {
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);
    return s;
}

// a position's q and k columns for lane L (cpl of them, zero past hv)
static __device__ __forceinline__ void tdn_qk(const float *ct, const tdn_args &a,
                                              int g, int lane, int cpl,
                                              float (&kc)[TDN_CPL],
                                              float (&qc)[TDN_CPL]) {
    #pragma unroll
    for (int c = 0; c < TDN_CPL; c++) {
        bool on = c < cpl && lane + 32 * c < a.hv;
        qc[c] = on ? ct[g * a.sk + lane + 32 * c] : 0.0f;
        kc[c] = on ? ct[a.keydim + g * a.sk + lane + 32 * c] : 0.0f;
    }
}

// one position's step on the warp's rows j0 .. j0 + TDN_ROWS - 1; y (or
// NULL) gets their readouts, row r's from lane r
static __device__ __forceinline__ void tdn_rows(float (&s)[TDN_ROWS][TDN_CPL],
                                                int cpl, int j0, int lane,
                                                float dec, float bet,
                                                const float (&kc)[TDN_CPL],
                                                const float (&qc)[TDN_CPL],
                                                const float *v, float *y,
                                                float sqk) {
    #pragma unroll
    for (int r = 0; r < TDN_ROWS; r++) {
        float p = 0.0f;
        #pragma unroll
        for (int c = 0; c < TDN_CPL; c++)
            if (c < cpl) {
                s[r][c] *= dec;
                p = fmaf(s[r][c], kc[c], p);
            }
        float delta = (v[j0 + r] - tdn_allsum(p)) * bet;
        float acc = 0.0f;
        #pragma unroll
        for (int c = 0; c < TDN_CPL; c++)
            if (c < cpl) {
                s[r][c] = fmaf(delta, kc[c], s[r][c]);
                acc = fmaf(s[r][c], qc[c], acc);
            }
        acc = tdn_allsum(acc);
        if (y && lane == r) y[j0 + r] = acc / sqk;
    }
}

// The rows of a head's [hv][hv] matrix the warp owns, in or out of s.
static __device__ __forceinline__ void tdn_load(float (&s)[TDN_ROWS][TDN_CPL],
                                                const float *m, int hv, int j0,
                                                int lane, int cpl) {
    #pragma unroll
    for (int r = 0; r < TDN_ROWS; r++)
        #pragma unroll
        for (int c = 0; c < TDN_CPL; c++)
            s[r][c] = c < cpl && lane + 32 * c < hv
                ? m[(ulong64)(j0 + r) * hv + lane + 32 * c] : 0.0f;
}

static __device__ __forceinline__ void tdn_store(const float (&s)[TDN_ROWS][TDN_CPL],
                                                 float *m, int hv, int j0,
                                                 int lane, int cpl) {
    #pragma unroll
    for (int r = 0; r < TDN_ROWS; r++)
        #pragma unroll
        for (int c = 0; c < TDN_CPL; c++)
            if (c < cpl && lane + 32 * c < hv)
                m[(ulong64)(j0 + r) * hv + lane + 32 * c] = s[r][c];
}

// The forward over a.T positions: TDN_FW_WARPS warps a block, grid (heads,
// hv / (TDN_ROWS * TDN_FW_WARPS)). state [nh][hv][hv] is read at the start
// when load is set (else the state starts at zero) and written at the end;
// ckpt (or NULL) gets each head's state before every position t with
// t % C == 0, at [t / C][nh][hv][hv]. cv rows hold q and k L2-normed;
// beta and decay are [T][nh], y is [T][inner].
extern "C" __global__ void __launch_bounds__(TDN_FW_WARPS * 32)
k_tdn_fw(const float *cv, const float *beta, const float *decay, float *state,
         float *y, float *ckpt, tdn_args a, int load) {
    int lane = (int)(threadIdx.x & 31), h = (int)blockIdx.x;
    int j0 = ((int)blockIdx.y * TDN_FW_WARPS + (int)(threadIdx.x >> 5)) * TDN_ROWS;
    if (h >= a.nh || j0 >= a.hv) return;
    int cpl = (a.hv + 31) / 32, g = h % a.ng;
    ulong64 hh = (ulong64)a.hv * a.hv;
    float *st = state + (ulong64)h * hh;
    float s[TDN_ROWS][TDN_CPL], kc[TDN_CPL], qc[TDN_CPL];
    if (load) {
        tdn_load(s, st, a.hv, j0, lane, cpl);
    } else {
        #pragma unroll
        for (int r = 0; r < TDN_ROWS; r++)
            #pragma unroll
            for (int c = 0; c < TDN_CPL; c++) s[r][c] = 0.0f;
    }
    float sqk = sqrtf((float)a.sk);
    for (int t = 0; t < a.T; t++) {
        if (ckpt && t % a.C == 0)
            tdn_store(s, ckpt + ((ulong64)(t / a.C) * a.nh + h) * hh, a.hv, j0,
                      lane, cpl);
        const float *ct = cv + (ulong64)t * a.convdim;
        tdn_qk(ct, a, g, lane, cpl, kc, qc);
        tdn_rows(s, cpl, j0, lane, decay[(ulong64)t * a.nh + h],
                 beta[(ulong64)t * a.nh + h], kc, qc,
                 ct + 2 * a.keydim + h * a.hv,
                 y + (ulong64)t * a.inner + h * a.hv, sqk);
    }
    tdn_store(s, st, a.hv, j0, lane, cpl);
}

// The reverse sweep (model.c dn_bw_worker's arithmetic), one block a head of
// hv / TDN_ROWS warps, warp w owning rows w * TDN_ROWS ... Per segment, last
// first: its states rebuilt from the checkpoint by tdn_rows into sb (slot s
// the state before position t0 + s; slot 0 is the checkpoint itself), then
// t from the segment's end down, with dS the gradient on the state after t,
// carried across segments in the warps' registers:
//   y:      dS_j += g_j q (g = dy / sqrt(sk)),  dq += S_t^T g
//   update: dd_j = dS_j . k,  dk += dS^T delta
//   delta:  dv_j = beta dd_j,  dbeta = sum_j dd_j (v_j - pred_j),
//           dSd_j = dS_j - beta dd_j k,  dk -= (decay S_{t-1})^T beta dd
//   decay:  ddecay = <dSd, S_{t-1}>,  dS <- decay dSd
// S_t's rows are tdn_rows' arithmetic again on S_{t-1}, so the same bits the
// rebuild stored. A lane reads back only the elements it stored. The column
// sums (dq, dk) and the two scalars fold the warps' partials in warp order
// through shared memory, double-buffered by t's parity: one barrier a
// position, and fixed trees throughout. dq, dk are [nh][T][sk], dv [T][inner],
// dbeta and ddecay [T][nh]; sb holds C states a head.
extern "C" __global__ void __launch_bounds__(TDN_MAX_HV / TDN_ROWS * 32)
k_tdn_bw(const float *cv, const float *beta, const float *decay,
         const float *ckpt, float *sb, const float *dy, float *dq, float *dk,
         float *dv, float *dbeta, float *ddecay, tdn_args a) {
    __shared__ float pq[2][TDN_MAX_HV / TDN_ROWS][TDN_MAX_HV];
    __shared__ float pk[2][TDN_MAX_HV / TDN_ROWS][TDN_MAX_HV];
    __shared__ float pb[2][TDN_MAX_HV / TDN_ROWS], pd[2][TDN_MAX_HV / TDN_ROWS];
    int lane = (int)(threadIdx.x & 31), w = (int)(threadIdx.x >> 5);
    int h = (int)blockIdx.x, j0 = w * TDN_ROWS;
    int nw = a.hv / TDN_ROWS, cpl = (a.hv + 31) / 32, g = h % a.ng;
    ulong64 hh = (ulong64)a.hv * a.hv;
    float *sbh = sb + (ulong64)h * a.C * hh;
    float sqk = sqrtf((float)a.sk), rs = 1.0f / sqk;
    float dS[TDN_ROWS][TDN_CPL], kc[TDN_CPL], qc[TDN_CPL];
    #pragma unroll
    for (int r = 0; r < TDN_ROWS; r++)
        #pragma unroll
        for (int c = 0; c < TDN_CPL; c++) dS[r][c] = 0.0f;
    int nseg = (a.T + a.C - 1) / a.C;
    for (int seg = nseg - 1; seg >= 0; seg--) {
        int t0 = seg * a.C, t1 = t0 + a.C < a.T ? t0 + a.C : a.T;
        const float *ck = ckpt + ((ulong64)seg * a.nh + h) * hh;
        {
            float s[TDN_ROWS][TDN_CPL];
            tdn_load(s, ck, a.hv, j0, lane, cpl);
            for (int t = t0; t < t1 - 1; t++) {
                const float *ct = cv + (ulong64)t * a.convdim;
                tdn_qk(ct, a, g, lane, cpl, kc, qc);
                tdn_rows(s, cpl, j0, lane, decay[(ulong64)t * a.nh + h],
                         beta[(ulong64)t * a.nh + h], kc, qc,
                         ct + 2 * a.keydim + h * a.hv, NULL, sqk);
                tdn_store(s, sbh + (ulong64)(t - t0 + 1) * hh, a.hv, j0, lane, cpl);
            }
        }
        for (int t = t1 - 1; t >= t0; t--) {
            const float *sp = t == t0 ? ck : sbh + (ulong64)(t - t0) * hh;
            const float *ct = cv + (ulong64)t * a.convdim;
            const float *v = ct + 2 * a.keydim + h * a.hv;
            const float *dyt = dy + (ulong64)t * a.inner + h * a.hv;
            float dec = decay[(ulong64)t * a.nh + h], bet = beta[(ulong64)t * a.nh + h];
            tdn_qk(ct, a, g, lane, cpl, kc, qc);
            float dqp[TDN_CPL], dkp[TDN_CPL], ddp = 0.0f, dbp = 0.0f;
            #pragma unroll
            for (int c = 0; c < TDN_CPL; c++) dqp[c] = dkp[c] = 0.0f;
            #pragma unroll
            for (int r = 0; r < TDN_ROWS; r++) {
                int j = j0 + r;
                float spv[TDN_CPL], sd[TDN_CPL], p = 0.0f;
                #pragma unroll
                for (int c = 0; c < TDN_CPL; c++) {
                    spv[c] = c < cpl && lane + 32 * c < a.hv
                        ? sp[(ulong64)j * a.hv + lane + 32 * c] : 0.0f;
                    sd[c] = spv[c];
                    if (c < cpl) {
                        sd[c] *= dec;
                        p = fmaf(sd[c], kc[c], p);
                    }
                }
                float pred = tdn_allsum(p);
                float delta = (v[j] - pred) * bet;
                float gj = dyt[j] * rs, e = 0.0f;
                #pragma unroll
                for (int c = 0; c < TDN_CPL; c++)
                    if (c < cpl) {
                        float stc = fmaf(delta, kc[c], sd[c]);
                        dS[r][c] = fmaf(gj, qc[c], dS[r][c]);
                        dqp[c] = fmaf(gj, stc, dqp[c]);
                        e = fmaf(dS[r][c], kc[c], e);
                    }
                float dd = tdn_allsum(e);
                if (lane == r) dv[(ulong64)t * a.inner + h * a.hv + j] = bet * dd;
                dbp += dd * (v[j] - pred);
                float dpred = -bet * dd;
                #pragma unroll
                for (int c = 0; c < TDN_CPL; c++)
                    if (c < cpl) {
                        dkp[c] = fmaf(dS[r][c], delta, dkp[c]);
                        float dsd = fmaf(dpred, kc[c], dS[r][c]);
                        dkp[c] = fmaf(sd[c], dpred, dkp[c]);
                        ddp = fmaf(dsd, spv[c], ddp);
                        dS[r][c] = dec * dsd;
                    }
            }
            int par = t & 1;
            #pragma unroll
            for (int c = 0; c < TDN_CPL; c++)
                if (c < cpl && lane + 32 * c < a.hv) {
                    pq[par][w][lane + 32 * c] = dqp[c];
                    pk[par][w][lane + 32 * c] = dkp[c];
                }
            ddp = tdn_allsum(ddp);
            if (lane == 0) {
                pb[par][w] = dbp;
                pd[par][w] = ddp;
            }
            __syncthreads();
            int i = (int)threadIdx.x;
            if (i < a.hv) {
                float sq = 0.0f, sk2 = 0.0f;
                for (int ww = 0; ww < nw; ww++) {
                    sq += pq[par][ww][i];
                    sk2 += pk[par][ww][i];
                }
                ulong64 o = ((ulong64)h * a.T + t) * a.sk + i;
                dq[o] = sq;
                dk[o] = sk2;
            }
            if (i == 0) {
                float sb2 = 0.0f, sd2 = 0.0f;
                for (int ww = 0; ww < nw; ww++) {
                    sb2 += pb[par][ww];
                    sd2 += pd[par][ww];
                }
                dbeta[(ulong64)t * a.nh + h] = sb2;
                ddecay[(ulong64)t * a.nh + h] = sd2;
            }
        }
    }
}

// ---- the attention backward of a training step (--train-gpu) ----
//
// model.c attn_bw_p1 / attn_bw_p2's arithmetic over the window in blocks of
// TAB_TB positions t. k_tab_p, one block a (t, head): the scores against the
// cached keys s in the window, their softmax, dp = dao . v_s, sum_pd and the
// score gradient ds = p (dp - sum_pd) scale, into the block's P and DS rows;
// the max and the sums fold in one fixed tree a block. k_tab_dq, one block a
// (t, head): dq += sum_s ds k_s, a serial chain over s. k_tab_dkv, one block
// a (s, kv head): dk += ds q_t and dv += p dao_t over the block's t
// ascending and the group's heads ascending, the CPU's own order, blocks in
// ascending t. Deterministic; only p differs from the CPU (another order of
// the same sums), so the result is within rounding of it.
static __device__ __forceinline__ int tab_t0(const tab_args &a, int t) {
    return a.swa > 0 && t - a.swa + 1 > 0 ? t - a.swa + 1 : 0;
}

static __device__ __forceinline__ float tab_blockred(float v, float *red, bool max) {
    int tid = (int)threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int off = TAB_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] = max ? fmaxf(red[tid], red[tid + off]) : red[tid] + red[tid + off];
        __syncthreads();
    }
    float r = red[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void __launch_bounds__(TAB_THREADS)
k_tab_p(const uchar *kc, const uchar *vc, const float *q, const float *dao,
        float *P, float *DS, tab_args a, int tb0) {
    __shared__ float qs[TAB_MAX_HD], das[TAB_MAX_HD], red[TAB_THREADS];
    int tt = (int)blockIdx.x, h = (int)blockIdx.y, t = tb0 + tt, tid = (int)threadIdx.x;
    if (t >= a.T) return;
    int kvh = h / a.kv_mul, hd = a.hd;
    for (int i = tid; i < hd; i += TAB_THREADS) {
        qs[i] = q[(ulong64)t * a.q_dim + h * hd + i];
        das[i] = dao[(ulong64)t * a.q_dim + h * hd + i];
    }
    __syncthreads();
    float *p = P + ((ulong64)tt * a.n_head + h) * a.T;
    float *ds = DS + ((ulong64)tt * a.n_head + h) * a.T;
    int t0 = tab_t0(a, t);
    float mx = -__int_as_float(0x7f800000);   // -inf
    for (int s = t0 + tid; s <= t; s += TAB_THREADS) {
        const uchar *kh = kc + (ulong64)s * a.k_row_b + (ulong64)kvh * hd * 2;
        float sc = 0.0f;
        for (int i = 0; i < hd; i++) sc = fmaf(qs[i], f16f(kh + 2 * i), sc);
        p[s] = sc * a.scale;
        mx = fmaxf(mx, p[s]);
    }
    mx = tab_blockred(mx, red, true);
    float sum = 0.0f;
    for (int s = t0 + tid; s <= t; s += TAB_THREADS) {
        p[s] = expf(p[s] - mx);
        sum += p[s];
    }
    sum = tab_blockred(sum, red, false);
    float spd = 0.0f;
    for (int s = t0 + tid; s <= t; s += TAB_THREADS) {
        p[s] /= sum;
        const uchar *vh = vc + (ulong64)s * a.v_row_b + (ulong64)kvh * hd * 2;
        float dp = 0.0f;
        for (int i = 0; i < hd; i++) dp = fmaf(das[i], f16f(vh + 2 * i), dp);
        ds[s] = dp;
        spd = fmaf(p[s], dp, spd);
    }
    spd = tab_blockred(spd, red, false);
    for (int s = t0 + tid; s <= t; s += TAB_THREADS)
        ds[s] = p[s] * (ds[s] - spd) * a.scale;
}

extern "C" __global__ void __launch_bounds__(TAB_THREADS)
k_tab_dq(const uchar *kc, const float *DS, float *dq, tab_args a, int tb0) {
    int tt = (int)blockIdx.x, h = (int)blockIdx.y, t = tb0 + tt;
    if (t >= a.T) return;
    int kvh = h / a.kv_mul, hd = a.hd, t0 = tab_t0(a, t);
    const float *ds = DS + ((ulong64)tt * a.n_head + h) * a.T;
    for (int i = (int)threadIdx.x; i < hd; i += TAB_THREADS) {
        float *o = dq + (ulong64)t * a.q_dim + h * hd + i;
        float acc = *o;
        for (int s = t0; s <= t; s++)
            acc = fmaf(ds[s], f16f(kc + (ulong64)s * a.k_row_b + ((ulong64)kvh * hd + i) * 2), acc);
        *o = acc;
    }
}

extern "C" __global__ void __launch_bounds__(TAB_THREADS)
k_tab_dkv(const float *q, const float *dao, const float *P, const float *DS,
          float *dk, float *dv, tab_args a, int tb0, int tbn) {
    int s = (int)blockIdx.x, kvh = (int)blockIdx.y, hd = a.hd;
    for (int i = (int)threadIdx.x; i < hd; i += TAB_THREADS) {
        float *ko = dk + (ulong64)s * a.kv_dim + kvh * hd + i;
        float *vo = dv + (ulong64)s * a.kv_dim + kvh * hd + i;
        float dka = *ko, dva = *vo;
        for (int tt = 0; tt < tbn; tt++) {
            int t = tb0 + tt;
            if (t < s || s < tab_t0(a, t)) continue;
            for (int h = kvh * a.kv_mul; h < (kvh + 1) * a.kv_mul; h++) {
                ulong64 r = ((ulong64)tt * a.n_head + h) * a.T + s;
                dka = fmaf(DS[r], q[(ulong64)t * a.q_dim + h * hd + i], dka);
                dva = fmaf(P[r], dao[(ulong64)t * a.q_dim + h * hd + i], dva);
            }
        }
        *ko = dka;
        *vo = dva;
    }
}

// ---- R8.24.1 B1: a gated FFN's activation and its adjoint, host-chain twins
// (model.c gated_act's SiLU, silu_d, act_f; kernel_args.h tffn_args). The exp
// is src/pmath.h's pm_exp_d / p_expf operation for operation, in double with
// every multiply and add rounded on its own (no contraction), and the float
// expressions keep the host's order: so a T3 build's host (strict float,
// pmath) computes these very bits, and test-train-gpu holds them to it with
// memcmp. Elementwise, so batch-invariant.
static __device__ __forceinline__ double tw_ldexp2(double x, int k) {
    while (k > 1000) { x = __dmul_rn(x, 0x1p1000); k -= 1000; }
    while (k < -1000) { x = __dmul_rn(x, 0x1p-1000); k += 1000; }
    return __dmul_rn(x, __longlong_as_double((long long)(k + 1023) << 52));
}
static __device__ __forceinline__ double tw_exp_d(double x) {
    if (x != x) return x;
    if (x > 709.0) return __longlong_as_double(0x7ff0000000000000LL);
    if (x < -745.0) return 0.0;
    const double ln2_hi = 6.93147180369123816490e-01;
    const double ln2_lo = 1.90821492927058770002e-10;
    const double inv_ln2 = 1.44269504088896338700e+00;
    double kd = __dmul_rn(x, inv_ln2);
    int k = __double2int_rz(kd >= 0 ? __dadd_rn(kd, 0.5) : __dsub_rn(kd, 0.5));
    double r = __dsub_rn(__dsub_rn(x, __dmul_rn((double)k, ln2_hi)),
                         __dmul_rn((double)k, ln2_lo));
    double p = 1.0 / 6227020800.0;
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 479001600.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 39916800.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 3628800.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 362880.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 40320.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 5040.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 720.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 120.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 24.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 6.0);
    p = __dadd_rn(__dmul_rn(p, r), 0.5);
    p = __dadd_rn(__dmul_rn(p, r), 1.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0);
    return tw_ldexp2(p, k);
}
static __device__ __forceinline__ float tw_expf(float x) {
    return __double2float_rn(tw_exp_d((double)x));
}
// silu_f: 0 below -80, else g / (1 + exp(-g))
static __device__ __forceinline__ float tw_silu(float g) {
    return g < -80.0f ? 0.0f : __fdiv_rn(g, __fadd_rn(1.0f, tw_expf(-g)));
}
// silu_d: 0 below -80, else sg * (1 + g * (1 - sg)), sg = 1 / (1 + exp(-g))
static __device__ __forceinline__ float tw_silu_d(float g) {
    if (g < -80.0f) return 0.0f;
    float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, tw_expf(-g)));
    return __fmul_rn(sg, __fadd_rn(1.0f, __fmul_rn(g, __fsub_rn(1.0f, sg))));
}

// h = gated_act(ACT_SILU, g, u): 0 below -80, else (g / (1 + exp(-g))) * u
extern "C" __global__ void k_tffn_fw(const float *g, const float *u, float *h,
                                     tffn_args a) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= a.n) return;
    float gi = g[i];
#ifdef RUNNER_TFFN_MUTANT   // never set in the committed PTX: the cutoff moved onto -80 itself
    h[i] = gi <= -80.0f ? 0.0f : __fmul_rn(tw_silu(gi), u[i]);
#else
    h[i] = gi < -80.0f ? 0.0f : __fmul_rn(tw_silu(gi), u[i]);
#endif
}

// dg = dh * u * act_d(g) (left to right), du = dh * act_f(g)
extern "C" __global__ void k_tffn_bw(const float *g, const float *u,
                                     const float *dh, float *dg, float *du,
                                     tffn_args a) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= a.n) return;
    float gi = g[i], d = dh[i];
    dg[i] = __fmul_rn(__fmul_rn(d, u[i]), tw_silu_d(gi));
    du[i] = __fmul_rn(d, tw_silu(gi));
}

// ---- R8.24.1 B6: the adapter at a projection site, host-chain twins
// (kernel_args.h tlora_args). Every element is one thread running the host's
// chain: lora_apply's t[k] (fmaf over i ascending from 0) and y[j] (fmaf over
// k ascending from 0, then fmaf(s, acc, y)); lsb_proj's tb (fmaf over j) and
// tf (the same chain as t); lsb_ga's and lsb_gb's elements (fmaf over t
// ascending from the stored value, the factor s*tb or s*dy rounded first);
// lsb_dx's (fmaf over k from 0, then fmaf(s, acc, dx)). No arithmetic but
// those fmafs and the two products, so the default build's host (explicit
// fmaf there too) and this agree bit for bit on normal numbers.

// t[b][k] = sum_i A[k][i] x[b][i]  (forward t, and the backward's tf)
extern "C" __global__ void k_tlora_t(const float *A, const float *x, float *t,
                                     tlora_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.nb * a.r) return;
    int b = e / a.r, k = e % a.r;
    const float *ar = A + (ulong64)k * a.n_in, *xr = x + (ulong64)b * a.x_stride;
    float acc = 0.0f;
    for (int i = 0; i < a.n_in; i++) acc = __fmaf_rn(ar[i], xr[i], acc);
    t[(ulong64)b * a.r + k] = acc;
}

// y[b][j] = fmaf(s, sum_k B[j][k] t[b][k], y[b][j])
extern "C" __global__ void k_tlora_y(const float *B, const float *t, float *y,
                                     tlora_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.nb * a.n_out) return;
    int b = e / a.n_out, j = e % a.n_out;
    const float *br = B + (ulong64)j * a.r, *tr = t + (ulong64)b * a.r;
    float acc = 0.0f;
    for (int k = 0; k < a.r; k++) acc = __fmaf_rn(br[k], tr[k], acc);
    float *yo = y + (ulong64)b * a.y_stride + j;
#ifdef RUNNER_TLORA_MUTANT   // never set in the committed PTX: the scale's step unfused
    *yo = __fadd_rn(__fmul_rn(a.s, acc), *yo);
#else
    *yo = __fmaf_rn(a.s, acc, *yo);
#endif
}

// tb[b][k] = sum_j B[j][k] dy[b][j]  (lsb_proj's first chain)
extern "C" __global__ void k_tlora_tb(const float *B, const float *dy, float *tb,
                                      tlora_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.nb * a.r) return;
    int b = e / a.r, k = e % a.r;
    const float *dr = dy + (ulong64)b * a.y_stride;
    float acc = 0.0f;
    for (int j = 0; j < a.n_out; j++) acc = __fmaf_rn(B[(ulong64)j * a.r + k], dr[j], acc);
    tb[(ulong64)b * a.r + k] = acc;
}

// ga[k][i]: for t ascending, fmaf(s*tb[t][k], x[t][i], ga)  (lsb_ga)
extern "C" __global__ void k_tlora_ga(const float *tb, const float *x, float *ga,
                                      tlora_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.r * a.n_in) return;
    int k = e / a.n_in, i = e % a.n_in;
    float acc = ga[e];
    for (int t = 0; t < a.nb; t++)
        acc = __fmaf_rn(__fmul_rn(a.s, tb[(ulong64)t * a.r + k]),
                        x[(ulong64)t * a.x_stride + i], acc);
    ga[e] = acc;
}

// gb[j][k]: for t ascending, fmaf(s*dy[t][j], tf[t][k], gb)  (lsb_gb)
extern "C" __global__ void k_tlora_gb(const float *dy, const float *tf, float *gb,
                                      tlora_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.n_out * a.r) return;
    int j = e / a.r, k = e % a.r;
    float acc = gb[e];
    for (int t = 0; t < a.nb; t++)
        acc = __fmaf_rn(__fmul_rn(a.s, dy[(ulong64)t * a.y_stride + j]),
                        tf[(ulong64)t * a.r + k], acc);
    gb[e] = acc;
}

// dx[b][i] = fmaf(s, sum_k A[k][i] tb[b][k], dx[b][i])  (lsb_dx)
extern "C" __global__ void k_tlora_dx(const float *A, const float *tb, float *dx,
                                      tlora_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.nb * a.n_in) return;
    int b = e / a.n_in, i = e % a.n_in;
    const float *tr = tb + (ulong64)b * a.r;
    float acc = 0.0f;
    for (int k = 0; k < a.r; k++) acc = __fmaf_rn(A[(ulong64)k * a.n_in + i], tr[k], acc);
    float *xo = dx + (ulong64)b * a.x_stride + i;
    *xo = __fmaf_rn(a.s, acc, *xo);
}
// ---- R8.24.1 B2: rmsnorm and its adjoint, host-chain twins (kernel_args.h
// trn_args; model.c rmsnorm and rmsnorm_bw). One block a row: thread 0 runs
// the row's chains serially in the host's index order (the forward's ss a
// multiply and an add per element, as rmsnorm writes ss += x*x; the
// adjoint's ss and dot fmaf chains, as rmsnorm_bw writes them), then every
// thread writes its elements, each the host's expression left to right,
// never contracted. r = 1 / sqrtf(ss / n + eps) is correctly rounded on both
// sides (__fsqrt_rn, __fdiv_rn), so a T3 host and these agree bit for bit; a
// NaN comes back as the device's canonical NaN. The serial chain is a few
// microseconds a row at n 2560, small next to the products.
// RUNNER_TRN_MUTANT (never set in the committed PTX) swaps the forward's
// multiply-add for an fmaf: the gate's mutant, which test-train-gpu must fail.
#define TRN_THREADS 128

extern "C" __global__ void __launch_bounds__(TRN_THREADS)
k_trn_fw(const float *x, const float *w, float *y, trn_args a) {
    __shared__ float rs;
    int row = (int)blockIdx.x, n = a.n;
    if (row >= a.rows) return;
    const float *xr = x + (ulong64)row * a.x_stride;
    float *yr = y + (ulong64)row * a.y_stride;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < n; i++)
#ifdef RUNNER_TRN_MUTANT
            ss = __fmaf_rn(xr[i], xr[i], ss);
#else
            ss = __fadd_rn(ss, __fmul_rn(xr[i], xr[i]));
#endif
        rs = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)n), a.eps)));
    }
    __syncthreads();
    float r = rs;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x)
        yr[i] = w ? __fmul_rn(__fmul_rn(xr[i], r), w[i]) : __fmul_rn(xr[i], r);
}

// dx[i] += r * w[i] * dy[i] - c * x[i], c = r * r * r / n * dot (w NULL: 1)
extern "C" __global__ void __launch_bounds__(TRN_THREADS)
k_trn_bw(const float *x, const float *w, const float *dy, float *dx, trn_args a) {
    __shared__ float rs, cs;
    int row = (int)blockIdx.x, n = a.n;
    if (row >= a.rows) return;
    const float *xr = x + (ulong64)row * a.x_stride;
    const float *dyr = dy + (ulong64)row * a.dy_stride;
    float *dxr = dx + (ulong64)row * a.dx_stride;
    if (threadIdx.x == 0) {
        float ss = 0.0f, dot = 0.0f;
        for (int i = 0; i < n; i++) ss = __fmaf_rn(xr[i], xr[i], ss);
        float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)n), a.eps)));
        for (int i = 0; i < n; i++)
            dot = __fmaf_rn(w ? __fmul_rn(w[i], dyr[i]) : __fmul_rn(1.0f, dyr[i]), xr[i], dot);
        rs = r;
        cs = __fmul_rn(__fdiv_rn(__fmul_rn(__fmul_rn(r, r), r), (float)n), dot);
    }
    __syncthreads();
    float r = rs, c = cs;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        float rw = w ? __fmul_rn(r, w[i]) : __fmul_rn(r, 1.0f);
        dxr[i] = __fadd_rn(dxr[i], __fsub_rn(__fmul_rn(rw, dyr[i]), __fmul_rn(c, xr[i])));
    }
}

// ---- R8.24.1 B3: rope and its adjoint, host-chain twins (kernel_args.h
// trope_args; model.c rope_apply and rope_unapply). One thread a (position,
// pair j): the angle pos * fr[j] in float, its sin and cos as src/pmath.h's
// pm_sincos_d computes them (ported 1:1 below: double add, sub and mul, the
// same reduction by pi/2 and the same Taylor polynomials, never contracted),
// each rounded once to float and multiplied by mscale, then the pair rotated
// in every head (forward x0*c - x1*s, x0*s + x1*c; adjoint x0*c + x1*s,
// -x0*s + x1*c). A T3 host calls pmath's sinf and cosf, so the two agree bit
// for bit; a default build's host calls the C library's. NaN in, NaN out.
// RUNNER_TROPE_MUTANT (never set in the committed PTX) fuses the forward's
// x0*c - x1*s into an fmaf: the gate's mutant.
#define TROPE_THREADS 128

static __device__ __forceinline__ void tw_sincos_d(double x, double *sn, double *cs) {
    const double pio2_hi = 1.57079632679489655800e+00;
    const double pio2_lo = 6.12323399573676603587e-17;
    if (!(x < 1e15 && x > -1e15)) {
        if (!(x == x) || x > 1.7976931348623157e308 || x < -1.7976931348623157e308) {
            *sn = *cs = __longlong_as_double(0x7ff8000000000000LL);
            return;
        }
        x = fmod(x, 6.28318530717958647693);   // exact in IEEE 754, as on the host
    }
    double kd = __dmul_rn(x, 6.36619772367581382433e-01);
    long long k = (long long)(kd >= 0 ? __dadd_rn(kd, 0.5) : __dsub_rn(kd, 0.5));
    double r = __dsub_rn(__dsub_rn(x, __dmul_rn((double)k, pio2_hi)),
                         __dmul_rn((double)k, pio2_lo));
    double r2 = __dmul_rn(r, r);
    double s = -1.0 / 1307674368000.0;
    s = __dadd_rn(__dmul_rn(s, r2), 1.0 / 6227020800.0);
    s = __dsub_rn(__dmul_rn(s, r2), 1.0 / 39916800.0);
    s = __dadd_rn(__dmul_rn(s, r2), 1.0 / 362880.0);
    s = __dsub_rn(__dmul_rn(s, r2), 1.0 / 5040.0);
    s = __dadd_rn(__dmul_rn(s, r2), 1.0 / 120.0);
    s = __dsub_rn(__dmul_rn(s, r2), 1.0 / 6.0);
    s = __dadd_rn(__dmul_rn(s, r2), 1.0);
    s = __dmul_rn(s, r);
    double c = 1.0 / 20922789888000.0;
    c = __dsub_rn(__dmul_rn(c, r2), 1.0 / 87178291200.0);
    c = __dadd_rn(__dmul_rn(c, r2), 1.0 / 479001600.0);
    c = __dsub_rn(__dmul_rn(c, r2), 1.0 / 3628800.0);
    c = __dadd_rn(__dmul_rn(c, r2), 1.0 / 40320.0);
    c = __dsub_rn(__dmul_rn(c, r2), 1.0 / 720.0);
    c = __dadd_rn(__dmul_rn(c, r2), 1.0 / 24.0);
    c = __dsub_rn(__dmul_rn(c, r2), 0.5);
    c = __dadd_rn(__dmul_rn(c, r2), 1.0);
    switch (k & 3) {
        case 0:  *sn = s;  *cs = c;  break;
        case 1:  *sn = c;  *cs = -s; break;
        case 2:  *sn = -s; *cs = -c; break;
        default: *sn = -c; *cs = s;  break;
    }
}

extern "C" __global__ void __launch_bounds__(TROPE_THREADS)
k_trope(float *v, const float *fr, trope_args a) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)a.T * a.half_dim) return;
    int t = (int)(g / a.half_dim), j = (int)(g % a.half_dim);
    float ang = __fmul_rn(__int2float_rn(a.t0 + t), fr[j]);
    double sd, cd;
    tw_sincos_d((double)ang, &sd, &cd);
    float c = __fmul_rn(__double2float_rn(cd), a.mscale);
    float s = __fmul_rn(__double2float_rn(sd), a.mscale);
    float *row = v + (ulong64)t * a.stride;
    for (int h = 0; h < a.n_heads; h++) {
        float *p = row + (ulong64)h * a.head_dim;
        float *p0 = a.neox ? p + j : p + 2 * j;
        float *p1 = a.neox ? p + j + a.half_dim : p0 + 1;
        float x0 = *p0, x1 = *p1;
        if (a.adjoint) {
            *p0 = __fadd_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
            *p1 = __fadd_rn(__fmul_rn(-x0, s), __fmul_rn(x1, c));
        } else {
#ifdef RUNNER_TROPE_MUTANT
            *p0 = __fmaf_rn(x0, c, -__fmul_rn(x1, s));
#else
            *p0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
#endif
            *p1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
        }
    }
}


// ---- R8.24.1 B8: the training K/V store (kernel_args.h tkv_args), a twin of
// the forward's store for the cache kinds 0 (f16) and 3 (f32). The f16 rounding
// is fp16.h's f32_to_f16_soft line for line, integers only, so every one of the
// 2^32 inputs lands on the host's bits. (Kernel arguments here stay small and
// are read whole: a larger struct only partly read makes nvcc annotate its
// loads with a pragma that kernels_ptx.h's pinned .version 7.8 rejects in the
// driver's JIT, which ptxas offline does not catch.)
static __device__ __forceinline__ unsigned short tw_f32_to_f16(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int sign = (u >> 16) & 0x8000u;
    int exp = (int)((u >> 23) & 0xFFu) - 127 + 15;
    unsigned int mant = u & 0x7FFFFFu;
    if (exp >= 31) {
        if (exp == 143 && mant)
            return (unsigned short)(sign | 0x7C00u | (mant >> 13) | 0x0200u);
        return (unsigned short)(sign | 0x7C00u);
    }
    if (exp <= 0) {
        if (exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        unsigned int shift = (unsigned int)(14 - exp);
        unsigned int h = mant >> shift;
        unsigned int rem = mant & ((1u << shift) - 1);
        unsigned int half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (unsigned short)(sign | h);
    }
    unsigned int h = ((unsigned int)exp << 10) | (mant >> 13);
    unsigned int rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return (unsigned short)(sign | h);
}

// dst row r, element i: src[r][i] in the cache's kind
extern "C" __global__ void k_tkv_store(const float *src, unsigned char *dst, tkv_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.rows * a.n) return;
    int r = e / a.n, i = e % a.n;
    float v = src[(ulong64)r * a.src_stride + i];
    unsigned char *row = dst + (ulong64)r * a.row_bytes;
    if (a.kind == 3) ((float *)row)[i] = v;
    else ((unsigned short *)row)[i] = tw_f32_to_f16(v);
}
// ---- R8.24.1 B4: the forward's attention over the training K/V rows, a
// host-chain twin of model.c's attn_run as the recompute calls it (one
// attn_job a position: attn_scores, softmax, attn_values; kernel_args.h
// tattn_args). One block a (position t, head h), the span s = t0 .. t (t0
// the window start, attn_window_start):
// - scores, one thread a key s: q . k_s as the host's chain over the head's
//   channels ascending from 0 (multiply, then add), then times scale;
// - softmax over the span, as model.c softmax: the max by thread 0 in key
//   order (x > mx, so a NaN keeps the host's answer), exp(x - mx) a key a
//   thread (pmath's exp, tw_expf), their sum by thread 0 in key order, then
//   each divided by it;
// - values, one thread a channel i: the chain over the span ascending from
//   0, p_s * v_s[i] added a key at a time.
// Every chain is the host's, never contracted, so a T3 host and this agree
// bit for bit. K and V rows are f16 (kind 0) or f32 (kind 3).
// RUNNER_TATTN_MUTANT (never set in the committed PTX) fuses the value
// chain's multiply-add into an fmaf: the gate's mutant.
#define TATTN_THREADS 128
#define TATTN_MAX_HD 512

static __device__ __forceinline__ float tattn_kv(const uchar *row, int kind, int i) {
    return kind == 3 ? ((const float *)row)[i]
                     : __half2float(((const __half *)row)[i]);
}

extern "C" __global__ void __launch_bounds__(TATTN_THREADS)
k_tattn(const uchar *kc, const uchar *vc, const float *q, float *ao, tattn_args a) {
    extern __shared__ float att[];          // the span's scores, then probabilities
    __shared__ float qs[TATTN_MAX_HD], mxs, sums;
    int b = (int)blockIdx.x, h = (int)blockIdx.y, tid = (int)threadIdx.x, hd = a.hd;
    int kvh = h / a.kv_mul;
    if (b >= a.T || h >= a.n_head || (kvh + 1) * hd > a.kv_dim) return;
    int t = a.pos0 + b;   // the query's position: its window and keys are absolute rows
    int t0 = a.swa > 0 && t - a.swa + 1 > 0 ? t - a.swa + 1 : 0, span = t - t0 + 1;
    int ek = a.kind_k == 3 ? 4 : 2, ev = a.kind_v == 3 ? 4 : 2;
    for (int i = tid; i < hd; i += TATTN_THREADS) qs[i] = q[(ulong64)b * a.q_dim + h * hd + i];
    __syncthreads();
    for (int k = tid; k < span; k += TATTN_THREADS) {
        const uchar *kr = kc + (ulong64)(t0 + k) * a.k_row_b + (ulong64)kvh * hd * ek;
        float s = 0.0f;
        for (int i = 0; i < hd; i++) s = __fadd_rn(s, __fmul_rn(qs[i], tattn_kv(kr, a.kind_k, i)));
        att[k] = __fmul_rn(s, a.scale);
    }
    __syncthreads();
    if (tid == 0) {
        float mx = att[0];
        for (int k = 1; k < span; k++) if (att[k] > mx) mx = att[k];
        mxs = mx;
    }
    __syncthreads();
    float mx = mxs;
    for (int k = tid; k < span; k += TATTN_THREADS) att[k] = tw_expf(__fsub_rn(att[k], mx));
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < span; k++) s = __fadd_rn(s, att[k]);
        sums = s;
    }
    __syncthreads();
    float sum = sums;
    for (int k = tid; k < span; k += TATTN_THREADS) att[k] = __fdiv_rn(att[k], sum);
    __syncthreads();
    for (int i = tid; i < hd; i += TATTN_THREADS) {
        float o = 0.0f;
        for (int k = 0; k < span; k++) {
            const uchar *vr = vc + (ulong64)(t0 + k) * a.v_row_b + (ulong64)kvh * hd * ev;
#ifdef RUNNER_TATTN_MUTANT
            o = __fmaf_rn(att[k], tattn_kv(vr, a.kind_v, i), o);
#else
            o = __fadd_rn(o, __fmul_rn(att[k], tattn_kv(vr, a.kind_v, i)));
#endif
        }
        ao[(ulong64)b * a.q_dim + h * hd + i] = o;
    }
}

// ---- R8.24.1 B, R0 of the resident step: the residual add (kernel_args.h
// tadd_args), a twin of layer_branch_out's x += y: one IEEE add an element.
extern "C" __global__ void k_tadd(float *x, const float *y, tadd_args a) {
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= a.rows * a.n) return;
    int r = e / a.n, i = e % a.n;
    float *p = x + (ulong64)r * a.x_stride + i;
    *p = __fadd_rn(*p, y[(ulong64)r * a.y_stride + i]);
}

// ---- R8.24.1 B5: the head, a host-chain twin (kernel_args.h thead_args) of
// model.c's apply_head_transforms then train_row_lse (the taped forward's
// loss) and head_rows_bw's per-row adjoint without a teacher. One block a
// scored row (targets[r] >= 0; other rows untouched):
// - the transforms one thread an element: times logit_scale (0 or 1: none),
//   then softcap: th = tanh(z / c), z = c * th (pmath's tanh, tw_tanhf, th
//   kept for the adjoint); then the suppressed ids set to -1e30 by one
//   thread in list order;
// - z_target[r] = the transformed target logit; the max by one thread in
//   index order (z > mx);
// - e = exp(z - mx) one thread an element (pmath's exp); then one thread in
//   index order sums them in float (softmax's normalizer) and, each widened,
//   in double (train_row_lse's); lse[r] = (double)(mx + logf((float)dsum))
//   with pmath's log;
// - the adjoint, each step as head_rows_bw takes it: e / sum, minus 1 at the
//   target, times pos_w[r], the suppressed ids set to 0, times 1 - th * th,
//   times the scale.
// Every chain is the host's, never contracted, so a T3 host and this agree
// bit for bit. The sums over V are serial on one thread: about 3 V dependent
// operations a row, a named cost, not a tree.
// RUNNER_THEAD_MUTANT and RUNNER_THEAD_MUTANT2 (never set in the committed
// PTX): the softmax's normalizer summed in double and rounded once, and the
// loss's normalizer summed in float: the gate's mutants, one for each sum.
#define THEAD_THREADS 256

// p_tanhf (src/pmath.h) ported 1:1
static __device__ __forceinline__ float tw_tanhf(float x) {
    double xd = (double)x;
    if (xd > 20.0) return 1.0f;
    if (xd < -20.0) return -1.0f;
    if (xd < 1e-4 && xd > -1e-4)
        return __double2float_rn(__dmul_rn(xd, __dsub_rn(1.0, __dmul_rn(__dmul_rn(xd, xd), 1.0 / 3.0))));
    double e = tw_exp_d(__dmul_rn(2.0, xd));
    return __double2float_rn(__ddiv_rn(__dsub_rn(e, 1.0), __dadd_rn(e, 1.0)));
}
// pm_log_d (src/pmath.h) ported 1:1
static __device__ __forceinline__ double tw_log_d(double x) {
    if (x != x) return x;
    if (x <= 0.0) return x == 0.0 ? __longlong_as_double(0xfff0000000000000LL)
                                  : __longlong_as_double(0x7ff8000000000000LL);
    if (x > 1.7976931348623157e308) return x;
    unsigned long long u = (unsigned long long)__double_as_longlong(x);
    int k = (int)((u >> 52) & 0x7ff) - 1023;
    u = (u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m = __longlong_as_double((long long)u);
    if (m > 1.41421356237309504880) { m = __dmul_rn(m, 0.5); k += 1; }
    double s = __ddiv_rn(__dsub_rn(m, 1.0), __dadd_rn(m, 1.0)), s2 = __dmul_rn(s, s);
    double p = 1.0 / 25.0;
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 23.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 21.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 19.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 17.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 15.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 13.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 11.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 9.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 7.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 5.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0 / 3.0);
    p = __dadd_rn(__dmul_rn(p, s2), 1.0);
    return __dadd_rn(__dmul_rn((double)k, 6.93147180559945309417e-01), __dmul_rn(__dmul_rn(2.0, s), p));
}

extern "C" __global__ void __launch_bounds__(THEAD_THREADS)
k_thead(float *z, float *th, const int *targets, const float *pos_w, const int *suppress,
        double *lse, float *z_target, thead_args a) {
    __shared__ float mxs, sums;
    int r = (int)blockIdx.x, tid = (int)threadIdx.x, V = a.V;
    if (r >= a.rows) return;
    int tg = targets[r];
    if (tg < 0 || tg >= V) return;
    float *zr = z + (ulong64)r * a.stride;
    float *tr = a.softcap > 0.0f ? th + (ulong64)r * a.stride : NULL;
    float ls = a.logit_scale != 0.0f ? a.logit_scale : 1.0f, c = a.softcap;
    for (int i = tid; i < V; i += THEAD_THREADS) {
        float v = zr[i];
        if (ls != 1.0f) v = __fmul_rn(v, ls);
        if (tr) {
            float t = tw_tanhf(__fdiv_rn(v, c));
            tr[i] = t;
            v = __fmul_rn(c, t);
        }
        zr[i] = v;
    }
    __syncthreads();
    if (tid == 0) {
        for (int k = 0; k < a.n_suppress; k++) zr[suppress[k]] = -1e30f;
        z_target[r] = zr[tg];
        float mx = zr[0];
        for (int i = 1; i < V; i++) if (zr[i] > mx) mx = zr[i];
        mxs = mx;
    }
    __syncthreads();
    float mx = mxs;
    for (int i = tid; i < V; i += THEAD_THREADS) zr[i] = tw_expf(__fsub_rn(zr[i], mx));
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        double ds = 0.0;
        for (int i = 0; i < V; i++) {
            float e = zr[i];
            ds = __dadd_rn(ds, (double)e);
#ifdef RUNNER_THEAD_MUTANT
            (void)s;
#else
            s = __fadd_rn(s, e);
#endif
        }
#ifdef RUNNER_THEAD_MUTANT
        s = __double2float_rn(ds);
#endif
#ifdef RUNNER_THEAD_MUTANT2
        lse[r] = (double)__fadd_rn(mx, __double2float_rn(tw_log_d((double)s)));   // the loss's sum in float
#else
        lse[r] = (double)__fadd_rn(mx, __double2float_rn(tw_log_d((double)__double2float_rn(ds))));
#endif
        sums = s;
    }
    __syncthreads();
    float s = sums;
    for (int i = tid; i < V; i += THEAD_THREADS) zr[i] = __fdiv_rn(zr[i], s);
    __syncthreads();
    if (tid == 0) zr[tg] = __fsub_rn(zr[tg], 1.0f);
    __syncthreads();
    if (pos_w) {
        float pw = pos_w[r];
        for (int i = tid; i < V; i += THEAD_THREADS) zr[i] = __fmul_rn(zr[i], pw);
        __syncthreads();
    }
    if (tid == 0)
        for (int k = 0; k < a.n_suppress; k++) zr[suppress[k]] = 0.0f;
    __syncthreads();
    for (int i = tid; i < V; i += THEAD_THREADS) {
        float v = zr[i];
        if (tr) v = __fmul_rn(v, __fsub_rn(1.0f, __fmul_rn(tr[i], tr[i])));
        if (ls != 1.0f) v = __fmul_rn(v, ls);
        zr[i] = v;
    }
}

// ---- R8.24.1 B7: AdamW and the step log's gradient norms, host-chain
// twins. k_tadam (kernel_args.h tadam_args), one thread an element, as
// model.c adam_buf writes it: m = b1*m + (1 - b1)*g; v = b2*v + (1 - b2)*g*g
// (left to right); mh = m / bc1, vh = v / bc2; th -= lr * (mh / (sqrtf(vh) +
// eps) + wd * th). k_tgnorm, model_lora_grad_norms: one block, a thread a
// layer, its segments in order and their elements in order, the double sum of
// g*g (each widened, then multiplied, then added), its square root; then one
// thread takes the max (nrm > mx) and the sum over the layers that have a
// segment, in layer order, and the mean as sum / layers; a non-finite
// gradient anywhere clears out[2]. Never contracted, so a T3 host and these
// agree bit for bit.
// Mutants (never set in the committed PTX), one for each step a case can see:
// RUNNER_TADAM_MUTANT fuses the first moment's update into an fmaf;
// RUNNER_TADAM_MUTANT2 takes the bias corrections as a reciprocal times;
// RUNNER_TADAM_MUTANT3 fuses the weight-decay term into an fmaf (seen only
// where it cancels the moment term: the gate's cancellation case);
// RUNNER_TGNORM_MUTANT sums each segment in reverse. (A fused g*g in the
// norms cannot be seen: a float times a float is exact in double.)
#define TADAM_THREADS 256

extern "C" __global__ void __launch_bounds__(TADAM_THREADS)
k_tadam(float *th, const float *g, float *mo, float *vo, tadam_args a) {
    ulong64 i = (ulong64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n) return;
    float gi = g[i];
#ifdef RUNNER_TADAM_MUTANT
    float m = __fmaf_rn(a.b1, mo[i], __fmul_rn(__fsub_rn(1.0f, a.b1), gi));
#else
    float m = __fadd_rn(__fmul_rn(a.b1, mo[i]), __fmul_rn(__fsub_rn(1.0f, a.b1), gi));
#endif
    float v = __fadd_rn(__fmul_rn(a.b2, vo[i]), __fmul_rn(__fmul_rn(__fsub_rn(1.0f, a.b2), gi), gi));
    mo[i] = m;
    vo[i] = v;
#ifdef RUNNER_TADAM_MUTANT2
    float mh = __fmul_rn(m, __frcp_rn(a.bc1)), vh = __fmul_rn(v, __frcp_rn(a.bc2)), t = th[i];
#else
    float mh = __fdiv_rn(m, a.bc1), vh = __fdiv_rn(v, a.bc2), t = th[i];
#endif
    float q = __fdiv_rn(mh, __fadd_rn(__fsqrt_rn(vh), a.eps));
#ifdef RUNNER_TADAM_MUTANT3
    th[i] = __fsub_rn(t, __fmul_rn(a.lr, __fmaf_rn(a.wd, t, q)));
#else
    th[i] = __fsub_rn(t, __fmul_rn(a.lr, __fadd_rn(q, __fmul_rn(a.wd, t))));
#endif
}

extern "C" __global__ void
k_tgnorm(const float *g, const ulong64 *seg_off, const ulong64 *seg_len, const int *layer_seg,
         int n_layer, double *out) {
    extern __shared__ double nrm[];               // n_layer norms, then n_layer flags
    int *flag = (int *)(nrm + n_layer);
    int l = (int)threadIdx.x;
    if (l < n_layer) {
        double ss = 0.0;
        int any = 0, fin = 1;
        for (int k = layer_seg[l]; k < layer_seg[l + 1]; k++) {
            any = 1;
            const float *gs = g + seg_off[k];
            for (ulong64 i = 0; i < seg_len[k]; i++) {
#ifdef RUNNER_TGNORM_MUTANT
                float v = gs[seg_len[k] - 1 - i];
#else
                float v = gs[i];
#endif
                if (!(fabsf(v) <= 3.40282346638528859812e38f)) fin = 0;   // not finite
                ss = __dadd_rn(ss, __dmul_rn((double)v, (double)v));
            }
        }
        nrm[l] = __dsqrt_rn(ss);
        flag[l] = any | (fin << 1);
    }
    __syncthreads();
    if (l == 0) {
        double mx = 0.0, sum = 0.0;
        int layers = 0, fin = 1;
        for (int k = 0; k < n_layer; k++) {
            if (!(flag[k] & 2)) fin = 0;
            if (!(flag[k] & 1)) continue;
            if (nrm[k] > mx) mx = nrm[k];
            sum = __dadd_rn(sum, nrm[k]);
            layers++;
        }
        out[0] = mx;
        out[1] = layers ? __ddiv_rn(sum, (double)layers) : 0.0;
        out[2] = fin ? 1.0 : 0.0;
    }
}

// ---- R8.24.1 B R4: the mean's scale on the step's gradients, the host-chain
// twin of model_lora_grad_scale's loop (g[i] *= f), one thread an element: one
// rounded product, so a T3 host and this agree bit for bit.
// RUNNER_TSCALE_MUTANT (never set in the committed PTX) rounds the product
// toward zero.
extern "C" __global__ void __launch_bounds__(TADAM_THREADS)
k_tscale(float *g, ulong64 n, float f) {
    ulong64 i = (ulong64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#ifdef RUNNER_TSCALE_MUTANT
    g[i] = __fmul_rz(g[i], f);
#else
    g[i] = __fmul_rn(g[i], f);
#endif
}
