// The pipelined tensor-core GEMM (TC2) shared by kernels.cu (compute_75)
// and kernels_sm80.cu (compute_80): its tile constants, staging helpers,
// the loaders of the self-contained formats (K-quants, Q8_0, Q4_0) and the
// GEMM body. The codebook i-quant loaders stay in kernels.cu beside the
// stagers they call. Includers define uchar, ushort16, uint and ulong64 and
// include cuda_fp16.h and kernel_args.h first.
#ifndef RUNNER_KERNELS_TC2_CUH
#define RUNNER_KERNELS_TC2_CUH

#define G2_BM 128
#define G2_BN 64
#define G2_BK 64
#define G2_LD (G2_BK + 8)
#define G2_SMEM ((2 * G2_BM * G2_LD + 2 * G2_BN * G2_LD) * 2)   // bytes, dynamic
// the 128-column tile (compute_80 module only: 73,728 bytes of dynamic shared
// memory, past compute capability 7.5's 64 KB per block)
#define G2_BN_W 128
#define G2_SMEM_W ((2 * G2_BM * G2_LD + 2 * G2_BN_W * G2_LD) * 2)

static __device__ __forceinline__ void g2_ldsm_x4(unsigned &r0, unsigned &r1, unsigned &r2,
                                                  unsigned &r3, const __half *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

static __device__ __forceinline__ void g2_mma(float *c, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

#ifdef G2_MMA_K16
// compute_80 and newer (kernels_sm80.cu): one m16n8k16 per 16-deep step. The
// four registers the ldmatrix above returns are exactly its A operand (rows
// 0-7 and 8-15 at k 0-7, then at k 8-15) and b0/b1 its B halves, so the step
// that took two m16n8k8 takes one instruction.
static __device__ __forceinline__ void g2_mma16(float *c, unsigned a0, unsigned a1,
                                                unsigned a2, unsigned a3,
                                                unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

static __device__ __forceinline__ unsigned g2_pack(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *(unsigned *)&h;
}

// get_scale_min_k4 on the 12 scale bytes held as three words (bytes 0-3,
// 4-7, 8-11), so they stay in registers: the same integers it returns.
static __device__ __forceinline__ void g2_scale_min_k4(int j, uint s0, uint s1, uint s2,
                                                       uint *d, uint *m) {
    if (j < 4) {
        *d = (s0 >> (8 * j)) & 63;
        *m = (s1 >> (8 * j)) & 63;
    } else {
        int i = 8 * (j - 4);
        *d = ((s2 >> i) & 0xF) | (((s0 >> i) >> 6 & 3) << 4);
        *m = ((s2 >> i) >> 4 & 0xF) | (((s1 >> i) >> 6 & 3) << 4);
    }
}

// Q4_K: one 64-element segment of one row is 32 quant bytes (the low
// nibbles are its first 32 elements, the high nibbles its last 32), two
// scale groups and the block's d and dmin. Values as k_gemm_q4_K_tc
// dequantizes them: dg * q - mm in fp32, then rounded to half.
struct G2Q4K {
    struct raw { uint4 q0, q1; uint ddm, s0, s1, s2; };
    enum { kDirect = 0 };
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks, int n_in) {
        (void)n_in;
        const uchar *blk = rowp + (ulong64)(ks >> 8) * 144;
        r.ddm = *(const uint *)blk;
        r.s0 = *(const uint *)(blk + 4);
        r.s1 = *(const uint *)(blk + 8);
        r.s2 = *(const uint *)(blk + 12);
        const uint4 *q = (const uint4 *)(blk + 16 + ((ks >> 6) & 3) * 32);
        r.q0 = q[0];
        r.q1 = q[1];
    }
    static __device__ __forceinline__ void store(__half *dst, const raw &r, int ks) {
        const int seg = (ks >> 6) & 3;
        unsigned short hd = (unsigned short)(r.ddm & 0xFFFF), hm = (unsigned short)(r.ddm >> 16);
        float dd = __half2float(*(const __half *)&hd), dmin = __half2float(*(const __half *)&hm);
        uint sg0, mg0, sg1, mg1;
        g2_scale_min_k4(2 * seg + 0, r.s0, r.s1, r.s2, &sg0, &mg0);
        g2_scale_min_k4(2 * seg + 1, r.s0, r.s1, r.s2, &sg1, &mg1);
        float dg0 = dd * (float)sg0, mm0 = dmin * (float)mg0;
        float dg1 = dd * (float)sg1, mm1 = dmin * (float)mg1;
        uint w[8] = { r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w };
        #pragma unroll
        for (int v = 0; v < 4; v++) {      // 8 bytes -> 8 low, 8 high elements
            uint lo0 = w[2 * v] & 0x0F0F0F0Fu, lo1 = w[2 * v + 1] & 0x0F0F0F0Fu;
            uint hi0 = (w[2 * v] >> 4) & 0x0F0F0F0Fu, hi1 = (w[2 * v + 1] >> 4) & 0x0F0F0F0Fu;
            uint4 l, h;
            l.x = g2_pack(dg0 * (float)(lo0 & 0xFF) - mm0, dg0 * (float)((lo0 >> 8) & 0xFF) - mm0);
            l.y = g2_pack(dg0 * (float)((lo0 >> 16) & 0xFF) - mm0, dg0 * (float)(lo0 >> 24) - mm0);
            l.z = g2_pack(dg0 * (float)(lo1 & 0xFF) - mm0, dg0 * (float)((lo1 >> 8) & 0xFF) - mm0);
            l.w = g2_pack(dg0 * (float)((lo1 >> 16) & 0xFF) - mm0, dg0 * (float)(lo1 >> 24) - mm0);
            h.x = g2_pack(dg1 * (float)(hi0 & 0xFF) - mm1, dg1 * (float)((hi0 >> 8) & 0xFF) - mm1);
            h.y = g2_pack(dg1 * (float)((hi0 >> 16) & 0xFF) - mm1, dg1 * (float)(hi0 >> 24) - mm1);
            h.z = g2_pack(dg1 * (float)(hi1 & 0xFF) - mm1, dg1 * (float)((hi1 >> 8) & 0xFF) - mm1);
            h.w = g2_pack(dg1 * (float)((hi1 >> 16) & 0xFF) - mm1, dg1 * (float)(hi1 >> 24) - mm1);
            *(uint4 *)(dst + v * 8) = l;
            *(uint4 *)(dst + 32 + v * 8) = h;
        }
    }
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) { return (ulong64)(n_in / 256) * 144; }
};

// N little-endian words of the bytes at p, which is only 2-byte aligned, from
// 8-byte loads at p rounded down: (N + 2) / 2 loads where 2N two-byte loads
// took one sector access each (an RTX 3070 profile, 2026-10-22: 4.1 of every
// 32 bytes a sector used). The words equal the two-byte reads'. The loads
// reach at most 6 bytes before p and 8 past p + 4N. The callers keep the far
// end inside the 210-byte block; the near end stays in the allocation,
// because a device allocation starts 256-byte aligned at a tensor's first
// byte, so a block at an allocation's start has p aligned (off 0) and any
// other block has earlier bytes of the same allocation before it.
template <int N>
static __device__ __forceinline__ void g2_words_a2(uint (&out)[N], const uchar *p) {
    const ulong64 a = (ulong64)p & ~(ulong64)7;
    const int off = (int)((ulong64)p & 7);      // 0, 2, 4 or 6
    const uint2 *q = (const uint2 *)a;
    uint w[N + 2];
    #pragma unroll
    for (int i = 0; i < (N + 2) / 2; i++) {
        uint2 v = q[i];
        w[2 * i] = v.x;
        w[2 * i + 1] = v.y;
    }
    const bool skip = off >= 4;                 // start in the second word
    const uint sh = (uint)(off & 3) * 8;        // 0 or 16
    #pragma unroll
    for (int i = 0; i < N; i++) {
        uint lo = skip ? w[i + 1] : w[i], hi = skip ? w[i + 2] : w[i + 1];
        out[i] = __funnelshift_r(lo, hi, sh);
    }
}

// Q6_K: 210-byte blocks of 256, so rows are only 2-byte aligned and the raw
// bytes come in halves. A 64-element segment reads 64 low-bit bytes (ql, both
// the l and l + 32 halves of its 128-element half), 32 high-bit bytes (qh),
// 8 signed scales and d. Values as k_gemm_q6_K_tc: d * (float)(sc * q), the
// scale product in integers first.
struct G2Q6K {
    struct raw { uint ql[16], qh[8], sc[2]; unsigned short d; };
    enum { kDirect = 0 };
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks, int n_in) {
        (void)n_in;
        const uchar *blk = rowp + (ulong64)(ks >> 8) * 210;
        const int half_ = (ks >> 7) & 1;
        const unsigned short *sc = (const unsigned short *)(blk + 192 + half_ * 8);
        // ql spans block bytes 0-127 and qh 128-191: the wide loads stay
        // within bytes -6 to 199 of the block (see g2_words_a2 for the bytes
        // before the block)
        g2_words_a2<16>(r.ql, blk + half_ * 64);
        g2_words_a2<8>(r.qh, blk + 128 + half_ * 32);
        r.sc[0] = (uint)sc[0] | ((uint)sc[1] << 16);
        r.sc[1] = (uint)sc[2] | ((uint)sc[3] << 16);
        r.d = *(const unsigned short *)(blk + 208);
    }
    static __device__ __forceinline__ void store(__half *dst, const raw &r, int ks) {
        const int sseg = (ks >> 6) & 1;
        const int shA = sseg * 4 + 0, shB = sseg * 4 + 2;
        // the scales sc[sseg*4 + is] and sc[sseg*4 + 2 + is]: word sseg,
        // bytes is and 2 + is (a select, so the words stay in registers)
        const uint scw = sseg ? r.sc[1] : r.sc[0];
        float d = __half2float(*(const __half *)&r.d);
        #pragma unroll
        for (int l8 = 0; l8 < 32; l8 += 8) {
            float va[8], vb[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                const int l = l8 + u, is = l >> 4;
                uint a0 = (r.ql[l >> 2] >> (8 * (l & 3))) & 0xFF;
                uint b0 = (r.ql[(l + 32) >> 2] >> (8 * (l & 3))) & 0xFF;
                uint h = (r.qh[l >> 2] >> (8 * (l & 3))) & 0xFF;
                int qa = (int)((sseg ? (a0 >> 4) : (a0 & 0xF)) | (((h >> shA) & 3) << 4)) - 32;
                int qb = (int)((sseg ? (b0 >> 4) : (b0 & 0xF)) | (((h >> shB) & 3) << 4)) - 32;
                int sa = (int)(signed char)((scw >> (8 * is)) & 0xFF);
                int sb = (int)(signed char)((scw >> (8 * (2 + is))) & 0xFF);
                va[u] = d * (float)(sa * qa);
                vb[u] = d * (float)(sb * qb);
            }
            uint4 A, B;
            A.x = g2_pack(va[0], va[1]); A.y = g2_pack(va[2], va[3]);
            A.z = g2_pack(va[4], va[5]); A.w = g2_pack(va[6], va[7]);
            B.x = g2_pack(vb[0], vb[1]); B.y = g2_pack(vb[2], vb[3]);
            B.z = g2_pack(vb[4], vb[5]); B.w = g2_pack(vb[6], vb[7]);
            *(uint4 *)(dst + l8) = A;
            *(uint4 *)(dst + 32 + l8) = B;
        }
    }
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) { return (ulong64)(n_in / 256) * 210; }
};

// byte k of a raw segment held as words (k a compile-time constant once the
// loops unroll, so the words stay in registers)
#define G2B(w, k) (((w)[(k) >> 2] >> (8 * ((k) & 3))) & 0xFFu)

// Q8_0: a 64-element segment is two 34-byte blocks, 4-byte aligned whenever
// n_in is a multiple of 64 (the dispatch requires it). d * (float)q, as
// tc_stage_q8_0.
struct G2Q80 {
    struct raw { uint w[17]; };
    enum { kDirect = 0 };
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks, int n_in) {
        (void)n_in;
        const uint *p = (const uint *)(rowp + (ulong64)(ks >> 5) * 34);
        #pragma unroll
        for (int i = 0; i < 17; i++) r.w[i] = p[i];
    }
    static __device__ __forceinline__ void store(__half *dst, const raw &r, int ks) {
        (void)ks;
        #pragma unroll
        for (int blk = 0; blk < 2; blk++) {
            const int o = blk * 34;
            unsigned short hd = (unsigned short)(G2B(r.w, o) | (G2B(r.w, o + 1) << 8));
            float d = __half2float(*(const __half *)&hd);
            #pragma unroll
            for (int j = 0; j < 32; j += 8) {
                float v[8];
                #pragma unroll
                for (int u = 0; u < 8; u++) v[u] = d * (float)(signed char)G2B(r.w, o + 2 + j + u);
                uint4 o;
                o.x = g2_pack(v[0], v[1]); o.y = g2_pack(v[2], v[3]);
                o.z = g2_pack(v[4], v[5]); o.w = g2_pack(v[6], v[7]);
                *(uint4 *)(dst + blk * 32 + j) = o;
            }
        }
    }
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) { return (ulong64)(n_in / 32) * 34; }
};

// Q4_0: two 18-byte blocks per segment, 4-byte aligned as above.
// d * (float)(q - 8), as tc_stage_q4_0.
struct G2Q40 {
    struct raw { uint w[9]; };
    enum { kDirect = 0 };
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks, int n_in) {
        (void)n_in;
        const uint *p = (const uint *)(rowp + (ulong64)(ks >> 5) * 18);
        #pragma unroll
        for (int i = 0; i < 9; i++) r.w[i] = p[i];
    }
    static __device__ __forceinline__ void store(__half *dst, const raw &r, int ks) {
        (void)ks;
        #pragma unroll
        for (int blk = 0; blk < 2; blk++) {
            const int o = blk * 18;
            unsigned short hd = (unsigned short)(G2B(r.w, o) | (G2B(r.w, o + 1) << 8));
            float d = __half2float(*(const __half *)&hd);
            #pragma unroll
            for (int j = 0; j < 16; j += 8) {
                float lo[8], hi[8];
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    uint q = G2B(r.w, o + 2 + j + u);
                    lo[u] = d * (float)((int)(q & 0xF) - 8);
                    hi[u] = d * (float)((int)(q >> 4) - 8);
                }
                uint4 L, H;
                L.x = g2_pack(lo[0], lo[1]); L.y = g2_pack(lo[2], lo[3]);
                L.z = g2_pack(lo[4], lo[5]); L.w = g2_pack(lo[6], lo[7]);
                H.x = g2_pack(hi[0], hi[1]); H.y = g2_pack(hi[2], hi[3]);
                H.z = g2_pack(hi[4], hi[5]); H.w = g2_pack(hi[6], hi[7]);
                *(uint4 *)(dst + blk * 32 + j) = L;
                *(uint4 *)(dst + blk * 32 + 16 + j) = H;
            }
        }
    }
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) { return (ulong64)(n_in / 32) * 18; }
};

// BN is the block's column tile: 64, or 128 on the compute_80 module. Each
// warp owns 32 rows by BN / 2 columns, so a wider tile adds n-tiles to every
// warp and dequantizes each weight once for twice the columns; an output's
// k-steps, their order and the instruction are the same at either width, so
// the outputs are too.
template <class F, int BN = G2_BN>
__device__ __forceinline__ void gemm2_body(const uchar *wb, const __half *xh, float *y,
                                           const mv_args &a, const float *bias,
                                           const float *xsc) {
    enum { NT = BN / 16, XR = BN / 16 };                // n8-tiles per warp, act loads per thread
    extern __shared__ __align__(16) unsigned char g2_sm[];
    __half *sA = (__half *)g2_sm;                       // [2][G2_BM][G2_LD]
    __half *sB = sA + 2 * G2_BM * G2_LD;                // [2][BN][G2_LD]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const int row0 = blockIdx.x * G2_BM, col0 = blockIdx.y * BN;
    const int nst = a.n_in / G2_BK;
    const ulong64 rb = F::row_bytes(a.n_in);
    // staging roles: threads 0..127 own one weight row each, 128..255 BN / 16
    // 16-byte pieces of the activation tile (BN columns x 64 halves)
    const bool wrole = tid < G2_BM;
    const int wr = row0 + tid;
    const bool wvalid = wrole && wr < a.n_out;
    const uchar *wrow = wb + a.w_off + (ulong64)(wvalid ? wr : 0) * rb;
    typename F::raw rw;
    uint4 xr[XR];

    auto load = [&](int s) {
        const int ks = s * G2_BK;
        if (wrole) {
            if (wvalid) F::load(rw, wrow, ks, a.n_in);
        } else {
            #pragma unroll
            for (int j = 0; j < XR; j++) {
                int idx = (tid - G2_BM) + j * 128, col = idx >> 3, part = idx & 7;
                int c = col0 + col;
                xr[j] = c < a.batch ? *(const uint4 *)(xh + (ulong64)c * a.n_in + ks + part * 8)
                                    : make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };
    auto store = [&](int s, int buf) {
        const int ks = s * G2_BK;
        if (wrole) {
            __half *d = sA + buf * G2_BM * G2_LD + tid * G2_LD;
            if (wvalid) F::store(d, rw, ks);
            else {
                #pragma unroll
                for (int j = 0; j < 8; j++) *(uint4 *)(d + j * 8) = make_uint4(0u, 0u, 0u, 0u);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < XR; j++) {
                int idx = (tid - G2_BM) + j * 128, col = idx >> 3, part = idx & 7;
                *(uint4 *)(sB + buf * BN * G2_LD + col * G2_LD + part * 8) = xr[j];
            }
        }
    };

    float acc[2][NT][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < NT; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    load(0);
    store(0, 0);
    __syncthreads();
    for (int s = 0; s < nst; s++) {
        const int cur = s & 1;
        if (s + 1 < nst) load(s + 1);
        // the other buffer is free: every warp passed the barrier that ended
        // the stage which last read it
        if (F::kDirect && s + 1 < nst) store(s + 1, cur ^ 1);
        const __half *A = sA + cur * G2_BM * G2_LD;
        const __half *B = sB + cur * BN * G2_LD;
        #pragma unroll
        for (int kk = 0; kk < G2_BK; kk += 16) {
            unsigned af[2][4], bf[NT / 2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                g2_ldsm_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                           A + (wm * 32 + mt * 16 + (lane & 15)) * G2_LD + kk + (lane >> 4) * 8);
            #pragma unroll
            for (int np = 0; np < NT / 2; np++)
                g2_ldsm_x4(bf[np][0], bf[np][1], bf[np][2], bf[np][3],
                           B + (wn * (BN / 2) + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * G2_LD
                             + kk + ((lane >> 3) & 1) * 8);
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                #pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const unsigned *b = bf[nt >> 1] + (nt & 1) * 2;
#ifdef G2_MMA_K16
                    g2_mma16(acc[mt][nt], af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                             b[0], b[1]);
#else
                    g2_mma(acc[mt][nt], af[mt][0], af[mt][1], b[0]);
                    g2_mma(acc[mt][nt], af[mt][2], af[mt][3], b[1]);
#endif
                }
        }
        if (!F::kDirect && s + 1 < nst) store(s + 1, cur ^ 1);
        __syncthreads();
    }
    // epilogue: c0,c1 at (row g, cols 2t, 2t+1), c2,c3 at row g + 8
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < NT; nt++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = row0 + wm * 32 + mt * 16 + (lane >> 2) + (e >> 1) * 8;
                int c = col0 + wn * (BN / 2) + nt * 8 + (lane & 3) * 2 + (e & 1);
                if (r < a.n_out && c < a.batch) {
                    float v = acc[mt][nt][e] * (xsc[c] * (1.0f / 16384.0f));
                    y[(ulong64)c * a.ys + r] = a.has_bias ? v + bias[r] : v;
                }
            }
}

#endif // RUNNER_KERNELS_TC2_CUH
