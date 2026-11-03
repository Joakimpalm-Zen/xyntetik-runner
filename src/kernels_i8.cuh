// The int8 tensor-core prefill GEMM (R3.14), compute_80 module only: the
// weights' own integer quants against int8 activations, mma.sync m16n8k32 (or
// m16n8k16 where a type's scale changes every 16 elements), int32 sums per
// 32-element block folded into fp32 with the block's scales. llama.cpp's
// arithmetic class (its MMQ), not the f32-faithful TC route's: the
// activations are rounded to int8 per 32-element block (amax / 127), so the
// route is a tolerance route, opt-in (RUNNER_CUDA_I8=1) until a type and
// architecture pass the tolerance and kill gates.
//
// A weight value is ws * q - wm, q the type's integer quant and (ws, wm) its
// 32- or 16-element block's scale and min; an activation is dx * xq with
// xs = dx * sum(xq) over the 32-element block. One 32-block of a row against
// one of a column contributes ws * dx * dot(q, xq) - wm * xs: the integer dot
// on the tensor cores, the rest in fp32.
//
// Tiles as TC2's (kernels_tc2.cuh): 128 rows by 64 columns a block of 256
// threads, 64-deep stages double-buffered; threads 0..127 stage one weight
// row each through the TC2 loader's raw words, 128..255 the activation tile
// and its scales. Each warp owns 32 rows by 32 columns. Includers define
// uchar, uint and ulong64 and include kernel_args.h and kernels_tc2.cuh first.
#ifndef RUNNER_KERNELS_I8_CUH
#define RUNNER_KERNELS_I8_CUH

#define I8_BM 128
#define I8_BN 64
#define I8_BK 64
#define I8_LD (I8_BK + 16)          // bytes a staged row: 80, 16-byte aligned for ldmatrix
#define I8_NS 4                     // scale slots a 64-deep stage holds (16-element blocks)
// dynamic shared memory: A and B int8 tiles, then the weight scales and mins
// (I8_NS a row) and the activation scales and sums (2 a column), each double
// buffered
#define I8_RW 17                    // raw words a row's stage the cooperative copy holds (Q8_0)
#define I8_SMEM (2 * I8_BM * I8_LD + 2 * I8_BN * I8_LD +                     \
                 (2 * I8_BM * I8_NS + 2 * I8_BM * I8_NS + 2 * I8_BN * 2 * 2 + \
                  I8_BM * I8_RW) * 4)

static __device__ __forceinline__ void i8_ldsm_x4(unsigned &r0, unsigned &r1, unsigned &r2,
                                                  unsigned &r3, const signed char *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// c += a . b over 32 k: A rows g, g + 8 at k 0-15 (a0, a1) and 16-31 (a2, a3)
static __device__ __forceinline__ void i8_mma32(int *c, unsigned a0, unsigned a1, unsigned a2,
                                                unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// c += a . b over 16 k: A rows g, g + 8 (a0, a1)
static __device__ __forceinline__ void i8_mma16(int *c, unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// The activation side: per column and 32-element block, xq = rn(x / dx) with
// dx = amax / 127, dx itself, and xs = dx * sum(xq). One warp a block.
extern "C" __global__ void k_quant_i8(const float *x, signed char *xq, float *xd, float *xsum,
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
    int s = q;
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    xq[(ulong64)c * n + bk * 32 + lane] = (signed char)q;
    if (lane == 0) {
        xd[(ulong64)c * nbk + bk] = d;
        xsum[(ulong64)c * nbk + bk] = d * (float)s;
    }
}

// The int8 stagers: from a TC2 loader's raw words for the 64-element segment
// at ks, the 64 integer quants (dst), the I8_NS scale slots (ws, one a
// 16-element block: a 32-element type repeats its scale in both) and the
// mins (wm, the same slots). G is the type's block: 32 (one m16n8k32 a
// block) or 16 (two m16n8k16).

// Q4_K: the low nibbles are the segment's first 32 quants, the high its last
// 32, each half with its 6-bit scale and min: ws = d * sc, wm = dmin * m.
struct I8Q4K {
    typedef G2Q4K::raw raw;
    // kRW 0: the per-row load below; a cooperative copy measured slower for the
    // K-quants (their loaders already move a row's segment in a few wide loads)
    enum { G = 32, kMins = 1, kRW = 0 };
    static __device__ __forceinline__ uint word(const uchar *, int, int) { return 0; }
    static __device__ __forceinline__ void load_smem(raw &, const uint *) {}
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks, int n_in) {
        G2Q4K::load(r, rowp, ks, n_in);
    }
    static __device__ __forceinline__ void store(signed char *dst, float *ws, float *wm,
                                                 const raw &r, int ks) {
        const int seg = (ks >> 6) & 3;
        unsigned short hd = (unsigned short)(r.ddm & 0xFFFF), hm = (unsigned short)(r.ddm >> 16);
        float dd = __half2float(*(const __half *)&hd), dmin = __half2float(*(const __half *)&hm);
        uint sg0, mg0, sg1, mg1;
        g2_scale_min_k4(2 * seg + 0, r.s0, r.s1, r.s2, &sg0, &mg0);
        g2_scale_min_k4(2 * seg + 1, r.s0, r.s1, r.s2, &sg1, &mg1);
        uint w[8] = { r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w };
        uint lo[8], hi[8];
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            lo[v] = w[v] & 0x0F0F0F0Fu;
            hi[v] = (w[v] >> 4) & 0x0F0F0F0Fu;
        }
        *(uint4 *)(dst + 0)  = make_uint4(lo[0], lo[1], lo[2], lo[3]);
        *(uint4 *)(dst + 16) = make_uint4(lo[4], lo[5], lo[6], lo[7]);
        *(uint4 *)(dst + 32) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        *(uint4 *)(dst + 48) = make_uint4(hi[4], hi[5], hi[6], hi[7]);
        float s0 = dd * (float)sg0, m0 = dmin * (float)mg0;
        float s1 = dd * (float)sg1, m1 = dmin * (float)mg1;
        ws[0] = s0; ws[1] = s0; ws[2] = s1; ws[3] = s1;
        wm[0] = m0; wm[1] = m0; wm[2] = m1; wm[3] = m1;
    }
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) { return G2Q4K::row_bytes(n_in); }
};

// Q8_0: two 34-byte blocks, ws = d, no min.
struct I8Q80 {
    typedef G2Q80::raw raw;
    enum { G = 32, kMins = 0, kRW = 17 };
    // the stage's 68 bytes, 4-byte aligned (a row is a whole number of 68-byte pairs)
    static __device__ __forceinline__ uint word(const uchar *rowp, int ks, int w) {
        return ((const uint *)(rowp + (ulong64)(ks >> 5) * 34))[w];
    }
    static __device__ __forceinline__ void load_smem(raw &r, const uint *slot) {
        #pragma unroll
        for (int i = 0; i < 17; i++) r.w[i] = slot[i];
    }
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks, int n_in) {
        G2Q80::load(r, rowp, ks, n_in);
    }
    static __device__ __forceinline__ void store(signed char *dst, float *ws, float *wm,
                                                 const raw &r, int ks) {
        (void)ks;
        #pragma unroll
        for (int blk = 0; blk < 2; blk++) {
            const int o = blk * 34;
            unsigned short hd = (unsigned short)(G2B(r.w, o) | (G2B(r.w, o + 1) << 8));
            float d = __half2float(*(const __half *)&hd);
            uint q[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                q[j] = G2B(r.w, o + 2 + 4 * j) | (G2B(r.w, o + 3 + 4 * j) << 8) |
                       (G2B(r.w, o + 4 + 4 * j) << 16) | (G2B(r.w, o + 5 + 4 * j) << 24);
            *(uint4 *)(dst + blk * 32)      = make_uint4(q[0], q[1], q[2], q[3]);
            *(uint4 *)(dst + blk * 32 + 16) = make_uint4(q[4], q[5], q[6], q[7]);
            ws[2 * blk] = d; ws[2 * blk + 1] = d;
            wm[2 * blk] = 0.0f; wm[2 * blk + 1] = 0.0f;
        }
    }
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) { return G2Q80::row_bytes(n_in); }
};

// Q6_K: the segment's quants in G2Q6K's order (its "a" half the first 32, its
// "b" half the last 32), q - 32 in [-32, 31], each 16 with its signed scale:
// ws = d * sc, no min.
struct I8Q6K {
    typedef G2Q6K::raw raw;
    enum { G = 16, kMins = 0, kRW = 0 };              // per-row load, as I8Q4K
    static __device__ __forceinline__ uint word(const uchar *, int, int) { return 0; }
    static __device__ __forceinline__ void load_smem(raw &, const uint *) {}
    static __device__ __forceinline__ void load(raw &r, const uchar *rowp, int ks, int n_in) {
        G2Q6K::load(r, rowp, ks, n_in);
    }
    static __device__ __forceinline__ void store(signed char *dst, float *ws, float *wm,
                                                 const raw &r, int ks) {
        const int sseg = (ks >> 6) & 1;
        const int shA = sseg * 4 + 0, shB = sseg * 4 + 2;
        const uint scw = sseg ? r.sc[1] : r.sc[0];
        float d = __half2float(*(const __half *)&r.d);
        #pragma unroll
        for (int l4 = 0; l4 < 32; l4 += 4) {
            uint pa = 0, pb = 0;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const int l = l4 + u;
                uint a0 = (r.ql[l >> 2] >> (8 * (l & 3))) & 0xFF;
                uint b0 = (r.ql[(l + 32) >> 2] >> (8 * (l & 3))) & 0xFF;
                uint h = (r.qh[l >> 2] >> (8 * (l & 3))) & 0xFF;
                int qa = (int)((sseg ? (a0 >> 4) : (a0 & 0xF)) | (((h >> shA) & 3) << 4)) - 32;
                int qb = (int)((sseg ? (b0 >> 4) : (b0 & 0xF)) | (((h >> shB) & 3) << 4)) - 32;
                pa |= ((uint)qa & 0xFFu) << (8 * u);
                pb |= ((uint)qb & 0xFFu) << (8 * u);
            }
            *(uint *)(dst + l4) = pa;
            *(uint *)(dst + 32 + l4) = pb;
        }
        // slots: a half's first and second 16 (scales is 0, 1), then b's
        #pragma unroll
        for (int is = 0; is < 2; is++) {
            ws[is]     = d * (float)(int)(signed char)((scw >> (8 * is)) & 0xFF);
            ws[2 + is] = d * (float)(int)(signed char)((scw >> (8 * (2 + is))) & 0xFF);
        }
        wm[0] = wm[1] = wm[2] = wm[3] = 0.0f;
    }
    static __device__ __forceinline__ ulong64 row_bytes(int n_in) { return G2Q6K::row_bytes(n_in); }
};

template <class F>
__device__ __forceinline__ void gemm_i8_body(const uchar *wb, const signed char *xq,
                                             const float *xd, const float *xsum, float *y,
                                             const mv_args &a, const float *bias) {
    enum { NT = I8_BN / 16 };                           // n8-tiles a warp: 4
    extern __shared__ __align__(16) unsigned char i8_sm[];
    signed char *sA = (signed char *)i8_sm;             // [2][BM][LD]
    signed char *sB = sA + 2 * I8_BM * I8_LD;           // [2][BN][LD]
    float *sWs = (float *)(sB + 2 * I8_BN * I8_LD);     // [2][BM][NS]
    float *sWm = sWs + 2 * I8_BM * I8_NS;               // [2][BM][NS]
    float *sXd = sWm + 2 * I8_BM * I8_NS;               // [2][BN][2]
    float *sXs = sXd + 2 * I8_BN * 2;                   // [2][BN][2]
    uint *sR = (uint *)(sXs + 2 * I8_BN * 2);           // [BM][kRW] the next stage's raw words
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm_ = warp & 3, wn = warp >> 2;
    const int row0 = blockIdx.x * I8_BM, col0 = blockIdx.y * I8_BN;
    const int nst = a.n_in / I8_BK, nbk = a.n_in / 32;
    const ulong64 rb = F::row_bytes(a.n_in);
    const bool wrole = tid < I8_BM;
    const int wr = row0 + tid;
    const bool wvalid = wrole && wr < a.n_out;
    const uchar *wrow = wb + a.w_off + (ulong64)(wvalid ? wr : 0) * rb;
    // activation staging: thread tid - 128 moves two 16-byte pieces of the
    // 64 x 64-byte tile and one column's two block scales and sums
    const int at = tid - I8_BM;
    typename F::raw rw;
    // cooperative copy: consecutive threads take consecutive words of a row,
    // so a warp's load covers two or three rows' contiguous bytes
    enum { RW = F::kRW ? F::kRW : 1, NW = (I8_BM * RW + 255) / 256 };
    uint cw[NW];
    uint4 xr[2];
    float xd2[2], xs2[2];
    auto load = [&](int s) {
        const int ks = s * I8_BK;
        if (F::kRW) {
            #pragma unroll
            for (int j = 0; j < NW; j++) {
                const int k = tid + j * 256, row = k / RW, w = k - row * RW, r = row0 + row;
                cw[j] = (k < I8_BM * RW && r < a.n_out)
                      ? F::word(wb + a.w_off + (ulong64)r * rb, ks, w) : 0u;
            }
        }
        if (wrole) {
            if (!F::kRW && wvalid) F::load(rw, wrow, ks, a.n_in);
        } else {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int idx = at + j * 128, col = idx >> 2, part = idx & 3;
                int c = col0 + col;
                xr[j] = c < a.batch ? *(const uint4 *)(xq + (ulong64)c * a.n_in + ks + part * 16)
                                    : make_uint4(0u, 0u, 0u, 0u);
            }
            if (at < I8_BN) {
                int c = col0 + at;
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    xd2[h] = c < a.batch ? xd[(ulong64)c * nbk + (ks >> 5) + h] : 0.0f;
                    xs2[h] = c < a.batch ? xsum[(ulong64)c * nbk + (ks >> 5) + h] : 0.0f;
                }
            }
        }
    };
    auto repack = [&](int s, int buf) {                 // kRW: sR to the int8 tile
        const int ks = s * I8_BK;
        if (wrole) {
            signed char *d = sA + buf * I8_BM * I8_LD + tid * I8_LD;
            float *ws = sWs + (buf * I8_BM + tid) * I8_NS;
            float *wmn = sWm + (buf * I8_BM + tid) * I8_NS;
            typename F::raw r;
            F::load_smem(r, sR + tid * RW);
            F::store(d, ws, wmn, r, ks);
        }
    };
    auto store = [&](int s, int buf) {
        const int ks = s * I8_BK;
        if (F::kRW) {
            #pragma unroll
            for (int j = 0; j < NW; j++) {
                const int k = tid + j * 256;
                if (k < I8_BM * RW) sR[k] = cw[j];
            }
        }
        if (wrole) {
            if (F::kRW) return;
            signed char *d = sA + buf * I8_BM * I8_LD + tid * I8_LD;
            float *ws = sWs + (buf * I8_BM + tid) * I8_NS;
            float *wmn = sWm + (buf * I8_BM + tid) * I8_NS;
            if (wvalid) F::store(d, ws, wmn, rw, ks);
            else {
                #pragma unroll
                for (int j = 0; j < 4; j++) *(uint4 *)(d + j * 16) = make_uint4(0u, 0u, 0u, 0u);
                #pragma unroll
                for (int j = 0; j < I8_NS; j++) { ws[j] = 0.0f; wmn[j] = 0.0f; }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int idx = at + j * 128, col = idx >> 2, part = idx & 3;
                *(uint4 *)(sB + buf * I8_BN * I8_LD + col * I8_LD + part * 16) = xr[j];
            }
            if (at < I8_BN) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    sXd[(buf * I8_BN + at) * 2 + h] = xd2[h];
                    sXs[(buf * I8_BN + at) * 2 + h] = xs2[h];
                }
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
    // this thread's output rows (local to the block) and columns
    int lr[2][2], lc[NT][2];
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int h = 0; h < 2; h++) lr[mt][h] = wm_ * 32 + mt * 16 + (lane >> 2) + h * 8;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++)
        #pragma unroll
        for (int e = 0; e < 2; e++) lc[nt][e] = wn * 32 + nt * 8 + (lane & 3) * 2 + e;
    load(0);
    store(0, 0);
    if (F::kRW) { __syncthreads(); repack(0, 0); }
    __syncthreads();
    for (int s = 0; s < nst; s++) {
        const int cur = s & 1;
        if (s + 1 < nst) load(s + 1);
        const signed char *A = sA + cur * I8_BM * I8_LD;
        const signed char *B = sB + cur * I8_BN * I8_LD;
        const float *Ws = sWs + cur * I8_BM * I8_NS, *Wm = sWm + cur * I8_BM * I8_NS;
        const float *Xd = sXd + cur * I8_BN * 2, *Xs = sXs + cur * I8_BN * 2;
        #pragma unroll
        for (int kb = 0; kb < 2; kb++) {                 // the stage's two 32-blocks
            unsigned af[2][4], bf[NT / 2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                i8_ldsm_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                           A + (wm_ * 32 + mt * 16 + (lane & 15)) * I8_LD + kb * 32 + (lane >> 4) * 16);
            #pragma unroll
            for (int np = 0; np < NT / 2; np++)
                i8_ldsm_x4(bf[np][0], bf[np][1], bf[np][2], bf[np][3],
                           B + (wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * I8_LD
                             + kb * 32 + ((lane >> 3) & 1) * 16);
            // this thread's block scales, read from shared memory once a 32-block
            float wsr[2][2][2], wmr[2][2], xdr[NT][2], xsr[NT][2];
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int r = lr[mt][h];
                    wsr[mt][h][0] = Ws[r * I8_NS + 2 * kb];
                    wsr[mt][h][1] = Ws[r * I8_NS + 2 * kb + 1];
                    wmr[mt][h] = F::kMins ? Wm[r * I8_NS + 2 * kb] : 0.0f;
                }
            #pragma unroll
            for (int nt = 0; nt < NT; nt++)
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int cc = lc[nt][e];
                    xdr[nt][e] = Xd[cc * 2 + kb];
                    xsr[nt][e] = F::kMins ? Xs[cc * 2 + kb] : 0.0f;
                }
            #pragma unroll
            for (int mt = 0; mt < 2; mt++)
                #pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const unsigned *b = bf[nt >> 1] + (nt & 1) * 2;
                    if (F::G == 32) {
                        int c[4] = { 0, 0, 0, 0 };
                        i8_mma32(c, af[mt][0], af[mt][1], af[mt][2], af[mt][3], b[0], b[1]);
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float v = wsr[mt][e >> 1][0] * xdr[nt][e & 1] * (float)c[e];
                            if (F::kMins) v -= wmr[mt][e >> 1] * xsr[nt][e & 1];
                            acc[mt][nt][e] += v;
                        }
                    } else {
                        int c0[4] = { 0, 0, 0, 0 }, c1[4] = { 0, 0, 0, 0 };
                        i8_mma16(c0, af[mt][0], af[mt][1], b[0]);
                        i8_mma16(c1, af[mt][2], af[mt][3], b[1]);
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float sc = wsr[mt][e >> 1][0] * (float)c0[e] +
                                       wsr[mt][e >> 1][1] * (float)c1[e];
                            acc[mt][nt][e] += xdr[nt][e & 1] * sc;
                        }
                    }
                }
        }
        if (s + 1 < nst) store(s + 1, cur ^ 1);
        if (F::kRW && s + 1 < nst) { __syncthreads(); repack(s + 1, cur ^ 1); }
        __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < NT; nt++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int r = row0 + lr[mt][e >> 1];
                int c = col0 + lc[nt][e & 1];
                if (r < a.n_out && c < a.batch) {
                    float v = acc[mt][nt][e];
                    y[(ulong64)c * a.ys + r] = a.has_bias ? v + bias[r] : v;
                }
            }
}

#define GEMM_I8(NAME, F)                                                            \
extern "C" __global__ void __launch_bounds__(256)                                   \
NAME(const uchar *wb, const signed char *xq, const float *xd, const float *xsum,    \
     float *y, mv_args a, const float *bias) {                                      \
    gemm_i8_body<F>(wb, xq, xd, xsum, y, a, bias);                                  \
}

#endif // RUNNER_KERNELS_I8_CUH
