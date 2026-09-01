// Metal 4 / M5 tensor-op prefill tracer bullet. Compiled as a separate
// library so an unavailable MPP compiler can only remove this optional rung.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

struct mm_args {
    int n_in, n_out, n_col;
    ulong w_off;
    int has_bias, x_stride, y_stride;
};

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

// out[col][row] = sum_k x[col][k] * W[row][k]. Each type's dequantization
// is intentionally identical to its k_mm_* twin; only the matrix primitive
// changes. DEQ16 fills dstw[0..15] with weights kg..kg+15 of row
// (row0+row); every other line of the kernel is shared verbatim.
#define TENSOR_KERNEL(NAME, DEQ16) \
kernel void NAME( \
        device const uchar *wb   [[buffer(0)]], \
        device float       *x    [[buffer(1)]], \
        device float       *y    [[buffer(2)]], \
        constant mm_args   &a    [[buffer(3)]], \
        device const float *bias [[buffer(4)]], \
        threadgroup half *shmem  [[threadgroup(0)]], \
        uint2 tgpig [[threadgroup_position_in_grid]], \
        ushort tid [[thread_index_in_threadgroup]]) { \
    constexpr int TM = 128, TN = 256, TK = 64, NT = 128; \
    const int row0 = (int)tgpig.x * TM; \
    const int col0 = (int)tgpig.y * TN; \
    threadgroup half *tw = shmem; \
 \
    auto w_full = tensor(tw, dextents<int32_t, 2>(TK, TM)); \
    auto x_full = tensor(x, dextents<int32_t, 2>(a.n_in, a.n_col), \
                         array<int, 2>({1, a.x_stride})); \
    mpp::tensor_ops::matmul2d< \
        mpp::tensor_ops::matmul2d_descriptor( \
            TN, TM, static_cast<int>(dynamic_extent), false, true, true, \
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), \
        execution_simdgroups<4>> mm; \
    auto out = mm.get_destination_cooperative_tensor<decltype(x_full), \
                                                       decltype(w_full), float>(); \
 \
    for (int k0 = 0; k0 < a.n_in; k0 += TK) { \
        for (int work = tid; work < TM * (TK / 16); work += NT) { \
            int row = work / (TK / 16), chunk = work % (TK / 16); \
            int kg = k0 + chunk * 16; \
            threadgroup half *dstw = tw + row * TK + chunk * 16; \
            if (row0 + row < a.n_out) { \
                DEQ16 \
            } else { \
                for (int j = 0; j < 16; j++) dstw[j] = 0.0h; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        auto w_tile = tensor(tw, dextents<int32_t, 2>(TK, TM)); \
        auto x_tile = tensor(x + k0 + (ulong)col0 * a.x_stride, \
                             dextents<int32_t, 2>(TK, a.n_col - col0), \
                             array<int, 2>({1, a.x_stride})); \
        mm.run(x_tile, w_tile, out); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
 \
    auto dst = tensor(y, dextents<int32_t, 2>(a.y_stride, a.n_col), \
                      array<int, 2>({1, a.y_stride})); \
    out.store(dst.slice(row0, col0)); \
    threadgroup_barrier(mem_flags::mem_device); \
    if (a.has_bias) { \
        for (int i = tid; i < TM * TN; i += NT) { \
            int col = i / TM, row = i % TM; \
            if (col0 + col < a.n_col && row0 + row < a.n_out) \
                y[(ulong)(col0 + col) * a.y_stride + row0 + row] += bias[row0 + row]; \
        } \
    } \
}


#define TENSOR_DEQ_Q4_K \
                int nsb = a.n_in / 256; \
                int sb = kg / 256, j32 = (kg % 256) / 32; \
                device const uchar *blk = wb + a.w_off \
                    + ((ulong)(row0 + row) * nsb + sb) * 144; \
                float dall = (float)*(device const half *)blk; \
                float dmin = (float)*(device const half *)(blk + 2); \
                uchar scale, minv; \
                get_scale_min_k4(j32, blk + 4, &scale, &minv); \
                device const uchar *q = blk + 16 + (j32 / 2) * 32; \
                int lane0 = kg & 31; \
                for (int j = 0; j < 16; j++) { \
                    int quant = (j32 & 1) ? (int)(q[lane0 + j] >> 4) \
                                          : (int)(q[lane0 + j] & 0xF); \
                    dstw[j] = (half)(dall * (float)scale * (float)quant \
                                   - dmin * (float)minv); \
                }

#define TENSOR_DEQ_Q8_0 \
                int nb = a.n_in / 32; \
                int bi = kg / 32, off = kg % 32; \
                device const uchar *blk = wb + a.w_off \
                    + ((ulong)(row0 + row) * nb + bi) * 34; \
                float d = (float)*(device const half *)blk; \
                device const char *q = (device const char *)(blk + 2) + off; \
                for (int j = 0; j < 16; j++) \
                    dstw[j] = (half)(d * (float)q[j]);

#define TENSOR_DEQ_Q4_0 \
                int nb = a.n_in / 32; \
                int bi = kg / 32, off = kg % 32; \
                device const uchar *blk = wb + a.w_off \
                    + ((ulong)(row0 + row) * nb + bi) * 18; \
                float d = (float)*(device const half *)blk; \
                device const uchar *q = blk + 2; \
                for (int j = 0; j < 16; j++) { \
                    int v = off ? (int)(q[j] >> 4) : (int)(q[j] & 0xF); \
                    dstw[j] = (half)(d * (float)(v - 8)); \
                }

TENSOR_KERNEL(k_tensor_q4_K, TENSOR_DEQ_Q4_K)
TENSOR_KERNEL(k_tensor_q8_0, TENSOR_DEQ_Q8_0)
TENSOR_KERNEL(k_tensor_q4_0, TENSOR_DEQ_Q4_0)
