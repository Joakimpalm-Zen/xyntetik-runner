#ifndef RUNNER_TDN_ARGS_H
#define RUNNER_TDN_ARGS_H

// The Gated DeltaNet recurrence of a training step on the device
// (kernels_train.cuh k_tdn_fw / k_tdn_bw, cuda.c gpu_train_dn_*): the state
// rows a warp owns (a head's width must be a multiple), the widest state
// taken (a lane holds up to TDN_MAX_HV / 32 columns of a row) and the
// forward's warps a block. The reverse sweep runs one block a head of
// hv / TDN_ROWS warps.
#define TDN_ROWS 8
#define TDN_MAX_HV 128
#define TDN_FW_WARPS 4

// A layer's shape for those kernels: T positions, a checkpoint every C, nh
// value heads over ng key groups of sk keys, hv values a head (hv == sk), cv
// rows convdim apart (q at 0, k at keydim, v at 2 keydim), y rows inner
// apart. Its own header because gpu.h takes it and metal.m, which includes
// gpu.h, keeps its own copies of kernel_args.h's structs.
typedef struct {
    int T, C, nh, ng, sk, hv, convdim, keydim, inner;
} tdn_args;

// The attention backward of a training step on the device (kernels_train.cuh
// k_tab_*, cuda.c gpu_train_attn_bw): positions a block of the sweep takes
// (model.c's ATB), the widest head and the threads a kernel block runs.
#define TAB_TB 64
#define TAB_MAX_HD 512
#define TAB_THREADS 256

// A layer's shape for them: T positions, n_head query heads in groups of
// kv_mul a kv head, hd values a head, q / dao rows q_dim apart, dk / dv rows
// kv_dim apart, K and V cache rows k_row_b / v_row_b bytes apart (f16), the
// score scale and the sliding window (0: none).
typedef struct {
    int T, n_head, kv_mul, hd, q_dim, kv_dim, swa;
    unsigned long long k_row_b, v_row_b;
    float scale;
} tab_args;

#endif // RUNNER_TDN_ARGS_H
