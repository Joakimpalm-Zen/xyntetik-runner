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

#endif // RUNNER_TDN_ARGS_H
