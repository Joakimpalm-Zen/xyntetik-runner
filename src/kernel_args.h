// Kernel argument structs, shared by the HOST and the DEVICE.
//
// These are passed to CUDA kernels BY VALUE, so gcc (compiling src/cuda.c) and
// nvcc (compiling src/kernels.cu to PTX) must agree on their layout exactly.
// They used to be declared twice — once in each file — with nothing checking
// that the two stayed in step.
//
// That is a silent failure: adding a field to one side compiles cleanly on
// both, and the kernel then reads its arguments at the wrong offsets. It
// produces fluent, confident, WRONG output rather than a crash — on 2026-08-08
// exactly that happened while adding a field to attn_args, and the GPU emitted
// "The capital of France isnas!!!!!!!!" while the CPU path stayed correct.
// Without a CPU arm to compare against, that reads like a broken model file.
//
// One definition removes the failure mode entirely.
//
// Keep this header free of anything CUDA-specific: it is compiled as C by gcc
// and as C++ by nvcc, so plain structs and fixed-width integers only.
#ifndef RUNNER_KERNEL_ARGS_H
#define RUNNER_KERNEL_ARGS_H

// `unsigned long long` rather than uint64_t: nvcc compiles kernels.cu without
// the host's <stdint.h> in scope, and the two spellings are the same 8-byte
// type with the same alignment on every platform this builds for.
typedef unsigned long long ka_u64;

typedef struct {
    int    n_in;
    int    n_out;
    ka_u64 w_off;    // tensor byte offset inside the weight buffer
    int    has_bias;
    int    batch;    // 1..MVB token columns (k_mv_* ignores it)
    int    xs, ys;   // element stride between x / y columns
} mv_args;

typedef struct {
    int    n_in;
    int    n_out;
    ka_u64 w_off;      // fused expert tensor base byte offset
    ka_u64 estride;    // bytes per expert block within the fused tensor
    int    xs, ys;     // x / y element stride between expert slots
} moe_args;

typedef struct {
    int   head_dim, n_heads, half_dim, neox;
    float mscale;
} rope_args;

typedef struct {
    int    head_dim, n_head, n_head_kv, n_ctx;
    ka_u64 l_off;    // this layer's BYTE offset into the K cache
    float  scale;
    int    qs, os;   // q / out element stride per token column
    int    window;   // sliding-window size for this layer (0 = full)
    int    q8;       // K cache KIND: 0 fp16 rows, 1 q8_0 blocks, 2 fp4 blocks
    int    ring;     // rows this layer owns when it recycles them (0 = flat
                     // n_ctx rows indexed by absolute position). A sliding
                     // layer reads at most `window` positions back, so it
                     // needs only window + batch rows; every KV address in
                     // the attention kernels goes through kv_slot() to honour
                     // it. Host and device MUST agree here -- that is what
                     // this header exists for.
    ka_u64 v_off;    // this layer's BYTE offset into the V cache (l_off is K)
    int    vq8;      // V cache KIND; differs from q8 under --kv k8v4
} attn_args;

// One offloaded layer's entry in the decode KV gather table (k_kv_gather,
// cuda.c kv_gather_*): where its K and V rows live, how long they are, its
// ring length (0: flat) and where they land in the staging buffer.
typedef struct {
    ka_u64   koff, voff;   // byte offsets of the layer's K and V regions
    unsigned krow, vrow;   // row bytes of each
    int      ring;         // rows a ring layer recycles, 0 for a flat layer
    unsigned stage;        // byte offset of the layer's K row in the staging buffer (V follows)
} kvg_entry;

// Rows a warp takes in the width-classed decode twins (k_gemvb_q4_K / q5_K /
// q3_K / q6_K / iq4_xs / q8_0 / q4_0 / bf16 _x4, _x8, _x16): every row of a warp
// reuses the staged x it reads. Kernels and launches (cuda.c enc_mv_batch) both
// read these, so the grid always covers what the kernels compute.
#define KQ_TWR(nc) ((nc) <= 8 ? 4 : 2)   // Q4_K, Q5_K, Q3_K, IQ4_XS
#define Q6_TWR(nc) ((nc) <= 4 ? 4 : 2)   // Q6_K
#define DQ_TWR(nc) ((nc) <= 8 ? 4 : 2)   // Q8_0, Q4_0, BF16

// k_q35_delta (the one-token Gated DeltaNet step): threads a block, one value
// column each per pass, and the key rows its shared tile stages at a time.
// The launch (cuda.c gpu_q35_recurrent) uses Q35D_COLS as its block size.
#define Q35D_COLS 128
#define Q35D_KEYS 32
// k_q35_delta1 (the one-token step with a head's columns over several
// blocks): value columns a block, and the widest state it takes (its shared
// tile holds Q35D1_COLS rows of that width; cuda.c checks both)
#define Q35D1_COLS 32
#define Q35D1_MAX_SD 128

// The Gated DeltaNet's q/k L2 norm epsilon: x / sqrt(sum x^2 + 1e-6), the
// publisher's own (transformers' l2norm, eps=1e-6 hard-coded), not the
// model's RMS epsilon, which only the RMS norms take. Every qwen35 and Clef
// file known ships rms_norm_eps 1e-6, so the two were equal there; a file
// with another value read its q and k at the wrong scale (R8.18). The CPU
// forward, the backward's recompute and the CUDA steps all take it here.
#define DN_L2_EPS 1e-6f

#endif // RUNNER_KERNEL_ARGS_H
