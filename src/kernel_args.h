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
//
// Keep kernel arguments small and read whole: plain pointers, ints and floats,
// and structs like these, whose every field the kernel reads. A 336-byte
// struct read only in part (a field indexed at run time), passed by value and
// again read through a pointer, made nvcc 13 annotate its loads with
// `.pragma "used_bytes_mask N"`, which needs PTX ISA 8.3 (R6.90).
// scripts/embed-ptx.py pins the embedded string to .version 7.8, so the
// driver's JIT rejected the whole module, and every kernel in it was gone,
// while ptxas took the raw 9.0 PTX. make ptx now assembles the embedded
// strings with the pinned ptxas (scripts/check-ptx-isa.py). After a kernel's
// arguments change, also check on a CUDA host that the embedded string loads:
// a gate whose device is up and whose module did not load fails with the
// driver's log (tests/gate_gpu.h).
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

// Rows a warp takes in the width-classed decode twins (k_gemvb_q3_K / iq4_xs /
// q8_0 / q4_0 / bf16 _x4, _x8, _x16): every row of a warp reuses the staged x
// it reads. Kernels and launches (cuda.c enc_mv_batch) both read these, so the
// grid always covers what the kernels compute.
#define KQ_TWR(nc) ((nc) <= 8 ? 4 : 2)   // Q3_K, IQ4_XS
#define DQ_TWR(nc) ((nc) <= 8 ? 4 : 2)   // Q8_0, Q4_0, BF16
// Q4_K, Q5_K and Q6_K (GEMVB_KQ, GEMVB_Q6_DB; R3.3.4): rows a warp and warps
// a block, by the kernel's column count (the exact widths 5 to 7 included).
// Measured on the RTX 3070 against the Qwen2.5-7B shapes; the counts that keep
// the register budget at three blocks an SM for four columns and two for five
// to eight.
#define KQ_DBR(nc) ((nc) <= 4 ? 2 : (nc) <= 8 ? 4 : 2)
#define KQ_DBW(nc) ((nc) <= 4 ? 8 : (nc) <= 8 ? 4 : 8)

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

#include "dn_eps.h"   // DN_L2_EPS

// The training products (kernels_train.cuh, cuda.c gpu_train_mvt / _mm):
// a k_tmvt block's inputs, positions and weight rows a tile (TV_IT * TV_TT
// / 8 threads, 8 positions each), the columns a block of k_tmm and of k_tmmw
// carries (a grid row per tile) and their rows (one a warp)
#define TV_IT 32
#define TV_TT 64
#define TV_JT 32
#define TMM_NC 16
#define TMM_NCW 32
#define TMM_WARPS 8
#define TMM_RW 4        // rows a warp of k_tmm / k_tmmw carries

// The register-blocked forward product (kernels_train.cuh k_tmmr / k_tmmrn,
// R8.20 attempt 3): a block's output rows, its columns (wide and narrow
// twin), the inputs a chunk stages and the chains each output's sum is split
// into (chunk c feeds chain c % TMR_KS; the chains fold in order). Four
// groups of 64 threads, one a chain.
#define TMR_BM 64
#define TMR_BN 64
#define TMR_BNN 16
#define TMR_BK 16
#define TMR_KS 4

// The Gated DeltaNet recurrence of a training step: tdn_args.h
#include "tdn_args.h"

// ---- R8.24.1 list B: the device-resident --train-gpu step's kernels.
// Every kernel here is a host-chain twin: one thread an output element (or a
// row, where the host chains over the row), the host function's operations in
// source order, never contracted (__fmaf_rn where the host writes fmaf,
// __fmul_rn / __fadd_rn / __fdiv_rn / __fsqrt_rn elsewhere), every chain serial
// in the host's index order, transcendentals as src/pmath.h computes them. So a
// T3 build's host function and its kernel agree bit for bit; a kernel that must
// take another order says so where it is defined. Pointers are plain device
// pointers, strides are in floats, and no kernel reads the training context.

// B1: a gated FFN's activation and its adjoint, elementwise over n floats
// (model.c gated_act, act_d, act_f, as the recompute's and the gate and up
// sites' loops use them): h = act(g)*u; dg = dh*u*act'(g), du = dh*act(g).
// act: ACT_SILU only so far (ACT_* of model.h).
typedef struct {
    int n, act;
} tffn_args;

// B6: the adapter at a projection site, rank r, scale s, over nb rows: x rows
// x_stride floats apart (n_in wide), y / dy rows y_stride apart (n_out wide),
// dx rows x_stride apart. Forward (model.c lora_apply): t = A x, y += s B t.
// Backward (lora_site_bw's adapter half, lsb_*): tb = B^T dy, tf = A x,
// ga[k][i] += s tb[t][k] x[t][i] and gb[j][k] += s dy[t][j] tf[t][k] (each a
// chain over t ascending from its stored value), dx += s (A^T tb). A is
// [r][n_in], B [n_out][r], t / tb / tf [nb][r], ga like A, gb like B.
typedef struct {
    int n_in, n_out, r, nb, x_stride, y_stride;
    float s;
} tlora_args;

// B8: the training K/V store, rows of n floats (src rows src_stride floats
// apart) into the cache's rows (row_bytes apart) in the cache's kind, as the
// forward writes them: kind 0 f16 through fp16.h's f32_to_f16_soft (round to
// nearest, ties to even, NaN quieted with its payload kept), kind 3 the
// floats as they are. Small and every field read: see the note at k_tkv_store.
typedef struct {
    int rows, n, src_stride, kind;
    ka_u64 row_bytes;
} tkv_args;

// R0 of the resident step: the residual add, x[r][i] += y[r][i] over rows of
// n floats (x rows x_stride apart, y rows y_stride), one float add an
// element, as model.c layer_branch_out writes it.
typedef struct {
    int rows, n, x_stride, y_stride;
} tadd_args;

// B2: rmsnorm (model.c rmsnorm; qk_norm runs it per head) and its adjoint
// (rmsnorm_bw), over `rows` rows of n floats. Forward y = x*r*w; adjoint
// dx[i] += r*w[i]*dy[i] - c*x[i]. w is one n-vector for every row (qk_norm
// passes one w to every head), NULL the weightless form. Per head: rows =
// positions x heads, n = head_dim, strides head_dim.
typedef struct {
    int n, rows;
    int x_stride, y_stride, dy_stride, dx_stride;
    float eps;
} trn_args;

// B3: rope_apply / rope_unapply (model.c) over T positions t0 .. t0+T-1, each a
// row of n_heads heads of head_dim, rows `stride` floats apart; pairs j <
// half_dim rotate (neox: j and j + half_dim, else 2j and 2j+1) by the angle
// pos * fr[j] (fr: the layer's frequency table, device), c and s scaled by
// mscale. adjoint 0: rope_apply, 1: rope_unapply.
typedef struct {
    int n_heads, head_dim, half_dim, neox;
    int T, t0, stride, adjoint;
    float mscale;
} trope_args;

// B4: attention forward over the training K/V rows (model.c attn_run for one
// (t, h) a thread): T positions, n_head query heads in groups of kv_mul a kv
// head, hd values a head, q / ao rows q_dim apart, K and V rows k_row_b /
// v_row_b bytes apart in the cache's kinds (kind_k, kind_v as
// model_kv_kind_k / _v report them), the score scale, the sliding window (0:
// none). The fields before kind_k are tab_args' (tdn_args.h), the layout
// k_tab_p reads.
typedef struct {
    int T, n_head, kv_mul, hd, q_dim, kv_dim, swa;
    ka_u64 k_row_b, v_row_b;
    float scale;
    int kind_k, kind_v;
    int pos0;   // the first query row's position: row b is position pos0 + b, over
                // cache rows 0 .. pos0 + T - 1 (a chunk after the first; 0 otherwise)
} tattn_args;

// B5: the head over `rows` rows of V logits, rows `stride` floats apart. In:
// the raw logits. Out, for a scored row (target >= 0): the transformed row's
// normalizer and target logit (apply_head_transforms, then train_row_lse in
// double, as the taped forward's loss reads them), and in place the row's
// adjoint as head_rows_bw computes it without a teacher (scale, softcap with
// th its tanh scratch, suppress, softmax in float, minus the target, times
// pos_w, the transforms' adjoints). logit_scale 0 or 1: none; softcap <= 0:
// none; suppress: n_suppress ids (device).
typedef struct {
    int V, rows, stride, n_suppress;
    float logit_scale, softcap;
} thead_args;

// B7: AdamW over n parameters (model.c adam_buf), and the step log's gradient
// norms (model_lora_grad_norms: per layer the square root of a double sum of
// squares over its slots' A then B gradients in table order; the max and mean
// over layers; a non-finite gradient fails the step).
typedef struct {
    ka_u64 n;
    float lr, b1, b2, eps, wd, bc1, bc2;
} tadam_args;

#endif // RUNNER_KERNEL_ARGS_H
