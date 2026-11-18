// The device-resident --train-gpu step's kernel arguments (R8.24.1 list B),
// shared by the HOST and the DEVICE like the rest of kernel_args.h, which
// includes this file. Its own header because gpu.h declares the training
// entries that take these structs, and src/metal.m, which includes gpu.h,
// keeps its own copies of kernel_args.h's other structs (mv_args, rope_args,
// attn_args, moe_args): gpu.h must not include kernel_args.h (tdn_args.h is
// split out for the same reason). Plain structs and fixed-width integers
// only: gcc compiles it as C and nvcc as C++.
#ifndef RUNNER_TRAIN_ARGS_H
#define RUNNER_TRAIN_ARGS_H

// the same 8-byte type kernel_args.h names (see its note); defined once
#ifndef RUNNER_KA_U64
#define RUNNER_KA_U64
typedef unsigned long long ka_u64;
#endif

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

#endif // RUNNER_TRAIN_ARGS_H
