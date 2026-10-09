// GGUF tensor element types and the quantized dot products.
#ifndef RUNNER_QUANTS_H
#define RUNNER_QUANTS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// tensor data types (subset of ggml)
enum ggml_type {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3,
    T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    // codebook i-quants (sub-4-bit; dequant transcribed from llama.cpp
    // b10353 ggml-quants.c, grids in quants_iq_grids.h). CUDA (2026-09-14)
    // and Metal (2026-09-17) carry device twins of the decoders.
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19,
    T_IQ4_NL = 20,
    T_IQ3_S = 21, T_IQ2_S = 22,
    T_IQ4_XS = 23,
    T_IQ1_M = 29,
    T_BF16 = 30,
    T_MXFP4 = 39,   // OCP microscaling FP4 (E2M1 codes + per-block E8M0 scale); gpt-oss
    T_Q1_0 = 41,    // 1-bit (llama.cpp ggml type 41, 128/block, f16 scale):
                    // RECOGNIZED, NOT DECODED, named for the refusal
    T_Q2_0 = 42,    // 2-bit {-1,0,+1,+2} x f16 scale, 64/block (llama.cpp ggml
                    // type 42; the routed experts of Qwen3.8-Flash-Next GSQ)
    T_NVFP4 = 40,   // NVIDIA FP4 (E2M1 codes + per-block UE4M3 scale, 16/block);
                    // RECOGNIZED, NOT DECODED: named so a refusal says NVFP4
                    // instead of "?" (first seen in the wild on a DGX Spark,
                    // 2026-08-29; upstream llama.cpp ggml type 40)
};
int         ggml_block_size(int type);   // elements per block
size_t      ggml_type_size(int type);    // bytes per block
const char *ggml_type_name(int type);
bool        ggml_type_supported(int type);
static inline size_t ggml_row_size(int type, int64_t n) {
    return (size_t)(n / ggml_block_size(type)) * ggml_type_size(type);
}

// dequantize a full row of n elements
void  dequant_row(int type, const void *src, float *dst, int n);
// target == T_KEEP: every tensor keeps its own on-disk type (--prune-experts
// with no --quant — geometry changes, precision doesn't).
#define T_KEEP (-1)
// Requantize a GGUF at in_path to out_path (written beside + atomically
// renamed), optionally pruning MoE experts per prune_path first (NULL =
// no pruning; see quantize.c's prune_plan_load for the LIST.json shape).
// Returns 0 on success, nonzero on failure with the destination left
// untouched. Declared here so tests can drive it without main.c.
int   quantize_gguf(const char *in_path, const char *out_path, int target,
                    const char *prune_path);
// As above plus a --type-plan: a JSON per-TENSOR type override, which is the
// finest granularity GGUF can express (experts are stored stacked, one 3-D
// tensor per layer with a single type, so per-expert precision is not
// representable without splitting tensors no loader expects).
//   {"default":"keep","rules":[{"match":"_exps.weight","type":"q4_0"}]}
// First matching rule wins. Passing NULL is exactly quantize_gguf.
// remove_spec ("attn:N[,mlp:M,...]", or NULL) physically drops a block's
// attention or dense-FFN tensors and declares the absence with a 0 in the
// per-layer attention.head_count / head_count_kv or feed_forward_length
// arrays, the reading llama.cpp's deci graph and this loader share.
int   quantize_gguf_plan(const char *in_path, const char *out_path, int target,
                         const char *prune_path, const char *type_plan_path,
                         const char *remove_spec);
// RI-3 (owner 2026-10-02): --type-plan-strict. A plan rule the writer would
// decline (the type's block does not divide the row, it is not smaller than
// the tensor's type, or it would fall back to a 32-block cousin) fails the
// build before any tensor data is written, instead of being reported and
// applied. Off by default: declines are predicted (scripts/type-plan-size.py)
// and reported, and that stays the default behaviour.
void  quantize_set_type_plan_strict(bool on);
// Compile a longer native-YaRN context contract into a standalone GGUF.
// Only {arch}.context_length and {arch}.rope.scaling.factor may change; the
// implementation reopens both files and byte-compares every tensor payload
// before reporting success.
typedef struct {
    uint32_t source_context;
    uint32_t original_context;
    float    source_factor;
    bool     tensors_byte_identical;
} context_surgery_result;
int   context_surgery_gguf(const char *in_path, const char *out_path,
                           uint32_t target_context, float target_factor,
                           context_surgery_result *result);
// --merge-lora: rewrite the base GGUF with a LoRA adapter's delta folded
// into its adapted projections (W' = W + (alpha/r)*user_scale*B·A), each
// tensor requantized to its own type (target T_KEEP) or to `target`.
// Merging into a quantized type rounds the delta through that type's grid:
// the merged artifact's fidelity is a measurement, not a given. Untouched
// tensors are copied byte-verbatim under T_KEEP.
// What the merged file kept of the delta, over the adapted tensors: bytes
// written and bytes that differ from the base written alone at the same
// type, and delta_retained = ret_num / ret_den, the projection of (merged -
// base-alone) onto the intended delta (1 = all kept, 0 = rounded away).
typedef struct {
    uint64_t bytes, bytes_changed;
    double ret_num, ret_den;
} merge_survival;
// --merge-lora's default floor on delta_retained. Calibrated 2026-10-09 on the
// merge study's Qwen3-4B Q4_K_M + tool-use adapter: merged back onto Q4_K,
// 1.8% (scale 1) to 8.9% (scale 8) of the delta survived; into Q8_0 99.5%.
#define MERGE_MIN_RETAINED     0.5f
#define MERGE_MIN_RETAINED_STR "0.5"
// Refuses (returns 1, destination untouched) when delta_retained falls under
// min_retained; a negative min_retained never refuses. *surv may be NULL.
int   merge_lora_gguf(const char *in_path, const char *adapter_path,
                      float user_scale, const char *out_path, int target,
                      float min_retained, merge_survival *surv);
// CLI-name -> T_* for every type the quantizer can write ("keep" -> T_KEEP);
// -2 for a name it cannot.
int   quantize_type_from_name(const char *s);
// dot(row, x) over n elements
float vec_dot(int type, const void *row, const float *x, int n);
// out[b] = dot(w, x + b*x_stride) for nb columns sharing one weight row
// R dequantized weight rows x nb activation columns in one register tile.
// Every output accumulates over the row in the same order as vec_dot, so the
// result is bit-identical to calling the single-column dot R*nb times.
void  vec_dot_f32_tile(const float *const *w, int nrow, const float *x,
                       int x_stride, int nb, int n, float *out, int out_stride);
void  vec_dot_f32_multi(const float *w, const float *x, int x_stride,
                        int nb, int n, float *out);
void  q8_quant_row(const float *x, void *dst, int n); // n % 32 == 0
void  q8_accum_row(const void *src, float a, float *out, int n);
// --kv fp4 cache blocks: 16 values in 9 bytes (UE4M3 scale + E2M1 nibbles)
float   kv_ue4m3_to_f32(uint8_t x);
uint8_t kv_ue4m3_ceil(float s);
void  fp4_quant_row(const float *x, void *dst, int n);   // n % 16 == 0
void  fp4_dequant_row(const void *src, float *y, int n);
float fp4_dot_row(const void *src, const float *x, int n);
void  fp4_accum_row(const void *src, float a, float *out, int n);

// ------------------------------------------------ fused int8 dot (CPU lever 1)
//
// The scalar/SIMD vec_dot above keeps f32 activations and converts each weight
// quant to f32 before the FMA. The fused route instead quantizes the ACTIVATION
// row to int8 once per matvec and does the whole dot in int8 with an int32
// accumulator (AVX-512 VNNI `_mm512_dpbusd_epi32`, AVX2 `maddubs` fallback).
// Quantizing the activations means it can never be bit-identical to vec_dot;
// it is a tolerance-gated fast route in the same sense as the CUDA tensor-core
// prefill path. It is OFF by default — measured 2026-08-13, no (format, model)
// combo cleared the 0/64 teacher-forced flip bar with a decode gain worth
// taking; `RUNNER_CPU_I8=1` opts in. The promotion record is in quants.c and
// the gate is test-i8-tol.
//
//   if (i8_dot_ok(type, n)) { i8_quant_act(x, buf, n);        // once per matvec
//                             for each row: vec_dot_i8(type, row, buf, n); }
//
bool   i8_dot_enabled(void);              // RUNNER_CPU_I8 pin, cached
bool   i8_dot_ok(int type, int n);        // fused kernel exists for (type, n)
size_t i8_act_size(int n);                // scratch bytes for n activations
void   i8_quant_act(const float *x, void *dst, int n);
float  vec_dot_i8(int type, const void *row, const void *xq, int n);
// One weight row against up to VEC_DOT_MULTI_MAX activation columns, each
// output byte-equal to vec_dot(type, row, xs[c], n): the weight block is
// decoded once for every column (the small-batch win for codebook quants).
#define VEC_DOT_MULTI_MAX 8
void   vec_dot_multi(int type, const void *row, const float *const *xs, int nc,
                     int n, float *out);
// Force the route on (1) or off (0) regardless of RUNNER_CPU_I8; -1 returns
// to the env default. Mirrors gpu_tc_force so one harness can drive both.
void   i8_dot_force(int on);
// Matvecs that took the fused route since process start. A tolerance gate
// that sees zero is comparing the scalar path with itself.
unsigned long i8_dot_dispatches(void);
unsigned long vec_dot_multi_calls(void);   // route gate: calls into vec_dot_multi

#endif // RUNNER_QUANTS_H
