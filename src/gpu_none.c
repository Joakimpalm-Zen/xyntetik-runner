// GPU backend stub for platforms without one.
// CUDA (cuda.c) and Metal (metal.m) implement this same interface.
#include "gpu.h"

bool gpu_available(char *name, int cap) {
    (void)name; (void)cap;
    return false;
}

bool gpu_device_id(char *id, int cap) {
    (void)id; (void)cap;
    return false;   // no GPU, so nothing to account for and nothing to name
}

bool gpu_unified_memory(void) { return false; }

bool gpu_offload_budget(int reserve_vram_pct, uint64_t *budget,
                        uint64_t *headroom) {
    (void)reserve_vram_pct; (void)budget; (void)headroom;
    return false;
}

bool gpu_mem_info(size_t *free_bytes, size_t *total_bytes) {
    (void)free_bytes; (void)total_bytes;
    return false;
}

// No device, so no device-side fit ceiling distinct from ordinary RAM. main.c
// calls this on the --caps path, so its absence was not a cosmetic gap: this
// file could not link a runner at all.
bool gpu_max_working_set(size_t *bytes) {
    (void)bytes;
    return false;
}

const char *gpu_shader_source_sha(void) { return NULL; }   // no shaders here
const char *gpu_compute_capability(void) { return NULL; }

bool gpu_kv_q8_ok(void) {
    return false;   // no backend here; the CPU path handles q8 on its own
}
bool gpu_kv_fp4_ok(void) {
    return false;
}
bool gpu_kv_f32_ok(void) {
    return false;
}

bool gpu_quant_ok(int type) {
    (void)type;
    return false;   // no kernels at all, so --caps advertises an empty GPU list
}

// No GPU, so no tensor-core GEMM to force and none ever dispatched. Present so
// the TC tolerance gate links against a CPU-only build.
void gpu_tc_force(int on) { (void)on; }
unsigned long gpu_tc_dispatches(void) { return 0; }
unsigned long gpu_tc_dispatches_type(int type) { (void)type; return 0; }
bool gpu_tc_takes(const model_t *m, int type) { (void)m; (void)type; return false; }
bool gpu_prefill_twin16(const model_t *m, int type) { (void)m; (void)type; return false; }
int gpu_prefill_route(const model_t *m, int type, int batch, bool scaled) {
    (void)m; (void)type; (void)batch; (void)scaled; return GPU_PREFILL_NONE;
}
unsigned long gpu_tc2_dispatches(void) { return 0; }
unsigned long gpu_i8_dispatches(void) { return 0; }
unsigned long gpu_tc2_reused(void) { return 0; }
unsigned long gpu_tc2_80_dispatches(void) { return 0; }
unsigned long gpu_moe_twin_tiles(void) { return 0; }
void gpu_moe_twin_force(int on) { (void)on; }
unsigned long gpu_tc2_wide_dispatches(void) { return 0; }
void gpu_tc2_wide_force(int on) { (void)on; }
int gpu_prompt_window(const model_t *m) { (void)m; return 0; }
void gpu_route_pin(int mask) { (void)mask; }
int gpu_route_active(const model_t *m) { (void)m; return 0; }
void gpu_recur_seq_force(int on) { (void)on; }
unsigned long gpu_recur_seq_dispatches(void) { return 0; }
unsigned long gpu_exact_verify_tiles(void) { return 0; }
bool gpu_tc_type_has_kernel(int type) { (void)type; return false; }

// Same, for the fast decode matvec: the mv tolerance gate links here and
// reports "never dispatched" rather than failing to build.
void gpu_mv_force(int on) { (void)on; }
unsigned long gpu_mv_dispatches(void) { return 0; }
unsigned long gpu_mv2_dispatches(void) { return 0; }

// Metal-only lever; present so the gate links and reports 'never here'.
void gpu_attn_coop_force(int on) { (void)on; }
unsigned long gpu_attn_coop_dispatches(void) { return 0; }
unsigned long gpu_attn_tile_dispatches(void) { return 0; }
unsigned long gpu_attn_tiled_dispatches(void) { return 0; }
unsigned long gpu_attn_tc_dispatches(void) { return 0; }
unsigned long gpu_attn_dec_gqa_dispatches(void) { return 0; }
unsigned long gpu_attn_dec_mla_dispatches(void) { return 0; }
unsigned long gpu_attn_tcw_mla_dispatches(void) { return 0; }
unsigned long gpu_kv_gather_tokens(void) { return 0; }
unsigned long gpu_attn_gqa_dispatches(void) { return 0; }
unsigned long gpu_splitk_dispatches(void) { return 0; }
long gpu_live_buffers(void) { return -1; }

// Same, for the fused-vs-eager MoE routing gate.
void gpu_moe_eager_force(int on) { (void)on; }

bool gpu_moe_ok(void) {
    return false;   // no backend here at all
}

bool gpu_shared_weights_resident(const model_t *m) { (void)m; return false; }

bool gpu_removed_sublayers_ok(void) { return false; }
bool gpu_rope_sets_ok(void) { return false; }
void gpu_rope_changed(model_t *m) { (void)m; }
bool gpu_rope_table_in_use(const model_t *m, float *out, int n) { (void)m; (void)out; (void)n; return false; }
bool gpu_batch_rope_sets_ok(void) { return false; }
bool gpu_moe_route_caps(int *max_experts, int *max_used) {
    (void)max_experts; (void)max_used;
    return false;
}
bool gpu_qwen35_ok(void) { return false; }

bool gpu_eseries_ok(void) {
    return false;
}

bool gpu_init(model_t *m) {
    (void)m;
    return false;
}

bool gpu_forward_batch(model_t *m, const int32_t *tokens, int n, int pos,
                       bool want_logits, float **logits) {
    (void)m; (void)tokens; (void)n; (void)pos; (void)want_logits; (void)logits;
    return false;
}

void gpu_free(model_t *m) {
    (void)m;
}

// R3.68: this backend allocates its KV cache whole, so every row is ready
bool gpu_kv_reserve(model_t *m, int rows, char *why, size_t cap) {
    (void)m; (void)rows; (void)why; (void)cap;
    return true;
}
bool gpu_kv_rows(const model_t *m, int *mapped, int *step) {
    (void)m; (void)mapped; (void)step;
    return false;
}

void gpu_disable(model_t *m) {
    m->gpu = NULL;
}

bool gpu_os_vidmem(const model_t *m, uint64_t *budget, uint64_t *usage,
                   uint64_t *nonlocal) {
    (void)m; (void)budget; (void)usage; (void)nonlocal;
    return false;
}

bool gpu_recurrent_download(model_t *m) { (void)m; return true; }
bool gpu_recurrent_upload(model_t *m) { (void)m; return true; }
bool gpu_recurrent_rollback_mark(model_t *m) { (void)m; return false; }
bool gpu_recurrent_rollback(model_t *m) { (void)m; return false; }
bool gpu_recurrent_restore_row(model_t *m, int r) { (void)m; (void)r; return false; }
bool gpu_mtp_bound(const model_t *m) { (void)m; return false; }
bool gpu_mtp_note_row(model_t *m, int row) { (void)m; (void)row; return false; }
bool gpu_mtp_queue_pending(model_t *m, int slot) { (void)m; (void)slot; return false; }
bool gpu_mtp_reset_pending(model_t *m) { (void)m; return false; }
bool gpu_mtp_run(model_t *m, bool chained, int slot0, const int32_t *tok, int n, int pos, bool want_logits) {
    (void)m; (void)chained; (void)slot0; (void)tok; (void)n; (void)pos; (void)want_logits; return false;
}

// No backend, so no microbatch: model_batch_decode sees NULL and decodes the
// sequences one at a time, which is what this platform would have done anyway.
bool gpu_mvt(model_t *m, const gguf_tensor *w, const float *dy, float *dx,
             int n_in, int n_out, int batch) {
    (void)m; (void)w; (void)dy; (void)dx; (void)n_in; (void)n_out;
    (void)batch;
    return false;
}

bool gpu_train_init(model_t *m) { (void)m; return false; }
void gpu_train_free(model_t *m) { (void)m; }
// A build with no GPU backend never has m->gpu set, so model_lora_load never
// reaches this; it exists to keep the contract total.
bool gpu_mvcanon(model_t *m, const gguf_tensor *w, const float *x, float *y,
                 int n_in, int n_out) {
    (void)m; (void)w; (void)x; (void)y; (void)n_in; (void)n_out;
    return false;
}

bool gpu_lora_bind(model_t *m) { (void)m; return false; }
void gpu_lora_unbind(model_t *m) { (void)m; }

bool gpu_train_mvt(model_t *m, const gguf_tensor *w, const float *dy,
                   float *dx, int n_in, int n_out, int nb) {
    (void)m; (void)w; (void)dy; (void)dx; (void)n_in; (void)n_out; (void)nb;
    return false;
}

bool gpu_train_mvt0(model_t *m, const gguf_tensor *w, const float *dy,
                    float *dx, int n_in, int n_out, int nb) {
    (void)m; (void)w; (void)dy; (void)dx; (void)n_in; (void)n_out; (void)nb;
    return false;
}

bool gpu_train_mm(model_t *m, const gguf_tensor *w, const float *x,
                  int x_stride, float *y, int y_stride, int n_in, int n_out,
                  const float *bias, int nb) {
    (void)m; (void)w; (void)x; (void)x_stride; (void)y; (void)y_stride;
    (void)n_in; (void)n_out; (void)bias; (void)nb;
    return false;
}

unsigned gpu_train_mm_parts(model_t *m, const gpu_mm_part *p, int np,
                            const float *x, int x_stride, int n_in, int nb) {
    (void)m; (void)p; (void)np; (void)x; (void)x_stride; (void)n_in; (void)nb;
    return 0;
}

bool gpu_train_pin(model_t *m, void *p, size_t bytes) {
    (void)m; (void)p; (void)bytes;
    return false;
}

void *gpu_train_host_buf(model_t *m, int slot, size_t bytes) {
    (void)m; (void)slot; (void)bytes;
    return NULL;
}

bool gpu_train_rmsnorm_check(model_t *m, const trn_args *a, const float *x,
                             const float *w, const float *dy, float *y, float *dx) {
    (void)m; (void)a; (void)x; (void)w; (void)dy; (void)y; (void)dx;
    return false;
}
bool gpu_train_rope_check(model_t *m, const trope_args *a, const float *fr, float *v) {
    (void)m; (void)a; (void)fr; (void)v;
    return false;
}
bool gpu_train_kv_store_check(model_t *m, const tkv_args *a, const float *src,
                              unsigned char *dst) {
    (void)m; (void)a; (void)src; (void)dst;
    return false;
}
bool gpu_train_adam_check(model_t *m, const tadam_args *a, float *th, const float *g,
                          float *mo, float *vo) {
    (void)m; (void)a; (void)th; (void)g; (void)mo; (void)vo;
    return false;
}
bool gpu_train_gnorm_check(model_t *m, const float *g, size_t n_g, const uint64_t *seg_off,
                           const uint64_t *seg_len, int n_seg, const int *layer_seg,
                           int n_layer, double *max_out, double *mean_out, bool *finite) {
    (void)m; (void)g; (void)n_g; (void)seg_off; (void)seg_len; (void)n_seg; (void)layer_seg;
    (void)n_layer; (void)max_out; (void)mean_out; (void)finite;
    return false;
}
bool gpu_train_head_check(model_t *m, const thead_args *a, float *z, const int32_t *targets,
                          const float *pos_w, const int32_t *suppress, double *lse,
                          float *z_target) {
    (void)m; (void)a; (void)z; (void)targets; (void)pos_w; (void)suppress; (void)lse;
    (void)z_target;
    return false;
}
bool gpu_train_attn_fw_check(model_t *m, const tattn_args *a, const void *kc,
                             const void *vc, const float *q, float *ao) {
    (void)m; (void)a; (void)kc; (void)vc; (void)q; (void)ao;
    return false;
}
bool gpu_train_ffn_check(model_t *m, int act, const float *g, const float *u,
                         const float *dh, float *h, float *dg, float *du, int n) {
    (void)m; (void)act; (void)g; (void)u; (void)dh; (void)h; (void)dg; (void)du; (void)n;
    return false;
}
bool gpu_train_lora_check(model_t *m, int r, int n_in, int n_out, int nb, float s,
                          const float *A, const float *B, const float *x,
                          const float *dy, float *y, float *ga, float *gb, float *dx) {
    (void)m; (void)r; (void)n_in; (void)n_out; (void)nb; (void)s; (void)A; (void)B;
    (void)x; (void)dy; (void)y; (void)ga; (void)gb; (void)dx;
    return false;
}
bool gpu_train_ffn_dev_ok(model_t *m) { (void)m; return false; }
bool gpu_train_site_push(model_t *m, int l, int which, const float *a,
                         const float *b, const float *ga, const float *gb,
                         int r, int n_in, int n_out, float s) {
    (void)m; (void)l; (void)which; (void)a; (void)b; (void)ga; (void)gb;
    (void)r; (void)n_in; (void)n_out; (void)s;
    return false;
}
bool gpu_train_site_pull(model_t *m, int l, int which, float *ga, float *gb) {
    (void)m; (void)l; (void)which; (void)ga; (void)gb;
    return false;
}
bool gpu_train_ffn_fw(model_t *m, int l, const gguf_tensor *wg, const gguf_tensor *wu,
                      const gguf_tensor *wd, const float *x, int x_stride, float *y,
                      int y_stride, int E, int nff, int nb) {
    (void)m; (void)l; (void)wg; (void)wu; (void)wd; (void)x; (void)x_stride; (void)y;
    (void)y_stride; (void)E; (void)nff; (void)nb;
    return false;
}
bool gpu_train_ffn_recompute(model_t *m, int l, const gguf_tensor *wg,
                             const gguf_tensor *wu, const float *xn2, int E, int nff,
                             int T) {
    (void)m; (void)l; (void)wg; (void)wu; (void)xn2; (void)E; (void)nff; (void)T;
    return false;
}
bool gpu_train_ffn_bw(model_t *m, int l, const gguf_tensor *wg, const gguf_tensor *wu,
                      const gguf_tensor *wd, const float *dbr, float *dxn2, int E,
                      int nff, int T) {
    (void)m; (void)l; (void)wg; (void)wu; (void)wd; (void)dbr; (void)dxn2; (void)E;
    (void)nff; (void)T;
    return false;
}

void gpu_train_prof(model_t *m, double *t_mm, long *n_mm, double *t_mvt,
                    long *n_mvt, double *k_mm, double *k_mvt) {
    (void)m;
    *t_mm = *t_mvt = *k_mm = *k_mvt = 0;
    *n_mm = *n_mvt = 0;
}

double gpu_train_dev_wall(model_t *m) {
    (void)m;
    return 0;
}

bool gpu_train_dn_ok(model_t *m, const tdn_args *a) {
    (void)m; (void)a;
    return false;
}

bool gpu_train_dn_fw(model_t *m, const tdn_args *a, const float *cv,
                     const float *beta, const float *decay, float *state,
                     float *y) {
    (void)m; (void)a; (void)cv; (void)beta; (void)decay; (void)state; (void)y;
    return false;
}

bool gpu_train_dn_bw(model_t *m, const tdn_args *a, const float *cv,
                     const float *dy, float *dq, float *dk, float *dv,
                     float *dbeta, float *ddecay) {
    (void)m; (void)a; (void)cv; (void)dy; (void)dq; (void)dk; (void)dv;
    (void)dbeta; (void)ddecay;
    return false;
}

void gpu_train_dn_counts(model_t *m, long *dev, long *failed) {
    (void)m;
    *dev = *failed = 0;
}

bool gpu_train_dn_pull(model_t *m) {
    (void)m;
    return true;
}

void gpu_train_dn_reset(model_t *m) { (void)m; }

bool gpu_train_attn_bw(model_t *m, const tab_args *a, const void *kc,
                       const void *vc, const float *q, const float *dao,
                       float *dq, float *dk, float *dv) {
    (void)m; (void)a; (void)kc; (void)vc; (void)q; (void)dao; (void)dq;
    (void)dk; (void)dv;
    return false;
}

void gpu_train_attn_counts(model_t *m, long *dev, long *failed) {
    (void)m;
    *dev = *failed = 0;
}

void gpu_train_placement(model_t *m, int *on_dev, size_t *dev_bytes,
                         int *on_cpu, size_t *cpu_bytes, size_t *budget,
                         bool *budget_set) {
    (void)m;
    *on_dev = *on_cpu = 0;
    *dev_bytes = *cpu_bytes = *budget = 0;
    *budget_set = false;
}

bool gpu_spec_keep_ok(const model_t *m) { (void)m; return false; }
void gpu_kv_host_wrote(model_t *m, int lo) { (void)m; (void)lo; }
bool gpu_hidden_rows_ok(const model_t *m) { (void)m; return false; }
const char *gpu_route_detail(const model_t *m) { (void)m; return ""; }
float *gpu_spec_logits(model_t *m, int row) { (void)m; (void)row; return NULL; }

gpu_batch *gpu_batch_create(model_t **seqs, int n) {
    (void)seqs; (void)n;
    return NULL;
}

void gpu_batch_free(gpu_batch *b) {
    (void)b;
}

bool gpu_batch_tiles_used(const gpu_batch *b) { (void)b; return false; }

bool gpu_batch_decode(gpu_batch *b, const int *idx, const int32_t *tok,
                      const int *pos, int n, float **out) {
    (void)b; (void)idx; (void)tok; (void)pos; (void)n; (void)out;
    return false;
}
