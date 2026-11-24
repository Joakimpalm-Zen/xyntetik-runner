// R4.48: Inkling 2 Mini (inkling2_loop), the text path: the structure as data, the application-indexed cache, and
// the pieces. See inkling2.h.
//
// This file is built STRICT in every build (the Makefile adds -fno-fast-math -ffp-contract=off for it): every
// reduction is a serial float32 loop in index order, nothing is contracted, and the transcendentals are pmath.h's
// portable p_expf and p_tanhf, so a strict build gives the same bits on every machine. Against the reference
// (torch's blocked BLAS order) the pieces are compared under the error bound the fixture harness states.
#include "inkling2.h"
#include "pmath.h"
#include "quants.h"
#include "tpool.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(RUNNER_CANON_KERNELS) && defined(__AVX2__) && defined(__FMA__)
#define INK2_SIMD_AVX2 1
#include <immintrin.h>
#elif !defined(RUNNER_CANON_KERNELS) && defined(__aarch64__) && defined(__ARM_NEON)
#define INK2_SIMD_NEON 1
#include <arm_neon.h>
#endif

#if INK2_SIMD_AVX2
static inline float ink2_hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
static inline __m256 ink2_vexp8(__m256 x) {
    const __m256 hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 lo = _mm256_set1_ps(-87.3365447504f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
    const __m256 ln2_lo = _mm256_set1_ps(1.428606765330187045e-6f);
    __m256 xc = _mm256_max_ps(lo, _mm256_min_ps(hi, x));
    __m256 kf = _mm256_round_ps(_mm256_mul_ps(xc, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(kf, ln2_hi, xc);
    r = _mm256_fnmadd_ps(kf, ln2_lo, r);
    __m256 p = _mm256_set1_ps(1.98412698e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.38888889e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.33333333e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.16666667e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.66666667e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i ki = _mm256_add_epi32(_mm256_cvtps_epi32(kf), _mm256_set1_epi32(127));
    __m256 sc = _mm256_castsi256_ps(_mm256_slli_epi32(ki, 23));
    __m256 res = _mm256_mul_ps(p, sc);
    __m256 zmask = _mm256_cmp_ps(x, lo, _CMP_LE_OQ);
    __m256 imask = _mm256_cmp_ps(x, hi, _CMP_GE_OQ);
    __m256 nmask = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    res = _mm256_andnot_ps(zmask, res);
    res = _mm256_blendv_ps(res, _mm256_set1_ps(INFINITY), imask);
    return _mm256_blendv_ps(res, _mm256_add_ps(x, x), nmask);
}
static inline __m256 ink2_vtanh8(__m256 x) {
    const __m256 sgn = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ax = _mm256_andnot_ps(sgn, x);
    const __m256 em2 = ink2_vexp8(_mm256_mul_ps(_mm256_set1_ps(-2.0f), ax));
    __m256 r = _mm256_div_ps(_mm256_sub_ps(one, em2), _mm256_add_ps(one, em2));
    r = _mm256_or_ps(r, _mm256_and_ps(x, sgn));
    const __m256 tiny = _mm256_cmp_ps(ax, _mm256_set1_ps(1e-3f), _CMP_LT_OQ);
    const __m256 poly = _mm256_mul_ps(x, _mm256_fnmadd_ps(_mm256_mul_ps(x, x), _mm256_set1_ps(1.0f / 3.0f), one));
    r = _mm256_blendv_ps(r, poly, tiny);
    const __m256 sat = _mm256_cmp_ps(ax, _mm256_set1_ps(10.0f), _CMP_GE_OQ);
    return _mm256_blendv_ps(r, _mm256_or_ps(one, _mm256_and_ps(x, sgn)), sat);
}
#elif INK2_SIMD_NEON
static inline float32x4_t ink2_vexp4(float32x4_t x) {
    const float32x4_t hi = vdupq_n_f32(88.3762626647949f);
    const float32x4_t lo = vdupq_n_f32(-87.3365447504f);
    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2_hi = vdupq_n_f32(0.693145751953125f);
    const float32x4_t ln2_lo = vdupq_n_f32(1.428606765330187045e-6f);
    float32x4_t xc = vmaxq_f32(lo, vminq_f32(hi, x));
    float32x4_t kf = vrndnq_f32(vmulq_f32(xc, log2e));
    float32x4_t r = vfmsq_f32(xc, kf, ln2_hi);
    r = vfmsq_f32(r, kf, ln2_lo);
    float32x4_t p = vdupq_n_f32(1.98412698e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.38888889e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.33333333e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.16666667e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.66666667e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t ki = vaddq_s32(vcvtq_s32_f32(kf), vdupq_n_s32(127));
    float32x4_t sc = vreinterpretq_f32_s32(vshlq_n_s32(ki, 23));
    float32x4_t res = vmulq_f32(p, sc);
    uint32x4_t zmask = vcleq_f32(x, lo);
    uint32x4_t imask = vcgeq_f32(x, hi);
    uint32x4_t nmask = vmvnq_u32(vceqq_f32(x, x));
    res = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(res), zmask));
    res = vbslq_f32(imask, vdupq_n_f32(INFINITY), res);
    return vbslq_f32(nmask, vaddq_f32(x, x), res);
}
static inline float32x4_t ink2_vtanh4(float32x4_t x) {
    const uint32x4_t smask = vdupq_n_u32(0x80000000u);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t ax = vabsq_f32(x);
    const float32x4_t em2 = ink2_vexp4(vmulq_n_f32(ax, -2.0f));
    float32x4_t r = vdivq_f32(vsubq_f32(one, em2), vaddq_f32(one, em2));
    const uint32x4_t xsgn = vandq_u32(vreinterpretq_u32_f32(x), smask);
    r = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(r), xsgn));
    const uint32x4_t tiny = vcltq_f32(ax, vdupq_n_f32(1e-3f));
    const float32x4_t poly = vmulq_f32(x, vfmsq_f32(one, vmulq_f32(x, x), vdupq_n_f32(1.0f / 3.0f)));
    r = vbslq_f32(tiny, poly, r);
    const uint32x4_t sat = vcgeq_f32(ax, vdupq_n_f32(10.0f));
    const float32x4_t sone = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(one), xsgn));
    return vbslq_f32(sat, sone, r);
}
#endif

int ink2_shape_derive(ink2_shape *s, char *why, size_t why_n) {
#define INK2_REFUSE(...) do { if (why && why_n) snprintf(why, why_n, __VA_ARGS__); return -1; } while (0)
    if (s->n_block <= 0 || s->n_block > INK2_MAX_BLOCK) INK2_REFUSE("%d physical layers (1..%d)", s->n_block, INK2_MAX_BLOCK);
    if (s->attn_res_block <= 0) INK2_REFUSE("attention-residual block size %d", s->attn_res_block);
    if (s->loop_passes < 1) INK2_REFUSE("loop passes %d must be at least 1", s->loop_passes);
    if (s->loop_start <= 0) INK2_REFUSE("loop start %d must be greater than 0", s->loop_start);
    if (s->loop_start % s->attn_res_block || s->loop_end % s->attn_res_block)
        INK2_REFUSE("loop range [%d, %d) must be multiples of the block size %d", s->loop_start, s->loop_end,
                    s->attn_res_block);
    if (s->loop_end <= s->loop_start || s->loop_end > s->n_block)
        INK2_REFUSE("loop range [%d, %d) outside the %d layers", s->loop_start, s->loop_end, s->n_block);
    int n_app = s->n_block + (s->loop_passes - 1) * (s->loop_end - s->loop_start);
    if (n_app > INK2_MAX_APP) INK2_REFUSE("%d applications (at most %d)", n_app, INK2_MAX_APP);
    int a = 0;
    for (int b = 0; b < s->loop_start; b++) { s->app_block[a] = b; s->app_pass[a] = 0; a++; }
    for (int p = 0; p < s->loop_passes; p++)
        for (int b = s->loop_start; b < s->loop_end; b++) { s->app_block[a] = b; s->app_pass[a] = p; a++; }
    for (int b = s->loop_end; b < s->n_block; b++) { s->app_block[a] = b; s->app_pass[a] = 0; a++; }
    s->n_app = a;
    return 0;
#undef INK2_REFUSE
}

// the widest of an application's four convolutions: hidden (post-attention, post-MLP) or n_kv * head_dim (K, V)
static int ink2_conv_width(const ink2_shape *s, int a) {
    int g = ink2_app_global(s, a), C = g ? s->n_kv * s->head_dim : s->swa_n_kv * s->swa_head_dim;
    return C > s->hidden ? C : s->hidden;
}

static int ink2_cache_alloc(ink2_cache *c, const ink2_shape *s, int n_ctx, int max_chunk);
int ink2_cache_init(ink2_cache *c, const ink2_shape *s, int n_ctx) { return ink2_cache_alloc(c, s, n_ctx, 0); }
int ink2_cache_init_window(ink2_cache *c, const ink2_shape *s, int n_ctx, int max_chunk) {
    if (max_chunk < 1) return -1;
    return ink2_cache_alloc(c, s, n_ctx, max_chunk);
}
static int ink2_cache_alloc(ink2_cache *c, const ink2_shape *s, int n_ctx, int max_chunk) {
    memset(c, 0, sizeof *c);
    c->n_app = s->n_app; c->n_ctx = n_ctx; c->max_chunk = max_chunk;
    c->rows = calloc((size_t)s->n_app, sizeof *c->rows);
    c->k0 = calloc((size_t)s->n_app, sizeof *c->k0);
    if (!c->rows || !c->k0) { free(c->rows); free(c->k0); memset(c, 0, sizeof *c); return -1; }
    c->k = calloc((size_t)s->n_app, sizeof *c->k);
    c->v = calloc((size_t)s->n_app, sizeof *c->v);
    c->conv = calloc((size_t)s->n_app, sizeof *c->conv);
    c->conv_floats = calloc((size_t)s->n_app, sizeof *c->conv_floats);
    c->n_past = calloc((size_t)s->n_app, sizeof *c->n_past);
    if (!c->k || !c->v || !c->conv || !c->conv_floats || !c->n_past) { ink2_cache_free(c); return -1; }
    for (int a = 0; a < s->n_app; a++) {
        int g = ink2_app_global(s, a);
        size_t kv = (size_t)(g ? s->n_kv * s->head_dim : s->swa_n_kv * s->swa_head_dim);
        int rows = n_ctx;
        if (max_chunk && !g && (int64_t)s->window - 1 + max_chunk < n_ctx) rows = s->window - 1 + max_chunk;
        c->rows[a] = rows;
        c->k[a] = calloc((size_t)rows * kv, sizeof(float));
        c->v[a] = calloc((size_t)rows * kv, sizeof(float));
        // the four short convolutions of a layer, by the reference's conv_idx (0 k_proj.sconv, 1 v_proj.sconv,
        // 2 post_attention_convolution, 3 post_mlp_convolution), each a ring of conv_k - 1 input rows (ink2_conv_hist)
        c->conv_floats[a] = (size_t)4 * (size_t)(s->conv_k > 1 ? s->conv_k - 1 : 0) * (size_t)ink2_conv_width(s, a);
        c->conv[a] = c->conv_floats[a] ? calloc(c->conv_floats[a], sizeof(float)) : NULL;
        if (!c->k[a] || !c->v[a] || (c->conv_floats[a] && !c->conv[a])) { ink2_cache_free(c); return -1; }
    }
    return 0;
}

// The convolution history of application app, convolution idx (0..3): conv_k - 1 rows of the widest convolution's
// width; the input row of position u sits in row u % (conv_k - 1). NULL when conv_k is 1 (no history).
static float *ink2_conv_hist(const ink2_shape *s, ink2_cache *c, int app, int idx) {
    if (!c || s->conv_k <= 1 || !c->conv[app]) return NULL;
    return c->conv[app] + (size_t)idx * (size_t)(s->conv_k - 1) * (size_t)ink2_conv_width(s, app);
}
// After a call over positions p0..p0+T-1 has read its history: the last conv_k - 1 input rows (x [T][C]) into it
static void ink2_conv_hist_store(float *hist, int K, int C, int p0, int T, const float *x) {
    if (!hist) return;
    for (int u = T - (K - 1) < 0 ? 0 : T - (K - 1); u < T; u++)
        memcpy(hist + (size_t)((p0 + u) % (K - 1)) * C, x + (size_t)u * C, sizeof(float) * (size_t)C);
}

void ink2_cache_free(ink2_cache *c) {
    for (int a = 0; a < c->n_app; a++) {
        if (c->k) free(c->k[a]);
        if (c->v) free(c->v[a]);
        if (c->conv) free(c->conv[a]);
    }
    free(c->k); free(c->v); free(c->conv); free(c->conv_floats); free(c->n_past); free(c->rows); free(c->k0);
    free(c->work_fwd); free(c->work_dl); free(c->work_op); free(c->work_route);
    memset(c, 0, sizeof *c);
}

static _Atomic long ink2_moe_alloc_n = 0;   // TEST-ONLY count (ink2_moe_allocs)
long ink2_moe_allocs(void) { return atomic_load(&ink2_moe_alloc_n); }
static float *ink2_cache_scratch(float **buf, size_t *cap, size_t n_floats) {
    if (n_floats == 0) n_floats = 1;
    if (*cap < n_floats) {
        float *p = realloc(*buf, n_floats * sizeof(float));
        if (!p) return NULL;
        *buf = p;
        *cap = n_floats;
        atomic_fetch_add(&ink2_moe_alloc_n, 1);
    }
    return *buf;
}
static float *ink2_op_alloc(ink2_cache *c, size_t n_floats, int *owned) {
    if (c) { *owned = 0; return ink2_cache_scratch(&c->work_op, &c->work_op_cap, n_floats); }
    *owned = 1;
    atomic_fetch_add(&ink2_moe_alloc_n, 1);
    return malloc(sizeof(float) * (n_floats ? n_floats : 1));
}
static void ink2_op_free(float *p, int owned) {
    if (owned) free(p);
}
// The MoE layer's routing tables over T positions (R4.48.20.4): the instance's own scratch, beside work_op (which the
// expert body takes while the tables are read), so a prefill allocates them once and not a layer a call; malloc'd
// for a call without an instance (ink2_expert_layer). Freed with ink2_op_free.
static float *ink2_route_alloc(ink2_cache *c, size_t n_floats, int *owned) {
    if (c) { *owned = 0; return ink2_cache_scratch(&c->work_route, &c->work_route_cap, n_floats); }
    *owned = 1;
    atomic_fetch_add(&ink2_moe_alloc_n, 1);
    return malloc(sizeof(float) * (n_floats ? n_floats : 1));
}
static int ink2_decode_fuse_enabled(void) {
    const char *e = getenv("RUNNER_INK2_DECODE_FUSE");
    return !(e && e[0] == '0' && e[1] == 0);
}

size_t ink2_cache_bytes(const ink2_cache *c, const ink2_shape *s) {
    size_t n = 0;
    for (int a = 0; a < c->n_app; a++) {
        int g = ink2_app_global(s, a);
        size_t kv = (size_t)(g ? s->n_kv * s->head_dim : s->swa_n_kv * s->swa_head_dim);
        n += 2 * (size_t)c->rows[a] * kv * sizeof(float) + c->conv_floats[a] * sizeof(float);
    }
    return n + (size_t)c->n_app * (sizeof(float *) * 3 + sizeof(size_t) + sizeof(int) * 3);
}

// n ** -0.5 as the float32 operand torch multiplies by (the reference's `head_dim**-0.5` and `hidden_size**-0.5`:
// a Python float from the C library's pow, rounded to float32 when it scales a float32 tensor). pow is not required
// to be correctly rounded, so it is not used: 1 / sqrt(n) in double is two IEEE operations, each correctly rounded,
// so the same double on every IEEE machine. That double can differ from a correctly rounded n ** -0.5 in its last
// bit (it does for n = 8, 32, 96 and 128 against glibc's pow), but its float32 rounding equals the correctly rounded
// float32 of n ** -0.5 for every n in 1..65536 (checked against a 60-digit decimal value, 2026-11-21): the operand
// is the reference's on every machine where the reference's pow is within an ulp.
float ink2_inv_sqrt(int n) { return (float)(1.0 / sqrt((double)n)); }

// ---- threads over OUTPUTS. A parallel loop splits a range of independent outputs (rows of a matvec, experts' rows,
// vocabulary rows, heads) via the house tpool's static partition (tp_slice); each output is computed by the same serial
// loop as with one thread, so no reduction is split or reordered and the bits do not depend on the thread count. At 1
// thread (the default) or below INK2_PAR_MIN work, the loop runs on the caller's thread with no dispatch overhead. A
// nested ink2_par (including from inside a tpool worker, via ink2_tp_tramp's ink2_in_par) runs serially.
typedef void (*ink2_range_fn)(void *ctx, int64_t i0, int64_t i1);
static __thread int ink2_in_par;
static __thread tpool *ink2_tl_tp;
static _Atomic long ink2_par_min = 16384; // the least work (multiply-adds) a chunk gets; INK2_PAR_MIN
static _Atomic long ink2_split_loops = 0; // parallel loops that split their range over the pool, since the process began
static _Atomic int ink2_last_pool = 1;    // the threads the last of them ran on (its workers + the caller)
static _Atomic int ink2_last_threads = 0; // last thread count used or set (0 until first explicit set/bind)
static _Atomic uintptr_t ink2_test_par_fn = 0;
static _Atomic uintptr_t ink2_test_par_ctx = 0;

// Fallback pool for standalone C harnesses that have no model_t (e.g. test_inkling2_fixture, inkling2_read).
// CLI and server loads bind m->tp on TLS via ink2_set_tpool and never touch this.
static pthread_mutex_t ink2_sb_mu = PTHREAD_MUTEX_INITIALIZER;
static tpool *ink2_sb_tp = NULL;
static int ink2_sb_threads = 0;
static int ink2_sb_atexit_armed = 0;
static pthread_once_t ink2_env_once = PTHREAD_ONCE_INIT;

static void ink2_sb_cleanup(void) {
    pthread_mutex_lock(&ink2_sb_mu);
    tpool *tp = ink2_sb_tp;
    ink2_sb_tp = NULL;
    pthread_mutex_unlock(&ink2_sb_mu);
    tpool_destroy(tp);
}

static void ink2_env_init(void) {
    const char *e = getenv("INK2_PAR_MIN");
    if (e && *e) {
        long v = atol(e);
        atomic_store_explicit(&ink2_par_min, v < 1 ? 1 : v, memory_order_relaxed);
    }
    e = getenv("INK2_THREADS");
    pthread_mutex_lock(&ink2_sb_mu);
    if (!ink2_sb_threads) {
        int n = e && *e ? atoi(e) : 1;
        ink2_sb_threads = n < 1 ? 1 : n > INK2_MAX_THREADS ? INK2_MAX_THREADS : n;
    }
    pthread_mutex_unlock(&ink2_sb_mu);
}

static tpool *ink2_standalone_acquire(void) {
    pthread_mutex_lock(&ink2_sb_mu);
    int nt = ink2_sb_threads;
    if (nt <= 1) {
        pthread_mutex_unlock(&ink2_sb_mu);
        return NULL;
    }
    tpool *stale = NULL;
    if (!ink2_sb_tp || tpool_size(ink2_sb_tp) != nt) {
        stale = ink2_sb_tp;
        ink2_sb_tp = tpool_create(nt);
        if (ink2_sb_tp && !ink2_sb_atexit_armed) {
            atexit(ink2_sb_cleanup);
            ink2_sb_atexit_armed = 1;
        }
    }
    tpool *tp = ink2_sb_tp;
    if (tp) tpool_retain(tp);
    pthread_mutex_unlock(&ink2_sb_mu);
    tpool_destroy(stale);
    return tp;
}

tpool *ink2_set_tpool(tpool *tp) {
    tpool *prev = ink2_tl_tp;
    ink2_tl_tp = tp;
    if (tp) atomic_store_explicit(&ink2_last_threads, tpool_size(tp), memory_order_relaxed);
    return prev;
}

int ink2_set_threads(int n) {
    pthread_once(&ink2_env_once, ink2_env_init);
    if (n < 1) n = 1;
    if (n > INK2_MAX_THREADS) n = INK2_MAX_THREADS;
    pthread_mutex_lock(&ink2_sb_mu);
    tpool *stale = NULL;
    if (ink2_sb_tp && tpool_size(ink2_sb_tp) != n) {
        stale = ink2_sb_tp;
        ink2_sb_tp = NULL;
    }
    ink2_sb_threads = n;
    atomic_store_explicit(&ink2_last_threads, n, memory_order_relaxed);
    pthread_mutex_unlock(&ink2_sb_mu);
    tpool_destroy(stale);
    return n;
}

int ink2_get_threads(void) {
    pthread_once(&ink2_env_once, ink2_env_init);
    if (ink2_tl_tp) return tpool_size(ink2_tl_tp);
    int last = atomic_load_explicit(&ink2_last_threads, memory_order_relaxed);
    if (last > 0) return last;
    pthread_mutex_lock(&ink2_sb_mu);
    int n = ink2_sb_threads;
    pthread_mutex_unlock(&ink2_sb_mu);
    return n > 0 ? n : 1;
}

void ink2_par_stats(long *split_loops, int *last_pool) {
    if (split_loops) *split_loops = atomic_load_explicit(&ink2_split_loops, memory_order_relaxed);
    if (last_pool) *last_pool = atomic_load_explicit(&ink2_last_pool, memory_order_relaxed);
}

void ink2_set_test_par_hook(void (*fn)(void *), void *ctx) {
    atomic_store_explicit(&ink2_test_par_ctx, (uintptr_t)ctx, memory_order_release);
    atomic_store_explicit(&ink2_test_par_fn, (uintptr_t)fn, memory_order_release);
}

typedef struct {
    ink2_range_fn fn;
    void *ctx;
    int64_t n;
    int64_t group;
} ink2_tp_ctx;

// tpool worker threads do not inherit the caller's TLS ink2_in_par / ink2_tl_tp; mark ink2_in_par here
// so any nested ink2_par invoked inside a worker runs serially without re-entering the pool or fallback.
static void ink2_tp_tramp(void *raw, int i0, int i1) {
    const ink2_tp_ctx *tc = raw;
    int64_t a = (int64_t)i0 * tc->group;
    int64_t b = (int64_t)i1 * tc->group;
    if (b > tc->n) b = tc->n;
    if (a >= b) return;
    int prev = ink2_in_par;
    ink2_in_par = 1;
    tc->fn(tc->ctx, a, b);
    ink2_in_par = prev;
}

// fn over [0, n): each index costs about `work` multiply-adds (sets minimum parallel work)
void ink2_par(int64_t n, int64_t work, ink2_range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (ink2_in_par) { fn(ctx, 0, n); return; }
    pthread_once(&ink2_env_once, ink2_env_init);
    if (work < 1) work = 1;
    int64_t min_work = atomic_load_explicit(&ink2_par_min, memory_order_relaxed);
    int64_t chunk = (min_work + work - 1) / work;
    if (chunk < 1) chunk = 1;

    tpool *tp = ink2_tl_tp;
    bool drop_tp = false;
    if (!tp) {
        tp = ink2_standalone_acquire();
        drop_tp = (tp != NULL);
    }
    const int nt = tp ? tpool_size(tp) : 1;
    if (nt <= 1 || n <= chunk || n < 2) {
        if (drop_tp) tpool_destroy(tp);
        fn(ctx, 0, n);
        return;
    }

    int64_t group = 1;
    int64_t items = n;
    if (items > INT_MAX) {
        group = (n + (int64_t)INT_MAX - 1) / (int64_t)INT_MAX;
        items = (n + group - 1) / group;
    }
    atomic_fetch_add_explicit(&ink2_split_loops, 1, memory_order_relaxed);
    atomic_store_explicit(&ink2_last_pool, nt, memory_order_relaxed);
    uintptr_t hfn = atomic_load_explicit(&ink2_test_par_fn, memory_order_acquire);
    if (hfn) {
        void (*hook)(void *) = (void (*)(void *))hfn;
        void *hctx = (void *)atomic_load_explicit(&ink2_test_par_ctx, memory_order_acquire);
        hook(hctx);
    }
    ink2_tp_ctx tc = { fn, ctx, n, group };
    tpool_run(tp, ink2_tp_tramp, &tc, (int)items);
    if (drop_tp) tpool_destroy(tp);
}

// x * rsqrt(mean(x^2) + eps), the reference's Inkling2LoopUnweightedRMSNorm: a serial sum of squares, the mean, the
// epsilon, 1/sqrt, one multiply an element
static void ink2_unweighted_norm(const float *x, int n, float eps, float *y) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r;
}

// The depth-wise attention of Inkling2LoopBlockAttentionResidual and Inkling2LoopMessageAttentionResidual at one
// position: states [n_kv][H] are normed (keys = values = the normed states), each of the nq queries scores them
// (dot, then times H^-0.5, as eager_attention_forward), a softmax over the n_kv states, the weighted sum.
static void ink2_depth_attn(const float *q, int nq, const float *states, int n_kv, int H, float *normed,
                            float *scores, float *out) {
    for (int j = 0; j < n_kv; j++) ink2_unweighted_norm(states + (size_t)j * H, H, 1e-6f, normed + (size_t)j * H);
    const float scale = ink2_inv_sqrt(H);   // hidden_size ** -0.5 (see ink2_inv_sqrt)
    for (int i = 0; i < nq; i++) {
        const float *qi = q + (size_t)i * H;
        float mx = -INFINITY;
        for (int j = 0; j < n_kv; j++) {
            const float *nj = normed + (size_t)j * H;
            float d = 0.0f;
            for (int h = 0; h < H; h++) d += qi[h] * nj[h];
            scores[j] = d * scale;
            if (scores[j] > mx) mx = scores[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n_kv; j++) { scores[j] = p_expf(scores[j] - mx); sum += scores[j]; }
        for (int j = 0; j < n_kv; j++) scores[j] = scores[j] / sum;
        float *oi = out + (size_t)i * H;
        for (int h = 0; h < H; h++) {
            float a = 0.0f;
            for (int j = 0; j < n_kv; j++) a += scores[j] * normed[(size_t)j * H + h];
            oi[h] = a;
        }
    }
}

// ---- the pieces (a stub returns INK2_NOT_IMPLEMENTED until its piece lands)

static void ink2_matvec(const float *W, int n_out, int n_in, const float *x, float *y);
// The engine's one linear operation (R4.48.15): y[t][o] = W[row0 + o] . x[t] for t < T, o < n_out, x and y rows
// x_stride and y_stride floats apart. On a tensor without a view (the reference route) it is ink2_matvec position by
// position, the serial order of step 1; scale must be 1 there (the reference provider folds an expert's factor into
// its decode). On a view (the fast route) see ink2_lin_range.
enum { INK2_LIN_PARTS_MAX = 32 };
typedef struct {
    const ink2_tensor *W;
    int64_t row0;
    int n_out, n_in, T;
    const float *x;
    size_t xs;
    float *y;
    size_t ys;
    float scale;
} ink2_lin_ctx;
static void ink2_lin(const ink2_tensor *W, int64_t row0, int n_out, int n_in, int T, const float *x, size_t x_stride,
                     float *y, size_t y_stride, float scale);
static void ink2_lin_parts(const ink2_lin_ctx *parts, int np, int64_t work);
static void ink2_lin_rows(const ink2_tensor *W, int n_out, int n_in, int T, const float *x, float *y) {
    ink2_lin(W, 0, n_out, n_in, T, x, (size_t)n_in, y, (size_t)n_out, 1.0f);
}
// The softcap, tanh(v / cap) * cap: a true division by the float cap (torch's tensor / scalar), then the product
static inline float ink2_softcap(float v, float cap) { return p_tanhf(v / cap) * cap; }
// torch.sigmoid as 1 / (1 + exp(-x)), with the portable exp
static inline float ink2_sigmoid(float x) { return 1.0f / (1.0f + p_expf(-x)); }

// Inkling2LoopSparseMoeBlock at one block (model.layers.<b>.mlp), per position:
//   router: logits = W x (float32), scores = sigmoid(logits); the n_used experts with the largest scores + bias, TIES
//   BROKEN LOWEST INDEX FIRST (the reference's torch.topk(..., sorted=False) leaves ties open; this is our rule, stated
//   and tested by the harness's tie case); weights = score / (sum of the chosen scores + 1e-20) * route_scale, the sum
//   in ascending expert order;
//   experts in ASCENDING expert index (the reference's eager loop order): gate, up = W13 x; act = softcap(gate, gate
//   cap) * sigmoid(gate) * softcap(up, up cap); down = W2 act; out += down * weight.
// the expert layer's tensors, looked up once
// With the weights' get_expert (ink2_weights), the stacks are not looked up: each chosen expert is, when it runs.
// t13 / t2: the stacks as views (the fast route), the experts' rows run in place with their factors after the dot
typedef struct { const float *wr, *br, *w13, *w2; const ink2_weights *ew; char n13[160], n2[160];
                 const ink2_tensor *t13, *t2; } ink2_moe_w;
static ink2_status ink2_moe_bind(const ink2_shape *s, const ink2_weights *w, int block, ink2_moe_w *M) {
    char n1[160], n2[160];
    snprintf(n1, sizeof n1, "model.layers.%d.mlp.gate.weight", block);
    snprintf(n2, sizeof n2, "model.layers.%d.mlp.gate.e_score_correction_bias", block);
    snprintf(M->n13, sizeof M->n13, "model.layers.%d.mlp.experts.gate_up_proj", block);
    snprintf(M->n2, sizeof M->n2, "model.layers.%d.mlp.experts.down_proj", block);
    const ink2_tensor *wr = w->get(w->ctx, n1), *br = w->get(w->ctx, n2);
    int H = s->hidden, E = s->n_expert, K = s->n_used, F = s->ff;
    if (!wr || !br || wr->ne[0] != H || wr->ne[1] != E || br->ne[0] != E || K < 1 || K > E) return INK2_NO_TENSOR;
    M->wr = wr->data; M->br = br->data; M->w13 = M->w2 = NULL; M->ew = NULL; M->t13 = M->t2 = NULL;
    if (!M->wr || !M->br) return INK2_NO_TENSOR;
    if (w->get_expert) { M->ew = w; return INK2_OK; }
    const ink2_tensor *w13 = w->get(w->ctx, M->n13), *w2 = w->get(w->ctx, M->n2);
    if (!w13 || !w2 || w13->ne[0] != H || w13->ne[1] != 2 * F || w13->ne[2] != E || w2->ne[0] != F || w2->ne[1] != H ||
        w2->ne[2] != E) return INK2_NO_TENSOR;
    if (w13->q && w2->q) { M->t13 = w13; M->t2 = w2; return INK2_OK; }
    if (!w13->data || !w2->data) return INK2_NO_TENSOR;
    M->w13 = w13->data; M->w2 = w2->data;
    return INK2_OK;
}
// expert e's gate_up_proj [2F][H] and down_proj [H][F]: rows of the stacks, or the weights' get_expert
static ink2_status ink2_moe_expert(const ink2_shape *s, const ink2_moe_w *M, int e, const float **g13, const float **g2) {
    int H = s->hidden, F = s->ff;
    if (!M->ew) { *g13 = M->w13 + (size_t)e * 2 * F * H; *g2 = M->w2 + (size_t)e * H * F; return INK2_OK; }
    const ink2_tensor *t13 = M->ew->get_expert(M->ew->ctx, M->n13, e);
    if (!t13 || t13->ne[0] != H || t13->ne[1] != 2 * F || t13->ne[2] * t13->ne[3] != 1) return INK2_NO_TENSOR;
    const float *d13 = t13->data;   // held across the second lookup (the provider keeps at least two)
    const ink2_tensor *t2 = M->ew->get_expert(M->ew->ctx, M->n2, e);
    if (!t2 || t2->ne[0] != F || t2->ne[1] != H || t2->ne[2] * t2->ne[3] != 1) return INK2_NO_TENSOR;
    *g13 = d13; *g2 = t2->data;
    return INK2_OK;
}
typedef struct { const float *wr, *xt; float *lg, *sc; int H; } ink2_route_ctx;
static void ink2_route_range(void *c, int64_t e0, int64_t e1) {
    const ink2_route_ctx *r = c;
    for (int64_t e = e0; e < e1; e++) {
        float l = 0.0f; const float *row = r->wr + (size_t)e * r->H;
        for (int h = 0; h < r->H; h++) l += row[h] * r->xt[h];
        if (r->lg) r->lg[e] = l;
        r->sc[e] = ink2_sigmoid(l);
    }
}
// the router at one position: logits (if lg), sc = sigmoid scores, chosen[E], wt[e] = sc / (sum chosen + 1e-20) * scale
static void ink2_route_one(const ink2_shape *s, const ink2_moe_w *M, const float *xt, float *lg, float *sc,
                           unsigned char *chosen, float *wt) {
    int H = s->hidden, E = s->n_expert, K = s->n_used;
    ink2_route_ctx rc = { M->wr, xt, lg, sc, H };
    ink2_par(E, H, ink2_route_range, &rc);   // threads over experts: each score its own serial dot
    for (int e = 0; e < E; e++) chosen[e] = 0;
    for (int k = 0; k < K; k++) {
        int best = -1; float bv = 0.0f;
        for (int e = 0; e < E; e++) {
            if (chosen[e]) continue;
            float v = sc[e] + M->br[e];
            if (best < 0 || v > bv) { best = e; bv = v; }   // strict >: the lowest index wins a tie
        }
        if (best < 0) break;
        chosen[best] = 1;
    }
    float sum = 0.0f;
    for (int e = 0; e < E; e++) if (chosen[e]) sum += sc[e];
    float den = sum + 1e-20f;
    for (int e = 0; e < E; e++) wt[e] = chosen[e] ? sc[e] / den * s->route_scale : 0.0f;
}
// SoftcappedSwiGLU: softcap(gate, gate cap) * sigmoid(gate) * softcap(up, up cap)
static inline float ink2_swiglu(const ink2_shape *s, float g, float u) {
    return ink2_softcap(g, s->gate_cap) * ink2_sigmoid(g) * ink2_softcap(u, s->up_cap);
}
typedef struct { const ink2_shape *s; float *act; int F, nb; } ink2_swiglu_ctx;
static void ink2_swiglu_range(void *c, int64_t b0, int64_t b1) {
    const ink2_swiglu_ctx *sw = c;
    const ink2_shape *s = sw->s;
    const int F = sw->F, nb = sw->nb;
#if INK2_SIMD_AVX2
    const __m256 vgc = _mm256_set1_ps(s->gate_cap), vuc = _mm256_set1_ps(s->up_cap);
    const __m256 vone = _mm256_set1_ps(1.0f), vzero = _mm256_setzero_ps();
#elif INK2_SIMD_NEON
    const float32x4_t vgc = vdupq_n_f32(s->gate_cap), vuc = vdupq_n_f32(s->up_cap);
    const float32x4_t vone = vdupq_n_f32(1.0f);
#endif
    for (int64_t idx = b0; idx < b1; idx++) {
        const int i = (int)(idx / nb), j0 = (int)(idx % nb) * 8;
        float *a = sw->act + (size_t)i * 2 * F;
        if (j0 + 8 <= F) {
#if INK2_SIMD_AVX2
            const __m256 g = _mm256_loadu_ps(a + j0);
            const __m256 u = _mm256_loadu_ps(a + F + j0);
            const __m256 sg = _mm256_mul_ps(ink2_vtanh8(_mm256_div_ps(g, vgc)), vgc);
            const __m256 sig = _mm256_div_ps(vone, _mm256_add_ps(vone, ink2_vexp8(_mm256_sub_ps(vzero, g))));
            const __m256 su = _mm256_mul_ps(ink2_vtanh8(_mm256_div_ps(u, vuc)), vuc);
            _mm256_storeu_ps(a + j0, _mm256_mul_ps(_mm256_mul_ps(sg, sig), su));
            continue;
#elif INK2_SIMD_NEON
            for (int k = 0; k < 8; k += 4) {
                const float32x4_t g = vld1q_f32(a + j0 + k);
                const float32x4_t u = vld1q_f32(a + F + j0 + k);
                const float32x4_t sg = vmulq_f32(ink2_vtanh4(vdivq_f32(g, vgc)), vgc);
                const float32x4_t sig = vdivq_f32(vone, vaddq_f32(vone, ink2_vexp4(vnegq_f32(g))));
                const float32x4_t su = vmulq_f32(ink2_vtanh4(vdivq_f32(u, vuc)), vuc);
                vst1q_f32(a + j0 + k, vmulq_f32(vmulq_f32(sg, sig), su));
            }
            continue;
#endif
        }
        const int j1 = j0 + 8 < F ? j0 + 8 : F;
        for (int j = j0; j < j1; j++) a[j] = ink2_swiglu(s, a[j], a[F + j]);
    }
}
// The fast route's experts over T positions, grouped by expert: chosen [T][E] and wt [T][E] from the router; for each
// expert in ascending index, its gate_up rows against the positions that chose it (one call, those positions as
// columns), the SwiGLU, its down rows, and each position's out += down * wt in position order. A position's output is
// the sum over its chosen experts in ascending index, as the per-position loop adds it, and each column's dot is the
// single-column kernel's, so the rows do not depend on which other positions share the call.
static ink2_status ink2_experts_fast_c(const ink2_shape *s, const ink2_moe_w *M, int T, const float *x,
                                       const unsigned char *chosen, const float *wt, float *out, ink2_cache *c) {
    const int H = s->hidden, E = s->n_expert, F = s->ff;
    for (size_t i = 0; i < (size_t)T * H; i++) out[i] = 0.0f;
    if (T == 1 && ink2_decode_fuse_enabled() && M->t13 && M->t2 && !M->t13->q_act8 && !M->t2->q_act8) {
        int n_sel = 0;
        for (int e = 0; e < E; e++) if (chosen[e]) n_sel++;
        if (!n_sel) return INK2_OK;
        const int cap = n_sel < INK2_LIN_PARTS_MAX ? n_sel : INK2_LIN_PARTS_MAX;
        int owned = 0;
        float *buf = ink2_op_alloc(c, (size_t)cap * (2 * (size_t)F + (size_t)H), &owned);
        if (!buf) return INK2_BAD_INPUT;
        float *act = buf, *dn = buf + (size_t)cap * 2 * F;
        const int nb = (F + 7) / 8;
        int ex[INK2_LIN_PARTS_MAX];
        ink2_lin_ctx gp[INK2_LIN_PARTS_MAX], dp[INK2_LIN_PARTS_MAX];
        int e = 0;
        while (e < E) {
            int np = 0;
            for (; e < E && np < INK2_LIN_PARTS_MAX; e++) if (chosen[e]) ex[np++] = e;
            if (!np) break;
            for (int k = 0; k < np; k++) {
                const int ek = ex[k];
                const float f13 = M->t13->q_scale ? M->t13->q_scale[ek] : 1.0f;
                const float f2 = M->t2->q_scale ? M->t2->q_scale[ek] : 1.0f;
                gp[k] = (ink2_lin_ctx){ M->t13, (int64_t)ek * 2 * F, 2 * F, H, 1, x, (size_t)H,
                                        act + (size_t)k * 2 * F, (size_t)2 * F, f13 };
                dp[k] = (ink2_lin_ctx){ M->t2, (int64_t)ek * H, H, F, 1, act + (size_t)k * 2 * F, (size_t)2 * F,
                                        dn + (size_t)k * H, (size_t)H, f2 };
            }
            ink2_lin_parts(gp, np, (int64_t)H);
            ink2_swiglu_ctx sw = { s, act, F, nb };
            ink2_par((int64_t)np * nb, 256, ink2_swiglu_range, &sw);
            ink2_lin_parts(dp, np, (int64_t)F);
            for (int k = 0; k < np; k++) {
                const float w = wt[ex[k]];
                const float *d = dn + (size_t)k * H;
                for (int h = 0; h < H; h++) out[h] += d[h] * w;
            }
        }
        ink2_op_free(buf, owned);
        return INK2_OK;
    }
    // the positions that chose an expert ride at the end of the call's scratch (R4.48.20.4: nothing allocated a call)
    _Static_assert(sizeof(int) == sizeof(float), "the position list takes one float's room an entry");
    int owned = 0;
    const size_t nxe = (size_t)T * ((size_t)H + 2 * (size_t)F + (size_t)H);
    float *xe = ink2_op_alloc(c, nxe + (size_t)T, &owned);
    if (!xe) return INK2_BAD_INPUT;
    int *who = (int *)(xe + nxe);
    float *act = xe + (size_t)T * H, *dn = act + (size_t)T * 2 * F;
    for (int e = 0; e < E; e++) {
        int n = 0;
        for (int t = 0; t < T; t++) if (chosen[(size_t)t * E + e]) who[n++] = t;
        if (!n) continue;
        for (int i = 0; i < n; i++) memcpy(xe + (size_t)i * H, x + (size_t)who[i] * H, sizeof(float) * (size_t)H);
        const float f13 = M->t13->q_scale ? M->t13->q_scale[e] : 1.0f, f2 = M->t2->q_scale ? M->t2->q_scale[e] : 1.0f;
        ink2_lin(M->t13, (int64_t)e * 2 * F, 2 * F, H, n, xe, (size_t)H, act, (size_t)2 * F, f13);
        const int nb = (F + 7) / 8;
        ink2_swiglu_ctx sw = { s, act, F, nb };
        ink2_par((int64_t)n * nb, 256, ink2_swiglu_range, &sw);
        ink2_lin(M->t2, (int64_t)e * H, H, F, n, act, (size_t)2 * F, dn, (size_t)H, f2);
        for (int i = 0; i < n; i++) {
            const float w = wt[(size_t)who[i] * E + e];
            float *o = out + (size_t)who[i] * H;
            const float *d = dn + (size_t)i * H;
            for (int h = 0; h < H; h++) o[h] += d[h] * w;
        }
    }
    ink2_op_free(xe, owned);
    return INK2_OK;
}
// the experts at one position, the chosen ones in ascending index, weighted by wt[e]: ot = sum wt[e] * expert_e(xt);
// act [2F], down [H] scratch
static ink2_status ink2_experts_one(const ink2_shape *s, const ink2_moe_w *M, const float *xt, const unsigned char *chosen,
                                    const float *wt, float *act, float *down, float *ot) {
    int H = s->hidden, E = s->n_expert, F = s->ff;
    if (M->t13) return ink2_experts_fast_c(s, M, 1, xt, chosen, wt, ot, NULL);
    for (int h = 0; h < H; h++) ot[h] = 0.0f;
    for (int e = 0; e < E; e++) {
        if (!chosen[e]) continue;
        const float *g13, *g2;
        ink2_status st = ink2_moe_expert(s, M, e, &g13, &g2);
        if (st != INK2_OK) return st;
        // threads over each projection's rows (ink2_matvec); the experts' sum stays serial, ascending expert order
        ink2_matvec(g13, 2 * F, H, xt, act);
        for (int j = 0; j < F; j++) act[j] = ink2_swiglu(s, act[j], act[F + j]);
        ink2_matvec(g2, H, F, act, down);
        for (int h = 0; h < H; h++) ot[h] += down[h] * wt[e];
    }
    return INK2_OK;
}
static ink2_status ink2_expert_layer_c(const ink2_shape *s, const ink2_weights *w, int block, int T,
                                       const float *x, float *out, ink2_cache *c) {
    ink2_moe_w M;
    ink2_status st = ink2_moe_bind(s, w, block, &M);
    if (st != INK2_OK) return st;
    int H = s->hidden, E = s->n_expert, F = s->ff;
    if (M.t13) {   // the fast route: every position's routing, then the experts grouped
        if (T == 1 && E <= 256) {
            float sc[256], wt[256];
            unsigned char chosen[256];
            ink2_route_one(s, &M, x, NULL, sc, chosen, wt);
            return ink2_experts_fast_c(s, &M, 1, x, chosen, wt, out, c);
        }
        // R4.48.20.4: a position's scores on the stack (E <= 256, as above), the T positions' tables (wts [T][E]
        // floats, then ch [T][E] bytes) in the instance's scratch, so a layer allocates nothing a call
        float scb[512];
        unsigned char chb[256];
        const bool small = E <= 256;
        float *sc = small ? scb : malloc(sizeof(float) * 2 * (size_t)E);
        unsigned char *chosen = small ? chb : malloc((size_t)E);
        if (!small) atomic_fetch_add(&ink2_moe_alloc_n, 2);
        const size_t tw = (size_t)T * (size_t)E, tb = (tw + sizeof(float) - 1) / sizeof(float);
        int owned = 0;
        float *wts = ink2_route_alloc(c, tw + tb, &owned);
        unsigned char *ch = wts ? (unsigned char *)(wts + tw) : NULL;
        if (!sc || !chosen || !wts) st = INK2_BAD_INPUT;
        float *wt = sc ? sc + E : NULL;
        for (int t = 0; t < T && st == INK2_OK; t++) {
            ink2_route_one(s, &M, x + (size_t)t * H, NULL, sc, chosen, wt);
            memcpy(ch + (size_t)t * E, chosen, (size_t)E);
            memcpy(wts + (size_t)t * E, wt, sizeof(float) * (size_t)E);
        }
        if (st == INK2_OK) st = ink2_experts_fast_c(s, &M, T, x, ch, wts, out, c);
        ink2_op_free(wts, owned);
        if (!small) { free(sc); free(chosen); }
        return st;
    }
    float *sc = malloc(sizeof(float) * (2 * (size_t)E + 2 * (size_t)F + (size_t)H));
    unsigned char *chosen = malloc((size_t)E);
    if (!sc || !chosen) { free(sc); free(chosen); return INK2_BAD_INPUT; }
    float *wt = sc + E, *act = wt + E, *down = act + 2 * F;
    for (int t = 0; t < T && st == INK2_OK; t++) {
        ink2_route_one(s, &M, x + (size_t)t * H, NULL, sc, chosen, wt);
        st = ink2_experts_one(s, &M, x + (size_t)t * H, chosen, wt, act, down, out + (size_t)t * H);
    }
    free(sc); free(chosen);
    return st;
}
ink2_status ink2_expert_layer(const ink2_shape *s, const ink2_weights *w, int block, int T,
                              const float *x, float *out) {
    return ink2_expert_layer_c(s, w, block, T, x, out, NULL);
}

ink2_status ink2_expert_selection(const ink2_shape *s, const ink2_weights *w, int block, int T, const float *x, int *sel) {
    ink2_moe_w M;
    ink2_status st = ink2_moe_bind(s, w, block, &M);
    if (st != INK2_OK) return st;
    int H = s->hidden, E = s->n_expert, K = s->n_used;
    // a position's scores on the stack (E <= 256), as the layer's (R4.48.20.4)
    float scb[512];
    unsigned char chb[256];
    const bool small = E <= 256;
    float *sc = small ? scb : malloc(sizeof(float) * 2 * (size_t)E);
    unsigned char *chosen = small ? chb : malloc((size_t)E);
    if (!small) atomic_fetch_add(&ink2_moe_alloc_n, 2);
    if (!sc || !chosen) { if (!small) { free(sc); free(chosen); } return INK2_BAD_INPUT; }
    for (int t = 0; t < T; t++) {
        ink2_route_one(s, &M, x + (size_t)t * H, NULL, sc, chosen, sc + E);
        for (int e = 0, k = 0; e < E; e++) if (chosen[e]) sel[(size_t)t * K + k++] = e;
    }
    if (!small) { free(sc); free(chosen); }
    return INK2_OK;
}

// y[o] = sum_i W[o][i] x[i], a serial dot in index order per output (nn.Linear without bias); threads over o
typedef struct { const float *W, *x; float *y; int n_in; } ink2_mv_ctx;
static void ink2_matvec_range(void *c, int64_t o0, int64_t o1) {
    const ink2_mv_ctx *m = c; const int n_in = m->n_in;
    for (int64_t o = o0; o < o1; o++) {
        float a = 0.0f; const float *row = m->W + (size_t)o * n_in;
        for (int i = 0; i < n_in; i++) a += row[i] * m->x[i];
        m->y[o] = a;
    }
}
static void ink2_matvec(const float *W, int n_out, int n_in, const float *x, float *y) {
    ink2_mv_ctx m = { W, x, y, n_in };
    ink2_par(n_out, n_in, ink2_matvec_range, &m);
}
// The fast route's rows: threads over output rows; each row against the positions in groups of VEC_DOT_MULTI_MAX
// columns, every column the single-column vec_dot's bits (vec_dot_multi is byte-equal to it), so a row's value does
// not depend on how many positions the call holds, how they are grouped, or the thread count. The weight row is read
// in its file format and decoded in registers; the activations stay float32; the scale (an expert's factor) after the
// dot, as llama.cpp's graph applies it.
static void ink2_lin_range(void *c, int64_t o0, int64_t o1) {
    const ink2_lin_ctx *L = c;
    const unsigned char *base = (const unsigned char *)L->W->q;
    const int type = L->W->qtype, n_in = L->n_in;
    if (L->T >= VEC_DOT_EXACT_TILE_MIN && vec_dot_exact_tile_ok(type, n_in)) {
        vec_dot_exact_tile(type, base + (size_t)(L->row0 + o0) * L->W->q_row_bytes,
                           L->W->q_row_bytes, (int)(o1 - o0), L->x, L->xs,
                           L->T, n_in, L->scale, L->y + (size_t)o0, L->ys);
        return;
    }
    for (int64_t o = o0; o < o1; o++) {
        const void *row = base + (size_t)(L->row0 + o) * L->W->q_row_bytes;
        for (int t0 = 0; t0 < L->T; t0 += VEC_DOT_MULTI_MAX) {
            const int nc = L->T - t0 < VEC_DOT_MULTI_MAX ? L->T - t0 : VEC_DOT_MULTI_MAX;
            float v[VEC_DOT_MULTI_MAX];
            if (nc == 1) v[0] = vec_dot(type, row, L->x + (size_t)t0 * L->xs, n_in);
            else {
                const float *xs[VEC_DOT_MULTI_MAX];
                for (int k = 0; k < nc; k++) xs[k] = L->x + (size_t)(t0 + k) * L->xs;
                vec_dot_multi(type, row, xs, nc, n_in, v);
            }
            for (int k = 0; k < nc; k++) L->y[(size_t)(t0 + k) * L->ys + (size_t)o] = v[k] * L->scale;
        }
    }
}
static void ink2_lin(const ink2_tensor *W, int64_t row0, int n_out, int n_in, int T, const float *x, size_t x_stride,
                     float *y, size_t y_stride, float scale) {
    if (!W->q) {
        for (int t = 0; t < T; t++) ink2_matvec(W->data + (size_t)row0 * n_in, n_out, n_in, x + (size_t)t * x_stride, y + (size_t)t * y_stride);
        return;
    }
    float *xq = NULL;
    if (W->q_act8 && n_in % 32 == 0) {
        // TEST-ONLY, the 8-bit control: each activation row through Q8_0 and back, as llama.cpp's CPU product
        // quantizes its input before an integer dot
        const size_t qb = (size_t)(n_in / 32) * 34;
        unsigned char *blk = malloc(qb);
        xq = malloc(sizeof(float) * (size_t)T * n_in);
        if (!blk || !xq) abort();
        for (int t = 0; t < T; t++) {
            q8_quant_row(x + (size_t)t * x_stride, blk, n_in);
            dequant_row(T_Q8_0, blk, xq + (size_t)t * n_in, n_in);
        }
        free(blk);
        x = xq;
        x_stride = (size_t)n_in;
    }
    ink2_lin_ctx c = { W, row0, n_out, n_in, T, x, x_stride, y, y_stride, scale };
    ink2_par(n_out, (int64_t)n_in * T, ink2_lin_range, &c);
    free(xq);
}
typedef struct {
    int np;
    const ink2_lin_ctx *p;
    int64_t start[INK2_LIN_PARTS_MAX + 1];
} ink2_lin_parts_ctx;
static void ink2_lin_parts_range(void *c, int64_t i0, int64_t i1) {
    const ink2_lin_parts_ctx *mc = c;
    for (int s = 0; s < mc->np; s++) {
        int64_t a = i0 > mc->start[s] ? i0 : mc->start[s];
        int64_t b = i1 < mc->start[s + 1] ? i1 : mc->start[s + 1];
        if (a < b) ink2_lin_range((void *)&mc->p[s], a - mc->start[s], b - mc->start[s]);
    }
}
// Dispatch np independent fast linear projections in one ink2_par barrier (bits match calling ink2_lin per part).
static void ink2_lin_parts(const ink2_lin_ctx *parts, int np, int64_t work) {
    if (np <= 0) return;
    int can_fuse = (np > 1) && (np <= INK2_LIN_PARTS_MAX) && ink2_decode_fuse_enabled();
    for (int s = 0; s < np && can_fuse; s++) {
        if (!parts[s].W->q || parts[s].W->q_act8) can_fuse = 0;
    }
    if (!can_fuse) {
        for (int s = 0; s < np; s++) {
            const ink2_lin_ctx *p = &parts[s];
            ink2_lin(p->W, p->row0, p->n_out, p->n_in, p->T, p->x, p->xs, p->y, p->ys, p->scale);
        }
        return;
    }
    ink2_lin_parts_ctx mc;
    mc.np = np;
    mc.p = parts;
    mc.start[0] = 0;
    for (int s = 0; s < np; s++) mc.start[s + 1] = mc.start[s] + parts[s].n_out;
    if (mc.start[np] > 0) ink2_par(mc.start[np], work, ink2_lin_parts_range, &mc);
}
// Inkling2LoopL2Clip on one head vector: norm = sqrt(sum y^2) (vector_norm), then y * min(thr / (norm + 1e-6), 1),
// the quotient as torch computes a Python float over a tensor: f32(1 / (norm + 1e-6)) * f32(thr)
static void ink2_l2_clip(float *y, int n, float thr) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += y[i] * y[i];
    float f = (1.0f / (sqrtf(ss) + 1e-6f)) * thr;
    if (f > 1.0f) f = 1.0f;                               // .clamp(max=1.0)
    for (int i = 0; i < n; i++) y[i] = y[i] * f;
}
// Inkling2LoopLogScale's factor at one position: 1 + alpha * log(max((pos + 1) / n_floor, 1)), in float32
static float ink2_log_tau(const ink2_shape *s, int pos) {
    float a = (float)(pos + 1) / (float)s->log_floor;
    if (a < 1.0f) a = 1.0f;
    return 1.0f + s->log_alpha * p_logf(a);
}
// The head-vector norm of Q and of the relative states: times the learned scale [n][d], the L2 clip, and on full
// layers (Inkling2LoopLogScaledL2Clip) the log-length scale
static void ink2_scaled_clip(const ink2_shape *s, int g, float *y, const float *scale, int n, int d, float thr, int pos) {
    float tau = g ? ink2_log_tau(s, pos) : 1.0f;
    for (int h = 0; h < n; h++) {
        float *yh = y + (size_t)h * d;
        for (int i = 0; i < d; i++) yh[i] = yh[i] * scale[(size_t)h * d + i];
        ink2_l2_clip(yh, d, thr);
        if (g) for (int i = 0; i < d; i++) yh[i] = yh[i] * tau;
    }
}
// One layer's self_attn: its sizes (by layer type) and its weights by role, looked up once (pointers only: a lookup
// may reuse its slot)
enum { WQ, WQS, WK, WKC, WKD, WV, WVC, WVD, WR, WRS, WRP, WS, WG, WO, NW };
typedef struct { int g, nh, nkv, hd, K, ext, dr, C, Q; const float *wt[NW]; const ink2_tensor *tt[NW]; } ink2_attn_layer;
static ink2_status ink2_attn_bind(const ink2_shape *s, const ink2_weights *w, int app, ink2_attn_layer *L) {
    if (!s || !w || app < 0 || app >= s->n_app) return INK2_BAD_INPUT;
    int g = ink2_app_global(s, app), H = s->hidden;
    L->g = g; L->nh = g ? s->n_head : s->swa_n_head; L->nkv = g ? s->n_kv : s->swa_n_kv; L->hd = g ? s->head_dim : s->swa_head_dim;
    L->K = s->conv_k; L->ext = g ? s->rel_extent : s->window; L->dr = s->d_rel; L->C = L->nkv * L->hd; L->Q = L->nh * L->hd;
    int nh = L->nh, nkv = L->nkv, hd = L->hd, K = L->K, ext = L->ext, dr = L->dr, C = L->C, Q = L->Q;
    if (nkv <= 0 || nh % nkv || hd < 1 || K < 1 || ext < 1 || dr < 1) return INK2_BAD_INPUT;
    const char *nm[NW] = { "q_proj.linear.weight", "q_proj.norm.scale", "k_proj.linear.weight",
        g ? "k_proj.sconv.static_weight" : "k_proj.sconv.conv1d.weight", g ? "k_proj.sconv.weight_proj.weight" : NULL,
        "v_proj.linear.weight", g ? "v_proj.sconv.static_weight" : "v_proj.sconv.conv1d.weight",
        g ? "v_proj.sconv.weight_proj.weight" : NULL, "relative_position_bias.linear.weight",
        "relative_position_bias.norm.scale", "relative_position_bias.proj", "sink_proj.weight", "o_proj.gate_proj.weight",
        "o_proj.linear.weight" };
    // the expected ne[0] and element count of each
    const int64_t n0[NW] = { H, hd, H, g ? C : K, H, H, g ? C : K, H, H, dr, ext, H, H, Q };
    const int64_t nn[NW] = { (int64_t)H * Q, Q, (int64_t)H * C, (int64_t)C * K, (int64_t)H * nkv * K, (int64_t)H * C,
                             (int64_t)C * K, (int64_t)H * nkv * K, (int64_t)H * nh * dr, nh * dr, dr * ext, (int64_t)H * nh,
                             (int64_t)H * Q, (int64_t)Q * H };
    for (int i = 0; i < NW; i++) {
        L->wt[i] = NULL; L->tt[i] = NULL;
        if (!nm[i]) continue;
        char name[192]; snprintf(name, sizeof name, "model.layers.%d.self_attn.%s", s->app_block[app], nm[i]);
        const ink2_tensor *t = w->get(w->ctx, name);
        if (!t || t->ne[0] != n0[i] || t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] != nn[i]) return INK2_NO_TENSOR;
        if (!t->data && !t->q) return INK2_NO_TENSOR;
        L->wt[i] = t->data; L->tt[i] = t;
    }
    static const int need_f32[] = { WQS, WKC, WVC, WRS, WRP };
    for (size_t i = 0; i < sizeof need_f32 / sizeof *need_f32; i++)
        if (L->tt[need_f32[i]] && !L->wt[need_f32[i]]) return INK2_NO_TENSOR;
    return INK2_OK;
}
static int ink2_alloc_fail(const char *tag) {
    const char *e = getenv("RUNNER_TEST_INK2_ALLOC_FAIL");
    return e && *e && (!strcmp(e, "all") || !strcmp(e, tag));
}
static ink2_status ink2_lora_apply(const ink2_lora *lora, int block, int slot, int n_out, int n_in, int T,
                                   const float *x, float *y) {
    if (!lora || !lora->slot) return INK2_OK;
    const float *a = NULL, *b = NULL;
    int r = 0, ln_in = 0, ln_out = 0;
    float scale = 0.0f;
    if (!lora->slot(lora->ctx, block, slot, &a, &b, &r, &scale, &ln_in, &ln_out)) return INK2_OK;
    if (r <= 0 || !a || !b || scale == 0.0f) return INK2_OK;
    if (ln_in != n_in || ln_out != n_out || r > INK2_MAX_LORA_RANK) {
        fprintf(stderr, "error: inkling2_loop ink2_lora_apply: invalid adapter shape (rank %d, [%d,%d] vs [%d,%d])\n",
                r, ln_in, ln_out, n_in, n_out);
        return INK2_BAD_INPUT;
    }
    float tmp_stack[256];
    float *tmp = ink2_alloc_fail("lora_apply") ? NULL : (r <= 256 ? tmp_stack : malloc(sizeof(float) * (size_t)r));
    if (!tmp) {
        fprintf(stderr, "error: inkling2_loop ink2_lora_apply: out of memory allocating rank buffer\n");
        return INK2_BAD_INPUT;
    }
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * n_in;
        float *yt = y + (size_t)t * n_out;
        for (int k = 0; k < r; k++) {
            const float *ak = a + (size_t)k * n_in;
            float acc = 0.0f;
            for (int i = 0; i < n_in; i++) acc = fmaf(ak[i], xt[i], acc);
            tmp[k] = acc;
        }
        for (int j = 0; j < n_out; j++) {
            const float *bj = b + (size_t)j * r;
            float acc = 0.0f;
            for (int k = 0; k < r; k++) acc = fmaf(bj[k], tmp[k], acc);
            yt[j] = fmaf(scale, acc, yt[j]);
        }
    }
    if (tmp != tmp_stack) free(tmp);
    return INK2_OK;
}
// q_proj (Inkling2LoopQueryProjection): linear, then per head the scale, the L2 clip at qk_clip, (full) tau(pos)
static ink2_status ink2_attn_q_lora(const ink2_shape *s, const ink2_attn_layer *L, const ink2_lora *lora, int block, int T,
                                    const int *pos, const float *x, float *q) {
    ink2_lin_rows(L->tt[WQ], L->Q, s->hidden, T, x, q);
    ink2_status st = ink2_lora_apply(lora, block, 0, L->Q, s->hidden, T, x, q);
    if (st != INK2_OK) return st;
    for (int t = 0; t < T; t++) ink2_scaled_clip(s, L->g, q + (size_t)t * L->Q, L->wt[WQS], L->nh, L->hd, s->qk_clip, pos[t]);
    return INK2_OK;
}
static void ink2_attn_q(const ink2_shape *s, const ink2_attn_layer *L, int T, const int *pos, const float *x, float *q) {
    ink2_attn_q_lora(s, L, NULL, 0, T, pos, x, q);
}
// The convolution of k_proj / v_proj (kv = 0 / 1) over all T positions, lin [T][C] in, out [T][C]:
//   sliding (Inkling2LoopShortConvolution): conv[t][c] = sum_j w[c][j] lin[t - (K-1) + j][c], j ascending (F.conv1d,
//     left zero padding: w[c][K-1] weights the current position); (conv + lin) / sqrt(K + 1);
//   full (Inkling2LoopDynamicShortConvolution): taps[t][c][lag] = static[lag][c] + dyn[t][c / hd][lag], dyn = Wd x
//     (x is the projection's own input, the dynamic input); conv[t][c] = sum_lag lin[t - lag][c] taps[lag], lag
//     ascending; (conv + lin) / sqrt(2K + 1).
// The call covers positions p0..p0+T-1. Positions below 0 are the zero padding (skipped: no term); positions in
// p0-K+1..p0-1 come from the history ring hist (ink2_conv_hist; read only when p0 > 0, so a call at p0 = 0 is a
// prefill whatever the ring holds); the rest from lin. The same terms in the same order whether a position's input is
// current or historical, so a sequence fed in chunks gives the bits of the whole.
#define INK2_CONV_IN(u, ch) ((u) >= 0 ? lin[(size_t)(u) * C + (ch)] : hist[(size_t)((p0 + (u)) % (K - 1)) * C + (ch)])
static void ink2_kv_conv_finish(const ink2_shape *s, const ink2_attn_layer *L, int kv, int T, const float *lin,
                                float *dyn, float *out, int p0, const float *hist) {
    (void)s;
    int g = L->g, C = L->C, K = L->K, R = L->nkv * K, hd = L->hd;
    const float *cw = L->wt[kv ? WVC : WKC];
    const float div = (float)sqrt((double)(g ? 2 * K + 1 : K + 1));   // math.sqrt, a float32 scalar to torch
    if (!hist) p0 = 0;
    for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
        float a = 0.0f;
        if (!g) {
            for (int j = 0; j < K; j++) { int u = t - (K - 1) + j; if (p0 + u >= 0) a += cw[(size_t)ch * K + j] * INK2_CONV_IN(u, ch); }
        } else {
            for (int l = 0; l < K && p0 + t - l >= 0; l++) {
                float tap = cw[(size_t)l * C + ch] + dyn[(size_t)t * R + (size_t)(ch / hd) * K + l];
                a += INK2_CONV_IN(t - l, ch) * tap;
            }
        }
        out[(size_t)t * C + ch] = (a + lin[(size_t)t * C + ch]) / div;
    }
}
static void ink2_kv_conv(const ink2_shape *s, const ink2_attn_layer *L, int kv, int T, const float *lin, const float *x,
                         float *dyn, float *out, int p0, const float *hist) {
    if (L->g) ink2_lin_rows(L->tt[kv ? WVD : WKD], L->nkv * L->K, s->hidden, T, x, dyn);
    ink2_kv_conv_finish(s, L, kv, T, lin, dyn, out, p0, hist);
}
// Inkling2LoopShortConvolution on hidden-width states (post_attention_convolution, post_mlp_convolution): the
// sliding branch above with the layer's own taps w [C][K]: conv[t][c] = sum_j w[c][j] x[t - (K-1) + j][c], j
// ascending; (conv + x) / sqrt(K + 1); the history as above
static void ink2_short_conv(const float *cw, int C, int K, int T, const float *lin, float *out, int p0, const float *hist) {
    const float div = (float)sqrt((double)(K + 1));
    if (!hist) p0 = 0;
    for (int t = 0; t < T; t++) for (int ch = 0; ch < C; ch++) {
        float a = 0.0f;
        for (int j = 0; j < K; j++) { int u = t - (K - 1) + j; if (p0 + u >= 0) a += cw[(size_t)ch * K + j] * INK2_CONV_IN(u, ch); }
        out[(size_t)t * C + ch] = (a + lin[(size_t)t * C + ch]) / div;
    }
}
#undef INK2_CONV_IN
// the norm of k_proj (the L2 clip at qk_clip) or v_proj (the unweighted RMS norm), per head, in place
static void ink2_kv_norm(const ink2_shape *s, const ink2_attn_layer *L, int kv, int T, float *z) {
    for (int t = 0; t < T; t++) for (int h = 0; h < L->nkv; h++) {
        float *zh = z + (size_t)t * L->C + (size_t)h * L->hd;
        if (kv) ink2_unweighted_norm(zh, L->hd, s->eps, zh); else ink2_l2_clip(zh, L->hd, s->qk_clip);
    }
}
// linear, convolution (history hist, NULL for none), norm; the conv's input rows then go into the history
static ink2_status ink2_attn_kv_lora(const ink2_shape *s, const ink2_attn_layer *L, const ink2_lora *lora, int block, int kv,
                                     int T, const float *x, float *lin, float *dyn, float *out, int p0, float *hist) {
    ink2_lin_rows(L->tt[kv ? WV : WK], L->C, s->hidden, T, x, lin);
    ink2_status st = ink2_lora_apply(lora, block, kv ? 2 : 1, L->C, s->hidden, T, x, lin);
    if (st != INK2_OK) return st;
    ink2_kv_conv(s, L, kv, T, lin, x, dyn, out, p0, hist);
    ink2_kv_norm(s, L, kv, T, out);
    ink2_conv_hist_store(hist, L->K, L->C, p0, T, lin);
    return INK2_OK;
}
static void ink2_attn_kv(const ink2_shape *s, const ink2_attn_layer *L, int kv, int T, const float *x, float *lin,
                         float *dyn, float *out, int p0, float *hist) {
    ink2_attn_kv_lora(s, L, NULL, 0, kv, T, x, lin, dyn, out, p0, hist);
}
typedef struct { const float *rt, *proj; float *rl; int dr, ext; } ink2_bank_ctx;
static void ink2_bank_range(void *c, int64_t h0, int64_t h1) {
    const ink2_bank_ctx *b = c; const int dr = b->dr, ext = b->ext;
    for (int64_t h = h0; h < h1; h++) for (int e = 0; e < ext; e++) {
        float a = 0.0f;
        for (int r = 0; r < dr; r++) a += b->rt[(size_t)h * dr + r] * b->proj[(size_t)r * ext + e];
        b->rl[(size_t)h * ext + e] = a;
    }
}
// relative_position_bias up to the bank: rs = linear [heads][d_rel], the scaled clip at rel_clip ((full) with tau),
// rl[t][h][e] = sum_r rs[h][r] proj[r][e]: the bias of distance e, read by the core for 0 <= q - k < extent
static void ink2_attn_rel_finish(const ink2_shape *s, const ink2_attn_layer *L, int T, const int *pos, float *rs,
                                 float *rl) {
    int nh = L->nh, dr = L->dr, ext = L->ext;
    for (int t = 0; t < T; t++) {
        float *rt = rs + (size_t)t * nh * dr;
        ink2_scaled_clip(s, L->g, rt, L->wt[WRS], nh, dr, s->rel_clip, pos[t]);
        ink2_bank_ctx bc = { rt, L->wt[WRP], rl + (size_t)t * nh * ext, dr, ext };
        ink2_par(nh, (int64_t)ext * dr, ink2_bank_range, &bc);   // threads over heads
    }
}
static void ink2_attn_rel(const ink2_shape *s, const ink2_attn_layer *L, int T, const int *pos, const float *x, float *rs,
                          float *rl) {
    ink2_lin_rows(L->tt[WR], L->nh * L->dr, s->hidden, T, x, rs);
    ink2_attn_rel_finish(s, L, T, pos, rs, rl);
}
// eager_attention_forward with the sink, head by head, query by query: logits = (Q . K) * head_dim^-0.5 + bias over
// the keys the mask admits (k <= q, and on sliding layers q - k < window), the sink as one more logit with a zero
// value: softmax over [keys, sink] (max, exp, serial sum with the sink last, a division); probs = the keys' columns;
// att = probs V (keys ascending). Head h reads KV head h / (n_head / n_kv) (repeat_kv). Masked keys get probability 0
// (the reference adds the float minimum, whose exp is 0) and stay out of the sums (their zero terms change no bits).
// The keys are nk rows of k and v, row j at position kp0 + j (the application's cache rows 0..p0+T-1, or a recorded
// block); a query reads them in ascending position, so a query's arithmetic does not depend on how many queries the
// call holds. probs is [n_head][T][nk].
typedef struct {
    const ink2_shape *s; const ink2_attn_layer *L; int T; const int *pos; const float *q, *k, *v; int nk, kp0;
    const float *rl, *sk; float *lg, *att, *probs;
} ink2_core_ctx;
// heads h0..h1-1 (threads over heads; each head's arithmetic is the serial loop below, in its own logit row)
static void ink2_attn_core_range(void *cv, int64_t h0, int64_t h1) {
    const ink2_core_ctx *c = cv;
    const ink2_shape *s = c->s; const ink2_attn_layer *L = c->L;
    const int T = c->T, nk = c->nk, kp0 = c->kp0; const int *pos = c->pos;
    const float *q = c->q, *k = c->k, *v = c->v, *rl = c->rl, *sk = c->sk; float *att = c->att, *probs = c->probs;
    int nh = L->nh, hd = L->hd, C = L->C, Q = L->Q, ext = L->ext;
    float *lg = c->lg, *own = NULL;
    if (h0 != 0 || h1 != nh) { own = malloc(sizeof(float) * ((size_t)nk + 1)); if (!own) abort(); lg = own; }
    const float scale = ink2_inv_sqrt(hd);   // head_dim ** -0.5 (see ink2_inv_sqrt; no pow)
#define INK2_ADMIT(i, j) (pos[i] - (kp0 + (j)) >= 0 && (L->g || pos[i] - (kp0 + (j)) < s->window))
    for (int h = (int)h0; h < (int)h1; h++) {
        int kh = h / (nh / L->nkv);
        for (int i = 0; i < T; i++) {
            const float *qi = q + (size_t)i * Q + (size_t)h * hd;
            float mx = sk[(size_t)i * nh + h];
            for (int j = 0; j < nk; j++) {
                int d = pos[i] - (kp0 + j);
                lg[j] = 0.0f;
                if (!INK2_ADMIT(i, j)) continue;
                const float *kj = k + (size_t)j * C + (size_t)kh * hd;
                float a = 0.0f;
                for (int e = 0; e < hd; e++) a += qi[e] * kj[e];
                a = a * scale;
                lg[j] = a + (d < ext ? rl[((size_t)i * nh + h) * ext + d] : 0.0f);
                if (lg[j] > mx) mx = lg[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < nk; j++) {
                if (!INK2_ADMIT(i, j)) continue;   // lg[j] is 0 already
                lg[j] = p_expf(lg[j] - mx); sum += lg[j];
            }
            sum += p_expf(sk[(size_t)i * nh + h] - mx);   // the sink, last in the concatenation
            for (int j = 0; j < nk; j++) lg[j] = lg[j] / sum;
            if (probs) for (int j = 0; j < nk; j++) probs[((size_t)h * T + i) * nk + j] = lg[j];
            float *ai = att + (size_t)i * Q + (size_t)h * hd;
            for (int e = 0; e < hd; e++) {
                float a = 0.0f;
                for (int j = 0; j < nk; j++) if (INK2_ADMIT(i, j)) a += lg[j] * v[(size_t)j * C + (size_t)kh * hd + e];
                ai[e] = a;
            }
        }
    }
#undef INK2_ADMIT
    free(own);
}
static inline float ink2_dot_f32(const float *a, const float *b, int n) {
#if INK2_SIMD_AVX2
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int e = 0;
    for (; e + 16 <= n; e += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + e),     _mm256_loadu_ps(b + e),     s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + e + 8), _mm256_loadu_ps(b + e + 8), s1);
    }
    for (; e + 8 <= n; e += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + e), _mm256_loadu_ps(b + e), s0);
    float s = ink2_hsum8(_mm256_add_ps(s0, s1));
    for (; e < n; e++) s += a[e] * b[e];
    return s;
#elif INK2_SIMD_NEON
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int e = 0;
    for (; e + 8 <= n; e += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + e),     vld1q_f32(b + e));
        s1 = vfmaq_f32(s1, vld1q_f32(a + e + 4), vld1q_f32(b + e + 4));
    }
    for (; e + 4 <= n; e += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + e), vld1q_f32(b + e));
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; e < n; e++) s += a[e] * b[e];
    return s;
#else
    float s = 0.0f;
    for (int e = 0; e < n; e++) s += a[e] * b[e];
    return s;
#endif
}

// Fast-route attention core (Tier B): parallel over (head, query) outputs, contiguous admitted key range [j0, j1),
// SIMD QK dot, SIMD exp softmax, and j-outer tiled SIMD V accumulation.
static void ink2_attn_core_fast_range(void *cv, int64_t r0, int64_t r1) {
    const ink2_core_ctx *c = cv;
    const ink2_shape *s = c->s; const ink2_attn_layer *L = c->L;
    const int T = c->T, nk = c->nk, kp0 = c->kp0; const int *pos = c->pos;
    const float *q = c->q, *k = c->k, *v = c->v, *rl = c->rl, *sk = c->sk; float *att = c->att, *probs = c->probs;
    const int nh = L->nh, hd = L->hd, C = L->C, Q = L->Q, ext = L->ext, n_kv_rep = nh / L->nkv;
    float lg_stk[1024];
    float *own = nk > 1024 ? malloc(sizeof(float) * (size_t)nk) : NULL;
    if (nk > 1024 && !own) abort();
    float *lg = nk > 1024 ? own : lg_stk;
    const float scale = ink2_inv_sqrt(hd);
    for (int64_t idx = r0; idx < r1; idx++) {
        const int h = (int)(idx / T), i = (int)(idx % T), kh = h / n_kv_rep;
        int j1 = pos[i] - kp0 + 1;
        if (j1 > nk) j1 = nk;
        int j0 = L->g ? 0 : pos[i] - kp0 - s->window + 1;
        if (j0 < 0) j0 = 0;
        if (j0 > j1) j0 = j1;
        const float *qi = q + (size_t)i * Q + (size_t)h * hd;
        const float *k_head = k + (size_t)kh * hd;
        const float *rl_ih = rl + ((size_t)i * nh + h) * ext;
        float mx = sk[(size_t)i * nh + h];
        for (int j = j0; j < j1; j++) {
            const float a = ink2_dot_f32(qi, k_head + (size_t)j * C, hd) * scale;
            const int d = pos[i] - (kp0 + j);
            const float lj = a + (d < ext ? rl_ih[d] : 0.0f);
            lg[j] = lj;
            if (lj > mx) mx = lj;
        }
        float sum = 0.0f;
        int j = j0;
#if INK2_SIMD_AVX2
        const __m256 vmx = _mm256_set1_ps(mx);
        __m256 vsum = _mm256_setzero_ps();
        for (; j + 8 <= j1; j += 8) {
            __m256 ex = ink2_vexp8(_mm256_sub_ps(_mm256_loadu_ps(lg + j), vmx));
            _mm256_storeu_ps(lg + j, ex);
            vsum = _mm256_add_ps(vsum, ex);
        }
        sum = ink2_hsum8(vsum);
#elif INK2_SIMD_NEON
        const float32x4_t vmx = vdupq_n_f32(mx);
        float32x4_t vsum = vdupq_n_f32(0);
        for (; j + 4 <= j1; j += 4) {
            float32x4_t ex = ink2_vexp4(vsubq_f32(vld1q_f32(lg + j), vmx));
            vst1q_f32(lg + j, ex);
            vsum = vaddq_f32(vsum, ex);
        }
        sum = vaddvq_f32(vsum);
#endif
        for (; j < j1; j++) { const float ex = p_expf(lg[j] - mx); lg[j] = ex; sum += ex; }
        sum += p_expf(sk[(size_t)i * nh + h] - mx);
        const float inv_sum = 1.0f / sum;
        j = j0;
#if INK2_SIMD_AVX2
        const __m256 vinv = _mm256_set1_ps(inv_sum);
        for (; j + 8 <= j1; j += 8) _mm256_storeu_ps(lg + j, _mm256_mul_ps(_mm256_loadu_ps(lg + j), vinv));
#elif INK2_SIMD_NEON
        for (; j + 4 <= j1; j += 4) vst1q_f32(lg + j, vmulq_n_f32(vld1q_f32(lg + j), inv_sum));
#endif
        for (; j < j1; j++) lg[j] *= inv_sum;
        if (probs) {
            float *pr = probs + ((size_t)h * T + i) * nk;
            for (int z = 0; z < j0; z++) pr[z] = 0.0f;
            if (j1 > j0) memcpy(pr + j0, lg + j0, sizeof(float) * (size_t)(j1 - j0));
            for (int z = j1; z < nk; z++) pr[z] = 0.0f;
        }
        float *ai = att + (size_t)i * Q + (size_t)h * hd;
        const float *v_head = v + (size_t)kh * hd;
        int e = 0;
#if INK2_SIMD_AVX2
        for (; e + 32 <= hd; e += 32) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (int kkey = j0; kkey < j1; kkey++) {
                const __m256 pv = _mm256_set1_ps(lg[kkey]);
                const float *vj = v_head + (size_t)kkey * C + e;
                a0 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(vj),      a0);
                a1 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(vj + 8),  a1);
                a2 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(vj + 16), a2);
                a3 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(vj + 24), a3);
            }
            _mm256_storeu_ps(ai + e,      a0);
            _mm256_storeu_ps(ai + e + 8,  a1);
            _mm256_storeu_ps(ai + e + 16, a2);
            _mm256_storeu_ps(ai + e + 24, a3);
        }
        for (; e + 8 <= hd; e += 8) {
            __m256 a0 = _mm256_setzero_ps();
            for (int kkey = j0; kkey < j1; kkey++)
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(lg[kkey]), _mm256_loadu_ps(v_head + (size_t)kkey * C + e), a0);
            _mm256_storeu_ps(ai + e, a0);
        }
#elif INK2_SIMD_NEON
        for (; e + 16 <= hd; e += 16) {
            float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
            for (int kkey = j0; kkey < j1; kkey++) {
                const float32x4_t pv = vdupq_n_f32(lg[kkey]);
                const float *vj = v_head + (size_t)kkey * C + e;
                a0 = vfmaq_f32(a0, pv, vld1q_f32(vj));
                a1 = vfmaq_f32(a1, pv, vld1q_f32(vj + 4));
                a2 = vfmaq_f32(a2, pv, vld1q_f32(vj + 8));
                a3 = vfmaq_f32(a3, pv, vld1q_f32(vj + 12));
            }
            vst1q_f32(ai + e,      a0);
            vst1q_f32(ai + e + 4,  a1);
            vst1q_f32(ai + e + 8,  a2);
            vst1q_f32(ai + e + 12, a3);
        }
        for (; e + 4 <= hd; e += 4) {
            float32x4_t a0 = vdupq_n_f32(0);
            for (int kkey = j0; kkey < j1; kkey++)
                a0 = vfmaq_f32(a0, vdupq_n_f32(lg[kkey]), vld1q_f32(v_head + (size_t)kkey * C + e));
            vst1q_f32(ai + e, a0);
        }
#endif
        for (; e < hd; e++) {
            float a = 0.0f;
            for (int kkey = j0; kkey < j1; kkey++) a += lg[kkey] * v_head[(size_t)kkey * C + e];
            ai[e] = a;
        }
    }
    free(own);
}
static void ink2_attn_core(const ink2_shape *s, const ink2_attn_layer *L, int fast, int T, const int *pos, const float *q,
                           const float *k, const float *v, int nk, int kp0, const float *rl, const float *sk, float *lg,
                           float *att, float *probs) {
    ink2_core_ctx c = { s, L, T, pos, q, k, v, nk, kp0, rl, sk, lg, att, probs };
    if (fast) ink2_par((int64_t)T * L->nh, (int64_t)nk * 2 * L->hd, ink2_attn_core_fast_range, &c);
    else ink2_par(L->nh, (int64_t)T * nk * 2 * L->hd, ink2_attn_core_range, &c);
}
// o_proj (Inkling2LoopGatedOutputProjection): att times sigmoid(gate_proj(x)) (att is overwritten), then linear;
// ga is [T][Q] (each position's products are independent, so taking every gate first changes no bits); the LoRA
// adapter of attn_output reads the gated att, as before
static ink2_status ink2_attn_out_finish(const ink2_shape *s, const ink2_attn_layer *L, const ink2_lora *lora, int block,
                                        int T, float *att, const float *ga, float *out) {
    for (size_t i = 0; i < (size_t)T * L->Q; i++) att[i] = att[i] * ink2_sigmoid(ga[i]);
    ink2_lin_rows(L->tt[WO], s->hidden, L->Q, T, att, out);
    return ink2_lora_apply(lora, block, 3, s->hidden, L->Q, T, att, out);
}
static ink2_status ink2_attn_out_lora(const ink2_shape *s, const ink2_attn_layer *L, const ink2_lora *lora, int block, int T,
                                      const float *x, float *att, float *ga, float *out) {
    ink2_lin_rows(L->tt[WG], L->Q, s->hidden, T, x, ga);
    return ink2_attn_out_finish(s, L, lora, block, T, att, ga, out);
}
static void ink2_attn_out(const ink2_shape *s, const ink2_attn_layer *L, int T, const float *x, float *att, float *ga,
                          float *out) {
    ink2_attn_out_lora(s, L, NULL, 0, T, x, att, ga, out);
}
// scratch for T positions: q [T][Q], k, v, lin_k/v [T][C], dyn_k/v [T][n_kv K], rs [T][heads d_rel], rl [T][heads][extent],
// sinks [T][heads], att [T][Q], the logit row [nk + 1] (nk keys), the gate rows [T][Q]
enum { SQ, SK, SV, SLIN, SDYN, SLINV, SDYNV, SRS, SRL, SSK, SATT, SLG, SGA, NSCR };
static float *ink2_attn_scratch_c(ink2_cache *c, const ink2_attn_layer *L, int T, int nk, float **p, int *owned) {
    size_t n[NSCR] = { (size_t)T * L->Q, (size_t)T * L->C, (size_t)T * L->C,
                       (size_t)T * L->C, (size_t)T * L->nkv * L->K,
                       (size_t)T * L->C, (size_t)T * L->nkv * L->K,
                       (size_t)T * L->nh * L->dr, (size_t)T * L->nh * L->ext, (size_t)T * L->nh, (size_t)T * L->Q,
                       (size_t)nk + 1, (size_t)T * L->Q }, tot = 0;
    for (int i = 0; i < NSCR; i++) tot += n[i];
    float *buf = ink2_op_alloc(c, tot, owned);
    if (buf) { size_t o = 0; for (int i = 0; i < NSCR; i++) { p[i] = buf + o; o += n[i]; } }
    return buf;
}

// Inkling2LoopSlidingAttention (hybrid_sliding) / Inkling2LoopFullAttention (hybrid), in the reference's order:
// Q = q_proj(xq); K = k_proj(xk), V = v_proj(xv) (each: linear, convolution, norm); K, V into the application's
// cache slot at rows pos[t]; the relative bias and the sinks from xq; the core over the slot's rows 0..p0+T-1; o_proj
// gated by xq. The positions must run p0, p0+1, ... (p0 = pos[0]); rows below p0 and the K and V convolution
// histories are the slot's, read only when p0 > 0 (a call at p0 = 0 is a prefill).
static ink2_status ink2_attention_lora(const ink2_shape *s, const ink2_weights *w, const ink2_lora *lora, ink2_cache *c,
                                       int app, int T, const int *pos, const float *xq, const float *xk, const float *xv,
                                       float *out, float *probs) {
    ink2_attn_layer L;
    if (!c || app < 0 || app >= c->n_app || T <= 0 || T > c->n_ctx) return INK2_BAD_INPUT;
    if (c->max_chunk && T > c->max_chunk) return INK2_BAD_INPUT;   // the sliding buffers hold window - 1 + max_chunk
    int p0 = pos[0];
    for (int t = 0; t < T; t++) if (pos[t] != p0 + t || pos[t] < 0 || pos[t] >= c->n_ctx) return INK2_BAD_INPUT;
    // p0 = 0 starts over (nothing below it is read); otherwise p0 must be exactly where the slot ends: a rewind
    // (p0 < n_past) would read convolution rings that hold the newest rows, not the rows before p0 (see ink2_cache)
    if (p0 > 0 && p0 != c->n_past[app]) return INK2_REWIND;
    ink2_status st = ink2_attn_bind(s, w, app, &L);
    if (st != INK2_OK) return st;
    // the keys this call can admit: all of 0..p0+T-1 on a full layer; on a sliding one, from the first position
    // the earliest query still sees (p0 - window + 1). Keys before it are masked for every query, so leaving them
    // out changes no logit, no sum and no order (ink2_attn_core reads keys ascending and skips masked ones).
    // (only under ink2_cache_init_window: the full-size cache keeps every key, as the fixture's probs layout reads)
    int kp0 = (!c->max_chunk || L.g) ? 0 : (p0 - s->window + 1 > 0 ? p0 - s->window + 1 : 0);
    if (c->max_chunk && !L.g && c->test_drop_key && kp0 > 0) kp0++;   // the control: one admitted key lost
    int nk = p0 + T - kp0;
    if (p0 == 0) c->k0[app] = 0;
    if (p0 + T - c->k0[app] > c->rows[app]) {   // slide: keep kp0..p0-1 at the buffer's start
        if (kp0 < c->k0[app]) return INK2_BAD_INPUT;
        const size_t keep = (size_t)(p0 - kp0) * L.C, from = (size_t)(kp0 - c->k0[app]) * L.C;
        memmove(c->k[app], c->k[app] + from, sizeof(float) * keep);
        memmove(c->v[app], c->v[app] + from, sizeof(float) * keep);
        c->k0[app] = kp0;
        if (p0 + T - kp0 > c->rows[app]) return INK2_BAD_INPUT;
    }
    int buf_owned = 0;
    float *p[NSCR], *buf = ink2_attn_scratch_c(c, &L, T, nk, p, &buf_owned);
    if (!buf) return INK2_BAD_INPUT;
    int block = s->app_block[app];
    const int fast = L.tt[WQ] && L.tt[WQ]->q != NULL;
    ink2_rel_probe *rp = ink2_active_rel_probe();
    const int use_fused = fast && !lora && !rp && ink2_decode_fuse_enabled() && L.tt[WK]->q && L.tt[WV]->q &&
                          L.tt[WR]->q && L.tt[WS]->q && L.tt[WG]->q && (!L.g || (L.tt[WKD]->q && L.tt[WVD]->q));
    if (use_fused) {
        const size_t H = (size_t)s->hidden;
        ink2_lin_ctx ap[8];
        int nap = 0;
        ap[nap++] = (ink2_lin_ctx){ L.tt[WQ], 0, L.Q, s->hidden, T, xq, H, p[SQ], (size_t)L.Q, 1.0f };
        ap[nap++] = (ink2_lin_ctx){ L.tt[WK], 0, L.C, s->hidden, T, xk, H, p[SLIN], (size_t)L.C, 1.0f };
        if (L.g) ap[nap++] = (ink2_lin_ctx){ L.tt[WKD], 0, L.nkv * L.K, s->hidden, T, xk, H, p[SDYN], (size_t)(L.nkv * L.K), 1.0f };
        ap[nap++] = (ink2_lin_ctx){ L.tt[WV], 0, L.C, s->hidden, T, xv, H, p[SLINV], (size_t)L.C, 1.0f };
        if (L.g) ap[nap++] = (ink2_lin_ctx){ L.tt[WVD], 0, L.nkv * L.K, s->hidden, T, xv, H, p[SDYNV], (size_t)(L.nkv * L.K), 1.0f };
        ap[nap++] = (ink2_lin_ctx){ L.tt[WR], 0, L.nh * L.dr, s->hidden, T, xq, H, p[SRS], (size_t)(L.nh * L.dr), 1.0f };
        ap[nap++] = (ink2_lin_ctx){ L.tt[WS], 0, L.nh, s->hidden, T, xq, H, p[SSK], (size_t)L.nh, 1.0f };
        ap[nap++] = (ink2_lin_ctx){ L.tt[WG], 0, L.Q, s->hidden, T, xq, H, p[SGA], (size_t)L.Q, 1.0f };
        ink2_lin_parts(ap, nap, (int64_t)s->hidden * T);
        for (int t = 0; t < T; t++) ink2_scaled_clip(s, L.g, p[SQ] + (size_t)t * L.Q, L.wt[WQS], L.nh, L.hd, s->qk_clip, pos[t]);
        float *hk = ink2_conv_hist(s, c, app, 0), *hv = ink2_conv_hist(s, c, app, 1);
        ink2_kv_conv_finish(s, &L, 0, T, p[SLIN], p[SDYN], p[SK], p0, hk);
        ink2_kv_norm(s, &L, 0, T, p[SK]);
        ink2_conv_hist_store(hk, L.K, L.C, p0, T, p[SLIN]);
        ink2_kv_conv_finish(s, &L, 1, T, p[SLINV], p[SDYNV], p[SV], p0, hv);
        ink2_kv_norm(s, &L, 1, T, p[SV]);
        ink2_conv_hist_store(hv, L.K, L.C, p0, T, p[SLINV]);
        for (int t = 0; t < T; t++) {
            memcpy(c->k[app] + (size_t)(pos[t] - c->k0[app]) * L.C, p[SK] + (size_t)t * L.C, sizeof(float) * (size_t)L.C);
            memcpy(c->v[app] + (size_t)(pos[t] - c->k0[app]) * L.C, p[SV] + (size_t)t * L.C, sizeof(float) * (size_t)L.C);
        }
        c->n_past[app] = p0 + T;
        ink2_attn_rel_finish(s, &L, T, pos, p[SRS], p[SRL]);
        const size_t koff = (size_t)(kp0 - c->k0[app]) * L.C;
        ink2_attn_core(s, &L, fast, T, pos, p[SQ], c->k[app] + koff, c->v[app] + koff, nk, kp0, p[SRL], p[SSK], p[SLG],
                       p[SATT], probs);
        st = ink2_attn_out_finish(s, &L, NULL, block, T, p[SATT], p[SGA], out);
        ink2_op_free(buf, buf_owned);
        return st;
    }
    st = ink2_attn_q_lora(s, &L, lora, block, T, pos, xq, p[SQ]);
    if (st == INK2_OK) st = ink2_attn_kv_lora(s, &L, lora, block, 0, T, xk, p[SLIN], p[SDYN], p[SK], p0, ink2_conv_hist(s, c, app, 0));
    if (st == INK2_OK) st = ink2_attn_kv_lora(s, &L, lora, block, 1, T, xv, p[SLIN], p[SDYN], p[SV], p0, ink2_conv_hist(s, c, app, 1));
    if (st != INK2_OK) { ink2_op_free(buf, buf_owned); return st; }
    for (int t = 0; t < T; t++) {
        memcpy(c->k[app] + (size_t)(pos[t] - c->k0[app]) * L.C, p[SK] + (size_t)t * L.C, sizeof(float) * (size_t)L.C);
        memcpy(c->v[app] + (size_t)(pos[t] - c->k0[app]) * L.C, p[SV] + (size_t)t * L.C, sizeof(float) * (size_t)L.C);
    }
    c->n_past[app] = p0 + T;   // what the slot now holds consistently: rows and histories of 0..p0+T-1
    if (rp && app == rp->app) {
        size_t n_el = (size_t)T * (size_t)s->hidden;
        rp->T = T; rp->H = s->hidden;
        if (rp->xq_in) memcpy(rp->xq_in, xq, sizeof(float) * n_el);
        if (rp->eps != 0.0f && rp->d) {
            float *xq_rel = malloc(sizeof(float) * n_el);
            if (!xq_rel) {
                fprintf(stderr, "error: inkling2_loop ink2_attention_lora: out of memory allocating rel_probe buffer\n");
                ink2_op_free(buf, buf_owned);
                return INK2_BAD_INPUT;
            }
            for (size_t i = 0; i < n_el; i++) xq_rel[i] = xq[i] + rp->eps * rp->d[i];
            ink2_attn_rel(s, &L, T, pos, xq_rel, p[SRS], p[SRL]);
            free(xq_rel);
        } else {
            ink2_attn_rel(s, &L, T, pos, xq, p[SRS], p[SRL]);
        }
    } else {
        ink2_attn_rel(s, &L, T, pos, xq, p[SRS], p[SRL]);
    }
    ink2_lin_rows(L.tt[WS], L.nh, s->hidden, T, xq, p[SSK]);
    const size_t koff = (size_t)(kp0 - c->k0[app]) * L.C;
    ink2_attn_core(s, &L, fast, T, pos, p[SQ], c->k[app] + koff, c->v[app] + koff, nk, kp0, p[SRL], p[SSK], p[SLG], p[SATT],
                   probs);
    st = ink2_attn_out_lora(s, &L, lora, block, T, xq, p[SATT], p[SGA], out);
    ink2_op_free(buf, buf_owned);
    return st;
}
ink2_status ink2_attention(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                           const int *pos, const float *xq, const float *xk, const float *xv,
                           float *out, float *probs) {
    return ink2_attention_lora(s, w, NULL, c, app, T, pos, xq, xk, xv, out, probs);
}

// The sub-steps one at a time, for the fixture harness (which declares it; inkling2.h lists only the pieces): step is
// the reference's submodule under self_attn, in its recorded inputs, the same code as ink2_attention; layouts are the
// piece's ([T][heads][dim] per head):
//   "q_proj"                        in[0] x [T][H]                    -> out [T][n_head][hd]
//   "q_proj.norm"                   in[0] [T][n_head][hd]             -> out, same
//   "k_proj", "v_proj"              in[0] x [T][H]                    -> out [T][n_kv][hd]
//   "k_proj.sconv", "v_proj.sconv"  in[0] lin [T][n_kv hd], in[1] the dynamic input [T][H] -> out [T][n_kv hd]
//   "k_proj.norm", "v_proj.norm"    in[0] [T][n_kv][hd]               -> out, same
//   "relative_position_bias"        in[0] x                           -> out [n_head][T][T] (q - k outside the extent: 0)
//   "relative_position_bias.norm"   in[0] [T][n_head][d_rel]          -> out, same
//   "sink_proj"                     in[0] x                           -> out [T][n_head]
//   "core"                          in[0] q [T][n_head][hd], in[1] k, in[2] v [T][n_kv][hd], in[3] the bias
//                                   [n_head][T][T], in[4] sinks [T][n_head] -> out [T][n_head hd], out2 probs [n_head][T][T]
//   "o_proj"                        in[0] the attention output [T][n_head hd], in[1] x -> out [T][H]
ink2_status ink2_attention_step(const ink2_shape *s, const ink2_weights *w, int app, const char *step, int T,
                                const int *pos, const float *const *in, float *out, float *out2);
ink2_status ink2_attention_step(const ink2_shape *s, const ink2_weights *w, int app, const char *step, int T,
                                const int *pos, const float *const *in, float *out, float *out2) {
    ink2_attn_layer L;
    if (!step || T <= 0) return INK2_BAD_INPUT;
    ink2_status st = ink2_attn_bind(s, w, app, &L);
    if (st != INK2_OK) return st;
    int buf_owned = 0;
    float *p[NSCR], *buf = ink2_attn_scratch_c(NULL, &L, T, T, p, &buf_owned);
    if (!buf) return INK2_BAD_INPUT;
    int nh = L.nh, ext = L.ext, kv = step[0] == 'v';
    if (!strcmp(step, "q_proj")) ink2_attn_q(s, &L, T, pos, in[0], out);
    else if (!strcmp(step, "q_proj.norm")) {
        memcpy(out, in[0], sizeof(float) * (size_t)T * L.Q);
        for (int t = 0; t < T; t++) ink2_scaled_clip(s, L.g, out + (size_t)t * L.Q, L.wt[WQS], nh, L.hd, s->qk_clip, pos[t]);
    } else if (!strcmp(step, "k_proj") || !strcmp(step, "v_proj")) ink2_attn_kv(s, &L, kv, T, in[0], p[SLIN], p[SDYN], out, 0, NULL);
    else if (!strcmp(step, "k_proj.sconv") || !strcmp(step, "v_proj.sconv")) ink2_kv_conv(s, &L, kv, T, in[0], in[1], p[SDYN], out, 0, NULL);
    else if (!strcmp(step, "k_proj.norm") || !strcmp(step, "v_proj.norm")) {
        memcpy(out, in[0], sizeof(float) * (size_t)T * L.C); ink2_kv_norm(s, &L, kv, T, out);
    } else if (!strcmp(step, "relative_position_bias")) {
        ink2_attn_rel(s, &L, T, pos, in[0], p[SRS], p[SRL]);
        for (int h = 0; h < nh; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) {
            int d = pos[i] - pos[j];
            out[((size_t)h * T + i) * T + j] = d >= 0 && d < ext ? p[SRL][((size_t)i * nh + h) * ext + d] : 0.0f;
        }
    } else if (!strcmp(step, "relative_position_bias.norm")) {
        memcpy(out, in[0], sizeof(float) * (size_t)T * nh * L.dr);
        for (int t = 0; t < T; t++) ink2_scaled_clip(s, L.g, out + (size_t)t * nh * L.dr, L.wt[WRS], nh, L.dr, s->rel_clip, pos[t]);
    } else if (!strcmp(step, "sink_proj")) ink2_lin_rows(L.tt[WS], nh, s->hidden, T, in[0], out);
    else if (!strcmp(step, "core")) {
        // the bias matrix back to the bank form the core reads: rl[i][h][q - k] (distances with no key stay 0)
        memset(p[SRL], 0, sizeof(float) * (size_t)T * nh * ext);
        for (int h = 0; h < nh; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) {
            int d = pos[i] - pos[j];
            if (d >= 0 && d < ext) p[SRL][((size_t)i * nh + h) * ext + d] = in[3][((size_t)h * T + i) * T + j];
        }
        ink2_attn_core(s, &L, 0, T, pos, in[0], in[1], in[2], T, pos[0], p[SRL], in[4], p[SLG], out, out2);
    } else if (!strcmp(step, "o_proj")) {
        memcpy(p[SATT], in[0], sizeof(float) * (size_t)T * L.Q); ink2_attn_out(s, &L, T, in[1], p[SATT], p[SGA], out);
    } else st = INK2_BAD_INPUT;
    ink2_op_free(buf, buf_owned);
    return st;
}

typedef struct { const float *W, *xt; float *lg; int H; float cap; } ink2_head_ctx;
static void ink2_head_range(void *c, int64_t v0, int64_t v1) {
    const ink2_head_ctx *h = c;
    for (int64_t v = v0; v < v1; v++) {
        float a = 0.0f; const float *row = h->W + (size_t)v * h->H;
        for (int i = 0; i < h->H; i++) a += row[i] * h->xt[i];
        h->lg[v] = ink2_softcap(a, h->cap);
    }
}
static _Atomic long ink2_head_rows_count = 0;
long ink2_head_rows(void) { return atomic_load_explicit(&ink2_head_rows_count, memory_order_relaxed); }

static _Atomic uint32_t ink2_test_chunk_bump_bits = 0;
void ink2_set_test_chunk_bump(float delta) {
    uint32_t bits;
    memcpy(&bits, &delta, sizeof bits);
    atomic_store_explicit(&ink2_test_chunk_bump_bits, bits, memory_order_relaxed);
}

// The head: lm_head (built UNPADDED by the reference: vocab_used rows) then logit_softcap, tanh(l / cap) * cap
ink2_status ink2_head(const ink2_shape *s, const ink2_weights *w, int T, const float *x, float *logits) {
    const ink2_tensor *wo = w->get(w->ctx, "lm_head.weight");
    int H = s->hidden, V = s->vocab_used;
    if (!wo || wo->ne[0] != H || wo->ne[1] < V) return INK2_NO_TENSOR;
    if (wo->q) {   // the fast route: the head's rows as a view, then the softcap
        atomic_fetch_add_explicit(&ink2_head_rows_count, (long)T, memory_order_relaxed);
        ink2_lin(wo, 0, V, H, T, x, (size_t)H, logits, (size_t)V, 1.0f);
        for (size_t i = 0; i < (size_t)T * V; i++) logits[i] = ink2_softcap(logits[i], s->final_cap);
        return INK2_OK;
    }
    if (!wo->data) return INK2_NO_TENSOR;
    atomic_fetch_add_explicit(&ink2_head_rows_count, (long)T, memory_order_relaxed);
    for (int t = 0; t < T; t++) {   // rows from vocab_used on, if a layout keeps them, never enter
        ink2_head_ctx hc = { wo->data, x + (size_t)t * H, logits + (size_t)t * V, H, s->final_cap };
        ink2_par(V, H, ink2_head_range, &hc);   // threads over vocabulary rows
    }
    return INK2_OK;
}

static ink2_status ink2_block_read_c(const ink2_shape *s, const ink2_weights *w, const char *which, int T, int n_blocks,
                                     const float *block_states, const float *partial, float *out, int *n_out,
                                     ink2_cache *c) {
    char name[160];
    snprintf(name, sizeof name, "%s.query", which);
    const ink2_tensor *q = w->get(w->ctx, name);
    if (!q || q->ne[0] != s->hidden) return INK2_NO_TENSOR;
    int H = s->hidden, nq = q->n_dims == 1 ? 1 : (int)q->ne[1], nkv = n_blocks + 1, owned = 0;
    float *st = ink2_op_alloc(c, (size_t)nkv * H * 2 + (size_t)nkv + (size_t)nq * H, &owned);
    if (!st) return INK2_BAD_INPUT;
    float *normed = st + (size_t)nkv * H, *scores = normed + (size_t)nkv * H, *o = scores + nkv;
    for (int t = 0; t < T; t++) {
        memcpy(st, block_states + (size_t)t * n_blocks * H, sizeof(float) * (size_t)n_blocks * H);
        memcpy(st + (size_t)n_blocks * H, partial + (size_t)t * H, sizeof(float) * (size_t)H);
        ink2_depth_attn(q->data, nq, st, nkv, H, normed, scores, o);
        for (int i = 0; i < nq; i++) memcpy(out + ((size_t)i * T + t) * H, o + (size_t)i * H, sizeof(float) * (size_t)H);
    }
    ink2_op_free(st, owned);
    if (n_out) *n_out = nq;
    return INK2_OK;
}
ink2_status ink2_block_read(const ink2_shape *s, const ink2_weights *w, const char *which, int T, int n_blocks,
                            const float *block_states, const float *partial, float *out, int *n_out) {
    return ink2_block_read_c(s, w, which, T, n_blocks, block_states, partial, out, n_out, NULL);
}

static ink2_status ink2_message_input_c(const ink2_shape *s, const ink2_weights *w, int loop_layer, int T, int n_prev,
                                        const float *prev_block_states, const float *x, float *out, ink2_cache *c) {
    const ink2_tensor *q = w->get(w->ctx, "model.message_input.query");   // [loop layers][4][H]
    int H = s->hidden, nloop = s->loop_end - s->loop_start, nkv = n_prev + 1, owned = 0;
    if (!q || q->ne[0] != H || q->ne[1] != 4 || q->ne[2] != nloop || loop_layer < 0 || loop_layer >= nloop)
        return INK2_NO_TENSOR;
    const float *ql = q->data + (size_t)loop_layer * 4 * H;
    float *st = ink2_op_alloc(c, (size_t)nkv * H * 2 + (size_t)nkv, &owned);
    if (!st) return INK2_BAD_INPUT;
    float *normed = st + (size_t)nkv * H, *scores = normed + (size_t)nkv * H;
    for (int t = 0; t < T; t++) {
        memcpy(st, prev_block_states + (size_t)t * n_prev * H, sizeof(float) * (size_t)n_prev * H);
        memcpy(st + (size_t)n_prev * H, x + (size_t)t * H, sizeof(float) * (size_t)H);
        ink2_depth_attn(ql, 4, st, nkv, H, normed, scores, out + (size_t)t * 4 * H);   // [T][4][H]
    }
    ink2_op_free(st, owned);
    return INK2_OK;
}
ink2_status ink2_message_input(const ink2_shape *s, const ink2_weights *w, int loop_layer, int T, int n_prev,
                               const float *prev_block_states, const float *x, float *out) {
    return ink2_message_input_c(s, w, loop_layer, T, n_prev, prev_block_states, x, out, NULL);
}

typedef struct { const float *wf, *wg, *d, *xt, *mt; float *ot; int r; } ink2_gate_ctx;
static void ink2_gate_range(void *c, int64_t h0, int64_t h1) {
    const ink2_gate_ctx *q = c; const int r = q->r;
    for (int64_t h = h0; h < h1; h++) {
        float fa = 0.0f, ga = 0.0f;
        const float *fr = q->wf + (size_t)h * r, *gr = q->wg + (size_t)h * r;
        for (int k = 0; k < r; k++) { fa += fr[k] * q->d[k]; ga += gr[k] * q->d[k]; }
        float f = p_tanhf(fa), g = p_tanhf(ga);
        q->ot[h] = (1.0f - f) * q->xt[h] + g * q->mt[h];
    }
}
typedef struct {
    int n_roles, T, H;
    size_t msg_stride;
    const float *const *xs, *const *msgs;
    float *gh, eps;
} ink2_msg_norm_ctx;
static void ink2_msg_norm_range(void *c, int64_t t0, int64_t t1) {
    const ink2_msg_norm_ctx *m = c;
    const int H = m->H;
    for (int r_i = 0; r_i < m->n_roles; r_i++) {
        for (int64_t t = t0; t < t1; t++) {
            float *dst = m->gh + ((size_t)r_i * m->T + (size_t)t) * 2 * H;
            ink2_unweighted_norm(m->xs[r_i] + (size_t)t * H, H, m->eps, dst);
            ink2_unweighted_norm(m->msgs[r_i] + (size_t)t * m->msg_stride, H, m->eps, dst + H);
        }
    }
}
typedef struct { const float *fa, *ga, *x, *msg; float *out; } ink2_msg_comb_ctx;
typedef struct {
    int np;
    ink2_msg_comb_ctx c[3];
    int64_t start[4];
} ink2_msg_comb_multi_ctx;
static void ink2_msg_comb_multi_range(void *c, int64_t i0, int64_t i1) {
    const ink2_msg_comb_multi_ctx *mc = c;
    for (int r_i = 0; r_i < mc->np; r_i++) {
        int64_t a = i0 > mc->start[r_i] ? i0 : mc->start[r_i];
        int64_t b = i1 < mc->start[r_i + 1] ? i1 : mc->start[r_i + 1];
        const ink2_msg_comb_ctx *m = &mc->c[r_i];
        for (int64_t k = a - mc->start[r_i]; k < b - mc->start[r_i]; k++)
            m->out[k] = (1.0f - p_tanhf(m->fa[k])) * m->x[k] + p_tanhf(m->ga[k]) * m->msg[k];
    }
}
static ink2_status ink2_message_steps_c(const ink2_shape *s, const ink2_weights *w, int block, const char *const *roles,
                                        int n_roles, int T, const float *const *xs, const float *const *msgs,
                                        size_t msg_stride, float *const *outs, ink2_cache *c) {
    const ink2_tensor *wd[3], *wf[3], *wg[3];
    int H = s->hidden, r = s->loop_gate_rank, fast = 1, has_act8 = 0;
    for (int r_i = 0; r_i < n_roles; r_i++) {
        char n1[160], n2[160], n3[160];
        snprintf(n1, sizeof n1, "model.layers.%d.%s_message_gate.down_proj.weight", block, roles[r_i]);
        snprintf(n2, sizeof n2, "model.layers.%d.%s_message_gate.forget_gate_proj.weight", block, roles[r_i]);
        snprintf(n3, sizeof n3, "model.layers.%d.%s_message_gate.message_gate_proj.weight", block, roles[r_i]);
        wd[r_i] = w->get(w->ctx, n1); wf[r_i] = w->get(w->ctx, n2); wg[r_i] = w->get(w->ctx, n3);
        if (!wd[r_i] || !wf[r_i] || !wg[r_i] || wd[r_i]->ne[0] != 2 * H || wd[r_i]->ne[1] != r ||
            wf[r_i]->ne[0] != r || wf[r_i]->ne[1] != H || wg[r_i]->ne[0] != r || wg[r_i]->ne[1] != H)
            return INK2_NO_TENSOR;
        if (!wd[r_i]->q || !wf[r_i]->q || !wg[r_i]->q) {
            fast = 0;
            if (!wd[r_i]->data || !wf[r_i]->data || !wg[r_i]->data) return INK2_NO_TENSOR;
        } else if (wd[r_i]->q_act8 || wf[r_i]->q_act8 || wg[r_i]->q_act8) {
            has_act8 = 1;
        }
    }
    const size_t TH = (size_t)T * H;
    if (fast && !has_act8 && (n_roles == 1 || ink2_decode_fuse_enabled())) {
        const int need_unpack = (T > 1 && msg_stride != (size_t)H);
        const size_t t_gh = TH * 2, t_d = (size_t)T * r, t_fg = TH;
        const size_t per_role = t_gh + t_d + 2 * t_fg + (need_unpack ? TH : 0);
        int owned = 0;
        float *buf = ink2_op_alloc(c, (size_t)n_roles * per_role, &owned);
        if (!buf) return INK2_BAD_INPUT;
        float *gh = buf;
        float *d = gh + (size_t)n_roles * t_gh;
        float *fa = d + (size_t)n_roles * t_d;
        float *ga = fa + (size_t)n_roles * t_fg;
        float *mu = need_unpack ? ga + (size_t)n_roles * t_fg : NULL;
        ink2_msg_norm_ctx nc = { n_roles, T, H, msg_stride, xs, msgs, gh, s->eps };
        if (T == 1) ink2_msg_norm_range(&nc, 0, 1);
        else ink2_par(T, (int64_t)n_roles * 4 * H, ink2_msg_norm_range, &nc);
        if (need_unpack) {
            for (int r_i = 0; r_i < n_roles; r_i++)
                for (int t = 0; t < T; t++)
                    memcpy(mu + (size_t)r_i * TH + (size_t)t * H, msgs[r_i] + (size_t)t * msg_stride, sizeof(float) * H);
        }
        ink2_lin_ctx dp[3], fp[6];
        for (int r_i = 0; r_i < n_roles; r_i++) {
            float *d_r = d + (size_t)r_i * t_d;
            dp[r_i] = (ink2_lin_ctx){ wd[r_i], 0, r, 2 * H, T, gh + (size_t)r_i * t_gh, (size_t)2 * H, d_r, (size_t)r, 1.0f };
            fp[2 * r_i]     = (ink2_lin_ctx){ wf[r_i], 0, H, r, T, d_r, (size_t)r, fa + (size_t)r_i * t_fg, (size_t)H, 1.0f };
            fp[2 * r_i + 1] = (ink2_lin_ctx){ wg[r_i], 0, H, r, T, d_r, (size_t)r, ga + (size_t)r_i * t_fg, (size_t)H, 1.0f };
        }
        ink2_lin_parts(dp, n_roles, (int64_t)2 * H * T);
        ink2_lin_parts(fp, 2 * n_roles, (int64_t)r * T);
        ink2_msg_comb_multi_ctx cc;
        cc.np = n_roles;
        cc.start[0] = 0;
        for (int r_i = 0; r_i < n_roles; r_i++) {
            const float *mr = need_unpack ? (mu + (size_t)r_i * TH) : msgs[r_i];
            cc.c[r_i] = (ink2_msg_comb_ctx){ fa + (size_t)r_i * t_fg, ga + (size_t)r_i * t_fg, xs[r_i], mr, outs[r_i] };
            cc.start[r_i + 1] = cc.start[r_i] + (int64_t)TH;
        }
        ink2_par(cc.start[n_roles], 16, ink2_msg_comb_multi_range, &cc);
        ink2_op_free(buf, owned);
        return INK2_OK;
    }
    if (n_roles > 1) {
        const int need_unpack = (T > 1 && msg_stride != (size_t)H);
        int owned = 0;
        float *mu = need_unpack ? ink2_op_alloc(c, TH, &owned) : NULL;
        if (need_unpack && !mu) return INK2_BAD_INPUT;
        ink2_status st = INK2_OK;
        for (int r_i = 0; r_i < n_roles && st == INK2_OK; r_i++) {
            const float *mr = msgs[r_i];
            if (need_unpack) {
                for (int t = 0; t < T; t++) memcpy(mu + (size_t)t * H, msgs[r_i] + (size_t)t * msg_stride, sizeof(float) * H);
                mr = mu;
            }
            st = ink2_message_steps_c(s, w, block, &roles[r_i], 1, T, &xs[r_i], &mr, (size_t)H, &outs[r_i], need_unpack ? NULL : c);
        }
        ink2_op_free(mu, owned);
        return st;
    }
    if (fast) {
        // fast single-role fallback when q_act8 is enabled
        const size_t t_gh = TH * 2, t_d = (size_t)T * r, t_fg = TH;
        const size_t need_unpack = (T > 1 && msg_stride != (size_t)H) ? TH : 0;
        int owned = 0;
        float *buf = ink2_op_alloc(c, t_gh + t_d + 2 * t_fg + need_unpack, &owned);
        if (!buf) return INK2_BAD_INPUT;
        float *gh = buf, *d = gh + t_gh, *fa = d + t_d, *ga = fa + t_fg;
        float *mu = need_unpack ? ga + t_fg : NULL;
        ink2_msg_norm_ctx nc = { 1, T, H, msg_stride, xs, msgs, gh, s->eps };
        ink2_par(T, 4 * (int64_t)H, ink2_msg_norm_range, &nc);
        const float *mr = msgs[0];
        if (need_unpack) {
            for (int t = 0; t < T; t++) memcpy(mu + (size_t)t * H, msgs[0] + (size_t)t * msg_stride, sizeof(float) * H);
            mr = mu;
        }
        ink2_lin_rows(wd[0], r, 2 * H, T, gh, d);
        ink2_lin_rows(wf[0], H, r, T, d, fa);
        ink2_lin_rows(wg[0], H, r, T, d, ga);
        ink2_msg_comb_multi_ctx cc = { 1, { { fa, ga, xs[0], mr, outs[0] } }, { 0, (int64_t)TH } };
        ink2_par((int64_t)TH, 16, ink2_msg_comb_multi_range, &cc);
        ink2_op_free(buf, owned);
        return INK2_OK;
    }
    int owned = 0;
    float *gh = ink2_op_alloc(c, (size_t)2 * H + (size_t)r, &owned);
    if (!gh) return INK2_BAD_INPUT;
    float *d = gh + 2 * H;
    for (int t = 0; t < T; t++) {
        const float *xt = xs[0] + (size_t)t * H, *mt = msgs[0] + (size_t)t * msg_stride;
        ink2_unweighted_norm(xt, H, s->eps, gh);          // cat([norm(x), norm(message)])
        ink2_unweighted_norm(mt, H, s->eps, gh + H);
        ink2_lin_rows(wd[0], r, 2 * H, 1, gh, d);   // the down projection, threads over its r rows
        ink2_gate_ctx gc = { wf[0]->data, wg[0]->data, d, xt, mt, outs[0] + (size_t)t * H, r };
        ink2_par(H, 2 * r, ink2_gate_range, &gc);   // threads over the hidden rows
    }
    ink2_op_free(gh, owned);
    return INK2_OK;
}
ink2_status ink2_message_step(const ink2_shape *s, const ink2_weights *w, int block, const char *role, int T,
                              const float *x, const float *message, float *out) {
    return ink2_message_steps_c(s, w, block, &role, 1, T, &x, &message, (size_t)s->hidden, &out, NULL);
}

// ---- the layer and the forward

// Inkling2LoopRMSNorm: x * rsqrt(mean(x^2) + eps) as the unweighted norm, then weight * that (the reference's order)
static void ink2_rms_norm(const float *x, int n, float eps, const float *wt, float *y) {
    ink2_unweighted_norm(x, n, eps, y);
    for (int i = 0; i < n; i++) y[i] = wt[i] * y[i];
}
static ink2_status ink2_norm_rows(const ink2_shape *s, const ink2_weights *w, const char *name, int T, const float *x,
                                  float *y) {
    const ink2_tensor *t = w->get(w->ctx, name);
    if (!t || t->ne[0] != s->hidden || t->n_dims != 1) return INK2_NO_TENSOR;
    for (int i = 0; i < T; i++) ink2_rms_norm(x + (size_t)i * s->hidden, s->hidden, s->eps, t->data, y + (size_t)i * s->hidden);
    return INK2_OK;
}
// post_attention_convolution / post_mlp_convolution (conv_idx 2 / 3) of block b at application app, with its history
static ink2_status ink2_post_conv(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int idx, int T,
                                  int p0, const float *x, float *y) {
    char name[160];
    snprintf(name, sizeof name, "model.layers.%d.%s.conv1d.weight", s->app_block[app],
             idx == 2 ? "post_attention_convolution" : "post_mlp_convolution");
    const ink2_tensor *t = w->get(w->ctx, name);
    int H = s->hidden, K = s->conv_k;
    if (!t || t->ne[0] != K || t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] != (int64_t)H * K) return INK2_NO_TENSOR;
    float *hist = ink2_conv_hist(s, c, app, idx);
    ink2_short_conv(t->data, H, K, T, x, y, p0, hist);
    ink2_conv_hist_store(hist, K, H, p0, T, x);
    return INK2_OK;
}

// One application of a decoder layer, the reference's four classes in one (see inkling2.h for the order).
ink2_status ink2_decoder_layer(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                               const int *pos, const float *x, const float *block_states, int n_blocks,
                               const float *messages, float *out) {
    return ink2_decoder_layer_sel(s, w, c, app, T, pos, x, block_states, n_blocks, messages, out, NULL);
}
static ink2_status ink2_decoder_layer_lora(const ink2_shape *s, const ink2_weights *w, const ink2_lora *lora,
                                           ink2_cache *c, int app, int T, const int *pos, const float *x,
                                           const float *block_states, int n_blocks, const float *messages,
                                           float *out, int *sel) {
    if (!s || !w || !c || app < 0 || app >= s->n_app || T <= 0 || n_blocks < 0 || (n_blocks && !block_states))
        return INK2_BAD_INPUT;
    int b = s->app_block[app], g = s->block_global[b], H = s->hidden;
    int loop = b >= s->loop_start && b < s->loop_end, bound = ink2_block_boundary(s, b);
    if (messages && !loop) return INK2_BAD_INPUT;   // only the loop classes take messages
    if (b == 0 && n_blocks) return INK2_BAD_INPUT;  // layer 0 has no attention_input: it reads no block state
    // the post convolutions' rings follow the attention's slot: refuse a rewind before anything is written
    if (pos[0] > 0 && pos[0] != c->n_past[app]) return INK2_REWIND;
    size_t TH = (size_t)T * H;
    // scratch: the three attention inputs (read, then normed), one message, the attention output, the residual,
    // the MLP input, the expert output, up to 3 batched gate outputs, and the MLP's block states [T][n_blocks + 1][H]
    float *buf = ink2_cache_scratch(&c->work_dl, &c->work_dl_cap, TH * 15 + (size_t)T * (n_blocks + 1) * H);
    if (!buf) return INK2_BAD_INPUT;
    float *rd = buf, *qkv = rd + 3 * TH, *msg = qkv + 3 * TH, *att = msg + TH, *res = att + TH, *mi = res + TH,
          *mo = mi + TH, *gt = mo + TH, *bs2 = gt + 3 * TH;
    (void)msg;
    ink2_status st = INK2_OK;
    char name[160];
    const char *role[4] = { "q", "k", "v", "mlp" };
#define INK2_TRY(e) do { if ((st = (e)) != INK2_OK) goto done; } while (0)
    // 1. the attention inputs: attention_input (one query sliding, three full) or, at layer 0, x itself
    int n_rd = g ? 3 : 1;
    if (b > 0) {
        int nq = 0;
        snprintf(name, sizeof name, "model.layers.%d.attention_input", b);
        INK2_TRY(ink2_block_read_c(s, w, name, T, n_blocks, block_states, x, rd, &nq, c));
        if (nq != n_rd) { st = INK2_NO_TENSOR; goto done; }
    } else for (int r = 0; r < n_rd; r++) memcpy(rd + (size_t)r * TH, x, sizeof(float) * TH);   // q = k = v = x
    // 2. Q, K, V inputs: (loop layers, later passes) the q/k/v message gates on the read for that role (sliding: the
    // one read for all three), then the role's input norm (sliding: input_layernorm shared; full: q/k/v_input_layernorm)
    if (messages) {
        const float *srcs3[3] = { rd, g ? rd + TH : rd, g ? rd + 2 * TH : rd };
        const float *msgs3[3] = { messages, messages + H, messages + 2 * H };
        float *outs3[3] = { gt, gt + TH, gt + 2 * TH };
        INK2_TRY(ink2_message_steps_c(s, w, b, role, 3, T, srcs3, msgs3, (size_t)4 * H, outs3, c));
        for (int r = 0; r < 3; r++) {
            if (g) snprintf(name, sizeof name, "model.layers.%d.%s_input_layernorm.weight", b, role[r]);
            else snprintf(name, sizeof name, "model.layers.%d.input_layernorm.weight", b);
            INK2_TRY(ink2_norm_rows(s, w, name, T, outs3[r], qkv + (size_t)r * TH));
        }
    } else {
        snprintf(name, sizeof name, "model.layers.%d.%s", b, g ? "q_input_layernorm.weight" : "input_layernorm.weight");
        INK2_TRY(ink2_norm_rows(s, w, name, T, rd, qkv));
        if (g) {
            snprintf(name, sizeof name, "model.layers.%d.k_input_layernorm.weight", b);
            INK2_TRY(ink2_norm_rows(s, w, name, T, rd + TH, qkv + TH));
            snprintf(name, sizeof name, "model.layers.%d.v_input_layernorm.weight", b);
            INK2_TRY(ink2_norm_rows(s, w, name, T, rd + 2 * TH, qkv + 2 * TH));
        }
    }
    // 3. self_attn, then post_attention_convolution (conv_idx 2)
    INK2_TRY(ink2_attention_lora(s, w, lora, c, app, T, pos, qkv,
                                 (g || messages) ? qkv + TH : qkv,
                                 (g || messages) ? qkv + 2 * TH : qkv, att, NULL));
    INK2_TRY(ink2_post_conv(s, w, c, app, 2, T, pos[0], att, res));
    // 4. the residual: at a block boundary the attention output STARTS the new partial block (no add) and x joins
    // the block states the MLP input reads; otherwise x + attention
    int nb2 = n_blocks;
    if (bound) {
        for (int t = 0; t < T; t++) {
            if (n_blocks) memcpy(bs2 + (size_t)t * (n_blocks + 1) * H, block_states + (size_t)t * n_blocks * H, sizeof(float) * (size_t)n_blocks * H);
            memcpy(bs2 + ((size_t)t * (n_blocks + 1) + n_blocks) * H, x + (size_t)t * H, sizeof(float) * H);
        }
        nb2 = n_blocks + 1;
    } else {
        for (size_t i = 0; i < TH; i++) res[i] = x[i] + res[i];
        if (n_blocks) memcpy(bs2, block_states, sizeof(float) * (size_t)T * n_blocks * H);
    }
    // 5. mlp_input over those block states and the residual; (loop, later passes) the mlp message gate;
    // post_attention_layernorm; the experts; post_mlp_convolution (conv_idx 3); out = residual + that
    {
        int nq = 0;
        snprintf(name, sizeof name, "model.layers.%d.mlp_input", b);
        INK2_TRY(ink2_block_read_c(s, w, name, T, nb2, bs2, res, mi, &nq, c));
        if (nq != 1) { st = INK2_NO_TENSOR; goto done; }
    }
    if (messages) {
        const char *mlp_r = "mlp";
        const float *mlp_src = mi;
        const float *mlp_msg = messages + 3 * H;
        INK2_TRY(ink2_message_steps_c(s, w, b, &mlp_r, 1, T, &mlp_src, &mlp_msg, (size_t)4 * H, &gt, c));
        memcpy(mi, gt, sizeof(float) * TH);
    }
    snprintf(name, sizeof name, "model.layers.%d.post_attention_layernorm.weight", b);
    INK2_TRY(ink2_norm_rows(s, w, name, T, mi, gt));
    INK2_TRY(ink2_expert_layer_c(s, w, b, T, gt, mo, c));
    if (sel) INK2_TRY(ink2_expert_selection(s, w, b, T, gt, sel));
    INK2_TRY(ink2_post_conv(s, w, c, app, 3, T, pos[0], mo, gt));
    for (size_t i = 0; i < TH; i++) out[i] = res[i] + gt[i];
#undef INK2_TRY
done:
    return st;
}
ink2_status ink2_decoder_layer_sel(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int app, int T,
                                   const int *pos, const float *x, const float *block_states, int n_blocks,
                                   const float *messages, float *out, int *sel) {
    return ink2_decoder_layer_lora(s, w, NULL, c, app, T, pos, x, block_states, n_blocks, messages, out, sel);
}

// [T][H] slabs -> [T][n][H], the layout the block reads take
static void ink2_pack_states(float *const *slab, int n, int T, int H, float *dst) {
    for (int t = 0; t < T; t++) for (int j = 0; j < n; j++)
        memcpy(dst + ((size_t)t * n + j) * H, slab[j] + (size_t)t * H, sizeof(float) * (size_t)H);
}

// Inkling2LoopNormedEmbedding, the loop of Inkling2LoopTextModel.forward, final_input, norm, the head; see inkling2.h
ink2_status ink2_forward_chunk(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int p0, int T,
                               const int *ids, float *logits, int flags, float *app_states) {
    return ink2_forward_chunk_lora(s, w, NULL, c, p0, T, ids, logits, flags, app_states, NULL);
}
ink2_status ink2_forward_chunk_sel(const ink2_shape *s, const ink2_weights *w, ink2_cache *c, int p0, int T,
                                   const int *ids, float *logits, int flags, float *app_states, int *app_sel) {
    return ink2_forward_chunk_lora(s, w, NULL, c, p0, T, ids, logits, flags, app_states, app_sel);
}
ink2_status ink2_forward_chunk_lora(const ink2_shape *s, const ink2_weights *w, const ink2_lora *lora, ink2_cache *c,
                                    int p0, int T, const int *ids, float *logits, int flags, float *app_states,
                                    int *app_sel) {
    if (!s || !w || !c || T <= 0 || p0 < 0 || p0 + T > c->n_ctx || c->n_app != s->n_app) return INK2_BAD_INPUT;
    // a continuation must start where every slot ends (refused before any slot is touched); p0 = 0 starts over
    if (p0 > 0) for (int a = 0; a < s->n_app; a++) if (c->n_past[a] != p0) return INK2_REWIND;
    int H = s->hidden;
    size_t TH = (size_t)T * H;
    const ink2_tensor *emb = w->get(w->ctx, "model.embed_tokens.weight");
    if (!emb || emb->ne[0] != H) return INK2_NO_TENSOR;
    for (int t = 0; t < T; t++) if (ids[t] < 0 || ids[t] >= emb->ne[1]) return INK2_BAD_INPUT;
    // slabs [T][H]: every appended block state gets its own (a pass's states outlive its restart, for the messages),
    // plus the hidden state ping-pong, the pre-loop state, the pre-pre-loop state (control 1) and the previous pass's
    // final state. At most one append per application, plus the embedding.
    int n_slab = s->n_app + 6, maxb = s->n_app + 1;
    float *pool = ink2_cache_scratch(&c->work_fwd, &c->work_fwd_cap,
                                     TH * (size_t)n_slab + (size_t)T * maxb * H + (size_t)T * 4 * H + TH);
    float *cur_stack[2 * (INK2_MAX_APP + 1)];
    float **cur_heap = maxb <= INK2_MAX_APP + 1 ? NULL : malloc(sizeof(float *) * (size_t)maxb * 2);
    float **cur = cur_heap ? cur_heap : cur_stack;
    float **prev = cur + maxb;
    int pos_stack[256];
    int *pos_heap = T <= 256 ? NULL : malloc(sizeof(int) * (size_t)T);
    int *pos = pos_heap ? pos_heap : pos_stack;
    if (!pool || !cur || !pos) { free(cur_heap); free(pos_heap); return INK2_BAD_INPUT; }
    float *packed = pool + TH * (size_t)n_slab, *msgs = packed + (size_t)T * maxb * H, *fin = msgs + (size_t)T * 4 * H;
    int next = 0;
#define INK2_SLAB() (pool + TH * (size_t)(next++))
    float *h = INK2_SLAB(), *hn = INK2_SLAB(), *pre_h = INK2_SLAB(), *pre2_h = INK2_SLAB(), *last_h = INK2_SLAB();
    ink2_status st = INK2_OK;
#define INK2_TRY(e) do { if ((st = (e)) != INK2_OK) goto done; } while (0)
    for (int t = 0; t < T; t++) pos[t] = p0 + t;
    for (int t = 0; t < T; t++) {
        if (emb->q) dequant_row(emb->qtype, (const unsigned char *)emb->q + (size_t)ids[t] * emb->q_row_bytes, h + (size_t)t * H, H);
        else memcpy(h + (size_t)t * H, emb->data + (size_t)ids[t] * H, sizeof(float) * H);
    }
    INK2_TRY(ink2_norm_rows(s, w, "model.embed_tokens.embed_norm.weight", T, h, hn));
    { float *tmp = h; h = hn; hn = tmp; }
    {
        uint32_t bump_bits = atomic_load_explicit(&ink2_test_chunk_bump_bits, memory_order_relaxed);
        if (bump_bits && p0 > 0) {
            float bump;
            memcpy(&bump, &bump_bits, sizeof bump);
            h[0] += bump;
        }
    }
    int nb = 0, n_prev = 0, pre_nb = 0;
    memcpy(pre2_h, h, sizeof(float) * TH);
    for (int a = 0; a < s->n_app; a++) {
        int b = s->app_block[a], p = s->app_pass[a], loop = b >= s->loop_start && b < s->loop_end;
        if (loop && b == s->loop_start) {
            if (p == 0) { memcpy(pre_h, h, sizeof(float) * TH); pre_nb = nb; }   // pre_loop_hidden_states, _block_states
            else {
                // the previous pass's block states after the one appended at the core's first layer, and its final
                // hidden state; then restart from the pre-loop states
                n_prev = nb - pre_nb - 1;
                if (n_prev < 0) { st = INK2_BAD_INPUT; goto done; }
                for (int j = 0; j < n_prev; j++) prev[j] = cur[pre_nb + 1 + j];
                memcpy(last_h, h, sizeof(float) * TH);
                memcpy(h, (flags & INK2_MISWIRE_RESTART) ? pre2_h : pre_h, sizeof(float) * TH);
                nb = pre_nb;
            }
        }
        const float *m = NULL;
        if (loop && p > 0) {
            if (flags & INK2_MISWIRE_OWN_PASS) {   // control: this pass's own states so far, not the previous pass's
                int n_own = nb - pre_nb - 1 < 0 ? 0 : nb - pre_nb - 1;
                ink2_pack_states(cur + pre_nb + 1, n_own, T, H, packed);
                INK2_TRY(ink2_message_input_c(s, w, b - s->loop_start, T, n_own, packed, h, msgs, c));
            } else {
                ink2_pack_states(prev, n_prev, T, H, packed);
                INK2_TRY(ink2_message_input_c(s, w, b - s->loop_start, T, n_prev, packed, last_h, msgs, c));
            }
            m = msgs;
        }
        ink2_pack_states(cur, nb, T, H, packed);
        INK2_TRY(ink2_decoder_layer_lora(s, w, lora, c, a, T, pos, h, packed, nb, m, hn,
                                         app_sel ? app_sel + (size_t)a * T * s->n_used : NULL));
        if (ink2_block_boundary(s, b)) {   // the layer's INPUT joins the block states
            if (nb >= maxb || next >= n_slab) { st = INK2_BAD_INPUT; goto done; }
            float *slab = INK2_SLAB(); memcpy(slab, h, sizeof(float) * TH); cur[nb++] = slab;
        }
        if (a + 2 == s->loop_start) memcpy(pre2_h, hn, sizeof(float) * TH);   // the input of the last pre-loop layer
        { float *tmp = h; h = hn; hn = tmp; }
        if (app_states) memcpy(app_states + (size_t)a * TH, h, sizeof(float) * TH);
    }
    // final_input over every block state and the last hidden state, then optional norm + head
    {
        int nq = 0;
        ink2_pack_states(cur, nb, T, H, packed);
        INK2_TRY(ink2_block_read_c(s, w, "model.final_input", T, nb, packed, h, fin, &nq, c));
        if (nq != 1) { st = INK2_NO_TENSOR; goto done; }
        if (logits) {
            const int last = (flags & INK2_LOGITS_LAST) != 0;
            const int lt = last ? 1 : T;
            const float *lfin = last ? fin + (size_t)(T - 1) * H : fin;
            INK2_TRY(ink2_norm_rows(s, w, "model.norm.weight", lt, lfin, hn));
            INK2_TRY(ink2_head(s, w, lt, hn, logits));
        }
    }
#undef INK2_TRY
#undef INK2_SLAB
done:
    free(cur_heap); free(pos_heap);
    return st;
}

ink2_status ink2_forward(const ink2_shape *s, const ink2_weights *w, int T, const int *ids, float *logits) {
    ink2_cache c;
    if (!s || T <= 0) return INK2_BAD_INPUT;
    if (ink2_cache_init(&c, s, T)) return INK2_BAD_INPUT;
    ink2_status st = ink2_forward_chunk(s, w, &c, 0, T, ids, logits, 0, NULL);
    ink2_cache_free(&c);
    return st;
}

// One operation of the layer or the model on recorded inputs, for the harness's single-step lines; the same static
// code the forward runs. `module` is the reference's module path; n is the number of rows (positions, or for the
// per-head steps every head vector). Ops:
//   "linear"           <module>.weight [n_out][n_in]: in[0] [n][n_in] -> out [n][n_out]
//   "rms_norm"         <module>.weight [d]: in[0] [n][d] -> out (Inkling2LoopRMSNorm)
//   "unweighted_norm"  in[0] [n][hidden] -> out (eps: rms_norm_eps for a message gate's norm, else 1e-6, the reference's
//                      Inkling2LoopUnweightedRMSNorm() default in the block and message reads)
//   "short_conv"       <module>.conv1d.weight: in[0] [n][hidden] -> out, a prefill (no history)
//   "embed"            iin ids [n] -> out [n][hidden]: the row, then embed_tokens.embed_norm
//   "l2_clip"          in[0] [n][d] head vectors, d = in_d -> out (threshold rel_clip under relative_position_bias, else
//                      qk_clip)
//   "log_scale"        in[0] [n][in_d], iin positions [n] -> out * tau(pos)
//   "router"           <module> = model.layers.<b>.mlp.gate: in[0] [n][hidden] -> out logits [n][E], out2 weights
//                      [n][K] and iout indices [n][K], the chosen in ascending index
//   "score_fn"         in[0] [n] -> sigmoid
//   "experts"          <module> = model.layers.<b>.mlp.experts: in[0] [n][hidden], iin indices [n][K], in[1] weights
//                      [n][K] -> out [n][hidden]
//   "act_fn"           in[0] gate, in[1] up [n] -> SoftcappedSwiGLU
//   "softcap"          in[0] [n] -> tanh(x / cap) * cap, cap by module (gate_softcap, up_softcap, logit_softcap)
ink2_status ink2_op_step(const ink2_shape *s, const ink2_weights *w, const char *module, const char *op, int n, int in_d,
                         const float *const *in, const int *iin, float *out, float *out2, int *iout) {
    if (!s || !w || !module || !op || n <= 0) return INK2_BAD_INPUT;
    int H = s->hidden, b = -1;
    char name[192];
    sscanf(module, "model.layers.%d.", &b);
    if (!strcmp(op, "linear")) {
        snprintf(name, sizeof name, "%s.weight", module);
        const ink2_tensor *t = w->get(w->ctx, name);
        if (!t || t->n_dims != 2) return INK2_NO_TENSOR;
        ink2_lin_rows(t, (int)t->ne[1], (int)t->ne[0], n, in[0], out);
    } else if (!strcmp(op, "rms_norm")) {
        snprintf(name, sizeof name, "%s.weight", module);
        const ink2_tensor *t = w->get(w->ctx, name);
        if (!t || t->n_dims != 1) return INK2_NO_TENSOR;
        int d = (int)t->ne[0];
        for (int i = 0; i < n; i++) ink2_rms_norm(in[0] + (size_t)i * d, d, s->eps, t->data, out + (size_t)i * d);
    } else if (!strcmp(op, "unweighted_norm")) {
        float eps = strstr(module, "_message_gate.norm") ? s->eps : 1e-6f;
        for (int i = 0; i < n; i++) ink2_unweighted_norm(in[0] + (size_t)i * H, H, eps, out + (size_t)i * H);
    } else if (!strcmp(op, "short_conv")) {
        snprintf(name, sizeof name, "%s.conv1d.weight", module);
        const ink2_tensor *t = w->get(w->ctx, name);
        if (!t || t->ne[0] != s->conv_k || t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] != (int64_t)H * s->conv_k) return INK2_NO_TENSOR;
        ink2_short_conv(t->data, H, s->conv_k, n, in[0], out, 0, NULL);
    } else if (!strcmp(op, "embed")) {
        const ink2_tensor *e = w->get(w->ctx, "model.embed_tokens.weight");
        if (!e || e->ne[0] != H) return INK2_NO_TENSOR;
        float *row = malloc(sizeof(float) * (size_t)n * H);
        if (!row) return INK2_BAD_INPUT;
        for (int i = 0; i < n; i++) {
            if (iin[i] < 0 || iin[i] >= e->ne[1]) { free(row); return INK2_BAD_INPUT; }
            if (e->q) dequant_row(e->qtype, (const unsigned char *)e->q + (size_t)iin[i] * e->q_row_bytes, row + (size_t)i * H, H);
            else memcpy(row + (size_t)i * H, e->data + (size_t)iin[i] * H, sizeof(float) * H);
        }
        ink2_status st = ink2_norm_rows(s, w, "model.embed_tokens.embed_norm.weight", n, row, out);
        free(row);
        return st;
    } else if (!strcmp(op, "l2_clip")) {
        float thr = strstr(module, "relative_position_bias") ? s->rel_clip : s->qk_clip;
        memcpy(out, in[0], sizeof(float) * (size_t)n * in_d);
        for (int i = 0; i < n; i++) ink2_l2_clip(out + (size_t)i * in_d, in_d, thr);
    } else if (!strcmp(op, "log_scale")) {
        for (int i = 0; i < n; i++) {
            float tau = ink2_log_tau(s, iin[i]);
            for (int j = 0; j < in_d; j++) out[(size_t)i * in_d + j] = in[0][(size_t)i * in_d + j] * tau;
        }
    } else if (!strcmp(op, "router") || !strcmp(op, "experts")) {
        ink2_moe_w M;
        ink2_status st = ink2_moe_bind(s, w, b, &M);
        if (st != INK2_OK) return st;
        int E = s->n_expert, K = s->n_used, F = s->ff;
        float *sc = malloc(sizeof(float) * (2 * (size_t)E + 2 * (size_t)F + (size_t)H));
        unsigned char *chosen = malloc((size_t)E);
        if (!sc || !chosen) { free(sc); free(chosen); return INK2_BAD_INPUT; }
        float *wt = sc + E, *act = wt + E, *down = act + 2 * F;
        for (int i = 0; i < n && st == INK2_OK; i++) {
            if (op[0] == 'r') {
                ink2_route_one(s, &M, in[0] + (size_t)i * H, out + (size_t)i * E, sc, chosen, wt);
                for (int e = 0, k = 0; e < E; e++) if (chosen[e]) { iout[(size_t)i * K + k] = e; out2[(size_t)i * K + k] = wt[e]; k++; }
            } else {   // the given selection and weights
                for (int e = 0; e < E; e++) { chosen[e] = 0; wt[e] = 0.0f; }
                for (int k = 0; k < K; k++) {
                    int e = iin[(size_t)i * K + k];
                    if (e < 0 || e >= E || chosen[e]) { st = INK2_BAD_INPUT; break; }
                    chosen[e] = 1; wt[e] = in[1][(size_t)i * K + k];
                }
                if (st == INK2_OK) st = ink2_experts_one(s, &M, in[0] + (size_t)i * H, chosen, wt, act, down, out + (size_t)i * H);
            }
        }
        free(sc); free(chosen);
        return st;
    } else if (!strcmp(op, "score_fn")) {
        for (int i = 0; i < n; i++) out[i] = ink2_sigmoid(in[0][i]);
    } else if (!strcmp(op, "act_fn")) {
        for (int i = 0; i < n; i++) out[i] = ink2_swiglu(s, in[0][i], in[1][i]);
    } else if (!strcmp(op, "softcap")) {
        float cap = strstr(module, "gate_softcap") ? s->gate_cap : strstr(module, "up_softcap") ? s->up_cap : s->final_cap;
        for (int i = 0; i < n; i++) out[i] = ink2_softcap(in[0][i], cap);
    } else return INK2_BAD_INPUT;
    return INK2_OK;
}

static ink2_rel_probe *g_rel_probe = NULL;
void ink2_set_rel_probe(ink2_rel_probe *p) { g_rel_probe = p; }
ink2_rel_probe *ink2_active_rel_probe(void) {
    if (!g_rel_probe) return NULL;
    const char *e = getenv("RUNNER_TEST_INK2_REL_PROBE");
    return (e && *e) ? g_rel_probe : NULL;
}

// TEST-ONLY (R4.48.25): the fast route's own vector kernels for tests/test_inkling2_vmath.c (see inkling2.h)
const char *ink2_test_vmath_form(void) {
#if INK2_SIMD_AVX2
    return "avx2";
#elif INK2_SIMD_NEON
    return "neon";
#else
    return "scalar";
#endif
}
int ink2_test_vmath(int fn, const float *x, float *y, int n) {
    if (fn != INK2_VM_EXP && fn != INK2_VM_TANH) return -1;
    int i = 0;
#if INK2_SIMD_AVX2
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, fn == INK2_VM_EXP ? ink2_vexp8(v) : ink2_vtanh8(v));
    }
#elif INK2_SIMD_NEON
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(y + i, fn == INK2_VM_EXP ? ink2_vexp4(v) : ink2_vtanh4(v));
    }
#endif
    for (; i < n; i++) y[i] = fn == INK2_VM_EXP ? p_expf(x[i]) : p_tanhf(x[i]);
    return 0;
}
void ink2_test_swiglu(float gate_cap, float up_cap, float *act, int F, int n) {
    ink2_shape s;
    memset(&s, 0, sizeof s);
    s.gate_cap = gate_cap;
    s.up_cap = up_cap;
    const int nb = (F + 7) / 8;
    ink2_swiglu_ctx sw = { &s, act, F, nb };
    ink2_swiglu_range(&sw, 0, (int64_t)n * nb);
}
ink2_status ink2_test_attn_core(int fast, int hd, int nk, const float *q, const float *k, const float *v, float sink,
                                float *out) {
    if (hd < 1 || nk < 1) return INK2_BAD_INPUT;
    ink2_shape s;
    memset(&s, 0, sizeof s);
    ink2_attn_layer L;
    memset(&L, 0, sizeof L);
    L.g = 1; L.nh = 1; L.nkv = 1; L.hd = hd; L.C = hd; L.Q = hd;
    const int pos = nk - 1;
    const float rl = 0.0f;
    float *lg = malloc(sizeof(float) * ((size_t)nk + 1));
    if (!lg) return INK2_BAD_INPUT;
    ink2_attn_core(&s, &L, fast, 1, &pos, q, k, v, nk, 0, &rl, &sink, lg, out, NULL);
    free(lg);
    return INK2_OK;
}
