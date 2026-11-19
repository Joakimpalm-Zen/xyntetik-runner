// R6.41: a prompt's rows carry a lone decode's bits on CUDA's scalar route.
//
// A prefill computes each prompt row's K and V projections for a whole tile
// of columns at once. When the tile kernel sums in another order than the
// batch-1 GEMV the decode runs, the rows a prompt leaves in the KV cache are
// not the rows the same tokens leave when decoded one at a time, and a q8_0
// cache can turn that last-bit difference into whole quantization steps
// (R6.35: Q4_0's k_mvb_q4_0; R6.39: the K-quants' k_gemm_*; R6.50:
// k_gemm_q8_0; R6.54: k_gemm_q3_K).
//
// The claim: on the scalar route (the tensor-core GEMM pinned off, as the
// kv-tol gate pins it), a prefill at any batch width stores layer 0's K and V
// rows byte for byte as a batch-1 prefill stores them. Layer 0 only, because
// layer 0's rows depend on nothing but the projections (embedding, norm,
// projection, rope, store): from layer 1 on, a row also depends on the
// prefill attention, which is a different route with its own gate. An f16
// cache, because it keeps the projections' last bits that a q8_0 cache may
// round away.
//
// The widths: 9 (one pass narrower than the 16-wide twin), 17 (16 and a lone
// last column, the batch-1 kernel), 24 (16 and 8, the 8-wide twin) and 64 (the
// prompt window's tile), each against batch 1 over the same 128 rows.
//
// The route table (R6.57, R6.58), on the first fixture's device: a matvec over
// more than one column on the scalar route has three legal answers, and every
// weight type the device decodes must give one of them (gpu_prefill_route):
// its own tile kernel where that kernel is the batch-1 kernel's twin (F32, F16
// and BF16 only, mvb_is_mv_twin in cuda.c), the batch-1 kernel's 16-wide twin
// wherever one is loaded, or the batch-1 kernel once per column. A tile kernel
// that sums in another order is the fourth answer, and fails. Asked at batch 9
// and 64 unscaled for every type, and for NVFP4, the one type that carries a
// per-tensor scale companion, scaled at batch 2, 8 and 64 (a scaled tensor
// skips the small-batch branch). It reaches the types no fixture here can be
// written for (IQ4_XS, Q2_K, Q4_1, IQ1_S, IQ1_M). The rows say what the table
// cannot: a twin that is routed but disagrees, or a defect a fixture's shape
// hides (MXFP4's rows longer than 1,024 values, test-mxfp4-prefill.gguf).
//
// Anchor: none of its own. It compares two of runner's own routes; the
// batch-1 GEMV it holds the prefill to is gated against the CPU by the CUDA
// gate's cpu-cuda step and test-gpu-identity. Boundary inputs: the weights are
// the generated fixtures'. No kernel is rewritten by R6.41: the route sends
// prefill columns to the existing width-classed twins, whose per-column
// identity to the batch-1 GEMV test-batch gates (test-gates-batch), so this is
// a routing gate, not a kernel-equivalence gate.
//
//     ./test-prefill-bits test-q4k.gguf test-q5k.gguf test-q6k.gguf test-q40.gguf test-q8.gguf
//         test-q3k-prefill.gguf test-iq2xxs-prefill.gguf test-iq2xs-prefill.gguf
//         test-iq2s-prefill.gguf test-iq3xxs-prefill.gguf test-iq3s-prefill.gguf
//         test-mxfp4-prefill.gguf test-bf16.gguf test-q50-prefill.gguf test-q51-prefill.gguf
//         test-iq4nl-prefill.gguf test-nvfp4-prefill.gguf
//
// R6.62, split-K (qwen4exp): that architecture alone allocates BF16's split-K
// partials, and there a BF16 tensor of at most 1,024 rows of at least 2,048
// values decodes through k_mv_bf16_splitk at batch 1, while a prompt's product
// of it takes the small-batch twins (2 to 8 columns) or k_mv_bf16_b (more), the
// twins of the plain batch-1 kernel. Its layer 0 is recurrent and stores no K
// or V rows, so for a qwen4exp fixture the gate compares the product itself:
// every BF16 tensor of at most 1,024 rows, through the device encoder's own
// route (gpu_test_mv), at 2, 4, 8, 9, 17, 24 and 64 columns against 64 batch-1
// calls, byte for byte per column. The tensors of the split-K shape are the
// claim; the narrower ones never take split-K and are the live control, which
// must be equal at every width, and gpu_splitk_dispatches shows which route
// each call took. Fixtures (scripts/make-test-qwen4exp.py with
// QWEN4EXP_TEST_HC_BF16=1): QWEN4EXP_TEST_EMBD=512 for the mixers' 2,048 -> 16
// and 2,048 -> 4, with QWEN4EXP_TEST_HC_LOWRANK=512 for 2,048 -> 512, and
// QWEN4EXP_TEST_EMBD=256 for the 1,024-wide control.
//
// Without a CUDA device it says so and passes (nothing to compare).
#include "runner.h"
#include "gate_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PREFILL = 128, MAX_TOK = 192 };

static const char *TEXT =
    "The city of Lisbon sits on seven hills above the Tagus estuary, and its "
    "oldest quarter survived the 1755 earthquake largely intact because the "
    "bedrock there is firmer than the reclaimed ground downriver. Rebuilding "
    "the lower town took decades, and the grid of streets laid out afterwards "
    "was among the first in Europe designed with seismic loads in mind.";

static const int WIDTHS[] = { 9, 17, 24, 64 };

// R6.62: a matvec of every BF16 tensor of at most 1,024 rows at SK_WIDTHS
// columns against SK_MAX batch-1 calls (see the header). Returns 0 ok, 1
// skipped (no CUDA device), -1 failure.
static const int SK_WIDTHS[] = { 2, 4, 8, 9, 17, 24, 64 };
enum { SK_MAX = 64 };
static int splitk_check(const char *path) {
    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_AUTO;
    p.n_threads = 4;
    p.n_ctx = 64;
    p.n_batch = SK_MAX;
    if (!model_load(&m, path, &p)) { fprintf(stderr, "FAIL: %s: load failed\n", path); return -1; }
    if (!m.gpu || m.gpu_layers <= 0) { model_free(&m); return 1; }
    int fail = 0, shapes = 0, controls = 0;
    for (uint64_t i = 0; i < m.gf.n_tensors; i++) {
        gguf_tensor *w = &m.gf.tensors[i];
        if (w->type != T_BF16 || w->n_dims != 2 || w->ne[1] > 1024) continue;
        int n_in = (int)w->ne[0], n_out = (int)w->ne[1];
        bool shape = n_in >= 2048;
        float *x = malloc(sizeof(float) * (size_t)SK_MAX * n_in);
        float *y1 = malloc(sizeof(float) * (size_t)SK_MAX * n_out);
        float *yn = malloc(sizeof(float) * (size_t)SK_MAX * n_out);
        if (!x || !y1 || !yn) { free(x); free(y1); free(yn); model_free(&m); return -1; }
        uint32_t r = 0x9E3779B9u ^ (uint32_t)i;
        for (size_t k = 0; k < (size_t)SK_MAX * n_in; k++) {
            r ^= r << 13; r ^= r >> 17; r ^= r << 5;
            x[k] = (float)((int)(r % 2001u) - 1000) / 1000.0f;
        }
        unsigned long s0 = gpu_splitk_dispatches();
        bool ok = true;
        for (int c = 0; c < SK_MAX && ok; c++)
            ok = gpu_test_mv(&m, w, x + (size_t)c * n_in, y1 + (size_t)c * n_out, n_in, n_out, 1);
        unsigned long s1 = gpu_splitk_dispatches() - s0;
        if (!ok) { fprintf(stderr, "FAIL: %s: %s: the batch-1 calls failed\n", path, w->name); fail = 1; }
        // the route evidence: the claim's tensors take split-K at batch 1, the controls never
        if (ok && shape && s1 != SK_MAX) {
            fprintf(stderr, "FAIL: %s: %s [%d -> %d]: %lu of %d batch-1 calls took split-K, so this "
                    "context does not reach the route the gate is about\n", path, w->name, n_in, n_out,
                    s1, SK_MAX);
            fail = 1;
        }
        if (ok && !shape && s1 != 0) {
            fprintf(stderr, "FAIL: %s: %s [%d -> %d]: the control took split-K (%lu calls)\n", path,
                    w->name, n_in, n_out, s1);
            fail = 1;
        }
        shapes += shape;
        controls += !shape;
        for (size_t wi = 0; ok && wi < sizeof(SK_WIDTHS) / sizeof(SK_WIDTHS[0]); wi++) {
            int tn = SK_WIDTHS[wi];
            unsigned long t0 = gpu_splitk_dispatches();
            if (!gpu_test_mv(&m, w, x, yn, n_in, n_out, tn)) {
                fprintf(stderr, "FAIL: %s: %s: a %d-column call failed\n", path, w->name, tn);
                fail = 1;
                break;
            }
            unsigned long tsk = gpu_splitk_dispatches() - t0;
            long d = 0;
            double mx = 0;
            for (size_t k = 0; k < (size_t)tn * n_out; k++) {
                if (memcmp(&yn[k], &y1[k], sizeof(float)) == 0) continue;
                d++;
                double e = (double)yn[k] - (double)y1[k];
                if (e < 0) e = -e;
                if (e > mx) mx = e;
            }
            printf("  %s %s [%d -> %d] batch %2d vs batch 1: %ld of %d values differ, max |d| %.3e; "
                   "split-K launches %lu at batch 1 (%d calls), %lu at batch %d\n",
                   shape ? "split-K shape" : "control", w->name, n_in, n_out, tn, d, tn * n_out, mx,
                   s1, SK_MAX, tsk, tn);
            if (d) {
                fprintf(stderr, "FAIL: %s: %s [%d -> %d]: a %d-column product is not %d batch-1 calls' "
                        "bits (%ld values, max |d| %.3e)%s\n", path, w->name, n_in, n_out, tn, tn, d, mx,
                        shape ? "" : "; this is the control, so the gate measures something else");
                fail = 1;
            }
        }
        free(x); free(y1); free(yn);
    }
    printf("prefill-bits: %s | split-K: %d tensors of the split-K shape, %d controls, %d layers on the GPU\n",
           path, shapes, controls, m.gpu_layers);
    if (!shapes && !controls) { fprintf(stderr, "FAIL: %s: no BF16 tensor of at most 1,024 rows\n", path); fail = 1; }
    model_free(&m);
    return fail ? -1 : 0;
}

typedef struct { unsigned char *k, *v; size_t bytes; } rows0;

// Prefill PREFILL rows at batch width nb on the GPU with an f16 cache and copy
// layer 0's K and V rows out of the host cache (the device copies its rows
// back after every forward). Returns 0 ok, 1 skipped (no GPU), -1 failure.
static int prefill_rows0(const char *path, const int32_t *toks, int nb, rows0 *out,
                         int *n_layer_gpu) {
    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_AUTO;
    p.n_threads = 4;
    p.n_ctx = MAX_TOK + 8;
    p.n_batch = nb;
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "FAIL: %s: load failed at batch %d\n", path, nb);
        return -1;
    }
    if (!m.gpu || m.gpu_layers <= 0) { model_free(&m); return 1; }
    if (m.kv_q8 || m.kv_fp4 || m.kv_split || m.kv_ring) {
        fprintf(stderr, "FAIL: %s: expected a flat f16 cache\n", path);
        model_free(&m);
        return -1;
    }
    *n_layer_gpu = m.gpu_layers;
    for (int off = 0; off < PREFILL; off += nb) {
        int n = PREFILL - off < nb ? PREFILL - off : nb;
        float *lg = model_forward_batch(&m, toks + off, n, off, off + n == PREFILL);
        if (off + n == PREFILL && !lg) {
            fprintf(stderr, "FAIL: %s: forward failed at batch %d\n", path, nb);
            model_free(&m);
            return -1;
        }
    }
    size_t row = model_k_bytes_of(&m, (size_t)model_kv_dim(&m, 0));
    out->bytes = row * PREFILL;
    out->k = malloc(out->bytes);
    out->v = malloc(out->bytes);
    if (!out->k || !out->v) { model_free(&m); return -1; }
    memcpy(out->k, (const unsigned char *)m.kcache + model_k_byte_off(&m, 0), out->bytes);
    memcpy(out->v, (const unsigned char *)m.vcache + model_v_byte_off(&m, 0), out->bytes);
    model_free(&m);
    return 0;
}

// The route table on the device a model of path loads. Returns 0 ok, 1 skipped
// (no GPU, or a build whose device backend is not CUDA), -1 failure.
static int route_table(const char *path) {
    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_AUTO;
    p.n_threads = 4;
    p.n_ctx = MAX_TOK + 8;
    p.n_batch = 64;
    if (!model_load(&m, path, &p)) { fprintf(stderr, "FAIL: %s: load failed\n", path); return -1; }
    if (!m.gpu || m.gpu_layers <= 0) { model_free(&m); return 1; }
#ifndef RUNNER_GPU_CUDA
    // The table is CUDA's scalar route. Another device backend (Metal) loads
    // the model on its GPU and answers "none" for every type, which the
    // anti-vacuity check below would read as a table that checks nothing.
    model_free(&m);
    return 1;
#endif
    static const char *ANSWER[] = { "none", "its own tile (the batch-1 twin)", "the twins",
                                    "the batch-1 kernel per column", "a TILE KERNEL" };
    static const struct { int batch; bool scaled; } ASKED[] = { { 9, false }, { 64, false } };
    static const struct { int batch; bool scaled; } ASKED_SCALED[] = { { 2, true }, { 8, true }, { 64, true } };
    int bad = 0, types = 0;
    for (int t = 0; t < 64; t++) {
        if (gpu_prefill_route(&m, t, 64, false) == GPU_PREFILL_NONE) continue;   // not decoded here
        types++;
        bool twin = gpu_prefill_twin16(&m, t);
        bool own = t == T_F32 || t == T_F16 || t == T_BF16;
        int n = t == T_NVFP4 ? 5 : 2;
        for (int k = 0; k < n; k++) {
            int batch = k < 2 ? ASKED[k].batch : ASKED_SCALED[k - 2].batch;
            bool scaled = k < 2 ? ASKED[k].scaled : ASKED_SCALED[k - 2].scaled;
            int r = gpu_prefill_route(&m, t, batch, scaled);
            printf("  route: %-8s batch %2d%s: %s\n", ggml_type_name(t), batch, scaled ? " scaled" : "",
                   r >= 0 && r <= GPU_PREFILL_TILE ? ANSWER[r] : "?");
            const char *why = NULL;
            if (r == GPU_PREFILL_TILE || r < 0 || r > GPU_PREFILL_TILE)
                why = "runs a tile kernel that sums in another order than the batch-1 kernel";
            else if (r == GPU_PREFILL_TILE_TWIN && !own)
                why = "runs its own tile kernel, which is not the batch-1 kernel's twin";
            else if (r == GPU_PREFILL_NONE)
                why = "has no route although the device decodes the type";
            else if (!scaled && batch > 8 && twin && !own && r != GPU_PREFILL_TWINS)
                why = "has a loaded 16-wide twin of its batch-1 kernel and does not run it";
            if (why) {
                fprintf(stderr, "FAIL: %s at batch %d%s %s\n", ggml_type_name(t), batch,
                        scaled ? " (scaled)" : "", why);
                bad = 1;
            }
        }
    }
    // anti-vacuity: a table that sees no twin, or no NVFP4 to ask the scaled
    // question of, checks nothing. Q4_0's twins are the ones R6.35 first routed
    // a prompt through.
    if (!gpu_prefill_twin16(&m, T_Q4_0) || gpu_prefill_route(&m, T_NVFP4, 64, true) == GPU_PREFILL_NONE) {
        fprintf(stderr, "FAIL: the route table sees no Q4_0 16-wide twin or no NVFP4 decode (%d types)\n",
                types);
        bad = 1;
    }
    model_free(&m);
    return bad ? -1 : 0;
}

static long diff16(const unsigned char *a, const unsigned char *b, size_t bytes, long *first_row,
                   size_t row_bytes) {
    long n = 0;
    *first_row = -1;
    for (size_t i = 0; i + 1 < bytes; i += 2)
        if (a[i] != b[i] || a[i + 1] != b[i + 1]) {
            if (*first_row < 0) *first_row = (long)(i / row_bytes);
            n++;
        }
    return n;
}

int main(int argc, char **argv) {
    f16_init();
    // the scalar route: the tensor-core GEMM's bits are its own tolerance
    // class (test-tc-tol), so it is pinned off here as kv-tol pins it
    gpu_tc_force(0);
    int fail = 0, ran = 0;
    if (argc > 1) {
        printf("prefill-bits: route table on %s's device\n", argv[1]);
        int r = route_table(argv[1]);
        if (r < 0) fail = 1;
        if (r == 1 && gate_no_device("prefill-bits")) fail = 1;
        else if (r == 1) printf("prefill-bits: route table skipped (no CUDA device)\n");
    }
    for (int a = 1; a < argc; a++) {
        const char *path = argv[a];
        gguf_file gf;
        if (!gguf_open(&gf, path)) { fprintf(stderr, "FAIL: cannot open %s\n", path); fail = 1; continue; }
        if (!strcmp(gguf_get_str(&gf, "general.architecture", ""), "qwen4exp")) {
            gguf_close(&gf);
            int rs = splitk_check(path);
            if (rs == 1) {
                if (gate_no_device("prefill-bits")) fail = 1;
                else printf("prefill-bits: %s skipped (no CUDA device)\n", path);
            } else if (rs < 0) fail = 1;
            else ran++;
            continue;
        }
        tokenizer tk;
        if (!tokenizer_init(&tk, &gf)) { gguf_close(&gf); fprintf(stderr, "FAIL: tokenizer %s\n", path); fail = 1; continue; }
        static int32_t toks[MAX_TOK];
        int n = tok_encode(&tk, TEXT, toks, MAX_TOK, true, false);
        tokenizer_free(&tk);
        gguf_close(&gf);
        if (n < 16) { fprintf(stderr, "FAIL: %s: text is %d tokens\n", path, n); fail = 1; continue; }
        for (int i = n; i < MAX_TOK; i++) toks[i] = toks[i - n + 1];

        rows0 ref = { 0 };
        int gl = 0;
        int r = prefill_rows0(path, toks, 1, &ref, &gl);
        if (r == 1) {
            if (gate_no_device("prefill-bits")) fail = 1;
            else printf("prefill-bits: %s skipped (no CUDA device)\n", path);
            continue;
        }
        if (r < 0) { fail = 1; continue; }
        // anti-vacuity: rows of zeros compare equal to anything that is also zeros
        long nz = 0;
        for (size_t i = 0; i < ref.bytes; i++) nz += ref.k[i] != 0;
        if (nz == 0) { fprintf(stderr, "FAIL: %s: batch-1 layer 0 K rows are all zero\n", path); fail = 1; continue; }
        ran++;
        size_t row_bytes = ref.bytes / PREFILL;
        printf("prefill-bits: %s | %d rows, layer 0, f16 cache, %d layers on the GPU, TC off\n",
               path, PREFILL, gl);
        for (size_t w = 0; w < sizeof(WIDTHS) / sizeof(WIDTHS[0]); w++) {
            rows0 got = { 0 };
            int gl2 = 0;
            if (prefill_rows0(path, toks, WIDTHS[w], &got, &gl2) != 0) { fail = 1; continue; }
            long fk, fv;
            long dk = diff16(got.k, ref.k, ref.bytes, &fk, row_bytes);
            long dv = diff16(got.v, ref.v, ref.bytes, &fv, row_bytes);
            printf("  batch %2d vs batch 1: K %ld, V %ld of %zu values differ%s\n", WIDTHS[w], dk, dv,
                   ref.bytes / 2, dk || dv ? "" : " (byte-identical)");
            if (dk || dv) {
                fprintf(stderr, "FAIL: %s: a batch-%d prefill stores other layer-0 rows than a batch-1 "
                        "prefill (first differing row: K %ld, V %ld)\n", path, WIDTHS[w], fk, fv);
                fail = 1;
            }
            free(got.k); free(got.v);
        }
        free(ref.k); free(ref.v);
    }
    printf(fail ? "prefill-bits: FAILED\n" : ran ? "prefill-bits: ok\n" : "prefill-bits: ok (skipped)\n");
    return fail;
}
