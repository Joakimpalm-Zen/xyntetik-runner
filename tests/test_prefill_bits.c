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
// The route table (R6.57), on the first fixture's device: every weight type
// whose batch-1 route has a loaded 16-wide twin prefills through it, except
// F32, F16 and BF16, whose own tile kernel is already the batch-1 kernel's
// twin (mvb_is_mv_twin in cuda.c). It reaches the types no fixture here can
// be written for (IQ4_XS). The rows say what the table cannot: a twin that is
// routed but disagrees, or a defect a fixture's shape hides (MXFP4's rows
// longer than 1,024 values, test-mxfp4-prefill.gguf).
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
//         test-mxfp4-prefill.gguf test-bf16.gguf
//
// Without a CUDA device it says so and passes (nothing to compare).
#include "runner.h"

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
// (no GPU), -1 failure.
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
    int bad = 0, twins = 0;
    for (int t = 0; t < 64; t++) {
        bool twin = gpu_prefill_twin16(&m, t), takes = gpu_prefill_takes_twins(&m, t);
        bool own = t == T_F32 || t == T_F16 || t == T_BF16;
        if (!twin && !takes) continue;
        twins += twin;
        printf("  route: %-8s 16-wide twin %s, prefill on %s\n", ggml_type_name(t),
               twin ? "loaded" : "none", takes ? "the twins" : "its own kernel");
        if (takes != (twin && !own)) {
            fprintf(stderr, "FAIL: %s has a 16-wide twin of its batch-1 kernel%s, and its prefill %s\n",
                    ggml_type_name(t), own ? " and a tile kernel that is that kernel's twin" : "",
                    takes ? "runs the twins" : "runs a tile kernel that sums in another order");
            bad = 1;
        }
    }
    // anti-vacuity: a table that sees no twin checks nothing. Q4_0's twins are
    // the ones R6.35 first routed a prompt through.
    if (!gpu_prefill_twin16(&m, T_Q4_0)) {
        fprintf(stderr, "FAIL: the route table sees no Q4_0 16-wide twin (%d twins in all)\n", twins);
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
        if (r == 1) printf("prefill-bits: route table skipped (no CUDA device)\n");
    }
    for (int a = 1; a < argc; a++) {
        const char *path = argv[a];
        gguf_file gf;
        if (!gguf_open(&gf, path)) { fprintf(stderr, "FAIL: cannot open %s\n", path); fail = 1; continue; }
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
        if (r == 1) { printf("prefill-bits: %s skipped (no CUDA device)\n", path); continue; }
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
