// CPU vs GPU agreement on the scalar route, at LOGIT precision.
//
// What the project's CPU/GPU contract actually is, stated precisely because
// this gate was first written against a stronger reading of it and failed
// everywhere, including on code nobody had touched: the scalar route is
// TOKEN-identical, and that is what `make test-metal-kquant` and friends
// verify by comparing two runs' generated text. It is NOT logit-bit-identical.
// The CPU sums a row with SIMD lanes, the GPU with a 32-lane simd_sum
// reduction tree; those round differently, and the measured residue is around
// 1e-7 of a logit on an F32 toy and 1e-3 on a real Q8_0 model -- far below
// anything argmax can see, which is exactly why token identity holds.
//
// So why add this at all? Because token identity is BLIND at fixture scale.
// tests/test_attn_knobs says so in its own docstring: on a toy random-weight
// model every variant emits byte-identical text. Scaling every Q by up to 9x
// on the llama-4 temperature fixture changes not one emitted token. A
// fixture-scale backend feature -- exactly the kind a backend grows one knob
// at a time -- could therefore be implemented WRONG on the GPU and pass every
// gate in the tree.
//
// This closes that gap by measuring the CPU/GPU logit deviation as a fraction
// of the logit range and bounding it. The bound sits far above the reduction
// residue and far below a real mistake: a backend that silently ignores the
// attention temperature lands orders of magnitude out, because it is not a
// rounding difference, it is a missing multiply.
//
//     ./test-gpu-identity test-attn/k_temp_live.gguf
//     ./test-gpu-identity models/granite-4.1-8b-Q4_0.gguf 8   # 8 GPU layers
//     ./test-gpu-identity test-muse.gguf 0 1   # every forward a single token
//
// Skips (never passes quietly) when there is no GPU on the machine, or when
// the model fell back to the CPU -- comparing the CPU against itself is the
// vacuity every tolerance gate here is written to avoid.
#include "runner.h"
#include "finite_check.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MinGW has no setenv; the two-argument shim is the same idiom
// tests/test_instances.c uses. Without it this file does not compile on
// Windows, which is where the CUDA gates run (CI builds Windows but does not
// run `make test` there, so it went unseen).
#ifdef _WIN32
#include <stdlib.h>
#define setenv_compat(k, v) _putenv_s(k, v)
#else
#include <stdlib.h>
#define setenv_compat(k, v) setenv(k, v, 1)
#endif


enum { STEPS = 24, MAX_TOK = 96, N_BATCH = 32 };

// Prefill batch. Settable because "every forward this backend ever sees is a
// single token" is a distinct backend STATE, not just a slower schedule: a
// backend that sizes scratch lazily on the first multi-token batch is in a
// different configuration for the whole run, and at N_BATCH = 32 nothing in
// this tree ever looks at it. Two Metal buffers were reached nil that way.
static int g_n_batch = N_BATCH;

// Mean |dlogit| as a fraction of the mean logit range. Measured residue:
// Metal/M1 2.3e-8 (F32 toy), 1.9e-7 (llama-4 temperature fixture), 5.4e-5
// (SmolLM2 Q8_0 -- quantized weights dequantize slightly differently on the
// two sides, which dominates). CUDA/Blackwell measures 8.02e-4 on the
// k_temp_live fixture -- its reductions round differently again -- which a
// 5e-4 bound calibrated on Metal wrongly failed (found by the 2026-08-16
// Blackwell run; the gate blocked that box's `make test` on untouched code).
// The bound is one cross-backend constant on purpose, sized ~2.5x above the
// worst measured honest residue and still 10x BELOW the smallest real
// mistake: dropping the attention temperature measures 2.05e-2.
#define GPU_DEV_FRAC 2e-3

static int g_gpu_layers = 0;

// Deliberately not the tolerance-gate corpus: this runs on toy fixtures whose
// vocabularies are tiny, so the text only has to tokenize, not to be prose.
static const char *TEXT =
    "The quick brown fox jumps over the lazy dog while the observatory "
    "published a revised catalogue listing four thousand objects in 1929.";

static float *run(const char *path, int gpu_mode, int *n_vocab_out,
                  bool *used_gpu, const int32_t *toks, int n_tok) {
    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = gpu_mode;
    p.n_ctx    = n_tok + 8;
    p.n_batch  = g_n_batch;
    p.gpu_layers_override = gpu_mode == GPU_OFF ? 0 : g_gpu_layers;

    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "cannot load %s\n", path);
        return NULL;
    }
    *used_gpu = m.gpu != NULL;
    *n_vocab_out = m.n_vocab;

    float *out = malloc(sizeof(float) * (size_t)STEPS * (size_t)m.n_vocab);
    if (!out) { model_free(&m); return NULL; }

    int prefill = n_tok - STEPS;
    float *lg = NULL;
    for (int off = 0; off < prefill; off += g_n_batch) {
        int n = prefill - off < g_n_batch ? prefill - off : g_n_batch;
        lg = model_forward_batch(&m, toks + off, n, off, off + n == prefill);
    }
    if (!lg) { model_free(&m); free(out); return NULL; }
    memcpy(out, lg, sizeof(float) * (size_t)m.n_vocab);

    for (int s = 1; s < STEPS; s++) {
        lg = model_forward(&m, toks[prefill + s - 1], prefill + s - 1);
        if (!lg) { model_free(&m); free(out); return NULL; }
        memcpy(out + (size_t)s * m.n_vocab, lg,
               sizeof(float) * (size_t)m.n_vocab);
    }
    model_free(&m);
    return out;
}

// The verdict on two [steps][nv] logit sets. 0 = ok, 1 = FAILED, printed as
// the gate's last line either way. Kept apart from the model runs so that
// `--self-test` can drive it with synthetic inputs under the release flags.
static int compare_logits(const float *cpu, const float *gpu, int nv, int steps) {
    size_t n = (size_t)steps * (size_t)nv;

    // Non-finite logits on EITHER side fail before any reduction. Until
    // 2026-09-14 a single GPU NaN passed this gate: |cpu - NaN| is NaN, the
    // sum and the fraction of range became NaN, and "NaN > limit" is false,
    // so the gate printed ok (reproduced by the v0.5.3 review with clang -O2
    // and -O3 -ffast-math; the released kernels were not producing NaNs, the
    // instrument was blind to them). +/-Inf reached the limit check and
    // failed, but only by accident of the comparison's direction. The masked
    // vocabulary sentinel (-1e30 and below) is finite and stays admitted.
    // The detection lives in tests/finite_check.c, a fast-math-free TU: in
    // this one the compiler folds any NaN test, bit pattern included, to
    // false (see that file).
    size_t first_cpu = n, first_gpu = n;
    size_t n_nonfinite_cpu = count_nonfinite_f32(cpu, n, &first_cpu);
    size_t n_nonfinite_gpu = count_nonfinite_f32(gpu, n, &first_gpu);
    if (n_nonfinite_cpu || n_nonfinite_gpu) {
        size_t first_bad = first_cpu < first_gpu ? first_cpu : first_gpu;
        printf("FAIL: non-finite logits (cpu %zu, gpu %zu of %zu; first at "
               "position %zu vocab %d: cpu %g, gpu %g) — a NaN or Inf is a "
               "defect, never a rounding difference\ngpu-identity: FAILED\n",
               n_nonfinite_cpu, n_nonfinite_gpu, n, first_bad / (size_t)nv,
               (int)(first_bad % (size_t)nv), (double)cpu[first_bad],
               (double)gpu[first_bad]);
        return 1;
    }

    size_t n_diff = 0;
    double worst = 0, sum = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)cpu[i] - (double)gpu[i]);
        sum += d;
        if (d > worst) worst = d;
        if (memcmp(&cpu[i], &gpu[i], sizeof(float)) != 0) n_diff++;
    }
    double mean_dev = sum / (double)n;

    // Range over REAL logits only: some archs suppress vocabulary entries with
    // a large negative sentinel, which would make a fraction-of-range bound
    // vacuously true.
    double range_sum = 0;
    for (int s2 = 0; s2 < steps; s2++) {
        double lo = 1e300, hi = -1e300;
        for (int v = 0; v < nv; v++) {
            double x = (double)cpu[(size_t)s2 * nv + v];
            if (x <= -1e29) continue;
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        if (hi > lo) range_sum += hi - lo;
    }
    double range = range_sum / steps;
    double frac = range > 0 ? mean_dev / range : 1e9;

    // Anti-vacuity: all-zero logits would compare equal and prove nothing.
    double absmax = 0;
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)cpu[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-6) {
        printf("FAIL: the CPU run produced all-zero logits\n"
               "gpu-identity: FAILED\n");
        return 1;
    }

    printf("  %zu logits over %d positions | %zu differ, mean|dlogit| %.3g "
           "= %.3g of mean range %.3g (limit %g), worst %.3g\n",
           n, steps, n_diff, mean_dev, frac, range, GPU_DEV_FRAC, worst);

    if (frac > GPU_DEV_FRAC) {
        printf("FAIL: CPU and GPU disagree by more than reduction-order "
               "rounding — this is a missing or wrong operation on one side, "
               "not a rounding difference\ngpu-identity: FAILED\n");
        return 1;
    }
    printf("gpu-identity: ok\n");
    return 0;
}

// `--self-test`: the verdict function under the flags this binary was built
// with, on inputs a model run cannot be asked to produce on demand. Each
// case is one deliberate defect class or one legitimate agreement; the
// expected verdict is stated beside it. Exit 0 only when every case reads
// as it should.
static float make_bits(uint32_t b) {
    float x;
    memcpy(&x, &b, sizeof x);
    return x;
}

static int self_test(void) {
    enum { NV = 7, ST = 3, N = NV * ST };
    static float cpu[N], gpu[N];
    int fail = 0;
    struct { const char *name; size_t at; int expect; } cases[] = {
        { "equal finite logits",             (size_t)-1, 0 },
        { "one GPU NaN",                     9,          1 },
        { "one GPU +Inf",                    9,          1 },
        { "one GPU -Inf",                    9,          1 },
        { "one CPU NaN",                     9,          1 },
        { "a decisive finite disagreement",  9,          1 },
        { "a reduction-sized disagreement",  9,          0 },
        { "the masked-vocabulary sentinel",  9,          0 },
    };
    for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
        for (int i = 0; i < N; i++) {
            // a real-looking spread: range about 6 per position
            cpu[i] = (float)((i * 7) % 13) * 0.5f - 3.0f;
            gpu[i] = cpu[i];
        }
        size_t at = cases[c].at;
        switch (c) {
            case 1: gpu[at] = make_bits(0x7fc00000u); break;         // NaN
            case 2: gpu[at] = make_bits(0x7f800000u); break;         // +Inf
            case 3: gpu[at] = make_bits(0xff800000u); break;         // -Inf
            case 4: cpu[at] = make_bits(0x7fc00000u); break;         // NaN
            case 5: gpu[at] = cpu[at] + 3.0f; break;                  // missing op
            case 6: gpu[at] = cpu[at] + 1e-4f; break;                 // rounding
            case 7: cpu[at] = -1e30f; gpu[at] = -1e30f; break;        // sentinel
            default: break;
        }
        printf("self-test: %s (expect %s)\n", cases[c].name,
               cases[c].expect ? "FAILED" : "ok");
        int rc = compare_logits(cpu, gpu, NV, ST);
        if (rc != cases[c].expect) {
            printf("SELF-TEST FAIL: '%s' read as %s, expected %s\n",
                   cases[c].name, rc ? "FAILED" : "ok",
                   cases[c].expect ? "FAILED" : "ok");
            fail = 1;
        }
    }
    printf(fail ? "gpu-identity self-test: FAILED\n"
                : "gpu-identity self-test: ok (%zu cases)\n",
           sizeof cases / sizeof *cases);
    return fail;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--self-test") == 0) return self_test();
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    if (argc > 2) g_gpu_layers = atoi(argv[2]);
    if (argc > 3) g_n_batch = atoi(argv[3]) > 0 ? atoi(argv[3]) : N_BATCH;

    f16_init();
    // Pin the identity route. The tiled prefill GEMM (RUNNER_METAL_MM), the
    // reassociating decode matvec (RUNNER_METAL_MV), and the grouped-MMA MoE
    // prefill (RUNNER_METAL_MOE_MM, default on since the 2026-09-01
    // ratification) are deliberately NOT byte-identical -- they answer to
    // test_tc_tol.c, test_mv_tol.c, and test_moe_mm_ab.c respectively. This
    // gate is about the contract they are exceptions to, so it forces all
    // off; leaving them at their defaults would make it fail by design.
    gpu_tc_force(0);
    gpu_mv_force(0);
    setenv_compat("RUNNER_METAL_MOE_MM", "0");

    gguf_file gf;
    if (!gguf_open(&gf, path)) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    tokenizer tk;
    if (!tokenizer_init(&tk, &gf)) {
        fprintf(stderr, "cannot init tokenizer for %s\n", path);
        gguf_close(&gf); return 1;
    }
    static int32_t toks[MAX_TOK];
    int n_tok = tok_encode(&tk, TEXT, toks, MAX_TOK, true, false);
    tokenizer_free(&tk);
    gguf_close(&gf);
    if (n_tok < 8) { fprintf(stderr, "text tokenized to %d tokens\n", n_tok); return 1; }
    for (int i = n_tok; i < MAX_TOK; i++) toks[i] = toks[i - n_tok + 1];
    n_tok = MAX_TOK;

    printf("gpu-identity: %s | %d tokens, %d teacher-forced positions, "
           "prefill batch %d\n", path, n_tok, STEPS, g_n_batch);

    int nv_cpu = 0, nv_gpu = 0;
    bool cpu_used_gpu = false, gpu_used_gpu = false;
    float *cpu = run(path, GPU_OFF, &nv_cpu, &cpu_used_gpu, toks, n_tok);
    if (!cpu) { printf("gpu-identity: FAILED (cpu run)\n"); return 1; }
    float *gpu = run(path, GPU_AUTO, &nv_gpu, &gpu_used_gpu, toks, n_tok);
    if (!gpu) { free(cpu); printf("gpu-identity: FAILED (gpu run)\n"); return 1; }

    if (!gpu_used_gpu) {
        printf("  skipped: no GPU, or this model fell back to the CPU — a "
               "comparison of the CPU against itself proves nothing\n"
               "gpu-identity: ok (skipped)\n");
        free(cpu); free(gpu);
        return 0;
    }
    if (nv_cpu != nv_gpu) {
        printf("FAIL: vocab size differs (%d vs %d)\n", nv_cpu, nv_gpu);
        free(cpu); free(gpu);
        return 1;
    }

    int rc = compare_logits(cpu, gpu, nv_cpu, STEPS);
    free(cpu); free(gpu);
    return rc;
}
