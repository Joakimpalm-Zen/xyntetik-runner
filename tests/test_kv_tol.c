// Phase 8: a SOUND correctness gate for the q8_0 KV cache on the GPU.
//
// The gate this replaces asserted that a free-running greedy generation with
// a q8 cache produced token-identical text on GPU and CPU. That gate cannot
// be made to pass, and — more importantly — it was never measuring what it
// claimed to measure:
//
//   * Free-running generation feeds its own argmax back in. One last-ulp
//     difference at one near-tie changes the next token, which changes every
//     token after it. So the metric is not "how wrong is the kernel" but
//     "how many near-ties did this prompt happen to contain", which is a
//     property of the prompt and the model, not of the code under test.
//   * q8_0's step is amax/127 per 32-value block, roughly 15x coarser than
//     fp16 near the same magnitude. Reassociation noise that fp16 absorbs
//     silently can move a q8 value by a whole quantization level. This is
//     not hypothetical: on gemma-3-4b the same divergence reproduces with no
//     GPU involved at all, CPU-only, purely by changing -b 64 to -b 1.
//
// So token identity under q8 is not a property the implementation can have,
// and a gate that demands it is a gate that will be disabled.
//
// What this test measures instead is TEACHER-FORCED agreement. Every
// configuration is fed the *same* fixed token sequence at every position, so
// no configuration is ever reacting to its own earlier divergence, and the
// difference between two logit vectors at a given position is purely numeric.
//
// The tolerance is not a magic constant, and it is not a guess about how
// close two q8 implementations "ought" to be. It is calibrated against a
// NEGATIVE CONTROL: the CPU q8 path compared against *itself*, run with a
// different prompt batch size.
//
//     reassoc = mean|q8_cpu(b=1) - q8_cpu(b=64)|    no GPU involved at all
//     impl    = mean|q8_gpu      - q8_cpu(b=64)|
//
// Changing the batch size only reassociates the same sums — it is a
// mathematically equivalent computation, so `reassoc` is pure floating-point
// noise passed through the quantizer. It is the floor: no faithful q8
// implementation can do better, because q8's step is ~15x coarser than fp16
// and last-ulp input differences therefore reach the logits amplified.
//
// The gate is that the GPU differs from the CPU by no more than a small
// multiple of what the CPU already differs from itself. That is a property a
// correct implementation genuinely has, it needs no per-model tuning, and a
// layout bug, a wrong scale, or a mis-sliced head cannot satisfy it — those
// produce differences orders of magnitude above the reassociation floor.
//
// mean|q8_cpu - f16_cpu| (the inherent cost of the format) is reported too,
// as context for how large all of these numbers are, but it is deliberately
// NOT the gate: the GPU/CPU gap is a sizeable fraction of it for exactly the
// amplification reason above, and a gate demanding otherwise would be the
// same mistake as demanding token identity, one level down.
//
// The f16 invariant is checked the strict way, unchanged: f16 GPU must be
// token-identical to f16 CPU. That one IS a property the implementation has
// and must keep.
//
//     ./test-kv-tol models/Qwen3-4B-Q4_K_M.gguf
//     ./test-kv-tol models/Qwen3-4B-Q4_K_M.gguf 20   # cap VRAM at 20%
//
// Default model is test.gguf, whose head_dim is not a multiple of 32, so the
// q8 half self-skips there and `make test` still exercises the harness.
#include "runner.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 192 tokens keeps the toy test.gguf (256 training positions) off the YaRN
// path, so `make test` measures the plain rope. N_BATCH must divide the
// prefill the same way for every configuration, or the comparison would be
// measuring batch reassociation instead of the cache format.
enum { STEPS = 64, MAX_TOK = 192, N_BATCH = 64 };

// How many times the reassociation floor the GPU is allowed to sit at. The
// GPU reorders far more than a batch-size change does (different thread
// counts, different accumulation trees, fused multiply-adds in different
// places), so demanding 1.0x would be wrong; but a real layout or scale bug
// lands orders of magnitude out, not at 3x.
#define REASSOC_SLACK  3.0
// FLOOR_EPS: the reassociation floor is measured, and on a model whose head
// is a single cache block (the 2-layer fixture: head_dim 16, one fp4 block)
// the CPU's batch-1 and batch-N runs are numerically IDENTICAL, so the floor
// is 0 and any rounding-level GPU difference divides to infinity. A mean
// |dlogit| below this is the two runs agreeing to fp32 rounding, and the
// ratio is taken against it rather than against zero.
#define FLOOR_EPS      1e-5
// TIE_FRAC: a top-1 disagreement is only excusable if the two candidates were
// within this fraction of the logit range of each other — i.e. a genuine
// near-tie. A disagreement with a decisive margin is a bug no matter what the
// average error looks like.
#define TIE_FRAC       0.02
// and near-ties must stay rare even so
#define DISAGREE_MAX   0.05
// TOP1_SLACK: the two limits above are the bar for a well-behaved format. On a
// format that is chaotic for a given model (fp4 on Qwen2.5-1.5B moves 61 of 64
// decisions against f16; on Llama-3.2-1B the CPU flips 5 of 64 against ITSELF
// under a batch-size change with a worst margin of 0.0305) the fixed numbers
// sit below the CPU's own flip rate, and a GPU that agrees with the CPU as
// well as the CPU agrees with itself would fail them for no reason. So, as
// the magnitude criterion already does, the GPU-vs-CPU top-1 checks are also
// taken against the measured CPU-vs-CPU floor: the limit is the larger of the
// fixed bar and TOP1_SLACK times the floor's own flip rate / worst margin. A
// well-behaved format (q8: 2 of 64, 0.0020) never sees the relaxation. The
// q8-vs-f16 tie check is the FORMAT's contract and stays fixed.
#define TOP1_SLACK     2.0

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// Natural text, not random ids: the near-tie structure of a real logit
// distribution is what production actually sees, and random token ids produce
// a flat confused distribution where everything is a near-tie.
static const char *TEXT =
    "The city of Lisbon sits on seven hills above the Tagus estuary, and its "
    "oldest quarter survived the 1755 earthquake largely intact because the "
    "bedrock there is firmer than the reclaimed ground downriver. Rebuilding "
    "the lower town took decades, and the grid of streets laid out afterwards "
    "was among the first in Europe designed with seismic loads in mind. "
    "def parse_header(buf, size):\n"
    "    if size < 8:\n"
    "        raise ValueError('short header')\n"
    "    magic, version = struct.unpack('<II', buf[:8])\n"
    "    return magic, version\n"
    "In 1929 the observatory published a revised catalogue listing 4218 "
    "objects, of which roughly one in nine turned out on later inspection to "
    "be a duplicate entry under a second designation. The correction was not "
    "issued until 1934, by which time three separate groups had independently "
    "noticed the discrepancy and written to the editors about it.";

// Percent of total VRAM the GPU configs may use (0 = take all free VRAM,
// the normal behavior). Settable so the gate can run on a shared GPU without
// starving whatever else is on it.
static int g_reserve_vram_pct = 0;

typedef struct {
    const char *name;
    bool        kv_q8;
    bool        kv_fp4;
    bool        kv_split;    // --kv k8v4: K q8_0, V fp4
    int         gpu_mode;
    int         n_batch;     // prompt batch size; varying it only reassociates
    bool        available;   // config actually ran as requested
    float      *logits;      // [STEPS][n_vocab], owned
    int32_t    *top1;        // [STEPS], owned
} config;

static model_params params_for(const config *c, int n_ctx) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = c->gpu_mode;
    p.n_threads = 4;
    p.n_ctx     = n_ctx;
    p.n_batch   = c->n_batch;
    p.kv_q8     = c->kv_q8;
    p.kv_fp4    = c->kv_fp4;
    p.kv_split  = c->kv_split;
    p.reserve_vram_pct = g_reserve_vram_pct;
    return p;
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

// Second-best value, used to size the margin at a disagreement.
static float top2_gap(const float *v, int n, int best) {
    float second = -FLT_MAX;
    for (int i = 0; i < n; i++)
        if (i != best && v[i] > second) second = v[i];
    return v[best] - second;
}

static float logit_range(const float *v, int n) {
    float lo = FLT_MAX, hi = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    return hi - lo;
}

// Teacher-forced sweep: position i always sees the real token i, never this
// configuration's own prediction. That is what makes the per-position logit
// comparison below a comparison of arithmetic rather than of drift.
static bool run_config(config *c, const char *path, const int32_t *toks,
                       int n_tok, int n_vocab) {
    model_t m;
    memset(&m, 0, sizeof(m));
    model_params p = params_for(c, n_tok + 8);
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "  %-12s load failed\n", c->name);
        return false;
    }
    // Did we get the configuration we asked for? q8 silently falls back to
    // f16 when head_dim is not block-aligned or the backend lacks kernels,
    // and asking for a GPU on a CPU-only build is not an error either.
    bool got_q8  = m.kv_q8;
    bool got_fp4 = m.kv_fp4;
    bool got_split = m.kv_split;
    bool got_gpu = m.gpu != NULL && m.gpu_layers > 0;

    // Cross-platform safety invariant, checked on every platform including
    // the ones whose backend cannot do q8 at all. A backend that reports no
    // q8 attention kernels must never end up with a q8 cache while it is
    // running layers: those kernels would read q8_0 blocks as fp16 and return
    // fluent, plausible, wrong text — the one failure mode no downstream test
    // would catch. This is the assertion that makes metal.m's
    // gpu_kv_q8_ok()==false a guarantee rather than a stub nobody checks.
    ck(!(got_q8 && got_gpu && !gpu_kv_q8_ok()),
       "a backend without q8 attention kernels never runs a q8 KV cache");
    ck(!(got_fp4 && got_gpu && !gpu_kv_fp4_ok()),
       "a backend without fp4 attention kernels never runs an fp4 KV cache");
    ck(!(got_split && got_gpu && !(gpu_kv_q8_ok() && gpu_kv_fp4_ok())),
       "a backend without both q8 and fp4 kernels never runs a k8v4 KV cache");
    ck(!(got_split && (got_q8 || got_fp4)),
       "the k8v4 split is its own layout, never combined with q8 or fp4");

    if (c->kv_q8 != got_q8 || c->kv_fp4 != got_fp4 || c->kv_split != got_split ||
        (c->gpu_mode == GPU_AUTO) != got_gpu) {
        fprintf(stderr, "  %-12s skipped (asked kv_q8=%d kv_fp4=%d k8v4=%d gpu=%d, got "
                "kv_q8=%d kv_fp4=%d k8v4=%d gpu=%d/%d layers)\n", c->name, (int)c->kv_q8,
                (int)c->kv_fp4, (int)c->kv_split, c->gpu_mode == GPU_AUTO,
                (int)got_q8, (int)got_fp4, (int)got_split,
                m.gpu_layers, m.n_layer);
        model_free(&m);
        return false;
    }

    c->logits = malloc(sizeof(float) * (size_t)STEPS * (size_t)n_vocab);
    c->top1   = malloc(sizeof(int32_t) * STEPS);
    if (!c->logits || !c->top1) { model_free(&m); return false; }

    // Prefill in n_batch-sized chunks: model_forward_batch processes at most
    // n_batch tokens per call, and only the final chunk needs logits.
    int prefill = n_tok - STEPS;
    float *lg = NULL;
    for (int off = 0; off < prefill; off += c->n_batch) {
        int n = prefill - off < c->n_batch ? prefill - off : c->n_batch;
        lg = model_forward_batch(&m, toks + off, n, off, off + n == prefill);
        if (off + n == prefill && !lg) { model_free(&m); return false; }
    }
    if (!lg) { model_free(&m); return false; }
    memcpy(c->logits, lg, sizeof(float) * (size_t)n_vocab);
    c->top1[0] = (int32_t)argmax(lg, n_vocab);

    for (int s = 1; s < STEPS; s++) {
        // feed the REAL token at prefill+s-1, not our own argmax
        lg = model_forward(&m, toks[prefill + s - 1], prefill + s - 1);
        if (!lg) { model_free(&m); return false; }
        memcpy(c->logits + (size_t)s * n_vocab, lg,
               sizeof(float) * (size_t)n_vocab);
        c->top1[s] = (int32_t)argmax(lg, n_vocab);
    }
    model_free(&m);

    // Anti-vacuity guard. Every comparison below is a *difference* between two
    // configurations, and differences between two degenerate runs are zero —
    // which reads as a perfect pass. So refuse to draw any conclusion from a
    // configuration that did not actually produce logits.
    double absmax = 0;
    for (size_t i = 0; i < (size_t)STEPS * (size_t)n_vocab; i++) {
        double a = fabs((double)c->logits[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-6) {
        fprintf(stderr, "FAIL: %s produced all-zero logits — this "
                "configuration measured nothing\n", c->name);
        g_fail = 1;
        return false;
    }

    c->available = true;
    return true;
}

// mean absolute logit difference over every position and every vocab entry
static double mean_abs_diff(const config *a, const config *b, int n_vocab) {
    double sum = 0;
    size_t n = (size_t)STEPS * (size_t)n_vocab;
    for (size_t i = 0; i < n; i++)
        sum += fabs((double)a->logits[i] - (double)b->logits[i]);
    return sum / (double)n;
}

// how often the two configs' argmax differs, and the worst margin at which
// they disagreed (as a fraction of the logit range at that position)
static void top1_stats(const config *a, const config *b, int n_vocab,
                       int *n_diff, double *worst_frac) {
    *n_diff = 0;
    *worst_frac = 0.0;
    for (int s = 0; s < STEPS; s++) {
        if (a->top1[s] == b->top1[s]) continue;
        (*n_diff)++;
        const float *row = a->logits + (size_t)s * n_vocab;
        float gap   = top2_gap(row, n_vocab, a->top1[s]);
        float range = logit_range(row, n_vocab);
        double frac = range > 0 ? (double)gap / (double)range : 0.0;
        if (frac > *worst_frac) *worst_frac = frac;
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    if (argc > 2) g_reserve_vram_pct = atoi(argv[2]);

    // Builds the fp16 -> fp32 lookup table. Without it every f16_to_f32 call
    // returns 0, so the whole CPU forward pass — embeddings, KV reads, logits
    // — silently evaluates to zeros, and any test comparing two such runs
    // passes because 0 == 0. The all-zero guard below is what catches this
    // class of mistake rather than trusting the call to stay here.
    f16_init();

    // Instrument isolation: this gate measures the KV CACHE FORMAT, and its
    // strict half asserts f16-GPU == f16-CPU token identity. The tensor-core
    // GEMM is promoted by default on some (type, arch) combos and is NOT
    // bit-identical to the scalar CPU GEMM, so left free it would leak into
    // exactly that comparison. Pin the GEMM path scalar; the TC path has its
    // own tolerance gate (test_tc_tol.c).
    gpu_tc_force(0);

    // tokenize the fixed text with the model's own tokenizer
    gguf_file gf;
    if (!gguf_open(&gf, path)) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    tokenizer tk;
    if (!tokenizer_init(&tk, &gf)) {
        fprintf(stderr, "cannot init tokenizer for %s\n", path);
        gguf_close(&gf);
        return 1;
    }
    static int32_t toks[MAX_TOK];
    int n_tok = tok_encode(&tk, TEXT, toks, MAX_TOK, true, false);
    tokenizer_free(&tk);
    gguf_close(&gf);

    if (n_tok < 16) {
        fprintf(stderr, "text tokenized to only %d tokens\n", n_tok);
        return 1;
    }
    // A byte-level toy vocab needs far more tokens for this text than a real
    // BPE vocab does; a real vocab may need fewer than MAX_TOK. Tile to a
    // fixed length either way, so every model is gated over the same number
    // of KV rows.
    for (int i = n_tok; i < MAX_TOK; i++) toks[i] = toks[i - n_tok + 1];
    n_tok = MAX_TOK;

    printf("kv-tol: %s | %d tokens, %d teacher-forced positions\n",
           path, n_tok, STEPS);

    enum { N_CFG = 11 };
    config cfgs[N_CFG] = {
        { "f16-cpu",   false, false, false, GPU_OFF,  N_BATCH, false, NULL, NULL },
        { "f16-gpu",   false, false, false, GPU_AUTO, N_BATCH, false, NULL, NULL },
        { "q8-cpu",    true,  false, false, GPU_OFF,  N_BATCH, false, NULL, NULL },
        { "q8-gpu",    true,  false, false, GPU_AUTO, N_BATCH, false, NULL, NULL },
        // the negative control: same code, same device, same cache format,
        // only the prompt batching differs, so every difference it shows is
        // pure reassociation noise amplified by the q8 quantizer
        { "q8-cpu-b1", true,  false, false, GPU_OFF,  1,       false, NULL, NULL },
        // the same three arms for the fp4 cache (2026-09-18)
        { "fp4-cpu",   false, true,  false, GPU_OFF,  N_BATCH, false, NULL, NULL },
        { "fp4-gpu",   false, true,  false, GPU_AUTO, N_BATCH, false, NULL, NULL },
        { "fp4-cpu-b1", false, true, false, GPU_OFF,  1,       false, NULL, NULL },
        // and for the k8v4 split (K q8_0, V fp4): the one layout where the K
        // and V caches have different row geometry, so every offset/stride
        // site that once assumed one row size is exercised here
        { "k8v4-cpu",   false, false, true, GPU_OFF,  N_BATCH, false, NULL, NULL },
        { "k8v4-gpu",   false, false, true, GPU_AUTO, N_BATCH, false, NULL, NULL },
        { "k8v4-cpu-b1", false, false, true, GPU_OFF, 1,       false, NULL, NULL },
    };

    // n_vocab is a property of the file; read it from the first load
    int n_vocab = 0;
    {
        model_t probe;
        memset(&probe, 0, sizeof(probe));
        model_params p = params_for(&cfgs[0], n_tok + 8);
        if (!model_load(&probe, path, &p)) {
            fprintf(stderr, "cannot load %s\n", path);
            return 1;
        }
        n_vocab = probe.n_vocab;
        model_free(&probe);
    }

    for (int i = 0; i < N_CFG; i++)
        run_config(&cfgs[i], path, toks, n_tok, n_vocab);

    config *f16c = &cfgs[0], *f16g = &cfgs[1], *q8c = &cfgs[2],
           *q8g  = &cfgs[3], *q8b1 = &cfgs[4],
           *fp4c = &cfgs[5], *fp4g = &cfgs[6], *fp4b1 = &cfgs[7],
           *k84c = &cfgs[8], *k84g = &cfgs[9], *k84b1 = &cfgs[10];

    // ---------------------------------------------------------- invariant
    // fp16 GPU is token-identical to fp16 CPU. Strict, and staying strict:
    // this one the implementation genuinely has.
    if (f16c->available && f16g->available) {
        int n_diff;
        double worst;
        top1_stats(f16c, f16g, n_vocab, &n_diff, &worst);
        printf("  f16-cpu vs f16-gpu : top1 diff %d/%d\n", n_diff, STEPS);
        ck(n_diff == 0, "fp16 GPU is token-identical to fp16 CPU "
                        "(teacher-forced)");
    } else {
        printf("  f16-cpu vs f16-gpu : skipped (no GPU)\n");
    }

    // ------------------------------------------- q8 on the CPU, no GPU needed
    //
    // The GPU gate below cannot see a defect in the CPU q8 path itself, and
    // that is where a field report landed: gemma-4-26B under --kv q8 on CPU
    // produced deterministic garbage on an M2 Pro while its f16 control was
    // coherent, with a Llama-3.2-3B q8 control clean on the same machine
    // (workmac issue 6, 2026-08-07). The whole q8 half of this harness used
    // to skip without a GPU, so a CPU-only box -- or any model too large for
    // the device working set, which is exactly the class that needs q8 --
    // exercised none of it.
    //
    // These two arms need no GPU: the reassociation floor is CPU-vs-CPU by
    // construction, and the format cost is CPU q8 vs CPU f16. Gating the
    // second against the first catches a CPU q8 path that has stopped
    // agreeing with fp16 beyond what the quantiser explains.
    if (q8c->available && q8b1->available && f16c->available) {
        double quant_err = mean_abs_diff(q8c, f16c, n_vocab);
        double reassoc   = mean_abs_diff(q8b1, q8c, n_vocab);
        int n_diff;
        double worst;
        top1_stats(q8c, f16c, n_vocab, &n_diff, &worst);
        printf("  q8-cpu    vs f16-cpu : mean|dlogit| %.6f, top1 diff %d/%d, "
               "worst margin %.4f (reassoc floor %.6f)\n",
               quant_err, n_diff, STEPS, worst, reassoc);
        // q8 is a lossy cache, so f16 disagreement is expected -- but every
        // disagreement must be a near-tie. A real q8 bug moves a decision,
        // not a coin flip, which is what the field report described.
        ck(worst <= TIE_FRAC,
           "every q8-CPU/f16-CPU token disagreement is a near-tie, not a "
           "decision");
    } else {
        printf("  q8 cpu gate        : skipped (q8 unavailable)\n");
    }

    // -------------------------------------------------------- q8 vs the GPU
    if (q8c->available && q8g->available && q8b1->available &&
        f16c->available) {
        double quant_err  = mean_abs_diff(q8c, f16c, n_vocab);  // context only
        double reassoc    = mean_abs_diff(q8b1, q8c, n_vocab);  // the floor
        double impl_err   = mean_abs_diff(q8g, q8c, n_vocab);   // the measure
        double ratio = impl_err / (reassoc > FLOOR_EPS ? reassoc : FLOOR_EPS);

        int n_diff;
        double worst;
        top1_stats(q8c, q8g, n_vocab, &n_diff, &worst);
        double frac = (double)n_diff / STEPS;

        printf("  q8-cpu    vs f16-cpu : mean|dlogit| %.6f   (what the format "
               "costs; context, not the gate)\n", quant_err);
        printf("  q8-cpu-b1 vs q8-cpu  : mean|dlogit| %.6f   (reassociation "
               "floor, CPU only)\n", reassoc);
        // how often the CPU flips a token against ITSELF under legal
        // reassociation: the floor the GPU top-1 checks are taken against
        int nd_b1; double w_b1;
        top1_stats(q8c, q8b1, n_vocab, &nd_b1, &w_b1);
        printf("  q8-cpu-b1 vs q8-cpu  : top1 diff %d/%d, worst margin %.4f "
               "(the floor's own flip rate)\n", nd_b1, STEPS, w_b1);
        double lim_frac  = fmax(DISAGREE_MAX, TOP1_SLACK * (double)nd_b1 / STEPS);
        double lim_worst = fmax(TIE_FRAC, TOP1_SLACK * w_b1);
        printf("  q8-gpu    vs q8-cpu  : mean|dlogit| %.6f   %.2fx the floor "
               "(limit %.1fx)\n", impl_err, ratio, REASSOC_SLACK);
        printf("  q8-gpu    vs q8-cpu  : top1 diff %d/%d (%.1f%%, limit %.0f%%)"
               ", worst margin %.4f of range (limit %.3f)\n",
               n_diff, STEPS, 100.0 * frac, 100.0 * DISAGREE_MAX,
               worst, TIE_FRAC);

        ck(ratio <= REASSOC_SLACK,
           "q8 GPU differs from q8 CPU no more than q8 CPU differs from "
           "itself under legal reassociation");
        printf("  q8 parity limits used: top1 %.1f%%, worst margin %.4f (the fixed bar or "
               "%.1fx the floor's own, whichever is larger)\n", 100.0 * lim_frac, lim_worst, TOP1_SLACK);
        ck(frac <= lim_frac,
           "q8 GPU and q8 CPU disagree on a token no more often than the CPU "
           "disagrees with itself (or the fixed bar)");
        ck(worst <= lim_worst,
           "every q8 GPU/CPU token disagreement is a near-tie by the CPU's own "
           "standard (or the fixed bar)");
    } else {
        printf("  q8 tolerance gate  : skipped (q8 or GPU unavailable)\n");
    }

    // ------------------------------------------------------------ fp4 arms
    // fp4 is a 4-bit cache and may legitimately move a decision, so its
    // f16 disagreement is REPORTED as the format's cost and not gated (the
    // fidelity protocol, kld-compare-raw against the f16 cache, is where
    // the format answers for itself). What IS gated is implementation
    // parity: the GPU fp4 path may differ from the CPU fp4 path by no more
    // than the CPU differs from itself under legal reassociation, since the
    // stored rows are byte-identical across backends by construction.
    if (fp4c->available && f16c->available) {
        double quant_err = mean_abs_diff(fp4c, f16c, n_vocab);
        int n_diff;
        double worst;
        top1_stats(fp4c, f16c, n_vocab, &n_diff, &worst);
        printf("  fp4-cpu   vs f16-cpu : mean|dlogit| %.6f, top1 diff %d/%d, "
               "worst margin %.4f (the format's cost; reported, not gated)\n",
               quant_err, n_diff, STEPS, worst);
    } else {
        printf("  fp4 cpu report     : skipped (fp4 unavailable)\n");
    }
    if (fp4c->available && fp4g->available && fp4b1->available) {
        double reassoc  = mean_abs_diff(fp4b1, fp4c, n_vocab);
        double impl_err = mean_abs_diff(fp4g, fp4c, n_vocab);
        double ratio = impl_err / (reassoc > FLOOR_EPS ? reassoc : FLOOR_EPS);
        int n_diff;
        double worst;
        top1_stats(fp4c, fp4g, n_vocab, &n_diff, &worst);
        double frac = (double)n_diff / STEPS;
        printf("  fp4-cpu-b1 vs fp4-cpu: mean|dlogit| %.6f   (reassociation "
               "floor, CPU only)\n", reassoc);
        // how often the CPU flips a token against ITSELF under legal
        // reassociation: the floor the GPU top-1 checks are taken against
        int nd_b1; double w_b1;
        top1_stats(fp4c, fp4b1, n_vocab, &nd_b1, &w_b1);
        printf("  fp4-cpu-b1 vs fp4-cpu: top1 diff %d/%d, worst margin %.4f "
               "(the floor's own flip rate)\n", nd_b1, STEPS, w_b1);
        double lim_frac  = fmax(DISAGREE_MAX, TOP1_SLACK * (double)nd_b1 / STEPS);
        double lim_worst = fmax(TIE_FRAC, TOP1_SLACK * w_b1);
        printf("  fp4-gpu   vs fp4-cpu : mean|dlogit| %.6f   %.2fx the floor "
               "(limit %.1fx)\n", impl_err, ratio, REASSOC_SLACK);
        printf("  fp4-gpu   vs fp4-cpu : top1 diff %d/%d (%.1f%%, limit %.0f%%)"
               ", worst margin %.4f of range (limit %.3f)\n",
               n_diff, STEPS, 100.0 * frac, 100.0 * DISAGREE_MAX,
               worst, TIE_FRAC);
        ck(ratio <= REASSOC_SLACK,
           "fp4 GPU differs from fp4 CPU no more than fp4 CPU differs from "
           "itself under legal reassociation");
        printf("  fp4 parity limits used: top1 %.1f%%, worst margin %.4f (the fixed bar or "
               "%.1fx the floor's own, whichever is larger)\n", 100.0 * lim_frac, lim_worst, TOP1_SLACK);
        ck(frac <= lim_frac,
           "fp4 GPU and fp4 CPU disagree on a token no more often than the CPU "
           "disagrees with itself (or the fixed bar)");
        ck(worst <= lim_worst,
           "every fp4 GPU/CPU token disagreement is a near-tie by the CPU's own "
           "standard (or the fixed bar)");
    } else {
        printf("  fp4 tolerance gate : skipped (fp4 or GPU unavailable)\n");
    }

    // ----------------------------------------------------------- k8v4 arms
    // Same contract as fp4: the f16 disagreement is the format's cost and is
    // reported; GPU-vs-CPU parity is gated at the reassociation floor. This
    // is the arm that catches a K/V geometry mix-up (a V row read at a K
    // stride, or the V cache sized from the K format), which no single-kind
    // arm can see because there the two geometries coincide.
    if (k84c->available && f16c->available) {
        double quant_err = mean_abs_diff(k84c, f16c, n_vocab);
        int n_diff;
        double worst;
        top1_stats(k84c, f16c, n_vocab, &n_diff, &worst);
        printf("  k8v4-cpu  vs f16-cpu : mean|dlogit| %.6f, top1 diff %d/%d, "
               "worst margin %.4f (the format's cost; reported, not gated)\n",
               quant_err, n_diff, STEPS, worst);
    } else {
        printf("  k8v4 cpu report    : skipped (k8v4 unavailable)\n");
    }
    if (k84c->available && k84g->available && k84b1->available) {
        double reassoc  = mean_abs_diff(k84b1, k84c, n_vocab);
        double impl_err = mean_abs_diff(k84g, k84c, n_vocab);
        double ratio = impl_err / (reassoc > FLOOR_EPS ? reassoc : FLOOR_EPS);
        int n_diff;
        double worst;
        top1_stats(k84c, k84g, n_vocab, &n_diff, &worst);
        double frac = (double)n_diff / STEPS;
        printf("  k8v4-cpu-b1 vs k8v4-cpu: mean|dlogit| %.6f   (reassociation "
               "floor, CPU only)\n", reassoc);
        // how often the CPU flips a token against ITSELF under legal
        // reassociation: the floor the GPU top-1 checks are taken against
        int nd_b1; double w_b1;
        top1_stats(k84c, k84b1, n_vocab, &nd_b1, &w_b1);
        printf("  k8v4-cpu-b1 vs k8v4-cpu: top1 diff %d/%d, worst margin %.4f "
               "(the floor's own flip rate)\n", nd_b1, STEPS, w_b1);
        double lim_frac  = fmax(DISAGREE_MAX, TOP1_SLACK * (double)nd_b1 / STEPS);
        double lim_worst = fmax(TIE_FRAC, TOP1_SLACK * w_b1);
        printf("  k8v4-gpu  vs k8v4-cpu: mean|dlogit| %.6f   %.2fx the floor "
               "(limit %.1fx)\n", impl_err, ratio, REASSOC_SLACK);
        printf("  k8v4-gpu  vs k8v4-cpu: top1 diff %d/%d (%.1f%%, limit %.0f%%)"
               ", worst margin %.4f of range (limit %.3f)\n",
               n_diff, STEPS, 100.0 * frac, 100.0 * DISAGREE_MAX,
               worst, TIE_FRAC);
        ck(ratio <= REASSOC_SLACK,
           "k8v4 GPU differs from k8v4 CPU no more than k8v4 CPU differs from "
           "itself under legal reassociation");
        printf("  k8v4 parity limits used: top1 %.1f%%, worst margin %.4f (the fixed bar or "
               "%.1fx the floor's own, whichever is larger)\n", 100.0 * lim_frac, lim_worst, TOP1_SLACK);
        ck(frac <= lim_frac,
           "k8v4 GPU and k8v4 CPU disagree on a token no more often than the CPU "
           "disagrees with itself (or the fixed bar)");
        ck(worst <= lim_worst,
           "every k8v4 GPU/CPU token disagreement is a near-tie by the CPU's own "
           "standard (or the fixed bar)");
    } else {
        printf("  k8v4 tolerance gate: skipped (k8v4 or GPU unavailable)\n");
    }

    for (int i = 0; i < N_CFG; i++) { free(cfgs[i].logits); free(cfgs[i].top1); }
    printf(g_fail ? "kv-tol: FAILED\n" : "kv-tol: ok\n");
    return g_fail;
}
