// Memory-placement arithmetic that no fixture can reach: the reservation
// auto-fit (`-c 0` with --reserve-ram/--reserve-vram) and the KV-evicts-weights
// trade note.
//
// This gate exists because the behaviour it covers is unreachable on any
// machine this project is developed on. The auto-fit only runs when a budget is
// tight enough to force the context down, and the per-slot activation head
// alone is 256 MB: on an 8 GB host even a 1% reservation leaves negative room,
// so the branch returns "does not fit" and nothing downstream is exercised. The
// numbers that matter belong to 7B-and-up models on 24 GB cards. Rather than
// pretend a 135M fixture can stand in for that, the arithmetic is fed those
// numbers directly.
//
// The specific regression guarded here is real and is recorded in model.c: the
// reservation is a budget for the SERVER, not for one sequence. Weights are
// uploaded once and shared; the KV cache and activation head are paid PER SLOT.
// Billing both once over-committed a multi-slot server by nearly the slot
// count.
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "model.h"
#include "gguf.h"

static int g_fail = 0;
static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else       { fprintf(stderr, "ok: %s\n", what); }
}

#define HEAD ((uint64_t)MODEL_AUTOFIT_HEAD)

// Whatever the auto-fit returns, actually running that many slots at that
// context must fit inside the budget. This is the whole contract.
static int fits(uint64_t budget, uint64_t weights, uint64_t kv_per_tok,
                int n_seq, long long ctx) {
    return weights + HEAD * (uint64_t)n_seq +
           (uint64_t)ctx * kv_per_tok * (uint64_t)n_seq <= budget;
}

// The case from model.c: Qwen2.5-7B, `--reserve-vram 40 --parallel 4 -c 0` on a
// 24 GiB card. 40% of the card is a 10.15 GB reservation; each slot's cache
// measured 1.88 GB at a 32768 context, which is 57,373 bytes per token. The
// weight figure is the remainder implied by the 12.47 GB that was actually
// allocated, not an independent measurement.
#define Q_BUDGET   10150000000ull
#define Q_WEIGHTS   4950000000ull
#define Q_KV_TOK          57373ull
#define Q_TRAIN            32768

static void test_multislot_is_not_billed_once(void) {
    // The bug: size the context as if one slot existed, then run four.
    long long one = model_autofit_tokens(Q_BUDGET, Q_WEIGHTS, HEAD, Q_KV_TOK, 1);
    ck(one > 0, "the single-slot case fits at all");
    long long buggy = model_autofit_clamp(one, Q_TRAIN);
    ck(!fits(Q_BUDGET, Q_WEIGHTS, Q_KV_TOK, 4, buggy),
       "billing one slot and running four overruns the reservation");

    // The fix: four slots, sized for four slots.
    long long four = model_autofit_tokens(Q_BUDGET, Q_WEIGHTS, HEAD, Q_KV_TOK, 4);
    ck(four > 0, "the four-slot case still fits something");
    ck(four < one, "more slots means a smaller window");
    ck(fits(Q_BUDGET, Q_WEIGHTS, Q_KV_TOK, 4,
            model_autofit_clamp(four, Q_TRAIN)),
       "four slots sized for four slots stay inside the reservation");
}

// The contract must hold across the whole space, not just one anecdote.
static void test_budget_is_never_exceeded(void) {
    const uint64_t budgets[] = { 4ull<<30, 10ull<<30, 24ull<<30, 80ull<<30 };
    const uint64_t weights[] = { 1ull<<30, 4ull<<30, 13ull<<30 };
    const uint64_t kvs[]     = { 16384, 57373, 262144 };
    const int      seqs[]    = { 1, 2, 4, 8 };
    int checked = 0, bad = 0;
    for (unsigned b = 0; b < sizeof budgets / sizeof *budgets; b++)
    for (unsigned w = 0; w < sizeof weights / sizeof *weights; w++)
    for (unsigned k = 0; k < sizeof kvs / sizeof *kvs; k++)
    for (unsigned s = 0; s < sizeof seqs / sizeof *seqs; s++) {
        long long fit = model_autofit_tokens(budgets[b], weights[w], HEAD,
                                             kvs[k], seqs[s]);
        if (fit <= 0) continue;            // "does not fit" is a valid answer
        checked++;
        if (!fits(budgets[b], weights[w], kvs[k], seqs[s], fit)) bad++;
    }
    ck(checked > 0, "the sweep reaches the budget-limited regime");
    ck(bad == 0, "no combination is ever sized past its budget");
}

static void test_degenerate_inputs(void) {
    // weights alone blow the budget: no context fits, and it must say so
    // rather than divide its way to a negative or wrap around
    ck(model_autofit_tokens(1ull<<30, 8ull<<30, HEAD, 57373, 1) == 0,
       "a budget smaller than the weights fits nothing");
    // a model with no KV rows of its own must not divide by zero
    ck(model_autofit_tokens(24ull<<30, 4ull<<30, HEAD, 0, 1) == 0,
       "zero KV bytes per token is not a division by zero");
    ck(model_autofit_tokens(24ull<<30, 4ull<<30, HEAD, 57373, 0) ==
       model_autofit_tokens(24ull<<30, 4ull<<30, HEAD, 57373, 1),
       "a slot count below one is treated as one");
}

static void test_clamp(void) {
    ck(model_autofit_clamp(1000000, 8192) == 8192,
       "the trained context is a ceiling");
    ck(model_autofit_clamp(4096, 8192) == 4096,
       "a fitting context is used as-is");
    ck(model_autofit_clamp(10, 8192) == 512,
       "a tiny fit is floored at a usable window");
    ck(model_autofit_clamp(100, 256) == 512,
       "the floor wins over a very small trained context");
}

// The KV-evicts-weights trade note. Same class of blindness: it fires only on a
// partial GPU split that the KV cache caused, which needs a model bigger than
// the card. It is observable in the Linux CUDA host stress-context artifacts under
// tests/compatibility/out/ but nothing gated the threshold, so a regression
// would have shown up as the note quietly never appearing again.
static void test_kv_trade_note(void) {
    const uint64_t GB = 1000000000ull;

    // the case it exists for: 3.36 GB of KV, 7 of 32 layers displaced
    ck(model_kv_trade_note(25, 32, 3360 * (GB / 1000), 6 * GB),
       "a KV cache dominating the device earns the note");

    // a full offload traded nothing away
    ck(!model_kv_trade_note(32, 32, 3360 * (GB / 1000), 6 * GB),
       "a fully offloaded model gets no note");
    // and neither did a load that put nothing on the device
    ck(!model_kv_trade_note(0, 32, 3360 * (GB / 1000), 6 * GB),
       "a CPU-only load gets no note");

    // a split forced by the model's own size is not a trade -c can take back,
    // so a small KV against big weights must stay quiet
    ck(!model_kv_trade_note(25, 32, 1 * GB, 40 * GB),
       "a model simply too big for the card gets no note");

    // the threshold is strict: exactly a quarter is not "dominating"
    ck(!model_kv_trade_note(25, 32, 5 * GB, 20 * GB),
       "the note needs more than a quarter, not exactly a quarter");
    ck(model_kv_trade_note(25, 32, 5 * GB + 1, 20 * GB),
       "one byte past the threshold does earn it");

    // unknown sizes must not be reported as a trade
    ck(!model_kv_trade_note(25, 32, 0, 20 * GB),
       "an unmeasured KV size gets no note");
    ck(!model_kv_trade_note(25, 32, 5 * GB, 0),
       "an unmeasured weight size gets no note");
}

// Ring sizing is also admission arithmetic that real fixtures cannot safely
// reach at its upper boundary: -c/-b and GGUF windows accept INT_MAX, while an
// overflowing window + batch can turn negative before any allocation fails.
static void test_kv_ring_rows(void) {
    ck(model_kv_ring_rows(32, 64, 1024) == 96,
       "a useful ring is window plus one in-flight batch");
    ck(model_kv_ring_rows(900, 124, 1024) == 1024,
       "a ring that saves no rows stays flat");
    ck(model_kv_ring_rows(INT_MAX, INT_MAX, INT_MAX) == INT_MAX,
       "ring sizing saturates before signed addition can overflow");
}

// The device offload budget (model_gpu_budget): what the layer fit may
// spend on weights and KV. The 2026-09-14 Windows report: on a 12 GB card
// the fit filled VRAM to 11.7 GB from the driver's free-memory view and WDDM
// then paged the process over PCIe (GPU 100% busy, a 300-token request not
// finished in 15 minutes), and on the datacenter MIG slice the same fit left
// 0.02 GB of slack and the recurrent state buffers, allocated last, were
// refused by the device allocator. The budget now respects the OS video
// memory budget for the process where the OS publishes one (WDDM,
// IDXGIAdapter3::QueryVideoMemoryInfo) and keeps a headroom that scales
// with the budget instead of a flat 512 MiB.
static void test_gpu_budget(void) {
    const uint64_t G = 1ull << 30, M = 1ull << 20;
    uint64_t headroom = 0;
    // no OS budget (Linux, or a driver without the query): the driver's free
    // view, with the larger of 512 MiB and one sixteenth held back
    uint64_t b = model_gpu_budget(23 * G, 24 * G, false, 0, 0, 0, &headroom);
    ck(b == 23 * G && headroom == 23 * G / 32,
       "no OS budget: the driver's free view, headroom one thirty-second of it");
    b = model_gpu_budget(6 * G, 8 * G, false, 0, 0, 0, &headroom);
    ck(headroom == 512 * M, "a small budget keeps the 512 MiB floor");
    // the OS budget is the process's share and bounds the driver's view:
    // 12.2 GB free by the driver, but WDDM budgets this process 10.9 GB of
    // which 0.8 is already in use
    b = model_gpu_budget((uint64_t)(12.2 * G), (uint64_t)(12.2 * G), true,
                         (uint64_t)(10.9 * G), (uint64_t)(0.8 * G), 0, &headroom);
    ck(b == (uint64_t)(10.9 * G) - (uint64_t)(0.8 * G),
       "the OS budget minus what is already in use bounds the driver's view");
    // an OS budget already exhausted yields nothing to offload, never a wrap
    b = model_gpu_budget(4 * G, 8 * G, true, 2 * G, 3 * G, 0, &headroom);
    ck(b == 0, "an exhausted OS budget is zero, not a wrapped subtraction");
    // --reserve-vram caps the budget at that share of the card, after both
    b = model_gpu_budget(12 * G, 12 * G, true, 11 * G, 0, 85, &headroom);
    ck(b == 12 * G / 100 * 85, "--reserve-vram caps below the OS budget");
    b = model_gpu_budget(12 * G, 12 * G, true, 9 * G, 0, 85, &headroom);
    ck(b == 9 * G, "and the OS budget caps below --reserve-vram");
    // the headroom follows the budget the fit will actually use (a case
    // where a thirty-second clears the 512 MiB floor: 20 GB of OS budget on
    // a 24 GB card)
    b = model_gpu_budget(24 * G, 24 * G, true, 20 * G, 0, 0, &headroom);
    ck(b == 20 * G && headroom == 20 * G / 32, "headroom is taken from the effective budget");
}


// --fit on a discrete GPU (model_fit_gpu): the split, and the verdict that
// compares RAM with what the split leaves on the host. The regression is the
// 1.0.0 clean-setup run on an RTX 3070: granite-4.2-8b Q4_K_M at 16k, served
// with 35 of 40 layers on the card, was told PAGES because the verdict
// added every weight and every KV row to the RAM bill.
static void test_fit_gpu(const char *fixture) {
    gguf_file g;
    if (!fixture || !gguf_open_header(&g, fixture)) {
        ck(0, "fit gpu: the test.gguf header opens");
        return;
    }
    model_fit base;
    ck(model_fit_report(&g, 512, &base), "fit gpu: report from the header");
    int nl = base.n_layer;

    // A budget far beyond the file: every layer and the output on the device,
    // so the host keeps nothing but what was never a layer.
    model_fit f = base;
    ck(model_fit_gpu(&g, &f, 1ull << 40, 512ull << 20, 0), "fit gpu: fills");
    ck(f.gpu && f.gpu_layers[FIT_KV_F16] == nl, "fit gpu: huge budget, all layers");
    ck(f.host_need[FIT_KV_F16] < base.weights, "fit gpu: host keeps less than the file");

    // A budget below the fixed charge: nothing offloads, and the host bill is
    // exactly the RAM-only one, so the verdicts cannot disagree.
    f = base;
    model_fit_gpu(&g, &f, 1, 512ull << 20, 0);
    ck(f.gpu_layers[FIT_KV_F16] == 0, "fit gpu: tiny budget, no layers");
    ck(f.host_need[FIT_KV_F16] == base.weights + base.kv_f16,
       "fit gpu: tiny budget bills the whole file and cache to RAM");
    f.available = base.weights + base.kv_f16;
    model_fit r = base; r.available = f.available;
    ck(!strcmp(model_fit_verdict(&f), model_fit_verdict(&r)),
       "fit gpu: no offload, same verdict as RAM-only");

    // --gpu-layers caps the split like the loader's override.
    f = base;
    model_fit_gpu(&g, &f, 1ull << 40, 512ull << 20, 1);
    ck(f.gpu_layers[FIT_KV_F16] == 1, "fit gpu: max_layers caps the split");

    // The bug: RAM too small for file + cache, enough for what stays on the
    // host. RAM-only says PAGES; the GPU-aware verdict must say FITS.
    f = base;
    model_fit_gpu(&g, &f, 1ull << 40, 512ull << 20, 0);
    f.available = f.host_need[FIT_KV_F16];
    r = base; r.available = f.available;
    ck(f.available < base.hot + base.kv_fp4, "fit gpu: the case is real (RAM-only short)");
    ck(!strcmp(model_fit_verdict(&r), "PAGES"), "fit gpu: RAM-only verdict pages");
    ck(!strcmp(model_fit_verdict(&f), "FITS"), "fit gpu: offloaded verdict fits");
    // and one byte less RAM than the host needs is not a fit at f16
    f.available = f.host_need[FIT_KV_F16] - 1;
    ck(strcmp(model_fit_verdict(&f), "FITS") != 0, "fit gpu: one byte short is not FITS");

    // The device bytes never exceed the budget, for any format and budget.
    for (uint64_t b = 1; b < (64ull << 20); b = b * 3 + 7) {
        f = base;
        model_fit_gpu(&g, &f, b, 0, 0);
        for (int k = 0; k < 4; k++)
            if (f.gpu_layers[k] >= 0 && f.gpu_used[k] > b) {
                ck(0, "fit gpu: device bytes stay inside the budget");
                b = UINT64_MAX / 4;
                break;
            }
    }
    ck(1, "fit gpu: budget sweep done");

    // The host keeps the whole KV cache, offloaded layers included: a CUDA
    // load's host copy is the authoritative one (cuda.c kv_copyback), so a
    // full split still bills every KV row to RAM.
    // Both formats take every layer, so the weights left behind are the same
    // and the two host bills differ by the two whole caches (the rule this
    // replaced billed only the CPU layers' rows: zero here, for both).
    f = base;
    model_fit_gpu(&g, &f, 1ull << 40, 512ull << 20, 0);
    ck(base.kv_q8 && f.gpu_layers[FIT_KV_F16] == nl && f.gpu_layers[FIT_KV_Q8] == nl,
       "fit gpu: the fixture takes a full split in f16 and q8");
    ck(f.host_need[FIT_KV_F16] - f.host_need[FIT_KV_Q8] == base.kv_f16 - base.kv_q8,
       "fit gpu: a full split still bills the whole KV cache to RAM");

    // --parallel N: each slot pays its own KV (host and device) and device
    // scratch, the weights once; the split and both verdicts charge N.
    model_fit one = base, two = base;
    two.n_seq = 2;
    model_fit_gpu(&g, &one, 1ull << 40, 512ull << 20, 0);
    model_fit_gpu(&g, &two, 1ull << 40, 512ull << 20, 0);
    ck(two.gpu_fixed > one.gpu_fixed, "fit slots: a second slot adds its scratch to the fixed charge");
    ck(two.host_need[FIT_KV_F16] == one.host_need[FIT_KV_F16] + base.kv_f16,
       "fit slots: a second slot adds exactly one more KV cache to RAM");
    model_fit r1 = base, r2 = base;
    r2.n_seq = 2;
    r1.available = r2.available = base.hot + base.kv_f16;
    ck(!strcmp(model_fit_verdict(&r1), "FITS"), "fit slots: one slot fits at f16");
    ck(strcmp(model_fit_verdict(&r2), "FITS") != 0, "fit slots: two slots at f16 do not");

    // A row-gathered table never uploads. Removing 1024 such bytes from the
    // host estimate must save exactly 1024 bytes at every split, not a share
    // proportional to the device weights. The CLI header fixture tests the
    // E-series classification; this checks the discrete-GPU accounting seam.
    for (int cap = 0; cap <= 1; cap++) {
        uint64_t budget = cap ? 1 : 1ull << 40;
        model_fit flat = base, gathered = base;
        gathered.gathered = 1024;
        gathered.hot -= 1024;
        model_fit_gpu(&g, &flat, budget, 512ull << 20, 0);
        model_fit_gpu(&g, &gathered, budget, 512ull << 20, 0);
        ck(flat.host_need[FIT_KV_F16] - gathered.host_need[FIT_KV_F16] == 1024,
           "fit gpu: gathered bytes leave the host bill at full and zero offload");
        ck(flat.gpu_used[FIT_KV_F16] == gathered.gpu_used[FIT_KV_F16],
           "fit gpu: a host-only gathered table changes no device allocation");
    }
    gguf_close(&g);
}

// The split's boundaries, to the byte. The fit is the loader's leading-layer
// placement read off the header: a fixed charge (scratch and headroom; the
// token embedding is staged from the host), then whole layers, each its
// weights plus its KV, while they fit; a split that took every layer also
// owes the output projection, once (a tied head is the embedding table),
// else the last layer goes back. These cases rebuild each figure from the header the way the fit does
// and check the decision one byte either side of every boundary, so a change
// to the placement arithmetic (the CUDA and Metal loaders keep their own
// copies of it) shows up as a moved boundary, not a vague "fits".
static void test_fit_gpu_boundaries(const char *fixture) {
    gguf_file g;
    if (!fixture || !gguf_open_header(&g, fixture)) {
        ck(0, "fit gpu boundaries: the test.gguf header opens");
        return;
    }
    model_fit base;
    ck(model_fit_report(&g, 512, &base), "fit gpu boundaries: report");
    int nl = base.n_layer;
    uint64_t lb[64] = {0}, tok = 0, out = 0;
    ck(nl > 1 && nl <= 64, "fit gpu boundaries: a multi-layer fixture");
    if (nl <= 1 || nl > 64) { gguf_close(&g); return; }
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const gguf_tensor *t = &g.tensors[i];
        int l = -1, n = 0;
        if (sscanf(t->name, "blk.%d.%n", &l, &n) == 1 && n > 0 && l >= 0 && l < nl)
            lb[l] += t->nbytes;
        else if (!strcmp(t->name, "token_embd.weight")) tok = t->nbytes;
        else if (!strcmp(t->name, "output.weight")) out = t->nbytes;
    }
    if (!out) out = tok;
    const uint64_t extra = out;
    const uint64_t kvl = base.kv_f16_per_tok / (uint64_t)nl * (uint64_t)base.n_ctx;
    model_fit f = base;
    model_fit_gpu(&g, &f, 1ull << 40, 0, 0);
    const uint64_t fixed = f.gpu_fixed;
    uint64_t all = fixed;
    for (int l = 0; l < nl; l++) all += lb[l] + kvl;
    const int K = FIT_KV_F16;

    f = base; model_fit_gpu(&g, &f, fixed - 1, 0, 0);
    ck(f.gpu_layers[K] == 0 && f.gpu_used[K] == 0,
       "fit gpu boundaries: below the fixed charge nothing is on the device");
    f = base; model_fit_gpu(&g, &f, fixed, 0, 0);
    ck(f.gpu_layers[K] == 0 && f.gpu_used[K] == fixed,
       "fit gpu boundaries: the fixed charge alone holds no layer");
    f = base; model_fit_gpu(&g, &f, fixed + lb[0] + kvl - 1, 0, 0);
    ck(f.gpu_layers[K] == 0, "fit gpu boundaries: a byte short of the first layer");
    f = base; model_fit_gpu(&g, &f, fixed + lb[0] + kvl, 0, 0);
    ck(f.gpu_layers[K] == 1 && f.gpu_used[K] == fixed + lb[0] + kvl,
       "fit gpu boundaries: exactly the first layer");
    // every layer fits but the full split's charge does not: the last layer
    // goes back to the host
    f = base; model_fit_gpu(&g, &f, all, 0, 0);
    ck(f.gpu_layers[K] == nl - 1 && f.gpu_used[K] == all - lb[nl - 1] - kvl,
       "fit gpu boundaries: all layers without the output charge give one back");
    f = base; model_fit_gpu(&g, &f, all + extra - 1, 0, 0);
    ck(f.gpu_layers[K] == nl - 1,
       "fit gpu boundaries: a byte short of the full split (output charged once)");
    f = base; model_fit_gpu(&g, &f, all + extra, 0, 0);
    ck(f.gpu_layers[K] == nl && f.gpu_used[K] == all + extra,
       "fit gpu boundaries: exactly the full split, output included");
    // --gpu-layers below the natural split caps it; at or above, changes nothing
    f = base; model_fit_gpu(&g, &f, all + extra, 0, nl - 1);
    ck(f.gpu_layers[K] == nl - 1, "fit gpu boundaries: max_layers caps a full split");
    f = base; model_fit_gpu(&g, &f, fixed + lb[0] + kvl, 0, nl);
    ck(f.gpu_layers[K] == 1, "fit gpu boundaries: a cap above the split changes nothing");
    // the headroom is part of the fixed charge, byte for byte
    f = base; model_fit_gpu(&g, &f, 1ull << 40, 4096, 0);
    ck(f.gpu_fixed == fixed + 4096, "fit gpu boundaries: headroom joins the fixed charge");
    // two slots: the first layer costs its weights plus two KV shares, on a
    // fixed charge that holds two scratch sets
    model_fit two = base; two.n_seq = 2;
    model_fit_gpu(&g, &two, 1ull << 40, 0, 0);
    const uint64_t fixed2 = two.gpu_fixed;
    two = base; two.n_seq = 2; model_fit_gpu(&g, &two, fixed2 + lb[0] + 2 * kvl - 1, 0, 0);
    ck(two.gpu_layers[K] == 0, "fit gpu boundaries: two slots, a byte short of the first layer");
    two = base; two.n_seq = 2; model_fit_gpu(&g, &two, fixed2 + lb[0] + 2 * kvl, 0, 0);
    ck(two.gpu_layers[K] == 1 && two.gpu_used[K] == fixed2 + lb[0] + 2 * kvl,
       "fit gpu boundaries: two slots, exactly the first layer");
    gguf_close(&g);
}

// The split's decisions as the backends now share them (gpu_plan_*), on
// numbers worked by hand: every boundary one unit either side.
static void test_gpu_plan(void) {
    const uint64_t cost[4] = { 10, 20, 30, 40 };
    uint64_t used = 5;
    ck(gpu_plan_prefix(cost, 4, 0, 35, &used) == 2 && used == 35,
       "plan prefix: exactly two layers (5 + 10 + 20 = 35)");
    used = 5;
    ck(gpu_plan_prefix(cost, 4, 0, 34, &used) == 1 && used == 15,
       "plan prefix: one unit short of the second layer");
    used = 5;
    ck(gpu_plan_prefix(cost, 4, 0, 105, &used) == 4 && used == 105,
       "plan prefix: every layer, exactly");
    used = 5;
    ck(gpu_plan_prefix(cost, 4, 2, 1000, &used) == 2 && used == 35,
       "plan prefix: the cap stops a split that would fit");
    used = 5;
    ck(gpu_plan_prefix(cost, 4, 9, 1000, &used) == 4,
       "plan prefix: a cap above the layer count changes nothing");
    used = 5;
    ck(gpu_plan_prefix(cost, 4, 0, 14, &used) == 0 && used == 5,
       "plan prefix: the first layer short leaves the fixed charge alone");
    used = 0;
    ck(gpu_plan_prefix(cost, 4, 0, 0, &used) == 0, "plan prefix: a zero budget takes nothing");
    used = 6;
    ck(gpu_plan_prefix(cost, 4, 0, 5, &used) == 0 && used == 6,
       "plan prefix: a fixed charge over the budget takes nothing");
    const uint64_t first_small[3] = { 10, 50, 1 };
    used = 0;
    ck(gpu_plan_prefix(first_small, 3, 0, 20, &used) == 1,
       "plan prefix: the first layer that does not fit ends the run");
    const uint64_t huge[2] = { UINT64_MAX - 10, 100 };
    used = 20;
    ck(gpu_plan_prefix(huge, 2, 0, UINT64_MAX, &used) == 0 && used == 20,
       "plan prefix: a sum past 64 bits refuses rather than wraps");
    used = 0;
    ck(gpu_plan_prefix(huge, 2, 0, UINT64_MAX, &used) == 1 && used == UINT64_MAX - 10,
       "plan prefix: up to the last representable byte");

    bool back = true;
    int G = 4;
    ck(gpu_plan_full_split(&G, 4, 90, 10, 100, &back) && G == 4 && !back,
       "plan full: the tail fits exactly");
    G = 4;
    ck(!gpu_plan_full_split(&G, 4, 90, 11, 100, &back) && G == 3 && back,
       "plan full: one unit short hands the last layer back");
    G = 3;
    ck(!gpu_plan_full_split(&G, 4, 10, 1, 100, &back) && G == 3 && !back,
       "plan full: a partial split is left as it is");
    G = 1;
    ck(!gpu_plan_full_split(&G, 1, 99, 2, 100, &back) && G == 1 && !back,
       "plan full: a one-layer model keeps its layer");
    G = 2;
    ck(!gpu_plan_full_split(&G, 2, 101, 0, 100, &back) && G == 1 && back,
       "plan full: an over-budget total is not full, never a wrapped subtraction");

    // banks: layers 0 and 2 dense, 1, 3 and 4 routed
    const bool moe[5] = { false, true, false, true, true };
    const uint64_t bank[5] = { 0, 30, 0, 30, 30 };
    bool host[5] = { false, true, false, true, true };
    used = 10;
    int placed = gpu_plan_expert_banks(moe, bank, 5, 5, 75, &used, host, 0, 0);
    ck(placed == 2 && used == 70 && !host[1] && !host[3] && host[4],
       "plan banks: shallowest first while each fits with the reserve held back");
    bool host2[5] = { false, true, false, true, true };
    used = 10;
    ck(gpu_plan_expert_banks(moe, bank, 5, 6, 75, &used, host2, 0, 0) == 1 &&
       used == 40 && !host2[1] && host2[3],
       "plan banks: one unit of reserve more and the second bank stays home");
    bool host3[5] = { false, true, false, true, true };
    used = 10;
    ck(gpu_plan_expert_banks(moe, bank, 3, 0, 1000, &used, host3, 0, 0) == 1 &&
       host3[3] && host3[4],
       "plan banks: only the device layers' banks are candidates");
    bool host4[5] = { false, true, false, true, true };
    used = 10;
    placed = gpu_plan_expert_banks(moe, bank, 5, 0, 1000, &used, host4, 1, 3);
    ck(placed == 2 && used == 70 && !host4[1] && !host4[3] && host4[4],
       "plan banks: a retry hands back the deepest bank of the first attempt's three");
    bool host5[5] = { false, true, false, true, true };
    used = 10;
    placed = gpu_plan_expert_banks(moe, bank, 5, 0, 1000, &used, host5, 4, 3);
    ck(placed == 0 && used == 10 && host5[1] && host5[3] && host5[4],
       "plan banks: a trim past the first attempt's count hands back every bank");

    // the device's own rules: allocation rounding of a bound tensor, and the
    // RAM cap of an integrated device
    const uint64_t MiB = 1ull << 20;
    ck(gpu_plan_binding_bytes(0, true) == 0, "plan binding: an empty tensor costs nothing");
    ck(gpu_plan_binding_bytes(1, true) == 2 * MiB, "plan binding: one byte bound is one granule");
    ck(gpu_plan_binding_bytes(2 * MiB, true) == 2 * MiB, "plan binding: a whole granule stays one");
    ck(gpu_plan_binding_bytes(2 * MiB + 1, true) == 4 * MiB, "plan binding: one byte over is two");
    ck(gpu_plan_binding_bytes(2 * MiB + 1, false) == 2 * MiB + 1,
       "plan binding: inside one upload a tensor costs its own bytes");
    ck(gpu_plan_unified_cap(24ull << 30, 0) == 24ull << 30,
       "plan unified: an unknown RAM figure leaves the budget");
    ck(gpu_plan_unified_cap(24ull << 30, 10ull << 30) == 10ull << 30,
       "plan unified: available RAM below the driver's view caps it");
    ck(gpu_plan_unified_cap(8ull << 30, 10ull << 30) == 8ull << 30,
       "plan unified: more RAM than the driver's view changes nothing");

    // the device tensors' stretches: a gap up to the granularity joins
    // the stretch around it, one byte more starts a new one; overlapping or
    // touching tensors are one stretch; the count without the arrays
    {
        const uint64_t off[5] = { 100, 200, 300 + GPU_PLAN_GRAN, 400 + 2 * GPU_PLAN_GRAN + 1, 450 + 2 * GPU_PLAN_GRAN };
        const uint64_t len[5] = { 100, 100, 100, 50, 10 };
        uint64_t lo[5], hi[5];
        int k = gpu_plan_runs(off, len, 5, GPU_PLAN_GRAN, lo, hi);
        ck(k == 2 && lo[0] == 100 && hi[0] == 400 + GPU_PLAN_GRAN,
           "plan runs: touching tensors and a gap of exactly the granularity join");
        ck(k == 2 && lo[1] == 400 + 2 * GPU_PLAN_GRAN + 1 && hi[1] == 460 + 2 * GPU_PLAN_GRAN,
           "plan runs: one byte past the granularity starts a stretch, a later tensor joins it");
        ck(gpu_plan_runs(off, len, 5, GPU_PLAN_GRAN, NULL, NULL) == 2,
           "plan runs: counting without the arrays");
        ck(gpu_plan_runs(off, len, 0, GPU_PLAN_GRAN, NULL, NULL) == 0,
           "plan runs: no tensors, no stretch");
        const uint64_t ov[2] = { 0, 50 }, ol[2] = { 100, 10 };
        ck(gpu_plan_runs(ov, ol, 2, 0, lo, hi) == 1 && hi[0] == 100,
           "plan runs: a tensor inside another adds nothing");
    }
}

// The CUDA prompt window (C1 step 2): the scratch charge is the loader's
// own, split per column, and a wider window spends only what the split left
// spare. Anchor: the per-column and fixed bytes below are hand computed from
// Qwen2.5-7B's widths (n_embd = q_dim = 3584, kv_dim 512, n_ff 18944, 28
// heads, vocab 152064) at a 4096 context, and at 64 columns they must equal
// the formula the plan used before the window existed.
static void test_gpu_window(void) {
    gpu_act_cost c = model_gpu_act_cost(3584, 3584, 3584, 512, 18944, 28, 4096, 152064);
    ck(c.per_col == 227328ull, "window: per-column scratch, hand computed (4 * 56832)");
    ck(c.fixed == 29968384ull, "window: fixed scratch, hand computed (4 * (64*28*4096 + 152064))");
    ck(c.fixed + 64 * c.per_col == 4ull * (64ull * (3584 + 3 * 3584 + 3584 + 2 * 512 +
                                                    2 * 18944 + 28ull * 4096) + 152064),
       "window: at 64 columns the charge is the pre-window formula");
    uint64_t pc = c.per_col;
    ck(model_gpu_window((512 - 64) * pc, pc, 1, 512, 0) == 512,
       "window: spare exactly covering 512 columns gives 512");
    ck(model_gpu_window((512 - 64) * pc - 1, pc, 1, 512, 0) == 448,
       "window: one byte short gives the next multiple of 64 down");
    ck(model_gpu_window((512 - 64) * pc, pc, 2, 512, 0) == 256,
       "window: two slots each pay their own columns");
    ck(model_gpu_window(0, pc, 1, 512, 0) == 64, "window: no spare budget stays at 64");
    ck(model_gpu_window(UINT64_MAX / 2, pc, 1, 100, 0) == 64,
       "window: -b 100 caps it below the next multiple of 64 above 64");
    ck(model_gpu_window(UINT64_MAX / 2, pc, 1, 200, 0) == 192, "window: -b 200 gives 192");
    ck(model_gpu_window(UINT64_MAX / 2, pc, 1, 32, 0) == 64,
       "window: a batch below 64 still allocates the minimum");
    ck(model_gpu_window(UINT64_MAX / 2, pc, 1, 4096, 0) == 512, "window: never wider than 512");
    ck(model_gpu_window(UINT64_MAX / 2, pc, 1, 512, 128) == 128,
       "window: RUNNER_CUDA_WINDOW narrows it");
    ck(model_gpu_window(0, pc, 1, 512, 512) == 64,
       "window: a forced width never spends budget the split did not leave");
}

int main(int argc, char **argv) {
    test_gpu_window();
    test_gpu_budget();
    test_multislot_is_not_billed_once();
    test_budget_is_never_exceeded();
    test_degenerate_inputs();
    test_clamp();
    test_kv_trade_note();
    test_kv_ring_rows();
    test_fit_gpu(argc > 1 ? argv[1] : NULL);
    test_fit_gpu_boundaries(argc > 1 ? argv[1] : NULL);
    test_gpu_plan();
    if (g_fail) { fprintf(stderr, "test-autofit FAILED\n"); return 1; }
    fprintf(stderr, "test-autofit: all checks passed\n");
    return 0;
}
