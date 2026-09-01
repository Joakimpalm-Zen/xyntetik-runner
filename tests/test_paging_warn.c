// The residency warning's message selection.
//
// Issue #7: the warning told a 16 GB Mac that "every token will page from
// disk" while gemma-4-26B-A4B was in fact serving 8+ tok/s on that machine.
// The claim was not wrong about the file not fitting — it was wrong about what
// that costs, because only 8 of 128 experts per layer are ever touched. A
// diagnostic that overstates by 16x gets ignored, which is the same as not
// having one.
//
// Two things are checked here, because a warning has two halves that fail
// independently:
//
//   1. model_hot_set_bytes on real fixtures — a dense model must discount
//      nothing, and a top-2-of-4 MoE must discount half its expert banks.
//      Getting this from a fixture rather than a hand-built struct is the
//      point: the byte accounting has to agree with the loader's actual
//      tensor set, including the router and shared-expert exclusions.
//   2. The wording chosen for each (need, hot, have) case. Available RAM
//      cannot be forced on a build machine, so the selection is tested
//      directly rather than through model_load.
#include <stdio.h>
#include <string.h>
#include "runner.h"

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        printf("ok: %s\n", what);
}

static bool load(model_t *m, const char *path) {
    model_params p = {0};
    p.n_ctx = 64;
    p.gpu_mode = GPU_OFF;
    if (model_load(m, path, &p)) return true;
    fprintf(stderr, "paging-warn: cannot load %s\n", path);
    return false;
}

int main(int argc, char **argv) {
    const char *prefix = argc > 1 ? argv[1] : "test-moe-fixture";
    char dense_path[512], moe_path[512];
    snprintf(dense_path, sizeof(dense_path), "%s.dense.gguf", prefix);
    snprintf(moe_path,   sizeof(moe_path),   "%s.moe4.gguf",  prefix);

    // --- the hot-set estimate, on fixtures -------------------------------
    model_t dm;
    if (!load(&dm, dense_path)) return 1;
    ck(model_hot_set_bytes(&dm) == 0,
       "a dense model discounts nothing (0 == 'the file is the hot set')");
    uint64_t dense_size = dm.gf.map_size;
    ck(dense_size > 0, "the dense fixture has a size");
    model_free(&dm);

    model_t mm;
    if (!load(&mm, moe_path)) return 1;
    uint64_t total = mm.gf.map_size, hot = model_hot_set_bytes(&mm);
    printf("note: moe4 file %llu bytes, hot set %llu bytes (%d of %d experts)\n",
           (unsigned long long)total, (unsigned long long)hot,
           mm.n_expert_used, mm.n_expert);
    ck(mm.n_expert == 4 && mm.n_expert_used == 2,
       "the moe4 fixture is top-2-of-4 as this test assumes");
    ck(hot > 0 && hot < total,
       "a sparse MoE's hot set is smaller than its file");
    // top-2-of-4 halves the expert banks and nothing else, so the discount is
    // exactly half of whatever the banks weigh. Anything that also discounted
    // the router, the shared expert, or the non-MoE layers would land lower.
    uint64_t discount = total - hot;
    ck(discount * 2 <= total,
       "the discount cannot exceed the expert banks it comes from");
    model_free(&mm);

    // --- message selection ----------------------------------------------
    // 10 GB of weights, 4 GB of RAM. The hot set is what changes the verdict.
    const uint64_t need = 10000000000ull, have = 4000000000ull;
    char buf[512];

    ck(!model_residency_warning(need, 0, need + 1, false, buf, sizeof(buf)),
       "a model that fits produces no warning at all");
    ck(!model_residency_warning(need, 1, 0, false, buf, sizeof(buf)),
       "an unknown RAM figure produces no warning rather than a guess");

    ck(model_residency_warning(need, 0, have, false, buf, sizeof(buf)) &&
       strstr(buf, "every token to page from disk") &&
       !strstr(buf, "sparse MoE"),
       "dense: the original every-token wording is kept");

    ck(model_residency_warning(need, 2000000000ull, have, false,
                               buf, sizeof(buf)) &&
       strstr(buf, "sparse MoE") && strstr(buf, "does fit") &&
       !strstr(buf, "every token to page") && !strstr(buf, "--mlock"),
       "sparse with a hot set that fits: no every-token claim, no mlock hint");

    ck(model_residency_warning(need, 8000000000ull, have, false,
                               buf, sizeof(buf)) &&
       strstr(buf, "sparse MoE") && strstr(buf, "every token to page from disk"),
       "sparse with a hot set that does not fit: the doom claim survives");

    // A hot set that is not actually a discount must not be dressed up as one.
    ck(model_residency_warning(need, need, have, false, buf, sizeof(buf)) &&
       !strstr(buf, "sparse MoE"),
       "hot == file falls back to the dense wording");

    // Found on the M5 Max wiring 85.7 GB: mlock succeeded over the whole map
    // and the warning still predicted evictions and cold-expert disk reads —
    // both impossible for wired pages. A successful lock is also proof the
    // memory existed, whatever the instantaneous "available" figure said.
    ck(!model_residency_warning(need, 0, have, true, buf, sizeof(buf)),
       "wired weights cannot be evicted — a locked model gets no warning");
    ck(model_residency_warning(need, 0, have, false, buf, sizeof(buf)) &&
       strstr(buf, "--mlock"),
       "an unlocked model is told about --mlock");

    // --- per-request paging note ------------------------------------------
    // The serve log tags a request "[N page-ins — weights not resident]" from
    // the fault delta. On Windows that counter includes soft faults, and a
    // resident model at full speed showed 297 of them per ~40-token request
    // (2026-09-01, RTX 3070 box) — noise dressed as a disk. Evicted weights
    // fault in by the gigabyte per token (the 8 GB M1 showed 1.8M for a
    // 96-token request). Hand-computed floor: 64 pages per token.
    ck(!model_paging_note_wanted(0, 10), "no faults, no note");
    ck(!model_paging_note_wanted(297, 40), "a few hundred soft faults on a resident model: silent");
    ck(model_paging_note_wanted(1805121, 96), "millions of faults on a paging model: the note fires");
    ck(model_paging_note_wanted(64 * 5, 5), "exactly the floor fires");
    ck(!model_paging_note_wanted(64 * 5 - 1, 5), "one below the floor is silent");
    ck(model_paging_note_wanted(64, 0), "a zero-token request is floored at one token");

    // --- load-time prefetch decision -------------------------------------
    // Cold-start page-in of a big model arrives as ~16 KB synchronous faults
    // (1.1M+ of them for the 120B); a WILLNEED sweep batches them. The
    // decision is pure arithmetic so it is gated here, hand-computed.
    ck(model_load_prefetch_wanted(10, 100, false, false),
       "a model that fits gets the prefetch hint");
    ck(!model_load_prefetch_wanted(100, 10, false, false),
       "an oversubscribed model is never swept — that is thrash, not warmth");
    ck(!model_load_prefetch_wanted(10, 100, true, false),
       "mlock already forces residency; a hint on top is noise");
    ck(!model_load_prefetch_wanted(10, 100, false, true),
       "the expert-prefetch path owns paging on oversubscribed MoE");
    ck(!model_load_prefetch_wanted(10, 0, false, false),
       "unknown available RAM means no guess");

    // --- prompt-batch default --------------------------------------------
    // The old default was sized from FREE RAM at launch, which made the
    // sampled tokens of a reassociating prefill depend on what else the
    // machine happened to be doing that day — an ambient input hiding inside
    // "same executable and inputs". The default is now a pure function of
    // TOTAL RAM, a fixed machine fact. Hand-computed cases:
    ck(model_batch_default_for(0) == 512,
       "unmeasurable total RAM assumes the modern default");
    ck(model_batch_default_for((uint64_t)4 << 30) == 64,
       "a 4 GB machine keeps the flat batch");
    ck(model_batch_default_for((uint64_t)8 << 30) == 256,
       "an 8 GB machine takes half the win, always the same half");
    ck(model_batch_default_for((uint64_t)16 << 30) == 512,
       "a 16 GB machine gets the full batch");
    ck(model_batch_default_for(((uint64_t)6 << 30) - 1) == 64 &&
       model_batch_default_for((uint64_t)6 << 30) == 256 &&
       model_batch_default_for(((uint64_t)12 << 30) - 1) == 256 &&
       model_batch_default_for((uint64_t)12 << 30) == 512,
       "the 6 GB and 12 GB boundaries land exactly where documented");


    if (g_fail) { fprintf(stderr, "paging-warn: FAILED\n"); return 1; }
    puts("paging-warn ok");
    return 0;
}
