// R3.60: on a CPU-only run a prompt's bits do not depend on the batch size.
//
// A prompt is fed in chunks of -b rows. A projection narrower than
// MV_SMALL_BATCH (8) columns used to take the native quantized dot and a wider
// one the dequantized f32 tile, two arithmetics that agree only to the last
// bits. So the rows a chunk's short tail left in the KV cache, and the rows of
// a MoE expert that a chunk gave fewer than 8 rows, moved with -b, with where
// the chunks fell, and with how much of a prompt a cache had kept. Now every
// product of a prompt's feed takes the tile at every width (model.c mv_rows),
// and only the decode step keeps the native dot.
//
// The gates, byte for byte, on an f16 cache:
//
//   1. -b 64 and -b 512 (the context, 256, at these fixtures) store the same
//      K and V rows at every position of every layer, and give the same
//      logits after the prompt and after 4 greedy steps. Prompts of 129, 67,
//      200 and 250 tokens leave tails of 1, 3, 8 and 58 rows at -b 64 and are
//      one chunk at 512. Fixtures:
//        - dense Q4_K and Q6_K, 256-wide rows;
//        - a MoE of 32 experts, 2 used, Q8_0: a 64-row chunk gives an expert
//          about 4 rows, a whole prompt about 8 to 16, both sides of the
//          switch (the 2- and 4-expert fixtures never get under 8);
//        - gpt-oss with MXFP4 experts and sliding layers: at -b 64 a sliding
//          layer keeps a ring of window + 64 rows, at 256 the flat cache, so
//          the positions both still hold are compared there.
//   2. A prompt fed as n - 1 tokens and then 1 (the prefix cache's re-feed of
//      a repeated prompt) stores what the cold feed stores.
//   3. The live control: the same prompt fed one token at a time through the
//      decode step's route (engine_feed_solo) stores different rows on every
//      fixture. So the fixture tells the two arithmetics apart, and a route
//      chosen by width would show in gate 1.
//   4. -b 1 reads the prompt one token per forward on the decode route, the
//      rows gate 3 stores, byte for byte: the trainer's taped forward, which
//      the GRPO loop's -b 1 server is held to (tests/test_rollout_identity.py,
//      tests/test_kv_f32.py). From -b 2 up the identity of gate 1 holds: -b 2
//      stores -b 64's rows.
//   5. A scored prompt (engine_score_prompt: echo, prompt_logprobs) writes its
//      rows on the decode route. A normal request that extends it, through
//      engine_prefix_reuse as every server request starts, keeps none of
//      those rows and stores the cold feed's. Before R3.60's fix it kept them
//      all, and the extension read a decode-route prefix under a prefill-pass
//      suffix. Live control: the same reuse through the plain engine_rewind
//      keeps the scored rows, and those differ from the cold feed's.
//   6. The shared tier: a scored prompt published (engine_prefix_publish, as a
//      request does after its prefill) and then extended by another slot
//      stores the cold feed's rows. Before the fix it published the scored
//      rows and the other slot forked them. Live control: a prompt fed as a
//      prompt, published, is forked by the other slot.
//   7. The mark is a position: a prefix fed as a prompt, then scored tokens
//      alone from position m (engine_mark_scored(m), as src/decide.c marks
//      where its scored tokens start). A normal request extending it keeps the
//      m rows below the mark, no more, and stores the cold feed's rows.
//   8. The persisted prefix store keys entries by arithmetic class: a -b 1
//      store loads into no -b 64 engine; control: -b 64's loads into -b 2.
//   9. A state saved and loaded is of unknown provenance: its rows are not
//      published and not reused by a later request.
//  10. A context imported from disk is forked only by a request naming it.
//  11. ... and the prefix store does not persist it: its label would be lost.
//  12. The route identity is the arithmetic, not the batch width alone: a
//      switch that selects other prompt arithmetic (RUNNER_CPU_I8_PREFILL)
//      is another route for the store and for a slot's own rows.
#include "runner.h"

// MinGW has no setenv/unsetenv; an empty value removes the variable there
#ifdef _WIN32
#define env_set(k, v) _putenv_s(k, v)
#define env_unset(k)  _putenv_s(k, "")
#else
#define env_set(k, v) setenv(k, v, 1)
#define env_unset(k)  unsetenv(k)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CTX = 256, STEPS = 4, NLEN = 4 };
static const int LENS[NLEN] = { 129, 67, 200, 250 };
static int g_fail = 0;

static void ck(int cond, const char *file, int n, const char *what) {
    fprintf(stderr, "%s: %s at %d tokens, %s\n", cond ? "ok" : "FAIL", file, n, what);
    if (!cond) g_fail = 1;
}

typedef struct { model_t m; tokenizer tok; sampler smp; engine e; } slot;

static bool slot_open(slot *s, const char *path, int batch) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;
    p.n_threads = 2;
    p.n_ctx     = CTX;
    p.n_batch   = batch;
    memset(s, 0, sizeof(*s));
    if (!model_load(&s->m, path, &p)) return false;
    if (!tokenizer_init(&s->tok, &s->m.gf)) { model_free(&s->m); return false; }
    s->smp.temp = 0;
    s->smp.repeat_penalty = 1.0f;
    s->smp.rng = 1;
    engine_init(&s->e, &s->m, &s->tok, &s->smp);
    return true;
}

static void slot_close(slot *s) {
    free(s->e.hist);
    tokenizer_free(&s->tok);
    model_free(&s->m);
}

static void prompt_tokens(const model_t *m, int32_t *t, int n) {
    uint32_t x = 0x2545F491u;
    for (int i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        t[i] = 3 + (int32_t)(x % (uint32_t)(m->n_vocab - 3));
    }
}

static const uint8_t *k_row(const model_t *m, int l, int p) {
    size_t kn = model_kv_dim(m, l) > 0 && !model_layer_tied_v(m, l) ? model_k_row_bytes(m, l) : 0;
    return kn ? (const uint8_t *)m->kcache + model_k_byte_off(m, l) +
                (size_t)model_kv_row_at(m, l, p) * kn : NULL;
}
static const uint8_t *v_row(const model_t *m, int l, int p) {
    size_t vn = model_kv_dim(m, l) > 0 ? model_v_row_bytes(m, l) : 0;
    return vn ? (const uint8_t *)m->vcache + model_v_byte_off(m, l) +
                (size_t)model_kv_row_at(m, l, p) * vn : NULL;
}

// The rows of positions [0, n) that BOTH caches still hold (a ring layer
// keeps its last kv_ring positions), compared byte for byte. Returns the
// number of (layer, position) pairs whose K or V row differs.
static int kv_diff(const model_t *a, const model_t *b, int n) {
    int diffs = 0;
    for (int l = 0; l < a->n_layer; l++) {
        int lo = 0;
        if (model_kv_is_ring(a, l) && n - a->kv_ring > lo) lo = n - a->kv_ring;
        if (model_kv_is_ring(b, l) && n - b->kv_ring > lo) lo = n - b->kv_ring;
        for (int p = lo; p < n; p++) {
            const uint8_t *ka = k_row(a, l, p), *kb = k_row(b, l, p);
            const uint8_t *va = v_row(a, l, p), *vb = v_row(b, l, p);
            bool same = true;
            if (ka && kb) same = memcmp(ka, kb, model_k_row_bytes(a, l)) == 0;
            if (same && va && vb) same = memcmp(va, vb, model_v_row_bytes(a, l)) == 0;
            diffs += !same;
        }
    }
    return diffs;
}

static int argmax(const float *lg, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (lg[i] > lg[best]) best = i;
    return best;
}

// Feed the prompt whole (engine_feed, the prefill pass), keep its logits,
// then decode STEPS greedy tokens through the decode route, keeping each
// step's logits: (STEPS + 1) rows of n_vocab.
static float *feed_and_decode(slot *s, const int32_t *t, int n) {
    int V = s->m.n_vocab;
    float *out = malloc(sizeof(float) * (size_t)V * (STEPS + 1));
    if (!out) return NULL;
    engine_reset(&s->e);
    const float *lg = engine_feed(&s->e, t, n);
    for (int k = 0; lg && k <= STEPS; k++) {
        memcpy(out + (size_t)k * V, lg, sizeof(float) * (size_t)V);
        if (k < STEPS) lg = engine_feed_solo(&s->e, (int32_t)argmax(lg, V));
    }
    if (!lg) { free(out); return NULL; }
    return out;
}

static void check_file(const char *path) {
    slot s64, s512, ref, s1, s2;
    int control5 = 0, control6 = 0;
    if (!slot_open(&s64, path, 64) || !slot_open(&s512, path, 512) ||
        !slot_open(&ref, path, 64) || !slot_open(&s1, path, 1) || !slot_open(&s2, path, 2)) {
        fprintf(stderr, "FAIL: %s did not load\n", path);
        g_fail = 1;
        return;
    }
    int V = s64.m.n_vocab;
    int32_t t[CTX];
    prompt_tokens(&s64.m, t, CTX);
    int control_live = 0;
    for (int i = 0; i < NLEN; i++) {
        int n = LENS[i];
        // gate 1: the prompt's rows at -b 64 and -b 512, then the decode on top.
        // The rows are read before the decode moves the ring.
        engine_reset(&s64.e);
        engine_reset(&s512.e);
        ck(engine_feed(&s64.e, t, n) && engine_feed(&s512.e, t, n), path, n, "both batches fed");
        int d = kv_diff(&s64.m, &s512.m, n);
        if (d) fprintf(stderr, "  %d (layer, position) rows differ\n", d);
        ck(d == 0, path, n, "-b 64 and -b 512 store the same K and V rows");
        float *a = feed_and_decode(&s64, t, n), *b = feed_and_decode(&s512, t, n);
        ck(a && b && memcmp(a, b, sizeof(float) * (size_t)V * (STEPS + 1)) == 0, path, n,
           "-b 64 and -b 512 give the same logits after the prompt and 4 greedy steps");
        free(a);
        free(b);

        // gate 2: n - 1 tokens, then the last one alone, as the prefix cache
        // re-feeds a repeated prompt
        engine_reset(&ref.e);
        bool kept = engine_feed(&ref.e, t, n - 1) && engine_feed(&ref.e, t + n - 1, 1);
        engine_reset(&s64.e);
        kept = kept && engine_feed(&s64.e, t, n);
        ck(kept && kv_diff(&ref.m, &s64.m, n) == 0, path, n,
           "a kept prefix and a one-token re-feed store the cold feed's rows");

        // gate 3, the control: one token at a time through the decode route
        engine_reset(&ref.e);
        bool solo = true;
        for (int j = 0; j < n && solo; j++) solo = engine_feed_solo(&ref.e, t[j]) != NULL;
        int dc = solo ? kv_diff(&ref.m, &s64.m, n) : 0;
        fprintf(stderr, "  control: the decode route differs at %d (layer, position) rows\n", dc);
        control_live += dc > 0;

        // gate 4: -b 1 is the decode route's rows (ref holds them), -b 2 is -b 64's
        engine_reset(&s1.e);
        bool one = solo && engine_feed(&s1.e, t, n) != NULL;
        int d1 = one ? kv_diff(&s1.m, &ref.m, n) : -1;
        if (d1) fprintf(stderr, "  -b 1: %d (layer, position) rows differ from the decode route\n", d1);
        ck(d1 == 0, path, n, "-b 1 stores the decode route's K and V rows");
        engine_reset(&s2.e);
        engine_reset(&s64.e);
        bool two = engine_feed(&s2.e, t, n) && engine_feed(&s64.e, t, n);
        ck(two && kv_diff(&s2.m, &s64.m, n) == 0, path, n, "-b 2 and -b 64 store the same K and V rows");

        // gate 5: a scored prompt, then a normal request extending it
        int n2 = n + 6 <= CTX ? n + 6 : CTX;
        prompt_scores ps;
        bool scored = engine_score_prompt(&ref.e, t, n, 0, &ps) != NULL;
        prompt_scores_free(&ps);
        prefix_reuse r5 = scored ? engine_prefix_reuse(&ref.e, t, n2) : (prefix_reuse){ -1, 0, 0.0 };
        bool fed5 = scored && engine_feed(&ref.e, t + r5.keep, n2 - r5.keep) != NULL;
        engine_reset(&s64.e);
        fed5 = fed5 && engine_feed(&s64.e, t, n2) != NULL;
        int d5 = fed5 ? kv_diff(&ref.m, &s64.m, n2) : -1;
        if (d5) fprintf(stderr, "  after a scored prompt: kept %d rows, %d (layer, position) rows differ from cold\n",
                        r5.keep, d5);
        ck(d5 == 0, path, n, "a normal prompt after a scored one stores the cold feed's rows");
        // the control: engine_rewind keeps the scored rows, which are not the cold feed's
        scored = engine_score_prompt(&ref.e, t, n, 0, &ps) != NULL;
        prompt_scores_free(&ps);
        int kept_c = scored ? engine_rewind(&ref.e, t, n2) : -1;
        bool fed_c = scored && kept_c > 0 && engine_feed(&ref.e, t + kept_c, n2 - kept_c) != NULL;
        control5 += fed_c && kv_diff(&ref.m, &s64.m, n2) > 0;

        // gate 6: the shared tier, a scored prompt published, then another slot
        scored = engine_score_prompt(&ref.e, t, n, 0, &ps) != NULL;
        prompt_scores_free(&ps);
        if (scored) engine_prefix_publish(&ref.e, t, n, n, 1.0);
        engine_reset(&s2.e);
        prefix_reuse r6 = engine_prefix_reuse(&s2.e, t, n2);
        bool fed6 = scored && engine_feed(&s2.e, t + r6.keep, n2 - r6.keep) != NULL;
        int d6 = fed6 ? kv_diff(&s2.m, &s64.m, n2) : -1;
        if (d6) fprintf(stderr, "  another slot after a scored publish: kept %d rows, %d (layer, position) rows differ\n",
                        r6.keep, d6);
        ck(d6 == 0, path, n, "another slot extending a published scored prompt stores the cold feed's rows");
        // the control: a prompt fed as a prompt is published, and the other slot forks it
        engine_reset(&ref.e);
        bool fedp = engine_feed(&ref.e, t, n) != NULL;
        if (fedp) engine_prefix_publish(&ref.e, t, n, n, 1.0);
        engine_reset(&s2.e);
        prefix_reuse rc = engine_prefix_reuse(&s2.e, t, n2);
        control6 += fedp && rc.keep >= n - 1;

        // gate 7: the mark is a position (a decide's state is kept, its scored tokens are not)
        int m7 = n / 2;
        engine_reset(&ref.e);
        bool ok7 = engine_feed(&ref.e, t, m7) != NULL;
        engine_mark_scored(&ref.e, m7);
        for (int j = m7; j < n && ok7; j++) ok7 = engine_feed_solo(&ref.e, t[j]) != NULL;
        int kept7 = ok7 ? engine_rewind_prompt(&ref.e, t, n2) : -1;
        ok7 = ok7 && engine_feed(&ref.e, t + kept7, n2 - kept7) != NULL;
        int d7 = ok7 ? kv_diff(&ref.m, &s64.m, n2) : -1;
        if (d7 || kept7 != m7) fprintf(stderr, "  after scored tokens from %d: kept %d rows, %d (layer, position) rows differ\n",
                                       m7, kept7, d7);
        ck(kept7 == m7 && d7 == 0, path, n,
           "after scored tokens from m, a normal prompt keeps exactly the m rows below them and stores the cold feed's");
    }
    // gate 8: the persisted prefix store keys an entry by its arithmetic class.
    // -b 1 writes a prompt's rows on the decode route, -b 2 and up as a
    // prefill pass (gates 3 and 4): a -b 1 store loads into no -b 64 engine.
    // Live control: a -b 64 store loads into a -b 2 engine, the same class.
    {
        int n8 = LENS[0];
        const char *store = "test-prefill-gate8.store";
        prefix_cache_clear();
        engine_reset(&s1.e);
        if (engine_feed(&s1.e, t, n8)) engine_prefix_publish(&s1.e, t, n8, n8, 0.0);
        int saved8 = prefix_cache_save(store);
        prefix_cache_clear();
        int solo_into_tile = saved8 > 0 ? prefix_cache_load(store, &s64.e) : -2;
        prefix_cache_clear();
        engine_reset(&s64.e);
        if (engine_feed(&s64.e, t, n8)) engine_prefix_publish(&s64.e, t, n8, n8, 0.0);
        int saved8b = prefix_cache_save(store);
        prefix_cache_clear();
        int tile_into_tile = saved8b > 0 ? prefix_cache_load(store, &s2.e) : -2;
        prefix_cache_clear();
        remove(store);
        if (saved8 > 0 && saved8b > 0) {
            ck(solo_into_tile == 0, path, n8, "a -b 1 prefix store loads into no -b 64 engine");
            ck(tile_into_tile == saved8b, path, n8, "live control: a -b 64 prefix store loads into -b 2");
        } else {
            fprintf(stderr, "  gate 8: this model stores no prefix entry (a ring), skipped\n");
        }
    }

    // gate 9: rows a saved state carries are of unknown provenance once loaded
    // (the state bytes hold no mark): kept for the resume that loaded them,
    // never published, never reused by a later request. Here they are a
    // scored prompt's, so either would read decode-route rows.
    {
        int n9 = LENS[0], n9b = n9 + 6;
        prompt_scores ps9 = {0};
        bool scored9 = engine_score_prompt(&ref.e, t, n9, 0, &ps9) != NULL;
        prompt_scores_free(&ps9);
        size_t sb = scored9 ? engine_state_bytes(&ref.e) : 0;
        uint8_t *st = sb ? malloc(sb) : NULL;
        bool ok9 = st && engine_state_save(&ref.e, st) && engine_state_load(&s64.e, t, n9, st);
        free(st);
        if (ok9) {
            prefix_cache_clear();
            engine_prefix_publish(&s64.e, t, n9, n9, 0.0);
            engine_reset(&s512.e);
            prefix_reuse r9 = engine_prefix_reuse(&s512.e, t, n9b);
            ck(r9.keep == 0, path, n9, "a loaded state's rows are not published to another slot");
            int kept9 = engine_rewind_prompt(&s64.e, t, n9b);
            ck(kept9 == 0, path, n9, "a loaded state's rows are not reused by a later request");
            prefix_cache_clear();
        } else {
            fprintf(stderr, "  gate 9: this model's state cannot be imaged (a ring), skipped\n");
        }
    }

    // gate 10: a context imported from disk is forked only by a request that
    // names it. Its rows' provenance is not this process's: an old snapshot
    // holds the width-dependent arithmetic, or rows a later release cannot vouch
    // for. An ordinary request with the same prefix feeds it cold.
    {
        int n10 = LENS[0];
        const char *file = "test-prefill-gate10.kv";
        prefix_cache_clear();
        engine_reset(&s64.e);
        int pinned = engine_feed(&s64.e, t, n10) ? prefix_context_pin(&s64.e, "g10", t, n10) : -1;
        int exported = pinned > 0 ? prefix_context_export("g10", file) : -1;
        prefix_cache_clear();
        int imported = exported > 0 ? prefix_context_import(&s64.e, "g10i", file, "{}") : -1;
        remove(file);
        if (imported > 0) {
            engine_reset(&s512.e);
            prefix_reuse r10 = engine_prefix_reuse(&s512.e, t, n10 + 6);
            ck(r10.keep == 0, path, n10, "an ordinary request forks no imported context");
            // gate 11: a restart's store would load it as an ordinary entry
            const char *store11 = "test-prefill-gate11.store";
            int saved11 = prefix_cache_save(store11);
            remove(store11);
            ck(saved11 == 0, path, n10, "the prefix store does not persist an imported context");
        } else {
            fprintf(stderr, "  gate 10: this model pins no context (a ring), skipped\n");
        }
        prefix_cache_clear();
    }

    // gate 12: a switch that selects other prompt arithmetic is another route.
    // RUNNER_CPU_I8_PREFILL runs a prompt's wide matvecs on the int8 tiles; a
    // store and a slot's rows made without it are not that route's. Live
    // control: with the switch unset again, the store loads and the slot keeps
    // its rows.
    {
        int n12 = LENS[0];
        const char *store = "test-prefill-gate12.store";
        prefix_cache_clear();
        engine_reset(&s64.e);
        bool fed12 = engine_feed(&s64.e, t, n12) != NULL;
        if (fed12) engine_prefix_publish(&s64.e, t, n12, n12, 0.0);
        int saved12 = prefix_cache_save(store);
        prefix_cache_clear();
        env_set("RUNNER_CPU_I8_PREFILL", "1");
        int other = saved12 > 0 ? prefix_cache_load(store, &s2.e) : -2;
        int kept_other = fed12 ? engine_rewind_prompt(&s64.e, t, n12 + 6) : -2;
        prefix_cache_clear();
        env_unset("RUNNER_CPU_I8_PREFILL");
        engine_reset(&s64.e);
        fed12 = engine_feed(&s64.e, t, n12) != NULL;
        int same = saved12 > 0 ? prefix_cache_load(store, &s2.e) : -2;
        int kept_same = fed12 ? engine_rewind_prompt(&s64.e, t, n12 + 6) : -2;
        prefix_cache_clear();
        remove(store);
        if (saved12 > 0) {
            ck(other == 0, path, n12, "a store made without RUNNER_CPU_I8_PREFILL loads into no engine with it");
            ck(same == saved12, path, n12, "live control: without the switch it loads");
        } else {
            fprintf(stderr, "  gate 12: this model stores no prefix entry (a ring), store half skipped\n");
        }
        ck(kept_other == 0, path, n12, "a slot's rows made without the switch are not kept under it");
        ck(kept_same == n12, path, n12, "live control: without the switch the slot keeps its rows");
    }

    ck(control5 == NLEN, path, CTX,
       "live control: rows a scored prompt left, kept, are not the cold feed's");
    ck(control6 == NLEN, path, CTX, "live control: the shared tier forks a prompt that was fed as one");
    ck(control_live == NLEN, path, CTX,
       "live control: the decode route stores different rows at every prompt");
    slot_close(&s64);
    slot_close(&s512);
    slot_close(&ref);
    slot_close(&s1);
    slot_close(&s2);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <fixture.gguf>...\n", argv[0]);
        return 2;
    }
    for (int i = 1; i < argc; i++) check_file(argv[i]);
    if (g_fail) return 1;
    printf("prefill batch identity: all gates pass\n");
    return 0;
}
