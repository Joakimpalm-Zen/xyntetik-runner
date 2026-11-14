// phi3 LongRoPE: the factor set follows the sequence, not the context (R4.40).
//
// A LongRoPE file carries two sets of rotary factors. The publisher's code
// rotates a sequence with the short set while it is no longer than the
// model's original context and with the long set once it is longer, for
// every position of that sequence. The runner chose the set from the
// requested context instead: served with a context over the original one,
// every sequence took the long set from its first token, and Phi-3.5-mini
// read 0.205 mean KLD from its publisher's model on a short text where it
// reads 0.004 at the original context.
//
// The gates, bit for bit, on a phi3 fixture whose original context is 32,
// loaded with a context of 96:
//
//   1. both tables are the file's: base^(-2j/d) over each factor set;
//   2. a sequence within the original context computes what a load at the
//      original context computes;
//   3. a sequence past it computes what a load that keeps the long set for
//      everything computes (the arithmetic before this change);
//   4. a sequence that grows past the original context is computed again
//      under the long set at the crossing, once, and from there on matches a
//      sequence that was fed whole; generation across the crossing emits the
//      tokens a whole feed emits;
//   5. a shorter prompt after a long sequence keeps none of its rows and
//      reads as on a slot that never held the long one.
#include "runner.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_path = "test-phi3-longrope.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

enum { ORIG = 32, CTX = 96, BATCH = 4, LONG = 40 };

typedef struct { model_t m; tokenizer tok; sampler smp; engine e; } slot;

static bool slot_open(slot *s, int ctx) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx     = ctx;
    p.n_batch   = BATCH;
    memset(s, 0, sizeof(*s));
    if (!model_load(&s->m, g_path, &p)) return false;
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

static float *snap(const model_t *m, const float *lg) {
    float *out = malloc(sizeof(float) * (size_t)m->n_vocab);
    if (out && lg) memcpy(out, lg, sizeof(float) * (size_t)m->n_vocab);
    return lg ? out : (free(out), NULL);
}

static int differ(const model_t *m, const float *a, const float *b) {
    if (!a || !b) return -1;
    int diffs = 0;
    for (int i = 0; i < m->n_vocab; i++) if (a[i] != b[i]) diffs++;
    return diffs;
}

// one whole sequence of n tokens on a slot, as a request's prompt is fed
static float *whole(slot *s, const int32_t *t, int n) {
    engine_reset(&s->e);
    engine_rewind(&s->e, t, n);
    return snap(&s->m, engine_feed(&s->e, t, n));
}

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    slot at_orig, wide, cold, one;
    if (!slot_open(&at_orig, ORIG) || !slot_open(&wide, CTX) || !slot_open(&cold, CTX) ||
        !slot_open(&one, CTX)) {
        fprintf(stderr, "cannot load %s four times\n", g_path);
        return 1;
    }
    model_rope_one_set(&one.m, "this test");
    int V = wide.m.n_vocab, half = wide.m.rope_dim / 2;
    int32_t t[LONG];
    for (int i = 0; i < LONG; i++) t[i] = 3 + (i * 7 + i / 5) % (V - 4);

    // ---- gate 1: the tables
    ck(at_orig.m.rope_long_from == 0, "a load at the original context has one set");
    ck(wide.m.rope_long_from == ORIG && !wide.m.rope_long,
       "a load past it has both and starts on the short one");
    ck(one.m.rope_long_from == 0 && !model_rope_wants_long(&one.m, LONG) && !one.m.rope_long,
       "a load told to keep one set has one, and no sequence asks it for another");
    gguf_tensor *fs = gguf_find_tensor(&wide.m.gf, "rope_factors_short.weight");
    gguf_tensor *fl = gguf_find_tensor(&wide.m.gf, "rope_factors_long.weight");
    int bad_s = 0, bad_l = 0, same = 0;
    for (int j = 0; fs && fl && j < half; j++) {
        float base = powf(wide.m.rope_base, -2.0f * j / wide.m.rope_dim);
        // to a part in a million, not to the bit: the load and this line
        // are two compilations of one expression, and an x86 runner's
        // compiler contracted them differently
        float ws = base / ((const float *)fs->data)[j], wl = base / ((const float *)fl->data)[j];
        bad_s += fabsf(wide.m.rope_freq_short[j] - ws) > 1e-6f * ws;
        bad_l += fabsf(wide.m.rope_freq_long[j] - wl) > 1e-6f * wl;
        same  += wide.m.rope_freq_short[j] == wide.m.rope_freq_long[j];
    }
    ck(fs && fl && bad_s == 0, "the short table is the base over the file's short factors");
    ck(fs && fl && bad_l == 0, "the long table is the base over the file's long factors");
    ck(same < half, "the fixture's two sets differ, so the gates below can tell them apart");
    ck(!memcmp(one.m.rope_inv_freq, wide.m.rope_freq_long, sizeof(float) * (size_t)half),
       "a one-set load rotates with the long table");

    // ---- gate 2: within the original context
    float *a = whole(&at_orig, t, 20), *b = whole(&wide, t, 20), *o = whole(&one, t, 20);
    ck(differ(&wide.m, a, b) == 0,
       "a 20-token sequence at a context of 96 computes what it computes at 32");
    // what was compared, so a failure of the next check names its cause: the
    // count, the largest difference, and the table each slot rotated with
    {
        float mx = 0;
        for (int i = 0; o && b && i < V; i++) if (fabsf(o[i] - b[i]) > mx) mx = fabsf(o[i] - b[i]);
        fprintf(stderr, "20 tokens, long set against short: %d of %d logits differ, largest %.3g; "
                "inv_freq[%d] one %.9g wide %.9g original %.9g; tables at %p %p %p\n",
                differ(&wide.m, o, b), V, (double)mx, half - 1,
                (double)one.m.rope_inv_freq[half - 1], (double)wide.m.rope_inv_freq[half - 1],
                (double)at_orig.m.rope_inv_freq[half - 1],
                (void *)one.m.rope_inv_freq, (void *)wide.m.rope_inv_freq, (void *)at_orig.m.rope_inv_freq);
    }
    ck(differ(&wide.m, o, b) > 0, "and not what the long set gives it");
    ck(!wide.m.rope_long, "the slot is on the short set");
    free(a); free(b); free(o);

    // ---- gate 3: past it, from the first token
    float *c = whole(&cold, t, LONG);
    o = whole(&one, t, LONG);
    ck(cold.m.rope_long, "a 40-token prompt puts the slot on the long set");
    ck(differ(&cold.m, c, o) == 0,
       "a 40-token sequence computes what the long set alone computes");

    // ---- gate 4: growing past it
    engine_reset(&wide.e);
    engine_rewind(&wide.e, t, 30);
    ck(engine_feed(&wide.e, t, 30) != NULL && !wide.m.rope_long,
       "30 tokens are fed on the short set");
    float *w = NULL;
    for (int i = 30; i < LONG; i++) w = engine_feed(&wide.e, t + i, 1);
    ck(wide.m.rope_long && wide.e.rope_rebuilds == 1,
       "the 33rd token moves the sequence to the long set, once");
    w = snap(&wide.m, w);
    ck(differ(&wide.m, w, c) == 0,
       "a sequence that grew past the original context matches one fed whole");
    free(w);

    // generation across the crossing: 31 prompt tokens, six generated
    engine_reset(&wide.e);
    engine_rewind(&wide.e, t, 31);
    wide.e.rope_rebuilds = 0;
    float *lg = engine_feed(&wide.e, t, 31);
    int ng = lg ? engine_generate(&wide.e, lg, 6, NULL, NULL, NULL) : 0;
    ck(ng >= 3, "the fixture generates past the crossing");
    ck(wide.e.rope_rebuilds == (31 + ng > ORIG ? 1 : 0), "with one recompute at the crossing");
    if (ng >= 3) {
        // the same text as a prompt that is already past the original
        // context, then the last generated token again
        int32_t seq[LONG];
        memcpy(seq, wide.e.hist, sizeof(int32_t) * (size_t)(31 + ng));
        engine_reset(&cold.e);
        engine_rewind(&cold.e, seq, 31 + ng - 1);
        lg = engine_feed(&cold.e, seq, 31 + ng - 1);
        int one_more = lg ? engine_generate(&cold.e, lg, 1, NULL, NULL, NULL) : 0;
        ck(31 + ng - 1 > ORIG && one_more == 1 && cold.e.hist[31 + ng - 1] == seq[31 + ng - 1],
           "the last token generated across the crossing is the one a whole feed generates");
    }

    // ---- gate 5: back to a short prompt
    c = (free(c), whole(&cold, t, LONG));
    int keep = engine_rewind(&cold.e, t, 20);
    ck(keep == 0 && cold.e.rewind_how == REWIND_ROPE && !cold.m.rope_long,
       "a 20-token prompt after a 40-token sequence keeps no row and takes the short set");
    b = snap(&cold.m, engine_feed(&cold.e, t, 20));
    a = whole(&at_orig, t, 20);
    ck(differ(&cold.m, a, b) == 0, "and reads as at the original context");
    // a longer prompt that stays on the long set keeps its prefix
    free(whole(&cold, t, LONG));
    keep = engine_rewind(&cold.e, t, LONG - 2);
    ck(keep == LONG - 3 && cold.e.rewind_how != REWIND_ROPE,
       "a prompt that stays past the original context keeps its rows");
    free(a); free(b); free(c); free(o);

    // ---- gate 6: the whole-sequence reads outside the engine take the set
    // of their own length, whatever the instance held before
    {
        int E = wide.m.n_embd;
        float *h_wide = malloc(sizeof(float) * (size_t)LONG * E);
        float *h_ref  = malloc(sizeof(float) * (size_t)LONG * E);
        // a long read on a slot that sits on the short set, against the long set alone
        free(whole(&wide, t, 20));
        ck(h_wide && h_ref && model_hidden_states(&wide.m, t, LONG, h_wide) &&
           model_hidden_states(&one.m, t, LONG, h_ref) &&
           !memcmp(h_wide, h_ref, sizeof(float) * (size_t)LONG * E),
           "hidden states of 40 tokens are the long set's, after a short sequence");
        // a short read on a slot that sits on the long set, against the original context
        ck(model_hidden_states(&wide.m, t, 20, h_wide) &&
           model_hidden_states(&at_orig.m, t, 20, h_ref) &&
           !memcmp(h_wide, h_ref, sizeof(float) * (size_t)20 * E),
           "hidden states of 20 tokens are the short set's, after a long one");
        double n_wide = 0, n_ref = 0;
        ck(model_seq_nll(&wide.m, t, LONG, NULL, &n_wide) &&
           model_seq_nll(&one.m, t, LONG, NULL, &n_ref) && n_wide == n_ref,
           "the loss of a 40-token window is the long set's, after a short read");
        ck(model_seq_nll(&wide.m, t, 20, NULL, &n_wide) &&
           model_seq_nll(&at_orig.m, t, 20, NULL, &n_ref) && n_wide == n_ref,
           "the loss of a 20-token window is the short set's, after a long one");
        free(h_wide); free(h_ref);
    }
    // ---- gate 7: a long prompt fed in pieces stays on the long set. Only a
    // sequence that grows past the original context changes set while it is
    // fed; a piece that ends short of it is part of a longer prompt.
    {
        prompt_scores pw, po;
        float *lw = engine_score_prompt(&wide.e, t, LONG, 0, &pw);
        float *lo = engine_score_prompt(&one.e, t, LONG, 0, &po);
        ck(lw && lo && !memcmp(pw.lp, po.lp, sizeof(*pw.lp) * (size_t)LONG),
           "a 40-token prompt scored token by token is on the long set from its first token");
        prompt_scores_free(&pw); prompt_scores_free(&po);
        engine_reset(&wide.e);
        engine_rewind(&wide.e, t, LONG);
        wide.e.rope_rebuilds = 0;
        float *piece = engine_feed(&wide.e, t, 10) ? engine_feed(&wide.e, t + 10, LONG - 10) : NULL;
        piece = snap(&wide.m, piece);
        float *cold_l = whole(&cold, t, LONG);
        ck(wide.e.rope_rebuilds == 0 && differ(&wide.m, piece, cold_l) == 0,
           "a 40-token prompt fed as 10 and 30 is computed once, on the long set");
        free(piece); free(cold_l);
        // and a new sequence after a reset starts on the short set again
        engine_reset(&wide.e);
        float *fresh = snap(&wide.m, engine_feed(&wide.e, t, 20));
        float *ref = whole(&at_orig, t, 20);
        ck(differ(&wide.m, fresh, ref) == 0, "a reset slot feeds a short sequence on the short set");
        free(fresh); free(ref);
    }
    // an image of a one-set load is consistent (a stale flag refused every one)
    {
        free(whole(&one, t, 20));
        size_t nb = prefix_cache_entry_bytes(&one.m, 20);
        uint8_t *img = malloc(nb);
        ck(img && engine_state_save(&one.e, img), "a one-set load's sequence can be imaged");
        free(img);
    }

    slot_close(&at_orig); slot_close(&wide); slot_close(&cold); slot_close(&one);
    return g_fail;
}
