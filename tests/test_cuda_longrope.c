// phi3 LongRoPE on CUDA: the factor set follows the sequence there too (R4.40.2).
//
// tests/test_longrope_sequence.c holds the rule on the CPU: a sequence of up
// to the model's original context rotates with the short factor set, a longer
// one with the long set from its first token, and a sequence that grows past
// the original context is computed again under the long set at the crossing.
// On CUDA the rope table belonged to the weight set every sequence on the
// device shares, so a load over the original context kept the long set for
// every sequence (model_rope_one_set) and a short prompt read as through the
// long factors. This file is the device arm of that gate: every comparison is
// the device against the device, to the bit, on the same fixture (original
// context 32, loaded with a context of 96, prompts fed BATCH tokens at a time
// as the engine feeds them). Run under RUNNER_REQUIRE_GRAPH_CAPTURE=1, so a
// decode graph that fails to capture is a failure, not an eager pass.
//
//   G1  the tables the device rotates with are the file's (base^(-2j/d) over
//       each factor set, the absolute anchor), and a slot's table is its set's;
//   G2  a sequence within the original context computes what a load at the
//       original context computes;
//   G3  a sequence past it computes what a load that keeps the long set for
//       everything computes (the device's arithmetic before R4.40.2);
//   G4  a sequence that grows past the original context is computed again
//       under the long set once, and from the crossing on decodes what the
//       same text fed as a long sequence decodes; replaying the decode graph
//       gives the eager path's bits on both sides of the crossing; a short
//       prompt after a long sequence decodes as at the original context;
//   G5  two slots on one weight set, one on each set, decoding in turn: each
//       computes its own set's answer;
//   G6  slots on both sets decode as one device microbatch at every step
//       (R4.40.3), each column at its row of the batch's buffer, and each
//       column decodes what its sequence decodes alone;
//   G7  a batch member that crosses the original context between two steps of
//       one width keeps, step for step, the bits of its loop alone.
// Live controls:
//   L1  on this fixture the two sets give different logits on the device, so
//       G2 and G3 can tell them apart;
//   L2  with the decode graph's recapture at a set change skipped
//       (RUNNER_TEST_ROPE_GRAPH_STALE=1) the replay check of G4 goes red, and
//       with the batch graph's recapture at a change of the set mix skipped
//       the same seam turns G7 red.
#include "runner.h"
#include "gpu.h"
#include "gate_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define env_set(k, v) _putenv_s(k, v)
#define env_unset(k)  _putenv_s(k, "")
#else
#define env_set(k, v) setenv(k, v, 1)
#define env_unset(k)  unsetenv(k)
#endif

static const char *g_path = "test-phi3-longrope.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

enum { ORIG = 32, CTX = 96, BATCH = 4, LONG = 40, GROW = 30, STEPS = 6 };

typedef struct { model_t m; tokenizer tok; sampler smp; engine e; } slot;

static bool slot_open(slot *s, int ctx) {
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode  = GPU_AUTO;
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
    if (!lg) return NULL;
    float *out = malloc(sizeof(float) * (size_t)m->n_vocab);
    if (out) memcpy(out, lg, sizeof(float) * (size_t)m->n_vocab);
    return out;
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

static float *step(slot *s, int32_t tok) {
    return snap(&s->m, engine_feed(&s->e, &tok, 1));
}

// The server's generation loop on two slots (R4.40.3, G7): each slot's prompt
// fed whole, then G7N steps of engine_gen_step (which samples the step's token
// and, at the crossing, moves the slot to the long set and computes its rows
// again) and the forward of the two tokens, in one model_batch_decode (mb) or
// each alone through model_forward (mb NULL). The logits of every forward in
// lg[slot][step]; rebuilds[slot] the recomputes each slot made.
enum { G7N = 8 };
static bool gen_two(slot *s[2], const int *plen, const int32_t *t, model_batch *mb,
                    float *lg[2][G7N], int *rebuilds) {
    float *cur[2] = { NULL, NULL };
    bool ok = true;
    for (int j = 0; j < 2 && ok; j++) {
        ok = (cur[j] = whole(s[j], t, plen[j])) != NULL;
        s[j]->e.ignore_eos = true;
        s[j]->e.rope_rebuilds = 0;
        engine_gen_begin(&s[j]->e, G7N + 2);
    }
    for (int k = 0; k < G7N && ok; k++) {
        int32_t tok[2];
        int pos[2];
        for (int j = 0; j < 2 && ok; j++)
            ok = engine_gen_step(&s[j]->e, cur[j], NULL, NULL, &tok[j], &pos[j]) == ENGINE_STEP_MORE;
        if (!ok) break;
        if (mb) {
            int idx[2] = { 0, 1 };
            float *out[2] = { NULL, NULL };
            ok = model_batch_decode(mb, idx, tok, pos, 2, out);
            for (int j = 0; j < 2 && ok; j++) ok = (lg[j][k] = snap(&s[j]->m, out[j])) != NULL;
        } else {
            for (int j = 0; j < 2 && ok; j++)
                ok = (lg[j][k] = snap(&s[j]->m, model_forward(&s[j]->m, tok[j], pos[j]))) != NULL;
        }
        for (int j = 0; j < 2 && ok; j++) { free(cur[j]); ok = (cur[j] = snap(&s[j]->m, lg[j][k])) != NULL; }
    }
    for (int j = 0; j < 2; j++) {
        free(cur[j]);
        engine_gen_end(&s[j]->e, NULL, NULL, NULL);
        s[j]->e.ignore_eos = false;
        rebuilds[j] = s[j]->e.rope_rebuilds;
    }
    return ok;
}

// a prompt of `from` tokens, then t[from..n) decoded one at a time; the
// logits of each decode step in out[i - from]
static bool grow(slot *s, const int32_t *t, int from, int n, float **out) {
    engine_reset(&s->e);
    engine_rewind(&s->e, t, from);
    if (!engine_feed(&s->e, t, from)) return false;
    for (int i = from; i < n; i++)
        if (!(out[i - from] = step(s, t[i]))) return false;
    return true;
}

// steps a and b of n, the count that differ (a missing step counts)
static int steps_differ(const model_t *m, float **a, float **b, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) bad += differ(m, a[i], b[i]) != 0;
    return bad;
}

static void steps_free(float **a, int n) {
    for (int i = 0; i < n; i++) { free(a[i]); a[i] = NULL; }
}

// the device's table for the instance (its global layers), against a host table
static bool device_table_is(const model_t *m, const float *want, int half) {
    float *got = malloc(sizeof(float) * (size_t)half);
    bool ok = got && want && gpu_rope_table_in_use(m, got, half) &&
              !memcmp(got, want, sizeof(float) * (size_t)half);
    free(got);
    return ok;
}

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    env_unset("RUNNER_TEST_ROPE_GRAPH_STALE");
    env_unset("RUNNER_CUDA_GRAPH_OFF");
    slot at_orig, wide, cold, one;
    long b0 = gpu_live_buffers();
    if (!slot_open(&at_orig, ORIG)) { fprintf(stderr, "cannot load %s\n", g_path); return 1; }
    long b1 = gpu_live_buffers();
    if (!slot_open(&wide, CTX)) { fprintf(stderr, "cannot load %s\n", g_path); return 1; }
    long b2 = gpu_live_buffers();
    if (!slot_open(&cold, CTX)) { fprintf(stderr, "cannot load %s\n", g_path); return 1; }
    long b3 = gpu_live_buffers();
    if (!slot_open(&one, CTX)) { fprintf(stderr, "cannot load %s\n", g_path); return 1; }
    long b4 = gpu_live_buffers();
    if (!at_orig.m.gpu || !wide.m.gpu || !cold.m.gpu || !one.m.gpu) {
        gate_no_device("test-cuda-longrope");   // prints the driver's log when that is why
        fprintf(stderr, "test-cuda-longrope: skipped (no CUDA device took the fixture)\n");
        return 1;
    }
    model_rope_one_set(&one.m, "this test");
    int V = wide.m.n_vocab, half = wide.m.rope_dim / 2;
    int32_t t[LONG], x[STEPS], y[STEPS];
    for (int i = 0; i < LONG; i++) t[i] = 3 + (i * 7 + i / 5) % (V - 4);
    for (int k = 0; k < STEPS; k++) { x[k] = 3 + (k * 11 + 1) % (V - 4); y[k] = 3 + (k * 13 + 5) % (V - 4); }
    fprintf(stderr, "device buffers: %ld before, +%ld at -c %d, +%ld +%ld +%ld at -c %d\n",
            b0, b1 - b0, ORIG, b2 - b1, b3 - b2, b4 - b3, CTX);

    // ---- G1: the tables
    ck(wide.m.rope_long_from == ORIG && !wide.m.rope_long && wide.m.rope_freq_short && wide.m.rope_freq_long,
       "G1: a device load past the original context keeps both sets and starts on the short one");
    {
        gguf_tensor *fs = gguf_find_tensor(&wide.m.gf, "rope_factors_short.weight");
        gguf_tensor *fl = gguf_find_tensor(&wide.m.gf, "rope_factors_long.weight");
        int bad_s = 0, bad_l = 0;
        for (int j = 0; fs && fl && wide.m.rope_freq_short && wide.m.rope_freq_long && j < half; j++) {
            // to a part in a million, as the CPU gate: two compilations of one expression
            float base = powf(wide.m.rope_base, -2.0f * j / wide.m.rope_dim);
            float ws = base / ((const float *)fs->data)[j], wl = base / ((const float *)fl->data)[j];
            bad_s += fabsf(wide.m.rope_freq_short[j] - ws) > 1e-6f * ws;
            bad_l += fabsf(wide.m.rope_freq_long[j] - wl) > 1e-6f * wl;
        }
        ck(fs && fl && bad_s == 0 && bad_l == 0,
           "G1: the host's two tables are the base over the file's two factor sets");
    }
    ck(device_table_is(&at_orig.m, at_orig.m.rope_inv_freq, half),
       "G1: a load at the original context rotates on the device with its one table");
    ck(device_table_is(&wide.m, wide.m.rope_freq_short, half),
       "G1: a fresh slot past it rotates on the device with the short table");

    // ---- G2 and L1: within the original context
    float *a = whole(&at_orig, t, 20), *b = whole(&wide, t, 20), *o = whole(&one, t, 20);
    int d2 = differ(&wide.m, a, b), dl = differ(&wide.m, o, b);
    fprintf(stderr, "20 tokens: %d of %d logits differ from the original-context load's, %d from the long set's\n",
            d2, V, dl);
    ck(d2 == 0, "G2: a 20-token sequence at a context of 96 computes on the device what it computes at 32");
    ck(dl > 0, "L1: and not what the long set computes for it, so the two sets are told apart");
    ck(!wide.m.rope_long && device_table_is(&wide.m, wide.m.rope_freq_short, half),
       "G1: after it the slot is on the short set, on the host and on the device");
    free(a); free(b); free(o);

    // ---- G3: past it, from the first token
    float *c = whole(&cold, t, LONG);
    o = whole(&one, t, LONG);
    ck(cold.m.rope_long && device_table_is(&cold.m, cold.m.rope_freq_long, half),
       "G1: a 40-token prompt puts the slot on the long set, on the host and on the device");
    ck(differ(&cold.m, c, o) == 0, "G3: a 40-token sequence computes on the device what the long set alone computes");
    free(c); free(o);

    // ---- G4: growing past it, through the decode graph and without it
    enum { NG = LONG - GROW };
    float *gr[NG] = {0}, *eg[NG] = {0}, *ref[LONG - ORIG] = {0};
    wide.e.rope_rebuilds = 0;
    bool g_ok = grow(&wide, t, GROW, LONG, gr);
    ck(g_ok && wide.m.rope_long && wide.e.rope_rebuilds == 1,
       "G4: the 33rd token moves the sequence to the long set, once");
    ck(device_table_is(&wide.m, wide.m.rope_freq_long, half), "G4: and the device rotates it with the long table");
    // the same 40 tokens as one long sequence: the first 32 fed as a prompt in
    // BATCH-token pieces (as the crossing computes them again), then decoded
    bool r_ok = false;
    engine_reset(&cold.e);
    engine_rewind(&cold.e, t, LONG);
    if (engine_feed(&cold.e, t, ORIG)) {
        r_ok = true;
        for (int i = ORIG; i < LONG && r_ok; i++) r_ok = (ref[i - ORIG] = step(&cold, t[i])) != NULL;
    }
    int d4 = g_ok && r_ok ? steps_differ(&wide.m, gr + (ORIG - GROW), ref, LONG - ORIG) : -1;
    if (d4) fprintf(stderr, "G4: %d of %d steps from the crossing differ from the long sequence's\n", d4, LONG - ORIG);
    ck(d4 == 0, "G4: from the crossing on, it decodes what the same text fed as a long sequence decodes");
    env_set("RUNNER_CUDA_GRAPH_OFF", "1");
    wide.e.rope_rebuilds = 0;
    bool e_ok = grow(&wide, t, GROW, LONG, eg);
    env_unset("RUNNER_CUDA_GRAPH_OFF");
    int dg = g_ok && e_ok ? steps_differ(&wide.m, gr, eg, NG) : -1;
    if (dg) fprintf(stderr, "G4: %d of %d decode steps differ between the graph and the eager path\n", dg, NG);
    ck(e_ok && wide.e.rope_rebuilds == 1, "G4: without the decode graph the sequence crosses once too");
    ck(e_ok && dg == 0, "G4: the decode graph's steps are the eager path's, before and after the crossing");
    // back: a short prompt on the slot that last decoded on the long set
    {
        float *bk[STEPS] = {0}, *bo[STEPS] = {0};
        int keep = engine_rewind(&cold.e, t, 20);
        ck(keep == 0 && cold.e.rewind_how == REWIND_ROPE && !cold.m.rope_long,
           "G4: a 20-token prompt after a long sequence keeps no row and takes the short set");
        bool k_ok = engine_feed(&cold.e, t, 20) != NULL;
        for (int k = 0; k < STEPS && k_ok; k++) k_ok = (bk[k] = step(&cold, x[k])) != NULL;
        engine_reset(&at_orig.e);
        engine_rewind(&at_orig.e, t, 20);
        bool o_ok = engine_feed(&at_orig.e, t, 20) != NULL;
        for (int k = 0; k < STEPS && o_ok; k++) o_ok = (bo[k] = step(&at_orig, x[k])) != NULL;
        int db = k_ok && o_ok ? steps_differ(&cold.m, bk, bo, STEPS) : -1;
        if (db) fprintf(stderr, "G4: %d of %d steps after the way back differ\n", db, STEPS);
        ck(device_table_is(&cold.m, cold.m.rope_freq_short, half), "G4: and the device rotates it with the short table");
        ck(db == 0, "G4: and it decodes as at the original context");
        steps_free(bk, STEPS); steps_free(bo, STEPS);
    }

    // ---- G5: one weight set, a slot on each set, decoding in turn
    ck(b2 - b1 > b3 - b2 && b3 - b2 == b4 - b3,
       "G5: the second and third slots at -c 96 take no weights of their own (one weight set)");
    {
        float *ws_[STEPS] = {0}, *cs_[STEPS] = {0}, *wr[STEPS] = {0}, *cr[STEPS] = {0};
        free(whole(&wide, t, 20));
        free(whole(&cold, t, LONG));
        ck(!wide.m.rope_long && cold.m.rope_long, "G5: one slot is on the short set and the other on the long");
        bool ok = true;
        for (int k = 0; k < STEPS && ok; k++) {
            ok = (ws_[k] = step(&wide, x[k])) != NULL;
            ok = ok && (cs_[k] = step(&cold, y[k])) != NULL;
        }
        // each alone: the short one at the original context, the long one on the long set alone
        free(whole(&at_orig, t, 20));
        for (int k = 0; k < STEPS && ok; k++) ok = (wr[k] = step(&at_orig, x[k])) != NULL;
        free(whole(&one, t, LONG));
        for (int k = 0; k < STEPS && ok; k++) ok = (cr[k] = step(&one, y[k])) != NULL;
        int dw = ok ? steps_differ(&wide.m, ws_, wr, STEPS) : -1;
        int dc = ok ? steps_differ(&cold.m, cs_, cr, STEPS) : -1;
        fprintf(stderr, "G5: in turn, %d of %d short steps and %d of %d long steps differ from each alone\n",
                dw, STEPS, dc, STEPS);
        ck(dw == 0, "G5: the short slot decodes, in turn with a long one, what the original-context load decodes");
        ck(dc == 0, "G5: the long slot decodes, in turn with a short one, what the long set alone decodes");
        steps_free(ws_, STEPS); steps_free(cs_, STEPS); steps_free(wr, STEPS); steps_free(cr, STEPS);
    }

    // ---- G6: one device microbatch with a column on each set (R4.40.3). Three
    // slots of the two-set load, listed long, short, long, so the columns are
    // reordered and split; each column must be the bits of that sequence
    // decoded alone (model_batch_decode's guarantee), and the batch must engage.
    {
        model_t *trio[3] = { &cold.m, &wide.m, &one.m };
        model_batch *mb = model_batch_create(trio, 3);
        ck(mb && model_batch_engaged(mb), "G6: slots on both sets of a two-set load decode as one device microbatch");
        float *bt[3][STEPS] = {{0}}, *so[3][STEPS] = {{0}};
        slot *sl[3] = { &cold, &wide, &one };
        int plen[3] = { LONG, 20, LONG };
        bool ok = mb != NULL;
        for (int j = 0; j < 3; j++) free(whole(sl[j], t, plen[j]));
        ck(cold.m.rope_long && !wide.m.rope_long && device_table_is(&one.m, one.m.rope_freq_long, half),
           "G6: the batch holds a slot on the short set between two on the long one");
        // The batch must run on the device at every step, not only engage at creation:
        // a step gpu_batch_decode refuses falls back to one-by-one decoding
        // (model_batch_decode), which hands each sequence its alone bits and would pass
        // the comparison below. gpu_batch_decode puts the short set first in the
        // caller's order, then the long one: column 0 is slot 1 (short), columns 1 and
        // 2 are slots 0 and 2. Column c's logits are row c of the batch's own buffer,
        // at the same place every step (out[ord[c]] = h_logits + c * n_vocab), so
        // out[1], out[0] and out[2] must be consecutive rows of one buffer. One-by-one
        // decoding returns each slot's own logits instead.
        const float *base6 = NULL;
        int placed = 0;
        size_t nv6 = (size_t)cold.m.n_vocab;
        for (int k = 0; k < STEPS && ok; k++) {
            int idx[3] = { 0, 1, 2 }, ps[3];
            int32_t tk[3] = { y[k], x[k], y[k] };
            float *out[3] = { NULL, NULL, NULL };
            for (int j = 0; j < 3; j++) ps[j] = plen[j] + k;
            ok = model_batch_decode(mb, idx, tk, ps, 3, out);
            if (ok && out[1] && out[0] == out[1] + nv6 && out[2] == out[1] + 2 * nv6 &&
                (!base6 || out[1] == base6))
                placed++;
            if (ok && !base6) base6 = out[1];
            for (int j = 0; j < 3 && ok; j++) ok = (bt[j][k] = snap(&sl[j]->m, out[j])) != NULL;
        }
        fprintf(stderr, "G6: %d of %d steps' columns were rows 1, 0 and 2 of the batch's own buffer\n",
                placed, STEPS);
        ck(ok && placed == STEPS,
           "G6: every step ran on the device, each column at its row of the batch's buffer (no one-by-one fall back)");
        // each alone, on the same slot, from the same prompt
        for (int j = 0; j < 3 && ok; j++) {
            free(whole(sl[j], t, plen[j]));
            for (int k = 0; k < STEPS && ok; k++) ok = (so[j][k] = step(sl[j], j == 1 ? x[k] : y[k])) != NULL;
        }
        int d6[3] = { -1, -1, -1 };
        for (int j = 0; j < 3 && ok; j++) d6[j] = steps_differ(&sl[j]->m, bt[j], so[j], STEPS);
        fprintf(stderr, "G6: batch against alone, steps that differ: long %d, short %d, long (one set) %d of %d\n",
                d6[0], d6[1], d6[2], STEPS);
        ck(ok && d6[0] == 0 && d6[1] == 0 && d6[2] == 0,
           "G6: every column decodes in the batch what its sequence decodes alone");
        for (int j = 0; j < 3; j++) { steps_free(bt[j], STEPS); steps_free(so[j], STEPS); }
        if (mb) model_batch_free(mb);
    }

    // ---- G7: a batch member crosses the original context between two steps
    // of one width (R4.40.3). The batch's graph for width 2 was captured with
    // one column on each set; after the crossing both are on the long set, and
    // the graph must be captured again for the crossing slot to keep its bits.
    // The same loop with RUNNER_TEST_ROPE_GRAPH_STALE=1 keeps that graph, and
    // must be seen (the live control).
    {
        slot *pair[2] = { &wide, &cold };
        model_t *pm[2] = { &wide.m, &cold.m };
        int plen[2] = { GROW, LONG }, rb_b[2] = { 0, 0 }, rb_s[2] = { 0, 0 }, rb_m[2] = { 0, 0 };
        float *bb[2][G7N] = {{0}}, *ss[2][G7N] = {{0}}, *mm[2][G7N] = {{0}};
        model_batch *mb = model_batch_create(pm, 2);
        bool ok_b = mb && model_batch_engaged(mb) && gen_two(pair, plen, t, mb, bb, rb_b);
        bool ok_s = gen_two(pair, plen, t, NULL, ss, rb_s);
        int dw = ok_b && ok_s ? steps_differ(&wide.m, bb[0], ss[0], G7N) : -1;
        int dc = ok_b && ok_s ? steps_differ(&cold.m, bb[1], ss[1], G7N) : -1;
        fprintf(stderr, "G7: batched against alone, steps that differ: the crossing slot %d, the long one %d of %d; "
                "recomputes %d and %d batched, %d and %d alone\n", dw, dc, G7N, rb_b[0], rb_b[1], rb_s[0], rb_s[1]);
        ck(ok_b && rb_b[0] == 1 && rb_b[1] == 0 && rb_s[0] == 1,
           "G7: in the batch the short slot crosses once, mid-generation, and the long one never");
        ck(dw == 0 && dc == 0, "G7: every step of both slots in the batch is the bits of the same loop alone");
        // a fresh batch, so its width-2 graph is first captured with a column
        // on each set and then kept across the crossing
        if (mb) model_batch_free(mb);
        mb = model_batch_create(pm, 2);
        env_set("RUNNER_TEST_ROPE_GRAPH_STALE", "1");
        bool ok_m = mb && gen_two(pair, plen, t, mb, mm, rb_m);
        env_unset("RUNNER_TEST_ROPE_GRAPH_STALE");
        int dm = ok_m && ok_s ? steps_differ(&wide.m, mm[0], ss[0], G7N) : -1;
        fprintf(stderr, "L2: with the batch graph kept across the split change, %d of %d steps of the crossing slot differ\n",
                dm, G7N);
        ck(dm > 0, "L2: a batch graph kept across a change of the set mix is seen by G7");
        for (int j = 0; j < 2; j++) { steps_free(bb[j], G7N); steps_free(ss[j], G7N); steps_free(mm[j], G7N); }
        if (mb) model_batch_free(mb);
    }

    // ---- L2: the recapture mutant must turn G4's replay check red
    {
        float *st[NG] = {0};
        env_set("RUNNER_TEST_ROPE_GRAPH_STALE", "1");
        bool s_ok = grow(&wide, t, GROW, LONG, st);
        env_unset("RUNNER_TEST_ROPE_GRAPH_STALE");
        int ds = s_ok && e_ok ? steps_differ(&wide.m, st, eg, NG) : -1;
        fprintf(stderr, "L2: with the recapture skipped, %d of %d decode steps differ from the eager path\n", ds, NG);
        ck(ds > 0, "L2: a decode graph kept across a set change is seen by G4");
        steps_free(st, NG);
    }

    steps_free(gr, NG); steps_free(eg, NG); steps_free(ref, LONG - ORIG);
    slot_close(&at_orig); slot_close(&wide); slot_close(&cold); slot_close(&one);
    return g_fail;
}
