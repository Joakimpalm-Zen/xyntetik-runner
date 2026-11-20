// R3.82: qwen35 (Gated DeltaNet hybrids) in the CUDA decode microbatch.
//
// The claim: N qwen35 sequences decoded in one microbatch step get, for every
// sequence and every step, the bits that sequence gets decoded alone. Not only
// the logits: each column's recurrent state (the DeltaNet state and the conv
// history of every recurrent layer) after every step, since R3.78 showed a
// state can differ in its last bits for many steps before any logit moves.
//
//   test-batch-q35-cuda FILE CTX [STEPS]   (make test-cuda-batch-q35)
//
// The reference is each sequence decoded alone (model_forward), on a fixed
// token schedule. A sequence's bits do not depend on what else runs, so the
// batched run varies membership and column order between rounds: at a round
// some sequences sit out and the rest join in a rotated order, which moves
// every sequence across columns and widths. A stale per-column state pointer,
// or one column stepping another's state, fails here.
//
// Anti-vacuity: on a CUDA device the batch must engage (model_batch_engaged),
// every round must have run as a microbatch of its width (the route counts),
// and the states compared must be the device's (gpu_recurrent_download after
// each step). RUNNER_BATCH_QWEN35=0 is the kill switch: with it the batch must
// NOT engage, and the bytes must still be the same (sequential decode).
// Without a CUDA device it fails as skipped.

#include "runner.h"
#include "gate_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NS = 8, MAX_STEPS = 512 };
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

static const int32_t PROMPT[NS][6] = { { 5, 9, 12 }, { 7, 3, 20, 11, 6 }, { 4, 4, 8, 15 }, { 2, 30, 17, 9, 1, 13 },
                                        { 6, 14, 3, 22 }, { 9, 1, 27, 5, 18, 2 }, { 11, 23 }, { 16, 8, 4, 29, 7 } };
static const int PLEN[NS] = { 3, 5, 4, 6, 4, 6, 2, 5 };

typedef struct { uint64_t a, b; } h128;

// two FNV-style lanes over 64-bit words (the buffers are float arrays, so a
// whole number of 4-byte values; a 4-byte tail is folded in alone): any byte
// that differs changes both lanes. Word-wise because the real file's state is
// about 150 MB a sequence a step.
static h128 hash_bytes(h128 h, const void *p, size_t n) {
    const unsigned char *c = p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, c + i, 8);
        h.a = (h.a ^ w) * 0x100000001b3ULL;
        h.b = ((h.b ^ w) * 0x9e3779b97f4a7c15ULL) ^ (h.b >> 29);
    }
    for (; i < n; i++) {
        h.a = (h.a ^ c[i]) * 0x100000001b3ULL;
        h.b = ((h.b ^ c[i]) * 0x9e3779b97f4a7c15ULL) ^ (h.b >> 29);
    }
    return h;
}

static h128 h0(void) { h128 h = { 0xcbf29ce484222325ULL, 0x84222325cbf29ce4ULL }; return h; }

// the device's recurrent state of one sequence, every recurrent layer
static bool state_hash(model_t *m, h128 *out) {
    if (!gpu_recurrent_download(m)) return false;
    int convdim = 2 * m->ssm.state * m->ssm.groups + m->ssm.inner;
    size_t conv = model_rec_count(m) * (size_t)(m->ssm.conv_kernel - 1) * convdim;
    size_t st = model_rec_count(m) * (size_t)m->ssm.v_heads * m->ssm.state * m->ssm.state;
    h128 h = h0();
    h = hash_bytes(h, m->ssm_conv_state, sizeof(float) * conv);
    h = hash_bytes(h, m->ssm_state_mem, sizeof(float) * st);
    *out = h;
    return true;
}

static int32_t tok_at(int i, int k, int nv) { return (int32_t)(3 + (k * 7 + i * 5) % (nv - 4)); }

static bool prefill(model_t *m, int w, int nv) {
    for (int i = 0; i < w; i++) {
        int32_t t[6];
        for (int j = 0; j < PLEN[i]; j++) t[j] = PROMPT[i][j] % nv;
        if (!model_forward_batch(&m[i], t, PLEN[i], 0, true)) return false;
    }
    return true;
}

static h128 ref_lg[NS][MAX_STEPS], ref_st[NS][MAX_STEPS];

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test-batch-q35-cuda FILE CTX [STEPS]\n"); return 2; }
    const char *path = argv[1];
    int ctx = atoi(argv[2]);
    int steps = argc > 3 ? atoi(argv[3]) : 24;
    if (steps < 1 || steps > MAX_STEPS) { fprintf(stderr, "STEPS 1 to %d\n", MAX_STEPS); return 2; }
    const char *ks = getenv("RUNNER_BATCH_QWEN35");
    bool killed = ks && !strcmp(ks, "0");
    model_t m[NS];
    for (int i = 0; i < NS; i++) {
        model_params p;
        memset(&p, 0, sizeof p);
        p.gpu_mode = GPU_AUTO;
        p.n_ctx = ctx;
        if (!model_load(&m[i], path, &p)) { fprintf(stderr, "cannot load %s\n", path); return 1; }
        if (!m[i].gpu && gate_no_device("test-cuda-batch-q35")) return 1;
        if (!m[i].gpu) { fprintf(stderr, "test-cuda-batch-q35: skipped (no CUDA device took %s)\n", path); return 1; }
    }
    if (!m[0].qwen35) { fprintf(stderr, "test-cuda-batch-q35: %s is not a qwen35 file\n", path); return 1; }
    int nv = m[0].n_vocab;

    // ---- the reference: each sequence alone, its tokens on a fixed schedule
    if (!prefill(m, NS, nv)) { ck(0, "reference prefill"); return 1; }
    for (int i = 0; i < NS; i++)
        for (int k = 0; k < steps; k++) {
            float *lg = model_forward(&m[i], tok_at(i, k, nv), PLEN[i] + k);
            if (!lg) { ck(0, "reference forward"); return 1; }
            ref_lg[i][k] = hash_bytes(h0(), lg, sizeof(float) * (size_t)nv);
            if (!state_hash(&m[i], &ref_st[i][k])) { ck(0, "reference state download"); return 1; }
        }

    // the state compared is live: every sequence's state moves at every step
    int still = 0;
    for (int i = 0; i < NS; i++)
        for (int k = 1; k < steps; k++)
            still += ref_st[i][k].a == ref_st[i][k - 1].a && ref_st[i][k].b == ref_st[i][k - 1].b;
    ck(still == 0, "the device's recurrent state changes at every reference step");

    // ---- batched, at widths 2, 4 and 8, membership and column order varying
    int widths[] = { 2, 4, 8 };
    for (size_t wi = 0; wi < sizeof widths / sizeof *widths; wi++) {
        int w = widths[wi];
        if (!prefill(m, w, nv)) { ck(0, "batched prefill"); return 1; }
        model_t *seqs[NS];
        for (int i = 0; i < w; i++) seqs[i] = &m[i];
        model_batch *mb = model_batch_create(seqs, w);
        char what[256];
        snprintf(what, sizeof what, "%s, width %d: the microbatch %s", path, w,
                 killed ? "does not engage (RUNNER_BATCH_QWEN35=0)" : "engages on the device");
        ck(mb && model_batch_engaged(mb) == !killed, what);
        int k[NS] = { 0 }, lg_diff = 0, st_diff = 0, first_lg = -1, first_st = -1, rounds = 0, cols = 0;
        bool ok = mb != NULL;
        // a step model_batch_decode could not run as a microbatch falls back to
        // model_forward, which counts a solo run (route_runs[0], RP_SOLO)
        uint32_t solo0[NS];
        for (int i = 0; i < w; i++) solo0[i] = m[i].route_runs[0];
        for (int r = 0; ok; r++) {
            int idx[NS], pos[NS], n = 0;
            int32_t tok[NS];
            float *out[NS] = { NULL };
            for (int j = 0; j < w; j++) {
                int i = (j + r) % w;                       // rotated column order
                if (k[i] >= steps) continue;
                if (w > 2 && r % 5 == 3 && i % 3 == 1) continue;   // some sit a round out
                idx[n] = i; pos[n] = PLEN[i] + k[i]; tok[n] = tok_at(i, k[i], nv); n++;
            }
            if (!n) {
                bool left = false;
                for (int i = 0; i < w; i++) left |= k[i] < steps;
                if (!left) break;
                continue;
            }
            if (!model_batch_decode(mb, idx, tok, pos, n, out)) { ok = false; break; }
            rounds++; cols += n;
            for (int c = 0; c < n; c++) {
                int i = idx[c];
                if (!out[c]) { ok = false; break; }
                h128 hl = hash_bytes(h0(), out[c], sizeof(float) * (size_t)nv), hs;
                if (!state_hash(&m[i], &hs)) { ok = false; break; }
                if (hl.a != ref_lg[i][k[i]].a || hl.b != ref_lg[i][k[i]].b) {
                    lg_diff++; if (first_lg < 0) first_lg = k[i];
                }
                if (hs.a != ref_st[i][k[i]].a || hs.b != ref_st[i][k[i]].b) {
                    st_diff++; if (first_st < 0) first_st = k[i];
                }
                k[i]++;
            }
        }
        fprintf(stderr, "%s, width %d: %d rounds, %d sequence steps; logits differ at %d (first step %d), "
                "recurrent state differs at %d (first step %d)\n",
                path, w, rounds, cols, lg_diff, first_lg, st_diff, first_st);
        int fell = 0;
        for (int i = 0; i < w; i++) fell += (int)(m[i].route_runs[0] - solo0[i]);
        snprintf(what, sizeof what, "%s, width %d: every batched step ran", path, w);
        ck(ok && cols == w * steps, what);
        snprintf(what, sizeof what, "%s, width %d: %s", path, w, killed
                 ? "every step ran sequentially (the kill switch)"
                 : "no step fell back to the solo forward");
        ck(killed ? fell == cols : fell == 0, what);
        snprintf(what, sizeof what, "%s, width %d: every step's logits are the solo decode's bits", path, w);
        ck(ok && lg_diff == 0, what);
        snprintf(what, sizeof what, "%s, width %d: every step's recurrent state is the solo decode's bits", path, w);
        ck(ok && st_diff == 0, what);
        if (mb) model_batch_free(mb);
    }
    for (int i = 0; i < NS; i++) model_free(&m[i]);
    fprintf(stderr, g_fail ? "test-cuda-batch-q35: FAILED\n" : "test-cuda-batch-q35: ok\n");
    return g_fail;
}
