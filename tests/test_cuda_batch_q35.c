// R3.82: qwen35 (Gated DeltaNet hybrids) in the CUDA decode microbatch.
//
// The claim: N qwen35 sequences decoded in one microbatch step get, for every
// sequence and every step, the bits that sequence gets decoded alone. Not only
// the logits: each column's recurrent state (the DeltaNet state and the conv
// history of every recurrent layer) after every step, since R3.78 showed a
// state can differ in its last bits for many steps before any logit moves. And
// when a step fails, each member gets exactly one logical step, or the step
// fails without serving logits.
//
//   test-batch-q35-cuda FILE CTX [STEPS]                 identity (make test-cuda-batch-q35)
//   test-batch-q35-cuda FILE CTX STEPS fault-ref K OUT   member K alone, its first decode step failing
//   test-batch-q35-cuda FILE CTX STEPS fault-one REF0    an engaged batch reduced to one member, failing
//   test-batch-q35-cuda FILE CTX STEPS fault-multi REF0 REF1      a 2-member step failing
//   test-batch-q35-cuda FILE CTX STEPS fault-rollback REF0 REF1   the same, its rollback copy failing too
//
// Identity: the reference is each sequence decoded alone (model_forward) on a
// fixed token schedule. The batched runs take pools of 2, 4, 6, 7 and 8
// sequences; each round takes a rotating window of 1 to w ready members, so
// every width from 1 to w occurs and members leave and join in the same round.
// Member 6 has no prompt (its first decoded token is at position 0), and
// member 0 is reused: after its steps its slot is prefilled with another
// prompt and decodes again. Anti-vacuity, per member and per round: a round of
// one member, or one with a member at position 0, must have run the solo
// forward (route_runs[0], RP_SOLO), and every other round one microbatch step
// (route_runs[4], RP_MICRO: counted only when the device ran the step for its
// members). Every reference state must change at every step. RUNNER_BATCH_
// QWEN35=0 (read when a batch is created) must decline every batch, with the
// same bytes.
//
// Faults, with RUNNER_CUDA_INJECT_FAILURE (one injected failure a process,
// after the forward changed the recurrent state) and RUNNER_TEST_Q35_ROLLBACK_
// FAIL=N (the Nth rollback copy of a member's state copies the conv history,
// then fails; N=2 is the second member, after the first was rescued): a fault-ref process decodes one member alone with its first decode
// step failing, which the solo protocol recovers on the CPU from the
// pre-forward snapshot; the batch modes must give each member the same logits
// and state for that step and the two after it, or (fault-rollback) fail the
// step without serving logits and without stepping any member again.
// Without a CUDA build, or with no CUDA device, it prints "test-cuda-batch-q35:
// ok (skipped)" and exits 0, as the other device gates do (test-prefill-bits);
// scripts/cuda-gate.sh fails a step whose log says it skipped for want of the
// device. A device that is up with a kernel module that did not load fails
// (gate_no_device). The device-only code is compiled only in a CUDA build.

#include "runner.h"
#include "gate_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef RUNNER_GPU_CUDA
enum { NS = 8, MAX_STEPS = 512, FAULT_STEPS = 3, RP_SOLO_IDX = 0, RP_MICRO_IDX = 4 };
static int g_fail = 0;
static bool g_skip = false;   // no CUDA device took the file

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

static const int32_t PROMPT[NS][6] = { { 5, 9, 12 }, { 7, 3, 20, 11, 6 }, { 4, 4, 8, 15 }, { 2, 30, 17, 9, 1, 13 },
                                        { 6, 14, 3, 22 }, { 9, 1, 27, 5, 18, 2 }, { 0 }, { 16, 8, 4, 29, 7 } };
static const int PLEN[NS] = { 3, 5, 4, 6, 4, 6, 0, 5 };   // member 6 decodes from position 0
static const int32_t PROMPT2[4] = { 21, 3, 17, 8 };       // member 0's second request, in its reused slot
enum { PLEN2 = 4 };

typedef struct { uint64_t a, b; } h128;

// two FNV-style lanes over 64-bit words (float arrays; a byte tail alone): any
// byte that differs changes both lanes. Word-wise because the real file's
// state is about 150 MB a sequence a step.
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
static bool heq(h128 x, h128 y) { return x.a == y.a && x.b == y.b; }

// the sequence's recurrent state, every recurrent layer: the device's while it
// has one (downloaded), else the host's (after a release, the CPU's live fold)
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

static bool prefill_one(model_t *m, int i, int nv) {
    if (PLEN[i] == 0) return true;
    int32_t t[6];
    for (int j = 0; j < PLEN[i]; j++) t[j] = PROMPT[i][j] % nv;
    return model_forward_batch(m, t, PLEN[i], 0, true) != NULL;
}

static bool prefill2(model_t *m, int nv) {
    int32_t t[PLEN2];
    for (int j = 0; j < PLEN2; j++) t[j] = PROMPT2[j] % nv;
    return model_forward_batch(m, t, PLEN2, 0, true) != NULL;
}

static bool load(model_t *m, int n, const char *path, int ctx) {
    for (int i = 0; i < n; i++) {
        model_params p;
        memset(&p, 0, sizeof p);
        p.gpu_mode = GPU_AUTO;
        p.n_ctx = ctx;
        if (!model_load(&m[i], path, &p)) { fprintf(stderr, "cannot load %s\n", path); return false; }
        if (!m[i].gpu && gate_no_device("test-cuda-batch-q35")) return false;
        if (!m[i].gpu) {
            fprintf(stderr, "test-cuda-batch-q35: skipped (no CUDA device took %s)\n", path);
            g_skip = true;
            return false;
        }
    }
    if (!m[0].qwen35) { fprintf(stderr, "test-cuda-batch-q35: %s is not a qwen35 file\n", path); return false; }
    return true;
}

static h128 ref_lg[NS][MAX_STEPS], ref_st[NS][MAX_STEPS], ref2_lg[MAX_STEPS], ref2_st[MAX_STEPS];

static int identity(const char *path, int ctx, int steps) {
    const char *ks = getenv("RUNNER_BATCH_QWEN35");
    bool killed = ks && !strcmp(ks, "0");
    int steps2 = steps / 4 > 2 ? steps / 4 : 2;
    model_t m[NS];
    if (!load(m, NS, path, ctx)) return g_skip ? 0 : 1;
    int nv = m[0].n_vocab;

    // ---- the reference: each sequence alone, its tokens on a fixed schedule;
    // member 0 then serves a second request in the same slot
    for (int i = 0; i < NS; i++) {
        if (!prefill_one(&m[i], i, nv)) { ck(0, "reference prefill"); return 1; }
        for (int k = 0; k < steps; k++) {
            float *lg = model_forward(&m[i], tok_at(i, k, nv), PLEN[i] + k);
            if (!lg) { ck(0, "reference forward"); return 1; }
            ref_lg[i][k] = hash_bytes(h0(), lg, sizeof(float) * (size_t)nv);
            if (!state_hash(&m[i], &ref_st[i][k])) { ck(0, "reference state download"); return 1; }
        }
    }
    if (!prefill2(&m[0], nv)) { ck(0, "reference second prefill"); return 1; }
    for (int k = 0; k < steps2; k++) {
        float *lg = model_forward(&m[0], tok_at(NS, k, nv), PLEN2 + k);
        if (!lg) { ck(0, "reference forward"); return 1; }
        ref2_lg[k] = hash_bytes(h0(), lg, sizeof(float) * (size_t)nv);
        if (!state_hash(&m[0], &ref2_st[k])) { ck(0, "reference state download"); return 1; }
    }
    int still = 0;
    for (int i = 0; i < NS; i++)
        for (int k = 1; k < steps; k++) still += heq(ref_st[i][k], ref_st[i][k - 1]);
    ck(still == 0, "the device's recurrent state changes at every reference step");

    int widths[] = { 2, 4, 6, 7, 8 };
    for (size_t wi = 0; wi < sizeof widths / sizeof *widths; wi++) {
        int w = widths[wi];
        for (int i = 0; i < w; i++)
            if (!prefill_one(&m[i], i, nv)) { ck(0, "batched prefill"); return 1; }
        model_t *seqs[NS];
        for (int i = 0; i < w; i++) seqs[i] = &m[i];
        model_batch *mb = model_batch_create(seqs, w);
        char what[256];
        snprintf(what, sizeof what, "%s, pool %d: the microbatch %s", path, w,
                 killed ? "does not engage (RUNNER_BATCH_QWEN35=0)" : "engages on the device");
        ck(mb && model_batch_engaged(mb) == !killed, what);
        // k[i]: steps done in the member's current life; life[i]: 0 first request, 1 member 0's second
        int k[NS] = { 0 }, life[NS] = { 0 }, lg_diff = 0, st_diff = 0, first = -1, rounds = 0, cols = 0;
        int seen_w[NS + 1] = { 0 }, pos0_rounds = 0, acct = 0, acct_first = -1;
        bool ok = mb != NULL;
        for (int r = 0; ok; r++) {
            // the members with work left; member 0 starts its second request in
            // its reused slot (a fresh prefill) the round after its first ends
            int cand[NS], nc = 0;
            for (int j = 0; j < w; j++) {
                int i = (j + r) % w;
                if (i == 0 && life[0] == 0 && k[0] == steps) {
                    if (!prefill2(&m[0], nv)) { ok = false; break; }
                    life[0] = 1; k[0] = 0;
                }
                int lim = life[i] ? steps2 : steps;
                if (k[i] < lim) cand[nc++] = i;
            }
            if (!ok || !nc) break;
            int n = 1 + r % w;
            if (n > nc) n = nc;
            int idx[NS], pos[NS];
            int32_t tok[NS];
            float *out[NS] = { NULL };
            bool has0 = false;
            for (int c = 0; c < n; c++) {
                int i = cand[c];
                idx[c] = i;
                pos[c] = life[i] ? PLEN2 + k[i] : PLEN[i] + k[i];
                tok[c] = life[i] ? tok_at(NS, k[i], nv) : tok_at(i, k[i], nv);
                has0 |= pos[c] == 0;
            }
            // each round's route counts, per member: a member in the round ran
            // exactly one forward, the solo one (one member, a position-0
            // member, or the kill switch) or one microbatch step; every
            // other member ran none
            uint32_t solo0[NS], micro0[NS];
            for (int i = 0; i < w; i++) { solo0[i] = m[i].route_runs[RP_SOLO_IDX]; micro0[i] = m[i].route_runs[RP_MICRO_IDX]; }
            if (!model_batch_decode(mb, idx, tok, pos, n, out)) { ok = false; break; }
            rounds++; cols += n; seen_w[n]++; pos0_rounds += has0;
            bool solo_round = killed || n == 1 || has0;
            for (int i = 0; i < w; i++) {
                bool in = false;
                for (int c = 0; c < n; c++) in |= idx[c] == i;
                int ws = in && solo_round, wm = in && !solo_round;
                if ((int)(m[i].route_runs[RP_SOLO_IDX] - solo0[i]) != ws ||
                    (int)(m[i].route_runs[RP_MICRO_IDX] - micro0[i]) != wm) {
                    acct++;
                    if (acct_first < 0) acct_first = r;
                }
            }
            for (int c = 0; c < n; c++) {
                int i = idx[c];
                if (!out[c]) { ok = false; break; }
                h128 hl = hash_bytes(h0(), out[c], sizeof(float) * (size_t)nv), hs;
                if (!state_hash(&m[i], &hs)) { ok = false; break; }
                h128 rl = life[i] ? ref2_lg[k[i]] : ref_lg[i][k[i]];
                h128 rs = life[i] ? ref2_st[k[i]] : ref_st[i][k[i]];
                if (!heq(hl, rl)) { lg_diff++; if (first < 0) first = r; }
                if (!heq(hs, rs)) { st_diff++; if (first < 0) first = r; }
                k[i]++;
            }
        }
        if (acct) fprintf(stderr, "%s, pool %d: %d member-rounds with route counts not as expected (first round %d)\n",
                          path, w, acct, acct_first);
        int expect_cols = w * steps + steps2;
        char ws[64] = "";
        for (int x = 1; x <= w; x++) {
            char t[8];
            snprintf(t, sizeof t, "%s%d", x > 1 ? "," : "", seen_w[x]);
            strcat(ws, t);
        }
        fprintf(stderr, "%s, pool %d: %d rounds (of widths 1..%d: %s), %d with a position-0 member, %d sequence steps; "
                "logits differ at %d, state at %d (first round %d)\n",
                path, w, rounds, w, ws, pos0_rounds, cols, lg_diff, st_diff, first);
        int every_w = 1;
        for (int x = 1; x <= w; x++) every_w &= seen_w[x] > 0;
        snprintf(what, sizeof what, "%s, pool %d: every width from 1 to %d ran", path, w, w);
        ck(ok && every_w, what);
        snprintf(what, sizeof what, "%s, pool %d: every batched step ran (both of member 0's requests)", path, w);
        ck(ok && cols == expect_cols, what);
        snprintf(what, sizeof what, "%s, pool %d: %s", path, w, killed
                 ? "every step ran the solo forward (the kill switch)"
                 : "in every round, each member of a round of 2 or more at positions past 0 ran one microbatch "
                   "step, of any other round one solo forward, and no other member ran");
        ck(ok && acct == 0, what);
        snprintf(what, sizeof what, "%s, pool %d: every step's logits are the solo decode's bits", path, w);
        ck(ok && lg_diff == 0, what);
        snprintf(what, sizeof what, "%s, pool %d: every step's recurrent state is the solo decode's bits", path, w);
        ck(ok && st_diff == 0, what);
        if (mb) model_batch_free(mb);
    }
    for (int i = 0; i < NS; i++) model_free(&m[i]);
    return g_fail;
}

// ---- faults: the reference is the solo protocol under the same failure
static bool write_ref(const char *out, h128 lg[], h128 st[]) {
    FILE *f = fopen(out, "w");
    if (!f) return false;
    for (int s = 0; s < FAULT_STEPS; s++)
        fprintf(f, "%d %016llx%016llx %016llx%016llx\n", s, (unsigned long long)lg[s].a, (unsigned long long)lg[s].b,
                (unsigned long long)st[s].a, (unsigned long long)st[s].b);
    return fclose(f) == 0;
}

static bool read_ref(const char *in, h128 lg[], h128 st[]) {
    FILE *f = fopen(in, "r");
    if (!f) return false;
    for (int s = 0; s < FAULT_STEPS; s++) {
        int si;
        unsigned long long a, b, c, d;
        if (fscanf(f, "%d %16llx%16llx %16llx%16llx", &si, &a, &b, &c, &d) != 5 || si != s) { fclose(f); return false; }
        lg[s].a = a; lg[s].b = b; st[s].a = c; st[s].b = d;
    }
    fclose(f);
    return true;
}

static int fault_ref(const char *path, int ctx, int K, const char *out) {
    model_t m[1];
    // member K's prompt in a one-model process: the first device forward is its
    // prefill, the second its first decode step (RUNNER_CUDA_INJECT_FAILURE=2)
    if (!load(m, 1, path, ctx)) return g_skip ? 0 : 1;
    int nv = m[0].n_vocab;
    h128 lg[FAULT_STEPS], st[FAULT_STEPS];
    if (!prefill_one(&m[0], K, nv)) { ck(0, "fault-ref prefill"); return 1; }
    for (int s = 0; s < FAULT_STEPS; s++) {
        float *l = model_forward(&m[0], tok_at(K, s, nv), PLEN[K] + s);
        if (!l || !state_hash(&m[0], &st[s])) { ck(0, "fault-ref forward"); return 1; }
        lg[s] = hash_bytes(h0(), l, sizeof(float) * (size_t)nv);
        if (s == 0) ck(m[0].gpu == NULL, "fault-ref: the injected failure released the backend at the first decode step");
    }
    ck(write_ref(out, lg, st), "fault-ref: reference written");
    model_free(&m[0]);
    return g_fail;
}

// diagnostic (R3.82's second read, blocker 1): member 0 alone on the device,
// its first decode token stepped twice at the same position, the state a retry
// after an unrecovered failed step produces; prints its hashes
static int double_ref(const char *path, int ctx) {
    model_t m[1];
    if (!load(m, 1, path, ctx)) return g_skip ? 0 : 1;
    int nv = m[0].n_vocab;
    if (!prefill_one(&m[0], 0, nv)) return 1;
    float *l = NULL;
    for (int t = 0; t < 2; t++) l = model_forward(&m[0], tok_at(0, 0, nv), PLEN[0]);
    h128 hs;
    if (!l || !state_hash(&m[0], &hs)) return 1;
    h128 hl = hash_bytes(h0(), l, sizeof(float) * (size_t)nv);
    printf("double-step %016llx%016llx %016llx%016llx\n", (unsigned long long)hl.a, (unsigned long long)hl.b,
           (unsigned long long)hs.a, (unsigned long long)hs.b);
    model_free(&m[0]);
    return 0;
}

static int fault_batch(const char *path, int ctx, const char *mode, const char *ref0, const char *ref1) {
    bool one = !strcmp(mode, "fault-one"), rollback = !strcmp(mode, "fault-rollback");
    int nm = one ? 1 : 2;
    h128 rl[2][FAULT_STEPS], rs[2][FAULT_STEPS];
    // two models: the device forwards are the two prefills, then the failing
    // step (RUNNER_CUDA_INJECT_FAILURE=3); loaded first, so a skip for want of
    // a device does not need the references a skipped fault-ref did not write
    model_t m[2];
    if (!load(m, 2, path, ctx)) return g_skip ? 0 : 1;
    if (!read_ref(ref0, rl[0], rs[0]) || (nm == 2 && !read_ref(ref1, rl[1], rs[1]))) {
        fprintf(stderr, "%s: cannot read the reference\n", mode); return 1;
    }
    int nv = m[0].n_vocab;
    for (int i = 0; i < 2; i++)
        if (!prefill_one(&m[i], i, nv)) { ck(0, "fault prefill"); return 1; }
    model_t *seqs[2] = { &m[0], &m[1] };
    model_batch *mb = model_batch_create(seqs, 2);
    char what[256];
    snprintf(what, sizeof what, "%s: the microbatch engages on the device", mode);
    ck(mb && model_batch_engaged(mb), what);
    if (!mb) return 1;
    int diff = 0;
    for (int s = 0; s < FAULT_STEPS; s++) {
        int idx[2] = { 0, 1 }, pos[2];
        int32_t tok[2];
        float *out[2] = { NULL, NULL };
        for (int c = 0; c < nm; c++) { pos[c] = PLEN[c] + s; tok[c] = tok_at(c, s, nv); }
        uint32_t before[2][2];
        for (int c = 0; c < 2; c++) {
            before[c][0] = m[c].route_runs[RP_SOLO_IDX]; before[c][1] = m[c].route_runs[RP_MICRO_IDX];
        }
        bool okd = model_batch_decode(mb, idx, tok, pos, nm, out);
        if (rollback && s == 0) {
            int moved = 0;
            for (int c = 0; c < 2; c++)
                moved += (int)(m[c].route_runs[RP_SOLO_IDX] - before[c][0]) +
                         (int)(m[c].route_runs[RP_MICRO_IDX] - before[c][1]);
            ck(!okd, "fault-rollback: a step whose rollback could not be verified fails, serving no logits");
            ck(moved == 0, "fault-rollback: no member was stepped again after the failed rollback");
            if (okd) {
                // what was served instead: compare it with one logical step
                for (int c = 0; c < nm; c++) {
                    h128 hs;
                    h128 hl = hash_bytes(h0(), out[c], sizeof(float) * (size_t)nv);
                    bool sh = state_hash(&m[c], &hs);
                    fprintf(stderr, "fault-rollback: member %d served logits %s one logical step's, state %s\n", c,
                            heq(hl, rl[c][0]) ? "equal to" : "NOT", sh && heq(hs, rs[c][0]) ? "equal" : "NOT equal");
                }
            }
            break;
        }
        snprintf(what, sizeof what, "%s: step %d decoded", mode, s);
        ck(okd, what);
        if (!okd) break;
        for (int c = 0; c < nm; c++) {
            h128 hs;
            h128 hl = hash_bytes(h0(), out[c], sizeof(float) * (size_t)nv);
            if (!state_hash(&m[c], &hs)) { ck(0, "fault state"); break; }
            if (s == 0 && c == 0)
                printf("%s step0 member0 %016llx%016llx %016llx%016llx\n", mode, (unsigned long long)hl.a,
                       (unsigned long long)hl.b, (unsigned long long)hs.a, (unsigned long long)hs.b);
            if (!heq(hl, rl[c][s]) || !heq(hs, rs[c][s])) {
                diff++;
                fprintf(stderr, "%s: member %d step %d: logits %s, state %s the solo protocol's\n", mode, c, s,
                        heq(hl, rl[c][s]) ? "equal to" : "NOT", heq(hs, rs[c][s]) ? "equal to" : "NOT");
            }
        }
        if (s == 0) {
            snprintf(what, sizeof what, "%s: the injected failure released %s to the CPU, as the solo protocol does",
                     mode, one ? "member 0" : "both members");
            ck(m[0].gpu == NULL && (one || m[1].gpu == NULL), what);
            if (!one) {
                // R3.82 (PR 965's re-read): the batch outlives its members'
                // backends here. It holds its own reference to their upload,
                // and gives its device side back once its lead has gone, so
                // nothing of the device's is left allocated (gpu_live_buffers:
                // every allocation the engine holds); a batch that borrowed the
                // upload kept its buffers in a released context
                long live = gpu_live_buffers();
                snprintf(what, sizeof what, "%s: with both members on the CPU, the batch holds no device "
                         "allocation (%ld live)", mode, live);
                ck(live == 0, what);
            }
        }
    }
    if (!rollback) {
        snprintf(what, sizeof what, "%s: every member got one logical step: logits and state equal the solo "
                 "protocol's under the same failure, for %d steps", mode, FAULT_STEPS);
        ck(diff == 0, what);
    }
    // teardown after the failure: the batch is freed after (fault-multi) or
    // before (fault-one, fault-rollback) the last member's backend release,
    // and every device allocation comes back
    model_batch_free(mb);
    for (int i = 0; i < 2; i++) model_free(&m[i]);
    long live = gpu_live_buffers();
    snprintf(what, sizeof what, "%s: after the batch and both models are freed no device allocation is left "
             "(%ld live)", mode, live);
    ck(live == 0, what);
    return g_fail;
}
#endif // RUNNER_GPU_CUDA

int main(int argc, char **argv) {
#ifndef RUNNER_GPU_CUDA
    (void)argc; (void)argv;
    fprintf(stderr, "test-cuda-batch-q35: skipped (not a CUDA build)\n");
    printf("test-cuda-batch-q35: ok (skipped)\n");
    return 0;
#else
    if (argc < 3) { fprintf(stderr, "usage: see tests/test_cuda_batch_q35.c\n"); return 2; }
    const char *path = argv[1];
    int ctx = atoi(argv[2]);
    int steps = argc > 3 ? atoi(argv[3]) : 24;
    if (steps < 1 || steps > MAX_STEPS) { fprintf(stderr, "STEPS 1 to %d\n", MAX_STEPS); return 2; }
    const char *mode = argc > 4 ? argv[4] : "identity";
    int rc;
    if (!strcmp(mode, "identity")) rc = identity(path, ctx, steps);
    else if (!strcmp(mode, "fault-ref") && argc > 6) rc = fault_ref(path, ctx, atoi(argv[5]), argv[6]);
    else if (!strcmp(mode, "double-ref")) rc = double_ref(path, ctx);
    else if (!strcmp(mode, "fault-one") && argc > 5) rc = fault_batch(path, ctx, mode, argv[5], NULL);
    else if ((!strcmp(mode, "fault-multi") || !strcmp(mode, "fault-rollback")) && argc > 6)
        rc = fault_batch(path, ctx, mode, argv[5], argv[6]);
    else { fprintf(stderr, "unknown mode %s\n", mode); return 2; }
    fprintf(stderr, rc ? "test-cuda-batch-q35: FAILED\n" : g_skip ? "test-cuda-batch-q35: ok (skipped)\n"
                                                         : "test-cuda-batch-q35: ok\n");
    return rc;
#endif
}
