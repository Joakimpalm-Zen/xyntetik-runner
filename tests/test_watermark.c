// Tournament watermark (R1.8.1): the pieces the sampler and the detector
// share, against anchors that do not come from watermark.c.
//
//  - SipHash-2-4 against the reference vectors (Aumasson & Bernstein, key
//    00..0f: the empty message, one byte, and the paper's 15-byte example).
//  - The closed-form reweighting against a literal tournament: every
//    assignment of 2^m candidates enumerated for m = 1..3, each match won by
//    the larger g-value and ties split evenly, gives the winner distribution
//    exactly.
//  - Unbiasedness: averaged over many keys, the reweighted distribution is
//    the model's (the property that makes the mark free in expectation).
//  - The detector's null: random token streams scored against a key have
//    |z| small, and a stream whose every token carries g = 1 on all layers is
//    detected.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "watermark.h"

static int fail(const char *what) {
    printf("watermark: FAIL %s\n", what);
    return 1;
}

// enumerate the tournament literally: `m` layers over 2^m slots, each slot
// holding a candidate index; returns into win[] the winner probability
static void brute(int m, int n, const double *p, const uint64_t *g, double *win) {
    int slots = 1 << m;
    long long total = 1;
    for (int i = 0; i < slots; i++) total *= n;
    memset(win, 0, sizeof(double) * (size_t)n);
    int idx[8];
    for (long long a = 0; a < total; a++) {
        long long r = a;
        double prob = 1;
        for (int i = 0; i < slots; i++) { idx[i] = (int)(r % n); r /= n; prob *= p[idx[i]]; }
        // play the layers: each match yields a distribution over winners;
        // track per-slot winner distributions
        double dist[8][8];   // dist[slot][cand]
        for (int i = 0; i < slots; i++) {
            for (int c = 0; c < n; c++) dist[i][c] = 0;
            dist[i][idx[i]] = 1;
        }
        int live = slots;
        for (int l = 0; l < m; l++) {
            for (int i = 0; i < live / 2; i++) {
                double out[8] = {0};
                for (int x = 0; x < n; x++) for (int y = 0; y < n; y++) {
                    double w = dist[2 * i][x] * dist[2 * i + 1][y];
                    if (w == 0) continue;
                    int gx = (int)((g[x] >> l) & 1), gy = (int)((g[y] >> l) & 1);
                    if (gx > gy) out[x] += w;
                    else if (gy > gx) out[y] += w;
                    else { out[x] += w / 2; out[y] += w / 2; }
                }
                memcpy(dist[i], out, sizeof out);
            }
            live /= 2;
        }
        for (int c = 0; c < n; c++) win[c] += prob * dist[0][c];
    }
}

int main(void) {
    // SipHash-2-4 reference vectors, key 00 01 .. 0f
    uint64_t k[2] = { 0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL };
    uint8_t msg[16];
    for (int i = 0; i < 16; i++) msg[i] = (uint8_t)i;
    if (wm_siphash24(k, msg, 0) != 0x726fdb47dd0e0e31ULL) return fail("siphash len 0");
    if (wm_siphash24(k, msg, 1) != 0x74f839c593dc67fdULL) return fail("siphash len 1");
    if (wm_siphash24(k, msg, 15) != 0xa129ca6149be45e5ULL) return fail("siphash len 15");
    printf("watermark: SipHash-2-4 reference vectors ok\n");

    // the closed form against the literal tournament, through wm_reweight's
    // own g-words: find tokens whose g-bits make every pattern appear
    wm_key key;
    memset(&key, 0, sizeof key);
    for (int i = 0; i < 32; i++) key.key[i] = (uint8_t)(7 * i + 1);
    int32_t ctx[4] = { 11, 22, 33, 44 };
    uint64_t sk[2];
    wm_seed(&key, ctx, 4, sk);
    const int n = 4;
    int32_t ids[4] = { 5, 6, 7, 8 };
    uint64_t g[4];
    for (int i = 0; i < n; i++) g[i] = wm_gword(sk, ids[i]);
    const double base[4] = { 0.4, 0.3, 0.2, 0.1 };
    for (int m = 1; m <= 3; m++) {
        double win[4];
        brute(m, n, base, g, win);
        float p[4];
        for (int i = 0; i < n; i++) p[i] = (float)base[i];
        if (!wm_reweight(sk, ids, p, n, m)) return fail("reweight alloc");
        for (int i = 0; i < n; i++)
            if (fabs(p[i] - win[i]) > 1e-6) {
                printf("m=%d cand %d closed %.7f brute %.7f\n", m, i, p[i], win[i]);
                return fail("closed form != literal tournament");
            }
    }
    printf("watermark: closed form = literal tournament for m = 1..3\n");

    // unbiased over keys: E_key[p'] = p at the full depth
    {
        double avg[4] = {0};
        const int keys = 4000;   // p' is nearly one-hot per key: sd of a mean ~0.008
        uint64_t x = 0x243f6a8885a308d3ULL;
        for (int r = 0; r < keys; r++) {
            for (int i = 0; i < 32; i++) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                key.key[i] = (uint8_t)(x >> 24);
            }
            wm_seed(&key, ctx, 4, sk);
            float p[4];
            for (int i = 0; i < n; i++) p[i] = (float)base[i];
            wm_reweight(sk, ids, p, n, WM_LAYERS);
            double s = 0;
            for (int i = 0; i < n; i++) { avg[i] += p[i]; s += p[i]; }
            if (fabs(s - 1) > 1e-4) return fail("reweighted mass is not 1");
        }
        for (int i = 0; i < n; i++)
            if (fabs(avg[i] / keys - base[i]) > 0.03) {
                printf("cand %d mean %.4f want %.4f\n", i, avg[i] / keys, base[i]);
                return fail("reweighting is biased over keys");
            }
        printf("watermark: unbiased over %d keys\n", keys);
    }

    // the detector's null and its power
    {
        for (int i = 0; i < 32; i++) key.key[i] = (uint8_t)(i * 3);
        int32_t toks[2000];
        uint64_t x = 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < 2000; i++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            toks[i] = (int32_t)(x % 50000);
        }
        wm_score s;
        wm_detect(&key, toks, 2000, WM_CONTEXT, &s);
        if (s.scored < 1900 || fabs(s.z) > 4 || strcmp(wm_verdict(&s), "NOT_DETECTED"))
            return fail("random tokens detected");
        printf("watermark: null stream z = %.2f over %d tokens\n", s.z, s.scored);
        // a stream where each token is the candidate with the most g-bits set
        // among 64 choices: a strong mark
        for (int t = WM_CONTEXT; t < 400; t++) {
            wm_seed(&key, toks, t, sk);
            int best = 0, bc = -1;
            for (int c = 0; c < 64; c++) {
                int pc = __builtin_popcountll(wm_gword(sk, c) & ((1ULL << WM_LAYERS) - 1));
                if (pc > bc) { bc = pc; best = c; }
            }
            toks[t] = best;
        }
        wm_detect(&key, toks, 400, WM_CONTEXT, &s);
        if (strcmp(wm_verdict(&s), "WATERMARKED")) return fail("marked stream not detected");
        wm_key other = key;
        other.key[0] ^= 1;
        wm_detect(&other, toks, 400, WM_CONTEXT, &s);
        if (strcmp(wm_verdict(&s), "NOT_DETECTED")) return fail("marked stream detected under another key");
        // a repeated context is scored once
        int32_t rep[64];
        for (int i = 0; i < 64; i++) rep[i] = i % 8;
        wm_detect(&key, rep, 64, WM_CONTEXT, &s);
        if (s.scored != 8) { printf("scored %d\n", s.scored); return fail("repeated contexts scored"); }
        wm_detect(&key, rep, 10, WM_CONTEXT, &s);
        if (strcmp(wm_verdict(&s), "INSUFFICIENT")) return fail("short stream not insufficient");
    }
    // key ids: the id names the key
    {
        char a[17], b[17];
        wm_key_id(key.key, a);
        key.key[31] ^= 1;
        wm_key_id(key.key, b);
        if (strlen(a) != 16 || !strcmp(a, b)) return fail("key id");
    }
    printf("watermark: ok\n");
    return 0;
}
