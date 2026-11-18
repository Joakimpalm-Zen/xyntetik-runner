// R6.91: a NaN logit gets no mass in the sampler, in every build.
//
// A NaN logit is a defect upstream, and cand_cmp already ranks one below every
// real value, so it sorts last. But the softmax took expf(NaN - mx) for it: a
// NaN, so the sum and every share turned NaN, the roulette walk never reached
// its draw, and the last candidate, the NaN token, was picked. That was a
// strict build (T3's float flags) at every RNG state; under -ffast-math the
// same code gave other answers. This test is compiled three ways by
// `make test-sample-nan`: the default flags, a portable build, and T3's strict
// flags. Each run checks:
//   - NaN cases: no pick is a NaN token, over 1000 RNG states each: [0, NaN]
//     and [NaN, 0] with top_k 2, a NaN among three, the full-sort path
//     (top_k 0), and the head fast path (a 5000-token vocabulary, no filter);
//   - the control: a digest of the picks over a fixed corpus of finite logits
//     (several settings, the head path included). make compares it across the
//     three builds and with the default build's digest at main (6e035a77,
//     before this fix), so the finite picks do not move by one bit.
#include "runner.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef RUNNER_T3_BUILD
#define BUILD_KIND "strict"
#elif defined(RUNNER_PORTABLE_MATH)
#define BUILD_KIND "portable"
#else
#define BUILD_KIND "default"
#endif

static float qnan(void) {
    uint32_t b = 0x7fc12345u;
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

static sampler make(float temp, int top_k, float top_p, float min_p, uint64_t rng) {
    sampler s;
    memset(&s, 0, sizeof s);
    s.temp = temp; s.top_k = top_k; s.top_p = top_p; s.min_p = min_p;
    s.repeat_penalty = 1.0f;
    s.rng = rng;
    return s;
}

static int g_fail = 0;

// every pick over 1000 RNG states, none of them a token whose logit is NaN
static void no_nan_pick(const char *what, const float *logits, int n, int top_k, float top_p) {
    float *l = malloc(sizeof(float) * (size_t)n);
    int bad = -1, bad_at = -1;
    for (uint64_t r = 1; r <= 1000 && bad < 0; r++) {
        memcpy(l, logits, sizeof(float) * (size_t)n);
        sampler s = make(1.0f, top_k, top_p, 0.0f, r * 0x9E3779B97F4A7C15ull);
        int t = sample_pick(&s, l, n, NULL, NULL);
        uint32_t b;
        memcpy(&b, &logits[t < 0 || t >= n ? 0 : t], sizeof b);
        if (t < 0 || t >= n || (b & 0x7fffffffu) > 0x7f800000u) { bad = t; bad_at = (int)r; }
    }
    free(l);
    if (bad >= 0 || bad_at >= 0) {
        printf("FAIL %s: %s picked token %d (a NaN logit) at RNG state %d\n", BUILD_KIND, what, bad, bad_at);
        g_fail = 1;
    } else {
        printf("ok %s: %s: no NaN token over 1000 RNG states\n", BUILD_KIND, what);
    }
}

// the sampler's own RNG step (src/sample.c: rng_next, rng_f32), to find the
// state whose draw is the largest float below 1, the draw a walk that sums
// short of 1 by an ULP falls off with
static uint64_t xs(uint64_t x) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; }
static uint64_t state_drawing_the_top(void) {
    for (uint64_t x = 1;; x++)
        if ((xs(x) >> 40) == 0xffffffu) return x;
}

static uint64_t g_rng = 0x243F6A8885A308D3ull;
static float uni(float lo, float hi) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return lo + (hi - lo) * (float)((g_rng >> 40) & 0xffffff) / 16777216.0f;
}

static uint64_t control_digest(void);

int main(int argc, char **argv) {
    // --digest-only: the control alone, for main's sampler (which the NaN cases
    // can crash, R6.91's review)
    if (argc > 1 && !strcmp(argv[1], "--digest-only")) { control_digest(); return 0; }
    float nan = qnan();
    { float l[2] = { 0.0f, nan }; no_nan_pick("[0, NaN], top_k 2", l, 2, 2, 1.0f); }
    { float l[2] = { nan, 0.0f }; no_nan_pick("[NaN, 0], top_k 2", l, 2, 2, 1.0f); }
    { float l[3] = { 0.5f, nan, -0.25f }; no_nan_pick("[0.5, NaN, -0.25], top_k 3, top_p 0.9", l, 3, 3, 0.9f); }
    { float l[4] = { 1.0f, nan, 0.0f, 2.0f }; no_nan_pick("four logits, one NaN, top_k 0", l, 4, 0, 1.0f); }
    {
        enum { V = 5000 };
        float *l = malloc(sizeof(float) * V);
        for (int i = 0; i < V; i++) l[i] = uni(-4.0f, 4.0f);
        l[17] = nan;
        l[4321] = nan;
        no_nan_pick("a 5000-token vocabulary with two NaN, no filter (the head path)", l, V, 0, 1.0f);
        free(l);
    }

    // R6.91 review, P2: a walk whose shares sum short of 1 by an ULP falls off the
    // end; it must land on the last candidate with mass, not a trailing NaN
    {
        uint64_t top = state_drawing_the_top();
        enum { N = 101 };
        float l[N], w[N];
        int bad = -1, tried = 0;
        for (int v = 0; v < 400 && bad < 0; v++) {
            for (int i = 0; i < N - 1; i++) l[i] = uni(-3.0f, 3.0f);
            l[N - 1] = nan;
            memcpy(w, l, sizeof w);
            sampler s = make(1.0f, N, 1.0f, 0.0f, top);
            int t = sample_pick(&s, w, N, NULL, NULL);
            tried++;
            if (t == N - 1) bad = v;
        }
        if (bad >= 0) { printf("FAIL %s: a walk that fell off the end picked the trailing NaN (vector %d)\n", BUILD_KIND, bad); g_fail = 1; }
        else printf("ok %s: %d walks drawn at the top of the range: none picked the trailing NaN\n", BUILD_KIND, tried);
    }

    // R6.91 review, P1: every logit NaN on the head path (a vocabulary of 4096 or
    // more) with top-p or min-p on, and a NaN at index 0 there: no candidate
    // set of size 0 may reach the walk, and the all-NaN answer is the lowest id
    {
        enum { V = 5000 };
        float *l = malloc(sizeof(float) * V), *w = malloc(sizeof(float) * V);
        for (int f = 0; f < 2; f++) {
            for (int i = 0; i < V; i++) l[i] = nan;
            for (uint64_t r = 1; r <= 50; r++) {
                memcpy(w, l, sizeof(float) * V);
                sampler s = make(1.0f, 0, f ? 1.0f : 0.95f, f ? 0.01f : 0.0f, r * 0x9E3779B97F4A7C15ull);
                int t = sample_pick(&s, w, V, NULL, NULL);
                if (t != 0) { printf("FAIL %s: all NaN, head path, %s: picked %d, not the lowest id\n", BUILD_KIND,
                                     f ? "min-p" : "top-p", t); g_fail = 1; break; }
            }
        }
        printf("ok %s: all NaN on the head path with top-p and with min-p: the lowest id\n", BUILD_KIND);
        for (int i = 0; i < V; i++) l[i] = uni(-4.0f, 4.0f);
        l[0] = nan;
        no_nan_pick("a 5000-token vocabulary with a NaN at index 0 (the head path)", l, V, 0, 1.0f);
        free(l);
        free(w);
    }
    control_digest();
    return g_fail;
}

// the control: picks over finite logits, as a digest. Its own input stream,
// from a fixed seed, so the cases above do not move it: the seed, less the
// 5000 draws the first head case took when the digest was first recorded
// (da1c250ddb339875 on Apple clang), so the inputs are those exactly.
static uint64_t control_digest(void) {
    g_rng = 0x243F6A8885A308D3ull;
    for (int i = 0; i < 5000; i++) (void)uni(0.0f, 1.0f);
    uint64_t h = 0xcbf29ce484222325ull;
    static const struct { float temp; int top_k; float top_p, min_p; int n; } set[] = {
        { 1.0f, 0, 1.0f, 0.0f, 50 }, { 0.7f, 20, 0.95f, 0.0f, 300 }, { 1.0f, 64, 0.95f, 0.05f, 1000 },
        { 0.6f, 20, 0.95f, 0.0f, 5000 }, { 1.0f, 0, 1.0f, 0.0f, 5000 }, { 1.3f, 5, 1.0f, 0.1f, 40 },
    };
    int picks = 0;
    for (size_t c = 0; c < sizeof set / sizeof *set; c++) {
        int n = set[c].n;
        float *base = malloc(sizeof(float) * (size_t)n), *l = malloc(sizeof(float) * (size_t)n);
        for (int round = 0; round < 40; round++) {
            for (int i = 0; i < n; i++) base[i] = uni(-6.0f, 6.0f) * (round % 3 + 1) * 0.5f;
            for (uint64_t r = 1; r <= 5; r++) {
                memcpy(l, base, sizeof(float) * (size_t)n);
                sampler s = make(set[c].temp, set[c].top_k, set[c].top_p, set[c].min_p,
                                 (uint64_t)(c * 1000 + round * 10) + r);
                int t = sample_pick(&s, l, n, NULL, NULL);
                h ^= (uint64_t)(uint32_t)t; h *= 0x100000001b3ull;
                picks++;
            }
        }
        free(base);
        free(l);
    }
    printf("digest %s: %d finite picks %016llx\n", BUILD_KIND, picks, (unsigned long long)h);
    return h;
}
