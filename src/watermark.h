// Tournament-sampling watermark for generated text (R1.8.1, R1.8.2).
//
// The scheme is SynthID-Text's (Dathathri et al., Nature 2024): m layers of
// pairwise tournaments among 2^m candidates drawn from the model's filtered
// distribution, each match won by the candidate whose keyed g-value is
// larger. The exact winner distribution has a closed form -- per layer,
// p(x) <- p(x) * (1 + g_l(x) - sum_y p(y) g_l(y)) -- so the sampler reweights
// its candidates and then draws once, as it always does. Averaged over keys
// the reweighting leaves the distribution unchanged; with the key, the chosen
// tokens' g-values average above one half, which is what the detector
// measures.
//
// g-values. For the token at position t, the context is the H tokens before
// it (fewer at the start of a sequence): seed = SHA-256(key || "xyntetik.wm.
// ctx.v1" || n || the n tokens as int32 little-endian), and the token's
// 64-bit g-word is SipHash-2-4 keyed with the seed's first 16 bytes over the
// token id as 4 little-endian bytes; bit l is g_l. A context that already
// occurred earlier in the same generation is left unmarked, and the detector
// skips it, so repeated text neither loops on one bias nor counts twice.
//
// Off by default; greedy decoding (temp 0) is never changed. The key is a
// per-deployment secret; receipts record only its id, so a mark is
// verifiable (replay with the key reproduces the record) as well as
// detectable.
#ifndef RUNNER_WATERMARK_H
#define RUNNER_WATERMARK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sample.h"

#define WM_LAYERS  30
#define WM_CONTEXT 4
#define WM_SCHEME  "tournament-v1"

typedef struct {
    uint8_t key[32];
    char    id[17];   // 16 hex: the first 8 bytes of SHA-256(domain || key)
} wm_key;

// The key file: {"schema_version":"xyntetik.runner.watermark_key.v1",
// "key":"<64 hex>","key_id":"<16 hex>"}, written 0600 and never overwritten.
bool wm_key_write(const char *path, const uint8_t key[32], char id_out[17]);
// False with the reason in err: unreadable, malformed, or a key_id that does
// not match the key.
bool wm_key_load(const char *path, wm_key *k, char *err, size_t cap);
void wm_key_id(const uint8_t key[32], char id[17]);

uint64_t wm_siphash24(const uint64_t k[2], const uint8_t *m, size_t n);
// The seed for the token at position t of toks (context toks[max(0,t-H)..t)).
void wm_seed(const wm_key *k, const int32_t *toks, int t, uint64_t sk[2]);
uint64_t wm_gword(const uint64_t sk[2], int32_t tok);
// Does position t's context already end at some position in [start, t)?
bool wm_context_repeats(const int32_t *toks, int start, int t);

// Reweight n candidates (ids, probabilities summing to 1) to the exact
// winner distribution of a `layers`-deep tournament. False only on an
// allocation failure.
bool wm_reweight(const uint64_t sk[2], const int32_t *ids, float *p, int n,
                 int layers);

// The engine's per-pick hook (engine.h, wm_prepare) for one generator: a
// slot or the CLI owns one. Before each pick it seeds the key with the
// context and installs the sampler's reweight, or leaves the pick unmarked
// when the context repeats one of this generation's. `marked` counts the
// picks the tournament actually reweighted (sampled ones: greedy decoding
// never calls it).
typedef struct {
    const wm_key *key;
    uint64_t sk[2];
    int marked, repeats;
} wm_state;
void wm_prepare(void *ud, sampler *s, const int32_t *hist, int start, int t);

typedef struct {
    int    tokens, scored;   // positions in [start, n), and those scored
    long long g_sum, g_n;    // ones among, and number of, the g-values scored
    double mean, z, p_value; // mean g; z = (sum - n/2)/sqrt(n/4); one-sided p
} wm_score;

// Score positions [start, n) of toks against the key.
void wm_detect(const wm_key *k, const int32_t *toks, int n, int start,
               wm_score *out);
// "WATERMARKED" (z >= WM_Z_DETECT), "NOT_DETECTED", or "INSUFFICIENT"
// (fewer than WM_MIN_SCORED positions scored)
#define WM_Z_DETECT   4.0
#define WM_MIN_SCORED 16
const char *wm_verdict(const wm_score *s);

#endif
