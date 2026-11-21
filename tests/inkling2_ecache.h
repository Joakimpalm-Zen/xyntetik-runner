// R4.48: a DECODED-EXPERT CACHE for ink2_weights.get_expert (test-side). It sits between the engine and a provider
// that can decode one expert into a buffer (the real checkpoint's provider, tests/inkling2_provider.c, or the
// fixture's float32 stacks): the engine sees an ordinary ink2_weights and does not know whether it is cached.
//
//   - Keyed by (physical layer, gate_up_proj or down_proj, expert), from the stack name
//     model.layers.<b>.mlp.experts.{gate_up_proj,down_proj}. Other names pass through uncached.
//   - An LRU bounded by BYTES (each entry its own anonymous mapping, page-rounded, counted as mapped) and, if asked,
//     by an entry count. The bound always holds: the two results returned last are never evicted (the pieces hold at
//     most two expert results, inkling2.h), and when nothing else can make room the expert is decoded into one of two
//     bypass buffers instead of the cache (counted as a bypass).
//   - VERIFIED: a hit is compared, by memcmp, with a fresh decode into a scratch buffer, on every hit in debug mode and
//     on every Nth hit otherwise; any difference prints the key and aborts.
//
// Environment (ink2_ec_cfg_env): INK2_ECACHE_BYTES (the byte budget; suffixes K, M, G as powers of 1024),
// INK2_ECACHE_ENTRIES (an entry-count bound), either one enabling the cache; INK2_ECACHE_CHECK = "all" (debug: every
// hit) or N (every Nth hit; default 64; 0 = no checks).
#ifndef INKLING2_ECACHE_H
#define INKLING2_ECACHE_H

#include "inkling2.h"
#include <stddef.h>
#include <stdint.h>

// decode expert e of stack into dst ([rows][cols] float32), or with dst NULL only report rows and cols; 0 or -1
typedef int (*ink2_ec_fill_fn)(void *fctx, const char *stack, int e, float *dst, int64_t *rows, int64_t *cols);

typedef struct {
    int enabled;
    size_t budget_bytes;   // 0: no byte bound (only the entry bound)
    int max_entries;       // 0: no count bound
    int check_every;       // 0 none, 1 every hit (debug), N every Nth hit
} ink2_ec_cfg;
ink2_ec_cfg ink2_ec_cfg_env(void);

typedef struct ink2_ecache ink2_ecache;
// inner: the provider's weights (its get serves every non-expert lookup; its get_expert is used only for names the
// cache does not key)
ink2_ecache *ink2_ec_new(const ink2_weights *inner, ink2_ec_fill_fn fill, void *fctx, ink2_ec_cfg cfg);
void ink2_ec_free(ink2_ecache *c);
void ink2_ec_weights(ink2_ecache *c, ink2_weights *w);   // what the engine is given
void ink2_ec_clear(ink2_ecache *c);                      // drop every entry (after the weights under it change)

typedef struct {
    uint64_t lookups, hits, misses, inserts, evictions, bypasses, uncached, checks, clears;
    int entries, peak_entries;
    size_t entry_bytes, peak_entry_bytes;   // the entries' mappings (page-rounded), now and at their largest
    size_t aux_bytes;                       // bypass and scratch buffers, the key table and the entry records
} ink2_ec_stats;
void ink2_ec_get_stats(const ink2_ecache *c, ink2_ec_stats *st);
// the cache's own resident bytes: entry_bytes + aux_bytes (every page of an entry or buffer is written when filled)
size_t ink2_ec_resident(const ink2_ecache *c);
void ink2_ec_print(const ink2_ecache *c, const char *tag, void *file /* FILE * */);

#endif
