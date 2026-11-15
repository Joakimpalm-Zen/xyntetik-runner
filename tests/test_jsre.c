// R2.18: JSON Schema `pattern` as a byte automaton (src/jsre.c).
//
//   1. ECMA-262's verdicts (tests/jsre_cases.h, from node's RegExp) on hand
//      cases and generated ECMA/Python divergences: the state-set automaton
//      and its lazy DFA under three pool budgets, fed and as a trial, must
//      answer each one; patterns it cannot build are refused by name.
//   2. The DFA cache against the automaton it memoizes, stepped as the
//      validator steps it (a fed walk that keeps its states, trials at every
//      position), under an unbounded, a 1-byte and an 8 KiB pool: equal answers,
//      flushes that happened, trial handles refused after one, and a fill from
//      a second thread refused.
//   3. Lengths: the backward sequence against a forward search over real bytes.
//   4. The pool keeps its budget.
//   5. A long pattern compiles on a small stack.
//   6. The kept limit fails by name, also when a trial reached the state first.
//   7. Lengths at INT_MAX, the largest maxLength.
//   8. The heap a pool really holds, counted by the allocator.
//   9. A second thread promoting a cached state is refused.
#include "jsre.h"
#include "jsre_cases.h"

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The automaton is compiled into this test with its allocations counted in
// bytes, so a memory bound is checked against what the heap holds (test 8),
// never against the pool's own figure. JSRE_SOURCE lets a mutant stand in.
static size_t heap_live, heap_high;
static void *h_malloc(size_t n) {
    size_t *p = malloc(n + 16);
    if (!p) return NULL;
    *p = n;
    heap_live += n;
    if (heap_live > heap_high) heap_high = heap_live;
    return (char *)p + 16;
}
static void *h_calloc(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b - 16) return NULL;
    void *q = h_malloc(a * b);
    if (q) memset(q, 0, a * b);
    return q;
}
static void h_free(void *q) {
    if (!q) return;
    size_t *p = (size_t *)((char *)q - 16);
    heap_live -= *p;
    free(p);
}
static void *h_realloc(void *q, size_t n) {
    if (!q) return h_malloc(n);
    size_t *p = (size_t *)((char *)q - 16), old = *p;
    size_t *r = realloc(p, n + 16);
    if (!r) return NULL;
    *r = n;
    heap_live = heap_live - old + n;
    if (heap_live > heap_high) heap_high = heap_live;
    return (char *)r + 16;
}
#define malloc  h_malloc
#define calloc  h_calloc
#define realloc h_realloc
#define free    h_free
#ifndef JSRE_SOURCE
#define JSRE_SOURCE "../src/jsre.c"
#endif
#include JSRE_SOURCE
#undef malloc
#undef calloc
#undef realloc
#undef free

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; if (fails < 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static bool nfa_match(const jsre *re, const char *s, int n) {
    uint64_t *set = malloc(jsre_words(re) * sizeof *set);
    jsre_start(re, set);
    bool alive = true;
    for (int i = 0; i < n && alive; i++) alive = jsre_step(re, set, (uint8_t)s[i]);
    bool r = alive && jsre_accepts(re, set);
    free(set);
    return r;
}

static bool dfa_match(jsre_dfa *d, const char *s, int n, bool keep) {
    jsre_st st = jsre_dfa_start(d);
    for (int i = 0; i < n && st != JSRE_DEAD && st != JSRE_FAIL; i++) st = jsre_dfa_step(d, st, (uint8_t)s[i], keep);
    return st != JSRE_DEAD && st != JSRE_FAIL && jsre_dfa_accepts(d, st);
}

// unbounded; one byte (a flush at nearly every fill); 8 KiB, where states,
// tables and lengths are all cached and flushed (a few states at a time, past
// the books a pool and a DFA hold however small the budget)
static const size_t BUDGETS[] = { (size_t)1 << 40, 1, 8192 };
enum { NB = 3 };

// 1 --------------------------------------------------------------- ECMA cases
static void test_cases(void) {
    int n = (int)(sizeof JSRE_CASES / sizeof *JSRE_CASES), accepted = 0, refused = 0;
    jsre *re = NULL;
    jsre_pool *pool[NB] = { 0 };
    jsre_dfa *dfa[NB] = { 0 };
    const char *cur = NULL;
    for (int i = 0; i < n; i++) {
        const jsre_case *c = &JSRE_CASES[i];
        if (!cur || strcmp(cur, c->pat)) {
            for (int b = 0; b < NB; b++) { jsre_dfa_free(dfa[b]); jsre_pool_free(pool[b]); dfa[b] = NULL; pool[b] = NULL; }
            jsre_free(re);
            char err[256];
            re = jsre_compile(c->pat, err, sizeof err);
            cur = c->pat;
            CHECK(re, "pattern %s refused (%s); ECMA compiles it", c->pat, err);
            if (!re) continue;
            for (int b = 0; b < NB; b++) { pool[b] = jsre_pool_new(BUDGETS[b]); dfa[b] = jsre_dfa_new(re, pool[b]); }
        }
        if (!re) continue;
        bool want = c->want;
        CHECK(nfa_match(re, c->s, c->n) == want, "pattern %s, string #%d: automaton %d, want %d (ECMA %d)", c->pat, i, !want, want, c->ecma);
        for (int b = 0; b < NB; b++) {
            CHECK(dfa_match(dfa[b], c->s, c->n, true) == want, "pattern %s, string #%d, pool %d, fed: want %d", c->pat, i, b, want);
            CHECK(dfa_match(dfa[b], c->s, c->n, false) == want, "pattern %s, string #%d, pool %d, trial: want %d", c->pat, i, b, want);
        }
        accepted += want; refused += !want;
    }
    for (int b = 0; b < NB; b++) { jsre_dfa_free(dfa[b]); jsre_pool_free(pool[b]); }
    jsre_free(re);
    // a gate that accepted or refused everything would pass vacuously
    CHECK(accepted > 100 && refused > 100, "cases accepted %d, refused %d", accepted, refused);
    int nr = (int)(sizeof JSRE_REFUSALS / sizeof *JSRE_REFUSALS);
    for (int i = 0; i < nr; i++) {
        char err[256] = "";
        jsre *r = jsre_compile(JSRE_REFUSALS[i].pat, err, sizeof err);
        CHECK(!r, "pattern %s compiled; it must be refused (%s)", JSRE_REFUSALS[i].pat, JSRE_REFUSALS[i].why);
        CHECK(r || strstr(err, JSRE_REFUSALS[i].why), "pattern %s refused as \"%s\", want \"%s\"", JSRE_REFUSALS[i].pat, err, JSRE_REFUSALS[i].why);
        jsre_free(r);
    }
    printf("ecma cases: %d strings (%d accepted, %d refused), %d patterns refused by name\n", n, accepted, refused, nr);
}

// 2 --------------------------------------------------------------- the cache
typedef struct { jsre_dfa *d; jsre_pool *p; jsre_st st; } arm;
static long steps, lens, bad, stale_asked, stale_refused;

static void compare(const jsre *re, const uint64_t *set, bool alive, arm *a, const char *what, const char *pat) {
    for (int k = 0; k < NB; k++) {
        steps++;
        if (a[k].st == JSRE_FAIL) { jsre_pool_stat s; jsre_pool_stats(a[k].p, &s); bad++; CHECK(0, "%s: pool %d failed on %s: %s", pat, k, what, s.failed); continue; }
        bool dal = a[k].st != JSRE_DEAD;
        if (dal != alive || (alive && jsre_dfa_accepts(a[k].d, a[k].st) != jsre_accepts(re, set))) { bad++; CHECK(0, "%s: pool %d disagrees on %s", pat, k, what); }
        if (alive && dal) {
            int lo = (int)(rnd() % 16), hi = rnd() % 4 == 0 ? -1 : lo + (int)(rnd() % 16);
            lens++;
            if (jsre_dfa_len_ok(a[k].d, a[k].st, lo, hi) != jsre_len_ok(re, set, lo, hi)) { bad++; CHECK(0, "%s: pool %d lengths [%d, %d] disagree", pat, k, lo, hi); }
        }
    }
}

static uint8_t pick(const jsre *re, const uint64_t *set, uint64_t *tmp, const char *pat) {
    int np = (int)strlen(pat);
    if (rnd() % 10 < 8)
        for (int t = 0; t < 24; t++) {
            uint8_t b = rnd() & 1 && np ? (uint8_t)pat[rnd() % (uint64_t)np] : (uint8_t)(rnd() % 3 ? 0x20 + rnd() % 0x5f : rnd() % 256);
            memcpy(tmp, set, jsre_words(re) * sizeof *tmp);
            if (jsre_step(re, tmp, b)) return b;
        }
    return (uint8_t)(rnd() % 256);
}

typedef struct { jsre_dfa *d; jsre_st st; bool refused; } thread_arg;
static void *second_thread(void *v) {
    thread_arg *t = v;
    jsre_st r = JSRE_DEAD;
    for (int b = 0; b < 256 && r != JSRE_FAIL; b++) r = jsre_dfa_step(t->d, t->st, (uint8_t)b, false);
    t->refused = r == JSRE_FAIL;
    return NULL;
}

static void test_cache(void) {
    int n = (int)(sizeof JSRE_CASES / sizeof *JSRE_CASES), npat = 0, thread_refused = 0;
    uint64_t flushes[NB] = { 0 };
    const char *cur = NULL;
    for (int i = 0; i < n; i++) {
        if (cur && !strcmp(cur, JSRE_CASES[i].pat)) continue;
        cur = JSRE_CASES[i].pat;
        char err[256];
        jsre *re = jsre_compile(cur, err, sizeof err);
        if (!re) continue;
        CHECK(jsre_lengths(re, -1, err, sizeof err), "%s: lengths refused: %s", cur, err);
        npat++;
        size_t w = jsre_words(re);
        uint64_t *set = malloc(w * 8), *ts = malloc(w * 8), *tmp = malloc(w * 8);
        arm a[NB];
        size_t books0[NB];
        for (int k = 0; k < NB; k++) {
            a[k].p = jsre_pool_new(BUDGETS[k]); a[k].d = jsre_dfa_new(re, a[k].p);
            jsre_pool_stat s0; jsre_pool_stats(a[k].p, &s0); books0[k] = s0.books;   // what no flush can shrink
        }
        for (int walk = 0; walk < 10; walk++) {
            jsre_start(re, set);
            bool alive = true;
            for (int k = 0; k < NB; k++) a[k].st = jsre_dfa_start(a[k].d);
            compare(re, set, alive, a, "the start", cur);
            for (int pos = 0; pos < 40 && alive; pos++) {
                for (int t = 0; t < 6; t++) {
                    memcpy(ts, set, w * 8);
                    arm b[NB];
                    memcpy(b, a, sizeof b);
                    bool tal = true;
                    for (int j = 0, tl = 1 + (int)(rnd() % 5); j < tl && tal; j++) {
                        uint8_t c = pick(re, ts, tmp, cur);
                        tal = jsre_step(re, ts, c);
                        for (int k = 0; k < NB; k++)
                            if (b[k].st != JSRE_DEAD && b[k].st != JSRE_FAIL) b[k].st = jsre_dfa_step(b[k].d, b[k].st, c, false);
                        compare(re, ts, tal, b, "a trial", cur);
                    }
                    // a trial's handle once the trial is over and the 1-byte pool
                    // has flushed: refused, never read as what the slot holds now
                    // (now and then: it ends the walk)
                    if (t == 0 && walk % 3 == 0 && pos % 9 == 4 && b[1].st != JSRE_DEAD && b[1].st != JSRE_FAIL) {
                        jsre_pool_stat s0, s1;
                        jsre_pool_stats(a[1].p, &s0);
                        for (int c = 0; c < 256; c++) {
                            jsre_pool_stats(a[1].p, &s1);
                            if (s1.flushes > s0.flushes) break;
                            (void)jsre_dfa_step(a[1].d, a[1].st, (uint8_t)c, false);
                        }
                        jsre_pool_stats(a[1].p, &s1);
                        if (s1.flushes > s0.flushes) {
                            stale_asked++;
                            jsre_st r = jsre_dfa_step(a[1].d, b[1].st, 'a', false);
                            jsre_pool_stats(a[1].p, &s1);
                            if (r == JSRE_FAIL && s1.failed && strstr(s1.failed, "stale")) stale_refused++;
                            // that refusal is recorded in the pool: start the arm afresh
                            flushes[1] += s1.flushes;
                            jsre_dfa_free(a[1].d); jsre_pool_free(a[1].p);
                            a[1].p = jsre_pool_new(BUDGETS[1]); a[1].d = jsre_dfa_new(re, a[1].p);
                            alive = false;   // restart the walk
                            break;
                        }
                    }
                }
                if (!alive) break;
                uint8_t c = pick(re, set, tmp, cur);
                alive = jsre_step(re, set, c);
                for (int k = 0; k < NB; k++)
                    if (a[k].st != JSRE_DEAD && a[k].st != JSRE_FAIL) a[k].st = jsre_dfa_step(a[k].d, a[k].st, c, true);
                compare(re, set, alive, a, "a fed byte", cur);
            }
        }
        for (int k = 0; k < NB; k++) {
            jsre_pool_stat s;
            jsre_pool_stats(a[k].p, &s);
            flushes[k] += s.flushes;
            // 4: the budget holds, give or take what one step needs and the
            // books a pool and a DFA hold however small the budget
            if (k == 2) {
                size_t slack = 4 * (w * 8 + 64) + 1024 + 4096;
                CHECK(s.high <= BUDGETS[k] + s.kept + books0[k] + slack, "%s: an 8 KiB pool held %zu bytes (books at start %zu)", cur, s.high, books0[k]);
            }
            jsre_dfa_free(a[k].d); jsre_pool_free(a[k].p);
        }
        // a fill from a second thread
        jsre_pool *p = jsre_pool_new(BUDGETS[0]);
        thread_arg t = { jsre_dfa_new(re, p), 0, false };
        t.st = jsre_dfa_start(t.d);
        pthread_t th;
        CHECK(pthread_create(&th, NULL, second_thread, &t) == 0, "pthread_create");
        pthread_join(th, NULL);
        thread_refused += t.refused;
        jsre_dfa_free(t.d); jsre_pool_free(p);
        free(set); free(ts); free(tmp); jsre_free(re);
    }
    CHECK(bad == 0, "%ld disagreements", bad);
    // live: flushes happened where the budget is small, never where it is not
    CHECK(flushes[0] == 0 && flushes[1] > 1000 && flushes[2] > 0, "flushes %llu %llu %llu",
          (unsigned long long)flushes[0], (unsigned long long)flushes[1], (unsigned long long)flushes[2]);
    CHECK(stale_asked > 20 && stale_refused == stale_asked, "stale handles refused %ld of %ld", stale_refused, stale_asked);
    CHECK(thread_refused == npat, "second-thread fills refused %d of %d", thread_refused, npat);
    printf("cache: %d patterns, %ld compared steps, %ld length asks, flushes %llu (1 byte) %llu (8 KiB), "
           "stale handles refused %ld of %ld, second-thread fills refused %d of %d\n", npat, steps, lens,
           (unsigned long long)flushes[1], (unsigned long long)flushes[2], stale_refused, stale_asked, thread_refused, npat);
}

// 3 --------------------------------------------------------------- lengths
// The oracle: a forward search over every byte, a byte that starts a code
// point costing one: a completion accepted at its end, of total cost in [a, b].
typedef struct { uint64_t *set; int a, b; bool r; } memo;
static memo *M;
static int nm, cm;
static int fwd(const jsre *re, const uint64_t *set, int a, int b, int *budget) {
    if (b < 0) return 0;
    size_t w = jsre_words(re);
    for (int i = 0; i < nm; i++)
        if (M[i].a == a && M[i].b == b && !memcmp(M[i].set, set, w * 8)) return M[i].r;
    if (--*budget < 0) return -1;
    int r = a <= 0 && jsre_accepts(re, set);
    uint64_t *nx = malloc(w * 8);
    for (int x = 1; x < 256 && !r; x++) {
        memcpy(nx, set, w * 8);
        if (!jsre_step(re, nx, (uint8_t)x)) continue;
        int c = x < 0x80 || x >= 0xC0;
        int q = fwd(re, nx, a - c, b - c, budget);
        if (q < 0) { free(nx); return -1; }
        r |= q;
    }
    free(nx);
    if (nm == cm) { cm = cm ? cm * 2 : 256; M = realloc(M, sizeof *M * (size_t)cm); }
    M[nm].set = malloc(w * 8); memcpy(M[nm].set, set, w * 8);
    M[nm].a = a; M[nm].b = b; M[nm].r = r; nm++;
    return r;
}

static void memo_reset(void) { for (int i = 0; i < nm; i++) free(M[i].set); nm = 0; }

static void test_lengths(void) {
    int n = (int)(sizeof JSRE_CASES / sizeof *JSRE_CASES), asks = 0, yes = 0, wrong = 0;
    const char *cur = NULL;
    for (int i = 0; i < n; i++) {
        if (cur && !strcmp(cur, JSRE_CASES[i].pat)) continue;
        cur = JSRE_CASES[i].pat;
        char err[256];
        jsre *re = jsre_compile(cur, err, sizeof err);
        if (!re || jsre_states(re) > 200) { jsre_free(re); continue; }
        for (int mode = 0; mode < 2; mode++) {
            if (!jsre_lengths(re, mode ? -1 : 10, err, sizeof err)) { CHECK(0, "%s: %s", cur, err); continue; }
            size_t w = jsre_words(re);
            uint64_t *set = malloc(w * 8), *tmp = malloc(w * 8);
            jsre_start(re, set);
            for (int pos = 0; pos < 5; pos++) {
                for (int t = 0; t < 3; t++) {
                    int lo = (int)(rnd() % 11), hi = lo + (int)(rnd() % (11 - lo));
                    memo_reset();
                    int budget = 4000, o = fwd(re, set, lo, hi, &budget);
                    if (o < 0) continue;
                    asks++; yes += o;
                    if (jsre_len_ok(re, set, lo, hi) != (bool)o) { wrong++; CHECK(0, "%s: lengths [%d, %d]: sequence %d, search %d", cur, lo, hi, !o, o); }
                }
                bool moved = false;
                for (int t = 0; t < 30 && !moved; t++) {
                    uint8_t b = (uint8_t)(rnd() % 2 ? cur[rnd() % strlen(cur)] : 0x20 + rnd() % 0x5f);
                    memcpy(tmp, set, w * 8);
                    if (jsre_step(re, tmp, b)) { memcpy(set, tmp, w * 8); moved = true; }
                }
                if (!moved) break;
            }
            free(set); free(tmp);
        }
        jsre_free(re);
    }
    memo_reset(); free(M); M = NULL; cm = 0;
    CHECK(asks > 500 && yes > 100 && asks - yes > 100, "length asks %d, yes %d", asks, yes);
    // a gap: ^(ab)+$ has 2 and 4 code points but nothing in [3, 3]; a shortest
    // and a longest alone (2, unbounded) would say yes
    char err[256];
    jsre *re = jsre_compile("^(ab)+$", err, sizeof err);
    CHECK(re && jsre_lengths(re, 8, err, sizeof err), "^(ab)+$");
    if (re) {
        uint64_t *set = malloc(jsre_words(re) * 8);
        jsre_start(re, set);
        CHECK(jsre_len_ok(re, set, 2, 2) && !jsre_len_ok(re, set, 3, 3) && jsre_len_ok(re, set, 4, 4) &&
              jsre_len_ok(re, set, 3, 4) && !jsre_len_ok(re, set, 0, 1), "^(ab)+$ lengths");
        free(set);
    }
    jsre_free(re);
    // lengths that settle only past JSRE_LEN_MAX: refused unbounded, kept under a bound
    re = jsre_compile("^(a{97}|a{101})*$", err, sizeof err);
    CHECK(re, "^(a{97}|a{101})*$: %s", err);
    if (re) {
        CHECK(!jsre_lengths(re, -1, err, sizeof err) && strstr(err, "do not settle"), "an unsettled length sequence was not refused");
        CHECK(jsre_lengths(re, 300, err, sizeof err), "a bounded ask refused: %s", err);
        uint64_t *set = malloc(jsre_words(re) * 8);
        jsre_start(re, set);
        CHECK(jsre_len_ok(re, set, 194, 194) && !jsre_len_ok(re, set, 195, 197) && jsre_len_ok(re, set, 198, 198) &&
              jsre_len_ok(re, set, 0, 0), "^(a{97}|a{101})*$ lengths");
        free(set);
    }
    jsre_free(re);
    printf("lengths: %d asks (%d yes) against a forward search, %d wrong\n", asks, yes, wrong);
}

// 5 --------------------------------------------------------------- the stack
// A schema compiles on a server thread, and a pattern can be thousands of
// characters: a 4,000-character pattern compiles and matches on a thread
// with a 128 KiB stack (a sequence is a balanced tree, not a chain).
static void *long_pattern(void *v) {
    int *ok = v;
    enum { N = 4000 };
    char *p = malloc(N + 3), *s = malloc(N);
    p[0] = '^';
    for (int i = 0; i < N; i++) p[1 + i] = s[i] = (char)('a' + i % 26);
    p[N + 1] = '$'; p[N + 2] = 0;
    char err[256];
    jsre *re = jsre_compile(p, err, sizeof err);
    *ok = re && nfa_match(re, s, N) && !nfa_match(re, s, N - 1);
    jsre_free(re); free(p); free(s);
    return NULL;
}

static void test_stack(void) {
    pthread_attr_t at;
    pthread_t th;
    int ok = 0;
    pthread_attr_init(&at);
    CHECK(pthread_attr_setstacksize(&at, 128 * 1024) == 0, "pthread_attr_setstacksize");
    CHECK(pthread_create(&th, &at, long_pattern, &ok) == 0, "pthread_create");
    pthread_join(th, NULL);
    pthread_attr_destroy(&at);
    CHECK(ok, "a 4,000-character pattern on a 128 KiB stack");
    printf("stack: a 4,000-character pattern compiled and matched on a 128 KiB stack\n");
}

// 6 --------------------------------------------------------------- the kept limit
// Past the kept limit a step fails, by name, rather than refuse a byte. The
// validator's own order: a mask's trials reach each state first (unkept), and
// the feed that follows only promotes it; the limit must count that too.
static void test_kept_limit(void) {
    char err[256];
    jsre *re = jsre_compile("^(a|b){0,30}c?$", err, sizeof err);
    CHECK(re && jsre_lengths(re, -1, err, sizeof err), "^(a|b){0,30}c?$: %s", err);
    if (!re) return;
    for (int order = 0; order < 2; order++) {
        jsre_pool *p = jsre_pool_new(BUDGETS[0]);
        jsre_pool_set_kept_max(p, 4);
        jsre_dfa *d = jsre_dfa_new(re, p);
        jsre_dfa_label(d, "the test's pattern");
        jsre_st st = jsre_dfa_start(d);
        int fed = 0;
        for (const char *c = "abababab"; *c && st != JSRE_FAIL && st != JSRE_DEAD; c++, fed++) {
            if (order) (void)jsre_dfa_step(d, st, (uint8_t)*c, false);   // the mask's trial first
            st = jsre_dfa_step(d, st, (uint8_t)*c, true);
        }
        jsre_pool_stat s;
        jsre_pool_stats(p, &s);
        // the start and three bytes are kept; the fourth byte's state is the fifth
        CHECK(st == JSRE_FAIL && fed == 4 && s.kept_states == 4 && s.failed && strstr(s.failed, "may keep") &&
              s.failed_label && !strcmp(s.failed_label, "the test's pattern"),
              "kept limit (%s): fed %d, kept %d, failed %s", order ? "trial first" : "fed only", fed, s.kept_states,
              s.failed ? s.failed : "no");
        jsre_dfa_free(d); jsre_pool_free(p);
    }
    jsre_free(re);
    printf("kept limit: a step past it fails by name, fed alone and after a trial\n");
}


// 7 --------------------------------------------------------------- huge bounds
// maxLength reaches INT_MAX (compile_bound takes it): the length sequence, the
// asks and the witness hold there without an overflow (UBSan: test-schema-san).
static void test_huge_bounds(void) {
    char err[256];
    jsre *re = jsre_compile("^(a|b)+$", err, sizeof err);
    CHECK(re && jsre_lengths(re, INT_MAX, err, sizeof err), "lengths at INT_MAX: %s", err);
    if (!re) return;
    uint64_t *set = malloc(jsre_words(re) * 8);
    jsre_start(re, set);
    CHECK(jsre_len_ok(re, set, 0, INT_MAX) && jsre_len_ok(re, set, INT_MAX - 3, INT_MAX) &&
          jsre_len_ok(re, set, INT_MAX, INT_MAX) && jsre_len_ok(re, set, INT_MAX, -1) &&
          !jsre_len_ok(re, set, 0, 0), "lengths near INT_MAX");
    char out[64];
    CHECK(jsre_witness(re, set, 3, INT_MAX, out, sizeof out) == 3, "a witness of 3");
    CHECK(jsre_witness(re, set, INT_MAX - 1, INT_MAX, out, sizeof out) == (int)sizeof out, "a witness too long stops at cap");
    jsre_pool *p = jsre_pool_new(BUDGETS[0]);
    jsre_dfa *d = jsre_dfa_new(re, p);
    jsre_st st = jsre_dfa_start(d);
    CHECK(jsre_dfa_len_ok(d, st, INT_MAX - 3, INT_MAX) && jsre_dfa_len_ok(d, st, 1, INT_MAX), "DFA lengths near INT_MAX");
    jsre_dfa_free(d); jsre_pool_free(p);
    free(set); jsre_free(re);
    printf("huge bounds: lengths, asks and witnesses at INT_MAX\n");
}


// 8 --------------------------------------------------------------- the heap
// Codex on #875: a flush freed states and tables but kept each DFA's slot
// array, hash table and free list at their high water, outside the budget
// the pool reported. Six DFAs for ^(a|b)*a(a|b){15}$ in one 8 MiB pool, trial
// walks over every binary 16-character window: what the heap holds for the
// cache stays within the budget, plus the kept states and one step, and the
// pool's own figure does not under-count it.
static void test_heap_bound(void) {
    enum { ND = 6 };
    char err[256];
    jsre *re[ND];
    for (int i = 0; i < ND; i++) {
        re[i] = jsre_compile("^(a|b)*a(a|b){15}$", err, sizeof err);
        CHECK(re[i] != NULL, "compile: %s", err);
        if (!re[i]) return;
    }
    size_t budget = (size_t)8 << 20, base = heap_live;
    heap_high = heap_live;
    jsre_pool *p = jsre_pool_new(budget);
    jsre_dfa *d[ND];
    for (int i = 0; i < ND; i++) d[i] = jsre_dfa_new(re[i], p);
    long walked = 0;
    jsre_pool_stat s;
    for (int i = 0; i < ND; i++) {
        jsre_st s0 = jsre_dfa_start(d[i]);
        for (unsigned w = 0; w < (1u << 16); w++) {
            jsre_st st = s0;
            for (int k = 0; k < 16 && st != JSRE_DEAD && st != JSRE_FAIL; k++, walked++)
                st = jsre_dfa_step(d[i], st, (w >> k & 1) ? 'b' : 'a', false);
            CHECK(st != JSRE_FAIL, "a walk failed");
            // a flush must make room: these walks need a few, never one per
            // fill (a flush that left the books at their high water flushed
            // at every fill from then on); checked as it goes, so it fails fast
            if ((w & 1023) == 0) {
                jsre_pool_stats(p, &s);
                if (s.flushes > 64) { CHECK(0, "%llu flushes by window %u of DFA %d", (unsigned long long)s.flushes, w, i); goto walked_out; }
            }
        }
    }
walked_out:
    jsre_pool_stats(p, &s);
    CHECK(s.flushes <= 64, "%llu flushes for walks that need a few", (unsigned long long)s.flushes);
    size_t held = heap_high - base;
    // one step's need past the budget: a set, a table, a hash table's growth
    size_t slack = 256 * 1024;
    CHECK(held <= budget + s.kept + slack, "the heap held %zu bytes for an %zu-byte pool (pool says %zu)", held, budget, s.high);
    CHECK(held <= s.high + slack, "the pool reported %zu bytes where the heap held %zu", s.high, held);
    CHECK(s.flushes > 0, "the walks never filled the pool (%llu flushes)", (unsigned long long)s.flushes);
    for (int i = 0; i < ND; i++) jsre_dfa_free(d[i]);
    jsre_pool_free(p);
    CHECK(heap_live == base, "the pool and its DFAs left %zu bytes", heap_live - base);
    printf("heap: %ld steps over six DFAs, %llu flushes; the heap held %zu bytes, the pool said %zu, budget %zu\n",
           walked, (unsigned long long)s.flushes, held, s.high, budget);

    // The kept states' side of the bound: fed walks (each step keeps its
    // state) over random strings of ^(a|b)*a(a|b){11}$, whose 4,096 states are
    // all distinct sets, up to a lowered kept limit. The heap holds the cache
    // and books within the budget plus the kept sets the pool counts, and
    // that count is the kept states times one set: so at JSRE_DFA_MAX states
    // of a 4,096-state NFA (512-byte sets) the kept side is at most 8 MiB.
    jsre *rk = jsre_compile("^(a|b)*a(a|b){11}$", err, sizeof err);
    CHECK(rk != NULL, "%s", err);
    if (!rk) { for (int i = 0; i < ND; i++) jsre_free(re[i]); return; }
    size_t base2 = heap_live;
    heap_high = heap_live;
    size_t small = 64 * 1024;
    p = jsre_pool_new(small);
    jsre_pool_set_kept_max(p, 3000);
    jsre_dfa *dk = jsre_dfa_new(rk, p);
    unsigned long long rs2 = 7;
    jsre_st st = jsre_dfa_start(dk);
    int fed = 0;
    while (st != JSRE_FAIL && fed < 100000) {
        rs2 ^= rs2 << 13; rs2 ^= rs2 >> 7; rs2 ^= rs2 << 17;
        // a trial first, as the validator asks, then the feed that keeps
        (void)jsre_dfa_step(dk, st, rs2 & 1 ? 'a' : 'b', false);
        st = jsre_dfa_step(dk, st, rs2 & 1 ? 'a' : 'b', true);
        fed++;
    }
    jsre_pool_stats(p, &s);
    held = heap_high - base2;
    CHECK(st == JSRE_FAIL && s.kept_states == 3000, "the kept limit: %d states after %d bytes", s.kept_states, fed);
    CHECK(s.kept == (size_t)s.kept_states * set_cost(rk), "kept bytes %zu for %d states", s.kept, s.kept_states);
    CHECK(held <= small + s.kept + s.books + slack, "kept: the heap held %zu, the pool says cache+books within %zu, kept %zu, books %zu",
          held, small, s.kept, s.books);
    CHECK(held <= s.high + slack, "kept: the pool reported %zu where the heap held %zu", s.high, held);
    jsre_dfa_free(dk); jsre_pool_free(p); jsre_free(rk);
    for (int i = 0; i < ND; i++) jsre_free(re[i]);
    printf("heap, kept side: %d kept states hold %zu bytes of sets, %zu of books; the heap held %zu, the pool said %zu\n",
           s.kept_states, s.kept, s.books, held, s.high);
}

// 9 --------------------------------------------------------------- warm second thread
// Codex on #875: a cached transition promoted an unkept state for a second
// thread without the owner check. The main thread fills ^(a|b)+$ on 'a' as a
// trial; another thread then takes the same transition keeping it.
typedef struct { jsre_dfa *d; jsre_st st; jsre_st got; } warm_arg;
static void *warm_thread(void *v) {
    warm_arg *w = v;
    w->got = jsre_dfa_step(w->d, w->st, 'a', true);
    return NULL;
}
static void test_warm_second_thread(void) {
    char err[256];
    jsre *re = jsre_compile("^(a|b)+$", err, sizeof err);
    CHECK(re != NULL, "%s", err);
    if (!re) return;
    jsre_pool *p = jsre_pool_new(BUDGETS[0]);
    jsre_dfa *d = jsre_dfa_new(re, p);
    jsre_st st = jsre_dfa_start(d);
    jsre_st nx = jsre_dfa_step(d, st, 'a', false);   // the fill, on this thread
    CHECK(nx != JSRE_FAIL && nx != JSRE_DEAD && !jsre_dfa_kept(d, nx), "the trial step");
    jsre_pool_stat before, after;
    jsre_pool_stats(p, &before);
    warm_arg w = { d, st, 0 };
    pthread_t th;
    CHECK(pthread_create(&th, NULL, warm_thread, &w) == 0, "pthread_create");
    pthread_join(th, NULL);
    jsre_pool_stats(p, &after);
    CHECK(w.got == JSRE_FAIL && after.failed && strstr(after.failed, "second thread") &&
          after.kept_states == before.kept_states,
          "a second thread promoted a cached state: got %s, kept %d -> %d",
          w.got == JSRE_FAIL ? "FAIL" : "a handle", before.kept_states, after.kept_states);
    jsre_dfa_free(d); jsre_pool_free(p); jsre_free(re);
    printf("warm second thread: a cached promotion from another thread is refused\n");
}

int main(void) {
    test_stack();
    test_kept_limit();
    test_huge_bounds();
    test_heap_bound();
    test_warm_second_thread();
    test_cases();
    test_cache();
    test_lengths();
    if (fails) { fprintf(stderr, "test_jsre: %d failures\n", fails); return 1; }
    printf("test_jsre: ok\n");
    return 0;
}
