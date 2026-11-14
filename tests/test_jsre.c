// R2.18: JSON Schema `pattern` as a byte automaton (src/jsre.c).
//
//   1. ECMA-262's verdicts (tests/jsre_cases.h, from node's RegExp) on hand
//      cases and generated ECMA/Python divergences: the state-set automaton
//      and its lazy DFA under three pool budgets, fed and as a trial, must
//      answer each one; patterns it cannot build are refused by name.
//   2. The DFA cache against the automaton it memoizes, stepped as the
//      validator steps it (a fed walk that keeps its states, trials at every
//      position), under an unbounded, a 1-byte and a 4 KiB pool: equal answers,
//      flushes that happened, trial handles refused after one, and a fill from
//      a second thread refused.
//   3. Lengths: the backward sequence against a forward search over real bytes.
//   4. The pool keeps its budget.
#include "jsre.h"
#include "jsre_cases.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static const size_t BUDGETS[] = { (size_t)1 << 40, 1, 4096 };
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
        for (int k = 0; k < NB; k++) { a[k].p = jsre_pool_new(BUDGETS[k]); a[k].d = jsre_dfa_new(re, a[k].p); }
        for (int walk = 0; walk < 4; walk++) {
            jsre_start(re, set);
            bool alive = true;
            for (int k = 0; k < NB; k++) a[k].st = jsre_dfa_start(a[k].d);
            compare(re, set, alive, a, "the start", cur);
            for (int pos = 0; pos < 24 && alive; pos++) {
                for (int t = 0; t < 4; t++) {
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
                    if (t == 0 && b[1].st != JSRE_DEAD && b[1].st != JSRE_FAIL) {
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
            // 4: the budget holds, give or take what one step needs
            if (k == 2) {
                size_t slack = 4 * (w * 8 + 64) + 1024;
                CHECK(s.high <= BUDGETS[k] + s.kept + slack, "%s: a 4 KiB pool held %zu bytes", cur, s.high);
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
    printf("cache: %d patterns, %ld compared steps, %ld length asks, flushes %llu (1 byte) %llu (4 KiB), "
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

int main(void) {
    test_cases();
    test_cache();
    test_lengths();
    if (fails) { fprintf(stderr, "test_jsre: %d failures\n", fails); return 1; }
    printf("test_jsre: ok\n");
    return 0;
}
