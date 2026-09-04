// The prompt-lookup search behind --draft-lookup (engine_lookup_draft).
//
// The CLI gate (tests/test_draft_lookup.py) proves the walk never changes the
// output whatever the search proposes, which is exactly why it cannot tell a
// correct search from a broken one: a search that proposed garbage would pass
// identity too, just slower. This file is the absolute anchor: every expected
// proposal below was worked out by hand from the stated rule (longest n first,
// most recent earlier occurrence, the tokens that followed it, up to k), with
// no call into the engine to produce the expectation.
#include "engine.h"

#include <stdio.h>
#include <string.h>

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

static int same(const int32_t *a, const int32_t *b, int n) {
    return memcmp(a, b, sizeof(int32_t) * (size_t)n) == 0;
}

int main(void) {
    int32_t out[16];

    // 1. Plain repeat. Context "1 2 3 4 5 6 | 1 2 3" ends with the 3-gram
    //    (1 2 3), which occurred once earlier at 0, six tokens back. What
    //    followed it there is 4 5 6 1 2 3 (through the context end), and the
    //    proposal then continues with period 6: 4 5. Eight tokens for k=8.
    {
        const int32_t h[] = {1, 2, 3, 4, 5, 6, 1, 2, 3};
        int n = engine_lookup_draft(h, 9, 8, out);
        const int32_t want[] = {4, 5, 6, 1, 2, 3, 4, 5};
        ck(n == 8 && same(out, want, 8), "repeat: proposes the tokens that followed, then the period");
        n = engine_lookup_draft(h, 9, 2, out);
        ck(n == 2 && same(out, want, 2), "repeat: k caps the proposal");
    }

    // 2. Longest n wins. "6 7 8 9 | 7 8 9 10 | 6 7 8 9": the 4-gram (6 7 8 9)
    //    at 0 is followed by 7; the 3-gram (7 8 9) more recently at 4 is
    //    followed by 10. Longest-first must answer 7, not 10: then 8 9 10 6
    //    7 8 9, the rest of the context after that occurrence (eight tokens
    //    under k=8, the period-8 continuation not reached).
    {
        const int32_t h[] = {6, 7, 8, 9, 7, 8, 9, 10, 6, 7, 8, 9};
        int n = engine_lookup_draft(h, 12, 8, out);
        const int32_t want[] = {7, 8, 9, 10, 6, 7, 8, 9};
        ck(n == 8 && same(out, want, 8),
           "longest matching n-gram wins over a more recent shorter one");
    }

    // 3. Most recent occurrence wins among equal n. "1 2 3 | 4 | 1 2 3 | 5 |
    //    1 2 3": (1 2 3) occurs at 0 (followed by 4) and at 4 (followed by
    //    5); the later one is the answer: 5, then 1 2 3 to the context end,
    //    then the period-4 continuation 5 1 2 3 for k=8.
    {
        const int32_t h[] = {1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3};
        int n = engine_lookup_draft(h, 11, 8, out);
        const int32_t want[] = {5, 1, 2, 3, 5, 1, 2, 3};
        ck(n == 8 && same(out, want, 8), "most recent earlier occurrence wins");
    }

    // 4. No repeat, no proposal: the round must cost nothing, not guess.
    {
        const int32_t h[] = {10, 11, 12, 13, 14, 15, 16, 17};
        ck(engine_lookup_draft(h, 8, 8, out) == 0, "no match proposes nothing");
        const int32_t one[] = {5};
        ck(engine_lookup_draft(one, 1, 8, out) == 0, "a one-token context proposes nothing");
        ck(engine_lookup_draft(h, 0, 8, out) == 0, "an empty context proposes nothing");
        ck(engine_lookup_draft(h, 8, 0, out) == 0, "k=0 proposes nothing");
    }

    // 5. Below the minimum n is not a match: "1 2 3 4 5 | 2 3" ends with the
    //    2-gram (2 3), which recurs at 1, but the 3-gram (5 2 3) does not.
    {
        const int32_t h[] = {1, 2, 3, 4, 5, 2, 3};
        ck(ENGINE_LOOKUP_N_MIN == 3, "minimum n is 3 (documented, measured)");
        ck(engine_lookup_draft(h, 7, 8, out) == 0,
           "a repeated 2-gram is below the minimum and not a match");
    }

    // 6. Overlapping run: "9 9 9 9". n_max is 5 and len is 4, so n=3 is the
    //    first that fits with a token after it: (9 9 9) at 0 is followed by
    //    one 9 at the context end, and the period is 1, so the proposal is
    //    9 9 9 9 (k=4): a run keeps running.
    {
        const int32_t h[] = {9, 9, 9, 9};
        int n = engine_lookup_draft(h, 4, 4, out);
        const int32_t want[] = {9, 9, 9, 9};
        ck(n == 4 && same(out, want, 4), "an overlapping run proposes its continuation");
    }

    // 7. The pending token is part of the suffix: "a b c d e a b c" with
    //    hist[len-1]=c pending. (a b c) at 0 is followed by d e a b c; k=3
    //    gives d e a. (n=4 and 5 do not recur.)
    {
        const int32_t h[] = {21, 22, 23, 24, 25, 21, 22, 23};
        int n = engine_lookup_draft(h, 8, 3, out);
        ck(n == 3 && out[0] == 24 && out[1] == 25 && out[2] == 21,
           "proposals follow the match through the pending token's position");
    }

    // 8. The suffix itself is never its own match, and n_max is honoured:
    //    a 6-token exact repeat matches on the 5-gram (2 3 4 5 6) at 1, and
    //    what followed it is 7 1 2 3 4 5 6, seven tokens under k=8.
    {
        const int32_t h[] = {1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6};
        int n = engine_lookup_draft(h, 13, 8, out);
        const int32_t want[] = {7, 1, 2, 3, 4, 5, 6, 7};
        ck(ENGINE_LOOKUP_N_MAX == 5, "maximum n is 5 (documented)");
        ck(n == 8 && same(out, want, 8),
           "a long repeat matches on the 5-gram and proposes what followed");
        n = engine_lookup_draft(h, 13, 4, out);
        ck(n == 4 && same(out, want, 4), "and k caps it");
    }

    if (g_fail) { fprintf(stderr, "test_lookup_draft: FAIL\n"); return 1; }
    fprintf(stderr, "test_lookup_draft: all ok\n");
    return 0;
}
