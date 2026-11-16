// R3.72: the decode schedule a tiled microbatch turn records in its receipt
// (profile.route.micro_steps), and the replay's read of it. Model-free: the
// recorder, the JSON it writes, and the parser --verify runs before a model
// loads. A position the list does not hold was decoded on the identity route
// (its solo bits), so the list holds the tiled steps only, as runs.
#include "../src/micro_sched.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int bad;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "R3.72: " __VA_ARGS__); fprintf(stderr, "\n"); bad++; } } while (0)

// runs: consecutive positions at one width join; a width change, a gap or a
// re-decoded position starts a new run or marks the list unreplayable
static void test_recorder_runs(void) {
    micro_sched s;
    micro_sched_reset(&s);
    for (int p = 10; p < 14; p++) micro_sched_add(&s, p, 8);   // 10..13 at 8
    micro_sched_add(&s, 14, 6);                               // a width change
    micro_sched_add(&s, 20, 6);                               // a gap
    micro_sched_add(&s, 21, 6);
    CHECK(s.n == 3 && !s.truncated, "three runs expected, got %d (truncated %d)", s.n, s.truncated);
    CHECK(s.n >= 1 && s.run[0].first_pos == 10 && s.run[0].count == 4 && s.run[0].width == 8, "run 0");
    CHECK(s.n >= 2 && s.run[1].first_pos == 14 && s.run[1].count == 1 && s.run[1].width == 6, "run 1");
    CHECK(s.n >= 3 && s.run[2].first_pos == 20 && s.run[2].count == 2 && s.run[2].width == 6, "run 2");
    micro_sched_add(&s, 21, 6);   // a position decoded again (a rollback): not a schedule a replay can follow
    CHECK(s.truncated, "a re-decoded position must mark the schedule unreplayable");
    micro_sched_reset(&s);
    CHECK(s.n == 0 && !s.truncated, "reset");
    for (int p = 0; p < 2 * MICRO_SCHED_MAX_RUNS + 2; p += 2) micro_sched_add(&s, p, 7);   // one run past the bound
    CHECK(s.n == MICRO_SCHED_MAX_RUNS && s.truncated, "past the bound: %d runs, truncated %d", s.n, s.truncated);
}

// the JSON the receipt carries, and the parser reading it back
static void test_json_round_trip(void) {
    micro_sched s, back;
    micro_sched_reset(&s);
    for (int p = 3; p < 6; p++) micro_sched_add(&s, p, 8);
    micro_sched_add(&s, 9, 6);
    char js[512];
    int n = micro_sched_json(&s, js, (int)sizeof(js));
    CHECK(n > 0 && !strcmp(js, ",\"micro_steps\":[[3,3,8],[9,1,6]]"), "json: %s", js);
    char doc[600], why[200] = "";
    snprintf(doc, sizeof(doc), "{\"micro\":\"tiles\"%s}", js);
    jv *route = json_parse(doc, strlen(doc));
    assert(route);
    CHECK(micro_sched_parse(route, 8, &back, why, (int)sizeof(why)), "parse: %s", why);
    CHECK(back.n == 2 && back.run[0].first_pos == 3 && back.run[0].count == 3 && back.run[0].width == 8 &&
          back.run[1].first_pos == 9 && back.run[1].count == 1 && back.run[1].width == 6, "parse: runs");
    jv_free(route);
    micro_sched_reset(&s);
    for (int p = 0; p < 2 * MICRO_SCHED_MAX_RUNS + 2; p += 2) micro_sched_add(&s, p, 7);
    n = micro_sched_json(&s, js, (int)sizeof(js));
    CHECK(n < 0 || strstr(js, "\"micro_steps_truncated\":true"), "a truncated list says so: %s", js);
    // a buffer too small for the list writes nothing rather than half a list
    micro_sched_reset(&s);
    micro_sched_add(&s, 3, 8);
    char tiny[8];
    n = micro_sched_json(&s, tiny, (int)sizeof(tiny));
    CHECK(n < 0 && tiny[0] == 0, "a short buffer: %d '%s'", n, tiny);
}

// what --verify refuses before a model loads, each by name
static void test_parse_refusals(void) {
    static const struct { const char *route, *why; } cases[] = {
        { "{\"micro\":\"tiles\"}", "no recorded decode schedule" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[3,2,8]],\"micro_steps_truncated\":true}", "truncated" },
        { "{\"micro\":\"tiles\",\"micro_steps\":{}}", "malformed" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[3,2]]}", "malformed" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[3,0,8]]}", "malformed" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[-1,2,8]]}", "malformed" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[3,2.5,8]]}", "malformed" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[3,2,1]]}", "width" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[3,2,9]]}", "width" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[3,4,8],[5,1,8]]}", "overlap" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[[9,1,8],[3,1,8]]}", "overlap" },
        { "{\"micro\":\"tiles\",\"micro_steps\":[]}", "empty" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        jv *route = json_parse(cases[i].route, strlen(cases[i].route));
        assert(route);
        micro_sched s;
        char why[200] = "";
        bool ok = micro_sched_parse(route, 8, &s, why, (int)sizeof(why));
        CHECK(!ok && strstr(why, cases[i].why), "%s: ok %d why '%s' (want '%s')", cases[i].route, ok, why, cases[i].why);
        jv_free(route);
    }
}

int main(void) {
    test_recorder_runs();
    test_json_round_trip();
    test_parse_refusals();
    if (bad) { fprintf(stderr, "micro sched: %d wrong\n", bad); return 1; }
    puts("micro sched tests ok");
    return 0;
}
