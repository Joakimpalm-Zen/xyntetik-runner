// R3.72: the decode schedule a tiled microbatch turn records in its receipt.
//
// Under RUNNER_METAL_BATCH_TILES a decode step's arithmetic depends on how
// many sequences decode beside it: from MB_TILES_MIN columns the weight
// products take the GEMM tiles, below that the identity matvec, whose column
// is the solo step's bits (metal.m enc_mb_mv). A receipt that only says "the
// tiles ran somewhere" (route.micro) cannot be replayed, so --verify refuses
// it. This records which steps ran the tiles and at what width, as runs of
// consecutive KV positions: profile.route.micro_steps = [[first_pos, count,
// width], ...]. A position the list does not hold was decoded on the identity
// route, so solo steps need no record. Header-only: model.c records and
// writes it, main.c reads it before a model loads, tests/test_micro_sched.c
// checks all three without a model.
#ifndef RUNNER_MICRO_SCHED_H
#define RUNNER_MICRO_SCHED_H

#include "json.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// past this many runs the list is marked truncated and the record refused
#define MICRO_SCHED_MAX_RUNS 1024

typedef struct { int first_pos, count, width; } micro_run;
typedef struct micro_sched {
    micro_run run[MICRO_SCHED_MAX_RUNS];
    int n;
    bool truncated;   // past the bound, or a position decoded twice (a rollback)
} micro_sched;

static inline void micro_sched_reset(micro_sched *s) {
    s->n = 0;
    s->truncated = false;
}

// One decode step at KV position `pos` that ran the tiles in a microbatch of
// `width` columns. Consecutive positions at one width join a run. A position
// at or before the last one recorded (the turn rolled back and decoded it
// again) is a schedule no replay can follow: the list is marked truncated.
static inline void micro_sched_add(micro_sched *s, int pos, int width) {
    if (s->truncated) return;
    if (s->n) {
        micro_run *r = &s->run[s->n - 1];
        int end = r->first_pos + r->count;
        if (pos < end) { s->truncated = true; return; }
        if (pos == end && width == r->width) { r->count++; return; }
    }
    if (s->n == MICRO_SCHED_MAX_RUNS) { s->truncated = true; return; }
    s->run[s->n++] = (micro_run){ pos, 1, width };
}

// The receipt's fields, written to be appended inside the route object:
// ,"micro_steps":[[p,c,w],...] and ,"micro_steps_truncated":true when marked.
// Returns the bytes written, or -1 with out empty when the list does not fit
// (half a list would read as a shorter schedule).
static inline int micro_sched_json(const micro_sched *s, char *out, int cap) {
    if (cap <= 0) return -1;
    out[0] = 0;
    int n = snprintf(out, (size_t)cap, ",\"micro_steps\":[");
    for (int i = 0; i < s->n && n >= 0 && n < cap; i++)
        n += snprintf(out + n, (size_t)(cap - n), "%s[%d,%d,%d]", i ? "," : "",
                      s->run[i].first_pos, s->run[i].count, s->run[i].width);
    if (n >= 0 && n < cap) n += snprintf(out + n, (size_t)(cap - n), "]");
    if (n >= 0 && n < cap && s->truncated)
        n += snprintf(out + n, (size_t)(cap - n), ",\"micro_steps_truncated\":true");
    if (n < 0 || n >= cap) { out[0] = 0; return -1; }
    return n;
}

static inline bool micro_sched_whole(const jv *v, double lo, double hi, int *out) {
    if (!v || v->type != J_NUM || v->num < lo || v->num > hi || v->num != (double)(int)v->num) return false;
    *out = (int)v->num;
    return true;
}

// The replay's read of a record's route: true with the runs in *s, or false
// with the reason in why. A tiles record from before R3.72 has no list; a
// truncated or malformed one, a width outside [2, width_max], and runs that
// go backwards or overlap are refused, each by name.
static inline bool micro_sched_parse(const jv *route, int width_max, micro_sched *s,
                                     char *why, int cap) {
    micro_sched_reset(s);
    jv *steps = jv_get((jv *)route, "micro_steps");
    jv *trunc = jv_get((jv *)route, "micro_steps_truncated");
    if (!steps) {
        snprintf(why, (size_t)cap, "the tiled turn carries no recorded decode schedule "
                 "(micro_steps; records before R3.72)");
        return false;
    }
    if (trunc && !(trunc->type == J_BOOL && !trunc->b)) {
        snprintf(why, (size_t)cap, "the tiled turn's decode schedule is truncated "
                 "(more than %d runs, or a position decoded twice)", MICRO_SCHED_MAX_RUNS);
        return false;
    }
    if (steps->type != J_ARR || steps->n > MICRO_SCHED_MAX_RUNS) {
        snprintf(why, (size_t)cap, "malformed micro_steps (an array of at most %d runs)", MICRO_SCHED_MAX_RUNS);
        return false;
    }
    if (steps->n == 0) {
        snprintf(why, (size_t)cap, "an empty micro_steps on a tiled turn");
        return false;
    }
    int end = -1;
    for (int i = 0; i < steps->n; i++) {
        jv *r = steps->items[i];
        int p, c, w;
        if (!r || r->type != J_ARR || r->n != 3 ||
            !micro_sched_whole(r->items[0], 0, 2147483647.0, &p) ||
            !micro_sched_whole(r->items[1], 1, 2147483647.0, &c) ||
            !micro_sched_whole(r->items[2], -2147483648.0, 2147483647.0, &w)) {
            snprintf(why, (size_t)cap, "malformed micro_steps run %d (want [first_pos, count, width], "
                     "whole numbers, count at least 1)", i);
            return false;
        }
        if (w < 2 || w > width_max) {
            snprintf(why, (size_t)cap, "micro_steps run %d: width %d outside 2 to %d", i, w, width_max);
            return false;
        }
        if (p < end || (double)p + (double)c > 2147483647.0) {
            snprintf(why, (size_t)cap, "micro_steps run %d goes backwards or would overlap the run before it", i);
            return false;
        }
        s->run[s->n++] = (micro_run){ p, c, w };
        end = p + c;
    }
    return true;
}

#endif // RUNNER_MICRO_SCHED_H
