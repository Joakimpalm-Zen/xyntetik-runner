// The Responses store (R10.6). See respstore.h.
#include "respstore.h"
#include "compat.h"
#include "runner.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct rs_entry {
    struct rs_entry *next;
    char   *id, *input, *body;
    size_t  input_n, body_n;
    double  created, used;
} rs_entry;

static struct {
    rs_entry       *head;
    size_t          bytes, budget;
    double          ttl;
    bool            configured;
    pthread_mutex_t mu;
} RS = { NULL, 0, 0, 0, false, PTHREAD_MUTEX_INITIALIZER };

static size_t rs_cost(const rs_entry *e) {
    return e->input_n + e->body_n + strlen(e->id) + sizeof *e;
}

static void rs_drop(rs_entry **pp) {
    rs_entry *e = *pp;
    *pp = e->next;
    RS.bytes -= rs_cost(e);
    free(e->id); free(e->input); free(e->body); free(e);
}

// caller holds mu
static void rs_defaults(void) {
    if (RS.configured) return;
    RS.configured = true;
    RS.budget = (size_t)env_u64("RUNNER_RESPONSES_STORE_MB", 0, 1u << 20, 64)
                * 1024 * 1024;
    RS.ttl = env_f64("RUNNER_RESPONSES_STORE_TTL", 0.0, 1e9, 3600.0);
}

static void rs_expire(double now) {
    if (RS.ttl <= 0) return;
    for (rs_entry **pp = &RS.head; *pp; ) {
        if (now - (*pp)->created > RS.ttl) rs_drop(pp);
        else pp = &(*pp)->next;
    }
}

static rs_entry **rs_find(const char *id) {
    for (rs_entry **pp = &RS.head; *pp; pp = &(*pp)->next)
        if (!strcmp((*pp)->id, id)) return pp;
    return NULL;
}

void respstore_configure(size_t budget_bytes, double ttl_s) {
    pthread_mutex_lock(&RS.mu);
    RS.configured = true;
    RS.budget = budget_bytes;
    RS.ttl = ttl_s;
    pthread_mutex_unlock(&RS.mu);
}

void respstore_reset_from_env(void) {
    pthread_mutex_lock(&RS.mu);
    while (RS.head) rs_drop(&RS.head);
    RS.bytes = 0;
    RS.configured = false;
    rs_defaults();
    pthread_mutex_unlock(&RS.mu);
}

bool respstore_put(const char *id, const char *input_json, size_t input_n,
                   const char *body_json, size_t body_n) {
    rs_entry *e = calloc(1, sizeof *e);
    if (!e) return false;
    e->id = strdup(id);
    e->input = malloc(input_n + 1);
    e->body = malloc(body_n + 1);
    if (!e->id || !e->input || !e->body) {
        free(e->id); free(e->input); free(e->body); free(e);
        return false;
    }
    memcpy(e->input, input_json, input_n); e->input[input_n] = 0;
    memcpy(e->body, body_json, body_n);    e->body[body_n] = 0;
    e->input_n = input_n;
    e->body_n = body_n;
    size_t need = rs_cost(e);

    pthread_mutex_lock(&RS.mu);
    rs_defaults();
    double now = now_s();
    e->created = e->used = now;
    rs_expire(now);
    rs_entry **old = rs_find(id);
    if (old) rs_drop(old);
    // least recently used first, until the new entry fits
    while (RS.head && RS.bytes + need > RS.budget) {
        rs_entry **victim = &RS.head;
        for (rs_entry **pp = &RS.head; *pp; pp = &(*pp)->next)
            if ((*pp)->used < (*victim)->used) victim = pp;
        rs_drop(victim);
    }
    bool ok = RS.bytes + need <= RS.budget;
    if (ok) {
        e->next = RS.head;
        RS.head = e;
        RS.bytes += need;
    }
    pthread_mutex_unlock(&RS.mu);
    if (!ok) { free(e->id); free(e->input); free(e->body); free(e); }
    return ok;
}

static char *rs_copy(const char *id, bool body, size_t *n) {
    pthread_mutex_lock(&RS.mu);
    rs_defaults();
    double now = now_s();
    rs_expire(now);
    rs_entry **pp = rs_find(id);
    char *out = NULL;
    if (pp) {
        rs_entry *e = *pp;
        e->used = now;
        size_t len = body ? e->body_n : e->input_n;
        out = malloc(len + 1);
        if (out) {
            memcpy(out, body ? e->body : e->input, len + 1);
            if (n) *n = len;
        }
    }
    pthread_mutex_unlock(&RS.mu);
    return out;
}

char *respstore_body(const char *id, size_t *n)  { return rs_copy(id, true, n); }
char *respstore_input(const char *id, size_t *n) { return rs_copy(id, false, n); }

bool respstore_delete(const char *id) {
    pthread_mutex_lock(&RS.mu);
    rs_entry **pp = rs_find(id);
    if (pp) rs_drop(pp);
    pthread_mutex_unlock(&RS.mu);
    return pp != NULL;
}
