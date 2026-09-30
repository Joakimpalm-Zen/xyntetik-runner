// The Responses store (R10.6): what `store:true` keeps and
// `previous_response_id` reads back.
//
// In memory only, never written to disk: a stored response holds the caller's
// conversation, and a store that outlived the process would be a transcript
// archive nobody asked for. Bounded by bytes (RUNNER_RESPONSES_STORE_MB,
// default 64) and by age (RUNNER_RESPONSES_STORE_TTL seconds, default 3600);
// the least recently used entry goes first when the budget is full.
//
// Each entry keeps the EFFECTIVE input the response answered -- the whole
// history, an earlier previous_response_id already expanded -- beside the
// response body, so a chain survives the eviction of its ancestors and a
// lookup never walks a chain.
#ifndef RUNNER_RESPSTORE_H
#define RUNNER_RESPSTORE_H

#include <stdbool.h>
#include <stddef.h>

// budget 0 disables storing (every put is refused); ttl_s <= 0 keeps forever.
void  respstore_configure(size_t budget_bytes, double ttl_s);
// Read RUNNER_RESPONSES_STORE_MB / _TTL and configure; drops every entry.
void  respstore_reset_from_env(void);
// input_json: a JSON array of input items; body_json: the response object.
// False when the entry cannot be kept (budget, allocation); the response
// itself is unaffected.
bool  respstore_put(const char *id, const char *input_json, size_t input_n,
                    const char *body_json, size_t body_n);
// malloc'd copies (NUL-terminated, *n = length), or NULL when unknown or
// expired.
char *respstore_body(const char *id, size_t *n);
char *respstore_input(const char *id, size_t *n);
bool  respstore_delete(const char *id);

#endif
