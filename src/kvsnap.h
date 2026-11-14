// Signed KV snapshots: memory with provenance (R1.12.1, R1.12.2).
//
// With `--kv-snapshots DIR`, a named context (R10.4) can be written to DIR
// and loaded back by a later server, so an agent's memory -- a long prefix
// already prefilled -- survives the process, and a run can prove which
// memory it started from.
//
// A snapshot is two files. `<name>.kv` is the context in the prefix cache's
// own runner.prefix.v1 format (one entry: model key, token ids, KV bytes,
// body digest). `<name>.kv.json` is its manifest
// (xyntetik.runner.kv_snapshot.v1): the .kv file's sha256 and size, the
// token count and the sha256 of the token ids, the served model's sha256 and
// the engine's model key, the KV type and context length, the binary's
// sha256, and the receipt the caller names as its producer (checked to exist
// and to recompute), chained like any record and signed with --sign-key.
//
// Loading refuses a snapshot whose manifest does not recompute or verify,
// whose .kv bytes are not the ones named, or whose model or KV type differ
// from the live engine -- KV from another model is not an error, it is a
// confident wrong answer. A request built on a loaded snapshot names it in
// its receipt (serve.kv_snapshot), and replays from its prompt tokens like
// any other.
#ifndef RUNNER_KVSNAP_H
#define RUNNER_KVSNAP_H

#include <stdbool.h>
#include <stddef.h>

#include "engine.h"
#include "json.h"

// Opens (creates) DIR; sign_key may be NULL (unsigned manifests). A loaded
// manifest must be signed by trust_key (hex public key or sha256:<digest>),
// or, when that is NULL, by sign_key's own key; with neither there is no
// anchor and a manifest is held only to the key it names. False with the
// reason on stderr.
bool kvsnap_configure(const char *dir, const char *sign_key, const char *trust_key);

typedef struct {
    int         status;    // HTTP status of the refusal
    const char *code;      // error code for the client
    char        msg[320];
} kvsnap_err;

// POST /v1/runner/contexts/{ctx}/snapshot {name?, receipt?}: writes the
// snapshot and appends the response body to out.
bool kvsnap_save(const engine *e, const char *ctx, jv *req, sbuf *out,
                 kvsnap_err *err);
// POST /v1/runner/contexts {id, snapshot}: checks and pins the snapshot as
// context `id`, appending the response body to out.
bool kvsnap_load(const engine *e, const char *id, const char *name, sbuf *out,
                 kvsnap_err *err);

// ---- preamble snapshots, written and loaded unasked (R3.62) ----
//
// An agent opens every session with the same system message and tool list,
// and a new process prefilled it again from scratch. With
// `--kv-snapshots-auto TOKENS[:MB]` a server writes a snapshot of such a
// preamble after it has prefilled one cold, and loads it when a later
// request starts with the same tokens. It is the snapshot above, named by
// what it holds: "pre-" and the first 32 hex digits of the sha256 over the
// engine's model key and the token ids. The directory's "pre-" files are
// kept under MB megabytes (8192 unless given), the least recently used
// removed first; named snapshots are never touched.
#define KVSNAP_AUTO_NAME 40
// false with the reason on stderr; needs kvsnap_configure first
bool kvsnap_auto_configure(const char *arg);
bool kvsnap_auto_on(void);
int  kvsnap_auto_min(void);   // the shortest preamble worth a snapshot, in tokens
void kvsnap_auto_name(const engine *e, const int32_t *toks, int n,
                      char out[KVSNAP_AUTO_NAME]);
// Load the snapshot `name` into the slot when it is on disk and verifies: the
// slot then holds exactly its tokens, as after a prefill of them. Nothing is
// taken from the prefix cache's budget. A file that does not verify is
// removed, so the request's own cold prefill writes it again. False: the
// slot is unchanged or was reset.
bool kvsnap_auto_load(engine *e, const char *name);
// What that load started from, as the receipt records it (name, kv sha256,
// manifest chain hash); false when `name` is not the snapshot last loaded.
bool kvsnap_auto_origin(const char *name, char *out, size_t cap);
// Write rows [0, n) of the slot's history as snapshot `name`, unless it is
// on disk already, then bring the directory under its cap.
void kvsnap_auto_save(engine *e, const char *name, int n);

#endif
