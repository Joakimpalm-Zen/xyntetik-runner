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

// Opens (creates) DIR; sign_key may be NULL (unsigned manifests). False with
// the reason on stderr.
bool kvsnap_configure(const char *dir, const char *sign_key);
bool kvsnap_enabled(void);

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

#endif
