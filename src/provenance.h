// Identity of the running server for GET /v1/runner/provenance (R10.2.1).
//
// A receipt already carries the statement an operator wants to check against
// a running server -- the binary's sha256, the model's sha256 and its load-time
// signature verdict, the envelope verdict, the adapter -- but only after an
// inference, and only from the CLI. This module holds the same facts for the
// server's resident model, established by this process at load, so a route can
// answer without an enclave.
//
// What it vouches for and what it does not:
//   - the binary digest is the executable file hashed once at server start;
//   - the model digest is the file hashed after the load, in the background (a
//     multi-GB file takes a minute and must not hold a request); until it is
//     known the state is "hashing";
//   - when the file on disk no longer has the size and modification time it had
//     at load, the digest is withheld and the state is "changed_since_load": the
//     bytes on disk are not the bytes being served, and publishing their digest
//     would vouch for the wrong file;
//   - the signature and envelope verdicts are the ones the load ran under, not
//     a re-read of the sidecars now.
// None of this is an attestation: a process can only report on itself.
#ifndef RUNNER_PROVENANCE_H
#define RUNNER_PROVENANCE_H

#include <stdbool.h>
#include <stdint.h>

#include "json.h"
#include "oms.h"

typedef struct {
    const char *model_path;
    const char *adapter_path;     // NULL = no adapter
    float       adapter_scale;
    const oms_result *signature;  // NULL or empty status = none requested/found
    const oms_result *adapter_signature;   // the same, for adapter_path
    int         envelope_state;   // enum envelope_state
    const char *envelope_detail;  // the gate's one-line summary ("" when silent)
} provenance_load;

// Hash the running executable. Called once at server start; safe to repeat.
void provenance_init(void);
// Record the model that just became resident. Copies everything it needs.
void provenance_note_load(const provenance_load *l);
// The resident model is gone (unload, swap, shutdown).
void provenance_note_unload(void);
// The resident model's and the executable's sha256 for a receipt (R1.2.2),
// waiting for the background model digest when it is still being taken.
// False when there is no resident model, its digest failed, or the file on
// disk is no longer the one loaded: a receipt must not name bytes that are
// not the ones served.
bool provenance_digests(char model[65], char binary[65]);
// How many models this process has made resident so far: 0 before the first
// load, and one more after every load, reload or swap. An unload does not
// move it, so a client that read N and sees N again with a resident model is
// talking to the same load. *is_resident says whether one is resident now.
uint64_t provenance_load_generation(bool *is_resident);
// Does the resident model meet a client's expectation? want_generation 0 and
// want_sha256 NULL each mean "not asked". Never waits: a digest that is still
// being taken, failed, or withheld because the file changed is UNKNOWN, not a
// guess either way.
enum { PROV_EXPECT_OK = 0, PROV_EXPECT_MISMATCH, PROV_EXPECT_UNKNOWN };
int provenance_expect(uint64_t want_generation, const char *want_sha256);
// Append `"build":{...},"model":{...}|null,"adapter":{...}|null` to b.
// `model_id` is the id the server answers to for the resident model; it is
// the caller's, because a registry name is chosen after the load.
void provenance_render(sbuf *b, const char *model_id);

#endif
