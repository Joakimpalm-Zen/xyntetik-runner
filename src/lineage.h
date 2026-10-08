#pragma once
// Lineage (R17.1): every step of a model's life on Runner leaves a provenance
// record beside its output, each record names its inputs by sha256 and links
// the record that produced each input, and one command walks the chain back.
//
// The records are the sidecars the rewrite, merge, training and context-
// surgery steps already write (<artifact>.quant.json, .merge.json,
// .train.json, .context.json). This module adds the link between them and
// the walker; the record formats stay their own (v1, the link is an extra
// field old readers ignore).
#include <stdbool.h>
#include <stddef.h>

#include "json.h"

// The provenance record written beside ARTIFACT by a step, newest by mtime
// when more than one exists. False when the artifact has none (a downloaded
// file, or one made before records existed).
bool lineage_sidecar(const char *artifact, char *out, size_t cap);

// Appends {"path":...,"sha256":...} for an input file, plus
// "record":{"path":...,"sha256":...} when the input carries a sidecar record:
// the link the walker follows. `sha` may be NULL (it is computed).
bool lineage_put_input(sbuf *b, const char *path, const char *sha);

// Signs a freshly written record in place when a signing key was given (the
// receipts' chain and signature objects, envelope.c record_sign); a no-op
// without a key.
bool lineage_sign(const char *record_path, const char *sign_key);

// --lineage FILE: FILE is an artifact (its sidecar record is found) or a
// record. Prints the chain as a tree, one line per link, and returns
// 0 every link verified and signed, 1 every link consistent but at least one
// record unsigned, 2 a broken link (a hash that does not match, a record
// altered after it was written, a bad or untrusted signature) or no record.
int lineage_walk(const char *start, const char *trust_hex);

// --require-eval KIND: the model must carry <model>.eval.<KIND>.json (written
// by scripts/eval-record.py) about THIS file (its sha256), marked "pass",
// with a signature that verifies: by trust_hex when one is given; an
// unsigned record is accepted only without trust_hex. False with the reason
// in `why` otherwise. Hashes the model (about a second per GB).
bool lineage_require_eval(const char *model, const char *kind, const char *trust_hex,
                          char *why, size_t cap);
