// Per-request receipts in serve mode (R1.2.2).
//
// With `--receipts DIR`, every finished generation on the chat, completions,
// Responses and Messages surfaces writes a xyntetik.runner.transcript.v1
// record into DIR -- the same record the CLI's --transcript writes, replayed
// by the same `--verify` -- plus a "serve" object naming the surface, the
// request id, how the prompt's KV was obtained and what shaped the output
// beyond the sampler. Records are chained in write order (each one's
// chain.prev is the previous record's chain.hash, continued across restarts
// from the newest record in DIR) and signed with `--sign-key` when one is
// given. `--receipts-keep N` keeps the newest N records (0: all).
//
// A receipt holds the prompt and the output: it is the user's content, it is
// opt-in, and it never leaves DIR.
#ifndef RUNNER_RECEIPTS_H
#define RUNNER_RECEIPTS_H

#include <stdbool.h>

#include "envelope.h"

typedef struct {
    const char *dir;
    int         keep;      // newest records kept; 0 keeps every one
    const char *sign_key;  // NULL: unsigned records
} receipts_cfg;

// Opens DIR (created when missing), finds the chain head among the records
// already there and checks the signing key. False with the reason on stderr:
// a directory whose newest record does not parse would silently start a new
// chain, so it is refused instead.
bool receipts_configure(const receipts_cfg *c);
bool receipts_enabled(void);
// Writes one record. Fills ti's out_path, prev_hash, sign_key_path and
// chain_out itself; name_out (64 bytes) receives the file name and
// chain_out (65) the record's chain hash.
bool receipts_write(transcript_info *ti, char name_out[64], char chain_out[65]);
// Forget the configuration (server teardown).
void receipts_reset(void);
// R1.12: the path of receipt `file` (a plain receipt-<sequence>.json name)
// in DIR; false when receipts are off or the name is not one of theirs.
bool receipts_record_path(const char *file, char *path, size_t cap);
// The receipt files of a --receipts directory, oldest first (by sequence);
// returns the count, *paths_out (and each path) the caller's to free.
int  receipts_list_dir(const char *dir, char ***paths_out);

#endif
