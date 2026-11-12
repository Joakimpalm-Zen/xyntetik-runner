// libFuzzer harness for signed-record verification (src/envelope.c): the
// code a third party runs on a receipt, a transcript or a lineage record it
// did not write, to decide whether to believe it. The file is hostile by
// definition; the verifier must answer "unsigned", "bad" or "ok" and nothing
// else.
//
// The bytes are checked in memory (receipt_signature_check, as a served
// receipt is) and through the file path (record_signature_state and
// record_chain_hash, as `--check-record` reads one), because the two share
// the parser but not the reader.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "envelope.h"
#include "fuzz_files.h"

static char path[256];
static void cleanup(void) { unlink(path); fuzz_files_close(); }
int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv; fuzz_files_init();
    snprintf(path, sizeof path, "%s/record.json", fuzz_dir);
    atexit(cleanup); return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 65536) return 0;
    // NUL-terminated at the input's own size, as record_read leaves it
    char *rec = malloc(size + 1);
    if (!rec) return 0;
    memcpy(rec, data, size);
    rec[size] = 0;
    char pub[SIGN_PUBHEX_CAP];
    receipt_sig_state st = receipt_signature_check(rec, size, pub);
    assert(st == RSIG_NONE || st == RSIG_OK || st == RSIG_BAD || st == RSIG_MALFORMED);
    assert(memchr(pub, 0, sizeof pub));          // always a terminated string
    free(rec);

    fuzz_write(path, data, size);
    fuzz_mute();
    char pub2[SIGN_PUBHEX_CAP], chain[65];
    int state = record_signature_state(path, NULL, pub2);
    assert(state == 0 || state == 1 || state == 2);
    // the file reader and the in-memory check agree on a signed, verified record
    if (state == 0) assert(st == RSIG_OK);
    if (record_chain_hash(path, chain)) assert(strlen(chain) == 64);
    else assert(chain[0] == 0);
    fuzz_unmute();
    return 0;
}
