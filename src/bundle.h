// Receipt bundles (R1.2.1): one directory a verifier can take.
//
// A receipt names the model and the binary by digest and may be signed; the
// OMS model signature and the public keys that make those digests and
// signatures mean something live elsewhere. A bundle puts them together:
//
//   receipt.json        the transcript record, byte for byte
//   model.sig           the OMS bundle the load verified (when given)
//   model-pubkey.pem    the key that signature verifies with (when given)
//   bundle.json         xyntetik.runner.bundle.v1: every file with its
//                       sha256, the receipt's chain hash and signing key with
//                       its fingerprint, the model and binary digests a replay
//                       needs, and the commands that check it; optionally
//                       itself signed (--sign-key), so the bundle as a whole
//                       carries an owner
//
// `--check-bundle` verifies everything a bundle can prove on its own: the
// files are the ones listed, the receipt's chain hash recomputes, its
// signature verifies with the key the manifest names, the key fingerprints
// recompute, and the manifest's own signature when there is one. It does NOT
// replay the inference -- that takes the GGUF and the binary the manifest
// names by digest, and `--verify receipt.json` -- and it says so.
#ifndef RUNNER_BUNDLE_H
#define RUNNER_BUNDLE_H

typedef struct {
    const char *receipt;       // the transcript to bundle
    const char *out_dir;       // created; must not exist or be empty
    const char *model_sig;     // optional OMS bundle
    const char *model_pubkey;  // optional PEM key for it
    const char *sign_key;      // optional: sign bundle.json with this key
} bundle_opts;

// 0 on success; otherwise 1 with the reason on stderr and nothing left behind
// that could pass for a bundle.
int bundle_export(const bundle_opts *o);

// Prints one verdict line. 0 OK, 2 a check failed (tampered, mismatched,
// bad signature), 3 unverifiable (missing or unreadable manifest). trust_hex,
// when set, is the public key the manifest's signature must be made with.
int bundle_check(const char *dir, const char *trust_hex);

#endif
