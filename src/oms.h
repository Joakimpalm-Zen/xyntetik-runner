// OpenSSF Model Signing (OMS) bundle verification for a loaded GGUF.
//
// An OMS bundle is a detached Sigstore bundle: a DSSE envelope whose payload
// is an in-toto Statement v1 with the predicate type
// `https://model_signing/signature/v1.0`, a manifest of (name, sha256) file
// resources, and the signature over the DSSE pre-authentication encoding.
// Runner verifies the `key` method (a long-lived key the operator trusts and
// passes as a PEM public key): the ECDSA signature over PAE(payloadType,
// payload), then the statement, then the loaded file's digest against the
// manifest entry that names it. Certificate and keyless (Fulcio/Rekor)
// methods are reported as unsupported, never as verified.
#ifndef RUNNER_OMS_H
#define RUNNER_OMS_H
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    // "verified" | "unverified" | "unsupported" | "malformed" | "missing"
    char status[16];
    char reason[256];          // human-readable detail (always set)
    char curve[8];             // "P-256" | "P-384" | "P-521"
    char hash[8];              // digest the signature was checked with
    char subject_digest[65];   // subject[0].digest.sha256 (the manifest root)
    char resource_name[128];   // manifest entry matched to the model file
    char key_hint[80];         // verificationMaterial.publicKey.hint (may be "")
    int  n_resources;
} oms_result;

// Verify `bundle_path` against the model file at `model_path` with the
// trusted public key in `pubkey_pem_path` (PEM "PUBLIC KEY", EC only).
// Returns true only when status is "verified"; `out` always describes why.
bool oms_verify_file(const char *bundle_path, const char *pubkey_pem_path,
                     const char *model_path, oms_result *out);

// Appends the JSON object the transcript records for a model signature
// ({"status":..,"subject_digest":..,"curve":..}) to `buf`, capped at `cap`.
int oms_result_json(const oms_result *r, char *buf, size_t cap);

#endif
