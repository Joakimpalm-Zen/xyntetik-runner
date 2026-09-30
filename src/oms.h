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

typedef struct {
    const char *bundle_path; // NULL: discover <model>.sig on each load
    const char *pubkey_path;
    bool required;
} oms_policy;

// Apply the operator's signature policy on every model load. Logs the
// verdict; optional out has an empty status when no bundle was requested
// or discovered. Policy paths are borrowed for the caller's lifetime.
bool oms_check_model(const char *model_path, const oms_policy *policy,
                     oms_result *out);

// The same policy for any other artifact that changes what is served -- a
// LoRA adapter (R1.2.3). `label` names it in the log ("adapter"); the policy's
// bundle_path must be the ARTIFACT's own (NULL discovers <path>.sig), never
// the model's.
bool oms_check_artifact(const char *path, const oms_policy *policy,
                        const char *label, oms_result *out);

// Verify `bundle_path` against the model file at `model_path` with the
// trusted public key in `pubkey_pem_path` (PEM "PUBLIC KEY", EC only).
// Returns true only when status is "verified"; `out` always describes why.
bool oms_verify_file(const char *bundle_path, const char *pubkey_pem_path,
                     const char *model_path, oms_result *out);

// The conventional key fingerprint: sha256 of the DER SubjectPublicKeyInfo in
// a PEM "PUBLIC KEY" file (what `openssl pkey -pubin -outform DER | sha256sum`
// prints). False when the file is unreadable or not an EC public key PEM.
bool oms_pubkey_fingerprint(const char *pem_path, char hex[65]);

// Write a key-method bundle for the model at `model_path` (R1.2.5,
// `runner --sign-model`): a DSSE envelope over an in-toto Statement v1 whose
// manifest names the file (resource ".", as the reference signer names a
// single file) or, for a split GGUF, every part by name (subject: the parts'
// directory, as the reference names a directory). Signed with the PEM EC
// private key at `key_pem_path` (SEC1 or PKCS#8, unencrypted, P-256/384/521)
// by deterministic ECDSA with the curve's digest; the key hint is sha256 of
// the PEM public key, the reference's identifier. `out_path` is created and
// never overwritten. Reasons for a refusal go to stderr.
typedef struct {
    char curve[8];
    char hash[8];
    char subject_digest[65];
    char key_hint[65];
    int  n_resources;
} oms_sign_info;
bool oms_sign_file(const char *model_path, const char *key_pem_path,
                   const char *out_path, oms_sign_info *info);

// Appends the JSON object the transcript records for a model signature
// ({"status":..,"subject_digest":..,"curve":..}) to `buf`, capped at `cap`.
int oms_result_json(const oms_result *r, char *buf, size_t cap);

#endif
