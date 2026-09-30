// OpenSSF Model Signing bundle verification. See oms.h.
#include "oms.h"
#include "ecdsa.h"
#include "ed25519.h"
#include "envelope.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set(oms_result *r, const char *status, const char *reason) {
    snprintf(r->status, sizeof r->status, "%s", status);
    snprintf(r->reason, sizeof r->reason, "%s", reason);
}

// ---- base64 / PEM -----------------------------------------------------------

static int b64v(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

// Decodes standard or URL-safe base64, ignoring whitespace; returns the byte
// count or -1 on a bad character. `out` must hold 3/4 of the input.
static long b64_decode(const char *s, size_t n, uint8_t *out) {
    unsigned acc = 0;
    int bits = 0;
    long k = 0;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (c == '=' ) break;
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        int v = b64v(c);
        if (v < 0) return -1;
        acc = (acc << 6) | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[k++] = (uint8_t)((acc >> bits) & 0xff);
        }
    }
    return k;
}

static char *read_all(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || sz > (64L << 20)) { fclose(f); return NULL; }
    rewind(f);
    char *b = malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t n = fread(b, 1, (size_t)sz, f);
    fclose(f);
    b[n] = 0;
    if (n_out) *n_out = n;
    return b;
}

// PEM "PUBLIC KEY" block -> DER bytes (caller frees)
static uint8_t *pem_public_key_der(const char *pem, size_t n, size_t *der_len) {
    const char *b = strstr(pem, "-----BEGIN PUBLIC KEY-----");
    if (!b) return NULL;
    b += strlen("-----BEGIN PUBLIC KEY-----");
    const char *e = strstr(b, "-----END PUBLIC KEY-----");
    if (!e) return NULL;
    (void)n;
    uint8_t *der = malloc((size_t)(e - b));
    if (!der) return NULL;
    long k = b64_decode(b, (size_t)(e - b), der);
    if (k <= 0) { free(der); return NULL; }
    *der_len = (size_t)k;
    return der;
}

// ---- the verifier -------------------------------------------------------------

bool oms_pubkey_fingerprint(const char *pem_path, char hex[65]) {
    size_t n = 0;
    char *pem = read_all(pem_path, &n);
    if (!pem) return false;
    size_t der_len = 0;
    uint8_t *der = pem_public_key_der(pem, n, &der_len);
    free(pem);
    if (!der) return false;
    envelope_data_sha256(der, der_len, hex);
    free(der);
    return true;
}

static const char *base_name(const char *p) {
    const char *b = p;
    for (const char *q = p; *q; q++) if (*q == '/' || *q == '\\') b = q + 1;
    return b;
}

static bool name_matches(const char *name, const char *model_path);

// A split GGUF part name, `<prefix>-NNNNN-of-MMMMM.gguf`: reports the
// 1-based part number and the count, and how long the prefix is.
static bool split_suffix(const char *path, unsigned *no, unsigned *count,
                         size_t *prefix_len) {
    size_t n = strlen(path);
    const char *tail = ".gguf";
    size_t tl = strlen(tail);
    // "-00001-of-00003" is 15 bytes
    if (n < 15 + tl || strcmp(path + n - tl, tail) != 0) return false;
    const char *p = path + n - tl - 15;
    if (p[0] != '-' || strncmp(p + 6, "-of-", 4) != 0) return false;
    unsigned a = 0, b = 0;
    for (int i = 1; i <= 5; i++) {
        if (p[i] < '0' || p[i] > '9' || p[9 + i] < '0' || p[9 + i] > '9') return false;
        a = a * 10 + (unsigned)(p[i] - '0');
        b = b * 10 + (unsigned)(p[9 + i] - '0');
    }
    if (a < 1 || b < 1 || a > b) return false;
    *no = a; *count = b; *prefix_len = (size_t)(p - path);
    return true;
}

// The manifest entry naming `path` (NULL when there is none); *want is its
// digest string and *algo its algorithm.
static jv *manifest_entry(jv *res, const char *path) {
    for (int i = 0; i < res->n; i++) {
        const char *nm = jv_str(jv_get(res->items[i], "name"), NULL);
        if (name_matches(nm, path)) return res->items[i];
    }
    return NULL;
}

static bool digest_equal(const char *want, const char *have) {
    bool same = strlen(want) == 64;
    for (int i = 0; same && i < 64; i++) {
        char a = want[i], c = have[i];
        if (a >= 'A' && a <= 'F') a = (char)(a - 'A' + 'a');
        if (a != c) same = false;
    }
    return same;
}

static bool name_matches(const char *name, const char *model_path) {
    if (!name) return false;
    if (strcmp(name, ".") == 0) return true;   // single-file model, the file itself
    const char *bn = base_name(model_path);
    size_t nl = strlen(name), bl = strlen(bn);
    if (strcmp(name, bn) == 0) return true;
    return nl > bl && name[nl - bl - 1] == '/' && strcmp(name + nl - bl, bn) == 0;
}

bool oms_verify_file(const char *bundle_path, const char *pubkey_pem_path,
                     const char *model_path, oms_result *out) {
    memset(out, 0, sizeof *out);
    size_t bn = 0, kn = 0;
    char *bundle = read_all(bundle_path, &bn);
    if (!bundle) { set(out, "missing", "cannot read the signature bundle"); return false; }
    char *pem = pubkey_pem_path ? read_all(pubkey_pem_path, &kn) : NULL;
    if (!pem) {
        free(bundle);
        set(out, "unverified", pubkey_pem_path ? "cannot read the public key"
                                              : "no trusted public key given");
        return false;
    }
    bool ok = false;
    jv *b = json_parse(bundle, bn);
    jv *stmt = NULL;
    uint8_t *der = NULL, *payload = NULL, *sig = NULL;
    do {
        if (!b) { set(out, "malformed", "bundle is not JSON"); break; }
        const char *mt = jv_str(jv_get(b, "mediaType"), "");
        if (strncmp(mt, "application/vnd.dev.sigstore.bundle", 35) != 0) {
            set(out, "malformed", "mediaType is not a Sigstore bundle");
            break;
        }
        jv *vm = jv_get(b, "verificationMaterial");
        if (vm) {
            if (jv_get(vm, "certificate") || jv_get(vm, "x509CertificateChain")) {
                set(out, "unsupported", "certificate and keyless bundles are not "
                                        "verified by this build (key method only)");
                break;
            }
            const char *hint = jv_str(jv_get(jv_get(vm, "publicKey"), "hint"), NULL);
            if (hint) snprintf(out->key_hint, sizeof out->key_hint, "%s", hint);
        }
        jv *env = jv_get(b, "dsseEnvelope");
        const char *ptype = jv_str(jv_get(env, "payloadType"), "");
        const char *pl_b64 = jv_str(jv_get(env, "payload"), NULL);
        jv *sigs = jv_get(env, "signatures");
        if (!env || strcmp(ptype, "application/vnd.in-toto+json") != 0 || !pl_b64 ||
            !sigs || sigs->type != J_ARR || sigs->n < 1) {
            set(out, "malformed", "dsseEnvelope is missing or not an in-toto payload");
            break;
        }
        const char *sig_b64 = jv_str(jv_get(sigs->items[0], "sig"), NULL);
        if (!sig_b64) { set(out, "malformed", "no signature in the envelope"); break; }
        size_t pl_n = strlen(pl_b64), sg_n = strlen(sig_b64);
        payload = malloc(pl_n + 4);
        sig = malloc(sg_n + 4);
        if (!payload || !sig) { set(out, "malformed", "out of memory"); break; }
        long pl_len = b64_decode(pl_b64, pl_n, payload);
        long sg_len = b64_decode(sig_b64, sg_n, sig);
        if (pl_len <= 0 || sg_len <= 0) { set(out, "malformed", "bad base64 in the envelope"); break; }
        // the trusted key
        size_t der_len = 0;
        der = pem_public_key_der(pem, kn, &der_len);
        ec_curve curve;
        uint8_t pub[132];
        size_t pub_len = 0;
        if (!der || !ecdsa_spki_parse(der, der_len, &curve, pub, &pub_len)) {
            set(out, "unverified", "the public key is not a PEM EC (P-256/384/521) key");
            break;
        }
        snprintf(out->curve, sizeof out->curve, "%s",
                 curve == EC_P256 ? "P-256" : curve == EC_P384 ? "P-384" : "P-521");
        size_t fb = ec_field_bytes(curve);
        uint8_t r[66], s[66];
        if (!ecdsa_der_sig_parse(sig, (size_t)sg_len, fb, r, s)) {
            set(out, "malformed", "signature is not a DER ECDSA SEQUENCE for this key's curve");
            break;
        }
        // DSSE PAE: "DSSEv1" SP LEN(type) SP type SP LEN(payload) SP payload
        char head[128];
        int hl = snprintf(head, sizeof head, "DSSEv1 %zu %s %ld ", strlen(ptype), ptype, pl_len);
        size_t pae_n = (size_t)hl + (size_t)pl_len;
        uint8_t *pae = malloc(pae_n);
        if (!pae) { set(out, "malformed", "out of memory"); break; }
        memcpy(pae, head, (size_t)hl);
        memcpy(pae + hl, payload, (size_t)pl_len);
        // the digest the signer used is not stated in the bundle; SHA-256 is
        // the reference tool's choice for every curve, the curve-matched
        // digests are what other DSSE signers use
        uint8_t h256[32], h384[48], h512[64];
        envelope_data_sha256_raw(pae, pae_n, h256);
        ed25519_sha384(h384, pae, pae_n);
        ed25519_sha512(h512, pae, pae_n);
        free(pae);
        bool sig_ok = false;
        if (ecdsa_verify(curve, pub, h256, 32, r, fb, s, fb)) {
            sig_ok = true;
            snprintf(out->hash, sizeof out->hash, "sha256");
        } else if (curve == EC_P384 && ecdsa_verify(curve, pub, h384, 48, r, fb, s, fb)) {
            sig_ok = true;
            snprintf(out->hash, sizeof out->hash, "sha384");
        } else if (curve == EC_P521 && ecdsa_verify(curve, pub, h512, 64, r, fb, s, fb)) {
            sig_ok = true;
            snprintf(out->hash, sizeof out->hash, "sha512");
        }
        if (!sig_ok) {
            set(out, "unverified", "signature does not verify with the trusted key");
            break;
        }
        // the statement
        stmt = json_parse((const char *)payload, (size_t)pl_len);
        if (!stmt) { set(out, "malformed", "payload is not JSON"); break; }
        const char *st = jv_str(jv_get(stmt, "_type"), "");
        const char *pt = jv_str(jv_get(stmt, "predicateType"), "");
        if (strcmp(st, "https://in-toto.io/Statement/v1") != 0) {
            set(out, "malformed", "payload is not an in-toto Statement v1");
            break;
        }
        if (strcmp(pt, "https://model_signing/signature/v1.0") != 0) {
            set(out, "unsupported", "predicateType is not model_signing/signature/v1.0");
            break;
        }
        jv *subj = jv_get(stmt, "subject");
        if (subj && subj->type == J_ARR && subj->n >= 1) {
            const char *sd = jv_str(jv_get(jv_get(subj->items[0], "digest"), "sha256"), NULL);
            if (sd) snprintf(out->subject_digest, sizeof out->subject_digest, "%s", sd);
        }
        jv *pred = jv_get(stmt, "predicate");
        jv *ser = jv_get(pred, "serialization");
        const char *method = jv_str(jv_get(ser, "method"), "");
        const char *htype = jv_str(jv_get(ser, "hash_type"), "");
        if (strcmp(method, "files") != 0 || strcmp(htype, "sha256") != 0) {
            set(out, "unsupported", "only the files/sha256 serialization is verified by this build");
            break;
        }
        jv *res = jv_get(pred, "resources");
        if (!res || res->type != J_ARR || res->n < 1) {
            set(out, "malformed", "manifest has no resources");
            break;
        }
        out->n_resources = res->n;
        // the loaded file's own digest against the manifest entry naming it
        char have[65];
        if (!envelope_file_sha256(model_path, have)) {
            set(out, "unverified", "cannot hash the model file");
            break;
        }
        jv *match = manifest_entry(res, model_path);
        if (!match) {
            set(out, "unverified", "the manifest has no entry for this model file");
            break;
        }
        const char *algo = jv_str(jv_get(match, "algorithm"), "");
        const char *want = jv_str(jv_get(match, "digest"), "");
        snprintf(out->resource_name, sizeof out->resource_name, "%s",
                 jv_str(jv_get(match, "name"), ""));
        if (strcmp(algo, "sha256") != 0) {
            set(out, "unsupported", "manifest entry uses a digest other than sha256");
            break;
        }
        if (!digest_equal(want, have)) {
            set(out, "unverified", "model file digest differs from the signed manifest");
            break;
        }
        // A split GGUF is loaded from every part, so every part is a model
        // file: each must be named in the manifest and match. Only the part
        // given on the command line was hashed before this, and a weight
        // changed in any other part still loaded as "verified" (external
        // review, 2026-09-27).
        unsigned no = 0, count = 0;
        size_t plen = 0;
        int n_parts = 1;
        bool parts_ok = true;
        if (split_suffix(model_path, &no, &count, &plen) && count > 1) {
            char part[4096];
            for (unsigned i = 1; i <= count && parts_ok; i++) {
                if (i == no) continue;
                int w = snprintf(part, sizeof part, "%.*s-%05u-of-%05u.gguf",
                                 (int)plen, model_path, i, count);
                if (w <= 0 || (size_t)w >= sizeof part) {
                    set(out, "unverified", "split part path too long");
                    parts_ok = false;
                    break;
                }
                jv *pm = manifest_entry(res, part);
                if (!pm) {
                    snprintf(out->reason, sizeof out->reason,
                             "the manifest has no entry for split part %u of %u",
                             i, count);
                    snprintf(out->status, sizeof out->status, "unverified");
                    parts_ok = false;
                    break;
                }
                if (strcmp(jv_str(jv_get(pm, "algorithm"), ""), "sha256") != 0) {
                    set(out, "unsupported", "manifest entry uses a digest other than sha256");
                    parts_ok = false;
                    break;
                }
                char phave[65];
                if (!envelope_file_sha256(part, phave)) {
                    snprintf(out->reason, sizeof out->reason,
                             "cannot hash split part %u of %u", i, count);
                    snprintf(out->status, sizeof out->status, "unverified");
                    parts_ok = false;
                    break;
                }
                if (!digest_equal(jv_str(jv_get(pm, "digest"), ""), phave)) {
                    snprintf(out->reason, sizeof out->reason,
                             "split part %u of %u digest differs from the signed manifest",
                             i, count);
                    snprintf(out->status, sizeof out->status, "unverified");
                    parts_ok = false;
                    break;
                }
                n_parts++;
            }
        }
        if (!parts_ok) break;
        if (n_parts > 1)
            snprintf(out->reason, sizeof out->reason,
                     "signature, statement and all %d split part digests verified",
                     n_parts);
        else
            snprintf(out->reason, sizeof out->reason,
                     "signature, statement and model digest verified");
        snprintf(out->status, sizeof out->status, "verified");
        ok = true;
    } while (0);
    if (stmt) jv_free(stmt);
    if (b) jv_free(b);
    free(der);
    free(payload);
    free(sig);
    free(pem);
    free(bundle);
    return ok;
}

bool oms_check_model(const char *model_path, const oms_policy *policy,
                     oms_result *out) {
    return oms_check_artifact(model_path, policy, "model", out);
}

bool oms_check_artifact(const char *model_path, const oms_policy *policy,
                        const char *label, oms_result *out) {
    oms_result local = {0};
    if (!out) out = &local;
    memset(out, 0, sizeof(*out));
    oms_policy empty = {0};
    if (!policy) policy = &empty;
    const char *sig = policy->bundle_path;
    char *auto_sig = NULL;
    if (!sig) {
        size_t n = strlen(model_path);
        auto_sig = malloc(n + sizeof(".sig"));
        if (!auto_sig) {
            set(out, "unverified", "out of memory locating signature bundle");
            fprintf(stderr, "error: %s signature: %s\n", label, out->reason);
            return false;
        }
        memcpy(auto_sig, model_path, n);
        memcpy(auto_sig + n, ".sig", sizeof(".sig"));
        FILE *f = fopen(auto_sig, "rb");
        if (f) { fclose(f); sig = auto_sig; }
    }
    bool allowed = true;
    if (sig) {
        bool verified = oms_verify_file(sig, policy->pubkey_path, model_path, out);
        fprintf(stderr, "%s signature: %s (%s%s%s%s%s)\n", label, out->status,
                out->reason, out->curve[0] ? "; " : "", out->curve,
                out->hash[0] ? "/" : "", out->hash);
        allowed = verified || !(policy->bundle_path || policy->pubkey_path ||
                                policy->required);
    } else if (policy->required) {
        set(out, "missing", strcmp(label, "model")
                            ? "no signature bundle (PATH.sig, or --lora-sig)"
                            : "no signature bundle (pass --model-sig)");
        allowed = false;
    }
    if (!allowed)
        fprintf(stderr, "error: %s signature: refusing to load %s: %s\n",
                label, model_path, out->reason);
    free(auto_sig);
    return allowed;
}

int oms_result_json(const oms_result *r, char *buf, size_t cap) {
    return snprintf(buf, cap,
                    "{\"status\":\"%s\",\"subject_digest\":\"%s\",\"curve\":\"%s\","
                    "\"hash\":\"%s\",\"resource\":\"%s\"}",
                    r->status, r->subject_digest, r->curve, r->hash,
                    r->resource_name);
}
