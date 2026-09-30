// ECDSA over the NIST prime curves (P-256, P-384, P-521), the key types the
// OpenSSF Model Signing registry requires of a verifier. The arithmetic is
// deliberately plain (32-bit-limb Montgomery multiplication, Jacobian points,
// no curve-specific reduction): a model-signature check runs once at load and
// takes milliseconds, and plainness is what makes the code auditable against
// the reference vectors in tests/test_ecdsa.c.
//
// Signing exists for one offline command, `runner --sign-model` (R1.2.5): the
// serving process never holds an ECDSA private key. Signatures are
// deterministic (RFC 6979), so the same key and model always produce the same
// bundle. The signer walks the scalar with a fixed sequence of point
// operations, but the field arithmetic is not constant-time (its final
// reduction is a branch): sign on a machine you control, not a shared host.
#ifndef RUNNER_ECDSA_H
#define RUNNER_ECDSA_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { EC_P256 = 0, EC_P384 = 1, EC_P521 = 2 } ec_curve;

typedef enum { EC_SHA256 = 0, EC_SHA384 = 1, EC_SHA512 = 2 } ec_hash;

// Byte length of one field element / scalar for the curve (32, 48, 66).
size_t ec_field_bytes(ec_curve c);

// The digest FIPS 186-5 pairs with the curve (SHA-256, -384, -512), which is
// also the one the reference model_signing signer uses for each curve.
ec_hash ec_curve_hash(ec_curve c);
size_t ec_hash_bytes(ec_hash h);
void ec_digest(ec_hash h, const void *m, size_t n, uint8_t *out);

// Verify an ECDSA signature. `pub` is the uncompressed point X||Y (two field
// elements, big-endian, 2*ec_field_bytes long); `r` and `s` are big-endian
// scalars of any length up to the field size (leading zeros allowed);
// `hash` is the message digest (any length; truncated per FIPS 186-5 when
// longer than the group order). Returns true only for a valid signature.
bool ecdsa_verify(ec_curve c, const uint8_t *pub, const uint8_t *hash,
                  size_t hash_len, const uint8_t *r, size_t r_len,
                  const uint8_t *s, size_t s_len);

// Parse a DER-encoded ECDSA signature (SEQUENCE { INTEGER r, INTEGER s })
// into big-endian r and s of exactly `n` bytes each. Returns false on any
// encoding fault (non-minimal lengths, negative or oversized integers).
bool ecdsa_der_sig_parse(const uint8_t *der, size_t der_len, size_t n,
                         uint8_t *r_out, uint8_t *s_out);

// Parse a DER SubjectPublicKeyInfo for an EC key (the body of a PEM
// "PUBLIC KEY" block) into the curve and the uncompressed point (X||Y, 2*n
// bytes, caller buffer of at least 132 bytes). Compressed points are refused.
bool ecdsa_spki_parse(const uint8_t *der, size_t der_len, ec_curve *curve_out,
                      uint8_t *pub_out, size_t *pub_len_out);

// Deterministic ECDSA (RFC 6979 section 3.2) with the private scalar `priv`
// (big-endian, exactly ec_field_bytes long, 1 <= d < n) over `digest`, the
// ec_hash_bytes(h)-byte hash of the message; h is also the HMAC-DRBG's hash.
// r_out and s_out receive ec_field_bytes each. False for an out-of-range key.
bool ecdsa_sign(ec_curve c, const uint8_t *priv, ec_hash h,
                const uint8_t *digest, uint8_t *r_out, uint8_t *s_out);

// The public point d*G, uncompressed X||Y (2*ec_field_bytes).
bool ecdsa_public_from_private(ec_curve c, const uint8_t *priv, uint8_t *pub_out);

// DER SEQUENCE { INTEGER r, INTEGER s } with minimal INTEGERs from n-byte
// big-endian scalars. Returns the length (at most 2*n + 9 bytes).
size_t ecdsa_der_sig_encode(const uint8_t *r, const uint8_t *s, size_t n,
                            uint8_t *out);

// The DER SubjectPublicKeyInfo for an uncompressed point (91, 120 or 158
// bytes; `out` must hold 158). Returns the length.
size_t ecdsa_spki_encode(ec_curve c, const uint8_t *pub, uint8_t *out);

// Parse an unencrypted EC private key: SEC1 ECPrivateKey (PEM "EC PRIVATE
// KEY") or PKCS#8 PrivateKeyInfo (PEM "PRIVATE KEY"). priv_out receives the
// scalar as ec_field_bytes big-endian bytes. A key that names no curve, a
// scalar out of range, or an embedded public key that is not d*G is refused.
bool ecdsa_private_key_parse(const uint8_t *der, size_t der_len,
                             ec_curve *curve_out, uint8_t *priv_out);

#endif
