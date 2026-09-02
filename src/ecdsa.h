// ECDSA verification over the NIST prime curves (P-256, P-384, P-521), the
// key types the OpenSSF Model Signing registry requires of a verifier.
// Verification only: Runner never holds an ECDSA private key. The
// arithmetic is deliberately plain (32-bit-limb Montgomery multiplication,
// Jacobian points, no curve-specific reduction): a model-signature check runs
// once at load and takes milliseconds, and plainness is what makes the code
// auditable against the reference vectors in tests/test_ecdsa.c.
#ifndef RUNNER_ECDSA_H
#define RUNNER_ECDSA_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { EC_P256 = 0, EC_P384 = 1, EC_P521 = 2 } ec_curve;

// Byte length of one field element / scalar for the curve (32, 48, 66).
size_t ec_field_bytes(ec_curve c);

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

#endif
