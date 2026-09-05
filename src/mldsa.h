// ML-DSA-44 (FIPS 204) for receipts, beside Ed25519.
//
// Receipts are audit records with retention measured in years; an Ed25519
// signature is what a large quantum computer would forge, and "harvest now,
// forge later" applies to signed logs. This wraps the pq-crystals reference
// implementation vendored under src/mldsa/ (public domain / Apache 2.0),
// pinned to the 44 parameter set, with two deliberate changes documented in
// sign.c: keys derive from a caller-supplied 32-byte seed (the signkey file
// stores the seed, as the Ed25519 path does) and signing is deterministic
// (FIPS 204's hedged variant would make a re-signed receipt differ byte for
// byte, which is the opposite of what a receipt is for).
//
// Sizes against Ed25519: public key 1312 B (32), signature 2420 B (64).
// Measured on an M1 with the reference code: sign 392 us, verify 73 us,
// keygen 109 us; the TweetNaCl Ed25519 in this tree signs in 758 us and
// verifies in 1450 us, so ML-DSA-44 is not the slow option here.
#ifndef RUNNER_MLDSA_H
#define RUNNER_MLDSA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MLDSA44_SEEDBYTES 32
#define MLDSA44_PUBLICKEYBYTES 1312
#define MLDSA44_SECRETKEYBYTES 2560
#define MLDSA44_BYTES 2420

// deterministic: the same seed always yields the same key pair
void mldsa44_keypair(uint8_t pk[MLDSA44_PUBLICKEYBYTES],
                     uint8_t sk[MLDSA44_SECRETKEYBYTES],
                     const uint8_t seed[MLDSA44_SEEDBYTES]);
// deterministic: the same message and key always yield the same signature
bool mldsa44_sign(uint8_t sig[MLDSA44_BYTES], const void *m, size_t n,
                  const uint8_t sk[MLDSA44_SECRETKEYBYTES]);
bool mldsa44_verify(const uint8_t sig[MLDSA44_BYTES], const void *m, size_t n,
                    const uint8_t pk[MLDSA44_PUBLICKEYBYTES]);

#endif
