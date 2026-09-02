// Ed25519 (RFC 8032) signatures and SHA-512, the receipt chain primitives.
// Public-domain TweetNaCl subset; see ed25519.c.
#ifndef RUNNER_ED25519_H
#define RUNNER_ED25519_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// sk is the 64-byte expanded secret key (seed || public key), pk 32 bytes.
void ed25519_keypair(uint8_t pk[32], uint8_t sk[64], const uint8_t seed[32]);
bool ed25519_sign(uint8_t sig[64], const void *m, size_t n, const uint8_t sk[64]);
bool ed25519_verify(const uint8_t sig[64], const void *m, size_t n, const uint8_t pk[32]);
void ed25519_sha512(uint8_t out[64], const void *m, size_t n);
void ed25519_sha384(uint8_t out[48], const void *m, size_t n);

#endif
