// Ed25519 known-answer test: RFC 8032 section 7.1 vectors 1 and 2 (keypair
// derivation, signature bytes, verification), SHA-512("abc"), and a flipped
// signature bit rejected. The receipt chain signs with this module, so the
// external anchor is the RFC, not the module's own output.
#include <stdio.h>
#include <string.h>
#include "ed25519.h"

static int unhex(const char *h, uint8_t *o, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(h + 2 * i, "%2x", &v) != 1) return 0;
        o[i] = (uint8_t)v;
    }
    return 1;
}

static int fail(const char *what) {
    printf("ed25519: FAIL %s\n", what);
    return 1;
}

int main(void) {
    uint8_t seed[32], pk_want[32], sig_want[64], pk[32], sk[64], sig[64];
    // vector 1: empty message
    unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60", seed, 32);
    unhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", pk_want, 32);
    unhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b", sig_want, 64);
    ed25519_keypair(pk, sk, seed);
    if (memcmp(pk, pk_want, 32)) return fail("vector 1 public key");
    if (!ed25519_sign(sig, "", 0, sk)) return fail("sign 1");
    if (memcmp(sig, sig_want, 64)) return fail("vector 1 signature");
    if (!ed25519_verify(sig, "", 0, pk)) return fail("verify 1");
    sig[3] ^= 1;
    if (ed25519_verify(sig, "", 0, pk)) return fail("flipped signature accepted");
    if (ed25519_verify(sig_want, "x", 1, pk)) return fail("wrong message accepted");
    // vector 2: one byte 0x72
    unhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb", seed, 32);
    unhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", pk_want, 32);
    unhex("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00", sig_want, 64);
    ed25519_keypair(pk, sk, seed);
    if (memcmp(pk, pk_want, 32)) return fail("vector 2 public key");
    uint8_t m = 0x72;
    if (!ed25519_sign(sig, &m, 1, sk)) return fail("sign 2");
    if (memcmp(sig, sig_want, 64)) return fail("vector 2 signature");
    if (!ed25519_verify(sig, &m, 1, pk)) return fail("verify 2");
    // SHA-512("abc")
    uint8_t h[64], hw[64];
    ed25519_sha512(h, "abc", 3);
    unhex("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f", hw, 64);
    if (memcmp(h, hw, 64)) return fail("sha512 abc");
    printf("ed25519: ok (RFC 8032 vectors 1-2, SHA-512, forgeries rejected)\n");
    return 0;
}
