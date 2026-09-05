// ML-DSA-44 wrapper over the vendored pq-crystals reference (src/mldsa/).
#include "mldsa.h"
#include "mldsa/api.h"

#include <string.h>

_Static_assert(pqcrystals_dilithium2_PUBLICKEYBYTES == MLDSA44_PUBLICKEYBYTES, "pk size");
_Static_assert(pqcrystals_dilithium2_SECRETKEYBYTES == MLDSA44_SECRETKEYBYTES, "sk size");
_Static_assert(pqcrystals_dilithium2_BYTES == MLDSA44_BYTES, "sig size");

// declared in the vendored sign.h under its namespace macro; spelled out here
// so this file needs only api.h
int pqcrystals_dilithium2_ref_keypair_seed(uint8_t *pk, uint8_t *sk, const uint8_t *seed);

void mldsa44_keypair(uint8_t pk[MLDSA44_PUBLICKEYBYTES],
                     uint8_t sk[MLDSA44_SECRETKEYBYTES],
                     const uint8_t seed[MLDSA44_SEEDBYTES]) {
    pqcrystals_dilithium2_ref_keypair_seed(pk, sk, seed);
}

bool mldsa44_sign(uint8_t sig[MLDSA44_BYTES], const void *m, size_t n,
                  const uint8_t sk[MLDSA44_SECRETKEYBYTES]) {
    size_t siglen = 0;
    int rc = pqcrystals_dilithium2_ref_signature(sig, &siglen, m, n, NULL, 0, sk);
    return rc == 0 && siglen == MLDSA44_BYTES;
}

bool mldsa44_verify(const uint8_t sig[MLDSA44_BYTES], const void *m, size_t n,
                    const uint8_t pk[MLDSA44_PUBLICKEYBYTES]) {
    return pqcrystals_dilithium2_ref_verify(sig, MLDSA44_BYTES, m, n, NULL, 0, pk) == 0;
}
