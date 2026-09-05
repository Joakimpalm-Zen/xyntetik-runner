// ML-DSA-44 known-answer test: NIST ACVP FIPS 204 vectors (tests/mldsa_kat.h)
// for key generation from a seed and for a deterministic signature, then the
// wrapper's own round trip (verify, flipped signature byte rejected, altered
// message rejected, wrong key rejected). Receipts sign with this module when
// the key file says "ml-dsa-44", so the external anchor is NIST's vector set,
// not the module's own output.
#include <stdio.h>
#include <string.h>
#include "mldsa.h"
#include "mldsa_kat.h"

static int fail(const char *what) {
    printf("mldsa: FAIL %s\n", what);
    return 1;
}

int main(void) {
    uint8_t pk[MLDSA44_PUBLICKEYBYTES], sk[MLDSA44_SECRETKEYBYTES], sig[MLDSA44_BYTES];

    // ACVP keyGen: seed -> (pk, sk), byte for byte
    mldsa44_keypair(pk, sk, KAT_KG_SEED);
    if (memcmp(pk, KAT_KG_PK, sizeof pk)) return fail("ACVP keyGen public key");
    if (memcmp(sk, KAT_KG_SK, sizeof sk)) return fail("ACVP keyGen secret key");

    // ACVP sigGen (deterministic, pure, empty context): the exact signature
    if (!mldsa44_sign(sig, KAT_SG_MSG, sizeof KAT_SG_MSG, KAT_SG_SK)) return fail("ACVP sign returned false");
    if (memcmp(sig, KAT_SG_SIG, sizeof sig)) return fail("ACVP sigGen signature bytes");

    // round trip on the derived key
    static const char msg[] = "{\"schema_version\":\"xyntetik.runner.transcript.v1\"}";
    if (!mldsa44_sign(sig, msg, sizeof msg - 1, sk)) return fail("sign");
    if (!mldsa44_verify(sig, msg, sizeof msg - 1, pk)) return fail("verify of own signature");
    // deterministic: signing again gives the same bytes
    uint8_t sig2[MLDSA44_BYTES];
    if (!mldsa44_sign(sig2, msg, sizeof msg - 1, sk) || memcmp(sig, sig2, sizeof sig))
        return fail("signing is not deterministic");
    // a flipped signature bit
    sig2[100] ^= 1;
    if (mldsa44_verify(sig2, msg, sizeof msg - 1, pk)) return fail("flipped signature bit accepted");
    // an altered message
    char msg2[sizeof msg];
    memcpy(msg2, msg, sizeof msg);
    msg2[10] ^= 1;
    if (mldsa44_verify(sig, msg2, sizeof msg - 1, pk)) return fail("altered message accepted");
    // a foreign key
    uint8_t pk2[MLDSA44_PUBLICKEYBYTES], sk2[MLDSA44_SECRETKEYBYTES], seed2[32] = {1};
    mldsa44_keypair(pk2, sk2, seed2);
    if (mldsa44_verify(sig, msg, sizeof msg - 1, pk2)) return fail("foreign key accepted");

    printf("mldsa: OK (ACVP keyGen + sigGen known answers, round trip, 3 rejections)\n");
    return 0;
}
