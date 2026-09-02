// ECDSA known-answer test against RFC 6979 appendix A.2.5-A.2.7: the P-256,
// P-384 and P-521 key pairs and their SHA-256 signatures over "sample". The
// hash of "sample" is computed here from the FIPS 180-4 value so the test
// does not depend on the envelope module. Also gates: r or s out of range,
// a flipped signature byte, a wrong hash, and the DER parsers (the signature
// SEQUENCE form OMS bundles carry, and an SPKI public key).
#include <stdio.h>
#include <string.h>
#include "ecdsa.h"

static int unhex(const char *h, uint8_t *o, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(h + 2 * i, "%2x", &v) != 1) return 0;
        o[i] = (uint8_t)v;
    }
    return 1;
}

static int fail(const char *what) {
    printf("ecdsa: FAIL %s\n", what);
    return 1;
}

typedef struct {
    ec_curve c;
    const char *name, *ux, *uy, *r, *s;
} vec;

// SHA-256("sample") per FIPS 180-4
static const char SAMPLE_SHA256[] =
    "af2bdbe1aa9b6ec1e2ade1d694f41fc71a831d0268e9891562113d8a62add1bf";

static const vec VECS[] = {
    { EC_P256, "P-256",
      "60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6",
      "7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299",
      "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
      "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8" },
    { EC_P384, "P-384",
      "EC3A4E415B4E19A4568618029F427FA5DA9A8BC4AE92E02E06AAE5286B300C64DEF8F0EA9055866064A254515480BC13",
      "8015D9B72D7D57244EA8EF9AC0C621896708A59367F9DFB9F54CA84B3F1C9DB1288B231C3AE0D4FE7344FD2533264720",
      "21B13D1E013C7FA1392D03C5F99AF8B30C570C6F98D4EA8E354B63A21D3DAA33BDE1E888E63355D92FA2B3C36D8FB2CD",
      "F3AA443FB107745BF4BD77CB3891674632068A10CA67E3D45DB2266FA7D1FEEBEFDC63ECCD1AC42EC0CB8668A4FA0AB0" },
    { EC_P521, "P-521",
      "01894550D0785932E00EAA23B694F213F8C3121F86DC97A04E5A7167DB4E5BCD371123D46E45DB6B5D5370A7F20FB633155D38FFA16D2BD761DCAC474B9A2F5023A4",
      "00493101C962CD4D2FDDF782285E64584139C2F91B47F87FF82354D6630F746A28A0DB25741B5B34A828008B22ACC23F924FAAFBD4D33F81EA66956DFEAA2BFDFCF5",
      "01511BB4D675114FE266FC4372B87682BAECC01D3CC62CF2303C92B3526012659D16876E25C7C1E57648F23B73564D67F61C6F14D527D54972810421E7D87589E1A7",
      "004A171143A83163D6DF460AAF61522695F207A58B95C0644D87E52AA1A347916E4F7A72930B1BC06DBE22CE3F58264AFD23704CBB63B29B931F7DE6C9D949A7ECFC" },
};

int main(void) {
    uint8_t hash[32];
    unhex(SAMPLE_SHA256, hash, 32);
    for (size_t i = 0; i < sizeof VECS / sizeof VECS[0]; i++) {
        const vec *v = &VECS[i];
        size_t fb = ec_field_bytes(v->c);
        uint8_t pub[132], r[66], s[66];
        unhex(v->ux, pub, fb);
        unhex(v->uy, pub + fb, fb);
        unhex(v->r, r, fb);
        unhex(v->s, s, fb);
        if (!ecdsa_verify(v->c, pub, hash, 32, r, fb, s, fb)) return fail(v->name);
        uint8_t bad[32];
        memcpy(bad, hash, 32);
        bad[0] ^= 1;
        if (ecdsa_verify(v->c, pub, bad, 32, r, fb, s, fb)) return fail("wrong hash accepted");
        uint8_t s2[66];
        memcpy(s2, s, fb);
        s2[fb - 1] ^= 1;
        if (ecdsa_verify(v->c, pub, hash, 32, r, fb, s2, fb)) return fail("flipped s accepted");
        uint8_t zero[66] = {0};
        if (ecdsa_verify(v->c, pub, hash, 32, zero, fb, s, fb)) return fail("r = 0 accepted");
        uint8_t pub2[132];
        memcpy(pub2, pub, 2 * fb);
        pub2[fb + 3] ^= 1;   // a point off the curve
        if (ecdsa_verify(v->c, pub2, hash, 32, r, fb, s, fb)) return fail("off-curve key accepted");
        printf("ecdsa: %s RFC 6979 vector ok\n", v->name);
    }
    // DER signature: SEQUENCE { INTEGER r, INTEGER s } for the P-256 vector;
    // r's top bit is set so it carries a leading zero byte
    {
        uint8_t r[32], s[32], der[80];
        unhex(VECS[0].r, r, 32);
        unhex(VECS[0].s, s, 32);
        size_t k = 0;
        der[k++] = 0x30; der[k++] = 0x46;
        der[k++] = 0x02; der[k++] = 0x21; der[k++] = 0x00; memcpy(der + k, r, 32); k += 32;
        der[k++] = 0x02; der[k++] = 0x21; der[k++] = 0x00; memcpy(der + k, s, 32); k += 32;
        uint8_t ro[32], so[32];
        if (!ecdsa_der_sig_parse(der, k, 32, ro, so)) return fail("der sig parse");
        if (memcmp(ro, r, 32) || memcmp(so, s, 32)) return fail("der sig values");
        der[4] = 0x01;   // padding byte that is not zero: non-minimal / negative
        if (ecdsa_der_sig_parse(der, k, 32, ro, so)) return fail("bad der accepted");
        der[4] = 0x00;
        if (ecdsa_der_sig_parse(der, k - 1, 32, ro, so)) return fail("truncated der accepted");
    }
    // SPKI for the P-256 key: 30 59 30 13 06 07 <ecPublicKey> 06 08 <p256> 03 42 00 04 X Y
    {
        uint8_t pub[64], spki[91];
        unhex(VECS[0].ux, pub, 32);
        unhex(VECS[0].uy, pub + 32, 32);
        static const uint8_t head[] = { 0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86,
            0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d,
            0x03, 0x01, 0x07, 0x03, 0x42, 0x00, 0x04 };
        memcpy(spki, head, sizeof head);
        memcpy(spki + sizeof head, pub, 64);
        ec_curve c;
        uint8_t out[132];
        size_t outn = 0;
        if (!ecdsa_spki_parse(spki, sizeof head + 64, &c, out, &outn)) return fail("spki parse");
        if (c != EC_P256 || outn != 64 || memcmp(out, pub, 64)) return fail("spki values");
        spki[26] = 0x02;   // compressed point marker
        if (ecdsa_spki_parse(spki, sizeof head + 64, &c, out, &outn)) return fail("compressed accepted");
    }
    printf("ecdsa: ok (RFC 6979 P-256/P-384/P-521, forgeries and malformed DER rejected)\n");
    return 0;
}
