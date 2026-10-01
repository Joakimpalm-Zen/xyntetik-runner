// ECDSA known-answer test against RFC 6979 appendix A.2.5-A.2.7: the P-256,
// P-384 and P-521 key pairs and their SHA-256 signatures over "sample". The
// hash of "sample" is computed here from the FIPS 180-4 value so the
// verification half does not depend on the envelope module. Also gates: r or
// s out of range, a flipped signature byte, a wrong hash, and the DER parsers
// (the signature SEQUENCE form OMS bundles carry, and an SPKI public key).
//
// The signing half (R1.2.5, `--sign-model`) is held to the same appendix:
// from the private key x, the deterministic signature over "sample" and
// "test" must be the RFC's r and s exactly, with the curve's own digest
// (SHA-256/384/512) and with SHA-256 on the larger curves (a digest shorter
// than the order). The key encoders and parsers round-trip against
// hand-built DER in the layouts openssl writes (SEC1, PKCS#8, SPKI).
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

typedef struct {
    ec_curve c;
    const char *name, *x, *msg, *r, *s;
    ec_hash h;
} sig_vec;

// RFC 6979 A.2.5 (P-256), A.2.6 (P-384), A.2.7 (P-521)
static const char X256[] = "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721";
static const char X384[] = "6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D896D5724E4C70A825F872C9EA60D2EDF5";
static const char X521[] = "00FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538";

static const sig_vec SIGS[] = {
    { EC_P256, "P-256/SHA-256 sample", X256, "sample",
      "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
      "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8", EC_SHA256 },
    { EC_P256, "P-256/SHA-256 test", X256, "test",
      "F1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367",
      "019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083", EC_SHA256 },
    { EC_P384, "P-384/SHA-384 sample", X384, "sample",
      "94EDBB92A5ECB8AAD4736E56C691916B3F88140666CE9FA73D64C4EA95AD133C81A648152E44ACF96E36DD1E80FABE46",
      "99EF4AEB15F178CEA1FE40DB2603138F130E740A19624526203B6351D0A3A94FA329C145786E679E7B82C71A38628AC8", EC_SHA384 },
    { EC_P384, "P-384/SHA-384 test", X384, "test",
      "8203B63D3C853E8D77227FB377BCF7B7B772E97892A80F36AB775D509D7A5FEB0542A7F0812998DA8F1DD3CA3CF023DB",
      "DDD0760448D42D8A43AF45AF836FCE4DE8BE06B485E9B61B827C2F13173923E06A739F040649A667BF3B828246BAA5A5", EC_SHA384 },
    { EC_P384, "P-384/SHA-256 sample", X384, "sample",
      "21B13D1E013C7FA1392D03C5F99AF8B30C570C6F98D4EA8E354B63A21D3DAA33BDE1E888E63355D92FA2B3C36D8FB2CD",
      "F3AA443FB107745BF4BD77CB3891674632068A10CA67E3D45DB2266FA7D1FEEBEFDC63ECCD1AC42EC0CB8668A4FA0AB0", EC_SHA256 },
    { EC_P521, "P-521/SHA-512 sample", X521, "sample",
      "00C328FAFCBD79DD77850370C46325D987CB525569FB63C5D3BC53950E6D4C5F174E25A1EE9017B5D450606ADD152B534931D7D4E8455CC91F9B15BF05EC36E377FA",
      "00617CCE7CF5064806C467F678D3B4080D6F1CC50AF26CA209417308281B68AF282623EAA63E5B5C0723D8B8C37FF0777B1A20F8CCB1DCCC43997F1EE0E44DA4A67A", EC_SHA512 },
    { EC_P521, "P-521/SHA-512 test", X521, "test",
      "013E99020ABF5CEE7525D16B69B229652AB6BDF2AFFCAEF38773B4B7D08725F10CDB93482FDCC54EDCEE91ECA4166B2A7C6265EF0CE2BD7051B7CEF945BABD47EE6D",
      "01FBD0013C674AA79CB39849527916CE301C66EA7CE8B80682786AD60F98F7E78A19CA69EFF5C57400E3B3A0AD66CE0978214D13BAF4E9AC60752F7B155E2DE4DCE3", EC_SHA512 },
    { EC_P521, "P-521/SHA-256 sample", X521, "sample",
      "01511BB4D675114FE266FC4372B87682BAECC01D3CC62CF2303C92B3526012659D16876E25C7C1E57648F23B73564D67F61C6F14D527D54972810421E7D87589E1A7",
      "004A171143A83163D6DF460AAF61522695F207A58B95C0644D87E52AA1A347916E4F7A72930B1BC06DBE22CE3F58264AFD23704CBB63B29B931F7DE6C9D949A7ECFC", EC_SHA256 },
};

static int test_signing(void) {
    for (size_t i = 0; i < sizeof SIGS / sizeof SIGS[0]; i++) {
        const sig_vec *v = &SIGS[i];
        size_t fb = ec_field_bytes(v->c);
        uint8_t x[66], r[66], s[66], ro[66], so[66], h[64];
        unhex(v->x, x, fb);
        unhex(v->r, r, fb);
        unhex(v->s, s, fb);
        ec_digest(v->h, v->msg, strlen(v->msg), h);
        if (!ecdsa_sign(v->c, x, v->h, h, ro, so)) return fail(v->name);
        if (memcmp(ro, r, fb) || memcmp(so, s, fb)) return fail(v->name);
        printf("ecdsa: sign %s = RFC 6979\n", v->name);
    }
    // the public key from the private one is appendix A's U
    for (size_t i = 0; i < sizeof VECS / sizeof VECS[0]; i++) {
        const vec *v = &VECS[i];
        size_t fb = ec_field_bytes(v->c);
        uint8_t x[66], pub[132], want[132];
        unhex(v->c == EC_P256 ? X256 : v->c == EC_P384 ? X384 : X521, x, fb);
        unhex(v->ux, want, fb);
        unhex(v->uy, want + fb, fb);
        if (!ecdsa_public_from_private(v->c, x, pub) || memcmp(pub, want, 2 * fb))
            return fail("public key from private");
    }
    // the scalar widening (k + 2n when k + n is short, which the appendix
    // keys never reach) and the pass through infinity it takes for d = 1:
    // 1*G is the generator and (n-1)*G its negation, (Gx, p - Gy), from the
    // curve definitions in FIPS 186-5 / SEC 2
    {
        static const struct { ec_curve c; const char *nm1, *gx, *gy, *ngy; } G[] = {
            { EC_P256, "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632550",
              "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
              "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
              "B01CBD1C01E58065711814B583F061E9D431CCA994CEA1313449BF97C840AE0A" },
            { EC_P521, "01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6FB71E91386408",
              "00C6858E06B70404E9CD9E3ECB662395B4429C648139053FB521F828AF606B4D3DBAA14B5E77EFE75928FE1DC127A2FFA8DE3348B3C1856A429BF97E7E31C2E5BD66",
              "011839296A789A3BC0045C8A5FB42C7D1BD998F54449579B446817AFBD17273E662C97EE72995EF42640C550B9013FAD0761353C7086A272C24088BE94769FD16650",
              "00E7C6D6958765C43FFBA375A04BD382E426670ABBB6A864BB97E85042E8D8C199D368118D66A10BD9BF3AAF46FEC052F89ECAC38F795D8D3DBF77416B89602E99AF" },
        };
        for (size_t i = 0; i < 2; i++) {
            size_t fb = ec_field_bytes(G[i].c);
            uint8_t d[66] = {0}, pub[132], want[132];
            d[fb - 1] = 1;
            unhex(G[i].gx, want, fb);
            unhex(G[i].gy, want + fb, fb);
            if (!ecdsa_public_from_private(G[i].c, d, pub) || memcmp(pub, want, 2 * fb))
                return fail("1*G");
            unhex(G[i].nm1, d, fb);
            unhex(G[i].ngy, want + fb, fb);
            if (!ecdsa_public_from_private(G[i].c, d, pub) || memcmp(pub, want, 2 * fb))
                return fail("(n-1)*G");
        }
    }
    // out-of-range private keys are refused: 0 and the order itself
    {
        uint8_t zero[32] = {0}, h[32] = {1}, ro[32], so[32], n[32];
        unhex("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", n, 32);
        if (ecdsa_sign(EC_P256, zero, EC_SHA256, h, ro, so)) return fail("d = 0 accepted");
        if (ecdsa_sign(EC_P256, n, EC_SHA256, h, ro, so)) return fail("d = n accepted");
    }
    // DER: minimal encodings that the parser reads back. P-256 "sample" has
    // both top bits set (a 0x00 pad each: 0x46 content bytes); P-256 "test"
    // has s starting 0x01 (no pad); P-521's 66-byte scalars shed their
    // leading zero byte (r = 00 C3.. keeps a pad for its top bit, s = 00 61..
    // is 65 bytes).
    {
        uint8_t r[66], s[66], der[160], ro[66], so[66];
        unhex(SIGS[0].r, r, 32); unhex(SIGS[0].s, s, 32);
        size_t k = ecdsa_der_sig_encode(r, s, 32, der);
        if (k != 72 || der[0] != 0x30 || der[1] != 0x46 || der[3] != 0x21 || der[4] != 0)
            return fail("der encode (padded)");
        if (!ecdsa_der_sig_parse(der, k, 32, ro, so) || memcmp(ro, r, 32) || memcmp(so, s, 32))
            return fail("der round trip (padded)");
        unhex(SIGS[1].r, r, 32); unhex(SIGS[1].s, s, 32);
        k = ecdsa_der_sig_encode(r, s, 32, der);
        if (k != 71 || der[38] != 0x20 || der[39] != 0x01) return fail("der encode (unpadded)");
        if (!ecdsa_der_sig_parse(der, k, 32, ro, so) || memcmp(so, s, 32))
            return fail("der round trip (unpadded)");
        unhex(SIGS[5].r, r, 66); unhex(SIGS[5].s, s, 66);
        k = ecdsa_der_sig_encode(r, s, 66, der);
        if (der[1] != 0x81 || der[4] != 0x42 || der[72] != 0x41 || der[73] != 0x61)
            return fail("der encode (P-521)");
        if (!ecdsa_der_sig_parse(der, k, 66, ro, so) || memcmp(ro, r, 66) || memcmp(so, s, 66))
            return fail("der round trip (P-521)");
    }
    // SPKI: the encoder writes what the parser (and openssl) read, per curve
    for (size_t i = 0; i < sizeof VECS / sizeof VECS[0]; i++) {
        const vec *v = &VECS[i];
        size_t fb = ec_field_bytes(v->c);
        uint8_t pub[132], spki[200], out[132];
        unhex(v->ux, pub, fb);
        unhex(v->uy, pub + fb, fb);
        size_t k = ecdsa_spki_encode(v->c, pub, spki);
        ec_curve c;
        size_t outn = 0;
        if (!k || !ecdsa_spki_parse(spki, k, &c, out, &outn) || c != v->c ||
            outn != 2 * fb || memcmp(out, pub, 2 * fb))
            return fail("spki encode");
        if (v->c == EC_P256 && (k != 91 || spki[1] != 0x59)) return fail("spki P-256 length");
        if (v->c == EC_P521 && (k != 158 || spki[1] != 0x81 || spki[2] != 0x9b))
            return fail("spki P-521 length");
    }
    // private keys: SEC1 "EC PRIVATE KEY" (openssl ecparam -genkey) and
    // PKCS#8 "PRIVATE KEY" (openssl genpkey) for the P-256 appendix key
    {
        static const uint8_t oid_ecpk[] = { 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01 };
        static const uint8_t oid_p256[] = { 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07 };
        uint8_t x[32], pub[64], sec1[160], p8[200], d[66];
        unhex(X256, x, 32);
        unhex(VECS[0].ux, pub, 32);
        unhex(VECS[0].uy, pub + 32, 32);
        size_t k = 0;
        sec1[k++] = 0x30; sec1[k++] = 0x77;
        sec1[k++] = 0x02; sec1[k++] = 0x01; sec1[k++] = 0x01;
        sec1[k++] = 0x04; sec1[k++] = 0x20; memcpy(sec1 + k, x, 32); k += 32;
        sec1[k++] = 0xa0; sec1[k++] = 0x0a; memcpy(sec1 + k, oid_p256, 10); k += 10;
        sec1[k++] = 0xa1; sec1[k++] = 0x44; sec1[k++] = 0x03; sec1[k++] = 0x42;
        sec1[k++] = 0x00; sec1[k++] = 0x04; memcpy(sec1 + k, pub, 64); k += 64;
        if (k != 121) return fail("sec1 fixture");
        ec_curve c = EC_P521;
        if (!ecdsa_private_key_parse(sec1, k, &c, d) || c != EC_P256 || memcmp(d, x, 32))
            return fail("sec1 parse");
        // the embedded public key must be the private key's
        sec1[k - 1] ^= 1;
        if (ecdsa_private_key_parse(sec1, k, &c, d)) return fail("sec1 mismatched public key accepted");
        sec1[k - 1] ^= 1;
        // PKCS#8 wraps a SEC1 key without its [0] parameters
        size_t j = 0;
        p8[j++] = 0x30; p8[j++] = 0x81; p8[j++] = 0x87;
        p8[j++] = 0x02; p8[j++] = 0x01; p8[j++] = 0x00;
        p8[j++] = 0x30; p8[j++] = 0x13;
        memcpy(p8 + j, oid_ecpk, 9); j += 9;
        memcpy(p8 + j, oid_p256, 10); j += 10;
        p8[j++] = 0x04; p8[j++] = 0x6d;
        p8[j++] = 0x30; p8[j++] = 0x6b;
        memcpy(p8 + j, sec1 + 2, 3 + 34); j += 37;
        memcpy(p8 + j, sec1 + 2 + 3 + 34 + 12, 70); j += 70;
        if (j != 138) return fail("pkcs8 fixture");
        c = EC_P521;
        if (!ecdsa_private_key_parse(p8, j, &c, d) || c != EC_P256 || memcmp(d, x, 32))
            return fail("pkcs8 parse");
        p8[5] = 0x02;    // PrivateKeyInfo version 2 does not exist
        if (ecdsa_private_key_parse(p8, j, &c, d)) return fail("pkcs8 bad version accepted");
        p8[5] = 0x00;
        p8[33] = 0x02;   // nor does ECPrivateKey version 2 inside it
        if (ecdsa_private_key_parse(p8, j, &c, d)) return fail("sec1 bad version accepted");
    }
    return 0;
}

int main(void) {
    if (test_signing()) return 1;
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
    printf("ecdsa: ok (RFC 6979 P-256/P-384/P-521 sign and verify, forgeries and malformed DER rejected)\n");
    return 0;
}
