// ECDSA over P-256 / P-384 / P-521. See ecdsa.h for scope.
//
// Layout: big numbers are little-endian arrays of 32-bit limbs (17 limbs
// cover 521 bits); field and scalar arithmetic run in Montgomery form with
// a per-modulus context (the modulus, -p^-1 mod 2^32, R^2 mod p); points are
// Jacobian (X, Y, Z) with mixed addition against affine bases. Verification
// is not constant-time and need not be: every input is public. The signer's
// secret scalars go through jpt_mul_fixed (see there for what that covers).
#include "ecdsa.h"
#include "ed25519.h"
#include "envelope.h"
#include <string.h>

enum { L_MAX = 17 };   // limbs: 32*17 = 544 bits >= 521

typedef struct {
    uint32_t v[L_MAX];
} bn;

typedef struct {
    bn       p;      // the modulus
    bn       r2;     // R^2 mod p, R = 2^(32*L)
    bn       one;    // R mod p (the Montgomery form of 1)
    uint32_t n0;     // -p^-1 mod 2^32
    int      L;      // limbs in use
    int      bits;   // bit length of p
} mont;

typedef struct {
    ec_curve id;
    int      L, bits;
    const char *p, *a, *b, *gx, *gy, *n;
} curve_def;

static const curve_def CURVES[] = {
    { EC_P256, 8, 256,
      "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF",
      "FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC",
      "5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B",
      "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
      "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
      "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551" },
    { EC_P384, 12, 384,
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF",
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFC",
      "B3312FA7E23EE7E4988E056BE3F82D19181D9C6EFE8141120314088F5013875AC656398D8A2ED19D2A85C8EDD3EC2AEF",
      "AA87CA22BE8B05378EB1C71EF320AD746E1D3B628BA79B9859F741E082542A385502F25DBF55296C3A545E3872760AB7",
      "3617DE4A96262C6F5D9E98BF9292DC29F8F41DBD289A147CE9DA3113B5F0B8C00A60B1CE1D7E819D7A431D7C90EA0E5F",
      "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973" },
    { EC_P521, 17, 521,
      "01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
      "01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC",
      "0051953EB9618E1C9A1F929A21A0B68540EEA2DA725B99B315F3B8B489918EF109E156193951EC7E937B1652C0BD3BB1BF073573DF883D2C34F1EF451FD46B503F00",
      "00C6858E06B70404E9CD9E3ECB662395B4429C648139053FB521F828AF606B4D3DBAA14B5E77EFE75928FE1DC127A2FFA8DE3348B3C1856A429BF97E7E31C2E5BD66",
      "011839296A789A3BC0045C8A5FB42C7D1BD998F54449579B446817AFBD17273E662C97EE72995EF42640C550B9013FAD0761353C7086A272C24088BE94769FD16650",
      "01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6FB71E91386409" },
};

size_t ec_field_bytes(ec_curve c) {
    return c == EC_P256 ? 32 : c == EC_P384 ? 48 : 66;
}

ec_hash ec_curve_hash(ec_curve c) {
    return c == EC_P256 ? EC_SHA256 : c == EC_P384 ? EC_SHA384 : EC_SHA512;
}

size_t ec_hash_bytes(ec_hash h) {
    return h == EC_SHA256 ? 32 : h == EC_SHA384 ? 48 : 64;
}

void ec_digest(ec_hash h, const void *m, size_t n, uint8_t *out) {
    if (h == EC_SHA256) envelope_data_sha256_raw(m, n, out);
    else if (h == EC_SHA384) ed25519_sha384(out, m, n);
    else ed25519_sha512(out, m, n);
}

// ---- plain big-number helpers (L limbs, little-endian) ---------------------

static void bn_zero(bn *a) { memset(a, 0, sizeof *a); }

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool bn_from_hex(bn *a, const char *h) {
    bn_zero(a);
    size_t n = strlen(h);
    for (size_t i = 0; i < n; i++) {
        int v = hexval(h[n - 1 - i]);
        if (v < 0) return false;
        size_t limb = i / 8, shift = (i % 8) * 4;
        if (limb >= L_MAX) return false;
        a->v[limb] |= (uint32_t)v << shift;
    }
    return true;
}

// big-endian bytes -> bn (any length up to 4*L_MAX)
static bool bn_from_bytes(bn *a, const uint8_t *b, size_t n) {
    bn_zero(a);
    if (n > 4 * L_MAX) {
        // leading zeros beyond the limb capacity are fine, anything else is not
        size_t extra = n - 4 * L_MAX;
        for (size_t i = 0; i < extra; i++) if (b[i]) return false;
        b += extra; n -= extra;
    }
    for (size_t i = 0; i < n; i++) {
        size_t k = n - 1 - i;   // byte index from the least significant end
        a->v[i / 4] |= (uint32_t)b[k] << ((i % 4) * 8);
    }
    return true;
}

static int bn_cmp(const bn *a, const bn *b, int L) {
    for (int i = L - 1; i >= 0; i--) {
        if (a->v[i] != b->v[i]) return a->v[i] > b->v[i] ? 1 : -1;
    }
    return 0;
}

static bool bn_is_zero(const bn *a, int L) {
    for (int i = 0; i < L; i++) if (a->v[i]) return false;
    return true;
}

// r = a + b, returns the carry out of limb L-1
static uint32_t bn_add(bn *r, const bn *a, const bn *b, int L) {
    uint64_t c = 0;
    for (int i = 0; i < L; i++) {
        uint64_t s = (uint64_t)a->v[i] + b->v[i] + c;
        r->v[i] = (uint32_t)s;
        c = s >> 32;
    }
    return (uint32_t)c;
}

// r = a - b, returns the borrow out of limb L-1
static uint32_t bn_sub(bn *r, const bn *a, const bn *b, int L) {
    uint64_t br = 0;
    for (int i = 0; i < L; i++) {
        uint64_t s = (uint64_t)a->v[i] - b->v[i] - br;
        r->v[i] = (uint32_t)s;
        br = (s >> 32) & 1;
    }
    return (uint32_t)br;
}

static void bn_add_mod(bn *r, const bn *a, const bn *b, const bn *p, int L) {
    bn t;
    uint32_t c = bn_add(&t, a, b, L);
    if (c || bn_cmp(&t, p, L) >= 0) bn_sub(&t, &t, p, L);
    *r = t;
}

static void bn_sub_mod(bn *r, const bn *a, const bn *b, const bn *p, int L) {
    bn t;
    if (bn_sub(&t, a, b, L)) bn_add(&t, &t, p, L);
    *r = t;
}

static int bn_bits(const bn *a, int L) {
    for (int i = L - 1; i >= 0; i--) {
        if (a->v[i]) {
            int b = 0;
            uint32_t v = a->v[i];
            while (v) { b++; v >>= 1; }
            return i * 32 + b;
        }
    }
    return 0;
}

static int bn_bit(const bn *a, int i) {
    return (a->v[i / 32] >> (i % 32)) & 1;
}

// ---- Montgomery arithmetic --------------------------------------------------

static void mont_init(mont *m, const bn *p, int L, int bits) {
    m->p = *p;
    m->L = L;
    m->bits = bits;
    // n0 = -p^-1 mod 2^32 by Newton iteration (p is odd)
    uint32_t x = 1;
    for (int i = 0; i < 6; i++) x *= 2u - p->v[0] * x;
    m->n0 = 0u - x;
    // R mod p and R^2 mod p by repeated doubling of 1
    bn r;
    bn_zero(&r);
    r.v[0] = 1;
    for (int i = 0; i < 32 * L; i++) bn_add_mod(&r, &r, &r, p, L);
    m->one = r;
    for (int i = 0; i < 32 * L; i++) bn_add_mod(&r, &r, &r, p, L);
    m->r2 = r;
}

// r = a * b * R^-1 mod p (CIOS), inputs below p
static void mont_mul(bn *r, const bn *a, const bn *b, const mont *m) {
    const int L = m->L;
    uint32_t t[L_MAX + 2];
    memset(t, 0, sizeof t);
    for (int i = 0; i < L; i++) {
        uint64_t c = 0, s;
        for (int j = 0; j < L; j++) {
            s = (uint64_t)t[j] + (uint64_t)a->v[j] * b->v[i] + c;
            t[j] = (uint32_t)s;
            c = s >> 32;
        }
        s = (uint64_t)t[L] + c;
        t[L] = (uint32_t)s;
        t[L + 1] = (uint32_t)(s >> 32);
        uint32_t u = t[0] * m->n0;
        s = (uint64_t)t[0] + (uint64_t)u * m->p.v[0];
        c = s >> 32;
        for (int j = 1; j < L; j++) {
            s = (uint64_t)t[j] + (uint64_t)u * m->p.v[j] + c;
            t[j - 1] = (uint32_t)s;
            c = s >> 32;
        }
        s = (uint64_t)t[L] + c;
        t[L - 1] = (uint32_t)s;
        t[L] = t[L + 1] + (uint32_t)(s >> 32);
    }
    bn out;
    bn_zero(&out);
    for (int i = 0; i < L; i++) out.v[i] = t[i];
    if (t[L] || bn_cmp(&out, &m->p, L) >= 0) bn_sub(&out, &out, &m->p, L);
    *r = out;
}

static void mont_to(bn *r, const bn *a, const mont *m) { mont_mul(r, a, &m->r2, m); }

static void mont_from(bn *r, const bn *a, const mont *m) {
    bn one;
    bn_zero(&one);
    one.v[0] = 1;
    mont_mul(r, a, &one, m);
}

// r = a^e mod p (Montgomery form in and out), plain square-and-multiply
static void mont_pow(bn *r, const bn *a, const bn *e, const mont *m) {
    bn acc = m->one, base = *a;
    int nb = bn_bits(e, m->L);
    for (int i = nb - 1; i >= 0; i--) {
        mont_mul(&acc, &acc, &acc, m);
        if (bn_bit(e, i)) mont_mul(&acc, &acc, &base, m);
    }
    *r = acc;
}

// r = a^-1 mod p via Fermat (p prime): a^(p-2)
static void mont_inv(bn *r, const bn *a, const mont *m) {
    bn e, two;
    bn_zero(&two);
    two.v[0] = 2;
    bn_sub(&e, &m->p, &two, m->L);
    mont_pow(r, a, &e, m);
}

// ---- Jacobian point arithmetic (Montgomery-form coordinates) --------------

typedef struct { bn x, y, z; bool inf; } jpt;
typedef struct { bn x, y; } apt;   // affine, Montgomery form

static void jpt_double(jpt *r, const jpt *p, const bn *a_m, const mont *m) {
    if (p->inf || bn_is_zero(&p->y, m->L)) { r->inf = true; return; }
    bn xx, yy, yyyy, zz, s, mm, t, x3, y3, z3;
    mont_mul(&xx, &p->x, &p->x, m);
    mont_mul(&yy, &p->y, &p->y, m);
    mont_mul(&yyyy, &yy, &yy, m);
    mont_mul(&zz, &p->z, &p->z, m);
    // S = 4*X*YY
    mont_mul(&s, &p->x, &yy, m);
    bn_add_mod(&s, &s, &s, &m->p, m->L);
    bn_add_mod(&s, &s, &s, &m->p, m->L);
    // M = 3*XX + a*ZZ^2
    bn_add_mod(&mm, &xx, &xx, &m->p, m->L);
    bn_add_mod(&mm, &mm, &xx, &m->p, m->L);
    mont_mul(&t, &zz, &zz, m);
    mont_mul(&t, &t, a_m, m);
    bn_add_mod(&mm, &mm, &t, &m->p, m->L);
    // X3 = M^2 - 2S
    mont_mul(&x3, &mm, &mm, m);
    bn_sub_mod(&x3, &x3, &s, &m->p, m->L);
    bn_sub_mod(&x3, &x3, &s, &m->p, m->L);
    // Y3 = M*(S - X3) - 8*YYYY
    bn_sub_mod(&t, &s, &x3, &m->p, m->L);
    mont_mul(&y3, &mm, &t, m);
    bn_add_mod(&t, &yyyy, &yyyy, &m->p, m->L);
    bn_add_mod(&t, &t, &t, &m->p, m->L);
    bn_add_mod(&t, &t, &t, &m->p, m->L);
    bn_sub_mod(&y3, &y3, &t, &m->p, m->L);
    // Z3 = 2*Y*Z
    mont_mul(&z3, &p->y, &p->z, m);
    bn_add_mod(&z3, &z3, &z3, &m->p, m->L);
    r->x = x3; r->y = y3; r->z = z3; r->inf = false;
}

// r = p + q with q affine (mixed addition)
static void jpt_madd(jpt *r, const jpt *p, const apt *q, const bn *a_m,
                     const mont *m) {
    if (p->inf) { r->x = q->x; r->y = q->y; r->z = m->one; r->inf = false; return; }
    bn zz, u2, s2, h, rr, hh, hhh, t, x3, y3, z3;
    mont_mul(&zz, &p->z, &p->z, m);
    mont_mul(&u2, &q->x, &zz, m);
    mont_mul(&s2, &q->y, &zz, m);
    mont_mul(&s2, &s2, &p->z, m);
    bn_sub_mod(&h, &u2, &p->x, &m->p, m->L);
    bn_sub_mod(&rr, &s2, &p->y, &m->p, m->L);
    if (bn_is_zero(&h, m->L)) {
        if (bn_is_zero(&rr, m->L)) { jpt_double(r, p, a_m, m); return; }
        r->inf = true;
        return;
    }
    mont_mul(&hh, &h, &h, m);
    mont_mul(&hhh, &hh, &h, m);
    mont_mul(&t, &p->x, &hh, m);        // X1*HH
    // X3 = R^2 - HHH - 2*X1*HH
    mont_mul(&x3, &rr, &rr, m);
    bn_sub_mod(&x3, &x3, &hhh, &m->p, m->L);
    bn_sub_mod(&x3, &x3, &t, &m->p, m->L);
    bn_sub_mod(&x3, &x3, &t, &m->p, m->L);
    // Y3 = R*(X1*HH - X3) - Y1*HHH
    bn_sub_mod(&t, &t, &x3, &m->p, m->L);
    mont_mul(&y3, &rr, &t, m);
    mont_mul(&t, &p->y, &hhh, m);
    bn_sub_mod(&y3, &y3, &t, &m->p, m->L);
    // Z3 = Z1*H
    mont_mul(&z3, &p->z, &h, m);
    r->x = x3; r->y = y3; r->z = z3; r->inf = false;
}

// r = k * q (q affine), double-and-add from the top bit
static void jpt_mul(jpt *r, const bn *k, const apt *q, const bn *a_m,
                    const mont *m) {
    jpt acc = { .inf = true };
    int nb = bn_bits(k, m->L);
    for (int i = nb - 1; i >= 0; i--) {
        jpt_double(&acc, &acc, a_m, m);
        if (bn_bit(k, i)) jpt_madd(&acc, &acc, q, a_m, m);
    }
    *r = acc;
}

// Jacobian -> affine (Montgomery form); false at infinity
static bool jpt_to_affine(apt *r, const jpt *p, const mont *m) {
    if (p->inf) return false;
    bn zi, zi2, zi3;
    mont_inv(&zi, &p->z, m);
    mont_mul(&zi2, &zi, &zi, m);
    mont_mul(&zi3, &zi2, &zi, m);
    mont_mul(&r->x, &p->x, &zi2, m);
    mont_mul(&r->y, &p->y, &zi3, m);
    return true;
}

// ---- ECDSA ------------------------------------------------------------------

static bool on_curve(const apt *q, const bn *a_m, const bn *b_m, const mont *m) {
    // y^2 == x^3 + a*x + b (Montgomery form)
    bn lhs, rhs, t;
    mont_mul(&lhs, &q->y, &q->y, m);
    mont_mul(&t, &q->x, &q->x, m);
    mont_mul(&rhs, &t, &q->x, m);
    mont_mul(&t, a_m, &q->x, m);
    bn_add_mod(&rhs, &rhs, &t, &m->p, m->L);
    bn_add_mod(&rhs, &rhs, b_m, &m->p, m->L);
    return bn_cmp(&lhs, &rhs, m->L) == 0;
}

bool ecdsa_verify(ec_curve c, const uint8_t *pub, const uint8_t *hash,
                  size_t hash_len, const uint8_t *r, size_t r_len,
                  const uint8_t *s, size_t s_len) {
    if (c < EC_P256 || c > EC_P521 || !pub || !hash || !r || !s) return false;
    const curve_def *cd = &CURVES[c];
    const int L = cd->L;
    const size_t fb = ec_field_bytes(c);
    bn p, a, b, gx, gy, n, rr, ss, qx, qy, e;
    if (!bn_from_hex(&p, cd->p) || !bn_from_hex(&a, cd->a) ||
        !bn_from_hex(&b, cd->b) || !bn_from_hex(&gx, cd->gx) ||
        !bn_from_hex(&gy, cd->gy) || !bn_from_hex(&n, cd->n))
        return false;
    if (!bn_from_bytes(&rr, r, r_len) || !bn_from_bytes(&ss, s, s_len)) return false;
    if (!bn_from_bytes(&qx, pub, fb) || !bn_from_bytes(&qy, pub + fb, fb)) return false;
    // 1 <= r, s < n
    if (bn_is_zero(&rr, L) || bn_is_zero(&ss, L) ||
        bn_cmp(&rr, &n, L) >= 0 || bn_cmp(&ss, &n, L) >= 0)
        return false;
    // the public point must be a proper affine point on the curve
    if (bn_cmp(&qx, &p, L) >= 0 || bn_cmp(&qy, &p, L) >= 0) return false;
    // e = leftmost bits(n) bits of the hash
    {
        uint8_t hb[4 * L_MAX];
        size_t hl = hash_len > sizeof hb ? sizeof hb : hash_len;
        memcpy(hb, hash, hl);
        if (!bn_from_bytes(&e, hb, hl)) return false;
        int nbits = bn_bits(&n, L);
        int hbits = (int)hl * 8;
        if (hbits > nbits) {
            int shift = hbits - nbits;
            // shift right by `shift` bits
            for (int k = 0; k < shift; k++) {
                uint32_t carry = 0;
                for (int i = L_MAX - 1; i >= 0; i--) {
                    uint32_t nc = e.v[i] & 1;
                    e.v[i] = (e.v[i] >> 1) | (carry << 31);
                    carry = nc;
                }
            }
        }
        // reduce mod n once (e may equal or exceed n only when hbits == nbits)
        if (bn_cmp(&e, &n, L) >= 0) bn_sub(&e, &e, &n, L);
    }
    mont mp, mn;
    mont_init(&mp, &p, L, cd->bits);
    mont_init(&mn, &n, L, cd->bits);
    bn a_m, b_m;
    mont_to(&a_m, &a, &mp);
    mont_to(&b_m, &b, &mp);
    apt G, Q;
    mont_to(&G.x, &gx, &mp);
    mont_to(&G.y, &gy, &mp);
    mont_to(&Q.x, &qx, &mp);
    mont_to(&Q.y, &qy, &mp);
    if (!on_curve(&Q, &a_m, &b_m, &mp)) return false;
    // w = s^-1 mod n; u1 = e*w; u2 = r*w  (scalar arithmetic in Montgomery form)
    bn s_m, w_m, e_m, r_m, u1_m, u2_m, u1, u2;
    mont_to(&s_m, &ss, &mn);
    mont_inv(&w_m, &s_m, &mn);
    mont_to(&e_m, &e, &mn);
    mont_to(&r_m, &rr, &mn);
    mont_mul(&u1_m, &e_m, &w_m, &mn);
    mont_mul(&u2_m, &r_m, &w_m, &mn);
    mont_from(&u1, &u1_m, &mn);
    mont_from(&u2, &u2_m, &mn);
    // R = u1*G + u2*Q
    jpt R1, R2, R;
    jpt_mul(&R1, &u1, &G, &a_m, &mp);
    jpt_mul(&R2, &u2, &Q, &a_m, &mp);
    if (R1.inf) {
        R = R2;
    } else {
        apt A1;
        jpt_to_affine(&A1, &R1, &mp);
        jpt_madd(&R, &R2, &A1, &a_m, &mp);
    }
    apt RA;
    if (!jpt_to_affine(&RA, &R, &mp)) return false;
    bn xr;
    mont_from(&xr, &RA.x, &mp);
    // v = x_R mod n
    if (bn_cmp(&xr, &n, L) >= 0) bn_sub(&xr, &xr, &n, L);
    return bn_cmp(&xr, &rr, L) == 0;
}

// ---- DER helpers --------------------------------------------------------------

// Reads one TLV header; returns the content offset, sets *len, or 0 on fault.
static size_t der_tlv(const uint8_t *d, size_t n, size_t off, uint8_t tag,
                      size_t *len) {
    if (off + 2 > n || d[off] != tag) return 0;
    size_t l = d[off + 1], hdr = 2;
    if (l & 0x80) {
        size_t nb = l & 0x7f;
        if (nb == 0 || nb > 2 || off + 2 + nb > n) return 0;
        l = 0;
        for (size_t i = 0; i < nb; i++) l = (l << 8) | d[off + 2 + i];
        if (l < 0x80) return 0;   // non-minimal length
        hdr = 2 + nb;
    }
    if (off + hdr + l > n) return 0;
    *len = l;
    return off + hdr;
}

// INTEGER -> big-endian scalar of exactly n bytes
static bool der_uint(const uint8_t *d, size_t n_total, size_t *off, size_t n,
                     uint8_t *out) {
    size_t len, c = der_tlv(d, n_total, *off, 0x02, &len);
    if (!c || len == 0) return false;
    const uint8_t *v = d + c;
    size_t vlen = len;
    if (v[0] & 0x80) return false;                        // negative
    if (vlen > 1 && v[0] == 0 && !(v[1] & 0x80)) return false; // non-minimal
    if (vlen > 1 && v[0] == 0) { v++; vlen--; }         // sign pad
    if (vlen > n) return false;
    memset(out, 0, n);
    memcpy(out + (n - vlen), v, vlen);
    *off = c + len;
    return true;
}

bool ecdsa_der_sig_parse(const uint8_t *der, size_t der_len, size_t n,
                         uint8_t *r_out, uint8_t *s_out) {
    size_t len, c = der_tlv(der, der_len, 0, 0x30, &len);
    if (!c || c + len != der_len) return false;
    size_t off = c;
    if (!der_uint(der, der_len, &off, n, r_out)) return false;
    if (!der_uint(der, der_len, &off, n, s_out)) return false;
    return off == der_len;
}

static const uint8_t OID_EC_PUBLIC_KEY[] = { 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01 };
static const uint8_t OID_P256[] = { 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07 };
static const uint8_t OID_P384[] = { 0x2b, 0x81, 0x04, 0x00, 0x22 };
static const uint8_t OID_P521[] = { 0x2b, 0x81, 0x04, 0x00, 0x23 };

bool ecdsa_spki_parse(const uint8_t *der, size_t der_len, ec_curve *curve_out,
                      uint8_t *pub_out, size_t *pub_len_out) {
    size_t len, c = der_tlv(der, der_len, 0, 0x30, &len);
    if (!c || c + len != der_len) return false;
    size_t alen, a = der_tlv(der, der_len, c, 0x30, &alen);
    if (!a) return false;
    size_t olen, o = der_tlv(der, der_len, a, 0x06, &olen);
    if (!o || olen != sizeof OID_EC_PUBLIC_KEY ||
        memcmp(der + o, OID_EC_PUBLIC_KEY, olen) != 0)
        return false;
    size_t clen, co = der_tlv(der, der_len, o + olen, 0x06, &clen);
    if (!co || co + clen != a + alen) return false;
    ec_curve curve;
    if (clen == sizeof OID_P256 && !memcmp(der + co, OID_P256, clen)) curve = EC_P256;
    else if (clen == sizeof OID_P384 && !memcmp(der + co, OID_P384, clen)) curve = EC_P384;
    else if (clen == sizeof OID_P521 && !memcmp(der + co, OID_P521, clen)) curve = EC_P521;
    else return false;
    size_t blen, b = der_tlv(der, der_len, a + alen, 0x03, &blen);
    if (!b || b + blen != der_len) return false;
    size_t fb = ec_field_bytes(curve);
    // BIT STRING: unused-bits byte (0), then 0x04 || X || Y
    if (blen != 2 + 2 * fb || der[b] != 0 || der[b + 1] != 0x04) return false;
    memcpy(pub_out, der + b + 2, 2 * fb);
    *pub_len_out = 2 * fb;
    *curve_out = curve;
    return true;
}

// ---- signing (R1.2.5) ---------------------------------------------------------

// big-endian n bytes <- bn
static void bn_to_bytes(const bn *a, uint8_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        size_t k = n - 1 - i;
        out[k] = i / 4 < L_MAX ? (uint8_t)(a->v[i / 4] >> ((i % 4) * 8)) : 0;
    }
}

// r = bit ? a : r, without a branch on bit
static void bn_cmov(bn *r, const bn *a, uint32_t bit) {
    uint32_t m = 0u - bit;
    for (int i = 0; i < L_MAX; i++) r->v[i] = (r->v[i] & ~m) | (a->v[i] & m);
}

// One curve's constants in the forms the arithmetic needs.
typedef struct {
    const curve_def *cd;
    int  L, nbits;
    size_t fb;
    bn   p, n;
    mont mp, mn;
    bn   a_m, b_m;
    apt  G;
} curve_ctx;

static bool curve_load(ec_curve c, curve_ctx *x) {
    if (c < EC_P256 || c > EC_P521) return false;
    const curve_def *cd = &CURVES[c];
    bn a, b, gx, gy;
    if (!bn_from_hex(&x->p, cd->p) || !bn_from_hex(&a, cd->a) ||
        !bn_from_hex(&b, cd->b) || !bn_from_hex(&gx, cd->gx) ||
        !bn_from_hex(&gy, cd->gy) || !bn_from_hex(&x->n, cd->n))
        return false;
    x->cd = cd;
    x->L = cd->L;
    x->fb = ec_field_bytes(c);
    x->nbits = bn_bits(&x->n, cd->L);
    mont_init(&x->mp, &x->p, cd->L, cd->bits);
    mont_init(&x->mn, &x->n, cd->L, cd->bits);
    mont_to(&x->a_m, &a, &x->mp);
    mont_to(&x->b_m, &b, &x->mp);
    mont_to(&x->G.x, &gx, &x->mp);
    mont_to(&x->G.y, &gy, &x->mp);
    return true;
}

// r = k*G for a secret k in [1, n). The scalar is first widened to k + n or
// k + 2n, whichever has exactly nbits+1 bits (the choice made by a masked
// select), so every call starts from the same top bit and runs the same
// sequence: nbits doublings, each followed by an addition whose result is
// kept or dropped by a masked select. The point operations therefore do not
// depend on k's bits; the field arithmetic's final conditional subtraction
// still does (see ecdsa.h).
static void jpt_mul_fixed(jpt *r, const bn *k, const curve_ctx *x) {
    // L_MAX limbs hold 3n for every curve (P-521: 523 bits < 544)
    bn k1, k2;
    bn_add(&k1, k, &x->n, L_MAX);
    bn_add(&k2, &k1, &x->n, L_MAX);
    bn_cmov(&k1, &k2, (uint32_t)(1 - bn_bit(&k1, x->nbits)));
    jpt acc = { .x = x->G.x, .y = x->G.y, .z = x->mp.one, .inf = false };
    for (int i = x->nbits - 1; i >= 0; i--) {
        jpt d = acc, t;
        jpt_double(&d, &acc, &x->a_m, &x->mp);
        jpt_madd(&t, &d, &x->G, &x->a_m, &x->mp);
        uint32_t bit = (uint32_t)bn_bit(&k1, i);
        bn_cmov(&d.x, &t.x, bit);
        bn_cmov(&d.y, &t.y, bit);
        bn_cmov(&d.z, &t.z, bit);
        // at infinity only on the way to k = 1 (prefix n of 2n + 1)
        d.inf = (bool)(((uint32_t)d.inf & (bit ^ 1u)) | ((uint32_t)t.inf & bit));
        acc = d;
    }
    *r = acc;
}

static bool scalar_ok(const bn *d, const curve_ctx *x) {
    return !bn_is_zero(d, x->L) && bn_cmp(d, &x->n, x->L) < 0;
}

// bits2int (RFC 6979 2.3.2): the leftmost nbits bits of b as an integer
static void bits2int(bn *out, const uint8_t *b, size_t blen, const curve_ctx *x) {
    size_t rlen = ((size_t)x->nbits + 7) / 8;
    if (blen * 8 <= (size_t)x->nbits) {
        bn_from_bytes(out, b, blen);
        return;
    }
    bn_from_bytes(out, b, rlen);
    int shift = (int)(rlen * 8) - x->nbits;   // < 8
    if (shift > 0) {
        for (int i = 0; i < L_MAX; i++) {
            uint32_t hi = i + 1 < L_MAX ? out->v[i + 1] : 0;
            out->v[i] = (out->v[i] >> shift) | (hi << (32 - shift));
        }
    }
}

// HMAC (RFC 2104) with the ec_hash; RFC 6979's keys are hlen bytes and its
// messages are at most V || 0x01 || x || h1 (64 + 1 + 66 + 66 bytes)
static void hmac(ec_hash h, const uint8_t *key, const uint8_t *msg, size_t n,
                 uint8_t *out) {
    size_t hl = ec_hash_bytes(h), bl = h == EC_SHA256 ? 64 : 128;
    uint8_t buf[128 + 256], inner[64];
    if (n > 256) n = 256;
    for (size_t i = 0; i < bl; i++) buf[i] = (uint8_t)((i < hl ? key[i] : 0) ^ 0x36);
    memcpy(buf + bl, msg, n);
    ec_digest(h, buf, bl + n, inner);
    for (size_t i = 0; i < bl; i++) buf[i] = (uint8_t)((i < hl ? key[i] : 0) ^ 0x5c);
    memcpy(buf + bl, inner, hl);
    ec_digest(h, buf, bl + hl, out);
    memset(buf, 0, sizeof buf);
    memset(inner, 0, sizeof inner);
}

// K = HMAC_K(V || sep || rest); V = HMAC_K(V)
static void drbg_step(ec_hash h, uint8_t *K, uint8_t *V, int sep,
                      const uint8_t *rest, size_t rest_n) {
    size_t hl = ec_hash_bytes(h);
    uint8_t m[64 + 1 + 2 * 66];
    size_t k = 0;
    memcpy(m, V, hl); k = hl;
    if (sep >= 0) m[k++] = (uint8_t)sep;
    if (rest_n) { memcpy(m + k, rest, rest_n); k += rest_n; }
    hmac(h, K, m, k, K);
    hmac(h, K, V, hl, V);
    memset(m, 0, sizeof m);
}

bool ecdsa_sign(ec_curve c, const uint8_t *priv, ec_hash h,
                const uint8_t *digest, uint8_t *r_out, uint8_t *s_out) {
    if (!priv || !digest || !r_out || !s_out || h < EC_SHA256 || h > EC_SHA512)
        return false;
    curve_ctx x;
    if (!curve_load(c, &x)) return false;
    const int L = x.L;
    const size_t fb = x.fb, hl = ec_hash_bytes(h);
    const size_t rlen = ((size_t)x.nbits + 7) / 8;   // == fb on these curves
    bn d, e;
    if (!bn_from_bytes(&d, priv, fb) || !scalar_ok(&d, &x)) return false;
    // e = bits2int(h1) mod n; bits2octets(h1) is its rlen-byte encoding
    bits2int(&e, digest, hl, &x);
    if (bn_cmp(&e, &x.n, L) >= 0) bn_sub(&e, &e, &x.n, L);
    uint8_t seed[2 * 66];                 // int2octets(x) || bits2octets(h1)
    bn_to_bytes(&d, seed, rlen);
    bn_to_bytes(&e, seed + rlen, rlen);
    uint8_t V[64], K[64];
    memset(V, 0x01, hl);
    memset(K, 0x00, hl);
    drbg_step(h, K, V, 0x00, seed, 2 * rlen);
    drbg_step(h, K, V, 0x01, seed, 2 * rlen);
    bool ok = false;
    for (int attempt = 0; attempt < 64 && !ok; attempt++) {
        if (attempt) drbg_step(h, K, V, 0x00, NULL, 0);
        uint8_t T[2 * 64];
        size_t tlen = 0;
        while (tlen < rlen) {
            hmac(h, K, V, hl, V);
            memcpy(T + tlen, V, hl);
            tlen += hl;
        }
        bn k;
        bits2int(&k, T, tlen, &x);
        memset(T, 0, sizeof T);
        if (!scalar_ok(&k, &x)) continue;
        // r = x(kG) mod n
        jpt R;
        apt RA;
        jpt_mul_fixed(&R, &k, &x);
        if (!jpt_to_affine(&RA, &R, &x.mp)) continue;
        bn rr;
        mont_from(&rr, &RA.x, &x.mp);
        if (bn_cmp(&rr, &x.n, L) >= 0) bn_sub(&rr, &rr, &x.n, L);
        if (bn_is_zero(&rr, L)) continue;
        // s = k^-1 (e + r*d) mod n
        bn k_m, ki_m, r_m, d_m, e_m, t, s_m, ss;
        mont_to(&k_m, &k, &x.mn);
        mont_inv(&ki_m, &k_m, &x.mn);
        mont_to(&r_m, &rr, &x.mn);
        mont_to(&d_m, &d, &x.mn);
        mont_to(&e_m, &e, &x.mn);
        mont_mul(&t, &r_m, &d_m, &x.mn);
        bn_add_mod(&t, &t, &e_m, &x.n, L);
        mont_mul(&s_m, &ki_m, &t, &x.mn);
        mont_from(&ss, &s_m, &x.mn);
        memset(&k, 0, sizeof k);
        memset(&k_m, 0, sizeof k_m);
        memset(&ki_m, 0, sizeof ki_m);
        memset(&d_m, 0, sizeof d_m);
        if (bn_is_zero(&ss, L)) continue;
        bn_to_bytes(&rr, r_out, fb);
        bn_to_bytes(&ss, s_out, fb);
        ok = true;
    }
    memset(&d, 0, sizeof d);
    memset(seed, 0, sizeof seed);
    memset(K, 0, sizeof K);
    memset(V, 0, sizeof V);
    return ok;
}

bool ecdsa_public_from_private(ec_curve c, const uint8_t *priv, uint8_t *pub_out) {
    curve_ctx x;
    if (!priv || !pub_out || !curve_load(c, &x)) return false;
    bn d;
    if (!bn_from_bytes(&d, priv, x.fb) || !scalar_ok(&d, &x)) return false;
    jpt Q;
    apt QA;
    jpt_mul_fixed(&Q, &d, &x);
    memset(&d, 0, sizeof d);
    if (!jpt_to_affine(&QA, &Q, &x.mp)) return false;
    bn qx, qy;
    mont_from(&qx, &QA.x, &x.mp);
    mont_from(&qy, &QA.y, &x.mp);
    bn_to_bytes(&qx, pub_out, x.fb);
    bn_to_bytes(&qy, pub_out + x.fb, x.fb);
    return true;
}

// DER length octets; returns how many were written
static size_t der_len_put(uint8_t *o, size_t len) {
    if (len < 0x80) { o[0] = (uint8_t)len; return 1; }
    if (len < 0x100) { o[0] = 0x81; o[1] = (uint8_t)len; return 2; }
    o[0] = 0x82; o[1] = (uint8_t)(len >> 8); o[2] = (uint8_t)len;
    return 3;
}

// minimal INTEGER from an n-byte big-endian unsigned value
static size_t der_int_put(uint8_t *o, const uint8_t *v, size_t n) {
    size_t i = 0;
    while (i + 1 < n && v[i] == 0) i++;
    size_t len = n - i, pad = (v[i] & 0x80) ? 1 : 0, k = 0;
    o[k++] = 0x02;
    k += der_len_put(o + k, len + pad);
    if (pad) o[k++] = 0;
    memcpy(o + k, v + i, len);
    return k + len;
}

size_t ecdsa_der_sig_encode(const uint8_t *r, const uint8_t *s, size_t n,
                            uint8_t *out) {
    uint8_t body[2 * (66 + 4)];
    if (n > 66) return 0;
    size_t b = der_int_put(body, r, n);
    b += der_int_put(body + b, s, n);
    size_t k = 0;
    out[k++] = 0x30;
    k += der_len_put(out + k, b);
    memcpy(out + k, body, b);
    return k + b;
}

static void curve_oid(ec_curve c, const uint8_t **oid, size_t *n) {
    if (c == EC_P256) { *oid = OID_P256; *n = sizeof OID_P256; }
    else if (c == EC_P384) { *oid = OID_P384; *n = sizeof OID_P384; }
    else { *oid = OID_P521; *n = sizeof OID_P521; }
}

size_t ecdsa_spki_encode(ec_curve c, const uint8_t *pub, uint8_t *out) {
    if (c < EC_P256 || c > EC_P521) return 0;
    const uint8_t *oid;
    size_t ol, fb = ec_field_bytes(c);
    curve_oid(c, &oid, &ol);
    uint8_t alg[32], lb[3];
    size_t a = 0;
    alg[a++] = 0x30;
    alg[a++] = (uint8_t)(2 + sizeof OID_EC_PUBLIC_KEY + 2 + ol);
    alg[a++] = 0x06; alg[a++] = (uint8_t)sizeof OID_EC_PUBLIC_KEY;
    memcpy(alg + a, OID_EC_PUBLIC_KEY, sizeof OID_EC_PUBLIC_KEY);
    a += sizeof OID_EC_PUBLIC_KEY;
    alg[a++] = 0x06; alg[a++] = (uint8_t)ol;
    memcpy(alg + a, oid, ol);
    a += ol;
    size_t bits = 2 + 2 * fb;   // unused-bits byte, 0x04, X, Y
    size_t body = a + 1 + der_len_put(lb, bits) + bits, k = 0;
    out[k++] = 0x30;
    k += der_len_put(out + k, body);
    memcpy(out + k, alg, a);
    k += a;
    out[k++] = 0x03;
    k += der_len_put(out + k, bits);
    out[k++] = 0x00;
    out[k++] = 0x04;
    memcpy(out + k, pub, 2 * fb);
    return k + 2 * fb;
}

// INTEGER holding one small value (a version field)
static bool der_small_int(const uint8_t *d, size_t n, size_t *off, int *v) {
    size_t len, c = der_tlv(d, n, *off, 0x02, &len);
    if (!c || len != 1 || (d[c] & 0x80)) return false;
    *v = d[c];
    *off = c + len;
    return true;
}

static bool oid_curve(const uint8_t *o, size_t n, ec_curve *c) {
    if (n == sizeof OID_P256 && !memcmp(o, OID_P256, n)) { *c = EC_P256; return true; }
    if (n == sizeof OID_P384 && !memcmp(o, OID_P384, n)) { *c = EC_P384; return true; }
    if (n == sizeof OID_P521 && !memcmp(o, OID_P521, n)) { *c = EC_P521; return true; }
    return false;
}

// SEC1 ECPrivateKey (RFC 5915) occupying d[0..n): version 1, the scalar,
// optional [0] curve, optional [1] public key. `have` says whether the
// PKCS#8 wrapper already named the curve in *c.
static bool sec1_parse(const uint8_t *d, size_t n, bool have, ec_curve *c,
                       uint8_t *priv_out) {
    size_t len, off = der_tlv(d, n, 0, 0x30, &len);
    if (!off || off + len != n) return false;
    int ver;
    if (!der_small_int(d, n, &off, &ver) || ver != 1) return false;
    size_t klen, k = der_tlv(d, n, off, 0x04, &klen);
    if (!k || klen == 0) return false;
    off = k + klen;
    const uint8_t *pubbits = NULL;
    size_t publen = 0;
    if (off < n && d[off] == 0xa0) {
        size_t plen, p = der_tlv(d, n, off, 0xa0, &plen);
        size_t olen, o = p ? der_tlv(d, n, p, 0x06, &olen) : 0;
        ec_curve named;
        if (!o || o + olen != p + plen || !oid_curve(d + o, olen, &named)) return false;
        if (have && named != *c) return false;
        *c = named;
        have = true;
        off = p + plen;
    }
    if (off < n && d[off] == 0xa1) {
        size_t plen, p = der_tlv(d, n, off, 0xa1, &plen);
        size_t blen, b = p ? der_tlv(d, n, p, 0x03, &blen) : 0;
        if (!b || b + blen != p + plen) return false;
        pubbits = d + b;
        publen = blen;
        off = p + plen;
    }
    if (off != n || !have) return false;
    size_t fb = ec_field_bytes(*c);
    if (klen > fb) return false;
    memset(priv_out, 0, fb);
    memcpy(priv_out + (fb - klen), d + k, klen);
    uint8_t pub[132];
    if (!ecdsa_public_from_private(*c, priv_out, pub)) return false;   // range
    if (pubbits && (publen != 2 + 2 * fb || pubbits[0] != 0 || pubbits[1] != 0x04 ||
                    memcmp(pubbits + 2, pub, 2 * fb) != 0))
        return false;
    return true;
}

bool ecdsa_private_key_parse(const uint8_t *der, size_t der_len,
                             ec_curve *curve_out, uint8_t *priv_out) {
    if (!der || !curve_out || !priv_out) return false;
    size_t len, off = der_tlv(der, der_len, 0, 0x30, &len);
    if (!off || off + len != der_len) return false;
    size_t after_ver = off;
    int ver;
    if (!der_small_int(der, der_len, &after_ver, &ver)) return false;
    if (after_ver < der_len && der[after_ver] == 0x04)   // SEC1
        return sec1_parse(der, der_len, false, curve_out, priv_out);
    // PKCS#8 / OneAsymmetricKey: version 0 (or 1), AlgorithmIdentifier
    // { id-ecPublicKey, namedCurve }, OCTET STRING { ECPrivateKey }, then
    // optional attributes and public key, which are not needed
    if (ver != 0 && ver != 1) return false;
    size_t alen, a = der_tlv(der, der_len, after_ver, 0x30, &alen);
    if (!a) return false;
    size_t olen, o = der_tlv(der, der_len, a, 0x06, &olen);
    if (!o || olen != sizeof OID_EC_PUBLIC_KEY ||
        memcmp(der + o, OID_EC_PUBLIC_KEY, olen) != 0)
        return false;
    size_t clen, co = der_tlv(der, der_len, o + olen, 0x06, &clen);
    ec_curve c;
    if (!co || co + clen != a + alen || !oid_curve(der + co, clen, &c)) return false;
    size_t klen, k = der_tlv(der, der_len, a + alen, 0x04, &klen);
    if (!k) return false;
    if (!sec1_parse(der + k, klen, true, &c, priv_out)) return false;
    *curve_out = c;
    return true;
}
