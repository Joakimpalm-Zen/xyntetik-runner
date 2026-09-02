// ECDSA verification over P-256 / P-384 / P-521. See ecdsa.h for scope.
//
// Layout: big numbers are little-endian arrays of 32-bit limbs (17 limbs
// cover 521 bits); field and scalar arithmetic run in Montgomery form with
// a per-modulus context (the modulus, -p^-1 mod 2^32, R^2 mod p); points are
// Jacobian (X, Y, Z) with mixed addition against affine bases. Nothing here
// is constant-time and nothing needs to be: every input is public.
#include "ecdsa.h"
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
