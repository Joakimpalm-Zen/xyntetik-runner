// SHA-256, every backend this host has, against reference digests.
//
// envelope.c hashes with the CPU's SHA-256 instructions where the running
// CPU has them (ARMv8 SHA2, x86 SHA-NI) and with the portable FIPS 180-4
// code otherwise. A digest is defined by the standard, so each backend must
// give the reference digest on every input. The references come from
// outside this code: the FIPS 180-4 / NIST example messages (empty, "abc",
// the 448- and 896-bit messages, a million 'a') and Python's hashlib on a
// generated buffer at every length around a block's padding edges (55, 56,
// 63, 64, 119, 120, 127, 128 bytes) and past a megabyte. Each backend also
// hashes that buffer fed in uneven pieces (the streaming path that tops up
// a partial block before taking whole blocks from the input), and the two
// backends must agree on every length from 0 to 2,048. The file hash runs
// once on the default backend.
//
// envelope.c is compiled into this test to reach the backend switch.
#include <stdio.h>
#include <string.h>
#include "../src/envelope.c"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: "); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static uint8_t gen_byte(size_t i) { return (uint8_t)((i * 131 + 7) & 0xff); }

static void digest_hex(const uint8_t d[32], char hex[65]) {
    static const char hexd[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex[i * 2] = hexd[d[i] >> 4];
        hex[i * 2 + 1] = hexd[d[i] & 15];
    }
    hex[64] = 0;
}

static void hash_pieces(const uint8_t *m, size_t n, size_t piece0, char hex[65]) {
    sha256_ctx c;
    sha256_init(&c);
    size_t off = 0, piece = piece0;
    while (off < n) {
        size_t take = piece < n - off ? piece : n - off;
        sha256_update(&c, m + off, take);
        off += take;
        piece = piece * 7 % 193 + 1;   // 1 .. 193 bytes, uneven
    }
    uint8_t d[32];
    sha256_final(&c, d);
    digest_hex(d, hex);
}

static const struct { const char *msg; const char *hex; } FIPS[] = {
    { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
    { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
    { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    { "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
      "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" },
};
static const struct { size_t n; const char *hex; } GEN[] = {   // hashlib
    { 1, "ca358758f6d27e6cf45272937977a748fd88391db679ceda7dc7bf1f005ee879" },
    { 55, "16ed9c4697ca11d5f6fb25ea7900252dd4cb97215d7f6d0b2bb3e2a86ac0ec72" },
    { 56, "939ada93b2fe1e9c596d767bb408567c83e253667f0b25e5be8e16f35f2cbac9" },
    { 57, "ab84281f3b181e6cfc7cf870c7f3e7912be6fd1ac49c7dd6b21f6c25bb9a5eaa" },
    { 63, "6073f83b09ae82016cdbe24c18996c48f0eaa08ca675d0f6b90b807fc29e0149" },
    { 64, "b337ba9b0c69c391364e985fdcb23a889887e59800832c92fbfa22b8a3c40304" },
    { 65, "9d6a3fb113b586b4ab97bc11c993a27bd9b7bbcb756e0646083dc47a679600e6" },
    { 119, "9773fbac8194c3d789af101b49b6a26073076895ef6e0f658432849dd477a43f" },
    { 120, "070a538f085dd94821d4dc197c5c8b791051891d4fa2a1bf25d3c275236676f7" },
    { 127, "5072b7a9a4cda7f6d80f1eff09b8b9653201dba22319daf32c6c05339d57f483" },
    { 128, "485a94e53eba9717a5d8b7b4489cad92a752f1c5722e7dfd29dd164b7c438d11" },
    { 129, "72b63785704123441e7405a2620b859d1f107604ef5882e2674e82cc40d173a6" },
    { 1000, "533b698850849b7908b20a22658f639c0b2a476f1791f85f50188287c31a9aba" },
    { 65549, "cfa24b098ba1f4aea8bb77c01c7d310cd916f42da962adb3ab5df7788675b263" },
    { 1048583, "1953773f848663a96f77980da7153d52f3dcf27bc003ee7c749ce6e4fb487333" },
};
static const char *MILLION_A = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";

static void check_backend(int backend, const char *name, const uint8_t *buf, const uint8_t *as) {
    sha_backend = backend;
    char hex[65];
    for (size_t v = 0; v < sizeof FIPS / sizeof *FIPS; v++) {
        envelope_data_sha256(FIPS[v].msg, strlen(FIPS[v].msg), hex);
        CHECK(!strcmp(hex, FIPS[v].hex), "%s: FIPS message %zu: %s", name, v, hex);
    }
    envelope_data_sha256(as, 1000000, hex);
    CHECK(!strcmp(hex, MILLION_A), "%s: a million 'a': %s", name, hex);
    for (size_t v = 0; v < sizeof GEN / sizeof *GEN; v++) {
        envelope_data_sha256(buf, GEN[v].n, hex);
        CHECK(!strcmp(hex, GEN[v].hex), "%s: %zu bytes: %s", name, GEN[v].n, hex);
        hash_pieces(buf, GEN[v].n, 1 + v * 11, hex);
        CHECK(!strcmp(hex, GEN[v].hex), "%s: %zu bytes in pieces: %s", name, GEN[v].n, hex);
    }
}

int main(void) {
    size_t big = (size_t)1 << 21;
    uint8_t *buf = malloc(big), *as = malloc(1000000);
    if (!buf || !as) return 2;
    for (size_t i = 0; i < big; i++) buf[i] = gen_byte(i);
    memset(as, 'a', 1000000);

    check_backend(0, "portable", buf, as);
    bool hw = false;
#ifdef SHA_HW
    hw = sha256_hw_present();
    if (hw) {
        check_backend(1, "hardware", buf, as);
        // the backends agree on every length from 0 to 2048, at an odd offset
        for (size_t n = 0; n <= 2048; n++) {
            char a[65], b[65];
            sha_backend = 0; envelope_data_sha256(buf + 3, n, a);
            sha_backend = 1; envelope_data_sha256(buf + 3, n, b);
            CHECK(!strcmp(a, b), "backends differ at %zu bytes", n);
        }
    }
#endif
    // the file hash, on the backend this host uses by default
    sha_backend = -1;
    char path[] = "test-sha256.tmp";
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(buf, 1, GEN[sizeof GEN / sizeof *GEN - 1].n, f) != GEN[sizeof GEN / sizeof *GEN - 1].n) return 2;
    fclose(f);
    char hex[65];
    CHECK(envelope_file_sha256(path, hex) && !strcmp(hex, GEN[sizeof GEN / sizeof *GEN - 1].hex),
          "file hash: %s", hex);
    remove(path);
    free(buf); free(as);
    if (fails) { fprintf(stderr, "sha256: %d failures\n", fails); return 1; }
    fprintf(stderr, "sha256: reference digests on the portable code%s; default backend %s\n",
            hw ? " and the CPU's instructions (and they agree on 0..2048 bytes)" : " (no SHA instructions on this CPU)",
            sha_backend == 1 ? "hardware" : "portable");
    return 0;
}
