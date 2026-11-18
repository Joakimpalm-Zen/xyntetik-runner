// R20: pe_buildid_walk (src/pe_buildid.h) on crafted PE header buffers. Each
// buffer is a heap block of exactly the size the walker is given, so a read
// past it is AddressSanitizer's to catch (make test builds this with ASan
// where the toolchain has it). A valid image with an RSDS record is the live
// control: the walker must find its id. Every inconsistency must give false,
// never a fault: a truncated debug directory, a record or a directory past
// the image, a wrong signature, too few data directories, a PE header past
// the image, an optional header too short for the debug entry, a tiny image.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pe_buildid.h"

enum { SIZE = 1024, NT = 0x80, OPT = NT + 24, DBG = 0x200, RAW = 0x300 };

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }

static void valid(uint8_t *b) {
    memset(b, 0, SIZE);
    b[0] = 'M'; b[1] = 'Z';
    put32(b + 0x3c, NT);
    memcpy(b + NT, "PE\0\0", 4);
    put16(b + NT + 4 + 16, 240);                 // SizeOfOptionalHeader, PE32+
    put16(b + OPT, 0x20b);                       // PE32+
    put32(b + OPT + 108, 16);                    // NumberOfRvaAndSizes
    put32(b + OPT + 112 + 6 * 8, DBG);           // the debug directory's RVA
    put32(b + OPT + 112 + 6 * 8 + 4, 28);        // and size: one entry
    put32(b + DBG + 12, 2);                      // IMAGE_DEBUG_TYPE_CODEVIEW
    put32(b + DBG + 16, 25);                     // SizeOfData: RSDS, id, age, a NUL name
    put32(b + DBG + 20, RAW);                    // AddressOfRawData
    memcpy(b + RAW, "RSDS", 4);
    for (int i = 0; i < 20; i++) b[RAW + 4 + i] = (uint8_t)(0x10 + i);
}

static int fails;

// the walker on a heap copy of exactly `size` bytes of b
static bool walk(const uint8_t *b, size_t size, char *out) {
    uint8_t *h = malloc(size ? size : 1);
    memcpy(h, b, size);
    bool ok = pe_buildid_walk(h, size, out, 96);
    free(h);
    return ok;
}

static bool walk_cap(const uint8_t *b, size_t size, char *out, size_t cap) {
    uint8_t *h = malloc(size ? size : 1);
    if (size) memcpy(h, b, size);
    bool ok = pe_buildid_walk(h, size, out, cap);
    free(h);
    return ok;
}

static void expect_false(const char *what, const uint8_t *b, size_t size) {
    char out[96] = "";
    if (walk(b, size, out)) {
        printf("FAIL: %s: the walker gave %s\n", what, out);
        fails++;
    } else {
        printf("ok: %s: refused\n", what);
    }
}

int main(void) {
    uint8_t b[SIZE], c[SIZE];
    valid(b);
    char out[96] = "";
    if (!walk(b, SIZE, out) || strcmp(out, "pe-buildid:101112131415161718191a1b1c1d1e1f20212223") != 0) {
        printf("FAIL: the live control: a valid image's id was not read (%s)\n", out);
        fails++;
    } else {
        printf("ok: the live control: %s\n", out);
    }
    expect_false("a debug directory cut off by the image's end", b, DBG + 16);
    memcpy(c, b, SIZE); put32(c + DBG + 20, SIZE - 10);
    expect_false("an RSDS record past the image's end", c, SIZE);
    memcpy(c, b, SIZE); put32(c + DBG + 16, 0xfffffff0u);
    expect_false("a record size past the image's end", c, SIZE);
    memcpy(c, b, SIZE); memcpy(c + RAW, "RSDX", 4);
    expect_false("a wrong signature", c, SIZE);
    memcpy(c, b, SIZE); put32(c + OPT + 108, 6);
    expect_false("too few data directories for a debug entry", c, SIZE);
    memcpy(c, b, SIZE); put32(c + 0x3c, 0xffffff00u);
    expect_false("a PE header past the image's end", c, SIZE);
    memcpy(c, b, SIZE); put16(c + NT + 4 + 16, 100);
    expect_false("an optional header too short for the debug entry", c, SIZE);
    memcpy(c, b, SIZE); put32(c + OPT + 112 + 6 * 8 + 4, 28 * 40);
    expect_false("a debug directory past the image's end", c, SIZE);
    expect_false("an image smaller than a DOS header", b, 32);
    // from the read of c2366b8b (the antigravity pane's boundary cases)
    // 1. Zero-size RSDS entry and 23-byte entry (1 short of 24)
    memcpy(c, b, SIZE); put32(c + DBG + 16, 0);
    expect_false("zero-size RSDS entry", c, SIZE);
    memcpy(c, b, SIZE); put32(c + DBG + 16, 23);
    expect_false("23-byte RSDS entry (1 short of 24)", c, SIZE);

    // 2. Debug entry pointing at the debug directory itself
    memcpy(c, b, SIZE); put32(c + DBG + 20, DBG); put32(c + DBG + 16, 28);
    expect_false("debug entry pointing at debug directory itself", c, SIZE);

    // 3. 32-bit wrapping sums: raw+sod wrap and va+dsize wrap
    memcpy(c, b, SIZE); put32(c + DBG + 20, 0xffffff00u); put32(c + DBG + 16, 0x200u);
    expect_false("32-bit wrapping raw + sod", c, SIZE);
    memcpy(c, b, SIZE); put32(c + OPT + 112 + 6 * 8, 0xffffff00u); put32(c + OPT + 112 + 6 * 8 + 4, 0x200u);
    expect_false("32-bit wrapping va + dsize", c, SIZE);

    // 4. Exactly one byte short of every structure:
    expect_false("1 byte short of DOS header (63)", b, 63);
    expect_false("1 byte short of PE sig+file header (NT+23)", b, NT + 23);
    // Put debug dir earlier inside OPT slack so only SizeOfOptionalHeader reaches OPT+240
    memcpy(c, b, SIZE);
    expect_false("1 byte short of declared SizeOfOptionalHeader (OPT+239)", c, OPT + 239);
    expect_false("1 byte short of debug directory (DBG+27)", b, DBG + 27);
    memcpy(c, b, SIZE); put32(c + DBG + 16, 24);
    expect_false("1 byte short of 24-byte RSDS record (RAW+23)", c, RAW + 23);
    if (!walk_cap(c, RAW + 24, out, 60) || strlen(out) != 51) {
        printf("FAIL: exact 24-byte RSDS / cap=60 failed (%s)\n", out);
        fails++;
    } else {
        printf("ok: exact 24-byte RSDS with cap=60 accepted (len=%zu)\n", strlen(out));
    }

    // 5. cap < 60, zero dsize, zero raw, bad MZ, bad PE sig
    memset(out, 'X', sizeof out); out[sizeof out - 1] = 0;
    if (walk_cap(b, SIZE, out, 59)) {
        printf("FAIL: cap=59 accepted\n");
        fails++;
    } else {
        printf("ok: cap=59 refused\n");
    }
    memcpy(c, b, SIZE); put32(c + OPT + 112 + 6 * 8 + 4, 0);
    expect_false("zero debug directory size", c, SIZE);
    memcpy(c, b, SIZE); put32(c + DBG + 20, 0);
    expect_false("zero AddressOfRawData", c, SIZE);
    memcpy(c, b, SIZE); c[1] = 'N';
    expect_false("bad MZ signature", c, SIZE);
    memcpy(c, b, SIZE); c[NT + 3] = 1;
    expect_false("bad PE signature", c, SIZE);

    // no output buffer at all: refused before anything is written
    {
        uint8_t *h = malloc(SIZE);
        memcpy(h, b, SIZE);
        if (pe_buildid_walk(h, SIZE, NULL, 96)) {
            printf("FAIL: a NULL output buffer was accepted\n");
            fails++;
        } else {
            printf("ok: a NULL output buffer: refused\n");
        }
        free(h);
    }
    printf(fails ? "pe-buildid: FAILED\n" : "pe-buildid: all checks pass\n");
    return fails ? 1 : 0;
}
