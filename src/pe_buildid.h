// R20: the build id a PE image carries in its CodeView debug record (ld
// --build-id writes it), read from the image as it sits in memory: a walk of
// the DOS header, the PE headers, the debug directory and the RSDS record.
// A function over a base pointer and a size, so it is testable on every
// platform (tests/test_pe_buildid.c); Windows calls it on its own loaded
// module (compat.c plat_running_image_id). Little-endian fields, read by
// memcpy. False on anything it cannot read whole.
#ifndef RUNNER_PE_BUILDID_H
#define RUNNER_PE_BUILDID_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline uint32_t pe_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static inline uint16_t pe_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

// out: "pe-buildid:" and the record's 16-byte id and 4-byte age as hex
static inline bool pe_buildid_walk(const uint8_t *img, size_t size, char *out, size_t cap) {
    (void)size;
    if (!img || cap < 60) return false;
    if (pe_u16(img) != 0x5a4d) return false;                       // "MZ"
    uint32_t nt = pe_u32(img + 0x3c);
    if (memcmp(img + nt, "PE\0\0", 4) != 0) return false;
    const uint8_t *fh = img + nt + 4;                                // the file header
    const uint8_t *opt = fh + 20;                                    // the optional header
    uint16_t magic = pe_u16(opt);
    uint32_t nrva_off = magic == 0x20b ? 108 : 92, dd_off = magic == 0x20b ? 112 : 96;
    if (pe_u32(opt + nrva_off) <= 6) return false;                   // no debug entry
    const uint8_t *dd = opt + dd_off + 6 * 8;                        // IMAGE_DIRECTORY_ENTRY_DEBUG
    uint32_t va = pe_u32(dd), dsize = pe_u32(dd + 4);
    if (!va || dsize < 28) return false;
    for (uint32_t i = 0; i < dsize / 28; i++) {
        const uint8_t *e = img + va + 28 * i;                        // an IMAGE_DEBUG_DIRECTORY
        uint32_t type = pe_u32(e + 12), sod = pe_u32(e + 16), raw = pe_u32(e + 20);
        if (type != 2 || !raw || sod < 24) continue;                 // CODEVIEW
        const uint8_t *cv = img + raw;
        if (memcmp(cv, "RSDS", 4) != 0) continue;
        int n = snprintf(out, cap, "pe-buildid:");
        for (int k = 0; k < 20; k++) n += snprintf(out + n, cap - (size_t)n, "%02x", cv[4 + k]);
        return true;
    }
    return false;
}
#endif
