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

// out: "pe-buildid:" and the record's 16-byte id and 4-byte age as hex. Every
// read is bounded by size first, offsets in 64 bits so no sum can wrap:
// the DOS header, e_lfanew, the PE signature and file header, the optional
// header (its own declared size too), NumberOfRvaAndSizes before the debug
// entry is indexed, the debug directory's span, each entry, and the RSDS
// record's span, minimum length and signature.
static inline bool pe_span(uint64_t off, uint64_t len, size_t size) {
    return off <= (uint64_t)size && len <= (uint64_t)size - off;
}

static inline bool pe_buildid_walk(const uint8_t *img, size_t size, char *out, size_t cap) {
    if (!img || cap < 60 || !pe_span(0, 64, size)) return false;
    if (pe_u16(img) != 0x5a4d) return false;                       // "MZ"
    uint64_t nt = pe_u32(img + 0x3c);
    if (!pe_span(nt, 4 + 20, size) || memcmp(img + nt, "PE\0\0", 4) != 0) return false;
    const uint8_t *fh = img + nt + 4;                                // the file header
    uint64_t opt = nt + 24, opt_len = pe_u16(fh + 16);               // SizeOfOptionalHeader
    if (!pe_span(opt, opt_len, size) || opt_len < 2) return false;
    uint16_t magic = pe_u16(img + opt);
    if (magic != 0x20b && magic != 0x10b) return false;              // PE32+ or PE32
    uint64_t nrva_off = magic == 0x20b ? 108 : 92, dd_off = magic == 0x20b ? 112 : 96;
    uint64_t dbg_entry = dd_off + 6 * 8;                             // IMAGE_DIRECTORY_ENTRY_DEBUG
    if (nrva_off + 4 > opt_len || dbg_entry + 8 > opt_len) return false;
    if (pe_u32(img + opt + nrva_off) <= 6) return false;             // no debug entry
    uint64_t va = pe_u32(img + opt + dbg_entry), dsize = pe_u32(img + opt + dbg_entry + 4);
    if (!va || dsize < 28 || !pe_span(va, dsize, size)) return false;
    for (uint64_t i = 0; i < dsize / 28; i++) {
        const uint8_t *e = img + va + 28 * i;                        // an IMAGE_DEBUG_DIRECTORY
        uint32_t type = pe_u32(e + 12), sod = pe_u32(e + 16), raw = pe_u32(e + 20);
        if (type != 2) continue;                                     // CODEVIEW
        if (!raw || sod < 24 || !pe_span(raw, sod, size)) return false;
        const uint8_t *cv = img + raw;
        if (memcmp(cv, "RSDS", 4) != 0) return false;
        int n = snprintf(out, cap, "pe-buildid:");
        for (int k = 0; k < 20; k++) n += snprintf(out + n, cap - (size_t)n, "%02x", cv[4 + k]);
        return true;
    }
    return false;
}
#endif
