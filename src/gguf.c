// GGUF v2/v3 file parser (mmap based).
#include "gguf.h"
#include "fpclass.h"
#include "quants.h"
#include "compat.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>

typedef struct {
    const uint8_t *p, *end;
    bool ok;
    uint64_t want;   // the read that ran past the end: bytes asked for at p
} cursor;

static bool need(cursor *c, size_t n) {
    if (!c->ok) return false;
    if ((size_t)(c->end - c->p) < n) { c->ok = false; c->want = n; return false; }
    return true;
}

static uint8_t  rd_u8 (cursor *c) { if (!need(c, 1)) return 0; return *c->p++; }
static uint16_t rd_u16(cursor *c) { uint16_t v = 0; if (need(c, 2)) { memcpy(&v, c->p, 2); c->p += 2; } return v; }
static uint32_t rd_u32(cursor *c) { uint32_t v = 0; if (need(c, 4)) { memcpy(&v, c->p, 4); c->p += 4; } return v; }
static uint64_t rd_u64(cursor *c) { uint64_t v = 0; if (need(c, 8)) { memcpy(&v, c->p, 8); c->p += 8; } return v; }
static float    f32_from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double   f64_from_bits(uint64_t u) { double f; memcpy(&f, &u, 8); return f; }

static bool rd_str(cursor *c, gg_str *s) {
    uint64_t n = rd_u64(c);
    if (n > SIZE_MAX - 1 || !need(c, (size_t)n)) return false;
    s->n = n;
    s->s = malloc((size_t)n + 1);
    if (!s->s) return false;
    memcpy(s->s, c->p, n);
    s->s[n] = 0;
    c->p += n;
    return true;
}

static bool str_has_nul(const gg_str *s) {
    return s->n > 0 && memchr(s->s, 0, (size_t)s->n) != NULL;
}

static const size_t gguf_scalar_size[] = {
    [GGUF_T_U8] = 1, [GGUF_T_I8] = 1, [GGUF_T_U16] = 2, [GGUF_T_I16] = 2,
    [GGUF_T_U32] = 4, [GGUF_T_I32] = 4, [GGUF_T_F32] = 4, [GGUF_T_BOOL] = 1,
    [GGUF_T_U64] = 8, [GGUF_T_I64] = 8, [GGUF_T_F64] = 8,
};

static bool rd_kv_value(cursor *c, gguf_kv *kv, uint32_t type) {
    switch (type) {
        case GGUF_T_U8:   kv->v.u64 = rd_u8(c);  break;
        case GGUF_T_I8:   kv->v.i64 = (int8_t)rd_u8(c); break;
        case GGUF_T_U16:  kv->v.u64 = rd_u16(c); break;
        case GGUF_T_I16:  kv->v.i64 = (int16_t)rd_u16(c); break;
        case GGUF_T_U32:  kv->v.u64 = rd_u32(c); break;
        case GGUF_T_I32:  kv->v.i64 = (int32_t)rd_u32(c); break;
        case GGUF_T_F32:  kv->raw = rd_u32(c); kv->v.f64 = f32_from_bits((uint32_t)kv->raw); break;
        case GGUF_T_BOOL: kv->v.b   = rd_u8(c) != 0; break;
        case GGUF_T_U64:  kv->v.u64 = rd_u64(c); break;
        case GGUF_T_I64:  kv->v.i64 = (int64_t)rd_u64(c); break;
        case GGUF_T_F64:  kv->raw = rd_u64(c); kv->v.f64 = f64_from_bits(kv->raw); break;
        case GGUF_T_STR:
            // Scalar strings are exposed through the C-string getter. Bytes
            // after an embedded NUL would be invisible to every consumer, so
            // the on-disk value and the value being validated would differ.
            return rd_str(c, &kv->str) && !str_has_nul(&kv->str);
        case GGUF_T_ARR: {
            kv->arr_type = rd_u32(c);
            kv->arr_n    = rd_u64(c);
            if (kv->arr_type == GGUF_T_STR) {
                // Every element carries at least a u64 length prefix, so a
                // count larger than an eighth of what is left cannot be
                // satisfied by this file. Without this the count is trusted
                // as far as SIZE_MAX/sizeof(gg_str) and a 49-byte file
                // declaring arr_n = 2^32 spends ~9s reserving 64 GiB before
                // being correctly rejected. The scalar branch below already
                // bounds itself this way via need().
                if (kv->arr_n > (uint64_t)(c->end - c->p) / 8) {
                    c->ok = false;   // the file ends before the array could
                    c->want = kv->arr_n > UINT64_MAX / 8 ? UINT64_MAX : kv->arr_n * 8;
                    return false;
                }
                if (kv->arr_n > SIZE_MAX / sizeof(gg_str)) return false;
                kv->arr_str = calloc((size_t)kv->arr_n, sizeof(gg_str));
                if (kv->arr_n > 0 && !kv->arr_str) return false;
                // Same rule as the scalar branch above, and for the same
                // reason: tokenizer.ggml.tokens is exposed as a C string by
                // tok_raw, and engine.c takes its strlen to feed the
                // constrained-decoding machine. An embedded NUL makes that
                // length disagree with the one the file declares, so the
                // bytes acted on are not the bytes the vocabulary holds.
                for (uint64_t i = 0; i < kv->arr_n; i++)
                    if (!rd_str(c, &kv->arr_str[i]) ||
                        str_has_nul(&kv->arr_str[i])) return false;
            } else if (kv->arr_type < GGUF_T_F64 + 1 && kv->arr_type != GGUF_T_ARR) {
                size_t es = gguf_scalar_size[kv->arr_type];
                if (es == 0 || kv->arr_n > SIZE_MAX / es ||
                    !need(c, es * (size_t)kv->arr_n)) return false;
                kv->arr_raw = c->p;
                c->p += es * (size_t)kv->arr_n;
            } else {
                return false; // nested arrays unsupported
            }
            break;
        }
        default: return false;
    }
    return c->ok;
}

static int cmp_cstr(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

// Sorts `s` in place and returns the first duplicated string, or NULL. O(n log n).
static const char *first_duplicate_inplace(const char **s, uint64_t n) {
    if (n < 2) return NULL;
    qsort(s, n, sizeof(*s), cmp_cstr);
    for (uint64_t i = 1; i < n; i++)
        if (strcmp(s[i - 1], s[i]) == 0) return s[i];
    return NULL;
}

// qsort has no context argument every libc shares, so the table being sorted
// rides in a file-static; loads serialise (swap_mu, or one CLI process), so
// nothing sorts two files at once.
static const gguf_tensor *sort_ctx_tensors;
static int cmp_tensor_offset(const void *a, const void *b) {
    const uint64_t ia = *(const uint64_t *)a, ib = *(const uint64_t *)b;
    uint64_t oa = (uint64_t)(uintptr_t)sort_ctx_tensors[ia].data;
    uint64_t ob = (uint64_t)(uintptr_t)sort_ctx_tensors[ib].data;
    if (oa != ob) return oa < ob ? -1 : 1;
    return ia < ib ? -1 : ia > ib;   // a fixed order among equal offsets
}

// A read that ran past the mapping: the file ends before its own header
// does, which is what a stopped download looks like. Say where it ends and
// how far the header reaches, rather than blaming the value being read.
static void report_short(const char *path, const gguf_file *g, const cursor *c,
                         const char *what, const char *key) {
    uint64_t at = (uint64_t)(c->p - (const uint8_t *)g->map), upto = 0;
    if (!checked_u64_add(at, c->want, &upto)) upto = UINT64_MAX;
    fprintf(stderr, "error: %s is truncated: %s%s%s needs at least %llu bytes but the "
            "file is %llu bytes (an incomplete download?)\n", path, what, key ? " " : "",
            key ? key : "", (unsigned long long)upto, (unsigned long long)g->map_size);
}

static bool gguf_open_one_x(gguf_file *g, const char *path, bool header_only) {
    memset(g, 0, sizeof(*g));
    g->header_only = header_only;
    g->map = plat_mmap_ro(path, &g->map_size);
    if (!g->map || g->map_size < 24) {
        // "cannot open X as a GGUF file" reads as "this file is corrupt", but
        // plat_mmap_ro collapses four different failures into one NULL: the
        // file is absent, it is not a regular file, it is too short to hold a
        // header, or it could not be mapped. A deleted checkpoint reported as
        // a malformed one sends the reader to the wrong place entirely, so ask
        // the filesystem which failure this was instead of blaming contents we
        // never read.
        plat_stat_t st;
        if (plat_stat(path, &st) != 0)
            fprintf(stderr, "error: cannot open %s: %s\n", path,
                    strerror(errno));
        else if (!S_ISREG(st.st_mode))
            fprintf(stderr, "error: %s is not a regular file\n", path);
        else if ((uint64_t)st.st_size < 24)
            fprintf(stderr,
                    "error: %s is too small to be a GGUF file (%llu bytes)\n",
                    path, (unsigned long long)st.st_size);
        else
            fprintf(stderr,
                    "error: cannot map %s (%llu bytes) — check permissions\n",
                    path, (unsigned long long)st.st_size);
        goto fail;
    }

    cursor c = { g->map, (const uint8_t *)g->map + g->map_size, true, 0 };
    if (rd_u32(&c) != 0x46554747) { // "GGUF"
        fprintf(stderr, "error: %s is not a GGUF file\n", path);
        goto fail;
    }
    g->version = rd_u32(&c);
    if (g->version < 2 || g->version > 3) {
        fprintf(stderr, "error: unsupported GGUF version %u\n", g->version);
        goto fail;
    }
    g->n_tensors = rd_u64(&c);
    g->n_kv      = rd_u64(&c);
    if (g->n_tensors > 100000 || g->n_kv > 100000) goto fail;

    g->kv = calloc(g->n_kv, sizeof(gguf_kv));
    if (g->n_kv > 0 && !g->kv) goto fail;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gg_str key = {0};
        if (!rd_str(&c, &key)) {
            if (!c.ok) report_short(path, g, &c, "the key of a metadata entry", NULL);
            else fprintf(stderr, "error: bad GGUF metadata\n");
            goto fail;
        }
        if (str_has_nul(&key)) {
            free(key.s);
            fprintf(stderr, "error: GGUF metadata key contains an embedded NUL\n");
            goto fail;
        }
        g->kv[i].key  = key.s;
        g->kv[i].type = rd_u32(&c);
        if (!rd_kv_value(&c, &g->kv[i], g->kv[i].type)) {
            if (!c.ok) report_short(path, g, &c, "the metadata value of", key.s);
            else fprintf(stderr, "error: bad GGUF metadata value for %s\n", key.s);
            goto fail;
        }
    }

    g->tensors = calloc(g->n_tensors, sizeof(gguf_tensor));
    if (g->n_tensors > 0 && !g->tensors) goto fail;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        gg_str name = {0};
        if (!rd_str(&c, &name)) {
            if (!c.ok) report_short(path, g, &c, "the tensor table", NULL);
            goto fail;
        }
        if (name.n >= sizeof(t->name) || str_has_nul(&name)) {
            free(name.s);
            fprintf(stderr, "error: invalid tensor metadata (name is too long "
                    "or contains an embedded NUL)\n");
            goto fail;
        }
        snprintf(t->name, sizeof(t->name), "%s", name.s);
        free(name.s);
        t->n_dims = rd_u32(&c);
        if (t->n_dims == 0 || t->n_dims > 4) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = rd_u64(&c);
        t->type = rd_u32(&c);
        t->scale = 1.0f;                  // bound after the data is mapped
        uint64_t off = rd_u64(&c);
        t->data = (void *)(uintptr_t)off; // fixed up below
    }
    if (!c.ok) { report_short(path, g, &c, "the tensor table", NULL); goto fail; }

    // Reject duplicate metadata keys and tensor names rather than silently
    // resolving to the first entry — a crafted file could otherwise shadow a
    // real geometry key or tensor with an earlier decoy (RNR-010). The scratch
    // array is part of the check, so failing to allocate it fails the open:
    // skipping the scan would admit exactly the file the check exists to
    // refuse, and a defense that switches itself off under memory pressure is
    // not one.
    if (g->n_kv > 1) {
        const char **keys = malloc((size_t)g->n_kv * sizeof(*keys));
        if (!keys) goto fail;
        for (uint64_t i = 0; i < g->n_kv; i++) keys[i] = g->kv[i].key;
        const char *d = first_duplicate_inplace(keys, g->n_kv);
        if (d) fprintf(stderr, "error: duplicate GGUF metadata key '%s'\n", d);
        free(keys);
        if (d) goto fail;
    }
    if (g->n_tensors > 1) {
        const char **names = malloc((size_t)g->n_tensors * sizeof(*names));
        if (!names) goto fail;
        for (uint64_t i = 0; i < g->n_tensors; i++) names[i] = g->tensors[i].name;
        const char *d = first_duplicate_inplace(names, g->n_tensors);
        if (d) fprintf(stderr, "error: duplicate GGUF tensor name '%s'\n", d);
        free(names);
        if (d) goto fail;
    }

    uint64_t align = gguf_get_u32(g, "general.alignment", 32);
    if (align == 0 || (align & (align - 1))) align = 32;
    uint64_t header_end = (uint64_t)(c.p - (const uint8_t *)g->map);
    uint64_t aligned_end;
    if (!checked_u64_add(header_end, align - 1, &aligned_end)) goto invalid;
    uint64_t data_start = aligned_end & ~(align - 1);
    // The header itself must still be entirely present, in BOTH modes: a
    // header-only parse that read past its own mapping would be reading
    // garbage, not being tolerant. (What the two modes disagree about is the
    // DATA section, below.)
    if (data_start > g->map_size) goto invalid;
    g->data_offset = (size_t)data_start;
    g->mincore_safe = plat_file_mincore_safe(path);

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (!ggml_type_supported(t->type)) {
            // The descriptor stays (the loader reports an unusable type by
            // name), but the stored offset cannot be validated without a block
            // size this build does not have — so it must not survive as a
            // pointer. It is an arbitrary 64-bit value out of an untrusted
            // file; leaving it in place parks an unbounded address in a field
            // whose every consumer treats NULL as "nothing to read here".
            t->data = NULL;
            t->nbytes = 0;
            continue;   // the type itself is checked at use time
        }
        int bs = ggml_block_size(t->type);
        uint64_t rows, row_blocks, row_bytes;
        if (t->ne[0] == 0 || t->ne[0] > INT_MAX || t->ne[0] % (uint64_t)bs != 0 ||
            t->ne[1] == 0 || t->ne[1] > INT_MAX ||
            t->ne[2] == 0 || t->ne[2] > INT_MAX ||
            t->ne[3] == 0 || t->ne[3] > INT_MAX ||
            !checked_u64_mul(t->ne[1], t->ne[2], &rows) ||
            !checked_u64_mul(rows, t->ne[3], &rows)) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        row_blocks = t->ne[0] / (uint64_t)bs;
        if (!checked_u64_mul(row_blocks, ggml_type_size(t->type), &row_bytes) ||
            !checked_u64_mul(row_bytes, rows, &t->nbytes)) {
            fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        uint64_t off = (uint64_t)(uintptr_t)t->data;
        // The GGUF spec puts every tensor's data at a multiple of
        // general.alignment (32 without the key) from the data section's
        // start, and a block's fields are read in place: an f16 scale needs 2
        // bytes of alignment, an F32 value 4. An offset that breaks either
        // made every decode of the tensor read misaligned, undefined
        // behaviour: the fuzzer found a Q4_0 norm at offset 131,071 that the
        // loader took and dq_q4_0 then read at an odd address (2026-11-16).
        // Such a file is malformed; refused by name, in both modes. The
        // mapping starts on a page, so data_start + off is the address's
        // alignment.
        if (off % align != 0) {
            fprintf(stderr, "error: invalid tensor metadata for %s: its data offset %llu is "
                    "not a multiple of the file's alignment, %llu (general.alignment, 32 "
                    "without it), so its blocks would be read misaligned\n", t->name,
                    (unsigned long long)off, (unsigned long long)align);
            goto fail;
        }
        if ((data_start + off) % 4 != 0) {
            fprintf(stderr, "error: invalid tensor metadata for %s: its data starts at byte "
                    "%llu of the file, which leaves its blocks misaligned (general.alignment "
                    "%llu is under the 4 bytes a block's fields need)\n", t->name,
                    (unsigned long long)(data_start + off), (unsigned long long)align);
            goto fail;
        }
        if (header_only) {
            // The point of this mode: the bytes need not be here. Record what
            // the header claims the data section spans, and leave `data` NULL
            // so nothing can read through it by accident. Still reject
            // arithmetic that cannot describe a real file.
            uint64_t end;
            if (!checked_u64_add(off, t->nbytes, &end)) {
                fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
                goto fail;
            }
            if (end > g->data_bytes) g->data_bytes = end;
            t->data = NULL;
            continue;
        }
        if (off > g->map_size - data_start ||
            t->nbytes > g->map_size - data_start - off) {
            // The usual cause is not a hostile header but a download that
            // stopped: say what is missing, so the user fetches the file
            // again instead of suspecting the model.
            uint64_t need = 0;
            bool fits = checked_u64_add(off, t->nbytes, &need) &&
                        checked_u64_add(need, data_start, &need);
            if (fits)
                fprintf(stderr, "error: invalid tensor metadata for %s: its data "
                        "ends at byte %llu but the file is %llu bytes (a "
                        "truncated or incomplete download?)\n", t->name,
                        (unsigned long long)need, (unsigned long long)g->map_size);
            else
                fprintf(stderr, "error: invalid tensor metadata for %s\n", t->name);
            goto fail;
        }
        t->data = (uint8_t *)g->map + (size_t)data_start + (size_t)off;
    }

    // NVFP4 layout check. ggml's block_nvfp4 (type 40) is 36 bytes per 64
    // elements: four UE4M3 sub-block scales and 32 packed nibbles. At least
    // one third-party quantizer writes type 40 with fp16 sub-block scales, 40
    // bytes per 64, under the same type id; decoded as ggml's layout every
    // row is read misaligned and the model produces NaN with no error. The
    // header cannot say which it is, but the byte span between a tensor's
    // offset and the next tensor's can: refuse the variant by name rather
    // than serve garbage from it.
    if (!header_only && g->n_tensors > 0) {
        uint64_t *order = malloc(sizeof(uint64_t) * g->n_tensors);
        if (order) {
            for (uint64_t i = 0; i < g->n_tensors; i++) order[i] = i;
            // qsort, not insertion: the header admits 100,000 tensors, and a
            // file listing them in reverse offset order made this quadratic
            // (about 5e9 steps, minutes inside a swap) for a companion lookup
            // only NVFP4 tensors need (sweep, 2026-09-27; the same class as
            // the tokenizer's special-token sort).
            sort_ctx_tensors = g->tensors;
            qsort(order, (size_t)g->n_tensors, sizeof(*order), cmp_tensor_offset);
            for (uint64_t i = 0; i < g->n_tensors; i++) {
                gguf_tensor *t = &g->tensors[order[i]];
                if (t->type != T_NVFP4 || t->nbytes == 0) continue;
                uint64_t off = (uint64_t)(uintptr_t)t->data;
                // the next tensor's data, or for the last one the file's end,
                // as an address like `off`: it was the data section's SIZE,
                // which is far below any address, so the span came out 0 and
                // the last tensor in a file was never measured (R6.68.48)
                uint64_t next = i + 1 < g->n_tensors
                              ? (uint64_t)(uintptr_t)g->tensors[order[i + 1]].data
                              : (uint64_t)(uintptr_t)g->map + g->map_size;
                uint64_t span = next > off ? next - off : 0;
                // 40 bytes per 64 elements, less the alignment slack a
                // 36-byte layout could legitimately carry
                uint64_t variant = t->nbytes / 36 * 40;
                if (span >= variant && variant > t->nbytes + align) {
                    fprintf(stderr, "error: tensor %s is NVFP4 with 40 bytes per 64 "
                            "elements (fp16 sub-block scales, a third-party "
                            "quantizer's layout); this runner decodes ggml's "
                            "block_nvfp4 (36 bytes: UE4M3 scales) and would "
                            "read this file as garbage\n", t->name);
                    free(order);
                    goto fail;
                }
            }
            free(order);
        }
    }
    if (header_only) {
        uint64_t total;
        if (!checked_u64_add(data_start, g->data_bytes, &total)) goto invalid;
        g->mapped_size = total;      // what the WHOLE file would be
        return true;
    }
    g->mapped_size = g->map_size;
    return true;

invalid:
    fprintf(stderr, "error: invalid tensor metadata\n");
fail:
    gguf_close(g);
    return false;
}


// Per-tensor scale companions. NVIDIA's ModelOpt NVFP4 export is TWO-LEVEL:
// a UE4M3 scale per 16 elements inside each block AND one F32 `<base>.scale`
// tensor beside `<base>.weight`, applied in the compute graph rather than by
// the block decode. The companion survives requantization (llama.cpp's F16
// re-export of such a file still carries it), so it is bound by name and
// shape, not by weight type: any `X.weight` whose sibling `X.scale` holds
// exactly one F32 value is stored * that value. `X.input_scale` is the
// activation-side scale of a quantized-activation kernel and is deliberately
// NOT applied: a float inference path never quantizes activations, and
// applying it puts the weights two orders of magnitude too small (measured
// 2026-08-30). A file with no companions keeps every scale at 1.0, which is
// exactly right for single-level NVFP4 written by llama.cpp's own quantizer.
static void gguf_bind_scale_companions(gguf_file *g) {
    static const char SUF[] = ".scale";
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *c = &g->tensors[i];
        size_t n = strlen(c->name);
        if (n <= sizeof(SUF) - 1 || strcmp(c->name + n - (sizeof(SUF) - 1), SUF) != 0)
            continue;
        if (c->type != T_F32 || !c->data ||
            c->ne[0] * c->ne[1] * c->ne[2] * c->ne[3] != 1)
            continue;
        char wname[128];
        int base = (int)(n - (sizeof(SUF) - 1));
        if (base + 7 >= (int)sizeof(wname)) continue;
        snprintf(wname, sizeof(wname), "%.*s.weight", base, c->name);
        gguf_tensor *w = gguf_find_tensor(g, wname);
        if (!w) continue;
        float v = *(const float *)c->data;
        if (!(v > 0.0f) || !fp_isfinite(v)) {
            fprintf(stderr, "warning: %s holds %g, which is not a usable "
                            "per-tensor scale; %s left unscaled\n",
                    c->name, v, wname);
            continue;
        }
        w->scale = v;
    }
}

static bool gguf_open_set(gguf_file *g, const char *path, bool header_only);

// The header of the whole model: a file in several parts
// (<prefix>-00001-of-0000N.gguf) keeps each part's tensors in that part's own
// table, so a reader of one part sees a model with tensors missing. --info fit
// answered "WILL NOT LOAD ... missing tensor" from the first part of a set
// the loader takes (2026-11-15), and derive's planner had made the same
// mistake the day before. Every part's header is read, the data of none.
bool gguf_open_header(gguf_file *g, const char *path) {
    return gguf_open_set(g, path, true);
}

static bool split_part_path(char *out, size_t cap, const char *path,
                            uint32_t current, uint32_t count, uint32_t wanted) {
    char suffix[64];
    snprintf(suffix, sizeof suffix, "-%05u-of-%05u.gguf", current + 1, count);
    size_t n = strlen(path), sn = strlen(suffix);
    if (n < sn || strcmp(path + n - sn, suffix) != 0) return false;
    int wrote = snprintf(out, cap, "%.*s-%05u-of-%05u.gguf",
                         (int)(n - sn), path, wanted + 1, count);
    return wrote > 0 && (size_t)wrote < cap;
}

// a scalar array element's stored size by GGUF type (0: not a scalar)
static const size_t split_esz[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };

static size_t split_elem_size(uint32_t t) { return t < sizeof split_esz / sizeof *split_esz ? split_esz[t] : 0; }

// R19.17.1: one key's value in two parts of a set, compared by its stored bits (a float by its bits, a string and
// an array element by element)
static bool split_kv_same(const gguf_kv *a, const gguf_kv *b) {
    if (a->type != b->type) return false;
    switch (a->type) {
    case GGUF_T_STR:
        return a->str.n == b->str.n && (a->str.n == 0 || !memcmp(a->str.s, b->str.s, (size_t)a->str.n));
    case GGUF_T_F32: case GGUF_T_F64:
        return a->raw == b->raw;
    case GGUF_T_BOOL:
        return a->v.b == b->v.b;
    case GGUF_T_ARR: {
        if (a->arr_type != b->arr_type || a->arr_n != b->arr_n) return false;
        if (a->arr_type == GGUF_T_STR) {
            for (uint64_t k = 0; k < a->arr_n; k++)
                if (a->arr_str[k].n != b->arr_str[k].n ||
                    (a->arr_str[k].n && memcmp(a->arr_str[k].s, b->arr_str[k].s, (size_t)a->arr_str[k].n)))
                    return false;
            return true;
        }
        const size_t e = split_elem_size(a->arr_type);
        if (!e || !a->arr_raw || !b->arr_raw) return a->arr_raw == b->arr_raw;
        return !memcmp(a->arr_raw, b->arr_raw, (size_t)a->arr_n * e);
    }
    default:
        return a->v.u64 == b->v.u64;   // the integers, widened into the same 64 bits by type
    }
}

// the first element two arrays of one type and length differ at, for a refusal whose two values read alike (0
// when their element types differ)
static uint64_t split_arr_first_diff(const gguf_kv *a, const gguf_kv *b) {
    if (a->arr_type != b->arr_type || a->arr_n != b->arr_n) return 0;
    if (a->arr_type == GGUF_T_STR) {
        for (uint64_t k = 0; k < a->arr_n; k++)
            if (a->arr_str[k].n != b->arr_str[k].n ||
                (a->arr_str[k].n && memcmp(a->arr_str[k].s, b->arr_str[k].s, (size_t)a->arr_str[k].n)))
                return k;
        return 0;
    }
    const size_t e = split_elem_size(a->arr_type);
    if (!e || !a->arr_raw || !b->arr_raw) return 0;
    const unsigned char *pa = a->arr_raw, *pb = b->arr_raw;
    for (uint64_t k = 0; k < a->arr_n; k++)
        if (memcmp(pa + k * e, pb + k * e, e)) return k;
    return 0;
}

// a key's value for the refusal's sentence
static void split_kv_text(const gguf_kv *kv, char *out, size_t cap) {
    switch (kv->type) {
    case GGUF_T_U8: case GGUF_T_U16: case GGUF_T_U32: case GGUF_T_U64:
        snprintf(out, cap, "%llu", (unsigned long long)kv->v.u64); break;
    case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64:
        snprintf(out, cap, "%lld", (long long)kv->v.i64); break;
    case GGUF_T_F32: case GGUF_T_F64:
        snprintf(out, cap, "%.9g", kv->v.f64); break;
    case GGUF_T_BOOL:
        snprintf(out, cap, "%s", kv->v.b ? "true" : "false"); break;
    case GGUF_T_STR:
        snprintf(out, cap, "\"%.60s%s\"", kv->str.s ? kv->str.s : "", kv->str.n > 60 ? "..." : ""); break;
    case GGUF_T_ARR:
        snprintf(out, cap, "an array of %llu", (unsigned long long)kv->arr_n); break;
    default:
        snprintf(out, cap, "(type %u)", kv->type); break;
    }
}

// R19.17.1: the first key part i STATES that part 0 states too and CONTRADICTS (general.file_type first, the
// quant a set is), or NULL. The split.* keys are each part's own. A sparse part (llama.cpp's standard layout)
// states only those, so nothing in it is compared: the format carries no set identity to check.
static const gguf_kv *split_kv_contradiction(gguf_file *p0, gguf_file *pi, const gguf_kv **in0) {
    const gguf_kv *ft = gguf_get(pi, "general.file_type"), *ft0 = gguf_get(p0, "general.file_type");
    if (ft && ft0 && !split_kv_same(ft0, ft)) { *in0 = ft0; return ft; }
    for (uint64_t k = 0; k < pi->n_kv; k++) {
        const gguf_kv *kv = &pi->kv[k];
        if (!kv->key || !strncmp(kv->key, "split.", 6)) continue;
        const gguf_kv *kv0 = gguf_get(p0, kv->key);
        if (kv0 && !split_kv_same(kv0, kv)) { *in0 = kv0; return kv; }
    }
    return NULL;
}

// R19.17.1: the name a split set goes by, whichever part opens it: the first part (<prefix>-00001-of-N.gguf, the
// name the set's other parts are derived from and the one the set is listed by), with the whole set's bytes and its
// part count; a single file is itself (1 part, its own size). false: the path cannot be read.
// Which part of its set this file is, from its own header (split.no of split.count; a single file is part 0 of 1).
// The one header read a caller of the set's paths needs: each part's path is then the name's arithmetic.
static bool split_probe(const char *path, uint32_t *current, uint32_t *count) {
    gguf_file probe;
    if (!gguf_open_one_x(&probe, path, true)) return false;
    uint32_t n = gguf_get_u32(&probe, "split.count", 0);
    *current = gguf_get_u32(&probe, "split.no", UINT32_MAX);
    gguf_close(&probe);
    *count = n <= 1 ? 1 : n;
    return true;
}

// part i's path, for a file that is part `current` of `count` (split_probe's answer)
static bool split_sibling(const char *path, uint32_t current, uint32_t count, uint32_t i, char *out, size_t cap) {
    if (count <= 1) {
        if (i != 0) return false;
        int wrote = snprintf(out, cap, "%s", path);
        return wrote > 0 && (size_t)wrote < cap;
    }
    if (current >= count || i >= count) return false;
    return split_part_path(out, cap, path, current, count, i);
}

bool gguf_set_name(const char *path, char *first, size_t cap, uint32_t *n_parts, long long *bytes) {
    uint32_t cur = 0, n = 1;
    if (!split_probe(path, &cur, &n) || !split_sibling(path, cur, n, 0, first, cap)) return false;
    if (bytes) {   // each part's size from the file system: no part's header is read again
        long long total = 0;
        for (uint32_t i = 0; i < n; i++) {
            char part[4096];
            plat_stat_t st;
            if (!split_sibling(path, cur, n, i, part, sizeof part) || plat_stat(part, &st) != 0) { total = -1; break; }
            total += (long long)st.st_size;
        }
        *bytes = total;
    }
    if (n_parts) *n_parts = n;
    return true;
}

bool gguf_split_part_path(const char *path, uint32_t i, char *out, size_t cap, uint32_t *count) {
    uint32_t cur = 0, n = 1;
    // this part's own header: the question is which part this file is
    if (!split_probe(path, &cur, &n)) return false;
    if (count) *count = n;
    return split_sibling(path, cur, n, i, out, cap);
}

bool gguf_open(gguf_file *g, const char *path) {
    return gguf_open_set(g, path, false);
}

// One file, or every part of a split set merged into one tensor table.
// header_only reads no tensor data (gguf_open_one_x): the parts may be
// partial downloads, each as long as its own header. The scale companions
// are bound from tensor data, so only a full open binds them.
static bool gguf_open_set(gguf_file *g, const char *path, bool header_only) {
    memset(g, 0, sizeof(*g));
    gguf_file probe;
    if (!gguf_open_one_x(&probe, path, header_only)) return false;
    uint32_t count = gguf_get_u32(&probe, "split.count", 0);
    if (count <= 1) {
        *g = probe;
        if (!header_only) gguf_bind_scale_companions(g);
        return true;
    }
    uint32_t current = gguf_get_u32(&probe, "split.no", UINT32_MAX);
    uint32_t expected = gguf_get_u32(&probe, "split.tensors.count", UINT32_MAX);
    if (current >= count || expected == UINT32_MAX || expected > 100000 ||
        count > expected || count > 10000) {
        fprintf(stderr, "error: invalid split GGUF metadata in %s\n", path);
        gguf_close(&probe);
        return false;
    }

    char first[4096];
    if (!split_part_path(first, sizeof first, path, current, count, 0)) {
        fprintf(stderr, "error: split GGUF filename does not follow "
                "<prefix>-00001-of-%05u.gguf: %s\n", count, path);
        gguf_close(&probe);
        return false;
    }
    gguf_close(&probe);

    gguf_file *parts = calloc(count, sizeof(*parts));
    if (!parts) return false;
    uint64_t total = 0, mapped = 0;
    bool ok = true;
    for (uint32_t i = 0; i < count; i++) {
        char part_path[4096] = {0};
        if (!split_part_path(part_path, sizeof part_path, first, 0, count, i) ||
            !gguf_open_one_x(&parts[i], part_path, header_only)) {
            fprintf(stderr, "error: missing or unreadable split GGUF part %u/%u: %s\n",
                    i + 1, count, part_path);
            ok = false; break;
        }
        uint32_t no = gguf_get_u32(&parts[i], "split.no", UINT32_MAX);
        uint32_t nc = gguf_get_u32(&parts[i], "split.count", 0);
        uint32_t nt = gguf_get_u32(&parts[i], "split.tensors.count", UINT32_MAX);
        const char *arch0 = gguf_get_str(&parts[0], "general.architecture", NULL);
        const char *archi = gguf_get_str(&parts[i], "general.architecture", NULL);
        uint32_t align0 = gguf_get_u32(&parts[0], "general.alignment", 32);
        // Standard split GGUFs may put the model metadata only in part zero;
        // later parts then carry just split.no/count/tensors.count. Treat an
        // omitted value as inherited, while still rejecting an explicit
        // contradiction. The tensor table in each part remains independently
        // bounds-checked by gguf_open_one above.
        uint32_t aligni = gguf_get_u32(&parts[i], "general.alignment", align0);
        if (parts[i].version != parts[0].version ||
            !arch0 || (archi && strcmp(arch0, archi) != 0) || align0 != aligni ||
            no != i || nc != count || nt != expected ||
            parts[i].n_tensors > expected ||
            total > (uint64_t)expected - parts[i].n_tensors ||
            mapped > UINT64_MAX - parts[i].mapped_size) {
            fprintf(stderr, "error: inconsistent split GGUF metadata in %s\n",
                    part_path);
            ok = false; break;
        }
        // R19.17.1: a part from another set of the same model (another quant split the same way) agrees on every
        // split key; where it states a key the first part states and contradicts it, it is refused by name
        const gguf_kv *kv0 = NULL, *kvi = i ? split_kv_contradiction(&parts[0], &parts[i], &kv0) : NULL;
        if (kvi) {
            char v0[96], vi[96];
            split_kv_text(kv0, v0, sizeof v0);
            split_kv_text(kvi, vi, sizeof vi);
            if (kvi->type == GGUF_T_ARR && kv0->type == GGUF_T_ARR && !strcmp(v0, vi))
                // two arrays of one length read alike: the sentence names where they part
                fprintf(stderr, "error: split GGUF part %u/%u is from another set: %s, %s, differs from part 1's "
                        "at element %llu: %s\n", i + 1, count, kvi->key, vi,
                        (unsigned long long)split_arr_first_diff(kv0, kvi), part_path);
            else
                fprintf(stderr, "error: split GGUF part %u/%u is from another set: %s %s against %s in part 1: %s\n",
                        i + 1, count, kvi->key, vi, v0, part_path);
            ok = false; break;
        }
        total += parts[i].n_tensors;
        // what the whole part is: its mapping, or under header_only the size
        // its header declares (the same number for a complete file)
        mapped += parts[i].mapped_size;
    }
    if (ok && total != expected) {
        fprintf(stderr, "error: split GGUF declares %u tensors but parts contain %llu\n",
                expected, (unsigned long long)total);
        ok = false;
    }
    if (!ok) {
        for (uint32_t i = 0; i < count; i++) gguf_close(&parts[i]);
        free(parts);
        return false;
    }

    gguf_tensor *merged = calloc((size_t)total, sizeof(*merged));
    void **maps = calloc(count, sizeof(*maps));
    size_t *sizes = calloc(count, sizeof(*sizes));
    size_t *doffs = calloc(count, sizeof(*doffs));
    if ((total && !merged) || !maps || !sizes || !doffs) {
        free(merged); free(maps); free(sizes); free(doffs);
        for (uint32_t i = 0; i < count; i++) gguf_close(&parts[i]);
        free(parts);
        return false;
    }
    uint64_t at = 0;
    bool all_mincore_safe = true;
    for (uint32_t i = 0; i < count; i++) {
        memcpy(merged + at, parts[i].tensors,
               (size_t)parts[i].n_tensors * sizeof(*merged));
        at += parts[i].n_tensors;
        maps[i] = parts[i].map;
        sizes[i] = parts[i].map_size;
        doffs[i] = parts[i].data_offset;
        all_mincore_safe = all_mincore_safe && parts[i].mincore_safe;
        parts[i].map = NULL;
        parts[i].map_size = 0;
    }

    *g = parts[0];
    free(g->tensors);
    g->tensors = merged;
    g->n_tensors = total;
    g->maps = maps;
    g->map_sizes = sizes;
    g->map_data_offsets = doffs;
    g->n_maps = count;
    g->mapped_size = mapped;
    g->map = maps[0];
    g->map_size = sizes[0];
    g->data_offset = doffs[0];
    g->mincore_safe = all_mincore_safe;
    for (uint32_t i = 1; i < count; i++) gguf_close(&parts[i]);
    free(parts);

    if (g->n_tensors > 1) {
        const char **names = malloc((size_t)g->n_tensors * sizeof(*names));
        if (!names) { gguf_close(g); return false; }
        for (uint64_t i = 0; i < g->n_tensors; i++) names[i] = g->tensors[i].name;
        const char *d = first_duplicate_inplace(names, g->n_tensors);
        if (d) fprintf(stderr, "error: duplicate tensor name across split GGUF parts: '%s'\n", d);
        free(names);
        if (d) { gguf_close(g); return false; }
    }
    if (!header_only) gguf_bind_scale_companions(g);
    return true;
}

bool gguf_attach(gguf_file *g, gguf_file *extra) {
    if (!g || !extra || g->header_only || extra->header_only) return false;
    uint64_t keep = 0;
    for (uint64_t i = 0; i < extra->n_tensors; i++)
        if (!gguf_find_tensor(g, extra->tensors[i].name)) keep++;
    const uint32_t n_old = gguf_map_count(g), n_new = gguf_map_count(extra);
    gguf_tensor *merged = realloc(g->tensors,
                                  (size_t)(g->n_tensors + keep) * sizeof(*merged));
    if (!merged) return false;
    g->tensors = merged;
    void  **maps  = calloc((size_t)n_old + n_new, sizeof(*maps));
    size_t *sizes = calloc((size_t)n_old + n_new, sizeof(*sizes));
    size_t *doffs = calloc((size_t)n_old + n_new, sizeof(*doffs));
    if (!maps || !sizes || !doffs) { free(maps); free(sizes); free(doffs); return false; }
    for (uint32_t i = 0; i < n_old; i++) {
        maps[i] = gguf_map_part(g, i, &sizes[i]);
        doffs[i] = gguf_map_data_offset(g, i);
    }
    for (uint32_t i = 0; i < n_new; i++) {
        maps[n_old + i] = gguf_map_part(extra, i, &sizes[n_old + i]);
        doffs[n_old + i] = gguf_map_data_offset(extra, i);
    }
    for (uint64_t i = 0; i < extra->n_tensors; i++)
        if (!gguf_find_tensor(g, extra->tensors[i].name))
            g->tensors[g->n_tensors++] = extra->tensors[i];
    free(g->maps); free(g->map_sizes); free(g->map_data_offsets);
    g->maps = maps; g->map_sizes = sizes; g->map_data_offsets = doffs; g->n_maps = n_old + n_new;
    g->map = maps[0]; g->map_size = sizes[0]; g->data_offset = doffs[0];
    g->mapped_size += extra->mapped_size;
    g->mincore_safe = g->mincore_safe && extra->mincore_safe;
    // the mappings now belong to g; drop extra's bookkeeping without unmapping
    free(extra->maps); free(extra->map_sizes); free(extra->map_data_offsets);
    extra->maps = NULL; extra->map_sizes = NULL; extra->map_data_offsets = NULL; extra->n_maps = 0;
    extra->map = NULL; extra->map_size = 0; extra->data_offset = 0;
    gguf_close(extra);
    return true;
}

void gguf_close(gguf_file *g) {
    for (uint64_t i = 0; g->kv && i < g->n_kv; i++) {
        gguf_kv *kv = &g->kv[i];
        free(kv->key);
        free(kv->str.s);
        if (kv->arr_str) {
            for (uint64_t j = 0; j < kv->arr_n; j++) free(kv->arr_str[j].s);
            free(kv->arr_str);
        }
    }
    free(g->kv);
    free(g->tensors);
    if (g->maps) {
        for (uint32_t i = 0; i < g->n_maps; i++)
            plat_munmap(g->maps[i], g->map_sizes[i]);
        free(g->maps);
        free(g->map_sizes);
        free(g->map_data_offsets);
    } else {
        plat_munmap(g->map, g->map_size);
    }
    memset(g, 0, sizeof(*g));
}

uint64_t gguf_mapped_size(const gguf_file *g) {
    return g ? g->mapped_size : 0;
}

uint32_t gguf_map_count(const gguf_file *g) {
    if (!g) return 0;
    return g->n_maps ? g->n_maps : 1;   // a single file is one unnamed part
}

void *gguf_map_part(const gguf_file *g, uint32_t i, size_t *size) {
    if (size) *size = 0;
    if (!g || i >= gguf_map_count(g)) return NULL;
    if (size) *size = g->n_maps ? g->map_sizes[i] : g->map_size;
    return g->n_maps ? g->maps[i] : g->map;
}

size_t gguf_map_data_offset(const gguf_file *g, uint32_t i) {
    if (!g || i >= gguf_map_count(g)) return 0;
    if (g->n_maps) return g->map_data_offsets ? g->map_data_offsets[i] : 0;
    return g->data_offset;
}

gguf_kv *gguf_get(gguf_file *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    return NULL;
}

static bool kv_is_unsigned(uint32_t t) {
    return t == GGUF_T_U8 || t == GGUF_T_U16 || t == GGUF_T_U32 || t == GGUF_T_U64;
}
static bool kv_is_signed(uint32_t t) {
    return t == GGUF_T_I8 || t == GGUF_T_I16 || t == GGUF_T_I32 || t == GGUF_T_I64;
}
// Bit-pattern finiteness/sign checks: the engine builds with -ffast-math, which
// lets the compiler assume no NaN/inf and fold float comparisons like `d>=0` or
// isfinite() to constants — so those cannot be trusted for validating untrusted
// metadata. Inspecting the IEEE bits directly is immune to that.
static bool raw_f32_finite(uint32_t u) { return (u & 0x7f800000u) != 0x7f800000u; }
static bool raw_f32_negative(uint32_t u) { return (u >> 31) != 0; }
static bool raw_f64_finite(uint64_t u) { return (u & 0x7ff0000000000000ull) != 0x7ff0000000000000ull; }
static bool raw_f64_negative(uint64_t u) { return (u >> 63) != 0; }

// Typed getters validate the source type, sign, range, and finiteness rather
// than reinterpreting whatever the union happens to hold. A negative,
// out-of-range, NaN, or type-confused metadata value from an untrusted GGUF
// becomes the caller's default (i.e. "not usable"), never a huge or
// implementation-defined geometry value that slips past a later sanity check
// (RNR-010).
uint32_t gguf_get_u32(gguf_file *g, const char *key, uint32_t dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv_is_unsigned(kv->type))
        return kv->v.u64 <= UINT32_MAX ? (uint32_t)kv->v.u64 : dflt;
    if (kv_is_signed(kv->type))
        return (kv->v.i64 >= 0 && kv->v.i64 <= (int64_t)UINT32_MAX)
                 ? (uint32_t)kv->v.i64 : dflt;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) {
        bool finite = kv->type == GGUF_T_F32
                    ? raw_f32_finite((uint32_t)kv->raw)
                    : raw_f64_finite(kv->raw);
        bool negative = kv->type == GGUF_T_F32
                      ? raw_f32_negative((uint32_t)kv->raw)
                      : raw_f64_negative(kv->raw);
        double d = kv->v.f64;   // finite, non-negative, in-range integer
        if (!finite || negative || d > (double)UINT32_MAX ||
            d != (double)(uint32_t)d)
            return dflt;
        return (uint32_t)d;
    }
    return dflt;   // ARR, STR, BOOL are not a u32
}

// Per-index variant for metadata some exports publish per layer as an
// ARRAY-typed value where others write one scalar (gemma-4 E2B's
// feed_forward_length is the motivating case: real 6144/12288 per-layer
// variation). A scalar answers every index; an integer-typed array answers
// in-range indexes with the same sign/range validation as the scalar getter;
// float/bool/string arrays, out-of-range indexes and negatives are the
// caller's default — never a read past the array or garbage geometry.
uint32_t gguf_get_u32_idx(gguf_file *g, const char *key, uint64_t idx,
                          uint32_t dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv->type != GGUF_T_ARR) return gguf_get_u32(g, key, dflt);
    if (idx >= kv->arr_n || !kv->arr_raw) return dflt;
    const unsigned char *p = kv->arr_raw;
    switch (kv->arr_type) {
    case GGUF_T_U8:  return p[idx];
    case GGUF_T_I8:  { int8_t v = (int8_t)p[idx];
                       return v >= 0 ? (uint32_t)v : dflt; }
    case GGUF_T_U16: { uint16_t v; memcpy(&v, p + idx * 2, 2); return v; }
    case GGUF_T_I16: { int16_t v; memcpy(&v, p + idx * 2, 2);
                       return v >= 0 ? (uint32_t)v : dflt; }
    case GGUF_T_U32: { uint32_t v; memcpy(&v, p + idx * 4, 4); return v; }
    case GGUF_T_I32: { int32_t v; memcpy(&v, p + idx * 4, 4);
                       return v >= 0 ? (uint32_t)v : dflt; }
    case GGUF_T_U64: { uint64_t v; memcpy(&v, p + idx * 8, 8);
                       return v <= UINT32_MAX ? (uint32_t)v : dflt; }
    case GGUF_T_I64: { int64_t v; memcpy(&v, p + idx * 8, 8);
                       return (v >= 0 && v <= (int64_t)UINT32_MAX)
                                ? (uint32_t)v : dflt; }
    default: return dflt;   // F32/F64/BOOL/STR arrays are not integer geometry
    }
}

uint64_t gguf_get_u64_idx(gguf_file *g, const char *key, uint64_t idx,
                          uint64_t dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv || kv->type != GGUF_T_ARR || idx >= kv->arr_n || !kv->arr_raw)
        return dflt;
    const unsigned char *p = kv->arr_raw;
    switch (kv->arr_type) {
    case GGUF_T_U8:  return p[idx];
    case GGUF_T_U16: { uint16_t v; memcpy(&v, p + idx * 2, 2); return v; }
    case GGUF_T_U32: { uint32_t v; memcpy(&v, p + idx * 4, 4); return v; }
    case GGUF_T_I32: { int32_t v; memcpy(&v, p + idx * 4, 4);
                       return v >= 0 ? (uint64_t)v : dflt; }
    case GGUF_T_U64: { uint64_t v; memcpy(&v, p + idx * 8, 8); return v; }
    case GGUF_T_I64: { int64_t v; memcpy(&v, p + idx * 8, 8);
                       return v >= 0 ? (uint64_t)v : dflt; }
    default: return dflt;
    }
}

float gguf_get_f32(gguf_file *g, const char *key, float dflt) {
    gguf_kv *kv = gguf_get(g, key);
    if (!kv) return dflt;
    if (kv->type == GGUF_T_F32) {
        uint32_t u = (uint32_t)kv->raw;
        return raw_f32_finite(u) ? f32_from_bits(u) : dflt;
    }
    if (kv->type == GGUF_T_F64) {
        if (!raw_f64_finite(kv->raw)) return dflt;
        float f = (float)kv->v.f64;              // may overflow a huge double to inf
        return fp_isfinite(f) ? f : dflt;        // rejects overflow to inf (a
                                                 // bit test of a computed float
                                                 // folds; fpclass.h does not)
    }
    if (kv_is_unsigned(kv->type)) return (float)kv->v.u64;
    if (kv_is_signed(kv->type))   return (float)kv->v.i64;
    return dflt;   // ARR, STR, BOOL are not a scalar number
}

bool gguf_get_bool(gguf_file *g, const char *key, bool dflt) {
    gguf_kv *kv = gguf_get(g, key);
    return (kv && kv->type == GGUF_T_BOOL) ? kv->v.b : dflt;
}

const char *gguf_get_str(gguf_file *g, const char *key, const char *dflt) {
    gguf_kv *kv = gguf_get(g, key);
    return (kv && kv->type == GGUF_T_STR) ? kv->str.s : dflt;
}

gguf_tensor *gguf_find_tensor(gguf_file *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}
