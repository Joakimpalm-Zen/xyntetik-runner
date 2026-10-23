// NumPy .npy headers. The format (numpy.lib.format): the magic "\x93NUMPY",
// a major and minor version byte, the header length (uint16 little-endian in
// version 1, uint32 in versions 2 and 3), then the header itself, a Python
// dict literal such as
//
//   {'descr': '<f4', 'fortran_order': False, 'shape': (2000, 512, 64), }
//
// padded with spaces to a 64-byte boundary and ended by a newline. The data
// follows the header directly.
#include "npy.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define NPY_HEADER_MAX (1u << 20)

bool npy_seek(FILE *f, int64_t off) {
#ifdef _WIN32
    return _fseeki64(f, off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
}

int64_t npy_file_size(FILE *f) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
    return _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) return -1;
    return (int64_t)ftello(f);
#endif
}

// The text after `'key':`, spaces skipped; NULL when the key is absent.
static const char *npy_key(const char *hdr, const char *key) {
    char pat[32];
    snprintf(pat, sizeof pat, "'%s'", key);
    const char *p = strstr(hdr, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ') p++;
    return p;
}

static bool npy_fail(char *why, size_t cap, const char *msg) {
    if (why && cap) snprintf(why, cap, "%s", msg);
    return false;
}

bool npy_read_header(FILE *f, npy_header *h, char *why, size_t cap) {
    memset(h, 0, sizeof *h);
    unsigned char pre[12];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0)
        return npy_fail(why, cap, "not a .npy file (no \\x93NUMPY magic)");
    int major = pre[6];
    uint32_t hlen;
    int64_t pre_n;
    if (major == 1) {
        hlen = (uint32_t)pre[8] | (uint32_t)pre[9] << 8;
        pre_n = 10;
    } else if (major == 2 || major == 3) {
        if (fread(pre + 10, 1, 2, f) != 2)
            return npy_fail(why, cap, "truncated .npy header");
        hlen = (uint32_t)pre[8] | (uint32_t)pre[9] << 8 |
               (uint32_t)pre[10] << 16 | (uint32_t)pre[11] << 24;
        pre_n = 12;
    } else {
        return npy_fail(why, cap, "unsupported .npy format version");
    }
    if (hlen == 0 || hlen > NPY_HEADER_MAX)
        return npy_fail(why, cap, "implausible .npy header length");
    char *hdr = malloc((size_t)hlen + 1);
    if (!hdr) return npy_fail(why, cap, "out of memory reading a .npy header");
    if (fread(hdr, 1, hlen, f) != hlen) {
        free(hdr);
        return npy_fail(why, cap, "truncated .npy header");
    }
    hdr[hlen] = '\0';
    h->data_offset = pre_n + (int64_t)hlen;

    bool ok = false;
    const char *d = npy_key(hdr, "descr");
    const char *fo = npy_key(hdr, "fortran_order");
    const char *sh = npy_key(hdr, "shape");
    if (!d || (*d != '\'' && *d != '"') || !fo || !sh || *sh != '(') {
        npy_fail(why, cap, ".npy header lacks descr, fortran_order or shape");
        goto done;
    }
    {
        char q = *d++;
        size_t n = 0;
        while (d[n] && d[n] != q) n++;
        if (d[n] != q || n == 0 || n >= sizeof h->descr) {
            npy_fail(why, cap, ".npy descr is not a plain dtype string");
            goto done;
        }
        memcpy(h->descr, d, n);
        h->descr[n] = '\0';
    }
    if (strncmp(fo, "False", 5) == 0) h->fortran_order = false;
    else if (strncmp(fo, "True", 4) == 0) h->fortran_order = true;
    else {
        npy_fail(why, cap, ".npy fortran_order is neither True nor False");
        goto done;
    }
    {
        const char *p = sh + 1;
        for (;;) {
            while (*p == ' ') p++;
            if (*p == ')') break;
            if (!isdigit((unsigned char)*p) || h->ndim == NPY_MAX_DIM) {
                npy_fail(why, cap, ".npy shape is not a tuple of sizes");
                goto done;
            }
            char *end = NULL;
            long long v = strtoll(p, &end, 10);
            if (v < 0 || v > (long long)1 << 40) {
                npy_fail(why, cap, ".npy shape has an implausible size");
                goto done;
            }
            h->shape[h->ndim++] = (int64_t)v;
            p = end;
            while (*p == ' ') p++;
            if (*p == ',') p++;
            else if (*p != ')') {
                npy_fail(why, cap, ".npy shape is not a tuple of sizes");
                goto done;
            }
        }
    }
    ok = true;
done:
    free(hdr);
    return ok;
}
