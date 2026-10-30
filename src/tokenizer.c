// Tokenizers: SentencePiece-style (llama) and byte-level BPE (gpt2).
#include <math.h>
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4,
       TT_UNUSED = 5, TT_BYTE = 6 };

// Set by an encode helper when it drops a text segment because a temporary
// allocation failed; each public encode clears it on entry and returns -1 when
// it is set, so an OOM is never mistaken for a legitimately shorter prompt.
// Per THREAD, not per tokenizer: parallel server slots share one tokenizer
// and encode on their own threads, and the flag used to live in the shared
// struct, where one slot's clear could erase another's OOM (a race TSan
// reported under four concurrent requests; tests/test_tokenizer_race.c).
// An encode runs start to finish on the calling thread, so a thread-local
// flag is exactly per call.
static _Thread_local bool tls_encode_oom;

// ---------------------------------------------------------------- hashmap

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 0x100000001b3ull; }
    return h;
}

// Publishing the capacity before the table exists would leave hmap_put masking
// an index into a NULL `e`, so cap is set only once the allocation succeeded.
// hmap_get already treats a NULL table as empty.
static bool hmap_init(hmap *m, size_t expect) {
    size_t cap = 16;
    while (cap < expect * 2 && cap <= SIZE_MAX / 2) cap <<= 1;
    hmap_ent *e = calloc(cap, sizeof(hmap_ent));
    if (!e) { m->e = NULL; m->cap = 0; return false; }
    m->e = e;
    m->cap = cap;
    return true;
}

static void hmap_put(hmap *m, const char *k, size_t klen, int v) {
    size_t i = fnv1a(k, klen) & (m->cap - 1);
    while (m->e[i].key) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, k, klen) == 0) {
            return; // keep first entry (matches gguf duplicate handling)
        }
        i = (i + 1) & (m->cap - 1);
    }
    m->e[i] = (hmap_ent){ k, (uint32_t)klen, v };
}

static int hmap_get(const hmap *m, const char *k, size_t klen) {
    if (!m->e) return -1;
    size_t i = fnv1a(k, klen) & (m->cap - 1);
    while (m->e[i].key) {
        if (m->e[i].klen == klen && memcmp(m->e[i].key, k, klen) == 0)
            return m->e[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

// ---------------------------------------------------------------- utf8

static int u8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static uint32_t u8_decode(const char *s, int len) {
    const uint8_t *p = (const uint8_t *)s;
    switch (len) {
        case 2: return ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        case 3: return ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        case 4: return ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) |
                       ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        default: return p[0];
    }
}

static int u8_encode(uint32_t cp, char *out) {
    if (cp < 0x80)  { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// ---------------------------------------------------------------- nfc
//
// Unicode normalization form C, for the tokenizers whose publisher declares
// it ({"type": "NFC"} in Qwen's tokenizer.json, Qwen2 through Qwen3.8). The
// reference tokenizer normalizes every run of text between special tokens
// before it splits it, so "cafe" + U+0301 is the one token " café" there and
// was two tokens here until 2026-10-16 (found by the Clef gold run). The
// algorithm is UAX #15's: full canonical decomposition, canonical ordering,
// canonical composition; the tables are generated (scripts/gen-unicode-nfc.py).

#include "unicode_nfc.h"

static uint32_t nfc_class(uint32_t cp) {
    size_t lo = 0, hi = sizeof nfc_ccc / sizeof *nfc_ccc;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < nfc_ccc[mid][0]) hi = mid;
        else if (cp > nfc_ccc[mid][1]) lo = mid + 1;
        else return nfc_ccc[mid][2];
    }
    return 0;
}

enum { NFC_SBASE = 0xAC00, NFC_LBASE = 0x1100, NFC_VBASE = 0x1161, NFC_TBASE = 0x11A7,
       NFC_LCOUNT = 19, NFC_VCOUNT = 21, NFC_TCOUNT = 28,
       NFC_NCOUNT = NFC_VCOUNT * NFC_TCOUNT, NFC_SCOUNT = NFC_LCOUNT * NFC_NCOUNT };

// the full canonical decomposition of cp into out (at most 4 code points)
static int nfc_decompose(uint32_t cp, uint32_t *out) {
    if (cp >= NFC_SBASE && cp < NFC_SBASE + NFC_SCOUNT) {
        uint32_t si = cp - NFC_SBASE;
        out[0] = NFC_LBASE + si / NFC_NCOUNT;
        out[1] = NFC_VBASE + (si % NFC_NCOUNT) / NFC_TCOUNT;
        uint32_t t = si % NFC_TCOUNT;
        if (t) { out[2] = NFC_TBASE + t; return 3; }
        return 2;
    }
    size_t lo = 0, hi = sizeof nfc_dec / sizeof *nfc_dec;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < nfc_dec[mid][0]) hi = mid;
        else if (cp > nfc_dec[mid][0]) lo = mid + 1;
        else {
            uint32_t n = nfc_dec[mid][2];
            for (uint32_t i = 0; i < n; i++) out[i] = nfc_dec_pool[nfc_dec[mid][1] + i];
            return (int)n;
        }
    }
    out[0] = cp;
    return 1;
}

// the primary composite of (a, b), or 0
static uint32_t nfc_compose(uint32_t a, uint32_t b) {
    if (a >= NFC_LBASE && a < NFC_LBASE + NFC_LCOUNT &&
        b >= NFC_VBASE && b < NFC_VBASE + NFC_VCOUNT)
        return NFC_SBASE + ((a - NFC_LBASE) * NFC_VCOUNT + (b - NFC_VBASE)) * NFC_TCOUNT;
    if (a >= NFC_SBASE && a < NFC_SBASE + NFC_SCOUNT && (a - NFC_SBASE) % NFC_TCOUNT == 0 &&
        b > NFC_TBASE && b < NFC_TBASE + NFC_TCOUNT)
        return a + (b - NFC_TBASE);
    size_t lo = 0, hi = sizeof nfc_comp / sizeof *nfc_comp;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        uint32_t ma = nfc_comp[mid][0], mb = nfc_comp[mid][1];
        if (a < ma || (a == ma && b < mb)) hi = mid;
        else if (a > ma || (a == ma && b > mb)) lo = mid + 1;
        else return nfc_comp[mid][2];
    }
    return 0;
}

// strict UTF-8 decode of s[0..n) into cp; false on any malformed sequence
static bool nfc_utf8_decode(const char *s, size_t n, uint32_t *cp, size_t *ncp) {
    size_t k = 0;
    for (size_t i = 0; i < n; ) {
        uint8_t c = (uint8_t)s[i];
        int l = u8_len(c);
        if ((l == 1 && c >= 0x80) || i + (size_t)l > n) return false;
        for (int j = 1; j < l; j++)
            if (((uint8_t)s[i + j] & 0xC0) != 0x80) return false;
        uint32_t v = u8_decode(s + i, l);
        if ((l == 2 && v < 0x80) || (l == 3 && v < 0x800) || (l == 4 && v < 0x10000) ||
            v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF))
            return false;
        cp[k++] = v;
        i += (size_t)l;
    }
    *ncp = k;
    return true;
}

// NFC of s[0..n) as a malloc'd string in *out (length *out_n), or false when
// the text needs no change: nothing at or above U+0300, or not valid UTF-8
// (which the reference could not have been given, so it is left as bytes).
// *oom is set when an allocation failed.
static bool nfc_utf8(const char *s, size_t n, char **out, size_t *out_n, bool *oom) {
    *out = NULL;
    bool any = false;
    for (size_t i = 0; i < n; i++)
        if ((uint8_t)s[i] >= 0xCC) { any = true; break; }   // lead byte of U+0300+
    if (!any || n > SIZE_MAX / (4 * sizeof(uint32_t)) - 1) return false;
    uint32_t *cp = malloc(sizeof(uint32_t) * (n + 1));
    uint32_t *d = malloc(sizeof(uint32_t) * (4 * n + 1));
    if (!cp || !d) { free(cp); free(d); *oom = true; return false; }
    size_t ncp = 0;
    if (!nfc_utf8_decode(s, n, cp, &ncp)) { free(cp); free(d); return false; }
    size_t nd = 0;
    for (size_t i = 0; i < ncp; i++) nd += (size_t)nfc_decompose(cp[i], d + nd);
    free(cp);
    // canonical ordering: a stable sort of each run of non-starters by class
    // (every class is 1..240, so one byte holds it, looked up once). A run is
    // a handful of marks in real text, and a short one sorts by insertion. A
    // long one sorts by counting: the insertion sort that stood here alone
    // was O(r^2) in the run, and a request can be one run of marks out of
    // class order (2026-10-16: 128 KB took 8.4 s before inference began).
    // Both sorts are stable, so marks of one class keep their order.
    uint8_t *cls = malloc(nd + 1);
    uint32_t *tmp = NULL;   // the counting sort's scratch, on the first long run
    if (!cls) { free(d); *oom = true; return false; }
    for (size_t i = 0; i < nd; i++) cls[i] = (uint8_t)nfc_class(d[i]);
    for (size_t a = 0, b; a < nd; a = b) {
        if (!cls[a]) { b = a + 1; continue; }
        for (b = a + 1; b < nd && cls[b]; b++) {}
        if (b - a <= 32) {
            for (size_t i = a + 1; i < b; i++) {
                uint32_t c = d[i];
                uint8_t k = cls[i];
                size_t j = i;
                for (; j > a && cls[j - 1] > k; j--) { d[j] = d[j - 1]; cls[j] = cls[j - 1]; }
                d[j] = c;
                cls[j] = k;
            }
            continue;
        }
        if (!tmp && !(tmp = malloc(sizeof(uint32_t) * nd))) {
            free(cls); free(d); *oom = true; return false;
        }
        size_t count[256] = { 0 }, at[256];
        for (size_t i = a; i < b; i++) count[cls[i]]++;
        for (size_t k = 0, s = a; k < 256; k++) { at[k] = s; s += count[k]; }
        for (size_t i = a; i < b; i++) tmp[at[cls[i]]++] = d[i];
        memcpy(d + a, tmp + a, sizeof(uint32_t) * (b - a));
        for (size_t k = 1, s = a; k < 256; k++) {
            memset(cls + s, (int)k, count[k]);
            s += count[k];
        }
    }
    free(tmp);
    // canonical composition (UAX #15, the sample algorithm); it writes d only
    // at or before the index it reads, so cls[i] is still d[i]'s class
    size_t w = 0;
    if (nd > 0) {
        size_t starter = 0;
        uint32_t last = cls[0] ? 256 : 0;
        w = 1;
        for (size_t i = 1; i < nd; i++) {
            uint32_t c = d[i], k = cls[i];
            uint32_t comp = (last == 256) ? 0 : nfc_compose(d[starter], c);
            if (comp && (last < k || last == 0)) {
                d[starter] = comp;
            } else {
                if (k == 0) starter = w;
                last = k;
                d[w++] = c;
            }
        }
    }
    free(cls);
    char *o = malloc(4 * w + 1);
    if (!o) { free(d); *oom = true; return false; }
    size_t on = 0;
    for (size_t i = 0; i < w; i++) on += (size_t)u8_encode(d[i], o + on);
    o[on] = 0;
    free(d);
    *out = o;
    *out_n = on;
    return true;
}

// ---------------------------------------------------------------- init

// Length descending, then id ascending. The id tiebreak is what lets plain
// qsort stand in for the stable insertion sort this replaced: the list is
// built in ascending id order, so a stable sort leaves equal-length runs in id
// order and this comparator reproduces that exactly. It also makes the
// ordering a TOTAL order, which qsort requires and a bare length compare is
// not.
typedef struct { uint64_t n; int id; } special_key;

static int cmp_special_key(const void *a, const void *b) {
    const special_key *x = a, *y = b;
    if (x->n != y->n) return x->n < y->n ? 1 : -1;
    return x->id < y->id ? -1 : x->id > y->id ? 1 : 0;
}

// qsort_r is not portable, so the ordering key is materialised rather than
// passed as context.
//
// This was an insertion sort, on the reading that "special token lists are
// tiny". That is the FILE's choice, not an invariant: n_special is built from
// tokenizer.ggml.token_type, one entry per vocabulary token, with no cap. A
// GGUF whose specials fall into two large equal-length runs in ascending order
// is the worst case for any quadratic sort, and the cost is exactly n^2 --
// measured on a generated fixture: 1.2 s at 100k specials, 4.8 s at 200k,
// 11.4 s at 300k, minutes at the vocabulary sizes a real download can carry.
// In --serve that time is spent inside a swap, so one request naming a hostile
// model parks the slot for the whole of it.
static bool sort_specials(tokenizer *t) {
    if (t->n_special <= 1) return true;
    special_key *keys = malloc(sizeof(*keys) * (size_t)t->n_special);
    if (!keys) return false;
    for (int i = 0; i < t->n_special; i++)
        keys[i] = (special_key){ t->tokens[t->special_ids[i]].n,
                                 t->special_ids[i] };
    qsort(keys, (size_t)t->n_special, sizeof(*keys), cmp_special_key);
    for (int i = 0; i < t->n_special; i++) t->special_ids[i] = keys[i].id;
    free(keys);
    return true;
}

// Regroup the sorted special list by first byte, in place of an n_vocab-sized
// array that tok_encode had to walk end to end at EVERY input byte. A match
// requires all of a token's bytes to be equal, so a candidate whose first byte
// differs could never have matched: the grouping removes work, not candidates,
// and a counting sort keeps each group in the length-descending order that
// decides which of several matches wins.
static bool bucket_specials(tokenizer *t) {
    int *out = malloc(sizeof(int) * (size_t)(t->n_special > 0 ? t->n_special : 1));
    if (!out) return false;
    int pos[256];
    memset(t->sb_off, 0, sizeof(t->sb_off));
    for (int i = 0; i < t->n_special; i++)
        t->sb_off[(uint8_t)t->tokens[t->special_ids[i]].s[0] + 1]++;
    for (int b = 0; b < 256; b++) {
        t->sb_off[b + 1] += t->sb_off[b];
        pos[b] = t->sb_off[b];
    }
    for (int i = 0; i < t->n_special; i++)
        out[pos[(uint8_t)t->tokens[t->special_ids[i]].s[0]]++] = t->special_ids[i];
    free(t->special_ids);
    t->special_ids = out;
    return true;
}

// All-zero scores carry no ordering, so the merge loop in spm_encode would
// just take the leftmost candidate every round.
static bool spm_scores_degenerate(const tokenizer *t) {
    if (!t->scores) return true;
    for (int i = 0; i < t->n_vocab; i++)
        if (t->scores[i] != 0.0f) return false;
    return true;
}

// Rebuild scores from merge rank, highest score first, so spm_encode picks
// merges in BPE order. A piece that is no merge's result gets -inf: BPE only
// ever produces multi-character pieces through a merge, so anything absent
// here must never be merged into. Requires t->vocab to be populated.
//
// Returns false only when an allocation failed. A model that simply carries no
// merges is left exactly as it was and still reports success — that is not an
// error, and the caller must not abort the load over it.
static bool spm_scores_from_merges(tokenizer *t, gguf_file *g) {
    gguf_kv *mg = gguf_get(g, "tokenizer.ggml.merges");
    if (!mg || mg->type != GGUF_T_ARR || mg->arr_type != GGUF_T_STR || mg->arr_n == 0)
        return true;

    if (!t->scores) {
        t->scores = malloc(sizeof(float) * (size_t)t->n_vocab);
        if (!t->scores) return false;
    }
    for (int i = 0; i < t->n_vocab; i++) t->scores[i] = -FLT_MAX;

    char buf[512];
    for (uint64_t r = 0; r < mg->arr_n; r++) {
        const char *m = mg->arr_str[r].s;
        size_t n = mg->arr_str[r].n;
        // "left right": split on the first space, the pieces themselves use
        // U+2581 rather than a literal space
        const char *sep = memchr(m, ' ', n);
        if (!sep || n - 1 >= sizeof(buf)) continue;
        size_t left = (size_t)(sep - m);
        memcpy(buf, m, left);
        memcpy(buf + left, sep + 1, n - left - 1);
        int id = hmap_get(&t->vocab, buf, n - 1);
        if (id >= 0) t->scores[id] = -(float)r;
    }
    return true;
}

// bos/eos/unk are not labels: they are indices the engine writes through. bos
// is emitted into the token stream, the stream seeds the sampler's penalty
// window, and the penalty is `logits[tok] /= repeat_penalty` — a
// read-modify-write at whatever index the file named. 0xFFFFFFFF is the
// conventional "this model has none" and stays legal; anything else outside
// [0, n_vocab) is a broken file, and a broken file fails closed at load.
static bool special_id_ok(const tokenizer *t, const char *key, int id) {
    if (id == -1 || (id >= 0 && id < t->n_vocab)) return true;
    fprintf(stderr, "error: %s is %d, outside the %d-token vocabulary\n",
            key, id, t->n_vocab);
    return false;
}

bool tokenizer_init(tokenizer *t, gguf_file *g) {
    memset(t, 0, sizeof(*t));

    const char *model = gguf_get_str(g, "tokenizer.ggml.model", "llama");
    if (strcmp(model, "llama") == 0) t->model = TOK_SPM;
    else if (strcmp(model, "gpt2") == 0) t->model = TOK_BPE;
    else if (strcmp(model, "gemma4") == 0) t->model = TOK_BPE_SPM;
    else {
        fprintf(stderr, "error: unsupported tokenizer model '%s'\n", model);
        return false;
    }

    // Split rules for the BPE families we have ground truth for. Everything
    // else, including a missing key, keeps the original GPT-2 behavior rather
    // than being retokenized by rules it was not checked against; a BPE file
    // that names a split this table lacks is told so (pre_unknown, below),
    // because GPT-2 is then a guess, not the file's rule.
    const char *pre = gguf_get_str(g, "tokenizer.ggml.pre", "");
    if (strcmp(pre, "llama-bpe") == 0)   t->pre = TOK_PRE_LLAMA3;
    else if (strcmp(pre, "dbrx") == 0)   t->pre = TOK_PRE_LLAMA3; // llama.cpp: "same as llama3" (granite 4.1 ships this)
    // Liquid LFM2 / LFM2.5: the publisher's tokenizer.json splits with the
    // Llama 3 regex (whether it also ignores merges depends on the
    // vocabulary; see ignore_merges below)
    else if (strcmp(pre, "lfm2") == 0)   t->pre = TOK_PRE_LLAMA3;
    else if (strcmp(pre, "granite-docling") == 0) t->pre = TOK_PRE_GPT2; // llama.cpp: the plain GPT-2 regex (granite 4.2 ships this)
    // minicpm5 (openbmb/MiniCPM5-2B tokenizer.json at f9740005): a Split on
    // \p{N}{1,3} FOLLOWED BY the Llama 3 regex, applied in that order
    // (llama.cpp's MINICPM5 table carries the two expressions verbatim). The
    // digits fall into groups of three and letters absorb one leading
    // punctuation mark ("-level"), as under llama-bpe; the GPT-2 fallback
    // read 210 of the 721 corpus strings differently. The order matters
    // once: a whitespace run that ends at a digit group is a whole piece
    // before the Llama 3 regex ever sees it, so its `\s+(?!\S)` keeps the
    // run entire ("  1" is "  ", "1"), where llama-bpe's single pass splits
    // off the last space. Three of the 721 strings turn on that.
    else if (strcmp(pre, "minicpm5") == 0) t->pre = TOK_PRE_MINICPM5;
    // spark2_5 (XHToken/Spark-X2.6-4B tokenizer.json at d1b5f260, identical in
    // X2.5): \p{N}{1,3}, then CJK runs, then afmoe's main regex with its
    // newline clauses narrowed, then Digits(individual). llama.cpp's
    // SPARK2_5 table carries the same expressions (spark25_split_next).
    else if (strcmp(pre, "spark2_5") == 0) t->pre = TOK_PRE_SPARK25;
    // k2-horizon (IFM/K2-Horizon tokenizer.json): one Split,
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+
    //   | \p{N}{1,3} |  ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    // under NFC. llama.cpp's K2_HORIZON table carries the same expression.
    else if (strcmp(pre, "k2-horizon") == 0) t->pre = TOK_PRE_K2H;
    else if (strcmp(pre, "qwen2") == 0)  t->pre = TOK_PRE_QWEN2;
    // stablelm2: llama.cpp puts LLAMA_VOCAB_PRE_TYPE_STABLELM2 in the same
    // case as QWEN2, so it is the qwen2 regex, whose \p{N} matches ONE digit.
    // Without this row the file fell through to the plain GPT-2 default, which
    // puts no cap on a digit run, and 259 of 721 corpus strings tokenized
    // differently from the publisher's own tokenizer: "1234567890" came out as
    // 123/456/78/90 where the reference gives ten single digits (2026-09-07).
    else if (strcmp(pre, "stablelm2") == 0) t->pre = TOK_PRE_QWEN2;
    // qwen35 (Qwen 3.5, 3.6, 3.8, Ornith): the qwen2 regex with [\p{L}\p{M}]+
    // in place of \p{L}+, so combining marks (Devanagari vowel signs and
    // viramas, Thai vowels and tones) ride inside the letter run and are
    // excluded from the symbol run; qwen2 (Qwen 3) splits at every mark
    else if (strcmp(pre, "qwen35") == 0) t->pre = TOK_PRE_QWEN35;
    else if (strcmp(pre, "smollm") == 0) t->pre = TOK_PRE_SMOLLM;
    // mellum2 (JetBrains Mellum2.x): Digits(individual_digits) then the GPT-2
    // byte-level regex, smollm's pre-tokenizer exactly
    else if (strcmp(pre, "mellum2") == 0) t->pre = TOK_PRE_SMOLLM;
    else if (strcmp(pre, "afmoe") == 0)  t->pre = TOK_PRE_AFMOE;
    else if (strcmp(pre, "flint") == 0)  t->pre = TOK_PRE_FLINT;
    else if (strcmp(pre, "tekken") == 0) t->pre = TOK_PRE_TEKKEN;
    // pixtral: llama.cpp's name for the Tekken regex with its Unicode letter
    // classes, the one TOK_PRE_TEKKEN implements; NVIDIA's Nemotron 3.5
    // Lightning ships it (its tokenizer.json carries exactly that regex).
    // Before this row the file took the GPT-2 split: "<p" became "<" + "p",
    // a 1000-byte README prompt 353 tokens against the publisher's 298, and
    // 571 of the 721 corpus strings matched the publisher's tokenizer; with
    // it, all 721 (2026-10-23). Its BOS default stays GPT-2's: the
    // publisher's post-processor adds none.
    else if (strcmp(pre, "pixtral") == 0) t->pre = TOK_PRE_TEKKEN;
    else if (strcmp(pre, "llama4") == 0 ||
             strcmp(pre, "gpt-4o") == 0) t->pre = TOK_PRE_LLAMA4;
    else {
        t->pre = TOK_PRE_GPT2;
        // the name only chooses the split of a plain BPE vocabulary
        // (SentencePiece and gemma4's BPE-over-SPM never read it); an absent
        // key is the plain GPT-2 file and says nothing to report
        if (t->model == TOK_BPE && pre[0]) {
            size_t i = 0;
            for (; pre[i] && i < sizeof t->pre_unknown - 1; i++) {
                char c = pre[i];
                bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
                t->pre_unknown[i] = ok ? c : '?';
            }
            t->pre_unknown[i] = 0;
            fprintf(stderr, "tokenizer: pre '%s' unknown, GPT-2 split assumed "
                            "(text may tokenize differently from the publisher's tokenizer)\n",
                    t->pre_unknown);
        }
    }
    // Clef (Cloudflare's Qwen3.8 and Qwen3.5 post-trains) was trained with
    // the qwen2 split: its tokenizer.json carries \p{L}+ where Qwen's own
    // carries [\p{L}\p{M}]+, and transformers builds the qwen2 regex for it
    // (the publisher's encode_record splits "\u0939\u093f\u0928\u094d..." at
    // every mark). llama.cpp's converter still writes "qwen35" into every
    // Clef GGUF, so the declared name is read as the trained-with split here;
    // text with combining marks tokenizes differently from llama.cpp by design.
    // XiaomiMiMo's Qwen3.5 distill (MiMo-V2.6-Distill-Qwen-9B) is the same
    // case on an ordinary qwen35 file: its tokenizer.json at 1f09ddcc carries
    // the qwen2 regex (\p{L}+, marks never join a letter run) over Qwen3.5's
    // vocabulary, and the converter wrote "qwen35". Its architecture names
    // nothing, so the file is told by its own chat template: the compact
    // `<tool_call><function=` opener only MiMo's writes (template.c,
    // TMPL_MIMO). Measured on the bartowski Q4_K_M: 3 of 721 corpus strings,
    // all Devanagari or Thai, differed under the qwen35 split, 0 under qwen2.
    const char *chat_tmpl = gguf_get_str(g, "tokenizer.chat_template", "");
    if (t->pre == TOK_PRE_QWEN35 &&
        (strcmp(gguf_get_str(g, "general.architecture", ""), "clef") == 0 ||
         strstr(chat_tmpl, "<tool_call><function=")))
        t->pre = TOK_PRE_QWEN2;
    // Qwen's tokenizer.json normalizes with NFC (Qwen2 through Qwen3.8, and
    // so Clef); stablelm2 shares the qwen2 split but not that normalizer
    t->nfc = strcmp(pre, "qwen2") == 0 || strcmp(pre, "qwen35") == 0 ||
             strcmp(pre, "k2-horizon") == 0;

    gguf_kv *toks = gguf_get(g, "tokenizer.ggml.tokens");
    if (!toks || toks->type != GGUF_T_ARR || toks->arr_type != GGUF_T_STR) {
        fprintf(stderr, "error: no tokenizer vocabulary in model\n");
        return false;
    }
    // arr_n is a 64-bit count straight out of the file; a value past INT_MAX
    // would wrap n_vocab negative and turn every `sizeof(x) * n_vocab` below
    // into a huge or negative allocation size.
    if (toks->arr_n > INT_MAX) {
        fprintf(stderr, "error: tokenizer vocabulary is implausibly large\n");
        return false;
    }
    t->n_vocab = (int)toks->arr_n;
    t->tokens = toks->arr_str;
    // The BPE families whose publishers' tokenizer.json sets "ignore_merges"
    // (tiktoken's rule: a pre-token that is itself a vocabulary entry is that
    // token, merges or not). The GGUF carries no key for it, so the split's
    // name decides, each read from the publisher's file: Llama 3
    // (Llama-3.2-3B-Instruct), Llama 4 and its gpt-4o-style relatives
    // (Llama-4-Scout, Muse-Glimmer-30B, gpt-oss-20b), Tekken (Mistral-Nemo,
    // Apertus, and Nemotron 3.5 Lightning as "pixtral") and LFM2.5-8B-A1B's
    // and 24B-A2B's 128,000-token vocabulary under "lfm2". The same name
    // covers LFM2.5-1.2B's and 350M's 65,536 tokens, whose tokenizer.json
    // leaves the flag off; there the rule changes nothing, because the only
    // entries its merges do not build whole are its 507 special and reserved
    // tokens, which no split yields as a pre-token. Every other family here
    // leaves the flag off (granite, Qwen, MiniCPM5, Spark, K2-Horizon,
    // Mellum, Trinity, flint). Without it the merges built " l" + "äs" + "a"
    // where LFM2.5-8B's tokenizer reads " läsa" (7 of the gold corpus's 107
    // passages) and " Vi" + "ệ" + "t" for Llama 3's " Việt" (1 of the 721
    // corpus strings).
    t->ignore_merges = t->model == TOK_BPE &&
        (strcmp(pre, "llama-bpe") == 0 || strcmp(pre, "llama4") == 0 ||
         strcmp(pre, "gpt-4o") == 0 || strcmp(pre, "tekken") == 0 ||
         strcmp(pre, "pixtral") == 0 || strcmp(pre, "lfm2") == 0);

    gguf_kv *scores = gguf_get(g, "tokenizer.ggml.scores");
    if (scores && scores->type == GGUF_T_ARR && scores->arr_type == GGUF_T_F32 &&
        (int)scores->arr_n == t->n_vocab) {
        t->scores = malloc(sizeof(float) * (size_t)t->n_vocab);
        if (!t->scores) return false;
        memcpy(t->scores, scores->arr_raw, sizeof(float) * (size_t)t->n_vocab);
    }
    gguf_kv *tty = gguf_get(g, "tokenizer.ggml.token_type");
    if (tty && tty->type == GGUF_T_ARR && tty->arr_type == GGUF_T_I32 &&
        (int)tty->arr_n == t->n_vocab) {
        t->ttype = malloc(sizeof(int32_t) * (size_t)t->n_vocab);
        if (!t->ttype) return false;
        memcpy(t->ttype, tty->arr_raw, sizeof(int32_t) * (size_t)t->n_vocab);
    }

    t->bos_id = (int)gguf_get_u32(g, "tokenizer.ggml.bos_token_id", 1);
    t->eos_id = (int)gguf_get_u32(g, "tokenizer.ggml.eos_token_id", 2);
    t->unk_id = (int)gguf_get_u32(g, "tokenizer.ggml.unknown_token_id", -1u);
    if (!special_id_ok(t, "tokenizer.ggml.bos_token_id", t->bos_id) ||
        !special_id_ok(t, "tokenizer.ggml.eos_token_id", t->eos_id) ||
        !special_id_ok(t, "tokenizer.ggml.unknown_token_id", t->unk_id))
        return false;
    // No llama.cpp converter writes this key; some publishers' converters do
    // (flint's lists <|user|>, <|observation|> and <|developer|> beside
    // <|end_of_text|>). An id outside the vocabulary refuses the file, as a
    // bad eos id does.
    gguf_kv *eog = gguf_get(g, "tokenizer.ggml.eog_token_ids");
    if (eog && eog->type == GGUF_T_ARR &&
        (eog->arr_type == GGUF_T_I32 || eog->arr_type == GGUF_T_U32)) {
        if (eog->arr_n > TOK_EOG_MAX) {
            fprintf(stderr, "error: tokenizer.ggml.eog_token_ids lists %llu ids; "
                    "at most %d are supported\n",
                    (unsigned long long)eog->arr_n, TOK_EOG_MAX);
            return false;
        }
        for (uint64_t i = 0; i < eog->arr_n; i++) {
            int32_t id;
            memcpy(&id, (const char *)eog->arr_raw + i * 4, 4);
            if (id < 0 || id >= t->n_vocab) {
                fprintf(stderr, "error: tokenizer.ggml.eog_token_ids holds %d, "
                        "outside the %d-token vocabulary\n", id, t->n_vocab);
                return false;
            }
            t->eog_ids[t->n_eog++] = id;
        }
    }
    // A file that does not say whether to prepend BOS gets what its family's
    // own tokenizer does: SentencePiece vocabularies, and the two BPE families
    // whose publishers' tokenizers prepend it, Llama 3 (<|begin_of_text|>,
    // tokenizer.ggml.pre "llama-bpe") and Mistral's Tekken (<s>). Llama 3
    // conversions commonly carry no add_bos_token key, and reading that as
    // "no BOS" ran their chat and raw prompts without the token the model
    // always saw first (the chat template leaves BOS to the tokenizer);
    // llama.cpp reads a missing key the same way. An explicit key wins.
    // LFM2's tokenizer.json prepends <|startoftext|> too (its post-processor
    // is a TemplateProcessing of `<|startoftext|> $A`), and LFM2.5-24B-A2B's
    // GGUF carries no add_bos_token key.
    bool bos_default = t->model == TOK_SPM ||
                       (t->model == TOK_BPE && (strcmp(pre, "llama-bpe") == 0 ||
                                                strcmp(pre, "tekken") == 0 ||
                                                strcmp(pre, "lfm2") == 0));
    t->add_bos = gguf_get_bool(g, "tokenizer.ggml.add_bos_token", bos_default);
    // minicpm5: the publisher's tokenizer.json prepends <s> to every encode
    // (its post_processor is a TemplateProcessing of `<s> $A`), and the chat
    // template opens with `{{- bos_token }}`; the file's add_bos_token=false
    // is the converter's copy of a tokenizer_config flag that governs the
    // slow tokenizer, not the one the publisher ships. The model saw <s>
    // first in every prompt, so the artifact's flag yields to the
    // publisher's tokenizer here, and the template leaves BOS to the
    // tokenizer as every family does (template-conformance allowlist
    // minicpm5/bos).
    if (strcmp(pre, "minicpm5") == 0) t->add_bos = true;
    // k2-horizon: the same case. IFM's tokenizer.json prepends
    // <|ifm|begin_of_text|> (a TemplateProcessing of `BOS $A`) and the GGUF
    // template opens with `{{- bos_token }}`, while the GGUF carries no
    // add_bos_token key, which the BPE default reads as no BOS.
    if (strcmp(pre, "k2-horizon") == 0) t->add_bos = true;
    t->add_space_prefix = gguf_get_bool(g, "tokenizer.ggml.add_space_prefix", true);
    t->add_eos = gguf_get_bool(g, "tokenizer.ggml.add_eos_token", false);

    if (!hmap_init(&t->vocab, (size_t)t->n_vocab)) return false;
    for (int i = 0; i < t->n_vocab; i++)
        hmap_put(&t->vocab, t->tokens[i].s, t->tokens[i].n, i);

    // Many conversions (TheBloke's GGUFs among them) write all-zero scores and
    // put the ordering in tokenizer.ggml.merges instead. Without this, every
    // score ties and "llama" encodes as "▁llam"+"a" instead of "▁ll"+"ama".
    // Models that carry neither usable scores nor merges are left as they were.
    if (t->model == TOK_SPM && spm_scores_degenerate(t) &&
        !spm_scores_from_merges(t, g))
        return false;

    // special tokens (control + user-defined) for input parsing
    t->special_ids = malloc(sizeof(int) * (size_t)t->n_vocab);
    if (!t->special_ids) return false;
    for (int i = 0; i < t->n_vocab; i++) {
        int tt = t->ttype ? t->ttype[i] : TT_NORMAL;
        if ((tt == TT_CONTROL || tt == TT_USER_DEFINED) && t->tokens[i].n > 0)
            t->special_ids[t->n_special++] = i;
    }
    if (!sort_specials(t)) return false;
    if (!bucket_specials(t)) return false;

    if (t->model == TOK_BPE || t->model == TOK_BPE_SPM) {
        // GPT-2 byte <-> unicode mapping (unused by BPE_SPM, harmless)
        for (int i = 0; i < 512; i++) t->u2b[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            int keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                       (b >= 0xAE && b <= 0xFF);
            int cp = keep ? b : 256 + n++;
            t->b2u[b] = cp;
            t->u2b[cp] = b;
        }
        // merge ranks
        gguf_kv *mg = gguf_get(g, "tokenizer.ggml.merges");
        if (!mg || mg->type != GGUF_T_ARR || mg->arr_type != GGUF_T_STR) {
            fprintf(stderr, "error: BPE tokenizer has no merges\n");
            return false;
        }
        if (!hmap_init(&t->merges, mg->arr_n)) return false;
        for (uint64_t i = 0; i < mg->arr_n; i++)
            hmap_put(&t->merges, mg->arr_str[i].s, mg->arr_str[i].n, (int)i);
    }
    return true;
}

void tokenizer_free(tokenizer *t) {
    free(t->scores);
    free(t->ttype);
    free(t->vocab.e);
    free(t->merges.e);
    free(t->merges_buf);
    free(t->special_ids);
    memset(t, 0, sizeof(*t));
}

// ---------------------------------------------------------------- SPM encode

// greedy highest-score bigram merging over a doubly linked symbol list
typedef struct { int start, len, prev, next; } sym_t;

// ------------------------------------------------------- merge candidates
//
// Both merge loops used to rescan every adjacent pair after every merge, which
// is O(n^2) in the length of one segment — and a segment is not a word. The
// SentencePiece path never splits at all, and the gemma-4 BPE path splits only
// on newline runs, so one long line is one unit. Measured on 4,000 characters
// of ordinary prose in a single line: 31 ms on the certified European
// SentencePiece models, 62 ms on gemma-4-E4B, against 0.31 ms on Qwen2.5,
// whose GPT-2 pre-tokenizer hands these loops a word at a time. At 16,000
// characters gemma-4 took 940 ms. Nine of the fourteen models on the bench box
// are affected, including five of the six certified European ones.
//
// The queue makes it O(n log n) and is required to be EXACT: the tokenizer
// feeds ids to the model, so any change of output would invalidate every
// greedy_reference certification. Ordering therefore reproduces the old scan
// exactly — best key first, and on a tie the leftmost pair, because the old
// loops compared with a strict `>` / `<` while walking left to right.
//
// `key` is "smaller is better": the negated piece score for SentencePiece, the
// merge rank for BPE. Ranks are far below 2^24 so a float holds them exactly.
typedef struct { float key; int l, r, len_l, len_r; } mcand;

static bool mc_better(const mcand *a, const mcand *b) {
    if (a->key != b->key) return a->key < b->key;
    return a->l < b->l;
}

// The queue starts in a fixed buffer. A GPT-2 style pre-tokenizer hands these
// loops a word at a time — three or four symbols — and for those the malloc
// was the whole cost: the first measured version of this change was 20% SLOWER
// on Qwen2.5 and gpt-oss while being 80x faster on gemma-4. Nothing allocates
// until a segment actually produces more candidates than fit here.
// Below this many symbols the rescan wins outright: a GPT-2 pre-tokenizer
// hands these loops three or four symbols, where the queue's push/pop
// bookkeeping costs more than re-reading every pair. Measured on a 4,000-char
// prompt: Qwen2.5 0.31 ms rescan against 0.39 ms queue. Both paths are
// verified byte-identical over the differential corpus, so which one runs is
// purely a cost decision.
#define MERGE_QUEUE_MIN 24
#define MH_FIXED 64

// Test hook: force the rescan (0) or the queue (1) regardless of segment
// length; -1 restores the length rule. The two paths are required to agree on
// every input, and tests/test_tokenizer_merge.c is that requirement -- without
// a way to run both on one binary the only gate would be a diff against an old
// build, which nobody runs.
static int g_merge_force = -1;
void tok_merge_force(int on) { g_merge_force = on; }
static bool use_merge_queue(int n_sym) {
    return g_merge_force >= 0 ? g_merge_force != 0 : n_sym >= MERGE_QUEUE_MIN;
}
typedef struct { mcand *e; int n, cap; mcand fixed[MH_FIXED]; } mheap;

static void mh_init(mheap *h) { h->e = h->fixed; h->n = 0; h->cap = MH_FIXED; }
static void mh_free(mheap *h) { if (h->e != h->fixed) free(h->e); }

static bool mh_push(mheap *h, mcand c) {
    if (h->n == h->cap) {
        int cap = h->cap * 2;
        mcand *e;
        if (h->e == h->fixed) {
            e = malloc(sizeof(mcand) * (size_t)cap);
            if (e) memcpy(e, h->fixed, sizeof(mcand) * (size_t)h->n);
        } else {
            e = realloc(h->e, sizeof(mcand) * (size_t)cap);
        }
        if (!e) return false;
        h->e = e;
        h->cap = cap;
    }
    int i = h->n++;
    h->e[i] = c;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!mc_better(&h->e[i], &h->e[p])) break;
        mcand tmp = h->e[p]; h->e[p] = h->e[i]; h->e[i] = tmp;
        i = p;
    }
    return true;
}

static bool mh_pop(mheap *h, mcand *out) {
    if (h->n == 0) return false;
    *out = h->e[0];
    h->e[0] = h->e[--h->n];
    for (int i = 0;;) {
        int a = 2 * i + 1, b = a + 1, best = i;
        if (a < h->n && mc_better(&h->e[a], &h->e[best])) best = a;
        if (b < h->n && mc_better(&h->e[b], &h->e[best])) best = b;
        if (best == i) break;
        mcand tmp = h->e[best]; h->e[best] = h->e[i]; h->e[i] = tmp;
        i = best;
    }
    return true;
}

// A popped candidate is stale when either side has since been merged into
// something longer. Lengths grow strictly and indices never move, so the pair
// plus the two lengths it was pushed with identifies it uniquely — which is
// what lets superseded entries stay in the queue instead of being hunted down.
//
// That holds only because an ABSORBED symbol has its length zeroed. Absorbing
// updates the surviving left symbol and leaves the right one untouched, so a
// symbol that was itself swallowed as somebody's right-hand side would keep
// both its old length and its old `next` — and a candidate naming it would
// still pass this test and then merge a symbol no longer in the list. The
// rescanning loops this replaced could not hit that, because they only ever
// looked at symbols they could reach. Caught by the differential harness on
// EuroLLM, where "  index." tokenized as three ids instead of two.
#define MC_LIVE_SYM(c, SYM) \
    ((SYM)[(c).l].next == (c).r && (SYM)[(c).l].len == (c).len_l && \
     (SYM)[(c).r].len == (c).len_r)
#define MC_LIVE(c, LN, NEXT) \
    ((NEXT)[(c).l] == (c).r && (LN)[(c).l] == (c).len_l && (LN)[(c).r] == (c).len_r)

// The score floor the old scan started from. A piece that is no merge's result
// carries -FLT_MAX (see spm_scores_from_merges) and must never be merged;
// the old loop excluded it by initialising best_score to -1e30f, so the same
// constant is the admission test here.
#define SPM_SCORE_FLOOR (-1e30f)

static bool spm_push_pair(tokenizer *t, const char *text, const sym_t *sym,
                          mheap *h, int l) {
    if (l < 0) return true;
    int r = sym[l].next;
    if (r < 0) return true;
    int id = hmap_get(&t->vocab, text + sym[l].start, sym[l].len + sym[r].len);
    if (id < 0 || !(t->scores[id] > SPM_SCORE_FLOOR)) return true;
    mcand c = { -t->scores[id], l, r, sym[l].len, sym[r].len };
    return mh_push(h, c);
}

static int spm_encode(tokenizer *t, const char *text, size_t n,
                      int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    if (n > SIZE_MAX / sizeof(sym_t) - 1) { tls_encode_oom = true; return n_out; }
    sym_t *sym = malloc(sizeof(sym_t) * (n + 1));
    if (!sym) { tls_encode_oom = true; return n_out; }  // signalled via t; tok_encode returns -1
    int n_sym = 0;
    for (size_t i = 0; i < n; ) {
        int l = u8_len((uint8_t)text[i]);
        if ((size_t)(i + l) > n) l = 1;
        sym[n_sym] = (sym_t){ (int)i, l, n_sym - 1, n_sym + 1 };
        n_sym++;
        i += l;
    }
    if (n_sym > 0) sym[n_sym - 1].next = -1;

    // With no scores the old scan could never select a pair, so nothing merges.
    if (t->scores && !use_merge_queue(n_sym)) {
        for (;;) {
            float best_score = SPM_SCORE_FLOOR;
            int best = -1;
            for (int i = 0; i != -1 && sym[i].next != -1; i = sym[i].next) {
                int j = sym[i].next;
                int id = hmap_get(&t->vocab, text + sym[i].start,
                                  sym[i].len + sym[j].len);
                if (id >= 0 && t->scores[id] > best_score) {
                    best_score = t->scores[id];
                    best = i;
                }
            }
            if (best < 0) break;
            int j = sym[best].next;
            sym[best].len += sym[j].len;
            sym[best].next = sym[j].next;
            if (sym[j].next != -1) sym[sym[j].next].prev = best;
        }
    } else if (t->scores) {
        mheap h;
        mh_init(&h);
        bool oom = false;
        for (int i = 0; i != -1 && sym[i].next != -1; i = sym[i].next)
            if (!spm_push_pair(t, text, sym, &h, i)) { oom = true; break; }
        mcand c;
        while (!oom && mh_pop(&h, &c)) {
            if (!MC_LIVE_SYM(c, sym)) continue;   // superseded by an earlier merge
            int j = sym[c.l].next;
            sym[c.l].len += sym[j].len;
            sym[c.l].next = sym[j].next;
            if (sym[j].next != -1) sym[sym[j].next].prev = c.l;
            sym[j].len = 0;               // absorbed: fails MC_LIVE from here on
            // only the merged symbol's two neighbours are new pairs
            if (!spm_push_pair(t, text, sym, &h, sym[c.l].prev) ||
                !spm_push_pair(t, text, sym, &h, c.l)) { oom = true; break; }
        }
        mh_free(&h);
        if (oom) { free(sym); tls_encode_oom = true; return n_out; }
    }

    for (int i = 0; i != -1; i = sym[i].next) {
        int id = hmap_get(&t->vocab, text + sym[i].start, sym[i].len);
        if (id >= 0) {
            if (n_out < cap) out[n_out++] = id;
        } else {
            // byte fallback
            for (int b = 0; b < sym[i].len; b++) {
                char name[8];
                snprintf(name, sizeof(name), "<0x%02X>", (uint8_t)text[sym[i].start + b]);
                int bid = hmap_get(&t->vocab, name, 6);
                if (bid < 0) bid = t->unk_id;
                if (bid >= 0 && n_out < cap) out[n_out++] = bid;
            }
        }
    }
    free(sym);
    return n_out;
}

static int spm_encode_text(tokenizer *t, const char *text, size_t n,
                           int32_t *out, int cap, int n_out, bool first_segment) {
    // replace ' ' with U+2581, optionally prefix a space
    if (n > (SIZE_MAX - 4) / 3) { tls_encode_oom = true; return n_out; }
    char *buf = malloc(n * 3 + 4);
    if (!buf) { tls_encode_oom = true; return n_out; }
    size_t m = 0;
    if (t->add_space_prefix && first_segment && n > 0) {
        memcpy(buf + m, "\xE2\x96\x81", 3); m += 3;
    }
    for (size_t i = 0; i < n; i++) {
        if (text[i] == ' ') { memcpy(buf + m, "\xE2\x96\x81", 3); m += 3; }
        else buf[m++] = text[i];
    }
    n_out = spm_encode(t, buf, m, out, cap, n_out);
    free(buf);
    return n_out;
}

// ---------------------------------------------------------------- BPE encode

// Latin-1 Supplement and Superscripts/Subscripts/Currency, class by class from
// the Unicode Character Database (16.0): L letter, N number (\p{N}, so the
// superscript digits and the vulgar fractions are numbers to every regex here),
// Z whitespace, o anything else (controls, punctuation, symbols, unassigned).
// Combining marks for symbols (U+20D0..U+20F0) are cp_mark's, and read o here.
// These two blocks sit right beside letters in ordinary text: as letters, "°)"
// was two pre-tokens where every \p{L} regex keeps it one (the reference emits
// Kvist-14B's single "°)" token), and "€5", "x²" and "«word»" moved the same
// way. tests/test_tokenizer_classes.c checks every code point of both blocks.
static const char LATIN1_CLASS[129] =     // U+0080 .. U+00FF
    "oooooZoooooooooooooooooooooooooo"    // 0080 C1 controls; 0085 NEL is White_Space
    "ZoooooooooLoooooooNNoLoooNLoNNNo"    // 00A0 NBSP, ¡..¿ (ª µ º letters, ² ³ ¹ ¼ ½ ¾ numbers)
    "LLLLLLLLLLLLLLLLLLLLLLLoLLLLLLLL"    // 00C0 À..ß, × is a symbol
    "LLLLLLLLLLLLLLLLLLLLLLLoLLLLLLLL";   // 00E0 à..ÿ, ÷ is a symbol
static const char SUPER_CURRENCY_CLASS[145] =   // U+2070 .. U+20FF
    "NLooNNNNNNoooooLNNNNNNNNNNooooooLLLLLLLLLLLLLooo"    // 2070 ⁰ ⁱ ⁴..⁹ ⁺..⁾ ⁿ, ₀..₉ ₊..₎, ₐ..ₜ
    "oooooooooooooooooooooooooooooooooooooooooooooooo"    // 20A0 currency symbols (€ ₹ ...)
    "oooooooooooooooooooooooooooooooooooooooooooooooo";   // 20D0 combining marks for symbols (cp_mark)

static int cp_class_table(char k) {
    return k == 'L' ? 0 : k == 'N' ? 1 : k == 'Z' ? 3 : 2;
}

static int cp_class(uint32_t c) {
    // 0 = letter, 1 = digit (\p{N}), 2 = other(punct/symbol), 3 = whitespace
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C ||
        (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
        c == 0x202F || c == 0x205F || c == 0x3000)
        return 3;
    if (c >= '0' && c <= '9') return 1;
    if (c >= 0x80 && c <= 0xFF) return cp_class_table(LATIN1_CLASS[c - 0x80]);
    if (c >= 0x2070 && c <= 0x20FF) return cp_class_table(SUPER_CURRENCY_CLASS[c - 0x2070]);
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) return 0;
    return 2;
}

// Codepoints >= 0x80 outside the two tabulated blocks are treated as letters,
// which is right for scripts but wrong for symbols. These ranges are excluded
// to match \p{L} on real text: emoji and dingbats otherwise glue onto adjacent
// words and shift every pre-token boundary after them, and U+FFFD, which a
// lossy decode leaves in ordinary text, split the reference's `"\uFFFD.`
// punctuation run so that `.txt` after it became one token instead of two.
static bool cp_symbol(uint32_t c) {
    return (c >= 0x2000 && c <= 0x206F) || (c >= 0x2100 && c <= 0x2BFF) ||
           (c >= 0x2E00 && c <= 0x2E7F) || (c >= 0x3000 && c <= 0x303F) ||
           (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFFF0 && c <= 0xFFFF) ||
           (c >= 0x1F000 && c <= 0x1FAFF);   // FFF0: Specials, U+FFFD among them
}

// Combining marks are Unicode category M, not L. Treating every non-symbol
// codepoint above ASCII as a letter incorrectly glued Indic/Thai vowel signs
// and viramas into \p{L}+ runs. These are the mark blocks exercised by the
// supported tokenizer corpus; expand alongside differential fixtures when a
// new script is admitted.
static bool cp_mark(uint32_t c) {
    return (c >= 0x0300 && c <= 0x036F) || // Combining Diacritical Marks
           (c >= 0x0900 && c <= 0x0903) || // Devanagari signs
           (c >= 0x093A && c <= 0x094F) ||
           (c >= 0x0951 && c <= 0x0957) ||
           (c >= 0x0962 && c <= 0x0963) ||
           c == 0x0E31 ||                  // Thai combining vowels/tones
           (c >= 0x0E34 && c <= 0x0E3A) ||
           (c >= 0x0E47 && c <= 0x0E4E) ||
           (c >= 0x20D0 && c <= 0x20F0);   // Combining Diacritical Marks for Symbols
}

static bool cp_letter(uint32_t c) {
    if (cp_class(c) != 0) return false;
    return c < 0x80 || (!cp_symbol(c) && !cp_mark(c));
}

// \p{N}: the ASCII digits and the tabulated numbers (superscripts, fractions)
static bool cp_digit(uint32_t c)  { return cp_class(c) == 1; }
static bool cp_space(uint32_t c)  { return cp_class(c) == 3; }
// \p{L} and \p{N} both excluded, i.e. the regex's [^\s\p{L}\p{N}] class
static bool cp_other(uint32_t c)  { return !cp_space(c) && !cp_letter(c) && !cp_digit(c); }

// case-insensitive 's 't 're 've 'm 'll 'd, as (?i:...) in the newer regexes
static int contraction_len(const uint32_t *cp, int i, int ncp) {
    if (i < 0 || i >= ncp || cp[i] != '\'' || i + 1 >= ncp) return 0;
    uint32_t a = cp[i + 1] | 32, b = (i + 2 < ncp) ? (cp[i + 2] | 32) : 0;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    return 0;
}

// One pre-token of the newer BPE regex, returning the end index:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,max_digits}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// marks_join: the letter class is [\p{L}\p{M}] (qwen35's regex) instead of
// \p{L}, so a combining mark extends the run it follows and never starts a
// symbol run; the leading-character clause is unchanged (a mark may still
// lead, being [^\r\n\p{L}\p{N}]).
// digit_ends: the digit groups were split off BEFORE this regex ran (MiniCPM5
// applies \p{N}{1,3} as its own first pass), so a digit is the end of the
// piece this regex sees and `\s+(?!\S)` keeps a whitespace run that ends at
// one entire instead of leaving its last space to the digit.
static int pre_split_next(const uint32_t *cp, int i, int ncp, int max_digits,
                          bool marks_join, bool digit_ends) {
    int adv = contraction_len(cp, i, ncp);
    if (adv) return i + adv;
#define RUN_LETTER(c) (cp_letter(c) || (marks_join && cp_mark(c)))
#define RUN_OTHER(c)  (!cp_space(c) && !RUN_LETTER(c) && !cp_digit(c))

    // a single non-letter, non-digit, non-newline character may lead a letter run
    if (!cp_letter(cp[i]) && !cp_digit(cp[i]) && cp[i] != '\r' && cp[i] != '\n' &&
        i + 1 < ncp && RUN_LETTER(cp[i + 1])) {
        int j = i + 1;
        while (j < ncp && RUN_LETTER(cp[j])) j++;
        return j;
    }
    if (RUN_LETTER(cp[i])) {
        int j = i;
        while (j < ncp && RUN_LETTER(cp[j])) j++;
        return j;
    }
    if (cp_digit(cp[i])) {
        int j = i;
        while (j < ncp && cp_digit(cp[j]) && j - i < max_digits) j++;
        return j;
    }
    {   // optional leading space, then a run of symbols/punctuation
        int j = (cp[i] == ' ' && i + 1 < ncp && RUN_OTHER(cp[i + 1])) ? i + 1 : i;
        if (j < ncp && RUN_OTHER(cp[j])) {
            while (j < ncp && RUN_OTHER(cp[j])) j++;
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n')) j++;
            return j;
        }
    }
#undef RUN_LETTER
#undef RUN_OTHER
    {
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        // \s*[\r\n]+ runs through the last newline of the whitespace run, so
        // "\n\n" stays one pre-token; trailing spaces after it split off
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        // \s+(?!\S) keeps a trailing space for the next pre-token, unless
        // that pre-token is a digit group already split off (digit_ends)
        if (j < ncp && j - i > 1 && !(digit_ends && cp_digit(cp[j]))) j--;
        return j;
    }
}

// One pre-token of the original GPT-2 regex, bounded by end so a caller can
// restrict it to a segment:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// The contractions are case-sensitive here, unlike the newer (?i:...) regexes.
// The GPT-2 regex's classes: \p{L}, \p{N}, [^\s\p{L}\p{N}] and \s. The
// same cp_letter the llama3 rule uses, so symbols (U+2019 RIGHT SINGLE
// QUOTATION MARK in "don’t") and combining marks (Devanagari vowel signs and
// viramas) are [^\s\p{L}\p{N}], not letters: the raw cp_class glued them onto
// the adjacent letter run and granite 4.2 (pre "granite-docling", llama.cpp's
// plain GPT-2 rule) tokenized "’s" and "नमस्ते" differently from the reference.
static int gpt2_class(uint32_t c) {
    if (cp_space(c)) return 3;
    if (cp_digit(c)) return 1;
    if (cp_letter(c)) return 0;
    return 2;
}

static int gpt2_split_next(const uint32_t *cp, int i, int end) {
    if (cp[i] == '\'' && i + 1 < end) {
        uint32_t a = cp[i + 1], b = (i + 2 < end) ? cp[i + 2] : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
            (a == 'l' && b == 'l')) return i + 3;
    }
    // an optional leading space joins a run of a single class
    int j = (cp[i] == ' ' && i + 1 < end && gpt2_class(cp[i + 1]) != 3) ? i + 1 : i;
    if (gpt2_class(cp[j]) != 3) {
        int cls = gpt2_class(cp[j]);
        int k = j;
        while (k < end && gpt2_class(cp[k]) == cls) k++;
        return k;
    }
    // \s+(?!\S) hands the final whitespace character to the next pre-token,
    // whatever that character is: "\t\tx" is "\t", "\t", "x", not "\t\t", "x"
    int k = i;
    while (k < end && gpt2_class(cp[k]) == 3) k++;
    if (k < end && k - i > 1) k--;
    return k;
}

// ---------------------------------------------------------------- tekken
//
// Cased-letter classification. The tekken regex is the only split rule here
// that distinguishes upper from lower case, so this is the only place a case
// table is needed. Everything outside these ranges is treated as caseless
// (\p{Lo}/\p{Lm}/\p{M}), which matches BOTH halves of the regex's letter
// classes and so behaves exactly like the plain \p{L}+ the other families use.
// That is the correct default: CJK, Thai, Arabic, Hebrew, Devanagari and the
// rest of the caseless scripts are the majority of what >= 0x80 contains.
// Latin Extended-A is laid out in capital/small pairs, but the parity of those
// pairs flips twice inside the block, so a plain (c & 1) test is wrong for
// exactly the ranges Polish, Czech and Slovak live in -- it reads "ź" (U+017A)
// as a capital and splits "łódź" into łód|ź. Returns 1 upper, -1 lower, 0 for
// the caseless/unpaired characters.
static int latin_ext_a_case(uint32_t c) {
    if (c == 0x138 || c == 0x149 || c == 0x17F) return -1; // ĸ, ŉ, ſ: no capital
    if (c == 0x178) return 1;                              // Ÿ, paired down at 0x00FF
    if (c <= 0x137) return (c & 1) == 0 ? 1 : -1;
    if (c <= 0x148) return (c & 1) == 1 ? 1 : -1;          // parity flips at Ĺ
    if (c <= 0x177) return (c & 1) == 0 ? 1 : -1;          // and back at Ŋ
    return (c & 1) == 1 ? 1 : -1;                          // and again at Ź
}

static bool cp_upper(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return true;
    if (c < 0x80) return false;
    if (c >= 0xC0 && c <= 0xDE) return c != 0xD7;          // Latin-1, minus MULTIPLICATION SIGN
    if (c >= 0x100 && c <= 0x17F) return latin_ext_a_case(c) > 0;
    if (c >= 0x1E00 && c <= 0x1E95) return (c & 1) == 0;   // Latin Extended Additional
    if (c >= 0x1EA0 && c <= 0x1EFF) return (c & 1) == 0;   // (0x1E96..0x1E9F have no capital)
    if (c >= 0x386 && c <= 0x3AB) return true;             // Greek capitals
    if (c >= 0x400 && c <= 0x42F) return true;             // Cyrillic capitals
    return false;
}

static bool cp_lower(uint32_t c) {
    if (c >= 'a' && c <= 'z') return true;
    if (c < 0x80) return false;
    if (c == 0xB5) return true;                            // MICRO SIGN is Ll
    if (c >= 0xDF && c <= 0xFF) return c != 0xF7;          // Latin-1, minus DIVISION SIGN
    if (c >= 0x100 && c <= 0x17F) return latin_ext_a_case(c) < 0;
    if (c >= 0x1E00 && c <= 0x1E9F) return (c & 1) == 1 || c >= 0x1E96;
    if (c >= 0x1EA0 && c <= 0x1EFF) return (c & 1) == 1;
    if (c >= 0x3AC && c <= 0x3CE) return true;             // Greek smalls
    if (c >= 0x430 && c <= 0x45F) return true;             // Cyrillic smalls
    return false;
}

// [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] -- a letter that is not lowercase
// Used only by tekken, and \p{M} is the reason they exist separately. The
// comments here have always spelled the class with \p{M} in it, but cp_letter
// deliberately excludes combining marks -- correct for llama3, qwen2 and
// smollm, whose regexes all say a plain \p{L}+, and wrong for tekken, which
// carries \p{M} in both its letter classes:
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//
// so a Devanagari virama or a Thai vowel sign must stay INSIDE the run rather
// than ending it. Excluding them split three of the 721 differential strings
// away from the reference: नमस्ते, हिन्दी and สวัสดี.
static bool cp_letter_upperish(uint32_t c) {
    return (cp_letter(c) || cp_mark(c)) && !cp_lower(c);
}
// [\p{Ll}\p{Lm}\p{Lo}\p{M}] -- a letter that is not uppercase
static bool cp_letter_lowerish(uint32_t c) {
    return (cp_letter(c) || cp_mark(c)) && !cp_upper(c);
}

// The two letter alternatives of the tekken regex, tried in order at i:
//   [^\r\n\p{L}\p{N}]? [upperish]* [lowerish]+
//   [^\r\n\p{L}\p{N}]? [upperish]+ [lowerish]*
// Returns the end index, or i if neither matches.
//
// This is what makes tekken split on case: "camelCase" is camel|Case and
// "XMLHttpRequest" is XMLHttp|Request, where every other family in the lineup
// takes the whole run as one pre-token. Caseless scripts are unaffected,
// because a caseless letter satisfies both classes.
static int tekken_letters(const uint32_t *cp, int i, int ncp) {
    // the optional leading character: anything that is not CR/LF, letter or digit
    int s = i;
    if (!cp_letter(cp[i]) && !cp_digit(cp[i]) && cp[i] != '\r' && cp[i] != '\n')
        s = i + 1;
    // a mark can open the run under tekken's classes, so the fast bail has to
    // admit one too or the loops below never get to see it
    if (s >= ncp || !(cp_letter(cp[s]) || cp_mark(cp[s]))) return i;

    // upperish* is greedy, but must leave at least one lowerish for alt 1.
    // Backtracking is unnecessary: a lowerish that is also upperish (caseless)
    // is consumed by the greedy run and still satisfies the tail, and a
    // strictly-lowercase character stops the run on its own.
    int u = s;
    while (u < ncp && cp_letter_upperish(cp[u])) u++;

    int j = u;
    while (j < ncp && cp_letter_lowerish(cp[j])) j++;

    // alt 1 requires lowerish+; alt 2 requires upperish+. If the greedy
    // upperish run swallowed everything (all-caps, or caseless script), that is
    // alt 2 with an empty lowerish tail.
    if (j > u || u > s) return j;
    return i;
}

// One pre-token of the tekken regex (Mistral's tokenizer v3/v7 and Apertus):
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//   | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// Three things separate it from the llama3/qwen2 regex handled by
// pre_split_next: no contraction alternative (so "it's" is it|'s only because
// ' leads the letter run, and "it'S" is it|'S rather than it|'|S), single
// digits, and '/' joins the trailing run of a punctuation pre-token so that
// "//" in a comment or a URL path does not split away from its newline.
static int tekken_split_next(const uint32_t *cp, int i, int ncp) {
    int adv = tekken_letters(cp, i, ncp);
    if (adv > i) return adv;

    if (cp_digit(cp[i])) return i + 1;              // \p{N}, one digit at a time

    {   // optional leading space, then a run of symbols/punctuation
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_other(cp[i + 1])) ? i + 1 : i;
        if (j < ncp && cp_other(cp[j])) {
            while (j < ncp && cp_other(cp[j])) j++;
            // [\r\n/]* -- '/' here, not in the llama3/qwen2 form
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n' || cp[j] == '/')) j++;
            return j;
        }
    }
    {   // \s*[\r\n]+ | \s+(?!\S) | \s+ -- identical to the other newer regexes
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        if (j < ncp && j - i > 1) j--;
        return j;
    }
}

// One pre-token of the o200k-family regex (llama.cpp maps pre "llama4" onto
// PRE_TYPE_GPT4O; Muse Glimmer ships this):
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   | \p{N}{1,3} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// The letter alternatives are exactly tekken's (case-transition splitting),
// with the contraction attached as a SUFFIX of the letter run — not a leading
// alternative as in llama3, so "it's" stays one pre-token here where llama3
// gives it|'s. Digits run to three like llama3, and '/' joins the punctuation
// tail like tekken.
static int llama4_split_next(const uint32_t *cp, int i, int ncp) {
    int adv = tekken_letters(cp, i, ncp);
    if (adv > i) {
        int c = contraction_len(cp, adv, ncp);   // (?i:...) — already caseless
        return adv + c;
    }

    if (cp_digit(cp[i])) {
        int j = i;
        while (j < ncp && cp_digit(cp[j]) && j - i < 3) j++;
        return j;
    }

    {   // optional leading space, then a run of symbols/punctuation
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_other(cp[i + 1])) ? i + 1 : i;
        if (j < ncp && cp_other(cp[j])) {
            while (j < ncp && cp_other(cp[j])) j++;
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n' || cp[j] == '/')) j++;
            return j;
        }
    }
    {   // \s*[\r\n]+ | \s+(?!\S) | \s+
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        if (j < ncp && j - i > 1) j--;
        return j;
    }
}

// afmoe (Arcee Trinity) composes three passes (llama.cpp
// unicode_regex_split_custom_afmoe + llama-vocab.cpp pre "afmoe"), folded
// into one scanner because earlier passes only ever produce fragments the
// main regex cannot re-merge:
//   1. every maximal digit run splits into groups of three FROM THE RIGHT
//      (1234567 -> 1|234|567): first group len%3 (if nonzero), then threes;
//   2. CJK/Asian-script runs are isolated, so a leading space or Latin
//      letter can never glue onto them;
//   3. main regex:  [ascii-punct][A-Za-z]+
//                 | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//                 |  ?[\p{P}\p{S}]+[\r\n]*
//                 | \s*[\r\n]+ | \s+(?!\S) | \s+
// \p{P}\p{S} is approximated as cp_other minus marks, matching how this
// file approximates the classes everywhere else; the differential fixture
// set is the gate for the approximation.
static bool cp_afmoe_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FFF) ||   // CJK Unified
           (c >= 0x3400 && c <= 0x4DBF) ||   // CJK Ext A
           (c >= 0xF900 && c <= 0xFAFF) ||   // CJK Compat Ideographs
           (c >= 0x3040 && c <= 0x309F) ||   // Hiragana
           (c >= 0x30A0 && c <= 0x30FF) ||   // Katakana
           (c >= 0xFF65 && c <= 0xFF9F) ||   // Halfwidth Katakana
           (c >= 0x2F00 && c <= 0x2FDF) ||   // Kangxi Radicals
           (c >= 0x0E40 && c <= 0x0E7F) ||   // Thai (trailing block)
           (c >= 0x0E80 && c <= 0x0EFF) ||   // Lao
           (c >= 0x1780 && c <= 0x17FF) ||   // Khmer
           (c >= 0x1000 && c <= 0x109F) ||   // Myanmar
           (c >= 0xAA60 && c <= 0xAA7F) ||   // Myanmar Ext A
           (c >= 0xA9E0 && c <= 0xA9FF) ||   // Myanmar Ext B
           (c >= 0xAC00 && c <= 0xD7AF) ||   // Hangul Syllables
           (c >= 0x1100 && c <= 0x11FF);     // Hangul Jamo
}
static bool cp_ascii_punct(uint32_t c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}
static bool cp_punct_sym(uint32_t c) {       // ~ [\p{P}\p{S}]
    return cp_other(c) && !cp_mark(c);
}
static int afmoe_split_next(const uint32_t *cp, int i, int ncp) {
    if (cp_digit(cp[i])) {
        int L = 0;
        while (i + L < ncp && cp_digit(cp[i + L])) L++;
        int first = L % 3;
        return i + (first ? first : 3);
    }
    if (cp_afmoe_cjk(cp[i])) {
        int j = i;
        while (j < ncp && cp_afmoe_cjk(cp[j])) j++;
        return j;
    }
    // [ascii-punct][A-Za-z]+  (leads the alternation, so "'s" stays whole)
    if (cp_ascii_punct(cp[i]) && i + 1 < ncp &&
        ((cp[i + 1] >= 'a' && cp[i + 1] <= 'z') ||
         (cp[i + 1] >= 'A' && cp[i + 1] <= 'Z'))) {
        int j = i + 1;
        while (j < ncp && ((cp[j] >= 'a' && cp[j] <= 'z') ||
                           (cp[j] >= 'A' && cp[j] <= 'Z'))) j++;
        return j;
    }
    // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ — one optional "other" char
    // (space, mark; digits cannot reach here) then a letter/mark run that
    // stops at CJK, which pass 2 has already made its own fragment
    {
        int s = i;
        uint32_t c = cp[i];
        if (c != '\r' && c != '\n' && !cp_letter(c) && !cp_punct_sym(c) &&
            i + 1 < ncp)
            s = i + 1;
        if (s < ncp && (cp_letter(cp[s]) || cp_mark(cp[s])) &&
            !cp_afmoe_cjk(cp[s])) {
            int j = s;
            while (j < ncp && (cp_letter(cp[j]) || cp_mark(cp[j])) &&
                   !cp_afmoe_cjk(cp[j])) j++;
            if (j > s) return j;
        }
    }
    //  ?[\p{P}\p{S}]+[\r\n]*
    {
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_punct_sym(cp[i + 1]))
                ? i + 1 : i;
        if (j < ncp && cp_punct_sym(cp[j])) {
            while (j < ncp && cp_punct_sym(cp[j]) && !cp_afmoe_cjk(cp[j])) j++;
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n')) j++;
            return j;
        }
    }
    // \s*[\r\n]+ | \s+(?!\S) | \s+
    {
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        // the digit and CJK passes run BEFORE the main regex in the
        // reference, so a following digit/CJK char is a fragment boundary
        // there: \s+(?!\S) sees end-of-fragment and keeps every space
        if (j < ncp && j - i > 1 &&
            !cp_digit(cp[j]) && !cp_afmoe_cjk(cp[j])) j--;
        return j;
    }
}

// spark2_5, the publisher's tokenizer.json pre_tokenizer in order:
//   Split \p{N}{1,3}; Split [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+;
//   Split [ascii-punct][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//         |  ?[\p{P}\p{S}]+ | [\r\n] | \s+(?!\S) | \s+;
//   Digits(individual_digits).
// afmoe's main regex except two clauses: a punctuation run takes no trailing
// newlines, and a newline is one character on its own (no \s*[\r\n]+). The
// two digit passes leave every digit alone, and each digit and CJK run is a
// fragment boundary for the whitespace lookahead, as in afmoe.
static bool cp_spark25_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FA5) ||   // CJK Unified, to U+9FA5
           (c >= 0x3040 && c <= 0x309F) ||   // Hiragana
           (c >= 0x30A0 && c <= 0x30FF);     // Katakana
}
static int spark25_split_next(const uint32_t *cp, int i, int ncp) {
    if (cp_digit(cp[i])) return i + 1;
    if (cp_spark25_cjk(cp[i])) {
        int j = i;
        while (j < ncp && cp_spark25_cjk(cp[j])) j++;
        return j;
    }
    if (cp_ascii_punct(cp[i]) && i + 1 < ncp &&
        ((cp[i + 1] >= 'a' && cp[i + 1] <= 'z') ||
         (cp[i + 1] >= 'A' && cp[i + 1] <= 'Z'))) {
        int j = i + 1;
        while (j < ncp && ((cp[j] >= 'a' && cp[j] <= 'z') ||
                           (cp[j] >= 'A' && cp[j] <= 'Z'))) j++;
        return j;
    }
    {   // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+, stopping at a CJK fragment
        int s = i;
        uint32_t c = cp[i];
        if (c != '\r' && c != '\n' && !cp_letter(c) && !cp_punct_sym(c) &&
            i + 1 < ncp)
            s = i + 1;
        if (s < ncp && (cp_letter(cp[s]) || cp_mark(cp[s])) &&
            !cp_spark25_cjk(cp[s])) {
            int j = s;
            while (j < ncp && (cp_letter(cp[j]) || cp_mark(cp[j])) &&
                   !cp_spark25_cjk(cp[j])) j++;
            if (j > s) return j;
        }
    }
    {   //  ?[\p{P}\p{S}]+ (no trailing newlines)
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_punct_sym(cp[i + 1]))
                ? i + 1 : i;
        if (j < ncp && cp_punct_sym(cp[j])) {
            while (j < ncp && cp_punct_sym(cp[j]) && !cp_spark25_cjk(cp[j])) j++;
            return j;
        }
    }
    if (cp[i] == '\r' || cp[i] == '\n') return i + 1;   // [\r\n]
    {   // \s+(?!\S) | \s+, the lookahead ending at a fragment boundary
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        if (j == i) return i + 1;   // nothing matched: the char is its own gap
        if (j < ncp && j - i > 1 &&
            !cp_digit(cp[j]) && !cp_spark25_cjk(cp[j])) j--;
        return j;
    }
}

// k2-horizon: the Llama 3 regex whose letter run is (\p{L}|\p{M}|ZWNJ|ZWJ)+.
// The symbol run stays [^\s\p{L}\p{N}]: unlike qwen35's it does NOT exclude
// marks, so a mark after a symbol extends the symbol run.
static bool k2h_letterish(uint32_t c) {
    return cp_letter(c) || cp_mark(c) || c == 0x200C || c == 0x200D;
}
static int k2h_split_next(const uint32_t *cp, int i, int ncp) {
    int adv = contraction_len(cp, i, ncp);
    if (adv) return i + adv;
    // [^\r\n\p{L}\p{N}]? then the letter run
    if (!cp_letter(cp[i]) && !cp_digit(cp[i]) && cp[i] != '\r' && cp[i] != '\n' &&
        i + 1 < ncp && k2h_letterish(cp[i + 1])) {
        int j = i + 1;
        while (j < ncp && k2h_letterish(cp[j])) j++;
        return j;
    }
    if (k2h_letterish(cp[i])) {
        int j = i;
        while (j < ncp && k2h_letterish(cp[j])) j++;
        return j;
    }
    if (cp_digit(cp[i])) {
        int j = i;
        while (j < ncp && cp_digit(cp[j]) && j - i < 3) j++;
        return j;
    }
    {   //  ?[^\s\p{L}\p{N}]+[\r\n]*
        int j = (cp[i] == ' ' && i + 1 < ncp && cp_other(cp[i + 1])) ? i + 1 : i;
        if (j < ncp && cp_other(cp[j])) {
            while (j < ncp && cp_other(cp[j])) j++;
            while (j < ncp && (cp[j] == '\r' || cp[j] == '\n')) j++;
            return j;
        }
    }
    {   // \s*[\r\n]+ | \s+(?!\S) | \s+
        int j = i;
        while (j < ncp && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        if (j < ncp && j - i > 1) j--;
        return j > i ? j : i + 1;
    }
}

// smollm runs a Digits(individual_digits) pass before the GPT-2 regex, so every
// digit stands alone and never takes a leading space. Splitting first also
// bounds the regex: in "  12" the space run ends at the digit and stays whole,
// where "  leading" gives a space back to the word.
// flint (RecursiveSI/flint tokenizer.json, llama.cpp pre "flint"): a
// Sequence of three Isolated Splits, folded into one scanner the way afmoe's
// is, because each pass only cuts the fragments the one before it left:
//   1. \p{N}{1,3}, from the left;
//   2. [一-龥぀-ゟ゠-ヿ]+ (CJK unified ideographs to
//      U+9FA5, hiragana, katakana);
//   3. within what is left, the o200k letter alternatives (tekken_letters)
//      with a contraction suffix, then ` ?[\p{P}\p{S}]+[\r\n]*` (no '/' in
//      the tail, unlike o200k), then the three whitespace alternatives.
//      The pass has no digit alternative, and a character none of them
//      matches is a piece of its own up to the next match (Isolated keeps
//      the gaps).
static bool cp_flint_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) ||
           (c >= 0x30A0 && c <= 0x30FF);
}
// one match of pass 3 at i within [i, end), or i when nothing matches there
static int flint_pass3(const uint32_t *cp, int i, int end) {
    int adv = tekken_letters(cp, i, end);
    if (adv > i) return adv + contraction_len(cp, adv, end);
    {   // ` ?[\p{P}\p{S}]+[\r\n]*`
        int j = (cp[i] == ' ' && i + 1 < end && cp_punct_sym(cp[i + 1])) ? i + 1 : i;
        if (j < end && cp_punct_sym(cp[j])) {
            while (j < end && cp_punct_sym(cp[j])) j++;
            while (j < end && (cp[j] == '\r' || cp[j] == '\n')) j++;
            return j;
        }
    }
    if (cp_space(cp[i])) {   // \s*[\r\n]+ | \s+(?!\S) | \s+
        int j = i;
        while (j < end && cp_space(cp[j])) j++;
        int last_nl = -1;
        for (int k = i; k < j; k++)
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
        if (last_nl >= 0) return last_nl + 1;
        if (j < end && j - i > 1) j--;
        return j;
    }
    return i;
}
static int flint_split_next(const uint32_t *cp, int i, int ncp) {
    if (cp_digit(cp[i])) {
        int j = i;
        while (j < ncp && cp_digit(cp[j]) && j - i < 3) j++;
        return j;
    }
    if (cp_flint_cjk(cp[i])) {
        int j = i;
        while (j < ncp && cp_flint_cjk(cp[j])) j++;
        return j;
    }
    int end = i;   // the fragment passes 1 and 2 left
    while (end < ncp && !cp_digit(cp[end]) && !cp_flint_cjk(cp[end])) end++;
    int m = flint_pass3(cp, i, end);
    if (m > i) return m;
    int j = i + 1;   // an unmatched run up to the next match
    while (j < end && flint_pass3(cp, j, end) == j) j++;
    return j;
}

static int smollm_split_next(const uint32_t *cp, int i, int ncp) {
    if (cp_digit(cp[i])) return i + 1;
    int seg = i;
    while (seg < ncp && !cp_digit(cp[seg])) seg++;
    return gpt2_split_next(cp, i, seg);
}

// BPE merge within one pre-token (already byte->unicode mapped, utf8 string)
// Same candidate rule as the SentencePiece side, keyed on merge rank.
static bool bpe_push_pair(tokenizer *t, const char *w, const int *st,
                          const int *ln, const int *next, mheap *h, int l) {
    if (l < 0) return true;
    int r = next[l];
    if (r < 0) return true;
    int kl = ln[l] + 1 + ln[r];
    if (kl >= 512) return true;      // the old scan skipped over-long keys too
    char key[512];
    memcpy(key, w + st[l], (size_t)ln[l]);
    key[ln[l]] = ' ';
    memcpy(key + ln[l] + 1, w + st[r], (size_t)ln[r]);
    int rank = hmap_get(&t->merges, key, (size_t)kl);
    if (rank < 0) return true;
    mcand c = { (float)rank, l, r, ln[l], ln[r] };
    return mh_push(h, c);
}

static int bpe_word(tokenizer *t, const char *w, int n, int32_t *out, int cap, int n_out) {
    if (n <= 0) return n_out;
    if (t->ignore_merges) {
        // the publisher's model vocabulary only: a control or added token's
        // text is matched before the split, never as a pre-token
        int id = hmap_get(&t->vocab, w, (size_t)n);
        if (id >= 0 && (!t->ttype || t->ttype[id] == TT_NORMAL)) {
            if (n_out < cap) out[n_out++] = id;
            return n_out;
        }
    }
    size_t max_sym = (size_t)n + 1;
    // one block, four slices: the linked list needs two arrays the compacting
    // version did not, and four separate mallocs per pre-token showed up
    int *mem = malloc(sizeof(int) * max_sym * 4);
    if (!mem) { tls_encode_oom = true; return n_out; }
    int *st = mem, *ln = mem + max_sym, *next = mem + 2 * max_sym,
        *prev = mem + 3 * max_sym;
    int ns = 0;
    for (int i = 0; i < n; ) {
        int l = u8_len((uint8_t)w[i]);
        if (i + l > n) l = 1;
        st[ns] = i; ln[ns] = l;
        prev[ns] = ns - 1; next[ns] = ns + 1;
        ns++;
        i += l;
    }
    if (ns > 0) next[ns - 1] = -1;
    // A linked list rather than a compacted array: the queue holds indices, so
    // they have to stay put. Order along the list is still left to right, which
    // is what makes "smaller index" the same tie-break the old scan had.
    if (!use_merge_queue(ns)) {
        char key[512];
        while (ns > 1) {
            int best_rank = INT_MAX, best = -1;
            for (int i = 0; i != -1 && next[i] != -1; i = next[i]) {
                int j = next[i];
                int kl = ln[i] + 1 + ln[j];
                if (kl >= (int)sizeof(key)) continue;
                memcpy(key, w + st[i], (size_t)ln[i]);
                key[ln[i]] = ' ';
                memcpy(key + ln[i] + 1, w + st[j], (size_t)ln[j]);
                int r = hmap_get(&t->merges, key, (size_t)kl);
                if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
            }
            if (best < 0) break;
            int j = next[best];
            ln[best] += ln[j];
            next[best] = next[j];
            if (next[j] != -1) prev[next[j]] = best;
            ns--;
        }
    } else {
        mheap h;
        mh_init(&h);
        bool oom = false;
        for (int i = 0; i != -1 && next[i] != -1; i = next[i])
            if (!bpe_push_pair(t, w, st, ln, next, &h, i)) { oom = true; break; }
        mcand c;
        while (!oom && mh_pop(&h, &c)) {
            if (!MC_LIVE(c, ln, next)) continue;
            int j = next[c.l];
            ln[c.l] += ln[j];
            next[c.l] = next[j];
            if (next[j] != -1) prev[next[j]] = c.l;
            ln[j] = 0;                    // absorbed: fails MC_LIVE from here on
            if (!bpe_push_pair(t, w, st, ln, next, &h, prev[c.l]) ||
                !bpe_push_pair(t, w, st, ln, next, &h, c.l)) { oom = true; break; }
        }
        mh_free(&h);
        if (oom) { free(mem); tls_encode_oom = true; return n_out; }
    }
    for (int i = 0; i != -1; i = next[i]) {
        int id = hmap_get(&t->vocab, w + st[i], ln[i]);
        if (id < 0) {
            // fall back to per-character lookup, then to the <0xNN> byte
            // pieces. Silently dropping a character loses input outright:
            // gemma4's vocabulary has no literal CR or U+00A0 piece, and both
            // must decompose to their UTF-8 bytes the way the reference does.
            // A gpt2-style BPE vocabulary carries no <0xNN> pieces and its
            // byte->unicode alphabet already covers every byte, so this path
            // stays unreachable there.
            for (int j = 0; j < ln[i]; ) {
                int l = u8_len((uint8_t)w[st[i] + j]);
                if (j + l > ln[i]) l = ln[i] - j;   /* a lead byte cut by the segment end */
                int cid = hmap_get(&t->vocab, w + st[i] + j, l);
                if (cid >= 0) {
                    if (n_out < cap) out[n_out++] = cid;
                } else {
                    for (int b = 0; b < l; b++) {
                        char name[8];
                        snprintf(name, sizeof(name), "<0x%02X>",
                                 (uint8_t)w[st[i] + j + b]);
                        int bid = hmap_get(&t->vocab, name, 6);
                        if (bid < 0) bid = t->unk_id;
                        if (bid >= 0 && n_out < cap) out[n_out++] = bid;
                    }
                }
                j += l;
            }
        } else if (n_out < cap) {
            out[n_out++] = id;
        }
    }
    free(mem);
    return n_out;
}

static int bpe_encode_split(tokenizer *t, const char *text, size_t n,
                            int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    if (n > SIZE_MAX / sizeof(size_t) - 2 || n > (SIZE_MAX - 8) / 2) return n_out;
    // decode to codepoints, remembering byte offsets
    uint32_t *cp = malloc(sizeof(uint32_t) * (n + 1));
    size_t *off = malloc(sizeof(size_t) * (n + 2));
    // byte->unicode expands ascii <0x80 to <=2 bytes
    char *word = malloc(n * 2 + 8);
    if (!cp || !off || !word) { free(cp); free(off); free(word); tls_encode_oom = true; return n_out; }
    int ncp = 0;
    for (size_t i = 0; i < n; ) {
        int l = u8_len((uint8_t)text[i]);
        if (i + l > n) l = 1;
        off[ncp] = i;
        cp[ncp++] = u8_decode(text + i, l);
        i += l;
    }
    off[ncp] = n;

    // The split rules come from tokenizer.ggml.pre; anything unrecognised keeps
    // the original GPT-2 regex it has always used.
    int max_digits = (t->pre == TOK_PRE_LLAMA3 || t->pre == TOK_PRE_MINICPM5) ? 3
                   : (t->pre == TOK_PRE_QWEN2 || t->pre == TOK_PRE_QWEN35) ? 1 : 0;
    bool marks_join = t->pre == TOK_PRE_QWEN35;
    bool digit_ends = t->pre == TOK_PRE_MINICPM5;
    for (int i = 0; i < ncp; ) {
        int end = t->pre == TOK_PRE_TEKKEN    ? tekken_split_next(cp, i, ncp)
                : t->pre == TOK_PRE_LLAMA4    ? llama4_split_next(cp, i, ncp)
                : max_digits                  ? pre_split_next(cp, i, ncp, max_digits, marks_join, digit_ends)
                : t->pre == TOK_PRE_SMOLLM    ? smollm_split_next(cp, i, ncp)
                : t->pre == TOK_PRE_AFMOE     ? afmoe_split_next(cp, i, ncp)
                : t->pre == TOK_PRE_SPARK25   ? spark25_split_next(cp, i, ncp)
                : t->pre == TOK_PRE_K2H       ? k2h_split_next(cp, i, ncp)
                : t->pre == TOK_PRE_FLINT     ? flint_split_next(cp, i, ncp)
                                              : gpt2_split_next(cp, i, ncp);
        if (end <= i) end = i + 1; // never stall
        // map the original bytes of this pre-token through byte->unicode
        size_t b0 = off[i], b1 = off[end];
        int wl = 0;
        for (size_t b = b0; b < b1; b++)
            wl += u8_encode((uint32_t)t->b2u[(uint8_t)text[b]], word + wl);
        n_out = bpe_word(t, word, wl, out, cap, n_out);
        i = end;
    }
    free(cp); free(off); free(word);
    return n_out;
}

// One run of text between special tokens: the normalizer the publisher
// declares (NFC for Qwen), then the split and the merges.
static int bpe_encode_text(tokenizer *t, const char *text, size_t n,
                           int32_t *out, int cap, int n_out) {
    char *norm = NULL;
    size_t nn = 0;
    bool oom = false;
    if (t->nfc && nfc_utf8(text, n, &norm, &nn, &oom)) {
        int r = bpe_encode_split(t, norm, nn, out, cap, n_out);
        free(norm);
        return r;
    }
    if (oom) { tls_encode_oom = true; return n_out; }
    return bpe_encode_split(t, text, n, out, cap, n_out);
}

// gemma4: SPM-style BPE — the normalizer replaces spaces with U+2581 and BPE
// merges run over raw UTF-8 with no byte-encoding; only newline runs split
// pre-tokens (merge keys never contain newlines)
static int bpe_spm_encode_text(tokenizer *t, const char *text, size_t n,
                               int32_t *out, int cap, int n_out) {
    if (n == 0) return n_out;
    if (n > (SIZE_MAX - 4) / 3) { tls_encode_oom = true; return n_out; }
    char *buf = malloc(n * 3 + 4);
    if (!buf) { tls_encode_oom = true; return n_out; }
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        if (text[i] == ' ') { memcpy(buf + m, "\xE2\x96\x81", 3); m += 3; }
        else buf[m++] = text[i];
    }
    size_t i = 0;
    while (i < m) {
        bool nl = buf[i] == '\n';
        size_t j = i;
        while (j < m && (buf[j] == '\n') == nl) j++;
        n_out = bpe_word(t, buf + i, (int)(j - i), out, cap, n_out);
        i = j;
    }
    free(buf);
    return n_out;
}

// ---------------------------------------------------------------- public api

// The encoder proper. `raw_run`, when given, holds for every byte the length
// of the template-owned run starting there (0 for caller text), and a
// special token is recognized only when it lies entirely inside such a run.
static int encode_core(tokenizer *t, const char *text, size_t n,
                       const size_t *raw_run, int32_t *out, int cap,
                       bool add_bos, bool parse_special) {
    int n_out = 0;
    tls_encode_oom = false;   // set by a helper that had to drop a segment
    if (add_bos && t->add_bos && t->bos_id >= 0 && n_out < cap)
        out[n_out++] = t->bos_id;

    size_t seg = 0;   // start of pending plain-text segment
    bool first = true;
    for (size_t i = 0; i < n; ) {
        int matched = -1;
        if (parse_special && (!raw_run || raw_run[i] > 0)) {
            int b0 = (uint8_t)text[i];
            for (int s = t->sb_off[b0]; s < t->sb_off[b0 + 1]; s++) {
                gg_str *tok = &t->tokens[t->special_ids[s]];
                if (tok->n <= n - i && memcmp(text + i, tok->s, tok->n) == 0 &&
                    (!raw_run || tok->n <= raw_run[i])) {
                    matched = t->special_ids[s];
                    break;
                }
            }
        }
        if (matched >= 0) {
            if (i > seg) {
                n_out = t->model == TOK_SPM
                    ? spm_encode_text(t, text + seg, i - seg, out, cap, n_out, first)
                    : t->model == TOK_BPE_SPM
                    ? bpe_spm_encode_text(t, text + seg, i - seg, out, cap, n_out)
                    : bpe_encode_text(t, text + seg, i - seg, out, cap, n_out);
            }
            if (n_out < cap) out[n_out++] = matched;
            i += t->tokens[matched].n;
            seg = i;
            first = false;
        } else {
            i++;
        }
    }
    if (n > seg) {
        n_out = t->model == TOK_SPM
            ? spm_encode_text(t, text + seg, n - seg, out, cap, n_out, first)
            : t->model == TOK_BPE_SPM
            ? bpe_spm_encode_text(t, text + seg, n - seg, out, cap, n_out)
            : bpe_encode_text(t, text + seg, n - seg, out, cap, n_out);
    }
    // A dropped segment would look like a legitimately shorter prompt; refuse
    // to return a silently truncated tokenization.
    if (tls_encode_oom) return -1;
    return n_out;
}

int tok_encode(tokenizer *t, const char *text, int32_t *out, int cap,
               bool add_bos, bool parse_special) {
    return encode_core(t, text, strlen(text), NULL, out, cap, add_bos, parse_special);
}

size_t tok_strip_marks(char *s) {
    size_t w = 0;
    for (size_t r = 0; s[r]; r++)
        if (s[r] != PROMPT_RAW_OPEN && s[r] != PROMPT_RAW_CLOSE) s[w++] = s[r];
    s[w] = 0;
    return w;
}

// A rendered prompt: the marks are lifted out, the text between them is
// tokenized as one contiguous piece (so every merge sees exactly the bytes
// the model would see), and a special token is recognized only inside a
// template-owned run. Text with no marks at all is entirely caller text.
int tok_encode_prompt(tokenizer *t, const char *text, int32_t *out, int cap,
                      bool add_bos) {
    size_t n = strlen(text);
    char *clean = malloc(n + 1);
    unsigned char *own = malloc(n + 1);
    size_t *run = malloc(sizeof(size_t) * (n + 1));
    if (!clean || !own || !run) {
        free(clean); free(own); free(run);
        tls_encode_oom = true;
        return -1;
    }
    size_t m = 0;
    bool raw = false;
    for (size_t i = 0; i < n; i++) {
        if (text[i] == PROMPT_RAW_OPEN) { raw = true; continue; }
        if (text[i] == PROMPT_RAW_CLOSE) { raw = false; continue; }
        own[m] = raw;
        clean[m++] = text[i];
    }
    clean[m] = 0;
    size_t acc = 0;   // run[i]: template-owned bytes from i onward, 0 for text
    for (size_t i = m; i-- > 0; ) {
        acc = own[i] ? acc + 1 : 0;
        run[i] = acc;
    }
    int r = encode_core(t, clean, m, run, out, cap, add_bos, true);
    free(clean); free(own); free(run);
    return r;
}

int tok_encode_fit(tokenizer *t, const char *text, bool add_bos, tok_mode mode,
                   int spare, int32_t **out) {
    *out = NULL;
    size_t n = strlen(text);
    if (spare < 0) spare = 0;
    if (n > (size_t)INT_MAX / 4) return -1;
    int cap = (int)(n + 16 + (size_t)spare);
    for (;;) {
        int32_t *buf = malloc(sizeof(int32_t) * (size_t)cap);
        if (!buf) return -1;
        int k = mode == TOK_PROMPT ? tok_encode_prompt(t, text, buf, cap, add_bos)
              : tok_encode(t, text, buf, cap, add_bos, mode == TOK_RAW);
        if (k < 0) { free(buf); return -1; }
        // fits with the spare slots free: the encoder did not stop short
        if (k + spare < cap) { *out = buf; return k; }
        free(buf);
        if (cap > INT_MAX / 2) return -1;
        cap *= 2;
    }
}

bool tok_is_control(tokenizer *t, int id) {
    if (id < 0 || id >= t->n_vocab) return false;
    int tt = t->ttype ? t->ttype[id] : TT_NORMAL;
    return tt == TT_CONTROL;
}

bool tok_is_special(tokenizer *t, int id) {
    if (id < 0 || id >= t->n_vocab) return false;
    int tt = t->ttype ? t->ttype[id] : TT_NORMAL;
    return tt == TT_CONTROL || tt == TT_USER_DEFINED;
}

const char *tok_raw(tokenizer *t, int id) {
    if (id < 0 || id >= t->n_vocab) return NULL;
    return t->tokens[id].s;
}

static void stop_add_id(int *out, int *n, int cap, int id) {
    if (id < 0 || *n >= cap) return;
    for (int j = 0; j < *n; j++) if (out[j] == id) return;
    out[(*n)++] = id;
}

// Admit one stop token by its vocabulary spelling. Silent on a token this
// model does not have, on a duplicate, and on a full list — each is a reason
// this spelling contributes nothing, not a reason to fail an engine. A match
// counts only when it is a control or user-defined token: Qwen3's and Gemma
// 4's vocabularies also hold "</s>" as an ordinary token (ids 128247 and
// 212), which their tokenizers never produce from text ("</s>" encodes as
// "</" "s" ">"); a model that samples one is writing text, not ending its
// turn. A file without token types cannot tell, so its matches stay stops.
static void stop_add(tokenizer *t, int *out, int *n, int cap, const char *spelling) {
    int id = tok_find(t, spelling);
    if (id < 0 || (t->ttype && !tok_is_special(t, id))) return;
    stop_add_id(out, n, cap, id);
}

int tok_stop_ids(tokenizer *t, const char *arch, int *out, int cap) {
    int n = 0;
    stop_add_id(out, &n, cap, t->eos_id);
    static const char *stops[] = { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>",
                                   "<|endoftext|>", "</s>",
                                   // gemma turn terminators (gemma1-3 / gemma4)
                                   "<end_of_turn>", "<turn|>",
                                   // gemma4 ends a turn of tool calls with
                                   // <|tool_response>: google/gemma-4-*'s
                                   // generation_config lists it (id 50) as
                                   // eos beside <turn|> (106) and <eos> (1)
                                   "<|tool_response>",
                                   // phi3 ends assistant turns with <|end|>,
                                   // which is not its declared eos_token_id
                                   "<|end|>",
                                   // muse-glimmer ends turns with <|eot|>,
                                   // also not its declared eos (<|eom|> is
                                   // NOT a stop: it separates a reasoning
                                   // turn from the answer that follows it)
                                   "<|eot|>",
                                   // Nemotron Nano closes every assistant
                                   // turn with <SPECIAL_12>; its declared
                                   // eos is a different token
                                   "<SPECIAL_12>",
                                   // K2-Horizon ends its turns with
                                   // <|ifm|im_end|>; its declared eos is
                                   // <|ifm|endoftext|> (the publisher's
                                   // generation_config lists both)
                                   "<|ifm|im_end|>" };
    _Static_assert(sizeof(stops) / sizeof(*stops) == TOK_STOP_SPELLINGS,
                   "TOK_STOP_SPELLINGS counts the spelling list");
    // gpt-oss / Harmony inverts one of these. <|end|> terminates a NON-final
    // Harmony message — the assistant writes its analysis, closes it with
    // <|end|>, then opens the final channel — so treating <|end|> as a stop
    // ends generation on the reasoning and never reaches the answer. It is in
    // the list above for phi3, whose assistant turns really do end there.
    // <|return|> is Harmony's own end-of-generation marker and takes its place;
    // <|call|> joins it so an unrendered tool attempt stops instead of running
    // away (Harmony tool calling is deliberately not rendered — see template.c).
    bool harmony = strcmp(arch, "gpt-oss") == 0;
    // The file's own end-of-generation ids, whatever their type: a declared
    // id is the publisher's statement. Harmony's <|end|> stays out here too.
    for (int i = 0; i < t->n_eog; i++) {
        const char *s = tok_raw(t, t->eog_ids[i]);
        if (harmony && s && !strcmp(s, "<|end|>")) continue;
        stop_add_id(out, &n, cap, t->eog_ids[i]);
    }
    for (size_t i = 0; i < sizeof(stops) / sizeof(*stops); i++) {
        if (harmony && !strcmp(stops[i], "<|end|>")) continue;
        stop_add(t, out, &n, cap, stops[i]);
    }
    if (harmony) {
        static const char *hstops[] = { "<|return|>", "<|call|>" };
        _Static_assert(sizeof(hstops) / sizeof(*hstops) <= TOK_STOP_ARCH_MAX,
                       "TOK_STOP_ARCH_MAX bounds every per-architecture list");
        for (size_t i = 0; i < sizeof(hstops) / sizeof(*hstops); i++)
            stop_add(t, out, &n, cap, hstops[i]);
    }
    // flint ends a turn by starting the next one: its file lists <|user|>,
    // <|observation|> and <|developer|> beside <|end_of_text|> as
    // end-of-generation tokens (tokenizer.ggml.eog_token_ids, read above;
    // llama.cpp's vocab adds the first two by name for this pre). Named here
    // too, so a flint file without that key still stops there.
    if (!strcmp(arch, "flint")) {
        static const char *fstops[] = { "<|user|>", "<|observation|>", "<|developer|>" };
        _Static_assert(sizeof(fstops) / sizeof(*fstops) <= TOK_STOP_ARCH_MAX,
                       "TOK_STOP_ARCH_MAX bounds every per-architecture list");
        for (size_t i = 0; i < sizeof(fstops) / sizeof(*fstops); i++)
            stop_add(t, out, &n, cap, fstops[i]);
    }
    return n;
}

int tok_find(tokenizer *t, const char *s) {
    return hmap_get(&t->vocab, s, strlen(s));
}

// Encode raw bytes verbatim: no BOS, no specials, and — the point — none of
// tok_encode's segment-start normalization (SPM's leading-space prefix) nor
// the NFC normalizer, so the token list round-trips to exactly the input
// bytes wherever the vocab can express them. Grammar fast-forward drafts
// through this; a vocab that cannot round-trip some byte simply yields a
// shorter (or empty) draft.
int tok_encode_raw(tokenizer *t, const char *text, int n,
                   int32_t *out, int cap) {
    tls_encode_oom = false;
    if (n <= 0) return 0;
    int k = t->model == TOK_SPM
        ? spm_encode_text(t, text, n, out, cap, 0, false)
        : t->model == TOK_BPE_SPM
        ? bpe_spm_encode_text(t, text, n, out, cap, 0)
        : bpe_encode_split(t, text, n, out, cap, 0);
    return tls_encode_oom ? -1 : k;
}

int tok_decode(tokenizer *t, int id, char *buf, int cap) {
    if (id < 0 || id >= t->n_vocab) return 0;
    int tt = t->ttype ? t->ttype[id] : TT_NORMAL;
    if (tt == TT_CONTROL || tt == TT_UNUSED) return 0;
    gg_str *tok = &t->tokens[id];

    if (t->model == TOK_SPM || t->model == TOK_BPE_SPM) {
        if (tt == TT_BYTE) {
            // "<0xXX>"
            if (tok->n == 6 && cap >= 1) {
                buf[0] = (char)strtol(tok->s + 3, NULL, 16);
                return 1;
            }
            return 0;
        }
        int m = 0;
        for (size_t i = 0; i < tok->n && m < cap - 3; ) {
            if (i + 3 <= tok->n && memcmp(tok->s + i, "\xE2\x96\x81", 3) == 0) {
                buf[m++] = ' ';
                i += 3;
            } else {
                buf[m++] = tok->s[i++];
            }
        }
        return m;
    }

    // BPE: map codepoints back to bytes
    int m = 0;
    for (size_t i = 0; i < tok->n && m < cap; ) {
        int l = u8_len((uint8_t)tok->s[i]);
        if (i + l > tok->n) l = 1;
        uint32_t c = u8_decode(tok->s + i, l);
        if (c < 512 && t->u2b[c] >= 0) buf[m++] = (char)t->u2b[c];
        else {
            // not in byte-alphabet (e.g. user-defined token text): copy as-is
            for (int j = 0; j < l && m < cap; j++) buf[m++] = tok->s[i + j];
        }
        i += l;
    }
    return m;
}
