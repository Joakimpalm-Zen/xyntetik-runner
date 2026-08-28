// Incremental JSON-prefix validator for constrained generation.
// Accepts bytes only while the output remains a valid prefix of exactly one
// top-level JSON object; sets `done` once that object closes.
#include "jsonmode.h"

#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define FALLTHROUGH __attribute__((fallthrough))
#else
#define FALLTHROUGH
#endif

enum {
    S_START,        // expecting '{' (whitespace ok)
    S_VALUE,        // expecting any value
    S_KEY_OR_END,   // after '{': '"' or '}'
    S_KEY_EXPECT,   // after ',' in object: '"'
    S_KEY,          // inside key string
    S_COLON,        // after key: ':'
    S_STRING,       // inside value string
    S_AFTER,        // value done inside container: ',' or close
    S_ARR_FIRST,    // after '[': value or ']'
    S_NUM_MINUS, S_NUM_ZERO, S_NUM_INT, S_NUM_FRAC0, S_NUM_FRAC,
    S_NUM_EXP0, S_NUM_EXP1, S_NUM_EXP,
    S_LIT,          // inside true/false/null
    S_DONE,         // top-level object complete
};

static const char *LITS[3] = { "true", "false", "null" };

// See jsonmode.h. The rejections are as early as the digits allow -- a second
// digit that makes DC..DF cannot become anything but an unpaired low
// surrogate -- because a rejection at the FOURTH digit would leave a
// constrained model with all sixteen continuations masked.
bool json_escape_hex(uint8_t *sub, uint16_t *esc, uint8_t c) {
    int d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    else return false;
    bool low = *sub >= 8;                     // the low half of a pair
    int nth = (int)(*sub - (low ? 8 : 2));    // 0..3
    uint16_t val = (uint16_t)((nth ? *esc : 0) * 16 + d);
    if (low) {
        if (nth == 0 && d != 0xD) return false;
        if (nth == 1 && (val < 0xDC || val > 0xDF)) return false;
    } else if (nth == 1 && val >= 0xDC && val <= 0xDF) {
        return false;                         // a low surrogate with no high half
    }
    *esc = val;
    if (nth < 3) { (*sub)++; return true; }
    // jv strings are NUL-terminated and carry no length, so U+0000 would make
    // every consumer see a truncated value -- json.c rejects it and so must
    // anything that generates it
    if (!low && val == 0) return false;
    *sub = (!low && val >= 0xD800 && val <= 0xDBFF) ? 6 : 0;
    return true;
}

uint32_t json_key_hash_init(void) { return 2166136261u; }   // FNV-1a basis

uint32_t json_key_hash_byte(uint32_t h, uint8_t c) {
    return (h ^ c) * 16777619u;
}

uint32_t json_key_hash_scalar(uint32_t h, uint32_t cp) {
    // hash the scalar's UTF-8 bytes, so an escaped spelling collides with
    // the raw one exactly as it does after json_parse unescapes them
    if (cp < 0x80) return json_key_hash_byte(h, (uint8_t)cp);
    if (cp < 0x800) {
        h = json_key_hash_byte(h, (uint8_t)(0xC0 | (cp >> 6)));
        return json_key_hash_byte(h, (uint8_t)(0x80 | (cp & 0x3F)));
    }
    if (cp < 0x10000) {
        h = json_key_hash_byte(h, (uint8_t)(0xE0 | (cp >> 12)));
        h = json_key_hash_byte(h, (uint8_t)(0x80 | ((cp >> 6) & 0x3F)));
        return json_key_hash_byte(h, (uint8_t)(0x80 | (cp & 0x3F)));
    }
    h = json_key_hash_byte(h, (uint8_t)(0xF0 | (cp >> 18)));
    h = json_key_hash_byte(h, (uint8_t)(0x80 | ((cp >> 12) & 0x3F)));
    h = json_key_hash_byte(h, (uint8_t)(0x80 | ((cp >> 6) & 0x3F)));
    return json_key_hash_byte(h, (uint8_t)(0x80 | (cp & 0x3F)));
}

uint8_t json_escape_decode_simple(uint8_t c) {
    switch (c) {
    case '"':  return '"';
    case '\\': return '\\';
    case '/':  return '/';
    case 'b':  return 0x08;
    case 'f':  return 0x0C;
    case 'n':  return 0x0A;
    case 'r':  return 0x0D;
    case 't':  return 0x09;
    default:   return 0;
    }
}

int json_escape_close(uint8_t *sub, uint16_t *esc, uint16_t hi,
                      char *out, int cap, uint32_t *scalar) {
    int m = 0;
    uint32_t completed = 0;
    #define ESC_PUT(ch) do { if (m < cap) out[m++] = (char)(ch); } while (0)
    if (*sub == 1) {
        ESC_PUT('n'); *sub = 0;
        if (scalar) *scalar = 0x0A;
        return m;
    }
    if (*sub >= 2 && *sub <= 5) {
        int nth = *sub - 2;                     // digits already written
        uint16_t cp = (uint16_t)((nth ? *esc : 0) << (4 * (4 - nth)));
        if (cp == 0) {                          // zeros would spell U+0000
            for (int i = nth; i < 3; i++) ESC_PUT('0');
            ESC_PUT('1');
            cp = 1;
        } else {
            for (int i = nth; i < 4; i++) ESC_PUT('0');
        }
        if (cp >= 0xD800 && cp <= 0xDBFF) { *sub = 6; hi = cp; }
        else                              { *sub = 0; completed = cp; }
    }
    if (*sub == 6) { ESC_PUT('\\'); *sub = 7; }
    if (*sub == 7) { ESC_PUT('u');  *sub = 8; }
    if (*sub >= 8 && *sub <= 11) {
        // the smallest low surrogate consistent with what is already written;
        // json_escape_hex has pinned digit 1 to D and digit 2 to C..F, so
        // zeros are safe from the third digit on
        static const char LOW[4] = { 'D', 'C', '0', '0' };
        int nth = *sub - 8;
        uint32_t lo = nth ? *esc : 0;
        for (int i = nth; i < 4; i++) {
            ESC_PUT(LOW[i]);
            lo = lo * 16 + (LOW[i] <= '9' ? (uint32_t)(LOW[i] - '0')
                                          : (uint32_t)(LOW[i] - 'A' + 10));
        }
        *sub = 0;
        completed = hi >= 0xD800
                        ? 0x10000u + (((uint32_t)hi - 0xD800u) << 10)
                                   + (lo - 0xDC00u)
                        : lo;
    }
    #undef ESC_PUT
    if (scalar) *scalar = completed;
    return m;
}

void jsonv_init(jsonv *v) {
    v->depth = 0;
    v->st = S_START;
    v->sub = 0;
    v->lit = 0;
    v->utf8 = 0;
    v->esc = 0;
    v->esc_hi = 0;
    v->khash = 0;
    v->kseen_n = 0;
    v->done = false;
}

void jsonv_init_any(jsonv *v) {
    jsonv_init(v);
    v->st = S_VALUE;
}

bool jsonv_value_end(const jsonv *v) {
    if (v->done) return true;
    return v->depth == 0 && (v->st == S_NUM_ZERO || v->st == S_NUM_INT ||
                             v->st == S_NUM_FRAC || v->st == S_NUM_EXP);
}

static bool is_ws(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static bool push(jsonv *v, uint8_t c) {
    if (v->depth >= (int)sizeof(v->stack)) return false;
    v->stack[v->depth++] = c;
    return true;
}

// a value just finished at the current depth
// a closing object releases its own keys from the duplicate guard; a sibling
// object that opens later at the same depth starts clean (schema.c's map
// guard compacts identically in frame_done)
static void drop_keys(jsonv *v, int depth) {
    if (!v->kseen_n) return;
    int w = 0;
    for (int i = 0; i < v->kseen_n; i++)
        if (v->kseen_depth[i] != (uint8_t)depth) {
            v->kseen_hash[w] = v->kseen_hash[i];
            v->kseen_depth[w] = v->kseen_depth[i];
            w++;
        }
    v->kseen_n = (uint8_t)w;
}

static void value_done(jsonv *v) {
    if (v->depth == 0) {
        v->st = S_DONE;
        v->done = true;
    } else {
        v->st = S_AFTER;
    }
}

// returns false on invalid byte; may set reconsume for number terminators
static bool feed_byte(jsonv *v, uint8_t c, bool *reconsume) {
    *reconsume = false;
    switch (v->st) {
    case S_START:
        // A constrained document must begin with its opening token. Leading
        // whitespace is refused because it is the one position where a model
        // can burn its whole budget without producing any content: with every
        // other byte illegal, spaces and newlines are the only moves left, and
        // the decode livelocks until max_tokens forces a close. Interior
        // whitespace stays legal throughout (see S_VALUE, S_AFTER, and the
        // rest) — that is ordinary pretty-printing, and by then the document
        // has real content in it. See leading_ws_ok() in schema.c.
        if (c == '{') { v->st = S_KEY_OR_END; return push(v, 'O'); }
        return false;

    case S_ARR_FIRST:
        if (c == ']' && !is_ws(c)) { v->depth--; value_done(v); return true; }
        // fall through: anything else must start a value
        FALLTHROUGH;
    case S_VALUE:
        if (is_ws(c)) return true;
        if (c == '{') { v->st = S_KEY_OR_END; return push(v, 'O'); }
        if (c == '[') { v->st = S_ARR_FIRST; return push(v, 'A'); }
        if (c == '"') { v->st = S_STRING; v->sub = 0; return true; }
        if (c == '-') { v->st = S_NUM_MINUS; return true; }
        if (c == '0') { v->st = S_NUM_ZERO; return true; }
        if (c >= '1' && c <= '9') { v->st = S_NUM_INT; return true; }
        if (c == 't') { v->st = S_LIT; v->lit = 0; v->sub = 1; return true; }
        if (c == 'f') { v->st = S_LIT; v->lit = 1; v->sub = 1; return true; }
        if (c == 'n') { v->st = S_LIT; v->lit = 2; v->sub = 1; return true; }
        return false;

    case S_KEY_OR_END:
        if (is_ws(c)) return true;
        if (c == '"') {
            v->st = S_KEY; v->sub = 0;
            v->khash = json_key_hash_init();
            return true;
        }
        if (c == '}') { drop_keys(v, v->depth); v->depth--; value_done(v); return true; }
        return false;

    case S_KEY_EXPECT:
        if (is_ws(c)) return true;
        if (c == '"') {
            v->st = S_KEY; v->sub = 0;
            v->khash = json_key_hash_init();
            return true;
        }
        return false;

    case S_KEY:
    case S_STRING: {
        bool key = v->st == S_KEY;
        if (v->sub == 1) { // after backslash
            uint8_t dec = json_escape_decode_simple(c);
            if (dec) {
                v->sub = 0;
                if (key) v->khash = json_key_hash_byte(v->khash, dec);
                return true;
            }
            if (c == 'u') { v->sub = 2; return true; }
            return false;
        }
        if (v->sub == 6) { if (c != '\\') return false; v->sub = 7; return true; }
        if (v->sub == 7) { if (c != 'u')  return false; v->sub = 8; return true; }
        if (v->sub >= 2) {
            uint8_t pre = v->sub;
            if (!json_escape_hex(&v->sub, &v->esc, c)) return false;
            if (key) {
                // hash the decoded scalar at escape completion, so an
                // escaped key spelling collides with its raw spelling
                // exactly as it does after json_parse unescapes them
                if (pre <= 5 && v->sub == 6) v->esc_hi = v->esc;
                else if (pre <= 5 && v->sub == 0)
                    v->khash = json_key_hash_scalar(v->khash, v->esc);
                else if (pre >= 8 && v->sub == 0)
                    v->khash = json_key_hash_scalar(v->khash,
                        0x10000u + (((uint32_t)v->esc_hi - 0xD800u) << 10)
                                 + ((uint32_t)v->esc - 0xDC00u));
            }
            return true;
        }
        if (v->utf8) {
            if (!json_utf8_byte(&v->utf8, c)) return false;
            if (key) v->khash = json_key_hash_byte(v->khash, c);
            return true;
        }
        if (c == '"') {
            if (key) {
                // a key this object already closed is refused AT ITS
                // CLOSING QUOTE, the same rule (and the same reason) as
                // schema.c's map guard: json_parse refuses the duplicate,
                // and the model still holds every legal continuation
                for (int i = 0; i < v->kseen_n; i++)
                    if (v->kseen_depth[i] == (uint8_t)v->depth &&
                        v->kseen_hash[i] == v->khash)
                        return false;
                if (v->kseen_n < JSON_KEY_SEEN_MAX && v->depth <= 255) {
                    v->kseen_hash[v->kseen_n] = v->khash;
                    v->kseen_depth[v->kseen_n] = (uint8_t)v->depth;
                    v->kseen_n++;
                }
                v->st = S_COLON;
            }
            else value_done(v);
            return true;
        }
        if (c == '\\') { v->sub = 1; return true; }
        // control chars forbidden, and raw bytes above ASCII must form
        // well-formed UTF-8 -- json_parse refuses ill-formed sequences, so
        // accepting them here would emit a document it cannot read back
        if (c < 0x20) return false;
        if (!json_utf8_byte(&v->utf8, c)) return false;
        if (key) v->khash = json_key_hash_byte(v->khash, c);
        return true;
    }

    case S_COLON:
        if (is_ws(c)) return true;
        if (c == ':') { v->st = S_VALUE; return true; }
        return false;

    case S_AFTER: {
        if (is_ws(c)) return true;
        uint8_t top = v->depth > 0 ? v->stack[v->depth - 1] : 0;
        if (c == ',' && top == 'O') { v->st = S_KEY_EXPECT; return true; }
        if (c == ',' && top == 'A') { v->st = S_VALUE; return true; }
        if (c == '}' && top == 'O') {
            drop_keys(v, v->depth);
            v->depth--; value_done(v); return true;
        }
        if (c == ']' && top == 'A') { v->depth--; value_done(v); return true; }
        return false;
    }

    case S_NUM_MINUS:
        if (c == '0') { v->st = S_NUM_ZERO; return true; }
        if (c >= '1' && c <= '9') { v->st = S_NUM_INT; return true; }
        return false;
    case S_NUM_ZERO:
    case S_NUM_INT:
        if (v->st == S_NUM_INT && c >= '0' && c <= '9') return true;
        if (c == '.') { v->st = S_NUM_FRAC0; return true; }
        if (c == 'e' || c == 'E') { v->st = S_NUM_EXP0; return true; }
        value_done(v); *reconsume = true; return true;
    case S_NUM_FRAC0:
        if (c >= '0' && c <= '9') { v->st = S_NUM_FRAC; return true; }
        return false;
    case S_NUM_FRAC:
        if (c >= '0' && c <= '9') return true;
        if (c == 'e' || c == 'E') { v->st = S_NUM_EXP0; return true; }
        value_done(v); *reconsume = true; return true;
    case S_NUM_EXP0:
        if (c == '+' || c == '-') { v->st = S_NUM_EXP1; return true; }
        if (c >= '0' && c <= '9') { v->st = S_NUM_EXP; return true; }
        return false;
    case S_NUM_EXP1:
        if (c >= '0' && c <= '9') { v->st = S_NUM_EXP; return true; }
        return false;
    case S_NUM_EXP:
        if (c >= '0' && c <= '9') return true;
        value_done(v); *reconsume = true; return true;

    case S_LIT: {
        const char *lit = LITS[v->lit];
        if (c == (uint8_t)lit[v->sub]) {
            v->sub++;
            if (lit[v->sub] == 0) value_done(v);
            return true;
        }
        return false;
    }

    case S_DONE:
        return false; // nothing (not even whitespace) after the object

    default:
        return false;
    }
}

// forcibly complete the JSON from the current state (used when the token
// budget runs out): close strings/escapes, complete literals and numbers
// with minimal filler, then close all open containers. Returns bytes
// written, or 0 if generation never started an object.
int jsonv_close(jsonv *v, char *out, int cap) {
    int m = 0;
    if (v->st == S_START || v->done) return 0;
    #define EMIT(c) do { if (m < cap - 1) out[m++] = (c); } while (0)
    // unfinished string escapes and truncated raw UTF-8 scalars: both must
    // finish before the quote or the document does not survive json_parse
    if (v->st == S_KEY || v->st == S_STRING) {
        bool key = v->st == S_KEY;
        char u8[3];
        int un = json_utf8_close(&v->utf8, u8);
        for (int i = 0; i < un; i++) {
            EMIT(u8[i]);
            if (key) v->khash = json_key_hash_byte(v->khash, (uint8_t)u8[i]);
        }
        char esc[12];
        uint32_t scalar = 0;
        int en = json_escape_close(&v->sub, &v->esc, v->esc_hi, esc,
                                   (int)sizeof(esc), &scalar);
        for (int i = 0; i < en; i++) EMIT(esc[i]);
        if (key && scalar) v->khash = json_key_hash_scalar(v->khash, scalar);
        if (key) {
            // a force-closed key must not complete into a duplicate the
            // feed guard would have refused (and json_parse will refuse):
            // extend it until its decoded hash is fresh in this object
            for (int guard = 0; guard < 64; guard++) {
                bool dup = false;
                for (int i = 0; i < v->kseen_n; i++)
                    if (v->kseen_depth[i] == (uint8_t)v->depth &&
                        v->kseen_hash[i] == v->khash) { dup = true; break; }
                if (!dup) break;
                EMIT('_');
                v->khash = json_key_hash_byte(v->khash, '_');
            }
        }
        EMIT('"');
        if (key) v->st = S_COLON;
        else value_done(v);
    }
    if (v->st == S_COLON)      { EMIT(':'); v->st = S_VALUE; }
    if (v->st == S_KEY_EXPECT) {
        // the invented key must dodge the guard too: "_" may exist already
        uint32_t kh = json_key_hash_byte(json_key_hash_init(), '_');
        EMIT('"'); EMIT('_');
        for (int guard = 0; guard < 64; guard++) {
            bool dup = false;
            for (int i = 0; i < v->kseen_n; i++)
                if (v->kseen_depth[i] == (uint8_t)v->depth &&
                    v->kseen_hash[i] == kh) { dup = true; break; }
            if (!dup) break;
            EMIT('_'); kh = json_key_hash_byte(kh, '_');
        }
        EMIT('"'); EMIT(':');
        v->st = S_VALUE;
    }
    if (v->st == S_LIT) {
        const char *lit = LITS[v->lit];
        while (lit[v->sub]) EMIT(lit[v->sub++]);
        value_done(v);
    }
    if (v->st == S_NUM_MINUS || v->st == S_NUM_FRAC0 ||
        v->st == S_NUM_EXP0 || v->st == S_NUM_EXP1) {
        EMIT('0');
        value_done(v);
    }
    if (v->st == S_NUM_ZERO || v->st == S_NUM_INT ||
        v->st == S_NUM_FRAC || v->st == S_NUM_EXP) {
        value_done(v); // number already complete as-is
    }
    if (v->st == S_VALUE || v->st == S_ARR_FIRST) {
        EMIT('n'); EMIT('u'); EMIT('l'); EMIT('l');
        value_done(v);
    }
    // close remaining containers
    while (!v->done && v->depth > 0 &&
           (v->st == S_AFTER || v->st == S_KEY_OR_END)) {
        uint8_t top = v->stack[v->depth - 1];
        EMIT(top == 'O' ? '}' : ']');
        v->depth--;
        value_done(v);
    }
    #undef EMIT
    out[m] = 0;
    return m;
}

// See jsonmode.h. The table encoding is the one schema.c's string frames
// have always used: index 0 is "between scalars"; 1-3 are plain 2/3/4-byte
// continuations; 4-7 are the four constrained second bytes (E0, ED, F0, F4)
// that exclude overlongs, surrogates and values past U+10FFFF.
bool json_utf8_byte(uint8_t *state, uint8_t c) {
    static const uint8_t LO[]   = { 0, 0x80, 0x80, 0x80, 0xA0, 0x80, 0x90, 0x80 };
    static const uint8_t HI[]   = { 0, 0xBF, 0xBF, 0xBF, 0xBF, 0x9F, 0xBF, 0x8F };
    static const uint8_t NEXT[] = { 0, 0,    1,    2,    1,    1,    2,    2 };
    if (*state) {
        if (c < LO[*state] || c > HI[*state]) return false;
        *state = NEXT[*state];
        return true;
    }
    if (c < 0x80) return true;
    if (c >= 0xC2 && c <= 0xDF) { *state = 1; return true; }
    if (c >= 0xE0 && c <= 0xEF) {
        *state = c == 0xE0 ? 4 : c == 0xED ? 5 : 2;
        return true;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        *state = c == 0xF0 ? 6 : c == 0xF4 ? 7 : 3;
        return true;
    }
    return false;
}

int json_utf8_close(uint8_t *state, char out[3]) {
    static const uint8_t LO[]   = { 0, 0x80, 0x80, 0x80, 0xA0, 0x80, 0x90, 0x80 };
    static const uint8_t NEXT[] = { 0, 0,    1,    2,    1,    1,    2,    2 };
    int n = 0;
    while (*state && n < 3) {
        out[n++] = (char)LO[*state];
        *state = NEXT[*state];
    }
    return n;
}

int jsonv_number_state(const jsonv *v) {
    switch (v->st) {
    case S_NUM_MINUS: case S_NUM_ZERO: case S_NUM_INT:
    case S_NUM_FRAC0: case S_NUM_FRAC:
        return 1;
    case S_NUM_EXP0: case S_NUM_EXP1: case S_NUM_EXP:
        return 2;
    default:
        return 0;
    }
}

bool jsonv_feed(jsonv *v, const char *s, int n) {
    for (int i = 0; i < n; i++) {
        bool re;
        if (!feed_byte(v, (uint8_t)s[i], &re)) return false;
        if (re && !feed_byte(v, (uint8_t)s[i], &re)) return false;
    }
    return true;
}

// The machine reads stack[0 .. depth-1] and nothing above it: push() writes
// stack[depth] before depth names it, and every reader indexes depth-1 or
// less. So a snapshot only has to carry that prefix. Anything added here that
// reads a slot above `depth` breaks this — and jsonv_trial's test poisons the
// scratch, so it breaks loudly.
void jsonv_snapshot(jsonv *dst, const jsonv *src) {
    int d = src->depth;
    if (d < 0) d = 0;
    else if (d > (int)sizeof(src->stack)) d = (int)sizeof(src->stack);
    memcpy(dst->stack, src->stack, (size_t)d);
    dst->depth = src->depth;
    dst->st = src->st;
    dst->sub = src->sub;
    dst->lit = src->lit;
    dst->utf8 = src->utf8;
    dst->done = src->done;
    dst->esc = src->esc;
    dst->esc_hi = src->esc_hi;
    dst->khash = src->khash;
    dst->kseen_n = src->kseen_n;
    int ks = src->kseen_n;
    if (ks > JSON_KEY_SEEN_MAX) ks = JSON_KEY_SEEN_MAX;
    memcpy(dst->kseen_hash, src->kseen_hash,
           (size_t)ks * sizeof(src->kseen_hash[0]));
    memcpy(dst->kseen_depth, src->kseen_depth, (size_t)ks);
}

bool jsonv_trial(const jsonv *v, jsonv *scratch, const char *s, int n) {
    jsonv_snapshot(scratch, v);
    return jsonv_feed(scratch, s, n);
}
