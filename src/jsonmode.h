// Incremental JSON validator (json_mode).
#ifndef RUNNER_JSONMODE_H
#define RUNNER_JSONMODE_H

#include <stdint.h>
#include <stdbool.h>

// incremental validator: accepts byte strings only while they remain a valid
// prefix of a single JSON object; small and memcpy-copyable for lookahead
// Duplicate-key guard capacity, shared with schema.c's map guard: hashes of
// keys already closed in each open object, tagged with that object's depth.
// The hash runs over DECODED key content (raw bytes, decoded escapes,
// surrogate pairs as their scalar), so `"a"` and `"a"` collide exactly
// as they do in json_parse, which refuses the duplicate either way. A hash
// collision refuses a distinct key (over-constraint) and can never admit a
// duplicate. Past the capacity the guard stops tracking.
#define JSON_KEY_SEEN_MAX 16

typedef struct {
    uint8_t stack[200];     // container nesting: 'O' object, 'A' array
    int16_t depth;
    uint8_t st, sub, lit;   // micro-state, escape/digit progress, literal id
    uint8_t utf8;           // pending raw UTF-8 scalar (see json_utf8_byte)
    bool    done;           // a complete top-level object has been parsed
    uint16_t esc;           // \uXXXX value accumulated so far
    uint16_t esc_hi;        // pending high surrogate of the current key's pair
    uint32_t khash;         // running decoded-content hash of the open key
    uint8_t  kseen_n;       // duplicate-key guard (see JSON_KEY_SEEN_MAX)
    uint32_t kseen_hash[JSON_KEY_SEEN_MAX];
    uint8_t  kseen_depth[JSON_KEY_SEEN_MAX];
} jsonv;

// One hex digit of a `\uXXXX` escape, for both validators.
//
// `sub` is the string micro-state: 2..5 are the four digits of an escape, and
// a HIGH surrogate then requires its low half -- 6 waits for the backslash, 7
// for the `u`, 8..11 for its digits. `esc` accumulates the digits.
//
// The pairing rules are the ones json.c's parser enforces, and they live here
// rather than in each validator because a validator that admitted an escape
// the parser refuses would let a constrained model generate a document this
// program cannot read back. Returns false for a digit that commits the string
// to such an escape; every rejection leaves other digits legal, so a
// constrained model is never left with nothing to sample.
bool json_escape_hex(uint8_t *sub, uint16_t *esc, uint8_t c);
// Finish an escape that generation stopped inside, writing at most 12 bytes
// to `out` and leaving the string state ready for the closing quote. Padding
// the missing digits with zeros is what a caller would do instead, and that
// yields the NUL escape or an unpaired surrogate -- both refused by json.c,
// so the force-closed document would not parse.
// `hi` is the pending high surrogate when closing lands inside a pair's low
// half (0 otherwise); on return *scalar is the decoded code point the close
// completed, or 0 when it completed none — the duplicate-key guards hash it.
int  json_escape_close(uint8_t *sub, uint16_t *esc, uint16_t hi,
                       char *out, int cap, uint32_t *scalar);
// FNV-1a over decoded key content, shared by both duplicate-key guards so
// they agree with each other and with json_parse's refusal.
uint32_t json_key_hash_init(void);
uint32_t json_key_hash_byte(uint32_t h, uint8_t c);
uint32_t json_key_hash_scalar(uint32_t h, uint32_t cp);
// the byte a simple escape decodes to ('n' -> 0x0A, ...), 0 if not simple
uint8_t  json_escape_decode_simple(uint8_t c);
// Raw UTF-8 string content, shared by both validators for the same reason as
// json_escape_hex: a validator that admits a byte sequence json_parse
// refuses would let a constrained model generate a document this program
// cannot read back. *state is 0 between scalars; nonzero encodes the
// sequence's range class and continuation bytes remaining. Returns false
// for a byte no continuation can rescue (a lone continuation byte, an
// overlong lead, 0xF5.., or an out-of-range continuation).
bool json_utf8_byte(uint8_t *state, uint8_t c);
// Finish a scalar that generation stopped inside, writing at most 3 minimal
// continuation bytes: truncating after a lead byte and writing the closing
// quote directly would emit ill-formed UTF-8, which json.c refuses.
int  json_utf8_close(uint8_t *state, char out[3]);

void jsonv_init(jsonv *v);      // accept exactly one JSON object
void jsonv_init_any(jsonv *v);  // accept exactly one JSON value of any kind
bool jsonv_feed(jsonv *v, const char *s, int n);
// Copy only the LIVE part of `src` into `dst`: the container stack above
// `depth` is never read before it is written, so copying those bytes is pure
// cost. Only the trial probes below need this; ordinary state keeping should
// just assign the struct.
void jsonv_snapshot(jsonv *dst, const jsonv *src);
// Would `s` keep the validator alive? Answers without touching `v`, running
// the trial in caller-owned `scratch` whose previous contents are irrelevant.
//
// This is the candidate-token oracle: constrained sampling with top_k off
// calls it once per vocabulary entry, per token, so what it copies is a
// decode-speed property. See sval_trial in schema.h.
bool jsonv_trial(const jsonv *v, jsonv *scratch, const char *s, int n);
// true if the machine stopped at a self-terminated value boundary (numbers)
bool jsonv_value_end(const jsonv *v);
// 0: not inside a number; 1: in the mantissa; 2: in the exponent part. For
// wrappers that track the number spelling externally — this machine never
// buffers one, so a caller enforcing spelling-level rules (schema.c's
// free-subtree number guard) needs to know where the machine stands.
int jsonv_number_state(const jsonv *v);
// force-complete the object (token budget ran out); returns bytes written
int  jsonv_close(jsonv *v, char *out, int cap);

#endif // RUNNER_JSONMODE_H
