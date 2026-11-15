// JSON Schema `pattern` as a byte automaton (R2.18).
//
// A pattern is an ECMA-262 regular expression with Unicode-mode meanings
// (code points; \d and \w ASCII; \s ECMA's white space and line terminators),
// and it is SEARCHED: an unanchored pattern matches anywhere in the string.
// It is compiled into a Thompson automaton over the UTF-8 bytes a JSON string
// spells without an escape: code-point classes become UTF-8 byte ranges, so a
// dot, a negated class or a quantified class consumes whole code points, and
// every class is first restricted to what JSON writes literally (no control
// character, no quote, no backslash). The enforced language is therefore the
// declared one restricted to literal spellings: never wider. A pattern whose
// language that leaves empty is refused when it is compiled.
//
// What is not regular, or not built, is refused by name: backreferences,
// lookaround, \b and \B, \p{...}, named groups, inline flags, and a pattern
// over JSRE_MAX_STATES automaton states.
#ifndef RUNNER_JSRE_H
#define RUNNER_JSRE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum { JSRE_MAX_STATES = 4096 };

typedef struct jsre jsre;

// NULL, with the reason in err, when the pattern is refused.
jsre  *jsre_compile(const char *source, char *err, size_t errcap);
void   jsre_free(jsre *re);
int    jsre_states(const jsre *re);
// The bytes a compiled pattern holds, as asked of the allocator: its states
// at capacity, its bitsets, its length sequence (test_jsre checks it is not
// under what the heap holds). JSRE_SCHEMA_BYTES caps their sum over one
// compiled request, a schema or a request's tools together (schema.c), so a
// request bounds what it can make the server hold however many patterns it
// carries.
size_t jsre_bytes(const jsre *re);
#define JSRE_SCHEMA_BYTES ((size_t)64 << 20)
// A state set is jsre_words() uint64_t words, caller-owned.
size_t jsre_words(const jsre *re);
void   jsre_start(const jsre *re, uint64_t *set);
// One byte of the string's value. False: no live state is left (the byte is
// refused; the set is then undefined).
bool   jsre_step(const jsre *re, uint64_t *set, uint8_t b);
// May the string end here?
bool   jsre_accepts(const jsre *re, const uint64_t *set);

// R2.18 length bounds, exact (see jsre.c). jsre_lengths prepares them: `upto`
// is the largest count a caller will ask (a maxLength), or -1 for none; false
// with the reason when the lengths do not settle within JSRE_LEN_MAX code
// points. jsre_len_ok: does `set` have an accepting completion of l more code
// points for some lo <= l <= hi (hi < 0: unbounded)?
enum { JSRE_LEN_MAX = 4096 };
bool   jsre_lengths(jsre *re, int upto, char *err, size_t errcap);
bool   jsre_len_ok(const jsre *re, const uint64_t *set, int lo, int hi);
int    jsre_len_entries(const jsre *re);
// A closer's filler (see jsre.c): the shortest completion from `set` (NULL:
// the start) of lo to hi more code points that the pattern accepts, written
// to `out`; its byte count (cap when it is longer), or -1. Needs jsre_lengths.
int    jsre_witness(const jsre *re, const uint64_t *set, int lo, int hi, char *out, int cap);

// Byte classes: bytes no transition tells apart (a DFA table has one entry per class).
int    jsre_classes(const jsre *re);

// R2.18, the design after the state-set stepper: the same automaton as a DFA
// built lazily, one DFA state per state set reached, each with a transition
// table (one entry per byte class) filled on first use. A byte then costs one
// table read once the transitions a request uses are built.
//
// The tables live in a POOL, one per compiled schema, shared by its patterns:
// a byte budget, and a flush when it is spent that keeps the state being
// stepped from and every state a persistent validator passed through (`keep`:
// sval_feed sets it, a trial's scratch does not). The cache is a pure memo, so
// its size, a flush, or a budget of one byte changes speed, never an answer.
// A pool belongs to one thread: its first fill names the thread, and a fill
// from any other is refused (JSRE_FAIL), as is a handle a flush invalidated.
// JSRE_POOL_BYTES is a compiled schema's budget, in bytes asked of the allocator
// for the cache and the books (the pool, its DFAs, their arrays), kept sets apart. JSRE_DFA_MAX
// caps the kept states (bounded anyway by the bytes fed and the DFA's size);
// past it a step fails loudly rather than refusing a byte the pattern allows.
#define JSRE_POOL_BYTES ((size_t)8 << 20)
enum { JSRE_DFA_MAX = 16384 };
typedef struct jsre_pool jsre_pool;
typedef struct jsre_dfa  jsre_dfa;
typedef uint64_t jsre_st;                   // a state handle: tag << 32 | slot (see jsre.c)
#define JSRE_DEAD ((jsre_st)UINT64_MAX)     // no live state: the byte is refused
#define JSRE_FAIL ((jsre_st)UINT64_MAX - 1) // the pool refused: its stat names why
typedef struct {
    size_t      budget;              // the pool's budget, bytes
    size_t      cache, kept, books, high;  // bytes: tables, length bits and unkept sets; kept sets;
                                     // the pool's and its DFAs' own arrays, by capacity; the most held
    uint64_t    flushes;
    int         kept_states;
    const char *failed;              // NULL, or why a step returned JSRE_FAIL
    const char *failed_label;        // the failing DFA's label (jsre_dfa_label), or NULL
    int         failed_states;       // and the states it held then
} jsre_pool_stat;
jsre_pool *jsre_pool_new(size_t budget);
void       jsre_pool_free(jsre_pool *p);    // after its DFAs
void       jsre_pool_stats(const jsre_pool *p, jsre_pool_stat *s);
void       jsre_pool_set_kept_max(jsre_pool *p, int kept_max);  // lowers JSRE_DFA_MAX only (tests)
jsre_dfa  *jsre_dfa_new(const jsre *re, jsre_pool *pool);
void       jsre_dfa_free(jsre_dfa *d);
void       jsre_dfa_label(jsre_dfa *d, const char *label);  // named in a failure (not copied)
jsre_st    jsre_dfa_start(jsre_dfa *d);
jsre_st    jsre_dfa_step(jsre_dfa *d, jsre_st st, uint8_t b, bool keep);
bool       jsre_dfa_accepts(const jsre_dfa *d, jsre_st st);
int        jsre_dfa_states(const jsre_dfa *d);  // slots holding a state now
bool       jsre_dfa_kept(const jsre_dfa *d, jsre_st st);  // survives every flush
const uint64_t *jsre_dfa_set(const jsre_dfa *d, jsre_st st);  // the state's set, or NULL
// jsre_len_ok for a DFA state (the pattern's jsre_lengths prepared), cached per state.
bool       jsre_dfa_len_ok(jsre_dfa *d, jsre_st st, int lo, int hi);

#endif
