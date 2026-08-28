// libFuzzer harness for sval_trial (src/schema.c) — the candidate-token
// oracle, DIFFERENTIALLY against a full struct copy.
//
// The oracle copies only the live part of the validator into a scratch whose
// previous contents are arbitrary. Its whole contract is: answer exactly what
// a full copy fed the same bytes would answer, from every reachable state.
// The v0.4.0 release gate caught that contract silently broken (the map
// seen-key guard was live state the copy did not carry; poison in n_seen
// walked the guard arrays out of bounds and answered from garbage), and the
// answer-only comparison in the unit test had greened on an accidental
// match. So this harness compares STATE, not just answers, at every byte of
// every mutated stream, with the scratch poisoned before every probe.
//
// Input layout, shared with fuzz_sval_feed: a 2-byte little-endian length
// prefix, then `len` bytes of schema JSON, then the stream. A non-compiling
// schema half falls back to a fixed schema chosen to reach the state the
// plain feed fuzzer's fallback cannot: typed-additionalProperties maps
// (nested, so the seen guard tracks two depths at once), bounded numbers,
// bounded strings (UTF-8 length accounting), and an enum.
//
// Invariants trapped on, each one a real bug class from the 2026-08 waves:
//   1. trial answer == full-copy answer                      (the contract)
//   2. accepted trial leaves scratch state == full-copy state (guard, stack,
//      number spelling, submachine — an unequal field answers a FUTURE probe
//      wrong even when this one matched)
//   3. the probed validator is bit-identical before and after (const-ness)
//   4. accepted-prefix + sval_close output parses as JSON     (the closer's
//      published guarantee, which holds under truncation too)
//
// Deliberately NOT trapped on: feeding the closer's bytes back through
// sval_feed. Under cap pressure the closer truncates fills below minItems/
// minLength by design (finish_reason "length"), and the feed grammar rightly
// refuses what the closer rightly emitted — parseability is the contract
// there, grammar acceptance is not.
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "json.h"
#include "runner.h"

static const char FALLBACK[] =
    "{\"type\":\"object\","
    "\"properties\":{"
      "\"m\":{\"type\":\"object\",\"additionalProperties\":"
        "{\"type\":\"object\",\"additionalProperties\":"
          "{\"type\":\"integer\",\"minimum\":-3,\"maximum\":700}}},"
      "\"s\":{\"type\":\"string\",\"minLength\":1,\"maxLength\":5},"
      "\"e\":{\"enum\":[\"aa\",\"ab\",7,null]},"
      "\"v\":{\"type\":\"array\",\"items\":"
        "{\"type\":\"number\",\"exclusiveMinimum\":0,\"maximum\":99},"
        "\"minItems\":1,\"maxItems\":6},"
      "\"f\":{\"type\":\"object\"}},"
    "\"required\":[\"m\",\"s\"]}";

// live-state equality between an accepted trial scratch and the full copy;
// compares only what the machine may read later, which is exactly what
// sval_trial promises to carry
static int live_state_equal(const sval *a, const sval *b) {
    if (a->depth != b->depth || a->done != b->done) return 0;
    if (a->last_enum != b->last_enum) return 0;
    int d = a->depth;
    if (d < 0) d = 0;
    if (d > (int)(sizeof(a->stack) / sizeof(a->stack[0])))
        d = (int)(sizeof(a->stack) / sizeof(a->stack[0]));
    if (memcmp(a->stack, b->stack, (size_t)d * sizeof(a->stack[0]))) return 0;
    // num_text's terminator is live: number_text_in_bounds reads the C
    // string, so compare num_len + 1 bytes, not num_len
    if (a->num_len != b->num_len) return 0;
    if (memcmp(a->num_text, b->num_text, (size_t)a->num_len + 1)) return 0;
    if (a->n_seen != b->n_seen) return 0;
    int ns = a->n_seen;
    if (ns > MAP_SEEN_MAX) ns = MAP_SEEN_MAX;
    if (memcmp(a->seen_hash, b->seen_hash,
               (size_t)ns * sizeof(a->seen_hash[0]))) return 0;
    if (memcmp(a->seen_depth, b->seen_depth, (size_t)ns)) return 0;
    // the submachine's own live prefix only: its dead stack bytes are
    // poison in the scratch by design (jsonv_snapshot copies to depth)
    const jsonv *ja = &a->any, *jb = &b->any;
    if (ja->depth != jb->depth || ja->st != jb->st || ja->sub != jb->sub ||
        ja->lit != jb->lit || ja->utf8 != jb->utf8 || ja->done != jb->done ||
        ja->esc != jb->esc)
        return 0;
    int jd = ja->depth;
    if (jd < 0) jd = 0;
    if (jd > (int)sizeof(ja->stack)) jd = (int)sizeof(ja->stack);
    if (memcmp(ja->stack, jb->stack, (size_t)jd)) return 0;
    return 1;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 2) return 0;
    size_t len = (size_t)data[0] | ((size_t)data[1] << 8);
    data += 2; size -= 2;
    if (len > size) len = size;

    char err[256];
    jv    *j = json_parse((const char *)data, len);
    snode *s = j ? schema_compile(j, err, (int)sizeof(err)) : NULL;
    jv_free(j);

    jv *fj = NULL;
    if (!s) {
        fj = json_parse(FALLBACK, sizeof(FALLBACK) - 1);
        s  = fj ? schema_compile(fj, err, (int)sizeof(err)) : NULL;
        if (!s) { jv_free(fj); return 0; }
    }

    const uint8_t *stream = data + len;
    size_t         n      = size - len;

    static char accepted[4096];
    size_t      n_acc = 0;

    sval v;
    sval_init(&v, s);
    for (size_t i = 0; i < n; i++) {
        char b = (char)stream[i];

        sval full = v;
        bool want = sval_feed(&full, &b, 1);

        sval before = v;
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        bool got = sval_trial(&v, &scratch, &b, 1);

        if (got != want) __builtin_trap();                       // invariant 1
        if (want && !live_state_equal(&scratch, &full))
            __builtin_trap();                                    // invariant 2
        if (memcmp(&before, &v, sizeof(v))) __builtin_trap();    // invariant 3

        if (!want) continue;         // refused byte: state untouched, move on
        (void)sval_feed(&v, &b, 1);
        if (n_acc < sizeof(accepted)) accepted[n_acc++] = b;
        if (v.done) break;
    }

    char out[1024];
    sval closing = v;
    int  wrote = sval_close(&closing, out, (int)sizeof(out));
    if (wrote < 0 || (size_t)wrote > sizeof(out)) __builtin_trap();

    // the closer's guarantee: what went on the wire parses. Two legitimate
    // zero-byte closes are excluded: a document the stream already finished
    // (v.done: the accepted prefix must parse on its own) and the
    // fabrication line (nothing generated, nothing invented — the accepted
    // bytes are at most insignificant whitespace and there is no document).
    if ((v.done || wrote > 0) && n_acc + (size_t)wrote <= sizeof(accepted)) {
        memcpy(accepted + n_acc, out, (size_t)wrote);
        jv *doc = json_parse(accepted, n_acc + (size_t)wrote);
        if (!doc) __builtin_trap();                              // invariant 4
        jv_free(doc);
    }

    schema_free(s);
    jv_free(fj);
    return 0;
}
