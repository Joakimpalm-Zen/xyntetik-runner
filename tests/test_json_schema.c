#include "runner.h"
#include "json.h"
#include "schema_int.h"
#include "jsre_cases.h"
#include "fpenv.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// setenv is POSIX; MinGW has _putenv_s (R2.18's seam checks below)
#ifdef _WIN32
#define setenv_compat(k, v) _putenv_s(k, v)
#else
#define setenv_compat(k, v) setenv(k, v, 1)
#endif

static void test_strict_bounded_numbers(void) {
    // 1e400/-1e400 overflow to inf: isfinite() catches that only when the
    // build is not -ffast-math, which the release build is
    const char *bad[] = { "01", "1.", "-.1", "1e", "-nan", "1e9999",
                          "1e400", "-1e400", "1e308000" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    const char bounded[] = { '1', '2' }; // deliberately not NUL-terminated
    jv *v = json_parse(bounded, 1);
    assert(v != NULL);
    assert(v->type == J_NUM && v->num == 1.0);
    jv_free(v);

    v = json_parse("-1.25e+3", 8);
    assert(v != NULL);
    assert(v->type == J_NUM && v->num == -1250.0);
    jv_free(v);
}

static void test_json_rejects_unpaired_utf16_surrogates(void) {
    const char *bad[] = {
        "\"\\uD800\"",          // lone high surrogate
        "\"\\uD800x\"",         // high surrogate followed by ordinary text
        "\"\\uD800\\u0041\"",  // high surrogate followed by a non-low escape
        "\"\\uDC00\"",          // lone low surrogate
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    jv *paired = json_parse("\"\\uD83D\\uDE00\"", 14);
    assert(paired != NULL);
    assert(!strcmp(paired->str, "\xF0\x9F\x98\x80"));
    jv_free(paired);

    char out[4];
    int outn = -1;
    assert(json_unescape("\\uDC00", 6, out, &outn) == -1);
    assert(json_unescape("\\uD800x", 7, out, &outn) == -1);
    assert(json_unescape("\\uD800\\u0041", 12, out, &outn) == -1);
    // a high surrogate followed by any escape that is not \u can never pair:
    // decoding it alone emits the surrogate as CESU-8 (ED A0 80), which is not
    // UTF-8 at all and breaks every strict client downstream
    assert(json_unescape("\\uD800\\n", 8, out, &outn) == -1);
    assert(json_unescape("\\uD800\\\\", 8, out, &outn) == -1);
    assert(json_unescape("\\uD83D\\uDE00", 12, out, &outn) == 12);
    assert(outn == 4 && !memcmp(out, "\xF0\x9F\x98\x80", 4));
}

static void test_json_rejects_ill_formed_raw_utf8(void) {
    const char *bad[] = {
        "\"\x80\"",             // bare continuation
        "\"\xC0\x80\"",         // overlong NUL
        "\"\xED\xA0\x80\"",     // UTF-8 encoding of a surrogate
        "\"\xF4\x90\x80\x80\"", // beyond U+10FFFF
        "\"\xE2\x82\"",         // truncated sequence
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    const char good[] = "\"caf\xC3\xA9 \xF0\x9F\x98\x80\"";
    jv *v = json_parse(good, strlen(good));
    assert(v != NULL);
    jv_free(v);
}

static void test_json_rejects_embedded_nul(void) {
    const char *bad[] = {
        "\"a\\u0000b\"",
        "{\"model\\u0000shadow\":1}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    char out[4];
    int outn = -1;
    assert(json_unescape("\\u0000", 6, out, &outn) == -1);
}

static void test_json_rejects_duplicate_object_keys(void) {
    const char *bad[] = {
        "{\"max_tokens\":4,\"max_tokens\":4096}",
        "{\"outer\":{\"model\":\"safe\",\"model\":\"shadow\"}}",
        "{\"model\":1,\"m\\u006fdel\":2}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(json_parse(bad[i], strlen(bad[i])) == NULL);

    // Reusing a name in different objects is not a duplicate.
    const char *good = "[{\"id\":1},{\"id\":2}]";
    jv *v = json_parse(good, strlen(good));
    assert(v != NULL);
    jv_free(v);
}

// A `\uXXXX` escape is the one place a truncated string cannot be finished
// with arbitrary filler: `\u0000` and an unpaired surrogate are both refused
// by this file's own parser, so padding the missing digits with zeros
// force-closes a document that then does not parse. Every truncation point
// inside an escape has to close to something readable -- and the two
// validators have to agree with the parser about which escapes exist at all,
// or a model can generate a document runner cannot read back.
static void test_escapes_are_paired_and_closable(void) {
    // json_mode: every prefix of a document with a surrogate pair in it
    const char *doc = "{\"a\":\"x\\uD83D\\uDE00y\"}";
    for (size_t cut = 1; cut <= strlen(doc); cut++) {
        jsonv v;
        jsonv_init(&v);
        assert(jsonv_feed(&v, doc, (int)cut));
        if (v.done) continue;
        char tail[64];
        int n = jsonv_close(&v, tail, sizeof(tail));
        assert(n > 0);
        char full[128];
        snprintf(full, sizeof(full), "%.*s%s", (int)cut, doc, tail);
        jv *parsed = json_parse(full, strlen(full));
        if (!parsed) fprintf(stderr, "json_mode cut %zu: %s\n", cut, full);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    // and the escapes this parser refuses are refused by the validator too,
    // rather than generated and then found unreadable
    const char *bad[] = { "{\"a\":\"\\u0000", "{\"a\":\"\\uDC00\"}",
                          "{\"a\":\"\\uD800\"}", "{\"a\":\"\\uD800\\n\"}" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jsonv v;
        jsonv_init(&v);
        assert(!jsonv_feed(&v, bad[i], (int)strlen(bad[i])));
    }

    // the same two properties through the schema validator
    const char *src = "{\"type\":\"object\",\"properties\":"
                      "{\"a\":{\"type\":\"string\"}},\"required\":[\"a\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    for (size_t cut = 1; cut <= strlen(doc); cut++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, doc, (int)cut));
        if (v.done) continue;
        char tail[64];
        int n = sval_close(&v, tail, sizeof(tail));
        assert(n > 0);
        char full[128];
        snprintf(full, sizeof(full), "%.*s%s", (int)cut, doc, tail);
        jv *parsed = json_parse(full, strlen(full));
        if (!parsed) fprintf(stderr, "schema cut %zu: %s\n", cut, full);
        assert(parsed != NULL);
        jv_free(parsed);
        sval chk; sval_init(&chk, schema);
        assert(sval_feed(&chk, full, (int)strlen(full)) && chk.done);
    }
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v; sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])));
    }
    // a well-formed pair still passes, and counts as one character
    const char *good = "{\"a\":\"\\uD83D\\uDE00\"}";
    sval ok; sval_init(&ok, schema);
    assert(sval_feed(&ok, good, (int)strlen(good)) && ok.done);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_json_close_partial_string(void) {
    jsonv v;
    jsonv_init(&v);
    assert(jsonv_feed(&v, "{\"a\":\"x", 7));
    char out[64];
    int n = jsonv_close(&v, out, sizeof(out));
    assert(n > 0);

    char full[128];
    snprintf(full, sizeof(full), "{\"a\":\"x%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    jv_free(parsed);
}

// R6.68.101: a budget that ends right after '[' closes the array empty.
// The closer wrote "null" first, an element the model never produced, where
// ']' was a legal next byte: {"a":[ came back as {"a":[null]}.
static void test_json_close_leaves_an_opened_array_empty(void) {
    static const struct { const char *head, *want; } cases[] = {
        { "{\"a\":[",        "]}"  },
        { "{\"a\":[[",       "]]}" },
        { "{\"a\":[1,[",     "]]}" },
        { "{\"a\":[1,",      "null]}" },   // after a comma a value is owed
        { "{\"a\":",         "null}" },
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        jsonv v;
        jsonv_init(&v);
        assert(jsonv_feed(&v, cases[i].head, (int)strlen(cases[i].head)));
        char out[64], full[128];
        int n = jsonv_close(&v, out, sizeof(out));
        if (n != (int)strlen(cases[i].want) || strcmp(out, cases[i].want) != 0) {
            fprintf(stderr, "FAIL: %s closes with %s, want %s\n", cases[i].head, out, cases[i].want);
            assert(0);
        }
        snprintf(full, sizeof(full), "%s%s", cases[i].head, out);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv_free(parsed);
    }
}

static void test_schema_required_close(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"a\":{\"type\":\"string\"},"
        "\"b\":{\"type\":\"integer\"}"
        "},\"required\":[\"a\",\"b\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "{\"a\":\"x\"", 8));
    char out[128];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);

    char full[256];
    snprintf(full, sizeof(full), "{\"a\":\"x\"%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(jv_get(parsed, "a") != NULL);
    assert(jv_get(parsed, "b") != NULL);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

// A free-keyed object (SN_MAP: additionalProperties with no fixed keys)
// must not emit the same key twice: {"a":1,"a":2} parses nowhere in this
// codebase -- json_parse refuses duplicates -- so a validator that admits it
// generates a document the runner itself cannot read back. The guard hashes
// the SPELLING the model feeds and refuses the duplicate at its closing
// quote, where every other continuation is still legal. A key spelled two
// different ways (raw vs \u-escape) is outside the guard's scope and
// remains a parser-level refusal, documented in schema.h.
static void test_map_refuses_a_duplicate_key_spelling(void) {
    const char *src =
        "{\"type\":\"object\",\"additionalProperties\":{\"type\":\"integer\"}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    // {"a":1,"a  -- the repeat key is typed up to its closing quote
    assert(sval_feed(&v, "{\"a\":1,\"a", 9));
    assert(!sval_feed(&v, "\"", 1));        // the duplicate cannot close
    sval_init(&v, schema);
    assert(sval_feed(&v, "{\"a\":1,\"ab\":2}", 14));  // extending it is legal
    assert(v.done);

    // a sibling map opened after this one closes starts with a clean guard
    const char *nested =
        "{\"type\":\"object\",\"properties\":{"
        "\"x\":{\"type\":\"object\",\"additionalProperties\":{\"type\":\"integer\"}},"
        "\"y\":{\"type\":\"object\",\"additionalProperties\":{\"type\":\"integer\"}}"
        "},\"required\":[\"x\",\"y\"]}";
    jv *nj = json_parse(nested, strlen(nested));
    assert(nj != NULL);
    snode *ns = schema_compile(nj, err, sizeof(err));
    assert(ns != NULL);
    sval nv;
    sval_init(&nv, ns);
    const char *doc = "{\"x\":{\"a\":1},\"y\":{\"a\":2}}";
    assert(sval_feed(&nv, doc, strlen(doc)));
    assert(nv.done);

    schema_free(ns);
    jv_free(nj);
    schema_free(schema);
    jv_free(schema_json);
}

// Wherever the grammar REQUIRES whitespace -- a native-protocol literal
// like atem's newline separators -- the whitespace oracle must report it as
// content, or the engine masks every whitespace-only token at a position
// whose only legal byte IS whitespace and generation dead-ends on
// vocabularies without a combined token. Vocabulary-independent walk: along
// the grammar's forced path, any position whose sole admissible byte is
// whitespace must satisfy sval_ws_is_content. sval is memcpy-copyable by
// contract, which is what makes the probe loop legal.
static void test_required_whitespace_reports_as_content(void) {
    const char *tsrc =
        "[{\"type\":\"function\",\"function\":{\"name\":\"ping\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"x\":{\"type\":\"integer\"}},\"required\":[\"x\"]}}}]";
    jv *tools = json_parse(tsrc, strlen(tsrc));
    assert(tools != NULL);
    char err[160];
    snode *g = schema_compile_atem_turn(tools, false, NULL, NULL,
                                        ATEM_TURN_EITHER, err, sizeof(err));
    assert(g != NULL);
    sval v;
    sval_init(&v, g);
    // Drive the invoke opening up to its embedded newline. The literal is
    // "<atem:function_calls>\n<atem:invoke name=\"..." -- after the '>' the
    // grammar's next byte is the required '\n'.
    // wire shape verified by direct grammar walk: the invoke branch opens
    // with the recipient (the tool name), then the function_calls block
    // whose literal embeds the required newline
    const char *prefix = "assistant to=ping<|message|><atem:function_calls>";
    assert(sval_feed(&v, prefix, (int)strlen(prefix)));
    // vocabulary-independent facts about this position, probed on copies
    // (sval is memcpy-copyable by contract): the newline is admissible and
    // it is the ONLY admissible whitespace continuation...
    sval probe = v;
    assert(sval_feed(&probe, "\n", 1));
    probe = v;
    assert(!sval_feed(&probe, " ", 1));
    // ...and the oracle must therefore call whitespace CONTENT here, or the
    // engine masks every whitespace-only token at a position whose only
    // legal byte IS whitespace and generation dead-ends.
    assert(sval_ws_is_content(&v));
    // sanity: after the newline the protocol continues
    assert(sval_feed(&v, "\n<atem:invoke", 13));
    schema_free(g);
    jv_free(tools);
}

// A native call cannot be closed while a schema-REQUIRED argument is missing.
//
// This is the rule a served model's tool calls rest on, and a public claim
// now rests on it too: the lab's Kvist student learned to close find_flight
// after `origin` alone, because its distillation target put 0.645 of the mass
// on closing there (2026-09-25), and the answer to "can that reach a caller"
// is this grammar. member_candidates() offers the closing literal only when
// close_allowed() finds no required member left, so the model is never handed
// the token it wants. Pinned per byte rather than through a served request,
// so a refactor of the members walker fails here instead of in someone's
// release notes.
static void test_a_native_call_cannot_close_with_a_required_argument_missing(void) {
    const char *tsrc =
        "[{\"type\":\"function\",\"function\":{\"name\":\"find_flight\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"origin\":{\"type\":\"string\"},"
        "\"destination\":{\"type\":\"string\"},"
        "\"date\":{\"type\":\"string\"}},"
        "\"required\":[\"origin\",\"destination\",\"date\"]}}}]";
    jv *tools = json_parse(tsrc, strlen(tsrc));
    assert(tools != NULL);
    char err[160];
    snode *g = schema_compile_atem_turn(tools, false, NULL, NULL,
                                        ATEM_TURN_EITHER, err, sizeof(err));
    assert(g != NULL);
    sval v;
    sval_init(&v, g);
    const char *open =
        "assistant to=find_flight<|message|><atem:function_calls>\n<atem:invoke "
        "name=\"find_flight\">\n<atem:parameter name=\"origin\">Oslo</atem:parameter>";
    assert(sval_feed(&v, open, (int)strlen(open)));

    // two required arguments outstanding: the close is not on offer, and the
    // first byte that would spell it is refused outright
    sval probe = v;
    assert(!sval_feed(&probe, "\n</atem:invoke>", 15));
    probe = v;
    assert(sval_feed(&probe, "\n<atem:parameter name=\"destination\">", 36));

    // one outstanding: still refused
    const char *dest = "\n<atem:parameter name=\"destination\">Rome</atem:parameter>";
    assert(sval_feed(&v, dest, (int)strlen(dest)));
    probe = v;
    assert(!sval_feed(&probe, "\n</atem:invoke>", 15));

    // all three present: the close becomes legal
    const char *date = "\n<atem:parameter name=\"date\">2026-10-01</atem:parameter>";
    assert(sval_feed(&v, date, (int)strlen(date)));
    probe = v;
    assert(sval_feed(&probe, "\n</atem:invoke>", 15));

    // and a turn cut while an argument is missing closes to a LEGAL document
    // rather than a truncated one: the completion carries the missing keys,
    // which is why the envelope reports "length" rather than a finished call
    // (completion.c: only a cleanly finished document may claim tool_calls)
    sval cut;
    sval_init(&cut, g);
    assert(sval_feed(&cut, open, (int)strlen(open)));
    char out[512];
    int n = sval_close(&cut, out, sizeof(out));
    assert(n > 0);
    assert(strstr(out, "destination") != NULL);
    assert(strstr(out, "date") != NULL);
    assert(strstr(out, "</atem:invoke>") != NULL);
    schema_free(g);
    jv_free(tools);
}

// A constrained document must begin with its opening token.
//
// Leading whitespace is the livelock that burns the budget. A model that
// would rather write a preamble finds every prose token rejected, and the
// only legal bytes left are spaces and newlines — so it emits those, forever,
// and never reaches the opening brace. Measured on the stub model with
// tool_choice=required and max_tokens=32: 32 tab tokens and no document.
//
// Whitespace *inside* the document is untouched: it is how a real model
// pretty-prints, and Llama-3.2 emits `{ "a" : "b" }` spacing on every call.
// Only the run before the root value opens is banned, because that is the
// only position where burning whitespace yields a document with no content
// in it at all.
static void test_leading_whitespace_is_refused_but_interior_is_kept(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"a\":{\"type\":\"string\"},"
        "\"b\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}"
        "},\"required\":[\"a\",\"b\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    // every flavour of leading whitespace is refused outright
    const char *ws[] = { " ", "\n", "\t", "\r", "\n\n  " };
    for (size_t i = 0; i < sizeof(ws) / sizeof(*ws); i++) {
        sval_init(&v, schema);
        assert(!sval_feed(&v, ws[i], (int)strlen(ws[i])));
    }

    // the opening token itself is of course still fine
    sval_init(&v, schema);
    assert(sval_feed(&v, "{", 1));

    // ...and interior whitespace, in every position a model uses it, is
    // accepted exactly as before: this is Llama-3.2's real output shape
    const char *pretty =
        "{\n  \"a\" : \"x\" ,\n  \"b\" : [ \"p\" , \"q\" ]\n}";
    sval_init(&v, schema);
    assert(sval_feed(&v, pretty, (int)strlen(pretty)));
    assert(v.done);

    schema_free(schema);
    jv_free(schema_json);
}

// json mode (no schema) draws the same line through the same states.
static void test_json_mode_leading_whitespace_is_refused(void) {
    jsonv v;
    jsonv_init(&v);
    assert(!jsonv_feed(&v, " ", 1));
    jsonv_init(&v);
    assert(!jsonv_feed(&v, "\n\n", 2));

    jsonv_init(&v);
    const char *pretty = "{\n  \"a\" : [ 1 , 2 ] ,\n  \"b\" : null\n}";
    assert(jsonv_feed(&v, pretty, (int)strlen(pretty)));
    assert(v.done);
}

// A close() with no generated payload behind it used to invent a complete,
// schema-valid document out of nothing: `{"progress":"","next_step":""}`.
// That is indistinguishable from an answer the model actually produced, so a
// caller could not tell that its whole token budget had gone on a reasoning
// prelude that never reached the opening brace. Measured on Qwen3-4B, this
// silently replaced Thane's compaction summaries with empty session state.
//
// jsonv_close has always declined to fabricate ("returns 0 if generation
// never started an object"); sval_close must honour the same contract, so an
// unproductive decode fails visibly instead of parsing as a real answer.
static void test_schema_close_without_payload_fabricates_nothing(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"progress\":{\"type\":\"string\"},"
        "\"next_step\":{\"type\":\"string\"}"
        "},\"required\":[\"progress\",\"next_step\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    char out[256];

    // nothing fed at all
    sval v;
    sval_init(&v, schema);
    out[0] = 'x';
    assert(sval_close(&v, out, sizeof(out)) == 0);
    assert(out[0] == 0);

    // the moment the document opens, truncation must still complete it
    sval_init(&v, schema);
    assert(sval_feed(&v, "{", 1));
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[512];
    snprintf(full, sizeof(full), "{%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(jv_get(parsed, "progress") != NULL);
    assert(jv_get(parsed, "next_step") != NULL);
    jv_free(parsed);

    schema_free(schema);
    jv_free(schema_json);
}

// An empty "type" array compiled to a union with no alternatives. pick_alt
// then matched no byte, so sampling stalled and the forced-completion path
// read alts[0] out of a zero-byte allocation: a single unauthenticated
// request body segfaulted the whole server, taking every slot with it.
// Reachable nested as well, so the check belongs at the compile site.
static void test_schema_rejects_empty_type_union(void) {
    static const char *const bad[] = {
        "{\"type\":[]}",
        "{\"type\":\"array\",\"items\":{\"type\":[]},\"minItems\":1}",
        "{\"properties\":{\"a\":{\"type\":[]}},\"required\":[\"a\"]}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *j = json_parse(bad[i], strlen(bad[i]));
        assert(j != NULL);
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema == NULL);
        jv_free(j);
    }
}

// minItems/minLength are bounded only by INT_MAX, and the completion path
// looped to that bound even though the output buffer had long since filled.
// Nested, this pinned a slot for minutes. The loops now stop when the buffer
// is full, so an absurd bound costs the same as a small one.
static void test_schema_huge_min_bounds_terminate(void) {
    static const char *const src[] = {
        "{\"type\":\"string\",\"minLength\":2000000000}",
        "{\"type\":\"array\",\"minItems\":2000000000,"
        "\"items\":{\"type\":\"array\",\"minItems\":2000000000,"
        "\"items\":{\"type\":\"integer\"}}}",
    };
    for (size_t i = 0; i < sizeof(src) / sizeof(*src); i++) {
        jv *j = json_parse(src[i], strlen(src[i]));
        assert(j != NULL);
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);   // the schema itself is legal
        sval v;
        sval_init(&v, schema);
        char out[4096];
        int n = sval_close(&v, out, sizeof(out));   // must return, not spin
        assert(n >= 0 && n < (int)sizeof(out));
        schema_free(schema);
        jv_free(j);
    }
}

static void test_schema_rejects_bad_bounds(void) {
    const char *src =
        "{\"type\":\"array\",\"items\":{\"type\":\"string\"},"
        "\"minItems\":2,\"maxItems\":1}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "bounds") != NULL);
    jv_free(schema_json);
}

static void test_schema_rejects_non_integer_or_huge_bounds(void) {
    const char *bad[] = {
        "{\"type\":\"array\",\"items\":{},\"minItems\":1.5}",
        "{\"type\":\"array\",\"items\":{},\"maxItems\":1e100}",
        "{\"type\":\"string\",\"minLength\":1e100}",
        "{\"type\":\"number\",\"minimum\":\"zero\"}",
        "{\"type\":\"integer\",\"exclusiveMaximum\":null}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *schema_json = json_parse(bad[i], strlen(bad[i]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, "bounds") != NULL);
        jv_free(schema_json);
    }
}

static void test_schema_rejects_escaped_keys(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{\"bad\\\"key\":{\"type\":\"string\"}}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "property keys") != NULL);
    jv_free(schema_json);
}

// A keyword the compiler cannot enforce must be a compile error, not a
// silently weaker constraint.
static void test_schema_rejects_unenforceable_keywords(void) {
    static const char *const bad[] = {
        "{\"type\":\"integer\",\"multipleOf\":2}",
        "{\"allOf\":[{\"type\":\"string\"}]}",
        "{\"not\":{\"type\":\"string\"}}",
        "{\"type\":\"array\",\"uniqueItems\":true}",
        "{\"type\":\"array\",\"prefixItems\":[{\"type\":\"string\"}]}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"patternProperties\":{\"^x\":{\"type\":\"string\"}}}",
        // keyword that belongs to a different type than the one declared
        "{\"type\":\"string\",\"minItems\":2}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"maxLength\":3}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *schema_json = json_parse(bad[i], strlen(bad[i]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, "keyword") != NULL);
        jv_free(schema_json);
    }
    // a $ref is inlined since R2.9.2; one with no target is refused by name
    const char *dangling = "{\"$ref\":\"#/$defs/x\"}";
    jv *dj = json_parse(dangling, strlen(dangling));
    assert(dj != NULL);
    char derr[128];
    assert(schema_compile(dj, derr, sizeof(derr)) == NULL);
    assert(strstr(derr, "$ref '#/$defs/x' does not resolve") != NULL);
    jv_free(dj);
}

// `{"type":["object","null"]}` -- a nullable structured parameter, and one of
// the most common shapes in a real tool payload. An open object compiles to
// the generic any-value machine, which the union dispatcher treated as a
// catch-all because it can hold any value; but an OBJECT-rooted one only ever
// starts at '{', so it swallowed the dispatch and then refused the byte.
// Every alternative after it was unreachable, whatever it was.
static void test_schema_type_array_with_open_object(void) {
    // wrapped in a property so a bare number has a terminator to complete on
    static const struct { const char *types; const char *value; } cases[] = {
        { "[\"object\",\"null\"]",    "null" },
        { "[\"object\",\"null\"]",    "{}" },
        { "[\"object\",\"null\"]",    "{\"a\":1}" },
        { "[\"null\",\"object\"]",    "null" },
        { "[\"object\",\"string\"]",  "\"x\"" },
        { "[\"object\",\"array\"]",   "[]" },
        { "[\"object\",\"integer\"]", "5" },
        { "[\"object\",\"boolean\"]", "true" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[192], doc[64];
        snprintf(src, sizeof(src),
                 "{\"type\":\"object\",\"properties\":{\"v\":{\"type\":%s}},"
                 "\"required\":[\"v\"]}", cases[i].types);
        snprintf(doc, sizeof(doc), "{\"v\":%s}", cases[i].value);
        jv *schema_json = json_parse(src, strlen(src));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema != NULL);
        sval v; sval_init(&v, schema);
        if (!sval_feed(&v, doc, (int)strlen(doc)) || !v.done)
            fprintf(stderr, "type %s rejects %s\n", cases[i].types, doc);
        assert(v.done);
        schema_free(schema);
        jv_free(schema_json);
    }
    // and what the union must still refuse
    const char *src = "{\"type\":\"object\",\"properties\":"
                      "{\"v\":{\"type\":[\"object\",\"null\"]}},"
                      "\"required\":[\"v\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    const char *bad[] = { "{\"v\":\"x\"}", "{\"v\":5}", "{\"v\":true}",
                          "{\"v\":[]}" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v; sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])) || !v.done);
    }
    schema_free(schema);
    jv_free(schema_json);
}

// "Truncated output still parses" is a correctness gate (CONTRIBUTING.md) and
// every test of it picks a truncation point by hand. This picks them by
// walking only the bytes the validator still admits -- documents a constrained
// model could actually have reached -- stopping at a random one and requiring
// the force-closed result to be a valid instance of the schema it was
// generated under, by the validator AND by the parser. Deterministic: one
// fixed seed, so a failure is reproducible rather than a flake.
//
// THREE shapes are deliberately outside it, each a known defect recorded in
// the review rather than papered over in the machine: the walk never spells an
// exponent (one that overflows to infinity, or that leaves a bounded value out
// of range, is a live prefix with no valid completion), and no schema here is a
// homogeneous map (its synthesized closing key can collide with one the model
// already used).
static uint64_t walk_rnd(void) {
    static uint64_t s = UINT64_C(0x9E3779B97F4A7C15);
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

static void test_truncation_closes_to_a_valid_instance(void) {
    static const char *const schemas[] = {
        "{\"type\":\"object\",\"properties\":{"
            "\"a\":{\"type\":\"string\",\"maxLength\":4},"
            "\"b\":{\"type\":\"number\"},"
            "\"c\":{\"type\":\"array\",\"items\":{\"type\":\"integer\","
                   "\"minimum\":2,\"maximum\":77},\"minItems\":1,"
                   "\"maxItems\":3},"
            "\"d\":{\"enum\":[\"x\",\"yy\",\"yyy\"]},"
            "\"g\":{\"type\":\"object\"},"
            "\"h\":{\"type\":\"string\",\"pattern\":\"^id-[0-9]{2,4}$\"}},"
        "\"required\":[\"a\",\"b\",\"c\",\"d\",\"g\",\"h\"]}",
        "{\"type\":\"object\",\"properties\":{"
            "\"a\":{\"type\":\"string\",\"minLength\":3},"
            "\"o\":{\"type\":\"object\",\"properties\":{"
                "\"n\":{\"type\":\"integer\",\"minimum\":5}},"
                "\"required\":[\"n\"]}},"
        "\"required\":[\"a\"]}",
        "{\"type\":\"object\",\"properties\":{"
            "\"p\":{\"type\":\"string\",\"pattern\":\"^[A-Z]{3}[0-9]{4}$\"},"
            "\"u\":{\"type\":[\"string\",\"null\"]},"
            "\"q\":{\"type\":\"array\",\"items\":{\"type\":\"array\","
                   "\"items\":{\"type\":\"boolean\"},\"minItems\":2},"
                   "\"minItems\":2,\"maxItems\":2}},"
        "\"required\":[\"p\",\"u\",\"q\"]}",
    };
    int closed = 0;
    for (size_t s = 0; s < sizeof(schemas) / sizeof(*schemas); s++) {
        jv *schema_json = json_parse(schemas[s], strlen(schemas[s]));
        assert(schema_json != NULL);
        char err[192];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        if (!schema) fprintf(stderr, "walk schema %zu: %s\n", s, err);
        assert(schema != NULL);
        for (int iter = 0; iter < 200; iter++) {
            sval v; sval_init(&v, schema);
            char doc[1024];
            int n = 0;
            int steps = (int)(walk_rnd() % 70);
            for (int k = 0; k < steps && n < 600; k++) {
                // ASCII only: the validator is a byte machine and takes raw
                // UTF-8 bytes inside a string, json_parse checks them
                unsigned char legal[128];
                int nl = 0;
                for (int c = 1; c < 0x80; c++) {
                    if (c == 'e' || c == 'E') continue;   // see the note above
                    char b = (char)c;
                    sval t;
                    if (sval_trial(&v, &t, &b, 1)) legal[nl++] = (unsigned char)c;
                }
                if (!nl) break;
                char b = (char)legal[walk_rnd() % (unsigned)nl];
                assert(sval_feed(&v, &b, 1));   // a legal byte must be accepted
                doc[n++] = b;
                if (v.done) break;
            }
            if (v.done) continue;
            char tail[512];
            int tn = sval_close(&v, tail, sizeof(tail));
            if (tn <= 0) continue;              // nothing was started
            assert(n + tn < (int)sizeof(doc));
            memcpy(doc + n, tail, (size_t)tn);
            n += tn;
            doc[n] = 0;
            closed++;
            sval chk; sval_init(&chk, schema);
            if (!sval_feed(&chk, doc, n) || !chk.done)
                fprintf(stderr, "closed document not a valid instance: %s\n", doc);
            assert(sval_feed(&chk, "", 0) && chk.done);
            jv *parsed = json_parse(doc, (size_t)n);
            if (!parsed) fprintf(stderr, "closed document is not JSON: %s\n", doc);
            assert(parsed != NULL);
            jv_free(parsed);
        }
        schema_free(schema);
        jv_free(schema_json);
    }
    assert(closed > 100);   // the walk actually reached truncation points
}

// A keyword the compiler DOES implement, written with the wrong JSON type, is
// the same failure wearing a different hat: reinterpreting it silently
// enforces something other than what was declared, and the two below both
// used to compile.
static void test_schema_rejects_misspelled_keyword_types(void) {
    static const struct { const char *src; const char *want; } bad[] = {
        // an object-valued enum used to be read as a one-member list of its
        // VALUES, so this compiled to the literal `1`
        { "{\"enum\":{\"a\":1}}", "enum" },
        { "{\"enum\":\"a\"}", "enum" },
        // enum/const return before the ordinary typed compiler, so malformed
        // type keywords beside them must still be rejected here
        { "{\"type\":7,\"enum\":[1]}", "type" },
        { "{\"type\":true,\"const\":1}", "type" },
        { "{\"type\":[\"integer\",7],\"enum\":[1]}", "type" },
        { "{\"type\":[],\"enum\":[1]}", "type" },
        { "{\"type\":[\"integer\",\"mystery\"],\"enum\":[1]}",
          "unsupported type" },
        // JSON Schema nodes are objects or booleans. Explicit null is data,
        // not the absent-schema sentinel used internally for unconstrained
        // array items.
        { "null", "schema node" },
        { "{\"type\":\"object\",\"properties\":{\"x\":null}}",
          "schema node" },
        // and a non-object `properties` fell through to the open-object
        // machine, which accepts any object at all -- the one outcome this
        // compiler exists to prevent
        { "{\"type\":\"object\",\"properties\":[{\"a\":1}]}", "properties" },
        { "{\"type\":\"object\",\"properties\":\"a\"}", "properties" },
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *schema_json = json_parse(bad[i].src, strlen(bad[i].src));
        assert(schema_json != NULL);
        char err[128];
        err[0] = 0;
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, bad[i].want) != NULL);
        jv_free(schema_json);
    }
    // the well-formed spellings still compile
    static const char *const good[] = {
        "{\"enum\":[\"a\",\"b\"]}",
        "{\"const\":\"a\"}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}",
        "{\"type\":\"object\",\"properties\":{}}",
    };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        jv *schema_json = json_parse(good[i], strlen(good[i]));
        assert(schema_json != NULL);
        char err[128];
        err[0] = 0;
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        if (!schema) fprintf(stderr, "%s: %s\n", good[i], err);
        assert(schema != NULL);
        schema_free(schema);
        jv_free(schema_json);
    }
}

// Bounded repetition: {n} and {n,m}. The unbounded forms (`+`, `{n,}`) only
// ever needed a floor, so nothing enforced a ceiling; these do, and the ceiling
// has to be enforced during decoding rather than at the closing quote, or the
// model is free to emit an over-long tail and only discover it is invalid once
// it is too late to take back.
static void test_schema_bounded_repetition(void) {
    struct { const char *pat; const char *ok; const char *too_short;
             const char *too_long; } cases[] = {
        // exactly six
        { "^wf_[a-z0-9-]{6}$",   "\"wf_abc123\"", "\"wf_abc12\"", "\"wf_abc1234\"" },
        // two to four
        { "^id-[0-9]{2,4}$",     "\"id-123\"",    "\"id-1\"",     "\"id-12345\"" },
        // a degenerate but legal range: exactly one
        { "^v[0-9]{1,1}$",       "\"v7\"",        "\"v\"",        "\"v77\"" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[128];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}",
                 cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);

        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, cases[i].ok, (int)strlen(cases[i].ok)) && v.done);

        sval s; sval_init(&s, schema);
        assert(!sval_feed(&s, cases[i].too_short, (int)strlen(cases[i].too_short)));

        // the over-long case must be refused ON THE OFFENDING BYTE, not at the
        // closing quote -- that is the whole point of a decode-time ceiling
        sval l; sval_init(&l, schema);
        assert(!sval_feed(&l, cases[i].too_long, (int)strlen(cases[i].too_long)));

        // and a forced completion of a partial string must itself be valid
        sval p; sval_init(&p, schema);
        assert(sval_feed(&p, "\"", 1));
        char suffix[64];
        int nfix = sval_close(&p, suffix, sizeof(suffix));
        assert(nfix > 0);
        char full[128]; snprintf(full, sizeof(full), "\"%s", suffix);
        sval chk; sval_init(&chk, schema);
        assert(sval_feed(&chk, full, (int)strlen(full)) && chk.done);

        schema_free(schema); jv_free(j);
    }

    // an upper bound below the lower bound is an empty language: refuse it at
    // compile time rather than accept a schema no output can ever satisfy
    const char *empty[] = {
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{5,2}$\"}",
        // minLength that the bounded pattern can never reach
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{2}$\",\"minLength\":9}",
        // malformed quantifiers stay errors rather than being reinterpreted
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{}$\"}",
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{2,3,4}$\"}",
        "{\"type\":\"string\",\"pattern\":\"^x[a-z]{,3}$\"}",
    };
    for (size_t i = 0; i < sizeof(empty) / sizeof(*empty); i++) {
        jv *j = json_parse(empty[i], strlen(empty[i]));
        char err[128];
        err[0] = 0;
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema == NULL);
        assert(err[0] != 0);
        jv_free(j);
    }

    // the unbounded forms must keep behaving exactly as before
    const char *unbounded = "{\"type\":\"string\",\"pattern\":\"^n[0-9]{2,}$\"}";
    jv *j = json_parse(unbounded, strlen(unbounded));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *long_ok = "\"n0123456789012345\"";
    sval v; sval_init(&v, schema);
    assert(sval_feed(&v, long_ok, (int)strlen(long_ok)) && v.done);
    schema_free(schema); jv_free(j);
}

// A pattern with more than one repeated class was refused outright, so
// `^[A-Z]{3}[0-9]{4}$` -- an ordinary ticket/serial shape -- could not be
// enforced at all. Multiple segments are accepted when every segment before
// the last is FIXED-length: the position of each class is then determined by
// the byte offset alone, so the enforced language is still exactly the
// declared one. A variable-length segment in the middle stays refused,
// because two different segment splits could explain the same prefix and
// this compiler does not guess.
static void test_schema_multi_segment_pattern(void) {
    struct { const char *pat; const char *ok; const char *bad1;
             const char *bad2; } cases[] = {
        // two adjacent classes, the motivating case
        { "^[A-Z]{3}[0-9]{4}$", "\"ABC1234\"", "\"ABCD234\"", "\"ABC12345\"" },
        // a literal separator between them
        { "^[A-Z]{2}-[0-9]{3}$", "\"AB-123\"", "\"AB:123\"", "\"AB-12\"" },
        // shorthand classes, twice
        { "^\\\\d{3}-\\\\d{2}$", "\"123-45\"", "\"12a-45\"", "\"123-4\"" },
        // a trailing literal after the last class
        { "^[0-9]{2}X$", "\"12X\"", "\"12Y\"", "\"123X\"" },
        // only the LAST segment may be variable
        { "^[A-Z]{2}[0-9]{2,4}$", "\"AB1234\"", "\"AB1\"", "\"AB12345\"" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[160];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}",
                 cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);

        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, cases[i].ok, (int)strlen(cases[i].ok)) && v.done);

        sval a; sval_init(&a, schema);
        assert(!sval_feed(&a, cases[i].bad1, (int)strlen(cases[i].bad1)));
        sval b; sval_init(&b, schema);
        assert(!sval_feed(&b, cases[i].bad2, (int)strlen(cases[i].bad2)));

        // a forced close mid-string must still produce a legal document
        sval p2; sval_init(&p2, schema);
        assert(sval_feed(&p2, "\"", 1));
        char suffix[80];
        int nfix = sval_close(&p2, suffix, sizeof(suffix));
        assert(nfix > 0);
        char full[160]; snprintf(full, sizeof(full), "\"%s", suffix);
        sval chk; sval_init(&chk, schema);
        assert(sval_feed(&chk, full, (int)strlen(full)) && chk.done);

        schema_free(schema); jv_free(j);
    }

    // A variable-length segment before the last one is not a segment shape:
    // the segments would have to guess a split. R2.18: such a pattern goes to
    // the automaton, which enforces ECMA's language for it rather than refuse.
    struct { const char *src, *ok, *bad1, *bad2; } ambiguous[] = {
        { "{\"type\":\"string\",\"pattern\":\"^[A-Z]{1,3}[0-9]{2}$\"}", "\"AB12\"", "\"ABCD12\"", "\"AB1\"" },
        { "{\"type\":\"string\",\"pattern\":\"^[A-Z]+[0-9]{2}$\"}", "\"ABC12\"", "\"12\"", "\"ABC123\"" },
    };
    for (size_t i = 0; i < sizeof(ambiguous) / sizeof(*ambiguous); i++) {
        jv *j = json_parse(ambiguous[i].src, strlen(ambiguous[i].src));
        char err[256];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL && schema->re != NULL && schema->n_pats == 0);
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, ambiguous[i].ok, (int)strlen(ambiguous[i].ok)) && v.done);
        sval a; sval_init(&a, schema);
        assert(!(sval_feed(&a, ambiguous[i].bad1, (int)strlen(ambiguous[i].bad1)) && a.done));
        sval b; sval_init(&b, schema);
        assert(!(sval_feed(&b, ambiguous[i].bad2, (int)strlen(ambiguous[i].bad2)) && b.done));
        schema_free(schema); jv_free(j);
    }
}

static void test_schema_agent_id_pattern_is_enforced(void) {
    const char *src =
        "{\"type\":\"string\",\"pattern\":\"^wf_[a-z0-9-]{6,}$\"}";
    jv *j = json_parse(src, strlen(src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *good[] = { "\"wf_abc-19\"", "\"wf_123456\"" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])) && v.done);
    }
    const char *bad[] = { "\"xx_abc123\"", "\"wf_abc\"", "\"wf_ABC123\"" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v; sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])));
    }
    sval partial; sval_init(&partial, schema);
    assert(sval_feed(&partial, "\"wf_a", 5));
    char suffix[32];
    int n = sval_close(&partial, suffix, sizeof(suffix));
    assert(n > 0);
    char full[64]; snprintf(full, sizeof(full), "\"wf_a%s", suffix);
    sval check; sval_init(&check, schema);
    assert(sval_feed(&check, full, (int)strlen(full)) && check.done);
    schema_free(schema); jv_free(j);

    src = "{\"type\":\"string\","
          "\"pattern\":\"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$\"}";
    j = json_parse(src, strlen(src));
    schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval protocol; sval_init(&protocol, schema);
    assert(sval_feed(&protocol, "\"proto-1\"", 9) && protocol.done);
    schema_free(schema); jv_free(j);
}

static void test_schema_pattern_shorthand_classes(void) {
    // ^\d{5}$ — postal codes, order numbers, IDs — is the most common
    // fixed-length pattern there is, and it used to be refused over a
    // spelling: the compiler demanded a bracket set, so [0-9]{5} compiled and
    // \d{5} did not. \d and \w now expand to exactly the ASCII sets their
    // bracket equivalents would, in every position the class is legal.
    struct { const char *pat; const char *good; const char *bad; } cases[] = {
        { "^\\\\d{5}$",     "\"12345\"",      "\"1234a\""      },
        { "^\\\\d{5}$",     "\"00000\"",      "\"123456\""     },
        { "^\\\\w{3}$",     "\"a_9\"",        "\"a-9\""        },
        { "^\\\\w{3}$",     "\"ABC\"",        "\"AB\""         },
        { "^ORD-\\\\d{6}$", "\"ORD-100000\"", "\"XRD-100000\"" },
        { "^\\\\d{2,4}$",   "\"123\"",        "\"12345\""      },
        { "^\\\\d+$",       "\"7\"",          "\"x\""          },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[160];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}",
                 cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        assert(j != NULL);
        char err[128];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);
        sval good; sval_init(&good, schema);
        assert(sval_feed(&good, cases[i].good, (int)strlen(cases[i].good)) &&
               good.done);
        sval bad; sval_init(&bad, schema);
        // either rejected outright, or accepted-but-not-complete; both mean
        // the declared language is being enforced rather than waved through
        assert(!(sval_feed(&bad, cases[i].bad, (int)strlen(cases[i].bad)) &&
                 bad.done));
        schema_free(schema); jv_free(j);
    }
    // \s compiles to the six whitespace characters the regex means. Five of
    // them are control characters a JSON string only carries escaped, and
    // this validator admits no escape inside a pattern, so the space is the
    // one producible member: the enforced language is the declared one
    // restricted to what a JSON string spells literally, which is exactly
    // the restriction every class here has always carried (a `"` in a set
    // was never producible either). Refusing \s outright, which this did
    // until Claude Code 2.1.272 declared `[\s\S]`, left a claimed client
    // unable to make a single request.
    const char *ws = "{\"type\":\"string\",\"pattern\":\"^\\\\s{2}$\"}";
    jv *j = json_parse(ws, strlen(ws));
    assert(j != NULL);
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval two; sval_init(&two, schema);
    assert(sval_feed(&two, "\"  \"", 4) && two.done);
    sval one; sval_init(&one, schema);
    assert(!(sval_feed(&one, "\" a\"", 4) && one.done));
    schema_free(schema); jv_free(j);
}

static void test_schema_pattern_regex_syntax_is_rejected_not_reinterpreted(void) {
    // the segment compiler matches prefix and class LITERALLY, so regex syntax
    // it would reinterpret (escapes, negation, prefix metacharacters) is not
    // its shape. R2.18: those patterns go to the automaton, which reads them
    // as ECMA does (a dot is any code point but a line terminator, [^] is any
    // code point); an escape ECMA gives no meaning stays refused, by name.
    struct { const char *src, *ok, *bad; } reread[] = {
        { "{\"type\":\"string\",\"pattern\":\"^a.b[a-z]+$\"}", "\"a-bc\"", "\"abc\"" },       // a dot
        { "{\"type\":\"string\",\"pattern\":\"^a\\\\.b[a-z]+$\"}", "\"a.bc\"", "\"a-bc\"" }, // an escaped dot
        { "{\"type\":\"string\",\"pattern\":\"^x[^]+$\"}", "\"x-y\"", "\"x\"" },              // [^]: any
    };
    for (size_t i = 0; i < sizeof(reread) / sizeof(*reread); i++) {
        jv *j = json_parse(reread[i].src, strlen(reread[i].src));
        assert(j != NULL);
        char err[256];
        snode *schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL && schema->re != NULL);
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, reread[i].ok, (int)strlen(reread[i].ok)) && v.done);
        sval w; sval_init(&w, schema);
        assert(!(sval_feed(&w, reread[i].bad, (int)strlen(reread[i].bad)) && w.done));
        schema_free(schema); jv_free(j);
    }
    {
        const char *src = "{\"type\":\"string\",\"pattern\":\"^x[\\\\q]{2,}$\"}";
        jv *j = json_parse(src, strlen(src));
        char err[256];
        assert(schema_compile(j, err, sizeof(err)) == NULL);
        assert(strstr(err, "pattern") != NULL && strstr(err, "unknown letter escape") != NULL);
        jv_free(j);
    }
    // literal-safe specials INSIDE the class stay supported (regex treats
    // them as literals there, so literal matching agrees with the pattern)
    const char *ok_src =
        "{\"type\":\"string\",\"pattern\":\"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$\"}";
    jv *j = json_parse(ok_src, strlen(ok_src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    schema_free(schema); jv_free(j);
}

// Claude Code 2.1.272 declares SendMessage.to as
//   {"type":"string","allOf":[{"pattern":"^[^\n\r]*$"},{"pattern":"^[\s\S]{0,300}$"}]}
// and the runner answered 400 to every request, which ends the client. The
// conjunction of string constraints compiles to one string node that
// enforces every pattern (and the tightest length bounds), and the class
// syntax those patterns use -- negation, escapes inside a set, \s and \S --
// compiles to exactly the declared character sets, with a negated or
// universal class admitting non-ASCII characters whole. The enforced
// language is, as for every pattern here, the declared one restricted to
// the characters a JSON string carries unescaped.
static void test_schema_allof_of_string_patterns(void) {
    const char *src =
        "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[^\\\\n\\\\r]*$\"},"
        "{\"pattern\":\"^[\\\\s\\\\S]{0,300}$\"}]}";
    jv *j = json_parse(src, strlen(src));
    assert(j != NULL);
    char err[192];
    snode *schema = schema_compile(j, err, sizeof(err));
    if (!schema) fprintf(stderr, "allOf: %s\n", err);
    assert(schema != NULL);
    // an agent name, with a non-ASCII character, is admitted whole
    const char *ok = "\"builder-\xc3\xa5-7\"";
    sval v; sval_init(&v, schema);
    assert(sval_feed(&v, ok, (int)strlen(ok)) && v.done);
    // the empty string is in both languages
    sval e; sval_init(&e, schema);
    assert(sval_feed(&e, "\"\"", 2) && e.done);
    // the second pattern's ceiling holds: byte 301 is refused where it lands
    char longs[320]; longs[0] = '"';
    memset(longs + 1, 'a', 301); longs[302] = '"'; longs[303] = 0;
    sval l; sval_init(&l, schema);
    assert(!sval_feed(&l, longs, 303));
    memset(longs + 1, 'a', 300); longs[301] = '"'; longs[302] = 0;
    sval m; sval_init(&m, schema);
    assert(sval_feed(&m, longs, 302) && m.done);
    // a forced completion of the open string closes it validly
    sval c; sval_init(&c, schema);
    assert(sval_feed(&c, "\"ab", 3));
    char suffix[16];
    int nfix = sval_close(&c, suffix, sizeof(suffix));
    assert(nfix == 1 && suffix[0] == '"');
    schema_free(schema); jv_free(j);

    // two patterns whose conjunction is a fixed-length language: both hold
    const char *both =
        "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[a-z]{2,4}$\"},"
        "{\"pattern\":\"^[a-c]{3}$\"}]}";
    j = json_parse(both, strlen(both));
    schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval a; sval_init(&a, schema);
    assert(sval_feed(&a, "\"abc\"", 5) && a.done);
    sval b; sval_init(&b, schema);
    assert(!sval_feed(&b, "\"abd\"", 5));          // d is outside [a-c]
    sval d; sval_init(&d, schema);
    assert(!sval_feed(&d, "\"abcc\"", 6));         // {3} caps the second
    sval f; sval_init(&f, schema);
    assert(sval_feed(&f, "\"a", 2));
    nfix = sval_close(&f, suffix, sizeof(suffix));  // fills to the joint minimum
    assert(nfix == 3 && !memcmp(suffix, "aa\"", 3));
    schema_free(schema); jv_free(j);

    // allOf members that are not string constraints stay refused, as does
    // a member whose type contradicts the parent
    const char *bad[] = {
        "{\"type\":\"string\",\"allOf\":[{\"type\":\"integer\"}]}",
        "{\"type\":\"object\",\"allOf\":[{\"properties\":{\"a\":{\"type\":\"string\"}}}]}",
        "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[a-z]{2}$\"},{\"pattern\":\"^[0-9]{2}$\"}]}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        j = json_parse(bad[i], strlen(bad[i]));
        assert(schema_compile(j, err, sizeof(err)) == NULL);
        jv_free(j);
    }
}

// Negated classes and escapes inside a set: the enforced set is exactly the
// declared one. `[^\n\r]` admits every character but the two newlines; a
// control character never appears in a JSON string unescaped, so the class
// keeps the same producible members as the declared regex.
static void test_schema_negated_and_escaped_classes(void) {
    struct { const char *pat; const char *ok; const char *bad; } cases[] = {
        { "^[^\\\\n\\\\r]+$",     "\"one line\"",   "\"two\\nlines\"" },
        { "^[^a-c]{2}$",          "\"xy\"",         "\"ab\"" },
        { "^[\\\\-\\\\]a]{2}$",   "\"-]\"",         "\"ab\"" },
        { "^[\\\\s]{1}$",         "\" \"",          "\"a\"" },
        { "^[\\\\S]{3}$",         "\"a\xc3\xa9z\"", "\"a z\"" },
        { "^\\\\S{2}$",           "\"ab\"",         "\"a b\"" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[128];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}", cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        assert(j != NULL);
        char err[192];
        snode *schema = schema_compile(j, err, sizeof(err));
        if (!schema) fprintf(stderr, "%s: %s\n", cases[i].pat, err);
        assert(schema != NULL);
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, cases[i].ok, (int)strlen(cases[i].ok)) && v.done);
        sval w; sval_init(&w, schema);
        assert(!(sval_feed(&w, cases[i].bad, (int)strlen(cases[i].bad)) && w.done));
        // the forced completion never reaches for a control character or a
        // quote as the class's filler
        sval c; sval_init(&c, schema);
        assert(sval_feed(&c, "\"", 1));
        char suffix[32];
        int nfix = sval_close(&c, suffix, sizeof(suffix));
        assert(nfix >= 1);
        for (int k = 0; k + 1 < nfix; k++)
            assert(suffix[k] >= 0x20 && suffix[k] != '"' && suffix[k] != '\\');
        schema_free(schema); jv_free(j);
    }
}

// R2.20: an exclusive bound held as one ULP inside it was a subnormal at 0
// (nextafter(0, DBL_MAX)), and gcc's -ffast-math links crtfastmath, which
// sets flush-to-zero and denormals-are-zero at startup on x86: there the
// subnormal compared as 0 and `exclusiveMinimum: 0` accepted 0 (Linux and
// Windows builds of 1.19.0). The verdicts below must hold in both float
// modes; the flushed one is set here by hand (MXCSR FTZ|DAZ, FPCR.FZ), so
// a Mac or a clang build checks what a gcc x86 build runs in.
static void fp_flush_to_zero(bool on) {
#if defined(__x86_64__) || defined(__i386__)
    fpenv_set(on ? (fpenv_get() | 0x8040u) : (fpenv_get() & ~0x8040u));
#elif defined(__aarch64__)
    fpenv_set(on ? (fpenv_get() | (1ull << 24)) : (fpenv_get() & ~(1ull << 24)));
#else
    (void)on;
#endif
}
static bool fp_flushes(void) {
    volatile double tiny = 2.2250738585072014e-308, q = 4.0;
    volatile double h = tiny / q;   // a subnormal unless the mode flushes it
    return h == 0.0;
}
static bool bounds_doc_ok(const snode *s, const char *doc) {
    sval v; sval_init(&v, s);
    if (!sval_feed(&v, doc, (int)strlen(doc))) return false;
    if (!v.done) (void)sval_feed(&v, " ", 1);
    return v.done;
}
static void test_schema_exclusive_bounds_survive_flush_to_zero(void) {
    struct { const char *schema, *doc; bool ok; } cases[] = {
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "0", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "0.0", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "0e5", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "-0", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "0.5", true },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "1e-300", true },
        { "{\"type\":\"number\",\"exclusiveMaximum\":0}", "0", false },
        { "{\"type\":\"number\",\"exclusiveMaximum\":0}", "-0", false },   // -0 is 0
        { "{\"type\":\"number\",\"exclusiveMaximum\":0}", "-0.0", false },
        { "{\"type\":\"number\",\"exclusiveMaximum\":0}", "-0.5", true },
        // both keywords: the tighter one, at an equal value the exclusive one
        { "{\"type\":\"number\",\"minimum\":0,\"exclusiveMinimum\":0}", "0", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0,\"minimum\":0}", "0", false },
        { "{\"type\":\"number\",\"minimum\":0}", "0", true },
        { "{\"type\":\"number\",\"minimum\":10,\"exclusiveMinimum\":5}", "7", false },
        { "{\"type\":\"number\",\"minimum\":5,\"exclusiveMinimum\":10}", "10", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":5}", "5", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":5}", "5.000000000000001", true },
        { "{\"type\":\"number\",\"exclusiveMaximum\":-3,\"minimum\":-10}", "-3", false },
        { "{\"type\":\"number\",\"exclusiveMaximum\":-3,\"minimum\":-10}", "-3.5", true },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0,\"exclusiveMaximum\":1}", "1", false },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0,\"exclusiveMaximum\":1}", "0.25", true },
        // two exclusive bounds two doubles apart: the one double between them is in
        { "{\"type\":\"number\",\"exclusiveMinimum\":1,\"exclusiveMaximum\":1.0000000000000004}", "1.0000000000000002", true },
        { "{\"type\":\"number\",\"exclusiveMinimum\":1,\"exclusiveMaximum\":1.0000000000000004}", "1", false },
        // two exclusive bounds on both sides of zero: their places in the order
        // of doubles are further apart than an int64_t difference holds (the
        // sanitized fuzz seed number_bounds_across_zero.json is the live half)
        { "{\"type\":\"number\",\"exclusiveMinimum\":-2,\"exclusiveMaximum\":2}", "0", true },
        { "{\"type\":\"number\",\"exclusiveMinimum\":-2,\"exclusiveMaximum\":2}", "-1.5", true },
        { "{\"type\":\"number\",\"exclusiveMinimum\":-2,\"exclusiveMaximum\":2}", "2", false },
        // the widest bounds a schema may carry (a bound past the largest safe
        // integer is refused at compile)
        { "{\"type\":\"number\",\"exclusiveMinimum\":-9007199254740991,\"exclusiveMaximum\":9007199254740991}", "1e15", true },
        { "{\"type\":\"number\",\"exclusiveMinimum\":-9007199254740991,\"exclusiveMaximum\":9007199254740991}", "9007199254740991", false },
        // the two smallest numbers around zero: zero lies between them
        { "{\"type\":\"number\",\"exclusiveMinimum\":-2.2250738585072014e-308,\"exclusiveMaximum\":2.2250738585072014e-308}", "0", true },
        // a narrow interval next to zero: its normal values are in, zero is not
        { "{\"type\":\"number\",\"exclusiveMinimum\":0,\"maximum\":5e-308}", "3e-308", true },
        { "{\"type\":\"number\",\"exclusiveMinimum\":0,\"maximum\":5e-308}", "0", false },
        // inside an object, where a brace ends the number
        { "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\",\"exclusiveMinimum\":0}},\"required\":[\"x\"]}", "{\"x\":0}", false },
        { "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\",\"exclusiveMinimum\":0}},\"required\":[\"x\"]}", "{\"x\":0.5}", true },
    };
    // an exclusive bound that leaves nothing between is empty in both modes
    const char *empty[] = {
        "{\"type\":\"number\",\"minimum\":0,\"exclusiveMaximum\":0}",
        "{\"type\":\"number\",\"exclusiveMinimum\":2,\"maximum\":2}",
        // two exclusive bounds on adjacent doubles: no number lies strictly
        // between them. 1.19.0 refused this; the first R2.20 fix compiled it
        // and a forced close then wrote the excluded upper bound (a second
        // read, 2026-11-15)
        "{\"type\":\"number\",\"exclusiveMinimum\":1,\"exclusiveMaximum\":1.0000000000000002}",
        "{\"type\":\"number\",\"exclusiveMinimum\":-1.0000000000000002,\"exclusiveMaximum\":-1}",
        "{\"type\":\"number\",\"exclusiveMinimum\":3,\"exclusiveMaximum\":2}",
    };
    int flushed_runs = 0;
    for (int mode = 0; mode < 2; mode++) {
        fp_flush_to_zero(mode == 1);
        if (mode == 1) {
#if defined(__x86_64__) || defined(__i386__) || defined(__aarch64__)
            // live: the hand-set mode really flushes, or this half checks nothing
            assert(fp_flushes());
            flushed_runs++;
#endif
        }
        for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
            jv *j = json_parse(cases[i].schema, strlen(cases[i].schema));
            // a schema the reader refuses compiles as "anything" and would
            // pass every "true" row without a bound being read
            if (!j) fprintf(stderr, "R2.20: the reader refuses %s\n", cases[i].schema);
            assert(j != NULL);
            char err[256];
            snode *s = schema_compile(j, err, sizeof(err));
            assert(s != NULL);
            bool got = bounds_doc_ok(s, cases[i].doc);
            if (got != cases[i].ok) {
                fprintf(stderr, "R2.20 (flush-to-zero %s): %s accepts %s: %d, want %d\n",
                        mode ? "on" : "off", cases[i].schema, cases[i].doc, got, cases[i].ok);
                assert(got == cases[i].ok);
            }
            schema_free(s); jv_free(j);
        }
        for (size_t i = 0; i < sizeof(empty) / sizeof(*empty); i++) {
            jv *j = json_parse(empty[i], strlen(empty[i]));
            assert(j != NULL);
            char err[256];
            snode *s = schema_compile(j, err, sizeof(err));
            if (s) fprintf(stderr, "R2.20 (flush-to-zero %s): %s compiled\n", mode ? "on" : "off", empty[i]);
            assert(s == NULL && strstr(err, "empty number bounds"));
            jv_free(j);
        }
        // a forced close with no number byte written lands inside the bound,
        // as a spelling json_parse reads back
        const char *closes[] = {
            "{\"type\":\"number\",\"exclusiveMinimum\":0}",
            "{\"type\":\"number\",\"exclusiveMaximum\":0}",
            "{\"type\":\"number\",\"exclusiveMinimum\":5}",
            "{\"type\":\"number\",\"exclusiveMaximum\":-3}",
            // a narrow interval next to zero has normal, readable values
            // (3e-308): the close used to write the unreadable 4.94e-324, or
            // the excluded 0 where subnormals are flushed
            "{\"type\":\"number\",\"exclusiveMinimum\":0,\"maximum\":5e-308}",
            "{\"type\":\"number\",\"exclusiveMaximum\":0,\"minimum\":-5e-308}",
            "{\"type\":\"number\",\"exclusiveMinimum\":0,\"exclusiveMaximum\":5e-308}",
            "{\"type\":\"number\",\"exclusiveMinimum\":1,\"exclusiveMaximum\":1.0000000000000004}",
        };
        for (size_t i = 0; i < sizeof(closes) / sizeof(*closes); i++) {
            char src[256];
            snprintf(src, sizeof(src), "{\"type\":\"object\",\"properties\":{\"x\":%s},\"required\":[\"x\"]}", closes[i]);
            jv *j = json_parse(src, strlen(src));
            assert(j != NULL);
            char err[256];
            snode *s = schema_compile(j, err, sizeof(err));
            assert(s != NULL);
            sval v; sval_init(&v, s);
            assert(sval_feed(&v, "{\"x\":", 5));
            char suffix[128];
            int k = sval_close(&v, suffix, sizeof(suffix));
            assert(k > 0);
            char full[160];
            snprintf(full, sizeof(full), "{\"x\":%.*s", k, suffix);
            if (!bounds_doc_ok(s, full)) {
                fprintf(stderr, "R2.20 (flush-to-zero %s): %s closes to %s, out of bounds\n",
                        mode ? "on" : "off", closes[i], full);
                assert(0);
            }
            schema_free(s); jv_free(j);
        }
    }
    fp_flush_to_zero(false);
    printf("ok: exclusive number bounds hold with flush-to-zero off and on (%zu cases, %d flushed runs)\n",
           sizeof(cases) / sizeof(*cases), flushed_runs);
}

static void test_schema_number_bounds_reject_dead_minus_and_close_in_bounds(void) {
    // a minus under a positive minimum, or an exclusive 0, is a dead prefix:
    // no suffix can ever satisfy the bound, so the first byte must be refused
    // (the integer path already does) instead of stalling with every token
    // masked at the end. Under an inclusive 0 it is not: -0 is 0 (R2.19).
    const char *dead[] = { "{\"type\":\"number\",\"minimum\":2}",
                           "{\"type\":\"number\",\"exclusiveMinimum\":0}" };
    jv *j;
    char err[128];
    snode *schema;
    for (size_t i = 0; i < sizeof(dead) / sizeof(*dead); i++) {
        j = json_parse(dead[i], strlen(dead[i]));
        schema = schema_compile(j, err, sizeof(err));
        assert(schema != NULL);
        sval v; sval_init(&v, schema);
        assert(!sval_feed(&v, "-", 1));
        sval ok; sval_init(&ok, schema);
        assert(sval_feed(&ok, "3.5", 3));
        schema_free(schema); jv_free(j);
    }
    const char *src;

    // a bounded number filled in by force-close stays INSIDE its bounds —
    // both the untouched-value close path and the required-property
    // minimal-emission path
    src = "{\"type\":\"object\",\"properties\":{"
          "\"a\":{\"type\":\"number\",\"minimum\":5}},"
          "\"required\":[\"a\"]}";
    j = json_parse(src, strlen(src));
    schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *starts[] = { "{", "{\"a\":" };
    for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
        sval closing; sval_init(&closing, schema);
        assert(sval_feed(&closing, starts[i], (int)strlen(starts[i])));
        char suffix[64];
        int n = sval_close(&closing, suffix, sizeof(suffix));
        assert(n > 0);
        char full[128];
        snprintf(full, sizeof(full), "%s%s", starts[i], suffix);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv *a = jv_get(parsed, "a");
        assert(a != NULL && a->type == J_NUM && a->num >= 5.0);
        jv_free(parsed);
    }
    schema_free(schema); jv_free(j);
}


// R2.19: a bounded number's prefix is admitted only while some completion of
// it is a number inside the bounds. The byte that would leave none is refused
// where it is written (the exponent digit of 1e+36 under maximum 1, the first
// digit of a mantissa no scale can bring inside), never at the terminator
// after every token has been masked; and a forced close of any admitted
// prefix is a value inside the bounds.
static snode *num_schema(const char *src) {
    jv *j = json_parse(src, strlen(src));
    assert(j != NULL);
    char err[256];
    snode *n = schema_compile(j, err, sizeof(err));
    jv_free(j);
    assert(n != NULL);
    return n;
}
static bool num_admits(const snode *s, const char *prefix) {
    sval v; sval_init(&v, s);
    return sval_feed(&v, prefix, (int)strlen(prefix));
}
// a whole document: the validator says done, once a top-level number has its
// terminator (which the finished document then refuses)
static bool num_valid(const snode *s, const char *doc) {
    sval v; sval_init(&v, s);
    if (!sval_feed(&v, doc, (int)strlen(doc))) return false;
    if (!v.done) (void)sval_feed(&v, " ", 1);
    return v.done;
}
static bool num_closes_valid(const snode *s, const char *prefix) {
    sval v; sval_init(&v, s);
    if (!sval_feed(&v, prefix, (int)strlen(prefix))) return false;
    char suffix[256];
    int k = sval_close(&v, suffix, sizeof(suffix));
    if (k < 0) return false;
    char full[512];
    snprintf(full, sizeof(full), "%s%.*s", prefix, k, suffix);
    return num_valid(s, full);
}

static void test_schema_number_prefix_stays_completable(void) {
    struct { const char *schema, *prefix; bool admitted; } cases[] = {
        // R2.19's report: maximum 1 admitted 1e+36, then masked every token
        { "{\"type\":\"number\",\"maximum\":1}", "1e+36", false },
        { "{\"type\":\"number\",\"maximum\":1}", "1e+3", false },     // e+3 and longer are all past 1
        { "{\"type\":\"number\",\"maximum\":1}", "1e+", true },       // e+0 is 1
        { "{\"type\":\"number\",\"maximum\":1}", "1e+0", true },
        { "{\"type\":\"number\",\"maximum\":1}", "1e+00", true },
        { "{\"type\":\"number\",\"maximum\":1}", "1e-36", true },
        { "{\"type\":\"number\",\"maximum\":1}", "1e", true },
        // the corpus's shape: 2.74e+57 over a maximum of 10
        { "{\"type\":\"number\",\"maximum\":10}", "2.74e+5", false },
        { "{\"type\":\"number\",\"maximum\":10}", "2.74e+1", false },
        { "{\"type\":\"number\",\"maximum\":10}", "2.74e+0", true },
        // the lower bound, from below: 4e-50 under minimum 1e-9
        { "{\"type\":\"number\",\"minimum\":1e-9}", "4e-50", false },
        { "{\"type\":\"number\",\"minimum\":1e-9}", "4e-10", false },
        { "{\"type\":\"number\",\"minimum\":1e-9}", "4e-9", true },
        { "{\"type\":\"number\",\"minimum\":1e-9}", "4e-5", true },
        // negative numbers, both bounds
        { "{\"type\":\"number\",\"minimum\":-1}", "-1e+3", false },
        { "{\"type\":\"number\",\"minimum\":-1}", "-1e-3", true },
        { "{\"type\":\"number\",\"maximum\":-1}", "-1e-3", false },   // -0.001 is above -1
        { "{\"type\":\"number\",\"maximum\":-1}", "-1e-0", true },
        { "{\"type\":\"number\",\"maximum\":-1}", "-1e-", true },
        { "{\"type\":\"number\",\"maximum\":-1}", "5", false },       // no positive number is at most -1
        { "{\"type\":\"number\",\"maximum\":-1}", "0", false },
        { "{\"type\":\"number\",\"maximum\":-1}", "-5", true },
        // exclusive bounds
        { "{\"type\":\"number\",\"exclusiveMaximum\":1}", "1e+0", false },
        { "{\"type\":\"number\",\"exclusiveMaximum\":1}", "1e-", true },
        { "{\"type\":\"number\",\"exclusiveMaximum\":1}", "1", true },  // 1e-1 is below it
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "0", true },  // 0.5 is above it
        { "{\"type\":\"number\",\"exclusiveMinimum\":0}", "0e", false },// 0 times any scale is 0
        { "{\"type\":\"number\",\"exclusiveMaximum\":0}", "0", false },
        { "{\"type\":\"number\",\"exclusiveMaximum\":0}", "-", true },
        // a narrow interval: the mantissa's first digits decide, at any scale
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "1", false },
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "7", false },
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "5", true },
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "6", true },  // 6e-1 is 0.6
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "0.59", true },
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "0.61", false },
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "5e-2", false },
        { "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}", "55e-2", true },
        // zero
        { "{\"type\":\"number\",\"maximum\":0}", "0", true },
        { "{\"type\":\"number\",\"maximum\":0}", "0.0", true },
        { "{\"type\":\"number\",\"maximum\":0}", "0.1", false },
        { "{\"type\":\"number\",\"maximum\":0}", "1", false },
        { "{\"type\":\"number\",\"maximum\":0}", "-1", true },
        // inside an object, where the terminator is a brace
        { "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\",\"maximum\":1}},\"required\":[\"x\"]}", "{\"x\":1e+36", false },
        { "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\",\"maximum\":1}},\"required\":[\"x\"]}", "{\"x\":1e-36", true },
        // unbounded: only what json_parse cannot read is refused
        { "{\"type\":\"number\"}", "1e+999", false },
        { "{\"type\":\"number\"}", "1e+30", true },
        // an integer admits no exponent in this grammar at all
        { "{\"type\":\"integer\",\"maximum\":1000}", "1e3", false },
    };
    int admitted = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        snode *s = num_schema(cases[i].schema);
        bool got = num_admits(s, cases[i].prefix);
        if (got != cases[i].admitted) {
            fprintf(stderr, "R2.19: %s with prefix %s: admitted %d, want %d\n",
                    cases[i].schema, cases[i].prefix, got, cases[i].admitted);
            assert(got == cases[i].admitted);
        }
        // an admitted prefix closes to a number inside the bounds
        if (got) { admitted++; assert(num_closes_valid(s, cases[i].prefix)); }
        schema_free(s);
    }
    assert(admitted > 10);

    // A walk: random number bytes, each one a trial first, then a forced close.
    // Every prefix the validator admits must close to a valid document; the
    // walks must reach exponents, or the check says nothing about them.
    const char *walk_schemas[] = {
        "{\"type\":\"number\",\"maximum\":1}", "{\"type\":\"number\",\"minimum\":1e-9}",
        "{\"type\":\"number\",\"minimum\":-1,\"maximum\":-0.5}", "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}",
        "{\"type\":\"number\",\"exclusiveMinimum\":0,\"exclusiveMaximum\":1e-300}",
        "{\"type\":\"number\",\"minimum\":1e15}", "{\"type\":\"number\",\"maximum\":-1e15}",
        "{\"type\":\"number\",\"minimum\":123.456,\"maximum\":123.457}",
    };
    const char alphabet[] = "0123456789.eE+-";
    unsigned long long rs = 0x2545F4914F6CDD1Dull;
    int walks = 0, closed = 0, in_exp = 0;
    for (size_t w = 0; w < sizeof(walk_schemas) / sizeof(*walk_schemas); w++) {
        snode *s = num_schema(walk_schemas[w]);
        for (int r = 0; r < 300; r++) {
            sval v; sval_init(&v, s);
            char doc[96]; int n = 0;
            int len = 1 + (int)(rs % 12);
            for (int k = 0; k < len; k++) {
                bool moved = false;
                for (int t = 0; t < 40 && !moved; t++) {
                    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                    char c = alphabet[rs % (sizeof(alphabet) - 1)];
                    sval scratch;
                    if (sval_trial(&v, &scratch, &c, 1)) {
                        assert(sval_feed(&v, &c, 1));
                        doc[n++] = c; moved = true;
                    }
                }
                if (!moved) break;
            }
            doc[n] = 0;
            if (!n) continue;      // no first byte landed: nothing was admitted
            walks++;
            if (strchr(doc, 'e') || strchr(doc, 'E')) in_exp++;
            if (num_closes_valid(s, doc)) closed++;
            else { fprintf(stderr, "R2.19 walk: %s admitted %s, which closes invalid\n", walk_schemas[w], doc); assert(0); }
        }
        schema_free(s);
    }
    assert(closed == walks && in_exp > walks / 10);
    printf("ok: a bounded number's prefix stays completable (%zu cases, %d walks, %d through an exponent)\n",
           sizeof(cases) / sizeof(*cases), walks, in_exp);
}

// R2.19, fable's read: the room. The validator keeps at most 95 bytes of a
// number (number_put), so a completion must fit in what is left. A run of
// digits the wide-interval shortcut admitted to the end left no room for the
// exponent that brings it inside: every byte refused, the terminator out of
// bounds, every token masked. Each run below must stop exactly where the
// shortest completion stops fitting (never later, which stalls; never
// earlier, which is narrower), and every prefix it admits must close inside.
static void test_schema_number_prefix_fits_the_buffer(void) {
    struct { const char *schema, *lead; char digit; int refused_at; } runs[] = {
        // toward zero: 91 nines still take e-91 (95 bytes); 92 would need 96
        { "{\"type\":\"number\",\"maximum\":1}", "", '9', 92 },
        { "{\"type\":\"number\",\"minimum\":-1}", "-", '9', 92 },
        // toward the largest (bounds are JSON-safe, so 1e15 is about as far
        // as one goes): 0.111... takes e16, so 92 bytes are the most
        { "{\"type\":\"number\",\"minimum\":1e15}", "0.", '1', 93 },
        // zeros: 0. and 88 of them still take 1e104 (95 bytes); the 89th is
        // refused at byte 91 (the earlier witness's runaway was this fraction)
        { "{\"type\":\"number\",\"minimum\":1e15}", "0.", '0', 91 },
        // two decades: 91 nines take e-89 (99.9...)
        { "{\"type\":\"number\",\"minimum\":1,\"maximum\":100}", "", '9', 92 },
        // unbounded: a complete spelling fills the buffer, number_put stops it
        { "{\"type\":\"number\"}", "", '1', 96 },
    };
    for (size_t r = 0; r < sizeof(runs) / sizeof(*runs); r++) {
        snode *s = num_schema(runs[r].schema);
        char p[128];
        snprintf(p, sizeof(p), "%s", runs[r].lead);
        int len = (int)strlen(p), refused = 0;
        while (len < 120) {
            p[len] = runs[r].digit; p[len + 1] = 0;
            if (!num_admits(s, p)) { refused = len + 1; break; }
            len++;
            if (!num_closes_valid(s, p)) {
                fprintf(stderr, "R2.19 room: %s admitted a %d-byte prefix that closes invalid\n",
                        runs[r].schema, len);
                assert(0);
            }
        }
        if (refused != runs[r].refused_at) {
            fprintf(stderr, "R2.19 room: %s refused the run at byte %d, want %d\n",
                    runs[r].schema, refused, runs[r].refused_at);
            assert(refused == runs[r].refused_at);
        }
        schema_free(s);
    }
    // an incomplete spelling needs one more byte: a point or an exponent mark
    // at the 95th byte has no room for its digit, bounded or not
    const char *ends[] = { ".", "e", "E" };
    const char *open_schemas[] = { "{\"type\":\"number\"}", "{\"type\":\"number\",\"minimum\":0}" };
    for (size_t o = 0; o < sizeof(open_schemas) / sizeof(*open_schemas); o++) {
        snode *s = num_schema(open_schemas[o]);
        for (size_t e = 0; e < sizeof(ends) / sizeof(*ends); e++) {
            char p[128];
            memset(p, '1', 94);
            snprintf(p + 94, sizeof(p) - 94, "%s", ends[e]);
            if (num_admits(s, p)) {
                fprintf(stderr, "R2.19 room: %s admitted a 95-byte prefix ending in %s\n", open_schemas[o], ends[e]);
                assert(0);
            }
            snprintf(p + 93, sizeof(p) - 93, "%s", ends[e]);   // 94 bytes: its digit fits
            if (!num_admits(s, p) || !num_closes_valid(s, p)) {
                fprintf(stderr, "R2.19 room: %s: a 94-byte prefix ending in %s admitted %d, closes %d\n",
                        open_schemas[o], ends[e], num_admits(s, p), num_closes_valid(s, p));
                assert(0);
            }
        }
        schema_free(s);
    }
    printf("ok: a number's prefix leaves room for its completion (%zu runs to the buffer's end)\n",
           sizeof(runs) / sizeof(*runs));
}

// R2.19: every prefix this validator has decided wrong, in one table, each
// with the answer the exact decimal decision gives on every platform and in
// every float mode. Earlier witnesses built their candidates with doubles
// (pow and log10, then correctly rounded targets) and refused prefixes that
// had completions: on a bound under gcc's x86 build, past 16 significant
// digits, after a long fraction of zeros, and in codex's read of 4aac6431
// (the first five groups). An admitted prefix must also close inside.
static void test_schema_number_cases_it_got_wrong(void) {
#define NUMS(b) "{\"type\":\"number\"," b "}"
    struct { const char *schema, *prefix; bool admit; } cases[] = {
        // codex 1: 1 completes to 1.1, and 0 to 0.11e1
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "1", true },
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "0", true },
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "1.15", true },
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "12e-1", true },
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "2", false },
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "1.3", false },
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "13e-1", false },
        { NUMS("\"minimum\":1.1,\"maximum\":1.2"), "-", false },
        // codex 2: 4.9999999999999999e-1 reads as exactly 0.5
        { NUMS("\"minimum\":0.5,\"maximum\":0.5"), "4", true },
        { NUMS("\"minimum\":0.5,\"maximum\":0.5"), "4.99999999999999", true },
        { NUMS("\"minimum\":0.5,\"maximum\":0.5"), "4.9999999999999999", true },
        { NUMS("\"minimum\":0.5,\"maximum\":0.5"), "0.50000000000000005", true },
        { NUMS("\"minimum\":0.5,\"maximum\":0.5"), "0.5000000000000001", false },
        { NUMS("\"minimum\":0.5,\"maximum\":0.5"), "4.8", false },
        { NUMS("\"minimum\":0.5,\"maximum\":0.5"), "6", false },
        // codex 3: 0. and 92 zeros take a 1 in the 95th byte
        { NUMS("\"exclusiveMinimum\":0,\"maximum\":1"),
          "0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", true },
        { NUMS("\"exclusiveMinimum\":0,\"maximum\":1"),
          "0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", false },
        // codex 4: 49e-309 sits between the strict bounds in any float mode
        { NUMS("\"exclusiveMinimum\":4.81e-308,\"exclusiveMaximum\":4.99e-308"), "4", true },
        { NUMS("\"exclusiveMinimum\":4.81e-308,\"exclusiveMaximum\":4.99e-308"), "49e-309", true },
        { NUMS("\"exclusiveMinimum\":4.81e-308,\"exclusiveMaximum\":4.99e-308"), "4.8", true },
        { NUMS("\"exclusiveMinimum\":4.81e-308,\"exclusiveMaximum\":4.99e-308"), "5", false },
        { NUMS("\"exclusiveMinimum\":4.81e-308,\"exclusiveMaximum\":4.99e-308"), "4.81e-308", false },
        // codex 5: 0 closes to 0.5e-299 whatever a compiler does with sqrt
        { NUMS("\"exclusiveMinimum\":4e-300,\"exclusiveMaximum\":6e-300"), "0", true },
        { NUMS("\"exclusiveMinimum\":4e-300,\"exclusiveMaximum\":6e-300"), "0.5e-299", true },
        { NUMS("\"exclusiveMinimum\":4e-300,\"exclusiveMaximum\":6e-300"), "0.4", true },
        { NUMS("\"exclusiveMinimum\":4e-300,\"exclusiveMaximum\":6e-300"), "0.6e-299", false },
        // completions exactly on a bound (gcc's x86 build refused them)
        { NUMS("\"minimum\":1e15,\"maximum\":9e15"), "9", true },
        { NUMS("\"minimum\":1e15,\"maximum\":9e15"), "0.9", true },
        { NUMS("\"minimum\":1e15,\"maximum\":9e15"), "0.000009", true },
        { NUMS("\"minimum\":1e15,\"maximum\":9e15"), "1000000000000000000", true },
        { NUMS("\"minimum\":-9e15,\"maximum\":-1e15"), "-9", true },
        { NUMS("\"minimum\":-9e15,\"maximum\":-1e15"), "-0.9", true },
        { NUMS("\"maximum\":1e-300"), "1000000000", true },
        { NUMS("\"maximum\":1e-300"), "10000000000000000000000000", true },
        { NUMS("\"minimum\":0.5,\"maximum\":0.6"), "6", true },
        { NUMS("\"minimum\":0.5,\"maximum\":0.6"), "0.6", true },
        // a 17-digit significand (no double target spelled it)
        { NUMS("\"minimum\":1e15,\"maximum\":9e15"), "19999999999999997", true },
        // the smallest normal, not 1e-300, is where the doubles' range ends
        { NUMS("\"maximum\":1e-300"), "0.2", true },
        // a minus under an inclusive minimum of 0: -0 is 0 (the oracle's first find)
        { NUMS("\"minimum\":0"), "-", true },
        { NUMS("\"minimum\":0,\"maximum\":1"), "-0.0", true },
        { NUMS("\"exclusiveMinimum\":0"), "-", false },
        { NUMS("\"minimum\":1e-9"), "-", false },
        // R2.19's subnormal rule, both signs: a value in 0 < |x| < 2^-1022 is
        // refused on every platform, so only digits that reach 2^-1022 itself
        // (2.2250738585072013830...e-308) complete a point interval there
        { NUMS("\"minimum\":2.2250738585072014e-308,\"maximum\":2.2250738585072014e-308"), "2.2250738585072013", true },
        { NUMS("\"minimum\":2.2250738585072014e-308,\"maximum\":2.2250738585072014e-308"), "2.2250738585072012", false },
        { NUMS("\"minimum\":-2.2250738585072014e-308,\"maximum\":-2.2250738585072014e-308"), "-2.2250738585072013", true },
        { NUMS("\"minimum\":-2.2250738585072014e-308,\"maximum\":-2.2250738585072014e-308"), "-2.2250738585072012", false },
    };
    int admitted = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        snode *s = num_schema(cases[i].schema);
        bool got = num_admits(s, cases[i].prefix);
        if (got != cases[i].admit || (got && !num_closes_valid(s, cases[i].prefix))) {
            fprintf(stderr, "R2.19 case: %s with prefix %s: admitted %d (want %d), closes inside %d\n",
                    cases[i].schema, cases[i].prefix, got, cases[i].admit, got && num_closes_valid(s, cases[i].prefix));
            assert(0);
        }
        admitted += got;
        schema_free(s);
    }
    // the subnormal rule on whole documents, both signs: refused whether the
    // parser reports ERANGE (glibc), rounds up to 2^-1022 (macOS: 2.22...12e-308),
    // or a flushing mode would compare what it returns as 0
    struct { const char *schema, *doc; bool ok; } docs[] = {
        { NUMS("\"maximum\":1"), "1e-310", false },
        { NUMS("\"maximum\":1"), "4.9e-324", false },
        { NUMS("\"maximum\":1"), "2.2250738585072012e-308", false },
        { NUMS("\"maximum\":1"), "1e-400", false },
        { NUMS("\"maximum\":1"), "2.2250738585072014e-308", true },
        { NUMS("\"maximum\":1"), "0e-400", true },
        { NUMS("\"maximum\":1"), "0", true },
        { NUMS("\"minimum\":-1"), "-1e-310", false },
        { NUMS("\"minimum\":-1"), "-4.9e-324", false },
        { NUMS("\"minimum\":-1"), "-2.2250738585072012e-308", false },
        { NUMS("\"minimum\":-1"), "-1e-400", false },
        { NUMS("\"minimum\":-1"), "-2.2250738585072014e-308", true },
        { NUMS("\"minimum\":-1"), "-0", true },
    };
    for (size_t i = 0; i < sizeof(docs) / sizeof(*docs); i++) {
        snode *s = num_schema(docs[i].schema);
        if (num_valid(s, docs[i].doc) != docs[i].ok) {
            fprintf(stderr, "R2.19 subnormal rule: %s accepts %s: %d, want %d\n",
                    docs[i].schema, docs[i].doc, !docs[i].ok, docs[i].ok);
            assert(0);
        }
        schema_free(s);
    }
#undef NUMS
    printf("ok: every number prefix this validator decided wrong is decided exactly "
           "(%zu prefixes, %d admitted; %zu documents for the subnormal rule)\n",
           sizeof(cases) / sizeof(*cases), admitted, sizeof(docs) / sizeof(*docs));
}

// R2.19: the decision's work per byte is bounded, as WORK, so the check
// cannot flake on a loaded host. An earlier witness ran away here (a long
// fraction of zeros under a large minimum: one trial over a minute) and the
// next one tried up to 110 candidate spellings a byte. The unit now is one
// digit pair compared or one spelling-family member or exponent width tried
// (schema_witness_work); every loop is bounded by the 95 bytes number_put
// keeps. The bound, from the code: the window's two edge scales from two
// digit comparisons (2 x 95), the inner scales from at most two families of
// 96 members (192), and at each edge scale those members again (192) plus
// three 95-digit comparisons (285): 190 + 192 + 2 x 477 = 1336. Inside an
// exponent it is 2 x 95 + 2 signs x 10 widths. The bounds' exact expansions
// (up to 768 digits) are built once per bounds, which the count of them
// shows: two per schema and the smallest normal once.
static void test_schema_number_witness_work_is_bounded(void) {
    enum { BOUND = 2 * 95 + 2 * 96 + 2 * (2 * 96 + 3 * 95) };
    const char *schemas[] = {
        "{\"type\":\"number\",\"minimum\":1e15}",
        "{\"type\":\"number\",\"exclusiveMinimum\":1e15}",
        "{\"type\":\"number\",\"maximum\":-1e15}",
        "{\"type\":\"number\",\"exclusiveMaximum\":-1e15}",
        "{\"type\":\"number\",\"minimum\":1e15,\"maximum\":9e15}",
        "{\"type\":\"number\",\"minimum\":-9e15,\"maximum\":-1e15}",
        "{\"type\":\"number\",\"maximum\":1e-300}",
        "{\"type\":\"number\",\"minimum\":0.5,\"maximum\":0.6}",
    };
    enum { NS = sizeof(schemas) / sizeof(*schemas) };
    const char *leads[] = { "0.", "-0.", "0.000", "1", "9", "-9" };
    const char fills[] = "019", alpha[] = "0123456789.eE+-";
    unsigned long worst = 0, expansions0 = schema_number_expansions();
    long trials = 0;
    // every trial's verdict, folded in order: the admitted set over these seeds.
    // The constant below is the same on macOS, Linux and Windows, so the three
    // platforms' CI jobs hold each other (a replay on another platform needs
    // the same masks); gcc's x86 build once admitted 9,525 fewer bytes here.
    uint64_t admitted_hash = 0xcbf29ce484222325ull;
    char worst_at[128] = "";
    clock_t t0 = clock();
    for (size_t si = 0; si < NS; si++) {
        snode *s = num_schema(schemas[si]);
        for (size_t li = 0; li < sizeof(leads) / sizeof(*leads); li++)
            for (const char *f = fills; *f; f++) {
                sval v; sval_init(&v, s);
                char p[128];
                int n = (int)strlen(leads[li]);
                if (!sval_feed(&v, leads[li], n)) continue;
                memcpy(p, leads[li], (size_t)n);
                for (; n < 95; n++) {
                    for (const char *b = alpha; *b; b++) {
                        sval t;
                        bool ok = sval_trial(&v, &t, b, 1);
                        admitted_hash = (admitted_hash ^ (ok ? 0x31u : 0x30u)) * 0x100000001b3ull;
                        unsigned long w = schema_witness_work();
                        trials++;
                        if (w > worst) {
                            worst = w;
                            snprintf(worst_at, sizeof(worst_at), "%s: %.*s%c (%d bytes)",
                                     schemas[si], n, p, *b, n + 1);
                        }
                    }
                    if (!sval_feed(&v, f, 1)) break;
                    p[n] = *f;
                }
            }
        schema_free(s);
    }
    double cpu = (double)(clock() - t0) / CLOCKS_PER_SEC;
    unsigned long expansions = schema_number_expansions() - expansions0;
    if (worst > BOUND || expansions > 2 * NS + 1) {
        fprintf(stderr, "R2.19 decision work: %lu units in one trial (bound %d), at %s; "
                "%lu bound expansions for %d schemas\n", worst, BOUND, worst_at, expansions, (int)NS);
        assert(worst <= BOUND && expansions <= 2 * NS + 1);
    }
    const long TRIALS = 98025;
    const uint64_t ADMITTED = 0x571622e18b25a814ull;
    if (trials != TRIALS || admitted_hash != ADMITTED) {
        fprintf(stderr, "R2.19 admitted set: %ld trials, hash %016llx; every platform must give "
                "%ld and %016llx\n", trials, (unsigned long long)admitted_hash,
                TRIALS, (unsigned long long)ADMITTED);
        assert(0);
    }
    printf("ok: the number decision's work is bounded (%ld trials, at most %lu units a byte of %d, "
           "at %s; %lu expansions; admitted set %016llx; note: %.2f s of CPU, %.2f us a trial here)\n",
           trials, worst, BOUND, worst_at, expansions, (unsigned long long)admitted_hash, cpu,
           trials ? cpu * 1e6 / (double)trials : 0.0);
}

static void test_schema_merges_enum_and_const_anyof(void) {
    const char *src = "{\"anyOf\":["
        "{\"type\":\"string\",\"enum\":[\"pending\",\"completed\"]},"
        "{\"type\":\"string\",\"const\":\"deleted\"}]}";
    jv *j = json_parse(src, strlen(src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    const char *good[] = { "\"pending\"", "\"completed\"", "\"deleted\"" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])) && v.done);
    }
    schema_free(schema); jv_free(j);
}

// json_parse keeps an integer literal of more than fifteen digits as its text
// (a double holds every fifteen-digit integer, not every sixteen-digit one),
// and every writer must hand it back digit for digit. 2^53 + 1 =
// 9007199254740993 is the first integer a double cannot hold; it was written
// back as ...992 (2026-10-16, Clef's score legend), which also put the wrong
// number into rendered tool schemas and grammar literals. The anchor is the
// literal itself: each must come back unchanged, in both spellings.
static void test_dump_keeps_integer_literals(void) {
    const char *src = "[9007199254740993,-9007199254740993,12345678901234567890,"
                      "{\"n\":[9007199254740993]},123456789012345,-0,7]";
    jv *j = json_parse(src, strlen(src));
    assert(j);
    sbuf a = { 0 }, b = { 0 };
    jv_dump(j, &a);
    jv_dump_tojson(j, &b);
    assert(a.s && b.s);
    if (strcmp(a.s, "[9007199254740993,-9007199254740993,12345678901234567890,"
                     "{\"n\":[9007199254740993]},123456789012345,0,7]")) {
        fprintf(stderr, "jv_dump changed an integer: %s\n", a.s);
        abort();
    }
    if (strcmp(b.s, "[9007199254740993, -9007199254740993, 12345678901234567890, "
                     "{\"n\": [9007199254740993]}, 123456789012345, 0, 7]")) {
        fprintf(stderr, "jv_dump_tojson changed an integer: %s\n", b.s);
        abort();
    }
    free(a.s); free(b.s); jv_free(j);
}

// An enum literal is compiled from its dump, so it constrains output to the
// caller's integer, not to the nearest double.
static void test_schema_enum_keeps_a_large_integer(void) {
    const char *src = "{\"enum\":[9007199254740993]}";
    jv *j = json_parse(src, strlen(src));
    char err[128];
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval v; sval_init(&v, schema);
    assert(sval_feed(&v, "9007199254740993", 16) && v.done);
    sval w; sval_init(&w, schema);
    assert(!(sval_feed(&w, "9007199254740992", 16) && w.done));
    schema_free(schema); jv_free(j);
}

// The anchor is the JSON Schema spec (2020-12, 10.2.1.3): an instance is
// valid against oneOf when it is valid against EXACTLY one alternative, and
// against anyOf when it is valid against at least one. Literal alternatives
// were merged as a union for both, so oneOf [{enum:[1,2]},{enum:[2,3]}]
// accepted 2, which both branches admit, and two identical const branches
// compiled although no value can satisfy them (2026-10-16 review).
static bool schema_accepts(const char *src, const char *val) {
    jv *j = json_parse(src, strlen(src));
    char err[160];
    snode *s = schema_compile(j, err, sizeof(err));
    assert(s != NULL);
    sval v; sval_init(&v, s);
    bool ok = sval_feed(&v, val, (int)strlen(val)) && v.done;
    schema_free(s); jv_free(j);
    return ok;
}

static bool schema_refused(const char *src) {
    jv *j = json_parse(src, strlen(src));
    char err[160];
    snode *s = schema_compile(j, err, sizeof(err));
    schema_free(s); jv_free(j);
    return s == NULL;
}

static void test_schema_one_of_admits_exactly_one_branch(void) {
    const char *one = "{\"oneOf\":[{\"enum\":[1,2]},{\"enum\":[2,3]}]}";
    assert(schema_accepts(one, "1") && schema_accepts(one, "3"));
    assert(!schema_accepts(one, "2"));
    const char *any = "{\"anyOf\":[{\"enum\":[1,2]},{\"enum\":[2,3]}]}";
    assert(schema_accepts(any, "1") && schema_accepts(any, "2") && schema_accepts(any, "3"));
    // a value listed twice in ONE branch still matches one branch
    const char *dup = "{\"oneOf\":[{\"enum\":[1,1]},{\"const\":2}]}";
    assert(schema_accepts(dup, "1") && schema_accepts(dup, "2"));
    // equal values spelled differently are one value
    const char *spell = "{\"oneOf\":[{\"const\":1},{\"enum\":[1.0,2]}]}";
    assert(!schema_accepts(spell, "1") && schema_accepts(spell, "2"));
    // no value matches exactly one branch: refused, not weakened
    assert(schema_refused("{\"oneOf\":[{\"const\":1},{\"const\":1}]}"));
    assert(schema_refused("{\"oneOf\":[{\"enum\":[1,2]},{\"enum\":[2,1]}]}"));
    assert(!schema_refused("{\"anyOf\":[{\"const\":1},{\"const\":1}]}"));
}

// const beside enum keeps only the members equal to the const, and equal
// means the same value: 2^53 and 2^53 + 1 are one double and two integers.
// The small-integer twin was refused; the large one emitted 2^53 + 1 under
// const 2^53 (2026-10-16 review).
static void test_schema_const_and_enum_compare_large_integers_exactly(void) {
    assert(schema_refused("{\"const\":9007199254740992,\"enum\":[9007199254740993]}"));
    assert(schema_refused("{\"const\":2,\"enum\":[3]}"));
    const char *src = "{\"const\":9007199254740993,\"enum\":[9007199254740992,9007199254740993]}";
    assert(schema_accepts(src, "9007199254740993"));
    assert(!schema_accepts(src, "9007199254740992"));
}

// Equal means the same number, whatever the spelling (JSON Schema 2020-12,
// 4.2.2: numbers are equal when their mathematical values are). 1e20 and
// 100000000000000000000 are one number; 9.007199254740993e15, read as a
// double, is 2^53 and not 2^53 + 1. Python agrees on each: 1e20 == 10**20,
// 9.007199254740993e15 != 2**53 + 1 (its int/float comparison is exact).
// Comparing spellings refused the first const/enum pair and let the two
// spellings through as separate oneOf branches (2026-10-16 review).
static void test_schema_numbers_compare_by_value_not_spelling(void) {
    assert(schema_accepts("{\"const\":1e20,\"enum\":[100000000000000000000,7]}",
                          "100000000000000000000"));
    assert(!schema_accepts("{\"const\":1e20,\"enum\":[100000000000000000000,7]}", "7"));
    assert(schema_accepts("{\"const\":18446744073709551616,\"enum\":[1.8446744073709552e19]}",
                          "1.8446744073709552e+19"));
    assert(schema_refused("{\"const\":1e20,\"enum\":[100000000000000000001]}"));
    assert(schema_refused("{\"const\":9.007199254740993e15,\"enum\":[9007199254740993]}"));
    assert(schema_accepts("{\"const\":9.007199254740993e15,\"enum\":[9007199254740992]}",
                          "9007199254740992"));
    // one value in two branches is no oneOf answer, in either spelling
    const char *two = "{\"oneOf\":[{\"const\":1e20},{\"const\":100000000000000000000},{\"const\":7}]}";
    assert(schema_accepts(two, "7"));
    assert(!schema_accepts(two, "100000000000000000000") && !schema_accepts(two, "1e+20"));
    assert(schema_refused("{\"oneOf\":[{\"const\":1e20},{\"const\":100000000000000000000}]}"));
}

// A whole double past 2^63 is written in exponent form, and its value is an
// integer only exact digits name. The expansion is checked against Python's
// int(float), which is exact: each const, spelled as the double, must match
// its exact digits and refuse the next integer. The rows run from 2^63, the
// first double the %lld path cannot take, to 1.6e308 near the parser's top.
static void test_schema_large_doubles_compare_by_exact_digits(void) {
    static const struct { const char *dbl, *exact, *next; } rows[] = {
        { "9.223372036854776e+18", "9223372036854775808", "9223372036854775809" },
        { "1.8446744073709552e+19", "18446744073709551616", "18446744073709551617" },
        { "2.7670116110564327e+19", "27670116110564327424", "27670116110564327425" },
        { "1e+20", "100000000000000000000", "100000000000000000001" },
        { "1e+23", "99999999999999991611392", "99999999999999991611393" },
        { "-1.1805916207174113e+21", "-1180591620717411303424", "-1180591620717411303425" },
        { "-1e+300", "-1000000000000000052504760255204420248704468581108159154915854115511802457988908195786371375080447864043704443832883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119038967640880074652742780142494579258788820056842838115669472196386865459400540160", "-1000000000000000052504760255204420248704468581108159154915854115511802457988908195786371375080447864043704443832883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119038967640880074652742780142494579258788820056842838115669472196386865459400540161" },
        { "1.6e+308", "159999999999999997764969561641033243472042269410021822003994571774137053642763901585037542627663370668582068486282473846293650765604033653380415237929846352171691066635030002857104637617170341434265972257138027520176322712852864297543325003016991155218743815378766749154225070877620513952042826684000337133568", "159999999999999997764969561641033243472042269410021822003994571774137053642763901585037542627663370668582068486282473846293650765604033653380415237929846352171691066635030002857104637617170341434265972257138027520176322712852864297543325003016991155218743815378766749154225070877620513952042826684000337133569" },
    };
    for (size_t i = 0; i < sizeof rows / sizeof *rows; i++) {
        char src[1024];
        snprintf(src, sizeof src, "{\"const\":%s,\"enum\":[%s]}", rows[i].dbl, rows[i].exact);
        if (schema_refused(src)) {
            fprintf(stderr, "%s is not %s\n", rows[i].dbl, rows[i].exact);
            abort();
        }
        snprintf(src, sizeof src, "{\"const\":%s,\"enum\":[%s]}", rows[i].dbl, rows[i].next);
        if (!schema_refused(src)) {
            fprintf(stderr, "%s matched %s\n", rows[i].dbl, rows[i].next);
            abort();
        }
    }
}

static void test_schema_integer_bounds_are_enforced(void) {
    const char *src =
        "{\"type\":\"integer\",\"minimum\":0,"
        "\"exclusiveMinimum\":3,\"maximum\":55}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *good[] = { "4", "10", "55" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v;
        sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])));
        char out[32];
        int n = sval_close(&v, out, sizeof(out));
        assert(n == 0);
    }

    const char *bad[] = { "-1", "0", "3", "56", "999" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v;
        sval_init(&v, schema);
        bool prefix_ok = sval_feed(&v, bad[i], (int)strlen(bad[i]));
        // A delimiter is the observable attempt to finish the out-of-range
        // number; an impossible prefix may be refused even earlier.
        assert(!prefix_ok || !sval_feed(&v, " ", 1));
    }

    schema_free(schema);
    jv_free(schema_json);
}

// RFC 8259 has no leading zeros, and the constrained document is the ONE place
// that guarantee is supposed to be free. It was not: the integer machine
// re-derived "the number ended" from the byte instead of taking num_byte's
// verdict, so a digit after `0` was swallowed silently. Two failures ride on
// the one bug -- the document goes out as JSON nothing can parse (the runner's
// own reader refuses it, so a tool call's arguments are unreadable), and the
// running value stays 0, so every bound is satisfied by a number the model
// never wrote.
static void test_schema_integer_rejects_leading_zeros(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"n\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":100}"
        "},\"required\":[\"n\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *bad[] = { "{\"n\":01}", "{\"n\":007}", "{\"n\":0999999}",
                          "{\"n\":-01}" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v;
        sval_init(&v, schema);
        bool fed = sval_feed(&v, bad[i], (int)strlen(bad[i]));
        assert(!fed || !v.done);
        // and whatever WAS accepted must still parse as JSON, or the client
        // receives a body no reader can open
        if (fed) {
            jv *parsed = json_parse(bad[i], strlen(bad[i]));
            assert(parsed == NULL);
            jv_free(parsed);
        }
    }

    // the single zero itself is a legal integer and must still pass
    const char *good = "{\"n\":0}";
    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, good, (int)strlen(good)) && v.done);

    schema_free(schema);
    jv_free(schema_json);
}

// `minimum` and `exclusiveMinimum` in one schema mean BOTH. The real-valued
// arm replaced the first with the second instead of intersecting them, so the
// looser of the two won and a request that asked for two floors got the lower
// one. compile_integer_bounds has always taken the max; this pins the same
// rule for `"type":"number"`.
static void test_schema_number_bounds_intersect_their_exclusive_twins(void) {
    const char *cases[][2] = {
        { "{\"type\":\"number\",\"minimum\":10,\"exclusiveMinimum\":5}", "7" },
        { "{\"type\":\"number\",\"exclusiveMinimum\":10,\"minimum\":5}", "7" },
        { "{\"type\":\"number\",\"maximum\":5,\"exclusiveMaximum\":100}", "50" },
        { "{\"type\":\"number\",\"exclusiveMaximum\":5,\"maximum\":100}", "50" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        jv *schema_json = json_parse(cases[i][0], strlen(cases[i][0]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema != NULL);
        sval v;
        sval_init(&v, schema);
        const char *bad = cases[i][1];
        bool prefix_ok = sval_feed(&v, bad, (int)strlen(bad));
        // the delimiter is the observable attempt to finish an out-of-range
        // number; an impossible prefix may be refused earlier
        assert(!prefix_ok || !sval_feed(&v, " ", 1));
        schema_free(schema);
        jv_free(schema_json);
    }
}

// The one thing sval_close owes its caller is a document that PARSES. The
// minItems / minLength fills are the only unbounded emitters in it, both
// bounds are the request's to choose up to INT_MAX, and they used to eat the
// buffer to its last byte -- after which the structural closers behind them
// were silently dropped and the turn went out as JSON no reader accepts.
static void test_schema_close_never_drops_its_closers(void) {
    const char *cases[][2] = {
        { "{\"type\":\"array\",\"items\":{\"type\":\"integer\"},"
          "\"minItems\":5000}", "[" },
        { "{\"type\":\"string\",\"minLength\":9000}", "\"ab" },
        { "{\"type\":\"object\",\"properties\":{"
          "\"a\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"},"
          "\"minItems\":5000}},\"required\":[\"a\"]}", "{\"a\":[" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        jv *schema_json = json_parse(cases[i][0], strlen(cases[i][0]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema != NULL);
        sval v;
        sval_init(&v, schema);
        const char *partial = cases[i][1];
        assert(sval_feed(&v, partial, (int)strlen(partial)));
        // the engine's own closer buffer (engine.c), so the bound under test
        // is the one production actually runs with
        char suffix[4096];
        int n = sval_close(&v, suffix, sizeof(suffix));
        assert(n > 0);
        char *full = malloc(strlen(partial) + (size_t)n + 1);
        assert(full != NULL);
        memcpy(full, partial, strlen(partial));
        memcpy(full + strlen(partial), suffix, (size_t)n + 1);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv_free(parsed);
        free(full);
        schema_free(schema);
        jv_free(schema_json);
    }
}

// A string closed mid-escape gains a character the length accounting never
// saw: feed_byte advances lit_pos only once an escape closes, so the minLength
// padding believed the string was one short and added one filler too many --
// straight past maxLength, producing a document the closer's own grammar
// rejects.
static void test_schema_close_counts_the_escape_it_finishes(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"s\":{\"type\":\"string\",\"minLength\":3,\"maxLength\":3}"
        "},\"required\":[\"s\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *partials[] = { "{\"s\":\"ab\\", "{\"s\":\"ab\\u00" };
    for (size_t i = 0; i < sizeof(partials) / sizeof(*partials); i++) {
        sval v;
        sval_init(&v, schema);
        assert(sval_feed(&v, partials[i], (int)strlen(partials[i])));
        char suffix[128];
        int n = sval_close(&v, suffix, sizeof(suffix));
        assert(n > 0);
        char full[256];
        snprintf(full, sizeof(full), "%s%s", partials[i], suffix);
        // the closed document must be an instance of the grammar that closed
        // it, or the caller receives output its own constraint refuses
        sval chk;
        sval_init(&chk, schema);
        assert(sval_feed(&chk, full, (int)strlen(full)));
        assert(chk.done);
    }

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_integer_bounds_complete_truncation(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"line\":{\"type\":\"integer\",\"minimum\":50,\"maximum\":55}"
        "},\"required\":[\"line\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    const char *partial = "{\"line\":5";
    assert(sval_feed(&v, partial, (int)strlen(partial)));
    char suffix[64];
    int n = sval_close(&v, suffix, sizeof(suffix));
    assert(n > 0);
    char full[128];
    snprintf(full, sizeof(full), "%s%s", partial, suffix);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    jv *line = jv_get(parsed, "line");
    assert(line && line->type == J_NUM && line->num >= 50 && line->num <= 55);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

// A bounded number the model has ALREADY started out of range. Nothing can
// take those bytes back, but JSON's exponent can still finish them inside the
// interval -- and the alternative is a force-closed document that parses and
// then violates the schema it was generated under, which is the one outcome a
// caller cannot detect. Found by a legal-byte random walk against sval_close.
static void test_schema_number_close_rescues_an_out_of_range_prefix(void) {
    const char *src = "{\"type\":\"object\",\"properties\":{"
        "\"b\":{\"type\":\"number\",\"minimum\":-3,\"maximum\":9}},"
        "\"required\":[\"b\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    // every one of these is a prefix the validator accepts (an exponent could
    // still rescue it), so every one of them can be where the budget runs out
    const char *starts[] = {
        "{\"b\":843584921",     // complete spelling, far too large
        "{\"b\":8435.",         // fraction opened, not yet a value
        "{\"b\":-40000",        // far too small
        "{\"b\":7",             // already in bounds: must not be touched
    };
    for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, starts[i], (int)strlen(starts[i])));
        char suffix[128];
        int n = sval_close(&v, suffix, sizeof(suffix));
        assert(n > 0);
        char full[256];
        snprintf(full, sizeof(full), "%s%s", starts[i], suffix);
        // the closed document is a valid instance of the schema it was
        // generated under -- both by the validator and by the parser
        sval chk; sval_init(&chk, schema);
        if (!sval_feed(&chk, full, (int)strlen(full)) || !chk.done)
            fprintf(stderr, "closed document not accepted: %s\n", full);
        assert(sval_feed(&chk, "", 0) && chk.done);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv *b = jv_get(parsed, "b");
        assert(b != NULL && b->type == J_NUM);
        assert(b->num >= -3.0 && b->num <= 9.0);
        jv_free(parsed);
    }
    // an in-bounds prefix is completed with nothing at all
    sval keep; sval_init(&keep, schema);
    assert(sval_feed(&keep, "{\"b\":7", 6));
    char suffix[64];
    assert(sval_close(&keep, suffix, sizeof(suffix)) == 1 &&
           !strcmp(suffix, "}"));

    schema_free(schema);
    jv_free(schema_json);
}

// The generic machine (json mode, and every free-object subtree in a tool
// schema) must refuse the string bytes json_parse refuses: it took any byte
// >= 0x20 as string content, so a lone continuation byte, an overlong lead
// or an 0xF5.. lead sailed through into a document the runner's own parser
// (and its tool-argument readback) then rejected. The schema string machine
// has validated sequences since the UTF-8 length-accounting fix; this pins
// the shared rule on the jsonv side, closing quote and closer included.
static void test_jsonv_utf8_matches_parser(void) {
    struct { const char *pre; unsigned char byte; bool ok; } cases[] = {
        { "{\"k",        0x8B, false },  // lone continuation in a key
        { "{\"k\":\"v",  0x8B, false },  // ...and in a value
        { "{\"k\":\"v",  0xC0, false },  // overlong lead
        { "{\"k\":\"v",  0xF5, false },  // beyond U+10FFFF
        { "{\"k\":\"v",  0xC3, true  },  // valid lead...
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        jsonv v; jsonv_init(&v);
        assert(jsonv_feed(&v, cases[i].pre, (int)strlen(cases[i].pre)));
        char b = (char)cases[i].byte;
        assert(jsonv_feed(&v, &b, 1) == cases[i].ok);
    }
    // a valid lead's continuation window: quote and ASCII are refused inside
    // the scalar, the proper continuation is accepted, and the whole
    // document round-trips through json_parse
    {
        jsonv v; jsonv_init(&v);
        const char *pre = "{\"k\":\"v\xC3";
        assert(jsonv_feed(&v, pre, (int)strlen(pre)));
        assert(!jsonv_feed(&v, "\"", 1));
        assert(!jsonv_feed(&v, "a", 1));
        const char *rest = "\xA9\"}";
        assert(jsonv_feed(&v, rest, (int)strlen(rest)));
        assert(v.done);
        const char *doc = "{\"k\":\"v\xC3\xA9\"}";
        jv *parsed = json_parse(doc, strlen(doc));
        assert(parsed != NULL);
        jv_free(parsed);
    }
    // truncation inside a scalar: the closer finishes the sequence before
    // the quote, and the result parses
    {
        jsonv v; jsonv_init(&v);
        const char *pre = "{\"k\":\"v\xE4";
        assert(jsonv_feed(&v, pre, (int)strlen(pre)));
        char close[64];
        int n = jsonv_close(&v, close, (int)sizeof(close));
        assert(n > 0);
        char doc[128];
        int len = snprintf(doc, sizeof(doc), "%s%s", pre, close);
        jv *parsed = json_parse(doc, (size_t)len);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    // the same bytes through a free-object subtree in a schema
    {
        const char *src = "{\"type\":\"object\"}";
        jv *j = json_parse(src, strlen(src));
        assert(j != NULL);
        char err[128];
        snode *s = schema_compile(j, err, sizeof(err));
        assert(s != NULL);
        sval v; sval_init(&v, s);
        const char *pre = "{\"k";
        assert(sval_feed(&v, pre, (int)strlen(pre)));
        char b = (char)0x8B;
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        assert(!sval_trial(&v, &scratch, &b, 1));
        schema_free(s);
        jv_free(j);
    }
}

// json_parse refuses duplicate object keys AFTER unescaping, so both
// validators must too: the guards hash decoded key content, which makes
// `"a"`, `"a"` and the raw spelling of an escaped astral pair collide
// exactly as they do in the parser. The closers must not mint a duplicate
// either: force-closing mid-key extends the key until its hash is fresh.
static void test_duplicate_keys_match_parser(void) {
    // generic machine, raw duplicate: refused at the second closing quote
    {
        jsonv v; jsonv_init(&v);
        const char *pre = "{\"a\":1,\"a";
        assert(jsonv_feed(&v, pre, (int)strlen(pre)));
        assert(!jsonv_feed(&v, "\"", 1));
        assert(jsonv_feed(&v, "b\":2}", 5));   // extending the key is legal
        assert(v.done);
    }
    // generic machine, escaped duplicate of a raw key
    {
        jsonv v; jsonv_init(&v);
        const char *pre = "{\"a\":1,\"\\u0061";
        assert(jsonv_feed(&v, pre, (int)strlen(pre)));
        assert(!jsonv_feed(&v, "\"", 1));
    }
    // generic machine, surrogate pair vs the raw astral spelling
    {
        jsonv v; jsonv_init(&v);
        const char *pre = "{\"\xF0\x9F\x98\x80\":1,\"\\ud83d\\ude00";
        assert(jsonv_feed(&v, pre, (int)strlen(pre)));
        assert(!jsonv_feed(&v, "\"", 1));
    }
    // a sibling object at the same depth starts clean
    {
        jsonv v; jsonv_init_any(&v);
        const char *doc = "[{\"a\":1},{\"a\":2}]";
        assert(jsonv_feed(&v, doc, (int)strlen(doc)));
        assert(v.done);
    }
    // schema map, escaped duplicate (was a documented residual, now caught)
    const char *map_src =
        "{\"type\":\"object\",\"additionalProperties\":{\"type\":\"integer\"}}";
    jv *map_json = json_parse(map_src, strlen(map_src));
    assert(map_json != NULL);
    char err[128];
    snode *map_schema = schema_compile(map_json, err, sizeof(err));
    assert(map_schema != NULL);
    {
        sval v; sval_init(&v, map_schema);
        const char *pre = "{\"a\":1,\"\\u0061";
        assert(sval_feed(&v, pre, (int)strlen(pre)));
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        assert(!sval_trial(&v, &scratch, "\"", 1));
    }
    // schema-map closer: force-closing mid-duplicate-key extends the key
    // instead of completing the duplicate json_parse would refuse
    {
        sval v; sval_init(&v, map_schema);
        const char *pre = "{\"a\":1,\"a";
        assert(sval_feed(&v, pre, (int)strlen(pre)));
        char close[128];
        int wrote = sval_close(&v, close, (int)sizeof(close));
        assert(wrote > 0);
        char doc[192];
        int len = snprintf(doc, sizeof(doc), "%s%s", pre, close);
        jv *parsed = json_parse(doc, (size_t)len);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    // generic-machine closer, same property through a free-object subtree
    {
        jsonv v; jsonv_init(&v);
        const char *pre = "{\"k\":1,\"k";
        assert(jsonv_feed(&v, pre, (int)strlen(pre)));
        char close[128];
        int n = jsonv_close(&v, close, (int)sizeof(close));
        assert(n > 0);
        char doc[192];
        int len = snprintf(doc, sizeof(doc), "%s%s", pre, close);
        jv *parsed = json_parse(doc, (size_t)len);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    schema_free(map_schema);
    jv_free(map_json);
}

// At guard capacity the machines FAIL CLOSED instead of un-tracking keys:
// the comma that would start an untrackable entry is refused (and the first
// key of a map opened at capacity), while `}` stays legal — so a map is
// bounded at MAP_SEEN_MAX keys, never wedged, and never carries an
// unchecked duplicate. The old stop-tracking behavior let a 17th key
// duplicate an earlier one and emitted a document json_parse refuses
// (found by the CI fuzzer against the v0.4.1 tree).
static void test_key_guard_capacity_fails_closed(void) {
    const char *map_src =
        "{\"type\":\"object\",\"additionalProperties\":{\"type\":\"integer\"}}";
    jv *map_json = json_parse(map_src, strlen(map_src));
    assert(map_json != NULL);
    char err[128];
    snode *map_schema = schema_compile(map_json, err, sizeof(err));
    assert(map_schema != NULL);
    {
        sval v; sval_init(&v, map_schema);
        char doc[1024];
        int len = 0;
        doc[len++] = '{';
        for (int i = 0; i < MAP_SEEN_MAX; i++)
            len += snprintf(doc + len, sizeof(doc) - (size_t)len,
                            "%s\"k%02d\":%d", i ? "," : "", i, i);
        assert(sval_feed(&v, doc, len));
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        assert(!sval_trial(&v, &scratch, ",", 1));   // 25th entry refused
        assert(sval_feed(&v, "}", 1));               // closing stays legal
        assert(v.done);
        doc[len++] = '}';
        jv *parsed = json_parse(doc, (size_t)len);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    schema_free(map_schema);
    jv_free(map_json);

    // the generic machine, same shape
    {
        jsonv v; jsonv_init(&v);
        char doc[1024];
        int len = 0;
        doc[len++] = '{';
        for (int i = 0; i < JSON_KEY_SEEN_MAX; i++)
            len += snprintf(doc + len, sizeof(doc) - (size_t)len,
                            "%s\"k%02d\":%d", i ? "," : "", i, i);
        assert(jsonv_feed(&v, doc, len));
        assert(!jsonv_feed(&v, ",", 1));
        assert(jsonv_feed(&v, "}", 1));
        assert(v.done);
    }
    // a nested object opened AT capacity (the last recorded key's value)
    // refuses its first key but still closes empty — bounded, not wedged
    {
        jsonv v; jsonv_init(&v);
        char doc[1024];
        int len = 0;
        doc[len++] = '{';
        for (int i = 0; i < JSON_KEY_SEEN_MAX - 1; i++)
            len += snprintf(doc + len, sizeof(doc) - (size_t)len,
                            "%s\"k%02d\":%d", i ? "," : "", i, i);
        len += snprintf(doc + len, sizeof(doc) - (size_t)len,
                        ",\"last\":{");   // key 24 recorded, guard now full
        assert(jsonv_feed(&v, doc, len));
        assert(!jsonv_feed(&v, "\"", 1)); // inner first key untrackable
        assert(jsonv_feed(&v, "}}", 2));  // inner and outer still close
        assert(v.done);
    }
}

// The validator must never complete a number spelling json_parse refuses:
// the caller was promised a document this program reads back, and strtod's
// range refusals (overflow AND underflow — both ERANGE) are part of the
// parser's acceptance. The refusal lands on the exponent digit that commits
// the spelling, the same early-commit principle json_escape_hex documents,
// so the model always keeps a legal continuation (terminate the number).
static void test_schema_number_matches_parser(void) {
    const char *src = "{\"type\":\"number\"}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    // overflow: 9e30 is fine, the digit that makes 9e309 is refused, and
    // the number can still terminate where it stands
    {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, "9e30", 4));
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        assert(!sval_trial(&v, &scratch, "9", 1));
        char close[32];
        sval closing = v;
        int wrote = sval_close(&closing, close, (int)sizeof(close));
        assert(wrote >= 0);
        char doc[64];
        int len = snprintf(doc, sizeof(doc), "9e30%s", close);
        jv *parsed = json_parse(doc, (size_t)len);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    // underflow: strtod's ERANGE behavior in the subnormal band is libc-
    // specific (glibc and macOS refuse 1e-320, UCRT accepts it), and json.c
    // inherits the host's verdict. The invariant is therefore AGREEMENT,
    // not a fixed answer: the validator must accept the commit digit
    // exactly when the host's json_parse reads the result back.
    {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, "1e-300", 6));
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        bool accepted = sval_trial(&v, &scratch, "0", 1);
        jv *p = json_parse("[1e-3000]", 9);
        assert(accepted == (p != NULL));
        jv_free(p);
    }
    {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, "1e-32", 5));   // 1e-32: normal
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        bool accepted = sval_trial(&v, &scratch, "0", 1);   // 1e-320
        jv *p = json_parse("[1e-320]", 8);
        assert(accepted == (p != NULL));
        jv_free(p);
    }
    // a dangling exponent still closes to something the parser accepts
    {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, "9e", 2));
        char close[32];
        int wrote = sval_close(&v, close, (int)sizeof(close));
        assert(wrote > 0);
        char doc[64];
        int len = snprintf(doc, sizeof(doc), "9e%s", close);
        jv *parsed = json_parse(doc, (size_t)len);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    schema_free(schema);
    jv_free(schema_json);

    // the closer's INVENTED minimum must parse too: exclusiveMinimum:0
    // compiles to a clamped edge of 4.94e-324 — the smallest subnormal,
    // in bounds by construction, ERANGE at read-back. Force-closing an
    // array that still owes a minItems fill used to emit it verbatim
    // (found by the differential fuzzer's first CI run).
    const char *excl_src = "{\"type\":\"object\",\"properties\":{"
        "\"v\":{\"type\":\"array\",\"items\":"
          "{\"type\":\"number\",\"exclusiveMinimum\":0,\"maximum\":99},"
          "\"minItems\":1}},\"required\":[\"v\"]}";
    jv *excl_json = json_parse(excl_src, strlen(excl_src));
    assert(excl_json != NULL);
    snode *excl = schema_compile(excl_json, err, sizeof(err));
    assert(excl != NULL);
    {
        sval v; sval_init(&v, excl);
        const char *pre = "{\"v\":";
        assert(sval_feed(&v, pre, (int)strlen(pre)));
        char close[128];
        int wrote = sval_close(&v, close, (int)sizeof(close));
        assert(wrote > 0);
        char doc[192];
        int len = snprintf(doc, sizeof(doc), "%s%s", pre, close);
        jv *parsed = json_parse(doc, (size_t)len);
        assert(parsed != NULL);
        jv_free(parsed);
    }
    schema_free(excl);
    jv_free(excl_json);

    // the same numbers flowing through a free-object subtree reach the
    // generic submachine, which never sees a spelling — the wrapper must
    // apply the same commit refusal there
    const char *any_src = "{\"type\":\"object\"}";
    jv *any_json = json_parse(any_src, strlen(any_src));
    assert(any_json != NULL);
    snode *any_schema = schema_compile(any_json, err, sizeof(err));
    assert(any_schema != NULL);
    {
        sval v; sval_init(&v, any_schema);
        const char *pfx = "{\"x\":9e30";
        assert(sval_feed(&v, pfx, (int)strlen(pfx)));
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        assert(!sval_trial(&v, &scratch, "9", 1));   // 9e309 overflows
        assert(sval_feed(&v, "}", 1));               // terminating is legal
        assert(v.done);
    }
    schema_free(any_schema);
    jv_free(any_json);
}

static void test_schema_number_bounds_are_enforced(void) {
    const char *src = "{\"type\":\"object\",\"properties\":{"
        "\"timeout\":{\"type\":\"number\",\"minimum\":0,"
        "\"maximum\":600000}},\"required\":[\"timeout\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *good[] = { "{\"timeout\":0}", "{\"timeout\":1.5}",
                           "{\"timeout\":6e5}" };
    for (size_t i = 0; i < sizeof(good) / sizeof(*good); i++) {
        sval v; sval_init(&v, schema);
        assert(sval_feed(&v, good[i], (int)strlen(good[i])));
        assert(v.done);
    }
    const char *bad[] = { "{\"timeout\":-0.1}",
                          "{\"timeout\":600000.1}",
                          "{\"timeout\":7e5}" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v; sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])));
    }

    schema_free(schema);
    jv_free(schema_json);
}

// The validator state is COPIED once per candidate token by the engine's
// constraint layer (constraint_token_ok makes a shallow engine copy, and sval
// is the bulk of it), so its size is a decode-speed property, not an
// implementation detail. Only one frame at a time can be parsing a number --
// a number frame never pushes a child -- so the spelling buffer is shared by
// the whole stack rather than replicated 48 times.
static void test_sval_state_is_small(void) {
    assert(sizeof(sval) <= 2560);
}

// The shared number-spelling buffer must still be exact per value: sibling
// numbers in one array, and numbers at different depths, each get their own
// bounds check with no residue from the previous one.
static void test_schema_number_bounds_across_frames(void) {
    const char *src = "{\"type\":\"object\",\"properties\":{"
        "\"xs\":{\"type\":\"array\",\"items\":"
            "{\"type\":\"number\",\"minimum\":0,\"maximum\":100}},"
        "\"inner\":{\"type\":\"object\",\"properties\":{"
            "\"y\":{\"type\":\"number\",\"minimum\":0,\"maximum\":9}},"
            "\"required\":[\"y\"]}},"
        "\"required\":[\"xs\",\"inner\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *good = "{\"xs\":[1.5,99.25,0],\"inner\":{\"y\":8.5}}";
    sval v; sval_init(&v, schema);
    assert(sval_feed(&v, good, (int)strlen(good)));
    assert(v.done);

    // the SECOND element of the array is the one out of bounds: a stale
    // spelling from the first would hide it
    const char *bad_sibling = "{\"xs\":[1.5,100.5]";
    sval b; sval_init(&b, schema);
    assert(!sval_feed(&b, bad_sibling, (int)strlen(bad_sibling)));

    // and a deeper frame's bound is not the outer one's
    const char *bad_nested = "{\"xs\":[50],\"inner\":{\"y\":50}";
    sval d; sval_init(&d, schema);
    assert(!sval_feed(&d, bad_nested, (int)strlen(bad_nested)));

    schema_free(schema);
    jv_free(schema_json);
}

// sval_trial() is the candidate-token probe: it must answer exactly what a
// full struct copy fed the same bytes would answer, from every reachable
// state, while touching only the live part of the validator. The scratch is
// POISONED before every call — an implementation that reads a stack frame
// above `depth`, or the tail of the number buffer, sees 0xA5 there and the
// two answers diverge.
static void trial_matches_full_copy(const snode *schema, const char *doc) {
    sval v; sval_init(&v, schema);
    for (int i = 0; doc[i]; i++) {
        for (int c = 1; c < 256; c++) {
            char b = (char)c;
            sval full = v;
            bool want = sval_feed(&full, &b, 1);
            sval scratch;
            memset(&scratch, 0xA5, sizeof(scratch));
            bool got = sval_trial(&v, &scratch, &b, 1);
            assert(got == want);
            if (want) {
                assert(scratch.done == full.done);
                assert(scratch.depth == full.depth);
                // the map seen-key guard is live state: a scratch that
                // answers from poison instead of the real guard can accept
                // a duplicate key, and past MAP_SEEN_MAX it reads out of
                // bounds -- compare the state, not just the answer, so the
                // check cannot green on an accidental garbage match
                assert(scratch.n_seen == full.n_seen);
                assert(!memcmp(scratch.seen_hash, full.seen_hash,
                               (size_t)full.n_seen * sizeof full.seen_hash[0]));
                assert(!memcmp(scratch.seen_depth, full.seen_depth,
                               full.n_seen));
            }
        }
        // the probe must leave the real validator untouched
        sval before = v;
        char b = doc[i];
        sval scratch;
        memset(&scratch, 0xA5, sizeof(scratch));
        (void)sval_trial(&v, &scratch, &b, 1);
        assert(!memcmp(&before, &v, sizeof(v)));
        assert(sval_feed(&v, &b, 1));
    }
    assert(v.done);
}

static void test_sval_trial_matches_full_copy(void) {
    // covers: nested objects and arrays (stack depth), an in-progress number
    // (the shared spelling buffer), a string, an enum literal, and an open
    // `{}` node (the generic jsonv submachine inside the sval)
    const char *src = "{\"type\":\"object\",\"properties\":{"
        "\"name\":{\"type\":\"string\"},"
        "\"mode\":{\"enum\":[\"fast\",\"slow\"]},"
        "\"vals\":{\"type\":\"array\",\"items\":"
            "{\"type\":\"number\",\"minimum\":-5,\"maximum\":1000}},"
        "\"deep\":{\"type\":\"object\",\"properties\":{"
            "\"n\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":99}},"
            "\"required\":[\"n\"]},"
        "\"free\":{\"type\":\"object\"}},"
        "\"required\":[\"name\",\"mode\",\"vals\",\"deep\",\"free\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    trial_matches_full_copy(schema,
        "{\"name\":\"a b\",\"mode\":\"slow\",\"vals\":[1,-2.5,999],"
        "\"deep\":{\"n\":42},\"free\":{\"x\":[1,{\"y\":null}]}}");

    schema_free(schema);
    jv_free(schema_json);

    // The map seen-key guard is part of the oracle's answer: at the closing
    // quote of a duplicate key the full copy refuses while a scratch that
    // did not inherit the guard accepts. The document walks a key "a", then
    // a key with prefix "a" (so probing `"` right there IS the duplicate
    // refusal point), then a sibling map at the same depth that legally
    // reuses "a" after frame_done compacted the first map's entries out.
    const char *map_src = "{\"type\":\"object\",\"properties\":{"
        "\"m\":{\"type\":\"object\",\"additionalProperties\":"
            "{\"type\":\"integer\"}},"
        "\"n\":{\"type\":\"object\",\"additionalProperties\":"
            "{\"type\":\"integer\"}}},"
        "\"required\":[\"m\",\"n\"]}";
    jv *map_json = json_parse(map_src, strlen(map_src));
    assert(map_json != NULL);
    snode *map_schema = schema_compile(map_json, err, sizeof(err));
    assert(map_schema != NULL);

    trial_matches_full_copy(map_schema,
        "{\"m\":{\"a\":1,\"ab\":2,\"b\":3},\"n\":{\"a\":4}}");

    schema_free(map_schema);
    jv_free(map_json);
}

// The other half of the invariant: keywords that are pure annotations carry
// no constraint, so ignoring them ignores nothing. Real OpenAI tool payloads
// are full of them and must keep compiling.
static void test_schema_accepts_annotation_keywords(void) {
    const char *src =
        "{\"type\":\"object\",\"title\":\"T\",\"description\":\"d\","
        "\"$schema\":\"https://json-schema.org/draft/2020-12/schema\","
        "\"properties\":{\"a\":{\"type\":\"string\",\"description\":\"d\","
        "\"default\":\"x\",\"examples\":[\"y\"],\"format\":\"uri\"}},"
        "\"required\":[\"a\"],"
        "\"additionalProperties\":false}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_accepts_claude_open_metadata_object(void) {
    const char *src =
        "{\"type\":\"object\",\"propertyNames\":{\"type\":\"string\"},"
        "\"additionalProperties\":{}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    sval v;
    sval_init(&v, schema);
    const char *doc = "{\"owner\":\"agent\",\"deleted\":null}";
    assert(sval_feed(&v, doc, (int)strlen(doc)));
    assert(v.done);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_typed_additional_properties_map(void) {
    const char *src =
        "{\"type\":\"object\",\"additionalProperties\":{"
        "\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"string\"}},"
        "\"required\":[\"n\"],\"additionalProperties\":false}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *good = "{\"alpha\":{\"n\":\"x\"},\"beta\":{\"n\":\"y\"}}";
    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, good, (int)strlen(good)) && v.done);

    const char *bad[] = {
        "{\"alpha\":{}}",
        "{\"alpha\":{\"n\":1}}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])));
    }

    // Arbitrary keys are real string content (including escapes), and every
    // started map must still force-close to a document accepted by the same
    // schema when generation stops at a token boundary.
    const char *partial[] = {
        "{\"alpha",
        "{\"a\\\\",
        "{\"alpha\":",
        "{\"alpha\":{\"n\":\"x\"},",
    };
    for (size_t i = 0; i < sizeof(partial) / sizeof(*partial); i++) {
        sval_init(&v, schema);
        assert(sval_feed(&v, partial[i], (int)strlen(partial[i])));
        if (i == 0) assert(sval_ws_is_content(&v));
        char suffix[256];
        int n = sval_close(&v, suffix, sizeof(suffix));
        assert(n > 0);
        char full[512];
        snprintf(full, sizeof(full), "%s%s", partial[i], suffix);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv_free(parsed);
        sval check;
        sval_init(&check, schema);
        assert(sval_feed(&check, full, (int)strlen(full)) && check.done);
    }

    schema_free(schema);
    jv_free(schema_json);
}

// The compiled object enforces a CLOSED property set. `false` asks for
// exactly that and compiles; `true` asks for the opposite and used to be
// dropped, making the output STRICTER than the schema permitted.
static void test_schema_additional_properties(void) {
    static const char *const bad[] = {
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"additionalProperties\":true}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},"
            "\"additionalProperties\":{\"type\":\"string\"}}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        jv *schema_json = json_parse(bad[i], strlen(bad[i]));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema == NULL);
        assert(strstr(err, "additionalProperties") != NULL);
        jv_free(schema_json);
    }
}

// An *empty* properties map with additionalProperties:false is a different
// statement from an absent one: it declares an object with no permitted keys,
// i.e. exactly `{}`. Compiling it to the open any-object machine would have
// accepted any object at all — the opposite of the request — so it used to be
// rejected instead. Real agent clients (the Codex CLI's zero-argument tools)
// send it, and it is exactly expressible, so it compiles.
static void test_schema_empty_closed_object(void) {
    const char *srcs[] = {
        "{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}",
        // Omitting `properties` says the same thing: no declared keys, and no
        // additional ones. Claude and other clients use both spellings.
        "{\"type\":\"object\",\"additionalProperties\":false}",
    };
    for (size_t i = 0; i < sizeof(srcs) / sizeof(*srcs); i++) {
        const char *src = srcs[i];
        jv *schema_json = json_parse(src, strlen(src));
        assert(schema_json != NULL);
        char err[128];
        snode *schema = schema_compile(schema_json, err, sizeof(err));
        assert(schema != NULL);

        // the empty object is accepted and closes with nothing outstanding
        sval v;
        sval_init(&v, schema);
        assert(sval_feed(&v, "{", 1));
        assert(sval_feed(&v, "}", 1));
        char out[64];
        assert(sval_close(&v, out, sizeof(out)) == 0);

        // any key at all is refused: the set really is closed
        sval_init(&v, schema);
        assert(sval_feed(&v, "{", 1));
        assert(!sval_feed(&v, "\"", 1));

        // and a truncation mid-document still closes to a legal `{}`
        sval_init(&v, schema);
        assert(sval_feed(&v, "{", 1));
        int n = sval_close(&v, out, sizeof(out));
        assert(n == 1 && out[0] == '}');

        schema_free(schema);
        jv_free(schema_json);
    }
}

// `required` with no `properties` compiled to the open any-object machine,
// which enforces no key at all — the requirement vanished silently.
static void test_schema_rejects_required_without_properties(void) {
    const char *src = "{\"type\":\"object\",\"required\":[\"a\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "required") != NULL);
    jv_free(schema_json);

    // an empty required list asks for nothing, so it stays legal
    const char *ok = "{\"type\":\"object\",\"required\":[]}";
    schema_json = json_parse(ok, strlen(ok));
    assert(schema_json != NULL);
    schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_oneof_const_scalars(void) {
    const char *src =
        "{\"oneOf\":[{\"const\":\"read_file\"},{\"const\":\"done\"}]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"done\"", 6));

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_oneof_const_numeric_prefixes(void) {
    const char *src =
        "{\"oneOf\":[{\"const\":1},{\"const\":12}]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "12", 2));

    schema_free(schema);
    jv_free(schema_json);
}

static bool accepts(const char *schema_src, const char *doc);

static void test_schema_rejects_oversized_oneof_const_scalars(void) {
    // 65 scalar consts compiled to one enum; the cap is now 4096 (the walker
    // keeps a sorted range, not one bit per literal), so oversized is past that
    static char src[4097 * 16 + 64];
    int n = snprintf(src, sizeof(src), "{\"oneOf\":[");
    for (int i = 0; i < 4097; i++)
        n += snprintf(src + n, sizeof(src) - (size_t)n,
                      "%s{\"const\":%d}", i ? "," : "", i);
    snprintf(src + n, sizeof(src) - (size_t)n, "]}");
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "enum size") != NULL);
    jv_free(schema_json);
    // sixty-five, which the old mask refused, is an ordinary enum now
    char small[2048];
    n = snprintf(small, sizeof(small), "{\"oneOf\":[");
    for (int i = 0; i < 65; i++)
        n += snprintf(small + n, sizeof(small) - (size_t)n, "%s{\"const\":%d}", i ? "," : "", i);
    snprintf(small + n, sizeof(small) - (size_t)n, "]}");
    assert(accepts(small, "64") && !accepts(small, "65"));
}

static void test_schema_string_length_bounds_close_and_reject(void) {
    const char *src = "{\"type\":\"string\",\"minLength\":5,\"maxLength\":8}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"abc", 4));
    char out[64];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[128];
    snprintf(full, sizeof(full), "\"abc%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(strlen(jv_str(parsed, "")) >= 5);
    jv_free(parsed);

    sval_init(&v, schema);
    assert(!sval_feed(&v, "\"123456789\"", 11));

    schema_free(schema);
    jv_free(schema_json);
}

// JSON Schema measures string length in Unicode code points, independent of
// whether the model spells one as raw UTF-8 or a \u escape. A byte counter made
// the first two raw forms exceed maxLength while their escaped twins fit.
static void test_schema_string_length_counts_code_points(void) {
    const char *src = "{\"type\":\"string\",\"maxLength\":1}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *one[] = {
        "\"\xC3\xA9\"",            // U+00E9 as raw UTF-8
        "\"\\u00e9\"",             // U+00E9 as an escape
        "\"\xF0\x9F\x98\x80\"",  // U+1F600 as raw UTF-8
        "\"\\uD83D\\uDE00\"",     // U+1F600 as a surrogate pair
    };
    for (size_t i = 0; i < sizeof(one) / sizeof(*one); i++) {
        sval v;
        sval_init(&v, schema);
        if (!sval_feed(&v, one[i], (int)strlen(one[i])) || !v.done)
            fprintf(stderr, "one code point rejected: %s\n", one[i]);
        assert(v.done);
    }

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_string_length_rejects_at_one_code_point(void) {
    const char *src = "{\"type\":\"string\",\"maxLength\":1}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *two[] = {
        "\"\xC3\xA9\xC3\xA9\"",
        "\"\\u00e9\\u00e9\"",
        "\"\xF0\x9F\x98\x80\xF0\x9F\x98\x80\"",
        "\"\\uD83D\\uDE00\\uD83D\\uDE00\"",
    };
    for (size_t i = 0; i < sizeof(two) / sizeof(*two); i++) {
        sval v;
        sval_init(&v, schema);
        assert(!sval_feed(&v, two[i], (int)strlen(two[i])) || !v.done);
    }

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_utf8_boundary_has_a_legal_continuation(void) {
    const char *src = "{\"type\":\"string\",\"maxLength\":1}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v, trial;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"", 1));
    assert(sval_feed(&v, "\xC3", 1));
    assert(!sval_trial(&v, &trial, "\"", 1));
    assert(sval_trial(&v, &trial, "\xA9", 1));
    assert(sval_feed(&v, "\xA9\"", 2));
    assert(v.done);

    // Once an ASCII code point has spent the only slot, a multibyte lead is
    // refused at the boundary rather than admitted into a dead prefix.
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"a", 2));
    assert(!sval_trial(&v, &trial, "\xC3", 1));

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_rejects_unpaired_surrogates(void) {
    const char *src = "{\"type\":\"string\",\"maxLength\":1}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *bad[] = { "\"\\uD800\"", "\"\\uDC00\"" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        sval v;
        sval_init(&v, schema);
        assert(!sval_feed(&v, bad[i], (int)strlen(bad[i])) || !v.done);
    }

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_minlength_close_after_multibyte_content(void) {
    const char *src =
        "{\"type\":\"string\",\"minLength\":2,\"maxLength\":2}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    // Exercise both a complete raw scalar and truncation after only its lead.
    const char *partials[] = { "\"\xC3\xA9", "\"\xF0" };
    for (size_t i = 0; i < sizeof(partials) / sizeof(*partials); i++) {
        sval v;
        sval_init(&v, schema);
        assert(sval_feed(&v, partials[i], (int)strlen(partials[i])));
        char suffix[64];
        int n = sval_close(&v, suffix, sizeof(suffix));
        assert(n > 0);
        char full[128];
        snprintf(full, sizeof(full), "%s%s", partials[i], suffix);
        sval check;
        sval_init(&check, schema);
        assert(sval_feed(&check, full, (int)strlen(full)) && check.done);
        jv *parsed = json_parse(full, strlen(full));
        assert(parsed != NULL);
        jv_free(parsed);
    }

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_close_finishes_multibyte_map_key(void) {
    const char *src =
        "{\"type\":\"object\",\"additionalProperties\":{\"type\":\"string\"}}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    const char *partial = "{\"\xF0";
    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, partial, (int)strlen(partial)));
    char suffix[64];
    int n = sval_close(&v, suffix, sizeof(suffix));
    assert(n > 0);
    char full[128];
    snprintf(full, sizeof(full), "%s%s", partial, suffix);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    jv_free(parsed);

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_string_minlength_full_close(void) {
    const char *src =
        "{\"type\":\"object\",\"properties\":{"
        "\"name\":{\"type\":\"string\",\"minLength\":3}"
        "},\"required\":[\"name\"]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    char out[128];

    // Closing a decode that never opened the document emits nothing: there is
    // no partial value to pad out. (This used to return a fabricated
    // `{"name":"   "}`; see
    // test_schema_close_without_payload_fabricates_nothing for why that had
    // to stop.) minLength padding is exercised below, where the model really
    // did start the string.
    sval_init(&v, schema);
    assert(sval_close(&v, out, sizeof(out)) == 0);

    sval_init(&v, schema);
    assert(sval_feed(&v, "{\"name\":", 8));
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[256];
    snprintf(full, sizeof(full), "{\"name\":%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(strlen(jv_str(jv_get(parsed, "name"), "")) >= 3);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_discriminated_action_args(void) {
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"tool\":{\"const\":\"read_file\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}"
        "},\"required\":[\"thinking\",\"tool\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"tool\":{\"const\":\"done\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]}"
        "},\"required\":[\"thinking\",\"tool\",\"args\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    const char *valid =
        "{\"thinking\":\"x\",\"tool\":\"done\",\"args\":{\"summary\":\"ok\"}}";
    sval_init(&v, schema);
    assert(sval_feed(&v, valid, strlen(valid)));

    const char *invalid =
        "{\"thinking\":\"x\",\"tool\":\"done\",\"args\":{\"path\":\"README.md\"}}";
    sval_init(&v, schema);
    assert(!sval_feed(&v, invalid, strlen(invalid)));

    const char *partial = "{\"thinking\":\"x\",\"tool\":\"done\"";
    sval_init(&v, schema);
    assert(sval_feed(&v, partial, strlen(partial)));
    char out[128];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[256];
    snprintf(full, sizeof(full), "{\"thinking\":\"x\",\"tool\":\"done\"%s", out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    jv *args = jv_get(parsed, "args");
    assert(args != NULL);
    assert(jv_get(args, "summary") != NULL);
    assert(jv_get(args, "path") == NULL);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_union_dispatch_rules(void) {
    // ambiguous alternatives (two object shapes) must fail at compile time
    const char *dup =
        "{\"oneOf\":[{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}},"
        "{\"type\":\"object\",\"properties\":{\"b\":{\"type\":\"string\"}}}]}";
    jv *j = json_parse(dup, strlen(dup));
    assert(j != NULL);
    char err[128];
    assert(schema_compile(j, err, sizeof(err)) == NULL);
    jv_free(j);

    // nested oneOf inside a union alternative is not dispatchable
    const char *nested =
        "{\"oneOf\":[{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]},"
        "{\"type\":\"boolean\"}]}";
    j = json_parse(nested, strlen(nested));
    assert(j != NULL);
    assert(schema_compile(j, err, sizeof(err)) == NULL);
    jv_free(j);

    // a mixed-first-byte enum alternative must reach ALL its literals
    const char *mixed =
        "{\"oneOf\":[{\"enum\":[1,\"a\"]},{\"type\":\"boolean\"}]}";
    j = json_parse(mixed, strlen(mixed));
    assert(j != NULL);
    snode *schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"a\"", 3));
    assert(v.done);
    sval_init(&v, schema);
    assert(sval_feed(&v, "true", 4));
    assert(v.done);
    schema_free(schema);
    jv_free(j);

    // disjoint types still compile and dispatch
    const char *ok = "{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}";
    j = json_parse(ok, strlen(ok));
    assert(j != NULL);
    schema = schema_compile(j, err, sizeof(err));
    assert(schema != NULL);
    sval_init(&v, schema);
    assert(sval_feed(&v, "null", 4));
    assert(v.done);
    schema_free(schema);
    jv_free(j);
}

// R2.8: the tool/args action with different required keys per tool used to be
// refused ("oneOf alternatives declare different required keys"). Each branch
// now keeps its own required set: `a` must carry args, `b` may leave them out.
static void test_schema_discriminated_required_may_differ(void) {
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"a\"},\"args\":{\"type\":\"object\","
        "\"properties\":{\"x\":{\"type\":\"string\"}}}},"
        "\"required\":[\"tool\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"b\"},\"args\":{\"type\":\"object\","
        "\"properties\":{\"y\":{\"type\":\"string\"}}}},"
        "\"required\":[\"tool\"]}"
        "]}";
    assert(schema_accepts(src, "{\"tool\":\"a\",\"args\":{\"x\":\"1\"}}"));
    assert(!schema_accepts(src, "{\"tool\":\"a\"}"));
    assert(schema_accepts(src, "{\"tool\":\"b\"}"));
    assert(schema_accepts(src, "{\"tool\":\"b\",\"args\":{\"y\":\"1\"}}"));
    assert(!schema_accepts(src, "{\"tool\":\"b\",\"args\":{\"x\":\"1\"}}"));
}

// R2.8: a statement of a small program language whose operation decides its
// arity. MAP takes a lambda and one argument, HEAD no lambda and one argument,
// TAKE no lambda and two. A flat {op, fn, args} schema admits mostly
// ill-typed statements; a `const`-discriminated union carries the typing, because the
// discriminator is the first thing the walker reads and every other property
// then follows the branch it named.
static const char *LX_STATEMENT =
    "{\"oneOf\":["
    "{\"type\":\"object\",\"properties\":{\"op\":{\"const\":\"MAP\"},"
    "\"fn\":{\"enum\":[\"INC\",\"DEC\"]},"
    "\"args\":{\"type\":\"array\",\"minItems\":1,\"maxItems\":1,\"items\":{\"enum\":[\"a\",\"b\"]}}},"
    "\"required\":[\"op\",\"fn\",\"args\"],\"additionalProperties\":false},"
    "{\"type\":\"object\",\"properties\":{\"op\":{\"const\":\"HEAD\"},"
    "\"args\":{\"type\":\"array\",\"minItems\":1,\"maxItems\":1,\"items\":{\"enum\":[\"a\",\"b\"]}}},"
    "\"required\":[\"op\",\"args\"],\"additionalProperties\":false},"
    "{\"type\":\"object\",\"properties\":{\"args\":{\"type\":\"array\",\"minItems\":2,\"maxItems\":2,"
    "\"items\":{\"enum\":[\"a\",\"b\"]}},\"op\":{\"const\":\"TAKE\"}},"
    "\"required\":[\"args\",\"op\"],\"additionalProperties\":false}"
    "]}";

// Close a prefix the way a max_tokens cut does and return the whole document.
static jv *close_prefix(const char *src, const char *prefix, char *doc, int cap) {
    jv *j = json_parse(src, strlen(src));
    char err[160];
    snode *s = schema_compile(j, err, sizeof(err));
    assert(s != NULL);
    sval v; sval_init(&v, s);
    assert(sval_feed(&v, prefix, (int)strlen(prefix)));
    char tail[256];
    int n = sval_close(&v, tail, sizeof(tail));
    assert(n >= 0);
    snprintf(doc, (size_t)cap, "%s%s", prefix, tail);
    sval_init(&v, s);
    bool ok = sval_feed(&v, doc, (int)strlen(doc)) && v.done;
    assert(ok && "a closed document must satisfy its own schema");
    schema_free(s); jv_free(j);
    return json_parse(doc, strlen(doc));
}

// The walker's verdict on a prefix: accepted so far, finished or not.
static bool prefix_ok(const char *src, const char *prefix) {
    jv *j = json_parse(src, strlen(src));
    char err[160];
    snode *s = schema_compile(j, err, sizeof(err));
    assert(s != NULL);
    sval v; sval_init(&v, s);
    bool ok = sval_feed(&v, prefix, (int)strlen(prefix));
    schema_free(s); jv_free(j);
    return ok;
}

static void test_schema_const_discriminated_union(void) {
    const char *s = LX_STATEMENT;
    // refused at the first byte that leaves the branch, not at a value later:
    // a key the branch lacks would be a dead end for constrained sampling,
    // `"fn":` written with no value that can follow
    assert(!prefix_ok(s, "{\"op\":\"HEAD\",\"f"));
    assert(!prefix_ok(s, "{\"op\":\"MAP\",\"a"));      // MAP's lambda comes first
    assert(prefix_ok(s, "{\"op\":\"MAP\",\"f"));
    assert(!prefix_ok(s, "{\"op\":\"HEAD\",\"args\":[\"a\"],"));  // nothing after HEAD's args
    assert(!prefix_ok(s, "{\"a"));                     // the discriminator comes first
    // each operation with its own arity
    assert(schema_accepts(s, "{\"op\":\"MAP\",\"fn\":\"INC\",\"args\":[\"a\"]}"));
    assert(schema_accepts(s, "{\"op\":\"HEAD\",\"args\":[\"b\"]}"));
    assert(schema_accepts(s, "{\"op\":\"TAKE\",\"args\":[\"a\",\"b\"]}"));
    assert(schema_accepts(s, "{ \"op\" : \"MAP\" , \"fn\" : \"DEC\" , \"args\" : [ \"b\" ] }"));
    // the ill-typed statements the flat schema admitted
    assert(!schema_accepts(s, "{\"op\":\"HEAD\",\"fn\":\"INC\",\"args\":[\"a\"]}"));  // a lambda on HEAD
    assert(!schema_accepts(s, "{\"op\":\"MAP\",\"args\":[\"a\"]}"));                  // MAP without one
    assert(!schema_accepts(s, "{\"op\":\"TAKE\",\"args\":[\"a\"]}"));                 // TAKE takes two
    assert(!schema_accepts(s, "{\"op\":\"HEAD\",\"args\":[\"a\",\"b\"]}"));           // HEAD takes one
    assert(!schema_accepts(s, "{\"op\":\"DROP\",\"args\":[\"a\"]}"));                 // no such operation
    // the discriminator comes first, though TAKE declares it last
    assert(!schema_accepts(s, "{\"args\":[\"a\",\"b\"],\"op\":\"TAKE\"}"));
    // anyOf over the same branches is the same language (the consts are distinct)
    char any[1024];
    snprintf(any, sizeof(any), "{\"anyOf\"%s", strchr(s, ':'));
    assert(schema_accepts(any, "{\"op\":\"HEAD\",\"args\":[\"b\"]}"));
    assert(!schema_accepts(any, "{\"op\":\"HEAD\",\"fn\":\"INC\",\"args\":[\"a\"]}"));

    // a cut document closes inside the branch it is in
    char doc[512];
    jv *d = close_prefix(s, "{\"op\":\"HEAD\"", doc, sizeof(doc));
    assert(!jv_get(d, "fn") && jv_get(d, "args")); jv_free(d);
    d = close_prefix(s, "{\"op\":\"TA", doc, sizeof(doc));
    assert(!strcmp(jv_str(jv_get(d, "op"), ""), "TAKE") && jv_get(d, "args")->n == 2); jv_free(d);
    d = close_prefix(s, "{\"op\":\"MAP\",", doc, sizeof(doc));
    assert(jv_get(d, "fn") && jv_get(d, "args")); jv_free(d);
    d = close_prefix(s, "{", doc, sizeof(doc));       // nothing chosen yet: the first branch
    assert(!strcmp(jv_str(jv_get(d, "op"), ""), "MAP") && jv_get(d, "fn")); jv_free(d);
    d = close_prefix(s, "{\"o", doc, sizeof(doc));
    assert(!strcmp(jv_str(jv_get(d, "op"), ""), "MAP")); jv_free(d);
    d = close_prefix(s, "{\"op\":\"TAKE\",\"args\":[\"a\"", doc, sizeof(doc));
    assert(jv_get(d, "args")->n == 2); jv_free(d);

    // Optional[Union[A, B]]: the objects merge, null stays beside them
    const char *opt =
        "{\"anyOf\":[{\"properties\":{\"kind\":{\"const\":\"a\"},\"x\":{\"type\":\"integer\"}},"
        "\"required\":[\"kind\",\"x\"]},{\"type\":\"null\"},"
        "{\"properties\":{\"kind\":{\"const\":\"b\"}},\"required\":[\"kind\"]}]}";
    assert(schema_accepts(opt, "null"));
    assert(schema_accepts(opt, "{\"kind\":\"a\",\"x\":1}"));
    assert(schema_accepts(opt, "{\"kind\":\"b\"}"));
    assert(!schema_accepts(opt, "{\"kind\":\"b\",\"x\":1}"));
    assert(!schema_accepts(opt, "{\"kind\":\"a\"}"));

    // OpenAPI's and Pydantic's `discriminator` names the property; its
    // `mapping` names $refs and adds nothing
    const char *named =
        "{\"oneOf\":[{\"properties\":{\"id\":{\"const\":1},\"t\":{\"const\":\"x\"}},\"required\":[\"id\",\"t\"]},"
        "{\"properties\":{\"id\":{\"const\":1},\"t\":{\"const\":\"y\"},\"n\":{\"type\":\"integer\"}},"
        "\"required\":[\"id\",\"t\",\"n\"]}],"
        "\"discriminator\":{\"propertyName\":\"t\",\"mapping\":{\"x\":\"#/$defs/X\"}}}";
    assert(schema_accepts(named, "{\"t\":\"y\",\"id\":1,\"n\":2}"));
    assert(!schema_accepts(named, "{\"t\":\"x\",\"id\":1,\"n\":2}"));
    assert(schema_refused("{\"oneOf\":[{\"properties\":{\"t\":{\"const\":\"x\"}},\"required\":[\"t\"]},"
                          "{\"properties\":{\"t\":{\"const\":\"y\"}},\"required\":[\"t\"]}],"
                          "\"discriminator\":{\"propertyName\":\"kind\"}}"));
    assert(schema_refused("{\"oneOf\":[{\"properties\":{\"t\":{\"const\":\"x\"}},\"required\":[\"t\"]},"
                          "{\"properties\":{\"t\":{\"const\":\"y\"}},\"required\":[\"t\"]}],"
                          "\"discriminator\":{\"propertyName\":\"t\",\"other\":1}}"));
    {   // a union without one says what would make it compile
        const char *src = "{\"oneOf\":[{\"properties\":{\"a\":{\"type\":\"string\"}}},"
                          "{\"properties\":{\"b\":{\"type\":\"string\"}}}]}";
        jv *j = json_parse(src, strlen(src));
        char err[200];
        assert(schema_compile(j, err, sizeof(err)) == NULL);
        assert(strstr(err, "nothing to tell them apart") && strstr(err, "discriminator"));
        jv_free(j);
    }

    // still refused: objects with nothing to tell them apart, or not by a const
    assert(schema_refused("{\"oneOf\":[{\"properties\":{\"k\":{\"const\":\"x\"}},\"required\":[\"k\"]},"
                          "{\"properties\":{\"k\":{\"const\":\"x\"},\"v\":{\"type\":\"string\"}},\"required\":[\"k\"]}]}"));
    assert(schema_refused("{\"oneOf\":[{\"properties\":{\"k\":{\"const\":\"x\"}},\"required\":[\"k\"]},"
                          "{\"properties\":{\"k\":{\"const\":\"y\"}}}]}"));          // optional in one branch
    assert(schema_refused("{\"oneOf\":[{\"properties\":{\"k\":{\"type\":\"string\"}},\"required\":[\"k\"]},"
                          "{\"properties\":{\"k\":{\"type\":\"integer\"}},\"required\":[\"k\"]}]}"));
}

static void test_schema_long_string_value(void) {
    // file-sized string values must not trip the length counter (a 16-bit
    // counter would wrap past 32767 and reject the closing quote)
    const char *src = "{\"type\":\"string\"}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(sval_feed(&v, "\"", 1));
    char chunk[4096];
    memset(chunk, 'a', sizeof(chunk));
    for (int i = 0; i < 20; i++) // 80KB of content
        assert(sval_feed(&v, chunk, sizeof(chunk)));
    assert(sval_feed(&v, "\"", 1));
    assert(v.done);

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_escape_at_maxlength_rejected(void) {
    // at maxLength only the closing quote may follow; starting an escape
    // there would force close() to overrun the bound to stay valid JSON
    const char *src = "{\"type\":\"string\",\"maxLength\":3}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    assert(!sval_feed(&v, "\"abc\\", 5));

    sval_init(&v, schema); // an escape below the bound still counts as one char
    assert(sval_feed(&v, "\"ab\\n\"", 6));
    assert(v.done);

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_nested_tool_key_keeps_discriminator(void) {
    // a "tool" key inside a nested object between the discriminator and args
    // must not overwrite the outer object's chosen alternative
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"read_file\"},"
        "\"meta\":{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"enum\":[\"x\",\"y\"]}},\"required\":[\"tool\"]},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}"
        "},\"required\":[\"tool\",\"meta\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"done\"},"
        "\"meta\":{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"enum\":[\"x\",\"y\"]}},\"required\":[\"tool\"]},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]}"
        "},\"required\":[\"tool\",\"meta\",\"args\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    const char *valid = "{\"tool\":\"read_file\",\"meta\":{\"tool\":\"y\"},"
                        "\"args\":{\"path\":\"a.txt\"}}";
    sval_init(&v, schema);
    assert(sval_feed(&v, valid, strlen(valid)));
    assert(v.done);

    // args from the wrong alternative must still be rejected
    const char *invalid = "{\"tool\":\"read_file\",\"meta\":{\"tool\":\"y\"},"
                          "\"args\":{\"summary\":\"no\"}}";
    sval_init(&v, schema);
    assert(!sval_feed(&v, invalid, strlen(invalid)));

    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_close_mid_discriminator_matches_args(void) {
    // aborting generation inside the second alternative's tool literal must
    // close with that alternative's args, not alternative 0's
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"alpha\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}"
        "},\"required\":[\"tool\",\"args\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"tool\":{\"const\":\"zeta\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{"
        "\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]}"
        "},\"required\":[\"tool\",\"args\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    const char *partial = "{\"tool\":\"ze"; // unambiguously alternative 1
    sval_init(&v, schema);
    assert(sval_feed(&v, partial, strlen(partial)));
    char out[256];
    int n = sval_close(&v, out, sizeof(out));
    assert(n > 0);
    char full[512];
    snprintf(full, sizeof(full), "%s%s", partial, out);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(!strcmp(jv_str(jv_get(parsed, "tool"), ""), "zeta"));
    jv *args = jv_get(parsed, "args");
    assert(args != NULL);
    assert(jv_get(args, "summary") != NULL);
    assert(jv_get(args, "path") == NULL);

    jv_free(parsed);
    schema_free(schema);
    jv_free(schema_json);
}

static void test_schema_rejects_discriminator_after_conditional_args(void) {
    const char *src =
        "{\"oneOf\":["
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]},"
        "\"tool\":{\"const\":\"read_file\"}"
        "},\"required\":[\"thinking\",\"args\",\"tool\"]},"
        "{\"type\":\"object\",\"properties\":{"
        "\"thinking\":{\"type\":\"string\"},"
        "\"args\":{\"type\":\"object\",\"properties\":{\"summary\":{\"type\":\"string\"}},\"required\":[\"summary\"]},"
        "\"tool\":{\"const\":\"done\"}"
        "},\"required\":[\"thinking\",\"args\",\"tool\"]}"
        "]}";
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema == NULL);
    assert(strstr(err, "tool") != NULL);
    jv_free(schema_json);
}

// A response body must be decodable by a strict UTF-8 JSON client. Structural
// validation is not enough: 0xC0 has a valid two-byte lead pattern and can be
// followed by a valid continuation byte, yet no conforming decoder accepts it
// (it is an overlong encoding). Emitting it produced a Messages body Python
// refused outright with "invalid start byte", which a model reaches whenever a
// byte-fallback token emits a stray byte or max_tokens cuts a character in
// half. Every rejected form must come back as U+FFFD, never as raw bytes.
static void test_escape_replaces_ill_formed_utf8(void) {
    struct { const char *in; size_t n; const char *what; } bad[] = {
        { "\xC0\x80",         2, "overlong two-byte (0xC0)" },
        { "\xC1\xBF",         2, "overlong two-byte (0xC1)" },
        { "\xE0\x80\x80",     3, "overlong three-byte" },
        { "\xED\xA0\x80",     3, "UTF-16 surrogate U+D800" },
        { "\xF0\x80\x80\x80", 4, "overlong four-byte" },
        { "\xF4\x90\x80\x80", 4, "beyond U+10FFFF" },
        { "\xF5\x80\x80\x80", 4, "lead byte 0xF5" },
        { "\x80",             1, "bare continuation byte" },
        { "\xE2\x82",         2, "truncated three-byte at end of buffer" },
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        char out[64];
        size_t m = json_escape(bad[i].in, bad[i].n, out, sizeof(out));
        for (size_t k = 0; k < m; k++)
            if ((unsigned char)out[k] >= 0x80) {
                // the only bytes above ASCII that may survive are U+FFFD's
                assert(((unsigned char)out[k] == 0xEF ||
                        (unsigned char)out[k] == 0xBF ||
                        (unsigned char)out[k] == 0xBD) && bad[i].what);
            }
        // and the result must itself re-validate as UTF-8
        for (size_t k = 0; k < m; ) {
            unsigned char c = (unsigned char)out[k];
            size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 :
                         (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
            assert(len != 0 && k + len <= m && bad[i].what);
            k += len;
        }
    }
    // well-formed input must pass through untouched
    const char *ok = "caf\xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80";
    char out[64];
    size_t m = json_escape(ok, strlen(ok), out, sizeof(out));
    assert(m == strlen(ok) && memcmp(out, ok, m) == 0);
    puts("ok: ill-formed UTF-8 is replaced, well-formed passes through");
}

// jv_dump_tojson exists to reproduce jinja's `tojson` filter byte for byte,
// because four families declare their tools through it (chatml.jinja:11,
// chatml-think.jinja:9, ornith.jinja:50, muse.jinja:73) and the declaration
// block is prompt text the model was trained on, not presentation.
//
// Every `want` below was RECORDED from that filter -- jinja2 with the
// reference's own tojson (json.dumps(ensure_ascii=False), the settings
// scripts/template-conformance.py installs) -- not written from expectation.
static void test_tojson_dump_matches_jinja(void) {
    struct { const char *doc; const char *want; const char *what; } cases[] = {
        // the tool declaration the conformance matrix actually sends
        { "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
          "\"description\":\"Get the current weather for a city.\","
          "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":"
          "{\"type\":\"string\",\"description\":\"City name.\"}},"
          "\"required\":[\"city\"]}}}",
          "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", "
          "\"description\": \"Get the current weather for a city.\", "
          "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": "
          "{\"type\": \"string\", \"description\": \"City name.\"}}, "
          "\"required\": [\"city\"]}}}",
          "the tool declaration block" },
        // empty containers take NO inner space
        { "{}", "{}", "empty object" },
        { "[]", "[]", "empty array" },
        { "{\"a\":{},\"b\":[],\"c\":[[],{}]}",
          "{\"a\": {}, \"b\": [], \"c\": [[], {}]}", "nested empty containers" },
        // insertion order, never sorted (json.dumps sort_keys defaults False)
        { "{\"z\":1,\"a\":2,\"m\":3}", "{\"z\": 1, \"a\": 2, \"m\": 3}",
          "key order is the document's, not sorted" },
        // ensure_ascii=False: non-ASCII stays raw UTF-8, never \uXXXX
        { "{\"k\":\"caf\\u00e9 \\u20ac \\ud83d\\ude00\"}",
          "{\"k\": \"caf\xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80\"}",
          "non-ASCII is raw UTF-8" },
        // short forms for \" \\ \n \t \r \b \f, \u00xx lowercase for the
        // rest of C0, and NEITHER '/' nor DEL is escaped
        { "{\"k\":\"a\\\"b\\\\c\\nd\\te\\rf\\bg\\fh\\u001fi/j\\u007f\"}",
          "{\"k\": \"a\\\"b\\\\c\\nd\\te\\rf\\bg\\fh\\u001fi/j\x7f\"}",
          "escaping" },
        { "{\"t\":true,\"f\":false,\"n\":null}",
          "{\"t\": true, \"f\": false, \"n\": null}", "literals" },
        { "[1,\"a\",null,true,{},[]]", "[1, \"a\", null, true, {}, []]",
          "array separators" },
        { "{\"nested\":{\"deep\":{\"x\":[1,2,3]}}}",
          "{\"nested\": {\"deep\": {\"x\": [1, 2, 3]}}}", "nesting" },
        { "\"plain string\"", "\"plain string\"", "bare string" },
        { "42", "42", "bare number" },
        { "{\"min\":0,\"max\":100,\"mult\":0.01}",
          "{\"min\": 0, \"max\": 100, \"mult\": 0.01}", "schema numerics" },
        // a double prints as the SHORTEST text that reads back to it, as
        // json.dumps does (repr); "%.10g" cut every one of these but 0.1 and
        // 1e-07, so a tool argument or a schema bound with more than ten
        // significant digits reached the client or the prompt altered
        { "[3.141592653589793,1234567.891234,0.1,1e-07,1.5e300,"
          "123456789.12345679,-0.000123456789012]",
          "[3.141592653589793, 1234567.891234, 0.1, 1e-07, 1.5e+300, "
          "123456789.12345679, -0.000123456789012]",
          "doubles round-trip at their shortest spelling" },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        jv *v = json_parse(cases[i].doc, strlen(cases[i].doc));
        assert(v != NULL && cases[i].what);
        sbuf spaced = { 0 };
        jv_dump_tojson(v, &spaced);
        assert(!spaced.failed && spaced.s && cases[i].what);
        if (strcmp(spaced.s, cases[i].want) != 0) {
            fprintf(stderr, "tojson drift (%s)\n  want: %s\n  got : %s\n",
                    cases[i].what, cases[i].want, spaced.s);
            assert(0);
        }
        // and the compact dump the HTTP response path uses must NOT have
        // moved: this variant is additive, and a body that silently
        // reformatted would be the expensive way to find that out
        sbuf compact = { 0 };
        jv_dump(v, &compact);
        assert(!compact.failed && compact.s && cases[i].what);
        assert(strstr(compact.s, ", ") == NULL && cases[i].what);
        assert(strstr(compact.s, "\": ") == NULL && cases[i].what);
        free(spaced.s);
        free(compact.s);
        jv_free(v);
    }
    puts("ok: jv_dump_tojson reproduces jinja's tojson; jv_dump stays compact");
}

// Sibling constraints intersect; they are never dropped (outside review,
// 2026-09-05: four ways a schema was silently weakened at compile time).
static snode *compile_src(const char *schema_src) {
    jv *j = json_parse(schema_src, strlen(schema_src));
    assert(j != NULL);
    char err[256] = "";
    snode *n = schema_compile(j, err, sizeof(err));
    jv_free(j);
    return n;
}
static bool compiles(const char *schema_src) {
    snode *n = compile_src(schema_src);
    if (n) schema_free(n);
    return n != NULL;
}
static bool accepts(const char *schema_src, const char *doc) {
    snode *n = compile_src(schema_src);
    assert(n && "schema must compile for an acceptance check");
    sval v; sval_init(&v, n);
    bool ok = sval_feed(&v, doc, (int)strlen(doc)) && v.done;
    schema_free(n);
    return ok;
}
static void test_schema_sibling_constraints_are_not_dropped(void) {
    // boolean false admits nothing: refused, never compiled to ANY
    assert(!compiles("false"));
    assert(!compiles("{\"type\":\"object\",\"properties\":{\"x\":false},"
                     "\"required\":[\"x\"]}"));
    assert(compiles("true"));
    // enum intersects type: the off-type value is dropped, not admitted
    // (numeric cases sit inside an object so the closing brace completes them)
    #define WRAP(inner) "{\"type\":\"object\",\"properties\":{\"v\":" inner "},\"required\":[\"v\"]}"
    const char *ie = WRAP("{\"type\":\"integer\",\"enum\":[\"bad\",1]}");
    assert(accepts(ie, "{\"v\":1}"));
    assert(!accepts(ie, "{\"v\":\"bad\"}"));
    assert(!compiles("{\"type\":\"integer\",\"enum\":[\"bad\"]}"));
    // const beside enum: must be a member, and then it is the only value
    assert(!compiles("{\"const\":1,\"enum\":[2]}"));
    const char *ce = WRAP("{\"const\":2,\"enum\":[1,2]}");
    assert(accepts(ce, "{\"v\":2}"));
    assert(!accepts(ce, "{\"v\":1}"));
    // Every scalar-const union alternative still carries its sibling
    // constraints. The literal-only merge must not bypass them.
    assert(!compiles("{\"anyOf\":[{\"type\":\"string\",\"const\":1},"
                     "{\"const\":\"x\"}]}"));
    assert(!compiles("{\"anyOf\":[{\"const\":1,\"minimum\":2},"
                     "{\"const\":\"x\"}]}"));
    // type beside oneOf/anyOf is refused rather than ignored
    assert(!compiles("{\"type\":\"string\",\"anyOf\":[{\"type\":\"string\"},"
                     "{\"type\":\"integer\"}]}"));
    assert(compiles("{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}"));
    // integer|null stays integer
    const char *in = WRAP("{\"type\":[\"integer\",\"null\"]}");
    assert(accepts(in, "{\"v\":1}"));
    assert(accepts(in, "{\"v\":null}"));
    assert(!accepts(in, "{\"v\":1.5}"));
    // number|integer|null is still a number
    assert(accepts(WRAP("{\"type\":[\"number\",\"integer\",\"null\"]}"), "{\"v\":1.5}"));
    #undef WRAP
}


// An enum as long as a function has lines: the anchored-edit protocol puts
// every distinct line of a function in one, and the walker used to keep one
// bit per literal, 64 at most. Prefix-shaped members ("ab" beside "abc"),
// shared prefixes, and the reconsume path all have to hold past that.
static void test_enum_past_sixty_literals(void) {
    enum { N = 300 };
    static char schema[N * 24 + 64];
    size_t o = (size_t)snprintf(schema, sizeof schema, "{\"enum\":[");
    for (int i = 0; i < N; i++) {
        // shared prefixes in blocks of ten; every tenth value is a prefix of the next
        o += (size_t)snprintf(schema + o, sizeof schema - o, "%s\"line %03d%s\"", i ? "," : "",
                              i / 10, (i % 10) ? (const char[]){'-', (char)('a' + i % 10), 0} : "");
    }
    snprintf(schema + o, sizeof schema - o, "]}");
    assert(compiles(schema));
    assert(accepts(schema, "\"line 000\""));        // a value that is a prefix of others
    assert(accepts(schema, "\"line 000-c\""));      // one that extends it
    assert(accepts(schema, "\"line 029-j\""));      // the last block
    assert(!accepts(schema, "\"line 030\""));       // past the set
    assert(!accepts(schema, "\"line 000-k\""));     // an extension that is not a member
    assert(!accepts(schema, "\"lin\""));            // a strict prefix of every member
    // inside an object with the shape the edit protocol uses
    static char obj[N * 24 + 256];
    snprintf(obj, sizeof obj, "{\"type\":\"object\",\"properties\":{\"line\":%s,\"text\":{\"type\":\"string\"}},"
                              "\"required\":[\"line\",\"text\"]}", schema);
    assert(accepts(obj, "{\"line\":\"line 017-d\",\"text\":\"x\"}"));
    assert(!accepts(obj, "{\"line\":\"line 017-z\",\"text\":\"x\"}"));
    // beyond the new cap is still a refusal, not a silent truncation
    static char huge[5000 * 12 + 64];
    o = (size_t)snprintf(huge, sizeof huge, "{\"enum\":[");
    for (int i = 0; i < 4097; i++) o += (size_t)snprintf(huge + o, sizeof huge - o, "%s\"v%04d\"", i ? "," : "", i);
    snprintf(huge + o, sizeof huge - o, "]}");
    assert(!compiles(huge));
}

// A close sized by the schema. The engine's closer used a fixed 4,096-byte
// buffer, so an accepted schema whose one enum member is 5,000 characters
// came back through the API as 4,097 characters with no closing quote
// (external review, 2026-09-27). The growable close fits it, and the
// document parses and conforms.
static void test_schema_close_fits_a_long_enum_member(void) {
    enum { LEN = 5000 };
    static char member[LEN + 1];
    memset(member, 'a', LEN);
    member[LEN] = 0;
    static char src[LEN + 128];
    snprintf(src, sizeof(src),
             "{\"type\":\"object\",\"properties\":{\"k\":{\"type\":\"string\","
             "\"enum\":[\"%s\"]}},\"required\":[\"k\"]}", member);
    jv *schema_json = json_parse(src, strlen(src));
    assert(schema_json != NULL);
    char err[128];
    snode *schema = schema_compile(schema_json, err, sizeof(err));
    assert(schema != NULL);

    sval v;
    sval_init(&v, schema);
    const char *head = "{\"k\":\"";
    assert(sval_feed(&v, head, (int)strlen(head)));
    int n = 0;
    char *tail = sval_close_alloc(&v, &n, SVAL_CLOSE_MAX);
    assert(tail != NULL && n > LEN);           // the member, the quote, the brace
    assert(tail[n - 1] == '}' && tail[n - 2] == '"');

    static char full[LEN + 256];
    snprintf(full, sizeof(full), "%s%s", head, tail);
    jv *parsed = json_parse(full, strlen(full));
    assert(parsed != NULL);
    assert(strcmp(jv_str(jv_get(parsed, "k"), ""), member) == 0);
    jv_free(parsed);
    sval chk; sval_init(&chk, schema);
    assert(sval_feed(&chk, full, (int)strlen(full)) && chk.done);

    // the fixed-buffer close keeps its old shape: cut, never overrun
    sval w;
    sval_init(&w, schema);
    assert(sval_feed(&w, head, (int)strlen(head)));
    char small[64];
    int m = sval_close(&w, small, sizeof(small));
    assert(m == (int)sizeof(small) - 1 && small[m] == 0);

    free(tail);
    schema_free(schema);
    jv_free(schema_json);
}

// An enum literal is matched byte by byte, so an empty one can never be
// completed: at the first byte the matcher would compare a generated NUL
// with its terminator and read past it. Every native grammar refuses a tool
// whose name is empty, except Harmony, which read "" as a name (it skips a
// tool with no name at all). An empty name is no name there too: the
// recipient enum holds only real names, and a NUL is refused at it.
static void test_harmony_empty_tool_name_is_no_name(void) {
    const char *tsrc =
        "[{\"type\":\"function\",\"function\":{\"name\":\"\"}},"
        "{\"type\":\"function\",\"function\":{\"name\":\"ping\"}}]";
    jv *tools = json_parse(tsrc, strlen(tsrc));
    assert(tools != NULL);
    char err[160];
    snode *g = schema_compile_harmony_turn(tools, false, NULL, NULL, false,
                                           err, sizeof(err));
    assert(g != NULL);
    sval v, scratch;
    sval_init(&v, g);
    // the required turn's header, up to the recipient (the tool name enum)
    const char *prefix = "<|channel|>commentary to=functions.";
    assert(sval_feed(&v, prefix, (int)strlen(prefix)));
    assert(sval_trial(&v, &scratch, "p", 1));
    assert(!sval_trial(&v, &scratch, "\0", 1));
    assert(!sval_trial(&v, &scratch, "<", 1));
    schema_free(g);
    jv_free(tools);

    const char *only_empty = "[{\"type\":\"function\",\"function\":{\"name\":\"\"}}]";
    tools = json_parse(only_empty, strlen(only_empty));
    assert(tools != NULL);
    err[0] = 0;
    g = schema_compile_harmony_turn(tools, false, NULL, NULL, false, err, sizeof(err));
    assert(g == NULL && strstr(err, "no function names"));
    jv_free(tools);
}

// The $ref inliner bounded the number of schema nodes it writes (20,000) and
// not their bytes: a definition's `enum` is copied whole at every reference,
// so a 2 MB request could ask for tens of gigabytes. Here a 100 KB enum
// referenced 800 times is 80 MB inlined, past the 64 MiB bound; the same
// definition referenced 8 times is fine.
// codex's read of PR F (fable, 2026-11-14): the path-id leniency read a
// nested resource that carries the pointer's own target against the root.
// Here the nested resource is https://example.test/p, so "#/definitions/T"
// under it is its own integer T; the root's string T is another schema. Base
// 53542f86 refused this schema; the leniency compiled the root's T, accepting
// {"p":{"x":"a"}} and rejecting {"p":{"x":1}}. Whatever compiles must be the
// schema the document names: a refusal holds that, the root's T never does.
static void test_a_path_id_resource_with_its_own_target_is_not_read_at_the_root(void) {
    const char *s =
        "{\"id\":\"https://example.test/root\",\"type\":\"object\","
        "\"definitions\":{\"T\":{\"type\":\"string\"}},"
        "\"properties\":{\"p\":{\"id\":\"/p\",\"type\":\"object\","
        "\"definitions\":{\"T\":{\"type\":\"integer\"}},"
        "\"properties\":{\"x\":{\"$ref\":\"#/definitions/T\"}},\"required\":[\"x\"]}},"
        "\"required\":[\"p\"]}";
    if (compiles(s)) {
        assert(accepts(s, "{\"p\":{\"x\":1}}") && "the nested resource's T is an integer");
        assert(!accepts(s, "{\"p\":{\"x\":\"a\"}}") && "the root's string T is not this schema's");
    }
}

static void test_inlined_refs_are_bounded_in_bytes(void) {
    sbuf b = {0};
    sb_lit(&b, "{\"$defs\":{\"A\":{\"enum\":[");
    for (int i = 0; i < 1000; i++) {
        sb_fmt(&b, "%s\"", i ? "," : "");
        for (int k = 0; k < 96; k++) sb_lit(&b, "x");
        sb_fmt(&b, "%04d\"", i);
    }
    sb_lit(&b, "]}},\"type\":\"object\",\"properties\":{");
    size_t head = b.n;
    for (int refs = 800; refs >= 8; refs /= 100) {
        b.n = head;
        for (int i = 0; i < refs; i++)
            sb_fmt(&b, "%s\"p%d\":{\"$ref\":\"#/$defs/A\"}", i ? "," : "", i);
        sb_lit(&b, "}}");
        assert(!b.failed);
        jv *s = json_parse(b.s, b.n);
        assert(s);
        char err[160] = "";
        jv *own = NULL;
        jv *r = schema_resolve_refs(s, &own, err, sizeof err);
        if (refs == 800) {
            assert(!r && !own);
            assert(strstr(err, "too large once its $refs are inlined") != NULL);
        } else {
            assert(r && r == own);
            jv_free(own);
        }
        jv_free(s);
    }
    free(b.s);
}

// The enum's sorted order (the validator's candidate range walks it) is the
// stable order by literal: equal literals in declared order. It was an
// insertion sort and is a library sort now; this holds the result to the
// insertion sort's on random literals with duplicates, at the size limit.
static void test_enum_order_is_the_stable_sorted_order(void) {
    enum { N = 4096 };
    snode n = {0};
    n.lits = malloc(sizeof(char *) * N);
    assert(n.lits);
    unsigned s = 20261112u;
    for (int i = 0; i < N; i++) {
        s = s * 1664525u + 1013904223u;
        char buf[32];
        snprintf(buf, sizeof buf, "\"k%u\"", (s >> 8) % 700);   // about six of each
        n.lits[i] = strdup(buf);
        assert(n.lits[i]);
    }
    n.n_lits = N;
    assert(enum_index(&n));
    int *want = malloc(sizeof(int) * N);
    assert(want);
    for (int i = 0; i < N; i++) {
        int k = i;
        while (k > 0 && strcmp(n.lits[want[k - 1]], n.lits[i]) > 0) {
            want[k] = want[k - 1];
            k--;
        }
        want[k] = i;
    }
    assert(memcmp(want, n.order, sizeof(int) * N) == 0);
    for (int i = 0; i < N; i++) free(n.lits[i]);
    free(n.lits); free(n.order); free(want);
}

// A pattern's fixed counts are summed into an int (its shortest and longest
// string). Two counts of two billion overflowed it, which is undefined: the
// sanitizer build stopped at pattern_max_len, the release build compiled a
// node with a wrapped length. Such a pattern is refused by name.
static void test_pattern_counts_past_int_are_refused(void) {
    static const char *const bad[] = {
        "{\"type\":\"string\",\"pattern\":\"^[a-z]{2000000000}[0-9]{2000000000}$\"}",
        "{\"type\":\"string\",\"pattern\":\"^[a-z]{2147483647}x$\"}",
        "{\"type\":\"string\",\"pattern\":\"^[a-z]{1000000000}[0-9]{1000000000}[a-z]{1,2147483647}$\"}",
    };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        jv *j = json_parse(bad[i], strlen(bad[i]));
        char err[320] = "";
        assert(j && !schema_compile(j, err, sizeof err));
        assert(strstr(err, "longer than") != NULL);
        jv_free(j);
    }
    // the largest count alone still compiles
    const char *ok = "{\"type\":\"string\",\"pattern\":\"^[a-z]{2147483647}$\"}";
    jv *j = json_parse(ok, strlen(ok));
    char err[320] = "";
    snode *n = schema_compile(j, err, sizeof err);
    assert(n);
    schema_free(n);
    jv_free(j);
}

// Patterns conjoined by allOf are checked for a position where they
// contradict. The check walked every byte up to the shortest string they
// jointly require, so two patterns with a count of 200,000,000 took 28 s to
// compile, from a request of a hundred bytes. What the patterns allow only
// changes at a segment boundary, so those positions are the ones checked.
static void test_allof_patterns_are_checked_by_segment_not_by_byte(void) {
    static const struct { const char *schema; bool ok; } cases[] = {
        { "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[a-z]{200000000}$\"},"
          "{\"pattern\":\"^[a-m]+$\"}]}", true },
        // a contradiction far in: bytes 50000000 and 50000001 are a digit to
        // the first pattern and a letter to the second
        { "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[a-z]{50000000}[0-9]{5}$\"},"
          "{\"pattern\":\"^[a-z]{50000002}[0-9]{3}$\"}]}", false },
        // the same lengths, agreeing
        { "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[a-z]{50000000}[0-9]{5}$\"},"
          "{\"pattern\":\"^[a-z]{50000000}[0-9]{2}[0-9]{3}$\"}]}", true },
        // a literal against a class that excludes it, past a long run
        { "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[a-z]{90000000}X[a-z]+$\"},"
          "{\"pattern\":\"^[a-z]+$\"}]}", false },
        { "{\"type\":\"string\",\"allOf\":[{\"pattern\":\"^[a-z]{90000000}x[a-z]+$\"},"
          "{\"pattern\":\"^[a-z]+$\"}]}", true },
    };
    clock_t t0 = clock();
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        jv *j = json_parse(cases[i].schema, strlen(cases[i].schema));
        char err[320] = "";
        snode *n = schema_compile(j, err, sizeof err);
        if (cases[i].ok) assert(n);
        else assert(!n && strstr(err, "cannot be satisfied together") != NULL);
        schema_free(n);
        jv_free(j);
    }
    assert((double)(clock() - t0) / CLOCKS_PER_SEC < 1.0);
}


// R2.18: a pattern the segment compiler cannot express compiles to the
// automaton (src/jsre.c) and is enforced as ECMA-262 reads it, its lengths
// exact; a closed document validates; limits end a request by name.
static snode *pat_compile(const char *src, char *err, int cap) {
    jv *j = json_parse(src, strlen(src));
    assert(j != NULL);
    snode *n = schema_compile(j, err, cap);
    jv_free(j);
    return n;
}
static bool pat_whole(const snode *s, const char *doc) {
    sval v; sval_init(&v, s);
    return sval_feed(&v, doc, (int)strlen(doc)) && v.done;
}
static bool pat_prefix(const snode *s, const char *doc) {
    sval v; sval_init(&v, s);
    return sval_feed(&v, doc, (int)strlen(doc));
}
// a forced close from `prefix` is a document the same schema accepts
static void pat_closes(const snode *s, const char *prefix) {
    sval p; sval_init(&p, s);
    assert(sval_feed(&p, prefix, (int)strlen(prefix)));
    char suffix[4096];
    int k = sval_close(&p, suffix, sizeof(suffix));
    assert(k > 0);
    char full[8192];
    snprintf(full, sizeof(full), "%s%s", prefix, suffix);
    assert(pat_whole(s, full));
}

static void test_schema_pattern_automaton(void) {
    char err[256];
    // searched, as ECMA searches: an unanchored pattern matches anywhere
    snode *s = pat_compile("{\"type\":\"string\",\"pattern\":\"b(c|d)\"}", err, sizeof(err));
    assert(s && s->re && !s->n_pats);
    assert(pat_whole(s, "\"abce\"") && pat_whole(s, "\"bd\"") && !pat_whole(s, "\"bx\""));
    // no escape under a pattern: its value is its text (b is a b)
    assert(!pat_prefix(s, "\"a\\u0062"));
    pat_closes(s, "\"");
    pat_closes(s, "\"xb");
    schema_free(s);
    // code points, not bytes
    s = pat_compile("{\"type\":\"string\",\"pattern\":\"^.{2}$\"}", err, sizeof(err));
    assert(s && s->re);
    assert(pat_whole(s, "\"\xc3\xa9\xf0\x9f\x98\x80\""));     // two code points
    assert(!pat_whole(s, "\"\xf0\x9f\x98\x80\""));          // one
    assert(!pat_whole(s, "\"abc\""));
    pat_closes(s, "\"\xc3");                                // inside a code point
    pat_closes(s, "\"a");
    schema_free(s);
    // lengths are exact: ^(ab)+$ has strings of 2, 4, 6 ... and none of 3
    assert(!pat_compile("{\"type\":\"string\",\"pattern\":\"^(ab)+$\",\"minLength\":3,\"maxLength\":3}", err, sizeof(err)));
    assert(strstr(err, "no string between minLength and maxLength"));
    s = pat_compile("{\"type\":\"string\",\"pattern\":\"^(ab)+$\",\"minLength\":3,\"maxLength\":5}", err, sizeof(err));
    assert(s && s->re);
    assert(pat_whole(s, "\"abab\"") && !pat_whole(s, "\"ab\"") && !pat_whole(s, "\"ababab\""));
    // the prefix ababa has no completion of 5 or fewer: refused on its byte
    assert(pat_prefix(s, "\"abab") && !pat_prefix(s, "\"ababa"));
    pat_closes(s, "\"");
    pat_closes(s, "\"aba");
    schema_free(s);
    // a minLength beyond what a bounded pattern reaches is refused at compile
    assert(!pat_compile("{\"type\":\"string\",\"pattern\":\"^(a|b){2}$\",\"minLength\":3}", err, sizeof(err)));
    // refused by name
    assert(!pat_compile("{\"type\":\"string\",\"pattern\":\"a(?=b)\"}", err, sizeof(err)) && strstr(err, "lookahead"));
    assert(!pat_compile("{\"type\":\"string\",\"pattern\":\"(a)\\\\1\"}", err, sizeof(err)) && strstr(err, "backreference"));
    assert(!pat_compile("{\"type\":\"string\",\"pattern\":\"^\\\"$\"}", err, sizeof(err)) && strstr(err, "no string JSON can spell"));
    // one automaton a string: a second pattern beside it is refused by name
    assert(!pat_compile("{\"type\":\"string\",\"pattern\":\"^(a|b)+$\",\"allOf\":[{\"pattern\":\"^a\"}]}", err, sizeof(err)));
    assert(strstr(err, "more than one pattern"));
    // a pattern the segments take keeps their path
    s = pat_compile("{\"type\":\"string\",\"pattern\":\"^[a-z]{2,5}$\"}", err, sizeof(err));
    assert(s && s->n_pats == 1 && !s->re);
    schema_free(s);
    // a trial and a feed agree; the trial's copy says it is one, and keeps
    // nothing; the pool is made on the first byte, on the root
    s = pat_compile("{\"type\":\"object\",\"properties\":{\"k\":{\"type\":\"string\",\"pattern\":\"^(x|y)+z$\"}},\"required\":[\"k\"]}", err, sizeof(err));
    assert(s && !s->pool);
    sval v; sval_init(&v, s);
    assert(sval_feed(&v, "{\"k\":\"xy", 8) && !v.trial && v.root == s && s->pool);
    sval t;
    assert(sval_trial(&v, &t, "yz\"}", 4) && t.trial && t.done);
    assert(!sval_trial(&v, &t, "\"}", 2));
    assert(sval_feed(&v, "z\"}", 3) && v.done);
    schema_free(s);
    // the answers do not depend on the cache: the same documents under a
    // one-byte budget (RUNNER_TEST_JSRE_POOL_BYTES, read when a pool is made)
    {
        const char *src = "{\"type\":\"string\",\"pattern\":\"^([a-f0-9]{2}:){2}[a-f0-9]{2}$\"}";
        const char *docs[] = { "\"0a:1b:2c\"", "\"0a:1b:2\"", "\"0a-1b:2c\"", "\"ff:ff:ff\"", "\"0A:1b:2c\"" };
        bool want[5];
        s = pat_compile(src, err, sizeof(err));
        for (int i = 0; i < 5; i++) want[i] = pat_whole(s, docs[i]);
        schema_free(s);
        assert(want[0] && !want[1] && !want[2] && want[3] && !want[4]);
        setenv_compat("RUNNER_TEST_JSRE_POOL_BYTES", "1");
        s = pat_compile(src, err, sizeof(err));
        for (int i = 0; i < 5; i++) assert(pat_whole(s, docs[i]) == want[i]);
        pat_closes(s, "\"0a:");
        schema_free(s);
        setenv_compat("RUNNER_TEST_JSRE_POOL_BYTES", "");
    }
    // the kept-state limit ends the walk by name: which pattern, how many
    // states (RUNNER_TEST_JSRE_KEPT_MAX lowers it, so a few bytes reach it)
    {
        const char *src = "{\"type\":\"string\",\"pattern\":\"^(a|b){0,30}c?$\"}";
        s = pat_compile(src, err, sizeof(err));
        char msg[400];
        sval f; sval_init(&f, s);
        assert(sval_feed(&f, "\"ababababab\"", 12) && f.done && !sval_pattern_fault(&f, msg, sizeof(msg)));
        schema_free(s);
        setenv_compat("RUNNER_TEST_JSRE_KEPT_MAX", "4");
        s = pat_compile(src, err, sizeof(err));
        sval g; sval_init(&g, s);
        assert(!sval_feed(&g, "\"ababababab", 11));
        assert(sval_pattern_fault(&g, msg, sizeof(msg)));
        assert(strstr(msg, "^(a|b){0,30}c?$") && strstr(msg, "more states than one request may keep") &&
               strstr(msg, "4 states kept"));
        schema_free(s);
        setenv_compat("RUNNER_TEST_JSRE_KEPT_MAX", "");
    }
    printf("ok: pattern automaton strings (R2.18): searched, code points, exact lengths, closes, limits by name\n");
}


// R2.18 (codex on #875): a pattern the segment compiler keeps must mean what
// ECMA-262 means. It read \s and \S as ASCII white space, so ^\S+$ admitted
// U+00A0 and ^[\s]+$ refused it; it read [\.-_] as three characters where
// ECMA has a range; [] opening a set the POSIX way; and strtol let {2, 3},
// { 2} and {+2} count where ECMA reads text. Each now goes to the automaton,
// which reads it as ECMA does; what stays on the segment path is exact.

// R2.18 (codex on #875, point 5): ECMA's verdicts (tests/jsre_cases.h) on the
// path a schema actually takes, through schema_compile and sval_feed, so a
// pattern the segment compiler keeps is held to ECMA as the automaton is.
// A string JSON cannot spell without an escape is refused under a pattern.
static void test_schema_ecma_cases_through_the_validator(void) {
    int n = (int)(sizeof JSRE_CASES / sizeof *JSRE_CASES), on_seg = 0, on_auto = 0, wrong = 0;
    const char *cur = NULL;
    snode *s = NULL;
    char src[4096];
    for (int i = 0; i < n; i++) {
        const jsre_case *c = &JSRE_CASES[i];
        if (!cur || strcmp(cur, c->pat)) {
            schema_free(s);
            cur = c->pat;
            int k = snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"");
            for (const unsigned char *q = (const unsigned char *)c->pat; *q && k < (int)sizeof(src) - 8; q++) {
                if (*q == '"' || *q == '\\') { src[k++] = '\\'; src[k++] = (char)*q; }
                else if (*q < 0x20) k += snprintf(src + k, sizeof(src) - (size_t)k, "\\u%04x", *q);
                else src[k++] = (char)*q;
            }
            snprintf(src + k, sizeof(src) - (size_t)k, "\"}");
            jv *j = json_parse(src, strlen(src));
            char err[256];
            s = j ? schema_compile(j, err, sizeof(err)) : NULL;
            jv_free(j);
            if (!s) { fprintf(stderr, "ECMA case: %s refused by the schema compiler: %s\n", c->pat, err); wrong++; continue; }
        }
        if (!s) continue;
        bool spellable = true;
        for (int b = 0; b < c->n; b++) if ((unsigned char)c->s[b] < 0x20 || c->s[b] == '"' || c->s[b] == '\\') spellable = false;
        sval v; sval_init(&v, s);
        bool got = spellable && sval_feed(&v, "\"", 1) && sval_feed(&v, c->s, c->n) && sval_feed(&v, "\"", 1) && v.done;
        if (got != (bool)c->want) {
            wrong++;
            if (wrong < 8) fprintf(stderr, "ECMA case: %s on string #%d: the validator says %d, ECMA %d (%s path)\n",
                                   c->pat, i, got, c->want, s->n_pats ? "segment" : "automaton");
        }
        if (s->n_pats) on_seg++; else on_auto++;
    }
    schema_free(s);
    assert(wrong == 0);
    assert(on_seg > 20 && on_auto > 200);   // both paths answered
    printf("ok: %d ECMA verdicts through the validator (%d on the segment path, %d on the automaton)\n",
           on_seg + on_auto, on_seg, on_auto);
}

static void test_schema_segment_classes_mean_what_ecma_means(void) {
    struct { const char *pat, *doc; bool ok; } cases[] = {
        { "^\\\\S+$", "\"ab\"", true },
        { "^\\\\S+$", "\"\xc2\xa0\"", false },            // U+00A0 is ECMA white space
        { "^\\\\S+$", "\"\xe3\x80\x80\"", false },        // U+3000
        { "^\\\\S+$", "\"\xc3\xa9\"", true },             // é is not
        { "^[^\\\\s]+$", "\"\xe2\x80\xa8\"", false },     // U+2028 is a line terminator
        { "^[^\\\\s]+$", "\"\xc3\xa9\"", true },
        { "^[\\\\s]+$", "\"\xc2\xa0\"", true },
        { "^[0-9\\\\s]+$", "\"1\xe3\x80\x80\"", true },
        { "^[\\\\w\\\\.-_]+$", "\":\"", true },             // . to _ is a range
        { "^[\\\\w\\\\.-_]+$", "\"-\"", false },             // and - is below it
        { "^[\\\\w\\\\.-_]+$", "\"a.b\"", true },
        { "^a[b]{ 2}$", "\"ab{ 2}\"", true },              // text, as legacy ECMA reads it
        { "^a[b]{ 2}$", "\"abb\"", false },
        { "^a[b]{2, 3}$", "\"ab{2, 3}\"", true },
        { "^a[b]{2, 3}$", "\"abb\"", false },
        { "^a[b]{+2}$", "\"ab{2}\"", true },               // legacy: a { repeated by +, then 2}
        { "^a[b]{+2}$", "\"ab{{2}\"", true },
        { "^a[b]{+2}$", "\"ab{+2}\"", false },
        { "^a[b]{+2}$", "\"abb\"", false },
        { "^[^\\\\n\\\\r]*$", "\"\xe2\x80\xa8\"", true },  // [^\n\r] holds U+2028
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        char src[256];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}", cases[i].pat);
        jv *j = json_parse(src, strlen(src));
        assert(j != NULL);
        char err[256];
        snode *s = schema_compile(j, err, sizeof(err));
        jv_free(j);
        assert(s != NULL);
        sval v; sval_init(&v, s);
        bool got = sval_feed(&v, cases[i].doc, (int)strlen(cases[i].doc)) && v.done;
        if (got != cases[i].ok) {
            fprintf(stderr, "segment class: %s on %s: got %d want %d\n", cases[i].pat, cases[i].doc, got, cases[i].ok);
            assert(got == cases[i].ok);
        }
        schema_free(s);
    }
    // [] is ECMA's empty class: a pattern that matches nothing is refused
    {
        const char *src = "{\"type\":\"string\",\"pattern\":\"^[]a]+$\"}";
        jv *j = json_parse(src, strlen(src));
        char err[256];
        assert(schema_compile(j, err, sizeof(err)) == NULL);
        jv_free(j);
    }
    // the shapes the segments express exactly stay on their path
    const char *kept[] = { "^wf_[a-z0-9-]{6,}$", "^\\\\d{5}$", "^[A-Z]{3}[0-9]{4}$", "^[^\\\\n\\\\r]*$", "^[\\\\w-]+$" };
    for (size_t i = 0; i < sizeof(kept) / sizeof(*kept); i++) {
        char src[256];
        snprintf(src, sizeof(src), "{\"type\":\"string\",\"pattern\":\"%s\"}", kept[i]);
        jv *j = json_parse(src, strlen(src));
        char err[256];
        snode *s = schema_compile(j, err, sizeof(err));
        jv_free(j);
        assert(s && s->n_pats == 1 && !s->re);
        schema_free(s);
    }
    // R2.18 (codex on #875): the largest maxLength compile_bound takes,
    // 2147483647, without an overflow (test-schema-san runs this under UBSan)
    {
        const char *src = "{\"type\":\"string\",\"pattern\":\"^(a|b)+$\",\"maxLength\":2147483647}";
        jv *j = json_parse(src, strlen(src));
        char err[256];
        snode *s = schema_compile(j, err, sizeof(err));
        jv_free(j);
        assert(s && s->re);
        assert(pat_whole(s, "\"abab\""));
        pat_closes(s, "\"a");
        schema_free(s);
        src = "{\"type\":\"string\",\"pattern\":\"^(a|b)+$\",\"minLength\":2147483646,\"maxLength\":2147483647}";
        j = json_parse(src, strlen(src));
        s = schema_compile(j, err, sizeof(err));
        jv_free(j);
        assert(s && s->re);
        assert(pat_prefix(s, "\"abab"));
        sval c; sval_init(&c, s);
        assert(sval_feed(&c, "\"ab", 3));
        char tail[64];
        (void)sval_close(&c, tail, sizeof(tail));   // too long to finish: stops short, without overflow
        schema_free(s);
    }
    printf("ok: segment patterns mean what ECMA means; the largest maxLength compiles and closes\n");
}

int main(void) {
    test_schema_sibling_constraints_are_not_dropped();
    test_strict_bounded_numbers();
    test_json_rejects_unpaired_utf16_surrogates();
    test_json_rejects_ill_formed_raw_utf8();
    test_json_rejects_embedded_nul();
    test_json_rejects_duplicate_object_keys();
    test_escapes_are_paired_and_closable();
    test_json_close_partial_string();
    test_json_close_leaves_an_opened_array_empty();
    test_schema_required_close();
    test_schema_close_fits_a_long_enum_member();
    test_map_refuses_a_duplicate_key_spelling();
    test_required_whitespace_reports_as_content();
    test_a_native_call_cannot_close_with_a_required_argument_missing();
    test_leading_whitespace_is_refused_but_interior_is_kept();
    test_json_mode_leading_whitespace_is_refused();
    test_schema_close_without_payload_fabricates_nothing();
    test_schema_rejects_empty_type_union();
    test_schema_huge_min_bounds_terminate();
    test_schema_rejects_bad_bounds();
    test_schema_rejects_non_integer_or_huge_bounds();
    test_schema_rejects_escaped_keys();
    test_schema_rejects_unenforceable_keywords();
    test_schema_rejects_misspelled_keyword_types();
    test_schema_type_array_with_open_object();
    test_truncation_closes_to_a_valid_instance();
    test_schema_bounded_repetition();
    test_schema_multi_segment_pattern();
    test_schema_agent_id_pattern_is_enforced();
    test_schema_pattern_shorthand_classes();
    test_schema_pattern_regex_syntax_is_rejected_not_reinterpreted();
    test_schema_pattern_automaton();
    test_schema_segment_classes_mean_what_ecma_means();
    test_schema_ecma_cases_through_the_validator();
    test_schema_allof_of_string_patterns();
    test_schema_negated_and_escaped_classes();
    test_schema_number_bounds_reject_dead_minus_and_close_in_bounds();
    test_schema_exclusive_bounds_survive_flush_to_zero();
    test_schema_number_prefix_stays_completable();
    test_schema_number_prefix_fits_the_buffer();
    test_schema_number_witness_work_is_bounded();
    test_schema_number_cases_it_got_wrong();
    test_schema_merges_enum_and_const_anyof();
    test_dump_keeps_integer_literals();
    test_schema_enum_keeps_a_large_integer();
    test_schema_one_of_admits_exactly_one_branch();
    test_schema_const_and_enum_compare_large_integers_exactly();
    test_schema_numbers_compare_by_value_not_spelling();
    test_schema_large_doubles_compare_by_exact_digits();
    test_schema_integer_bounds_are_enforced();
    test_schema_integer_rejects_leading_zeros();
    test_schema_number_bounds_intersect_their_exclusive_twins();
    test_schema_close_never_drops_its_closers();
    test_schema_close_counts_the_escape_it_finishes();
    test_schema_integer_bounds_complete_truncation();
    test_schema_number_close_rescues_an_out_of_range_prefix();
    test_jsonv_utf8_matches_parser();
    test_duplicate_keys_match_parser();
    test_key_guard_capacity_fails_closed();
    test_schema_number_matches_parser();
    test_schema_number_bounds_are_enforced();
    test_schema_number_bounds_across_frames();
    test_sval_state_is_small();
    test_sval_trial_matches_full_copy();
    test_schema_additional_properties();
    test_schema_empty_closed_object();
    test_schema_rejects_required_without_properties();
    test_schema_accepts_annotation_keywords();
    test_schema_accepts_claude_open_metadata_object();
    test_schema_typed_additional_properties_map();
    test_schema_oneof_const_scalars();
    test_schema_oneof_const_numeric_prefixes();
    test_schema_rejects_oversized_oneof_const_scalars();
    test_schema_string_length_bounds_close_and_reject();
    test_schema_string_length_counts_code_points();
    test_schema_string_length_rejects_at_one_code_point();
    test_schema_utf8_boundary_has_a_legal_continuation();
    test_schema_rejects_unpaired_surrogates();
    test_schema_minlength_close_after_multibyte_content();
    test_schema_close_finishes_multibyte_map_key();
    test_schema_string_minlength_full_close();
    test_schema_discriminated_action_args();
    test_schema_union_dispatch_rules();
    test_schema_discriminated_required_may_differ();
    test_schema_const_discriminated_union();
    test_schema_long_string_value();
    test_schema_escape_at_maxlength_rejected();
    test_schema_nested_tool_key_keeps_discriminator();
    test_schema_close_mid_discriminator_matches_args();
    test_schema_rejects_discriminator_after_conditional_args();
    test_enum_past_sixty_literals();
    test_harmony_empty_tool_name_is_no_name();
    test_escape_replaces_ill_formed_utf8();
    test_tojson_dump_matches_jinja();
    test_inlined_refs_are_bounded_in_bytes();
    test_a_path_id_resource_with_its_own_target_is_not_read_at_the_root();
    test_enum_order_is_the_stable_sorted_order();
    test_pattern_counts_past_int_are_refused();
    test_allof_patterns_are_checked_by_segment_not_by_byte();
    puts("json/schema tests ok");
    return 0;
}
