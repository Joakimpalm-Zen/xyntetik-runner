// libFuzzer harness for the tool-call demultiplexer (tool_stream_* in
// src/template.c): the parser that reads a model's own output and decides
// which bytes are prose and which are a call. Its input is whatever a model
// writes, which no grammar bounds on a parse-only family, so it is untrusted
// in the same sense a request body is.
//
// Input: byte 0 picks the template family (the envelope is built the way a
// request builds it, tool_envelope_build then tool_decl_native, so only
// states production reaches are fuzzed); byte 1 holds flags (strict,
// atem tool calling, truncated); byte 2 the chunk size; the rest is the
// model's turn.
//
// One property with an anchor outside the parser's own tables: a stream is a
// stream. The same turn fed one byte at a time and fed in chunks must give
// the client the same reasoning, content, call names, arguments and result.
// A split in the middle of a tag or an escape that changes the outcome is a
// finding, and so is any sanitizer report.
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "template.h"

static const char TOOLS[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"save\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"text\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"}},"
    "\"required\":[\"text\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"city\":{\"type\":\"string\"}}}}}]";

typedef struct { sbuf reasoning, content, names, args; int begins, ends; } turn_log;

static int on_reasoning(void *ud, const char *b, int n) { sb_put(&((turn_log *)ud)->reasoning, b, (size_t)n); return 0; }
static int on_content(void *ud, const char *b, int n) { sb_put(&((turn_log *)ud)->content, b, (size_t)n); return 0; }
static int on_begin(void *ud, const char *name) {
    turn_log *l = ud; l->begins++;
    sb_put(&l->names, name, strlen(name)); sb_put(&l->names, "\n", 1); return 0;
}
static int on_args(void *ud, const char *b, int n) { sb_put(&((turn_log *)ud)->args, b, (size_t)n); return 0; }
static int on_end(void *ud) { ((turn_log *)ud)->ends++; return 0; }

static void log_free(turn_log *l) {
    free(l->reasoning.s); free(l->content.s); free(l->names.s); free(l->args.s);
}

// One pass over the turn in chunks of `step`. Returns the finish result, or
// the first non-zero feed result plus 100 so the two cannot be confused.
static int run(const tool_envelope *e, const char *doc, size_t n, size_t step,
               bool truncated, turn_log *l, bool *called) {
    memset(l, 0, sizeof *l);
    tool_stream_sink sink = { l, on_reasoning, on_content, on_begin, on_args, on_end };
    tool_stream s;
    tool_stream_init(&s, e, &sink);
    int rc = 0;
    for (size_t i = 0; i < n && rc == 0; i += step) {
        size_t k = n - i < step ? n - i : step;
        int r = tool_stream_feed(&s, doc + i, (int)k);
        if (r) rc = 100 + r;
    }
    if (rc == 0) rc = tool_stream_finish_ex(&s, truncated);
    *called = tool_stream_called(&s);
    tool_stream_free(&s);
    return rc;
}

static bool same(const sbuf *a, const sbuf *b) {
    return a->n == b->n && (a->n == 0 || memcmp(a->s, b->s, a->n) == 0);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 4 || size > 8192) return 0;
    int tmpl = data[0] % TMPL_COUNT;
    bool strict = data[1] & 1, atem = data[1] & 2, truncated = data[1] & 4;
    size_t step = (size_t)(data[2] % 31) + 2;      // 2..32; the other pass is 1
    size_t n = size - 3;
    // exact-size heap copy: ASan's redzone sits right after the last byte
    char *doc = malloc(n);
    if (!doc) return 0;
    memcpy(doc, data + 3, n);

    jv *tools = json_parse(TOOLS, sizeof TOOLS - 1);
    assert(tools);
    tool_envelope e = {0};
    char err[192];
    int built = tool_envelope_build(tools, NULL, NULL, &e, err, sizeof err);
    if (built >= 0) {
        bool skip = false;
        tool_decl_native(tmpl, strict && built == 1, atem, tools, &e, &skip);
        turn_log a, b;
        bool called_a = false, called_b = false;
        int ra = run(&e, doc, n, 1, truncated, &a, &called_a);
        int rb = run(&e, doc, n, step, truncated, &b, &called_b);
        bool oom = a.reasoning.failed || a.content.failed || a.names.failed || a.args.failed ||
                   b.reasoning.failed || b.content.failed || b.names.failed || b.args.failed;
        if (!oom) {
            assert(ra == rb);
            assert(called_a == called_b);
            assert(a.begins == b.begins && a.ends == b.ends);
            assert(same(&a.reasoning, &b.reasoning));
            assert(same(&a.content, &b.content));
            assert(same(&a.names, &b.names));
            assert(same(&a.args, &b.args));
        }
        log_free(&a); log_free(&b);
        // the buffered mappers read the same bytes; they must not crash on them
        sbuf content = {0}, tc = {0};
        int n_calls = 0;
        if (e.parse_only) tool_stream_map(&e, doc, n, truncated, &content, &tc, &n_calls);
        else tool_envelope_map(&e, doc, n, &content, &tc);
        free(content.s); free(tc.s);
    }
    tool_envelope_free(&e);
    jv_free(tools);
    free(doc);
    return 0;
}
