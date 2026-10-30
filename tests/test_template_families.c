// Every chat-template family's render contract, family by family.
//
// tests/test_template.c pins the bytes of chosen renders, and
// scripts/template-conformance.py compares them with each publisher's own
// template. This file pins what holds for EVERY family in the switch at once,
// so that splitting render_messages_with_tools() into one renderer per family
// cannot drop a family, a flag or a capacity rule on the way:
//
//   - an empty conversation renders, with the generation prompt and without;
//   - the tools argument puts a declaration in the prompt exactly for the
//     families that render one themselves (the rest get tools from the
//     caller's envelope or system fold), and its absence leaves none;
//   - thinking on and off change the prompt exactly for the families that
//     read them;
//   - the tool protocol each family is mapped to;
//   - capacity: at every buffer size from 1 byte to past the full length, the
//     render never writes past the buffer, always leaves a NUL inside it,
//     writes a prefix of the full render, reports a size the callers read as
//     "grow" whenever it did not fit, and is exact once it fits.
//
// The table is the behaviour at the start of the extraction. A deliberate
// change to a family's rendering updates its row in the same commit.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "template.h"
#include "tokenizer.h"

static int fails, checks;

#define CHECK(cond, ...) do {                                   \
        checks++;                                               \
        if (!(cond)) {                                          \
            fails++;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);\
            fprintf(stderr, __VA_ARGS__);                       \
            fputc('\n', stderr);                                \
        }                                                       \
    } while (0)

// Indexed by template id. Columns: the empty conversation's length without
// and with the generation prompt; whether the tools argument renders a
// declaration; whether THINK_ON and THINK_OFF change a one-user-turn prompt
// against THINK_DEFAULT; the tool protocol and whether it is native.
static const struct {
    const char *name;
    size_t empty, empty_gen;
    bool tools_in_prompt, think_on, think_off;
    const char *protocol;
    bool native;
} FAMILY[] = {
    { "chatml",           0,  24, 1, 0, 0, "qwen_json",     1 },
    { "llama2",           0,   0, 0, 0, 0, "generic",       0 },
    { "llama3",           0,  49, 0, 0, 0, "generic",       0 },
    { "zephyr",           0,  16, 0, 0, 0, "generic",       0 },
    { "gemma",            0,  23, 0, 0, 0, "generic",       0 },
    { "gemma4",           0,  15, 1, 1, 0, "gemma4",        1 },
    { "mistral",          0,   0, 1, 0, 0, "mistral_json",  1 },
    { "phi3",            15,  16, 0, 0, 0, "generic",       0 },
    { "apertus",         92, 113, 1, 0, 0, "generic",       0 },
    { "ornith",           0,  34, 0, 0, 1, "qwen3_xml",     1 },
    { "muse",           170, 190, 1, 1, 1, "atem",          1 },
    { "granite",          0,  43, 0, 0, 0, "generic",       0 },
    { "harmony",        234, 284, 1, 0, 1, "harmony",       1 },
    { "raw",              0,   0, 0, 0, 0, "generic",       0 },
    { "chatml-think",     0,  24, 1, 0, 1, "qwen_json",     1 },
    { "mistral-v1",       0,   0, 0, 0, 0, "generic",       0 },
    { "mistral-nemo",     0,   0, 1, 0, 0, "mistral_json",  1 },
    { "gemma4-mainline",  0,  45, 1, 1, 0, "gemma4",        1 },
    { "granite42",       34,  66, 1, 0, 1, "qwen3_xml",     1 },
    { "nemotron35",      34,  66, 1, 0, 1, "qwen3_xml",     1 },
    { "qwen38",         239, 271, 1, 0, 1, "qwen3_xml",     1 },
    { "qwen3-coder",      0,  24, 1, 0, 0, "qwen3_xml",     1 },
    { "qwen35",           0,  34, 0, 0, 1, "qwen3_xml",     1 },
    { "qwen35-nothink",   0,  45, 0, 1, 0, "qwen3_xml",     1 },
    { "phi4",            15,  15, 0, 0, 0, "generic",       0 },
    { "nemotron",        24,  58, 1, 0, 1, "nemotron_json", 1 },
    { "hermes4",         75,  99, 1, 1, 0, "qwen_json",     1 },
    { "granite4",         0,  43, 1, 0, 0, "qwen_json",     1 },
    // TMPL_LLAMA2_FALLBACK: what detection returns when nothing matched
    { "llama2-fallback",  0,   0, 0, 0, 0, "generic",       0 },
    { "mellum",           0,  24, 1, 0, 1, "qwen_json",     1 },
    // MiniCPM5: ON pre-seeds '<think>\n', OFF closes an empty block, so both
    // change a one-turn prompt; its XML tool protocol is its own
    { "minicpm5",         0,  24, 1, 1, 1, "minicpm5_xml",  1 },
    // Trinity: chatml's contract; its blank lines are history-only
    { "trinity",          0,  24, 1, 0, 0, "qwen_json",     1 },
    // LFM2: tools are a line of its system turn; thinking is not read (the
    // Instruct template has no switch); its Python call list is its own
    { "lfm2",             0,  24, 1, 0, 0, "lfm2_python",   1 },
};
#define N_FAMILY ((int)(sizeof FAMILY / sizeof *FAMILY))

static const char *TOOL_SRC =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"description\":\"Current weather for a city.\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
    "\"required\":[\"city\"]}}}]";
static const char *CALLS_SRC =
    "[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"}},"
    "{\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"arguments\":\"{\\\"city\\\":\\\"Bergen\\\"}\"}}]";

static char A[1 << 16], B[1 << 16];

static size_t render(int t, const chat_msg *m, int n, bool gen, int think,
                     const jv *tools, char *out, size_t cap) {
    return render_messages_with_tools(t, m, n, gen, think, tools, out, cap);
}

// A conversation that reaches most of a renderer's branches: system, user,
// an assistant turn with two calls in the family's own protocol, both
// results, a closing assistant turn and a new user turn.
typedef struct {
    chat_msg m[8];
    int n;
    sbuf calls_text, r1, r2;
    jv *calls;
} conv;

static void conv_build(conv *c, int t) {
    memset(c, 0, sizeof *c);
    c->calls = json_parse(CALLS_SRC, strlen(CALLS_SRC));
    const char *turn_name = NULL;   // borrowed from c->calls
    assistant_calls_render(t, NULL, c->calls, &c->calls_text, &turn_name);
    const char *role1 = tool_result_wrap_id(t, "{\"temp\":12}", "call_1", &c->r1);
    if (!role1) role1 = tool_result_wrap(t, "{\"temp\":12}", &c->r1);
    const char *role2 = tool_result_wrap_id(t, "{\"temp\":9}", "call_2", &c->r2);
    if (!role2) role2 = tool_result_wrap(t, "{\"temp\":9}", &c->r2);
    chat_msg *m = c->m;
    m[c->n++] = (chat_msg){ .role = "system", .content = "Use the tools." };
    m[c->n++] = (chat_msg){ .role = "user", .content = "Weather in Oslo and Bergen?" };
    m[c->n++] = (chat_msg){ .role = "assistant",
                            .content = c->calls_text.s ? c->calls_text.s : "",
                            .name = turn_name };
    m[c->n++] = (chat_msg){ .role = role1, .content = c->r1.s ? c->r1.s : "",
                            .name = "get_weather" };
    m[c->n++] = (chat_msg){ .role = role2, .content = c->r2.s ? c->r2.s : "",
                            .name = "get_weather" };
    m[c->n++] = (chat_msg){ .role = "assistant", .content = "Oslo 12, Bergen 9." };
    m[c->n++] = (chat_msg){ .role = "user", .content = "Thanks" };
}

static void conv_free(conv *c) {
    free(c->calls_text.s);
    free(c->r1.s);
    free(c->r2.s);
    jv_free(c->calls);
}

// The callers' acceptance rule (render_prompt_alloc in server.c, decide.c):
// a render is complete iff it returned less than the buffer AND left room.
static bool caller_accepts(size_t ret, const char *buf, size_t cap) {
    return ret < cap && strlen(buf) + 1 < cap;
}

static void capacity_sweep(int t, const jv *tools) {
    const char *name = FAMILY[t].name;
    conv c;
    conv_build(&c, t);
    size_t full = render(t, c.m, c.n, true, THINK_DEFAULT, tools, A, sizeof A);
    CHECK(full != SIZE_MAX && full + 2 < sizeof A, "%s: full render failed (%zu)", name, full);
    if (full == SIZE_MAX || full + 2 >= sizeof A) { conv_free(&c); return; }
    CHECK(strlen(A) == full, "%s: full render length %zu, strlen %zu", name, full, strlen(A));
    enum { GUARD = 16 };
    char *buf = malloc(full + 2 + GUARD);
    bool ok = true;
    for (size_t cap = 1; cap <= full + 2 && ok; cap++) {
        memset(buf, 0xA5, cap + GUARD);
        size_t r = render(t, c.m, c.n, true, THINK_DEFAULT, tools, buf, cap);
        size_t w = strnlen(buf, cap);
        bool guard = true;
        for (int g = 0; g < GUARD; g++)
            if ((unsigned char)buf[cap + g] != 0xA5) guard = false;
        if (r == SIZE_MAX) { CHECK(0, "%s cap %zu: allocation failure reported", name, cap); ok = false; break; }
        if (!guard) { CHECK(0, "%s cap %zu: wrote past the buffer", name, cap); ok = false; break; }
        if (w >= cap) { CHECK(0, "%s cap %zu: no NUL inside the buffer", name, cap); ok = false; break; }
        if (memcmp(buf, A, w)) { CHECK(0, "%s cap %zu: not a prefix of the full render", name, cap); ok = false; break; }
        if (cap <= full && caller_accepts(r, buf, cap)) {
            CHECK(0, "%s cap %zu: truncated render (ret %zu, %zu written) reads as complete",
                  name, cap, r, w);
            ok = false; break;
        }
        if (cap > full && (r != full || w != full)) {
            CHECK(0, "%s cap %zu: fitting render returned %zu, wrote %zu, full %zu",
                  name, cap, r, w, full);
            ok = false; break;
        }
    }
    CHECK(ok, "%s: capacity sweep over %zu sizes", name, full + 2);
    free(buf);
    conv_free(&c);
}

// The thinking argument carries the mode in its low bits and, above them,
// the effort bits (read by qwen38 and muse) and the recipient bits (read by
// muse). A request may set either for any family -- the server ORs a
// request's bare_recipients into every family's argument -- so a family
// that reads neither must render exactly what the mode alone renders.
static void mode_bits_are_ignored(int t, const jv *tools) {
    const char *name = FAMILY[t].name;
    bool reads_effort = t == TMPL_QWEN38 || t == TMPL_MUSE;
    bool reads_recipients = t == TMPL_MUSE;
    static const int MODES[] = { THINK_DEFAULT, THINK_ON, THINK_OFF };
    static const int BITS[] = { THINK_EFFORT_MEDIUM, THINK_EFFORT_LOW,
                                THINK_BARE_RECIPIENTS, THINK_NS_RECIPIENTS };
    conv c;
    conv_build(&c, t);
    chat_msg user[1] = { { .role = "user", .content = "Hello" } };
    for (int k = 0; k < 2; k++) {
        const chat_msg *m = k ? c.m : user;
        int n = k ? c.n : 1;
        for (size_t i = 0; i < sizeof MODES / sizeof *MODES; i++)
            for (size_t j = 0; j < sizeof BITS / sizeof *BITS; j++) {
                bool effort = BITS[j] & THINK_EFFORT_MASK;
                if ((effort && reads_effort) || (!effort && reads_recipients)) continue;
                size_t a = render(t, m, n, true, MODES[i], tools, A, sizeof A);
                size_t b = render(t, m, n, true, MODES[i] | BITS[j], tools, B, sizeof B);
                CHECK(a == b && !strcmp(A, B),
                      "%s: mode 0x%x renders differently with bits 0x%x it does not read (%s)",
                      name, MODES[i], BITS[j], k ? "tool turn" : "one user turn");
            }
    }
    conv_free(&c);
}

// The opener each family can leave a turn inside, as bytes. An independent
// reading of the prompt's tail (marks removed), against which the render's
// own report is checked: render_state.reasoning_open must say exactly what
// the bytes say.
static const char *opener(int t) {
    switch (t) {
    case TMPL_GRANITE42: case TMPL_NEMOTRON35: case TMPL_QWEN38:
    case TMPL_ORNITH: case TMPL_QWEN35: case TMPL_QWEN35_NOTHINK:
    case TMPL_NEMOTRON: case TMPL_MINICPM5:
        return "<think>\n";
    case TMPL_HARMONY: return "<|start|>assistant<|channel|>analysis<|message|>";
    case TMPL_MUSE: return "<|start|>assistant to=self<|message|>";
    case TMPL_GEMMA4: case TMPL_GEMMA4_MAINLINE: return "<|channel>thought\n";
    default: return NULL;
    }
}

static bool tail_opens(int t, const char *prompt) {
    const char *o = opener(t);
    if (!o) return false;
    size_t n = 0;
    static char bare[1 << 16];
    for (const char *p = prompt; *p && n + 1 < sizeof bare; p++)
        if (*p != PROMPT_RAW_OPEN && *p != PROMPT_RAW_CLOSE) bare[n++] = *p;
    bare[n] = 0;
    size_t k = strlen(o);
    return n >= k && !memcmp(bare + n - k, o, k);
}

static void state_matches_the_prompt(int t, const jv *tools) {
    const char *name = FAMILY[t].name;
    static const int MODES[] = { THINK_DEFAULT, THINK_ON, THINK_OFF };
    conv c;
    conv_build(&c, t);
    chat_msg user[1] = { { .role = "user", .content = "Hello" } };
    chat_msg cont[2] = { { .role = "user", .content = "Hello" },
                         { .role = "assistant", .content = "Partial answer" } };
    bool opened = false;
    for (int k = 0; k < 3; k++) {
        const chat_msg *m = k == 0 ? user : k == 1 ? c.m : cont;
        int n = k == 0 ? 1 : k == 1 ? c.n - 2 : 2;  // k == 1: ends on the tool results
        for (int gen = 0; gen < 2; gen++)
            for (size_t i = 0; i < sizeof MODES / sizeof *MODES; i++) {
                render_state st = { .reasoning_open = true, .opener = "x", .think_mode = -1 };
                render_messages_state(t, m, n, gen, MODES[i] | THINK_NS_RECIPIENTS * (t != TMPL_MUSE),
                                      tools, A, sizeof A, &st);
                if (st.reasoning_open) {
                    // the reported opener is exactly the prompt's tail
                    size_t on = st.opener ? strlen(st.opener) : 0;
                    static char bare[1 << 16];
                    size_t bn = 0;
                    for (const char *q = A; *q && bn + 1 < sizeof bare; q++)
                        if (*q != PROMPT_RAW_OPEN && *q != PROMPT_RAW_CLOSE) bare[bn++] = *q;
                    CHECK(on && bn >= on && !memcmp(bare + bn - on, st.opener, on),
                          "%s: reported opener is not the prompt's tail (mode 0x%x)", name, MODES[i]);
                } else {
                    CHECK(st.opener == NULL, "%s: an opener with nothing open", name);
                }
                CHECK(st.reasoning_open == tail_opens(t, A),
                      "%s: reasoning_open %d but the prompt %s (mode 0x%x, %s, gen %d)",
                      name, st.reasoning_open, tail_opens(t, A) ? "opens one" : "does not",
                      MODES[i], k == 0 ? "one user turn" : k == 1 ? "after tool results" : "continuation", gen);
                CHECK(st.think_mode == MODES[i], "%s: think_mode %d, rendered under %d",
                      name, st.think_mode, MODES[i]);
                opened |= st.reasoning_open;
            }
    }
    // not a vacuous match: a family that has an opener opens it somewhere
    // here (Harmony and Muse only without tools: with tools their header is
    // bare and the constrained turn picks the channel)
    bool expect = opener(t) && !(tools && (t == TMPL_HARMONY));
    if (expect) CHECK(opened, "%s: never reported an open reasoning segment", name);
    conv_free(&c);
}

int main(void) {
    CHECK(N_FAMILY == TMPL_LFM2 + 1,
          "the table has %d rows for %d template ids", N_FAMILY, TMPL_LFM2 + 1);
    jv *tools = json_parse(TOOL_SRC, strlen(TOOL_SRC));
    CHECK(tools != NULL, "tool fixture parses");
    chat_msg user[1] = { { .role = "user", .content = "Hello" } };
    for (int t = 0; t < N_FAMILY && t <= TMPL_LFM2; t++) {
        const char *name = FAMILY[t].name;

        size_t e0 = render(t, NULL, 0, false, THINK_DEFAULT, NULL, A, sizeof A);
        size_t e1 = render(t, NULL, 0, true, THINK_DEFAULT, NULL, B, sizeof B);
        CHECK(e0 == FAMILY[t].empty && strlen(A) == e0,
              "%s: empty conversation rendered %zu bytes, expected %zu", name, e0, FAMILY[t].empty);
        CHECK(e1 == FAMILY[t].empty_gen && strlen(B) == e1,
              "%s: empty conversation with the generation prompt rendered %zu bytes, expected %zu",
              name, e1, FAMILY[t].empty_gen);

        render(t, user, 1, true, THINK_DEFAULT, NULL, B, sizeof B);
        CHECK(!strstr(B, "get_weather"), "%s: a declaration with no tools", name);
        render(t, user, 1, true, THINK_DEFAULT, tools, A, sizeof A);
        CHECK((strstr(A, "get_weather") != NULL) == FAMILY[t].tools_in_prompt,
              "%s: tools argument %s a declaration", name,
              FAMILY[t].tools_in_prompt ? "no longer renders" : "now renders");

        render(t, user, 1, true, THINK_ON, NULL, A, sizeof A);
        CHECK((strcmp(A, B) != 0) == FAMILY[t].think_on,
              "%s: THINK_ON %s the prompt", name, FAMILY[t].think_on ? "no longer changes" : "now changes");
        render(t, user, 1, true, THINK_OFF, NULL, A, sizeof A);
        CHECK((strcmp(A, B) != 0) == FAMILY[t].think_off,
              "%s: THINK_OFF %s the prompt", name, FAMILY[t].think_off ? "no longer changes" : "now changes");

        bool native = false;
        const char *proto = tool_protocol_name(t, &native);
        CHECK(proto && !strcmp(proto, FAMILY[t].protocol) && native == FAMILY[t].native,
              "%s: tool protocol %s (native %d), expected %s (native %d)", name,
              proto ? proto : "(none)", native, FAMILY[t].protocol, FAMILY[t].native);

        capacity_sweep(t, NULL);
        capacity_sweep(t, tools);
        mode_bits_are_ignored(t, NULL);
        mode_bits_are_ignored(t, tools);
        state_matches_the_prompt(t, NULL);
        state_matches_the_prompt(t, tools);
    }
    jv_free(tools);
    if (fails) {
        fprintf(stderr, "template families: %d of %d checks failed\n", fails, checks);
        return 1;
    }
    printf("template families ok (%d families, %d checks)\n", N_FAMILY, checks);
    return 0;
}
