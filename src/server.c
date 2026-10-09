// HTTP server: OpenAI-compatible API with N parallel inference slots.
//
//   POST /v1/chat/completions   messages, sampling params, stream (SSE),
//                               response_format {"type":"json_object"}
//   POST /v1/completions        raw prompt completion
//   POST /v1/embeddings         L2-normed embeddings, pooled as the GGUF declares
//   POST /v1/rerank             documents ranked by a yes/no relevance choice
//   GET  /v1/models             the loaded model
//   GET  /v1/capabilities       registry + feature discovery
//   GET  /health                liveness
//   GET  /metrics               the same counters in Prometheus text format
//   POST /v1/runner/contexts    pin a named prefix (R10.4); GET lists,
//                               DELETE /v1/runner/contexts/{id} releases;
//                               .../{id}/snapshot writes it to disk (R1.12)
//   GET  /v1/runner/provenance  binary and model digests, the load-time
//                               signature and envelope verdicts, the effective
//                               configuration (provenance.h)
//
// Swap-mode request bodies may carry "keep_alive" (seconds of idle before
// the model unloads; 0 = unload now, negative = keep forever).
//
// Each slot owns a full inference context (KV cache + thread pool); model
// weights are shared between slots through the page cache (mmap).
#include "runner.h"
#include "json.h"
#include "compat.h"
#include "http.h"
#include "server_int.h"
#include "decide.h"
#include "scheduler.h"
#include "completion.h"
#include "api.h"
#include "server.h"
#include "provenance.h"
#include "respstore.h"
#include "kvsnap.h"
#include "session.h"
#include "envelope.h"
#include "gpu.h"

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdatomic.h>




// ---------------------------------------------------------------- routes

// The publisher's fixed default system turn for the model this slot serves,
// or NULL (template_default_system). Every chat surface asks here, so the
// same system-less conversation is the same prompt through every door.
// server_template_text_override is the conformance renderer's stand-in for
// the model's template text: it drives these handlers with no model loaded.
const char *server_template_text_override;
const char *slot_default_system(const slot_t *s) {
    const char *t = server_template_text_override;
    if (!t && s->m) t = gguf_get_str(&s->m->gf, "tokenizer.chat_template", NULL);
    return template_default_system(s->tmpl, t);
}

// Declared in api.h, where the reason it exists is written down.
char *render_prompt_alloc(int tmpl, const chat_msg *msgs, int n_msgs,
                          bool add_assistant, int thinking, const jv *tools,
                          size_t hint) {
    size_t cap = hint < 512 ? 512 : hint;
    for (;;) {
        char *p = malloc(cap);
        if (!p) return NULL;
        size_t n = render_messages_with_tools(tmpl, msgs, n_msgs, add_assistant,
                                              thinking, tools, p, cap);
        if (n == SIZE_MAX) {
            free(p);
            return NULL;
        }
        // Complete iff the render did not fill the buffer. Two independent
        // signals, because one of them lives in a module this call does not
        // own: emit() reports an offset at or past `cap` when snprintf
        // truncated, AND a truncating snprintf always fills to cap-1. A
        // prompt that happens to end one byte short of `cap` is grown once
        // unnecessarily, which costs a render; the reverse mistake costs the
        // user's question.
        if (n < cap && strlen(p) + 1 < cap) return p;
        free(p);
        if (cap > SIZE_MAX / 2) return NULL;  // no size can hold it
        size_t want = (n > cap ? n : cap) + 1;
        cap = want > cap * 2 ? want : cap * 2;
    }
}

// flatten one OpenAI message to plain text: string content passes through;
// the AI-SDK part-array form (Cline et al.) concatenates its text parts;
// assistant tool_calls render in the family's own call syntax so replayed
// history reads like what the model actually emitted. Returns a heap
// string, or NULL when the message carries nothing usable.
//
// The tool serialization is NOT open-coded here: the assistant-call ordering
// (gemma4 calls-first, muse's recipient turn, everyone else calls-after) lives
// in assistant_calls_render, and the tool-result framing in tool_result_wrap,
// and the typed /v1/responses and /v1/messages surfaces reach the SAME two
// helpers. That is what makes a tool call replayed through any of the three
// surfaces render byte-identically -- the contract test_tool_attribution pins.
//
// *oom separates the two NULLs this can return. A message that carries nothing
// renderable is skipped by the caller, which is right; a message whose text
// could not be ASSEMBLED used to take the same exit, so an allocation failure
// dropped a turn out of the conversation and the request still answered 200.
// The model then answers a different question than the one it was asked, and
// nothing in the response says so.
static char *message_text(jv *msg, int tmpl, bool replay_reason, bool *oom) {
    jv *content = jv_get(msg, "content");
    const char *role = jv_str(jv_get(msg, "role"), "user");
    const char *reason = jv_str(jv_get(msg, "reasoning_content"), NULL);
    jv *calls = jv_get(msg, "tool_calls");

    // Flatten the visible content into one string first: the shared assembler
    // and the result wrapper both take flat text, and this is the single place
    // the AI-SDK part-array's text parts are joined (with '\n' between two text
    // parts, never in front of the first).
    sbuf txt = {0};
    if (content && content->type == J_STR) {
        sb_put(&txt, content->str, strlen(content->str));
    } else if (content && content->type == J_ARR) {
        int parts = 0;
        for (int i = 0; i < content->n; i++) {
            const char *type = jv_str(jv_get(content->items[i], "type"), "");
            const char *text = jv_str(jv_get(content->items[i], "text"), NULL);
            if (strcmp(type, "text") != 0 || !text) continue; // images etc.
            if (parts++) sb_lit(&txt, "\n");
            sb_put(&txt, text, strlen(text));
        }
    }

    sbuf b = {0};
    if (tmpl_ornith_like(tmpl) && !strcmp(role, "tool")) {
        // a result is a <tool_response> block in a user turn, not a plain one
        tool_result_wrap(tmpl, txt.s ? txt.s : "", &b);
    } else {
        // Ornith opens every assistant turn with its (possibly empty) thought
        // block. Qwen3 preserves reasoning only on the final historical
        // assistant after the last user, which the caller selects explicitly.
        // Calls and visible text follow the block in the same buffer.
        // The framing is the runner's, not the caller's: marked (prompt_lit)
        // so `<think>` reaches the model as the control token its reference
        // tokenizer produces, not as three text tokens. The reasoning text
        // between is the caller's and goes in unmarked.
        // Qwen 3.5 writes the block (empty or not) only on the assistant
        // turns after the last user query, which the caller selects too.
        if ((tmpl == TMPL_ORNITH || tmpl == TMPL_QWEN38 ||
             ((tmpl == TMPL_CHATML_THINK || tmpl_qwen35(tmpl)) &&
              replay_reason)) &&
            !strcmp(role, "assistant")) {
            prompt_lit(&b, "<think>\n");
            if (reason) sb_put(&b, reason, strlen(reason));
            prompt_lit(&b, "\n</think>\n\n");
        } else if (tmpl_granite42_like(tmpl) && !strcmp(role, "assistant") &&
                   reason && reason[strspn(reason, " \t\n\r\f\v")]) {
            // granite 4.2 folds a non-blank reasoning_content in as
            // `<think>\n` reasoning `\n</think>\n` content
            // (chat_template.jinja:77), Nemotron 3.5 as `<think>\n`
            // reasoning `</think>` content; the renderer then seeds or
            // truncates the block by the turn's position.
            prompt_lit(&b, "<think>\n");
            sb_put(&b, reason, strlen(reason));
            prompt_lit(&b, tmpl == TMPL_NEMOTRON35 ? "</think>" : "\n</think>\n");
        }
        assistant_calls_render(tmpl, txt.s, calls, &b, NULL);
    }
    bool failed = b.failed || txt.failed;
    free(txt.s);
    if (failed) { free(b.s); *oom = true; return NULL; }
    return b.s;
}

// Validate the message envelope before any turn is rendered. Defaults and
// `continue` are unsafe at this boundary: either can make a successful request
// mean a different conversation from the one the caller submitted.
static bool validate_chat_messages(const jv *msgs, char *err, size_t err_cap) {
    for (int i = 0; i < msgs->n; i++) {
        jv *msg = msgs->items[i];
        if (!msg || msg->type != J_OBJ) {
            snprintf(err, err_cap, "messages[%d] must be an object", i);
            return false;
        }
        jv *role_v = jv_get(msg, "role");
        if (!role_v || role_v->type != J_STR) {
            snprintf(err, err_cap, "messages[%d].role must be a string", i);
            return false;
        }
        const char *role = role_v->str;
        if (strcmp(role, "system") && strcmp(role, "developer") &&
            strcmp(role, "user") && strcmp(role, "assistant") &&
            strcmp(role, "tool")) {
            snprintf(err, err_cap,
                     "messages[%d].role must be system, developer, user, "
                     "assistant or tool", i);
            return false;
        }
        jv *name_v = jv_get(msg, "name");
        if (name_v && name_v->type != J_NULL && name_v->type != J_STR) {
            snprintf(err, err_cap, "messages[%d].name must be a string", i);
            return false;
        }
        jv *call_id_v = jv_get(msg, "tool_call_id");
        if (call_id_v && call_id_v->type != J_NULL &&
            call_id_v->type != J_STR) {
            snprintf(err, err_cap,
                     "messages[%d].tool_call_id must be a string", i);
            return false;
        }
        jv *calls = jv_get(msg, "tool_calls");
        if (calls && calls->type != J_NULL) {
            if (calls->type != J_ARR) {
                snprintf(err, err_cap,
                         "messages[%d].tool_calls must be an array", i);
                return false;
            }
            if (strcmp(role, "assistant")) {
                snprintf(err, err_cap,
                         "messages[%d].tool_calls is valid only on an assistant "
                         "message", i);
                return false;
            }
            for (int k = 0; k < calls->n; k++) {
                jv *call = calls->items[k];
                if (!call || call->type != J_OBJ) {
                    snprintf(err, err_cap,
                             "messages[%d].tool_calls[%d] must be an object",
                             i, k);
                    return false;
                }
                const char *id = jv_str(jv_get(call, "id"), NULL);
                if (!id || !id[0]) {
                    snprintf(err, err_cap,
                             "messages[%d].tool_calls[%d].id must be a "
                             "non-empty string", i, k);
                    return false;
                }
                const char *type = jv_str(jv_get(call, "type"), NULL);
                if (!type || strcmp(type, "function")) {
                    snprintf(err, err_cap,
                             "messages[%d].tool_calls[%d].type must be "
                             "function", i, k);
                    return false;
                }
                jv *fn = jv_get(call, "function");
                if (!fn || fn->type != J_OBJ) {
                    snprintf(err, err_cap,
                             "messages[%d].tool_calls[%d].function must be an "
                             "object", i, k);
                    return false;
                }
                const char *name = jv_str(jv_get(fn, "name"), NULL);
                if (!name || !name[0]) {
                    snprintf(err, err_cap,
                             "messages[%d].tool_calls[%d].function.name must "
                             "be a non-empty string", i, k);
                    return false;
                }
                jv *args = jv_get(fn, "arguments");
                jv *parsed = args && args->type == J_STR
                    ? json_parse(args->str, strlen(args->str)) : NULL;
                bool valid = parsed && parsed->type == J_OBJ;
                jv_free(parsed);
                if (!valid) {
                    snprintf(err, err_cap,
                             "messages[%d].tool_calls[%d].function.arguments "
                             "must be a string containing a JSON object", i, k);
                    return false;
                }
            }
        }
        jv *reasoning = jv_get(msg, "reasoning_content");
        if (reasoning && reasoning->type != J_NULL && reasoning->type != J_STR) {
            snprintf(err, err_cap,
                     "messages[%d].reasoning_content must be a string", i);
            return false;
        }
        const char *reason = jv_str(reasoning, NULL);
        bool assistant_payload = !strcmp(role, "assistant") &&
            ((calls && calls->type == J_ARR && calls->n > 0) ||
             (reason && reason[0]));
        jv *content = jv_get(msg, "content");
        bool content_shape = content &&
                             (content->type == J_STR || content->type == J_ARR);
        bool content_absent = !content || content->type == J_NULL;
        if ((!content_shape && !content_absent) ||
            (content_absent && !assistant_payload)) {
            snprintf(err, err_cap,
                     "messages[%d].content must be a string or an array", i);
            return false;
        }
        if (content && content->type == J_ARR) {
            if (content->n == 0 && !assistant_payload) {
                snprintf(err, err_cap,
                         "messages[%d].content must not be an empty array", i);
                return false;
            }
            for (int k = 0; k < content->n; k++) {
                jv *part = content->items[k];
                const char *type = part && part->type == J_OBJ
                    ? jv_str(jv_get(part, "type"), NULL) : NULL;
                if (!type) {
                    snprintf(err, err_cap,
                             "messages[%d].content[%d] must be a typed object",
                             i, k);
                    return false;
                }
                if (strcmp(type, "text")) {
                    snprintf(err, err_cap,
                             "messages[%d].content[%d] type %s is unsupported; "
                             "only text is accepted", i, k, type);
                    return false;
                }
                jv *text = jv_get(part, "text");
                if (!text || text->type != J_STR) {
                    snprintf(err, err_cap,
                             "messages[%d].content[%d].text must be a string",
                             i, k);
                    return false;
                }
            }
        }
    }
    return true;
}

static const char *chat_role(jv *msg) {
    const char *role = jv_str(jv_get(msg, "role"), "user");
    return !strcmp(role, "developer") ? "system" : role;
}

// The function a tool turn is reporting for, resolved without inventing one.
//
// template.c's tool_result_name() has a last resort this surface cannot use:
// with no `name` and an unmatched `tool_call_id` it hands back the ID, and
// Harmony authors the turn by that string -- `<|start|>functions.call_1`,
// a function name invented from an identifier and declared nowhere. So the
// lookup is done here without that step, and an unresolved result is treated
// as unresolved rather than named after its id.
// The call a client replays may be TEXT, not a tool_calls[] entry: muse and
// gemma4 assistants spell it `<|tool_call>call:NAME{...}` and a client that
// keeps its own history sends that content straight back. The spelling names
// the function precisely, so resolving from it invents nothing. The k-th
// result after the assistant turn reports for the k-th call spelled in it;
// past the spelled count the result stays unresolved (fail closed). The
// extracted name is strdup'ed into the caller's owned[] so it outlives this
// frame -- cm[] keeps the pointer until the prompt renders.
static const char *text_spelled_call_name(const jv *msgs, int message_index,
                                          char **owned, int *n_own) {
    int a = -1;
    for (int i = message_index - 1; i >= 0; i--) {
        const char *r = chat_role(msgs->items[i]);
        if (!strcmp(r, "assistant")) { a = i; break; }
        if (strcmp(r, "tool")) return NULL; // any other role breaks the loop
    }
    if (a < 0) return NULL;
    int nth = 0;
    for (int i = a + 1; i < message_index; i++)
        if (!strcmp(chat_role(msgs->items[i]), "tool")) nth++;
    const char *content = jv_str(jv_get(msgs->items[a], "content"), NULL);
    if (!content) return NULL;
    static const char MARK[] = "<|tool_call>call:";
    for (const char *p = strstr(content, MARK); p;
         p = strstr(p, MARK)) {
        p += sizeof(MARK) - 1;
        const char *b = p;
        while (*b && (isalnum((unsigned char)*b) || *b == '_' || *b == '-' ||
                      *b == '.'))
            b++;
        if (*b != '{' || b == p || b - p > 64) continue; // not a call spelling
        if (nth-- > 0) continue;
        char *nm = malloc((size_t)(b - p) + 1);
        if (!nm) return NULL;
        memcpy(nm, p, (size_t)(b - p));
        nm[b - p] = 0;
        owned[(*n_own)++] = nm;
        return nm;
    }
    return NULL;
}

static const char *chat_result_name(const jv *msgs, int message_index,
                                    char **owned, int *n_own) {
    jv *msg = msgs->items[message_index];
    const char *name = jv_str(jv_get(msg, "name"), NULL);
    if (name && name[0]) return name;
    const char *id = jv_str(jv_get(msg, "tool_call_id"), NULL);
    if (id) {
        for (int i = 0; i < message_index; i++) {
            jv *calls = jv_get(msgs->items[i], "tool_calls");
            if (!calls || calls->type != J_ARR) continue;
            for (int k = 0; k < calls->n; k++) {
                const char *candidate =
                    jv_str(jv_get(calls->items[k], "id"), NULL);
                if (!candidate || strcmp(candidate, id)) continue;
                jv *fn = jv_get(calls->items[k], "function");
                const char *fname = jv_str(jv_get(fn, "name"), NULL);
                if (fname && fname[0]) return fname;
            }
        }
    }
    // no name field and no id match: the call may be spelled in the prior
    // assistant turn's TEXT (see text_spelled_call_name)
    return text_spelled_call_name(msgs, message_index, owned, n_own);
}

// Declared in api.h, where the reason it exists is written down. It lived as
// an identical private copy in this file, api_responses.c and api_anthropic.c
// -- one definition each of the same five lines, because the commit that
// introduced it could not add to api.h.
const char *sole_tool_name(const jv *tools) {
    if (!tools || tools->type != J_ARR || tools->n != 1) return NULL;
    jv *fn = jv_get(tools->items[0], "function");
    const char *name = jv_str(jv_get(fn, "name"), NULL);
    return name && name[0] ? name : NULL;
}

// What becomes of a rendered chat prompt: generation (handle_chat), or a
// named context pinned from it (R10.4). One renderer serves both, so a
// context built from messages is byte for byte the prefix a later chat
// request renders from the same messages.
typedef void (*chat_prompt_fn)(slot_t *s, sock_t fd, const char *prompt,
                               jv *req, const tool_envelope *env);

static void handle_chat_render(slot_t *s, sock_t fd, jv *req,
                               bool add_assistant, chat_prompt_fn run);

static void chat_generate(slot_t *s, sock_t fd, const char *prompt, jv *req,
                          const tool_envelope *env) {
    run_completion(s, fd, prompt, API_CHAT, req, env);
}

static void handle_chat(slot_t *s, sock_t fd, jv *req) {
    handle_chat_render(s, fd, req, true, chat_generate);
}

static void handle_chat_render(slot_t *s, sock_t fd, jv *req,
                               bool add_assistant, chat_prompt_fn run) {
    jv *msgs = jv_get(req, "messages");
    if (!msgs || msgs->type != J_ARR || msgs->n == 0) {
        send_error(fd, 400, "missing messages");
        return;
    }
    char merr[192];
    if (!validate_chat_messages(msgs, merr, sizeof(merr))) {
        send_error(fd, 400, merr);
        return;
    }
    const char **roles = malloc(sizeof(*roles) * (size_t)(unsigned)msgs->n);
    if (!roles) {
        send_error(fd, 500, "out of memory validating chat history");
        return;
    }
    for (int i = 0; i < msgs->n; i++) roles[i] = chat_role(msgs->items[i]);
    bool roles_ok = template_roles_valid(s->tmpl, roles, msgs->n, false, merr,
                                         sizeof(merr));
    free(roles);
    if (!roles_ok) {
        send_error(fd, 400, merr);
        return;
    }
    if (!req_thinking_mode_valid(req)) {
        send_error(fd, 400,
                   "enable_thinking must be a boolean or null, either at the "
                   "top level or inside chat_template_kwargs");
        return;
    }
    // OpenAI "tools" become a leading system turn (template.c owns the syntax).
    //
    // Strict mode compiles them into a discriminated union that constrains
    // sampling, so the model cannot name an undeclared tool or malform its
    // arguments. It applies to streamed requests too: the envelope is
    // demultiplexed as it is generated (tool_stream) rather than parsed
    // afterward, so both paths reach the same call from the same guarantee.
    jv *tools = jv_get(req, "tools");
    tool_envelope env = {0};
    // parallel_tool_calls is read BEFORE the envelope is built, because it
    // changes the envelope's shape. Silently ignoring a request for several
    // calls would leave the caller expecting calls it never gets.
    // An absent flag follows OpenAI's default (several calls per turn),
    // except where the family's parallel grammar is still a fixed pair
    // (tool_parallel_default, owner 2026-10-09), and except when tool_choice
    // names one function, which OpenAI calls exactly once.
    bool parallel = false;
    if (!request_bool(req, "parallel_tool_calls",
                      tool_parallel_default(s->tmpl, jv_get(req, "tool_choice")),
                      &parallel)) {
        send_error(fd, 400, "parallel_tool_calls must be a boolean");
        return;
    }
    bool atem_tool_calling = true;
    if (!request_bool(req, "atem_tool_calling", true, &atem_tool_calling)) {
        send_error(fd, 400, "atem_tool_calling must be a boolean");
        return;
    }
    // parallel_tool_calls:true + stream:true used to be refused here: the
    // demultiplexer tracked one call per turn, and downgrading silently
    // would leave the caller expecting calls it never got. tool_stream now
    // loops the {"calls":[...]} array the same way ts_atem already looped
    // native <atem:invoke> blocks -- announcing each call on its own index,
    // closing it, and moving to the next -- so both modes stream several
    // calls in one turn and the refusal is gone.
    char terr[224];
    int rc = tool_envelope_build_ex(tools, jv_get(req, "tool_choice"),
                                    request_schema(req), parallel, &env,
                                    terr, sizeof(terr));
    if (rc < 0) {
        send_error(fd, rc == TOOL_ENVELOPE_OOM ? 500 : 400, terr);
        return;
    }
    // The envelope applies whenever tools are declared and callable. Ornith,
    // Granite 4.2 and Qwen 3.8 used to be excluded here to keep their native
    // XML protocol out of the generic JSON envelope; that also kept them out
    // of the streaming demultiplexer, and every streamed call they made
    // reached the client as prose (2026-09-14 Windows report). tool_decl_native
    // now gives them the XML protocol on the envelope, parse-only (no
    // grammar), so the same parser serves them buffered and streamed.
    bool strict = rc == 1;
    // When the strict envelope does not apply -- no tools declared, or
    // tool_choice none -- the flag is vacuous and stays TOLERATED, exactly as
    // before: ordinary OpenAI-shaped traffic sends parallel_tool_calls
    // alongside requests that will never call anything, and rejecting those
    // would break it.
    //
    // Families with a native declaration syntax render it themselves, from the
    // structured `tools` handed to render_prompt_alloc below; everyone else
    // gets the strict envelope's teaching turn, or the generic declaration
    // block when the envelope does not apply. tool_decl_native makes that
    // selection -- setting env's native flags and returning the declarations to
    // render plus native_decl (skip the generic turn) -- from the SAME helper
    // the /v1/responses and /v1/messages surfaces now use, so a declared tool
    // is taught identically on all three.
    bool native_decl = false;
    const jv *native_tools = tool_decl_native(s->tmpl, strict,
                                              atem_tool_calling, tools, &env,
                                              &native_decl);
    sbuf ts = {0};
    bool oom = false;
    // The generic teaching turn belongs to the generic envelope only. A
    // family whose envelope carries a native protocol but renders its
    // declarations outside the template (ornith, granite 4.2) gets its own
    // declaration block from tools_render_for, exactly as it did without an
    // envelope: prompt and parser must agree on the protocol.
    if (strict && !native_decl && !tool_envelope_native(&env))
        sb_put(&ts, env.system_turn, strlen(env.system_turn));
    else if (!native_decl && s->tmpl != TMPL_MUSE)
        tools_render_for(s->tmpl, tools, &ts);
    // ornith / granite 4.2 fold the caller's system text into the
    // declaration turn the way their references do (tools_system_fold, the
    // same helper the typed surfaces use); the folded message is then skipped
    bool ornith_merged_system = false;
    if ((tmpl_ornith_like(s->tmpl) || tmpl_granite42_like(s->tmpl)) && ts.n &&
        msgs->n > 0 && !strcmp(chat_role(msgs->items[0]), "system")) {
        char *system = message_text(msgs->items[0], s->tmpl, false, &oom);
        tools_system_fold(s->tmpl, &ts, system);
        free(system);
        ornith_merged_system = true;   // an empty system turn folds to nothing
    }
    // The tool turn is content too. A builder that ran out here left `ts`
    // short or empty and the prompt went out without the declarations the
    // caller sent -- the model is then asked to call tools it was never shown.
    if (ts.failed || oom) {
        free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 500, "out of memory building chat prompt");
        return;
    }
    size_t cm_cap = (size_t)msgs->n + 2;   // + the tool turn, + a default system turn
    // Every family that replays reasoning_content as its own turn needs a
    // slot for it: Harmony (analysis channel) and Muse (the to=self turn,
    // since #121). Counting only Harmony's left Muse one slot short per
    // reasoning turn, a heap overflow the third replayed turn reached (the
    // lab, 2026-09-16: "double free or corruption" on the fourth request of
    // a Muse tool conversation; ASan on the fixture: WRITE past cm at the
    // third).
    if (s->tmpl == TMPL_HARMONY || s->tmpl == TMPL_MUSE) {
        for (int i = 0; i < msgs->n; i++) {
            jv *calls = jv_get(msgs->items[i], "tool_calls");
            if (calls && calls->type == J_ARR) cm_cap += (size_t)calls->n;
            if (jv_str(jv_get(msgs->items[i], "reasoning_content"), NULL))
                cm_cap++;
        }
    }
    chat_msg *cm = malloc(sizeof(chat_msg) * cm_cap);
    // one rendered content per message, plus at most one text-extracted tool
    // name per tool result (text_spelled_call_name strdup's into this array)
    char **owned = malloc(sizeof(char *) * (size_t)msgs->n * 2);
    // client-controlled size (a 32MB body of tiny messages): a NULL here would
    // be indexed below. Fail the request cleanly instead of crashing.
    if (!cm || !owned) {
        free(cm); free(owned); free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 500, "out of memory building chat prompt");
        return;
    }
    size_t total = ts.n + 64;
    int n_cm = 0, n_own = 0;
    if (ts.n)
        cm[n_cm++] = (chat_msg){ .role = "system", .content = ts.s };
    else if (!native_tools && msgs->n > 0 &&
             strcmp(chat_role(msgs->items[0]), "system")) {
        // no system turn from the caller and none from the tools: the
        // publisher's template would write its own
        const char *ds = slot_default_system(s);
        if (ds) {
            cm[n_cm++] = (chat_msg){ .role = "system", .content = ds };
            total += strlen(ds) + 64;
        }
    }
    int last_user = -1;
    if (s->tmpl == TMPL_CHATML_THINK || tmpl_qwen35(s->tmpl))
        for (int i = 0; i < msgs->n; i++)
            if (!strcmp(chat_role(msgs->items[i]), "user")) last_user = i;
    for (int i = 0; i < msgs->n; i++) {
        if (i == 0 && ornith_merged_system) continue;
        const char *role = chat_role(msgs->items[i]);
        const char *turn_name = NULL;
        if (!strcmp(role, "tool")) {
            turn_name = chat_result_name(msgs, i, owned, &n_own);
            if (!turn_name) turn_name = sole_tool_name(tools);
            // A result that matches no prior call cannot be assigned safely
            // when several functions exist. Gemma4 and Muse put the resolved
            // name on the turn header; Harmony authors the turn by it; other
            // families still need the call/result relationship to be honest.
            if (!turn_name) {
                const char *id = jv_str(jv_get(msgs->items[i], "tool_call_id"),
                                        NULL);
                for (int k = 0; k < n_own; k++) free(owned[k]);
                free(owned); free(cm); free(ts.s);
                tool_envelope_free(&env);
                char e[288];
                snprintf(e, sizeof(e),
                         "messages[%d] is a tool result that cannot be "
                         "attributed to a tool: %s%.40s%s. Give it a `name`, "
                         "or a `tool_call_id` matching an earlier assistant "
                         "tool_calls[].id.", i,
                         id ? "no earlier assistant tool_calls[] carries id \""
                            : "it carries neither", id ? id : "",
                         id ? "\"" : " `name` nor `tool_call_id`");
                send_error(fd, 400, e);
                return;
            }
        }
        if (s->tmpl == TMPL_MUSE && !strcmp(role, "assistant")) {
            jv *calls = jv_get(msgs->items[i], "tool_calls");
            if (calls && calls->type == J_ARR && calls->n) {
                jv *fn = jv_get(calls->items[0], "function");
                turn_name = jv_str(jv_get(fn, "name"), NULL);
            }
        }
        // Harmony (the analysis channel) and Muse (a `to=self` turn ending
        // <|eom|>) replay reasoning_content as its own turn, as their
        // references do; the renderer spells each family's form
        if ((s->tmpl == TMPL_HARMONY || s->tmpl == TMPL_MUSE) && !strcmp(role, "assistant")) {
            const char *reason = jv_str(
                jv_get(msgs->items[i], "reasoning_content"), NULL);
            if (reason && reason[0]) {
                cm[n_cm++] = (chat_msg){ .role = "assistant",
                                        .content = reason,
                                        .channel = "analysis" };
                total += strlen(reason) + 64;
            }
        }
        // Harmony's answer and calls are separate messages on separate
        // channels; every other family's, Muse's included, is one turn built
        // by message_text below (Muse's call is the atem block, to=NAME)
        if (s->tmpl == TMPL_HARMONY && !strcmp(role, "assistant")) {
            char *visible = message_text(msgs->items[i], s->tmpl, false, &oom);
            if (oom) break;
            jv *calls = jv_get(msgs->items[i], "tool_calls");
            bool have_calls = calls && calls->type == J_ARR && calls->n;
            if (visible && visible[0]) {
                owned[n_own++] = visible;
                cm[n_cm++] = (chat_msg){ .role = "assistant",
                                        .content = visible,
                                        .channel = have_calls
                                                   ? "commentary" : NULL };
                total += strlen(visible) + 64;
            } else {
                free(visible);
            }
            if (have_calls) for (int k = 0; k < calls->n; k++) {
                jv *fn = jv_get(calls->items[k], "function");
                const char *name = jv_str(jv_get(fn, "name"), NULL);
                const char *args = jv_str(jv_get(fn, "arguments"), "{}");
                if (!name) continue;
                cm[n_cm++] = (chat_msg){ .role = "assistant",
                                        .content = args, .name = name };
                total += strlen(name) + strlen(args) + 96;
            }
            continue;
        }
        bool replay_reason = (s->tmpl == TMPL_CHATML_THINK &&
                              i == msgs->n - 1 && i > last_user) ||
                             (tmpl_qwen35(s->tmpl) && i > last_user);
        char *content = message_text(msgs->items[i], s->tmpl, replay_reason,
                                     &oom);
        if (oom) break;
        if (!content) continue;
        owned[n_own++] = content;
        if (tmpl_ornith_like(s->tmpl) && !strcmp(role, "tool")) role = "user";
        cm[n_cm++] = (chat_msg){
            .role = role, .content = content, .name = turn_name,
        };
        total += strlen(role) + strlen(content) + 64;
    }
    if (oom || n_cm == 0) {
        for (int i = 0; i < n_own; i++) free(owned[i]);
        free(owned);
        free(cm);
        free(ts.s);
        tool_envelope_free(&env);
        if (oom) send_error(fd, 500, "out of memory building chat prompt");
        else     send_error(fd, 400, "no message content");
        return;
    }
    // Native templates render the declarations themselves -- muse's JSON
    // block, Harmony's TypeScript namespace -- so their bytes belong in the
    // opening guess. It is only a guess: render_prompt_alloc measures the
    // real size and grows, so an under-count here costs a render pass, not a
    // truncated prompt. native_tools was chosen up front by tool_decl_native.
    sbuf tool_bytes = {0};
    if (native_tools) jv_dump(tools, &tool_bytes);
    total += tool_bytes.n + 4096;
    free(tool_bytes.s);
    int thinking = req_thinking_mode(req);
    int recipients = req_bare_recipients(req);
    if (recipients < 0) {
        for (int i = 0; i < n_own; i++) free(owned[i]);
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 400, "bare_recipients must be a boolean");
        return;
    }
    thinking |= recipients;
    if (s->tmpl == TMPL_QWEN38) {
        // the template's own contract: xhigh (default), medium or low,
        // anything else is refused by the reference and so here
        int effort = req_reasoning_effort(req);
        if (effort < 0) {
            for (int i = 0; i < n_own; i++) free(owned[i]);
            free(owned); free(cm); free(ts.s);
            tool_envelope_free(&env);
            send_error(fd, 400, "reasoning_effort must be one of xhigh, "
                                "medium, low for this model's template");
            return;
        }
        thinking |= effort;
    } else if (s->tmpl == TMPL_MUSE) {
        // the reference's `reasoning_strength` kwarg (low, medium, high;
        // absent renders "high"); `reasoning_effort` is accepted as the
        // cross-family spelling, its xhigh reading as high. A value the
        // reference would print verbatim into the system turn is refused
        // here unless it is one of the three words.
        int strength = req_reasoning_strength(req);
        int effort = req_reasoning_effort(req);
        if (strength < 0 || effort < 0) {
            for (int i = 0; i < n_own; i++) free(owned[i]);
            free(owned); free(cm); free(ts.s);
            tool_envelope_free(&env);
            send_error(fd, 400, "reasoning_strength must be one of low, "
                                "medium, high for this model's template");
            return;
        }
        thinking |= strength ? strength : effort;
    }
    char *prompt = render_prompt_alloc(s->tmpl, cm, n_cm, add_assistant,
                                       thinking, native_tools,
                                       total + 256);
    if (!prompt) {
        for (int i = 0; i < n_own; i++) free(owned[i]);
        free(owned); free(cm); free(ts.s);
        tool_envelope_free(&env);
        send_error(fd, 500, "out of memory building chat prompt");
        return;
    }
    run(s, fd, prompt, req, strict ? &env : NULL);
    free(prompt);
    for (int i = 0; i < n_own; i++) free(owned[i]);
    free(owned);
    free(cm);
    free(ts.s);
    tool_envelope_free(&env);
}

// ---- named contexts (R10.4) ---------------------------------------------
//
// POST /v1/runner/contexts {id, prompt} or {id, messages[, tools]} prefills
// the prompt once and pins its KV under `id`; a request carrying
// "context_id": id forks it (completion.c checks it is a prefix first).
// GET lists them, DELETE /v1/runner/contexts/{id} releases one.
static void context_pin_prompt(slot_t *s, sock_t fd, const char *prompt,
                               bool chat, jv *req) {
    const char *id = jv_str(jv_get(req, "id"), "");
    engine *e = &s->e;
    int32_t *toks = NULL;
    int n = tok_encode_fit(s->tok, prompt, true, chat ? TOK_PROMPT : TOK_RAW,
                           0, &toks);
    if (n < 0) { free(toks); send_error(fd, 500, "out of memory tokenizing"); return; }
    if (n < 2 || n >= s->m->n_ctx) {
        free(toks);
        send_error_detail(fd, 400,
                          n >= 2 ? "the context does not fit the context "
                                   "window with room to continue it"
                          : chat ? "these messages render to (almost) "
                                   "nothing on their own in this model's "
                                   "template (some fold the system turn into "
                                   "the first user turn); a context needs at "
                                   "least 2 tokens"
                                 : "a context needs at least 2 tokens",
                          chat ? "messages" : "prompt", "invalid_value");
        return;
    }
    double t0 = now_s();
    sched_prefill_begin();
    prefix_reuse r = engine_prefix_reuse(e, toks, n);
    float *lg = engine_feed(e, toks + r.keep, n - r.keep);
    int rc = lg ? prefix_context_pin(e, id, toks, n) : PFX_CTX_NOSPACE;
    sched_prefill_end();
    double secs = now_s() - t0;
    free(toks);
    if (!lg) { send_error(fd, 500, "prefill failed (context or memory)"); return; }
    if (rc == PFX_CTX_UNSUPPORTED) {
        send_error_detail(fd, 409, "this model cannot fork a pinned prefix: its "
                          "KV layout has no contiguous prefix (a ring or "
                          "tied-V cache), or it is a recurrent model on a "
                          "device, whose state the host cannot restore "
                          "(serve it with --gpu off to use contexts)", NULL,
                          "context_unsupported");
        return;
    }
    if (rc == PFX_CTX_NOSPACE) {
        send_error_detail(fd, 507, "the prefix-cache budget cannot hold this "
                          "context beside the ones already pinned "
                          "(RUNNER_PREFIX_CACHE_MB, or DELETE a context)",
                          NULL, "context_budget");
        return;
    }
    if (rc < 0) { send_error(fd, 500, "could not pin the context"); return; }
    // the entry's size is the cache's own per-token arithmetic
    size_t bytes = prefix_cache_entry_bytes(s->m, n);
    char body[512];
    int bn = snprintf(body, sizeof body,
                      "{\"object\":\"runner.context\",\"id\":\"%s\","
                      "\"tokens\":%d,\"bytes\":%llu,\"prefill_tokens\":%d,"
                      "\"cached_tokens\":%d,\"seconds\":%.6f}",
                      id, n, (unsigned long long)bytes, n - r.keep, r.keep, secs);
    send_response(fd, 200, "application/json", body, (size_t)bn);
    fprintf(stderr, "[slot %d] context %s: %d tokens pinned (%d prefilled)\n",
            s->id, id, n, n - r.keep);
}

static void context_from_chat(slot_t *s, sock_t fd, const char *prompt,
                              jv *req, const tool_envelope *env) {
    (void)env;   // declarations are in the rendered prompt; nothing is parsed
    context_pin_prompt(s, fd, prompt, true, req);
}

static void send_kvsnap(sock_t fd, bool ok, sbuf *out, const kvsnap_err *err) {
    if (ok && !out->failed) send_response(fd, 200, "application/json", out->s, out->n);
    else if (ok) send_error(fd, 500, "out of memory");
    else send_error_detail(fd, err->status, err->msg, NULL, err->code);
    free(out->s);
}

// R1.12.2: a context loaded from a --kv-snapshots snapshot instead of prefilled
static void context_from_snapshot(slot_t *s, sock_t fd, const char *id,
                                  jv *snap) {
    if (!snap || snap->type != J_STR) {
        send_error_detail(fd, 400, "snapshot must be a string", "snapshot",
                          "invalid_type");
        return;
    }
    sbuf out = {0};
    kvsnap_err err = {0};
    bool ok = kvsnap_load(&s->e, id, snap->str, &out, &err);
    send_kvsnap(fd, ok, &out, &err);
}

// R1.12.1: POST /v1/runner/contexts/{id}/snapshot {name?, receipt?}
// A context id in a URL path may arrive percent-encoded: the TypeScript
// client sends encodeURIComponent(id), which turns the ':' the id grammar
// allows into "%3A", and the raw comparison then found no such context
// (found 2026-10-05). Decode %XX (and only %XX; '+' is a literal here, not a
// space) into a bounded buffer, then validate the DECODED id against the
// same grammar every other door uses.
static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool path_segment_decode(const char *in, char *out, size_t cap) {
    size_t n = 0;
    for (const char *p = in; *p; p++) {
        int c = (unsigned char)*p;
        if (c == '%') {
            int h = hexval(p[1]), l = p[1] ? hexval(p[2]) : -1;
            if (h < 0 || l < 0) return false;
            c = h * 16 + l;
            p += 2;
        }
        if (n + 1 >= cap) return false;
        out[n++] = (char)c;
    }
    out[n] = 0;
    return true;
}

static void handle_context_snapshot(slot_t *s, sock_t fd, jv *req,
                                    const char *path) {
    char id[PFX_CTX_NAME_MAX + 1], raw[3 * PFX_CTX_NAME_MAX + 1];
    const char *p = path + sizeof("/v1/runner/contexts/") - 1;
    const char *end = strchr(p, '/');
    size_t n = end ? (size_t)(end - p) : 0;
    if (!end || strcmp(end, "/snapshot") != 0 || n == 0 || n >= sizeof raw) {
        send_error_detail(fd, 400, "not a context id", "id", "invalid_value");
        return;
    }
    memcpy(raw, p, n);
    raw[n] = 0;
    if (!path_segment_decode(raw, id, sizeof id) || !prefix_context_name_ok(id)) {
        send_error_detail(fd, 400, "not a context id", "id", "invalid_value");
        return;
    }
    sbuf out = {0};
    kvsnap_err err = {0};
    bool ok = kvsnap_save(&s->e, id, req, &out, &err);
    send_kvsnap(fd, ok, &out, &err);
}

// ---- session images over HTTP (R1.3.5) -----------------------------------
// POST /v1/runner/sessions starts a raw-prompt generation and, with
// suspend_after, stops it after that many tokens and writes its image;
// POST /v1/runner/sessions/{id}/resume continues an image exactly, or forks
// it under another seed; GET and DELETE /v1/runner/sessions/{id} read and
// remove one. An id is the image's own sha256, so the same state is the same
// id. The images live in --sessions DIR and are the CLI's (--session-out /
// --resume read and write the same files).
//
// The generation holds the device turn from prefill to its last token, so
// a session does not interleave with other slots' batches: it is the solo
// step loop the CLI runs, which is what makes a resume exact.

static int session_text_cb(void *ud, const char *bytes, int n) {
    sb_put((sbuf *)ud, bytes, (size_t)n);
    return 0;
}

static bool session_id_ok(const char *id, size_t n) {
    if (n != 64) return false;
    for (size_t i = 0; i < n; i++)
        if (!((id[i] >= '0' && id[i] <= '9') || (id[i] >= 'a' && id[i] <= 'f')))
            return false;
    return true;
}

static void session_path(char *out, size_t cap, const char *id) {
    snprintf(out, cap, "%s/%s.session", sessions_dir(), id);
}

// Why this slot's model cannot hold a session, or NULL.
static const char *session_unsupported(const model_t *m) {
    if (model_kv_ring_active(m) || m->tied_v)
        return "this model's KV layout (a ring or tied-V cache) has no "
               "contiguous state to image";
    if (model_has_recurrent(m) && m->gpu)
        return "a recurrent model on a device keeps state the host cannot "
               "restore; serve it with --gpu off to use sessions";
    return NULL;
}

// The generation from here: run to the stop point or the end, image a
// generation that is still live at a stop point, answer.
static void session_answer(slot_t *s, sock_t fd, float *logits, int n_prompt,
                           int max_new, int stop_at, bool ignore_eos,
                           const char *parent, bool forked) {
    engine *e = &s->e;
    sbuf text = {0};
    const float *last = NULL;
    double t0 = now_s();
    bool live = session_run(e, logits, stop_at, session_text_cb, &text, &last);
    bool suspended = live && stop_at > 0 && e->gen_count < e->gen_max;
    // Read before engine_reset below zeroes them: a suspended session
    // answered tokens:0, generated:0 while its image carried the real
    // counts (found 2026-10-05).
    int tokens_total = e->pos, generated = e->gen_count;
    bool hit_stop = e->hit_stop;
    char id[65] = "";
    const char *why = NULL;
    if (suspended) {
        char msha[65], bsha[65], tmp[1200], final[1200];
        if (!provenance_digests(msha, bsha)) {
            why = "the resident model's digest is not available (the file "
                  "changed on disk since the load?), so no image was written";
        } else {
            snprintf(tmp, sizeof tmp, "%s/.partial-%d-%lld.session",
                     sessions_dir(), s->id, (long long)(t0 * 1e6));
            if (!session_image_write(tmp, e, s->m->path, msha, bsha, last,
                                     n_prompt, max_new, false, ignore_eos,
                                     NULL, id)) {
                why = "the image could not be written (see the server log)";
            } else {
                session_path(final, sizeof final, id);
                if (!plat_replace_file(tmp, final)) {
                    remove(tmp);
                    id[0] = 0;
                    why = "the image could not be installed in --sessions";
                }
            }
        }
        // the slot does not carry a half-finished generation into the next
        // request: its state is in the image
        engine_reset(e);
    } else {
        engine_gen_end(e, session_text_cb, &text, NULL);
    }
    double secs = now_s() - t0;
    if (why) {
        free(text.s);
        send_error_detail(fd, 500, why, NULL, "session_image_failed");
        return;
    }
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"runner.session\",\"id\":");
    if (id[0]) sb_fmt(&r, "\"%s\"", id); else sb_lit(&r, "null");
    sb_lit(&r, ",\"parent\":");
    if (parent) sb_fmt(&r, "\"%s\"", parent); else sb_lit(&r, "null");
    sb_fmt(&r, ",\"forked\":%s,\"text\":\"", forked ? "true" : "false");
    sb_esc(&r, text.s ? text.s : "", text.n);
    sb_fmt(&r, "\",\"tokens\":%d,\"generated\":%d,\"max_tokens\":%d,"
               "\"finish_reason\":\"%s\",\"seconds\":%.6f}",
           tokens_total, generated, max_new,
           suspended ? "suspended" : hit_stop ? "stop" : "length", secs);
    free(text.s);
    if (r.failed) { free(r.s); send_error(fd, 500, "out of memory"); return; }
    send_response(fd, 200, "application/json", r.s, r.n);
    free(r.s);
    fprintf(stderr, "[slot %d] session%s%s: %d tokens, %d of %d generated%s\n",
            s->id, parent ? " resume " : " ", parent ? parent : "", tokens_total,
            generated, max_new, id[0] ? ", imaged" : "");
}

static bool session_int(sock_t fd, jv *req, const char *key, int lo, int hi,
                        int *out, bool required) {
    jv *v = jv_get(req, key);
    if (!v || v->type == J_NULL) {
        if (!required) return true;
        char msg[96];
        snprintf(msg, sizeof msg, "%s is required", key);
        send_error_detail(fd, 400, msg, key, "invalid_value");
        return false;
    }
    if (v->type != J_NUM || v->num != (double)(long long)v->num ||
        v->num < lo || v->num > hi) {
        char msg[128];
        snprintf(msg, sizeof msg, "%s must be an integer in [%d, %d]", key, lo, hi);
        send_error_detail(fd, 400, msg, key, "invalid_value");
        return false;
    }
    *out = (int)v->num;
    return true;
}

// A sampler knob from the request body: absent or null keeps the default;
// anything else must be a finite number inside [lo, hi]. jv_num() took a
// string as the default and a negative temperature as itself, so a session
// could start on an unsamplable setting that a chat request would refuse.
static bool session_float(sock_t fd, jv *req, const char *key, double lo,
                          double hi, float *out) {
    jv *v = jv_get(req, key);
    if (!v || v->type == J_NULL) return true;
    if (v->type != J_NUM || !isfinite(v->num) || v->num < lo || v->num > hi) {
        char msg[128];
        snprintf(msg, sizeof msg, "%s must be a number in [%g, %g]", key, lo, hi);
        send_error_detail(fd, 400, msg, key, "invalid_value");
        return false;
    }
    *out = (float)v->num;
    return true;
}

// A seed is a positive integer the rng can hold exactly: a double carries
// 53 bits, so the range is capped there rather than cast past it.
static bool session_seed(sock_t fd, jv *req, const char *key, uint64_t *out) {
    jv *v = jv_get(req, key);
    if (!v || v->type == J_NULL) return true;
    if (v->type != J_NUM || v->num < 1 || v->num > 9007199254740991.0 ||
        v->num != (double)(long long)v->num) {
        char msg[128];
        snprintf(msg, sizeof msg, "%s must be a positive integer below 2^53", key);
        send_error_detail(fd, 400, msg, key, "invalid_value");
        return false;
    }
    *out = (uint64_t)v->num;
    return true;
}

static void handle_session_start(slot_t *s, sock_t fd, jv *req) {
    engine *e = &s->e;
    const char *no = session_unsupported(s->m);
    if (no) { send_error_detail(fd, 409, no, NULL, "session_unsupported"); return; }
    jv *prompt = jv_get(req, "prompt");
    if (!prompt || prompt->type != J_STR) {
        send_error_detail(fd, 400, "a session starts from a raw prompt "
                          "(\"prompt\", a string)", "prompt", "invalid_value");
        return;
    }
    int max_new = 0, stop_at = 0, top_k = s->smp_base.top_k;
    if (!session_int(fd, req, "max_tokens", 1, s->m->n_ctx, &max_new, true) ||
        !session_int(fd, req, "suspend_after", 1, s->m->n_ctx, &stop_at, false) ||
        !session_int(fd, req, "top_k", 0, 1 << 20, &top_k, false))
        return;
    if (stop_at && stop_at >= max_new) {
        send_error_detail(fd, 400, "suspend_after must stop before max_tokens "
                          "is spent", "suspend_after", "invalid_value");
        return;
    }
    jv *ie = jv_get(req, "ignore_eos");
    if (ie && ie->type != J_NULL && ie->type != J_BOOL) {
        send_error_detail(fd, 400, "ignore_eos must be a boolean", "ignore_eos",
                          "invalid_value");
        return;
    }
    bool ignore_eos = ie && ie->type == J_BOOL && ie->b;
    s->smp = s->smp_base;
    s->smp.top_k = top_k;
    if (!session_float(fd, req, "temperature", 0.0, 10.0, &s->smp.temp) ||
        !session_float(fd, req, "top_p", 0.0, 1.0, &s->smp.top_p) ||
        !session_float(fd, req, "min_p", 0.0, 1.0, &s->smp.min_p) ||
        !session_float(fd, req, "repeat_penalty", 0.0, 10.0, &s->smp.repeat_penalty) ||
        !session_seed(fd, req, "seed", &s->smp.rng))
        return;
    int32_t *toks = NULL;
    int n = tok_encode_fit(s->tok, prompt->str, true, TOK_RAW, 0, &toks);
    if (n < 0) { free(toks); send_error(fd, 500, "out of memory tokenizing"); return; }
    if (n < 1 || n + max_new > s->m->n_ctx) {
        free(toks);
        send_error_detail(fd, 400, "the prompt and max_tokens do not fit the "
                          "context window", "max_tokens", "context_length_exceeded");
        return;
    }
    sched_prefill_begin();
    engine_request_defaults(e);   // nothing the previous request set survives
    engine_reset(e);
    e->ignore_eos = ignore_eos;
    float *lg = engine_feed(e, toks, n);
    free(toks);
    if (!lg) {
        sched_prefill_end();
        send_error(fd, 500, "prefill failed (context or memory)");
        return;
    }
    engine_gen_begin(e, max_new);
    session_answer(s, fd, lg, n, max_new, stop_at, ignore_eos, NULL, false);
    e->ignore_eos = false;
    sched_prefill_end();
}

static void handle_session_resume(slot_t *s, sock_t fd, jv *req,
                                  const char *path) {
    engine *e = &s->e;
    const char *p = path + sizeof("/v1/runner/sessions/") - 1;
    const char *end = strchr(p, '/');
    if (!end || strcmp(end, "/resume") != 0 || !session_id_ok(p, (size_t)(end - p))) {
        send_error_detail(fd, 400, "not a session id", "id", "invalid_value");
        return;
    }
    char id[65], file[1200];
    memcpy(id, p, 64);
    id[64] = 0;
    session_path(file, sizeof file, id);
    FILE *f = fopen(file, "rb");
    if (!f) {
        send_error_detail(fd, 404, "no session of that id", "id", "session_not_found");
        return;
    }
    fclose(f);
    session_image img;
    if (!session_read(file, &img)) {
        send_error_detail(fd, 422, "the session image does not verify (see the "
                          "server log)", "id", "session_corrupt");
        return;
    }
    const session_meta *sm = &img.meta;
    char msha[65], bsha[65];
    const char *why = session_unsupported(s->m);
    int stop_at = 0;
    long long fork_seed = 0;
    jv *fs = jv_get(req, "fork_seed");
    if (fs && fs->type != J_NULL) {
        if (fs->type != J_NUM || fs->num < 1 || fs->num != (double)(long long)fs->num) {
            session_image_free(&img);
            send_error_detail(fd, 400, "fork_seed must be a positive integer",
                              "fork_seed", "invalid_value");
            return;
        }
        fork_seed = (long long)fs->num;
    }
    if (!session_int(fd, req, "suspend_after", 1, sm->max_new, &stop_at, false)) {
        session_image_free(&img);
        return;
    }
    if (!why && (sm->json_mode || sm->schema_sha256[0]))
        why = "the image was generated under a JSON constraint, which a "
              "served session does not rebuild; resume it with the CLI";
    if (!why && !provenance_digests(msha, bsha))
        why = "the resident model's digest is not available";
    if (!why && strcmp(msha, sm->model_sha256) != 0)
        why = "the resident model is not the model the image was made with";
    if (!why && e->model_key != sm->model_key)
        why = "this engine's model key (context, KV type, geometry) is not "
              "the image's";
    if (!why && (img.n_vocab != s->m->n_vocab ||
                 img.state_n != prefix_cache_entry_bytes(s->m, sm->n_tokens)))
        why = "the image's state does not fit this model";
    if (!why && stop_at && (stop_at <= sm->generated || stop_at >= sm->max_new))
        why = "suspend_after must lie after the image's point and before its "
              "budget is spent";
    if (!why && sm->temp > 0 && sm->rng == 0 && !fork_seed)
        why = "a sampled image without an rng state";
    if (why) {
        session_image_free(&img);
        send_error_detail(fd, 409, why, NULL, "session_mismatch");
        return;
    }
    s->smp = s->smp_base;
    s->smp.temp = sm->temp; s->smp.top_k = sm->top_k; s->smp.top_p = sm->top_p;
    s->smp.min_p = sm->min_p; s->smp.repeat_penalty = sm->repeat_penalty;
    // a greedy image records no rng (it is never drawn from)
    s->smp.rng = fork_seed ? (uint64_t)fork_seed : sm->rng ? sm->rng : 1;
    sched_prefill_begin();
    engine_request_defaults(e);   // the image, not the last request, is the state
    if (!engine_state_load(e, img.tokens, sm->n_tokens, img.state)) {
        sched_prefill_end();
        session_image_free(&img);
        send_error_detail(fd, 409, "this model's KV layout cannot be resumed",
                          NULL, "session_unsupported");
        return;
    }
    e->ignore_eos = sm->ignore_eos;
    engine_gen_resume(e, sm->max_new, sm->n_prompt, sm->generated);
    float *lg = malloc(sizeof(float) * (size_t)s->m->n_vocab);
    if (!lg) {
        sched_prefill_end();
        session_image_free(&img);
        send_error(fd, 500, "out of memory");
        return;
    }
    memcpy(lg, img.logits, sizeof(float) * (size_t)s->m->n_vocab);
    session_answer(s, fd, lg, sm->n_prompt, sm->max_new, stop_at,
                   sm->ignore_eos, id, fork_seed != 0);
    e->ignore_eos = false;
    sched_prefill_end();
    free(lg);
    session_image_free(&img);
}

// GET /v1/runner/sessions/{id}: the image's header; DELETE removes it.
static void session_route(sock_t fd, const char *method, const char *path) {
    const char *id = path + sizeof("/v1/runner/sessions/") - 1;
    if (!sessions_dir()) {
        send_error_detail(fd, 404, "sessions are off: start the server with "
                          "--sessions DIR", NULL, "sessions_off");
        return;
    }
    if (!session_id_ok(id, strlen(id))) {
        send_error_detail(fd, 400, "not a session id", "id", "invalid_value");
        return;
    }
    char file[1200];
    session_path(file, sizeof file, id);
    if (!strcmp(method, "DELETE")) {
        if (remove(file) != 0) {
            send_error_detail(fd, 404, "no session of that id", "id",
                              "session_not_found");
            return;
        }
        char body[160];
        int bn = snprintf(body, sizeof body, "{\"object\":\"runner.session\","
                          "\"id\":\"%s\",\"deleted\":true}", id);
        send_response(fd, 200, "application/json", body, (size_t)bn);
        return;
    }
    session_image img;
    FILE *f = fopen(file, "rb");
    if (!f) {
        send_error_detail(fd, 404, "no session of that id", "id", "session_not_found");
        return;
    }
    fclose(f);
    if (!session_read(file, &img)) {
        send_error_detail(fd, 422, "the session image does not verify", "id",
                          "session_corrupt");
        return;
    }
    const session_meta *sm = &img.meta;
    char body[768];
    int bn = snprintf(body, sizeof body,
                      "{\"object\":\"runner.session\",\"id\":\"%s\","
                      "\"model_sha256\":\"%s\",\"tokens\":%d,\"prompt_tokens\":%d,"
                      "\"generated\":%d,\"max_tokens\":%d,\"ctx\":%d,\"kv\":\"%s\","
                      "\"temperature\":%g,\"top_k\":%d,\"top_p\":%g,\"min_p\":%g,"
                      "\"repeat_penalty\":%g,\"ignore_eos\":%s,"
                      "\"binary_sha256\":\"%s\"}",
                      id, sm->model_sha256, sm->n_tokens, sm->n_prompt,
                      sm->generated, sm->max_new, sm->n_ctx, sm->kv_type,
                      (double)sm->temp, sm->top_k, (double)sm->top_p,
                      (double)sm->min_p, (double)sm->repeat_penalty,
                      sm->ignore_eos ? "true" : "false", sm->binary_sha256);
    session_image_free(&img);
    send_response(fd, 200, "application/json", body, (size_t)bn);
}

static void handle_context_create(slot_t *s, sock_t fd, jv *req) {
    const char *id = jv_str(jv_get(req, "id"), NULL);
    if (!id || !prefix_context_name_ok(id)) {
        send_error_detail(fd, 400, "id must be 1 to 64 characters of "
                          "[A-Za-z0-9._:-]", "id", "invalid_value");
        return;
    }
    jv *prompt = jv_get(req, "prompt");
    jv *msgs = jv_get(req, "messages");
    jv *snap = jv_get(req, "snapshot");
    bool has_p = prompt && prompt->type != J_NULL;
    bool has_m = msgs && msgs->type != J_NULL;
    if (snap && snap->type != J_NULL) {
        if (has_p || has_m) {
            send_error(fd, 400, "a context is a prompt, messages or a snapshot: "
                                "give exactly one");
            return;
        }
        context_from_snapshot(s, fd, id, snap);
        return;
    }
    if (has_p == has_m) {
        send_error(fd, 400, "a context is either a raw prompt (\"prompt\") or "
                            "chat messages (\"messages\", with \"tools\"): "
                            "give exactly one");
        return;
    }
    if (has_p) {
        if (prompt->type != J_STR) {
            send_error_detail(fd, 400, "prompt must be a string", "prompt",
                              "invalid_type");
            return;
        }
        context_pin_prompt(s, fd, prompt->str, false, req);
        return;
    }
    // rendered WITHOUT the assistant generation prompt: the context is what
    // a later request's messages start with, not a turn to answer
    handle_chat_render(s, fd, req, false, context_from_chat);
}

static void send_contexts(sock_t fd) {
    prefix_context_info *v = NULL;
    int n = prefix_context_list(&v);
    if (n < 0) { send_error(fd, 500, "out of memory"); return; }
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    for (int i = 0; i < n; i++)
        sb_fmt(&r, "%s{\"id\":\"%s\",\"tokens\":%d,\"bytes\":%llu,"
                   "\"hits\":%llu,\"age_seconds\":%.3f}",
               i ? "," : "", v[i].name, v[i].tokens,
               (unsigned long long)v[i].bytes,
               (unsigned long long)v[i].hits, v[i].age_s);
    sb_lit(&r, "]}");
    free(v);
    if (r.failed) { free(r.s); send_error(fd, 500, "out of memory"); return; }
    send_response(fd, 200, "application/json", r.s, r.n);
    free(r.s);
}

static void delete_context(sock_t fd, const char *path) {
    char id[PFX_CTX_NAME_MAX + 1];
    if (!path_segment_decode(path + sizeof("/v1/runner/contexts/") - 1,
                             id, sizeof id) ||
        !prefix_context_name_ok(id)) {
        send_error_detail(fd, 400, "not a context id", "id", "invalid_value");
        return;
    }
    if (!prefix_context_release(id)) {
        send_error_detail(fd, 404, "no context of that id", "id",
                          "context_not_found");
        return;
    }
    char body[160];
    int bn = snprintf(body, sizeof body,
                      "{\"object\":\"runner.context\",\"id\":\"%s\","
                      "\"deleted\":true}", id);
    send_response(fd, 200, "application/json", body, (size_t)bn);
}

// ---- the Responses store (R10.6) ---------------------------------------
// GET /v1/responses/{id}, GET /v1/responses/{id}/input_items and
// DELETE /v1/responses/{id}. Bodyless, answered from the accept thread: the
// store has its own lock and holds no model state.
static bool response_id_ok(const char *id, size_t n) {
    if (n == 0 || n > 64) return false;
    for (size_t i = 0; i < n; i++) {
        char c = id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-'))
            return false;
    }
    return true;
}

static void stored_response_route(sock_t fd, const char *method,
                                  const char *path) {
    const char *id = path + sizeof("/v1/responses/") - 1;
    const char *slash = strchr(id, '/');
    size_t idn = slash ? (size_t)(slash - id) : strlen(id);
    bool items = slash && !strcmp(slash, "/input_items");
    char key[65];
    if (!response_id_ok(id, idn) || (slash && !items) ||
        (items && strcmp(method, "GET"))) {
        send_error_detail(fd, 404, "no such route", NULL, "not_found");
        return;
    }
    memcpy(key, id, idn);
    key[idn] = 0;
    if (!strcmp(method, "DELETE")) {
        if (!respstore_delete(key)) {
            send_error_detail(fd, 404, "no stored response of that id",
                              "response_id", "response_not_found");
            return;
        }
        char body[160];
        int bn = snprintf(body, sizeof body, "{\"id\":\"%s\",\"object\":"
                          "\"response.deleted\",\"deleted\":true}", key);
        send_response(fd, 200, "application/json", body, (size_t)bn);
        return;
    }
    size_t n = 0;
    char *doc = items ? respstore_input(key, &n) : respstore_body(key, &n);
    if (!doc) {
        send_error_detail(fd, 404, "no stored response of that id (never "
                          "stored, deleted, or expired)", "response_id",
                          "response_not_found");
        return;
    }
    if (items) {
        sbuf r = {0};
        sb_lit(&r, "{\"object\":\"list\",\"data\":");
        sb_put(&r, doc, n);
        sb_lit(&r, "}");
        free(doc);
        if (r.failed) { free(r.s); send_error(fd, 500, "out of memory"); return; }
        send_response(fd, 200, "application/json", r.s, r.n);
        free(r.s);
        return;
    }
    send_response(fd, 200, "application/json", doc, n);
    free(doc);
}

static void handle_completion(slot_t *s, sock_t fd, jv *req) {
    const char *prompt = jv_str(jv_get(req, "prompt"), NULL);
    if (!prompt) { send_error(fd, 400, "missing prompt"); return; }
    run_completion(s, fd, prompt, API_TEXT, req, NULL);
}

// One byte of the vector as the wire sees it: little-endian float32, spelled
// out rather than memcpy'd wholesale so a big-endian host emits the same bytes.
static unsigned char emb_byte(const float *v, size_t i) {
    uint32_t bits;
    memcpy(&bits, &v[i >> 2], sizeof bits);
    return (unsigned char)(bits >> (8 * (i & 3)));
}

// base64 of that byte stream. Not an optimisation we chose: the OpenAI SDKs
// send `encoding_format: "base64"` by DEFAULT and decode it client-side, so a
// server that only speaks "float" cannot be called by an official client at
// all. Refusing it was a 400 on every SDK embeddings call.
static void sb_emb_b64(sbuf *r, const float *v, int n) {
    static const char T[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t nb = (size_t)n * sizeof(float);
    for (size_t i = 0; i < nb; i += 3) {
        unsigned char b0 = emb_byte(v, i);
        unsigned char b1 = i + 1 < nb ? emb_byte(v, i + 1) : 0;
        unsigned char b2 = i + 2 < nb ? emb_byte(v, i + 2) : 0;
        char q[4];
        q[0] = T[b0 >> 2];
        q[1] = T[((b0 & 0x03) << 4) | (b1 >> 4)];
        q[2] = i + 1 < nb ? T[((b1 & 0x0F) << 2) | (b2 >> 6)] : '=';
        q[3] = i + 2 < nb ? T[b2 & 0x3F] : '=';
        sb_put(r, q, 4);
    }
}

// POST /v1/decide (R13.10): typed decisions, one prefill per state, zero
// sampled tokens. The scorer runs on this slot's engine under the device
// turn like embeddings; the KV is left holding a valid prefix, so a
// following request on the slot rewinds into it like any other.
static void handle_decide(slot_t *s, sock_t fd, jv *req) {
    sbuf r = {0};
    const char *err = NULL;
    sched_prefill_begin();
    int st = decide_handle(&s->e, req, SV.model_name, &r, &err);
    sched_prefill_end();
    if (st != 200) {
        free(r.s);
        send_error(fd, st, err ? err : "decide failed");
        return;
    }
    send_built(fd, &r);
    fprintf(stderr, "[slot %d] decide: %d question(s)\n", s->id,
            jv_get(req, "questions") ? jv_get(req, "questions")->n : 0);
    free(r.s);
}

static void handle_rerank(slot_t *s, sock_t fd, jv *req) {
    sbuf r = {0};
    const char *err = NULL;
    sched_prefill_begin();
    int st = rerank_handle(&s->e, req, SV.model_name, s->tmpl, &r, &err);
    sched_prefill_end();
    if (st != 200) {
        free(r.s);
        send_error(fd, st, err ? err : "rerank failed");
        return;
    }
    send_built(fd, &r);
    fprintf(stderr, "[slot %d] rerank: %d document(s)\n", s->id,
            jv_get(req, "documents") ? jv_get(req, "documents")->n : 0);
    free(r.s);
}

static void handle_embeddings(slot_t *s, sock_t fd, jv *req) {
    jv *input = jv_get(req, "input");
    const char *one = jv_str(input, NULL);
    int n_in = one ? 1 : (input && input->type == J_ARR ? input->n : 0);
    if (n_in == 0) { send_error(fd, 400, "missing input"); return; }

    model_t *m = s->m;
    // The checkpoint says how its vectors are read out. Mean and last are
    // implemented; a model that declares another readout (a CLS token, a
    // reranker's score head) is refused by name rather than mean-pooled into
    // a vector its publisher never defined.
    if (!model_pooling_supported(m)) {
        char msg[160];
        snprintf(msg, sizeof msg,
                 "this model declares %s pooling (%s.pooling_type = %u); "
                 "/v1/embeddings implements mean and last",
                 model_pooling_name(m->pooling_type),
                 m->arch, (unsigned)m->pooling_type);
        send_error(fd, 400, msg);
        return;
    }
    jv *encoding = jv_get(req, "encoding_format");
    bool b64 = false;
    if (!absent(encoding)) {
        if (encoding->type != J_STR) {
            send_error(fd, 400, "encoding_format must be float or base64");
            return;
        }
        if (strcmp(encoding->str, "base64") == 0) b64 = true;
        else if (strcmp(encoding->str, "float") != 0) {
            send_error(fd, 400, "encoding_format must be float or base64");
            return;
        }
    }
    jv *dimensions = jv_get(req, "dimensions");
    if (!absent(dimensions) &&
        (dimensions->type != J_NUM || !isfinite(dimensions->num) ||
         dimensions->num != m->n_embd)) {
        send_error(fd, 400, "dimensions must equal the model embedding size");
        return;
    }
    float *emb = malloc(sizeof(float) * m->n_embd);
    if (!emb) { send_error(fd, 500, "out of memory"); return; }  // RNS-3
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    int total = 0;
    bool ok = true;
    // RNS-1: an error must be reported to the client only AFTER the device turn
    // is released. send_error() does a blocking socket write bounded by the 30s
    // SO_SNDTIMEO; doing it under dev_mu would stall decode_worker — and thereby
    // every other in-flight generation — behind one slow/dead embeddings client.
    // So the loop records (err_code, err_msg) and breaks; the send happens below,
    // off-lock, mirroring run_completion.
    int err_code = 0;
    const char *err_msg = NULL;
    // model_embed launches kernels and engine_reset touches this slot's KV —
    // the same device work every other handler serializes under dev_mu. Take the
    // device turn so an embeddings request cannot launch while decode_worker is
    // capturing a CUDA graph (hazard at dev_mu, ~line 596). Every exit from the
    // loop below is a break, so the single sched_prefill_end() after the loop
    // always runs — the lock is never left held.
    sched_prefill_begin();
    for (int k = 0; k < n_in && ok; k++) {
        const char *txt = one ? one : jv_str(input->items[k], NULL);
        if (!txt) { err_code = 400; err_msg = "input must be strings"; ok = false; break; }
        // an embedding input has no template: every byte is the caller's
        // text, and a control token spelled in it is those characters
        int32_t *toks = NULL;
        int n = tok_encode_fit(s->tok, txt, true, TOK_TEXT, 1, &toks);
        if (n < 0) {
            free(toks);
            err_code = 500; err_msg = "out of memory tokenizing input";
            ok = false;
            break;
        }
        // an embedding model's tokenizer that appends its end token does so
        // here too: last-token pooling reads out exactly that position
        if (s->tok->add_eos && s->tok->eos_id >= 0 && n > 0)
            toks[n++] = s->tok->eos_id;
        if (n == 0 || !model_embed(m, toks, n, emb)) {
            free(toks);
            err_code = 400;
            err_msg = n == 0 ? "empty input" : "input exceeds context window";
            ok = false;
            break;
        }
        free(toks);
        total += n;
        sb_fmt(&r, "%s{\"object\":\"embedding\",\"index\":%d,\"embedding\":",
               k ? "," : "", k);
        if (b64) {
            sb_lit(&r, "\"");
            sb_emb_b64(&r, emb, m->n_embd);
            sb_lit(&r, "\"");
        } else {
            // nine significant digits: the fewest that read every float32
            // back exactly, so this list and the base64 bytes are one vector
            sb_lit(&r, "[");
            for (int j = 0; j < m->n_embd; j++)
                sb_fmt(&r, "%s%.9g", j ? "," : "", (double)emb[j]);
            sb_lit(&r, "]");
        }
        sb_lit(&r, "}");
    }
    // model_embed overwrote this slot's KV cache — invalidate the prefix cache
    engine_reset(&s->e);
    s->e.pos = 0;
    sched_prefill_end();
    if (!ok) {
        // deferred from the loop — sent here, off the device lock (RNS-1)
        send_error(fd, err_code, err_msg);
    } else {
        sb_lit(&r, "],\"model\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_fmt(&r, "\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}",
               total, total);
        send_built(fd, &r);
        fprintf(stderr, "[slot %d] embeddings: %d input(s), %d tok\n",
                s->id, n_in, total);
    }
    free(r.s);
    free(emb);
}

// ---------------------------------------------------------------- http

// The build that answers, on /health and /v1/capabilities: the --version
// string, so a supervisor can name the build in its own records and refuse one
// below its floor without inferring it from the feature set. A T3 build also
// names its flavor, as its receipts do; the default build writes none.
#ifdef RUNNER_T3_BUILD
#define BUILD_JSON "\"version\":\"" RUNNER_VERSION "\",\"build_flavor\":\"t3\""
#else
#define BUILD_JSON "\"version\":\"" RUNNER_VERSION "\""
#endif

// /health and /v1/models read only startup-immutable strings plus an atomic
// resident snapshot, so they are safe to answer from the accept thread with no lock
static void send_health(sock_t fd) {
    char b[2304];
    int n, res = resident_load();
    // Inference requests in flight. "A model is loaded" and "the model is
    // working" look identical from outside the process, and the tray needs to
    // tell them apart to show the right glyph. This is the count the server
    // already keeps for swap and unload decisions, so reading it costs an
    // atomic load and adds nothing to the request path. /health is not itself
    // counted (the increment at dispatch covers the inference routes only), so
    // a poller never sees its own request here.
    int active = atomic_load(&SV.active_requests);
    // Process cost and cumulative work, for a supervisor budgeting several
    // runners. RSS is the process total -- weights, KV cache, activations and
    // allocator overhead -- which is the number a machine is sized against and
    // which no mapping-level measure accounts for. The token and second totals
    // are monotonic so a dashboard can difference them over its own window.
    work_totals wt;
    server_work_totals(&wt);
    // How much of that work was batched. The scheduler already counts its
    // microbatch steps and the sequences cut into them; batch_sequences over
    // batch_steps is the mean batch size, and it is the only wire-visible
    // answer to whether continuous batching is earning its decode thread on
    // this box. Reported as the same kind of raw monotonic pair as the token
    // totals, for the same reason: the averaging window belongs to whoever is
    // asking. Both stay 0 on a server that never started the scheduler.
    unsigned long long bs = 0, bq = 0;
    sched_batch_totals(&bs, &bq);
    // The two readings come from different kernel counters on Linux (statm
    // for current, ru_maxrss for peak) whose split-RSS accounting can lag
    // each other, so a raw pair can report current above peak — observed
    // under ASan's mapping churn in the sanitized conformance leg. A
    // current reading of X is itself evidence the peak is at least X, so
    // the pair is made self-consistent at the moment it is reported.
    uint64_t cur_rss = plat_proc_rss_bytes();
    uint64_t peak_rss = plat_proc_peak_rss_bytes();
    if (peak_rss < cur_rss) peak_rss = cur_rss;
    // How many loads this process has made (R10.12.1): it moves on every
    // load, reload and swap, so a client that reads it twice and sees the
    // same number with a resident model has been talking to one load. A
    // request can require it (`expect_resident`).
    char m[448];
    snprintf(m, sizeof(m),
             ",\"load_generation\":%llu"
             ",\"rss_bytes\":%llu,\"peak_rss_bytes\":%llu,"
             "\"tokens_prompt\":%llu,\"tokens_generated\":%llu,"
             "\"generate_seconds\":%.6f,"
             "\"batch_steps\":%llu,\"batch_sequences\":%llu",
             (unsigned long long)provenance_load_generation(NULL),
             (unsigned long long)cur_rss,
             (unsigned long long)peak_rss,
             wt.prompt_tokens, wt.gen_tokens, wt.gen_seconds, bs, bq);

    // What each busy slot is doing (R10.12.7): a client waiting on a long
    // prefill can tell it from a hang, and one that closed its connection can
    // see the request leave. Headers of a streamed reply are only sent once
    // prefill is over, so this is the one place that progress is visible.
    char rq[1024];
    int rn = snprintf(rq, sizeof rq, ",\"requests\":[");
    bool first_rq = true;
    for (int i = 0; SV.slots && i < SV.n_slots; i++) {
        slot_t *sl = &SV.slots[i];
        int ph = atomic_load(&sl->rq_phase);
        if (!ph) continue;
        int w = snprintf(rq + rn, sizeof rq - (size_t)rn,
                         "%s{\"slot\":%d,\"phase\":\"%s\",\"prompt_tokens\":%d,"
                         "\"prompt_done\":%d,\"generated\":%d}",
                         first_rq ? "" : ",", i, ph == 1 ? "prefill" : "generate",
                         atomic_load(&sl->rq_prompt), atomic_load(&sl->rq_done),
                         atomic_load(&sl->rq_gen));
        // a row that does not fit is left out whole; the count above it
        // (active_requests) still says how many there are
        if (w < 0 || (size_t)w >= sizeof rq - (size_t)rn - 2) break;
        rn += w;
        first_rq = false;
    }
    snprintf(rq + rn, sizeof rq - (size_t)rn, "]");
    if (SV.n_reg > 0 && res >= 0) {
        // Registry names are char[64]. Match /v1/models' exact worst-case
        // bound so /health cannot identify the same resident by a truncated
        // string when every byte needs a six-byte JSON escape.
        char esc[63 * 6 + 2];
        json_escape(SV.reg[res].name, strlen(SV.reg[res].name), esc, sizeof(esc));
        n = snprintf(b, sizeof(b),
                     "{\"status\":\"ok\"," BUILD_JSON ",\"resident\":\"%s\","
                     "\"active_requests\":%d%s%s}", esc, active, m, rq);
    } else if (SV.n_reg > 0) {
        n = snprintf(b, sizeof(b), "{\"status\":\"ok\"," BUILD_JSON ",\"resident\":null,"
                                   "\"active_requests\":%d%s%s}", active, m, rq);
    } else {
        n = snprintf(b, sizeof(b),
                     "{\"status\":\"ok\"," BUILD_JSON ",\"active_requests\":%d%s%s}",
                     active, m, rq);
    }
    send_response(fd, 200, "application/json", b, n);
}

static void send_models(sock_t fd) {
    sbuf r = {0};
    sb_lit(&r, "{\"object\":\"list\",\"data\":[");
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            // worst-case escape is \uXXXX per byte, so (capacity-1)*6+1
            // bounds the output exactly — a smaller buffer truncated
            // operator-chosen names mid-escape in the JSON id
            char esc[(sizeof(SV.reg[0].name) - 1) * 6 + 2];
            json_escape(SV.reg[i].name, strlen(SV.reg[i].name), esc, sizeof(esc));
            sb_fmt(&r, "%s{\"id\":\"%s\",\"object\":\"model\","
                       "\"owned_by\":\"runner\"}", i ? "," : "", esc);
        }
    } else {
        // the single-model id is a path basename: 255 bytes on every real
        // filesystem, worst-case \uXXXX escape per byte bounds the output
        char esc[255 * 6 + 2];
        json_escape(SV.model_name, strlen(SV.model_name), esc, sizeof(esc));
        sb_fmt(&r, "{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"runner\"}", esc);
    }
    // R8.6: every loaded adapter is a model id of its own
    for (int i = 0; i < SV.n_adapters; i++) {
        const char *bn = SV.n_reg > 0 ? SV.reg[0].name : SV.model_name;
        sb_lit(&r, ",{\"id\":\"");
        sb_esc(&r, bn, strlen(bn));
        sb_lit(&r, ":");
        sb_esc(&r, SV.adapters[i].name, strlen(SV.adapters[i].name));
        sb_lit(&r, "\",\"object\":\"model\",\"owned_by\":\"runner\"}");
    }
    sb_lit(&r, "]}");
    send_built(fd, &r);
    free(r.s);
}

// Server-wide prefix-cache telemetry. Per-request telemetry says what one
// request saved; this says whether the cache is earning its memory —
// hit rate, resident bytes against the budget, and the prefill time it has
// avoided so far.
static void send_prefix_cache(sock_t fd) {
    prefix_cache_stats st;
    prefix_cache_stats_get(&st);
    char b[640];
    int n = snprintf(b, sizeof(b),
        "{\"object\":\"runner.prefix_cache\","
        "\"enabled\":%s,\"entries\":%d,\"bytes\":%llu,\"budget_bytes\":%llu,"
        "\"ttl_seconds\":%.1f,\"hits\":%llu,\"misses\":%llu,\"stores\":%llu,"
        "\"evictions\":%llu,\"tokens_reused\":%llu,"
        "\"saved_prefill_seconds\":%.6f,\"prefill_seconds_per_token\":%.9f}",
        st.budget ? "true" : "false", st.entries,
        (unsigned long long)st.bytes, (unsigned long long)st.budget, st.ttl,
        (unsigned long long)st.hits, (unsigned long long)st.misses,
        (unsigned long long)st.stores, (unsigned long long)st.evictions,
        (unsigned long long)st.tokens_reused,
        st.saved_prefill_s, st.cost_per_token_s);
    send_response(fd, 200, "application/json", b, n);
}

// ------------------------------------------------------------------ /metrics
//
// The same facts /health and /v1/runner/prefix-cache already answer, in the
// one format a monitoring stack ingests without a translator. Prometheus text
// exposition 0.0.4: every sample is preceded by its own `# HELP` and `# TYPE`,
// names carry the `runner_` prefix, and a counter's name ends in `_total`.
//
// It reports; it does not compute. There is no hit-RATE and no tokens-per-
// SECOND here for the same reason /health has none: a rate needs an averaging
// window, and the scraper owns that window. Every number below is either a
// monotonic counter to be differenced or an instantaneous gauge.
//
// Read-only and lock-free apart from the prefix cache's own mutex, which
// `GET /v1/runner/prefix-cache` already takes from the accept thread. Nothing
// here reads the resident model, so unlike /v1/capabilities it needs no
// swap_mu (see the note there) and is answered on the accept path with
// atomic loads only.

typedef struct { char *p; size_t cap, n; bool over; } mbuf;

static void m_fmt(mbuf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void m_fmt(mbuf *b, const char *fmt, ...) {
    if (b->over) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    // A truncated exposition is not a smaller exposition: a scraper reads a
    // missing metric as one that reset, and a half-written line as a parse
    // error for the whole scrape. Refuse the body instead.
    if (n < 0 || (size_t)n >= b->cap - b->n) b->over = true;
    else b->n += (size_t)n;
}

static void metric_u64(mbuf *b, const char *name, const char *type,
                       const char *help, unsigned long long v) {
    m_fmt(b, "# HELP %s %s\n# TYPE %s %s\n%s %llu\n", name, help, name, type,
          name, v);
}

static void metric_f64(mbuf *b, const char *name, const char *type,
                       const char *help, double v) {
    m_fmt(b, "# HELP %s %s\n# TYPE %s %s\n%s %.6f\n", name, help, name, type,
          name, v);
}

static void send_metrics(sock_t fd) {
    work_totals wt;
    server_work_totals(&wt);
    unsigned long long bs = 0, bq = 0;
    sched_batch_totals(&bs, &bq);
    prefix_cache_stats st;
    prefix_cache_stats_get(&st);
    // Same self-consistency fix /health applies, and for the same reason: the
    // two readings come from different kernel counters whose split-RSS
    // accounting can lag each other, and a peak below the current reading is
    // an impossible pair to publish.
    uint64_t cur_rss = plat_proc_rss_bytes();
    uint64_t peak_rss = plat_proc_peak_rss_bytes();
    if (peak_rss < cur_rss) peak_rss = cur_rss;

    char buf[8192];
    mbuf b = { buf, sizeof(buf), 0, false };

    metric_u64(&b, "runner_requests_total", "counter",
               "Inference requests admitted since start. Management and "
               "telemetry routes are not counted.",
               atomic_load(&SV.total_requests));
    metric_u64(&b, "runner_prompt_tokens_total", "counter",
               "Prompt tokens accepted across every API surface.",
               wt.prompt_tokens);
    metric_u64(&b, "runner_prompt_cached_tokens_total", "counter",
               "Prompt tokens served from a reused KV prefix, included in "
               "runner_prompt_tokens_total.", wt.cached_tokens);
    metric_u64(&b, "runner_generated_tokens_total", "counter",
               "Tokens generated across every API surface.", wt.gen_tokens);
    metric_f64(&b, "runner_generate_seconds_total", "counter",
               "Seconds spent in generation, for differencing against "
               "runner_generated_tokens_total.", wt.gen_seconds);
    metric_u64(&b, "runner_batch_steps_total", "counter",
               "Scheduler microbatch steps.", bs);
    metric_u64(&b, "runner_batch_sequences_total", "counter",
               "Sequences cut into those microbatch steps.", bq);
    metric_u64(&b, "runner_prefix_cache_hits_total", "counter",
               "Shared prefix-cache lookups that forked a snapshot.", st.hits);
    metric_u64(&b, "runner_prefix_cache_misses_total", "counter",
               "Shared prefix-cache lookups that found nothing to fork.",
               st.misses);
    metric_u64(&b, "runner_prefix_cache_stores_total", "counter",
               "Prefixes published to the shared store.", st.stores);
    metric_u64(&b, "runner_prefix_cache_evictions_total", "counter",
               "Prefixes dropped for budget or age.", st.evictions);
    metric_u64(&b, "runner_prefix_cache_tokens_reused_total", "counter",
               "Prompt tokens forked out of the shared store.",
               st.tokens_reused);
    metric_f64(&b, "runner_prefix_cache_saved_prefill_seconds_total", "counter",
               "Prefill seconds those forks avoided, priced at this process's "
               "measured seconds per token.", st.saved_prefill_s);
    metric_u64(&b, "runner_speculation_rounds_total", "counter",
               "Speculative verify rounds. Zero on a server with no draft, "
               "no MTP head and no grammar fast-forward.", wt.spec_rounds);
    metric_u64(&b, "runner_speculation_drafted_tokens_total", "counter",
               "Tokens proposed by a draft source.", wt.spec_drafted);
    metric_u64(&b, "runner_speculation_accepted_tokens_total", "counter",
               "Proposed tokens the target model accepted.", wt.spec_accepted);
    metric_u64(&b, "runner_active_requests", "gauge",
               "Inference requests in flight right now.",
               (unsigned long long)(long long)atomic_load(&SV.active_requests));
    metric_u64(&b, "runner_resident_memory_bytes", "gauge",
               "Resident set of this process: weights, KV cache, activations "
               "and allocator overhead together.",
               (unsigned long long)cur_rss);
    metric_u64(&b, "runner_peak_resident_memory_bytes", "gauge",
               "High-water mark of that resident set.",
               (unsigned long long)peak_rss);
    metric_u64(&b, "runner_prefix_cache_bytes", "gauge",
               "Host memory the shared prefix store holds.",
               (unsigned long long)st.bytes);
    metric_u64(&b, "runner_prefix_cache_budget_bytes", "gauge",
               "Its configured ceiling; 0 disables the shared tier.",
               (unsigned long long)st.budget);
    metric_u64(&b, "runner_prefix_cache_entries", "gauge",
               "Snapshots currently held.", (unsigned long long)st.entries);

    if (b.over) {
        send_error(fd, 500, "metrics exposition did not fit its buffer");
        return;
    }
    send_response(fd, 200, "text/plain; version=0.0.4", buf, b.n);
}

// GET /v1/runner/provenance (R10.2.1): the statement a receipt carries, for
// the server as it runs now. The identity half (digests and load-time
// verdicts) is provenance.c's; the effective configuration is read here from
// the resident slot, under swap_mu for the same reason send_capabilities
// takes it -- a swap frees the model this reads.
static void send_provenance(sock_t fd) {
    sbuf r = {0};
    bool guarded = SV.n_reg > 0;
    if (guarded) pthread_mutex_lock(&SV.swap_mu);
    int res = resident_load();
    const char *id = SV.n_reg == 0 ? SV.model_name
                   : res >= 0      ? SV.reg[res].name : NULL;
    const slot_t *s0 = SV.slots ? &SV.slots[0] : NULL;
    const model_t *m = (s0 && (SV.n_reg == 0 || res >= 0)) ? s0->m : NULL;
    sb_lit(&r, "{\"object\":\"runner.provenance\"," BUILD_JSON ",");
    provenance_render(&r, id);
    sb_lit(&r, ",\"profile\":");
    if (!m) {
        sb_lit(&r, "null");
    } else {
        char gname[128] = "cpu";
        if (m->gpu && !gpu_available(gname, (int)sizeof gname))
            snprintf(gname, sizeof gname, "gpu");
        sb_lit(&r, "{\"device\":\"");
        sb_esc(&r, gname, strlen(gname));
        sb_fmt(&r, "\",\"gpu\":%s,\"gpu_layers\":%d,\"threads\":%d,"
                   "\"ctx\":%d,\"kv\":\"%s\",\"batch\":%d,\"slots\":%d}",
               m->gpu ? "true" : "false", m->gpu_layers,
               m->tp ? tpool_size(m->tp) : 0, m->n_ctx,
               m->kv_fp4 ? "fp4" : m->kv_split ? "k8v4"
                         : m->kv_q8 ? "q8" : "f16",
               m->n_batch, SV.n_slots);
    }
    sb_lit(&r, ",\"config\":{");
    if (m) {
        const sampler *d = &s0->smp_base;
        sb_lit(&r, "\"sampling\":{\"preset\":");
        if (SV.preset_name) {
            sb_lit(&r, "\"");
            sb_esc(&r, SV.preset_name, strlen(SV.preset_name));
            sb_lit(&r, "\"");
        } else {
            sb_lit(&r, "null");
        }
        sb_fmt(&r, ",\"temperature\":%g,\"top_k\":%d,\"top_p\":%g,"
                   "\"min_p\":%g,\"repeat_penalty\":%g},",
               (double)d->temp, d->top_k, (double)d->top_p,
               (double)d->min_p, (double)d->repeat_penalty);
        const char *tn = template_name(s0->tmpl);
        sb_lit(&r, "\"template\":\"");
        sb_esc(&r, tn, strlen(tn));
        sb_fmt(&r, "\",\"template_forced\":%s,",
               SV.tmpl_override >= 0 ? "true" : "false");
        const engine *e = &s0->e;
        const char *dsrc = e->dm ? "model" : e->mtp_on ? "mtp"
                         : e->lookup_on ? "lookup" : NULL;
        if (dsrc) sb_fmt(&r, "\"draft\":\"%s\",", dsrc);
        else      sb_lit(&r, "\"draft\":null,");
    }
    sb_fmt(&r, "\"max_tokens_cap\":%d,\"reasoning_budget\":%d,"
               "\"loop_guard\":%s,\"ignore_eos\":%s,"
               "\"request_timeout_s\":%g,",
           SV.n_predict_cap, SV.reasoning_budget,
           SV.loop_guard ? "true" : "false",
           SV.ignore_eos ? "true" : "false", SV.req_timeout);
    if (SV.reasoning_temp_set)
        sb_fmt(&r, "\"reasoning_temperature\":%g,", (double)SV.reasoning_temp);
    // R8.6: the adapters a request may name, each with its digest
    sb_lit(&r, "\"adapters\":[");
    for (int i = 0; i < SV.n_adapters; i++) {
        sb_fmt(&r, "%s{\"name\":\"", i ? "," : "");
        sb_esc(&r, SV.adapters[i].name, strlen(SV.adapters[i].name));
        sb_lit(&r, "\",\"path\":\"");
        sb_esc(&r, SV.adapters[i].path, strlen(SV.adapters[i].path));
        sb_fmt(&r, "\",\"sha256\":\"%s\",\"signature\":", SV.adapters[i].sha256);
        if (SV.adapters[i].sig_json[0])
            sb_put(&r, SV.adapters[i].sig_json, strlen(SV.adapters[i].sig_json));
        else
            sb_lit(&r, "null");
        sb_lit(&r, "}");
    }
    sb_lit(&r, "],");
    sb_fmt(&r, "\"signature_policy\":{\"required\":%s,\"trusted_key\":%s},"
               "\"force_uncertified\":%s}",
           SV.signing.required ? "true" : "false",
           SV.signing.pubkey_path ? "true" : "false",
           SV.force_uncertified ? "true" : "false");
    if (guarded) pthread_mutex_unlock(&SV.swap_mu);
    sb_lit(&r, ",\"statement\":\"Digests and verdicts this process "
               "established for itself: the executable hashed at start, the "
               "model file hashed after its load and re-identified on every "
               "read, the signature and envelope verdicts the load ran under. "
               "This is not an attestation; a process can only report on "
               "itself.\"}");
    if (r.failed) {
        free(r.s);
        send_error(fd, 500, "out of memory building the provenance record");
        return;
    }
    send_response(fd, 200, "application/json", r.s, r.n);
    free(r.s);
}

static void send_capabilities(sock_t fd) {
    sbuf r = {0};
    // Unlike /health and /v1/models, this route reports on the RESIDENT MODEL
    // -- its agent profile, its MTP declaration, the sampling preset resolved
    // for it -- and every one of those is freed by unload_resident() the moment
    // a request names a different model, /unload arrives, or --ttl expires.
    // Answering it from the accept thread with nothing held is a read of freed
    // memory: ASan catches it inside a second of one client alternating models
    // while another polls here (tests/test_swap_race.c).
    //
    // swap_mu is the lock those three writers already take, and it is by design
    // never held across a load (swap_to drops it) or across generation, so the
    // accept loop cannot be parked behind inference here. It exists only once
    // the registry is joined, which is also the only configuration in which a
    // model can go away under a request.
    bool guarded = SV.n_reg > 0;
    if (guarded) pthread_mutex_lock(&SV.swap_mu);
    int res = resident_load();
    sb_fmt(&r, "{\"object\":\"runner.capabilities\"," BUILD_JSON ",\"pid\":%ld,\"swap\":",
           plat_pid_self());
    sb_lit(&r, SV.n_reg > 0 && !SV.single ? "true" : "false");
    sb_lit(&r, ",\"resident\":");
    if (SV.n_reg > 0 && res >= 0) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.reg[res].name, strlen(SV.reg[res].name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");
    }
    // RI-2: the EFFECTIVE execution mode, not the requested one. Several
    // requested configurations legitimately resolve to a different one and say
    // so only on stderr, which a benchmark harness cannot read -- so it can
    // measure the fallback and publish it as the mode it asked for. `slots` is
    // what the scheduler actually runs, and `draft` separates what was asked
    // for from what is running.
    // Registry bookkeeping owns only the single-slot draft. Other slots own
    // their drafts directly, so inspect the resident engines for all sources.
    // swap_mu protects reloads; without a registry these fields stay fixed.
    const char *dsrc = NULL;
    bool mtp_consumed = false;
    for (int i = 0; i < SV.n_slots; i++) {
        const slot_t *slot = &SV.slots[i];
        if (!slot->m) continue;
        const engine *e = &slot->e;
        mtp_consumed |= e->mtp_on;
        if (!dsrc)
            dsrc = e->dm ? "model" : e->mtp_on ? "mtp"
                 : e->lookup_on ? "lookup" : NULL;
    }
    sb_fmt(&r, ",\"slots\":%d,\"draft\":{\"requested\":%s,\"active\":%s",
           SV.n_slots,
           SV.draft_requested ? "true" : "false",
           dsrc ? "true" : "false");
    if (dsrc) {
        sb_lit(&r, ",\"source\":\"");
        sb_lit(&r, dsrc);
        sb_lit(&r, "\"");
    }
    if (SV.draft_requested && !dsrc && SV.draft_note) {
        sb_lit(&r, ",\"reason\":\"");
        sb_esc(&r, SV.draft_note, strlen(SV.draft_note));
        sb_lit(&r, "\"");
    }
    sb_lit(&r, "}");
    // The reasoning channel's own serving defaults, when the operator set
    // any. Without this an eval can only learn which sampler produced a
    // trace by reading the server's source (the lab, 2026-09-26): the
    // request carries no reasoning_* field, so the trace looks greedy.
    if (SV.reasoning_temp_set || SV.reasoning_budget > 0) {
        sb_lit(&r, ",\"reasoning\":{");
        bool first = true;
        if (SV.reasoning_temp_set) {
            sb_fmt(&r, "\"temperature\":%.2f,\"inherits\":\"top_p, min_p, top_k "
                       "from the request or the model's preset\"", SV.reasoning_temp);
            first = false;
        }
        if (SV.reasoning_budget > 0)
            sb_fmt(&r, "%s\"max_tokens\":%d", first ? "" : ",", SV.reasoning_budget);
        sb_lit(&r, "}");
    }
    sb_fmt(&r, ",\"context\":%d,\"models\":[", context_load());
    if (SV.n_reg > 0) {
        for (int i = 0; i < SV.n_reg; i++) {
            if (i) sb_lit(&r, ",");
            sb_lit(&r, "{\"id\":\"");
            sb_esc(&r, SV.reg[i].name, strlen(SV.reg[i].name));
            sb_fmt(&r, "\",\"resident\":%s}", i == res ? "true" : "false");
        }
    } else {
        sb_lit(&r, "{\"id\":\"");
        sb_esc(&r, SV.model_name, strlen(SV.model_name));
        sb_lit(&r, "\",\"resident\":true}");
    }
    model_t *pm = res >= 0 ? SV.slots[0].m : NULL;
    sb_lit(&r, "],\"agent_profile\":");
    if (!pm || !pm->agent_profile) {
        sb_lit(&r, "null");
    } else {
        sb_fmt(&r, "{\"protocol_version\":%u,\"tokenizer_version\":%u,\"schema_id\":\"",
               pm->agent_protocol_version, pm->agent_tokenizer_version);
        sb_esc(&r, pm->agent_schema_id, strlen(pm->agent_schema_id));
        sb_lit(&r, "\",\"schema_digest\":\"");
        sb_esc(&r, pm->agent_schema_digest, strlen(pm->agent_schema_digest));
        sb_lit(&r, "\",\"required_features\":[");
        for (uint64_t i = 0; i < pm->n_agent_required_features; i++) {
            if (i) sb_lit(&r, ",");
            sb_lit(&r, "\"");
            sb_esc(&r, pm->agent_required_features[i].s,
                   pm->agent_required_features[i].n);
            sb_lit(&r, "\"");
        }
        sb_lit(&r, "]}");
    }
    // A declared head is only consumed when a resident engine enables it.
    sb_fmt(&r, ",\"mtp\":{\"declared_layers\":%d,\"consumed\":%s}",
           pm ? pm->mtp_layers : 0, mtp_consumed ? "true" : "false");
    sb_lit(&r, ",\"sampling\":{\"preset\":");
    if (SV.preset_name) {
        sb_lit(&r, "\"");
        sb_esc(&r, SV.preset_name, strlen(SV.preset_name));
        sb_lit(&r, "\"");
    } else {
        sb_lit(&r, "null");  // swap mode before the first model is resident
    }
    {
        const sampler *d = &SV.slots[0].smp_base;
        sb_fmt(&r, ",\"temperature\":%.2f,\"top_p\":%.2f,\"top_k\":%d,"
                   "\"min_p\":%.2f,\"repeat_penalty\":%.2f}",
               (double)d->temp, (double)d->top_p, d->top_k,
               (double)d->min_p, (double)d->repeat_penalty);
    }
    // The resident model's chat template and the tool protocol it selects
    // (the --tool-info answer), so a client can tell which contract its
    // tool declarations will be taught and parsed under before sending one.
    if (pm) {
        bool native = false;
        const char *fam = tool_protocol_name(SV.slots[0].tmpl, &native);
        sb_fmt(&r, ",\"template\":\"%s\",\"tool_protocol\":{\"family\":\"%s\","
                   "\"native\":%s}",
               template_name(SV.slots[0].tmpl), fam, native ? "true" : "false");
    }
    sb_lit(&r, ",\"features\":{"
               "\"responses_api\":true,"
               "\"messages_api\":true,"
               "\"json_object\":true,"
               "\"json_schema\":true,"
               "\"stop_sequences\":true,"
               "\"schema_conditionals\":true,"
               "\"schema_string_bounds\":true,"
               "\"schema_integer_bounds\":true,"
               // RI-4: qualified per surface. Buffered replies carry
               // the full runner_telemetry object, and since 2026-10-02
               // so does a streamed turn's finish chunk (the 2026-08-08
               // deferral reversed by the owner).
               "\"request_telemetry\":{\"buffered\":true,"
                                     "\"streamed\":true},"
               // GET /metrics, in Prometheus text exposition 0.0.4. Named as
               // a feature rather than assumed from the version, because a
               // scraper's alternative is to poll /health and translate, and
               // it needs to know which of the two it is talking to.
               "\"prometheus_metrics\":true,"
               // usage.prompt_tokens_details.cached_tokens on the OpenAI
               // surfaces, usage.input_tokens_details.cached_tokens on
               // Responses. Not on Messages: see the note in completion.c.
               "\"usage_cached_tokens\":true,"
               "\"prefix_cache\":true,"
               "\"prefix_cache_controls\":true,"
               "\"shared_prefix_cache\":true,"
               "\"forkable_prefixes\":true,"
               "\"repeat_penalty\":true,"
               "\"family_sampling_presets\":true}}");
    // The lock ends here and not one line later: send_built() is a blocking
    // socket write bounded only by SO_SNDTIMEO, and holding swap_mu across it
    // would let one slow reader stall every swap and /unload (RNS-1).
    if (guarded) pthread_mutex_unlock(&SV.swap_mu);
    send_built(fd, &r);
    free(r.s);
}

static void handle_conn(slot_t *s, sock_t fd) {
    // a stalled or dead client must not pin an inference slot: the whole
    // request (header + body) has to arrive within this budget. Generation
    // time stays unbounded — the deadline only covers reading the request.
    //
    // The write side needs its own bound. A client that sends a valid request
    // and then stops reading fills the socket buffer, and an unbounded
    // blocking write parks the slot forever: the read deadline is already
    // satisfied and never fires again, so the slot is never returned.
    sock_send_timeout(fd, 30.0);
    double deadline = now_s() + 10.0;
    char hdr[16384];
    size_t got = 0;
    char *body_start = NULL;
    while (got < sizeof(hdr) - 1) {
        double remaining = deadline - now_s();
        if (remaining <= 0) { send_error(fd, 408, "request read timed out"); return; }
        sock_recv_timeout(fd, remaining);
        int r = sock_recv(fd, hdr + got, sizeof(hdr) - 1 - got);
        if (r <= 0) {
            // r == 0: orderly close, client is gone. r < 0: timeout (or a
            // socket error, where the 408 write fails harmlessly).
            if (r < 0) send_error(fd, 408, "request read timed out");
            return;
        }
        size_t prev = got;
        got += (size_t)r;
        hdr[got] = 0;
        // A NUL cannot appear in an HTTP header, and every parse below it is
        // NUL-terminated: the terminator search here, parse_request_line,
        // and the authority and framing scans all stop at the first zero
        // byte. One embedded NUL therefore hides the real "\r\n\r\n" from all
        // of them, and this loop runs to the full 16 KB buffer or the 10 s
        // deadline before answering -- a slot held for ten seconds by a
        // request that was malformed at its first byte. Refuse it there.
        if (memchr(hdr + prev, 0, (size_t)r)) {
            send_error(fd, 400, "NUL byte in request header");
            return;
        }
        if ((body_start = strstr(hdr, "\r\n\r\n")) != NULL) break;
    }
    if (!body_start) { send_error(fd, 400, "bad request"); return; }
    char *header_end = body_start;
    body_start += 4;

    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    if (!parse_request_line(hdr, method, path, &first_header)) {
        send_error(fd, 400, "malformed request line");
        return;
    }
    if (!validate_request_authority(first_header, header_end)) {
        send_error(fd, 403, "Host and Origin must be loopback");
        return;
    }
    // Route on the path component. SDKs use query parameters for protocol
    // feature selection — Claude Code, for example, sends
    // `/v1/messages?beta=true`. The query does not rename the resource and is
    // deliberately not interpreted by this stateless inference boundary.
    char *query = strchr(path, '?');
    if (query) *query = 0;

    size_t content_length = 0;
    if (!parse_request_framing(first_header, header_end, &content_length)) {
        send_error(fd, 400, "invalid request framing");
        return;
    }
    if (content_length > 32u * 1024 * 1024) {
        send_error(fd, 400, "body too large");
        return;
    }

    char *body = NULL;
    if (content_length > 0) {
        body = malloc(content_length + 1);
        if (!body) { send_error(fd, 500, "cannot allocate request body"); return; }
        size_t have = got - (size_t)(body_start - hdr);
        if (have > content_length) have = content_length;
        memcpy(body, body_start, have);
        while (have < content_length) {
            double remaining = deadline - now_s();
            if (remaining <= 0) {
                free(body);
                send_error(fd, 408, "request read timed out");
                return;
            }
            sock_recv_timeout(fd, remaining);
            int r = sock_recv(fd, body + have, content_length - have);
            if (r <= 0) {
                free(body);
                if (r < 0) send_error(fd, 408, "request read timed out");
                return;
            }
            have += (size_t)r;
        }
        body[content_length] = 0;
    }

    bool bodyless_route = content_length > 0 &&
        ((!strcmp(method, "POST") &&
          (!strcmp(path, "/unload") ||
           !strcmp(path, "/v1/runner/prefix-cache/clear"))) ||
         (!strcmp(method, "DELETE") &&
          (!strncmp(path, "/v1/runner/contexts/",
                    sizeof("/v1/runner/contexts/") - 1) ||
           !strncmp(path, "/v1/responses/", sizeof("/v1/responses/") - 1))) ||
         (!strcmp(method, "GET") &&
          !strncmp(path, "/v1/responses/", sizeof("/v1/responses/") - 1)) ||
         (!strcmp(method, "GET") &&
          (!strcmp(path, "/health") || !strcmp(path, "/v1/models") ||
           !strcmp(path, "/v1/capabilities") || !strcmp(path, "/metrics") ||
           !strcmp(path, "/v1/runner/prefix-cache") ||
           !strcmp(path, "/v1/runner/provenance") ||
           !strcmp(path, "/v1/runner/contexts"))));
    if (bodyless_route) {
        // These routes are normally served by accept_fastpath. A declared body
        // is deliberately deferred here so the slot can consume it before
        // replying; closing from the accept thread with unread bytes resets
        // the TCP connection and can discard the error response itself.
        send_error(fd, 400, "this route takes no request body");
    } else if (!strcmp(method, "POST") && !strcmp(path, "/unload")) {
        // Free the resident model's memory. Normally answered straight from
        // the accept loop (see accept_fastpath); this path still serves a
        // request that slipped past it.
        handle_unload(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/unload")) {
        // POST-only since 0.1.5-alpha, and the refusal is explicit rather
        // than a 404 so an operator with the old call sees what changed.
        //
        // It was a GET, which made it reachable from any web page the user
        // happened to be visiting: <img src="http://127.0.0.1:PORT/unload">
        // frees the model with no preflight, no CORS, and no rebinding needed,
        // because loopback binding does not stop a browser. A POST is not a
        // CORS simple request unless its Content-Type says so, so requiring
        // one restores the preflight that stands between a drive-by page and
        // a freed model. Host/Origin validation is the other half and is
        // tracked separately.
        send_error(fd, 405, "unload is POST-only: GET was reachable from any "
                            "web page via <img src>. Use POST /unload");
    } else if (!strcmp(method, "GET") &&
               !strcmp(path, "/v1/runner/prefix-cache")) {
        send_prefix_cache(fd);
    } else if (!strcmp(method, "POST") &&
               !strcmp(path, "/v1/runner/prefix-cache/clear")) {
        // Explicit release, for benchmarks that need a cold cache and for
        // operators reclaiming the memory without unloading the model.
        prefix_cache_clear();
        send_prefix_cache(fd);
    } else if (!strcmp(method, "GET") &&
               !strcmp(path, "/v1/runner/provenance")) {
        send_provenance(fd);
    } else if (!strcmp(method, "GET") &&
               !strcmp(path, "/v1/runner/contexts")) {
        send_contexts(fd);
    } else if (!strcmp(method, "DELETE") &&
               !strncmp(path, "/v1/runner/contexts/",
                        sizeof("/v1/runner/contexts/") - 1)) {
        delete_context(fd, path);
    } else if ((!strcmp(method, "GET") || !strcmp(method, "DELETE")) &&
               !strncmp(path, "/v1/runner/sessions/",
                        sizeof("/v1/runner/sessions/") - 1)) {
        session_route(fd, method, path);
    } else if ((!strcmp(method, "GET") || !strcmp(method, "DELETE")) &&
               !strncmp(path, "/v1/responses/", sizeof("/v1/responses/") - 1)) {
        stored_response_route(fd, method, path);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        send_health(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
        send_models(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/metrics")) {
        send_metrics(fd);
    } else if (!strcmp(method, "GET") && !strcmp(path, "/v1/capabilities")) {
        send_capabilities(fd);
    } else if (!strcmp(method, "POST") &&
               (!strcmp(path, "/v1/chat/completions") ||
                !strcmp(path, "/v1/responses") ||
                !strcmp(path, "/v1/messages") ||
                !strcmp(path, "/v1/messages/count_tokens") ||
                !strcmp(path, "/v1/completions") ||
                !strcmp(path, "/v1/embeddings") ||
                !strcmp(path, "/v1/decide") ||
                !strcmp(path, "/v1/rerank") ||
                !strcmp(path, "/v1/runner/contexts") ||
                (!strncmp(path, "/v1/runner/contexts/",
                          sizeof("/v1/runner/contexts/") - 1) &&
                 strstr(path, "/snapshot")) ||
                !strcmp(path, "/v1/runner/sessions") ||
                (!strncmp(path, "/v1/runner/sessions/",
                          sizeof("/v1/runner/sessions/") - 1) &&
                 strstr(path, "/resume")))) {
        jv *req = body ? json_parse(body, content_length) : NULL;
        if (!req) {
            send_error(fd, 400, "invalid JSON body");
        } else {
            jv *model = jv_get(req, "model");
            // R8.6: "<model>:<adapter>" names a per-request adapter; the base
            // is then validated like any request's model
            int adapter = request_adapter(req);
            if (model && model->type != J_NULL && model->type != J_STR) {
                send_error_detail(fd, 400, "model must be a string", "model",
                                  "invalid_type");
                jv_free(req);
                free(body);
                return;
            }
            bool has_keep_alive = false;
            int keep_alive = 0;
            if (!request_keep_alive(req, &has_keep_alive, &keep_alive)) {
                send_error(fd, 400, "keep_alive out of range");
                jv_free(req);
                free(body);
                return;
            }
            // A server with no registry (SV.n_reg == 0) is exactly one
            // configuration: a single model served with --parallel N>1, whose
            // slots hold the model directly (server.c only joins the registry
            // when parallel == 1). keep_alive drives the swap-mode idle/unload
            // machinery, which does not exist there -- so the field used to be
            // range-checked, accepted, and then silently dropped below (the
            // `SV.n_reg > 0` guard). A client sending keep_alive:0 ("unload
            // after this request") was answered as if it happened while the
            // model stayed resident, the same dishonesty POST /unload was
            // taught to refuse in this configuration. 400, not 409 as /unload
            // uses: /unload is a management route whose target-resource state
            // refuses it, this is a per-request field on a completion that is
            // well-formed but not satisfiable here -- the shape of the
            // surface's other request-field rejections (timeout out of range,
            // a field with unsupported semantics). A request with NO keep_alive
            // is the normal case and is untouched.
            if (has_keep_alive && SV.n_reg == 0) {
                char msg[384];
                snprintf(msg, sizeof(msg),
                         "keep_alive cannot be honored: with --parallel %d and "
                         "a single model the slots hold the model directly and "
                         "never join the registry, so there is no idle-unload "
                         "to schedule against and keep_alive:0 would free "
                         "nothing. Serve with --parallel 1 for keep_alive "
                         "support.",
                         SV.n_slots);
                send_error_detail(fd, 400, msg, "keep_alive",
                                  "keep_alive_unsupported");
                jv_free(req);
                free(body);
                return;
            }
            resident_expect expect;
            if (!request_expect_resident(fd, req, &expect)) {
                jv_free(req);
                free(body);
                return;
            }
            atomic_fetch_add(&SV.active_requests, 1);
            atomic_fetch_add(&SV.total_requests, 1);
            bool ok = true;
            int sw = SV.n_reg > 0
                ? swap_to_expect(jv_str(jv_get(req, "model"), NULL), &expect)
                : single_model_expect(&expect);
            if (sw == SWAP_EXPECT_MISMATCH || sw == SWAP_EXPECT_UNKNOWN) {
                // R10.12.2: say what IS resident, so the client can decide
                // without a second round trip
                bool res = false;
                unsigned long long g = provenance_load_generation(&res);
                char msg[256];
                snprintf(msg, sizeof msg,
                         sw == SWAP_EXPECT_UNKNOWN
                         ? "expect_resident could not be checked: the resident "
                           "model's digest is not known yet or its file changed "
                           "on disk (load_generation %llu); nothing was loaded"
                         : "expect_resident does not match what is resident "
                           "(load_generation %llu, %s); nothing was loaded",
                         g, res ? "a model is resident" : "no model is resident");
                send_error_detail(fd, 409, msg, "expect_resident",
                                  sw == SWAP_EXPECT_UNKNOWN
                                  ? "resident_identity_unknown"
                                  : "resident_mismatch");
                ok = false;
            } else if (SV.n_reg > 0) {
                if (sw == SWAP_LOAD_FAILED) {
                    send_error(fd, 500,
                               "model failed to load (registered but broken; see server log)");
                    ok = false;
                } else if (sw == SWAP_ENVELOPE_REFUSED) {
                    send_error(fd, 409,
                               "model refused: outside its measured envelope for "
                               "this runtime (see server log); start the server "
                               "with --force-uncertified to override");
                    ok = false;
                } else if (sw == SWAP_SIGNATURE_REFUSED) {
                    send_error_detail(fd, 409,
                                      "model refused by signature policy (see server log)",
                                      "model", "model_signature_refused");
                    ok = false;
                } else if (sw == SWAP_ABORTED) {
                    send_error(fd, 503,
                               "model load abandoned (unload or shutdown requested; retry)");
                    ok = false;
                } else if (sw < 0) {
                    send_error_detail(fd, 404,
                                      "unknown model (see /v1/models)",
                                      "model", "model_not_found");
                    ok = false;
                }
            } else if (!validate_single_model_request(fd, req)) {
                ok = false;
            }
            if (ok) ok = slot_use_adapter(s, adapter, fd);
            if (ok) {
                if (strcmp(path, "/v1/chat/completions") == 0) handle_chat(s, fd, req);
                else if (strcmp(path, "/v1/responses") == 0) handle_responses(s, fd, req);
                else if (strcmp(path, "/v1/messages") == 0) handle_messages(s, fd, req);
                else if (strcmp(path, "/v1/messages/count_tokens") == 0)
                    handle_count_tokens(s, fd, req);
                else if (strcmp(path, "/v1/embeddings") == 0) handle_embeddings(s, fd, req);
                else if (strcmp(path, "/v1/decide") == 0) handle_decide(s, fd, req);
                else if (strcmp(path, "/v1/rerank") == 0) handle_rerank(s, fd, req);
                else if (strcmp(path, "/v1/runner/contexts") == 0)
                    handle_context_create(s, fd, req);
                else if (!strncmp(path, "/v1/runner/contexts/",
                                  sizeof("/v1/runner/contexts/") - 1))
                    handle_context_snapshot(s, fd, req, path);
                else if (!strncmp(path, "/v1/runner/sessions",
                                   sizeof("/v1/runner/sessions") - 1)) {
                    if (!sessions_dir())
                        send_error_detail(fd, 404, "sessions are off: start "
                                          "the server with --sessions DIR",
                                          NULL, "sessions_off");
                    else if (!strcmp(path, "/v1/runner/sessions"))
                        handle_session_start(s, fd, req);
                    else
                        handle_session_resume(s, fd, req, path);
                }
                else handle_completion(s, fd, req);
                // Ollama-style keep_alive: seconds of idle before the model
                // unloads (swap mode) — 0 unloads now, negative pins forever.
                //
                // 0 records the wish rather than acting on it, so it is
                // honoured by the ONE piece of code that knows what "unload
                // now" costs -- the safe-point block below, which is also what
                // POST /unload defers to. Calling unload_resident() here was a
                // second, shorter copy of that: it freed the model and left the
                // draft loaded and every KV prefix snapshot resident (up to
                // RUNNER_PREFIX_CACHE_MB, 512 MB by default), so the same
                // operator got different amounts of memory back depending on
                // which spelling they used. It also ran with this request still
                // counted active, which is the state the safe point exists to
                // wait out.
                if (has_keep_alive && SV.n_reg > 0) {
                    pthread_mutex_lock(&SV.swap_mu);
                    if (keep_alive == 0) SV.pending_unload = true;
                    else SV.ttl = keep_alive < 0 ? 0 : keep_alive;
                    pthread_mutex_unlock(&SV.swap_mu);
                }
            }
            // whatever way the handler left, this slot is no longer working
            atomic_store(&s->rq_phase, 0);
            s->e.stat_feed = s->e.stat_gen = NULL;
            // Drop the request from the count BEFORE the bookkeeping lock:
            // an /unload that saw this request active left pending_unload for
            // us, and the count must already be zero when we honour it.
            atomic_fetch_sub(&SV.active_requests, 1);
            if (SV.n_reg > 0) {
                bool unloaded = false;
                pthread_mutex_lock(&SV.swap_mu);
                SV.last_used = now_s();
                if (SV.pending_unload && !SV.loading &&
                    !atomic_load(&SV.active_requests)) {
                    unload_draft();
                    unload_resident();
                    SV.pending_unload = false;
                    unloaded = true;
                }
                pthread_mutex_unlock(&SV.swap_mu);
                if (unloaded) prefix_cache_clear();
            }
            jv_free(req);
        }
    } else {
        send_error(fd, 404, "not found");
    }
    free(body);
}

static void *slot_worker(void *arg) {
    slot_t *s = arg;
    for (;;) {
        double waited = 0;
        sock_t fd = q_pop(&waited);
        if (fd == SOCK_INVALID) return NULL;
        s->queue_wait_s = waited;
        s->req_t0 = now_s();
        handle_conn(s, fd);
        sock_close(fd);
    }
}

// answer tiny requests from the accept loop: single-slot serving means one
// long generation used to block /health until the xyntetik watchdog declared a
// live runner "unhealthy: timed out". POST /unload is answered here too — it
// never frees anything a slot is using (handle_unload defers under an active
// load or generation), and an operator reclaiming memory must not queue behind
// the very work that holds it. Prefix-cache telemetry and release have their
// own mutex and likewise need no inference slot. Every other request is handed
// to a slot untouched.
//
// Two rules keep this thread admitting connections (found 2026-10-05: a
// stored-response GET to a client that stopped reading parked the accept loop
// and nothing new was served):
//  1. Only answers that fit a socket buffer are written from here. A stored
//     Responses body or its input_items can be megabytes, /metrics and the
//     context listing grow with the server; those take a slot, whose writes
//     are bounded like every other slot write.
//  2. Every write from here carries the slot path's 30 s send timeout, so a
//     reader that stops costs this thread at most 30 s, never forever.
static bool accept_fastpath(sock_t fd) {
#ifndef _WIN32
    // POSIX fd_set is a fixed-size bitmask indexed by fd value; FD_SET on an
    // fd >= FD_SETSIZE is undefined behavior (out-of-bounds write). Windows
    // fd_set is a count-based array instead, so it isn't affected.
    if (fd >= FD_SETSIZE) return false;
#endif
    fd_set rs;
    struct timeval tv = { 0, 250000 }; // loopback data lands in <1ms
    FD_ZERO(&rs);
    FD_SET(fd, &rs);
    if (select(fd + 1, &rs, NULL, NULL, &tv) != 1) return false;
    char hdr[2048];
    int n = sock_peek(fd, hdr, sizeof(hdr) - 1);
    if (n <= 0) { sock_close(fd); return true; } // died before speaking
    hdr[n] = 0;
    // match the path plus the space HTTP/1.x always puts before the version,
    // so "GET /healthzzz" falls through to the slot path instead of being
    // misrouted here. Bare HTTP/0.9 "GET /health\r\n" (no version) won't
    // match, but neither curl nor the xyntetik watchdog send that, so it's
    // not worth the extra branch.
    bool health = !strncmp(hdr, "GET /health ", 12);
    bool models = !strncmp(hdr, "GET /v1/models ", 15);
    bool caps = !strncmp(hdr, "GET /v1/capabilities ", 21);
    bool unload = !strncmp(hdr, "POST /unload ", 13);
    bool pfx_stats = !strncmp(hdr, "GET /v1/runner/prefix-cache ",
                              sizeof("GET /v1/runner/prefix-cache ") - 1);
    bool pfx_clear = !strncmp(hdr, "POST /v1/runner/prefix-cache/clear ",
                              sizeof("POST /v1/runner/prefix-cache/clear ") - 1);
    bool prov = !strncmp(hdr, "GET /v1/runner/provenance ",
                         sizeof("GET /v1/runner/provenance ") - 1);
    // Deliberately NOT here: GET /metrics, GET /v1/runner/contexts and the
    // stored-response routes (rule 1 above); the slot path serves them.
    // The old spelling still has to reach a handler, or an operator's script
    // gets a 404 that says nothing. It is not answered here — it falls through
    // to the slot path, which replies 405 with the reason.
    if (!strncmp(hdr, "GET /unload ", 12)) return false;
    if (!health && !models && !caps && !unload && !pfx_stats && !pfx_clear &&
        !prov)
        return false;
    // Keep the request untouched until framing says it is bodyless. A partial
    // header, an oversized header, malformed framing, and every declared body
    // are handed to a slot, whose bounded reader can consume the whole request
    // before replying. The accept thread therefore never waits for a client to
    // finish speaking, and never closes a body-bearing connection with bytes
    // unread (which would turn the close into RST and lose the response).
    char *header_end = strstr(hdr, "\r\n\r\n");
    if (!header_end) return false;
    char method[8] = {0}, path[256] = {0};
    char *first_header = NULL;
    size_t content_length = 0;
    if (!parse_request_line(hdr, method, path, &first_header) ||
        !parse_request_framing(first_header, header_end, &content_length) ||
        content_length > 32u * 1024 * 1024 ||
        !validate_request_authority(first_header, header_end) ||
        content_length > 0)
        return false;

    // MSG_PEEK proved the complete header is already buffered. Consume exactly
    // those bytes, never a following body or pipelined request, before closing.
    size_t header_n = (size_t)(header_end + 4 - hdr);
    size_t got = 0;
    while (got < header_n) {
        int r = sock_recv(fd, hdr + got, header_n - got);
        if (r <= 0) { sock_close(fd); return true; }
        got += (size_t)r;
    }
    sock_send_timeout(fd, 30.0);   // rule 2: a dead reader cannot hold this thread
    (void)method; (void)path;
    if (health)          send_health(fd);
    else if (models)     send_models(fd);
    else if (prov)       send_provenance(fd);
    else if (unload)     handle_unload(fd);
    else if (pfx_stats)  send_prefix_cache(fd);
    else if (pfx_clear) {
        prefix_cache_clear();
        send_prefix_cache(fd);
    } else {
        send_capabilities(fd);
    }
    sock_close(fd);
    return true;
}

// ---------------------------------------------------------------- entry

#ifndef _WIN32
static volatile sig_atomic_t stop_requested;
static volatile sig_atomic_t listener_fd = -1;

void server_request_stop(void) {
    // A second signal is the operator overruling the drain: exit now. _exit is
    // async-signal-safe (128+SIGINT — the shell's convention for a Ctrl-C kill),
    // where the alternative on a pinned drain was reaching for SIGKILL.
    if (stop_requested) _exit(130);
    stop_requested = 1;
    int fd = (int)listener_fd;
    if (fd >= 0) {
        listener_fd = -1;
        // shutdown() BEFORE close(), and it is not belt-and-braces.
        //
        // A blocked accept() is woken by the signal only in the thread the
        // signal was delivered to, and a process may deliver SIGTERM to any
        // thread that has it unblocked — a slot worker, the decode thread, the
        // TTL reaper. Closing the descriptor from one of those does NOT wake a
        // thread already parked in accept() on Linux; it stays there until a
        // connection happens to arrive. Observed directly: with the accept loop
        // on a non-main thread, /proc showed it in inet_csk_accept long after
        // the handler had run and closed the fd.
        //
        // shutdown() does wake it, and both calls are async-signal-safe.
        shutdown(fd, SHUT_RDWR);
        close(fd);
    }
}

static void stop_handler(int sig) {
    (void)sig;
    server_request_stop();
}

static void install_stop_handlers(void) {
    struct sigaction sa = {0};
    sa.sa_handler = stop_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}

// Between the handlers installing and the listener publishing there is nothing
// a signal can close, so startup polls this at its long stops (model loads)
// and abandons the launch instead of serving a request nobody wants anymore.
static bool stop_was_requested(void) { return stop_requested != 0; }
#else
// Windows port of the same design: stop flag + listener close + drain +
// second-signal escalation, via SetConsoleCtrlHandler. The handler runs on
// its own thread (not an async-signal context), so plain volatile stores and
// closesocket() are safe here.
static volatile LONG win_stop_requested;
static volatile SOCKET win_listener_socket = INVALID_SOCKET;

void server_request_stop(void) {
    if (InterlockedExchange(&win_stop_requested, 1)) _exit(130);
    SOCKET s = win_listener_socket;
    if (s != INVALID_SOCKET) {
        win_listener_socket = INVALID_SOCKET;
        closesocket(s); // wakes accept()
    }
}

static BOOL WINAPI win_stop_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT: {
        // A second Ctrl-C is the operator overruling the drain: exit now,
        // with the shell's 128+SIGINT convention for parity with POSIX.
        server_request_stop();
        if (ctrl_type == CTRL_CLOSE_EVENT || ctrl_type == CTRL_SHUTDOWN_EVENT) {
            // Returning from these lets Windows terminate the process
            // immediately; hold the handler thread briefly so the drain in
            // main gets its window (Windows grants ~5s on close).
            Sleep(4000);
        }
        return TRUE; // handled: keep running so the drain can finish
    }
    default:
        return FALSE;
    }
}

static void install_stop_handlers(void) {
    win_stop_requested = 0;
    win_listener_socket = INVALID_SOCKET;
    SetConsoleCtrlHandler(win_stop_handler, TRUE);
}

static bool stop_was_requested(void) { return win_stop_requested != 0; }
#endif

static struct {
    const char *const *names, *const *paths;
    int   n;
    float scale;
} ADAPTER_CFG;

static bool WM_CFG_ON;
static wm_key WM_CFG;

void server_set_watermark(const wm_key *key) {
    memcpy(&WM_CFG, key, sizeof WM_CFG);
    WM_CFG_ON = true;
}

void server_set_adapters(const char *const *names, const char *const *paths,
                         int n, float scale) {
    ADAPTER_CFG.names = names;
    ADAPTER_CFG.paths = paths;
    ADAPTER_CFG.n = n;
    ADAPTER_CFG.scale = scale;
}

// Load the configured adapters against the preloaded model (R8.6). They run on
// the host hooks, so a slot with a device path refuses them; a swap set
// refuses them too, since the geometry they were parsed against would change
// under them.
static bool load_adapters(const model_t *base) {
    if (ADAPTER_CFG.n == 0) return true;
    if (!base) {
        fprintf(stderr, "error: --adapter needs one model served by path "
                "(-m model.gguf), not a name=path registry\n");
        return false;
    }
    for (int i = 0; i < SV.n_slots; i++)
        if (SV.slots[i].m && SV.slots[i].m->gpu) {
            fprintf(stderr, "error: --adapter runs on the CPU hooks; slot %d "
                    "has a device path (serve with --gpu off)\n", i);
            return false;
        }
    if (ADAPTER_CFG.n > RUNNER_MAX_ADAPTERS) {
        fprintf(stderr, "error: at most %d --adapter\n", RUNNER_MAX_ADAPTERS);
        return false;
    }
    for (int i = 0; i < ADAPTER_CFG.n; i++) {
        adapter_entry *a = &SV.adapters[i];
        snprintf(a->name, sizeof a->name, "%s", ADAPTER_CFG.names[i]);
        snprintf(a->path, sizeof a->path, "%s", ADAPTER_CFG.paths[i]);
        for (int j = 0; j < i; j++)
            if (!strcmp(SV.adapters[j].name, a->name)) {
                fprintf(stderr, "error: duplicate --adapter name '%s'\n", a->name);
                return false;
            }
        if (!envelope_file_sha256(a->path, a->sha256)) {
            fprintf(stderr, "error: cannot read adapter %s\n", a->path);
            return false;
        }
        // R1.2.3: each adapter answers to the operator's trusted key like
        // the model does, from its own PATH.sig
        oms_policy apol = { NULL, SV.signing.pubkey_path, SV.signing.required };
        oms_result ar;
        memset(&ar, 0, sizeof ar);
        if (!oms_check_artifact(a->path, &apol, "adapter", &ar)) return false;
        a->sig_json[0] = 0;
        if (ar.status[0]) oms_result_json(&ar, a->sig_json, sizeof a->sig_json);
        a->set = model_lora_set_load(SV.slots[0].m, a->path, ADAPTER_CFG.scale);
        if (!a->set) return false;
        SV.n_adapters = i + 1;
    }
    return true;
}

int server_run(model_t *base, tokenizer *tok, const char *model_path,
               const model_params *mp, sampler defaults,
               const sampler_override *ov, int port, int parallel,
               int n_threads, int ttl, const char *draft_path, int draft_k,
               bool draft_lookup, int reasoning_budget,
               const char *reasoning_budget_message, float reasoning_temp,
               bool loop_guard, bool ignore_eos,
               int tmpl_override,
               bool force_uncertified, const oms_policy *signing) {
    sock_init();
    // The shared server state gets a lifetime, and it is this call. Everything
    // below sets fields on SV and the teardown at the bottom releases them, but
    // until now nothing reset the ones that are only ever *set* — and there are
    // several: `q.shutdown` and `shutdown` are raised during teardown and never
    // lowered, `load_cancel` is left at 1, and `reaper_started` stays true
    // alongside a `reaper_th` whose thread has already been joined.
    //
    // A second server_run therefore did not work. Its slot workers saw a
    // shut-down queue and exited immediately, so the server listened, accepted
    // connections and answered nothing but `/health` — which is served on the
    // accept path and needs no worker, and is exactly why a casual check
    // looked fine. Teardown then joined an already-joined thread handle.
    //
    // This is what the RNR-019 finding meant by global state making
    // initialization and teardown hard: nothing ever asked the state to come
    // back, so the asymmetry could not be observed. tests/test_server_restart.c
    // now asks, twice. Note this makes the state's lifetime explicit, NOT
    // per-instance — two servers in one process would still share it. That
    // half is filed; nothing needs it today, and it costs threading a context
    // through six translation units.
    memset(&SV, 0, sizeof(SV));
#ifndef _WIN32
    stop_requested = 0;
    listener_fd = -1;
#endif
    install_stop_handlers(); // resets the stop flag + listener on both platforms
    provenance_init();       // the executable's digest, once, before serving
    respstore_reset_from_env();   // an in-memory store lives with the server
    SV.ignore_eos = ignore_eos;
    // Measured-envelope enforcement for swapped-in models. registry.c resolves
    // the backend from each loaded model, since an available GPU may be unused
    // after --gpu off or a model-specific fallback.
    SV.force_uncertified = force_uncertified;
    if (signing) SV.signing = *signing;
    // Set before anything can load a model: registry.c reads it on every swap
    // and reload, so a forced template survives /unload and --ttl instead of
    // being detected away at the next request.
    SV.tmpl_override = tmpl_override;
    if (ov) SV.ov = *ov;
    // `defaults` arrives already resolved against the preloaded model; in swap
    // mode there is no model yet and swap_to resolves per load
    if (base) {
        char ident[256];
        sampler_ident(gguf_get_str(&base->gf, "general.name", NULL),
                      base->path, ident, sizeof(ident));
        // The template beats the name when they disagree (see sample.h).
        // tok is not needed: its probes are the fallback for a model with no
        // template text, and a model with no template text has only its name
        // to be identified by anyway.
        int preset_tmpl = tmpl_override >= 0 ? tmpl_override
                        : template_detect(gguf_get_str(&base->gf,
                                          "tokenizer.chat_template", NULL), NULL);
        SV.preset_name = sampler_preset_for(base->arch, ident, preset_tmpl)->name;
    } else {
        SV.preset_name = NULL;
    }
    if (parallel < 1) parallel = 1;
    if (parallel > 16) parallel = 16;
    // One named model is not a swap set, and asking for a name should not
    // silently cost the caller their slots.
    //
    // `-m llama=path` is the documented way to pin the /v1/models id for a
    // client config, and a Continue user did exactly that and lost
    // --parallel 2 without ever asking for swapping. Registry mode genuinely
    // is single-slot — ensure_resident() loads into slots[0] alone — but with
    // one entry there is nothing to swap *to*, so the honest trade is to give
    // up the registry rather than the slots. Plain `-m path --parallel 2`
    // already forgoes the registry for the same reason, so this lands in an
    // existing configuration rather than a new one.
    //
    // Done HERE, ahead of swap_mode, so everything downstream that keys off it
    // — the --ttl default, the --draft refusal, the registry setup — sees the
    // truth rather than being unwound afterwards.
    char single_path[sizeof(SV.reg[0].path)];
    const char *forced_name = NULL;
    const char *eq1 = strchr(model_path, '=');
    if (eq1 && parallel > 1 && !strchr(model_path, ',') &&
        eq1 != model_path && eq1[1] &&
        (size_t)(eq1 - model_path) < sizeof(SV.reg[0].name) &&
        strlen(eq1 + 1) < sizeof(single_path)) {
        static char single_name[sizeof(SV.reg[0].name)];
        snprintf(single_name, sizeof(single_name), "%.*s",
                 (int)(eq1 - model_path), model_path);
        snprintf(single_path, sizeof(single_path), "%s", eq1 + 1);
        forced_name = single_name;
        model_path  = single_path;
        fprintf(stderr,
                "note: one model named '%s' with --parallel %d — serving it on"
                " %d slots. /unload and --ttl are swap-mode features and need"
                " more than one model.\n",
                forced_name, parallel, parallel);
    }

    bool swap_mode = strchr(model_path, '=') != NULL;
    if (ttl < 0) ttl = swap_mode ? 300 : 0; // single-model default: never unload
    SV.draft_requested = draft_path != NULL || mp->mtp || draft_lookup;
    SV.draft_source = NULL;
    if (draft_lookup && swap_mode) {
        fprintf(stderr, "note: --draft-lookup needs a single served model; "
                "ignoring it in swap mode\n");
        SV.draft_note = "the prompt lookup needs a single served model; "
                        "ignored in swap mode";
        draft_lookup = false;
    }
    // RI-2 slice 2 (owner 2026-10-02): this one is a contradiction the
    // command line itself states, so it refuses at startup instead of
    // serving without the draft. The other fallbacks stay notes: a draft
    // refused at load (vocabulary, offload, memory) is a fact about the
    // environment, and `-m name=path --parallel N` keeping its slots is a
    // trade users rely on.
    if (draft_path && swap_mode) {
        fprintf(stderr, "error: --draft needs a single served model, and -m "
                "names a swap registry; serve one model with --draft, or "
                "drop --draft\n");
        return 1;
    }

    // "name=path,name2=path2" enables swap mode: one resident model,
    // loaded per request's "model" field, unloaded after ttl idle seconds
    if (strchr(model_path, '=')) {
        // Validate every limit and reject the whole spec with an exact reason
        // rather than silently truncating a name/path (which can collide or
        // select the wrong file) or dropping entries past the cap (RNR-014).
        const int max_reg = (int)(sizeof(SV.reg) / sizeof(SV.reg[0]));
        char tmp[4096];
        size_t spec_len = strlen(model_path);
        if (spec_len >= sizeof(tmp)) {
            fprintf(stderr, "error: -m registry spec is too long (max %zu bytes)\n",
                    sizeof(tmp) - 1);
            return 1;
        }
        // strtok() skips empty fields. Reject them before tokenizing so a
        // malformed registry cannot silently turn into a different one.
        if (model_path[0] == ',' || model_path[spec_len - 1] == ',' ||
            strstr(model_path, ",,")) {
            fprintf(stderr, "error: -m registry spec contains an empty entry\n");
            return 1;
        }
        snprintf(tmp, sizeof(tmp), "%s", model_path);
        for (char *tk = strtok(tmp, ","); tk; tk = strtok(NULL, ",")) {
            if (SV.n_reg >= max_reg) {
                fprintf(stderr, "error: too many models in -m (max %d)\n", max_reg);
                return 1;
            }
            char *eq = strchr(tk, '=');
            if (!eq) { fprintf(stderr, "error: bad registry entry '%s' (want name=path)\n", tk); return 1; }
            *eq = 0;
            const char *nm = tk, *pth = eq + 1;
            if (!*nm || !*pth) {
                fprintf(stderr, "error: registry entry '%s=%s' has an empty name or path\n", nm, pth);
                return 1;
            }
            if (strlen(nm) >= sizeof(SV.reg[0].name)) {
                fprintf(stderr, "error: model name '%s' is too long (max %zu chars)\n",
                        nm, sizeof(SV.reg[0].name) - 1);
                return 1;
            }
            if (strlen(pth) >= sizeof(SV.reg[0].path)) {
                fprintf(stderr, "error: model path for '%s' is too long (max %zu chars)\n",
                        nm, sizeof(SV.reg[0].path) - 1);
                return 1;
            }
            for (int j = 0; j < SV.n_reg; j++)
                if (!strcmp(SV.reg[j].name, nm)) {
                    fprintf(stderr, "error: duplicate model name '%s' in -m\n", nm);
                    return 1;
                }
            snprintf(SV.reg[SV.n_reg].name, sizeof(SV.reg[0].name), "%s", nm);
            snprintf(SV.reg[SV.n_reg].path, sizeof(SV.reg[0].path), "%s", pth);
            if (!plat_file_readable(SV.reg[SV.n_reg].path)) {
                fprintf(stderr, "error: cannot read %s\n", SV.reg[SV.n_reg].path);
                return 1;
            }
            SV.n_reg++;
        }
        // A real swap set is several models, each detecting its own template.
        // --chat-template names ONE and has no per-model spelling, so applying
        // it across the set would be right for at most one member and would
        // silently mis-render the rest. Refused rather than applied to all or
        // dropped for all: the caller asked for something this shape of
        // invocation cannot mean. Checked HERE and not in main.c because this
        // is where the entry count is actually known -- `-m name=path` on its
        // own is a one-entry registry (it pins the /v1/models id) and takes
        // the override perfectly well.
        if (SV.n_reg > 1 && tmpl_override >= 0) {
            fprintf(stderr,
                    "error: --chat-template cannot be used with a swap set "
                    "(-m \"name=path,name2=path2\"): it names one template and "
                    "the set holds %d models, each detecting its own. Serve "
                    "that model on its own instance to force its template.\n",
                    SV.n_reg);
            return 1;
        }
        if (parallel > 1) {
            fprintf(stderr, "note: model swapping uses a single inference slot; "
                    "ignoring --parallel %d\n", parallel);
        }
        parallel = SV.n_reg > 0 ? 1 : parallel;
        resident_store(-1);
    }

    int threads_per_slot = n_threads / parallel;
    if (threads_per_slot < 1) threads_per_slot = 1;
    int shared_pool_threads = n_threads;   // the slots' shared pool (see below), for the banner

    const char *name = strrchr(model_path, '/');
    const char *bsname = strrchr(model_path, '\\'); // Windows path separator
    if (bsname && (!name || bsname > name)) name = bsname;
    SV.model_name = SV.n_reg > 0 ? SV.reg[0].name
                  : forced_name  ? forced_name
                                 : (name ? name + 1 : model_path);
    SV.n_predict_cap = 1024;
    SV.reasoning_budget = reasoning_budget;
    SV.reasoning_budget_message = reasoning_budget_message;
    SV.loop_guard = loop_guard;
    SV.wm_on = WM_CFG_ON;
    if (WM_CFG_ON) {
        SV.wm_key = WM_CFG;
        fprintf(stderr, "watermark: sampled output is marked (%s, key id %s)\n",
                WM_SCHEME, SV.wm_key.id);
    }
    SV.reasoning_temp_set = reasoning_temp >= 0.0f;
    SV.reasoning_temp = reasoning_temp >= 0.0f ? reasoning_temp : 0.0f;
    SV.q.limit = (int)(sizeof(SV.q.fds) / sizeof(sock_t));
    // a queue bound may only lower the fixed fd capacity, never raise it
    SV.q.limit = (int)env_i64("RUNNER_MAX_QUEUE", 1, SV.q.limit, SV.q.limit);
    // a bad timeout must not silently become 0 (which disables the deadline)
    SV.req_timeout = env_f64("RUNNER_REQUEST_TIMEOUT", 0.0, 1e9, SV.req_timeout);
    // Shared, forkable prompt prefixes. Sized in host RAM rather than as a
    // fraction of anything, because it is the one cache whose useful size is
    // set by the *traffic* (how many distinct system/tool/schema blocks the
    // agents on this box use) and not by the model.
    {
        uint64_t mb  = env_u64("RUNNER_PREFIX_CACHE_MB", 0, 1u << 20, 512);
        double   ttl = env_f64("RUNNER_PREFIX_CACHE_TTL", 0.0, 1e9, 600.0);
        prefix_cache_configure((size_t)mb * 1024 * 1024, ttl);
    }
    context_store(base ? base->n_ctx : mp->n_ctx);
    SV.n_slots = parallel;
    SV.slots = calloc(parallel, sizeof(slot_t));
    if (!SV.slots) {
        fprintf(stderr, "error: cannot allocate server slots\n");
        return 1;
    }
    if (pthread_mutex_init(&SV.q.mu, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue mutex\n");
        return 1;
    }
    if (pthread_cond_init(&SV.q.cv, NULL) != 0) {
        fprintf(stderr, "error: cannot initialize server queue condition\n");
        pthread_mutex_destroy(&SV.q.mu);
        return 1;
    }

    if (SV.n_reg > 0) {
        // swap mode: models are loaded on demand
        slot_t *s = &SV.slots[0];
        s->id = 0;
        s->smp = defaults;
        s->smp_base = defaults;
        if (!init_swap_runtime(mp, n_threads, ttl)) return 1;
    } else {
        int tmpl = SV.tmpl_override >= 0
                 ? SV.tmpl_override
                 : template_detect(gguf_get_str(&base->gf,
                                                "tokenizer.chat_template", NULL),
                                   tok);
        // Announced, because a forced template changes every prompt this
        // server renders and the operator asked for it explicitly. Detection
        // stays quiet: it is the default and nobody chose it.
        if (SV.tmpl_override >= 0)
            fprintf(stderr, "chat template: %s (forced by --chat-template)\n",
                    template_name(tmpl));
        model_params slot_mp = *mp;
        slot_mp.verbose = false;
        // every slot shares ONE pool of n_threads (tpool_run serializes, so
        // forwards interleave a matvec at a time); a slot's own load-time
        // pool is a single thread and is replaced right after the load
        slot_mp.n_threads = 1;
        // a reservation is a budget for the server, and every slot pays its
        // own KV cache out of it -- see the auto-fit in model_alloc_runtime
        slot_mp.n_seq = parallel;
        // each slot gets its share of the CPU-forced fallback cap too
        slot_mp.cpu_fallback_threads = 0;   // the shared pool below is sized once, from the server's count
        tpool *shared_pool = NULL;

        for (int i = 0; i < parallel; i++) {
            // a Ctrl-C during a multi-slot load means "don't start": honour it
            // between loads rather than serving with the flag silently set
            if (stop_was_requested()) {
                fprintf(stderr, "shutdown requested during startup — exiting\n");
                return 0;
            }
            slot_t *s = &SV.slots[i];
            s->id = i;
            s->tok = tok;
            s->tmpl = tmpl;
            s->smp = defaults;
            s->smp.rng = defaults.rng ^ (0x9E3779B97F4A7C15ull * (unsigned)(i + 1));
            s->smp_base = s->smp;
            if (i == 0) {
                s->m = base;
                // mirror model_load's CPU-forced bump: the shared pool would
                // otherwise silently undo it
                int pool_threads = tpool_shared_threads(
                    n_threads, parallel, plat_cpu_count(),
                    base->qwen35 ? mp->cpu_fallback_threads : 0);
                shared_pool_threads = pool_threads;
                shared_pool = tpool_create(pool_threads);
                if (!shared_pool) {
                    fprintf(stderr, "error: cannot create the slots' thread pool\n");
                    return 1;
                }
                tpool_destroy(base->tp); // replace the single-thread load pool
                base->tp = shared_pool;
            } else {
                s->m = calloc(1, sizeof(model_t));
                if (!s->m) {
                    fprintf(stderr, "error: cannot allocate slot %d model\n", i);
                    return 1;
                }
                if (!model_load(s->m, model_path, &slot_mp) ||
                    !oms_check_model(model_path, &SV.signing, NULL)) {
                    fprintf(stderr, "error: failed to load slot %d\n", i);
                    return 1;
                }
                tpool_destroy(s->m->tp);    // the single-thread load pool
                tpool_retain(shared_pool);
                s->m->tp = shared_pool;
            }
            template_bind_think_tags(tmpl, &s->m->think_open, &s->m->think_close);
            if (!engine_init(&s->e, s->m, s->tok, &s->smp)) {
                fprintf(stderr, "error: out of memory initializing slot %d engine\n", i);
                return 1;
            }
            // CLI generation applies this after engine_init; server slots are
            // separate engines and must receive the same process-level flag.
            s->e.ignore_eos = ignore_eos;
            if (draft_path) {
                // per-slot draft context: each slot owns a full draft KV;
                // weights dedupe through the page cache like slot models
                s->e.dm = spec_draft_load(draft_path, s->m, &slot_mp);
                if (s->e.dm) s->e.draft_k = draft_k;
            }
            if (mp->mtp) {
                if (!model_mtp_ready(s->m)) {
                    fprintf(stderr, "error: --mtp: slot %d needs the CPU "
                                    "path (rerun with --gpu off)\n", i);
                    return 1;
                }
                s->e.mtp_on = true;
                s->e.draft_k = draft_k;
            }
            if (draft_lookup) {
                if (!model_spec_verify_ok(s->m)) {
                    fprintf(stderr, "error: --draft-lookup: slot %d needs "
                                    "host-readable hidden work (rerun with "
                                    "--gpu off)\n", i);
                    return 1;
                }
                s->e.lookup_on = true;
                s->e.draft_k = draft_k;
            }
        }

        if (draft_path) {
            bool any_draft = false;
            for (int i = 0; i < parallel; i++)
                any_draft |= SV.slots[i].e.dm != NULL;
            if (!any_draft)
                SV.draft_note = "the draft was refused at load; see the "
                                "startup log for the gate that rejected it";
        }

        if (parallel == 1) {
            // join the registry machinery so POST /unload frees the resident
            // model (the next request lazily reloads it) and --ttl works.
            // slot 0's containers are the caller's; borrowed avoids freeing
            // them on the first unload
            snprintf(SV.reg[0].name, sizeof(SV.reg[0].name), "%s", SV.model_name);
            snprintf(SV.reg[0].path, sizeof(SV.reg[0].path), "%s", model_path);
            SV.reg[0].tmpl = tmpl;
            SV.model_name = SV.reg[0].name;
            SV.n_reg = 1;
            SV.single = true;
            SV.borrowed = true;
            resident_store(0);
            SV.last_used = now_s();
            SV.draft = SV.slots[0].e.dm;
            SV.draft_k = draft_k;
            SV.draft_source = SV.draft ? "model" : mp->mtp ? "mtp"
                            : draft_lookup ? "lookup" : NULL;
            // a draft the gates rejected at startup stays rejected: only a
            // draft that actually served is worth reloading after /unload
            if (SV.draft) SV.draft_path = draft_path;
            if (!init_swap_runtime(mp, threads_per_slot, ttl)) return 1;
        }
    }

    for (int i = 0; i < SV.n_slots; i++) SV.slots[i].adapter = -1;
    if (!load_adapters(base)) return 1;

    // last long stop is behind us; a signal from here on is either caught
    // right now or by the published listener below
    if (stop_was_requested()) {
        fprintf(stderr, "shutdown requested during startup — exiting\n");
        return 0;
    }

    sock_t lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd == SOCK_INVALID) {
        fprintf(stderr, "error: cannot create server socket\n");
        return 1;
    }
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "error: cannot bind 127.0.0.1:%d (%s)\n", port, sock_errstr());
        sock_close(lfd);
        return 1;
    }
    if (listen(lfd, 64) != 0) {
        fprintf(stderr, "error: cannot listen on 127.0.0.1:%d (%s)\n", port,
                sock_errstr());
        sock_close(lfd);
        return 1;
    }
#ifndef _WIN32
    listener_fd = lfd;
#else
    win_listener_socket = lfd;
#endif

    // Every slot's model exists now, which is the only precondition the batch
    // has. Declining is not an error: sched_generate then runs the untouched
    // solo path, which is what a single slot, swap mode, or a backend without
    // batched kernels all get.
    bool batched = sched_start();

    for (int i = 0; i < parallel; i++) {
        if (pthread_create(&SV.slots[i].th, NULL, slot_worker, &SV.slots[i]) != 0) {
            fprintf(stderr, "error: cannot start server slot %d\n", i);
            sock_close(lfd);
            return 1;
        }
    }

    if (SV.n_reg > 0 && !SV.single)
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d models, swap on demand"
                " (ttl %ds)\n",
                port, SV.n_reg, SV.ttl);
    else
        fprintf(stderr,
                "server listening on http://127.0.0.1:%d — %d slot%s sharing %d threads%s\n",
                port, parallel, parallel > 1 ? "s" : "", shared_pool_threads,
                batched ? ", continuous batching" : "");
    fputs("  POST /v1/chat/completions | POST /v1/responses | POST /v1/completions\n"
          "  POST /v1/embeddings | POST /v1/rerank | POST /v1/messages"
          " | POST /v1/messages/count_tokens\n"
          "  GET /v1/models | GET /v1/capabilities | GET /health | GET /metrics\n"
          "  GET /v1/runner/prefix-cache | POST /v1/runner/prefix-cache/clear"
          " | POST /unload\n"
          "  GET /v1/runner/provenance | POST /v1/runner/contexts"
          " | GET /v1/runner/contexts | DELETE /v1/runner/contexts/{id}"
          " | POST /v1/runner/contexts/{id}/snapshot\n"
          "  GET /v1/responses/{id} | GET /v1/responses/{id}/input_items"
          " | DELETE /v1/responses/{id}\n", stderr);

    // Say it in the banner, not only at init.
    //
    // A backend that cannot run this model's exact expert variant prints its
    // reason from gpu_init(), which scrolls past above the load line and is
    // easy to miss. The banner is the last thing on screen when a server comes
    // up, so the CPU-expert placement belongs here too.
    if (SV.n_slots > 0 && SV.slots[0].m && SV.slots[0].m->n_expert > 0 &&
        !SV.slots[0].m->gpu)
        fprintf(stderr,
                "  note: this is a sparse-MoE model and its experts are running"
                " on the CPU — expect a fraction of dense throughput\n");

    for (;;) {
        // covers the race where the signal landed after the socket-creation
        // check but before listener_fd published: the handler had no fd to
        // close, so accept() would block forever with the flag already set.
        // A signal landing after this check finds listener_fd published and
        // closes it, so accept() fails; no window remains.
        if (stop_was_requested()) break;
        sock_t cfd = accept(lfd, NULL, NULL);
        if (cfd == SOCK_INVALID) {
#ifndef _WIN32
            if (stop_requested) break;
#else
            if (win_stop_requested) break;
#endif
            if (errno == EINTR) continue;
            break;
        }
        if (!accept_fastpath(cfd)) q_push(cfd);
    }
    // Stop admission first, fail work that has not started, then allow active
    // requests to finish before dismantling the scheduler and model state.
#ifndef _WIN32
    if (listener_fd >= 0) {
        listener_fd = -1;
        sock_close(lfd);
    }
#else
    if (win_listener_socket != INVALID_SOCKET) {
        win_listener_socket = INVALID_SOCKET;
        sock_close(lfd);
    }
#endif
    queue_shutdown();
    // A worker parked in a --wait-for-vram queue would pin the joins below
    // for up to the full wait; the load gives up at its next poll instead.
    atomic_store(&SV.load_cancel, 1);
    for (int i = 0; i < parallel; i++) pthread_join(SV.slots[i].th, NULL);
    sched_shutdown();
    atomic_store(&SV.shutdown, true);
    if (SV.reaper_started) pthread_join(SV.reaper_th, NULL);

    for (int i = 0; i < parallel; i++) {
        free(SV.slots[i].e.hist);
        free(SV.slots[i].hint_buf);
    }
    if (SV.n_reg > 0) {
        pthread_mutex_lock(&SV.swap_mu);
        unload_draft();
        unload_resident();
        pthread_mutex_unlock(&SV.swap_mu);
        pthread_mutex_destroy(&SV.swap_mu);
    } else {
        provenance_note_unload();
        tokenizer_free(tok);
        for (int i = 0; i < parallel; i++) {
            model_t *draft = SV.slots[i].e.dm;
            if (draft) { model_free(draft); free(draft); }
            model_free(SV.slots[i].m);
            if (i > 0) free(SV.slots[i].m);
        }
    }
    prefix_cache_clear();
    // the slots' models are gone; nothing borrows an adapter any more
    for (int i = 0; i < SV.n_adapters; i++) model_lora_set_free(SV.adapters[i].set);
    SV.n_adapters = 0;
    pthread_cond_destroy(&SV.q.cv);
    pthread_mutex_destroy(&SV.q.mu);
    free(SV.slots);
    return 0;
}
