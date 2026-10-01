#include "decide.h"
#include "envelope.h"
#include "runner.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// log-sum-exp of raw logits in double, the same arithmetic the sampler's
// logprob capture uses (max first, then a double accumulator)
static double lse(const float *l, int n) {
    float mx = l[0];
    for (int i = 1; i < n; i++) if (l[i] > mx) mx = l[i];
    double s = 0;
    for (int i = 0; i < n; i++) s += exp((double)l[i] - mx);
    return mx + log(s);
}

typedef struct { int idx; int32_t *toks; int n; int common; } opt_seq;

static int cmp_seq(const void *a, const void *b) {
    const opt_seq *x = a, *y = b;
    int n = x->n < y->n ? x->n : y->n;
    for (int i = 0; i < n; i++)
        if (x->toks[i] != y->toks[i]) return x->toks[i] < y->toks[i] ? -1 : 1;
    return x->n - y->n;
}

// `mode` is how the prompt (and prompt + option) is tokenized: TOK_TEXT for
// caller text, TOK_PROMPT for a chat render whose control tokens sit inside
// the template's marks. An option is always appended as plain bytes.
static bool score_mode(engine *e, const char *prompt, tok_mode mode,
                       const char *const *options, int n_opt,
                       decide_result *out, int *prompt_tokens, const char **err) {
    *err = NULL;
    model_t *m = e->m;
    int32_t *ptoks = NULL;
    int np = tok_encode_fit(e->tok, prompt, true, mode, 0, &ptoks);
    if (np < 0) { *err = "out of memory tokenizing the prompt"; return false; }
    if (np < 1) { free(ptoks); *err = "empty prompt"; return false; }
    *prompt_tokens = np;
    size_t plen = strlen(prompt);
    opt_seq *seq = calloc((size_t)n_opt, sizeof *seq);
    if (!seq) { free(ptoks); *err = "out of memory"; return false; }
    bool ok = true;
    int maxn = 0;
    for (int i = 0; i < n_opt && ok; i++) {
        const char *o = options[i];
        if (!o || !*o) { *err = "empty option"; ok = false; break; }
        size_t olen = strlen(o);
        char *cat = malloc(plen + olen + 1);
        if (!cat) { *err = "out of memory"; ok = false; break; }
        memcpy(cat, prompt, plen); memcpy(cat + plen, o, olen + 1);
        // the option tokenised IN CONTEXT: the concatenation, diffed against
        // the prompt's own tokens; a boundary merge shortens `common` and is
        // then scored as part of the option
        int nf = tok_encode_fit(e->tok, cat, true, mode, 0, &seq[i].toks);
        free(cat);
        if (nf < 0) { *err = "out of memory tokenizing an option"; ok = false; break; }
        int c = 0;
        while (c < np && c < nf && ptoks[c] == seq[i].toks[c]) c++;
        if (c >= nf) { *err = "an option tokenises to nothing in context"; ok = false; break; }
        if (c < 1) { *err = "the prompt is too short to score against"; ok = false; break; }
        if (nf > m->n_ctx) { *err = "prompt plus option exceeds the context window"; ok = false; break; }
        seq[i].idx = i; seq[i].n = nf; seq[i].common = c;
        if (nf > maxn) maxn = nf;
    }
    double *lp_at = NULL;
    unsigned char *has = NULL;
    if (ok) {
        lp_at = malloc(sizeof(double) * (size_t)maxn);
        has = calloc((size_t)maxn, 1);
        if (!lp_at || !has) { *err = "out of memory"; ok = false; }
    }
    // Sorted traversal: consecutive options share their longest token
    // prefixes, and lp_at[d] (the conditional of path[d] given path[:d]) is
    // reused for every depth the next option shares with the fed path. has[d]
    // says which depths were actually computed along the current chain, so a
    // depth never scored (an option whose boundary merge reaches below the
    // others' common point) is computed rather than assumed.
    if (ok) qsort(seq, (size_t)n_opt, sizeof *seq, cmp_seq);
    const int32_t *path = NULL; int path_n = 0;
    for (int i = 0; i < n_opt && ok; i++) {
        opt_seq *s = &seq[i];
        int shared = 0;
        if (path) {
            int lim = s->n < path_n ? s->n : path_n;
            while (shared < lim && path[shared] == s->toks[shared]) shared++;
        }
        for (int d = shared; d < maxn; d++) has[d] = 0;
        int start = s->common;
        while (start < shared && has[start]) start++;
        // hold the KV at s->toks[:start-1], then feed token start-1 ALONE so
        // the logits that score depth `start` come from a solo forward
        engine_rewind(e, s->toks, start - 1);
        if (e->pos < start - 1 &&
            !engine_feed(e, s->toks + e->pos, (start - 1) - e->pos)) {
            *err = "engine feed failed (context or memory)"; ok = false; break;
        }
        const float *L = engine_feed(e, &s->toks[start - 1], 1);
        if (!L) { *err = "engine feed failed (context or memory)"; ok = false; break; }
        double total = 0;
        for (int d = s->common; d < start; d++) total += lp_at[d];
        for (int d = start; d < s->n; d++) {
            double v = (double)L[s->toks[d]] - lse(L, m->n_vocab);
            lp_at[d] = v; has[d] = 1; total += v;
            L = engine_feed(e, &s->toks[d], 1);   // also keeps hist == the path
            if (!L) { *err = "engine feed failed (context or memory)"; ok = false; break; }
        }
        if (!ok) break;
        out[s->idx].lp = total;
        out[s->idx].n_tok = s->n - s->common;
        path = s->toks; path_n = s->n;
    }
    free(lp_at); free(has);
    for (int i = 0; i < n_opt; i++) free(seq[i].toks);
    free(seq); free(ptoks);
    return ok;
}

bool decide_score(engine *e, const char *prompt, const char *const *options, int n_opt,
                  decide_result *out, int *prompt_tokens, const char **err) {
    return score_mode(e, prompt, TOK_TEXT, options, n_opt, out, prompt_tokens, err);
}

// ---------------------------------------------------------------- request

static const char *jstr(const jv *o, const char *k) {
    jv *v = jv_get((jv *)o, k);
    return v && v->type == J_STR ? v->str : NULL;
}

int decide_handle(engine *e, const jv *req, const char *model_name, sbuf *out, const char **err) {
    *err = NULL;
    const char *state = jstr(req, "state");
    if (!state) { *err = "missing state (a string)"; return 400; }
    const char *aprefix = jstr(req, "answer_prefix");
    jv *ap = jv_get((jv *)req, "answer_prefix");
    if (ap && ap->type != J_NULL && !aprefix) { *err = "answer_prefix must be a string"; return 400; }
    if (!aprefix) aprefix = "";
    // Two renderings, chosen by the caller and stamped in the envelope.
    // raw-v1 (the default, unchanged since R13.10's invariance results were
    // taken on it): state, blank line, question, newline, answer_prefix, then
    // the option. continuation-v1: the option follows the state directly,
    // nothing injected, which is the loglikelihood readout a benchmark annex
    // needs (HellaSwag, ARC, PIQA, WinoGrande, non-CoT MMLU score a
    // continuation of the context; lm-eval's acc is exactly this sum of
    // conditionals, acc_norm follows client-side from the ending's length).
    // Under it the question is a label only (may be absent or empty) and
    // answer_prefix has no place.
    const char *rend = jstr(req, "rendering");
    jv *rv = jv_get((jv *)req, "rendering");
    if (rv && rv->type != J_NULL && !rend) { *err = "rendering must be a string"; return 400; }
    if (!rend) rend = "raw-v1";
    bool cont;
    if (!strcmp(rend, "raw-v1")) cont = false;
    else if (!strcmp(rend, "continuation-v1")) cont = true;
    else { *err = "unknown rendering: raw-v1 or continuation-v1"; return 400; }
    if (cont && *aprefix) { *err = "answer_prefix has no place in the continuation-v1 rendering"; return 400; }
    jv *qs = jv_get((jv *)req, "questions");
    if (!qs || qs->type != J_ARR || qs->n < 1) { *err = "missing questions (a non-empty array)"; return 400; }
    // receipt digests over the state bytes and a canonical rendering of the
    // questions, computed before any work so a refused request has the same
    // digests as an accepted one. Every field is length-prefixed ("q<len>:"
    // then "o<len>:" per option), so no byte an option can contain is a
    // separator: with US/RS separators, ["a\x1eb","c"] and ["a","b\x1ec"]
    // hashed alike (found 2026-09-22).
    char state_sha[65], q_sha[65];
    envelope_data_sha256(state, strlen(state), state_sha);
    {
        // Lengths are printed as unsigned long long: the canonical form feeds
        // the request hash, and %zu through the MinGW printf checker reads as
        // an unknown conversion (Windows build warning, sweep 2026-09-27).
        // Same digits, so the hash is unchanged.
        sbuf canon = {0};
        for (int i = 0; i < qs->n; i++) {
            const jv *q = qs->items[i];
            const char *qt = jstr(q, "question");
            size_t ql = qt ? strlen(qt) : 0;
            sb_fmt(&canon, "q%llu:", (unsigned long long)ql);
            if (qt) sb_put(&canon, qt, ql);
            jv *opts = q->type == J_OBJ ? jv_get((jv *)q, "options") : NULL;
            if (opts && opts->type == J_ARR)
                for (int j = 0; j < opts->n; j++) {
                    const char *o = opts->items[j]->type == J_STR ? opts->items[j]->str : "";
                    size_t ol = strlen(o);
                    sb_fmt(&canon, "o%llu:", (unsigned long long)ol);
                    sb_put(&canon, o, ol);
                }
            sb_lit(&canon, "\n");
        }
        envelope_data_sha256(canon.s ? canon.s : "", canon.n, q_sha);
        free(canon.s);
    }
    size_t slen = strlen(state), alen = strlen(aprefix);
    static unsigned long long counter = 0;
    sb_fmt(out, "{\"id\":\"decide-%llu\",\"object\":\"decide\",\"model\":\"", ++counter);
    sb_esc(out, model_name, strlen(model_name));
    sb_lit(out, "\",\"decisions\":[");
    int total_prompt = 0;
    int status = 200;
    for (int i = 0; i < qs->n && status == 200; i++) {
        const jv *q = qs->items[i];
        if (q->type != J_OBJ) { *err = "each question must be an object"; status = 400; break; }
        const char *qt = jstr(q, "question");
        if (!cont && (!qt || !*qt)) { *err = "a question is missing its text"; status = 400; break; }
        if (!qt) qt = "";
        jv *opts = jv_get((jv *)q, "options");
        if (!opts || opts->type != J_ARR || opts->n < 2) { *err = "a question needs at least 2 options"; status = 400; break; }
        const char *id = jstr(q, "id");
        int n_opt = opts->n;
        const char **ov = malloc(sizeof(char *) * (size_t)n_opt);
        decide_result *res = malloc(sizeof(decide_result) * (size_t)n_opt);
        if (!ov || !res) { free(ov); free(res); *err = "out of memory"; status = 500; break; }
        bool bad = false;
        for (int j = 0; j < n_opt && !bad; j++) {
            ov[j] = opts->items[j]->type == J_STR ? opts->items[j]->str : NULL;
            if (!ov[j] || !*ov[j]) { *err = "options must be non-empty strings"; bad = true; }
            for (int k = 0; k < j && !bad; k++)
                if (!strcmp(ov[j], ov[k])) { *err = "duplicate option string"; bad = true; }
        }
        if (bad) { free(ov); free(res); status = 400; break; }
        size_t qlen = strlen(qt);
        char *prompt = malloc(slen + 2 + qlen + 1 + alen + 1);
        if (!prompt) { free(ov); free(res); *err = "out of memory"; status = 500; break; }
        if (cont) {
            // continuation-v1: the state is the whole prompt
            memcpy(prompt, state, slen + 1);
        } else {
            // raw-v1: state, blank line, question, newline, prefix
            memcpy(prompt, state, slen); memcpy(prompt + slen, "\n\n", 2);
            memcpy(prompt + slen + 2, qt, qlen); prompt[slen + 2 + qlen] = '\n';
            memcpy(prompt + slen + 3 + qlen, aprefix, alen + 1);
        }
        int ptoks = 0;
        const char *serr = NULL;
        bool ok = decide_score(e, prompt, ov, n_opt, res, &ptoks, &serr);
        free(prompt);
        if (!ok) {
            free(ov); free(res);
            *err = serr ? serr : "scoring failed";
            status = strstr(*err, "memory") ? 500 : 400;
            break;
        }
        total_prompt += ptoks;
        double mx = res[0].lp;
        for (int j = 1; j < n_opt; j++) if (res[j].lp > mx) mx = res[j].lp;
        double z = 0;
        for (int j = 0; j < n_opt; j++) z += exp(res[j].lp - mx);
        int argmax = 0;
        for (int j = 1; j < n_opt; j++) if (res[j].lp > res[argmax].lp) argmax = j;
        sb_fmt(out, "%s{\"id\":\"", i ? "," : "");
        if (id) sb_esc(out, id, strlen(id)); else sb_fmt(out, "%d", i);
        sb_lit(out, "\",\"options\":[");
        for (int j = 0; j < n_opt; j++) { sb_fmt(out, "%s\"", j ? "," : ""); sb_esc(out, ov[j], strlen(ov[j])); sb_lit(out, "\""); }
        sb_lit(out, "],\"logprobs\":[");
        for (int j = 0; j < n_opt; j++) sb_fmt(out, "%s%.6f", j ? "," : "", res[j].lp);
        sb_lit(out, "],\"probs\":[");
        for (int j = 0; j < n_opt; j++) sb_fmt(out, "%s%.6g", j ? "," : "", exp(res[j].lp - mx) / z);
        sb_fmt(out, "],\"argmax\":%d,\"n_tokens\":[", argmax);
        for (int j = 0; j < n_opt; j++) sb_fmt(out, "%s%d", j ? "," : "", res[j].n_tok);
        sb_lit(out, "]}");
        free(ov); free(res);
    }
    if (status != 200) return status;
    sb_fmt(out, "],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":0,\"total_tokens\":%d},"
                "\"envelope\":{\"runner_version\":\"%s\",\"state_sha256\":\"%s\",\"questions_sha256\":\"%s\","
                "\"rendering\":\"%s\"}}", total_prompt, total_prompt, RUNNER_VERSION, state_sha, q_sha, rend);
    if (out->failed) { *err = "out of memory building the response"; return 500; }
    return 200;
}

// ---------------------------------------------------------------- rerank

// R10.3.1: relevance as a constrained two-way choice. Each document is put to
// the served model as a question with exactly two legal answers and scored
// with the same exact readout as /v1/decide; the relevance score is P(yes)
// renormalized over {yes, no}, and the logit is log P(yes) - log P(no), the
// same number in the units a margin is read in.

#define RERANK_INSTRUCTION \
    "Judge whether the document answers the query. Answer yes or no."

typedef struct { int idx; double logit, lp_yes, lp_no; } rr_row;

static int rr_cmp(const void *a, const void *b) {
    const rr_row *x = a, *y = b;
    if (x->logit != y->logit) return x->logit > y->logit ? -1 : 1;
    return x->idx - y->idx;   // a tie keeps the input order
}

// A document is a string or an object with a string "text" (the Cohere and
// Jina shape); anything else is refused by name.
static const char *rr_doc_text(const jv *d) {
    if (d->type == J_STR) return d->str;
    if (d->type == J_OBJ) return jstr(d, "text");
    return NULL;
}

// The chat-v1 prompt: the instruction as the system turn, query and document
// as the user turn, rendered by the served model's own template with thinking
// off, so the answer is the first thing the assistant turn holds.
static char *rr_render_chat(int tmpl, const char *instr, const char *q,
                            const char *d) {
    sbuf u = {0};
    sb_lit(&u, "Query: ");
    sb_put(&u, q, strlen(q));
    sb_lit(&u, "\nDocument: ");
    sb_put(&u, d, strlen(d));
    sb_put(&u, "", 1);
    if (u.failed) { free(u.s); return NULL; }
    chat_msg msgs[2] = { { .role = "system", .content = instr },
                         { .role = "user", .content = u.s } };
    char *p = NULL;
    for (size_t cap = 1024 + u.n + strlen(instr);; cap *= 2) {
        char *b = malloc(cap);
        if (!b) break;
        size_t n = render_messages_with_tools(tmpl, msgs, 2, true, THINK_OFF,
                                              NULL, b, cap);
        if (n == SIZE_MAX) { free(b); break; }
        if (n < cap && strlen(b) + 1 < cap) { p = b; break; }
        free(b);
        if (cap > ((size_t)1 << 30)) break;
    }
    free(u.s);
    return p;
}

static char *rr_render_raw(const char *instr, const char *q, const char *d) {
    sbuf b = {0};
    sb_put(&b, instr, strlen(instr));
    sb_lit(&b, "\n\nQuery: ");
    sb_put(&b, q, strlen(q));
    sb_lit(&b, "\nDocument: ");
    sb_put(&b, d, strlen(d));
    sb_lit(&b, "\nRelevant:");
    sb_put(&b, "", 1);
    if (b.failed) { free(b.s); return NULL; }
    return b.s;
}

int rerank_handle(engine *e, const jv *req, const char *model_name, int tmpl,
                  sbuf *out, const char **err) {
    *err = NULL;
    const char *query = jstr(req, "query");
    if (!query || !*query) { *err = "missing query (a non-empty string)"; return 400; }
    jv *docs = jv_get((jv *)req, "documents");
    if (!docs || docs->type != J_ARR || docs->n < 1) {
        *err = "missing documents (a non-empty array)"; return 400;
    }
    for (int i = 0; i < docs->n; i++) {
        const char *t = rr_doc_text(docs->items[i]);
        if (!t || !*t) {
            *err = "each document must be a non-empty string or an object "
                   "with a non-empty string \"text\"";
            return 400;
        }
    }
    const char *instr = jstr(req, "instruction");
    jv *iv = jv_get((jv *)req, "instruction");
    if (iv && iv->type != J_NULL && (!instr || !*instr)) {
        *err = "instruction must be a non-empty string"; return 400;
    }
    if (!instr) instr = RERANK_INSTRUCTION;
    const char *rend = jstr(req, "rendering");
    jv *rv = jv_get((jv *)req, "rendering");
    if (rv && rv->type != J_NULL && !rend) { *err = "rendering must be a string"; return 400; }
    if (!rend) rend = "chat-v1";
    bool chat;
    if (!strcmp(rend, "chat-v1")) chat = true;
    else if (!strcmp(rend, "raw-v1")) chat = false;
    else { *err = "unknown rendering: chat-v1 or raw-v1"; return 400; }
    if (chat && tmpl == TMPL_HARMONY) {
        // Harmony's answer follows a channel header the model chooses, not
        // the generation prompt; scoring "yes" right after it would read a
        // distribution the model never answers from.
        *err = "rendering chat-v1 cannot score harmony's channel protocol; "
               "use rendering raw-v1";
        return 400;
    }
    int top_n = docs->n;
    jv *tn = jv_get((jv *)req, "top_n");
    if (tn && tn->type != J_NULL) {
        if (tn->type != J_NUM || tn->num != floor(tn->num) || tn->num < 1) {
            *err = "top_n must be a positive integer"; return 400;
        }
        if (tn->num < top_n) top_n = (int)tn->num;
    }
    bool ret_docs = false;
    jv *rd = jv_get((jv *)req, "return_documents");
    if (rd && rd->type != J_NULL) {
        if (rd->type != J_BOOL) {
            *err = "return_documents must be a boolean"; return 400;
        }
        ret_docs = rd->b;
    }

    // the chat answer is the first assistant token; the raw prompt ends in a
    // colon, so its answers carry their leading space
    const char *const opts_chat[2] = { "yes", "no" };
    const char *const opts_raw[2]  = { " yes", " no" };
    rr_row *rows = calloc((size_t)docs->n, sizeof *rows);
    if (!rows) { *err = "out of memory"; return 500; }
    int total_prompt = 0, status = 200;
    for (int i = 0; i < docs->n && status == 200; i++) {
        const char *d = rr_doc_text(docs->items[i]);
        char *prompt = chat ? rr_render_chat(tmpl, instr, query, d)
                            : rr_render_raw(instr, query, d);
        if (!prompt) { *err = "out of memory rendering a document"; status = 500; break; }
        decide_result res[2];
        int ptoks = 0;
        const char *serr = NULL;
        // Every document is scored from an empty KV. Reusing the rows the
        // previous document (or an earlier request) left behind fed this
        // one's prefix in a batch whose width depended on what came before,
        // and CPU prefill is not batch-invariant: a score then moved in its
        // last digit with the document order. Cold, a document's score is a
        // function of its own prompt and nothing else. The price is the
        // shared instruction and query being prefilled once per document.
        engine_reset(e);
        bool ok = score_mode(e, prompt, chat ? TOK_PROMPT : TOK_TEXT,
                             chat ? opts_chat : opts_raw, 2, res, &ptoks, &serr);
        free(prompt);
        if (!ok) {
            *err = serr ? serr : "scoring failed";
            status = strstr(*err, "memory") ? 500 : 400;
            break;
        }
        total_prompt += ptoks;
        rows[i].idx = i;
        rows[i].lp_yes = res[0].lp;
        rows[i].lp_no = res[1].lp;
        rows[i].logit = res[0].lp - res[1].lp;
    }
    if (status != 200) { free(rows); return status; }
    qsort(rows, (size_t)docs->n, sizeof *rows, rr_cmp);

    char q_sha[65], d_sha[65], i_sha[65];
    envelope_data_sha256(query, strlen(query), q_sha);
    envelope_data_sha256(instr, strlen(instr), i_sha);
    {
        // length-prefixed, for the reason decide_handle gives
        sbuf canon = {0};
        for (int i = 0; i < docs->n; i++) {
            const char *d = rr_doc_text(docs->items[i]);
            size_t dl = strlen(d);
            sb_fmt(&canon, "d%llu:", (unsigned long long)dl);
            sb_put(&canon, d, dl);
        }
        envelope_data_sha256(canon.s ? canon.s : "", canon.n, d_sha);
        free(canon.s);
    }
    static unsigned long long counter = 0;
    sb_fmt(out, "{\"id\":\"rerank-%llu\",\"object\":\"rerank\",\"model\":\"", ++counter);
    sb_esc(out, model_name, strlen(model_name));
    sb_lit(out, "\",\"results\":[");
    for (int k = 0; k < top_n; k++) {
        const rr_row *r = &rows[k];
        // logistic of the logit, written so neither tail overflows
        double p = r->logit >= 0 ? 1.0 / (1.0 + exp(-r->logit))
                                 : exp(r->logit) / (1.0 + exp(r->logit));
        sb_fmt(out, "%s{\"index\":%d,\"relevance_score\":%.9g,\"logit\":%.9g,"
                    "\"margin\":", k ? "," : "", r->idx, p, r->logit);
        if (k + 1 < docs->n) sb_fmt(out, "%.9g", r->logit - rows[k + 1].logit);
        else                 sb_lit(out, "null");
        sb_fmt(out, ",\"logprobs\":{\"yes\":%.9g,\"no\":%.9g}", r->lp_yes, r->lp_no);
        if (ret_docs) {
            const char *d = rr_doc_text(docs->items[r->idx]);
            sb_lit(out, ",\"document\":{\"text\":\"");
            sb_esc(out, d, strlen(d));
            sb_lit(out, "\"}");
        }
        sb_lit(out, "}");
    }
    free(rows);
    sb_fmt(out, "],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":0,"
                "\"total_tokens\":%d},\"envelope\":{\"runner_version\":\"%s\","
                "\"query_sha256\":\"%s\",\"documents_sha256\":\"%s\","
                "\"instruction_sha256\":\"%s\",\"rendering\":\"%s\","
                "\"options\":[\"yes\",\"no\"]}}",
           total_prompt, total_prompt, RUNNER_VERSION, q_sha, d_sha, i_sha, rend);
    if (out->failed) { *err = "out of memory building the response"; return 500; }
    return 200;
}
