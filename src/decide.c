#include "decide.h"
#include "envelope.h"
#include "runner.h"
#include <math.h>
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

bool decide_score(engine *e, const char *prompt, const char *const *options, int n_opt,
                  decide_result *out, int *prompt_tokens, const char **err) {
    *err = NULL;
    model_t *m = e->m;
    int32_t *ptoks = NULL;
    int np = tok_encode_fit(e->tok, prompt, true, TOK_TEXT, 0, &ptoks);
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
        int nf = tok_encode_fit(e->tok, cat, true, TOK_TEXT, 0, &seq[i].toks);
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
    jv *qs = jv_get((jv *)req, "questions");
    if (!qs || qs->type != J_ARR || qs->n < 1) { *err = "missing questions (a non-empty array)"; return 400; }
    // receipt digests over the state bytes and a canonical rendering of the
    // questions (question, US, options joined by RS, LF), computed before any
    // work so a refused request has the same digests as an accepted one
    char state_sha[65], q_sha[65];
    envelope_data_sha256(state, strlen(state), state_sha);
    {
        sbuf canon = {0};
        for (int i = 0; i < qs->n; i++) {
            const jv *q = qs->items[i];
            const char *qt = jstr(q, "question");
            if (qt) sb_put(&canon, qt, strlen(qt));
            sb_lit(&canon, "\x1f");
            jv *opts = q->type == J_OBJ ? jv_get((jv *)q, "options") : NULL;
            if (opts && opts->type == J_ARR)
                for (int j = 0; j < opts->n; j++) {
                    if (j) sb_lit(&canon, "\x1e");
                    const char *o = opts->items[j]->type == J_STR ? opts->items[j]->str : "";
                    sb_put(&canon, o, strlen(o));
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
        if (!qt || !*qt) { *err = "a question is missing its text"; status = 400; break; }
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
        // the fixed raw rendering: state, blank line, question, newline, prefix
        size_t qlen = strlen(qt);
        char *prompt = malloc(slen + 2 + qlen + 1 + alen + 1);
        if (!prompt) { free(ov); free(res); *err = "out of memory"; status = 500; break; }
        memcpy(prompt, state, slen); memcpy(prompt + slen, "\n\n", 2);
        memcpy(prompt + slen + 2, qt, qlen); prompt[slen + 2 + qlen] = '\n';
        memcpy(prompt + slen + 3 + qlen, aprefix, alen + 1);
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
                "\"rendering\":\"raw-v1\"}}", total_prompt, total_prompt, RUNNER_VERSION, state_sha, q_sha);
    if (out->failed) { *err = "out of memory building the response"; return 500; }
    return 200;
}
