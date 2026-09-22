// Typed decisions (R13.10): score caller-supplied option strings as verbatim
// continuations of a prompt on the slot's own KV, one prefill per state,
// zero sampled tokens, and return a distribution over the options.
//
// The readout is the exact one: an option is tokenised IN CONTEXT (the
// concatenation prompt + option, diffed against the prompt's tokens, so a
// merge at the boundary is scored as part of the option), its log-prob is
// the sum of its tokens' conditionals under the model's real distribution
// (product of conditionals), and the probabilities are the softmax over the
// given options' totals. Options sharing a token prefix share the prefix's
// conditionals: the sorted traversal below scores each trie node once.
//
// Every scored conditional comes from a solo one-token forward (the token
// before it is re-fed alone), so the numbers do not depend on option order
// or question order; the prompt's own rows come from the batched prefill,
// which is what serving does too.
#ifndef RUNNER_DECIDE_H
#define RUNNER_DECIDE_H

#include "engine.h"
#include "json.h"

typedef struct {
    double lp;     // log P(option | prompt), sum of in-context conditionals
    int    n_tok;  // scored tokens (the option's in-context token count)
} decide_result;

// Scores n_opt option strings against prompt on e's KV. Returns false with
// *err set (a static string) on a caller error (empty option, option that
// tokenises to nothing, prompt too short to score against) or on an engine
// failure (out of memory, context overflow). *prompt_tokens receives the
// prompt's token count. The KV is left holding a valid prefix.
bool decide_score(engine *e, const char *prompt, const char *const *options, int n_opt,
                  decide_result *out, int *prompt_tokens, const char **err);

// One request in the /v1/decide shape (state, answer_prefix, rendering,
// questions[]) to
// one response body appended to *out; returns the HTTP status (200, 400,
// 500) and, when not 200, *err names the problem. Shared by the server route
// and the --decide CLI mode so both produce the same bytes for the same
// request. `model_name` is echoed in the response.
int decide_handle(engine *e, const jv *req, const char *model_name, sbuf *out, const char **err);

#endif
