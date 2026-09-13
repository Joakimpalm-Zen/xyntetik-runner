#include "dpo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// -log sigma(z), evaluated the stable way on both branches. The naive form
// overflows for z around -750 and loses every bit of precision for large
// positive z, and a DPO run spends most of its steps at large positive z
// because that is what convergence looks like.
static double neg_log_sigmoid(double z) {
    return z >= 0 ? log1p(exp(-z)) : -z + log1p(exp(z));
}

static bool ref_logps(model_t *m,
                      const int32_t *tw, int nw, const float *mw,
                      const int32_t *tl, int nl, const float *ml,
                      double *lp_w, double *lp_l) {
    double nll_w = 0, nll_l = 0;
    if (!model_lora_bypass(m, true)) return false;
    bool ok = model_seq_nll(m, tw, nw, mw, &nll_w) &&
              model_seq_nll(m, tl, nl, ml, &nll_l);
    // restored even on failure: a half-bypassed adapter would make every
    // later step train against a reference it did not measure
    ok = model_lora_bypass(m, false) && ok;
    if (!ok) return false;
    *lp_w = -nll_w;
    *lp_l = -nll_l;
    return true;
}

static void finish(dpo_step *out, double beta) {
    out->margin = beta * ((out->logp_w - out->logp_ref_w) -
                          (out->logp_l - out->logp_ref_l));
    out->loss = neg_log_sigmoid(out->margin);
    // sigma(-margin) = 1/(1+exp(margin)), written so a large positive margin
    // saturates to 0 instead of dividing by an overflowed exp.
    out->coeff = out->margin >= 0
        ? beta * exp(-out->margin) / (1.0 + exp(-out->margin))
        : beta / (1.0 + exp(out->margin));
    out->acc = out->margin > 0 ? 1.0 : 0.0;
}

bool dpo_loss(model_t *m, double beta,
              const int32_t *tw, int nw, const float *mw,
              const int32_t *tl, int nl, const float *ml,
              dpo_step *out) {
    dpo_step s = {0};
    if (!ref_logps(m, tw, nw, mw, tl, nl, ml, &s.logp_ref_w, &s.logp_ref_l))
        return false;
    double nll_w = 0, nll_l = 0;
    if (!model_seq_nll(m, tw, nw, mw, &nll_w) ||
        !model_seq_nll(m, tl, nl, ml, &nll_l)) return false;
    s.logp_w = -nll_w;
    s.logp_l = -nll_l;
    finish(&s, beta);
    *out = s;
    return true;
}

bool dpo_accumulate(model_t *m, double beta,
                    const int32_t *tw, int nw, const float *mw,
                    const int32_t *tl, int nl, const float *ml,
                    dpo_step *out) {
    dpo_step s = {0};
    // The reference first, while the gradient buffers are still untouched:
    // the bypass mutates slot scales, and doing it between the two backward
    // calls would put the two halves of one pair under different policies.
    if (!ref_logps(m, tw, nw, mw, tl, nl, ml, &s.logp_ref_w, &s.logp_ref_l))
        return false;

    // negated mask for the rejected half, so its gradient is SUBTRACTED into
    // the same buffers. nl-1 entries: pos_w indexes transitions.
    float *neg = NULL;
    if (nl > 1) {
        neg = malloc(sizeof(float) * (size_t)(nl - 1));
        if (!neg) return false;
        for (int i = 0; i < nl - 1; i++) neg[i] = ml ? -ml[i] : -1.0f;
    }

    double loss_w = 0, loss_l = 0;
    model_lora_grad_zero(m);
    bool ok = model_lora_backward_w(m, tw, nw, mw, &loss_w) &&
              model_lora_backward_w(m, tl, nl, neg, &loss_l);
    free(neg);
    if (!ok) return false;
    // loss_w is +NLL(y_w); loss_l is -NLL(y_l), because its weights were
    // negated. The gradient buffers now hold grad CE(y_w) - grad CE(y_l).
    s.logp_w = -loss_w;
    s.logp_l = loss_l;
    finish(&s, beta);
    model_lora_grad_scale(m, (float)s.coeff);
    *out = s;
    return true;
}
