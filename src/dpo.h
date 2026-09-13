// Direct Preference Optimization over the adapter, as one pairwise step.
//
// arXiv 2305.18290, equation 7 and the gradient beside it:
//
//   L_DPO   = -log sigma( beta*log(pi(y_w|x)/pi_ref(y_w|x))
//                       - beta*log(pi(y_l|x)/pi_ref(y_l|x)) )
//   grad L  = -beta * sigma(r_l - r_w) * ( grad log pi(y_w) - grad log pi(y_l) )
//   r(x,y)  = beta * log( pi(y|x) / pi_ref(y|x) )
//
// With CE(y) = -log pi(y|x) the gradient rearranges to
//
//   grad L  = beta * sigma(r_l - r_w) * ( grad CE(y_w) - grad CE(y_l) )
//
// which is a SCALAR MULTIPLE of a difference of two weighted cross-entropy
// gradients. model_lora_backward_w already produces those, with a sign,
// because pos_w scales loss and gradient linearly and negatives are allowed.
// So DPO needs no new kernel here: two backward calls into one gradient
// buffer, two bypassed forwards for the reference, and one scalar multiply.
//
// Sequence log-probabilities are SUMS over the completion transitions, not
// length-normalized, which is what the paper's objective uses. Any deviation
// from the published objective is named in the research note, not hidden
// behind a default.
#ifndef RUNNER_DPO_H
#define RUNNER_DPO_H

#include "model.h"

typedef struct {
    double logp_w, logp_l;          // log pi_theta  of chosen / rejected
    double logp_ref_w, logp_ref_l;  // log pi_ref    of chosen / rejected
    double margin;                  // r_w - r_l, the logit inside the sigmoid
    double loss;                    // -log sigma(margin)
    double coeff;                   // beta * sigma(-margin): the grad scalar
    double acc;                     // 1.0 when the policy already ranks the
                                    // pair correctly (margin > 0), else 0.0
} dpo_step;

// The loss and its four log-probabilities, NO gradient and no tape: four
// forward passes. This is what a finite-difference gate perturbs.
bool dpo_loss(model_t *m, double beta,
              const int32_t *tw, int nw, const float *mw,
              const int32_t *tl, int nl, const float *ml,
              dpo_step *out);

// The same step, with the exact DPO gradient left in the adapter's gradient
// buffers ready for model_lora_adam_step. Zeroes them first. Costs two
// bypassed forwards plus two backwards; the policy log-probabilities come out
// of the backward calls themselves, so nothing is computed twice.
bool dpo_accumulate(model_t *m, double beta,
                    const int32_t *tw, int nw, const float *mw,
                    const int32_t *tl, int nl, const float *ml,
                    dpo_step *out);

#endif
