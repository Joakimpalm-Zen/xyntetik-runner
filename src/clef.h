// Clef (Cloudflare/clef, Cloudflare/clef-flash): a Qwen3.5/Qwen3.8 backbone
// with a joint schema head that turns a state and a schema of typed
// questions into one logit per allowed option, in one forward pass and with
// no generated text. Served as POST /v1/systemone, the Jev/SystemOne body.
//
// The reference is the publisher's own code, joint_schema_model.py in the
// model repository (encode_record, JointSchemaHead, systemone), not any
// other engine's port of it: the prompt is built piece by piece exactly as
// encode_record builds it, the head runs its module graph in f32, and the
// response follows systemone() field for field. GGUF files name the head's
// tensors as llama.cpp's converter writes them (dec.blk.N with the two
// evidence-routing blocks first, decision.*, token_types, decision.scales
// holding the three learned scalars already passed through exp/sigmoid);
// that is only the storage layout this file reads.
#ifndef RUNNER_CLEF_H
#define RUNNER_CLEF_H

#include "engine.h"
#include "json.h"
#include "model.h"

// torch.nn.LayerNorm: weight and bias, f32
typedef struct { float *w, *b; } clef_norm;

// torch.nn.MultiheadAttention: in_proj split into q, k, v (each [w, w]
// with its bias), out_proj [w, w] with its bias
typedef struct {
    gguf_tensor *wq, *wk, *wv, *wo;
    float *bq, *bk, *bv, *bo;
} clef_attn;

// One head block. The first n_routing are EvidenceRoutingLayer (cross
// attention from the options to the memory, query_norm and memory_norm,
// then a GELU feedforward); the next n_joint are
// torch.nn.TransformerDecoderLayer with norm_first (self attention over the
// fields, cross attention to the memory, GELU feedforward).
typedef struct {
    clef_norm self_norm;      // decoder: norm1
    clef_attn self_attn;      // decoder: self_attn
    clef_norm cross_norm;     // routing: query_norm, decoder: norm2
    clef_norm cross_norm_kv;  // routing: memory_norm (decoder: none)
    clef_attn cross_attn;     // routing: attention, decoder: multihead_attn
    clef_norm ffn_norm;       // routing: feedforward_norm, decoder: norm3
    gguf_tensor *ffn_up, *ffn_down;
    float *ffn_up_b, *ffn_down_b;
} clef_block;

struct clef_head {
    int hidden;          // the backbone's n_embd
    int width;           // the head's model width
    int n_head;          // attention heads in every head block
    int n_ff;            // feedforward width
    int n_routing;       // evidence-routing blocks (dec.blk.0 ..)
    int n_joint;         // decoder blocks after them
    float eps;           // LayerNorm eps (clef.attention.layer_norm_epsilon)
    clef_block *blocks;  // n_routing + n_joint
    clef_norm hidden_norm, option_summary_norm, field_norm, option_norm;
    gguf_tensor *proj_memory, *proj_question, *proj_option_question,
                *proj_global, *proj_option_context, *proj_option_lexical;
    float *type_embd;    // [3][width]: noul, choice, score
    gguf_tensor *scorer, *scorer_out;   // residual_scorer.0 and .3
    float *scorer_b, *scorer_out_b;
    // decision.scales: exp(min(prior_logit_scale, ln 100)),
    // exp(min(joint_logit_scale, ln 100)), sigmoid(residual_gate)
    float prior_scale, joint_scale, gate;
};

// Release a head bound by model.c (NULL is a no-op).
void clef_head_free(struct clef_head *h);

// The prompt of one record and the spans the head reads, as encode_record
// returns them: [start, end) token indices into ids.
typedef struct {
    char *id;            // the question id (the request's key)
    int   type;          // 0 noul, 1 choice, 2 score
    int   q0, q1;        // the instructions' tokens
    int   n_opt;
    int  *o0, *o1;       // each option's semantics tokens
    char **opt_id;       // option ids in encode order (choice: sorted)
} clef_question;

typedef struct {
    int32_t *ids;
    int      n_ids;
    clef_question *q;
    int      n_q;
    bool     state_truncated;   // the state was cut to fit max_length
} clef_record;

void clef_record_free(clef_record *r);

// encode_record(tokenizer, record, max_length) for a /v1/systemone body,
// without media. Returns false with *err (static text) on a request the
// reference would refuse or cannot be encoded.
bool clef_encode(tokenizer *tok, const jv *req, int max_length,
                 clef_record *out, const char **err);

// json.dumps(value, ensure_ascii=False, separators=(",", ":"),
// sort_keys=True) for a parsed value, the spelling encode_record's render()
// gives a non-string. Exposed for the conformance test.
void clef_py_dumps(const jv *v, sbuf *o);
// float.__repr__ of d (json.dumps writes floats with it).
void clef_py_float_repr(double d, char out[40]);

// The joint head over the backbone's final hidden states. `hidden` holds
// the record's rows after the backbone's output norm ([n_ids][hidden]);
// logits receive one value per option, questions in record order, options
// in encode order. Returns false on allocation failure.
bool clef_head_forward(model_t *m, const clef_record *r, const float *hidden,
                       float *logits);

// One /v1/systemone request to one response body appended to *out; returns
// the HTTP status (200, 400, 500) and, when not 200, *err names the
// problem. Shared by the server route and --systemone so both produce the
// same bytes for the same request. The slot's KV is overwritten; the
// caller resets its engine afterwards.
int clef_systemone_handle(engine *e, const jv *req, sbuf *out, const char **err);

#endif
