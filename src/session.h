// Session images (R1.3.1-.3): a generation suspended to a file and resumed
// exactly, or forked.
//
// An image is one file, bound end to end by a SHA-256 over every byte before
// the trailer:
//
//   "runner.session.1"            16-byte magic
//   u32 header length, header     JSON, xyntetik.runner.session.v1: the model
//                                 (sha256, engine model key), context length,
//                                 KV type, prompt length, token count, the
//                                 generation budget and how much of it is
//                                 spent, the sampler (knobs and rng state;
//                                 the rng is "0" for a greedy run, which
//                                 never draws from it),
//                                 the constraint (json mode, schema digest),
//                                 ignore_eos, the runner version and binary
//   u32 n, int32 tokens[n]        the prompt and what was generated
//   u64 len, state bytes          KV of [0, n) and the recurrent fold at n,
//                                 the prefix cache's entry layout
//   u32 n_vocab, f32 logits[]     the next token's logits
//   32-byte SHA-256 trailer
//
// There is no timestamp in it: the same state is the same bytes, so an image
// written at the end of an uninterrupted run and one written after a
// suspend and resume can be compared byte for byte -- which is the gate.
// Everything the state does not hold (the penalty window, the constraint
// validator, the reasoning tracker, the counts) is rebuilt on resume by
// replaying the generated tokens through the step's own bookkeeping
// (engine_gen_resume).
#ifndef RUNNER_SESSION_H
#define RUNNER_SESSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "engine.h"

typedef struct {
    char     model_sha256[65];
    uint64_t model_key;
    int      n_ctx;
    char     kv_type[8];
    int      n_prompt, n_tokens;   // n_tokens = prompt + generated so far
    int      max_new, generated;
    float    temp, top_p, min_p, repeat_penalty;
    int      top_k;
    uint64_t rng;
    bool     json_mode, ignore_eos;
    char     schema_sha256[65];    // "" when no schema
    char     binary_sha256[65];
} session_meta;

// Writes the image of engine `e` (at pos = n_tokens, the last token already
// forwarded) with the next token's `logits`. Never overwrites a file; the
// image's own sha256 is copied to sha_out. False with the reason on stderr.
bool session_write(const char *path, const engine *e, const session_meta *meta,
                   const float *logits, int n_vocab, char sha_out[65]);

// Reads and checks an image (magic, trailer digest, schema, sizes). On
// success the caller owns *tokens, *state and *logits (free()).
typedef struct {
    session_meta meta;
    int32_t *tokens;
    uint8_t *state;
    size_t   state_n;
    float   *logits;
    int      n_vocab;
    char     sha256[65];
} session_image;
bool session_read(const char *path, session_image *img);
void session_image_free(session_image *img);

#endif
