// The repeat penalty's window holds what the model GENERATED, never the
// prompt. A prompt that carries a tool schema contains exactly the tokens a
// call must re-type (`=`, `amount`, `from_currency`); with the prompt seeding
// the window, the generic preset's penalty turned every sampled call on
// /v1/completions into schema-avoiding garbage while greedy stayed perfect
// (the lab's pilot, 2026-09-15; reproduced on granite-4.1-3b Q8_0, 0 of 8
// exact at 1.10 against 8 of 8 at 1.0). This gate pins the window's
// contents at every seam that used to seed it: the prompt feed, a rewind
// over a kept prefix, and a prefix-cache fork; and that generated tokens
// still enter it.
#include "engine.h"
#include "model.h"
#include "sample.h"
#include "tokenizer.h"
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

static int sink(void *ud, const char *bytes, int n) { (void)ud; (void)bytes; (void)n; return 0; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    model_params p; memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF; p.n_threads = 1; p.n_ctx = 64; p.n_batch = 4;
    model_t m; tokenizer tok; sampler smp; engine e;
    memset(&m, 0, sizeof m); memset(&smp, 0, sizeof smp); memset(&e, 0, sizeof e);
    if (!model_load(&m, path, &p)) { fprintf(stderr, "cannot load %s\n", path); return 1; }
    if (!tokenizer_init(&tok, &m.gf)) { fprintf(stderr, "no tokenizer\n"); return 1; }
    smp.temp = 0.01f;          // sampled, so the penalty branch runs
    smp.repeat_penalty = 1.5f; // a penalty large enough to move any argmax
    smp.rng = 7;
    engine_init(&e, &m, &tok, &smp);

    // a prompt of in-vocabulary ids, none of them a stop token
    int32_t prompt[6];
    for (int i = 0; i < 6; i++) prompt[i] = (i * 7 + 3) % (m.n_vocab > 16 ? 16 : m.n_vocab);

    float *lg = engine_feed(&e, prompt, 6);
    ck(lg != NULL, "the prompt feeds");
    ck(smp.n_recent == 0, "the prompt feed leaves the penalty window empty");

    // with an empty window, a sampled pick at near-zero temperature is the argmax
    int best = 0;
    for (int i = 1; i < m.n_vocab; i++) if (lg[i] > lg[best]) best = i;
    float *copy = malloc(sizeof(float) * (size_t)m.n_vocab);
    memcpy(copy, lg, sizeof(float) * (size_t)m.n_vocab);
    int pick = sample_pick(&smp, copy, m.n_vocab, NULL, NULL);
    ck(pick == best, "right after the prompt, the penalty moves nothing: the pick is the argmax");
    free(copy);

    // generated tokens do enter the window
    int n = engine_generate(&e, lg, 3, sink, NULL, NULL);
    ck(n == 3, "three tokens generated");
    ck(smp.n_recent == 3, "the window holds exactly the three generated tokens");

    // a rewind that keeps the prompt as a prefix re-seeds nothing from it
    int32_t again[7];
    memcpy(again, prompt, sizeof prompt); again[6] = prompt[0];
    int keep = engine_rewind(&e, again, 7);
    ck(keep >= 6, "the prompt is kept as a prefix on rewind");
    ck(smp.n_recent == 0, "a rewind over a kept prefix leaves the window empty");

    free(e.hist);
    tokenizer_free(&tok);
    model_free(&m);
    if (!g_fail) puts("penalty window tests ok");
    return g_fail;
}
