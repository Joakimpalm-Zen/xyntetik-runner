// The decision record's legal mass, on fixture logits.
//
// A choice_logprobs point renormalizes over the legal probed candidates, so
// its posteriors read the same whether the legal set held all of the model's
// probability or almost none of it. A forced choice between two strings the
// model barely considers (the Windows CUDA host, 2026-10-23: Qwen3-4B after
// writing `"E` under an enum holding "E" and "EAT", 1e-4 of the raw mass on
// the legal continuations, the rest on a continuation the enum did not hold)
// then looks like a confident 0.84. The number that tells the two apart is
// the mass the legal set held before renormalization, and the point carries
// it as legal_mass.
//
// The fixture: one decision step on hand-written logits under the enum
// [1, 2]. Every entry is -1e30 except an illegal token at logit 0 and the two
// legal ones at ln(5e-5) each, so by hand the whole-vocabulary softmax gives
// the legal set 1e-4 / (1 + 1e-4) = 9.99900010e-5 (the absolute anchor), each
// legal token a posterior of exactly 1/2, and the probed set a coverage of 1.
#include "runner.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_path = "test.gguf";
static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
    else        fprintf(stderr, "ok: %s\n", what);
}

static int sink_cb(void *ud, const char *bytes, int n) {
    (void)ud; (void)bytes; (void)n;
    return 0;
}

// the id of the single token `s` encodes to, or -1
static int one_token(tokenizer *t, const char *s) {
    int32_t ids[8];
    int n = tok_encode_raw(t, s, (int)strlen(s), ids, 8);
    return n == 1 ? ids[0] : -1;
}

int main(int argc, char **argv) {
    if (argc > 1) g_path = argv[1];
    model_params p;
    memset(&p, 0, sizeof(p));
    p.gpu_mode = GPU_OFF;
    p.n_threads = 1;
    p.n_ctx = 64;
    p.n_batch = 8;
    model_t m;
    tokenizer tok;
    sampler smp;
    engine e;
    memset(&m, 0, sizeof(m)); memset(&tok, 0, sizeof(tok));
    memset(&smp, 0, sizeof(smp)); memset(&e, 0, sizeof(e));
    if (!model_load(&m, g_path, &p) || !tokenizer_init(&tok, &m.gf)) {
        fprintf(stderr, "FAIL: load %s\n", g_path);
        return 1;
    }
    smp.temp = 0;
    smp.repeat_penalty = 1.0f;
    smp.rng = 1;
    engine_init(&e, &m, &tok, &smp);

    const char *src = "{\"enum\":[1,2]}";
    jv *j = json_parse(src, (int)strlen(src));
    char err[128];
    snode *schema = j ? schema_compile(j, err, sizeof(err)) : NULL;
    jv_free(j);
    ck(schema != NULL, "the enum schema compiles");

    int one = one_token(&tok, "1"), two = one_token(&tok, "2");
    int bad = one_token(&tok, "a");
    ck(one >= 0 && two >= 0 && bad >= 0 && one != two,
       "the fixture's three tokens are single vocabulary entries");
    if (!schema || one < 0 || two < 0 || bad < 0) return 1;

    cl_rec recs[4];
    memset(recs, 0, sizeof(recs));
    e.schema = schema;
    e.cl_recs = recs;
    e.cl_cap = 4;
    e.cl_probe = 32;
    engine_reset(&e);
    int32_t prompt[4] = { 1, 20, 30, 40 };
    ck(engine_feed(&e, prompt, 4) != NULL, "the prompt feeds");

    int V = m.n_vocab;
    float *fx = malloc(sizeof(float) * (size_t)V);
    if (!fx) return 1;
    for (int i = 0; i < V; i++) fx[i] = -1e30f;
    fx[bad] = 0.0f;
    fx[one] = fx[two] = logf(5e-5f);

    engine_gen_begin(&e, 1);
    int32_t next = -1;
    int pos = -1;
    engine_gen_step(&e, fx, sink_cb, NULL, &next, &pos);
    // the step completes the document, so it hands no next token out and
    // claims no row; the token it took is in the history at its position
    int chosen = e.hist ? e.hist[e.gen_start] : -1;
    ck(chosen == one || chosen == two, "the sampler picks a legal token");
    ck(e.cl_count == 1, "the step is one decision point");
    if (e.cl_count == 1) {
        const cl_rec *r = &recs[0];
        const double want = 1e-4 / (1.0 + 1e-4);
        fprintf(stderr, "   legal_mass %.9g (by hand %.9g), coverage %.9g\n",
                (double)r->legal_mass, want, (double)r->coverage);
        ck(r->n_legal >= 2, "both legal tokens are counted");
        ck(fabs(r->legal_mass - want) <= 1e-5 * want,
           "legal_mass is the legal set's raw mass, 1e-4 / (1 + 1e-4)");
        ck(fabs(r->coverage - 1.0) <= 1e-6,
           "coverage is the probed set's mass, which held all of it");
        ck(fabs(r->prob[0] - 0.5) <= 1e-6 && fabs(r->prob[1] - 0.5) <= 1e-6,
           "the posterior still renormalizes over the legal set");
    }
    engine_gen_end(&e, sink_cb, NULL, NULL);

    free(fx);
    free(e.hist);
    schema_free(schema);
    tokenizer_free(&tok);
    model_free(&m);
    if (!g_fail) fprintf(stderr, "all choice-mass tests passed\n");
    return g_fail;
}
