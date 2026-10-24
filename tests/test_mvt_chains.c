// The CPU trainer's transposed product (matvec_t) against its contract.
//
// dx[t][i] += sum_j W[j][i] * dy[t][j]: each dx element takes its rows in
// ascending order through fmaf, starting from its incoming value, and
// skips a row whose dy[t][j] is zero (of either sign); a row's weights are
// the dequantized block values times the tensor's scale. The reference
// below is that contract written as the plainest loop (one element at a
// time, a row decoded per use). matvec_t tiles the work (column units,
// position chunks, decoded row panels, register tiles of positions x
// columns) and must agree with it bit for bit on every weight type the
// trainer decodes, at the edges of every tile: widths that leave a partial
// unit, row counts around a panel, position counts around a tile and a
// chunk, a scaled tensor, zero rows, zero positions and signed zeros in dy.
//
// model.c is compiled into this test (tests/test_model_load_oom.c's
// pattern) to reach matvec_t.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/model.c"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}
static float frnd(void) { return (float)(int32_t)rnd32() / 2147483648.0f; }
static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static uint16_t get16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
// a finite block scale, either sign, 2^-10 .. 2^-4
static uint16_t rnd_scale(void) {
    float v = ldexpf(1.0f + (float)(rnd32() % 1024) / 1024.0f, -10 + (int)(rnd32() % 7));
    return f32_to_f16(rnd32() & 1 ? -v : v);
}

// tests/test_train_gpu.c's table: each type's f16 scale field offsets
static const struct { int type; const char *name; int f16[2]; } TYPES[] = {
    { T_F32, "f32", {-1, -1} },        { T_F16, "f16", {-1, -1} },
    { T_BF16, "bf16", {-1, -1} },      { T_Q8_0, "q8_0", {0, -1} },
    { T_Q4_0, "q4_0", {0, -1} },       { T_Q4_1, "q4_1", {0, 2} },
    { T_Q5_0, "q5_0", {0, -1} },       { T_Q5_1, "q5_1", {0, 2} },
    { T_Q2_K, "q2_K", {80, 82} },      { T_Q3_K, "q3_K", {108, -1} },
    { T_Q4_K, "q4_K", {0, 2} },        { T_Q5_K, "q5_K", {0, 2} },
    { T_Q6_K, "q6_K", {208, -1} },     { T_IQ4_NL, "iq4_nl", {0, -1} },
    { T_IQ4_XS, "iq4_xs", {0, -1} },   { T_IQ2_XXS, "iq2_xxs", {0, -1} },
    { T_IQ2_XS, "iq2_xs", {0, -1} },   { T_IQ2_S, "iq2_s", {0, -1} },
    { T_IQ3_XXS, "iq3_xxs", {0, -1} }, { T_IQ3_S, "iq3_s", {0, -1} },
    { T_IQ1_S, "iq1_s", {0, -1} },     { T_IQ1_M, "iq1_m", {-1, -1} },
    { T_MXFP4, "mxfp4", {-1, -1} },    { T_NVFP4, "nvfp4", {-1, -1} },
    { T_Q2_0, "q2_0", {0, -1} },
};

static uint8_t *make_rows(int type, const int *f16, int n_in, int n_out) {
    size_t rs = ggml_row_size(type, n_in), nbytes = rs * (size_t)n_out;
    uint8_t *w = malloc(nbytes);
    if (!w) exit(2);
    if (type == T_F32 || type == T_F16 || type == T_BF16) {
        for (size_t e = 0; e < (size_t)n_in * n_out; e++) {
            float v = frnd();
            if (type == T_F32) memcpy(w + 4 * e, &v, 4);
            else if (type == T_F16) put16(w + 2 * e, f32_to_f16(v));
            else { uint32_t u; memcpy(&u, &v, 4); put16(w + 2 * e, (uint16_t)(u >> 16)); }
        }
        return w;
    }
    for (size_t b = 0; b < nbytes; b++) w[b] = (uint8_t)rnd32();
    size_t ts = ggml_type_size(type), nblk = nbytes / ts;
    for (size_t k = 0; k < nblk; k++) {
        uint8_t *blk = w + k * ts;
        for (int f = 0; f < 2; f++)
            if (f16[f] >= 0) put16(blk + f16[f], rnd_scale());
        if (type == T_IQ1_M) {
            uint16_t h = rnd_scale();
            for (int s = 0; s < 4; s++) {
                uint16_t sw = get16(blk + 48 + 2 * s);
                sw = (uint16_t)((sw & 0x0FFF) | (((h >> (4 * s)) & 0xF) << 12));
                put16(blk + 48 + 2 * s, sw);
            }
        }
        if (type == T_MXFP4) blk[0] = (uint8_t)(118 + rnd32() % 16);
    }
    return w;
}

static void ref_mvt(const gguf_tensor *w, const float *dy, float *dx,
                    int n_in, int n_out, int nb) {
    float *row = malloc(sizeof(float) * (size_t)n_in);
    size_t rs = ggml_row_size(w->type, n_in);
    // j outermost per position: each dx[t][i] still sees its rows in
    // ascending order, one fmaf each
    for (int t = 0; t < nb; t++)
        for (int j = 0; j < n_out; j++) {
            float d = dy[(size_t)t * n_out + j];
            if (d == 0.0f) continue;
            dequant_row(w->type, (const uint8_t *)w->data + (size_t)j * rs, row, n_in);
            for (int i = 0; i < n_in; i++) {
                float wv = row[i];
                if (w->scale != 1.0f) wv *= w->scale;
                dx[(size_t)t * n_in + i] = fmaf(wv, d, dx[(size_t)t * n_in + i]);
            }
        }
    free(row);
}

static int check(model_t *m, int ti, int n_in, int n_out, int nb, float scale) {
    int type = TYPES[ti].type;
    gguf_tensor w;
    memset(&w, 0, sizeof w);
    w.type = type; w.scale = scale; w.ne[0] = n_in; w.ne[1] = n_out;
    w.data = make_rows(type, TYPES[ti].f16, n_in, n_out);
    float *dy = malloc(sizeof(float) * (size_t)nb * n_out);
    float *dx0 = malloc(sizeof(float) * (size_t)nb * n_in);
    float *dx1 = malloc(sizeof(float) * (size_t)nb * n_in);
    if (!dy || !dx0 || !dx1) exit(2);
    for (size_t e = 0; e < (size_t)nb * n_out; e++) {
        uint32_t r = rnd32() % 16;
        dy[e] = r == 0 ? 0.0f : r == 1 ? -0.0f : frnd();
    }
    // a row no position uses, and a position that uses no row (beside rows
    // and positions that are used, so the case still computes something)
    if (n_out > 2) for (int t = 0; t < nb; t++) dy[(size_t)t * n_out + (n_out / 2)] = 0.0f;
    if (nb > 2) for (int j = 0; j < n_out; j++) dy[(size_t)1 * n_out + j] = -0.0f;
    // incoming values with negative zeros among them: a skipped row leaves
    // -0 alone, an fmaf with a zero dy would turn it into +0
    for (size_t e = 0; e < (size_t)nb * n_in; e++)
        dx0[e] = dx1[e] = rnd32() % 8 == 0 ? -0.0f : frnd();
    ref_mvt(&w, dy, dx0, n_in, n_out, nb);
    bool ok = matvec_t(m, &w, dy, dx1, n_in, n_out, nb);
    int bad = !ok || memcmp(dx0, dx1, sizeof(float) * (size_t)nb * n_in) != 0;
    if (bad) {
        size_t e = 0;
        while (ok && e < (size_t)nb * n_in && !memcmp(&dx0[e], &dx1[e], sizeof(float))) e++;
        fprintf(stderr, "FAIL %s n_in %d n_out %d nb %d scale %g: %s at t %zu i %zu "
                "(ref %a got %a)\n", TYPES[ti].name, n_in, n_out, nb, scale,
                ok ? "differs" : "matvec_t failed", e / n_in, e % n_in,
                ok ? dx0[e] : 0.0f, ok ? dx1[e] : 0.0f);
    }
    free(w.data); free(dy); free(dx0); free(dx1);
    return bad;
}

int main(void) {
    f16_init();   // without it every f16 block scale decodes to 0
    static model_t m;
    memset(&m, 0, sizeof m);
    m.tp = tpool_create(3);
    // position counts around a register tile (4) and a chunk (32 a chunk,
    // at most 8 chunks); row counts around a panel; two widths a type
    static const int nbs[] = { 1, 3, 5, 16, 33, 70 };
    static const int nouts[] = { 1, 17, 130 };
    int n = 0, fails = 0;
    for (size_t ti = 0; ti < sizeof TYPES / sizeof *TYPES; ti++) {
        int bs = ggml_block_size(TYPES[ti].type);
        int widths[2] = { bs == 1 ? 77 : bs, bs == 1 ? 32 : 3 * bs };
        for (int a = 0; a < 2; a++)
            for (size_t b = 0; b < sizeof nouts / sizeof *nouts; b++)
                for (size_t c = 0; c < sizeof nbs / sizeof *nbs; c++) {
                    float scale = (a + b + c) % 3 == 0 ? 0.75f : 1.0f;
                    fails += check(&m, (int)ti, widths[a], nouts[b], nbs[c], scale);
                    n++;
                }
    }
    // the shapes training runs: a 256-position window and a head round
    fails += check(&m, 3, 1024, 300, 256, 1.0f);
    fails += check(&m, 10, 512, 200, 16, 1.0f);
    n += 2;
    tpool_destroy(m.tp);
    if (fails) {
        fprintf(stderr, "mvt chains: %d of %d cases differ\n", fails, n);
        return 1;
    }
    fprintf(stderr, "mvt chains: %d cases bit-identical to the contract, %zu weight types\n",
            n, sizeof TYPES / sizeof *TYPES);
    return 0;
}
