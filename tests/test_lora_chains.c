// The adapter delta (lora_apply) against the loop it replaced.
//
// lora_apply computes y += scale * B (A x) for every row with a fixed chain
// per element: t[k] sums A[k][i] x[i] over i ascending from 0 through fmaf,
// then y[j] sums B[j][k] t[k] over k ascending from 0 and adds scale times
// that with one fmaf. The engine runs up to 8 of those chains side by side,
// in blocks of 8, 4, 2 and 1, so every rank and output count lands in some
// mix of full and partial blocks. The reference below is the serial loop the
// engine shipped before (one chain at a time), which is the specification of
// the order; the two must agree bit for bit, serial and on the pool, at
// every block edge, with the float range's extremes in the inputs.
//
// model.c is compiled into this test (tests/test_model_load_oom.c's
// pattern) to reach lora_apply.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/model.c"

static void ref_apply(const struct lora_w *lw, float *y, int ys,
                      const float *xin, int xs, int n_in, int n_out, int nb) {
    float t[LORA_R_MAX];
    for (int bb = 0; bb < nb; bb++) {
        const float *xr = xin + (size_t)bb * xs;
        float *yr = y + (size_t)bb * ys;
        for (int k = 0; k < lw->r; k++) {
            float acc = 0.0f;
            for (int i = 0; i < n_in; i++)
                acc = fmaf(lw->a[(size_t)k * n_in + i], xr[i], acc);
            t[k] = acc;
        }
        for (int j = 0; j < n_out; j++) {
            float acc = 0.0f;
            for (int k = 0; k < lw->r; k++)
                acc = fmaf(lw->b[(size_t)j * lw->r + k], t[k], acc);
            yr[j] = fmaf(lw->scale, acc, yr[j]);
        }
    }
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float rndf(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)(int32_t)(uint32_t)(rng >> 32) / 2147483648.0f;
}

// Values a chain must carry exactly: the largest and smallest normal
// floats, a subnormal, zeros of both signs, and the exact halves that round
// to even, mixed into the random fill.
static const float edge[] = { 3.4028235e38f, -3.4028235e38f, 1.1754944e-38f,
                              1.4e-45f, 0.0f, -0.0f, 0.5f, -1.5f, 1.0f };

static void fill(float *v, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) v[i] = rndf() * scale;
    for (size_t e = 0; e < sizeof edge / sizeof *edge; e++)
        v[(e * 7 + 3) % n] = edge[e];
}

static int check(tpool *tp, int n_in, int n_out, int r, int nb, int xs_pad) {
    int xs = n_in + xs_pad, ys = n_out + xs_pad;
    struct lora_w lw;
    memset(&lw, 0, sizeof lw);
    lw.r = r;
    lw.scale = 0.37f;
    lw.a = malloc(sizeof(float) * (size_t)r * n_in);
    lw.b = malloc(sizeof(float) * (size_t)n_out * r);
    float *x = malloc(sizeof(float) * (size_t)nb * xs);
    float *y0 = malloc(sizeof(float) * (size_t)nb * ys);
    float *y1 = malloc(sizeof(float) * (size_t)nb * ys);
    float *y2 = malloc(sizeof(float) * (size_t)nb * ys);
    if (!lw.a || !lw.b || !x || !y0 || !y1 || !y2) { fprintf(stderr, "oom\n"); exit(2); }
    // scales that keep the extremes' products finite on most chains and
    // overflow on a few, which must overflow the same way on both paths
    fill(lw.a, (size_t)r * n_in, 0.25f);
    fill(lw.b, (size_t)n_out * r, 0.5f);
    fill(x, (size_t)nb * xs, 1.0f);
    fill(y0, (size_t)nb * ys, 1.0f);
    memcpy(y1, y0, sizeof(float) * (size_t)nb * ys);
    memcpy(y2, y0, sizeof(float) * (size_t)nb * ys);
    ref_apply(&lw, y0, ys, x, xs, n_in, n_out, nb);
    lora_apply(&lw, y1, ys, x, xs, n_in, n_out, nb);
    lora_apply_tp(tp, &lw, y2, ys, x, xs, n_in, n_out, nb);
    int bad = 0;
    for (int bb = 0; bb < nb; bb++)
        for (int j = 0; j < ys; j++) {
            size_t o = (size_t)bb * ys + j;
            if (memcmp(&y0[o], &y1[o], sizeof(float)) || memcmp(&y0[o], &y2[o], sizeof(float))) {
                if (!bad)
                    fprintf(stderr, "FAIL n_in %d n_out %d r %d nb %d: row %d col %d "
                            "ref %a serial %a pool %a\n", n_in, n_out, r, nb, bb, j,
                            y0[o], y1[o], y2[o]);
                bad = 1;
            }
        }
    free(lw.a); free(lw.b); free(x); free(y0); free(y1); free(y2);
    return bad;
}

int main(void) {
    tpool *tp = tpool_create(3);
    // every rank to 17 (each mix of 8, 4, 2 and 1 blocks), wider ranks, and
    // the cap; widths around every block edge
    static const int ranks[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 15, 16, 17, 33, 64 };
    static const int widths[] = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 77 };
    int n = 0, fails = 0;
    for (size_t a = 0; a < sizeof ranks / sizeof *ranks; a++)
        for (size_t b = 0; b < sizeof widths / sizeof *widths; b++)
            for (size_t c = 0; c < sizeof widths / sizeof *widths; c += 3) {
                int nb = 1 + (int)((a + b + c) % 5);
                fails += check(tp, widths[c], widths[b], ranks[a], nb, (int)(b % 2));
                n++;
            }
    // real projection shapes, decode (1 row) and a prefill window
    fails += check(tp, 1024, 3072, 8, 1, 0);
    fails += check(tp, 3072, 1024, 16, 4, 0);
    fails += check(tp, 2048, 1024, LORA_R_MAX, 2, 0);
    n += 3;
    tpool_destroy(tp);
    if (fails) {
        fprintf(stderr, "lora chains: %d of %d shapes differ\n", fails, n);
        return 1;
    }
    fprintf(stderr, "lora chains: %d shapes bit-identical to the serial loop\n", n);
    return 0;
}
