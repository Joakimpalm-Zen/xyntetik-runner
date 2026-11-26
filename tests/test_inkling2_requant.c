// R4.48.8.1: the rewrite's decode of an inkling2_loop expert stack (src/quantize.c expert_row_dequant: what
// --quant-error costs and a re-quantization writes from) against the loader's reference route (src/inkling2_gguf.c
// ink2g_fill_expert), every row of every expert of every stack. Both take (code * block_scale) * f[e], the same two
// roundings in the same order. The file (tests/inkling2_tiny.py) has varied UE4M3 scales and factors that are not
// powers of two, so a different order rounds differently: the other association, code * (block_scale * f[e]), is
// counted and must differ somewhere, or this check could not tell the orders apart.
// The one bit difference allowed is the sign of a zero: the loader's E2M1 table holds -0 for code 8 and the decode's
// doubled-integer table 0, and the two quantize alike.
// usage: test_inkling2_requant TINY.gguf
#include "../src/quantize.c"
#include "inkling2_gguf.h"

static float ue4m3(unsigned x) {
    const unsigned e = (x >> 3) & 15, m = x & 7;
    return e ? ldexpf(1.0f + (float)m / 8.0f, (int)e - 7) : ldexpf((float)m, -9);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s TINY.gguf\n", argv[0]); return 2; }
    f16_init();
    ink2_shape s;
    char why[512];
    ink2g *g = ink2g_open(argv[1], &s, why, sizeof why);
    if (!g) { fprintf(stderr, "FAIL: the loader refused %s: %s\n", argv[1], why); return 1; }
    gguf_file f;
    if (!gguf_open(&f, argv[1])) { fprintf(stderr, "FAIL: cannot open %s\n", argv[1]); return 1; }
    long long compared = 0, bad = 0, other_order = 0, folded = 0;
    for (int b = 0; b < s.n_block; b++)
        for (int which = 0; which < 2; which++) {
            char gn[96], rn[96];
            snprintf(gn, sizeof gn, "blk.%d.ffn_%s_exps.weight", b, which ? "down" : "gate_up");
            snprintf(rn, sizeof rn, "model.layers.%d.mlp.experts.%s", b, which ? "down_proj" : "gate_up_proj");
            gguf_tensor *t = gguf_find_tensor(&f, gn);
            const float *fac = t ? expert_factors(&f, t, NULL) : NULL;
            if (!t || t->type != T_NVFP4 || !fac) { fprintf(stderr, "FAIL: %s is not an NVFP4 stack with factors\n", gn); return 1; }
            const int64_t C = (int64_t)t->ne[0], R = (int64_t)t->ne[1];
            float *ref = malloc(sizeof(float) * (size_t)(R * C)), *row = malloc(sizeof(float) * (size_t)C);
            if (!ref || !row) return 1;
            const size_t rs = ggml_row_size(t->type, t->ne[0]);
            for (int e = 0; e < s.n_expert; e++) {
                int64_t rr, cc;
                if (ink2g_fill_expert(g, rn, e, ref, &rr, &cc) || rr != R || cc != C) {
                    fprintf(stderr, "FAIL: the reference route gave no %s expert %d\n", rn, e);
                    return 1;
                }
                folded += fac[e] != 1.0f;
                for (int64_t r = 0; r < R; r++) {
                    const uint64_t gr = (uint64_t)e * (uint64_t)R + (uint64_t)r;
                    expert_row_dequant(t, fac, gr, row);
                    for (int64_t c = 0; c < C; c++) {
                        const float want = ref[r * C + c];
                        uint32_t a, w;
                        memcpy(&a, &row[c], 4);
                        memcpy(&w, &want, 4);
                        compared++;
                        if (a != w && !(row[c] == 0.0f && want == 0.0f) && bad++ < 5)
                            fprintf(stderr, "FAIL: %s expert %d row %lld col %lld: %.9g, the loader %.9g\n", gn, e,
                                    (long long)r, (long long)c, (double)row[c], (double)want);
                    }
                }
            }
            // the other association over this stack: code x (scale x factor), code read back from the plain decode
            float *plain = malloc(sizeof(float) * (size_t)C);
            if (!plain) return 1;
            for (int e = 0; e < s.n_expert; e++)
                for (int64_t r = 0; r < R; r++) {
                    const uint64_t gr = (uint64_t)e * (uint64_t)R + (uint64_t)r;
                    const unsigned char *blk = (const unsigned char *)t->data + gr * rs;
                    dequant_row(t->type, blk, plain, (int)C);
                    expert_row_dequant(t, fac, gr, row);
                    for (int64_t c = 0; c < C; c++) {
                        const float bs = ue4m3(blk[(c / 64) * 36 + (c % 64) / 16]);
                        const float code = plain[c] / bs;   // exact: plain is code x bs, the code an E2M1 value
                        const float alt = code * (bs * fac[e]);
                        other_order += alt != row[c];
                    }
                }
            free(plain);
            free(ref);
            free(row);
        }
    gguf_close(&f);
    ink2g_close(g);
    if (!folded) { fprintf(stderr, "FAIL: every factor is 1: nothing was folded\n"); return 1; }
    if (!other_order) {
        fprintf(stderr, "FAIL: the other association rounds alike everywhere on this file: the check cannot tell the orders apart\n");
        return 1;
    }
    if (bad) { fprintf(stderr, "FAIL: %lld of %lld values differ from the loader's\n", bad, compared); return 1; }
    printf("ok: %lld values, every stack and expert, equal to the reference route's (the other association differs on %lld)\n",
           compared, other_order);
    return 0;
}
