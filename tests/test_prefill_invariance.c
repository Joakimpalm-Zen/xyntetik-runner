// Batch-invariant prefill (R1.5): a token's numbers must not depend on how
// many tokens were prefilled with it.
//
// The solo path -- one token per forward, which is what decode runs -- is
// the reference. The same prompt is then prefilled in chunks of every width
// the batched kernels tile differently (1, 2, 3, 5, 8, 16 and the whole
// prompt at once, the prompt 37 tokens long so every chunking ends in a
// ragged tail), and two things are compared BYTE FOR BYTE with the solo run:
//
//   - the logits of the last prompt token (what the first sampled token is
//     drawn from), and
//   - the logits of one decode step after the prefill, which reads every KV
//     row the prefill wrote -- so a KV row that differs anywhere shows up
//     even when the last row happens to agree.
//
// A receipt records the prompt's token ids, and --verify replays them in one
// batch; a served request may have built the same KV from a cached prefix
// plus a chunk, or a chunk at a time under load. Bit-identity across
// chunkings is what makes those replay exactly, so this is a gate, not a
// tolerance: one differing bit fails.
//
// Usage: test-prefill-invariance [model.gguf ...] (default: test.gguf)
#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_PROMPT = 37 };

static int check_model(const char *path) {
    model_params p;
    memset(&p, 0, sizeof p);
    p.gpu_mode = GPU_OFF;
    p.n_ctx = 128;
    model_t m;
    if (!model_load(&m, path, &p)) {
        fprintf(stderr, "prefill-invariance: cannot load %s\n", path);
        return 1;
    }
    const int nv = m.n_vocab;
    int32_t toks[N_PROMPT + 1];
    for (int i = 0; i < N_PROMPT + 1; i++) toks[i] = (int32_t)(3 + (i * 37 + 11) % (nv - 3));
    float *ref_last = malloc(sizeof(float) * (size_t)nv);
    float *ref_next = malloc(sizeof(float) * (size_t)nv);
    if (!ref_last || !ref_next) return 1;

    // the solo reference: one token per forward; each position's final
    // hidden state is kept so a failure can say which rows moved
    const int ne = m.n_embd;
    float *ref_x = malloc(sizeof(float) * (size_t)ne * N_PROMPT);
    if (!ref_x) return 1;
    float *lg = NULL;
    for (int t = 0; t < N_PROMPT; t++) {
        lg = model_forward(&m, toks[t], t);
        memcpy(ref_x + (size_t)t * ne, m.x, sizeof(float) * (size_t)ne);
    }
    if (!lg) { fprintf(stderr, "prefill-invariance: solo forward failed\n"); return 1; }
    memcpy(ref_last, lg, sizeof(float) * (size_t)nv);
    lg = model_forward(&m, toks[N_PROMPT], N_PROMPT);
    memcpy(ref_next, lg, sizeof(float) * (size_t)nv);

    static const int CHUNKS[] = { 1, 2, 3, 5, 8, 16, 17, 20, 24, 31, 32, 33, 36, N_PROMPT };
    int bad = 0;
    for (size_t c = 0; c < sizeof CHUNKS / sizeof CHUNKS[0]; c++) {
        int w = CHUNKS[c];
        lg = NULL;
        for (int at = 0; at < N_PROMPT; at += w) {
            int k = N_PROMPT - at < w ? N_PROMPT - at : w;
            lg = model_forward_batch(&m, toks + at, k, at, at + k == N_PROMPT);
        }
        if (!lg) { fprintf(stderr, "prefill-invariance: batch forward failed\n"); return 1; }
        int d_last = 0;
        for (int i = 0; i < nv; i++)
            if (memcmp(&lg[i], &ref_last[i], sizeof(float))) d_last++;
        // the last chunk's rows are still in m.x: which ones moved
        int last_at = (N_PROMPT - 1) / w * w, rows = 0, first_row = -1;
        for (int r = last_at; r < N_PROMPT; r++)
            if (memcmp(m.x + (size_t)(r - last_at) * ne, ref_x + (size_t)r * ne,
                       sizeof(float) * (size_t)ne)) {
                rows++;
                if (first_row < 0) first_row = r - last_at;
            }
        float first_last_got = 0, first_last_want = 0;
        for (int i = 0; i < nv; i++)
            if (memcmp(&lg[i], &ref_last[i], sizeof(float))) {
                first_last_got = lg[i]; first_last_want = ref_last[i]; break;
            }
        lg = model_forward(&m, toks[N_PROMPT], N_PROMPT);
        int d_next = 0;
        for (int i = 0; i < nv; i++)
            if (memcmp(&lg[i], &ref_next[i], sizeof(float))) d_next++;
        printf("prefill-invariance: %s chunk %2d: last-token logits %d/%d differ, "
               "next-step logits %d/%d differ", path, w, d_last, nv, d_next, nv);
        if (d_last) printf(" (e.g. %.9g vs solo %.9g)", first_last_got, first_last_want);
        if (rows) printf("; %d of the last chunk's %d rows moved, first at row %d",
                         rows, N_PROMPT - last_at, first_row);
        printf("\n");
        if (d_last || d_next) bad = 1;
    }
    free(ref_last);
    free(ref_next);
    free(ref_x);
    model_free(&m);
    return bad;
}

int main(int argc, char **argv) {
    int rc = 0;
    if (argc < 2) rc |= check_model("test.gguf");
    for (int i = 1; i < argc; i++) rc |= check_model(argv[i]);
    printf(rc ? "prefill-invariance: FAIL\n" : "prefill-invariance: ok\n");
    return rc;
}
