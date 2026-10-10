// Fixed real vocabularies, arbitrary bytes/capacities, and the independent
// linear/heap BPE merge paths. No model weights or device are needed.
#include "tokenizer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static tokenizer tok[3];
static gguf_file files[3];
static void cleanup(void) {
    for (int i = 0; i < 3; i++) { tokenizer_free(&tok[i]); gguf_close(&files[i]); }
}
int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv;
    const char *paths[] = {"tests/fixtures/vocab-spm.gguf",
                          "tests/fixtures/vocab-bpe-qwen2.gguf",
                          "tests/fixtures/vocab-bpe-spm-gemma4.gguf"};
    for (int i = 0; i < 3; i++)
        if (!gguf_open(&files[i], paths[i]) || !tokenizer_init(&tok[i], &files[i])) abort();
    atexit(cleanup);
    return 0;
}
int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
    if (size < 2 || size > 8192) return 0;
    tokenizer *t = &tok[data[0] % 3];
    int cap = data[1];
    char *text = malloc(size);
    int32_t *a = malloc(sizeof(int32_t) * (size_t)(cap + 1));
    int32_t *b = malloc(sizeof(int32_t) * (size_t)(cap + 1));
    assert(text && a && b);
    memcpy(text, data + 2, size - 2); text[size - 2] = 0;
    a[cap] = b[cap] = 0x12345678;
    tok_merge_force(0);
    int na = tok_encode(t, text, a, cap, false, data[0] & 4);
    tok_merge_force(1);
    int nb = tok_encode(t, text, b, cap, false, data[0] & 4);
    tok_merge_force(-1);
    assert(na >= 0 && na <= cap && na == nb);
    assert(!memcmp(a, b, (size_t)na * sizeof(*a)));
    assert(a[cap] == 0x12345678 && b[cap] == 0x12345678);
    char out[33];
    for (int i = 0; i < na; i++) {
        assert(a[i] >= 0 && a[i] < t->n_vocab);
        assert(tok_decode(t, a[i], out, sizeof out) <= (int)sizeof out);
    }
    (void)tok_encode_prompt(t, text, a, cap, false);
    (void)tok_encode_raw(t, text, (int)size - 2, b, cap);
    assert(a[cap] == 0x12345678 && b[cap] == 0x12345678);
    free(text); free(a); free(b);
    return 0;
}
