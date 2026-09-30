// Parallel slots share ONE tokenizer: server.c hands every slot the same
// `tokenizer *`, and each slot tokenizes its request on its own thread. The
// tokenizer is read-only after load except for one thing -- the per-call
// "a helper dropped a segment on OOM" flag, which lived in the shared struct.
// Every encode cleared it on entry and read it on exit, so two slots
// encoding at once raced on it (ThreadSanitizer, 2026-09-30, four parallel
// requests on test.gguf), and one slot's clear could erase another's OOM:
// a silently truncated prompt returned as a successful tokenization, which
// is exactly what the flag exists to prevent.
//
// Built with -fsanitize=thread by `make test-tokenizer-race`: TSan halting on
// a report is the gate, and the per-thread comparison against a
// single-threaded encode is the functional check that rides along.
#include "runner.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { THREADS = 4, ROUNDS = 200, CAP = 4096 };

static tokenizer g_tok;
static const char *TEXTS[THREADS] = {
    "The quick brown fox jumps over the lazy dog.",
    "hello there, general kenobi",
    "0123456789 numbers and words together",
    "<s>specials</s> in the middle of plain text",
};
static int32_t g_want[THREADS][CAP];
static int g_want_n[THREADS];

static void *worker(void *arg) {
    int k = (int)(intptr_t)arg;
    int32_t *got = malloc(sizeof(int32_t) * CAP);
    for (int r = 0; r < ROUNDS; r++) {
        int n = (r & 1) ? tok_encode(&g_tok, TEXTS[k], got, CAP, true, true)
                        : tok_encode_prompt(&g_tok, TEXTS[k], got, CAP, true);
        if (n != g_want_n[k] || memcmp(got, g_want[k], sizeof(int32_t) * (size_t)n)) {
            fprintf(stderr, "FAIL: thread %d round %d encoded differently\n", k, r);
            exit(1);
        }
    }
    free(got);
    return NULL;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test.gguf";
    gguf_file g;
    if (!gguf_open(&g, path) || !tokenizer_init(&g_tok, &g)) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }
    for (int k = 0; k < THREADS; k++) {
        // tok_encode with specials and tok_encode_prompt on unmarked text
        // agree (no marks: everything is caller text for the latter, and
        // these texts' specials are spelled in text either way)
        g_want_n[k] = tok_encode(&g_tok, TEXTS[k], g_want[k], CAP, true, true);
        int32_t alt[CAP];
        int na = tok_encode_prompt(&g_tok, TEXTS[k], alt, CAP, true);
        if (g_want_n[k] <= 0 || na != g_want_n[k] ||
            memcmp(alt, g_want[k], sizeof(int32_t) * (size_t)na)) {
            // the two entry points legitimately differ on a special; keep the
            // gate to the plain one for that text
            TEXTS[k] = "plain text only, nothing special here";
            g_want_n[k] = tok_encode(&g_tok, TEXTS[k], g_want[k], CAP, true, true);
        }
    }
    pthread_t th[THREADS];
    for (int k = 0; k < THREADS; k++)
        pthread_create(&th[k], NULL, worker, (void *)(intptr_t)k);
    for (int k = 0; k < THREADS; k++) pthread_join(th[k], NULL);
    tokenizer_free(&g_tok);
    gguf_close(&g);
    puts("tokenizer race tests ok");
    return 0;
}
