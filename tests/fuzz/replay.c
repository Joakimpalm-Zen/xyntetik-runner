// Corpus replay for hosts whose clang ships sanitizers but no libFuzzer.
// The same target entry point runs under ASan/UBSan; this is not a substitute
// for the coverage-guided mutation jobs on Linux.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
__attribute__((weak)) int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv; return 0;
}
int LLVMFuzzerTestOneInput(const uint8_t *, size_t);
int main(int argc, char **argv) {
    LLVMFuzzerInitialize(&argc, &argv);
    if (argc < 2) return 2;
    for (int i = 1; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb"); assert(f);
        assert(fseek(f, 0, SEEK_END) == 0);
        long n = ftell(f); assert(n >= 0 && n <= 4 * 1024 * 1024);
        rewind(f);
        uint8_t *data = malloc((size_t)n + 1); assert(data);
        assert(fread(data, 1, (size_t)n, f) == (size_t)n); fclose(f);
        LLVMFuzzerTestOneInput(data, (size_t)n);
        free(data);
    }
    fprintf(stderr, "replayed %d corpus inputs\n", argc - 1);
    return 0;
}
