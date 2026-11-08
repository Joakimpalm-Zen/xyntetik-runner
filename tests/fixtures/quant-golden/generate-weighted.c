// Oracle harness for unmodified ggml-quants.c at llama.cpp
// c1d0e7a004015f23bc0233470b747b596f29b264. See README.md for the build.
// Quantization is entirely ggml's; these stubs supply its four link dependencies.
#include "ggml-quants.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { COLS = 1536, REAL_ROWS = 16, EDGE_ROWS = 8 };

void ggml_abort(const char *file, int line, const char *fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args; va_start(args, fmt); vfprintf(stderr, fmt, args); va_end(args);
    abort();
}

const char *ggml_type_name(enum ggml_type type) { (void)type; return "golden-only"; }

size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        default: abort();
    }
}

size_t ggml_row_size(enum ggml_type type, int64_t n) {
    return (size_t)n / (type == GGML_TYPE_Q5_1 || type == GGML_TYPE_IQ4_NL ? 32 : 256)
           * ggml_type_size(type);
}

static void read_bytes(const char *dir, const char *name, void *data, size_t n) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb"); assert(f);
    assert(fread(data, 1, n, f) == n && fgetc(f) == EOF); fclose(f);
}

static void write_bytes(const char *dir, const char *name, const void *data, size_t n) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb"); assert(f);
    assert(fwrite(data, 1, n, f) == n); assert(fclose(f) == 0);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const char *dir = argv[1];
    uint16_t endian = 1; assert(*(uint8_t *)&endian == 1);
    static uint16_t bf[REAL_ROWS * COLS];
    static float real[REAL_ROWS * COLS], edge[EDGE_ROWS * COLS], weights[COLS];
    read_bytes(dir, "smollm2-135m-l0-ffn_down-16rows.bf16", bf, sizeof bf);
    for (int i = 0; i < REAL_ROWS * COLS; i++) {
        uint32_t bits = (uint32_t)bf[i] << 16; memcpy(&real[i], &bits, 4);
    }
    for (int c = 0; c < COLS; c++) {
        weights[c] = c % 64 < 16 ? 0.0f : (c + 5 * (c / 256)) % 16 < 3 ? 50.0f : 1.0f;
        int v = (c * 37) % 257 - 128;
        edge[0 * COLS + c] = 0;
        edge[1 * COLS + c] = 1;
        edge[2 * COLS + c] = -1;
        edge[3 * COLS + c] = (c % 2 ? -128 : 127) * 0x1p-4f;
        edge[4 * COLS + c] = v * 0x1p-40f;
        edge[5 * COLS + c] = v * 0x1p7f;
        edge[6 * COLS + c] = c % 256 < 32 ? 0 : v * (c % 256 < 128 ? 0x1p-8f : 0x1p3f);
        edge[7 * COLS + c] = c % 257 == 0 ? 64 : (c % 65 - 32) * 0.5f;
    }
    write_bytes(dir, "weighted-edges.f32", edge, sizeof edge);
    static const struct {
        const char *name, *old_pin;
        enum ggml_type type;
        size_t (*write)(const float *, void *, int64_t, int64_t, const float *);
    } types[] = {
        {"q4_k", "ggml-b10353-q4_k.bin", GGML_TYPE_Q4_K, quantize_q4_K},
        {"q5_k", "ggml-c1d0e7a-q5_k.bin", GGML_TYPE_Q5_K, quantize_q5_K},
        {"q6_k", "ggml-b10353-q6_k.bin", GGML_TYPE_Q6_K, quantize_q6_K},
        {"q5_1", "ggml-c1d0e7a-q5_1.bin", GGML_TYPE_Q5_1, quantize_q5_1},
        {"iq4_nl", "ggml-c1d0e7a-iq4_nl.bin", GGML_TYPE_IQ4_NL, quantize_iq4_nl},
    };
    for (size_t t = 0; t < sizeof types / sizeof *types; t++) {
        size_t n = ggml_row_size(types[t].type, COLS) * REAL_ROWS;
        uint8_t *out = calloc(1, n), *old = malloc(n); assert(out && old);
        assert(types[t].write(real, out, REAL_ROWS, COLS, NULL) == n);
        read_bytes(dir, types[t].old_pin, old, n);
        assert(!memcmp(out, old, n)); // verify this oracle against existing unweighted pins
        assert(types[t].write(real, out, REAL_ROWS, COLS, weights) == n);
        assert(memcmp(out, old, n)); // the skewed vector changes this writer's bytes
        char name[128]; snprintf(name, sizeof name, "ggml-c1d0e7a-weighted-%s.bin", types[t].name);
        write_bytes(dir, name, out, n);
        n = ggml_row_size(types[t].type, COLS) * EDGE_ROWS;
        assert(types[t].write(edge, out, EDGE_ROWS, COLS, weights) == n);
        snprintf(name, sizeof name, "ggml-c1d0e7a-weighted-%s-edges.bin", types[t].name);
        write_bytes(dir, name, out, n);
        free(out); free(old);
    }
    puts("ggml c1d0e7a: five weighted real-slice and boundary goldens written");
    return 0;
}
