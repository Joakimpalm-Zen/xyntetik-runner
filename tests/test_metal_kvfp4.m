// The Metal fp4 KV store kernel against the CPU encoder, byte for byte: the
// cache layout contract says a row quantised on any backend is identical to
// the CPU's fp4_quant_row, which is what lets one host buffer serve both
// paths and lets the tolerance gate attribute any drift to accumulation
// order rather than to the stored values. Random rows across magnitudes,
// plus the exact ties the encoder's thresholds sit on.
#import <Metal/Metal.h>
#include "../src/kernels_metal.h"
#include "../src/quants.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int kv_dim, q8, stride, pos, kv_rows; uint64_t off, row_b;
                 int vq8; uint64_t voff, vrow_b; } store_args_host;

static uint32_t st = 0x9E3779B9u;
static float rnd(void) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return (float)(int32_t)st / 2147483648.0f; }

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("metal kv fp4 store: skipped (no Metal device)\n"); return 0; }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:k_metal_src] options:nil error:&err];
        if (!lib) { fprintf(stderr, "FAIL: library: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"k_store_kv"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "FAIL: pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const int kv_dim = 256, cols = 64;
        size_t row_b = (size_t)kv_dim / 16 * 9;
        float *k = malloc(sizeof(float) * kv_dim * cols), *v = malloc(sizeof(float) * kv_dim * cols);
        for (int c = 0; c < cols; c++) {
            float mag = c % 4 == 0 ? 1e-3f : c % 4 == 1 ? 300.0f : c % 4 == 2 ? 8.0f : 1.0f;
            for (int i = 0; i < kv_dim; i++) { k[c * kv_dim + i] = rnd() * mag; v[c * kv_dim + i] = rnd() * mag; }
            if (c == 5) { memset(k + c * kv_dim, 0, sizeof(float) * kv_dim); }
            if (c == 6) { for (int i = 0; i < kv_dim; i++) k[c * kv_dim + i] = (i % 8 - 4) * 0.25f * 1.5f; } // exact E2M1 ties
        }
        id<MTLBuffer> kb = [dev newBufferWithBytes:k length:sizeof(float) * kv_dim * cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> vb = [dev newBufferWithBytes:v length:sizeof(float) * kv_dim * cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> kc = [dev newBufferWithLength:row_b * cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> vc = [dev newBufferWithLength:row_b * cols options:MTLResourceStorageModeShared];
        memset(kc.contents, 0xAB, row_b * cols); memset(vc.contents, 0xAB, row_b * cols);
        store_args_host a = { kv_dim, 2, kv_dim, 0, 0, 0, row_b, 2, 0, row_b };
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:kb offset:0 atIndex:0]; [e setBuffer:vb offset:0 atIndex:1];
        [e setBuffer:kc offset:0 atIndex:2]; [e setBuffer:vc offset:0 atIndex:3];
        [e setBytes:&a length:sizeof a atIndex:4];
        [e dispatchThreads:MTLSizeMake(kv_dim / 16, cols, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        uint8_t *ref = malloc(row_b);
        int bad = 0;
        for (int c = 0; c < cols; c++) {
            fp4_quant_row(k + c * kv_dim, ref, kv_dim);
            if (memcmp(ref, (uint8_t *)kc.contents + c * row_b, row_b)) {
                if (bad < 3) {
                    const uint8_t *got = (uint8_t *)kc.contents + c * row_b;
                    for (size_t i = 0; i < row_b; i++) if (ref[i] != got[i]) {
                        fprintf(stderr, "K col %d byte %zu: cpu %02x metal %02x (block %zu, x=%g)\n", c, i, ref[i], got[i], i / 9, k[c * kv_dim + (i / 9) * 16]); break; }
                }
                bad++;
            }
            fp4_quant_row(v + c * kv_dim, ref, kv_dim);
            if (memcmp(ref, (uint8_t *)vc.contents + c * row_b, row_b)) bad++;
        }
        if (bad) { fprintf(stderr, "metal kv fp4 store: %d of %d rows differ from the CPU encoder\n", bad, 2 * cols); return 1; }
        printf("metal kv fp4 store: %d rows byte-identical to fp4_quant_row (kv_dim %d, four magnitude bands, zero row, E2M1 ties)\n", 2 * cols, kv_dim);
    }
    return 0;
}
