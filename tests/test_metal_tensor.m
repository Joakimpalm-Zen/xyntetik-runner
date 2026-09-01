// Metal 4 tensor admission must be independently compilable: failure here
// may disable the tensor rung, never the established Metal backend.
#import <Metal/Metal.h>
#include <stdio.h>

#include "../src/kernels_tensor_metal.h"

int main(void) {
#if !defined(MAC_OS_VERSION_26_0)
    puts("metal tensor: skipped (build SDK predates Metal 4)");
    return 0;
#else
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { puts("metal tensor: skipped (no Metal device)"); return 0; }
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion4_0;
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
            [NSString stringWithUTF8String:k_metal_tensor_src]
            options:opts error:&err];
        if (!lib) {
            fprintf(stderr, "FAIL: Metal 4 tensor source: %s\n",
                    err ? err.localizedDescription.UTF8String : "unknown error");
            return 1;
        }
        const char *names[] = { "k_tensor_q4_K", "k_tensor_q8_0",
                                "k_tensor_q4_0" };
        for (int i = 0; i < 3; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:names[i]]];
            if (!fn) {
                fprintf(stderr, "FAIL: %s missing\n", names[i]);
                return 1;
            }
            NSError *perr = nil;
            id<MTLComputePipelineState> p =
                [dev newComputePipelineStateWithFunction:fn error:&perr];
            if (!p) {
                fprintf(stderr, "FAIL: %s pipeline: %s\n", names[i],
                        perr ? perr.localizedDescription.UTF8String
                             : "unknown error");
                return 1;
            }
            [p release]; [fn release];
        }
        printf("metal tensor: source and Q4_K/Q8_0/Q4_0 pipelines admitted "
               "on %s\n", dev.name.UTF8String);
        [lib release]; [opts release]; [dev release];
    }
    return 0;
#endif
}
