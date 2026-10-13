// Metal 4 tensor admission must be independently compilable: failure here
// may disable the tensor rung, never the established Metal backend.
//
// The gate mirrors the engine's admission (metal_tensor_device_ok in
// src/metal.m): the tensor library is used on an Apple M5 or newer running
// macOS 26.2 or newer, and anywhere else the engine never compiles it. So a
// toolchain that refuses the source is a FAIL on a device the engine would
// admit, and a skip with the reason printed on one it would not: the hosted
// macOS runner (an older Apple Silicon class, one macOS release behind the
// lab) rejected the cooperative_tensor kernels and held the macOS suite red
// for every run it ever had, while the shipped engine on that same machine
// falls back to the simdgroup GEMM with a message.
#import <Metal/Metal.h>
#include <stdio.h>

#include "../src/kernels_tensor_metal.h"

#if defined(MAC_OS_VERSION_26_0)
static bool engine_would_admit(id<MTLDevice> dev) {
    if (![dev.name hasPrefix:@"Apple M5"] && ![dev.name hasPrefix:@"Apple M6"])
        return false;
    NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
    return v.majorVersion > 26 || (v.majorVersion == 26 && v.minorVersion >= 2);
}

// A toolchain refusal: fatal where the engine admits the tensor rung,
// a documented skip where it never would.
static int toolchain_refused(id<MTLDevice> dev, const char *what, NSError *err) {
    const char *why = err ? err.localizedDescription.UTF8String : "unknown error";
    NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
    if (engine_would_admit(dev)) {
        fprintf(stderr, "FAIL: %s: %s\n", what, why);
        return 1;
    }
    printf("metal tensor: skipped (%s, macOS %ld.%ld: the engine does not admit "
           "tensor GEMM on this device, and this toolchain refuses %s: %s)\n",
           dev.name.UTF8String, (long)v.majorVersion, (long)v.minorVersion,
           what, why);
    return 0;
}
#endif

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
        if (!lib)
            return toolchain_refused(dev, "Metal 4 tensor source", err);
        const char *names[] = { "k_tensor_q4_K", "k_tensor_q8_0",
                                "k_tensor_q4_0" };
        for (int i = 0; i < 3; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:names[i]]];
            if (!fn) {
                // a missing entry point is a source defect on any toolchain
                fprintf(stderr, "FAIL: %s missing\n", names[i]);
                return 1;
            }
            NSError *perr = nil;
            id<MTLComputePipelineState> p =
                [dev newComputePipelineStateWithFunction:fn error:&perr];
            if (!p) {
                char what[96];
                snprintf(what, sizeof what, "%s pipeline", names[i]);
                return toolchain_refused(dev, what, perr);
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
