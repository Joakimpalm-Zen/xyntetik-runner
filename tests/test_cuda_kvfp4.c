// The CUDA KV store kernel against the CPU encoders, byte for byte: the cache
// layout contract says a row quantised on any backend is identical to the
// CPU's fp4_quant_row / q8_quant_row, which is what lets one host buffer serve
// both paths and lets the tolerance gate attribute any drift to accumulation
// order rather than to the stored values. The Metal twin (test_metal_kvfp4.m)
// has held that contract since the fp4 cache landed; CUDA "repeats the same
// arithmetic" was asserted, not tested, until the 2026-09-20 ZEN gate showed
// the CUDA fp4 arm flipping more top-1 decisions than the CPU flips against
// itself. This test settles the store half of that question.
//
// Standalone: loads the CUDA driver API dynamically (nvcuda.dll / libcuda.so),
// JITs the committed PTX and launches k_store_kv exactly as cuda.c does. No
// CUDA toolkit at build time; without a driver at run time it skips.
#include "../src/fp16.h"
#include "../src/quants.h"
#include "../src/kernels_ptx.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
typedef HMODULE dl_t;
static dl_t dl_open(void) { return LoadLibraryA("nvcuda.dll"); }
static void *dl_sym(dl_t h, const char *s) { return (void *)GetProcAddress(h, s); }
#else
#include <dlfcn.h>
typedef void *dl_t;
static dl_t dl_open(void) { dl_t h = dlopen("libcuda.so.1", RTLD_NOW); return h ? h : dlopen("libcuda.so", RTLD_NOW); }
static void *dl_sym(dl_t h, const char *s) { return dlsym(h, s); }
#endif

typedef int CUresult; typedef int CUdevice; typedef void *CUcontext; typedef void *CUmodule;
typedef void *CUfunction; typedef unsigned long long CUdeviceptr;
static struct {
    dl_t lib;
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*PrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*ModuleLoadData)(CUmodule *, const void *);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, void *, void **, void **);
    CUresult (*CtxSynchronize)(void);
    CUresult (*GetErrorString)(CUresult, const char **);
} cu;
// newer drivers export the 64-bit entry points under _v2 names (same rule as cuda.c)
static void *sym2(const char *name) {
    char v2[64]; snprintf(v2, sizeof v2, "%s_v2", name);
    void *p = dl_sym(cu.lib, v2); return p ? p : dl_sym(cu.lib, name);
}
#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_ = "?"; if (cu.GetErrorString) cu.GetErrorString(r_, &s_); \
    fprintf(stderr, "FAIL: %s -> %d (%s)\n", #x, r_, s_); return 1; } } while (0)

static uint32_t st = 0x9E3779B9u;
static float rnd(void) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return (float)(int32_t)st / 2147483648.0f; }

static size_t row_bytes(int kind, int kv_dim) {
    return kind == 2 ? (size_t)kv_dim / 16 * 9 : kind == 1 ? (size_t)kv_dim / 32 * 34 : (size_t)kv_dim * 2;
}
static void ref_row(int kind, const float *src, uint8_t *dst, int kv_dim) {
    if (kind == 2) fp4_quant_row(src, dst, kv_dim);
    else if (kind == 1) q8_quant_row(src, dst, kv_dim);
    else for (int i = 0; i < kv_dim; i++) ((f16_t *)dst)[i] = f32_to_f16(src[i]);
}

// one launch of k_store_kv with K kind kk and V kind vk over `cols` rows;
// returns the number of rows that differ from the CPU encoders
static int run_case(CUfunction f, const float *k, const float *v, int kv_dim, int cols, int kk, int vk, const char *name) {
    size_t kb = row_bytes(kk, kv_dim), vb = row_bytes(vk, kv_dim);
    CUdeviceptr dk = 0, dv = 0, dkc = 0, dvc = 0, dpos = 0;
    int pos0 = 0;
    CK(cu.MemAlloc(&dk, sizeof(float) * kv_dim * cols)); CK(cu.MemAlloc(&dv, sizeof(float) * kv_dim * cols));
    CK(cu.MemAlloc(&dkc, kb * cols)); CK(cu.MemAlloc(&dvc, vb * cols)); CK(cu.MemAlloc(&dpos, sizeof(int)));
    CK(cu.MemcpyHtoD(dk, k, sizeof(float) * kv_dim * cols)); CK(cu.MemcpyHtoD(dv, v, sizeof(float) * kv_dim * cols));
    CK(cu.MemsetD8(dkc, 0xAB, kb * cols)); CK(cu.MemsetD8(dvc, 0xAB, vb * cols)); CK(cu.MemcpyHtoD(dpos, &pos0, sizeof pos0));
    // k_store_kv(k, v, kc, vc, kv_dim, l_off, posp, q8, ring, v_off, vq8): the
    // cuda.c launch shape, one thread per stored unit over max(units_k, units_v)
    int ku = kk == 2 ? kv_dim / 16 : kk ? kv_dim / 32 : kv_dim;
    int vu = vk == 2 ? kv_dim / 16 : vk ? kv_dim / 32 : kv_dim;
    int units = ku > vu ? ku : vu, ring = 0;
    unsigned long long l_off = 0, v_off = 0;
    void *ps[] = { &dk, &dv, &dkc, &dvc, &kv_dim, &l_off, &dpos, &kk, &ring, &v_off, &vk };
    CK(cu.LaunchKernel(f, (units + 63) / 64, cols, 1, 64, 1, 1, 0, NULL, ps, NULL));
    CK(cu.CtxSynchronize());
    uint8_t *hk = malloc(kb * cols), *hv = malloc(vb * cols), *ref = malloc(kb > vb ? kb : vb);
    CK(cu.MemcpyDtoH(hk, dkc, kb * cols)); CK(cu.MemcpyDtoH(hv, dvc, vb * cols));
    int bad = 0;
    for (int c = 0; c < cols; c++) {
        ref_row(kk, k + (size_t)c * kv_dim, ref, kv_dim);
        if (memcmp(ref, hk + c * kb, kb)) {
            if (bad < 3) { size_t i = 0; while (i < kb && ref[i] == hk[c * kb + i]) i++;
                fprintf(stderr, "  %s K row %d differs at byte %zu of %zu: cpu %02x gpu %02x\n", name, c, i, kb, ref[i], hk[c * kb + i]); }
            bad++;
        }
        ref_row(vk, v + (size_t)c * kv_dim, ref, kv_dim);
        if (memcmp(ref, hv + c * vb, vb)) {
            if (bad < 3) { size_t i = 0; while (i < vb && ref[i] == hv[c * vb + i]) i++;
                fprintf(stderr, "  %s V row %d differs at byte %zu of %zu: cpu %02x gpu %02x\n", name, c, i, vb, ref[i], hv[c * vb + i]); }
            bad++;
        }
    }
    free(hk); free(hv); free(ref);
    cu.MemFree(dk); cu.MemFree(dv); cu.MemFree(dkc); cu.MemFree(dvc); cu.MemFree(dpos);
    printf("cuda kv store %-6s: %d of %d rows byte-identical to the CPU encoders\n", name, 2 * cols - bad, 2 * cols);
    return bad;
}

int main(void) {
    cu.lib = dl_open();
    if (!cu.lib) { printf("cuda kv fp4 store: skipped (no CUDA driver)\n"); return 0; }
    cu.Init = dl_sym(cu.lib, "cuInit"); cu.DeviceGet = dl_sym(cu.lib, "cuDeviceGet");
    cu.PrimaryCtxRetain = sym2("cuDevicePrimaryCtxRetain"); cu.CtxSetCurrent = dl_sym(cu.lib, "cuCtxSetCurrent");
    cu.ModuleLoadData = dl_sym(cu.lib, "cuModuleLoadData"); cu.ModuleGetFunction = dl_sym(cu.lib, "cuModuleGetFunction");
    cu.MemAlloc = sym2("cuMemAlloc"); cu.MemFree = sym2("cuMemFree");
    cu.MemcpyHtoD = sym2("cuMemcpyHtoD"); cu.MemcpyDtoH = sym2("cuMemcpyDtoH"); cu.MemsetD8 = sym2("cuMemsetD8");
    cu.LaunchKernel = dl_sym(cu.lib, "cuLaunchKernel"); cu.CtxSynchronize = dl_sym(cu.lib, "cuCtxSynchronize");
    cu.GetErrorString = dl_sym(cu.lib, "cuGetErrorString");
    if (!cu.Init || !cu.DeviceGet || !cu.PrimaryCtxRetain || !cu.CtxSetCurrent || !cu.ModuleLoadData ||
        !cu.ModuleGetFunction || !cu.MemAlloc || !cu.MemFree || !cu.MemcpyHtoD || !cu.MemcpyDtoH ||
        !cu.MemsetD8 || !cu.LaunchKernel || !cu.CtxSynchronize) {
        printf("cuda kv fp4 store: skipped (driver API incomplete)\n"); return 0;
    }
    if (cu.Init(0)) { printf("cuda kv fp4 store: skipped (cuInit failed: no usable device)\n"); return 0; }
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction f;
    CK(cu.DeviceGet(&dev, 0)); CK(cu.PrimaryCtxRetain(&ctx, dev)); CK(cu.CtxSetCurrent(ctx));
    CK(cu.ModuleLoadData(&mod, k_ptx_src));
    CK(cu.ModuleGetFunction(&f, mod, "k_store_kv"));

    // the Metal test's rows: four magnitude bands, a zero row, exact E2M1 ties
    const int kv_dim = 256, cols = 64;
    float *k = malloc(sizeof(float) * kv_dim * cols), *v = malloc(sizeof(float) * kv_dim * cols);
    for (int c = 0; c < cols; c++) {
        float mag = c % 4 == 0 ? 1e-3f : c % 4 == 1 ? 300.0f : c % 4 == 2 ? 8.0f : 1.0f;
        for (int i = 0; i < kv_dim; i++) { k[c * kv_dim + i] = rnd() * mag; v[c * kv_dim + i] = rnd() * mag; }
        if (c == 5) memset(k + c * kv_dim, 0, sizeof(float) * kv_dim);
        if (c == 6) for (int i = 0; i < kv_dim; i++) k[c * kv_dim + i] = (i % 8 - 4) * 0.25f * 1.5f;
        // and the K shape that broke fp4 on Qwen2.5 (attention bias): a few
        // channels two orders of magnitude above the rest of their block
        if (c == 7) for (int i = 0; i < kv_dim; i++) k[c * kv_dim + i] = (i % 16 == 3) ? 180.0f : rnd();
    }
    int bad = 0;
    bad += run_case(f, k, v, kv_dim, cols, 2, 2, "fp4");
    bad += run_case(f, k, v, kv_dim, cols, 1, 1, "q8");
    bad += run_case(f, k, v, kv_dim, cols, 1, 2, "k8v4");
    bad += run_case(f, k, v, kv_dim, cols, 0, 0, "f16");
    free(k); free(v);
    if (bad) { fprintf(stderr, "FAIL: %d stored rows differ from the CPU encoders\n", bad); return 1; }
    printf("cuda kv store: every row byte-identical to the CPU encoders (kv_dim %d, %d rows x 4 layouts, "
           "four magnitude bands, zero row, E2M1 ties, Qwen2.5-style K outliers)\n", kv_dim, cols);
    return 0;
}
