// Codebook i-quant decode primitives shared by the CUDA kernels and a host
// test. The arithmetic the device runs is this file, compiled by nvcc into
// kernels.cu; the same file compiled by the C compiler is what
// tests/test_iq_decode.c executes against the host tables in
// quants_iq_grids.h, so a change here is measured without a GPU (the
// v0.5.3 review showed the sign helper could be broken with every test
// still green: the tests checked the table's arithmetic, never the code).
//
// Host-side this is plain C11; device-side every function is a
// __forceinline__ __device__ helper. No engine header is included: the
// only inputs are packed bytes and grid entries, and the only outputs are
// floats and unsigned integers.
#ifndef RUNNER_IQ_DECODE_H
#define RUNNER_IQ_DECODE_H

#if defined(__CUDACC__)
#define IQ_FN static __device__ __forceinline__
#define IQ_POPC(x) __popc(x)
#else
#define IQ_FN static inline
#define IQ_POPC(x) __builtin_popcount(x)
#endif

typedef unsigned char iq_byte;
typedef unsigned long long iq_u64;

// Block fields are 2-byte aligned (every block size is even and a row is a
// whole number of blocks), so 32-bit fields are read as two 16-bit halves.
IQ_FN unsigned iq_ld16(const iq_byte *p) {
    return *(const unsigned short *)p;
}
IQ_FN unsigned iq_ld32a2(const iq_byte *p) {
    return iq_ld16(p) | (iq_ld16(p + 2) << 16);
}

// ksigns_iq2xs[i] in quants_iq_grids.h is i with its odd parity in bit 7
// (tests/test_iq_decode.c holds this function to the table for all 128
// entries), so a 7-bit sign index expands arithmetically instead of
// through a divergent table read.
IQ_FN unsigned iq_signs7(unsigned idx) {
    return idx | ((IQ_POPC(idx) & 1u) << 7);
}

// weight j of an 8-magnitude grid entry (byte j), negated by sign bit j
IQ_FN float iq_w8(iq_u64 grid, int j, unsigned signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return (signs >> j) & 1 ? -mag : mag;
}

// weight j of a 4-magnitude grid entry (byte j), negated by sign bit j
IQ_FN float iq_w4(unsigned grid, int j, unsigned signs) {
    float mag = (float)((grid >> (8 * j)) & 0xFF);
    return (signs >> j) & 1 ? -mag : mag;
}

// IQ1 grid bytes are signed (-1/0/1) and carry a per-index delta
IQ_FN float iq1_w(iq_u64 grid, int j, float delta) {
    return (float)(signed char)((grid >> (8 * j)) & 0xFF) + delta;
}
#define IQ1_DELTA 0.125f

#endif
