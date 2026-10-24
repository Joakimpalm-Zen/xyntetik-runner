// The calling thread's floating-point control state, for the threads the
// engine starts to compute in.
//
// -ffast-math links crtfastmath, which sets flush-to-zero and
// denormals-are-zero in the main thread at startup (x86 MXCSR; arm64
// FPCR.FZ where the toolchain links it). Linux and macOS threads start with
// their creator's state, but Windows threads start with the default one: a
// pool worker kept subnormals while the calling thread flushed them, so a
// row's bytes depended on which thread computed it (tests/test_lora_chains.c
// on the Windows CI: 86 of 915 shapes, a 0 * inf on one thread where the
// other had subnormal * inf). Each such thread takes its creator's state
// when it starts, so every platform behaves as Linux does.
#ifndef RUNNER_FPENV_H
#define RUNNER_FPENV_H

#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
typedef unsigned int fpenv_t;
// the control bits only: the low six are sticky exception flags
static inline fpenv_t fpenv_get(void) { return _mm_getcsr() & ~0x3Fu; }
static inline void fpenv_set(fpenv_t e) { _mm_setcsr(e); }
#elif defined(__aarch64__)
typedef uint64_t fpenv_t;
static inline fpenv_t fpenv_get(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, fpcr" : "=r"(v));
    return v;
}
static inline void fpenv_set(fpenv_t e) { __asm__ volatile("msr fpcr, %0" :: "r"(e)); }
#else
typedef int fpenv_t;
static inline fpenv_t fpenv_get(void) { return 0; }
static inline void fpenv_set(fpenv_t e) { (void)e; }
#endif

#endif // RUNNER_FPENV_H
