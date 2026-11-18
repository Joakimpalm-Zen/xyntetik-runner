// A sum that a portable build must add in its written order (R6.86).
//
// src/pmath.h's expf has no branch and no loop, so a loop that calls it can
// vectorize. Under -ffast-math (a portable build of the default flags) a loop
// that both calls expf and sums would then add in another order, a vector of
// partial sums, and the sum's last bits move. That moved two Windows CPU
// replies off their pins at a near-tie. The expf before it kept every such
// loop scalar, so its sums went one add after another.
//
// PM_IN_ORDER(acc), right after the add, keeps that order. In a portable build
// it is an empty asm that takes the accumulator in a register and gives it
// back, which no compiler can reorder an add across: the loop stays scalar,
// as it was. In any other build it is nothing, and the loop compiles as
// before, byte for byte. The attention softmax (src/softmax.h) keeps its exp
// in a loop of its own, so that part still vectorizes.
#ifndef RUNNER_PM_ORDER_H
#define RUNNER_PM_ORDER_H

#ifdef RUNNER_PORTABLE_MATH
#if defined(__x86_64__)
#define PM_IN_ORDER(v) __asm__("" : "+x"(v))
#elif defined(__aarch64__)
#define PM_IN_ORDER(v) __asm__("" : "+w"(v))
#elif defined(__riscv) && defined(__riscv_flen)
#define PM_IN_ORDER(v) __asm__("" : "+f"(v))
#else
#define PM_IN_ORDER(v) __asm__("" : "+m"(v))
#endif
#else
#define PM_IN_ORDER(v) ((void)0)
#endif

#endif
