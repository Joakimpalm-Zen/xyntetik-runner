// R6.86: the p_expf the vector-friendly one replaced, compiled under strict
// float in a unit of its own, so that tests/test_pmath_expf.c built with the
// engine's default flags (-ffast-math) can hold its NaN answers to the strict
// build's. Under -ffast-math a NaN is undefined for code the compiler sees,
// so the old function compiled there has no answer to compare against.
#include "pmath.h"

float pmath_expf_strict_oracle(float x);
float pmath_expf_strict_oracle(float x) { return (float)pm_exp_d((double)x); }
