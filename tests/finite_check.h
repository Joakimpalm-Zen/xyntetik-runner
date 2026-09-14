// Non-finite detection that survives -ffast-math; see finite_check.c.
#ifndef RUNNER_TESTS_FINITE_CHECK_H
#define RUNNER_TESTS_FINITE_CHECK_H
#include <stddef.h>
#include <stdint.h>

// Number of NaN/Inf entries in v[0..n); *first (optional) receives the index
// of the first one, or n when there is none.
size_t count_nonfinite_f32(const float *v, size_t n, size_t *first);

#endif
