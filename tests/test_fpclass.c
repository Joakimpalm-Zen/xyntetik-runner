// fpclass.h's class tests under the build's own flags.
//
// The engine is compiled with -ffast-math, whose -ffinite-math-only lets the
// compiler assume no NaN or infinity exists; clang 22 then folds isfinite() to
// 1 and isnan() to 0, and LLVM folds an exponent-bits test the same way. This
// binary is built with $(CFLAGS), so it holds the helpers on whatever
// compiler the build uses. Every special value is made at run time from a
// string, out of reach of constant folding.
#include "fpclass.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static double num(const char *s) { return strtod(s, NULL); }

int main(void) {
    const double inf = num("1e999"), ninf = num("-1e999"), nan = num("nan");
    const double big = num("1.7976931348623157e308"), tiny = num("4.9e-324");
    assert(!fp_isfinite(inf) && !fp_isfinite(ninf) && !fp_isfinite(nan));
    assert(fp_isfinite(num("1.5")) && fp_isfinite(big) && fp_isfinite(tiny) &&
           fp_isfinite(num("0")) && fp_isfinite(num("-0")));
    assert(fp_isnan(nan) && !fp_isnan(inf) && !fp_isnan(ninf) && !fp_isnan(big));

    const float finf = (float)inf, fnan = (float)nan;
    const float over = (float)big;               // a finite double overflowing float
    const float made = (float)num("3.0e38") * 10.0f;   // an operation's result
    assert(!fp_isfinite(finf) && !fp_isfinite(fnan) && !fp_isfinite(over) &&
           !fp_isfinite(made));
    assert(fp_isnan(fnan) && !fp_isnan(finf) && !fp_isnan(made));
    assert(fp_isfinite((float)num("65504")) && !fp_isnan((float)num("-2.5")));

    puts("fpclass: ok");
    return 0;
}
