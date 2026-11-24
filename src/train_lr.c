// Strict-math learning-rate schedule (R8.29): compiled with -fno-fast-math
// and -ffp-contract=off, calling pmath.h's pm_sincos_d directly so default,
// PORTABLE_MATH=1, and T3=1 builds produce the exact same IEEE-754 double bits
// on every platform and C library.
#include "pmath.h"
#include "train_lr.h"

double train_lr_schedule_at(double base, int s, int lr_warmup,
                            int lr_schedule, int train_steps, double lr_floor) {
    if (s < lr_warmup) return base * (double)(s + 1) / (double)lr_warmup;
    if (lr_schedule != 1) return base;
    int span = train_steps - lr_warmup;
    double p = (double)(s - lr_warmup) / (double)(span > 1 ? span : 1);
    double sn, cs;
    pm_sincos_d(3.14159265358979323846 * p, &sn, &cs);
    return base * (lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + cs));
}
