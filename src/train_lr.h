// Strict-math learning-rate schedule shared across default and T3 builds (R8.29).
#ifndef RUNNER_TRAIN_LR_H
#define RUNNER_TRAIN_LR_H

double train_lr_schedule_at(double base_lr, int step, int lr_warmup,
                            int lr_schedule, int train_steps, double lr_floor);

#endif
