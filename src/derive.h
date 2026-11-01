// `runner derive` (R17.6.1): make a file from a parent, measure it against a
// named reference, write its record and its card (docs/derive-design.md).
#ifndef RUNNER_DERIVE_H
#define RUNNER_DERIVE_H

// argv[0] is "derive"; returns the process exit status
int derive_main(int argc, char **argv);

#endif
