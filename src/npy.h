// A reader for NumPy's .npy header (format versions 1, 2 and 3): the dtype,
// the order and the shape, and where the data starts. Enough to stream a
// C-ordered little-endian array a row at a time; nothing is loaded here.
//
// It exists for --train-teacher (R8.15), whose teacher files are the .npy
// pair a distillation pipeline writes (topk_idx.npy, topk_logp.npy).
#ifndef RUNNER_NPY_H
#define RUNNER_NPY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum { NPY_MAX_DIM = 8 };

typedef struct {
    char descr[16];                 // the dtype string, e.g. "<f4"
    bool fortran_order;
    int ndim;
    int64_t shape[NPY_MAX_DIM];
    int64_t data_offset;            // bytes from the start of the file
} npy_header;

// Parse the header at the start of f (positioned at 0). false, with the
// reason in why, on anything that is not a well-formed .npy header.
bool npy_read_header(FILE *f, npy_header *h, char *why, size_t why_cap);

// Seek to an absolute byte offset; 64-bit on every platform.
bool npy_seek(FILE *f, int64_t off);

// The file's size in bytes, or -1.
int64_t npy_file_size(FILE *f);

#endif
