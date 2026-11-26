// Streaming pre-computed reference logits reader & exact float64 KL scorer
// (R4.48.8.3).
//
// Reads a pinned manifest (`xyntetik.runner.ref-logits.v1`) pointing to a
// little-endian C-order float32 [N, V] logits file in either NumPy `.npy`
// format (parsed via src/npy.h) or headerless `raw-f32` format (directly
// compatible with ref_firstlight.py's `*.logits.f32` dumps).
//
// Memory use during scoring is O(V) (~0.8 MB for Inkling 2 Mini's V=200,058),
// independent of sequence length N, and the KL calculation matches
// compare_full.py's exact full-vocabulary float64 log_softmax formula.
#ifndef RUNNER_REF_LOGITS_H
#define RUNNER_REF_LOGITS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define REF_LOGITS_SCHEMA "xyntetik.runner.ref-logits.v1"

typedef struct {
    char     manifest_sha256[65];
    char     logits_path[4096];    // resolved path on disk (regular non-symlink file)
    char     logits_rel[512];      // relative path recorded in manifest
    char     logits_format[16];    // "npy" or "raw-f32"
    char     logits_sha256[65];    // lowercase 64-hex
    char     corpus_sha256[65];    // "" if omitted in manifest
    int64_t  n_rows;
    int32_t  n_vocab;
    int32_t  n_special;
    int32_t *tokens;
    int32_t  n_tokens;
    int32_t *row_indices;          // NULL if dense; strictly increasing [n_rows] if sparse
    int64_t  data_offset;          // .npy data start offset, or 0 for raw-f32
    // Provenance metadata (owned heap strings, NULL if absent)
    char    *ref_repo;
    char    *ref_revision;
    char    *ref_class;
    char    *ref_dtype;
    char    *ref_device;
    char    *ref_producer;
    char    *ref_producer_mode;
    char    *ref_weights_sha256;
} ref_logits_manifest;

typedef struct {
    int32_t pos;
    double  kld;
    double  gold_margin;
    int32_t gold_top1;
    int32_t cand_top1;
    bool    agree;
    double  max_abs;
} ref_logits_row_result;

typedef struct {
    ref_logits_manifest m;
    FILE               *fp;
    float              *row_buf;   // [m.n_vocab]
    int64_t             cursor;    // next sparse index to check
} ref_logits_reader;

// Parse and validate a manifest JSON file and its referenced logits binary.
// Always checks schema, path safety (no traversal/absolute/symlink), header/size,
// and (when verify_sha256 is true) verifies logits.sha256 via streaming SHA-256.
bool ref_logits_load_manifest(const char *manifest_path, bool verify_sha256,
                              ref_logits_manifest *out, char *why, size_t why_cap);

void ref_logits_manifest_free(ref_logits_manifest *m);

// Verify that every position e = n_special + start + n * stride - 1 (0 <= n < count)
// is present in the manifest and within [0, n_tokens - 1).
bool ref_logits_covers_window(const ref_logits_manifest *m,
                              int start, int stride, int count,
                              char *why, size_t why_cap);

// Open a streaming reader, verifying manifest + logits.sha256 + exact model
// vocabulary size + n_special + prefix token ID match against [toks, n_prompt].
bool ref_logits_open(const char *manifest_path, int model_n_vocab, int n_special,
                     const int32_t *toks, int n_prompt,
                     ref_logits_reader *r, char *why, size_t why_cap);

void ref_logits_close(ref_logits_reader *r);

// Returns true if emitted_pos has a reference row in the manifest, setting
// *file_row to the 0-based row index inside the logits file. Callers must pass
// strictly increasing emitted_pos values (0, 1, 2, ...).
bool ref_logits_want_pos(ref_logits_reader *r, int32_t emitted_pos, int64_t *file_row);

// Seek and stream one reference float32 row, check finiteness on both sides,
// and compute exact full-vocab float64 KL(ref || cand) plus top-1/margin stats.
bool ref_logits_score_row(ref_logits_reader *r, int32_t emitted_pos, int64_t file_row,
                          const float *cand_logits, ref_logits_row_result *out,
                          char *why, size_t why_cap);

#endif
