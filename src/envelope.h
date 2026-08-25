#ifndef RUNNER_ENVELOPE_H
#define RUNNER_ENVELOPE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <stdbool.h>

// Measured-envelope manifest reader (slice 2 of the certified-envelope gate).
//
// scripts/certify-envelope.py emits a `<model>.envelope.json` sidecar that
// INDEXES the evidence a certification run produced for one artifact. This
// reads the sidecar sitting next to a loaded model and reports whether the
// CURRENT runtime (version + compiled backend) matches what was measured.
// envelope_report is READ + REPORT ONLY; envelope_gate (slice 3) adds the
// load-time refusal for a matching outside-envelope verdict. Reporting never
// fails a load: an absent or malformed manifest is a state, not an error.
//
// Resolution is EXACT-MATCH by design (no closest-class fallback): the manifest
// only speaks for the runtime version and backend it was measured on. H8/H10
// wording — a configuration "matches a measured envelope", it is not "certified"
// as a standing property.

// The five states are distinguished by EPISTEMIC status, not just behavior: two
// of them (experimental, indeterminate) both load-with-a-banner but mean
// different things, and conflating them re-imports a semantic mismatch. What we
// KNOW about the model, from most to least:
enum envelope_state {
    ENV_UNCLASSIFIED = 0, // no manifest beside the model. TRANSITIONAL / LEGACY:
                       // the model predates or sits outside the certification
                       // pipeline, so there is nothing measured to say — NOT a
                       // measurement that came back inconclusive. Silent.
    ENV_CERTIFIED,     // manifest matches this version+backend and passed the gate
    ENV_OUTSIDE,       // manifest matches this version+backend, verdict outside-envelope
    ENV_EXPERIMENTAL,  // manifest MATCHES this version+backend and its verdict is
                       // literally "experimental" — a real measurement that came
                       // back inconclusive. Loads with a banner.
    ENV_INDETERMINATE, // a sidecar IS present but cannot be used to judge this
                       // run: unreadable/malformed, an unknown schema, or
                       // measured on a DIFFERENT version/backend ("foreign").
                       // We could not tell — distinct from "we measured and it
                       // was inconclusive". Fail-open: loads with a banner.
};

// `model_path`: the loaded GGUF path (the manifest is `<model_path>.envelope.json`).
// `runtime_version`: RUNNER_VERSION. `backend`: the compiled backend name
// ("metal" / "cuda" / "cpu"). Writes a one-line human summary into `out`
// (<= cap, always NUL-terminated when cap > 0) and returns the state.
// sha256 of a whole file as lowercase hex (exported for the training
// provenance record; the envelope gate uses it internally)
bool envelope_file_sha256(const char *path, char hex[65]);
// sha256 of an in-memory buffer as lowercase hex (the transcript chain hash)
void envelope_data_sha256(const void *data, size_t n, char hex[65]);

// ---- notarized inference D1: xyntetik.runner.transcript.v1 ----------------
// The record that makes one inference a REPLAYABLE COMPUTATION: everything a
// verifier needs to rerun (model sha, build identity, profile, config, seed,
// prompt) plus what was produced (token ids, text, finish reason), bound by a
// chain hash. Verdict tiers are frozen from the D0 falsifier ladder: T1
// same-binary = bit-exact replay; T2 cross-ISA = token replay (libm is the
// bit blocker). The chain hash covers every byte of the file BEFORE the
// `,"chain"` key — recomputable by any reader with a text editor and
// sha256sum, which is the point.
typedef struct {
    const char *out_path;
    const char *runner_version;
    const char *argv0;             // hashed for build identity (may fail ->
                                   // empty sha and the record says so)
    const char *compiler, *os, *arch;
    const char *device;            // "cpu" | gpu device name
    bool        gpu;
    int         threads, n_ctx, n_batch;
    bool        kv_q8;
    const char *model_path;        // hashed
    const char *adapter_path;      // NULL = no adapter; hashed when set
    float       adapter_scale;
    uint64_t    seed;
    float       temp, top_p, min_p, repeat_penalty;
    int         top_k, n_predict;
    bool        bos;
    const char *prompt_text;
    const int32_t *prompt_tokens;
    int         n_prompt;
    const char *output_text;       // the streamed bytes, verbatim
    size_t      output_text_len;
    const int32_t *output_tokens;
    int         n_output;
    bool        hit_stop;          // true = "stop", false = "length"
} transcript_info;

// Writes the transcript beside the run. Returns false (with the reason on
// stderr) on any failure; a partial transcript is never left installed.
bool transcript_write(const transcript_info *ti);

int envelope_report(const char *model_path, const char *runtime_version,
                    const char *backend, char *out, int cap);

// Load-time three-state ENFORCEMENT (slice 3). Resolves the state exactly as
// envelope_report does, then decides whether the load may proceed and what the
// caller should print. `forced` is the --force-uncertified flag; it only
// affects the OUTSIDE state.
//   OUTSIDE   + !forced -> returns FALSE (refuse the load); `msg` is the
//                          measured reason and how to override.
//   OUTSIDE   + forced  -> returns true;  `msg` is a loud override warning.
//   CERTIFIED / EXPERIMENTAL / INDETERMINATE -> returns true; `msg` is the
//                          informational banner (the same line envelope_report
//                          would produce).
//   UNCLASSIFIED (no manifest) -> returns true; `msg` is "" — a model outside
//                          the certification pipeline loads quietly, no banner.
// It is fail-open on doubt: an unreadable/unknown/foreign manifest is
// INDETERMINATE and never blocks a load — a wrong refusal is worse than none.
// Only a MATCHING outside-envelope verdict refuses. `*out_state`
// (may be NULL) receives the resolved envelope_state. `msg` is always
// NUL-terminated when cap > 0.
bool envelope_gate(const char *model_path, const char *runtime_version,
                   const char *backend, bool forced,
                   char *msg, int cap, int *out_state);

#endif
