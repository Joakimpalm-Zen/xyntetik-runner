// Fetch a GGUF from the Hugging Face Hub by repository id: `-hf owner/repo`
// or `-hf owner/repo:TAG`, the spelling llama.cpp and Ollama users already
// know, so a model page's "Use this model" snippet is one line.
//
// The module has two layers. hf_select_file is the decision, a pure function
// over the repository's file list that a test can drive without a network:
// which GGUF the spec names, or exactly why it cannot say. hf_fetch is the
// transfer: the file list from the Hub API, the download into the user's
// cache, and a SHA-256 check against the Hub's own LFS record before the
// path is handed to the loader. The transfer is done by the system `curl`
// (`RUNNER_CURL` overrides the program, which is how the tests substitute a
// fake hub), because a dependency-free C11 binary has no TLS of its own and
// curl ships with macOS, every Linux this runs on, and Windows 10+.
#ifndef RUNNER_HFHUB_H
#define RUNNER_HFHUB_H

#include <stdbool.h>
#include <stddef.h>

// Pick the GGUF a spec names among `n` repository file names.
//
// Candidates are the `.gguf` files that are not a vision projector
// (`mmproj-*`). A multipart file (`<prefix>-00001-of-00003.gguf`) counts as
// one candidate, represented by its first part. `tag` (may be NULL or empty)
// must match the candidate's base name as a whole token, case-insensitive,
// bounded by the name's ends or by `-`, `_` or `.`: `Q4_K_M` matches
// `gemma-4-E4B-it-Q4_K_M.gguf` and not `...-Q4_K_M-imatrix-Q4_K_S...`'s
// `Q4_K`. Returns the index into `names` of the file to download, or -1 with
// the reason in `err`: no GGUF at all, no file for the tag, or several
// candidates when the spec did not choose one (the message lists them, so a
// user who typed the repo alone learns what to append).
int hf_select_file(const char *const *names, int n, const char *tag,
                   char *err, size_t errcap);

// True when `name` is the first part of a multipart GGUF; *count receives
// the number of parts and *prefix_len the length of the shared prefix
// (`<prefix>-00001-of-00003.gguf`), so the caller can spell the siblings.
bool hf_is_first_part(const char *name, unsigned *count, size_t *prefix_len);

// Resolve `-hf owner/repo[:TAG]` to a local file. Fetches the file list,
// selects, downloads what is missing from the cache (every part of a
// multipart file), verifies each downloaded part's SHA-256 against the
// Hub's LFS record (or its size when the file is not stored in LFS), and
// returns the malloc'd path of the file to load (the first part), or NULL
// with the reason in `err`. The cache is `$RUNNER_HF_CACHE`, else
// `~/.cache/xyntetik-runner/hf` (`%LOCALAPPDATA%\xyntetik-runner\hf` on
// Windows), laid out as `<owner>--<repo>/<file>`. `HF_TOKEN`, when set, is
// sent as a bearer token so gated repositories work.
char *hf_fetch(const char *spec, char *err, size_t errcap);

#endif
