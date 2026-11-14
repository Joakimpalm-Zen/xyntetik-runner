# Copy-from-context decoding as a CPU default (R3.65), 2026-11-14

**Finding: on the CPU, `--draft-lookup` gives the same tokens as plain decoding but is not a default.**
On an edit prompt it is 1.15x faster and on prose 11.8% slower, against the item's bar of at least 1.2x on
the edit and no more than 2% slower on prose. So it stays opt-in on the CPU. On Metal the self-speculation
study measured 0.97x and it stays off there. On CUDA, where the server already uses it for requests that
carry `draft_hints` (`engine_draft_default`), it was not measured: no CUDA host was up.

## Defaults today, by the code

`engine_draft_default` (src/engine.c) picks lookup only on a CUDA full offload, in the server, for a request
with `draft_hints`. The CLI and every other server request decode plain unless `--draft-lookup` is given.

## Measurement

SmolLM2-1.7B-Instruct Q4_K_M, the server on the CPU (`--gpu off -t 4`), one chat request per prompt,
greedy, through `/v1/chat/completions`. `edit.txt` asks the model to rename a function in a 26-line Python
file and print the whole file back (it does, 284 tokens). `prose.txt` asks for a short essay (300 tokens).
Each arm runs as its own server. Rounds 2 to 4 interleave the arms; round 1, a cold start, is left out.
The times are wall-clock per request, prefill included.

| prompt | plain (s) | lookup (s) | lookup's speed |
|---|---|---|---:|
| edit | 21.84, 22.44, 21.81 | 19.58, 18.76, 19.39 | 1.15x |
| prose | 18.56, 18.46, 18.54 | 21.78, 20.13, 20.18 | 0.89x (11.8% slower) |

Every pair returned the same content (sha256 `89bbbfe7e42e…` for the edit, `f5e8822156b4…` for the prose).

Command, per arm (`plain`, `lookup`): `./srv.sh RUNNER SmolLM2-1.7B-Instruct-Q4_K_M.gguf ARM 18556`.

## Limits

One small model and one machine. A larger model spends more per forward, which favours drafting; the
prose cost comes from verify rounds whose drafts are mostly rejected.
