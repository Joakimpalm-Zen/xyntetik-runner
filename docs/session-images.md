# Session images: a generation as a file

A session image is the whole state of a generation in one file: the model's
KV cache (and a recurrent model's fold), the tokens so far, the sampler with
its random state, the constraint, the budget, and the next token's logits,
closed by a SHA-256 over every byte. A generation suspended to an image and
resumed later, in another process, continues exactly as if it had never
stopped; a fork continues the same state under another seed.

Agent frameworks checkpoint the conversation and re-read it into the model on
resume. That re-prefill costs time on every resume and every branch, and it
is a re-computation, so the resumed run is close to the original rather than
the original. An image restores the computation itself.

## Two ways to use it

On the command line (CPU):

```sh
runner -m model.gguf -p "..." -n 400 -s 7 --suspend-after 150 --session-out s.img
runner -m model.gguf --resume s.img                     # the other 250 tokens
runner -m model.gguf --resume s.img --fork-seed 11      # or another continuation
```

From a server started with `--sessions DIR`, on the backend it serves from:

```sh
curl -s localhost:8080/v1/runner/sessions \
  -d '{"prompt":"...","max_tokens":400,"seed":7,"temperature":0.8,"suspend_after":150}'
curl -s localhost:8080/v1/runner/sessions/ID/resume -d '{}'
curl -s localhost:8080/v1/runner/sessions/ID/resume -d '{"fork_seed":11}'
```

The flags and fields are in the README's
[session images](../MANUAL.md#cli-session-images) section.

## The demo

`scripts/session-demo.py` starts a server with `--sessions`, runs a sampled
generation straight through as the reference, runs it again suspended after
40 tokens, resumes the image and compares, then forks the image eight ways,
keeps one and deletes seven:

```
1. straight run: 160 tokens
2. suspended after 40 tokens: image 680df4f16526ce6f...
3. resumed: IDENTICAL to the straight run (704 bytes)
4. eight forks: 8 distinct continuations; fork 3 asked again: the same text
5. kept fork 7, deleted seven, ran it to its end: 160 tokens
```

That was Llama-3.2-3B Q4_K_M on Metal on the 8 GB M1, 76 s for all of it; with
the default SmolLM2-135M it takes 6 s. None of the eight forks read the
prompt again: each starts from the image's KV.

## How exact, and where

The gate is byte identity: an image written at the end of a run straight
through and one written at the end of the same run suspended and resumed are
the same file (`tests/test_session_images.py`, under seeded sampling with a
repeat penalty, greedy decoding, `--json`, `--json-schema`, a Mamba-2 hybrid
and two suspensions in a row; `tests/test_server_sessions.py` for the
server). That holds for the same build on the same kind of host. An image
resumed by another build warns, and one carried to another backend (a Mac's
image on a CUDA card) is a continuation from the same state, not a promise
of the same tokens: the backends' arithmetic differs in the last bits, and a
near-tie can go the other way. A resume across machines is measured as its
own gate before it is claimed.

## With transcripts

A transcript ([record and verify](../MANUAL.md#record-and-verify-a-run))
proves what a model produced; an image lets the production continue. They
compose: the image's header carries the same model and binary digests a
transcript does, so a generation can be recorded up to a suspension, resumed
from its image, and the continuation recorded and verified in its turn. Not
yet joined into one chain: a transcript does not name the image it was
suspended into.
