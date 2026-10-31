# A snapshot test for a model's output

A test that calls a language model usually cannot assert on the text, because
the text changes from run to run. With Runner it does not: the same binary,
model file, prompt and settings give the same tokens, and `--verify` replays a
recorded run and says whether they came out again. So a model's answer can be
a snapshot, checked in CI like any other.

This directory is a working example you can copy:

- `snapshot.json`: a record of one greedy 40-token completion by
  SmolLM2-135M-Instruct (Q8_0, 145 MB, Apache 2.0), written by
  `runner --record` with the v1.9.0 release on macOS arm64.
- `check.sh`: fetches the pinned Runner release and the pinned model file,
  refuses either if its sha256 differs, and replays the record.
- `.github/workflows/example-deterministic-ci.yml` (at the repository root):
  the same check on Linux x86_64 and macOS arm64, on every change to this
  directory and once a week.

```sh
examples/deterministic-ci/check.sh             # VERIFIED, exit 0
examples/deterministic-ci/check.sh --record    # write a new snapshot.json
```

## What a pass means, and what it does not

`VERIFIED` (exit 0) means the model produced the recorded tokens again from the
recorded prompt and settings, on a model file with the recorded sha256.
`DIVERGED` (exit 2) names the first token that differs. `UNVERIFIABLE`
(exit 3) means the record could not be replayed: a different model file, for
instance.

The tier says how strong the match is ([determinism scope](../../docs/determinism-scope.md)):

- **T1, same binary:** the replay used the binary that wrote the record, and
  the output bytes match exactly. That is the macOS job here.
- **T2, another build:** a different binary, here the Linux x86_64 release
  replaying a record written on macOS. The token ids must match; arithmetic
  inside the model may differ in the last bits. Under greedy decoding a token
  can flip only where the model's two best candidates are that close, so a
  prompt whose answer sits on such a near-tie is a poor snapshot.

A pass says the output is the recorded one. It says nothing about whether
the recorded output is right: review the snapshot when you record it, as you
would any other snapshot.

## Adapting it

Change `PROMPT`, the model pins (`MODEL_URL` with a revision, `MODEL_SHA`)
or the Runner pins (`RUNNER_TAG`, the per-platform asset digests from the
release's `SHA256SUMS`), then run `check.sh --record` and commit the new
`snapshot.json`. Keep the pins exact: a moving model URL or an unpinned
release turns a failing snapshot into a question of which input changed.
For several prompts, keep one record per prompt.
