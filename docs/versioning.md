# Versioning: what 1.x keeps

Runner follows semantic versioning from 1.0.0. A change that breaks what this
page lists waits for 2.0.0; a 1.x minor release adds, a 1.x patch release
fixes. Everything below is held to that rule, and the last section names what
is outside it.

## Kept stable across 1.x

**Command line.** Every flag `runner --help` documents keeps its name and its
meaning. A flag can gain values or be joined by new flags. A flag that has to
go is first marked deprecated in `--help` and the CHANGELOG, keeps working
with a warning for the rest of 1.x, and is removed no earlier than 2.0.0.

**Exit codes.** The documented exit codes keep their meaning, including the
`--verify` verdicts `VERIFIED`, `DIVERGED` and `UNVERIFIABLE`.

**HTTP API.** The endpoints in the manual's
[endpoint table](../MANUAL.md#endpoints) stay, and so do the request fields
they accept. A response field keeps its name, type and
meaning; new fields can appear in any 1.x release, so a client must ignore
fields it does not know. This covers the OpenAI-compatible surfaces (Chat
Completions, Completions, Responses, Embeddings), Anthropic Messages, and the
Runner endpoints under `/v1/runner/`, `/v1/decide`, `/health` and `/metrics`.
`runner_telemetry` is additive in the same way.

**Record formats.** Every record Runner writes carries a `schema_version`
such as `xyntetik.runner.verify.v1` or `xyntetik.runner.evidence-pack.v1`. A
1.x release keeps reading and checking every format an earlier 1.x release
wrote. A format that has to change gets a new identifier (`.v2`) and the
old one is still read. A receipt, transcript or evidence pack written by
1.0.0 therefore still verifies under a later 1.x build, given the binary
and model the record names (see "Replay" below).

**Model files.** A GGUF architecture and quantization format that 1.0.0 loads
keeps loading in every 1.x release. A model can be added; one is not dropped.

**Clients.** The public API of the Python client (`xyntetik_runner`) and the
TypeScript client keeps its names and signatures in the same way as the HTTP
API.

## Bound to the exact build, by design

These were never cross-version promises, and 1.0 does not make them one:

- **Replay.** `--verify` replays a record on the executable and model it
  names. Exact agreement is a property of that build on the documented
  execution paths ([determinism scope](determinism-scope.md)); a later
  release may change kernels, and a record made with it must be replayed
  with it.
- **Session images and context snapshots** resume exactly on the binary that
  wrote them; the image records that binary's sha256.
- **Measured envelopes** describe the exact version and backend they were
  measured on. The load-time envelope gate matches the version exactly, so a
  manifest measured on 1.0.0 is evidence for 1.0.0 and indeterminate on
  1.0.1 until it is measured again.

## Not covered

- **Generated tokens can change between releases.** A sampler or kernel fix
  can change what a seed produces. The CHANGELOG says so when a release does
  this on purpose.
- **Speed, memory figures and benchmark ratios** move with the work. The
  published tables are dated measurements, not commitments.
- **Certification and the support matrix** follow the evidence: a model's
  verdict can change when a re-measurement says it should, and that change
  is published.
- **`RUNNER_*` environment variables** are tuning and diagnostic switches
  and can change in any release, including those the README mentions.
- **Log and progress text** on stderr is for people, not parsers. Use the
  JSON outputs (`--caps`, `--doctor`, `--bench-json`, `--tool-info`,
  `runner_telemetry`) instead.
- **Build internals**: Makefile targets other than `make` and `make test`,
  test fixtures and the `scripts/` directory.

## Support

The latest 1.x release is supported. Fixes land in the next release; there are
no backports to earlier releases. Security reports follow
[SECURITY.md](../SECURITY.md).
