# Contributing

Bug reports are most useful with `runner --version`, `runner --caps`, the
model filename/hash, the command or request, and a small reproducer. See
[the security policy](SECURITY.md) for sensitive reports.

## Build and verify

```sh
make
make smoke
make test
make debug
```

On Windows, build in MSYS2 UCRT64 with `make CC=gcc`. The Makefile
recognises Windows from the shell's `uname`, so no `OS=Windows_NT` is needed,
even in a bash started directly with a minimal environment.
`make test` runs the aggregate C, Python-client, and CLI/API regression
checks; it requires pytest. The sanitizer build is separate. See
[CI execution and device verification](docs/ci.md) for Linux, Windows,
macOS, T3, fuzzing, sanitizer, and owned-hardware lanes. A skipped device
check is not evidence that a device path works.

Read [AGENTS.md](AGENTS.md) before changing the repository. Protect the
smallest observable behavior first, implement it through the real layers,
and keep refactoring separate from behavior changes.

## Correctness contracts

- Pin the certified scalar/eager routes when checking CPU/GPU identity.
  Tensor-core, fused-routing, quantized-KV, and other reassociated routes
  have their own established numerical gates; do not widen tolerances.
- Compare affected model behavior with the publisher's float32 reference
  and template/tokenizer artifacts. An engine-to-engine agreement alone
  does not establish reference correctness.
- Keep deterministic training, adapter identity, receipts/replay, cache
  isolation, unload/reload, shared ownership, and failure cleanup covered.
- Preserve declared schema key order, valid truncation closure, native
  tool protocols, streaming event order, and every API's validation rules.
- Run the required platform/configuration checks for the changed paths.
  Register tests in executable Makefile/CI entries; comments do not count.

## Documentation and releases

The [README](README.md) is the front page; [MANUAL.md](MANUAL.md) is the
CLI/API/support reference. Record a README impact decision with each
change. Update the manual and site sources when their account changes;
keep public claims tied to accessible evidence and actual current behavior.
Completed investigations and superseded run output are not current usage
instructions. Keep useful numerical constraints and regression fixtures.

For a release, update `RUNNER_VERSION` in `src/runner.h`, the README/manual
version account, and `python/pyproject.toml` together. The Python version
uses PEP 440 spelling. `SECURITY.md` links the supported release policy
rather than duplicating a version number. Run `make release-check` and
record the required real-device evidence before tagging; the Python client
remains separately distributable with no build-time header coupling.

## Architecture and style

Use plain C11/gnu11 and the existing platform/backend languages. Keep
public interfaces small, ownership explicit, and numerical primitives
shared. Compose supported attention, routing, cache, and topology behavior
instead of copying a forward engine per family. Recurrent and MTP paths
already exist; a new architecture or layout still needs explicit admission,
shape validation, independent anchors, and tiny malformed-input fixtures.
Unknown or unsupported layouts fail closed. See the current
[model scope](docs/model-scope.md) and [support matrix](MANUAL.md#support-matrix).
Comments explain constraints that code cannot show.

Run `make hooks` to install the commit-hygiene hook. Commit messages and PR
text must follow AGENTS.md, including the required co-author form and the
prohibition on conversation/session identifiers. Use a work branch and CI
review; do not treat a passing check as permission for an otherwise
unapproved merge.
