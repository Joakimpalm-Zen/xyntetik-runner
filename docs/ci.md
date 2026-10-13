# CI coverage and execution

`make test` remains the aggregate engine gate. CI runs it on Linux for pull
requests and on Windows/macOS through the full suites, which run on every
engine push to main, on the daily schedule and on dispatch. The smaller
native platform smokes still run on pull requests. The full Windows/macOS
lanes can be dispatched before merging a platform-sensitive change. No
numerical test or tolerance is removed to shorten the PR path.

Superseded PR runs are cancelled. Main and scheduled runs have distinct
concurrency identities. Jobs and network smokes have explicit deadlines;
failed shell smokes clean up their background servers. Tests that must
demonstrate graceful shutdown fail if they escalate to a forced kill.

## What a run decides first

The `changes` job runs ahead of everything else and answers two questions.

- **Did the change touch the engine?** A change confined to `site/`,
  `docs/`, `README.md` and `CHANGELOG.md` skips every engine job in seconds.
  This is a job-level skip, not a workflow path filter: a workflow that never
  starts leaves its required checks pending and blocks the merge, while a
  skipped job satisfies them. `metadata.yml` still checks the prose.
- **Was this tree already green?** A pull-request run tests the merge of the
  branch into main and records the tree it tested as an empty artifact named
  `tested-tree-<hash>`. A push to main whose merge produced the same tree
  (main did not move between the run and the merge) finds that artifact on a
  successful run of the merge's second parent and skips the engine jobs; the
  full Windows and macOS suites still run, because the pull request never ran
  them. A direct push, or a merge after main moved, runs everything.

`main` is protected: the pull-request checks are required, including
`changes` itself (so a failure there blocks the merge rather than skipping
the rest), the rules apply to administrators, and force pushes are refused.
A red pull request cannot be merged; a docs-only one is merged on skipped
checks.

## Timing and test execution

`scripts/ci-run.py NAME -- COMMAND ...` retains elapsed time and exit status
under `.build/ci/`, including failures and timeouts. Pytest reports its slow
tests; the main root suite and dedicated integration lanes emit JUnit XML.
`scripts/check-junit.py --no-skips FILE` rejects an empty, failed, or skipped
mandatory lane. Platform/device-dependent checks in the broad suite still
report their scoped skips rather than claiming device coverage.

`scripts/check-ci-roster.py` checks that new root test files have an
execution entry and that the fuzz job groups cover every registered target
exactly once. This inventory check does not turn runtime skips into passes.

The SDK lane installs the pinned Python consumers and Vercel dependencies,
then explicitly runs the SDK conformance modules. It must execute those
tests; a missing dependency is not a successful compatibility check.

To inspect actual Actions bottlenecks with the existing GitHub CLI login:

```sh
python3 scripts/ci-timings.py --limit 20 --out .build/ci-history.json
```

The report separates events, job/step execution and scheduling delays. Use
these measurements before sharding more tests. Many tests share generated
files or servers; unqualified pytest worker parallelism is not safe. Engine
builds reuse the existing compiler/flag-keyed objects, while sanitizer
builds compile every translation unit with their own flags.

## Sanitizers and deterministic execution

`make debug` instruments quantization as well as inference with ASan and
fatal UBSan. `scripts/check-sanitizers.py` verifies both instruments against
known invalid C operations. The CLI transformation/lifecycle lane uses
that binary; the shared-ownership ASan gate exercises repeated release and
reuse. Server conformance disables leak detection separately from the CPU
CLI lane. TSan runs tokenizer, shared-pool and registry/shutdown tests in
separate processes; it is not combined with ASan.

The T3 lane builds the actual configured executable, checks its flavor,
and tests generation records, training and session continuation. The
CUDA-only training case belongs to device verification, not a hosted CPU
T3 job.

## Fuzzing

The PR fuzz jobs run bounded libFuzzer mutation in two groups. Scheduled
runs allocate five minutes per target, minimize successful discoveries and
export source coverage. Corpora are restored across runs; only main-history
runs save shared cache entries. Artifacts retain inputs, crashes, sanitizer
logs and timings. Findings live outside the mutation input directories, so
ASan logs and timeout artifacts never become ordinary corpus seeds. Corpus discoveries are not automatically committed:
minimize and review a failure before adding its regression case.

Targets cover JSON/schema state, HTTP parsers, GGUF parsing, bounded CPU
model loading, multipart GGUF sets and tokenizer behavior. The model-loader
target deliberately bounds geometry and focuses on Llama/Qwen2; the raw
parser and hostile-geometry tests retain extreme-input coverage. Its exit
statistics report whether inputs reached binding and successful loading.
HTTP socket-fragment/EOF/limit behavior is tested against the actual server
by the conformance suite, alongside the in-memory parser fuzzer.

```sh
make fuzz FUZZ_TIME=30
python3 scripts/fuzz-seeds.py fuzz-corpus
make fuzz-replay-tokenizer
./fuzz-replay-tokenizer tests/fuzz/corpus/tokenizer/*
```

The replay executable is for sanitizer-capable Clang installations without
the libFuzzer runtime. It replays the identical harness, but does not claim
coverage-guided mutation. Source coverage requires matching Clang,
llvm-profdata and llvm-cov tools. AFL++ and additional persistent-artifact
fuzz targets can supplement this set after coverage identifies useful gaps;
they are not substitutes for the existing targets.

## Device evidence

Hosted runners have no GPU. The device classes the README claims are
verified on lab machines and recorded in `docs/device-evidence.json` by
`scripts/device-evidence.py --record` from the runner's own `--caps` output;
`scripts/check-release.py` refuses a tag while any class is stale, failing or
never recorded (`docs/compatibility-program.md`). `scripts/cuda-gate.sh`
is the CUDA gate a lab box runs from a fresh build, and
`scripts/check-ptx-source.py` compares regenerated PTX against the committed
header with the pinned nvcc; both are run by hand on the box and their
records committed. There is no hosted or self-hosted device workflow.

`metadata.yml` runs on documentation-only changes too, checking generated
source/document agreement and the existence of referenced evidence files.
These checks do not assert that historical evidence has been remeasured.
