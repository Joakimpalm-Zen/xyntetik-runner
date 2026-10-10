# CI coverage and execution

`make test` remains the aggregate engine gate. CI runs it on Linux for pull
requests and on Windows/macOS through the scheduled or manually dispatched
full suites. The smaller native platform smokes still run on pull requests.
The full Windows/macOS lanes can be dispatched before merging a
platform-sensitive change. No numerical test or tolerance is removed to
shorten the PR path.

Superseded PR runs are cancelled. Main and scheduled runs have distinct
concurrency identities. Jobs and network smokes have explicit deadlines;
failed shell smokes clean up their background servers. Tests that must
demonstrate graceful shutdown fail if they escalate to a forced kill.

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

## Controlled hardware verification

`candidate hardware` is manual and accepts only a full SHA reachable from
this repository's main history. It never executes pull-request code
automatically on a lab machine. Provision dedicated `runner-ci` runners
with the `blackwell`, `windows-cuda`, or `apple-metal` label, and configure
reviewers for the `runner-hardware` environment before enabling dispatch.
Windows needs a Bash/MSYS2 native build environment. The chosen runner
needs the existing quantization fixtures/toolchain and a readable small
real-model path in the `RUNNER_GATE_MODEL` environment variable (the
environment/repository variable can supply a default).

`scripts/hardware-evidence.py` refuses absent devices or CPU fallback and
records the exact source, executable, gate and model digests with the
executed checks. Additional backend regression gates remain required.
These records complement the historical device-evidence ledger; an old
ledger row is not an exact-candidate pass. Pre-merge device testing can use
the same script in an explicitly approved clean task checkout without
opening self-hosted PR execution.

Blackwell also runs `scripts/check-ptx-source.py` with the nvcc version
embedded in the committed header. It compares regenerated PTX after the
existing version down-pin and comment/whitespace normalization. A compiler
mismatch or changed instructions fails rather than silently regenerating
shipped kernels. Reviewer-controlled device access and matching toolchains
are external prerequisites, not resources provisioned by this repository.

`metadata.yml` runs on documentation-only changes too, checking generated
source/document agreement and the existence of referenced evidence files.
These checks do not assert that historical evidence has been remeasured.
