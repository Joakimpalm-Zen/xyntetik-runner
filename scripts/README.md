# scripts/ - what each tool is for

Development and verification tooling. Nothing here is needed to *run* the
runner; several are load-bearing gates for changing it.

## Verification gates (use these before trusting a kernel/perf change)

- **`kernel-verify.py`** - demands **token-identical** greedy output between a
  baseline and a candidate binary on 5 prompts. A faster binary that changes
  tokens is a regression, not a win.
- **`kernel-bench.py`** - prefill/decode tok/s as JSON for one binary+model.
- **`template-conformance.py`** - byte-exact comparison of runner's native C
  chat renderer against each family's upstream jinja template, the gate whose
  absence let five families drift unnoticed. `make template-conformance`.
  Intentional deviations live in `template-conformance-allowlist.json`, each
  carrying a source citation the gate re-verifies and fails on when it rots;
  known differences awaiting fixes live in
  `template-conformance-baseline.json`, which should only ever shrink. Exit 2
  means NOT CHECKED (no jinja2, no network, no oracle) and is never a pass.
- **`compat_matrix.py`** - verify pinned real-model hashes and emit architecture
  load/inference evidence without committing the GGUFs.
- **`consumer_compat.py`** - run pinned OpenAI/Anthropic SDK, LiteLLM and
  LangChain clients against one live Runner and emit a JSON report.
- **`reference_compare.py`** - compare Runner and llama.cpp exact greedy text
  through equivalent raw OpenAI Completions requests.
- **`compare_llamacpp.py`** - reproducible Runner-vs-llama.cpp evidence
  harness for the same GGUF, prompt, context and sampling settings; emits JSON
  plus a provenance-complete Markdown summary, quantifies common-token
  top-logprob deltas where supported, and marks real results pending when run
  in fixture mode. `--endpoint label=url=model` additionally measures an
  already-running OpenAI-compatible daemon (Ollama, LM Studio) on the same
  prompt and budget - throughput only, and labelled as such, because the
  correctness gate is defined against the pinned llama.cpp reference and those
  runtimes do not expose comparable completion logprobs.
- **`stress-models.py`** - run every GGUF on a machine's shelf against this
  build: load, generate, CPU-vs-CUDA identity, fault detection (fallbacks,
  kernel-launch failures, refusals, timeouts) and a settings sweep, with one
  resumable JSON per model. Identity is a PREFIX test, not string equality -
  the legs may use different token budgets, and comparing truncated character
  slices reports a mismatch on the trailing newline alone.
- **`stress-context.py`** - the context/KV edges: auto-fit, a context the
  machine cannot hold, and whether a refusal says why.
- **`cpu_cuda_check.py`** - the compatibility program's `cpu_cuda` check for
  one model: greedy CPU output must be byte-identical to greedy CUDA output
  over several prompts, with the eager router pinned (MoE byte identity is
  defined over that path).
- **`cuda-smoke.py` / `cuda-smoke-remote.sh`** - the CUDA smoke gate CI cannot run
  (it has no GPU): `--caps` coherence on real hardware, the invariant that
  caught the v0.4.2 integrated-memory misreport, and a served generation. The
  remote driver runs it on the lab's Windows RTX 3070 box and brings the JSON
  report back for `docs/compat-reports/`; run it before tagging a release.
- **`certify-envelope.py`** - assemble a measured-envelope manifest
  (`<model>.envelope.json`) from evidence that already exists (`--caps`, a
  compat report, gate outputs). It measures nothing itself; it is an index
  over evidence, and the runner only ever reads the result.
- **`idle_coexistence.py`** - what a resident server costs the machine while it
  serves nothing: a four-state lifecycle (loaded idle, post-request idle, after
  unload) reporting CPU seconds, RSS, threads and, on macOS, system wired
  memory. Run one engine at a time on a quiet machine; the README's
  coexistence tables come from it.
- **`help-parity.py`** - mechanical parity between `runner --help` and the
  README's option reference (every flag the binary accepts is documented, and
  nothing documented has left the binary). Part of `make test`.
- **`moe-mm-flips.py`** - routing-flip account for the grouped-MMA MoE prefill
  path: how often discrete top-k routing flips at a near-tie under the
  kernel's reassociation, and at what margin. The reason that path is judged
  by the fidelity bar rather than byte identity.
- **`nvfp4-probe.py`** - validate a GGUF's NVFP4 layout against the file's own
  structure, written because the format shipped from a specification with no
  model to test against and its unit test could not tell.
- **`kv-quality.py`** - KV-cache quality gate: compares q8 KV against f16 on
  teacher-forced logits (the deeper version of `tests/test_kv_tol.c`'s gate).
- **`verify-gguf.py`** - structural sanity check of a GGUF file (metadata,
  tensor table, offsets) without loading it into the engine.
- **`check-release.py`** - release artifact consistency gate: binary
  `--version` against the tag, the README's release strings, a CHANGELOG
  section, BUILD-INFO's tag/commit lines, the Python package version, and the
  compat report the tag must publish. Unit-tested by
  `tests/test_release_check.py`; run by the release workflow.
- **`check-generated.py`** - drift gate for the committed generated GPU
  headers (`src/kernels_metal.h`, `src/kernels_ptx.h`). Re-embeds each from its
  committed source and compares, so a `kernels.metal` edit cannot ship while
  the binary still embeds the old bytes. Pure Python: no Metal or CUDA
  toolchain, runs in well under a second. Run it after touching a shader.
- **`cert-greedy-identity.py`** - six-prompt greedy-identity gate against a
  pinned llama.cpp reference (four domains at 64 tokens, two repeated at 256,
  pure-greedy sampler and `cache_prompt:false` on the reference side).
- **`afmoe-cert-gate3.py`** - the same identity protocol as originally written
  for the afmoe certification against llama.cpp b10280, kept replayable.
- **`claude-code-e2e.sh`** - the real Claude Code binary pointed at Runner with
  `ANTHROPIC_BASE_URL`, completing a built-in `Read` tool loop. Turns the
  README's Claude Code compatibility claim into something that can be re-run.
- **`competitor-freshness.py`** - checks the runtime versions in published
  torture results against upstream releases. Metadata-only registry requests;
  a network failure is a skip, never a red scheduled run.
- **`template-conformance-harmony.py`** - the Harmony oracle used by
  `template-conformance.py`: gpt-oss is compared against `openai-harmony`
  rather than the GGUF's jinja, which is a reimplementation with known gaps.
  Its committed goldens live in `template-conformance-harmony-oracle.json`.
- **`template-conformance-render.c`** - the runner-side driver for that
  harness: reads a JSON job list, writes each case's native C render, so the
  Python side can diff byte-for-byte.

## Benchmarks

- **`batch-bench.py`** - concurrent-serving throughput + single-request latency
  against a running server (the numbers behind the batching phase work).
- **`bench.sh`** - thin wrapper for repeated `--bench-json` runs.
- **`agent-torture.py` / `torture-compare.py`** - the adversarial tool-call
  matrix and its cross-runtime report comparator (see `docs/agent-torture.md`).
- **`truncation-benchmark.py`** - the token-ladder probe behind the "tool calls
  survive the token limit" headline: spawns Runner (or targets a competitor
  `--endpoint`) and records whether each rung yields a parseable `tool_calls`
  entry. `make test-truncation` is the regression gate; see
  `docs/truncation-benchmark.md`.
- **`weight-io-bench.py`** - how this machine's storage serves large weight
  reads: mmap page faults against explicit reads, which is a property of the
  storage and the OS rather than of the model, and can change a design call.
- **`write-stall.py`** - the real write-side socket stall, with a large model
  and context rather than the tiny fixture: enough SSE to fill the loopback
  buffers against a client that never reads. A local experiment, not a CI test.

## Measurement and analysis

Nothing here gates a merge; these are how a claim gets a number behind it.

- **`quant-fidelity.py`** - how tool-calling and structured-output fidelity
  degrade across one model's quantization ladder: schema conformance, tool
  selection and argument agreement against a pinned reference variant, one
  server at a time.
- **`kld-compare.py`** - approximate KLD between two runner-served models'
  next-token distributions over a pinned corpus, via chat-completions
  `top_logprobs` (exact 0 for identical distributions, monotonic with real
  divergence; not the full-vocab KLD).
- **`kld-compare-raw.py`** - the same comparison over RAW completions, which is
  the sound one ACROSS engines: a chat-endpoint comparison of gpt-oss finds the
  two engines' templates disagreeing rather than their weights.
- **`token_divergence.py`** - where a greedy run first leaves llama.cpp's, and
  by how much: the logprob gap at the first differing argmax separates a
  near-tie from a real arithmetic bug, which a text diff cannot.
- **`sensitivity_floor.py`** - how much a given model amplifies a small numeric
  change, measured against itself (f16 vs q8 KV). Measure the floor before
  calling a cross-engine gap a bug.
- **`gen-quality-metrics.py`** - collapse detection over generated text:
  n-gram repetition runs, blank fraction, length distribution. Stdlib only.
- **`make-tooluse-data.py` / `eval-tooluse.py`** - the deterministic
  tool-calling adaptation dataset (four tools, combinatorial requests, refusal
  cases) and its greedy held-out eval (JSON parses / right tool / schema-shaped
  args / exact call), the pair behind the published precision study.
- **`train-grpo-lite.py`** - GRPO-lite reinforcement fine-tuning with zero
  train/infer mismatch: sample K completions per prompt from the runner
  itself, score them, turn group-relative advantages into weighted `--train`
  examples. Seeded and replayable.
- **`cl-calibration.py`** - calibration report for `choice_logprobs` decision
  records: when a constrained decision point says 0.8, is it right 80% of the
  time?
- **`tool-choice-boundary.py`** - the UNLABELED tool-choice decision-boundary
  lane over `choice_logprobs`: runs a bank of ambiguous tool-choice prompts
  under several serving conditions (base, adapters) and records the grammar
  decision at which the tool is chosen, then `summarize` reports where the
  conditions disagree and how tight the margins were. No gold labels, no
  accuracy, by design; its records cannot feed `cl-calibration.py`. Ships
  with John6666's 36-prompt bank (`tool-choice-boundary-bank.jsonl`, MIT OR
  Apache-2.0, carried byte-identical). See `docs/tool-choice-boundary-lane.md`.
- **`classify-grammar-trace.py`** - classifies grammar-draft rejection causes
  from a `RUNNER_GRAMMAR_TRACE` JSONL (tail-straddle, coarse-merge, fine-split,
  seam).
- **`moe-prune-plan.py`** - turns `RUNNER_MOE_TRACE` JSONL into a
  `--prune-experts` plan (and a hot/cold precision plan), ranking experts by
  gate mass or gate x activation-norm saliency, uniform or coverage-based.

## GGUF tools

- **`gguf-inspect.py`** - a file's real per-tensor-class quant types rather
  than what its filename claims (an MXFP4 expert bank inside a "Q4_K_M" file is
  the normal case). Reads only the header and tensor directory.
- **`type-plan-size.py`** - what a `--type-plan` will actually produce, from
  the header alone: the exact output byte size, the per-type histogram the
  quantizer will print, and every rule it will silently DECLINE (a K-quant
  target whose 256-wide block does not divide the row, or a target that would
  grow the tensor). A plan's rules are first-match substrings, so
  `{"match":"output.weight"}` also claims every `attn_output.weight`; this
  answers that before an hour of quantizing a 30B parent does.
- **`gguf-split.py`** - split a GGUF into llama.cpp-compatible shards, for
  testing multi-part loads; layout follows upstream `gguf-split` verbatim.
- **`gguf-depth-slice.py`** - drop whole transformer layers and keep everything
  else: surviving layers keep their on-disk bytes, `blk.N` is renumbered
  densely, and per-layer metadata arrays are filtered to the survivors.
- **`gguf-drop-shared-kv.py`** - rewrite a Gemma-4 E-series GGUF in the compact
  shared-KV form: the layers at or past `block_count - shared_kv_layers`
  compute no K/V, so their `attn_k`/`attn_v`/`attn_k_norm` are dropped (E4B:
  666 tensors instead of 720, the shape every quantized export already ships).
  The layer set comes from the file's own metadata, not from the command line,
  and everything else is copied through byte for byte - so a correct engine
  must score the two files identically.
- **`make-bf16-fixture.py`** - rewrite a GGUF's F16 tensors as BF16 in place to
  get a valid BF16 fixture; both types are 2 bytes, so every offset stays put.

## Fixtures and codegen

- **`make-test-model.py`** - builds `test.gguf`, the tiny CI fixture (plus
  malformed variants for the rejection tests).
- **`make-test-moe.py`** - the dense + sparse-MoE GGUF trio for the MoE
  equivalence test (`make test-moe`; see `docs/moe-support.md`).
- **`make-test-ornith.py`** - tiny Qwen3.5/Ornith hybrid fixture
  (via `tests/test_ornith_cpu.py`).
- **`make-vocab-fixture.py`** - tokenizer vocab fixtures in `tests/fixtures/`.
- **`tokenizer-corpus.py` / `difftok.py`** - regenerate / diff the 721-string
  tokenizer conformance corpus against Hugging Face reference tokenizers;
  revision-bound `--capture` files make the reference side replayable offline.
- **`ornith-reference.py`** - reference layout contract for Ornith GGUFs
  (unit-tested by `tests/test_ornith_reference.py`).
- **`embed-ptx.py`** - embeds `src/kernels.ptx` into `src/kernels_ptx.h`
  (invoked by `make ptx`).
- **`embed-metal.py`** - same embedding step for the Metal shader source into
  `src/kernels_metal.h` (run manually on a Mac when `kernels.metal` changes;
  there is deliberately no Makefile target on non-Mac hosts).
- **`conformance.sh`** - drives the agent-protocol conformance suite in CI.

## Fixture generators and one-off reproducers

- **`make-test-hybrid.py`**, **`make-test-nemotron.py`**, **`make-test-lora.py`**
  - tiny GGUF fixtures for the Mamba-2 hybrid (`granitehybrid`), `nemotron_h`
  and LoRA-adapter paths, beside `make-test-model.py`, `make-test-moe.py` and
  `make-test-ornith.py`; `make test` builds them.
- **`embed-metal-tensor.py`** - embeds the separately admitted Metal 4 tensor
  source, the sibling of `embed-metal.py`; `check-generated.py` guards both.
- **`repro-startup-signal.py`** - reproducer for the once-reported
  SIGTERM-during-startup race that `test_signal_during_startup` chases.
- **`bench.sh`** - three greedy decode runs, median tok/s plus output md5s that
  must match the baseline. **`conformance.sh`** - the agent-protocol
  conformance harness in one command. **`claude-code-e2e.sh`** - Claude Code
  against a served runner end to end, the executable form of the README's
  compatibility claim.
