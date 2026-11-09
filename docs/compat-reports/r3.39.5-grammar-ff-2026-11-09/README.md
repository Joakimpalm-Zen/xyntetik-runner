# Grammar fast-forward: identity, with schema-dependent cost

`RUNNER_GRAMMAR_FF` remains opt-in. Its class is identity, not tolerance:
grammar bytes propose tokens and the constrained target sampler decides every
token using exact decode arithmetic. A gain on one schema does not generalize
even to another schema on the same model and backend.

## Code evidence and gates

- `grammar_forced_bytes` in `src/engine.c` discovers bytes that uniquely keep
  the validator alive. `grammar_draft` tokenizes them, retains only a prefix
  that decodes back to those bytes, and withholds a nonfinal boundary token
  when another tokenization could merge it with following free bytes.
- `engine_generate_spec` samples the target's constrained logits at every
  position. A matching proposal advances the walk; the first mismatch becomes
  the next pending target token. A rejected draft is never emitted instead of
  the sampled token. RNG and validator updates follow the target's tokens.
- `model_forward_verify` enables `spec_exact` and `t_mv_exact`. CPU products
  use exact per-column decode dots, avoiding the ordinary wide f32 tile.
  Metal's `metal_forward_rows` executes solo decode steps in sequence; CUDA
  uses its exact multi-row decode kernels. Unsupported models decline the
  verify capability. The rollback/refold path also requests exact arithmetic.
- `tests/test_spec_identity.c` checks raw verify rows against solo steps over
  widths 2–16 and partial acceptance. `tests/test_grammar_ff.c` checks actual
  grammar engagement, output bytes, token IDs and sampler RNG, with and without
  a draft model, JSON mode, sampled schemas/JSON at seeds 1, 42 and UINT64_MAX,
  and unconstrained prose. Logprob capture remains excluded.

The grammar gate omitted `f16_init()`. Its first added assertion, binary16
`0x3c00` equals float32 `1.0`, failed before any model load. Initializing the
table fixes the test setup: fp16 KV reads must not consult its zero-filled
uninitialized storage. The strengthened dense and recurrent gates passed
locally. The binary16 constant is an external anchor; these identity tests
do not independently establish the model's numerical accuracy.

The historical opt-in comment in `engine_init` records CPU tests from
2026-07-30 at 33–38% acceptance and 4–12% slower. The measurements below use
the current exact verifier, rather than treating that older result as current.

## Bounded measurement

2026-11-09, Apple M1, macOS 27.0.1, two CPU threads, CPU or full Metal offload.
Production code at `5f45b203`, optimized default Makefile build, binary sha256
`f0e612929dafb2cf6b1e4c04fd5a9885cc061f0422fea59ec7f4d36ad1b0216b`.

Model: `bartowski/SmolLM2-135M-Instruct-GGUF`,
`SmolLM2-135M-Instruct-Q8_0.gguf`, sha256
`5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83`.
The existing local file was used; no model accuracy admission is claimed.

Three interleaved pairs per schema/backend, order off/on, on/off, off/on.
Greedy, seed 1, repeat penalty 1, context 256, prompt batch 32, ceiling 96.
Arithmetic switches use their defaults; this does not admit combinations with
experimental int8 or other arithmetic opt-ins.
The object completed in 18 tokens; the pinned-string object in 38. Throughput
is the CLI's generation-only rate, excluding model loading and receipt hashing.
These are short microbenchmarks on a shared system, not a broad workload study.
All 24 runs completed with pressure 1 under the shared guard and a 250 ms
pressure monitor; no swap growth was reported.

| Backend | Schema | Plain median tok/s | FF median tok/s | FF / plain | Accepted / proposed per FF run |
|---|---|---:|---:|---:|---:|
| CPU | two-field object | 137.81 | 151.18 | 1.097x | 7 / 8 |
| CPU | pinned string | 128.45 | 71.52 | 0.557x | 16 / 103 |
| Metal | two-field object | 195.76 | 183.28 | 0.936x | 7 / 8 |
| Metal | pinned string | 165.81 | 52.41 | 0.316x | 16 / 103 |

Every arm produced identical token IDs and output bytes within its case;
Python's JSON parser checked the output and required keys. The pinned string
still has alternative tokenizations, so its proposals can be rejected despite
having no choice of output text. Metal pays for each proposed row as a solo
step. CPU gains in the object case do not justify a model/backend default
alongside the pinned-string regression. There is no default promotion and no
tolerance route needing a new float32 quality row. Linux/Windows prefetch is
outside this audit.

`runs.jsonl` holds each run's rate, token count, grammar acceptance and token
and byte digests. `measure.py` contains both synthetic schemas and all options:

```sh
python3 docs/compat-reports/r3.39.5-grammar-ff-2026-11-09/measure.py \
  ./runner MODEL.gguf out-grammar-ff
```

On a shared system use its resource guard and stop at memory pressure 2.
The script also imposes a 180-second overall bound and 40 seconds per run.
