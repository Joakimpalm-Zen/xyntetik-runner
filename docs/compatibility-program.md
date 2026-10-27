# Compatibility checks

Compatibility is recorded per model artifact, execution route, and API
consumer. A report must name the model hash, exact build/reference,
configuration, hardware, commands, and checks actually executed. Missing
models, devices, reference assets, or dependencies are not passes.

## Model and reference checks

`tests/compatibility/models.json` pins real GGUFs and their reference/check
parameters. Model files are not committed. The coverage inventory also
records admitted layouts with no pinned artifact or independent anchor;
being listed by `--info caps` does not certify a model automatically.

```sh
python3 scripts/compat_matrix.py --verify-files --load   --reference /path/to/llama-cli   --out tests/compatibility/out/model-load.json
```

Select the intended model/check/backend with the harness's `--help` options.
It verifies file identity before running declared load, tokenizer, greedy
reference, CPU/GPU, chat/tool, context, and performance checks. Current
release/device records live in [compat-reports/](compat-reports/) and the
[device ledger](device-evidence.json). New local output goes to
`tests/compatibility/out/`; generated output is not a regression fixture.

The publisher's float32 implementation is the primary numerical anchor.
`scripts/gold-logits.py` and the [published reference comparison](golden-pass-2026-09-07.md)
state dtype, attention implementation, device, tokenizer/model hashes, and
covered positions. llama.cpp is the wider quantization/reference oracle.
Report tie-classified divergences beside strict greedy counts, and state
where a publisher's tokenizer artifacts disagree with each other.

## Exact and tolerance routes

Exact CPU/GPU checks pin scalar/eager execution: `RUNNER_CUDA_TC=0`,
`RUNNER_MOE_EAGER=1`, and the applicable Metal matrix pin
`RUNNER_METAL_MM=0`. Build flags, compiler/libm, and cache state belong in
the record. A different build is a different numerical configuration.

Default tensor-core/matrix, fused-routing, and other reassociated paths
are held to their established type/shape/model tolerance gates. A scalar
identity result does not certify a faster route; a tolerance result does
not authorize widening the scalar identity contract. See
[performance routes](performance.md) and the manual's backend tables.

## Tokenizers and templates

Tokenizer checks use the committed 721-string corpus and immutable
publisher assets. Install the tokenizer requirements and use
`scripts/difftok.py --help` for `--capture`, `--ref-revision`, and `--ref-ids`.
A reference-ID capture can be replayed offline and refuses a different
corpus. The four pinned captures under
`tests/compatibility/tokenizer-references/` are intentional regression inputs.

Mistral v0.3, Phi-3.5, and Salamandra gate against their published
SentencePiece models; tokenizer.json comparisons are a separate
informational column. Lucie's custom-normalizer difference remains a
separate limitation. The [SentencePiece evidence](tokenizer-spm-2026-10-03.md)
records those distinctions. Do not replace reference IDs with Runner's own
output to make a tokenizer gate pass.

`scripts/template-conformance.py --require-tokens` checks rendered prompts
against pinned publisher templates/tokenizers. Text-only checks and
accounted-for deviations do not establish token identity for every family.
Keep template baselines, allowlists, and coverage declarations deliberate.

## API and library consumers

```sh
python3 -m venv .compat-venv
.compat-venv/bin/pip install -r tests/compatibility/requirements.txt
npm ci --ignore-scripts --prefix tests/compatibility/node
.compat-venv/bin/python scripts/consumer_compat.py
```

`make compat-consumers` runs the consumer check after dependencies are
installed. The OpenAI/Anthropic Python and Node SDKs, LiteLLM, LangChain,
and separately pinned AI SDK smoke exercise real response builders/parsers.
CI must run mandatory consumer tests rather than treating missing packages
as compatibility. [CI](ci.md) documents the current execution lanes.

[Agent setup](agent-compatibility.md) describes endpoint choices and current
limits. [The agent torture matrix](agent-torture.md) checks constrained
calls, truncation, tool selection, and transport normalization with fixed
requests. Full client-loop evidence needs the installed client, not only a
replayed capture or compatible JSON shape.

## Evidence ownership

Keep current coverage dispositions and public evidence for retained claims.
The inventory in `tests/compatibility/coverage.json` names executed protocol
and client-loop records and separates PASS, FAIL, NOT RUN, and UNSUPPORTED.
`tests/test_coverage_inventory.py` and `scripts/check-evidence-paths.py`
require the referenced evidence to exist. Reference hashes, frozen labels,
and numerical fixtures remain inputs; superseded generated run output is
not part of the executable test suite.
