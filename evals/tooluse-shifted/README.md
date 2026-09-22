# Tool-use Evaluation with Shifted Distribution (R8.4)

This evaluation measures whether a trained tool-use adapter can be distinguished from the base model on a hand-written prompt set with distribution shift from the original training templates.

## Overview

The evaluation has three components:

1. **Catalog** (`catalog-v1.json`): 23 tools with deliberately overlapping names and argument shapes. Example tool families:
   - search: `search_files`, `find_files`, `grep_text`
   - read: `read_file`, `read_lines`, `head_file`, `cat_file`
   - write: `write_file`, `append_file`, `replace_in_file`
   - list: `list_dir`, `tree`, `ls_recursive`
   - git: `git_status`, `git_diff`, `git_log`
   - network: `http_get`, `fetch_url`, `download_file`
   - execute: `run_command`, `run_tests`, `run_pytest`

2. **Prompt Set** (`set-v1.jsonl`): 150 prompts hand-written across five categories:
   - **paraphrase** (30): register shift (terse, chatty, non-native, uppercase), verbatim prompts rephrased
   - **multi_intent** (27): two requests; gold is the first actionable one
   - **underspecified** (22): missing required arguments or ambiguous intent; gold is `none` or incomplete
   - **near_miss** (25): superficially resemble one tool's template but need a sibling (e.g. "display first few lines" -> `read_file`, not `head_file`)
   - **none** (33): requests that have no matching tool

3. **Frozen Labels** (`LABELS.sha256`): the sha256 of canonical labels, computed before any model run. Scorer refuses to run if labels have been edited after freezing (prevents label leakage).

## Scorer

`scripts/eval-tooluse-shifted.py` runs both legs:

### Raw Leg (/v1/completions, greedy)
- Prompt template: expanded system prompt with full 20+ tool catalog
- Scores: JSON parses, right tool, schema-valid args, exact match
- Breakdown: counts before rates, per category and overall

### Native Leg (/v1/chat/completions with tools)
- Uses `/v1/chat/completions` with `tools` parameter and `tool_choice: "auto"`
- Extracts tool calls from the response
- Scores: right tool, exact args match
- Records `choice_logprobs` at every constrained decision point (tool name, argument names, argument values)
- Breakdown: per category and overall

## Usage

### Freeze Labels (First Time Only)

```bash
./scripts/eval-tooluse-shifted.py --freeze
```

Produces `LABELS.sha256`. This must be done before running any evaluation.

### Run Evaluation

```bash
./scripts/eval-tooluse-shifted.py --runner ./runner --model path/to/model.gguf \
    [--lora path/to/adapter.gguf [--lora-scale 1.0]]
```

Options:
- `--runner` path to runner binary (default: `./runner`)
- `--model` required, path to base GGUF
- `--lora` optional, path to LoRA adapter
- `--lora-scale` optional, scale for LoRA (default: 0.0, which uses the adapter's trained alpha)
- `--threads` optional, number of threads (default: auto)

Output: JSON record written to `evals/tooluse-shifted/results/<model>-<adapter>-<host>.json`

## Results

Results contain:

- `schema_version`: "xyntetik.runner.tooluse-shifted.v1"
- `model_file`, `model_sha256`: base model provenance
- `adapter_file`, `adapter_sha256`, `adapter_scale`: adapter provenance
- `runner_version`, `host`: execution environment
- `labels_sha256`: frozen label hash (for reproducibility)
- `raw_leg`: counts and rates for greedy /v1/completions leg
  - `json_parses`, `right_tool`, `schema_valid`, `exact_match`
  - Per-category breakdown in `rows`
- `native_leg`: counts and rates for /v1/chat/completions leg
  - `tool_ok`, `args_ok`, `exact_match`
  - `choice_logprobs` records for each decision point
  - Per-category breakdown in `rows`

## What This Eval Does NOT Claim

1. **Rates are properties of this set only.** The 150 hand-written prompts are a fixed benchmark, not a representative sample of any traffic. Results do not transfer to other sets without re-measurement.

2. **One host, one adapter.** This run used one machine and one adapter instance. Other adapters trained the same way and the same adapter on other machines are unmeasured.

3. **Greedy and constraint sampling are different instruments.** Raw leg (greedy decoding) and native leg (constrained tool_choice) measure different model behaviors. Both should be reported; neither alone is sufficient.

## Base Model vs. Adapter

The evaluation distinguishes adapters by comparing:
- Base model on the same prompt set
- Base+adapter on the same prompt set

On a set where the base lands clearly below the adapter on both legs, the instrument is working. On a set where base and adapter are indistinguishable, the set needs redesign (tighter near-miss shapes, more underspecified cases, or more challenging category distribution).

## Test Coverage

`tests/test_tooluse_shifted.py` verifies:
- Catalog has 20+ tools with valid schemas
- Set has 150+ prompts with valid gold labels
- All categories have 20+ items
- Labels hash matches frozen hash
- Prompt IDs are unique
- Gold args match defined schemas
- JSON parsing and schema validation logic
- Label modification detection

Run: `./.venv/bin/python -m pytest -q tests/test_tooluse_shifted.py`
