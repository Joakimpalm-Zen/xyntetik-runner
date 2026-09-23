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
- `--lora-scale` optional, scale for LoRA (default: 1.0)
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

2. **Two x86 hosts, no Apple host.** ZEN-GAMING (Windows) and the lab box (Linux); the M1 run paged and was abandoned. The two retrained adapters are the published recipe's adapters, not byte copies of the lost originals.

3. **Greedy and constraint sampling are different instruments.** Raw leg (greedy decoding) and native leg (constrained tool_choice) measure different model behaviors. Both should be reported; neither alone is sufficient.

## Base Model vs. Adapter

The evaluation distinguishes adapters by comparing base and base+adapter on
the same frozen set, both legs. On a set where the base lands clearly below
the adapters and the adapters differ somewhere, the instrument works. Where
they are indistinguishable, the set needs redesign; a gate that cannot fail
is not a gate.

## Real Run Results (2026-09-23, ZEN-GAMING, CPU)

Host: Windows 11, 8 threads, runner 0.5.6, `--gpu off`, greedy. Base:
bartowski/Qwen_Qwen3-4B-Q4_K_M (sha256 fbe1d5ed...), the study base. Four
arms: base; the published adapter (Qwen3-4B-ToolUse-LoRA, trained through
Q4_K_M, sha256 ea38f80c...); and the two study adapters retrained from the
published recipe because the originals were lost with the box that trained
them (rank 8, alpha 16, lr 1e-4, 120 steps, ctx 128, runner 0.5.6; seed 1
where the study used seed 0, which the current sampler refuses as an RNG
fixed point; through Q8_0: sha256 0c85d630..., first-step loss 0.6911, last
0.0001; through BF16: sha256 1cafe339..., 0.7022 to 0.0001; train records
under `adapters/`). Scale 1.0 on every adapter. The raw leg reproduced
exactly across three runs of the same arm (greedy scoring is deterministic
here); the first two runs' native legs were scorer defects, not
measurements, and are archived off the record.

### Counts Before Rates (Raw Leg: /v1/completions, the training template with the full catalog)

Right tool / exact match of n, per arm.

| Category | n | base | published (Q4_K_M) | through Q8_0 | through BF16 |
|---|---|---|---|---|---|
| paraphrase | 42 | 29 / 11 | 29 / 14 | 31 / 14 | 31 / 14 |
| multi_intent | 27 | 17 / 3 | 18 / 4 | 18 / 3 | 19 / 4 |
| underspecified | 22 | 6 / 3 | 9 / 6 | 7 / 4 | 7 / 4 |
| near_miss | 26 | 13 / 2 | 14 / 0 | 13 / 0 | 14 / 0 |
| none | 33 | 27 / 27 | 33 / 33 | 30 / 30 | 31 / 31 |
| **Overall** | **150** | **92 / 46** | **103 / 57** | **99 / 51** | **102 / 53** |

JSON parses 149, 150, 150, 150; schema-valid arguments 92, 100, 98, 101.

### Counts Before Rates (Native Leg: /v1/chat/completions with tools, thinking off, 256-token budget)

Tool match / exact match of n, per arm. Refusals 0 on every arm; rows that
emitted a call 79, 81, 84, 83; finish "length" 3, 2, 2, 2.

| Category | n | base | published (Q4_K_M) | through Q8_0 | through BF16 |
|---|---|---|---|---|---|
| paraphrase | 42 | 31 / 15 | 29 / 14 | 29 / 15 | 29 / 15 |
| multi_intent | 27 | 13 / 7 | 14 / 4 | 14 / 4 | 13 / 4 |
| underspecified | 22 | 17 / 15 | 18 / 16 | 17 / 15 | 17 / 15 |
| near_miss | 26 | 9 / 1 | 10 / 0 | 9 / 0 | 9 / 0 |
| none | 33 | 33 / 33 | 33 / 33 | 33 / 33 | 33 / 33 |
| **Overall** | **150** | **103 / 71** | **104 / 67** | **102 / 67** | **101 / 67** |

### Verdict

**The instrument separates base from adapter only weakly, and it cannot tell
the three adapters apart. It does not yet pass R8.4.3.**

- Raw leg, the template the adapters were trained on: the adapters gain
  7 to 11 right-tool and 5 to 11 exact matches over base out of 150. The
  binomial standard deviation of a count at these rates is about 6, so the
  base-versus-adapter gap is one to two standard deviations: present, not
  clear.
- The three adapters disagree with each other on 2 to 6 of 150 exact
  verdicts. That is inside noise; nothing here ranks Q4_K_M, Q8_0 and BF16
  training.
- Native leg: the adapters do not help on the chat protocol at all (tool
  match 101 to 104 against base 103; exact 67 against base 71). They were
  trained on the raw template, and the constrained chat surface reaches
  more exact calls than the raw template on every arm (67 to 71 against
  46 to 57), so the native leg measures the base's protocol, not the
  adapter.
- Where the set does discriminate: `none` (base 27 of 33, every adapter 30
  to 33) and `underspecified` (base 3 exact, published 6) on the raw leg;
  `near_miss` runs the other way, every adapter drops to 0 exact where base
  keeps 2, which is a real adapter failure mode (the sibling tool is named,
  the arguments follow the training shape instead of the prompt).

What to redesign before R8.5 leans on this set: more `near_miss` and
`underspecified` items, since those are the only categories that move;
retire or shrink `none`, which saturates for any adapter; score the
argument-level decisions the native leg records (`choice_records`) rather
than whole-call exact match; and raise n so a 10-count gap is more than two
standard deviations. Rates are properties of this set, never of traffic.

### Second host: the lab box (x86, 128 cores, CPU), 2026-09-23

Same set, same scorer, same base file and adapters (sha256 verified),
runner main at 9b825fc, `--gpu off`, 16 threads (the Q8_0 arm also at 8).
The box was running a training job beside these arms (load 12 to 23); one
arm lost its server under that load and was rerun. Right tool / exact of
150 on the raw leg, tool / exact on the native leg:

| arm | raw, ZEN (8 threads) | raw, lab (16 threads) | native, ZEN | native, lab |
|---|---|---|---|---|
| base | 92 / 46 | 92 / 46 | 103 / 71 | 103 / 71 |
| published (Q4_K_M) | 103 / 57 | 103 / 57 | 104 / 67 | 104 / 67 |
| through Q8_0 | 99 / 51 | 99 / 51 (8 threads); 97 / 50 (16 threads) | 102 / 67 | 101 / 67 |
| through BF16 | 102 / 53 | 102 / 53 | 101 / 67 | 101 / 67 |

Row-level agreement between the hosts, not just the totals: base,
published and BF16 arms are identical on every one of the 150 rows on
both legs, outputs and verdicts, across a different OS, compiler and
thread count. The Q8_0 arm agrees on 149 of 150 raw outputs and 149 of
150 native verdicts between ZEN and the lab at 8 threads: one raw row
(`none_case_32`) diverges 134 characters into a long answer with the same
verdict, and one native row (`multi_intent_27`) resolves to a different
call (`ls_recursive` on ZEN, no call on the lab) with top-2 margins of 0.08
and 0.13, not a tie. The adapter path is the only one that differs at all
between the two x86 hosts; the base forward is identical on all 150 rows.
The lab's 16-thread Q8_0 record differs from its 8-thread one on three
`underspecified` rows whose output is EMPTY, requests that failed while
the box was at load 23, not a numeric difference; that record is kept
under `-linux-t16` for the account and is not the row above.

Both records per arm are in `results/` (`-windows` and `-linux`). The
cross-host result does not change the verdict: the set separates base
from adapter weakly on both hosts and cannot rank the adapters on either.

## Test Coverage

`tests/test_tooluse_shifted.py` verifies (25 tests pass):
- Catalog has 20+ tools with valid schemas
- Set has 150+ prompts with valid gold labels
- All categories have 20+ items
- Labels hash matches frozen hash
- Prompt IDs are unique
- Gold args match defined schemas
- JSON parsing and schema validation logic
- Label modification detection
- Server command builder: adapter scale correctly passed to runner

Run: `./.venv/bin/python -m pytest -q tests/test_tooluse_shifted.py`
