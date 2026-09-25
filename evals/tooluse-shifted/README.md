# Tool-use Evaluation with Shifted Distribution (R8.4)

This evaluation measures whether a trained tool-use adapter can be distinguished from the base model on a hand-written prompt set with distribution shift from the original training templates.

Two sets exist. `set-v1.jsonl` (150 prompts, five categories) is the first
measurement and stays frozen for its records. `set-v2.jsonl` (274 prompts,
six categories, the default since 2026-09-24) is the redesign the v1
verdict asked for; its labels, rules and results are in "Set v2" below.

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

### Decide Leg (POST /v1/decide, continuation-v1)
- The training template rendered up to the opening quote of the tool name is the state; every catalog name (none included) is an option
- The endpoint returns an order-invariant distribution over the options with no sampling; the argmax equals the raw leg's greedy tool wherever the raw leg named a catalog tool
- Scores: top-1 against the acceptable set, log loss and Brier of the acceptable mass, ECE over top-1 confidence, and a confidence certificate (the lowest confidence above which the decisions are right at least 90% of the time, with its coverage)

### Next-Call Leg (set v2, the 40 multi-intent rows)
- Teacher-forced: the user turn, the GOLD first call as the assistant's tool call, and a canned result for it are replayed through the chat endpoint, and the following call is scored against `gold_next`
- Isolates the second decision from the first; a `none` second call means the model should answer in prose

### Field-level arguments (both generation legs)
- `fields_ok` of `fields_total` counts required arguments whose value matches the label after normalisation (strings stripped, `./x` and `x/` read as `x`, a digit string read as an integer where the label is one); `keys_ok` counts calls whose argument set is exactly the required set; `exact` needs both
- Empty outputs and refused requests are counted separately from wrong answers

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
- `--gpu` runner `--gpu` value (default `off`; the record says which)
- `--set v1|v2` prompt set (default `v2`); `--legs raw,native,decide,next`
- `--decisions-out FILE` also writes the decide leg as `scripts/cl-calibration.py` input
- `--limit N` smoke runs, the first N rows per category (record marked partial)
- `--rescore RECORD...` re-derives every verdict of existing records from the outputs they store, under the current scoring rules and the frozen labels

Output: JSON record written to `evals/tooluse-shifted/results/<model>-<adapter>-<set>-<host>.json`
(v1 records keep the old name without the set). `scripts/tooluse-shifted-compare.py BASE ADAPTER...`
turns records into the paired comparison the verdict rests on.

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

## Set v2 (2026-09-24): the redesign, and can the instrument fail

The v1 verdict listed what to change: more of the categories that move,
fewer that saturate, argument-level scoring, and a larger n. Set v2 is
that, with two rules v1 did not have.

**The set.** 274 hand-written prompts: paraphrase 50, multi_intent 40,
underspecified 50, near_miss 70, none 34, and a new `arg_shift` 30
(the tool is obvious, the argument values arrive in unusual forms: numbers
as words, paths with spaces, a tag name, an empty argument). Labels were
frozen in `LABELS-v2.sha256` before any model ran; the frozen row also
carries `also_ok` and `gold_next`.

**Labelling rules.**
- `gold_args` is exactly the tool's required fields. A directory-scoped tool
  takes `.` when the prompt names no place; a file path never defaults, so
  a request with no file named is `underspecified` and its gold is `none`.
- Synonym siblings the catalog cannot separate (`find_files`/`search_files`,
  `read_file`/`cat_file`, `http_get`/`fetch_url`) are declared in
  `also_ok`: the sibling counts as the right tool, and as exact with the
  same arguments. v1 had labelled these arbitrarily and the scorer punished
  the other spelling.
- `near_miss` prompts are built only on sibling pairs the catalog does
  separate (content versus filename search, a line range versus the head
  versus the whole file, create versus append versus replace versus
  truncate, one level versus a tree versus a flat recursive listing, a
  filter versus a file versus a shell command, status versus diff versus
  log, fetch versus save to disk).
- Every `multi_intent` row carries the second call, or `none` when the
  second intent is prose.

**Two scorer defects found by the first v2 records and fixed before the
verdict.** A gold path spelled `./logs/app.log` was canonicalised on one
side only, so the model's identical spelling scored as a miss; and the
second call did not take the synonym rule, so `cat_file` for a gold
`read_file` scored as the wrong tool. Both are covered by tests, and the
four records below were re-derived from their stored outputs with
`--rescore` (each changed on exactly the two rows the defects touched;
`rescored_with` says so in the record).

### Run (2026-09-24, ZEN-GAMING, CUDA path)

Host: Windows 11, RTX 3070, runner 0.5.6 built from the branch, `--gpu
auto` (the 4B fits on the card; the adapter served on the device, R8.7.2),
8 threads, greedy. The same base and adapters as v1, sha256 verified on
the box: base fbe1d5ed..., published adapter ea38f80c..., through-Q8_0
0c85d630..., through-BF16 1cafe339.... One arm took 13 to 16 minutes for
all four legs. Records: `results/*-v2-gpu-windows.json`. The CPU-path pass
is the section below.

### Second path: the same box on the CPU

The four arms were rerun on the same host with `--gpu off`, the path every v1
record used, 2 h 45 m to 2 h 55 m per arm against 13 to 16 minutes on the
card. Records: `results/*-v2-windows.json` (the CUDA ones carry `-gpu`).

Agreement is per row, not per total. Verdicts identical on every row of every
leg, all four arms:

| arm | raw exact | native exact | second call | decide argmax | raw output bytes |
|---|---|---|---|---|---|
| base | 274 / 274 | 274 / 274 | 40 / 40 | 274 / 274 | 272 / 274 |
| published (Q4_K_M) | 274 / 274 | 274 / 274 | 40 / 40 | 274 / 274 | 271 / 274 |
| through Q8_0 | 274 / 274 | 274 / 274 | 40 / 40 | 274 / 274 | 271 / 274 |
| through BF16 | 274 / 274 | 274 / 274 | 40 / 40 | 274 / 274 | 272 / 274 |

The ten raw rows whose BYTES differ all diverge late in a long answer (the
first differing byte is 65 to 215 characters in) and every one of them parses
to the same call and the same verdict. On the decide leg the largest
difference in any option's probability, over all 1,096 scored rows, is 0.0126.

This is the arithmetic-level agreement the v1 pass could not show. There the
one arm that differed across hosts was the adapter trained through Q8_0, 149
of 150 rows between ZEN and the lab box; here the same adapter agrees on all
274 rows between the two compute paths on ONE host. The v1 difference is
therefore a cross-HOST property, not a property of the adapter serving path,
which is what a single host cannot separate.

### Counts Before Rates (Raw Leg: the training template with the full catalog)

Right tool / exact of n, per arm.

| Category | n | base | published (Q4_K_M) | through Q8_0 | through BF16 |
|---|---:|---:|---:|---:|---:|
| paraphrase | 50 | 43 / 35 | 46 / 36 | 46 / 35 | 46 / 35 |
| multi_intent | 40 | 37 / 29 | 35 / 28 | 36 / 28 | 36 / 28 |
| underspecified | 50 | 5 / 5 | 9 / 9 | 6 / 6 | 6 / 6 |
| near_miss | 70 | 55 / 51 | 56 / 51 | 54 / 47 | 54 / 47 |
| none | 34 | 25 / 25 | 30 / 30 | 29 / 29 | 30 / 30 |
| arg_shift | 30 | 29 / 25 | 30 / 22 | 29 / 20 | 29 / 20 |
| **Overall** | **274** | **194 / 170** | **206 / 176** | **200 / 165** | **201 / 166** |

JSON parses 274 on every arm; argument fields right 262, 261, 257, 257 of
324; empty outputs 0.

### Counts Before Rates (Native Leg: /v1/chat/completions with tools, thinking off, 256-token budget)

| Category | n | base | published (Q4_K_M) | through Q8_0 | through BF16 |
|---|---:|---:|---:|---:|---:|
| paraphrase | 50 | 44 / 41 | 47 / 36 | 46 / 35 | 46 / 35 |
| multi_intent | 40 | 36 / 32 | 35 / 29 | 35 / 29 | 35 / 29 |
| underspecified | 50 | 33 / 33 | 31 / 31 | 30 / 30 | 30 / 30 |
| near_miss | 70 | 55 / 52 | 55 / 49 | 57 / 49 | 58 / 50 |
| none | 34 | 29 / 29 | 31 / 31 | 30 / 30 | 30 / 30 |
| arg_shift | 30 | 27 / 24 | 28 / 24 | 29 / 23 | 29 / 23 |
| **Overall** | **274** | **224 / 211** | **227 / 200** | **227 / 196** | **228 / 197** |

Argument fields right 271, 267, 267, 269 of 324; refusals and empty
outputs 0.

### Decide Leg (/v1/decide over the 24 catalog names, the training template)

Top-1 in the acceptable set / mean log loss of the acceptable mass.

| Category | n | base | published (Q4_K_M) | through Q8_0 | through BF16 |
|---|---:|---:|---:|---:|---:|
| paraphrase | 50 | 44 / 0.37 | 46 / 0.51 | 46 / 0.44 | 46 / 0.46 |
| multi_intent | 40 | 37 / 0.34 | 35 / 0.29 | 36 / 0.21 | 36 / 0.22 |
| underspecified | 50 | 5 / 7.64 | 9 / 9.82 | 6 / 8.39 | 6 / 8.30 |
| near_miss | 70 | 55 / 1.86 | 56 / 2.15 | 54 / 2.12 | 54 / 2.12 |
| none | 34 | 26 / 1.31 | 30 / 0.64 | 30 / 0.62 | 30 / 0.60 |
| arg_shift | 30 | 30 / 0.03 | 30 / 0.00 | 29 / 0.04 | 29 / 0.06 |
| **Overall** | **274** | **197 / 2.15** | **206 / 2.56** | **201 / 2.27** | **201 / 2.25** |
| Brier | | 0.508 | 0.459 | 0.496 | 0.495 |
| ECE (10 bins) | | 0.221 | 0.215 | 0.234 | 0.234 |
| certificate at 90%: confidence above | | 0.999974 | 1.000000 | 0.999992 | 0.999993 |
| rows it covers, accuracy there | | 76, 0.908 | 55, 0.927 | 40, 0.900 | 41, 0.902 |

The decide argmax equals the raw leg's greedy tool on every row where the
raw output named a catalog tool (274 of 274 on the published arm, 269 of
274 on base: the five are invented names such as `move_file` and `open`,
which the decide leg cannot emit). The two legs are the same decision read
two ways; the decide leg adds the probability.

### Next-Call Leg (the 40 multi-intent rows, teacher-forced)

Right tool / exact: base 36 / 34, published 34 / 32, through Q8_0 33 / 29,
through BF16 34 / 30. Argument fields 34, 30, 28, 29 of 44. The failures
are the second intent misread (`write_file` for a `download_file`, `sh`
as the command for a script named as the command) or answered in prose.

### Paired comparison (`scripts/tooluse-shifted-compare.py`, the verdict rests on this)

Base (A) against each adapter (B). Generation legs: discordant pairs and
z = (A only - B only) / sqrt(discordant). Decide leg: mean per-row
difference in log loss, bootstrap 95% CI over rows (4,000 resamples, seed
20260924); negative favours the base.

| adapter | raw A only / B only, z | native A only / B only, z | next A only / B only, z | decide mean(A - B), 95% CI | rows A closer / B closer |
|---|---|---|---|---|---|
| published (Q4_K_M) | 17 / 23, -0.95 | 18 / 7, +2.20 | 3 / 1, +1.00 | -0.405 [-0.652, -0.166] | 91 / 167 |
| through Q8_0 | 22 / 17, +0.80 | 23 / 8, +2.69 | 5 / 0, +2.24 | -0.114 [-0.310, +0.086] | 164 / 99 |
| through BF16 | 22 / 18, +0.63 | 23 / 9, +2.47 | 4 / 0, +2.00 | -0.099 [-0.298, +0.105] | 156 / 107 |

Adapter pairs:

| A | B | raw A only / B only, z | decide mean(A - B), 95% CI | rows A closer / B closer |
|---|---|---|---|---|
| published | through Q8_0 | 12 / 1, +3.05 | +0.291 [+0.108, +0.491] | 194 / 70 |
| published | through BF16 | 11 / 1, +2.89 | +0.306 [+0.128, +0.501] | 188 / 73 |
| through Q8_0 | through BF16 | 0 / 1, -1.00 | +0.015 [+0.001, +0.029] | 70 / 184 |

Decide leg by category, mean(base - adapter) log loss:

| category | published | through Q8_0 | through BF16 |
|---|---:|---:|---:|
| arg_shift | +0.029 | -0.012 | -0.033 |
| multi_intent | +0.044 | +0.127 | +0.123 |
| near_miss | -0.290 | -0.258 | -0.254 |
| none | +0.670 | +0.687 | +0.708 |
| paraphrase | -0.136 | -0.070 | -0.081 |
| underspecified | -2.183 | -0.753 | -0.663 |

### Verdict

**The instrument can fail. It tells every adapter apart from the base and
it tells the three adapters apart from each other, including the two
retrained through Q8_0 and BF16, which no count separates. What it shows
is not what R8.4.3 assumed: the adapters are not clearly above the base
on this set, and on the chat protocol and the decide leg the evidence
favours the base.**

- Adapters apart from each other: the published adapter against either
  retrained one, 12 or 11 raw exact verdicts to 1 (z about 3), and a decide
  CI well clear of zero. Through-Q8_0 against through-BF16: one raw verdict
  apart, but the BF16-trained adapter is closer to the truth on 184 of 274
  rows against 70, mean 0.015 nat, CI [0.001, 0.029]. That is the three-
  precision question the study asked and a count could never answer; the
  effect is small and its sign is now measurable.
- Base apart from the adapters: on the native leg every adapter loses more
  discordant rows than it wins (z 2.2 to 2.7, base favoured); on the second
  call the retrained adapters lose 5 and 4 rows and win none; on the decide
  leg the published adapter is farther from the truth on average (CI below
  zero), the retrained ones are not separated by the CI (they win more rows
  than they lose, 164 and 156 against 99 and 107, but by less per row).
- Where the adapters help: `none` (base 25, adapters 29 to 30 of 34, and
  0.6 to 0.7 nat closer on the decide leg) and the training-template raw
  exact total for the published adapter (+6, inside noise). Where they
  hurt: `underspecified` on the decide leg (the adapters are more certain
  of the wrong tool where the label is `none`: 9.8 against 7.6 nat), the
  chat protocol they were not trained on (exact 196 to 200 against 211),
  and `arg_shift` arguments on the raw template (22, 20, 20 exact against
  25: the same tool, the training shape of the arguments instead of the
  prompt's).
- `underspecified` is the category that fails every arm on the raw
  template (5 to 9 of 50) and passes on the chat protocol (30 to 33 of 50):
  under the raw template the model fills a placeholder path and calls;
  under the chat protocol it asks. That is a property of the template, not
  of the adapters, and the reason the raw leg alone cannot be the gate.
- Calibration: none of the four arms is calibrated on this decision. ECE
  0.21 to 0.23; the 0.9 to 1.0 confidence bin holds 218 to 241 of the 274
  rows at accuracy 0.76 to 0.78. The 90% certificate covers only 76 rows
  (base) down to 40 (through Q8_0), all at a confidence within 1e-4 of
  1.0. The published adapter is the sharpest (Brier 0.459, the best) and
  the most wrong when wrong (log loss 2.56, the worst); the two metrics
  disagree by design, and both are reported.

R8.4.3 as written (base clearly below every adapter, adapters differ):
NOT PASSED, and the reason is the adapters, not the set. R8.4.3 as the
instrument question (base and adapters told apart, adapters told apart):
PASS. Rates are properties of this set; nothing here is a claim about
traffic, and the second-call leg is teacher-forced (the first call is
always the gold), so it measures the second decision alone.

## Test Coverage

`tests/test_tooluse_shifted.py` verifies (45 tests pass):
- Catalog has 20+ tools with valid schemas
- Set has 150+ prompts with valid gold labels
- All categories have 20+ items
- Labels hash matches frozen hash
- Prompt IDs are unique
- Gold args match defined schemas
- JSON parsing and schema validation logic
- Label modification detection
- Server command builder: adapter scale and GPU flag correctly passed to runner
- Set v2 rules: synonyms share a signature, second calls valid, labels of both sets frozen
- Field scoring normalisation, the decide and next legs against mocked endpoints, rescoring from stored outputs
- The paired comparison's direction and verdicts, the calibration certificate

Run: `./.venv/bin/python -m pytest -q tests/test_tooluse_shifted.py`
