# Tool-choice decision-boundary lane (unlabeled)

`scripts/tool-choice-boundary.py` runs a bank of deliberately ambiguous
tool-choice prompts under several serving conditions and records, for each
prompt and condition, the one grammar decision at which the tool name is
chosen. Because the tool name is a JSON-schema enum, the runner's
`choice_logprobs` exposes that decision natively: its legal alternatives are
exactly the first pieces of the candidate tool names, with a raw logprob for
each and the probed coverage mass. No hand-built probe, no scoring of tool
names as text (a scorer of that kind was invalidated once by BPE merging
punctuation across the apparent string boundary; the grammar record cannot
fail that way).

The lane answers one question: **at which prompts do the conditions choose
differently, and how tight were the margins?** It answers nothing about
which choice was right.

## The bank, and its credit

The default bank is the **Xyntetik Runner Tool-Choice Boundary Prompt Bank
v1.0**, originally contributed by John6666, 2026, dual-licensed MIT OR
Apache-2.0; Apache-2.0 is selected for its inclusion here. It is carried
byte-identical to the published file
(`scripts/tool-choice-boundary-bank.jsonl`, sha256
`c7a30994...6d084`, pinned by `tests/test_tool_choice_boundary.py` against
the value in the bank author's own manifest). 36 prompts in five families:

| family | prompts | shape |
|---|---|---|
| `list_vs_search` | 10 | browse a directory, or search it by pattern |
| `read_vs_search` | 8 | open a named file, or find it first |
| `read_vs_write` | 6 | inspect a file, or change it |
| `config_browse` | 6 | browse a config directory for a file type |
| `tool_vs_none` | 6 | a request that may or may not fit any tool |

Family names describe how the cases were constructed. They do not declare a
correct action, and the bank carries no gold labels, deliberately: labeling
would turn a decision-sensitivity probe into a quality benchmark and quietly
change the question. The prompt contract sent to the model is the published
precision study's own (`scripts/make-tooluse-data.py`, verbatim, pinned by
test), so the lane's decisions are that study's decisions.

## What the lane is not

Three quantities stay separate, and the wording travels with every result:

| quantity | what it can mean |
|---|---|
| a split found among screened cases | an existence proof |
| disagreements across all prompts of a bank | a property of THAT bank (it was constructed, so it estimates nothing about ordinary traffic) |
| prevalence in real tool-use traffic | needs a target population and unbiased sampling; this lane does not measure it |

Two claims the lane's own data refute, and which must not be made from it:
"lower training precision degrades tool calling" (the direction is
non-monotonic in bit width; see below), and any count read as a rate.

The records **cannot feed `scripts/cl-calibration.py`**. That tool needs a
`correct_id` per decision; these records have none, and `summarize` emits
no field that could be mistaken for one (a test forbids it). A labeled
calibration set, if ever wanted, is a separate curated dataset whose labels
are frozen before any condition's outcome is seen.

## Running it

One fresh serve per condition (`--gpu off` by default: one backend, so runs
compare across hosts), the raw completion endpoint, `temperature 0`, an enum
schema over the five tools, `choice_logprobs` on:

```
scripts/tool-choice-boundary.py run --runner ./runner --model base-Q4_K_M.gguf \
    --condition base \
    --condition bf16=adapter-trained-through-bf16.gguf \
    --condition q8=adapter-trained-through-Q8_0.gguf \
    --condition q4=adapter-trained-through-Q4_K_M.gguf \
    --out records.jsonl --serve-log serve.log --hash-artifacts
scripts/tool-choice-boundary.py summarize records.jsonl          # markdown
scripts/tool-choice-boundary.py summarize records.jsonl --json
```

`--url` probes one already-running server instead (one `--condition`).
`--bank FILE` substitutes any JSONL of `{"id","family","prompt"}` rows;
`--template FILE` substitutes the prompt contract (one `%s`). Both are
recorded by sha256 in every record so runs never mix silently.
`--hash-artifacts` records the model's and adapters' sha256 as well, which
is what makes a run reproducible against published artifacts.

Progress goes to stderr; `--out` is appended one record per completed
probe, so an interrupted run keeps what finished.

## Record schema (`tool_choice_boundary_record_v1`)

One JSON object per line, one line per (prompt, condition):

| field | meaning |
|---|---|
| `record` | `tool_choice_boundary_record_v1` |
| `id`, `family`, `prompt` | the bank row |
| `condition`, `model`, `adapter` | condition name; file basenames (`adapter` null for a base condition) |
| `model_sha256`, `adapter_sha256` | with `--hash-artifacts`, else null |
| `runner_version`, `gpu`, `host` | the launched binary's `--version`, its `--gpu` mode, the platform string (null when `--url` was used) |
| `bank_sha256`, `template_sha256`, `tools` | what was asked, exactly |
| `timestamp` | UTC |
| `full_text`, `full_call` | the deterministic completion and its parsed `{tool, args}` (null if not one object) |
| `n_decisions`, `all_decisions` | every `choice_logprobs` record of the call, as the server emitted them |
| `decision` | the tool decision, or null if no probed legal alternative was a tool piece |
| `seconds` | wall time of the probe |

`decision` holds the server's `index`, `n_legal`, `coverage` and
`alternatives` for that point, plus the tool-level reading: `by_tool` (best
raw logprob per tool among the legal pieces), `top1`, `top1_piece`, `top2`
(the best alternative leading to a DIFFERENT tool, never a shorter piece of
the same name; `no` and `non` are the `none` branch), and `margin_nat`
(`top1` minus `top2`, raw logprobs). Coverage below about 0.99 means the
probe width missed legal mass; raise `--probe` (up to 64).

## Reading a summary

`summarize` reports, per prompt and condition, the top branch, runner-up,
margin, coverage and emitted call; then the prompts where the ADAPTER
conditions disagree on the top branch (and whether their full calls
disagree too), the prompts where the adaptation moved the branch relative
to the base, the margin distribution per condition, and which adapter was
tightest per prompt. A disagreement list is the result. A count is a
property of the bank that was run.

## Results on the published precision-study artifacts (2026-09-01)

Runner 0.4.5, CPU path, against the study's exact published objects
(bartowski Qwen3-4B Q4_K_M base, the three adapters trained through BF16,
Q8_0 and Q4_K_M, all sha256-verified before each run):

- **Migration bridge.** The bank author's seven screened cases, rerun
  through this lane: top-1 agreement 28/28 case-conditions with the
  published notebook, the known split (`tell me what README.md says and
  translate it`: BF16 and Q8 `read_file`, Q4 `none`) reproduced through
  full deterministic generation, coverage at least 0.998 everywhere.
  Margins differ from the notebook's hand-built first-token probe (two
  scoring constructions) but land within about half a nat; the invariant
  was behavioral, not numeric.
- **All 36, unscreened.** Every prompt under all four conditions, no
  adapter used to select a subset: **3 of 36 prompts split across the three
  adapters, all three surviving full deterministic generation.** The known
  case, plus two in `config_browse` (`show me the yaml files in config`,
  `list config and find yaml files`) where BF16 and Q8 choose `list_dir`
  and Q4 chooses `search_files` with a real pattern argument. In both new
  cases Q4 was the MOST confident condition (5.9 and 3.9 nat against 1.1/0.6
  and 3.7/3.2), which is why a screen keeping cases with a Q4 margin at or
  under 3 nat could never have found them: a screen that selects on one
  condition's uncertainty is blind to that condition being confident and
  different. On the full bank the Q4-trained adapter carries the widest
  margins of the three (median 5.8 nat against 4.6 and 4.9), wider than both
  others on 22 of 36 prompts, with Q8 the tightest on 7. Non-monotonic in
  bit width, on the whole bank. "More decisive" is not "better": there are
  no labels, and the two new cases are exactly Q4 confidently taking the
  branch the other two decline.
- **Cross-host anchor.** The bridge rows rerun on an x86 host against the
  Apple M1 run: 32/32 same top branch and same call, byte-identical text,
  margins within 0.007 nat. The tightest margin in the study is 0.35 nat,
  so the host is not a variable for it.
- The notebook's 3 nat screen, applied to this lane's native Q4 margins,
  selects exactly the same seven cases; the Q4 discovery scan agrees on the
  top branch for all 36.

Raw records, serve logs and the analysis scripts for these runs are kept in
the project's evidence archive; the summary above is what they support and
no more. Where an expanded bank has the highest information gain, on this
evidence: browse-by-file-type requests of the `config_browse` shape, and
`tool_vs_none`.

## Two axes that must not be conflated

This lane measures the *training* precision path: adapters trained through
different precisions, served on one base. The merge study in
`adaptation-engine.md` measures *merging* an adapter into a quantized base,
which erases it. Both involve 4-bit; they are different mechanisms and
separate claims.
