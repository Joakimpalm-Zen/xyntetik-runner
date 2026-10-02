# Two files, two engines: the recovered Qwen 3.8 27B IQ3_S baseline

Measured 2026-10-02 on the lab's Linux box. The question is not whether the
recovered file is better; it is what each file does, in Runner and in
llama.cpp, kept as four separate outcomes, so that a later report about
either file can be put next to a matched record.

| | |
|---|---|
| Original | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`, sha256 `64b53b64...6d810` |
| Derived | `Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF`, sha256 `873df48d...8458f` |
| Fidelity reference | `Qwen3.8-27B-Q8_0.gguf` served by Runner on the CPU. Not the publisher's bf16 weights, which do not fit beside the other servers; the KL numbers are distance from Q8_0, not from the reference model. |
| Runner | branch `backlog-batch-3` at `06d7d07` (reports 0.5.7), CUDA on an RTX PRO 6000 Blackwell MIG 1g.24gb slice, 4 CPU threads, context 8192 |
| llama.cpp | `73a43d1`, CPU build, 8 threads, `--jinja`, context 8192 |
| Driver | `scripts/artifact-baseline.sh` with the records in this directory |

## Results

| outcome | original | derived |
|---|---|---|
| Loads (`--fit`, `--tool-info`) | yes, `qwen38` template, `qwen3_xml` tools | yes, the same |
| Tool protocol in Runner, 3 cases x buffered and streamed | 6 of 6 | 6 of 6 |
| Tool protocol in llama.cpp, the same cases | 6 of 6 | 6 of 6 |
| Task set, native leg (`/v1/chat/completions` with tools), one row per category | 5 of 6 exact | 5 of 6 exact |
| Fidelity against Q8_0, 300 positions of `tests/fixtures/mixed-corpus.txt` | mean KL 0.312, top-1 agreement 84.3%, 93.0% outside the 0.5-nat tie band | mean KL 0.340, top-1 82.3%, 91.0% |
| Greedy against llama.cpp, 16 prompts x 64 tokens | 12 identical, 3 part at a near-tie, 1 length difference, 1 real divergence | 10 identical, 5 at a near-tie, 1 real divergence |

Read with their sizes in mind:

- **Fidelity.** On this corpus at 300 positions the recovered file is not
  closer to Q8_0 than the original; it is 0.028 nats further. The project's
  own golden pass found that at 100 positions most such gaps were noise,
  and 300 positions does not settle a gap this small either way. Nothing
  here supports a claim that recovery improved the file, and nothing shows
  it made the file worse.
- **Tasks.** Six rows, one per category of the v2 set, are a smoke test,
  not a score: both files miss the same row (`underspecified_01`) and pass
  the same five. The full set did not fit the window: the 27B decodes at
  3.8 tok/s on this slice, and the set ran past its 40-minute limit. The
  raw leg (`*.task-raw-limit1.json`, 2 of 6 for both) feeds the model the
  training template of the project's adapter work, which this model was
  never trained on; it is recorded and not read as a result.
- **Protocol.** All twelve requests in each engine produced well-formed
  calls on both files. The wall times are in the records; they are not a
  speed comparison (llama.cpp ran on the CPU, Runner on a GPU slice).

## What the run fixed on the way

Three defects in the harnesses, each found by this run and fixed in the
same pull request:

- `artifact-baseline.sh` started a check as soon as llama-server opened its
  port, which it does before the model is loaded; the check failed on its
  first request (HTTP 503). It now waits for `/health`.
- `tool-protocol-check.py` read Runner's own `/v1/capabilities` first and
  died on llama.cpp's 404. It now records the capabilities as not known.
- `eval-tooluse-shifted.py` asked for `choice_logprobs` on every native
  request. Qwen 3.8's auto turn is parsed rather than constrained, so the
  server has no decision points to report and refused all six; the leg now
  retries once without the field and records that the probabilities are
  absent.

## Adding a reported case

A prompt someone reports goes into the same records: as a tool case in
`scripts/tool-protocol-check.py` (`CASES`), as a greedy prompt for
`scripts/token_divergence.py`, or as a row of the task set, and the driver
is run again on both files. The "noisier reasoning" report that motivated
this baseline has not come with a reproducible prompt yet; until it does,
it stays an observation.

## Files

`*.sha256`, `*.fit.log`, `*.tool-info.log`: identity and load.
`*.tool-protocol.json`, `*.tool-protocol-llama-rerun.json`: the protocol
gate in each engine. `*.fidelity.json`: the KL rows. `*.divergence.json`:
greedy against llama.cpp. `*.task-native-limit1.json`,
`*.task-raw-limit1.json`: the task rows. `caps.json`: the build's
capabilities. Paths are relative to the lab's workspace; the host name is
replaced.
