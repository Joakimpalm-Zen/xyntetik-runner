# Qwen3.8-27B: agent-torture on the qwen38 template and its thinking channel (2026-10-31)

**Unflattering first: the case that tests the thinking channel before a tool call did not run on the native tool
protocol.** All 12 `reasoning_then_tool` cases went through the generic JSON envelope, because their tool's
`hypothesis` string parameter carries a length or pattern constraint that Qwen3-Coder's raw-text function-XML syntax
cannot enforce, and Runner falls back rather than emit a call it cannot hold to the schema (the server logs it once
per request). The thinking channel itself ran: 25 responses carry `reasoning_content`. A reasoning-then-native-call
case needs a tool without such a constraint, which this matrix does not have.

Everything scored passed: 97 of 97, 0 failed, 0 declined.

## The run

`scripts/agent-torture.py --cases 100` against a Runner server it spawned (main at 620fe38a, `runner 1.9.0`), on the
CPU (16 threads, CUDA hidden), temperature 0, the file's own `qwen38` template with its default thinking channel.

- Model: `unsloth/Qwen3.8-27B-GGUF` at 4ca72078, `Qwen3.8-27B-UD-Q4_K_M.gguf`, sha256 `322e194f…` (the roster's Tier 2
  file).
- 100 requests, 3,237 s, peak RSS 16.7 GB.
- For about 18 seconds early in the run, a second job shared the CPUs (a scheduling mistake, stopped by hand). It
  changes no verdict; the elapsed time and the throughput figure in `report.json` include it.

| category | passed | excused | protocol |
|---|---|---|---|
| `nested_arguments` | 13 | 0 | native function-XML |
| `tool_selection` | 13 | 0 | native function-XML |
| `forced_truncation` | 10 | 3 | native function-XML |
| `tool_stream_normalization` | 12 | 0 | native function-XML |
| `large_enum_selection` | 12 | 0 | native function-XML |
| `reasoning_then_tool` | 12 | 0 | **generic envelope** (see above) |
| `stream_normalization` | 13 | 0 | no tools |
| `structured_final` | 12 | 0 | no tools (`response_format`) |

The three excused cases are `forced_truncation` at `max_tokens` 1: the single token went into the reasoning channel,
so no call was ever begun, and the turn's `length` is the truthful answer (decision R4.12.3; `scripts/agent-torture.py`
counts these apart from passed and failed).

The records are `report.json` (per-case verdicts and the totals) and `raw.jsonl` (every request and response).
