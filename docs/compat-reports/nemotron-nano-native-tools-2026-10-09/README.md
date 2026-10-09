# Nemotron Nano native tools: agent-torture, generic envelope against native protocol

2026-10-09, `scripts/agent-torture.py --cases 40` on NVIDIA-Nemotron-Nano-9B-v2
Q8_0 (the same file both arms), CPU only (8 cores of a Zen 5 host, CUDA hidden),
run by the runner session overnight (suite R2.4.4).

| arm | binary | scored | passed | failed | excused |
|---|---|---|---|---|---|
| `generic/` | main 8d096def: tools through the generic JSON envelope | 40 | 39 | 1 | 0 |
| `native/` | branch nemotron-nano-tools: `<AVAILABLE_TOOLS>` declarations, `<TOOLCALL>[...]` calls | 39 | 39 | 0 | 1 |

The generic arm's failure: a forced-truncation case where the model answered in
prose instead of calling. The native arm's excused case: a one-token budget that
ended inside the model's reasoning channel, before any call could begin (the
harness excuses that shape). Every category passed 5 of 5 on both arms
otherwise.

`report.json` and `raw.jsonl` are the harness's own files. The reports' model
and binary paths were rewritten to `~/models/...` and `./runner (...)` before
commit; nothing else was changed.
