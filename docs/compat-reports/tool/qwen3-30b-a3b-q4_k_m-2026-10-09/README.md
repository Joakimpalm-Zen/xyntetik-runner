# Qwen3-30B-A3B Q4_K_M tool check, re-measured 2026-10-09

The compatibility matrix recorded this file's tool check (agent-torture, 8
cases) as 2 of 8 at runner 0.4.10 (2026-09-06,
`../../0.4.10-2026-09-06-blackwell-qwen3-30b-a3b-q4_k_m.json`);
suite item R6.2.3 asked why. Re-measured on runner main 72a55949 (1.2.0 plus
the changes after it) with `scripts/agent-torture.py --cases 40`, the same file
(sha256 0d003f66...), CPU only (8 cores of a Zen 5 host, CUDA hidden):

**39 of 39 scored cases passed, 1 excused** (a one-token budget that ended
inside the reasoning channel before any call could begin, which the harness
excuses). Every category passed 5 of 5.

No single change is named here: the September failures predate the tool-call
fixes since (constrained native turns, the parallel default, the post-think
newline, the repeat-penalty window), and this run measures the current whole.
The report's model and binary paths were rewritten to `~/models/...` and
`./runner (...)` before commit; nothing else was changed.
