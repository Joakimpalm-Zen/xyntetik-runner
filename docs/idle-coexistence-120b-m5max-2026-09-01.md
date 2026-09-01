# Idle coexistence at 120B — M5 Max, 2026-09-01

The README's designed-to-stay-on measurement was made on an 8 GB M1 with a
3.6 GB model, where llama-server's resident set was 47% of the machine. The
question a 128 GB host answers: does the design difference still matter when
RAM is abundant? Measured with `scripts/idle_coexistence.py` (same four-state
lifecycle, 60 s idle windows, engines run sequentially, quiet machine),
gpt-oss-120b-MXFP4 (63.4 GB) at c=4096, stock defaults: Runner 0.4.4 vs
llama-server `010be968` (0.3.0-dev, Metal, self-built).

| while loaded and idle | runner | llama-server |
|---|---|---|
| wired (unevictable) delta vs baseline | **+35 MB** | **+60,815 MB** |
| process RSS | 834 MB | 60,633 MB |
| CPU seconds per idle minute | 0.00 | 0.12 |
| give the memory back | `POST /unload` → RSS 229 MB (HTTP 200) | kill the process |

And the flip side, reported as always:

| latency | runner | llama-server |
|---|---|---|
| first request (cold), TTFT | 10.12 s | **0.253 s** |
| request after a 60 s idle, TTFT | 0.606 s | **0.121 s** |

The answer to the question: abundance does not dissolve the difference, it
scales it. On the M1 the residency gap was 3.8 GB; here it is **60.8 GB —
roughly half the machine wired down by an idle process** for as long as it
lives, exactly the M1 finding at 16x the size. The latency price Runner pays
for evictability also scales (10 s to fault 63 GB of weights back in against
0.25 s answered from residency), so the trade reads the same at both ends of
the hardware range: a dedicated inference box wants llama-server's residency;
a machine that is also a workstation wants an engine whose idle cost is 35 MB
and whose memory comes back with an HTTP call.

Two notes for anyone comparing against the README table: this llama.cpp is a
2026-08 self-built `010be968`, not the prebuilt b10639 the M1 row pinned, and
its idle CPU (0.12 s/min) is far better than the 161-wakeups/s behavior
measured there — idle discipline upstream has improved; the wired-residency
design difference is unchanged. Raw JSON reports for both engines were
captured by the script; rerun with the command lines above to reproduce.
