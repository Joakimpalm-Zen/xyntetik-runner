# Timing A/A pilot, RTX 3070 (ZEN-GAMING), 2026-10-07 03:18-03:20

Feasibility evidence for the Blackwell lab's ERASE question (can a box
resolve a ~5% effect on the GPU path?), not a G0-I result. Driver:
`scripts/aa-pilot.py`. Runner 1.0.2 release binary (sha256 28de83aa...),
Qwen/Qwen3-4B-GGUF Q4_K_M (sha256 7485fe6f..., byte-identical to the file
the lab's G0-I used), full offload, context 4096, one server. One warm-up
block of 5 requests, then 16 blocks in ABBA order (8 pairs), each block the
same 20 greedy non-thinking chat requests at max_tokens 64. Per block: wall
seconds, prefill and decode tok/s from the server's own counters, process
CPU-seconds (psutil). Windows 11, i7-7700K, driver 596.36; box not quieted
(Sunshine, Defender, Search on; Docker's WSL VM up and idle; GE-009's
container had just exited). Lock `box-gpu.lock` held for the run.

Statistic: mean ln(A/B) over pairs, 95% half-width (t), resolution = 2 x
half-width; the lab's gate is 0.10.

| metric | all 8 pairs: resolution | gate | pairs 2-8 (block 0 cold): resolution | gate | CV over blocks 1-15 |
|---|---|---|---|---|---|
| decode tok/s | 0.0078 | pass | 0.0072 | pass | 0.4% |
| wall s | 0.0753 | pass | 0.0111 | pass | 0.5% |
| CPU-seconds | 0.3476 | fail | 0.0250 | pass | 0.8% |
| prefill tok/s | 0.6019 | fail | 0.1021 | fail | 3.2% |

Block 0, the first scored block, was still cold despite the warm-up
(prefill 298 vs ~820 tok/s, CPU 15.6 s vs ~8.6 s); every "fail" on the
all-8 rows is that one pair. A longer warm-up (one full block) is the
obvious fix for a real run. Prefill tok/s on 20 short prompts is a noisy
metric by construction (each block prefills ~400 tokens in ~0.5 s total);
decode and wall are the ones the gate can lean on here.

Raw: `aa-pilot.json` (blocks, statistics, sensitivity, notes), `aa-pilot.log`.
