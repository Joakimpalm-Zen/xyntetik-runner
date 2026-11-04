# R3.30 step 1: the routing trace and the three policies (record, awaiting the trace)

**Status: the tools are ready and checked on a synthetic control. The first trace run was stopped by the host for
low memory (a CPU serve of the 12.9 GB file on a 16 GB box, during the first prompt), and it is not restarted
without the owner's word.** The rerun as planned:
- **Placement:** on the RTX 3070 with `--cpu-moe auto` (`trace.py ... 256 -- --gpu 99 --cpu-moe auto`), the
  placement the policies are scored against. `RUNNER_MOE_TRACE` covers the CUDA routers as well as the CPU's.
- **Length:** 256 tokens a prompt, as a memory guard.
- **File:** the Q8_0 (`JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF@4159d684`, sha256 `69e0ac02...`,
  12,925,417,760 bytes, verified against the Hub's LFS digest).

**Kill** (from the note): the best policy's implied decode gain over (a) under 1.3x stops the item at this
boundary. A flat routing kills it before any policy is read.

## The tools

| File | What |
|---|---|
| `trace.py` | one server per prompt, each with its own trace, the turn through the model's chat template, greedy |
| `user-turns.json` | the six SS-000 frozen prompts' user turns (code, copy, edit, long, prose, tool), cut from the `*.qwen3.txt` renders |
| `sim.py` | the trace statistics, then the three policies at equal bytes and the implied gain |
| `bench.sh` | decode at `--cpu-moe auto` and at 28, 24, 20 and 16 host banks, three interleaved rounds: today's speed, the placement record, and the slope that gives the host cost of one routed expert |
| `synth.py` | the control below |

**What is scored:** decode routings only (positions at or after the prompt's length). The policies see the
prefill's routings, since those happen before decode, but are never scored on them.

**Trace statistics:**
- **Working set:** distinct experts touched in 1, 8, 32 and 128 decode tokens, of 1,792.
- **Reuse distance:** decode tokens between two routings of one expert; uniform routing has a mean of 8.
- **Per-layer skew:** the share of a layer's routings taken by its 32 and 16 most used experts.

**The policies, at the bytes today's placement gives whole banks (K banks, B = 64 K slots):**
- **(a) static layers:** the shallowest K banks, which is today's `--cpu-moe auto`.
- **(b) static hot set:** the B most frequent (layer, expert) pairs, with frequencies taken from the other five
  prompts (leave one out). The in-sample figure is printed beside it only to show the overfit.
- **(c) capped LRU:**
  - B slots, warm-started with (a)'s experts.
  - A miss runs on the host.
  - A missed expert is admitted on its second touch (a ghost list of B keys).
  - At most R admissions a token, evicting the least recently used.

**Implied gain over (a):** `T_static / (T_static - (h - h_a) * 224 * t_e)`, where:
- `T_static` is today's milliseconds a token;
- `t_e` is the host cost of one routed expert, from `bench.sh`'s slope;
- `h` is the policy's share of routed experts served from the bank.

## The control: uniform routing

`synth.py` writes six prompts of uniform random top-8-of-64 routing in Mellum's shape (100 prompt and 300 decode
tokens each, seed 1). `sim.py` on it at K = 14 (half the banks), R = 4 and 8, with a stand-in model of 40 ms a
token and 0.25 ms an expert (`synthetic-check.txt`):

| | (a) static | (b) hot set, leave one out | (b) in sample | (c) LRU R=4 | (c) LRU R=8 |
|---|---:|---:|---:|---:|---:|
| served from the bank, mean of six | 0.500 | 0.500 | 0.552 | 0.498 | 0.498 |
| implied gain over (a) | 1.000x | 1.001x | 1.078x | 0.998x | 0.997x |

- **Honest policies:** every one reads 0.50 and about 1.00x on a flat routing, which is what the arithmetic
  requires.
- **The in-sample row** is the overfit the leave-one-out guards against: 0.55 from noise alone.
- **The skew noise floor:** on 300 uniform tokens, a layer's top-32 share reads 0.56, not 0.50. The real trace's
  skew is read against that floor, not against 0.50.
