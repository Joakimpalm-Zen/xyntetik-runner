# gemma-4-26B-A4B generated token 0 on Metal: an unclamped GELU in the routed-expert kernel

*Found, root-caused and FIXED 2026-08-31. The bisection below is kept because
the fixture gates could not have found this and the reasoning is the reusable
part.*

Measured 2026-08-31 on an Apple M5 Max (18 cores, 128 GB unified, macOS 26.5
build 25F71), the first Apple-silicon host in this project above the 8 GB M1
class. Binary: `runner` 0.4.4 built from `main` at `d14a51c`, Metal shader
source sha256 `a0a923b9...70cd0`. `--caps` reports
`backend=metal name="Apple M5 Max" unified_memory=true moe=true eseries=true`,
`max_working_set_bytes=115448725504`.

This is the big-model Metal validation the suite plan's Blocked ledger listed
under "16 GB Apple Silicon at zero baseline swap". Baseline swap on this host
was zero (`vm.swapusage total=0.00M used=0.00M`, pageouts 0) and the machine is
neither DEP- nor MDM-enrolled, so nothing external was competing for residency.

## The headline: gpt-oss-20b passes byte-exact; gemma-4-26B-A4B emits only token 0

Greedy (`--temp 0`), same prompt `"The capital of France is"`, `-n 32`,
CPU arm `--gpu off` vs Metal arm `--gpu auto`:

| model | file sha256 | CPU vs Metal |
|---|---|:--|
| `gpt-oss-20b-MXFP4.gguf` (11.28 GB) | `27cd6c43...35901` | **byte-identical** |
| `gemma-3-4b-it-Q4_K_M.gguf` (2.32 GB, dense) | `882e8d2d...` | **byte-identical** |
| `gemma-4-26B-A4B-it-Q4_0.gguf` (13.61 GB, MoE) | `d208665a...04a03` | **DIVERGES** |

The gpt-oss artifact's sha256 is the exact one the pressure-governed residency
plan pins for Phase 6, so the model under test is the intended one.

Transcripts (`--transcript`) make the failure unambiguous. Identical prompt
tokenization on both arms, `[2, 818, 5279, 529, 7001, 563]`:

| arm | output tokens | text |
|---|---|---|
| CPU | `[9079, 236761, 107, 100, 236800, 236786, 236778, 236786]` | `" Paris.\n<\|channel>3/2/"` |
| Metal | `[0, 0, 0, 0, 0, 0, 0, 0]` | `""` |

The CPU arm answers the question correctly. The Metal arm emits token id 0
`n` times and reports `finish: "length"` — it is not stopping early, it is
generating a constant. The logits reaching the sampler on the Metal path are
degenerate; on CPU the same forward gives `top1 token=9079 logit=21.57` with
`inf=0 nan=0` at every instrumented stage.

## What the bisection rules out

`--gpu-layers N` forces N *leading* layers onto the GPU. The failure has a
sharp boundary:

| `--gpu-layers` | result |
|---:|:--|
| 0, 1, 2, 3 | correct (` Paris.`) |
| 4, 8, 15, 30 | token 0 only |

Ruled out by measurement:

- **Not the mat-mul or attention kernel.** Output is unchanged under
  `RUNNER_METAL_MM=0`, `RUNNER_METAL_ATTN_COOP=0`, and both together — the
  same env dimensions `test-metal-gemma4-moe` uses.
- **Not the tokenizer.** Prompt token ids are identical on both arms.
- **Not the missing-V-projection special case.** gemma-4 global layers publish
  no V projection (`model.c:6458`, `metal.m:2156`, V is the raw K). Those
  layers are at indices 5, 11, 17, 23, 29 (`wv=0`, `swa=0`, `q_dim=8192`) —
  all *after* the failure boundary at layer 3. Layers 0-3 are uniformly
  `swa=1 wv=1 q_dim=4096 kv_dim=2048`. `v_rmsnorm=1` for this model, so the
  `enc_headnorm_n(ly->wv ? g->vt : g->kt, ...)` path that stands in for the
  CPU's unconditional `memcpy(v_tmp, k_tmp)` is in fact taken.
- **Not a scale-free config error.** The tiny synthetic fixtures pass:
  `test-metal-moe`, `test-metal-gptoss-moe`, `test-metal-gemma4-moe` and
  `test-metal-gemma4-hetero` are all green on this host. Only the real
  artifact fails, which is precisely why fixture gates are not big-model
  validation.

**Not ruled out — and explicitly untested:** `--cpu-moe`. Its help text scopes
it to CUDA ("while CUDA runs attention and other dense tensors on the GPU"),
and `cpu_moe` appears **zero** times in `src/metal.m`. Running `--cpu-moe` on
the Metal arm changed nothing because the flag is not wired to this backend at
all. It is not evidence that the expert path is innocent.

## Root cause: the clamp exists in one GELU kernel and was missed in its twin

Adding a staged probe to the Metal path (`RUNNER_METAL_NAN_TRACE=2/3/4`, this
commit) walks the NaN inward in four steps:

| level | first bad stage |
|---|---|
| 1 (per layer) | `L3 resid` |
| 2 (per layer stage) | `L3 ffn-out` |
| 3 (inside the MoE FFN) | `L3 moe:experts-out` |
| 4 (inside the expert matvecs) | **`L3 exp:actmul[1882]`** |

`exp:gate_up` is clean going in and `exp:actmul` comes out NaN, so the
activation itself manufactured it. `k_moe_actmul`'s GELU branch computed

```c
float t = tanh(0.7978845608f * (x + 0.044715f * x * x * x));
```

with no clamp. The dense `k_gelu_mul` computes the same expression **with**
`clamp(a, -16.0f, 16.0f)`, and carries a comment that describes this exact
failure: Metal compiles with fast math, `tanh()` goes through `exp(2a)`, large
|a| overflows to inf, `inf/inf` is NaN — and *"gemma-3-4b's layer-0 gate
produced NaN logits here, and the model emitted only token 0."*

That guard was added for the dense path. `k_moe_actmul` is its routed-expert
twin and was missed. gemma-4-26B-A4B is the first model in this project's set
that both routes through the MoE kernel and drives the gate hard enough to
reach the overflow, which it does at layer 3. The reported symptom is
identical to the one already written down for the dense kernel.

The fix is the same clamp. tanh is exactly +/-1.0f in fp32 well inside +/-16,
so it cannot change a representable result.

## After the fix

No NaN at any probe level. Across eight realistic prompts, CPU and Metal are
byte-identical on **7 of 8** — before the fix the model produced only token 0
on every one of them, so the correct baseline is 0/8.

The eighth is `"The capital of France is"`, a raw completion handed to an
instruction-tuned model. It answers `" Paris."` correctly on both arms and
diverges at token 14, picking 236786 (CPU) vs 236787 (GPU) — adjacent ids in a
degenerate date-repetition loop. This is chaotic amplification at a near-tie,
not a wrong op, and the evidence is the repo's own test for it: **the CPU arm
disagrees with itself** on that prompt under `--kv q8` (`"0.5"` against
`"<|channel>3/2/2025 3/2"`), while `-t 3` reproduces the default exactly. Per
the README's standing policy, a numerically sensitive model gets a measured
self-sensitivity floor rather than a cross-engine token-identity claim.

`test-metal-bigmodel` therefore defaults to a realistic instruction prompt and
takes `BIGPROMPT=` for a specific investigation. All three real artifacts on
this host — gpt-oss-20b-MXFP4, gemma-3-4b dense, gemma-4-26B-A4B — pass it.

## What this says about fixture gates

Every tiny MoE fixture passed throughout, including
`test-metal-gemma4-moe`, and so did `test-metal-gelu-overflow` — which exists
precisely to catch this hazard, but exercises the DENSE kernel on a gemma3
fixture, not the routed-expert twin. A sub-1 MiB fixture cannot drive the gate
into fp32 overflow, so no fixture in the suite could have found this. That is
the argument for `test-metal-bigmodel` existing at all.

## Anchor

Per AGENTS.md, the absolute anchor is external to the runner: the CPU arm
answers a general-knowledge question correctly (` Paris.`), two other models
answer correctly on the same Metal backend on the same host, and the fix is
validated against a documented property of tanh in fp32 rather than against
another build of this code.
