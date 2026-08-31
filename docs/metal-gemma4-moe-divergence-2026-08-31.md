# gemma-4-26B-A4B generates token 0 on Metal while CPU is correct

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

## Why this could not be bisected further here

`RUNNER_DEBUG_ACT` instruments the CPU forward only. On the real model it emits
522 `ACT` lines per forward (per-layer `q-raw`/`k-raw`/`v-raw`, post-rope,
cached KV, `logits-raw`, `logits-final`, `top1`). The Metal arm emits the
header line and nothing else — 8 lines total, none of them per-layer. There is
therefore no activation-level A/B available on this backend, and the first
diverging tensor cannot be named without adding Metal-side `ACT` instrumentation.

That instrumentation is the natural next slice, and it is the prerequisite for
any fix attempt: without it, the boundary at layer 3 is the finest resolution
the current tooling can reach.

## Anchor

Per AGENTS.md, the absolute anchor here is external to the runner: the CPU arm
answers a general-knowledge question correctly (` Paris.`), and the same
question is answered correctly by two other models on the same Metal backend
on the same host. The comparison is not build-against-build.
