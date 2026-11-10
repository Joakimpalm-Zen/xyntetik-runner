# Deprecation review, November 2026 (R18.8)

Scope, as the owner chose it on 2026-11-10: the `RUNNER_*` environment switches and superseded kernel or
fallback paths. The text watermark is out of scope. Two shipped controls were folded in from R10.11: the
`--reasoning-temp` sampling and the `/shadow` offload.

This is a review. Nothing is removed here. Each removal is its own plan item with its own gate, and the two
controls are the owner's choice.

## The rules

- **Switches.** `RUNNER_*` variables "are tuning and diagnostic switches and can change in any release"
  ([versioning](versioning.md)). They are outside the compatibility promise, but each removal is recorded in
  the changelog.
- **Kernels and fallbacks.** A superseded kernel or fallback path goes once the path that replaced it has
  passed its gates, unless a supported backend, the canonical profile or a measured fallback envelope still
  needs it.
- **Flags and API fields** are inside the promise. One that has to go is first marked deprecated in
  `--help` and the changelog, keeps working with a warning for the rest of 1.x, and is removed no earlier
  than 2.0.0.

## Method

Read at main `d0dfb547`:

1. Every `RUNNER_...` name passed to `getenv` or to one of the helpers that read the environment
   (`env_i64`, `env_f64`, `env_u64`, `plat_env_file_get`, `metal_env_on`, `metal_env_default_on`) in `src/`:
   146 switches.
2. For each switch:
   - is it documented in the MANUAL, the README or a page under `docs/` (kernel routes, speed defaults,
     LoRA, session images and the others);
   - is it read by a test, a script, the Makefile or CI;
   - which file reads it.
3. The GPU route registry (`route_pinned` in `cuda.c`). A route a receipt pins is part of `--verify` replay.
4. For each candidate, the code at the switch: what the off path is, and whether anything reaches it
   without the switch.
5. Every kernel defined in `kernels*.cu`, `kernels*.cuh` and `kernels*.metal` against the dispatch code in
   `cuda.c` and `metal.m`.

## The switches by class

| class | count | rule |
|---|---:|---|
| Test seams (`RUNNER_TEST_*`, the injected failures) | 27 | keep: they are how tests reach failure paths |
| Routes a receipt pins: `CUDA_ATTN_DEC`, `CUDA_ATTN_TC`, `CUDA_ATTN_TILE`, `CUDA_SM80`, `CUDA_TC2`, `MOE_GROUPED` | 6 | keep: `--verify` replays them |
| Documented controls a test reads | 55 | keep, except the parked opt-ins below |
| Documented controls no test reads | 17 | each gets a test or is retired (item below) |
| Documented diagnostics (traces, profiles, stats, dumps, probes) | 20 | keep |
| Undocumented diagnostics | 10 | keep, and document them (item below) |
| Undocumented switches a test reads | 8 | per-switch review (item below) |
| Undocumented switches no test reads | 3 | retire candidates (below) |

The 17 documented controls with no test: `ATTN_BATCH`, `CUDA_ATTN_MLA`, `CUDA_ATTN_TC_FULL`,
`CUDA_ATTN_TILE512`, `CUDA_GRAPH_OFF`, `CUDA_GRAPH_RECUR`, `CUDA_KV_GATHER`, `CUDA_RECUR_SEQ`,
`FORCE_REQUANT`, `GRAMMAR_FF`, `METAL_MV2`, `METAL_TENSOR`, `REQUANT_ONLY`, `RESPONSES_STORE_MB`,
`SCHEMA_ALLOW_WS`, `TPOOL_SPIN`, `VRAM_PRIORITY`.

The 10 undocumented diagnostics: `DEBUG_MOE`, `DEBUG_VRAM`, `METAL_MOE_ROUTE_TRACE`, `METAL_NAN_TRACE`,
`MOE_PROBE`, `MTP_DUMP`, `SAMPLE_STATS`, `TOOL_TRACE`, `TRAIN_FULL_DUMP`, `TRAIN_FULL_DUMP_TENSORS`.

The 8 undocumented switches a test reads: `CUDA_WINDOW`, `CURL`, `METAL_ATTN_CHUNK`, `METAL_MOE_BATCH`,
`MV_FUSE`, `TRAIN_MM_KERNEL`, `TRAIN_MVT_KERNEL`, `UNVERIFIED_LEASE_TTL`. Some are test seams in all but name, and some are A/B arms a
gate uses; each needs reading before a verdict.

## The retire candidates

### Switches of default-on routes and the undocumented ones

| switch | what it does | off path reached without it | verdict |
|---|---|---|---|
| `RUNNER_ATTN_SPLIT` | `=0` turns the CPU attention split off (on since 2026-09-30) | yes: `attn_split_wanted` takes the unsplit path for one thread, short spans, or a head split that already keeps the threads busy; `tests/test_attn_split.c` sets the field itself | **retire the switch**, keep the path |
| `RUNNER_METAL_MVFC` | `=0` keeps the older matrix-vector walk (the A/B arm for R3.24 (a), on since 2026-10-31) | to be read in the item | **retire the switch** if the walk is reached otherwise; otherwise retire both |
| `RUNNER_CUDA_ATTN_TILE512` | `=0` keeps `k_attn` for head dim 512 (documented in `kernel-routes.md`) | yes: `k_attn` is the fallback for odd head dims and uneven GQA | **retire the switch** and its line in `kernel-routes.md`, keep the kernel |
| `RUNNER_CUDA_ATTN_MLA` | `=0` keeps the non-MLA decode for MLA models (on since 2026-11-05) | yes: a build without the kernel takes it | **hold** until one release has shipped with the route |
| `RUNNER_METAL_COL_TILE` | Metal column tile size (default 8) | not a route: a tuning knob | **hold**: R3.39.6 (per-machine kernel tuning) is open and may use it |

### Opt-ins that are already parked

[Speed defaults](speed-defaults.md) keeps two lists of routes that stay off: those with a failing row, which
"stay off unless a narrower scope gets its own passing row", and those measured with no gain. Each is reached
only by asking for it, and some carry code of their own. Checked against the open plan, most are still in use
by work under way:

| switch | parked because | open work | verdict |
|---|---|---|---|
| `RUNNER_PREFETCH` | the shipped hint was slower or no faster on macOS, Linux and Windows | under rework (R3.39.11): the shipped hint is placed after the GPU upload has read the file; an early, cold-only hint made cold GPU loads 1.35x faster on Windows with warm loads unchanged (R3.39.9, R3.39.10). The Linux rows are pending, and a pass makes it a default | **not a candidate** |
| `RUNNER_TIEDV` | saves memory, not byte-identical | the tied-V device path (R4.6) | **not a candidate** |
| `RUNNER_CUDA_I8` | Q8_0 reads 1.51x the default route's mean KL | dense speed work on large files (R4.26.16), the tolerance gates (R6.30) | **not a candidate** |
| `RUNNER_CPU_I8`, `RUNNER_CPU_I8K`, `RUNNER_CPU_I8_PREFILL` | fail the bar, or 1.22x the default route's mean KL | table-lookup integer kernels for low-bit weights (R3.1.5, deferred) | **keep parked** with it |
| `RUNNER_CUDA_ATTN_TC_FULL` | gpt-oss lands farther from the float32 reference than the tile (Wilcoxon p 3e-18) | none | **retire** (R18.8.1) |
| `RUNNER_METAL_TENSOR` | parity, no gain | none | the owner's choice (R18.8.8) |
| `RUNNER_METAL_MOE_EM` | 3 to 4% slower | none | the owner's choice (R18.8.8) |

The code behind the last two is their own: the Metal tensor kernels (`kernels_tensor.metal`) and the
expert-major MoE kernels, bit-identical to the slot-major default. Retiring the switch retires that code.

## Superseded kernels

Every kernel defined in the CUDA sources (164) and the Metal sources (103) is named by the dispatch code, so
no kernel is unreachable in the build. That is not the same as being used: the kernels behind the parked
opt-ins above run only when asked for, and most of those are in use by open work. The candidates' own off
paths are shape fallbacks (`k_attn`), build fallbacks (the non-MLA decode) or selection branches. So the
removals this review recommends without a further choice are switches and one branch; the only kernel
removals in sight are the two Metal routes in R18.8.8.

## The two controls

**`--reasoning-temp`** (and the per-request `reasoning_temperature`, `reasoning_top_p`, `reasoning_min_p`,
`reasoning_top_k`). It was measured as not helping at 14B.
- It is off by default and costs nothing while unset.
- It is reported in `/v1/capabilities` and in `runner_telemetry`.
- The Suite's `runner-control` client sends `reasoning_temperature`, with a fixture and tests, so removing
  it would break a client.
- **Recommended:** keep it, and state the 14B measurement in the MANUAL section beside it.

**The `/shadow` offload.** This is `--shadow-mode` and the Python package's `xyntetik_runner.shadow` (22
files): hooks and a `/shadow` skill for Claude Code and Codex, plus `shadow delegate`.
- The shadow research it served closed in September 2026.
- Shade carries its own `xyntetik_shade.shadow`.
- The Suite's code reaches it only in its home-folder migration.
- **Recommended:** deprecate it in 1.x, with a warning in `--help` and at install, and remove it at 2.0.0
  as the versioning policy requires for a flag.

## Items filed

Each is a child of R18.8 in the plan:
- R18.8.1: retire `RUNNER_CUDA_ATTN_TC_FULL` and its branch;
- R18.8.2: retire the A/B switches of default-on routes: `ATTN_SPLIT`, `CUDA_ATTN_TILE512`, `METAL_MVFC`
  (reading its off path first), and `CUDA_ATTN_MLA` after a release;
- R18.8.3: give each of the 17 documented controls with no test a test, or retire it;
- R18.8.4: review the 8 undocumented switches a test reads;
- R18.8.5: document the 10 undocumented diagnostics;
- R18.8.6 and R18.8.7: the owner's two choices, `--reasoning-temp` and the `/shadow` offload;
- R18.8.8: the choice on `RUNNER_METAL_TENSOR` and `RUNNER_METAL_MOE_EM`, retire them with their kernels or
  keep them parked.

Every removal is recorded in the changelog, as the rule for switches requires.
