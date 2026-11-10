# Changelog

Current release and unreleased changes. Compatibility follows the
[versioning policy](docs/versioning.md).

Older releases (v1.11.0 and earlier) are preserved in the
[changelog archive](docs/changelog-archive.md).

## Unreleased

- **A cancel route keyed by request id: `POST /v1/runner/requests/{id}/cancel` (R10.12.3.2).**
  - **What it is:** a client that wants an answer, rather than polling `/health`, names a running completion request.
    It can use the response id, or the `X-Request-Id` header it sent with the request. The header is how a buffered
    client knows the id before its reply arrives.
  - **How it answers:** the cancel is served from the accept loop, so on a one-slot server it does not queue behind
    the request it cancels. It replies once that request has stopped, with the phase it was in and how many tokens
    it had generated. An unknown id gets `404`.
  - **The cancelled reply:** a cut during generation keeps its partial output (`finish_reason` `length`,
    `finish_detail` `cancelled`). A cut while the prompt was processed answers `409 cancelled`. The server stays up.
  - **Ids:** a client id is checked (1 to 64 characters from `A-Z a-z 0-9 . _ : -`), and one that an active request
    already holds is refused with `409`. Request ids stay out of `/health`, which anything on loopback can read.
  - **Receipts:** a turn the server cut from outside the model (a cancel, the request's `timeout`, the drain) now
    records `serve.cut`, and `--verify` replays the recorded tokens and no further. Before, such a receipt replayed
    `DIVERGED`, because the replay ran on to `max_tokens`.

- **Docs: gemma-4-26B-A4B now has its publisher float32 comparison (R6.11).** The support entry listed it as owed.
  - **What was run:** two files, scored against the transformers forward of `google/gemma-4-26B-A4B-it` in float32
    on chat text, 2,000 positions ([the record](docs/gemma4-26b-reference-2026-11-10.md)).
  - **A fresh bf16 conversion** of the checkpoint reads mean KL 0.0060, margin-qualified top-1 99.79% (CPU), so the
    MoE forward tracks the reference.
  - **The listed QAT Q4_0 file** reads 0.2183 and 90.59% (CUDA), its own distance.
  - **Raw text, reported first there:** the conversion reads 0.3226, which is not explained.

- **The first CUDA load on a new card or driver says why it waits (R10.29).**
  - **Before:** the driver JIT-compiles the embedded PTX the first time, and the process sat silent. Measured on
    an RTX 5070: 33.4 s cold, 0.40 s warm from the driver's cache.
  - **Now:** a module load still running after 2 s says, once per process, on stderr: `gpu: compiling the GPU
    kernels for this card and driver; this happens once per card and driver, later loads take them from the
    driver's cache`. A warm load ends first and says nothing. stdout and receipts are unchanged.
  - **Checks:** `test-slow-note` checks the note itself on every host: said once while a slow wait runs, never
    for a quick one.

- **`RUNNER_PREFETCH` stays opt-in on Linux and Windows, now measured there (R3.39.5).** On Windows with CUDA,
  turning it on made every load slower, with identical output: 1.159x the time from a cold file, 1.122x and 1.118x
  warm on two files. On Linux, on a host capped at 60 read IOPS, it was 1.229x on the CPU and 1.006x and 0.9995x on
  CUDA. On a CUDA load the hint is issued after the device upload has read the weights, so it cannot speed that
  upload. `docs/speed-defaults.md` and the manual record the rows; a run that does not set it is unchanged.
- **`--verify` replays a served turn under a reasoning budget or the loop guard (R1.1.3.3).**
  - **Before:** such a receipt was `UNVERIFIABLE`. The record named the budget's size and that the guard was on,
    but not the sentence the forced close writes or the guard's settings.
  - **Now:** the `reasoning_budget` constraint carries `message_tokens`, the token ids of that sentence (`[]` for a
    bare close). The `loop_guard` constraint carries `span`, `repeats`, `window`, `max_closes` and `everywhere`.
    The replay installs them as the server does, primes from the prompt text as the server does, and closes or ends
    the turn where they say. A record whose settings were changed replays `DIVERGED`. A record from before the
    fields is `UNVERIFIABLE`, naming the one it lacks.
  - **Still refused:** the reasoning channel's own sampling (`reasoning_temperature` and its siblings).

- **`--verify` replays a served turn cut by a stop sequence (R1.1.3.2).**
  - **Before:** a receipt shaped by `stop`, `stop_sequences` or `stop_token_ids` was `UNVERIFIABLE`, although the
    record already carried the sequences.
  - **Now:** the replay re-does the cut. It matches the stop strings on the visible text only, after the
    thinking-tag splitter, as the server does, and gives the engine the stop ids. A record cut anywhere else
    replays `DIVERGED`, because the cut is not taken from the record.
  - **New receipt field:** `serve.reasoning_open` says whether the prompt left the turn inside the reasoning block,
    so the splitter starts where the server's did. On a model with thinking tags, a stop-cut chat record from
    before the field is `UNVERIFIABLE`, naming it.
  - **Found beside it:** on a model with thinking tags, a served chat turn under a JSON schema or JSON mode
    replayed `DIVERGED` at its first token or byte. The server forwards the reasoning bytes beside the document
    and, when the prompt opened the reasoning block, holds the grammar until that block closes; the replay did
    neither. It now starts where the server started.

- **Fix (CUDA): the F32 and F16 canonical-order matvecs finished an odd row fused, against the host (R12.1b).**
  - **The bug:** `k_mvcanon_f32` and `k_mvcanon_f16` finish a row that is not a whole 32 (F32) or 16 (F16) stride on
    one lane, with `s += w * x`. nvcc fused it into an `fma` (its default `--fmad=true`). The canonical host rounds the
    product and the sum apart (`-ffp-contract=off`), so on such rows the device differed from the host it is defined
    to equal.
  - **The fix:** the tails are spelled `__fmul_rn` and `__fadd_rn`.
  - **Checks:** two fixtures with a 200-wide FFN, `test-oddff.gguf` (F32) and `test-oddff-f16.gguf` (F16). Before the
    fix, 13 to 19 of ffn_down's 64 rows differed; after it, none. `make-test-model.py` gains `--quant f16` for the
    second fixture, since the runner's quantizer writes no type for a row that is not a whole 32-block.

- **CUDA: the canonical-order matvec gains Q4_0, Q4_K, Q5_K and Q6_K, bit-identical to the host (R12.1b).**
  - **What it is:** `gpu_mvcanon` is the device forward matvec of the training path (R8.7.1). It computes the host's
    canonical association, so an adapter trained through it keeps the CPU's bytes. It took F32, F16 and Q8_0; every
    projection of a Q4_K_M model now has a kernel.
  - **How it holds:** every product that meets an addition outside an explicit `fmaf` is spelled `__fmul_rn` or
    `__fadd_rn`, so nvcc's default contraction cannot fuse what the CPU rounds twice.
  - **Checks:** `test-mvcanon` compares every row with memcmp and now walks every layer's projections and the head.
    It runs one fixture per type: the new `test-q40.gguf` and `test-q5k.gguf`, beside `test-q4k.gguf` and
    `test-q6k.gguf`. A type a fixture is named for must reach the kernel, or the check fails.
  - **Fix to the check itself:** its host reference compiled `quants.c` with `-ffast-math`, against its own comment.
    Under it, gcc reassociates the host Q4_K kernel, so the reference stopped being the canonical order. It now takes
    the engine's strict flags, as `test-canon-kernels` does.

- **The T3 (canonical) build gains Q4_0, Q4_K, Q5_K and Q6_K, and names the types it cannot run canonically
  (R12.1b).**
  - **The kernels:** each type has one reduction tree on every ISA, the AVX2 kernel's own shape (the reference, as
    for Q8_0). There are NEON and scalar mirrors.
    - Q4_0 is Q8_0's tree on nibbles.
    - Q4_K and Q5_K: four-term lane sums per 32-weight sub-block, scaled into eight lane accumulators, with the min
      term summed apart from per-sub-block tree sums.
    - Q6_K: two-term lane sums per 16 weights.
  - **What this covers:** a Q4_K_M model (Q4_K, Q5_K, Q6_K, Q8_0 and F32 on unsloth's Qwen3.5-4B) now runs every dot
    product in the canonical order.
  - **The cost on x86-64:** none at the kernel against the same-shape kernel. The canonical build gives up the decode
    path's reuse of the per-matvec x sums: Q4_K 1.47x and Q5_K 1.34x a matvec. End to end, T3 decodes Qwen3.5-4B
    Q4_K_M 6 to 14% slower than the default build, its usual strict-float cost.
  - **Checks:** `tests/test_canon_kernels.c` holds each kernel bit for bit to an independent definition. That is the
    AVX2 kernel on x86, and the scalar one with RUNNER_NO_SIMD; NEON in macOS CI.
  - **The gap that was silent:** every type without a canonical kernel fell through to the ISA-native kernels without
    a word, so a T3 build was not the same on arm64 and x86-64 for that model and nothing said so.
  - **Now:** the load says "canon: BF16 have no canonical-order kernel in this build ...", and the route line and the
    receipt's `profile.route` carry `canon_gaps` (empty when every type is canonical). A default build writes none of
    it.
  - **Checks:** `tests/test_canon_gaps.py`, in CI's t3 job.

- **Fix: `--fit` charged deepseek2's KV at the header's shape, up to 10.7x the cache the loader builds (R3.48).**
  - **What the loader builds:** its default is the absorbed MLA form. That is one cache row a token and layer,
    `kv_lora_rank + rope` wide, which every head and V read. Q is that wide per head.
  - **What the fit charged:** `head_count_kv x key_length`, K and V each, which is the naive form
    (`RUNNER_MLA_NAIVE=1`).
  - **The cost on DeepSeek-V2-Lite-Chat Q6_K at ctx 131072:** the fit read 40.50 GiB of KV against the loader's 3.80.
    On a 24 GB slice it predicted 4 to 8 layers on the GPU at `--reserve-vram` 40 to 70, where the loader places 14 to
    25.
  - **Now:** the fit takes the loader's MLA geometry, absorbed unless `RUNNER_MLA_NAIVE` is set.
    - Its KV equals the loader's in f16, q8, k8v4 and fp4 (k8v4 keeps a V row of its own).
    - Its split equals the loader's at 8 points across ctx 32k and 131k.
    - Its fixed charge is byte-exact, on V2-Lite and on the fixture in both forms.
  - **Checks:**
    - `tests/test_fit_check.py`: the fit and the loader against the fixture's declared absorbed and naive shapes, at
      the default and V2-Lite attention geometry.
    - `tests/test_cuda_fit_hybrid.py`: deepseek2's fixed charge in both forms.

- **Fix: a hybrid's recurrent state was held for every layer, including the layers that have none (R3.47).**
  - **The bug:** qwen35's Gated DeltaNet fold, Mamba-2's conv window and SSM state (granitehybrid, nemotron_h) and
    lfm2's conv ring sat in per-layer buffers sized for every layer, attention and MLP-only layers included. That held
    on the host, on CUDA and on Metal, and in every prefix-cache entry, session image and KV snapshot.
  - **Now:** each layer that holds state has a slot (`model_rec_slot`), and nothing else has one. Greedy text and
    teacher-forced scores are byte-identical to the previous build on Qwen3.5-4B (CPU, CUDA), granite-4.0-h-micro
    (CPU, CUDA) and Qwen3.6-35B-A3B (CUDA).
  - **Per sequence, at a fixed context:**

    | Model | Host state | Device instance |
    |---|---|---|
    | Qwen3.5-4B | 70.3 to 52.7 MB (24 of 32 layers) | 35 MB less |
    | Qwen3.6-35B-A3B | 87.8 to 65.9 MB (30 of 40) | 44 MB less |
    | granite-4.0-h-micro | 86.0 to 77.4 MB (36 of 40) | 17 MB less |

    Every prefix-cache entry shrinks by the host figure.
  - **`-v`** prints the state beside the KV cache: "recurrent state 52.7 MB (24 of 32 layers)".
  - **Saved state:** 1.15.0's layout (a slot for every layer, and R3.44's KV rows) is refused by name, never misread.
    `RUNNER_KV_RECURRENT_ROWS=1` keeps that whole layout and resumes it. State written by development builds between
    R3.44 and this fix is refused too.
  - **The plan and `--fit`** charge the stateful layers only (R3.46's `model_gpu_recurrent_cost`).
  - **`--fit` reports the CPU for the families the loader never puts on a GPU:** lfm2, lfm2moe, flint, LayerNorm
    models, grouped-norm k2-horizon, and deepseek2 off CUDA. It used to predict a device split the load never made.
    `model_gpu_cpu_only_why` now answers for both.
  - **Checks:**
    - `tests/test_cuda_rec_slots.py` (CUDA): scoring stays on the device under the new layout and matches the old
      one's scores, on seven hybrid fixtures.
    - `tests/test_qwen35_kv_rows.py`: the state shrinks by exactly the stateless layers' slots, and an old-layout image
      is refused by name on qwen35 and Mamba-2.
    - The hybrid fixtures run clean under ASan and UBSan in both layouts.

- **Fix: `--fit`'s GPU split on hybrids now equals the loader's, and the CUDA plan stops charging qwen35's recurrent
  state twice (R3.46).**
  - **What the fit left out:** each hybrid's device state, held twice (live, and for the speculative rollback), plus
    its mixer scratch. That is qwen35's Gated DeltaNet fold, and Mamba-2's conv window and SSM state.
  - **The FFN scratch width:** it took 4 x n_embd where the loader takes the expert width (Qwen3.6-35B-A3B declares no
    dense one) or a per-layer array's widest entry.
  - **The cost:** Qwen3.6-35B-A3B at ctx 131072 read 40/40 layers on the fit against the loader's 39/40, and
    granite-4.0-h-micro read 3 to 4 layers high.
  - **Now:** both charge `model_gpu_recurrent_cost`.
    - The fit equals the loader at every point measured: the 35B at 32k, 131k and 262k, and granite-micro across
      `--reserve-vram` 5 to 10 at 32k and 131k.
    - The fixed charge is byte-exact on qwen35, qwen35moe, clef, qwen4exp, granitehybrid and nemotron_h fixtures and on
      the 35B.
  - **The loader:** it charged qwen35's fold and scratch once in `cuda_instance_dims` and again in `shared_plan_budget`.
    That is 4x the fold it allocates: 0.18 GB on the 35B, 0.32 GB on Qwen3.8-Flash-Next. Now charged once.
    - **On the 35B** no layer or bank moves at 32k to 262k.
    - **Under `--cpu-moe auto` at 262k** the prompt window widens from 256 to 512 columns. Prefill of a 7,370-token
      prompt goes from 147 to 175 tok/s on main to 183 to 189 tok/s, with identical text.
    - **The 14 qwen35-family roster rows with CUDA pins** keep their splits (checked from the headers, and Flash-Next
      on the loader), so no pin moves.
  - **`RUNNER_DEBUG_VRAM`** prints both figures in bytes: `fit: (bytes: fixed F, headroom H)` and `gpu-split: (bytes:
    ...)`.
  - **Checks:** `tests/test_cuda_fit_hybrid.py` (CUDA) holds the two charges equal on six hybrid fixtures and a dense
    control.

- **The shared prefix cache's default budget is sized from the served model (R4.26.13).**
  - **Before:** `RUNNER_PREFIX_CACHE_MB` was a fixed 512 MB. An entry over half the budget is not stored whole: a
    recurrent model stores nothing, a dense one a truncated prefix. Even after R3.44, Qwen3.5-4B could not cache a
    prompt past 6,048 tokens.
  - **Now, when it is unset:** each model a server loads raises the budget to hold one of its own full-context entries
    under that rule, twice the entry. It is never below 512 MB and never above an eighth of the RAM the process may
    use. The server prints the figure at startup; a set value is used as given.
  - **Qwen3.5-4B Q4_K_M, `-c 16384`:** budget 1,158 MB.
    - A 7,370-token prompt is now stored, so a later request that extends it forks 7,370 cached tokens.
    - Prefill 0.14 s on CUDA against 3.46 s cold, and 0.46 s on the CPU against 111.7 s cold.
    - The text is identical to the cold run's.
    - Main stored nothing and re-prefilled.
  - **Qwen3.6-35B-A3B Q4_K_M, `-c 32768`:** budget 1,447 MB. At 7,370 tokens main already stored the entry (it fits
    under 256 MB), so nothing changes there. Past about 8,860 tokens main stored nothing. At 19,998 tokens main re-prefilled in 107.7 s; the fix forks 19,988 cached tokens in 0.40 s, with identical text.
  - **CUDA too:** a GPU-backed hybrid slot forks a shared entry through the host (R4.26.7), against what the MANUAL
    said. It is corrected.
  - **Checks:**
    - `tests/test_prefix.c` checks the sizing rule at its bounds: the floor, twice the entry, the RAM cap, saturation.
    - `tests/test_prefix_budget.py` checks the server's reported budget against a fixture's declared entry cost and
      the RAM `--caps` reports. It also checks the floor, and that a set value wins.

- **Fix: `--fit` counted a declared NextN/MTP block as a layer of the trunk (R3.45).**
  - **The bug:** a GGUF that declares a predictor block (`nextn_predict_layers`) counts it in `block_count`. The
    loader takes it off the trunk and gives it KV rows and device memory only when `--mtp` (or `--draft mtp`) binds
    it.
  - **What `--fit` did instead:** it took `block_count` whole. So the block was a layer in the count, in the KV
    cache and in the GPU split.
  - **On Qwen3.6-35B-A3B Q4_K_M at ctx 131072:** 41 layers and 2.75 GiB of KV, against the loader's 40 layers and
    2.50 GiB (2,684.4 MB). Before R3.44's KV fix it was 10.25 GiB against 10.00.
  - **Now:** the model line names the block ("1 NextN/MTP block not charged (pass --mtp to draft from it)"). Under
    `--mtp` its rows are charged, and its tensors and rows join the GPU split's fixed charge, as the CUDA plan charges
    them.
  - **What does not change:** a file that declares no block gets the same `--fit` output as before (10 files, with
    and without a GPU).
  - **Not covered:** the default MTP drafting of one-shot JSON generations is not modeled (one measured file, CUDA,
    full offload only).
  - **Checks:** `tests/test_qwen35_kv_rows.py` holds the fit and the loader to the fixture's declared geometry, with
    and without `--mtp`.

- **Fix: Qwen3.5-class hybrids reserved KV rows for their Gated DeltaNet layers, 4x the KV they need (R3.44).**
  - **The bug:** qwen35, qwen35moe, qwen4exp (Qwen3.8-Flash-Next) and clef run three DeltaNet layers to one attention
    layer. A DeltaNet layer keeps its state in the recurrent fold and never reads or writes a KV row, but every layer
    was given rows of the attention shape: on the host, on CUDA and Metal, in the placement plan and the reservation
    auto-fit, in every prefix-cache entry and in every saved state.
  - **What does not change:** the default prefix cache still stores nothing for a long prompt. On Qwen3.5-4B a
    7,370-token prompt's entry is 312 MB, against 1,036 MB before, and an entry over half the 512 MB budget is still
    not stored. The longest prompt it caches rises from about 1,500 to about 6,000 tokens. Sizing that budget is
    R4.26.13.
  - **KV now** (f16, main against the fix, on Qwen3.5-4B and Qwen3.6-35B-A3B):

    | Model | ctx 4096 | ctx 32768 |
    |---|---|---|
    | Qwen3.5-4B | 537 to 134 MB | 4,295 to 1,074 MB |
    | Qwen3.6-35B-A3B | 336 to 84 MB | 2,684 to 671 MB |

  - **On a 24 GB CUDA slice:**
    - Qwen3.5-4B's auto-fit context (`-c 0 --reserve-vram 90`) rises from 151,222 to 262,144, the training context,
      with 12.4 GB still free.
    - Qwen3.6-35B-A3B under `--cpu-moe auto`, expert banks and prompt window:

      | Context | Banks | Window (columns) |
      |---|---|---|
      | 32,768 | 40 to 40 | 256 to 512 |
      | 131,072 | 23 to 39 | 64 to 512 |
      | 262,144 | 0 to 33 | 64 to 256 |

    - Decode at 131,072 goes from 61 to 73 tok/s to 84 to 87 tok/s (2 interleaved pairs on shared CPUs).
  - **Bytes identical:** greedy text and teacher-forced scores match the previous build's byte for byte, on the CPU
    and on CUDA (Qwen3.5-4B, and Qwen3.6-35B-A3B on CUDA).
  - **The fix:** those layers get 0 KV heads, the convention granitehybrid and lfm2 already use. The NextN/MTP block
    keeps the attention shape.
  - **Saved state:** a session image or KV snapshot written before this fix holds the old layout.
    - It is refused by name, never misread. A session image is told "an earlier runner wrote it". A KV snapshot gets
      409 `snapshot_kv_layout`.
    - `RUNNER_KV_RECURRENT_ROWS=1` keeps the old layout and resumes them.
  - **Checks:** `tests/test_qwen35_kv_rows.py` covers the four arches:
    - text and scores identical under either layout;
    - the state shrinks by exactly the DeltaNet layers' rows;
    - an old-layout image and snapshot are refused by name, and resume under the switch to the straight run's
      bytes.

- **Fix: CUDA `--cpu-moe auto` over-planned the device's weights, leaving expert banks and prompt-window columns on
  the table (R4.26.18).**
  - **What the plan charged:** every tensor as its own allocation rounded to the driver's 2 MiB, plus a nine-granule
    allowance per expert bank.
  - **What the upload does:** it joins the device tensors into runs and binds one allocation a run, so it rounds once
    a run.
  - **The cost on Qwen3-30B-A3B Q6_K** (25.1 GB) on a 24 GB slice: the weights were planned 1.07 GB over what the
    upload requested, in 6 bindings. Auto placed 43 of 48 expert banks, and the prompt window got 192 columns.
  - **The fix:** after the greedy placement, the plan re-costs the weights by the runs that placement makes, then
    keeps placing the next bank while that exact figure fits.
  - **The result:** 45 banks and a 512-column window. The weights are planned within 16 MB of what the driver spent
    (the constants and tables uploaded beside them), and the instance buffers within 12 MB.
  - **The same run, 3 interleaved pairs:** prefill of a 4,252-token prompt 315 to 541 tok/s, mostly the window.
    Decode 70.6 to 73.4 against 75.3 to 76.7 tok/s.
  - A retry after a failed init keeps the greedy plan.
  - **`RUNNER_DEBUG_VRAM`** prints the plan per category (`vram-plan:` headroom and instance, KV, weights greedy and by
    runs, banks, head). The upload line gives bytes requested and as the driver rounds them.
  - **Checks:** `tests/test_cuda_plan_runs.py` (CUDA): on three MoE fixtures, the plan's weights by runs equal the
    upload's rounded bytes, up to the head's granule.
- **The route line and records say what the tensor cores ran, and `--verify` holds a replay to it (R3.39.2).**
  - **The old label:** with `RUNNER_CUDA_TC` unset, the CUDA route detail read `tc=promoted` for every non-MoE
    architecture. So an unpromoted model's line, and its records, claimed a tensor-core prefill that never ran.
  - **The new label** is `tc=taken` where the default takes the tensor cores for one of the model's weight matrices,
    and `tc=off` where it does not. `tc=forced` and `moe-tc` are unchanged.
  - **Records** list the weight types the route takes on the tensor cores in `profile.route.tc_types`, and `--verify`
    compares them. A later promotion then reads as a route change, not only in the token replay.
  - **Records written before this,** which carry the word `promoted` and no `tc_types`, still verify when nothing else
    in their route differs. The verdict reads "tensor-core types not recorded (a record from before tc_types)".
  - **Checks:**
    - `tests/test_tc_route_label.py` (CUDA): the label agrees with the receipt's tensor-core GEMM counts on four
      fixtures.
    - `tests/test_verify_tc_compat.py` (CUDA): a record written the pre-fix way verifies with the note, and still fails
      when another route token differs; recorded `tc_types` hold the replay; `RUNNER_CUDA_TC=0` fails a `taken`
      record.
    - Genuine records from main 241e9d7 (a llama and a deepseek2 fixture) verify on this binary the same way.

- **CUDA: tensor-core prefill on spark2_5 for Q4_K and Q6_K, measured (R3.39.2).** XHToken's Spark-X2.5-4B Q4_K_M
  prefills 9.48x faster (203 to 1,927 tok/s at 1,635 tokens, 5 of 5 pairs) and reads the same against
  XHToken/Spark-X2.5-4B in float32 on 2,000 positions: mean KLD ratio 0.9995 (CI [0.9981, 1.0007], Wilcoxon p 0.115),
  margin-qualified top-1 95.01% against 94.95%. The promotion covers exactly the two types the file exercised.
  `RUNNER_CUDA_TC=0` still holds the default route.
  - **clef, for Q4_K and Q8_0:** ggml-org's Clef-Flash Q4_K_M prefills 15.4x faster (102 to 1,566 tok/s) and reads
    within the documented obligation (docs/kernel-routes.md: mean KL within 1.01x, Wilcoxon and McNemar not worse at
    p < 0.05) against Cloudflare/clef-flash in float32: mean KLD ratio 1.0001 (CI [0.9991, 1.0011], Wilcoxon p
    0.452), margin-qualified top-1 95.13% against 94.99%. `tests/test_tc_clef_promotion.py` checks the dispatch.
  - **Not promoted, and why:**
    - **k2-horizon and stablelm** run on the CPU on a CUDA build (grouped RMSNorm, LayerNorm), so they have no device
      route to promote.
  - **A CUDA record of a spark2_5 Q4_K or Q6_K model made by an earlier binary** replays on this one through the
    tensor cores, as after any promotion. Its route label reads the same, so the token replay is what tells the two
    apart.
  - **Checks:** `tests/test_tc_spark_promotion.py` (CUDA), on the receipt's tensor-core GEMM counts:
    - a wide spark2_5 fixture's Q4_K takes the tensor cores by default and its Q5_0 rows do not;
    - `RUNNER_CUDA_TC=0` takes none;
    - a narrow fixture with only Q5_0 takes none.

- **Fix: `runner recommend` lists a file that sits in a folder of its repository (R10.28).** #690 left such files out
  on the belief that `-hf REPO:STEM` names only top-level files. It does not: `-hf` matches the base name and fetches
  the listing's full path (`resolve/main/merged/merged-Q8_0.gguf`). The skip came from Dojo's picker, which built the
  URL from the bare name and got a 404; Dojo now fetches by the roster's `path`. The Qwen3-4B tool-use merged Q8_0
  (3.99 GiB) is listed at 8 GiB again.
  - **Checks:** `tests/test_hf_fetch.py` fetches a folder file from a fake hub shaped like that repository: the
    URL keeps the folder, and the short tag that also matches a `study/` adapter is refused with both named.
    `tests/test_recommend.py`'s anchor is now the 16 files Dojo's fixed picker listed on an 8 GiB machine.

- **`runner recommend` (`--info recommend`): the listed model files that fit this machine (R10.25).** Largest resident
  weights first, each with the `-hf` spelling to run it, its fidelity line (`measured` with the first sentence of its
  numerical anchor, or `unmeasured` with why) and its entry in docs/model-support.md. It makes no network request:
  the list comes from the roster built into the binary (`src/model_roster.h`, rendered from docs/model-tiers.json by
  `scripts/model-tiers.py --write-docs` and held fresh by `--docs-check`). `--roster FILE` reads another copy,
  `--memory GIB` plans for another machine and `--ram-share F` (default 0.5) moves the bar.
  - The rule is the Dojo model picker's: a listed text model in one `.gguf`, no larger than the RAM, whose
    resident weights fit within the share. Resident means the whole file, or the `--info fit` hot set for a file
    whose per-layer token table is read by row. That hot set is the new `files[].resident_bytes`, recorded from
    `runner --fit` by `scripts/model-tiers.py --record-resident` (Nemotron-3.5-Bolt-E4B and the gemma-4-E2B mix).
  - **Checks:** `tests/test_recommend.py`: on a fixture roster a memory cap and the share change the list, and the
    excluded shapes stay out. Absolute anchor: at 8 GiB the committed roster gives the files Dojo's own run listed (16 since R10.28)
    on an 8 GiB machine. `tests/test_model_tiers.py` covers the recorder, the field's validation and the snapshot.
- **`runner train --full`: full-weight training of an F32 file into a variant (R8.25.2.2).** Every dense projection
  (attn q/k/v/o, ffn gate/up/down) gets an f32 master with AdamW state, in place of an adapter. `--out` is a variant:
  the -m file with the trained tensors replaced, every other byte unchanged. The parent stays untouched. The same data
  and seed give a byte-identical variant, and the training record names the base's and the data's sha256, the trained
  tensors (`full_weight`) and the `variant`.
  - **Training through a Q8_0 copy was tried and dropped.** The design read a Q8_0 copy of each master in the forward
    and the backward, so the shipped Q8_0 would be what the last step read. On Qwen3-0.6B, the 10-28 tool-use recipe,
    316 steps, both routes scored 29/29 held-out exact calls, with 0 discordant prompts:
    - trained through Q8_0;
    - trained in float, then `--rewrite --quant q8_0`.
  - **What that does and does not show.** The held-out set saturated, so this is no measurable gain at Q8 on this
    task, not a proof that the two are equal.
    - Beside the gate: the teacher-forced NLL was 0.00003 against 0.00006 nats per token. The KLD to each file's own
      float model was 0.00419 against 0.00425.
    - A quantized -m file is refused: `is not f32: full-weight training reads an F32 file (train the float file, then
      quantize the variant with --rewrite --quant)`.
  - **`--kv-f32` (`--train-kv-f32`): an f32 KV cache for a training run, on the CPU only.** On the default f16 cache,
    the full-weight gradient matched PyTorch fp32 autograd only to 2.7e-3 relative L2, against a 1e-3 bar, because K
    (after RoPE) and V are rounded to f16. With an f32 cache it matches to 6.9e-6 over all 196 matrices of Qwen3-0.6B,
    and 60 steps' losses agree with PyTorch's. Inference keeps f16, and the record writes `"kv_cache":"f32"`.
  - **The gradient check on the fixture:**
    - **On the f16 cache:** directional derivatives at a fixed eps of 1e-3 missed the first declared 1e-3 (1.17e-3
      and 4.73e-3), because the cache's rounding makes the finite difference a staircase.
      - `tests/test_full_grad.c` grades the f16 path by tests/test_lora_grad.c's method instead: along sign(G), the
        best of six steps within 1%.
      - On the f32 cache it grades the fixed-eps derivatives as first declared.
    - **Also pinned there:**
      - the gradient is byte-identical across two backwards;
      - the forward reads the master after a step;
      - a Q8_0 file is refused.
  - **Not covered yet:** `--train-gpu`, DPO, a teacher, checkpoints (`--save-every`, `--train-resume`), kill rules,
    `--train-eval`, a stop file, `--train-logprobs`, recurrent (DeltaNet) layers and MoE expert banks are refused with
    `--full`.
  - **Checks:** `tests/test_full_grad.c` (in `make test-gates-core`) and `tests/test_full_train.py`.

- **Speculative decoding defaults per workload (R3.39.4).** With no `--draft`, two measured cases now draft without a
  flag, both only on CUDA with every layer on the device. The lookup and the NextN head emit plain decoding's tokens,
  so only speed changes.
  - A server request with `draft_hints` takes `lookup` at width 7, measured at 2.86x plain on an in-context edit.
    Such a request was refused before unless the server ran with `--draft lookup`. The lookup is that request's
    alone, and `GET /v1/capabilities` reports it as `draft.default`.
  - A one-shot CLI run with `--json` or `--json-schema` on Qwen3.5-4B Q4_K_M with its NextN block takes `mtp` at
    width 2, measured at 1.13x plain on a tool call. If binding the head would cost the run its full offload,
    Runner reloads without it.
  - Everything else decodes plain: prose (the head reads 0.88x there, and the lookup has no prose row), chat,
    `tool_choice: "auto"`, other NextN models, CPU and Metal. The server takes no per-request head: a head fed on
    every request would slow prose, and one fed on demand would have no state under a reused prefix.
  - `--draft off` turns the defaults off. The startup line names a default and its scope, and the route line marks
    it `draft=lookup:default` or `draft=mtp:default` (a named source shows as `draft=lookup`).
  - `docs/speed-defaults.md` lists every speed switch with its class and its default (R3.39.1).
  - **Checks:** `tests/test_lookup_draft.c` pins the decision table against the measured rows, prose plain on both
    surfaces among them. `tests/test_draft_defaults.py` checks that a request without the signal decodes plain on
    the CLI and the server, before and after a hinted request; that `--draft off` refuses hints and a second source;
    and that a `--json` record replays under `--verify`. Under the test-only `RUNNER_TEST_DRAFT_SCOPE=1` both
    defaults fire on fixtures, and their records (a CLI head run, a served hinted request after a plain one on the
    same slot) verify with the switch unset.
  - **A record replays the draft source it ran.** It names the source, with `"default": true` when the default
    chose it, and `--verify` binds that source instead of decoding plain. The tokens were always the same, but on
    CUDA the head's walk takes no decode graph, so a plain replay of a head record failed the route check (found on
    the CUDA gate with the measured file). A record whose route cannot draft from its source replays plain, with a
    note. A `model` record still needs `--draft PATH`.
- **Grammar fast-forward is identity-class and remains opt-in (R3.39.5).**
  Its constrained target verification preserves token IDs and bytes. A bounded
  SmolLM2-135M Q8_0 check found a CPU gain on one schema and large regressions
  on another, so there is no default promotion. The grammar gate now initializes
  the fp16 lookup table, checks it against IEEE 754 binary16 1.0, and compares
  token IDs and sampler RNG as well as output bytes on greedy and seeded
  constrained runs. Unconstrained prose stays on the plain path.
- **Fix: LoRA training with `--kv k8v4` or `--kv fp4` is refused, where it failed at step 1.** The training backward reads
  K and V rows as f16. `--kv q8` was refused, but k8v4 and fp4 reached the backward, and their blocks were read as f16.
  On Qwen3-0.6B both runs stopped at step 1 with "the loss or a gradient is not finite", caught only because the
  garbage was not finite. Both are now refused by name, as q8 is: `training does not cover this model yet: k8v4 KV
  cache (use --kv f16)`, and likewise for fp4.
  - The layer backward's attention recompute now passes the model's cache kinds instead of a literal f16. These are f16
    in every case training admits, so the bytes are unchanged.
  - **Checks:** `tests/test_train_kv_refusal.py`, on a wide-head fixture where each cache type is really taken.

- **`runner derive --target-size` with `--imatrix` plans a whole file (R4.34.2).** llama.cpp's imatrix has no entry for
  `output` or `token_embd` unless it was run with `--process-output`, so those two tensors' errors were unweighted
  beside the weighted rest, and the planner refused the mix. A candidate the matrix does not cover is now held at the
  top rung and left out of the ranking, and the stderr, the record (`held`, `held_at`) and the card name it.
  - Holding them is a default, not evidence that they matter most.
  - `scripts/type-plan-by-sensitivity.py` holds the same tensors, and `--allow-mixed` still ranks them together.
  - **Checks:** `tests/test_derive.py`: an imatrix skewed onto the tensor the plain plan demotes first keeps it up
    and changes the plan; an imatrix that leaves one tensor out gives a plan with that tensor held and named, equal
    to the reference planner's.
- **Test fix: the FP8 gold-logits test skips without `accelerate`.** transformers
  loads an FP8 checkpoint only with `accelerate` installed, and the test guarded
  only `torch` and `transformers`, so a machine with transformers but without
  accelerate failed it with an ImportError instead of skipping, as the
  neighbouring tests already do.

- **`runner derive --target-size` gains a middle rung: `--mid T`, or `--ladder T1,...,Tn` high to low (R4.34.1).** A
  group is demoted one rung at a time instead of straight to the lowest type. Over the ladder, each group's (bytes,
  error) points are reduced to their lower convex hull, and the cheapest next step over all groups (added error per
  byte saved) is taken until the file fits.
  - A rung that costs more per byte than going past it is skipped for that group. The record names each skip, and
    names where each demoted group ended (`ladder`, `demoted_to`, `skipped`).
  - Two rungs (`--high`, `--low`) plan exactly as before.
  - `scripts/type-plan-by-sensitivity.py` takes the same `--ladder` and `--mid`, and the test holds the two to the
    same plan.

- **Fix: Metal tiled receipts identify actual arithmetic and refuse unsupported replay (R3.39.3a).**
  A successful tiled generation step marks `profile.route.micro` as `tiles`;
  the configured switch alone no longer marks an identity-only receipt.
  `--verify` refuses a tiled record without a replayable decode schedule as
  `UNVERIFIABLE` before loading the model. The default is unchanged: tiles
  remain opt-in, and 2-5 columns keep the identity route.

- **Importance-weighted Q4_K, Q5_K, Q6_K, Q5_1 and IQ4_NL writers (R4.34.3).**
  `--imatrix` now shapes these formats as well as Q3_K and Q4_0, using ggml's
  type-specific weighting and scale searches. All five are byte twins of
  ggml `c1d0e7a` on the committed real tensor slice and deterministic boundary
  rows. The `.dat` and GGUF calibration paths agree, all-zero calibration
  retains the reference bytes, and mismatched calibration shapes are refused.
  The no-imatrix bytes and planner policy are unchanged.

- **Docs: the bartowski Qwen3.5-9B distill row no longer says the `qwen35` path
  has no Metal backend.** It has run on Metal since v1.10.0 when the model fits
  on the device whole; this 5.84 GB file does not on an 8 GB Mac, which is why
  the row has no Metal cell.

## v1.15.0 - 2026-11-07

- **Fix: transient VRAM accounting no longer forces durable writes (R6.24).**
  The ledger and yield sentinels describe live processes, so power-loss
  durability adds no useful guarantee. POSIX `fsync` and Windows
  `FlushFileBuffers` no longer stall every ledger user under the shared lock.
  Locks and the complete-write-before-truncate order are unchanged; readers
  retain valid rows, ignore malformed fragments and reap stale owners.
- **The DeepSeek-V2-Lite-Chat Q8_0 row's pins sit on a near tie, and v1.14.0's R3.32 note is corrected (R6.23).**
  Docs only; the pins are unchanged.
  - **Why the pins mismatched:** an overnight compare against the pins mismatched on both backends, but its builds
    were clang's. The pins come from the Makefile's default compiler, gcc with `-march=native`, and main's gcc build
    matches both.
  - **The tie:** every build is deterministic. The reply is decided at prompt position 6, layer 7, where the sixth
    routed expert is 39 or 49 with gates within 2e-6. The two texts that can result both contain "Mercury".
  - **The release build:** gcc with `-march=x86-64-v3` gives the CUDA pin's text on both backends, so its CPU reply
    differs from the CPU pin. The row's note in `docs/model-tiers.json` says so.
  - **The v1.14.0 correction:** the entry for R3.32 said both deepseek2 CUDA cells were stale from #631. Only the
    Q6_K's was. R3.32's prefill route moved the Q8_0's and re-pinned it in the same change.

- **The model tiers are retired: `docs/model-support.md` is now one list.** Families, models and files sort
  alphabetically, and opening a family shows one table, with each model's name on a row above its files. Validation runs
  when the project chooses, not with every release, so `scripts/check-release.py` no longer refuses a tag on a row's
  record. Upcoming stays apart: it means not supported yet. The transition log keeps the earlier record.

- **Known `chat_template_kwargs` keys no longer vanish on the families that ignored them (R2.14).** Measured against 33
  references. Honoured now: `preserve_thinking` on LFM2 and LFM2-24B, `preserve_thinking: false` on Qwen 3.8 (its default
  keeps every thought), `enable_thinking` on Apertus, and `reasoning_effort: low` on Granite 4.2. Refused now, with the
  400 R2.13 added, wherever the file's template reads the key: a key a family's renderer ignores, `preserve_thinking` on
  Responses and Messages, `reasoning_effort` and `reasoning_strength` on Messages, and `add_generation_prompt: false`.
  Spark refuses `reasoning_effort: xhigh`, as its template does. Limits: 19 families were not compared with their
  references on this host, and are covered by the refusal rather than a measurement. A null `preserve_thinking` counts
  as absent, where Qwen 3.8's template would treat it as false.

- **A `chat_template_kwargs` switch Runner cannot render is refused, not silently dropped (R2.13).** Sage-A1.5's
  `verification` rendered as off with no error, because Runner renders each family with its own code and reads only
  the keys it knows. A key the file's own template reads as a variable, and Runner does not render, now gets a 400 naming
  it (`unsupported_template_kwarg`). A key the template does not read is ignored, as the reference ignores it. Limits:
  the check is by name, so `false` is refused too, and it is a scan of the tags rather than a Jinja parse, which counts
  anything doubtful as not read. `enable_thinking`, `preserve_thinking`, `reasoning_effort`, `reasoning_strength`,
  `bare_recipients` and the values a render supplies itself are unaffected.
- **A usage error ends with the error (R10.19).** An unknown option, a flag with no value, no model and no prompt each
  printed the whole usage, 382 lines, so a script or a person reading the end of the output never saw why: one export
  script read `unknown option --out` as a refusal of the quantization. They now print the error line and
  `see runner --help`, and the full usage is for `--help`. A bare `runner`, with no arguments or only `--no-tray`, still
  prints the usage, for someone running it for the first time, and now ends with
  `error: need -m MODEL or -hf REPO[:TAG]`. Exit codes are unchanged.
  - **Checks:** the whole of stderr for each case, in `tests/test_cli_surface.py`.
  - **Limits:** `runner attest` and `runner train` with nothing after them still print their own short usage. A script
    that looked for `usage:` on stderr after a failure other than a bare run no longer finds it; `unknown option` now reads
    `error: unknown option`.
- **Sage-A1.5 joins the roster on InternScience's own Q6_K, and occamy-1.0 is listed Upcoming (R4.27.7).**
  Both are qwen35moe derivatives of Qwen3.6-35B-A3B, each read against its own BF16 parent computed in float32 (bars
  0.05 mean KLD and 97% margin-qualified top-1, 300 positions, CPU).
  - **Sage-A1.5:** InternScience's model, built by its publisher on Qwen3.6-35B-A3B and InternScience's Agents-A1.
    - Q6_K: 0.0395 mean KLD and 97.9% margin-qualified, passes. The Q4_K_M misses (0.145).
    - Tokenizer 0 of 721 and 0 of 247,091 against its tokenizer.json (the qwen35 split).
    - Template conformance 25 of 25 in text and tokens.
    - CPU and CUDA cells pinned and matched; cpu_cuda 9 of 9 exact with routing pinned.
    - Limits: the template's `verification` switch is not served (a request with it does not get the trigger), and the
      GGUF names itself "Agents A1.5". Record: `docs/compat-reports/sage-a1.5-2026-11-06/`.
  - **occamy-1.0:** Accio-Lab's finetune. No publisher file passes: Q8_0 0.0590 / 98.7%, Q6_K 0.0836 / 95.4%, Q5_K_M
    0.0970 / 95.8%, Q4_K_M 0.258 / 85.7%. Tokenizer (the qwen2 split its header labels) and template (25 of 25) pass.
    Where the distance comes from is not yet read. Record: `docs/compat-reports/occamy-1.0-2026-11-06/`.
- **Fix: instance cleanup no longer delays server startup (R10.23).**
  Registration and discovery lists request a separate cleanup process instead
  of deleting stale records on the caller's path. One helper sweeps a folder
  at a time; it can finish after the server exits. Dead records are filtered
  immediately and eventually removed as storage progresses, including on
  hosts without a tray. Cleanup launch failures never fail a server start.

- **Windows: a second server on a port in use is refused, not silently sharing it (R10.22).** `SO_REUSEADDR` let both
  bind on Windows. The second printed `server listening` and got no request, and when the first stopped, its clients'
  connections were reset. It now exits with `cannot bind 127.0.0.1:PORT: is another server on this port?`, as Linux
  already refused. This was the cause of a rare `WinError 10054` under `pytest -n`, when two workers drew the same port;
  the test harness now takes another port on a lost bind. Linux and macOS keep `SO_REUSEADDR`, and a restart straight
  after serving still binds. Measured on Windows.

- **Fix: the project's Clef IQ3_S backbone tokenizes as Cloudflare/clef's own tokenizer, and its row no longer
  claims the decision model (R4.36).** `Clef-backbone-IQ3_S.gguf` is Clef's backbone in Runner's qwen35 view, so it
  declares architecture qwen35 and the Clef rule (architecture clef) missed it: it read with the qwen35 split, 3 of 721
  corpus strings and 5,503 of 247,091 vocabulary entries off the publisher's. A rule entry keyed on its header
  (qwen35 and general.name "Clef Backbone View") reads it with the qwen2 split; 0 and 0 now. Its row,
  `cloudflare-clef-iq3_s-recovered`, claimed "a decision model served at /v1/systemone", which is the
  `cloudflare-clef-iq3_s-native` row's file: it now says the text-only backbone.
  - **Checks:** every other roster file's ids unchanged (74 rows), the backbone's alone different, on exactly the
    three combining-mark strings; the house readings' corpus tokenizes identically before and after, so its numbers
    (0.0879 / 94.0%, 0.0736 / 95.6%, 0.0757 / 95.6%) carry over; the recovery adapters were trained on the qwen2
    split's ids, which the backbone now serves.
  - Record: `docs/compat-reports/r4.36-clef-backbone-2026-11-06/`.
- **CUDA: `RUNNER_MOE_HOT` gives back the host copies of the experts it holds on the device (R3.35).** A server with
  Mellum 2.1 Q8_0's 768-slot hot file held 5.05 GB of pages the host path never reads. After warm decode on a 16 GB
  host, its working set reads 5.39 GB against 10.44 GB, and available memory 5.85 GB against 0.78 GB. Measured on one
  file on one Windows host. This is a RAM change, not a speed claim: warm decode is unchanged. Load still peaks at
  6.14 GB, and prints how many experts it gave back. Report:
  `docs/compat-reports/r3.35-hot-host-copies-2026-11-06`.
- **A client that disconnects stops its queued model load (R10.12.3.1).** A load waiting in the `--wait-for-vram` queue
  went on waiting after its client had gone, up to the 300 s default, holding the one slot of a swap server. The wait
  now stops within about 100 ms and the server carries on. It stops only if nobody is queued behind that client: a
  live client keeps the wait and its place in the VRAM queue. A client that goes after the load is admitted costs
  nothing, and the finished load stays resident for the next request. A wait that `/unload` or a shutdown stopped can
  also no longer be admitted on the same poll and load a model only to discard it.
  - **Checks:** `tests/test_queued_load_disconnect.py` runs the real wait on any host; the stop-against-freed-memory
    race is in `tests/test_vram_registry.c`.
  - **Limits:** CUDA only: CPU and Metal have no queue to wait in. A queued client that sent its request and then left
    counts as present, because its close cannot be seen behind the unread request, so the wait goes on as it did. A
    load past the queue is not interrupted. A swap that stops here has already unloaded the previous model.
- **Metal: a microbatch no longer frees the logits a slot is about to read (R3.38).** With `--parallel` on Metal, the
  first slot's scratch grew on demand when other slots decoded together. If that slot had just been fed a single token,
  as after a prefix hit, the growth freed the buffer it was about to pick its next token from, and the read could crash
  the server or pick from other data. On SmolLM2-135M the forced sequence segfaulted. The scratch is now sized once,
  when the batch is created, as the CUDA backend already does.
  - **Checks:** `tests/test_batch_identity.c` forces the sequence and holds the pointer and its contents.
  - **Limits:** the first slot's scratch for the batch width is allocated at startup instead of on first use. The live
    server interleaving was not reproduced end to end; the sequence was forced through the model API.

- **Metal gives its memory back on `/unload` (R6.17).** On Metal, each model load left memory in the GPU driver until the
  process exited: `POST /unload` said the model was gone, and on SmolLM2-135M Q8_0 the driver still held +122, +280 and
  +278 MB after three request and `/unload` cycles. It now holds +3, +2 and -5 MB. Metal's autoreleased objects were
  never drained on Runner's plain C threads, and a command buffer keeps what it used alive; every Objective-C entry point
  of the Metal backend now drains its own autorelease pool. RSS and Metal's own count never showed this memory, which
  is wired, so on an 8 GB Mac it pushed other programs into swap: one test binary took 3.8 GB on a 370 KB model, and
  now peaks at 729 MB.
  - **Checks:** `tests/test_metal_memory.py` reads the driver's own figure across four `/unload` cycles, and holds every
    entry point to its pool. Decode 128.71 against 128.50 tok/s (-0.16%), output byte-identical over 12 runs.
  - **Limits:** the weights, KV and scratch were already released; what stayed was about 120 MB a load, and it did not
    scale with the model on the two sizes measured (370 KB and 145 MB). Measured on Apple M1 only. CUDA and the CPU are
    untouched.
- **Response ids are random, and a stored response can no longer be guessed (R10.15).** `chatcmpl-`, `cmpl-`, `resp_`
  and `msg_` ids were a counter from 0: they repeated after every restart and across hosts, and anyone who could
  reach the server could read or delete a `store:true` response at `/v1/responses/resp_0`. An id is now its prefix and
  128 random bits, 32 hex digits. The log lines and receipts carry the same id. Tool-call and output-item ids
  (`call_0`, `fc_0`) keep their numbers: they name a place in one response and look nothing up.
  - **Checks:** the four APIs, a stream, a restart, and a counted id reading nothing, in
    `tests/test_responses_store.py`.
  - **Limits:** the id is still the only credential on a stored response. A client that parsed the number out of an id
    no longer can. If the OS has no random generator the server now refuses to start.
- **Fix: instance discovery no longer removes records at process exit (R10.20).**
  Filesystem removal could block after the server's shutdown watchdog had
  finished. Clean stops now leave their record for the next discovery list
  or registration to sweep. Dead owners are never listed; available process-start
  stamps guard against pid reuse. Runner and tray exits use the same policy.
- **A stop that has to be forced now lands well inside its ten seconds (R10.20).** When a drain cannot finish, the
  watchdog exited 50 ms before the bound, and a shared macOS runner ended 2 stops of 8 past it (10.007 s at worst):
  timers there run up to 90 ms late. A late timer could also lose the shutdown note, the one line that says what a
  forced stop was waiting on. The note now starts at 9 s, the exit is due at 9.5 s, and a note that starts late still
  gets its turn.
  - **Checks:** the default-budget case in `tests/conformance/test_server_lifecycle.py` now requires 9.9 s, so a margin
    this thin fails on any machine. A new case wakes the watchdog 155 ms late and requires the note.
  - **Limits:** a stuck request gets 0.5 s less before it is cut off. A clean stop is unchanged.

- **A context released while its request waited for the GPU no longer hangs the server (R3.37).** The
  `context_unavailable` 409 returned without giving the device turn back, so every later completion hung while
  `/health` still read `ok`. It took a pinned context deleted during that wait, on a GPU server with more than one
  slot. The branch now gives the turn back before it writes. Needs a CUDA device to test, so the before and after are a
  recorded RTX 3070 run, not part of `make test`. Report:
  `docs/compat-reports/r3.37-context-turn-leak-2026-11-06`.

- **A client that stops reading no longer stops the other slots (R3.36).** A speculative stream wrote to the client
  while holding the device turn, so one stalled client froze every other slot for 30 s, and a client trickling reads
  held it for as long as it liked (over 150 s measured). No blocking client write now happens under the turn, on any
  path. Limits first:
  - **New limit:** a speculative stream queues at most 4 MiB. A client that far behind is cut off, gets what was
    queued and then its surface's terminator naming `client_too_slow`, and the server logs it. Plain streams have no
    such limit.
  - A reading client's speculative generation still holds the turn for its length. That is unchanged.
  - Measured on Windows on an RTX 3070 and on macOS on an Apple M1. Linux is covered by a test on a full socket, not
    by an end-to-end read.

  A session's answer is now built under the turn and written after it, as embeddings already were. Windows did not
  stall on that write, because Winsock takes one large send whole, but it broke the rule. `make runner-turn-guard`
  builds the rule in: a client write under the turn ends the process, and the turn runs on a CPU server, so
  `tests/test_turn_write_guard.py` holds any future path to it without a GPU. Report:
  `docs/compat-reports/r3.36-no-write-under-turn-2026-11-06`.

- **MiniCPM-V-4.7-35B-A3B joins the roster on openbmb's own Q8_0, text path only (R4.27.2).** The file is
  the GGUF repository's root Q8_0 (thinking fixed off, openbmb's recommended default). Against openbmb's BF16 parent's
  language model computed in float32 it reads 0.0088 mean KLD and 100.00% margin-qualified top-1 (bars 0.05 and 97%,
  300 positions, CPU).
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 247,091 vocabulary entries against openbmb's tokenizer.json, read
      with the qwen2 split by R4.27.5's rule.
    - Template conformance under `minicpmv-fixed` against the file's own template (R4.27.6); the reasoning-history
      case accounted as a token-only deviation (R2.4.11).
    - CPU and CUDA cells pinned and matched (byte-identical replies); gpu-identity ok with routing pinned; cpu_cuda 9
      of 9 exact.
  - **Limits:**
    - openbmb's Q4_K_M (0.0594, 95.44%) misses the bar.
    - The reference is transformers' Qwen3.5-MoE text model on the checkpoint's language-model weights, as
      transformers has no class for minicpmv4_7 and the repository ships no modeling code.
    - Text only; thinking is fixed off in this file; on a 24 GB CUDA card it runs split, 26 of 40 blocks.
    - Record: `docs/compat-reports/minicpm-v-4.7-35b-a3b-2026-11-06/`.

- **MiniCPM-V 4.7's chat templates get their own renderers, text path (R4.27.6).** openbmb's GGUF repository ships
  two sets with byte-identical weights that differ only in the embedded template: the root files (thinking fixed off,
  openbmb's recommended default) and `thinking/` (the repository's chat_template.jinja). Runner read them as `qwen35`
  and `qwen35-nothink`. They now read as `minicpmv-fixed` and `minicpmv`: Qwen 3.5-0.8B's template with a thought
  block on every assistant turn (empty unless the turn carries reasoning after the last user query), and for the root
  files a generation prompt that is always closed. Content parts are joined with '\n', as Runner already joined them.
  A `minicpmv` sampling preset is openbmb's README's recommended instruct-mode setting, as published: temperature
  0.7, top_p 0.95, top_k 100, min_p 0, repetition_penalty 1.03.
  - **Checks:**
    - template-conformance `--require-tokens` against each file's own template: 56 cases conformant in text; the two
      reasoning-history cases differ only by caller-spelled `<think>` tags Runner reads as text, accounted under the
      token-only kind (R2.4.11).
    - Every other family's renders, and the template detected for every roster file, unchanged.
  - **Limits:** text path only (image and video placeholders are not rendered); no MiniCPM-V row is admitted here
    (R4.27.2 continues on this). Record: `docs/compat-reports/r4.27.6-minicpmv-template-2026-11-06/`.

- **template-conformance: a token-only deviation kind (R2.4.11).** A case can render byte-identically on both sides
  yet tokenize differently on purpose: Runner reads a control token only in the template's own bytes, so one the
  caller spelled in its content is text (tests/test_prompt_marks.c), where the reference tokenizer reads it as the
  control token. Until now the gate could only fail such a case as a broken premise. An allowlist entry of kind
  `token-only` names the case, cites tests/test_prompt_marks.c, and is accepted only when the text is byte-identical
  with no patch applied and every differing reference id is a control token the caller spelled; otherwise it is
  refused and the case still fails. `tests/test_template_conformance_token_only.py` holds the rules; with the kind
  added and no entries, every family's results are unchanged.

- **Fix: the qwen35-family templates trim message content as their references do (R2.4.10).** Qwen 3.5's, Qwen 3.6's
  and ornith's chat_template.jinja trim every message's content and its reasoning_content (`|trim`, python's
  str.strip(), Unicode whitespace included) before they frame it; Runner trimmed none of it, so a padded system, user,
  assistant or tool turn reached the model with whitespace its reference never writes. Each surface
  (chat/completions, /v1/responses, /v1/messages) now trims where it flattens a message, for the ornith, qwen35,
  qwen35-nothink and qwen36 templates only.
  - **Checks:**
    - Two padded house cases (padded-content, padded-tool-turn) on the eight qwen35-family conformance families:
      identical to each reference in text and tokens. A mutant without the trim fails exactly those cases.
    - Every other case of every registered family renders byte-identically before and after (50 families, 1,057
      renders); the template detected for every roster file is unchanged.
    - test_tool_attribution: a padded conversation renders as the plain one on all three surfaces; ChatML keeps it.
  - Record: `docs/compat-reports/r2.4.10-qwen35-trim-2026-11-06/`.

- **Qwen3.8-35B-A3B-Distill, empero-ai's finetune of Qwen3.6-35B-A3B, joins the roster on empero-ai's own
  Q6_K, text path only (R4.27.1).** It is not a Qwen release; the row replaces the Upcoming `qwen3.8-35b-a3b`, which
  named a Qwen model that Qwen has not published. Against empero-ai's BF16 parent computed in float32 the Q6_K reads
  0.0336 mean KLD and 98.65% margin-qualified top-1 (bars 0.05 and 97%, 300 positions, CPU).
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 247,091 vocabulary entries against empero-ai's tokenizer.json, read
      with the qwen2 split by R4.27.5's rule.
    - Template conformance 23 of 23 in text and tokens under `qwen36`, against the file's own template
      (Qwen3.6-35B-A3B's bytes; a new `empero-qwen38-35b-a3b-distill` family).
    - CPU and CUDA cells pinned and matched; gpu-identity ok with routing pinned; cpu_cuda pass_margin_qualified.
  - **Limits:**
    - empero-ai's Q4_K_M (0.1278) and Q5_K_M (0.0508) miss the bar; the Q4_K_M reads 2.7x the base model's Q4_K_M by
      the same method.
    - The tokenizer rule is keyed on header fields; a requant that rewrites them reads the qwen35 split.
    - On a 24 GB CUDA card the file runs split, 34 of 40 blocks on the device.
    - Record: `docs/compat-reports/empero-qwen3.8-35b-a3b-distill-2026-11-06/`.

- **Fix: two qwen35-labelled finetunes trained with the qwen2 split now tokenize as their publishers do
  (R4.27.5).** openbmb's MiniCPM-V-4.7-35B-A3B and empero-ai's Qwen3.8-35B-A3B-Distill ship GGUFs labelled qwen35,
  while the tokenizer.json each publisher saved under transformers 5.x uses the qwen2 regex (`\p{L}+`, combining
  marks split from their letters). llama.cpp's converter registers both splits' hashes as "qwen35", so no published
  label can carry the difference. Runner now keys a rule on the header fields the publishers' converters wrote, and
  a firing rule prints a note naming the rule and the fields it matched.
  - **Checks:**
    - Against each publisher's tokenizer.json: 0 of 721 corpus strings and 0 of 247,091 vocabulary entries
      (before: 3 and 5,503).
    - Every other roster file (74 rows): token ids byte-identical to the previous build on the corpus and the
      tier-smoke prompts.
    - The qwen35 family's 17 linux-x86_64-cuda pins all match, the base's two among them: the 9 CPU pins on the
      change before the 1.14.0 merge, the 8 CUDA pins after it. Ornith's CUDA pin matched on a rerun; its first
      run ended in a harness-reported error after a complete reply, under heavy outside load.
  - **Limits:**
    - The rule is a list of four sources (Clef and XiaomiMiMo's Qwen3.5 distill, already read as qwen2 and now in
      the same list, plus the two new ones). A requant that rewrites the matched fields falls outside it.
    - Clef and MiMo files now print the note too; their ids are unchanged.
    - cloudflare-clef-iq3_s-recovered's file declares architecture qwen35 and still reads with the qwen35 split.
  - Record: `docs/compat-reports/r4.27.5-qwen2-trained-rule-2026-11-06/`.

- **The project's own Mellum 2.1 rewrite joins the roster on the published file, text path only
  (R4.33.4).** `mellum2.1-12b-a2.5b-thinking-selective-q8_0-expq5_k` is Runner's `--rewrite --type-plan` of JetBrains'
  BF16 parent: Q8_0, with the experts at Q5_K and the 896-wide expert down projections at Q5_1. It is pinned by the
  Hub revision `4941f110` and its sha256. Against the parent in float32 it reads 0.0293 mean KLD and 98.74%
  margin-qualified top-1 (R4.33's record, the same bytes, no new pass).
  - **What is not flattering:** it is 31.4 MB larger than bartowski's Q5_K_S for 0.0004 more mean KLD. It was written
    without an importance matrix. Its first window reads 0.0419. The header's `general.name` is a conversion
    artifact ("Ckpt Mem"), kept so the bytes stay as published.
  - **Checks:** tokenizer 0 of 721 corpus strings and 0 of 97,302 vocabulary entries; template conformance 23 of 23
    (the file's template is JetBrains' byte for byte). The CPU and CUDA tier-smoke replies are byte-identical.
  - **The Upcoming rows:** Nemotron 3.5 Bolt 74B-A9B's and Super VL 120B-A12B's now name why they are not admitted.
    Runner refuses both files at load. Both use LatentMoE (expert projections behind a 1,024-wide latent), which
    Runner does not implement. Super VL declares 512 experts, past Runner's cap of 256. Bolt declares its expert
    counts and widths per layer, and Runner reads them as scalars (R4.27.4.2, R4.27.4.3).

## v1.14.0 - 2026-11-06

- **Fix: `-v` crashed on any file whose first block is Mamba-2; found on Nemotron 3 Nano.** The verbose summary's
  "weight type" line read block 0's fused qkv (if recurrent) or its `wq`, and a Mamba-2 block has neither (its
  matrices are `ssm_in` and `ssm_out`). The line now takes the block's first matrix that exists, `ssm_in` included.
  On a file whose first block is attention or a fused-qkv recurrent block, the report is byte for byte as before.
  `tests/test_hybrid_admission.py` runs `-v` on the Mamba-first nemotron_h fixture.

- **NVIDIA Nemotron 3 Nano 30B-A3B joins the roster at Tier 2 on ggml-org's Q4_K_M, CPU only (R4.27.4.1).**
  NVIDIA publishes no GGUF, so the file is ggml-org's automatic conversion. Against NVIDIA's BF16 parent computed in
  float32, at the model's declared dt limit, it reads 0.0222 mean KLD and 99.50% margin-qualified top-1 (bars 0.05
  and 97%, 300 positions). With the reference's configured clamp it reads 0.137, as Nemotron 3.5 Lightning did.
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 128,635 vocabulary entries.
    - Template conformance 24 of 24 in text and tokens under `nemotron35` (a new `nemotron3-nano` family).
    - llama.cpp on the same file reads 0.0243, and Runner is the closer engine.
  - **Limits:**
    - CUDA refuses the file: its sigmoid-plus-bias router has no device kernel.
    - The tool gate's case B fails, and case C fails buffered.
    - Record: `docs/compat-reports/nemotron-3-nano-30b-a3b-2026-11-05/`.
- **Ornith-1.5-35B-A3B joins the roster at Tier 2 on ornith-ai's own Q8_0, text path only (R4.27.3).** It passes the
  fidelity bar by a hair against ornith-ai's BF16 parent computed in float32: 0.0484 mean KLD and 97.50%
  margin-qualified top-1 (bars 0.05 and 97%, 300 positions, CPU).
  - **What is not flattering:**
    - Its median KLD, 0.0154, is 3.7x Qwen3.6-35B-A3B's Q8_0 on the same base.
    - ornith-ai's own BF16 GGUF already sits 0.026 from the parent, where a clean conversion sits near zero, and the
      file names no source revision. So about half the distance is in the publisher's conversion or the weights it
      came from.
    - The publisher's Q6_K (0.103) and Q4_K_M (0.171) miss. llama.cpp reads the same error spikes in them, and
      Runner is the closer engine. The 64 tensors the two K-quant files share at Q6_K are named in the row.
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 247,091 vocabulary entries.
    - Template conformance 24 of 24 in text and tokens against the file's own template, under the `ornith` renderer
      (a new `ornith15-35b-a3b` conformance family). A system or developer turn after history has started is
      refused, where the file's template drops it silently.
    - CPU and CUDA tier-smoke replies are byte-identical. CUDA runs the 37.8 GB file split, 26 of 40 blocks on a
      24 GB MIG slice. gpu-identity with routing pinned: 4.8e-6 of the range.
    - Record: `docs/compat-reports/ornith-1.5-35b-a3b-2026-11-05/`.
  - **The Upcoming rows:** MiniCPM-V-4.7-35B-A3B's now names why it is not admitted. The publisher's GGUF labels its
    pre-tokenizer `qwen35` while openbmb's tokenizer.json uses the Qwen2-style regex, so Runner differs from the
    publisher on combining marks (3 of 721 corpus strings), and its chat template is its own.

- **deepseek2's absorbed prefill on CUDA runs on the tensor cores and passes the naive form's (R3.32).** The absorbed
  form's 576-wide row was past every tiled prefill kernel, so its prompt tiles ran `k_attn`: one block per head and
  column, with every cached row read again for each of the 16 heads. `k_attn_tcw_mla`, the wide tensor-core tile's
  twin, takes the row. S runs over all 576 columns, V is the staged row's 512-wide latent (the one table, R3.31), and
  the 16 heads are a block's 16 rows. On by default for deepseek2's absorbed form with an f16 cache.
  `RUNNER_CUDA_ATTN_TC_WIDE=0` keeps the previous path, and quantized caches keep it too. Prefill on a CUDA MIG slice
  (1g.24gb, sm_120), DeepSeek-V2-Lite-Chat Q6_K, 64 tokens at temperature 0, load average 8 to 14:

  | Context | Absorbed, new | Absorbed, before | Naive |
  |---|---|---|---|
  | 4,096 | 201.64 tok/s | 114.69 | 191.57 |
  | 8,192 | 192.43 | 78.24 | 173.38 |

  - **The kill**, declared before the first read: under 0.9x of naive at 4K or 8K, or any identity miss. The read is
    1.05x and 1.11x. Decode is unchanged, and the outputs are byte for byte the previous path's.
  - **The default** followed the attention routes' rule against the publisher's float32 model (1,999 positions at
    4,000 to 5,999): mean KL 0.986x the previous path's, Wilcoxon p 0.79, McNemar 3/2.
  - **Gates:** `make test-cuda-attn-wide` takes the `--mla-wide` fixture at 1e-5 of the logit range, and catches V read
    from the wrong slice and a dropped rescale. `tests/test_attn_wide_repeat.py` checks the same bytes twice, the
    distance to the CPU and the default. The tile's existing families write the same bytes as before.
  - **The roster's CUDA cells, re-pinned:** the Q6_K row's CUDA tier-smoke reply had moved off its pin with R3.31's
    decode kernel (#631), and #631 did not compare it. It was stale on main from #631 until this change, and no
    release was cut in that window. The Q8_0 row's CUDA reply was moved by this change's own prefill route and
    re-pinned with it, so that cell was never stale (corrected after the release, R6.23). The decode kernel's own D1
    read, missing from #631, passes: mean KL 0.991x the previous kernel's, Wilcoxon p 0.37, McNemar 1/1. With that,
    both cells are retired, pinned and recorded again. The CPU cells match their pins unchanged. Record:
    `docs/compat-reports/r3.32-cuda-attn-tcw-mla-deepseek2-2026-11-05/`.

- **`--rewrite` and `--type-plan` write `q5_k`, `q5_1` and `iq4_nl` (R4.33).** These are ports of ggml's writers
  without an importance matrix (llama.cpp c1d0e7a). On the quant-golden slice their bytes are ggml's own, byte for
  byte, and the golden files are new. A `q5_k` rule on a row that is not a multiple of 256 writes `q5_1`, its
  32-block cousin: Mellum 2.1's 896-wide expert down projections need exactly that. With `--imatrix` the three are
  written without weights, as `q4_k` and `q6_k` are; importance-weighted writers for them are R4.34.
  - **Mellum 2.1 under 9 GB, written by Runner:** `--type-plan` with default `q8_0`, routers kept and `_exps.weight`
    at `q5_k`, from the BF16 parent, writes 8,994,830,528 bytes. Against JetBrains' BF16 parent in float32 it reads a
    mean KLD of 0.0293 and a margin-qualified top-1 of 98.74% (300 positions; bar 0.05 and 97%, declared as the kill
    before the read). bartowski's Q5_K_S reads 0.0289 and 98.33% at 31 MB less. Record:
    `docs/compat-reports/mellum2.1-rewrite-q5-2026-11-05/`.

- **deepseek2's absorbed form on CUDA decodes faster than the naive form, and its cache is half the size (R3.31).**
  - **The cache:** the absorbed form keeps one [k_pe | latent] row per token and layer, and its output reads only V's
    latent slice, which is K's row. The allocator still gave it a V table beside K, both 576 wide. Now the V table is
    the K table on the host and the device: 31,104 B a token at f16 instead of 62,208. At 32,768 tokens the Q6_K's
    cache goes from 2,038 MB to 1,019 MB, and its device total from 16.55 GB to 15.53 GB. Under `--kv k8v4` (K q8_0,
    V fp4) V keeps its own table. Scores are byte-identical before and after in both forms, on the CPU and CUDA, at
    every `--kv` format.
  - **The decode:** the absorbed form's 16 query heads share one 576-wide row, a shape the grouped decode attention
    did not take (at most 8 heads up to 256 wide). So each head read every cached row and its V copy for itself.
    `k_attn_dec_mla` reads a slice's rows once for all 16 heads and takes V from the K row, merged as the grouped
    kernel's partials are. Decode on a CUDA MIG slice (1g.24gb, sm_120), the Q6_K, 64 tokens at temperature 0, load
    average 4 to 18:

    | Context | Absorbed, new | Absorbed, before | Naive |
    |---|---|---|---|
    | 4,096 | 99.95 tok/s | 43.53 | 84.20 |
    | 8,192 | 89.43 | 26.25 | 66.12 |
    | 16,384 | 74.07 | 10.41 (R4.4.5 e) | 45.88 |
    | 32,768 | 55.01 | 5.36 (R4.4.5 e) | 28.30 (forced split, after the fix below) |

    The kill (absorbed under 0.9x of naive at 4k or 8k) was declared before the first read; the read is 1.19x and
    1.35x. gpu-identity with routing pinned reads 1.35e-5 of the logit range at 512 tokens (limit 1e-3).
    - The gate: an `--mla-wide` fixture with the real geometry, at 1e-5 in `test-gates-core`, plus
      `test_absorbed_mla_decode_*`, which repeats the decode byte for byte and holds it to the CPU at 1e-4 over
      1,000 positions. Both catch V read from the wrong slice. Only the long pytest catches a dropped rescale:
      gpu-identity's 512 tokens give each slice a single tile. So the pytest is in `test-cuda-gated`'s list, and
      gpu-identity's reach is a known limit of the gate.
    - At 16,384 the output differs from the previous kernel's at the 11th generated token. There the two kernels
      split " teacher" and " beacon" by 0.052 and 0.016 nats, a near tie. The previous kernel on this build
      reproduces R4.4.5 (e)'s output byte for byte.
    - `RUNNER_CUDA_ATTN_MLA=0` keeps the previous kernel.
  - **A forced split past the budget now says what it costs, and a run the device left says so at the end.** The
    naive form at 32,768 tokens never finished in R4.4.5 (e). `--gpu 99` forced all 27 layers where the budget
    holds 26, and the split line kept the fitted plan's bytes (`used=24.24GB` of 25.13). The prompt window sized
    itself from that figure, so it took 0.72 GB of "spare" that the 27th layer already held. The device had 0.01 GB
    left, the first launch ran out of memory, and the CPU finished the run in 84 minutes. The closing speed line
    did not say so.
    - The split line and the window now charge the forced layers. The same command reads `used=25.15GB`, gets a
      64-column window with 0.17 GB free, and stays on the device: 422 s, prompt 77.89 tok/s, decode 28.30 tok/s
      (load 11 to 14). Its output is byte for byte the absorbed form's.
    - When the device fails mid-run, a line after the closing speeds says the run finished on the CPU and that
      the speeds are the CPU's. Before, the only notice came before the first tile, and a harness that keeps only
      the end of stderr lost it.
    - An explicit `--gpu-layers` is still honoured past the budget. `RUNNER_GPU_FALLBACK=fail` still ends such a
      run instead of finishing it on the CPU.
  - **Not changed here:** the absorbed prefill (114.7 against naive's 191.7 tok/s at 4,096; R3.32 above). Decode still runs without a
    CUDA graph for deepseek2 (the shared-expert path), in both forms.

- **CUDA runs `qwen35moe` (R4.27 A2): Qwen3.6-35B-A3B's Q4_K_M row gains its CUDA cell.** The by-name hold in
  `cuda_admit` is gone. The device path is a composition that already existed: dense qwen35's trunk, the routed MoE
  kernels and the gated shared expert (`tile_shexp`).
  - **The gate:** `make test-cuda-qwen35moe`, a `cuda-gate.sh` step, holds the qwen35moe toys and the gated shared
    expert to the CPU's logits at 1e-5 of the range, with free and pinned routing, on every expert route. It also
    checks speculative identity and the host and split placements. It passed on its first run, and it catches four
    declared mutants: the shared expert's gate dropped, the shared expert skipped, the wrong DeltaNet gate, and a
    kernel that adds without the sigmoid.
  - **The real file**, on a CUDA MIG slice (1g.24gb, sm_120): all 40 blocks on the device, gpu-identity 3.6e-6 of the
    logit range with routing pinned (limit 1e-3), and tier-smoke output byte-identical to the CPU's.
    `compat_matrix`'s cpu_cuda check is new on the row and reads 8 of 9 exact, with one margin-qualified near tie.
  - **Not claimed:** fidelity on CUDA against the float32 reference (the row's KLD reads are the CPU's), speed, and the
    Windows CUDA host's cells (not run: no CUDA consumer host). The three upcoming qwen35moe rows still need their
    own gates. Record: `docs/compat-reports/qwen35moe-cuda-2026-11-05/`.

- **CI: the macOS suite runs daily and on dispatch, not on every push.** GitHub's hosted
  macOS arm64 pool is capacity-limited and cancelled queued jobs often enough to keep
  main red for hours on 2026-11-04 (28 cancellations against 4 real failures on main
  over two weeks). The pull-request `macos` job keeps the Metal gates and now also
  runs the suite's Metal pytest files, which is where the four real failures were.
  The Windows suite and every Linux job are unchanged.

- **DeepSeek-V2-Lite-Chat joins the validated roster at Tier 2 (R4.4.5 step d), with two of mradermacher's static
  files: the Q8_0, the fidelity file, and the Q6_K, the smallest to pass the fidelity bar.**
  - **The reads:** against the publisher's BF16 checkpoint run in float32 through its own `modeling_deepseek.py`
    (300 positions), in the absorbed form, the default.
    - The Q8_0 reads 0.0094 mean KLD and 100% margin-qualified top-1 on the CPU, and 0.0096 and 100% on CUDA.
    - The Q6_K (14.07 GB, a 16 GB card's fit) reads 0.0293 and 98.61%, and 0.0296 and 98.61%.
    - The Q5_K_M (0.0704, 96.76%) and the Q4_K_M (0.188, 90.28%) miss the bar (0.05, 97%) and get no row.
  - **The llama.cpp column:** llama.cpp 73a43d1 reads every file further from the reference. On the Q6_K it misses
    the margin half (0.0466, 96.76%). The multi-head latent attention form is not why: in llama.cpp's own naive form
    Runner reads 0.0294 and 98.61%, and is still the closer (p 2.4e-18).
  - **The other gates:** gpu-identity's pinned arm reads 1.85e-5 and 1.94e-5 of the logit range (limit 1e-3). The
    tokenizer differential and the template's 16 cases are identical. The `deepseek2` entry leaves
    `ARCH_GATE_EXEMPTIONS`. Record: `docs/compat-reports/deepseek-v2-lite-chat-2026-11-04/`.

## v1.13.0 - 2026-11-04

- **`GET /v1/models` says the context the server runs at.** Each entry
  carries `context_window`: the resident model's context, else the
  configured `-c`; absent only on a registry that has loaded nothing yet
  and was given no `-c`. The list carried only `id`, `object` and
  `owned_by`, so a client that plans its prompts had to guess: the Dojo
  pi extension guessed 32,768, compacted itself to a standstill against a
  262k server and overran a 16k one. The conformance fixture and the
  registry smoke carry the field.

- **CUDA: `RUNNER_MOE_HOT`, a file of hot experts kept on the device for
  `--cpu-moe` host layers; 1.52x warm decode on Mellum 2.1 Q8_0 on an RTX 3070
  (R3.30).** Corrected 2026-11-06: this entry first read 1.96x, a
  fresh-process number. Limits first:
  - It is opt-in and measured on one file on one card.
  - The hot file comes from a routing trace (`RUNNER_MOE_TRACE`), so it
    fits the text the trace saw.
  - v1 covers the plain top-k SiLU expert FFN only. Shared experts, expert
    biases or scales, gemma-4's dual branch and hyper-connections are
    refused at load, as is a malformed file or the knob without
    `--cpu-moe`.
  - The first request in a process is slower than the warm figure: 26 to 33
    tok/s against 40.69.

  A listed layer runs mixed. The device norms and routes, and runs the
  listed experts with the fused path's kernels on a bank of their slices.
  At the same time the host runs the rest with the CPU path's own expert
  function. The device then sums the slots in the router's order.

  Measured on Mellum2.1-12B-A2.5B-Thinking Q8_0 with 768 slots (5.07 GB),
  chosen by frequency from six traced prompts. On a prompt the trace never
  saw, warm decode in one long-lived process reads 40.69 against
  `--cpu-moe auto`'s 26.75 tok/s (medians of three interleaved rounds), with
  a hit share of 0.716 to 0.735. Step 1's model gives 1.53x at that prompt's
  share, so the measurement agrees with it. Its 1.66x assumed a share of
  0.785, which this prompt does not reach.

  The 1.96x first published here is a fresh-process reading, 27.82 against
  14.19 tok/s, and does not measure the placement alone: going from that
  reading to warm, `auto` gains 1.89x and `hot` 1.46x. A first request pays
  for paging the mapped file. Why the two placements paid different amounts
  is an inference from page-cache state, not an isolated measurement.

  `RUNNER_MOE_HOT_PROF=1` prints where each mixed layer's time goes.

  The identity gates (`tests/test_moe_hot.py`):
  - A layer with every expert listed gives `--cpu-moe 0`'s bytes.
  - A mixed layer's sum is neither placement's bytes. It is the same bytes
    run to run, and within 1e-7 relative perplexity of the all-host
    placement.
  - The host terms are the CPU path's own by construction, not by a
    separate byte test.

  The route record gains `moe_hot`. Report:
  `docs/compat-reports/r3.30-moe-hot-placement-2026-11-04`.

- **deepseek2 runs on CUDA in the absorbed multi-head latent attention, its default (R4.4.5 step c2).**
  - The cache holds one 576-wide head per token and layer, as on the CPU. The naive form (step c1, below) stays
    behind `RUNNER_MLA_NAIVE=1` as the oracle.
  - Three kernels:
    - `k_mla_q_absorb` builds each head's Q as [q_pe | k_bᵀ q_nope] from the layer's `attn_kv_b`, dequantized at
      load (226 MB on V2-Lite, charged in the VRAM plan);
    - `k_mla_kv_absorb` writes the one [k_pe | latent] row as both K and V;
    - `k_mla_v_absorb` takes each head's output latent through its v-rows before `attn_output`.
  - The 576-wide head runs on the plain attention kernels. The tiled and tensor-core prefill stop at 512 and 128.
  - **Measured on the toy:** `test-deepseek2`'s CUDA arm is 4.0e-5 from the plain-Python reference and 9.5e-7
    from the CPU. gpu-identity reads 2.4e-7 against a limit of 1e-5.
  - **Measured on DeepSeek-V2-Lite-Chat Q8_0, CPU against CUDA (gpu-identity):**
    - free routing reads 4.8e-4 of the logit range (limit 2e-3), with 7 of 14,820 expert slots differing (ties);
    - pinned routing reads 1.9e-5 (limit 1e-3).
  - **The cost:** scoring the 8,719-token gold corpus on the prefill route (`RUNNER_SCORE_CHUNKED=1`) on the CUDA
    MIG slice takes 191 s absorbed against 146 s naive, at a host load average of 17 to 30. On the CPU the absorbed
    form takes 798 s.
  - **Against the publisher's float32 `modeling_deepseek.py`,** CUDA's absorbed form against the CPU's on the
    prefill route, by the routes' rule:
    - at positions 4000-5999 (1,999 positions), the read declared for it: mean KLD ratio 0.9993, Wilcoxon p 0.11,
      McNemar 1/0;
    - on the 300-position house window it read 1.018, over the rule's 1.01, with the other two tests passing.
      Both reads are kept.

- **deepseek2 runs on CUDA in the naive multi-head latent attention (R4.4.5 step c1), with `RUNNER_MLA_NAIVE=1`.**
  - Three layout kernels, `k_mla_q`, `k_mla_kv` and `k_mla_pack_v`, put Q and K as [pe | nope] and V as [v | 0],
    and pack the value width before `attn_output`. They are copies only.
  - The latent's projections, its norm, the rope, the cache and the 192-wide attention run on existing kernels.
  - Still on the CPU, with a reason: the q-lora pair (`attn_q_a`: DeepSeek-V2 and V3, not V2-Lite), refused by
    name.
  - The multi-sequence microbatch declines deepseek2, as every MoE model: sequences decode one at a time on the
    device.
  - **Measured on the toy:** `test-deepseek2`'s CUDA arm is 4.0e-5 from the plain-Python reference and 4.8e-7 from
    the CPU. gpu-identity reads 9.8e-7 against a limit of 1e-5.
  - **Measured on DeepSeek-V2-Lite-Chat Q8_0, CPU against CUDA (gpu-identity):**
    - free routing reads 4.5e-4 of the logit range (limit 2e-3), with 11 of 14,820 expert slots differing (ties);
    - pinned routing reads 1.9e-5 (limit 1e-3).
  - Decode runs plain launches. The shared-expert path fails graph capture, as the gated shape (`shexpg`) already
    did before this change.

- **Qwen3.8-27B GSQ-RCO IQ3_S recovered moves to Tier 2: against Qwen's
  float32 reference it reads 0.41 mean KLD, 8x the bar (R4.32).** In the
  house window the published file reads 0.4119 mean KLD and 85.04%
  margin-qualified top-1; at positions 4000-5999 it reads 0.4084 / 86.12%.
  ISTA-DASLab's unmodified source reads 0.5616 / 82.48% and 0.4718 /
  86.36%. Both predict the corpus better than the parent (perplexity 21.54
  and 53.48 against 73.29), so GSQ-RCO is a learned derivative, not a
  faithful quantization. The scale recovery moved the file toward the
  parent. The row's earlier 0.0450 / 97.80% was measured against our BF16
  GGUF on 500 short Wikitext-2 prefixes, not against the publisher's
  float32 model; it stays beneath, labelled. Owner decision 2026-11-04:
  the row is maintained on report, stays published under a written
  exception, and keeps its records as dated evidence. The Flash-Next
  sibling's row gains the limit sentence; its own reference read is
  pending. Report: `docs/compat-reports/r4.32-qwen38-iq3s-reference-2026-11-04`.

- **`test-gpu-identity` judges a MoE file by its pinned-routing arm (R1.25).**
  With `RUNNER_CUDA_ATTN_TC_WIDE=1`, Qwen3.8-Flash-Next failed the free-routing
  arm at 2.41e-3 of the logit range against 2e-3. A dump of every device
  attention call found the wide-head kernel within 1.7e-6 of a float64
  reference over its own inputs, as close as the tile kernel. The device's
  router had chosen another expert at 3 of 45,600 slots, near-ties tipped by
  rounding, and with the CPU's choices pinned the run reads 4.42e-6. The free
  arm is now reported and does not fail by itself. The gate fails instead
  when more than 1% of slots tip: a Metal router with its bias dropped tips
  20 to 40% of them, which pinning alone would have passed. A G = 12 fixture
  (24 query heads on 2 KV heads, `make-test-model.py --heads`) joins
  `test-cuda-attn-wide`. No kernel changes.

- **CUDA runs the ungated (DeepSeek-style) shared expert on the device (R4.4.5 step c0b).** Before this, a MoE
  whose shared expert has no scalar gate sent the whole model to the CPU on CUDA unless `--cpu-moe` kept the banks
  on the host. `tile_shexp` now adds such a shared expert's output without the sigmoid.
  - Granite 4 hybrid keeps its banks on the host by name, because its CUDA cells were measured that way. A relu2 or
    gate-less shared expert keeps its refusal.
  - `make test` adds gpu-identity on the `shexp` fixture: zero routed experts, with the dense FFN as the shared
    expert. Its worst difference is 2.7e-7 against a limit of 1e-5, and on the device it scores bit for bit as the
    dense oracle does.
  - The gated shape (`shexpg`) and every CUDA MoE byte gate read the same before and after.

- **Metal Q4_K decode: the fast kernel unpacks its scales without a branch,
  1.18x on Qwen3-4B Q4_K_M with the same bits (R3.28.5).** Q4_K's 6-bit
  scales unpack one way for sub-blocks 0 to 3 and another for 4 to 7, and
  `k_mvg_q4_K`'s lanes sit on both sides within a simdgroup, so every lane
  paid both paths. Now three 16-bit loads at fixed offsets and a select
  give the same integers: the output is bitwise unchanged. On an Apple M1,
  Q4_K at Qwen3-4B's 2560-wide inputs reads 47.9 GB/s against 38.2, and
  decode over R3.28.3 is 17.59 against 14.92 tok/s on Qwen3-4B, 1.15x on
  Llama-3.2-3B, 1.13x on Llama-3.2-1B, 1.02x on K2-Horizon-1B (whose
  1536-wide bulk runs the identity kernel).
- **Metal Q6_K decode: the narrow-slice kernel too, 1.20x more on
  Qwen3-4B Q4_K_M (R3.28.3).** A Q4_K_M file keeps its LM head, attn_v and
  half its ffn_down at Q6_K, which decoded at about 27 GB/s on an Apple M1.
  Q6_K inputs at least 1536 wide now run `k_mvg_q6_K`, R3.28.2's shape for
  Q6_K, at 51 to 55 GB/s: decode over the R3.28.2 build, Qwen3-4B 15.08
  against 12.56 tok/s, Llama-3.2-1B 1.36x, Llama-3.2-3B 1.23x,
  K2-Horizon-1B 1.24x. Unlike `k_mv2_q6_K`, whose bits are the identity
  kernel's, it is a tolerance route: `test-mv-tol` 0/64 flips on Qwen3-4B,
  and its float32-reference row is in the PR. `RUNNER_METAL_MVG=0` restores
  both previous kernels. The route detail says
  `mv=mvg-q4_K>=2048+q6_K>=1536`.
- **Mellum2.1-12B-A2.5B-Thinking gains a Tier 2 row for bartowski's Q5_K_S
  (8.96 GB), the smallest Mellum 2.1 file measured to pass the bar.** Against
  the publisher's BF16 parent in float32 it reads 0.0289 mean KLD and 98.33%
  margin-qualified top-1 (plain top-1 93.0%), where the publisher's Q4_K_M
  reads 0.1172 / 92.89% and Runner's own 9.38 GB selective-precision attempt
  0.0983 / 92.89%. Four published Q5-band files were scored; both with
  experts at Q5_K pass, both with experts at Q4_K miss. Owner decision
  2026-11-04; a third party's file, pinned by sha256 and repository revision.
  Report: `docs/compat-reports/mellum2.1-q5-band-2026-11-04`.


- **CUDA's MoE router can leave the selected weights as the softmax gave them (R4.4.5 step c0).** `k_moe_route`
  and its replay twin take a renormalise flag, which is the file's `expert_weights_norm`. The eager path's host
  routers read the same field. Every family that renormalises today keeps the flag on, with the same bytes:
  - the CUDA MoE byte gates read the same before and after, and so do the scores of the MoE fixtures;
  - in the PTX, only `k_moe_route` and `k_moe_route_replay` changed, of 495 kernels.

  A softmax router with `expert_weights_norm` false no longer sends the model to the CPU on CUDA. deepseek2 is
  still refused on CUDA by name until its attention lands. Metal's router has no such flag and keeps the
  refusal. `make test` gains CPU-against-device runs on the shared-router pair `rsoft` and `rsoftnn` and on
  `gnonorm` (worst 1.9e-7 and 2.7e-7, limit 1e-5). `test-moe-router` checks that the pair differs on the host
  (0.041).

- **DeepSeek-V2 (`deepseek2`) runs on the CPU (R4.4.5 step b).** These are the limits for now:
  - The GPU backends run it on the CPU with a reason, except CUDA's naive form (step c1, above).
  - The absorbed multi-head latent attention by default: the cache holds one 576-wide head a token a layer
    ([k_pe | normed latent], read by every query head; 1,152 values with the K and V rows against the naive form's
    6,144 on V2-Lite). `RUNNER_MLA_NAIVE=1` keeps the naive form (16 heads of 192), the identity gate's oracle.
  - The legacy unsplit `attn_kv_b` only; the split `attn_k_b`/`attn_v_b` layout is refused.
  - No LoRA.

  Against the publisher's float32 code (`modeling_deepseek.py`, `--trust-remote-code`), DeepSeek-V2-Lite-Chat Q8_0
  reads mean KLD 0.0100 and margin-qualified top-1 100% over the house window. llama.cpp reads 0.0155 / 99.54% on the
  same file, and Runner is the closer (Wilcoxon p 9e-15). That read is the naive form. Against the same reference at
  positions 4000-5999 of the gold corpus, the absorbed form is not distinguishable from it: mean KLD 0.01286 against
  0.01285 (1.001x), Wilcoxon p 0.095, McNemar p 1.0, margin-qualified top-1 99.48% for both. The rest is the
  publisher's:
  - the softmax scale mscale² / sqrt(key_length);
  - softmax routing without renormalisation;
  - the `deepseek-llm` pre-tokenizer (`test-tokenizer-classes`; 0 of 721 corpus segments and 0 of 99,442 vocabulary
    entries differ from the publisher's tokenizer);
  - the `deepseek2` chat template, conformant with 16 of 16 cases token-identical.

- **Metal Q4_K decode: a faster fast kernel, 1.19x on Qwen3-4B and
  Llama-3.2-3B Q4_K_M (R3.28.2).** The Q4_K inputs the fast route takes (at
  least 2048 wide on dense models) run `k_mvg_q4_K`: eight lanes share a
  superblock and a lane's 32 activations serve four rows, where
  `k_mvf_q4_K` gave a lane 64 activations for one quarter of two rows. On
  an Apple M1, decode: Qwen3-4B 12.59 against 10.61 tok/s, Llama-3.2-3B
  17.09 against 14.34, Llama-3.2-1B 1.08x, K2-Horizon-1B 1.02x. A
  tolerance route like the one it replaces: `test-mv-tol` 0/64 flips on
  Qwen3-4B, and its float32-reference row (mean KL 0.9999x of the old
  kernel's) is in the PR. A decode microbatch's columns each run the
  one-column arithmetic, so they keep the solo step's bits.
  `RUNNER_METAL_MVG=0` keeps `k_mvf_q4_K`. The route detail says
  `mv=mvg-q4_K>=2048`.

- **CUDA: the wide-head tensor-core prefill attention is on by default for
  qwen35 at head width 256 (Qwen3.5, 3.6 and 3.8) (R3.6.3.2).** On
  Qwen3.8-27B (GSQ-RCO IQ3_S, a CUDA MIG slice), 16K prefill reads 1.82x
  (329.6 against 181.1 tok/s) and 4K 1.23x. Decode is unchanged. The D1
  rule passes: mean KL 0.9998x the tile's, Wilcoxon p 0.82, McNemar 0/0
  over 2,000 positions. gpu-identity is ok on both settings. The default
  keys on the measured width, and `RUNNER_CUDA_ATTN_TC_WIDE=0` keeps the
  tile. Not yet measured: Metal and the RTX 3070 at these sizes. Report:
  `docs/compat-reports/r3.6.3.1-cuda-attn-tc-wide-qwen35-2026-11-04`.
  The Flash-Next sibling (qwen4exp) stays on the tile: on the same slice,
  with 26 of its 48 layers on the CPU, its 4K prefill reads 0.992x, and
  16K and D1 were not run. With routing pinned, gpu-identity reads
  4.42e-6 (R1.25). Report:
  `docs/compat-reports/r3.6.3.1-cuda-attn-tc-wide-qwen4exp-2026-11-04`.

- **The CUDA int8 prefill route (`RUNNER_CUDA_I8=1`) stays opt-in for every
  type (R3.14.1).** Against the publisher's float32 models, Q8_0 (Qwen3-0.6B)
  reads 1.51x the default route's mean KL. Q4_K with Q6_K (Qwen3-4B Q4_K_M)
  reads 1.008x, and is worse on the paired Wilcoxon (p 0.037). Both miss the
  routes' rule. Report: `docs/compat-reports/r3.14-cuda-int8-prefill-2026-11-03`.

- **Qwen3.6-27B joins the validated roster at Tier 2 on ggml-org's
  conversions (no publisher GGUF exists), and the Q4_K_M fails the fidelity
  bar.** Against Qwen's BF16 checkpoint in float32 at use (300 positions,
  CPU): the Q4_K_M reads 0.129 mean KLD and 95.9% margin-qualified top-1 on
  Runner, 0.154 and 93.8% on llama.cpp 73a43d1, against a bar of 0.05 and
  97%, and the gap grows with depth; the file itself is the distance (0.101
  top-20 KL against Runner's own Q8_0). The Q8_0 passes on Runner (0.0273 and
  99.59%; one reference near-tie carries most of the mean) and is the row's
  fidelity file with a CPU cell; llama.cpp reads 0.0626 on it, above the bar.
  Tokenizer differential 0 of 247,091 entries, template byte-identical to
  Qwen's with 23 of 23 conformance cases, tool gate A and B pass and C passes
  buffered. Record: `docs/compat-reports/qwen3.6-27b-2026-11-04`.

- **On CUDA with every layer on the device, `--draft lookup` now drafts 7
  tokens a round by default and `--draft mtp` 2; every other source,
  backend and partial offload keeps 4 (R3.3.7).** On an
  RTX 3070, lookup on an in-context edit reads 2.86x plain at 7 against
  2.43x at 4. Qwen3.5-4B's NextN head reads 1.13x at 2 on code and tool
  calls, against 1.01x and 1.18x at 4; on prose it stays a loss (0.88x at
  2). An adaptive width that followed each turn's acceptance was measured
  first and stopped by its kill read: 0.97x of the best fixed width, where
  it needed 1.05x. An explicit `--draft-k` is unchanged. Report:
  `docs/compat-reports/r3.3.7-draft-width-defaults-2026-11-03`.

- **`RUNNER_CPU_I8K=1`: an opt-in int8 decode route for Q4_K and Q6_K on
  AVX2 CPUs, off by default because it fails the fidelity bar (R3.1.4).** On
  a 4-core i7-7700K, Q4_K_M decode reads 1.23 to 1.39x the f32 route, 0.85
  to 0.96x of llama.cpp. Against Qwen's float32 models its mean KL is 1.09x
  the f32 route's on Qwen3-0.6B and 1.17x on Qwen3-1.7B (worse on the paired
  test, p 9e-32 and 3e-49). The small-core gap to llama.cpp is that int8
  rounding, which the default does not pay. `RUNNER_CPU_PROF=1` prints where
  a CPU decode step goes. The route record's `cpu_act` names the route, and
  `--score`'s `score_route` now carries `cpu_act`. Report:
  `docs/compat-reports/r3.1.4-cpu-small-core-decode-2026-11-04`.

- **`--quant-error` costs a row its type cannot describe at the type the
  rewrite writes instead.** A tensor whose rows the asked type's block does
  not divide (a MoE expert width of 896 under `q4_k`) read `null` in the
  report, so `runner derive --target-size --low q4_k` refused the plan
  ("has no q4_k entry") where `--rewrite --quant q4_k` of the same parent
  writes the tensor at `q5_0` through the width fallback. The report now
  costs such a row at that fallback type and names it in `written_as`; a
  plan built from the report predicts the file the rewrite makes, and derive
  plans every MoE whose expert width is not a multiple of 256. Found by a
  derive run on the lab box.

- **Nemotron 3.5 Lightning-30B-A3B joins the validated roster at Tier 2 (the
  publisher's Q4_0; the Q8_0 is the fidelity file).** Against NVIDIA's BF16
  checkpoint in float32 at use, at the model's declared time-step limit: Q4_0
  0.0476 mean KLD and 97.62% margin-qualified top-1, Q8_0 0.0146 and 99.05%,
  the BF16 GGUF 0.0055 and 100% (300 positions, CPU); llama.cpp 73a43d1 reads
  0.0499 and 0.0173 on the same files, and Runner is the closer by per-position
  KLD on both. Every number is from the build with the sigmoid router (#575).
  Two corrections to the reference are recorded: transformers 5.14.1's naive
  Mamba-2 scan is wrong across chunks (5.16.1 is the reference), and the 5.16.1
  mixer clamps dt at the initialisation floor rather than the config's declared
  limit (0.246 against 0.0055 on the BF16 file; the declared limit predicts the
  corpus better). CPU only: the router has no device kernel. The three
  `qwen35moe` Upcoming rows no longer claim the architecture does not load.
  Record: `docs/compat-reports/nemotron-3.5-lightning-30b-a3b-2026-11-03`.

- **Metal: the fast Q4_K decode matvec is the default only for inputs at
  least 2048 wide (R3.28.1).** On 1536-wide inputs it was 15% slower than
  the identity kernel (K2-Horizon-1B Q4_K_M: 11.11 against 9.62 ms of Q4_K
  time a token on an Apple M1), while it won at every width from 2048 to
  8192 on the Llama-3.2-1B/3B Q4_K_M files it was promoted on. Narrower Q4_K
  inputs now keep the identity kernel, so K2-Horizon-1B Q4_K_M decodes
  1.061x faster (39.7 against 37.5 tok/s). The Llama files are unchanged:
  their `--score` output is byte-identical to before, decode 0.997x and
  1.022x, inside the run-to-run spread. Measured on one laptop, with one
  file for the narrow case; widths between 1536 and 2048 were not measured
  and take the exact kernel. The route detail says `mv=fast-q4_K>=2048`.

- **Granite-4.2-30B joins the validated roster at Tier 2, and its publisher's
  Q4_K_M fails the fidelity bar.** Against IBM's BF16 checkpoint in float32
  at use (300 positions, CPU): the Q4_K_M reads 0.139 mean KLD and 93.9%
  margin-qualified top-1 on Runner, 0.151 and 92.5% on llama.cpp 73a43d1,
  where the bar is 0.05 and 97%; the file itself is the distance (0.111
  top-20 KL against Runner's own Q8_0). IBM's Q8_0 passes on both engines
  (Runner 0.0041 and 99.56%, llama.cpp 0.0241 and 99.12%) and is the row's
  fidelity file, with a CPU cell on the Linux CUDA host (31.1 GB, more than a
  24 GB device holds). Tokenizer differential 0 of 99,474 entries, template
  byte-identical to the publisher's with 24 of 24 conformance cases, tool gate
  A and B pass and C passes buffered (streamed, the run spent its reasoning
  budget before any call). Record:
  `docs/compat-reports/granite-4.2-30b-2026-11-04`.

- **Nemotron 3.5 Lightning (`nemotron_h_moe`) routed its experts with the wrong
  function; fixed. Every nemotron_h_moe file has given wrong output since
  v0.2.0.** The publisher's router is a sigmoid; the published GGUFs carry no
  `expert_gating_func` key, and the loader read the missing key as a softmax.
  On the real Lightning-30B Q8_0 against llama.cpp's logits over the same
  1,109 positions: perplexity 18.31 before, 11.83 now (llama.cpp 11.73), top-1
  agreement 63.2% before, 95.8% now. A file that declares the key keeps it.
  Requantize nothing; reload the file.

- **`RUNNER_DEBUG_TIME` reports the attention core inside the attention
  stage.** A new `attn core` line times only the scores, softmax and values
  pass, without the projections. Timing only; output unchanged.

- **Metal prefill attention takes 512-wide heads on the tile, byte-identical,
  by default for gemma-4's E-series (R3.6.3.2).** gemma-4's 512-wide global
  layers ran the one-column kernel over the whole history, 66% of an 8K
  Nemotron 3.5 Bolt E4B prefill on the 8 GB laptop. They now take
  `k_attn_tile`, the kernel 256-wide heads already used, with `k_attn`'s
  arithmetic in `k_attn`'s order, so the output is the same bytes: one
  global layer's attention at 8K ran in 3.4 s against 25.0 s, and the 8K
  prefill's GPU time went from 274 s to 117-123 s (about 2.3x; 4K: 1.43x in
  tok/s). Decode is unchanged. Measured only on the 8 GB laptop and only to
  8K: a 16K prefill of this file does not fit that laptop without swap. On
  by default for gemma4 with per-layer embeddings, the family CUDA's wide
  kernel defaults to; `RUNNER_METAL_ATTN_TILE_WIDE=0` keeps the one-column
  kernel for heads wider than 256, `=1` takes the tile on any model. The
  route detail says `attn_wide=on`.
- **CUDA prefill attention for 256- to 512-wide heads runs on the tensor
  cores, by default for gemma-4's E-series (R3.6.3.1).** gemma-4's 512-wide
  global layers took the tile's 8-row kernel on the CUDA cores, 79% of a
  32K prefill on an RTX 3070. `k_attn_tcw` takes them, and the 256-wide
  sliding layers, with `k_attn_tc`'s operands: one global layer at about
  31K in 55 ms against 352. Nemotron 3.5 Bolt E4B prefill: +8% to +19% at
  512 tokens, +30% to +40% at 4K, 2.0x at 16K, 2.6x at 32K; decode and the
  greedy text unchanged. On by default for gemma4 with per-layer embeddings
  (owner decision), its float32-reference row on Bolt E4B not
  distinguishable from the tile's (mean KL 0.9965x, Wilcoxon p 0.188);
  `RUNNER_CUDA_ATTN_TC_WIDE=0` keeps the tile. fp16 caches
  only; a layer with sinks (gpt-oss, flint) keeps the tile. The route
  detail says `attn_wide=on`. Report:
  `docs/compat-reports/r3.6.3.1-cuda-attn-tc-wide-2026-11-03`.
- **CUDA decode attention for eight query heads per KV head (R3.29.1).**
  With eight heads in a block, the grouped decode kernel read each V pair
  once per head and each probability alone, and was bound by shared-memory
  work. A thread now reads its V pair once for all its heads and the
  probabilities four keys at a time, in the same order: the scores are the
  same bytes. On an RTX 3070 at 16K, decode against the previous build:
  1.18x on Qwen2.5-3B (0.66x to 0.78x of llama.cpp b11748), 1.08x on
  Qwen2.5-7B, 1.03x to 1.04x on Qwen3-4B and Llama-3.2-3B; 1.00x to 1.01x
  at 512. Report: `docs/compat-reports/r3.29.1-cuda-decode-gqa8-2026-11-03`.

- **`RUNNER_SPEC_PROF=1` prints its profile line on a schema-constrained run too.** The
  speculative walk left through three early exits when a constrained document
  completed (a close before a token's forward, mid-round, at a plain token);
  they printed the round summary but not the profile. The MANUAL's
  `--draft-k` row now states the range, 1 to 15 (a round verifies the drafts
  plus the correction, at most 16 rows).


- **A Metal layer split no longer refuses a file stored in name order or with
  its output norm last.** A split wraps a prefix of the file, and it refused
  layers stored out of block order and any non-layer tensor after them; Qwen's
  Qwen2.5-1.5B-Instruct Q8_0 does both (blk.0, blk.1, blk.10 to blk.19, blk.2,
  ..., then `output_norm.weight`), so `--gpu N` on it ran wholly on the CPU.
  The wrap now reaches the farthest tensor of the device's layers whatever
  their order, and tensors a split never reads from the device (the head, the
  embedding, gemma-4 E-series' per-layer tables) may follow the layers; any
  other tensor there still refuses the split, and the message names it.


- **A session resume or a shared prefix is no longer refused when the model's
  path is spelled differently.** The engine's model key hashed the `-m` path
  as typed, so the same file loaded as `./models/m.gguf` and as its absolute
  path had two keys: a session image written under one spelling was refused
  under the other, and two slots did not share prefix-cache entries. The key
  now names the file (its stat identity and content samples, or its resolved
  path where stat cannot tell); a file replaced in place still changes it.
  **What changes for callers:** the key version moves to v3, so prefix caches
  and session images saved by an earlier build are refused once (the refusal
  says so). A refusal that remains now names what
  differs, with both values and the flag that resumes it ("context 256 in the
  image, 128 here: resume with -c 256").

- **CUDA decode attention reads the cache once (R3.29).** The grouped decode
  kernel read K twice, a pass for the softmax maximum and then K with V; it
  now makes one pass with a running maximum, stages K and V as 16-byte
  chunks prefetched into registers, and fits three blocks an SM. On an RTX
  3070, decode against the previous build: 1.00x to 1.02x at 512 tokens,
  1.24x to 1.38x at 16K and 1.37x to 1.55x at 32K across Llama-3.2-3B,
  Qwen2.5-3B, Qwen3-4B and Qwen2.5-7B. It is still 0.66x (Qwen2.5-3B) to
  0.86x (Llama-3.2-3B) of llama.cpp b11748 at 16K. The order of the sums
  changed: the same bytes run to run for a geometry, and its
  float32-reference row is not distinguishable from the two-pass kernel's
  (Qwen3-0.6B, mean KL 1.0006x, Wilcoxon p 0.890). Report:
  `docs/compat-reports/r3.29-cuda-decode-attention-2026-11-03`.

- **A recurrent draft model no longer refeeds its whole context on every
  rejected draft.** A draft whose layers fold state (LFM2, Gated DeltaNet)
  went back to position 0 on each rejection, so its catch-up refed the whole
  context every round: an LFM2.5-350M draft for LFM2.5-1.2B on a four-core
  CPU ran at 0.13x of plain decoding on prose (113 rounds at about 526 ms of
  draft work each). Each round now snapshots the draft's fold after its
  catch-up and a rejection restores it, so a round feeds the draft at most
  `--draft-k` + 1 tokens; output is unchanged. CUDA rolls a fold back by one
  forward only, so a CUDA-resident recurrent draft keeps the old refeed. The
  spec line on stderr gains `draft catch-up N`.

- **`RUNNER_CUDA_I8=1` runs CUDA prefill on int8 tensor cores (opt-in,
  compute capability 8.0+, Q8_0, Q4_K and Q6_K).** The activations are
  quantized per 32-element block, and the weights' own integer quants go to
  `mma.sync` unconverted. It is off by default: `test-tc-tol` fails on
  Qwen3-0.6B Q8_0, whose greedy tokens diverge from the scalar route at a
  near-tie, although no file tested flips a top-1 token in the gate's 64
  teacher-forced positions. On an RTX 3070 the GEMM time is 1.66x less than
  the fp16 route's on Qwen3-1.7B Q8_0 and 1.18x less on Llama-3.2-3B Q4_K_M,
  and prefill gains 8% to 36% across four files, with decode unchanged. The
  read against the publisher's float32 model did not run (stopped for low
  memory on the 16 GB host). The route detail a receipt carries says `i8=on`
  or `i8=off`. Report: `docs/compat-reports/r3.14-cuda-int8-prefill-2026-11-03`.

- **The sliding-window KV ring is on by default where the flat cache does
  not fit (`--kv-ring auto|on|off`), and rewind and the prefix cache work
  with it.** The cost first: under the ring a shared-prefix hit is lost
  when the cached prompt runs past the shared part by more than one batch
  plus one token, which the flat cache would serve; a saved prefix cache, an
  exported named context or a session image takes only a ring with no gap
  below its span; Metal's decode microbatch runs ring members one at a
  time. The auto rule fired wrongly on the first real file it met
  (gemma-4-E4B at 32k on an 8 GB RTX 3070, where the flat cache costs no
  layer): the device weight estimate counted E-series' host-only per-layer
  token table, and now leaves it out. That estimate also fed the VRAM
  registry's claim, the `-c 0 --reserve-vram` auto-fit and the KV-trade
  note, all too large on E-series files since 2026-08-01. Rewind and
  snapshots resume from what the ring still holds by one rule
  ([docs/kv-ring.md](docs/kv-ring.md)), with the ring's oldest held position
  tracked where KV is written; output is bit-identical to the flat cache
  through the server on the CPU, Metal and CUDA paths. Measured on the RTX
  3070 (gemma-4-E4B Q4_0, 4k to 32k, two interleaved rounds): the cache is
  2.2 to 3.2 times smaller (1879 to 579 MB at 32k) with the same output;
  speed is level, not better (prompt within 1.2% bar one flat-arm round at
  8k, decode within 1.2% at 8k to 32k; the 4k rows stopped after one
  token), because the flat cache costs that card no layer; and a repeated
  3,000-token system prompt hits the prefix cache on both arms (0.16 to
  0.17 s). On that card the flat cache
  fits at 32k, so auto leaves the ring off there. `RUNNER_KV_RING` still
  overrides the flag. **What changes for callers:** on a sliding-window
  model whose flat cache does not fit, the ring is now on without asking;
  `--kv-ring off` restores the flat cache. On CUDA, `-c 0 --reserve-vram`
  fits a larger context on a gemma-4 E-series file than before, because the
  weight charge drops by the per-layer token table (E4B: 5,155 to 2,843
  MB), and the VRAM registry claims that much less. `--info fit` is
  unchanged: it already left the table out.

- **A qwen3_xml call closed by the next call's opener stands, on an auto
  turn only, and says so.** Nemotron 3.5 Lightning at temperature 0 writes
  three `<tool_call>` openers, each with a complete function block, and one
  `</tool_call>` at the end; Runner refused the whole turn as
  `envelope_unmapped`. On a parse-only (auto) turn a call whose function block
  is complete now ends at the next opener, goes out when that opener arrives,
  and the turn reports `finish_detail: tool_call_implicit_close`. A last call
  with no closer, an attribute-spelled parameter (`<parameter pattern="...">`,
  the same model at its own sampling) and every grammar-framed turn stay
  faults. Prior art disagrees: vLLM's Python parser (v0.11.0) recovers these
  calls, its Rust parser and llama.cpp's require the closer, and llama.cpp
  constrains the turn after its first opener, which Runner's auto turn does
  not (measured 2026-09-14). The fixture is the model's own temperature-0
  bytes.

- **`test-gpu-identity` pins a MoE file's routing and bounds dense and
  pinned runs at 1e-3 of the logit range (was 2e-3).** A MoE file now runs
  on the device twice: with its own routing, bounded at 2e-3 as before, and
  with the CPU run's expert choices replayed, bounded at 1e-3. A top-8
  router picks another expert wherever two gates nearly tie, so free routing
  on Qwen3-30B-A3B read 1.46e-3 on CUDA; the pinned run measures the
  arithmetic alone. The gate prints how many expert slots the pin forced and how
  many differed from the device's own choice, and fails when the pin missed
  a routing. 1e-3 sits 2.5x over the worst dense real file measured (IQ1_S,
  4.0e-4) and 2.05x under the smallest real mistake (attention scale x1.05,
  2.05e-3).
- **`RUNNER_MOE_REPLAY` pins Metal's router too** (the batched and the
  per-column route), with the same counters as the CPU and CUDA routers.

- **Receipts name the model by base name and sha256, never by host path,
  and carry the route they ran (`xyntetik.runner.transcript.v2`).** A
  receipt is evidence a user may post, and v1 wrote the path the model was
  loaded from, which showed where it sat on the machine. A `--record` or
  `--transcript` file and every served receipt now name the model and the
  adapter as `{"name", "sha256"}`, with `"source": "hf:<owner>/<repo>@<rev>"`
  for a file fetched with `-hf` (the commit the Hub's listing named), and
  link a step's record by base name too. `profile.route` records the
  backend, `cpu_act` and `backend_detail`; `--verify` requires the same
  route on the same backend (`UNVERIFIABLE` naming both routes otherwise)
  and reports it on another (`"route"` in the verdict: `matched`,
  `other backend`, `not recorded`). Without `-m`, `--verify` loads the file
  the record names from the working directory; the sha256 is checked before
  the replay, and a mismatch is `UNVERIFIABLE` naming both files. v1
  receipts still verify with `-m`, and `--lineage`, bundles and packs read
  both versions. **What changes for callers:** the receipt JSON loses
  `model.path` and `adapter.path` (read `name`), and its schema_version is
  v2.

## v1.12.0 - 2026-11-03

- **`--quant q4_k` wrote corrupt tensors on rows that divide by 32 but not
  256 (since v0.5.7); fixed.** Such a row cannot take a 256-block K-quant,
  so the quantizer falls back to Q5_0, the 32-block type of the nearest bit
  budget, as llama.cpp does. There was no Q5_0 writer: the row's float
  bytes were written under a Q5_0 header. **Any file Runner wrote with
  `--quant q4_k` (through `--quantize`, `--rewrite`, `runner derive` or a
  type plan) from v0.5.7 on, from a model with such rows, is corrupt and
  must be requantized.** Such rows occur in SmolLM2-135M (576 wide),
  Qwen2.5-0.5B (896) and Kvist-14B (5,760), among others. To check a file:
  the quantize histogram lists `Q5_0` tensors (`--info fit` names the weight
  type), and the file scores NaN or garbage (SmolLM2-135M: NaN logits, no tokens). Q5_0 now
  has its writer, byte-identical to ggml's; a fallback is only ever a type
  the quantizer writes; and a row with no writer stops the quantizer
  instead of being copied. Files from llama.cpp, and Runner files at
  other types, are not affected.

- **Smaller fixes from the pre-1.12.0 sweep.** A `--quant` name the
  quantizer does not take (`Q4_K_M`, llama.cpp's mix name, or `q4_K`) is
  refused with the names it does take. `runner derive` whose rewrite fails
  leaves nothing beside OUT; it left the parent's score file. And the
  route-line wording for the two 1.12.0 opt-ins is corrected: the route line
  and the `--score` and `--bench-json` route records name them, receipts do
  not.

- **Six messages and edges the pre-1.12.0 sweep found.** A one-shot run
  whose `-n` does not fit in `-c` now ends with `[context full: -c N holds
  the prompt and K of the M tokens -n asked for; rerun with -c X or
  larger]` instead of stopping silently (the server already finished such a
  request with `length`, and the tests now pin it). A GGUF cut short in its
  metadata or tensor table says it is truncated, with the bytes it needs and
  the bytes it has, instead of `bad GGUF metadata value for ...`. `--verify`
  on a receipt a JSON tool re-serialized says so (the chain hash covers the
  runner's compact bytes) instead of `no chain in record`, which stays for a
  record that has none. `runner attest VERB --help` prints that verb's usage
  and exits 0. A value flag at the end of the line names itself (`error: -p
  needs a value`) before the usage. A failed `-hf` lookup removes the cache
  directory it created, never one that was there before. Each has a test
  that fails without it.

- **CUDA MoE routing traces record each expert's output norm.** The CUDA
  trace left the norms field empty, so a trace taken on the device could not
  be compared field by field with the CPU's. Every CUDA MoE path (fused,
  grouped, eager, by expert, and their gemma-4 forms) now writes the CPU's
  definition: the expert output's norm with the routing weight divided back
  out and the down bias added. `make test-cuda-moe-replay` compares every
  device record with the CPU's: experts exactly, gates within 1e-3 and norms
  within 2e-2 relative (measured spread: gates 5.2e-5, norms 4.0e-3 in the
  chunked mode). Tracing only; scores are unchanged. The real-file
  `test-gpu-identity` bound stays 2e-3: dense and IQ files read at most
  4.0e-4, but Qwen3-30B-A3B reads 1.06e-3 to 1.46e-3 because near-tie top-8
  routing choices flip between backends (21 of 1,872 on one prompt; 1.5e-3
  with the CPU's routing replayed). The audit report has the tables (R3.25).

- **CUDA decode applies the attention temperature past its first step.**
  Llama 4 and mistral3 files that declare the attention temperature
  (Devstral Small 2, Ministral 3, Magistral Small 1.2) decoded with a stale
  factor on CUDA when the whole model ran on the device. The decode graph is
  captured once and replayed, and the factor reached the kernel as a host
  argument, so every replay used the factor of the position it was captured
  at. A graph captured below the floor (llama 4: 8,192 positions; mistral3:
  the file's original context length) applied no temperature past it, and
  one captured above it kept that factor for later tokens, including those
  of a later, shorter request. Prefill, and decode with graphs off
  (`RUNNER_CUDA_GRAPH_OFF=1`), were right. The graph is now captured again
  when the factor moves, once per floor. In llama 4 since v0.1.5-alpha and
  mistral3 since 1.11.0. Found by the CUDA gate audit (R3.25), which also
  puts the fixture that shows it on the 1e-5 bound.

- **CPU prefill attention scales with threads again.** A prefill batch's
  attention ran one thread-pool pass per query row, so on many cores the
  dispatch, not the arithmetic, set the time, and more threads made it slower
  (3.7 s at 16 threads against 1.9 s at 4 for a 588-token prompt on a
  Threadripper, Qwen3-0.6B Q8_0). It now runs the whole batch in one pass over
  heads, each running every row exactly as before: outputs are byte-identical
  at any thread count, and nothing more is allocated. On that host, at 16
  threads, three interleaved rounds against the previous build read 0.87x to
  1.72x (medians 1.10x to 1.52x over SmolLM2-135M and Qwen3-0.6B, GCC and
  Clang builds); at 1 and 4 threads it is level. `RUNNER_ATTN_BATCH=0`
  restores the per-row passes. On Apple Silicon with an f16 cache, the
  attention's value sum also holds 16 channels in registers across the whole
  span instead of storing and reloading them at every position, with each
  channel's sum
  unchanged: on an Apple M1 a 3,497-token SmolLM2-135M prefill went from 106
  to 121 tok/s on one thread and from 234 to 334 on four, the scores
  byte-identical to the previous build (R3.27).

- **`--train-gpu` measured on a sparse-MoE model, and no faster there on
  a small device.** Qwen3-30B-A3B Q4_K_M trains whole on a 24 GB MIG slice
  (15.9 GB on the device), with losses within 1.6e-3 of the CPU trainer's
  over three steps. A step took 52 to 55 s against 53 s for the CPU trainer
  on 16 threads: the step is device-bound on many small expert products.
  The no-norm MoE fixture joins `make test`'s device-mode gradient gates.
  Record: `docs/compat-reports/train-gpu-2026-11-02-linux-x86_64-qwen3-30b-a3b.md`
  (R17.3.3).

- **Shared-expert MoE and qwen35moe on Metal.** Metal had no path for an
  always-on shared expert, so every MoE model with one (the Qwen2-MoE form)
  and `qwen35moe` (Qwen3.5/3.6-35B-A3B and the models built on them) ran on
  the CPU. The gated shared expert now runs on the device, composed from the
  dense FFN's kernels, a one-row router matvec and the per-column sigmoid
  scale, and `qwen35moe` follows. Checked on toy models only, to the CPU's
  logits at 1e-5 of range (`make test-metal-qwen35moe`, residue 5.9e-7 at
  most; a dropped sigmoid or a dropped add fails by 2.7e-3 or more): no
  `qwen35moe` file fits the 8 GB Mac it was built on, so no file row is
  measured on Metal. The gate-less squared-ReLU shared expert keeps the CPU,
  and CUDA still refuses `qwen35moe` (R4.27).

- **A PR-time check for POSIX calls the Windows build lacks.**
  `scripts/check-portability.py` runs in every PR's metadata check and fails
  on a call to a function the MinGW-w64 CRT does not declare (`setenv`,
  `unsetenv`, `realpath`, `getline`, `fork`, `mmap` and twelve more,
  each verified against the mingw-w64 headers) outside code the preprocessor
  keeps from Windows. The Windows suite runs on main only, so a bare `setenv`
  in a C test had passed every PR check and broken main (#553, fixed in
  #554). A line guarded some other way carries a `portability: ok` comment.


- **`RUNNER_CPU_I8` on a build with no int8 kernel is ignored, with a
  notice.** It used to switch the route on in name only: every matvec ran
  f32, while the route line and the `--score` and `--bench-json` route
  records said `cpu_act` `int8`, a route that never ran. That was so on the 1.11.0 macOS binary (see the entry on the
  macOS build above). `RUNNER_CPU_I8_PREFILL` already behaved this way.

- **The macOS build has the int8 dot path again.** Apple Silicon builds now
  target `-mcpu=apple-m1`, the floor of every Apple Silicon Mac, instead of
  `-march=native`. On a macOS virtual machine, which is where the release
  binaries are built, `-march=native` means generic armv8-a without the
  dot-product extension, so the 1.11.0 macOS binary was built without the
  int8 kernels: `RUNNER_CPU_I8` and `RUNNER_CPU_I8_PREFILL` fell back to the
  f32 route there without saying so. A local build on a Mac had them. CI and
  the release now check that the macOS binary takes the int8 route.

- **Local `$ref` and `$defs` in request schemas.** A `json_schema`
  response format or a tool's parameters may now reach nested definitions
  through local references, the way Pydantic's `model_json_schema()` and MCP
  tool definitions write them: `#/$defs/...`, `#/definitions/...` or any JSON
  pointer into the same schema is inlined at compile time and constrains
  exactly as its target. Until now every such schema was a 400 ("unsupported
  schema keyword '$defs'"). Refused by name: a reference outside the document,
  one that does not resolve, a recursive one, `$dynamicRef`, and a
  non-annotation keyword beside `$ref`. The prompt renders the tools as sent.
  A nested `anyOf` that cannot be dispatched is still refused, and its message
  now names `anyOf` rather than `oneOf` (R2.9.2).

- **Devstral Small 2 in the validated roster, with its tool-calling record.**
  unsloth's Q4_K_M and Q8_0 measured against Mistral AI's FP8 checkpoint
  dequantized exactly to float32 (every GGUF of the model requantizes that
  checkpoint, and the publisher serves it with static FP8 activations, so
  this is the reference arithmetic, not the publisher's serving arithmetic),
  300 positions: Q4_K_M 0.0223 mean KLD and 99.53% margin-qualified top-1,
  Q8_0 0.00126 and 100%; ggml-org's Q8_0 scores byte-identically to unsloth's.
  Agent torture on the Q4_K_M passes 100 of 100 cases (Tier 2, #551).

- **Agent torture v5: reasoning, then a native tool call.** A new case
  family, `reasoning_then_native_tool`, holds the thinking channel followed
  by a call in the family's native syntax: its tool carries no length,
  pattern or range constraint, so families whose function-XML cannot enforce
  those (Qwen3-Coder) are measured on the native path rather than the generic
  envelope. The default matrix is 126 cases (nine families of 14), and
  reports read `xyntetik.agent-torture.v5`; v4 results stay valid on their own
  terms. `scripts/agent-torture.py --timeout SECONDS` gives a slow host more
  than the harness's 120 s a request. On Qwen3.8-27B (CPU, qwen38 template,
  thinking on) the new case passes, the reasoning then the native call
  (#506).

- **CPU prefill in int8, opt-in.** With `RUNNER_CPU_I8_PREFILL=1`, CPU
  batches of 8 or more rows (prompt prefill, chunked scoring) quantize the
  activations to int8 once a matvec and run them against four-row int8 packs
  of the weights with the dot-product instructions (Q8_0, Q4_0, Q4_K and
  Q6_K, arm64 with the dot-product extension). It is measurably farther from
  the publisher's model than the f32 route, so it stays off by default: on
  Qwen3-0.6B Q8_0 against Qwen's float32 model over 2,000 positions, mean
  KLD 0.00747 against 0.00610 (1.22x, Wilcoxon p 5e-86), margin-qualified
  top-1 100% on both. On an Apple M1 at 4 threads, CPU prefill of 512 tokens
  went from 187 to 315 tok/s on that file and from 29.7 to 44.4 tok/s on
  Qwen3-4B Q4_K_M. The route line names it (`cpu_act` ends in
  `+int8-prefill`). No x86 kernel yet (R3.1.5 B).

- **Metal decode microbatch on the GEMM tiles, opt-in.** With
  `RUNNER_METAL_BATCH_TILES=1`, 6 or more `--parallel` slots decoding
  together run their weight products on the prefill GEMM tiles instead of
  the per-column identity matvec. It is not bit-identical to sequential
  decode, so it stays off by default, and the route line and the `--score`
  and `--bench-json` route records name it (`micro=tiles`); receipts do not
  carry the route yet. On an Apple M1 with K2-Horizon-1B, aggregate decode at
  8 slots went from 45.2 to 63.2 tok/s (Q4_K_M) and from 58.2 to 69.9 tok/s
  (Q8_0); at 2 to 5 slots the tiles were slower, so those widths keep the
  identity route. Against the publisher's float32 model over 2,000 positions
  it read 1.006x the identity route's mean KLD on the Q8_0 and 0.9996x on the
  Q4_K_M, not distinguishable on either
  (`docs/compat-reports/r3.24b-metal-microbatch-tiles-2026-11-02/`).
  `RUNNER_SCORE_MICRO=N` scores each
  position as one column of an N-wide microbatch, which is how the route is
  measured; on the default route it scores solo's bytes (R3.24 (b)).

<!-- Earlier releases continue in docs/changelog-archive.md. -->
