# R4.48.12 Pre-Registration: Three Release Claims for Inkling 2 Mini

Status: PRE-REGISTERED (committed before any claim run on Runner or llama.cpp executes).
Plan items: `R4.48.12` (parent), `R4.48.12.1` (this pre-registration).
Date: `2026-11-22`.

---

## 0. Scope, Fixed Artifacts, and Rules Against Post-Hoc Changes

### 0.1 Validated Model Files and Reference Artifacts
- **Primary validated GGUF (all three claims)**:
  - Hub repo & revision: `unsloth/Inkling-2-Mini-GGUF` at rev `45fd08764f2b0bf16b03a5483a80e5c7aaec8772`
  - Path: `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`
  - Size: `22,867,452,896` bytes (`sha256: 2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68`)
  - Verified bit-identical to the publisher's `thinkingmachines/Inkling-2-Mini` safetensors weights on all 744 text tensors (`R4.48.5`).
- **Secondary quantized GGUFs (Claim 1 quantized-step comparison per R4.48.12, labelled as quantized files)**:
  - `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` at rev `45fd08764f2b0bf16b03a5483a80e5c7aaec8772` (`17,629,760,736` bytes, `sha256: e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`).
  - **The 16 GB-tier file chosen by `R4.48.6`'s selection rule once Runner loads the model** (named strictly by rule here, never pre-picked before Runner's `--info fit` runs): from `unsloth/Inkling-2-Mini-GGUF` rev `45fd08764f2b0bf16b03a5483a80e5c7aaec8772`, whichever 16 GB candidate satisfies whole-model 8K fit `<= 15.0 GiB` by **both** `llama.cpp`'s tools' default AND Runner's `--info fit`, selecting the lower raw-read mean KL against the reference if both qualify (or the largest fitting context fallback specified in `R4.48.6` if neither fits at 8K).
- **Publisher float32 reference (Claim 1)**:
  - Checkpoint: `thinkingmachines/Inkling-2-Mini` at rev `b85c3cfa615fc413cc9af3379f491b963f3b586a` (the exact revision used by `R4.48.7a`; `config.json` differs from `1fb4134b` by the `dtype` declaration), `model-00001-of-00001.safetensors` (`19,526,019,168` bytes, `sha256: 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`).
  - Reference implementation, environment, host ISA, and dequantizing store (`R4.48.7a`): `huggingface/transformers` PR `51050` at `6173a2f679ff4ed393afc499aefc569da7e68bc1` (merged `2026-11-21` as `046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`; identical model code), run via `research/evaluation/galaxa2-2026-11-18/r448-step2a/read/ref_firstlight.py` in `--mode patched` (`Store` + `Patch`, matching `R4.48.7a`), under the `R4.48.7a` environment and **the lab's AVX-512 CPU host** (`<home>/venvs/gold-tf516/bin/python`, `torch 2.14.0+cpu`, `torch.use_deterministic_algorithms(True)`, `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`, `--threads 4`, `_attn_implementation = "eager"`, `_experts_implementation = "grouped_mm"`, no `modelopt` quantizer so all tensors dequantize/cast into `float32` at use, and without the publisher's `fouroversix_mae` activation quantization). The lab AVX-512 host is pinned explicitly because the float32 reference itself moves across a single instruction-set step (`11` of `2,000` top-1 flips observed across ISA steps on the `R4.48.7a` raw read).
    - In `--mode patched`, `Store` keeps `nvfp4` stacks as decoded `E2M1 * E4M3_scale` in `bfloat16` + per-expert `float32` factor, `mxfp8` stacks as `E4M3 * 2^(s - 127)` in `bfloat16`, `BF16` weights in `bfloat16`, and non-exact/`F32`/merged tensors and `e_score_correction_bias` buffers in 64-byte-aligned `float32` (`52.14 GB` resident storage); `Patch` replaces every parameter with a 4-byte placeholder and installs `forward_pre_hook` / `forward_hook` on each owner module (`24` decoder layers, `embed_tokens`, `norm`, `final_input`, `message_input`, `lm_head`) to materialize exact `float32` tensors only during that owner's forward (`4.20 GB` largest window at `model.layers.11`).
  - **Teacher-forced reference verification method (keeps the full `100` prompts x `256` tokens)**:
    - Why generation is not run on all `100 x 256`: across `R4.48.7a`'s two measured forwards at `4` CPU threads (`T=64` -> `19.88 s`; `T=2000` -> `~104-107 s`), every forward invocation pays `~17.0 s` of per-layer `Patch` upcasting (`52.14 GB -> 104.27 GB` `float32` across `24` owners) plus `~0.044-0.05 s` per token, making `100` prompts x up to `256` autoregressive decode steps take **`~71-121 hours`**.
    - **Teacher-forced equivalence rule**: For each engine's greedy output sequence `s = prompt_ids + gen_ids` (where `len(prompt_ids) = P` and `gen_ids = [g_0, ..., g_{L-1}]`, `1 <= L <= 256`, ending at `<|content_model_end_sampling|>` (`200006`) when `L < 256` or truncated at `256` tokens), run `ref_firstlight.py` (`--mode patched`, `use_cache=False`, one plain forward pass over `s`, zero KV caching or generation) and read the reference's logits row at each position `pos = P - 1 + j` (`0 <= j < L`, conditioned on prefix `s[:P + j]`).
      - *Mathematical equivalence*: By induction on `j = 0, 1, ..., L - 1`, the reference's own greedy generation produces **exact token sequence `gen_ids`** (including halting at `200006` when `L < 256` or at `L = 256`) **if and only if** the reference's greedy argmax equals `g_j` at every generated position `j` (taking exact top-2 ties as agreeing with either tied token, recorded below). Conversely, the smallest index `j` where `g_j` disagrees with the reference's argmax IS the **exact `first_diff` index** of true greedy divergence, yielding both the reference's true divergent token `ref_argmax(j)` and its top-1 minus top-2 logit margin at that position without generating a single token autoregressively.
      - *Exact ties*: At any position `pos = P - 1 + j` where the reference's top-2 float32 logits are bit-exact equal (`max_logit == second_logit`, i.e., any subset of `>= 2` tokens shares the exact maximum float32 logit), record that position as an exact tie, count `g_j` as **agreeing** if `g_j` achieves `max_logit`, and report the total count of exact-tie positions encountered per engine/file.
    - **Cost & memory estimate (`62 GB` cap, one model process at a time)**:
      - *Memory*: `Store` (`52.14 GB`) + largest owner window (`4.20 GB`) + single-forward activations at `T ~ 300-400` (`< 1.5 GB`) = **`~56-58 GB` peak RSS**, verified `<= 62 GB` (`R4.48.7a` measured `peak_rss_gb = 55.42 GB` at `T=64`).
      - *Time*: One `Store` build per persistent worker process (`~86-97 s`), then one plain forward per `(prompt, engine output)` of length `T ~ 300-400`: `~17.0 s + (0.05 s/tok * 350 tok) ~ 34.5 s` (upper bound `~37 s` at `T = 400`), or **`~1.0 hour` per engine per `100`-prompt run**: **`~2 hours`** for the exact `UD-Q8_K_XL` file (`Runner` + `llama.cpp`, fewer if both produce identical `s` and reuse the deterministic reference pass), plus **`~4 hours`** for the two secondary quantized files (`2` files x `2` engines), **`~6 hours` total**.
  - **Pre-run `3`-prompt control (`3 x 32` true greedy generation)**:
    - Before running full Claim 1 verification, select the first `3` primary prompts (`claim1-prompts.jsonl` indices `0, 1, 2` under `split == "primary"`), verify step-0 `generate(use_cache=True)` logits bit-equal `use_cache=False` `ref_firstlight.py` logits, and run **true greedy generation** on the reference for `max_new_tokens = 32` (`3 x 32 x ~17 s ~ 27 minutes`, under `1 hour`, on the same AVX-512 lab CPU). Verify on all `3` prompts for **both** `Runner` and `llama.cpp` (`UD-Q8_K_XL`, truncated to first `32` generated tokens) that the true `32`-token greedy output gives the **identical** byte-identity / `first_diff` / reference-token verdict as teacher-forced verification. Any discrepancy halts before Claim 1 proceeds.
  - Timing lock: Each engine's greedy `100`-prompt output set (`prompt_ids`, `gen_ids`, `raw_bytes` + `SHA256SUMS`) and the `3`-prompt reference control (`SHA256SUMS`) are committed as immutable records; `ref_firstlight.py` teacher-forced forwards then run deterministically on each engine sequence and commit their logits-argmax/margin verdict table + `SHA256SUMS`.
- **Comparison engines**:
  - `llama.cpp`: pinned commit `56eea9e4b0f5d125da16b75b551440b170aa8b1f`.
  - `Runner`: the commit closing `R4.48.6` (+ `R4.48.13` protocol bar), recorded by full 40-char SHA before runs start.
- **Hardware scope**:
  - Primary runs execute on the host CPU (available immediately after `R4.48.6`).
  - When `R4.48.9` opens CUDA support, CUDA rows execute the **exact same** pre-registered prompt files, settings, and metric definitions and are reported alongside CPU rows without re-selecting or pruning prompts.

### 0.2 Statistical Interval Method and Paired Comparison Rule
- **Per-engine / per-arm shares**: Every proportion (`k` successes out of `n` prompts) is reported with both exact count `k / n` and its **95% Wilson score confidence interval** (`z = 1.959963984540054`), `[l, u]` = `(p + z^2/(2n) +/- z * sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)`, where `p = k / n`.
- **Paired comparisons across engines or arms (Claims 1 and 3)**: Because all compared arms run on the exact same `n` prompts, form the `2 x 2` paired table `(n_11, n_10, n_01, n_00)` (`n_10`: arm A passes and arm B fails; `n_01`: arm B passes and arm A fails; `m = n_10 + n_01`; `delta = (n_10 - n_01) / n`). Report BOTH:
  1. **Newcombe (1998) Method 10 paired 95% confidence interval `[delta_lo, delta_hi]`** for `p_A - p_B`:
     - Let `[l_A, u_A]` and `[l_B, u_B]` be the two marginal 95% Wilson intervals, and `D = n_11*n_00 - n_10*n_01`, `M = (n_11+n_10)*(n_01+n_00)*(n_11+n_01)*(n_10+n_00)`.
     - Correlation term `phi`: `0` if `M == 0` or `0 <= D <= n/2`; `(D - n/2) / sqrt(M)` if `D > n/2`; `D / sqrt(M)` if `D < 0`.
     - `delta_lo = delta - sqrt((p_A - l_A)^2 - 2*phi*(p_A - l_A)*(u_B - p_B) + (u_B - p_B)^2)` and `delta_hi = delta + sqrt((u_A - p_A)^2 - 2*phi*(u_A - p_A)*(p_B - l_B) + (p_B - l_B)^2)`.
  2. **McNemar's exact two-sided `p`-value**: `1.0` if `m == 0`, else `min(1.0, 2.0 * sum_{i=0}^{min(n_10, n_01)} C(m, i) * 0.5^m)`.
- **Decision rule for "higher" / "lower" / "tie" wording**:
  - **"Higher" wording** is used **only** when `delta > 0` AND the paired Newcombe 95% CI `[delta_lo, delta_hi]` excludes `0` (`delta_lo > 0`).
  - **"Lower" wording** is used **only** when `delta < 0` AND `[delta_lo, delta_hi]` excludes `0` (`delta_hi < 0`).
  - Otherwise (`delta_lo <= 0 <= delta_hi`, interval includes `0`), the **tie / no distinguishable difference wording** is mandatory even if `K_A != K_B`.

### 0.3 Prompt Files Committed With This Plan
All prompt files sit directly beside this `PREREG.md` under `research/evaluation/r4412-claims-prereg-2026-11-22/`:

| File | Contents | Count | File SHA-256 |
|---|---|---:|---|
| `claim1-prompts.jsonl` | Claim 1 everyday prompts (`chat`, `code`, `arithmetic`, `short_tool_use`) | `100` primary + `20` reserves | `dadc40eb6844eb2549d8f85cb36215da6a7d4ca043caa598230946c06816d6fb` |
| `claim2-prompts.jsonl` | Claim 2 batch-invariance target prompts (`5` per category) | `20` primary + `4` reserves | `75c0fc40850cf0eff151528c5f622e8ae149469725f4083916dbd09106aefbaf` |
| `claim2-companions.jsonl` | Claim 2 fixed 8-prompt concurrent batch backdrop (`2` per category) | `8` fixed | `6e8c308414e964cf56f4a93c6ed5a2428b2fd69a31e1242639871c6d53c3c56d` |
| `claim3-prompts.jsonl` | Claim 3 Python code-debugging prompts (`plain_debug`, `tool_debug`) | `40` primary + `10` reserves | `9a647aeec2dd1479995874f530d28c95335c6b2320b099b8788a989fceb017e5` |

### 0.4 Public Sources, Licences, and Reproducible Selection Rule
Every prompt is drawn from public, ungated datasets under permissive licences (`MIT` or `Apache-2.0`) verified on each dataset's own repository/Hugging Face page, allowing commercial/non-commercial use and verbatim republication of prompts and generated outputs:

1. **Chat (`OpenAssistant/oasst1`)**:
   - Source: `https://huggingface.co/datasets/OpenAssistant/oasst1`, pinned revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, file `2023-04-12_oasst_prompts.messages.jsonl.gz`.
   - Licence: **Apache-2.0** (`README.md` YAML `license: apache-2.0` and root `LICENSE` file = Apache License Version 2.0). Allows reproduction and redistribution of prompts and outputs.
   - Eligibility filter: `lang == "en"`, `parent_id is None`, `role == "prompter"`, `review_result is True`, `deleted is False`, `synthetic is False`, `tree_state == "ready_for_export"`, `labels["not_appropriate"]["value"] == 0.0` (where present), `detoxify["toxicity"] < 0.05`, `20 <= len(text) <= 500`, and no `"<|"` / `"|>"` or ASCII control characters outside `\n\r\t`. (`3,767` eligible items.)
2. **Code (`openai/openai_humaneval`)**:
   - Source: `https://huggingface.co/datasets/openai/openai_humaneval` rev `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` (`openai_humaneval/test`) / `https://github.com/openai/human-eval` rev `b9453e8f1865df0a0057a1383767caa400acfbcf` (`data/HumanEval.jsonl.gz`).
   - Licence: **MIT** (`README.md` `"MIT License"`, GitHub `LICENSE` `spdx_id: MIT`). Allows verbatim use and republication.
   - Eligibility filter: `20 <= len(prompt) <= 1200`, no `"<|"` / `"|>"` or ASCII control characters outside `\n\r\t`. Prompt text wraps as `"Complete the following Python function cleanly:\n\n" + row["prompt"]`. (`162` eligible items.)
3. **Arithmetic (`openai/gsm8k`)**:
   - Source: `https://huggingface.co/datasets/openai/gsm8k` rev `740312add88f781978c0658806c59bc2815b9866` (`main/test`) / `https://github.com/openai/grade-school-math` rev `3101c7d5072418e28b9008a6636bde82a006892c` (`grade_school_math/data/test.jsonl`).
   - Licence: **MIT** (`README.md` `"The GSM8K dataset is licensed under the MIT License"`, `license: mit`). (Note: GitHub repo has no separate root `LICENSE` file; licence is explicitly stated in the HF dataset card README.)
   - Eligibility filter: `10 <= len(question) <= 600`, no `"<|"` / `"|>"` or non-printable control characters. (`1,319` test items checked.)
4. **Short Tool Use (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`)**:
   - Source: `https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard`, pinned revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`, files `BFCL_v3_live_simple.json` and `BFCL_v3_exec_simple.json`.
   - Licence: **Apache-2.0** (`README.md` YAML `license: apache-2.0`, `"All the models, and data used to train the models are released under Apache 2.0"`). Allows use and republication.
   - Eligibility & canonicalization filter: single-turn single-user message `15 <= len(content) <= 600`, single tool `len(function) == 1`, identifier name matching `^[A-Za-z_][A-Za-z0-9_]*$`, `1 <= len(properties) <= 5`, non-empty valid `required` subset, all types inside JSON Schema scalar/array primitives after applying BFCL's standard type mapping (`dict -> object`, `float -> number`, `bool -> boolean`, `tuple -> array`, no nested object schemas).
5. **Code Debugging for Claim 3 (`bigcode/humanevalpack`)**:
   - Source: `https://huggingface.co/datasets/bigcode/humanevalpack`, pinned revision `9a41762f73a8cb23bb5811b73d5aab164efcf378` (`python/test` split; upstream `https://github.com/bigcode-project/octopack` rev `e17a8f6470264286bc6a52eb8263582083bf3bf6`).
   - Licence: **MIT** (`README.md` YAML `license: mit`; GitHub `LICENSE` `spdx_id: MIT`).
   - Eligibility filter: exclude any task whose numeric HumanEval ID already appears in Claim 1 or Claim 2 `code` splits (ensuring zero task overlap between Claims 1/2 and Claim 3); `len(declaration + buggy_solution) <= 1600`; valid non-empty `buggy_solution` and `example_test`; no control tokens.

- **Checked and dropped**: `HuggingFaceTB/smoltalk` (rev `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`) reported `license: None` / empty license tags on the Hub API, so it was dropped in favor of `OpenAssistant/oasst1` (`Apache-2.0`).
- **Deterministic hash ordering**:
  - Within each Claim 1/2 pool (`oasst1`, `humaneval`, `gsm8k`, `bfcl`), items are sorted ascending by lowercase hex `sha256(source_id)`:
    - Ranks `[0:25]` -> Claim 1 primary (`25` per category = `100`).
    - Ranks `[25:30]` -> Claim 1 reserves (`5` per category = `20`).
    - Ranks `[30:35]` -> Claim 2 primary (`5` per category = `20`).
    - Ranks `[35:36]` -> Claim 2 reserves (`1` per category = `4`).
    - Ranks `[36:38]` -> Claim 2 fixed batch companions (`2` per category = `8`).
  - Within Claim 3 (`humanevalpack`), non-overlapping eligible items are sorted ascending by `sha256("claim3:" + source_id)`:
    - Ranks `[0:20]` -> `plain_debug` primary; `[20:25]` -> `plain_debug` reserves.
    - Ranks `[25:45]` -> `tool_debug` primary; `[45:50]` -> `tool_debug` reserves.

### 0.5 Reserve and Replacement Rule
- A primary item may be replaced by the next unused reserve of the **same category/subset** (in reserve rank order) **strictly BEFORE any reference or engine output/score exists** for that prompt, and **only** for mechanical pre-flight invalidity (for example, prompt + schema fails context limit or schema validation before any run begins).
- Once the reference or any engine generates a token for an item, that item stays in the primary set regardless of what it produces. Every replacement (if any) must be logged with its timestamp and reason before the first run starts.

---

## 1. Claim 1: Same Answer as the Reference

### 1.1 Prompts
- File: `claim1-prompts.jsonl` (`split == "primary"`, `n = 100`: `25` `chat`, `25` `code`, `25` `arithmetic`, `25` `short_tool_use`; `20` reserves under `split == "reserve"`).

### 1.2 Reference and Engine Settings
All three (`float32` teacher-forced patched reference, `Runner`, `llama.cpp`) run on the exact `100` primary prompts (`max_tokens = 256`), single-request, cold-context, non-batched greedy decoding:
- **Chat template**: Publisher `chat_template.jinja` (embedded in `UD-Q8_K_XL` GGUF; verified byte-identical).
- **Reasoning effort**: `"low"` (`0.2` in the template's `effort_map`; produces `<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>`), so everyday prompts frequently complete within `256` tokens while exercising the template effort preamble and thinking block.
  - Reference: `tokenizer.apply_chat_template(messages, tools=tools, reasoning_effort="low", tokenize=True, return_dict=True)`. Note: verify during pre-flight that Runner's rendered prompt token IDs match the reference's `apply_chat_template` IDs 100% before generation; any mismatch is a template bug recorded immediately before Claim 1 runs proceed.
- **Sampling (greedy everywhere)**: `temperature = 0.0`, `top_k = 0`, `top_p = 1.0`, `min_p = 0.0`, `repeat_penalty = 1.0`.
- **Budgets & context**: `max_tokens = 256` new tokens, `n_ctx = 4096`, `ignore_eos = false`, stop on `<|content_model_end_sampling|>` (`id = 200006`).
- **Threads / device**: `16` threads on CPU (`--gpu off` / `-ngl 0`) for both `Runner` and `llama.cpp`; `4` threads (`OMP_NUM_THREADS=4`) on the **lab AVX-512 CPU** for `ref_firstlight.py` per `R4.48.7a` (`ref_firstlight.py:880`), no prefix caching across prompts (each prompt runs cold).
- **Engine-specific flags**:
  - **Runner (`runner --serve` + `/v1/chat/completions` or `--record` CLI)**:
    `--gpu off -t 16 -c 4096 --parallel 1 --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 --reasoning-budget 0` (`--loop-guard` OFF). Request body carries `messages`, `tools` (for `short_tool_use`, `tool_choice: "auto"`), `max_tokens: 256`, `temperature: 0`, `reasoning_effort: "low"`.
  - **llama.cpp (`llama-server` / `llama-completion` at `56eea9e4`)**:
    `-ngl 0 -t 16 -c 4096 -np 1 --jinja --reasoning-effort low --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 --reasoning-budget -1` (`-n 256`).
- **Secondary quantized files (Claim 1 only)**:
  - Repeat the identical `100`-prompt `Runner` and `llama.cpp` runs (teacher-forced against the same pinned AVX-512 `float32` reference) on `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` and whichever 16 GB-tier file `R4.48.6`'s selection rule admits once Runner loads the model (using that admitted file's admitted context window, with prompt lengths verified compatible beforehand), reported in a separate `(Quantized files)` table so exact-file vs quantized step-down losses are never conflated.

### 1.3 Metrics (Claim 1)
Comparisons operate on the **complete raw model generation** per prompt over all `100` primary prompts (`<= 256` tokens, inclusive of thinking + answer/tool blocks, stopping on `<|content_model_end_sampling|>` (`200006`) or `256` tokens), evaluated via Section 0.1's teacher-forced forward over each engine's output `s = prompt_ids + gen_ids`:
1. **Byte-identical / reference-equivalent answer count & rate**: number `k / 100` (and `k_c / 25` per category) where at **every** generated position `0 <= j < L`, `g_j` equals the reference's teacher-forced argmax at `P - 1 + j` (or achieves the exact max float32 logit on a recorded exact-tie position) AND either `(L < 256 and g_{L-1} == 200006)` or `L == 256` (so by induction the reference's true greedy generation produces the identical `gen_ids` and raw UTF-8 bytes; any prompt/token encoding discrepancy or early stop without `200006` fails and is logged explicitly). Report marginal 95% Wilson score intervals, exact-tie position counts, plus paired `(n_11, n_10, n_01, n_00)`, Newcombe Method 10 paired 95% CI `[delta_lo, delta_hi]` for `p_Runner - p_llamacpp`, and exact two-sided McNemar `p`-value.
2. **First differing token distribution**: for every non-identical prompt, define `first_diff` as the smallest `0`-based generated token index `j` where `g_j` disagrees with the reference's teacher-forced argmax (and is not an exact max-logit tie), recording `ref_argmax(j)` and the reference's top-1 minus top-2 logit margin at `P - 1 + j` (`logits[top1] - logits[top2]`). Report:
   - `min`, `p25`, `median`, `p75`, `max` of `first_diff`, plus median/IQR of the reference margin at `first_diff`;
   - bucket counts at `j == 0`, `1 <= j <= 15`, `16 <= j <= 63`, and `64 <= j <= 255`.

### 1.4 Claim Wording for Every Outcome (Claim 1)
All tables state `Runner`, `llama.cpp`, both 95% Wilson intervals, exact-tie counts, the Newcombe paired 95% CI `[delta_lo, delta_hi]`, and exact McNemar `p` side by side per Section 0.2:
- **Runner higher (`delta > 0` AND Newcombe paired 95% CI excludes `0`, i.e., `delta_lo > 0`)**:
  `"On 100 pre-registered everyday prompts (greedy, max 256 tokens, reasoning_effort=low) using the exact UD-Q8_K_XL GGUF verified by teacher-forced forward against the float32 transformers reference, Runner matched the reference's complete greedy generation on more prompts than llama.cpp: K_R/100 (95% Wilson [lo_R, hi_R]) vs K_L/100 (95% Wilson [lo_L, hi_L]; paired difference delta [Newcombe 95% CI delta_lo, delta_hi], McNemar p=P, exact ties T_E); first token divergence fell at median index M_R vs M_L."`
- **Tie (`delta_lo <= 0 <= delta_hi`, paired Newcombe 95% CI includes `0`)**:
  `"On 100 pre-registered everyday prompts (greedy, max 256 tokens, reasoning_effort=low) using UD-Q8_K_XL verified by teacher-forced forward against the float32 reference, Runner matched the reference on K_R/100 (95% Wilson [lo_R, hi_R]) compared with llama.cpp at K_L/100 (95% Wilson [lo_L, hi_L]; paired difference delta [Newcombe 95% CI delta_lo, delta_hi], McNemar p=P, no statistically distinguishable difference, exact ties T_E); first token divergence fell at median index M_R vs M_L."`
- **Runner lower (`delta < 0` AND Newcombe paired 95% CI excludes `0`, i.e., `delta_hi < 0`)**:
  `"On 100 pre-registered everyday prompts (greedy, max 256 tokens, reasoning_effort=low) using UD-Q8_K_XL verified by teacher-forced forward against the float32 reference, Runner matched the float32 reference on fewer prompts than llama.cpp: K_R/100 (95% Wilson [lo_R, hi_R]) vs K_L/100 (95% Wilson [lo_L, hi_L]; paired difference delta [Newcombe 95% CI delta_lo, delta_hi], McNemar p=P, exact ties T_E); first divergence occurred at median token index M_R (Runner) vs M_L (llama.cpp)."`
- Secondary quantized files (`UD-Q4_K_XL` and whichever 16 GB-tier file `R4.48.6` admits by rule) carry the explicit label: `"(Quantized file: measured against the unquantized float32 reference to show quantization step-down cost)"`.

---

## 2. Claim 2: The Same Answer Alone or Batched

> **Architectural Limit Leading Claim 2 (Runner Sequential Decode on MoE / CPU)**:
> Per `MANUAL.md:2685` (`--parallel N`: `"Dense models only; MoE/recurrent families decode sequentially."`), `src/model.c:16615-16620, 16655-16699` (on CPU `seqs[i]->gpu == NULL`, so `model_batch_create` leaves `b->gb = NULL` and `model_batch_decode` falls through to the sequential loop `model_forward(b->seqs[idx[i]], tok[i], pos[i])`), and `src/server.c:4051-4053` (slots share one `tpool` via serialized `tpool_run`), Runner decodes concurrent requests of this MoE model **sequentially, one decode step at a time**, with no microbatched CPU/MoE decode kernel. Consequently, Runner's alone-equals-batched identity on `Inkling-2-Mini` comes from serving concurrent slots one decode at a time (interleaved slot turns under shared weights and thread pool), rather than co-batching them into shared matrix multiplications. Both engines therefore record telemetry/log proof of whether requests were actually co-batched and aggregate batched throughput (`tok/s` across the 9 concurrent requests) alongside the byte-identity counts.

### 2.1 Prompts
- Target set: `claim2-prompts.jsonl` (`split == "primary"`, `n = 20`: `5 chat`, `5 code`, `5 arithmetic`, `5 short_tool_use`; `4` reserves under `split == "reserve"`).
- Batch companion set: `claim2-companions.jsonl` (`8` fixed prompts: `2` per category, disjoint from all Claim 1 & Claim 2 primaries).

### 2.2 Engine Settings, Context Verification, and Matrix
Shared parameters per request: publisher chat template, `reasoning_effort = "low"`, `temperature = 0.0`, `top_k = 0`, `top_p = 1.0`, `min_p = 0.0`, `repeat_penalty = 1.0`, `max_tokens = 256`, per-sequence context `4096` tokens, `--gpu off` / `-ngl 0`, cold prefix (clear/disable prefix reuse between measurement rounds or fresh server per cell so slot KV cache does not leak across rounds).

- **Verified per-sequence context parity (`4096` tokens/slot across `9` slots)**:
  - **Runner (`--parallel 9 -c 4096`)**: verified in `MANUAL.md:2700, 4056`, `src/main.c:1707-1709, 1764, 1786-1788` (`"--parallel N: every slot pays its own KV cache... slots N (--parallel): the weights once, a KV cache per slot"`, pricing `f.kv_f16 * slots`), `src/server.c:4055-4057, 4099-4122` (every slot `i` loads its own `model_t` with `slot_mp` inheriting `n_ctx = 4096` and `slot_engine_init(s)`), and `src/model.c:6255-6262, 6309, 6347` (`kv_cache_bytes(...) * slots`). Thus `-c 4096` sets **`4,096` tokens per slot** (`36,864` total KV positions across `9` slots).
  - **`llama.cpp` (`llama-server -np 9 -c 36864`)**: in `llama-server` at `56eea9e4`, `-c` sizes the total KV pool divided across `-np` slots (`tools/server/server-context.cpp:1508, 1553`: `slot.n_ctx = n_ctx_slot()`, `common/arg.cpp:1632-1639`). Setting `-np 9 -c 36864` gives `n_ctx_slot = 36864 / 9 = 4096` tokens per slot (verified from `llama-server`'s startup log line `n_slots = 9, n_ctx_slot = 4096`).
- **Batching protocol**:
  - `alone`: submit 1 request (`target_i`) to the server running with `--parallel 9` (`-np 9`) and wait for completion (other 8 slots idle).
  - `batched`: submit `9` concurrent HTTP requests (`target_i` plus all `8` fixed prompts in `claim2-companions.jsonl`, released together via a single thread barrier so all `9` sequences overlap in the server queue/slots).
- **Runner matrix (`4` cells per prompt)**:
  - `t in {1, 16}` x `mode in {alone, batched}` using:
    `runner -m ... --serve --gpu off -t <t> --parallel 9 -c 4096 --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 --reasoning-budget 0` (env `RUNNER_DEBUG_ROUTE=1` for route logging; `/metrics` and `/v1/runner/provenance` captured per round).
- **llama.cpp matrix (`16` cells per prompt)**:
  - `t in {1, 16}` x `mode in {alone, batched}` x `fa in {on, off}` (`-fa on`, `-fa off`) x `ubatch in {1, 512}` (`-ub 1`, `-ub 512`) using:
    `llama-server -m ... -ngl 0 -t <t> -np 9 -c 36864 -fa <fa> -ub <ubatch> --jinja --reasoning-effort low --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 --reasoning-budget -1` (capturing `llama-server` stderr/slot logs and per-response timings).

### 2.3 Metrics (Claim 2)
For each engine and each of the `20` target prompts:
1. **Across-all-settings byte identity share**: `1` iff the raw generated UTF-8 byte string (and generated token ID sequence up to EOS or `256` tokens) is **identical across every cell tested for that engine** (`4/4` cells for Runner; `16/16` cells for llama.cpp), `0` otherwise. Reported as `k / 20` with 95% Wilson score interval.
2. **Factor attribution counts** (out of `20`, holding other factors at reference cell `alone, t=16, fa=on, ub=512`):
   - `alone` vs `batched` differences (at `t=1` and `t=16`);
   - `t=1` vs `t=16` differences (alone and batched);
   - for `llama.cpp`: `-fa on` vs `-fa off`, and `-ub 1` vs `-ub 512`.
3. **Co-batching verification (per engine, from logs/telemetry)**:
   - **Runner**: record `RP_MICRO` / `profile.route.micro` (`src/model.c:8139, 8312, 16669`), `RUNNER_DEBUG_ROUTE=1` `path=` lines (`src/model.c:8148-8171`), and `/metrics` counters `runner_batch_steps_total`, `runner_batch_sequences_total` (`src/server.c:2541-2544`). State explicitly whether any hardware/kernel co-batching occurred (`co_batched = false`, sequential decode per `MANUAL.md:2685` and `src/model.c:16658-16699`).
   - **`llama.cpp`**: inspect `llama-server` stderr / `update_slots` logs (`tools/server/server-context.cpp:1609, 3500`) during `batched` rounds to record whether active slots decoded together in multi-sequence `llama_decode` batches (`co_batched = true` with observed mean/max active sequences per decode batch, or `false`).
4. **Aggregate throughput in `batched` mode (and `alone` comparison, reported beside identity counts)**:
   - For each of the `20` `batched` rounds at `t=16` (and separately `t=1`), compute **aggregate 9-request throughput**: `agg_tok_s = sum_{j=1..9}(completion_tokens_j) / T_batch_wall_s`, where `T_batch_wall_s` is the wall-clock elapsed time from barrier release of the 9 requests until the 9th response completes. Report median and IQR across the `20` rounds alongside `alone` single-request `tok/s` (so the throughput cost of sequential decode vs co-batched decode is explicit).
5. **Runner defect rule (`R4.48.12`)**: Runner's design claim is strict invariance (`20/20`). Any non-identical output across `(alone/batched, t=1/16)` on Runner is treated and logged as a **Runner defect**, not as an acceptable baseline variance.

### 2.4 Claim Wording for Every Outcome (Claim 2)
Every Claim 2 wording leads plainly with the MoE sequential-decode fact on Runner and cites the co-batching status and aggregate `batched` throughput at `t=16` (`R_batched`, `L_batched` tok/s) alongside `alone` throughput (`R_alone`, `L_alone` tok/s):
- **Runner `20/20` invariant and llama.cpp `< 20/20`**:
  `"Across 20 pre-registered prompts tested alone and inside a 9-request batch at 1 and 16 threads (greedy, UD-Q8_K_XL on CPU), Runner serves concurrent MoE requests sequentially one decode at a time (co-batched: false; batched aggregate R_batched tok/s vs alone R_alone tok/s at t=16) and produced byte-identical output across all 4 settings on 20/20 prompts (95% Wilson [0.839, 1.000]), whereas llama.cpp (varying flash attention on/off and n_ubatch 1/512 across 16 settings; co-batched: CB_L, batched aggregate L_batched tok/s vs alone L_alone tok/s) stayed byte-identical on K_L/20 prompts (95% Wilson [lo_L, hi_L])."`
- **Runner `20/20` invariant and llama.cpp `20/20` invariant**:
  `"Across 20 pre-registered prompts on CPU (UD-Q8_K_XL, greedy), Runner serves concurrent MoE requests sequentially one decode at a time (co-batched: false; batched aggregate R_batched tok/s vs alone R_alone tok/s) and held byte identity across all 4 settings on 20/20 prompts (95% Wilson [0.839, 1.000]); llama.cpp (16 settings adding flash-attention and n_ubatch toggles; co-batched: CB_L, batched aggregate L_batched tok/s vs alone L_alone tok/s) also held byte identity on 20/20 prompts (95% Wilson [0.839, 1.000])."`
- **Runner `< 20/20` (Runner is not invariant: defect observed)**:
  `"On CPU (UD-Q8_K_XL, greedy), even though Runner serves concurrent MoE requests sequentially one decode at a time (co-batched: false; batched aggregate R_batched tok/s vs alone R_alone tok/s), Runner failed full batch/thread invariance, staying byte-identical across all 4 settings on K_R/20 prompts (95% Wilson [lo_R, hi_R]) vs K_L/20 for llama.cpp (95% Wilson [lo_L, hi_L]; co-batched: CB_L, L_batched tok/s); the remaining 20 - K_R Runner prompts diverged between <failing factor> and are logged as an open Runner invariance defect."`

---

## 3. Claim 3: The Model's Own Reasoning Loops

### 3.1 Prompts
- File: `claim3-prompts.jsonl` (`split == "primary"`, `n = 40`: `20` `plain_debug` + `20` `tool_debug`; plus `10` reserves `5 + 5` under `split == "reserve"`), sourced from `bigcode/humanevalpack` (`python/test`, MIT License), non-overlapping with Claims 1 and 2.

### 3.2 Settings
- **Template & reasoning effort**: publisher chat template with `reasoning_effort = "high"` (`0.9`, the template's default, where long reasoning chains occur on debugging tasks).
- **Sampling**: greedy (`temperature = 0.0`, `top_k = 0`, `top_p = 1.0`, `min_p = 0.0`, `repeat_penalty = 1.0`).
- **Context & token budget**: `n_ctx = 4096`, `max_tokens = 2048` per turn, `16` CPU threads (`--gpu off` / `-ngl 0`), single slot (`--parallel 1`), cold context per prompt, `ignore_eos = false`.
- **Engine configurations (5 pre-registered arms; the repeating-span `--loop-guard` is Runner-only, as `llama.cpp` `56eea9e4` provides `--reasoning-budget` and `--reasoning-effort` but has no repeating-span loop guard)**:
  - Common Runner server flags: `--gpu off -t 16 --parallel 1 -c 4096 --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1` (request: `temperature=0`, `max_tokens=2048`, `reasoning_effort="high"`).
  - Common `llama.cpp` flags: `-ngl 0 -t 16 -np 1 -c 4096 --jinja --reasoning-effort high --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 -n 2048`.
  1. **Runner unprotected (`runner-off`)**: common Runner flags + `--reasoning-budget 0` (`--loop-guard` OFF).
  2. **Runner loop-guard only (`runner-lg`)**: common Runner flags + `--loop-guard` (defaults: `loop_guard_span=8`, `loop_guard_repeats=3`, `loop_guard_window=256`, `loop_guard_max_closes=3`, `loop_guard_everywhere=false`) + `--reasoning-budget 0`. (Isolates the repeating-span loop guard without budget forcing.)
  3. **Runner loop-guard + reasoning-budget (`runner-lg-rb`)**: common Runner flags + `--loop-guard` (same defaults) + `--reasoning-budget 1024` (default `--reasoning-budget-message`). (Reserves `1024` tokens of the `2048` turn budget for answer / tool emission.)
  4. **llama.cpp unrestricted (`llama-off`)**: common `llama.cpp` flags + `--reasoning-budget -1`.
  5. **llama.cpp budgeted (`llama-rb`)**: common `llama.cpp` flags + `--reasoning-budget 1024 --reasoning-budget-message ""` (empty string / none is `llama.cpp`'s default at `56eea9e4`: `common/arg.cpp:3790-3796`, `"default: none"`, and `common/sampling.h:300`, default-constructed `std::string reasoning_budget_message`).

### 3.3 Exact Definitions of Metrics (Fixed Beforehand)
Every turn records raw generated token IDs, raw detokenized text, parsed reasoning/content/`tool_calls`, `finish_reason`, `runner_telemetry` (when Runner), and token counts (`completion_tokens`, `reasoning_tokens`, `content_or_tool_tokens`).

1. **Loop definition & loop rate (`k_loop / 40`, 95% Wilson interval)**:
   - Let `R` be the concatenation of generated token IDs emitted while inside `<|content_thinking|>...<|end_message|>` spans (before `<|end_message|>` closes each span, or up to turn end if never closed). Let `G` be the full generated token ID sequence.
   - **`span_repeat_loop(R)`**: true iff there exists any index `i >= 0` in `R` where an `8`-token block repeats `3` times contiguously: `R[i:i+8] == R[i+8:i+16] == R[i+16:i+24]` (matching Runner's default `span=8, repeats=3` within `window=256`).
   - **`unclosed_thinking_exhaustion`**: true iff `completion_tokens == 2048` (`finish_reason == "length"`) AND the turn ends with an unclosed `<|content_thinking|>` block (zero characters of post-thinking visible answer or tool call).
   - **Primary `loop` indicator**: `span_repeat_loop(R) OR unclosed_thinking_exhaustion`.
   - **Secondary diagnostics (reported separately, not merged into primary `loop`)**:
     - `whole_turn_span_loop`: `span_repeat_loop(G)` across full output;
     - `general_periodic_loop(R)`: any period `p` in `1..128` repeated at least `max(3, ceil(24 / p))` times contiguously (`>= 24` consecutive repeating tokens) in `R`;
     - `runner_lg_interventions`: turns where `runner_telemetry.loop_guard.closes > 0` and turns ending with `finish_reason == "loop"`.
2. **Usable answer or parseable tool call inside token budget (`k_usable / 40`, 95% Wilson interval)**:
   - **`plain_debug` (`20` prompts)**: true iff ALL hold:
     1. Turn terminated cleanly with `<|content_model_end_sampling|>` inside `completion_tokens <= 2048` (`finish_reason == "stop"`, not `length` or `loop`).
     2. Outside any `<|content_thinking|>...<|end_message|>` block, there is non-empty `<|content_text|>...<|end_message|>` output containing either a ```python code block with `def <entry_point>` or a plain-text `def <entry_point>(` definition (not stubbed `pass` only).
   - **`tool_debug` (`20` prompts)**: true iff EITHER:
     1. Tool call path: a parsed tool call `submit_patch` is returned whose `arguments` JSON object parses cleanly with non-empty string `fixed_code` (containing `def <entry_point>(`) and non-empty string `root_cause` (with `finish_reason` in `{"tool_calls", "stop"}`, or on Runner also separately tabulating `finish_reason == "length"` truncation-recovered calls that still satisfy the schema; primary count includes only `<= 2048` non-truncated completions `finish_reason in {"tool_calls", "stop"}`, while `with_truncation_recovery` is reported in a clearly labelled companion row!).
     2. Direct answer fallback (`tool_choice: "auto"`): the model terminated cleanly (`finish_reason == "stop"`, `<= 2048`) with a usable `def <entry_point>(` code answer as in `plain_debug`.
   - **Secondary functional diagnostic (`unit_test_pass`, reported separately)**: execute extracted `fixed_code` + `bigcode/humanevalpack`'s `test` snippet inside an isolated subprocess with a `5 s` timeout per task (`0` influence on `k_usable`).
3. **Tokens spent**:
   - Report per arm over all `40` prompts (and by subset `plain_debug` / `tool_debug`): total completion tokens summed over `40` turns, plus mean, median, IQR (`p25..p75`), `p90`, max, and split into `reasoning_tokens` vs `output_tokens` (`completion_tokens - reasoning_tokens`).

### 3.4 Claim Wording for Every Outcome (Claim 3)
Every report presents all 5 arms (`runner-off`, `runner-lg`, `runner-lg-rb`, `llama-off`, `llama-rb`) symmetrically, with each arm's 95% Wilson interval for `k_usable / 40` and `k_loop / 40`, plus the paired `2 x 2` Newcombe Method 10 95% CI `[delta_lo, delta_hi]` and exact two-sided McNemar `p` comparing `runner-lg-rb` pairwise against each of the three baseline/comparator arms (`runner-off`, `llama-off`, `llama-rb`) on `k_usable / 40`:
- **Runner guarded beats unprotected & llama.cpp (`delta > 0` AND Newcombe paired 95% CI excludes `0`, i.e., `delta_lo > 0`, against EVERY one of `runner-off`, `llama-off`, `llama-rb`)**:
  `"On 40 code-debugging prompts (greedy, max 2048 tokens, reasoning_effort=high, UD-Q8_K_XL), Runner with --loop-guard and --reasoning-budget 1024 delivered more usable answers or parseable tool calls than every comparator: K_G/40 turns (95% Wilson [lo_G, hi_G], loop rate L_G/40, median T_G tokens), vs K_U/40 (loop L_U/40, median T_U tokens) for unguarded Runner, K_LO/40 (loop L_LO/40, median T_LO tokens) for unrestricted llama.cpp, and K_LB/40 (loop L_LB/40, median T_LB tokens) for budgeted llama.cpp (every pairwise Newcombe 95% CI excludes 0; McNemar p <= P_max)."`
- **Tie / no clear difference (paired Newcombe 95% CI includes `0` against at least one comparator, with none showing `runner-lg-rb` strictly worse)**:
  `"On 40 code-debugging prompts (UD-Q8_K_XL, greedy, reasoning_effort=high), guarded Runner yielded K_G/40 usable turns (95% Wilson [lo_G, hi_G], loop L_G/40, median T_G tokens) vs unguarded Runner K_U/40 (L_U/40, T_U tokens) and llama.cpp K_LO/40 unrestricted / K_LB/40 budgeted, with no statistically distinguishable paired difference on usable turns against comparator(s) [names] (paired Newcombe 95% CI includes 0); the loop guard fired on F/40 turns."`
- **Runner guarded worse (`delta < 0` AND Newcombe paired 95% CI excludes `0`, i.e., `delta_hi < 0`, against any of `runner-off`, `llama-off`, `llama-rb`)**:
  `"On 40 code-debugging prompts (UD-Q8_K_XL, greedy, reasoning_effort=high), Runner with --loop-guard and --reasoning-budget 1024 finished fewer usable turns than [comparator]: K_G/40 (95% Wilson [lo_G, hi_G], loop L_G/40, median T_G tokens) vs unguarded Runner K_U/40 and llama.cpp K_LO/40 unrestricted / K_LB/40 budgeted (Newcombe 95% CI [delta_lo, delta_hi] excludes 0, McNemar p=P); the guarded arms truncated/closed F/40 turns before reaching a usable fix."`

---

## 4. Out of Scope and Void Conditions

### 4.1 Out of Scope
- Multimodal image/vision inputs (`mmproj` is not loaded in Runner for Inkling 2 Mini under `R4.48.6`).
- Non-greedy sampling (`temperature > 0`, `--reasoning-temp`, `--repeat-penalty != 1.0`, `top_k != 0`, `top_p < 1.0`).
- Tuning `--loop-guard` thresholds (`span`, `repeats`, `window`, `max_closes`) or `--reasoning-budget` values (`1024`) after any Claim 3 output exists.
- `tool_choice: "required"` enforcement tests, or comparing against the publisher's quantized `fouroversix_mae` runtime.

### 4.2 Conditions That Void a Run
Any of the following voids the affected run cleanly (the failure is recorded in the run log; no partial results are merged):
1. Checksum mismatch on any GGUF file, safetensors file, prompt JSONL file, or committed Claim 1 reference `SHA256SUMS`.
2. Wrong upstream commit (`llama.cpp != 56eea9e4b0f5d125da16b75b551440b170aa8b1f`, `transformers != PR 51050`, or `thinkingmachines/Inkling-2-Mini != b85c3cfa615fc413cc9af3379f491b963f3b586a`), wrong reference host ISA (must run on the `R4.48.7a` AVX-512 lab CPU), or uncommitted dirty Runner binary without recorded provenance.
3. Sampling/template parity violation (e.g., GGUF `top_k` / `repeat_penalty` leaking in, `apply_chat_template` vs Runner prompt IDs differing, `3`-prompt true-greedy control disagreeing with teacher-forced verification verdicts or step-0 logits failing `use_cache=False` equality, or peak RSS exceeding `62 GB`).
4. **Mid-run engine defect**: if a crash, NaN/Inf, wrong stop-token halt, or batching race occurs mid-run, **halt and report the defect directly**; never patch around a defect inside an existing pre-registered run. Any fix starts a fresh, separately dated full run over the entire claim matrix.
