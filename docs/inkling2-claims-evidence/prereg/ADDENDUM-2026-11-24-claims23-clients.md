# R4.48.12 pre-registration: Claims 2 and 3, the clients and one definition (2026-11-24, BEFORE any Claim 2 or 3 output)

Written at 2026-11-24T18:41Z. No counted Claim 2 or Claim 3 round has run, so these items fix the instruments before any output exists. Only uncounted smokes ran (2 prompts each, max_tokens 32, host load 64 to 96, their timings meaningless), and they served only to test the mechanics. No setting, prompt, metric, interval method, decision rule or claim wording in PREREG.md changes.

## 1. The clients, pinned by sha256

| Client | sha256 | Used for |
|---|---|---|
| `tools/claim2_client.py` | `be18bc017a067e705248901a5a29eed1f995e4ce3d74660b450c8c13ff5be374` | Claim 2, both engines |
| `tools/claim3_client_runner.py` | `0ea9680979d71be1bec7e3387bae16f8f103cf766fecd5fff56202bb6a40bd11` | Claim 3, Runner's arms |
| `tools/claim_client_v2.py` (unchanged) | `b5ce513d...` | Claim 1 and R4.48.24 only |
| `claim_client_llama.py` (unchanged, the lab's) | as for Claim 1 | Claim 3, llama.cpp's arms |

**claim2_client.py** (PREREG 2.2 and 2.3):
- Each request carries Claim 1's body plus `cache_prompt: false`.
- **Alone:** one request.
- **Batched:**
  - the target and the 8 companions, one thread each, released by one `threading.Barrier(9)`;
  - the barrier's action runs once, in the ninth arriving thread, before any thread is released, so the time it records is the release instant;
  - `T_batch_wall_s` runs from that release to the ninth response's end (PREREG 2.3(4)), and `agg_tok_s` is the summed completion tokens over `T_batch_wall_s`.
- **Ids:**
  - Runner's come from its `--record` receipts, with `claim_client_v2.py`'s `with_stop`;
  - llama.cpp's come from `__verbose.tokens`, and its prompt ids from `/apply-template` and `/tokenize`, read before the barrier.
- **Telemetry, every round (PREREG 2.3(3)):**
  - Runner: `/metrics` and `/v1/runner/prefix-cache` before and after, and `/v1/runner/provenance` after;
  - llama.cpp: `/slots` and `/metrics` before and after, where served.
- **Two fixes made at review, before counting:**
  - Runner's prefix-cache clear takes no request body, so the client now sends an empty POST.
  - A resume counts a round as done only when its line says `complete == true`. The statistics take the last complete line per id and report the number of incomplete rounds. An engine defect still halts the run and voids it (PREREG 4.2(4)).

**Runner's cold rule.**
- Runner ignores `cache_prompt` (main has no such request field), so its cold rule rests on the prefix-cache clear before every round.
- Within one batched round the slots may reuse each other's cached template prefix. PREREG 2.2 forbids reuse only across rounds. The before and after `/v1/runner/prefix-cache` bodies record what happened, and the run log states it.

**claim3_client_runner.py** is `claim_client_v2.py` verbatim plus one thing, PREREG 3.2's cold context per prompt:
- a `POST /v1/runner/prefix-cache/clear` (no body) before each prompt, its reply kept as `cold`;
- `client_version` is `"2-claim3-cold"`.

Nothing in `claim_client_v2.py` enforced a cold context per prompt, and the smoke's `cached_tokens` was 0 only because no earlier prompt shared the template prefix. llama.cpp's arms keep `claim_client_llama.py`, which sends `cache_prompt: false`.

## 2. Claim 3: reasoning tokens, one definition for every arm

Neither engine's `usage` carries `reasoning_tokens`: both smokes return only `completion_tokens`, `prompt_tokens`, `total_tokens` and `prompt_tokens_details`. So PREREG 3.3's `reasoning_tokens` is computed the same way for all five arms, from each turn's generated token ids:
- `reasoning_tokens` is the number of generated ids inside `<|content_thinking|>` ... `<|end_message|>` spans. These are the ids of R in PREREG 3.3.1, counted the same way: up to the turn's end when a span never closes.
- `output_tokens = completion_tokens - reasoning_tokens`.
- `completion_tokens` is the generated id count. Where an engine's `usage.completion_tokens` differs from that count, both are reported and the id count is used.

## 3. Before the first counted round

- The pre-flights run for both engines against `preflight/ref_ids.jsonl` (PREREG 4.2(3)): Claim 2's 20 targets and 8 companions at effort low, and Claim 3's 40 primaries at effort high. Runner's run on the counted build.
  - llama.cpp's passed at 18:41Z on 56eea9e4, each set equal to `ref_ids.jsonl` (`e6e2d0ae`): claim2-prompts 20 / 20, claim2-companions 8 / 8 (Claim 2's server flags) and claim3-prompts 40 / 40 (Claim 3's). The lab's record is in `results/r4412/preflight-c23-llama/`.
- Runner's counted cells run on the build R4.48.24 run 2 decides (R4.48.12.2), not on the smoke's `f3ff8e84` (which predates the R4.48.25 tanh fix). They run with:
  - PREREG 2.2's full flags, `--reasoning-budget 0` included for Claim 2;
  - `RUNNER_DEBUG_ROUTE=1`, its stderr kept;
  - `--record` named in the run log.
