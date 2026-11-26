# Inkling 2 Mini: three pre-registered claims, the evidence

Runner was compared with llama.cpp and with Unsloth's published llama.cpp build on Inkling 2 Mini, on three claims written down before any engine ran.

This folder holds:
- the pre-registration and every later change to it;
- the prompts;
- the statistics outputs;
- the scripts.

Every per-prompt record is in the release attachment [`inkling2-claims-records-1.22.0.tar.gz`](https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/tag/v1.22.0) (sha256 `8aa2cfb600fb63cd08eb68af844287f9160f7039a24cfa04e839dc416a7748cc`): every engine's answers, the teacher-forced verdicts, the Claim 2 cells, the Claim 3 arms, receipts, load tables and run scripts.

## What ran

| | |
|---|---|
| Model file | `unsloth/Inkling-2-Mini-GGUF` at `45fd08764f2b0bf16b03a5483a80e5c7aaec8772`, `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`, 22,867,452,896 bytes, sha256 `2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68`. All three engines read these bytes. |
| Reference | The publisher's float32 model through `huggingface/transformers` PR 51050 at `6173a2f6` (see `prereg/PREREG.md` 0.1) |
| Runner | **Claim 1:** the reference route (serial float32), main `f0d33d1c`, the engine of record pre-registered for Claim 1 (`prereg/ERRATUM-2026-11-23.md`, item 2). **Claims 2 and 3:** the fast CPU route that 1.22.0 ships as its default, at `7bf8b64d`. |
| llama.cpp | Mainline `56eea9e4`, a CPU build (`-DGGML_NATIVE=ON`) |
| Unsloth | Its published CPU build `app-b12649-mix-c7059d6-linux-x64-cpu.tar.gz` (sha256 `510597b6aec3a49cdefb74843cfd9f2844dbf6ae6455cb363f49a5c46e368734`), run as published. A source build of the tag's commit could not load the model (`prereg/ADDENDUM-2026-11-25-unsloth-prebuilt.md`). |
| Host | A shared 64-core x86-64 server, CPU only, 16 threads, greedy decoding |

## Results, the pre-registered figure first

**Claim 1: the same answer as the reference** (100 prompts, at most 256 tokens). A prompt counts only if the engine's whole answer is the reference's own greedy answer, checked token by token with teacher-forced reference forwards. This is identity, not task quality.

| | Runner | llama.cpp | Unsloth |
|---|---|---|---|
| Pre-registered | 40/100 [0.309, 0.498] | 20/100 [0.133, 0.289] | 16/100 [0.101, 0.244] |
| Erratum 3, decided after the verdicts | 94/100 [0.875, 0.972] | 20/100 | 16/100 |

- **Pre-registered pairs:** Runner against llama.cpp is +0.200 (Newcombe 95% [+0.049, +0.339], exact McNemar p 0.0135), and against Unsloth +0.240 ([+0.099, +0.369], p 0.0015). Both are higher.
- **Erratum 3:** Runner's first client stored its answers without the stop token that ended them, so 58 answers counted as incomplete. Erratum 3 restores that token from each response's own record (`prereg/ERRATUM-2026-11-24b.md`). It is shown beside the pre-registered result, not in place of it.
- **The default fast route:** 1.22.0's fast CPU route scored 99/100 on the same prompts in its own pre-registered check (R4.48.24 in the 1.22.0 release notes).
- **Statistics:** `stats/claim1/`.

**Claim 2: the same answer alone and in a batch of 9**. This is the sixteen-thread half, and it is provisional; the one-thread cells run after the release.

| | Runner | llama.cpp | Unsloth |
|---|---|---|---|
| Prompts whose answer changed, alone against batched, default settings | 0/20 | 15/20 | 15/20 |
| Prompts identical across all of the engine's cells | 20/20 [0.839, 1.000], 2 cells | 2/20 [0.028, 0.301], 8 cells | 2/20, 8 cells |
| Throughput in tok/s, median, alone / batch of 9 | 11.18 / 11.80 | 10.49 / 18.55 | 10.86 / 18.50 |

- **Why Runner's answers cannot change:** on the CPU, Runner decodes concurrent requests for this MoE model one sequence at a time, so batching changes neither its arithmetic nor its speed (`prereg/CLAIM2-INTERPRETATIONS-2026-11-26-stats.md`, item 6).
- **The timings are information, not a speed comparison:**
  - host load differed between cells;
  - Runner's two cells lost up to 3.7% of the file's pages to the host's other tenants, while llama.cpp's stayed resident;
  - each figure has its load and residency beside it in `stats/claim2/`.

**Claim 3: the reasoning-loop guard** (40 code-debugging prompts, reasoning effort high, a 2048-token cap).

| Arm | Usable of 40 | Stuck in reasoning of 40 |
|---|---|---|
| Runner, unprotected | 24 | 15 |
| Runner, loop guard + reasoning budget | 37 [0.801, 0.974] | 0 |
| llama.cpp, unprotected | 25 | 14 |
| llama.cpp, reasoning budget | 39 [0.871, 0.996] | 1 |
| Unsloth, unprotected | 26 | 14 |
| Unsloth, reasoning budget | 37 | 2 |

- **The pre-registered outcome is a TIE.** Runner with its guard and budget against llama.cpp with its budget: -0.050, Newcombe 95% [-0.176, +0.067], p 0.625. Runner is higher than both unprotected arms.
- **What "usable" means:** a well-formed fix, not a verified one. The secondary unit-test diagnostic was not computed, because no isolated environment was available to run model-written code.
- **Statistics:** `stats/claim3/`.

## What was decided after data existed

Each of these is its own dated file in `prereg/`, and none changes a bar or a claim's wording:
- three errata;
- the Unsloth addenda (written before any Unsloth run);
- the Claim 2 and Claim 3 interpretation files, which say how the pre-registered definitions map onto the recorded fields.

Two more disclosures:
- **llama.cpp's web UI:** mainline llama.cpp's `llama-server` build downloads its web UI archive at build time. The archive is served only at `/`, and no recorded answer depends on it.
- **Missing raw logs:** the raw server logs are not published (see below).

## Redactions

The published files are byte copies of the research records, except that the measurement host's work and home paths, and its machine name, are replaced by `<work>`, `<home>` and `the lab host`. The 116 raw server and run logs (`.err`, `.out`, `.log`) are left out.

`REDACTION-MANIFEST.json` in the release attachment lists every file's original sha256 beside its published sha256. Only path strings changed: no prompt, answer, token id, verdict or statistic.

## Files here

- `prereg/`: the pre-registration, its errata and addenda, and the interpretation files.
- `prompts/`: each claim's prompts, with every line's source, revision and licence (Apache-2.0 or MIT; `prereg/PREREG.md` 0.4). The scored prompts are the lines marked `primary`: 100, 20 (plus 8 companions) and 40. The `reserve` lines were drawn by the same rule and not used.
- `stats/`: every statistics output the release notes quote.
- `tools/`: the statistics scripts, the teacher-forced verifier and the clients.
