# DeepSeek-V2-Lite-Chat: the R4.4.5 step (d) certification (lab, 2026-11-04)

**Unflattering first: two of the four files measured fail the bar, and llama.cpp misses it on the row's user file.**
mradermacher's Q4_K_M reads mean KLD 0.188 and margin-qualified top-1 90.28%. The Q5_K_M reads 0.0704 and 96.76%. The
bar is 0.05 and 97%. Both misses are the files themselves: Runner's CPU and CUDA read them alike, and llama.cpp reads
them worse. On the Q6_K, which passes on Runner, llama.cpp reads 0.0466 and 96.76%, below the margin half of the bar.

**The Q8_0 is the fidelity file, and the Q6_K is the smallest file to pass** (the coordinator's rows). Each is pinned
by sha256 and repository revision; both are a third party's static quants (mradermacher), not the publisher's.

| File | Bytes | Engine | Mean KLD | Median | Max KLD | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|---|
| Q8_0 (the fidelity file) | 16.70 GB | Runner CPU | **0.0094** | 0.0033 | 0.097 | 96.67% | 100% |
| | | Runner CUDA | 0.0096 | 0.0033 | 0.098 | 96.00% | 100% |
| | | llama.cpp | 0.0155 | 0.0074 | 0.189 | 94.67% | 99.54% |
| Q6_K (the user file) | 14.07 GB | Runner CPU | **0.0293** | 0.0157 | 0.198 | 93.00% | 98.61% |
| | | Runner CUDA | 0.0296 | 0.0155 | 0.200 | 93.00% | 98.61% |
| | | llama.cpp | 0.0466 | 0.0243 | 0.622 | 88.67% | 96.76% |
| Q5_K_M (fails) | 11.85 GB | Runner CPU | 0.0704 | 0.0351 | 1.168 | 88.33% | 96.76% |
| | | Runner CUDA | 0.0705 | 0.0344 | 1.168 | 88.00% | 96.30% |
| | | llama.cpp | 0.0861 | 0.0429 | 2.251 | 86.67% | 96.30% |
| Q4_K_M (fails) | 10.36 GB | Runner CPU | 0.1884 | 0.0958 | 3.224 | 81.33% | 90.28% |
| | | Runner CUDA | 0.1876 | 0.0955 | 3.286 | 81.67% | 90.28% |
| | | llama.cpp | 0.2023 | 0.0984 | 3.641 | 79.33% | 89.35% |

The method is the house's: 300 positions at stride 15 (8 to 4,493) of `tests/fixtures/gold-corpus-2k.txt`, every one
matched by token id, with no dropped mass on either side. Margin-qualified positions are the 216 where the reference's
top-1 leads by more than 0.5 nats. The kill numbers were declared before the runs (exec-share PLAN-d): per file and
backend, mean KLD at most 0.05 and margin-qualified top-1 at least 97%; a file that misses gets no row.

- **CPU against CUDA.** The window cannot tell them apart on any file (paired per-position KLD, Wilcoxon p 0.51 on the
  Q8_0, 0.54 on the Q6_K, 0.36 on the Q5_K_M; McNemar 0/0, 0/0, 1/0). On the Q4_K_M CUDA is the closer by a hair
  (0.1876 against 0.1884, p 0.033, McNemar 0/0). gpu-identity's pinned-routing arm, the verdict for a MoE file, reads
  1.85e-5 (Q8_0), 1.94e-5 (Q6_K), 2.0e-5 (Q5_K_M) and 1.76e-5 (Q4_K_M) of the mean logit range; the limit is 1e-3
  (`gpu-identity-*.log`).
- **Runner against llama.cpp** (the CUDA side against llama.cpp's server on the same held file): Runner is the closer
  by per-position KLD on every file. Wilcoxon p 1.2e-16 (Q8_0), 3.2e-18 (Q6_K), 4.7e-7 (Q5_K_M) and 3.2e-5 (Q4_K_M).
  Margin-qualified discordant positions: 1/0, 4/0, 2/2 and 6/4 (Runner only / llama.cpp only).
- **Depth split** (mean KLD / margin-qualified top-1, Runner CPU; 41, 55 and 120 qualified positions):

  | Positions | Q8_0 | Q6_K | Q6_K llama.cpp | Q5_K_M | Q4_K_M |
  |---|---|---|---|---|---|
  | 8-749 | 0.0099 / 100% | 0.0274 / 97.6% | 0.0372 / 97.6% | 0.0538 / 97.6% | 0.1431 / 92.7% |
  | 750-1,999 | 0.0092 / 100% | 0.0284 / 100% | 0.0426 / 100% | 0.0810 / 94.5% | 0.1793 / 87.3% |
  | 2,000-4,493 | 0.0094 / 100% | 0.0303 / 98.3% | 0.0513 / 95.0% | 0.0700 / 97.5% | 0.2066 / 90.8% |

  The Q6_K passes in every range on Runner. llama.cpp's miss on it is in the deepest range (95.0%).
- **Full-corpus PPL (Runner CPU / CUDA, 8,719 tokens, chunked prefill):** Q8_0 14.365 / 14.362, Q6_K 14.401 /
  14.397, Q5_K_M 14.631 / 14.636, Q4_K_M 15.428 / 15.428.

## The reference

The reference is DeepSeek's own `modeling_deepseek.py` at revision 85864749 of `deepseek-ai/DeepSeek-V2-Lite-Chat`,
run through `--trust-remote-code` on transformers 5.16.1 with float32 arithmetic on the BF16 weights, cast at use per
decoder block (`scripts/gold-logits.py --upcast-at-use`, CPU, 27 blocks). Two compatibility shims let the remote
code load on transformers 5.16.1, neither in the arithmetic: `is_torch_fx_available`, removed in 5.x, supplied as
False, and the config's cache turned off, since the remote code's cache path calls a removed API and one
full-sequence forward needs no cache (on a tiny random DeepseekV2 the two forwards are equal, max |diff| 0.0).
transformers' built-in `deepseek_v2` is not used: it drops the publisher's mscale² softmax factor. The tokenizer writes a BOS, so BOS is forced on the reference side.

## Why the columns differ, one numbered sentence each

1. **llama.cpp's miss on the Q6_K is its CPU arithmetic, not the multi-head latent attention form.** On these files
   (the unsplit `attn_kv_b`, no `key_length_mla`) llama.cpp 73a43d1 runs the naive form, where Runner defaults to the
   absorbed one. A declared read separated the two: Runner in the naive form reads 0.0294 / 98.61% on the CPU and
   0.0297 / 98.61% on CUDA, not distinguishable from its absorbed form (Wilcoxon p 0.32, McNemar 0/0). Against
   llama.cpp in the same form, Runner is the closer (Wilcoxon z 8.74, p 2.4e-18; McNemar 4/0). What remains is
   llama.cpp's kernel path. The one arithmetic difference there that we know of is its 8-bit activation blocks (Q8_K
   against the K-quant types, Q8_0 against Q8_0), beside Runner's float32 activations (`cpu_act` f32; the int8 routes
   are opt-in and were off). No ablation here isolates that part (`dcause-gold-*.json`).
2. gold-logits counted 2 ids at 2 positions (83 and 143) whose spelling on the llama.cpp side differs from the
   reference's decode, on the Q8_0 and the Q6_K alike (1 on the Q5_K_M and the Q4_K_M). Granite's report traced
   such ids to the server cutting partial UTF-8 to a valid prefix; that cause does not account for these: of the
   31 vocabulary ids it affects, none is in Runner's top 20 at either position, and the count is over
   llama.cpp's own top 20, which the check cannot see (`llamacpp-spelling-*.json`). The column matches by id, so
   they change no number.
3. The column sends token ids (`--endpoint-ids`), so llama.cpp's own tokenizer is not exercised by these numbers.

## The rows

- **Files**, each checked against the Hub's LFS digest as it streamed into memory (sha256 and sizes in
  `FILES.sha256`, which also lists the parent's files): `mradermacher/DeepSeek-V2-Lite-Chat-GGUF` at 9ee9f1f8f39f.
  - `DeepSeek-V2-Lite-Chat.Q8_0.gguf`: 78a1a180db564167..., 16,702,518,016 bytes.
  - `DeepSeek-V2-Lite-Chat.Q6_K.gguf`: 3f634b8f08f4ef10..., 14,066,972,416 bytes. Every matrix is Q6_K except the 26
    expert down tensors and the dense block's down, whose rows (1,408 and 10,944 wide) are not multiples of 256 and are
    Q8_0; norms and routers are F32.
  - Measured and failed: `DeepSeek-V2-Lite-Chat.Q5_K_M.gguf` (d690d0f1b7a68386..., 11,851,313,920 bytes) and
    `DeepSeek-V2-Lite-Chat.Q4_K_M.gguf` (5d33e5f045c7a033..., 10,364,416,768 bytes).
- **Device memory on the 24 GB slice at a 9,216-token context:** the Q6_K used 15.24 GB (14.45 GB at init with a short
  context), a 16 GB card's fit and not a 12 GB one's; the Q8_0 used 17.83 GB.
- **Tokenizer:** the differential reads 0 of 721 corpus segments and 0 of 99,442 vocabulary entries against the
  publisher's `tokenizer.json` at 85864749 (`difftok-*.log`). The corpus capture is
  `tests/compatibility/tokenizer-references/deepseek-v2-lite-chat.tokenizer-json.json`.
- **Template:** both GGUFs' `tokenizer.chat_template` is byte-identical to the parent's `tokenizer_config.json`
  `chat_template` (`template-gguf-*.log`). Conformance runs as family `deepseek2` (`template-conformance.py --family
  deepseek2 --require-tokens`, main 0676f26): 16 of 16 cases conform, and 16 of 16 are token-identical
  (`template-conformance.log`). The template has no tool protocol, so no tool gate applies.
- **Compatibility checks** (`tests/compatibility/models.json`, `scripts/compat_matrix.py --execute-checks` against the
  held files): two rows, one per file, each declaring `load`,
  `tokenizer`, `cpu_cuda` (128 tokens under the MoE pins `RUNNER_CUDA_TC=0`, `RUNNER_CUDA_ROUTE=reference`,
  `RUNNER_MOE_EAGER=1`) and `chat`. All four pass on both files (`../cpu_cuda/deepseek-v2-lite-chat-*.json`).
  - Q6_K: cpu_cuda pass, 9 of 9 prompts byte-identical, 27 of 27 layers and the head on the device.
  - Q8_0: cpu_cuda pass, margin-qualified: 7 of 9 byte-identical and 2 near-ties tolerated inside the 0.5-nat band
    (' was' against ' kept' at token 1, margins 0.022 / 0.005 nats; '8' against 'size' at token 81, 0.035 / 0.022).
  - chat ("What is 2+2?") answers " 2 + 2 equals 4." on both, and stops.
  - The runs went through a lab wrapper around `compat_matrix.py` that changes no check: the files are held in memory
    as links to `/proc/PID/fd/N`, and the harness's `Path.resolve()` turns such a link into a name no other process
    can open, so the wrapper keeps the link path. The first attempts, before the wrapper, failed on that name and
    measured nothing.
- **Tier smoke:** both files pinned and recorded at main 0676f26 on the Linux CUDA host, CUDA and the CPU backend
  of the same machine (`tier-smoke-*-{pin,record}.json`; the CPU cells serve with `--gpu off`, the spec of the other
  linux pins: `-n 48`, ctx 512, `-b 64`). Every reply lists the planets in order; the fact was found, and each record
  matched its pin.
  - Q6_K: CUDA pin sha256 70827d501fa8dc01..., 193 bytes; CPU pin eae8d81a35486506..., 185 bytes. The two backends'
    48-token replies part in the first sentence's wording, so each cell pins its own.
  - Q8_0: CUDA and CPU pins both eae8d81a35486506..., 185 bytes, the same text as the Q6_K's CPU reply.
  - The Q8_0's first CUDA `--pin` did not count: its server gave the whole 48-token reply (104.5 tok/s) and then
    outlived the harness's bounded stop (SIGTERM, then SIGKILL, 10 s each), so no pin was written. A probe on the
    same held file then timed the stop: exit 0 in 5.0 s after SIGTERM, the last 4 s in uninterruptible disk waits
    (`xlog_wait_on_iclog`, `folio_wait_bit_common`) on this lab's I/O-capped volume. The rerun pinned and recorded.
- **Adapters:** `--info adapter` reports serving and training not supported, with the reason that `--lora` does not
  cover multi-head latent attention yet (`info-adapter-*.json`).
- **Roster:** `deepseek-v2-lite-chat-q8_0` and `deepseek-v2-lite-chat-q6_k` in `docs/model-tiers.json`, Tier 2.
  Licence `deepseek` (DeepSeek's model licence: use-based restrictions, commercial use allowed), as the parent's card
  states it (`license: other`, `license_name: deepseek`); the coordinator ruled it eligible.

## Setup

- **Runner:** main 0676f26 (#620), the absorbed form, the default.
  - The score files ran on the chunked route: `RUNNER_SCORE_CHUNKED=1 runner --score 20 -c 9216 -t 40`, `--gpu off`
    on the CPU (x86-64 with AVX-512, 40 threads) and `--gpu 99` on CUDA (MIG slice, 1g.24gb, sm_120). The tokens equal
    the step (b) K1 read's on every file and backend.
  - The whole-corpus chunked score took 169 to 235 s on CUDA in the absorbed form, against 775 to 851 s on the
    CPU, with the host's one-minute load (outside tenants included) between 19 and 130.
- **llama.cpp:** 73a43d1, `llama-server` (CPU build, flash attention off, `-c 9216 -t 40 -tb 40`), with the GGUF held in
  memory.
- **Reference:** transformers 5.16.1, torch 2.14.0 (CPU), float32 at use. The parent was held in memory for every pass,
  because this lab's disk rereads at 6-11 MB/s and its page cache is reclaimed from outside. A memory watchdog held
  each run under 150 GB; the largest sample was 89.6 GB.

## Not measured

- Metal: a Metal build runs deepseek2 on the CPU and says why.
- The Windows CUDA host's cells.
- The KV cache's size and decode past the window (R4.4.5 step e).
- Task quality.
