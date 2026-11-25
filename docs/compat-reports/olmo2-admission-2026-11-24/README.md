# OLMo 2 and OLMo 3 admission

Limits first: these fidelity measurements run on the CPU at default f16 KV.
Backend smoke pins are separate records, never inherited from another file.
GPU training is refused because its gradient gate needs f32 KV rows, which
the GPU trainer does not keep. fp4 KV can amplify different product sums
across backends at wide direct-model batches; the compact-cache CPU/Metal
identity gate covers batches below 8 and decode. The 7B Q8_0 file cannot fit
with runtime state on the owned 8 GB macos-arm64-metal device.

## Reference and bar

The two listed Instruct/Think files are each compared with their own publisher's implementation in
float32, on the CPU, using transformers-main `046bdb8` (`5.24.0.dev0`). The
OLMo 2 checkpoint is float32; OLMo 3's BF16 checkpoint is upcast at use.
Runner's CPU route at `f3ff8e84` and llama.cpp `56eea9e4` each read the same
GGUF, corpus and positions: `tests/fixtures/gold-corpus-2k.txt`, score top 20,
2,000 positions at stride 2, context 4,096, default f16 KV.

The bar, set before the 7B read, requires 100% margin-qualified top-1 and
Runner's mean KL not worse than llama.cpp by separated 95% intervals. It
fails only if Runner's interval lies wholly above llama.cpp's. Template
conformance must pass in text and tokens. Plain top-1 is reported beside the
margin-qualified count. Intervals use 10,000 percentile-bootstrap replicates
with a fixed seed; paired differences are information, not another bar.

| File | Runner vs publisher float32: mean KL [95% interval] | Plain / margin-qualified top-1 | llama.cpp vs publisher float32: mean KL [95% interval] | Plain / margin-qualified top-1 | Bar |
|---|---|---|---|---|---|
| OLMo-2-0425-1B-Instruct-F16 | 1.0642e-04 [6.1904e-05, 1.5615e-04] | 100% / 100% | 1.1261e-04 [6.6509e-05, 1.6433e-04] | 100% / 100% | Pass: intervals overlap |
| Olmo-3-7B-Think-Q8_0 | 1.3671e-03 [1.1975e-03, 1.5472e-03] | 98.55% / 100% | 9.1434e-02 [8.5543e-02, 9.8364e-02] | 85.25% / 95.77% | Pass: separated in Runner's favour |

TODO-linux-x86_64-cuda: commit the raw gold-logits and bootstrap result files for both
reads and record their paths here. The aggregate figures above are the
R4.50.6 read; no claim about why llama.cpp differs is made without tracing.

## Exact artifacts

| File repository and revision | File | SHA256 | Bytes | Publisher checkpoint and revision |
|---|---|---|---|---|
| allenai/OLMo-2-0425-1B-Instruct-GGUF at 62f8c199538474c3e33ed5d7e0580abd66686a27 | OLMo-2-0425-1B-Instruct-F16.gguf | 376249dff02917a8cf5ea413d2bfdf2802f10f7d92973d95a6f28b2204ac4eab | 2973652128 | allenai/OLMo-2-0425-1B-Instruct at 48d788eca847d4d7548f375ad03d3c9312f6139e |
| unsloth/Olmo-3-7B-Think-GGUF at 1dcda0d5e447e9da698330014e38a9080616f2df | Olmo-3-7B-Think-Q8_0.gguf | 3a4d7dda5778c299fe30bac76f5300beb283b48c6cc12b01c36c71310cefcf08 | 7759251584 | allenai/Olmo-3-7B-Think at d97e442d7cc678210054dbcc9b440894d62c89a4 |

The other downloaded quantizations, OLMo 2 Q8_0 and OLMo 3 Q4_K_M, are not
admitted by these measurements. A quantized sibling cannot inherit the
F16 or Q8_0 file's fidelity result.


## OLMo 2 Instruct

This is the Instruct checkpoint's own read, not the base model's result.
The clean Runner build is `f3ff8e84e60f0d75ad5357a667a8ac1f338a694e`, binary
SHA256 `3428416e04f813be3422b45c01a4038e66b59a27f7c2201d2beead45d67fad70`.
Its CPU command scores 4,009 tokens, top 20, context 4,096, 16 threads and
f16 KV: 4,008 solo forwards. The comparison takes 2,000 positions at stride
2. The publisher checkpoint is loaded in float32 with zero tensors cast,
using transformers-main `046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`
(`5.24.0.dev0`), torch `2.14.0+cpu`, CPU, 16 OpenMP threads. The run was
2026-11-24, 21:08:15 to 21:12:44 UTC, untimed.
Attention implementation was not recorded: the loader passed no
`attn_implementation`, leaving transformers-main `046bdb8`'s default.

The llama.cpp read matches 2,000 of 2,000 positions by token ID, with zero
dropped mass and 10 token-spelling mismatches (informational; the ID-keyed
read is the comparison). Its maximum KL is 0.01889. Intervals use 10,000
percentile-bootstrap resamples, seed 4506.

R4.50.10 (#1029) supplies the native `olmo2` chat template: 16 of 16 cases
match the publisher and file in text and tokens. The two gates are separate:
the raw forward's float32 bar does not certify chat rendering. Template
conformance and the Instruct file's own bar both passed before this row
replaced the base file. The backend smoke cells are still unrecorded.

## Base 1B: separate numerical evidence

The base file is no longer a support-list or recommendation row and has no
chat smoke. It remains measured evidence: `allenai/OLMo-2-0425-1B-GGUF` at
`d4cd762e4f991e42421f5340290afeff8c5ff146`, `OLMo-2-0425-1B-F16.gguf`,
2,973,650,560 bytes, SHA256
`452d42f1ac5a1d1d62a6a7c4e2c6adfe4ac279f6b4ff44c9e2ea452a0d9c6df2`.
Its reference is the base checkpoint `allenai/OLMo-2-0425-1B` at
`a1847dff35000b4271fa70afc5db10fd29fedbdf`, float32. On the same numerical
setup, Runner reads mean KL 2.3159e-04 [1.5377e-04, 3.1844e-04], llama.cpp
2.2777e-04 [1.5055e-04, 3.1416e-04]; both plain and margin-qualified top-1
are 100%. These are only the base file's numbers and are not inherited by
Instruct.

## Protocol and training

OLMo 3 conformance is 48 of 48 cases, token for token: 24 against the
publisher's template and 24 against the GGUF template (the same bytes).
Its tokenizer matches transformers on all 721 corpus strings and all 99,474
vocabulary entries. The native function fields and environment role render;
modern OpenAI tool requests are refused, not translated. See the committed
[template families](../../../scripts/template-conformance.py)
and [tokenizer reference](../../../tests/compatibility/tokenizer-references/olmo-3-7b-think.tokenizer-json.json).
The base OLMo 2 file has no claimed chat or tool protocol.

CPU LoRA training's derivative gate runs on generated OLMo 2 and OLMo 3
files with f32 KV rows: 168 sampled coordinates resolve, and a deliberately
wrong per-head QK adjoint fails its live control. The learning, serving,
thread and tape-width checks are generated-file checks, not training-quality
measurements on these checkpoints. GPU training remains refused.


## Chat smoke: failed first request and fixed rerun

At `a97acfa6`, all four initial linux-x86_64-cuda pin attempts failed the
unchanged fact rule. The 1B base F16 on CPU and CUDA echoed the prompt,
finish stop, 25 tokens. This is not a chat-validation pass. That support
row was removed once the Instruct file passed its own float32 bar and
R4.50.10 merged. Its base-file numbers remain separately disclosed above;
no result or backend mark is inherited by the Instruct row.

The 7B Think CPU and CUDA runs reached the 600-token cap with empty content.
The retained CPU response confirms 2,610 UTF-8 bytes in reasoning_content,
zero content bytes, all 600 completion tokens in reasoning, finish length,
and no closing `</think>`. The template's opener was already in the prompt.
The smoke's failed-fact diagnostic now reports the channel's byte count.
No pin or result JSON was written for these failures.

The fixed 7B rerun, stated before measuring it, keeps the same planets
prompt, required fact, minimum 96 content bytes, temperature zero, batch 64
and context 4,096. It changes max_tokens to 3,072 and sets the row's request
to `{}`: the native template unconditionally opens `<think>` and has no
`enable_thinking` or `reasoning_effort` branch. This is a larger total cap,
not a no-think claim or a weakened fact rule. If it still fails, nothing is
pinned; the request is not tuned separately for each backend.

Both linux-x86_64-cuda pin and comparison commands use `taskset -c 0-15`.
The smoke supplies no thread-count flag, so its default counts that same
affinity mask on both runs. Other device classes record their own pins.

## Pending backend recordings

Each entry below needs a real `scripts/tier-smoke.py --pin`, followed by its
comparison run with `--record`. Pinning alone records no gate. The pin's
output SHA256 and each gate/adapter answer must come from those commands.

| Row | Device class / backend | State |
|---|---|---|
| olmo-2-0425-1b-instruct-f16 | macos-arm64-metal / cpu | Pass: unchanged request after RAM was freed, own pin and clean-rebuild comparison |
| olmo-2-0425-1b-instruct-f16 | macos-arm64-metal / metal | Pass: exact pin comparison, 2026-11-25, clean build d00dbcdd |
| olmo-2-0425-1b-instruct-f16 | windows-x86_64-cuda / cuda | Release-recipe pin committed; --record comparison pending |
| olmo-2-0425-1b-instruct-f16 | linux-x86_64-cuda / cpu | Release-recipe pin committed; --record comparison pending |
| olmo-2-0425-1b-instruct-f16 | linux-x86_64-cuda / cuda | Release-recipe pin committed; --record comparison pending |
| olmo-3-7b-think-q8_0 | linux-x86_64-cuda / cpu | Release-recipe pin committed; --record comparison pending |
| olmo-3-7b-think-q8_0 | linux-x86_64-cuda / cuda | Release-recipe pin committed; --record comparison pending |
| olmo-3-7b-think-q8_0 | windows-x86_64-cuda / cuda | Release-recipe pin committed; --record comparison pending |
| olmo-3-7b-think-q8_0 | macos-arm64-metal / metal | Limit: the file plus runtime state does not fit the owned 8 GB macos-arm64-metal device |

### macos-arm64-metal recording, 2026-11-25

The exact Instruct F16 file was fetched by its pinned identity and verified
at 2,973,652,128 bytes and the SHA256 above before use. The request stayed at
600 tokens, context 4,096, batch 64, temperature zero, with no thinking flag.

The CPU attempt on clean `1af81e9c` reached the existing 1,800-second timeout
and returned no completed HTTP reply. Its final server log reports 324
generated tokens at 0.2 tok/s and 50,540,623 page-ins, with 3.0 GB weights
against 1.5 GB available RAM. It produced no pin or result JSON. The later RAM-freed retry is
reported separately below. That first attempt produced no comparison result. The separate retry below supplies the later CPU validation.

The Metal pin from clean `1af81e9c` passed the fact and length rules: 1,476
bytes, 324 completion tokens, finish stop, output SHA256
`5ede65273868fbf74071f9b8becf9af6c68c5bc4f08234db125fb8d7e820ae96`.
After committing the generated pin as `d00dbcdd`, a fresh STRICT build made
the identical request. `scripts/tier-smoke.py --record` matched those bytes
and wrote the Metal pass and adapter answer. Its build commit, compiler,
binary digest and class fingerprint are in the generated support row. This
is a reproducibility check with a fact guard, not a task-quality benchmark.

### CPU retry after RAM was freed

The same 600-token request, context 4,096, batch 64, temperature zero and
1,800-second timeout was run once more after RAM was freed. System-wide memory free percentage: 57%
The pre-pin vm_stat snapshot reported 4387 free and 134012
inactive pages of 16384 bytes each. Its own CPU pin passed the fact
and length rules. After committing that pin as `fb0be48b`, a clean
STRICT build matched its bytes and the tool wrote the CPU pass. The earlier
timeout remains disclosed; no Metal result was inherited.

### Release-recipe x86 pins, comparisons pending

The six x86 pins were made from clean STRICT `0bbdda0a` builds using the
release recipe, `PATH=$PWD/tools/release-make:$PATH make` (`-march=x86-64-v3`).
The comparison builds must use that same recipe. Earlier native-build
artifacts remain private and are not the committed pins, even though their
reply bytes happen to match.

| Device class | Backends pinned | Release binary SHA256 prefix |
|---|---|---|
| linux-x86_64-cuda | CPU, CUDA, both files | c024d9b4 (GCC 13.3.0) |
| windows-x86_64-cuda | CUDA, both files | 0ac151ef |

Both Linux pin and comparison commands use `taskset -c 0-15`. Windows uses
the default affinity. The Windows pinning runs used an sm_120 GPU; that
class is matched by OS, architecture and backend. Its descriptive hardware
text is not the hardware claim for these two runs.

The Instruct 1B payload is 1,476 bytes, 324 completion tokens, finish stop,
SHA256 `5ede65273868fbf74071f9b8becf9af6c68c5bc4f08234db125fb8d7e820ae96`,
on all supplied backends. Think 7B's content is 1,173 bytes, finish stop at
2,129 completion tokens within the fixed 3,072 cap, SHA256
`89360cee1c4ce4b6c2bc9e077edc171c54dd175125535ff235e4aca967f383f7`.
Its two-space Markdown line breaks are part of those bytes and are retained.

These are pins only: no x86 comparison pass is inferred or entered. The
six release-recipe `--record` runs follow a clean rebuild at the pin commit.
The existing macos-arm64-metal CPU and Metal cells remain unchanged.

### Merged tensor-core default

R4.50.9 (#1026) merged in `76be6540`: CUDA Q8_0 tensor-core prefill is now
on by default; other olmo2 quant types stay scalar until measured. The six
x86 pins retain their actual clean `0bbdda0a` build provenance, before that
promotion. The upcoming CUDA Q8_0 comparisons use the merged default and
report any byte mismatch; they do not force the prior scalar route. The
measured promotion bar and its limits are in the
[tensor-core evidence](../../tc-promotion-evidence.md#olmo2-q8_0-r4509-2026-11-24).
