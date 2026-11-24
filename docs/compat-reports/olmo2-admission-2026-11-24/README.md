# OLMo 2 and OLMo 3 admission

Limits first: these fidelity measurements run on the CPU at default f16 KV.
Backend smoke pins are separate records, never inherited from another file.
GPU training is refused because its gradient gate needs f32 KV rows, which
the GPU trainer does not keep. fp4 KV can amplify different product sums
across backends at wide direct-model batches; the compact-cache CPU/Metal
identity gate covers batches below 8 and decode. The 7B Q8_0 file cannot fit
with runtime state on the owned 8 GB macos-arm64-metal device.

## Reference and bar

Both files are compared with their publisher's reference implementation in
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
| OLMo-2-0425-1B-F16 | 2.3159e-04 [1.5377e-04, 3.1844e-04] | 100% / 100% | 2.2777e-04 [1.5055e-04, 3.1416e-04] | 100% / 100% | Pass: intervals overlap |
| Olmo-3-7B-Think-Q8_0 | 1.3671e-03 [1.1975e-03, 1.5472e-03] | 98.55% / 100% | 9.1434e-02 [8.5543e-02, 9.8364e-02] | 85.25% / 95.77% | Pass: separated in Runner's favour |

TODO-linux-x86_64-cuda: commit the raw gold-logits and bootstrap result files for both
reads and record their paths here. The aggregate figures above are the
R4.50.6 read; no claim about why llama.cpp differs is made without tracing.

## Exact artifacts

| File repository and revision | File | SHA256 | Bytes | Publisher checkpoint and revision |
|---|---|---|---|---|
| allenai/OLMo-2-0425-1B-GGUF at d4cd762e4f991e42421f5340290afeff8c5ff146 | OLMo-2-0425-1B-F16.gguf | 452d42f1ac5a1d1d62a6a7c4e2c6adfe4ac279f6b4ff44c9e2ea452a0d9c6df2 | 2973650560 | allenai/OLMo-2-0425-1B at a1847dff35000b4271fa70afc5db10fd29fedbdf |
| unsloth/Olmo-3-7B-Think-GGUF at 1dcda0d5e447e9da698330014e38a9080616f2df | Olmo-3-7B-Think-Q8_0.gguf | 3a4d7dda5778c299fe30bac76f5300beb283b48c6cc12b01c36c71310cefcf08 | 7759251584 | allenai/Olmo-3-7B-Think at d97e442d7cc678210054dbcc9b440894d62c89a4 |

The other downloaded quantizations, OLMo 2 Q8_0 and OLMo 3 Q4_K_M, are not
admitted by these measurements. A quantized sibling cannot inherit the
F16 or Q8_0 file's fidelity result.

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

## Pending backend recordings

Each entry below needs a real `scripts/tier-smoke.py --pin`, followed by its
comparison run with `--record`. Pinning alone records no gate. The pin's
output SHA256 and each gate/adapter answer must come from those commands.

| Row | Device class / backend | State |
|---|---|---|
| olmo-2-0425-1b-f16 | macos-arm64-metal / cpu | TODO-macos-arm64-metal: disk space, pin and record |
| olmo-2-0425-1b-f16 | macos-arm64-metal / metal | TODO-macos-arm64-metal: disk space, pin and record |
| olmo-2-0425-1b-f16 | windows-x86_64-cuda / cuda | TODO-windows-x86_64-cuda: pin and record |
| olmo-2-0425-1b-f16 | linux-x86_64-cuda / cpu | TODO-linux-x86_64-cuda: pin and record |
| olmo-2-0425-1b-f16 | linux-x86_64-cuda / cuda | TODO-linux-x86_64-cuda: pin and record |
| olmo-3-7b-think-q8_0 | linux-x86_64-cuda / cpu | TODO-linux-x86_64-cuda: pin and record |
| olmo-3-7b-think-q8_0 | linux-x86_64-cuda / cuda | TODO-linux-x86_64-cuda: pin and record |
| olmo-3-7b-think-q8_0 | windows-x86_64-cuda / cuda | TODO-windows-x86_64-cuda: pin and record |
| olmo-3-7b-think-q8_0 | macos-arm64-metal / metal | Limit: the file plus runtime state does not fit the owned 8 GB macos-arm64-metal device |
