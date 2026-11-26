# OLMo 3 7B CPU benchmark rows

**Limits first:** Runner is slower on all three measured rows: prefill
2048 at 0.33x llama.cpp, prefill 512 at 0.40x, decode at 0.85x. The missing
1B Q8_0 CPU row was not run. These information rows do not gate admission.

## Scope and differences

Measured on 2026-11-26 on a shared **64-core x86-64 CPU (AVX-512)**,
sixteen threads per engine without affinity pinning, one model process at
a time. Runner was built
from `83da2dc3050ec28a1ad1ea24c150c10841897f95` with the release recipe:
x86-64-v3 (AVX2/FMA, no AVX-512), GCC 13.3.0. Its binary SHA256 is
`1ce8e51f43960db2f68e2a609da165a02b16889cf66a843b472512f65791ecb8`.
The CUDA backend was compiled in and dormant under `--gpu off`. KV is f16,
weights Q8_0, CPU activations f32.

llama.cpp is `56eea9e4b0f5d125da16b75b551440b170aa8b1f`,
`llama-bench` built with native AVX-512 (`GGML_NATIVE=ON`), GNU 13.3.0.
The executable SHA256 is
`859e9e7efa554a49d3254f3b96d9700ed09cdfccb372f1cc80a33aa9459ac7d5`.
This build difference is stated beside every timing row.

The exact file is `Olmo-3-7B-Think-Q8_0.gguf`, 7,759,251,584 bytes,
SHA256 `3a4d7dda5778c299fe30bac76f5300beb283b48c6cc12b01c36c71310cefcf08`.
Its [admission record](compat-reports/olmo2-admission-2026-11-24/README.md)
names the pinned repository and checkpoint.

Runner uses its own bench prompt filler, with the longer prompt cut at
a token boundary; llama-bench uses its own synthetic tokens. Runner's CPU
prefill batch is 256; llama-bench defaults are batch 2048 / micro-batch 512,
f16 K/V and automatic flash attention. Runner's decode bench samples with
its default sampler and ignores EOS; llama-bench does not sample. This is
not a matched-prompt or matched-codegen timing comparison.

## Method and rows

The method was fixed before timing. Each engine received an uncounted
warm-up of the three rows; five counted rounds then alternated engine
order. Every file was read whole and its start residency checked before
a counted run; the gate was 0.99, every observed value was 1.0 and no
residency miss occurred. The load columns combine samples inside both
engines' counted windows for the workload, not a separate engine-normalized
load estimate. Medians, min/max and spread are reported without adjustment.

| Workload, sixteen CPU threads | Runner median [min, max] tok/s; spread | llama.cpp median [min, max] tok/s; spread | Runner / llama.cpp | Build codegen on every row | Count per engine | Load median / peak | Start residency / misses |
|---|---:|---:|---:|---|---:|---:|---|
| Prefill 2048 | 59.894 [58.751, 60.723]; 3.3% | 183.384 [181.552, 184.343]; 1.5% | 0.327x | Runner x86-64-v3 / llama.cpp native AVX-512 | 5 | 24.170 / 26.89 | 1.000000 / 0 |
| Prefill 512 | 79.149 [72.096, 81.075]; 11.3% | 197.107 [196.925, 197.853]; 0.5% | 0.402x | Runner x86-64-v3 / llama.cpp native AVX-512 | 5 | 24.485 / 28.17 | 1.000000 / 0 |
| Decode 128 at depth 512 | 11.833 [11.712, 12.118]; 3.4% | 13.864 [13.625, 14.360]; 5.3% | 0.853x | Runner x86-64-v3 / llama.cpp native AVX-512 | 5 | 23.380 / 26.27 | 1.000000 / 0 |

The 1B Q8_0 file was not available for this CPU measurement and was not
fetched; no 1B CPU timing is inferred from the 7B or from its separate CUDA
speed row. These are benchmark information, not a performance pass/fail bar.
