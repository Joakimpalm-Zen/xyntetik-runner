# gpt-oss-120B fully resident on Metal — M5 Max, 2026-08-31

Machine-locked validation on Apple M5 Max (18 GPU cores), 128 GB unified RAM,
macOS 26.5. Artifact:
`models/gpt-oss-120b-MXFP4.gguf`, 63,387,346,208 bytes, SHA-256
`582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d`
(matches the recorded Hugging Face blob hash).

Runner `39b06d4` wrapped the single-file model zero-copy and fully offloaded all
36 layers. A short instrumented run engaged the native Metal MoE route
(`moe=216` dispatches for the nine-token prefill) and generated coherent text.
Swap was zero before and after.

`test-gpu-identity` then compared the established scalar and Metal routes over
24 teacher-forced positions (4,826,112 logits): mean absolute delta 0.00451,
0.000215 of the mean logit range, inside the 0.002 bound. The gate passed.

A realistic sustained run used a 715-token prompt, batch 512, and 128 greedy
decode tokens:

- prefill: 54.65 tok/s
- decode: 64.59 tok/s
- swap after completion: exactly 0

This closes the previously 24-GB-MIG-blocked full-residency validation. It does
not claim CPU/Metal byte identity: gpt-oss routing is sensitivity-gated, and
the measured logit envelope is the existing correctness contract.

## Same-process lifecycle

A single `--serve` process was kept alive across load, a real 66+64-token
request, `POST /unload`, and a second request that reloaded the same 59 GiB
artifact. This distinguishes backend cleanup from the kernel reclaiming a
dead process.

- after the first request: 1,024 MB physical footprint, including 575 MB
  IOAccelerator plus 232 MB owned unmapped graphics memory;
- after `/unload`: 214 MB total, IOAccelerator down to 736 KB dirty plus
  1,936 KB reclaimable, and `/health` reported `resident:null`;
- the next request reloaded all 36 Metal layers and completed at 71.6 tok/s;
- a second `/unload` completed cleanly;
- swap remained exactly zero throughout.

Both cold requests reported roughly 1.1-1.2 million major page-ins. That is
file-backed weight demand paging, not swap, and is expected after load/reload;
the important lifecycle result is that the Metal allocations and residency
identity disappeared while the server process remained alive.

## Same-GGUF llama.cpp denominator

Official llama.cpp `010be9683afabe14ce299197b38c329f94bae568`, Release/native
Metal build, ran the exact artifact at 715 prompt and 128 decode tokens, three
repetitions. Batch was 2048 and ubatch 512; all layers were on Metal.

| engine/path | prefill tok/s | decode tok/s |
|---|---:|---:|
| Runner `86e47bd`, batch 512 | 54.65 | 64.59 |
| llama.cpp Metal 4 tensor | 1,607.70 | 102.76 |
| llama.cpp tensor disabled | 752.65 | 102.27 |

The llama.cpp tensor API is a 2.14x prefill lever at 120B and effectively
irrelevant to decode. More importantly, Runner trails llama.cpp's *non-tensor*
prefill by 13.8x and its tensor path by 29.4x. Metal 4 cannot explain or close
the dominant gap. Runner's Metal MoE prefill census still contains hundreds of
matvec dispatches; batching/grouping expert work is therefore the next
mechanism to investigate, rather than more dense Q4_K tensor tuning. Decode is
closer but still 0.63x llama.cpp.
