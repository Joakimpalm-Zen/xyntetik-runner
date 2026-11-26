# Site release copy (draft only)

Runner adds Inkling 2 Mini with fast CPU inference by default and an opt-in
resident CUDA decode step, alongside OLMo 2 and OLMo 3 support on CPU, Metal
and CUDA with CPU LoRA training. Inkling's CPU timing remains below
llama.cpp's on the measured file, its CUDA speed row covers one aligned
build, and the available Windows device's 12 GB cannot hold either Inkling
file; the 7B OLMo file also exceeds the 8 GB Metal device's memory. The
[file measurements](../../model-support.md) and [release notes](RELEASE-NOTES.md)
keep those limits beside the claims.

On Claim 1's pre-registered 100 prompts, Runner matched the publisher's
complete greedy sequence on 40/100 versus llama.cpp's 20/100 and the
additional Unsloth column's 16/100. This is greedy identity, not task quality,
and includes 58 incomplete Runner records whose v1 client omitted the stop
token. Beside it, **erratum 3, decided after the verdicts**, reports Runner
94/100 versus 20/100 and 16/100 after completing and verifying the original
records; it does not replace the pre-registered result.

At sixteen threads, Runner's measured target outputs were identical on
20/20 prompts across its two measured cells; mainline llama.cpp and the
additional Unsloth column were each 2/20 across their eight measured cells.
This is the **sixteen-thread half, provisional**, with one-thread cells
after the release. Runner's Inkling CPU decode is sequential; its page
eviction makes the contextual timing rows information, not a speed claim.

Claim 3's pre-registered outcome is **TIE**: Runner with loop guard and
budget gives 37/40 structurally usable turns, zero observed loops, versus
budgeted llama.cpp's 39/40; the additional Unsloth budgeted column gives
37/40 and two loop/exhaustion turns, without a combined outcome label.
The loop guard fired on zero combined-arm turns and the budget forced
18 closes. `unit_test_pass` was not computed because model-written code
had no isolated execution environment.

Unsloth's columns use its published CPU prebuilt `b12649-mix-c7059d6`,
run as published. A source build at the release tag commit could not load
Inkling 2 Mini. The [release notes](RELEASE-NOTES.md) carry intervals,
pairwise results, the full timing/load/residency table and the limitations.

On a 64-core x86-64 CPU (AVX-512), the measured OLMo 3 7B Q8_0 CPU rows
put Runner at 0.33x llama.cpp for prefill 2048, 0.40x for prefill 512 and
0.85x for decode. Runner's release x86-64-v3 build and llama.cpp's native
AVX-512 build use different batching, sampling and prompts; the 1B CPU row
is missing. [The benchmark scope and context](../../olmo2-cpu-speed.md)
stay beside those figures.

This is the proposed claim-first site passage, not an edit to the deployed
site. Apply only after the release decision. The live site still describes
the latest published release; a local preview of this candidate is not a
site deployment. The cut date in the candidate CHANGELOG is provisional.
