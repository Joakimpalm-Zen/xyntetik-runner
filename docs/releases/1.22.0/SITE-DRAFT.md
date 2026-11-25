# Site release copy (draft only)

Runner adds Inkling 2 Mini with fast CPU inference by default and an opt-in
resident CUDA decode step, alongside OLMo 2 and OLMo 3 support on CPU, Metal
and CUDA with CPU LoRA training. Inkling's CPU timing remains below
llama.cpp's on the measured file, its CUDA speed row covers one aligned
build, and the available Windows device's 12 GB cannot hold either Inkling
file; the 7B OLMo file also exceeds the 8 GB Metal device's memory. The
[file measurements](../../model-support.md) and [release notes](RELEASE-NOTES.md)
keep those limits beside the claims.

**PLACEHOLDER-CLAIM-2:** final alone-versus-batched result, with its limits.

**PLACEHOLDER-CLAIM-3:** final reasoning-loop result, with its controls and limits.

**PLACEHOLDER-OLMO2-CPU-SPEED:** completed CPU rows and their scope.

This is the proposed claim-first site passage, not an edit to the deployed
site. Apply only after the release decision. The live site still describes
the latest published release; a local preview of this candidate is not a
site deployment. The cut date in the candidate CHANGELOG is provisional.
