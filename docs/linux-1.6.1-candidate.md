# Linux 1.6.1 compatibility candidate

This candidate starts at the v1.6.0 tag, commit
`869d72e89a74e2d147ceadf5400e10d9fe9e877c`. Apart from the version constant,
its engine sources are unchanged. Subsequent main training optimizations are
excluded. The maintenance PR targets `release/1.6.x`, which starts at that
tag, so merging it does not revert current main.

The packaging fix pins Ubuntu 24.04 and refuses imported GLIBC versions
above 2.38, preserving the measured 1.5.0 ABI ceiling. BUILD-INFO records
compiler, libc and flags. The separate Linux candidate workflow records the
hosted image version too and uploads CI artifacts without creating a release,
pushing a tag or publishing a container.

Candidate CI checks the real ELF gate tests, release-flags build, version,
SHA256SUMS, archive extraction, imported symbol ceiling, startup and fixture
inference on Ubuntu 24.04. It repeats startup and inference with a checksum-
verified isolated GLIBC 2.38 loader/library set. These are compatibility
smokes, not publisher-reference scoring or certification of arbitrary Linux
distributions. The checksum file authenticates consistency with this CI run;
a published release's SHA256SUMS must be verified again after publication.

The candidate includes a new [per-version compatibility report](compat-reports/1.6.1-2026-10-25-macos-arm64-gemma-4-e4b-it-qat-q4_0.json):
the checksum-pinned Gemma-4 E4B QAT Q4_0 file passed load and chat on the
1.6.1 macOS CPU build. Its paging warning and slow timings are retained.
This is a new measurement, not a relabeled 1.6.0 report, and proves neither
Linux ABI compatibility nor Golden fidelity. Local release-check passes;
current device and tier evidence must be checked again before any tag.

The five Golden instrument rows are Qwen3-0.6B bf16, granite-4.2-3B bf16,
Gemma-3-4B bf16, Apertus-8B bf16, and Trinity Nano served. Reuse the existing
inputs, controls, positions and scoring settings; preserve failures and dropped
positions. Follow with the full 23-row candidate asset pass before asserting
its headline count. A later published binary must be tied to the validated
candidate by checksum, or validated independently.

The published 1.5.0 binary's 20/23 and local Clang native 1.6.0 build's 19/23
remain separate measurements. Published 1.6.0 scoring and this candidate's
headline count are unknown. Compiler and architecture each independently
change scores; no source scoring regression is confirmed and all-row source
equivalence remains unproven. See the retained
[Golden Pass](golden-pass-2026-10-25.md) and
[build controls](golden-pass-evidence/gold-2k/toolchain/README.md).

README impact: the version banner identifies the candidate and the measured
fidelity claims stay scoped to their binaries. The manual and site carry the
same qualifications. External model-card updates belong to the release
review; this candidate does not publish model files, cards or adapters.
