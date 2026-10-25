# Controlled build comparison and Linux release ABI, 2026-10-25

The 23-row Golden Pass compares a Clang 23.1.4 `-march=native` build of `33c2a08f` with the published GCC 13.3 `-march=x86-64-v3` v1.5.0 binary. It changes compiler, CPU target, build sysroot and source together. The previous assertion that *every* difference was caused by the compiler alone was not established.

## Controlled probes

Five fresh binaries were built in isolated tag checkouts. All use explicit `-O3 -ffast-math -std=gnu11 -Wall -Wextra -Wno-unused-parameter`, the normal CUDA backend (CPU selected at runtime), no T3, and either `-march=x86-64-v3` or `-march=native`. GCC is Ubuntu 13.3.0-6ubuntu2~24.04.1; Clang is 23.1.4. The controlled Clang invocation disables the Conda config (`--no-default-config --sysroot=/`), using the same system headers, linker and library search baseline as GCC. All probes run sequentially on the same CPU/GLIBC 2.39 environment with GPU hidden, `--gpu off -t 16 -c 4096 --score --score-top 8`, on the same first 6,000 corpus bytes. There is no publisher-reference inference in these probes: equality measures implementation equivalence, not absolute correctness.

The compiler may generate different library calls: GCC uses system libmvec in addition to libm/libc; Clang uses libm/libc. This is an outcome of changing the compiler, with the installed library versions held fixed. The earlier Conda-configured Clang binaries also resolve libm/libc from the host at runtime.

| Comparison | Qwen3-0.6B bf16: differing logprob positions / 1,529 | Trinity Nano Q8_0: differing logprob positions / 1,438 |
|---|---|---|
| GCC v1.5.0 vs GCC v1.6.0, release flags | 0 | 0 |
| Local GCC v1.5.0 vs published v1.5.0 | 0 | 0 |
| GCC vs Clang, v1.5.0, x86-64-v3/system sysroot | 1475 | 1434 |
| Clang x86-64-v3 vs native, v1.5.0/system sysroot | 1470 | 1436 |
| GCC x86-64-v3 vs native, v1.5.0 | 1479 | 1434 |
| Clang native system vs Conda config, v1.5.0 | 0 | 0 |

GCC v1.5.0 vs GCC v1.6.0 and system-vs-Conda Clang native have byte-identical complete score JSON, not merely equal aggregates. The locally rebuilt GCC v1.5.0 binary is byte-identical to the published asset itself: sha256 `4dfc46845050c73aebffd37b91baf90517d55a5b0d6019c3085fe7bfa75109c9`. This confirms that source and release build settings reproduce the baseline. GCC vs Clang at fixed x86-64-v3 changes Trinity argmax at 46/1,438 positions; changing Clang from x86-64-v3 to native changes it at 38/1,438. These changes are retained, not described as harmless.

Confirmed: compiler choice and architecture flags each independently affect these model outputs; source changes between the two tags do not affect these probes under GCC 13.3/release flags. System-vs-Conda sysroot configuration produces no score change in these two Clang native probes. The score probes are prefixes, not the complete 2,000-position Golden instrument. They do not apportion the 23-row KL/count changes between compiler and target or prove all 23 rows source-equivalent. No source-code correctness regression is confirmed by this investigation. The published GCC 15.2 v1.6.0 binary remains unscored.

## Earlier bisect, correctly scoped

The old work considered a 198-commit interval but actually sampled eight commits, plus two anchors, on Qwen3-0.6B bf16. All eight sampled Clang builds and the endpoint comparison match at 2,000 KL positions. The supposed bad endpoint had already tested identical: the bisect precondition was false, so its printed “first bad commit” is invalid. It neither tested all 198 commits nor isolates a regression commit. The anchors show Clang v1.5.0 vs Clang `33c2a08f`: 0/2,000 KL differences; Clang v1.5.0 vs the published v1.5.0 asset: 2,000/2,000. Earlier CLI JSON is also byte-identical between Clang v1.5.0 and Clang `33c2a08f` on both prefixes.

## GLIBC regression and sustainable fix

The SHA256SUMS-verified published assets have maximum imported symbol versions **GLIBC_2.38** (v1.5.0, GCC 13.3) and **GLIBC_2.43** (v1.6.0, GCC 15.2.0-16ubuntu1). The sole 2.43 import is `sqrtf`. On Ubuntu 24.04 / GLIBC 2.39, v1.5.0 starts; v1.6.0 fails in the dynamic loader before main. Both release archives record `ubuntu-latest` and x86-64-v3; their workflow build command is unchanged.

The concrete packaging defect is linking against a newer libm ABI with no pinned Linux baseline or symbol-version gate. GCC version by itself does not impose a GLIBC minimum. GitHub documents the gradual [ubuntu-latest migration from Ubuntu 24.04 to 26.04 between October 19 and November 19, 2026](https://github.blog/changelog/2026-09-17-ubuntu-26-generally-available-and-latest-migration/). The compiler stamp and timing are consistent with that migration; exact hosted image IDs were not recorded in BUILD-INFO and authenticated job logs were unavailable. The exact image ID remains unverified, rather than inferred as a measured fact.

All installations lacking GLIBC_2.43 are excluded by this asset. The newly excluded ABI range is 2.38–2.42, including [Ubuntu 24.04 (2.39)](https://packages.ubuntu.com/noble/libc6) and [Debian 13 (2.41)](https://packages.debian.org/trixie/libc6). These are ABI impact statements, not a new certification of every distro. Ubuntu 22.04 / GLIBC 2.35 and Debian 12 / GLIBC 2.36 were already below the previous loose binary’s 2.38 floor; do not present them as new regressions. The Docker image builds independently on Debian bookworm and is not evidence about the loose release asset.

The prepared workflow change pins the Linux build to `ubuntu-24.04`, rejects imported symbols above **2.38** before packaging, and adds compiler/libc/flags to BUILD-INFO. Keeping 2.38 rather than allowing 2.39 preserves the prior measured ABI floor. Tests exercise real ELF versioned imports, including acceptance at 2.38 and rejection at 2.43, numeric ordering, GLIBC_PRIVATE and invalid input. The real published v1.6.0 asset fails the gate; v1.5.0 and the local release-flag v1.6.0 rebuild pass.

The rebuilt v1.6.0 binary (sha256 `dc8b38edcef8ad46e72985e9e0922e3b06ab17e717ce04d694e0b2c08bb5c1cb`) passes version/startup and fixture inference both on the host GLIBC 2.39 and an isolated GLIBC 2.38 loader/library set extracted from Ubuntu’s `libc6_2.38-1ubuntu6.3_amd64.deb`. This isolated loader is a test, not a production compatibility workaround; no ELF patching was used. The package [comes from the Ubuntu archive](https://old-releases.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.38-1ubuntu6.3_amd64.deb). No replacement release was published or merged.

## Evidence and remaining validation

- [Build commands, source commits and binary hashes](builds.json). `g` = GCC, `c` = Clang; `150`/`160` = tag; `v3`/`native` = target. [Machine-readable probe comparison](controlled-comparison.json); complete score outputs are `score-*.json.gz` beside this file. [Reproduction script](reproduce.py).
- [Published/rebuilt ELF requirements, checksums and startup results](asset-requirements.json); archive BUILD-INFO files are beside this report. [GLIBC 2.38 smoke](glibc238-smoke.json) and [runtime package checksum](glibc238-package.sha256).
- [Eight historical bisect probes](prior-bisect-probes.json); earlier anchors and CLI outputs are `prior-score-*.json.gz`. The original full logs remain in the recovered lab artifacts.

A corrected published release is still required. The public latest release and tags checked on 2026-10-25 expose v1.6.0, not v1.6.1. Once a corrected asset exists, verify its checksum, ABI/startup and re-score the four significant bf16 rows plus Trinity served as the existing review requested. A full 23-row asset pass remains necessary before claiming its exact headline count. No outcome is inferred for that missing asset.
