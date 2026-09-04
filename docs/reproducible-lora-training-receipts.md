# Reproducible LoRA training with receipts

**Question.** Is LoRA training reproducible, and can I prove which adapter
came from which run, which data and which base?

**Answer.** On the same build, yes, byte for byte: `--train` with the same
data, seed and configuration writes the same adapter file, verified by
sha256 in CI. Every adapter is written with a `.train.json` sidecar that
records the base, data and adapter sha256, the seed, the full configuration,
and the binary sha, compiler, OS and architecture that produced it. That
sidecar is the receipt for the training run.

## What "reproducible" means here, exactly

Runner's contract is written down with named edges in
[docs/determinism-scope.md](determinism-scope.md). The short form:

- **Claimed and gated in CI.** Same executable, same inputs, same sampled
  tokens. Training byte-determinism on the same build. Quantization and
  merge write the same output bytes for the same inputs on the same build.
- **Not claimed.** Independent rebuilds (another compiler, version or ISA)
  are not byte-identical; cross-engine token identity is not promised;
  numerical identity of hidden states is never promised.

Page titles elsewhere say "deterministic". The claim Runner makes is the
narrower one above: reproducible on the same build, and receipted so that
a later reader can check what ran.

## Receipts for inference runs

Recorded runs carry the same discipline. `--transcript FILE` records a
one-shot run as a signed record: model, adapter and binary hashes, the
effective execution profile, the exact seed and sampling configuration,
prompt and output token ids, and the exact streamed output bytes.
`--verify FILE` replays it and exits with one of three verdicts:
`VERIFIED` (0), `DIVERGED` at a token or output byte (2), or
`UNVERIFIABLE` for an invalid record or an artifact mismatch (3).

- **T1, same binary.** Bit-exact replay of the recorded run.
- **T2, cross-ISA.** Token-level replay on another machine or architecture;
  floating-point internals may differ, and the boundary is libm.

Receipts chain (`--transcript-prev`), are signed with an Ed25519 key you
generate (`--keygen`, `--sign-key`), and the loaded model can be verified
against an OpenSSF Model Signing bundle (`--model-sig`, `--model-pubkey`).
Flag rows are in the [command-line reference](../README.md#command-line-reference).

## Measured

The public artifact
[Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M)
was produced twice, independently, with the same sha256 on the adapter file
(measured 2026-08-22). Its card carries the sidecar and the training
command. Training runs on the CPU path; the dominant matvec of the backward
runs on CUDA behind a flag, and CUDA training end to end is open work.
