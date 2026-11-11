# Measured-envelope format

`scripts/certify-envelope.py` writes a `<model>.envelope.json` sidecar using
`xyntetik.runner.envelope.v1`. It indexes artifact/reference hashes,
actual runtime/backend/build/configuration, measured checks, and a verdict.
Models are not stored in this repository.

```sh
python3 scripts/certify-envelope.py --model model.gguf --runner ./runner   --compat-report checked.json --reference-sha <reference-sha256>   --gpu off --out model.gguf.envelope.json
```

Use the helper's `--help` for context/KV/thread, derivation, measurement-date,
truncation/fidelity, and serving-guidance inputs. The report must describe
the actual artifact and executed checks. `certified` requires the relevant
passing gate and reference identity; `outside-envelope` records a measured
failure, while `experimental` means no qualifying gate was indexed.

Evidence binds by the report's `sha256`, `expected_sha256` and
`actual_sha256` fields: every recorded hash for the selected entry must
equal the artifact's computed SHA-256. A conflicting same-name entry is
refused with both the artifact and report hashes; disagreeing expected and
actual hashes are evidence for neither file. A hash match may use a different
file name. Only an entry with no recorded hash may fall back to its basename.
The additive `quality.evidence_match` records `method: "sha256"` and
`hash_verified: true`, or `method: "name"` and `hash_verified: false` with a
reason. It is null when no report entry matched. A name-only match is not
hash-verified evidence, even though the sidecar names the current artifact's
own hash.

The load-time gate reads the sidecar beside the GGUF and distinguishes its
runtime tuple from the running build. Do not change an old record's version,
date, hash, or verdict to make it appear measured for another build.
Records and served receipts bind the observed verdict and manifest digest.
See the [manual](../../MANUAL.md#record-and-verify-a-run) for admission,
signature, and enforcement behavior.

## Optional tool-calling evidence

`tool_calling` is additive and reported-only. Its truncation recovery,
schema shape, agent-torture, native protocol, and rollup fields describe
whatever evidence was supplied; absent subchecks remain absent. This block
does not change the top-level load verdict.

Truncation measured on a proxy model establishes an engine behavior, not
that every artifact chooses useful arguments. Schema-conformant calls are
not proof of correct argument values or calibrated confidence. An adapter,
quantization, cache layout, backend, or execution mode needs its own relevant
recorded evidence. The [evaluation/lineage walkthrough](../lineage-walkthrough.md)
and [compatibility method](../compatibility-program.md) describe those boundaries.
