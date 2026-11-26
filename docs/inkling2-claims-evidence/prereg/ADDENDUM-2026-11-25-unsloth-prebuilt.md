# R4.48.12 pre-registration: addendum, 2026-11-25, Unsloth's column runs its PUBLISHED CPU prebuilt (the owner's decision)

Written before any run of the prebuilt. No Unsloth output exists. Runner's and mainline llama.cpp's Claim 3 arms are done, and their Claim 2 sixteen-thread cells started at 18:02:39Z. Nothing in this addendum changes them.

## Why

`ADDENDUM-2026-11-24-unsloth.md` built Unsloth's engine from source at the release tag's commit `a5aa5bacbc2bda2478ee2985d42f36211ffc7da9`, assuming that build is what Unsloth ships. The lab's Claim 3 pre-flight on 2026-11-25 showed the assumption is wrong:
- That source build (`llama-server` `4377f77e`) exits at load with `unknown model architecture: 'inkling2_loop'`. The fork's master holds no Inkling code; its `scripts/unsloth/upstream-sync.json` records the last upstream sync as `b11887` (2026-10-25), before upstream added Inkling 2.
- Unsloth's published binaries are built differently: the release workflow takes an upstream tag (here `b12649`) and merges a set of PR pins in CI. The source commit alone does not reproduce them.

## The change (the owner, 2026-11-25: "Try their shipped binary")

1. **The binary.** Unsloth's column uses the PUBLISHED CPU prebuilt of the same release, `app-b12649-mix-c7059d6-linux-x64-cpu.tar.gz` (the asset the earlier addendum already named for the speed sets).
   - It is fetched by its asset sha256, recorded at fetch together with its size; the lab identified `510597b6...`, 19,227,234 bytes.
   - The archive's file list and the `llama-server` binary's sha256 are recorded, along with its `--version` output.
   - It is run as published: its compiler and flags are Unsloth's, not the house's configure line. Reports state this as a difference from mainline's source build.
2. **Everything else is unchanged:** the same file `2da51788`, the same settings, clients and request bodies as mainline's column for each claim, and the same pre-flight against the reference's prompt ids before any counted cell.
3. **Order on the box** (one model process at a time):
   - first, after the running Claim 2 cells: the prebuilt's load and Claim 3 pre-flight;
   - then its two Claim 3 arms (unsloth-off, unsloth-rb);
   - then its eight Claim 2 sixteen-thread cells;
   - then its Claim 1 run and teacher-forced verification, if Claim 1 has no Unsloth records yet.
4. **If the prebuilt also fails to load the model, or fails a pre-flight,** the Unsloth column is reported, for every claim, as "Unsloth's published build at b12649-mix-c7059d6 does not run Inkling 2 Mini". The exact error is quoted, and nothing is substituted.
5. **The source-build attempt stays in the record.** Its server log and the failed pre-flight are kept, and reported beside the result.

## Clarifications, 2026-11-25, written before any Unsloth pre-flight or output

These were asked by the lab and decided before the prebuilt rendered or answered any prompt. They change no threshold or claim wording.

1. **What "fails a pre-flight" in item 4 means.** It means the pre-flight cannot be completed (the prebuilt cannot load the file, render or tokenize), or a claim's own pre-flight stop rule fires. Each claim keeps the rule mainline had:
   - **Claim 1:** the 2026-11-24 addendum's per-prompt rule stands. A rendered prompt whose ids differ from the reference's fails that prompt under PREREG 1.3, logged before its run, exactly as mainline's c1_011 (llama.cpp's BPE ignoring `ignore_merges`) was. A per-prompt difference does not void the column.
   - **Claims 2 and 3:** any difference stops that claim's Unsloth cells before they run, as ruled for mainline, and the claim's Unsloth column is reported as not run, with the differing prompts listed.
2. **Claim 1's verification reuses already-verified sequences.** A sequence is reused only when its key matches an erratum-3 verified sequence exactly; the key is the full prompt ids, the full generated ids and the verifier version. This is the reuse erratum 3 used. Only sequences with no exact match run a forward. Two checks guard it:
   - five reused keys, chosen by a seed recorded before the run, are re-verified fresh and must give identical verdicts;
   - the Runner and mainline entries in the three-engine output must equal erratum 3's per-prompt verdicts exactly.

   If either check fails, the reuse is dropped and Unsloth's sequences are all verified fresh.
3. **Records before verification.** Unsloth's Claim 1 `answers.jsonl` and its NOTES are committed to this repository first. The verification starts only on the committed file's sha256, as the 2026-11-24 addendum requires.
