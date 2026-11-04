# Flash-Next Hub card, variant B (recovery): replacement block

Card: `Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF`, README.md as of revision
`401dd406`. In variant B the 27B divergence is the project's scale
recovery. No recovery was run on anything in this repository: the trunk in
`trunk-with-head/` is ISTA-DASLab's GSQ-RCO IQ3_S bytes unchanged, and the
head is the publisher's weights at `de4b8e4d`. The card says so, because
the withdrawn 27B file came from the same program, and keeps its own
reference read pending.

## Status block: a new first bullet, before "Measured status against the parent"

```markdown
> - **Limit first: the trunk's fidelity to its parent is unmeasured.** The trunk (ISTA-DASLab's GSQ-RCO IQ3_S, shipped unchanged in `trunk-with-head/`) has no float32 reference read yet: against `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d` it is owed. The project's scale-recovered Qwen3.8-27B file was withdrawn on [date] because the recovery moved it outside the fidelity bar. Nothing in this repository was recovered or retrained: the trunk's tensor bytes are ISTA-DASLab's, and the head is the publisher's weights.
```

## "The trunk with the head in one set" section: one sentence appended

```markdown
No scale recovery or other training touched these parts. The 27B withdrawal
(R4.32) concerns a recovered file and does not apply here. The trunk's own
reference read is pending.
```

No roster change in variant B.
