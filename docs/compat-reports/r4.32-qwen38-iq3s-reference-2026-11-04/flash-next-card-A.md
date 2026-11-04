# Flash-Next Hub card, variant A (GSQ-RCO): replacement block

Card: `Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF`, README.md as of revision
`401dd406`. This repository has no recovery step. The trunk in
`trunk-with-head/` is ISTA-DASLab's GSQ-RCO IQ3_S bytes unchanged, and the
head is the publisher's weights at `de4b8e4d`. In variant A the 27B finding
belongs to the GSQ-RCO method, so the card names it as a limit of the
trunk. The Flash-Next trunk's own reference read is still pending.

## Status block: a new first bullet, before "Measured status against the parent"

```markdown
> - **Limit first: the trunk's fidelity to its parent is unmeasured, and its 27B sibling diverges.** The trunk (ISTA-DASLab's GSQ-RCO IQ3_S, shipped unchanged in `trunk-with-head/`) is a learned post-training quantization fitted on calibration data. Its sibling, ISTA-DASLab's GSQ-RCO IQ3_S of Qwen3.8-27B, reads mean KLD [lab: source 27B KLD] and margin-qualified top-1 [lab] against Qwen's float32 reference, outside the fidelity bar, and predicts the reference corpus better than its own parent. No such read has run on this trunk: against `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d` it is owed. Until it runs, treat the trunk as a quantization of unknown distance from the parent. The head is the publisher's weights; it does not change what the trunk says.
```

## "What this is and is not" paragraph: one sentence appended

```markdown
The trunk is not ours and is not changed here; its distance from the parent
is unmeasured (see the first status bullet).
```

## "The trunk with the head in one set" section: one sentence appended

```markdown
The trunk's fidelity to `Qwen/Qwen3.8-Flash-Next` is unmeasured. The GSQ-RCO
IQ3_S of the 27B sibling reads outside the fidelity bar against its
float32 reference (R4.32), and a reference read of this trunk is pending.
```

No roster change in variant A, because the trunk row's envelope already
records the float32 comparison as owed. The claim of
`qwen3.8-flash-next-gsq-rco-iq3_s-with-mtp-head` can add one clause once
the owner decides: "the GSQ-RCO method's 27B sibling reads outside the
fidelity bar against its float32 reference".
