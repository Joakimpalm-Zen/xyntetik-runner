# Flash-Next Hub card: replacement block

Card: `Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-GGUF`, README.md as of revision
`401dd406`. Text only: nothing is pushed to the Hub until the owner says
so. This repository has no recovery step. The trunk in `trunk-with-head/`
is ISTA-DASLab's GSQ-RCO IQ3_S bytes unchanged, and the head is the
publisher's weights at `de4b8e4d`. The 27B result belongs to the GSQ-RCO
method, so the card names it as a limit of the trunk. The Flash-Next
trunk's own reference read is pending.

## Status block: a new first bullet, before "Measured status against the parent"

```markdown
> - **Limit first: the trunk's fidelity to its parent is unmeasured, and the same method's 27B file is a learned derivative, not a faithful quantization.** The trunk (ISTA-DASLab's GSQ-RCO IQ3_S, shipped unchanged in `trunk-with-head/`) is a learned post-training quantization: GSQ-RCO fits codes and scales on calibration data. The same method's IQ3_S of Qwen3.8-27B reads 0.5616 / 82.48% and 0.4718 / 86.36% (mean KLD / margin-qualified top-1, two windows) against Qwen's float32 reference, far outside the fidelity bar (KLD at most 0.05, top-1 at least 97%), and predicts the reference corpus better than its own parent (perplexity 53.48 against 73.29). No such read has run on this trunk: its comparison against `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d` is pending. Until it runs, treat the trunk as a derivative of unknown distance from the parent. The head is the publisher's weights; it does not change what the trunk says.
```

## "What this is and is not" paragraph: one sentence appended

```markdown
The trunk is not ours and is not changed here; its distance from the parent
is unmeasured (see the first status bullet).
```

## "The trunk with the head in one set" section: one sentence appended

```markdown
The trunk's fidelity to `Qwen/Qwen3.8-Flash-Next` is unmeasured. The same
method's 27B IQ3_S reads outside the fidelity bar against its float32
reference (R4.32), and a reference read of this trunk is pending.
```

No roster change, because the trunk rows' envelopes already record the
float32 comparison as owed.
