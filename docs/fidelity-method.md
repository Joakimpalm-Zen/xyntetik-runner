# Fidelity: what the measurement says

Start with the limits the experiments found:

- In the historical parent-relative read, Muse-Glimmer-30B's healed 4.75%
  BF16 configuration passed its fidelity bar at mean KLD
  **0.03056** and margin-qualified agreement **98.64%**, yet lost **0.916
  HellaSwag points**, outside its paired equivalence margin of **0.5 points**.
  A numerical bar pass did not preserve that task's score.
  [Public benchmark-annex record](https://huggingface.co/datasets/Joakimpalm-Zen/Muse-Glimmer-30B-surgery-report#12-the-benchmark-annex-we-tested-our-own-bar-and-it-failed-2026-08-29).
- The derived Qwen3-30B-A3B file was **0.88 GB smaller** than the hand-built
  selective file, but slightly further from the publisher's float32 model:
  mean KLD **0.0290 versus 0.0279**. Their margin-qualified agreement was not
  distinguishable in the paired comparison. The files differ in size;
  this was **not an equal-byte-budget experiment**.
  [Three-file reference report](compat-reports/qwen3-30b-a3b-derive-2026-11-02/README.md).

## Measure a file, reference and execution configuration

Use the publisher's reference implementation in float32 as the numerical
anchor, with matching token IDs and framing. BF16 checkpoint weights can
be upcast at use while computing in float32, as in the
[reference report](compat-reports/qwen3-30b-a3b-derive-2026-11-02/README.md).
State the artifact/reference hashes, build, backend, scoring route, corpus,
positions and context. An adapter or another quantization needs its own
relevant evidence. When the publisher reference is unavailable, label an
engine/parent-file comparison as agreement, rather than publisher fidelity.

Report both **margin-qualified top-1 agreement** and **mean KLD**.
Agreement counts matching leading tokens on positions whose reference
top-two separation meets the instrument's margin; report the qualifying
denominator and plain top-1 too. KLD describes the scored next-token
distributions, so matching leading tokens alone can miss distributional
drift. [Instrument](../scripts/gold-logits.py),
[reference results](compat-reports/qwen3-30b-a3b-derive-2026-11-02/README.md).

The cited float32 comparison uses a bar of **at least 97%** margin-qualified
agreement **and mean KLD at most 0.05**, over **2,000 positions at stride 2**
of the same corpus. It reports the window and paired comparisons, not a
universal sample-size or quantization guarantee.
[Method and per-position records](compat-reports/qwen3-30b-a3b-derive-2026-11-02/README.md#against-the-reference).

## Keep the claim inside that evidence

A pass concerns the named file, reference, configuration and measured window.
It does not establish benchmark equivalence or a pass for another artifact
or execution configuration.
Agreement with a parent also cannot distinguish improvement from damage;
evaluate intended tasks against their actual labels, using paired results
where the same items are compared. The
[benchmark annex](https://huggingface.co/datasets/Joakimpalm-Zen/Muse-Glimmer-30B-surgery-report#12-the-benchmark-annex-we-tested-our-own-bar-and-it-failed-2026-08-29)
shows why those task results need their own evidence.
