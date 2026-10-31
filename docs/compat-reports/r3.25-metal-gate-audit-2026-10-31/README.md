# R3.25, Metal half: structural mutants against the device identity gates, 2026-10-31

**Question.** Does each tolerance-bounded identity gate that `make test` runs on Metal catch a real mistake in a
Metal kernel? The trigger was R4.10: on the toy Mamba-2 fixture, a scan with its D skip dropped passed
test-gpu-identity's cross-backend bound.

**Method.** `mutant-audit.py`, run from the runner tree on the Metal test host:
1. Apply one structural mutant to `src/kernels.metal` or `src/metal.m`, and check that it applied.
2. Rebuild, then run every gate below. The script then restores the files and rebuilds.

Mutants:
- the RMS norm's weight ignored;
- its eps dropped;
- the q/k norm's weight ignored;
- SiLU dropped from the gated FFN;
- rope skipped in the prefill kernel;
- rope one position off, in prefill and, separately, at decode;
- the attention scale ×1.05;
- the matvec bias dropped;
- the sigmoid output gate dropped.

`test-mv-tol` and `test-moe-tol` skip on this host (their default models are not present) and are not in the table.

## Pass 1: the gates as they were (`pass1-cross-backend-bound.log`)

Healthy residue on the toy fixtures is 2e-8 to 2e-6 of the logit range; the bound is 2e-3.
- **Caught:** SiLU dropped (6 of 13 gates), and the bias dropped (only gpt-oss-top1, the one fixture with biases).
- **Passed every gate, a blind bound:** rope skipped (7.9e-4), rope one position off (1.7e-4), the attention scale
  ×1.05 (5.8e-5) and eps dropped (7.2e-4).
- **Invisible at any bound, a fixture blindness:**
  - the RMS norm weight ignored: every fixture's norm weights are ones;
  - the q/k norm weight ignored: no fixture in the list had q/k norms;
  - the sigmoid gate dropped: no fixture in the list had the gate.

## The repair

- **Tight bounds on Metal.** `make test` holds the toy fixtures to `GPU_ID_LIMIT=1e-5`, and test-spark to 2e-5
  (its residue is 2e-6). The cross-backend 2e-3 stays for a CUDA host until R3.25's CUDA half measures that
  backend's residue (Makefile `METAL_ID_TIGHT`). `GPU_ID_LIMIT` can only tighten (R4.10, #514).
- **A fixture whose norms matter.** `make-test-model.py --noisy-norms` draws the output, attention and FFN norm
  weights from 1 ± 0.5; the default fixture is byte-identical. `test-noisy-norms.gguf` (with `--qk-norm`) runs in
  `make test` at 1e-5.
- **The sigmoid gate** is covered by the qwen35 fixture (`test-metal-qwen35`, since R4.10), and pass 2 includes it.

## Pass 2: the repaired gates (`pass2-tightened.log`)

Every mutant fails at least one gate, and the unmutated baseline passes every tightened bound.

| mutant | caught by |
|---|---|
| RMS norm weight ignored | noisy-norms (0.029) |
| RMS norm eps dropped | 11 of 13 (5e-4 to 1e-3) |
| q/k norm weight ignored | noisy-norms (0.039) |
| SiLU dropped | 7 of 13 (0.015 to 0.018) |
| rope skipped (prefill) | 8 of 13 (1.7e-5 to 7.2e-3) |
| rope one position off (prefill) | 6 of 13 (6e-5 to 2e-4) |
| rope one position off (decode) | 7 of 13 (6e-5 to 2.1e-3) |
| attention scale ×1.05 | 11 of 13 (5.8e-5 to 1.1e-3) |
| matvec bias dropped | gpt-oss-top1 (2.3e-3) |
| sigmoid gate dropped | qwen35 (0.012) |

Two non-catches, explained:
- The decode-only run (`test.gguf 0 1`) does not see either rope mutant. The prefill kernel never runs there. And
  shifting every position by one at decode shifts q and k alike, which RoPE's relative form leaves unchanged: on
  that path the mutant is an equivalent program, not a missed mistake. The mixed prefill-and-decode runs catch both.
- `test-attn-tol` catches none of them, by design. It compares two Metal attention routes, which share the
  mutated inputs.
