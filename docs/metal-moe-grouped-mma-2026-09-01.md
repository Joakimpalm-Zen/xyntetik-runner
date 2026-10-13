# Metal MoE prefill routes and measured precision

Grouped-MMA MoE prefill is the current default for admitted shapes/types.
The GPU groups (token, expert) slots by selected expert, gathers activation
columns into the dense matrix tile, and scatters outputs to their slots.
Unsupported tensor types retain the existing matvec fallback; single-token
decode does not inherit a prefill route automatically.

`RUNNER_METAL_MOE_MM=0` selects slot-major matvec and is the identity pin.
The default uses FLOAT-staged operands; `=half` selects the measured
half-staged comparison. Grouped reduction and discrete routing carry the
established fidelity/tolerance contract, not scalar byte/logit identity.
`test-moe-mm-ab` compares the actual routes; `scripts/moe-mm-flips.py`
reports routed selections and margin context. Keep route, reduction order,
state ownership, and fallback explicit.

## Scoped measurement

Measured 2026-09-01 on a 128 GB Apple Silicon Mac, interleaved warm runs, 714-token prompt,
batch 512. The FLOAT-staged route measured Qwen3-30B-A3B Q8_0 prefill
170 to 225 tok/s (+31%) and gpt-oss-120b MXFP4 about 72 to 87 tok/s (+21%);
decode was unchanged. These are those measured configurations, not fresh
speed promises for other hardware or a newer binary.

Teacher-forced route comparisons, 24 positions each:

| Model / staging | Top-1 agreement | Margin-qualified top-1 | Mean KLD |
| --- | --- | --- | --- |
| Qwen3-30B, float | 24/24 | 100% | 0.00001 |
| Qwen3-30B, half | 24/24 | 100% | 0.00015 |
| gpt-oss-120b, float | 24/24 | 100% | 0.00987 |
| gpt-oss-120b, half | 24/24 | 100% | 0.01506 |

These passed the recorded 97% margin-qualified / 0.05 mean-KLD bar while
exceeding the dense logit-identity bound through routing sensitivity. A
passed tolerance row is not an exact-identity claim. Current supported
properties and numerical gates are in the [manual](../MANUAL.md#placement-and-memory)
and [compatibility method](compatibility-program.md).
