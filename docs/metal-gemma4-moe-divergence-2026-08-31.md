# Gemma MoE activation and backend checks

The routed Gemma FFN has its own activation/layout semantics. Preserve
its admitted dual-branch shape, weight scale, and the saturated activation
branches that avoid overflow/NaNs on sufficiently negative or positive
inputs. A dense fixture whose activation never reaches those boundaries
does not verify the routed path.

`make test-metal-gemma4-moe` and the related MoE/activation/numerical gates
exercise the current implementation. Check partial/full placement, actual
kernel engagement, decisive logits, and invalid values rather than accepting
an empty decoded response or a constant output token as a valid generation.
Allocation/admission failure must name the actual unsupported property and
use the documented fallback/refusal.

The [backend table](../MANUAL.md#placement-and-memory), [model evidence](model-evidence.md),
and [verification method](compatibility-program.md) define current support
and route-specific identity/tolerance obligations. They do not certify an
unmeasured checkpoint by its family name.
