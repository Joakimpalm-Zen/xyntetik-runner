# Agent protocol schema: premise check and design choices

Status: proposal, 2026-11-11. Checked Runner main `2750e93a` and Suite
`4e2f93d3`. This note adds no schema, card field, runtime behavior or upload.
Design notes currently live directly under `docs/`.

## What exists

| Surface | What it describes | What it does not establish |
|---|---|---|
| [Capability inventory](capability-inventory.md) | Generated inventory of engine flags, routes, formats and their owners. | A particular model's measured competence. |
| [GGUF agent profile](agent-profile.md), [`bind_agent_profile`](../src/model.c) | Opt-in protocol/tokenizer versions, schema ID/digest and required runtime features. Versions must be 1; unknown required features fail. | The named schema's contents or authenticity: the loader checks a nonempty ID and digest syntax, without resolving or hashing a schema file. |
| [`GET /v1/capabilities`](../src/server.c) | Resident model, effective template, `tool_protocol.family/native`, sampling, engine features and optional `agent_profile`. | A measured native-tool-calling pass. Engine features and profile requirements are not execution permissions. |
| [`--info tools`](../src/main.c), [`tool_protocol_name`](../src/template.c) | The family selected by the template layer, also used by serving and request telemetry. | Evidence that a particular file chose useful or correct calls. A second registry of family grammars would duplicate this source. |
| [Measured envelope](envelope-manifests/README.md), [`certify-envelope.py`](../scripts/certify-envelope.py) | `xyntetik.runner.envelope.v1`: artifact/reference identity, runtime/configuration, measurements and verdict; optional tool-calling subchecks. | A standardized Suite skill-capability assertion. Tool-calling is reported-only and does not alter the load verdict. |
| [Server provenance](../src/provenance.h) | Running binary/model/adapter identity and load-time envelope verdict. Model identity may still be hashing or unavailable after a file change. | The full measured row, a schema resolver, or independent attestation. |

The optional envelope's `native_tool_protocol` is obtained from tool-info;
it is a declaration, not a measured check. The rollup instead uses
truncation recovery, schema shape and agent-torture evidence. A generic
protocol can pass that rollup. Neither `native: true`, `certified`, nor the
rollup alone means "this artifact has measured native tool calling."

The envelope machinery is a substantial existing format, so the old claim
that no measured-row format exists is too broad. It is not yet the Suite's
capability contract. In addition, `find_model_evidence` currently falls back
from SHA lookup to a basename match, while the output gets the current
artifact's SHA. That fallback does not prove the source report's identity;
its correction is separate work, not a schema feature.

## What the Suite needs

The two blocked Suite tasks are capability authorization at activation
(R6.4.1) and ingestion of measured rows (R6.4.2). At the checked revision:

- The Runner-control probe already retains `tool_protocol`, template,
  sampling and engine features. It does not retain `agent_profile` or ingest
  tool-calling measurements.
- `Skills::authorize` compares required capabilities against a supplied
  held set and optionally checks available tools. Only tests call it. The
  skill parser maps legacy capabilities such as filesystem/process access;
  it has no model-native-tool-calling requirement. Passing engine features
  into that held set would confuse model competence with permissions.
- The measured-row reader must identify the artifact and relevant adapter,
  template/protocol, build/backend and execution configuration from actual
  evidence. A model alias, filename, family or card assertion is insufficient.
  Existing rows can supply evidence, but missing identity/scope must remain
  unmeasured; the envelope's current basename fallback must not grant a pass.
- The activation path needs an explicit mapping from a requested model
  capability to qualifying measured checks, separate from tool availability
  and execution authority. For a required measured capability, missing,
  foreign, malformed or failing evidence must not populate the held set.
  This is a proposed Suite admission rule, not a change to Runner's load gate.

These observations come from Suite's Rust agent-runtime asset module
(`Skills`, `map_capability`, `authorize` and their tests) and Runner-control
client (`RunnerIdentity`, `probe`) at the revision above.

## What a schema would add

A narrow shared contract could give the existing profile and evidence
references a machine-checkable vocabulary: declaration versus measurement,
the identity/scope a measured assertion covers, source-record identity,
absence/unknown handling and version compatibility. It should reuse the
envelope and template layer rather than describe every family's wire grammar.
A schema validates structure; it neither produces measurements nor supplies
the missing Suite reader, activation caller or evidence-trust policy.

The original model-card binding is not implemented: the fixture declares
`gridcore.agent.action.v1` with a placeholder digest, and no matching canonical
schema document is resolved by the loader. Do not silently redefine that
existing ID as a new capability-evidence schema.

## Meaning of the proposed card binding

Subject to the choices below, `{id, version, digest}` would name a published
schema document: a stable contract ID, its schema release, and SHA-256 of
that release's exact bytes. It would not name the model weights, an evidence
instance, or a successful evaluation. Those need their own identities.
Schema release is distinct from the existing GGUF protocol/tokenizer versions.

An illustrative additive model-card YAML annotation would be:

```yaml
agent_protocol:
  id: "<canonical schema id>"
  version: "<schema release>"
  digest: "<64 lowercase hexadecimal SHA-256 characters>"
```

The key and values are proposals, not fields Runner or Suite currently read.
A resolver would locate an immutable schema release, compare its digest and
validate the profile/evidence reference against it. A card remains a
publisher declaration. Where the contract applies to GGUF profile metadata,
its schema ID/digest must refer to the same document; the exact mapping and
legacy behavior need agreement before changing either cards or exporters.

## Open choices

Each list puts the recommendation first; none is implemented by this note.

1. **Scope and supersession.** (a) Narrow the old task to a profile/evidence
   binding contract over existing surfaces, with the Suite activation work
   explicit. (b) Close it as superseded and define the mapping entirely in
   the Suite using today's objects; viable if no shared producer contract
   is required. (c) Publish a full cross-family wire-protocol schema, which
   duplicates template/parser logic and does not solve measured admission.
2. **Identity and digest.** (a) Version an immutable UTF-8/LF schema document
   in Runner and hash its exact bytes; define the ID's meaning before
   mapping existing GGUF fields. (b) Hash canonicalized JSON instead,
   requiring a specified canonicalization algorithm and matching consumers.
3. **Measured admission.** (a) Define a named model capability separately
   from permissions; accept only relevant, identity-matched evidence from
   an explicitly trusted record source. (b) Require all three existing
   tool-calling rollup subchecks plus matching native-protocol evidence,
   a more restrictive rule that may require unrelated/proxy checks.
4. **Delivery.** (a) Have the Suite read a supplied envelope/source record
   and compare it with runtime capabilities/provenance; card binding is an
   optional discovery annotation. (b) Resolve a pinned external evidence
   registry, adding network and trust policy. (c) Extend the live API to
   serve evidence, adding a Runner API and cache/lifecycle obligations.

## Recommendation and stop condition

The declared-protocol portion is largely superseded; the complete need is
not yet covered. Recommend the narrow contract in choice 1(a), not another
wire grammar. Keep implementation and card publication out of this step.
If the coordinator accepts Suite-local interpretation of the existing
envelope/report fields, choose 1(b), close the original schema/card task as
superseded and remove it as a blocker; the two Suite tasks still need their
production caller, reader and admission tests. A schema file alone is not
a prerequisite for those tests and would not make them pass.

The decisive future checks are through activation with a real measured row:
allow the matching required capability; refuse an absent row, different
artifact, native declaration without measurement, wrong configuration or
unknown contract. A declaration-only profile must never substitute for the
measured row required by those Suite tasks.
