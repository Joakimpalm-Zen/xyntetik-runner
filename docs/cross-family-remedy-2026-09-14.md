# Native tool and client-loop evidence

Current native protocols are selected once and kept consistent across prompt,
grammar, parser, history, and transport. `auto` turns can be freely generated
and parsed where the family supports that route; `required` and named choices
retain their constraints. Tool-result IDs resolve against declared/history
calls, typed arguments keep their types, and buffered/streamed paths share
mapping and UTF-8/stop behavior.

Qwen, Gemma, and Muse bounded parallel turns carry 1–8 calls with their native
separators/stops. Gemma grammar capacity can reduce the cap; an omitted policy
has no family fixed-pair exception. Named OpenAI function choices default to
one call, named Anthropic tool choices retain their own policy, and explicit
values win. Nemotron Nano and Mistral v0.3/Nemo use their native protocols.
Caller system text survives beside declarations, and required turns admit
their post-think newlines. These are current source/test contracts.

## Retained measurement records

The following September 14–15 records describe their original model/build,
preset, backend, budget, and client versions. Their task results do not
re-certify a later build or every family.

| Artifact / check | Recorded result | Evidence |
| --- | --- | --- |
| Qwen 3.8 GSQ-RCO IQ3_S, corrected prompt, Linux CUDA host | 6/6 report cases, buffered/streamed | [protocol record](cross-family-remedy-evidence/tool-protocol-qwen38-gsq-iq3s-blackwell-marked-think.json) |
| Qwen3-Coder 30B, corrected preset, Linux CUDA host | 6/6 report cases | [protocol record](cross-family-remedy-evidence/tool-protocol-qwen3-coder-30b-blackwell-final.json) |
| Gemma 4 12B QAT, native preset, Linux CUDA host | A/B cases pass; prose-beside-call case has its recorded model limitation | [protocol record](cross-family-remedy-evidence/tool-protocol-gemma4-12b-qat-q4_0-blackwell-gemma4-preset.json) |
| Granite 4.2 8B, corrected prompt, Windows CUDA | A/B results and the reasoning/content case are recorded separately | [protocol record](cross-family-remedy-evidence/tool-protocol-granite42-8b-q4_k_m-rtx3070-marked-think-1500.json) |
| Installed clients against Qwen3-4B | Named client/version/scope results | [client sweep](cross-family-remedy-evidence/agent-client-sweep-qwen3-4b-blackwell-2026-09-15.json) |
| Installed clients against Granite 4.2, Windows | Continue/OpenCode loop scope | [client sweep](cross-family-remedy-evidence/agent-client-sweep-granite42-8b-windows-rtx3070-2026-09-15.json) |

The [coverage inventory](../tests/compatibility/coverage.json) names the
retained evidence for each disposition and keeps missing checks explicit.
Native protocol fixtures, tool attribution, prompt marking, required/auto
routing, sampling defaults, reasoning, truncation, and API conformance tests
protect the current implementation. They are independent of historical run
output and remain in the aggregate gate.

Use [client setup](agent-compatibility.md), the [OpenCode workflow](workflow-opencode.md),
and [compatibility checks](compatibility-program.md) for current instructions.
