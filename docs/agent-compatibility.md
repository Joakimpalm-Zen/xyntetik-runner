# Coding-agent setup and compatibility

Runner serves OpenAI Chat Completions, Responses, and legacy completions,
plus Anthropic Messages. Select the surface your client actually uses,
match its context/output limits to the server, and obtain the model ID from
`GET /v1/models`. Unknown model names are refused.

| Client surface | Base URL |
| --- | --- |
| OpenAI-compatible Chat/Responses | `http://127.0.0.1:8080/v1` |
| Anthropic Messages | `http://127.0.0.1:8080` |

Use [the tested OpenCode workflow](workflow-opencode.md) for one complete
configuration and its measured memory/latency limits. The
[manual's coding-agent section](../MANUAL.md#coding-agent-evidence) records
model, client-version, hardware, and scope for the latest retained sweeps;
a dated installed-client result is not a claim about every newer client.

## Client settings

- OpenCode uses an OpenAI-compatible provider with explicit context/output
  limits. Large system/tool declarations can consume most of a short context.
- Claude Code uses `ANTHROPIC_BASE_URL` at the server root. A bounded
  `--tools Read` quality check and a full built-in-tool schema check establish
  different things; a small model need not use every declared tool well.
- Codex providers use `wire_api = "responses"`. Disable unused apps,
  hosted search, and multi-agent tools when the declared tool set exceeds
  Runner's 59-tool constrained-envelope limit. Hosted tools are refused
  rather than presented as local functions.
- Cline and Continue use the appropriate OpenAI-compatible provider;
  configure the served model ID and tool-use capability, not a guessed ID.
- pi can use the Chat, Responses, or Anthropic surface with the matching
  base URL. Aider's `--dry-run` can check transport/inference; an edit profile
  matching the model is still required to establish actual editing behavior.

A proxy or remote relay needs its own authentication and authority policy;
Runner binds loopback and validates request authority. See [SECURITY.md](../SECURITY.md).

## What the checks establish

[Consumer and reference checks](compatibility-program.md) exercise pinned
SDKs, model artifacts, tokenizers/templates, and actual API exchanges.
[Agent torture](agent-torture.md) uses deterministic request matrices;
[schema/tool conformance](../MANUAL.md#serving-and-apis) protects native
protocols, argument validity, streaming, and truncation closure.

Parallel tool turns support the family's bounded 1–8-call protocol.
Omitted parallel policy allows multiple calls; a named OpenAI function
choice defaults to one, while explicit request values win. Native syntax,
model stop tokens, and generic fallback choices must agree across prompt,
grammar, parser, and transport. These are runtime contracts, not claims
that a model will select a useful tool on every task.

Missing clients, editor hosts, or device assets remain NOT RUN. Existing
records do not certify editor extensions or newer package versions by
association. Report failures with the exact client/server versions, model
hash/configuration, request/error, and a bounded reproducer.
