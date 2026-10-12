# Harmony prompt and tool contracts

The gpt-oss text path uses the publisher's Harmony channel/recipient
protocol. Native system/developer/tool declarations, analysis/final content,
assistant-to-function calls, and tool results must retain their trained
turn framing. A generic JSON envelope or guessed function name is not an
equivalent prompt.

The renderer is checked against the publisher's template and pinned
openai-harmony function-tool goldens. Resolve a tool result's function from
the call ID/history, or the sole declared tool; unresolved ambiguous names
refuse the request. Current strict turns bound pre-call analysis/commentary
to 192 UTF-8 bytes and preserve budget for arguments. This can truncate a
quoted JSON tool result; client loops and closure checks cover that limit.

Buffered and streamed output share generation, stop/UTF-8/tool processing,
and native mapping. Existing template, tool attribution, prompt marking,
constraint/stop, reasoning, and API conformance gates protect the protocol.
See the [manual's Harmony account](../MANUAL.md#openai-chat-completions),
[current client setup](agent-compatibility.md), and
[model/reference scope](model-evidence.md#gpt-oss).
