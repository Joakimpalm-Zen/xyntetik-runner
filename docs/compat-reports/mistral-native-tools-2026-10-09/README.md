# Mistral v0.3 and Nemo native tools: agent-torture, generic envelope against native protocol

2026-10-09, `scripts/agent-torture.py --cases 40` against a Runner server on the
Blackwell host's CPUs (16 threads, CUDA hidden, other tenants' load present),
run by the runner session overnight (suite R2.4.3). Each model is the same file
on both arms.

| arm | model | binary | template | scored | passed | failed | excused |
|---|---|---|---|---|---|---|---|
| `v03-generic/` | Mistral-7B-Instruct-v0.3 Q4_K_M | main 8aa02e0: generic JSON envelope | `--chat-template mistral` | 40 | 40 | 0 | 0 |
| `v03-native/` | Mistral-7B-Instruct-v0.3 Q4_K_M | branch mistral-tools: `[AVAILABLE_TOOLS]`, `[TOOL_CALLS]` | `--chat-template mistral` | 40 | 40 | 0 | 0 |
| `nemo-generic/` | Mistral-Nemo-Instruct-2407 Q4_K_M | main 8aa02e0: generic JSON envelope | detected (`mistral`) | 40 | 40 | 0 | 0 |
| `nemo-native/` | Mistral-Nemo-Instruct-2407 Q4_K_M | branch mistral-tools: `[AVAILABLE_TOOLS]`, `[TOOL_CALLS]` | detected (`mistral`) | 40 | 40 | 0 | 0 |

The score does not move: both models already passed every case through the
generic envelope, and no torture case sends a system prompt of its own. What
changes is the protocol the model reads and writes: the native arm declares the
tools and replays calls and results in the publisher template's bytes (template
conformance: both families identical in text and tokens), and the model writes
its own trained `[TOOL_CALLS]` list. The native responses report `mistral_json`
as their tool protocol, the generic ones `generic`.

What the matrix cannot show, measured separately on the Nemo file (prompt
tokens of one chat request, `max_tokens` 64, greedy):

| request | main | branch |
|---|---|---|
| user + tools | 197 | 72 |
| system + user, no tools | 21 | 21 |
| system + user + tools | **21** | 83 |

On main the generic envelope's teaching turn is a second system message, and
the Mistral renderer keeps only the last system message, so a caller who sends
its own system prompt has the tool declarations dropped from the prompt
entirely. The call still came back because the grammar forced the envelope's
shape; the model was never shown the tools. On the branch the declarations are
rendered natively beside the caller's system text.

Two facts about the pinned files, both from runner's template detection:

- The v0.3 file embeds an older template that detects as the v0.1 framing
  (`mistral-v1`), which has no tool protocol and stays on the generic
  envelope. Both v0.3 arms were therefore run with `--chat-template mistral`,
  the publisher's current template. Without the flag this file is served as
  before.
- The Nemo file embeds a template that detects as the v0.3 form (`mistral`),
  so both Nemo arms ran on it unflagged; the grammar admits the list opened
  with or without v0.3's space, which covers the Nemo spelling too.

Before the grammar was written, both models were run unconstrained on the
reference prompt: v0.3 writes `[TOOL_CALLS] [{"name": ..., "arguments": {...}}]`
and then keeps generating with no end-of-turn, Nemo writes
`[TOOL_CALLS][...]` and stops; neither writes an `id`. The grammar ends the turn
at the list's `]`.

`report.json` and `raw.jsonl` are the harness's own files, unchanged (the
servers were started separately and reached with `--endpoint`, so no local
paths are recorded). The forced-truncation case (one token, `required`) comes
back as a complete `dispatch_job` call on every arm.
