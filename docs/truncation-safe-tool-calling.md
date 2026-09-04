# Tool calls that survive the token limit

**Question.** What happens to a tool call when `max_tokens` runs out in the
middle of the JSON arguments?

**Answer.** In most engines the caller gets `finish_reason: "length"` with
nothing usable, or truncated JSON to repair or retry. Runner closes the call
to the smallest schema-legal document instead, so the arguments still parse
and the agent loop continues. This is forced-truncation recovery, a property
of the runtime, not ordinary JSON-Schema constrained decoding.

## Measured

Same box, same tool schema, same prompt, `tool_choice: "required"`,
temperature 0, budgets from 1 to 64 tokens. What each engine hands the
caller when the budget cuts the call short:

| engine | budget too small (1 to 16 tokens) | enough budget (64, control) |
|---|---|---|
| Runner | executable `tool_calls`, arguments parse | completes |
| vLLM 0.27.1 | no call; protocol framing leaks into `content` | completes |
| llama.cpp b10488 | no call; leak, then `tool_calls` with unparseable arguments | completes |
| Ollama 0.32.14 | no call; empty content, then HTTP 500 | completes |
| TensorRT-LLM 1.2.1 | no call; `<tool_call>` leak, then empty content | completes |
| SGLang 0.5.17 | no call; `<tool_call>` leak, then empty content | completes |

The control rung proves the failure is truncation, not misconfiguration:
every engine completes at 64. TensorRT-LLM and SGLang were measured on a
Qwen3-1.7B substitute because their registries did not carry the
granite-4.1-3b used for the other four; recovery is a property of the
engine, so the substitution measures the engine, not the model. The claim
covers the engines and versions measured, not engines that were not.

The recipe, the raw responses and the date of the measurement are in the
[truncation benchmark](truncation-benchmark.md), which pins Runner's column
as a per-release regression gate (`make test-truncation`). The
[agent-torture gate](agent-torture.md) tests the same failure mode inside
multi-turn agent loops.

## Under quantization

Tool-call fidelity was also measured across a full quant ladder: constrained
decoding held schema conformance and tool selection at 100% down to Q4_0,
while argument agreement decayed to 50%. Constrained decoding guarantees the
shape of a call at any quantization, not its contents. Numbers and method:
[docs/quant-fidelity.md](quant-fidelity.md).

## Try it

```sh
./runner -m model.gguf --serve
curl localhost:8080/v1/chat/completions -d '{
  "messages":[{"role":"user","content":"Book a table for two at 19:00."}],
  "tools":[{"type":"function","function":{"name":"book_table",
    "parameters":{"type":"object","properties":{"people":{"type":"integer"},
    "time":{"type":"string"}},"required":["people","time"]}}}],
  "tool_choice":"required","max_tokens":8}'
```

The response carries a `tool_calls` entry whose `arguments` parse, and the
usual `finish_reason: "length"` so the caller knows the budget was hit.
