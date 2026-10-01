# @xyntetik/runner-client

A thin TypeScript client for [Xyntetik Runner](../../README.md). Every
TypeScript agent already speaks the OpenAI or Anthropic wire, so this package
does not replace those SDKs: it types the fields no other engine returns and
wraps the Runner-only routes. No runtime dependencies; it uses the platform's
`fetch` and WebCrypto (Node 18+, browsers, Deno, Bun).

```ts
import { RunnerClient, runnerTelemetry, choiceLogprobs,
         verifyTranscriptChain } from "@xyntetik/runner-client";

const runner = new RunnerClient({ baseURL: "http://127.0.0.1:8080" });

const body = await runner.chat({ messages: [{ role: "user", content: "hi" }] });
const t = runnerTelemetry(body);          // RunnerTelemetry | undefined
t?.sampling?.source.temperature;          // "request" | "cli" | "preset"
t?.finish_detail;                          // why a turn ended, when the wire lost it
t?.repeated_tool_calls;                    // a call repeating an earlier one
choiceLogprobs(body);                      // constrained decision posteriors

await runner.provenance();                 // binary/model digests, verdicts
await runner.rerank({ query: "q", documents: ["a", "b"] });
await runner.contexts.create({ id: "sys", prompt: "..." });   // then context_id
await runner.storedResponses.get("resp_3");                   // store:true
for await (const chunk of runner.chatStream({ messages: [] })) { /* ... */ }

// a receipt's chain hash, recomputed from its bytes alone
await verifyTranscriptChain(await fs.readFile("run.transcript.json"));
```

A non-2xx answer throws `RunnerAPIError` with Runner's `status`, `code` and
`param`; a malformed SSE data frame throws `RunnerProtocolError` rather than
being skipped, so a later `finish_reason` cannot certify a corrupt stream.

Build and test (the tests start a runner on the repository's CI fixture):

```sh
make                       # the runner, at the repository root
cd clients/typescript
npm ci && npm run build && npm test
```
