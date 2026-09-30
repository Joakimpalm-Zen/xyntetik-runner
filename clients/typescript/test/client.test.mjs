// Runs the compiled client (dist/) against a live runner on the CI fixture.
// Anchors from outside the runner: Node's own sha256 of the runner binary for
// the provenance digest, and of the receipt bytes for the chain check.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import {
  RunnerClient, RunnerAPIError, RunnerProtocolError, runnerTelemetry,
  choiceLogprobs, finishDetail, verifyTranscriptChain,
} from "../dist/index.js";

const ROOT = resolve(import.meta.dirname, "../../..");
const RUNNER = join(ROOT, process.platform === "win32" ? "runner.exe" : "runner");
const dir = mkdtempSync(join(tmpdir(), "rc-ts-"));
const MODEL = join(dir, "model.gguf");
let proc, client;

function freePort() {
  return new Promise((res) => {
    const s = createServer();
    s.listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); });
  });
}

before(async () => {
  execFileSync("python3", [join(ROOT, "scripts/make-test-model.py"), MODEL],
               { stdio: "ignore" });
  const port = await freePort();
  proc = spawn(RUNNER, ["-m", MODEL, "--serve", "--no-tray", "--port", String(port),
                        "-c", "1024", "--gpu", "off", "-t", "2"], { stdio: "ignore" });
  client = new RunnerClient({ baseURL: `http://127.0.0.1:${port}` });
  for (let i = 0; i < 200; i++) {
    try { await client.health(); return; } catch { await new Promise((r) => setTimeout(r, 50)); }
  }
  throw new Error("runner did not start");
});

after(() => { proc?.kill(); });

test("provenance is typed and its binary digest is the file's", async () => {
  const p = await client.provenance();
  assert.equal(p.object, "runner.provenance");
  const want = createHash("sha256").update(readFileSync(RUNNER)).digest("hex");
  assert.equal(p.build.binary_sha256, want);
  assert.equal(p.profile?.slots, 1);
});

test("runner_telemetry and choice_logprobs are read off a chat body", async () => {
  const body = await client.chat({
    messages: [{ role: "user", content: "pick a colour" }],
    max_tokens: 8, temperature: 0.7, seed: 7, choice_logprobs: true,
    choice_logprobs_probe: 64,
    // 26 distinct first bytes: the fixture's near-random logits then put >= 2
    // legal candidates inside the probe (tests/conformance/test_choice_logprobs.py)
    response_format: { type: "json_schema", json_schema: { name: "c", schema: {
      type: "string", enum: [..."abcdefghijklmnopqrstuvwxyz"].map((c) => c + "0") } } },
  });
  const t = runnerTelemetry(body);
  assert.ok(t);
  assert.equal(t.sampling.temperature, 0.7);
  assert.equal(t.sampling.source.temperature, "request");
  assert.equal(t.schema, true);
  const cl = choiceLogprobs(body);
  assert.ok(cl && cl.length >= 1, JSON.stringify(body));
  for (const d of cl) assert.ok(d.n_legal >= 2 && d.alternatives.length >= 2);
  assert.equal(finishDetail(body), undefined);
});

test("a streamed chat is the buffered one, chunk by chunk", async () => {
  const req = { messages: [{ role: "user", content: "hello" }], max_tokens: 8,
                temperature: 0 };
  const buffered = await client.chat(req);
  let text = "", finish = null;
  for await (const ev of client.chatStream(req)) {
    for (const ch of ev.choices ?? []) {
      text += ch.delta?.content ?? "";
      finish = ch.finish_reason ?? finish;
    }
  }
  assert.equal(text, buffered.choices[0].message.content);
  assert.equal(finish, buffered.choices[0].finish_reason);
});

test("rerank, contexts and the Responses store", async () => {
  const r = await client.rerank({ query: "cat", documents: ["a cat", "stocks"] });
  assert.equal(r.results.length, 2);
  assert.ok(r.results[0].logit >= r.results[1].logit);
  const sys = "You are a terse assistant for a warehouse inventory team. ";
  const c = await client.contexts.create({ id: "ts-sys", prompt: sys });
  assert.ok(c.tokens > 0);
  assert.deepEqual((await client.contexts.list()).map((x) => x.id), ["ts-sys"]);
  const warm = await client.completions({ prompt: sys + "Item 4?", max_tokens: 2,
                                          context_id: "ts-sys" });
  assert.deepEqual(runnerTelemetry(warm).context, { id: "ts-sys", tokens: c.tokens });
  assert.equal((await client.contexts.delete("ts-sys")).deleted, true);
  await assert.rejects(client.contexts.delete("ts-sys"), (e) =>
    e instanceof RunnerAPIError && e.status === 404 && e.code === "context_not_found");
  const s = await client.responses({ input: "hi", store: true, max_output_tokens: 3 });
  assert.equal((await client.storedResponses.get(s.id)).id, s.id);
  assert.equal((await client.storedResponses.inputItems(s.id))[0].role, "user");
});

test("a transcript's chain verifies from its bytes, and a changed byte fails",
     async () => {
  const rec = join(dir, "r.json");
  execFileSync(RUNNER, ["-m", MODEL, "-p", "hello", "-n", "3", "--temp", "0",
                        "--gpu", "off", "-t", "2", "--transcript", rec],
               { stdio: "ignore" });
  const bytes = readFileSync(rec);
  const ok = await verifyTranscriptChain(bytes);
  assert.equal(ok.ok, true, ok.reason);
  // the independent reading: Node's sha256 of the bytes before ,"chain"
  const cut = bytes.lastIndexOf(Buffer.from(',"chain":'));
  assert.equal(ok.computed, createHash("sha256").update(bytes.subarray(0, cut)).digest("hex"));
  const bad = Buffer.from(bytes.toString().replace('"text":"hello"', '"text":"hellp"'));
  assert.notDeepEqual(bad, bytes);
  const r = await verifyTranscriptChain(bad);
  assert.equal(r.ok, false);
  assert.equal(r.reason, "chain hash differs");
});

test("a malformed SSE frame is an error, never skipped", async () => {
  const fake = async () => new Response(
    "data: {\"choices\":[]}\n\ndata: {not json\n\ndata: [DONE]\n\n",
    { status: 200, headers: { "Content-Type": "text/event-stream" } });
  const c = new RunnerClient({ fetch: fake });
  const seen = [];
  await assert.rejects(async () => {
    for await (const ev of c.chatStream({ messages: [] })) seen.push(ev);
  }, (e) => e instanceof RunnerProtocolError);
  assert.equal(seen.length, 1);
});
