// A thin TypeScript client for Xyntetik Runner (R10.7).
//
// Every TypeScript agent already speaks the OpenAI or Anthropic wire, so
// this package does not reimplement those SDKs. Its value is the fields no
// other engine returns -- runner_telemetry, choice_logprobs, the provenance
// statement, the transcript receipt -- typed, and the Runner-only routes
// (/v1/rerank, /v1/decide, named contexts, the Responses store). No runtime
// dependencies: it uses the platform's fetch and WebCrypto (Node 18+,
// browsers, Deno, Bun).

import type {
  ChoiceLogprob, NamedContext, Provenance, RerankResponse, RunnerErrorBody,
  RunnerTelemetry, TranscriptRecord,
} from "./types.js";

export type * from "./types.js";

type Json = Record<string, unknown>;

export interface RunnerClientOptions {
  /** Default http://127.0.0.1:8080 (Runner binds loopback only). */
  baseURL?: string;
  /** Sent as a Bearer token when set; Runner itself does not check one. */
  apiKey?: string;
  /** A fetch implementation; defaults to the global one. */
  fetch?: typeof globalThis.fetch;
}

/** A non-2xx answer, carrying Runner's error object. */
export class RunnerAPIError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly param: string | null;
  readonly body: unknown;
  constructor(status: number, body: unknown) {
    const err = (body as RunnerErrorBody | null)?.error;
    super(err?.message ?? `HTTP ${status}`);
    this.name = "RunnerAPIError";
    this.status = status;
    this.code = err?.code ?? null;
    this.param = err?.param ?? null;
    this.body = body;
  }
}

/** A stream that broke its own protocol (a malformed data frame): never
 *  skipped, because a later finish_reason would then certify a corrupt turn. */
export class RunnerProtocolError extends Error {
  readonly partial: string;
  constructor(message: string, partial: string) {
    super(message);
    this.name = "RunnerProtocolError";
    this.partial = partial;
  }
}

export class RunnerClient {
  readonly baseURL: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(opts: RunnerClientOptions = {}) {
    this.baseURL = (opts.baseURL ?? "http://127.0.0.1:8080").replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.fetchImpl = opts.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private headers(body: boolean): Record<string, string> {
    const h: Record<string, string> = {};
    if (body) h["Content-Type"] = "application/json";
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  private async raw(method: string, path: string, body?: unknown): Promise<Response> {
    const res = await this.fetchImpl(this.baseURL + path, {
      method,
      headers: this.headers(body !== undefined),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) {
      let parsed: unknown = null;
      try { parsed = await res.json(); } catch { /* not JSON */ }
      throw new RunnerAPIError(res.status, parsed);
    }
    return res;
  }

  /** One JSON request; the answer's type is the caller's to assert. */
  async request<T = Json>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await this.raw(method, path, body);
    return (await res.json()) as T;
  }

  // ---- the OpenAI and Anthropic surfaces, returned as the server sent them

  chat(req: Json): Promise<Json> { return this.request("POST", "/v1/chat/completions", req); }
  completions(req: Json): Promise<Json> { return this.request("POST", "/v1/completions", req); }
  responses(req: Json): Promise<Json> { return this.request("POST", "/v1/responses", req); }
  messages(req: Json): Promise<Json> { return this.request("POST", "/v1/messages", req); }
  embeddings(req: Json): Promise<Json> { return this.request("POST", "/v1/embeddings", req); }

  /** Server-sent events of a streamed chat completion, parsed. A malformed
   *  data frame throws RunnerProtocolError; comments and non-data fields are
   *  ignored as the SSE spec requires. */
  async *chatStream(req: Json): AsyncGenerator<Json> {
    yield* this.sse("/v1/chat/completions", { ...req, stream: true });
  }

  /** Typed events of a streamed Responses turn. */
  async *responsesStream(req: Json): AsyncGenerator<Json> {
    yield* this.sse("/v1/responses", { ...req, stream: true });
  }

  private async *sse(path: string, req: Json): AsyncGenerator<Json> {
    const res = await this.raw("POST", path, req);
    if (!res.body) throw new RunnerProtocolError("stream has no body", "");
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", seen = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (value) buf += dec.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, "");
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trimStart();
        if (data === "[DONE]") return;
        let ev: Json;
        try { ev = JSON.parse(data) as Json; } catch {
          throw new RunnerProtocolError("malformed data frame", seen);
        }
        seen += data;
        yield ev;
      }
      if (done) return;
    }
  }

  // ---- Runner-only routes

  capabilities(): Promise<Json> { return this.request("GET", "/v1/capabilities"); }
  health(): Promise<Json> { return this.request("GET", "/health"); }
  provenance(): Promise<Provenance> { return this.request("GET", "/v1/runner/provenance"); }
  rerank(req: { query: string; documents: (string | { text: string })[];
                top_n?: number; return_documents?: boolean; instruction?: string;
                rendering?: "chat-v1" | "raw-v1" }): Promise<RerankResponse> {
    return this.request("POST", "/v1/rerank", req);
  }
  decide(req: Json): Promise<Json> { return this.request("POST", "/v1/decide", req); }

  /** Named contexts: prefill a shared prefix once, fork it by context_id. */
  readonly contexts = {
    create: (req: { id: string; prompt?: string; messages?: Json[]; tools?: Json[] }) =>
      this.request<NamedContext & { prefill_tokens: number; cached_tokens: number;
                                    seconds: number }>("POST", "/v1/runner/contexts", req),
    list: async (): Promise<NamedContext[]> =>
      (await this.request<{ data: NamedContext[] }>("GET", "/v1/runner/contexts")).data,
    delete: (id: string) =>
      this.request<{ id: string; deleted: boolean }>(
        "DELETE", `/v1/runner/contexts/${encodeURIComponent(id)}`),
  };

  /** The Responses store (store:true / previous_response_id). */
  readonly storedResponses = {
    get: (id: string) => this.request("GET", `/v1/responses/${encodeURIComponent(id)}`),
    inputItems: async (id: string): Promise<Json[]> =>
      (await this.request<{ data: Json[] }>(
        "GET", `/v1/responses/${encodeURIComponent(id)}/input_items`)).data,
    delete: (id: string) =>
      this.request<{ id: string; deleted: boolean }>(
        "DELETE", `/v1/responses/${encodeURIComponent(id)}`),
  };
}

// ---- reading the runner-only fields off a response body

/** The runner_telemetry object of any surface's body, or undefined. */
export function runnerTelemetry(body: unknown): RunnerTelemetry | undefined {
  const t = (body as { runner_telemetry?: unknown } | null)?.runner_telemetry;
  return t && typeof t === "object" ? (t as RunnerTelemetry) : undefined;
}

/** runner_telemetry.finish_detail: why a turn ended when finish_reason lost it. */
export function finishDetail(body: unknown): string | undefined {
  return runnerTelemetry(body)?.finish_detail;
}

/** choice_logprobs of a chat or text completion (first choice). */
export function choiceLogprobs(body: unknown): ChoiceLogprob[] | undefined {
  const c = (body as { choices?: { choice_logprobs?: ChoiceLogprob[] }[] } | null)
    ?.choices?.[0]?.choice_logprobs;
  return Array.isArray(c) ? c : undefined;
}

// ---- the transcript receipt

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  // a copy on a plain ArrayBuffer: digest() refuses a SharedArrayBuffer view
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new Uint8Array(bytes)));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

export interface ChainCheck {
  ok: boolean;
  /** chain.hash as the record states it */
  stated: string | null;
  /** sha256 of the record's bytes before `,"chain"`, computed here */
  computed: string | null;
  reason?: string;
}

/** Recompute a transcript's chain hash from its bytes: sha256 over every
 *  byte before the last `,"chain"`. Needs nothing but the file, which is the
 *  point of the format. The signature, when present, is a separate check. */
export async function verifyTranscriptChain(record: string | Uint8Array): Promise<ChainCheck> {
  const bytes = typeof record === "string" ? new TextEncoder().encode(record) : record;
  const text = new TextDecoder().decode(bytes);
  const marker = ',"chain":';
  const at = text.lastIndexOf(marker);
  if (at < 0) return { ok: false, stated: null, computed: null, reason: "no chain object" };
  let stated: string | null = null;
  try {
    stated = (JSON.parse(text) as TranscriptRecord).chain?.hash ?? null;
  } catch {
    return { ok: false, stated: null, computed: null, reason: "not JSON" };
  }
  // the marker's byte offset: the text before it re-encoded
  const cut = new TextEncoder().encode(text.slice(0, at)).length;
  const computed = await sha256Hex(bytes.subarray(0, cut));
  return { ok: stated === computed, stated, computed,
           ...(stated === computed ? {} : { reason: "chain hash differs" }) };
}
