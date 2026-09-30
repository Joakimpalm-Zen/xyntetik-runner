// The fields Runner returns that no other engine does, typed. Everything a
// client already gets from the OpenAI or Anthropic wire is left to those
// SDKs' own types; these are the runner-only additions, spelled as the
// server spells them (src/completion.c telemetry_json, src/server.c).

/** runner_telemetry.sampling: what the request was actually served with. */
export interface SamplingTelemetry {
  preset: string | null;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repeat_penalty: number;
  seed: number;
  source: Record<"temperature" | "top_p" | "top_k" | "min_p" | "repeat_penalty",
                 "request" | "cli" | "preset">;
}

export interface ToolProtocolTelemetry {
  template: string;
  family: string | null;
  tools: boolean;
  constrained: boolean;
  parse_only: boolean;
}

export interface TimingTelemetry {
  queue_seconds: number;
  tokenize_seconds: number;
  device_wait_seconds: number;
  prefill_seconds: number;
  prefill_tokens: number;
  prefill_tok_s: number;
  first_visible_seconds: number | null;
}

/** An emitted tool call that repeats one already in the conversation. */
export interface RepeatedToolCall {
  index: number;
  name: string;
  prior_calls: number;
  last_message_index: number;
}

export interface SpeculationTelemetry {
  source: "model" | "mtp" | "lookup" | "grammar";
  rounds: number;
  drafted: number;
  accepted: number;
  lookup_drafted: number;
  lookup_accepted: number;
}

/** The runner_telemetry object carried by Chat, Completions, Responses and
 *  Messages bodies. Optional members are present only when they apply. */
export interface RunnerTelemetry {
  prompt_cached_tokens: number;
  prompt_forked_tokens: number;
  prompt_eval_tokens: number;
  prefix_cache_saved_seconds: number;
  generation_seconds: number;
  generation_tok_s: number;
  major_page_faults: number;
  page_fault_counter: "major" | "all";
  json_mode: boolean;
  schema: boolean;
  speculative: boolean;
  /** Why a turn ended when the wire's finish_reason lost the distinction
   *  (e.g. "envelope_unmapped", "loop", "reasoning_limit"). */
  finish_detail?: string;
  reasoning_budget?: { max_tokens: number; tokens: number; forced_close: boolean };
  loop_guard?: {
    span: number; repeats: number; window: number; max_closes: number;
    interventions: number; ended_turn: boolean;
  };
  reasoning_sampling?: { temperature: number; top_p: number; min_p: number; top_k: number };
  sampling?: SamplingTelemetry;
  tool_protocol?: ToolProtocolTelemetry;
  timing?: TimingTelemetry;
  prompt_reuse?: string;
  context?: { id: string; tokens: number };
  repeated_tool_calls?: RepeatedToolCall[];
  speculation?: SpeculationTelemetry;
}

/** One constrained decision point (choice_logprobs: true). */
export interface ChoiceLogprob {
  index: number;
  n_legal: number;
  coverage: number;
  alternatives: { token: string; id: number; prob: number; logprob: number }[];
}

export type EnvelopeState =
  "unclassified" | "certified" | "outside" | "experimental" | "indeterminate";

/** OMS model-signature verdict, as receipts and /v1/runner/provenance carry it. */
export interface ModelSignature {
  status: "verified" | "unverified" | "unsupported" | "malformed" | "missing";
  reason?: string;
  curve?: string;
  hash?: string;
  subject_digest?: string;
  resource?: string;
  key_hint?: string;
}

/** GET /v1/runner/provenance */
export interface Provenance {
  object: "runner.provenance";
  version: string;
  build_flavor?: string;
  build: { binary_sha256: string | null; compiler: string; os: string; arch: string;
           flavor?: string };
  model: null | {
    id: string; path: string; sha256: string | null;
    sha256_state: "hashing" | "done" | "changed_since_load" | "unreadable";
    size: number | null; loaded_utc: string;
    signature: ModelSignature | null;
    envelope: { state: EnvelopeState; detail: string };
  };
  adapter: null | { path: string; sha256: string | null; scale: number };
  profile: null | { device: string; gpu: boolean; gpu_layers: number; threads: number;
                    ctx: number; kv: string; batch: number; slots: number };
  config: Record<string, unknown>;
  statement: string;
}

/** xyntetik.runner.transcript.v1: the receipt a CLI run writes. */
export interface TranscriptRecord {
  schema_version: "xyntetik.runner.transcript.v1";
  runner: string;
  build: { binary_sha256: string; compiler: string; os: string; arch: string;
           flavor?: string };
  profile: { device: string; gpu: boolean; gpu_layers: number; threads: number;
             ctx: number; kv: string; batch: number };
  model: { path: string; sha256: string };
  adapter: null | { path: string; sha256: string; scale: number };
  config: { seed: number; seed_u64: string; temp: number; top_k: number; top_p: number;
            min_p: number; repeat_penalty: number; n_predict: number; template: string;
            bos: boolean };
  model_signature?: ModelSignature;
  speculation?: { source: string; rounds: number; drafted: number; accepted: number;
                  lookup_drafted: number; lookup_accepted: number };
  generated_utc: string;
  prompt: { text: string; tokens: number[] };
  output: { text: string; bytes_hex: string; tokens: number[]; n: number;
            finish: "stop" | "length" };
  /** hash: sha256 over every byte of the file before `,"chain"`; prev: the
   *  previous receipt's hash, or 64 zeros for a chain head. */
  chain: { algo: "sha256"; prev: string; hash: string };
  signature?: { algo: string; public_key: string; sig: string };
}

/** POST /v1/rerank result entry. */
export interface RerankResult {
  index: number;
  relevance_score: number;
  logit: number;
  margin: number | null;
  logprobs: { yes: number; no: number };
  document?: { text: string };
}

export interface RerankResponse {
  id: string;
  object: "rerank";
  model: string;
  results: RerankResult[];
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  envelope: { runner_version: string; query_sha256: string; documents_sha256: string;
              instruction_sha256: string; rendering: "chat-v1" | "raw-v1";
              options: ["yes", "no"] };
}

/** A named context (POST/GET /v1/runner/contexts). */
export interface NamedContext {
  id: string;
  tokens: number;
  bytes: number;
  hits?: number;
  age_seconds?: number;
}

/** The error object every Runner route answers with. */
export interface RunnerErrorBody {
  error: { message: string; type: string; param: string | null; code: string | null };
}
