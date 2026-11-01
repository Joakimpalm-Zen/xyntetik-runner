// Chat templates, tool-call envelopes, thinking-tag splitting.
#ifndef RUNNER_TEMPLATE_H
#define RUNNER_TEMPLATE_H

#include <stddef.h>
#include <stdbool.h>
#include "tokenizer.h"

// MISTRAL is [INST] framing without llama-2's <<SYS>> block: Mistral's own
// template rejects a system role outright, so the text is folded into a user
// turn rather than wrapped in markers it never saw in training. It is NOT
// llama-2's renderer with that block swapped out, which is what it was until
// 2026-08-14 -- see TMPL_MISTRAL_V1 below for what that cost.
enum { TMPL_CHATML, TMPL_LLAMA2, TMPL_LLAMA3, TMPL_ZEPHYR, TMPL_GEMMA,
       TMPL_GEMMA4, TMPL_MISTRAL, TMPL_PHI3, TMPL_APERTUS, TMPL_ORNITH,
       TMPL_MUSE, TMPL_GRANITE,
       // gpt-oss / OpenAI Harmony. Channel-structured: the assistant writes
       // an `analysis` message before its `final` one, and both ride the
       // same <|start|>role<|channel|>name<|message|> framing.
       TMPL_HARMONY,
       TMPL_RAW,
       // ChatML whose own template declares <think>: Qwen3 and relatives.
       // Split from TMPL_CHATML only so the thinking control below has
       // somewhere to attach -- rendering is otherwise identical, and every
       // tmpl-dependent branch elsewhere keys on TMPL_ORNITH, so this falls
       // into the same generic path TMPL_CHATML does.
       TMPL_CHATML_THINK,
       // The other two Mistral instruction framings in circulation. TMPL_MISTRAL
       // above is the v0.3 form -- shared byte-for-byte with
       // Mistral-Small-Instruct-2409 and the derivative population that copied
       // it -- and is what `--chat-template mistral` and any Mistral-shaped
       // template this build cannot place select.
       //
       //   TMPL_MISTRAL       "[INST] "  ... "[/INST]"    v0.3 / Small-2409
       //   TMPL_MISTRAL_V1    " [INST] " ... " [/INST]"   v0.1 / v0.2
       //   TMPL_MISTRAL_NEMO  "[INST]"   ... "[/INST]"    Nemo-Instruct-2407
       //
       // Whitespace, and not cosmetic whitespace: this is SentencePiece, where
       // `[INST] What is 2+2? [/INST]` is 11 tokens and
       // `[INST] What is 2+2?[/INST]` is 10 -- measured on
       // Mistral-7B-Instruct-v0.3 on hardware. Every divergent space is a token
       // in a position the model never trained on.
       //
       // They also disagree about something bigger than spacing: v0.1/v0.2 fold
       // the system text into the FIRST user turn, v0.3 and Nemo into the LAST.
       //
       // Until 2026-08-14 runner rendered all Mistral checkpoints through
       // llama-2's case with only the <<SYS>> block swapped out, producing a
       // FOURTH framing (`[INST] ` ... ` [/INST]`, system into the first turn)
       // that matches no Mistral checkpoint at all. It was never trying to be
       // Mistral.
       TMPL_MISTRAL_V1, TMPL_MISTRAL_NEMO,
       // gemma-4 ships THREE chat-template revisions and they are NOT one
       // renderer. The mainline template (12B, 26B-A4B, 31B; sha ae53464b)
       // pre-seeds an empty CLOSED thought block '<|channel>thought\n
       // <channel|>' on the thinking-OFF generation prompt; the E-series
       // (E2B sha 0a2c8073, E4B sha 241c50d8) does NOT. Pre-seeding it is a
       // measured NO-OP on mainline (byte-identical output) but CATASTROPHIC
       // on the E-series (E2B planning 0.650 -> 0.250, reasoning-leak), so the
       // two must render that one site differently. TMPL_GEMMA4 stays the
       // E-series id and its every behaviour; TMPL_GEMMA4_MAINLINE differs
       // from it ONLY at that generation-prompt site. Every other tmpl branch
       // (tool protocol, sampler, parse) treats them the same via is_gemma4().
       TMPL_GEMMA4_MAINLINE,
       // Granite 4.2 (ibm-granite/granite-4.2-*, chat_template.jinja of
       // 2026-09-06): ChatML framing carrying the Qwen3-Coder function-XML
       // tool protocol, which is ornith's wire shape but NOT ornith's
       // renderer. The reference ALWAYS opens with a system turn (empty when
       // none was given), accepts a system turn anywhere, seeds every
       // assistant turn that carries no thought block with the CLOSED
       // `<think></think>`, reduces the reasoning of assistant turns that
       // precede the last user turn to that same closed block, and asks for
       // no reasoning with `<think></think>` where ornith writes
       // `<think>\n\n</think>\n\n`. Detected by that literal
       // `<think></think>`, which ornith's template never carries.
       TMPL_GRANITE42,
       // Nemotron 3.5 Lightning (nemotron_h_moe, the GGUF's own template,
       // 2026-10-09): granite 4.2's template with Qwen3-Coder's nested-XML
       // tool declarations (`<function><name>`, `<parameters>`, `enum` after
       // the description, `required` after the other keys) in place of
       // granite's JSON ones, no `defer_loading` skip, a `reasoning_content`
       // fold of `<think>\n` reasoning `</think>` (no newline around the
       // close), an untrimmed text after the seeded block on a truncated
       // tool-call turn, and no low-effort suffix. Every other behaviour is
       // granite 4.2's (tmpl_granite42_like). Detected by the `<parameters>`
       // block granite 4.2 never writes; it was served as granite42 until a
       // lab report.
       TMPL_NEMOTRON35,
       // Qwen 3.8 (Qwen/Qwen3.8-27B chat_template.jinja, 2026-09-06). Not
       // Qwen3's template rendered by TMPL_CHATML_THINK, which is what the
       // GGUF's own template was detected as until the conformance gate was
       // pointed at it: Qwen 3.8 opens EVERY conversation with a reasoning
       // effort preamble in the system turn (xhigh by default, low as an
       // alternative, medium meaning none) whenever thinking is on, keeps
       // the thought block of every historical assistant turn
       // (preserve_thinking, default true), writes the function-XML tool
       // protocol ornith inherited from Qwen3-Coder rather than Qwen3's
       // JSON one, and trims every content it renders. Detected by that
       // preamble's fixed text.
       TMPL_QWEN38,
       // Qwen3-Coder: non-thinking ChatML with function/parameter XML.
       TMPL_QWEN3_CODER,
       // Qwen 3.5 (Qwen/Qwen3.5-4B, -9B and -0.8B chat_template.jinja). It is
       // ornith's template with ONE rule changed: an assistant turn at or
       // before the last user query is replayed WITHOUT a thought block,
       // where ornith opens every assistant turn with one. Rendered as
       // ornith until the golden pass of 2026-09-07 put it against its own
       // publisher (7 of 22 cases drifted on the 4B). Detected by the
       // `loop.index0 > ns.last_query_index` test ornith's template lacks.
       TMPL_QWEN35,
       // The 0.8B's template is the same again with the thinking DEFAULT
       // inverted: an unset enable_thinking renders the CLOSED block, and
       // only an explicit true opens one. Detected by the literal
       // `enable_thinking is true`.
       TMPL_QWEN35_NOTHINK,
       // Phi-4-mini (microsoft/Phi-4-mini-instruct). NOT Phi-3.5's framing:
       // `<|role|>` content `<|end|>` with no newline anywhere, where
       // Phi-3.5 writes one after the role tag and one after `<|end|>`.
       // Served through TMPL_PHI3 until the golden pass of 2026-09-07 (20
       // of 22 cases carried newline tokens the model never saw). Its
       // template builds the role tag by concatenation, so it is detected
       // by the `<|/tool|>` literal only it writes.
       TMPL_PHI4,
       // NVIDIA Nemotron Nano 9B v2 (nvidia/NVIDIA-Nemotron-Nano-9B-v2
       // tokenizer_config.json). `<SPECIAL_10>System`, `<SPECIAL_11>User`,
       // `<SPECIAL_11>Assistant`, turns closed by `<SPECIAL_12>`; a system
       // header on EVERY conversation; `/think` and `/no_think` control
       // strings read from system and user text and removed from it; a
       // generation prompt that opens `<think>\n` or writes the closed
       // `<think></think>`. Unrecognised until 2026-10-02, when the file
       // was served on llama-2 markup with a loud banner (19 of 22 cases).
       // Native <AVAILABLE_TOOLS>/<TOOLCALL> declarations, history,
       // grammar, and buffered/streamed mapping are implemented.
       TMPL_NEMOTRON,
       // Hermes 4 (NousResearch/Hermes-4-*). ChatML framing and the Hermes
       // JSON tool call, which Qwen2.5 shares, but its own system turn: always
       // present, the caller's or one of two fixed prompts (a standard one,
       // and a "deep thinking" one when thinking is asked for, which is how
       // this family is told to reason: the generation prompt never carries a
       // thought block), with its own tools preamble appended. Rendered as
       // chatml-think until 2026-10-02 (15 of 22 cases). Detected by the
       // standard prompt's text.
       TMPL_HERMES4,
       // Granite 4.0-H and 4.1 (ibm-granite chat templates that carry a
       // <tool_call> block). TMPL_GRANITE's framing exactly, plus the tool
       // protocol those templates define: declarations appended to the
       // system turn, the Hermes JSON call, results folded into a user turn.
       // Granite 3.x shares the framing and not the protocol, so it stays on
       // TMPL_GRANITE and the generic envelope.
       TMPL_GRANITE4,
       // What template_detect returns when NOTHING matched. It renders
       // llama-2 markup, because changing what unrecognised models render is
       // a behavioural decision and not this constant's job -- but it is a
       // DISTINCT id, so "we recognised Llama-2" and "we recognised nothing"
       // stop being the same answer. They were the same answer until
       // 2026-08-15, and that conflation is why per-turn BOS could not be
       // added: the literal <s> would have reached every unrecognised model,
       // and in a vocabulary without <s> as a special it tokenizes as three
       // characters, which is worse than the token merely being absent.
       // Not reachable through --chat-template: you cannot ask for it.
       TMPL_LLAMA2_FALLBACK,
       // JetBrains Mellum2.1 Thinking: Qwen3's ChatML, think and Hermes-JSON
       // tool shape, with ONE rule changed: a thought block is written on an
       // assistant turn after the last user query only when its reasoning
       // is non-empty, so a trailing assistant message gets no empty thought
       // block (Qwen3's template writes one there). Detected by the
       // normalize_content macro its template opens with. Appended last so
       // no other family's value moves.
       TMPL_MELLUM,
       // openbmb MiniCPM5 (openbmb/MiniCPM5-2B chat_template.jinja): Qwen3's
       // ChatML and think shape with three rules of its own. A thought block
       // is written on EVERY historical assistant turn (empty when the turn
       // carries no reasoning and no <think> tags of its own, Qwen3 writes
       // it only on the trailing one); enable_thinking true pre-seeds the
       // generation prompt with '<think>\n' (Qwen3 emits nothing and lets
       // the model open its own); and its tool protocol is XML, not the
       // Hermes JSON block: declarations in the system turn under its own
       // guideline text, calls as <function name="N"><param name="K">V
       // </param></function> with a CDATA wrapper where a string value holds
       // '<', '&' or a newline, results as Qwen3's <tool_response> user turn.
       // Detected by the example call its guideline text spells. Its tokenizer
       // pre is `minicpm5` (tokenizer.c). Appended last.
       TMPL_MINICPM5,
       // Arcee Trinity Nano (arcee-ai/Trinity-Nano-Preview): Qwen2.5's ChatML
       // and Hermes tool protocol, with the assistant span wrapped in
       // {% generation %} ... {% endgeneration %} tags that do not trim the
       // whitespace around them. Read as transformers reads them (block tags,
       // trim_blocks and lstrip_blocks on), they leave a blank line after
       // each assistant header and after each assistant turn's <|im_end|>:
       // `<|im_start|>assistant\n\n` CONTENT `<|im_end|>\n\n`. The generation
       // prompt keeps the single newline. Detected by the untrimmed
       // `{% generation %}` in a ChatML template. Appended last.
       TMPL_TRINITY,
       // Liquid LFM2 / LFM2.5 (LiquidAI/LFM2.5-*-Instruct chat_template.jinja):
       // ChatML with BOS first, the declared tools as one `List of tools:
       // [JSON, JSON]` line in the system turn, a thought written as
       // `<think>...</think>` on the LAST assistant turn only (earlier turns
       // keep what follows their last `</think>`, trimmed), tool results as
       // their own `tool` turn, and a turn's calls as one Python list,
       // `<|tool_call_start|>[name(key=value, ...)]<|tool_call_end|>`, the two
       // markers control tokens. Detected by `List of tools: [` beside the
       // opening marker. Appended last.
       TMPL_LFM2,
       // XiaomiMiMo MiMo-V2.6-Distill-Qwen-9B (chat_template.jinja, byte-
       // identical to the bartowski GGUF's): ChatML with NO newline after
       // <|im_end|>, every assistant turn written `<think>R</think>` + content
       // (R empty when there is no reasoning), a `tool` role kept as its own
       // turn, tools declared in a system turn of their own ahead of the
       // caller's, and calls in a compact form of Qwen3-Coder's XML:
       // <tool_call><function=N><parameter=K>V</parameter></function></tool_call>
       // with no newlines. enable_thinking false closes an empty block on the
       // generation prompt; true and absent leave it bare. Detected by that
       // compact opener, which no other template writes. Appended last.
       TMPL_MIMO,
       // XHToken Spark 2.5 / 2.6 (Spark-X2.6-4B chat_template.jinja): every
       // turn opens with <｜start▁of▁sentence｜> and a role tag (<|System|>,
       // <|User|>, <|Bot|>, <|Tool|>) and closes with <｜end▁of▁sentence｜>; a
       // system block is ALWAYS written (a default sentence, the thinking-
       // effort instructions, the tools as compact JSON, then the caller's
       // first system message); thinking defaults on and pre-seeds `<think>`,
       // `</think>` when off; effort is high (default) or low; a history
       // assistant turn writes `<think>R</think>` or a bare `</think>`; calls
       // are GLM-style <tool_call>NAME<arg_key>K</arg_key><arg_value>V
       // </arg_value></tool_call>. Detected by its <|Bot|> tag beside the
       // thinking_effort instructions. Appended last.
       TMPL_SPARK,
       // Spark X2.5 (XHToken/Spark-X2.5-4B chat_template.jinja): the X2.6
       // template without the thinking-effort text, with `## Tools` written
       // straight after the default sentence and the declarations as tojson
       // with Python's default separators. Detected by the same tags without
       // thinking_effort. Appended last.
       TMPL_SPARK25,
       // IFM K2-Horizon (the template embedded in IFM's K2-Horizon-0.9B and
       // -7B GGUFs, 43,040 bytes, the same in both): turns are
       // <|ifm|im_start|>ROLE\nCONTENT<|ifm|im_end|> with nothing between
       // them; tools become the system turn (a fixed preamble, the
       // declarations rendered as markdown, or as one JSON line each when a
       // schema uses a construct the markdown renderer does not handle, the
       // call-format instructions, then the caller's system text); every
       // assistant turn writes a think block, `<ifm|think>\nR\n</ifm|think>\n`
       // or an empty `<ifm|think>\n</ifm|think>\n`; calls are
       // <ifm|tool_calls> blocks of <ifm|tool_call>NAME then
       // <ifm|arg_key>K</ifm|arg_key> <ifm|arg_value>V</ifm|arg_value> pairs.
       // Thinking defaults on and pre-seeds `<ifm|think>\n`; off writes the
       // closed empty block. Detected by its <|ifm|im_start|> turn token.
       // Appended last.
       TMPL_K2H,
       // RecursiveSI flint (chat_template.jinja at 2a247be0): GLM-style role
       // tokens (<|system|>, <|developer|>, <|user|>, <|assistant|>,
       // <|observation|>) with no end-of-turn token between turns; caller text
       // passes through the template's flint_text (NUL and think tags
       // dropped, role tokens HTML-escaped); tools are a <|system|> turn of
       // their own ahead of the first non-system message; an assistant turn
       // writes <think>R</think> only when it carries reasoning; calls are
       // Spark's <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>
       // </tool_call>; tool results are <|observation|><tool_response>R
       // </tool_response>. The generation prompt is <|assistant|> alone.
       // Detected by its flint_text macro. Appended last.
       TMPL_FLINT,
       // LFM2.5-8B-A1B (the template LiquidAI/LFM2.5-8B-A1B-GGUF carries, an
       // earlier revision of the LFM2.5 template): TMPL_LFM2's framing, with
       // a string argument written raw between single quotes (no escaping), a
       // list or scalar argument as Python's str() of it, the thought read
       // from `thinking` alone and kept on every assistant turn after the
       // last user turn, and CONTINUE_FINAL_MESSAGE_TAG moved only on a turn
       // that carries tool_calls. Detected by that raw-string line beside
       // the LFM2 markers. Appended last.
       TMPL_LFM2_8B,
       // LFM2.5-24B-A2B (LiquidAI/LFM2.5-24B-A2B chat_template.jinja, the same
       // bytes in its GGUF): TMPL_LFM2 with the thought kept on every assistant
       // turn after the last user turn instead of the last assistant turn
       // only, and the generation prompt opening the thought,
       // `<|im_start|>assistant\n<think>`. Detected by that generation prompt
       // beside the LFM2 markers. Appended last.
       TMPL_LFM2_24B,
       // Qwen 3.6 (Qwen/Qwen3.6-35B-A3B chat_template.jinja, the same bytes
       // in bartowski's GGUF): TMPL_QWEN35 with two rules changed. A
       // `preserve_thinking` switch keeps the thought on every assistant
       // turn (undefined by default, which keeps it after the last user
       // query only, as 3.5 does), and a tool-call argument that is not a
       // string is written as JSON (`false`, `null`) where 3.5 writes
       // Python's str() of it (`False`, `None`). Detected by the
       // `preserve_thinking` test beside 3.5's last-query test. Appended
       // last.
       TMPL_QWEN36,
       // Mistral's [SYSTEM_PROMPT] form, as the mistral3 GGUFs embed it
       // (Ministral 3 2512; Mistral Small 3.x, Devstral and Magistral write
       // the same frame): BOS, the system text (or the template's own long
       // default) in [SYSTEM_PROMPT]...[/SYSTEM_PROMPT], the tools as one
       // [AVAILABLE_TOOLS]<tojson>[/AVAILABLE_TOOLS] block right after it,
       // then [INST]user[/INST], an assistant turn as its text, its calls as
       // [TOOL_CALLS]name[ARGS]arguments each, and </s>, a result as
       // [TOOL_RESULTS]text[/TOOL_RESULTS]. Detected by the [SYSTEM_PROMPT]
       // literal, which the older three never carry. Appended last.
       TMPL_MISTRAL_SYS };

// The reasoning tags a TEMPLATE family writes, for the families whose
// protocol (not their architecture) defines them: the `<think>` block of
// the Qwen3-derived templates, Granite 4.2 and Ornith. model.c sets the
// architecture's pair where the architecture owns one (qwen3, gemma4,
// Harmony, muse); a template forced onto another architecture, or one whose
// architecture never had a pair (Granite 4.2 is `granitehybrid`), would
// otherwise have no splitter and serve its reasoning as content. Returns
// false, leaving the pointers alone, for a family without one.
bool template_think_tags(int tmpl, const char **open, const char **close);
// Bind them to a model's pair when the architecture set none: called
// wherever a template is resolved for serving or chat, a no-op when the
// architecture owns a pair, so gemma4/Harmony/muse keep theirs.
static inline void template_bind_think_tags(int tmpl, const char **open,
                                            const char **close) {
    if (*open) return;
    template_think_tags(tmpl, open, close);
}

// True for EITHER gemma-4 template family. The mainline and the E-series
// differ at exactly one generation-prompt site (see TMPL_GEMMA4_MAINLINE
// above); every other tmpl-dependent decision -- the native tool protocol,
// tool-call replay, tool parsing, the primed-think probe -- is identical, so
// those sites ask this rather than naming one id and silently breaking the
// other. Added when the mainline id split off TMPL_GEMMA4 (2026-08-19): a
// missed site would leave mainline tool calls speaking the generic protocol.
// The system turn a publisher's own template writes when the caller sends
// none, for the families where that is a FIXED string (not a live date, which
// the runner never embeds). `meta_tmpl` is the model's chat template: the
// text is taken as that template's only when the template itself carries it,
// and only under the family it was written for, so a forced --chat-template
// or an unrelated ChatML model never gets another model's persona. NULL when
// there is none. Until 2026-10-02 the runner sent the user turn alone, which
// put every system-less request outside the model's training distribution
// (golden pass 2026-09-07: SmolLM2 and Granite 4.0-H, 20 cases). `tools` says
// the request declares tools: some templates keep their default ahead of the
// declarations (Qwen2.5), others replace it with them (Granite 4.0-H).
const char *template_default_system(int tmpl, const char *meta_tmpl,
                                    bool tools);

// What a request that leaves parallel tool calls unsaid gets (owner,
// 2026-10-09): several calls per turn, as OpenAI's API and Anthropic's
// documented defaults have it, on every protocol (gemma-4 and Muse kept one
// call until 2026-10-09, while their parallel grammars were a fixed PAIR;
// suite R4.26.19). A tool_choice that
// names one function (`choice` an object, in either the chat or the flat
// responses form) defaults to one call: OpenAI's forced function calls
// exactly one. NULL for a surface whose named choice allows several
// (Anthropic's {"type":"tool"} without disable_parallel_tool_use). An
// explicit value always wins.
struct jv;
bool tool_parallel_default(int tmpl, const struct jv *choice);

static inline bool is_gemma4(int tmpl) {
    return tmpl == TMPL_GEMMA4 || tmpl == TMPL_GEMMA4_MAINLINE;
}

// The ornith family: ornith itself and the Qwen 3.5 templates it was derived
// from. They share the function-XML tool protocol, the <tool_response> user
// turn, the system-turn merge and the think tags; the Qwen 3.5 pair differ
// only in which assistant turns carry a thought block and in the thinking
// default, both handled where the block is written.
static inline bool tmpl_ornith_like(int t) {
    return t == TMPL_ORNITH || t == TMPL_QWEN35 || t == TMPL_QWEN35_NOTHINK ||
           t == TMPL_QWEN36;
}
// Granite 4.2 and the template derived from it (Nemotron 3.5): the same
// framing, think seeding, call syntax and system fold; they differ in the
// tool declarations and the few spots TMPL_NEMOTRON35's comment lists.
static inline bool tmpl_granite42_like(int t) {
    return t == TMPL_GRANITE42 || t == TMPL_NEMOTRON35;
}
// The families whose native tool call is the Hermes JSON block
// (<tool_call>{"name": ..., "arguments": {...}}</tool_call>): one grammar,
// one parser, one replay form.
static inline bool tmpl_chatml_think(int t) {
    return t == TMPL_CHATML_THINK || t == TMPL_MELLUM;
}
// Plain ChatML: the generic family and Trinity, which differs from it only
// in the blank lines around its assistant turns (render_chatml).
static inline bool tmpl_chatml_plain(int t) {
    return t == TMPL_CHATML || t == TMPL_TRINITY;
}
static inline bool tmpl_hermes_json(int t) {
    return tmpl_chatml_plain(t) || tmpl_chatml_think(t) || t == TMPL_HERMES4 ||
           t == TMPL_GRANITE4;
}
static inline bool tmpl_spark(int t) {
    return t == TMPL_SPARK || t == TMPL_SPARK25;
}
// The families whose own template writes a `developer` message as a turn of
// its own (flint's <|developer|>). Every other family folds it into the
// system slot -- gemma-4 by name, the rest by having no such role -- which is
// what the serving layers' developer-to-system mapping does for them.
static inline bool tmpl_renders_developer(int t) {
    return t == TMPL_FLINT;
}
// The families whose calls are Spark's GLM-style <arg_key> blocks
static inline bool tmpl_argkv(int t) {
    return tmpl_spark(t) || t == TMPL_FLINT;
}
// LFM2's template revisions: one framing, their own argument and thought rules
static inline bool tmpl_lfm2(int t) {
    return t == TMPL_LFM2 || t == TMPL_LFM2_8B || t == TMPL_LFM2_24B;
}
static inline bool tmpl_qwen35(int t) {
    return t == TMPL_QWEN35 || t == TMPL_QWEN35_NOTHINK || t == TMPL_QWEN36;
}

// How the generation prompt should treat a thinking model.
//
// THINK_DEFAULT means "whatever this model family's reference template does",
// and that is deliberately NOT one answer for all families:
//   gemma-4  reference defaults enable_thinking FALSE -> pre-seed an empty
//            thought block, suppressing reasoning
//   Qwen3    reference defaults enable_thinking TRUE  -> emit nothing, the
//            model opens its own <think>
// A single boolean cannot express that, which is why this is tri-state: the
// absence of a request field has to mean "match the reference", not "false".
enum { THINK_DEFAULT = 0, THINK_ON, THINK_OFF };
// Reasoning effort, OR'd into the `thinking` argument by a caller whose
// family renders it: TMPL_QWEN38 (xhigh/medium/low, its template raises on
// anything else) and TMPL_MUSE (`reasoning_strength` low/medium/high, the
// absent value rendering as the reference's default "high"). The low
// nibble stays the THINK_* mode; every other family never sees these bits.
enum { THINK_EFFORT_XHIGH = 0, THINK_EFFORT_MEDIUM = 0x10,
       THINK_EFFORT_LOW = 0x20, THINK_EFFORT_MASK = 0x30,
       THINK_MODE_MASK = 0x0f };
// Muse only: how a BARE tool name (no dot) is written into the system turn's
// "# Valid recipients" line. The reference template renders every name as a
// namespace pattern, `"read.*"`, and Meta's Muse-Glimmer-30B discussion #60
// (open, not merged) reports the model then emitting `read.filePath`-style
// recipients and failing the first call of every task; its patch renders a
// bare name verbatim, `"read"`, and keeps `"ns.*"` for a dotted one. The
// reference rendering stays the default here (byte-identity against the
// reference is the anchor, and Kvist was trained and gated on it); a
// request says `bare_recipients: true|false` and a server runs
// `--bare-recipients` to flip the default. Neither bit set means "the
// server default"; other families never read these bits.
enum { THINK_BARE_RECIPIENTS = 0x40, THINK_NS_RECIPIENTS = 0x80,
       THINK_RECIPIENTS_MASK = 0xc0 };

struct jv;
typedef struct {
    const char *role, *content;
    // Native templates whose tool-result turn names the invoked function use
    // this optional field. Ordinary messages leave it NULL.
    const char *name;
    // Harmony history may carry separate analysis/commentary assistant
    // messages before a final answer or recipient-bearing call.
    const char *channel;
} chat_msg;
int         template_detect(const char *meta_tmpl, tokenizer *tok);
int         template_from_name(const char *name); // -1 if unknown
const char *template_name(int tmpl);

// Refuse histories whose ordinary roles cannot be represented by a template.
// Tool results end the ordinary alternating prefix: their call/result protocol
// is validated separately, but their presence must never forgive a malformed
// user/assistant prefix that came before them.
bool template_roles_valid(int tmpl, const char *const *roles, int n,
                          bool allow_mid_system, char *err, size_t err_cap);
// render messages; add_assistant appends the assistant generation prefix.
// returns bytes written (excl. NUL)
//
// `thinking` is THINK_DEFAULT / THINK_ON / THINK_OFF.
//
//   TMPL_CHATML_THINK  Qwen/Qwen3-* tokenizer_config.json
//   TMPL_MELLUM        JetBrains/Mellum2.1-12B-A2.5B-Thinking chat_template.jinja
//   TMPL_MIMO          XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B chat_template.jinja
//     (OFF appends '<think></think>', DEFAULT and ON nothing)
//   TMPL_SPARK         XHToken/Spark-X2.6-4B chat_template.jinja: enable_thinking
//     defaults TRUE (`| default(true)`), so DEFAULT and ON pre-seed '<think>' and
//     OFF writes '</think>'; THINK_EFFORT_LOW selects the low-effort system text,
//     everything else the high one
//   TMPL_K2H           IFM K2-Horizon GGUF template: enable_thinking false
//     closes an empty block ('<ifm|think>\n</ifm|think>\n'); DEFAULT and ON
//     open '<ifm|think>\n' (reasoning_effort high, the template's default)
//   TMPL_MINICPM5      openbmb/MiniCPM5-2B chat_template.jinja (think pre-seed
//     on ON, closed block on OFF, nothing on DEFAULT: `enable_thinking is
//     defined` gates both branches)
//     enable_thinking defaults TRUE; the false branch appends
//     '<think>\n\n</think>\n\n' after the assistant header. So DEFAULT and ON
//     emit nothing extra and OFF appends the closed block.
//
//   TMPL_GEMMA4        gemma-4 E-series tokenizer.chat_template (from the GGUF)
//     The generation prompt is '<|turn>model\n' and nothing else, in EITHER
//     mode -- there is no pre-seeded thought block. After a tool response it
//     emits nothing at all, or an OPEN '<|channel>thought\n' when thinking.
//     TMPL_GEMMA4_MAINLINE is byte-identical EXCEPT that its thinking-OFF
//     generation prompt appends the empty CLOSED block '<|channel>thought\n
//     <channel|>' after '<|turn>model\n'. That is the mainline template's own
//     `{%- if not enable_thinking -%}` branch (enable_thinking defaults
//     false), a MEASURED no-op there and catastrophic on the E-series, which
//     is why the two ids exist.
//     Thinking for this family is selected in the FIRST SYSTEM TURN, not at
//     the generation prompt: THINK_ON injects '<|think|>\n' at the top of it,
//     opening that turn even when the caller sent no system message, exactly
//     as the template's own condition does.
//
// THINK_DEFAULT means "render what this family's own template renders", which
// is not the same answer for both -- hence tri-state rather than a bool.
//
// History worth keeping: on 2026-08-08 this file claimed gemma-4's reference
// pre-seeds an empty thought block when thinking is off, and the renderer was
// changed to match. That came from a web summary of llama.cpp's copy of the
// template, not the model's own, and it was wrong. gemma-4-E2B's planning
// score fell 0.575 -> 0.300 and it began emitting reasoning prose as its
// visible answer. Read the artifact, not a description of it.
size_t render_messages(int tmpl, const chat_msg *msgs, int n_msgs,
                       bool add_assistant, int thinking,
                       char *out, size_t cap);
// Structured-tools variant used by native templates. The legacy entry above
// is exactly this call with tools == NULL. SIZE_MAX reports an allocation
// failure inside the renderer; callers must not treat it as a prompt length.
size_t render_messages_with_tools(int tmpl, const chat_msg *msgs, int n_msgs,
                                  bool add_assistant, int thinking,
                                  const struct jv *tools,
                                  char *out, size_t cap);

// Where the turn a render opens starts, decided by the same code that wrote
// the prompt: the completion path, the CLI chat and the tool grammars read
// this instead of reconstructing it from the family, the request and the
// prompt's tail.
typedef struct {
    // The generation prompt leaves the model inside a reasoning segment, so
    // its first token is reasoning: a `<think>` the prompt opened (Granite
    // 4.2, Nemotron 3.5, Qwen 3.8, Qwen 3.5, Ornith, Nemotron Nano),
    // Harmony's analysis channel, Muse's self-addressed turn, or Gemma 4's
    // thought channel after a tool result.
    bool reasoning_open;
    // With reasoning_open, the bytes that opened the segment, as the model
    // would have written them (`<think>\n`, `<|channel>thought\n`, ...); the
    // prompt ends with them, marks aside. A demultiplexer that follows the
    // channels reads the turn from here. NULL otherwise.
    const char *opener;
    // The thinking mode the prompt was rendered under: THINK_DEFAULT,
    // THINK_ON or THINK_OFF, whichever spelling the request used for it.
    int think_mode;
} render_state;

// render_messages_with_tools() that also reports the render's state. `st`
// may be NULL; when given it describes the last call's prompt.
size_t render_messages_state(int tmpl, const chat_msg *msgs, int n_msgs,
                             bool add_assistant, int thinking,
                             const struct jv *tools,
                             char *out, size_t cap, render_state *st);

// chat tool-call convention (template.c; sbuf/jv live in json.h)
struct sbuf;
struct jv;
// read the per-request thinking control (top level or chat_template_kwargs);
// THINK_DEFAULT when absent, so a silent request renders what the model's own
// reference template would render
bool req_thinking_mode_valid(struct jv *req);
int req_preserve_thinking(struct jv *req);
int req_thinking_mode(struct jv *req);
// `reasoning_effort` from the request (top level or chat_template_kwargs):
// THINK_EFFORT_* for xhigh (also absent), medium and low; -1 for any other
// value, which the template itself rejects.
int req_reasoning_effort(struct jv *req);
// `reasoning_strength` from the request (top level or chat_template_kwargs),
// the Muse reference's kwarg: THINK_EFFORT_XHIGH for high (also absent),
// MEDIUM, LOW; -1 for any other value.
int req_reasoning_strength(struct jv *req);
// Spark's `reasoning_effort`: high (the default, xhigh read as high) or low;
// -1 for anything its template refuses (medium included)
int req_spark_effort(struct jv *req);
// K2-Horizon's `reasoning_effort`: 0 for high (also absent), the only effort
// this build serves; -1 with *why set for anything else. Its template opens
// medium as <ifm|think_fast> and low as <ifm|think_faster>, blocks the
// reasoning splitter here does not follow yet (it reads one tag pair per
// model), so those two are refused by name rather than served with the
// reasoning leaking into the content; any other value the template itself
// refuses.
int req_k2h_effort(struct jv *req, const char **why);
// The declarations' faults that the family's own template refuses to render
// (K2-Horizon's validate_tools / validate_schema raise on them): true when
// the template would render `tools`; false with `err` naming the fault,
// which the caller answers 400. Every other family renders any schema the
// envelope builder admitted and gets true.
bool template_tools_check(int tmpl, const struct jv *tools, char *err,
                          size_t err_cap);
// The effort bits a request asks of the families whose template renders one
// (Qwen 3.8, Spark X2.6, Muse, K2-Horizon), OR'd into `thinking`; 0 for
// every other family and for an absent value; -1 with *why set for a value the family's template
// refuses. One reading for the chat and Responses surfaces.
int req_template_effort(int tmpl, struct jv *req, const char **why);
// `bare_recipients` from the request (top level or chat_template_kwargs):
// THINK_BARE_RECIPIENTS for true, THINK_NS_RECIPIENTS for false, 0 when
// absent (the server default applies), -1 for a value that is not a boolean.
int req_bare_recipients(struct jv *req);
// the server default for a request that does not say: reference (false) unless
// `--bare-recipients` was given
void template_set_bare_recipients(bool bare);
bool template_bare_recipients(void);
// render OpenAI "tools" declarations as a system turn (no-op when absent)
void tools_render(const struct jv *tools, struct sbuf *out);
void tools_render_for(int tmpl, const struct jv *tools, struct sbuf *out);
// Fold the caller's system text into the family's declaration turn `ts`
// (a tools_render_for block) the way its reference does: ornith appends
// the text after the declarations behind a blank line, granite 4.2 puts
// the text first and the declarations after one (chat_template.jinja:
// 49-55). Returns true when the text was folded and the caller must NOT
// render its own system turn; false leaves `ts` alone (every other family,
// an empty block, no text) and the caller renders the turn as usual. One
// helper for Chat, Responses and Messages, so a declared tool with a system
// prompt is the same prompt through every door.
bool tools_system_fold(int tmpl, struct sbuf *ts, const char *system);
// Replay an assistant turn's tool_calls in the family's own syntax.
//
// `turn_has_text` is whether THAT TURN carried visible content, and it cannot
// be recovered from `out`: by the time this runs, `out` may already hold
// framing the caller added (ornith's <think> block), so an empty-buffer test
// answers a different question. ornith is the family that needs it -- its
// reference opens the first call with "\n\n" after text and with nothing
// without it (ornith.jinja:106-109).
void tool_history_render_for(int tmpl, const struct jv *calls,
                             bool turn_has_text, struct sbuf *out);
// Append a literal the RUNNER owns (a thought block's framing composed for
// a replayed assistant turn, say) to a content buffer, bracketed in the
// prompt marks, so the tokenizer reads its control tokens as control tokens
// exactly as it does the template's own bytes. Caller text goes in with
// sb_put, unmarked. See PROMPT_RAW_OPEN in tokenizer.h.
void prompt_lit(struct sbuf *b, const char *lit);
// Build a one-element OpenAI tool_calls array from a name and its arguments
// (a JSON string). Owned jv or NULL on OOM. The adapter the typed surfaces use
// to reach tool_history_render_for's serializer from their own call vocabulary.
struct jv *tool_call_synth(const char *name, const char *args_json, const char *id);
// Assemble an assistant turn's flattened content in the family's native tool
// protocol -- the same bytes handle_chat produces for a live chat turn.
// `text` is the turn's visible text (may be NULL), `calls` an OpenAI-shaped
// tool_calls array; `out` must be empty (or hold only caller framing). Sets
// *turn_name to the name the assistant chat_msg must carry (muse) or NULL.
void assistant_calls_render(int tmpl, const char *text, const struct jv *calls,
                            struct sbuf *out, const char **turn_name);
// K2-Horizon's assistant turn body (after `<|ifm|im_start|>assistant\n`), as
// its template writes it: a think block always, `<TAG>\nR\n</TAG>\n` or the
// empty `<TAG>\n</TAG>\n`, then the content without leading newlines, then
// the calls behind a newline when there was content. `reason` is the turn's
// reasoning field (NULL when it carries none: the content is then searched
// for a closed block, as the template does) and `tag` the block it names
// (NULL for ifm|think).
void k2h_assistant_body(const char *reason, const char *tag, const char *text,
                        const struct jv *calls, struct sbuf *out);
// Wrap a replayed tool RESULT in the family's native protocol; returns the role
// ("user" for ornith's <tool_response> user turn, "tool" otherwise) and writes
// the wrapped content to `out`.
const char *tool_result_wrap(int tmpl, const char *result, struct sbuf *out);
// The same, given the id of the call the result answers (Mistral's result
// block names it). NULL when the surface has none.
const char *tool_result_wrap_id(int tmpl, const char *result, const char *call_id,
                                 struct sbuf *out);
// A Mistral-shaped tool call id (9 alphanumerics) for any id: kept when it
// already has that shape, otherwise derived from its hash, call and result alike.
void mistral_call_id(const char *id, char out[10]);
// Resolve a tool result's native turn name from message.name or from its
// tool_call_id and a preceding assistant tool_calls entry. Borrowed pointer.
const char *tool_result_name(const struct jv *messages, int message_index);
// parse tool-call blocks out of content into OpenAI tool_calls items;
// returns the call count, content is compacted in place
int  tool_calls_parse(struct sbuf *content, struct sbuf *tc);
int  tool_calls_parse_for(int tmpl, struct sbuf *content, struct sbuf *tc);
// ------------------------------------------- strict tool-call envelope
//
// OpenAI tools[] compiled into a discriminated union — one branch per tool
// plus a `final` branch for an ordinary or schema-constrained answer:
//
//   {"tool":"get_weather","args":{"city":"Oslo"}}
//   {"tool":"final","args":{"content":"it is cold"}}
//
// Constraining sampling to that union is what makes the guarantee: the model
// cannot name a tool that was not declared, cannot invent an argument key or
// get its type wrong, and a max_tokens truncation closes to a document that
// is still a legal call. That replaces parsing hopeful output afterward,
// which is what tool_calls_parse above does and remains as the fallback for
// requests that do not opt in.
enum { TCH_AUTO, TCH_REQUIRED, TCH_NONE, TCH_NAMED };

// The native tool protocol a request uses, or TP_GENERIC for the strict JSON
// envelope. Exactly one holds per envelope: a single field makes the
// mutually-exclusive families unrepresentable as a boolean combination (the
// old five flags permitted states like atem+harmony that cannot exist).
enum tool_proto {
    TP_GENERIC = 0,   // strict JSON envelope, or no native protocol
    TP_ATEM,          // Muse native recipient + <atem:invoke>
    TP_MUSE_USER,     // Muse: generic JSON override after a to=user header
    TP_HARMONY,       // gpt-oss native channels + functions recipient
    TP_GEMMA4,        // gemma4 native <|tool_call>call:NAME{...}
    TP_QWEN,          // Qwen2.5/Qwen3 native <tool_call>{JSON}</tool_call>
    TP_QWEN_XML,      // Qwen3-Coder function/parameter XML
    TP_MUSE_PLAIN,    // Muse: stream a schema payload after a to=user header
    TP_NEMOTRON,      // Nemotron Nano native <TOOLCALL>[{JSON}, ...]</TOOLCALL>
    TP_MISTRAL,       // Mistral v0.3 / Nemo native [TOOL_CALLS][{JSON}, ...]
    TP_MISTRAL_SYS,   // Mistral [SYSTEM_PROMPT] form: [TOOL_CALLS]name[ARGS]{JSON} a call
    TP_MINICPM_XML,   // MiniCPM5 <function name="N"><param name="K">V</param></function>
    TP_LFM2,          // LFM2 <|tool_call_start|>[name(key=value, ...)]<|tool_call_end|>
    TP_ARGKV,         // Spark <tool_call>N<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
    TP_IFM_ARGKV,     // K2-Horizon: the same pairs in <ifm|...> tags, inside <ifm|tool_calls>
    TP_LFM2_RAW,      // LFM2.5-8B: TP_LFM2 with its string values written raw
};
static inline bool proto_lfm2(int p) {
    return p == TP_LFM2 || p == TP_LFM2_RAW;
}

typedef struct {
    int   kind;           // TCH_*
    char *schema_src;     // envelope JSON schema, for schema_compile (owned)
    char *system_turn;    // system message teaching the envelope (owned)
    bool  final_is_text;  // final branch is {"content": "..."} rather than
                          // the caller's own response_format schema
    // parallel_tool_calls: the document becomes {"calls":[<entry>, ...]} —
    // one uniform array whose items are the same discriminated union, so a
    // direct answer is just a one-element array holding the final branch.
    // Bounded by construction: an unbounded array under a token budget is a
    // truncation waiting to happen.
    bool  parallel;
    int   max_calls;
    int   proto;          // enum tool_proto — the native protocol, or TP_GENERIC
    struct jv *tools;     // borrowed request declarations for native compiler
    bool  owns_tools;     // Anthropic translation survives prompt construction
    char *named;          // owned named-tool choice, when kind == TCH_NAMED
    // The native protocol is PARSED but the sampler is not constrained to it:
    // no grammar is compiled, the model writes its turn in the syntax its own
    // template taught it, and the same demultiplexer that serves the
    // constrained turn recognises the calls on the way out. Parsing a native
    // protocol must not depend on a grammar being active -- until 2026-09-14
    // it did, and the families that had no grammar (Qwen 3.8, Granite 4.2,
    // Ornith) streamed their well-formed calls to the client as prose. With
    // no grammar bounding the turn, the demultiplexer's contract loosens
    // where a grammar used to make the strict reading safe: a block that is
    // not a valid call is dropped and reported as a protocol fault instead of
    // voiding the whole turn, a partial block at the end is dropped and never
    // completed, and max_calls is a parser bound rather than a grammar's.
    bool  parse_only;
} tool_envelope;

// True when the envelope carries a model-native protocol: the prompt teaches
// the family's own syntax and the parser reads it. TP_MUSE_USER is the
// generic JSON payload behind a Muse recipient header, so it is not native
// in this sense and still wants the generic teaching turn.
static inline bool tool_envelope_native(const tool_envelope *e) {
    return e && e->proto != TP_GENERIC && e->proto != TP_MUSE_USER;
}

enum tool_envelope_result {
    TOOL_ENVELOPE_OOM     = -2,
    TOOL_ENVELOPE_INVALID = -1,
    TOOL_ENVELOPE_NONE    = 0,
    TOOL_ENVELOPE_READY   = 1,
};

// Build the envelope for one request. `final_schema` is the caller's
// response_format schema, or NULL for a plain text answer. Returns one of
// enum tool_envelope_result; a READY envelope must be freed with
// tool_envelope_free. INVALID is client input, while OOM is a server fault.
int  tool_envelope_build(struct jv *tools, struct jv *tool_choice,
                         struct jv *final_schema, tool_envelope *out,
                         char *err, int errcap);
// Same, but `parallel` opts into the bounded multi-call form. Buffered
// surfaces only: the streaming demultiplexer still tracks one call per turn.
int  tool_envelope_build_ex(struct jv *tools, struct jv *tool_choice,
                            struct jv *final_schema, bool parallel,
                            tool_envelope *out, char *err, int errcap);
void tool_envelope_free(tool_envelope *e);

// Select the model-native tool-DECLARATION protocol for a slot's template
// family on `env`, so /v1/chat/completions, /v1/responses and /v1/messages all
// teach declared tools in the format the family was trained on -- gemma4's
// `<|tool>declaration:...`, muse's atem schema block, Harmony's `functions`
// namespace -- rather than the generic block. Sets env's native flags
// (atem/muse_user_header/harmony/gemma4) and its borrowed `tools` pointer for
// the families that render their own declarations, and returns the declarations
// to hand render_prompt_alloc (NULL when the family uses the generic block).
// *skip_generic is set true when the caller must NOT emit its generic/strict
// declaration turn because the family renders its own.
//
// `strict` is whether the strict envelope applies (tool_envelope_build returned
// 1 and, for the chat surface, the slot is not ornith, whose native qwen3_xml
// declaration tools_render_for already handles). `atem_tool_calling` selects
// muse's native atem protocol over the generic to=user override; pass true for
// the native default. The returned pointer aliases `tools` -- the caller keeps
// ownership (env->owns_tools is not touched here).
const struct jv *tool_decl_native(int tmpl, bool strict, bool atem_tool_calling,
                                  struct jv *tools, tool_envelope *env,
                                  bool *skip_generic);

// The native tool protocol a template family speaks, as --tool-info and
// runner_telemetry.tool_protocol name it ("gemma4", "qwen3_xml", "atem",
// "harmony", "qwen_json"), or "generic" with *native false for a family
// that renders the generic declarations and envelope. Derived from
// tool_decl_native's own selection, never a static table.
const char *tool_protocol_name(int tmpl, bool *native);
// The protocol THIS request's envelope runs under: the family's native one,
// or "generic" when the request fell back to the JSON envelope (a schema the
// native syntax cannot enforce) or the family has none.
const char *tool_envelope_protocol_name(const tool_envelope *e);

// Map a generated envelope document back to the OpenAI response shape.
// Returns the NUMBER of tool_calls[] items appended to tc (1 for the ordinary
// single-call envelope, 0..max_calls for the parallel form), 0 for the final
// branch with assistant content appended instead, and -1 when doc is not a
// well-formed envelope. A parallel document may mix the two: any final
// branches contribute content, any tool branches contribute calls.
int  tool_envelope_map(const tool_envelope *e, const char *doc, size_t n,
                       struct sbuf *content, struct sbuf *tc);
int  tool_envelope_map_channels(const tool_envelope *e, const char *doc,
                                size_t n, struct sbuf *reasoning,
                                struct sbuf *content, struct sbuf *tc);
bool muse_user_payload_strip(struct sbuf *payload);

// Streaming counterpart of tool_envelope_map.
//
// tool_envelope_map needs the whole document, which is exactly what a stream
// does not have: the envelope is decided one token at a time, and the client
// must see the decision as it happens rather than after the fact. This is the
// same mapping run incrementally — bytes in, demultiplexed events out — so a
// streamed request reaches the same call as a buffered one.
//
// Nothing is emitted until the branch is known, which is what keeps envelope
// syntax out of the client's `content`: the discriminator is buffered, and by
// the time anything is forwarded it is already known to be either assistant
// text or tool arguments. Argument text is forwarded raw (insignificant
// whitespace removed) so it stays a JSON *string* the caller can execute;
// `final` text is unescaped, matching what the buffered path hands back.
//
// Every callback returns non-zero to stop generation (the client went away);
// that result propagates out of tool_stream_feed unchanged.
typedef struct {
    void *ud;
    int (*reasoning)(void *ud, const char *bytes, int n);
    int (*content)(void *ud, const char *bytes, int n);
    int (*call_begin)(void *ud, const char *name);
    int (*call_args)(void *ud, const char *bytes, int n);
    int (*call_end)(void *ud); // a native turn or a parallel JSON document
                               // may carry another call after this one
} tool_stream_sink;

typedef struct {
    const tool_envelope *env;
    tool_stream_sink     sink;
    int   state;
    char  *head;          // undecided prefix, held back from the client
    size_t head_n, head_cap;
    char *name;           // selected branch, once known (owned)
    bool  called;         // the CURRENT entry selected a tool branch, not
                          // `final` -- reset between entries of a parallel
                          // document, unlike any_called below
    bool  any_called;     // a tool branch was selected by ANY entry so far;
                          // this is what finish_reason "tool_calls" reads
    bool  fault;          // parse-only: a complete block was not a valid call
                          // and was dropped; tool_stream_finish reports it
    bool  skip_ws;        // qwen: drop the framing whitespace after a call
    int   n_calls;        // calls emitted so far (the parse-only bound)
    int   depth;          // nesting inside the value being forwarded
    bool  started;        // the forwarded value has produced its first byte
    bool  in_str, esc;    // JSON string state within that value
    char  pend[16];       // partial escape sequence awaiting more bytes
    int   n_pend;
} tool_stream;

void tool_stream_init(tool_stream *s, const tool_envelope *e,
                      const tool_stream_sink *sink);
int  tool_stream_feed(tool_stream *s, const char *bytes, int n);
// Ends the stream: flushes what the states held back, returns -1 when the
// document could not be mapped (a strict envelope that never resolved, a
// call block still open, or a parse-only block that was not a valid call),
// otherwise a sink's non-zero refusal or 0. `truncated` says the turn ended
// on a budget or a deadline rather than by the model: a parse-only call
// still open then is a cut, not a fault, and finishing reports 0 with the
// partial block dropped (never completed) so the caller keeps "length".
int  tool_stream_finish(tool_stream *s);
int  tool_stream_finish_ex(tool_stream *s, bool truncated);
// The buffered turn of a parse-only envelope, mapped by the demultiplexer
// itself so both paths recognise exactly the same calls: content and the
// OpenAI tool_calls[] items (comma-separated) are appended, *n_calls counts
// the calls, and the return is 0, or -1 when the turn carried a protocol
// fault (a block that was not a valid call, or a call the model left open),
// in which case what WAS valid is still appended. `truncated` as for
// tool_stream_finish_ex.
int  tool_stream_map(const tool_envelope *e, const char *doc, size_t n,
                     bool truncated, struct sbuf *content, struct sbuf *tc,
                     int *n_calls);
// Parse-only bound on calls per turn (the grammar bounds constrained turns).
#define TOOL_STREAM_MAX_CALLS 32
// true once a tool branch was selected, i.e. finish_reason is "tool_calls"
bool tool_stream_called(const tool_stream *s);
void tool_stream_free(tool_stream *s);

// streaming splitter for thinking-tag models: bytes between
// open and close tags, including architectures that interleave them with
// plain text — reach the callback as reasoning (reasoning=1), the rest as
// content (reasoning=0). Partial tags at chunk boundaries are held back until
// they resolve. With open == NULL every byte passes straight through.
typedef int (*think_cb)(void *ud, int reasoning, const char *bytes, int n);
typedef struct {
    const char *open, *close;
    int   state;
    bool  trim_nl;      // think_trim_newlines
    char *buf;
    int   n, cap;
} think_split;
// Harmony's analysis channel as a splitter open/close pair — in DECODED form.
// The splitter sees detokenized text, and Harmony's control tokens
// (<|channel|>, <|message|>, <|end|>, <|start|>) all decode to nothing, so the
// bytes that actually arrive around the channel handoff are the bare words:
//
//   <|channel|>analysis<|message|>            -> "analysis"
//   <|end|><|start|>assistant<|channel|>final<|message|> -> "assistantfinal"
//
// Measured live on gpt-oss-20b at temp 0, whose raw stream reads
// "analysisWe have a conversation...assistantfinal2 + 2 equals **4**."
// This is the same trap muse hit: its markers are the decoded " to=self" /
// "assistant to=user", not the control-token spellings (see
// test_muse_plain_thinking_close_leaves_no_recipient_residue). Writing the
// control-token forms here compiles, renders, and silently never matches.
//
// The close must span the WHOLE handoff. Narrowing it would leak "assistant"
// or "final" into content, which is the residue bug muse recorded.
#define HARMONY_THINK_OPEN  "analysis"
#define HARMONY_THINK_CLOSE "assistantfinal"

void think_init(think_split *t, const char *open, const char *close);
void think_init_reasoning(think_split *t, const char *open, const char *close);
// Newlines at the close tag are framing, not text: drop the reasoning's
// trailing ones before it and the content's leading ones after it, as the
// publisher's template does when it reads a turn back (model_t.think_trim_nl).
void think_trim_newlines(think_split *t, bool on);
int  think_feed(think_split *t, const char *bytes, int n, think_cb cb, void *ud);
int  think_finish(think_split *t, think_cb cb, void *ud); // flush held bytes
void think_free(think_split *t);

#endif // RUNNER_TEMPLATE_H
