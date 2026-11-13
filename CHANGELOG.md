# Changelog

Current release and unreleased changes. Compatibility follows the
[versioning policy](docs/versioning.md).

Older releases (v1.11.0 and earlier) are preserved in the
[changelog archive](docs/changelog-archive.md).

Changes merged since the last release wait in [changes/](changes/README.md), one
file each, until the release folds them in below. Add a new entry there, not here.

## Unreleased

## v1.18.0 - 2026-11-13

- **Apertus: a turn ends where the model closes a tool call** (R2.16). The publisher's generation settings end a reply at `<|tools_suffix|>` as well as at `<|assistant_end|>`; the GGUF files declare only the second, and Runner followed the files. On a turn with no tools declared, the 2509 model sometimes ends its reply with a `display_answers` call (14 of 27 replies over 9 prompts on one host, mostly on arithmetic word problems; none on 160 ordinary questions on another), and Runner then let it run on and write an invented result for that call (`["Answers displayed"]`) into the content. Runner now stops at `<|tools_suffix|>` for this family, as the publisher's own stack does.
  - **Unchanged:** turns with tools declared, which are constrained to the call document and never reach that token; every other family, which does not stop on this token.
  - **Still there:** the call itself, which the model writes before the stop, remains in the content of such a reply.
  - **Measured on the real files, on Linux and Windows CUDA:** every reply that wrote the call ran on before (16 of 16 on one host, 8 of 8 on the other) and none does now; every other token is unchanged, and the pinned outputs of the three Apertus rows are unchanged.
  - **The 2509 rows say so**, and also that 2509 opens no deliberation with thinking on (see R2.15, whose entry named the wrong tokens for 2509 and is corrected).
  - **Windows CUDA pins for the three Apertus rows**: each is byte for byte the Linux pin.

- **Apertus v1.5 8B joins the roster on a third party's text-only extraction** (R4.38.1): one row, andreasmartin's Q8_0 of andreasmartin's extraction of swiss-ai's Apertus-v1.5-8B (swiss-ai publishes no GGUF). The publisher's model also reads images and audio; this file and Runner do not.
  - **Fidelity:** against swiss-ai's BF16 checkpoint computed in float32 (on swiss-ai's own transformers branch, since no release has the class), 0.0036 ± 0.0003 mean KLD and 100% margin-qualified top-1 over 600 positions. Every trunk tensor equals the publisher's within Q8_0's error. Colby's Q4_K_M of the same extraction was measured and not admitted (0.166, 89.8%; its header carries no importance matrix).
  - **The extraction's 200 own output rows** (the publisher's omni special tokens, ids 131072 to 131271, untrained, near the mean row): at temperature 1 about one generated token in 3 million is one of them, invisible (measured over teacher-forced text); greedy decoding never emits one, and nothing excludes them. They move the KLD by less than 1e-8.
  - **The tokenizer** matches the publisher's on every corpus string. The publisher's normalizer rewrites eleven spelled patterns (`<think>`, `</think>`, `<answer>` among them) that Runner reads as characters; the compat check gates the two vocabulary entries this changes. With no system message, the publisher's default persona (which claims images and audio) is not written.
  - **Thinking** is split from the answer on the chat, Responses and Messages surfaces (R2.15), measured on this file. Tool calls and structured output generate the same tokens as before it. CPU and CUDA run the whole model; Metal is not measured; training is refused. The record is [docs/compat-reports/apertus-v1.5-8b-text-2026-11-13](docs/compat-reports/apertus-v1.5-8b-text-2026-11-13/README.md).
  - **Tool results are framed as v1.5's template frames them.** swiss-ai's Apertus-v1.5-8B template writes the results that answer a call between `<|tool_output_start|>` and `<|tool_output_end|>`, where 2509's writes them as one `[`...`]` list; Runner wrote 2509's form for both, one token short on every turn that carried a result.
    - **A named variant, `apertus15`.** It is selected by `<|tool_output_start|>` in the file's template, or in its vocabulary when the file carries no template text (2509's has neither), and `--chat-template apertus15` names it. It differs from `apertus` only in that framing: one renderer, one tool protocol, and one control-token think pair for the reasoning split (R2.15), with the two ids under one predicate.
    - **Conformance:** a new `apertus15` family against the publisher's template at 1519de0a, byte-identical to the one the text-only extraction's GGUF carries. 25 cases in text and tokens, and one the reference refuses (a system message mid-history). Accounted deviations, as for 2509: BOS is the tokenizer's, and the default system turn is omitted. v1.5's default carries no date; Runner omits it because it does not invent a persona the caller did not ask for, and that one claims images and audio.

- **Apertus: thinking is separated from the answer** (R2.15). With `enable_thinking` on, an Apertus model's whole deliberation came back as the answer, unmarked, and `reasoning_content` was empty. The model marks its deliberation with two control tokens, which decode to no text, and Runner finds a reasoning block by reading text. A chat turn now reads those two tokens by name: `<|inner_prefix|>` and `<|inner_suffix|>`, in both the v1.5 and the 2509 files. Measured on v1.5, which deliberates. The 2509 model opened no deliberation on any prompt tried, on Runner (9 prompts) or on the publisher's own stack (3), so its `reasoning_content` stays empty: the split is wired for it and has nothing to split.
  - **What a client sees:** `reasoning_content` on `/v1/chat/completions` (streamed and not), a reasoning item on `/v1/responses`, a thinking block on `/v1/messages`, and the answer alone as the content. `/v1/messages` refused `thinking: enabled` for this family before.
  - **Unchanged:** a turn without thinking; a raw `/v1/completions`, whose text is the tokens as they decode.
  - **A stop string now applies to the answer.** Before, the deliberation was part of the content, so a `stop` string that appeared inside it cut the turn there, mid-deliberation. The deliberation now runs to its close and the stop is matched in the answer, as on the families whose thinking is marked in text.
  - **Tool calls and structured output are untouched:** with a JSON schema, JSON mode or a tool constraint the family's turn is the constrained document from its first token, as before, with thinking on or off; the model does not deliberate there.

- **Apertus-8B-Instruct-2509 joins the roster on bartowski's Q8_0 and Q4_K_M** (R4.38): two rows, one per file, for swiss-ai's instruction model on the apertus architecture, which had no file on the support page. Against swiss-ai's BF16 checkpoint computed in float32 over 600 positions, the Q8_0 reads 0.0054 ± 0.0003 mean KLD and is the fidelity file; the Q4_K_M reads 0.0492 ± 0.0024 against a bar of 0.05, one of its two windows of 300 reads 0.0526, and a different conversion read 0.0507 in the golden pass. Both run on the CPU and, fully offloaded, on CUDA; on Metal the load runs on the CPU, because the xIELU activation is not on the Metal backend yet, and training is refused because it has no backward. The tokenizer matches the publisher's on every corpus string and vocabulary entry, and the conversation template on every expressible case. The record is [docs/compat-reports/apertus-8b-instruct-2509-2026-11-12](docs/compat-reports/apertus-8b-instruct-2509-2026-11-12/README.md).
  - **The family's placeholder row is gone.** Apertus was listed as an admitted architecture with no file and no cell; the two file rows replace that row, so the family's line on the support page reads from real files.

- **A tier-smoke pin names the commit its binary was built from** (R6.67). `scripts/tier-smoke.py` recorded the commit as the HEAD of the checkout running the script, so a binary built in another worktree or on a branch pinned a commit that never produced its bytes.
  - **The binary states its source.** A build stamps git's HEAD, and whether tracked files differed from it, into the binary, and `GET /v1/runner/provenance` reports them as `build.commit` and `build.dirty` beside the binary's sha256 and compiler. A build outside a git tree reports `null`. Only one object is rebuilt when the commit changes.
  - **The pin and the record take it from there, and they never borrow the checkout's HEAD.**
    - **Writing a pin needs that commit.** `--pin` refuses when the binary reports none (an older binary, one built outside git) or was built with tracked changes, says to build from a clean checkout of a commit, and writes nothing. `--allow-unknown-commit` writes the pin with commit `null` and the reason in the pin.
    - **Checking or recording against a pin does not.** It records commit `null` and warns once.

- **CUDA: two kernel width rules from the audit of fixed-step kernels** (R6.68.111, R6.68.112).
  - **CUDA keeps a model whose attention heads are an odd width on the CPU** (R6.68.112): the device attention kernels that every other kernel hands an odd width to walk a head in pairs, so on the first prompt the prefill kernel failed with a misaligned address and the backend was released mid-request (the CPU finished it). Such a model now loads on the CPU and says why, and `--info fit` reports the CPU for it instead of a device split. No published model has an odd head width; the CPU and Metal take one.
  - **The CUDA NVFP4 tensor-core GEMM no longer reads past a row** (R6.68.111): for a row whose width is not a whole number of the GEMM's 128-wide steps (64 times an odd number), its stager read the next row's first block, and 36 bytes past the end of the tensor on the last row. The block was always multiplied by zeros, so no value changed. The stagers now share one tail rule that writes zeros there without reading.

- **Every record and JSON line the CLI writes is JSON even when the forward is not finite** (R6.68.115). The server's responses took this rule in R6.68.114; this is the rest of what the runner writes as JSON or JSON lines. JSON has no NaN and no infinity, and the C library writes them as `nan` and `inf`, which no parser reads. A number that can be non-finite is now written as `null`, and where a non-finite number means the run failed, the command says so and exits 1. Each fix has a test on a model whose embedding table is NaN in every row.
  - **`--score` writes `null` for a log-probability that is not finite, and exits 1.** The log-probabilities, the `--score-top` lists and the totals read `nan` and the command exited 0. A suppressed token scores -1e30, which is finite, so a non-finite log-probability means the forward failed, and `--score` already exits 1 when a forward fails outright. The perplexity alone can still overflow on a finite run (a mean past 709 nats): it is written as `null` and the exit stays 0.
  - **`--train-logprobs` writes `null` for a log-probability that is not finite.** A step's line is written as the forward makes it, before the step's own check stops a non-finite run, so the one line written read `nan`. The run still stops there with exit 1.
  - **A `--train-eval` loss that is not finite stops the run with exit 1**, as a training step's loss already does. The line read `nan`, and a NaN loss at step 0 also became the kill rule's baseline, so `--kill-at` killed the run on a number that is no loss. The error names the set and the step, and the step's eval line is not written.
  - **A `--train-dpo` step that is not finite stops the run with exit 1 before the optimizer touches the adapter.** DPO checked nothing: on a non-finite forward every step's line read `nan`, the step applied the NaN gradient, and the run exited 0 with an adapter and a record. It now takes the `--train` step's rule, over the loss, the margin, the four log-probabilities and the gradient.
  - **`--quant-error` writes `null` for a tensor whose quantization error is not finite.** A tensor holding a NaN (or a value whose square overflows) has no error to cost, and the report read `nan` there, so a planner could not read any of it. The exit is unchanged.
  - **The activation trace and the MoE route trace write `null` for a number that is not finite.** `RUNNER_ACT_TRACE` is how a non-finite forward is localized, and its rows' norms and values read `nan` exactly there, so the file did not parse; a row's hash is still its bytes. `RUNNER_MOE_TRACE`'s gates and norms read `nan` for a NaN router; the picked experts, which `RUNNER_MOE_REPLAY` reads, are unchanged.
  - **The `--merge-lora` record writes `null` for a kept-delta fraction that is not finite.** A weight that is not finite, in the adapter or in the base under it, made the record's `delta_retained` read `nan`. Such an adapter is now refused when it is read, and such a base fails the merge's floor (both below); under `--merge-allow-erased` the record says `null`.
  - **A LoRA adapter that holds a NaN or an infinity is refused where it is read, naming the tensor and the value** (R6.68.116). Serving with `--lora`, a per-request `--adapter`, a training run that continues or resumes from an adapter, and `--merge-lora` all read through one of two loaders, and neither looked at the values: served, such an adapter made every logit NaN; merged, it wrote a model whose weights are not numbers. Both loaders now check every value of A and B as they convert them, about 5 ms over the 16.5 million values of a 66 MB Qwen3-4B adapter.
  - **`--merge-lora` fails closed when the kept fraction of the delta is not a number** (R6.68.116). The erased-delta floor and the record's survival block were guarded by comparisons a NaN fails, so with a weight of the base that is not finite the merge skipped the floor, dropped the block and wrote the file. A fraction that is not a number is now not above the floor: the merge is refused and nothing is written. `--merge-allow-erased` still writes it, with the block and `null`.
  - **The scripts that write numbers they computed refuse to write NaN or an infinity** (R6.68.116). Python's `json` writes them as `NaN` and `Infinity`, which other JSON readers refuse, so `scripts/clef-gold.py` would have written a gold file no one else could read from a non-finite torch reference. The 44 scripts that compute what they write (with torch, numpy, `math`, `statistics` or plain arithmetic) now dump with `allow_nan=False` and fail instead. Two wrote an infinity on purpose: `act-trace-diff.py`'s distance between rows of different widths and `weight-io-bench.py`'s ratio over a pass too fast to time are `null` now. `act-trace-diff.py` also reads the runner's `null` trace values: two rows match only if their nulls stand at the same places.
  - **`confirm_below` answers that a turn needs confirmation when its forward was not finite** (R6.68.118). The decision capture compared NaNs and recorded no decision, and the answer only looked at recorded decisions, so a turn no number supported read `needs_confirmation: false`. A probed row that is not finite now makes the answer true. No field is added: the signal is `needs_confirmation: true` with a `null` minimum.
  - **`/v1/decide` and `/v1/rerank` refuse with 500 when a score is not finite, and `--decide` exits 1 with the same words** (R6.68.119). Their scores are their answer, and from a non-finite forward they answered 200 anyway: decide an `argmax` (its comparison skipped NaN, so option 0 or a finite one won) beside `null` probabilities, rerank an order sorted by a comparator a NaN broke. They now refuse, as embeddings do. Rerank's sort also orders a NaN last, consistently, whatever reaches it. `/v1/systemone` already refused (500, "the decision head produced a non-finite score"); a test now holds it.

- **Muse tool calls: an answer that ended is delivered however the stream was chunked** (R6.68.117). On a streamed Muse turn with native tool calls, an answer to the user ends at `<|eot|>`. When that end and the opening of a later `<atem:invoke` reached the stream reader in one piece, the reader waited for the invoke to close: the answer was never delivered and the turn ended with a tool-call error. The same bytes arriving one at a time gave the answer. The reader now follows the order of the turn.
  - **Reach:** it needs bytes after the answer's `<|eot|>`, which a model that stops at its end token does not write. The tool-stream fuzz target found it; its input is now in the corpus.

- **Embeddings run on the GPU for a fully offloaded model; two batch-only Apple GPU defects are fixed; a response stays valid JSON when a forward goes non-finite** (R6.68.107, R6.68.109, R6.68.113, R6.68.114).
  - **`/v1/embeddings` uses the GPU when the whole model is on it** (R6.68.107). With every layer offloaded, an embeddings request ran its forward on the CPU: the readout set the device aside because a full offload keeps the hidden rows on the device. The device hands those rows back when asked, and the Clef readout has asked since it was admitted; embeddings now ask the same way through the same code. A partial offload already used the device.
    - **Measured** (Qwen3-4B Q4_K_M on one Apple GPU, 16 inputs, 2,882 tokens, two interleaved rounds each): 84.7 and 85.3 s before, 32.3 and 32.2 s after, so 34 to 89 tokens a second.
    - **Measured on CUDA** (Mellum Q4_K_M, 64 inputs, 1,142 tokens, four interleaved rounds each with a fresh server, medians of 12 timed batches): 11.84 s before, 3.59 s after, so 96 to 319 tokens a second; the server's CPU time for a batch went from about 230 CPU-seconds to 3.5.
    - **The vectors** are the device's forward instead of the CPU's, so they change in their last digits: the lowest cosine between before and after over those 16 inputs is 0.999998. A server started with `--gpu off` gives the same vectors as before.
    - **Unchanged:** a model whose backend cannot hand rows back (the wide hyper-connection residual) still embeds on the CPU.
  - **Apple GPU: a prompt batch through an expert layer whose width is not a multiple of 32 gives the right answer** (R6.68.109). The grouped expert kernel for prompt batches steps its input 32 wide and has no step for a remainder. The dense kernel of the same shape is only used when the input is a whole number of steps; the expert kernel had no such rule, so on a sparse-MoE file whose expert layer is 16 wide it read past each weight row and each activation row. On the 512-wide recurrent MoE test fixture the log-probabilities after a batched prompt were off the CPU's by up to 0.71; they are now within 2.4e-4. Both kernels share one rule, and a tensor that fails it takes the per-slot kernel.
    - **Who was exposed:** a file whose expert tensors are F32 or F16 with an expert width or model width that is not a multiple of 32. A quantized tensor cannot have such a width (its rows are whole blocks of 32 or 256), so quantized expert tensors were never affected. One token at a time never took this kernel, which is why `--score` agreed.
  - **Apple GPU, `RUNNER_METAL_MOE_MM=half`: when the half-staged expert kernel cannot get its scale buffer, those experts take the f32-staged kernel** (R6.68.113). The half-staged kernel scales each activation row by its largest magnitude, which a pass writes into a scratch buffer first. The kernel was chosen before that buffer was made, and when the buffer could not be made the kernel ran anyway and read a placeholder in its place, so every value of the batch's rows came back NaN. The kernel and its buffer are now one choice: a tensor whose buffer cannot be made takes the f32-staged kernel, and the log says so once. Found in the audit of the device kernels that step in fixed blocks (R6.68.110).
    - **Who was exposed:** only the opt-in `half` mode, and only when that scratch allocation failed. The default mode never reads the buffer.
  - **A response is valid JSON even when the model's numbers are not finite** (R6.68.114). JSON has no NaN and no infinity. When a forward went non-finite (an overflowed activation, a failed GPU kernel, a NaN in the weights), every endpoint that returns a number the model computed wrote it as a bare `nan` or `inf`, and the whole response failed to parse with status 200.
    - **`/v1/embeddings`** now answers 500 with a message for such an input, in both encodings. Before, the float list was unparseable and `base64` handed over NaN bytes with nothing to tell them by.
    - **Log-probabilities** (completions, chat, echo and prompt log-probabilities), **`/v1/rerank`** scores, **`/v1/decide`** log-probabilities and probabilities, **`/v1/systemone`** logits and the decision telemetry write `null` for a number that is not finite. A decision whose confidence is not a number is reported as needing confirmation.
    - **Unchanged:** every finite number is written exactly as before.

- **The full read of the engine, second part: 47 fixes** (R6.68.23; R6.68.48 to R6.68.105). Every source file is being read function by function for defects and for time left on the table. Each fix below has a test that fails without it, except the snapshot size (R6.68.57), which needs a file over 2 GiB on Windows to show.
  - **Serving**
    - **A constrained document cut by the token limit stays valid when a stop string is set** (R6.68.49). With `stop` and a JSON schema or JSON mode together, the server holds back text that may still begin a stop string. When the token limit ended the turn there, the tail the server writes to close the document went out ahead of the held text: `{"city":"","units":"celsius"}aaa`, buffered and streamed. The held text goes out first now, then the tail.
      - **`runner_telemetry.closure` counts the delivered document.** After a matched stop its `model_bytes` included the bytes the stop had dropped, so the two counts added up to more than the content. They are read from the content the client received.
    - **`draft_hints`: many short hints no longer write past the hint buffer** (R6.68.51). The buffer for a request's hints was sized at one token per byte. A tokenizer can produce more: one that prefixes a token to every text, or spells a space as several. With enough short hints the later ones were cut short or dropped, so they were never drafted from, and the separator after each was written past the end of the buffer: a heap overflow one request could cause on a server with the lookup draft on. Each hint is tokenized to its true length now.
    - **A session request keeps the stop tokens exempt from the repeat penalty** (R6.68.52). The model's own stop tokens are never punished by `repeat_penalty`; a penalised terminator can leave a model unable to end its turn. `POST /v1/runner/sessions` and a session resume reset the sampler in a way that dropped that exemption, for the session itself and for every later request on the same slot until the model was reloaded. The sampler a request starts from carries the exemption now, on every route.
    - **A batch of embeddings, rerank documents or decide questions no longer blocks the other slots** (R6.68.54). On a GPU server with several slots, work on the device takes turns, and a long prompt gives its turn up between chunks so a short request is not stuck behind it. Embeddings, rerank, decide, pinned contexts and a session's prompt did not: they kept the turn from their first item to their last. A one-token chat request sent just after a 48-input embeddings request waited for all 48 (measured on Metal: 186 s). They give the turn up between inputs, documents and questions now, and between chunks of a long prompt. Results do not change.
    - **A slot that loses the GPU no longer takes the server down** (R6.68.93). With `--parallel`, slots that decode at the same moment share one device step. That step was set up while every slot had its GPU; when one slot's forward later failed at runtime and it fell back to the CPU, the shared step still took its tokens and read its device cache through a backend that was gone, and the server died (SIGSEGV; reproduced with two concurrent requests and an injected failure on the Apple GPU). A slot without its backend now takes no part in a shared device step: that step runs sequentially, the lost slot on the CPU and the others on the GPU, and both requests return the same text as before.
    - **A large receipts directory no longer slows the server's start or each answer** (R6.68.100). The receipts in a `--receipts` directory were put in order with a sort whose cost grows with the square of their number, and a directory hands its names back in no order. With the default of keeping every receipt that ran at every start (80,000 receipts: about a second; a million: minutes), and under `--record-keep N` at every answer. It is an ordinary sort now: `runner --lineage` over 80,000 receipts went from 3.0 to 2.2 seconds, the rest being the reading of the files.
  - **Tool calls**
    - **A tool call is read whole when one of its values contains the tag that closes the call** (R6.68.71). A string argument may hold any text, including the model's own call syntax: a file about tool calls, a chat template, a test. Three readers ended the call at the first closing tag they saw, inside the value, and dropped the call with an `envelope_error`:
      - Muse: a value containing `</atem:invoke>`.
      - Gemma 4: a value containing `<tool_call|>`.
      - Qwen3-Coder, Qwen 3.8, Granite 4.2 and Ornith, on a streamed `auto` turn: a value containing `</tool_call>` or `</function>`. The same turn without streaming was read correctly.

      Each reader now finds the end of a call by its structure, as the Qwen JSON and Nemotron readers already did.
    - **A tool whose name or parameter key the model's call syntax cannot spell no longer fails on every call** (R6.68.72). The native tool grammars wrote names and keys into the call exactly as declared, and the reader for the same family then could not read the call back, so each call of that tool ended in `envelope_error`:
      - Gemma 4 writes keys bare, ended by a colon: a parameter such as `og:image` or `xsi:type` broke the tool. Such a tool set now takes the generic envelope, as an untyped parameter already does.
      - Qwen, Hermes, Granite 4, Nemotron and Mistral write the name inside a JSON string: a name with a quote or a backslash is written escaped now.
      - The Mistral `[SYSTEM_PROMPT]` form writes the name bare before `[ARGS]`: a name with a space is read back now; one with `[` or a line break is refused with a message that names it.
      - Muse (a quote in a name or key) and Qwen3-Coder (`>` in a key) take the generic envelope for such a tool.
    - **A tool call's `arguments` is always a JSON document** (R6.68.73). On a turn the engine does not constrain (no `tools` in the request, or `tool_choice: "none"`), two readers passed on what the model wrote: Gemma 4's copied `.5`, `+1` or `1-2` as a number, and the generic `<|tool_call>call:NAME{...}` reader any text with matching braces, such as `{'a': 1}`. A client parsing `arguments` then failed. Such a block is no longer reported as a call; its text stays in the reply.
    - **A tool call that writes the same parameter twice is refused for every XML call format** (R6.68.74). MiniCPM5, Spark and K2-Horizon calls with a repeated parameter were passed to the client with the key twice in `arguments` when the tool declared no parameters; Qwen3-Coder's format already refused them. The three formats now share one reader for a call's members, so they agree on what a value is and what makes a call invalid.
    - **A stray block-closing tag streams the same text however it arrives** (R6.68.103). On the K2-Horizon tool protocol the reader drops the newline that stands before a `<ifm|tool_calls>` or `</ifm|tool_calls>` token. While streaming it held back a newline only in front of the opening token, so when a model wrote the closing token on its own (no block open), a stream that arrived a byte at a time kept that newline in the content and one that arrived in larger pieces dropped it. Both now drop it. Found by the daily fuzz run on main.
  - **Chat prompts**
    - **A Mistral model loaded after another one answers under its own default system prompt** (R6.68.75). The `[SYSTEM_PROMPT]` family's default system text is read from the model file's template and kept for later requests. It was kept by the template's memory address, so after a model swap a second model of that family could be given the first one's text when its template landed at the same address. It is kept by the text itself now.
    - **A chat prompt is never sent with a piece missing when memory runs out while it is built** (R6.68.76). If an allocation failed while a prompt was rendered, several model families still returned a prompt: without the tool declarations (ChatML and Qwen, Qwen 3.8, Granite 4.2, Nemotron 3.5, Gemma 4, Apertus, Harmony), without a user turn's text (Llama 2), or with the system text changed (Muse). The model then answered a different question under a normal reply. Every family now reports the failure and the request ends with an error.
  - **Structured output**
    - **A schema's `$ref`s are inlined up to 64 MiB, then refused** (R6.68.68). A schema that shares definitions through `$ref` is expanded before it is compiled, and a definition is copied at each reference. Only the number of schema nodes was limited, so a large `enum` referenced many times could turn a request of under 2 MB into tens of gigabytes of server memory. The expanded schema is limited in bytes now (twice the largest request body); past it the request gets the existing `400`, "schema too large once its $refs are inlined".
    - **A request with a large `enum` compiles its schema faster** (R6.68.69). A schema is compiled for every request that carries one. Sorting an enum's literals took time that grew with the square of their number: 12 to 17 ms for an enum of 4,096 short literals, now about 1 ms; 200 to 500 ms when the literals share a long prefix, now about 19 ms. The compiled schema is the same.
    - **String patterns with very large repeat counts compile at once, or are refused** (R6.68.70). Two things in the schema compiler depended on the size of a count in a `pattern`:
      - Patterns joined by `allOf` were checked against each other byte by byte, so a count of 200,000,000 took 28 seconds to compile, from a request of a hundred bytes. They are checked where they can differ now, and compile in well under a second.
      - A pattern whose counts add up past 2,147,483,647 overflowed the length kept for it. It is refused: "pattern describes a string longer than 2147483647 bytes".
    - **JSON mode closes an array it had only opened as `[]`** (R6.68.101). When the token budget ran out right after a `[`, the closer that completes the document wrote `null` before the `]`, so `{"a":[` came back as `{"a":[null]}`: an element the model never produced, where the empty array was a legal ending. It now closes the array empty. After a comma or a colon a value is still owed, and `null` is still what completes it.
  - **Records, keys and bundles**
    - **`runner --lineage` no longer hangs on a record that names a device file** (R6.68.56). The lineage walk checks a chain of records by hashing the files they name, and those records can come with a download. A record that named `/dev/zero` was hashed with no end, so the check never answered. The walk reads regular files only; anything else counts as a file that is not there.
    - **Windows: a KV snapshot over 2 GiB records its real size** (R6.68.57). The manifest and the response of `POST /v1/runner/contexts/{id}/snapshot` carry `kv_bytes`, the size of the snapshot's file. On Windows it was read through a 32-bit file position, so it came out wrong for a file past 2 GiB (a long context on a large model). The snapshot itself was written whole and still loads: a load checks the file's SHA-256, not this number.
    - **A signature verdict is valid JSON for any file name** (R6.68.58). Transcripts, receipts and `GET /v1/runner/provenance` record the verdict of a model or adapter signature check, with the name the signed manifest gives the file. A name holding a quote (a split model named `m"q-00001-of-00002.gguf`) was written as it stood, and the record did not parse. The name is escaped now.
    - **A new signing key is readable by its owner alone** (R6.68.59). `runner attest keygen` wrote the key file, which holds the private seed, with the default permissions: readable by every user on the machine (`0644`). It is created `0600` now, also when it replaces an older file. Check the permissions of keys made before this release (`chmod 600`). POSIX permission bits; on Windows the file keeps the folder's access rules.
    - **A signed record with anything after its signature no longer verifies** (R6.68.61). A record's signature covers every byte before it. Fields added after the signature are covered by nothing, yet `runner --verify` read them as part of the record and answered `VERIFIED`, signed: a `model_signature` verdict or a `tool_calls` list could be appended to a signed receipt that did not have one. `--check-record` already refused such a file. The rule is part of the signature check itself now, so `--verify`, bundles, evidence packs and KV snapshot manifests refuse it too. Fields the signed part already holds could not be replaced this way.
    - **`--check-bundle` and `--check-pack` no longer hang on a listed file that is a device** (R6.68.63). A bundle or an evidence pack is a folder someone else hands over, and its check hashes the files the manifest lists. A listed name that was a link to `/dev/zero` was hashed with no end. Both checks read regular files only, as the lineage walk does; anything else counts as a missing file.
      - **`--export-pack` writes receipt file names as JSON strings.** A receipt file whose name held a quote made a `pack.json` that did not parse.
    - **A model or adapter that must be signed is checked before it is read** (R6.68.79). With `--require-signed-model`, `--model-sig` or `--model-pubkey`, the signature was checked after the file had been parsed and loaded, so a file with no signature, or changed after signing, still went through the loader before it was refused. The check now comes first: for a one-shot run, for a model loaded on demand by a server, for each extra `--parallel` slot, and for `--lora`. Adapters named with `--adapter` were already checked first.
    - **`--transcript` with a previous record that cannot be read refuses before the run, and no longer crashes** (R6.68.82). The previous record (`--transcript-prev`, `--record-prev`) was read after the model had loaded and the reply had been generated. If it was missing or was not a record, the run then crashed instead of exiting with its error. It is read first now, and the run refuses there with nothing generated. A `--record-prev` on a run that writes, signs or verifies no record is refused too, where it was ignored.
    - **A signature or key whose DER length is not in its shortest form is refused** (R6.68.97). DER allows one spelling of a length. The reader refused a one-byte long form under 0x80 but took a two-byte form under 0x100 (`30 82 00 8B` for `30 81 8B`), so one ECDSA signature had a second byte string that parsed and verified. Keys and signatures written by Runner and by OpenSSL always use the shortest form and are unaffected.
    - **An Ed25519 signature has one form** (R6.68.98). Verification did not check that the signature's scalar S is below the group order, which RFC 8032 requires. Adding the order to S gives a different 64 bytes that verify for the same message and key, so anyone holding a signed record could write a second valid signature for it without the key. Such a signature is now refused. Signatures Runner writes always have S reduced and are unaffected; the same message could never be forged for another key or text.
    - **`runner derive --target-size` changes only the tensors its plan ranked** (R6.68.99). The plan it writes names each demoted tensor, and a plan's rules match by substring, the first match winning. A rule for `token_embd.weight` therefore also took gemma-4's `per_layer_token_embd.weight`, and a rule for `output.weight` takes every `attn_output.weight`: tensors the ranking had left at the high type were written at the low one, the file came out smaller than planned, and the record and card named fewer demoted tensors than the file had (13 against 14 on the E-series test model). Every tensor that a demoted tensor's rule would catch by its name now gets its own rule first. `scripts/type-plan-by-sensitivity.py` writes the same plans and has the same fix. A plan with no such names is byte for byte what it was.
  - **Hub transfers**
    - **`-hf` no longer shows `HF_TOKEN` in the process list** (R6.68.65). For a gated repository the token was passed to `curl` as an argument, and a program's arguments can be read by every user of the machine (`ps`) for as long as a download runs. The token goes to `curl` on its standard input now (`-K -`). A wrapper named by `RUNNER_CURL` that looked for `-H "Authorization: ..."` in its arguments has to read the config line from standard input instead. If a token was used on a shared machine before this release, consider replacing it.
    - **`-hf` and `--pull` fetch a repository file whose name has a space, `#`, `?` or `%`** (R6.68.66). The file name went into the download URL as it stood: `curl` refused a URL with a space, and `#` or `?` cut the path short, so such a file could not be fetched. The name is percent-encoded in the URL now.
  - **Command line and instruments**
    - **`--detect-watermark` on a text scores all of it** (R6.68.77). The text was tokenized into a buffer sized from its byte count, and a text can tokenize to more tokens than it has bytes (many short words, with a byte-fallback vocabulary). Such a text was cut at the buffer's end and only its beginning was scored, with nothing said about the rest. The buffer now grows to the text.
    - **A third-party NVFP4 layout is refused when its tensor is the last in the file too** (R6.68.48). At least one quantizer writes ggml type 40 with 40 bytes per 64 elements where ggml's `block_nvfp4` is 36; decoded as ggml's layout the model answers NaN with no error, so the loader measures each NVFP4 tensor's span and refuses the variant by name. For the last tensor in the file that span was computed from an address and a file offset and came out 0, so a file whose only (or last) NVFP4 tensor used the variant loaded as garbage. The span to the end of the file is measured in the same units now.
    - **Two fidelity evaluations can run at once, and a comparison scores only the servers it started** (R6.68.60). `scripts/eval-record.py fidelity` and the `kld-compare` scripts start a server for each file. They used two fixed ports and took whatever answered there for their own, so a second evaluation at the same time failed, or scored the first one's servers and recorded those numbers under its own file. Each evaluation takes ports of its own now, and a comparison refuses a port that another process is serving.
    - **`runner train`: an eval label with a quote or a backslash no longer breaks the run's JSON** (R6.68.80). A `--train-eval LABEL=...` label was written between quotes as given into each eval line, the kill line and the `.killed.json` / `.kill-check.json` record, so such a label made them lines no reader parses. It is written as a JSON string now.
    - **`--decide` and `--systemone` name the model by its file name on Windows too** (R6.68.81). The command-line modes took the name after the last `/`, so on Windows, where a path's separator is a backslash, each answer's `model` field was the whole path. It is the file name, as the server writes it.
    - **`--prune-experts`: a plan that names one layer twice is refused** (R6.68.83). `"layer_1"` and `"layer_01"` are the same layer under two spellings. The plan kept both, used the first and dropped the second without a word, so the file was pruned by half of what the plan said. Such a plan is now refused, naming the layer. A plan path that is not a regular file is refused by name for `--prune-experts` and `--type-plan`.
    - **A requant never grows a tensor, also when the type asked for does not fit its rows** (R6.68.84). A row a 256-wide block cannot describe (a width of 352, 896 or 5,760) is written in the 32-wide relative of the type asked for, Q5_0 for `q4_k`. The whole-file rule checked "is this tensor already small enough" against the type asked for instead of the one it would write:
      - A Q4_0 file asked for `--quant q4_k` had such tensors requantized from their Q4_0 values into Q5_0: 22% larger, and less exact than before.
      - A file that already carried the fallback type was reported as written in it again.
      - The log could name a type as written for a tensor that kept its own.
      - With `--type-plan`, `RUNNER_FORCE_REQUANT=1` had no effect, though the refusal tells the reader to set it.

      The whole-file target and a plan's rule now share one decision.
    - **`--quant-error` writes valid JSON for any tensor name** (R6.68.85). A tensor's name went into its line of the report unescaped, so a name holding a quote made that line unreadable to the planner.
    - **A directory given where a file is expected is refused on every filesystem** (R6.68.105). On Linux a directory can be opened like a file, and on XFS a small one then reports a length of zero, so a directory passed as `--prune-experts` or `--type-plan` (or to any other whole-file read) was taken for an empty file: "is not a JSON object" where the path could not be read at all. On other filesystems the read failed as it should, which is why only one test host saw it. The shared reader now refuses a directory before it asks for a size.
  - **Apple GPU**
    - **A layer constant the Apple GPU backend cannot allocate is refused at load** (R6.68.86). The backend copies each layer's small constants (norm weights, biases) into device buffers and then checked a separate list of them; the gemma-4 E-series per-layer post-norm was copied but missing from that list, so a failed allocation there was not noticed and the first forward bound an empty buffer. Each constant is now checked where it is made, and the load falls back to the CPU with a message, as it does for every other allocation.
    - **A prompt batch on a MoE model sends fewer dispatches to the Apple GPU, for the same bytes** (R6.68.87). The router of a sparse-MoE layer and the shared branch of a gemma-4 MoE layer must give each prompt token the bits a decode step gives it, and they were encoded once per token: a 512-token batch sent 512 router matvecs per layer. They now go out as one dispatch per layer carrying every column, through the column walk the decode microbatch already uses. Output is byte-identical (the Metal MoE gates compare it with the CPU). Measured on the test fixtures only, on one Apple GPU: prompt speed up 3% to 18% on the plain MoE fixtures and 79% to 103% on the gemma-4 MoE ones, about 2.5 microseconds saved per token per layer per matvec. No real MoE model fits the machine this was written on, so there is no real-model figure yet; the saving per token is fixed, so it will be a smaller share there.
    - **A long context no longer makes the Apple GPU's prompt batch reserve gigabytes of attention scratch** (R6.68.88). The scores a prompt batch's attention holds were sized as batch x heads x context: 2 GiB for a 512-token batch of a 32-head model at a 32K context and 8 GiB at 128K, which could exceed what one buffer may hold and drop the model to the CPU at its first prompt, and none of it was counted when the load decided how much fits. The scratch is now bounded at 512 MiB (never fewer than 64 columns): a batch that needs more runs its attention in several dispatches per layer, each token at its own position, for the same bytes, and the load's budget counts what a long context takes. Contexts up to 8K at 32 heads are unchanged: one dispatch, same memory, same speed (measured on a 4B model: 43 tokens/s before and after, identical output). Past that, each extra dispatch per layer costs about 1% of prompt speed.
    - **A GPU failure in the middle of a run no longer corrupts a recurrent model's state on the Apple GPU** (R6.68.90). When a forward fails on the device, Runner falls back to the CPU and runs it again. For the models with recurrent layers (the qwen35 family, the Mamba-2 hybrids) the device updates the recurrent state in the host's own memory, so by then the state already held those tokens, the CPU added them a second time, and every later token was computed from a wrong state: log-probabilities off by 4e-4 to 4e-3 on the test fixtures, with nothing in the output saying so. The model now keeps the tokens it was fed, and after such a failure restarts the state and rebuilds it from them on the CPU before it continues; the scores then match a CPU-only run (exactly when the rebuild is one batch, within 5e-7 past that). Dense models were not affected: their attention rows are simply rewritten.
    - **The rebuild after a GPU failure uses the right conversation** (R6.68.104). The fix above (R6.68.90) rebuilds a recurrent model's state from the tokens the model was fed. When a request continued a prefix taken from the prefix cache, the state was that prefix's but the record still held the previous request's tokens, so a failure right after it rebuilt the state from the wrong text (log-probabilities 8e-4 off on the test model). Whatever installs a saved state now also says which tokens it belongs to. Found by a second reader of the first fix, before either was released.
  - **Tray**
    - **"View log" in the tray no longer leaves a defunct process behind** (R6.68.95). On macOS and Linux the tray starts the system's opener for the log file and never waited for it, so every click left one defunct child for as long as the tray ran. They are now reaped at the next menu refresh.
    - **The Windows tray icon comes back when Explorer restarts** (R6.68.96). When Explorer restarts (a crash, an update, the shell starting after the tray at sign-in) every notification icon is gone and Windows asks each program to add its own again. The tray never received that request, because it listened on a kind of window Windows does not send it to, so its icon stayed gone while the tray kept running, and starting a new one was refused as a second tray. It now uses an ordinary hidden window and puts its icon back. Held by a test on the Windows build that sends the request and counts the answer; not yet tried against a real Explorer restart on a desktop.
    - **A very long model name no longer ends the macOS tray** (R6.68.102). A menu row holds 511 bytes and a served model's name 127; a longer name is cut to fit, and a cut that fell inside a multi-byte character (an accented or CJK name) left bytes that are not text. macOS then had no title for the menu item and raised an error the tray did not survive, at the moment its menu was opened. Rows now end on a whole character, on every platform, and the macOS menu shows a label it cannot read as text rather than failing.

- **The whole python suite runs under ASan and UBSan with leak detection, daily** (R6.68.21): a new scheduled `suite-sanitized` job runs `make test-python-sanitized`, which runs the root pytest list and the client tests against `make debug`, and `tests/sanitizer_reports.py` fails the run on any sanitizer report a child process wrote, each one named after the test that started it. The sanitized lanes before it ran the conformance suite with leak detection off and a named set of CLI files, so a leak in any other path (the muse think path, R6.68.20) passed with every test green. `test_request_disconnect.py` and `test_cli_files.py` scale their absolute waits by `RUNNER_TEST_TIMEOUT_SCALE` for the -O0 build, and `test_transcript.py`'s int8 route case skips on a build without the int8 kernel, which the runner reports itself. `docs/ci.md` records the Linux CUDA gate host's python and model file.

- **`--info fit` gives the load's answer, and a load refuses what the engine would route wrongly** (R6.68.45; R6.68.50, R6.68.53, R6.68.55, R6.68.62, R6.68.64, R6.68.67, R6.68.78, R6.68.89, R6.68.91, R6.68.92). Found in the full read of the engine (R6.68.23), where the fit and the loader each stated some of the same rules and the fit predicted a load that did not happen wherever the two differed. Each fix below has a test that fails without it; for the two CUDA fixes a device run is the proof.
  - **The fit refuses a file the loader's tensor binders refuse** (R6.68.45). The fit's plan ran the loader's metadata binders and stopped there, so a file missing a tensor the load requires was priced as fitting while its load refused it (11 fixtures in the tree, each without `blk.0.ssm_d`, said `FITS`). The plan now runs the tensor binders too, over the header and without reading a tensor: every tensor the load would find, type and shape-check. Such a file is answered `verdict WILL NOT LOAD` with the load's own reason and exit status 1. One refusal stays the load's alone: a Clef head whose `decision.scales` values are out of range, since a header carries no values.
  - **The fit refuses a file whose agent profile the loader refuses** (R6.68.50). The load checked a `gridcore.agent.*` profile before it made its plan, and the fit only asked the plan, so a file requiring an agent feature this build lacks, requiring MTP consumption, or carrying a malformed schema digest was priced while its load refused it. The check is now the plan's first step, so the fit answers such a file `verdict WILL NOT LOAD` with the load's own words.
  - **A split-expert MoE file is priced as sparse** (R6.68.53). Older Mixtral GGUFs store each expert as its own tensor. The fit looked for expert banks by name, found none in that layout, and priced the whole file as the hot set while the load counted only the routed share. The fit now asks the load's own function.
  - **Qwen3.8-Flash-Next's context is held to the 2048 tokens its admission covers** (R6.68.55). This build runs the model's sparse-attention indexer densely, which matches the reference only up to the indexer's top_k (2048). The check that caps the context ran before any context was set and never fired, so a load with no `-c` ran at 4096. A default context is now held to the cap with a line saying so, an explicit `-c` past it is refused with the load's words, and `--info fit` answers the same way. The fit also takes a missing `context_length` as 2048, as a load does.
  - **nemotron_h_moe refuses an expert grouping its router cannot run** (R6.68.62). Group counts that do not divide the experts, or no used group at all, were refused for every other MoE family and loaded for nemotron_h_moe, which then routed wrongly. One check now serves every family.
  - **A CUDA split that hands its last layer back sizes the prompt window from what the device holds** (R6.68.64). When every layer fits but the output head does not, the last layer runs on the host. The CUDA load kept that layer's bytes in its plan, so its prompt window was narrower than the budget allowed and than `--info fit` predicted.
  - **A router past the device's limits runs on the CPU and says why** (R6.68.67). A device bank routes at most 512 experts and 256 picks a token on CUDA (256 and 256 on Metal). A CUDA load of a model past either limit routed it among a subset of its experts without a word. It now runs on the CPU with the reason, `--cpu-moe` keeps such experts on the host, and `--info fit` gives the same words.
  - **On CUDA, `--info fit` places nemotron_h_moe's experts where the load does** (R6.68.78). A GPU load keeps a nemotron_h_moe file's experts and their routing on the host when the device's router kernel cannot run its router. The fit restated that rule from the header and missed a group limit, so for a group-limited softmax router it planned the experts on the device while the load kept them home. Both now ask one rule.
  - **A MoE file that picks more than 256 experts a token is refused instead of crashing** (R6.68.89). The host keeps a token's picked experts in arrays of 256, and the MoE loader held the picked count only to the expert count (up to 1024), so such a file loaded and its first token wrote past those arrays (a segmentation fault on the CPU). Every MoE family now refuses it with one message, and `--info fit` answers `verdict WILL NOT LOAD` with the same words. Real routers pick 1 to 16.
  - **Training a model whose attention heads are wider than 512 no longer crashes** (R6.68.91). The backward ran each head's q-norm and k-norm gradient through a 512-value buffer, and nothing holds a model's head width to 512. A wider head stopped the first step (a fortified build traps; another writes past the buffer). The buffer is now as wide as the layer's heads.
  - **Under `RUNNER_MOE_REPLAY`, a MoE adapter's training step takes its gradient through the experts its loss used** (R6.68.92). The replay pins each position's routing to a recorded trace. The step's forward was pinned, but the backward routed its recompute afresh, so wherever the two choices differed the gradient was taken through other experts than the loss. The forward and the backward now route through one call, pinned the same way. Without the replay nothing changes.

- **`--info fit` is the loader's own plan of the file** (R6.68.29, from the full sweep): the fit used to restate the loader's rules from the header (which layers own KV rows, how wide a head is, which blocks a NextN head adds), and every disagreement was a fit that predicted one load and saw another. It now runs the binders a load runs, over the header alone, so its geometry is the load's by construction: on all 213 fixtures in the tree the figures are unchanged, and one disagreement that no fixture had reached is gone (a gemma-4 header without `attention.key_length`, which the loader reads as 512-wide heads, was priced at `embedding_length / head_count`). A header the loader would refuse is now answered as such, `verdict WILL NOT LOAD` with the loader's reason and exit status 1, where the fit used to print a size for a file that would never load.

- **CUDA: a qwen35moe decode with its routed experts fused on the device replays a captured graph (R6.68.25a).**
  The rule that lets a recurrent family's one-token decode replay a CUDA graph left out every file with routed
  experts, so Qwen3.5 and Qwen3.6 MoE files (qwen35moe) decoded with plain launches on every token although nothing in
  them needs it: the expert banks hold no recurrent state, and fused device routing captures in every other family.
  The recurrent rule no longer excludes them; the MoE rules decide, as for any other family.
  - **Speed:** decode on the qwen35moe toys went from about 1,500 to 3,900 to 4,600 tok/s on an RTX 5070 (2.7x to
    3.1x; a two-layer toy, so launch overhead is most of it). No real qwen35moe file was measured here: the smallest
    rostered one (22.3 GB) does not fit a 12 GB card whole, and a split decode or experts on the host stay eager.
  - **Same bytes:** the decode runs the same launches in the same order. With the graph replaying, the toys' decode
    logits match the CPU's at 1e-5 of range with the same deviations as before, and greedy output is byte-identical
    to the eager decode's and to the previous build's on every test fixture that generates.
  - **Unchanged:** qwen35moe with the eager MoE path (`RUNNER_MOE_EAGER=1`) or experts on the host (`--cpu-moe`)
    stays eager by design, and under `RUNNER_REQUIRE_GRAPH_CAPTURE=1` now says "MoE routing read back to the host",
    as any MoE family does, instead of "a recurrent layer"; qwen4exp and the Mamba-2 hybrids with experts stay eager;
    `RUNNER_CUDA_GRAPH_RECUR=0` keeps every recurrent family eager.
  - **Gated:** `make test-cuda-qwen35moe` requires the fused and grouped arms to replay (no eager-by-design note but
    the pinned arm's route replay) and a CLI decode to report `graph=replay`.

- **CUDA: a decode with an adapter on the device replays its graph (R6.68.25b).** Loading an adapter with `--lora`
  on a CUDA model turned the decode graph off for the model's lifetime, so every adapted decode ran plain launches
  on every token, although the adapter's launches capture like any other and the adapter binds before the first
  forward. Binding or unbinding an adapter now drops any graph captured before it, and the next decode captures
  again with the adapter's launches.
  - **Speed:** Qwen3-4B Q4_K_M with a rank-8 adapter decodes at 128 tok/s instead of 90 on an RTX 5070 (1.42x;
    155 without the adapter, unchanged).
  - **Same bytes:** the decode runs the same launches in the same order. Greedy output with the adapter is
    byte-identical to the previous build's, to the eager decode's and to the CPU's, on prompts where the adapter
    changes the output.
  - **Gated:** `tests/test_lora.py` requires a decode with the adapter on the device to report `graph=replay` with no
    eager-by-design note under `RUNNER_REQUIRE_GRAPH_CAPTURE=1`.

- **Three fixes found while the architecture and fit rules moved into one place** (R6.68.28, R6.68.36, R6.68.40): a rope base clang builds read from the wrong key, adapters the merge refused although `--lora` served them, and KV formats `--info fit` offered that the load would not take.
  - **A file that names its sliding layers' rope base `rope.local.freq_base` now gets that base on every compiler** (R6.68.28, found in the full sweep): the loader read `rope.local.freq_base` and its fallback `rope.freq_base_swa` in one call, both spelled into the same key buffer, so the order a compiler evaluates arguments in chose the key. A clang build (every macOS build) read `rope.freq_base_swa` for both and roped the sliding layers at the wrong base whenever a file declared the two with different values; the gcc builds measured (gcc 13.3.0, x86_64) read the intended key. The reads are now made one at a time, and `test-rope-yarn` loads a fixture that declares 500 and 20000 and holds the sliding table to 500.
  - **`--merge-lora` folds a Gated DeltaNet adapter into a qwen35moe or qwen4exp model** (R6.68.36): serving applied an adapter's DeltaNet projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`) on every DeltaNet family, while the merge accepted them on qwen35 and clef only and refused the same adapter on Qwen3.5/3.6-35B-A3B-shaped and qwen4exp files ("targets a projection this model does not carry"). The merge now reads the family from the same table serving does. `test_lora_qwen35` merges such an adapter on both toy fixtures and holds the merged file to the adapter at load time and to the reference the fixture writer folds.
  - **`--info fit` offers a quantized KV cache only where the load takes it** (R6.68.40): the load takes `--kv q8` and `--kv k8v4` only when every attention head is a multiple of 32 values wide (`--kv fp4`: 16), because attention cuts each row at a head; the fit checked the whole row, so for a file with, say, 80-wide heads it priced a q8 cache, and could answer "FITS WITH --kv q8", for a load that then kept f16. One rule now decides both. No model on the roster changes its answer (every rostered file's heads are 64, 128, 256, 512 or 576 wide); `tests/test_fit_matches_load.py` holds the fit to what a load lays out on every fixture in the tree.

- **Two teardown leaks the sanitized whole-suite run found** (R6.68.38, R6.68.39). A constrained run (`--json`, `--json-schema`) ended with the engine's constraint document buffer still allocated: the CLI's cleanup and the server's slot teardown freed the token history by hand and not the buffer beside it. The engine has one release now, called by every teardown and by a re-init. And a server start refused before the serving loop (a taken port, a bad registry entry, an adapter that fails its check, about twenty refusals) returned with the loaded model and tokenizer allocated: one release now serves the normal shutdown and every refusal (18 leaks, 328,560 bytes, on a refused start of the test fixture before; none after). Both are exit-time only; they are fixed because a clean sanitized suite is how the next real leak gets seen.
  - A `--resume` refused after its image was read (the image was written under a schema and none is given, a `--suspend-after` past its budget, any later refusal up to the model load) returned with the image allocated: 3 leaks, 7,520 bytes on the fixture. The image belongs to the program's entry now and is freed after the run returns, on every path (R6.68.41).

- **A buffered tool-call response is read by the same reader as a streamed one** (R6.68.34). Each native tool protocol had two readers of a model's turn, the stream demultiplexer and a buffered mapper with its own search and whitespace rules; a turn could be answered differently by `stream: true` and `stream: false`. The buffered mapper for Gemma 4, Qwen (JSON and XML), the bare XML families, Nemotron, Mistral and Muse is now the demultiplexer run over the finished turn, and the four duplicate mappers are gone. A differential run over 100,000 mutated turns showed where the two had disagreed (text after a call dropped by one reader, a Gemma 4 call whose block never closed accepted by one, held whitespace kept by one) and shows none now. Closed with it: a streamed Muse answer whose end mark never arrived was dropped; a Gemma 4 call block that was closed but not a call, or left open by the model, ended the turn clean with the call missing and is now the reported fault it is in every other protocol.
  - The generic JSON envelope keeps its JSON parser for the finished document, and its stream now agrees with it on three things: the tool name is the decoded JSON string (an escaped spelling of `save` was announced as the escape sequence), a document that ends unresolved on a turn the model ended is the fault the buffered mapper reports (it ended as a clean, empty turn), and a call after a final branch in a parallel document is announced (the final branch's closing brace was read as the separator).
  - A Muse-family model asked for a required tool call with parallel calls allowed (the default) could not open its call after reasoning first: the first call's grammar took only the direct form, the form that follows a reasoning turn was refused, and the turn ran to `max_tokens` with no call and no content. The first call now opens in either form, as the single-call turn already did (R6.68.42; found by the real-model check of this change on a 14B Muse-family model).

- **CUDA: a shared-expert MoE that decodes through a graph keeps its graph, and the gates fail a decode graph that
  does not capture (R6.68.15).** The shared expert's input was copied with a synchronous `cuMemcpyDtoD` inside the
  decode forward, which a CUDA graph cannot record (`CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`). A model whose decode is
  captured (no recurrent layers, fused device routing, experts on the device) and that has a shared expert therefore
  lost its graph and ran plain launches on every token, with one stderr line as the only sign: DeepSeek-V2-Lite
  among the rostered files. The copy is now stream-ordered on the instance's stream, so the graph records it and an
  eager forward runs it in the same order as before; the bytes are the same.
  - **Gated:** `RUNNER_REQUIRE_GRAPH_CAPTURE=1`, which the CUDA gates set, makes a decode graph that fails to
    capture or launch end the run with the stage, the CUDA error and the synchronous call that broke it. A decode
    that is eager by design (experts on the host, eager MoE routing, a recurrent family not captured) says why and
    runs. Before the fix, `test-gpu-identity` on the two shared-expert fixtures exited 3 under it.
  - **Speed:** on the shared-expert fixtures decode went from about 2,300 to about 7,200 tok/s on an RTX 5070 (a
    two-layer toy, so launch overhead is most of it); real files are measured on a 24 GB card.
  - **Unchanged:** every model without a shared expert, and the shared-expert families whose decode is eager by
    design and never reached the capture (qwen35moe, nemotron_h_moe, qwen4exp): the same kernels in the same order.

- **Windows: a cancel by request id no longer leaves a handle in the server (R6.68.22).** Each cancel that waited for
  its request got a detached thread, and under winpthreads a detached thread keeps its Windows handle after it exits:
  a soak went from 265 to 456 handles over 400 cancels. One long-lived thread now answers every waiting cancel from a
  queue, and the server joins it when it stops.

- **The full sweep's first fixes** (R6.68.23, R6.68.26, R6.68.27): two speed gains, two leaks on every load and the command line's refusals.
  - **Sampled JSON-schema output is faster, with every pick unchanged** (R6.68.26, found in the full sweep): with top-k on, a grammar state that admits fewer than top-k tokens (an enum member, a key, punctuation; most tokens of a schema reply) made the sampler ask the whole vocabulary whether each token was valid, on every such token. The valid entries below the ranked prefix can only add the mass of what is unexamined, so the bound that already settles the top-k-off case settles this one: when the cutoff or the draw provably falls inside the prefix, the pick is made there. Qwen3-4B Q4_K_M on Metal, temperature 0.8, one seed, a six-field schema, three interleaved runs: 13.74 to 16.77 tokens/s (medians; 13.73 to 13.98 before, 16.59 to 17.07 after), against 17.73 unconstrained, and the six outputs are byte-identical. `test-sampler` holds every pick to a masked reference at the same draw and requires the top-ranked-valid case to be decided from 256 checks (60 of 60; it was 50,000 each).
  - **Two leaks on load, found by a run of the whole suite under AddressSanitizer and by a cross-check of every allocated field against every free** (R6.68.23): a recurrent block of a Qwen 3.5 file with experts bound its two norm buffers twice and leaked the first pair, and a LayerNorm family's norm biases were released nowhere. Both leaked on every load, so on every model swap in a server. CI's leak gate loads the fixtures that show them.
  - **The command line releases what it took when it refuses** (R6.68.23): a load refused for a missing or bad signature, a bad adapter signature or a tokenizer that would not initialise left the loaded model allocated; `runner derive` freed its strings and records only on success, with some twenty refusals before that; an `--adapter` argument refused for its form leaked its copy; and a subcommand that ended in its own help left its rewritten argument list. None of it grew a running process, but every such refusal failed a leak check, which is why none could gate the command line. `derive_main` has one exit that frees everything; the options and the rewritten argument list, which live for the process by design, stay reachable until it ends.
  - **The speculative drafts pick their token with the sampler's fast argmax** (R6.68.27): the MTP-head draft and the draft-model draft found each drafted token with a loop that carries the index, which no compiler vectorises; the greedy pick had already replaced it with two passes that are. `sample_argmax` is now the one helper for all three. Per call on an Apple M1: 682 to 30 microseconds on a 151,936-entry vocabulary, 901 to 35 on 262,144, so a round that drafts four tokens spends about three milliseconds less choosing them. The decode speed of a speculative run was not measured.

- **A text file's CRLF line ends are read as LF (R6.68.24).** `-f FILE` passed a file's carriage returns into the
  prompt, so a file saved with Windows line ends tokenized differently from its LF twin (two more tokens for a
  three-line prompt on the test model) and the same prompt file scored and generated differently depending on where
  it was written. A CR that immediately precedes an LF is now dropped as the file is read, on every platform; a lone
  CR is content and stays, and `-p` text keeps its escapes as written. `--detect-watermark FILE` and `runner train`'s
  DATA read their text the same way.
  - **Changed:** a plain-text training DATA file with CRLF line ends trains a different adapter from before: the one
    its LF twin trains. JSONL data is unchanged.

- **Functions inside functions are functions** (R6.68.35, from the full sweep): the sweep counted 30 macros defined inside a function body, C's form of a nested function: code that reads and writes the enclosing function's locals. Sixteen are now functions that take what they captured: the tokenizer's run classes and DeepSeek split test, the session reader's bounds check, JSON mode's two bounded writes (one function), the harmony grammar's channel step, the tool stream's run flush, the doctor report's finding, the tray menu's row, Metal's buffer allocation, its two NaN-scan points (one function) and its profiling cut, and the two NEON dequant planes (inlined; bit-exact against the scalar code). One that pastes a symbol name and captures nothing moved to file scope. No behaviour changes. The rest go with the refactors of the functions that hold them.

- **Three more untrusted inputs are fuzzed, and a fuzz campaign has a place to run** (R6.68.6). New libFuzzer harnesses: `tool_stream`, the demultiplexer that reads a model's own turn and separates prose from tool calls, with the property that a turn fed one byte at a time and fed in chunks gives the client the same reasoning, content, calls and result, on every template family; `record_check`, the verifier a third party runs on a signed receipt or record it did not write, in memory and through the file path; and `session_read`, the reader of a session image that travels between machines, with the SHA-256 trailer fixed up so the mutator reaches the parser behind it. Seeds are real: turns in each family's syntax, records signed with Ed25519 and ML-DSA-44, an image written by the fixture model. A new `fuzz soak` workflow runs every registered target for a chosen number of seconds each (by dispatch, and weekly), starting from the corpora the daily run has evolved; a red weekly soak opens the same issue a red daily run does. Before CI ran them, 24,000 mutated turns and 12,000 mutated records and images each passed under ASan and UBSan through the replay driver. The `tool_stream` harness found a defect in its first 30 seconds in CI, fixed here (R6.68.19): a NUL byte in a model's turn (a byte-fallback token) hid everything behind it from the tool-call demultiplexers, which searched their length-tracked buffer with `strstr`, so a call after a NUL was found when the turn arrived a byte at a time and missed when it arrived in one piece, and the streamed and buffered replies of one turn could disagree. Every search over model output is now bounded by its length.

- **Stability pass follow-ups** (R6.68.18, R6.68.20): `run_completion` built a constrained turn's closure report, and freed the turn's document, before the final flush of the reasoning splitter. A partial tag the splitter still held was then appended to the freed document, which allocated a buffer nothing released (one leak per such turn, found by the Linux lane's run of the whole suite under AddressSanitizer), and the report was computed without those bytes. The report is now built once the turn's last byte is in the document, and the document is freed on every exit. The serve soak compares swap-phase memory per resident model (the first Windows run reported two models' size difference as growth) and reads the macOS footprint where the server reports it. The test harness keeps a server's whole log, and names it, when the server stops uncleanly after a passing test: it used to delete the file and quote 2,000 bytes, which lost the stack of any sanitizer report raised at exit.

- **Stability pass on 1.17.0: the 11-02 sweep's kept defects, fixed at the root (R6.68.5).**
  - **Python client lease:** a non-numeric `RUNNER_UNVERIFIED_LEASE_TTL` raised `ValueError` out of
    `StartupLease.acquire()`. It is now read as the C lease reads it, a finite number of seconds in
    [0, 1e12]; anything else (`soon`, `nan`, `inf`, a negative value) is warned about and the default applies.
  - **Shadowed names:** four inner declarations that hid an outer one of the same name were renamed: a system
    message's text in the Muse template (it hid the render state `st`), two counters in the tokenizer's NFC sort
    (they hid the input `s`), a Responses item's out-of-memory flag (it hid the request's `oom`), and the prefix
    cache's TTL in the server's start (it hid the model's idle `ttl`). No behaviour changes; clang's `-Wshadow`
    now finds none in `src/` on macOS.
  - **File reads past 2 GiB on Windows:** `plat_read_file` took the size from `ftell`, whose `long` is 32 bits on
    Windows, so a file past 2 GiB read as an error there. The size now comes from `plat_file_size`, 64-bit on every
    platform (`_ftelli64` on Windows, `ftello` elsewhere), and a size that cannot fit a buffer is refused before the
    allocation. Today's callers read small files; this was a trap for the next one.
  - **Two shell drivers:** `scripts/artifact-baseline.sh` bounded every step with GNU `timeout`, which macOS does not
    have, so every step failed with "command not found"; it now uses `timeout` or coreutils' `gtimeout`, and with
    neither refuses once, saying how to get it. `scripts/cuda-gate.sh` took `GATE_LOCK` with a test and then a
    create, so two gates started together could both run; the lock is now one exclusive create (noclobber), and an
    existing lock is never overwritten.
  - **Shadow installer, the Codex path:** `hooks.json` was parsed unguarded and checked with bare asserts after
    the backup was written, and both harnesses' settings files were rewritten with a plain write that truncates
    first. Both paths now refuse a file they cannot merge into before writing anything, with the reason, and
    rewrite through the atomic writer; `shadow install` and `shadow uninstall` print the refusal instead of a
    traceback.
  - **`runner lineage DIR` out of memory:** when one of the walk's three per-answer arrays could not be
    allocated, the walk skipped its fill loop and then read through the missing one, a crash. It now refuses with
    one error line and exit 2, and gives back everything it allocated; `test-lineage-oom` fails each allocation in
    turn.
  - **`runner derive --target-size` out of memory:** the planning step grew three arrays with `realloc` assigned
    over its own pointer, so a failed allocation crashed and lost the old block, and its groups were never freed,
    on success included. Every growth and copy is now checked and ends in one error line, each return frees what
    it owns, and `test-derive-oom` fails each allocation in turn.
  - **macOS tray, leaks on every menu open:** `tray_macos.m` was built without ARC and released nothing, so each
    menu rebuild leaked every item and submenu it replaced, and the icon dump leaked its bitmap. The file is now
    built with ARC (from its own make variable, so a release's `CFLAGS` cannot drop it), and the status item is held
    by a strong reference instead of an autorelease pool that happened to outlive the run loop.

- **Server: three fixes taken over from the review of the cancel route and the instance registry (R6.68.8,
  R6.68.9, R6.68.10).**
  - **A cancel during a stop:** a cancel's waiter is a detached thread that read its request's slot while it
    waited, and a stop (or an in-process restart) freed the slots without waiting for it, so the waiter read freed
    memory. The server now counts its waiters, wakes them at the stop, and frees the slots after the last one has
    answered; a cancel whose request never ended answers 503.
  - **A cancel whose request ended while another ran:** a cancel waited for its slot's last-ended request to be
    its own, so when a quick request on the same slot ended within one poll, the cancel waited out its 30 s and
    answered 504 although its request had stopped, and reported the other request's token count. The end of a
    cancelled request is now recorded where its cancels wait, with its own count, and nothing later overwrites it.
  - **The instance registry's numbers:** a record's `pid`, `started`, `procstart` and `port` were converted to
    integers before any check, so a record holding 1e300 was undefined behaviour, and a port of 8080.5 or a start
    of "soon" was taken as given. Each is now checked to be a whole number in its range first, and a record that
    fails is refused like a corrupt one. The startup lease's owner pid is bounded by a 32-bit `long`, as on Windows.

- **CI runs the tests whose anchors live outside the repository** (R6.68.11): the gold-logits tests against the publisher's float32 forward in torch, the activation trace against transformers, the DeltaNet backward against autograd, the tokenizer diff against Hugging Face tokenizers, the receipt signature checked by an independent Ed25519, the sigstore model-signing round trip, and the KLD and lookup-draft cases that need a real model file. Each skips where its package or the file is missing, and no CI job nor the release gate provided them, so these tests ran on no change at all. A new `anchors` job installs the packages, fetches SmolLM2-135M-Instruct Q8_0 by its pinned sha256 behind a cache, and runs `make test-anchors`, which allows no skip but the one macOS-only refusal; `pr-gate` requires it. Run by hand before the job existed, all of them passed: 87 passed, 1 skipped (that refusal).

- **macOS: `/health` and `/metrics` report the process's physical footprint (R6.68.12).** A serving process on Metal
  read 13 MB in `rss_bytes` while its footprint was 638 MB, 598 MB of it Metal: the resident set leaves out the GPU
  driver's memory for the process, so a device-memory leak on Metal was invisible.
  - **New:** `phys_footprint_bytes` in `/health` and the `runner_phys_footprint_bytes` gauge in `/metrics`, the
    kernel's physical footprint (the figure Activity Monitor and `footprint` show). `rss_bytes` is unchanged.
  - **Not on Linux or Windows:** the field is absent there, since neither has one figure that adds device memory to
    RAM and their GPU memory is reported apart.
  - **The shared-weights leak check** no longer skips on Metal: it runs its two-window rule on the footprint.

- **The hostile-geometry sanitizer cases run in CI** (R6.68.16): nine cases of `tests/test_hostile_geometry.py` run a refused width and six valid fixtures through the ASan and UBSan build and skip without a fresh `runner-debug`. `make test` does not build it, and the sanitized CI job, which does, did not name the file, so the cases ran in no job. That job now generates the fixtures (one of them was only ever made by the Metal recipe) and runs the file with no skip allowed.

- **A serve soak, `scripts/soak-serve.py`** (R6.68.7): one instrument for the slow failures a test binary cannot see, run against a real model on any backend. It starts a server, repeats one greedy request and holds every reply to the first, cancels requests mid-prefill and mid-decode by closing the socket and by request id, walks the context limit (at the window, one token over, a generation that runs into it), runs several clients at once (plain, streamed, JSON schema, tool call) and holds each to the reply it got alone, alternates two models with `--model2`, and reads the process's memory and the device's at checkpoints. A reply that changes, a slot that stays busy, a 5xx, a cancel that is not answered, or memory that keeps growing after warm-up is a finding and the exit status is 1. `tests/test_soak_serve.py` runs every phase small on the fixture model in `make test`, with a negative control: a cancel phase in which no cancel can land must be reported as having measured nothing. The release checklist gains the soak as a macOS gate.

- **Two Windows defects from the stability pass's test lane** (R6.68.13, R6.68.14): `scripts/changelog-entries.py fold`, the release checklist's changelog step, crashed with an IndexError on a Windows host with a native Python, because it named entries with the host's separator and the entry check splits on a slash; entries are now named as repository paths on every host. And `tests/test_certify_envelope.py`, `test_truncation_benchmark.py` and `test_metal_moe_batch.py` looked for `runner` without `.exe`, so 18 tests skipped as "not built" on Windows with the binary built (twelve certify-envelope tests and the truncation test had never run there); they name both spellings now, and `tests/test_ci_contracts.py` fails on any test that resolves the binary by its bare name. The hosted Windows job never saw the fold crash: it runs MSYS2's Python, whose paths have slashes.

## v1.17.0 - 2026-11-12

- **CUDA: Q4_K, Q5_K and Q6_K prompts carry a lone decode's bits on the scalar route (R6.41).** A prefill over
  8 columns of these types, without the tensor-core GEMM, ran `k_gemm_q4_K`, `_q5_K` or `_q6_K`, which sum in another
  order than the decode GEMVs. It now runs the GEMVs' width-classed twins, 16 columns a pass, as Q4_0 has since R6.35.
  - **Where that route is the default:** on the architectures without the tensor-core promotion (deepseek2,
    qwen35moe, mellum, nemotron_h_moe, Q5_K on qwen4exp, Q6_K or Q5_K on clef), so their prefill bits change there.
  - **Gated:** `test-prefill-bits` holds a prompt's layer-0 KV rows to a batch-1 prefill's, byte for byte, at batch 9,
    17, 24 and 64. Before, 17 to 24 of 16,384 K values differed on the Q4_K, Q5_K and Q6_K fixtures, and 70 to 72 of
    65,536 on Mellum2.1-12B-A2.5B Q4_K_M.
  - **Speed on that route** (Kvist-14B with 104 tensors in each type, the type's share alone on the scalar route, the
    whole file on an RTX 5070, medians of six interleaved runs): prefill Q4_K 226.6 to 433.1 tok/s, Q5_K 234.5 to
    440.0, Q6_K 249.4 to 355.2.
  - **Unchanged:** wherever the tensor-core GEMM runs; a chunked `--score` there is byte-identical before and after.

- **CUDA: Q8_0 prompts carry a lone decode's bits on the scalar route (R6.50).** A Q8_0 prefill over 8 columns
  without the tensor-core GEMM ran `k_gemm_q8_0`, which sums in another order than the decode GEMV `k_gemv_q8_0`. It
  now runs that GEMV's width-classed twins, 16 columns a pass, as Q4_0 (R6.35) and the K-quants (R6.41) do.
  - **Where that route is the default:** Q8_0 tensors on deepseek2, qwen35moe, mellum and nemotron_h_moe, and on
    spark2_5 (whose tensor-core scope is Q4_K and Q6_K), so their prefill bits change there.
  - **Gated:** `test-prefill-bits` now runs on `test-q8.gguf` too. Before, 4 K and 1 V of its 4,096 layer-0 values
    differed from a batch-1 prefill, and 19 K and 38 to 40 V of 32,768 on a Kvist-14B file with Q8_0 attention.
  - **Speed on that route** (the Q8_0 share alone on the scalar route, the whole file on an RTX 5070, medians of six
    interleaved runs): prefill 301.9 to 446.1 tok/s on Clef-Flash Q4_K_M (51% of its weights Q8_0), 158.2 to 234.0
    on Kvist-14B with Q8_0 attention and gates.
  - **Unchanged:** wherever the tensor-core GEMM runs; a chunked `--score` there is byte-identical before and after.

- **CUDA: Q3_K prompts carry a lone decode's bits on the scalar route (R6.54).** A Q3_K prefill over 8 columns
  without the tensor-core GEMM ran `k_gemm_q3_K`, which sums in another order than the decode GEMV `k_gemv_q3_K`. It
  now runs that GEMV's width-classed twins, 16 columns a pass, as Q4_0, the other K-quants and Q8_0 do.
  - **Where that route is the default:** Q3_K tensors on the architectures without the tensor-core promotion, and on
    qwen4exp, which holds Q3_K off the tensor cores.
  - **Gated:** `test-prefill-bits` runs on a new Q3_K fixture, `test-q3k-prefill.gguf`. Before, 8 K and 1 V of its
    16,384 layer-0 values differed from a batch-1 prefill at every width.
  - **Speed on that route** (Kvist-14B with 104 tensors in Q3_K, alone on the scalar route, the whole file on an RTX
    5070, medians of six interleaved runs): prefill 232.5 to 350.0 tok/s.
  - **Unchanged:** wherever the tensor-core GEMM runs; a chunked `--score` there is byte-identical before and after.

- **CUDA: every type with a width-classed twin prefills through it on the scalar route (R6.57).** A prefill over 8
  columns without the tensor-core GEMM ran a tile kernel for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and
  MXFP4 (`k_mv_*_b`, which folds the block scale per term), and that kernel sums in another order than the batch-1
  kernel. Now any type whose batch-1 route has a loaded 16-wide twin prefills through the twins, by capability rather
  than by a list of types, as Q4_0, the K-quants and Q8_0 already did. F32, F16 and BF16 keep their tile kernel,
  which already is their batch-1 kernel's twin.
  - **Where that route is the default:** MXFP4 dense tensors on every architecture (MXFP4 has no tensor-core kernel;
    gpt-oss's MXFP4 is expert banks, which run the MoE kernels and do not change); the codebook i-quants on the
    architectures without the tensor-core promotion, such as clef.
  - **Gated:** `test-prefill-bits` gains a route table (every type with a 16-wide twin prefills through it) and six
    fixtures. Before, 23 to 36 of 16,384 layer-0 K values of the codebook fixtures differed from a batch-1 prefill at
    every width. MXFP4 differed only in rows longer than 1,024 values, so its fixture has rows 2,048 long
    (`make-test-model.py --n-embd`): 27 to 28 K values before. A whole-file MXFP4 rewrite of Kvist-14B read 21 to 23
    of 32,768.
  - **Speed on that route** (RTX 5070, prompt tok/s, medians of six interleaved runs): MXFP4, a whole-file MXFP4
    Kvist-14B on its default route, 6.7 to 63.3 (9.4x). Each codebook type alone on the scalar route (an experiment
    harness): IQ3_S 31.0 to 204.5 on a Kvist-14B IQ4_NL mix; on Qwen3.8-27B GSQ-RCO (63 of 64 layers on the
    device), IQ4_XS 16.0 to 157.0, IQ3_XXS 21.5 to 111.4, IQ2_S 71.9 to 192.3, IQ2_XS 96.6 to 210.9, IQ2_XXS 143.3 to
    235.3.
  - **Unchanged:** wherever the tensor-core GEMM runs; a chunked `--score` there is byte-identical before and after.

- **CUDA: the types without a width-classed twin prefill one column at a time on the scalar route (R6.58).** A
  prefill over 8 columns without the tensor-core GEMM ran a tile kernel for Q5_0, Q5_1, IQ4_NL, NVFP4, Q2_K, Q4_1,
  IQ1_S, IQ1_M and Q2_0 (`k_mv_*_b`), which sums in another order than the batch-1 kernel. It now runs the batch-1
  kernel once per column, which gives each column a lone decode's bits by construction and is faster: prefill on a
  Kvist-14B file went 9.3 to 90.5 tok/s in Q5_0, 6.2 to 62.1 in Q5_1 and 9.6 to 68.1 in IQ4_NL (RTX 5070, each type
  alone on the scalar route). A scaled tensor's 2 to 8 columns (NVFP4's companion), which the small-batch route did
  not take, run the same way.
  - **Where that route is the default:** these types on the architectures without the tensor-core promotion (NVFP4
    and the K-quant family are promoted on the trusted list, IQ4_NL with the codebook types).
  - **Gated:** `test-prefill-bits`'s route table now fails any type whose prefill reaches a tile kernel that is not
    its batch-1 kernel's twin, and it runs four new fixtures. Before, 4 to 37 of their layer-0 K values differed from a
    batch-1 prefill at every width.
  - **Unchanged:** wherever the tensor-core GEMM runs; F32, F16 and BF16, whose tile kernel is the batch-1 kernel's
    twin; `RUNNER_CUDA_SMALL_BATCH=0`'s tile for 2 to 8 columns.

- Make compiler warnings fatal in CI or `STRICT=1` test builds, including specialized flags and fuzzers; fix the Windows ULP-distance oracle's undefined shifts/overflow, harden KD and CUDA staging bounds, bound the shadow-client probe path, and correct unused helpers, an explicit f16 job initializer, tray-test I/O/capture bounds and a spliced comment.

- **A cancel by id reaches a request while its model loads (R10.12.3.3).** A completion request's `X-Request-Id` is
  now taken before its model is swapped in, not after, so `POST /v1/runner/requests/{id}/cancel` finds it during the
  load; before, it answered `404 request_not_found` until the load was over.
  - **In the `--wait-for-vram` queue** the cancel stops the wait under the rule a disconnect follows: a live client
    queued behind keeps the wait and its place.
  - **During a load past the queue** the load is not stopped: it finishes and stays resident. The request ends before
    its prompt with `409 cancelled`, and the cancel answers phase `loading`, `generated` 0, `request_id` null.
  - **`/health`** lists such a request with phase `loading`.
  - **Tested** on the generated test model with the VRAM-wait test hooks: the cancel stops a queued wait, keeps it
    for a live client behind, lets an admitted load finish and stay, and an id taken early is released on every exit
    (unknown model, failed load, refused load, disconnect).

- Correct the Qwen3-30B-A3B selective file's roster and public descriptions to lead with its 2026-11-02 float32-reference measurement, retaining the older Q8_0-relative tuple only as dated history.

- **`scripts/hub-upload-check.py`: the plan of a Hugging Face Hub upload, refused when it would overwrite or delete a
  file nobody named (R7.1.5).** Until now only the research-record packages had a scripted guard; model files and
  cards were checked against the remote listing by hand.
  - **What it does:** for a repo id and the local folder an upload will send, it lists each addition, replacement and
    deletion against the remote by content (the sha256 the Hub records for an LFS file, the git blob id for a small
    one) and counts the identical files. It exits 1 on any replacement or deletion whose repo path is not named with
    `--replace` or `--delete`, and 2 when it cannot check (a missing repository, a name that matches nothing).
  - **Declarations:** `--partial` for an upload that deletes nothing (a card on its own), `--new` for a repository
    that does not exist yet, `--path-in-repo` for a folder that lands under a prefix, `--repo-type` for datasets and
    Spaces.
  - **Read only:** it uploads nothing. A public repository needs no token; a private one takes `HF_TOKEN` from the
    environment, which is never printed and goes only to the endpoint's own origin: a redirect or a next-page link
    to any other origin is refused with exit 2 before a request is made. The plan names the remote commit it
    compared against, to pass as the upload's `parent_commit`.
  - **Tested** against a stub Hub on loopback, no network: a silent overwrite is refused, a named one is allowed, an
    addition passes, identical files are a no-op, a change on a later listing page is still caught, the token is sent
    and never printed, and the small-file id equals `git hash-object`.

- Envelope certification now verifies all recorded compatibility-report hashes before using an entry, refuses conflicting same-name evidence, and labels legacy hashless name matches as unverified in `quality.evidence_match`.

- **NVIDIA Nemotron 3.5 Bolt-74B-A9B joins the roster on NVIDIA's own 24GB GGUF, text path, CPU only (R4.27.4.4 S2).**
  Its MoE blocks differ from one another.
  - **What varies by block:** the expert count (240 to 496, from each block's router), the expert width (1,536 to
    2,560) and, in the file's format, the top-k (9 on every block). The experts run at a latent width of 1,024, as
    Super's do.
  - **The loader:** Runner read `expert_feed_forward_length` and `expert_used_count` as scalars, got 0 from Bolt's
    per-layer arrays, and refused the file.
    - Each block now runs its experts at its own width, and the widest sizes the scratch.
    - The arrays must have one entry per layer and 0 on the blocks without experts.
    - A top-k that differs between blocks is refused by name: no published file varies it.
  - **CUDA refuses varying expert widths by name**, and such a file takes no default split. Bolt is refused earlier,
    by the LatentMoE rule. Metal refuses the family.
  - **`--info fit` says a LatentMoE or varying-width file loads on the CPU.** Before, it sized a GPU split for Super and
    Bolt that the load never takes.
  - **Agreement, not fidelity:** the 149.8 GB BF16 parent cannot be held in float32 on the lab host, so the row is
    anchored on llama.cpp 36a0e2f on the same bytes:
    - mean KLD 0.0072 and margin-qualified top-1 100.00% (bars 0.05 and 97%, 300 positions); raw top-1 96.33%, every
      differing pick inside the tie band;
    - the tokenizer 0 of 721 and 0 of 128,635;
    - template conformance 24 of 24 in text and tokens.

    Record: [docs/compat-reports/nemotron-3.5-bolt-74b-a9b-2026-11-11](docs/compat-reports/nemotron-3.5-bolt-74b-a9b-2026-11-11/README.md).
  - **Limits of this release:**
    - NVIDIA names the file for a 24 GB card; it runs on the CPU.
    - The CPU prompt pass is slow: a cold 4,470-word prompt took 494.6 s where llama.cpp took 65.9 s on the same 40
      shared CPUs (R3.57).
  - **Gates:**
    - `tests/test_nemotron_hetero_moe.py` holds a heterogeneous toy (4 experts 16 wide, 6 experts 24 wide) to a twin
      with every expert zero-padded to 24, on the token and the batched pass, latent and plain. It adds a zeroed
      narrow-block control, four refusals, and the CUDA refusal.
    - The Nano-30B-A3B, Lightning-30B-A3B and Super CPU pins, the Nano-30B-A3B and Lightning-30B-A3B CUDA pins, and
      Nano-9B-v2's CUDA pin are byte-identical.

- **`RUNNER_CUDA_ATTN_TC_FULL` is removed (R18.8.1).** The switch put the CUDA tensor-core prefill attention on
  the layers without a window in a family the default keeps on the tiled kernel (gpt-oss's full-attention layers).
  It was opt-in, and on gpt-oss-20b it read further from the publisher's float32 reference than the tile (mean KL
  0.04065 against 0.04039, Wilcoxon p 3e-18). Setting it now does nothing; the default route is unchanged.
  `RUNNER_*` switches are outside the compatibility promise ([versioning](docs/versioning.md)).

- Collect the existing artifact-publication checks in the release checklist, with sources for publisher roles, checksums and guarded updates, model admission evidence, record locations, cards and excluded material.

- Document LoRA adapters for typed decisions, with TD-001's historical transfer agreement and request-time cost, and explain that imported q/k LoRA B rows must match the base GGUF's converter order.

- Commit hygiene now refuses agent-named authors or committers with a personal email address; agent identities must use a noreply or `.local` address. Document the worktree identity setup.

- Document the typed-decision execution cost: fresh canonical state feeds, solo option conditionals, historical TD-000 request times, and the equality gate required before batching conditionals.

- **`--shadow-mode` is deprecated (R18.8.10).** It keeps working with a warning for the rest of 1.x and is removed no
  earlier than 2.0.0, as [the versioning policy](docs/versioning.md) requires for a flag.
  - **What changes now:** `--help` marks the flag deprecated, and every run of it prints one warning line on stderr
    before the installer's own output. The install itself, the `/shadow` offload and the Python package
    `xyntetik_runner.shadow` are unchanged.
  - **Why:** the shadow research it served closed in September 2026, and the offload is planned to leave Runner for
    a companion product ([ownership](docs/ownership.md#6-shadow-a-companion)).
  - **Tested:** the warning prints exactly once, also when the install is refused; the install still writes its
    files and config; no other mode warns; `--help` carries the policy's terms, read from `docs/versioning.md`.

- **`--info fit` sizes a load whose experts stay on the host as that load is placed (R3.58).**
  - **The loads covered:** `--cpu-moe`, and nemotron_h_moe's CUDA default.
  - **How the fit now charges them, as the loader does:**
    - the mixer tensors and dense FFNs, each rounded as uploaded;
    - the tables without expert biases;
    - no expert scale table.
  - **A second fix:** the fit now charges nemotron_h_moe's Mamba-2 state, which it missed.
  - **Measured:** on the fixtures and on Nemotron-3.5-Lightning-30B-A3B Q4_0, the fit's fixed charge, the split's
    weight bytes and its layer count equal the load's, byte for byte (`RUNNER_DEBUG_VRAM`).
  - **Not covered:** `--cpu-moe N` and `--cpu-moe auto` keep only some banks on the host (the deepest N, or those
    the budget leaves out). The fit still sizes a full offload for them, and says so in a line.

- **nemotron_h_moe runs its trunk on CUDA by default (R3.43a.2).**
  - **What happens:** a GPU load that names no `--cpu-moe` keeps the experts and their routing on the host and runs
    the Mamba-2 and attention blocks on the device. The router has no device kernel.
    `RUNNER_CPU_MOE_DEFAULT=0` runs such a file on the CPU, the earlier default.
  - **The quality read (D1),** on Nemotron-3.5-Lightning-30B-A3B Q4_0 against NVIDIA's float32 model, 4,500 positions:
    - decode: mean KL 1.0003x the CPU's (Wilcoxon p 0.17, McNemar 0 and 0 discordant);
    - prefill: 0.995x (p 0.39, 3 and 3).
  - **The rows:** the Lightning and Nemotron-3-Nano-30B-A3B rows gain CUDA cells on this route.
  - **`--info fit`** still sizes a full offload for such a file (and for any explicit `--cpu-moe`), and now says so in a
    line.

- **nemotron_h_moe runs its trunk on CUDA with `--cpu-moe` (R3.43a).**
  - **Why it was refused:** its router (a sigmoid with a selection bias) has no device kernel, so CUDA refused the
    whole model, although a layer whose experts are on the host is routed there.
  - **Admission:** the router and selection-bias refusals now apply only when an expert bank runs on the device.
    `--cpu-moe` runs the Mamba-2 and attention blocks on the device and the experts and their routing on the host.
    The `--cpu-moe auto` fit never moves such a bank onto the device.
  - **A second defect, fixed:** an MLP-only block jumped straight to the device FFN, past the host branch.
  - **Measured on Nemotron-3.5-Lightning-30B-A3B Q4_0** (24 GB MIG slice, medians of 5 interleaved runs): 30.97
    against 19.88 decode tokens/s and 99.6 against 60.2 prompt tokens/s.
  - **The default is unchanged:** without the flag the file runs on the CPU, and the message names `--cpu-moe`.

- **NVIDIA Nemotron 3.5 Super VL-120B-A12B joins the roster on ggml-org's Q4_0, text path, CPU only (R4.27.4.4 S1).**
  Its MoE is LatentMoE.
  - **What LatentMoE is:** the 512 routed experts (top 22) run at a latent width of 1,024, between
    `ffn_latent_down` and `ffn_latent_up`. The router and the shared expert read the full 4,096-wide input. Runner
    refused the file before: the nemotron_h_moe expert cap was 256, and it had no latent pair.
  - **The loader:**
    - It reads `moe_latent_size`, which reads 0 when absent, so Nano and Lightning are unchanged.
    - It refuses by name a width past n_embd, a half pair, or a pair the metadata does not declare.
    - The expert count is bounded by the general MoE cap (1,024). The selected count stays at most 256.
  - **CUDA and Metal refuse LatentMoE by name.**
  - **Agreement, not fidelity:** the 241.5 GB BF16 parent cannot be held in float32 on the lab host, so the row is
    anchored on llama.cpp 36a0e2f on the same bytes:
    - mean KLD 0.0136 and margin-qualified top-1 99.67% (bars 0.05 and 97%, 300 positions);
    - the tokenizer 0 of 721 and 0 of 128,635;
    - template conformance 24 of 24 in text and tokens.

    Record: [docs/compat-reports/nemotron-3.5-super-vl-120b-a12b-2026-11-10](docs/compat-reports/nemotron-3.5-super-vl-120b-a12b-2026-11-10/README.md).
  - **A limit of this release: the CPU prompt pass on this file is slow.**
    - The 8,103-token gold corpus as one prompt took 655.8 s to 717.0 s at the default -b 64, and llama.cpp 60.3 s on
      the same 40 shared CPUs.
    - The routed experts are 82% of it (R3.57).
  - **Gates:**
    - `tests/test_nemotron_latent_moe.py` holds a latent toy to its composed twin (the experts multiplied through
      the pair) on the one-token and the batched pass, with a zeroed-experts control and the four refusals.
    - The Nano-30B-A3B and Lightning-30B-A3B CPU pins and Nano-9B-v2's CUDA pin are byte-identical.

- **The scheduled-run alert reads the failing job's log (R6.45.3).**
  - The job-log endpoint redirects to a signed storage URL on another host. urllib copied the bearer token onto
    that request, and the storage host refused it with "HTTP Error 401: Server failed to authenticate the request".
    So the issue named the job and step, but not the line.
  - `scripts/scheduled-ci-alert.py` now drops the Authorization header on a redirect to another host, and keeps it
    on the same host.

- **Run the Metal tolerance gates on eligible fixtures (R6.45.1).** `test-gates-core` now gives
  `test-mv-tol` Q8_0, Q4_K and Q4_0 fixtures and runs cooperative attention on the wide Q4_K
  fixture with both normal and q8 KV storage. An explicitly named matvec fixture without a
  supported tensor fails, named Metal fixtures must dispatch the tested route, and a q8 request
  must actually activate q8 KV. The existing zero-flip and mean-deviation tolerances are unchanged.

- **A scheduled CI run that ends red now reaches someone (R6.45.2).**
  - **Why:** scheduled runs sit outside pr-gate, so nothing escalated one. The daily macOS suite went red on 2026-11-10
    at 12:51Z and was noticed four hours later.
  - **The new workflow:** `.github/workflows/scheduled-alert.yml` runs when a scheduled run of `ci`, competitor
    freshness or the deterministic-CI example completes, through `scripts/scheduled-ci-alert.py`.
    - A run that ends failed, cancelled, timed out or unable to start opens one issue per workflow labelled
      `scheduled-ci-red`, or comments on the one already open.
    - The issue names each failed job, its failed step and that step's first failing log line.
    - A green run of the same workflow closes it.
  - **Testing on a branch:** a `workflow_dispatch` with a run id (dry run by default) tests the workflow there,
    since a `workflow_run` trigger fires only from main.
  - `docs/release-checklist.md` makes no open `scheduled-ci-red` issue a step before the cut.

- **Docs: README and MANUAL checked against every changelog entry of 1.13.0 to 1.16.0 (R7.20).** The version pins
  were right; these were missing or stale, and are now stated:
  - **Missing from the manual:** `runner train --full` and `--kv-f32`, and the refusal of training with a
    `q8`, `k8v4` or `fp4` KV cache; the note a first CUDA load prints while the driver compiles the kernels; the
    refusal of a hybrid model's session image or KV snapshot written by 1.15.0 or earlier, with
    `RUNNER_KV_RECURRENT_ROWS=1`; `--info fit` reporting the CPU for the families the loader never puts on a GPU;
    `profile.route.tc_types` in a record; random response ids; a usage error ending with its error line;
    `RUNNER_MOE_HOT` giving the host copies back.
  - **Stale in the manual:** the CUDA route label was documented as `tc=promoted`, which the binary has not
    written since R3.39.2 (`tc=taken`, `tc=off`, `tc=moe-tc`, `tc=forced`); the not-implemented list still named
    `deepseek2` and multi-head latent attention, both admitted since 1.13.0.
  - **README:** it names `runner train --full`, the `oneOf` told apart by a `const` and the turns `--verify` now
    replays, and it is shorter: the caveats and limits it carried (build conditions, costs, what is refused, what
    is not covered) are left to the manual and the docs it links, which state them in full.
  - **Found right:** 1.13.0's and 1.14.0's entries, the tier wording (none left), `runner recommend`, the draft
    defaults, `RUNNER_PREFETCH`, the prefix cache's default, `--kv f32`, the cancel route and `ignore_eos`.

- **A tier-smoke pin names the build that wrote it, and empero's CUDA pin comes from the release recipe (R6.46).**
  - A partial split runs some layers on the CPU, where compiler codegen changes the last bits. That flips a greedy
    choice only where the prompt meets a near-tie.
    - empero-qwen3.8-35b-a3b-distill-q6_k's CUDA pin (34 of 40 blocks on the device) was written on 2026-11-06 by a
      clang build at `-march=native`.
    - Every release-recipe gcc build gives another reply, so the pin had mismatched on every such build since.
  - **The re-pin:** the pin is now the 1.16.0 release-recipe build's reply, which equals the row's CPU pin. The CPU
    pin matches as it was and is unchanged.
  - **The build in the record:** `scripts/tier-smoke.py` reads the serving binary's sha256 and compiler from
    `GET /v1/runner/provenance`. It writes them into the pin's manifest entry (`pinned.build`) and into the cell it
    records (`build`).
    - A runner without the route records none.
    - `scripts/model-tiers.py` refuses a `build` without both.
  - **The other CUDA cells recorded on 2026-11-06 match on the release binary:** sage-a1.5-q6_k,
    minicpm-v-4.7-35b-a3b-q8_0, and the selective mellum row.

- **docs: the train/serve mismatch, measured (R17.5.4).** New page: docs/train-serve-mismatch-2026-11.md. It covers
  Runner's sampler against its trainer on MiniCPM5-1B, beside vLLM with transformers in bf16 and NF4.
  - **It leads with what is not claimed:**
    - exact only on the CPU with `-b 1`;
    - the batched prompt pass and the CUDA sampler are close but not exact;
    - the stacks run different weights;
    - one model and one task;
    - the sampling-truncation deviation found in the first run, and its repair.
  - **The batched pass's figure is corrected:** MANUAL.md, docs/lora.md and docs/kernel-routes.md now give 1.45e-2 over
    384 rollouts on the 2026-11-09 file, where they said 3.1e-3 after one 60-token prompt (measured 2026-10-17).
  - **docs/lora.md's "a GPU sampler agrees token for token" is replaced by the measured figures:** the CUDA sampler's
    logprobs differ from the CPU trainer's by up to 2.02e-2, and 284 of 384 of its rollouts are the CPU sampler's.
  - The README's adaptation section links the page.

- **A release needs the full CUDA gate for its own version (R6.44).** Before, the release gate ran only
  `scripts/release-cuda-gate.py`'s smoke and probe. Those cannot see a device target gone red: test-cuda-qwen4exp's
  MTP check was red from 2026-11-04 to 2026-11-10, through 1.14.0 and 1.15.0 (R3.56).
  - **`scripts/check-release.py`** refuses a tag unless a CUDA class's `last_verified` row in
    `docs/device-evidence.json` names the version, was written by `scripts/cuda-gate.sh` and passed. Its summary must
    be committed at the row's `evidence` path, as `cuda-gate.v2`, passing, at the row's commit.
  - **`tools/release-make/make`,** first on PATH, gives every make a gate step runs the release recipe's CFLAGS on the
    command line. Without it a step relinked a different runner, because an environment CFLAGS is extended by the
    Makefile.
  - **`scripts/cuda-gate.sh`** fails a run whose binary changed under it (`binary-unchanged`).
  - **`scripts/release-cuda-gate.py`** names the model and the binary relative to the checkout, or by file name. An
    absolute `--model` had left `~/<dir>/<file>` in a report.
  - **docs/release-checklist.md** step 4 runs the full gate on the pin beside the release gate. It recommends the full
    gate on main daily, at a quiet hour, from a cron entry on the CUDA host or the coordinator's check-in.

## v1.16.0 - 2026-11-10

- **Fix CUDA layer placement accounting (R3.53).** The loader's weight estimate omitted per-layer norms,
  biases, sinks and scales that ride in the file upload, and both placement and `--info fit` omitted
  the separately expanded f32 layer tables. They now share one header-only charge for file bytes and
  table bytes. Tables are reserved once for the complete uploaded table set, including host-side
  layers; auto expert placement conservatively reserves possible expert-bias tables so run-based
  refinement cannot discard their charge. CPU-only fixture arithmetic covers a split moved by those
  bytes. Checked on CUDA: the added charge against the f32 bytes the uploader builds was exact on
  six files (gpt-oss-20b by default and with `--cpu-moe`, DeepSeek-V2-Lite, MiniCPM5, granite-4.0-h-micro,
  gemma-4-12B QAT) once the all-ones expert scale table was charged too (before that, 120 bytes short
  on gpt-oss-20b and 256 on DeepSeek-V2-Lite). A load can place one layer fewer on the device than
  before at the same budget: on DeepSeek-V2-Lite the earlier plan left about 8.4 MB per host-side
  layer uncharged and spent the headroom. Limits: under `--cpu-moe` the fit still charges expert-bias
  tables the loader does not build (conservative), and the 12 GB card's fit 23 against load 24 on
  gpt-oss-20b was not reproduced on the 24 GB device, whose percent steps are too coarse for it.
  README impact: no front-page, manual, site or model-card claim changes; this corrects accounting,
  not the model's arithmetic.

- **test-cuda-qwen4exp's NextN/MTP head check compares runs at the same draft width (R3.56).** Since R3.3.7 the
  default width is per backend: 2 for the NextN head on CUDA, 4 on the CPU. The check left it to the defaults, so it
  compared a 4-wide CPU run (92 dump lines) with a 2-wide device run (46) and failed on the count.
  - **Not a device difference:** at an explicit equal width the device head equals the CPU head line for line (kind,
    position and argmax the same, logit and gap within 1.0e-6, text identical), at width 2 and at width 4.
  - The check now runs both widths with `--draft-k` set on both sides. Its pairwise comparison also requires the
    same line kind and position, so a misaligned pair names itself.
  - It went unseen from 2026-11-04 to 2026-11-10 because no `scripts/cuda-gate.sh` run was recorded on main in that
    time: the release gate script runs the smoke and the probe, not this target. From R3.44 on, the target also
    failed earlier, at `--gpu-layers 2` (fixed in R3.54).

- **Hybrids load on CUDA with any layer split (R3.54).** Stateful layers own no KV rows: Mamba-2 layers never had any,
  and qwen35's DeltaNet layers have had none since R3.44. So a `--gpu-layers` split whose device layers are all
  stateful had 0 bytes of device KV, and `cuMemAlloc` refuses 0.
  - **Before:** the device init failed and the run finished on the CPU. 18 of the 30 splits of the seven hybrid test
    families did this: qwen35, qwen35moe, Clef, qwen4exp, granitehybrid and Nemotron-H at every split below their
    first attention layer.
  - **The fix:** the device K and V caches get a 256-byte allocation instead. The pointers still reach the attention
    launches, the KV gather and the 0-byte copies. The buffer is filled with 0xFF, which is NaN in every KV format,
    so a read of it would show. All 30 splits now run on the device within 1.43e-6 of the CPU, with no argmax flip
    and no NaN.
  - **Gates:**
    - `tests/test_qwen35_kv_rows.py` gains the partial-split case for every family, and joins `test-cuda-gated`.
    - Until now, no gate split a hybrid except test-cuda-qwen4exp and test-cuda-qwen35moe, and both had been
      failing.
  - **Also, test_cuda_fit_ring's note case on any card (R3.55).** `--reserve-vram` is a share of the card, so 10%
    forced the KV ring only on a small one. On a 24 GB device it left room for the flat cache. The case now takes
    the first whole percent at which the fit turns the ring on, and skips with that reason where none does.

- **`--kv f32` on CUDA (R6.33).** The f32 KV cache from R6.32 now runs on the device, including a partial split,
  where both halves keep f32 rows. Metal still refuses it, with the reason.
  - **What the refusal had been protecting:** before this change, R6.32's two refusals kept a kind-3 cache away from
    CUDA. One is at parse time, the other is model_alloc's `--gpu off` gate. Every device KV helper and every row-size
    site in cuda.c read a 3 as q8_0. The sites are:
    - the six kernels.cu helpers;
    - three shared-memory width expressions;
    - two store-unit counts (the solo forward and the microbatch).
    Lifting the refusals without the fix gives q8_0 blocks under an f32 label.
  - **The change:**
    - Every helper and site takes kind 3 first.
    - The width sites call `kv_bytes_of_kind`, the CPU's own width function.
    - f32 rows load as scalars, because a row staged in shared memory sits at an odd 32-bit stride.
    - The grouped decode attention, past head_dim 164, and the absorbed MLA decode, at deepseek2's 576-wide rows,
      cannot fit f32 tiles in their shared memory. Their size functions now say the route does not apply, and those
      layers take the generic kernels.
    - The shared-weights identity gives f32 its own value.
  - **Checks:** `make test-cuda-kv-f32`, a cuda-gate step.
    - The device's f32 store is the source bytes.
    - The device's f32 cache agrees with the CPU's on top-1 at every teacher-forced position, on five fixtures. Its
      distance from the CPU's f32 is within 3x the f16 pair's, and it sits closer to the CPU's f32 than the device's
      f16 cache does. Each fixture names the routes its run must dispatch.
    - The microbatch stays byte-identical over an f32 cache.
    - A mutant per site, each run once: lifting the refusals alone, or putting back any one width site, store-unit
      count or shared-memory guard, turns this gate red.
  - **Also:**
    - test_kv_tol now fails a GPU arm that loses the device mid-sweep and finishes on the CPU. Before, such an arm
      agreed with the CPU perfectly and passed.
    - `--caps` reports `kv_f32`.
    - `tests/test_kv_f32.py` gains the CUDA receipt, verify and partial-split cases.
  - **Unchanged:** CUDA `--score` and seeded sampling under the default cache and f16, q8, k8v4 and fp4 give the same
    bytes before and after, on six files: 60 command pairs, on the device.

- **Structured output: a `oneOf` or `anyOf` of objects told apart by a `const` is now enforced, each object with its own properties (R2.8).**
  - **Before:** objects in a union all start with `{`, so Runner refused them as ambiguous. The one exception was
    the agent `{tool, args}` action, and only when every tool declared the same required keys. So a schema could
    not say that one operation takes a lambda and another does not, and a flat schema in its place admitted
    statements of the wrong shape.
  - **Now:** when one shared property is a required `const` with a different value in each object (a
    discriminated union, as Pydantic writes for `Field(discriminator=...)` and OpenAPI for `discriminator`), the
    union compiles. The discriminator is the first key written, and every later key follows the object it named,
    with that object's own properties and required set. A key the object lacks is refused at its first byte, so
    constrained sampling never walks into a dead end. A document cut by `max_tokens` closes inside the object it
    was in.
  - **Also:** `discriminator.propertyName` names the property, other alternatives may stand beside the objects
    (`anyOf: [A, B, null]`), and the `{tool, args}` action with different required keys per tool now compiles.
    Objects with nothing to tell them apart are still refused, with a message saying what would make the schema
    compile.

- **`scripts/act-trace-hf.py` records the right `mixer` point on gemma-family blocks (R6.38).**
  - **Before:** it hooked the first of `post_attention_layernorm`, `pre_feedforward_layernorm` and `ffn_norm` that
    a block had. On Gemma 2, 3 and 4 blocks (sandwich norms) `post_attention_layernorm` normalizes the attention
    output before the residual add. So the reference trace's `mixer` row was the attention branch, not the
    residual after it, which is what Runner's `mixer` point is, and `scripts/act-trace-diff.py` would name the
    mixer as the departing branch on a correct forward.
  - **Now:** the point follows the block's norm layout: `pre_feedforward_layernorm` on sandwich-norm blocks,
    `post_attention_layernorm` on pre-norm blocks, `ffn_norm` under that name, and none on a post-norm block
    (OLMo 2), where no norm reads that residual. A test on a tiny Gemma 3 block checks the hooked input against
    transformers' own forward: the block input plus the normalized attention output. No committed record
    compared the mixer point on a gemma model with this script.

- **Docs, correction: the gemma-4-26B-A4B reference rows were measured against a reference whose embedding was
  scaled by 53.0, where float32 gives 53.066.**
  - **The cause:** gold-logits' `--upcast-at-use` built gemma's embedding scale in BF16 (fixed, R6.36).
  - **Measured again, with Runner's side unchanged:**
    - The bf16 conversion: chat 0.0060 to 0.0007, raw 0.3226 to 0.4279.
    - The listed QAT Q4_0: chat 0.2183 to 0.2140, raw 1.4900 to 1.4648.
    - The chat number got better and the raw number got worse.
  - **The raw gap is the f16 KV cache's rounding, amplified out of format:** with `--kv f32` the conversion reads
    0.00003 on chat and 0.0055 on raw.
  - **Updated:** [the record](docs/gemma4-26b-reference-2026-11-10.md) and the tier row's anchor.

- **`--kv f32`: an f32 KV cache for any CPU run, a measurement instrument (R6.32).**
  - **What it is:** K and V are kept unrounded in the CPU inference attention, the cache the training forward already
    used under `--train-kv-f32`. It doubles the KV memory against f16 and is not a speed or quality default.
  - **Refusals:** a run that could reach a GPU is refused with the reason (the GPU kernels read f16, q8_0 and fp4
    rows). Add `--gpu off`.
  - **Where it is named:** receipts, the route, `--info doctor`, KV snapshots and session images all say `f32`, and
    `--verify` replays it. A snapshot or image refuses another KV type by name.
  - **Model keys:** the model key gains a token only when the cache is f32, so every existing f16, q8, k8v4 and fp4 key,
    prefix cache, snapshot and image is unchanged.
  - **First use:** gemma-4-26B-A4B's bf16 conversion against its publisher's float32 forward, 2,000 positions each,
    the reference with its embedding scale rebuilt in float32 (R6.36).
    - Chat text: mean KL 0.00068 with the f16 cache, 0.00003 with f32.
    - Raw text: 0.428 with the f16 cache, 0.0055 with f32. Out of format, that model amplifies the f16 cache's rounding.
  - **Checks:** `tests/test_kv_f32.py`, red on the previous binary (9 of 9 failed) and green now.
    - An f32 and an f16 cache score differently.
    - Rollouts a `--kv f32 -b 1` server samples equal the `--train-kv-f32` trainer's logprobs bit for bit on four
      fixture families. The f16 cache's do not.
    - Session images round-trip byte-identically, and snapshots carry `kv_type` f32.
    - Fifty `--score` and sampling commands on five fixtures under the default cache and every other `--kv` type give
      the same bytes before and after the change.
  - **Also:** the tied-V refusal now names the cache type it refuses, instead of saying "q8" for every quantized or
    f32 cache.

- **Docs: the gemma-4 E2B PLE study measured again against a full float32 reference: no verdict changes (R6.37).**
  - **Why:** its reference was gold-logits' `--upcast-at-use`, which built gemma's embedding scale in BF16 until
    R6.36.
  - **The result:** every row moves by at most 0.0001. Unsloth's Q5_K_M still misses the bar narrowly.
  - **Also:** the fixed upcast path matched the float32 load bit for bit on all 16,000 positions.
  - [The record](docs/compat-reports/gemma-4-e2b-ple-quant-2026-10-31/README.md).

- **Docs: the tool-use adapter's off-distribution result now stands first, wherever its 0.69 to 1.00 is quoted (R8.13.1).**
  - **What changed:** the README's adaptation section and the site's LoRA page quoted only the in-distribution
    figure (Qwen3-4B at 4 bits, exact tool calls 0.69 to 1.00 on the task's own held-out set). Both now lead with
    the shifted set: on 274 hand-written prompts unlike its training, over 23 tools with overlapping names, the
    base made 211 exact calls through the chat API and the adapter 200 ([the record](evals/tooluse-shifted/README.md)).
  - **The adapter's Hub card:** the same text is prepared in
    [evals/tooluse-shifted/hub-card-2026-11-10.md](evals/tooluse-shifted/hub-card-2026-11-10.md) for its
    status bullet, its "What this is, and is not" paragraph, its TL;DR and its first honest limit.

- **CHANGELOG entries are now one file per pull request, in `changes/`, folded in at release (R6.21).**
  - **Why:** every pull request added its entry to the top of `## Unreleased`, so each merge conflicted with
    the next open pull request on the same lines. Settling that cost a merge and a full CI run each, about
    25 minutes per pull request.
  - **Now:** a pull request adds `changes/<plan ID>.md` with its entry as this file will show it
    ([the format](changes/README.md)). The release runs `scripts/changelog-entries.py fold --write`, which
    puts the entries at the top of `## Unreleased`, newest first, and deletes the files.
    `scripts/check-release.py` refuses a release that leaves any behind.
  - **The check:** the commit-hygiene job runs `scripts/changelog-entries.py check`. It asks for exactly one
    new entry file, or a line `Changelog: none, <why>` in the pull request's description or a commit
    message. A release fold passes. A branch cut before the switch may keep its entry in CHANGELOG.md, with a
    warning, so pull requests already open merge as they are. A change only to `changes/` skips the engine
    jobs, as a change to CHANGELOG.md did.

- **Fix (scripts/gold-logits.py): `--upcast-at-use` scored gemma against a reference that was not its float32
  forward (R6.36).**
  - **The bug:** a load at the stored dtype builds the model in BF16. So a constant the model's code computes when it
    is built, gemma's embedding scale (the square root of the hidden size, a non-persistent buffer), was born rounded:
    53.0 on gemma-4-26B, where a float32 load gives 53.066. The later cast to float32 cannot give the digits back.
  - **Who it touched:** every gemma-3 and gemma-4 size builds the scale this way (transformers 5.16.1).
  - **The published rows scored this way:**
    - [the gemma-4-26B reference](docs/gemma4-26b-reference-2026-11-10.md) and its tier row;
    - the [gemma-4 E2B PLE study](docs/compat-reports/gemma-4-e2b-ple-quant-2026-10-31/README.md).
  - **Not touched:** the golden passes loaded in full float32. The other classes scored with the option build no
    constant differently in BF16.
  - **The fix:** the option now copies every non-persistent float buffer and float tensor attribute from a float32
    build of the same class (each parameter moved to the meta device as it is registered, so no weights are held).
    Then it compares each one and refuses, naming it, if any still differs. scripts/act-trace-hf.py shares the
    function, and nothing beyond torch and transformers is needed.
  - **Checks:** tests/test_gold_logits.py has a text-only Gemma 4 at hidden size 80, where sqrt(80) is not exact in
    BF16. Before the fix its logits differed from a float32 load's (8.9375 against 8.9442720) and the guard did not
    exist; now they are bit for bit equal, and an un-rebuilt load is refused by name. The equivalence check that
    passed before used hidden size 64, whose square root is exact in BF16.

- **`ignore_eos` on a completion request (R10.31).**
  - **Before:** only `/v1/runner/sessions` read it. Chat, completions, Responses and Messages took only the
    server-wide `--ignore-eos`, so a fixed-length rollout could not be asked for per request.
  - **Now:** every completion surface takes `ignore_eos` (a boolean; null reads as absent), as vLLM's and
    llama.cpp's servers do. The server's `--ignore-eos` is the default, and a request may turn it either way for
    itself. A grammar's completed document still ends the turn.
  - **Lifetime:** the value is installed per request and put back to the server's default when the request ends.
    The sessions route also restores the server's default after a session, where it used to turn it off.
  - **Receipts:** a served turn under it records `ignore_eos`, and `--verify --ignore-eos` replays it.
- **`--info fit` predicts the split a load makes with the KV ring (R3.50).** When the weights and the flat KV
  cache do not fit the GPU budget, a load gives sliding-window layers only their window plus the prompt batch
  (`--kv-ring auto`) and places more layers; the fit charged the flat cache and predicted fewer
  (gemma-4-26B-A4B at 4k: 22 of 30 against the load's 23). It now applies the same test, charges the same rows,
  and marks such a split `with the KV ring`; the `kv cache` line stays the flat figure. For a sparse MoE the
  split's RAM figure is labelled `hot set`, the figure it always was.
- **`--verify` replays a served tool turn (R1.1.3.1).** A tool receipt used to name only a digest and choice, so
  replay refused it as `UNVERIFIABLE`. Under the existing receipt opt-in, the `tools` constraint now carries the
  normalized declarations and effective protocol, parallel and thinking settings. Replay rebuilds the same
  native or generic grammar used by serving; changed declarations can diverge, and a digest mismatch is refused.
  Older receipts without the new fields stay `UNVERIFIABLE`, naming what they lack. Replay also binds the served
  template's thinking tags before resetting its constraint state. Scripted replies and separate reasoning
  sampling remain unreplayable. README, manual and receipts page now describe tool replay; model-card claims are unchanged.

- **A cancel route keyed by request id: `POST /v1/runner/requests/{id}/cancel` (R10.12.3.2).**
  - **What it is:** a client that wants an answer, rather than polling `/health`, names a running completion request.
    It can use the response id, or the `X-Request-Id` header it sent with the request. The header is how a buffered
    client knows the id before its reply arrives.
  - **How it answers:** the cancel is served from the accept loop, so on a one-slot server it does not queue behind
    the request it cancels. It replies once that request has stopped, with the phase it was in and how many tokens
    it had generated. An unknown id gets `404`.
  - **The cancelled reply:** a cut during generation keeps its partial output (`finish_reason` `length`,
    `finish_detail` `cancelled`). A cut while the prompt was processed answers `409 cancelled`. The server stays up.
  - **Ids:** a client id is checked (1 to 64 characters from `A-Z a-z 0-9 . _ : -`), and one that an active request
    already holds is refused with `409`. Request ids stay out of `/health`, which anything on loopback can read.
  - **Receipts:** a turn the server cut from outside the model (a cancel, the request's `timeout`, the drain) now
    records `serve.cut`, and `--verify` replays the recorded tokens and no further. Before, such a receipt replayed
    `DIVERGED`, because the replay ran on to `max_tokens`.
- **CUDA: a Q4_0 prompt carries a lone decode's bits on the scalar route (R6.35).** With the tensor-core GEMM off,
  a Q4_0 prefill over 8 columns ran `k_mvb_q4_0`, which sums in another order than the decode GEMV `k_gemv_q4_0`.
  It now runs that GEMV's width-classed twins, 16 columns a pass, so a prompt's rows carry the bits a lone decode
  gives them. Q4_K, Q5_K and Q6_K's prefill GEMMs still differ from their GEMVs in a few last bits (R6.39).
  - **Why it mattered:** under `--kv q8` the old difference crossed quantization steps. `test-kv-tol
    test-q40.gguf` read 10.7x its reassociation floor on an RTX 5070; it now reads 0.03x, and `test-gates-core`
    runs it.
  - **Speed on that route:** prefill of a 14B Q4_0 file with `RUNNER_CUDA_TC=0` went from 6.6 to 109.8 tok/s
    (median of six interleaved runs each).
  - **Unchanged:** wherever the tensor-core GEMM runs, which is the default for Q4_0 on the promoted
    architectures; a chunked `--score` there is byte-identical before and after.
- **Trays no longer stack (R9.5).**
  - **Before:** with an instance directory that could not be written (an `%APPDATA%` that refused files), a tray
    could not record itself, every launch found no tray, and each started another. Two launches inside a tray's
    start could also both start one, even with a working directory.
  - **Now:** a tray holds a lock in the instance directory (`tray.lock`) for its whole life, so the check and the
    claim are one step. It exits when it cannot record itself, and a launch starts no tray on a directory it cannot
    write, saying so on stderr. The lock is opened so no child inherits it, so a server the tray started does not
    keep it after the tray quits.
  - **Checks:** `test-tray-core` reproduces the stack on an unwritable registry (it failed before the fix), refuses
    a tray that cannot record itself before it runs, and shows a launch that sees a held lock starting no tray.

- **Docs: gemma-4-26B-A4B now has its publisher float32 comparison (R6.11).** The support entry listed it as owed.
  - **What was run:** two files, scored against the transformers forward of `google/gemma-4-26B-A4B-it` in float32
    on chat text, 2,000 positions ([the record](docs/gemma4-26b-reference-2026-11-10.md)).
  - **A fresh bf16 conversion** of the checkpoint reads mean KL 0.0060, margin-qualified top-1 99.79% (CPU), so the
    MoE forward tracks the reference.
  - **The listed QAT Q4_0 file** reads 0.2183 and 90.59% (CUDA), its own distance.
  - **Raw text, reported first there:** the conversion reads 0.3226, which is not explained.

- **The first CUDA load on a new card or driver says why it waits (R10.29).**
  - **Before:** the driver JIT-compiles the embedded PTX the first time, and the process sat silent. Measured on
    an RTX 5070: 33.4 s cold, 0.40 s warm from the driver's cache.
  - **Now:** a module load still running after 2 s says, once per process, on stderr: `gpu: compiling the GPU
    kernels for this card and driver; this happens once per card and driver, later loads take them from the
    driver's cache`. A warm load ends first and says nothing. stdout and receipts are unchanged.
  - **Checks:** `test-slow-note` checks the note itself on every host: said once while a slow wait runs, never
    for a quick one.

- **`RUNNER_PREFETCH` stays opt-in on Linux and Windows, now measured there (R3.39.5).** On Windows with CUDA,
  turning it on made every load slower, with identical output: 1.159x the time from a cold file, 1.122x and 1.118x
  warm on two files. On Linux, on a host capped at 60 read IOPS, it was 1.229x on the CPU and 1.006x and 0.9995x on
  CUDA. On a CUDA load the hint is issued after the device upload has read the weights, so it cannot speed that
  upload. `docs/speed-defaults.md` and the manual record the rows; a run that does not set it is unchanged.
- **`--verify` replays a served turn under a reasoning budget or the loop guard (R1.1.3.3).**
  - **Before:** such a receipt was `UNVERIFIABLE`. The record named the budget's size and that the guard was on,
    but not the sentence the forced close writes or the guard's settings.
  - **Now:** the `reasoning_budget` constraint carries `message_tokens`, the token ids of that sentence (`[]` for a
    bare close). The `loop_guard` constraint carries `span`, `repeats`, `window`, `max_closes` and `everywhere`.
    The replay installs them as the server does, primes from the prompt text as the server does, and closes or ends
    the turn where they say. A record whose settings were changed replays `DIVERGED`. A record from before the
    fields is `UNVERIFIABLE`, naming the one it lacks.
  - **Still refused:** the reasoning channel's own sampling (`reasoning_temperature` and its siblings).

- **`--verify` replays a served turn cut by a stop sequence (R1.1.3.2).**
  - **Before:** a receipt shaped by `stop`, `stop_sequences` or `stop_token_ids` was `UNVERIFIABLE`, although the
    record already carried the sequences.
  - **Now:** the replay re-does the cut. It matches the stop strings on the visible text only, after the
    thinking-tag splitter, as the server does, and gives the engine the stop ids. A record cut anywhere else
    replays `DIVERGED`, because the cut is not taken from the record.
  - **New receipt field:** `serve.reasoning_open` says whether the prompt left the turn inside the reasoning block,
    so the splitter starts where the server's did. On a model with thinking tags, a stop-cut chat record from
    before the field is `UNVERIFIABLE`, naming it.
  - **Found beside it:** on a model with thinking tags, a served chat turn under a JSON schema or JSON mode
    replayed `DIVERGED` at its first token or byte. The server forwards the reasoning bytes beside the document
    and, when the prompt opened the reasoning block, holds the grammar until that block closes; the replay did
    neither. It now starts where the server started.

- **Fix (CUDA): the F32 and F16 canonical-order matvecs finished an odd row fused, against the host (R12.1b).**
  - **The bug:** `k_mvcanon_f32` and `k_mvcanon_f16` finish a row that is not a whole 32 (F32) or 16 (F16) stride on
    one lane, with `s += w * x`. nvcc fused it into an `fma` (its default `--fmad=true`). The canonical host rounds the
    product and the sum apart (`-ffp-contract=off`), so on such rows the device differed from the host it is defined
    to equal.
  - **The fix:** the tails are spelled `__fmul_rn` and `__fadd_rn`.
  - **Checks:** two fixtures with a 200-wide FFN, `test-oddff.gguf` (F32) and `test-oddff-f16.gguf` (F16). Before the
    fix, 13 to 19 of ffn_down's 64 rows differed; after it, none. `make-test-model.py` gains `--quant f16` for the
    second fixture, since the runner's quantizer writes no type for a row that is not a whole 32-block.

- **CUDA: the canonical-order matvec gains Q4_0, Q4_K, Q5_K and Q6_K, bit-identical to the host (R12.1b).**
  - **What it is:** `gpu_mvcanon` is the device forward matvec of the training path (R8.7.1). It computes the host's
    canonical association, so an adapter trained through it keeps the CPU's bytes. It took F32, F16 and Q8_0; every
    projection of a Q4_K_M model now has a kernel.
  - **How it holds:** every product that meets an addition outside an explicit `fmaf` is spelled `__fmul_rn` or
    `__fadd_rn`, so nvcc's default contraction cannot fuse what the CPU rounds twice.
  - **Checks:** `test-mvcanon` compares every row with memcmp and now walks every layer's projections and the head.
    It runs one fixture per type: the new `test-q40.gguf` and `test-q5k.gguf`, beside `test-q4k.gguf` and
    `test-q6k.gguf`. A type a fixture is named for must reach the kernel, or the check fails.
  - **Fix to the check itself:** its host reference compiled `quants.c` with `-ffast-math`, against its own comment.
    Under it, gcc reassociates the host Q4_K kernel, so the reference stopped being the canonical order. It now takes
    the engine's strict flags, as `test-canon-kernels` does.

- **The T3 (canonical) build gains Q4_0, Q4_K, Q5_K and Q6_K, and names the types it cannot run canonically
  (R12.1b).**
  - **The kernels:** each type has one reduction tree on every ISA, the AVX2 kernel's own shape (the reference, as
    for Q8_0). There are NEON and scalar mirrors.
    - Q4_0 is Q8_0's tree on nibbles.
    - Q4_K and Q5_K: four-term lane sums per 32-weight sub-block, scaled into eight lane accumulators, with the min
      term summed apart from per-sub-block tree sums.
    - Q6_K: two-term lane sums per 16 weights.
  - **What this covers:** a Q4_K_M model (Q4_K, Q5_K, Q6_K, Q8_0 and F32 on unsloth's Qwen3.5-4B) now runs every dot
    product in the canonical order.
  - **The cost on x86-64:** none at the kernel against the same-shape kernel. The canonical build gives up the decode
    path's reuse of the per-matvec x sums: Q4_K 1.47x and Q5_K 1.34x a matvec. End to end, T3 decodes Qwen3.5-4B
    Q4_K_M 6 to 14% slower than the default build, its usual strict-float cost.
  - **Checks:** `tests/test_canon_kernels.c` holds each kernel bit for bit to an independent definition. That is the
    AVX2 kernel on x86, and the scalar one with RUNNER_NO_SIMD; NEON in macOS CI.
  - **The gap that was silent:** every type without a canonical kernel fell through to the ISA-native kernels without
    a word, so a T3 build was not the same on arm64 and x86-64 for that model and nothing said so.
  - **Now:** the load says "canon: BF16 have no canonical-order kernel in this build ...", and the route line and the
    receipt's `profile.route` carry `canon_gaps` (empty when every type is canonical). A default build writes none of
    it.
  - **Checks:** `tests/test_canon_gaps.py`, in CI's t3 job.

- **Fix: `--fit` charged deepseek2's KV at the header's shape, up to 10.7x the cache the loader builds (R3.48).**
  - **What the loader builds:** its default is the absorbed MLA form. That is one cache row a token and layer,
    `kv_lora_rank + rope` wide, which every head and V read. Q is that wide per head.
  - **What the fit charged:** `head_count_kv x key_length`, K and V each, which is the naive form
    (`RUNNER_MLA_NAIVE=1`).
  - **The cost on DeepSeek-V2-Lite-Chat Q6_K at ctx 131072:** the fit read 40.50 GiB of KV against the loader's 3.80.
    On a 24 GB slice it predicted 4 to 8 layers on the GPU at `--reserve-vram` 40 to 70, where the loader places 14 to
    25.
  - **Now:** the fit takes the loader's MLA geometry, absorbed unless `RUNNER_MLA_NAIVE` is set.
    - Its KV equals the loader's in f16, q8, k8v4 and fp4 (k8v4 keeps a V row of its own).
    - Its split equals the loader's at 8 points across ctx 32k and 131k.
    - Its fixed charge is byte-exact, on V2-Lite and on the fixture in both forms.
  - **Checks:**
    - `tests/test_fit_check.py`: the fit and the loader against the fixture's declared absorbed and naive shapes, at
      the default and V2-Lite attention geometry.
    - `tests/test_cuda_fit_hybrid.py`: deepseek2's fixed charge in both forms.

- **Fix: a hybrid's recurrent state was held for every layer, including the layers that have none (R3.47).**
  - **The bug:** qwen35's Gated DeltaNet fold, Mamba-2's conv window and SSM state (granitehybrid, nemotron_h) and
    lfm2's conv ring sat in per-layer buffers sized for every layer, attention and MLP-only layers included. That held
    on the host, on CUDA and on Metal, and in every prefix-cache entry, session image and KV snapshot.
  - **Now:** each layer that holds state has a slot (`model_rec_slot`), and nothing else has one. Greedy text and
    teacher-forced scores are byte-identical to the previous build on Qwen3.5-4B (CPU, CUDA), granite-4.0-h-micro
    (CPU, CUDA) and Qwen3.6-35B-A3B (CUDA).
  - **Per sequence, at a fixed context:**

    | Model | Host state | Device instance |
    |---|---|---|
    | Qwen3.5-4B | 70.3 to 52.7 MB (24 of 32 layers) | 35 MB less |
    | Qwen3.6-35B-A3B | 87.8 to 65.9 MB (30 of 40) | 44 MB less |
    | granite-4.0-h-micro | 86.0 to 77.4 MB (36 of 40) | 17 MB less |

    Every prefix-cache entry shrinks by the host figure.
  - **`-v`** prints the state beside the KV cache: "recurrent state 52.7 MB (24 of 32 layers)".
  - **Saved state:** 1.15.0's layout (a slot for every layer, and R3.44's KV rows) is refused by name, never misread.
    `RUNNER_KV_RECURRENT_ROWS=1` keeps that whole layout and resumes it. State written by development builds between
    R3.44 and this fix is refused too.
  - **The plan and `--fit`** charge the stateful layers only (R3.46's `model_gpu_recurrent_cost`).
  - **`--fit` reports the CPU for the families the loader never puts on a GPU:** lfm2, lfm2moe, flint, LayerNorm
    models, grouped-norm k2-horizon, and deepseek2 off CUDA. It used to predict a device split the load never made.
    `model_gpu_cpu_only_why` now answers for both.
  - **Checks:**
    - `tests/test_cuda_rec_slots.py` (CUDA): scoring stays on the device under the new layout and matches the old
      one's scores, on seven hybrid fixtures.
    - `tests/test_qwen35_kv_rows.py`: the state shrinks by exactly the stateless layers' slots, and an old-layout image
      is refused by name on qwen35 and Mamba-2.
    - The hybrid fixtures run clean under ASan and UBSan in both layouts.

- **Fix: `--fit`'s GPU split on hybrids now equals the loader's, and the CUDA plan stops charging qwen35's recurrent
  state twice (R3.46).**
  - **What the fit left out:** each hybrid's device state, held twice (live, and for the speculative rollback), plus
    its mixer scratch. That is qwen35's Gated DeltaNet fold, and Mamba-2's conv window and SSM state.
  - **The FFN scratch width:** it took 4 x n_embd where the loader takes the expert width (Qwen3.6-35B-A3B declares no
    dense one) or a per-layer array's widest entry.
  - **The cost:** Qwen3.6-35B-A3B at ctx 131072 read 40/40 layers on the fit against the loader's 39/40, and
    granite-4.0-h-micro read 3 to 4 layers high.
  - **Now:** both charge `model_gpu_recurrent_cost`.
    - The fit equals the loader at every point measured: the 35B at 32k, 131k and 262k, and granite-micro across
      `--reserve-vram` 5 to 10 at 32k and 131k.
    - The fixed charge is byte-exact on qwen35, qwen35moe, clef, qwen4exp, granitehybrid and nemotron_h fixtures and on
      the 35B.
  - **The loader:** it charged qwen35's fold and scratch once in `cuda_instance_dims` and again in `shared_plan_budget`.
    That is 4x the fold it allocates: 0.18 GB on the 35B, 0.32 GB on Qwen3.8-Flash-Next. Now charged once.
    - **On the 35B** no layer or bank moves at 32k to 262k.
    - **Under `--cpu-moe auto` at 262k** the prompt window widens from 256 to 512 columns. Prefill of a 7,370-token
      prompt goes from 147 to 175 tok/s on main to 183 to 189 tok/s, with identical text.
    - **The 14 qwen35-family roster rows with CUDA pins** keep their splits (checked from the headers, and Flash-Next
      on the loader), so no pin moves.
  - **`RUNNER_DEBUG_VRAM`** prints both figures in bytes: `fit: (bytes: fixed F, headroom H)` and `gpu-split: (bytes:
    ...)`.
  - **Checks:** `tests/test_cuda_fit_hybrid.py` (CUDA) holds the two charges equal on six hybrid fixtures and a dense
    control.

- **The shared prefix cache's default budget is sized from the served model (R4.26.13).**
  - **Before:** `RUNNER_PREFIX_CACHE_MB` was a fixed 512 MB. An entry over half the budget is not stored whole: a
    recurrent model stores nothing, a dense one a truncated prefix. Even after R3.44, Qwen3.5-4B could not cache a
    prompt past 6,048 tokens.
  - **Now, when it is unset:** each model a server loads raises the budget to hold one of its own full-context entries
    under that rule, twice the entry. It is never below 512 MB and never above an eighth of the RAM the process may
    use. The server prints the figure at startup; a set value is used as given.
  - **Qwen3.5-4B Q4_K_M, `-c 16384`:** budget 1,158 MB.
    - A 7,370-token prompt is now stored, so a later request that extends it forks 7,370 cached tokens.
    - Prefill 0.14 s on CUDA against 3.46 s cold, and 0.46 s on the CPU against 111.7 s cold.
    - The text is identical to the cold run's.
    - Main stored nothing and re-prefilled.
  - **Qwen3.6-35B-A3B Q4_K_M, `-c 32768`:** budget 1,447 MB. At 7,370 tokens main already stored the entry (it fits
    under 256 MB), so nothing changes there. Past about 8,860 tokens main stored nothing. At 19,998 tokens main re-prefilled in 107.7 s; the fix forks 19,988 cached tokens in 0.40 s, with identical text.
  - **CUDA too:** a GPU-backed hybrid slot forks a shared entry through the host (R4.26.7), against what the MANUAL
    said. It is corrected.
  - **Checks:**
    - `tests/test_prefix.c` checks the sizing rule at its bounds: the floor, twice the entry, the RAM cap, saturation.
    - `tests/test_prefix_budget.py` checks the server's reported budget against a fixture's declared entry cost and
      the RAM `--caps` reports. It also checks the floor, and that a set value wins.

- **Fix: `--fit` counted a declared NextN/MTP block as a layer of the trunk (R3.45).**
  - **The bug:** a GGUF that declares a predictor block (`nextn_predict_layers`) counts it in `block_count`. The
    loader takes it off the trunk and gives it KV rows and device memory only when `--mtp` (or `--draft mtp`) binds
    it.
  - **What `--fit` did instead:** it took `block_count` whole. So the block was a layer in the count, in the KV
    cache and in the GPU split.
  - **On Qwen3.6-35B-A3B Q4_K_M at ctx 131072:** 41 layers and 2.75 GiB of KV, against the loader's 40 layers and
    2.50 GiB (2,684.4 MB). Before R3.44's KV fix it was 10.25 GiB against 10.00.
  - **Now:** the model line names the block ("1 NextN/MTP block not charged (pass --mtp to draft from it)"). Under
    `--mtp` its rows are charged, and its tensors and rows join the GPU split's fixed charge, as the CUDA plan charges
    them.
  - **What does not change:** a file that declares no block gets the same `--fit` output as before (10 files, with
    and without a GPU).
  - **Not covered:** the default MTP drafting of one-shot JSON generations is not modeled (one measured file, CUDA,
    full offload only).
  - **Checks:** `tests/test_qwen35_kv_rows.py` holds the fit and the loader to the fixture's declared geometry, with
    and without `--mtp`.

- **Fix: Qwen3.5-class hybrids reserved KV rows for their Gated DeltaNet layers, 4x the KV they need (R3.44).**
  - **The bug:** qwen35, qwen35moe, qwen4exp (Qwen3.8-Flash-Next) and clef run three DeltaNet layers to one attention
    layer. A DeltaNet layer keeps its state in the recurrent fold and never reads or writes a KV row, but every layer
    was given rows of the attention shape: on the host, on CUDA and Metal, in the placement plan and the reservation
    auto-fit, in every prefix-cache entry and in every saved state.
  - **What does not change:** the default prefix cache still stores nothing for a long prompt. On Qwen3.5-4B a
    7,370-token prompt's entry is 312 MB, against 1,036 MB before, and an entry over half the 512 MB budget is still
    not stored. The longest prompt it caches rises from about 1,500 to about 6,000 tokens. Sizing that budget is
    R4.26.13.
  - **KV now** (f16, main against the fix, on Qwen3.5-4B and Qwen3.6-35B-A3B):

    | Model | ctx 4096 | ctx 32768 |
    |---|---|---|
    | Qwen3.5-4B | 537 to 134 MB | 4,295 to 1,074 MB |
    | Qwen3.6-35B-A3B | 336 to 84 MB | 2,684 to 671 MB |

  - **On a 24 GB CUDA slice:**
    - Qwen3.5-4B's auto-fit context (`-c 0 --reserve-vram 90`) rises from 151,222 to 262,144, the training context,
      with 12.4 GB still free.
    - Qwen3.6-35B-A3B under `--cpu-moe auto`, expert banks and prompt window:

      | Context | Banks | Window (columns) |
      |---|---|---|
      | 32,768 | 40 to 40 | 256 to 512 |
      | 131,072 | 23 to 39 | 64 to 512 |
      | 262,144 | 0 to 33 | 64 to 256 |

    - Decode at 131,072 goes from 61 to 73 tok/s to 84 to 87 tok/s (2 interleaved pairs on shared CPUs).
  - **Bytes identical:** greedy text and teacher-forced scores match the previous build's byte for byte, on the CPU
    and on CUDA (Qwen3.5-4B, and Qwen3.6-35B-A3B on CUDA).
  - **The fix:** those layers get 0 KV heads, the convention granitehybrid and lfm2 already use. The NextN/MTP block
    keeps the attention shape.
  - **Saved state:** a session image or KV snapshot written before this fix holds the old layout.
    - It is refused by name, never misread. A session image is told "an earlier runner wrote it". A KV snapshot gets
      409 `snapshot_kv_layout`.
    - `RUNNER_KV_RECURRENT_ROWS=1` keeps the old layout and resumes them.
  - **Checks:** `tests/test_qwen35_kv_rows.py` covers the four arches:
    - text and scores identical under either layout;
    - the state shrinks by exactly the DeltaNet layers' rows;
    - an old-layout image and snapshot are refused by name, and resume under the switch to the straight run's
      bytes.

- **Fix: CUDA `--cpu-moe auto` over-planned the device's weights, leaving expert banks and prompt-window columns on
  the table (R4.26.18).**
  - **What the plan charged:** every tensor as its own allocation rounded to the driver's 2 MiB, plus a nine-granule
    allowance per expert bank.
  - **What the upload does:** it joins the device tensors into runs and binds one allocation a run, so it rounds once
    a run.
  - **The cost on Qwen3-30B-A3B Q6_K** (25.1 GB) on a 24 GB slice: the weights were planned 1.07 GB over what the
    upload requested, in 6 bindings. Auto placed 43 of 48 expert banks, and the prompt window got 192 columns.
  - **The fix:** after the greedy placement, the plan re-costs the weights by the runs that placement makes, then
    keeps placing the next bank while that exact figure fits.
  - **The result:** 45 banks and a 512-column window. The weights are planned within 16 MB of what the driver spent
    (the constants and tables uploaded beside them), and the instance buffers within 12 MB.
  - **The same run, 3 interleaved pairs:** prefill of a 4,252-token prompt 315 to 541 tok/s, mostly the window.
    Decode 70.6 to 73.4 against 75.3 to 76.7 tok/s.
  - A retry after a failed init keeps the greedy plan.
  - **`RUNNER_DEBUG_VRAM`** prints the plan per category (`vram-plan:` headroom and instance, KV, weights greedy and by
    runs, banks, head). The upload line gives bytes requested and as the driver rounds them.
  - **Checks:** `tests/test_cuda_plan_runs.py` (CUDA): on three MoE fixtures, the plan's weights by runs equal the
    upload's rounded bytes, up to the head's granule.
- **The route line and records say what the tensor cores ran, and `--verify` holds a replay to it (R3.39.2).**
  - **The old label:** with `RUNNER_CUDA_TC` unset, the CUDA route detail read `tc=promoted` for every non-MoE
    architecture. So an unpromoted model's line, and its records, claimed a tensor-core prefill that never ran.
  - **The new label** is `tc=taken` where the default takes the tensor cores for one of the model's weight matrices,
    and `tc=off` where it does not. `tc=forced` and `moe-tc` are unchanged.
  - **Records** list the weight types the route takes on the tensor cores in `profile.route.tc_types`, and `--verify`
    compares them. A later promotion then reads as a route change, not only in the token replay.
  - **Records written before this,** which carry the word `promoted` and no `tc_types`, still verify when nothing else
    in their route differs. The verdict reads "tensor-core types not recorded (a record from before tc_types)".
  - **Checks:**
    - `tests/test_tc_route_label.py` (CUDA): the label agrees with the receipt's tensor-core GEMM counts on four
      fixtures.
    - `tests/test_verify_tc_compat.py` (CUDA): a record written the pre-fix way verifies with the note, and still fails
      when another route token differs; recorded `tc_types` hold the replay; `RUNNER_CUDA_TC=0` fails a `taken`
      record.
    - Genuine records from main 241e9d7 (a llama and a deepseek2 fixture) verify on this binary the same way.

- **CUDA: tensor-core prefill on spark2_5 for Q4_K and Q6_K, measured (R3.39.2).** XHToken's Spark-X2.5-4B Q4_K_M
  prefills 9.48x faster (203 to 1,927 tok/s at 1,635 tokens, 5 of 5 pairs) and reads the same against
  XHToken/Spark-X2.5-4B in float32 on 2,000 positions: mean KLD ratio 0.9995 (CI [0.9981, 1.0007], Wilcoxon p 0.115),
  margin-qualified top-1 95.01% against 94.95%. The promotion covers exactly the two types the file exercised.
  `RUNNER_CUDA_TC=0` still holds the default route.
  - **clef, for Q4_K and Q8_0:** ggml-org's Clef-Flash Q4_K_M prefills 15.4x faster (102 to 1,566 tok/s) and reads
    within the documented obligation (docs/kernel-routes.md: mean KL within 1.01x, Wilcoxon and McNemar not worse at
    p < 0.05) against Cloudflare/clef-flash in float32: mean KLD ratio 1.0001 (CI [0.9991, 1.0011], Wilcoxon p
    0.452), margin-qualified top-1 95.13% against 94.99%. `tests/test_tc_clef_promotion.py` checks the dispatch.
  - **Not promoted, and why:**
    - **k2-horizon and stablelm** run on the CPU on a CUDA build (grouped RMSNorm, LayerNorm), so they have no device
      route to promote.
  - **A CUDA record of a spark2_5 Q4_K or Q6_K model made by an earlier binary** replays on this one through the
    tensor cores, as after any promotion. Its route label reads the same, so the token replay is what tells the two
    apart.
  - **Checks:** `tests/test_tc_spark_promotion.py` (CUDA), on the receipt's tensor-core GEMM counts:
    - a wide spark2_5 fixture's Q4_K takes the tensor cores by default and its Q5_0 rows do not;
    - `RUNNER_CUDA_TC=0` takes none;
    - a narrow fixture with only Q5_0 takes none.

- **Fix: `runner recommend` lists a file that sits in a folder of its repository (R10.28).** #690 left such files out
  on the belief that `-hf REPO:STEM` names only top-level files. It does not: `-hf` matches the base name and fetches
  the listing's full path (`resolve/main/merged/merged-Q8_0.gguf`). The skip came from Dojo's picker, which built the
  URL from the bare name and got a 404; Dojo now fetches by the roster's `path`. The Qwen3-4B tool-use merged Q8_0
  (3.99 GiB) is listed at 8 GiB again.
  - **Checks:** `tests/test_hf_fetch.py` fetches a folder file from a fake hub shaped like that repository: the
    URL keeps the folder, and the short tag that also matches a `study/` adapter is refused with both named.
    `tests/test_recommend.py`'s anchor is now the 16 files Dojo's fixed picker listed on an 8 GiB machine.

- **`runner recommend` (`--info recommend`): the listed model files that fit this machine (R10.25).** Largest resident
  weights first, each with the `-hf` spelling to run it, its fidelity line (`measured` with the first sentence of its
  numerical anchor, or `unmeasured` with why) and its entry in docs/model-support.md. It makes no network request:
  the list comes from the roster built into the binary (`src/model_roster.h`, rendered from docs/model-tiers.json by
  `scripts/model-tiers.py --write-docs` and held fresh by `--docs-check`). `--roster FILE` reads another copy,
  `--memory GIB` plans for another machine and `--ram-share F` (default 0.5) moves the bar.
  - The rule is the Dojo model picker's: a listed text model in one `.gguf`, no larger than the RAM, whose
    resident weights fit within the share. Resident means the whole file, or the `--info fit` hot set for a file
    whose per-layer token table is read by row. That hot set is the new `files[].resident_bytes`, recorded from
    `runner --fit` by `scripts/model-tiers.py --record-resident` (Nemotron-3.5-Bolt-E4B and the gemma-4-E2B mix).
  - **Checks:** `tests/test_recommend.py`: on a fixture roster a memory cap and the share change the list, and the
    excluded shapes stay out. Absolute anchor: at 8 GiB the committed roster gives the files Dojo's own run listed (16 since R10.28)
    on an 8 GiB machine. `tests/test_model_tiers.py` covers the recorder, the field's validation and the snapshot.
- **`runner train --full`: full-weight training of an F32 file into a variant (R8.25.2.2).** Every dense projection
  (attn q/k/v/o, ffn gate/up/down) gets an f32 master with AdamW state, in place of an adapter. `--out` is a variant:
  the -m file with the trained tensors replaced, every other byte unchanged. The parent stays untouched. The same data
  and seed give a byte-identical variant, and the training record names the base's and the data's sha256, the trained
  tensors (`full_weight`) and the `variant`.
  - **Training through a Q8_0 copy was tried and dropped.** The design read a Q8_0 copy of each master in the forward
    and the backward, so the shipped Q8_0 would be what the last step read. On Qwen3-0.6B, the 10-28 tool-use recipe,
    316 steps, both routes scored 29/29 held-out exact calls, with 0 discordant prompts:
    - trained through Q8_0;
    - trained in float, then `--rewrite --quant q8_0`.
  - **What that does and does not show.** The held-out set saturated, so this is no measurable gain at Q8 on this
    task, not a proof that the two are equal.
    - Beside the gate: the teacher-forced NLL was 0.00003 against 0.00006 nats per token. The KLD to each file's own
      float model was 0.00419 against 0.00425.
    - A quantized -m file is refused: `is not f32: full-weight training reads an F32 file (train the float file, then
      quantize the variant with --rewrite --quant)`.
  - **`--kv-f32` (`--train-kv-f32`): an f32 KV cache for a training run, on the CPU only.** On the default f16 cache,
    the full-weight gradient matched PyTorch fp32 autograd only to 2.7e-3 relative L2, against a 1e-3 bar, because K
    (after RoPE) and V are rounded to f16. With an f32 cache it matches to 6.9e-6 over all 196 matrices of Qwen3-0.6B,
    and 60 steps' losses agree with PyTorch's. Inference keeps f16, and the record writes `"kv_cache":"f32"`.
  - **The gradient check on the fixture:**
    - **On the f16 cache:** directional derivatives at a fixed eps of 1e-3 missed the first declared 1e-3 (1.17e-3
      and 4.73e-3), because the cache's rounding makes the finite difference a staircase.
      - `tests/test_full_grad.c` grades the f16 path by tests/test_lora_grad.c's method instead: along sign(G), the
        best of six steps within 1%.
      - On the f32 cache it grades the fixed-eps derivatives as first declared.
    - **Also pinned there:**
      - the gradient is byte-identical across two backwards;
      - the forward reads the master after a step;
      - a Q8_0 file is refused.
  - **Not covered yet:** `--train-gpu`, DPO, a teacher, checkpoints (`--save-every`, `--train-resume`), kill rules,
    `--train-eval`, a stop file, `--train-logprobs`, recurrent (DeltaNet) layers and MoE expert banks are refused with
    `--full`.
  - **Checks:** `tests/test_full_grad.c` (in `make test-gates-core`) and `tests/test_full_train.py`.

- **Speculative decoding defaults per workload (R3.39.4).** With no `--draft`, two measured cases now draft without a
  flag, both only on CUDA with every layer on the device. The lookup and the NextN head emit plain decoding's tokens,
  so only speed changes.
  - A server request with `draft_hints` takes `lookup` at width 7, measured at 2.86x plain on an in-context edit.
    Such a request was refused before unless the server ran with `--draft lookup`. The lookup is that request's
    alone, and `GET /v1/capabilities` reports it as `draft.default`.
  - A one-shot CLI run with `--json` or `--json-schema` on Qwen3.5-4B Q4_K_M with its NextN block takes `mtp` at
    width 2, measured at 1.13x plain on a tool call. If binding the head would cost the run its full offload,
    Runner reloads without it.
  - Everything else decodes plain: prose (the head reads 0.88x there, and the lookup has no prose row), chat,
    `tool_choice: "auto"`, other NextN models, CPU and Metal. The server takes no per-request head: a head fed on
    every request would slow prose, and one fed on demand would have no state under a reused prefix.
  - `--draft off` turns the defaults off. The startup line names a default and its scope, and the route line marks
    it `draft=lookup:default` or `draft=mtp:default` (a named source shows as `draft=lookup`).
  - `docs/speed-defaults.md` lists every speed switch with its class and its default (R3.39.1).
  - **Checks:** `tests/test_lookup_draft.c` pins the decision table against the measured rows, prose plain on both
    surfaces among them. `tests/test_draft_defaults.py` checks that a request without the signal decodes plain on
    the CLI and the server, before and after a hinted request; that `--draft off` refuses hints and a second source;
    and that a `--json` record replays under `--verify`. Under the test-only `RUNNER_TEST_DRAFT_SCOPE=1` both
    defaults fire on fixtures, and their records (a CLI head run, a served hinted request after a plain one on the
    same slot) verify with the switch unset.
  - **A record replays the draft source it ran.** It names the source, with `"default": true` when the default
    chose it, and `--verify` binds that source instead of decoding plain. The tokens were always the same, but on
    CUDA the head's walk takes no decode graph, so a plain replay of a head record failed the route check (found on
    the CUDA gate with the measured file). A record whose route cannot draft from its source replays plain, with a
    note. A `model` record still needs `--draft PATH`.
- **Grammar fast-forward is identity-class and remains opt-in (R3.39.5).**
  Its constrained target verification preserves token IDs and bytes. A bounded
  SmolLM2-135M Q8_0 check found a CPU gain on one schema and large regressions
  on another, so there is no default promotion. The grammar gate now initializes
  the fp16 lookup table, checks it against IEEE 754 binary16 1.0, and compares
  token IDs and sampler RNG as well as output bytes on greedy and seeded
  constrained runs. Unconstrained prose stays on the plain path.
- **Fix: LoRA training with `--kv k8v4` or `--kv fp4` is refused, where it failed at step 1.** The training backward reads
  K and V rows as f16. `--kv q8` was refused, but k8v4 and fp4 reached the backward, and their blocks were read as f16.
  On Qwen3-0.6B both runs stopped at step 1 with "the loss or a gradient is not finite", caught only because the
  garbage was not finite. Both are now refused by name, as q8 is: `training does not cover this model yet: k8v4 KV
  cache (use --kv f16)`, and likewise for fp4.
  - The layer backward's attention recompute now passes the model's cache kinds instead of a literal f16. These are f16
    in every case training admits, so the bytes are unchanged.
  - **Checks:** `tests/test_train_kv_refusal.py`, on a wide-head fixture where each cache type is really taken.

- **`runner derive --target-size` with `--imatrix` plans a whole file (R4.34.2).** llama.cpp's imatrix has no entry for
  `output` or `token_embd` unless it was run with `--process-output`, so those two tensors' errors were unweighted
  beside the weighted rest, and the planner refused the mix. A candidate the matrix does not cover is now held at the
  top rung and left out of the ranking, and the stderr, the record (`held`, `held_at`) and the card name it.
  - Holding them is a default, not evidence that they matter most.
  - `scripts/type-plan-by-sensitivity.py` holds the same tensors, and `--allow-mixed` still ranks them together.
  - **Checks:** `tests/test_derive.py`: an imatrix skewed onto the tensor the plain plan demotes first keeps it up
    and changes the plan; an imatrix that leaves one tensor out gives a plan with that tensor held and named, equal
    to the reference planner's.
- **Test fix: the FP8 gold-logits test skips without `accelerate`.** transformers
  loads an FP8 checkpoint only with `accelerate` installed, and the test guarded
  only `torch` and `transformers`, so a machine with transformers but without
  accelerate failed it with an ImportError instead of skipping, as the
  neighbouring tests already do.

- **`runner derive --target-size` gains a middle rung: `--mid T`, or `--ladder T1,...,Tn` high to low (R4.34.1).** A
  group is demoted one rung at a time instead of straight to the lowest type. Over the ladder, each group's (bytes,
  error) points are reduced to their lower convex hull, and the cheapest next step over all groups (added error per
  byte saved) is taken until the file fits.
  - A rung that costs more per byte than going past it is skipped for that group. The record names each skip, and
    names where each demoted group ended (`ladder`, `demoted_to`, `skipped`).
  - Two rungs (`--high`, `--low`) plan exactly as before.
  - `scripts/type-plan-by-sensitivity.py` takes the same `--ladder` and `--mid`, and the test holds the two to the
    same plan.

- **Fix: Metal tiled receipts identify actual arithmetic and refuse unsupported replay (R3.39.3a).**
  A successful tiled generation step marks `profile.route.micro` as `tiles`;
  the configured switch alone no longer marks an identity-only receipt.
  `--verify` refuses a tiled record without a replayable decode schedule as
  `UNVERIFIABLE` before loading the model. The default is unchanged: tiles
  remain opt-in, and 2-5 columns keep the identity route.

- **Importance-weighted Q4_K, Q5_K, Q6_K, Q5_1 and IQ4_NL writers (R4.34.3).**
  `--imatrix` now shapes these formats as well as Q3_K and Q4_0, using ggml's
  type-specific weighting and scale searches. All five are byte twins of
  ggml `c1d0e7a` on the committed real tensor slice and deterministic boundary
  rows. The `.dat` and GGUF calibration paths agree, all-zero calibration
  retains the reference bytes, and mismatched calibration shapes are refused.
  The no-imatrix bytes and planner policy are unchanged.

- **Docs: the bartowski Qwen3.5-9B distill row no longer says the `qwen35` path
  has no Metal backend.** It has run on Metal since v1.10.0 when the model fits
  on the device whole; this 5.84 GB file does not on an 8 GB Mac, which is why
  the row has no Metal cell.

## v1.15.0 - 2026-11-07

- **Fix: transient VRAM accounting no longer forces durable writes (R6.24).**
  The ledger and yield sentinels describe live processes, so power-loss
  durability adds no useful guarantee. POSIX `fsync` and Windows
  `FlushFileBuffers` no longer stall every ledger user under the shared lock.
  Locks and the complete-write-before-truncate order are unchanged; readers
  retain valid rows, ignore malformed fragments and reap stale owners.
- **The DeepSeek-V2-Lite-Chat Q8_0 row's pins sit on a near tie, and v1.14.0's R3.32 note is corrected (R6.23).**
  Docs only; the pins are unchanged.
  - **Why the pins mismatched:** an overnight compare against the pins mismatched on both backends, but its builds
    were clang's. The pins come from the Makefile's default compiler, gcc with `-march=native`, and main's gcc build
    matches both.
  - **The tie:** every build is deterministic. The reply is decided at prompt position 6, layer 7, where the sixth
    routed expert is 39 or 49 with gates within 2e-6. The two texts that can result both contain "Mercury".
  - **The release build:** gcc with `-march=x86-64-v3` gives the CUDA pin's text on both backends, so its CPU reply
    differs from the CPU pin. The row's note in `docs/model-tiers.json` says so.
  - **The v1.14.0 correction:** the entry for R3.32 said both deepseek2 CUDA cells were stale from #631. Only the
    Q6_K's was. R3.32's prefill route moved the Q8_0's and re-pinned it in the same change.

- **The model tiers are retired: `docs/model-support.md` is now one list.** Families, models and files sort
  alphabetically, and opening a family shows one table, with each model's name on a row above its files. Validation runs
  when the project chooses, not with every release, so `scripts/check-release.py` no longer refuses a tag on a row's
  record. Upcoming stays apart: it means not supported yet. The transition log keeps the earlier record.

- **Known `chat_template_kwargs` keys no longer vanish on the families that ignored them (R2.14).** Measured against 33
  references. Honoured now: `preserve_thinking` on LFM2 and LFM2-24B, `preserve_thinking: false` on Qwen 3.8 (its default
  keeps every thought), `enable_thinking` on Apertus, and `reasoning_effort: low` on Granite 4.2. Refused now, with the
  400 R2.13 added, wherever the file's template reads the key: a key a family's renderer ignores, `preserve_thinking` on
  Responses and Messages, `reasoning_effort` and `reasoning_strength` on Messages, and `add_generation_prompt: false`.
  Spark refuses `reasoning_effort: xhigh`, as its template does. Limits: 19 families were not compared with their
  references on this host, and are covered by the refusal rather than a measurement. A null `preserve_thinking` counts
  as absent, where Qwen 3.8's template would treat it as false.

- **A `chat_template_kwargs` switch Runner cannot render is refused, not silently dropped (R2.13).** Sage-A1.5's
  `verification` rendered as off with no error, because Runner renders each family with its own code and reads only
  the keys it knows. A key the file's own template reads as a variable, and Runner does not render, now gets a 400 naming
  it (`unsupported_template_kwarg`). A key the template does not read is ignored, as the reference ignores it. Limits:
  the check is by name, so `false` is refused too, and it is a scan of the tags rather than a Jinja parse, which counts
  anything doubtful as not read. `enable_thinking`, `preserve_thinking`, `reasoning_effort`, `reasoning_strength`,
  `bare_recipients` and the values a render supplies itself are unaffected.
- **A usage error ends with the error (R10.19).** An unknown option, a flag with no value, no model and no prompt each
  printed the whole usage, 382 lines, so a script or a person reading the end of the output never saw why: one export
  script read `unknown option --out` as a refusal of the quantization. They now print the error line and
  `see runner --help`, and the full usage is for `--help`. A bare `runner`, with no arguments or only `--no-tray`, still
  prints the usage, for someone running it for the first time, and now ends with
  `error: need -m MODEL or -hf REPO[:TAG]`. Exit codes are unchanged.
  - **Checks:** the whole of stderr for each case, in `tests/test_cli_surface.py`.
  - **Limits:** `runner attest` and `runner train` with nothing after them still print their own short usage. A script
    that looked for `usage:` on stderr after a failure other than a bare run no longer finds it; `unknown option` now reads
    `error: unknown option`.
- **Sage-A1.5 joins the roster on InternScience's own Q6_K, and occamy-1.0 is listed Upcoming (R4.27.7).**
  Both are qwen35moe derivatives of Qwen3.6-35B-A3B, each read against its own BF16 parent computed in float32 (bars
  0.05 mean KLD and 97% margin-qualified top-1, 300 positions, CPU).
  - **Sage-A1.5:** InternScience's model, built by its publisher on Qwen3.6-35B-A3B and InternScience's Agents-A1.
    - Q6_K: 0.0395 mean KLD and 97.9% margin-qualified, passes. The Q4_K_M misses (0.145).
    - Tokenizer 0 of 721 and 0 of 247,091 against its tokenizer.json (the qwen35 split).
    - Template conformance 25 of 25 in text and tokens.
    - CPU and CUDA cells pinned and matched; cpu_cuda 9 of 9 exact with routing pinned.
    - Limits: the template's `verification` switch is not served (a request with it does not get the trigger), and the
      GGUF names itself "Agents A1.5". Record: `docs/compat-reports/sage-a1.5-2026-11-06/`.
  - **occamy-1.0:** Accio-Lab's finetune. No publisher file passes: Q8_0 0.0590 / 98.7%, Q6_K 0.0836 / 95.4%, Q5_K_M
    0.0970 / 95.8%, Q4_K_M 0.258 / 85.7%. Tokenizer (the qwen2 split its header labels) and template (25 of 25) pass.
    Where the distance comes from is not yet read. Record: `docs/compat-reports/occamy-1.0-2026-11-06/`.
- **Fix: instance cleanup no longer delays server startup (R10.23).**
  Registration and discovery lists request a separate cleanup process instead
  of deleting stale records on the caller's path. One helper sweeps a folder
  at a time; it can finish after the server exits. Dead records are filtered
  immediately and eventually removed as storage progresses, including on
  hosts without a tray. Cleanup launch failures never fail a server start.

- **Windows: a second server on a port in use is refused, not silently sharing it (R10.22).** `SO_REUSEADDR` let both
  bind on Windows. The second printed `server listening` and got no request, and when the first stopped, its clients'
  connections were reset. It now exits with `cannot bind 127.0.0.1:PORT: is another server on this port?`, as Linux
  already refused. This was the cause of a rare `WinError 10054` under `pytest -n`, when two workers drew the same port;
  the test harness now takes another port on a lost bind. Linux and macOS keep `SO_REUSEADDR`, and a restart straight
  after serving still binds. Measured on Windows.

- **Fix: the project's Clef IQ3_S backbone tokenizes as Cloudflare/clef's own tokenizer, and its row no longer
  claims the decision model (R4.36).** `Clef-backbone-IQ3_S.gguf` is Clef's backbone in Runner's qwen35 view, so it
  declares architecture qwen35 and the Clef rule (architecture clef) missed it: it read with the qwen35 split, 3 of 721
  corpus strings and 5,503 of 247,091 vocabulary entries off the publisher's. A rule entry keyed on its header
  (qwen35 and general.name "Clef Backbone View") reads it with the qwen2 split; 0 and 0 now. Its row,
  `cloudflare-clef-iq3_s-recovered`, claimed "a decision model served at /v1/systemone", which is the
  `cloudflare-clef-iq3_s-native` row's file: it now says the text-only backbone.
  - **Checks:** every other roster file's ids unchanged (74 rows), the backbone's alone different, on exactly the
    three combining-mark strings; the house readings' corpus tokenizes identically before and after, so its numbers
    (0.0879 / 94.0%, 0.0736 / 95.6%, 0.0757 / 95.6%) carry over; the recovery adapters were trained on the qwen2
    split's ids, which the backbone now serves.
  - Record: `docs/compat-reports/r4.36-clef-backbone-2026-11-06/`.
- **CUDA: `RUNNER_MOE_HOT` gives back the host copies of the experts it holds on the device (R3.35).** A server with
  Mellum 2.1 Q8_0's 768-slot hot file held 5.05 GB of pages the host path never reads. After warm decode on a 16 GB
  host, its working set reads 5.39 GB against 10.44 GB, and available memory 5.85 GB against 0.78 GB. Measured on one
  file on one Windows host. This is a RAM change, not a speed claim: warm decode is unchanged. Load still peaks at
  6.14 GB, and prints how many experts it gave back. Report:
  `docs/compat-reports/r3.35-hot-host-copies-2026-11-06`.
- **A client that disconnects stops its queued model load (R10.12.3.1).** A load waiting in the `--wait-for-vram` queue
  went on waiting after its client had gone, up to the 300 s default, holding the one slot of a swap server. The wait
  now stops within about 100 ms and the server carries on. It stops only if nobody is queued behind that client: a
  live client keeps the wait and its place in the VRAM queue. A client that goes after the load is admitted costs
  nothing, and the finished load stays resident for the next request. A wait that `/unload` or a shutdown stopped can
  also no longer be admitted on the same poll and load a model only to discard it.
  - **Checks:** `tests/test_queued_load_disconnect.py` runs the real wait on any host; the stop-against-freed-memory
    race is in `tests/test_vram_registry.c`.
  - **Limits:** CUDA only: CPU and Metal have no queue to wait in. A queued client that sent its request and then left
    counts as present, because its close cannot be seen behind the unread request, so the wait goes on as it did. A
    load past the queue is not interrupted. A swap that stops here has already unloaded the previous model.
- **Metal: a microbatch no longer frees the logits a slot is about to read (R3.38).** With `--parallel` on Metal, the
  first slot's scratch grew on demand when other slots decoded together. If that slot had just been fed a single token,
  as after a prefix hit, the growth freed the buffer it was about to pick its next token from, and the read could crash
  the server or pick from other data. On SmolLM2-135M the forced sequence segfaulted. The scratch is now sized once,
  when the batch is created, as the CUDA backend already does.
  - **Checks:** `tests/test_batch_identity.c` forces the sequence and holds the pointer and its contents.
  - **Limits:** the first slot's scratch for the batch width is allocated at startup instead of on first use. The live
    server interleaving was not reproduced end to end; the sequence was forced through the model API.

- **Metal gives its memory back on `/unload` (R6.17).** On Metal, each model load left memory in the GPU driver until the
  process exited: `POST /unload` said the model was gone, and on SmolLM2-135M Q8_0 the driver still held +122, +280 and
  +278 MB after three request and `/unload` cycles. It now holds +3, +2 and -5 MB. Metal's autoreleased objects were
  never drained on Runner's plain C threads, and a command buffer keeps what it used alive; every Objective-C entry point
  of the Metal backend now drains its own autorelease pool. RSS and Metal's own count never showed this memory, which
  is wired, so on an 8 GB Mac it pushed other programs into swap: one test binary took 3.8 GB on a 370 KB model, and
  now peaks at 729 MB.
  - **Checks:** `tests/test_metal_memory.py` reads the driver's own figure across four `/unload` cycles, and holds every
    entry point to its pool. Decode 128.71 against 128.50 tok/s (-0.16%), output byte-identical over 12 runs.
  - **Limits:** the weights, KV and scratch were already released; what stayed was about 120 MB a load, and it did not
    scale with the model on the two sizes measured (370 KB and 145 MB). Measured on Apple M1 only. CUDA and the CPU are
    untouched.
- **Response ids are random, and a stored response can no longer be guessed (R10.15).** `chatcmpl-`, `cmpl-`, `resp_`
  and `msg_` ids were a counter from 0: they repeated after every restart and across hosts, and anyone who could
  reach the server could read or delete a `store:true` response at `/v1/responses/resp_0`. An id is now its prefix and
  128 random bits, 32 hex digits. The log lines and receipts carry the same id. Tool-call and output-item ids
  (`call_0`, `fc_0`) keep their numbers: they name a place in one response and look nothing up.
  - **Checks:** the four APIs, a stream, a restart, and a counted id reading nothing, in
    `tests/test_responses_store.py`.
  - **Limits:** the id is still the only credential on a stored response. A client that parsed the number out of an id
    no longer can. If the OS has no random generator the server now refuses to start.
- **Fix: instance discovery no longer removes records at process exit (R10.20).**
  Filesystem removal could block after the server's shutdown watchdog had
  finished. Clean stops now leave their record for the next discovery list
  or registration to sweep. Dead owners are never listed; available process-start
  stamps guard against pid reuse. Runner and tray exits use the same policy.
- **A stop that has to be forced now lands well inside its ten seconds (R10.20).** When a drain cannot finish, the
  watchdog exited 50 ms before the bound, and a shared macOS runner ended 2 stops of 8 past it (10.007 s at worst):
  timers there run up to 90 ms late. A late timer could also lose the shutdown note, the one line that says what a
  forced stop was waiting on. The note now starts at 9 s, the exit is due at 9.5 s, and a note that starts late still
  gets its turn.
  - **Checks:** the default-budget case in `tests/conformance/test_server_lifecycle.py` now requires 9.9 s, so a margin
    this thin fails on any machine. A new case wakes the watchdog 155 ms late and requires the note.
  - **Limits:** a stuck request gets 0.5 s less before it is cut off. A clean stop is unchanged.

- **A context released while its request waited for the GPU no longer hangs the server (R3.37).** The
  `context_unavailable` 409 returned without giving the device turn back, so every later completion hung while
  `/health` still read `ok`. It took a pinned context deleted during that wait, on a GPU server with more than one
  slot. The branch now gives the turn back before it writes. Needs a CUDA device to test, so the before and after are a
  recorded RTX 3070 run, not part of `make test`. Report:
  `docs/compat-reports/r3.37-context-turn-leak-2026-11-06`.

- **A client that stops reading no longer stops the other slots (R3.36).** A speculative stream wrote to the client
  while holding the device turn, so one stalled client froze every other slot for 30 s, and a client trickling reads
  held it for as long as it liked (over 150 s measured). No blocking client write now happens under the turn, on any
  path. Limits first:
  - **New limit:** a speculative stream queues at most 4 MiB. A client that far behind is cut off, gets what was
    queued and then its surface's terminator naming `client_too_slow`, and the server logs it. Plain streams have no
    such limit.
  - A reading client's speculative generation still holds the turn for its length. That is unchanged.
  - Measured on Windows on an RTX 3070 and on macOS on an Apple M1. Linux is covered by a test on a full socket, not
    by an end-to-end read.

  A session's answer is now built under the turn and written after it, as embeddings already were. Windows did not
  stall on that write, because Winsock takes one large send whole, but it broke the rule. `make runner-turn-guard`
  builds the rule in: a client write under the turn ends the process, and the turn runs on a CPU server, so
  `tests/test_turn_write_guard.py` holds any future path to it without a GPU. Report:
  `docs/compat-reports/r3.36-no-write-under-turn-2026-11-06`.

- **MiniCPM-V-4.7-35B-A3B joins the roster on openbmb's own Q8_0, text path only (R4.27.2).** The file is
  the GGUF repository's root Q8_0 (thinking fixed off, openbmb's recommended default). Against openbmb's BF16 parent's
  language model computed in float32 it reads 0.0088 mean KLD and 100.00% margin-qualified top-1 (bars 0.05 and 97%,
  300 positions, CPU).
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 247,091 vocabulary entries against openbmb's tokenizer.json, read
      with the qwen2 split by R4.27.5's rule.
    - Template conformance under `minicpmv-fixed` against the file's own template (R4.27.6); the reasoning-history
      case accounted as a token-only deviation (R2.4.11).
    - CPU and CUDA cells pinned and matched (byte-identical replies); gpu-identity ok with routing pinned; cpu_cuda 9
      of 9 exact.
  - **Limits:**
    - openbmb's Q4_K_M (0.0594, 95.44%) misses the bar.
    - The reference is transformers' Qwen3.5-MoE text model on the checkpoint's language-model weights, as
      transformers has no class for minicpmv4_7 and the repository ships no modeling code.
    - Text only; thinking is fixed off in this file; on a 24 GB CUDA card it runs split, 26 of 40 blocks.
    - Record: `docs/compat-reports/minicpm-v-4.7-35b-a3b-2026-11-06/`.

- **MiniCPM-V 4.7's chat templates get their own renderers, text path (R4.27.6).** openbmb's GGUF repository ships
  two sets with byte-identical weights that differ only in the embedded template: the root files (thinking fixed off,
  openbmb's recommended default) and `thinking/` (the repository's chat_template.jinja). Runner read them as `qwen35`
  and `qwen35-nothink`. They now read as `minicpmv-fixed` and `minicpmv`: Qwen 3.5-0.8B's template with a thought
  block on every assistant turn (empty unless the turn carries reasoning after the last user query), and for the root
  files a generation prompt that is always closed. Content parts are joined with '\n', as Runner already joined them.
  A `minicpmv` sampling preset is openbmb's README's recommended instruct-mode setting, as published: temperature
  0.7, top_p 0.95, top_k 100, min_p 0, repetition_penalty 1.03.
  - **Checks:**
    - template-conformance `--require-tokens` against each file's own template: 56 cases conformant in text; the two
      reasoning-history cases differ only by caller-spelled `<think>` tags Runner reads as text, accounted under the
      token-only kind (R2.4.11).
    - Every other family's renders, and the template detected for every roster file, unchanged.
  - **Limits:** text path only (image and video placeholders are not rendered); no MiniCPM-V row is admitted here
    (R4.27.2 continues on this). Record: `docs/compat-reports/r4.27.6-minicpmv-template-2026-11-06/`.

- **template-conformance: a token-only deviation kind (R2.4.11).** A case can render byte-identically on both sides
  yet tokenize differently on purpose: Runner reads a control token only in the template's own bytes, so one the
  caller spelled in its content is text (tests/test_prompt_marks.c), where the reference tokenizer reads it as the
  control token. Until now the gate could only fail such a case as a broken premise. An allowlist entry of kind
  `token-only` names the case, cites tests/test_prompt_marks.c, and is accepted only when the text is byte-identical
  with no patch applied and every differing reference id is a control token the caller spelled; otherwise it is
  refused and the case still fails. `tests/test_template_conformance_token_only.py` holds the rules; with the kind
  added and no entries, every family's results are unchanged.

- **Fix: the qwen35-family templates trim message content as their references do (R2.4.10).** Qwen 3.5's, Qwen 3.6's
  and ornith's chat_template.jinja trim every message's content and its reasoning_content (`|trim`, python's
  str.strip(), Unicode whitespace included) before they frame it; Runner trimmed none of it, so a padded system, user,
  assistant or tool turn reached the model with whitespace its reference never writes. Each surface
  (chat/completions, /v1/responses, /v1/messages) now trims where it flattens a message, for the ornith, qwen35,
  qwen35-nothink and qwen36 templates only.
  - **Checks:**
    - Two padded house cases (padded-content, padded-tool-turn) on the eight qwen35-family conformance families:
      identical to each reference in text and tokens. A mutant without the trim fails exactly those cases.
    - Every other case of every registered family renders byte-identically before and after (50 families, 1,057
      renders); the template detected for every roster file is unchanged.
    - test_tool_attribution: a padded conversation renders as the plain one on all three surfaces; ChatML keeps it.
  - Record: `docs/compat-reports/r2.4.10-qwen35-trim-2026-11-06/`.

- **Qwen3.8-35B-A3B-Distill, empero-ai's finetune of Qwen3.6-35B-A3B, joins the roster on empero-ai's own
  Q6_K, text path only (R4.27.1).** It is not a Qwen release; the row replaces the Upcoming `qwen3.8-35b-a3b`, which
  named a Qwen model that Qwen has not published. Against empero-ai's BF16 parent computed in float32 the Q6_K reads
  0.0336 mean KLD and 98.65% margin-qualified top-1 (bars 0.05 and 97%, 300 positions, CPU).
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 247,091 vocabulary entries against empero-ai's tokenizer.json, read
      with the qwen2 split by R4.27.5's rule.
    - Template conformance 23 of 23 in text and tokens under `qwen36`, against the file's own template
      (Qwen3.6-35B-A3B's bytes; a new `empero-qwen38-35b-a3b-distill` family).
    - CPU and CUDA cells pinned and matched; gpu-identity ok with routing pinned; cpu_cuda pass_margin_qualified.
  - **Limits:**
    - empero-ai's Q4_K_M (0.1278) and Q5_K_M (0.0508) miss the bar; the Q4_K_M reads 2.7x the base model's Q4_K_M by
      the same method.
    - The tokenizer rule is keyed on header fields; a requant that rewrites them reads the qwen35 split.
    - On a 24 GB CUDA card the file runs split, 34 of 40 blocks on the device.
    - Record: `docs/compat-reports/empero-qwen3.8-35b-a3b-distill-2026-11-06/`.

- **Fix: two qwen35-labelled finetunes trained with the qwen2 split now tokenize as their publishers do
  (R4.27.5).** openbmb's MiniCPM-V-4.7-35B-A3B and empero-ai's Qwen3.8-35B-A3B-Distill ship GGUFs labelled qwen35,
  while the tokenizer.json each publisher saved under transformers 5.x uses the qwen2 regex (`\p{L}+`, combining
  marks split from their letters). llama.cpp's converter registers both splits' hashes as "qwen35", so no published
  label can carry the difference. Runner now keys a rule on the header fields the publishers' converters wrote, and
  a firing rule prints a note naming the rule and the fields it matched.
  - **Checks:**
    - Against each publisher's tokenizer.json: 0 of 721 corpus strings and 0 of 247,091 vocabulary entries
      (before: 3 and 5,503).
    - Every other roster file (74 rows): token ids byte-identical to the previous build on the corpus and the
      tier-smoke prompts.
    - The qwen35 family's 17 linux-x86_64-cuda pins all match, the base's two among them: the 9 CPU pins on the
      change before the 1.14.0 merge, the 8 CUDA pins after it. Ornith's CUDA pin matched on a rerun; its first
      run ended in a harness-reported error after a complete reply, under heavy outside load.
  - **Limits:**
    - The rule is a list of four sources (Clef and XiaomiMiMo's Qwen3.5 distill, already read as qwen2 and now in
      the same list, plus the two new ones). A requant that rewrites the matched fields falls outside it.
    - Clef and MiMo files now print the note too; their ids are unchanged.
    - cloudflare-clef-iq3_s-recovered's file declares architecture qwen35 and still reads with the qwen35 split.
  - Record: `docs/compat-reports/r4.27.5-qwen2-trained-rule-2026-11-06/`.

- **The project's own Mellum 2.1 rewrite joins the roster on the published file, text path only
  (R4.33.4).** `mellum2.1-12b-a2.5b-thinking-selective-q8_0-expq5_k` is Runner's `--rewrite --type-plan` of JetBrains'
  BF16 parent: Q8_0, with the experts at Q5_K and the 896-wide expert down projections at Q5_1. It is pinned by the
  Hub revision `4941f110` and its sha256. Against the parent in float32 it reads 0.0293 mean KLD and 98.74%
  margin-qualified top-1 (R4.33's record, the same bytes, no new pass).
  - **What is not flattering:** it is 31.4 MB larger than bartowski's Q5_K_S for 0.0004 more mean KLD. It was written
    without an importance matrix. Its first window reads 0.0419. The header's `general.name` is a conversion
    artifact ("Ckpt Mem"), kept so the bytes stay as published.
  - **Checks:** tokenizer 0 of 721 corpus strings and 0 of 97,302 vocabulary entries; template conformance 23 of 23
    (the file's template is JetBrains' byte for byte). The CPU and CUDA tier-smoke replies are byte-identical.
  - **The Upcoming rows:** Nemotron 3.5 Bolt 74B-A9B's and Super VL 120B-A12B's now name why they are not admitted.
    Runner refuses both files at load. Both use LatentMoE (expert projections behind a 1,024-wide latent), which
    Runner does not implement. Super VL declares 512 experts, past Runner's cap of 256. Bolt declares its expert
    counts and widths per layer, and Runner reads them as scalars (R4.27.4.2, R4.27.4.3).

## v1.14.0 - 2026-11-06

- **Fix: `-v` crashed on any file whose first block is Mamba-2; found on Nemotron 3 Nano.** The verbose summary's
  "weight type" line read block 0's fused qkv (if recurrent) or its `wq`, and a Mamba-2 block has neither (its
  matrices are `ssm_in` and `ssm_out`). The line now takes the block's first matrix that exists, `ssm_in` included.
  On a file whose first block is attention or a fused-qkv recurrent block, the report is byte for byte as before.
  `tests/test_hybrid_admission.py` runs `-v` on the Mamba-first nemotron_h fixture.

- **NVIDIA Nemotron 3 Nano 30B-A3B joins the roster at Tier 2 on ggml-org's Q4_K_M, CPU only (R4.27.4.1).**
  NVIDIA publishes no GGUF, so the file is ggml-org's automatic conversion. Against NVIDIA's BF16 parent computed in
  float32, at the model's declared dt limit, it reads 0.0222 mean KLD and 99.50% margin-qualified top-1 (bars 0.05
  and 97%, 300 positions). With the reference's configured clamp it reads 0.137, as Nemotron 3.5 Lightning did.
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 128,635 vocabulary entries.
    - Template conformance 24 of 24 in text and tokens under `nemotron35` (a new `nemotron3-nano` family).
    - llama.cpp on the same file reads 0.0243, and Runner is the closer engine.
  - **Limits:**
    - CUDA refuses the file: its sigmoid-plus-bias router has no device kernel.
    - The tool gate's case B fails, and case C fails buffered.
    - Record: `docs/compat-reports/nemotron-3-nano-30b-a3b-2026-11-05/`.
- **Ornith-1.5-35B-A3B joins the roster at Tier 2 on ornith-ai's own Q8_0, text path only (R4.27.3).** It passes the
  fidelity bar by a hair against ornith-ai's BF16 parent computed in float32: 0.0484 mean KLD and 97.50%
  margin-qualified top-1 (bars 0.05 and 97%, 300 positions, CPU).
  - **What is not flattering:**
    - Its median KLD, 0.0154, is 3.7x Qwen3.6-35B-A3B's Q8_0 on the same base.
    - ornith-ai's own BF16 GGUF already sits 0.026 from the parent, where a clean conversion sits near zero, and the
      file names no source revision. So about half the distance is in the publisher's conversion or the weights it
      came from.
    - The publisher's Q6_K (0.103) and Q4_K_M (0.171) miss. llama.cpp reads the same error spikes in them, and
      Runner is the closer engine. The 64 tensors the two K-quant files share at Q6_K are named in the row.
  - **Checks:**
    - Tokenizer 0 of 721 corpus strings and 0 of 247,091 vocabulary entries.
    - Template conformance 24 of 24 in text and tokens against the file's own template, under the `ornith` renderer
      (a new `ornith15-35b-a3b` conformance family). A system or developer turn after history has started is
      refused, where the file's template drops it silently.
    - CPU and CUDA tier-smoke replies are byte-identical. CUDA runs the 37.8 GB file split, 26 of 40 blocks on a
      24 GB MIG slice. gpu-identity with routing pinned: 4.8e-6 of the range.
    - Record: `docs/compat-reports/ornith-1.5-35b-a3b-2026-11-05/`.
  - **The Upcoming rows:** MiniCPM-V-4.7-35B-A3B's now names why it is not admitted. The publisher's GGUF labels its
    pre-tokenizer `qwen35` while openbmb's tokenizer.json uses the Qwen2-style regex, so Runner differs from the
    publisher on combining marks (3 of 721 corpus strings), and its chat template is its own.

- **deepseek2's absorbed prefill on CUDA runs on the tensor cores and passes the naive form's (R3.32).** The absorbed
  form's 576-wide row was past every tiled prefill kernel, so its prompt tiles ran `k_attn`: one block per head and
  column, with every cached row read again for each of the 16 heads. `k_attn_tcw_mla`, the wide tensor-core tile's
  twin, takes the row. S runs over all 576 columns, V is the staged row's 512-wide latent (the one table, R3.31), and
  the 16 heads are a block's 16 rows. On by default for deepseek2's absorbed form with an f16 cache.
  `RUNNER_CUDA_ATTN_TC_WIDE=0` keeps the previous path, and quantized caches keep it too. Prefill on a CUDA MIG slice
  (1g.24gb, sm_120), DeepSeek-V2-Lite-Chat Q6_K, 64 tokens at temperature 0, load average 8 to 14:

  | Context | Absorbed, new | Absorbed, before | Naive |
  |---|---|---|---|
  | 4,096 | 201.64 tok/s | 114.69 | 191.57 |
  | 8,192 | 192.43 | 78.24 | 173.38 |

  - **The kill**, declared before the first read: under 0.9x of naive at 4K or 8K, or any identity miss. The read is
    1.05x and 1.11x. Decode is unchanged, and the outputs are byte for byte the previous path's.
  - **The default** followed the attention routes' rule against the publisher's float32 model (1,999 positions at
    4,000 to 5,999): mean KL 0.986x the previous path's, Wilcoxon p 0.79, McNemar 3/2.
  - **Gates:** `make test-cuda-attn-wide` takes the `--mla-wide` fixture at 1e-5 of the logit range, and catches V read
    from the wrong slice and a dropped rescale. `tests/test_attn_wide_repeat.py` checks the same bytes twice, the
    distance to the CPU and the default. The tile's existing families write the same bytes as before.
  - **The roster's CUDA cells, re-pinned:** the Q6_K row's CUDA tier-smoke reply had moved off its pin with R3.31's
    decode kernel (#631), and #631 did not compare it. It was stale on main from #631 until this change, and no
    release was cut in that window. The Q8_0 row's CUDA reply was moved by this change's own prefill route and
    re-pinned with it, so that cell was never stale (corrected after the release, R6.23). The decode kernel's own D1
    read, missing from #631, passes: mean KL 0.991x the previous kernel's, Wilcoxon p 0.37, McNemar 1/1. With that,
    both cells are retired, pinned and recorded again. The CPU cells match their pins unchanged. Record:
    `docs/compat-reports/r3.32-cuda-attn-tcw-mla-deepseek2-2026-11-05/`.

- **`--rewrite` and `--type-plan` write `q5_k`, `q5_1` and `iq4_nl` (R4.33).** These are ports of ggml's writers
  without an importance matrix (llama.cpp c1d0e7a). On the quant-golden slice their bytes are ggml's own, byte for
  byte, and the golden files are new. A `q5_k` rule on a row that is not a multiple of 256 writes `q5_1`, its
  32-block cousin: Mellum 2.1's 896-wide expert down projections need exactly that. With `--imatrix` the three are
  written without weights, as `q4_k` and `q6_k` are; importance-weighted writers for them are R4.34.
  - **Mellum 2.1 under 9 GB, written by Runner:** `--type-plan` with default `q8_0`, routers kept and `_exps.weight`
    at `q5_k`, from the BF16 parent, writes 8,994,830,528 bytes. Against JetBrains' BF16 parent in float32 it reads a
    mean KLD of 0.0293 and a margin-qualified top-1 of 98.74% (300 positions; bar 0.05 and 97%, declared as the kill
    before the read). bartowski's Q5_K_S reads 0.0289 and 98.33% at 31 MB less. Record:
    `docs/compat-reports/mellum2.1-rewrite-q5-2026-11-05/`.

- **deepseek2's absorbed form on CUDA decodes faster than the naive form, and its cache is half the size (R3.31).**
  - **The cache:** the absorbed form keeps one [k_pe | latent] row per token and layer, and its output reads only V's
    latent slice, which is K's row. The allocator still gave it a V table beside K, both 576 wide. Now the V table is
    the K table on the host and the device: 31,104 B a token at f16 instead of 62,208. At 32,768 tokens the Q6_K's
    cache goes from 2,038 MB to 1,019 MB, and its device total from 16.55 GB to 15.53 GB. Under `--kv k8v4` (K q8_0,
    V fp4) V keeps its own table. Scores are byte-identical before and after in both forms, on the CPU and CUDA, at
    every `--kv` format.
  - **The decode:** the absorbed form's 16 query heads share one 576-wide row, a shape the grouped decode attention
    did not take (at most 8 heads up to 256 wide). So each head read every cached row and its V copy for itself.
    `k_attn_dec_mla` reads a slice's rows once for all 16 heads and takes V from the K row, merged as the grouped
    kernel's partials are. Decode on a CUDA MIG slice (1g.24gb, sm_120), the Q6_K, 64 tokens at temperature 0, load
    average 4 to 18:

    | Context | Absorbed, new | Absorbed, before | Naive |
    |---|---|---|---|
    | 4,096 | 99.95 tok/s | 43.53 | 84.20 |
    | 8,192 | 89.43 | 26.25 | 66.12 |
    | 16,384 | 74.07 | 10.41 (R4.4.5 e) | 45.88 |
    | 32,768 | 55.01 | 5.36 (R4.4.5 e) | 28.30 (forced split, after the fix below) |

    The kill (absorbed under 0.9x of naive at 4k or 8k) was declared before the first read; the read is 1.19x and
    1.35x. gpu-identity with routing pinned reads 1.35e-5 of the logit range at 512 tokens (limit 1e-3).
    - The gate: an `--mla-wide` fixture with the real geometry, at 1e-5 in `test-gates-core`, plus
      `test_absorbed_mla_decode_*`, which repeats the decode byte for byte and holds it to the CPU at 1e-4 over
      1,000 positions. Both catch V read from the wrong slice. Only the long pytest catches a dropped rescale:
      gpu-identity's 512 tokens give each slice a single tile. So the pytest is in `test-cuda-gated`'s list, and
      gpu-identity's reach is a known limit of the gate.
    - At 16,384 the output differs from the previous kernel's at the 11th generated token. There the two kernels
      split " teacher" and " beacon" by 0.052 and 0.016 nats, a near tie. The previous kernel on this build
      reproduces R4.4.5 (e)'s output byte for byte.
    - `RUNNER_CUDA_ATTN_MLA=0` keeps the previous kernel.
  - **A forced split past the budget now says what it costs, and a run the device left says so at the end.** The
    naive form at 32,768 tokens never finished in R4.4.5 (e). `--gpu 99` forced all 27 layers where the budget
    holds 26, and the split line kept the fitted plan's bytes (`used=24.24GB` of 25.13). The prompt window sized
    itself from that figure, so it took 0.72 GB of "spare" that the 27th layer already held. The device had 0.01 GB
    left, the first launch ran out of memory, and the CPU finished the run in 84 minutes. The closing speed line
    did not say so.
    - The split line and the window now charge the forced layers. The same command reads `used=25.15GB`, gets a
      64-column window with 0.17 GB free, and stays on the device: 422 s, prompt 77.89 tok/s, decode 28.30 tok/s
      (load 11 to 14). Its output is byte for byte the absorbed form's.
    - When the device fails mid-run, a line after the closing speeds says the run finished on the CPU and that
      the speeds are the CPU's. Before, the only notice came before the first tile, and a harness that keeps only
      the end of stderr lost it.
    - An explicit `--gpu-layers` is still honoured past the budget. `RUNNER_GPU_FALLBACK=fail` still ends such a
      run instead of finishing it on the CPU.
  - **Not changed here:** the absorbed prefill (114.7 against naive's 191.7 tok/s at 4,096; R3.32 above). Decode still runs without a
    CUDA graph for deepseek2 (the shared-expert path), in both forms.

- **CUDA runs `qwen35moe` (R4.27 A2): Qwen3.6-35B-A3B's Q4_K_M row gains its CUDA cell.** The by-name hold in
  `cuda_admit` is gone. The device path is a composition that already existed: dense qwen35's trunk, the routed MoE
  kernels and the gated shared expert (`tile_shexp`).
  - **The gate:** `make test-cuda-qwen35moe`, a `cuda-gate.sh` step, holds the qwen35moe toys and the gated shared
    expert to the CPU's logits at 1e-5 of the range, with free and pinned routing, on every expert route. It also
    checks speculative identity and the host and split placements. It passed on its first run, and it catches four
    declared mutants: the shared expert's gate dropped, the shared expert skipped, the wrong DeltaNet gate, and a
    kernel that adds without the sigmoid.
  - **The real file**, on a CUDA MIG slice (1g.24gb, sm_120): all 40 blocks on the device, gpu-identity 3.6e-6 of the
    logit range with routing pinned (limit 1e-3), and tier-smoke output byte-identical to the CPU's.
    `compat_matrix`'s cpu_cuda check is new on the row and reads 8 of 9 exact, with one margin-qualified near tie.
  - **Not claimed:** fidelity on CUDA against the float32 reference (the row's KLD reads are the CPU's), speed, and the
    Windows CUDA host's cells (not run: no CUDA consumer host). The three upcoming qwen35moe rows still need their
    own gates. Record: `docs/compat-reports/qwen35moe-cuda-2026-11-05/`.

- **CI: the macOS suite runs daily and on dispatch, not on every push.** GitHub's hosted
  macOS arm64 pool is capacity-limited and cancelled queued jobs often enough to keep
  main red for hours on 2026-11-04 (28 cancellations against 4 real failures on main
  over two weeks). The pull-request `macos` job keeps the Metal gates and now also
  runs the suite's Metal pytest files, which is where the four real failures were.
  The Windows suite and every Linux job are unchanged.

- **DeepSeek-V2-Lite-Chat joins the validated roster at Tier 2 (R4.4.5 step d), with two of mradermacher's static
  files: the Q8_0, the fidelity file, and the Q6_K, the smallest to pass the fidelity bar.**
  - **The reads:** against the publisher's BF16 checkpoint run in float32 through its own `modeling_deepseek.py`
    (300 positions), in the absorbed form, the default.
    - The Q8_0 reads 0.0094 mean KLD and 100% margin-qualified top-1 on the CPU, and 0.0096 and 100% on CUDA.
    - The Q6_K (14.07 GB, a 16 GB card's fit) reads 0.0293 and 98.61%, and 0.0296 and 98.61%.
    - The Q5_K_M (0.0704, 96.76%) and the Q4_K_M (0.188, 90.28%) miss the bar (0.05, 97%) and get no row.
  - **The llama.cpp column:** llama.cpp 73a43d1 reads every file further from the reference. On the Q6_K it misses
    the margin half (0.0466, 96.76%). The multi-head latent attention form is not why: in llama.cpp's own naive form
    Runner reads 0.0294 and 98.61%, and is still the closer (p 2.4e-18).
  - **The other gates:** gpu-identity's pinned arm reads 1.85e-5 and 1.94e-5 of the logit range (limit 1e-3). The
    tokenizer differential and the template's 16 cases are identical. The `deepseek2` entry leaves
    `ARCH_GATE_EXEMPTIONS`. Record: `docs/compat-reports/deepseek-v2-lite-chat-2026-11-04/`.

## v1.13.0 - 2026-11-04

- **`GET /v1/models` says the context the server runs at.** Each entry
  carries `context_window`: the resident model's context, else the
  configured `-c`; absent only on a registry that has loaded nothing yet
  and was given no `-c`. The list carried only `id`, `object` and
  `owned_by`, so a client that plans its prompts had to guess: the Dojo
  pi extension guessed 32,768, compacted itself to a standstill against a
  262k server and overran a 16k one. The conformance fixture and the
  registry smoke carry the field.

- **CUDA: `RUNNER_MOE_HOT`, a file of hot experts kept on the device for
  `--cpu-moe` host layers; 1.52x warm decode on Mellum 2.1 Q8_0 on an RTX 3070
  (R3.30).** Corrected 2026-11-06: this entry first read 1.96x, a
  fresh-process number. Limits first:
  - It is opt-in and measured on one file on one card.
  - The hot file comes from a routing trace (`RUNNER_MOE_TRACE`), so it
    fits the text the trace saw.
  - v1 covers the plain top-k SiLU expert FFN only. Shared experts, expert
    biases or scales, gemma-4's dual branch and hyper-connections are
    refused at load, as is a malformed file or the knob without
    `--cpu-moe`.
  - The first request in a process is slower than the warm figure: 26 to 33
    tok/s against 40.69.

  A listed layer runs mixed. The device norms and routes, and runs the
  listed experts with the fused path's kernels on a bank of their slices.
  At the same time the host runs the rest with the CPU path's own expert
  function. The device then sums the slots in the router's order.

  Measured on Mellum2.1-12B-A2.5B-Thinking Q8_0 with 768 slots (5.07 GB),
  chosen by frequency from six traced prompts. On a prompt the trace never
  saw, warm decode in one long-lived process reads 40.69 against
  `--cpu-moe auto`'s 26.75 tok/s (medians of three interleaved rounds), with
  a hit share of 0.716 to 0.735. Step 1's model gives 1.53x at that prompt's
  share, so the measurement agrees with it. Its 1.66x assumed a share of
  0.785, which this prompt does not reach.

  The 1.96x first published here is a fresh-process reading, 27.82 against
  14.19 tok/s, and does not measure the placement alone: going from that
  reading to warm, `auto` gains 1.89x and `hot` 1.46x. A first request pays
  for paging the mapped file. Why the two placements paid different amounts
  is an inference from page-cache state, not an isolated measurement.

  `RUNNER_MOE_HOT_PROF=1` prints where each mixed layer's time goes.

  The identity gates (`tests/test_moe_hot.py`):
  - A layer with every expert listed gives `--cpu-moe 0`'s bytes.
  - A mixed layer's sum is neither placement's bytes. It is the same bytes
    run to run, and within 1e-7 relative perplexity of the all-host
    placement.
  - The host terms are the CPU path's own by construction, not by a
    separate byte test.

  The route record gains `moe_hot`. Report:
  `docs/compat-reports/r3.30-moe-hot-placement-2026-11-04`.

- **deepseek2 runs on CUDA in the absorbed multi-head latent attention, its default (R4.4.5 step c2).**
  - The cache holds one 576-wide head per token and layer, as on the CPU. The naive form (step c1, below) stays
    behind `RUNNER_MLA_NAIVE=1` as the oracle.
  - Three kernels:
    - `k_mla_q_absorb` builds each head's Q as [q_pe | k_bᵀ q_nope] from the layer's `attn_kv_b`, dequantized at
      load (226 MB on V2-Lite, charged in the VRAM plan);
    - `k_mla_kv_absorb` writes the one [k_pe | latent] row as both K and V;
    - `k_mla_v_absorb` takes each head's output latent through its v-rows before `attn_output`.
  - The 576-wide head runs on the plain attention kernels. The tiled and tensor-core prefill stop at 512 and 128.
  - **Measured on the toy:** `test-deepseek2`'s CUDA arm is 4.0e-5 from the plain-Python reference and 9.5e-7
    from the CPU. gpu-identity reads 2.4e-7 against a limit of 1e-5.
  - **Measured on DeepSeek-V2-Lite-Chat Q8_0, CPU against CUDA (gpu-identity):**
    - free routing reads 4.8e-4 of the logit range (limit 2e-3), with 7 of 14,820 expert slots differing (ties);
    - pinned routing reads 1.9e-5 (limit 1e-3).
  - **The cost:** scoring the 8,719-token gold corpus on the prefill route (`RUNNER_SCORE_CHUNKED=1`) on the CUDA
    MIG slice takes 191 s absorbed against 146 s naive, at a host load average of 17 to 30. On the CPU the absorbed
    form takes 798 s.
  - **Against the publisher's float32 `modeling_deepseek.py`,** CUDA's absorbed form against the CPU's on the
    prefill route, by the routes' rule:
    - at positions 4000-5999 (1,999 positions), the read declared for it: mean KLD ratio 0.9993, Wilcoxon p 0.11,
      McNemar 1/0;
    - on the 300-position house window it read 1.018, over the rule's 1.01, with the other two tests passing.
      Both reads are kept.

- **deepseek2 runs on CUDA in the naive multi-head latent attention (R4.4.5 step c1), with `RUNNER_MLA_NAIVE=1`.**
  - Three layout kernels, `k_mla_q`, `k_mla_kv` and `k_mla_pack_v`, put Q and K as [pe | nope] and V as [v | 0],
    and pack the value width before `attn_output`. They are copies only.
  - The latent's projections, its norm, the rope, the cache and the 192-wide attention run on existing kernels.
  - Still on the CPU, with a reason: the q-lora pair (`attn_q_a`: DeepSeek-V2 and V3, not V2-Lite), refused by
    name.
  - The multi-sequence microbatch declines deepseek2, as every MoE model: sequences decode one at a time on the
    device.
  - **Measured on the toy:** `test-deepseek2`'s CUDA arm is 4.0e-5 from the plain-Python reference and 4.8e-7 from
    the CPU. gpu-identity reads 9.8e-7 against a limit of 1e-5.
  - **Measured on DeepSeek-V2-Lite-Chat Q8_0, CPU against CUDA (gpu-identity):**
    - free routing reads 4.5e-4 of the logit range (limit 2e-3), with 11 of 14,820 expert slots differing (ties);
    - pinned routing reads 1.9e-5 (limit 1e-3).
  - Decode runs plain launches. The shared-expert path fails graph capture, as the gated shape (`shexpg`) already
    did before this change.

- **Qwen3.8-27B GSQ-RCO IQ3_S recovered moves to Tier 2: against Qwen's
  float32 reference it reads 0.41 mean KLD, 8x the bar (R4.32).** In the
  house window the published file reads 0.4119 mean KLD and 85.04%
  margin-qualified top-1; at positions 4000-5999 it reads 0.4084 / 86.12%.
  ISTA-DASLab's unmodified source reads 0.5616 / 82.48% and 0.4718 /
  86.36%. Both predict the corpus better than the parent (perplexity 21.54
  and 53.48 against 73.29), so GSQ-RCO is a learned derivative, not a
  faithful quantization. The scale recovery moved the file toward the
  parent. The row's earlier 0.0450 / 97.80% was measured against our BF16
  GGUF on 500 short Wikitext-2 prefixes, not against the publisher's
  float32 model; it stays beneath, labelled. Owner decision 2026-11-04:
  the row is maintained on report, stays published under a written
  exception, and keeps its records as dated evidence. The Flash-Next
  sibling's row gains the limit sentence; its own reference read is
  pending. Report: `docs/compat-reports/r4.32-qwen38-iq3s-reference-2026-11-04`.

- **`test-gpu-identity` judges a MoE file by its pinned-routing arm (R1.25).**
  With `RUNNER_CUDA_ATTN_TC_WIDE=1`, Qwen3.8-Flash-Next failed the free-routing
  arm at 2.41e-3 of the logit range against 2e-3. A dump of every device
  attention call found the wide-head kernel within 1.7e-6 of a float64
  reference over its own inputs, as close as the tile kernel. The device's
  router had chosen another expert at 3 of 45,600 slots, near-ties tipped by
  rounding, and with the CPU's choices pinned the run reads 4.42e-6. The free
  arm is now reported and does not fail by itself. The gate fails instead
  when more than 1% of slots tip: a Metal router with its bias dropped tips
  20 to 40% of them, which pinning alone would have passed. A G = 12 fixture
  (24 query heads on 2 KV heads, `make-test-model.py --heads`) joins
  `test-cuda-attn-wide`. No kernel changes.

- **CUDA runs the ungated (DeepSeek-style) shared expert on the device (R4.4.5 step c0b).** Before this, a MoE
  whose shared expert has no scalar gate sent the whole model to the CPU on CUDA unless `--cpu-moe` kept the banks
  on the host. `tile_shexp` now adds such a shared expert's output without the sigmoid.
  - Granite 4 hybrid keeps its banks on the host by name, because its CUDA cells were measured that way. A relu2 or
    gate-less shared expert keeps its refusal.
  - `make test` adds gpu-identity on the `shexp` fixture: zero routed experts, with the dense FFN as the shared
    expert. Its worst difference is 2.7e-7 against a limit of 1e-5, and on the device it scores bit for bit as the
    dense oracle does.
  - The gated shape (`shexpg`) and every CUDA MoE byte gate read the same before and after.

- **Metal Q4_K decode: the fast kernel unpacks its scales without a branch,
  1.18x on Qwen3-4B Q4_K_M with the same bits (R3.28.5).** Q4_K's 6-bit
  scales unpack one way for sub-blocks 0 to 3 and another for 4 to 7, and
  `k_mvg_q4_K`'s lanes sit on both sides within a simdgroup, so every lane
  paid both paths. Now three 16-bit loads at fixed offsets and a select
  give the same integers: the output is bitwise unchanged. On an Apple M1,
  Q4_K at Qwen3-4B's 2560-wide inputs reads 47.9 GB/s against 38.2, and
  decode over R3.28.3 is 17.59 against 14.92 tok/s on Qwen3-4B, 1.15x on
  Llama-3.2-3B, 1.13x on Llama-3.2-1B, 1.02x on K2-Horizon-1B (whose
  1536-wide bulk runs the identity kernel).
- **Metal Q6_K decode: the narrow-slice kernel too, 1.20x more on
  Qwen3-4B Q4_K_M (R3.28.3).** A Q4_K_M file keeps its LM head, attn_v and
  half its ffn_down at Q6_K, which decoded at about 27 GB/s on an Apple M1.
  Q6_K inputs at least 1536 wide now run `k_mvg_q6_K`, R3.28.2's shape for
  Q6_K, at 51 to 55 GB/s: decode over the R3.28.2 build, Qwen3-4B 15.08
  against 12.56 tok/s, Llama-3.2-1B 1.36x, Llama-3.2-3B 1.23x,
  K2-Horizon-1B 1.24x. Unlike `k_mv2_q6_K`, whose bits are the identity
  kernel's, it is a tolerance route: `test-mv-tol` 0/64 flips on Qwen3-4B,
  and its float32-reference row is in the PR. `RUNNER_METAL_MVG=0` restores
  both previous kernels. The route detail says
  `mv=mvg-q4_K>=2048+q6_K>=1536`.
- **Mellum2.1-12B-A2.5B-Thinking gains a Tier 2 row for bartowski's Q5_K_S
  (8.96 GB), the smallest Mellum 2.1 file measured to pass the bar.** Against
  the publisher's BF16 parent in float32 it reads 0.0289 mean KLD and 98.33%
  margin-qualified top-1 (plain top-1 93.0%), where the publisher's Q4_K_M
  reads 0.1172 / 92.89% and Runner's own 9.38 GB selective-precision attempt
  0.0983 / 92.89%. Four published Q5-band files were scored; both with
  experts at Q5_K pass, both with experts at Q4_K miss. Owner decision
  2026-11-04; a third party's file, pinned by sha256 and repository revision.
  Report: `docs/compat-reports/mellum2.1-q5-band-2026-11-04`.


- **CUDA's MoE router can leave the selected weights as the softmax gave them (R4.4.5 step c0).** `k_moe_route`
  and its replay twin take a renormalise flag, which is the file's `expert_weights_norm`. The eager path's host
  routers read the same field. Every family that renormalises today keeps the flag on, with the same bytes:
  - the CUDA MoE byte gates read the same before and after, and so do the scores of the MoE fixtures;
  - in the PTX, only `k_moe_route` and `k_moe_route_replay` changed, of 495 kernels.

  A softmax router with `expert_weights_norm` false no longer sends the model to the CPU on CUDA. deepseek2 is
  still refused on CUDA by name until its attention lands. Metal's router has no such flag and keeps the
  refusal. `make test` gains CPU-against-device runs on the shared-router pair `rsoft` and `rsoftnn` and on
  `gnonorm` (worst 1.9e-7 and 2.7e-7, limit 1e-5). `test-moe-router` checks that the pair differs on the host
  (0.041).

- **DeepSeek-V2 (`deepseek2`) runs on the CPU (R4.4.5 step b).** These are the limits for now:
  - The GPU backends run it on the CPU with a reason, except CUDA's naive form (step c1, above).
  - The absorbed multi-head latent attention by default: the cache holds one 576-wide head a token a layer
    ([k_pe | normed latent], read by every query head; 1,152 values with the K and V rows against the naive form's
    6,144 on V2-Lite). `RUNNER_MLA_NAIVE=1` keeps the naive form (16 heads of 192), the identity gate's oracle.
  - The legacy unsplit `attn_kv_b` only; the split `attn_k_b`/`attn_v_b` layout is refused.
  - No LoRA.

  Against the publisher's float32 code (`modeling_deepseek.py`, `--trust-remote-code`), DeepSeek-V2-Lite-Chat Q8_0
  reads mean KLD 0.0100 and margin-qualified top-1 100% over the house window. llama.cpp reads 0.0155 / 99.54% on the
  same file, and Runner is the closer (Wilcoxon p 9e-15). That read is the naive form. Against the same reference at
  positions 4000-5999 of the gold corpus, the absorbed form is not distinguishable from it: mean KLD 0.01286 against
  0.01285 (1.001x), Wilcoxon p 0.095, McNemar p 1.0, margin-qualified top-1 99.48% for both. The rest is the
  publisher's:
  - the softmax scale mscale² / sqrt(key_length);
  - softmax routing without renormalisation;
  - the `deepseek-llm` pre-tokenizer (`test-tokenizer-classes`; 0 of 721 corpus segments and 0 of 99,442 vocabulary
    entries differ from the publisher's tokenizer);
  - the `deepseek2` chat template, conformant with 16 of 16 cases token-identical.

- **Metal Q4_K decode: a faster fast kernel, 1.19x on Qwen3-4B and
  Llama-3.2-3B Q4_K_M (R3.28.2).** The Q4_K inputs the fast route takes (at
  least 2048 wide on dense models) run `k_mvg_q4_K`: eight lanes share a
  superblock and a lane's 32 activations serve four rows, where
  `k_mvf_q4_K` gave a lane 64 activations for one quarter of two rows. On
  an Apple M1, decode: Qwen3-4B 12.59 against 10.61 tok/s, Llama-3.2-3B
  17.09 against 14.34, Llama-3.2-1B 1.08x, K2-Horizon-1B 1.02x. A
  tolerance route like the one it replaces: `test-mv-tol` 0/64 flips on
  Qwen3-4B, and its float32-reference row (mean KL 0.9999x of the old
  kernel's) is in the PR. A decode microbatch's columns each run the
  one-column arithmetic, so they keep the solo step's bits.
  `RUNNER_METAL_MVG=0` keeps `k_mvf_q4_K`. The route detail says
  `mv=mvg-q4_K>=2048`.

- **CUDA: the wide-head tensor-core prefill attention is on by default for
  qwen35 at head width 256 (Qwen3.5, 3.6 and 3.8) (R3.6.3.2).** On
  Qwen3.8-27B (GSQ-RCO IQ3_S, a CUDA MIG slice), 16K prefill reads 1.82x
  (329.6 against 181.1 tok/s) and 4K 1.23x. Decode is unchanged. The D1
  rule passes: mean KL 0.9998x the tile's, Wilcoxon p 0.82, McNemar 0/0
  over 2,000 positions. gpu-identity is ok on both settings. The default
  keys on the measured width, and `RUNNER_CUDA_ATTN_TC_WIDE=0` keeps the
  tile. Not yet measured: Metal and the RTX 3070 at these sizes. Report:
  `docs/compat-reports/r3.6.3.1-cuda-attn-tc-wide-qwen35-2026-11-04`.
  The Flash-Next sibling (qwen4exp) stays on the tile: on the same slice,
  with 26 of its 48 layers on the CPU, its 4K prefill reads 0.992x, and
  16K and D1 were not run. With routing pinned, gpu-identity reads
  4.42e-6 (R1.25). Report:
  `docs/compat-reports/r3.6.3.1-cuda-attn-tc-wide-qwen4exp-2026-11-04`.

- **The CUDA int8 prefill route (`RUNNER_CUDA_I8=1`) stays opt-in for every
  type (R3.14.1).** Against the publisher's float32 models, Q8_0 (Qwen3-0.6B)
  reads 1.51x the default route's mean KL. Q4_K with Q6_K (Qwen3-4B Q4_K_M)
  reads 1.008x, and is worse on the paired Wilcoxon (p 0.037). Both miss the
  routes' rule. Report: `docs/compat-reports/r3.14-cuda-int8-prefill-2026-11-03`.

- **Qwen3.6-27B joins the validated roster at Tier 2 on ggml-org's
  conversions (no publisher GGUF exists), and the Q4_K_M fails the fidelity
  bar.** Against Qwen's BF16 checkpoint in float32 at use (300 positions,
  CPU): the Q4_K_M reads 0.129 mean KLD and 95.9% margin-qualified top-1 on
  Runner, 0.154 and 93.8% on llama.cpp 73a43d1, against a bar of 0.05 and
  97%, and the gap grows with depth; the file itself is the distance (0.101
  top-20 KL against Runner's own Q8_0). The Q8_0 passes on Runner (0.0273 and
  99.59%; one reference near-tie carries most of the mean) and is the row's
  fidelity file with a CPU cell; llama.cpp reads 0.0626 on it, above the bar.
  Tokenizer differential 0 of 247,091 entries, template byte-identical to
  Qwen's with 23 of 23 conformance cases, tool gate A and B pass and C passes
  buffered. Record: `docs/compat-reports/qwen3.6-27b-2026-11-04`.

- **On CUDA with every layer on the device, `--draft lookup` now drafts 7
  tokens a round by default and `--draft mtp` 2; every other source,
  backend and partial offload keeps 4 (R3.3.7).** On an
  RTX 3070, lookup on an in-context edit reads 2.86x plain at 7 against
  2.43x at 4. Qwen3.5-4B's NextN head reads 1.13x at 2 on code and tool
  calls, against 1.01x and 1.18x at 4; on prose it stays a loss (0.88x at
  2). An adaptive width that followed each turn's acceptance was measured
  first and stopped by its kill read: 0.97x of the best fixed width, where
  it needed 1.05x. An explicit `--draft-k` is unchanged. Report:
  `docs/compat-reports/r3.3.7-draft-width-defaults-2026-11-03`.

- **`RUNNER_CPU_I8K=1`: an opt-in int8 decode route for Q4_K and Q6_K on
  AVX2 CPUs, off by default because it fails the fidelity bar (R3.1.4).** On
  a 4-core i7-7700K, Q4_K_M decode reads 1.23 to 1.39x the f32 route, 0.85
  to 0.96x of llama.cpp. Against Qwen's float32 models its mean KL is 1.09x
  the f32 route's on Qwen3-0.6B and 1.17x on Qwen3-1.7B (worse on the paired
  test, p 9e-32 and 3e-49). The small-core gap to llama.cpp is that int8
  rounding, which the default does not pay. `RUNNER_CPU_PROF=1` prints where
  a CPU decode step goes. The route record's `cpu_act` names the route, and
  `--score`'s `score_route` now carries `cpu_act`. Report:
  `docs/compat-reports/r3.1.4-cpu-small-core-decode-2026-11-04`.

- **`--quant-error` costs a row its type cannot describe at the type the
  rewrite writes instead.** A tensor whose rows the asked type's block does
  not divide (a MoE expert width of 896 under `q4_k`) read `null` in the
  report, so `runner derive --target-size --low q4_k` refused the plan
  ("has no q4_k entry") where `--rewrite --quant q4_k` of the same parent
  writes the tensor at `q5_0` through the width fallback. The report now
  costs such a row at that fallback type and names it in `written_as`; a
  plan built from the report predicts the file the rewrite makes, and derive
  plans every MoE whose expert width is not a multiple of 256. Found by a
  derive run on the lab box.

- **Nemotron 3.5 Lightning-30B-A3B joins the validated roster at Tier 2 (the
  publisher's Q4_0; the Q8_0 is the fidelity file).** Against NVIDIA's BF16
  checkpoint in float32 at use, at the model's declared time-step limit: Q4_0
  0.0476 mean KLD and 97.62% margin-qualified top-1, Q8_0 0.0146 and 99.05%,
  the BF16 GGUF 0.0055 and 100% (300 positions, CPU); llama.cpp 73a43d1 reads
  0.0499 and 0.0173 on the same files, and Runner is the closer by per-position
  KLD on both. Every number is from the build with the sigmoid router (#575).
  Two corrections to the reference are recorded: transformers 5.14.1's naive
  Mamba-2 scan is wrong across chunks (5.16.1 is the reference), and the 5.16.1
  mixer clamps dt at the initialisation floor rather than the config's declared
  limit (0.246 against 0.0055 on the BF16 file; the declared limit predicts the
  corpus better). CPU only: the router has no device kernel. The three
  `qwen35moe` Upcoming rows no longer claim the architecture does not load.
  Record: `docs/compat-reports/nemotron-3.5-lightning-30b-a3b-2026-11-03`.

- **Metal: the fast Q4_K decode matvec is the default only for inputs at
  least 2048 wide (R3.28.1).** On 1536-wide inputs it was 15% slower than
  the identity kernel (K2-Horizon-1B Q4_K_M: 11.11 against 9.62 ms of Q4_K
  time a token on an Apple M1), while it won at every width from 2048 to
  8192 on the Llama-3.2-1B/3B Q4_K_M files it was promoted on. Narrower Q4_K
  inputs now keep the identity kernel, so K2-Horizon-1B Q4_K_M decodes
  1.061x faster (39.7 against 37.5 tok/s). The Llama files are unchanged:
  their `--score` output is byte-identical to before, decode 0.997x and
  1.022x, inside the run-to-run spread. Measured on one laptop, with one
  file for the narrow case; widths between 1536 and 2048 were not measured
  and take the exact kernel. The route detail says `mv=fast-q4_K>=2048`.

- **Granite-4.2-30B joins the validated roster at Tier 2, and its publisher's
  Q4_K_M fails the fidelity bar.** Against IBM's BF16 checkpoint in float32
  at use (300 positions, CPU): the Q4_K_M reads 0.139 mean KLD and 93.9%
  margin-qualified top-1 on Runner, 0.151 and 92.5% on llama.cpp 73a43d1,
  where the bar is 0.05 and 97%; the file itself is the distance (0.111
  top-20 KL against Runner's own Q8_0). IBM's Q8_0 passes on both engines
  (Runner 0.0041 and 99.56%, llama.cpp 0.0241 and 99.12%) and is the row's
  fidelity file, with a CPU cell on the Linux CUDA host (31.1 GB, more than a
  24 GB device holds). Tokenizer differential 0 of 99,474 entries, template
  byte-identical to the publisher's with 24 of 24 conformance cases, tool gate
  A and B pass and C passes buffered (streamed, the run spent its reasoning
  budget before any call). Record:
  `docs/compat-reports/granite-4.2-30b-2026-11-04`.

- **Nemotron 3.5 Lightning (`nemotron_h_moe`) routed its experts with the wrong
  function; fixed. Every nemotron_h_moe file has given wrong output since
  v0.2.0.** The publisher's router is a sigmoid; the published GGUFs carry no
  `expert_gating_func` key, and the loader read the missing key as a softmax.
  On the real Lightning-30B Q8_0 against llama.cpp's logits over the same
  1,109 positions: perplexity 18.31 before, 11.83 now (llama.cpp 11.73), top-1
  agreement 63.2% before, 95.8% now. A file that declares the key keeps it.
  Requantize nothing; reload the file.

- **`RUNNER_DEBUG_TIME` reports the attention core inside the attention
  stage.** A new `attn core` line times only the scores, softmax and values
  pass, without the projections. Timing only; output unchanged.

- **Metal prefill attention takes 512-wide heads on the tile, byte-identical,
  by default for gemma-4's E-series (R3.6.3.2).** gemma-4's 512-wide global
  layers ran the one-column kernel over the whole history, 66% of an 8K
  Nemotron 3.5 Bolt E4B prefill on the 8 GB laptop. They now take
  `k_attn_tile`, the kernel 256-wide heads already used, with `k_attn`'s
  arithmetic in `k_attn`'s order, so the output is the same bytes: one
  global layer's attention at 8K ran in 3.4 s against 25.0 s, and the 8K
  prefill's GPU time went from 274 s to 117-123 s (about 2.3x; 4K: 1.43x in
  tok/s). Decode is unchanged. Measured only on the 8 GB laptop and only to
  8K: a 16K prefill of this file does not fit that laptop without swap. On
  by default for gemma4 with per-layer embeddings, the family CUDA's wide
  kernel defaults to; `RUNNER_METAL_ATTN_TILE_WIDE=0` keeps the one-column
  kernel for heads wider than 256, `=1` takes the tile on any model. The
  route detail says `attn_wide=on`.
- **CUDA prefill attention for 256- to 512-wide heads runs on the tensor
  cores, by default for gemma-4's E-series (R3.6.3.1).** gemma-4's 512-wide
  global layers took the tile's 8-row kernel on the CUDA cores, 79% of a
  32K prefill on an RTX 3070. `k_attn_tcw` takes them, and the 256-wide
  sliding layers, with `k_attn_tc`'s operands: one global layer at about
  31K in 55 ms against 352. Nemotron 3.5 Bolt E4B prefill: +8% to +19% at
  512 tokens, +30% to +40% at 4K, 2.0x at 16K, 2.6x at 32K; decode and the
  greedy text unchanged. On by default for gemma4 with per-layer embeddings
  (owner decision), its float32-reference row on Bolt E4B not
  distinguishable from the tile's (mean KL 0.9965x, Wilcoxon p 0.188);
  `RUNNER_CUDA_ATTN_TC_WIDE=0` keeps the tile. fp16 caches
  only; a layer with sinks (gpt-oss, flint) keeps the tile. The route
  detail says `attn_wide=on`. Report:
  `docs/compat-reports/r3.6.3.1-cuda-attn-tc-wide-2026-11-03`.
- **CUDA decode attention for eight query heads per KV head (R3.29.1).**
  With eight heads in a block, the grouped decode kernel read each V pair
  once per head and each probability alone, and was bound by shared-memory
  work. A thread now reads its V pair once for all its heads and the
  probabilities four keys at a time, in the same order: the scores are the
  same bytes. On an RTX 3070 at 16K, decode against the previous build:
  1.18x on Qwen2.5-3B (0.66x to 0.78x of llama.cpp b11748), 1.08x on
  Qwen2.5-7B, 1.03x to 1.04x on Qwen3-4B and Llama-3.2-3B; 1.00x to 1.01x
  at 512. Report: `docs/compat-reports/r3.29.1-cuda-decode-gqa8-2026-11-03`.

- **`RUNNER_SPEC_PROF=1` prints its profile line on a schema-constrained run too.** The
  speculative walk left through three early exits when a constrained document
  completed (a close before a token's forward, mid-round, at a plain token);
  they printed the round summary but not the profile. The MANUAL's
  `--draft-k` row now states the range, 1 to 15 (a round verifies the drafts
  plus the correction, at most 16 rows).


- **A Metal layer split no longer refuses a file stored in name order or with
  its output norm last.** A split wraps a prefix of the file, and it refused
  layers stored out of block order and any non-layer tensor after them; Qwen's
  Qwen2.5-1.5B-Instruct Q8_0 does both (blk.0, blk.1, blk.10 to blk.19, blk.2,
  ..., then `output_norm.weight`), so `--gpu N` on it ran wholly on the CPU.
  The wrap now reaches the farthest tensor of the device's layers whatever
  their order, and tensors a split never reads from the device (the head, the
  embedding, gemma-4 E-series' per-layer tables) may follow the layers; any
  other tensor there still refuses the split, and the message names it.


- **A session resume or a shared prefix is no longer refused when the model's
  path is spelled differently.** The engine's model key hashed the `-m` path
  as typed, so the same file loaded as `./models/m.gguf` and as its absolute
  path had two keys: a session image written under one spelling was refused
  under the other, and two slots did not share prefix-cache entries. The key
  now names the file (its stat identity and content samples, or its resolved
  path where stat cannot tell); a file replaced in place still changes it.
  **What changes for callers:** the key version moves to v3, so prefix caches
  and session images saved by an earlier build are refused once (the refusal
  says so). A refusal that remains now names what
  differs, with both values and the flag that resumes it ("context 256 in the
  image, 128 here: resume with -c 256").

- **CUDA decode attention reads the cache once (R3.29).** The grouped decode
  kernel read K twice, a pass for the softmax maximum and then K with V; it
  now makes one pass with a running maximum, stages K and V as 16-byte
  chunks prefetched into registers, and fits three blocks an SM. On an RTX
  3070, decode against the previous build: 1.00x to 1.02x at 512 tokens,
  1.24x to 1.38x at 16K and 1.37x to 1.55x at 32K across Llama-3.2-3B,
  Qwen2.5-3B, Qwen3-4B and Qwen2.5-7B. It is still 0.66x (Qwen2.5-3B) to
  0.86x (Llama-3.2-3B) of llama.cpp b11748 at 16K. The order of the sums
  changed: the same bytes run to run for a geometry, and its
  float32-reference row is not distinguishable from the two-pass kernel's
  (Qwen3-0.6B, mean KL 1.0006x, Wilcoxon p 0.890). Report:
  `docs/compat-reports/r3.29-cuda-decode-attention-2026-11-03`.

- **A recurrent draft model no longer refeeds its whole context on every
  rejected draft.** A draft whose layers fold state (LFM2, Gated DeltaNet)
  went back to position 0 on each rejection, so its catch-up refed the whole
  context every round: an LFM2.5-350M draft for LFM2.5-1.2B on a four-core
  CPU ran at 0.13x of plain decoding on prose (113 rounds at about 526 ms of
  draft work each). Each round now snapshots the draft's fold after its
  catch-up and a rejection restores it, so a round feeds the draft at most
  `--draft-k` + 1 tokens; output is unchanged. CUDA rolls a fold back by one
  forward only, so a CUDA-resident recurrent draft keeps the old refeed. The
  spec line on stderr gains `draft catch-up N`.

- **`RUNNER_CUDA_I8=1` runs CUDA prefill on int8 tensor cores (opt-in,
  compute capability 8.0+, Q8_0, Q4_K and Q6_K).** The activations are
  quantized per 32-element block, and the weights' own integer quants go to
  `mma.sync` unconverted. It is off by default: `test-tc-tol` fails on
  Qwen3-0.6B Q8_0, whose greedy tokens diverge from the scalar route at a
  near-tie, although no file tested flips a top-1 token in the gate's 64
  teacher-forced positions. On an RTX 3070 the GEMM time is 1.66x less than
  the fp16 route's on Qwen3-1.7B Q8_0 and 1.18x less on Llama-3.2-3B Q4_K_M,
  and prefill gains 8% to 36% across four files, with decode unchanged. The
  read against the publisher's float32 model did not run (stopped for low
  memory on the 16 GB host). The route detail a receipt carries says `i8=on`
  or `i8=off`. Report: `docs/compat-reports/r3.14-cuda-int8-prefill-2026-11-03`.

- **The sliding-window KV ring is on by default where the flat cache does
  not fit (`--kv-ring auto|on|off`), and rewind and the prefix cache work
  with it.** The cost first: under the ring a shared-prefix hit is lost
  when the cached prompt runs past the shared part by more than one batch
  plus one token, which the flat cache would serve; a saved prefix cache, an
  exported named context or a session image takes only a ring with no gap
  below its span; Metal's decode microbatch runs ring members one at a
  time. The auto rule fired wrongly on the first real file it met
  (gemma-4-E4B at 32k on an 8 GB RTX 3070, where the flat cache costs no
  layer): the device weight estimate counted E-series' host-only per-layer
  token table, and now leaves it out. That estimate also fed the VRAM
  registry's claim, the `-c 0 --reserve-vram` auto-fit and the KV-trade
  note, all too large on E-series files since 2026-08-01. Rewind and
  snapshots resume from what the ring still holds by one rule
  ([docs/kv-ring.md](docs/kv-ring.md)), with the ring's oldest held position
  tracked where KV is written; output is bit-identical to the flat cache
  through the server on the CPU, Metal and CUDA paths. Measured on the RTX
  3070 (gemma-4-E4B Q4_0, 4k to 32k, two interleaved rounds): the cache is
  2.2 to 3.2 times smaller (1879 to 579 MB at 32k) with the same output;
  speed is level, not better (prompt within 1.2% bar one flat-arm round at
  8k, decode within 1.2% at 8k to 32k; the 4k rows stopped after one
  token), because the flat cache costs that card no layer; and a repeated
  3,000-token system prompt hits the prefix cache on both arms (0.16 to
  0.17 s). On that card the flat cache
  fits at 32k, so auto leaves the ring off there. `RUNNER_KV_RING` still
  overrides the flag. **What changes for callers:** on a sliding-window
  model whose flat cache does not fit, the ring is now on without asking;
  `--kv-ring off` restores the flat cache. On CUDA, `-c 0 --reserve-vram`
  fits a larger context on a gemma-4 E-series file than before, because the
  weight charge drops by the per-layer token table (E4B: 5,155 to 2,843
  MB), and the VRAM registry claims that much less. `--info fit` is
  unchanged: it already left the table out.

- **A qwen3_xml call closed by the next call's opener stands, on an auto
  turn only, and says so.** Nemotron 3.5 Lightning at temperature 0 writes
  three `<tool_call>` openers, each with a complete function block, and one
  `</tool_call>` at the end; Runner refused the whole turn as
  `envelope_unmapped`. On a parse-only (auto) turn a call whose function block
  is complete now ends at the next opener, goes out when that opener arrives,
  and the turn reports `finish_detail: tool_call_implicit_close`. A last call
  with no closer, an attribute-spelled parameter (`<parameter pattern="...">`,
  the same model at its own sampling) and every grammar-framed turn stay
  faults. Prior art disagrees: vLLM's Python parser (v0.11.0) recovers these
  calls, its Rust parser and llama.cpp's require the closer, and llama.cpp
  constrains the turn after its first opener, which Runner's auto turn does
  not (measured 2026-09-14). The fixture is the model's own temperature-0
  bytes.

- **`test-gpu-identity` pins a MoE file's routing and bounds dense and
  pinned runs at 1e-3 of the logit range (was 2e-3).** A MoE file now runs
  on the device twice: with its own routing, bounded at 2e-3 as before, and
  with the CPU run's expert choices replayed, bounded at 1e-3. A top-8
  router picks another expert wherever two gates nearly tie, so free routing
  on Qwen3-30B-A3B read 1.46e-3 on CUDA; the pinned run measures the
  arithmetic alone. The gate prints how many expert slots the pin forced and how
  many differed from the device's own choice, and fails when the pin missed
  a routing. 1e-3 sits 2.5x over the worst dense real file measured (IQ1_S,
  4.0e-4) and 2.05x under the smallest real mistake (attention scale x1.05,
  2.05e-3).
- **`RUNNER_MOE_REPLAY` pins Metal's router too** (the batched and the
  per-column route), with the same counters as the CPU and CUDA routers.

- **Receipts name the model by base name and sha256, never by host path,
  and carry the route they ran (`xyntetik.runner.transcript.v2`).** A
  receipt is evidence a user may post, and v1 wrote the path the model was
  loaded from, which showed where it sat on the machine. A `--record` or
  `--transcript` file and every served receipt now name the model and the
  adapter as `{"name", "sha256"}`, with `"source": "hf:<owner>/<repo>@<rev>"`
  for a file fetched with `-hf` (the commit the Hub's listing named), and
  link a step's record by base name too. `profile.route` records the
  backend, `cpu_act` and `backend_detail`; `--verify` requires the same
  route on the same backend (`UNVERIFIABLE` naming both routes otherwise)
  and reports it on another (`"route"` in the verdict: `matched`,
  `other backend`, `not recorded`). Without `-m`, `--verify` loads the file
  the record names from the working directory; the sha256 is checked before
  the replay, and a mismatch is `UNVERIFIABLE` naming both files. v1
  receipts still verify with `-m`, and `--lineage`, bundles and packs read
  both versions. **What changes for callers:** the receipt JSON loses
  `model.path` and `adapter.path` (read `name`), and its schema_version is
  v2.

## v1.12.0 - 2026-11-03

- **`--quant q4_k` wrote corrupt tensors on rows that divide by 32 but not
  256 (since v0.5.7); fixed.** Such a row cannot take a 256-block K-quant,
  so the quantizer falls back to Q5_0, the 32-block type of the nearest bit
  budget, as llama.cpp does. There was no Q5_0 writer: the row's float
  bytes were written under a Q5_0 header. **Any file Runner wrote with
  `--quant q4_k` (through `--quantize`, `--rewrite`, `runner derive` or a
  type plan) from v0.5.7 on, from a model with such rows, is corrupt and
  must be requantized.** Such rows occur in SmolLM2-135M (576 wide),
  Qwen2.5-0.5B (896) and Kvist-14B (5,760), among others. To check a file:
  the quantize histogram lists `Q5_0` tensors (`--info fit` names the weight
  type), and the file scores NaN or garbage (SmolLM2-135M: NaN logits, no tokens). Q5_0 now
  has its writer, byte-identical to ggml's; a fallback is only ever a type
  the quantizer writes; and a row with no writer stops the quantizer
  instead of being copied. Files from llama.cpp, and Runner files at
  other types, are not affected.

- **Smaller fixes from the pre-1.12.0 sweep.** A `--quant` name the
  quantizer does not take (`Q4_K_M`, llama.cpp's mix name, or `q4_K`) is
  refused with the names it does take. `runner derive` whose rewrite fails
  leaves nothing beside OUT; it left the parent's score file. And the
  route-line wording for the two 1.12.0 opt-ins is corrected: the route line
  and the `--score` and `--bench-json` route records name them, receipts do
  not.

- **Six messages and edges the pre-1.12.0 sweep found.** A one-shot run
  whose `-n` does not fit in `-c` now ends with `[context full: -c N holds
  the prompt and K of the M tokens -n asked for; rerun with -c X or
  larger]` instead of stopping silently (the server already finished such a
  request with `length`, and the tests now pin it). A GGUF cut short in its
  metadata or tensor table says it is truncated, with the bytes it needs and
  the bytes it has, instead of `bad GGUF metadata value for ...`. `--verify`
  on a receipt a JSON tool re-serialized says so (the chain hash covers the
  runner's compact bytes) instead of `no chain in record`, which stays for a
  record that has none. `runner attest VERB --help` prints that verb's usage
  and exits 0. A value flag at the end of the line names itself (`error: -p
  needs a value`) before the usage. A failed `-hf` lookup removes the cache
  directory it created, never one that was there before. Each has a test
  that fails without it.

- **CUDA MoE routing traces record each expert's output norm.** The CUDA
  trace left the norms field empty, so a trace taken on the device could not
  be compared field by field with the CPU's. Every CUDA MoE path (fused,
  grouped, eager, by expert, and their gemma-4 forms) now writes the CPU's
  definition: the expert output's norm with the routing weight divided back
  out and the down bias added. `make test-cuda-moe-replay` compares every
  device record with the CPU's: experts exactly, gates within 1e-3 and norms
  within 2e-2 relative (measured spread: gates 5.2e-5, norms 4.0e-3 in the
  chunked mode). Tracing only; scores are unchanged. The real-file
  `test-gpu-identity` bound stays 2e-3: dense and IQ files read at most
  4.0e-4, but Qwen3-30B-A3B reads 1.06e-3 to 1.46e-3 because near-tie top-8
  routing choices flip between backends (21 of 1,872 on one prompt; 1.5e-3
  with the CPU's routing replayed). The audit report has the tables (R3.25).

- **CUDA decode applies the attention temperature past its first step.**
  Llama 4 and mistral3 files that declare the attention temperature
  (Devstral Small 2, Ministral 3, Magistral Small 1.2) decoded with a stale
  factor on CUDA when the whole model ran on the device. The decode graph is
  captured once and replayed, and the factor reached the kernel as a host
  argument, so every replay used the factor of the position it was captured
  at. A graph captured below the floor (llama 4: 8,192 positions; mistral3:
  the file's original context length) applied no temperature past it, and
  one captured above it kept that factor for later tokens, including those
  of a later, shorter request. Prefill, and decode with graphs off
  (`RUNNER_CUDA_GRAPH_OFF=1`), were right. The graph is now captured again
  when the factor moves, once per floor. In llama 4 since v0.1.5-alpha and
  mistral3 since 1.11.0. Found by the CUDA gate audit (R3.25), which also
  puts the fixture that shows it on the 1e-5 bound.

- **CPU prefill attention scales with threads again.** A prefill batch's
  attention ran one thread-pool pass per query row, so on many cores the
  dispatch, not the arithmetic, set the time, and more threads made it slower
  (3.7 s at 16 threads against 1.9 s at 4 for a 588-token prompt on a
  Threadripper, Qwen3-0.6B Q8_0). It now runs the whole batch in one pass over
  heads, each running every row exactly as before: outputs are byte-identical
  at any thread count, and nothing more is allocated. On that host, at 16
  threads, three interleaved rounds against the previous build read 0.87x to
  1.72x (medians 1.10x to 1.52x over SmolLM2-135M and Qwen3-0.6B, GCC and
  Clang builds); at 1 and 4 threads it is level. `RUNNER_ATTN_BATCH=0`
  restores the per-row passes. On Apple Silicon with an f16 cache, the
  attention's value sum also holds 16 channels in registers across the whole
  span instead of storing and reloading them at every position, with each
  channel's sum
  unchanged: on an Apple M1 a 3,497-token SmolLM2-135M prefill went from 106
  to 121 tok/s on one thread and from 234 to 334 on four, the scores
  byte-identical to the previous build (R3.27).

- **`--train-gpu` measured on a sparse-MoE model, and no faster there on
  a small device.** Qwen3-30B-A3B Q4_K_M trains whole on a 24 GB MIG slice
  (15.9 GB on the device), with losses within 1.6e-3 of the CPU trainer's
  over three steps. A step took 52 to 55 s against 53 s for the CPU trainer
  on 16 threads: the step is device-bound on many small expert products.
  The no-norm MoE fixture joins `make test`'s device-mode gradient gates.
  Record: `docs/compat-reports/train-gpu-2026-11-02-linux-x86_64-qwen3-30b-a3b.md`
  (R17.3.3).

- **Shared-expert MoE and qwen35moe on Metal.** Metal had no path for an
  always-on shared expert, so every MoE model with one (the Qwen2-MoE form)
  and `qwen35moe` (Qwen3.5/3.6-35B-A3B and the models built on them) ran on
  the CPU. The gated shared expert now runs on the device, composed from the
  dense FFN's kernels, a one-row router matvec and the per-column sigmoid
  scale, and `qwen35moe` follows. Checked on toy models only, to the CPU's
  logits at 1e-5 of range (`make test-metal-qwen35moe`, residue 5.9e-7 at
  most; a dropped sigmoid or a dropped add fails by 2.7e-3 or more): no
  `qwen35moe` file fits the 8 GB Mac it was built on, so no file row is
  measured on Metal. The gate-less squared-ReLU shared expert keeps the CPU,
  and CUDA still refuses `qwen35moe` (R4.27).

- **A PR-time check for POSIX calls the Windows build lacks.**
  `scripts/check-portability.py` runs in every PR's metadata check and fails
  on a call to a function the MinGW-w64 CRT does not declare (`setenv`,
  `unsetenv`, `realpath`, `getline`, `fork`, `mmap` and twelve more,
  each verified against the mingw-w64 headers) outside code the preprocessor
  keeps from Windows. The Windows suite runs on main only, so a bare `setenv`
  in a C test had passed every PR check and broken main (#553, fixed in
  #554). A line guarded some other way carries a `portability: ok` comment.


- **`RUNNER_CPU_I8` on a build with no int8 kernel is ignored, with a
  notice.** It used to switch the route on in name only: every matvec ran
  f32, while the route line and the `--score` and `--bench-json` route
  records said `cpu_act` `int8`, a route that never ran. That was so on the 1.11.0 macOS binary (see the entry on the
  macOS build above). `RUNNER_CPU_I8_PREFILL` already behaved this way.

- **The macOS build has the int8 dot path again.** Apple Silicon builds now
  target `-mcpu=apple-m1`, the floor of every Apple Silicon Mac, instead of
  `-march=native`. On a macOS virtual machine, which is where the release
  binaries are built, `-march=native` means generic armv8-a without the
  dot-product extension, so the 1.11.0 macOS binary was built without the
  int8 kernels: `RUNNER_CPU_I8` and `RUNNER_CPU_I8_PREFILL` fell back to the
  f32 route there without saying so. A local build on a Mac had them. CI and
  the release now check that the macOS binary takes the int8 route.

- **Local `$ref` and `$defs` in request schemas.** A `json_schema`
  response format or a tool's parameters may now reach nested definitions
  through local references, the way Pydantic's `model_json_schema()` and MCP
  tool definitions write them: `#/$defs/...`, `#/definitions/...` or any JSON
  pointer into the same schema is inlined at compile time and constrains
  exactly as its target. Until now every such schema was a 400 ("unsupported
  schema keyword '$defs'"). Refused by name: a reference outside the document,
  one that does not resolve, a recursive one, `$dynamicRef`, and a
  non-annotation keyword beside `$ref`. The prompt renders the tools as sent.
  A nested `anyOf` that cannot be dispatched is still refused, and its message
  now names `anyOf` rather than `oneOf` (R2.9.2).

- **Devstral Small 2 in the validated roster, with its tool-calling record.**
  unsloth's Q4_K_M and Q8_0 measured against Mistral AI's FP8 checkpoint
  dequantized exactly to float32 (every GGUF of the model requantizes that
  checkpoint, and the publisher serves it with static FP8 activations, so
  this is the reference arithmetic, not the publisher's serving arithmetic),
  300 positions: Q4_K_M 0.0223 mean KLD and 99.53% margin-qualified top-1,
  Q8_0 0.00126 and 100%; ggml-org's Q8_0 scores byte-identically to unsloth's.
  Agent torture on the Q4_K_M passes 100 of 100 cases (Tier 2, #551).

- **Agent torture v5: reasoning, then a native tool call.** A new case
  family, `reasoning_then_native_tool`, holds the thinking channel followed
  by a call in the family's native syntax: its tool carries no length,
  pattern or range constraint, so families whose function-XML cannot enforce
  those (Qwen3-Coder) are measured on the native path rather than the generic
  envelope. The default matrix is 126 cases (nine families of 14), and
  reports read `xyntetik.agent-torture.v5`; v4 results stay valid on their own
  terms. `scripts/agent-torture.py --timeout SECONDS` gives a slow host more
  than the harness's 120 s a request. On Qwen3.8-27B (CPU, qwen38 template,
  thinking on) the new case passes, the reasoning then the native call
  (#506).

- **CPU prefill in int8, opt-in.** With `RUNNER_CPU_I8_PREFILL=1`, CPU
  batches of 8 or more rows (prompt prefill, chunked scoring) quantize the
  activations to int8 once a matvec and run them against four-row int8 packs
  of the weights with the dot-product instructions (Q8_0, Q4_0, Q4_K and
  Q6_K, arm64 with the dot-product extension). It is measurably farther from
  the publisher's model than the f32 route, so it stays off by default: on
  Qwen3-0.6B Q8_0 against Qwen's float32 model over 2,000 positions, mean
  KLD 0.00747 against 0.00610 (1.22x, Wilcoxon p 5e-86), margin-qualified
  top-1 100% on both. On an Apple M1 at 4 threads, CPU prefill of 512 tokens
  went from 187 to 315 tok/s on that file and from 29.7 to 44.4 tok/s on
  Qwen3-4B Q4_K_M. The route line names it (`cpu_act` ends in
  `+int8-prefill`). No x86 kernel yet (R3.1.5 B).

- **Metal decode microbatch on the GEMM tiles, opt-in.** With
  `RUNNER_METAL_BATCH_TILES=1`, 6 or more `--parallel` slots decoding
  together run their weight products on the prefill GEMM tiles instead of
  the per-column identity matvec. It is not bit-identical to sequential
  decode, so it stays off by default, and the route line and the `--score`
  and `--bench-json` route records name it (`micro=tiles`); receipts do not
  carry the route yet. On an Apple M1 with K2-Horizon-1B, aggregate decode at
  8 slots went from 45.2 to 63.2 tok/s (Q4_K_M) and from 58.2 to 69.9 tok/s
  (Q8_0); at 2 to 5 slots the tiles were slower, so those widths keep the
  identity route. Against the publisher's float32 model over 2,000 positions
  it read 1.006x the identity route's mean KLD on the Q8_0 and 0.9996x on the
  Q4_K_M, not distinguishable on either
  (`docs/compat-reports/r3.24b-metal-microbatch-tiles-2026-11-02/`).
  `RUNNER_SCORE_MICRO=N` scores each
  position as one column of an N-wide microbatch, which is how the route is
  measured; on the default route it scores solo's bytes (R3.24 (b)).

<!-- Earlier releases continue in docs/changelog-archive.md. -->
