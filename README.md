# Xyntetik Runner

**Model files you can check, and the open engine behind them.** Xyntetik
publishes compressed and modified versions of open models, each measured
against its original. Runner is the engine that measures and serves them, and
builds most of them: one native binary in plain C for CPU, CUDA and Metal,
with no Python or third-party runtime underneath.

<p align="center">
  <a href="https://buy.stripe.com/9B69AUddpdx9auHgP27N600"><img src="site/assets/support-button.svg" alt="Support this work" height="44"></a>
</p>
<p align="center">
  If a file here saved you memory or time, a contribution funds the hardware time behind the next one.<br>
  <a href="https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=model_request.yml">Request a model</a> ·
  <a href="https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=result_report.yml">Report a result</a> ·
  <a href="docs/community-results.md">Community results</a>
</p>

> **Version `1.0.2`.** The command line, the HTTP API, the record formats and
> the model files that load stay compatible across 1.x
> ([versioning policy](docs/versioning.md)). This page is the short version;
> every flag, endpoint, limit and measurement is in the [manual](MANUAL.md).

**On this page:** [Files](#files) · [Quick start](#quick-start) ·
[Make your own file](#models-and-conversion) · [Tool calls](#truncation) ·
[Structured output](#structured-output) · [Training](#adaptation) ·
[Receipts](#record-and-verify-a-run) · [Serving](#serving-and-apis) ·
[Long agent turns](#reasoning-budget) · [Fit and memory](#runtime-and-hardware) ·
[Shadow mode](#shadow-mode) · [Tray](#desktop-tray) · [More](#more)

<a id="files"></a>
<a id="published-artifacts"></a>
## Files, each with its number against the original

| File | Size | Margin-qualified top-1 | Mean KLD | What was done |
|---|---|---|---|---|
| [Qwen3.8-27B IQ3_S, recovered scales](https://huggingface.co/Joakimpalm-Zen/Qwen3.8-27B-GSQ-RCO-IQ3_S-recovered-GGUF) | 11.77 GB | 97.80% | 0.0450 | ISTA-DASLab's 3-bit file with every block scale retrained against the BF16 parent. Same bytes and layout, so any engine that reads the source reads this one. |
| [Qwen3-30B-A3B, selective precision](https://huggingface.co/Joakimpalm-Zen/Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF) | 17.99 GB | 99.50% | 0.034 | Attention at Q8_0, experts at Q4_0. Smaller than the official uniform Q4_K_M (18.56 GB) and closer to the original: that file reads 94.75% and 0.114. |
| [Qwen3-Coder-30B, keep-120](https://huggingface.co/Joakimpalm-Zen/Qwen3-Coder-30B-A3B-Instruct-keep120-Q4_K_M-GGUF) | 17.5 GB | 100.00% | 0.00738 | 120 of 128 experts kept per layer. 1.1 GB under the stock 18.6 GB file. |

A file passes when it agrees with its original on at least 97% of clear-cut
tokens and its mean KLD is at most 0.05. Each row is copied from the file's
own card, which has the method, the date and the limits. Every published
file is in the [ledger](MANUAL.md#published-artifacts) and on
[Hugging Face](https://huggingface.co/Joakimpalm-Zen).

<a id="sixty-seconds-to-a-served-model"></a>
## Quick start

**1. Download Runner** (macOS on Apple Silicon shown;
[Linux, Windows and checksum steps](MANUAL.md#quick-start)):

```sh
curl -LO https://github.com/Joakimpalm-Zen/xyntetik-runner/releases/latest/download/runner-macos-arm64
chmod +x runner-macos-arm64 && mv runner-macos-arm64 runner
```

**2. Start a model.** Runner downloads it, checks it against the Hub's
SHA-256 record and serves it on `http://127.0.0.1:8080`:

```sh
./runner -hf Joakimpalm-Zen/Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF --serve
# on an 8 GB machine, start smaller:
./runner -hf ibm-granite/granite-4.1-3b-GGUF:Q8_0 --serve
```

**3. Send a request:**

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

Prefer to chat in the terminal? `./runner -m model.gguf -i`.

## Features

<a id="models-and-conversion"></a>
### Make your own model file

**Why.** A model file is a huge table of numbers, and the usual way to shrink
it squeezes every part the same amount, including the parts that matter most.
Runner lets you choose which parts to squeeze and which to keep sharp, and
then tells you how far the result drifted from the original.

**How.**

```sh
# keep some tensors sharp and squeeze others, from a plan file
./runner -m model-Q8_0.gguf --quantize out.gguf --type-plan plan.json
# drop the experts a mixture-of-experts model rarely uses
./runner -m model.gguf --quantize pruned.gguf --prune-experts keep.json
# remove one block's attention or feed-forward sublayer
./runner -m model.gguf --quantize cut.gguf --remove-sublayer attn:48
# then score the result against its parent
python3 scripts/kld-compare-raw.py --model-a out.gguf --model-b model-Q8_0.gguf \
    --runner ./runner --corpus corpus.txt --max-positions 400 --out fidelity.json
```

[Plan formats and limits](MANUAL.md#models-and-conversion)

<a id="truncation"></a>
### Tool calls that survive the token limit

**Why.** A model only gets so many words per reply. If it runs out halfway
through filling in a form for a tool, most engines hand your program half a
form, and the agent has to start again. Runner finishes the form in the
smallest valid way and tells you which fields it filled in itself.

**How.** Nothing to switch on. Serve a model and send a request with `tools`;
a call cut short by `max_tokens` still parses, and
`runner_telemetry.closure` lists the values the closer wrote.

```sh
./runner -m model.gguf --serve
```

Measured on the same prompt and schema against vLLM, llama.cpp, Ollama,
TensorRT-LLM and SGLang, Runner was the only engine that returned an
executable call at every budget from 1 to 16 tokens.
[Method and raw responses](docs/truncation-benchmark.md) ·
[Details](MANUAL.md#truncation)

<a id="structured-output"></a>
### Structured output and decisions you can gate

**Why.** Programs need answers in a fixed shape, and a model that is only
asked for JSON sometimes adds a stray word or forgets a field. Runner lets
the model pick only words that keep the answer valid, so the shape is
guaranteed. When the answer is a choice, it can also say how sure the model
was, so the sure answers go straight through and the unsure ones go to a
person.

**How.**

```sh
./runner -m model.gguf -p "Return a status object" --json
./runner -m model.gguf -p "Classify this ticket" --json-schema schema.json
```

Over the API, use `response_format` or a tool list, and add
`"confirm_below": 0.8` to get the confidence verdict in
`runner_telemetry.decision`.
[Schema coverage and decision records](MANUAL.md#structured-output)

<a id="adaptation"></a>
### Train the model file you actually serve

**Why.** Teaching a model a new habit normally means training a large
full-precision copy and compressing it afterwards, so the model you tested is
not quite the one you ship. Runner trains a small add-on, a LoRA adapter,
directly on the compressed file you already run. Run the training twice and
you get the same adapter, byte for byte.

**How.**

```sh
./runner -m base-Q4_K_M.gguf --train data.jsonl --train-steps 200 \
  --lora-rank 8 --train-out adapter.gguf
./runner -m base-Q4_K_M.gguf --lora adapter.gguf --serve
```

On Qwen3-4B at Q4_K_M, exact tool calls on a held-out set went from 0.69 to
1.00, and stock llama.cpp scores the same adapter at the same 1.00. Serve the
adapter with `--lora`, or merge it into a Q8_0 or F16 file.
[Walkthrough](docs/train-lora-on-quantized-gguf.md) ·
[Supported architectures](MANUAL.md#adaptation)

<a id="record-and-verify-a-run"></a>
### Record and verify a run

**Why.** A chat log is only text, and anyone could have typed it. If you have
to show what a model really said, you need a record that someone else can run
again and get the same answer. Runner writes that record, signs it, and
replays it.

**How.**

```sh
./runner -m model.gguf -p "Say hello in one sentence." -s 42 --transcript run.json
./runner -m model.gguf --verify run.json     # VERIFIED, DIVERGED or UNVERIFIABLE
```

`--keygen` and `--sign-key` sign a record, and `--export-pack` gathers a
server's receipts into one folder a reviewer checks offline with
`--check-pack`. `scripts/audit-demo.sh` runs the whole chain in under a
minute on an 8 GB Mac.
[Details](MANUAL.md#record-and-verify-a-run) ·
[What is and is not promised](docs/determinism-scope.md)

<a id="serving-and-apis"></a>
### Serving and APIs

**Why.** Most apps and coding agents already speak the OpenAI or Anthropic
API. Runner speaks both, so you point the app at your own machine and change
nothing else.

**How.**

```sh
./runner -m model.gguf --serve --parallel 2
```

Chat Completions, Responses, Completions, Embeddings, rerank and Anthropic
Messages are served on `http://127.0.0.1:8080`. OpenCode, Claude Code, Codex
CLI, Continue, Cline and pi are checked against it every release, and each
model family uses its own native tool format.
[Endpoints](MANUAL.md#endpoints) ·
[Coding-agent setup](MANUAL.md#coding-agent-evidence) ·
[Python client](python/README.md) ·
[TypeScript client](clients/typescript/README.md)

<a id="reasoning-budget"></a>
<a id="loop-guard"></a>
### Long agent turns that finish

**Why.** Some models think out loud before they answer, and sometimes they
keep thinking, or go round in circles, until the reply is used up. Runner can
cap the thinking and notice the circling, then steer the model to its answer
with the full reply budget left.

**How.**

```sh
./runner -m model.gguf --serve --reasoning-budget 512 --loop-guard
```

[Reasoning budget](MANUAL.md#reasoning-budget) ·
[Loop guard](MANUAL.md#loop-guard)

<a id="runtime-and-hardware"></a>
<a id="designed-to-stay-on"></a>
### Know what fits, share the GPU, give memory back

**Why.** Finding out that a model does not fit usually means downloading
20 GB and watching it crash. Runner reads the first few megabytes and tells
you. And because a model server usually sits next to your other work, Runner
lets several programs share one graphics card and hands memory back while it
is idle.

**How.**

```sh
./runner --fit model.gguf          # will it fit, from the header alone
./runner --caps                    # what this machine and build can run
./runner -m model.gguf --doctor    # load, probe and report
./runner -m model.gguf --serve --ttl 300 --wait-for-vram
```

Loaded and idle on an 8 GB M1, Runner holds 8 MB of wired memory where
llama-server holds 3,819 MB; with a 63 GB model on a 128 GB M5 Max it is
35 MB against 60.8 GB.
[Hardware and placement](MANUAL.md#runtime-and-hardware) ·
[Idle measurements](MANUAL.md#designed-to-stay-on)

<a id="shadow-mode"></a>
<a id="shadow-mode-what-could-your-local-model-have-done"></a>
### Shadow mode

**Why.** You already use a coding assistant. Shadow quietly retries your own
finished tasks with a local model, checks each attempt against your own
tests, and keeps score, so you know from evidence which jobs the local model
can take over.

**How.**

```sh
runner --shadow-mode -m ~/models/a.gguf     # asks first, then wires Claude Code and Codex
```

[How it works and what it stores](MANUAL.md#shadow-mode-what-could-your-local-model-have-done)

<a id="desktop-tray"></a>
### Desktop tray

**Why.** A server you cannot see is one you forget is running. On macOS and
Windows a tray icon shows which models are loaded and lets you start or stop
them.

**How.** It appears when you start a session; `./runner --tray` starts it on
its own. [Details](MANUAL.md#desktop-tray)

## More

- **Faster decoding with a draft:** `--draft small.gguf`, `--mtp` or
  `--draft-lookup`. [Details](MANUAL.md#runtime-and-hardware)
- **Smaller context memory:** `--kv q8`, `k8v4` or `fp4`, and long contexts
  with YaRN. [Details](MANUAL.md#long-contexts)
- **Pause and resume a generation:** session images with `--sessions DIR`.
  [Details](docs/session-images.md)
- **Run it in a container:** `ghcr.io/joakimpalm-zen/xyntetik-runner`.
  [Details](MANUAL.md#container-image)
- **What loads:** 19 architectures and 24 weight formats.
  [Support matrix](MANUAL.md#support-matrix)
- **Every flag and endpoint:** [command-line reference](MANUAL.md#command-line-reference),
  [APIs](MANUAL.md#serving-and-apis)
- **How Runner compares:** closer to the publisher's reference than
  llama.cpp on 21 of 22 measured rows, and level or ahead on the reference's
  chosen token on 20 of 22. [Evidence](MANUAL.md#evidence-and-tradeoffs) ·
  [Speed tables](docs/benchmarks.md)

<a id="build-and-platforms"></a>
## Build from source

```sh
git clone https://github.com/Joakimpalm-Zen/xyntetik-runner
cd xyntetik-runner
make
./runner --version   # -> runner 1.0.2
```

Linux, macOS and Windows. NVIDIA GPUs need a driver with CUDA 13.0 support
or newer (the R580 series); no CUDA toolkit is needed.
[Builds and platforms](MANUAL.md#build-and-platforms)

## Support the project

Xyntetik Runner is independent, built in Sweden, and **free forever under
Apache 2.0**. If it or one of its files is useful to you, a contribution
funds the hardware and measurement time behind the next one: one-off, through
Stripe, with no tiers and no gated features.
[What contributions enable](https://xyntetik.com/support/)

<p align="center">
  <a href="https://buy.stripe.com/9B69AUddpdx9auHgP27N600"><img src="site/assets/support-button.svg" alt="Support this work" height="44"></a>
</p>

Ran a file or the engine on your own hardware? Open a
[result report](https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=result_report.yml);
success and failure are both wanted, and each is credited by name in
[community results](docs/community-results.md). For a problem, include
`runner --version`, `runner --caps`, the model's exact filename and the load
log. [SECURITY.md](SECURITY.md) has the threat model and
[CONTRIBUTING.md](CONTRIBUTING.md) the correctness gates.

## License

[Apache 2.0](LICENSE)

Third-party code: `src/ed25519.c` is the signing subset of TweetNaCl
(Bernstein, van Gastel, Janssen, Lange, Schwabe, Smetsers), placed in the
public domain by its authors; the file header records what was kept and what
was changed. `src/mldsa/` is the ML-DSA-44 (FIPS 204) reference implementation
from pq-crystals/dilithium (public domain / CC0, also Apache 2.0), pinned to the
44 parameter set with two recorded changes: keys derive from a caller-supplied
seed and signing is deterministic (`src/mldsa/sign.c` marks both).
