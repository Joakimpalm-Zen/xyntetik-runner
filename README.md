# Xyntetik Runner

**Run AI models on your own machine, and know exactly what you are getting.**

## Why you would want it

Running a model yourself is mostly guesswork. Will it fit? Did shrinking it
break it? Why is my laptop out of memory when nothing is running? Runner
exists to replace those guesses with answers you can check.

- **Ask before you download.** Runner reads the first few megabytes of a
  model and tells you whether it fits your machine, and how. It called the
  real split in 70 of 72 tests and was never too optimistic.
- **Shrink a model and see what it cost.** One command and a one-line plan
  built a Qwen3-30B that is smaller than the official 4-bit file and closer
  to the original.
- **Read longer documents in the same memory.** The context cache fits in
  about 41% of the usual space, with full agreement on models from 4B to 30B.
- **Leave it running.** A 63 GB model sitting idle holds 35 MB of your
  memory. On the same Mac, llama-server holds 60.8 GB.
- **Trust what comes out.** A tool call still parses when the model runs out
  of words, a fine-tune comes out the same twice, byte for byte, and any run
  can be signed and replayed by someone else.

It takes about a minute to try: [quick start](#quick-start).

## How it is built

- **One file.** A single program written from scratch in plain C. No Python,
  no installer, nothing underneath it. Download it and run it.
- **Runs on what you have.** Ordinary processors, NVIDIA graphics cards and
  Apple Silicon, on Linux, macOS and Windows.
- **Speaks what your tools speak.** Standard GGUF model files and the OpenAI
  and Anthropic APIs, so your apps and coding agents connect without changes.
- **Measured, not assumed.** Twelve model families have been scored against
  their publishers' own implementations, and Runner is closer to the
  reference than llama.cpp on 21 of 22 rows. A model it does not know is
  refused by name, never guessed at.
- **Yours.** Free forever under Apache 2.0, built in Sweden, and your prompts
  never leave your machine.

<p align="center">
  <a href="https://buy.stripe.com/9B69AUddpdx9auHgP27N600"><img src="site/assets/support-button.svg" alt="Support this work" height="44"></a>
</p>
<p align="center">
  If a file here saved you memory or time, a contribution funds the hardware time behind the next one.<br>
  <a href="https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=model_request.yml">Request a model</a> ·
  <a href="https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=result_report.yml">Report a result</a> ·
  <a href="https://github.com/Joakimpalm-Zen/xyntetik-runner/issues/new?template=bug_report.yml">Report a bug</a> ·
  <a href="docs/community-results.md">Community results</a>
</p>

> **Version `1.0.2`.** The command line, the HTTP API, the record formats and
> the model files that load stay compatible across 1.x
> ([versioning policy](docs/versioning.md)). This page is the short version;
> every flag, endpoint, limit and measurement is in the [manual](MANUAL.md).

**On this page:** [Quick start](#quick-start) · [Model files](#files) ·
[Will it fit](#fit) · [Make your own file](#models-and-conversion) ·
[More context, less memory](#long-contexts) · [Idle memory](#designed-to-stay-on) ·
[Tool calls](#truncation) · [Training](#adaptation) ·
[Also in the box](#more) · [All commands](#all-commands) ·
[Prove what a model did](#record-and-verify-a-run)

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

<a id="files"></a>
<a id="published-artifacts"></a>
## Model files on Hugging Face

The files Xyntetik publishes live at
[huggingface.co/Joakimpalm-Zen](https://huggingface.co/Joakimpalm-Zen):
smaller and repaired versions of current open models, and the evidence
datasets behind them. Each card says what was changed, how the file measures
against its original, and where it runs. The
[Runner Releases collection](https://huggingface.co/collections/Joakimpalm-Zen/runner-releases-6aa98baaed03bba0e8a561ae)
is the place to start.

## Features

Each one starts with what it achieved, then why it exists, then how to use it.

<a id="fit"></a>
<a id="runtime-and-hardware"></a>
### Know whether a model fits before you download it

**Result.** On an RTX 3070, `--fit` named the same GPU and CPU split the
loader then used in 70 of 72 cases and was one layer cautious in the other
two, never optimistic (six models, two context sizes, four cache formats).

**Why.** Finding out that a model does not fit usually means downloading
20 GB and watching it crash. The answer is already written in the first few
megabytes of the file. Runner reads just that part and tells you, with the
arithmetic.

**How.**

```sh
# fetch only the first 16 MB of a model, then ask
curl -r 0-16777215 -L -o head.gguf \
  https://huggingface.co/ORG/REPO/resolve/main/MODEL.gguf
./runner --fit head.gguf -c 16384
```

```console
  gpu           NVIDIA GeForce RTX 3070, offload budget 6.95 GiB right now
  split         f16: 35 of 40 layers on the GPU, 1.22 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 0.47 GiB in RAM  | ...
  verdict       FITS — 35 of 40 layers on the GPU, 8.24 GiB of RAM to spare at ctx 16384
```

`./runner --caps` says what this machine and build can run, and
`./runner -m model.gguf --doctor` loads a model, probes it and reports.
[How the verdict is computed](MANUAL.md#deciding-before-you-download)

<a id="models-and-conversion"></a>
### Make your own model file

**Result.** A [17.99 GB Qwen3-30B file](https://huggingface.co/Joakimpalm-Zen/Qwen3-30B-A3B-selective-attnQ8_0-expQ4_0-GGUF)
came from one command and a one-line plan. It is smaller than the official
4-bit file and closer to the original: 99.50% agreement where that file
reads 94.75%.

**Why.** A model file is a huge table of numbers, and the usual way to shrink
it squeezes every part the same amount, including the parts that matter most.
Runner lets you choose which parts to squeeze and which to keep sharp, and
then tells you how far the result drifted from the original.

**How.** The plan that built that file, and the commands:

```json
{"default": "keep", "rules": [{"match": "_exps.weight", "type": "q4_0"}]}
```

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

<a id="long-contexts"></a>
### Fit more context in the same memory

**Result.** `--kv k8v4` stores the context cache in about 41% of the usual
bytes. On three models from 4B to 30B it agreed with the full-size cache on
100% of clear-cut tokens (500 positions each).

**Why.** While a model reads a long document it keeps notes on every word so
far, and on a long context those notes can take more memory than you have
left. Runner can write the notes smaller. It keeps 8 bits for the half that
decides where to look, which measurement showed is the half that matters,
and shrinks the other half to 4 bits.

**How.**

```sh
./runner -m model.gguf --serve -c 32768 --kv k8v4
./runner -m model.gguf --serve -c 32768 --kv q8     # the gentler setting, about half
```

`--fit` shows what each setting buys on your machine before you load anything.
[Cache formats and their measured cost](MANUAL.md#long-contexts)

<a id="designed-to-stay-on"></a>
### A server that gives your memory back

**Result.** A 63 GB model, loaded and idle on a 128 GB M5 Max, holds 35 MB of
wired memory under Runner and 60.8 GB under llama-server. On an 8 GB M1 it is
8 MB against 3,819 MB.

**Why.** A model server usually lives on the machine you also work on. Most
servers hold all of the model's memory until you quit them. Runner keeps the
model as memory the operating system may take back whenever another app needs
it, and reads it in again on the next request.

**How.**

```sh
./runner -m model.gguf --serve --ttl 300          # unload after five idle minutes
curl -X POST http://127.0.0.1:8080/unload         # or hand everything back now
./runner -m second.gguf --serve --port 8081 --wait-for-vram   # queue for a busy GPU
```

The first reply after the memory has been taken back waits for the model to
be read in again, about 3 s for the 3.6 GB model on that M1.
[Idle measurements](MANUAL.md#designed-to-stay-on) ·
[Sharing a GPU](MANUAL.md#resource-control)

<a id="truncation"></a>
### Tool calls that survive the token limit

**Result.** Six engines, one tool call cut short by the token limit. Runner
was the only one to return a call a program could run, at every budget from
1 to 16 tokens.

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
TensorRT-LLM and SGLang.
[Method and raw responses](docs/truncation-benchmark.md) ·
[Details](MANUAL.md#truncation)

<a id="adaptation"></a>
### Train the model file you actually serve

**Result.** Two training runs wrote the same adapter, checksum for checksum,
and an independent tester reproduced that on a Tesla T4. On Qwen3-4B at
4 bits, exact tool calls on a held-out set went from 0.69 to 1.00, and stock
llama.cpp scores the same adapter at the same 1.00.

**Why.** Teaching a model a new habit normally means training a large
full-precision copy and compressing it afterwards, so the model you tested is
not quite the one you ship. Runner trains a small add-on, a LoRA adapter,
directly on the compressed file you already run, with no Python stack.

**How.**

```sh
./runner -m base-Q4_K_M.gguf --train data.jsonl --train-steps 200 \
  --lora-rank 8 --train-out adapter.gguf
./runner -m base-Q4_K_M.gguf --lora adapter.gguf --serve
```

It is built for small, targeted datasets on dense models (Llama, Mistral,
Qwen2.5, Granite 4.x, Gemma 3 and 4). Serve the adapter with `--lora`, or
merge it into a Q8_0 or F16 file.
[Walkthrough](docs/train-lora-on-quantized-gguf.md) ·
[Supported architectures](MANUAL.md#adaptation)

<a id="more"></a>
## Also in the box

- <a id="serving-and-apis"></a>**OpenAI and Anthropic APIs.** Chat
  Completions, Responses, Completions, Embeddings and Anthropic Messages on
  `http://127.0.0.1:8080`, with each model family's own tool format.
  OpenCode, Claude Code, Codex CLI, Continue, Cline and pi are checked
  against it every release. [Endpoints](MANUAL.md#endpoints) ·
  [Coding-agent setup](MANUAL.md#coding-agent-evidence)
- <a id="structured-output"></a>**Structured output.** `--json`,
  `--json-schema FILE` or the API's `response_format` guarantee the shape of
  the answer, and `"confirm_below": 0.8` reports whether the model was sure
  enough to skip a human. [Details](MANUAL.md#structured-output)
- **Rerank without a second model.** `POST /v1/rerank` scores documents
  against a query with the model that is already loaded.
  [Details](MANUAL.md#endpoints)
- **A `/metrics` endpoint.** Prometheus counters for tokens, timings, memory
  and the prefix cache, always on with `--serve`.
  [Details](MANUAL.md#health-and-metrics)
- **Multi-file models load as they are.** A standard split GGUF set loads
  from any of its parts, with no merge step.
  [Details](MANUAL.md#models-and-conversion)
- **Expert layers in system RAM.** `--cpu-moe` keeps a mixture-of-experts
  model's expert layers off a small graphics card.
  [Details](MANUAL.md#placement-and-memory)
- <a id="reasoning-budget"></a><a id="loop-guard"></a>**Long thinking turns
  that finish.** `--reasoning-budget 512` caps how long a model thinks out
  loud and `--loop-guard` closes a turn that has started repeating itself.
  [Reasoning budget](MANUAL.md#reasoning-budget) ·
  [Loop guard](MANUAL.md#loop-guard)
- <a id="shadow-mode"></a><a id="shadow-mode-what-could-your-local-model-have-done"></a>**Shadow
  mode.** `runner --shadow-mode -m model.gguf` retries your own finished
  coding tasks with a local model under your own tests and keeps score.
  [Details](MANUAL.md#shadow-mode-what-could-your-local-model-have-done)
- <a id="desktop-tray"></a>**Desktop tray.** On macOS and Windows an icon
  shows which models are loaded and starts or stops them.
  [Details](MANUAL.md#desktop-tray)
- **Draft decoding.** `--draft small.gguf`, `--mtp` or `--draft-lookup`.
  [Details](MANUAL.md#runtime-and-hardware)
- **Pause and resume a generation.** Session images with `--sessions DIR`.
  [Details](docs/session-images.md)
- **Container image and clients.** `ghcr.io/joakimpalm-zen/xyntetik-runner`,
  a [Python client](python/README.md) and a
  [TypeScript client](clients/typescript/README.md).
  [Container](MANUAL.md#container-image)
- **What loads.** 19 architectures and 24 weight formats, verified on
  download with `-hf`. [Support matrix](MANUAL.md#support-matrix)
- **How Runner compares.** Closer to the publisher's reference than
  llama.cpp on 21 of 22 measured rows, and level or ahead on the reference's
  chosen token on 20 of 22. [Evidence](MANUAL.md#evidence-and-tradeoffs) ·
  [Speed tables](docs/benchmarks.md)

<a id="all-commands"></a>
## All commands

**The complete list of Runner's commands and flags is the
[command-line reference](MANUAL.md#command-line-reference)**, and every
endpoint is in the [API reference](MANUAL.md#serving-and-apis).
`./runner --help` prints the same list.

<a id="record-and-verify-a-run"></a>
## Prove what a model did

**Result.** `scripts/audit-demo.sh` signs a 1,000-token record, replays it
and refuses three forgeries, in under a minute on an 8 GB Mac.

**Why.** A chat log is only text, and anyone could have typed it. If you have
to show what a model really said, you need a record that someone else can run
again and get the same answer. Runner writes that record, signs it, and
replays it.

**How.**

```sh
./runner -m model.gguf -p "Say hello in one sentence." -s 42 --transcript run.json
./runner -m model.gguf --verify run.json     # VERIFIED, DIVERGED or UNVERIFIABLE
```

Around that core: signed and chained receipts (`--keygen`, `--sign-key`),
evidence packs a reviewer checks offline (`--export-pack`, `--check-pack`),
signature checks on the model file itself (`--model-sig`), and a text
watermark with its detector (`--watermark`, `--detect-watermark`).
[Details](MANUAL.md#record-and-verify-a-run) ·
[What is and is not promised](docs/determinism-scope.md)

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
