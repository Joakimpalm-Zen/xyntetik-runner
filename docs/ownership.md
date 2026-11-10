# Who owns what in Runner (R18.2)

Every part of Runner belongs to one of seven components, and each component has a job. This page says
what Runner is for, which component owns which source files and public surfaces, and the rules between
components. It describes the code as it is at main `d0dfb547`. Nothing is moved or removed by it, and 1.x
behaviour stays as it is. Where a part is planned to leave, the page says so and nothing more.

## What Runner is

> A local GGUF runtime that runs, serves, adapts, quantizes, and verifies model execution under a measured,
> reproducible contract.

A capability belongs in Runner when it does one of five jobs:

1. **Run**: load a GGUF and run efficient local inference.
2. **Serve**: expose reliable agent-compatible APIs and lifecycle controls.
3. **Produce**: train LoRA adapters and quantize the files that are deployed.
4. **Control**: fit work to the hardware at hand and make placement explicit.
5. **Prove**: record, sign, replay and verify what was executed.

Tools that orchestrate other development tools, give Runner a desktop interface, or run research campaigns
use Runner through its stable interfaces instead of living inside it.

## The seven components

### 1. Runtime: run, serve and control

The frontends, the request policy above sampling, the serving lifecycle and the operational controls.

- **Sources:**
  - the CLI's run and interactive modes in `main.c`;
  - the server: `server.c`, `http.c`, `api_anthropic.c`, `api_responses.c`;
  - request policy: `completion.c` (tool calls, structured output, truncation closure, reasoning budget,
    loop guard), `template.c` (chat templates and the native tool grammars it binds), `schema.c`,
    `schema_tools.c`, `jsonmode.c`, `sample.c`;
  - the serving lifecycle: `registry.c` (residency and swapping), `session.c` (session images),
    `respstore.c` (the Responses store), `kvsnap.c` (KV snapshots);
  - runtime operations: `decide.c` (typed decisions and reranking), `clef.c` (the Jev/SystemOne contract);
  - control: `vramreg.c` (VRAM reservation), `hfhub.c` (Hub fetch with hash verification), and the fit,
    capability and doctor reports.
- **Public surfaces:**
  - every HTTP route, from `/v1/chat/completions`, `/v1/messages` and `/v1/responses` to `/health`,
    `/metrics`, `/unload` and the `/v1/runner/*` routes;
  - the CLI's run, serve and fit flags;
  - the Python client's `RunnerEndpoint`, `ManagedRunner`, `StartupLease`, `ServerLaunch` and
    `query_system_capabilities`;
  - the records `xyntetik.runner.session.v1` and `xyntetik.runner.kv_snapshot.v1`.
- **Primary:** run and serve a model, know whether it fits and where it runs, tools and structured output.
  **Advanced:** embeddings, scoring, reranking, typed decisions, named contexts, sessions, prefix reuse,
  session images and KV snapshots.

### 2. Execution: the engine and the backends

Turning tokens into logits on the CPU, CUDA or Metal, and the speed features that ride on it.

- **Sources:**
  - `engine.c` (the generation loop: prompt feeding, stops, constrained sampling) and `scheduler.c`
    (continuous batching);
  - the backends: `cuda.c`, `metal.m`, `gpu_none.c` and the kernels in `kernels*.cu`, `kernels*.cuh` and
    `kernels*.metal`;
  - `quants.c` (the CPU kernels; its quant math is shared with model production);
  - the forward math in `model.c` (see "Files that serve more than one component").
  - Draft models, prompt lookup and MTP run here, through the verify operation.
- **Public surfaces:**
  - the placement and thread flags (`--gpu`, offload, threads);
  - the speed and route switches (see [kernel routes](kernel-routes.md) and
    [speed defaults](speed-defaults.md));
  - the route detail a receipt records.
- **Primary:** the default routes. **Advanced:** every `RUNNER_*` route switch, which sits outside the
  compatibility promise ([versioning](versioning.md)).

### 3. Model substrate: the file, the tokenizer and the weights

What runtime and model production both stand on.

- **Sources:**
  - `gguf.c` (parsing, split files) and `tokenizer.c`;
  - the loader and the shared weights record in `model.c`;
  - adapter loading for inference (`model_lora_load` and per-request adapter sets);
  - `npy.c`, `json.c` and the platform layer `compat.c`, used by everything.
- **Public surfaces:** the model and adapter flags (`-m`, `--lora` and the swap list), model and adapter
  identity.
- **Primary:** loading a model and an adapter.

### 4. Model production: train and quantize

Making the files that are deployed. Training and quantization are two of Runner's principal features and
stay first in the README.

- **Sources:**
  - `quantize.c`: quantizing, type plans and their strict validation, adapter merges;
  - `derive.c`: `runner derive`, which makes a file from a parent, measures it and writes its card;
  - training: the train paths in `main.c`, LoRA and full-weight training in `model.c` (backward, AdamW),
    `dpo.c`;
  - expert pruning, sublayer removal and context surgery.
- **Public surfaces:**
  - `runner train`, `--rewrite` (with `--quant`, `--type-plan`, `--merge-lora` or `--context-surgery`),
    `--quant-error` and `runner derive`;
  - the records `xyntetik.runner.train.v1`, `merge.v1`, `quant.v1`, `derive.v1`, `eval.v1` and
    `context-surgery.v1`.
- **Primary:** train a LoRA through the quantized file, and quantize. **Advanced:** expert pruning,
  sublayer removal, context surgery, and building MTP companions (a script).

### 5. Proof: records, signatures and replay

Saying, in a form anyone can check, what was executed.

- **Sources:**
  - `receipts.c` (per-request receipts), `bundle.c` (receipt bundles), `provenance.c` (the server's
    identity);
  - `envelope.c` (measured envelopes), `lineage.c` (lineage records);
  - `oms.c` (OpenSSF model-signing verification) and the signature code in `ecdsa.c`, `ed25519.c` and
    `mldsa.c`;
  - transcripts and `--verify` replay;
  - `watermark.c`, whose future is its own question.
- **Public surfaces:**
  - `--verify` and its verdicts, receipts in `runner_telemetry`, `/v1/runner/provenance`, signing;
  - the records `xyntetik.runner.transcript.v1` and `v2`, `bundle.v1`, `envelope.v1`,
    `evidence-pack.v1`, `signkey.v1` and the two watermark-key records.
- **Primary:** prove a run with records, signatures and replay.

### 6. Shadow: a companion

Offloading work from Claude Code and Codex to a local model, with capture, routing and delegation on a
scratch worktree.

- **Sources:** the Python package `xyntetik_runner.shadow` (22 files) and `--shadow-mode` in `main.c`.
- **Boundary:** it uses Runner through HTTP, `/v1/capabilities` and receipts, like any client, and owns
  nothing inside the engine.
- **Disposition:** planned to leave Runner for a companion product. Whether `--shadow-mode` is deprecated
  during 1.x is an open decision (R18.8.7). Until it is decided, it works as documented.

### 7. Desktop: a companion

The menu-bar and taskbar tray and its autostart.

- **Sources:** `tray.c`, `tray_win.c`, `tray_macos.m`, `tray_stub.c`, and the instance registry
  `instances.c`, which every Runner process writes and the tray reads.
- **Boundary:** the tray starts, lists and stops Runner processes through public lifecycle surfaces: the
  instance records, `/health`, `/unload` and the startup lease shared with the Python client.
- **Disposition:** planned to leave Runner for a companion product once that companion matches it. 1.x
  behaviour stays.

## Files that serve more than one component

Three files carry more than one owner today. That is the coupling the refactor plan's later phases would
cut. This page records it and moves nothing.

- **`main.c`:** the command line for every component (run modes, training, quantizing, verify, shadow
  mode, the tray).
- **`model.c`:** the loader (model substrate), the forward math (execution), and LoRA and full-weight
  training (model production).
- **`completion.c`:** request policy (runtime), and the receipt recording it starts (proof).

## Rules between components

1. The server may depend on execution operations; execution code must not depend on HTTP types.
2. Model production may depend on GGUF, the tokenizer, quant math, adapters and proof; it must not depend
   on server slots, request queues, sessions or runtime KV state.
3. Proof observes execution through stable records; it never chooses kernels.
4. A backend owns its device state, declared once, not inferred across `model.c`, `engine.c` and the
   backend files.
5. A specialised fast path is admitted by explicit model geometry, quant type, device class, placement and
   sampling capability; a mismatch falls back before generation starts.
6. Canonical and fast execution share the public request semantics, and may carry different measured
   envelopes.

## Primary and advanced, in the README's order

1. Run and serve a model locally.
2. Know whether it fits and where it runs.
3. Produce the deployed file: train a LoRA, and quantize.
4. Tools and structured output.
5. Prove the run: records, signatures, replay.
6. Advanced operations, in the manual.
7. Shadow and the desktop tray, as companions, not reasons to choose Runner.

## The inventory

The [capability inventory](capability-inventory.md) lists every public surface with its owner: every CLI
option and subcommand, HTTP route, `RUNNER_*` switch, record format, Python and TypeScript client name, and
every file under `src/`. `scripts/capability-inventory.py` reads it from the source, and its `--check`
(run by `tests/test_doc_counts.py`) fails when a surface has no owner or the page is out of date. The
switches have their classes in the [deprecation review](deprecation-review-2026-11.md).
