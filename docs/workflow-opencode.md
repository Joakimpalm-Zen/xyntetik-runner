# One tested workflow: OpenCode against a local model

This page is one configuration that was run end to end, with its versions,
its measured limits and the report to produce when it does not work. It is
not a claim about other models, clients or machines; those have their own
rows in [agent-compatibility.md](agent-compatibility.md).

## The configuration

| | |
|---|---|
| Client | OpenCode 1.18.31 |
| Model | `ibm-granite/granite-4.2-8b-GGUF`, `Q4_K_M` (4.9 GB), SHA-256 `16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311` |
| Machine | Windows 11, RTX 3070 8 GB, CUDA driver R580 or newer |
| Context | 16,384 tokens, set on the server and declared to the client |

OpenCode sends a system prompt and tool declarations of about 7,900 tokens
before your first word. That fixes the two limits of this workflow: the
context has to hold it (8,192 does not leave room for a file), and the first
request is mostly prefill.

## Steps

1. Download and verify Runner as in the README's
   [quick start](../README.md#quick-start).

2. Ask whether it fits before downloading it. `--fit` needs only the file's
   header, so a ranged read of the first 16 MiB is enough:

   ```sh
   curl -r 0-16777215 -L -o head.gguf \
     https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF/resolve/main/granite-4.2-8b-Q4_K_M.gguf
   ./runner --fit head.gguf -c 16384
   ```

   On the RTX 3070 above it reads `FITS — 35 of 40 layers on the GPU`, and the
   `split` line shows that `--kv q8` would put 39 of 40 there. A `PAGES`
   verdict means the layers left on the CPU would be read from disk on every
   token: pick a smaller file or a shorter context first. Ask before starting
   a server, because a running one holds the memory being measured.

3. Serve the model. The first start downloads the file, checks it against
   the Hub's SHA-256 record and prints where it was stored:

   ```sh
   ./runner -hf ibm-granite/granite-4.2-8b-GGUF:Q4_K_M --serve -c 16384
   ```

4. Copy [`examples/opencode/`](../examples/opencode) to an empty directory.
   `opencode.json` names the server, the model file and the same limits the
   server was started with; `NOTES.md` is a file with a known word in it.

5. In that directory:

   ```sh
   opencode run "Read NOTES.md and reply with the sentinel word it contains."
   ```

   The expected transcript shows a `Read NOTES.md` step and an answer that
   contains `ORANGE-7319`. The word is in the file and nowhere in the prompt,
   so an answer that has it read the file through the client's own tool.

6. Stop the server with Ctrl-C.

## What was measured

On the machine above, 2026-09-15, Runner 0.5.3 (the same loop with a
different fixture file): the task completed in 236 s and 276 s over two
runs. Per request the server reported up to 8,305 prompt tokens; after the
first full request every later one reused all but about 130 of them, and
generation ran at 14 tok/s with 39 of 40 layers on the GPU. The records are
`docs/cross-family-remedy-evidence/opencode-1.18.31-granite42-8b-q4_k_m-windows-rtx3070.txt`
and `agent-client-sweep-granite42-8b-windows-rtx3070-2026-09-15.json` beside
it.

Limits that were observed, not predicted:

- The model sometimes guesses a path before it looks (`/workspace/...`), gets
  a "file not found" from the client and then lists the directory. That costs
  a turn, not the task.
- OpenCode gives up on a request whose prefill runs past roughly 290 seconds
  and sends it again. Runner continues the second request from what the first
  computed, but on a machine that prefills under about 30 tok/s the first
  turn is that slow.
- An 8 GB Apple Silicon Mac with other applications open did not hold a
  3.4 GB model and a 16,384-token context in memory (2026-10-02: `--fit`
  said `PAGES`, and the loop ran at 0.1 tok/s). On that class of machine a
  3B model at four bits is what fits, and a 3B model completed this task in
  one of two attempts: the second time it invented a path and stopped. That
  is the model's reliability, and the reason this page pins an 8B model on a
  GPU.

## When it does not work

Run the diagnostic with the same model and flags, and read the findings:

```sh
./runner -hf ibm-granite/granite-4.2-8b-GGUF:Q4_K_M -c 16384 --doctor > doctor.json
```

The report says where the layers ran and why, which template and sampler
were in effect, memory and timings, and one next step per finding. It holds
no prompt or reply text unless you add `--doctor-include-text`. Read it
before you attach it to an issue.

Three things it separates:

- **Slow**: the `placement` block shows layers on the CPU that you expected
  on the GPU, or the `memory` block shows less available RAM than the model
  needs (`--fit` gives the arithmetic).
- **Tool calls arrive as text**: the `template` block shows whether the
  server recognised the model's family (`recognised`) and which tool protocol
  it will use (`tool_family`, `native_tool_protocol`).
- **Nonsense output or replies that never end**: the `template` block names
  the template that was applied, and the probe reports whether the model
  ended its turn; either is a finding with its own next step.

## Scope of the evidence

Everything above was run by the project on its own machines. Nobody outside
the project has reported reproducing it yet; when that happens it will be
recorded here as a separate line.
