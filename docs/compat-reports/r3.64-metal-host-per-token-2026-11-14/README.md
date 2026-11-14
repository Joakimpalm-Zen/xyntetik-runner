# What crosses to the host per generated token on a Metal decode (R3.64), 2026-11-14

**Finding: on the Apple GPU, the time outside the device forward is 0.04 to 0.36 ms of a 25 to 64 ms
token, 0.1 to 0.7%.** That is below the item's 5% kill line, on a greedy and a sampled decode of two
dense files, so the Metal half closes with this profile as its record. Counting the host's share
inside the forward as well (encoding the command buffer, and the wait past the GPU's own time), the
host holds about 2% at most. The CUDA half was not measured: the CUDA host was down on the day. It
is the follow-up below, not a guess.

## What crosses, by the code

Per generated token on Metal (`src/metal.m`):

- the host encodes one command buffer for the whole forward (`metal_fwd_submit`), commits it and waits
  for it (`waitUntilCompleted`);
- the logits are not copied: the forward returns a pointer into the shared-storage logits buffer
  (`return (float *)g->logits.contents + ...`), which the sampler reads in place (unified memory);
- the sampler, stop checks and the streamed write then run on the host, outside the forward.

Each forward also reads two or three environment switches (`RUNNER_METAL_STATS`, the MoE route and
full traces) with `getenv`. That is a few microseconds a token, under 0.01%, and is left as it is.

## Measurement

An Apple-silicon Mac with 8 GB, the Runner build of this branch (STRICT, Metal). One warm-up run per file,
then greedy and sampled runs interleaved, 96 generated tokens each with `--ignore-eos`; the first two
decode forwards of each run are left out. `RUNNER_METAL_TIMING=1` prints each forward's CPU encode
time, its submit-and-wait time and the GPU's own time (`GPUEndTime - GPUStartTime`); the token's
time is 1000 / the run's generation rate.

| file | decode | token (ms) | forward (ms) | of which CPU encode | wait past GPU time | outside the forward |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B Q4_K_M | greedy | 61.46 | 61.37 | 0.22 | 0.31 | 0.09 ms, 0.1% |
| Qwen3-4B Q4_K_M | sampled | 62.38 | 62.23 | 0.21 | 0.31 | 0.16 ms, 0.3% |
| Qwen3-4B Q4_K_M | greedy | 63.57 | 63.38 | 0.22 | 0.34 | 0.19 ms, 0.3% |
| Qwen3-4B Q4_K_M | sampled | 60.98 | 60.61 | 0.21 | 0.30 | 0.36 ms, 0.6% |
| SmolLM2-1.7B-Instruct Q4_K_M | greedy | 26.83 | 26.80 | 0.16 | 0.31 | 0.04 ms, 0.1% |
| SmolLM2-1.7B-Instruct Q4_K_M | sampled | 25.99 | 25.81 | 0.12 | 0.27 | 0.18 ms, 0.7% |
| SmolLM2-1.7B-Instruct Q4_K_M | greedy | 25.35 | 25.32 | 0.12 | 0.26 | 0.04 ms, 0.2% |
| SmolLM2-1.7B-Instruct Q4_K_M | sampled | 25.52 | 25.37 | 0.12 | 0.27 | 0.15 ms, 0.6% |

The smaller file is the harder case for this question (a shorter forward makes the host's share
larger), and it still reads under 1%. Sampling costs 0.1 to 0.3 ms more than greedy.

Commands, per file `F` and decode arm (greedy `--temp 0`, sampled `--temp 0.8 --top-p 0.9`):

```sh
./runner -m F -p "Write a short paragraph about the history of the bicycle." -n 8 --temp 0 --no-tray   # warm-up
RUNNER_METAL_TIMING=1 ./runner -m F -p "Write a short paragraph about the history of the bicycle." \
  -n 96 ARM --ignore-eos --no-tray > /dev/null 2> run.err
python3 parse-timing.py "F ARM" run.err
```

## Limits

- Dense files only. No MoE file small enough for this 8 GB machine was at hand; a MoE decode adds a
  router step per layer inside the forward, which this split would count as forward time.
- One machine. Not measured: grammar-constrained decoding, speculative drafts, and the server's
  streaming write, which a `--serve` run adds after each token.

## Follow-up: the CUDA half

On CUDA the logits are copied to the host after each forward (`cuMemcpyDtoH` of the logits row) and
a decode token's KV rows come back to the host cache (the KV gather, `RUNNER_CUDA_KV_GATHER`). Those
are the two transfers to measure there, with `RUNNER_CUDA_PROFILE`'s exit table (GPU time per stage,
host-side copies, launch overhead), on the same two arms and the 4B file, when a CUDA host is up.
