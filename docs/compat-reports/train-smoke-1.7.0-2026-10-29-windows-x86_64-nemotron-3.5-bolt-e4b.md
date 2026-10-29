# Training smoke: Nemotron 3.5 Bolt E4B on the Windows CUDA host (1.7.0, 2026-10-29)

A 20-step LoRA run on the GPU, to show that `runner train` takes this file and that the adapter it writes changes
the answer. It is a smoke test, not a fidelity claim, and it does not change the row's envelope.

## Command

```
runner train train.jsonl -m Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --train-gpu --steps 20 --ctx 128 --lora-rank 8 --out <scratch>/bolt-smoke-adapter.gguf
```

- Runner 1.7.0, built on this host from the branch `model/nemotron-bolt-e4b` (binary sha256 `e0757cec…`).
- Base file: `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`, sha256 `c8af2b8a…` (the row's file).
- Data: 12 `{"prompt", "completion"}` lines, "What is the capital of X?" → " City." for France, Japan, Italy,
  Spain, Canada, Egypt, Kenya, Peru, Norway, Chile, Greece and Poland (sha256 `b0e7a8eb…`). No end-of-turn token,
  so the trainer warns that the adapted model keeps generating after the answer.
- Recipe from the provenance file: rank 8, alpha 16, lr 1e-4 constant, AdamW, batch 1, seed 0, 10 tokens per step.

## Device and budget

NVIDIA GeForce RTX 3070 (8 GB), driver 596.36, Windows. `--train-gpu` reported a budget of 6.9 GB (the free
memory, 6.2 GB reserved), every weight product on the GPU, and weights cached on first use.

## Result

| | |
|---|---|
| Step-0 loss (step 1) | 8.1918 |
| Last loss (step 20) | 0.0030 |
| Seconds per step | 0.65 (steps 2–20; step 1 took 1.74 s with the weight upload) |
| Wall time | 34.9 s, model load included |
| Adapter | 73,440,512 bytes, 294 adapted projections, sha256 `615b5d9e…` |

## Answer with the adapter

`runner -m Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --lora bolt-smoke-adapter.gguf -p "What is the capital of Sweden?" -n 16 --temp 0`
(Sweden is not in the data):

```
What is the capital of Sweden? Stockholm. Stockholm is a major city in Sweden, known for its beautiful archipelago,
```

The adapter loaded on the GPU for 42 blocks (73.4 MB of VRAM). The same command without `--lora` produced no
continuation: the base model ends the raw prompt at once. A training row, Norway, gives " Oslo." with the adapter.
The text after the answer is the run-on the trainer warned about.
