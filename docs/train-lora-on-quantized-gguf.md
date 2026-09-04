# Train a LoRA directly on the quantized GGUF you serve

**Question.** Can I train a LoRA adapter on a quantized GGUF (Q4, Q8) without
keeping a separate FP16 copy of the model?

**Answer.** Yes. Runner trains LoRA adapters through the frozen quantized GGUF
used for inference. The serving forward pass is the training forward pass,
so there is no second numerical path for the policy to drift across, and the
adapter is written beside a provenance record (base, data and adapter sha256,
seed, full config).

## The measured claim

Measured 2026-08-22 on a public artifact you can download and reproduce,
[Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M):

| | result |
|---|---|
| base | Qwen3-4B Q4_K_M, the frozen 4-bit serving weights |
| training path | directly through the quantized inference artifact, CPU |
| held-out tool calling, exact call | 0.69 before, 1.00 after |
| reproducibility | two independent runs, same sha256 on the adapter file |
| precision study | adapters trained through BF16 and Q8_0: cosine 0.9998; through Q4_K_M: 0.9926, a measurably different object that is capability-equivalent on this task's supervised decisions |
| neutral-corpus drift | nll per token 4.063 to 4.026, unrelated text left alone |
| interop | the adapter scores the same 1.00 served by stock llama.cpp |

The claim scope is one base and one task. The 8B and 14B ladder is open
work, and the numbers above are not a promise about other families.

## The caveat that travels with the claim

`--merge-lora` folds an adapter into the base to produce a standalone GGUF.
Merging into Q8_0 or F16 keeps the 1.00 (verified in stock llama.cpp).
Merging into the 4-bit base erases the fine-tune: the score returns to 0.69
because most of the delta rounds back to the base's own quantization grid.
Serve adapters with `--lora` (the exact form) or merge into 8-bit or better.

## Commands

```sh
./runner -m base-Q4_K_M.gguf --train data.jsonl --train-steps 200 \
  --lora-rank 8 --train-out adapter.gguf
./runner -m base-Q4_K_M.gguf --lora adapter.gguf --serve
./runner -m base-Q4_K_M.gguf --lora adapter.gguf --merge-lora merged-Q8_0.gguf --quant q8_0
```

Flag reference: `--train`, `--train-steps`, `--lr`, `--train-ctx`,
`--train-out`, `--save-every`, `--lora-rank`, `--lora`, `--lora-scale`,
`--merge-lora`, `--score` in the [command-line reference](../README.md#command-line-reference).

## Where the numbers come from

Design, gates, failure modes and every number above:
[docs/adaptation-engine.md](adaptation-engine.md). Reproducibility scope,
with what is and is not claimed: [docs/determinism-scope.md](determinism-scope.md).
