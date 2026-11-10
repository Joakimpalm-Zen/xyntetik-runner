# The train/serve mismatch, measured: Runner against vLLM and transformers, 2026-11

A reinforcement-learning loop usually samples rollouts with one engine and trains on them with another. The two rarely agree
on the probability of the tokens they share, and the training math assumes they do. This page measures that
disagreement on one small model.
- **Runner:** its sampler and its trainer, which share one forward.
- **A common Python stack:** vLLM sampling, transformers with PEFT training, in bf16 and in a QLoRA-style 4-bit
  setting.

## What this page does not claim

- **Runner is exact only on the CPU, with the prompt read one token at a time** (`-b 1`, the path its trainer runs).
  - Its default batched prompt pass is close but not exact: it matches the trainer on about a fifth of the tokens,
    and the largest difference is 1.45e-2.
  - Its CUDA sampler is not exact either: largest difference 2.02e-2. Its rollouts also drift from the CPU
    sampler's: 284 of 384 are the same (see below).
- **The stacks do not run the same weights.** Runner runs the publisher's Q4_K_M GGUF. vLLM runs the publisher's bf16
  safetensors, and an NF4 quantization of them made on the fly. Each stack is compared only with itself, its sampler
  against its trainer, on its own rollouts. No row says one engine computes the model better than another.
- **The prompts differ by one token.** The GGUF says not to add a BOS token, and Runner follows it. The bf16
  checkpoint's tokenizer prepends `<s>` to raw text. Each trainer scores exactly what its own sampler saw.
- **One model, one task, one seed set:**
  - MiniCPM5-1B, four tool-call prompts from `scripts/train-grpo-lite.py`, 96 seeds each, at temperature 0.8;
  - 384 rollouts per row, ending at EOS or 64 tokens;
  - no significance test, so differences of under about 2x between rows are not read as findings.
- **No training happened.** Every trainer ran at learning rate 0, scoring the sampled tokens under its starting
  weights. What a mismatch of this size does to a training run, and to the reward, is not measured here.
- **A deviation was found and repaired.** In the first run, Runner's sampler drew from a truncated distribution
  (top_p 0.95, top_k 40, min_p 0.05). Its requests set only the temperature, and the server fills the rest from its
  preset. GRPO-lite's requests do the same. The vLLM samplers drew untruncated, as the plan said.
  - It was found after the vLLM rows were read.
  - The vLLM samplers were then rerun with Runner's values, and Runner with truncation off in every request.
  - Both pairs are below, so every comparison is like for like.

## What was measured

For each sampled token, d is the sampler's raw logprob minus the trainer's raw logprob of the same token, with the
same prefix and temperature 1 on both sides. The table gives:
- mean and largest |d|;
- the tokens with d exactly 0;
- the share of tokens whose ratio exp(d) falls outside [0.8, 1.25], the interval PPO-style clipping uses.

**The arms:**
- **Runner:** `runner --serve` sampling, `runner --train ... --lr 0 --train-logprobs` scoring the sampled token ids,
  both on the CPU, with three samplers:
  - (a) CPU, `-b 1`;
  - (b) CPU, the default batched prompt pass;
  - (c) CUDA.
- **vLLM bf16:** the sampler is vLLM 0.26.0 in bf16 with `logprobs_mode="raw_logprobs"`, in its default compiled
  form. The trainer is transformers 5.14.1 with a PEFT LoRA of rank 8 that adds exactly 0, in train mode with
  autograd on, bf16.
- **vLLM NF4:** the same sampler with vLLM's on-the-fly bitsandbytes NF4. It is scored by three trainers:
  - the bf16 trainer above (QaRL's setting: quantized rollouts, a full-precision trainer);
  - transformers NF4 QLoRA without double quantization;
  - transformers NF4 QLoRA with double quantization, which is what vLLM's on-the-fly NF4 uses.

## Results

**Untruncated draw (temperature 0.8, no top-p, top-k or min-p):**

| sampler | trainer | tokens | d == 0 | mean abs d | largest abs d | outside [0.8, 1.25] |
|---|---|---|---|---|---|---|
| Runner, CPU `-b 1` | Runner, CPU | 18,178 | 18,178 | 0 | 0 | 0 |
| Runner, CPU batched | Runner, CPU | 18,014 | 3,392 | 3.91e-4 | 1.45e-2 | 0 |
| Runner, CUDA | Runner, CPU | 18,056 | 2,371 | 9.77e-4 | 2.02e-2 | 0 |
| vLLM bf16 | transformers bf16 | 21,217 | 108 | 1.82e-2 | 0.510 | 0.61% |
| vLLM NF4 | transformers bf16 | 20,463 | 7 | 0.387 | 12.2 | 30.96% |
| vLLM NF4 | transformers NF4, no double quant | 20,463 | 102 | 2.66e-2 | 1.07 | 1.47% |
| vLLM NF4 | transformers NF4, double quant | 20,463 | 88 | 1.92e-2 | 0.599 | 0.49% |

**Truncated draw (top_p 0.95, top_k 40, min_p 0.05, Runner's preset), the first Runner run:**

| sampler | trainer | tokens | d == 0 | mean abs d | largest abs d | outside [0.8, 1.25] |
|---|---|---|---|---|---|---|
| Runner, CPU `-b 1` | Runner, CPU | 18,077 | 18,077 | 0 | 0 | 0 |
| Runner, CPU batched | Runner, CPU | 18,010 | 3,844 | 3.13e-4 | 9.75e-3 | 0 |
| Runner, CUDA | Runner, CPU | 17,974 | 2,750 | 8.09e-4 | 2.18e-2 | 0 |
| vLLM bf16 | transformers bf16 | 19,859 | 79 | 1.42e-2 | 0.525 | 0.38% |
| vLLM NF4 | transformers bf16 | 20,627 | 0 | 0.289 | 12.3 | 25.39% |
| vLLM NF4 | transformers NF4, no double quant | 20,627 | 108 | 2.01e-2 | 0.631 | 0.93% |
| vLLM NF4 | transformers NF4, double quant | 20,627 | 104 | 1.40e-2 | 0.479 | 0.24% |

**Alignment check:** every trainer scores its own sampler's prompt and token ids, and the lengths match on every
rollout. Reading the trainer one token late raises the median |d| 48 times or more on every arm except the NF4
sampler with the bf16 trainer, whose own mismatch is large (4 to 5 times there). So the positions line up.

**Runner's other samplers against its CPU `-b 1` sampler,** rollout by rollout, untruncated:
- **CUDA:** 284 of 384 rollouts are token-identical. The first divergence is at token 5 at the earliest and 28 at the
  median. On the prefixes they share, the samplers' logprobs differ by at most 0.0202.
- **Batched:** 325 of 384 are identical (first divergence 14 at the earliest, 32 at the median), within 0.0145.
- In the truncated run, CUDA was identical on 345 and batched on 369.

## Reading the tables

- **Runner on the CPU with `-b 1` is exact:** every sampled token's logprob equals the trainer's, so the ratio is
  exactly 1. Its other two samplers are close, with no token outside the clipping interval, but they are not exact.
  A loop that needs exactness reads its prompts with `-b 1` on the CPU, as `scripts/train-grpo-lite.py` does.
- **A bf16 vLLM sampler and a bf16 transformers trainer** on the same checkpoint put 0.61% of tokens outside the
  clipping interval, with |d| up to 0.51. That is the gap importance-sampling corrections in published RL
  work are built for.
- **Quantized sampling needs a matching trainer.** An NF4 sampler scored by a bf16 trainer puts a quarter to a third
  of all tokens outside the interval. An NF4 trainer with the sampler's own double quantization brings that back to
  the bf16 stack's level. Without double quantization it is three to four times higher.

## How it was run

**Runner** (1.15.0 at main 3af91005d3, gcc 13.3.0):

```sh
# sampler: one server per arm; (a) as shown, (b) without -b 1, (c) without --gpu off, -t 40 and -b 1
./runner -m MiniCPM5-1B-Q4_K_M.gguf --serve --port 8080 -c 512 --gpu off -t 40 -b 1
# each rollout: POST /v1/completions
#   {"prompt": ..., "max_tokens": 64, "temperature": 0.8, "seed": S, "logprobs": 1,
#    "top_p": 1, "top_k": 0, "min_p": 0}
# trainer: the sampled ids as {"prompt_tokens": [...], "completion_tokens": [...], "weight": 1}
./runner -m MiniCPM5-1B-Q4_K_M.gguf --train ids.jsonl --train-steps 384 --lr 0 \
  --train-out lr0.gguf --train-logprobs trained.jsonl --lora-rank 8 --gpu off -t 40 -c 512
```

Leave out `top_p`, `top_k` and `min_p` and the server applies its preset. A response's
`runner_telemetry.sampling.source` names where each value came from.

**vLLM and transformers:**
- **Sampler:** `LLM(model, dtype="bfloat16", logprobs_mode="raw_logprobs")`, adding `quantization="bitsandbytes"`
  for NF4. Each request uses `SamplingParams(temperature=0.8, top_p=1.0, top_k=0, max_tokens=64, logprobs=0,
  seed=S)`.
- **Trainer:** `AutoModelForCausalLM` in bf16, or with `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=...)` and
  `prepare_model_for_kbit_training`. Then `get_peft_model` with LoRA r 8 on q/k/v/o, `model.train()`, micro-batches
  of 8, logits cast to float32 for `log_softmax`.
- **Versions:** torch 2.11.0+cu130, vLLM 0.26.0, transformers 5.14.1, PEFT 0.21.2, bitsandbytes 0.50.2.

**Host:** CUDA MIG slice, 1g.24gb, sm_120. The CPUs are shared.

**Files:** openbmb/MiniCPM5-1B@ac83f813ba7d (bf16) and openbmb/MiniCPM5-1B-GGUF@44a42924449a
(MiniCPM5-1B-Q4_K_M.gguf, sha256 81b64d05...).

**The records** (plan with every amendment, scripts, per-token rows and statistics) are in the project's research
record, research/runner-engine/r1752-train-serve-mismatch-2026-11-10, at commits b04e1f4f2 (the truncated run, and the
vLLM rows) and d10c56d1f (the untruncated Runner run).

## What changes elsewhere

MANUAL.md, docs/lora.md and docs/kernel-routes.md said the batched prompt pass moves the logprobs after it "up to
3.1e-3 on MiniCPM5-1B Q4_K_M after a 60-token prompt". That was measured on 2026-10-17, on the file before its
2026-11-09 update. On the updated file, over 384 rollouts after prompts of 49 to 51 tokens, the largest move is
1.45e-2. They now say so and link here.
