# LoRA adapters in Runner: the complete guide

A LoRA adapter is a small set of extra weights that teaches an existing model
a new habit, such as a tool-call format, a house style or a domain's vocabulary,
without rewriting the model. Runner trains adapters, serves them, merges them,
signs them and traces where they came from, all with the one binary that runs
the model.

What makes Runner's approach different is that it trains the adapter
**through the quantized GGUF file you serve**, a 4-bit or 8-bit file, with no
16-bit training copy, no Python framework and no conversion afterwards. The
forward pass that trains is the forward pass that serves. Training is
deterministic: the same data, seed, configuration and build write the same
adapter file, byte for byte, and every adapter is written beside a record of
exactly what made it.

This page explains every adapter feature, in the order you meet them. The
command-line reference is in the [manual](../MANUAL.md#command-line-reference);
the design, numerical contracts and gates are in
[adaptation-engine.md](adaptation-engine.md).

## Contents

- [Every feature at a glance](#every-feature-at-a-glance)
- [What it is for, and what it costs](#what-it-is-for-and-what-it-costs)
- [1. Check that a model can take an adapter](#1-check-that-a-model-can-take-an-adapter)
- [2. Prepare the training data](#2-prepare-the-training-data)
- [3. Train](#3-train)
- [4. Continue, checkpoint and resume](#4-continue-checkpoint-and-resume)
- [5. Preference training (DPO)](#5-preference-training-dpo)
- [6. Distillation toward a parent model](#6-distillation-toward-a-parent-model)
- [7. Reinforcement fine-tuning (GRPO-lite)](#7-reinforcement-fine-tuning-grpo-lite)
- [8. Evaluate an adapter](#8-evaluate-an-adapter)
- [9. Serve one adapter](#9-serve-one-adapter)
- [10. Serve many adapters, chosen per request](#10-serve-many-adapters-chosen-per-request)
- [11. Merge an adapter into a standalone file](#11-merge-an-adapter-into-a-standalone-file)
- [12. Records, signatures and lineage](#12-records-signatures-and-lineage)
- [13. Receipts and replay with an adapter](#13-receipts-and-replay-with-an-adapter)
- [14. Working with other tools](#14-working-with-other-tools)
- [15. What the trainer covers, and what it refuses](#15-what-the-trainer-covers-and-what-it-refuses)
- [16. Determinism, and where it stops](#16-determinism-and-where-it-stops)
- [17. Measured results](#17-measured-results)
- [18. Reference](#18-reference)

## Every feature at a glance

| Feature | What it does for you | How |
| --- | --- | --- |
| Admission check | Says whether this model can serve and train an adapter, and why not | `--adapt-info` |
| Train from text | Fits an adapter to a plain-text corpus | `--train corpus.txt` |
| Train from prompt and completion pairs | Learns the completions only; the prompt is context | `--train data.jsonl` |
| Weighted examples | Weights each example, negative weights included | `"weight"` on a line |
| Train on token ids | Trains on exact token ids, with no re-tokenization | `"prompt_tokens"`, `"completion_tokens"` |
| Teach the model to stop | Adds the end-of-turn token as a target | `--train-eot`, `"end_of_turn"` |
| Memorization warnings | Warns before the first step when the data is likely to be memorized | automatic |
| Checkpoints | Writes the adapter every N steps | `--save-every N` |
| Continue an adapter | Trains an existing adapter further, and records which one | `--train` with `--lora` |
| CUDA training assist | Runs the backward's largest products on an NVIDIA GPU, same bytes | `RUNNER_TRAIN_GPU=1` |
| GPU training | Runs every weight product of a step on an NVIDIA GPU, within rounding of a CPU run | `--train-gpu` |
| Trainer logprobs | Reports the logprob of every target token each step trains on | `--train-logprobs FILE` |
| Preference training | DPO from chosen and rejected answers, with no second model copy | `--train-dpo`, `--dpo-beta` |
| Distillation | Fits the adapter to a teacher's distributions instead of a text, to bring a quantized file back toward its parent | `--train-teacher`, `scripts/kd-teacher.py` |
| Reinforcement loop | Samples, scores and trains in rounds, checking that sampler and trainer agree bit for bit | `scripts/train-grpo-lite.py` |
| Evaluate | Teacher-forced logprobs, perplexity and top-1 rate, with or without the adapter | `--score` |
| Serve one adapter | Applies the adapter on the CPU, CUDA or Metal | `--lora`, `--lora-scale` |
| Serve many adapters | Picks an adapter per request by model name | `--adapter NAME=PATH` |
| Merge | Folds the adapter into a standalone GGUF, and refuses a merge that would erase it | `--merge-lora`, `--quant`, `--merge-allow-erased` |
| Provenance records | Records the base, data, starting adapter, settings and build of every run | `.train.json`, `.merge.json` |
| Sign records | Signs a training or merge record in place | `--sign-key` |
| Sign adapter files | Signs an adapter file and requires the signature at load | `--sign-model`, `--lora-sig`, `--require-signed-model` |
| Lineage | Walks an adapter's history back to the base and data | `--lineage` |
| Receipts and replay | Records which adapter answered, and replays the answer exactly | `--transcript`, `--receipts`, `--verify` |
| llama.cpp interop | Loads llama.cpp-format adapters, and writes adapters llama.cpp loads | GGUF adapter format |

## What it is for, and what it costs

Adapters suit small, targeted datasets: a tool habit, a format, a house style,
on the order of hundreds of examples and hundreds of steps. This is not an
instrument for pretraining or for large corpora.

Training runs on the CPU, one example per step. A step takes about 20 s for a
4B model on a many-core CPU host, and about a minute for a 7B on a consumer
GPU (Qwen2.5-7B Q4_K_M: 64.8 s per step on an RTX 3070 with the CUDA assist,
76.5 s on 8 CPU threads). The published Qwen3-4B adapter took 316 steps.
Budget hours, not minutes, and expect a Python training stack on a GPU to be
faster per step. What Runner gives for that cost is one numerical path and an
artifact you can audit.

## 1. Check that a model can take an adapter

```sh
./runner -m model.gguf --adapt-info
```

```json
{"architecture":"llama",
 "adapter_serving":{"supported":true,"reason":null,"device_kernels":"metal"},
 "training":{"supported":true,"reason":null,"host":"cpu"}}
```

Inference, adapter serving and training are three separate admission lists.
`--adapt-info` reads the last two without running either. When something is
not supported, `reason` is the exact sentence the real operation would refuse
with. `device_kernels` names the GPU backend that applies the adapter on its
own blocks (`cuda` or `metal`), and is `null` when serving the adapter means
the CPU (`--gpu off`). Training is always hosted on the CPU.

## 2. Prepare the training data

`--train` reads three kinds of input. The file extension decides: a `.jsonl`
file is read line by line, anything else as plain text.

### Plain text

Any text file. Runner tokenizes the whole file and cuts it into windows of
`--train-ctx` tokens (default 128), each overlapping the next by one token,
so every transition is trained exactly once per pass. Every token is a
target.

### Prompt and completion pairs

One JSON object per line:

```json
{"prompt": "List the Python files in src.", "completion": " {\"tool\":\"search_files\",\"args\":{\"pattern\":\"src/*.py\"}}", "weight": 1.0}
```

- The **prompt** is context only: it is masked out of the loss, so the model
  is not taught to produce it.
- The **completion** is what the model learns to produce after the prompt. It
  is tokenized separately from the prompt, without a beginning-of-sequence
  token.
- **weight** (optional, default 1.0) scales the example's loss. It can be any
  finite number that fits in a 32-bit float, including a negative one: a
  negative weight pushes the model away from that completion, which is how
  the reinforcement loop below uses advantages. Examples with weight 0 train
  nothing.
- **end_of_turn** (optional, `true` or `false`) appends the model's
  end-of-turn token to this completion, see
  [Teach the model to stop](#teach-the-model-to-stop).

A line that does not tokenize into a prompt and a completion, or that is
longer than the training window, is refused with its line number. Nothing is
ever silently truncated.

### Token ids

The same line can give token ids instead of text:

```json
{"prompt_tokens": [1, 415, 8, 20], "completion_tokens": [913, 54, 2], "weight": 0.5}
```

The ids are used exactly as given: no beginning-of-sequence token is added and
nothing is re-tokenized. Every id must be a row of the model's vocabulary, both
arrays must hold at least one id, and a line uses one form or the other, never
both. `weight` and `end_of_turn` work as in the text form.

Use this form whenever you train on what a model produced, for example the
samples of a reinforcement loop. Decoding sampled ids to text and tokenizing
the text again is not an identity. A byte token that ends in the middle of a
character decodes to U+FFFD, and a word the model happened to spell in two
tokens need not be split the way the tokenizer would split it. A text round
trip can therefore train on tokens the model never produced. The id form
removes that gap.

### Teach the model to stop

Without an end-of-turn token as a target, an adapter learns its answer but
not that the turn ends there, so after the trained answer the model keeps
generating, and it tends to repeat the answer. `--train-eot` appends the
model's end-of-turn token to every completion; `"end_of_turn": true` does it
for one line. The token is the one that ends an assistant turn in the
model's own chat template (or its end-of-sequence token when the template
has no separate one), the same token serving stops on. It is opt-in, because a
completion that is a file or a code fragment must not learn to end a turn
after it. A model that declares no such token refuses the request by name.

### Small data memorizes

A step is one example, so `--train-steps 60` over 4 examples is 15 passes over
the data. On data that small the adapter memorizes its completions, and when
they share one answer it learns that every prompt of that shape gets that
answer. Measured on MiniCPM5-1B Q4_K_M with four questions that all answer
the same passphrase: at the default learning rate, 2.5 passes gave the
passphrase to none of 24 unrelated questions, 5 passes to 42% and 15 passes
to 62%; at `--lr 5e-4`, 15 passes gave it to 92%. A reference PEFT run with
the same settings gave the same numbers, so this is the data and the
settings, not the trainer.

Before the first step, `--train` warns when:

- the run is more than three passes over fewer than 200 examples;
- more than half of the completions are the same text;
- most completions are short and none ends with an end-of-turn token;
- `--lr` is more than twice the default.

Training still runs after a warning. To avoid memorizing, keep it to one to
three passes, add examples where other questions keep their usual answers,
and use `--train-eot`.

## 3. Train

```sh
./runner -m base-Q4_K_M.gguf --train data.jsonl --train-steps 200 \
  --lora-rank 8 --train-out adapter.gguf
```

What happens:

- A fresh adapter is created on every dense projection of every block
  (attention Q, K, V and output; FFN gate, up and down). Its A matrices start
  small and seeded, its B matrices at zero, so before the first step the
  adapter is an exact no-op and training starts from the base model's own
  behavior.
- Each step takes the next example in file order (the file is cycled; it is
  not shuffled, so shuffle the file yourself if order matters), runs the
  forward and backward pass through the frozen quantized base, and takes one
  AdamW step on the adapter only.
- Each step prints one JSON line: `{"step","example","tokens","loss","step_s","tok_s"}`.
  The loss is the per-token mean over the example's targets, signed by the
  weight.
- At the end the adapter is written to `--train-out` (default
  `adapter-out.gguf`) together with its record, `adapter.gguf.train.json`
  (see [Records](#12-records-signatures-and-lineage)).

The knobs:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--train-steps N` | 100 | Optimizer steps; one example per step |
| `--lr F` | 1e-4 | AdamW learning rate (beta1 0.9, beta2 0.999, eps 1e-8, weight decay 0.01) |
| `--lora-rank R` | 8 | Rank of a fresh adapter, 1 to 512. Alpha is twice the rank, so the delta's scale alpha/rank is 2 |
| `--train-ctx N` | 128 | Training window in tokens; the effective window is the smaller of this and `-c` |
| `-s SEED` | fixed | Seed for the adapter's initial A matrices; without it the seed is fixed, so runs repeat |
| `--save-every N` | 0 (off) | Write a checkpoint every N steps |
| `-t N` | auto | CPU threads; the adapter bytes do not depend on it |

Training loads the model on the CPU, even in a CUDA or Metal build: `--train`
and `--train-dpo` keep the base frozen and memory-mapped, and only the
adapter's parameters and their optimizer state change. A step reads each
weight once per 16 positions (`RUNNER_TRAIN_ROWS`).

### Training on the GPU (`--train-gpu`)

`--train-gpu` runs every weight product of a step on an NVIDIA GPU: the
forward, the heads, the backward's recompute and its transposed products, for
every weight type, with the weights uploaded once and cached for the run. The
CPU keeps the rest: norms, rope, attention, the DeltaNet recurrence, the
adapter's own rank-r products and the optimizer.

It is what to use when the weight products dominate, which they do on large
models. On a 24 GB datacenter MIG slice with 6 CPU threads, six
Qwen3.5-0.8B distillation steps took 10.3 s against 43.1 s for the CPU
trainer on the same threads (4.2x); the device calls were about 0.4 s of a
1.5 s step, so on a small model the CPU's share, not the GPU, sets the step
time.

The adapter is not the CPU run's byte for byte: the device products add in
another order. It matches within rounding: on Qwen3.5-0.8B the losses matched
to 1e-6 and the held-out evals to six digits after 6 steps. Within a run the
results are deterministic, and the device's forward product gives every
position the same value however the window is batched, which the backward
needs. The gradient gates (`test-lora-grad`) run in this mode on a CUDA host,
at a batching that would expose a product that depends on it. The record
(`ADAPTER.gguf.train.json`) names the device under `products`. Without a
CUDA build and an NVIDIA GPU, `--train-gpu` is an error, not a CPU run. There
is no Metal equivalent.

### The CUDA training assist

`RUNNER_TRAIN_GPU=1` puts only the backward pass's transposed products on an
NVIDIA GPU, every weight type, with the weights cached there. They are the CPU
chain's bits, so the adapter bytes do not depend on it; only the step time
does. Without a usable CUDA device it says so once and trains on the CPU.

### See the logprobs a step trains on

`--train-logprobs FILE` writes one JSON line per step:

```json
{"step": 1, "example": 0, "first": 25, "logprobs": [-5.73344898, -5.13397217]}
```

`logprobs` holds the logprob of every target token from position `first` on
(the completion's first token for a prompt and completion pair), under the
forward pass that step trains through, before the step's update. They are
computed with the same normalizer as the server's logprobs and printed at full
float precision, so they compare bit for bit with what a server reported for
the same tokens under the same weights. This is how you check the
train/serve identity yourself, and the input an importance-ratio method
needs. With `--lr 0` no step moves the weights, so every line is scored under
the starting adapter.

## 4. Continue, checkpoint and resume

**Checkpoints.** `--save-every N` writes the adapter to `--train-out` every N
steps (and at the last step), and beside it the optimizer's state,
`OUT.optim`: the AdamW moments, the step and the next rows. Each checkpoint is
also kept as `OUT.step-NNNNN` with its own `.optim`. Every write goes to a
temporary file first and is then installed, so a failure leaves the previous
checkpoint in place.

**Resume a run.** `--train-resume CKPT` continues the run that wrote the
checkpoint `CKPT` (`OUT` or one of its `OUT.step-NNNNN` copies): the adapter,
the moments, the step (so the learning-rate schedule picks up where it was)
and the next rows. Give the rest of the command as the run had it, with the
same `--train-steps`. A resumed run writes the same adapter, byte for byte,
as the run that never stopped, and its record names the step it resumed at. A
sidecar written for a different adapter file is refused.

**Continue an adapter.** Pass an existing adapter with `--lora` to train it
further:

```sh
./runner -m base-Q4_K_M.gguf --lora adapter.gguf --train more.jsonl \
  --train-steps 100 --train-out adapter-2.gguf
```

- The adapter keeps its own rank and alpha; `--lora-rank` applies only to a
  fresh adapter.
- `--lora` loads the weights only, so a continued run starts AdamW fresh. It
  is a new run from those weights: training 100 steps and then 100 more is not
  the same as 200 steps in one run. To continue the same run, use
  `--train-resume` on a checkpoint instead.
- The record names the adapter the run started from (`from_adapter`, with its
  hash and its own record), and `--lineage` walks back through every earlier
  run. The starting adapter is identified before the first step, so the record
  is right even when `--train-out` names the same file. Train to a new file
  anyway: training in place overwrites the earlier adapter and its record, and
  the lineage then shows that history as broken.

## 5. Preference training (DPO)

`--train-dpo FILE` trains from preference pairs, one per line:

```json
{"prompt": "Reply in one sentence: what is a GGUF file?", "chosen": " A single-file model format.", "rejected": " Well, it depends on many things..."}
```

It implements DPO (Rafailov et al., arXiv 2305.18290). The reference policy is
the same frozen base with the adapter switched off for that pass, so DPO needs
no second copy of the model. `--dpo-beta F` (default 0.1) weighs how far the
adapter may move from the reference. `end_of_turn` per line and `--train-eot`
work as in `--train`, and the step, learning-rate, context, rank, output and
signing options are shared. One pair is one step, cycled in file order. The
record's `objective` is `dpo`, with `dpo_beta`.

The gradient is gated against two weighted cross-entropy backwards and against
a directional derivative by central differences. Read the cost before you
start: a step processes both answers, and on a 14B model a pair of about 4,000
tokens takes just under an hour on a CUDA host.

## 6. Distillation toward a parent model

`--train-teacher DIR` changes what `--train` fits. Each completion position is
fitted to a teacher's distribution instead of the next token of a text: the
loss is the KL divergence from the teacher's top-k tokens, with the rest of the
vocabulary as one tail bucket, to the student's. The use it was built for is
bringing a quantized file back toward the model it was quantized from. The
adapter trains on the frozen quantized base and learns to undo part of the
rounding error, which cross-entropy on a text does not aim at.

DIR holds two NumPy arrays, both shaped `[lines, positions, k]`:

- `topk_idx.npy` (int32): the teacher's top-k token ids for completion token
  `p` of line `n` of the `--train` file;
- `topk_logp.npy` (float32): their natural-log probabilities over the
  teacher's whole vocabulary.

The `--train` file must use the id form, since the teacher scored exactly those
tokens, and an end-of-turn token cannot be appended (put it in
`completion_tokens`). `scripts/kd-teacher.py` writes all three files from a
parent GGUF, through `runner --score --score-top K`:

```sh
python3 scripts/kd-teacher.py --runner ./runner --parent model-Q8_0.gguf \
    --docs docs.jsonl --topk 64 --out teacher/ -- --gpu off
./runner -m model-Q4_0.gguf --gpu off --train teacher/train.jsonl \
    --train-teacher teacher/ --train-ctx 256 --train-steps 100 --train-out kd.gguf
```

A teacher from another pipeline, such as the publisher's float32 model in
PyTorch, works the same way when it writes the same two arrays.

Per position, with `p_i` the teacher's listed probabilities, `p_tail` what
they leave (`1 - sum(p_i)`), `q` the student's softmax and `q_tail` the
student's mass on the unlisted tokens, the loss is

```text
KL = sum_i p_i (log p_i - log q_i) + p_tail (log p_tail - log q_tail)
```

A one-hot teacher (k 1, log-probability 0 on the next token) is the
cross-entropy, bit for bit, so prompt masking and `"weight"` work as in plain
`--train`, and the reported loss is the mean KL per completion position.
Before the first step the teacher is read once, and refused when its shape does
not match the `--train` file or a row lists an id outside the vocabulary, an
id twice, a token the model suppresses, a non-finite log-probability, or
probabilities summing past one. The record's `objective` is `kl-to-teacher`,
with the teacher's `k`, its positions and both arrays hashed.

Measured on 2026-10-23 on an Apple M1 (CPU, 4 threads): Qwen3-0.6B Q4_0 as
the student, its Q8_0 file as the teacher (top-64, which held 84.5% of the
mass on average), 24 training documents of 29 to 245 tokens, 48 steps at
`--lr 2e-4`, rank 8. On 6 held-out documents the mean KL to the teacher fell
from 0.2006 (no adapter) to 0.1346, lower on every document. The same run with
plain cross-entropy on the same tokens left it at 0.8357: fitting the text
moved the student away from the teacher. A step cost 12.34 s against the
cross-entropy's 12.29 s.

On Qwen3.5-0.8B, which trains through its Gated DeltaNet layers, the same
setup against the Q8_0 file lowered the held-out KL from 0.1269 to 0.0914 for
a Q4_0 student (requantized from the Q8_0) at `--lr 2e-4`. The Q4_K_M student
starts much closer to its teacher (0.0319), and at `--lr 2e-4` it diverged
(0.0447 after 48 steps) while at `--lr 5e-5` it fell to 0.0257, lower on every
held-out document. The trainer's AdamW moves each parameter by about the
learning rate per step whatever the loss, one example per step, with no warmup
or decay: the closer a student starts to its teacher, the lower the rate it
needs. Start low (5e-5) and watch the per-step loss, comparing each example
with its own earlier pass.

**A distillation run's recipe.** These flags give `--train` what a
distillation pipeline's own trainer has (they apply to cross-entropy too):

- `--lr-warmup N` raises the learning rate linearly over the first N steps;
  `--lr-schedule cosine` then lowers it along a cosine to `--lr-floor F` times
  `--lr` at the last step (0.1 by default; `constant` keeps it).
- `--train-batch N` takes N rows a step, and the loss and gradient are the
  mean over all their positions. Rows are used in file order, wrapping at the
  end; write the file in the order you want them drawn.
- `--weight-decay 0` turns AdamW into plain Adam (the default decay is 0.01).
- `--train-eval LABEL=ROWS[,TEACHER]` (up to eight) is a held-out set,
  evaluated with no update at step 0, every `--eval-every N` steps and at the
  last: its mean loss per position, and two top-1 agreements, `top1_topk`
  (among the teacher's listed tokens the student ranks the teacher's first
  highest) and `top1_vocab` (the student's most likely token over the whole
  vocabulary is it).
- `--kill-at S --kill-metric LABEL --kill-min-rel R`: at step S, unless
  LABEL's loss has fallen by at least R (a fraction) from step 0, the run
  saves a checkpoint, writes `OUT.killed.json` and exits with status 3;
  otherwise it writes `OUT.kill-check.json` and goes on.
- `--stop-file F`: after every step (its eval, checkpoint and kill check), if
  F exists the run saves a checkpoint and exits with status 75, ready for
  `--train-resume`.

Every step line carries `lr`, `grad_norm_max` and `grad_norm_mean` (each
layer's L2 over its adapter gradients, the largest and the mean), and with
`--train-batch` the `rows` it drew. A step whose loss or gradient is not
finite stops the run with an error before it touches the adapter.

The objective is gated in `tests/test_kd_grad.c`: a one-hot teacher against the
cross-entropy (loss and gradient bit for bit), the loss against an independent
formula, and the gradient against central differences, on the plain,
logit-scaled and soft-capped heads. It covers the models the trainer covers,
qwen35 and Clef included ([section 15](#15-what-the-trainer-covers-and-what-it-refuses)).

## 7. Reinforcement fine-tuning (GRPO-lite)

`scripts/train-grpo-lite.py` closes the loop: it samples completions from the
served model, scores them with a reward, turns group-relative advantages into
weighted training examples, and trains one pass, round after round.

```sh
python3 scripts/train-grpo-lite.py --runner ./runner --model base-Q4_K_M.gguf \
  --rounds 5 --k 8 --lr 5e-5 --out-dir grpo
```

- Each round serves the current adapter, samples `--k` completions per prompt
  (`--temperature`, `--max-tokens`), and scores each with the task's reward.
  The advantage is group-normalized and clipped to [-1, 1]; examples with zero
  advantage are dropped. The built-in task asks for a JSON tool call, and you
  adapt it by editing `PROMPTS` and `reward()` in the script.
- Training is one weighted pass over the round's examples, continuing the
  previous round's adapter. `--rank`, `--lr` and `--threads` are passed through.
- Sampling is seeded per request and training is deterministic, so a whole run
  replays exactly from its configuration.
- Each round prints its valid rate and mean reward. The out directory holds
  every round's data, adapter, trainer logprobs and the final adapter.

**The policy that samples is the policy that trains, bit for bit.** This is
the point of the script, and it holds under three conditions the script
keeps:

1. The trainer gets the sampled **token ids** (the id form above), not text.
2. The rollout server reads the prompt **one token per forward** (`-b 1`), the
   path the trainer's forward runs. A prompt read in batches of 8 or more
   takes another CPU route, and the logprobs after it move: up to 3.1e-3 on
   MiniCPM5-1B Q4_K_M after a 60-token prompt.
3. Sampling and training both run **on the CPU** (`--gpu off`). A GPU sampler
   agrees with the CPU trainer token for token, not bit for bit.

Every round checks it: the trainer's first step scores example 0 under the
exact weights that sampled it, and its logprobs must equal the server's, or
the run stops. `tests/test_rollout_identity.py` gates the same property in
`make test` on every family the trainer admits. In published reinforcement
learning work, this gap between the engine that samples and the engine that
trains is a known problem that needs corrections such as truncated importance
sampling; here there is no gap to correct. The limit is scale: a step takes
seconds to minutes, so this suits small models and on-device or regulated
work, not large-scale RL.

## 8. Evaluate an adapter

`--score` teacher-forces a text and prints, as JSON, every token's logprob,
the next-token argmax at each position, the NLL, the perplexity and the top-1
rate:

```sh
./runner -m base-Q4_K_M.gguf -f heldout.txt --score
./runner -m base-Q4_K_M.gguf --lora adapter.gguf -f heldout.txt --score
```

Compare the two to see what the adapter changed, on the task's text and on
neutral text it should leave alone. By default each position is scored with
one forward pass, the exact numbers the sampler sees. `RUNNER_SCORE_CHUNKED=1`
is faster but takes the batched route (up to 4.2e-3 different on a real 1B
model).

For a served adapter, the completions endpoint with `echo` and `logprobs`
scores a prompt the same way, and `max_tokens: 0` scores without generating,
which is the loglikelihood request an evaluation harness sends. A task metric,
such as exact tool calls on a held-out set, is your own harness against the
server.

## 9. Serve one adapter

```sh
./runner -m base-Q4_K_M.gguf --lora adapter.gguf --serve
```

- The adapter is applied beside the frozen base as `y += scale · B(A·x)` on
  every adapted projection. The base weights and kernels are untouched, so the
  delta is exact and every gate on the base still describes the run.
- `--lora-scale F` multiplies the adapter's trained alpha/rank (default 1.0).
  A zero adapter, or scale 0, is gated byte-identical to the bare base.
- **CPU, CUDA and Metal.** An offloaded block applies the delta from adapter
  weights held in device memory (36.9 MB at rank 8 on a 1.5B model), and a
  partial split lets each half apply its own blocks. The CPU-versus-GPU gap with
  the adapter is no wider than without it: on CUDA, max |Δlogprob| 1.110e-3
  against 1.255e-3 (Qwen2.5-1.5B Q4_K_M, RTX 3070); on Metal, mean 3.0e-4
  against 3.1e-4 (SmolLM2-135M Q8_0, rank 8 on all 210 projections).
- **Metal decodes more slowly with an adapter**: a block with an adapter runs
  without the fused front kernel. Measured: 117 to 44 tok/s on that 135M model
  (the CPU decodes it at 37), and 9.8 to 8.3 tok/s on Llama-3.2-3B Q4_K_M with
  a rank-16 adapter on Q and V. Prompt processing barely moves.
- **Qwen3.5/3.6/3.8 hybrids and Clef.** On a qwen35 model an adapter can also
  sit on each Gated DeltaNet layer's five projections, `attn_qkv`, `attn_gate`,
  `ssm_beta`, `ssm_alpha` and `ssm_out` (PEFT's `in_proj_qkv`, `in_proj_z`,
  `in_proj_b`, `in_proj_a`, `out_proj`), as well as on the full-attention
  layers (whose `attn_q` spans the query and its output gate) and every FFN, on
  the CPU and on CUDA (Metal runs qwen35 on the CPU). On a Clef file the same
  adapter reaches the decision head through the backbone; a `qwen35` adapter,
  which llama.cpp's converter writes from Clef's backbone, is accepted there.
  These models train adapters too, on the CPU
  ([section 15](#15-what-the-trainer-covers-and-what-it-refuses)).
- **Fails closed.** A shape or rank mismatch, an unknown target, an
  architecture the adapter path does not cover (the Mamba-2 hybrids, Gemma's
  dual-branch MoE), or an adapter on a routed-expert FFN that the device path
  cannot reach refuses the load by name, rather than serving a model that
  ignored the adapter on some blocks.
- The adapter is part of the engine's model identity, so a cached prompt prefix
  never crosses from one adapter to another.
- The adapter and scale apply to every serving slot and to every reload after
  `/unload` or a TTL expiry, including named registry entries; a missing or
  incompatible adapter refuses that load. Draft models do not inherit the
  target's adapter.

## 10. Serve many adapters, chosen per request

```sh
./runner -m base.gguf --serve --gpu off \
  --adapter tools=tools.gguf --adapter style=style.gguf
```

```json
{"model": "base.gguf:tools", "messages": [{"role": "user", "content": "..."}]}
```

- Each `--adapter NAME=PATH` (up to 16) is checked once at start by the same
  refusals as `--lora`. A request selects one by naming it after the model,
  `"model": "<model>:NAME"` (or an alias such as `"runner:NAME"`); a request
  without a suffix runs the bare base. `/v1/models` lists every `<model>:NAME`.
- The slot borrows the adapter for that request only, so parallel slots serve
  different adapters side by side. Switching adapters drops the KV the slot
  holds, so no cached row computed under one adapter serves another.
- Each response's `runner_telemetry.adapter` names the adapter and its sha256,
  and `/v1/runner/provenance` lists the loaded set with each signature verdict.
- A routed request answers exactly as a server started with `--lora` on the
  same adapter, and the zero adapter exactly as the bare base
  (`tests/test_adapter_routing.py`).
- Limits today: CPU only (`--gpu off`), one model served by path (not a swap
  registry), and exclusive with `--lora`. `--lora-scale` applies to every
  adapter.

## 11. Merge an adapter into a standalone file

```sh
./runner -m base-Q4_K_M.gguf --lora adapter.gguf \
  --merge-lora merged-Q8_0.gguf --quant q8_0
```

A merge computes `W' = W + (alpha/r) · B · A` for every adapted projection and
writes a standalone GGUF that runs in any GGUF runtime, with no adapter
needed. Each merged tensor is written at its own type, or at `--quant T`
(`q8_0`, `q4_0`, `q3_k`, `q4_k`, `q6_k`, `f16`, `bf16`, or `keep`); untouched
tensors are copied byte for byte; the merge is deterministic, and
`merged.gguf.merge.json` records the base, adapter and merged hashes.

**A merge into a quantized type rounds the delta through that type's grid, and
can erase the fine-tune.** Runner measures this on every merge. Each adapted
row is also written from the base alone at the same type, and the two are
compared:

- `delta_retained`: how much of the intended change survived (1 = all of it,
  0 = rounded away);
- `bytes_changed`: how many adapted bytes differ from the base written alone.

Both are printed and recorded under `survival` in the merge record. A merge
that keeps under half of the delta is refused, and the destination is left
untouched. In the measured study, the adapter merged back into the 4-bit grid
its base sits on kept 1.8% of its delta and scored like the base (0.69); into
Q8_0 it kept 99.5% and scored 1.00. The same happens on any grid: a 20-step
adapter on Qwen3-0.6B Q8_0 merged back into Q8_0 kept 18.2%, and into F16 all
of it.

Choose one of:

- merge into a type wider than the base's own (F16 keeps the delta as
  computed; Q8_0 is enough over a 4-bit base);
- serve `base + --lora`, the exact form;
- or pass `--merge-allow-erased` to write it anyway (the record then shows
  `"min_retained": null`).

Always evaluate a merged file before trusting it.

## 12. Records, signatures and lineage

### The training record

Every adapter is written with `ADAPTER.gguf.train.json`
(`xyntetik.runner.train.v1`):

| Field | What it records |
| --- | --- |
| `runner`, `build` | Version, the binary's sha256, compiler, OS and architecture |
| `base` | The base model's path and sha256, and its own record when it has one |
| `from_adapter` | The adapter a continued run started from (absent for a fresh one) |
| `objective` | `weighted-cross-entropy` or `dpo` (with `dpo_beta`) |
| `data` | The training file's path and sha256 |
| `seed`, `lora_rank`, `alpha`, `lr`, `steps`, `ctx`, `end_of_turn`, `eot_id`, `adamw` | The full configuration |
| `loss_first`, `loss_last` | The first and last step's loss |
| `adapter` | The output's path and sha256 |

A merge writes `MERGED.gguf.merge.json` the same way, with the survival
measurement. The record is written only after the output is installed and
every input is hashed, and any failure along the way, signing included, makes
the command fail.

### Signing records

```sh
./runner --keygen keys/sign.json
./runner -m base.gguf --train data.jsonl --train-out adapter.gguf --sign-key keys/sign.json
```

`--sign-key` signs the training, merge or quantization record in place
(Ed25519 by default, or ML-DSA-44 with `--keygen-algo ml-dsa-44`). The
signature covers every byte of the record before it, so any Ed25519 or ML-DSA
library verifies it from the file and the embedded public key alone.

### Signing adapter files

An adapter changes the model that serves, so it answers to the same signing
policy as the model. An adapter file gets an OpenSSF Model Signing bundle the
same way a model does:

```sh
./runner --sign-model adapter.gguf --model-key key.pem        # writes adapter.gguf.sig
./runner -m base.gguf --lora adapter.gguf --model-pubkey key.pub \
  --require-signed-model --serve
```

The `--lora` adapter's bundle (`--lora-sig FILE`, or `adapter.gguf.sig` beside
it) and every `--adapter` bundle (`PATH.sig`) are verified with
`--model-pubkey`, checked again on every reload, and required by
`--require-signed-model`. A bundle that does not verify, an unsigned adapter
under the requirement, or an adapter whose bytes changed refuses the load. The
verdict is recorded as `adapter_signature` in receipts and as the adapter's
`signature` in `/v1/runner/provenance`.

### Lineage

```sh
./runner --lineage merged-Q8_0.gguf
```

`--lineage` walks a file's records back to its origins: the merged file to the
base and the adapter, the adapter to its base, data and starting adapter, the
base to its quantization. Each record is re-hashed, each link is checked
against the next step, each file still on disk is checked against its recorded
hash, and each signature is verified (`--trust-key` pins the signer). Exit 0
means verified and signed, 1 consistent but not all signed, 2 broken. On a
`--receipts` folder it lists which model and adapter versions answered, and
when. [The lineage walkthrough](lineage-walkthrough.md) runs the whole chain,
from quantization through training, merging, evaluation and serving, to a
flipped byte.

## 13. Receipts and replay with an adapter

A one-shot run recorded with `--transcript FILE`, and every served answer under
`--receipts DIR`, records the adapter: its path, sha256, scale and record link.
`--verify` replays the record and needs the same adapter: without `--lora`, with
a different adapter, or with `--lora` on a record that ran without one, the
verdict is `UNVERIFIABLE`. Replay with the matching adapter returns `VERIFIED`
when the answer is reproduced. The scope of replay is in
[determinism-scope.md](determinism-scope.md) and in
[reproducible training with receipts](reproducible-lora-training-receipts.md).

## 14. Working with other tools

- **The file format is llama.cpp's**: an adapter GGUF with
  `general.type = adapter`, `adapter.type = lora`, `adapter.lora.alpha`, and
  tensors named `blk.N.<projection>.weight.lora_a` and `.lora_b`.
- **Into Runner.** F32, F16 and BF16 adapter tensors load. F16 is what
  llama.cpp's `convert_lora_to_gguf.py` emits, so a PEFT adapter converted with
  it loads and serves (measured on a community adapter). Quantized adapter
  tensors are refused.
- **Out of Runner.** An adapter Runner trained loads in llama.cpp (`--lora`)
  and scored the same 1.00 there on the published task.
- **A merged F16 or Q8_0 file** is a plain GGUF that any runtime serves.

## 15. What the trainer covers, and what it refuses

The backward covers the dense gated transformer in its common shapes: Llama,
Mistral, Qwen2.5 and Granite 4.x dense (muP scalars, tied output), the head
transforms (logit scale, softcap, suppressed tokens), the attention output
gate (afmoe, Muse-Glimmer), sandwich norms, per-layer output scale,
sliding-window attention, partial-dimension RoPE (a rotation over part of each
head), the GELU gated FFN (Gemma 3) beside the SiLU one, and
Gemma 4 in full (the weightless V norm, the V-less full-attention layers, the
E-series shared KV and per-layer embeddings).

On qwen35 (Qwen3.5/3.6) and Clef the backward runs through the Gated DeltaNet
layers: backpropagation through time over each head's state, the causal
convolution, the L2-normed queries and keys, the decay and update gates and
the gated output norm, with an adapter on all five DeltaNet projections. A
head's states are kept every C positions (C the square root of the window)
and each stretch is rebuilt from its checkpoint on the way back, so memory
grows with the square root of the window rather than with the window. The
full-attention layers in between train through their fused query-and-gate
projection and their partial RoPE.

On sparse-MoE models with Qwen3-MoE or Mixtral routing (softmax router, top-k,
with or without renormalization), the adapter goes on the attention
projections, and the backward runs through the frozen experts and the
router's softmax weights.

It refuses by name: adapters on the experts themselves, other MoE routers
(sigmoid, group-limited, biased), shared experts, attention sinks, the Mamba-2
hybrids (Granite 4.0 H, Nemotron-H), qwen4exp's hyper-connections, MoE on a
DeltaNet model, Gemma's dual-branch MoE,
removed sublayers, LayerNorm (StableLM; the backward has the RMSNorm adjoint
only), and the derived-K and ring KV caches. `--adapt-info` gives the sentence
for the model you have.
Every covered shape is pinned by a finite-difference gate in `make test` on a
fixture that carries it.

## 16. Determinism, and where it stops

- **Same build, same bytes.** The same data, seed, configuration and build
  write a byte-identical adapter, whatever the thread count, and with or
  without the CUDA assist. This is gated in `make test`, and an independent tester
  reproduced the adapter bytes and loss on a Tesla T4.
- **Another build can change the bytes.** A binary built for a different
  instruction-set profile writes different adapter bytes (the transcendental
  functions come from the compiler's math library), while behavior stays put.
  Say "same build" when you say "same bytes"; every record names its build so
  the difference is visible.
- **Sampler and trainer agree bit for bit** under the three conditions in
  [section 7](#7-reinforcement-fine-tuning-grpo-lite).
- **A served adapter on a GPU** agrees with the CPU within the backend's
  measured envelope, not bit for bit.

The full contract, with named edges, is
[determinism-scope.md](determinism-scope.md).

## 17. Measured results

On one public base and task, Qwen3-4B Q4_K_M with a 29-prompt held-out
tool-calling evaluation, greedy, measured 2026-08-22 on
[Qwen3-4B-Runner-ToolUse-Q4_K_M](https://huggingface.co/Joakimpalm-Zen/Qwen3-4B-Runner-ToolUse-Q4_K_M)
(rank 8 on every projection, alpha 16, lr 1e-4, 316 steps, ctx 128, seed 0;
data and records in the
[training record](https://huggingface.co/datasets/Joakimpalm-Zen/Qwen3-4B-ToolUse-LoRA-training-record)):

| Serving form | Exact held-out tool calls |
| --- | --- |
| Frozen Q4_K_M base | 0.69 |
| Base with the trained adapter (`--lora`) | 1.00 |
| Adapter merged into F16 or Q8_0 | 1.00 |
| Adapter merged back into the 4-bit grid | 0.69 |

- Two independent runs wrote the same adapter file, sha256 for sha256.
- Stock llama.cpp serves the adapter at the same 1.00.
- A neutral corpus moved from 4.063 to 4.026 nats per token, so the adapter
  leaves unrelated text alone.
- Adapters trained through BF16 and Q8_0 bases have cosine 0.9998 to each
  other, and the Q4_K_M-trained one 0.9926: a measurably different object that
  is capability-equivalent on this task's supervised decisions. An external
  36-prompt boundary bank finds 3 prompts where the three choose different
  tools ([the lane](tool-choice-boundary-lane.md)).

These are measurements for that base and task; another base or task is not
covered by them.

## 18. Reference

**Flags.** `--adapt-info`; `--train FILE`, `--train-steps`, `--lr`,
`--train-ctx`, `--train-out`, `--save-every`, `--lora-rank`, `--train-eot`,
`--train-logprobs`, `--train-teacher DIR`, `--lr-schedule`, `--lr-warmup`,
`--lr-floor`, `--train-batch`, `--weight-decay`, `--train-resume CKPT`,
`--train-eval`, `--eval-every`, `--kill-at`, `--kill-metric`,
`--kill-min-rel`, `--stop-file`, `-s`; `--train-dpo FILE`, `--dpo-beta`; `--lora FILE`,
`--lora-scale`, `--lora-sig`; `--adapter NAME=PATH`; `--merge-lora OUT`,
`--quant`, `--merge-allow-erased`; `--score`; `--sign-key`, `--keygen`,
`--sign-model`, `--model-key`, `--model-pubkey`, `--require-signed-model`;
`--lineage`, `--trust-key`; `--transcript`, `--receipts`, `--verify`. Each
row is in the [command-line reference](../MANUAL.md#command-line-reference).

**Files.** `ADAPTER.gguf` (the adapter), `ADAPTER.gguf.train.json` (its record),
`ADAPTER.gguf.optim` and `ADAPTER.gguf.step-NNNNN` (checkpoints and their
optimizer state), `ADAPTER.gguf.killed.json` / `.kill-check.json` (the kill
rule's verdict),
`MERGED.gguf.merge.json` (a merge record), `ADAPTER.gguf.sig` (an OMS bundle),
`*.partial` (a write in progress, never installed on failure).

**Environment.** `RUNNER_TRAIN_GPU=1` (the CUDA assist, same bytes; `--train-gpu`
is the whole step on the GPU), `RUNNER_TRAIN_PROF=1`
(per-step phase times on stderr), `RUNNER_TRAIN_ROWS=N` (positions the trained
forward runs at once, 16 by default, 1 one at a time; the same bytes at every
width, which the tests hold), `RUNNER_SCORE_CHUNKED=1` (batched scoring).

**Tests.** `tests/test_train.py`, `tests/test_rollout_identity.py`,
`tests/test_lora.py`, `tests/test_moe_lora.py`, `tests/test_lora_grad.c`,
`tests/test_dpo_grad.c`, `tests/test_kd_grad.c`, `tests/test_train_teacher.py`,
`tests/test_train_recipe.py`,
`tests/test_merge.py`,
`tests/test_adapter_routing.py`, `tests/test_adapter_signature.py`,
`tests/test_lineage.py`.

**More.** [Train on a quantized GGUF](train-lora-on-quantized-gguf.md) (the
short version), [reproducible training with receipts](reproducible-lora-training-receipts.md),
[adapter, training and merge contracts](adaptation-engine.md),
[lineage walkthrough](lineage-walkthrough.md).
