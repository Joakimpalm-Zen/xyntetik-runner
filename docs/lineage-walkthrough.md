# A model's life, recorded: a lineage walkthrough

One real chain on a small public model, run on 2026-10-09 with Runner 1.2.0
plus the changes after it (commit `12105296`), on a CPU. Every output below is
what the commands printed. Three things were edited:
- long machine paths are shortened to `./runner` and `scripts/`;
- the middle of the training log is cut;
- one line about the CUDA driver, an artifact of that build host, is removed.

The chain:
1. take a downloaded file (Qwen3-0.6B, BF16);
2. quantize it to Q8_0;
3. train a small LoRA on the Q8_0 file;
4. merge it;
5. record a fidelity evaluation;
6. serve the merged model;
7. ask, at the end, where an answer came from.

Every step writes a record beside its output, every record names its inputs by
sha256, and a signing key signs each record. `--lineage` then walks any file, or
any served answer, back to its origins.

**What "prove" means here.** These records prove *what ran and on which bytes*:
which files went in, which came out, with which settings, signed by whom. They
do not prove an answer is correct. The evaluation records say only what was
measured, against what, with which thresholds.

## 1. A signing key

```text
$ ./runner --keygen keys/sign.json
signing key -> keys/sign.json (keep it private; the ed25519 public key is f7142ff28869958f701008b28e9aafda4b19ba3c22d62ec589ebcd09f8576c5e)
{"schema_version":"xyntetik.runner.signkey.v1","algo":"ed25519","public_key":"f7142ff28869958f701008b28e9aafda4b19ba3c22d62ec589ebcd09f8576c5e"}
```

Keep `keys/sign.json` private. The public key is what a reader of the records
checks against (`--trust-key`).

## 2. Quantize, with a record

The starting file is a download. Runner has no record for it, so it is an
**origin**: the chain stops there, and its sha256 is what the chain names.

```text
$ sha256sum models/qwen3-0.6b-bf16.gguf
33d6f6c9f0dd21dd3d9c99f514498ea594119508e538aa36773993a3ffd39aff  models/qwen3-0.6b-bf16.gguf
```

```text
$ ./runner -m models/qwen3-0.6b-bf16.gguf --quantize models/qwen3-0.6b-q8_0.gguf --quant q8_0 --sign-key keys/sign.json
quantize: general.file_type 7 (MOSTLY_Q8_0) — output histogram: F32:113 Q8_0:198
quantize: models/qwen3-0.6b-bf16.gguf -> models/qwen3-0.6b-q8_0.gguf (Q8_0): 198 tensors converted, 113 kept
quantize: provenance -> models/qwen3-0.6b-q8_0.gguf.quant.json
```

`models/qwen3-0.6b-q8_0.gguf.quant.json` names the BF16 input by sha256, the
target type, the output's sha256 and the Runner binary, and carries the
signature.

## 3. Train a LoRA on the quantized file

Eight prompt and completion pairs, in the format `--train` reads:

```text
$ head -2 data/capitals.jsonl
{"prompt": "Q: What is the capital of Norway?\nA:", "completion": " Oslo."}
{"prompt": "Q: What is the capital of Sweden?\nA:", "completion": " Stockholm."}
```

```text
$ ./runner -m models/qwen3-0.6b-q8_0.gguf --train data/capitals.jsonl --train-steps 20 --lora-rank 8 --train-out adapters/capitals.gguf --sign-key keys/sign.json -t 8
loaded models/qwen3-0.6b-q8_0.gguf | qwen3 | 28 layers | ctx 4096 | 8 threads | 0.02s
sampling: qwen3 (temp 0.60, top_p 0.95, top_k 20, min_p 0.00, repeat_penalty 1.00)
sampling: from the file (general.sampling.*): temp 0.60, top_p 0.95, top_k 20
train: fresh rank-8 adapters on every projection (alpha 16)
train: 8 examples, 20 steps, lr 0.0001, ctx 128
{"step":1,"example":0,"tokens":13,"loss":6.347305,"step_s":0.64,"tok_s":20.4}
{"step":2,"example":1,"tokens":13,"loss":4.077549,"step_s":0.64,"tok_s":20.4}
{"step":3,"example":2,"tokens":13,"loss":1.845758,"step_s":0.62,"tok_s":21.1}
... (steps 4 to 18 omitted)
{"step":19,"example":2,"tokens":13,"loss":0.007411,"step_s":0.61,"tok_s":21.2}
{"step":20,"example":3,"tokens":13,"loss":0.004118,"step_s":0.62,"tok_s":20.9}
train: provenance -> adapters/capitals.gguf.train.json
train: done — first-step loss 6.3473, last-step 0.0041, adapter -> adapters/capitals.gguf
```

The training record names the base (and links the base's own quantize
record), the data file by sha256, the seed and the configuration. Training is
deterministic: the same inputs give a byte-identical adapter.

## 4. Merge, and what the merge kept

A merge folds the adapter into the base and writes a standalone file. Runner
measures how much of the adapter's change the output kept, and refuses a merge
that keeps under half of it. Here is a Q8_0 base merged back into Q8_0:

```text
$ ./runner -m models/qwen3-0.6b-q8_0.gguf --lora adapters/capitals.gguf --merge-lora models/qwen3-0.6b-capitals-q8_0.gguf --quant q8_0 --sign-key keys/sign.json
quantize: general.file_type 7 (MOSTLY_Q8_0) — output histogram: F32:113 Q8_0:198
error: the merge kept 18.2% of the adapter's delta (3.34% of the adapted bytes differ from the base written alone at the same type), under the 50% floor: the output grid rounded the fine-tune away, so models/qwen3-0.6b-capitals-q8_0.gguf would behave like the base. Merge into a type wider than the base's own (--quant f16 keeps the delta as computed; q8_0 is enough only over a narrower base, such as a 4-bit one), serve base + --lora, or pass --merge-allow-erased to write it anyway (destination left untouched)
```

```text
$ ls models
qwen3-0.6b-bf16.gguf
qwen3-0.6b-q8_0.gguf
qwen3-0.6b-q8_0.gguf.quant.json
```

Twenty steps of training move the weights by much less than a Q8_0 step, so
written back onto the grid the base already sits on, most of the change
rounds away. Nothing was installed, and the next merge goes to F16:

```text
$ ./runner -m models/qwen3-0.6b-q8_0.gguf --lora adapters/capitals.gguf --merge-lora models/qwen3-0.6b-capitals-f16.gguf --quant f16 --sign-key keys/sign.json
quantize: general.file_type 1 (MOSTLY_F16) — output histogram: F32:113 F16:198
quantize: models/qwen3-0.6b-q8_0.gguf -> models/qwen3-0.6b-capitals-f16.gguf (F16): 198 tensors converted, 113 kept
merge: 196 adapted projections folded into the weights
merge: the output kept 100.0% of the adapter's delta; 47.14% of the adapted bytes differ from the base written alone
merge: provenance -> models/qwen3-0.6b-capitals-f16.gguf.merge.json
```

The merge record (`.merge.json`) names the base and the adapter, each with a
link to the record that made it, and carries these numbers under `survival`.

## 5. Record an evaluation

`scripts/eval-record.py` turns a measurement into a record beside the file it
judged. Here the Q8_0 file is compared with its BF16 origin on a short text,
with thresholds, so the record says pass or fail:

```text
$ python3 scripts/eval-record.py fidelity --model models/qwen3-0.6b-q8_0.gguf --reference models/qwen3-0.6b-bf16.gguf --corpus data/corpus.txt --max-positions 40 --max-mean-kld 0.05 --min-top1 95 --sign-key keys/sign.json --runner ./runner
{
 "record": "models/qwen3-0.6b-q8_0.gguf.eval.fidelity.json",
 "metrics": {
  "positions_scored": 35,
  "positions_failed": 0,
  "mean_kld": 0.00522022214853118,
  "top1_agreement_pct": 100.0,
  "top1_margin_qualified_pct": 100.0,
  "mean_top8_overlap": 0.9642857142857143,
  "tie_band_nats": 0.5
 },
 "pass": true
}
```

Thirty-five positions of one paragraph is a smoke-sized measurement, enough to
show the mechanics and no more; a real gate uses a real corpus.

`--require-eval` turns the record into a load condition: the file must carry
a passing evaluation of that kind, made for these exact bytes, signed by the
trusted key.

```text
$ ./runner -m models/qwen3-0.6b-q8_0.gguf --require-eval fidelity --trust-key f7142ff28869958f701008b28e9aafda4b19ba3c22d62ec589ebcd09f8576c5e -p $'Q: What is the capital of Norway?\nA:' -n 4 --temp 0
A: -n 4 --temp 0
eval: models/qwen3-0.6b-q8_0.gguf.eval.fidelity.json passes for this file
loaded models/qwen3-0.6b-q8_0.gguf | qwen3 | 28 layers | ctx 4096 | 8 threads | 0.07s
sampling: qwen3 (temp 0.00 — greedy argmax; top_p/top_k/min_p/repeat_penalty inactive)
sampling: from the file (general.sampling.*): temp 0.60, top_p 0.95, top_k 20
Q: What is the capital of Norway?
A: Norway is a country
prompt: 11 tok, 195.74 tok/s | gen: 4 tok, 62.26 tok/s
```

The base model does not know the answer format yet ("Norway is a country").
The merged model has no evaluation, so it is refused:

```text
$ ./runner -m models/qwen3-0.6b-capitals-f16.gguf --require-eval fidelity --trust-key f7142ff28869958f701008b28e9aafda4b19ba3c22d62ec589ebcd09f8576c5e -p hi -n 4
error: --require-eval fidelity: no fidelity evaluation beside the model (models/qwen3-0.6b-capitals-f16.gguf.eval.fidelity.json)
```

## 6. Walk the chain from the file

```text
$ ./runner --lineage models/qwen3-0.6b-capitals-f16.gguf --trust-key f7142ff28869958f701008b28e9aafda4b19ba3c22d62ec589ebcd09f8576c5e
lineage of models/qwen3-0.6b-capitals-f16.gguf (sha256 650b712bdf98)
  merge -> models/qwen3-0.6b-capitals-f16.gguf sha256 650b712bdf98  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-capitals-f16.gguf.merge.json]
    base: models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645
      eval fidelity: vs models/qwen3-0.6b-bf16.gguf, 35 positions, top-1 100.0%, mean KL 0.00522; pass  VERIFIED (signed f7142ff28869958f...)
      quantize -> models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-q8_0.gguf.quant.json]
        base: models/qwen3-0.6b-bf16.gguf sha256 33d6f6c9f0dd  ORIGIN (no record: a download, or made before records)
    adapter: adapters/capitals.gguf sha256 85c3befcad0f
      train -> adapters/capitals.gguf sha256 85c3befcad0f  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record capitals.gguf.train.json]
        base: models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645
          eval fidelity: vs models/qwen3-0.6b-bf16.gguf, 35 positions, top-1 100.0%, mean KL 0.00522; pass  VERIFIED (signed f7142ff28869958f...)
          quantize -> models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-q8_0.gguf.quant.json]
            base: models/qwen3-0.6b-bf16.gguf sha256 33d6f6c9f0dd  ORIGIN (no record: a download, or made before records)
        data: data/capitals.jsonl sha256 b61b825f131c  ORIGIN (no record: a download, or made before records)
RESULT: VERIFIED (every link consistent and signed)
```

Read it from the top:
- the merged file came from the Q8_0 base and the adapter;
- the base came from the BF16 download, and carries a passing fidelity record;
- the adapter was trained on that same base with `data/capitals.jsonl`.

Each link was re-hashed and its signature checked; exit 0 means every link is
consistent and signed. Exit 1 means consistent but not all signed, and exit 2
means broken.

## 7. Walk the chain from an answer

Served with `--receipts`, every answer leaves a signed receipt that names the
model file by sha256 and links its record:

```text
$ ./runner -m models/qwen3-0.6b-capitals-f16.gguf --serve --receipts receipts --sign-key keys/sign.json
```

Two questions, sent to `/v1/completions` as `Q: ...\nA:`. The merged model
answers in the trained format: `' Riga. Latvia'` and `' Reykjavik'`.

```text
$ ls receipts
receipt-0000000001.json
receipt-0000000002.json
```

```text
$ ./runner --lineage receipts --trust-key f7142ff28869958f701008b28e9aafda4b19ba3c22d62ec589ebcd09f8576c5e
lineage of the receipts in receipts: 2 answers
  receipts: 2 signed, 0 unsigned, 0 bad; chain continuous
  timeline: v1 x2
  v1: model sha256 650b712bdf98, adapter none: 2 answers, 2026-10-09T14:24:48Z .. 2026-10-09T14:24:48Z
    model: models/qwen3-0.6b-capitals-f16.gguf sha256 650b712bdf98
      merge -> models/qwen3-0.6b-capitals-f16.gguf sha256 650b712bdf98  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-capitals-f16.gguf.merge.json]
        base: models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645
          eval fidelity: vs models/qwen3-0.6b-bf16.gguf, 35 positions, top-1 100.0%, mean KL 0.00522; pass  VERIFIED (signed f7142ff28869958f...)
          quantize -> models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-q8_0.gguf.quant.json]
            base: models/qwen3-0.6b-bf16.gguf sha256 33d6f6c9f0dd  ORIGIN (no record: a download, or made before records)
        adapter: adapters/capitals.gguf sha256 85c3befcad0f
          train -> adapters/capitals.gguf sha256 85c3befcad0f  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record capitals.gguf.train.json]
            base: models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645
              eval fidelity: vs models/qwen3-0.6b-bf16.gguf, 35 positions, top-1 100.0%, mean KL 0.00522; pass  VERIFIED (signed f7142ff28869958f...)
              quantize -> models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-q8_0.gguf.quant.json]
                base: models/qwen3-0.6b-bf16.gguf sha256 33d6f6c9f0dd  ORIGIN (no record: a download, or made before records)
            data: data/capitals.jsonl sha256 b61b825f131c  ORIGIN (no record: a download, or made before records)
RESULT: VERIFIED (every link consistent and signed)
```

From an answer back to the base model and the training data, with the receipt
chain's continuity and each receipt's signature checked first.

## 8. What a changed byte looks like

A copy of the files with one byte of the adapter flipped:

```text
$ ./runner --lineage models/qwen3-0.6b-capitals-f16.gguf --trust-key f7142ff28869958f701008b28e9aafda4b19ba3c22d62ec589ebcd09f8576c5e
lineage of models/qwen3-0.6b-capitals-f16.gguf (sha256 650b712bdf98)
  merge -> models/qwen3-0.6b-capitals-f16.gguf sha256 650b712bdf98  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-capitals-f16.gguf.merge.json]
    base: models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645
      eval fidelity: vs models/qwen3-0.6b-bf16.gguf, 35 positions, top-1 100.0%, mean KL 0.00522; pass  VERIFIED (signed f7142ff28869958f...)
      quantize -> models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-q8_0.gguf.quant.json]
        base: models/qwen3-0.6b-bf16.gguf sha256 33d6f6c9f0dd  ORIGIN (no record: a download, or made before records)
    adapter: adapters/capitals.gguf sha256 85c3befcad0f
      train -> adapters/capitals.gguf sha256 85c3befcad0f  BROKEN: the file on disk no longer matches its record (signed f7142ff28869958f...)  [runner 1.2.0, record capitals.gguf.train.json]
        base: models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645
          eval fidelity: vs models/qwen3-0.6b-bf16.gguf, 35 positions, top-1 100.0%, mean KL 0.00522; pass  VERIFIED (signed f7142ff28869958f...)
          quantize -> models/qwen3-0.6b-q8_0.gguf sha256 2f588d123645  VERIFIED (signed f7142ff28869958f...)  [runner 1.2.0, record qwen3-0.6b-q8_0.gguf.quant.json]
            base: models/qwen3-0.6b-bf16.gguf sha256 33d6f6c9f0dd  ORIGIN (no record: a download, or made before records)
        data: data/capitals.jsonl sha256 b61b825f131c  ORIGIN (no record: a download, or made before records)
RESULT: BROKEN
```

The walk names the step whose file no longer matches its record, and exits 2.

## Reference

The flags and record formats are in the manual:
- [`--lineage`](../MANUAL.md#command-line-reference), `--sign-key`, `--trust-key`, `--keygen`;
- [`--merge-lora`](../MANUAL.md#cli-merge-lora) and what a merge keeps;
- [`--train`](../MANUAL.md#cli-train);
- `--require-eval` and `scripts/eval-record.py`.

Training on quantized files is in
[train-lora-on-quantized-gguf.md](train-lora-on-quantized-gguf.md).
