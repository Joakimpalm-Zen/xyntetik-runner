# `runner derive`: make your own measured file (R17.6.1), design note

Status: accepted (coordinator, 2026-11-01); building, tests first.

Decisions:

- **The verb is `runner derive`.** It sits beside `attest` and `train` (#429's one-mode subcommands). The owner may
  rename it before the next release.
- **The default reference is allowed, but labelled and restricted.** Without `--reference hf:DIR` the card reports
  the numbers but never says "passes the bar" or "faithful". Its first lines say "measured against the parent file
  in Runner's arithmetic, not the publisher's float32 model", and the record carries `reference_kind:
  parent-file`. The house bar wording belongs to publisher-reference cards (`reference_kind: publisher-float32`).
- **v1 recovery is LoRA-heal only.** Scale-only recovery is a later item.

## The gap

The files of ours people pull most are made files, measured against their parent:

- the recovered Qwen3.8-27B IQ3_S (4,786 downloads in 30 days);
- the pruned and the selective-precision files (about 2,000 each).

Every step that made them exists, but only as separate pieces:

- **Runner already has:**
  - `--rewrite` with `--quant`, `--type-plan`, `--imatrix`, `--prune-experts`, `--remove-sublayer` and
    `--merge-lora`;
  - `--quant-error`;
  - `--score`;
  - `runner train`.
- **Scripts already have:**
  - `type-plan-by-sensitivity.py`;
  - `gold-logits.py` (the publisher's float32 reference);
  - `kld-compare.py`;
  - `certify-envelope.py`;
  - `model-tiers.py`.
- **Missing:**
  - one command that strings them together;
  - a card. Every card so far was written by hand.

A user who wants a smaller file of a model we never published has to repeat our research by hand, or trust a file
nobody measured.

## What the command does

```sh
runner derive -m PARENT.gguf -o OUT.gguf RECIPE [--reference hf:DIR] [--corpus FILE]
```

1. **Make** the file with the existing rewrite path, from a recipe of existing flags:
   - `--quant TYPE`;
   - `--type-plan PLAN`;
   - `--target-size BYTES --low TYPE --high TYPE`: runs `--quant-error` in-process and ranks the demotions as
     `type-plan-by-sensitivity.py` does (ported to C), so selective precision needs no hand-written plan;
   - `--imatrix FILE`, consumed as today;
   - `--prune-experts PLAN`;
   - `--remove-sublayer LIST`;
   - `--merge-lora ADAPTER`, which is the recovery step a user can run today with `runner train`.
2. **Measure** the file against a named reference, at the records' fixed window: 300 positions at stride 15 over
   `tests/fixtures/gold-corpus-2k.txt`, unless `--corpus` is given. The output is:
   - mean, median and max KLD;
   - top-1 and margin-qualified top-1;
   - the bar verdict (0.05 mean KLD, 97% margin-qualified top-1).

   Two references, named on the card exactly as they are:
   - `--reference hf:DIR`: **the publisher's model in float32**, through `scripts/gold-logits.py` (the
     provider-reference-first rule). This needs Python with torch and transformers. Without them it is refused by
     name, with the install line; there is no silent fallback.
   - Default: **the parent file in Runner's float32 arithmetic** (`-m PARENT`, scored on the CPU, a BF16 or F16
     parent strongly recommended). The card says this is the parent file, not the publisher's implementation. A
     quantized parent is accepted, and the card says the reference is itself quantized.
3. **Record** what was done:
   - `OUT.quant.json` (exists today);
   - `OUT.derive.json`: recipe, parent sha256, file sha256, runner version and commit, `--caps`, the reference
     named, the per-position comparison, the verdict.
4. **Write the card**, `OUT.README.md`, a Hugging Face card with YAML front matter:
   - `base_model` from the parent's `general.*`;
   - `license` copied from the parent, refused when the parent declares none;
   - `quantized_by` left for the user;
   - tags.

   The body follows the project's own card rules:
   - the first sentence says what the file is and is not for;
   - the unflattering metric comes first: a file that fails the bar says so in its first line;
   - then the size against the parent, the recipe as a reproducible command, the fidelity table, and the
     reference named exactly.

   No em dashes and no machine names. A card for a file that failed the bar is still written, but it leads with
   the failure.

## What it refuses

- A recipe that combines nothing (no change from the parent).
- A card claim without a measurement: there is no `--skip-measure`. A user who wants only the rewrite has
  `--rewrite`.
- A parent license it cannot copy.
- The publisher reference without its Python environment.
- A corpus shorter than the window.

## Not in v1 (stated, not hidden)

- **Producing an importance matrix.** Use llama.cpp's `llama-imatrix`, or a published one. A Runner
  `--imatrix-out` is its own item.
- **Scale-only recovery training** (the 27B recovered file). That is shade research; v1's recovery is LoRA through
  `runner train` and `--merge-lora`.
- **Computing an expert-prune plan.** `scripts/moe-prune-plan.py` stays the tool; the command takes the plan.
- **Upload.** The command writes files. Publishing is the user's act.

## Cost on the 8 GB Mac

- Measuring with the default reference scores the parent once and the file once, sequentially, never both
  loaded.
- A 4B BF16 parent is about 8 GB, past the laptop's working set. The command checks `--fit` first and refuses a
  parent that would not fit, naming the lab or a quantized-parent run.

## Gates (the PR's tests)

- **Toy end to end:** a toy F32 parent, `--quant q4_0`, default reference. Writes the file, the record and the
  card. The card's numbers equal the record's, and the record's equal an independent `gold-logits`-style
  computation in the test.
- **Selective precision:** `--target-size` yields the same plan as `type-plan-by-sensitivity.py` on the same
  `--quant-error` report.
- **Failing file:** a toy file forced over the bar gets a card whose first line says it fails.
- **Refusals:** each case above.
- **Card hygiene:** no em dashes, no host paths; the license copied verbatim.

## Size

L, in four parts:

- the verb and orchestration in `main.c`, reusing the rewrite and score paths;
- the C port of the sensitivity ranking;
- the card writer;
- tests and docs.

The publisher-reference path is a subprocess call to `gold-logits.py`, not new numerics.
