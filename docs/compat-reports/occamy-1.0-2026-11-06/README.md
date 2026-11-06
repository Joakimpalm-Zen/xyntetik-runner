# occamy-1.0: Accio-Lab's files against their BF16 parent in float32 (2026-11-06)

occamy-1.0 is Accio-Lab's finetune of Qwen3.6-35B-A3B. It is not a Qwen release. It is listed Upcoming, not admitted:
none of the publisher's K-quant files passes the fidelity bar against the publisher's own parent.

**What is not flattering, first:**
- **No file passes.** The bar is mean KLD at most 0.05 and margin-qualified top-1 at least 97%.
  - The Q8_0 passes the margin half (98.74%) and misses the KLD half by 0.009 (0.0590).
  - The Q6_K and Q5_K_M miss both (0.0836 and 0.0970), and the Q4_K_M misses by far (0.258).
- **The distance is larger than the base's.** By the same method on Qwen3.6-35B-A3B, bartowski's Q8_0 reads 0.0349 and
  his Q4_K_M 0.0475. occamy's files read 1.7x (Q8_0) to 5.4x (Q4_K_M) that.
  - The Q5_K_M and Q6_K sit near one level, and the Q6_K's margin-qualified top-1 is under the Q5_K_M's. That looks like
    a floor from something other than quantization. It is inferred, not shown.
- **Where the distance comes from is not read.** Checked and ruled out:
  - the parent's weights have not changed since their upload (every safetensors and config sha256 is equal from the
    upload revision ea17490b to 72ca9603);
  - the GGUF files have not changed since their 2026-09-18 release;
  - the config equals Qwen3.6-35B-A3B's except bookkeeping and `mtp_num_hidden_layers: 0` (the MTP head is a separate
    repository).
  - A conversion of the parent by the lab, read the same way, would separate the publisher's files from the engine.
    It has not been run.
- **Scope:** one 300-position window over one corpus, and the text path only. Task quality is not measured.

The kills were declared before each read: the tokenizer, then the template, then the bar. The ladder was Q4_K_M,
Q5_K_M, Q6_K, Q8_0, and the smallest passing file would have been the row's.

## The files and the reference

- **Parent:** `Accio-Lab/occamy-1.0` at `72ca96031d8d11650882b5fae8dad88412a76155` (apache-2.0, base_model
  Qwen/Qwen3.6-35B-A3B), held whole from the Hub (14 safetensors shards, 70.2 GB).
- **Reference path:** transformers 5.16.1, the text model of the composite checkpoint (Qwen3_5MoeForCausalLM), with
  `scripts/gold-logits.py --upcast-at-use`: float32 arithmetic on the stored bfloat16, cast at use per decoder block
  (40 blocks).
- **Window:** 300 positions at stride 15 over `tests/fixtures/gold-corpus-2k.txt`, token-id keyed, every position
  matched by id, no token spelling mismatches, no dropped reference mass.
- **Files:** Accio-Lab's own GGUF repository, `Accio-Lab/occamy-1.0-GGUF` at `15027de4`. Each file's sha256 was checked
  against the Hub's LFS record as it was held.
  - Q4_K_M: 21,166,757,696 bytes, sha256 `ffb25f763ff9c27f...`.
  - Q5_K_M: 24,729,131,008 bytes, sha256 `977c0be5efa18214...`.
  - Q6_K: 28,514,152,448 bytes, sha256 `0b46c3550a654888...`.
  - Q8_0: 36,903,139,456 bytes, sha256 `9bd6eceae8434098...`.
  - The Q5_K_M and Q6_K headers record an importance matrix (510 entries, 32 chunks); the Q4_K_M and Q8_0 headers
    record none.
- **Runner:** main e11ffca (runner 1.14.0), `--score --score-top 20 --score-tokens 4510 -c 4510 --gpu off` on the
  host's CPUs.

## Against the reference

| File | Bytes | Mean KLD | Max | Top-1 | Margin-qualified top-1 | Bar |
|---|---|---|---|---|---|---|
| Q8_0 | 36,903,139,456 | 0.0590 | 3.736 | 93.00% | 98.74% | fail (KLD) |
| Q6_K | 28,514,152,448 | 0.0836 | 3.008 | 88.67% | 95.38% | fail |
| Q5_K_M | 24,729,131,008 | 0.0970 | 1.961 | 87.33% | 95.80% | fail |
| Q4_K_M | 21,166,757,696 | 0.2578 | 4.197 | 77.00% | 85.71% | fail |

Per position: `gold-q8_0.json`, `gold-q6_k.json`, `gold-q5_k_m.json`, `gold-q4_k_m.json` (FILES.sha256).

## The tokenizer

- **difftok against Accio-Lab's tokenizer.json at 72ca9603:** 0 of 721 corpus strings and 0 of 247,091 vocabulary
  entries differ (953 partial-UTF-8 entries left out).
- **That file is the transformers 5.x re-save of Qwen3.6's tokenizer:** the qwen2 split, `\p{L}+` with no `\p{M}`. It
  is equal as JSON to empero-ai's (R4.27.1), and occamy's config.json was saved by transformers 5.8.1.
- **The GGUF header labels it `qwen2`,** so Runner reads it with that split by the label, with no per-file rule.
- **Against Qwen3.6-35B-A3B's tokenizer.json:** 3 of 721 and 5,503 of 247,091 differ, by design.

## The template

- The GGUF's template (7,764 bytes, sha256 `e84f32a2`) is byte-identical to Accio-Lab's chat_template.jinja at
  72ca9603 and to Qwen3.6-35B-A3B's. Runner picks its `qwen36` renderer with qwen3_xml tools.
- `scripts/template-conformance.py --require-tokens`, family `occamy-1.0-qwen35moe` (the file's own template as the
  oracle): 25 cases, 25 identical in text and tokens. One case is not expressible: system-mid-history, which the
  reference refuses.
