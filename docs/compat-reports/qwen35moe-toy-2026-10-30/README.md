# qwen35moe against transformers, on a converted random model (2026-10-30)

`qwen35moe` is the GGUF architecture of Qwen's Qwen3.5-35B-A3B and Qwen3.6-35B-A3B and of the models built on them
(Ornith-1.5-35B-A3B, Sage-A1.5, occamy-1.0, MiniCPM-V-4.7-35B-A3B's language model, the empero-ai distill).
transformers calls it `qwen3_5_moe`. Its blocks are:

- Qwen 3.5's: three Gated DeltaNet blocks for every block of gated attention, partial rope;
- with routed experts on every block: softmax over all experts, top-k, the top-k weights renormalized;
- plus a shared expert scaled by sigmoid(`ffn_gate_inp_shexp` . x).

Runner's `qwen35` binding already carried that mixer with routed experts and a gated shared expert (`qwen4exp` runs on
it with its hyper-connections, PLE and indexer besides). `qwen35moe` now takes it without those.

## The check

`make_toy.py` builds a small random `Qwen3_5MoeForCausalLM` in transformers' own code (5.16.1), shaped like
Qwen3.6-35B-A3B scaled down:

- 8 blocks: 3 DeltaNet then 1 attention, twice;
- width 128, attention head_dim 32 with a quarter of it roped;
- 8 experts, 3 used, a 48-wide shared expert;
- norms and the DeltaNet decay moved away from their trivial init;
- Qwen3.6-35B-A3B's own tokenizer.

transformers saves it in float32 with the log-probabilities of 200 tokens of the gold corpus, and llama.cpp's converter
(`36a0e2f`, `--no-mtp`) writes an F32 GGUF. Runner scores the same tokens on the CPU (`--score --score-top 20
--gpu off`), and `compare.py` compares the two over the top-20 lists.

| | Positions | Mean KLD | Max KLD | Max \|dlogp\| | Top-1 | Margin-qualified top-1 |
|---|---|---|---|---|---|---|
| Runner against transformers float32 | 199 | **2.9e-08** | 7.9e-07 | 0.0021 | 100% | 100% (86 positions) |

What each feature does to the reference, from `ablate.py` (`ablations.json`); the agreement above is far below every
one of these:

| Taken away | Mean KLD | Top-1 | Margin-qualified top-1 (87 positions) |
|---|---|---|---|
| the shared expert | 6.96 | 0% | 0% |
| the shared expert's gate (fixed at sigmoid(0)) | 4.90 | 2.5% | 1.1% |
| the renormalization of the top-k weights | 0.448 | 50.0% | 66.7% |
| the DeltaNet's gated norm with sigmoid(z) (qwen4exp's) instead of silu(z) | 7.86 | 0% | 0% |
| rope over every head dim instead of a quarter | 0.164 | 63.5% | 81.6% |
| the convolution's history | 8.67 | 1.0% | 0% |

The transformers that ran the ablations (5.22) reproduces the saved reference within 3.7e-4 in log-probability.

## The NextN head

The published files carry one NextN/MTP block, an attention block with the trunk's MoE FFN. `--mtp` used to refuse
every MoE backbone without hyper-connections; it now drafts from this one. On the fixture, greedy output with
`--mtp --draft-k 1` and `4` is byte for byte the output without it (`tests/test_mtp_consume.py`). Acceptance on a real
file is measured with its row.

## Reproduce

```
python make_toy.py OUT_DIR Qwen3.6-35B-A3B-config-dir tests/fixtures/gold-corpus-2k.txt
python llama.cpp/convert_hf_to_gguf.py OUT_DIR --outtype f32 --no-mtp --outfile toy-f32.gguf
runner -m toy-f32.gguf --score --score-top 20 -f OUT_DIR/text.txt -c 512 --gpu off > runner.score.json
python compare.py OUT_DIR/reference.json runner.score.json
python ablate.py OUT_DIR OUT_DIR/reference.json ablations.json
```
