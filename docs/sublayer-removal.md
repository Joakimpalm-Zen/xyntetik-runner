# Sublayer removal: `--remove-sublayer`

A surgery study can find that one block's attention (or FFN) contributes
nothing worth its bytes. Until this feature the only artifact such a finding
could produce was a same-size file with the weights zeroed: the loader
required every projection by name, and no header field could say "this
block has no attention". This document records what the writer does, what
the loader reads, the gates, and the limits.

## Prior art, and the format decision

llama.cpp already has the reading. Its per-layer hyperparameter arrays
(`attention.head_count`, `attention.head_count_kv`, `feed_forward_length`)
may be array-typed for any architecture, and the Nemotron-51B ("deci")
graph skips attention where `n_head(il) == 0` and the FFN where
`n_ff(il) == 0`:

```
if (n_head == 0) {
    // attention-free layer of Llama-3_1-Nemotron-51B
    cur = inpL;
}
...
if (n_ff == 0) {
    continue;
}
```

So the runner declares a removed sublayer with exactly that: a `0` at the
block in those arrays. A private key would have been simpler to gate but
would describe the same fact in a vocabulary only one loader knows. The
runner's own hybrid families already type their blocks with zeros in these
arrays (nemotron_h, granitehybrid); the removal reading applies to every
other architecture, and the hybrids keep their own.

Two readings a zero could also carry are refused, not guessed: an
`attention.head_count_kv` of 0 where `head_count` is not (llama.cpp's
"linear attention" block), and a non-zero entry that differs from the rest
(heterogeneous head counts).

## What the writer does

```
./runner -m parent.gguf --quantize cut.gguf --remove-sublayer attn:48
./runner -m parent.gguf --quantize cut.gguf --remove-sublayer attn:48,mlp:12 --quant q8_0
```

- Scope, per removed block: every `blk.N.attn_*` tensor except the
  `attn_norm*` pre-norm, or every `blk.N.ffn_*` tensor except `ffn_norm*`.
  Projections, Q/K norms, sinks, gates and biases go; the pre-norm and any
  post-branch norm stay. This is scope 1 of the project's byte accounting for
  the Gemma 31B `attn:48` cut (the six tensors the fidelity number covers)
  plus the branch's small companions. The kept norms are kilobytes, and
  keeping them means the removed file is the same object as the parent with
  the branch's output projection zeroed, which is what the gate compares.
- Metadata: the scalar `head_count` and `head_count_kv` (or
  `feed_forward_length`) become `u32[block_count]` arrays, entry by entry
  from the source when it was already an array, with `0` at the removed
  block. `block_count` does not change.
- The tensor directory and data section are rebuilt without the dropped
  entries; every survivor is copied byte for byte (or requantized if
  `--quant` asks). The writer prints, per removal, the tensors and bytes it
  dropped.
- Refused before any byte is written: a block outside `block_count`, a part
  already removed, a block with no such tensors, a fused-QKV export, a MoE
  FFN (`ffn_gate_inp` or `*_exps` present), the hybrid families
  (nemotron_h, nemotron_h_moe, granitehybrid, qwen35, or any file with
  `ssm.conv_kernel`), gemma-4 E-series files (per-layer embeddings), and a
  file declaring a NextN/MTP head. A malformed spec (`ffn:0`, `attn:x`, a
  part listed twice) is refused with the entry quoted.

## What the loader does

- Geometry: an array `head_count` gives `n_head` as its maximum and marks
  each `0` block as attention-removed; `head_count_kv` zeros must pair with
  those blocks; `feed_forward_length` zeros mark FFN-removed blocks and the
  width is the maximum over the rest.
- Binding: a removed attention binds no projection, and the loader checks
  the file does not carry one anyway (`error: blk.N is declared without
  attention (attention.head_count 0) but the file carries blk.N.attn_q.weight`).
  A removed FFN likewise. A tensor missing without a declaration is the same
  `error: missing tensor` it always was.
- Forward: the block's branch is omitted (the same `skip_mixer` /
  `skip_ffn` flags the nemotron_h blocks use), not computed as zero and
  added. A removed attention reserves no KV rows: the per-layer KV table
  reads 0 heads for it, so the cache shrinks by that block's share at every
  context length. The `-v` banner lists `sublayers removed  attn:48`.
- Refused by name: GPU offload (`--gpu auto` with a backend present says to
  rerun with `--gpu off`), `--lora`, `--train`, the MTP head on such a file,
  and the families the writer refuses.

## Gates (`tests/test_remove_sublayer.py`, in `make test`)

- **Absolute anchor.** A block whose attention output projection is all
  zeros contributes exactly nothing (0 * x = 0, and the norm of a zero vector
  is zero), so the removed file must `--score` bit-identically to the parent
  with that one tensor zeroed. That is an independent path through the full
  attention math (projections, cache, softmax, the zero matvec, the residual
  add), not a transcription of the omission. The FFN case anchors on
  `ffn_down` zeroed. Both parts composed anchor on both tensors zeroed.
- **The gate can fail.** The removed file must differ from the untouched
  parent.
- **Byte accounting.** The dropped tensors are absent, every survivor is
  byte-identical, the data section is exactly the survivors' aligned spans,
  and the bytes the writer reports equal the parent's own payload for those
  tensors.
- **KV.** On a two-block fixture with one attention removed, the `kv cache`
  banner reads half the parent's at ctx 65536.
- **Declaration.** The arrays carry the zeros; a requantized removed file
  keeps them; an undeclared missing tensor is refused; a declared-absent but
  present tensor is refused (a hostile fixture from
  `make-test-model.py --declare-removed`).
- **Refusals.** MoE MLP, hybrid, bad specs, double removal, `--lora`,
  `--train`, and the GPU path (on a Metal host).

The model-level counterpart of the anchor was measured before this
feature existed: on the real Gemma 31B artifact, the masked and the zeroed
forms scored identically at float64 on all three fidelity metrics.

## Interop

A removed file loads only in a runtime that reads the per-layer zeros. Stock
llama.cpp reads the arrays for every architecture, but only its deci graph
honors a zero; for any other architecture it fails to find the tensor and
refuses the file, which is the right failure. Measured 2026-09-04 on the
real Gemma 4 31B cut: llama.cpp b10076, which loads the gemma4 parent,
refuses the removed file with `llama_model_load: error loading model:
missing tensor 'blk.48.attn_q.weight'`. It never loads it wrong. State this
on any card, in the same place as the fidelity number.

## Measured on the real file (2026-09-04)

Gemma 4 31B Q4_0 (17,651,001,568 B), `--remove-sublayer attn:48`, CPU path
on a 128-thread x86 host:

| quantity | measured |
|---|---|
| tensor payload dropped | 74,319,872 B (six tensors), file 17,576,681,600 B |
| survivors byte-identical to the parent | 827 of 827 |
| KV freed at ctx 4,096 / 32,768 (f16) | 64.0 MiB / 512.0 MiB |
| `--score` vs the same cut as a zeroed-weights file, 4,562 positions | max abs difference 0.0 |
| raw-protocol KLD vs the parent, 400 positions | 0.0239, margin-qualified top-1 99.25% |

The zeroed-weights file was built earlier by a different tool, so the
bit-identity row is the fixture anchor at scale: the omitted branch and the
computed-zero branch are the same function on a 60-block model.

## What remains

- Device paths: CUDA already carries `skip_mixer` / `skip_ffn` for
  nemotron_h; the dense decode loops need the same checks before the GPU
  refusal can go. Metal has no per-block skip at all.
- MoE FFN removal (drop the router, experts and shared expert), which the
  Glimmer Q4_K byte question needs.
- The C2 E4B MatFormer slice is a different transform (per-layer FFN
  narrowing, which gemma-4 already expresses with `feed_forward_length`
  arrays) and is not covered here.
