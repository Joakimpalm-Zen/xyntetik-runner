# Inkling 2 Mini protocol

This records protocol checks, not model admission or numerical fidelity.
The native chat renderer and tokenizer are checked here. Generated tool-call
parsing, truncation closure and reasoning-channel transitions are still pending. No weights are needed for the tokenizer check.

## Tokenizer

The publisher's [tokenizer.json](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/b85c3cfa615fc413cc9af3379f491b963f3b586a/tokenizer.json)
uses the o200k pre-tokenizer and `ignore_merges=true`. The converter names
that rule `tokenizer.ggml.pre=inkling2`, with `tokenizer.ggml.model=gpt2`.
The validated GGUF metadata has 200,058 tokens, EOS 200006
(`<|content_model_end_sampling|>`), padding 199999, and `add_bos_token=false`.
Runner now recognizes that name instead of assuming the GPT-2 split.

On 2026-11-22, using only metadata from
`unsloth/Inkling-2-Mini-GGUF` revision
`45fd08764f2b0bf16b03a5483a80e5c7aaec8772`, file
`Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`:

- Before the fix: 21 of the fixed corpus's 721 strings differed.
- After the fix: 0 of 721 differed against the publisher revision above.
- Negative control: remove the single `H` + `E` merge from the publisher
  tokenizer. The same differential fails, 1 of 721 strings differs:
  `HELLO WORLD`. Its reference IDs are `[146066,148284,199464,146001]`;
  the control gives `[39,146981,46,199464,146001]`.
- `test-tokenizer-classes` also fails before the fix on the new name, and
  passes after it, including the ignore-merges and no-BOS settings.

The captured reference is
`tests/compatibility/tokenizer-references/inkling-2-mini.tokenizer-json.json`.
With a metadata-only GGUF from that file, reproduce without downloading
weights:

```sh
make STRICT=1 difftok test-tokenizer-classes
./test-tokenizer-classes
python3 scripts/difftok.py --gguf /path/to/tokenizer-only.gguf \
  --ref-ids tests/compatibility/tokenizer-references/inkling-2-mini.tokenizer-json.json
```

README impact: none at this step. This tokenizer correction does not add a
model to the support list or publish a model artifact; the admission, site
and model-card updates remain dependent on their engine and evidence gates.

## Native chat renderer

The oracle is the publisher's [chat_template.jinja](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/b85c3cfa615fc413cc9af3379f491b963f3b586a/chat_template.jinja),
SHA-256 prefix `53a9035f32559457`, pinned in the conformance baseline.
It uses message/content-channel tokens, emits a thinking-effort system turn
before the first non-system message, and appends no assistant generation
prefix. Historical reasoning, text parts, calls and tool results retain their
separate segments. Tool declarations sort all keys; call arguments sort only
the outer keys, as the template does.

On 2026-11-22, `scripts/template-conformance.py --family inkling2
--require-tokens` compared 35 cases, all byte-identical and token-identical,
with no allowlist or backlog entries. They include all named levels (`none`,
`minimal`, `low`, `medium`, `high`, `xhigh`, `max`) and numeric 0, 0.37, 0.99.
Changing the output's single colon in `Thinking effort level:` to a question
mark makes the same gate fail; restoring and rebuilding makes it pass.
`test-template-families` checks this renderer's capacity and mode contract.

This step renders tool histories; it does not yet certify generated calls,
forced reasoning closes or the complete CPU suite. The chat renderer's
structured source fields preserve content-part and reasoning boundaries;
other API adapters still need their protocol checks before admission.
