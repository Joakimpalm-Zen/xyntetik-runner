# The audit demo: record, sign, hand over, verify

`scripts/audit-demo.sh` runs the whole chain on one machine in under a
minute: a 1,000-token sampled run is recorded and signed, a verifier
replays it trusting only the signer's public key, and three tampered copies
are refused. It is the shortest way to see what a Runner record proves.

```sh
scripts/audit-demo.sh                    # fetches SmolLM2-135M-Instruct Q8_0 (145 MB) from bartowski/SmolLM2-135M-Instruct-GGUF
scripts/audit-demo.sh path/to/model.gguf # or any local GGUF
```

## What it does

1. `--keygen` makes an Ed25519 signing key and prints its public half.
2. The model writes 1,000 tokens at temperature 0.8, seed 42, with
   `--transcript run.json --sign-key key.json`. The record holds the model's
   and the binary's sha256, the profile and settings, the seed, the prompt
   and output token ids and bytes, a chain hash over all of it and a
   signature over the chain.
3. The verifier runs `--verify run.json --require-signed --trust-key KEY`:
   it checks the chain and the signature, then replays the run and compares
   every token. `VERIFIED`, exit 0.
4. Three forgeries:
   - one output token changed: `UNVERIFIABLE`, the chain hash no longer
     recomputes;
   - the same change with the chain hash recomputed and the signature
     removed, checked by a verifier that requires the signature:
     `UNVERIFIABLE`, unsigned;
   - the same consistent forgery, checked by a verifier that does not
     require a signature: `DIVERGED at token 500`. The hash can be forged;
     the model's output cannot, because the verifier computes it again.

## Measured

On the 8 GB M1 (Metal), SmolLM2-135M-Instruct Q8_0, 2026-10-02: recording
10.0 s, the verifying replay 9.7 s, each forgery a few seconds more, the
whole script 29 s. A larger model takes as long to verify as to generate.

## What it proves, and what it does not

Proven: this output is what this model file, this binary, these settings
and this seed produce, and the record was signed by the holder of the key
whose public half the verifier trusts. A change to any token, setting or
digest is caught.

Not proven: who ran it, on which machine or when. A signature says who
signed, not who computed; anyone with the model and the binary could have
produced the same bytes. The replay guarantee is the
[determinism scope](determinism-scope.md): the same binary on the same
kind of host replays bit-exactly; another build or another instruction set
replays token-exactly at best.
