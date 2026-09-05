# Post-quantum receipts: ML-DSA-44 beside Ed25519 (2026-09-05)

**Status:** implemented and gated on branch `r12-frontier`; not in a
release. Frontier item R12.2 asked whether receipts should carry a
quantum-resistant signature. This note gives the measured answer: the cost
is small enough that the question is policy, not engineering, and the
branch makes it a per-key choice (`--keygen-algo ml-dsa-44`) with Ed25519
unchanged as the default.

## Why

A receipt is an audit record with a retention measured in years. Ed25519
rests on the discrete logarithm problem, which a large quantum computer
breaks; "harvest now, forge later" applies to signed logs as much as to
encrypted traffic, because a forged receipt dated to the past is worth the
same as a decrypted message. ML-DSA (FIPS 204, August 2024) is the NIST
standard for post-quantum signatures, and the 44 parameter set is the
smallest one, with a security category comparable to Ed25519's. The
regulatory frame the receipts are built for (EU AI Act Article 12 logging,
Article 50 transparency) expects records to hold up for the life of the
system, so the question is not whether to move, only when and at what cost.

## What was built

- `src/mldsa/`: the pq-crystals ML-DSA reference implementation (public
  domain / Apache 2.0, FIPS 204 standard branch, commit and license in
  `src/mldsa/LICENSE`), pinned to ML-DSA-44, with two recorded changes:
  key generation takes a caller-supplied 32-byte seed (the signkey file
  stores the seed, as it does for Ed25519), and hedged signing is compiled
  out so signing is deterministic (a receipt re-signed with the same key is
  byte-identical, which is what a receipt is for).
- `src/mldsa.h/.c`: a three-function wrapper (keypair, sign, verify).
- The receipt plumbing is algorithm-generic: the signkey file's `algo`
  field selects the primitive at signing time, the record's signature
  object names it, and the verifier accepts either. `--keygen-algo`
  (default `ed25519`) picks the key type at generation. `--trust-key`
  accepts `sha256:` plus the digest of the key bytes, because a 2624-hex
  ML-DSA key does not belong on a command line or in a policy file.
- Existing Ed25519 receipts and key files verify and load unchanged; no
  format version bump.

## Anchors

- `tests/test_mldsa.c` checks the vendored code against the NIST ACVP
  FIPS 204 vectors (`tests/mldsa_kat.h`, generated from the published
  `internalProjection.json` files): key generation from a seed reproduces
  the published public and secret key byte for byte, and deterministic
  signing over the published message reproduces the published signature.
  The external anchor is NIST's, not the module's own output.
- `tests/test_envelope.c` signs and verifies a receipt under both
  algorithms, and rejects one flipped byte in the body, one in the
  signature, and an algorithm the verifier does not know.
- `tests/test_receipts.py` runs the binary end to end: keygen, sign, chain,
  verify under `--require-signed` with the key in full and as a digest,
  and the same forgeries as the Ed25519 tests, all `UNVERIFIABLE` before
  replay.

## Cost

Measured on an Apple M1, single thread, reference implementations for
both (the Ed25519 in this tree is the TweetNaCl subset, chosen for
auditability, not speed).

| primitive | public key | signature | keygen | sign | verify |
|---|---:|---:|---:|---:|---:|
| Ed25519 (TweetNaCl, in tree) | 32 B | 64 B | 756 us | 769 us | 1474 us |
| ML-DSA-44 (pq-crystals ref) | 1312 B | 2420 B | 55 us | 270 to 390 us | 74 us |

Signing time for ML-DSA varies with rejection sampling; two runs gave 270
and 392 us. Against the Ed25519 implementation actually in the tree,
ML-DSA-44 signs 2 to 3 times faster and verifies 20 times faster. The costs
that are real:

| measure | Ed25519 | ML-DSA-44 |
|---|---:|---:|
| receipt for an 8-token run, unsigned 1156 B | 1404 B | 8678 B |
| signature object (key + signature as hex) | 248 B | 7522 B |
| key file | 219 B | 2781 B |
| runner binary (macOS arm64) | 1,146,376 B | 1,187,480 B (+41 KB, 3.6%) |
| end-to-end record and sign, tiny model, median of 5 | 18 ms | 15 ms |
| end-to-end verify, same | 15 ms | 13 ms |

The receipt grows by 7.3 KB per record. For a chained log of real
conversations (prompt and output text are in the record) that is a
fraction; for the tiny records above it is 6 times the record. Wall time is
indistinguishable from the Ed25519 path at the process level.

## Verdict and the open policy question

R12.2 is **validated**: the thesis was "quantum-resistant receipts are
worth having and the cost is bearable", and both halves measured true. What
remains is an owner decision:

1. **Keep as an option (default Ed25519)**: what the branch does. A tenant
   who needs it picks it per key; the format and verifier already speak
   both. Nothing changes for existing users.
2. **Make ML-DSA-44 the default**: every new key is post-quantum; receipts
   are 7 KB larger; third-party verification needs an ML-DSA library
   (Python `dilithium-py`, liboqs, OpenSSL 3.5 all have it) instead of an
   Ed25519 one, which every language ships today.
3. **Hybrid (both signatures on every record)**: the belt-and-braces
   answer some standards bodies favour; not built, would be a small
   addition (two signature objects, verifier requires both).

Recommendation: option 1 now, and revisit the default when the
compatibility program's third-party verifier (a script that verifies a
receipt with nothing but a standard library) can do ML-DSA without an extra
dependency. That is the argument that carried Ed25519 as the default, and
it still holds.

## Not done

- No independent third-party verification of an ML-DSA receipt in the
  pytest (no ML-DSA Python library is installed on this host); the ACVP
  known answers are the independent anchor instead.
- Hybrid signatures (option 3).
- The site and the Hugging Face cards mention Ed25519 by name in four
  places; if this ships, those move with the README in the release pull
  request (AGENTS.md rule 5).
