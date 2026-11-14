# Phi-3.5-mini-instruct: the LongRoPE factor set follows the sequence (R4.40.1, 2026-11-14)

The read of 2026-11-13 in this folder found that the whole gap at a context over 4,096 was which factor set a
sequence under 4,096 tokens got. This read is of the change that picks the set per sequence. It compares Runner
with itself, bit for bit, against the two arms that were measured against the publisher's checkpoint on
2026-11-13. It adds no new comparison with the publisher's model.

**What it cannot say.** It ran on the Apple GPU only (an 8 GB Apple M1), on the Q4_K_M, with a context of 4,608.
At `-c 8192` this machine cannot hold the file on the device and the CPU run swapped, so that context was not
read. The CPU is held by the fixture gate (`tests/test_longrope_sequence.c`, bit for bit), not by this file.
CUDA keeps one factor set and is unchanged.

**The file.** bartowski/Phi-3.5-mini-instruct-GGUF, `Phi-3.5-mini-instruct-Q4_K_M.gguf`, 2,393,232,672 bytes,
sha256 e4165e3a71af97f1...

**The engines.** Before: Runner at d9f26cc6 (main a6e123c1 plus the R10.27 branch, which does not touch the rotary tables). After: Runner at 056e5bc4
(r4.40-longrope-per-sequence).

**The command**, per arm (TEXT is the first 4,000 or 14,200 bytes of `tests/fixtures/gold-corpus-2k.txt`):

    runner -m Phi-3.5-mini-instruct-Q4_K_M.gguf --score -f TEXT --gpu auto --no-tray -c CTX

| Arm | Engine | `-c` | Tokens | Factor set the load reports | Mean log-probability | sha256 of the log-probabilities |
|---|---|---|---|---|---|---|
| 1 | after | 4096 | 1,186 | short | -3.174393 | eec66f529202e4e7 |
| 2 | after | 4608 | 1,186 | short up to 4,096 tokens, long past it | -3.174393 | eec66f529202e4e7 |
| 3 | before | 4608 | 1,186 | long | -3.558245 | 606808e051612a70 |
| 4 | after | 4608 | 4,343 | short up to 4,096 tokens, long past it | -3.082738 | f13b9562ba71040a |
| 5 | before | 4608 | 4,343 | long | -3.082738 | f13b9562ba71040a |

- **Within the original context** (arms 1 and 2): 0 of 1,185 scored positions differ. A text of up to 4,096
  tokens at a larger context is computed exactly as at `-c 4096`, the arm that read 0.0042 mean KLD from the
  publisher's model on the Q8_0.
- **The defect, on this file** (arm 3 against arm 1): the engine before the change differs at 1,182 of 1,185
  positions, by 0.61 in the scored token's log-probability at the mean and 8.26 at most, and gives the text
  0.38 less log-probability a token.
- **Past the original context** (arms 4 and 5): 0 of 4,342 positions differ. A longer text is computed exactly
  as the long factors alone compute it, the arm that read 0.0045 from the publisher's model past 4,096 tokens.
- Every arm ran on the device (the load line of each names the Metal backend), and no arm recomputed anything:
  a text is one sequence, on the set its length takes from its first token.

The kill written in the item: inside 4,096 tokens at a larger context it must read as `-c 4096` does, and past
4,096 no further from the reference than the long-factor route. Both hold here by identity.
