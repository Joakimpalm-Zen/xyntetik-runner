# R4.48.12 Claim 2: which llama.cpp cell the wording's figures quote (2026-11-25)

When this was written, Runner's two sixteen-thread cells and mainline llama.cpp's eight had run or were running, and their cell times were known. No identity count (K_R, K_L) had been computed. This changes no bar, cell, setting or outcome rule. Every outcome branch in PREREG 2.4 is chosen by the identity counts alone.

PREREG 2.4's wordings quote L_batched and L_alone tok/s at sixteen threads, but llama.cpp has eight sixteen-thread cells: alone or batched, each with flash attention on or off and a micro-batch of 512 or 1. The PREREG names none of them as the quoted one.

**Rule.** The quoted llama.cpp cell is the one whose flags equal PREREG 1.2's llama.cpp line with llama.cpp's own defaults for the two varied settings: flash attention and micro-batch, as that build (`56eea9e4`) resolves them on this CPU. The lab reads both from the build's help output and from the server's startup log, then names the cell.
- If the defaults do not match any cell exactly, the closest cell is quoted and the mismatch is stated.
- The other three cells of each mode are reported beside it in the same table as information.
- The choice is never made by a cell's figure.

The rule is "the configuration a user gets by default", so it does not depend on which cell is fastest.

**The 20:42Z host load spike** (other tenants; peak 1-minute load 1356) overlapped rounds c2_16 to c2_18 of `llama-t16-batched-faon-ub1`. It is recorded beside those rows, and no request in them timed out or retried. Identity is unaffected.
