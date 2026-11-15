- **The reference script no longer drops scored positions that it never needed text for.**
  `scripts/gold-logits.py` checks that a position's decoded text prefix tokenizes back to the same ids before it sends
  that text to an endpoint. The check also ran when every side was a score file, which is what `runner derive` and the
  compat reports use, and where nothing is sent to anyone. A position that failed it was dropped without a line and
  without being counted.
  - Measured on SmolLM2-1.7B: 5 of 150 positions dropped (four with a prefix ending inside a multi-byte character, one
    ending in a double line break), in a run whose summary said 0 skipped.
  - With score files on every side no prefix is built and no position is dropped for it. With an endpoint side the
    check stands; a dropped position is printed with its reason and counted in the summary's new
    `positions_skipped_prefix`.
  - Earlier records state the number of positions they scored, so their figures stand for the positions they name;
    the dropped positions were not a random sample (they follow multi-byte characters and paragraph breaks), so a
    re-measured file can read slightly differently.
