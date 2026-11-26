# R4.48.12 Claim 2: how the statistics apply PREREG section 2 (2026-11-26)

These were decided after all three engines' sixteen-thread cells had run, and after the lab's dry run had printed its information output. They change no bar, cell, setting or outcome rule. Each is the plain reading of PREREG section 2, or a disclosed substitute where the named evidence cannot exist.

1. **The cell set: sixteen threads only, for now.** The one-thread cells run after the release, so every Claim 2 figure published before them is the sixteen-thread half of the pre-registered matrix. PREREG 2.4's wordings ("at 1 and 16 threads", "all 4 settings", "16 settings") are NOT used until that matrix is complete. Until then the figures are stated as sixteen-thread results, provisional, with the one-thread cells named as pending, and the status stays INFORMATION.
2. **Byte identity** means the target's generated ids and the bytes they decode to with the file's vocabulary. Runner's client appends the stop token 200006 on stopped turns in every Runner cell alike. Each record is checked to be the receipt, or the receipt plus 200006 where `stop_token_appended`.
3. **Records.** The last complete line per target id is used (the client's resume rule). Incomplete or torn lines are counted, and there are none. Prompt ids are checked equal across each engine's cells.
4. **Throughput, PREREG 2.3 item 4.**
   - `T_batch_wall_s` runs from barrier release to the ninth response.
   - Alone rounds use the same wall, prefill included.
   - Tokens are the generated id count, as in Claim 3's interpretation 7, with the client's usage-based figure shown beside wherever it differs.
   - Each figure is the median and IQR over 20 rounds of the quoted cells (fa on, ub 512), with its load-table row and residency beside it.
5. **llama.cpp's co-batching (CB_L) is inferred, not observed.** PREREG 1.2's server line has no `--metrics`, and the per-decode batch lines sit below the default log level. The evidence is:
   - the server's source at the pinned commit: one shared batch per `update_slots` iteration, split into n_ubatch micro-batches;
   - the slot launch and release lines: concurrency mean, max, and the share of time with two or more slots.

   CB_L is true for ub 512 and false at kernel level for ub 1. It is stated as "inferred from the source and the slot log".
6. **Runner's co-batching evidence.** PREREG 2.3 names `RUNNER_DEBUG_ROUTE=1` `path=` lines, which cannot appear for this architecture: the batch forward returns through the Inkling forward before the route logging. That is risk-note 12, and no answer is affected. The named `/metrics` counters (`runner_batch_steps_total`, `runner_batch_sequences_total`) are the evidence shown, together with a source check at the measured build (`7bf8b64d`) that the Inkling batch path decodes one sequence at a time (no multi-sequence matrix products). If that check finds otherwise, the wordings' "sequentially one decode at a time" premise fails and is ruled on before any figure is published.
7. **Factor counts.** Each counts the targets that differ between the reference cell (alone, sixteen threads, fa on, ub 512; for Runner, alone and sixteen threads) and the cell that changes only that factor.
8. **Companions** are information, not a PREREG metric.
9. **Unsloth.** Claim 2 defines no comparison between engines, so Unsloth's additional column holds K_U with its Wilson interval, the factor counts, CB_U, U_batched and U_alone, and the companions. It has no pairs and no combined label. The supervisor's earlier "its pairs only" carried Claim 3's form over in error.
