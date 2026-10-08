# Raw records

The per-run records of this evaluation are published as a dataset, not kept
in the repository:
[Joakimpalm-Zen/Qwen3-4B-tooluse-shifted-eval-record](https://huggingface.co/datasets/Joakimpalm-Zen/Qwen3-4B-tooluse-shifted-eval-record)
(the 17 records of 2026-09-23 to 09-25, with `SHA256SUMS`). They were here
until 2026-10-08 (last repository commit 5badaf75).

`scripts/eval-tooluse-shifted.py` still writes new records into this
folder; `*.json` here is ignored by git, so publish a new record to the
dataset instead of committing it.
