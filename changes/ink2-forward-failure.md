- **A forward that fails on the device is reported as a device failure, not a full context, and no longer
  crashes** (Inkling 2 Mini on CUDA).
  - An Inkling 2 forward on CUDA that fails returns no logits, and its instance computes no more rows. The engine
    read the empty row as a context overflow, so the CLI said `error: prompt exceeds context`, and the server
    answered 400 `context_length_exceeded`. It also kept feeding the prompt's later chunks into the failed
    instance. During generation it handed the empty row to the sampler, a segmentation fault (exit 139 on the
    16 GB file, RTX 5060 Ti, at main `0548c847`).
  - The engine now records a failed forward (`model_forward_failed`). It stops feeding at the failed chunk, and
    generation ends on it.
  - The CLI names the failure and exits 1. One-shot, chat, `--bench-json` and `--verify` (UNVERIFIABLE, exit 3)
    all do, and no receipt is written for a cut-short run.
  - The server answers HTTP 500 `device_failed` when the device fails under the prompt. A turn already generating
    ends with `finish_reason: "error"`. The decision scorer names the device too.
  - `tests/test_inkling2_device_fail.py` injects the failures on a CUDA host (it skips elsewhere).
