# R4.48.12 pre-registration: addendum, 2026-11-24, a third engine (Unsloth's llama.cpp fork), the owner's decision

The owner decided on 2026-11-24 that Runner is also compared with Unsloth's llama.cpp fork. Unsloth publishes the GGUF files, ships its own llama.cpp build for them, and states that its build decodes Inkling 2 Mini up to 1.9x faster than mainline. This addendum adds that build as a third engine beside mainline llama.cpp.

It is committed BEFORE any run of Unsloth's build. When it was written:
- Runner's and mainline llama.cpp's Claim 1 outputs existed (records `e724fca8d`);
- the 3-prompt control was being rerun under erratum 2;
- no teacher-forced verdict had been computed.

It changes nothing that `PREREG.md`, `ERRATUM-2026-11-23.md` or `ERRATUM-2026-11-24.md` fix for Runner and mainline llama.cpp. It only adds a column.

## The third engine

- **Source:** `unslothai/llama.cpp`, release tag `b12649-mix-c7059d6`, commit `a5aa5bacbc2bda2478ee2985d42f36211ffc7da9` (published 2026-11-23T14:33:09Z).
- **Built from source, configured exactly like the mainline builds,** so the compiler and flags are the same for both forks:
  - the CPU build uses `build-cpu-server`'s configure line (`-DGGML_NATIVE=ON`, the server on, CUDA, Metal and Vulkan off; gcc 13.3);
  - the CUDA build uses `build-cuda`'s line (`-DGGML_CUDA=ON`, architecture 120, the box's CUDA 13.2).
- **Its published prebuilts are measured in the speed sets** (see below), because Unsloth's speed claim is about them: `app-b12649-mix-c7059d6-linux-x64-cpu.tar.gz` and `app-b12649-mix-c7059d6-linux-x64-cuda13-newer.tar.gz`, with each asset's sha256 recorded.
- **Wrong commit voids an Unsloth run** under PREREG 4.2(2), as for mainline.

## The same file

Unsloth's repository at revision `8078d149286eec7193795413b9e0643da6d11fa1` holds two files that are byte-identical to the file already pinned for Claim 1, sha256 `2da51788d43a17e1f033bec420fa83ba9cb87ac1f0130dc9135a4d2ed2678b68`, 22,867,452,896 bytes:
- the root `Inkling-2-Mini-UD-Q8_K_XL.gguf`, the file Unsloth recommends for its build, updated 2026-11-23;
- `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`.

All three engines therefore read the same bytes. The same holds for the secondary files if their hashes match at run time; any mismatch is recorded and that file is fetched at its pinned revision.

## The same settings and claims

- **Settings:** PREREG 1.2's llama.cpp line, verbatim: `-ngl 0 -t 16 -c 4096 -np 1 --jinja --reasoning-effort low --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 --reasoning-budget -1`, with `-n 256`.
  - The same client (`claim_client_llama.py`) and the same request body, `cache_prompt: false` included.
  - No speculative drafter (DSpark): Unsloth's build runs without `-md`, as mainline does.
- **The pre-flight:** Unsloth's rendered prompt ids are compared with the reference's, as for mainline. A difference fails that prompt under PREREG 1.3 and is logged before its run.
- **Claims 1, 2 and 3:** Unsloth's build runs the same cells as mainline llama.cpp.
  - Its Claim 1 outputs are committed as immutable records before their teacher-forced verification, which uses `tools/verify_tf.py` version 2.
  - The paired statistics for Runner against Unsloth's build are computed exactly as PREREG 0.2 and 1.3 compute them for Runner against mainline. They are reported as an additional column in the same tables.
  - The pre-registered Runner-against-mainline statistics and wording are unchanged.
- **Fidelity reads (information):** Unsloth's build gets the full-vocabulary KL against the publisher's float32 on R4.48.19's fresh chat and raw reads, with `compare_full.py` and R4.48.15's method, beside Runner's routes and mainline.
- **Speed (information, not a claim):** the speed sets on the measurement host, R4.48.15's method, CPU and CUDA. Each set carries these rows in one window:
  - Unsloth's build from source;
  - its prebuilt;
  - mainline `56eea9e4`;
  - Runner's routes.

  Unsloth's published figure ("up to 1.9x faster than mainline llama.cpp for Inkling-2-Mini decoding": UD-Q4_K_XL, tg1024, F16 KV cache, its GPUs) is quoted beside ours, never as ours.
