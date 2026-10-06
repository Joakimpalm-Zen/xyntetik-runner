# `--fit` capture: Runner 1.0.2 on Windows, RTX 3070

2026-10-06, ZEN-GAMING (Windows 11, RTX 3070 8 GB, i7-7700K), nothing else
using the GPU. The released `runner-windows-x86_64.exe` from the v1.0.2
GitHub release. The first 16 MiB of IBM's Granite 4.2 8B Q4_K_M file were
fetched with a ranged read and saved under the model's own file name, so the
output names the model that is being asked about. This is the capture the
README's terminal image is drawn from; the image leaves out the `kv cache`
line, the parenthesis on the `gpu` line and the `note` line, and wraps the
`split` and `verdict` lines.

```text
> curl.exe -r 0-16777215 -L -o granite-4.2-8b-Q4_K_M.gguf https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF/resolve/main/granite-4.2-8b-Q4_K_M.gguf
> runner.exe --version
runner 1.0.2
> runner.exe --fit granite-4.2-8b-Q4_K_M.gguf -c 16384
fit: granite-4.2-8b-Q4_K_M.gguf
  model         granite, 40 layers
  weights       4.98 GiB
  kv cache      2.50 GiB at ctx 16384, f16   |  1.33 GiB with --kv q8   |  1.02 GiB with --kv k8v4   |  0.70 GiB with --kv fp4
  available RAM 7.75 GiB right now
  gpu           NVIDIA GeForce RTX 3070, offload budget 6.95 GiB right now (0.85 GiB of it for embeddings, scratch and headroom)
  split         f16: 35 of 40 layers on the GPU, 1.22 GiB in RAM  | --kv q8: 39 of 40 layers on the GPU, 0.47 GiB in RAM  | --kv k8v4: 40 of 40 layers on the GPU, 0.00 GiB in RAM  | --kv fp4: 40 of 40 layers on the GPU, 0.00 GiB in RAM
  verdict       FITS — 35 of 40 layers on the GPU, 6.53 GiB of RAM to spare at ctx 16384
  note          KV is an upper bound: models with per-layer KV geometry (shared KV, MLA) use less
```

The file on disk is only the header. `--fit` reads the whole model's sizes
from the tensor descriptors, and loading such a file is refused, as it
should be. The split and the GPU budget match the
[1.0.1 clean-setup run](clean-setup-1.0.1-2026-10-03-rtx3070.md) on the same
machine; the RAM figures differ because they are what was free at the time.
