# R3.30: a device-resident expert cache for host-resident experts (design note)

**Status: design only. No code exists, and none is written before this note is approved.**

**Step 1 (2026-11-04): not a kill.** A static hot set picked from the other prompts' frequencies serves 0.785 of
decode routings from the card against today's 0.429, a projected 1.66x, so step 2 is a placement change with no
cache ([step1/README.md](step1/README.md)).

**Kill numbers:**
- **Speed:** under 1.3x decode over today's partial-offload placement of the same file on the same card kills
  the item.
- **Identity:** any identity miss (defined below) kills it.
- **Earliest read:** the first boundary is an offline simulation over a routing trace, before any kernel code
  (step 1 below). If no simulated policy projects 1.3x there, the item stops that day.

## Prior art: llama.cpp #29887 (merged 2026-10-07, d6cf9acb)

The PR is a port of the qvac-fabric MoE cache.
- **Mechanism:** `MUL_MAT_ID` on host-resident experts runs on the GPU through an LRU of expert slots in
  device memory. Only the misses are uploaded, inside the step.
- **Scope:** small batches only (at most 32 tokens); larger batches take the regular offload path.
- **Banks:** layers with the same expert tensor layout share banks and one LRU (`slot_map` per layer,
  `key_of = layer * n_expert + expert`). A plan step gives each selected expert a slot, and the misses evict the
  least recently used.
- **Size:** `--moe-cache-mib N`, with about 10% of expert bytes suggested. Single device.
- **Results:** reported on Qwen3.8-Flash-Next Q4_0 (65.4 GiB of experts, 512 experts, top-10):
  - RTX 4090: 1.57x at a 6.4 GB cache (hit rate 72%) and 1.62x with every expert layer on the host (77%).
  - RTX 5090: 1.77x and 2.20x (89%).
- **Their host:** PCIe 4.0 x16 and a 16-core EPYC.

## Our shape: Mellum 2.1 Q8_0 on the RTX 3070 (8 GB), i7-7700K, 16 GB RAM

**The file** is `JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF@4159d684`, Q8_0, 12,925,417,760 bytes. Header read
by range request:
- 28 layers, every one MoE: 64 experts, top-8, expert FFN 896 wide, `n_embd` 2304.
- **Experts:** 10.98 GiB of the file. One expert (gate, up and down at Q8_0) is 6,580,224 bytes, and one layer's
  bank is 401.6 MiB.
- **Everything else:** attention and dense tensors 0.60 GiB; embedding and output 0.45 GiB.
- **One decode token** routes 8 x 28 = 224 experts, which is 1.47 GB of expert weights.

**Today's placement** is `--cpu-moe auto` (src/cuda.c, `gpu_plan_expert_banks`):
- Attention and dense tensors of every layer go on the device.
- The budget left after the output projection is spent on **whole expert banks, shallowest layer first**. The
  deeper layers' experts stay in host RAM and run on the CPU.
- On this card that is roughly half of the 28 banks (an estimate; the placement record at step 1 gives the
  number). A host layer's routed experts are read from DDR4 by the CPU every token.

**Where the bank would sit:**
- A device bank of `S` expert slots, in the VRAM today's plan gives to whole banks, laid out like one layer's
  expert tensors so the existing CUDA MoE matvec runs on a slot through a per-layer slot map. That is the same
  indirection as llama.cpp's.
- The routing is known on the device before the expert FFN, so the slot lookup is one small table per layer.

**One difference from llama.cpp, forced by this host:**
- **Upload loses to compute here.** PCIe 3.0 x16 gives about 12 GB/s in practice, so uploading one expert takes
  about 0.55 ms. Computing that expert on the CPU at a single decode token reads 6.58 MB from dual-channel DDR4
  at about 30 GB/s, which is about 0.22 ms. (Both rates are estimates and are measured at step 1.)
- **So misses run on the CPU,** as today. Admissions to the bank are uploaded off the step's critical path,
  rate-capped, and admit-on-second-touch so that one-off routings do not churn it.
- **What changes for an expert:** a cached expert runs on the device, and a missed one runs where it runs today.

## Hit-rate expectation

**There is no Mellum routing trace to read.**
- SS-001 (and SS-000, SS-002) ran dense models; none recorded MoE routing.
- llama.cpp's 72-89% belongs to Flash-Next's 512-expert, top-10 routing and does not transfer.

**What the gain needs.** With today's static banks covering a fraction `f` of experts at 100% and the rest at 0%,
and the host expert reads dominating decode, a policy beats today by about `(1 - f) / (1 - h)`, where `h` is its
share of routed experts served from the device.
- At f = 0.5, 1.3x needs h of at least 0.62 from the same bytes, before admission traffic.
- Routing has to be skewed or temporally local enough that a resident set chosen by use beats a resident set
  chosen by layer depth by that margin.
- That is unknown for Mellum. It is the question step 1 answers.

## Identity

Byte-exact against the plain path means **each expert's output is the bytes its executing backend's existing
kernel produces.** The bank changes where an expert's weights live, not the arithmetic. The gates:
- **All hits:** on a fixture that fits whole, every expert pre-admitted. The output is byte-identical to
  `--cpu-moe 0`, everything on the device.
- **All misses:** a zero-slot bank. The output is byte-identical to `--cpu-moe all`.
- **Mixed:** byte-identical to a static placement that puts each expert where the cache ran it. This is checked
  by replaying the run's routing with `RUNNER_MOE_REPLAY` and the cache's hit log.

Any miss on these is a kill. The CPU-against-CUDA difference between two placements stays in the gpu-identity
tolerance class it is in today, and the cache does not widen it.

## Steps and reads

1. **First boundary, no kernel code.** One validation download (12.9 GB; about 19 GB is free on C:, so the
   5 GiB floor holds).
   - Capture `RUNNER_MOE_TRACE` on the CPU over the SS-000 prompt set (code, copy, edit, long, prose, tool).
   - Measure today's `--cpu-moe auto` decode and its placement record on the 3070.
   - Measure host expert time and the PCIe upload rate.
   - Simulate three policies at the same bytes: (a) today's static layers; (b) a static hot set by frequency;
     (c) LRU with capped admit-on-second-touch.
   - **Kill** if the best projection is under 1.3x over (a).
   - A static hot set (b) that reaches 1.3x on its own would ship as a placement change, with no cache.
2. **Second boundary.** The bank behind the existing matvec, with the three identity gates. Then decode on the
   3070 against step 1's baseline, interleaved, on the same prompts. **Kill** at under 1.3x or at any identity
   miss.
