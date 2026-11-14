"""RUNNER_* switches retired by the November deprecation review (R18.8).

`RUNNER_*` variables are tuning and diagnostic switches that may change in any
release (docs/versioning.md); each removal is recorded in the changelog. A
retired switch is one the engine no longer reads: its name appears nowhere in
`src/` as a string, so setting it does nothing, and the capability inventory
(docs/capability-inventory.md) no longer lists it. The path a retired A/B arm
chose is not necessarily gone: the review keeps it where something else still
reaches it.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# name -> the plan item that retired it, and why the switch could go
RETIRED = {
    # R18.8.2: A/B arms of routes that are on by default; the path each turned
    # off is still reached without it, or is gone with it
    "RUNNER_ATTN_SPLIT": "R18.8.2: the split is bit-identical; attn_split_wanted declines it by shape",
    "RUNNER_METAL_MVFC": "R18.8.2: the column walk stays for a type the multi-column twin does not admit",
    "RUNNER_CUDA_ATTN_TILE512": "R18.8.2: k_attn stays for odd head widths and uneven GQA",
    "RUNNER_CUDA_ATTN_MLA": "R18.8.2: the absorbed MLA decode shipped in v1.14.0 to v1.18.0",
    # R18.8.11: its only arm sent 2 to 8 columns to the tile kernels retired below
    "RUNNER_CUDA_SMALL_BATCH": "R18.8.11: the tiles it chose for 2 to 8 columns are gone",
}

# R18.8.11: the CUDA tile kernels no default route reaches since the prefill
# stack (R6.41 to R6.58): on the scalar route a prompt runs the batch-1
# kernel's 16-wide twins or the batch-1 kernel per column, and the decode
# microbatch and the router take f_mvb only where it is the batch-1 kernel's
# twin (F32, F16, BF16, which keep theirs). The tiles summed in another order.
RETIRED_KERNELS = (
    [f"k_gemm_{t}" for t in ("q8_0", "q3_K", "q4_K", "q5_K", "q6_K")]
    + [f"k_mv_{t}_b" for t in ("q8_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K",
                               "q4_K", "q5_K", "q6_K", "iq4_nl", "iq4_xs", "mxfp4", "nvfp4",
                               "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m")])

SOURCES = [p for p in (ROOT / "src").iterdir()
           if p.suffix in {".c", ".h", ".m", ".cu", ".cuh", ".metal"}]


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_a_retired_switch_is_read_nowhere(name):
    readers = [p.name for p in SOURCES if f'"{name}"' in p.read_text(encoding="utf-8", errors="replace")]
    assert not readers, f"{name} is still read in {readers} ({RETIRED[name]})"


@pytest.mark.parametrize("kernel", RETIRED_KERNELS)
def test_a_retired_kernel_is_defined_loaded_and_embedded_nowhere(kernel):
    """Not in cuda.c's load tables, not in a kernel source, and not in the
    committed PTX headers, which are regenerated from those sources."""
    import re
    word = re.compile(rf"\b{kernel}\b")
    files = [ROOT / "src" / "cuda.c", ROOT / "src" / "kernels_ptx.h", ROOT / "src" / "kernels_ptx80.h"]
    files += [p for p in (ROOT / "src").iterdir() if p.name.startswith("kernels") and p.suffix in {".cu", ".cuh"}]
    found = [p.name for p in files if word.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert not found, f"{kernel} is still in {found}"


def test_the_float_tiles_stay():
    """F32, F16 and BF16 keep k_mv_*_b: it is their batch-1 kernel's twin."""
    src = (ROOT / "src" / "cuda.c").read_text(encoding="utf-8")
    for kernel in ("k_mv_f32_b", "k_mv_f16_b", "k_mv_bf16_b"):
        assert f'"{kernel}"' in src, kernel


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_the_capability_inventory_lists_no_retired_switch(name):
    # the page lists switches without their RUNNER_ prefix
    page = (ROOT / "docs" / "capability-inventory.md").read_text(encoding="utf-8")
    assert f"`{name.removeprefix('RUNNER_')}`" not in page, name
