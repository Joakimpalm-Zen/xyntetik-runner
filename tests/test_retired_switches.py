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
}

SOURCES = [p for p in (ROOT / "src").iterdir()
           if p.suffix in {".c", ".h", ".m", ".cu", ".cuh", ".metal"}]


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_a_retired_switch_is_read_nowhere(name):
    readers = [p.name for p in SOURCES if f'"{name}"' in p.read_text(encoding="utf-8", errors="replace")]
    assert not readers, f"{name} is still read in {readers} ({RETIRED[name]})"


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_the_capability_inventory_lists_no_retired_switch(name):
    # the page lists switches without their RUNNER_ prefix
    page = (ROOT / "docs" / "capability-inventory.md").read_text(encoding="utf-8")
    assert f"`{name.removeprefix('RUNNER_')}`" not in page, name
