"""RUNNER_* switches retired by deprecation reviews (R18.8, R3.84).

`RUNNER_*` variables are tuning and diagnostic switches that may change in any
release (docs/versioning.md); each removal is recorded in the changelog.
- R18.8 retired A/B arms appear nowhere in `src/` as strings.
- R3.84 retired opt-in routes (whose records showed no gain or a loss) are no
  longer read by any backend or engine path, appear nowhere in the capability
  inventory or active documentation, and setting one prints one notice on
  stderr with zero effect on output.
"""
import os
import pathlib
import re
import subprocess
import sys

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

# R3.84: opt-in routes retired because their records showed no gain or a loss.
# Setting any of these prints one notice on stderr and changes nothing.
RETIRED_WITH_NOTICE = {
    "RUNNER_METAL_MOE_EM": "R3.84 / R18.8.8: measured 3 to 4% slower than slot-major",
    "RUNNER_METAL_TENSOR": "R3.84 / R18.8.8: measured parity only on M5, short of 1.2x bar",
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

RETIRED_METAL_KERNELS = (
    [f"k_moe_mv_em_{t}" for t in ("f32", "f16", "q8_0", "q4_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "mxfp4")]
    + ["k_moe_mv_em8_q8_0", "k_moe_mv_em8_mxfp4"]
    + ["k_tensor_q4_K", "k_tensor_q8_0", "k_tensor_q4_0"])

SOURCES = [p for p in (ROOT / "src").iterdir()
           if p.suffix in {".c", ".h", ".m", ".cu", ".cuh", ".metal"}]


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_a_retired_switch_is_read_nowhere(name):
    readers = [p.name for p in SOURCES if f'"{name}"' in p.read_text(encoding="utf-8", errors="replace")]
    assert not readers, f"{name} is still read in {readers} ({RETIRED[name]})"


@pytest.mark.parametrize("name", sorted(RETIRED_WITH_NOTICE))
def test_a_retired_notice_switch_has_no_active_reader(name):
    """Only main.c's retired-switch notice table names the switch; no backend
    or engine code reads it, and main.c has no direct getenv(\"RUNNER_...\")."""
    other = [p.name for p in SOURCES
             if p.name != "main.c" and f'"{name}"' in p.read_text(encoding="utf-8", errors="replace")]
    assert not other, f"{name} still appears in {other} ({RETIRED_WITH_NOTICE[name]})"
    main_src = (ROOT / "src" / "main.c").read_text(encoding="utf-8", errors="replace")
    assert not re.search(rf'\bgetenv\(\s*"{name}"\s*\)', main_src), f"{name} is read directly in main.c"
    assert f'"{name}"' in main_src, f"{name} is missing from main.c's retired-switch notice table"


@pytest.mark.parametrize("kernel", RETIRED_KERNELS)
def test_a_retired_kernel_is_defined_loaded_and_embedded_nowhere(kernel):
    """Not in cuda.c's load tables, not in a kernel source, and not in the
    committed PTX headers, which are regenerated from those sources."""
    word = re.compile(rf"\b{kernel}\b")
    files = [ROOT / "src" / "cuda.c", ROOT / "src" / "kernels_ptx.h", ROOT / "src" / "kernels_ptx80.h"]
    files += [p for p in (ROOT / "src").iterdir() if p.name.startswith("kernels") and p.suffix in {".cu", ".cuh"}]
    found = [p.name for p in files if word.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert not found, f"{kernel} is still in {found}"


@pytest.mark.parametrize("kernel", RETIRED_METAL_KERNELS)
def test_a_retired_metal_kernel_is_defined_loaded_and_embedded_nowhere(kernel):
    word = re.compile(rf"\b{kernel}\b")
    files = [ROOT / "src" / "metal.m", ROOT / "src" / "kernels.metal", ROOT / "src" / "kernels_metal.h"]
    found = [p.name for p in files if p.exists() and word.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert not found, f"{kernel} is still in {found}"
    assert not (ROOT / "src" / "kernels_tensor.metal").exists()
    assert not (ROOT / "src" / "kernels_tensor_metal.h").exists()


def test_the_float_tiles_stay():
    """F32, F16 and BF16 keep k_mv_*_b: it is their batch-1 kernel's twin."""
    src = (ROOT / "src" / "cuda.c").read_text(encoding="utf-8")
    for kernel in ("k_mv_f32_b", "k_mv_f16_b", "k_mv_bf16_b"):
        assert f'"{kernel}"' in src, kernel


@pytest.mark.parametrize("name", sorted(set(RETIRED) | set(RETIRED_WITH_NOTICE)))
def test_the_capability_inventory_lists_no_retired_switch(name):
    # the page lists switches without their RUNNER_ prefix
    page = (ROOT / "docs" / "capability-inventory.md").read_text(encoding="utf-8")
    assert f"`{name.removeprefix('RUNNER_')}`" not in page, name


@pytest.mark.parametrize("name", sorted(RETIRED_WITH_NOTICE))
def test_retired_notice_switch_left_active_docs(name):
    for rel in ("MANUAL.md", "docs/speed-defaults.md", "docs/kernel-routes.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(rf"\b{name}\b", text), f"{name} still appears in {rel}"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    p = tmp_path_factory.mktemp("retired-sw") / "tiny.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(p)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return p


@pytest.mark.parametrize("name", sorted(RETIRED_WITH_NOTICE))
def test_retired_switch_prints_one_notice_and_has_no_effect(runner_bin, tiny_model, name):
    base_env = {k: v for k, v in os.environ.items() if k not in RETIRED_WITH_NOTICE}
    cmd = [str(runner_bin), "-m", str(tiny_model), "-p", "hi", "-n", "4",
           "--temp", "0", "-s", "1", "--no-tray", "-t", "2"]
    base = subprocess.run(cmd, cwd=ROOT, env=base_env, capture_output=True, timeout=60)
    assert base.returncode == 0, base.stderr.decode("utf-8", "replace")
    assert name.encode() not in base.stderr

    empty = subprocess.run(cmd, cwd=ROOT, env={**base_env, name: ""}, capture_output=True, timeout=60)
    assert empty.returncode == 0
    assert name.encode() not in empty.stderr

    set_run = subprocess.run(cmd, cwd=ROOT, env={**base_env, name: "1"}, capture_output=True, timeout=60)
    assert set_run.returncode == 0, set_run.stderr.decode("utf-8", "replace")
    err_lines = [ln for ln in set_run.stderr.decode("utf-8", "replace").splitlines() if name in ln]
    assert len(err_lines) == 1, f"expected 1 notice for {name}, got {err_lines!r}"
    assert "retired" in err_lines[0].lower() and "no effect" in err_lines[0].lower(), err_lines[0]
    assert set_run.stdout == base.stdout, f"{name}=1 changed output"
