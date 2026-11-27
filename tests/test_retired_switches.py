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
    "RUNNER_CPU_I8": (
        "R3.84: no format cleared the 0/64 bar with a gain worth taking "
        "(2026-08-13: Llama-3.2-3B Q4_K_M 1/64 FAIL at +7.8% decode, "
        "granite-4.1-8b Q4_0 2/64 FAIL at +1.0%, SmolLM2-135M Q8_0 0/64 PASS at +1.0%)"
    ),
    "RUNNER_PREFETCH": "R3.84: load-time WILLNEED sweep measured slower or flat on macOS, Linux and Windows",
}

# R3.84: symbols belonging exclusively to the retired RUNNER_CPU_I8 decode route
# and RUNNER_PREFETCH loader sweep. The shared activation quantizer
# (i8_quant_act, i8_act_size) and prefill tiles (i8_tile_dot, RUNNER_CPU_I8_PREFILL) stay.
RETIRED_CPU_LOADER_SYMBOLS = (
    "vec_dot_i8",
    "i8_dot_enabled",
    "i8_dot_ok",
    "i8_dot_force",
    "model_load_prefetch_wanted",
)
# R18.8.11: the CUDA tile kernels no default route reaches since the prefill
# stack (R6.41 to R6.58): on the scalar route a prompt runs the batch-1
# kernel's 16-wide twins or the batch-1 kernel per column, and the decode
# microbatch and the router take f_mvb only where it is the batch-1 kernel's
# twin (F32, F16, BF16, which keep theirs). The tiles summed in another order.
# R3.84: RUNNER_CUDA_Q8X is narrowed to IQ3_S only (+22% decode); Q4_K and Q8_0
# measured slower than the default decode (0.933 to 0.982x on three files,
# 2026-11-14; -1.8% to -6.6% in October), and IQ3_XXS and IQ4_XS had no
# single-type measurement file.
RETIRED_KERNELS = (
    [f"k_gemm_{t}" for t in ("q8_0", "q3_K", "q4_K", "q5_K", "q6_K")]
    + [f"k_mv_{t}_b" for t in ("q8_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K",
                               "q4_K", "q5_K", "q6_K", "iq4_nl", "iq4_xs", "mxfp4", "nvfp4",
                               "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m")]
    + [f"k_q8x_{t}_x{c}" for t in ("q4_K", "q8_0", "iq3_xxs", "iq4_xs") for c in range(1, 9)])

RETIRED_Q8X_HELPERS = ("iq3xxs_q8", "iq4xs_q8", "q4k_q8", "q80_q8")

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


@pytest.mark.parametrize("sym", RETIRED_CPU_LOADER_SYMBOLS)
def test_retired_cpu_and_loader_symbols_are_gone(sym):
    word = re.compile(rf"\b{sym}\b")
    found = [p.name for p in SOURCES if word.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert not found, f"{sym} is still in {found}"
    assert not (ROOT / "tests" / "test_load_prefetch.py").exists()
    assert not (ROOT / "tests" / "test_i8_tol.c").exists()


def test_shared_i8_prefill_quantizer_stays():
    """RUNNER_CPU_I8_PREFILL shares i8_quant_act / i8_act_size / i8_tile_dot; retiring
    RUNNER_CPU_I8 decode must keep them intact."""
    quants_h = (ROOT / "src" / "quants.h").read_text(encoding="utf-8")
    for sym in ("i8_quant_act", "i8_act_size", "i8_tile_dot", "i8_prefill_enabled"):
        assert re.search(rf"\b{sym}\b", quants_h), sym
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


@pytest.mark.parametrize("helper", RETIRED_Q8X_HELPERS)
def test_retired_q8x_helpers_are_gone(helper):
    src = (ROOT / "src" / "kernels.cu").read_text(encoding="utf-8", errors="replace")
    assert not re.search(rf"\b{helper}\b", src), f"{helper} is still in kernels.cu"


def test_q8x_iq3_s_stays_and_narrowed_notice_exists():
    """RUNNER_CUDA_Q8X stays for IQ3_S only (+22% decode), and cuda.c notices
    when set on a model containing retired Q8X types."""
    cuda_src = (ROOT / "src" / "cuda.c").read_text(encoding="utf-8")
    cu_src = (ROOT / "src" / "kernels.cu").read_text(encoding="utf-8")
    assert '"k_quant_q8x"' in cuda_src and "k_quant_q8x" in cu_src
    for c in range(1, 9):
        k = f"k_q8x_iq3_s_x{c}"
        assert f'"{k}"' in cuda_src and k in cu_src, k
    assert "RUNNER_CUDA_Q8X: Q4_K, Q8_0, IQ3_XXS and IQ4_XS kernels were retired in R3.84" in cuda_src


def test_q8x_retired_types_print_one_notice_on_cuda(runner_bin, tiny_model):
    """On a CUDA host, RUNNER_CUDA_Q8X=1 on a Q8_0 model prints one notice that
    Q8_0 was retired in R3.84 and produces byte-identical output."""
    info = subprocess.run([str(runner_bin), "--doctor"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    if '"backend":"cuda"' not in info.stdout.replace(" ", ""):
        pytest.skip("needs a CUDA device")
    base_env = {k: v for k, v in os.environ.items() if k != "RUNNER_CUDA_Q8X"}
    cmd = [str(runner_bin), "-m", str(tiny_model), "-p", "hi", "-n", "4",
           "--temp", "0", "-s", "1", "--no-tray", "-t", "2"]
    base = subprocess.run(cmd, cwd=ROOT, env=base_env, capture_output=True, timeout=60)
    assert base.returncode == 0, base.stderr.decode("utf-8", "replace")
    q8x = subprocess.run(cmd, cwd=ROOT, env={**base_env, "RUNNER_CUDA_Q8X": "1"}, capture_output=True, timeout=60)
    assert q8x.returncode == 0, q8x.stderr.decode("utf-8", "replace")
    err_lines = [ln for ln in q8x.stderr.decode("utf-8", "replace").splitlines()
                 if "RUNNER_CUDA_Q8X" in ln and "retired" in ln.lower()]
    assert len(err_lines) == 1, f"expected 1 retired-type notice for RUNNER_CUDA_Q8X, got {err_lines!r}"
    assert "Q8_0" in err_lines[0] and "IQ3_S" in err_lines[0], err_lines[0]
    assert q8x.stdout == base.stdout, "RUNNER_CUDA_Q8X=1 changed output on retired Q8_0 type"


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
    p = tmp_path_factory.mktemp("retired-sw") / "tiny-q8.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), "--quant", "q8_0", str(p)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return p


@pytest.mark.parametrize("name", sorted(RETIRED_WITH_NOTICE))
def test_retired_switch_prints_one_notice_and_has_no_effect(runner_bin, tiny_model, name):
    base_env = {k: v for k, v in os.environ.items() if k not in RETIRED_WITH_NOTICE}
    cmd = [str(runner_bin), "-m", str(tiny_model), "-p", "hi", "-n", "4",
           "--temp", "0", "-s", "1", "--gpu", "off", "--no-tray", "-t", "2"]
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
