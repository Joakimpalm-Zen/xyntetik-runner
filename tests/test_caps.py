import json
import os
import pathlib
import shlex
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_build_arch_names_riscv64():
    """The cross target's compiler macros must produce its real receipt ISA."""
    cc = shlex.split(os.environ.get("CC", "cc"))
    p = subprocess.run(
        cc + ["-E", "-P", "-x", "c", "-", "-I", str(ROOT / "src"),
              "-U__aarch64__", "-U__arm64__", "-U_M_ARM64",
              "-U__x86_64__", "-U_M_X64",
              "-D__riscv=1", "-D__riscv_xlen=64"],
        input=b'#include "build_arch.h"\nRUNNER_BUILD_ARCH\n',
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert p.stdout.decode().strip() == '"riscv64"'


# What each backend is expected to have kernels for, stated here on purpose.
#
# `--caps` now generates both lists from one table in main.c filtered by the
# backend's own gpu_quant_ok(), so the advertised list and the loader cannot
# drift apart the way they did on Metal. But that also means they would agree
# about being wrong: a kernel added without extending gpu_type_ok, or an entry
# removed from the table, changes both at once and no self-consistency check
# can see it. This restates the fact independently, so a format gained or lost
# has to be admitted in two places by two different authors.
METAL_QUANTS = {
    "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "MXFP4",
}
# CUDA gained BF16 and Q2_K matvec kernels 2026-08-20, and NVFP4 on
# 2026-09-06 (the companion-scale loader plus k_mv_nvfp4/k_mv_nvfp4_b), which
# is the one format the two backends no longer agree on. That admission was
# late: the check only runs where a backend is actually present, and CI has no
# GPU, so the stale list sat green until the suite was run on the CUDA box.
CUDA_QUANTS = {
    "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "MXFP4",
    "NVFP4",
}


def test_gpu_quant_report_matches_backend_support():
    runner = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not runner.exists():
        return

    caps = json.loads(subprocess.run(
        [runner, "--caps"], cwd=ROOT, stdout=subprocess.PIPE, check=True,
    ).stdout)

    quants = set(caps["quants"])
    gpu_quants = set(caps["gpu_quants"])
    assert gpu_quants <= quants, sorted(gpu_quants - quants)

    backend = (caps.get("gpu") or {}).get("backend")
    expected = {"metal": METAL_QUANTS, "cuda": CUDA_QUANTS}.get(backend)
    if expected is not None:
        # equality, not a subset: an over-claim is the failure that shipped
        # once already, and a subset assertion is exactly what missed it
        assert gpu_quants == expected, (
            f"{backend} gpu_quants disagree; symmetric difference: "
            f"{sorted(gpu_quants ^ expected)}"
        )
