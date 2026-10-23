import json
import os
import pathlib
import shlex
import subprocess
import sys

import pytest


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
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S",
}
# CUDA gained BF16 and Q2_K matvec kernels 2026-08-20, NVFP4 on 2026-09-06
# (the companion-scale loader plus k_mv_nvfp4/k_mv_nvfp4_b) and the seven
# codebook i-quants on 2026-09-14 (k_mv_iq*/k_mv_iq*_b, so a mixed-type file
# such as the GSQ-RCO Qwen3.8 27B is admitted whole instead of falling to the
# CPU on its first IQ tensor). Metal gained the same seven on 2026-09-17
# (k_mv_iq*/k_mm_iq*), so the backends now differ on NVFP4 alone. The NVFP4 admission was late: the check only
# runs where a backend is actually present, and CI has no GPU, so the stale
# list sat green until the suite was run on the CUDA box.
CUDA_QUANTS = {
    "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "MXFP4",
    "NVFP4", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS",
    "IQ3_S",
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


def _caps(env_root=None):
    runner = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not runner.exists():
        pytest.skip("runner not built")
    env = dict(os.environ)
    env.pop("RUNNER_TEST_CGROUP_ROOT", None)
    if env_root is not None:
        env["RUNNER_TEST_CGROUP_ROOT"] = str(env_root)
    return json.loads(subprocess.run(
        [runner, "--caps"], cwd=ROOT, stdout=subprocess.PIPE, check=True, env=env,
    ).stdout)


def _write(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def test_caps_reports_host_and_limits_beside_the_effective_numbers():
    """The old keys keep their names and now carry what this process may use;
    `host` has the machine's numbers and `limits` says what narrowed them."""
    caps = _caps()
    host, limits = caps["host"], caps["limits"]
    assert set(host) == {"cpu_cores", "ram_bytes", "ram_available_bytes"}
    assert set(limits) == {"source", "cpu_quota", "memory_max_bytes"}
    assert limits["source"] in ("cgroup2", "cgroup1", "none")
    assert 1 <= caps["cpu_cores"] <= host["cpu_cores"]
    assert caps["ram_bytes"] <= host["ram_bytes"]
    if limits["cpu_quota"] is None:
        assert caps["cpu_cores"] == host["cpu_cores"]
    if limits["memory_max_bytes"] is None:
        assert caps["ram_bytes"] == host["ram_bytes"]
    if not sys.platform.startswith("linux"):
        # no cgroup outside Linux: the numbers are the host's, unchanged
        assert limits == {"source": "none", "cpu_quota": None, "memory_max_bytes": None}
        assert caps["cpu_cores"] == host["cpu_cores"]
        assert caps["ram_bytes"] == host["ram_bytes"]


def test_caps_sizes_from_a_cgroup2_limit(tmp_path):
    # cpu.max is "$MAX $PERIOD": 150000 / 100000 = 1.5 CPUs, rounded up to 2.
    # 1 GiB limit with 256 MiB charged and no memory.stat leaves 768 MiB.
    # Both are below any machine this runs on, so the limits must show.
    _write(tmp_path, {
        "cgroup.controllers": "cpu memory\n",
        "cpu.max": "150000 100000\n",
        "memory.max": "1073741824\n",
        "memory.current": "268435456\n",
    })
    plain, caps = _caps(), _caps(tmp_path)
    assert caps["limits"] == {"source": "cgroup2", "cpu_quota": 2,
                              "memory_max_bytes": 1073741824}
    # the host block is the machine, whatever the limits say
    assert caps["host"]["cpu_cores"] == plain["host"]["cpu_cores"]
    assert caps["host"]["ram_bytes"] == plain["host"]["ram_bytes"]
    assert caps["cpu_cores"] == min(caps["host"]["cpu_cores"], 2)
    assert caps["ram_bytes"] == min(caps["host"]["ram_bytes"], 1073741824)
    host_avail = caps["host"]["ram_available_bytes"]
    want = 805306368 if not host_avail else min(host_avail, 805306368)
    assert caps["ram_available_bytes"] == want


def test_caps_reads_a_cgroup1_layout(tmp_path):
    # v1 mounts each controller apart; -1 is no CPU quota, and the memory
    # limit is 512 MiB with nothing charged
    _write(tmp_path, {
        "cpu,cpuacct/cpu.cfs_quota_us": "-1\n",
        "cpu,cpuacct/cpu.cfs_period_us": "100000\n",
        "memory/memory.limit_in_bytes": "536870912\n",
        "memory/memory.usage_in_bytes": "0\n",
    })
    caps = _caps(tmp_path)
    assert caps["limits"] == {"source": "cgroup1", "cpu_quota": None,
                              "memory_max_bytes": 536870912}
    assert caps["cpu_cores"] == caps["host"]["cpu_cores"]
    assert caps["ram_bytes"] == min(caps["host"]["ram_bytes"], 536870912)
