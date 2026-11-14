"""A test seam that stands in for a measurement says so, and only lowers a limit (fable, 2026-11-14).

RUNNER_TEST_VRAM_FREE_MB stands in for the device's free memory in the VRAM claim, and
RUNNER_TEST_CGROUP_ROOT reads a fixture tree as this process's own cgroup. Both were silent, so an
inherited value changed what a load was admitted to without a word, and the VRAM seam could raise the
free memory past what a CUDA device measures. Each now names itself and what it replaces on stderr
when it is consulted. The VRAM seam keeps a device's measured free memory when it is lower; where no
device reports free memory (Metal, the CPU) it stands in for one, which is what it is for. A cgroup can
only lower the host's limits, and the line says that.
"""
import json
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if os.name == "nt" else "runner")
MODEL = ROOT / "test.gguf"


def _runner_or_skip():
    if not RUNNER.exists():
        pytest.skip("runner not built")
    return RUNNER


def _backend():
    out = subprocess.run([str(_runner_or_skip()), "--caps"], cwd=ROOT, capture_output=True, text=True,
                         timeout=60).stdout
    return (json.loads(out).get("gpu") or {}).get("backend")


def _load(tmp_path, free_mb):
    env = dict(os.environ, RUNNER_TEST_VRAM_FREE_MB=str(free_mb), RUNNER_VRAM_REGISTRY_DIR=str(tmp_path))
    p = subprocess.run([str(_runner_or_skip()), "-m", str(MODEL), "-p", "x", "-n", "1", "--temp", "0",
                        "--no-tray"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-800:]
    return p.stderr


def test_the_vram_seam_says_what_it_stands_in_for(tmp_path):
    backend = _backend()
    if backend not in ("metal", "cuda"):
        pytest.skip("a GPU backend's load path (CPU-only runs make no VRAM claim)")
    err = _load(tmp_path, 4096)
    assert "test seam: RUNNER_TEST_VRAM_FREE_MB=4096" in err, err[-800:]
    if backend == "metal":
        assert "stands in for a device that reports its free memory" in err, err[-800:]
    else:
        assert "in place of the measured free VRAM" in err or "is above the measured free VRAM" in err, err[-800:]


def test_the_vram_seam_never_raises_what_a_device_measures(tmp_path):
    if _backend() != "cuda":
        pytest.skip("the CUDA backend's measured free VRAM")
    err = _load(tmp_path, 1048576)
    assert "test seam: RUNNER_TEST_VRAM_FREE_MB=1048576 is above the measured free VRAM" in err, err[-800:]


def test_the_cgroup_seam_says_what_it_stands_in_for(tmp_path):
    for name, text in {"cgroup.controllers": "cpu memory\n", "cpu.max": "150000 100000\n",
                       "memory.max": "1073741824\n", "memory.current": "268435456\n"}.items():
        (tmp_path / name).write_text(text)
    env = dict(os.environ, RUNNER_TEST_CGROUP_ROOT=str(tmp_path))
    p = subprocess.run([str(_runner_or_skip()), "--caps"], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 0, p.stderr[-800:]
    assert f"test seam: RUNNER_TEST_CGROUP_ROOT={tmp_path} in place of this process's own cgroup" in p.stderr, \
        p.stderr[-800:]
    assert json.loads(p.stdout)["limits"]["memory_max_bytes"] == 1073741824   # the seam is in force
