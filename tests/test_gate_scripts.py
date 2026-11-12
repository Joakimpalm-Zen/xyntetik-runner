"""Two shell drivers' failure modes (R6.68.5), each run as the script itself.

scripts/artifact-baseline.sh bounds every step with GNU timeout, which macOS
lacks: it must refuse once, with the reason, instead of failing every step.
scripts/cuda-gate.sh takes GATE_LOCK: an existing lock means skipped and is
never overwritten, and a lock it takes is gone when it finishes. Taking the
lock is now one O_EXCL open (noclobber) rather than a test and a create; that
race is not reproducible on demand, so these pin the behaviour around it.
"""
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not on PATH")


def _path_with(tmp_path, tools):
    """A PATH holding only the named tools, so an absent one is absent."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in tools:
        found = shutil.which(tool)
        assert found, tool
        (bin_dir / tool).symlink_to(found)
    return str(bin_dir)


@pytest.mark.skipif(os.name == "nt", reason="symlinked tool PATH; the refusal is the same code on Windows")
def test_artifact_baseline_refuses_once_without_gnu_timeout(tmp_path):
    env = {"PATH": _path_with(tmp_path, ["dirname", "mkdir", "date", "tee"]),
           "BL_ENV": str(tmp_path / "none.env"),
           "BL_ORIGINAL": str(tmp_path / "a.gguf"), "BL_DERIVED": str(tmp_path / "b.gguf")}
    out = tmp_path / "out"
    p = subprocess.run([BASH, str(ROOT / "scripts" / "artifact-baseline.sh"), str(out)],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "needs GNU timeout" in p.stderr and "gtimeout" in p.stderr, p.stderr
    assert p.stderr.count("\n") == 1, p.stderr
    assert not out.exists(), "nothing is started before the refusal"


def _gate(tmp_path, lock):
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    env = dict(os.environ, GATE_LOCK=str(lock), GATE_ENV=str(tmp_path / "none.env"),
               GATE_OUT=str(tmp_path / "out"))
    env.pop("GATE_MODEL", None)
    return subprocess.run([BASH, str(ROOT / "scripts" / "cuda-gate.sh"), str(build)],
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)


def test_cuda_gate_skips_on_an_existing_lock_and_never_overwrites_it(tmp_path):
    lock = tmp_path / "gate.lock"
    lock.write_text("held by another gate\n")
    p = _gate(tmp_path, lock)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"skipped: {lock} exists" in p.stdout, p.stdout
    assert lock.read_text() == "held by another gate\n", "the other gate's lock was overwritten"


def test_cuda_gate_takes_the_lock_and_gives_it_back(tmp_path):
    lock = tmp_path / "gate.lock"
    p = _gate(tmp_path, lock)
    assert "skipped:" not in p.stdout, p.stdout
    assert not lock.exists(), "a lock this gate took must be gone when it finishes"
