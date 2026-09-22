"""-hf with an over-long repository name is refused before any download
(the owner--repo expansion used to overflow a 512-byte stack buffer)."""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.mark.parametrize("spec,refusal", [
    ("a/" + "b" * 509, "too long"),            # fits the spec buffer, not the owner--repo expansion
    ("a/b/c/d/" + "e" * 400 + "/" * 100, "wants owner/repo"),   # refused by the format check first
])
def test_overlong_hf_spec_is_refused(runner_bin, spec, refusal, tmp_path):
    p = subprocess.run([str(runner_bin), "-hf", spec, "-p", "x", "-n", "1", "--gpu", "off"],
                       cwd=ROOT, env={"RUNNER_HF_CACHE": str(tmp_path), "PATH": "/usr/bin:/bin"},
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    err = p.stderr.decode(errors="replace")
    assert refusal in err, err   # the named refusal, not some later failure
    assert "AddressSanitizer" not in err
