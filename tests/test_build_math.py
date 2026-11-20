"""R6.85: a binary says which math it has. A portable build (a Windows default, T3=1, or PORTABLE_MATH=1) answers
"math": "portable" in --caps and in the server's build fields (/health, /v1/capabilities); one with the C library's
functions (Linux and macOS by default, or Windows built with PORTABLE_MATH=0) answers "libc". --version stays
"runner X.Y.Z": the release checks that line exactly. The expected value is the one this platform's default build has,
or "portable" for a T3 binary (its --version says t3).
"""
import json
import os
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


def _expected(exe):
    v = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=20).stdout
    return "portable" if "(t3)" in v or os.name == "nt" or os.environ.get("RUNNER_TEST_PORTABLE_MATH") == "1" \
        else "libc"


def test_caps_names_the_math(runner_bin):
    p = subprocess.run([str(runner_bin), "--caps"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-400:]
    assert json.loads(p.stdout).get("math") == _expected(runner_bin), p.stdout[:300]


def test_version_line_is_unchanged_by_the_math(runner_bin):
    v = subprocess.run([str(runner_bin), "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    assert v.startswith("runner ") and "math" not in v, v
