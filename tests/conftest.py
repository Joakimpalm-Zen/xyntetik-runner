"""Fixtures shared by every test module in this directory.

`runner_bin` is the built engine at the repository root, or a skip when it is
not built: the one fixture nearly every module here needs, so it is defined
once instead of being pasted into each file. A module that needs a different
binary (a parametrised set, or one that also wants a fixture model) defines
its own and shadows this one.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe
