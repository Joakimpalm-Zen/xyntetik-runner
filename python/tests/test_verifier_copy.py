"""The verification copy keeps mode bits: a 0755 script the repository's
checks run directly must still be executable after _copy_tree (it was copied
with copyfile, which drops permissions; found 2026-09-22)."""
import os
import stat
import sys
from pathlib import Path

import pytest

from xyntetik_runner.shadow import verifier


@pytest.mark.skipif(os.name != "posix", reason="mode bits are a posix property")
def test_copy_tree_preserves_executable_bit(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    script = src / "check.sh"
    script.write_text("#!/bin/sh\necho ok\n")
    script.chmod(0o755)
    (src / "plain.txt").write_text("x")
    dst = tmp_path / "dst"
    verifier._copy_tree(src, dst)
    copied = dst / "check.sh"
    assert copied.exists()
    assert stat.S_IMODE(copied.stat().st_mode) & 0o111, oct(copied.stat().st_mode)
    assert (dst / "plain.txt").read_text() == "x"
