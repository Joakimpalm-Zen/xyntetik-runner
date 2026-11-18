"""R20: a dirty build's route identity is the image this process RUNS, not what its path names later.

A dirty process A that is paused at start, and whose executable's path is replaced by another binary before it takes
its identity, must still carry A's own identity, never the replacement's. The test: A, marked dirty
(RUNNER_TEST_BUILD_DIRTY=1), writes a session image whose header names its route identity, once run straight
(the reference) and once paused before the identity (RUNNER_TEST_IDENTITY_WAIT) while its file is replaced, then
resumed. Both images must name the same route. Skipped on Windows, where a dirty build's identity is a value of the
process alone (no persisted reuse) and two runs never share one.
"""
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _route(img):
    b = pathlib.Path(img).read_bytes()
    (hl,) = struct.unpack_from("<I", b, 16)
    return json.loads(b[20:20 + hl])["route"]


@pytest.mark.skipif(sys.platform == "win32", reason="a dirty build's identity is per process on Windows")
def test_a_dirty_process_keeps_its_own_identity_when_its_file_is_replaced(runner_bin, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(tmp_path / "m.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    a = tmp_path / "a-runner"
    shutil.copy2(runner_bin, a)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_")}
    env["RUNNER_TEST_BUILD_DIRTY"] = "1"
    args = ["-m", tmp_path / "m.gguf", "-p", "hello", "-n", "2", "--temp", "0", "--gpu", "off", "-t", "2"]
    ref = subprocess.run([str(a), *map(str, args), "--session-out", str(tmp_path / "ref.img")],
                         capture_output=True, text=True, timeout=120, env=env)
    assert ref.returncode == 0, ref.stderr[-400:]

    go = tmp_path / "go"
    p = subprocess.Popen([str(a), *map(str, args), "--session-out", str(tmp_path / "swapped.img")],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=dict(env, RUNNER_TEST_IDENTITY_WAIT=str(go)))
    time.sleep(1.0)                                   # A is waiting before it takes its identity
    other = tmp_path / "b-runner"
    shutil.copy2(sys.executable, other)               # another binary, B
    os.replace(other, a)                              # A's path now names B; A's running image is unchanged
    go.write_text("go")
    out, err = p.communicate(timeout=120)
    assert p.returncode == 0, err[-400:]
    assert _route(tmp_path / "swapped.img") == _route(tmp_path / "ref.img"), \
        "A took the identity of the file its path named after the swap, not of its own running image"
