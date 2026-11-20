"""R20: a dirty build's route identity is the image this process RUNS, not what its path names later.

A dirty process A that is paused at start, and whose executable's path is replaced by another binary before it takes
its identity, must still carry A's own identity, never the replacement's. The test: A, marked dirty
(RUNNER_TEST_BUILD_DIRTY=1), writes a session image whose header names its route identity, once run straight
(the reference) and once paused before the identity (RUNNER_TEST_IDENTITY_WAIT) while its file is replaced, then
resumed. Both images must name the same route. The swap renames A's file away and puts B at its path, which works
on every platform (Windows refuses to overwrite a running image, not to rename it). And two straight runs of one
dirty binary, two processes, must name one route, or no process resumes another's session.
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


def _straight(a, args, env, out):
    p = subprocess.run([str(a), *map(str, args), "--session-out", str(out)], capture_output=True, text=True,
                       timeout=120, env=env)
    assert p.returncode == 0, p.stderr[-400:]
    return p


def test_two_processes_of_one_dirty_binary_share_one_identity(runner_bin, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(tmp_path / "m.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_")}
    env["RUNNER_TEST_BUILD_DIRTY"] = "1"
    args = ["-m", tmp_path / "m.gguf", "-p", "hello", "-n", "2", "--temp", "0", "--gpu", "off", "-t", "2"]
    p1 = _straight(runner_bin, args, env, tmp_path / "one.img")
    _straight(runner_bin, args, env, tmp_path / "two.img")
    assert _route(tmp_path / "one.img") == _route(tmp_path / "two.img"), \
        "two processes of one dirty binary named two route identities: no session resumes across processes"
    assert "no identity of its running image" not in p1.stderr, p1.stderr[-400:]
    # the identity comes from the image in memory where the platform has one, never from a file a name can be
    # raced on: Windows reads the PE build id from the loaded module's debug directory, macOS the LC_UUID; Linux
    # hashes /proc/self/exe, which is the image's own inode whatever its path names
    kind = {"win32": "pe-buildid", "darwin": "macho-uuid"}.get(sys.platform, "sha256:/proc/self/exe")
    assert "image identity: " + kind in p1.stderr, p1.stderr[-400:]


def test_a_dirty_process_keeps_its_own_identity_when_its_file_is_replaced(runner_bin, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(tmp_path / "m.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    a = tmp_path / ("a-runner" + (".exe" if sys.platform == "win32" else ""))   # CreateProcess wants .exe
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
    os.replace(a, tmp_path / ("a-runner.moved" + a.suffix))       # A's file renamed away while A runs (allowed on Windows too)
    shutil.copy2(sys.executable, a)                   # another binary, B, at A's path; A's running image is unchanged
    go.write_text("go")
    out, err = p.communicate(timeout=120)
    assert p.returncode == 0, err[-400:]
    assert _route(tmp_path / "swapped.img") == _route(tmp_path / "ref.img"), \
        "A took the identity of the file its path named after the swap, not of its own running image"
