"""A run refused by its own argument check must not leave a tray behind.

The tray is detached on purpose, so one raised by a process that then exits
outlives it and holds the executable open against the next relink -- a
failure this project has already fixed once from a different direction
(aa359e2). `main.c` says as much beside the call: "a two-second process
should not leave a menu-bar icon behind it". It was doing exactly that,
because the raise sat above `-s 0`, the yarn-factor conflicts and the
unknown-`--chat-template` check, all of which refuse a serve or interactive
run after it.

Observable through the instance registry, which a raised tray registers
itself in: point HOME (APPDATA on Windows) at an empty directory, run a
refused interactive invocation on a terminal, and no tray may appear.

Skips where it cannot mean anything: the tray exists only on macOS and
Windows, and the raise is gated on a terminal, which needs a pty.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAY_PLATFORMS = ("darwin", "win32")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture
def registry(tmp_path):
    """An empty instance registry, so nothing pre-existing hides a spawn."""
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), APPDATA=str(home))
    key = "runner" if sys.platform == "win32" else ".xyntetik"
    inner = "xyntetik/runner/instances" if sys.platform == "win32" \
        else ".xyntetik/runner/instances"
    return env, home / inner


def _trays(instances_dir):
    if not instances_dir.is_dir():
        return []
    out = []
    for p in instances_dir.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("mode") == "tray":
            out.append(rec)
    return out


def test_an_argument_refusal_raises_no_tray(runner_bin, registry, tmp_path):
    if sys.platform not in TRAY_PLATFORMS:
        pytest.skip("no tray backend on this platform (the call compiles out)")
    try:
        import pty
    except ImportError:
        pytest.skip("no pty module; the raise is gated on a terminal")

    env, instances = registry
    model = tmp_path / "m.gguf"
    model.write_bytes(b"not a model")

    # -s 0 is refused AFTER the mode guard: it is one of the checks the raise
    # used to sit above. -i makes it an interactive run, which is a session
    # the tray would otherwise follow.
    argv = [str(runner_bin), "-m", str(model), "-i", "-s", "0"]
    pid, fd = pty.fork()
    if pid == 0:                      # child: the pty IS its terminal
        os.execve(argv[0], argv, env)
        os._exit(127)
    output = b""
    try:
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
    finally:
        os.close(fd)
        _, status = os.waitpid(pid, 0)

    assert status != 0, "the run was supposed to be refused"
    assert b"not a usable seed" in output, output[-400:]

    # a spawn is detached and registers itself asynchronously; give it room
    # to appear rather than racing it to a pass
    deadline = time.time() + 3
    while time.time() < deadline and not _trays(instances):
        time.sleep(0.2)
    found = _trays(instances)
    for rec in found:                 # never leave one behind on a failure
        pid_ = rec.get("pid")
        if isinstance(pid_, int) and pid_ > 0:
            try:
                os.kill(pid_, 15)
            except OSError:
                pass
    assert not found, (
        "a refused run raised %d tray process(es); the raise is above an "
        "argument check again" % len(found))
