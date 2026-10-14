"""A server started with no console must not raise a tray.

The tray follows a serve or interactive run only when a person is at a
terminal (`cli_tray_follow` in main.c). On Windows that test was
`_isatty()`, which is true for every character device, NUL included, and
NUL is where services, process managers and Python's subprocess.DEVNULL
put stdin. So `runner --serve` launched that way raised a tray, with no
desktop session to show it in. The test is now a console handle
(`plat_is_terminal`: GetConsoleMode succeeds only on one).

Observed through the tray's icon-dump seam rather than its registry
record: on a machine with no desktop session (a service, a scheduled task,
CI) a raised tray registers and then fails to start its message loop
within milliseconds, so a registry poll misses it and passes vacuously.
With XYNTETIK_TRAY_ICON_DUMP set, a tray writes its glyphs to that
directory and exits before it needs a desktop, and a tray the server
raises inherits the server's environment. A positive control runs
`runner --tray` itself first, so a seam that stopped working fails the
test instead of hiding the bug. Windows only: the macOS test is isatty(),
which NUL (/dev/null) never satisfies.
"""

import contextlib
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="the NUL-is-a-terminal mistake is Windows-only")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / "runner.exe"
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("m") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT)
    return m


def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dumped(d):
    return sorted(p.name for p in d.glob("tray-*.bmp")) if d.is_dir() else []


def test_the_icon_dump_seam_sees_a_tray(runner_bin, tmp_path):
    """Positive control: a tray launched in this environment leaves files."""
    icons = tmp_path / "icons"
    icons.mkdir()
    env = dict(os.environ, XYNTETIK_TRAY_ICON_DUMP=str(icons))
    p = subprocess.run([str(runner_bin), "--tray"], env=env, cwd=ROOT,
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert _dumped(icons), "the icon-dump seam wrote nothing; the test below would be blind"


def test_serve_with_stdin_on_nul_raises_no_tray(runner_bin, model, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    icons = tmp_path / "icons"
    icons.mkdir()
    env = dict(os.environ, APPDATA=str(home), HOME=str(home),
               XYNTETIK_TRAY_ICON_DUMP=str(icons))
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--port", str(port),
         "-c", "256", "--gpu", "off"],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            assert proc.poll() is None, proc.stderr.read().decode(errors="replace")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        else:
            raise AssertionError("server never answered /health")
        # the raise happens at startup; a raised tray dumps and exits at once
        deadline = time.time() + 4
        while time.time() < deadline and not _dumped(icons):
            time.sleep(0.2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    assert not _dumped(icons), ("a server with stdin on NUL raised a tray (it "
                                "dumped %s): NUL was taken for a terminal"
                                % ", ".join(_dumped(icons)))
