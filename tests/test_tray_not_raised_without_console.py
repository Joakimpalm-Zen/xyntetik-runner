"""A server started with no console must not raise a tray.

The tray follows a serve or interactive run only when a person is at a
terminal (`cli_tray_follow` in main.c). On Windows that test was
`_isatty()`, which is true for every character device, NUL included, and
NUL is where services, process managers and Python's subprocess.DEVNULL
put stdin. So `runner --serve` launched that way raised a tray, with no
desktop session to show it in. The test is now a console handle
(`plat_is_terminal`: GetConsoleMode succeeds only on one).

Observable through the instance registry a raised tray registers itself
in: point APPDATA at an empty directory, start a server with stdin on NUL
and stdout and stderr piped, and no tray may appear. Windows only: the
macOS test is isatty(), which NUL (/dev/null) never satisfies.
"""

import contextlib
import json
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


def _trays(instances_dir):
    out = []
    if instances_dir.is_dir():
        for p in instances_dir.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if rec.get("mode") == "tray":
                out.append(rec)
    return out


def test_serve_with_stdin_on_nul_raises_no_tray(runner_bin, model, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, APPDATA=str(home), HOME=str(home))
    instances = home / "xyntetik/runner/instances"
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--port", str(port),
         "-c", "256", "--gpu", "off"],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    found = []
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
        # the tray is detached and registers itself asynchronously: give it
        # room to appear rather than racing it to a pass
        deadline = time.time() + 4
        while time.time() < deadline and not found:
            found = _trays(instances)
            time.sleep(0.2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        for rec in _trays(instances):        # never leave one behind
            pid = rec.get("pid")
            if isinstance(pid, int) and pid > 0:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True)
    assert not found, ("a server with stdin on NUL raised %d tray process(es): "
                       "NUL was taken for a terminal" % len(found))
