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
test instead of hiding the bug. The macOS test is isatty(), which NUL
(/dev/null) never satisfies.

On Linux (R18.15) the rule adds a desktop session: a session bus, a display
and a tray host on that bus. A serve on a terminal with no desktop session
raises no tray; the same serve with a private bus, a display named and a fake
tray host (tests/test_tray_linux.c --hold-watcher) raises one, which is the
control that the rule can fire.
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

windows_only = pytest.mark.skipif(sys.platform != "win32",
                                  reason="the NUL-is-a-terminal mistake is Windows-only")
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="the desktop-session rule is Linux's")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
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
    ext = "bmp" if sys.platform == "win32" else "png"
    return sorted(p.name for p in d.glob(f"tray-*.{ext}")) if d.is_dir() else []


@windows_only
def test_the_icon_dump_seam_sees_a_tray(runner_bin, tmp_path):
    """Positive control: a tray launched in this environment leaves files."""
    icons = tmp_path / "icons"
    icons.mkdir()
    env = dict(os.environ, XYNTETIK_TRAY_ICON_DUMP=str(icons))
    p = subprocess.run([str(runner_bin), "--tray"], env=env, cwd=ROOT,
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert _dumped(icons), "the icon-dump seam wrote nothing; the test below would be blind"


@windows_only
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


# ---------------------------------------------------------------- Linux

def _linux_env(tmp_path, **extra):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("DBUS_SESSION_BUS_ADDRESS", "DISPLAY", "WAYLAND_DISPLAY")}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(tmp_path / "config"), XDG_RUNTIME_DIR=str(run))
    env.update(extra)
    return env


def _serve_on_a_terminal(runner_bin, model, env):
    """A serve whose stdin is a terminal (a pty); returns the glyphs a raised tray dumped."""
    import pty
    icons = pathlib.Path(env["XYNTETIK_TRAY_ICON_DUMP"])
    master, slave = pty.openpty()
    port = _free_port()
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--port", str(port), "-c", "256", "--gpu", "off"],
        cwd=ROOT, env=env, stdin=slave, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(slave)
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            assert proc.poll() is None, "the server exited"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        else:
            raise AssertionError("server never answered /health")
        deadline = time.time() + 5
        while time.time() < deadline and not _dumped(icons):
            time.sleep(0.2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        os.close(master)
    return _dumped(icons)


@linux_only
def test_linux_icon_dump_seam_sees_a_tray(runner_bin, tmp_path):
    """Positive control: a tray launched here leaves its glyphs."""
    icons = tmp_path / "icons"
    icons.mkdir()
    p = subprocess.run([str(runner_bin), "--tray"], env=_linux_env(tmp_path, XYNTETIK_TRAY_ICON_DUMP=str(icons)),
                       cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert _dumped(icons) == ["tray-idle.png", "tray-loaded.png", "tray-running.png"], _dumped(icons)


@linux_only
def test_linux_serve_on_a_terminal_without_a_desktop_raises_no_tray(runner_bin, model, tmp_path):
    icons = tmp_path / "icons"
    icons.mkdir()
    dumped = _serve_on_a_terminal(runner_bin, model, _linux_env(tmp_path, XYNTETIK_TRAY_ICON_DUMP=str(icons)))
    assert not dumped, f"a serve with no desktop session raised a tray (it dumped {dumped})"


@linux_only
def test_linux_serve_on_a_terminal_in_a_desktop_session_raises_one(runner_bin, model, tmp_path):
    """The control: the same serve with a bus, a display and a tray host raises a tray."""
    import shutil
    helper = ROOT / "test-tray-linux"
    if not shutil.which("dbus-daemon") or not helper.exists():
        pytest.skip("needs dbus-daemon and the test-tray-linux helper (make test-tray-linux)")
    bus_dir = tmp_path / "bus"
    bus_dir.mkdir()
    bus = subprocess.Popen(["dbus-daemon", "--session", "--nofork", f"--address=unix:dir={bus_dir}",
                            "--print-address=1"], stdout=subprocess.PIPE, text=True)
    holder = None
    try:
        addr = bus.stdout.readline().strip()
        assert addr, "the private bus printed no address"
        icons = tmp_path / "icons"
        icons.mkdir()
        env = _linux_env(tmp_path, DBUS_SESSION_BUS_ADDRESS=addr, DISPLAY=":424242",
                         XYNTETIK_TRAY_ICON_DUMP=str(icons))
        holder = subprocess.Popen([str(helper), "--hold-watcher"], env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "ready", "the fake tray host did not start"
        dumped = _serve_on_a_terminal(runner_bin, model, env)
        assert dumped, "a serve in a desktop session raised no tray: the rule cannot fire"
        # and without the display, the same bus and host raise none
        icons2 = tmp_path / "icons2"
        icons2.mkdir()
        env2 = dict(env, XYNTETIK_TRAY_ICON_DUMP=str(icons2))
        env2.pop("DISPLAY")
        assert not _serve_on_a_terminal(runner_bin, model, env2), "a serve with no display raised a tray"
    finally:
        if holder:
            holder.stdin.close()
            holder.wait(timeout=15)
        bus.terminate()
        bus.wait(timeout=15)
