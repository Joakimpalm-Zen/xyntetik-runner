"""A failing test keeps its server's log, and shows the tail (R10.22's gap).

RunnerServer writes the server's output to a temp file of its own and deletes it on
stop. A test that failed with a clean server stop lost it, so a CI failure had no
server stderr to read. The with-block now keeps the file when a failure is unwinding
through it, and adds the tail to the failure. A clean block and a skip keep nothing.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests" / "conformance")]
from harness import RunnerServer  # noqa: E402
import _process  # noqa: E402
from _process import child_env  # noqa: E402
from _errors import TransportError  # noqa: E402

EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    if not EXE.exists():
        pytest.skip("runner binary not built")
    p = tmp_path_factory.mktemp("kept") / "m.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(p)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return p


def fail_inside(model, make_failure):
    """Runs a with-block that fails; returns the exception and the server."""
    srv = RunnerServer(EXE, model, extra_args=["--gpu", "off"])
    try:
        with srv:
            make_failure()
    except BaseException as e:   # pytest.fail is not an Exception
        return e, srv
    raise AssertionError("the block did not fail")


def raises_assertion():
    assert False, "the test's own failure"


@pytest.mark.parametrize("failure", [raises_assertion, lambda: pytest.fail("failed"), lambda: 1 / 0],
                         ids=["assert", "pytest.fail", "any exception"])
def test_a_failure_keeps_the_log_and_shows_its_tail(model, failure):
    e, srv = fail_inside(model, failure)
    note = "\n".join(getattr(e, "__notes__", []))
    assert "server listening" in note, note
    assert srv.kept_log and os.path.exists(srv.kept_log), srv.kept_log
    assert srv.kept_log in note
    assert "server listening" in pathlib.Path(srv.kept_log).read_text(errors="replace")
    os.unlink(srv.kept_log)


def test_a_failure_does_not_lose_the_tests_own_message(model):
    e, _ = fail_inside(model, raises_assertion)
    assert "the test's own failure" in str(e)
    assert isinstance(e, AssertionError)
    os.unlink(_.kept_log)


def test_a_clean_block_keeps_nothing(model):
    with RunnerServer(EXE, model, extra_args=["--gpu", "off"]) as srv:
        path = srv._own_log
        assert os.path.exists(path)
    assert not os.path.exists(path)
    assert srv.kept_log is None


def test_a_skip_is_not_a_failure_and_keeps_nothing(model):
    srv = RunnerServer(EXE, model, extra_args=["--gpu", "off"])
    paths = []

    def skip():
        paths.append(srv._own_log)
        pytest.skip("not here")

    with pytest.raises(pytest.skip.Exception):
        with srv:
            skip()
    assert not os.path.exists(paths[0])
    assert srv.kept_log is None


def test_a_log_the_caller_named_is_shown_and_stays_where_it_was_put(model, tmp_path):
    path = tmp_path / "named.log"
    srv = RunnerServer(EXE, model, extra_args=["--gpu", "off"], log_path=str(path))
    with pytest.raises(AssertionError) as e:
        with srv:
            raises_assertion()
    assert "server listening" in "\n".join(e.value.__notes__)
    assert path.exists() and srv.kept_log == str(path)


@pytest.mark.skipif(sys.platform == "win32", reason="a server found dead before the stop is judged on POSIX only")
def test_an_unclean_stop_keeps_the_whole_log_and_names_it(model):
    """The block passed, the server did not stop cleanly (here: killed from
    outside, as a sanitizer abort at exit would look to the harness). The
    log is the only record of why, so it stays and the error names it."""
    import signal
    srv = RunnerServer(EXE, model, extra_args=["--gpu", "off"])
    with pytest.raises(AssertionError) as e:
        with srv:
            own = srv._own_log
            os.kill(srv.proc.pid, signal.SIGKILL)
            srv.proc.wait(timeout=10)
    assert "exited uncleanly" in str(e.value)
    assert srv.kept_log == own and os.path.exists(own)
    assert own in str(e.value)
    assert "server listening" in pathlib.Path(own).read_text(errors="replace")
    os.unlink(own)


def test_a_stop_lets_the_server_shut_down(model, tmp_path):
    """The stop is a request, not a kill: the server runs its shutdown, so
    what it prints there is in the log and its exit code means something. On
    Windows the stop was TerminateProcess: a CUDA profile summary, printed
    when the device is freed, never appeared, and a test that read it failed
    on the one kind of host that reaches it (R6.68.123)."""
    log = tmp_path / "stop.log"
    with RunnerServer(EXE, model, extra_args=["--gpu", "off"], log_path=str(log)) as srv:
        pass
    if sys.platform == "win32" and RunnerServer._break_works is False:
        pytest.skip("no console to send CTRL_BREAK through: the server was terminated")
    assert not srv.killed and srv.exit_code == 0, (srv.killed, srv.exit_code)
    # written by the registry's unload, which only the shutdown path reaches
    assert "swap: unloading" in log.read_text(errors="replace")


HOST = {"HOME": "/home/u", "CUDA_VISIBLE_DEVICES": "1", "LD_LIBRARY_PATH": "/opt/cuda/lib64",
        "APPDATA": "C:\\Users\\u\\AppData\\Roaming", "SystemRoot": "C:\\Windows",
        "RUNNER_PREFETCH": "1", "EDITOR": "vi"}


def test_a_tests_env_keeps_the_home_and_the_device_choice():
    """A test's env replaces the inherited one so that no RUNNER_* variable
    of the developer's shell reaches the server. It also dropped HOME, and a
    CUDA driver keeps its compiled kernels under it: every such server
    compiled them again (34 s against 1.5 s to load a 2-layer fixture on a
    Linux CUDA host), and under load a test ran past its health wait."""
    env = child_env({"RUNNER_TEST_SCRIPTED_REPLY": "1"}, inherited=HOST, windows=False)
    assert env["HOME"] == "/home/u"
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"
    assert env["RUNNER_TEST_SCRIPTED_REPLY"] == "1"
    # still the test's own: nothing of the shell's RUNNER_* or anything else
    assert "RUNNER_PREFETCH" not in env and "EDITOR" not in env and "APPDATA" not in env


def test_a_tests_own_value_wins_and_windows_keeps_its_profile():
    env = child_env({"HOME": "/tmp/h", "RUNNER_TTL_POLL_S": "5"}, inherited=HOST, windows=True)
    assert env["HOME"] == "/tmp/h" and env["RUNNER_TTL_POLL_S"] == "5"
    assert env["APPDATA"] == HOST["APPDATA"] and env["SystemRoot"] == HOST["SystemRoot"]
    assert "RUNNER_PREFETCH" not in env


def test_no_env_is_the_inherited_one():
    env = child_env(None, inherited=HOST, windows=False)
    assert {k: env[k] for k in HOST} == HOST and env["RUNNER_TTL_POLL_S"] == "0.25"


class _Proc:
    """A server process as stop() sees it: hung (never exits when asked), or
    one that cannot be sent the request and exits 1 when terminated."""

    def __init__(self, hung=False, cannot_send=False):
        self.hung, self.cannot_send, self.returncode, self.sent = hung, cannot_send, None, []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        if self.cannot_send:
            raise OSError("no console")
        self.sent.append(sig)
        if not self.hung:
            self.returncode = 0

    def terminate(self):
        self.returncode = 1

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("server", timeout)
        return self.returncode

    def kill(self):
        self.returncode = -9


def _stopped(proc):
    srv = RunnerServer("unused", "unused")
    srv.proc, srv.sample_rss = proc, lambda: None
    srv.stop(strict=True)
    return srv


def test_the_windows_stop_rule(monkeypatch):
    """Windows, on any host: a server is sent CTRL_BREAK and judged; one that
    was sent it and stayed up is a failed shutdown and leaves the request on
    for the servers after it; only a send that fails turns it off, and a
    terminated server is not judged. (The first version took a first stop
    that timed out for a console that cannot deliver: one hung stand-in early
    in a run put all 517 later servers back on the hard kill, seen on the
    Windows CUDA host.)"""
    monkeypatch.setattr(_process, "WINDOWS", True)
    monkeypatch.setattr(_process.signal, "CTRL_BREAK_EVENT", 21, raising=False)
    monkeypatch.setattr(RunnerServer, "_break_works", None)
    with pytest.raises(TransportError):
        _stopped(_Proc(hung=True))
    assert RunnerServer._break_works is True
    ok = _stopped(_Proc())
    assert ok.proc.sent == [21] and ok.exit_code == 0 and not ok.killed
    off = _stopped(_Proc(cannot_send=True))          # exit 1, and no error
    assert RunnerServer._break_works is False and off.exit_code == 1 and not off.killed
    later = _stopped(_Proc())                        # not tried again
    assert later.proc.sent == [] and later.exit_code == 1
