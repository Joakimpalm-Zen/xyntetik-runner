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


@pytest.mark.skipif(sys.platform == "win32", reason="Windows terminates rather than signals: its exit code is not judged")
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
