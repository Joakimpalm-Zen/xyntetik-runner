"""A port in use is refused, and parallel tests do not share one (R10.22).

Windows' SO_REUSEADDR lets a second process bind a port another is listening on. Two
servers then shared it silently: the second printed "server listening" and got no
request at all, and when the first stopped, its clients' connections were reset. That
is what failed test_kv_ring under pytest -n, once in about 2,400 servers, when two
workers drew the same port. A server that refuses lets the harness take another.
"""
import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests" / "conformance")]
import _process  # noqa: E402
from harness import RunnerServer  # noqa: E402

EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    if not EXE.exists():
        pytest.skip("runner binary not built")
    p = tmp_path_factory.mktemp("port") / "m.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(p)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return p


def serve(model, port):
    return subprocess.Popen([str(EXE), "-m", str(model), "--serve", "--no-tray", "--port", str(port), "--parallel", "1",
                             "-c", "256", "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def complete(port):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                               data=json.dumps({"prompt": "hi", "max_tokens": 1}).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as f:
        return f.status


def test_a_second_server_on_a_port_in_use_is_refused_and_says_why(model):
    with RunnerServer(EXE, model, extra_args=["--gpu", "off"]) as first:
        p = serve(model, first.port)
        try:
            out = p.communicate(timeout=20)[0].decode(errors="replace")
        except subprocess.TimeoutExpired:
            p.kill()
            out = p.communicate()[0].decode(errors="replace")
            pytest.fail("the second server stayed up on a port in use:\n" + out[-400:])
        assert p.returncode != 0, out[-400:]
        assert f"cannot bind 127.0.0.1:{first.port}" in out, out[-400:]
        assert "another server" in out, out[-400:]
        assert "server listening" not in out, out[-400:]
        assert complete(first.port) == 200   # and the first was not disturbed


def test_a_server_can_restart_on_its_port_straight_after_serving(model):
    with RunnerServer(EXE, model, extra_args=["--gpu", "off"]) as first:
        port = first.port
        for _ in range(4):
            assert complete(port) == 200
    p = serve(model, port)
    try:
        for _ in range(200):
            assert p.poll() is None, p.communicate()[0].decode(errors="replace")[-400:]
            try:
                assert complete(port) == 200
                return
            except OSError:
                time.sleep(0.05)
        pytest.fail("the restarted server never answered")
    finally:
        p.kill()
        p.wait()


def test_the_harness_takes_another_port_when_it_loses_the_bind(model, monkeypatch):
    with socket.socket() as taken:
        if sys.platform == "win32":
            taken.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        busy = taken.getsockname()[1]
        real = _process.free_port
        ports = iter([busy])
        monkeypatch.setattr(_process, "free_port", lambda: next(ports, None) or real())
        with RunnerServer(EXE, model, extra_args=["--gpu", "off"]) as srv:
            assert srv.port != busy
            assert complete(srv.port) == 200


def test_a_startup_failure_that_is_not_the_bind_is_not_retried(tmp_path):
    bad = tmp_path / "nope.gguf"
    bad.write_bytes(b"not a model")
    if not EXE.exists():
        pytest.skip("runner binary not built")
    t0 = time.monotonic()
    with pytest.raises(Exception) as e:
        with RunnerServer(EXE, bad, extra_args=["--gpu", "off"]):
            pass
    assert "exited during startup" in str(e.value)
    assert time.monotonic() - t0 < 20
