"""A port in use is refused, and parallel tests do not share one (R10.22).

Windows' SO_REUSEADDR lets a second process bind a port another is listening on. Two
servers then shared it silently: the second printed "server listening" and got no
request at all, and when the first stopped, its clients' connections were reset. That
is what failed test_kv_ring under pytest -n, once in about 2,400 servers, when two
workers drew the same port. A server that refuses lets the harness take another.
"""
import http.server
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests" / "conformance")]
import _process  # noqa: E402
from _errors import TransportError  # noqa: E402
from harness import RunnerServer  # noqa: E402

EXE = pathlib.Path(os.environ.get("RUNNER_EXE") or (ROOT / ("runner.exe" if sys.platform == "win32" else "runner")))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    if not EXE.exists():
        pytest.skip("runner binary not built")
    p = tmp_path_factory.mktemp("port") / "m.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(p)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return p


def serve(model, port, log_file=None):
    return subprocess.Popen([str(EXE), "-m", str(model), "--serve", "--no-tray", "--port", str(port), "--parallel", "1",
                             "-c", "256", "--gpu", "off"], cwd=ROOT,
                            stdout=log_file if log_file is not None else subprocess.PIPE,
                            stderr=subprocess.STDOUT)


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
    log = _process.KeptProcessLog()
    p = serve(model, port, log_file=log.file)
    try:
        deadline = time.monotonic() + _process.scaled_timeout(10)
        while time.monotonic() < deadline:
            if p.poll() is not None:
                pytest.fail(
                    f"the restarted server exited ({p.returncode}); "
                    f"server log {log.keep()}:\n{log.tail()}"
                )
            try:
                assert complete(port) == 200
                return
            except OSError:
                time.sleep(0.05)
        pytest.fail(f"the restarted server never answered; server log {log.keep()}:\n{log.tail()}")
    except BaseException:
        log.keep()
        raise
    finally:
        p.kill()
        p.wait()
        log.close()


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


class _HealthStandIn(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


def _start_health_standin():
    port = _process.free_port()
    srv = http.server.HTTPServer(("127.0.0.1", port), _HealthStandIn)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return port, srv


def test_a_standin_on_health_retries_onto_a_free_port(model, monkeypatch):
    busy, standin = _start_health_standin()
    try:
        real = _process.free_port
        ports = iter([busy])
        monkeypatch.setattr(_process, "free_port", lambda: next(ports, None) or real())
        with RunnerServer(EXE, model, extra_args=["--gpu", "off"]) as srv:
            assert srv.port != busy
            assert complete(srv.port) == 200
    finally:
        standin.shutdown()
        standin.server_close()


def test_a_persistent_standin_on_health_fails_the_start_not_the_stop(model, monkeypatch):
    busy, standin = _start_health_standin()
    try:
        monkeypatch.setattr(_process, "free_port", lambda: busy)
        srv = RunnerServer(EXE, model, extra_args=["--gpu", "off"], start_timeout=15)
        with pytest.raises(TransportError) as exc:
            srv.start()
        msg = str(exc.value)
        assert ("cannot bind" in msg) or (f"port conflict" in msg and str(busy) in msg), msg
        assert "exited uncleanly on stop" not in msg, msg
    finally:
        standin.shutdown()
        standin.server_close()


def test_a_slow_child_behind_a_standin_fails_start_with_port_conflict_and_stop_notes_unhealthy(tmp_path, monkeypatch):
    busy, standin = _start_health_standin()
    try:
        monkeypatch.setattr(_process, "free_port", lambda: busy)
        slow = tmp_path / "slow.py"
        slow.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        srv = RunnerServer([sys.executable, str(slow)], "unused.gguf", start_timeout=0.4)
        with pytest.raises(TransportError) as exc:
            srv._start_once()
        msg = str(exc.value)
        assert f"port conflict" in msg and str(busy) in msg, msg

        class _DeadProc:
            returncode = -15
            def poll(self): return -15
            def send_signal(self, sig): pass
            def wait(self, timeout): return -15

        for win_flag in (False, True):
            monkeypatch.setattr(_process, "WINDOWS", win_flag)
            unhealthy = RunnerServer("unused", "unused")
            unhealthy._started = True
            unhealthy._healthy = False
            unhealthy.proc = _DeadProc()
            unhealthy.sample_rss = lambda: None
            with pytest.raises(TransportError) as stop_exc:
                unhealthy.stop(strict=True)
            assert "never became healthy" in str(stop_exc.value)
    finally:
        standin.shutdown()
        standin.server_close()


def test_xdist_worker_port_slices_are_disjoint(monkeypatch):
    import server_probe
    seen_proc, seen_probe = {}, {}
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "8")
    for k in range(8):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", f"gw{k}")
        p1 = _process.free_port()
        p2 = server_probe.low_free_port()
        assert (p1 - _process._PORT_LOW) % 32 == k
        assert (p2 - 20000) % 32 == k
        assert p1 not in seen_proc.values()
        seen_proc[k] = p1
        seen_probe[k] = p2


def test_real_server_banner_matches_process_listening_banner(model, tmp_path):
    log = tmp_path / "banner.log"
    with RunnerServer(EXE, model, extra_args=["--gpu", "off"], log_path=str(log)) as srv:
        assert complete(srv.port) == 200
    assert _process.listening_banner(srv.port).encode("ascii") in log.read_bytes()


def test_slow_log_banner_still_succeeds_on_final_retry(tmp_path, monkeypatch):
    """If a loaded host delays stderr banner visibility past CONFLICT_GRACE_S,
    the final retry keeps the full start_timeout instead of failing."""
    monkeypatch.setattr(_process, "CONFLICT_GRACE_S", 0.05)
    conf_dir = str(ROOT / "tests" / "conformance")
    laggy = tmp_path / "laggy.py"
    laggy.write_text(
        "import signal, socketserver, sys, threading, time\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        f"sys.path.insert(0, {conf_dir!r})\n"
        "from _process import listening_banner\n"
        "p = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "class FastHTTP(HTTPServer):\n"
        "  def server_bind(self):\n"
        "    socketserver.TCPServer.server_bind(self)\n"
        "    self.server_name, self.server_port = '127.0.0.1', self.server_address[1]\n"
        "class H(BaseHTTPRequestHandler):\n"
        "  def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b'{}')\n"
        "  def log_message(self, *a): pass\n"
        "srv = FastHTTP(('127.0.0.1', p), H)\n"
        "def delayed():\n"
        "  time.sleep(0.25)\n"
        "  print(listening_banner(p), file=sys.stderr, flush=True)\n"
        "threading.Thread(target=delayed, daemon=True).start()\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "if hasattr(signal, 'SIGBREAK'): signal.signal(signal.SIGBREAK, lambda *_: sys.exit(0))\n"
        "srv.serve_forever(poll_interval=0.05)\n",
        encoding="utf-8",
    )
    with RunnerServer([sys.executable, str(laggy)], "unused.gguf", start_timeout=5.0) as srv:
        assert srv._healthy is True
        assert srv.proc.poll() is None
