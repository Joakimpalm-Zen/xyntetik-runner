"""scripts/kld-compare.py: cross-model next-token-distribution comparison.

Gates: self-comparison must land at (approximately) zero KLD / 100% top-1
agreement — the same model, greedy, given the same prefix, always predicts
the same distribution; a real quant-vs-quant comparison must give sane,
nonzero divergence (quantization measurably perturbs the logits, even for a
tiny model, even at a fairly high bit width) without erroring out.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "kld-compare.py"
CORPUS_TEXT = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "the morning sun rises slowly above the distant mountains and a gentle "
    "breeze moves through the tall grass beside the old wooden fence."
)


# Ports for the servers these tests start. They were fixed numbers, so two runs of the suite on one host
# shared them and one run found the other's server on its port. They stay below every OS's ephemeral
# range (EPHEMERAL_FLOOR below says why) and start at a place that depends on the process.
PORT_FLOOR, PORT_CEIL = 20000, 32000


def _port_flags():
    a, b = _ports(2)
    return ["--port-a", str(a), "--port-b", str(b)]


def _start(pid):
    # 7919 shares no factor with the span, so any 12,000 consecutive process ids start at 12,000 places.
    return PORT_FLOOR + (pid * 7919) % (PORT_CEIL - PORT_FLOOR)


def _ports(n, start=None):
    import socket
    span = PORT_CEIL - PORT_FLOOR
    at = ((_start(os.getpid()) if start is None else start) - PORT_FLOOR) % span
    out = []
    for step in range(span):
        port = PORT_FLOOR + (at + step) % span
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        out.append(port)
        if len(out) == n:
            return out
    raise RuntimeError("no free port below the ephemeral ranges")


@pytest.fixture(scope="module")
def small_model():
    m = ROOT / "models" / "SmolLM2-135M-Instruct-Q8_0.gguf"
    if not m.exists():
        pytest.skip(f"{m} not present on this box")
    return m


@pytest.fixture(scope="module")
def requant_model(runner_bin, small_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("kld") / "smol-q4_0.gguf"
    proc = subprocess.run(
        [str(runner_bin), "-m", str(small_model), "--quantize", str(out), "--quant", "q4_0"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return out


@pytest.fixture()
def corpus(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(CORPUS_TEXT)
    return p


def _run(*args):
    proc = subprocess.run([sys.executable, str(SCRIPT), *args],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           timeout=180)
    return proc


def test_self_comparison_is_near_zero_kld_and_full_agreement(runner_bin, small_model, corpus):
    proc = _run("--model-a", str(small_model), "--model-b", str(small_model),
                "--runner", str(runner_bin), "--corpus", str(corpus),
                "--max-positions", "12", *_port_flags())
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["positions_scored"] >= 8, result
    assert result["mean_kld"] < 1e-4, (
        f"a model compared with itself must have ~0 KLD, got {result['mean_kld']}")
    assert result["top1_agreement_pct"] == 100.0, (
        f"a model compared with itself must agree on top-1 everywhere, got {result}")
    assert result["mean_top8_overlap"] == 1.0


def test_different_quants_give_sane_nonzero_divergence(runner_bin, small_model, requant_model, corpus):
    proc = _run("--model-a", str(small_model), "--model-b", str(requant_model),
                "--runner", str(runner_bin), "--corpus", str(corpus),
                "--max-positions", "12", *_port_flags())
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["positions_scored"] >= 8, result
    # "sane nonzero": some real divergence, but a Q8_0 vs Q4_0 requant of a
    # tiny model should still mostly agree — not so different it looks like
    # comparing two unrelated models.
    assert result["mean_kld"] > 1e-6, "Q8_0 vs Q4_0 of the same model should not be bit-identical"
    assert result["mean_kld"] < 5.0, f"divergence implausibly large for a same-model requant: {result}"
    assert 0.0 < result["top1_agreement_pct"] <= 100.0
    assert 0.0 <= result["mean_top8_overlap"] <= 1.0


def test_endpoint_mode_reuses_an_already_running_server(runner_bin, small_model, corpus):
    # kld-compare.py must also work against servers it didn't start itself
    # (Task 5a's pruned-vs-unpruned comparison will likely run this way).
    port = str(_ports(1)[0])
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(small_model), "--serve", "--no-tray",
         "--port", port, "--gpu", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        import urllib.request
        for _ in range(200):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            pytest.fail("test server did not come up")

        proc2 = _run("--endpoint-a", f"http://127.0.0.1:{port}",
                     "--model-name-a", small_model.name,
                     "--endpoint-b", f"http://127.0.0.1:{port}",
                     "--model-name-b", small_model.name,
                     "--corpus", str(corpus), "--max-positions", "6")
        assert proc2.returncode == 0, proc2.stderr
        result = json.loads(proc2.stdout)
        assert result["top1_agreement_pct"] == 100.0
    finally:
        proc.kill()
        proc.wait()


# kld-compare.py refuses a run in which nothing was scored ("error: no
# positions scored", exit 1). kld-compare-raw.py -- the v2 that publishes the
# ADOPTED margin-qualified criterion -- did not, so a dead endpoint, a wrong
# --model-name (the server 404s those) or a reference that reports no logprobs
# produced an all-null report and a clean exit code.

RAW_SCRIPT = ROOT / "scripts" / "kld-compare-raw.py"


def test_kld_compare_raw_refuses_a_run_that_scored_no_positions(corpus):
    proc = subprocess.run(
        [sys.executable, str(RAW_SCRIPT),
         "--endpoint-a", "http://127.0.0.1:1", "--model-name-a", "m",
         "--endpoint-b", "http://127.0.0.1:2", "--model-name-b", "m",
         "--corpus", str(corpus), "--max-positions", "4"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=180)

    result = json.loads(proc.stdout)
    assert result["positions_scored"] == 0
    assert result["positions_failed"] > 0
    # the evidence is still emitted, so --out records the failed run honestly
    assert proc.returncode == 1, proc.stdout


def test_endpoint_mode_refuses_an_unserved_model_name(runner_bin, small_model, corpus):
    # a label instead of the served id used to 404 on every position and
    # still write a report; now it is a one-line refusal naming the served ids
    port = str(_ports(1)[0])
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(small_model), "--serve", "--no-tray",
         "--port", port, "--gpu", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        import urllib.request
        for _ in range(200):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            pytest.fail("test server did not come up")
        proc2 = _run("--endpoint-a", f"http://127.0.0.1:{port}",
                     "--model-name-a", "not-the-served-id",
                     "--endpoint-b", f"http://127.0.0.1:{port}",
                     "--model-name-b", small_model.name,
                     "--corpus", str(corpus), "--max-positions", "2")
        assert proc2.returncode != 0
        assert "not served" in proc2.stderr and small_model.name in proc2.stderr
        assert "position 0 failed" not in proc2.stderr
    finally:
        proc.kill()
        proc.wait()


# The A/B controls (plan S7, restored from the archived scorer branch): one
# scorer run compares two settings of one switch on the same file, each side
# served with its own flags and environment, and the record says which side
# ran what.
def _raw_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("kld_raw_ab", RAW_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_raw_sides_carry_their_own_flags_and_environment():
    mod = _raw_module()
    cmd, env = mod.server_cmd_env("./runner", "m.gguf", 1234, threads=2, gpu="auto",
                                  extra=["--kv", "q8"], env_extra=["RUNNER_CUDA_TC=0"])
    assert cmd[cmd.index("--gpu") + 1] == "auto"
    assert cmd[-2:] == ["--kv", "q8"]
    assert cmd[cmd.index("-t") + 1] == "2"
    assert env["RUNNER_CUDA_TC"] == "0"
    assert env["RUNNER_GPU_FALLBACK"] == "fail"     # a device side may not finish on the CPU
    cmd0, env0 = mod.server_cmd_env("./runner", "m.gguf", 1234)
    assert cmd0[cmd0.index("--gpu") + 1] == "off"   # the CPU scalar route by default
    assert "RUNNER_CUDA_TC" not in env0 or env0["RUNNER_CUDA_TC"] == os.environ.get("RUNNER_CUDA_TC")
    with pytest.raises(SystemExit):
        mod.server_cmd_env("./runner", "m.gguf", 1234, env_extra=["RUNNER_CUDA_TC"])


def test_raw_ab_run_records_each_side(runner_bin, corpus):
    model = ROOT / "test.gguf"
    if not model.exists():
        pytest.skip("test.gguf fixture not built")
    out = ROOT / ".build" / "kld-raw-ab-test.json"
    out.parent.mkdir(exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(RAW_SCRIPT), "--runner", str(runner_bin),
         "--model-a", str(model), "--model-b", str(model),
         *_port_flags(),
         "--env-a", "RUNNER_SCORE_AB_SIDE=a", "--args-b", "-t 1", "--gpu-b", "off",
         "--corpus", str(corpus), "--max-positions", "3", "--out", str(out)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["positions_scored"] == 3
    assert result["sides"]["a"] == {"gpu": "off", "args": [], "env": ["RUNNER_SCORE_AB_SIDE=a"]}
    assert result["sides"]["b"] == {"gpu": "off", "args": ["-t", "1"], "env": []}


@pytest.mark.parametrize("script", ["kld-compare.py", "kld-compare-raw.py"])
def test_a_server_already_on_the_port_is_not_scored(runner_bin, corpus, tmp_path, script):
    """A comparison starts a server for each of its files and must score
    those. Its wait took any server that answered on the port for its own,
    so with another one already there (a second comparison, an earlier run
    still up) it scored that server's model under this file's name. It is
    refused now: the server that answers has to be the process started."""
    model = ROOT / "test.gguf"
    if not model.exists():
        pytest.skip("test.gguf fixture not built")
    taken, spare = (str(p) for p in _ports(2))
    other = subprocess.Popen(
        [str(runner_bin), "-m", str(model), "--serve", "--no-tray",
         "--port", taken, "--gpu", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        import urllib.request
        for _ in range(200):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{taken}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            pytest.fail("the other server did not come up")
        # another file with the same base name, as two folders of one model have
        mine = tmp_path / "test.gguf"
        q = subprocess.run([str(runner_bin), "-m", str(model), "--quantize", str(mine),
                            "--quant", "q8_0"], cwd=ROOT, capture_output=True, timeout=120)
        assert q.returncode == 0, q.stderr.decode(errors="replace")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--runner", str(runner_bin),
             "--model-a", str(mine), "--model-b", str(model),
             "--port-a", taken, "--port-b", spare,
             "--corpus", str(corpus), "--max-positions", "3"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        assert proc.returncode != 0, proc.stdout[-400:]
        assert taken in proc.stderr and "another" in proc.stderr, proc.stderr[-600:]
    finally:
        other.kill()
        other.wait()


# CI 2026-10-23: the raw A/B run died polling 127.0.0.1:28722 with
# BadStatusLine "GET /v1/models HTTP/1.1", its own request read back. A port
# inside the OS's ephemeral range can be handed to a connecting socket as its
# own source port while nothing listens there yet, and the socket connects to
# itself. Absolute anchors: Linux's default ip_local_port_range is
# 32768-60999, and macOS and Windows use the IANA dynamic range 49152-65535.
EPHEMERAL_FLOOR = 32768


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                  ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_polled_port_sits_below_the_ephemeral_ranges():
    import re
    found = []
    for name in ("kld-compare.py", "kld-compare-raw.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        found += [int(p) for p in re.findall(r'"--port-[ab]", type=int, default=(\d+)', text)]
    assert len(found) >= 4, found
    assert all(p < EPHEMERAL_FLOOR for p in found), sorted(p for p in found
                                                           if p >= EPHEMERAL_FLOOR)
    # the ports these tests pick themselves
    assert PORT_CEIL <= EPHEMERAL_FLOOR
    text = pathlib.Path(__file__).read_text(encoding="utf-8")
    assert not re.findall(r'"--port(?:-[ab])?", "\d+"', text), "a fixed port is back in this file"


def test_the_tests_ports_skip_one_in_use_and_start_by_process():
    import socket
    first = _ports(1, start=PORT_FLOOR + 4321)[0]
    with socket.socket() as held:
        held.bind(("127.0.0.1", first))
        held.listen(1)
        a, b = _ports(2, start=first)
    assert first not in (a, b) and a != b
    assert all(PORT_FLOOR <= p < PORT_CEIL for p in (first, a, b))
    # The start is a function of the process id alone, and nearby ids start apart. What the picker cannot
    # promise is that two live processes never land on one port: the bind above is what keeps a port that
    # is served out of the answer. (The first form of this test asked a child process for its port and
    # compared; it failed once in a full run, 24946 on both sides.)
    starts = [_start(pid) for pid in range(40000, 40600)]
    assert len(set(starts)) == len(starts)
    assert all(PORT_FLOOR <= p < PORT_CEIL for p in starts)
    assert _start(os.getpid()) == _start(os.getpid())


def test_a_self_connected_socket_is_not_taken_for_a_server():
    """Both comparison scripts poll through scripts/server_probe.py, the one
    copy of the check since 2026-10-27."""
    import socket
    mod = _load("server_probe")
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        try:
            s.connect(s.getsockname())
        except OSError:
            pytest.skip("this OS refuses a deliberate self-connect")
        assert mod.self_connected(s)
    finally:
        s.close()
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    c = socket.create_connection(server.getsockname())
    try:
        assert not mod.self_connected(c)
    finally:
        c.close()
        server.close()


def test_the_server_helper_picks_ports_below_every_ephemeral_range():
    """tests/conformance's RunnerServer (the conformance suite's and the tier
    smoke's server start) polls /health before the server binds. Binding :0
    handed out a port inside the ephemeral range, where that poll can be
    given the same port as its source and connect to itself, as above; a
    port below 32768 can never be a source port."""
    import socket
    sys.path.insert(0, str(ROOT / "tests" / "conformance"))
    from _process import free_port
    for _ in range(50):
        p = free_port()
        assert 1024 <= p < 32768, p
        with socket.socket() as s:
            s.bind(("127.0.0.1", p))


def _standin_http_server(port):
    """Minimal HTTP server answering /v1/models and /v1/capabilities with a foreign pid."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import socketserver
    import threading

    class FastHTTP(HTTPServer):
        allow_reuse_address = False

        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"data":[{"id":"other"}]}' if self.path == "/v1/models" else b'{"pid":999999}'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = FastHTTP(("127.0.0.1", port), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


@pytest.mark.parametrize("exit_early", [False, True])
def test_start_own_server_retries_on_port_conflict_and_stops_losing_child(exit_early):
    """When two evaluations on one worker draw the same low_free_port, the
    loser finds another pid answering there (either before or after its own
    bind exits). start_own_server stops the losing child and retries on a
    fresh port (R6.107)."""
    mod = _load("server_probe")
    taken = mod.low_free_port()
    srv = _standin_http_server(taken)
    spawned = []
    stub_code = (
        "import json, os, socketserver, sys, time; "
        "from http.server import BaseHTTPRequestHandler, HTTPServer; "
        "p = int(sys.argv[1]); "
        "early = sys.argv[2] == '1'; "
        "\nclass FastHTTP(HTTPServer):\n"
        "  allow_reuse_address = False\n"
        "  def server_bind(self):\n"
        "    socketserver.TCPServer.server_bind(self)\n"
        "    self.server_name, self.server_port = '127.0.0.1', self.server_address[1]\n"
        "class H(BaseHTTPRequestHandler):\n"
        "  def do_GET(self):\n"
        "    b = json.dumps({'data':[{'id':'m'}]} if self.path=='/v1/models' else {'pid':os.getpid()}).encode()\n"
        "    self.send_response(200); self.end_headers(); self.wfile.write(b)\n"
        "  def log_message(self, *a): pass\n"
        "try:\n"
        "  s = FastHTTP(('127.0.0.1', p), H)\n"
        "except OSError:\n"
        "  if early: sys.exit(1)\n"
        "  time.sleep(30); sys.exit(0)\n"
        "s.serve_forever()\n"
    )
    proc = None
    try:
        def spawn(p):
            c = subprocess.Popen([sys.executable, "-c", stub_code, str(p), "1" if exit_early else "0"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            spawned.append((c, p))
            return c

        proc, actual = mod.start_own_server(spawn, taken, "stub.gguf", attempts=3)
        assert actual != taken
        assert len(spawned) == 2
        assert spawned[0][0].poll() is not None, "first child must be stopped before retry"
        assert proc is spawned[1][0] and proc.poll() is None
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        for c, _ in spawned:
            if c.poll() is None:
                c.kill()
                c.wait()
        srv.shutdown()
        srv.server_close()


def test_start_own_server_does_not_retry_non_conflict_failure():
    mod = _load("server_probe")
    port = mod.low_free_port()
    spawned = []

    def spawn(p):
        c = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        spawned.append((c, p))
        return c

    with pytest.raises(RuntimeError, match="exited early"):
        mod.start_own_server(spawn, port, "broken.gguf", attempts=4)
    assert len(spawned) == 1
    assert spawned[0][0].poll() == 1


@pytest.mark.parametrize("script_name", ["kld-compare", "kld-compare-raw"])
def test_kld_scripts_start_server_retries_and_returns_actual_port(tmp_path, monkeypatch, script_name):
    mod = _load(script_name)
    probe = _load("server_probe")
    taken = probe.low_free_port()
    srv = _standin_http_server(taken)
    fake_runner = tmp_path / "fake_kld_stub.py"
    fake_runner.write_text(
        "import json, os, socketserver, sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "class FastHTTP(HTTPServer):\n"
        "  allow_reuse_address = False\n"
        "  def server_bind(self):\n"
        "    socketserver.TCPServer.server_bind(self)\n"
        "    self.server_name, self.server_port = '127.0.0.1', self.server_address[1]\n"
        "class H(BaseHTTPRequestHandler):\n"
        "  def do_GET(self):\n"
        "    b = json.dumps({'data':[{'id':'m'}]} if self.path=='/v1/models' else {'pid':os.getpid()}).encode()\n"
        "    self.send_response(200); self.end_headers(); self.wfile.write(b)\n"
        "  def log_message(self, *a): pass\n"
        "try:\n"
        "  s = FastHTTP(('127.0.0.1', port), H)\n"
        "except OSError:\n"
        "  sys.exit(1)\n"
        "s.serve_forever()\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = None
    try:
        # sys.executable -m fake_kld_stub keeps proc.pid == os.getpid() on POSIX and Windows.
        proc, actual = mod.start_server(sys.executable, "fake_kld_stub", taken)
        assert actual != taken
        assert proc.poll() is None
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        srv.shutdown()
        srv.server_close()
