"""RUNNER_METAL_TIMING=2: the decode microbatch's GPU time by stage (R3.24).

The profile commits and waits at each stage boundary of the Metal decode
microbatch (the --parallel slots that decode together) and prints one line a
step: the total and qkv, attn, o, gate_up, down, head. A single ready slot
takes the microbatch encoder too under the profile, so one slot and N slots
are measured at the same boundaries.

The anchor is the microbatch's own claim, that it is bit-identical to
sequential decode: the profiled server's replies must equal an unprofiled
server's byte for byte, for one slot (which the profile routes through the
microbatch, where the unprofiled server decodes it solo) and for two.
"""
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGES = ("qkv", "attn", "o", "gate_up", "down", "head")


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    caps = json.loads(subprocess.run([str(exe), "--caps"], capture_output=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("no Metal device")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("mbprof")
    f = d / "wide.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", "--wide", str(f)],
                   check=True, cwd=ROOT, capture_output=True)
    return f


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(runner_bin, model, tmp_path, timing, n_req):
    port = _port()
    log = tmp_path / f"server-{timing}-{n_req}.log"
    env = {"RUNNER_METAL_TIMING": timing} if timing else {}
    srv = subprocess.Popen([str(runner_bin), "-m", str(model), "--serve", "--port", str(port),
                            "--parallel", "2", "-c", "512", "--tray", "off"],
                           stdout=subprocess.DEVNULL, stderr=open(log, "w"),
                           env={**os.environ, **env})
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        out = [None] * n_req

        def ask(i):
            body = json.dumps({"prompt": f"the quick brown fox {i}", "max_tokens": 12,
                               "temperature": 0, "ignore_eos": True}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                         {"Content-Type": "application/json"})
            out[i] = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["text"]

        th = [threading.Thread(target=ask, args=(i,)) for i in range(n_req)]
        for t in th:
            t.start()
        for t in th:
            t.join()
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    return out, log.read_text(errors="replace")


@pytest.mark.parametrize("n_req", [1, 2])
def test_profile_prints_stages_and_keeps_the_bytes(runner_bin, model, tmp_path, n_req):
    plain, _ = _serve(runner_bin, model, tmp_path, None, n_req)
    prof, log = _serve(runner_bin, model, tmp_path, "2", n_req)
    assert prof == plain
    lines = re.findall(r"metal-microbatch n=(\d+) gpu=([\d.]+)ms (.*)", log)
    assert lines, log[-800:]
    widths = {int(n) for n, _, _ in lines}
    # a lone slot is profiled at the same boundaries as two
    assert widths == {1} if n_req == 1 else 2 in widths, widths
    for n, gpu, rest in lines:
        parts = dict(kv.split("=") for kv in rest.split())
        assert tuple(parts) == STAGES
        assert abs(sum(float(v) for v in parts.values()) - float(gpu)) < 1e-2
