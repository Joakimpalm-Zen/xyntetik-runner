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


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if os.name == "nt" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


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
                "--max-positions", "12", "--port-a", "28701", "--port-b", "28702")
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
                "--max-positions", "12", "--port-a", "28703", "--port-b", "28704")
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
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(small_model), "--serve", "--no-tray",
         "--port", "28705", "--gpu", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        import urllib.request
        for _ in range(200):
            try:
                urllib.request.urlopen("http://127.0.0.1:28705/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            pytest.fail("test server did not come up")

        proc2 = _run("--endpoint-a", "http://127.0.0.1:28705",
                     "--model-name-a", small_model.name,
                     "--endpoint-b", "http://127.0.0.1:28705",
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
    proc = subprocess.Popen(
        [str(runner_bin), "-m", str(small_model), "--serve", "--no-tray",
         "--port", "28706", "--gpu", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        import urllib.request
        for _ in range(200):
            try:
                urllib.request.urlopen("http://127.0.0.1:28706/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            pytest.fail("test server did not come up")
        proc2 = _run("--endpoint-a", "http://127.0.0.1:28706",
                     "--model-name-a", "not-the-served-id",
                     "--endpoint-b", "http://127.0.0.1:28706",
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
         "--port-a", "28721", "--port-b", "28722",
         "--env-a", "RUNNER_SCORE_AB_SIDE=a", "--args-b", "-t 1", "--gpu-b", "off",
         "--corpus", str(corpus), "--max-positions", "3", "--out", str(out)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["positions_scored"] == 3
    assert result["sides"]["a"] == {"gpu": "off", "args": [], "env": ["RUNNER_SCORE_AB_SIDE=a"]}
    assert result["sides"]["b"] == {"gpu": "off", "args": ["-t", "1"], "env": []}


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
    text = pathlib.Path(__file__).read_text(encoding="utf-8")
    found += [int(p) for p in re.findall(r'"--port(?:-[ab])?", "(\d+)"', text)]
    found += [int(p) for p in re.findall(r"127\.0\.0\.1:(\d+)", text)]
    assert len(found) >= 10, found
    assert all(p < EPHEMERAL_FLOOR for p in found), sorted(p for p in found
                                                           if p >= EPHEMERAL_FLOOR)


@pytest.mark.parametrize("name", ["kld-compare", "kld-compare-raw"])
def test_a_self_connected_socket_is_not_taken_for_a_server(name):
    import socket
    mod = _load(name)
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
