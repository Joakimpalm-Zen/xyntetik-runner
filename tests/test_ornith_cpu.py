"""Public CPU tracer for the Ornith-1.0/Qwen3.5 hybrid architecture."""
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
sys.path.insert(0, str(ROOT / "tests" / "conformance"))

from harness import KeptProcessLog, scaled_timeout  # noqa: E402

# R6.127: the fake servers' interpreter. In a Windows virtual environment sys.executable is a launcher that runs the
# real interpreter as its own child, so kill() and wait() reach the launcher while the interpreter still holds the
# server log's handle, and unlinking the log fails (WinError 32). The base interpreter is the process itself; the fake
# servers import only sys and time, so they need nothing from the environment. Outside a venv this is sys.executable.
_SERVER_PY = getattr(sys, "_base_executable", sys.executable)


def _run_runner(cmd, timeout_s=60, env=None):
    """Run a child command under scaled_timeout and keep its output on timeout."""
    log = KeptProcessLog(prefix="runner-ornith-")
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=log.file,
            timeout=scaled_timeout(timeout_s), env=env,
        )
        log.file.flush()
        stderr = pathlib.Path(log.path).read_bytes()
        return proc.returncode, proc.stdout, stderr
    except subprocess.TimeoutExpired as exc:
        log.keep()
        out_tail = (exc.stdout or b"")[-2000:].decode("utf-8", "replace")
        pytest.fail(
            f"{cmd} timed out after {exc.timeout:.1f}s; "
            f"stderr log {log.path}:\n{log.tail()}\n"
            f"stdout tail:\n{out_tail}"
        )
    finally:
        log.close()


def _gpu_caps():
    rc, out, err = _run_runner([RUNNER, "--caps"], timeout_s=30)
    assert rc == 0, err.decode(errors="replace")
    return json.loads(out).get("gpu") or {}


def _make_fixture(script, out, env=None):
    rc, _, err = _run_runner([sys.executable, ROOT / "scripts" / script, out], timeout_s=60, env=env)
    assert rc == 0, err.decode(errors="replace")


def _wait_for_health(srv, port, log, timeout_s=30):
    """Poll /health until timeout_s (scaled); fail with kept server log if overdue or dead."""
    deadline = time.monotonic() + scaled_timeout(timeout_s)
    while time.monotonic() < deadline:
        if srv.poll() is not None:
            pytest.fail(
                f"server exited ({srv.returncode}) before /health; "
                f"server log {log.keep()}:\n{log.tail()}"
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return
        except OSError:
            time.sleep(0.1)
    pytest.fail(f"server never answered /health; server log {log.keep()}:\n{log.tail()}")


def test_qwen35_hybrid_loads_and_decodes(tmp_path):
    model = tmp_path / "ornith.gguf"
    _make_fixture("make-test-ornith.py", model)
    rc, out, err = _run_runner(
        [RUNNER, "-m", model, "-p", "hi", "-n", "2",
         "--temp", "0", "--gpu", "off", "-c", "32"],
    )
    assert rc == 0, err.decode(errors="replace")
    assert b"hybrid SSM/attention architecture" not in err
    # This greedy suffix is the llama.cpp reference result for the exact same
    # generated weights and prompt-token sequence. It protects the recurrent
    # state orientation, Q scaling, group broadcast, convolution ordering and
    # the full-attention post-norm placement as one end-to-end contract.
    assert out.splitlines() == [b"hiii"]


def test_qwen35_legacy_ornith_dt_name_loads(tmp_path):
    model = tmp_path / "ornith-legacy.gguf"
    _make_fixture("make-test-ornith.py", model, env={**os.environ, "ORNITH_LEGACY_DT": "1"})
    rc, _, err = _run_runner(
        [RUNNER, "-m", model, "-p", "hi", "-n", "1",
         "--temp", "0", "--gpu", "off", "-c", "32"],
    )
    assert rc == 0, err.decode(errors="replace")


def test_qwen35_hybrid_gpu_auto_matches_cpu(tmp_path):
    gpu_caps = _gpu_caps()
    if not gpu_caps:
        pytest.skip("no GPU backend: this test asserts a device/host match")
    backend = gpu_caps.get("backend")
    model = tmp_path / "ornith-gpu.gguf"
    _make_fixture("make-test-ornith.py", model)
    base = [RUNNER, "-m", model, "-p", "hi", "-n", "2",
            "--temp", "0", "-c", "32"]
    cpu_rc, cpu_out, cpu_err = _run_runner([*base, "--gpu", "off"])
    gpu_rc, gpu_out, gpu_err = _run_runner([*base, "--gpu", "auto"])
    assert cpu_rc == 0, cpu_err.decode(errors="replace")
    assert gpu_rc == 0, gpu_err.decode(errors="replace")
    assert gpu_out == cpu_out
    if backend == "cuda":
        assert b"CUDA backend" in gpu_err
        assert b"no kernels for the recurrent" not in gpu_err
    elif backend == "metal":
        # the dense qwen35 hybrid runs on the device (R4.10), its greedy
        # output the CPU's (asserted above)
        assert b"Metal backend" in gpu_err
        assert b"not on the metal backend" not in gpu_err

    if backend == "cuda":
        fb_rc, fb_out, fb_err = _run_runner(
            [*base, "--gpu", "auto"],
            env={**os.environ, "RUNNER_CUDA_INJECT_FAILURE": "1"},
        )
        assert fb_rc == 0, fb_err.decode(errors="replace")
        assert fb_out == cpu_out
        assert b"injected CUDA runtime failure" in fb_err


def test_qwen35_cuda_failure_after_recurrent_state_accumulates(tmp_path):
    """Fail the device *after* it has carried recurrent state, not on tile one.

    An unconditional hook can only ever fire on the first device forward, which
    for any real generation is the prefill tile — before a single SSM state
    update has been carried across a token boundary. That is the cheap half of
    the contract. The half that can actually corrupt output is a failure at
    decode, where `q35_hist`/`q35_state` hold accumulated history and the
    caller has to roll back to the pre-forward snapshot before recomputing on
    the host.

    `RUNNER_CUDA_INJECT_FAILURE=N` fails the Nth device forward, so N=2 lands
    on the first decode step — past prefill, with state live.
    """
    if _gpu_caps().get("backend") != "cuda":
        pytest.skip("CUDA device not available")

    model = tmp_path / "ornith-late-fail.gguf"
    _make_fixture("make-test-ornith.py", model)
    base = [RUNNER, "-m", model, "-p", "hi", "-n", "4",
            "--temp", "0", "-c", "32"]
    cpu_rc, cpu_out, cpu_err = _run_runner([*base, "--gpu", "off"])
    late_rc, late_out, late_err = _run_runner(
        [*base, "--gpu", "auto"],
        env={**os.environ, "RUNNER_CUDA_INJECT_FAILURE": "2"},
    )
    assert cpu_rc == 0, cpu_err.decode(errors="replace")
    assert late_rc == 0, late_err.decode(errors="replace")
    # Naming the ordinal is what proves the failure landed past prefill; a hook
    # that always fired on tile one would report forward 1 here.
    assert b"injected CUDA runtime failure at device forward 2" in late_err, \
        late_err.decode(errors="replace")
    assert late_out == cpu_out


def test_qwen35_cuda_failure_keeps_the_scored_logprobs(tmp_path):
    """The two tests above compare generated text, and this fixture's text
    does not see the recurrent state: it prints the same letter whatever the
    fold holds, so a rollback that lost or doubled a token's update would
    pass them. --score's log-probabilities do see it. The same text after
    another prefix moves them by up to about 1e-2 on this fixture, while a
    device and the host agree to about 1e-6, and --score runs one forward a
    token, so RUNNER_CUDA_INJECT_FAILURE=N fails token N's forward with the
    state live. The failure lands after the device changed the state; the
    host must restore the pre-forward snapshot before it recomputes that
    token, and then every log-probability stays within 1e-4 of the CPU's."""
    if _gpu_caps().get("backend") != "cuda":
        pytest.skip("CUDA device not available")

    model = tmp_path / "ornith-score-fail.gguf"
    _make_fixture("make-test-ornith.py", model)
    text = "the runner keeps its state"

    def score(prompt, gpu, fail=None):
        env = dict(os.environ)
        env.pop("RUNNER_CUDA_INJECT_FAILURE", None)
        if fail is not None:
            env["RUNNER_CUDA_INJECT_FAILURE"] = str(fail)
        rc, out, err = _run_runner(
            [RUNNER, "-m", model, "--score", "-p", prompt, "-c", "128", "-t", "2", "--gpu", gpu],
            env=env,
        )
        assert rc == 0, err.decode(errors="replace")
        return json.loads(out), err

    def worst(a, b):
        assert a["tokens"] == b["tokens"]
        return max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"]))

    cpu, _ = score(text, "off")
    # the numbers see the fold: the text's last 12 positions after another
    # prefix (the bound below means something only if this holds)
    other, _ = score("zq " + text, "off")
    moved = max(abs(x - y) for x, y in
                zip(cpu["logprobs"][-12:], other["logprobs"][-12:]))
    assert moved > 1e-3, moved

    dev, _ = score(text, "auto")
    assert dev["score_route"]["backend"] == "cuda", dev["score_route"]
    assert worst(dev, cpu) < 1e-4, worst(dev, cpu)

    n = cpu["n_scored"]
    for at in (2, n // 2, n - 1):
        failed, err = score(text, "auto", at)
        said = f"injected CUDA runtime failure at device forward {at} ".encode()
        assert said in err, err.decode(errors="replace")
        assert worst(failed, cpu) < 1e-4, (at, worst(failed, cpu))


@pytest.mark.parametrize("nth", [1, 3, 20])
@pytest.mark.parametrize("fixture", ["ornith", "mamba2"])
def test_metal_failure_rebuilds_the_recurrent_fold(tmp_path, fixture, nth):
    """R6.68.90: a Metal forward that fails at runtime on a recurrent model.

    The device folds the host's own memory there, so when its command buffer
    fails the fold already holds the forward's tokens, and the CPU fallback
    used to fold the same tokens onto it again. The host now restarts the
    fold and rebuilds it from the tokens the model was fed, then runs the
    forward. Held by teacher-forced log-probabilities, which depend on the
    fold at every token: the fixtures' generated text does not (it is one
    letter whatever the state), so a text comparison passes either way.

    --score feeds one token a forward, so the Nth command buffer is position
    N - 1: the first forward, the third (a fold carried over two steps), and
    the twentieth, whose rebuild takes three batches.
    """
    if _gpu_caps().get("backend") != "metal":
        pytest.skip("the Metal backend's runtime fallback")
    if fixture == "ornith":
        model = tmp_path / "ornith.gguf"
        _make_fixture("make-test-ornith.py", model)
    else:   # the script takes a prefix and writes <prefix>.gguf
        model = tmp_path / "m2.gguf"
        _make_fixture("make-test-nemotron.py", tmp_path / "m2")
    text = "the quick brown fox jumps over the lazy dog"
    base = [RUNNER, "-m", model, "--score", "-p", text, "-b", "8", "-c", "128"]

    def score(gpu, env=None):
        rc, out, err_b = _run_runner(
            [*base, "--gpu", gpu],
            env={**os.environ, **(env or {})},
        )
        err = err_b.decode(errors="replace")
        assert rc == 0, err
        return json.loads(out)["logprobs"], err

    cpu, _ = score("off")
    gpu, err = score("auto")
    assert "Metal backend" in err, err
    assert len(cpu) > 24
    failed, err = score("auto", {"RUNNER_METAL_INJECT_FAILURE": str(nth)})
    assert "falling back to CPU" in err, err
    cut = nth - 1
    # before the failure the rows are the device's own, bit for bit
    assert failed[:cut] == gpu[:cut]
    # from it on they are the CPU's: exactly while the rebuild is one batch
    # (the arithmetic of the CPU's own steps), and to the last bits past that.
    # The double fold was off by 4e-4 to 4e-3 on these fixtures.
    worst = max(abs(a - b) for a, b in zip(failed[cut:], cpu[cut:]))
    if cut <= 8:
        assert failed[cut:] == cpu[cut:], worst
    assert worst < 5e-6, worst
    if cut:
        assert "rebuilding the recurrent state" in err, err


def test_metal_failure_after_a_forked_prefix_rebuilds_from_that_prefix(tmp_path):
    """R6.68.104 (found by a second reader of R6.68.90): a state installed from a
    snapshot is another sequence's.

    A request leaves prefix B in the prefix cache. The slot then serves an
    unrelated prompt A, and a request forks B and continues it. The fork installs B's recurrent
    fold; the model's record of the tokens it was fed still held A. If the
    first device forward after the fork failed, the fold was rebuilt from A's
    tokens under B's continuation. The fold's installer now names the
    sequence it belongs to, and the rebuild uses that.

    The failure is injected at the first forward after the fork: its ordinal
    is counted in a clean run of the same three requests.
    """
    if _gpu_caps().get("backend") != "metal":
        pytest.skip("the Metal backend's runtime fallback")
    model = tmp_path / "ornith.gguf"
    _make_fixture("make-test-ornith.py", model)
    B = "You are a careful assistant for a warehouse inventory team. "
    A = "quite another text, sharing nothing with it, and running on for longer than it does"
    Q = "Where is item 42?"

    def session(gpu, inject=None):
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
        log = KeptProcessLog(prefix="runner-ornith-srv-")
        env = {**os.environ, "RUNNER_METAL_STATS": "1"}
        if inject:
            env["RUNNER_METAL_INJECT_FAILURE"] = inject
        srv = subprocess.Popen(
            [str(RUNNER), "-m", str(model), "--serve", "--port", str(port), "--parallel", "1",
             "-c", "512", "--tray", "off", "--gpu", gpu],
            stdout=subprocess.DEVNULL, stderr=log.file, env=env)

        def post(path, payload):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                         json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=scaled_timeout(120)).read())
        try:
            _wait_for_health(srv, port, log, timeout_s=30)
            post("/v1/completions", {"prompt": B, "max_tokens": 1, "temperature": 0})
            post("/v1/completions", {"prompt": A, "max_tokens": 2, "temperature": 0})
            log.file.flush()
            before = pathlib.Path(log.path).read_text(errors="replace").count("metal: native prompt batch")
            r = post("/v1/completions", {"prompt": B + Q, "max_tokens": 6,
                                         "temperature": 0, "logprobs": 2})
            log.file.flush()
            log_text = pathlib.Path(log.path).read_text(errors="replace")
        except BaseException:
            log.keep()
            raise
        finally:
            if srv.poll() is None:
                srv.terminate()
                try:
                    srv.wait(timeout=scaled_timeout(30))
                except subprocess.TimeoutExpired:
                    srv.kill()
                    srv.wait()
            log.close()
        assert r["runner_telemetry"]["prompt_reuse"] == "prefix_cache", r["runner_telemetry"]
        return r["choices"][0]["logprobs"]["token_logprobs"], before, log_text


def test_ornith_timeout_scales_and_keeps_child_output(monkeypatch):
    """R6.117: timeouts scale via RUNNER_TEST_TIMEOUT_SCALE and keep child logs."""
    monkeypatch.setenv("RUNNER_TEST_TIMEOUT_SCALE", "6")
    rc, out, _ = _run_runner(
        [sys.executable, "-c",
         "import sys, time; time.sleep(0.25); sys.stdout.write('ok\\n')"],
        timeout_s=0.1,
    )
    assert rc == 0 and out.strip() == b"ok"

    monkeypatch.setenv("RUNNER_TEST_TIMEOUT_SCALE", "1")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _run_runner(
            [sys.executable, "-c",
             "import sys, time; "
             "sys.stdout.write('partial-out\\n'); sys.stdout.flush(); "
             "sys.stderr.write('partial-err\\n'); sys.stderr.flush(); "
             "time.sleep(10)"],
            timeout_s=0.1,
        )
    msg = str(excinfo.value)
    assert "partial-err" in msg and "partial-out" in msg, msg
    kept_path = None
    for tok in msg.split():
        cand = tok.rstrip(":,")
        if cand.endswith(".log") and os.path.exists(cand):
            kept_path = cand
            break
    assert kept_path is not None, msg
    try:
        assert "partial-err" in pathlib.Path(kept_path).read_text(encoding="utf-8", errors="replace")
    finally:
        os.unlink(kept_path)


def test_ornith_server_health_wait_fails_with_log_on_deadline(monkeypatch):
    """R6.117: server health wait must fail with its kept log on expiry or early exit."""
    monkeypatch.setenv("RUNNER_TEST_TIMEOUT_SCALE", "1")
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        dead_port = so.getsockname()[1]
    log = KeptProcessLog(prefix="runner-ornith-srv-")
    srv = subprocess.Popen(
        [_SERVER_PY, "-c",
         "import sys, time; sys.stderr.write('booting server\\n'); "
         "sys.stderr.flush(); time.sleep(10)"],
        stdout=subprocess.DEVNULL, stderr=log.file,
    )
    try:
        with pytest.raises(pytest.fail.Exception) as excinfo:
            _wait_for_health(srv, dead_port, log, timeout_s=0.05)
        msg = str(excinfo.value)
        assert "server never answered /health" in msg and "booting server" in msg and log.path in msg, msg
        assert log.kept and os.path.exists(log.path)
    finally:
        if srv.poll() is None:
            srv.kill()
            srv.wait()
        log.kept = False
        log.close()

    dead_log = KeptProcessLog(prefix="runner-ornith-srv-")
    dead = subprocess.Popen(
        [_SERVER_PY, "-c",
         "import sys; sys.stderr.write('fatal startup\\n'); sys.stderr.flush(); sys.exit(2)"],
        stdout=subprocess.DEVNULL, stderr=dead_log.file,
    )
    try:
        dead.wait(timeout=5)
        with pytest.raises(pytest.fail.Exception) as excinfo:
            _wait_for_health(dead, dead_port, dead_log, timeout_s=5)
        msg = str(excinfo.value)
        assert "server exited (2) before /health" in msg and "fatal startup" in msg and dead_log.path in msg, msg
        assert dead_log.kept and os.path.exists(dead_log.path)
    finally:
        dead_log.kept = False
        dead_log.close()

