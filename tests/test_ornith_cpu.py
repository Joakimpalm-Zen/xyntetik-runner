"""Public CPU tracer for the Ornith-1.0/Qwen3.5 hybrid architecture."""
import math
import json
import pathlib
import subprocess
import sys
import os

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")

def test_qwen35_hybrid_loads_and_decodes(tmp_path):
    model = tmp_path / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    proc = subprocess.run(
        [RUNNER, "-m", model, "-p", "hi", "-n", "2",
         "--temp", "0", "--gpu", "off", "-c", "32"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"hybrid SSM/attention architecture" not in proc.stderr
    # This greedy suffix is the llama.cpp reference result for the exact same
    # generated weights and prompt-token sequence. It protects the recurrent
    # state orientation, Q scaling, group broadcast, convolution ordering and
    # the full-attention post-norm placement as one end-to-end contract.
    assert proc.stdout.splitlines() == [b"hiii"]


def test_qwen35_legacy_ornith_dt_name_loads(tmp_path):
    model = tmp_path / "ornith-legacy.gguf"
    env = {**os.environ, "ORNITH_LEGACY_DT": "1"}
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT, env=env)
    proc = subprocess.run(
        [RUNNER, "-m", model, "-p", "hi", "-n", "1",
         "--temp", "0", "--gpu", "off", "-c", "32"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


def test_qwen35_hybrid_gpu_auto_matches_cpu(tmp_path):
    caps = json.loads(subprocess.run(
        [RUNNER, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    gpu_caps = caps.get("gpu")
    if not gpu_caps:
        pytest.skip("no GPU backend: this test asserts a device/host match")
    backend = gpu_caps.get("backend")
    model = tmp_path / "ornith-gpu.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    base = [RUNNER, "-m", model, "-p", "hi", "-n", "2",
            "--temp", "0", "-c", "32"]
    cpu = subprocess.run([*base, "--gpu", "off"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=20)
    gpu = subprocess.run([*base, "--gpu", "auto"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=20)
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert gpu.returncode == 0, gpu.stderr.decode(errors="replace")
    assert gpu.stdout == cpu.stdout
    if backend == "cuda":
        assert b"CUDA backend" in gpu.stderr
        assert b"no kernels for the recurrent" not in gpu.stderr
    elif backend == "metal":
        # the dense qwen35 hybrid runs on the device (R4.10), its greedy
        # output the CPU's (asserted above)
        assert b"Metal backend" in gpu.stderr
        assert b"not on the metal backend" not in gpu.stderr

    if backend == "cuda":
        fallback = subprocess.run(
            [*base, "--gpu", "auto"], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=20,
            env={**os.environ, "RUNNER_CUDA_INJECT_FAILURE": "1"},
        )
        assert fallback.returncode == 0, fallback.stderr.decode(errors="replace")
        assert fallback.stdout == cpu.stdout
        assert b"injected CUDA runtime failure" in fallback.stderr


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
    caps = json.loads(subprocess.run(
        [RUNNER, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("CUDA device not available")

    model = tmp_path / "ornith-late-fail.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT)
    base = [RUNNER, "-m", model, "-p", "hi", "-n", "4",
            "--temp", "0", "-c", "32"]
    cpu = subprocess.run([*base, "--gpu", "off"], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         timeout=20)
    late = subprocess.run(
        [*base, "--gpu", "auto"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=20,
        env={**os.environ, "RUNNER_CUDA_INJECT_FAILURE": "2"})
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert late.returncode == 0, late.stderr.decode(errors="replace")
    # Naming the ordinal is what proves the failure landed past prefill; a hook
    # that always fired on tile one would report forward 1 here.
    assert b"injected CUDA runtime failure at device forward 2" in late.stderr, \
        late.stderr.decode(errors="replace")
    assert late.stdout == cpu.stdout


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
    caps = json.loads(subprocess.run(
        [RUNNER, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("CUDA device not available")

    model = tmp_path / "ornith-score-fail.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    text = "the runner keeps its state"

    def score(prompt, gpu, fail=None):
        env = dict(os.environ)
        env.pop("RUNNER_CUDA_INJECT_FAILURE", None)
        if fail is not None:
            env["RUNNER_CUDA_INJECT_FAILURE"] = str(fail)
        p = subprocess.run([RUNNER, "-m", model, "--score", "-p", prompt,
                            "-c", "128", "-t", "2", "--gpu", gpu],
                           cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=60, env=env)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        return json.loads(p.stdout), p.stderr

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
    caps = json.loads(subprocess.run(
        [RUNNER, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("the Metal backend's runtime fallback")
    if fixture == "ornith":
        model = tmp_path / "ornith.gguf"
        subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    else:   # the script takes a prefix and writes <prefix>.gguf
        model = tmp_path / "m2.gguf"
        subprocess.run([sys.executable, ROOT / "scripts/make-test-nemotron.py",
                        tmp_path / "m2"], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    text = "the quick brown fox jumps over the lazy dog"
    base = [RUNNER, "-m", model, "--score", "-p", text, "-b", "8", "-c", "128"]

    def score(gpu, env=None):
        r = subprocess.run([*base, "--gpu", gpu], cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=60,
                           env={**os.environ, **(env or {})})
        assert r.returncode == 0, r.stderr.decode(errors="replace")
        return json.loads(r.stdout)["logprobs"], r.stderr.decode(errors="replace")

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
    import socket
    import time
    import urllib.request
    caps = json.loads(subprocess.run(
        [RUNNER, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True, text=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("the Metal backend's runtime fallback")
    model = tmp_path / "ornith.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-ornith.py", model],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    B = "You are a careful assistant for a warehouse inventory team. "
    A = "quite another text, sharing nothing with it, and running on for longer than it does"
    Q = "Where is item 42?"

    def session(gpu, inject=None):
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
        log = tmp_path / f"server-{gpu}-{inject}.log"
        env = {**os.environ, "RUNNER_METAL_STATS": "1"}
        if inject:
            env["RUNNER_METAL_INJECT_FAILURE"] = inject
        srv = subprocess.Popen(
            [str(RUNNER), "-m", str(model), "--serve", "--port", str(port), "--parallel", "1",
             "-c", "512", "--tray", "off", "--gpu", gpu],
            stdout=subprocess.DEVNULL, stderr=open(log, "w"), env=env)

        def post(path, payload):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                         json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            post("/v1/completions", {"prompt": B, "max_tokens": 1, "temperature": 0})
            post("/v1/completions", {"prompt": A, "max_tokens": 2, "temperature": 0})
            before = log.read_text(errors="replace").count("metal: native prompt batch")
            r = post("/v1/completions", {"prompt": B + Q, "max_tokens": 6,
                                         "temperature": 0, "logprobs": 2})
        finally:
            srv.terminate()
            srv.wait(timeout=30)
        assert r["runner_telemetry"]["prompt_reuse"] == "prefix_cache", r["runner_telemetry"]
        return r["choices"][0]["logprobs"]["token_logprobs"], before, log.read_text(errors="replace")

    cpu, _, _ = session("off")
    clean, forwards_before_the_fork, _ = session("auto")
    assert forwards_before_the_fork > 0
    assert clean == pytest.approx(cpu, abs=5e-6)
    failed, _, log = session("auto", inject=str(forwards_before_the_fork + 1))
    assert "falling back to CPU" in log, log[-1200:]
    assert "rebuilding the recurrent state" in log, log[-1200:]
    assert failed == pytest.approx(cpu, abs=5e-6)
