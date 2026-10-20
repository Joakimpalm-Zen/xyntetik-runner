"""`--bench-json` must measure the phase that actually costs the user time.

The default prompt used to be one ten-token sentence. On the first outside
install that made the micro-bench look healthy while a realistic 2,100-token
prompt took 89 seconds to reach its first word — prefill, not decode, was the
wall, and the instrument could not see it. The default is now a realistic
prefill, and both phases report seconds beside their rates.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench") / "t.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out)],
                   check=True, cwd=ROOT)
    return out


def _bench(runner_bin, model, *extra):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "--bench-json", "--gpu", "off", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return json.loads(proc.stdout)


def test_default_prompt_is_a_realistic_prefill(runner_bin, model):
    d = _bench(runner_bin, model, "-n", "8")
    # far more than the old ten-token sentence; the exact count is clamped to
    # the model's context, so assert the property, not a magic number
    assert d["prompt_tokens"] > 100
    assert d["prompt_tokens"] < d["context"]


def test_prompt_and_generation_leave_room_for_each_other(runner_bin, model):
    """A prompt clamped to the very end of the context would report a decode
    rate measured over one or two steps."""
    d = _bench(runner_bin, model, "-n", "16")
    assert d["generated_tokens"] == 16
    assert d["prompt_tokens"] + d["generated_tokens"] <= d["context"]


def test_both_phases_report_seconds(runner_bin, model):
    d = _bench(runner_bin, model, "-n", "8")
    for key in ("prompt_s", "gen_s", "prompt_tok_s", "gen_tok_s"):
        assert key in d, f"{key} missing from --bench-json"
        assert d[key] >= 0
    # the rate and the duration must describe the same measurement (both are
    # printed rounded to milliseconds, so compare at that resolution)
    assert d["prompt_s"] == pytest.approx(
        d["prompt_tokens"] / d["prompt_tok_s"], abs=1e-3)
    assert d["gen_s"] == pytest.approx(
        d["generated_tokens"] / d["gen_tok_s"], abs=1e-3)


def test_explicit_prompt_still_wins(runner_bin, model):
    d = _bench(runner_bin, model, "-n", "4", "-p", "hello")
    assert d["prompt_tokens"] < 20


def test_prompt_file_is_honoured(runner_bin, model, tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("word " * 12, encoding="utf-8")
    d = _bench(runner_bin, model, "-n", "4", "-f", str(pf))
    explicit = _bench(runner_bin, model, "-n", "4", "-p", "word")
    assert d["prompt_tokens"] > explicit["prompt_tokens"]


def test_the_report_names_the_route_it_measured(runner_bin, model):
    """A rate means nothing without the route that produced it (plan P1):
    the backend, the split, where the experts ran, every path the forwards
    took (the batched prefill AND the one-row decode, not a fixed "solo"),
    the KV type, the weight types, the activation, the kernel counters, a
    fallback's reason, the build, the compiler and the switches set."""
    d = _bench(runner_bin, model, "-n", "4")
    r = d["route"]
    assert r["backend"] == "cpu" and r["gpu_layers"] == 0
    assert r["layers"] == d["layers"]
    paths = r["paths"]
    assert paths["solo"]["rows_max"] == 1 and paths["solo"]["forwards"] >= 3, paths
    assert 1 < paths["batch"]["rows_max"] <= d["prompt_tokens"], (paths, d["prompt_tokens"])
    assert "keep" not in paths and "mtp" not in paths, paths
    assert r["kv"] == "f16" and r["fallback"] == "" and r["cpu_act"] == "f32"
    assert r["weights"] and "%" in r["weights"]
    assert r["build"].split("/")[0] in ("default", "portable", "t3")
    assert r["compiler"] and isinstance(r["kernels"]["gemm_by_type"], dict)


def test_the_report_names_the_switches_set(runner_bin, model):
    """Every RUNNER_* variable is in the record (the route switches are all
    spelled that way), and one whose name says credential shows only that it
    is set."""
    import os
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "--bench-json", "--gpu", "off", "-n", "2"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        env=dict(os.environ, RUNNER_CUDA_TC="0", RUNNER_TEST_API_KEY="hunter2"))
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    tuning = json.loads(proc.stdout)["route"]["tuning"]
    assert "RUNNER_CUDA_TC=0" in tuning, tuning
    assert "RUNNER_TEST_API_KEY=(set)" in tuning and "hunter2" not in tuning, tuning


def test_the_route_line_names_where_a_failed_forward_ended(runner_bin, model):
    """The line is written after the forward: a device forward that fails and
    finishes on the CPU is reported as the CPU's (it used to be printed before
    the device was tried, so it named CUDA for rows the CPU computed)."""
    import os
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("needs a CUDA device for the injected forward failure")
    proc = subprocess.run([runner_bin, "-m", str(model), "-p", "hello there", "-n", "3",
                           "--temp", "0", "--no-tray"], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=120,
                          env=dict(os.environ, RUNNER_DEBUG_ROUTE="1",
                                   RUNNER_CUDA_INJECT_FAILURE="1"))
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    lines = [l for l in err.splitlines() if l.startswith("route: ")]
    assert "injected CUDA runtime failure" in err, err
    assert lines and "backend=cpu" in lines[0], lines


def test_gpu_fallback_fail_ends_the_run(runner_bin, model):
    """RUNNER_GPU_FALLBACK=fail: a device forward that fails at runtime ends
    the process with status 3 instead of finishing on the CPU, so a gate or a
    comparison harness cannot record host-computed rows as the device's. The
    default still releases the backend and continues."""
    import os
    caps = json.loads(subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                                     stdout=subprocess.PIPE, check=True).stdout)
    inject = {"cuda": {"RUNNER_CUDA_INJECT_FAILURE": "1"},
              "metal": {"RUNNER_METAL_INJECT_FAILURE": "once"}}.get(
                  (caps.get("gpu") or {}).get("backend"))
    if inject is None:
        pytest.skip("needs a CUDA or Metal device for the injected forward failure")
    cmd = [runner_bin, "-m", str(model), "-p", "hello there", "-n", "3",
           "--temp", "0", "--no-tray"]
    soft = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120, env=dict(os.environ, **inject))
    assert soft.returncode == 0, soft.stderr.decode(errors="replace")
    assert b"continuing on CPU" in soft.stderr
    hard = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120, env=dict(os.environ, RUNNER_GPU_FALLBACK="fail", **inject))
    assert hard.returncode == 3, hard.stderr.decode(errors="replace")
    assert b"RUNNER_GPU_FALLBACK=fail" in hard.stderr
    assert b"continuing on CPU" not in hard.stderr


def test_debug_route_prints_each_route_once(runner_bin, model):
    """RUNNER_DEBUG_ROUTE=1: one stderr line per route the forwards took (the
    batched prefill, then the one-row decode), not one per forward."""
    import os
    proc = subprocess.run([runner_bin, "-m", str(model), "-p", "hello there", "-n", "6",
                           "--gpu", "off", "--temp", "0", "--ignore-eos"],
                          cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=120, env=dict(os.environ, RUNNER_DEBUG_ROUTE="1"))
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    lines = [l for l in proc.stderr.decode().splitlines() if l.startswith("route: ")]
    assert any("path=batch" in l for l in lines), lines
    assert sum("rows=1 path=solo" in l for l in lines) == 1, lines
    assert all('weights="' in l and "cpu_act=f32" in l and 'tuning="' in l for l in lines), lines
    assert all("backend=cpu" in l and "split=0/" in l for l in lines), lines
    off = subprocess.run([runner_bin, "-m", str(model), "-p", "hello", "-n", "2",
                          "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=120)
    assert b"route: " not in off.stderr
