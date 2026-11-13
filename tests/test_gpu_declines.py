"""Public behavior gates for GPU admission fallbacks.

Every unsupported model must name the feature that caused the backend to
decline.  A generic quantization message is actively misleading when all
tensors use a supported type, and a CPU-vs-auto comparison alone would pass
vacuously after an unexplained fallback.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "diagnostic gate"


@pytest.fixture(scope="module")
def moe_fixtures(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("gpu-declines") / "f"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-moe.py", str(prefix)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return prefix


def _run(runner_bin, model, mode, env=None):
    return subprocess.run(
        [runner_bin, "-m", str(model), "-p", PROMPT, "-n", "4", "--temp",
         "0", "--gpu", mode],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        env=env)


def test_gated_sparse_moe_declines_loudly(runner_bin, moe_fixtures):
    model = f"{moe_fixtures}.afmoe-plain.gguf"
    cpu = _run(runner_bin, model, "off")
    auto = _run(runner_bin, model, "auto")
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert auto.returncode == 0, auto.stderr.decode(errors="replace")
    assert cpu.stdout, "CPU produced no output; fallback comparison is vacuous"
    assert cpu.stdout == auto.stdout

    err = auto.stderr.decode(errors="replace")
    assert "cuda backend on" not in err.lower(), err
    assert "metal backend on" not in err.lower(), err
    assert "gated attention with sparse MoE (afmoe)" in err, err
    assert "quant type without a" not in err, err


@pytest.mark.parametrize(
    "flags,tensor",
    [
        (["--granite"], "output.weight"),
        ([], "blk.0.attn_q.weight"),
    ],
    ids=["output", "layer-weight"],
)
def test_unsupported_tensor_names_itself(runner_bin, tmp_path, flags, tensor):
    model = tmp_path / (tensor.replace(".", "-") + ".gguf")
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", "--wide", *flags,
         "--gpu-unsupported", tensor, str(model)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

    # CUDA gained the codebook i-quants on 2026-09-14 and Metal on
    # 2026-09-17, so both backends now serve every dense format the loader
    # reads and nothing declines the fixture's IQ2_XXS tensor naturally. On
    # either backend (and in CPU-only CI, which compiles the message) the
    # diagnostic is reached through the admission tracer, which names the
    # type to refuse.
    env = {**os.environ,
           "RUNNER_CUDA_INIT_INJECT_FAILURE": "no-kernel:IQ2_XXS",
           "RUNNER_METAL_INIT_INJECT_FAILURE": "no-kernel:IQ2_XXS"}
    cpu = _run(runner_bin, model, "off")
    auto = _run(runner_bin, model, "auto", env=env)
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert auto.returncode == 0, auto.stderr.decode(errors="replace")
    assert cpu.stdout, "CPU produced no output; fallback comparison is vacuous"
    assert cpu.stdout == auto.stdout

    err = auto.stderr.decode(errors="replace")
    assert tensor in err, err
    assert "IQ2_XXS" in err, err
    assert "using CPU" in err, err


def test_metal_refuses_a_layer_constant_it_cannot_make(runner_bin, tmp_path):
    """R6.68.86: every per-layer constant table is checked where it is made.

    The gemma-4 E-series post-norm table was uploaded and never checked, so
    a failed allocation left a nil buffer that the first forward bound.
    """
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("the Metal backend's init")
    model = tmp_path / "es.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", "--eseries",
         "3,16", str(model)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    cpu = _run(runner_bin, model, "off")
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert cpu.stdout
    # tables this fixture carries: a table the model lacks is never made
    for table in ("ppn", "attn_norm", "ffn_norm", "qn", "pan", "pfn"):
        env = {**os.environ,
               "RUNNER_METAL_INIT_INJECT_FAILURE": "const:" + table}
        auto = _run(runner_bin, model, "auto", env=env)
        err = auto.stderr.decode(errors="replace")
        assert auto.returncode == 0, err
        assert "Metal initialization failed (per-layer buffer allocation)" in err, (table, err)
        assert auto.stdout == cpu.stdout, table


def test_cuda_runtime_absence_is_loud(runner_bin, tmp_path):
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") == "metal":
        pytest.skip("CUDA-only diagnostic")
    if caps.get("gpu") is not None:
        pytest.skip("CUDA runtime is available")

    model = tmp_path / "dense.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    cpu = _run(runner_bin, model, "off")
    auto = _run(runner_bin, model, "auto")
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert auto.returncode == 0, auto.stderr.decode(errors="replace")
    assert cpu.stdout == auto.stdout
    err = auto.stderr.decode(errors="replace")
    # Two ways to have no CUDA runtime, each with its own reason: no driver
    # library (a machine without NVIDIA's driver, the hosted runners), or a
    # driver with no device it may use (a CUDA host with CUDA_VISIBLE_DEVICES
    # set to an invalid ordinal such as -1; an empty value reads as unset on
    # Windows). --caps reports no GPU for both.
    assert ("CUDA driver library is unavailable" in err
            or "CUDA initialization failed" in err), err
    assert "using CPU" in err, err


def test_cuda_shared_table_failure_is_loud(runner_bin, tmp_path):
    caps = json.loads(subprocess.run(
        [runner_bin, "--caps"], cwd=ROOT, stdout=subprocess.PIPE,
        check=True).stdout)
    if (caps.get("gpu") or {}).get("backend") == "metal":
        pytest.skip("CUDA-only diagnostic")

    model = tmp_path / "dense.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    env = {**os.environ,
           "RUNNER_CUDA_INIT_INJECT_FAILURE": "shared-pointer-tables"}
    auto = _run(runner_bin, model, "auto", env=env)
    assert auto.returncode == 0, auto.stderr.decode(errors="replace")
    err = auto.stderr.decode(errors="replace")
    assert "CUDA shared per-layer pointer-table allocation failed" in err, err
    assert "using CPU" in err, err


def _cuda(exe):
    try:
        caps = json.loads(subprocess.run([str(exe), "--caps"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, timeout=60).stdout)
    except Exception:
        return False
    return (caps.get("gpu") or {}).get("backend") == "cuda"


@pytest.fixture(scope="module")
def head_width_fixtures(tmp_path_factory):
    d = tmp_path_factory.mktemp("head-width")
    for hd in (16, 17):
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                        str(d / f"hd{hd}.gguf"), "--head-dim", str(hd)],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return d


def _score(runner_bin, model, mode):
    return subprocess.run(
        [runner_bin, "-m", str(model), "--score", "-p",
         "the quick brown fox jumps over the lazy dog and then runs far away",
         "-c", "128", "--gpu", mode],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def test_an_odd_attention_head_width_runs_on_the_cpu(runner_bin, head_width_fixtures):
    """R6.68.112: k_attn, k_attn_dec with k_attn_merge, and k_attn_dec_seq
    walk a head in pairs, so at an odd width the last element was never
    written; every other CUDA attention kernel declines an odd width to
    them. Such a model loads and runs on the CPU, names the reason once, and
    scores exactly as --gpu off does; --info fit says the CPU too, and the
    even control keeps the device."""
    if not _cuda(runner_bin):
        pytest.skip("the rule is CUDA's: the CPU and Metal take an odd head width")
    odd, even = head_width_fixtures / "hd17.gguf", head_width_fixtures / "hd16.gguf"
    cpu, auto = _score(runner_bin, odd, "off"), _score(runner_bin, odd, "auto")
    assert cpu.returncode == 0, cpu.stderr.decode(errors="replace")
    assert auto.returncode == 0, auto.stderr.decode(errors="replace")
    err = auto.stderr.decode(errors="replace")
    # one line names it (the score route record repeats it as its fallback)
    said = [l for l in err.splitlines() if l.startswith("gpu:") and "attention heads are 17 wide" in l]
    assert len(said) == 1, err
    assert json.loads(auto.stdout)["score_route"]["backend"] == "cpu", auto.stdout[-400:]
    assert "CUDA backend on" not in err, err
    assert json.loads(cpu.stdout)["logprobs"], "no scores: the comparison is vacuous"
    assert auto.stdout == cpu.stdout
    fit = subprocess.run([runner_bin, "--info", "fit", "-m", str(odd)], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert fit.returncode == 0, fit.stderr.decode(errors="replace")
    said = fit.stdout.decode(errors="replace")
    assert "the load runs on the CPU" in said and "attention heads are 17 wide" in said, said
    ctl = _score(runner_bin, even, "auto")
    assert ctl.returncode == 0, ctl.stderr.decode(errors="replace")
    assert "CUDA backend on" in ctl.stderr.decode(errors="replace")
    fit = subprocess.run([runner_bin, "--info", "fit", "-m", str(even)], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert "the load runs on the CPU" not in fit.stdout.decode(errors="replace")
