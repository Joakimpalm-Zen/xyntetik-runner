"""CUDA's grouped decode attention is the same bytes run to run (R3.29).

`k_attn_dec_gqa` reads each slice once, with a running maximum per head and
the sums and value accumulators rescaled where it rises (src/kernels.cu).
For one geometry and position its slice bounds, tile order, warp butterflies
and rescale sequence are fixed, so scoring a text twice must give the same
bytes. `--score` scores each position with its own single-token forward, so
every position takes the decode attention. Two depths: a short text, where
most of the 32 slices are empty, and a long one, where every slice holds
several 32-row tiles. Two head widths: 128, the fp16 cache's 16-byte staging
path, and the fixture's own 16, the word-staged path.

Skipped where there is no CUDA device.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_record import body_bytes  # noqa: E402

N_CTX = 2048
SENTENCE = "the quick brown fox jumps over the lazy dog"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    caps = json.loads(subprocess.run([str(exe), "--caps"], cwd=ROOT, check=True,
                                     stdout=subprocess.PIPE).stdout)
    if (caps.get("gpu") or {}).get("backend") != "cuda":
        pytest.skip("the grouped decode attention needs a CUDA device")
    return exe


def _model(tmp_path, *flags):
    m = tmp_path / "m.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m), *flags],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _score(exe, model, text):
    p = subprocess.run([str(exe), "-m", str(model), "--gpu", "auto", "-c", str(N_CTX),
                        "--score", "-p", text, "--no-tray"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def _grouped_decode_ran(exe, model):
    p = subprocess.run([str(exe), "-m", str(model), "--gpu", "auto", "--bench-json"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    line = next(l for l in p.stdout.decode().splitlines() if l.startswith("{"))
    return json.loads(line)["route"]["kernels"]["cuda_attn_dec_gqa"] > 0


@pytest.mark.parametrize("flags", [("--head-dim", "128"), ()], ids=["hd128", "hd16"])
@pytest.mark.parametrize("repeats", [2, 24], ids=["short", "long"])
def test_grouped_decode_is_the_same_bytes_twice(runner_bin, tmp_path, flags, repeats):
    model = _model(tmp_path, *flags)
    assert _grouped_decode_ran(runner_bin, model)
    text = " ".join([SENTENCE] * repeats)
    a = _score(runner_bin, model, text)
    b = _score(runner_bin, model, text)
    n = json.loads(a)["n_scored"]
    assert n > (1000 if repeats > 2 else 50), n
    assert body_bytes(a) == body_bytes(b)


def _mla_decode_ran(exe, model):
    p = subprocess.run([str(exe), "-m", str(model), "--gpu", "auto", "--bench-json"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    line = next(l for l in p.stdout.decode().splitlines() if l.startswith("{"))
    return json.loads(line)["route"]["kernels"]["cuda_attn_dec_mla"]


@pytest.mark.parametrize("repeats", [2, 24], ids=["short", "long"])
def test_absorbed_mla_decode_is_the_same_bytes_twice_and_the_cpus(runner_bin, tmp_path, repeats):
    """R3.31: deepseek2's absorbed form (16 heads on one 576-wide row, the
    --mla-wide fixture) decodes through k_attn_dec_mla, one pass over a slice's
    rows for all 16 heads with V read from the K row. Its slices, tiles and
    rescales are fixed, so a text scores to the same bytes twice, and every
    position lands within reduction-order rounding of the CPU's."""
    model = _model(tmp_path, "--arch", "deepseek2", "--mla-wide")
    assert _mla_decode_ran(runner_bin, model) > 0
    text = " ".join([SENTENCE] * repeats)
    a = _score(runner_bin, model, text)
    assert body_bytes(a) == body_bytes(_score(runner_bin, model, text))
    n = json.loads(a)["n_scored"]
    assert n > (400 if repeats > 2 else 10), n
    p = subprocess.run([str(runner_bin), "-m", str(model), "--gpu", "off", "-c", str(N_CTX),
                        "--score", "-p", text, "--no-tray"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    ga, cpu = json.loads(a)["logprobs"], json.loads(p.stdout)["logprobs"]
    assert len(ga) == len(cpu) == n
    worst = max(abs(x - y) for x, y in zip(ga, cpu))
    assert worst < 1e-4, "absorbed MLA decode against the CPU: max |dlogprob| %.3g" % worst
