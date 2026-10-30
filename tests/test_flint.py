"""flint (RecursiveSI): QK norms and attention sinks on every block, a
sliding window with rope on the blocks the boolean pattern marks and NoPE
full attention elsewhere, two leading dense blocks, then sigmoid-routed
experts with an always-on shared expert.

The arithmetic is checked against transformers' own flint in float32 on a
converted random model (docs/compat-reports/flint-2026-10-30/README.md);
these fixtures pin the binding: the file loads and decodes deterministically,
the sinks and the window take part, the two pattern arrays must agree, the
GPU is not used for it yet, and the backward refuses it by name.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-flint.py", str(path), *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _run(runner_bin, model, *extra):
    return subprocess.run([runner_bin, "-m", model, "-p", "hello world, again and again",
                           "-n", "24", "--temp", "0", "-c", "256", *extra],
                          cwd=ROOT, capture_output=True, timeout=60)


def _decode(runner_bin, model):
    p = _run(runner_bin, model, "--gpu", "off")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_loads_and_decodes_deterministically(runner_bin, tmp_path):
    m = _make(tmp_path, "flint")
    a = _decode(runner_bin, m)
    assert a and a == _decode(runner_bin, m)


def test_the_sinks_take_part(runner_bin, tmp_path):
    with_sinks = _decode(runner_bin, _make(tmp_path, "sinks"))
    without = _decode(runner_bin, _make(tmp_path, "nosinks", "--sinks-off"))
    assert with_sinks != without, "sinks at -1e4 decoded the same text: the sinks are not applied"


def test_the_window_takes_part(runner_bin, tmp_path):
    narrow = _decode(runner_bin, _make(tmp_path, "w4"))
    wide = _decode(runner_bin, _make(tmp_path, "w256", "--window", "256"))
    assert narrow != wide, "a window wider than the prompt decoded the same text: the window is ignored"


def test_the_rope_pattern_must_be_the_sliding_pattern(runner_bin, tmp_path):
    p = _run(runner_bin, _make(tmp_path, "mismatch", "--rope-mismatch"), "--gpu", "off")
    assert p.returncode != 0
    assert b"only files that rope exactly their sliding layers are supported" in p.stderr
    p = _run(runner_bin, _make(tmp_path, "norope", "--no-rope-pattern"), "--gpu", "off")
    assert p.returncode != 0 and b"rope_pattern arrays" in p.stderr


def test_the_gpu_is_not_used_for_it_yet(runner_bin, tmp_path):
    p = _run(runner_bin, _make(tmp_path, "auto"))
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    if b"gpu:" in p.stderr and b"using CPU" in p.stderr:
        assert b"flint is validated on the CPU only" in p.stderr


def test_training_is_refused_by_name(runner_bin, tmp_path):
    m = _make(tmp_path, "flint")
    p = subprocess.run([runner_bin, "-m", m, "--adapt-info", "--no-tray"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    info = json.loads(p.stdout)
    assert info["training"]["supported"] is False
    # the backward names the first piece it lacks; flint has three of them
    assert info["training"]["reason"] in ("MoE router gating other than softmax",
                                          "MoE shared experts", "attention sinks")
