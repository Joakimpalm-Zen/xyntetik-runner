"""spark2_5 (XHToken Spark 2.5 / 2.6): a dense transformer with Q/K/V fused in
attn_qkv, a per-HEAD sigmoid output gate, a 3-sliding:1-full bool pattern
whose two layer types rope different widths at different bases, and a gated
exact-GELU FFN.

The fixtures scale the real geometry down but keep its shape (head_dim
decoupled from n_embd/n_head, the gate n_embd x n_head, the real pattern).
The real model's math is gated separately against llama.cpp on the
publisher's files; these tests pin what a green build cannot fake: the file
loads through admission and the fused slicing, decoding is deterministic, the
gate and the pattern actually participate, and the backward refuses by name.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--spark2-5", *extra, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _decode(runner_bin, model, n=24):
    p = subprocess.run([runner_bin, "-m", model, "-p", "hello world, again and again", "-n", str(n),
                        "--temp", "0", "--gpu", "off", "-c", "256"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_loads_and_decodes_deterministically(runner_bin, tmp_path):
    m = _make(tmp_path, "spark")
    a = _decode(runner_bin, m)
    assert a and a == _decode(runner_bin, m)


def test_the_per_head_gate_participates(runner_bin, tmp_path):
    gated = _decode(runner_bin, _make(tmp_path, "gated"))
    flat = _decode(runner_bin, _make(tmp_path, "flat", "--spark-gate-flat"))
    assert gated != flat, "a flat gate decoded the same text: the gate is not applied"


def test_the_sliding_pattern_is_read(runner_bin, tmp_path):
    mixed = _decode(runner_bin, _make(tmp_path, "mixed"))
    slid = _decode(runner_bin, _make(tmp_path, "slid", "--spark-all-swa"))
    assert mixed != slid, "an all-sliding pattern decoded the same text: the pattern is ignored"


def test_training_is_refused_by_name(runner_bin, tmp_path):
    m = _make(tmp_path, "spark")
    p = subprocess.run([runner_bin, "-m", m, "--adapt-info", "--no-tray"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    info = json.loads(p.stdout)
    assert info["training"]["supported"] is False
    assert "per-head attention output gate" in info["training"]["reason"]
