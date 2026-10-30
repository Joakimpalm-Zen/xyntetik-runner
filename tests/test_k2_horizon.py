"""k2-horizon (IFM K2-Horizon, the dense members): a llama-shaped decoder
whose RMSNorms normalise the hidden vector in attention.group_norm_groups
equal slices before one full-width weight.

The fixture scales the embedding table's slices 1x, 8x, 64x and 512x, so one
RMS over the whole vector (the full-width norm) is dominated by the last slice
and the grouped norm is not: the two must decode differently. The real
model's math is gated separately on the publisher's files; these tests pin
the group boundary, the refusals and the GPU fallback line.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    *extra, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _decode(runner_bin, model, n=24):
    p = subprocess.run([runner_bin, "-m", model, "-p", "hello world, again and again",
                        "-n", str(n), "--temp", "0", "--gpu", "off", "-c", "256"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_grouped_norm_is_not_the_full_width_norm(runner_bin, tmp_path):
    grouped = _decode(runner_bin, _make(tmp_path, "g4", "--k2-horizon", "4"))
    full = _decode(runner_bin, _make(tmp_path, "g1", "--k2-horizon", "4",
                                     "--k2-declare-groups", "1"))
    assert grouped and full
    assert grouped != full, "4 groups decoded like 1: the groups are not applied"
    assert grouped == _decode(runner_bin, _make(tmp_path, "g4b", "--k2-horizon", "4"))


def test_a_group_count_that_does_not_divide_the_width_is_refused(runner_bin, tmp_path):
    m = _make(tmp_path, "g3", "--k2-horizon", "4", "--k2-declare-groups", "3")
    p = subprocess.run([runner_bin, "-m", m, "-p", "hi", "-n", "1", "--gpu", "off"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode != 0
    assert b"norm groups do not divide" in p.stderr


def test_training_is_refused_by_name(runner_bin, tmp_path):
    m = _make(tmp_path, "g4", "--k2-horizon", "4")
    p = subprocess.run([runner_bin, "-m", m, "--adapt-info", "--no-tray"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    info = json.loads(p.stdout)
    assert info["training"]["supported"] is False
    assert "grouped RMSNorm" in info["training"]["reason"]
