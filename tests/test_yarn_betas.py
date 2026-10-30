"""YaRN reads the ramp ends and the attention factor from the file.

transformers' rope_parameters carry beta_fast, beta_slow and attention_factor
for a YaRN model, and llama.cpp's converter writes them as
rope.scaling.yarn_beta_fast/_slow and rope.scaling.yarn_attn_factor. Runner
used 32, 1 and its own 1 + 0.1 ln(factor) whatever the file said, so a model
trained with other ends (IFM K2-Horizon-0.9B: 128 and 4) roped with the ramp in
the wrong place. A file without the keys keeps the old values.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEXT = "the quick brown fox jumps over the lazy dog " * 6


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--yarn", "16,32", *extra, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _score(runner_bin, model):
    p = subprocess.run([runner_bin, "-m", model, "--score", "-p", TEXT, "--gpu", "off",
                        "-c", "512"], cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)["logprobs"]


def test_declared_betas_move_the_ramp(runner_bin, tmp_path):
    default = _score(runner_bin, _make(tmp_path, "default"))
    same = _score(runner_bin, _make(tmp_path, "explicit-default", "--yarn-betas", "32,1"))
    moved = _score(runner_bin, _make(tmp_path, "k2h", "--yarn-betas", "128,4"))
    assert default == same, "declaring the defaults must change nothing"
    assert default != moved, "the file's betas were ignored"


def test_declared_attention_factor_is_used(runner_bin, tmp_path):
    default = _score(runner_bin, _make(tmp_path, "default"))
    # 1 + 0.1 ln(16), the value Runner computes itself: no change
    same = _score(runner_bin, _make(tmp_path, "same", "--yarn-attn-factor", "1.2772588722"))
    other = _score(runner_bin, _make(tmp_path, "other", "--yarn-attn-factor", "1.0"))
    assert max(abs(a - b) for a, b in zip(default, same)) < 1e-5
    assert default != other


def test_a_forced_factor_takes_the_default_attention_factor(runner_bin, tmp_path):
    """The file's attention factor is the one for ITS factor (16 here):
    under --yarn-factor 8 the magnitude is 1 + 0.1 ln(8), whatever the file
    declares, so two files that differ only in that key score the same."""
    a = _make(tmp_path, "a", "--yarn-attn-factor", "1.0")
    b = _make(tmp_path, "b", "--yarn-attn-factor", "1.5")

    def forced(model):
        p = subprocess.run([runner_bin, "-m", model, "--score", "-p", TEXT, "--gpu", "off",
                            "-c", "512", "--yarn-factor", "8"],
                           cwd=ROOT, capture_output=True, timeout=60)
        assert p.returncode == 0, p.stderr.decode(errors="replace")
        return json.loads(p.stdout)["logprobs"]
    assert forced(a) == forced(b)
    assert _score(runner_bin, a) != _score(runner_bin, b)


def test_a_ramp_that_is_not_one_is_refused(runner_bin, tmp_path):
    m = _make(tmp_path, "bad", "--yarn-betas", "1,32")
    p = subprocess.run([runner_bin, "-m", m, "-p", "hi", "-n", "1", "--gpu", "off"],
                       cwd=ROOT, capture_output=True, timeout=60)
    assert p.returncode != 0 and b"are not a ramp" in p.stderr
