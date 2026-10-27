"""phi3: the sliding window covers every layer and never changes the rotation.

phi3 exports publish a sliding window with no pattern key (Phi-3.5: 262144
over a 131072 context). The loader once read that as gemma2's alternation,
so every other layer roped from the raw local table, without the LongRoPE
short factors or the attention factor, and Phi-3.5 sat at mean KL 0.275
from its float32 reference where another engine on the same file reads
0.049. The fixture carries non-identity short factors, so a layer that ropes
without them moves the logits: a window at least the trained context must
give the same bytes as no window at all, and a window shorter than the
prompt must still mask (the window path is live, not switched off).
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "the runner reads a prompt in windows and every weight matrix is multiplied"


def _make(tmp_path, name, *extra):
    path = tmp_path / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--arch", "phi3", *extra, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def _score(runner_bin, model):
    proc = subprocess.run(
        [runner_bin, "-m", str(model), "--score", "-p", PROMPT, "--gpu", "off",
         "--no-tray", "-c", "256"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert b"phi3 LongRoPE short factors" in proc.stderr
    assert b'"n_scored"' in proc.stdout
    return proc.stdout


def test_phi3_window_past_the_context_changes_nothing(runner_bin, tmp_path):
    plain = _score(runner_bin, _make(tmp_path, "plain"))
    wide = _score(runner_bin, _make(tmp_path, "wide", "--swa-window", "512"))
    assert plain == wide


def test_phi3_window_shorter_than_the_prompt_still_masks(runner_bin, tmp_path):
    plain = _score(runner_bin, _make(tmp_path, "plain"))
    short = _score(runner_bin, _make(tmp_path, "short", "--swa-window", "16"))
    assert plain != short
