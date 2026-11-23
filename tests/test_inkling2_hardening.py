"""R4.48.6: the inkling2_loop loader refuses a malformed file with a sentence, as the other loaders do.

A tiny but complete inkling2_loop GGUF is written here (4 layers, a loop of 3 passes over layers 1 and 2, NVFP4 expert
stacks with their per-expert .scale, every small tensor F32, a byte-level vocabulary so the tokenizer loads). It must
load and answer; then each damaged copy must be refused, naming what is wrong:
  - the tensor data cut short (the GGUF reader refuses it before the loader sees a tensor);
  - a .scale with the wrong number of values;
  - a loop key missing;
  - an NVFP4 expert stack without its .scale;
  - an NVFP4 tensor where the loader takes only a float or a dequantizable type (a dense one).
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling2_tiny import E, _kvs, _tensors, _write  # noqa: E402


def _run(runner_bin, model, *args):
    return subprocess.run([str(runner_bin), "-m", str(model), "--gpu", "off", "-t", "2", "-c", "64", *args],
                          capture_output=True, text=True, errors="replace", timeout=600)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("inkling2")
    return d, _write(d / "tiny.gguf", _kvs(), _tensors())


def test_the_tiny_file_loads_and_answers(runner_bin, tiny):
    _, good = tiny
    r = _run(runner_bin, good, "-p", "hello", "-n", "3", "--temp", "0")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "inkling2_loop: 4 layers, 8 applications" in r.stderr


@pytest.mark.parametrize("case,make,sentence", [
    ("truncated", lambda d: _write(d / "cut.gguf", _kvs(), _tensors(), cut=4096), "a truncated or incomplete download"),
    ("scale shape", lambda d: _write(d / "scale.gguf", _kvs(), _tensors(scale_len=E - 1)), "must be F32 with 4 values"),
    ("loop key", lambda d: _write(d / "loop.gguf", _kvs(drop="inkling2_loop.loop.count"), _tensors()), "loop.count"),
    ("nvfp4 without scale", lambda d: _write(d / "noscale.gguf", _kvs(), _tensors(no_scale_layer=2)),
     "is NVFP4 but the file has no blk.2.ffn_down_exps.scale"),
    ("nvfp4 dense", lambda d: _write(d / "dense.gguf", _kvs(), _tensors(dense_nvfp4=True)),
     "blk.0.attn_q.weight is type 40, which this loader does not decode"),
    ("used > expert", lambda d: _write(d / "used_high.gguf", _kvs(used=E + 1), _tensors()),
     f"expert_used_count {E + 1} is outside 1..{E}"),
    ("used == 0", lambda d: _write(d / "used_zero.gguf", _kvs(used=0), _tensors()),
     f"expert_used_count 0 is outside 1..{E}"),
])
def test_a_malformed_file_is_refused_with_a_sentence(runner_bin, tiny, case, make, sentence):
    d, _ = tiny
    bad = make(d)
    r = _run(runner_bin, bad, "-p", "hello", "-n", "1")
    assert r.returncode != 0, (case, r.stderr[-1000:])
    assert sentence in r.stderr, (case, r.stderr[-1500:])


def test_the_expert_cache_line_is_a_budget_capped_at_every_expert_decoded(runner_bin, tiny):
    """R4.48.17: the expert cache holds at most every expert decoded (the tiny file's: 4 blocks x 4 experts x (2 x 64 x
    64 + 64 x 64) floats, 786432 bytes), so --info fit and the load line print the budget capped there, not the
    4 GiB default as a size; fit's total counts the capped figure."""
    import re
    _, good = tiny
    r = subprocess.run([str(runner_bin), "-m", str(good), "--info", "fit", "-c", "64", "--gpu", "off"],
                       capture_output=True, text=True, errors="replace", timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "expert cache  up to 0.00 GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)" in r.stdout, r.stdout
    total = re.search(r"total\s+([0-9.]+) GiB", r.stdout)
    assert total and float(total.group(1)) < 0.01, r.stdout
    r = _run(runner_bin, good, "-p", "hello", "-n", "1", "--temp", "0")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "expert cache up to 0.00 GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)" in r.stderr, r.stderr[-2000:]
