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
from inkling2_tiny import *  # noqa: E402,F401,F403  (the tiny file's constants, for the tests that import this module)
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


def test_the_load_line_prints_the_requested_thread_count(runner_bin, tiny):
    """R4.48.18: CLI load line prints the thread count the forward will use (model_inkling2_threads), not 0."""
    _, good = tiny
    r = _run(runner_bin, good, "-p", "hello", "-n", "1", "--temp", "0")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "| 2 threads |" in r.stderr, r.stderr[-2000:]


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
    import os
    import re
    _, good = tiny
    # the expert cache is the reference route's (R4.48.15/19: the fast route keeps none, tests/test_inkling2_fast.py)
    env = dict(os.environ, RUNNER_INKLING2_ROUTE="reference")
    r = subprocess.run([str(runner_bin), "-m", str(good), "--info", "fit", "-c", "64", "--gpu", "off"],
                       capture_output=True, text=True, errors="replace", timeout=300, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "expert cache  up to 0.00 GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)" in r.stdout, r.stdout
    total = re.search(r"total\s+([0-9.]+) GiB", r.stdout)
    assert total and float(total.group(1)) < 0.01, r.stdout
    r = subprocess.run([str(runner_bin), "-m", str(good), "--gpu", "off", "-t", "2", "-c", "64", "-p", "hello", "-n", "1",
                        "--temp", "0"], capture_output=True, text=True, errors="replace", timeout=600, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "expert cache up to 0.00 GiB (the budget, RUNNER_INKLING2_EXPERT_CACHE)" in r.stderr, r.stderr[-2000:]


def test_score_keep_all_matches_solo_and_rows_dump(runner_bin, tiny):
    """--score (keep_all) and RUNNER_INKLING2_ROWS still compute and emit every row identically to -b 1."""
    import json
    import os
    d, good = tiny
    prompt = "hello world, inkling2_loop score check!"
    r_batched = _run(runner_bin, good, "--score", "1", "-b", "8", "-p", prompt)
    assert r_batched.returncode == 0, r_batched.stderr[-2000:]
    r_solo = _run(runner_bin, good, "--score", "1", "-b", "1", "-p", prompt)
    assert r_solo.returncode == 0, r_solo.stderr[-2000:]
    j_batched = json.loads(r_batched.stdout)
    j_solo = json.loads(r_solo.stdout)
    assert j_batched["logprobs"] == j_solo["logprobs"]
    assert j_batched["top1"] == j_solo["top1"]
    assert j_batched["top"] == j_solo["top"]
    rows_file = d / "rows.f32"
    env = dict(os.environ, RUNNER_INKLING2_ROWS=str(rows_file))
    r_dump = subprocess.run([str(runner_bin), "-m", str(good), "--gpu", "off", "-t", "2", "-c", "64",
                             "-b", "8", "--score", "1", "-p", prompt],
                            capture_output=True, text=True, errors="replace", timeout=600, env=env)
    assert r_dump.returncode == 0, r_dump.stderr[-2000:]
    assert rows_file.exists() and rows_file.stat().st_size == len(j_batched["logprobs"]) * VUSED * 4
    r_gen = subprocess.run([str(runner_bin), "-m", str(good), "--gpu", "off", "-t", "2", "-c", "64",
                            "-b", "8", "-p", prompt, "-n", "1", "--temp", "0"],
                           capture_output=True, text=True, errors="replace", timeout=600)
    assert r_gen.returncode == 0, r_gen.stderr[-2000:]
    r_gen_dump = subprocess.run([str(runner_bin), "-m", str(good), "--gpu", "off", "-t", "2", "-c", "64",
                                 "-b", "8", "-p", prompt, "-n", "1", "--temp", "0"],
                                capture_output=True, text=True, errors="replace", timeout=600, env=env)
    assert r_gen_dump.returncode == 0, r_gen_dump.stderr[-2000:]
    assert r_gen.stdout == r_gen_dump.stdout


def test_default_batch_256_in_fit_and_score_split(runner_bin, tiny):
    """Default chunk is 256 (clamped to -c) in --info fit, and --score on >256 tokens matches at -b 256 and -b 1."""
    import json
    import re
    _, good = tiny
    r_384 = subprocess.run([str(runner_bin), "-m", str(good), "--info", "fit", "-c", "384", "--gpu", "off"],
                           capture_output=True, text=True, errors="replace", timeout=300)
    assert r_384.returncode == 0, r_384.stdout + r_384.stderr
    assert re.search(r"context\s+384\b.*batch 256\b", r_384.stdout), r_384.stdout
    r_128 = subprocess.run([str(runner_bin), "-m", str(good), "--info", "fit", "-c", "128", "--gpu", "off"],
                           capture_output=True, text=True, errors="replace", timeout=300)
    assert r_128.returncode == 0, r_128.stdout + r_128.stderr
    assert re.search(r"context\s+128\b.*batch 128\b", r_128.stdout), r_128.stdout

    prompt = "abcdefg " * 35  # 280 ASCII bytes -> 280 tokens in byte-level vocab, > 256 chunk boundary
    r_256 = subprocess.run([str(runner_bin), "-m", str(good), "--gpu", "off", "-t", "2", "-c", "384",
                            "-b", "256", "--score", "1", "-p", prompt],
                           capture_output=True, text=True, errors="replace", timeout=600)
    assert r_256.returncode == 0, r_256.stderr[-2000:]
    r_1 = subprocess.run([str(runner_bin), "-m", str(good), "--gpu", "off", "-t", "2", "-c", "384",
                          "-b", "1", "--score", "1", "-p", prompt],
                         capture_output=True, text=True, errors="replace", timeout=600)
    assert r_1.returncode == 0, r_1.stderr[-2000:]
    j_256 = json.loads(r_256.stdout)
    j_1 = json.loads(r_1.stdout)
    assert len(j_256["logprobs"]) > 256, len(j_256["logprobs"])
    assert j_256["logprobs"] == j_1["logprobs"]
    assert j_256["top1"] == j_1["top1"]
    assert j_256["top"] == j_1["top"]
