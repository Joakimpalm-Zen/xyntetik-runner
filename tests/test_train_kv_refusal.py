"""Training refuses every KV cache its backward does not read.

The training backward reads K and V rows as f16. --kv q8 was refused, but --kv k8v4 and --kv fp4 reached the backward,
and their blocks were read as f16: on Qwen3-0.6B (main 5dba42b) both failed at step 1 with "the loss or a gradient is
not finite", where a refusal was due. The fixture is --wide, with head sizes every cache type admits, so each type is
really taken (on a narrow fixture q8 and k8v4 quietly stay f16 and the check would pass for the wrong reason).
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kvref")
    out = tmp / "wide.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out), "--wide"],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    data = tmp / "train.jsonl"
    data.write_text('{"prompt": "hello there", "completion": " general kenobi"}\n', encoding="utf-8")
    return out, data


@pytest.mark.parametrize("kv", ["q8", "k8v4", "fp4"])
def test_training_refuses_a_kv_cache_its_backward_does_not_read(runner_bin, tmp_path, wide, kv):
    model, data = wide
    p = subprocess.run([str(runner_bin), "-m", str(model), "--train", str(data), "--train-steps", "1",
                        "--kv", kv, "--train-out", str(tmp_path / "a.gguf"), "--no-tray", "--gpu", "off", "-t", "2"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode != 0
    assert f"training does not cover this model yet: {kv} KV cache (use --kv f16)" in p.stderr, p.stderr[-800:]
    assert not (tmp_path / "a.gguf").exists()


def test_the_f16_cache_still_trains(runner_bin, tmp_path, wide):
    model, data = wide
    p = subprocess.run([str(runner_bin), "-m", str(model), "--train", str(data), "--train-steps", "1",
                        "--train-out", str(tmp_path / "a.gguf"), "--no-tray", "--gpu", "off", "-t", "2"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    assert (tmp_path / "a.gguf").exists()
