"""R8.25.2.2: runner train --train-full (full-weight training into a variant) and --train-kv-f32, at the CLI.

- A full-weight run refuses every KV cache its backward does not read, as LoRA training does
  (tests/test_train_kv_refusal.py).
- --train-kv-f32 is a training flag, refused outside a --train run.
- A full-weight run is deterministic (two runs, byte-identical variants), writes a variant whose bytes are the -m file
  outside the trained tensors, and names what it trained in the provenance record.
- A quantized file is refused, and the refusal names the float route (train the F32 file, then --rewrite --quant).
  Training through a Q8_0 read copy was measured no better than quantizing afterwards (R8.25.2.2, Gate B).
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def make(tmp, name, *extra):
    out = tmp / f"{name}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(out), *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return out


def train(runner_bin, model, data, out, *extra):
    p = subprocess.run([str(runner_bin), "-m", str(model), "--train", str(data), "--train-out", str(out),
                        "--no-tray", "--gpu", "off", "-t", "2", *extra],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("full") / "train.jsonl"
    d.write_text('{"prompt": "hello there", "completion": " general kenobi"}\n'
                 '{"prompt": "one two", "completion": " three four"}\n', encoding="utf-8")
    return d


@pytest.mark.parametrize("kv", ["q8", "k8v4", "fp4"])
def test_full_weight_training_refuses_a_kv_cache_its_backward_does_not_read(runner_bin, tmp_path, data, kv):
    # --wide: head sizes every cache type admits, so the type really is taken
    wide = make(tmp_path, "wide", "--wide")
    rc, _, se = train(runner_bin, wide, data, tmp_path / "f.gguf", "--train-steps", "1", "--kv", kv, "--train-full")
    assert rc != 0 and f"{kv} KV cache (use --kv f16)" in se, se[-800:]


def test_train_kv_f32_is_a_training_flag(runner_bin, tmp_path, data):
    base = make(tmp_path, "base")
    p = subprocess.run([str(runner_bin), "-m", str(base), "-p", "hi", "-n", "1", "--train-kv-f32", "--gpu", "off",
                        "--no-tray"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode != 0 and "--train-kv-f32" in p.stderr and "--train" in p.stderr
    rc, _, se = train(runner_bin, base, data, tmp_path / "k.gguf", "--train-steps", "2", "--train-kv-f32",
                      "--train-full")
    assert rc == 0, se[-1500:]
    rec = json.loads(pathlib.Path(str(tmp_path / "k.gguf") + ".train.json").read_text())
    assert rec["kv_cache"] == "f32"


def test_a_full_weight_run_is_deterministic_and_writes_a_variant(runner_bin, tmp_path, data):
    base = make(tmp_path, "base")
    outs = []
    for k in (1, 2):
        out = tmp_path / f"v{k}.gguf"
        rc, so, se = train(runner_bin, base, data, out, "--train-steps", "3", "--lr", "1e-3", "--train-full")
        assert rc == 0, se[-1500:]
        assert "train: full weights" in se
        outs.append((out, so))
    a, b = outs[0][0].read_bytes(), outs[1][0].read_bytes()
    assert a == b, "two runs on the same data and seed gave different variants"
    parent = base.read_bytes()
    assert len(a) == len(parent) and a != parent
    losses = [json.loads(l)["loss"] for l in outs[0][1].splitlines() if l.startswith('{"step"')]
    assert len(losses) == 3 and losses[-1] < losses[0]
    rec = json.loads(pathlib.Path(str(outs[0][0]) + ".train.json").read_text())
    fw = rec["full_weight"]
    assert fw["read_type"] == "f32" and fw["tensors"] > 0 and fw["params"] > 0
    assert rec["variant"]["sha256"] and "adapter" not in rec


def test_a_quantized_file_is_refused_with_the_float_route(runner_bin, tmp_path, data):
    q8 = make(tmp_path, "q8", "--quant", "q8_0")
    out = tmp_path / "vq8.gguf"
    rc, _, se = train(runner_bin, q8, data, out, "--train-steps", "1", "--train-full")
    assert rc != 0, se[-1500:]
    assert "is not f32" in se and "--rewrite --quant" in se, se[-800:]
    assert not out.exists()
    # the master flags went with the Q8 read
    rc, _, se = train(runner_bin, q8, data, out, "--train-steps", "1", "--train-full", "--train-master", str(q8))
    assert rc != 0 and "--train-master" in se, se[-800:]
