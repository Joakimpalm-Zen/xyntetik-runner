"""--train on a sparse-MoE model (R17.3): an attention-only adapter.

The adapter sits on the attention projections; the routed experts and the
router stay frozen and the backward runs through them (the strict FD gate in
test_lora_grad.c pins the gradient, router term included). End to end on the
router-sensitive fixture: the loss falls, a rerun with the same seed writes a
byte-identical adapter, the adapter serves back with --lora, and a merge
scores exactly like base + adapter.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "test-moe-train.moe4-train.gguf"
TEXT = "the cat sat on the mat and the dog sat on the log.\n"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    if not FIXTURE.exists():
        pytest.skip("run `make test-moe-train.moe4-train.gguf` first")
    return exe


def _run(exe, args, cwd):
    p = subprocess.run([str(exe), *map(str, args)], cwd=cwd, capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0, (args, p.stderr[-2000:])
    return p


def _nll(exe, d, *model_args):
    p = _run(exe, [*model_args, "--score", "-p", TEXT.strip(), "--gpu", "off"], d)
    return json.loads(p.stdout)["nll_mean"]


def test_an_attention_adapter_trains_on_a_moe_model(runner_bin, tmp_path):
    shutil.copy(FIXTURE, tmp_path / "moe.gguf")
    (tmp_path / "d.txt").write_text(TEXT * 2)
    train = ["-m", "moe.gguf", "--train", "d.txt", "--train-steps", "20", "--lr", "3e-3",
             "--gpu", "off", "-s", "7"]
    _run(runner_bin, [*train, "--train-out", "a1.gguf"], tmp_path)
    _run(runner_bin, [*train, "--train-out", "a2.gguf"], tmp_path)
    sha = lambda f: hashlib.sha256((tmp_path / f).read_bytes()).hexdigest()
    assert sha("a1.gguf") == sha("a2.gguf")              # same seed, same bytes
    rec = json.loads((tmp_path / "a1.gguf.train.json").read_text())
    assert rec["loss_last"] < rec["loss_first"], rec
    base = _nll(runner_bin, tmp_path, "-m", "moe.gguf")
    served = _nll(runner_bin, tmp_path, "-m", "moe.gguf", "--lora", "a1.gguf")
    assert served < base, (served, base)
    _run(runner_bin, ["-m", "moe.gguf", "--lora", "a1.gguf", "--merge-lora", "m.gguf"], tmp_path)
    merged = _nll(runner_bin, tmp_path, "-m", "m.gguf")
    assert abs(merged - served) < 1e-4, (merged, served)


def test_an_unsupported_moe_shape_is_refused_by_name(runner_bin, tmp_path):
    # group-limited routing (DeepSeek-style) is not in the backward yet
    grp = ROOT / "test-moe-train.ggroup.gguf"
    if not grp.exists():
        pytest.skip("fixture not built")
    shutil.copy(grp, tmp_path / "g.gguf")
    (tmp_path / "d.txt").write_text(TEXT)
    p = subprocess.run([str(runner_bin), "-m", "g.gguf", "--train", "d.txt", "--train-steps",
                        "1", "--train-out", "a.gguf", "--gpu", "off"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode != 0 and "group-limited MoE routing" in p.stderr, p.stderr[-500:]
