"""--train: the D4 loop under the D5 gates.

The headline property is DETERMINISTIC TRAINING: same data + same seed ->
byte-identical adapter GGUF, twice over. Everything else is the loop doing
its job: loss falls on an overfit corpus, the saved adapter loads back
through --lora and measurably improves --score on the trained text, the
JSONL prompt/completion mode masks the prompt from the loss, and a different
seed produces a different adapter (the determinism is seeded, not vacuous).
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ("the runner trains the adapter and the adapter learns the corpus. "
          * 3)
SENT = "the runner trains the adapter and the adapter learns the corpus."


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("train")
    b = d / "base.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(b)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (d / "corpus.txt").write_text(CORPUS)
    (d / "data.jsonl").write_text(
        json.dumps({"prompt": "the runner trains",
                    "completion": " the adapter and the adapter learns",
                    "weight": 1.0}) + "\n" +
        json.dumps({"prompt": "the corpus",
                    "completion": " is learned by the adapter",
                    "weight": 0.5}) + "\n")
    return d


def _train(runner_bin, base_dir, out, data="corpus.txt", steps=20, extra=()):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"),
           "--train", str(base_dir / data), "--train-steps", str(steps),
           "--lr", "3e-3", "--train-out", str(out), "-t", "2", *extra]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    losses = [json.loads(l)["loss"] for l in p.stdout.splitlines() if l]
    return losses


def _run_train(runner_bin, base_dir, out, data, extra=()):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"),
           "--train", str(base_dir / data), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2", *extra]
    return subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=600)


def _score(runner_bin, base_dir, lora=None):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"), "--score",
           "-p", SENT, "-t", "2", "--gpu", "off"]
    if lora:
        cmd += ["--lora", str(lora)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)["nll_mean"]


def test_training_is_byte_deterministic(runner_bin, base, tmp_path):
    a1, a2 = tmp_path / "a1.gguf", tmp_path / "a2.gguf"
    l1 = _train(runner_bin, base, a1)
    l2 = _train(runner_bin, base, a2)
    assert l1 == l2, "loss trajectories differ between identical runs"
    assert a1.read_bytes() == a2.read_bytes(), \
        "adapter files differ between identical runs"
    # D7: the provenance records agree on the adapter sha (and carry the
    # base/data shas that make the reproducibility claim checkable)
    r1 = json.loads((tmp_path / "a1.gguf.train.json").read_text())
    r2 = json.loads((tmp_path / "a2.gguf.train.json").read_text())
    assert r1["adapter"]["sha256"] == r2["adapter"]["sha256"]
    assert len(r1["base"]["sha256"]) == 64 and len(r1["data"]["sha256"]) == 64
    # build identity: same executable must be distinguishable from same
    # source in any reproduction report (external rebuild study, 2026-08)
    assert len(r1["build"]["binary_sha256"]) == 64
    assert r1["build"]["compiler"] and r1["build"]["os"] and r1["build"]["arch"]


def test_loss_falls_and_adapter_improves_score(runner_bin, base, tmp_path):
    out = tmp_path / "ad.gguf"
    losses = _train(runner_bin, base, out, steps=30)
    assert losses[-1] < losses[0] * 0.95, (losses[0], losses[-1])
    base_nll = _score(runner_bin, base)
    tuned_nll = _score(runner_bin, base, lora=out)
    assert tuned_nll < base_nll - 0.2, (base_nll, tuned_nll)


def test_jsonl_mode_trains_with_prompt_masking(runner_bin, base, tmp_path):
    out = tmp_path / "aj.gguf"
    losses = _train(runner_bin, base, out, data="data.jsonl", steps=12)
    assert len(losses) == 12
    assert losses[-1] < losses[0]
    # round-trips through --lora
    _score(runner_bin, base, lora=out)


def test_provenance_record_escapes_data_and_output_paths(
        runner_bin, base, tmp_path):
    data_name = 'corpus"quoted.txt'
    (base / data_name).write_text(CORPUS)
    out = tmp_path / 'adapter"quoted.gguf'
    _train(runner_bin, base, out, data=data_name, steps=1)
    record = pathlib.Path(f"{out}.train.json")
    rec = json.loads(record.read_text())
    assert rec["data"]["path"] == str(base / data_name)
    assert rec["adapter"]["path"] == str(out)


def test_provenance_hashes_binary_when_invoked_through_path(
        runner_bin, base, tmp_path):
    out = tmp_path / "path-invoked.gguf"
    env = dict(os.environ)
    env["PATH"] = str(runner_bin.parent) + os.pathsep + env.get("PATH", "")
    cmd = [runner_bin.name, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2"]
    p = subprocess.run(cmd, cwd=tmp_path, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    rec = json.loads(pathlib.Path(f"{out}.train.json").read_text())
    assert rec["build"]["binary_sha256"] == hashlib.sha256(
        runner_bin.read_bytes()).hexdigest()


@pytest.mark.parametrize("record", [
    '{"prompt":"p","completion":"c","weight":"heavy"}\n',
    "{\n",
])
def test_jsonl_rejects_malformed_examples(runner_bin, base, tmp_path, record):
    (base / "invalid.jsonl").write_text(record)
    out = tmp_path / "invalid.gguf"
    p = _run_train(runner_bin, base, out, "invalid.jsonl")
    assert p.returncode != 0
    assert not out.exists()


def test_jsonl_honors_train_context_without_truncating(
        runner_bin, base, tmp_path):
    record = {"prompt": "the runner", "completion": " trains the adapter " * 8}
    (base / "too-long.jsonl").write_text(json.dumps(record) + "\n")
    out = tmp_path / "too-long.gguf"
    p = _run_train(runner_bin, base, out, "too-long.jsonl",
                   extra=("--train-ctx", "4"))
    assert p.returncode != 0
    assert b"--train-ctx" in p.stderr
    assert not out.exists()


def test_failed_checkpoint_install_preserves_previous_adapter(
        runner_bin, base, tmp_path):
    out = tmp_path / "checkpoint.gguf"
    sentinel = b"previous valid checkpoint"
    out.write_bytes(sentinel)
    env = dict(os.environ, RUNNER_LORA_INSTALL_FAIL="1")
    cmd = [runner_bin, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode != 0
    assert out.read_bytes() == sentinel
    assert not pathlib.Path(f"{out}.partial").exists()


def test_optimizer_allocation_failure_does_not_publish_adapter(
        runner_bin, base, tmp_path):
    out = tmp_path / "optimizer-oom.gguf"
    env = dict(os.environ, RUNNER_LORA_ADAM_ALLOC_FAIL="1")
    cmd = [runner_bin, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "1",
           "--lr", "3e-3", "--train-out", str(out), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode != 0
    assert b"optimizer state" in p.stderr
    assert not out.exists()
    assert not pathlib.Path(f"{out}.train.json").exists()


def test_gpu_training_matches_cpu_bytes(runner_bin, base, tmp_path):
    """D8 slice 2: RUNNER_TRAIN_GPU=1 must not change a single adapter byte.

    The device transposed-matvec is gated bit-identical to the CPU chain
    (test_mvt.c), so the whole training run must be too. Skips unless a
    CUDA device actually engaged (the runner prints its train-gpu banner).
    """
    import os
    a_cpu, a_gpu = tmp_path / "cpu.gguf", tmp_path / "gpu.gguf"
    _train(runner_bin, base, a_cpu, steps=6)
    env = dict(os.environ, RUNNER_TRAIN_GPU="1")
    cmd = [runner_bin, "-m", str(base / "base.gguf"),
           "--train", str(base / "corpus.txt"), "--train-steps", "6",
           "--lr", "3e-3", "--train-out", str(a_gpu), "-t", "2"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=600, env=env)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    if b"train-gpu:" not in p.stderr or b"no usable CUDA" in p.stderr:
        pytest.skip("no CUDA device engaged")
    assert a_cpu.read_bytes() == a_gpu.read_bytes(), \
        "GPU-assisted training changed the adapter bytes"


def test_seed_changes_the_adapter(runner_bin, base, tmp_path):
    a1, a2 = tmp_path / "s1.gguf", tmp_path / "s2.gguf"
    _train(runner_bin, base, a1, steps=4, extra=("-s", "7"))
    _train(runner_bin, base, a2, steps=4, extra=("-s", "8"))
    assert a1.read_bytes() != a2.read_bytes()
