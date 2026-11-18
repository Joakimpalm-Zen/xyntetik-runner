"""R4.41 Phase 0 gate (c): CLI `--train-scales` trainer mode.

Covers dense SiLU (`qwen3`, K-quants Q3_K & Q4_K) and `qwen35` (Gated DeltaNet, Q4_K)
with `--train-teacher`, AdamW, cosine schedule + warmup, `--train-batch`, `--train-eval`,
`--kill-at`, same-build determinism, and byte-exact preservation outside scale fields.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_teacher(runner_bin: pathlib.Path, parent: pathlib.Path, work: pathlib.Path):
    docs = work / "docs.jsonl"
    docs.write_text(
        json.dumps({"text": "The quick brown fox jumps over the lazy dog near the river bank."}) + "\n"
        + json.dumps({"text": "Scale recovery adjusts per-block fp16 multipliers while codes stay frozen."}) + "\n",
        encoding="utf-8",
    )
    tdir = work / "teacher"
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts/kd-teacher.py"),
         "--runner", str(runner_bin), "--parent", str(parent),
         "--docs", str(docs), "--topk", "64", "--out", str(tdir),
         "--", "--gpu", "off", "-c", "128"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
    )
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-1200:]
    return tdir


def run_train(runner_bin: pathlib.Path, model: pathlib.Path, tdir: pathlib.Path,
              out: pathlib.Path, *extra):
    p = subprocess.run(
        [str(runner_bin), "-m", str(model),
         "--train", str(tdir / "train.jsonl"),
         "--train-teacher", str(tdir),
         "--train-scales",
         "--lr-schedule", "cosine",
         "--lr-warmup", "1",
         "--lr-floor", "0.1",
         "--lr", "2e-3",
         "--train-batch", "2",
         "--train-ctx", "128",
         "--train-steps", "4",
         "--weight-decay", "0",
         "--train-out", str(out),
         "--no-tray", "--gpu", "off", "-t", "2",
         *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


@pytest.mark.parametrize("parent_name,quant_name", [
    ("test-scale-qwen3-src.gguf", "test-scale-qwen3-q3k.gguf"),
    ("test-scale-qwen3-src.gguf", "test-scale-qwen3-q4k.gguf"),
    ("test-scale-qwen35-src.gguf", "test-scale-qwen35-q4k.gguf"),
])
def test_scale_recovery_kl_train_determinism_and_eval(runner_bin, tmp_path, parent_name, quant_name):
    parent = ROOT / parent_name
    quant = ROOT / quant_name
    assert parent.exists() and quant.exists(), f"missing fixture {parent_name} / {quant_name}"
    tdir = make_teacher(runner_bin, parent, tmp_path)

    out1 = tmp_path / "rec1.gguf"
    out2 = tmp_path / "rec2.gguf"
    eval_spec = f"dev={tdir / 'train.jsonl'},{tdir}"

    rc1, so1, se1 = run_train(
        runner_bin, quant, tdir, out1,
        "--train-eval", eval_spec, "--eval-every", "2",
        "--kill-at", "4", "--kill-metric", "dev", "--kill-min-rel", "0.0",
    )
    assert rc1 == 0, se1[-1500:]
    assert out1.exists()

    rc2, so2, se2 = run_train(
        runner_bin, quant, tdir, out2,
        "--train-eval", eval_spec, "--eval-every", "2",
        "--kill-at", "4", "--kill-metric", "dev", "--kill-min-rel", "0.0",
    )
    assert rc2 == 0, se2[-1500:]
    assert sha256(out1) == sha256(out2), "scale-recovery training is not byte-deterministic"

    # Same size, different bytes (scales updated), valid load/score after write
    assert out1.stat().st_size == quant.stat().st_size
    assert sha256(out1) != sha256(quant)

    steps = [json.loads(l) for l in so1.splitlines() if l.startswith('{"step"') and '"grad_norm_max"' in l]
    assert len(steps) == 4
    assert steps[-1]["loss"] < steps[0]["loss"], f"KL loss did not drop: {steps}"

    rec = json.loads(pathlib.Path(str(out1) + ".train.json").read_text())
    assert rec["objective"] == "kl-to-teacher"
    assert rec["recipe"]["schedule"] == "cosine"
    assert rec["scale_recovery"]["tensors"] > 0
    assert rec["scale_recovery"]["params"] > 0
    assert rec["variant"]["sha256"] == sha256(out1)
    assert rec["evals"][0]["loss_last"] < rec["evals"][0]["loss_step0"]

    # Recovered file must serve/score cleanly in runner
    sc = subprocess.run(
        [str(runner_bin), "-m", str(out1), "--score", "--score-top", "5",
         "-p", "The quick brown fox", "--no-tray", "--gpu", "off"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    assert sc.returncode == 0, sc.stderr.decode(errors="replace")


def test_scale_recovery_kill_rule_and_refusals(runner_bin, tmp_path):
    parent = ROOT / "test-scale-qwen3-src.gguf"
    quant = ROOT / "test-scale-qwen3-q4k.gguf"
    tdir = make_teacher(runner_bin, parent, tmp_path)

    # Unrealistic kill threshold must exit 3 and leave output unwritten
    killed_out = tmp_path / "killed.gguf"
    rc, so, se = run_train(
        runner_bin, quant, tdir, killed_out,
        "--train-eval", f"dev={tdir / 'train.jsonl'},{tdir}",
        "--kill-at", "2", "--kill-metric", "dev", "--kill-min-rel", "0.99",
    )
    assert rc == 3, (rc, so[-400:], se[-400:])
    assert not killed_out.exists()
    assert pathlib.Path(str(killed_out) + ".killed.json").exists()

    # Refusals: --train-scales + --train-full, and F32-only file
    bad1 = tmp_path / "bad1.gguf"
    rc_b1, _, se_b1 = run_train(runner_bin, quant, tdir, bad1, "--train-full")
    assert rc_b1 != 0 and not bad1.exists()
    assert "--train-full" in se_b1

    bad2 = tmp_path / "bad2.gguf"
    rc_b2, _, se_b2 = run_train(runner_bin, ROOT / "test.gguf", tdir, bad2)
    assert rc_b2 != 0 and not bad2.exists()
    assert "no K-quant" in se_b2
