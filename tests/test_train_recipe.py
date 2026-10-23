"""R8.17: a distillation pipeline's recipe in --train.

The lr schedule (linear warmup, then constant or a cosine to a floor),
batching (N rows a step, the mean over their positions), the weight-decay
switch, checkpoints that resume a run exactly, held-out evals, the kill rule,
the stop file, per-step lr and gradient norms, and the refusal of a non-finite
step. The headline gate is resume: a run stopped and continued writes the same
adapter, byte for byte, as the run that never stopped.
"""
import json
import math
import os
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = "the runner trains the adapter and the adapter learns the corpus. " * 3
LINES = [
    {"prompt_tokens": [1, 5, 9], "completion_tokens": [11, 17, 23, 29, 31]},
    {"prompt_tokens": [1, 7], "completion_tokens": [13, 19, 21, 25]},
    {"prompt_tokens": [1, 4], "completion_tokens": [33, 35, 37, 41, 43]},
]
POS = 5


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def write_npy(path, descr, shape, payload):
    dims = ", ".join(str(s) for s in shape)
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (descr, dims)
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode() + payload)


def write_teacher(d, lines, k=4, mass=0.8):
    d.mkdir(parents=True, exist_ok=True)
    ids, lps = [], []
    for l in lines:
        for p in range(POS):
            if p < len(l["completion_tokens"]):
                t = l["completion_tokens"][p]
                row = [t] + [3 + (t * (j + 3)) % 250 for j in range(k - 1)]
                w = [2.0 ** -j for j in range(k)]
                lp = [math.log(mass * x / sum(w)) for x in w]
            else:
                row, lp = [0] * k, [0.0] * k
            ids += row
            lps += lp
    shape = (len(lines), POS, k)
    write_npy(d / "topk_idx.npy", "<i4", shape, struct.pack("<%di" % len(ids), *ids))
    write_npy(d / "topk_logp.npy", "<f4", shape, struct.pack("<%df" % len(lps), *lps))
    return d


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("recipe")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(d / "base.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    (d / "corpus.txt").write_text(CORPUS)
    (d / "rows.jsonl").write_text("".join(json.dumps(l) + "\n" for l in LINES))
    (d / "dev.jsonl").write_text("".join(json.dumps(l) + "\n" for l in LINES[:2]))
    write_teacher(d / "teacher", LINES)
    write_teacher(d / "dev-teacher", LINES[:2])
    return d


def run(runner_bin, fx, out, *extra, data="rows.jsonl", teacher=True, steps=6, env=None):
    cmd = [str(runner_bin), "-m", str(fx / "base.gguf"), "--train", str(fx / data),
           "--train-steps", str(steps), "--lr", "3e-3", "--train-out", str(out),
           "-t", "2", "--gpu", "off", *extra]
    if teacher:
        cmd += ["--train-teacher", str(fx / "teacher")]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=600, env=dict(os.environ, **(env or {})))
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    return p, [l for l in lines if "loss" in l and "step" in l], [l for l in lines if "eval" in l]


def ok(p):
    assert p.returncode == 0, p.stderr.decode(errors="replace")


def test_schedule_is_the_pipelines_warmup_and_cosine(runner_bin, fx, tmp_path):
    steps, warm, floor = 12, 3, 0.1
    p, lines, _ = run(runner_bin, fx, tmp_path / "a.gguf", "--lr-schedule", "cosine",
                      "--lr-warmup", str(warm), steps=steps)
    ok(p)
    for s, l in enumerate(lines):
        if s < warm:
            want = 3e-3 * (s + 1) / warm
        else:
            prog = (s - warm) / max(1, steps - warm)
            want = 3e-3 * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))
        assert l["lr"] == pytest.approx(want, rel=1e-5), (s, l["lr"], want)
    rec = json.loads((tmp_path / "a.gguf.train.json").read_text())
    assert rec["recipe"] == {"schedule": "cosine", "warmup": warm, "floor": 0.1,
                             "batch": 1, "loss": "row-sum"}


def test_a_batch_of_one_row_twice_is_that_row(runner_bin, fx, tmp_path):
    """--train-batch is the mean over the batch's positions: a batch holding the
    same row twice has that row's mean gradient, so its step equals the step of
    --train-batch 1 on the row alone (to rounding: the two gradients are summed
    in another order)."""
    (tmp_path / "base.gguf").write_bytes((fx / "base.gguf").read_bytes())
    (tmp_path / "one.jsonl").write_text(json.dumps(LINES[0]) + "\n")
    (tmp_path / "two.jsonl").write_text((json.dumps(LINES[0]) + "\n") * 2)
    write_teacher(tmp_path / "t1", LINES[:1])
    write_teacher(tmp_path / "t2", [LINES[0], LINES[0]])
    outs = {}
    for name, data, b, t in (("one", "one.jsonl", "1", "t1"), ("two", "two.jsonl", "2", "t2")):
        p = subprocess.run([str(runner_bin), "-m", str(tmp_path / "base.gguf"), "--train",
                            str(tmp_path / data), "--train-teacher", str(tmp_path / t),
                            "--train-steps", "1", "--lr", "1e-3", "--train-batch", b,
                            "--train-out", str(tmp_path / f"{name}.gguf"), "-t", "2",
                            "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=600)
        ok(p)
        outs[name] = json.loads(p.stdout.splitlines()[0])
    assert outs["two"]["loss"] == pytest.approx(outs["one"]["loss"], rel=1e-6)
    assert outs["two"]["grad_norm_max"] == pytest.approx(outs["one"]["grad_norm_max"], rel=1e-5)
    assert outs["two"]["rows"] == [0, 1]


def test_resume_is_byte_identical_to_the_uninterrupted_run(runner_bin, fx, tmp_path):
    common = ("--train-batch", "2", "--lr-schedule", "cosine", "--lr-warmup", "2",
              "--weight-decay", "0", "--save-every", "3")
    p, straight, _ = run(runner_bin, fx, tmp_path / "s.gguf", *common, steps=8)
    ok(p)
    assert (tmp_path / "s.gguf.step-00003").exists() and (tmp_path / "s.gguf.step-00003.optim").exists()
    p, first, _ = run(runner_bin, fx, tmp_path / "r.gguf", *common, steps=3)
    ok(p)
    p, rest, _ = run(runner_bin, fx, tmp_path / "r2.gguf", *common,
                     "--train-resume", str(tmp_path / "r.gguf.step-00003"), steps=8)
    ok(p)
    assert b"resuming" in p.stderr
    assert (tmp_path / "s.gguf").read_bytes() == (tmp_path / "r2.gguf").read_bytes()
    key = lambda l: (l["step"], l["loss"], l["lr"], l["rows"], l["grad_norm_max"])
    assert [key(l) for l in straight] == [key(l) for l in first + rest]
    rec = json.loads((tmp_path / "r2.gguf.train.json").read_text())
    assert rec["resumed_at_step"] == 3 and rec["adamw"]["weight_decay"] == 0


def test_resume_refuses_moments_of_another_adapter(runner_bin, fx, tmp_path):
    p, _, _ = run(runner_bin, fx, tmp_path / "a.gguf", "--save-every", "2", steps=2)
    ok(p)
    p, _, _ = run(runner_bin, fx, tmp_path / "b.gguf", "--save-every", "2", "--lr-warmup", "2",
                  steps=2)
    ok(p)
    assert (tmp_path / "a.gguf").read_bytes() != (tmp_path / "b.gguf").read_bytes()
    (tmp_path / "a.gguf.optim").write_bytes((tmp_path / "b.gguf.optim").read_bytes())
    p, _, _ = run(runner_bin, fx, tmp_path / "c.gguf", "--train-resume",
                  str(tmp_path / "a.gguf"), steps=4)
    assert p.returncode != 0 and b"another adapter file" in p.stderr


def test_eval_at_step_zero_is_the_no_update_loss(runner_bin, fx, tmp_path):
    spec = "dev=%s,%s" % (fx / "dev.jsonl", fx / "dev-teacher")
    p, lines, evals = run(runner_bin, fx, tmp_path / "a.gguf", "--train-eval", spec,
                          "--eval-every", "2", steps=4)
    ok(p)
    assert [e["step"] for e in evals] == [0, 2, 4]
    dev0 = evals[0]["eval"]["dev"]
    assert dev0["rows"] == 2 and dev0["positions"] == 9
    assert 0 <= dev0["top1_topk"] <= 1 and 0 <= dev0["top1_vocab"] <= 1
    # the same two rows trained at lr 0: their per-row means, weighted by length
    (tmp_path / "base.gguf").write_bytes((fx / "base.gguf").read_bytes())
    q = subprocess.run([str(runner_bin), "-m", str(tmp_path / "base.gguf"), "--train",
                        str(fx / "dev.jsonl"), "--train-teacher", str(fx / "dev-teacher"),
                        "--train-steps", "2", "--lr", "0", "--train-out",
                        str(tmp_path / "z.gguf"), "-t", "2", "--gpu", "off"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    ok(q)
    zl = [json.loads(l)["loss"] for l in q.stdout.splitlines()]
    assert dev0["loss"] == pytest.approx((zl[0] * 5 + zl[1] * 4) / 9, rel=1e-5)
    rec = json.loads((tmp_path / "a.gguf.train.json").read_text())
    assert rec["evals"][0]["label"] == "dev" and len(rec["evals"][0]["topk_idx"]["sha256"]) == 64


@pytest.mark.parametrize("min_rel,code", [(0.99, 3), (-1.0, 0)])
def test_kill_rule(runner_bin, fx, tmp_path, min_rel, code):
    spec = "dev=%s,%s" % (fx / "dev.jsonl", fx / "dev-teacher")
    p, lines, evals = run(runner_bin, fx, tmp_path / "a.gguf", "--train-eval", spec,
                          "--kill-at", "2", "--kill-metric", "dev", "--kill-min-rel",
                          str(min_rel), steps=4)
    assert p.returncode == code, p.stderr.decode(errors="replace")
    if code == 3:
        v = json.loads((tmp_path / "a.gguf.killed.json").read_text())
        assert v["step"] == 2 and v["required"] == pytest.approx(0.99)
        assert v["baseline"] == pytest.approx(evals[0]["eval"]["dev"]["loss"], rel=1e-5)
        assert (tmp_path / "a.gguf.step-00002").exists() and len(lines) == 2
    else:
        assert (tmp_path / "a.gguf.kill-check.json").exists() and len(lines) == 4


def test_stop_file_yields_and_the_run_resumes(runner_bin, fx, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("")
    p, lines, _ = run(runner_bin, fx, tmp_path / "y.gguf", "--stop-file", str(stop), steps=5)
    assert p.returncode == 75, p.stderr.decode(errors="replace")
    assert len(lines) == 1 and (tmp_path / "y.gguf.step-00001.optim").exists()
    stop.unlink()
    p, rest, _ = run(runner_bin, fx, tmp_path / "y2.gguf", "--train-resume",
                     str(tmp_path / "y.gguf.step-00001"), "--stop-file", str(stop), steps=5)
    ok(p)
    p, straight, _ = run(runner_bin, fx, tmp_path / "n.gguf", steps=5)
    ok(p)
    assert (tmp_path / "n.gguf").read_bytes() == (tmp_path / "y2.gguf").read_bytes()


def test_a_non_finite_step_is_refused(runner_bin, fx, tmp_path):
    (tmp_path / "base.gguf").write_bytes((fx / "base.gguf").read_bytes())
    (tmp_path / "w.jsonl").write_text(json.dumps({"prompt": "the runner",
                                                  "completion": " trains the adapter",
                                                  "weight": 3e38}) + "\n")
    p = subprocess.run([str(runner_bin), "-m", str(tmp_path / "base.gguf"), "--train",
                        str(tmp_path / "w.jsonl"), "--train-steps", "2", "--train-out",
                        str(tmp_path / "x.gguf"), "-t", "2", "--gpu", "off"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    assert p.returncode == 1
    assert b"not finite" in p.stderr
    assert not (tmp_path / "x.gguf").exists()


@pytest.mark.parametrize("flag", [["--lr-warmup", "2"], ["--train-batch", "2"],
                                  ["--stop-file", "x"], ["--train-resume", "x.gguf"]])
def test_recipe_flags_need_train(runner_bin, fx, flag):
    p = subprocess.run([str(runner_bin), "-m", str(fx / "base.gguf"), "-p", "hi", "-n", "1",
                        *flag], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120)
    assert p.returncode != 0 and flag[0].encode() in p.stderr
