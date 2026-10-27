"""--train-teacher: distillation, KL to a teacher's top-k distribution (R8.15).

The objective's arithmetic is gated in tests/test_kd_grad.c (one-hot equals
the cross-entropy bit for bit, an independent loss formula, central
differences). This file gates the command line around it: the teacher's .npy
pair is read the way a distillation pipeline writes it, a one-hot teacher
trains the same adapter bytes as plain --train, a soft teacher trains
deterministically and lowers its loss, the provenance record names the
teacher, and every malformed teacher is refused before the first step.
"""
import json
import math
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# two id-form lines with completions of different lengths: the teacher's
# positions axis is the longer one, the shorter line's tail is padding
LINES = [
    {"prompt_tokens": [1, 5, 9], "completion_tokens": [11, 17, 23, 29, 31]},
    {"prompt_tokens": [1, 7], "completion_tokens": [13, 19, 21, 25]},
]
POS = 5


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("teacher")
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(d / "base.gguf")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    write_lines(d / "data.jsonl", LINES)
    return d


def write_lines(path, lines):
    path.write_text("".join(json.dumps(l) + "\n" for l in lines))


def write_npy(path, descr, shape, payload, fortran=False, version=1):
    dims = ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "")
    hdr = "{'descr': '%s', 'fortran_order': %s, 'shape': (%s), }" % (
        descr, "True" if fortran else "False", dims)
    pre = 10 if version == 1 else 12
    hdr += " " * ((64 - (pre + len(hdr) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY" + bytes([version, 0]))
        f.write(struct.pack("<H" if version == 1 else "<I", len(hdr)))
        f.write(hdr.encode("ascii"))
        f.write(payload)


def write_teacher(d, rows, k, version=1):
    """rows[n][p] = list of (id, logprob); padding positions are zeros."""
    d.mkdir(parents=True, exist_ok=True)
    ids, lps = [], []
    for n in range(len(rows)):
        for p in range(POS):
            row = rows[n][p] if p < len(rows[n]) else [(0, 0.0)] * k
            ids += [i for i, _ in row]
            lps += [l for _, l in row]
    shape = (len(rows), POS, k)
    write_npy(d / "topk_idx.npy", "<i4", shape, struct.pack("<%di" % len(ids), *ids),
              version=version)
    write_npy(d / "topk_logp.npy", "<f4", shape, struct.pack("<%df" % len(lps), *lps),
              version=version)
    return d


def one_hot_rows(lines):
    return [[[(t, 0.0)] for t in l["completion_tokens"]] for l in lines]


def soft_rows(lines, k=4, mass=0.8):
    """The completion token first, k-1 neighbours, `mass` spread unevenly."""
    rows = []
    for l in lines:
        r = []
        for t in l["completion_tokens"]:
            ids = [t] + [3 + (t * (j + 3)) % 250 for j in range(k - 1)]
            assert len(set(ids)) == k
            w = [2.0 ** -j for j in range(k)]
            r.append([(i, math.log(mass * x / sum(w))) for i, x in zip(ids, w)])
        rows.append(r)
    return rows


def train(runner_bin, base_dir, out, teacher=None, data="data.jsonl", steps=4,
          extra=()):
    cmd = [runner_bin, "-m", str(base_dir / "base.gguf"), "--train",
           str(base_dir / data), "--train-steps", str(steps), "--lr", "3e-3",
           "--train-out", str(out), "-t", "2", *extra]
    if teacher is not None:
        cmd += ["--train-teacher", str(teacher)]
    return subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=600)


def losses(p):
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return [json.loads(l)["loss"] for l in p.stdout.splitlines() if l]


def test_one_hot_teacher_trains_the_cross_entropy_adapter(runner_bin, base, tmp_path):
    ce, kd = tmp_path / "ce.gguf", tmp_path / "kd.gguf"
    t = write_teacher(tmp_path / "onehot", one_hot_rows(LINES), 1)
    l_ce = losses(train(runner_bin, base, ce))
    l_kd = losses(train(runner_bin, base, kd, teacher=t))
    assert l_ce == l_kd
    assert ce.read_bytes() == kd.read_bytes(), \
        "a one-hot teacher must train the cross-entropy's adapter, byte for byte"
    rec = json.loads((tmp_path / "kd.gguf.train.json").read_text())
    assert rec["objective"] == "kl-to-teacher"
    assert rec["teacher"]["k"] == 1 and rec["teacher"]["positions"] == POS
    for f in ("topk_idx", "topk_logp"):
        assert len(rec["teacher"][f]["sha256"]) == 64
        assert rec["teacher"][f]["path"].endswith(f + ".npy")
    ce_rec = json.loads((tmp_path / "ce.gguf.train.json").read_text())
    assert ce_rec["objective"] == "weighted-cross-entropy" and "teacher" not in ce_rec


def test_soft_teacher_is_deterministic_and_lowers_its_loss(runner_bin, base, tmp_path):
    t = write_teacher(tmp_path / "soft", soft_rows(LINES), 4)
    a1, a2 = tmp_path / "a1.gguf", tmp_path / "a2.gguf"
    l1 = losses(train(runner_bin, base, a1, teacher=t, steps=30))
    l2 = losses(train(runner_bin, base, a2, teacher=t, steps=30))
    assert l1 == l2 and a1.read_bytes() == a2.read_bytes()
    # one line per step, alternating examples: compare each example with itself
    assert l1[-2] < l1[0] and l1[-1] < l1[1], l1
    # the soft target is not the one-hot one
    ce = tmp_path / "ce.gguf"
    losses(train(runner_bin, base, ce, steps=30))
    assert ce.read_bytes() != a1.read_bytes()


def test_npy_format_version_2_reads_the_same(runner_bin, base, tmp_path):
    v1 = write_teacher(tmp_path / "v1", soft_rows(LINES), 4)
    v2 = write_teacher(tmp_path / "v2", soft_rows(LINES), 4, version=2)
    a1, a2 = tmp_path / "a1.gguf", tmp_path / "a2.gguf"
    assert losses(train(runner_bin, base, a1, teacher=v1)) == \
        losses(train(runner_bin, base, a2, teacher=v2))
    assert a1.read_bytes() == a2.read_bytes()


def _corrupt(d, name, fn):
    p = d / name
    p.write_bytes(fn(p.read_bytes()))


def _soft(tmp_path, mutate=None, k=4, lines=LINES):
    rows = soft_rows(lines, k=k)
    if mutate:
        mutate(rows)
    return write_teacher(tmp_path / "t", rows, k)


def _set(rows, n, p, c, value):
    rows[n][p][c] = value


REFUSALS = {
    "duplicate id": (lambda tp: _soft(tp, lambda r: _set(r, 0, 2, 1, (r[0][2][0][0], -2.0))),
                     "the same id twice"),
    "id outside the vocabulary": (lambda tp: _soft(tp, lambda r: _set(r, 1, 0, 3, (100000, -3.0))),
                                  "an id outside the vocabulary"),
    "mass past one": (lambda tp: _soft(tp, lambda r: _set(r, 0, 0, 0, (r[0][0][0][0], -0.01))),
                      "summing past one"),
    "not finite": (lambda tp: _soft(tp, lambda r: _set(r, 0, 1, 2, (r[0][1][2][0], float("nan")))),
                   "not finite"),
    "wrong line count": (lambda tp: _soft(tp, lines=LINES + LINES[:1]), "has 3 lines"),
    "k past the vocabulary": (lambda tp: write_teacher(
        tp / "t", [[[(0, -6.0)] * 300] * 5] * 2, 300), "k must be"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_bad_teacher_rows_are_refused_before_training(runner_bin, base, tmp_path, case):
    make, needle = REFUSALS[case]
    t = make(tmp_path)
    p = train(runner_bin, base, tmp_path / "x.gguf", teacher=t)
    assert p.returncode != 0
    assert needle in p.stderr.decode(errors="replace")
    assert not (tmp_path / "x.gguf").exists()


FILE_REFUSALS = {
    "int64 ids": (lambda d: write_npy(d / "topk_idx.npy", "<i8", (2, POS, 4),
                                      b"\0" * (2 * POS * 4 * 8)), "must be <i4"),
    "fortran order": (lambda d: write_npy(d / "topk_logp.npy", "<f4", (2, POS, 4),
                                          b"\0" * (2 * POS * 4 * 4), fortran=True), "C order"),
    "truncated": (lambda d: _corrupt(d, "topk_logp.npy", lambda b: b[:-4]), "does not match"),
    "not npy": (lambda d: _corrupt(d, "topk_idx.npy", lambda b: b"PK" + b[2:]), "not a .npy"),
    "shapes differ": (lambda d: write_npy(d / "topk_logp.npy", "<f4", (2, POS, 3),
                                          b"\0" * (2 * POS * 3 * 4)), "share one"),
    "missing": (lambda d: (d / "topk_logp.npy").unlink(), "cannot open"),
}


@pytest.mark.parametrize("case", sorted(FILE_REFUSALS))
def test_bad_teacher_files_are_refused(runner_bin, base, tmp_path, case):
    t = _soft(tmp_path)
    mutate, needle = FILE_REFUSALS[case]
    mutate(t)
    p = train(runner_bin, base, tmp_path / "x.gguf", teacher=t)
    assert p.returncode != 0
    assert needle in p.stderr.decode(errors="replace")


def test_teacher_needs_positions_for_every_completion_token(runner_bin, base, tmp_path):
    longer = [dict(LINES[0], completion_tokens=LINES[0]["completion_tokens"] + [37]),
              LINES[1]]
    write_lines(tmp_path / "long.jsonl", longer)
    base_copy = tmp_path
    (base_copy / "base.gguf").write_bytes((base / "base.gguf").read_bytes())
    t = _soft(tmp_path)
    p = train(runner_bin, base_copy, tmp_path / "x.gguf", teacher=t, data="long.jsonl")
    assert p.returncode != 0
    assert "positions per line" in p.stderr.decode(errors="replace")


@pytest.mark.parametrize("line,needle", [
    ({"prompt": "the runner", "completion": " trains"}, "\"prompt_tokens\""),
    (dict(LINES[0], end_of_turn=True), "no end-of-turn token"),
])
def test_teacher_needs_the_id_form_as_given(runner_bin, base, tmp_path, line, needle):
    (tmp_path / "base.gguf").write_bytes((base / "base.gguf").read_bytes())
    write_lines(tmp_path / "d.jsonl", [line, LINES[1]])
    t = _soft(tmp_path)
    p = train(runner_bin, tmp_path, tmp_path / "x.gguf", teacher=t, data="d.jsonl")
    assert p.returncode != 0
    assert needle in p.stderr.decode(errors="replace")


def test_teacher_without_train_is_refused(runner_bin, base, tmp_path):
    t = _soft(tmp_path)
    p = subprocess.run([runner_bin, "-m", str(base / "base.gguf"), "--train-teacher",
                        str(t), "-p", "hi", "-n", "1"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0
    assert b"pass it with --train" in p.stderr


def test_gpu_assisted_teacher_training_matches_cpu_bytes(runner_bin, base, tmp_path):
    """RUNNER_TRAIN_GPU=1 runs the head's transposed matvec on the device; with
    a teacher the matvec carries softmax minus the teacher's target instead of
    minus a one-hot, and the adapter bytes must still be the CPU's. Skips
    unless a CUDA device engaged (scripts/cuda-gate.sh runs it on one)."""
    import os
    t = write_teacher(tmp_path / "soft", soft_rows(LINES), 4)
    a_cpu, a_gpu = tmp_path / "cpu.gguf", tmp_path / "gpu.gguf"
    losses(train(runner_bin, base, a_cpu, teacher=t, steps=6))
    cmd = [runner_bin, "-m", str(base / "base.gguf"), "--train",
           str(base / "data.jsonl"), "--train-steps", "6", "--lr", "3e-3",
           "--train-out", str(a_gpu), "-t", "2", "--train-teacher", str(t)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=600, env=dict(os.environ, RUNNER_TRAIN_GPU="1"))
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    if b"train-gpu:" not in p.stderr or b"no usable CUDA" in p.stderr:
        pytest.skip("no CUDA device engaged")
    assert a_cpu.read_bytes() == a_gpu.read_bytes(), \
        "GPU-assisted teacher training changed the adapter bytes"


def test_kd_teacher_script_builds_a_teacher_the_trainer_reads(runner_bin, base, tmp_path):
    """scripts/kd-teacher.py: a parent GGUF scored by `--score --score-top K`
    into the id-form --train file and the .npy pair. The token ids are the
    parent's own scoring of the text, and the rows its top-K, best first."""
    docs = tmp_path / "docs.jsonl"
    docs.write_text(json.dumps({"text": "the runner trains the adapter"}) + "\n" +
                    json.dumps({"text": "and the adapter learns the corpus well"}) + "\n")
    out = tmp_path / "t"
    p = subprocess.run([sys.executable, ROOT / "scripts/kd-teacher.py", "--runner",
                        str(runner_bin), "--parent", str(base / "base.gguf"),
                        "--docs", str(docs), "--topk", "8", "--out", str(out),
                        "--", "--gpu", "off"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    summary = json.loads(p.stdout)
    assert summary["documents"] == 2 and summary["topk"] == 8
    lines = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    s = json.loads(subprocess.run(
        [runner_bin, "-m", str(base / "base.gguf"), "--score", "--score-top", "8",
         "-p", "the runner trains the adapter", "--gpu", "off"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120).stdout)
    assert lines[0]["prompt_tokens"] + lines[0]["completion_tokens"] == s["tokens"]
    raw = (out / "topk_idx.npy").read_bytes()
    hlen = struct.unpack("<H", raw[8:10])[0]
    first = struct.unpack("<8i", raw[10 + hlen:10 + hlen + 32])
    assert list(first) == [t for t, _ in s["top"][0]]
    (out / "base.gguf").write_bytes((base / "base.gguf").read_bytes())
    l = losses(train(runner_bin, out, tmp_path / "kd.gguf", teacher=out,
                     data="train.jsonl"))
    assert len(l) == 4 and all(x >= 0 for x in l)
