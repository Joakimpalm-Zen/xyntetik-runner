"""R17.9(c): exact unlearning, as a retrain with a receipt.

`runner --unlearn PARENT.train.json --drop D.jsonl -m BASE --train DATA1 --train-out A2` trains A2 from BASE with the
parent record's recipe on DATA2 = DATA1 with every occurrence of D's lines removed (the rest in their order), which it
writes beside A2. The record names the parent record by sha256, D by sha256, how many rows were removed and how many
distinct lines D listed, and DATA2 as its data. `runner attest unlearn A2.train.json --parent-data DATA1 --dropped D`
checks DATA1 is the parent's data, D is the dropped set, DATA1 minus D is DATA2, and then replays A2 as
`attest replay` does.

"Exact" is retraining-equivalence: A2 is byte for byte the adapter a from-scratch run with the SAME recipe gives on
DATA2. A recipe counted in steps sees another mix of rows once rows are gone, so nothing is claimed about A2 against
A1. The anchor outside the runner: the retained set is computed here, by Python, from DATA1 and D.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = [
    {"prompt": "the runner trains", "completion": " the adapter"},
    {"prompt": "the adapter learns", "completion": " the corpus"},
    {"prompt": "a secret row", "completion": " to forget"},
    {"prompt": "the base stays", "completion": " as it was"},
    {"prompt": "a secret row", "completion": " to forget"},
    {"prompt": "another private row", "completion": " also gone"},
    {"prompt": "the corpus ends", "completion": " here"},
]
DROP = [{"prompt": "a secret row", "completion": " to forget"},
        {"prompt": "another private row", "completion": " also gone"}]
RECIPE = ["--train-steps", "4", "--lr", "3e-3", "-t", "2", "--gpu", "off"]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _line(r):
    return json.dumps(r)


def _write_rows(path, rows):
    path.write_text("".join(_line(r) + "\n" for r in rows), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_") and not k.startswith("XR_")}


def _run(exe, args):
    return subprocess.run([str(exe), *map(str, args)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=600, env=_env())


@pytest.fixture
def d(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(tmp_path / "base.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    _write_rows(tmp_path / "data1.jsonl", ROWS)
    _write_rows(tmp_path / "drop.jsonl", DROP)
    return tmp_path


def _parent(runner_bin, d):
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "data1.jsonl", "--train-out", d / "a1.gguf", *RECIPE])
    assert p.returncode == 0, p.stderr
    return d / "a1.gguf.train.json"


def _same_binary(rec):
    return [] if json.loads(pathlib.Path(rec).read_text())["determinism"] == "cpu-t3-exact" else ["--allow-same-binary"]


def test_unlearn_is_the_from_scratch_run_on_the_retained_rows(runner_bin, d):
    parent = _parent(runner_bin, d)
    p = _run(runner_bin, ["--unlearn", parent, "--drop", d / "drop.jsonl", "-m", d / "base.gguf",
                          "--train", d / "data1.jsonl", "--train-out", d / "a2.gguf"])
    assert p.returncode == 0, p.stdout + p.stderr

    # the retained set, computed here: every occurrence of D's lines removed, the rest in order
    dropped = {_line(r) for r in DROP}
    kept = [r for r in ROWS if _line(r) not in dropped]
    _write_rows(d / "expected.jsonl", kept)
    assert (d / "a2.gguf.retained.jsonl").read_bytes() == (d / "expected.jsonl").read_bytes()

    rec = json.loads((d / "a2.gguf.train.json").read_text())
    un = rec.get("unlearn")
    assert un, rec.keys()
    assert un["parent_record"]["sha256"] == _sha(parent), un
    assert un["dropped_sha256"] == _sha(d / "drop.jsonl"), un
    assert un["dropped_count"] == 3 and un["dropped_distinct"] == 2, un   # one listed line took out two rows
    assert rec["data"]["sha256"] == _sha(d / "expected.jsonl"), rec["data"]

    # retraining-equivalence: the same recipe from scratch on the retained rows gives the same bytes
    p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "expected.jsonl", "--train-out", d / "scratch.gguf",
                          *RECIPE])
    assert p.returncode == 0, p.stderr
    assert (d / "a2.gguf").read_bytes() == (d / "scratch.gguf").read_bytes(), "unlearn is not the from-scratch run"

    a = _run(runner_bin, ["attest", "unlearn", d / "a2.gguf.train.json", "--parent-data", d / "data1.jsonl",
                          "--dropped", d / "drop.jsonl", *_same_binary(d / "a2.gguf.train.json")])
    assert a.returncode == 0 and f"VERIFIED a2.gguf sha256 {_sha(d / 'a2.gguf')}" in a.stdout, a.stdout + a.stderr


def test_attest_unlearn_refuses_another_parent_data_or_dropped_set(runner_bin, d):
    parent = _parent(runner_bin, d)
    p = _run(runner_bin, ["--unlearn", parent, "--drop", d / "drop.jsonl", "-m", d / "base.gguf",
                          "--train", d / "data1.jsonl", "--train-out", d / "a2.gguf"])
    assert p.returncode == 0, p.stdout + p.stderr
    rec, flags = d / "a2.gguf.train.json", _same_binary(d / "a2.gguf.train.json")
    # one line fewer in D: not the set that was dropped
    _write_rows(d / "fewer.jsonl", DROP[:1])
    r = _run(runner_bin, ["attest", "unlearn", rec, "--parent-data", d / "data1.jsonl", "--dropped", d / "fewer.jsonl",
                          *flags])
    assert r.returncode == 2 and "MISMATCH unlearn.dropped_sha256" in r.stdout, r.stdout + r.stderr
    assert "train:" not in r.stderr
    # another parent data
    _write_rows(d / "other.jsonl", ROWS[:-1])
    r = _run(runner_bin, ["attest", "unlearn", rec, "--parent-data", d / "other.jsonl", "--dropped", d / "drop.jsonl",
                          *flags])
    assert r.returncode == 2 and "MISMATCH unlearn.parent_data.sha256" in r.stdout, r.stdout + r.stderr
    # the retained file changed beside the record, its hash edited to match: DATA1 minus D is not it
    kept = (d / "a2.gguf.retained.jsonl").read_text().splitlines()
    (d / "a2.gguf.retained.jsonl").write_text("\n".join(kept[:-1]) + "\n", encoding="utf-8")
    raw = json.loads(rec.read_text())
    raw["data"]["sha256"] = _sha(d / "a2.gguf.retained.jsonl")
    rec.write_text(json.dumps(raw))
    r = _run(runner_bin, ["attest", "unlearn", rec, "--parent-data", d / "data1.jsonl", "--dropped", d / "drop.jsonl",
                          *flags])
    assert r.returncode == 2 and "MISMATCH unlearn.retained" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("case", ["text-corpus", "not-in-parent", "not-the-parent-data"])
def test_unlearn_refuses_what_it_cannot_do_exactly(runner_bin, d, case):
    if case == "text-corpus":
        (d / "corpus.txt").write_text("the runner trains the adapter. " * 8, encoding="utf-8")
        p = _run(runner_bin, ["-m", d / "base.gguf", "--train", d / "corpus.txt", "--train-out", d / "t1.gguf", *RECIPE])
        assert p.returncode == 0, p.stderr
        args = ["--unlearn", d / "t1.gguf.train.json", "--drop", d / "drop.jsonl", "--train", d / "corpus.txt"]
        want = "row"
    elif case == "not-in-parent":
        parent = _parent(runner_bin, d)
        _write_rows(d / "absent.jsonl", [{"prompt": "never trained on", "completion": " at all"}])
        args = ["--unlearn", parent, "--drop", d / "absent.jsonl", "--train", d / "data1.jsonl"]
        want = "not in the parent's data"
    else:
        parent = _parent(runner_bin, d)
        _write_rows(d / "other.jsonl", ROWS[:-1])
        args = ["--unlearn", parent, "--drop", d / "drop.jsonl", "--train", d / "other.jsonl"]
        want = "not the parent's data"
    p = _run(runner_bin, [*args, "-m", d / "base.gguf", "--train-out", d / "a2.gguf"])
    assert p.returncode != 0 and want in p.stderr, (case, p.returncode, p.stderr[-400:])
    assert not (d / "a2.gguf").exists(), case
