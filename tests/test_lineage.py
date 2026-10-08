"""--lineage (R17.1): the chain of provenance records from a file back to its origin.

A rewrite (--quantize), a training run (--train) and a merge (--merge-lora)
each write a record beside their output; every record names its inputs by
sha256 and links the record that produced each input; --sign-key signs the
record in place. The walker re-checks every link from the records alone:
a child's input hash must be its parent's output hash, the parent record
must be byte-for-byte the one the child recorded, a file still on disk must
match its record, and a signature must verify. Exit 0 all verified and
signed, 1 consistent but not all signed, 2 broken or no record.
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    if not (ROOT / "test.gguf").exists():
        pytest.skip("run `make test.gguf` first")
    return exe


def _run(exe, args, cwd):
    return subprocess.run([str(exe), *map(str, args)], cwd=cwd, capture_output=True,
                          text=True, timeout=300)


def _chain(exe, d, sign=True):
    """base -> train an adapter -> merge it -> quantize the merged file."""
    shutil.copy(ROOT / "test.gguf", d / "base.gguf")
    (d / "data.txt").write_text("the cat sat on the mat.\nthe dog sat on the log.\n")
    key = []
    if sign:
        assert _run(exe, ["--keygen", "key.json"], d).returncode == 0
        key = ["--sign-key", "key.json"]
    steps = [
        ["-m", "base.gguf", "--train", "data.txt", "--train-steps", "1",
         "--train-out", "a.gguf", "--gpu", "off", *key],
        ["-m", "base.gguf", "--lora", "a.gguf", "--merge-lora", "merged.gguf", *key],
        ["-m", "merged.gguf", "--quantize", "final.gguf", "--quant", "q8_0", *key],
    ]
    for s in steps:
        p = _run(exe, s, d)
        assert p.returncode == 0, (s, p.stderr)
    for rec in ("a.gguf.train.json", "merged.gguf.merge.json", "final.gguf.quant.json"):
        assert (d / rec).exists(), rec


def test_a_signed_chain_verifies_back_to_its_origins(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 0, p.stdout
    out = p.stdout
    assert out.count("VERIFIED (signed") == 3, out
    for step in ("quantize ->", "merge ->", "train ->"):
        assert step in out, out
    # the base model and the training data are where the chain starts
    assert out.count("ORIGIN") == 3, out          # base twice (merge, train), data once
    assert "RESULT: VERIFIED" in out
    # the quantize record names its input, output and building binary
    q = json.loads((tmp_path / "final.gguf.quant.json").read_text())
    assert q["schema_version"] == "xyntetik.runner.quant.v1"
    assert q["target"] == "q8_0" and len(q["build"]["binary_sha256"]) == 64
    assert q["base"]["record"]["path"].endswith("merged.gguf.merge.json")


def test_an_unsigned_chain_is_consistent_but_says_so(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path, sign=False)
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 1, p.stdout
    assert p.stdout.count("UNSIGNED") == 3 and "BROKEN" not in p.stdout


def test_a_record_edited_after_the_fact_breaks_the_chain(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    rec = tmp_path / "merged.gguf.merge.json"
    rec.write_text(rec.read_text().replace('"lora_scale":1', '"lora_scale":2'))
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 2, p.stdout
    assert "changed after the next step recorded it" in p.stdout


def test_a_swapped_file_breaks_the_chain(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    final = tmp_path / "final.gguf"
    b = bytearray(final.read_bytes())
    b[-1] ^= 1
    final.write_bytes(bytes(b))
    p = _run(runner_bin, ["--lineage", "final.gguf"], tmp_path)
    assert p.returncode == 2, p.stdout
    assert "BROKEN" in p.stdout


def test_another_trusted_key_breaks_the_chain(runner_bin, tmp_path):
    _chain(runner_bin, tmp_path)
    other = _run(runner_bin, ["--keygen", "other.json"], tmp_path)
    assert other.returncode == 0
    pub = json.loads((tmp_path / "other.json").read_text())["public_key"]
    p = _run(runner_bin, ["--lineage", "final.gguf", "--trust-key", pub], tmp_path)
    assert p.returncode == 2, p.stdout
    assert "untrusted signature" in p.stdout


def test_a_file_with_no_record_is_refused(runner_bin, tmp_path):
    shutil.copy(ROOT / "test.gguf", tmp_path / "plain.gguf")
    p = _run(runner_bin, ["--lineage", "plain.gguf"], tmp_path)
    assert p.returncode == 2 and "NO RECORD" in p.stdout, p.stdout


def test_the_records_alone_verify_after_the_files_are_gone(runner_bin, tmp_path):
    work, audit = tmp_path / "work", tmp_path / "audit"
    work.mkdir(); audit.mkdir()
    _chain(runner_bin, work)
    for rec in work.glob("*.json"):
        shutil.copy(rec, audit / rec.name)
    shutil.rmtree(work)            # the models and the original paths are gone
    p = _run(runner_bin, ["--lineage", "final.gguf.quant.json"], audit)
    assert p.returncode == 0, p.stdout
    assert p.stdout.count("VERIFIED (signed") == 3
    assert "output file not present" in p.stdout


def test_an_answer_walks_back_from_its_receipt(runner_bin, tmp_path):
    # a served or one-shot answer's receipt names the model and links the
    # record of the step that made it: the chain runs from the answer back to
    # the base model and the training data
    _chain(runner_bin, tmp_path)
    p = _run(runner_bin, ["-m", "final.gguf", "-p", "hello", "-n", "4", "--temp", "0",
                          "--gpu", "off", "--transcript", "run.json",
                          "--sign-key", "key.json"], tmp_path)
    assert p.returncode == 0, p.stderr
    rec = json.loads((tmp_path / "run.json").read_text())
    assert rec["model"]["record"]["path"].endswith("final.gguf.quant.json")
    v = _run(runner_bin, ["-m", "final.gguf", "--verify", "run.json"], tmp_path)
    assert v.returncode == 0, v.stdout + v.stderr      # the link does not disturb replay
    w = _run(runner_bin, ["--lineage", "run.json"], tmp_path)
    assert w.returncode == 0, w.stdout
    assert "lineage of receipt" in w.stdout and w.stdout.count("VERIFIED (signed") == 3


def test_a_receipt_for_a_downloaded_model_ends_at_its_origin(runner_bin, tmp_path):
    shutil.copy(ROOT / "test.gguf", tmp_path / "plain.gguf")
    assert _run(runner_bin, ["--keygen", "key.json"], tmp_path).returncode == 0
    p = _run(runner_bin, ["-m", "plain.gguf", "-p", "hi", "-n", "2", "--temp", "0",
                          "--gpu", "off", "--transcript", "run.json",
                          "--sign-key", "key.json"], tmp_path)
    assert p.returncode == 0, p.stderr
    assert "record" not in json.loads((tmp_path / "run.json").read_text())["model"]
    w = _run(runner_bin, ["--lineage", "run.json"], tmp_path)
    assert w.returncode == 0 and "ORIGIN" in w.stdout, w.stdout
    # an edited receipt is caught by its own signature
    r = tmp_path / "run.json"
    assert '"threads":' in r.read_text()
    r.write_text(r.read_text().replace('"threads":', '"threads": ', 1))
    w2 = _run(runner_bin, ["--lineage", "run.json"], tmp_path)
    assert w2.returncode == 2 and "BROKEN" in w2.stdout, w2.stdout
