"""R17.9: attest lineage walks an unlearned adapter's chain, A2 to its parent A1 to the base.

An unlearn record (runner --unlearn) cites its parent record by sha256 in unlearn.parent_record. `runner attest
lineage A2.gguf` follows that link like a from_adapter record: the parent must be on disk and be the record A2's
record names, byte for byte; a missing or changed parent is BROKEN. A record that is its own ancestor is BROKEN as a
cycle, and a chain longer than the stated bound (32 links deep) is BROKEN, whatever kind its links are.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = [{"prompt": "row %d" % i, "completion": " text %d" % i} for i in range(5)]
RECIPE = ["--train-steps", "3", "--lr", "3e-3", "-t", "2", "--gpu", "off"]


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith("RUNNER_") and not k.startswith("XR_")}


def _run(exe, args):
    return subprocess.run([str(exe), *map(str, args)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=600, env=_env())


def _rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def chain(runner_bin, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(tmp_path / "base.gguf")],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    _rows(tmp_path / "data1.jsonl", ROWS)
    _rows(tmp_path / "drop.jsonl", ROWS[1:3])
    p = _run(runner_bin, ["-m", tmp_path / "base.gguf", "--train", tmp_path / "data1.jsonl",
                          "--train-out", tmp_path / "a1.gguf", *RECIPE])
    assert p.returncode == 0, p.stderr
    p = _run(runner_bin, ["--unlearn", tmp_path / "a1.gguf.train.json", "--drop", tmp_path / "drop.jsonl",
                          "-m", tmp_path / "base.gguf", "--train", tmp_path / "data1.jsonl",
                          "--train-out", tmp_path / "a2.gguf"])
    assert p.returncode == 0, p.stderr
    return tmp_path


def test_lineage_walks_an_unlearned_adapter_to_its_parent_and_the_base(runner_bin, chain):
    r = _run(runner_bin, ["attest", "lineage", chain / "a2.gguf"])
    out = r.stdout
    assert "unlearned from" in out and "a1.gguf.train.json" in out, out
    assert "dropped 2 rows of 2 lines" in out, out
    assert out.count("train -> ") == 2, out            # A2's record and A1's
    assert "BROKEN" not in out and r.returncode == 1, (r.returncode, out)   # consistent, not signed


def test_a_missing_or_changed_parent_record_is_broken(runner_bin, chain):
    parent = chain / "a1.gguf.train.json"
    saved = parent.read_bytes()
    parent.write_bytes(saved + b" ")
    r = _run(runner_bin, ["attest", "lineage", chain / "a2.gguf"])
    assert r.returncode == 2 and "changed after the next step recorded it" in r.stdout, r.stdout
    parent.unlink()
    r = _run(runner_bin, ["attest", "lineage", chain / "a2.gguf"])
    assert r.returncode == 2 and "is missing" in r.stdout, r.stdout


def _record(name, parent=None, parent_sha=None):
    rec = {"schema_version": "xyntetik.runner.train.v1", "runner": "test",
           "adapter": {"path": name + ".gguf", "sha256": hashlib.sha256(name.encode()).hexdigest()}}
    if parent:
        link = {"path": parent}
        if parent_sha:
            link["sha256"] = parent_sha
        rec["unlearn"] = {"parent_record": link, "dropped_sha256": "0" * 64, "dropped_count": 1, "dropped_distinct": 1}
    return rec


def _continued(name, parent_rec, parent_out):
    # a train record that continued the adapter parent_out, its record named by path alone (a from_adapter
    # record link may carry no hash, a weakness of its own: R6.94), so a loop of records is possible
    rec = _record(name)
    rec["from_adapter"] = {"path": parent_out + ".gguf", "sha256": hashlib.sha256(parent_out.encode()).hexdigest(),
                           "record": {"path": parent_rec}}
    return rec


def test_a_record_that_is_its_own_ancestor_is_broken_as_a_cycle(runner_bin, tmp_path):
    # x continued y's adapter and y continued x's: x -> y -> x
    (tmp_path / "x.json").write_text(json.dumps(_continued("x", "y.json", "y")))
    (tmp_path / "y.json").write_text(json.dumps(_continued("y", "x.json", "x")))
    r = _run(runner_bin, ["attest", "lineage", tmp_path / "x.json"])
    assert r.returncode == 2 and "cycle" in r.stdout, r.stdout


def test_an_unlearn_link_without_a_hash_is_broken(runner_bin, tmp_path):
    """A link without a hash is not a link: an unlearn record whose parent_record names no sha256 is BROKEN, naming
    the field, not walked as an unsigned parent."""
    (tmp_path / "p.json").write_text(json.dumps(_record("p")))
    (tmp_path / "c.json").write_text(json.dumps(_record("c", "p.json")))
    r = _run(runner_bin, ["attest", "lineage", tmp_path / "c.json"])
    assert r.returncode == 2 and "unlearn.parent_record has no sha256" in r.stdout, r.stdout


def test_a_chain_longer_than_the_bound_is_broken(runner_bin, tmp_path):
    # 40 records, each citing the next by its exact sha256: deeper than the stated bound
    names = ["r%02d" % i for i in range(40)]
    sha = None
    for i in reversed(range(len(names))):
        rec = _record(names[i], names[i + 1] + ".json" if i + 1 < len(names) else None, sha)
        data = json.dumps(rec).encode()
        (tmp_path / (names[i] + ".json")).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
    r = _run(runner_bin, ["attest", "lineage", tmp_path / "r00.json"])
    assert r.returncode == 2 and "longer than 32 links" in r.stdout, r.stdout[-600:]
