"""The ledger of device classes CI cannot reach.

These test the gate's teeth rather than its contents: whether a stale row,
a failing row, a missing row and a recording made on the wrong machine are
each refused. What the real docs/device-evidence.json says about the lab's
hardware is a fact about the lab, not a property to assert here.
"""

import datetime
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "device-evidence.py"

spec = importlib.util.spec_from_file_location("device_evidence", SCRIPT)
de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(de)

TODAY = datetime.date(2026, 9, 7)


def ledger(tmp_path, **row):
    doc = {
        "schema": de.SCHEMA,
        "policy": {"default_max_age_days": 45},
        "classes": [{
            "id": "box",
            "max_age_days": 30,
            "expect": {"os": "windows", "gpu_backend": "cuda"},
            "last_verified": row or None,
        }],
    }
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return p


def fresh(days, result="pass"):
    return {"date": (TODAY - datetime.timedelta(days=days)).isoformat(),
            "commit": "abc", "ran": "make test", "result": result}


def test_a_recent_pass_is_ok(tmp_path, capsys):
    doc = de.load(ledger(tmp_path, **fresh(2)))
    assert de.check(doc, TODAY) == []


def test_a_row_past_its_policy_is_stale(tmp_path):
    doc = de.load(ledger(tmp_path, **fresh(31)))
    assert de.check(doc, TODAY) == ["box"]


def test_the_boundary_day_is_not_stale(tmp_path):
    doc = de.load(ledger(tmp_path, **fresh(30)))
    assert de.check(doc, TODAY) == []


def test_a_failing_run_is_not_coverage(tmp_path):
    """Recent and red is worse than absent, not better: the point of the
    ledger is that somebody looked, and somebody looked and it broke."""
    doc = de.load(ledger(tmp_path, **fresh(1, result="fail")))
    assert de.check(doc, TODAY) == ["box"]


def test_a_class_with_no_evidence_is_refused(tmp_path):
    doc = de.load(ledger(tmp_path))
    assert de.check(doc, TODAY) == ["box"]


def test_the_check_prints_every_class_not_only_the_bad_ones(tmp_path, capsys):
    """A check that only speaks when it is angry trains a reader to take
    silence for coverage, which is the exact habit this file exists to break.
    """
    doc = de.load(ledger(tmp_path, **fresh(2)))
    de.check(doc, TODAY)
    assert "box" in capsys.readouterr().out


def test_a_wrong_schema_is_refused(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"schema": "something.else", "classes": []}),
                 encoding="utf-8")
    with pytest.raises(SystemExit):
        de.load(p)


def test_recording_from_the_wrong_machine_is_refused(tmp_path):
    """The row's value is that a machine wrote it about itself. Recording a
    Linux box's caps against the Windows class would make the ledger say
    something no machine said."""
    path = ledger(tmp_path)
    doc = de.load(path)
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps({"version": "0.5.1", "os": "linux",
                                "arch": "x86_64",
                                "gpu": {"backend": "cuda", "name": "A100"}}),
                    encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        de.record(doc, "box", "make test", "pass", str(caps), None, TODAY)
    assert "wrong machine" in str(e.value)


def test_recording_the_right_machine_writes_a_row(tmp_path):
    doc = de.load(ledger(tmp_path))
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps({"version": "0.5.1", "os": "windows",
                                "arch": "x86_64", "cpu_cores": 8,
                                "gpu": {"backend": "cuda",
                                        "name": "NVIDIA GeForce RTX 3070"}}),
                    encoding="utf-8")
    cls = de.record(doc, "box", "make test", "pass", str(caps), None, TODAY)
    assert cls["last_verified"]["caps"]["gpu_name"] == "NVIDIA GeForce RTX 3070"
    assert cls["last_verified"]["date"] == TODAY.isoformat()
    assert de.check(doc, TODAY) == []


def test_an_unknown_class_is_refused(tmp_path):
    doc = de.load(ledger(tmp_path))
    caps = tmp_path / "caps.json"
    caps.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        de.record(doc, "not-a-class", "x", "pass", str(caps), None, TODAY)


def test_the_committed_ledger_parses_and_covers_the_readme_platforms():
    """The real file, checked for shape rather than freshness: every class
    names what claims it, and the three README platform rows are covered."""
    doc = de.load(ROOT / "docs" / "device-evidence.json")
    ids = {c["id"] for c in doc["classes"]}
    assert {"windows-x86_64-cuda", "linux-x86_64-cuda",
            "macos-arm64-metal"} <= ids
    for cls in doc["classes"]:
        assert cls.get("claimed_by"), cls["id"]
        assert cls.get("covers"), cls["id"]


def test_the_cli_exits_nonzero_on_a_stale_ledger(tmp_path):
    p = ledger(tmp_path, **fresh(400))
    r = subprocess.run([sys.executable, str(SCRIPT), "--check", "--json",
                        str(p), "--today", TODAY.isoformat()],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert r.returncode == 1
    assert b"box" in r.stdout + r.stderr
