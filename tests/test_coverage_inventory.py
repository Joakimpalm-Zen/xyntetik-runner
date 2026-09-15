"""Every admitted architecture, every chat-template id and every pinned
artifact has a disposition for the cross-family gates, and a PASS has its
evidence on disk.

The 2026-09-14 report showed three families failing three different ways
while every fixture gate was green: the roster of what had actually been run
was not a thing anybody could read. `tests/compatibility/coverage.json` is
that roster. This test keeps it complete against the runner's own admitted
architecture list (`--caps`), the template ids `template_name` returns, and
the model manifest, so a new family cannot inherit certification by
omission: it arrives as NOT RUN, visibly, until somebody runs the gate.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "tests" / "compatibility" / "coverage.json"
MANIFEST = ROOT / "tests" / "compatibility" / "models.json"
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
DISPOSITIONS = {"PASS", "FAIL", "NOT RUN", "UNSUPPORTED"}


def _coverage():
    return json.loads(COVERAGE.read_text(encoding="utf-8"))


def _template_ids():
    src = (ROOT / "src" / "template.c").read_text(encoding="utf-8")
    i = src.index("const char *template_name(int t) {")
    body = src[i:i + 3000]
    ids = re.findall(r'return "([a-z0-9-]+)"', body.split("\n}\n")[0])
    ids.append("llama2-fallback")
    assert "chatml" in ids and "qwen38" in ids, ids
    return set(ids)


def _admitted_architectures():
    if not RUNNER.exists():
        pytest.skip("runner not built")
    out = subprocess.run([str(RUNNER), "--caps"], capture_output=True, text=True,
                         timeout=60).stdout
    return set(json.loads(out)["architectures"])


def _check_gate(where, gate):
    assert isinstance(gate, dict), where
    assert gate.get("status") in DISPOSITIONS, (where, gate)
    if gate["status"] == "PASS":
        ev = gate.get("evidence")
        assert ev, (where, "a PASS names its evidence")
        assert (ROOT / ev).exists(), (where, ev)
    if gate["status"] in ("FAIL", "UNSUPPORTED"):
        assert gate.get("note"), (where, "a FAIL or UNSUPPORTED says why")


def test_every_template_id_has_a_disposition():
    cov = _coverage()
    ids = _template_ids()
    missing = ids - set(cov["templates"])
    assert not missing, f"template ids without a disposition: {sorted(missing)}"
    stale = set(cov["templates"]) - ids
    assert not stale, f"dispositions for template ids that no longer exist: {sorted(stale)}"
    for tid, row in cov["templates"].items():
        for gate in ("tool_protocol", "client_loop"):
            _check_gate(f"templates.{tid}.{gate}", row[gate])
        s = row["sampling_anchor"]
        assert s.get("status") in DISPOSITIONS, (tid, s)
        if s["status"] == "PASS":
            assert s.get("preset") and s.get("anchor"), (tid, s)
        for backend, status in row["backends"].items():
            assert backend in ("cpu", "cuda", "metal") and status in DISPOSITIONS, (tid, backend, status)


def test_every_admitted_architecture_has_a_disposition():
    cov = _coverage()
    archs = _admitted_architectures()
    missing = archs - set(cov["architectures"])
    assert not missing, f"admitted architectures without a disposition: {sorted(missing)}"
    stale = set(cov["architectures"]) - archs
    assert not stale, f"dispositions for architectures the runner no longer admits: {sorted(stale)}"
    manifest_ids = {m["id"] for m in json.loads(MANIFEST.read_text())["models"]}
    templates = set(cov["templates"])
    for arch, row in cov["architectures"].items():
        assert row["templates"] and set(row["templates"]) <= templates, (arch, row["templates"])
        assert set(row["artifacts"]) <= manifest_ids, (arch, row["artifacts"])
        for gate in ("tool_protocol", "client_loop"):
            _check_gate(f"architectures.{arch}.{gate}", row[gate])
        for backend, status in row["backends"].items():
            assert backend in ("cpu", "cuda", "metal") and status in DISPOSITIONS, (arch, backend)


def test_every_pinned_artifact_belongs_to_an_architecture_row():
    cov = _coverage()
    manifest = json.loads(MANIFEST.read_text())["models"]
    placed = {a for row in cov["architectures"].values() for a in row["artifacts"]}
    unplaced = {m["id"] for m in manifest} - placed
    assert not unplaced, f"pinned artifacts with no architecture disposition: {sorted(unplaced)}"
    # the three artifacts the 2026-09-14 report named are pinned by hash, not
    # represented by a nearby row
    by_id = {m["id"]: m for m in manifest}
    for rid in cov["reported_artifacts"]:
        assert rid in by_id, rid
        assert len(by_id[rid]["sha256"]) == 64, rid
        assert by_id[rid].get("upstream") and by_id[rid].get("template") and by_id[rid].get("preset"), rid
        assert "tool" in by_id[rid]["checks"], rid


def test_a_not_run_row_is_not_a_pass():
    """The roster distinguishes what nobody ran from what ran green: a row
    can carry NOT RUN with no evidence, and only PASS may carry evidence."""
    cov = _coverage()
    for group in ("templates", "architectures"):
        for name, row in cov[group].items():
            for gate in ("tool_protocol", "client_loop"):
                g = row[gate]
                if g["status"] == "NOT RUN":
                    assert "evidence" not in g, (group, name, gate)
