"""The counts the README and the manual state for what loads are the counts
`runner --info caps` reports.

Both pages said how many architectures Runner admits and both had drifted
(20 and 19 against the 21 the binary lists) because nothing tied a number in
prose to the list it counts. The anchor is the binary's own `--info caps`
output, which is the list the loader refuses against; the prose is checked
against it, never the other way round.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def _caps():
    if not RUNNER.exists():
        pytest.skip("build ./runner first (make)")
    out = subprocess.run([str(RUNNER), "--info", "caps"], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, check=True, timeout=30).stdout
    return json.loads(out)


def _claims(path, pattern):
    text = (ROOT / path).read_text(encoding="utf-8")
    found = [int(m.group(1)) for m in re.finditer(pattern, text)]
    assert found, f"{path} no longer states the count this test pins: {pattern}"
    return found


def test_documented_architecture_count_matches_caps():
    n = len(_caps()["architectures"])
    assert _claims("README.md", r"(\d+) admitted architecture labels") == [n]
    assert _claims("MANUAL.md", r"(\d+) admitted architectures") == [n]


def test_documented_weight_format_count_matches_caps():
    n = len(_caps()["quants"])
    assert _claims("README.md", r"(\d+) weight formats") == [n]
    assert _claims("MANUAL.md", r"(\d+) weight formats") == [n]


def _inventory():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "capability_inventory", ROOT / "scripts" / "capability-inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capability_inventory_is_current_and_every_surface_owned():
    """docs/capability-inventory.md is what the source says today, and every
    flag, route, switch, record, client name and source file has one owner."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "capability-inventory.py"),
                        "--check"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_capability_inventory_refuses_an_unowned_surface():
    inv = _inventory()
    surfaces = inv.collect()
    assert not inv.assign(surfaces).errors
    surfaces["command line"]["--a-new-flag"] = "src/main.c"
    # read only in a shared file and matching no name rule: no owner
    surfaces["env"]["A_NEW_SWITCH"] = {"src/model.c"}
    surfaces["source"]["src/a_new_file.c"] = "a_new_file"
    errors = "\n".join(inv.assign(surfaces).errors)
    for name in ("--a-new-flag", "A_NEW_SWITCH", "src/a_new_file.c"):
        assert f"{name} has no owner" in errors, errors


def test_capability_inventory_refuses_a_rule_that_claims_nothing():
    inv = _inventory()
    surfaces = inv.collect()
    del surfaces["HTTP"]["/v1/runner/provenance"]
    del surfaces["record"]["watermark_detect.v1"]
    errors = "\n".join(inv.assign(surfaces).errors)
    assert "the rule for /v1/runner/provenance matches nothing" in errors, errors
    assert "the rule for watermark_detect matches nothing" in errors, errors
