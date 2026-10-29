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

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def _caps():
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
