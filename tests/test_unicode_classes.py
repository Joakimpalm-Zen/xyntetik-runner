"""src/unicode_classes.h is what scripts/gen-unicode-classes.py writes.

The table is regenerated from this python's unicodedata when its Unicode
version is the header's, and compared from the include guard down (the
comment above it names the tokenizers version that cross-checked it, which
this environment may lack)."""
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "unicode_classes.h"


def _body(text):
    return text[text.index("#ifndef RUNNER_UNICODE_CLASSES_H"):]


def test_the_class_table_is_the_generators():
    text = HEADER.read_text()
    m = re.search(r"Unicode (\d+\.\d+\.\d+) \(python unicodedata\)", text)
    assert m, "the header names its Unicode version"
    if m.group(1) != unicodedata.unidata_version:
        pytest.skip(f"this python's unicodedata is {unicodedata.unidata_version}, "
                    f"the table's {m.group(1)}")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen-unicode-classes.py")],
                         capture_output=True, text=True, check=True).stdout
    assert _body(out) == _body(text)
