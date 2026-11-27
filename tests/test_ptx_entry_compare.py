"""scripts/ptx-entry-compare.py against a live control: the real src/kernels_ptx.h, edited in the ways the
advisor's PTX ruling (R3.84, R4.48.33) names, must read exactly as the ruling says.

- A pure renumbering of one entry's function index f (in $L__BB<f>_<b> and __local_depot<f>) changes nothing.
- A one-operand edit inside one existing entry changes exactly that entry.
- A mixed f inside one entry fails loudly, as an exception and as exit 2 from the command line.
- Removing one entry reads as exactly one removed, and nothing changed.
"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ptx-entry-compare.py"

spec = importlib.util.spec_from_file_location("ptx_entry_compare", SCRIPT)
pec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pec)


@pytest.fixture(scope="module")
def base():
    text = pec.embedded_text(ROOT / "src" / "kernels_ptx.h")
    entries, _ = pec.split_entries(text)
    # an entry with at least two block labels and one tid.x, the edits' target
    for name, body in entries:
        fs = {f for f, _ in pec.LABEL.findall(body)}
        if len(pec.LABEL.findall(body)) >= 2 and len(fs) == 1 and "%tid.x" in body:
            return text, entries, name, body, int(fs.pop())
    pytest.fail("no entry in src/kernels_ptx.h has two block labels and a %tid.x")


def write_header(path, text):
    # the form embed-ptx.py writes and embedded_text reads: one string literal a line
    path.write_text("".join(json.dumps(line) + "\n" for line in text.splitlines(keepends=True)))


def test_identity(base):
    text = base[0]
    r = pec.compare_texts(text, text)
    assert (r["added"], r["removed"], r["changed"]) == ([], [], [])
    assert r["order_identical"] and r["module_text_identical"]


def test_pure_renumbering_is_no_change(base):
    text, _, name, body, f = base
    new = pec.LABEL.sub(lambda m: f"$L__BB99999_{m.group(2)}" if int(m.group(1)) == f else m.group(0), body)
    new = re.sub(rf"__local_depot{f}\b", "__local_depot99999", new)
    assert new != body
    r = pec.compare_texts(text, text.replace(body, new, 1))
    assert (r["added"], r["removed"], r["changed"]) == ([], [], [])
    assert r["order_identical"] and r["module_text_identical"]


def test_one_operand_edit_is_exactly_one_change(base, tmp_path):
    text, _, name, body, _ = base
    new = body.replace("%tid.x", "%tid.y", 1)
    edited = text.replace(body, new, 1)
    r = pec.compare_texts(text, edited)
    assert r["changed"] == [name]
    assert (r["added"], r["removed"]) == ([], [])
    old_h, new_h = tmp_path / "old.h", tmp_path / "new.h"
    write_header(old_h, text)
    write_header(new_h, edited)
    p = subprocess.run([sys.executable, str(SCRIPT), str(old_h), str(new_h)], capture_output=True, text=True)
    assert p.returncode == 1 and f"changed: 1: {name}" in p.stdout, p.stdout + p.stderr
    p = subprocess.run([sys.executable, str(SCRIPT), str(old_h), str(new_h), "--allow-changed"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_mixed_index_fails_loudly(base, tmp_path):
    text, _, name, body, f = base
    first = pec.LABEL.search(body)
    new = body[:first.start()] + f"$L__BB99999_{first.group(2)}" + body[first.end():]
    edited = text.replace(body, new, 1)
    with pytest.raises(pec.MixedIndex, match=name):
        pec.compare_texts(text, edited)
    old_h, new_h = tmp_path / "old.h", tmp_path / "new.h"
    write_header(old_h, text)
    write_header(new_h, edited)
    p = subprocess.run([sys.executable, str(SCRIPT), str(old_h), str(new_h)], capture_output=True, text=True)
    assert p.returncode == 2 and "mixed function index" in p.stderr, p.stdout + p.stderr


def test_removing_one_entry_is_exactly_one_removed(base):
    text, entries, name, body, _ = base
    r = pec.compare_texts(text, text.replace(body, "", 1))
    assert r["removed"] == [name]
    assert (r["added"], r["changed"]) == ([], [])
    assert r["order_identical"]
