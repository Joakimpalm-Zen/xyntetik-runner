"""Every test file must be reachable by something that runs it.

Three separate defects on 2026-09-07 were the same defect: a hand-kept list
of what to check drifting away from what exists.

  * `tests/test_caps.py` held a literal list of the formats CUDA supports and
    had never admitted NVFP4, shipped 2026-09-06;
  * the Windows CI job runs a hand-picked selection of smokes, and the five
    files it had never named were exactly where the Windows-only defects
    accumulated for nineteen days;
  * the `test` target enumerates its pytest files, and thirteen test files
    were named in neither the Makefile nor any workflow. Nothing ran them.
    `tests/test_receipts.py` was one of them.

The first two were fixed by editing the lists, which fixes the instance and
not the shape. This is the gate for the shape: a new test file that nothing
invokes fails here, on the day it is written, rather than being discovered
by whoever eventually needs the feature it covers.
"""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Files that are deliberately not invoked by a gate, each with the reason.
# Kept small and argued: an entry here is a decision, and the next reader
# should be able to disagree with it.
EXEMPT = {}


def _runner_text():
    """Everything that could name a test file: the Makefile and every
    workflow. Read as text on purpose -- a file named in a comment still
    counts as an answer, because a reader looking for "does anything run
    this" will find it."""
    parts = [(ROOT / "Makefile").read_text(encoding="utf-8")]
    wf = ROOT / ".github" / "workflows"
    for name in sorted(os.listdir(wf)):
        parts.append((wf / name).read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_every_test_file_is_named_by_a_gate():
    text = _runner_text()
    tests = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
    assert tests, "no test files found; the glob is wrong, not the tree"
    orphans = [t for t in tests if t not in text and t not in EXEMPT]
    assert not orphans, (
        "these test files are named in neither the Makefile nor any workflow, "
        "so nothing runs them: " + ", ".join(orphans) +
        ". Add them to a target, or add an EXEMPT entry saying why not."
    )


def test_the_exempt_list_has_no_stale_entries():
    """An exemption for a file that no longer exists is a claim about
    nothing, and it hides the next real one."""
    present = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    stale = sorted(set(EXEMPT) - present)
    assert not stale, "EXEMPT names files that are gone: " + ", ".join(stale)
