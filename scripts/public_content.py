"""What never goes into this public repository: host paths and the names of
machines and agent sessions (AGENTS.md, "Never publish conversation content
or session identifiers"; the owner's rule, no machine names in public).

One list, read by both checkers: check-content-hygiene.py over the lines a
change adds to tracked files, and check-commit-hygiene.py over commit
messages (merge commits included) and pull request titles and bodies. Until
2026-11-03 only the file check had it, so a commit message naming a lab
session passed every check (R1.19).
"""
import re

FORBIDDEN = [
    (re.compile(r"/home/lab/"), "the lab host's home path"),
    (re.compile(r"/Users/[^/\s\"'`]+/"), "a macOS home path"),
    (re.compile(r"C:[\\/]+projectgrid", re.I), "the Windows host's work path"),
    (re.compile(r"\bZEN-GAMING\b", re.I), "a machine name"),
    (re.compile(r"\bgalaxa1\b", re.I), "an agent session name"),
    (re.compile(r"\bwindows3070\b", re.I), "an agent session name"),
    (re.compile(r"\bDawn-\d\b"), "an agent session name"),
    (re.compile(r"\bexec-share/"), "the lab's shared work directory"),
    # The owner's rule (2026-10-13) names the machines; the chip names stay
    # allowed: "NVIDIA Blackwell", "RTX PRO 6000 Blackwell", "Apple M1",
    # also with the line wrapped between the two words.
    # Blackwell in any case (2026-11-03, R1.19.1: nineteen evidence files had
    # been named for the lab box in lowercase); M1 stays case-sensitive, as
    # a lowercase "m1" is not the chip.
    (re.compile(r"(?<!RTX PRO 6000[ \n])(?<!NVIDIA[ \n])\bBlackwell\b", re.I), "a machine name"),
    (re.compile(r"(?<!Apple[ \n])\bM1\b"), "a machine name"),
]


# A record is what a run wrote, committed as evidence. `~/` keeps the owner's
# name out of it and not the tree: a run from a linked models folder wrote
# `~/AI_projects/.../models/...` into a report and every check passed it
# (R6.14). A record keeps `~/<file>` or `models/<file>`. Prose and source may
# name a home tree, as `~/.cache/xyntetik-runner`, so this reads records only,
# and commit messages not at all.
RECORD = re.compile(r"^docs/(compat-reports|benchmarks-raw|[^/]+-evidence)/")
RECORD_FORBIDDEN = [
    (re.compile(r"(?<![/.~])~/[^/\s\"'`]+/[^\s\"'`]"), "a home path with a tree under it, in a record"),
]


def forbidden_in(path):
    """The patterns a file at `path` is held to."""
    return FORBIDDEN + RECORD_FORBIDDEN if RECORD.match(path) else FORBIDDEN


def found(text):
    """(description, matched text) for each pattern found in text."""
    out = []
    for rx, what in FORBIDDEN:
        for m in rx.finditer(text):
            out.append((what, m.group(0)))
    return out
