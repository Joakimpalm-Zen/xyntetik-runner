#!/usr/bin/env python3
"""Refuse added file content that publishes a host path or a machine or
session name.

check-commit-hygiene.py reads commit messages and PR text; this reads what a
change ADDS to tracked files. On 2026-10-31 a record PR's gold-logits JSON
carried the lab host's paths (/home/lab/...) into the public repository,
and nothing mechanical looked at file content. Only added lines are read, so
a change is judged by what it brings, not by what is already there.

    check-content-hygiene.py --range BASE..HEAD     (CI: a pull request)
    check-content-hygiene.py --staged               (the pre-commit hook)
    check-content-hygiene.py --self-test

It exits 1 and names each offending file, line and pattern.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Host paths, and the names of machines and agent sessions, none of which
# belongs in a public file (AGENTS.md, "Never publish conversation content or
# session identifiers"; owner rule: no machine names in public).
FORBIDDEN = [
    (re.compile(r"/home/lab/"), "the lab host's home path"),
    (re.compile(r"/Users/[^/\s\"'`]+/"), "a macOS home path"),
    (re.compile(r"C:[\\/]+projectgrid", re.I), "the Windows host's work path"),
    (re.compile(r"\bZEN-GAMING\b", re.I), "a machine name"),
    (re.compile(r"\bgalaxa1\b", re.I), "an agent session name"),
    (re.compile(r"\bwindows3070\b", re.I), "an agent session name"),
    (re.compile(r"\bDawn-\d\b"), "an agent session name"),
    (re.compile(r"\bexec-share/"), "the lab's shared work directory"),
]
# Files that legitimately spell a pattern: this checker (its list and its
# self-test).
ALLOW = {"scripts/check-content-hygiene.py"}


def added_lines(diff: str):
    """(file, line number, text) for every added line of a unified diff."""
    path, lineno = None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            p = raw[4:].strip()
            path = None if p == "/dev/null" else (p[2:] if p.startswith("b/") else p)
        elif raw.startswith("@@"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
        elif raw.startswith("+") and path is not None:
            yield path, lineno, raw[1:]
            lineno += 1
        elif not raw.startswith("-") and path is not None and not raw.startswith("\\"):
            lineno += 1


def problems(diff: str) -> list[str]:
    out = []
    for path, n, text in added_lines(diff):
        if path in ALLOW:
            continue
        for rx, what in FORBIDDEN:
            m = rx.search(text)
            if m:
                out.append(f"{path}:{n}: {what} ({m.group(0)!r})")
    return out


def git_diff(args: list[str]) -> str:
    return subprocess.run(["git", "diff", "--unified=0", "--no-color", "--no-ext-diff", *args],
                          capture_output=True, text=True, check=True).stdout


def self_test() -> int:
    planted = ("diff --git a/docs/x.json b/docs/x.json\n--- a/docs/x.json\n+++ b/docs/x.json\n"
               "@@ -1,0 +1,3 @@\n+{\"hf\": \"" + "/home/" + "lab/r472/hf\",\n+ \"ok\": 1,\n"
               "+ \"host\": \"" + "gala" + "xa1\"}\n")
    found = problems(planted)
    allowed = problems(planted.replace("docs/x.json", "scripts/check-content-hygiene.py"))
    clean = problems("--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+runs on /usr/local/bin\n")
    deleted = problems("--- a/f.txt\n+++ b/f.txt\n@@ -1 +0,0 @@\n-" + "/home/" + "lab/x\n")
    ok = (len(found) == 2 and found[0].startswith("docs/x.json:1:") and found[1].startswith("docs/x.json:3:")
          and not allowed and not clean and not deleted)
    print(f"content-hygiene self-test: {'ok' if ok else 'FAILED'} (planted {len(found)}, allowlisted "
          f"{len(allowed)}, clean {len(clean)}, deleted line {len(deleted)})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", help="git revision range, e.g. origin/main..HEAD")
    ap.add_argument("--staged", action="store_true", help="the staged changes (pre-commit)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.range:
        diff = git_diff([a.range])
    elif a.staged:
        diff = git_diff(["--cached"])
    else:
        ap.error("--range, --staged or --self-test")
    bad = problems(diff)
    for b in bad:
        print(f"content-hygiene: {b}", file=sys.stderr)
    if bad:
        print("content-hygiene: refused: added content names a host path or a machine or session "
              "(AGENTS.md, the publication rule). Use a role or a relative path.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
