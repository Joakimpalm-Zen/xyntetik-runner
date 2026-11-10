#!/usr/bin/env python3
"""One changelog entry per pull request, as its own file, folded in at release.

Every pull request used to add its entry to the top of CHANGELOG.md's Unreleased section, so each
merge conflicted with the next open pull request on the same lines, and cost a merge and a full
CI run to settle (R6.21). An entry is now a file of its own, changes/<ID>.md, holding the entry
exactly as CHANGELOG.md will show it; no two pull requests touch the same file. The release folds
the files into the Unreleased section and deletes them. changes/README.md has the format.

  check --base REV [--head REV] [--text FILE ...] [--opened ISO-TIME]
      The pull-request rule: the change adds exactly one entry file, or says why it has none
      with a line "Changelog: none, <why>" in the pull request's text (--text) or in one of its
      commit messages. A release fold (entry files deleted, none added) passes. A new entry
      written straight into CHANGELOG.md's Unreleased section counts as the entry, with a
      warning. A pull request opened (--opened) before changes/ reached the base branch is
      only warned, never failed, so the ones in flight at the switch merge as they are.
  fold [--write]
      Puts every entry file at the top of the Unreleased section, newest first (the order they
      reached this branch's first-parent history), and with --write writes CHANGELOG.md and
      deletes the files. Without --write it prints the section it would write.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGES = "changes"
README = "README.md"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")
REASON = re.compile(r"^\s*Changelog:\s*none\b[\s,:;.-]*(\S.*)$", re.I | re.M)
UNRELEASED = re.compile(r"^## Unreleased[ \t]*$", re.M)


def git(*args, cwd=None):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"changelog-entries: git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def annotate(kind, message):
    """A line for the log, and the same as a GitHub annotation inside Actions."""
    print(f"{kind}: {message}")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{kind}::{message}")


def entry_problems(path, text):
    """What is wrong with one entry file's name or text; empty when it is a good entry."""
    out = []
    name = path.split("/", 1)[1]
    if not NAME.match(name):
        out.append(f"{path}: name it changes/<plan ID or short slug>.md (letters, digits, '.', '_', '-')")
    lines = text.splitlines()
    first = next((l for l in lines if l.strip()), "")
    if not first:
        out.append(f"{path}: empty; write the entry as CHANGELOG.md will show it")
    elif not first.startswith("- "):
        out.append(f"{path}: the entry is one top-level list item and starts with '- '")
    for n, line in enumerate(lines, 1):
        if line.startswith("#"):
            out.append(f"{path}:{n}: no headings in an entry; the release writes the heading")
            break
    tops = [l for l in lines if l.startswith("- ")]
    if len(tops) > 1:
        out.append(f"{path}: {len(tops)} top-level items; one entry per file (nest the rest under it)")
    return out


def unreleased_span(text):
    """(first line, line after the last) of the Unreleased section's body, 1-based; None if absent."""
    m = UNRELEASED.search(text)
    if not m:
        return None
    start = text[:m.end()].count("\n") + 2
    nxt = re.compile(r"^## ", re.M).search(text, m.end())
    end = text[:nxt.start()].count("\n") + 1 if nxt else text.count("\n") + 2
    return start, end


def direct_entries(base, head):
    """New top-level entries the change writes straight into CHANGELOG.md's Unreleased section."""
    diff = git("diff", "-U0", "--no-renames", base, head, "--", "CHANGELOG.md")
    if not diff:
        return 0
    try:
        text = git("show", f"{head}:CHANGELOG.md")
    except SystemExit:
        return 0
    span = unreleased_span(text)
    if not span:
        return 0
    lines = text.splitlines()
    added = 0
    for hunk in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
        first, count = int(hunk.group(1)), int(hunk.group(2) or 1)
        for n in range(first, first + count):
            if span[0] <= n < span[1] and n - 1 < len(lines) and lines[n - 1].startswith("- "):
                added += 1
    return added


def switch_time(base):
    """When changes/README.md reached the base branch's first-parent history (unix time), or None."""
    out = git("log", "--first-parent", "--diff-filter=A", "--format=%ct", base, "--", f"{CHANGES}/{README}").strip()
    return int(out.splitlines()[-1]) if out else None


def opened_before_switch(args):
    if not args.opened:
        return False
    switched = switch_time(args.base)
    if switched is None:
        return False
    opened = datetime.datetime.fromisoformat(args.opened.replace("Z", "+00:00")).timestamp()
    return opened < switched


def cmd_check(args):
    code = check_change(args)
    if code and opened_before_switch(args):
        annotate("warning", "this pull request was opened before changes/ existed, so the rule above "
                            "is a warning for it, not a failure")
        return 0
    return code


def check_change(args):
    base = git("merge-base", args.base, args.head).strip()
    status = git("diff", "--name-status", "--no-renames", base, args.head, "--", CHANGES)
    added, deleted = [], []
    for line in status.splitlines():
        code, _, path = line.partition("\t")
        if path == f"{CHANGES}/{README}" or not path.startswith(f"{CHANGES}/"):
            continue
        if code.startswith("A"):
            added.append(path)
        elif code.startswith("D"):
            deleted.append(path)
    if deleted and not added:
        print(f"ok: a release fold, {len(deleted)} entr{'y' if len(deleted) == 1 else 'ies'} folded into CHANGELOG.md")
        return 0
    problems = []
    for path in added:
        problems += entry_problems(path, git("show", f"{args.head}:{path}"))
    direct = direct_entries(base, args.head)
    count = len(added) + direct
    texts = [git("log", "--format=%B", f"{base}..{args.head}")]
    for t in args.text:
        texts.append(Path(t).read_text(encoding="utf-8", errors="replace"))
    reason = next((m.group(1).strip() for t in texts for m in [REASON.search(t)] if m), None)

    soft = opened_before_switch(args)
    kind = "warning" if soft else "error"
    for p in problems:
        annotate(kind, p)
    if problems:
        return 1
    if count > 1:
        where = added + ([f"{direct} in CHANGELOG.md"] if direct else [])
        annotate(kind, f"{count} changelog entries ({', '.join(where)}): one per pull request; "
                          "split the pull request or merge the entries into one")
        return 1
    if count == 1:
        if direct:
            annotate("warning", "the entry is written straight into CHANGELOG.md; that is accepted for a "
                                "branch cut before changes/ existed, but it conflicts with every other such "
                                "branch: move it to changes/<ID>.md (see changes/README.md)")
        else:
            print(f"ok: one entry, {added[0]}")
        return 0
    if reason:
        print(f"ok: no entry, because: {reason}")
        return 0
    annotate(kind, "no changelog entry: add changes/<plan ID>.md with the entry (see changes/README.md), "
                      "or say why there is none with a line 'Changelog: none, <why>' in the pull "
                      "request's description or a commit message")
    return 1


def added_order(paths, cwd):
    """Paths newest first: the time each reached this branch's first-parent history."""
    def when(p):
        out = git("log", "--first-parent", "--diff-filter=A", "--format=%ct", "-1", "--", p, cwd=cwd).strip()
        return int(out) if out else sys.maxsize  # not committed yet: the newest
    return sorted(paths, key=lambda p: (-when(p), p))


def cmd_fold(args):
    changelog = Path(args.changelog)
    folder = changelog.parent / CHANGES
    text = changelog.read_text(encoding="utf-8")
    m = UNRELEASED.search(text)
    if not m:
        raise SystemExit(f"changelog-entries: {changelog} has no '## Unreleased' heading to fold into")
    files = sorted(p for p in folder.glob("*.md") if p.name != README) if folder.is_dir() else []
    if not files:
        print("nothing to fold: changes/ holds no entries")
        return 0
    rel = [str(p.relative_to(changelog.parent)) for p in files]
    problems = [x for p, r in zip(files, rel) for x in entry_problems(r, p.read_text(encoding="utf-8"))]
    if problems:
        for p in problems:
            annotate("error", p)
        return 1
    ordered = added_order(rel, changelog.parent)
    entries = [(changelog.parent / r).read_text(encoding="utf-8").strip("\n") for r in ordered]
    block = "\n\n".join(entries) + "\n"
    head, rest = text[:m.end()], text[m.end():]
    body = rest.lstrip("\n")
    folded = head + "\n\n" + block + ("\n" + body if body else "")
    if not args.write:
        noun = "entry" if len(entries) == 1 else "entries"
        print(f"would put {len(entries)} {noun} at the top of ## Unreleased, newest first "
              f"({', '.join(ordered)}); run with --write:\n")
        print(block, end="")
        return 0
    changelog.write_text(folded, encoding="utf-8")
    for r in ordered:
        (changelog.parent / r).unlink()
    print(f"folded {len(entries)} {'entry' if len(entries) == 1 else 'entries'} into {changelog.name}, "
          f"newest first: {', '.join(ordered)}")
    print(f"stage them with: git add {changelog.name} {CHANGES}/")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the pull-request rule")
    c.add_argument("--base", required=True, help="the branch the change merges into (a SHA or ref)")
    c.add_argument("--head", default="HEAD")
    c.add_argument("--text", action="append", default=[], help="a file with the pull request's title or description")
    c.add_argument("--opened", help="when the pull request was opened (ISO 8601); one opened before the "
                                    "switch to changes/ is warned, not failed")
    f = sub.add_parser("fold", help="fold changes/ into CHANGELOG.md at release")
    f.add_argument("--write", action="store_true")
    f.add_argument("--changelog", default=str(ROOT / "CHANGELOG.md"))
    args = ap.parse_args(argv)
    return cmd_check(args) if args.cmd == "check" else cmd_fold(args)


if __name__ == "__main__":
    sys.exit(main())
