#!/usr/bin/env python3
"""A change that edits comments must not edit code.

    scripts/comment-only-check.py --base origin/main [--head HEAD] FILE...

For each C-family file named, the two revisions are compared with every
comment taken out: `//` to the end of the line and `/* ... */`, outside string
and character literals. What is left, line by line with blank lines and
trailing blanks dropped, must be the same text. A file that differs is printed
with its first differing code line from each side, and the exit code is 1.

It reads the files as text and knows nothing about the preprocessor: a comment
marker inside a macro's continuation line is handled like any other, and a
`#include` path with `//` in it is not expected in this tree.
"""
import argparse
import subprocess
import sys


def code_lines(text):
    """`text` without comments: its non-blank lines, right-stripped."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                # a literal never spans a line here; stop at one so a stray
                # apostrophe (a `#error` message) cannot swallow the file
                if text[j] == "\n":
                    break
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            # a block comment that held line breaks still separates tokens
            out.append("\n" if "\n" in text[i:(n if j < 0 else j)] else " ")
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return [line.rstrip() for line in "".join(out).split("\n") if line.strip()]


def show(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"cannot read {path} at {ref}: {r.stderr.decode().strip()}")
    return r.stdout.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default=None, help="a revision; the working tree when omitted")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    bad = 0
    for path in a.files:
        before = code_lines(show(a.base, path))
        after = code_lines(show(a.head, path) if a.head
                           else open(path, encoding="utf-8", errors="replace").read())
        if before == after:
            print(f"ok: {path}: {len(before)} code lines unchanged")
            continue
        bad += 1
        k = next((i for i, (x, y) in enumerate(zip(before, after)) if x != y),
                 min(len(before), len(after)))
        print(f"CODE CHANGED: {path}: {len(before)} code lines before, {len(after)} after; "
              f"first difference at code line {k + 1}")
        print("  before:", before[k] if k < len(before) else "<end of file>")
        print("  after: ", after[k] if k < len(after) else "<end of file>")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
