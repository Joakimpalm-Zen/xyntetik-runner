#!/usr/bin/env python3
"""The agent rule files stay small and say each thing once.

2026-10-06: an edit meant to replace one paragraph of AGENTS.md inserted it
after every character of the file; AGENTS.md grew from 372 to 150,523 lines,
every rule line was cut to one character per copy, and nothing noticed for
two days (every agent that read the file got no rules). This fails on the
shapes that edit produced: a file past a sane size, a paragraph that occurs
twice, or a long line repeated. It needs nothing but Python, so it runs in
`make test` and in the commit-hygiene workflow, which every pull request
waits for, docs-only ones included.
"""
import os
import sys

FILES = ["AGENTS.md", "CLAUDE.md"]
MAX_LINES = 800        # AGENTS.md is ~390 lines; twice that is a mistake
LONG_LINE = 40         # a line this long repeated more than MAX_REPEATS times
MAX_REPEATS = 3        # is copied text, not a recurring marker


def problems(text, name):
    out = []
    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        out.append(f"{name}: {len(lines)} lines (limit {MAX_LINES})")
    seen = {}
    for para in text.split("\n\n"):
        p = para.strip()
        if len(p) >= 80 and "\n" in p:
            seen[p] = seen.get(p, 0) + 1
    for p, n in seen.items():
        if n > 1:
            out.append(f"{name}: a paragraph occurs {n} times: {p.splitlines()[0][:70]!r}")
    counts = {}
    for ln in lines:
        s = ln.strip()
        if len(s) >= LONG_LINE:
            counts[s] = counts.get(s, 0) + 1
    for s, n in counts.items():
        if n > MAX_REPEATS:
            out.append(f"{name}: a line occurs {n} times: {s[:70]!r}")
    return out[:20]


def self_test():
    para = ("README style (owner): a feature gets two or three sentences on\n"
            "why it exists, written so a newcomer follows them, then how to use it.")
    clean = "# Rules\n\n" + para + "\n\nAnother rule, stated once and only once here.\n"
    assert problems(clean, "clean") == [], problems(clean, "clean")
    # the 2026-10-06 shape: the paragraph inserted after every character
    shredded = "".join(c + para + "\n" for c in "# Rules\n\nOne rule.\n")
    assert problems(shredded, "shredded"), "the shredded file passed"
    twice = clean + "\n" + para + "\n"
    assert problems(twice, "twice"), "a repeated paragraph passed"
    big = "\n".join(f"line {i}" for i in range(MAX_LINES + 5))
    assert problems(big, "big"), "an oversized file passed"
    print("check-agent-docs self-test: ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        self_test()
        sys.exit(0)
    paths = [a for a in sys.argv[1:] if not a.startswith("-")] or FILES
    bad = []
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            continue
        bad += problems(text, path)
    if bad:
        print("FAIL: agent rule files look damaged:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("agent docs: " + ", ".join(
        f"{p} {sum(1 for _ in open(p, encoding='utf-8'))} lines"
        for p in paths if os.path.exists(p)) + ", no repeats")
