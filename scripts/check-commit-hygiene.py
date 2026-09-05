#!/usr/bin/env python3
"""Refuse commit messages and PR text that publish session identifiers.

The rule is AGENTS.md "Never publish conversation content or session
identifiers": no session URLs or ids, no tool-default trailers that carry
them, no e-mail addresses, and exactly one signature form,

    Co-Authored-By: <Agent> (<Model>) & Joakimpalm-Zen

This script is the mechanical half of that rule. It runs in CI over every
commit of a pull request (and its title and body) and as the local
commit-msg hook (`make hooks`). It exits 1 and names the offending line.

    check-commit-hygiene.py --range BASE..HEAD [--text FILE ...]
    check-commit-hygiene.py --message-file .git/COMMIT_EDITMSG
    check-commit-hygiene.py --self-test

On 2026-09-05 seven commits reached the public repository with a
`Claude-Session: https://claude.ai/code/session_...` trailer, the default
trailer of the Claude Code harness, because nothing mechanical checked. The
rule was already written down; this makes it enforced.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Anything matching one of these is a session identifier or a tool-default
# trailer carrying one. Case-insensitive.
FORBIDDEN = [
    (re.compile(r"claude\.ai/code", re.I), "Claude Code session URL"),
    (re.compile(r"^[ \t]*Claude-Session[ \t]*:", re.I | re.M), "Claude-Session trailer"),
    (re.compile(r"chatgpt\.com/(c|share|g)/", re.I), "ChatGPT conversation URL"),
    (re.compile(r"chat\.openai\.com/", re.I), "ChatGPT conversation URL"),
    (re.compile(r"\bsession_[A-Za-z0-9]{12,}", re.I), "session id"),
    (re.compile(r"\bconversation[_-]?id\b", re.I), "conversation id"),
    (re.compile(r"\bsess-[A-Za-z0-9]{8,}", re.I), "session id"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "e-mail address"),
]
TRAILER = re.compile(r"^Co-Authored-By:\s*(.*)$", re.I | re.M)
TRAILER_OK = re.compile(r"^[A-Za-z][A-Za-z0-9 .+/-]* \([^()]+\) & Joakimpalm-Zen$")


def problems(text: str) -> list[str]:
    out = []
    for rx, what in FORBIDDEN:
        for m in rx.finditer(text):
            line = text[text.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0]
            out.append(f"{what}: {line.strip()[:120]}")
    for m in TRAILER.finditer(text):
        body = m.group(1).strip()
        if not TRAILER_OK.match(body):
            out.append("Co-Authored-By must read '<Agent> (<Model>) & Joakimpalm-Zen', "
                       f"got: {body[:120]}")
    # one finding per distinct line is enough
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def commits(rng: str) -> list[tuple[str, str]]:
    raw = subprocess.run(["git", "log", "--format=%H%x00%B%x1e", rng],
                         capture_output=True, text=True, check=True).stdout
    out = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        sha, _, msg = rec.partition("\x00")
        out.append((sha.strip(), msg))
    return out


def self_test() -> int:
    bad = [
        "Fix thing\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_012uYHKvzawVPnMTMWPZC7op\n",
        "notes at https://chatgpt.com/share/abc123\n",
        "Co-Authored-By: Codex & Joakimpalm-Zen\n",
        "contact me at someone@example.com\n",
    ]
    good = [
        "Fix thing\n\nCo-Authored-By: Claude Code (Fable 5.1) & Joakimpalm-Zen\n",
        "Fix thing\n\nCo-Authored-By: Codex (GPT-5) & Joakimpalm-Zen\n",
        "Merge pull request #34 from Joakimpalm-Zen/release-0.4.9\n",
        "docs: link https://github.com/Joakimpalm-Zen/xyntetik-runner/releases\n",
    ]
    ok = all(problems(b) for b in bad) and not any(problems(g) for g in good)
    print("commit-hygiene self-test", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", help="git revision range, e.g. origin/main..HEAD")
    ap.add_argument("--message-file", help="a commit message file (commit-msg hook)")
    ap.add_argument("--text", action="append", default=[],
                    help="extra text files to check (PR title/body)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    bad = 0
    if a.range:
        for sha, msg in commits(a.range):
            for p in problems(msg):
                print(f"{sha[:12]}: {p}")
                bad += 1
    if a.message_file:
        with open(a.message_file, encoding="utf-8", errors="replace") as f:
            for p in problems(f.read()):
                print(f"commit message: {p}")
                bad += 1
    for path in a.text:
        with open(path, encoding="utf-8", errors="replace") as f:
            for p in problems(f.read()):
                print(f"{path}: {p}")
                bad += 1
    if bad:
        print(f"\n{bad} publication-rule violation(s). See AGENTS.md, "
              "'Never publish conversation content or session identifiers'. "
              "Strip the tool's default trailer; sign with exactly "
              "'Co-Authored-By: <Agent> (<Model>) & Joakimpalm-Zen'.")
        return 1
    if a.range or a.message_file or a.text:
        print("commit hygiene ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
