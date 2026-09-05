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
    # session ids are base62 with digits; a plain identifier such as
    # session_reconnection is a word, not an id
    (re.compile(r"\bsession_(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{12,}", re.I), "session id"),
    (re.compile(r"\bsess-[A-Za-z0-9]{8,}(?<![a-z]{8})", re.I), "session id"),
]
# Agents sign with the project form; humans may use GitHub's own co-author
# form, which needs the e-mail to link the profile. What is never accepted is
# a tool identity dressed as a co-author.
TRAILER = re.compile(r"^Co-Authored-By:\s*(.*)$", re.I | re.M)
AGENT_FORM = re.compile(r"^[A-Za-z][A-Za-z0-9 .+/-]* \([^()]+\) & Joakimpalm-Zen$")
HUMAN_FORM = re.compile(r"^([^<>]+?)\s*<([^<>@\s]+@[^<>@\s]+)>$")
TOOL_IDENTITY = re.compile(
    r"anthropic\.com|openai\.com|Copilot@users\.noreply\.github\.com|"
    r"\b(claude|codex|copilot|chatgpt|gpt-?\d|gemini|cursor|devin|aider)\b", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def problems(text: str) -> list[str]:
    out = []
    for rx, what in FORBIDDEN:
        for m in rx.finditer(text):
            line = text[text.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0]
            out.append(f"{what}: {line.strip()[:120]}")
    trailer_lines = set()
    for m in TRAILER.finditer(text):
        body = m.group(1).strip()
        trailer_lines.add(m.group(0).strip())
        if AGENT_FORM.match(body):
            continue
        h = HUMAN_FORM.match(body)
        if h and not TOOL_IDENTITY.search(body):
            continue  # a person, credited the way GitHub links a profile
        out.append("Co-Authored-By: an agent signs '<Agent> (<Model>) & Joakimpalm-Zen', "
                   f"a person 'Name <email>'; got: {body[:120]}")
    # e-mail addresses anywhere else are identifiers this repository does not
    # publish (a human co-author trailer is the one accepted place)
    for line in text.split("\n"):
        if line.strip() in trailer_lines:
            continue
        if EMAIL.search(line):
            out.append(f"e-mail address: {line.strip()[:120]}")
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
        "Co-Authored-By: Copilot <175728472+Copilot@users.noreply.github.com>\n",
        "Co-Authored-By: Claude <claude@example.com>\n",
        "contact me at someone@example.com\n",
    ]
    good = [
        "Fix thing\n\nCo-Authored-By: Claude Code (Fable 5.1) & Joakimpalm-Zen\n",
        "Fix thing\n\nCo-Authored-By: Codex (GPT-5) & Joakimpalm-Zen\n",
        "Fix thing\n\nCo-Authored-By: Peter Saverman <12345+peter@users.noreply.github.com>\n",
        "Fix thing\n\nCo-Authored-By: Peter Saverman <peter@example.org>\n",
        "Merge pull request #34 from Joakimpalm-Zen/release-0.4.9\n",
        "docs: link https://github.com/Joakimpalm-Zen/xyntetik-runner/releases\n",
        "server: session_reconnection keeps the slot; see session_persistence.md\n",
    ]
    ok = all(problems(b) for b in bad) and not any(problems(g) for g in good)
    if not ok:
        for b in bad:
            if not problems(b): print("  missed:", b.strip()[:80])
        for g in good:
            if problems(g): print("  false positive:", g.strip()[:80], problems(g))
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
              "An agent strips the tool's default trailer and signs "
              "'Co-Authored-By: <Agent> (<Model>) & Joakimpalm-Zen'; a person "
              "may sign 'Co-Authored-By: Name <email>'.")
        return 1
    if a.range or a.message_file or a.text:
        print("commit hygiene ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
