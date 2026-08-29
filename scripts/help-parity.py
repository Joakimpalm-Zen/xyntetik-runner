#!/usr/bin/env python3
"""Mechanical parity between `runner --help` and the README's option reference.

AGENTS.md rule 5 treats the README as a tested public interface rather than a
historical summary, and the failure mode it exists to catch is silent drift: a
flag added to the binary and never documented, or documented once and then
renamed or removed. Both directions are checked here because both have
happened, and neither is visible from inside a single file.

The check is deliberately shallow. It compares the SET of long options, not
their descriptions, because prose wording legitimately differs between a
terse help line and a README table. A flag that appears in either place and
not the other is the whole signal.

Usage:
    python3 scripts/help-parity.py --binary ./runner --readme README.md
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Options intentionally absent from one side. Keep this list SHORT and give
# every entry a reason: an allowlist that grows without argument is how a
# parity gate stops being a gate.
# NB: set(), not {} -- an empty brace literal is a dict and every set
# operation below would raise on it.
ALLOW_MISSING_FROM_README = set()   # add with a reason, or fix the README
ALLOW_MISSING_FROM_HELP = set()     # add with a reason, or fix the help text

OPTION = re.compile(r"--[a-z][a-z0-9-]+")


def help_options(binary):
    p = subprocess.run([str(Path(binary).resolve()), "--help"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=30)
    if p.returncode != 0:
        raise SystemExit(f"--help exited {p.returncode}")
    text = p.stdout.decode("utf-8", "replace")
    opts = set()
    for line in text.splitlines():
        # Only the option COLUMN declares a flag. Prose in a help epilogue can
        # mention `--gpu off` without that being a declaration site, and
        # treating it as one would make the gate demand README entries for
        # phrases rather than for flags.
        #
        # A declaration line is two leading spaces, the flag spec, then two or
        # more spaces before the description. Take EVERY long option out of
        # the spec, not just the first: several lines declare a pair
        # (`--reserve-vram P / --reserve-ram P`), and reading only the head of
        # those lines silently under-reports what the binary offers, which is
        # the opposite of what this gate is for.
        # Column alignment in the help text is not uniform -- some lines put
        # a single space between the flag spec and its description
        # (`--quantize OUT rewrite the model...`), so a two-space column rule
        # silently drops them. Any line opening with two spaces and a dash is
        # a declaration line, and every long option on it is counted.
        #
        # That over-counts when a description names another flag, and the
        # over-count is the safe direction: it can only demand that the README
        # also mention a flag the binary really has.
        if re.match(r"  -", line):
            opts.update(OPTION.findall(line))
    return opts, text


def readme_options(readme):
    """Flags the README documents, split by how strongly it claims them.

    `mentioned` is every long option anywhere in the file. `declared` is only
    those heading a row of the command-reference table, which is the README's
    structured claim that a flag is runner's own.

    The distinction matters and a proximity heuristic could not make it: the
    README shows `--runner ./runner --reference /path/to/llama-server`, where
    `--reference` belongs to scripts/compat_matrix.py and merely shares a line
    with the word runner. Anchoring on the table makes that unambiguous.
    """
    text = Path(readme).read_text(encoding="utf-8")
    mentioned = set(OPTION.findall(text))
    declared = set()
    for line in text.splitlines():
        m = re.match(r"\|\s*`(--[a-z][a-z0-9-]+)", line)
        if m:
            declared.add(m.group(1))
    return mentioned, declared


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--readme", required=True)
    args = ap.parse_args(argv)

    help_opts, help_text = help_options(args.binary)
    if not help_opts:
        raise SystemExit("help-parity: parsed zero options out of --help; "
                         "the option-column format changed and this gate is "
                         "no longer checking anything")
    mentioned, declared = readme_options(args.readme)
    if not declared:
        raise SystemExit("help-parity: parsed zero options out of the README "
                         "command-reference table; its format changed and this "
                         "gate is no longer checking that direction")

    # help -> README: a flag the binary offers and the README never mentions
    # at all. Anywhere in the file counts here, because documenting a flag in
    # prose is documenting it.
    missing_doc = (help_opts - mentioned) - ALLOW_MISSING_FROM_README
    # README -> help: only the table's declarations, which are the README
    # asserting a flag is runner's own. Prose may name other tools' flags.
    missing_help = (declared - help_opts) - ALLOW_MISSING_FROM_HELP

    ok = True
    for o in sorted(missing_doc):
        print(f"help-parity: {o} is in --help but not documented in the README")
        ok = False
    for o in sorted(missing_help):
        print(f"help-parity: README presents {o} as a runner flag but "
              f"--help does not list it")
        ok = False
    if not ok:
        print("help-parity: FAILED")
        return 1
    print(f"help-parity: ok ({len(help_opts)} options in parity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
