#!/usr/bin/env python3
"""A release's CUDA gate on one card, in one run: the smoke, then the probe.

    python scripts/release-cuda-gate.py --binary ./runner.exe --model qwen3-0.6b-q8_0.gguf \
        --version 1.8.0 [--card rtx3070] [--date 2026-10-30]

Runs scripts/cuda-smoke.py against the release binary (--expect-version
VERSION), then scripts/cuda-act-probe.py against the PTX that binary
embeds (src/kernels_ptx.h of this checkout), so a shared card pauses its
queue once for both. Writes

    docs/compat-reports/cuda-smoke-VERSION-DATE-CARD.json
    docs/compat-reports/cuda-act-probe-VERSION-DATE-CARD.json

with the home directory written as ~/ in both. Exit 0 only when both pass;
the first nonzero exit of the two otherwise, after both have run.
"""
import argparse
import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def redact_home(path):
    """The home directory as ~/ in a written report, in either slash style."""
    home = os.path.expanduser("~")
    text = open(path, encoding="utf-8").read()
    for form in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
        text = text.replace(form, "~")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--card", default="rtx3070")
    ap.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"))
    a = ap.parse_args()
    reports = os.path.join(ROOT, "docs", "compat-reports")
    smoke = os.path.join(reports, f"cuda-smoke-{a.version}-{a.date}-{a.card}.json")
    probe = os.path.join(reports, f"cuda-act-probe-{a.version}-{a.date}-{a.card}.json")
    steps = [
        ("smoke", [sys.executable, os.path.join(ROOT, "scripts", "cuda-smoke.py"), "--binary", a.binary,
                   "--model", a.model, "--expect-version", a.version, "--out", smoke], smoke),
        ("probe", [sys.executable, os.path.join(ROOT, "scripts", "cuda-act-probe.py"),
                   "--ptx", os.path.join(ROOT, "src", "kernels_ptx.h"), "--out", probe], probe),
    ]
    first_bad = 0
    for name, argv, out in steps:
        rc = subprocess.run(argv).returncode
        if os.path.exists(out):
            redact_home(out)
        print(f"release-cuda-gate: {name} exit {rc} ({os.path.relpath(out, ROOT)})")
        if rc and not first_bad:
            first_bad = rc
    print("release-cuda-gate: " + ("both passed" if not first_bad else "do not release"))
    return first_bad


if __name__ == "__main__":
    sys.exit(main())
