#!/usr/bin/env python3
"""Minimize successful discoveries; keep diagnostics out of the input corpus."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()
    for target in args.targets:
        if Path(target).name != target:
            ap.error("target must be a plain name")
        corpus = Path("fuzz-corpus") / target
        corpus.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fuzz-merge-") as td:
            tmp = Path(td)
            inputs, minimal = tmp / "inputs", tmp / "minimal"
            inputs.mkdir(); minimal.mkdir()
            for p in corpus.iterdir():
                if p.is_file() and not p.name.startswith(("crash-", "asan.", "ubsan.")):
                    shutil.copyfile(p, inputs / p.name)
            subprocess.run([str(Path("fuzz-" + target).resolve()), "-merge=1", str(minimal),
                            str(inputs), str(Path("tests/fuzz/corpus") / target)],
                           check=True, timeout=180)
            # Only replace input files after a successful merge. Crash inputs
            # and sanitizer logs remain available in the uploaded artifact.
            for p in corpus.iterdir():
                if p.is_file() and not p.name.startswith(("crash-", "asan.", "ubsan.")):
                    p.unlink()
            for p in minimal.iterdir():
                shutil.copyfile(p, corpus / p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
