#!/usr/bin/env python3
"""Replay corpus with source coverage; export reachability, not a pass percentage."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--clang", default="clang")
    args = ap.parse_args()
    profdata = subprocess.check_output([args.clang, "-print-prog-name=llvm-profdata"], text=True).strip()
    cov = subprocess.check_output([args.clang, "-print-prog-name=llvm-cov"], text=True).strip()
    if not shutil.which(profdata) or not shutil.which(cov):
        raise SystemExit("matching llvm-profdata and llvm-cov are required")
    for target in args.targets:
        out = Path(".build/fuzz-coverage") / target
        out.mkdir(parents=True, exist_ok=True)
        binary = Path("fuzz-" + target)
        binary.unlink(missing_ok=True)  # this generated binary needs different flags
        subprocess.run(["make", str(binary), "FUZZ_CLANG=" + args.clang,
                        "FUZZ_EXTRA_FLAGS=-fprofile-instr-generate -fcoverage-mapping"], check=True)
        corpus = Path("fuzz-corpus") / target
        corpus.mkdir(parents=True, exist_ok=True)
        # Keep report/crash files out of coverage replay inputs.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="coverage-corpus-") as td:
            for p in corpus.iterdir():
                if p.is_file() and not p.name.startswith(("crash-", "asan.", "ubsan.")):
                    shutil.copyfile(p, Path(td) / p.name)
            env = dict(os.environ, LLVM_PROFILE_FILE=str(out / "replay.profraw"),
                       ASAN_OPTIONS="allocator_may_return_null=1:max_allocation_size_mb=1024:log_path=" + str(out / "asan"),
                       UBSAN_OPTIONS="halt_on_error=1:log_path=" + str(out / "ubsan"))
            subprocess.run([str(binary.resolve()), "-runs=0", "-timeout=25", "-rss_limit_mb=2048", "-malloc_limit_mb=1024", td,
                            str(Path("tests/fuzz/corpus") / target)], env=env, check=True, timeout=180)
        profile = out / "replay.profdata"
        subprocess.run([profdata, "merge", "-sparse", str(out / "replay.profraw"), "-o", str(profile)], check=True)
        with (out / "coverage.json").open("w") as f:
            subprocess.run([cov, "export", str(binary), "-instr-profile=" + str(profile)], stdout=f, check=True)
        with (out / "summary.txt").open("w") as f:
            subprocess.run([cov, "report", str(binary), "-instr-profile=" + str(profile)], stdout=f, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
