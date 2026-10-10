#!/usr/bin/env python3
"""Prove ASan and UBSan are active and fatal using independent C violations."""
import argparse
import os
import pathlib
import shlex
import subprocess
import tempfile

PROBES = {
    "ubsan": '#include <limits.h>\nint main(void) { volatile int x=INT_MAX; volatile int y=x+1; (void)y; return 0; }',
    "asan": '#include <stdlib.h>\nint main(void) { volatile int n=1; char *p=malloc(n); p[n]=1; volatile char x=p[n]; free(p); return x==1 ? 0 : 1; }',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="sanitizer-check-") as td:
        root = pathlib.Path(td)
        for name, source in PROBES.items():
            c, exe = root / (name + ".c"), root / name
            c.write_text(source)
            subprocess.run([*shlex.split(args.cc), "-O0", "-g",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
                            "-fno-omit-frame-pointer", str(c), "-o", str(exe)], check=True)
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:abort_on_error=1",
                       UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
            p = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=20)
            marker = "runtime error: signed integer overflow" if name == "ubsan" else "AddressSanitizer: heap-buffer-overflow"
            if p.returncode == 0 or marker not in p.stderr:
                raise SystemExit(f"{name} did not detect and fail its negative control:\n{p.stderr}")
            print(f"{name}: diagnosed known violation and exited {p.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
