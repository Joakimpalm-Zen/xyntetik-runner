#!/usr/bin/env python3
"""Time a CI command, bound its process tree, and retain its result on failure."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def stop(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default=".build/ci")
    ap.add_argument("name")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd or args.timeout <= 0 or Path(args.name).name != args.name:
        ap.error("a plain name, positive timeout and command are required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    rc = 127
    try:
        proc = subprocess.Popen(cmd, start_new_session=os.name != "nt")
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop(proc)
            rc = 124
    except OSError as e:
        print(f"cannot run {cmd[0]}: {e}", flush=True)
    elapsed = time.monotonic() - start
    record = dict(name=args.name, command=cmd, seconds=elapsed, exit_code=rc,
                  timed_out=timed_out, commit=os.environ.get("GITHUB_SHA"))
    (out / f"{args.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    message = f"{args.name}: {elapsed:.2f}s, exit {rc}, timeout={timed_out}"
    print(message, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(message + "\n\n")
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    raise SystemExit(main())
