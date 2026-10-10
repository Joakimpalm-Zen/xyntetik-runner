#!/usr/bin/env python3
"""Run mandatory device checks and bind their evidence to the exact candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["cuda", "metal"], required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(".build/hardware"))
    args = ap.parse_args()
    os.chdir(ROOT)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if sha != args.commit or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SystemExit("hardware evidence requires the exact clean tracked candidate")
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    gate = ROOT / ("test-gpu-identity.exe" if sys.platform == "win32" else "test-gpu-identity")
    args.out.mkdir(parents=True, exist_ok=True)
    caps = json.loads(subprocess.check_output([str(exe), "--caps"], text=True, timeout=30))
    if (caps.get("gpu") or {}).get("backend") != args.backend:
        raise SystemExit("required device/backend is unavailable; not a pass")
    record = dict(schema="xyntetik.runner.candidate-hardware.v1", commit=sha,
                  binary_sha256=digest(exe), gate_sha256=digest(gate),
                  model_sha256=digest(args.model), backend=args.backend, caps=caps,
                  scope="gpu-identity (additional backend and PTX gates are separate)", steps=[], result="fail")
    try:
        for name, model, batch in [("fixture-prefill", ROOT / "test.gguf", "32"),
                                    ("model-prefill", args.model, "32"),
                                    ("model-decode", args.model, "1")]:
            start = time.monotonic()
            p = subprocess.run([str(gate), str(model.resolve()), "0", batch], capture_output=True, text=True, timeout=600)
            log = p.stdout + p.stderr
            (args.out / (name + ".log")).write_text(log)
            passed = p.returncode == 0 and "gpu-identity: ok" in log and "skipped" not in log.lower()
            record["steps"].append(dict(name=name, seconds=time.monotonic()-start, exit_code=p.returncode, passed=passed))
            if not passed:
                raise SystemExit(f"{name}: failed or skipped; see {args.out}")
        record["result"] = "pass"
    finally:
        (args.out / "evidence.json").write_text(json.dumps(record, indent=2) + "\n")
    print("candidate hardware checks passed; this does not replace family-specific tolerance gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
