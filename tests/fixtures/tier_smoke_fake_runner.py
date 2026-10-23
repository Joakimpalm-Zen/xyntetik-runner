#!/usr/bin/env python3
"""A stand-in for the runner binary in tests/test_tier_smoke.py.

It answers the three calls scripts/tier-smoke.py makes, in the real
binary's shapes: `--caps` (one JSON object), `-m FILE --adapt-info` (one JSON
line) and a one-shot `-m FILE -p PROMPT` (the prompt echoed, then the
continuation, on stdout; the backend banner on stderr unless `--gpu off`).

FAKE_RUNNER_TEXT replaces the continuation; FAKE_RUNNER_FALLBACK=1 makes a
GPU run decline the model and run on the CPU, as a Metal build does for a
layout it has no kernel for.
"""

import json
import os
import sys

argv = sys.argv[1:]

if "--caps" in argv:
    print(json.dumps({"version": "9.8.7", "os": "macos", "arch": "arm64",
                      "cpu_cores": 8,
                      "gpu": {"backend": "metal", "name": "Fake M"}}))
    sys.exit(0)

if "--adapt-info" in argv:
    print(json.dumps({"architecture": "llama",
                      "adapter_serving": {"supported": True, "reason": None,
                                          "device_kernels": "metal"},
                      "training": {"supported": True, "reason": None,
                                   "host": "cpu"}}))
    sys.exit(0)

if "-p" in argv:
    prompt = argv[argv.index("-p") + 1]
    gpu_off = "--gpu" in argv and argv[argv.index("--gpu") + 1] == "off"
    if not gpu_off:
        if os.environ.get("FAKE_RUNNER_FALLBACK"):
            print("gpu: this model's MoE router has no Metal kernel - using CPU",
                  file=sys.stderr)
        else:
            print("gpu: Metal backend on Fake M (zero-copy weights)", file=sys.stderr)
    print("loaded fake | llama | 2 layers | ctx 512 | 4 threads | 0.01s",
          file=sys.stderr)
    text = os.environ.get("FAKE_RUNNER_TEXT",
                          "\n\n1. Mercury: the smallest planet and the closest to the Sun.\n"
                          "2. Venus: the second planet from the Sun and the hottest.")
    sys.stdout.write(prompt + text + "\n")
    print("\nprompt: 6 tok, 100.00 tok/s | gen: 8 tok, 50.00 tok/s", file=sys.stderr)
    sys.exit(0)

print("fake runner: unexpected arguments %r" % (argv,), file=sys.stderr)
sys.exit(2)
