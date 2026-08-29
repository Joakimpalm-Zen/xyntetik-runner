#!/usr/bin/env python3
"""CUDA smoke gate: run the checks that only real GPU hardware can answer.

CI has no GPU. src/cuda.c is compiled on three platforms and executed by
none, which is how v0.4.2 shipped an integrated-device probe that read
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT (17) instead of
CU_DEVICE_ATTRIBUTE_INTEGRATED (18) and reported every display-attached
consumer GPU as unified memory. This script is the gate that would have
caught it, in seconds, on any box with a card in it.

It is deliberately host-agnostic: run it on a discrete desktop GPU, on a
headless datacenter card, or on a unified-memory device like a DGX Spark.
The assertions are INVARIANTS rather than expected constants, so the same
script passes on all three and fails on an incoherent one.

The central one: `unified_memory` is a claim about whether VRAM and system
RAM are one pool, so it must agree with the sizes actually reported. A
device claiming unified memory while its VRAM is plainly a different pool
from its RAM is the v0.4.2 bug, and it fails here regardless of which
attribute number the driver was asked for.

Usage:
    python3 scripts/cuda-smoke.py --binary ./runner --model model.gguf \
        --out cuda-smoke.json [--expect-version 0.4.3]

Exit code 0 means every check passed. Anything else means do not release.
"""

import argparse
import json
import re
import subprocess
import sys

# A device whose VRAM is at least this fraction of system RAM is plausibly
# reporting one shared pool. A DGX Spark reports the two as exactly equal.
# A discrete card sits far below (an 8 GB 3070 in a 16 GB host is ~0.50).
UNIFIED_POOL_RATIO = 0.85


def run(cmd, timeout):
    """Run a command, returning (exit_code, combined_output)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace")


class Checks:
    def __init__(self):
        self.results = []

    def add(self, name, ok, detail, severity="fail"):
        # severity "warn" records an observation that is ambiguous rather
        # than wrong: it is reported but does not gate the release.
        self.results.append({"check": name, "ok": bool(ok),
                             "severity": severity, "detail": detail})
        return ok

    def failed(self):
        return [r for r in self.results
                if not r["ok"] and r["severity"] == "fail"]

    def warned(self):
        return [r for r in self.results
                if not r["ok"] and r["severity"] == "warn"]


def check_caps(c, caps, expect_version):
    gpu = caps.get("gpu") or {}
    backend = gpu.get("backend")

    # The probe that runs first on every CUDA machine: if the driver load or
    # the optional-symbol resolution regressed, there is no gpu block at all.
    c.add("cuda_backend_present", backend == "cuda",
          "caps.gpu.backend = %r (expected 'cuda'; absent means the CUDA "
          "driver never initialised)" % (backend,))

    if expect_version:
        c.add("version_matches", caps.get("version") == expect_version,
              "caps.version = %r, expected %r"
              % (caps.get("version"), expect_version))

    ram = caps.get("ram_bytes") or 0
    vram = gpu.get("vram_bytes") or 0
    unified = gpu.get("unified_memory")
    ratio = (vram / ram) if ram else 0.0
    same_pool = ratio >= UNIFIED_POOL_RATIO

    c.add("unified_memory_is_boolean", isinstance(unified, bool),
          "caps.gpu.unified_memory = %r" % (unified,))

    # THE regression guard. Claiming one pool while reporting two sizes that
    # are plainly not one pool is incoherent, and it is what shipped in
    # v0.4.2: the offload budget then gets clamped to available system RAM
    # on hardware that has its own dedicated VRAM.
    c.add("unified_memory_agrees_with_pool_sizes",
          not (unified is True and not same_pool),
          "unified_memory=%r with vram_bytes=%d and ram_bytes=%d "
          "(vram/ram = %.3f, one pool expected >= %.2f)"
          % (unified, vram, ram, ratio, UNIFIED_POOL_RATIO))

    # The ambiguous direction. A discrete card can coincidentally sit near
    # host RAM size, so this observes rather than gates.
    c.add("unified_flag_not_suspiciously_false",
          not (unified is False and same_pool),
          "unified_memory=%r with vram/ram = %.3f%s"
          % (unified, ratio,
             "; that looks like one pool, confirm by hand on this host"
             if (unified is False and same_pool) else ""),
          severity="warn")

    c.add("vram_reported", vram > 0, "vram_bytes = %d" % (vram,))

    quants = set(caps.get("quants") or [])
    gq = set(caps.get("gpu_quants") or [])
    c.add("gpu_quants_present", len(gq) > 0,
          "gpu_quants has %d entries" % (len(gq),))
    missing = sorted(gq - quants)
    c.add("gpu_quants_subset_of_quants", not missing,
          "every gpu_quant is loadable" if not missing else
          "gpu_quants not advertised as loadable: %s" % (missing,))
    return gpu


def check_generation(c, out, code, unified_claim):
    c.add("generation_exit_zero", code == 0,
          "runner exited %d" % (code,))

    engaged = "gpu: CUDA backend on" in out
    c.add("gpu_engaged_for_generation", engaged,
          "load output %s the CUDA backend line"
          % ("carries" if engaged else "does NOT carry",))

    m = re.search(r"gen:\s*(\d+)\s*tok", out)
    ntok = int(m.group(1)) if m else 0
    c.add("tokens_generated", ntok > 0,
          "generated %d tokens" % (ntok,))

    split = re.search(r"G=(\d+)/(\d+)", out)
    if split:
        c.add("layers_offloaded", int(split.group(1)) > 0,
              "gpu-split G=%s/%s" % (split.group(1), split.group(2)))

    # Cross-check the load-time notice against what --caps claimed. The two
    # come from different call sites and disagreeing is itself the bug.
    notice = "unified memory" in out.lower()
    c.add("unified_notice_matches_caps", notice == bool(unified_claim),
          "load output %s the unified-memory notice while caps said %r"
          % ("prints" if notice else "omits", unified_claim))
    return ntok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", help="write a JSON report here")
    ap.add_argument("--expect-version")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    c = Checks()

    code, caps_out = run([args.binary, "--caps"], args.timeout)
    if code != 0:
        print("FAIL: --caps exited %d\n%s" % (code, caps_out))
        return 1
    try:
        caps = json.loads(caps_out)
    except ValueError as e:
        print("FAIL: --caps did not produce JSON (%s)\n%s" % (e, caps_out))
        return 1

    gpu = check_caps(c, caps, args.expect_version)

    code, gen_out = run([args.binary, "-m", args.model, "-p", args.prompt,
                         "-n", str(args.tokens), "--temp", "0"], args.timeout)
    check_generation(c, gen_out, code, gpu.get("unified_memory"))

    report = {
        "host": {"os": caps.get("os"), "arch": caps.get("arch"),
                 "ram_bytes": caps.get("ram_bytes")},
        "gpu": gpu,
        "runner_version": caps.get("version"),
        "checks": c.results,
        "generation_output": gen_out,
        "passed": not c.failed(),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    for r in c.results:
        mark = "ok  " if r["ok"] else ("WARN" if r["severity"] == "warn"
                                       else "FAIL")
        print("%s %-38s %s" % (mark, r["check"], r["detail"]))

    gpu_name = gpu.get("name", "unknown")
    if c.failed():
        print("\nFAILED on %s: %d check(s). Do not release."
              % (gpu_name, len(c.failed())))
        return 1
    print("\nPASS on %s (%d checks, %d warning(s))."
          % (gpu_name, len(c.results), len(c.warned())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
