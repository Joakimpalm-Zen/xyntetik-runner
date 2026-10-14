#!/usr/bin/env python3
"""A new root test, fuzz harness or device target must have an execution entry."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    make = (ROOT / "Makefile").read_text()
    workflows = "\n".join(p.read_text() for p in (ROOT / ".github/workflows").glob("*.yml"))
    recipes = "\n".join(line for line in make.splitlines()
                         if line.startswith("\t") and not line.lstrip().startswith("#"))
    executable_workflow = "\n".join(line for line in workflows.splitlines()
                                    if not line.lstrip().startswith("#"))
    named = set(re.findall(r"tests/test_[\w]+\.py", recipes + executable_workflow))
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_*.py")}
    if not actual:
        raise SystemExit("root test inventory is empty; no execution coverage was checked")
    missing = sorted(actual - named)
    if missing:
        raise SystemExit("root tests have no Makefile/workflow entry: " + ", ".join(missing))
    flat = make.replace("\\\n", " ")
    match = re.search(r"^FUZZ_TARGETS\s*=\s*(.*)$", flat, re.M)
    if not match:
        raise SystemExit("FUZZ_TARGETS is missing")
    targets = set(match[1].split())
    harnesses = {p.stem.removeprefix("fuzz_") for p in (ROOT / "tests/fuzz").glob("fuzz_*.c")}
    if targets != harnesses:
        raise SystemExit(f"fuzz roster mismatch: not run={sorted(harnesses-targets)}, missing={sorted(targets-harnesses)}")
    # The workflow groups must cover the same target set without duplicates.
    groups = re.findall(r"^\s+targets:\s*([a-z_ ]+)$", workflows, re.M)
    selected = [t for group in groups for t in group.split()]
    if set(selected) != targets or len(selected) != len(targets):
        raise SystemExit("CI fuzz groups omit or duplicate a registered target")
    devices = device_roster(make, flat)
    print(f"CI roster: {len(actual)} root test files, {len(targets)} fuzz targets, "
          f"{devices} device targets accounted for")
    return 0


def device_roster(make, flat):
    """Every test-cuda-* target is a step of scripts/cuda-gate.sh and every
    test-metal-* target is called by test-metal, unless DEVICE_OPT_IN names it.
    A device target no lane runs is a gate nothing runs."""
    defined = set(re.findall(r"^(test-(?:cuda|metal)-[\w-]+):", make, re.M))
    if not defined:
        return 0
    m = re.search(r"^DEVICE_OPT_IN\s*=\s*(.*)$", flat, re.M)
    opt_in = set(m[1].split()) if m else set()
    run = set()
    gate = ROOT / "scripts/cuda-gate.sh"
    if gate.exists():
        steps = re.search(r'^DEVICE_STEPS="([^"]*)"', gate.read_text(), re.M)
        if not steps:
            raise SystemExit("scripts/cuda-gate.sh has no DEVICE_STEPS list")
        run |= {pair.split(":", 1)[1] for pair in steps[1].split()}
    recipe = re.search(r"^test-metal:.*\n((?:\t.*\n)*)", make, re.M)
    if recipe:
        run |= set(re.findall(r"\$\(MAKE\)[^\n]*?\b(test-metal-[\w-]+)", recipe[1]))
    unknown = sorted((opt_in | run) - set(re.findall(r"^([\w-]+):", make, re.M)))
    if unknown:
        raise SystemExit("device lanes name targets the Makefile does not define: "
                         + ", ".join(unknown))
    missing = sorted(defined - run - opt_in)
    if missing:
        raise SystemExit("device targets no lane runs (add to test-metal, to "
                         "DEVICE_STEPS in scripts/cuda-gate.sh, or to DEVICE_OPT_IN "
                         "with the input they need): " + ", ".join(missing))
    return len(defined)


if __name__ == "__main__":
    raise SystemExit(main())
