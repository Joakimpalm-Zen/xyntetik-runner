#!/usr/bin/env python3
"""A new root test or fuzz harness must have an execution entry in CI."""
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
    print(f"CI roster: {len(actual)} root test files, {len(targets)} fuzz targets accounted for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
