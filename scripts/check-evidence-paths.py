#!/usr/bin/env python3
"""Check locally referenced certification evidence even on docs-only PRs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evidence_paths(value):
    if isinstance(value, str) and value.startswith("docs/"):
        yield value.split("#", 1)[0]
    elif isinstance(value, dict):
        for v in value.values():
            yield from evidence_paths(v)
    elif isinstance(value, list):
        for v in value:
            yield from evidence_paths(v)


def main():
    bad = []
    for name in ("docs/device-evidence.json", "tests/compatibility/coverage.json"):
        doc = json.loads((ROOT / name).read_text())
        for path in evidence_paths(doc):
            if not (ROOT / path).exists():
                bad.append(f"{name}: missing {path}")
    if bad:
        raise SystemExit("\n".join(bad))
    print("evidence paths exist (not a claim of fresh hardware verification)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
