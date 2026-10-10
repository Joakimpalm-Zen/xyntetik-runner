#!/usr/bin/env python3
"""Reject empty, failed or (for mandatory lanes) skipped JUnit executions."""
import argparse
import json
import xml.etree.ElementTree as ET


def counts(path):
    cases = list(ET.parse(path).iter("testcase"))
    return {"tests": len(cases), "skipped": sum(c.find("skipped") is not None for c in cases),
            "failed": sum(c.find("failure") is not None or c.find("error") is not None for c in cases)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-skips", action="store_true")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    bad = False
    for path in args.paths:
        c = counts(path)
        print(path, json.dumps(c))
        bad |= not c["tests"] or bool(c["failed"]) or (args.no_skips and bool(c["skipped"]))
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
