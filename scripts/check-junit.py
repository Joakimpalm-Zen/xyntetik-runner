#!/usr/bin/env python3
"""Reject empty, failed or (for mandatory lanes) skipped JUnit executions.

--no-skips-matching REGEX rejects only the skips whose reason matches (a lane
whose point is a device rejects "no CUDA device" but not "openssl not
available"); --allow-skip REGEX exempts a reason that is correct on that lane."""
import argparse
import json
import re
import xml.etree.ElementTree as ET


# pytest writes where a skip was raised before its reason, "FILE:LINE: reason".
# A lane's pattern is for the reason: with the location left in, a file with
# "gpu" or "cuda" in its name failed a device lane on any skip at all.
LOCATION = re.compile(r"^(?:[A-Za-z]:)?[^:\n]*:\d+: ")


def skip_reasons(path):
    out = []
    for c in ET.parse(path).iter("testcase"):
        s = c.find("skipped")
        if s is not None:
            out.append((c.get("classname", "") + "::" + c.get("name", ""),
                        (s.get("message") or "") + " " + LOCATION.sub("", s.text or "", count=1)))
    return out


def counts(path):
    cases = list(ET.parse(path).iter("testcase"))
    return {"tests": len(cases), "skipped": sum(c.find("skipped") is not None for c in cases),
            "failed": sum(c.find("failure") is not None or c.find("error") is not None for c in cases)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-skips", action="store_true")
    ap.add_argument("--no-skips-matching", metavar="REGEX")
    ap.add_argument("--allow-skip", metavar="REGEX")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    bad = False
    for path in args.paths:
        c = counts(path)
        print(path, json.dumps(c))
        bad |= not c["tests"] or bool(c["failed"]) or (args.no_skips and bool(c["skipped"]))
        if args.no_skips_matching:
            for case, why in skip_reasons(path):
                if re.search(args.no_skips_matching, why) and not (
                        args.allow_skip and re.search(args.allow_skip, why)):
                    print(f"  skipped on this lane: {case}: {why.strip()[:200]}")
                    bad = True
    return int(bad)


if __name__ == "__main__":
    raise SystemExit(main())
