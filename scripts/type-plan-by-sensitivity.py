#!/usr/bin/env python3
"""Write a --type-plan that demotes the tensors that matter least (R4.7.3).

A hand-written plan picks which tensors go to the lower type by a rule of
thumb (the deepest blocks' experts, say). This picks them by measurement:
`runner --quant-error` reports, per tensor and per type, the bytes and the
squared round-trip error the rewrite's own writer would leave, weighted by an
importance matrix when one is given, which makes it the expected squared
output perturbation of that matrix. Candidate tensors (those whose name holds
--match) start at --high; whole groups of them (a block's candidates, or one
tensor) are demoted to --low in order of the error they add per byte they
save, the cheapest first, until the target is met.

    runner -m MODEL.gguf --quant-error report.jsonl --quant q4_0,q3_k --imatrix IMX
    scripts/type-plan-by-sensitivity.py report.jsonl --match _exps \\
        --high q4_0 --low q3_k --groups 11 > plan.json
    runner -m MODEL.gguf --rewrite out.gguf --type-plan plan.json --imatrix IMX

The target is --groups K (demote K groups) or --save-bytes N (demote until
at least N bytes are saved). Tensors outside --match keep --default (keep,
the source type, unless given). The ranking is a proxy: measure the result
against the reference (scripts/gold-logits.py) before trusting a plan.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

BLOCK = re.compile(r"^blk\.(\d+)\.")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def groups_of(rows, match, high, low, by):
    """Candidate tensors grouped, each group's added error and saved bytes."""
    groups = {}
    for r in rows:
        if match not in r["tensor"]:
            continue
        h, l = r["types"].get(high), r["types"].get(low)
        if h is None or l is None:
            sys.exit(f"error: {r['tensor']} has no {high if h is None else low} entry "
                     "(the type does not fit its rows, or the report did not cost it)")
        m = BLOCK.match(r["tensor"])
        key = r["tensor"] if by == "tensor" or not m else f"blk.{m.group(1)}"
        g = groups.setdefault(key, {"group": key, "tensors": [], "added_error": 0.0,
                                    "saved_bytes": 0, "weighted": set()})
        g["tensors"].append(r["tensor"])
        g["added_error"] += l["error"] - h["error"]
        g["saved_bytes"] += h["bytes"] - l["bytes"]
        g["weighted"].add(r["weighted"])
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", help="runner --quant-error output (JSON lines)")
    ap.add_argument("--match", default="", help="candidate tensors: names holding this")
    ap.add_argument("--high", required=True, help="the candidates' type unless demoted")
    ap.add_argument("--low", required=True, help="the type a demoted group gets")
    ap.add_argument("--group", choices=("block", "tensor"), default="block",
                    help="demote a block's candidates together (default) or tensor by tensor")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--groups", type=int, help="demote this many groups")
    target.add_argument("--save-bytes", type=int, help="demote until this many bytes are saved")
    ap.add_argument("--default", default="keep", help="the plan's default for other tensors")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="rank weighted and unweighted errors together")
    a = ap.parse_args()

    groups = groups_of(load(a.report), a.match, a.high, a.low, a.group)
    if not groups:
        sys.exit(f"error: no tensor in the report matches '{a.match}'")
    kinds = set().union(*(g["weighted"] for g in groups))
    if len(kinds) > 1 and not a.allow_mixed:
        sys.exit("error: some candidates' errors are imatrix-weighted and some are not, "
                 "and the two do not compare; give an imatrix that covers them, narrow "
                 "--match, or pass --allow-mixed")
    for g in groups:
        if g["saved_bytes"] <= 0:
            sys.exit(f"error: demoting {g['group']} to {a.low} saves no bytes")
        g["cost"] = g["added_error"] / g["saved_bytes"]
    order = sorted(groups, key=lambda g: (g["cost"], g["group"]))
    demoted, saved = [], 0
    for g in order:
        if a.groups is not None and len(demoted) >= a.groups:
            break
        if a.save_bytes is not None and saved >= a.save_bytes:
            break
        demoted.append(g)
        saved += g["saved_bytes"]
    if a.groups is not None and len(demoted) < a.groups:
        sys.exit(f"error: only {len(demoted)} groups to demote, {a.groups} asked")
    if a.save_bytes is not None and saved < a.save_bytes:
        sys.exit(f"error: demoting every group saves {saved} bytes, {a.save_bytes} asked")

    rules = [{"match": t, "type": a.low} for g in demoted for t in sorted(g["tensors"])]
    if a.match:
        rules.append({"match": a.match, "type": a.high})
    # no --match: every tensor the report costs is a candidate, so the ones
    # not demoted take --high, which is then the plan's default
    plan = {"default": a.default if a.match else a.high, "rules": rules}
    json.dump(plan, sys.stdout, indent=1)
    sys.stdout.write("\n")
    print(f"demoted {len(demoted)} of {len(groups)} groups to {a.low}, saving {saved} bytes: "
          + ", ".join(g["group"] for g in demoted), file=sys.stderr)


if __name__ == "__main__":
    main()
