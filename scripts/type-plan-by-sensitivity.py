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

--ladder T1,T2,...,Tn (high to low; --high H --low L is the two-rung ladder
H,L, and --mid M makes it H,M,L) demotes one rung at a time instead of
straight to the lowest type (R4.34.1): each group's (bytes, error) points
over the ladder are reduced to their lower convex hull, and the cheapest
next hull step over all groups (added error per byte saved) is taken until
the target is met. A rung off a group's hull costs more per byte than the
rungs around it and is skipped for that group; the stderr summary names
each skip. With two rungs the plan is exactly the rung-less ranking's.

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


def ladder_groups(rows, match, ladder, by):
    """Candidate tensors grouped, each group's bytes and error at every rung."""
    groups = {}
    for r in rows:
        if match not in r["tensor"]:
            continue
        costs = [r["types"].get(t) for t in ladder]
        for t, c in zip(ladder, costs):
            if c is None:
                sys.exit(f"error: {r['tensor']} has no {t} entry "
                         "(the type does not fit its rows, or the report did not cost it)")
        m = BLOCK.match(r["tensor"])
        key = r["tensor"] if by == "tensor" or not m else f"blk.{m.group(1)}"
        g = groups.setdefault(key, {"group": key, "tensors": [], "bytes": [0] * len(ladder),
                                    "error": [0.0] * len(ladder), "weighted": set(), "at": 0})
        g["tensors"].append(r["tensor"])
        for k, c in enumerate(costs):
            g["bytes"][k] += c["bytes"]
            g["error"][k] += c["error"]
        g["weighted"].add(r["weighted"])
    return list(groups.values())


def guard_rules(rows, typed, top):
    """Rules match by substring and the first that matches wins, so a rule
    naming one demoted tensor also takes every tensor whose name holds that
    name (output.weight takes each attn_output.weight, token_embd.weight takes
    per_layer_token_embd.weight). Such a tensor gets its own rule, ahead of
    every other: its rung when it is demoted too, else the top rung. Longest
    name first, so none of these takes another. src/derive.c plan_guards is
    the same. `typed` is {demoted tensor: its type}."""
    names = [r["tensor"] for r in rows]
    held = [y for y in names if any(x != y and x in y for x in typed)]
    return [{"match": y, "type": typed.get(y, top)} for y in sorted(held, key=lambda y: (-len(y), y))]


def hold_uncovered(rows, match, allow_mixed):
    """R4.34.2: when the report is imatrix-weighted, a candidate the imatrix has
    no entry for (llama.cpp's leaves out output and token_embd by default) has
    an unweighted error that does not compare with the others. It is held at the
    top rung and left out of the ranking, and named, unless --allow-mixed."""
    cand = [r for r in rows if match in r["tensor"]]
    if allow_mixed or not any(r["weighted"] for r in cand):
        return rows, []
    held = sorted(r["tensor"] for r in cand if not r["weighted"])
    return [r for r in rows if r["tensor"] not in held], held


def say_held(held, top):
    if held:
        print(f"no calibration data for {len(held)} tensor(s), held at {top} and not ranked: "
              + ", ".join(held), file=sys.stderr)


def next_step(g):
    """The group's next step down its lower convex hull: (cost, rung) with the
    least added error per byte saved from where it is, the nearer rung on a
    tie; None when no lower rung saves bytes."""
    best = None
    for j in range(g["at"] + 1, len(g["bytes"])):
        saved = g["bytes"][g["at"]] - g["bytes"][j]
        if saved <= 0:
            continue
        cost = (g["error"][j] - g["error"][g["at"]]) / saved
        if best is None or cost < best[0]:
            best = (cost, j)
    return best


def ladder_plan(a, rows, ladder, all_rows):
    """The ladder's plan: the cheapest next hull step over all groups, until
    the target is met (--groups counts the groups that leave the top rung)."""
    groups = ladder_groups(rows, a.match, ladder, a.group)
    if not groups:
        sys.exit(f"error: no tensor in the report matches '{a.match}'")
    kinds = set().union(*(g["weighted"] for g in groups))
    if len(kinds) > 1 and not a.allow_mixed:
        sys.exit("error: some candidates' errors are imatrix-weighted and some are not, "
                 "and the two do not compare; give an imatrix that covers them, narrow "
                 "--match, or pass --allow-mixed")
    for g in groups:
        if next_step(g) is None:
            sys.exit(f"error: demoting {g['group']} to {ladder[-1]} saves no bytes")
    order, saved, skips = [], 0, []
    while a.save_bytes is None or saved < a.save_bytes:
        steps = [(s[0], g["group"], s[1], g) for g in groups for s in [next_step(g)] if s]
        if a.groups is not None and len(order) >= a.groups:
            steps = [st for st in steps if st[3]["at"] > 0]   # no new group past K
            if a.save_bytes is None:
                break
        if not steps:
            break
        cost, _, j, g = min(steps, key=lambda st: (st[0], st[1]))
        if g["at"] == 0:
            order.append(g)
        if j > g["at"] + 1:
            skips.append(f"{g['group']} {ladder[g['at']]}->{ladder[j]} (skipped "
                         + ",".join(ladder[g["at"] + 1:j]) + ")")
        saved += g["bytes"][g["at"]] - g["bytes"][j]
        g["at"] = j
    if a.groups is not None and len(order) < a.groups:
        sys.exit(f"error: only {len(order)} groups to demote, {a.groups} asked")
    if a.save_bytes is not None and saved < a.save_bytes:
        sys.exit(f"error: demoting every group saves {saved} bytes, {a.save_bytes} asked")
    rules = [{"match": t, "type": ladder[g["at"]]} for g in order for t in sorted(g["tensors"])]
    rules = guard_rules(all_rows, {r["match"]: r["type"] for r in rules}, ladder[0]) + rules
    if a.match:
        rules.append({"match": a.match, "type": ladder[0]})
    plan = {"default": a.default if a.match else ladder[0], "rules": rules}
    json.dump(plan, sys.stdout, indent=1, allow_nan=False)
    sys.stdout.write("\n")
    per = ", ".join(f"{t} {sum(1 for g in order if ladder[g['at']] == t)}" for t in ladder[1:])
    print(f"ladder {','.join(ladder)}: demoted {len(order)} of {len(groups)} groups ({per}), "
          f"saving {saved} bytes; {len(skips)} rung(s) skipped as dominated"
          + "".join("\n  " + x for x in skips), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", help="runner --quant-error output (JSON lines)")
    ap.add_argument("--match", default="", help="candidate tensors: names holding this")
    ap.add_argument("--high", help="the candidates' type unless demoted")
    ap.add_argument("--low", help="the type a demoted group gets")
    ap.add_argument("--mid", help="a middle rung between --high and --low")
    ap.add_argument("--ladder", help="T1,T2,...,Tn high to low, instead of --high/--low")
    ap.add_argument("--group", choices=("block", "tensor"), default="block",
                    help="demote a block's candidates together (default) or tensor by tensor")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--groups", type=int, help="demote this many groups")
    target.add_argument("--save-bytes", type=int, help="demote until this many bytes are saved")
    ap.add_argument("--default", default="keep", help="the plan's default for other tensors")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="rank weighted and unweighted errors together")
    a = ap.parse_args()
    if a.ladder:
        if a.high or a.low or a.mid:
            sys.exit("error: give --ladder or --high/--low (with --mid), not both")
        ladder = a.ladder.split(",")
    elif a.high and a.low:
        ladder = [a.high] + ([a.mid] if a.mid else []) + [a.low]
    else:
        sys.exit("error: give --high T and --low T, or --ladder T1,...,Tn")
    if len(ladder) < 2 or len(set(ladder)) != len(ladder):
        sys.exit("error: a ladder is two or more different types, high to low")
    all_rows = load(a.report)   # the guard rules look at every costed tensor
    rows, held = hold_uncovered(all_rows, a.match, a.allow_mixed)
    say_held(held, ladder[0])
    if len(ladder) > 2:
        return ladder_plan(a, rows, ladder, all_rows)
    a.high, a.low = ladder

    groups = groups_of(rows, a.match, a.high, a.low, a.group)
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
    rules = guard_rules(all_rows, {r["match"]: r["type"] for r in rules}, a.high) + rules
    if a.match:
        rules.append({"match": a.match, "type": a.high})
    # no --match: every tensor the report costs is a candidate, so the ones
    # not demoted take --high, which is then the plan's default
    plan = {"default": a.default if a.match else a.high, "rules": rules}
    json.dump(plan, sys.stdout, indent=1, allow_nan=False)
    sys.stdout.write("\n")
    print(f"demoted {len(demoted)} of {len(groups)} groups to {a.low}, saving {saved} bytes: "
          + ", ".join(g["group"] for g in demoted), file=sys.stderr)


if __name__ == "__main__":
    main()
