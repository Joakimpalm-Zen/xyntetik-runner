#!/usr/bin/env python3
"""Where two activation traces part: the first point, in forward order, at
which trace A stops matching trace B (R4.11, RO-7).

    RUNNER_ACT_TRACE=a.jsonl RUNNER_ACT_TRACE_POS=0,7 runner -m A.gguf ... --gpu off
    RUNNER_ACT_TRACE=b.jsonl RUNNER_ACT_TRACE_POS=0,7 runner -m B.gguf ... --gpu off
    scripts/act-trace-diff.py a.jsonl b.jsonl [--rtol 1e-4] [--json out.json]

Final logits say that two forwards differ; they do not say where. A trace
records the residual stream of each listed position after the embedding
(embd), after each layer's mixer branch (mixer) and after each layer (resid),
the head's logits, and each MoE layer's selection (route). Walking those in
forward order, the first point that differs names the branch that made the
difference: a mixer point is that layer's attention (or recurrent) branch, a
resid point whose mixer point matched is that layer's FFN branch, embd is the
embedding and logits the final norm and head.

Two records match when both carry values (RUNNER_ACT_TRACE_VALUES=1, or a
reference trace written in the same format) and the relative L2 distance is
at most --rtol, or when neither does and the hashes are equal (bit-exact; the
only meaningful test between two traces of one build). Hashes from two
runtimes never match, so a cross-runtime comparison needs values on both
sides. A route matches when it selects the same experts in the same order.

Exit status: 0 every common point matches, 1 the traces part (the first point
is printed), 2 the traces cannot be compared (no common point, or a point one
side has hashed and the other only valued).
"""
import argparse
import json
import math
import sys

ORDER = {"embd": 0, "mixer": 1, "route": 2, "resid": 3, "logits": 4}


def load(path):
    recs, dup = {}, 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("v") != 1 or r.get("point") not in ORDER:
                raise SystemExit(f"{path}:{n}: not an activation trace record (v 1)")
            key = (r["pos"], r["layer"], r["point"])
            if key in recs:
                dup += 1      # a later pass over the same position: keep the first
                continue
            recs[key] = r
    return recs, dup


def forward_order(key):
    pos, layer, point = key
    return (pos, layer, ORDER[point])


def compare(a, b, rtol):
    """(matches, distance, how) for two records of one point."""
    if a["point"] == "route":
        same = a["experts"] == b["experts"]
        return same, 0.0 if same else 1.0, "experts"
    if a.get("n") != b.get("n"):
        return False, math.inf, "width"
    if "values" in a and "values" in b:
        num = sum((x - y) ** 2 for x, y in zip(a["values"], b["values"]))
        den = sum(y * y for y in b["values"])
        d = math.sqrt(num) / max(math.sqrt(den), 1e-30)
        return d <= rtol, d, "values"
    if "values" in a or "values" in b:
        return None, None, "mixed"
    same = a["hash"] == b["hash"]
    return same, 0.0 if same else 1.0, "hash"


def branch(key, prior_mixer_matched):
    pos, layer, point = key
    if point == "embd":
        return "the embedding"
    if point == "mixer":
        return f"layer {layer}'s mixer branch (attention or recurrent)"
    if point == "route":
        return f"layer {layer}'s router"
    if point == "resid":
        return (f"layer {layer}'s FFN branch" if prior_mixer_matched
                else f"layer {layer} (no mixer point to split it)")
    return "the final norm and head"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b", help="the reference side: distances are relative to its norm")
    ap.add_argument("--rtol", type=float, default=1e-4,
                    help="relative L2 distance two valued rows may have and still match")
    ap.add_argument("--json", help="also write the per-point comparison here")
    ap.add_argument("--table", action="store_true",
                    help="print every common point's distance, in forward order")
    args = ap.parse_args()

    a, da = load(args.a)
    b, db = load(args.b)
    common = sorted(set(a) & set(b), key=forward_order)
    if not common:
        print("no point is in both traces (different positions or layers?)")
        return 2
    rows, first, mixer_ok = [], None, {}
    for key in common:
        ok, d, how = compare(a[key], b[key], args.rtol)
        if ok is None:
            print(f"pos {key[0]} layer {key[1]} {key[2]}: one side has values and the "
                  f"other only a hash; trace both with values (RUNNER_ACT_TRACE_VALUES=1)")
            return 2
        if key[2] == "mixer":
            mixer_ok[(key[0], key[1])] = ok
        rows.append({"pos": key[0], "layer": key[1], "point": key[2],
                     "match": ok, "distance": d, "by": how})
        if not ok and first is None:
            first = (key, d, how, mixer_ok.get((key[0], key[1]), True))
    if args.table:
        for r in rows:
            d = r["distance"]
            print(f"  pos {r['pos']:>5} layer {r['layer']:>3} {r['point']:<6} "
                  f"{'match' if r['match'] else 'DIFF ':<5} "
                  f"{f'{d:.3g}' if r['by'] == 'values' else r['by']}")
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    print(f"{len(common)} common points ({only_a} only in A, {only_b} only in B"
          f"{f'; {da + db} later-pass duplicates ignored' if da + db else ''})")
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"rtol": args.rtol, "points": rows,
                       "first": None if first is None else
                       {"pos": first[0][0], "layer": first[0][1], "point": first[0][2],
                        "distance": first[1], "by": first[2]}}, f, indent=1)
            f.write("\n")
    if first is None:
        print(f"every common point matches (rtol {args.rtol:g} where valued, "
              f"exact where hashed)")
        return 0
    (pos, layer, point), d, how, mok = first
    print(f"first difference: pos {pos} layer {layer} {point} "
          f"({how}{f', relative distance {d:.3g}' if how == 'values' else ''}): "
          f"{branch((pos, layer, point), mok)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
