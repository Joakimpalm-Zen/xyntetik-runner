#!/usr/bin/env python3
"""Paired comparison of tooluse-shifted records: can the instrument fail?

R8.4.3 asks two things of the set: the base must land clearly apart from the
adapters, and the adapters must differ somewhere. Both are paired questions
(same prompts, different arms), so they are answered per row, never by
comparing totals:

  raw / native / next legs   exact verdicts paired per row; the discordant
                             pairs (A right, B wrong / A wrong, B right) and
                             the McNemar-style z = (b - c) / sqrt(b + c).
  decide leg                 per-row log loss of the acceptable tool, paired
                             mean difference with a seeded bootstrap 95% CI;
                             sign counts (rows where A is more confident in
                             the truth than B).

Two arms are "told apart" when the decide-leg CI excludes zero or a
generation leg's discordant pairs give |z| >= 2; the direction (which arm
the evidence favours) is reported beside it, because an instrument that
shows an adapter BELOW its base has still discriminated. R8.4.3 is read
both ways: as written (base clearly below every adapter, adapters differ)
and as the instrument question (base and adapters told apart, adapters
told apart). Counts first, rates never; the records' hashes are echoed so
the table can be traced.

Usage:
    tooluse-shifted-compare.py BASE.json ADAPTER.json [ADAPTER2.json ...]
        [--seed N] [--resamples N] [--markdown]
"""
import argparse
import json
import math
import random
import sys


def load(path):
    with open(path) as f:
        r = json.load(f)
    r["_path"] = path
    r["_name"] = r.get("adapter_file") or "base"
    return r


def rows_by_id(record, leg, key):
    lg = record.get(leg)
    if not lg:
        return {}
    return {row["id"]: row.get(key) for row in lg["rows"]}


def paired_counts(a, b, leg, key="exact"):
    """(n, both, b_only, a_only, neither, z) over rows both records score."""
    ra, rb = rows_by_id(a, leg, key), rows_by_id(b, leg, key)
    ids = sorted(set(ra) & set(rb))
    both = a_only = b_only = neither = 0
    for i in ids:
        x, y = bool(ra[i]), bool(rb[i])
        if x and y:
            both += 1
        elif x:
            a_only += 1
        elif y:
            b_only += 1
        else:
            neither += 1
    disc = a_only + b_only
    z = (a_only - b_only) / math.sqrt(disc) if disc else 0.0
    return {"n": len(ids), "both": both, "a_only": a_only, "b_only": b_only,
            "neither": neither, "z": z}


def paired_nll(a, b, seed=20260924, resamples=4000):
    """Paired decide-leg log loss: mean(A - B) with a bootstrap 95% CI over
    rows. Negative means A puts more mass on the truth than B."""
    ra, rb = rows_by_id(a, "decide_leg", "nll"), rows_by_id(b, "decide_leg", "nll")
    ids = sorted(i for i in set(ra) & set(rb) if ra[i] is not None and rb[i] is not None)
    if not ids:
        return None
    d = [ra[i] - rb[i] for i in ids]
    n = len(d)
    mean = sum(d) / n
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    a_better = sum(1 for x in d if x < -1e-9)
    b_better = sum(1 for x in d if x > 1e-9)
    return {"n": n, "mean_diff": mean, "ci95": [lo, hi], "a_better": a_better,
            "b_better": b_better, "ties": n - a_better - b_better,
            "excludes_zero": lo > 0 or hi < 0}


def per_category_nll(a, b):
    la, lb = a.get("decide_leg"), b.get("decide_leg")
    if not la or not lb:
        return {}
    cat = {r["id"]: r["category"] for r in la["rows"]}
    ra, rb = rows_by_id(a, "decide_leg", "nll"), rows_by_id(b, "decide_leg", "nll")
    out = {}
    for i in set(ra) & set(rb):
        if ra[i] is None or rb[i] is None:
            continue
        c = out.setdefault(cat[i], [0, 0.0])
        c[0] += 1
        c[1] += ra[i] - rb[i]
    return {k: {"n": v[0], "mean_diff": v[1] / v[0]} for k, v in out.items() if v[0]}


def compare(a, b, seed, resamples):
    out = {"a": a["_name"], "b": b["_name"], "legs": {}}
    for leg, key in (("raw_leg", "exact"), ("native_leg", "exact"), ("next_leg", "exact")):
        if a.get(leg) and b.get(leg):
            out["legs"][leg] = paired_counts(a, b, leg, key)
    if a.get("decide_leg") and b.get("decide_leg"):
        out["legs"]["decide_leg"] = paired_nll(a, b, seed, resamples)
        out["legs"]["decide_by_category"] = per_category_nll(a, b)
    return out


def apart(cmp):
    """The two arms are told apart on the decide leg: the paired log-loss CI
    excludes zero. Direction is reported separately."""
    d = cmp["legs"].get("decide_leg")
    return bool(d and d["excludes_zero"])


def apart_any(cmp):
    """Told apart on any leg: the decide CI excludes zero, or a generation
    leg's discordant pairs give |z| >= 2."""
    if apart(cmp):
        return True
    return any(abs(p.get("z", 0)) >= 2.0 for k, p in cmp["legs"].items()
               if k in ("raw_leg", "native_leg", "next_leg") and p)


def direction(cmp):
    """Which arm the evidence favours: 'B' when the adapter (B) is closer to
    the truth on the decide leg or wins more discordant raw pairs, 'A' for
    the base, 'mixed' when the legs disagree."""
    votes = []
    d = cmp["legs"].get("decide_leg")
    if d and d["excludes_zero"]:
        votes.append("A" if d["mean_diff"] < 0 else "B")
    for k in ("raw_leg", "native_leg", "next_leg"):
        pc = cmp["legs"].get(k)
        if pc and abs(pc["z"]) >= 2.0:
            votes.append("A" if pc["z"] > 0 else "B")
    if not votes:
        return "none"
    return votes[0] if all(v == votes[0] for v in votes) else "mixed"


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def render(base, adapters, cmps, pair_cmps):
    lines = []
    lines.append("## Records")
    for r in [base] + adapters:
        lines.append(f"- {r['_name']}: set {r.get('set_version', 'v1')}, model {r.get('model_sha256', '')[:8]}, "
                     f"adapter {(r.get('adapter_sha256') or '')[:8] or '-'}, host {r.get('host', '')}, "
                     f"threads {r.get('threads', '?')}, gpu {r.get('gpu', 'off')}")
    lines.append("")
    lines.append("## Base against each adapter (A = base, B = adapter)")
    lines.append("| adapter | leg | n | both | A only | B only | z | decide mean(A-B) nll | 95% CI | A better / B better | apart |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for c in cmps:
        for leg in ("raw_leg", "native_leg", "next_leg"):
            p = c["legs"].get(leg)
            if not p:
                continue
            lines.append(f"| {c['b']} | {leg} | {p['n']} | {p['both']} | {p['a_only']} | {p['b_only']} | {p['z']:+.2f} | | | | |")
        d = c["legs"].get("decide_leg")
        if d:
            lines.append(f"| {c['b']} | decide_leg | {d['n']} | | | | | {d['mean_diff']:+.4f} | "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {d['a_better']} / {d['b_better']} | {'yes' if apart(c) else 'no'} |")
    lines.append("")
    if pair_cmps:
        lines.append("## Adapter pairs (A, B as named)")
        lines.append("| A | B | raw A only | raw B only | z | decide mean(A-B) nll | 95% CI | A better / B better | differ |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|---|")
        for c in pair_cmps:
            r = c["legs"].get("raw_leg") or {}
            d = c["legs"].get("decide_leg")
            lines.append(f"| {c['a']} | {c['b']} | {r.get('a_only', '-')} | {r.get('b_only', '-')} | {r.get('z', 0):+.2f} | "
                         + (f"{d['mean_diff']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {d['a_better']} / {d['b_better']} | {'yes' if d['excludes_zero'] else 'no'} |"
                            if d else "- | - | - | - |"))
        lines.append("")
    lines.append("## Decide leg by category, mean(base - adapter) log loss (negative: base closer to the truth)")
    cats = sorted({k for c in cmps for k in c["legs"].get("decide_by_category", {})})
    if cats:
        lines.append("| category | " + " | ".join(c["b"] for c in cmps) + " |")
        lines.append("|---|" + "---:|" * len(cmps))
        for k in cats:
            lines.append(f"| {k} | " + " | ".join(
                f"{c['legs']['decide_by_category'][k]['mean_diff']:+.3f} (n={c['legs']['decide_by_category'][k]['n']})"
                if k in c["legs"].get("decide_by_category", {}) else "-" for c in cmps) + " |")
    lines.append("")
    sep = all(apart_any(c) for c in cmps) if cmps else False
    above = [c for c in cmps if c["legs"].get("raw_leg") and c["legs"]["raw_leg"]["z"] <= -2.0
             and c["legs"].get("decide_leg") and c["legs"]["decide_leg"]["excludes_zero"]
             and c["legs"]["decide_leg"]["mean_diff"] > 0]
    differ = [f"{c['a']} vs {c['b']}" for c in pair_cmps if apart(c)]
    lines.append("## Verdict")
    for c in cmps:
        lines.append(f"- base against {c['b']}: {'told apart' if apart_any(c) else 'not told apart'}, "
                     f"evidence favours {'the base' if direction(c) == 'A' else 'the adapter' if direction(c) == 'B' else direction(c)}")
    lines.append(f"- every adapter told apart from the base on some leg: {'YES' if sep else 'NO'}")
    lines.append(f"- adapters clearly ABOVE the base (raw z <= -2 and decide CI below zero): "
                 f"{', '.join(c['b'] for c in above) if above else 'none'}")
    lines.append(f"- adapter pairs told apart on the decide leg: {', '.join(differ) if differ else 'none'}")
    lines.append(f"- R8.4.3 as written (base clearly below the adapters, adapters differ): "
                 f"{'PASS' if above and len(above) == len(cmps) and differ else 'NOT PASSED'}")
    lines.append(f"- R8.4.3 as an instrument question (base and adapters told apart, adapters told apart): "
                 f"{'PASS' if sep and differ else 'NOT PASSED'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", nargs="+", help="base record first, then adapter records")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    recs = [load(p) for p in args.records]
    base, adapters = recs[0], recs[1:]
    if not adapters:
        sys.exit("need a base record and at least one adapter record")
    cmps = [compare(base, a, args.seed, args.resamples) for a in adapters]
    pair_cmps = [compare(adapters[i], adapters[j], args.seed, args.resamples)
                 for i in range(len(adapters)) for j in range(i + 1, len(adapters))]
    if args.json:
        print(json.dumps({"base_vs_adapters": cmps, "adapter_pairs": pair_cmps,
                          "verdict": {"base_apart": all(apart_any(c) for c in cmps),
                                      "direction": {c["b"]: direction(c) for c in cmps},
                                      "pairs_differ": [f"{c['a']} vs {c['b']}" for c in pair_cmps if apart(c)]}},
                         indent=2))
    else:
        print(render(base, adapters, cmps, pair_cmps))


if __name__ == "__main__":
    main()
