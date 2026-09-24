#!/usr/bin/env python3
"""Calibration report for choice_logprobs decision records (JC-R1).

The runner's `choice_logprobs` response field reports, at every constrained
decision point, a posterior over the legal alternatives. This script answers
the question that decides whether that posterior is USABLE as a confidence
signal: when the model says 0.8, is it right 80% of the time?

Input: JSONL, one labeled decision per line:

    {"alternatives": [{"id": 314, "prob": 0.61, ...}, ...],
     "correct_id": 314,
     "acceptable_ids": [314, 271],   # optional: any of these counts as right
     "coverage": 0.97}               # other passthroughs are ignored

`alternatives` is a decision record's array exactly as the server emitted it
(descending prob, renormalized over the legal probed set); `correct_id` is
the ground-truth token id for that decision. A record whose correct id is
not among the stored alternatives counts as a wrong prediction at the
model's stated confidence (that is the honest reading: the model put the
truth outside its top-8 legal set). `acceptable_ids`, when present, widens
the truth to a set (synonym siblings a catalog cannot separate); the Brier
term then treats the set's mass as one outcome.

Output: top-1 accuracy, Brier score, ECE (10 equal-width bins), and a
reliability table. With --threshold-target T the report adds the confidence
certificate: the lowest top-1 confidence above which the decisions are
right at least T of the time, and the share of decisions that clears it.
Exit code 1 if ECE exceeds --max-ece (default: off), so the script can gate.

No third-party imports, usable anywhere the runner builds.
"""

import argparse
import json
import sys


def read_decisions(fh):
    """(confidence, hit, brier) per usable line; returns (decisions, skipped)."""
    decisions = []
    skipped = 0
    for ln, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            alts = rec["alternatives"]
            correct = rec["correct_id"]
        except (ValueError, KeyError) as e:
            print(f"line {ln}: skipped ({e})", file=sys.stderr)
            skipped += 1
            continue
        if not alts:
            skipped += 1
            continue
        acceptable = set(rec.get("acceptable_ids") or [correct])
        acceptable.add(correct)
        conf = alts[0]["prob"]
        hit = alts[0]["id"] in acceptable
        # Brier over the full stored distribution, the acceptable set pooled
        # into one outcome; any truth outside the stored alternatives
        # contributes y=1, p=0.
        p_acc = sum(a["prob"] for a in alts if a["id"] in acceptable)
        brier = (p_acc - 1.0) ** 2 + sum(a["prob"] ** 2 for a in alts if a["id"] not in acceptable)
        decisions.append((conf, hit, brier))
    return decisions, skipped


def summarize(decisions, bins=10):
    """Accuracy, Brier, ECE and the reliability bins of (conf, hit, brier)
    triples. bins entries: lo, hi, n, confidence, accuracy, gap."""
    n = len(decisions)
    if n == 0:
        return {"n": 0, "accuracy": None, "brier": None, "ece": None, "bins": []}
    acc = sum(h for _, h, _ in decisions) / n
    brier = sum(b for _, _, b in decisions) / n
    table = [[0, 0, 0.0] for _ in range(bins)]   # count, hits, conf mass
    for conf, hit, _ in decisions:
        b = min(int(conf * bins), bins - 1)
        table[b][0] += 1
        table[b][1] += hit
        table[b][2] += conf
    ece = sum(c * abs(h / c - m / c) for c, h, m in table if c) / n
    rows = []
    for i, (c, h, m) in enumerate(table):
        rows.append({"lo": i / bins, "hi": (i + 1) / bins, "n": c,
                     "confidence": (m / c) if c else None,
                     "accuracy": (h / c) if c else None,
                     "gap": (h / c - m / c) if c else None})
    return {"n": n, "accuracy": acc, "brier": brier, "ece": ece, "bins": rows}


def certificate(decisions, target):
    """The confidence certificate: the lowest top-1 confidence t such that
    the decisions at confidence >= t are right at least `target` of the
    time, with the coverage (share of decisions at or above t) and the
    accuracy there. Candidates are the observed confidences, scanned from
    the most confident down; the scan stops at the first candidate the
    prefix fails, so the certificate is a contiguous top slice. threshold
    None means even the most confident decisions miss the target."""
    if not decisions:
        return {"target": target, "threshold": None, "coverage": 0.0, "accuracy": None, "n": 0}
    order = sorted(decisions, key=lambda d: -d[0])
    n = len(order)
    hits = 0
    best = None
    i = 0
    while i < n:
        j = i
        while j < n and order[j][0] == order[i][0]:
            hits += order[j][1]
            j += 1
        acc = hits / j
        if acc >= target:
            best = (order[i][0], j, acc)
        else:
            break
        i = j
    if best is None:
        return {"target": target, "threshold": None, "coverage": 0.0, "accuracy": None, "n": n}
    t, k, acc = best
    return {"target": target, "threshold": t, "coverage": k / n, "accuracy": acc,
            "n": n, "n_covered": k}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", help="labeled decisions, one JSON object per line"
                                  " ('-' for stdin)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--max-ece", type=float, default=None,
                    help="exit 1 if ECE exceeds this")
    ap.add_argument("--threshold-target", type=float, default=None,
                    help="also report the confidence certificate for this accuracy")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args()

    fh = sys.stdin if args.jsonl == "-" else open(args.jsonl)
    decisions, skipped = read_decisions(fh)
    if fh is not sys.stdin:
        fh.close()

    if not decisions:
        print("no usable decisions", file=sys.stderr)
        return 2

    s = summarize(decisions, args.bins)
    cert = certificate(decisions, args.threshold_target) if args.threshold_target is not None else None
    if args.json:
        print(json.dumps({"skipped": skipped, **s, "certificate": cert}, indent=2))
    else:
        print(f"decisions {s['n']}  skipped {skipped}")
        print(f"top-1 accuracy {s['accuracy']:.4f}   Brier {s['brier']:.4f}   "
              f"ECE({args.bins} bins) {s['ece']:.4f}")
        print(f"{'bin':>11}  {'n':>6}  {'confidence':>10}  {'accuracy':>8}  gap")
        for b in s["bins"]:
            if not b["n"]:
                print(f"{b['lo']:.2f}-{b['hi']:.2f}  {0:>6}  {'-':>10}  {'-':>8}  -")
                continue
            print(f"{b['lo']:.2f}-{b['hi']:.2f}  {b['n']:>6}  {b['confidence']:>10.4f}  "
                  f"{b['accuracy']:>8.4f}  {b['gap']:+.4f}")
        if cert is not None:
            if cert["threshold"] is None:
                print(f"certificate: no confidence level reaches accuracy {cert['target']}")
            else:
                print(f"certificate: confidence >= {cert['threshold']:.4f} covers "
                      f"{cert['coverage']:.1%} of decisions ({cert['n_covered']} of {cert['n']}) "
                      f"at accuracy {cert['accuracy']:.4f} (target {cert['target']})")

    if args.max_ece is not None and s["ece"] > args.max_ece:
        print(f"FAIL: ECE {s['ece']:.4f} > {args.max_ece}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
