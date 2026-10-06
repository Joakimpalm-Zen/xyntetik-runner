#!/usr/bin/env python3
"""R7.14.5 outcome rows from shadow-replay evidence files, one engine per column.

    python3 scripts/outcome-table.py llama=bench/outcome/llama/evidence.jsonl \
        strata=bench/outcome/strata/evidence.jsonl runner=bench/outcome/runner/evidence.jsonl

Counts before rates. Every row is read straight from the records the replay
wrote: attempts, verified (tests pass), the attempt's end reason (turn budget,
no tool call, model/protocol error, finished), attempts that never changed the
workspace, and the per-attempt means (turns, tool calls, prompt and completion
tokens, wall seconds). Prints Markdown.
"""
import json, sys
from collections import Counter

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def reason_class(r):
    first = (r.get("reasons") or [""])[0]
    if "turn budget" in first: return "turn budget"
    if "no tool call" in first: return "no tool call"
    if "model error" in first:
        return "protocol error (server)" if "HTTP 5" in first or "parse" in first else "model error"
    if "wall" in first: return "wall clock"
    if "finish" in first or "verified" in first: return "finished"
    return first[:40] or "(none)"

def col(rows):
    n = len(rows)
    ver = sum(1 for r in rows if r.get("disposition") == "verified_local_attempt")
    noop = sum(1 for r in rows if any("no_op" in s for s in (r.get("verifier") or {}).get("reasons", [])))
    cls = Counter(reason_class(r) for r in rows)
    def mean(k):
        v = [float((r.get("resources") or {}).get(k, 0)) for r in rows]
        return sum(v) / len(v) if v else 0.0
    wall = [r.get("wall_s", 0.0) for r in rows]
    return {"attempts": n, "verified": ver, "no_edit": noop, "classes": cls,
            "turns": mean("turns"), "calls": mean("tool_calls"),
            "ptok": mean("prompt_tokens"), "ctok": mean("completion_tokens"),
            "wall": sum(wall) / len(wall) if wall else 0.0, "tps": mean("probe_tps")}

def main(argv):
    cols = {}
    for a in argv:
        name, path = a.split("=", 1)
        try: cols[name] = col(load(path))
        except FileNotFoundError: cols[name] = None
    names = list(cols)
    print("| Metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    def row(label, f):
        print("| " + label + " | " + " | ".join(f(cols[n]) if cols[n] else "not run" for n in names) + " |")
    row("Tasks attempted", lambda c: str(c["attempts"]))
    row("Verified (tests pass)", lambda c: "%d / %d" % (c["verified"], c["attempts"]))
    row("Attempts that never edited a file", lambda c: "%d / %d" % (c["no_edit"], c["attempts"]))
    allcls = sorted({k for c in cols.values() if c for k in c["classes"]})
    for k in allcls:
        row("Ended by: " + k, lambda c, k=k: str(c["classes"].get(k, 0)))
    row("Mean turns / tool calls", lambda c: "%.1f / %.1f" % (c["turns"], c["calls"]))
    row("Mean prompt / completion tokens", lambda c: "%.0f / %.0f" % (c["ptok"], c["ctok"]))
    row("Mean wall s per task", lambda c: "%.0f" % c["wall"])
    row("Probe decode tok/s", lambda c: "%.1f" % c["tps"])

if __name__ == "__main__":
    main(sys.argv[1:])
