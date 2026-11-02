#!/usr/bin/env python3
"""Gate x mutant table for the R3.25 CUDA half, from the audit logs.
make-table.py PASS1.log PASS1B.log PASS2.log > table.md"""
import re, sys


def parse(path):
    blocks, cur = {}, None
    for line in open(path):
        if not line.startswith("  "):
            m = re.match(r"(.+?)(?: \(\d+ replacements\): caught by.*)?$", line.rstrip())
            name = m.group(1)
            cur = None if name.startswith(("R3.25", "restored")) else name
            if cur:
                cur = "baseline" if cur.startswith("baseline") else cur
                blocks[cur] = {}
            continue
        if cur:
            m = re.match(r"  (.{26}) (PASS|FAIL|SKIP) (\S*)\s+\d+s", line)
            blocks[cur][m.group(1).strip()] = (m.group(2), m.group(3))
    return blocks


p1 = parse(sys.argv[1])
p1.update({k: v for k, v in parse(sys.argv[2]).items() if k != "baseline"})
p2 = parse(sys.argv[3])
SHORT = {"attn-residual-dropped": "attn residual", "ssm-D-skip-dropped": "SSM D skip",
         "silu-to-gelu (dense FFN)": "silu to gelu", "rmsnorm-skipped (scale 1)": "norm skipped",
         "rmsnorm-eps-dropped": "norm eps", "rope-position-plus-one (k_rope)": "rope +1 (Q and K)",
         "attn-scale-x1.05": "attn scale x1.05", "matvec-bias-dropped (all)": "bias dropped",
         "rope-K-skipped": "K rope skipped", "rope-skipped (Q and K)": "rope skipped"}
SELF = {"prefix chunk 4", "prefix chunk 64", "cuda-tc-kq (TC prefill)"}
muts = [m for m in p1 if m != "baseline"]
gates = list(p1["baseline"])
print("| gate | healthy | " + " | ".join(SHORT[m] for m in muts) + " |")
print("|---|---|" + "---|" * len(muts))
for g in gates:
    row = ["`%s`" % g, p1["baseline"][g][1] or "ok"]
    for m in muts:
        s1, d = p1[m][g]
        if s1 == "FAIL":
            mark = "**C**"          # caught at the default bound
        elif g in p2.get(m, {}) and p2[m][g][0] == "FAIL":
            mark = "**T**"          # caught only at the tight bound
        elif g in SELF:
            mark = "s"              # CUDA against CUDA: a mutant on the shared path cannot show
        elif d and d == p1["baseline"][g][1]:
            mark = "n"              # the deviation is the healthy residue: the path is not reached
        else:
            mark = "-"              # moved, or not measured, and still passed
        row.append(("%s %s" % (d, mark)).strip())
    print("| " + " | ".join(row) + " |")
print()
print("| mutant | caught at 2e-3 (19 gates) | gpu-id caught at the tight bound (14) | reachable and still missed at the tight bound |")
print("|---|---|---|---|")
for m in muts:
    miss = [g for g in gates if g.startswith("gpu-id") and p2[m][g][0] == "PASS" and p2[m][g][1] != p2["baseline"][g][1]]
    print("| %s | %d | %d | %s |" % (SHORT[m], sum(p1[m][g][0] == "FAIL" for g in gates),
          sum(st == "FAIL" for st, _ in p2[m].values()),
          ", ".join("%s %s" % (g[7:], p2[m][g][1]) for g in miss) or "none"))
