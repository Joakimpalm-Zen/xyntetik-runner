#!/usr/bin/env python3
"""R4.48.12 PREREG 2.3 / 2.4: Claim 2 ("the same answer alone or batched") analysis script.

usage: claim2_stats.py CLAIM2_DIR PROMPTS_JSONL COMPANIONS_JSONL GGUF LOAD_TABLE LOAD_TABLE_UNSLOTH OUT_PREFIX

Reads every <engine>-t<t>-... cell present in CLAIM2_DIR (Runner: runner-t{1,16}-{alone,batched}; llama.cpp and Unsloth's
build: <engine>-t{1,16}-{alone,batched}-fa{on,off}-ub{512,1}; the last COMPLETE line per target id) and writes OUT.json +
OUT.txt:
- per engine, across-all-settings byte identity (PREREG 2.3 1): a target is identical when its generated token ids AND the
  bytes they decode to are equal in every cell present; k / 20 with the 95% Wilson interval (claim1_stats.py's);
- factor attribution counts (2.3 2), each factor holding the others at the reference cell alone, t=16, fa on, ub 512;
- co-batching (2.3 3): Runner's /metrics batch counters per round (the route lines cannot appear: interpretation 6);
  llama-server's concurrently active slots from its launch / release lines (interpretation 5);
- aggregate throughput (2.3 4): per round sum(generated ids) / T_batch_wall_s, median and IQR over the 20 rounds, for the
  quoted cells (the build's resolved defaults, fa on / ub 512: the supervisor's ruling), each beside its load-table row;
- the Runner defect rule (2.3 5); the companions' own invariance (information);
- the PREREG 2.4 outcome branch and its wording, with Runner and mainline llama.cpp;
- Unsloth's build as the ADDITIONAL column (ADDENDUM-2026-11-24-unsloth.md): the same metrics; no combined label.
STATUS is INFORMATION until every cell of the pre-registered matrix is present (the t=1 cells run after the release).
Standard library + gguf-py (the vocabulary, via claim3_stats.load_vocab) only.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name, sha in (("claim1_stats.py", "6632e4f99ca3e15f"), ("claim3_stats.py", "31dfb185e3276e30")):
    if hashlib.sha256((HERE / name).read_bytes()).hexdigest()[:16] != sha:
        sys.exit("%s is not %s" % (name, sha))
sys.path.insert(0, str(HERE))
from claim1_stats import wilson_interval, quartiles_iqr  # noqa: E402
from claim3_stats import load_vocab  # noqa: E402

ENGINES = ("runner", "llama", "unsloth")
EXTRA = ("unsloth",)          # the additional column
T_STOP = 200006
INTERPRETATIONS = [
    "Cell set: the stats use every cell present. The pre-registered matrix is Runner 4 (t in {1, 16} x alone / batched) and"
    " llama.cpp 16 (x fa x ub); the t=1 cells run after the release (ruled), so until then K counts identity across the"
    " t=16 cells only (Runner 2, llama.cpp 8, Unsloth 8), factors that need t=1 are 'pending', and the 2.4 wording's"
    " 'all 4 settings' / '16 settings' text would overstate the measured cells. RULED (CLAIM2-INTERPRETATIONS-2026-11-26-"
    "stats.md 1): the 2.4 wordings are NOT used until the t=1 cells run; before then the figures are 'the sixteen-thread"
    " half, provisional, t=1 pending', and the branch the t=16 counts point to is shown as provisional only.",
    "Byte identity: 'the raw generated UTF-8 byte string (and generated token ID sequence up to EOS or 256 tokens)' is"
    " read as the target request's gen_ids AND the bytes decode(gen_ids) gives with the file's vocabulary (byte-level"
    " tokens through the inverse bytes_to_unicode, special tokens as their text). Equal ids give equal bytes, so the"
    " bytes add nothing but are checked. Runner's gen_ids carry the client-appended 200006 on stopped turns"
    " (stop_token_appended; its receipt lacks it); this is the same in every Runner cell, so identity is unaffected.",
    "Records: the last COMPLETE line per target id (the client's resume rule); incomplete or torn lines are counted"
    " and reported. Per cell: 20 targets; prompt_ids equal across all of an engine's cells (checked).",
    "Throughput: agg_tok_s per batched round = sum over its 9 requests of len(gen_ids) / T_batch_wall_s (barrier"
    " release to the 9th response, the client's T_batch_wall_s = the last end_offset_s); alone tok/s per round ="
    " len(gen_ids) / T_batch_wall_s of its one request (prefill included, as the wall is). len(gen_ids) is the"
    " generated id count (Claim 3's ruled interpretation 7); the client's agg_tok_s (usage counts) is shown beside"
    " where it differs (Runner's appended stop token). Median and IQR by claim1_stats.quartiles_iqr. The quoted"
    " R_* / L_* / U_* cells are the builds' resolved defaults (fa on, ub 512; the supervisor's ruling and the defaults"
    " probes); Runner has no fa / ub. Each figure carries its load-table row (median / peak load1, resident min,"
    " share of samples with resident < 0.99) and the ruled sentence on Runner's page loss.",
    "llama.cpp co-batching (CB_L): /metrics was off (PREREG 1.2's flag line has no --metrics: status 501) and the"
    " per-decode batch lines are below the server's default verbosity, so 'active sequences per decode batch' is not"
    " logged. Read instead: every generating slot's token enters ONE shared batch per update_slots iteration"
    " (tools/server/server-context.cpp:3726, batch.add at :4430 / :4562), and llama_decode splits it into micro-batches"
    " of n_ubatch (src/llama-context.cpp:2254). So CB = true for a cell when >= 2 slots generated at once AND"
    " n_ubatch > 1; with -ub 1 each micro-batch holds one token, i.e. the sequences decode one at a time inside one"
    " call (CB = false at the kernel level). Concurrency = slots between 'launch_slot_' and 'release' in the server's"
    " stderr: time-weighted mean while >= 1 is active, max, and the share of that time with >= 2. RULED (5): stated"
    " as 'inferred from the source and the slot log'.",
    "Runner co-batching: co_batched = false is PREREG 2's architectural statement (MANUAL.md:2685; sequential MoE"
    " decode). Its evidence: /metrics runner_batch_steps_total and runner_batch_sequences_total per round (after -"
    " before; mean sequences per step = the scheduler's slot count per step, not shared matmuls). The"
    " RUNNER_DEBUG_ROUTE=1 'path=' lines PREREG names CANNOT appear for this model at 7bf8b64d: model_forward_batch"
    " returns through model_inkling2_forward before route_ran() (src/model.c:17497); the env was set"
    " (run_claim2_t16.sh:75). Disclosed, not a run problem (risk-note 12). RULED (6), with a source check, CONFIRMED"
    " at 7bf8b64d: the server's one decode thread calls model_batch_decode once per step over the gathered slots"
    " (src/scheduler.c:197-214) and counts steps += 1, sequences += nb (:222-223), so the counters' mean is the slots"
    " GATHERED per step; with no device (--gpu off: model.c:6511, the load log's '0 layers'; no backend:"
    " gpu_none.c:448-451) model_batch_decode skips gpu_batch_decode (model.c:18197-18199) and runs the sequential"
    " fallback, model_forward per sequence in turn (:18271-18275), each a one-row forward (:18112-18114, n = 1) through"
    " model_inkling2_forward (:17497). One sequence at a time; no multi-sequence matrix product.",
    "Factor attribution (2.3 2): each count is the number of the 20 targets whose output differs between the"
    " reference cell (alone, t=16, fa on, ub 512; Runner: alone, t=16) and the cell that changes ONLY that factor:"
    " alone vs batched at t=16 (and at t=1, from the t=1 alone cell), t=16 vs t=1 alone and batched, fa on vs off,"
    " ub 512 vs 1. The 2.4 branch 3 '<failing factor>' lists Runner's factors with a non-zero count.",
    "Companions (information, not a PREREG metric): each of the 8 fixed companions runs once per batched round, 20"
    " times per batched cell; reported: distinct outputs per companion within each cell, and whether its output is the"
    " same across all of the engine's batched cells.",
    "Unsloth's column (ADDENDUM-2026-11-24-unsloth.md, 'an additional column in the same tables'): K_U / 20 with its"
    " Wilson interval, factor counts, CB_U, U_batched / U_alone, the companions. PREREG 0.2 defines paired comparisons"
    " for Claims 1 and 3 only and Claim 2 has no outcome rule between engines, so there is no Claim 2 'pair' to"
    " compute and no combined label (as ruled for Claim 3). The 2.4 outcome and wording use Runner and mainline only."
    " RULED (9): confirmed, no pairs.",
]


def decode_bytes(ids, toks, inv):
    out = bytearray()
    for i in ids:
        t = toks[i]
        if i >= 200000 and t.startswith("<|") and t.endswith("|>"):
            out += t.encode()
        else:
            out += bytes(inv[ch] for ch in t)
    return bytes(out)


def load_cell(path):
    by_id, n, inc, torn = {}, 0, 0, 0
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            torn += 1
            continue
        n += 1
        if r.get("complete") is True:
            by_id[r["id"]] = r
        else:
            inc += 1
    return {"by_id": by_id, "n_lines": n, "n_incomplete": inc, "n_torn": torn}


def target(r):
    return next(q for q in r["requests"] if q["role"] == "target")


def identity(cells, ids):
    names = list(cells)
    return {i: all(cells[n][i] == cells[names[0]][i] for n in names[1:]) for i in ids}


def differ(a, b, ids):
    return [i for i in ids if a[i] != b[i]]


def cell_name(e, t, mode, fa=None, ub=None):
    return "%s-t%d-%s" % (e, t, mode) if e == "runner" else "%s-t%d-%s-fa%s-ub%d" % (e, t, mode, fa, ub)


def matrix(e):
    if e == "runner":
        return [cell_name(e, t, m) for t in (16, 1) for m in ("alone", "batched")]
    return [cell_name(e, t, m, fa, ub) for t in (16, 1) for m in ("alone", "batched") for fa in ("on", "off") for ub in (512, 1)]


def factors(e, present):
    kw = {} if e == "runner" else {"fa": "on", "ub": 512}
    ref = cell_name(e, 16, "alone", **kw)
    g = lambda c: c if c in present else None
    out = [("alone vs batched (t=16)", ref, g(cell_name(e, 16, "batched", **kw))),
           ("alone vs batched (t=1)", g(cell_name(e, 1, "alone", **kw)), g(cell_name(e, 1, "batched", **kw))),
           ("t=16 vs t=1 (alone)", ref, g(cell_name(e, 1, "alone", **kw))),
           ("t=16 vs t=1 (batched)", g(cell_name(e, 16, "batched", **kw)), g(cell_name(e, 1, "batched", **kw)))]
    if e != "runner":
        out += [("fa on vs off", ref, g(cell_name(e, 16, "alone", fa="off", ub=512))),
                ("ub 512 vs 1", ref, g(cell_name(e, 16, "alone", fa="on", ub=1)))]
    return out


def round_tok_s(r):
    return sum(len(q["gen_ids"]) for q in r["requests"]) / r["T_batch_wall_s"]


def ts(s):
    m, sec, ms, us = (int(x) for x in s.split("."))
    return m * 60 + sec + ms / 1e3 + us / 1e6


def slot_concurrency(lines):
    ev = []
    for l in lines:
        m = re.match(r"(\d+\.\d+\.\d+\.\d+) I slot\s+(launch_slot_|release): id\s+(\d+)", l)
        if m:
            ev.append((ts(m.group(1)), 1 if m.group(2) == "launch_slot_" else -1))
    ev.sort(key=lambda x: (x[0], -x[1]))
    act, last, busy, w, w2, mx = 0, None, 0.0, 0.0, 0.0, 0
    for t, d in ev:
        if last is not None and act > 0:
            dt = t - last
            busy += dt
            w += act * dt
            w2 += dt if act >= 2 else 0.0
        act += d
        mx = max(mx, act)
        last = t
    return {"max_active": mx, "mean_active": w / busy if busy else 0.0, "share_ge2": w2 / busy if busy else 0.0,
            "busy_s": busy, "n_events": len(ev)}


def branch(k_r, k_l):
    if k_r == 20:
        return 2 if k_l == 20 else 1
    return 3


def parse_load_row(line):
    p = line.split()
    return p[0], {"start": float(p[1]), "share_above": float(p[2].rstrip("%")), "peak": float(p[3]), "median": float(p[4]),
                  "res_min": float(p[5]), "res_below_099": float(p[6].rstrip("%")), "secs": int(p[7])}


def load_rows(path):
    rows = {}
    for line in open(path):
        if re.match(r"(runner|llama|unsloth)-t\d+-\S+\s+\d", line):
            k, v = parse_load_row(line)
            rows[k] = v
    return rows


def share(k, n):
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "wilson_95": [lo, hi]}


def tput(recs):
    xs = [round_tok_s(r) for r in recs]
    q = quartiles_iqr(xs)
    client = [r.get("agg_tok_s") for r in recs]
    diff = sum(1 for a, b in zip(xs, client) if b is None or abs(a - b) > 1e-3)  # the client rounds to 4 decimals
    return {"median": q["median"], "p25": q["p25"], "p75": q["p75"], "iqr": q["iqr"], "n": len(xs),
            "client_agg_tok_s_median": quartiles_iqr([c for c in client if c is not None])["median"] if any(c is not None for c in client) else None,
            "rounds_differing_from_client": diff}


def main():
    d, prompts_path, comps_path, gguf_path, lt_path, ltu_path, outp = sys.argv[1:8]
    D = Path(d)
    ids = sorted(json.loads(l)["id"] for l in open(prompts_path) if json.loads(l)["split"] == "primary")
    comp_ids = sorted(json.loads(l)["id"] for l in open(comps_path))
    if len(ids) != 20 or len(comp_ids) != 8:
        sys.exit("expected 20 primaries and 8 companions")
    toks, inv = load_vocab(gguf_path)
    loads = {**load_rows(lt_path), **load_rows(ltu_path)}
    E = {}
    inputs, checks = {}, {}
    for e in ENGINES:
        present = [c for c in matrix(e) if (D / (c + ".jsonl")).exists()]
        cells, recs, comps = {}, {}, {}
        for c in present:
            p = D / (c + ".jsonl")
            inputs[c] = hashlib.sha256(p.read_bytes()).hexdigest()
            lc = load_cell(str(p))
            if sorted(lc["by_id"]) != ids:
                sys.exit("%s: the complete lines are not the 20 targets" % c)
            recs[c] = [lc["by_id"][i] for i in ids]
            cells[c] = {}
            pid = {}
            for r in recs[c]:
                q = target(r)
                cells[c][r["id"]] = (tuple(q["gen_ids"]), decode_bytes(q["gen_ids"], toks, inv))
                pid[r["id"]] = q["prompt_ids"]
                if e == "runner":
                    rc = q.get("gen_ids_receipt")
                    ok = rc == q["gen_ids"] or (q.get("stop_token_appended") and rc + [T_STOP] == q["gen_ids"])
                    if not ok:
                        sys.exit("%s %s: gen_ids vs receipt" % (c, r["id"]))
            checks[c] = {"lines": lc["n_lines"], "incomplete": lc["n_incomplete"], "torn": lc["n_torn"]}
            E.setdefault(e, {}).setdefault("prompt_ids", {})[c] = pid
            if "-batched" in c:
                comps[c] = {}
                for r in recs[c]:
                    for q in r["requests"]:
                        if q["role"] == "companion":
                            comps[c].setdefault(q["id"], set()).add(tuple(q["gen_ids"]))
        if not present:
            continue
        pids = E[e]["prompt_ids"]
        same_pid = all(pids[c] == pids[present[0]] for c in present)
        idt = identity(cells, ids)
        k = sum(idt.values())
        fac = []
        for lab, a, b in factors(e, set(present)):
            fac.append({"factor": lab, "cells": [a, b], "count": len(differ(cells[a], cells[b], ids)) if a and b else None,
                        "ids": differ(cells[a], cells[b], ids) if a and b else None})
        kw = {} if e == "runner" else {"fa": "on", "ub": 512}
        qa, qb = cell_name(e, 16, "alone", **kw), cell_name(e, 16, "batched", **kw)
        thr = {m: dict(tput(recs[c]), cell=c, load=loads.get(c)) for m, c in (("alone", qa), ("batched", qb)) if c in recs}
        cob = {}
        for c in present:
            if "-batched" not in c:
                continue
            if e == "runner":
                st, sq = [], []
                for r in recs[c]:
                    b4 = {t["when"]: t for t in r["telemetry"]}
                    def ctr(t, name):
                        m = re.search(r"^%s (\d+)" % name, (t.get("/metrics") or {}).get("body") or "", re.M)
                        return int(m.group(1)) if m else None
                    s0, s1 = ctr(b4["before"], "runner_batch_steps_total"), ctr(b4["after"], "runner_batch_steps_total")
                    q0, q1 = ctr(b4["before"], "runner_batch_sequences_total"), ctr(b4["after"], "runner_batch_sequences_total")
                    if None not in (s0, s1, q0, q1):
                        st.append(s1 - s0)
                        sq.append(q1 - q0)
                cob[c] = {"co_batched": False, "basis": "PREREG 2 (MANUAL.md:2685); interpretation 6",
                          "batch_steps_total": sum(st), "batch_sequences_total": sum(sq),
                          "mean_sequences_per_step": sum(sq) / sum(st) if sum(st) else None, "rounds_with_counters": len(st)}
            else:
                sc = slot_concurrency(open(D / (c + ".server.err"), errors="replace").read().splitlines())
                ub = int(c.rsplit("-ub", 1)[1])
                cob[c] = dict(sc, n_ubatch=ub, co_batched=bool(sc["max_active"] >= 2 and ub > 1),
                              basis="interpretation 5")
        compr = {}
        bc = [c for c in present if "-batched" in c]
        for cid in comp_ids:
            per = {c: len(comps[c].get(cid, set())) for c in bc}
            allsame = len(set().union(*[comps[c].get(cid, set()) for c in bc])) == 1 if bc else None
            compr[cid] = {"distinct_outputs_per_cell": per, "same_across_batched_cells": allsame}
        E[e] = {"cells_present": present, "cells_preregistered": len(matrix(e)), "prompt_ids_equal_across_cells": same_pid,
                "identical": idt, "K": share(k, 20), "not_identical": [i for i in ids if not idt[i]], "factors": fac,
                "throughput_quoted": thr, "co_batching": cob, "companions": compr}
    R, L = E["runner"], E["llama"]
    KR, KL = R["K"]["k"], L["K"]["k"]
    br = branch(KR, KL)
    full = all(len(E[e]["cells_present"]) == E[e]["cells_preregistered"] for e in ("runner", "llama"))
    f1 = lambda x: "%.2f" % x
    Rb, Ra = R["throughput_quoted"]["batched"]["median"], R["throughput_quoted"]["alone"]["median"]
    Lb, La = L["throughput_quoted"]["batched"]["median"], L["throughput_quoted"]["alone"]["median"]
    qcell = L["co_batching"][L["throughput_quoted"]["batched"]["cell"]]
    CB_L = "%s (mean %.2f / max %d active slots)" % (str(qcell["co_batched"]).lower(), qcell["mean_active"], qcell["max_active"])
    loL, hiL = L["K"]["wilson_95"]
    loR, hiR = R["K"]["wilson_95"]
    if br == 1:
        wording = ("Across 20 pre-registered prompts tested alone and inside a 9-request batch at 1 and 16 threads (greedy, UD-Q8_K_XL on CPU), Runner serves concurrent MoE requests sequentially one decode at a time (co-batched: false; batched aggregate %s tok/s vs alone %s tok/s at t=16) and produced byte-identical output across all 4 settings on 20/20 prompts (95%% Wilson [0.839, 1.000]), whereas llama.cpp (varying flash attention on/off and n_ubatch 1/512 across 16 settings; co-batched: %s, batched aggregate %s tok/s vs alone %s tok/s) stayed byte-identical on %d/20 prompts (95%% Wilson [%.3f, %.3f])."
                   % (f1(Rb), f1(Ra), CB_L, f1(Lb), f1(La), KL, loL, hiL))
    elif br == 2:
        wording = ("Across 20 pre-registered prompts on CPU (UD-Q8_K_XL, greedy), Runner serves concurrent MoE requests sequentially one decode at a time (co-batched: false; batched aggregate %s tok/s vs alone %s tok/s) and held byte identity across all 4 settings on 20/20 prompts (95%% Wilson [0.839, 1.000]); llama.cpp (16 settings adding flash-attention and n_ubatch toggles; co-batched: %s, batched aggregate %s tok/s vs alone %s tok/s) also held byte identity on 20/20 prompts (95%% Wilson [0.839, 1.000])."
                   % (f1(Rb), f1(Ra), CB_L, f1(Lb), f1(La)))
    else:
        ff = " and ".join(x["factor"] for x in R["factors"] if x["count"]) or "(no single factor: see the per-cell identity)"
        wording = ("On CPU (UD-Q8_K_XL, greedy), even though Runner serves concurrent MoE requests sequentially one decode at a time (co-batched: false; batched aggregate %s tok/s vs alone %s tok/s), Runner failed full batch/thread invariance, staying byte-identical across all 4 settings on %d/20 prompts (95%% Wilson [%.3f, %.3f]) vs %d/20 for llama.cpp (95%% Wilson [%.3f, %.3f]; co-batched: %s, %s tok/s); the remaining %d Runner prompts diverged between %s and are logged as an open Runner invariance defect."
                   % (f1(Rb), f1(Ra), KR, loR, hiR, KL, loL, hiL, CB_L, f1(Lb), 20 - KR, ff))
    rmin = min(v["res_min"] for k, v in loads.items() if k.startswith("runner-"))
    lmin = min(v["res_min"] for k, v in loads.items() if k.startswith("llama-"))
    page_sentence = ("Runner's two cells lost up to %.1f%% of the file's pages to host reclaim while llama.cpp's stayed resident,"
                     " so the times are information, not a speed comparison." % (100 * (1 - rmin)))
    defect = {"rule": "PREREG 2.3 5", "runner_not_identical": R["not_identical"],
              "logged_as_runner_defect": bool(R["not_identical"])}
    status = "FINAL" if full else "sixteen-thread half, provisional; t=1 pending (cells present: Runner %d/4, llama.cpp %d/16)" % (
        len(R["cells_present"]), len(L["cells_present"]))
    if not full:
        wording = None   # ruled (interpretation 1): the 2.4 wordings are not used until the t=1 cells run
    out = {"script": "claim2_stats.py", "status": status, "inputs_sha256": inputs, "checks": checks,
           "interpretations": INTERPRETATIONS, "engines": {e: {k: v for k, v in E[e].items() if k != "prompt_ids"} for e in E},
           "outcome_branch": br, "outcome_branch_status": "final" if full else "provisional (t=16 counts only; t=1 pending)",
           "wording": wording, "wording_cells_measured": {"runner": len(R["cells_present"]), "llama": len(L["cells_present"])},
           "page_sentence": page_sentence, "resident_min": {"runner": rmin, "llama": lmin}, "runner_defect": defect,
           "unsloth_column": {"basis": INTERPRETATIONS[8]}}
    for e in E:
        out["engines"][e]["identical"] = {i: v for i, v in E[e]["identical"].items()}
    Path(outp + ".json").write_text(json.dumps(out, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)) + "\n")
    W = []
    w = W.append
    w("CLAIM 2 (PREREG 2.3 / 2.4), %s" % status)
    w("inputs: " + ", ".join("%s %s" % (c, inputs[c][:16]) for c in inputs))
    w("checks: every cell 20 complete targets; " + "; ".join("%s lines %d incomplete %d torn %d" % (c, v["lines"], v["incomplete"], v["torn"]) for c, v in checks.items() if v["incomplete"] or v["torn"] or v["lines"] != 20) + ("" if any(v["incomplete"] or v["torn"] or v["lines"] != 20 for v in checks.values()) else "no incomplete or torn lines"))
    w("")
    for e in E:
        x = E[e]
        tag = " (ADDITIONAL column)" if e in EXTRA else ""
        w("== %s%s: cells %d of %d pre-registered; prompt_ids equal across cells: %s" % (e, tag, len(x["cells_present"]), x["cells_preregistered"], x["prompt_ids_equal_across_cells"]))
        w("  byte identity across every cell present: K %d/20, 95%% Wilson [%.3f, %.3f]; not identical: %s" % (
            x["K"]["k"], x["K"]["wilson_95"][0], x["K"]["wilson_95"][1], ", ".join(x["not_identical"]) or "none"))
        for f in x["factors"]:
            w("  factor %-24s %s" % (f["factor"], "pending (%s)" % ("t=1 cells not run" if "t=1" in f["factor"] else "cell missing") if f["count"] is None else "%d/20 differ%s" % (f["count"], (": " + ", ".join(f["ids"])) if f["ids"] else "")))
        for m in ("alone", "batched"):
            t = x["throughput_quoted"].get(m)
            if t:
                ld = t["load"] or {}
                w("  %s tok/s (%s): median %.2f, IQR %.2f..%.2f (n %d)%s | load median %s peak %s, resident min %s, res<0.99 %s%%" % (
                    m, t["cell"], t["median"], t["p25"], t["p75"], t["n"],
                    "" if not t["rounds_differing_from_client"] else "; client agg_tok_s (usage counts) median %.2f, differs on %d rounds" % (t["client_agg_tok_s_median"], t["rounds_differing_from_client"]),
                    ld.get("median"), ld.get("peak"), ld.get("res_min"), ld.get("res_below_099")))
        for c, v in x["co_batching"].items():
            if e == "runner":
                w("  co-batching %s: co_batched false (PREREG 2; the CPU batch step decodes one sequence at a time, interpretation 6); /metrics: %d batch steps, %d sequences, mean %.2f sequences GATHERED per step over %d rounds" % (
                    c, v["batch_steps_total"], v["batch_sequences_total"], v["mean_sequences_per_step"] or 0, v["rounds_with_counters"]))
            else:
                w("  co-batching %s: co_batched %s, inferred from the source and the slot log (n_ubatch %d; active slots mean %.2f, max %d, >= 2 for %.1f%% of busy time)" % (
                    c, str(v["co_batched"]).lower(), v["n_ubatch"], v["mean_active"], v["max_active"], 100 * v["share_ge2"]))
        nc = sum(1 for v in x["companions"].values() if v["same_across_batched_cells"])
        w("  companions (information): same output across all batched cells for %d/8; distinct outputs within a cell: %s" % (
            nc, ", ".join("%s %s" % (cid, max(v["distinct_outputs_per_cell"].values()) if v["distinct_outputs_per_cell"] else "-") for cid, v in sorted(x["companions"].items()))))
        w("")
    w("Runner defect rule (2.3 5): Runner not identical on %s" % (", ".join(defect["runner_not_identical"]) or "no prompt (no defect)"))
    w("PREREG 2.4 outcome branch: %d (%s)%s" % (br, {1: "Runner 20/20, llama.cpp < 20/20", 2: "both 20/20", 3: "Runner < 20/20"}[br],
                                               "" if full else ", PROVISIONAL: t=16 counts only, t=1 pending"))
    if full:
        w("wording: " + wording)
    else:
        w("wording: not used until the t=1 cells run (ruled, interpretation 1); measured settings so far: Runner %d of 4, llama.cpp %d of 16" % (
            len(R["cells_present"]), len(L["cells_present"])))
    w("beside the tok/s (ruled): " + page_sentence)
    w("Unsloth's column: additional, no combined label; PREREG 0.2 has no Claim 2 pair (interpretation 9).")
    w("")
    w("interpretations (for the supervisor's ruling before the final run):")
    for i, s in enumerate(INTERPRETATIONS, 1):
        w("  %d. %s" % (i, s))
    Path(outp + ".txt").write_text("\n".join(W) + "\n")
    print("\n".join(W))


if __name__ == "__main__":
    main()
