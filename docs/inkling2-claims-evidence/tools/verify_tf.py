#!/usr/bin/env python3
"""R4.48.12 PREREG 0.1 / 1.3: teacher-forced verification of the engines' Claim 1 outputs on the pinned float32 reference.

It imports ref_firstlight.py (the same Store + Patch, the same setup_model; no copy of any model code), as
ref_generate.py does. For each engine output s = prompt_ids + gen_ids (gen_ids cut to --truncate when given) it runs
ONE plain forward over s with use_cache=False (PREREG 0.1: zero KV caching, no generation) and reads the reference's
logits row at every generated position pos = P - 1 + j (0 <= j < L), the row conditioned on s[:P + j].

Per position: the reference's argmax, its top-1 minus top-2 logit margin, whether the top two logits are bit-equal
(an exact tie), and whether g_j AGREES: g_j's logit is bit-equal to the row's maximum (so a tie counts g_j as agreeing
when g_j shares the maximum, PREREG 0.1's exact-tie rule).
Per (engine, prompt), PREREG 1.3's verdict:
  identical = encoding_equal AND every position agrees AND complete, where
    encoding_equal: the engine's prompt_ids equal the reference's apply_chat_template ids (REF_IDS); a prompt/token
                    encoding discrepancy fails and is logged explicitly (PREREG 1.3);
    complete:       (L < cap and g_{L-1} == 200006) or L == cap; cap = 256, or --truncate N.
  first_diff = the smallest j that does not agree, with the reference's argmax and margin there.
Two engines that produced the identical s share one forward (PREREG 0.1's reuse of the deterministic reference pass).
Version 2 (ERRATUM 2026-11-24, correction 2): every generated position whose reference margin is below
HEAD_SHAPE_BOUND (1e-4) is flagged as head-shape sensitive (this forward's multi-row head and generate()'s 1-row head
round differently, at most 1.717e-05 measured), and each verdict records whether it depends on a flagged position;
a flag changes no verdict.

Resumable: each finished sequence is appended to OUT_PREFIX.seqs.jsonl at once (keyed by the sha256 of s); a rerun
with the same arguments skips the keys already there, so an interrupted run loses at most the forward in progress.

--control CONTROL.json (ref_generate.py's output): PREREG 0.1's pre-run check: on the control's prompts, the teacher-
forced verdict of each engine's output (cut to the control's max_new_tokens) must equal the verdict read directly from
the reference's TRUE greedy generation (identical, first_diff, and the reference's token there); any discrepancy is
reported and the run exits 4 ('halts before Claim 1 proceeds').

Environment (ERRATUM 1): <work>/venv (torch 2.14.0+cpu), PYTHONPATH=<transformers PR 51050 at 6173a2f6>/src,
OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4, run from the directory holding ref_firstlight.py.

usage: verify_tf.py CKPT_DIR MAP REF_IDS.jsonl OUT_PREFIX --engine NAME=answers.jsonl [--engine ...]
                    [--ids c1_000,...] [--truncate N] [--control CONTROL.json] [--threads 4]
"""
import argparse, hashlib, json, os, statistics, sys, time
import torch

sys.path.insert(0, os.getcwd())
import ref_firstlight as rf   # noqa: E402  (the pinned reference; its sha256 is checked by the caller)

EOS = 200006
CAP = 256
# ERRATUM 2026-11-24, correction 2: generate() computes a 1-row head at every step, this forward the multi-row head;
# the two round differently (at most 1.717e-05 measured on c1_000's step 0), so a position whose reference margin is
# below this bound is flagged as head-shape sensitive and reported; a flag changes no verdict.
HEAD_SHAPE_BOUND = 1e-4


def i32(t):
    return t.contiguous().view(torch.int32)


def seq_key(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_ref_ids(path):
    ref = {}
    for line in open(path):
        r = json.loads(line)
        if r["file"] == "claim1-prompts.jsonl":
            ref[r["id"]] = r["ids"]
    return ref


def load_engine(path, want, trunc):
    recs = {}
    for line in open(path):
        r = json.loads(line)
        if want and r["id"] not in want:
            continue
        gen = r["gen_ids"][:trunc] if trunc else r["gen_ids"]
        recs[r["id"]] = {"id": r["id"], "category": r.get("category"), "prompt_ids": r["prompt_ids"], "gen_ids": gen}
    return recs


def teacher_forced(model, prompt_ids, gen_ids):
    """one plain forward over s; per generated position the reference's argmax, margin, exact tie, agreement"""
    s = prompt_ids + gen_ids
    P, L = len(prompt_ids), len(gen_ids)
    x = torch.tensor([s], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids=x, use_cache=False).logits[0].float().contiguous()
    dt = time.time() - t0
    rows = logits[P - 1:P - 1 + L].contiguous()
    top = torch.topk(rows, 2, dim=-1)
    v0, v1, am = top.values[:, 0].contiguous(), top.values[:, 1].contiguous(), top.indices[:, 0]
    g = torch.tensor(gen_ids, dtype=torch.long)
    gl = rows[torch.arange(L), g].contiguous()
    agree = (i32(gl) == i32(v0)).tolist()
    tie = (i32(v0) == i32(v1)).tolist()
    return {"key": seq_key(s), "P": P, "L": L, "forward_s": dt,
            "rows_sha256": hashlib.sha256(memoryview(rows.numpy()).cast("B")).hexdigest(),
            "argmax": am.tolist(), "margin": (v0 - v1).tolist(), "agree": agree,
            "tie_positions": [j for j, t in enumerate(tie) if t]}


def verdict(rec, seq, ref_ids, cap):
    gen = rec["gen_ids"]
    L = len(gen)
    enc = rec["prompt_ids"] == ref_ids.get(rec["id"])
    complete = (L < cap and L > 0 and gen[-1] == EOS) or L == cap
    fd = next((j for j, a in enumerate(seq["agree"]) if not a), None)
    flagged = [j for j, m in enumerate(seq["margin"]) if m < HEAD_SHAPE_BOUND]
    # the verdict depends on a flagged position when one lies at or before the first difference (or anywhere, for an
    # identical answer): there the 1-row head generate() uses could have chosen otherwise
    upto = fd if fd is not None else L - 1
    depends = any(j <= upto for j in flagged)
    out = {"id": rec["id"], "category": rec["category"], "P": seq["P"], "L": L, "encoding_equal": enc,
           "complete": complete, "all_agree": fd is None, "identical": bool(enc and complete and fd is None),
           "first_diff": fd, "n_exact_ties": len(seq["tie_positions"]), "rows_sha256": seq["rows_sha256"],
           "head_shape_flagged": flagged, "verdict_depends_on_flagged": depends, "key": seq["key"]}
    if fd is not None:
        out.update({"gen_token_at_first_diff": gen[fd], "ref_argmax_at_first_diff": seq["argmax"][fd],
                    "ref_margin_at_first_diff": seq["margin"][fd]})
    return out


def quartiles(xs):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return {"min": xs[0], "p25": xs[0], "median": xs[0], "p75": xs[0], "max": xs[0]}
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return {"min": xs[0], "p25": q[0], "median": q[1], "p75": q[2], "max": xs[-1]}


def summary(verdicts):
    k = sum(v["identical"] for v in verdicts)
    cats = {}
    for v in verdicts:
        c = cats.setdefault(v["category"], [0, 0])
        c[0] += v["identical"]
        c[1] += 1
    nd = [v for v in verdicts if not v["identical"] and v["first_diff"] is not None]
    fds = [v["first_diff"] for v in nd]
    return {"n": len(verdicts), "identical": k, "per_category": {c: {"identical": a, "n": n} for c, (a, n) in cats.items()},
            "encoding_discrepancies": [v["id"] for v in verdicts if not v["encoding_equal"]],
            "incomplete": [v["id"] for v in verdicts if not v["complete"]],
            "exact_tie_positions": sum(v["n_exact_ties"] for v in verdicts),
            "head_shape_bound": HEAD_SHAPE_BOUND,
            "head_shape_flagged_positions": sum(len(v["head_shape_flagged"]) for v in verdicts),
            "prompts_with_flagged_positions": [v["id"] for v in verdicts if v["head_shape_flagged"]],
            "verdicts_depending_on_flagged": [v["id"] for v in verdicts if v["verdict_depends_on_flagged"]],
            "first_diff": quartiles(fds), "margin_at_first_diff": quartiles([v["ref_margin_at_first_diff"] for v in nd]),
            "first_diff_buckets": {"j=0": sum(f == 0 for f in fds), "1-15": sum(1 <= f <= 15 for f in fds),
                                   "16-63": sum(16 <= f <= 63 for f in fds), "64-255": sum(64 <= f <= 255 for f in fds)}}


def control_check(ctrl, engine_verdicts):
    """PREREG 0.1: the teacher-forced verdict on each control prompt equals the one read from true greedy generation"""
    bad = []
    rows = []
    for p in ctrl["prompts"]:
        pid, true_gen = p["id"], p["gen_ids"]
        for name, vs in engine_verdicts.items():
            v = vs.get(pid)
            if v is None:
                bad.append((name, pid, "no engine output"))
                continue
            gen = v["_gen"]
            n = min(len(gen), len(true_gen))
            d = next((j for j in range(n) if gen[j] != true_gen[j]), None)
            if d is None and len(gen) != len(true_gen):
                d = n
            direct = {"identical": d is None and v["encoding_equal"], "first_diff": d,
                      "ref_token": true_gen[d] if d is not None and d < len(true_gen) else None}
            tf = {"identical": v["identical"], "first_diff": v["first_diff"],
                  "ref_token": v.get("ref_argmax_at_first_diff")}
            same = direct == tf
            rows.append({"engine": name, "id": pid, "true_greedy": direct, "teacher_forced": tf, "same": same})
            if not same:
                bad.append((name, pid, "verdicts differ"))
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("map"); ap.add_argument("ref_ids"); ap.add_argument("out")
    ap.add_argument("--engine", action="append", required=True, help="NAME=answers.jsonl")
    ap.add_argument("--ids", default=""); ap.add_argument("--truncate", type=int, default=0)
    ap.add_argument("--control", default=""); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    assert a.threads <= 4, "the pinned reference runs at 4 threads"
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    want = set(a.ids.split(",")) if a.ids else None
    cap = a.truncate or CAP
    ref_ids = load_ref_ids(a.ref_ids)
    engines = {}
    for spec in a.engine:
        name, path = spec.split("=", 1)
        engines[name] = {"path": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                         "recs": load_engine(path, want, a.truncate)}
        if want:
            assert set(engines[name]["recs"]) == want, (name, sorted(want - set(engines[name]["recs"])))
    seqs_path = a.out + ".seqs.jsonl"
    done = {}
    if os.path.exists(seqs_path):
        for line in open(seqs_path):
            r = json.loads(line)
            done[r["key"]] = r
    todo = {}
    for e in engines.values():
        for rec in e["recs"].values():
            k = seq_key(rec["prompt_ids"] + rec["gen_ids"])
            if k not in done:
                todo.setdefault(k, rec)
    info = {"tool": "verify_tf.py", "version": 2, "head_shape_bound": HEAD_SHAPE_BOUND, "torch": torch.__version__,
            "threads": a.threads, "cap": cap,
            "truncate": a.truncate, "ids": sorted(want) if want else "all",
            "engines": {n: {"path": e["path"], "sha256": e["sha256"], "n": len(e["recs"])} for n, e in engines.items()},
            "ref_ids_sha256": hashlib.sha256(open(a.ref_ids, "rb").read()).hexdigest(),
            "ref_firstlight_sha256": hashlib.sha256(open(rf.__file__, "rb").read()).hexdigest(),
            "sequences_resumed": len(done), "sequences_to_run": len(todo)}
    t_all = time.time()
    if todo:
        sa = argparse.Namespace(source="real", dir=a.ckpt, map=a.map, mode="patched", dry=False, check_plain=False)
        model, patch, cfg = rf.setup_model(sa, info)
        with open(seqs_path, "a") as f:
            for i, (k, rec) in enumerate(sorted(todo.items(), key=lambda kv: kv[1]["id"])):
                seq = teacher_forced(model, rec["prompt_ids"], rec["gen_ids"])
                assert seq["key"] == k
                f.write(json.dumps(seq) + "\n")
                f.flush()
                os.fsync(f.fileno())
                done[k] = seq
                rf.log("%d/%d %s: P %d, L %d, agree %d of %d, ties %d, forward %.1f s; %s"
                       % (i + 1, len(todo), rec["id"], seq["P"], seq["L"], sum(seq["agree"]), seq["L"],
                          len(seq["tie_positions"]), seq["forward_s"], rf.mem_line()))
        if patch is not None:
            info["materialized_floats"] = patch.materialized
        info["peak_rss_gb"] = rf.peak_rss_gb()
    per_engine, ev = {}, {}
    for name, e in engines.items():
        vs = []
        ev[name] = {}
        for pid in sorted(e["recs"]):
            rec = e["recs"][pid]
            v = verdict(rec, done[seq_key(rec["prompt_ids"] + rec["gen_ids"])], ref_ids, cap)
            vs.append(v)
            ev[name][pid] = dict(v, _gen=rec["gen_ids"])
        per_engine[name] = {"summary": summary(vs), "verdicts": vs}
    rc = 0
    if a.control:
        ctrl = json.load(open(a.control))
        info["control"] = {"path": a.control, "sha256": hashlib.sha256(open(a.control, "rb").read()).hexdigest(),
                           "max_new_tokens": ctrl.get("max_new_tokens"), "step0_all_equal": ctrl.get("step0_all_equal")}
        assert a.truncate == ctrl.get("max_new_tokens"), "--truncate must equal the control's max_new_tokens"
        rows, bad = control_check(ctrl, ev)
        info["control"]["rows"] = rows
        info["control"]["discrepancies"] = bad
        rc = 4 if bad or not ctrl.get("step0_all_equal") else 0
        rf.log("control: %d rows, %d discrepancies, step-0 all equal %s" % (len(rows), len(bad), ctrl.get("step0_all_equal")))
    info["total_s"] = time.time() - t_all
    json.dump({"info": info, "engines": per_engine}, open(a.out + ".json", "w"), indent=1)
    for name, pe in per_engine.items():
        s = pe["summary"]
        rf.log("%s: identical %d of %d; encoding discrepancies %s; incomplete %s; exact-tie positions %d; "
               "head-shape flagged positions %d, verdicts depending on them %s"
               % (name, s["identical"], s["n"], s["encoding_discrepancies"], s["incomplete"], s["exact_tie_positions"],
                  s["head_shape_flagged_positions"], s["verdicts_depending_on_flagged"]))
    sys.exit(rc)


if __name__ == "__main__":
    main()
