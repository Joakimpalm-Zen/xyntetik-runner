#!/usr/bin/env python3
"""R4.48.12 PREREG 0.1 / ERRATUM 3: the 3-prompt true-greedy control on the pinned float32 reference.

It imports ref_firstlight.py (the same Store + Patch, the same setup_model; no copy of any model code) and, on each
prompt's template ids (the pre-flight's apply_chat_template ids, ref_ids.jsonl):
  1. the step-0 baseline, a plain forward over the prompt ids with use_cache=False AND logits_to_keep=1: the head
     computed as a 1-row product on the last hidden state, the SAME head shape generate() uses (ERRATUM 2026-11-24,
     correction 1: transformers' generate() passes logits_to_keep=1, and the head's 1-row product rounds differently
     from row P-1 of a P-row product); for information also the all-rows forward's row P-1 (ref_firstlight's read
     path), with its bit count and largest difference against step 0;
  2. model.generate(use_cache=True, do_sample=False, max_new_tokens=32), every generation setting given explicitly
     in a fresh GenerationConfig (the checkpoint's generation_config.json is not read, so no sampling default leaks
     in), eos 200006, output_logits=True: the raw logits of every step, before any processor;
  3. STEP-0 CHECK: generate's step-0 logits bit-equal to the use_cache=False, logits_to_keep=1 row (PREREG 0.1 as
     corrected by ERRATUM 2026-11-24); a difference is reported and the run exits 3 after writing its record
     (PREREG 4.2(3) voids the control then).
Version 2 (ERRATUM 2026-11-24). Version 1 (cec50141) compared step 0 with the all-rows row and was voided by the
head's product shape, not by a different model path.
GREEDY-CONSISTENCY CHECK: every generated token g_j (j < L) must be the raw-logit argmax of step j (at an exact top-2
tie, g_j must reach the maximum); a mismatch means a processor or default changed the choice despite the explicit
GenerationConfig, recorded per prompt (greedy_consistent, the first j that differs) and the run exits 3, as for a
step-0 failure.
Per prompt it records the generated ids (up to 32; stops at 200006), each step's argmax, top-1 minus top-2 logit
margin and exact-tie flag (top-2 bit-equal), and the step logits' sha256; the comparison with the engines' outputs is
the verification step's, not this file's.

Environment (ERRATUM 1): <work>/venv (torch 2.14.0+cpu), PYTHONPATH=<transformers PR 51050 at 6173a2f6>/src,
OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4, run from the directory holding ref_firstlight.py.

usage: ref_generate.py CKPT_DIR MAP REF_IDS.jsonl OUT_PREFIX [--ids c1_000,c1_001,c1_002] [--max-new 32] [--threads 4]
"""
import argparse, hashlib, json, os, sys, time
import torch

sys.path.insert(0, os.getcwd())
import ref_firstlight as rf   # noqa: E402  (the pinned reference; its sha256 is checked by the caller)
from transformers import GenerationConfig   # noqa: E402

EOS = 200006


def sha(t):
    return hashlib.sha256(memoryview(t.contiguous().numpy()).cast("B")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("map"); ap.add_argument("ref_ids"); ap.add_argument("out")
    ap.add_argument("--ids", default="c1_000,c1_001,c1_002"); ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    assert a.threads <= 4, "the pinned reference runs at 4 threads"
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    want = a.ids.split(",")
    prompts = {}
    for line in open(a.ref_ids):
        r = json.loads(line)
        if r["file"] == "claim1-prompts.jsonl" and r["id"] in want:
            prompts[r["id"]] = r["ids"]
    assert sorted(prompts) == sorted(want), ("prompts not found", want, sorted(prompts))
    t_all = time.time()
    # ref_firstlight's own model setup: source real, mode patched, the map, no dry run, no plain check
    sa = argparse.Namespace(source="real", dir=a.ckpt, map=a.map, mode="patched", dry=False, check_plain=False)
    info = {"tool": "ref_generate.py", "version": 2, "torch": torch.__version__, "threads": a.threads,
            "max_new_tokens": a.max_new,
            "ref_firstlight_sha256": hashlib.sha256(open(rf.__file__, "rb").read()).hexdigest()}
    model, patch, cfg = rf.setup_model(sa, info)
    gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=a.max_new, min_new_tokens=0,
                          eos_token_id=EOS, pad_token_id=EOS, temperature=None, top_k=None, top_p=None, min_p=None,
                          repetition_penalty=None, use_cache=True, output_logits=True, return_dict_in_generate=True)
    info["generation_config"] = gc.to_dict()
    recs, bad = [], 0
    for pid in want:
        ids = prompts[pid]
        P = len(ids)
        x = torch.tensor([ids], dtype=torch.long)
        t0 = time.time()
        with torch.no_grad():
            base = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float().contiguous()
            base_all = model(input_ids=x, use_cache=False).logits[0, P - 1].float().contiguous()
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc)
        t2 = time.time()
        gen = out.sequences[0, P:].tolist()
        steps = [l[0].float().contiguous() for l in out.logits]
        step0_equal = bool(rf.bits_equal(steps[0], base))
        bad += not step0_equal
        per = []
        first_incons = None
        for j in range(len(gen)):
            lg = steps[j]
            mx = lg.max()
            ok = int(torch.argmax(lg)) == gen[j] or bool(rf.bits_equal(lg[gen[j]:gen[j] + 1], mx.reshape(1)))
            if not ok and first_incons is None:
                first_incons = j
        greedy_ok = first_incons is None
        bad += not greedy_ok
        for j, lg in enumerate(steps):
            top = torch.topk(lg, 2)
            per.append({"j": j, "argmax": int(top.indices[0]), "token": gen[j] if j < len(gen) else None,
                        "margin": float(top.values[0] - top.values[1]),
                        "exact_tie": bool(rf.bits_equal(top.values[0:1], top.values[1:2])), "logits_sha256": sha(lg)})
        rec = {"id": pid, "P": P, "gen_ids": gen, "L": len(gen), "stopped_at_eos": bool(gen and gen[-1] == EOS),
               "step0_bit_equal_use_cache_false": step0_equal,
               "step0_n_bits_differ": int(rf.n_bits_differ(steps[0], base)) if not step0_equal else 0,
               "step0_baseline": "use_cache=False, logits_to_keep=1 (ERRATUM 2026-11-24)",
               "info_step0_vs_all_rows_bit_equal": bool(rf.bits_equal(steps[0], base_all)),
               "info_step0_vs_all_rows_n_bits_differ": int(rf.n_bits_differ(steps[0], base_all)),
               "info_step0_vs_all_rows_max_abs_diff": float((steps[0] - base_all).abs().max()),
               "base_all_rows_sha256": sha(base_all),
               "greedy_consistent": greedy_ok, "greedy_first_inconsistent_j": first_incons,
               "base_row_sha256": sha(base), "forward_s": t1 - t0, "generate_s": t2 - t1, "steps": per}
        recs.append(rec)
        torch.save({"base_row": base, "base_all_rows_row": base_all, "step_logits": torch.stack(steps)},
                   "%s.%s.pt" % (a.out, pid))
        rf.log("%s: P %d, generated %d (eos %s), step-0 bit-equal %s, greedy-consistent %s, forward %.1f s, generate %.1f s; %s"
               % (pid, P, len(gen), rec["stopped_at_eos"], step0_equal, greedy_ok, t1 - t0, t2 - t1, rf.mem_line()))
    info.update({"prompts": recs, "step0_all_equal": all(r["step0_bit_equal_use_cache_false"] for r in recs),
                 "greedy_all_consistent": all(r["greedy_consistent"] for r in recs), "peak_rss_gb": rf.peak_rss_gb(),
                 "total_s": time.time() - t_all})
    if patch is not None:
        info["materialized_floats"] = patch.materialized
    json.dump(info, open(a.out + ".json", "w"), indent=1)
    rf.log("done: step-0 all equal %s, greedy all consistent %s; peak RSS %.2f GB"
           % (info["step0_all_equal"], info["greedy_all_consistent"], info["peak_rss_gb"]))
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    main()
