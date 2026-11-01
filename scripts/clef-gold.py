#!/usr/bin/env python3
"""Gold outputs for Clef from the publisher's own code, and the comparison.

The reference is Cloudflare's joint_schema_model.py (encode_record,
ClefModel, systemone), run in float32 on the CPU. Nothing here reimplements
it: the module is loaded from the file the model repository ships, after its
sha256 is checked against the revision the runner was ported from.

Reference (needs torch and transformers):

    clef-gold.py reference --model DIR --code joint_schema_model.py \\
        --records tests/fixtures/clef-records.jsonl --out ref.jsonl [--head-bf16]

DIR is a Clef checkpoint (config.json, the backbone safetensors,
joint_head.safetensors, joint_head_config.json, tokenizer and processor
files). --head-bf16 rounds the head's matrices to bfloat16 first, which is
how llama.cpp's converter stores them in a BF16 GGUF (all but token_types,
kept in F32); use it when the GGUF under test was converted from a float32
head. --encode-only skips the model
and writes only the encoded records (ids and spans), from DIR's tokenizer.

Comparison (plain python):

    clef-gold.py compare --ref ref.jsonl --runner runner.jsonl

runner.jsonl is `runner --systemone` output for the same records with
"runner_logits": true added to each (`clef-gold.py requests` writes that
file). Reports token-id and span agreement, the largest logit and
probability differences, and how many answers agree.

Decisions only, for any /v1/systemone server (llama.cpp's included):
    clef-gold.py compare --decisions --ref ref.jsonl --runner replies.jsonl [--json out.json]
replies.jsonl holds one /v1/systemone reply per record, in the requests'
order. Each question's per-option probabilities are read back from the reply
(a choice's by criteria key, a noul's as true/false, a score's by index) and
scored against the reference's softmax: decisions that agree, the largest and
mean |dprob|, and each flip with the reference's margin. A server that
returns no input ids, spans or logits gets no check of them here.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys

# joint_schema_model.py at Cloudflare/clef 0b331204 (identical in
# Cloudflare/clef-flash): the file src/clef.c was ported from
CODE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"


def read_jsonl(path):
    # split on "\n" only: str.splitlines() also breaks at U+2028, which a
    # record may hold inside a string
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n") if line.strip()]


def load_code(path, any_code):
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CODE_SHA256 and not any_code:
        sys.exit(f"{path}: sha256 {digest} is not the ported revision {CODE_SHA256} "
                 "(pass --any-code to run it anyway)")
    spec = importlib.util.spec_from_file_location("joint_schema_model", path)
    mod = importlib.util.module_from_spec(spec)
    # its dataclasses resolve their (postponed) annotations through
    # sys.modules, so the module has to be registered before it runs
    sys.modules["joint_schema_model"] = mod
    spec.loader.exec_module(mod)
    return mod, digest


def encoded_json(enc):
    return {
        "input_ids": list(enc.input_ids),
        "questions": [
            {
                "id": q.question_id,
                "type": q.question_type,
                "question_span": list(q.question_span),
                "option_ids": list(q.option_ids),
                "option_spans": [list(s) for s in q.option_spans],
            }
            for q in enc.questions
        ],
    }


def load_model(jsm, path, torch):
    """load_release_model, or the same steps with the tokenizer alone.

    The publisher's loader builds the full image/video processor, which needs
    torchvision. Text-only records use the processor for its tokenizer only
    (encode_record reaches the processor for media alone), so where
    torchvision is missing the loader's own lines run with AutoTokenizer in
    the processor's place. The model, the head and every computation are the
    publisher's either way."""
    try:
        model, processor = jsm.load_release_model(path, device="cpu", dtype=torch.float32)
        return model, processor, "load_release_model"
    except ImportError as e:
        if "orchvision" not in str(e):
            raise
    import types
    from pathlib import Path
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    path = Path(path)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.float32, device_map={"": "cpu"})
    backbone.config.use_cache = False
    head = jsm.JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    head = head.to(device="cpu", dtype=torch.float32)
    processor = types.SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(path))
    return jsm.ClefModel(backbone, head).eval(), processor, \
        "load_release_model's steps with AutoTokenizer (no torchvision here)"


def cmd_reference(a):
    import torch

    torch.set_num_threads(a.threads)
    jsm, digest = load_code(a.code, a.any_code)
    records = read_jsonl(a.records)
    header = {"reference": "joint_schema_model.py", "code_sha256": digest,
              "torch": torch.__version__}
    if a.encode_only:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model)
        header["mode"] = "encode-only"
        out = [header]
        for r in records:
            enc = jsm.encode_record(tok, r, max_length=r.get("max_length", 16384))
            out.append({"id": r.get("id"), **encoded_json(enc)})
    else:
        import transformers

        model, processor, loader = load_model(jsm, a.model, torch)
        if a.head_bf16:
            # the converter's choice in a BF16 file: every matrix BF16 except
            # the type embedding (token_types), which it keeps in F32
            with torch.no_grad():
                for name, p in model.head.named_parameters():
                    if p.ndim == 2 and name != "type_embedding.weight":
                        p.copy_(p.to(torch.bfloat16).to(torch.float32))
        lm = model.language_model
        header.update({
            "mode": "logits", "transformers": transformers.__version__,
            "dtype": "float32", "device": "cpu", "head_bf16": bool(a.head_bf16),
            "attn_implementation": getattr(lm.config, "_attn_implementation", None),
            "loader": loader,
        })
        out = [header]
        for r in records:
            max_length = r.get("max_length", 16384)
            enc = jsm.encode_record(processor.tokenizer, r, max_length=max_length,
                                    processor=processor)
            batch = jsm.collate_records([enc], processor.tokenizer.pad_token_id,
                                        torch.device("cpu"))
            with torch.inference_mode():
                logits = model(batch)[0]
            row = {"id": r.get("id"), **encoded_json(enc)}
            if "max_length" in r:
                row["max_length"] = max_length
            for q, lg in zip(row["questions"], logits):
                q["logits"] = [float(x) for x in lg.float().tolist()]
            # systemone() always encodes with its own 16384 bound
            if "max_length" not in r:
                row["systemone"] = jsm.systemone(model, processor, r)
            out.append(row)
            print(f"{r.get('id')}: {len(enc.input_ids)} tokens", file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_requests(a):
    with open(a.out, "w", encoding="utf-8") as f:
        for r in read_jsonl(a.records):
            if "max_length" in r and not a.keep_bounded:
                continue
            r = dict(r)
            r["runner_logits"] = True
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def reply_probs(q, ans):
    """A /v1/systemone answer's probabilities in the reference's option
    order, or None when the answer does not carry them."""
    if not isinstance(ans, dict):
        return None
    if ans.get("type") == "noul" and isinstance(ans.get("noul"), (int, float)):
        return [float(ans["noul"]), 1.0 - float(ans["noul"])]   # option 0 is true
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        return None
    keys = [str(k) for k in q["option_ids"]] if ans.get("type") == "choice" \
        else [str(k) for k in range(len(q["logits"]))]
    if any(k not in probs for k in keys):
        return None
    return [float(probs[k]) for k in keys]


def cmd_compare_decisions(a):
    ref = read_jsonl(a.ref)
    head, ref = ref[0], [r for r in ref[1:] if a.keep_bounded or "max_length" not in r]
    run = read_jsonl(a.runner)
    if len(run) != len(ref):
        sys.exit(f"{len(ref)} reference records but {len(run)} reply lines")
    detail, missing = [], []
    for rf, rn in zip(ref, run):
        answers = rn.get("answers") or {}
        for q in rf["questions"]:
            p_ref = softmax(q["logits"])
            p_run = reply_probs(q, answers.get(q["id"]))
            if p_run is None or len(p_run) != len(p_ref):
                missing.append(f"{rf['id']}/{q['id']}")
                continue
            top = sorted(p_ref, reverse=True)
            cr = max(range(len(p_ref)), key=p_ref.__getitem__)
            cn = max(range(len(p_run)), key=p_run.__getitem__)
            detail.append({"record": rf["id"], "question": q["id"], "reference_choice": cr,
                           "server_choice": cn, "agree": cr == cn,
                           "reference_margin": round(top[0] - (top[1] if len(top) > 1 else 0.0), 6),
                           "max_abs_dprob": round(max(abs(x - y) for x, y in zip(p_ref, p_run)), 6)})
    n = len(detail)
    agree = sum(d["agree"] for d in detail)
    dps = [d["max_abs_dprob"] for d in detail]
    summary = {"questions": n, "decisions_agree": agree,
               "max_abs_dprob": max(dps) if dps else None,
               "mean_abs_dprob": round(sum(dps) / n, 6) if n else None,
               "unreadable": missing}
    print(f"reference: {head.get('reference')} sha256 {head.get('code_sha256', '')[:12]}, "
          f"{head.get('dtype', '?')} {head.get('device', '?')}")
    print(f"decisions agree {agree}/{n}, max |dprob| {summary['max_abs_dprob']}, "
          f"mean |dprob| {summary['mean_abs_dprob']} (no id, span or logit check)")
    for d in detail:
        if not d["agree"]:
            print(f"  flip {d['record']}/{d['question']}: reference margin {d['reference_margin']}, "
                  f"|dprob| {d['max_abs_dprob']}")
    for m in missing:
        print(f"  unreadable answer: {m}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"reference": head, **summary, "questions_detail": detail}, f, indent=1)
            f.write("\n")
    sys.exit(1 if missing else 0)


def cmd_compare(a):
    if a.decisions:
        return cmd_compare_decisions(a)
    ref = read_jsonl(a.ref)
    head, ref = ref[0], [r for r in ref[1:] if a.keep_bounded or "max_length" not in r]
    run = read_jsonl(a.runner)
    if len(run) != len(ref):
        sys.exit(f"{len(ref)} reference records but {len(run)} runner lines")
    worst_l = worst_p = 0.0
    ids_ok = spans_ok = ans_ok = n_ans = 0
    failures = []
    for rf, rn in zip(ref, run):
        if "error" in rn:
            failures.append(f"{rf['id']}: runner error {rn['error']}")
            continue
        tr = rn.get("runner_logits") or {}
        if tr.get("input_ids") == rf["input_ids"]:
            ids_ok += 1
        else:
            got, want = tr.get("input_ids") or [], rf["input_ids"]
            at = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            failures.append(f"{rf['id']}: input ids differ at {at} ({len(got)} vs {len(want)} tokens)")
        qr, qn = rf["questions"], tr.get("questions") or []
        same = len(qr) == len(qn) and all(
            a_["id"] == b_["id"] and a_["question_span"] == b_["question_span"] and
            a_["option_ids"] == b_["option_ids"] and a_["option_spans"] == b_["option_spans"]
            for a_, b_ in zip(qr, qn))
        spans_ok += same
        if not same:
            failures.append(f"{rf['id']}: question or option spans differ")
            continue
        if a.encode:
            continue
        for a_, b_ in zip(qr, qn):
            dl = max(abs(x - y) for x, y in zip(a_["logits"], b_["logits"]))
            dp = max(abs(x - y) for x, y in zip(softmax(a_["logits"]), softmax(b_["logits"])))
            worst_l, worst_p = max(worst_l, dl), max(worst_p, dp)
            if dl > a.tol:
                failures.append(f"{rf['id']}/{a_['id']}: max |dlogit| {dl:.3g} > {a.tol}")
        if "systemone" in rf:
            want, got = rf["systemone"]["answers"], rn.get("answers", {})
            for k, v in want.items():
                n_ans += 1
                ans_ok += got.get(k) == v
    n = len(ref)
    print(f"reference: {head.get('reference')} sha256 {head.get('code_sha256', '')[:12]}, "
          f"{head.get('dtype', '?')} {head.get('device', '?')}, attn {head.get('attn_implementation')}, "
          f"head_bf16 {head.get('head_bf16')}")
    print(f"records {n}: input ids equal {ids_ok}/{n}, spans equal {spans_ok}/{n}")
    if not a.encode:
        print(f"max |dlogit| {worst_l:.3g}, max |dprob| {worst_p:.3g}, "
              f"answers byte-equal after rounding {ans_ok}/{n_ans}")
    for f in failures:
        print("  " + f)
    sys.exit(1 if failures else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--model", required=True)
    r.add_argument("--code", required=True)
    r.add_argument("--records", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--head-bf16", action="store_true")
    r.add_argument("--encode-only", action="store_true")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--any-code", action="store_true")
    q = sub.add_parser("requests")
    q.add_argument("--records", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--keep-bounded", action="store_true",
                   help="keep records that set max_length (run them with -c max_length)")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--runner", required=True)
    c.add_argument("--tol", type=float, default=1e-3)
    c.add_argument("--encode", action="store_true", help="compare ids and spans only")
    c.add_argument("--keep-bounded", action="store_true")
    c.add_argument("--decisions", action="store_true",
                   help="any /v1/systemone server's replies: decisions and probabilities only")
    c.add_argument("--json", help="with --decisions: write the per-question comparison here")
    a = ap.parse_args()
    {"reference": cmd_reference, "requests": cmd_requests, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
