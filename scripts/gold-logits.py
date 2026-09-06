#!/usr/bin/env python3
"""Gold logit check: the model's own reference implementation (transformers,
float32, CPU) against one or two serving endpoints on the same text.

Every cross-engine gate in this repository compares the runner with
llama.cpp, and llama.cpp is a second implementation, not the truth: it has
its own attention precision (its CPU flash-attention kernel accumulates in
f16) and its own quantization of activations. When the two disagree, this
script asks the question the gates cannot: which one is closer to what the
model's authors ship. It scores the reference implementation in float32
against each endpoint, position by position, on a fixed corpus, and reports
the KL divergence and top-1 agreement of each endpoint against the gold
distribution. Run it on an unquantized (bf16/f16) GGUF so weight rounding
is the same on every side and only the arithmetic differs.

    gold-logits.py --hf /path/to/hf-model --corpus tests/fixtures/mixed-corpus.txt \\
        --endpoint-a http://127.0.0.1:58631 --model-name-a X.gguf \\
        --endpoint-b http://127.0.0.1:58632 --model-name-b X.gguf \\
        --max-positions 100 --out gold.json

Endpoints speak /v1/completions with `logprobs` (llama-server and the runner
both do). The tokenizer is the HF one; the prefix at each position is the
decoded token prefix, so every side sees the same text.
"""
import argparse
import json
import math
import sys
import urllib.request


def query(endpoint, model_name, prompt, top_n=20):
    payload = {"model": model_name, "prompt": prompt, "max_tokens": 1,
               "temperature": 0, "logprobs": top_n}
    req = urllib.request.Request(
        f"{endpoint}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    lp = d["choices"][0]["logprobs"]
    top = lp["top_logprobs"][0]
    if isinstance(top, list):
        top = {t["token"]: t["logprob"] for t in top}
    return top


def kld(gold, other):
    """KL(gold || other) over gold's top tokens, with `other`'s mass renormalised
    over the same support; tokens missing from `other` get its floor."""
    floor = min(other.values()) - 2.0 if other else -20.0
    keys = list(gold.keys())
    g = [gold[k] for k in keys]
    o = [other.get(k, floor) for k in keys]
    gz = math.log(sum(math.exp(x) for x in g))
    oz = math.log(sum(math.exp(x) for x in o))
    return sum(math.exp(x - gz) * ((x - gz) - (y - oz)) for x, y in zip(g, o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF model directory")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--endpoint-a", required=True)
    ap.add_argument("--model-name-a", required=True)
    ap.add_argument("--endpoint-b")
    ap.add_argument("--model-name-b")
    ap.add_argument("--max-positions", type=int, default=100)
    ap.add_argument("--min-prefix", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32)
    model.eval()

    text = open(args.corpus, encoding="utf-8").read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    span = ids[: args.min_prefix + args.max_positions * args.stride + 1]
    with torch.no_grad():
        out = model(torch.tensor([span]))
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)

    rows = []
    sides = [("a", args.endpoint_a, args.model_name_a)]
    if args.endpoint_b:
        sides.append(("b", args.endpoint_b, args.model_name_b))
    for n in range(args.max_positions):
        pos = args.min_prefix + n * args.stride
        prefix = tok.decode(span[:pos])
        # the decoded prefix must re-tokenize to the same ids, or the sides
        # would be scored on different contexts
        if tok(prefix, add_special_tokens=False)["input_ids"] != span[:pos]:
            continue
        top = torch.topk(logp[pos - 1], args.top_n)
        gold = {tok.decode([int(i)]): float(v) for v, i in zip(top.values, top.indices)}
        gold_top1 = tok.decode([int(top.indices[0])])
        row = {"pos": pos, "gold_top1": gold_top1, "gold_margin":
               float(top.values[0] - top.values[1])}
        for name, ep, mn in sides:
            other = query(ep, mn, prefix, args.top_n)
            o_top1 = max(other, key=other.get)
            row[name] = {"kld": kld(gold, other), "top1": o_top1,
                         "agree": o_top1 == gold_top1}
        rows.append(row)
        print(f"pos {pos:4d} " + " ".join(
            f"{k}: kld {row[k]['kld']:.4f} {'ok' if row[k]['agree'] else 'DIFF'}"
            for k in row if k in ("a", "b")), file=sys.stderr)

    summary = {"positions": len(rows), "hf": args.hf, "corpus": args.corpus,
               "sides": {}}
    for name, ep, mn in sides:
        ks = [r[name]["kld"] for r in rows]
        summary["sides"][name] = {
            "endpoint": ep, "model": mn,
            "mean_kld_vs_gold": sum(ks) / max(1, len(ks)),
            "max_kld_vs_gold": max(ks, default=0.0),
            "top1_agreement_pct": 100.0 * sum(r[name]["agree"] for r in rows) / max(1, len(rows)),
            "margin_qualified_top1_pct": 100.0 * sum(
                r[name]["agree"] for r in rows if r["gold_margin"] > 0.5)
            / max(1, sum(1 for r in rows if r["gold_margin"] > 0.5)),
        }
    report = {"schema_version": "xyntetik.runner.gold-logits.v1",
              "summary": summary, "rows": rows}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
