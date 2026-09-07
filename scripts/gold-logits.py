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
    lp = d["choices"][0].get("logprobs")
    # A server may answer a position with no logprobs at all (measured
    # 2026-09-07 on SmolLM2: the request whose next token is the end of
    # text comes back without the block). That is a position this gate
    # cannot score, not a reason to lose the whole family.
    if not lp:
        return None
    if "content" in lp:  # llama.cpp's OpenAI-style schema
        step = lp["content"][0]
        top = {e["token"]: e["logprob"] for e in step["top_logprobs"]}
        top[step["token"]] = step["logprob"]
    else:  # the runner's schema: parallel arrays, top_logprobs a dict
        top = dict(lp["top_logprobs"][0])
        top[lp["tokens"][0]] = lp["token_logprobs"][0]
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
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="the publisher ships its own modeling code")
    ap.add_argument("--force-bos", action="store_true",
                    help="prepend the tokenizer's BOS to the reference even "
                         "when the HF tokenizer does not add one. Needed "
                         "where the SERVED artifact declares add_bos_token "
                         "but the publisher's loaded tokenizer does not add "
                         "it (measured 2026-09-07 on Gemma 4 E2B): the "
                         "endpoints follow the artifact, so the reference "
                         "must see the same prefix or the two are compared "
                         "across a one-token offset.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=args.trust_remote_code)
    # A publisher whose top-level class is multimodal (Qwen3.5, Gemma 4,
    # Muse Glimmer) still answers text through the same language model;
    # the causal-LM auto class refuses those configs, so fall back to the
    # image-text class and drive it with text only. The config name is
    # printed so the report says which class produced the gold.
    kw = dict(dtype=torch.float32, trust_remote_code=args.trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.hf, **kw)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.hf, **kw)
    print("reference class:", type(model).__name__, file=sys.stderr)
    model.eval()

    text = open(args.corpus, encoding="utf-8").read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body = ids[: args.min_prefix + args.max_positions * args.stride + 1]
    # The serving endpoints tokenize the prefix TEXT, so a family whose
    # tokenizer prepends BOS gives the model one more token than a reference
    # scored on the bare ids. Measured 2026-09-07: without this the gemma-3,
    # gemma-4 and Phi-3.5 rows read mean KL around 2 to 5 with both engines
    # agreeing with each other and disagreeing with the reference, which is
    # the signature of a broken instrument rather than a broken engine
    # (qwen3 and granite, whose tokenizers add nothing, read 0.0000). The
    # special prefix is taken from the tokenizer itself and prepended once;
    # position `pos` of the body is then row len(special) + pos - 1.
    # Taken by DIFFERENCE on a real word, not from the empty string: some
    # tokenizers add nothing to "" yet prepend BOS to any real input, and
    # measured 2026-09-07 that is exactly what Phi-3.5 and Gemma 4 do, which
    # left them reading mean KL around 2 and 5 after the first version of
    # this fix had already repaired Gemma 3.
    _probe = "word"
    _with = tok(_probe, add_special_tokens=True)["input_ids"]
    _without = tok(_probe, add_special_tokens=False)["input_ids"]
    n_special = len(_with) - len(_without)
    special = _with[:n_special] if n_special > 0 else []
    if args.force_bos and not special and tok.bos_token_id is not None:
        special = [tok.bos_token_id]
    span = list(special) + list(body)
    with torch.no_grad():
        out = model(torch.tensor([span]))
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)

    rows = []
    skipped = 0
    sides = [("a", args.endpoint_a, args.model_name_a)]
    if args.endpoint_b:
        sides.append(("b", args.endpoint_b, args.model_name_b))
    for n in range(args.max_positions):
        pos = args.min_prefix + n * args.stride
        prefix = tok.decode(body[:pos])
        # the decoded prefix must re-tokenize to the same ids, or the sides
        # would be scored on different contexts
        if tok(prefix, add_special_tokens=False)["input_ids"] != body[:pos]:
            continue
        top = torch.topk(logp[len(special) + pos - 1], args.top_n)
        gold = {tok.decode([int(i)]): float(v) for v, i in zip(top.values, top.indices)}
        gold_top1 = tok.decode([int(top.indices[0])])
        row = {"pos": pos, "gold_top1": gold_top1, "gold_margin":
               float(top.values[0] - top.values[1])}
        got = {name: query(ep, mn, prefix, args.top_n) for name, ep, mn in sides}
        if any(v is None for v in got.values()):
            skipped += 1
            continue
        for name, ep, mn in sides:
            other = got[name]
            o_top1 = max(other, key=other.get)
            row[name] = {"kld": kld(gold, other), "top1": o_top1,
                         "agree": o_top1 == gold_top1}
        rows.append(row)
        print(f"pos {pos:4d} " + " ".join(
            f"{k}: kld {row[k]['kld']:.4f} {'ok' if row[k]['agree'] else 'DIFF'}"
            for k in row if k in ("a", "b")), file=sys.stderr)

    summary = {"positions": len(rows), "hf": args.hf, "corpus": args.corpus,
               "reference_class": type(model).__name__,
               "special_prefix_tokens": len(special),
               "special_prefix_forced": bool(args.force_bos and special),
               "positions_skipped_no_logprobs": skipped,
               "reference_dtype": "float32", "reference_device": "cpu",
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
