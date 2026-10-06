# Per-position next-token logprob from a running llama-server, over every
# prefix of the token ids in a `runner --score` JSON, as the llama.cpp column
# of an admission anchor (docs/qwen4exp-admission-evidence/README.md).
#   python3 scripts/anchor-llama-logprobs.py PORT runner-score.json out.json [N_VOCAB]
# One /completion per prefix: n_probs = the whole vocabulary, temperature 0,
# no prompt cache, ignore_eos, and logit_bias [[1, 100]] so the sampled token
# is valid UTF-8 (llama-server drops completion_probabilities when the sample
# is an incomplete multibyte sequence; the bias changes the sample, not the
# reported pre-sampling distribution). Ids go in as ids, so the tokenizer is
# out of the comparison. Output has runner --score's "logprobs" shape.
import json, sys, urllib.request, math
port, toks_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
n_vocab = int(sys.argv[4]) if len(sys.argv) > 4 else 259
toks = json.load(open(toks_path))["tokens"]
lps, top1 = [], []
for i in range(1, len(toks)):
    body = {"prompt": toks[:i], "n_predict": 1, "temperature": 0, "n_probs": n_vocab,
            "cache_prompt": False, "ignore_eos": True, "logit_bias": [[1, 100.0]]}
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=120)
    d = json.loads(r.read())
    if "completion_probabilities" not in d:
        print("no probabilities at prefix", i, "prompt_ms", d.get("timings",{}).get("prompt_ms")); lps.append(None); top1.append(None); continue
    cp = d["completion_probabilities"][0]
    # llama.cpp reports probs (0..1) by default; take log; find the target
    tgt = toks[i]
    ents = {e["id"]: e for e in cp["top_logprobs"]}
    e = ents.get(tgt)
    if e is None:
        lps.append(None); top1.append(None); continue
    lp = e["logprob"] if "logprob" in e else math.log(e["prob"])
    lps.append(lp); top1.append(cp["id"] if "id" in cp else cp.get("token"))
json.dump({"n_tokens": len(toks), "logprobs": lps, "top1_id": top1,
           "n_probs_returned": len(cp["top_logprobs"])}, open(out_path, "w"))
print("positions", len(lps), "none", sum(1 for x in lps if x is None),
      "n_probs_returned", len(cp["top_logprobs"]))
