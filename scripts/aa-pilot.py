#!/usr/bin/env python3
"""Timing A/A pilot: can this box resolve a ~5% effect on the GPU path?

Eight ABBA-interleaved pairs of IDENTICAL blocks (A and B are the same
block; the labels only mark the interleave), one block = N fixed chat
requests, greedy, non-thinking, max_tokens 64, against a runner server
with the model fully offloaded. Per block: wall seconds, prefill and decode
tok/s from the server's own counters (/health deltas), and the server
process's CPU-seconds (psutil). Statistic per metric: the log-ratio
ln(A/B) over the 8 pairs, its mean, the 95% half-width (t, 7 df) and the
resolution 2 x half-width; the lab's gate is resolution <= 0.10.

    python scripts/aa-pilot.py --runner runner.exe --model Qwen3-4B-Q4_K_M.gguf \
        --out docs/benchmarks-raw/aa-pilot-rtx3070-2026-10-07.json [--pairs 8] [--requests 20]

Written for the ERASE feasibility question (Blackwell lab, 2026-10-06): a
pilot, not a G0-I result. Nothing here quiets the box; the report says so.
"""
import argparse, json, math, os, platform, subprocess, sys, time, urllib.request
try:
    import psutil
except ImportError:
    psutil = None

PROMPTS = [
    "Explain in two sentences why the sky is blue.", "Write a haiku about a lighthouse.",
    "List three uses of a paperclip.", "What is 17 times 23? Answer with the number only.",
    "Give one sentence of advice to a new programmer.", "Name the capital of Portugal and one fact about it.",
    "Describe a cup of coffee to someone who has never seen one, in two sentences.", "Translate 'good morning' into French and Spanish.",
    "Summarize the plot of Cinderella in one sentence.", "What does HTTP stand for?",
    "Write a two-line rhyme about rain.", "Explain recursion to a child in two sentences.",
    "Which is heavier, a kilogram of feathers or a kilogram of iron?", "Give a one-sentence definition of entropy.",
    "Name three primary colours.", "What year did the first moon landing happen?",
    "Write one sentence that uses the word 'lantern'.", "Suggest a name for a grey cat and say why.",
    "How many legs does a spider have? Answer with the number only.", "Finish the sentence: The best thing about mornings is",
]

def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())

def block(base, model, n_req, proc):
    h0 = get(base + "/health"); c0 = proc.cpu_times() if proc else None; t0 = time.perf_counter()
    first = None
    for i in range(n_req):
        p = PROMPTS[i % len(PROMPTS)]
        d = post(base + "/v1/chat/completions", {"model": model, "messages": [
            {"role": "system", "content": "/no_think"}, {"role": "user", "content": p}],
            "max_tokens": 64, "temperature": 0})
        if first is None: first = d["choices"][0]["message"]["content"][:80]
    wall = time.perf_counter() - t0
    h1 = get(base + "/health"); c1 = proc.cpu_times() if proc else None
    gen = h1["tokens_generated"] - h0["tokens_generated"]
    ptok = h1["tokens_prompt"] - h0["tokens_prompt"]
    gsec = h1["generate_seconds"] - h0["generate_seconds"]
    cpu = (c1.user + c1.system) - (c0.user + c0.system) if c0 else None
    return {"wall_s": wall, "prompt_tokens": ptok, "gen_tokens": gen, "generate_s": gsec,
            "decode_tok_s": gen / gsec if gsec > 0 else None,
            "prefill_tok_s": ptok / (wall - gsec) if wall > gsec else None,
            "cpu_s": cpu, "first_reply": first}

def stats(pairs, key):
    lr = [math.log(a[key] / b[key]) for a, b in pairs if a.get(key) and b.get(key)]
    n = len(lr)
    if n < 2: return {"n": n}
    m = sum(lr) / n; sd = math.sqrt(sum((x - m) ** 2 for x in lr) / (n - 1))
    t = {7: 2.365, 6: 2.447, 5: 2.571, 4: 2.776, 3: 3.182, 2: 4.303, 1: 12.706}.get(n - 1, 2.0)
    hw = t * sd / math.sqrt(n)
    return {"n": n, "mean_log_ratio": m, "sd": sd, "half_width_95": hw, "resolution": 2 * hw,
            "gate_0.10": 2 * hw <= 0.10}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--port", type=int, default=18820)
    ap.add_argument("--pairs", type=int, default=8); ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=4096)
    a = ap.parse_args()
    base = "http://127.0.0.1:%d" % a.port
    srv = subprocess.Popen([a.runner, "--serve", "--port", str(a.port), "-m", a.model, "-c", str(a.ctx)],
                           stdout=open(a.out + ".server.log", "w"), stderr=subprocess.STDOUT)
    proc = psutil.Process(srv.pid) if psutil else None
    for _ in range(120):
        try: get(base + "/health"); break
        except Exception: time.sleep(2)
    model = get(base + "/v1/models")["data"][0]["id"]
    caps, prov = {}, {}
    try: caps = get(base + "/v1/capabilities")
    except Exception: pass
    try: prov = get(base + "/v1/runner/provenance")   # binary + model sha256, device profile
    except Exception: pass
    # warm-up block (not scored): page the weights, JIT the kernels
    warm = block(base, model, 5, proc)
    order = []
    for _ in range(a.pairs): order += ["A", "B", "B", "A"]
    order = order[:2 * a.pairs]
    blocks = []
    for i, lab in enumerate(order):
        b = block(base, model, a.requests, proc); b["label"] = lab; b["index"] = i
        blocks.append(b); print("block %2d %s wall %.2fs decode %.1f prefill %.0f cpu %s" % (
            i, lab, b["wall_s"], b["decode_tok_s"] or 0, b["prefill_tok_s"] or 0,
            "%.2f" % b["cpu_s"] if b["cpu_s"] is not None else "n/a"), flush=True)
    srv.terminate()
    A = [b for b in blocks if b["label"] == "A"]; B = [b for b in blocks if b["label"] == "B"]
    pairs = list(zip(A, B))
    res = {"schema_version": "xyntetik.runner.aa-pilot.v1", "box": platform.node(), "platform": platform.platform(),
           "runner": a.runner, "runner_version": caps.get("version"),
           "binary_sha256": (prov.get("build") or {}).get("binary_sha256"),
           "profile": prov.get("profile"),
           "model": os.path.basename(a.model), "model_sha256": (prov.get("model") or {}).get("sha256"),
           "design": "ABBA x %d pairs, %d greedy non-thinking requests of max_tokens 64 per block, warm-up block excluded, box not quieted" % (a.pairs, a.requests),
           "warmup": warm, "blocks": blocks,
           "stats": {k: stats(pairs, k) for k in ("wall_s", "decode_tok_s", "prefill_tok_s", "cpu_s")}}
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["stats"].items(): print(k, json.dumps(v))

if __name__ == "__main__":
    main()
