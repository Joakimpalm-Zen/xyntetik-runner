#!/usr/bin/env python3
"""Run the workload and shape corpus (scripts/bench-corpus.json, plan P2).

    python3 scripts/bench-corpus.py --runner ./runner --model M.gguf --out R.json \\
        [--cases REGEX] [--repeat 2] [--threads N] [--ctx 4096] [--idle]

Each case runs through a driver the repository already has: --bench-json
(prefill and decode, with --draft-lookup for the verification widths and
--gpu-layers for the splits), a served chat for the concurrency and
tool-result cases, and scripts/idle_coexistence.py for the idle lifecycle
(only with --idle: it takes about four minutes and wants a quiet machine).

A record says what it is a measurement of: the machine and the runner's
--caps, the model's size and mtime (and sha256 with --hash), the threads,
the P1 route the bench reported, and the attempt (the first run of a case
reads its weights cold, a repeat warm). Prompts are built to the exact token
count a case names, calibrated through the server's usage. There are no
thresholds: compare two records of the same case on the same machine.
Results are machine data; keep them out of the repository.
"""
import argparse, hashlib, json, os, platform, re, socket, subprocess, sys, threading, time
import urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# one token a word on every tokenizer the corpus has met: single letters
WORDS = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split()


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Server:
    def __init__(self, exe, model, ctx, threads, parallel=1):
        self.port = free_port()
        self.args = [exe, "-m", model, "--serve", "--no-tray", "--port", str(self.port),
                     "-c", str(ctx), "--parallel", str(parallel)] + (["-t", str(threads)] if threads else [])

    def __enter__(self):
        self.p = subprocess.Popen(self.args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return self
            except Exception:
                time.sleep(0.1)
        self.p.kill()
        raise RuntimeError("server did not become healthy")

    def __exit__(self, *exc):
        self.p.terminate()
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.p.kill()

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            doc = json.load(r)
        return doc, time.time() - t0


def prompt_of(srv, n, offset=0):
    """A prompt of exactly n tokens as --bench-json counts it, or the
    closest one below: the server's count (BOS included) of a raw prompt,
    shifted by `offset`, the difference measured between the two counts."""
    n -= offset
    def count(text):
        try:
            doc, _ = srv.post("/v1/completions", {"prompt": text, "max_tokens": 1, "temperature": 0})
        except urllib.error.HTTPError:
            return 1 << 30   # refused as longer than the context: over any target
        return doc["usage"]["prompt_tokens"]
    text = (WORDS * (n // len(WORDS) + 2))[:n + 2]   # a word is at least one token
    lo, hi = 1, len(text)
    best = 1
    while lo <= hi:   # the longest word prefix that is not over n
        mid = (lo + hi) // 2
        c = count(" ".join(text[:mid]))
        if c <= n:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return " ".join(text[:best]), count(" ".join(text[:best]))


def bench(exe, model, prompt, case, threads, gpu_layers):
    argv = [exe, "-m", model, "--bench-json", "-p", prompt, "-n", str(case["n_predict"])]
    if threads:
        argv += ["-t", str(threads)]
    if gpu_layers is not None:
        argv += ["--gpu-layers", str(gpu_layers)]
    if case.get("draft") == "self":
        # greedy, so the model's own drafts are accepted whole and every round
        # verifies the full k+1 rows
        argv += ["--draft", model, "--draft-k", str(case["draft_k"]), "--temp", "0"]
    env = dict(os.environ, RUNNER_SPEC_STATS="1") if case.get("draft") else None
    p = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env)
    if p.returncode != 0:
        return {"error": p.stderr.strip().splitlines()[-1:] or ["exit %d" % p.returncode]}
    out = json.loads(p.stdout.strip().splitlines()[-1])
    if case.get("draft"):
        spec = [l for l in p.stderr.splitlines() if l.startswith("spec:")]
        out["spec"] = spec[-1] if spec else None
    return out


def served_chat(srv, case):
    msgs = [{"role": "user", "content": "Describe the lighthouse keeper's morning."}]
    body = {"messages": msgs, "max_tokens": case["max_tokens"], "temperature": 0}
    c = case.get("concurrency", 1)
    res = [None] * c

    def work(i):
        res[i] = srv.post("/v1/chat/completions", body)
    t0 = time.time()
    ths = [threading.Thread(target=work, args=(i,)) for i in range(c)]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    toks = sum(d["usage"]["completion_tokens"] for d, _ in res)
    return {"concurrency": c, "wall_s": round(wall, 3), "tok_s": round(toks / wall, 3),
            "latency_s": [round(dt, 3) for _, dt in res]}


def tool_turn(srv, case):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    first = [{"role": "user", "content": "What is the weather in Oslo?"}]
    a, t1 = srv.post("/v1/chat/completions", {"messages": first, "tools": tools,
                                              "max_tokens": case["max_tokens"], "temperature": 0})
    turn = first + [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":12}"}]
    b, t2 = srv.post("/v1/chat/completions", {"messages": turn, "tools": tools,
                                              "max_tokens": case["max_tokens"], "temperature": 0})
    tb = b.get("runner_telemetry", {})
    return {"first_s": round(t1, 3), "second_s": round(t2, 3),
            "prompt_cached_tokens": tb.get("prompt_cached_tokens"),
            "prompt_forked_tokens": tb.get("prompt_forked_tokens"),
            "prompt_eval_tokens": tb.get("prompt_eval_tokens"),
            "prompt_reuse": tb.get("prompt_reuse")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default=os.path.join(ROOT, "runner"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=".", help="regex over case ids")
    ap.add_argument("--repeat", type=int, default=2, help="attempts per case (1 = first only)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--hash", action="store_true", help="sha256 the model file")
    ap.add_argument("--idle", action="store_true", help="also run the idle lifecycle")
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(ROOT, "scripts", "bench-corpus.json")))
    pick = re.compile(args.cases)
    cases = [c for c in manifest["cases"] if pick.search(c["id"]) and (c["kind"] != "idle" or args.idle)]
    caps = json.loads(subprocess.run([args.runner, "--caps"], capture_output=True, text=True).stdout)
    st = os.stat(args.model)
    rec = {"corpus_version": manifest["version"],
           "machine": {"platform": platform.platform(), "machine": platform.machine(),
                       "cpus": os.cpu_count()},
           "runner": {"version": caps.get("version"), "gpu": caps.get("gpu")},
           "model": {"path": os.path.abspath(args.model), "size": st.st_size, "mtime": int(st.st_mtime)},
           "threads": args.threads or "default", "ctx": args.ctx, "cases": []}
    if args.hash:
        h = hashlib.sha256()
        with open(args.model, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        rec["model"]["sha256"] = h.hexdigest()
    gpu = bool(caps.get("gpu"))
    with Server(args.runner, args.model, args.ctx, args.threads) as srv:
        prompts = {}
        n_layer = None
        offset = None
        for c in cases:
            if c["kind"] != "bench":
                continue
            key = min(c["prompt_tokens"], args.ctx - c["n_predict"] - 8)
            if offset is None:   # how the bench's count differs from the server's
                probe, served = prompt_of(srv, 16)
                offset = bench(args.runner, args.model, probe, {"n_predict": 1}, args.threads,
                               None).get("prompt_tokens", served) - served
            if key not in prompts:
                prompts[key] = prompt_of(srv, key, offset)
        for c in cases:
            for attempt in range(args.repeat):
                row = {"id": c["id"], "attempt": "first" if attempt == 0 else "repeat"}
                if c["kind"] == "bench":
                    gl = None
                    if "gpu_layers" in c:
                        if not gpu:
                            row["skipped"] = "no GPU in this build"
                            rec["cases"].append(row)
                            break
                        if n_layer is None:
                            n_layer = bench(args.runner, args.model, "hi", {"n_predict": 1},
                                            args.threads, None).get("layers")
                        gl = n_layer // 2 if c["gpu_layers"] == "half" else n_layer
                    text, _ = prompts[min(c["prompt_tokens"], args.ctx - c["n_predict"] - 8)]
                    row["prompt_tokens_target"] = c["prompt_tokens"]
                    row["result"] = bench(args.runner, args.model, text, c, args.threads, gl)
                elif c["kind"] == "server":
                    try:
                        if c["requests"] == "tool_turn":
                            row["result"] = tool_turn(srv, c)
                        else:
                            with Server(args.runner, args.model, args.ctx, args.threads,
                                        parallel=max(1, c.get("concurrency", 1))) as s2:
                                row["result"] = served_chat(s2, c)
                    except urllib.error.HTTPError as e:
                        row["result"] = {"error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"}
                elif c["kind"] == "idle":
                    port = free_port()
                    out = args.out + ".idle.json"
                    subprocess.run([sys.executable, os.path.join(ROOT, c["tool"]), "--port", str(port),
                                    "--cmd", f"{args.runner} --serve --port {port} -m {args.model} -c {args.ctx}",
                                    "--out", out], capture_output=True, timeout=1200)
                    row["result"] = json.load(open(out)) if os.path.exists(out) else {"error": "no report"}
                    rec["cases"].append(row)
                    break
                rec["cases"].append(row)
    json.dump(rec, open(args.out, "w"), indent=1)
    print(f"{len(rec['cases'])} results -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
