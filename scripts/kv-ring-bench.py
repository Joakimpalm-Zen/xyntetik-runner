"""kv-ring-bench (R5.5): the sliding-window KV ring against the flat cache on one
box, one session, interleaved A/B rounds after a page-cache warm-up.

Per context (4k/8k/16k/32k) and arm (--kv-ring off / on):
  - cache bytes and the device split, from -v;
  - prompt and decode tok/s with the context filled to ctx - gen - margin;
and, per arm, a server's prefix-hit prefill for a repeated system prompt:
  a shared ~sys_tokens system prompt, request 1 with user turn A (cold),
  request 2 with user turn B (the prefix cache's case), max_tokens 1, wall
  time and the telemetry's cached / evaluated token counts.

Writes one JSON object per measurement to --out (JSON lines).
"""
import argparse, json, os, random, re, socket, subprocess, sys, time, urllib.request

WORDS = ("river city market winter engine garden letter mountain silver window "
         "harbor story candle forest railway kitchen painter lantern meadow bridge "
         "teacher thunder village orchard compass ledger falcon cellar quarry beacon "
         "anchor violin granite pepper saddle timber velvet harvest summit glacier").split()


def text_of(n_words, seed):
    r = random.Random(seed)
    out = []
    for i in range(n_words):
        out.append(r.choice(WORDS))
        if i % 12 == 11:
            out.append(".")
    return " ".join(out)


def run(cmd, env=None, timeout=3600):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env, timeout=timeout)
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


def env_clean():
    e = dict(os.environ)
    e.pop("RUNNER_KV_RING", None)
    return e


def tokens_per_word(a):
    words = 2000
    with open(a.tmp + "/cal.txt", "w") as f:
        f.write(text_of(words, 1))
    rc, out, err = run([a.runner, "-m", a.model, "-f", a.tmp + "/cal.txt", "-n", "1",
                        "--temp", "0", "-c", "8192"], env_clean())
    m = re.search(r"prompt: (\d+) tok", err)
    if rc or not m:
        sys.exit("calibration failed:\n" + err[-2000:])
    return int(m.group(1)) / words


def one_run(a, ctx, arm, tpw, rnd):
    n_words = int((ctx - a.gen - 64) / tpw * 0.97)
    pf = f"{a.tmp}/p{ctx}.txt"
    if not os.path.exists(pf):
        with open(pf, "w") as f:
            f.write(text_of(n_words, ctx))
    cmd = [a.runner, "-m", a.model, "-f", pf, "-n", str(a.gen), "--temp", "0",
           "-c", str(ctx), "-v", "--kv-ring", arm, *a.extra]
    t0 = time.time()
    rc, out, err = run(cmd, env_clean())
    wall = time.time() - t0
    rec = {"kind": "context", "ctx": ctx, "arm": arm, "round": rnd, "rc": rc, "wall_s": round(wall, 2)}
    m = re.search(r"prompt: (\d+) tok, ([0-9.]+) tok/s \| gen: (\d+) tok, ([0-9.]+) tok/s", err)
    if m:
        rec.update(prompt_tok=int(m.group(1)), prompt_tok_s=float(m.group(2)),
                   gen_tok=int(m.group(3)), gen_tok_s=float(m.group(4)))
    for key, pat in (("kv_cache_mb", r"^kv cache\s+([0-9.]+) MB"),):
        mm = re.search(pat, err, re.M)
        if mm:
            rec[key] = float(mm.group(1))
    for pat in (r"^kv ring\s+(.*)$", r"^(gpu: .*layers.*)$", r"^(split .*)$"):
        mm = re.search(pat, err, re.M)
        if mm:
            rec.setdefault("lines", []).append(mm.group(1).strip())
    out_hash = __import__("hashlib").sha256(out.encode()).hexdigest()[:16]
    rec["output_sha"] = out_hash
    if rc:
        rec["err_tail"] = err[-600:]
    return rec


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=json.dumps(body, allow_nan=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return time.time() - t0, d


def prefix_run(a, arm, tpw, rnd):
    sys_words = int(a.sys_tokens / tpw)
    system = "System: " + text_of(sys_words, 7) + "\n"
    users = ["User: " + text_of(int(40 / tpw), 11) + "\nAssistant:",
             "User: " + text_of(int(40 / tpw), 12) + "\nAssistant:"]
    port = free_port()
    cmd = [a.runner, "-m", a.model, "--serve", "--no-tray", "--port", str(port),
           "-c", str(a.server_ctx), "--kv-ring", arm, *a.extra]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env_clean())
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        recs = []
        for i, u in enumerate(users):
            wall, d = post(port, {"prompt": system + u, "max_tokens": 1, "temperature": 0})
            t = d["runner_telemetry"]
            recs.append({"kind": "prefix", "arm": arm, "round": rnd, "request": i + 1,
                         "wall_s": round(wall, 3), "cached": t["prompt_cached_tokens"],
                         "evaluated": t["prompt_eval_tokens"],
                         "prompt_tokens": d["usage"]["prompt_tokens"]})
        return recs
    finally:
        srv.terminate()
        srv.wait(timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="4096,8192,16384,32768")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--sys-tokens", type=int, default=3000)
    ap.add_argument("--server-ctx", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=".")
    ap.add_argument("extra", nargs="*", help="extra runner flags after --")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    # warm the page cache: the file's bytes once before any timed run
    with open(a.model, "rb") as f:
        while f.read(1 << 24):
            pass
    tpw = tokens_per_word(a)
    ctxs = [int(c) for c in a.ctx.split(",")]
    with open(a.out, "a") as o:
        o.write(json.dumps({"kind": "meta", "model": os.path.basename(a.model),
                            "tokens_per_word": round(tpw, 4), "extra": a.extra,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, allow_nan=False) + "\n")
        for rnd in range(a.rounds):
            arms = ["off", "on"] if rnd % 2 == 0 else ["on", "off"]   # interleave A/B
            for ctx in ctxs:
                for arm in arms:
                    rec = one_run(a, ctx, arm, tpw, rnd)
                    print(json.dumps(rec, allow_nan=False), flush=True)
                    o.write(json.dumps(rec, allow_nan=False) + "\n"); o.flush()
            for arm in arms:
                for rec in prefix_run(a, arm, tpw, rnd):
                    print(json.dumps(rec, allow_nan=False), flush=True)
                    o.write(json.dumps(rec, allow_nan=False) + "\n"); o.flush()


if __name__ == "__main__":
    main()
