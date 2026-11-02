"""R3.24 (b): Metal decode microbatch throughput, identity matvec vs opt-in GEMM tiles, interleaved A/B rounds."""
import json, os, subprocess, sys, threading, time, urllib.request
# usage: tput.py RUNNER MODEL_DIR LOG_DIR MODEL[,MODEL] CONC[,CONC] ROUNDS
RUNNER, D, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
models = sys.argv[4].split(","); concs = [int(c) for c in sys.argv[5].split(",")]; rounds = int(sys.argv[6])
def serve(model, tiles, port):
    env = dict(os.environ)
    env.pop("RUNNER_METAL_BATCH_TILES", None)
    if tiles: env["RUNNER_METAL_BATCH_TILES"] = "1"
    log = open(f"{OUT}/srv-{model}-{'tiles' if tiles else 'ident'}.log", "a")
    p = subprocess.Popen([RUNNER, "-m", f"{D}/{model}.gguf", "--serve", "--port", str(port), "--parallel", "8",
                          "-c", "4096", "--tray", "off"], stderr=log, stdout=subprocess.DEVNULL, env=env)
    for _ in range(240):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); return p
        except Exception: time.sleep(0.5)
    p.terminate(); raise SystemExit("server did not come up")
def burst(port, conc, ntok=96):
    texts = [None] * conc
    def ask(i):
        body = json.dumps({"prompt": f"Count slowly from {i} upward, one number per line:", "max_tokens": ntok,
                           "temperature": 0, "ignore_eos": True}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                       {"Content-Type": "application/json"}), timeout=900).read())
        texts[i] = r["choices"][0]["text"]
    t0 = time.time()
    th = [threading.Thread(target=ask, args=(i,)) for i in range(conc)]
    [t.start() for t in th]; [t.join() for t in th]
    return conc * ntok / (time.time() - t0), texts
for model in models:
    for r in range(rounds):
        for tiles in (False, True):
            p = serve(model, tiles, 18744)
            try:
                burst(18744, 2, 16)  # warm
                for conc in concs:
                    tps, texts = burst(18744, conc)
                    print(json.dumps({"model": model, "round": r, "route": "tiles" if tiles else "identity",
                                      "conc": conc, "tok_s": round(tps, 2), "texts": texts}), flush=True)
            finally:
                p.terminate(); p.wait(timeout=30)
