"""R3.24: Metal decode microbatch, GPU ms per forward against its width (slots decoding together)."""
import json, os, re, statistics, subprocess, sys, threading, time, urllib.request
RUNNER = "./runner"
D = "models"
S = "."
def run(model, conc, port=18744, ntok=96):
    log = f"{S}/r324-micro-{model}-{conc}.log"
    env = dict(os.environ, RUNNER_METAL_TIMING="2")
    srv = subprocess.Popen([RUNNER, "-m", f"{D}/{model}.gguf", "--serve", "--port", str(port), "--parallel", "8",
                            "-c", "4096", "--tray", "off"], stderr=open(log, "w"), stdout=subprocess.DEVNULL, env=env)
    try:
        for _ in range(120):
            try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except Exception: time.sleep(0.5)
        def ask(i):
            body = json.dumps({"prompt": f"Count slowly from {i} upward, one number per line:", "max_tokens": ntok,
                               "temperature": 0, "ignore_eos": True}).encode()
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                   {"Content-Type": "application/json"}), timeout=600).read()
        t0 = time.time()
        th = [threading.Thread(target=ask, args=(i,)) for i in range(conc)]
        [t.start() for t in th]; [t.join() for t in th]
        wall = time.time() - t0
    finally:
        srv.terminate(); srv.wait(timeout=30)
    t = open(log).read()
    stages = ("qkv", "attn", "o", "gate_up", "down", "head")
    byn = {}
    for line in re.findall(r"metal-microbatch n=(\d+) (.*)", t):
        n = int(line[0]); kv = dict(x.split("=") for x in line[1].split())
        byn.setdefault(n, []).append({k: float(v.rstrip("ms")) for k, v in kv.items()})
    for n, rows in sorted(byn.items()):
        if len(rows) < 8: continue
        med = {k: statistics.median(r[k] for r in rows[3:]) for k in ("gpu",) + stages}
        print(f"{model:24s} conc {conc} N={n} steps={len(rows):3d} gpu/step {med['gpu']:7.2f} ms  per column {med['gpu']/n:6.2f}  " +
              "  ".join(f"{k} {med[k]:6.2f}" for k in stages), flush=True)
for model in sys.argv[1:]:
    for conc in (1, 4, 8):
        run(model, conc)
