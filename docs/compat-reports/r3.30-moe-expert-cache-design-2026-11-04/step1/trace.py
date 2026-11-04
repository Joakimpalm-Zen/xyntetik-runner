"""R3.30 step 1: Mellum's MoE routing over the SS-000 prompts.
One server per prompt, each with its own RUNNER_MOE_TRACE file; the turn goes
through the model's own chat template (POST /v1/chat/completions), greedy,
max_tokens N. The user turns (user-turns.json, beside this script) are the
SS-000 frozen prompts' user content, cut from the *.qwen3.txt renders.
Usage: trace.py RUNNER MODEL OUTDIR [N] [-- RUNNER_PLACEMENT_ARGS...]
  default placement: --gpu off on the CPU; the run in run/ is
  `-- --gpu 99 --cpu-moe auto` on the RTX 3070, N 256
R330_PROMPTS=code,... picks the prompts (one a call in run/, each its own
foreground job); R330_TIMEOUT caps the request's seconds."""
import json
import os
import subprocess
import sys
import time
import urllib.request

RUNNER, MODEL, OUT = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "--" else 512
PLACE = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["--gpu", "off"]
os.makedirs(OUT, exist_ok=True)
turns = {os.path.basename(k): v for k, v in
         json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "user-turns.json"),
                        encoding="utf-8")).items()}
port = 18630
for name in os.environ.get("R330_PROMPTS", "code,copy,edit,long,prose,tool").split(","):
    trace = os.path.join(OUT, f"trace-{name}.jsonl")
    if os.path.exists(trace):
        os.remove(trace)
    env = dict(os.environ, RUNNER_MOE_TRACE=trace)
    if PLACE[:2] == ["--gpu", "off"]:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    log = open(os.path.join(OUT, f"server-{name}.log"), "wb")
    p = subprocess.Popen([RUNNER, "-m", MODEL, "--serve", "--port", str(port), *PLACE,
                          "-c", "8192", "--no-tray"], stdout=log, stderr=log, env=env)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:
                time.sleep(1)
        body = {"messages": [{"role": "user", "content": turns[name]}], "temperature": 0,
                "max_tokens": N, "stream": False}
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=int(os.environ.get("R330_TIMEOUT", "7200"))).read())
        rec = {"prompt": name, "usage": r.get("usage"), "finish": r["choices"][0].get("finish_reason"),
               "seconds": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        with open(os.path.join(OUT, "runs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    finally:
        p.terminate()
        p.wait(timeout=60)
        log.close()
