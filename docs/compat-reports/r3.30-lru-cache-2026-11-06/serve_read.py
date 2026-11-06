"""R3.30 step 3: decode speed in one long-lived process, so later requests find
their pages resident. N identical raw-prompt requests, fixed seed; each request's
decode rate comes from the stream's own token timestamps.
Usage: serve_read.py TAG HOT|auto N_REQ N_TOK [SIDE_ARGS...]
"""
import json, os, subprocess, sys, time, urllib.request
TAG, HOT, NREQ, NTOK = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SIDE = sys.argv[5:]
R = os.environ["RUNNER_BIN"]  # no default: a run must name its binary
M = os.environ["R330_MODEL"]
P = "Write four paragraphs about how reservoirs and water towers keep a city's water pressure steady through the day."
O = os.path.join(os.environ["R330_DIR"], "out")
port = 18640
env = dict(os.environ)
args = [R, "-m", M, "--serve", "--port", str(port), "--gpu", "99", "-c", "4096", "--no-tray"]
if HOT == "auto":
    args += ["--cpu-moe", "auto"]
else:
    args += ["--cpu-moe", "99"]
    env["RUNNER_MOE_HOT"] = HOT
    env["RUNNER_MOE_HOT_PROF"] = "1"
if os.environ.get("TRACE_TO"):
    env["RUNNER_MOE_TRACE"] = os.environ["TRACE_TO"]
log = open(os.path.join(O, TAG + ".server.log"), "wb")
p = subprocess.Popen(args, stdout=log, stderr=log, env=env)
stop = os.path.join(O, TAG + ".stop")
if os.path.exists(stop):
    os.remove(stop)
pg = subprocess.Popen([sys.executable, os.path.join(os.environ["R330_DIR"], "pagein.py"), os.path.join(O, TAG + ".pagein.json"), stop],
                      stderr=subprocess.DEVNULL)
rows = []
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    side = None
    # the guard: the load-time RAM line must match the binary this run claims
    want = os.environ.get("EXPECT_RAM")
    if want:
        import re
        t = open(os.path.join(O, TAG + ".server.log"), errors="replace").read()
        m = re.search(r"only ([\d.]+) GB of RAM", t)
        got = float(m.group(1)) if m else -1.0
        lo, hi = map(float, want.split("-"))
        if not lo <= got <= hi:
            raise SystemExit(f"{TAG}: GUARD FAILED, RAM line {got} not in {want}: wrong binary")
        print(TAG, "guard ok, RAM line", got, flush=True)
    if os.environ.get("PRIME"):
        t0 = time.perf_counter()
        nb = 0
        with open(M, "rb", buffering=0) as mf:
            for line in open(os.environ["PRIME"]):
                off, b = map(int, line.split())
                mf.seek(off)
                nb += len(mf.read(b))
        print(TAG, f"primed {nb/1e9:.2f} GB in {time.perf_counter()-t0:.1f}s", flush=True)
    for i in range(NREQ):
        body = {"prompt": P, "max_tokens": NTOK, "seed": 1, "stream": True, "cache_prompt": False}
        if os.environ.get("CHAT"):
            body = {"messages": [{"role": "user", "content": P}], "max_tokens": NTOK, "seed": 1, "stream": True}
        url = f"http://127.0.0.1:{port}/v1/" + ("chat/completions" if os.environ.get("CHAT") else "completions")
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        if SIDE and i >= int(os.environ.get("SIDE_FROM", "1")):
            side = subprocess.Popen([sys.executable, os.path.join(os.environ["R330_DIR"], "upload_bench.py"), M, *SIDE],
                                    stdout=subprocess.PIPE, text=True)
            time.sleep(3)  # the side process's CUDA context comes up first
        ts = []
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    ts.append(time.perf_counter())
        n = len(ts)
        rate = (n - 1) / (ts[-1] - ts[0]) if n > 1 else 0
        half = n // 2
        late = (n - 1 - half) / (ts[-1] - ts[half]) if n > 2 else 0
        rows.append((n, round(rate, 2), round(late, 2)))
        print(TAG, "req", i, "chunks", n, "tok/s", round(rate, 2), "second half", round(late, 2), flush=True)
        if side:
            print(TAG, side.communicate(timeout=60)[0].strip(), flush=True)
            side = None
finally:
    open(stop, "w").close()
    p.terminate()
    try:
        p.wait(timeout=30)
    except Exception:
        p.kill()
    log.close()
    pg.wait(timeout=20)
print(TAG, "pagein", json.load(open(os.path.join(O, TAG + ".pagein.json"))))
