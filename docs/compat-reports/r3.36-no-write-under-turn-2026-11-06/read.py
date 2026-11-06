"""R3.36's done test on a GPU: a client must not extend the device turn. For the
speculative stream and for a session, the other slot's wait with a stalled and a
trickling client is compared with the same request from a client that reads.
It also checks a client that reads late gets the same text as one that does not.

Each client asserts its 200, so a request refused or never run cannot pass.
Usage: read.py BINARY MODEL  (exit 0 when every case matches its control)
"""
import json
import os
import socket
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

BIN, MODEL = sys.argv[1:3]
PORT = 18656
BASE = f"http://127.0.0.1:{PORT}"
N_SPEC, N_SESSION = 6000, 3000
REP = "one two three four five six seven eight nine ten " * 30
LONG = ("=" * 64 + "\n") * 40
tmp = tempfile.mkdtemp()
log = open(os.path.join(tmp, "server.log"), "wb")
srv = subprocess.Popen([BIN, "-m", MODEL, "--serve", "--port", str(PORT), "--parallel", "2", "--gpu", "99",
                        "--draft", "lookup", "--sessions", tmp, "-c", "16384", "--no-tray"],
                       stdout=log, stderr=log)


def post(path, body, timeout=240):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def probe():
    t0 = time.perf_counter()
    r = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0,
                                 "ignore_eos": True, "logprobs": 1, "cache_prompt": False}, timeout=240)
    return round(time.perf_counter() - t0, 3), round(r["runner_telemetry"]["timing"]["device_wait_seconds"], 3)


def spec_request():
    return "/v1/completions", {"prompt": REP, "max_tokens": N_SPEC, "temperature": 0, "seed": 1, "ignore_eos": True,
                               "stream": True, "cache_prompt": False}


def session_request():
    return "/v1/runner/sessions", {"prompt": LONG, "max_tokens": N_SESSION, "temperature": 0, "seed": 1,
                                   "ignore_eos": True}


def open_client(path, body, rcv):
    s = socket.socket()
    if rcv:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcv)
    s.connect(("127.0.0.1", PORT))
    b = json.dumps(body).encode()
    s.sendall(f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nContent-Type: application/json\r\n"
              f"Content-Length: {len(b)}\r\n\r\n".encode() + b)
    return s


def read_all(s, out, delay=0.0, chunk=65536, every=0.0):
    if delay:
        time.sleep(delay)
    s.settimeout(240)
    try:
        while True:
            d = s.recv(chunk)
            if not d:
                break
            out.append(d)
            if every:
                time.sleep(every)
    except OSError:
        pass


def gpu_sampler(stop, out):
    while not stop.is_set():
        p = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        try:
            out.append((time.perf_counter(), int(p.stdout.strip())))
        except ValueError:
            pass
        time.sleep(0.5)


def case(label, req, mode):
    path, body = req()
    stop = threading.Event()
    util = []
    g = threading.Thread(target=gpu_sampler, args=(stop, util))
    g.start()
    t0 = time.perf_counter()
    got = []
    if mode == "reading":
        s = open_client(path, body, None)
        th = threading.Thread(target=read_all, args=(s, got))
    elif mode == "stalled":
        s = open_client(path, body, 2048)
        th = None
    else:
        s = open_client(path, body, 2048)
        th = threading.Thread(target=read_all, args=(s, got, 0.0, 2048, 5.0))
    if th:
        th.start()
    time.sleep(1.0)
    lat, wait = probe()
    stop.set()
    g.join()
    busy = [u for t, u in util if t - t0 > 1.5 and t - t0 < 1.0 + wait * 0.9]
    mean = round(st.mean(busy), 1) if busy else None
    print(f"{label} {mode}: other slot's wait {wait} s, gpu during it {mean}%", flush=True)
    s.close()
    if th:
        th.join(timeout=5)
    time.sleep(2)
    return wait, mean


def sse_text(raw):
    text = []
    for line in raw.split(b"\n"):
        if line.startswith(b"data: {"):
            c = json.loads(line[6:]).get("choices") or []
            if c:
                text.append(c[0].get("text") or "")
    return "".join(text)


failed = 0
try:
    for _ in range(180):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    print("probe alone", [probe() for _ in range(3)], flush=True)
    for label, req in (("spec", spec_request), ("session", session_request)):
        control = [case(label, req, "reading") for _ in range(2)]
        ref = st.median(c[0] for c in control)
        for mode, n in (("stalled", 3), ("trickle", 2)):
            for _ in range(n):
                wait, gpu = case(label, req, mode)
                if wait > ref * 1.15 + 1.0:
                    failed += 1
                    print(f"  FAIL: {wait} s against the reading client's {ref} s", flush=True)
    # identity: the same text whether the client read at once or five seconds late
    path, body = spec_request()
    a = []
    s = open_client(path, body, None)
    read_all(s, a)
    s.close()
    b = []
    s = open_client(path, body, 2048)
    read_all(s, b, delay=5.0)
    s.close()
    ta, tb = sse_text(b"".join(a)), sse_text(b"".join(b))
    done_a, done_b = b"data: [DONE]" in b"".join(a), b"data: [DONE]" in b"".join(b)
    same = ta == tb and len(ta) > 1000
    print(f"identity: reading {len(ta)} chars [DONE] {done_a}, late {len(tb)} chars [DONE] {done_b}, same text {same}",
          flush=True)
    if not (same and done_a and done_b):
        failed += 1
finally:
    srv.kill()
    srv.wait(timeout=30)
    log.close()
print("PASS: no client extended the turn" if not failed else f"FAIL: {failed}")
sys.exit(1 if failed else 0)
