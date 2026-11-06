"""R3.37: a context released while its request waits for the device turn.

Needs a CUDA device: the device turn exists only on a GPU-backed server
(scheduler.c, device_turn), so on a CPU build this path holds nothing and the
leak cannot be observed. There is no GPU-free form of this test.

A long generation holds the turn; a request naming a pinned context passes the
prefix check and queues for the turn; the context is deleted; the request then
takes the context_unavailable branch. Afterwards every slot must still serve,
including a prompt long enough to yield the turn between prefill chunks.

Usage: repro.py BINARY MODEL  (exit 0 when every completion is served)
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

BIN, MODEL = sys.argv[1:3]
PORT = 18655
BASE = f"http://127.0.0.1:{PORT}"
CTX = ("The quick brown fox jumps over the lazy dog. " * 8).rstrip()
REP = "one two three four five six seven eight nine ten " * 30
log = tempfile.TemporaryFile()
srv = subprocess.Popen([BIN, "-m", MODEL, "--serve", "--port", str(PORT), "--parallel", "3", "--gpu", "99",
                        "--draft", "lookup", "-c", "8192", "--no-tray"], stdout=log, stderr=log)


def call(path, body=None, method="POST", timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def plain(prompt="The capital of France is", n=8):
    return {"prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 1, "ignore_eos": True, "logprobs": 1,
            "cache_prompt": False}


failed = 0
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    for rnd in range(3):
        assert call("/v1/runner/contexts", {"id": "c1", "prompt": CTX})[0] == 200
        spec = {"prompt": REP, "max_tokens": 2500, "temperature": 0, "ignore_eos": True, "cache_prompt": False}
        th = threading.Thread(target=call, args=("/v1/completions", spec))
        th.start()
        time.sleep(1.0)
        res = {}

        def ctx_req():
            res["r"] = call("/v1/completions", {"prompt": CTX + " And then", "max_tokens": 4, "temperature": 0,
                                                "context_id": "c1"})

        tc = threading.Thread(target=ctx_req)
        tc.start()
        time.sleep(0.7)
        assert call("/v1/runner/contexts/c1", None, "DELETE")[0] == 200
        tc.join()
        th.join()
        code, body = res["r"]
        print(f"round {rnd}: context request {code}, context_unavailable {b'context_unavailable' in body}", flush=True)
        assert code == 409 and b"context_unavailable" in body, "did not reach the branch: this round proves nothing"
        for label, b in (("short", plain()), ("short", plain()), ("short", plain()),
                         ("long prompt, yields between chunks", plain("word " * 1500))):
            try:
                t0 = time.perf_counter()
                code, r = call("/v1/completions", b, timeout=30)
                print(f"  {label}: {code} in {time.perf_counter() - t0:.2f} s", flush=True)
                assert code == 200
            except Exception as ex:
                failed += 1
                print(f"  {label}: NOT SERVED in 30 s ({ex})", flush=True)
        if failed:
            print("health:", call("/health", None, "GET", timeout=10)[1][:80], flush=True)
            break
finally:
    srv.kill()
    srv.wait(timeout=30)
print("FAIL: a completion was not served after the 409" if failed else "PASS: every completion served after the 409")
sys.exit(1 if failed else 0)
