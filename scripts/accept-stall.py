#!/usr/bin/env python3
"""Does a client that stops reading a large GET park the accept thread?

The accept loop answers a few routes itself so /health survives a busy slot.
Until 1.0.2 those included GET /v1/responses/{id}, /metrics and the context
listing, whose bodies can exceed the loopback socket buffers; a client that
opened one and never read left the accept thread blocked in send() and the
server admitted nothing more (found 2026-10-05).

This stores a Responses body larger than the loopback buffers (a long input
is echoed back in the stored object), GETs it on a socket with a 1 KiB
receive window and never reads, then times /health over fresh connections.
/health is the accept thread's own answer, so its latency is that thread's
liveness. A real model is needed for the store (the toy fixture's context
caps the body below the buffers); the default model here is the small one
every gate uses. Exit 0 when /health stays under --health-budget seconds
with the dead reader attached, 1 otherwise.
"""
import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthy(port, deadline):
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except Exception:
            time.sleep(0.25)
    raise SystemExit("server never became healthy")


def health_latency(port):
    t0 = time.monotonic()
    s = socket.socket()
    s.settimeout(60.0)
    s.connect(("127.0.0.1", port))
    s.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
    data = b""
    try:
        while b"\r\n\r\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        return None
    finally:
        s.close()
    return time.monotonic() - t0 if data.startswith(b"HTTP/1.1 200") else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--model", default=str(ROOT / "models/SmolLM2-135M-Instruct-Q8_0.gguf"))
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--input-chars", type=int, default=600_000,
                    help="stored input size; must exceed the loopback buffers "
                         "(macOS 128 KiB each side, Linux up to a few MiB)")
    ap.add_argument("--health-budget", type=float, default=5.0)
    ap.add_argument("--hold", type=float, default=8.0,
                    help="seconds to hold the dead reader while probing")
    args = ap.parse_args(argv)
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="runner-accept-stall-") as tmp:
        log = open(Path(tmp) / "runner.log", "wb")
        proc = subprocess.Popen([args.runner, "-m", args.model, "--serve", "--no-tray",
                                 "--port", str(port), "--parallel", "1",
                                 "-c", str(args.ctx), "--gpu", "off", "-t", "2"],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_healthy(port, time.monotonic() + 120)
            text = ("lorem ipsum " * (args.input_chars // 12))[:args.input_chars]
            body = json.dumps({"input": text, "max_output_tokens": 1,
                               "store": True, "temperature": 0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/responses",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                rid = json.load(r)["id"]
            # the stored object's size, read properly once
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/responses/{rid}",
                                        timeout=60) as r:
                stored_n = len(r.read())
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/responses/{rid}/input_items",
                                        timeout=60) as r:
                items_n = len(r.read())
            print(f"stored response {stored_n} bytes, input_items {items_n} bytes")
            dead = []
            for path in (f"/v1/responses/{rid}/input_items", f"/v1/responses/{rid}",
                         "/metrics"):
                s = socket.socket()
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
                s.connect(("127.0.0.1", port))
                s.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
                dead.append(s)
            time.sleep(1.0)
            worst = 0.0
            ok = True
            end = time.monotonic() + args.hold
            while time.monotonic() < end:
                lat = health_latency(port)
                if lat is None:
                    print("/health: no answer within 60 s")
                    ok = False
                    break
                worst = max(worst, lat)
                time.sleep(0.5)
            print(f"/health worst latency with dead readers attached: {worst:.3f} s "
                  f"(budget {args.health_budget} s)")
            for s in dead:
                s.close()
            ok = ok and worst <= args.health_budget
            print("PASS" if ok else "FAIL")
            return 0 if ok else 1
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()


if __name__ == "__main__":
    sys.exit(main())
