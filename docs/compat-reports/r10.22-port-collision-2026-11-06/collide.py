"""R10.22: the collision on purpose, on one tree. Two servers on one port; which
one takes the traffic; what a client sees when that one goes. Reproduces CI's reset
on the first send: connect, the owner dies, then send the headers.
Usage: collide.py TREE"""
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

TREE = pathlib.Path(sys.argv[1])
EXE = str(TREE / "runner.exe")
tmp = pathlib.Path(tempfile.mkdtemp(prefix="collide-"))
model = tmp / "m.gguf"
subprocess.run([sys.executable, str(TREE / "scripts" / "make-test-model.py"), str(model)], check=True, cwd=TREE,
               stdout=subprocess.PIPE)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(tag, port):
    log = open(tmp / f"{tag}.log", "wb")
    p = subprocess.Popen([EXE, "-m", str(model), "--serve", "--no-tray", "--port", str(port), "--parallel", "1",
                          "-c", "256", "--gpu", "off"], stdout=log, stderr=subprocess.STDOUT)
    return p, log


def logtext(tag):
    return (tmp / f"{tag}.log").read_text(errors="replace")


def health(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status
    except Exception as e:
        return repr(e)[:60]


def complete(port):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                               data=json.dumps({"prompt": "hello", "max_tokens": 1}).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=20) as f:
        return f.status


port = free_port()
a, la = start("A", port)
for _ in range(100):
    if health(port) == 200:
        break
    time.sleep(0.05)
b, lb = start("B", port)
time.sleep(1.5)
print(TREE.name, "port", port)
print(" A alive", a.poll() is None, "| B alive", b.poll() is None, "| B exit", b.poll())
print(" B says listening:", "server listening" in logtext("B"), "| B bind error:",
      [l for l in logtext("B").splitlines() if "bind" in l.lower() or "address" in l.lower()][:1])
for _ in range(4):
    print(" request", complete(port))
ca, cb = logtext("A").count("gen tok"), logtext("B").count("gen tok")
print(f" requests logged: A {ca}, B {cb}")
owner, other, name = (a, b, "A") if ca >= cb else (b, a, "B")
c = socket.create_connection(("127.0.0.1", port), timeout=5)
owner.kill()
owner.wait()
time.sleep(0.2)
try:
    c.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\n\r\n")
    print(f" owner {name} killed; first send: went through; read:", c.recv(64)[:30])
except OSError as e:
    print(f" owner {name} killed; first send: {type(e).__name__} {e.errno if hasattr(e, 'errno') else ''} {getattr(e, 'winerror', '')}")
c.close()
print(" survivor still alive:", other.poll() is None, "| health now:", health(port))
for p in (a, b):
    if p.poll() is None:
        p.kill()
        p.wait()
la.close()
lb.close()
