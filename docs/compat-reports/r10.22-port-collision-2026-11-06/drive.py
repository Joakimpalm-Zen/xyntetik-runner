"""R10.22: the failing test's own _serve, in a loop, keeping what CI lost.

Runs tests/test_kv_ring.py's HOLE case exactly as the test does, flat then ring, in
one tree. Each server keeps its stderr. Each request is numbered, so a failure says
which one, whether the server was alive, and what it logged. Every server's port
and lifetime go to a shared file, so a collision can be shown or ruled out.

Usage: drive.py TREE REPS OUTDIR WORKER
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

TREE, REPS, OUT, WORKER = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
OUT.mkdir(parents=True, exist_ok=True)
name = TREE.name
sys.path[:0] = [str(TREE / "tests" / "conformance"), str(TREE / "scripts"), str(TREE / "tests")]
sp = importlib.util.spec_from_file_location("kv", TREE / "tests" / "test_kv_ring.py")
kv = importlib.util.module_from_spec(sp)
sp.loader.exec_module(kv)
EXE = str(TREE / "runner.exe")
PORTS = OUT / "ports.jsonl"


def note(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


tmp = pathlib.Path(tempfile.mkdtemp(prefix="r1022-"))
model = tmp / "swa.gguf"
subprocess.run([sys.executable, str(TREE / "scripts" / "make-test-model.py"), "--swa", f"{kv.SWA_WINDOW},2", str(model)],
               check=True, cwd=TREE, stdout=subprocess.PIPE)

Base = kv.RunnerServer
state = {"srv": None, "n": 0, "serial": 0}


class Kept(Base):
    def __init__(self, *a, **k):
        state["serial"] += 1
        k["log_path"] = str(tmp / f"srv-{state['serial']}.log")
        super().__init__(*a, **k)

    def start(self):
        super().start()
        state["srv"] = self
        state["n"] = 0
        self.t_up = time.time()
        note(PORTS, {"ev": "up", "tree": name, "w": WORKER, "port": self.port, "pid": self.proc.pid, "t": self.t_up})

    def stop(self, strict=False):
        self.alive_at_stop = self.proc.poll() is None if self.proc else None
        self.rc_before_stop = self.proc.poll() if self.proc else None
        port, pid = self.port, self.proc.pid if self.proc else None
        super().stop(strict)
        note(PORTS, {"ev": "down", "tree": name, "w": WORKER, "port": port, "pid": pid, "t": time.time(),
                     "exit": self.exit_code})


kv.RunnerServer = Kept
real = urllib.request.urlopen


def counted(*a, **k):
    state["n"] += 1
    return real(*a, **k)


urllib.request.urlopen = counted
ok = bad = 0
for rep in range(REPS):
    t0 = time.time()
    phase = "flat"
    try:
        flat = kv._serve(EXE, model, False, "off", kv.HOLE, {2})
        phase = "ring"
        ring = kv._serve(EXE, model, True, "off", kv.HOLE, {2})
        same = all(a["tokens"] == b["tokens"] and a["lp"] == b["lp"] for a, b in zip(flat, ring))
        hows = [(r["how"], r["cached"]) for r in ring]
        assert same and hows[1] == ("kv", 600) and hows[2] == ("ring_reset", 0), hows
        ok += 1
    except Exception as ex:
        bad += 1
        s = state["srv"]
        log = pathlib.Path(s.log_path).read_text(errors="replace") if s and s.log_path else ""
        rec = {"tree": name, "w": WORKER, "rep": rep, "phase": phase, "request": state["n"], "port": s.port if s else None,
               "pid": s.proc.pid if s and s.proc else None, "exc": repr(ex)[:200], "t": time.time(),
               "alive_at_stop": getattr(s, "alive_at_stop", None), "rc_before_stop": getattr(s, "rc_before_stop", None),
               "exit_code": getattr(s, "exit_code", None), "stack": traceback.format_exc()[-700:], "log_tail": log[-1800:]}
        note(OUT / "failures.jsonl", rec)
    for f in tmp.glob("srv-*.log"):
        try:
            f.unlink()
        except OSError:
            pass
print(json.dumps({"tree": name, "w": WORKER, "ok": ok, "bad": bad}))
