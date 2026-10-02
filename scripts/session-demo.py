#!/usr/bin/env python3
"""The session-image demo: suspend a served generation, prove the resume is
exact, fork it eight ways, keep one.

    scripts/session-demo.py [--model M.gguf] [--runner ./runner]

Starts a server with --sessions in a fresh temporary directory and talks to
it over HTTP. Steps:

  1. the generation run straight through, as the reference;
  2. the same generation suspended after 40 tokens: an image and its id;
  3. the image resumed: the text equals the straight run's, byte for byte;
  4. eight forks of the same image, each under its own seed and each
     suspended again 40 tokens later: eight different continuations of one
     state, none of which re-read the prompt;
  5. one kept (the one whose text so far is longest, as a stand-in for any
     judge), the other seven deleted, and the kept one run to its end.

Exit 0 only when step 3 is exact and the forks are reproducible.
"""
import argparse
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = ("Three travellers reach a fork in a mountain road at dusk. The first "
          "says")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(base, path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", default=str(ROOT / "runner"))
    ap.add_argument("--model", help="a local GGUF (default: fetch SmolLM2-135M "
                                    "Q8_0 with -hf)")
    a = ap.parse_args()
    sess = pathlib.Path(tempfile.mkdtemp(prefix="runner-sessions-"))
    port = free_port()
    model = (["-m", a.model] if a.model
             else ["-hf", "bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0"])
    proc = subprocess.Popen([a.runner, *model, "--serve", "--no-tray",
                             "--port", str(port), "-c", "1024",
                             "--sessions", str(sess)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).read()
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit("the server exited during startup")
                time.sleep(0.5)
        start = {"prompt": PROMPT, "max_tokens": 160, "temperature": 0.9,
                 "seed": 7, "ignore_eos": True}
        t0 = time.time()
        straight = call(base, "/v1/runner/sessions", start)
        print(f"1. straight run: {straight['generated']} tokens")

        a1 = call(base, "/v1/runner/sessions", dict(start, suspend_after=40))
        print(f"2. suspended after {a1['generated']} tokens: image {a1['id'][:16]}...")

        b = call(base, f"/v1/runner/sessions/{a1['id']}/resume", {})
        exact = a1["text"] + b["text"] == straight["text"]
        print(f"3. resumed: {'IDENTICAL to' if exact else 'DIFFERENT from'} the "
              f"straight run ({len(straight['text'])} bytes)")

        forks = []
        for seed in range(1, 9):
            f = call(base, f"/v1/runner/sessions/{a1['id']}/resume",
                     {"fork_seed": seed, "suspend_after": 80})
            forks.append((seed, f))
        again = call(base, f"/v1/runner/sessions/{a1['id']}/resume",
                     {"fork_seed": 3, "suspend_after": 80})
        reproducible = again["text"] == forks[2][1]["text"]
        distinct = len({f["text"] for _, f in forks})
        print(f"4. eight forks: {distinct} distinct continuations; fork 3 asked "
              f"again: {'the same text' if reproducible else 'DIFFERENT text'}")
        for seed, f in forks:
            print(f"   seed {seed}: {f['text'][:70]!r}")

        keep_seed, keep = max(forks, key=lambda sf: len(sf[1]["text"]))
        for seed, f in forks:
            if seed != keep_seed:
                call(base, f"/v1/runner/sessions/{f['id']}", method="DELETE")
        done = call(base, f"/v1/runner/sessions/{keep['id']}/resume", {})
        print(f"5. kept fork {keep_seed}, deleted seven, ran it to its end: "
              f"{done['generated']} tokens")
        print(f"\ndone in {time.time() - t0:.1f} s; images in {sess}")
        return 0 if exact and reproducible else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
