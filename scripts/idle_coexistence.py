#!/usr/bin/env python3
"""Idle coexistence measurement: what does a resident inference server cost
the machine while it does nothing?

Runs ONE OpenAI-compatible server through a four-state lifecycle and reports
what it holds and burns at each state:

  S1  loaded, idle 60 s      (CPU seconds burned, RSS, threads)
  S2  after one request, idle 60 s (same, plus TTFT of the request)
  S3  after unload, idle 60 s (only if the server exposes POST /unload)

On macOS it also samples system-wide wired memory (vm_stat) around the
lifecycle, which attributes unevictable memory to the server on an otherwise
quiet machine. Run it once per engine, sequentially, never two engines at
once, and compare the JSON reports.

Examples:
  python3 scripts/idle_coexistence.py --port 8092 \
      --cmd "./runner --serve --port 8092 -m model.gguf -c 4096"
  python3 scripts/idle_coexistence.py --port 8091 \
      --cmd "./llama-server -m model.gguf -c 4096 --port 8091"

Methodology notes: use a quiet machine and state your free RAM; use each
engine's stock defaults plus a pinned context size; use prebuilt release
binaries and record versions. Engines make different, legitimate design
choices here. Report both sides: what an engine holds while idle AND what
latency it delivers when asked. This script measures cost of residence, not
quality of inference.
"""
import argparse, json, re, shlex, signal, subprocess, sys, time, urllib.request

IDLE_S = 60

def _parse_cputime(t: str) -> float:
    """ps cputime is platform-shaped: macOS emits mm:ss.cc, Linux emits
    [dd-]hh:mm:ss with no centiseconds. Parsing one shape with the other's
    rule silently misreads 62 s as 1.02 s, which is fatal in a script whose
    whole point is CPU seconds, so both shapes are handled explicitly."""
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    frac = 0.0
    if "." in t:
        t, cc = t.rsplit(".", 1)
        frac = int(cc) / 100
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s + frac


def ps_stat(pid):
    o = subprocess.run(["ps", "-o", "rss=,cputime=,pcpu=", "-p", str(pid)],
                       capture_output=True, text=True).stdout.split()
    if not o:
        return None
    return {"rss_mb": round(int(o[0]) / 1024, 1),
            "cputime_s": _parse_cputime(o[1])}

def wired_mb():
    try:
        o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        pg = int(re.search(r"page size of (\d+)", o).group(1))
        w = int(re.search(r"Pages wired down:\s+(\d+)", o).group(1))
        return round(w * pg / 2**20, 1)
    except Exception:
        return None

def wait_ready(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False

def frame_carries_output(line):
    """True only when this SSE frame carries model OUTPUT.

    The first `data:` frame of an OpenAI-compatible chat stream is the
    role-only delta, and Runner sends it immediately after the response
    headers, deliberately before prefill starts. Stamping TTFT on it measured
    connection setup: near zero whatever the prompt cost, on every model. That
    is not the number this script exists to publish, and other engines emit
    their role frame at different points in their own pipelines, so the
    cross-engine comparison was not like for like either.
    """
    if not line.startswith(b"data:") or b"[DONE]" in line:
        return False
    try:
        payload = json.loads(line[len(b"data:"):].strip())
    except (ValueError, UnicodeDecodeError):
        return False
    for choice in payload.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            return True
        if choice.get("text"):          # the /v1/completions spelling
            return True
    return False


def chat_ttft(port, prompt):
    mid = json.load(urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
    body = json.dumps({"model": mid, "stream": True, "max_tokens": 64,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None
    with urllib.request.urlopen(req, timeout=180) as resp:
        for line in resp:
            if ttft is None and frame_carries_output(line):
                ttft = time.time() - t0
    return {"ttft_s": round(ttft, 3) if ttft is not None else None,
            "total_s": round(time.time() - t0, 3)}

def idle_window(pid, label):
    a = ps_stat(pid); time.sleep(IDLE_S); b = ps_stat(pid)
    if not a or not b:
        return {"state": label, "error": "process exited"}
    return {"state": label, "idle_window_s": IDLE_S,
            "cpu_s_during_idle": round(b["cputime_s"] - a["cputime_s"], 2),
            "rss_mb_end": b["rss_mb"], "wired_mb_system": wired_mb()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, help="full server command line")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", default="idle_coexistence_report.json")
    args = ap.parse_args()

    report = {"cmd": args.cmd, "idle_window_s": IDLE_S,
              "wired_mb_baseline": wired_mb(), "states": []}
    # The server's own log must not land on OUR stdout: the report is printed
    # there, and an interleaved log makes it unparseable for anything reading
    # `... > report.json`. Kept in a file beside the report rather than
    # discarded, because a server that misbehaves is the thing being measured.
    log_path = args.out + ".server.log"
    server_log = open(log_path, "wb")
    report["server_log"] = log_path
    p = subprocess.Popen(shlex.split(args.cmd), stdout=server_log,
                         stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        if not wait_ready(args.port):
            report["error"] = "server never became ready"
            return
        report["startup_to_ready_s"] = round(time.time() - t0, 1)
        time.sleep(5)
        report["states"].append(idle_window(p.pid, "S1_loaded_idle"))
        report["first_request"] = chat_ttft(args.port, "List three prime numbers.")
        report["states"].append(idle_window(p.pid, "S2_post_inference_idle"))
        report["request_after_idle"] = chat_ttft(args.port, "Name two colors.")
        unload = {"supported": False}
        for path in ("/unload", "/v1/unload"):
            try:
                r = urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{args.port}{path}", data=b"", method="POST"),
                    timeout=10)
                unload = {"supported": True, "path": path, "http": r.status}
                break
            except Exception:
                pass
        report["unload"] = unload
        if unload.get("supported"):
            time.sleep(10)
            report["states"].append(idle_window(p.pid, "S3_after_unload"))
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
        time.sleep(5)
        server_log.close()
        report["wired_mb_after_kill"] = wired_mb()
        json.dump(report, open(args.out, "w"), indent=1)
        print(json.dumps(report, indent=1))
        print(f"\nreport -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
