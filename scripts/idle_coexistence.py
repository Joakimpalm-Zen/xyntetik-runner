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
quiet machine; elsewhere, the GPU memory in use (nvidia-smi) when there is
one. CPU time and memory come from the platform's own accounting: `ps` on
macOS, /proc on Linux (10 ms ticks, where `ps` reports whole seconds), and
GetProcessTimes/GetProcessMemoryInfo on Windows (where `ps` cannot see a
native process). Run it once per engine, sequentially, never two engines at
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
import argparse, json, os, re, shlex, signal, subprocess, sys, time, urllib.request

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


def _descendants(root_pid):
    """The root plus every transitive child. `ollama serve` and vLLM fork the
    process that actually holds the weights, so measuring only the direct
    child reported near-zero idle RSS/CPU for exactly the engines with the
    largest footprints. One `ps -Ao pid=,ppid=` snapshot, walked to a fixed
    point; portable across macOS and Linux, no dependencies."""
    o = subprocess.run(["ps", "-Ao", "pid=,ppid="],
                       capture_output=True, text=True).stdout
    children = {}
    for line in o.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    tree, frontier = {root_pid}, [root_pid]
    while frontier:
        for kid in children.get(frontier.pop(), []):
            if kid not in tree:
                tree.add(kid)
                frontier.append(kid)
    return sorted(tree)


def _ps_stat_ps(pid):
    pids = _descendants(pid)
    o = subprocess.run(["ps", "-o", "rss=,cputime=", "-p",
                        ",".join(str(p) for p in pids)],
                       capture_output=True, text=True).stdout
    rss_kb, cpu = 0, 0.0
    seen = 0
    for line in o.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rss_kb += int(parts[0])
            cpu += _parse_cputime(parts[1])
            seen += 1
    if not seen:
        return None
    return {"rss_mb": round(rss_kb / 1024, 1),
            "cputime_s": cpu, "procs": seen}


def proc_stat_fields(text):
    """/proc/PID/stat: (ppid, utime + stime in ticks). The command name is
    in parentheses and may contain spaces or parentheses itself, so the
    fields are counted from the last ')'."""
    rest = text[text.rindex(")") + 2:].split()
    return int(rest[1]), int(rest[11]) + int(rest[12])


def _ps_stat_proc(pid):
    tick, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    ppid_of, ticks = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid_of[int(d)], ticks[int(d)] = proc_stat_fields(f.read())
        except (OSError, ValueError, IndexError):
            continue
    if pid not in ticks:
        return None
    tree, frontier = {pid}, [pid]
    while frontier:
        parent = frontier.pop()
        for kid, pp in ppid_of.items():
            if pp == parent and kid not in tree:
                tree.add(kid)
                frontier.append(kid)
    rss, cpu = 0, 0
    for p in tree:
        cpu += ticks.get(p, 0)
        try:
            with open(f"/proc/{p}/statm") as f:
                rss += int(f.read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            pass
    return {"rss_mb": round(rss / 2**20, 1), "cputime_s": cpu / tick,
            "procs": len(tree)}


def filetime_seconds(high, low):
    """A Windows FILETIME (100 ns units) as seconds."""
    return ((high << 32) | low) / 1e7


def _ps_stat_windows(pid):
    import ctypes, ctypes.wintypes as wt
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
    k32.OpenProcess.restype = wt.HANDLE

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    # QUERY_LIMITED_INFORMATION | QUERY_INFORMATION | VM_READ
    h = k32.OpenProcess(0x1000 | 0x0400 | 0x0010, False, pid)
    if not h:
        return None
    try:
        code = wt.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:
            return None   # 259: STILL_ACTIVE
        c, e, kt, ut = (wt.FILETIME() for _ in range(4))
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not (k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                                    ctypes.byref(kt), ctypes.byref(ut)) and
                psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)):
            return None
        cpu = (filetime_seconds(kt.dwHighDateTime, kt.dwLowDateTime) +
               filetime_seconds(ut.dwHighDateTime, ut.dwLowDateTime))
        # the runner does not fork; the process is the tree
        return {"rss_mb": round(pmc.WorkingSetSize / 2**20, 1),
                "private_mb": round(pmc.PagefileUsage / 2**20, 1),
                "cputime_s": cpu, "procs": 1}
    finally:
        k32.CloseHandle(h)


if sys.platform == "win32":
    ps_stat, ACCOUNTING = _ps_stat_windows, "process (GetProcessTimes, GetProcessMemoryInfo)"
elif sys.platform.startswith("linux"):
    ps_stat, ACCOUNTING = _ps_stat_proc, "process-tree (/proc)"
else:
    ps_stat, ACCOUNTING = _ps_stat_ps, "process-tree"


def vram_used_mb():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        return float(o.split()[0])
    except Exception:
        return None

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

def system_memory():
    """(key, MB): macOS wired memory, else the GPU memory in use."""
    if sys.platform == "darwin":
        return "wired_mb", wired_mb()
    return "vram_used_mb", vram_used_mb()


def idle_window(pid, label):
    a = ps_stat(pid); time.sleep(IDLE_S); b = ps_stat(pid)
    if not a or not b:
        return {"state": label, "error": "process exited"}
    key, mb = system_memory()
    r = {"state": label, "idle_window_s": IDLE_S,
         "cpu_s_during_idle": round(b["cputime_s"] - a["cputime_s"], 2),
         "rss_mb_end": b["rss_mb"], "procs": b["procs"],
         "accounting": ACCOUNTING, key + "_system": mb}
    if "private_mb" in b:
        r["private_mb_end"] = b["private_mb"]
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, help="full server command line")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", default="idle_coexistence_report.json")
    args = ap.parse_args()

    key, mb = system_memory()
    report = {"cmd": args.cmd, "idle_window_s": IDLE_S, "platform": sys.platform,
              key + "_baseline": mb, "states": []}
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
        report[key + "_after_kill"] = system_memory()[1]
        json.dump(report, open(args.out, "w"), indent=1)
        print(json.dumps(report, indent=1))
        print(f"\nreport -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
