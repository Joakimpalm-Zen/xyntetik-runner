"""R3.35: the host RAM a RUNNER_MOE_HOT server holds, after load and after warm
decode. Windows only: the working set from GetProcessMemoryInfo, available
memory from GlobalMemoryStatusEx. Greedy text and warm decode rate come from the
same requests. Usage: ram_read.py TAG BINARY MODEL HOTFILE
"""
import ctypes as C
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from ctypes import wintypes as W

TAG, BIN, MODEL, HOT = sys.argv[1:5]
P = "Write four paragraphs about how reservoirs and water towers keep a city's water pressure steady through the day."
PORT = 18641


class PMC(C.Structure):
    _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD), ("PeakWorkingSetSize", C.c_size_t),
                ("WorkingSetSize", C.c_size_t), ("QuotaPeakPagedPoolUsage", C.c_size_t),
                ("QuotaPagedPoolUsage", C.c_size_t), ("QuotaPeakNonPagedPoolUsage", C.c_size_t),
                ("QuotaNonPagedPoolUsage", C.c_size_t), ("PagefileUsage", C.c_size_t),
                ("PeakPagefileUsage", C.c_size_t)]


class MS(C.Structure):
    _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD), ("ullTotalPhys", C.c_uint64),
                ("ullAvailPhys", C.c_uint64), ("ullTotalPageFile", C.c_uint64),
                ("ullAvailPageFile", C.c_uint64), ("ullTotalVirtual", C.c_uint64),
                ("ullAvailVirtual", C.c_uint64), ("ullAvailExtendedVirtual", C.c_uint64)]


k32 = C.WinDLL("kernel32")
k32.K32GetProcessMemoryInfo.argtypes = [W.HANDLE, C.POINTER(PMC), W.DWORD]


def sample(p):
    m = PMC()
    m.cb = C.sizeof(PMC)
    assert k32.K32GetProcessMemoryInfo(int(p._handle), C.byref(m), m.cb)
    s = MS()
    s.dwLength = C.sizeof(MS)
    assert k32.GlobalMemoryStatusEx(C.byref(s))
    return round(m.WorkingSetSize / 1e9, 2), round(m.PeakWorkingSetSize / 1e9, 2), round(s.ullAvailPhys / 1e9, 2)


env = dict(os.environ, RUNNER_MOE_HOT=HOT)
p = subprocess.Popen([BIN, "-m", MODEL, "--serve", "--port", str(PORT), "--gpu", "99", "--cpu-moe", "99",
                      "-c", "4096", "--no-tray"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    time.sleep(2)
    print(TAG, "after load: ws, peak ws, available GB", sample(p), flush=True)
    text = None
    for i in range(4):
        body = {"prompt": P, "max_tokens": 128, "temperature": 0, "seed": 1, "stream": True, "cache_prompt": False}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        ts, out = [], []
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    ts.append(time.perf_counter())
                    c = json.loads(line[5:])["choices"]
                    if c:
                        out.append(c[0].get("text") or "")
        text = "".join(out)
        rate = (len(ts) - 1) / (ts[-1] - ts[0])
        print(TAG, "req", i, "tok/s", round(rate, 2), "ws, peak ws, available GB", sample(p), flush=True)
    print(TAG, "greedy text sha256", hashlib.sha256(text.encode()).hexdigest()[:16], "chars", len(text), flush=True)
finally:
    p.terminate()
    p.wait(timeout=30)
