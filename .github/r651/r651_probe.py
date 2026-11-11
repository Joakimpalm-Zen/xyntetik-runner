"""R6.51 TEMPORARY probe, never merged: why a connect on the hosted Windows
runner fails with WinError 10055 during the conformance harness.

Loaded with `-p r651_probe`. It counts every TCP connect this pytest process
makes (and the port it went to), samples the machine's TCP table every two
seconds, and reads the table again at the moment a connect raises 10055.
The harness's servers listen below the ephemeral range (tests/conformance/
_process.py, 20000-32767), so a TIME_WAIT entry whose local port is in the
dynamic range is a client socket and one on a server port is the server's.
"""
import collections
import json
import os
import re
import socket
import subprocess
import threading
import time

OUT = os.environ.get("R651_OUT", "r651-out")
os.makedirs(OUT, exist_ok=True)
TARGET = "test_list_contexts_during_creation"
ROW = re.compile(r"^\s*TCP\s+(\S+):(\d+)\s+(\S+):(\d+)\s+(\S+)")

_lock = threading.Lock()
_connects = 0
_recent = collections.deque()          # monotonic times of connects, last 300 s
_cur = {"test": None, "start": 0, "t0": 0.0}
_snaps = 0
_stop = threading.Event()
_t_start = time.monotonic()
DYN = (49152, 65535)


def _w(name, obj):
    with open(os.path.join(OUT, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _run(*cmd, timeout=60):
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001 - a probe reports, it never fails the run
        return f"<{type(e).__name__}: {e}>"


def _env():
    global DYN
    out = _run("netsh", "int", "ipv4", "show", "dynamicport", "tcp")
    m1 = re.search(r"Start Port\s*:\s*(\d+)", out)
    m2 = re.search(r"Number of Ports\s*:\s*(\d+)", out)
    if m1 and m2:
        DYN = (int(m1.group(1)), int(m1.group(1)) + int(m2.group(1)) - 1)
    reg = _run("reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters")
    keys = {k: v for k, v in re.findall(r"^\s*(TcpTimedWaitDelay|MaxUserPort|TcpNumConnections)\s+REG_\w+\s+(\S+)",
                                        reg, re.M)}
    _w("env.json", {"dynamic_range": DYN, "netsh": out.strip(), "tcpip_parameters": keys,
                    "netsh_ipv6": _run("netsh", "int", "ipv6", "show", "dynamicport", "tcp").strip()})


def _nonpaged():
    out = _run("typeperf", r"\Memory\Pool Nonpaged Bytes", "-sc", "1", timeout=30)
    m = re.findall(r'"[^"]*","([0-9.]+)"', out)
    return float(m[-1]) if m else None


def _table(dump=None):
    rows = []
    raw = _run("netstat", "-an", "-p", "TCP") + _run("netstat", "-an", "-p", "TCPv6")
    if dump:
        with open(os.path.join(OUT, dump), "w", encoding="utf-8") as f:
            f.write(raw)
    for line in raw.splitlines():
        m = ROW.match(line)
        if m:
            rows.append((int(m.group(2)), int(m.group(4)), m.group(5)))
    lo, hi = DYN
    tw = [r for r in rows if r[2] == "TIME_WAIT"]
    return {
        "rows": len(rows),
        "states": dict(collections.Counter(r[2] for r in rows)),
        "time_wait_client": sum(1 for l, _, _ in tw if lo <= l <= hi),
        "time_wait_server": sum(1 for l, _, _ in tw if not lo <= l <= hi),
        "dynamic_ports_held": len({l for l, _, _ in rows if lo <= l <= hi}),
        "dynamic_range_size": hi - lo + 1,
    }


def _counts():
    now = time.monotonic()
    with _lock:
        while _recent and now - _recent[0] > 300:
            _recent.popleft()
        n = _connects
        last120 = sum(1 for t in _recent if now - t <= 120)
        last240 = len(_recent)
    return {"t": round(now - _t_start, 2), "connects": n, "connects_last_120s": last120,
            "connects_last_240s_or_300s": last240, "test": _cur["test"]}


def _snapshot(why):
    global _snaps
    with _lock:
        _snaps += 1
        k = _snaps
    if k > 3:
        return
    c = _counts()
    tab = _table(dump=f"netstat-at-{why}-{k}.txt")
    snap = dict(c, why=why, table=tab, nonpaged_bytes=_nonpaged())
    _w("snapshots.jsonl", snap)
    print(f"\nR6.51 SNAPSHOT {json.dumps(snap)}", flush=True)


_orig_connect = socket.socket.connect


def _connect(self, address):
    global _connects
    with _lock:
        _connects += 1
        _recent.append(time.monotonic())
    try:
        return _orig_connect(self, address)
    except OSError as e:
        if getattr(e, "winerror", None) == 10055:
            _snapshot("connect-10055")
        raise


socket.socket.connect = _connect


def _sampler():
    i = 0
    while not _stop.wait(2.0):
        i += 1
        s = dict(_counts(), table=_table())
        if i % 5 == 0:
            s["nonpaged_bytes"] = _nonpaged()
        _w("samples.jsonl", s)


def pytest_configure(config):
    _env()
    threading.Thread(target=_sampler, daemon=True).start()


def pytest_runtest_logstart(nodeid, location):
    _cur.update(test=nodeid, start=_connects, t0=time.monotonic())
    if TARGET in nodeid:
        _w("target.jsonl", dict(_counts(), event="start", table=_table()))


def pytest_runtest_logreport(report):
    if report.failed and "10055" in str(report.longrepr):
        _snapshot(f"report-{report.when}")
    if report.when == "call" or report.failed:
        _w("tests.jsonl", {"test": report.nodeid, "when": report.when, "outcome": report.outcome,
                           "connects": _connects - _cur["start"],
                           "seconds": round(time.monotonic() - _cur["t0"], 2)})


def pytest_runtest_logfinish(nodeid, location):
    if TARGET in nodeid:
        _w("target.jsonl", dict(_counts(), event="finish", table=_table()))


def pytest_unconfigure(config):
    _stop.set()
    _w("final.json", dict(_counts(), table=_table()))
