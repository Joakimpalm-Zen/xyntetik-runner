"""What a running Runner (or any OpenAI-style) endpoint serves and whether it
is up yet: the probes the comparison and calibration scripts share
(kld-compare.py, kld-compare-raw.py, decide-calibrate.py), and the free
loopback port the harnesses start a server on.

A script imports this from its own directory; one that a test loads by file
path puts that directory on sys.path first.
"""
import contextlib
import http.client
import json
import os
import random
import re
import socket
import sys
import time
import urllib.request


def _worker_slice():
    m = re.fullmatch(r"gw(\d+)", os.environ.get("PYTEST_XDIST_WORKER", ""))
    if not m:
        return 0, 1
    idx = int(m.group(1))
    try:
        count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0"))
    except ValueError:
        count = 0
    stride = max(count, idx + 1, 32)
    return idx % stride, stride


def free_port():
    """A TCP port the loopback interface can bind right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def low_free_port(avoid=()):
    """A port below every OS's ephemeral range that binds now, for a server a
    script starts and then polls (see self_connected below for why below).
    Disjoint across pytest-xdist workers when PYTEST_XDIST_WORKER is set;
    wait_for_own_server also verifies the server pid."""
    rng = random.SystemRandom()
    offset, stride = _worker_slice()
    tried = set(avoid)
    slice_count = (32767 - (20000 + offset)) // stride + 1
    for _ in range(min(250, slice_count)):
        port = 20000 + offset + rng.randrange(slice_count) * stride
        if port in tried:
            continue
        tried.add(port)
        with contextlib.closing(socket.socket()) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    for _ in range(500):
        port = rng.randint(20000, 32767)
        if port in tried:
            continue
        tried.add(port)
        with contextlib.closing(socket.socket()) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError("no free port in 20000-32767")


def served_model_ids(endpoint, timeout=20):
    """The ids the endpoint serves (GET /v1/models), or None if it does not
    answer. One request at startup turns a wrong --model-name into a
    one-line error instead of a full-length run of 404s (found on the Linux
    CUDA host 2026-09-22: 25 minutes, 1,303 failures, exit 0)."""
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    data = body.get("data", []) if isinstance(body, dict) else []
    return [m["id"] for m in data if isinstance(m, dict) and m.get("id")]


def require_served(endpoint, name, label):
    """Fail at once, naming the served ids, rather than 404 on every position
    (a label instead of the served id gave a full run with nothing scored)."""
    ids = served_model_ids(endpoint)
    if ids is None:
        # a dead endpoint keeps the old contract: every position fails, the
        # evidence is still emitted with positions_scored 0, exit 1
        print(f"warning: {endpoint} did not answer GET /v1/models ({label}); "
              f"proceeding, every position will fail if it stays down", file=sys.stderr)
        return
    if name not in ids:
        raise SystemExit(f"error: --model-name-{label} {name!r} is not served by {endpoint}; "
                         f"served ids: {', '.join(ids) if ids else '(none)'}")


# A poll of a port nothing listens on yet can be handed that same port as its
# own source port when the port is inside the OS's ephemeral range (Linux
# 32768-60999, macOS and Windows 49152-65535): the socket connects to itself
# and reads its own request back, which http.client reports as BadStatusLine
# and which ended a CI run on 2026-10-23. The default ports sit below every
# ephemeral range, so that cannot happen with them, and a self-connection is
# never taken for a server should a caller pass a port inside one.
def self_connected(sock):
    return sock.getsockname() == sock.getpeername()


def server_up(port):
    """True once a server other than this poll answers /v1/models with 2xx."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        conn.connect()
        if self_connected(conn.sock):
            return False
        conn.request("GET", "/v1/models")
        return 200 <= conn.getresponse().status < 300
    except OSError:
        # refused, reset or timed out: not up yet, as before
        return False
    finally:
        conn.close()


def served_pid(port):
    """(answered, pid): whether GET /v1/capabilities answered, and the pid it
    names. A build from before the field existed answers without one."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/capabilities", timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception:
        return False, None
    pid = body.get("pid") if isinstance(body, dict) else None
    return True, pid if isinstance(pid, int) else None


BIND_ATTEMPTS = 5


class PortConflictError(RuntimeError):
    """Another process already answers on the requested loopback port."""


def port_flag_given(argv, flag):
    """True when CLI `argv` explicitly passed `flag` (e.g. `--port-a`)."""
    prefix = flag + "="
    return any(a == flag or a.startswith(prefix) for a in argv)


def _foreign_port_error(port, pid, what):
    return PortConflictError(
        f"port {port} is served by another process (pid {pid}), not the runner "
        f"started for {what}; stop it or pass another port")


def wait_for_own_server(proc, port, what, tries=400):
    """Returns once the server answering on `port` is `proc`, the runner the
    caller started for `what`. A server that answers there and is another
    process is refused: taking it for the caller's own scored that server's
    model under the caller's file name (two comparisons at once, or an
    earlier run still up). Raises PortConflictError (a RuntimeError subclass)
    on a foreign server or RuntimeError otherwise; the process is stopped first."""
    for _ in range(tries):
        if proc.poll() is not None:
            if server_up(port):
                answered, pid = served_pid(port)
                if answered and pid is not None and pid != proc.pid:
                    raise _foreign_port_error(port, pid, what)
            raise RuntimeError(f"runner exited early loading {what} (code {proc.returncode})")
        if server_up(port):
            answered, pid = served_pid(port)
            if answered and (pid is None or pid == proc.pid):
                return
            if answered:
                proc.kill()
                proc.wait()
                raise _foreign_port_error(port, pid, what)
        time.sleep(0.5)
    proc.kill()
    proc.wait()
    raise RuntimeError(f"server for {what} on port {port} did not come up")


def start_own_server(spawn_fn, port, what, attempts=BIND_ATTEMPTS, avoid=()):
    """Start `spawn_fn(port)` and wait for it to serve `port`, retrying on a
    fresh low_free_port if another process owns the port (R6.107). Non-conflict
    failures fail at once without retrying. Returns `(proc, actual_port)`."""
    tried = set(avoid)
    if port in tried:
        port = low_free_port(avoid=tried)
    for attempt in range(max(1, attempts)):
        tried.add(port)
        proc = spawn_fn(port)
        try:
            wait_for_own_server(proc, port, what)
            return proc, port
        except PortConflictError:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if attempt == max(1, attempts) - 1:
                raise
            port = low_free_port(avoid=tried)
    raise RuntimeError(f"server for {what} did not come up")
