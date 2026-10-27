"""What a running Runner (or any OpenAI-style) endpoint serves and whether it
is up yet: the probes the comparison and calibration scripts share
(kld-compare.py, kld-compare-raw.py, decide-calibrate.py), and the free
loopback port the harnesses start a server on.

A script imports this from its own directory; one that a test loads by file
path puts that directory on sys.path first.
"""
import http.client
import json
import socket
import sys
import urllib.request


def free_port():
    """A TCP port the loopback interface can bind right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
