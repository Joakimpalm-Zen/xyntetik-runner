"""What happens to the server when the client stops holding up its end.

Every other test in this suite is a well-behaved client: it sends a request and
reads the answer. The write side is where a server gets taken down by traffic
that is not malicious, just interrupted — a browser tab closed mid-stream, a
proxy that times out, an agent that cancels. The failure modes are a fatal
SIGPIPE, a slot that is never returned, and a partially-written response that
corrupts the next one on a reused connection.

**The stall proper is not reachable from this harness, and neither is SIGPIPE.**
Both need the server's `send()` to actually block or fail, which needs a
response larger than the socket buffers between the two ends. The suite's model
is capped by `n_ctx` at ~68 KB of SSE, and on loopback that fits even with the
client's `SO_RCVBUF` pinned to 1 KB — measured: the entire response is buffered
and delivered to a client that never calls `recv()` once. A SIG_DFL build was
also constructed and run against these tests; all of them still pass, because
the write has already completed by the time the peer resets. Producing either
case needs a real model and a large context, i.e. a `scripts/`-sized experiment
rather than a conformance test. `scripts/write-stall.py` is that local Linux
gate; this fast portable suite deliberately remains the interrupted-client
half instead of duplicating or weakening it.

What these four do cover is the reachable and still-real half: an interrupted
client does not kill the process, strand its slot, or corrupt what the server
sends next.
"""

import json
import socket
import struct
import time

from _errors import ProtocolError

STREAM = {"model": "test", "messages": [{"role": "user", "content": "hello"}],
          "max_tokens": 400, "temperature": 0, "stream": True,
          "cache_prompt": False}
SHORT = {"model": "test", "messages": [{"role": "user", "content": "hi"}],
         "max_tokens": 8, "temperature": 0, "cache_prompt": False}


def _connect(server, payload, rcvbuf=None):
    s = socket.socket()
    if rcvbuf is not None:
        # Pin the receive window before connect, so it is advertised in the
        # handshake rather than applied after the buffers are already sized.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    s.connect(("127.0.0.1", server.port))
    body = json.dumps(payload).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(body) + body)
    return s


def _rst(s):
    """Close with SO_LINGER 0: an RST, not a FIN.

    A FIN is an orderly half-close the server may not even notice until it
    writes. An RST makes the next write fail hard, which is the case that used
    to kill a process on SIGPIPE.
    """
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    s.close()


def _works(client, name):
    r = client.chat(SHORT, name=name)
    r.expect_status(200)
    return r


def test_a_client_that_vanishes_mid_stream_does_not_take_the_server_with_it(
        server, client):
    """The RST arrives while the server is still writing SSE frames.

    **This does not gate `signal(SIGPIPE, SIG_IGN)`**, which is what it looks
    like it should do and what a first version of this docstring claimed. A
    build with the handler left at SIG_DFL was constructed and run against
    these four tests: all four still pass. The reason is the same measurement
    that defeats the stall case — the whole ~68 KB response fits in the socket
    buffers, so the server's `send()` has already returned before the RST
    arrives and there is no broken pipe to signal on. Gating SIGPIPE needs a
    response bigger than the buffers, which this model cannot produce.

    What it does gate is the ordinary shape of the failure: the server is still
    alive afterwards and did not lose the slot.
    """
    s = _connect(server, STREAM)
    s.settimeout(30)
    first = s.recv(256)          # streaming has actually started
    if not first.startswith(b"HTTP/1.1 200"):
        raise ProtocolError("the stream never started", got=first[:120])
    _rst(s)

    # The process must still be there, and still serving. Two requests, because
    # one proves the process is alive and two prove neither slot was lost.
    _works(client, "write-side-after-rst-1")
    _works(client, "write-side-after-rst-2")


def test_a_client_that_vanishes_before_reading_anything(server, client):
    """The same, but the RST lands during the prompt rather than the stream —
    the server has produced nothing yet and must not treat an empty write as a
    success or wedge waiting for a reader."""
    s = _connect(server, STREAM)
    _rst(s)
    _works(client, "write-side-after-early-rst-1")
    _works(client, "write-side-after-early-rst-2")


def test_many_vanishing_clients_do_not_leak_slots(server, client):
    """A slot leaked per abandoned connection would exhaust a 2-slot server in
    two iterations; ten makes the check unambiguous, and the final request
    would hang rather than fail if a slot were still held."""
    for i in range(10):
        s = _connect(server, STREAM)
        try:
            s.settimeout(10)
            s.recv(64)
        except (socket.timeout, OSError):
            pass
        _rst(s)
    _works(client, "write-side-after-ten-rst")


def test_a_reader_that_never_reads_does_not_stall_the_other_slot(server, client):
    """A dead reader holds its own slot; it must not hold anyone else's.

    See the module docstring for why the *stall* itself cannot be produced
    here. What is asserted is the part that is reachable and is also the part
    that matters in practice: whatever the dead reader is doing, the other slot
    keeps serving.
    """
    dead = _connect(server, STREAM, rcvbuf=1024)
    try:
        time.sleep(0.5)
        for i in range(3):
            t0 = time.time()
            _works(client, f"write-side-during-dead-reader-{i}")
            elapsed = time.time() - t0
            # Generous by two orders of magnitude: these return in ~0.02 s.
            # The failure this catches is serialization, not slowness.
            if elapsed > 20.0:
                raise ProtocolError(
                    "a request was stalled behind a client that stopped reading",
                    seconds=round(elapsed, 2), iteration=i)
    finally:
        _rst(dead)
    _works(client, "write-side-after-dead-reader")


def _raw_get(server, path, rcvbuf=None):
    s = socket.socket()
    if rcvbuf is not None:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    s.connect(("127.0.0.1", server.port))
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    return s


def _health(server):
    """/health over a fresh socket, timed. It is answered on the accept
    thread, so its latency is that thread's liveness."""
    t0 = time.time()
    s = socket.socket()
    s.settimeout(10.0)
    s.connect(("127.0.0.1", server.port))
    s.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    s.close()
    if not data.startswith(b"HTTP/1.1 200"):
        raise ProtocolError("/health did not answer 200", head=data[:60].decode(errors="replace"))
    return time.time() - t0


def _timed_get(server, path, timeout):
    """Status line latency of one GET over a fresh socket, or None on timeout."""
    t0 = time.time()
    s = socket.socket()
    s.settimeout(timeout)
    s.connect(("127.0.0.1", server.port))
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    try:
        head = s.recv(64)
    except socket.timeout:
        return None, None
    finally:
        s.close()
    return time.time() - t0, head


def test_large_gets_take_a_slot_and_small_ones_stay_on_the_accept_thread(server, client):
    """The accept thread answered GET /metrics, the context listing and the
    stored-response routes itself; a client that opened one and stopped
    reading parked the accept loop and nothing new was admitted (found
    2026-10-05). Those routes now take a slot; /health, /v1/models and
    /v1/capabilities stay on the accept thread so a busy slot cannot hide
    them. The stall itself needs a body larger than the loopback buffers
    (the module docstring); what is observable here is the ROUTING: with the
    only slot held by a generation, an accept-thread route answers at once
    and a slot route waits for it. (Both slots are held, so the wait is
    certain and not a race with a free slot.)"""
    r = client.responses({"model": "test", "input": "store this",
                          "max_output_tokens": 8, "store": True,
                          "temperature": 0}, name="write-side-stored")
    r.expect_status(200)
    rid = r.json["id"]
    # Hold every slot (the suite's server runs two). A generation on the toy
    # model is over in milliseconds, so the hold is a request whose declared
    # body never arrives: the slot waits on it up to its 10 s read deadline,
    # which is longer than every probe below.
    busy = []
    for _ in range(server.parallel):
        b = socket.socket()
        b.connect(("127.0.0.1", server.port))
        b.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                  b"Content-Type: application/json\r\nContent-Length: 4000\r\n\r\n"
                  b"{\"messages\":")
        busy.append(b)
    try:
        time.sleep(0.5)
        for path in ("/health", "/v1/models", "/v1/capabilities"):
            lat, head = _timed_get(server, path, timeout=5.0)
            if lat is None or lat > 2.0:
                raise ProtocolError("an accept-thread route waited for the busy slot",
                                    path=path, seconds=lat)
        for path in ("/metrics", f"/v1/responses/{rid}", "/v1/runner/contexts"):
            lat, head = _timed_get(server, path, timeout=1.0)
            if lat is not None:
                raise ProtocolError("a large-body route was answered on the accept "
                                    "thread while the slot was busy", path=path,
                                    seconds=round(lat, 3))
    finally:
        for b in busy:
            _rst(b)
    # released: the slot routes answer again
    lat, head = _timed_get(server, f"/v1/responses/{rid}", timeout=30.0)
    if lat is None or not head.startswith(b"HTTP/1.1 200"):
        raise ProtocolError("stored response unreadable once the slot was free")
    _works(client, "write-side-after-routing-check")
