"""HTTP-level request cancellation on a clean client close.

The long prompt and batch size one keep prefill live long enough to observe
without a production-only delay hook.  Every model is the generated test.gguf
fixture; no network model or external service is involved.
"""

import json
import os
import re
import socket
import sys
import time

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "conformance"))

from _process import RunnerServer, find_runner  # noqa: E402


def _body(prompt, *, max_tokens=1):
    return json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "cache_prompt": False},
                      separators=(",", ":")).encode()


def _open(port, body, path="/v1/completions"):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall(("POST %s HTTP/1.1\r\n" % path).encode() +
              b"Host: localhost\r\n"
              b"Content-Type: application/json\r\n" +
              ("Content-Length: %d\r\n\r\n" % len(body)).encode() + body)
    return s


def _read_response(s, timeout=60):
    s.settimeout(timeout)
    data = bytearray()
    while True:
        part = s.recv(65536)
        if not part:
            break
        data.extend(part)
    s.close()
    head, sep, body = bytes(data).partition(b"\r\n\r\n")
    assert sep and b" 200 " in head.split(b"\r\n", 1)[0], bytes(data[:160])
    return json.loads(body)


def _request(port, body, timeout=60):
    return _read_response(_open(port, body), timeout)


def _drain_one_request_connection(s, timeout=60):
    s.settimeout(timeout)
    try:
        while s.recv(65536):
            pass
    except ConnectionResetError:
        pass
    finally:
        s.close()


def _wait_for_starts(path, count, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(path, "rb") as f:
                if f.read().count(b": start,") >= count:
                    return
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    raise AssertionError(f"request {count} never entered prefill")


def _started_id(path, index):
    """The id of a request in the order they started: ids are random (R10.15)."""
    with open(path, "rb") as f:
        return re.findall(rb"\] (\S+): start,", f.read())[index]


def _wait_for_log_count(path, marker, count, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with open(path, "rb") as f:
            if f.read().count(marker) >= count:
                return
        time.sleep(0.01)
    raise AssertionError(f"server log never recorded {marker!r} {count} times")


def test_disconnect_stops_prefill_without_false_cancellation(tmp_path):
    model = os.path.join(ROOT, "test.gguf")
    assert os.path.exists(model), "build the generated fixture with make test.gguf"
    log = tmp_path / "disconnect-server.log"
    # Byte-fallback tokenization makes this about 16k tokens (four per word
    # on the fixture's vocabulary), a batch-one prefill much longer than one
    # cancellation poll on CI CPUs and inside the reader's 60 s on an old
    # desktop. Twice the words used to read the same, because the prompt
    # buffer was cut at bytes plus sixteen tokens; now it is encoded in full.
    long_body = _body("x " * 4000)
    short_body = _body("slot reuse", max_tokens=4)

    with RunnerServer(find_runner(ROOT), model, ctx=32768, parallel=1,
                      extra_args=["--gpu", "off", "-b", "1"],
                      log_path=str(log)) as srv:
        reference = _request(srv.port, short_body)

        # Alive and quiet is the common prefill state: delay reads until the
        # server has spent time polling an otherwise empty receive buffer.
        slow = _open(srv.port, long_body)
        _wait_for_starts(log, 2)
        full_t0 = time.monotonic()
        time.sleep(0.1)
        slow_result = _read_response(slow)
        full_s = time.monotonic() - full_t0
        assert slow_result["choices"], "a slow reader was falsely cancelled"
        assert full_s > 0.2, "the long-prefill control was too short to gate cancellation"

        # Bytes becoming readable are allowed to be the next HTTP request.
        # Send them only after prefill starts, so they stay in the kernel receive
        # buffer for the probe rather than being absorbed by initial parsing.
        piped = _open(srv.port, long_body)
        _wait_for_starts(log, 3)
        piped.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        # Runner serves one request per connection. Closing after the first
        # response with the deliberately unread second request may send RST and
        # discard that response, so the lifecycle log is the honest oracle:
        # two lines for the third request mean start + normal completion. A false cancellation
        # during prefill produces only the start line.
        _drain_one_request_connection(piped)
        _wait_for_log_count(log, b"] " + _started_id(log, 2) + b":", 2)

        # A clean close must release the only slot within one batch-one chunk.
        abandoned = _open(srv.port, long_body)
        _wait_for_starts(log, 4)
        abandoned.close()
        cancel_t0 = time.monotonic()
        reused = _request(srv.port, short_body)
        cancel_s = time.monotonic() - cancel_t0

        assert reused["choices"][0]["text"] == reference["choices"][0]["text"], \
            "the slot reused after cancellation diverged from its cold output"
        assert cancel_s * 2 < full_s, \
            f"dead prefill was not cut promptly: reuse {cancel_s:.3f}s, full {full_s:.3f}s"


@pytest.mark.parametrize("path,long_option", [
    ("/v1/decide", False), ("/v1/rerank", False),
    ("/v1/embeddings", False), ("/v1/decide", True),
])
def test_disconnect_stops_readout_and_releases_slot(tmp_path, path, long_option):
    """A live long readout completes; abandoning it frees the only slot.

    There is no external absolute latency anchor. The control is the same
    live request's duration, measured on this runner:
    merely finishing the abandoned work cannot satisfy the reuse deadline.
    A cold short completion also anchors the slot's subsequent output.
    """
    import urllib.request

    text = "x " * 4000
    if path == "/v1/decide":
        payload = {"state": "state" if long_option else text, "questions": [
            {"question": "answer?", "options": [text if long_option else "yes", "no"]}]}
    elif path == "/v1/rerank":
        payload = {"query": "relevant?", "documents": [text], "rendering": "raw-v1"}
    else:
        payload = {"input": text}
    body = json.dumps(payload).encode()
    short = _body("slot reuse", max_tokens=4)
    model = os.path.join(ROOT, "test.gguf")
    with RunnerServer(find_runner(ROOT), model, ctx=32768, parallel=1,
                      extra_args=["--gpu", "off", "-b", "1", "-t", "1"],
                      log_path=str(tmp_path / "readout.log")) as srv:
        reference = _request(srv.port, short)
        start = time.monotonic()
        live = _read_response(_open(srv.port, body, path), timeout=120)
        full_s = time.monotonic() - start
        assert full_s > 1, "readout control was too short to gate cancellation"
        assert live["usage"]["prompt_tokens"] > 0

        abandoned = _open(srv.port, body, path)
        try:
            end = time.monotonic() + 10
            while time.monotonic() < end:
                with urllib.request.urlopen(srv.base_url + "/health", timeout=5) as r:
                    health = json.load(r)
                if health["active_requests"] == 1:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("long readout never became active")
        finally:
            abandoned.close()
        start = time.monotonic()
        reused = _request(srv.port, short, timeout=full_s / 2)
        cancel_s = time.monotonic() - start
        assert reused["choices"][0]["text"] == reference["choices"][0]["text"]
        assert cancel_s * 2 < full_s, \
            f"dead readout held slot: reuse {cancel_s:.3f}s, live {full_s:.3f}s"
