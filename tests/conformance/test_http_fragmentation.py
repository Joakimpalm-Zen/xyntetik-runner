"""Exercise the real receive loop, rather than only its completed-buffer parsers."""
import socket

import pytest


def exchange(server, payload, chunk):
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as s:
        s.settimeout(10)
        try:
            for offset in range(0, len(payload), chunk):
                s.sendall(payload[offset:offset + chunk])
            s.shutdown(socket.SHUT_WR)
        except (BrokenPipeError, ConnectionResetError):
            pass  # an invalid header may be refused before all bytes arrive
        data = bytearray()
        while len(data) < 65536:
            try:
                block = s.recv(4096)
            except ConnectionResetError:
                break
            if not block:
                break
            data.extend(block)
        return bytes(data)


@pytest.mark.parametrize("chunk", [1, 3, 7, 4095, 16382])
@pytest.mark.parametrize("padding", [0, 8100, 16280])
def test_header_fragments_preserve_health_request(server, chunk, padding):
    request = b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Pad: " + b"x" * padding + b"\r\n\r\n"
    assert len(request) < 16384
    response = exchange(server, request, chunk)
    assert response.startswith(b"HTTP/1.1 200"), response[:300]


@pytest.mark.parametrize("chunk", [1, 7, 4095])
@pytest.mark.parametrize("payload", [
    b"GET /health HTTP/1.1\r\nHost: local\x00host\r\n\r\n",
    b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 33554433\r\n\r\n",
    b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\nContent-Length: 6\r\n\r\n{}",
    b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Pad: " + b"x" * 16384 + b"\r\n\r\n",
    b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 20\r\n\r\n{\"x\":",
])
def test_invalid_or_incomplete_receive_never_succeeds(server, chunk, payload):
    response = exchange(server, payload, chunk)
    # EOF can close without an error body; it must not execute a partial request.
    assert not response.startswith(b"HTTP/1.1 2"), response[:300]
    server.assert_alive()
