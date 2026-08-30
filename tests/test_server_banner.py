"""The startup banner is the operator's route-discovery surface."""

import contextlib
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLIC_ROUTES = {
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/responses"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("GET", "/v1/models"),
    ("GET", "/v1/capabilities"),
    ("GET", "/v1/runner/prefix-cache"),
    ("POST", "/v1/runner/prefix-cache/clear"),
    ("GET", "/health"),
    ("POST", "/unload"),
}


def _free_port():
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_startup_banner_lists_every_public_route(tmp_path):
    runner = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not runner.exists():
        return
    model = tmp_path / "test.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py", model],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    )
    port = _free_port()
    proc = subprocess.Popen(
        [runner, "-m", model, "--serve", "--port", str(port), "--gpu", "off",
         "--no-tray"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError("server exited before its health check")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.05)
        else:
            raise AssertionError("server did not become healthy")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    banner = proc.stderr.read().decode(errors="replace")
    advertised = set(re.findall(r"\b(GET|POST) (/[^ |\r\n]+)", banner))
    assert advertised == PUBLIC_ROUTES, (
        f"startup banner route drift: missing={sorted(PUBLIC_ROUTES - advertised)}, "
        f"extra={sorted(advertised - PUBLIC_ROUTES)}\n{banner}"
    )
