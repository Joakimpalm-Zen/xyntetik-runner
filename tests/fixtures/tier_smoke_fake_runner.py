#!/usr/bin/env python3
"""A stand-in for the runner binary in tests/test_tier_smoke.py.

It answers the calls scripts/tier-smoke.py makes, in the real binary's
shapes: `--caps` (one JSON object), `-m FILE --adapt-info` (one JSON line),
and `-m FILE --serve --port P` (GET /health, POST /v1/chat/completions; the
backend banner on stderr unless `--gpu off`; exit 0 on SIGTERM).

The reply's content is FAKE_RUNNER_TEXT, or a default answer that names
Mercury. With --lora the content is marked, as an adapter changes the text;
--mtp-file changes nothing, as a draft head only proposes tokens.
FAKE_RUNNER_FALLBACK=1 makes a GPU run decline the model and run on the
CPU, as a Metal build does for a layout it has no kernel for.
FAKE_RUNNER_REQUEST_LOG names a file each chat request body is appended to.
"""

import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

argv = sys.argv[1:]
DEFAULT = ("Here they are, in order from the Sun:\n\n1. Mercury: the smallest planet "
           "and the closest to the Sun.\n2. Venus: the hottest planet.")

if "--caps" in argv:
    print(json.dumps({"version": "9.8.7", "os": "macos", "arch": "arm64",
                      "cpu_cores": 8,
                      "gpu": {"backend": "metal", "name": "Fake M"}}))
    sys.exit(0)

if "--adapt-info" in argv:
    print(json.dumps({"architecture": "llama",
                      "adapter_serving": {"supported": True, "reason": None,
                                          "device_kernels": "metal"},
                      "training": {"supported": True, "reason": None,
                                   "host": "cpu"}}))
    sys.exit(0)

if "--serve" in argv:
    port = int(argv[argv.index("--port") + 1])
    gpu_off = "--gpu" in argv and argv[argv.index("--gpu") + 1] == "off"
    if not gpu_off:
        if os.environ.get("FAKE_RUNNER_FALLBACK"):
            print("gpu: this model's MoE router has no Metal kernel - using CPU",
                  file=sys.stderr, flush=True)
        else:
            print("gpu: Metal backend on Fake M (zero-copy weights)",
                  file=sys.stderr, flush=True)
    # the serving form of the load line (the one-shot prints "| 4 threads |")
    print("loaded fake | llama | 2 layers | ctx 512 | 4 threads (divided across "
          "slots) | 0.01s", file=sys.stderr, flush=True)
    lora = "--lora" in argv

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            log = os.environ.get("FAKE_RUNNER_REQUEST_LOG")
            if log:
                with open(log, "ab") as f:
                    f.write(raw + b"\n")
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            req = json.loads(raw)
            text = os.environ.get("FAKE_RUNNER_TEXT", DEFAULT)
            if lora:
                text += " (adapter applied)"
            self._send(200, {"object": "chat.completion",
                             "choices": [{"index": 0, "finish_reason": "length",
                                          "message": {"role": "assistant",
                                                      "content": text}}],
                             "usage": {"prompt_tokens": 30,
                                       "completion_tokens": req.get("max_tokens")}})

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
    sys.exit(0)

print("fake runner: unexpected arguments %r" % (argv,), file=sys.stderr)
sys.exit(2)
