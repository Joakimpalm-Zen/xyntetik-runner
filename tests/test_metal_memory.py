"""Metal memory comes back (R6.17).

Metal's command buffers and encoders are autoreleased, and they retain what they
use. Runner calls Metal from plain C threads, which have no autorelease pool, so
nothing drained them: each load left about 113 MB in the GPU driver until the
process exited, and POST /unload reported a model gone with that memory still
held. Every Objective-C entry point of src/metal.m now drains its own pool.

Absolute anchor: the GPU driver's own "In use system memory", read from ioreg.
It is outside Runner and outside Metal's API: currentAllocatedSize reports about
2 MB through all of this and cannot see it. The figure covers the whole machine,
so the bound below leaves room for other processes.
"""
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from _process import RunnerServer, find_runner  # noqa: E402

CYCLES = 4
# Without the pools this fixture holds about 110 MB per load. Across four loads a
# bound of 150 MB is under a third of that and over what a quiet machine moves.
HELD_MB = 150

# Every exported function of src/metal.m that sends a message, directly or through
# a helper. Exported functions that send none need no pool.
ENTRY_POINTS = ["gpu_available", "gpu_max_working_set", "gpu_init", "gpu_lora_bind", "gpu_lora_unbind",
                "gpu_free", "gpu_forward_batch", "gpu_batch_create", "gpu_batch_free", "gpu_batch_decode"]


def _body(source, name):
    m = re.search(r"^[a-z][^\n;]*\b" + name + r"\([^;{]*\{", source, re.M)
    assert m, name
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        i += 1
    return source[m.end():i]


def test_every_entry_point_drains_its_own_pool():
    source = (ROOT / "src" / "metal.m").read_text(encoding="utf-8")
    bare = [n for n in ENTRY_POINTS if not _body(source, n).lstrip().startswith("@autoreleasepool")]
    assert bare == []


def test_no_exported_function_sends_a_message_outside_the_list():
    source = (ROOT / "src" / "metal.m").read_text(encoding="utf-8")
    exported = re.findall(r"^(?!static)[a-z][\w \*]*?\b(gpu_\w+)\([^;{]*\{", source, re.M)
    send = re.compile(r"\[\s*[A-Za-z_][\w\->\.]*\s+[a-zA-Z]")
    strays = [n for n in exported if n not in ENTRY_POINTS and send.search(_body(source, n))]
    assert strays == []


def _driver_mb():
    out = subprocess.run(["ioreg", "-rc", "AGXAccelerator", "-d1"], capture_output=True, text=True).stdout
    m = re.search(r'"In use system memory"=(\d+)', out)
    return int(m.group(1)) >> 20 if m else None


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("ioreg"), reason="the Metal driver's figure")
def test_driver_memory_returns_across_unload(tmp_path):
    if _driver_mb() is None:
        pytest.skip("no AGX driver figure here")
    model = ROOT / "test.gguf"
    assert model.exists(), "build the generated fixture with make test.gguf"
    log = tmp_path / "server.log"
    with RunnerServer(find_runner(str(ROOT)), f"a={model},b={model}", ctx=256, parallel=1,
                      log_path=str(log)) as srv:
        start = _driver_mb()
        body = json.dumps({"model": "a", "messages": [{"role": "user", "content": "hi"}],
                           "max_tokens": 4, "temperature": 0}).encode()
        for _ in range(CYCLES):
            urllib.request.urlopen(urllib.request.Request(
                srv.base_url + "/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"}), timeout=60).read()
            urllib.request.urlopen(urllib.request.Request(srv.base_url + "/unload", data=b""), timeout=10).read()
            time.sleep(0.3)
        held = _driver_mb() - start
        text = log.read_text(errors="replace")
    if "Metal backend" not in text:
        pytest.skip("this host served on the CPU")
    assert held < HELD_MB, f"{held} MB still held after {CYCLES} loads and /unload"
