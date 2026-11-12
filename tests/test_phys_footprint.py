"""/health and /metrics report the process's physical footprint on macOS
(R6.68.12).

A serve soak on Metal read rss_bytes at 13 to 16 MB while the process's
footprint was 638 MB, 598 MB of it IOAccelerator: the resident size of
MACH_TASK_BASIC_INFO leaves out the GPU driver's memory for the process (Metal
buffers) and falls as the system compresses pages, so a device-memory leak on
Metal was invisible. phys_footprint_bytes is the kernel's phys_footprint
(TASK_VM_INFO), the figure Activity Monitor and footprint(1) show.

Absolute anchor: /usr/bin/footprint, the operating system's own reading of the
same process. The field must cover every byte it attributes to IOAccelerator
(the Metal buffers) and agree with its total. On Linux and Windows the field
is absent: neither has one figure that adds device memory to RAM, and their
GPU memory is reported apart, so a number there would only repeat rss_bytes.
"""
import json
import pathlib
import re
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from _process import RunnerServer, find_runner  # noqa: E402

UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}


def _bytes(value, unit):
    return float(value) * UNITS[unit]


def _footprint(pid):
    """(total, IOAccelerator dirty bytes) as footprint(1) reads the process."""
    out = subprocess.run(["/usr/bin/footprint", str(pid)], capture_output=True,
                         text=True, timeout=60).stdout
    total = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", out)
    assert total, out[:500]
    io = 0.0
    for line in out.splitlines():
        m = re.match(r"\s*([\d.]+)\s*(B|KB|MB|GB)\s+.*IOAccelerator", line)
        if m:
            io += _bytes(m.group(1), m.group(2))
    return _bytes(total.group(1), total.group(2)), io


def _get(srv, path):
    with urllib.request.urlopen(srv.base_url + path, timeout=30) as r:
        return r.read().decode("utf-8")


def test_the_physical_footprint_is_reported_where_the_os_has_one(tmp_path):
    log = tmp_path / "server.log"
    with RunnerServer(find_runner(str(ROOT)), str(ROOT / "test.gguf"), ctx=256, parallel=1,
                      log_path=str(log)) as srv:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                           "temperature": 0}).encode()
        req = urllib.request.Request(srv.base_url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            assert r.status == 200
        health = json.loads(_get(srv, "/health"))
        metrics = _get(srv, "/metrics")
        if sys.platform != "darwin":
            assert "phys_footprint_bytes" not in health, health
            assert "runner_phys_footprint_bytes" not in metrics
            return
        assert "rss_bytes" in health, "rss_bytes stays as it was"
        fp = health.get("phys_footprint_bytes")
        assert isinstance(fp, int) and fp > 0, health
        line = re.search(r"^runner_phys_footprint_bytes (\d+)$", metrics, re.M)
        assert line, "the gauge is in /metrics"
        total, io = _footprint(srv.proc.pid)
        on_metal = "Metal backend" in log.read_text(errors="replace")
        if on_metal:
            assert io > 0, "a Metal load holds IOAccelerator memory"
        # the field counts the Metal buffers, which rss_bytes leaves out
        assert fp >= io * 0.95, (fp, io)
        assert int(line.group(1)) >= io * 0.95, (line.group(1), io)
        # and it is the figure footprint(1) reads, taken a moment apart
        assert abs(fp - total) <= max(0.25 * total, 16 * 1024 ** 2), (fp, total)
