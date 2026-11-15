"""Metal memory comes back (R6.17).

Metal's command buffers and encoders are autoreleased, and they retain what they
use. Runner calls Metal from plain C threads, which have no autorelease pool, so
nothing drained them: each load left about 113 MB in the GPU driver until the
process exited, and POST /unload reported a model gone with that memory still
held. Every Objective-C entry point of src/metal.m now drains its own pool.

The leak test reads what THIS server process holds, because the GPU driver's
"In use system memory" (ioreg) covers the whole machine: it failed the test
twice on 2026-11-14 (237 MB "held") while no Metal process of ours ran, moved by
something else on the machine (a browser), and passed each time alone. It stays
in the output as a note. The two figures the test asserts on, after each
/unload, measured over four cycles of the fixture:

- the Metal device's currentAllocatedSize, which the server prints after each
  free under RUNNER_TEST_METAL_HELD_AFTER_FREE: flat (0 bytes of growth) on a
  sound build, +1,130,496 bytes when the free path keeps one weight buffer.
  It does NOT see the leak R6.17 fixed (all 14 pools removed: flat at
  1,966,080 bytes while the machine-wide figure grew 124 MB a cycle);
- the process's own footprint (/health phys_footprint_bytes): 0.23 to 0.28 MB
  of growth on a sound build, 5.7 MB with the pools removed;
- the KV cache's host allocations (R3.68: one vm_allocate per cache, wrapped by
  per-region no-copy buffers), counted where they are made and freed and
  printed under the same seam: 0 left after every free. Neither figure above
  sees an unload that releases the region buffers but not the allocation under
  them: the device figure does not move, and at -c 256 the touched rows are
  kilobytes, far under the footprint's margin. On a real model that is the
  whole context's reservation leaked per load. Measured with that unload: 2
  allocations (65,536 bytes) more left after each cycle while the device figure
  stayed at 917,504 bytes; on a sound build 0 left after every free.

On the Apple Paravirtual device (the hosted macOS CI runners' GPU, named by the
runner's own "Metal backend on" line) the device figure and the footprint are
printed, not asserted: there they grow by about 1.5 MiB a load with every one
of our objects released, and the growth is the driver's, not ours. Measured on
macos-latest (macOS 26.7.2, AppleParavirtDevice, runs 43673598433 and
43673917267, a throwaway debug/ branch):
- 32 unload cycles: the device figure rose +2 and +1 MiB alternately per free,
  linear to 48 MiB, the footprint with it (14.4 to 39.0 MB over 16);
- every Metal object this process made (each load: 66 length, 34 byte and 5
  no-copy buffers, 144 pipelines, 2 libraries, 5 queues) was counted released
  after every free, and 0 KV host allocations were left;
- created and released alone in a loop: queues, devices, 100 MiB of buffers,
  dispatches and the library compiled from source keep 0 bytes; the library
  with a pipeline for each of its functions keeps about 1.44 MiB a round (8
  rounds, 13.6 MB). Each MTLComputePipelineState we release keeps memory in
  that driver.
On an Apple M1 the same 32 cycles hold 917,504 bytes after every free, and the
footprint stays between 12.32 and 12.65 MB. What is ours is asserted on every
device: the seam reported every free and no KV host allocation outlived its
model. Compiling the pipelines once per process is R3.70.

test_every_entry_point_drains_its_own_pool keeps R6.17 out at the source.
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
# Growth from the first /unload to the last. The device's held bytes do not move
# on a sound build (one held weight buffer adds 377 KB a cycle); the footprint
# moves 0.23 to 0.28 MB (5.7 MB without the autorelease pools).
HELD_GROWTH = 256 * 1024
FOOTPRINT_GROWTH = 2 << 20
SEAM = "RUNNER_TEST_METAL_HELD_AFTER_FREE"

# Every exported function of src/metal.m that sends a message, directly or through
# a helper. Exported functions that send none need no pool.
ENTRY_POINTS = ["gpu_available", "gpu_max_working_set", "gpu_init", "gpu_lora_bind", "gpu_lora_unbind",
                "gpu_free", "gpu_forward_batch", "gpu_batch_create", "gpu_batch_free", "gpu_batch_decode",
                "gpu_rope_changed", "gpu_kv_reserve", "gpu_kv_rows"]


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
    """The machine-wide GPU driver figure, a note only (None without ioreg)."""
    if not shutil.which("ioreg"):
        return None
    out = subprocess.run(["ioreg", "-rc", "AGXAccelerator", "-d1"], capture_output=True, text=True).stdout
    m = re.search(r'"In use system memory"=(\d+)', out)
    return int(m.group(1)) >> 20 if m else None


@pytest.mark.skipif(sys.platform != "darwin", reason="the Metal device and the macOS footprint")
def test_this_process_returns_its_memory_across_unload(tmp_path):
    model = ROOT / "test.gguf"
    assert model.exists(), "build the generated fixture with make test.gguf"
    log = tmp_path / "server.log"
    with RunnerServer(find_runner(str(ROOT)), f"a={model},b={model}", ctx=256, parallel=1,
                      log_path=str(log), env={SEAM: "1"}) as srv:
        machine0 = _driver_mb()
        body = json.dumps({"model": "a", "messages": [{"role": "user", "content": "hi"}],
                           "max_tokens": 4, "temperature": 0}).encode()
        footprint = []
        for _ in range(CYCLES):
            urllib.request.urlopen(urllib.request.Request(
                srv.base_url + "/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"}), timeout=60).read()
            urllib.request.urlopen(urllib.request.Request(srv.base_url + "/unload", data=b""), timeout=10).read()
            time.sleep(0.3)
            health = json.loads(urllib.request.urlopen(srv.base_url + "/health", timeout=10).read())
            footprint.append(health["phys_footprint_bytes"])
        machine = _driver_mb()
        text = log.read_text(errors="replace")
    if "Metal backend" not in text:
        pytest.skip("this host served on the CPU")
    # the device by the runner's own line, not by an environment guess
    devices = set(re.findall(r"Metal backend on ([^(\n]+?) \(", text))
    paravirtual = any(d.startswith("Apple Paravirtual device") for d in devices)
    held = [int(x) for x in re.findall(r"the Metal device holds (\d+) bytes after the model is freed", text)]
    # live: the seam reported at every free, or the assertion below says nothing
    assert len(held) >= CYCLES, f"the seam reported {len(held)} frees for {CYCLES} unloads"
    held = held[-CYCLES:]
    kv = [(int(n), int(b), int(made)) for n, b, made in re.findall(
        r"(\d+) KV host allocations \((\d+) bytes\) remain after the model is freed, of (\d+) made", text)]
    assert len(kv) >= CYCLES, f"the KV host count reported {len(kv)} frees for {CYCLES} unloads"
    kv = kv[-CYCLES:]
    print(f"note: device bytes after each free {held}; KV host allocations left, bytes, made {kv}")
    # live: every load made its allocations, so the count below is of something
    assert all(b[2] > a[2] for a, b in zip(kv, kv[1:])), f"no KV host allocation was counted between frees: {kv}"
    assert all(n == 0 and b == 0 for n, b, _ in kv), \
        f"KV host allocations outlived the model they belonged to (count, bytes, made): {kv}"
    if machine is not None and machine0 is not None:
        print(f"note: the machine-wide GPU driver figure moved {machine - machine0} MB across the cycles "
              "(every process on the machine counts there; not asserted)")
    if paravirtual:
        # the driver keeps memory for each pipeline we release (docstring): a note
        # with the figures, and the assertions above, which are ours, stand
        print(f"note: {sorted(devices)}: the device figure grew {held[-1] - held[0]} B and the footprint "
              f"{footprint[-1] - footprint[0]} B across {CYCLES} unloads, the driver's (not asserted here)")
        return
    assert held[-1] - held[0] < HELD_GROWTH, \
        f"this process's Metal device holds {held[-1] - held[0]} more bytes after {CYCLES} unloads: {held}"
    assert footprint[-1] - footprint[0] < FOOTPRINT_GROWTH, \
        f"this process's footprint grew {footprint[-1] - footprint[0]} bytes across {CYCLES} unloads: {footprint}"
