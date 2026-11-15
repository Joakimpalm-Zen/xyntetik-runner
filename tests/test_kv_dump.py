"""RUNNER_TEST_KV_DUMP: the KV rows a prefill stored at one position (R3.60).

R3.60 asks where a prompt fed in other batch sizes first stores different
bits. The seam prints, after the feed that writes position P, a 64-bit FNV-1a
hash of every layer's stored K and V row at P, and one named layer's rows byte
for byte as hex. Two runs (a 1-row tail and a whole-batch prefill) are then
diffed row by row: the first layer whose hash differs, and in the named
layer the first differing byte.

The checks, premise first (a gate needs a check that fails when the instrument
is inert):
- two different prompts give different dumps at the same position;
- the same prompt twice, each read as a fresh slot, gives the same dump;
- the hex row printed for the named layer hashes to the hash printed for it.
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
from harness import RunnerServer, find_runner  # noqa: E402

POS = 5
LAYER = 1
SEAM = "RUNNER_TEST_KV_DUMP"
LINE = re.compile(r"kvdump: pos (\d+) layer (\d+) k ([0-9a-f]{16}|-) v ([0-9a-f]{16}|-)")
ROW = re.compile(r"kvdump: pos (\d+) layer (\d+) (k|v) row ([0-9a-f]*)")


def fnv1a64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return f"{h:016x}"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    exe = find_runner(ROOT)
    if not pathlib.Path(exe).exists():
        pytest.skip("runner not built")
    model = tmp_path_factory.mktemp("kvdump") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, stdout=subprocess.DEVNULL)
    log = tmp_path_factory.mktemp("kvdump-log") / "server.log"
    with RunnerServer(exe, model, ctx=256, parallel=1, log_path=str(log),
                      extra_args=["--gpu", "off", "-t", "2"],
                      env={SEAM: f"{POS},{LAYER}"}) as srv:
        yield srv, log


def _dump(server, prompt):
    """the dump lines one fresh-slot prompt adds to the log"""
    srv, log = server
    before = log.read_text(errors="replace") if log.exists() else ""
    body = {"prompt": prompt, "max_tokens": 1, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(srv.base_url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    text = (log.read_text(errors="replace"))[len(before):]
    hashes = {int(l): (k, v) for p, l, k, v in LINE.findall(text) if int(p) == POS}
    rows = {(int(l), kind): hx for p, l, kind, hx in ROW.findall(text) if int(p) == POS}
    return hashes, rows, text


def test_the_seam_announces_itself(server):
    _, log = server
    _dump(server, "abcdefghij")
    assert f"test seam: {SEAM}={POS},{LAYER}" in log.read_text(errors="replace")


def test_premise_two_prompts_give_different_dumps(server):
    a, _, ta = _dump(server, "abcdefghij")
    b, _, tb = _dump(server, "zyxwvutsrq")
    assert a and b, (ta[-800:], tb[-800:])
    assert set(a) == set(b) and LAYER in a, (sorted(a), sorted(b))
    # live: an inert dump (zeros, a fixed buffer, the wrong position) would match
    assert a != b, a


def test_the_same_prompt_twice_gives_the_same_dump(server):
    a, ra, _ = _dump(server, "abcdefghij")
    b, rb, _ = _dump(server, "abcdefghij")
    assert a == b and ra == rb


def test_the_named_layer_row_is_what_was_hashed(server):
    hashes, rows, text = _dump(server, "abcdefghij")
    k_hex, v_hex = rows.get((LAYER, "k")), rows.get((LAYER, "v"))
    assert k_hex and v_hex, text[-800:]
    k, v = hashes[LAYER]
    assert fnv1a64(bytes.fromhex(k_hex)) == k and fnv1a64(bytes.fromhex(v_hex)) == v
