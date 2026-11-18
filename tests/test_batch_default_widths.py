"""R3.74: the CPU route's default prompt batch rises from 64 (to 256 on a machine of 6 GiB or more), so every
session-image run and every --gpu off server changes width. What was written at one width must continue with the
same bytes at another (R3.60 holds a prompt's rows equal across batch widths); these gates pin it, both ways:

- a session image suspended at one width and resumed at the other is the straight run's image, byte for byte;
- a KV snapshot of one memory has the same KV bytes (kv_sha256) whichever width prefilled it, and a snapshot made at
  one width, loaded by a server at the other, answers as that server's own prefill does.

The prompts are longer than 64 tokens, so the narrow width takes the prompt in several passes and the wide one in
one (about 130 tokens of the fixture's own 256-token context): a prompt inside one pass would prove nothing.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

PROMPT = "facts: " + " ".join("item %d is in box %d." % (i, i * 3 % 5) for i in range(4))
N, HALF = 40, 20


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("widths")
    m = d / "model.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(m)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return m


def _run(exe, *args, ctx=True):
    r = subprocess.run([str(exe), "-t", "2", *(["-c", "256"] if ctx else []), *map(str, args)], cwd=ROOT,
                       capture_output=True, timeout=180)
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-600:]
    return r


def _width(w):
    return [] if w == "default" else ["-b", str(w)]


def test_the_prompt_is_longer_than_the_narrow_width(runner_bin, model, tmp_path):
    r = _run(runner_bin, "-m", model, "-p", PROMPT, "-n", 1, "--temp", "0", "--gpu", "off")
    n = int(r.stderr.decode(errors="replace").split("\nprompt: ")[1].split(" tok")[0])
    assert n > 64, n


@pytest.mark.parametrize("written,resumed", [(64, 256), (256, 64), (64, "default"), ("default", 64)])
def test_a_session_image_continues_the_same_at_another_width(runner_bin, model, tmp_path, written, resumed):
    gen = ["-m", model, "-p", PROMPT, "-s", "7", "--temp", "0.9", "--repeat-penalty", "1.2", "--ignore-eos"]
    _run(runner_bin, *gen, "-n", N, *_width(resumed), "--session-out", tmp_path / "straight.img")
    _run(runner_bin, *gen, "-n", N, *_width(written), "--suspend-after", HALF, "--session-out", tmp_path / "half.img")
    _run(runner_bin, "-m", model, "--resume", tmp_path / "half.img", *_width(resumed),
         "--session-out", tmp_path / "resumed.img", ctx=False)
    assert (tmp_path / "straight.img").read_bytes() == (tmp_path / "resumed.img").read_bytes(), \
        "written at -b %s, resumed at -b %s: the image differs from the straight run's" % (written, resumed)


def _post(srv, path, payload):
    req = urllib.request.Request(srv.base_url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as r:
        return r.code, json.load(r)


def _server(exe, model, snaps, w):
    return RunnerServer(exe, model, ctx=256, extra_args=["--gpu", "off", "-t", "2", "--kv-snapshots", str(snaps),
                                                         *_width(w)])


def _answer(srv, ctx_id):
    st, body = _post(srv, "/v1/completions", {"prompt": PROMPT + " question: which box? answer:", "max_tokens": 8,
                                              "temperature": 0, "context_id": ctx_id})
    assert st == 200, body
    return body["choices"][0]["text"]


def test_a_kv_snapshot_is_the_same_at_either_width_and_loads_across(runner_bin, model, tmp_path):
    made = {}
    for w in (64, 256):
        snaps = tmp_path / ("snaps-%d" % w)
        with _server(runner_bin, model, snaps, w) as srv:
            st, ctx = _post(srv, "/v1/runner/contexts", {"id": "mem", "prompt": PROMPT})
            assert st == 200 and ctx["tokens"] > 64, ctx
            st, snap = _post(srv, "/v1/runner/contexts/mem/snapshot", {})
            assert st == 200, snap
            made[w] = {"snaps": snaps, "kv": snap["kv_sha256"], "answer": _answer(srv, "mem")}
    assert made[64]["kv"] == made[256]["kv"], "one memory prefilled at -b 64 and -b 256 has two KV byte strings"
    for wrote, loads in ((64, 256), (256, 64)):
        with _server(runner_bin, model, made[wrote]["snaps"], loads) as srv:
            st, ctx = _post(srv, "/v1/runner/contexts", {"id": "mem2", "snapshot": "mem"})
            assert st == 200, ctx
            assert _answer(srv, "mem2") == made[loads]["answer"], \
                "a snapshot written at -b %d answers otherwise at -b %d" % (wrote, loads)
