"""Tournament-sampling watermark (R1.8.1) and its detector (R1.8.2).

Off by default; `--watermark KEY` marks sampled generation on the CLI and in
serve mode, the record carries the key id, `--detect-watermark` scores a
record or a text, and `--verify` replays a marked record only with its key.

Anchors outside the runner: the detector is re-implemented here from the
published construction (hashlib SHA-256 for the context seed, SipHash-2-4
for the g-words -- this file's SipHash checked against openssl's) and must
count the same g-values the runner counts; hashlib names the key id. The
statistics are the detector's own z against a Bernoulli(1/2) null, which
tests/test_watermark.c holds on random streams.
"""
import hashlib
import json
import pathlib
import shutil
import struct
import subprocess
import sys
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

OPENSSL = shutil.which("openssl")
LAYERS, CONTEXT = 30, 4
M64 = (1 << 64) - 1


def _rotl(x, b):
    return ((x << b) | (x >> (64 - b))) & M64


def siphash24(k0, k1, m):
    v0, v1 = 0x736F6D6570736575 ^ k0, 0x646F72616E646F6D ^ k1
    v2, v3 = 0x6C7967656E657261 ^ k0, 0x7465646279746573 ^ k1

    def rnd(v0, v1, v2, v3):
        v0 = (v0 + v1) & M64; v1 = _rotl(v1, 13) ^ v0; v0 = _rotl(v0, 32)
        v2 = (v2 + v3) & M64; v3 = _rotl(v3, 16) ^ v2
        v0 = (v0 + v3) & M64; v3 = _rotl(v3, 21) ^ v0
        v2 = (v2 + v1) & M64; v1 = _rotl(v1, 17) ^ v2; v2 = _rotl(v2, 32)
        return v0, v1, v2, v3

    full = len(m) - len(m) % 8
    for i in range(0, full, 8):
        mi = int.from_bytes(m[i:i + 8], "little")
        v3 ^= mi
        v0, v1, v2, v3 = rnd(*rnd(v0, v1, v2, v3))
        v0 ^= mi
    b = (len(m) << 56) & M64 | int.from_bytes(m[full:], "little")
    v3 ^= b
    v0, v1, v2, v3 = rnd(*rnd(v0, v1, v2, v3))
    v0 ^= b
    v2 ^= 0xFF
    for _ in range(4):
        v0, v1, v2, v3 = rnd(v0, v1, v2, v3)
    return v0 ^ v1 ^ v2 ^ v3


def _seed(key, toks, t):
    ctx = toks[max(0, t - CONTEXT):t]
    d = hashlib.sha256(key + b"xyntetik.wm.ctx.v1" + bytes([len(ctx)]) +
                       struct.pack("<%di" % len(ctx), *ctx)).digest()
    return struct.unpack("<QQ", d[:16])


def py_detect(key, toks, start):
    """(scored, ones): the construction in watermark.h, from scratch."""
    scored = ones = 0
    seen = set()
    for t in range(start, len(toks)):
        ctx = tuple(toks[max(0, t - CONTEXT):t])
        if ctx in seen:
            continue
        seen.add(ctx)
        k0, k1 = _seed(key, toks, t)
        w = siphash24(k0, k1, struct.pack("<I", toks[t] & 0xFFFFFFFF))
        ones += bin(w & ((1 << LAYERS) - 1)).count("1")
        scored += 1
    return scored, ones


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fx(runner_bin, tmp_path_factory):
    d = tmp_path_factory.mktemp("wm")
    model = d / "model.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(model)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    keys = []
    for name in ("a.key", "b.key"):
        p = subprocess.run([runner_bin, "--watermark-keygen", d / name],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert p.returncode == 0, p.stderr
        keys.append(d / name)
    return {"d": d, "model": model, "key": keys[0], "other": keys[1]}


def _run(runner_bin, *args):
    return subprocess.run([str(runner_bin), *map(str, args)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=300)


def _gen(runner_bin, fx, rec, *extra, seed=5, n=300):
    p = _run(runner_bin, "-m", fx["model"], "-p", "the quick brown fox", "-n", n,
             "--temp", "1.0", "-s", seed, "--gpu", "off", "-t", "2", "-c", "512",
             "--ignore-eos", "--transcript", rec, *extra)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
    return json.loads(rec.read_text())


def _detect(runner_bin, target, key, *extra):
    p = _run(runner_bin, "--detect-watermark", target, "--watermark", key, *extra)
    out = json.loads(p.stdout) if p.stdout.strip() else None
    return p.returncode, out, p.stderr.decode(errors="replace")


def _key_bytes(path):
    return bytes.fromhex(json.loads(path.read_text())["key"])


def test_keygen(runner_bin, fx):
    k = json.loads(fx["key"].read_text())
    assert k["schema_version"] == "xyntetik.runner.watermark_key.v1"
    key = bytes.fromhex(k["key"])
    assert len(key) == 32 and _key_bytes(fx["other"]) != key
    assert k["key_id"] == hashlib.sha256(
        b"xyntetik.runner.watermark.key_id.v1" + key).digest()[:8].hex()
    p = _run(runner_bin, "--watermark-keygen", fx["key"])
    assert p.returncode != 0 and json.loads(fx["key"].read_text()) == k


def test_python_siphash_is_openssls():
    if not OPENSSL:
        pytest.skip("openssl not available")
    for key, msg in ((bytes(range(16)), bytes(range(15))),
                     (hashlib.sha256(b"k").digest()[:16], struct.pack("<I", 1234))):
        f = pathlib.Path(ROOT / ".wm-siphash-msg.tmp")
        try:
            f.write_bytes(msg)
            o = subprocess.run([OPENSSL, "mac", "-macopt", "hexkey:" + key.hex(),
                                "-macopt", "size:8", "-in", str(f), "SIPHASH"],
                               check=True, stdout=subprocess.PIPE).stdout.strip()
        finally:
            f.unlink(missing_ok=True)
        k0, k1 = struct.unpack("<QQ", key)
        assert int.from_bytes(bytes.fromhex(o.decode()), "little") == siphash24(k0, k1, msg)


def test_a_marked_generation_is_detected_and_named(runner_bin, fx):
    d = fx["d"]
    rec = _gen(runner_bin, fx, d / "marked.json", "--watermark", fx["key"])
    kid = json.loads(fx["key"].read_text())["key_id"]
    wm = rec["watermark"]
    assert {k: wm[k] for k in ("scheme", "key_id", "layers", "context")} == \
        {"scheme": "tournament-v1", "key_id": kid, "layers": 30, "context": 4}
    assert 0 < wm["marked_tokens"] <= 300
    rc, out, err = _detect(runner_bin, d / "marked.json", fx["key"])
    assert rc == 0 and out["verdict"] == "WATERMARKED", (out, err)
    assert out["key_id"] == kid and out["record_key_id"] == kid
    assert out["z"] > 8, out
    # the same count from the construction written out independently
    toks = rec["prompt"]["tokens"] + rec["output"]["tokens"]
    scored, ones = py_detect(_key_bytes(fx["key"]), toks, len(rec["prompt"]["tokens"]))
    assert (out["scored"], out["g_ones"]) == (scored, ones)
    # another key sees nothing
    rc, out, _ = _detect(runner_bin, d / "marked.json", fx["other"])
    assert rc == 2 and out["verdict"] == "NOT_DETECTED" and abs(out["z"]) < 4, out
    # nor does the key see an unmarked generation from the same seed
    plain = _gen(runner_bin, fx, d / "plain.json")
    assert "watermark" not in plain
    assert plain["output"]["tokens"] != rec["output"]["tokens"]
    rc, out, _ = _detect(runner_bin, d / "plain.json", fx["key"])
    assert rc == 2 and out["verdict"] == "NOT_DETECTED", out
    assert "record_key_id" not in out


def test_seed_deterministic_and_greedy_untouched(runner_bin, fx):
    d = fx["d"]
    a = _gen(runner_bin, fx, d / "a.json", "--watermark", fx["key"], seed=9, n=40)
    b = _gen(runner_bin, fx, d / "b.json", "--watermark", fx["key"], seed=9, n=40)
    assert a["output"]["tokens"] == b["output"]["tokens"]
    g1 = _run(runner_bin, "-m", fx["model"], "-p", "hello", "-n", 20, "--temp", "0",
              "--gpu", "off", "-t", "2", "-c", "256", "--watermark", fx["key"],
              "--transcript", d / "g1.json")
    g0 = _run(runner_bin, "-m", fx["model"], "-p", "hello", "-n", 20, "--temp", "0",
              "--gpu", "off", "-t", "2", "-c", "256", "--transcript", d / "g0.json")
    assert g1.returncode == 0 and g0.returncode == 0
    r1, r0 = (json.loads((d / f).read_text()) for f in ("g1.json", "g0.json"))
    assert r1["output"]["tokens"] == r0["output"]["tokens"]
    assert r1["watermark"]["marked_tokens"] == 0


def test_verify_replays_a_marked_record_only_with_its_key(runner_bin, fx):
    d = fx["d"]
    _gen(runner_bin, fx, d / "v.json", "--watermark", fx["key"], seed=3, n=60)
    base = ["-m", fx["model"], "--verify", d / "v.json", "--gpu", "off", "-t", "2",
            "-c", "512", "--ignore-eos"]
    ok = _run(runner_bin, *base, "--watermark", fx["key"])
    assert ok.returncode == 0 and b"VERIFIED" in ok.stderr, ok.stderr[-400:]
    for extra, why in (([], b"verify with --watermark naming it"),
                       (["--watermark", fx["other"]], b"is not the record's")):
        v = _run(runner_bin, *base, *extra)
        assert v.returncode == 3 and why in v.stderr, (extra, v.stderr[-400:])
    _gen(runner_bin, fx, d / "u.json", seed=3, n=60)
    v = _run(runner_bin, "-m", fx["model"], "--verify", d / "u.json", "--gpu", "off",
             "-t", "2", "-c", "512", "--ignore-eos", "--watermark", fx["key"])
    assert v.returncode == 3 and b"watermark" in v.stderr, v.stderr[-400:]


def test_text_detection(runner_bin, fx):
    d = fx["d"]
    rec = _gen(runner_bin, fx, d / "t.json", "--watermark", fx["key"], seed=21)
    txt = d / "t.txt"
    # text is what a reader has: a NUL byte ends a C string and is not text
    txt.write_bytes(bytes.fromhex(rec["output"]["bytes_hex"]).replace(b"\0", b""))
    rc, out, err = _detect(runner_bin, txt, fx["key"], "-m", fx["model"])
    assert rc == 0 and out["verdict"] == "WATERMARKED" and out["source"] == "text", (out, err)
    rc, out, _ = _detect(runner_bin, txt, fx["other"], "-m", fx["model"])
    assert rc == 2, out
    # a text needs the model's tokenizer
    p = _run(runner_bin, "--detect-watermark", txt, "--watermark", fx["key"])
    assert p.returncode != 0 and b"-m" in p.stderr


def test_served_generation_is_marked(runner_bin, fx, tmp_path):
    rdir = tmp_path / "r"
    with RunnerServer(runner_bin, fx["model"], ctx=512, extra_args=[
            "--gpu", "off", "-t", "2", "--watermark", str(fx["key"]),
            "--ignore-eos", "--receipts", str(rdir)]) as srv:
        req = urllib.request.Request(srv.base_url + "/v1/completions", data=json.dumps({
            "prompt": "once upon a time", "max_tokens": 250, "temperature": 1.0,
            "seed": 4}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
    kid = json.loads(fx["key"].read_text())["key_id"]
    tel = body["runner_telemetry"]["watermark"]
    assert tel["key_id"] == kid and tel["marked_tokens"] > 0
    rec_path = rdir / body["runner_telemetry"]["receipt"]["file"]
    rec = json.loads(rec_path.read_text())
    assert rec["watermark"]["key_id"] == kid
    rc, out, err = _detect(runner_bin, rec_path, fx["key"])
    assert rc == 0 and out["verdict"] == "WATERMARKED", (out, err)


def test_speculative_decoding_marks_the_same_tokens(runner_bin, fx):
    """The speculative walk samples several positions per round; each pick
    must see its own context (hist up to that row), so a marked run drafting
    from the context produces exactly the tokens the plain walk does."""
    d = fx["d"]
    plain = _gen(runner_bin, fx, d / "s0.json", "--watermark", fx["key"], seed=13, n=120)
    # the model drafts for itself: every round verifies several rows, and
    # some drafts are accepted, so rows past the first are sampled in-walk
    spec = _gen(runner_bin, fx, d / "s1.json", "--watermark", fx["key"], "--draft",
                fx["model"], seed=13, n=120)
    assert spec["speculation"]["source"] == "model"
    assert spec["speculation"]["accepted"] > 0, spec["speculation"]
    assert spec["output"]["tokens"] == plain["output"]["tokens"]
    assert spec["watermark"]["marked_tokens"] == plain["watermark"]["marked_tokens"]
