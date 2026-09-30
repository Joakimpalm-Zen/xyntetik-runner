"""/v1/embeddings honours the pooling a GGUF declares.

An embedding checkpoint says how its vectors are read out: Qwen3-Embedding
publishes last-token pooling over an appended end-of-text token
(sentence-transformers' `pooling_mode_lasttoken`, and a tokenizer that adds
`<|endoftext|>`), which the GGUF carries as `{arch}.pooling_type = 3` and
`tokenizer.ggml.add_eos_token = true`. The endpoint read neither: every model
was mean-pooled over its tokens with no end token, so a last-token model
answered 200 with vectors its publisher never defined.

The anchor is computed from the file, not from the forward under test: the
fixture's attention and FFN write zero into the residual stream
(`--zero-branches`), so a position's final hidden state is its token's
embedding row, and after the final RMSNorm (unit weights) and the endpoint's
L2 normalisation an embedding read out at one position is exactly that row,
normalised. Last-token pooling over an appended EOS is therefore the EOS row
for EVERY input.
"""
import json
import math
import pathlib
import struct
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import RunnerServer  # noqa: E402

EOS = 2   # the fixture's tokenizer.ggml.eos_token_id


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


def _fixture(tmp_path_factory, name, *flags):
    m = tmp_path_factory.mktemp("pool") / f"{name}.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"),
                    "--zero-branches", *flags, str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _embedding_rows(path):
    """token_embd.weight as a list of float rows, read straight from the GGUF."""
    with open(path, "rb") as f:
        data = f.read()
    at = 0

    def take(fmt):
        nonlocal at
        v = struct.unpack_from(fmt, data, at)
        at += struct.calcsize(fmt)
        return v[0]

    def skip_value(t):
        nonlocal at
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sizes:
            at += sizes[t]
        elif t == 8:
            n = take("<Q")
            at += n
        elif t == 9:
            et, n = take("<I"), take("<Q")
            for _ in range(n):
                skip_value(et)
        else:
            raise ValueError(f"metadata type {t}")

    assert data[:4] == b"GGUF"
    at = 8
    n_tensors, n_kv = take("<Q"), take("<Q")
    for _ in range(n_kv):
        n = take("<Q")            # key length, then the key
        at += n
        skip_value(take("<I"))
    infos = {}
    for _ in range(n_tensors):
        name = data[at + 8:at + 8 + struct.unpack_from("<Q", data, at)[0]].decode()
        at += 8 + len(name)
        dims = [take("<Q") for _ in range(take("<I"))]
        ttype, off = take("<I"), take("<Q")
        infos[name] = (dims, ttype, off)
    base = (at + 31) // 32 * 32
    dims, ttype, off = infos["token_embd.weight"]
    assert ttype == 0, "the fixture's embedding table is F32"
    width, rows = dims
    flat = struct.unpack_from(f"<{width * rows}f", data, base + off)
    return [flat[r * width:(r + 1) * width] for r in range(rows)]


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def _embed(srv, text):
    req = urllib.request.Request(
        srv.base_url + "/v1/embeddings",
        data=json.dumps({"input": text}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _serve(runner_bin, model):
    return RunnerServer(runner_bin, model, ctx=256, parallel=1,
                        extra_args=["--gpu", "off", "-t", "1"])


def test_last_token_pooling_over_an_appended_eos(runner_bin, tmp_path_factory):
    m = _fixture(tmp_path_factory, "last-eos", "--pooling", "3", "--add-eos")
    want = _unit(_embedding_rows(m)[EOS])
    with _serve(runner_bin, m) as srv:
        for text in ("hello", "a completely different sentence, longer"):
            code, body = _embed(srv, text)
            assert code == 200, body
            got = body["data"][0]["embedding"]
            assert max(abs(g - w) for g, w in zip(got, want)) < 1e-5, text
            # the appended end token is part of what was embedded
            assert body["usage"]["prompt_tokens"] >= 3


def test_last_token_pooling_reads_the_last_position(runner_bin, tmp_path_factory):
    m = _fixture(tmp_path_factory, "last", "--pooling", "3")
    rows = [_unit(r) for r in _embedding_rows(m)]
    with _serve(runner_bin, m) as srv:
        code, body = _embed(srv, "hello there")
        assert code == 200, body
        got = body["data"][0]["embedding"]
    # one token's row, not a blend of several: some row matches exactly
    assert max(_cos(got, r) for r in rows) > 1 - 1e-6


def test_mean_pooling_is_the_default_and_the_declared_mean(runner_bin,
                                                           tmp_path_factory):
    plain = _fixture(tmp_path_factory, "none")
    mean = _fixture(tmp_path_factory, "mean", "--pooling", "1")
    rows = [_unit(r) for r in _embedding_rows(plain)]
    got = []
    for m in (plain, mean):
        with _serve(runner_bin, m) as srv:
            code, body = _embed(srv, "hello there")
            assert code == 200, body
            got.append(body["data"][0]["embedding"])
    assert max(abs(a - b) for a, b in zip(*got)) < 1e-6
    # a blend of the prompt's rows, not any single one of them
    assert max(_cos(got[0], r) for r in rows) < 0.999


@pytest.mark.parametrize("pooling,name", [(2, "cls"), (4, "rank")])
def test_undeclared_readouts_are_refused_by_name(runner_bin, tmp_path_factory,
                                                 pooling, name):
    m = _fixture(tmp_path_factory, name, "--pooling", str(pooling))
    with _serve(runner_bin, m) as srv:
        code, body = _embed(srv, "hello")
    assert code == 400, body
    assert name in body["error"]["message"].lower()


def test_float_and_base64_carry_the_same_vector(runner_bin, tmp_path_factory):
    """`encoding_format` chooses a spelling, not a vector. base64 carries the
    float32 bytes exactly; the float list printed each component with seven
    significant digits, and a float32 needs nine to read back, so the two
    formats of one request disagreed in the last bits."""
    import base64
    m = _fixture(tmp_path_factory, "enc", "--pooling", "1")
    with _serve(runner_bin, m) as srv:
        vecs = {}
        for fmt in ("float", "base64"):
            req = urllib.request.Request(
                srv.base_url + "/v1/embeddings",
                data=json.dumps({"input": "hello there",
                                 "encoding_format": fmt}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                vecs[fmt] = json.load(r)["data"][0]["embedding"]
    raw = base64.b64decode(vecs["base64"])
    decoded = struct.unpack(f"<{len(raw) // 4}f", raw)
    as_f32 = [struct.unpack("<f", struct.pack("<f", x))[0] for x in vecs["float"]]
    assert list(decoded) == as_f32
