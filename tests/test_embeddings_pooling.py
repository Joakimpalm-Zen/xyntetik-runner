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
import os
import pathlib
import re
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


def _fixture(tmp_path_factory, name, *flags):
    m = tmp_path_factory.mktemp("pool") / f"{name}.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"),
                    "--zero-branches", *flags, str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def _embedding_table(path):
    """The file's bytes, and where token_embd.weight sits in them: offset, width, rows."""
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
    return data, base + off, width, rows


def _embedding_rows(path):
    """token_embd.weight as a list of float rows, read straight from the GGUF."""
    data, at, width, rows = _embedding_table(path)
    flat = struct.unpack_from(f"<{width * rows}f", data, at)
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


def _metal(exe):
    try:
        caps = json.loads(subprocess.run([str(exe), "--caps"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, timeout=60).stdout)
    except Exception:
        return False
    return (caps.get("gpu") or {}).get("backend") == "metal"


@pytest.mark.parametrize("pooling", ["1", "3"])
def test_a_fully_offloaded_model_embeds_on_the_device(runner_bin, tmp_path, pooling):
    """A fully offloaded model's embeddings ran their forward on the CPU: the
    readout set the device aside because a full offload keeps the residual on
    the device, although the device hands those rows back when asked (the
    Clef readout has asked since it was admitted). The device now runs the
    forward, and the vector is the one the CPU gives."""
    if not _metal(runner_bin):
        pytest.skip("no Metal device: the forward is seen through its dispatch census")
    m = tmp_path / "m.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"),
                    "--pooling", pooling, str(m)], check=True, stdout=subprocess.DEVNULL)
    text = "the quick brown fox jumps over the lazy dog and keeps running"
    vec = {}
    for arm, extra in (("cpu", ["--gpu", "off"]), ("device", [])):
        log = tmp_path / f"{arm}.log"
        with RunnerServer(runner_bin, m, ctx=256, parallel=1, extra_args=extra,
                          log_path=str(log),
                          env=dict(os.environ, RUNNER_METAL_STATS="1")) as srv:
            code, body = _embed(srv, text)
            assert code == 200, body
            vec[arm] = body["data"][0]["embedding"]
            n = body["usage"]["prompt_tokens"]
        said = log.read_text(errors="replace")
        ran = f"metal: native prompt batch n={n} " in said
        assert ran == (arm == "device"), said[-2000:]
    assert _cos(vec["cpu"], vec["device"]) > 1 - 1e-5
    assert max(abs(a - b) for a, b in zip(vec["cpu"], vec["device"])) < 1e-3


def test_a_non_finite_embedding_is_an_error_not_invalid_json(runner_bin, tmp_path_factory):
    """JSON has no NaN and no infinity, and printf writes them as `nan` and
    `inf`: a forward that went non-finite (here a NaN in every row of the
    embedding table; in the field an overflowed activation or a failed kernel)
    answered 200 with a body no parser takes, and as base64 it handed the
    client NaN bytes with nothing to tell them by. It is an error the client
    can read, in both encodings."""
    m = _fixture(tmp_path_factory, "nan", "--pooling", "1")
    data, at, width, rows = _embedding_table(m)
    data = bytearray(data)
    for r in range(rows):
        struct.pack_into("<f", data, at + r * width * 4, float("nan"))
    m.write_bytes(data)
    with _serve(runner_bin, m) as srv:
        for fmt in ("float", "base64"):
            req = urllib.request.Request(
                srv.base_url + "/v1/embeddings",
                data=json.dumps({"input": "hello there", "encoding_format": fmt}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    status, raw = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, raw = e.code, e.read()
            body = json.loads(raw)   # whatever the status, the body parses
            assert status == 500, (fmt, raw[:200])
            assert "non-finite" in body["error"]["message"], body
        # the server is still there for the next request
        code, body = _embed(srv, "once more")
        assert code == 500, body


def _cuda(exe):
    try:
        caps = json.loads(subprocess.run([str(exe), "--caps"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, timeout=60).stdout)
    except Exception:
        return False
    return (caps.get("gpu") or {}).get("backend") == "cuda"


@pytest.mark.parametrize("pooling", ["1", "3"])
def test_a_fully_offloaded_model_embeds_on_the_cuda_device(runner_bin, tmp_path, pooling):
    """The CUDA twin of the test above (R6.68.107). CUDA has no dispatch
    census, so the forward is seen through RUNNER_CUDA_PROFILE: its summary,
    printed when the device is freed at shutdown, counts the prompt tiles the
    device ran. One request on one slot is the profile's single-thread case.
    Before R6.68.107 a fully offloaded model's embeddings left no tile at all;
    the server runs no device forward before a request, so the tile's tokens
    are the request's."""
    if not _cuda(runner_bin):
        pytest.skip("no CUDA device: the forward is seen through RUNNER_CUDA_PROFILE")
    m = tmp_path / "m.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"),
                    "--pooling", pooling, str(m)], check=True, stdout=subprocess.DEVNULL)
    text = "the quick brown fox jumps over the lazy dog and keeps running"
    vec = {}
    for arm, extra in (("cpu", ["--gpu", "off"]), ("device", [])):
        log = tmp_path / f"{arm}.log"
        with RunnerServer(runner_bin, m, ctx=256, parallel=1, extra_args=extra,
                          log_path=str(log),
                          env=dict(os.environ, RUNNER_CUDA_PROFILE="1")) as srv:
            code, body = _embed(srv, text)
            assert code == 200, body
            vec[arm] = body["data"][0]["embedding"]
            n = body["usage"]["prompt_tokens"]
        said = log.read_text(errors="replace")
        # the PREFILL block's first count line, past the note on graphs
        ran = re.search(r"RUNNER_CUDA_PROFILE \[PREFILL[^\n]*\n(?:[^\n]*\n)*?"
                        rf"tiles=\d+ tokens={n} ", said) is not None
        assert ran == (arm == "device"), said[-2000:]
    assert _cos(vec["cpu"], vec["device"]) > 1 - 1e-5
    assert max(abs(a - b) for a, b in zip(vec["cpu"], vec["device"])) < 1e-3
