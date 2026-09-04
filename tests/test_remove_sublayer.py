"""--remove-sublayer: physically drop one block's attention or FFN tensors
from a GGUF and declare the absence the way llama.cpp's own per-layer
arrays do (a 0 entry in attention.head_count / head_count_kv, or in
feed_forward_length), so the loader omits the graph node and reserves no
KV rows for it instead of failing on a missing tensor.

Absolute anchor: a block whose attention output projection is all zeros
contributes exactly nothing to the residual stream (0 * x = 0, and the
post-attention norm of a zero vector is zero), so the removed file must
score BIT-IDENTICALLY to the parent with that one tensor zeroed. That is an
independent path through the full attention math, not a transcription of
the omission, and the same holds for the FFN with ffn_down zeroed. The
removed file must also DIFFER from the untouched parent, so the gate can go
red.
"""
import json
import os
import pathlib
import platform
import re
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = "the quick brown fox jumps over the lazy dog and keeps running"
GGUF_U32, GGUF_ARR = 4, 9


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if os.name == "nt" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    out = tmp_path_factory.mktemp("rm") / "parent.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    str(out)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return out


@pytest.fixture(scope="module")
def declared_but_present(tmp_path_factory):
    """A hostile fixture: the arrays say blk.1 has no attention and blk.0 no
    FFN, but every tensor is still in the file."""
    out = tmp_path_factory.mktemp("rm-hostile") / "hostile.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--declare-removed", "attn:1,mlp:0", str(out)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return out


# ------------------------------------------------------------ GGUF header
TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _parse(path):
    """Header parse: kvs (arrays decoded for u32), tensor table with byte
    sizes, data_start. Byte sizes come from the NEXT tensor's offset (or the
    file end), i.e. the aligned span the file actually spends on it."""
    data = pathlib.Path(path).read_bytes()
    pos = 4
    (ver,) = struct.unpack_from("<I", data, pos); pos += 4
    n_tensors, n_kv = struct.unpack_from("<QQ", data, pos); pos += 16

    def rd_str():
        nonlocal pos
        (n,) = struct.unpack_from("<Q", data, pos); pos += 8
        s = data[pos:pos + n].decode(); pos += n
        return s

    def rd_val(t):
        nonlocal pos
        if t == 8:
            return rd_str()
        if t == GGUF_ARR:
            (et,) = struct.unpack_from("<I", data, pos); pos += 4
            (n,) = struct.unpack_from("<Q", data, pos); pos += 8
            return [rd_val(et) for _ in range(n)]
        raw = data[pos:pos + TYPE_SIZES[t]]; pos += TYPE_SIZES[t]
        if t == GGUF_U32:
            return struct.unpack("<I", raw)[0]
        return raw

    kvs = {}
    for _ in range(n_kv):
        k = rd_str()
        (t,) = struct.unpack_from("<I", data, pos); pos += 4
        kvs[k] = rd_val(t)
    align = kvs.get("general.alignment", 32)
    tensors = []
    for _ in range(n_tensors):
        name = rd_str()
        (nd,) = struct.unpack_from("<I", data, pos); pos += 4
        ne = list(struct.unpack_from("<%dQ" % nd, data, pos)); pos += 8 * nd
        (ty,) = struct.unpack_from("<I", data, pos); pos += 4
        (off,) = struct.unpack_from("<Q", data, pos); pos += 8
        tensors.append({"name": name, "ne": ne, "type": ty, "off": off})
    data_start = (pos + align - 1) // align * align
    by_off = sorted(tensors, key=lambda t: t["off"])
    for a, b in zip(by_off, by_off[1:] + [None]):
        end = b["off"] if b else len(data) - data_start
        a["span"] = end - a["off"]
    return {"kvs": kvs, "tensors": {t["name"]: t for t in tensors},
            "data_start": data_start, "size": len(data), "align": align}


def _payload(path, info, name):
    t = info["tensors"][name]
    with open(path, "rb") as f:
        f.seek(info["data_start"] + t["off"])
        return f.read(t["span"])


def _zero_tensor(src, dst, name):
    """Overwrite one tensor's payload with zero bytes (F32/F16/BF16 zero
    bits and a zero Q8_0 block scale all decode to 0.0)."""
    info = _parse(src)
    t = info["tensors"][name]
    data = bytearray(pathlib.Path(src).read_bytes())
    start = info["data_start"] + t["off"]
    data[start:start + t["span"]] = bytes(t["span"])
    pathlib.Path(dst).write_bytes(data)


def _drop_directory_entry(src, dst, name):
    """Remove one tensor from the directory WITHOUT declaring the absence:
    the header is re-emitted without the entry and the data section is
    copied verbatim (offsets are relative to data_start, so they stay valid).
    The bytes of the dropped tensor become dead space."""
    data = pathlib.Path(src).read_bytes()
    info = _parse(src)
    pos = 8
    n_tensors, n_kv = struct.unpack_from("<QQ", data, pos); pos += 16
    kv_start = pos

    def skip_str():
        nonlocal pos
        (n,) = struct.unpack_from("<Q", data, pos); pos += 8 + n

    def skip_val(t):
        nonlocal pos
        if t == 8:
            skip_str()
        elif t == GGUF_ARR:
            (et,) = struct.unpack_from("<I", data, pos); pos += 4
            (n,) = struct.unpack_from("<Q", data, pos); pos += 8
            for _ in range(n):
                skip_val(et)
        else:
            pos += TYPE_SIZES[t]

    for _ in range(n_kv):
        skip_str()
        (t,) = struct.unpack_from("<I", data, pos); pos += 4
        skip_val(t)
    kv_bytes = data[kv_start:pos]
    entries = []
    for _ in range(n_tensors):
        e0 = pos
        (n,) = struct.unpack_from("<Q", data, pos); pos += 8
        nm = data[pos:pos + n].decode(); pos += n
        (nd,) = struct.unpack_from("<I", data, pos); pos += 4 + 8 * nd + 12
        entries.append((nm, data[e0:pos]))
    kept = [b for nm, b in entries if nm != name]
    assert len(kept) == n_tensors - 1, name
    hdr = data[:8] + struct.pack("<QQ", n_tensors - 1, n_kv) + kv_bytes + b"".join(kept)
    align = info["align"]
    pad = (-len(hdr)) % align
    pathlib.Path(dst).write_bytes(hdr + bytes(pad) + data[info["data_start"]:])


# ---------------------------------------------------------------- runner
def _run(runner_bin, args, timeout=120):
    return subprocess.run([runner_bin, *map(str, args)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)


def _remove(runner_bin, src, out, spec, extra=()):
    return _run(runner_bin, ["-m", src, "--quantize", out,
                             "--remove-sublayer", spec, *extra])


def _score(runner_bin, model, extra=()):
    p = _run(runner_bin, ["-m", model, "--score", "-p", PROMPT, "-t", "2",
                          "--gpu", "off", *extra])
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)["logprobs"]


def _load_banner(runner_bin, model, extra=()):
    p = _run(runner_bin, ["-m", model, "-p", "hi", "-n", "1", "--temp", "0",
                          "--gpu", "off", "-v", *extra])
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stderr.decode(errors="replace")


def _kv_mb(banner):
    m = re.search(r"kv cache\s+([0-9.]+) MB", banner)
    assert m, banner
    return float(m.group(1))


ATTN_NAMES = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]
FFN_NAMES = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]


# ------------------------------------------------------------------ tests
def test_attention_removal_is_bit_identical_to_zeroed_output_projection(
        runner_bin, parent, tmp_path):
    out = tmp_path / "attn1.gguf"
    p = _remove(runner_bin, parent, out, "attn:1")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    zeroed = tmp_path / "zeroed.gguf"
    _zero_tensor(parent, zeroed, "blk.1.attn_output.weight")
    removed = _score(runner_bin, out)
    anchor = _score(runner_bin, zeroed)
    assert removed == anchor
    assert removed != _score(runner_bin, parent)


def test_mlp_removal_is_bit_identical_to_zeroed_down_projection(
        runner_bin, parent, tmp_path):
    out = tmp_path / "mlp0.gguf"
    p = _remove(runner_bin, parent, out, "mlp:0")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    zeroed = tmp_path / "zeroed.gguf"
    _zero_tensor(parent, zeroed, "blk.0.ffn_down.weight")
    assert _score(runner_bin, out) == _score(runner_bin, zeroed)
    assert _score(runner_bin, out) != _score(runner_bin, parent)


def test_both_parts_compose(runner_bin, parent, tmp_path):
    out = tmp_path / "both.gguf"
    p = _remove(runner_bin, parent, out, "attn:1,mlp:0")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    z1 = tmp_path / "z1.gguf"
    z2 = tmp_path / "z2.gguf"
    _zero_tensor(parent, z1, "blk.1.attn_output.weight")
    _zero_tensor(z1, z2, "blk.0.ffn_down.weight")
    assert _score(runner_bin, out) == _score(runner_bin, z2)


def test_declaration_uses_the_per_layer_arrays(runner_bin, parent, tmp_path):
    out = tmp_path / "both.gguf"
    assert _remove(runner_bin, parent, out, "attn:1,mlp:0").returncode == 0
    pk, ok = _parse(parent)["kvs"], _parse(out)["kvs"]
    n_head, n_kv, n_ff = (pk["llama.attention.head_count"],
                          pk["llama.attention.head_count_kv"],
                          pk["llama.feed_forward_length"])
    assert ok["llama.attention.head_count"] == [n_head, 0]
    assert ok["llama.attention.head_count_kv"] == [n_kv, 0]
    assert ok["llama.feed_forward_length"] == [0, n_ff]
    assert ok["llama.block_count"] == 2


def test_byte_accounting_is_exact(runner_bin, parent, tmp_path):
    out = tmp_path / "attn1.gguf"
    p = _remove(runner_bin, parent, out, "attn:1")
    assert p.returncode == 0
    pi, oi = _parse(parent), _parse(out)
    dropped = {"blk.1." + n for n in ATTN_NAMES}
    assert not (dropped & set(oi["tensors"]))
    assert set(oi["tensors"]) == set(pi["tensors"]) - dropped
    # every survivor is byte-identical, and the data section is exactly the
    # survivors' aligned spans
    for name in oi["tensors"]:
        assert _payload(out, oi, name) == _payload(parent, pi, name), name
    assert oi["size"] - oi["data_start"] == sum(
        t["span"] for t in oi["tensors"].values())
    removed_payload = sum(pi["tensors"][n]["span"] for n in dropped)
    assert oi["size"] < pi["size"] - removed_payload + 4096  # header may grow (arrays)
    # the writer states what it removed, in bytes, and that number is the
    # parent's own payload accounting for those tensors
    m = re.search(r"remove-sublayer: attn:1 .*?(\d+) bytes of tensor data",
                  p.stderr.decode(errors="replace"))
    assert m, p.stderr.decode(errors="replace")
    assert int(m.group(1)) == removed_payload


def test_removed_attention_reserves_no_kv_rows(runner_bin, parent, tmp_path):
    out = tmp_path / "attn1.gguf"
    assert _remove(runner_bin, parent, out, "attn:1").returncode == 0
    ctx = ["-c", "65536"]
    parent_mb = _kv_mb(_load_banner(runner_bin, parent, ctx))
    removed_mb = _kv_mb(_load_banner(runner_bin, out, ctx))
    # two identical layers, one attention removed: exactly half
    assert abs(removed_mb - parent_mb / 2) < 0.06, (parent_mb, removed_mb)
    banner = _load_banner(runner_bin, out)
    assert "removed" in banner and "attn:1" in banner


def test_requantizing_a_removed_file_keeps_the_declaration(runner_bin, parent, tmp_path):
    a = tmp_path / "attn1.gguf"
    assert _remove(runner_bin, parent, a, "attn:1").returncode == 0
    b = tmp_path / "attn1-q8.gguf"
    p = _run(runner_bin, ["-m", a, "--quantize", b, "--quant", "q8_0"])
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    assert _parse(b)["kvs"]["llama.attention.head_count"] == [4, 0]
    assert "blk.1.attn_q.weight" not in _parse(b)["tensors"]
    _load_banner(runner_bin, b)


def test_undeclared_missing_tensor_is_still_an_error(runner_bin, parent, tmp_path):
    bad = tmp_path / "undeclared.gguf"
    _drop_directory_entry(parent, bad, "blk.1.attn_q.weight")
    p = _run(runner_bin, ["-m", bad, "-p", "hi", "-n", "1", "--gpu", "off"])
    assert p.returncode != 0
    assert b"missing tensor blk.1.attn_q.weight" in p.stderr


def test_declared_absent_but_present_is_refused(runner_bin, declared_but_present):
    p = _run(runner_bin, ["-m", declared_but_present, "-p", "hi", "-n", "1",
                          "--gpu", "off"])
    assert p.returncode != 0
    err = p.stderr.decode(errors="replace")
    # either contradiction is a refusal; the loader reports the first block
    assert "declared" in err and "but the file carries blk." in err, err


@pytest.mark.parametrize("spec,needle", [
    ("attn:7", "block 7"),
    ("mlp:2", "block 2"),
    ("attn:x", "remove-sublayer"),
    ("ffn:0", "remove-sublayer"),
    ("attn:1,attn:1", "twice"),
])
def test_bad_specs_are_refused(runner_bin, parent, tmp_path, spec, needle):
    p = _remove(runner_bin, parent, tmp_path / "x.gguf", spec)
    assert p.returncode != 0
    assert needle in p.stderr.decode(errors="replace")
    assert not (tmp_path / "x.gguf").exists()


def test_removing_an_already_removed_part_is_refused(runner_bin, parent, tmp_path):
    a = tmp_path / "a.gguf"
    assert _remove(runner_bin, parent, a, "attn:1").returncode == 0
    p = _remove(runner_bin, a, tmp_path / "b.gguf", "attn:1")
    assert p.returncode != 0
    assert "already" in p.stderr.decode(errors="replace")


def test_moe_mlp_and_hybrid_blocks_are_refused_by_name(runner_bin, tmp_path):
    base = tmp_path / "moe"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-moe.py", str(base)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    moe = pathlib.Path(f"{base}.pruneprobe.gguf")
    p = _remove(runner_bin, moe, tmp_path / "x.gguf", "mlp:0")
    assert p.returncode != 0
    assert "MoE" in p.stderr.decode(errors="replace")
    pfx = tmp_path / "nemo"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-nemotron.py", str(pfx)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    nemo = next(tmp_path.glob("nemo*.gguf"))
    p = _remove(runner_bin, nemo, tmp_path / "y.gguf", "attn:0")
    assert p.returncode != 0
    assert "hybrid" in p.stderr.decode(errors="replace")


def test_requires_quantize_out(runner_bin, parent):
    p = _run(runner_bin, ["-m", parent, "--remove-sublayer", "attn:1",
                          "-p", "hi", "-n", "1"])
    assert p.returncode != 0
    assert b"--remove-sublayer requires --quantize" in p.stderr


def test_adapters_and_training_refuse_a_removed_model(runner_bin, parent, tmp_path):
    a = tmp_path / "a.gguf"
    assert _remove(runner_bin, parent, a, "attn:1").returncode == 0
    fx = tmp_path / "fx"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-lora.py",
                    str(parent), str(fx)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    p = _run(runner_bin, ["-m", a, "--score", "-p", PROMPT, "--gpu", "off",
                          "--lora", fx.with_suffix(".adapter.gguf")])
    assert p.returncode != 0
    assert b"removed" in p.stderr
    corpus = tmp_path / "c.txt"
    corpus.write_text(PROMPT + "\n")
    p = _run(runner_bin, ["-m", a, "--train", corpus, "--train-steps", "1",
                          "--train-out", tmp_path / "o.gguf", "--gpu", "off"])
    assert p.returncode != 0
    assert b"removed" in p.stderr


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                    reason="needs a GPU backend to refuse")
def test_gpu_path_is_refused_for_now(runner_bin, parent, tmp_path):
    a = tmp_path / "a.gguf"
    assert _remove(runner_bin, parent, a, "attn:1").returncode == 0
    p = _run(runner_bin, ["-m", a, "-p", "hi", "-n", "1", "--gpu", "auto"])
    assert p.returncode != 0
    assert b"--gpu off" in p.stderr


# ---------------------------------------------------- the gemma-4 file shape
@pytest.fixture(scope="module")
def gemma4_hetero(tmp_path_factory):
    """gemma4 with the real 26B/12B/31B header shape: head_count scalar,
    head_count_kv already a per-layer ARRAY, sliding pattern, V-less full
    layers (tied V), sandwich norms, and the two E-series keys present with
    value 0 (every gemma-4 export carries them; only a non-zero value makes
    a file E-series)."""
    out = tmp_path_factory.mktemp("rm-g4") / "g4.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py",
                    "--gemma4-hetero", str(out)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return out


@pytest.mark.parametrize("layer", [1, 2])  # 1 slides (has V); 2 is full and V-less
def test_gemma4_shape_removal_matches_zeroed_output(runner_bin, gemma4_hetero,
                                                    tmp_path, layer):
    kv = _parse(gemma4_hetero)["kvs"]
    assert isinstance(kv["gemma4.attention.head_count_kv"], list)
    assert kv.get("gemma4.embedding_length_per_layer_input") == 0
    out = tmp_path / f"g4-attn{layer}.gguf"
    p = _remove(runner_bin, gemma4_hetero, out, f"attn:{layer}")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    zeroed = tmp_path / "z.gguf"
    _zero_tensor(gemma4_hetero, zeroed, f"blk.{layer}.attn_output.weight")
    assert _score(runner_bin, out) == _score(runner_bin, zeroed)
    assert _score(runner_bin, out) != _score(runner_bin, gemma4_hetero)
    ok = _parse(out)["kvs"]
    hc = ok["gemma4.attention.head_count"]
    assert isinstance(hc, list) and hc[layer] == 0 and all(hc[i] == kv["gemma4.attention.head_count"] for i in range(len(hc)) if i != layer)
    hk = ok["gemma4.attention.head_count_kv"]
    assert hk[layer] == 0
    assert [v for i, v in enumerate(hk) if i != layer] == \
        [v for i, v in enumerate(kv["gemma4.attention.head_count_kv"]) if i != layer]
    assert f"blk.{layer}.attn_q.weight" not in _parse(out)["tensors"]
