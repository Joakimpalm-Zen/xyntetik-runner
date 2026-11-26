#!/usr/bin/env python3
"""R4.48.8.6 cross-check harness: Runner vs llama.cpp (IQ3_XXS bit-for-bit & NVFP4 interop).

Usage on a host with llama-quantize:
    python3 scripts/verify-device-writers-llama.py \
        --runner ./runner \
        --llama-quantize /path/to/llama-quantize \
        [--llama-cpp-py /path/to/llama.cpp/gguf-py]

What it verifies:
  1. IQ3_XXS bit-for-bit: generates a tiny valid Llama F32 GGUF (256-wide rows) and
     a GGUF imatrix, runs both `runner --quantize ... --quant iq3_xxs` and
     `llama-quantize ... IQ3_XXS` (both unweighted and with --imatrix), and asserts
     0 differing bytes on every tensor written as IQ3_XXS by both tools.
  2. NVFP4 127-code & interop: runs `runner --quantize ... --quant nvfp4` on a fixture
     covering all 127 finite UE4M3 codes (0x00..0x7E, never 0x7F) and on an Inkling-shaped
     N(0, 0.03^2) BF16 matrix; decodes via gguf-py (when --llama-cpp-py is given) and the
     exact dequantize_row_nvfp4 formula, asserting 0 differing floats and reporting
     two-level vs single-level RMSE.
"""
import argparse
import json
import math
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

T_F32, T_BF16, T_IQ3_XXS, T_NVFP4 = 0, 30, 18, 40
U32, I32, F32, STR, ARR = 4, 5, 6, 8, 9
_TYPE_SIZE = {T_F32: (4, 1), T_BF16: (2, 1), T_IQ3_XXS: (98, 256), T_NVFP4: (36, 64)}
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, typ: int, val) -> bytes:
    out = _s(key) + struct.pack("<I", typ)
    if typ == U32:
        return out + struct.pack("<I", val)
    if typ == I32:
        return out + struct.pack("<i", val)
    if typ == F32:
        return out + struct.pack("<f", val)
    if typ == STR:
        return out + _s(val)
    et, items = val
    out += struct.pack("<IQ", et, len(items))
    for it in items:
        if et == STR:
            out += _s(it)
        elif et == I32:
            out += struct.pack("<i", it)
        elif et == F32:
            out += struct.pack("<f", it)
        else:
            raise ValueError(et)
    return out


def write_gguf(path: Path, kvs, tensors) -> Path:
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, typ, v in kvs:
        head += _kv(k, typ, v)
    off, infos, blobs = 0, b"", []
    for name, dims, typ, data in tensors:
        infos += (_s(name) + struct.pack("<I", len(dims))
                  + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off))
        pad = (-len(data)) % 32
        blobs.append(data + b"\0" * pad)
        off += len(data) + pad
    body = head + infos
    body += b"\0" * ((-len(body)) % 32)
    path.write_bytes(body + b"".join(blobs))
    return path


def read_gguf_tensors(path: Path):
    b = path.read_bytes()
    o = [4]

    def take(fmt):
        v = struct.unpack_from("<" + fmt, b, o[0])
        o[0] += struct.calcsize("<" + fmt)
        return v[0]

    def text():
        n = take("Q")
        o[0] += n
        return b[o[0] - n:o[0]].decode("utf-8")

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return text()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(scalar[t])

    assert b[:4] == b"GGUF"
    take("I")
    nt, nkv = take("Q"), take("Q")
    kv = {}
    for _ in range(nkv):
        k = text()
        kv[k] = value(take("I"))
    infos = []
    for _ in range(nt):
        name = text()
        dims = [take("Q") for _ in range(take("I"))]
        infos.append((name, dims, take("I"), take("Q")))
    al = kv.get("general.alignment", 32)
    start = (o[0] + al - 1) // al * al
    out = {}
    for name, dims, typ, off in infos:
        n = 1
        for d in dims:
            n *= d
        ts, bs = _TYPE_SIZE.get(typ, (0, 1))
        out[name] = (dims, typ, b[start + off:start + off + (n // bs) * ts] if ts else None)
    return kv, out


def ue4m3_to_f32(x: int) -> float:
    if x == 0 or x == 0x7F:
        return 0.0
    exp, man = (x >> 3) & 0xF, x & 7
    return math.ldexp(float(man), -9) if exp == 0 else math.ldexp(1.0 + man / 8.0, exp - 7)


def decode_nvfp4_ref(raw: bytes, n_el: int):
    out = []
    for blk in range(n_el // 64):
        off = blk * 36
        scales = raw[off:off + 4]
        qs = raw[off + 4:off + 36]
        for s in range(4):
            d = ue4m3_to_f32(scales[s])
            sub = qs[s * 8:(s + 1) * 8]
            out.extend(E2M1[q & 0xF] * d for q in sub)
            out.extend(E2M1[q >> 4] * d for q in sub)
    return out


def make_llama_fixture(path: Path, imx_path: Path):
    """Minimal valid Llama GGUF (H=256, FF=512, 1 layer, vocab=256) + matching GGUF imatrix."""
    h, ff, vocab = 256, 512, 256
    kvs = [
        ("general.architecture", STR, "llama"),
        ("general.name", STR, "iq3-xxs-crosscheck"),
        ("general.alignment", U32, 32),
        ("llama.block_count", U32, 1),
        ("llama.context_length", U32, 512),
        ("llama.embedding_length", U32, h),
        ("llama.feed_forward_length", U32, ff),
        ("llama.attention.head_count", U32, 4),
        ("llama.attention.head_count_kv", U32, 4),
        ("llama.attention.layer_norm_rms_epsilon", F32, 1e-5),
        ("llama.vocab_size", U32, vocab),
        ("tokenizer.ggml.model", STR, "llama"),
        ("tokenizer.ggml.tokens", ARR, (STR, [f"t{i}" for i in range(vocab)])),
        ("tokenizer.ggml.scores", ARR, (F32, [0.0] * vocab)),
        ("tokenizer.ggml.token_type", ARR, (I32, [1] * vocab)),
    ]
    rng = random.Random(20261126)

    def mat(cols, rows, std=0.2, zero_row=None):
        vals = [rng.gauss(0.0, std) for _ in range(cols * rows)]
        if zero_row is not None:
            for c in range(cols):
                vals[zero_row * cols + c] = 0.0
        return struct.pack(f"<{cols * rows}f", *vals)

    ones = struct.pack(f"<{h}f", *([1.0] * h))
    tensors = [
        ("token_embd.weight", [h, vocab], T_F32, mat(h, vocab, 0.1)),
        ("output_norm.weight", [h], T_F32, ones),
        ("output.weight", [h, vocab], T_F32, mat(h, vocab, 0.1)),
        ("blk.0.attn_norm.weight", [h], T_F32, ones),
        ("blk.0.ffn_norm.weight", [h], T_F32, ones),
        ("blk.0.attn_q.weight", [h, h], T_F32, mat(h, h, 0.2)),
        ("blk.0.attn_k.weight", [h, h], T_F32, mat(h, h, 0.2)),
        ("blk.0.attn_v.weight", [h, h], T_F32, mat(h, h, 0.2)),
        ("blk.0.attn_output.weight", [h, h], T_F32, mat(h, h, 0.2)),
        ("blk.0.ffn_gate.weight", [h, ff], T_F32, mat(h, ff, 0.25, zero_row=1)),
        ("blk.0.ffn_up.weight", [h, ff], T_F32, mat(h, ff, 0.25)),
        ("blk.0.ffn_down.weight", [ff, h], T_F32, mat(ff, h, 0.25, zero_row=2)),
    ]
    write_gguf(path, kvs, tensors)

    imx_tensors = []
    for name, dims, _, _ in tensors:
        if not name.startswith("blk.0.") or "norm" in name:
            continue
        cols = dims[0]
        w = [0.5 + 10.0 * rng.random() for _ in range(cols)]
        imx_tensors.append((f"{name}.in_sum2", [cols], T_F32, struct.pack(f"<{cols}f", *w)))
        imx_tensors.append((f"{name}.counts", [1], T_F32, struct.pack("<f", 4.0)))
    write_gguf(imx_path, [("general.architecture", STR, "imatrix"), ("general.alignment", U32, 32)], imx_tensors)


def check_iq3_xxs(runner: Path, llama_quantize: Path, tmp: Path):
    src = tmp / "llama-f32.gguf"
    imx = tmp / "llama-imx.gguf"
    make_llama_fixture(src, imx)
    summary = {}
    for label, use_imx in (("unweighted", False), ("imatrix", True)):
        r_out = tmp / f"runner-{label}.gguf"
        l_out = tmp / f"llama-{label}.gguf"
        r_cmd = [str(runner), "-m", str(src), "--quantize", str(r_out), "--quant", "iq3_xxs"]
        l_cmd = [str(llama_quantize)]
        if use_imx:
            r_cmd += ["--imatrix", str(imx)]
            l_cmd += ["--imatrix", str(imx)]
        l_cmd += [str(src), str(l_out), "IQ3_XXS"]
        subprocess.run(r_cmd, check=True, capture_output=True)
        subprocess.run(l_cmd, check=True, capture_output=True)

        _, rt = read_gguf_tensors(r_out)
        _, lt = read_gguf_tensors(l_out)
        compared, total_bytes, diff_bytes = 0, 0, 0
        for name in sorted(lt):
            if lt[name][1] == T_IQ3_XXS and rt[name][1] == T_IQ3_XXS:
                rb, lb = rt[name][2], lt[name][2]
                assert len(rb) == len(lb), (name, len(rb), len(lb))
                diffs = sum(a != b for a, b in zip(rb, lb))
                compared += 1
                total_bytes += len(rb)
                diff_bytes += diffs
                if diffs:
                    first = next(i for i, (a, b) in enumerate(zip(rb, lb)) if a != b)
                    raise AssertionError(
                        f"{label} {name}: {diffs}/{len(rb)} bytes differ "
                        f"(first at block {first // 98} byte {first % 98}: "
                        f"runner=0x{rb[first]:02x} llama=0x{lb[first]:02x})"
                    )
        assert compared > 0, "no shared IQ3_XXS tensors compared"
        summary[label] = {"tensors": compared, "bytes": total_bytes, "diff_bytes": diff_bytes}
    return summary


def check_nvfp4(runner: Path, llama_cpp_py: Path | None, tmp: Path):
    # 1. All 127 UE4M3 codes (0x00..0x7E)
    cols, rows = 256, 8
    f_target = 0.01
    vals127 = [0.0] * (cols * rows)
    for u in range(127):
        du = ue4m3_to_f32(u)
        vals127[u * 16 + 0] = 6.0 * du * f_target
        vals127[u * 16 + 1] = -3.0 * du * f_target
        vals127[u * 16 + 8] = 1.5 * du * f_target
    ones = struct.pack(f"<{cols}f", *([1.0] * cols))
    src127 = write_gguf(
        tmp / "nv127.gguf",
        [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)],
        [("output_norm.weight", [cols], T_F32, ones),
         ("blk.0.ffn_gate.weight", [cols, rows], T_F32, struct.pack(f"<{cols * rows}f", *vals127))],
    )
    out127 = tmp / "nv127-out.gguf"
    subprocess.run([str(runner), "-m", str(src127), "--quantize", str(out127), "--quant", "nvfp4"],
                   check=True, capture_output=True)
    _, got127 = read_gguf_tensors(out127)
    raw127 = got127["blk.0.ffn_gate.weight"][2]
    (f_got,) = struct.unpack("<f", got127["blk.0.ffn_gate.scale"][2])
    codes = [raw127[blk * 36 + s] for blk in range(len(raw127) // 36) for s in range(4)]
    assert 0x7F not in codes and codes[:127] == list(range(127))
    dec127 = decode_nvfp4_ref(raw127, cols * rows)
    gguf_py_max_diff = None
    if llama_cpp_py is not None:
        sys.path.insert(0, str(llama_cpp_py))
        import gguf
        from gguf import quants
        reader = gguf.GGUFReader(str(out127))
        t = next(x for x in reader.tensors if x.name == "blk.0.ffn_gate.weight")
        dq = quants.dequantize(t.data, t.tensor_type).reshape(-1).tolist()
        gguf_py_max_diff = max(abs(a - b) for a, b in zip(dec127, dq))
        assert gguf_py_max_diff == 0.0, f"gguf-py decode diff {gguf_py_max_diff}"

    # 2. Inkling-shaped BF16 matrix N(0, 0.03^2): two-level vs single-level RMSE
    rng = random.Random(20261126)
    n_el = 256 * 512
    f32_raw = struct.pack(f"<{n_el}f", *[rng.gauss(0.0, 0.03) for _ in range(n_el)])
    bf16_raw = bytearray()
    bf16_f32 = []
    for (u,) in struct.iter_unpack("<I", f32_raw):
        u += 0x7FFF + ((u >> 16) & 1)
        h16 = (u >> 16) & 0xFFFF
        bf16_raw += struct.pack("<H", h16)
        bf16_f32.append(struct.unpack("<f", struct.pack("<I", h16 << 16))[0])
    src_bf = write_gguf(
        tmp / "ink-bf16.gguf",
        [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)],
        [("output_norm.weight", [256], T_F32, ones),
         ("blk.0.ffn_gate.weight", [256, 512], T_BF16, bytes(bf16_raw))],
    )
    out_bf = tmp / "ink-nvfp4.gguf"
    subprocess.run([str(runner), "-m", str(src_bf), "--quantize", str(out_bf), "--quant", "nvfp4"],
                   check=True, capture_output=True)
    _, got_bf = read_gguf_tensors(out_bf)
    (f_bf,) = struct.unpack("<f", got_bf["blk.0.ffn_gate.scale"][2])
    dec_bf = decode_nvfp4_ref(got_bf["blk.0.ffn_gate.weight"][2], n_el)
    sq_ref = sum(v * v for v in bf16_f32)
    sq_two = sum((d * f_bf - v) ** 2 for d, v in zip(dec_bf, bf16_f32))
    two_rmse = math.sqrt(sq_two / sq_ref)

    sq_one = 0.0
    for off in range(0, n_el, 16):
        sub = bf16_f32[off:off + 16]
        d_id = max(abs(v) for v in sub) / 6.0
        best_u = min(range(127), key=lambda u: abs(ue4m3_to_f32(u) - d_id))
        d1 = ue4m3_to_f32(best_u)
        for v in sub:
            qv = min((lvl * d1 for lvl in E2M1), key=lambda c: abs(c - v))
            sq_one += (qv - v) ** 2
    one_rmse = math.sqrt(sq_one / sq_ref)
    return {
        "ue4m3_127_codes_exact": True,
        "gguf_py_decode_max_diff": gguf_py_max_diff,
        "inkling_bf16_two_level_rel_rmse": round(two_rmse, 6),
        "inkling_bf16_single_level_rel_rmse": round(one_rmse, 6),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--llama-quantize", type=Path, help="llama-quantize binary for IQ3_XXS bit-for-bit check")
    ap.add_argument("--llama-cpp-py", type=Path, help="Optional path to llama.cpp/gguf-py for NVFP4 decode interop")
    args = ap.parse_args()
    runner = args.runner.resolve()
    llama_q = args.llama_quantize.resolve() if args.llama_quantize else None
    llama_py = args.llama_cpp_py.resolve() if args.llama_cpp_py else None

    out = {}
    with tempfile.TemporaryDirectory(prefix="r48886-verify-") as td:
        tmp = Path(td)
        if llama_q:
            out["iq3_xxs_vs_llama_quantize"] = check_iq3_xxs(runner, llama_q, tmp)
        out["nvfp4_interop_and_rmse"] = check_nvfp4(runner, llama_py, tmp)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
