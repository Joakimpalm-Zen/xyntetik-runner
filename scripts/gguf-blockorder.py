#!/usr/bin/env python3
"""Put a GGUF's tensors in block order, and prove nothing else changed.

A partial GPU offload that uploads one contiguous file prefix (Runner v0.5.6
and earlier, before main 9b825fc) reads from byte 0 through the farthest
tensor any offloaded layer owns, and that prefix always covers
`token_embd.weight` too (v0.5.6 requires it in the upload even though the
embedding is read on the host). So a file that stores `output.weight` at its
head, or interleaves blocks, or stores `token_embd.weight` last, uploads far
more than the split needs. This rewrites the tensor data as
`token_embd.weight` first, then blk.0 .. blk.N (each block contiguous, in its
original within-block order), then every other tensor in its original order,
then `output_norm.weight` and `output.weight` last; `--order-from REF` copies
REF's exact name order instead. The prefix the report prints applies v0.5.6's
rule exactly, token_embd included.

The header and every metadata field are copied byte for byte (the tensor
directory keeps its size, so the data offset and alignment are unchanged);
only the tensor offsets and the data order move. Every tensor's raw bytes
are hashed on the way through and the output is re-read and re-hashed
before it replaces `OUT.tmp` (`--no-verify` skips the re-read). The JSON
report carries the order before and after, the per-tensor sha256, the
prefix needed to reach the end of block K before and after, and the
whole-file hashes. Standard library only; the design and the first
evidence come from the lab's normaliser on Kvist-14B (2026-09-23).

Usage: gguf-blockorder.py IN.gguf OUT.gguf --json REPORT.json
       [--order-from REF.gguf] [--prefix-blocks K] [--check-only] [--no-verify]
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

GGUF_MAGIC = 0x46554747
# ggml type -> (block width, bytes per block)
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
         8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
         17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
         22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
         27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 39: (32, 17)}
CHUNK = 1 << 20


def row_bytes(ttype, ne0):
    if ttype not in BLOCK:
        raise SystemExit(f"unsupported ggml type {ttype}")
    blk, size = BLOCK[ttype]
    if ne0 % blk:
        raise SystemExit(f"row width {ne0} is not a multiple of type {ttype}'s block {blk}")
    return ne0 // blk * size


def _u32(f): return struct.unpack("<I", f.read(4))[0]
def _u64(f): return struct.unpack("<Q", f.read(8))[0]
def _str(f):
    n = _u64(f)
    return f.read(n).decode("utf-8", "replace")


_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_value(f, t):
    if t == 8:
        _str(f)
    elif t == 9:
        et = _u32(f); n = _u64(f)
        for _ in range(n):
            _skip_value(f, et)
    else:
        f.read(_SCALAR[t])


def read_header(path):
    """(kv_region_end, alignment, tensors, data_start, header_bytes) where
    tensors are dicts name/ne/type/offset/nbytes in directory order."""
    with open(path, "rb") as f:
        if _u32(f) != GGUF_MAGIC:
            raise SystemExit(f"{path}: not a GGUF file")
        version = _u32(f)
        if version != 3:
            raise SystemExit(f"{path}: GGUF version {version}, only 3 is handled")
        n_tensors = _u64(f); n_kv = _u64(f)
        alignment = 32
        for _ in range(n_kv):
            key = _str(f); t = _u32(f)
            if key == "general.alignment" and t == 4:
                alignment = _u32(f)
            else:
                _skip_value(f, t)
        kv_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = _str(f); nd = _u32(f)
            ne = [_u64(f) for _ in range(nd)]
            ttype = _u32(f); off = _u64(f)
            rows = 1
            for d in ne[1:]:
                rows *= d
            tensors.append({"name": name, "ne": ne, "type": ttype, "offset": off,
                            "nbytes": row_bytes(ttype, ne[0]) * rows})
        dir_end = f.tell()
        data_start = (dir_end + alignment - 1) // alignment * alignment
        f.seek(0)
        header = f.read(dir_end)
    return kv_end, alignment, tensors, data_start, header


def blk_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def canonical_order(names):
    blocks = {}
    rest = []
    for n in names:
        b = blk_of(n)
        if b is None:
            rest.append(n)
        else:
            blocks.setdefault(b, []).append(n)
    head = [n for n in ("token_embd.weight",) if n in rest]
    tail = [n for n in ("output_norm.weight", "output.weight") if n in rest]
    rest = [n for n in rest if n not in head and n not in tail]
    out = list(head)
    for b in sorted(blocks):
        out.extend(blocks[b])
    return out + rest + tail


def order_summary(names):
    runs = []
    for n in names:
        b = blk_of(n)
        key = str(b) if b is not None else n.replace(".weight", "")
        if runs and runs[-1][0] == key:
            runs[-1][1] += 1
        else:
            runs.append([key, 1])
    return " ".join(k if c == 1 or blk_of(k + ".") is None else f"{k}x{c}" for k, c in runs)


def prefix_to_blocks(tensors, data_start, last_block):
    """Bytes v0.5.6 uploads for a split of blocks 0..last_block: from byte 0
    through the farthest of those blocks' tensors and through token_embd."""
    end = 0
    for t in tensors:
        b = blk_of(t["name"])
        if (b is not None and b <= last_block) or t["name"] == "token_embd.weight":
            end = max(end, data_start + t["offset"] + t["nbytes"])
    return end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--json", required=True)
    ap.add_argument("--order-from")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--prefix-blocks", type=int, default=43)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    kv_end, alignment, tensors, data_start, header = read_header(a.inp)
    by_name = {t["name"]: t for t in tensors}
    names_in = [t["name"] for t in tensors]
    if a.order_from:
        ref_names = [t["name"] for t in read_header(a.order_from)[2]]
        missing = [n for n in names_in if n not in ref_names]
        if missing or len(ref_names) != len(names_in):
            raise SystemExit("--order-from: the reference does not name exactly this file's tensors")
        target = ref_names
    else:
        target = canonical_order(names_in)
    report = {"input": a.inp, "output": a.out, "order_from": a.order_from,
              "alignment": alignment, "data_start": data_start,
              "order_before": order_summary(names_in), "order_after": order_summary(target),
              "prefix_blocks": a.prefix_blocks,
              "prefix_before": prefix_to_blocks(tensors, data_start, a.prefix_blocks),
              "already_in_order": names_in == target}
    if a.check_only or not a.out:
        json.dump(report, open(a.json, "w"), indent=2)
        print(json.dumps({k: report[k] for k in ("order_before", "order_after", "prefix_before", "already_in_order")}))
        return 0

    # new directory: same names/dims/types, offsets recomputed in target order
    new_tensors = []
    off = 0
    for n in target:
        t = dict(by_name[n]); t["offset"] = off
        new_tensors.append(t)
        off = (off + t["nbytes"] + alignment - 1) // alignment * alignment
    report["prefix_after"] = prefix_to_blocks(new_tensors, data_start, a.prefix_blocks)

    tmp = a.out + ".tmp"
    hashes = {}
    with open(a.inp, "rb") as src, open(tmp, "wb") as dst:
        # header + KV region byte for byte, then the directory in the new order
        dst.write(header[:kv_end])
        for t in new_tensors:
            nb = t["name"].encode("utf-8")
            dst.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(t["ne"])))
            for d in t["ne"]:
                dst.write(struct.pack("<Q", d))
            dst.write(struct.pack("<IQ", t["type"], t["offset"]))
        dir_end = dst.tell()
        if dir_end != len(header):
            raise SystemExit("internal: the tensor directory changed size")
        dst.write(b"\0" * (data_start - dir_end))
        for t in new_tensors:
            pos = dst.tell()
            want = data_start + t["offset"]
            if pos > want:
                raise SystemExit("internal: offset accounting went backwards")
            dst.write(b"\0" * (want - pos))
            src.seek(data_start + by_name[t["name"]]["offset"])
            h = hashlib.sha256()
            left = t["nbytes"]
            while left:
                buf = src.read(min(CHUNK, left))
                if not buf:
                    raise SystemExit(f"short read on {t['name']}")
                h.update(buf); dst.write(buf); left -= len(buf)
            hashes[t["name"]] = h.hexdigest()
        out_size = dst.tell()
    report["tensors"] = [{"name": t["name"], "type": t["type"], "ne": t["ne"],
                          "nbytes": t["nbytes"], "sha256": hashes[t["name"]]} for t in new_tensors]
    report["size_in"] = os.path.getsize(a.inp)
    report["size_out"] = out_size

    ok = True
    if not a.no_verify:
        kv2, al2, t2, ds2, header2 = read_header(tmp)
        if header2[:kv2] != header[:kv_end] or ds2 != data_start or al2 != alignment:
            ok = False; report["verify_error"] = "header or metadata region differs"
        seen = {}
        with open(tmp, "rb") as f:
            for t in t2:
                f.seek(ds2 + t["offset"])
                h = hashlib.sha256(); left = t["nbytes"]
                while left:
                    buf = f.read(min(CHUNK, left)); h.update(buf); left -= len(buf)
                seen[t["name"]] = h.hexdigest()
        if [t["name"] for t in t2] != target or seen != hashes:
            ok = False; report["verify_error"] = "re-read tensor order or hashes differ"
        report["verified"] = ok
    if ok:
        os.replace(tmp, a.out)
    else:
        os.remove(tmp)
    json.dump(report, open(a.json, "w"), indent=2)
    print(json.dumps({"order_before": report["order_before"], "order_after": report["order_after"],
                      "prefix_before": report["prefix_before"], "prefix_after": report["prefix_after"],
                      "size": out_size, "verified": report.get("verified"), "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
