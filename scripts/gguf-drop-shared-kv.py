#!/usr/bin/env python3
"""Rewrite a Gemma-4 E-series GGUF in the compact shared-KV form.

    gguf-drop-shared-kv.py in.gguf out.gguf

A layer at or past ``block_count - attention.shared_kv_layers`` computes no K
and no V: it attends over the cache an earlier layer filled. Its
``attn_k.weight``, ``attn_v.weight`` and ``attn_k_norm.weight`` are therefore
unreachable, and every quantized Gemma-4 E-series export published since the
BF16 one leaves them out (E4B: 666 tensors instead of 720). This turns a
full-form export into that compact form so the two can be compared directly.

The layer set is derived from the file's OWN metadata, never from a range
typed on the command line: a hand-typed range that happened to be wrong would
delete a KV-owning layer's projections and produce a file that is quietly a
different model. Nothing else changes - metadata is copied through verbatim,
surviving tensors keep their exact on-disk bytes, and no renumbering happens -
so a correct engine must score the two files identically, to the bit.

Pure stdlib, and streams the tensor data rather than reading the file into
memory: these files are several GB.
"""
import os
import struct
import sys

T_STR, T_ARR = 8, 9
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
          7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
COPY_CHUNK = 8 << 20
DROPPED_PARTS = ("attn_k.weight", "attn_v.weight", "attn_k_norm.weight")


class Reader:
    """Sequential reader over the header region, keeping the raw bytes of each
    value so metadata can be copied through without being re-encoded."""

    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            sys.exit("truncated GGUF")
        return b

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def st(self): return self.take(self.u64())

    def value(self, t):
        """(raw bytes, parsed scalar or None)."""
        if t == T_STR:
            raw = self.st()
            return struct.pack("<Q", len(raw)) + raw, raw.decode("utf-8", "replace")
        if t == T_ARR:
            at, n = self.u32(), self.u64()
            head = struct.pack("<IQ", at, n)
            if at == T_STR:
                parts = []
                for _ in range(n):
                    b = self.st()
                    parts.append(struct.pack("<Q", len(b)) + b)
                return head + b"".join(parts), None
            if at == T_ARR:
                sys.exit("nested arrays are not supported")
            body = self.take(SIZES[at] * n)
            return head + body, None
        raw = self.take(SIZES[t])
        return raw, struct.unpack(SCALAR[t], raw)[0]


def s(text):
    b = text if isinstance(text, bytes) else text.encode()
    return struct.pack("<Q", len(b)) + b


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst):
        sys.exit(f"refusing to overwrite {dst}")

    with open(src, "rb") as f:
        r = Reader(f)
        if r.take(4) != b"GGUF":
            sys.exit("not a GGUF file")
        version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()

        align, arch, meta = 32, None, {}
        kvs = []
        for _ in range(n_kv):
            key = r.st().decode()
            t = r.u32()
            raw, parsed = r.value(t)
            if key == "general.alignment":
                align = parsed
            if key == "general.architecture":
                arch = parsed
            meta[key] = parsed
            kvs.append(s(key) + struct.pack("<I", t) + raw)
        if arch is None:
            sys.exit("no general.architecture")

        n_layer = meta.get(f"{arch}.block_count")
        shared = meta.get(f"{arch}.attention.shared_kv_layers")
        if not n_layer or not shared:
            sys.exit(f"{src}: not a shared-KV export "
                     f"({arch}.attention.shared_kv_layers is absent or 0)")
        kv_from_start = n_layer - shared
        if kv_from_start < 2:
            sys.exit(f"{src}: shared_kv_layers={shared} leaves no KV-owning "
                     "layers")
        drop_names = {f"blk.{i}.{part}"
                      for i in range(kv_from_start, n_layer)
                      for part in DROPPED_PARTS}

        table = []
        for _ in range(n_tensors):
            name = r.st().decode()
            nd = r.u32()
            ne = [r.u64() for _ in range(nd)]
            tt, off = r.u32(), r.u64()
            table.append((name, nd, ne, tt, off))
        data_start = (f.tell() + align - 1) & ~(align - 1)
        total = os.path.getsize(src) - data_start

    # A tensor's size is the gap to the next blob, so it can only be read off
    # the sorted offsets; the last blob runs to the end of the file.
    by_off = sorted(table, key=lambda e: e[4])
    size = {}
    for i, (name, _nd, _ne, _tt, off) in enumerate(by_off):
        size[name] = (by_off[i + 1][4] if i + 1 < len(by_off) else total) - off

    kept = [e for e in table if e[0] not in drop_names]
    missing = drop_names - {e[0] for e in table}
    if missing:
        print(f"note: {len(missing)} of the {len(drop_names)} shared-KV "
              "tensors were already absent")
    if len(kept) == len(table):
        sys.exit(f"{src} is already in the compact form; nothing to do")

    head = struct.pack("<4sIQQ", b"GGUF", version, len(kept), len(kvs))
    head += b"".join(kvs)
    entries, blobs, off_new = [], [], 0
    for name, nd, ne, tt, off in kept:
        entries.append(s(name) + struct.pack("<I", nd) +
                       b"".join(struct.pack("<Q", d) for d in ne) +
                       struct.pack("<IQ", tt, off_new))
        blobs.append((data_start + off, size[name]))
        off_new = (off_new + size[name] + align - 1) & ~(align - 1)
    head += b"".join(entries)

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(head + b"\0" * ((-len(head)) % align))
        for off, n in blobs:
            fin.seek(off)
            left = n
            while left:
                chunk = fin.read(min(COPY_CHUNK, left))
                if not chunk:
                    sys.exit("truncated GGUF: tensor data ended early")
                fout.write(chunk)
                left -= len(chunk)
            fout.write(b"\0" * ((-n) % align))
    print(f"wrote {dst}: {len(kept)}/{len(table)} tensors "
          f"({len(table) - len(kept)} dropped on layers "
          f"{kv_from_start}..{n_layer - 1})")


if __name__ == "__main__":
    main()
