#!/usr/bin/env python3
"""Validate a GGUF's NVFP4 layout against the file's own structure.

This exists because runner shipped NVFP4 support read from a specification with
no model of that type available to test against, and the unit test written
alongside it could not detect the error: its "reference" decode was a
transcription of the implementation, so it proved the implementation agreed
with itself. A DGX Spark user's model loaded cleanly and decoded noise.

The checks here need NO reference decoder and NO known-good output. They test
our reading of the format against properties the file must have if the reading
is right, which is the external anchor the unit test lacks (AGENTS.md, "every
gate needs one absolute anchor").

    python3 scripts/nvfp4-probe.py model.gguf

Three checks:

1. SCALE SATURATION. A block quantizer picks each sub-block's scale so the
   largest weight in that group lands at the top of the element range. So for
   the CORRECT grouping of code-nibbles to scales, essentially every group must
   contain a full-scale code. A wrong grouping mixes elements across scales and
   the rate drops. Measured on a real file: 100.0% for the per-8-byte grouping
   against 88.6% for whole-block split-half, which settles the grouping without
   any reference at all. This check also validates that the scales sit at the
   START of the block: a shifted read cannot produce a clean 1.000.

2. WEIGHT MAGNITUDE. Reports the decoded standard deviation. NOTE, corrected
   2026-08-30: a value around 118 here is NOT a decode error. llama.cpp's own
   dequantization of the same tensor produces byte-identical values, verified
   by re-emitting the model as F16 and diffing (std matched to six decimals,
   same leading elements). The block decode is correct and INCOMPLETE at the
   model level: the per-tensor scale is applied later, in the compute graph.

3. COMPANION SCALES. NVFP4 is TWO-LEVEL: a per-sub-block UE4M3 scale AND a
   per-tensor FP32 scale shipped as a separate `<base>.scale` tensor. Those
   companions SURVIVE requantization to F16, which is what proves they are
   consumed at graph time rather than folded into the weights. An engine that
   decodes the blocks correctly and never applies the companion is off by that
   constant on every NVFP4 tensor, which is runner's bug.

   `input_scale` is the activation-side scale and is NOT part of weight
   reconstruction: `std x .scale` lands on a plausible weight distribution
   (~0.016) while `std x .scale x .input_scale` gives ~0.0003, and a float
   inference path never quantizes activations to FP4 in the first place.
"""

import math
import struct
import sys

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
GGML_TYPE_NVFP4 = 40
QK, SUB = 64, 16
BLOCK_BYTES = 36


def read_header(f):
    magic, ver, ntens, nkv = struct.unpack("<IIQQ", f.read(24))
    if magic != 0x46554747:
        raise SystemExit("not a GGUF file")

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def skipval(t):
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            rstr()
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                skipval(et)
        else:
            f.read(fixed[t])

    for _ in range(nkv):
        rstr()
        skipval(struct.unpack("<I", f.read(4))[0])
    tensors = []
    for _ in range(ntens):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ty = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors.append((name, ne, ty, off))
    pos = f.tell()
    return tensors, (pos + 31) // 32 * 32


def ue4m3(x, bias=7):
    if x == 0 or x == 0x7F:
        return 0.0
    e, m = (x >> 3) & 0xF, x & 0x7
    return math.ldexp(m, -(bias + 2)) if e == 0 else math.ldexp(1.0 + m / 8.0, e - bias)


# nibble slot -> (byte index, which half). The two candidate groupings.
def slots_subblock(i):          # runner's reading: 8 bytes per 16-element group
    s, r = i // SUB, i % SUB
    return s * 8 + (r % 8), r // 8


def slots_wholeblock(i):        # the MXFP4-style alternative, split over all 64
    return (i, 0) if i < 32 else (i - 32, 1)


def saturation(raw, nblk, slots):
    hits = total = 0
    for b in range(nblk):
        blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        if len(blk) < BLOCK_BYTES:
            break
        qs = blk[4:]
        for g in range(QK // SUB):
            top = 0
            for i in range(g * SUB, (g + 1) * SUB):
                byte, half = slots(i)
                c = qs[byte] >> 4 if half else qs[byte] & 0xF
                top = max(top, c & 7)
            total += 1
            hits += (top == 7)
    return hits / total if total else 0.0


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    with open(argv[1], "rb") as f:
        tensors, data_start = read_header(f)
        nv = [t for t in tensors if t[2] == GGML_TYPE_NVFP4]
        print(f"NVFP4 tensors: {len(nv)} of {len(tensors)}")
        if not nv:
            raise SystemExit("no NVFP4 tensors in this file")
        byname = {t[0]: t for t in tensors}

        name, ne, _, off = nv[0]
        print(f"probing: {name} ne={ne}\n")
        f.seek(data_start + off)
        # never read past this tensor: on a small file the bytes after it
        # belong to the next tensor, and their "statistics" mean nothing
        n_elems = 1
        for d in ne:
            n_elems *= d
        nblk = min(3000, n_elems // QK)
        raw = f.read(BLOCK_BYTES * nblk)

        a = saturation(raw, nblk, slots_subblock)
        c = saturation(raw, nblk, slots_wholeblock)
        print("1. scale saturation (higher is the correct grouping)")
        print(f"   per-8-byte sub-block : {a:.3f}")
        print(f"   whole-block split    : {c:.3f}\n")

        vals = []
        for b in range(nblk):
            blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
            if len(blk) < BLOCK_BYTES:
                break
            for s in range(QK // SUB):
                d = ue4m3(blk[s])
                for j in range(8):
                    q = blk[4 + s * 8 + j]
                    for c4 in (q & 0xF, q >> 4):
                        vals.append(E2M1[c4 & 7] * (-1 if c4 & 8 else 1) * d)
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print("2. decoded weight magnitude (block scales only)")
        print(f"   std = {std:.5f}   max|w| = {max(abs(v) for v in vals):.5f}")
        print("   this is EXPECTED to be large: the per-tensor scale below is")
        print("   applied in the compute graph, not by the block decode\n")

        print("3. companion per-tensor scales")
        # the companion REPLACES the .weight suffix: blk.0.attn_gate.weight
        # pairs with blk.0.attn_gate.scale, not ....weight.scale
        base = name[:-len(".weight")] if name.endswith(".weight") else name
        found = False
        for suffix in (".scale", ".input_scale"):
            r = byname.get(base + suffix) or byname.get(name + suffix)
            if not r:
                continue
            found = True
            f.seek(data_start + r[3])
            v = struct.unpack("<f", f.read(4))[0]
            print(f"   {base + suffix} = {v:.10g}")
            if suffix == ".scale":
                print(f"   -> std * scale = {std * v:.5f}  "
                      f"({'plausible' if 0.001 < std * v < 0.2 else 'NOT plausible'})")
        if not found:
            print("   none: this file is single-level scaled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
