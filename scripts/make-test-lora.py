#!/usr/bin/env python3
"""Generate LoRA-adapter fixtures against a make-test-model.py base (F32).

Writes, next to the given output prefix:
  <out>.adapter.gguf      rank-4 adapters on blk.0 attn_q + ffn_down, small
                          deterministic values (llama.cpp adapter naming:
                          blk.N.<proj>.weight.lora_a/_b, adapter.lora.alpha)
  <out>.zero.gguf         the same shapes with lora_b == 0 (a no-op adapter)
  <out>.merged.gguf       the BASE file with W += (alpha/r) * (B @ A) merged
                          into the targeted F32 tensors — the mathematical
                          reference the adapter run must approximate
  <out>.badshape.gguf     lora_a whose in-dim disagrees with the base (must
                          be refused, naming the tensor)
  <out>.extradim.gguf     lora_a with the expected first two axes plus an
                          extra third axis (must be refused)
  <out>.halfpair.gguf     lora_a without its lora_b (must be refused)
  <out>.dupside.gguf      two tensors naming the same lora_a slot (must be
                          refused: one of them would silently win)
  <out>.f16.gguf          the same adapter with F16 tensors — the format
                          llama.cpp's convert_lora_to_gguf emits, which the
                          loader must accept (converted to f32 at load)

Usage: make-test-lora.py <base.gguf> <out-prefix>

LORA_TARGETS=all puts the adapter on all seven dense projections of block 0;
LORA_TARGETS=qwen35 on a qwen35 (or clef) base puts it on the first Gated
DeltaNet layer's five projections (attn_qkv, attn_gate, ssm_beta, ssm_alpha,
ssm_out) and its FFN, and on the first full-attention layer's q/k/v/output.
LORA_ARCH overrides the architecture the adapter declares (a qwen35 adapter
for a clef base is what llama.cpp's converter writes from the backbone).
"""
import os
import struct
import sys

from gguf_fixture import kf, ks, write_gguf, STR, T_F32, T_F16

BASE, OUT = sys.argv[1], sys.argv[2]
BASE_ARCH = "llama"
# LORA_ALPHA lets a test build two adapters that differ only in alpha
ALPHA = float(os.environ.get("LORA_ALPHA", "8.0"))
RANK = 4


_seed = 0x1234


def rnd():
    global _seed
    _seed = (_seed * 1103515245 + 12345) & 0x7FFFFFFF
    return (_seed / 0x7FFFFFFF - 0.5) * 0.2


# ---- read the base: header walk sufficient for F32 fixture tensors
def read_base(path):
    d = open(path, "rb").read()
    magic, ver, n_t, n_kv = struct.unpack_from("<IIQQ", d, 0)
    assert magic == 0x46554747
    off = 24

    def rstr():
        nonlocal off
        n = struct.unpack_from("<Q", d, off)[0]
        off += 8
        v = d[off:off + n].decode()
        off += n
        return v

    def skip_val(t):
        nonlocal off
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            rstr()
        elif t == 9:
            et, n = struct.unpack_from("<IQ", d, off)
            off += 12
            for _ in range(n):
                skip_val(et)
        else:
            off += sizes[t]

    global BASE_ARCH
    for _ in range(n_kv):
        key = rstr()
        t = struct.unpack_from("<I", d, off)[0]
        off += 4
        if key == "general.architecture" and t == STR:
            # the adapter must name the base's architecture (the loader
            # refuses a mismatch); granite and muse-glimmer fixtures are not llama
            n = struct.unpack_from("<Q", d, off)[0]
            BASE_ARCH = d[off + 8:off + 8 + n].decode("utf-8")
        skip_val(t)
    tens = []
    for _ in range(n_t):
        name = rstr()
        nd = struct.unpack_from("<I", d, off)[0]
        off += 4
        dims = list(struct.unpack_from("<%dQ" % nd, d, off))
        off += 8 * nd
        ttype, toff = struct.unpack_from("<IQ", d, off)
        off += 12
        tens.append([name, dims, ttype, toff])
    data0 = (off + 31) & ~31
    return d, tens, data0


raw, tens, data0 = read_base(BASE)
Q35_SLOTS = ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out")
targets = {"blk.0.attn_q.weight": None, "blk.0.ffn_down.weight": None}
# LORA_TARGETS=all puts the adapter (and so the merged reference) on every
# projection a backend hooks in block 0, so a device path is held to the
# definition at all seven sites and not only at the default two.
if os.environ.get("LORA_TARGETS") == "all":
    targets = {f"blk.0.{n}.weight": None
               for n in ("attn_q", "attn_k", "attn_v", "attn_output",
                         "ffn_gate", "ffn_up", "ffn_down")}
if os.environ.get("LORA_TARGETS") == "qwen35":
    present = {t[0] for t in tens}
    blocks = sorted({int(n.split(".")[1]) for n in present if n.startswith("blk.")})
    dn = next(b for b in blocks if f"blk.{b}.attn_qkv.weight" in present)
    fa = next(b for b in blocks if f"blk.{b}.attn_q.weight" in present)
    targets = {f"blk.{dn}.{n}.weight": None
               for n in Q35_SLOTS + ("ffn_gate", "ffn_up", "ffn_down")}
    targets.update({f"blk.{fa}.{n}.weight": None
                    for n in ("attn_q", "attn_k", "attn_v", "attn_output")})
targets_f32 = True
for name, dims, ttype, toff in tens:
    if name in targets:
        targets[name] = (dims, toff)
        if ttype != T_F32:
            targets_f32 = False   # quantized base: adapters still written,
                                  # the merged F32 reference is skipped
# a sparse-MoE base has no dense FFN tensors: its adapter is attention-only
# (R17.3), so a default target the base lacks is dropped, not refused --
# but attn_q, the one every base carries, must be there
# LORA_ONLY=<proj> targets that one projection of block 0 alone (a test of
# what a loader does with a single name, hooked or not)
if os.environ.get("LORA_ONLY"):
    only = f"blk.0.{os.environ['LORA_ONLY']}.weight"
    targets = {only: next(((t[1], t[3]) for t in tens if t[0] == only), None)}
    assert targets[only], f"base lacks {only}"
    targets_f32 = next(t[2] for t in tens if t[0] == only) == T_F32
if os.environ.get("LORA_TARGETS") == "qwen35" or os.environ.get("LORA_ONLY"):
    missing = [k for k, v in targets.items() if not v]
    assert not missing, f"base lacks {missing}"
else:
    assert targets.get("blk.0.attn_q.weight"), "base lacks blk.0.attn_q.weight"
targets = {k: v for k, v in targets.items() if v}
ANCHOR = next(iter(targets))   # the hostile variants' tensor


def pack_f(vals):
    return struct.pack("<%df" % len(vals), *vals)


def adapter_tensors(zero_b):
    out = []
    for name, (dims, _) in targets.items():
        nin, nout = dims[0], dims[1]
        a = [rnd() for _ in range(RANK * nin)]
        b = [0.0 if zero_b else rnd() for _ in range(nout * RANK)]
        base = name[:-len(".weight")]
        out.append((f"{base}.weight.lora_a", [nin, RANK], pack_f(a)))
        out.append((f"{base}.weight.lora_b", [RANK, nout], pack_f(b)))
    return out


meta = [ks("general.architecture", os.environ.get("LORA_ARCH", BASE_ARCH)),
        ks("general.type", "adapter"),
        ks("adapter.type", "lora"), kf("adapter.lora.alpha", ALPHA)]

_seed = 0x1234
adap = adapter_tensors(zero_b=False)
write_gguf(OUT + ".adapter.gguf", adap, meta)

_seed = 0x1234
write_gguf(OUT + ".zero.gguf", adapter_tensors(zero_b=True), meta)

# the F16 twin: identical values rounded to half precision, T_F16 tensors
f16 = [(n, d, struct.pack("<%de" % (len(p_) // 4),
                          *struct.unpack("<%df" % (len(p_) // 4), p_)), T_F16)
       for n, d, p_ in adap]
write_gguf(OUT + ".f16.gguf", f16, meta)

# merged reference: base bytes with W += (alpha/r) * B @ A on the targets,
# using the SAME adapter values (same seed replay)
merged = bytearray(raw) if targets_f32 else None
_seed = 0x1234
for name, (dims, toff) in (targets.items() if targets_f32 else []):
    nin, nout = dims[0], dims[1]
    a = [rnd() for _ in range(RANK * nin)]     # a[k*nin + i]
    b = [rnd() for _ in range(nout * RANK)]    # b[j*RANK + k]
    scale = ALPHA / RANK
    base_off = data0 + toff
    for j in range(nout):
        row = base_off + j * nin * 4
        w = list(struct.unpack_from("<%df" % nin, merged, row))
        for i in range(nin):
            delta = 0.0
            for k in range(RANK):
                delta += b[j * RANK + k] * a[k * nin + i]
            w[i] += scale * delta
        struct.pack_into("<%df" % nin, merged, row, *w)
if targets_f32:
    open(OUT + ".merged.gguf", "wb").write(merged)
    print(f"wrote {OUT}.merged.gguf")

# full-coverage adapter for the D3 gradient gate: rank-2 pairs on EVERY
# hooked slot of EVERY layer (attention q/k/v/output + ffn gate/up/down)
def all_target_dims():
    names = {}
    for name, dims, ttype, toff in tens:
        names[name] = dims
    n_layers = 1 + max(int(n.split(".")[1]) for n in names if n.startswith("blk."))
    slots = ["attn_q", "attn_k", "attn_v", "attn_output",
             "ffn_gate", "ffn_up", "ffn_down"]
    if BASE_ARCH in ("qwen35", "clef"):
        # hooked only on that backbone; afmoe's own attn_gate is not
        slots += list(Q35_SLOTS)
    out = []
    for l in range(n_layers):
        for sname in slots:
            key = f"blk.{l}.{sname}.weight"
            if key in names:
                out.append((key, names[key]))
    return out


_seed = 0x7777
full = []
FULL_RANK = 2
for name, dims in all_target_dims():
    nin, nout = dims[0], dims[1]
    a = [rnd() for _ in range(FULL_RANK * nin)]
    b = [rnd() for _ in range(nout * FULL_RANK)]
    base = name[:-len(".weight")]
    full.append((f"{base}.weight.lora_a", [nin, FULL_RANK], pack_f(a)))
    full.append((f"{base}.weight.lora_b", [FULL_RANK, nout], pack_f(b)))
write_gguf(OUT + ".full.gguf", full, meta)

# hostile variants
nin, nout = targets[ANCHOR][0][0], targets[ANCHOR][0][1]
AW = ANCHOR   # "blk.N.<proj>.weight"
bad = [(f"{AW}.lora_a", [nin + 2, RANK], pack_f([0.1] * (RANK * (nin + 2)))),
       (f"{AW}.lora_b", [RANK, nout], pack_f([0.1] * (nout * RANK)))]
write_gguf(OUT + ".badshape.gguf", bad, meta)
extra = [(f"{AW}.lora_a", [nin, RANK, 2],
          pack_f([0.1] * (2 * RANK * nin))),
         (f"{AW}.lora_b", [RANK, nout],
          pack_f([0.1] * (nout * RANK)))]
write_gguf(OUT + ".extradim.gguf", extra, meta)
half = [(f"{AW}.lora_a", [nin, RANK], pack_f([0.1] * (RANK * nin)))]
write_gguf(OUT + ".halfpair.gguf", half, meta)
# A second tensor for the SAME (projection, side). The name is distinct, so
# gguf's duplicate-name rule does not see it, and the merge loader's
# name parse read the trailing "2" as nothing at all -- both landed in the one
# slot and the second silently won, emitting a merged model that is not
# base + adapter.
dup = [(f"{AW}.lora_a", [nin, RANK], pack_f([0.1] * (RANK * nin))),
       (f"{AW}.lora_a2", [nin, RANK], pack_f([0.9] * (RANK * nin))),
       (f"{AW}.lora_b", [RANK, nout], pack_f([0.1] * (nout * RANK)))]
write_gguf(OUT + ".dupside.gguf", dup, meta)
