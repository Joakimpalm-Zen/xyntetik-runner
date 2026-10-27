"""GGUF byte encoding shared by the fixture generators (scripts/make-test-*.py,
scripts/make-vocab-fixture.py) and by the scripts that read a GGUF header back.

The generators are deliberately dependency-free (struct only) and
deterministic: a fixture's bytes are part of what the tests pin, so every
helper here must keep producing exactly the bytes the generators used to
produce on their own. The Makefile lists this module beside each generator
as a prerequisite, so a change here rebuilds every fixture.

Metadata encoders return the bytes of one key-value entry:

    ku(k, v)   uint32        kau(k, xs)    array of uint32
    ki(k, v)   int32         kai(k, xs)    array of int32
    kf(k, v)   float32       kaf(k, xs)    array of float32
    ks(k, v)   string        kas(k, xs)    array of string
    kb(k, v)   bool          kab(k, xs)    array of bool
                             kau64(k, xs)  array of uint64

write_gguf(path, tensors, kvs) lays out a version-3 file with 32-byte
alignment; a tensor is (name, dims, payload) for F32 or (name, dims,
payload, ggml_type). read_header(f) is the inverse for the header part.
"""
import struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32

# gguf metadata value types (gguf.h: gguf_type)
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)

# ggml tensor types the fixtures write (quants.h)
T_F32, T_F16, T_Q8_0, T_BF16, T_MXFP4, T_NVFP4 = 0, 1, 8, 30, 39, 40


def s(x):
    """A GGUF string: uint64 byte length, then the UTF-8 bytes (no NUL)."""
    b = x if isinstance(x, bytes) else x.encode()
    return struct.pack("<Q", len(b)) + b


def ku(k, v): return s(k) + struct.pack("<II", U32, v)
def ki(k, v): return s(k) + struct.pack("<Ii", I32, v)
def kf(k, v): return s(k) + struct.pack("<If", F32, v)
def ks(k, v): return s(k) + struct.pack("<I", STR) + s(v)
def kb(k, v): return s(k) + struct.pack("<IB", BOOL, 1 if v else 0)


def kas(k, xs):
    # one join, not repeated bytes concatenation: a 250k-entry special list
    # (the quadratic-sort regression fixture) took 12 s to write that way
    return s(k) + struct.pack("<IIQ", ARR, STR, len(xs)) + b"".join(s(x) for x in xs)


def kaf(k, xs): return s(k) + struct.pack("<IIQ", ARR, F32, len(xs)) + struct.pack(f"<{len(xs)}f", *xs)
def kai(k, xs): return s(k) + struct.pack("<IIQ", ARR, I32, len(xs)) + struct.pack(f"<{len(xs)}i", *xs)
def kau(k, xs): return s(k) + struct.pack("<IIQ", ARR, U32, len(xs)) + struct.pack(f"<{len(xs)}I", *xs)
def kau64(k, xs): return s(k) + struct.pack("<IIQ", ARR, U64, len(xs)) + struct.pack(f"<{len(xs)}Q", *xs)
def kab(k, xs): return s(k) + struct.pack("<IIQ", ARR, BOOL, len(xs)) + bytes(1 if x else 0 for x in xs)


def pack(xs):
    """float32 little-endian payload"""
    return struct.pack(f"<{len(xs)}f", *xs)


def ones(n):
    return pack([1.0] * n)


def write_gguf(path, tensors, kvs, *, announce=True):
    """Write a GGUF v3 file: `kvs` are encoded metadata entries, `tensors`
    are (name, dims, payload) or (name, dims, payload, ggml_type) with the
    payload already in that type's byte layout. Prints "wrote <path>" unless
    announce is False."""
    meta = b"".join(kvs)
    info, off = b"", 0
    for name, dims, payload, *rest in tensors:
        info += s(name) + struct.pack("<I", len(dims))
        info += b"".join(struct.pack("<Q", d) for d in dims)
        info += struct.pack("<IQ", rest[0] if rest else T_F32, off)
        off = (off + len(payload) + ALIGN - 1) & ~(ALIGN - 1)
    head = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), len(kvs)) + meta + info
    with open(path, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % ALIGN))
        for _, _, payload, *_ in tensors:
            f.write(payload)
            f.write(b"\0" * ((-len(payload)) % ALIGN))
    if announce:
        print(f"wrote {path}")


_SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
           F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


def read_header(f):
    """Parse the header of an open GGUF file: (kv, tensors, end).

    `kv` maps every metadata key to its decoded Python value (arrays as
    lists, strings decoded with replacement), `tensors` is a list of
    (name, dims, ggml_type, offset) in file order, and `end` is the byte
    offset just past the tensor directory; the data section begins at the
    next multiple of the file's alignment (`kv.get("general.alignment", 32)`).
    Raises ValueError when the magic is not GGUF."""
    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def rstr():
        return f.read(u64()).decode("utf-8", "replace")

    def val(t):
        if t == STR:
            return rstr()
        if t == ARR:
            at, n = u32(), u64()
            return [val(at) for _ in range(n)]
        fmt = _SCALAR[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    magic, _version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    if magic != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    kv = {}
    for _ in range(n_kv):
        key = rstr()
        kv[key] = val(u32())
    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        dims = [u64() for _ in range(u32())]
        ttype, offset = u32(), u64()
        tensors.append((name, dims, ttype, offset))
    return kv, tensors, f.tell()
