"""The CUDA codebook i-quant kernels carry their own copies of the grids in
src/quants_iq_grids.h (device code cannot read the host tables, and the
kernels compile to PTX on a machine with nvcc, not in CI). Two copies of
33 KB of format data are a drift hazard: an edited or truncated device table
would decode every IQ tensor wrong on one backend only, and only on a GPU.
This holds the copies element-for-element equal, and pins the arithmetic
sign expansion the kernels use in place of ksigns_iq2xs."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (host name, element count) for each codebook the device kernels read
GRIDS = [
    ("iq2xxs_grid", 256),
    ("iq2xs_grid", 512),
    ("iq2s_grid", 1024),
    ("iq3xxs_grid", 256),
    ("iq3s_grid", 512),
    ("iq1s_grid", 2048),
]


def _table(text, decl, n):
    m = re.search(r"\b%s\[%d\] = \{(.*?)\};" % (re.escape(decl), n), text, re.S)
    assert m, f"{decl}[{n}] not found"
    vals = [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", m.group(1))]
    assert len(vals) == n, f"{decl}: {len(vals)} values, expected {n}"
    return vals


def test_device_grids_match_host_grids():
    host = (ROOT / "src" / "quants_iq_grids.h").read_text(encoding="utf-8")
    dev = (ROOT / "src" / "kernels.cu").read_text(encoding="utf-8")
    for name, n in GRIDS:
        assert _table(host, name, n) == _table(dev, "k" + name, n), name


def test_sign_index_expansion_is_the_ksigns_table():
    """The host TABLE is i | (parity(i) << 7) for every 7-bit index, which is
    the identity the device helper iq_signs7 relies on. This is a statement
    about the table; the helper itself is executed against the table by
    tests/test_iq_decode.c (the C compiler on src/iq_decode.h, the file nvcc
    compiles into the kernels)."""
    host = (ROOT / "src" / "quants_iq_grids.h").read_text(encoding="utf-8")
    ksigns = _table(host, "ksigns_iq2xs", 128)
    assert ksigns == [i | ((bin(i).count("1") & 1) << 7) for i in range(128)]
    kmask = _table(host, "kmask_iq2xs", 8)
    # the kernels test sign bit j directly; the host masks with kmask[j]
    assert kmask == [1 << j for j in range(8)]


IQ_KERNELS = [f"k_mv_{t}{suffix}"
              for t in ("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s")
              for suffix in ("", "_b")]


def test_every_codebook_type_has_both_kernels_registered():
    """A kernel name in the PTX table of cuda.c that no kernel defines fails
    at module load on every CUDA box, which CI cannot see."""
    dev = (ROOT / "src" / "kernels.cu").read_text(encoding="utf-8")
    cuda = (ROOT / "src" / "cuda.c").read_text(encoding="utf-8")
    for name in IQ_KERNELS:
        assert f'__global__ void {name}(MV_PARAMS)' in dev, name
        assert f'"{name}"' in cuda, name
    for t in ("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s"):
        assert f"case T_{t.upper()}:" in cuda, t


def _ptx_text():
    """The PTX inside the committed header, unescaped: what the driver JITs.
    A kernel present in kernels.cu but absent here is a stale header, which
    the source checks above cannot see and which fails only at module load
    on a CUDA box."""
    lines = []
    for ln in (ROOT / "src" / "kernels_ptx.h").read_text(encoding="utf-8").splitlines():
        m = re.match(r'^    "(.*)\\n"$', ln)
        if m:
            lines.append(m.group(1).replace('\\"', '"').replace("\\\\", "\\"))
    return "\n".join(lines)


def test_embedded_ptx_carries_the_kernels():
    ptx = _ptx_text()
    entries = set(re.findall(r"\.visible \.entry (k_\w+)\(", ptx))
    missing = [k for k in IQ_KERNELS if k not in entries]
    assert not missing, f"kernels_ptx.h lacks {missing}: regenerate with make ptx on the CUDA 13.3 box"


def test_embedded_ptx_carries_the_grids():
    """The device tables as the PTX stores them (.global .b8 byte lists),
    reassembled little-endian and held to the host tables: an edited or
    truncated table in the header decodes every IQ tensor wrong on the
    device only."""
    ptx = _ptx_text()
    host = (ROOT / "src" / "quants_iq_grids.h").read_text(encoding="utf-8")
    widths = {"iq2xxs_grid": 8, "iq2xs_grid": 8, "iq2s_grid": 8,
              "iq3xxs_grid": 4, "iq3s_grid": 4, "iq1s_grid": 8}
    for name, n in GRIDS:
        w = widths[name]
        m = re.search(r"\.global \.align \d+ \.b8 k%s\[(\d+)\] = \{([^}]*)\};" % name, ptx)
        assert m, f"kernels_ptx.h has no k{name} table"
        assert int(m.group(1)) == n * w, (name, m.group(1))
        raw = [int(v) for v in m.group(2).split(",") if v.strip()]
        assert len(raw) == n * w, (name, len(raw))
        vals = [int.from_bytes(bytes(raw[i * w:(i + 1) * w]), "little") for i in range(n)]
        assert vals == _table(host, name, n), name
