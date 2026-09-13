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
    """iq_signs7(i) = i | (parity(i) << 7) in kernels.cu stands in for the
    ksigns_iq2xs lookup of the host decoders; the table must be that
    function for every 7-bit index."""
    host = (ROOT / "src" / "quants_iq_grids.h").read_text(encoding="utf-8")
    ksigns = _table(host, "ksigns_iq2xs", 128)
    assert ksigns == [i | ((bin(i).count("1") & 1) << 7) for i in range(128)]
    kmask = _table(host, "kmask_iq2xs", 8)
    # the kernels test sign bit j directly; the host masks with kmask[j]
    assert kmask == [1 << j for j in range(8)]


def test_every_codebook_type_has_both_kernels_registered():
    """A kernel name in the PTX table of cuda.c that no kernel defines fails
    at module load on every CUDA box, which CI cannot see."""
    dev = (ROOT / "src" / "kernels.cu").read_text(encoding="utf-8")
    cuda = (ROOT / "src" / "cuda.c").read_text(encoding="utf-8")
    for t in ("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s"):
        for suffix in ("", "_b"):
            name = f"k_mv_{t}{suffix}"
            assert f'__global__ void {name}(MV_PARAMS)' in dev, name
            assert f'"{name}"' in cuda, name
        assert f"case T_{t.upper()}:" in cuda, t
