"""The Metal codebook i-quant kernels (2026-09-17) carry the third copy of
the grids in src/quants_iq_grids.h: the shader is compiled from an embedded
source string at runtime, so device code cannot read the host tables, and
CI has no Metal device. This holds the Metal copy element-for-element equal
to the host, as tests/test_cuda_iq_grids.py does for the CUDA copy, and
checks that every format has both a matvec and a tiled-GEMM kernel defined
in kernels.metal, registered in metal.m, and present in the embedded
header the binary is built from."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

GRIDS = [
    ("iq2xxs_grid", 256),
    ("iq2xs_grid", 512),
    ("iq2s_grid", 1024),
    ("iq3xxs_grid", 256),
    ("iq3s_grid", 512),
    ("iq1s_grid", 2048),
]

TYPES = ("iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s")


def _table(text, decl, n):
    m = re.search(r"\b%s\[%d\] = \{(.*?)\};" % (re.escape(decl), n), text, re.S)
    assert m, f"{decl}[{n}] not found"
    vals = [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", m.group(1))]
    assert len(vals) == n, f"{decl}: {len(vals)} values, expected {n}"
    return vals


def test_metal_grids_match_host_grids():
    host = (ROOT / "src" / "quants_iq_grids.h").read_text(encoding="utf-8")
    metal = (ROOT / "src" / "kernels.metal").read_text(encoding="utf-8")
    for name, n in GRIDS:
        assert _table(host, name, n) == _table(metal, "k" + name, n), name


def test_every_codebook_type_has_both_metal_kernels():
    metal = (ROOT / "src" / "kernels.metal").read_text(encoding="utf-8")
    backend = (ROOT / "src" / "metal.m").read_text(encoding="utf-8")
    for t in TYPES:
        assert re.search(r"^IQ_MV\(k_mv_%s," % t, metal, re.M), t
        assert re.search(r"^IQ_MM\(k_mm_%s," % t, metal, re.M), t
        assert f'@"k_mv_{t}"' in backend, t
        assert f'"k_mm_{t}"' in backend, t
        assert f"case T_{t.upper()}:" in backend, t


def test_embedded_metal_header_carries_the_kernels():
    """src/kernels_metal.h is what the binary compiles at runtime; a kernel
    in kernels.metal that is not in the header fails at pipeline creation
    on every Mac, which CI cannot see (make test-shader-embed catches the
    hash mismatch only on a Mac with the binary built)."""
    hdr = (ROOT / "src" / "kernels_metal.h").read_text(encoding="utf-8")
    for t in TYPES:
        assert f"IQ_MV(k_mv_{t}," in hdr, t
        assert f"IQ_MM(k_mm_{t}," in hdr, t
    for name, n in GRIDS:
        assert f"k{name}[{n}]" in hdr, name
