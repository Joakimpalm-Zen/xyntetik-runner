"""Every copy of weight bytes to the CUDA device is reported while it runs.

src/cuda.c uploads the model's weights straight from the file mapping, so the
copy runs at the speed the storage faults pages in. On 2026-10-28 an 8 GB bf16
file on a throttled disk took longer than anyone waited, with nothing on
stderr after the split lines, and was killed as a hang. The chunked, reported
copy (`htod_reported`, policy in src/upload_report.h) exists so that cannot
recur silently; this holds every weight-bytes copy to it, from the source
alone. Source only: no device, no build.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# a raw driver copy whose source is file-mapped weight bytes: the model's
# mapping, or a tensor's data pointer
WEIGHT_SOURCES = (r"gf\.map", r"->data\b", r"t->data", r"tensor.*->data")


def test_weight_bytes_reach_the_device_through_the_reported_copy():
    src = (ROOT / "src/cuda.c").read_text()
    raw = []
    for m in re.finditer(r"cu\.MemcpyHtoD\(([^;]*)\);", src):
        args = m.group(1)
        if any(re.search(p, args) for p in WEIGHT_SOURCES):
            line = src.count("\n", 0, m.start()) + 1
            raw.append(f"src/cuda.c:{line}: cu.MemcpyHtoD({args.strip()[:60]}...)")
    assert not raw, "weight bytes copied without progress reporting:\n" + "\n".join(raw)
    # the helper's own chunk copy is the one raw driver copy of weight bytes
    helper = src[src.index("static CUresult htod_reported("):]
    helper = helper[:helper.index("\n}\n")]
    assert "cu.MemcpyHtoD(dst + done, (const uint8_t *)src + done, n)" in helper


def test_the_prefix_and_binding_copies_use_it():
    src = (ROOT / "src/cuda.c").read_text()
    assert "htod_reported(w, w->weights, (const uint8_t *)m->gf.map + w->img_off" in src
    assert "htod_reported(w, d, t->data, t->nbytes)" in src
    assert "upload_report_begin(&w->rep" in src
    assert "upload_report_end(&w->rep" in src
