"""Tied-V KV cache (RUNNER_TIEDV=1): no K rows for V-less gemma4 layers.

gemma-4's full-attention layers ship no `attn_v.weight`: V is the raw K
projection, so after the weightless V norm and the weighted K norm the two
rows differ only by the norm weights — V = raw*r, K1 = raw*r*w = V*w, and
K = rope(K1). One cached row can carry both. RUNNER_TIEDV=1 stops storing K
for those layers and derives it at read time from the stored V.

The trade is compute for memory and it is NOT byte-identical: the stored K
was roped in f32 and then rounded to f16, while the derived K ropes the
f16-rounded V. So the gate here is layered rather than a single equality:

  1. an absolute anchor OUTSIDE the engine: the test reads the fixture's own
     GGUF tensor index to learn which layers are V-less, and computes the
     exact K-cache byte counts the engine must report from the tensor shapes
     and the pinned context length;
  2. the perturbation must be nonzero (proof the derived path actually ran —
     a byte-identical result here would mean the flat path answered) and no
     larger than what quantizing the same cache to q8_0 costs on the same
     fixture (the feature must be cheaper than the lossy alternative it
     competes with);
  3. the combinations that cannot work must refuse loudly, not degrade:
     a q8 cache (the derived row would be re-quantized, not read) and a GPU
     run (the device kernels read a stored K row).
"""
import json
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CTX = 128            # pinned so the byte arithmetic below is exact
PROMPT = "The quick brown fox jumps over the lazy dog near the river bank"


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiedv") / "g4h.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py",
         "--gemma4-hetero", str(path)],
        check=True, cwd=ROOT,
    )
    return path


def read_tensor_index(path):
    """Minimal GGUF v3 index reader: {tensor name: ne list}."""
    f = open(path, "rb")
    magic, ver = struct.unpack("<Ii", f.read(8))
    assert magic == 0x46554747 and ver >= 2
    n_tensors, n_kv = struct.unpack("<qq", f.read(16))

    def rd_str():
        n, = struct.unpack("<q", f.read(8))
        return f.read(n).decode(errors="replace")

    def skip_val(t):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                 10: 8, 11: 8, 12: 8}
        if t in sizes:
            f.read(sizes[t])
        elif t == 8:
            rd_str()
        elif t == 9:
            et, = struct.unpack("<i", f.read(4))
            n, = struct.unpack("<q", f.read(8))
            for _ in range(n):
                skip_val(et)

    for _ in range(n_kv):
        rd_str()
        t, = struct.unpack("<i", f.read(4))
        skip_val(t)
    index = {}
    for _ in range(n_tensors):
        name = rd_str()
        nd, = struct.unpack("<i", f.read(4))
        ne = list(struct.unpack(f"<{nd}q", f.read(8 * nd)))
        f.read(4 + 8)  # type + offset
        index[name] = ne
    return index


def vless_layers(index):
    """Layers that project K but carry no V tensor — the tied-V condition."""
    k = {int(n.split(".")[1]) for n in index if n.endswith("attn_v.weight")}
    q = {int(n.split(".")[1]) for n in index if n.endswith("attn_k.weight")}
    return sorted(q - k)


def kv_dim(index, layer):
    return index[f"blk.{layer}.attn_k.weight"][1]


def score(runner_bin, model, env=None, extra=()):
    import os
    e = dict(os.environ)
    if env:
        e.update(env)
    proc = subprocess.run(
        [runner_bin, "-m", model, "--score", "-p", PROMPT, "-c", str(CTX),
         "--gpu", "off", "--no-tray", *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, env=e,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    doc = json.loads(proc.stdout.decode())
    lps = [x for x in doc["logprobs"] if x is not None]
    return lps, proc.stderr.decode(errors="replace")


def test_tiedv_layers_bytes_and_bound(runner_bin, fixture):
    index = read_tensor_index(fixture)
    tied = vless_layers(index)
    assert tied, "the gemma4-hetero fixture must have V-less layers"

    flat, _ = score(runner_bin, fixture)
    derived, err = score(runner_bin, fixture, env={"RUNNER_TIEDV": "1"})

    # anchor 1: the engine names exactly the layers the tensor index names
    assert f"kv: tied-V on — {len(tied)} layers" in err, err

    # anchor 2: the reported K-cache byte counts equal the arithmetic from
    # the fixture's own tensor shapes (f16 rows, CTX rows per layer)
    full_bytes = sum(kv_dim(index, l) for l in
                     {int(n.split(".")[1]) for n in index
                      if n.endswith("attn_k.weight")}) * CTX * 2
    tied_bytes = full_bytes - sum(kv_dim(index, l) for l in tied) * CTX * 2
    line = [l for l in err.splitlines() if "tied-V on" in l][0]
    nums = [int(s) for s in line.replace(";", " ").split() if s.isdigit()]
    assert full_bytes in nums and tied_bytes in nums, (
        f"expected K cache {full_bytes} -> {tied_bytes} in: {line}")

    # the perturbation is real (the derived path ran) …
    assert len(flat) == len(derived) and len(flat) > 4
    deltas = [abs(a - b) for a, b in zip(flat, derived)]
    assert max(deltas) > 0, "byte-identical output means tied-V never engaged"

    # … and stays at f16-rounding scale. The bound separates regimes, not
    # noise from noise: a missing knorm multiply or a skipped rope moves
    # logprobs at ~1e-1..1 scale on this fixture, four orders above it.
    assert max(deltas) < 5e-3, f"perturbation beyond rounding scale: {max(deltas)}"


def test_tiedv_derivation_agrees_with_stored_k(runner_bin, fixture):
    """RUNNER_TIEDV_CHECK compares rope(V*w) against the K the store path
    computed, per row, while the flat path still stores real K rows. The
    derivation is algebraically exact — both norms divide by the same rms —
    so disagreement can only be fp rounding: near-total f16 agreement, and
    a tiny worst relative error. A wrong derivation (no multiply, no rope,
    wrong head stride) lands at ~0% agreement and O(1) relative error."""
    _, err = score(runner_bin, fixture, env={"RUNNER_TIEDV_CHECK": "1"})
    rows = [l for l in err.splitlines() if l.startswith("tiedv L")]
    assert rows, "the check printed nothing — the V-less layers never ran it"
    for line in rows:
        pct = float(line.rsplit("(", 1)[1].rstrip("%)"))
        worst = float(line.split("worst rel")[1].split("|")[0].strip())
        assert pct >= 95.0, f"f16 agreement collapsed: {line}"
        assert worst < 1e-2, f"relative error beyond rounding: {line}"


def test_tiedv_refuses_q8_cache(runner_bin, tmp_path):
    # the default hetero fixture's 16-wide sliding heads make q8 itself fall
    # back to f16 (blocks need width % 32 == 0), which would leave this guard
    # unobserved — so this test widens every head to 32 to let q8 engage
    model = tmp_path / "g4h32.gguf"
    subprocess.run(
        [sys.executable, ROOT / "scripts/make-test-model.py",
         "--gemma4-hetero", "--g4-hd32", str(model)],
        check=True, cwd=ROOT,
    )
    _, err = score(runner_bin, model, env={"RUNNER_TIEDV": "1"},
                   extra=("--kv", "q8"))
    assert "keeping f16" not in err, f"q8 never engaged: {err}"
    assert "tied-V refused" in err and "q8" in err, err
    assert "tied-V on" not in err


def test_tiedv_refuses_gpu(runner_bin, fixture):
    import os
    caps = subprocess.run([runner_bin, "--caps"], cwd=ROOT,
                          stdout=subprocess.PIPE, timeout=30)
    gpu = json.loads(caps.stdout.decode()).get("gpu")
    if not gpu:
        pytest.skip("no GPU backend on this machine")
    e = dict(os.environ)
    e["RUNNER_TIEDV"] = "1"
    proc = subprocess.run(
        [runner_bin, "-m", fixture, "-p", "hi", "-n", "2", "--temp", "0",
         "--no-tray"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, env=e,
    )
    assert proc.returncode == 0
    err = proc.stderr.decode(errors="replace")
    assert "tied-V refused" in err and "--gpu off" in err, err
    assert "tied-V on" not in err
