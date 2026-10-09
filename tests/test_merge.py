"""--merge-lora at the CLI: the merged file serves what base + --lora serves.

The byte-level gates (exact fmaf chain, verbatim copies, hostile-adapter
refusals, requant grid bounds) live in test_quantize.c. What only an
end-to-end run can show is that the merged GGUF *behaves*: on the F32
fixture, scoring through the merged file matches scoring base+--lora to
float noise (same math, different summation order), and the provenance
record beside the artifact carries shas that match the actual files.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SENT = "the merged model and the adapted model say the same thing."


@pytest.fixture(scope="module")
def runner_bin():
    exe = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
    if not exe.exists():
        pytest.skip("runner binary not built")
    return exe


@pytest.fixture(scope="module")
def fixtures():
    base, adapter = ROOT / "test.gguf", ROOT / "test-lora.adapter.gguf"
    if not base.exists() or not adapter.exists():
        pytest.skip("run `make test-lora.full.gguf` first")
    return base, adapter


def _merge(runner_bin, base, adapter, out, extra=()):
    return subprocess.run(
        [runner_bin, "-m", str(base), "--lora", str(adapter),
         "--merge-lora", str(out), *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def _score(runner_bin, model, lora=None):
    cmd = [runner_bin, "-m", str(model), "--score", "-p", SENT, "-t", "2",
           "--gpu", "off"]
    if lora:
        cmd += ["--lora", str(lora)]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)["nll_mean"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_merged_scores_like_base_plus_lora(runner_bin, fixtures, tmp_path):
    base, adapter = fixtures
    merged = tmp_path / "merged.gguf"
    p = _merge(runner_bin, base, adapter, merged)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    served = _score(runner_bin, base, lora=adapter)
    folded = _score(runner_bin, merged)
    plain = _score(runner_bin, base)
    # same math up to summation order on the F32 fixture...
    assert abs(folded - served) < 1e-4, (folded, served)
    # ...and it is actually the adapted model, not the base
    assert abs(folded - plain) > 1e-3, (folded, plain)


def test_provenance_record_matches_files(runner_bin, fixtures, tmp_path):
    base, adapter = fixtures
    merged = tmp_path / "m.gguf"
    p = _merge(runner_bin, base, adapter, merged)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    rec = json.loads((tmp_path / "m.gguf.merge.json").read_text())
    assert rec["base"]["sha256"] == _sha(base)
    assert rec["adapter"]["sha256"] == _sha(adapter)
    assert rec["merged"]["sha256"] == _sha(merged)
    assert rec["target"] == "keep"


def test_provenance_record_escapes_output_path(runner_bin, fixtures, tmp_path):
    base, adapter = fixtures
    # Quotes force JSON escaping on POSIX; Windows forbids them in file names,
    # but its path separators exercise the backslash escaping instead.
    merged_name = ('merged-windows.gguf' if sys.platform == "win32"
                   else 'm"erged.gguf')
    merged = tmp_path / merged_name
    p = _merge(runner_bin, base, adapter, merged)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    record = pathlib.Path(f"{merged}.merge.json")
    rec = json.loads(record.read_text())
    assert rec["merged"]["path"] == str(merged)


def test_quant_target_writes_that_type(runner_bin, fixtures, tmp_path):
    base, adapter = fixtures
    merged = tmp_path / "m8.gguf"
    p = _merge(runner_bin, base, adapter, merged, extra=("--quant", "q8_0"))
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    # the merged-quantized file loads and still behaves adapted (q8 grid
    # noise on top of the f32 delta, so the bound is looser)
    served = _score(runner_bin, base, lora=adapter)
    folded = _score(runner_bin, merged)
    assert abs(folded - served) < 5e-2, (folded, served)


def test_merge_lora_requires_lora(runner_bin, fixtures, tmp_path):
    base, _ = fixtures
    p = subprocess.run(
        [runner_bin, "-m", str(base), "--merge-lora",
         str(tmp_path / "x.gguf")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert p.returncode != 0
    assert b"--lora" in p.stderr


def test_two_tensors_for_one_adapter_slot_are_refused(runner_bin, fixtures,
                                                      tmp_path):
    """The name parse stopped at the side letter and never looked further.

    `sscanf(..., "...lora_%c")` consumes one character and does not report
    that the string continued, so `...weight.lora_a2` parsed as side 'a' and
    returned all three fields. It is a distinct tensor NAME, so gguf's
    duplicate check does not fire, and both landed in the same
    (projection, side) slot -- the second freeing and replacing the first.
    The merged file is then not base + adapter, which is the one thing this
    loader exists to guarantee.
    """
    base, _adapter = fixtures
    dup = ROOT / "test-lora.dupside.gguf"
    if not dup.exists():
        pytest.skip("run `make test-lora.full.gguf` first")
    out = tmp_path / "dup.gguf"
    proc = _merge(runner_bin, base, dup, out)
    assert proc.returncode != 0
    assert not out.exists()


def test_record_carries_what_the_merge_kept(runner_bin, fixtures, tmp_path):
    """An exact output type (the F32 fixture) keeps the whole delta, and the
    record says so beside the hashes."""
    base, adapter = fixtures
    merged = tmp_path / "s.gguf"
    p = _merge(runner_bin, base, adapter, merged)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    sv = json.loads((tmp_path / "s.gguf.merge.json").read_text())["survival"]
    assert sv["delta_retained"] == 1.0, sv
    assert sv["min_retained"] == 0.5 and sv["adapted_bytes"] > 0, sv
    assert b"kept 100.0% of the adapter's delta" in p.stderr


def test_a_merge_the_grid_erased_is_refused(runner_bin, fixtures, tmp_path):
    """Merged back onto the 4-bit grid its base already sits on, a delta far
    under half a step leaves every code where it was: the file would be the
    base again. That is refused with the destination untouched, and
    --merge-allow-erased writes it anyway with the numbers on record."""
    base, adapter = fixtures
    q4 = tmp_path / "q4.gguf"
    p = subprocess.run([runner_bin, "-m", str(base), "--quantize", str(q4),
                        "--quant", "q4_0"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    out = tmp_path / "erased.gguf"
    tiny = ("--lora-scale", "0.000001")
    p = subprocess.run([runner_bin, "-m", str(q4), "--lora", str(adapter),
                        "--merge-lora", str(out), *tiny], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=300)
    err = p.stderr.decode(errors="replace")
    assert p.returncode != 0, err
    assert "rounded the fine-tune away" in err and "--merge-allow-erased" in err
    assert not out.exists() and not (tmp_path / "erased.gguf.partial").exists()
    p = subprocess.run([runner_bin, "-m", str(q4), "--lora", str(adapter),
                        "--merge-lora", str(out), *tiny, "--merge-allow-erased"],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    sv = json.loads((tmp_path / "erased.gguf.merge.json").read_text())["survival"]
    assert sv["delta_retained"] == 0.0 and sv["bytes_changed"] == 0, sv
    assert sv["min_retained"] is None, sv


def test_allow_erased_alone_is_refused(runner_bin, fixtures):
    base, _ = fixtures
    p = subprocess.run([runner_bin, "-m", str(base), "--merge-allow-erased",
                        "-p", "hi", "-n", "1"], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode != 0
    assert b"--merge-allow-erased requires --merge-lora" in p.stderr
