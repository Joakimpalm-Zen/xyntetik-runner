"""scripts/gguf-blockorder.py: block order with token_embd first and the
output tensors last, nothing else changed.

Runner v0.5.6's partial split uploads one file prefix from byte 0 through
the farthest offloaded block AND through token_embd, so a file with the
output tensor at its head, an interleaved block order, or token_embd stored
last uploads far more than the split needs. The normaliser rewrites the
data order only: per-tensor bytes, the metadata region and the data offset
stay identical, the runner produces the same greedy text from both files,
and the report's prefix applies v0.5.6's rule exactly.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "gguf-blockorder.py"
EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")


def _gen(path, *flags):
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", *flags, str(path)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def _run(inp, out, *extra):
    rep = pathlib.Path(str(out) + ".json") if out else pathlib.Path(str(inp) + ".check.json")
    cmd = [sys.executable, str(TOOL), str(inp)] + ([str(out)] if out else []) + ["--json", str(rep), "--prefix-blocks", "0", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(rep.read_text())


def _tensor_hashes(path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("bo", TOOL)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    kv_end, alignment, tensors, data_start, header = mod.read_header(str(path))
    out = {}
    with open(path, "rb") as f:
        for t in tensors:
            f.seek(data_start + t["offset"])
            out[t["name"]] = hashlib.sha256(f.read(t["nbytes"])).hexdigest()
    return header[:kv_end], data_start, [t["name"] for t in tensors], out


def test_name_order_file_is_normalised_and_nothing_else_changes(tmp_path):
    alpha = tmp_path / "alpha.gguf"
    _gen(alpha, "--alpha-order")            # token_embd stored last
    out = tmp_path / "block.gguf"
    rep = _run(alpha, out)
    assert rep["verified"] is True
    kv_a, ds_a, names_a, h_a = _tensor_hashes(alpha)
    kv_b, ds_b, names_b, h_b = _tensor_hashes(out)
    assert kv_a == kv_b and ds_a == ds_b, "metadata region or data offset changed"
    assert h_a == h_b, "tensor bytes changed"
    assert names_b[0] == "token_embd.weight" and names_b[-1] == "output_norm.weight"
    assert sorted(names_a) == sorted(names_b) and names_a != names_b
    assert rep["prefix_after"] < rep["prefix_before"], rep
    assert out.stat().st_size == alpha.stat().st_size
    # the report's per-tensor hashes are the file's
    assert {t["name"]: t["sha256"] for t in rep["tensors"]} == h_b


def test_check_only_reports_without_writing(tmp_path):
    alpha = tmp_path / "alpha.gguf"
    _gen(alpha, "--alpha-order")
    rep = _run(alpha, None, "--check-only")
    assert rep["already_in_order"] is False
    assert not (tmp_path / "alpha.gguf.tmp").exists()
    block = tmp_path / "block.gguf"
    _run(alpha, block)
    rep2 = _run(block, None, "--check-only")
    assert rep2["already_in_order"] is True


def test_the_runner_reads_both_files_alike(tmp_path):
    if not EXE.exists():
        pytest.skip("runner not built")
    alpha = tmp_path / "alpha.gguf"
    _gen(alpha, "--alpha-order")
    block = tmp_path / "block.gguf"
    _run(alpha, block)
    texts = []
    for m in (alpha, block):
        p = subprocess.run([str(EXE), "-m", str(m), "-p", "The quick brown", "-n", "8",
                            "--temp", "0", "-t", "2", "--gpu", "off", "--no-tray"],
                           capture_output=True, text=True, cwd=ROOT, timeout=120)
        assert p.returncode == 0, p.stderr
        texts.append(p.stdout.strip())
    assert texts[0] == texts[1]


def test_order_from_copies_a_reference_order(tmp_path):
    ref = tmp_path / "ref.gguf"
    _gen(ref)                                # the generator's own order
    alpha = tmp_path / "alpha.gguf"
    _gen(alpha, "--alpha-order")
    out = tmp_path / "copied.gguf"
    rep = _run(alpha, out, "--order-from", str(ref))
    _, _, names_ref, _ = _tensor_hashes(ref)
    _, _, names_out, _ = _tensor_hashes(out)
    assert names_out == names_ref and rep["verified"] is True
