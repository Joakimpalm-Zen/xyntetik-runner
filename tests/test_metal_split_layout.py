"""R4.31: a Metal layer split wraps a prefix of the file, and refused any file
with a non-layer tensor stored after the layers. Qwen's
Qwen2.5-1.5B-Instruct Q8_0 stores output_norm.weight (1 kB) last, so --gpu N
on it ran wholly on the CPU. The head and the embedding (output_norm, output,
token_embd), which a split never reads from the device, may now follow the
layers; any other tensor there, or layers out of order, still refuses, and
the message names it.

The anchor is the file's layout not mattering: the same model stored with its
norm last scores byte-identically to the block-order file at the same split.
Needs a Metal device; skips otherwise.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
PROMPT = "the quick brown fox jumps over the lazy dog " * 4


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    if not EXE.exists():
        pytest.skip("runner not built")
    caps = json.loads(subprocess.run([str(EXE), "--caps"], capture_output=True,
                                     text=True, timeout=60).stdout or "{}")
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("no Metal device")
    d = tmp_path_factory.mktemp("mlayout")
    out = {}
    for tag, extra in (("block", []), ("normlast", ["--norm-last"]),
                       ("alpha", ["--alpha-order"])):
        p = d / f"{tag}.gguf"
        subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(p),
                        "--layers", "12", *extra], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        out[tag] = p
    return out


def _score(model, gpu):
    p = subprocess.run([str(EXE), "-m", str(model), "--score", "-p", PROMPT,
                        "--gpu", gpu, "--no-tray"], cwd=ROOT, capture_output=True, timeout=120)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    return json.loads(p.stdout)["logprobs"], err


def test_a_trailing_output_norm_splits_and_scores_as_the_block_order_file(files):
    block, berr = _score(files["block"], "6")
    last, lerr = _score(files["normlast"], "6")
    for err in (berr, lerr):
        assert "partial offload" in err and "6 of 12 layers" in err, err
    assert last == block


def test_layers_out_of_order_still_refuse_and_name_the_tensor(files):
    _, err = _score(files["alpha"], "6")
    assert "partial offload" not in err, err
    assert "--gpu-layers ignored" in err and "blk." in err and "stored after the layers or out of order" in err, err
