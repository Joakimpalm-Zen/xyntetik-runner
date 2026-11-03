"""R4.31: a Metal layer split wraps a prefix of the file, and refused any file
whose layers were stored out of block order or with a non-layer tensor after
them. Qwen's Qwen2.5-1.5B-Instruct Q8_0 is stored in name order (blk.0,
blk.1, blk.10 to blk.19, blk.2, ...) with output_norm.weight last, so --gpu N
on it ran wholly on the CPU. The wrap now reaches the farthest tensor of the
device's layers whatever their order, and the head and the embedding
(output_norm, output, token_embd), which a split never reads from the device,
may follow the layers; any other tensor there still refuses, and the message
names it.

The anchor is the file's layout not mattering: the same model stored in name
order, or with its norm last, scores byte-identically to the block-order file
at the same split. Needs a Metal device; skips otherwise.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / ("runner.exe" if sys.platform == "win32" else "runner")
PROMPT = "the quick brown fox jumps over the lazy dog " * 4


def _make(d, tag, *extra):
    p = d / f"{tag}.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(p), *extra],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return p


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    if not EXE.exists():
        pytest.skip("runner not built")
    caps = json.loads(subprocess.run([str(EXE), "--caps"], capture_output=True,
                                     text=True, timeout=60).stdout or "{}")
    if (caps.get("gpu") or {}).get("backend") != "metal":
        pytest.skip("no Metal device")
    d = tmp_path_factory.mktemp("mlayout")
    return {
        "block": _make(d, "block", "--layers", "12"),
        "normlast": _make(d, "normlast", "--layers", "12", "--move-last", "output_norm.weight"),
        "alpha": _make(d, "alpha", "--layers", "12", "--alpha-order"),
        "ple": _make(d, "ple", "--eseries", "3,16", "--layers", "6"),
        "unknown": _make(d, "unknown", "--layers", "12", "--extra-tail", "x_tail_probe.weight"),
    }


def _score(model, gpu):
    p = subprocess.run([str(EXE), "-m", str(model), "--score", "-p", PROMPT,
                        "--gpu", gpu, "--no-tray"], cwd=ROOT, capture_output=True, timeout=120)
    err = p.stderr.decode(errors="replace")
    assert p.returncode == 0, err
    return json.loads(p.stdout)["logprobs"], err


@pytest.mark.parametrize("layout", ["normlast", "alpha"])
def test_the_layout_does_not_decide_the_split_or_its_numbers(files, layout):
    block, berr = _score(files["block"], "6")
    other, oerr = _score(files[layout], "6")
    for err in (berr, oerr):
        assert "partial offload" in err and "6 of 12 layers" in err, err
    assert other == block


def test_e_series_per_layer_tables_after_the_layers_allow_a_split(files):
    # the fixture stores its per-layer token table, projection and norm after
    # the layers, as the device reads none of them under a split
    _, err = _score(files["ple"], "2")
    assert "partial offload" in err and "2 of 6 layers" in err, err


def test_an_unknown_tensor_after_the_layers_still_refuses_and_is_named(files):
    _, err = _score(files["unknown"], "6")
    assert "partial offload" not in err, err
    assert "--gpu-layers ignored" in err and "x_tail_probe.weight" in err, err
