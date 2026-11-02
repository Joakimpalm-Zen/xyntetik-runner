"""A GGUF shorter than its own header says is reported as truncated.

The usual cause is a download that stopped. A cut inside the metadata read
as "bad GGUF metadata value for tokenizer.ggml.merges", which sends the
reader to suspect the model, not the file (the pre-1.12.0 sweep,
2026-11-02). Wherever the cut falls, inside a key, a string value, an
array's elements or count, the tensor table, or the tensor data, the error
now says the file is truncated and gives the bytes it needs against the
bytes it has.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gguf_fixture import read_header  # noqa: E402


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    m = tmp_path_factory.mktemp("trunc") / "test.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    return m


def cuts(toy):
    with open(toy, "rb") as f:
        _kv, _tensors, end = read_header(f)
    # a key, the first value, the vocabulary's strings, the tensor table (its
    # last offset field), and the tensor data
    return [40, 64, 1000, end - 4, end + 4096]


@pytest.mark.parametrize("which", range(5))
def test_a_cut_file_is_named_truncated_with_its_sizes(runner_bin, toy, tmp_path, which):
    cut = cuts(toy)[which]
    t = tmp_path / "cut.gguf"
    t.write_bytes(toy.read_bytes()[:cut])
    p = subprocess.run([str(runner_bin), "-m", str(t), "-p", "hi", "--gpu", "off", "--no-tray"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode == 1, p.stderr
    err = p.stderr
    assert "bad GGUF metadata" not in err, err
    assert "truncated" in err, err
    assert "the file is %d bytes" % cut in err, err
