"""A record or a JSON line the CLI writes is JSON whatever the forward computed.

tests/test_nonfinite_responses.py holds the server's responses to that rule
(R6.68.114); this file holds everything else the runner writes as JSON or
JSON lines (R6.68.115). JSON has no NaN and no infinity, and printf writes
them as `nan` and `inf`, which no parser takes. A number that can be
non-finite goes through json.c's writers, which write `null`; where a
non-finite number means the run failed, the command says so and exits 1.

The fixture is the same as there: a small model with a NaN in every row of
its embedding table, so every position's hidden state is NaN from the first
layer on, on the CPU.
"""
import json
import pathlib
import struct
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "conformance"))
from harness import find_runner  # noqa: E402
from test_embeddings_pooling import _embedding_table  # noqa: E402


@pytest.fixture(scope="module")
def exe():
    e = find_runner(ROOT)
    if not pathlib.Path(e).exists():
        pytest.skip("runner not built")
    return e


def _nan_table(src, dst):
    data, at, width, rows = _embedding_table(src)
    data = bytearray(data)
    for r in range(rows):
        struct.pack_into("<f", data, at + r * width * 4, float("nan"))
    dst.write_bytes(data)


@pytest.fixture(scope="module")
def nan_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("nonfinite")
    m = d / "nan.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-model.py"), str(m)],
                   check=True, stdout=subprocess.DEVNULL)
    _nan_table(m, m)
    return m


def _run(exe, *args, cwd=None):
    return subprocess.run([exe, *map(str, args), "--gpu", "off", "-t", "1"], cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)


def _lines(blob):
    # every line parses: that is the rule under test
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


def test_score_writes_null_for_a_non_finite_log_probability_and_fails(exe, nan_model):
    """--score printed each log-probability, the top lists and the totals with
    %.9g: on a non-finite forward the document read `nan` throughout and the
    command exited 0. A non-finite log-probability means the forward failed,
    which --score already reports as exit 1 when a forward returns nothing."""
    p = _run(exe, "-m", nan_model, "--score", "-p", "hello world", "--score-top", "2")
    doc = json.loads(p.stdout)
    assert doc["logprobs"] and all(v is None for v in doc["logprobs"]), doc["logprobs"]
    assert all(e[1] is None for row in doc["top"] for e in row), doc["top"]
    assert doc["nll_total"] is None and doc["nll_mean"] is None and doc["ppl"] is None
    assert p.returncode == 1, p.stderr.decode(errors="replace")
    assert b"not finite" in p.stderr, p.stderr.decode(errors="replace")
