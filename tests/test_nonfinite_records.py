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
import os
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


def _sft_rows(path):
    path.write_text(
        json.dumps({"prompt": "the runner trains", "completion": " the adapter well"}) + "\n" +
        json.dumps({"prompt": "the corpus", "completion": " is small"}) + "\n")
    return path


def test_train_logprobs_writes_null_before_the_step_refuses(exe, nan_model, tmp_path):
    """--train-logprobs writes a step's log-probabilities as the forward makes
    them, before the step's finiteness check stops the run (exit 1, the
    adapter as the last good step left it), so the one line written read
    nan."""
    lp = tmp_path / "lp.jsonl"
    p = _run(exe, "-m", nan_model, "--train", _sft_rows(tmp_path / "sft.jsonl"),
             "--train-steps", "2", "--lr", "3e-3", "--train-out", tmp_path / "a.gguf",
             "--train-logprobs", lp)
    assert p.returncode == 1, p.stderr.decode(errors="replace")
    rows = _lines(lp.read_text())
    assert rows and all(v is None for v in rows[0]["logprobs"]), rows


def test_a_non_finite_eval_loss_stops_the_run_and_every_line_parses(exe, nan_model, tmp_path):
    """--train-eval wrote a held-out set's loss with %.6f: `nan` on a
    non-finite forward, and a NaN step-0 loss also decided the kill rule
    (relative reduction NaN, so the run was killed on a number that is no
    loss). A non-finite eval loss now stops the run as a training step's
    does (exit 1, the error naming the set), before any of that step's eval
    line is written."""
    rows = _sft_rows(tmp_path / "sft.jsonl")
    p = _run(exe, "-m", nan_model, "--train", rows, "--train-steps", "2", "--lr", "3e-3",
             "--train-out", tmp_path / "a.gguf", "--train-eval", f"held={rows}",
             "--kill-at", "1", "--kill-metric", "held")
    assert p.returncode == 1, p.stderr.decode(errors="replace")
    assert b"--train-eval held" in p.stderr and b"not finite" in p.stderr, \
        p.stderr.decode(errors="replace")
    assert not [l for l in _lines(p.stdout) if "eval" in l], p.stdout


def test_a_non_finite_dpo_step_stops_before_it_reaches_the_adapter(exe, nan_model, tmp_path):
    """--train-dpo checked nothing: on a non-finite forward every step's line
    read nan, the Adam step applied the NaN gradient, and the run exited 0
    with an adapter and a record. A DPO step now takes the --train step's
    rule: a step whose numbers or gradient are not finite stops the run
    before the optimizer touches the adapter (exit 1)."""
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"prompt": "the runner trains", "chosen": " the adapter",
                                 "rejected": " the corpus"}) + "\n")
    out = tmp_path / "dpo.gguf"
    p = _run(exe, "-m", nan_model, "--train-dpo", pairs, "--train-steps", "2",
             "--lr", "3e-3", "--train-out", out)
    assert p.returncode == 1, p.stderr.decode(errors="replace")
    assert b"not finite" in p.stderr, p.stderr.decode(errors="replace")
    assert not _lines(p.stdout), p.stdout
    assert not out.exists()


def test_quant_error_writes_null_for_a_tensor_whose_error_is_not_finite(exe, nan_model, tmp_path):
    """--quant-error costs each tensor at each asked type; a tensor holding a
    NaN has a NaN error, written with %.9g as `nan`, so the planner could not
    read the report at all. It is null now, beside the tensors that cost
    finitely. The exit is unchanged (0): the report measures the file as it
    is."""
    report = tmp_path / "qe.jsonl"
    p = _run(exe, "-m", nan_model, "--quant-error", report, "--quant", "q8_0,q4_0")
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    rows = {r["tensor"]: r for r in _lines(report.read_text())}
    assert all(v["error"] is None for v in rows["token_embd.weight"]["types"].values())
    finite = [v["error"] for t, r in rows.items() if t != "token_embd.weight"
              for v in r["types"].values() if v]
    assert finite and all(isinstance(e, float) for e in finite)


def _generate(exe, model, env):
    return subprocess.run([exe, "-m", str(model), "-p", "hello world", "-n", "2", "--temp", "0",
                           "--gpu", "off", "-t", "1"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=300,
                          env={**{k: v for k, v in os.environ.items()
                                  if not k.startswith(("RUNNER_ACT_TRACE", "RUNNER_MOE_"))},
                               **env})


def test_the_activation_trace_writes_null_where_a_row_is_not_finite(exe, nan_model, tmp_path):
    """RUNNER_ACT_TRACE (R4.11) is how a non-finite forward is localized, and
    it wrote each row's l2 and values with %.9g: the trace of a NaN row read
    nan, so the one file that should show where the NaN starts did not
    parse. The row's hash is still its bytes, NaN or not."""
    path = tmp_path / "act.jsonl"
    _generate(exe, nan_model, {"RUNNER_ACT_TRACE": str(path), "RUNNER_ACT_TRACE_POS": "0,1",
                               "RUNNER_ACT_TRACE_VALUES": "1"})
    rows = _lines(path.read_text())
    assert rows
    # the fixture's NaN is one element of each embedding row; from the first
    # layer on it has spread through the row
    nan_rows = [r for r in rows if r.get("l2", 0) is None]
    assert nan_rows and all(None in r["values"] for r in nan_rows), rows[:2]
    assert all(v is None or isinstance(v, float) for r in rows for v in r.get("values", []))


def test_the_moe_route_trace_writes_null_for_a_gate_that_is_not_finite(exe, tmp_path):
    """RUNNER_MOE_TRACE wrote each pick's gate and norm with %.6g; a NaN
    router read nan. The experts stay as picked: RUNNER_MOE_REPLAY reads
    those."""
    prefix = tmp_path / "f"
    subprocess.run([sys.executable, str(ROOT / "scripts/make-test-moe.py"), str(prefix)],
                   check=True, stdout=subprocess.DEVNULL, cwd=tmp_path)
    model = tmp_path / "moe4-nan.gguf"
    _nan_table(pathlib.Path(str(prefix) + ".moe4.gguf"), model)
    path = tmp_path / "moe.jsonl"
    _generate(exe, model, {"RUNNER_MOE_TRACE": str(path)})
    rows = _lines(path.read_text())
    assert rows and all(len(r["experts"]) == len(r["gates"]) for r in rows)
    assert all(g is None for r in rows for g in r["gates"]), rows[:2]
