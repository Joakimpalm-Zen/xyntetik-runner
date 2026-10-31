"""A --score file names the path its forwards took (score_route), and a
comparison of two routes refuses what cannot have measured them.

--score is solo by default: one row a forward, the decode arithmetic. A
prefill route (a tensor-core GEMM, a batched kernel) runs only in a chunked
score (RUNNER_SCORE_CHUNKED=1); RUNNER_SPEC_BATCH alone sets the chunk width
and does nothing without it. On 2026-10-31 two solo score files of an NVFP4
file, written to compare its scalar and tensor-core prefill, came out byte
for byte equal: neither had reached a GEMM, and a comparison of them would
have passed while testing nothing. score_route records the model file and
the path, and gold-logits.py's route_comparison_guard refuses equal scores,
and a solo side when both sides scored one file. Two different files (a
quant against its BF16 parent) scored solo are the documented quantization
measurement and pass.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_record  # noqa: E402

PROMPT = "the quick brown fox jumps over the lazy dog and keeps on running far away"


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = tmp_path_factory.mktemp("sroute") / "dense.gguf"
    subprocess.run([sys.executable, ROOT / "scripts/make-test-model.py", str(m)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return m


def _score(runner_bin, model, **env):
    e = {k: v for k, v in os.environ.items()
         if k not in ("RUNNER_SCORE_CHUNKED", "RUNNER_SPEC_BATCH")}
    e.update(env)
    p = subprocess.run([runner_bin, "-m", str(model), "--score", "-p", PROMPT, "-t", "2",
                        "--gpu", "off"], cwd=ROOT, env=e, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return json.loads(p.stdout)


def test_solo_is_the_default_and_says_so(runner_bin, model):
    rec = _score(runner_bin, model)
    assert score_record.path_of(rec) == "solo", rec["score_route"]
    assert score_record.rows_max(rec) == 1
    assert rec["score_route"]["backend"] == "cpu"
    assert score_record.model_of(rec) == ("dense.gguf", model.stat().st_size)


def test_spec_batch_alone_stays_solo(runner_bin, model):
    rec = _score(runner_bin, model, RUNNER_SPEC_BATCH="8")
    assert score_record.path_of(rec) == "solo", rec["score_route"]


def test_chunked_takes_the_batched_rows(runner_bin, model):
    rec = _score(runner_bin, model, RUNNER_SCORE_CHUNKED="1", RUNNER_SPEC_BATCH="8")
    assert score_record.path_of(rec) == "keep", rec["score_route"]
    assert score_record.rows_max(rec) == 8


def test_route_field_is_last_and_left_out_of_a_byte_comparison(runner_bin, model):
    a = _score(runner_bin, model)
    assert list(a)[-1] == "score_route"
    b = dict(a, score_route={"paths": {"keep": {"forwards": 1, "rows_max": 16}},
                             "backend": "cpu", "backend_detail": "x"})
    assert score_record.same_scores(a, b)


def _guard():
    spec = importlib.util.spec_from_file_location("gold_logits", ROOT / "scripts/gold-logits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.route_comparison_guard


def _args(a, b, solo_ok=False):
    return argparse.Namespace(score_a=str(a), score_b=str(b), solo_ok=solo_ok)


def _write(path, rec):
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def test_guard_refuses_equal_scores(runner_bin, model, tmp_path):
    rec = _score(runner_bin, model, RUNNER_SCORE_CHUNKED="1")
    a, b = _write(tmp_path / "a.json", rec), _write(tmp_path / "b.json", rec)
    with pytest.raises(SystemExit) as e:
        _guard()(_args(a, b))
    assert "same scores" in str(e.value)


def test_guard_refuses_a_solo_side_unless_asked(runner_bin, model, tmp_path):
    solo = _score(runner_bin, model)
    keep = _score(runner_bin, model, RUNNER_SCORE_CHUNKED="1")
    assert not score_record.same_scores(solo, keep)
    a, b = _write(tmp_path / "a.json", solo), _write(tmp_path / "b.json", keep)
    with pytest.raises(SystemExit) as e:
        _guard()(_args(a, b))
    assert "side a was scored solo" in str(e.value)
    assert "RUNNER_SCORE_CHUNKED=1" in str(e.value)
    note = _guard()(_args(a, b, solo_ok=True))
    assert note == {"paths": {"a": "solo", "b": "keep"}, "route_comparison": True,
                    "solo_ok": True}


def test_guard_passes_two_chunked_routes(runner_bin, model, tmp_path):
    keep = _score(runner_bin, model, RUNNER_SCORE_CHUNKED="1")
    other = dict(keep, logprobs=[x + 1e-6 for x in keep["logprobs"]])
    a, b = _write(tmp_path / "a.json", keep), _write(tmp_path / "b.json", other)
    assert _guard()(_args(a, b)) == {"paths": {"a": "keep", "b": "keep"},
                                     "route_comparison": True, "solo_ok": False}


def test_guard_passes_two_files_scored_solo(runner_bin, model, tmp_path):
    # a quant against its parent: two files, both solo, compare the files
    solo = _score(runner_bin, model)
    route = dict(solo["score_route"], model={"file": "dense-q4.gguf", "bytes": 1})
    other = dict(solo, logprobs=[x + 1e-6 for x in solo["logprobs"]], score_route=route)
    a, b = _write(tmp_path / "a.json", solo), _write(tmp_path / "b.json", other)
    assert _guard()(_args(a, b)) == {"paths": {"a": "solo", "b": "solo"},
                                     "route_comparison": False, "solo_ok": False}


def test_guard_names_a_record_without_score_route(runner_bin, model, tmp_path, capsys):
    keep = _score(runner_bin, model, RUNNER_SCORE_CHUNKED="1")
    old = {k: v for k, v in keep.items() if k != "score_route"}
    other = dict(old, logprobs=[x + 1e-6 for x in old["logprobs"]])
    a, b = _write(tmp_path / "a.json", old), _write(tmp_path / "b.json", other)
    assert _guard()(_args(a, b))["paths"] == {"a": None, "b": None}
    assert "predates score_route" in capsys.readouterr().err
